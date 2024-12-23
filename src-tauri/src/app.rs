use std::{
    collections::{HashMap, HashSet},
    fs::create_dir_all,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use anyhow::{anyhow, Result};
use candle_core::{IndexOp, Tensor};
use rayon::{
    iter::{IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use serde::{Deserialize, Serialize};
use tauri::{
    async_runtime::{channel, Mutex, Receiver, RwLock, Sender},
    Emitter, Window,
};

use crate::{
    docs::{Extractor, ExtractorEvt},
    embed::Embed,
    gen::{Generator, QueryMore},
    store::{FileKind, Store, StoreDataRepr},
    utils::select_device,
};

pub enum Event {
    Search((String, SearchConfig, Window)),
    Index((PathBuf, Window)),
}

pub enum OpResult {
    Status(StatusData),
    Result(SearchResult),
    Error(String),
    Indexing(IndexStatus),
}

#[derive(Clone, Default, Serialize)]
pub struct StatusData {
    head: String,
    hint: Option<String>,
    body: String,
    time_s: Option<f32>,
}

#[derive(Clone, Default, Serialize)]
pub struct IndexStatus {
    msg: &'static str,
    progress: f32,
    pages: usize,
    files: usize,
}

#[derive(Clone)]
pub struct App {
    gen: Arc<Mutex<Option<Generator>>>,
    send: Sender<Event>,
    store: Arc<RwLock<Store>>,
    embed: Arc<Mutex<Embed>>,
    modeldir: PathBuf,
}

impl App {
    // max tokens allowed in context
    const MAX_CTX_TOK: usize = 2800;

    /// Create a new instance of the app
    pub fn new(appdir: &Path, models_dir: &Path) -> Result<Self> {
        let storage_dir = appdir.join("store");
        if !storage_dir.is_dir() {
            create_dir_all(&storage_dir)?;
        }

        let (send, recv) = channel::<Event>(32);
        let store = Arc::new(RwLock::new(Store::load_from_file(
            storage_dir.as_path(),
            None,
            None,
        )?));
        let embed = Arc::new(Mutex::new(Embed::new(Path::new("../models"))?));

        let app = Self {
            gen: Arc::new(Mutex::new(None)),
            send,
            store,
            embed,
            modeldir: models_dir.to_path_buf(),
        };

        let arced = Arc::new(app.clone());
        tauri::async_runtime::spawn(async move {
            Self::listen(arced, recv).await;
        });

        Ok(app)
    }

    /// A method to `send` events/ commands to the app state
    pub async fn send(&self, e: Event) -> Result<()> {
        self.send.send(e).await?;

        Ok(())
    }

    // The internal listner - executes incoming tasks received
    async fn listen(app: Arc<Self>, recv: Receiver<Event>) {
        let mut recv = recv;
        while let Some(evt) = recv.recv().await {
            match evt {
                Event::Search((qry, cfg, w)) => {
                    if let Err(e) = app.search(&qry, &cfg, &w).await {
                        println!("Error while searching: {e:?}");
                    }
                }
                Event::Index((dir, w)) => {
                    if let Err(e) = app.index(dir.as_path(), &w).await {
                        println!("Error while indexing {dir:?}: {e:?}");
                    }
                }
            }
        }
    }

    // Triggers indexing workflow
    async fn index(&self, path: &Path, w: &Window) -> Result<()> {
        {
            // Drop generator module to save some VRAM
            let has_gen = { self.gen.lock().await.is_some() };

            if has_gen {
                let mut l = self.gen.lock().await;
                *l = None;
            }
        }

        println!("Initializing indexing ..");
        let mut to_index = vec![];
        // Create list of files
        path.read_dir()?
            .filter_map(|f| {
                if let Ok(p) = f {
                    if p.metadata().map_or(false, |f| f.is_file()) {
                        Some(p)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .for_each(|f| {
                let path = f.path();
                if let Some(ext) = path.extension() {
                    if ext == "txt" || ext == "pdf" {
                        to_index.push(path);
                    }
                }
            });

        let mut indexing = IndexStatus {
            msg: "Indexing",
            progress: 0.,
            pages: 0,
            files: to_index.len(),
        };

        Self::send_event(w, OpResult::Indexing(indexing.clone())).await?;

        let mut f2t = vec![];
        // For simplicity let's assume that processing each page is 50% of the total processing
        {
            indexing.msg = "Analyzing page layout and extracting pages!";
            let (send, recv) = std::sync::mpsc::channel();
            let model_dir = self.modeldir.clone();
            let to_index = to_index;

            std::thread::spawn(move || {
                let model_dir = model_dir;
                let to_index = to_index;

                let extractor = Extractor::new(&model_dir, &to_index[..]).unwrap();
                extractor.estimate(send.clone());

                if let Err(e) = extractor.extract(send.clone()) {
                    println!("App.index: trying to call extract: {e:?}");
                }
            });

            let mut pagesparsed = 0.;

            while let Ok(d) = recv.recv() {
                match d {
                    ExtractorEvt::Estimate(p) => {
                        indexing.pages = p;
                        Self::send_event(w, OpResult::Indexing(indexing.clone())).await?;
                    }
                    ExtractorEvt::Page => {
                        pagesparsed += 1.;
                        indexing.progress = pagesparsed / indexing.pages as f32 * 50.;
                        Self::send_event(w, OpResult::Indexing(indexing.clone())).await?;
                    }
                    ExtractorEvt::Data(d) => match d {
                        Ok(d) => {
                            f2t = d.concat();
                        }
                        Err(e) => {
                            println!("App.index: error while parsing data: {e:?}");
                        }
                    },
                }
            }
        }

        let chunks = f2t.len() as f32;
        let mut chunks_done = 0.;

        indexing.msg = "Chunking, encoding and embedding extracted data!";

        let mut embed = self.embed.lock().await;

        let mut data = vec![];
        for (txt, f) in f2t.iter() {
            let t = embed
                .split_text_and_encode(txt)
                .iter()
                .map(|(s, t)| (s.to_owned(), t.to_owned(), f.to_owned()))
                .collect::<Vec<_>>();

            data = [data, t].concat();

            chunks_done += 1.;
            indexing.progress = ((chunks_done / chunks) * 50.) + 50.;

            Self::send_event(w, OpResult::Indexing(indexing.clone())).await?;
        }

        let mut data = data.drain(..);

        let mut writer = self.store.write().await;
        let (mut f, _) = writer.files()?;

        writer.insert(&mut f, &mut data)?;

        println!("Indexing done ..");

        Ok(())
    }
}

#[derive(Debug, Deserialize)]
pub struct SearchConfig {
    with_bm25: bool,
    allow_without_evidence: bool,
    max_result: usize,
    ann_cutoff: Option<f32>,
    n_sub_qry: usize,
    k_adjacent: usize,
    relevance_cutoff: f32,
}

#[derive(Debug, Serialize, Default)]
pub struct SearchResult {
    qry: String,
    files: Vec<String>,
    evidence: Vec<Evidence>,
    answer: String,
    elapsed: f32,
}

#[derive(Debug, Serialize, Default)]
pub struct Evidence {
    text: String,
    file: String,
    page: Option<usize>,
}

impl App {
    // A separate method to spawn and send events, the main thread seems to be blocking
    async fn send_event(window: &Window, msg: OpResult) -> Result<()> {
        println!("Sending event to window!");
        match msg {
            OpResult::Status(s) => window.emit("status", &s)?,
            OpResult::Result(s) => window.emit("result", &s)?,
            OpResult::Error(e) => window.emit("error", &e)?,
            OpResult::Indexing(m) => window.emit("indexing", &m)?,
        }

        Ok(())
    }

    // A utility to ensure LLM is loaded
    async fn ensure_generator(&self, window: &Window) -> Result<()> {
        let mut gen = self.gen.lock().await;
        if gen.is_none() {
            Self::send_event(
                window,
                OpResult::Status(StatusData {
                    head: "Loading LLaMA ..".to_string(),
                    body: String::new(),
                    ..Default::default()
                }),
            )
            .await?;

            let start = Instant::now();
            *gen = Some(Generator::new(&self.modeldir, &select_device()?)?);
            Self::send_event(
                window,
                OpResult::Status(StatusData {
                    head: "LLaMA Ready".to_string(),
                    body: String::new(),
                    time_s: Some((Instant::now() - start).as_secs_f32()),
                    ..Default::default()
                }),
            )
            .await?;
        }
        Ok(())
    }

    // Utility to run query preprocessing and sub query decomposition
    async fn query_preproc(
        &self,
        qry: &str,
        n_subqry: usize,
        window: &Window,
    ) -> Result<(QueryMore, Vec<String>, Vec<Tensor>)> {
        Self::send_event(
            window,
            OpResult::Status(StatusData {
                head: "Subquery decomposition".to_string(),
                body: format!("Generating {} subqueries..", n_subqry),
                ..Default::default()
            }),
        )
        .await?;

        let (qry_more, start) = if n_subqry == 0 {
            (QueryMore::new(qry), Instant::now())
        } else {
            let mut gen = self.gen.lock().await;
            let llm = if let Some(gen) = gen.as_mut() {
                gen
            } else {
                println!("App::search: LLaMA not loaded!");
                return Err(anyhow!("Error Loading Generator"));
            };

            (llm.query_preproc(qry, n_subqry)?, Instant::now())
        };

        let (q_txt, q_tensor) = {
            let queries = qry_more.queries();
            let mut emb = self.embed.lock().await;
            let t = emb.query(&queries)?;

            let tensor = (0..queries.len())
                .map(|i| {
                    t.i(i)
                        .unwrap()
                        .to_device(&candle_core::Device::Cpu)
                        .unwrap()
                        .unsqueeze(0)
                        .unwrap()
                })
                .collect::<Vec<_>>();

            (queries, tensor)
        };

        Self::send_event(
            window,
            OpResult::Status(StatusData {
                head: format!("Generated {} subqueries", qry_more.sub_queries().len()),
                body: if n_subqry > 0 {
                    format!(
                        "<b>Topic:</b>\n{}\n\n<b>Subqueries:</b>\n- {}",
                        qry_more.topic(),
                        qry_more.sub_queries().join("\n- ")
                    )
                } else {
                    String::new()
                },
                time_s: Some((Instant::now() - start).as_secs_f32()),
                ..Default::default()
            }),
        )
        .await?;

        Ok((qry_more, q_txt, q_tensor))
    }

    // A function to run the `relevance` pass
    async fn find_relevant(
        &self,
        qry: &[String],
        cutoff: f32,
        res: &[StoreDataRepr<'_>],
        window: &Window,
    ) -> Result<Vec<usize>> {
        // A relevance cutoff greater than `0` means we have activated the flow
        if cutoff == 0. {
            return Ok(Vec::new());
        }

        Self::send_event(
            window,
            OpResult::Status(StatusData {
                head: "Starting `Relevance` and Re-ranking pass".to_string(),
                body: format!("Relevance cutoff: {}", cutoff),
                ..Default::default()
            }),
        )
        .await?;

        let mut gen = self.gen.lock().await;
        let llm = if let Some(gen) = gen.as_mut() {
            gen
        } else {
            return Err(anyhow!("generator not found"));
        };

        let start = Instant::now();
        // Sometimes the LLM ends up returning duplicates, this is to clean them out
        let mut unq = HashSet::new();

        // If we send ALL our response, we'll probably run out of context length
        // So, let's chunk this
        let mut relevant = res
            .chunks(8)
            .filter_map(|c| {
                let batched = c.par_iter().map(|k| (k.0, k.2.clone())).collect::<Vec<_>>();
                llm.find_relevant(qry, &batched).ok()
            })
            .flatten()
            .filter(|(idx, score)| {
                if unq.contains(idx) || *score < cutoff {
                    false
                } else {
                    unq.insert(*idx);
                    true
                }
            })
            .collect::<Vec<_>>();

        relevant.par_sort_by(|a, b| b.1.total_cmp(&a.1));

        println!("Relevant: {relevant:?}");

        Self::send_event(
            window,
            OpResult::Status(StatusData {
                head: format!("Filtered {} relevant results", relevant.len()),
                body: String::new(),
                time_s: Some((Instant::now() - start).as_secs_f32()),
                ..Default::default()
            }),
        )
        .await?;

        Ok(relevant.iter().map(|(idx, _)| *idx).collect::<Vec<_>>())
    }

    // Rethurns k_adjacent text if its > 0
    async fn k_adjacent(
        &self,
        k_adjacent: usize,
        data: &[StoreDataRepr<'_>],
        window: &Window,
    ) -> Result<Vec<(usize, String)>> {
        if k_adjacent == 0 {
            return Ok(data
                .iter()
                .map(|(idx, _, txt, _)| (*idx, txt.to_owned()))
                .collect::<Vec<_>>());
        }

        Self::send_event(
            window,
            OpResult::Status(StatusData {
                head: "Context enhancement: Expanding search context".to_string(),
                body: format!("<i>K</i> Adjacent: {}", k_adjacent),
                ..Default::default()
            }),
        )
        .await?;

        let store = self.store.read().await;
        let start = Instant::now();
        let enhanced = data
            .iter()
            .filter_map(|(idx, _, _, _)| {
                let a = store.with_k_adjacent(*idx, k_adjacent).ok()?;
                let txt = [a.0.join("\n").as_str(), &a.1, a.2.join("\n").as_str()].join("\n\n");

                Some((*idx, txt))
            })
            .collect::<Vec<_>>();

        Self::send_event(
            window,
            OpResult::Status(StatusData {
                head: format!("Context enhanced with {k_adjacent} adjacent"),
                body: String::new(),
                time_s: Some((Instant::now() - start).as_secs_f32()),
                ..Default::default()
            }),
        )
        .await?;

        Ok(enhanced)
    }

    // So we have a total context length of *4096* tokens
    // leave 1/4th of that for the final *answer* which leaves us with *3072* tokens for our *context* (including the system prompt).
    // Now, our default prompt without any aditional data takes *~250* tokens, which means we are left with around *2800* tokens for our context.
    // So, we'll define our `threshold` as follows:
    // ```
    // while total context > 2500:
    //  Summarize largest chunk
    //```
    async fn create_context(
        &self,
        qry: &str,
        data: &mut [(usize, String)],
        window: &Window,
    ) -> Result<String> {
        Self::send_event(
            window,
            OpResult::Status(StatusData {
                head: "Context generation:".to_string(),
                body: "Generating final context".to_string(),
                ..Default::default()
            }),
        )
        .await?;

        let start = Instant::now();

        let (mut total_tokens, _, mut max_token_idx) = self.compute_tokens(data).await?;

        // Tracking the number of summaries generated
        let mut iter = 0;

        while total_tokens > Self::MAX_CTX_TOK {
            // Break if we have visited at-least data.len() of summaries
            // Nothing more can be done with this
            iter += 1;
            println!("Pre loop[{iter}]:  {total_tokens} {max_token_idx}");
            if iter > data.len() {
                break;
            }

            // This scope is required because the `.lock()` will block and next iterations of tokens will not be computed
            {
                // We need to run a summarization pass for max tokens
                let mut g = self.gen.lock().await;
                let gen = if let Some(g) = g.as_mut() {
                    g
                } else {
                    return Err(anyhow!("Generator not ready!"));
                };

                Self::send_event(
                    window,
                    OpResult::Status(StatusData {
                        head: "Context generation: Summarizing a datapoint".to_string(),
                        body: "Generating summary of a text chunk to fit it in context!"
                            .to_string(),
                        ..Default::default()
                    }),
                )
                .await?;

                let summarystart = Instant::now();
                let summary = gen.summarize(qry, &data.get(max_token_idx).unwrap().1)?;

                data[max_token_idx] = (
                    data[max_token_idx].0,
                    format!("## {}\n{}", summary.heading(), summary.summary()),
                );

                Self::send_event(
                    window,
                    OpResult::Status(StatusData {
                        head: "Context generation: Datapoint summarized ".to_string(),
                        time_s: Some((Instant::now() - summarystart).as_secs_f32()),
                        ..Default::default()
                    }),
                )
                .await?;
            }

            (total_tokens, _, max_token_idx) = self.compute_tokens(data).await?;
            println!("In loop[{iter}]: {total_tokens} {max_token_idx}");
        }

        println!("Begining context generation!");

        let ctx = data
            .iter()
            .map(|(idx, txt)| format!("Source: {idx}\n{}\n-------------\n", txt.trim()))
            .collect::<Vec<_>>()
            .join("")
            .trim()
            .to_string();

        Self::send_event(
            window,
            OpResult::Status(StatusData {
                head: "Context generated".to_string(),
                time_s: Some((Instant::now() - start).as_secs_f32()),
                ..Default::default()
            }),
        )
        .await?;

        Ok(ctx)
    }

    // Computes `token` count related information of given text
    // returns: (total_tokens, max_tokens, max_token_idx)
    async fn compute_tokens(&self, data: &[(usize, String)]) -> Result<(usize, usize, usize)> {
        let mut g = self.gen.lock().await;
        let gen = if let Some(g) = g.as_mut() {
            g
        } else {
            return Err(anyhow!("Generator not ready!"));
        };

        // Total size of encoded tokens
        let mut total_tokens = 0;
        // Chunk with max tokens
        let mut max_token_idx = 0;
        let mut max_tokens = 0;

        data.iter().enumerate().for_each(|(i, (_, txt))| {
            let tokenized = gen.tokenize(txt).unwrap().len();
            total_tokens += tokenized;
            if tokenized > max_tokens {
                max_token_idx = i;
                max_tokens = tokenized;
            }
        });

        Ok((total_tokens, max_tokens, max_token_idx))
    }

    // Trigger the search flow - the search pipeline
    async fn search(&self, qry: &str, cfg: &SearchConfig, res_send: &Window) -> Result<()> {
        let mut final_result = SearchResult {
            qry: qry.to_string(),
            ..Default::default()
        };

        if let Err(e) = self.ensure_generator(res_send).await {
            println!("App::search: error while loading LLaMA: {e:?}");
            Self::send_event(
                res_send,
                OpResult::Error("Error Loading Generator".to_string()),
            )
            .await?;

            return Err(anyhow!("Error Loading Generator"));
        }

        let search_start = Instant::now();

        // Step 1: query preprocessing
        let (qry_more, q_txt, q_tensor) =
            match self.query_preproc(qry, cfg.n_sub_qry, res_send).await {
                Ok(r) => r,
                Err(e) => {
                    println!("App::search: error during sub query decomposition: {e:?}");

                    Self::send_event(
                        res_send,
                        OpResult::Error("Error during subquery decomposition".to_string()),
                    )
                    .await?;

                    return Err(anyhow!("Error during subquery decomposition"));
                }
            };

        // Step 2: Approximate nearest neighbor search
        Self::send_event(
            res_send,
            OpResult::Status(StatusData {
                head: "Firing Approx. Nearest Neighbor search".to_string(),
                body: format!(
                    "<b>BM25:</b> {} | <b>ANN Cutoff:</b> {}",
                    cfg.with_bm25,
                    cfg.ann_cutoff.map_or(0., |c| c)
                ),
                ..Default::default()
            }),
        )
        .await?;

        let store = self.store.read().await;
        let (res, elapsed) = {
            let start = Instant::now();

            let res = store.search(
                &q_tensor,
                &[qry_more.topic().to_string()],
                cfg.max_result,
                cfg.ann_cutoff,
                cfg.with_bm25,
            )?;

            (res, (Instant::now() - start).as_secs_f32())
        };

        Self::send_event(
            res_send,
            OpResult::Status(StatusData {
                head: format!("ANN Search yielded {} results", res.len()),
                body: String::new(),
                time_s: Some(elapsed),
                ..Default::default()
            }),
        )
        .await?;

        // Keep initial findings, if the search errors out
        let mut res_map = HashMap::new();
        res.iter().for_each(|r| {
            res_map.insert(r.0, r.to_owned());
        });

        // Step 3: Check for relevance and re-rank
        let relevant = match self
            .find_relevant(&q_txt, cfg.relevance_cutoff, &res[..], res_send)
            .await
        {
            Ok(r) => {
                if r.is_empty() {
                    res
                } else {
                    r.iter()
                        .filter_map(|idx| {
                            let dp = res_map.get(idx)?;
                            Some(dp.to_owned())
                        })
                        .collect::<Vec<_>>()
                }
            }
            Err(e) => {
                println!("App::search: error during relevance filtering: {e:?}");

                Self::send_event(
                    res_send,
                    OpResult::Error("Error during relevance filtering".to_string()),
                )
                .await?;

                return Err(anyhow!("Error during relevance filtering"));
            }
        };

        // Step 4: context augmentation - get adjacent data
        let mut enhanced = match self
            .k_adjacent(cfg.k_adjacent, &relevant[..], res_send)
            .await
        {
            Ok(e) => e,
            Err(e) => {
                println!(
                    "App::search: error during fetching of {} adjacent: {e:?}",
                    cfg.k_adjacent
                );

                Self::send_event(
                    res_send,
                    OpResult::Error("Error during context enhancement".to_string()),
                )
                .await?;

                return Err(anyhow!("Error during context enhancement"));
            }
        };

        // We have enhanced context now, let's summarize the context if needed
        let qry_str = q_txt.join("\n");
        let ctx = match self
            .create_context(&qry_str, &mut enhanced[..], res_send)
            .await
        {
            Ok(c) => c,
            Err(e) => {
                println!("App::search: generating context: {e:?}");
                Self::send_event(
                    res_send,
                    OpResult::Error("Error generating context".to_string()),
                )
                .await?;

                return Err(anyhow!("Error during context generation"));
            }
        };

        if ctx.is_empty() && !cfg.allow_without_evidence {
            return Self::send_event(res_send, OpResult::Error("Nothing found!".to_string())).await;
        }

        // Step 5: Finally the answer
        Self::send_event(
            res_send,
            OpResult::Status(StatusData {
                head: "Generating answer!".to_string(),
                body: String::new(),
                ..Default::default()
            }),
        )
        .await?;

        let (ans, elapsed) = {
            let mut gen = self.gen.lock().await;
            let llm = if let Some(gen) = gen.as_mut() {
                gen
            } else {
                return Err(anyhow!("generator not found"));
            };

            let start = Instant::now();
            let answer = llm.answer(qry_more.topic(), qry_more.source(), &ctx)?;

            (answer, (Instant::now() - start).as_secs_f32())
        };

        Self::send_event(
            res_send,
            OpResult::Status(StatusData {
                head: "Finally, generated answer!".to_string(),
                body: String::new(),
                time_s: Some(elapsed),
                ..Default::default()
            }),
        )
        .await?;

        final_result.answer = ans.answer().to_string();

        if ctx.is_empty() {
            final_result.files = Vec::new();
            final_result.evidence = Vec::new();
        } else {
            let mut file_list = HashSet::new();
            println!("Num Evidence: {}", ans.evidence().len());
            final_result.evidence = ans
                .evidence()
                .iter()
                .filter_map(|e| {
                    let evidence = res_map.get(&e.index())?.1.file();
                    let (file, page) = match evidence {
                        FileKind::Pdf((pth, pg)) => {
                            file_list.insert(pth.to_owned());
                            (pth.to_str()?.to_string(), Some(*pg))
                        }
                        FileKind::Text(pth) => {
                            file_list.insert(pth.to_owned());
                            (pth.to_str()?.to_string(), None)
                        }
                        FileKind::Html(pth) => {
                            file_list.insert(pth.to_owned());
                            (pth.to_str()?.to_string(), None)
                        }
                    };

                    Some(Evidence {
                        text: e.text().to_string(),
                        file,
                        page,
                    })
                })
                .collect::<Vec<_>>();

            final_result.files = file_list
                .iter()
                .filter_map(|f| f.to_str().map(|s| s.to_string()))
                .collect::<Vec<_>>();

            println!("{:?}", final_result.files);
        }

        final_result.elapsed = (Instant::now() - search_start).as_secs_f32();

        Self::send_event(res_send, OpResult::Result(final_result)).await?;

        Ok(())
    }
}
