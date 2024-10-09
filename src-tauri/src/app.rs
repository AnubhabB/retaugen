use std::{
    fs::create_dir_all,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{anyhow, Result};
use candle_core::IndexOp;
use rayon::slice::ParallelSliceMut;
use tauri::async_runtime::{self, channel, Mutex, Receiver, RwLock, Sender};

use crate::{
    docs::files_to_text, embed::Embed, gen::Generator, store::Store, utils::select_device,
};

pub enum Event {
    Search(String),
    Index(PathBuf),
}

#[derive(Clone)]
pub struct App {
    gen: Arc<Mutex<Option<Generator>>>,
    send: Sender<Event>,
    store: Arc<RwLock<Store>>,
    embed: Arc<Mutex<Embed>>,
    // appdir: PathBuf,
    modeldir: PathBuf,
}

const MAX_RESULTS: usize = 8;
const K_ADJACENT: usize = 1;

impl App {
    /// Create a new instance of the app
    pub fn new(appdir: &Path, models_dir: &Path) -> Result<Self> {
        let storage_dir = appdir.join("store");
        if !storage_dir.is_dir() {
            create_dir_all(&storage_dir)?;
        }

        let (send, recv) = channel(32);
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
            // appdir: appdir.to_path_buf(),
            modeldir: models_dir.to_path_buf(),
        };

        let arced = Arc::new(app.clone());
        async_runtime::spawn(async move {
            Self::listen(arced, recv).await;
        });

        Ok(app)
    }

    pub async fn send(&self, e: Event) -> Result<()> {
        self.send.send(e).await?;
        Ok(())
    }

    async fn listen(app: Arc<Self>, recv: Receiver<Event>) {
        let mut recv = recv;
        while let Some(evt) = recv.recv().await {
            match evt {
                Event::Search(qry) => {
                    if let Err(e) = app.search(&qry).await {
                        eprintln!("Error while searching: {e:?}");
                    }
                }
                Event::Index(dir) => {
                    if let Err(e) = app.index(dir.as_path()).await {
                        eprintln!("Error while indexing {dir:?}: {e:?}");
                    }
                }
            }
        }
    }

    // Trigger the search flow - the search pipeline
    async fn search(&self, qry: &str) -> Result<()> {
        let mut gen = self.gen.lock().await;
        if gen.is_none() {
            println!("Loading generator ..");
            *gen = Some(Generator::new(&self.modeldir, &select_device()?)?)
        }

        let llm = if let Some(gen) = gen.as_mut() {
            gen
        } else {
            return Err(anyhow!("generator not found"));
        };

        // Step 1: query preprocessing
        println!("Step 1: query preprocessing ..");
        let qry_more = llm.query_preproc(qry, 4)?;
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
        println!("{qry_more:?}");

        // Step 2: Approximate nearest neighbor search
        println!("Step 2: ANN Search ..");
        let store = self.store.read().await;
        let res = store.search(
            &q_tensor,
            &[qry_more.topic().to_string()],
            MAX_RESULTS,
            Some(0.75),
        )?;
        println!("Step 2: ANN search returned {} results", res.len());

        // Step 3: Check for relevance and re-rank
        // If we send ALL our response, we'll probably run out of context length
        // So, let's chunk this
        println!("Step 3: filtering out relevant results and reranking ..");
        let mut relevant = res
            .chunks(6)
            .enumerate()
            .filter_map(|(i, c)| {
                let batched = c.iter().map(|k| (k.0, k.2.clone())).collect::<Vec<_>>();
                println!("Relevance: A batch[{i}] begins!");
                llm.find_relevant(&q_txt, &batched).ok()
            })
            .flatten()
            .collect::<Vec<_>>();
        relevant.par_sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .map_or(std::cmp::Ordering::Equal, |o| o)
        });
        println!("Step 3: Found {} relevant results", relevant.len());

        // Step 4: context augmentation - get adjacent data
        println!("Step 4: getting {K_ADJACENT} adjacent data ..");
        let context = relevant
            .iter()
            .filter_map(|(idx, cfd)| {
                let a = store.with_k_adjacent(*idx, K_ADJACENT).ok()?;
                println!(
                    "{idx}[{cfd}]: Before: {}\nThis: {}\nNext: [{}]",
                    a.0.len(),
                    a.1,
                    a.2.len()
                );
                // println!("Before: {:?}", &a.0.len());
                // println!("This: {:?}", a.1);
                // println!("After: {:?}", &a.2.len());
                // let summary_before = if !a.0.is_empty() {
                //     let s = llm.summarize(&a.0.join("\n\n")).ok()?;
                //     format!("## {}\n\n{}", s.heading(), s.summary())
                // } else {
                //     String::new()
                // };
                // let summary_after = if !a.2.is_empty() {
                //     let s = llm.summarize(&a.2.join("\n\n")).ok()?;
                //     format!("## {}\n\n{}", s.heading(), s.summary())
                // } else {
                //     String::new()
                // };

                Some([a.0.join("\n").as_str(), &a.1, a.2.join("\n").as_str()].join("\n\n"))
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        // println!("Context:\n{context}");
        // Step 5: Finally the answer
        let answer = llm.answer(qry_more.topic(), qry_more.source(), &context)?;
        println!("{answer:?}");

        Ok(())
    }

    // Triggers indexing workflow
    async fn index(&self, path: &Path) -> Result<()> {
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

        println!("Index {} files", to_index.len());

        let f2t = files_to_text(self.modeldir.as_path(), &to_index[..])?.concat();

        println!(
            "Text exteacion done.. {} text blocks. Begin split and encode ..",
            f2t.len()
        );

        let mut embed = self.embed.lock().await;

        let mut data = f2t.iter().flat_map(|(txt, f)| {
            let t = embed
                .split_text_and_encode(txt)
                .iter()
                .map(|(s, t)| (s.to_owned(), t.to_owned(), f.to_owned()))
                .collect::<Vec<_>>();
            t
        });

        let mut writer = self.store.write().await;
        let (mut f, _) = writer.files()?;

        writer.insert(&mut f, &mut data)?;

        println!("Indexing done ..");

        Ok(())
    }
}
