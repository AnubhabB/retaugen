use core::f32;
use std::{
    cmp::Ordering,
    fs::{self, File, OpenOptions},
    io::{BufReader, Read, Seek, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{anyhow, Result};
use bm25::{Document, Language, SearchEngine, SearchEngineBuilder};
use candle_core::{safetensors, Tensor};
use dashmap::DashMap;
use rayon::{
    iter::{IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use serde::{Deserialize, Serialize};

use crate::{ann::ANNIndex, utils::dedup_text};

const EMBED_FILE: &str = "embed.data";
const TEXT_FILE: &str = "text.data";
const STORE_FILE: &str = ".store";
const EMBED_TENSOR_NAME: &str = "embed";

/// The store would represent data that is indexed.
/// Upon initiation it'd read (or create) a .store, text.data, embed.data file
/// In `text.data` file we'll maintain bytes of text split into embedding chunks. The start index byte, the length of the chunk and some more metadata will be maintained
/// In `embed.data` file we'll maintain byte representations of the tensor, one per each segment.
/// `.store` file will maintain a bincode serialized representation of the `struct Store`
#[derive(Serialize, Deserialize, Default)]
pub struct Store {
    data: Vec<Data>,
    dir: PathBuf,
    text_size: usize,
    #[serde(skip)]
    data_file: Option<Arc<Mutex<BufReader<File>>>>,
    #[serde(skip)]
    index: Option<ANNIndex>,
    #[serde(skip)]
    bm25: Option<SearchEngine<usize>>,
}

/// in `struct Data` we maintain the source file of the data and along with it the chunk of text being indexed
#[derive(Serialize, Deserialize, Debug)]
pub struct Data {
    file: FileKind,
    start: usize,
    length: usize,
    deleted: bool,
    indexed: bool,
}

impl Data {
    pub fn file(&self) -> &FileKind {
        &self.file
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum FileKind {
    Html(PathBuf),
    Pdf((PathBuf, usize)),
    Text(PathBuf),
}

impl Store {
    /// Loads data and initializes indexes if files are present in the given directory or creates them
    pub fn load_from_file(
        dir: &Path,
        num_trees: Option<usize>,
        max_size: Option<usize>,
    ) -> Result<Self> {
        let storefile = dir.join(STORE_FILE);
        let store = if storefile.is_file() {
            let mut store = fs::File::open(storefile)?;
            let mut buf = vec![];
            store.read_to_end(&mut buf)?;

            let mut store = bincode::deserialize::<Store>(&buf)?;

            store.data_file = Some(Arc::new(Mutex::new(BufReader::new(File::open(
                dir.join(TEXT_FILE),
            )?))));

            if let Ok(m) = dir.join(EMBED_FILE).metadata() {
                if m.len() > 0 {
                    store.build_index(num_trees.map_or(16, |n| n), max_size.map_or(16, |sz| sz))?;
                }
            }

            store
        } else {
            fs::File::create_new(storefile)?;
            fs::File::create_new(dir.join(EMBED_FILE))?;
            fs::File::create_new(dir.join(TEXT_FILE))?;

            let mut store = Self {
                dir: dir.to_path_buf(),
                ..Default::default()
            };

            store.data_file = Some(Arc::new(Mutex::new(BufReader::new(File::open(
                dir.join(TEXT_FILE),
            )?))));

            store.save()?;

            store
        };

        Ok(store)
    }

    /// Accepts an iterator of (text, embeddings of text, FileKind) and adds it to the indexed list
    pub fn insert(
        &mut self,
        text_file: &mut File,
        data: &mut impl Iterator<Item = (String, Tensor, FileKind)>,
    ) -> Result<()> {
        // Initialize with a large size to avoid frequent re-allocation
        let mut txt_data = Vec::with_capacity(4096);
        let mut tensors = Vec::with_capacity(64);

        while let Some((txt, tensor, file)) = data.by_ref().next() {
            let txt = txt.as_bytes();
            let data = Data {
                file,
                start: self.text_size,
                length: txt.len(),
                deleted: false,
                indexed: true,
            };
            tensors.push(tensor);
            txt_data = [&txt_data, txt].concat();
            self.data.push(data);
            self.text_size += txt.len();
        }

        Tensor::stack(&tensors, 0)?
            .save_safetensors(EMBED_TENSOR_NAME, self.dir.join(EMBED_FILE))?;
        text_file.write_all(&txt_data)?;
        text_file.sync_all()?;

        self.save()?;

        Ok(())
    }

    /// Serializes and saves the core struct
    pub fn save(&self) -> Result<()> {
        let storebytes = bincode::serialize(&self)?;
        std::fs::write(self.dir.join(STORE_FILE), &storebytes)?;

        Ok(())
    }

    /// API for search into the index
    pub fn search(
        &self,
        qry: &[Tensor],
        qry_str: &[String],
        top_k: usize,
        ann_cutoff: Option<f32>,
        with_bm25: bool,
    ) -> Result<Vec<(usize, &Data, String, f32)>> {
        // Giving 75% weightage to the ANN search and 25% to BM25 search
        const ALPHA: f32 = 0.75;

        // Let's get the ANN scores and BM25 scores in parallel
        let (ann, bm25) = rayon::join(
            || {
                let ann = DashMap::new();
                if let Some(index) = &self.index {
                    qry.par_iter().for_each(|q| {
                        let res = match index.search_approximate(q, top_k * 4, ann_cutoff) {
                            Ok(d) => d,
                            Err(e) => {
                                eprintln!("Error in search_approximate: {e}");
                                return;
                            }
                        };

                        res.iter().for_each(|(idx, score)| {
                            let idx = *idx;
                            if let Some(d) = self.data.get(idx) {
                                let txt = if let Ok(c) = self.chunk(d) {
                                    c
                                } else {
                                    return;
                                };

                                let mut e = ann.entry(idx).or_insert((d, txt, *score));
                                if e.2 < *score {
                                    e.2 = *score;
                                }
                            }
                        });
                    });
                }

                ann
            },
            || {
                if !with_bm25 {
                    return None;
                }
                let bm25 = DashMap::new();
                if let Some(b) = self.bm25.as_ref() {
                    qry_str.par_iter().for_each(|qs| {
                        let res = b.search(qs, top_k * 4);
                        res.par_iter().for_each(|r| {
                            let mut e = bm25.entry(r.document.id).or_insert(r.score);

                            if *e < r.score {
                                *e = r.score;
                            }
                        });
                    });
                };

                Some(bm25)
            },
        );

        // Now, we have the highest ANN and BM25 scores for the set of queries
        // We'll need to create a `combined` score of the two
        // Based on https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques_runnable_scripts/fusion_retrieval.py
        // the steps are:
        // 1. Normalize the vector search score
        // 2. Normalize the bm25 score
        // 3. combined_scores = some alpha * vector_scores + (1 - alpha) * bm25_scores

        // To normalize the ANN Scores, let's go ahead and get the Max/ Min
        let mut ann_max = 0_f32;
        let mut ann_min = f32::MAX;

        ann.iter().for_each(|j| {
            ann_max = j.2.max(ann_max);
            ann_min = j.2.min(ann_min);
        });

        let ann_div = ann_max - ann_min;

        // And same for bm25 scores
        let mut bm25_max = 0_f32;
        let mut bm25_min = f32::MAX;

        let has_bm_25 = bm25.as_ref().map_or(false, |b| b.is_empty());

        let bm25_div = if has_bm_25 {
            if let Some(b) = bm25.as_ref() {
                b.iter().for_each(|j| {
                    bm25_max = j.max(bm25_max);
                    bm25_min = j.min(bm25_min);
                });

                bm25_max - bm25_min
            } else {
                f32::MIN
            }
        } else {
            f32::MIN
        };

        // Ok, to time to normalize our scores and create a combined score for each of them
        let mut combined = ann
            .par_iter()
            .map(|j| {
                let id = *j.key();
                let ann_score = 1. - (j.2 - ann_min) / ann_div;
                let bm25_score = if has_bm_25 {
                    let bm25_score = if let Some(b) = bm25.as_ref().and_then(|b| b.get(&id)) {
                        (*b - bm25_min) / bm25_div
                    } else {
                        // Some very small number if not present
                        0.
                    };

                    bm25_score
                } else {
                    0.
                };

                let combined = ALPHA * ann_score + (1. - ALPHA) * bm25_score;

                (id, j.0, j.1.clone(), combined)
            })
            .collect::<Vec<_>>();

        combined.par_sort_unstable_by(|a, b| b.3.total_cmp(&a.3));

        Ok(combined[0..top_k.min(combined.len())].to_vec())
    }

    /// Given an index `idx` returns `k` adjacent chunks before and after the index
    /// Returns k text blocks before with overlap removed, the current text with overlap removed and k text blocks after, again overlap removed
    pub fn with_k_adjacent(
        &self,
        idx: usize,
        k: usize,
    ) -> Result<(Vec<String>, String, Vec<String>)> {
        // Let's collect all indices that need to be fethed
        // We have to ensure indices that are in the SAME source file
        let start = idx.saturating_sub(k);
        let end = (idx + k + 1).min(self.data.len());

        let trg_data = if let Some(d) = self.data.get(idx) {
            d
        } else {
            eprintln!("Nothing found for index {idx}. Corrupt store!");
            return Err(anyhow!("corrupt store!"));
        };

        let trg_src = match &trg_data.file {
            FileKind::Text(p) => p.as_path(),
            FileKind::Pdf((p, _)) => p.as_path(),
            FileKind::Html(p) => p.as_path(),
        };

        let mut chunks: Vec<(String, usize)> = Vec::with_capacity(end - start);

        (start..end).for_each(|index| {
            let data = if index == idx {
                trg_data
            } else if let Some(d) = self.data.get(index) {
                d
            } else {
                eprintln!("Nothing found for data point {index}");
                return;
            };

            let src = match &data.file {
                FileKind::Text(p) => p.as_path(),
                FileKind::Pdf((p, _)) => p.as_path(),
                FileKind::Html(p) => p.as_path(),
            };

            // Not neighbors if indices are not from the same source file
            if src != trg_src {
                return;
            }

            let txt = if let Ok(txt) = self.chunk(data) {
                txt
            } else {
                return;
            };

            if !chunks.is_empty() {
                let i = chunks.len() - 1;
                chunks[i].0 = if let Ok(t) = dedup_text(&chunks[i].0, &txt) {
                    t
                } else {
                    return;
                }
            }

            chunks.push((txt, index));
        });

        // We have deduplicated text, let's prepare them in the before after kind of structure
        let mut result = (vec![], String::new(), vec![]);

        chunks.into_iter().for_each(|(s, i)| match i.cmp(&idx) {
            Ordering::Less => result.0.push(s),
            Ordering::Equal => result.1 = s,
            Ordering::Greater => result.2.push(s),
        });

        Ok(result)
    }

    /// Given a datapoint, returns the text chunk for that datapoint
    pub fn chunk(&self, data: &Data) -> Result<String> {
        let df = if let Some(df) = self.data_file.as_ref() {
            df
        } else {
            return Err(anyhow!("Store not initialized!"));
        };

        let mut f = df
            .lock()
            .map_err(|e| anyhow!("error acquiring data file lock: {e:?}"))?;
        f.seek(std::io::SeekFrom::Start(data.start as u64))?;

        let mut buf = vec![0; data.length];
        f.read_exact(&mut buf)?;

        String::from_utf8(buf).map_err(|e| anyhow!(e))
    }

    /// Returns the files associated with this store
    pub fn files(&self) -> Result<(File, File)> {
        let text = OpenOptions::new()
            .append(true)
            .open(self.dir.join(TEXT_FILE))?;

        let embed = OpenOptions::new()
            .append(true)
            .open(self.dir.join(EMBED_FILE))?;

        Ok((text, embed))
    }

    // We break apart the index builders to separate functions and build the ANN and BM25 index in parallel
    fn build_index(&mut self, num_trees: usize, max_size: usize) -> Result<()> {
        let (ann, bm25) = rayon::join(
            || {
                Self::build_ann(
                    &self.dir.join(EMBED_FILE),
                    num_trees,
                    max_size,
                    self.data.len(),
                )
            },
            || {
                let docs = self
                    .data
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, d)| {
                        let chunk = match self.chunk(d) {
                            Ok(c) => c,
                            Err(e) => {
                                eprintln!("Error while reading chunk: {e:?}");
                                return None;
                            }
                        };

                        Some(Document {
                            id: idx,
                            contents: chunk,
                        })
                    })
                    .collect::<Vec<_>>();

                Self::build_bm25(docs)
            },
        );

        self.index = Some(ann?);
        self.bm25 = Some(bm25?);

        Ok(())
    }

    // Builds the ANN index
    fn build_ann(
        file: &Path,
        num_trees: usize,
        max_size: usize,
        data_len: usize,
    ) -> Result<ANNIndex> {
        let tensors = unsafe {
            safetensors::MmapedSafetensors::new(file)?
                .load(EMBED_TENSOR_NAME, &candle_core::Device::Cpu)?
        };

        let ann = ANNIndex::build_index(
            num_trees,
            max_size,
            &tensors,
            &(0..data_len).collect::<Vec<_>>(),
        )?;

        Ok(ann)
    }

    // Builds the BM25 index
    fn build_bm25(docs: Vec<Document<usize>>) -> Result<SearchEngine<usize>> {
        let engine = SearchEngineBuilder::<usize>::with_documents(Language::English, docs).build();

        Ok(engine)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Seek},
        path::Path,
    };

    use anyhow::Result;
    use candle_core::IndexOp;
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
    use serde::Deserialize;

    use crate::embed::Embed;

    use super::{FileKind, Store};

    #[derive(Deserialize)]
    struct WikiNews {
        title: String,
        text: String,
    }

    #[test]
    fn storage_init() -> Result<()> {
        let data = {
            let data = std::fs::read_to_string("../test-data/wiki-smoll.json")?;
            serde_json::from_str::<Vec<WikiNews>>(&data)?
        };

        let mut store = Store::load_from_file(Path::new("../test-data"), None, None)?;
        let mut embed = Embed::new(Path::new("../models"))?;

        let mut chunks = data
            .chunks(Embed::STELLA_MAX_BATCH)
            .take(128)
            .filter_map(|c| {
                let batch = c
                    .par_iter()
                    .map(|t| format!("## {}\n{}", t.title, t.text))
                    .collect::<Vec<_>>();

                if let Ok(e) = embed.embeddings(&batch) {
                    let data = batch
                        .par_iter()
                        .enumerate()
                        .map(|(i, t)| {
                            (
                                t.to_string(),
                                e.i(i).unwrap(),
                                FileKind::Text(
                                    Path::new("../test-data/wiki-smoll.json").to_path_buf(),
                                ),
                            )
                        })
                        .collect::<Vec<_>>();

                    Some(data)
                } else {
                    None
                }
            })
            .flatten();

        let (mut text_file, _) = store.files()?;
        store.insert(&mut text_file, &mut chunks)?;

        // Ok, now let's test we have saved it right or not
        for (i, d) in store.data.iter().enumerate() {
            let mut df = store.data_file.as_ref().unwrap().lock().unwrap();
            let mut buf = vec![0_u8; d.length];
            df.seek(std::io::SeekFrom::Start(d.start as u64))?;
            df.read_exact(&mut buf)?;
            let str = std::str::from_utf8(&buf)?;
            assert_eq!(str, format!("## {}\n{}", data[i].title, data[i].text));
        }

        Ok(())
    }

    #[test]
    fn storage_read() -> Result<()> {
        let store = Store::load_from_file(Path::new("../test-data"), Some(16), Some(16))?;

        let mut embed = Embed::new(Path::new("../models"))?;
        let qry = embed
            .query(&[
                "What are the latest news about Iraq?".to_string(),
                "Latest news on ISIS in Iraq?".to_string(),
                "Iraq news and current events".to_string(),
            ])?
            .to_device(&candle_core::Device::Cpu)?;

        let b = qry.dims2()?.0;
        let qry = (0..b)
            .filter_map(|i| qry.get(i).ok().and_then(|t| t.unsqueeze(0).ok()))
            .collect::<Vec<_>>();

        qry.iter().for_each(|q| println!("{:?}", q.shape()));

        let res = store.search(&qry[..], &["Iraq".to_string()], 8, Some(0.36), true)?;

        println!("Response length: {}", res.len());
        res.iter().for_each(|(idx, _, txt, score)| {
            println!("Match[{idx}] score[{score}] ----------\n{txt}\n------------------\n")
        });
        Ok(())
    }
}
