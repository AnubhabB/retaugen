use std::{
    cmp::Ordering, collections::HashMap, fs::{self, File, OpenOptions}, io::{BufReader, Read, Seek, Write}, path::{Path, PathBuf}, sync::{Arc, Mutex}
};

use anyhow::{anyhow, Result};
use bm25::{Document, Language, SearchEngine, SearchEngineBuilder};
use candle_core::{safetensors, Tensor};
use serde::{Deserialize, Serialize};

use crate::{ann::ANNIndex, docs, utils::dedup_text};

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
    bm25: Option<SearchEngine<usize>>
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

            store.build_index(num_trees.map_or(16, |n| n), max_size.map_or(16, |sz| sz))?;

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
        qry: &Tensor,
        qry_str: &str,
        top_k: usize,
        ann_cutoff: Option<f32>,
    ) -> Result<Vec<(usize, &Data, String, f32)>> {
        let ann = if let Some(idx) = &self.index {
            idx
            .search_approximate(qry, top_k, ann_cutoff)?
            .iter()
            .filter_map(|(idx, score)| {
                let idx = *idx;
                if let Some(d) = self.data.get(idx) {
                    let txt = self.chunk(d).ok()?;
                    Some((idx, (d, txt, *score)))
                    // Some((*idx, d, txt, *score))
                } else {
                    None
                }
            })
            .collect::<HashMap<_, _>>()
        } else {
            return Err(anyhow!("ANN Index not ready!"));
        };

        let bm25 = self.bm25.as_ref().map(|bm25| {
            let res = bm25.search(qry_str, top_k * 2);
            res.iter().map(|r| (r.document.id, r.score)).collect::<HashMap<_, _>>()
        });
        
        let result = ann.into_iter().map(|(idx, (data, txt, score))| {

        })
        .collect::<Vec<_>>();
        // println!("{bm25:?}");
        todo!()
        // Ok(ann)
    }

    /// Given an index `idx` returns `k` adjacent chunks before and after the index
    /// Returns k text blocks before with overlap removed, the current text with overlap removed and k text blocks after, again overlap removed
    pub fn with_k_adjacent(&self, idx: usize, k: usize) -> Result<(Vec<String>, String, Vec<String>)> {
        // Let's collect all indices that need to be fethed
        // We have to ensure indices that are in the SAME source file
        let start = if k > idx { 0 } else { idx - k };
        let end = (idx + k).max(self.data.len());

        let trg_data = if let Some(d) = self.data.get(idx) {
            d
        } else {
            eprintln!("Nothing found for index {idx}. Corrupt store!");
            return Err(anyhow!("corrupt store!"))
        };

        let trg_src = match &trg_data.file {
            FileKind::Text(p) => p.as_path(),
            FileKind::Pdf((p, _)) => p.as_path(),
            FileKind::Html(p) => p.as_path(),
        };

        let mut chunks: Vec<(String, usize)> = Vec::with_capacity(end - start);

        (start .. end)
        .for_each(|index| {
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
        let mut result = (
            vec![],
            String::new(),
            vec![]
        );

        chunks.into_iter().for_each(|(s, i)| {
            match i.cmp(&idx) {
                Ordering::Less => result.0.push(s),
                Ordering::Equal => result.1 = s,
                Ordering::Greater => result.2.push(s),
            }
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

        let mut f = df.lock().map_err(|e| anyhow!("error acquiring data file lock: {e:?}"))?;
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

    fn build_index(&mut self, num_trees: usize, max_size: usize) -> Result<()> {
        
        let (ann, bm25) = rayon::join(|| {
            Self::build_ann(&self.dir.join(EMBED_FILE), num_trees, max_size, self.data.len())
        }, || {
            let docs = self.data.iter().enumerate().filter_map(|(idx, d)| {
                let chunk = match self.chunk(d) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Error while reading chunk: {e:?}");
                        return None;
                    }
                };

                Some(Document { id: idx, contents: chunk })
            }).collect::<Vec<_>>();

            Self::build_bm25(docs)
        });

        self.index = Some(ann?);
        self.bm25 = Some(bm25?);

        Ok(())
    }

    fn build_ann(file: &Path, num_trees: usize, max_size: usize, data_len: usize) -> Result<ANNIndex> {
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

    fn build_bm25(docs: Vec<Document<usize>>) -> Result<SearchEngine<usize>> {
        let engine = SearchEngineBuilder::<usize>::with_documents(
            Language::English,
            docs,
        )
        .build();

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

    use crate::{embed::Embed, stella::Config};

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
            .chunks(Config::STELLA_MAX_BATCH)
            .take(128)
            .filter_map(|c| {
                let batch = c
                    .par_iter()
                    .map(|t| format!("## {}\n{}", t.title, t.text))
                    .collect::<Vec<_>>();

                if let Ok(e) = embed.embeddings(crate::embed::ForEmbed::Docs(&batch)) {
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
            .query("What are the latest news about Iraq?")?
            .to_device(&candle_core::Device::Cpu)?;

        let res = store.search(&qry, "What are the latest news about Iraq?", 8, Some(0.36))?;

        println!("Response length: {}", res.len());
        res.iter().for_each(|(idx, _, txt, score)| {
            println!("Match[{idx}] score[{score}] ----------\n{txt}\n------------------\n")
        });
        Ok(())
    }
}
