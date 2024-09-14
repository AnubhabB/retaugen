use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, Read, Seek, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{anyhow, Result};
use candle_core::{safetensors, Tensor};
use serde::{Deserialize, Serialize};

use crate::ann::ANNIndex;

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

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum FileKind {
    Html(PathBuf),
    Pdf(PathBuf),
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

            store.build_index(num_trees.map_or(16, |n| n), max_size.map_or(16, |sz| sz))?;
            store.data_file = Some(Arc::new(Mutex::new(BufReader::new(File::open(
                dir.join(TEXT_FILE),
            )?))));

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
        top_k: usize,
        cutoff: Option<f32>,
    ) -> Result<Vec<(&Data, String, f32)>> {
        let index = if let Some(idx) = &self.index {
            idx
        } else {
            return Err(anyhow!("ANN Index not ready!"));
        };

        let res = index
            .search_approximate(qry, top_k, cutoff)?
            .iter()
            .filter_map(|(idx, score)| {
                if let Some(d) = self.data.get(*idx) {
                    if let Some(f) = &self.data_file {
                        match f.lock() {
                            Ok(mut l) => {
                                l.seek(std::io::SeekFrom::Start(d.start as u64)).unwrap();
                                let mut txt = vec![0_u8; d.length];
                                l.read_exact(&mut txt).unwrap();

                                Some((d, String::from_utf8(txt).unwrap(), *score))
                            }
                            Err(e) => {
                                println!("Error acquiring lock!: {e:?}");
                                None
                            }
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        Ok(res)
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
        let tensors = unsafe {
            safetensors::MmapedSafetensors::new(self.dir.join(EMBED_FILE))?
                .load(EMBED_TENSOR_NAME, &candle_core::Device::Cpu)?
        };

        let ann = ANNIndex::build_index(
            num_trees,
            max_size,
            &tensors,
            &(0..self.data.len()).collect::<Vec<_>>(),
        )?;
        self.index = Some(ann);
        Ok(())
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
        let mut embed = Embed::new()?;

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

        println!("Begin insert!");
        let (mut text_file, _) = store.files()?;
        store.insert(&mut text_file, &mut chunks)?;
        println!("Preparing to test ..{}", store.data.len());

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

        let mut embed = Embed::new()?;
        let qry = embed
            .query("What are the latest news about Iraq?")?
            .to_device(&candle_core::Device::Cpu)?;

        let res = store.search(&qry, 4, Some(0.4))?;

        println!("Response length: {}", res.len());
        res.iter().for_each(|(_, txt, score)| {
            println!("Match score[{score}] ----------\n{txt}\n------------------\n")
        });
        Ok(())
    }
}
