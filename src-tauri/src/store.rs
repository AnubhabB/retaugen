use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, Read, Seek, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{anyhow, Result};
use candle_core::Tensor;
use serde::{Deserialize, Serialize};

use crate::ann::ANNIndex;

/// The store would represent data that is indexed.
/// Upon initiation it'd read (or create) a .store, text.data, embed.data file
/// In `text.data` file we'll maintain bytes of text split into embedding chunks. The start index byte, the length of the chunk and some more metadata will be maintained
/// in `struct Data`
/// In `embedding.data` file we'll maintain byte representations of the tensor, one per each segment.
/// `.store` file will maintain a bincode serialized representation of the `struct Store`
#[derive(Serialize, Deserialize, Default)]
pub struct Store {
    data: Vec<Data>,
    dir: PathBuf,
    text_end: usize,
    embed_end: usize,
    #[serde(skip)]
    data_file: Option<Arc<Mutex<BufReader<File>>>>,
    #[serde(skip)]
    index: Option<ANNIndex>,
}

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

const EMBED_FILE: &str = "embed.data";
const TEXT_FILE: &str = "text.data";
const STORE_FILE: &str = ".store";

impl Store {
    pub fn load_from_file(dir: &Path) -> Result<Self> {
        let storefile = dir.join(STORE_FILE);
        let store = if storefile.is_file() {
            let mut store = fs::File::open(storefile).unwrap();
            let mut buf = vec![];
            store.read_to_end(&mut buf).unwrap();

            let mut store = bincode::deserialize::<Store>(&buf).unwrap();

            store.build_index().unwrap();
            store.data_file = Some(Arc::new(Mutex::new(BufReader::new(
                File::open(dir.join(TEXT_FILE)).unwrap(),
            ))));

            store
        } else {
            fs::File::create_new(storefile).unwrap();
            fs::File::create_new(dir.join(EMBED_FILE)).unwrap();
            fs::File::create_new(dir.join(TEXT_FILE)).unwrap();

            let store = Self {
                dir: dir.to_path_buf(),
                ..Default::default()
            };

            store.save().unwrap();

            store
        };

        Ok(store)
    }

    pub fn insert(
        &mut self,
        text_file: &mut File,
        embed_file: &mut File,
        data: &[(String, Tensor, FileKind)],
    ) -> Result<()> {
        let mut txt_data = vec![];
        let mut embed_data = vec![];

        for (txt, tensor, file) in data.iter() {
            let mut txtbytes = txt.as_bytes().to_vec();
            // text_file.write_all(txtbytes).unwrap();
            let data = Data {
                file: file.clone(),
                start: self.text_end,
                length: txtbytes.len(),
                deleted: false,
                indexed: false,
            };
            self.text_end += txtbytes.len();
            let mut tensorbytes = vec![];
            tensor.write_bytes(&mut tensorbytes).unwrap();

            self.embed_end += tensorbytes.len();

            self.data.push(data);

            txt_data.append(&mut txtbytes);
            embed_data.append(&mut tensorbytes);
        }

        self.save().unwrap();
        embed_file.write_all(&embed_data)?;
        text_file.write_all(&txt_data)?;

        embed_file.sync_all().unwrap();
        text_file.sync_all().unwrap();

        Ok(())
    }

    pub fn save(&self) -> Result<()> {
        let storebytes = bincode::serialize(&self).unwrap();
        std::fs::write(self.dir.join(STORE_FILE), &storebytes).unwrap();

        Ok(())
    }

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
            .search_approximate(qry, top_k, cutoff)
            .unwrap()
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

    pub fn files(&self) -> Result<(File, File)> {
        let text = OpenOptions::new()
            .append(true)
            .open(self.dir.join(TEXT_FILE))
            .unwrap();

        let embed = OpenOptions::new()
            .append(true)
            .open(self.dir.join(EMBED_FILE))
            .unwrap();

        Ok((text, embed))
    }

    fn build_index(&mut self) -> Result<()> {
        let mut reader = fs::File::open(self.dir.join(EMBED_FILE)).unwrap();
        let mut ids = vec![];
        let mut tensors = vec![];

        for (i, _) in self.data.iter().enumerate() {
            let mut buffer = [0_u8; 4096];
            reader.read_exact(&mut buffer).unwrap();

            let tensor = Tensor::from_raw_buffer(
                &buffer,
                candle_core::DType::F32,
                &[1024],
                &candle_core::Device::Cpu,
            )
            .unwrap();

            ids.push(i);
            tensors.push(tensor);
        }

        let tensors = Tensor::stack(&tensors[..], 0).unwrap();
        let ann = ANNIndex::build_index(32, 32, &tensors, &ids).unwrap();
        self.index = Some(ann);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Seek},
        path::Path,
        time::Instant,
    };

    use anyhow::Result;
    use candle_core::IndexOp;
    use serde::Deserialize;

    use crate::{embed::Embed, stella::STELLA_MAX_BATCH};

    use super::{FileKind, Store};

    #[derive(Deserialize)]
    struct WikiNews {
        title: String,
        text: String,
    }

    #[test]
    fn storage_init() -> Result<()> {
        let mut store = Store::load_from_file(Path::new("../test-data")).unwrap();

        let data = {
            let data = std::fs::read_to_string("../test-data/wiki-smoll.json").unwrap();
            serde_json::from_str::<Vec<WikiNews>>(&data).unwrap()
        };

        let mut embed = Embed::new().unwrap();

        let chunks = data.chunks(STELLA_MAX_BATCH);

        println!("Begin insert!");
        let (mut text_file, mut embed_file) = store.files()?;

        for (i, c) in chunks.enumerate().skip(85).take(128) {
            let c = c
                .iter()
                .map(|t| format!("## {}\n{}", t.title, t.text))
                .collect::<Vec<_>>();

            let s = Instant::now();
            let tensor = if let Ok(e) = embed.embeddings(&c) {
                e
            } else {
                continue;
            };
            println!(
                "Embedding generation @{i}: {}ms",
                (Instant::now() - s).as_millis()
            );

            let tosave = c
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    (
                        t.to_string(),
                        tensor.i(i).unwrap(),
                        FileKind::Text(Path::new("../test-data/wiki-smoll.json").to_path_buf()),
                    )
                })
                .collect::<Vec<_>>();
            let s = Instant::now();
            store
                .insert(&mut text_file, &mut embed_file, &tosave[..])
                .unwrap();
            println!("Store insert @{i}: {}ms", (Instant::now() - s).as_millis());
        }

        println!("Preparing to test ..{}", store.data.len());

        // Ok, now let's test we have saved it right or not
        for (i, d) in store.data.iter().enumerate() {
            let mut df = store.data_file.as_ref().unwrap().lock().unwrap();
            let mut buf = vec![0_u8; d.length];
            df.seek(std::io::SeekFrom::Start(d.start as u64)).unwrap();
            df.read_exact(&mut buf).unwrap();
            let str = std::str::from_utf8(&buf).unwrap();
            // println!("{str} ## {}\n{}", data[i].title, data[i].text);
            assert_eq!(str, format!("## {}\n{}", data[i].title, data[i].text));
        }

        Ok(())
    }

    #[test]
    fn storage_read() -> Result<()> {
        let store = Store::load_from_file(Path::new("../test-data")).unwrap();

        let mut embed = Embed::new().unwrap();
        let qry = embed
            .query("What are some latest news about Iraq?")
            .unwrap()
            .to_device(&candle_core::Device::Cpu)
            .unwrap();

        let res = store.search(&qry, 4, None).unwrap();
        println!("{:?}", res);
        Ok(())
    }
}
