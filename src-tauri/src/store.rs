// Reference for this implementation: https://fennel.ai/blog/vector-search-in-200-lines-of-rust/
// Code: https://github.com/fennel-ai/fann

use std::{
    collections::HashSet,
    fs::{self, OpenOptions},
    io::{BufReader, BufWriter, Read, Seek, Write},
    path::{Path, PathBuf},
};

use anyhow::Result;
use candle_core::{IndexOp, Tensor};
use rand::seq::SliceRandom;
use rayon::{slice::ParallelSliceMut, vec};
use serde::{Deserialize, Serialize};

struct HyperPlane {
    coefficients: Tensor,
    constant: f32,
}

impl HyperPlane {
    pub fn point_is_above(&self, point: &Tensor) -> candle_core::Result<bool> {
        Ok((self.coefficients.mul(point)?.sum(0)?.to_scalar::<f32>()? + self.constant) >= 0.)
    }
}

enum Node {
    Inner(Box<InnerNode>),
    Leaf(Box<LeafNode>),
}
struct LeafNode(Vec<usize>);
struct InnerNode {
    hyperplane: HyperPlane,
    left_node: Node,
    right_node: Node,
}
pub struct ANNIndex {
    trees: Vec<Node>,
    ids: Vec<usize>,
    values: Tensor,
}

impl ANNIndex {
    fn build_hyperplane(
        indexes: &[usize],
        all_vecs: &Tensor,
    ) -> candle_core::Result<(HyperPlane, Vec<usize>, Vec<usize>)> {
        let sample = indexes
            .choose_multiple(&mut rand::thread_rng(), 2)
            .collect::<Vec<_>>();
        // cartesian eq for hyperplane n * (x - x_0) = 0
        // n (normal vector) is the coefs x_1 to x_n
        let (a, b) = (*sample[0], *sample[1]);
        let coefficients = (all_vecs.i(b) - &all_vecs.i(a)?)?;
        let point_on_plane = ((all_vecs.i(b)? + &all_vecs.i(a)?)? / 2.)?;

        let constant = coefficients
            .mul(&point_on_plane)?
            .sum(0)?
            .to_scalar::<f32>()?;

        let hyperplane = HyperPlane {
            coefficients,
            constant,
        };
        let (mut above, mut below) = (vec![], vec![]);
        for &id in indexes.iter() {
            if hyperplane
                .point_is_above(&all_vecs.i(id).unwrap())
                .map_or(false, |d| d)
            {
                above.push(id)
            } else {
                below.push(id)
            };
        }
        Ok((hyperplane, above, below))
    }

    fn build_a_tree(
        max_size: usize,
        indexes: &[usize],
        data: &Tensor,
    ) -> candle_core::Result<Node> {
        if indexes.len() <= max_size {
            return Ok(Node::Leaf(Box::new(LeafNode(indexes.to_vec()))));
        }
        let (plane, above, below) = Self::build_hyperplane(indexes, data).unwrap();
        let node_above = Self::build_a_tree(max_size, &above, data).unwrap();
        let node_below = Self::build_a_tree(max_size, &below, data).unwrap();

        Ok(Node::Inner(Box::new(InnerNode {
            hyperplane: plane,
            left_node: node_below,
            right_node: node_above,
        })))
    }

    pub fn build_index(
        num_trees: usize,
        max_size: usize,
        data: &Tensor,
        vec_ids: &[usize],
    ) -> candle_core::Result<Self> {
        // Trees hold an index into the [unique_vecs] list which is not
        // necessarily its id, if duplicates existed
        let trees = (0..num_trees)
            .map(|_| Self::build_a_tree(max_size, vec_ids, data).unwrap())
            .collect::<Vec<_>>();
        Ok(Self {
            trees,
            ids: vec_ids.to_owned(),
            values: data.to_owned(),
        })
    }

    fn tree_result(
        query: &Tensor,
        n: usize,
        tree: &Node,
        candidates: &mut HashSet<usize>,
    ) -> usize {
        // take everything in node, if still needed, take from alternate subtree
        match tree {
            Node::Leaf(box_leaf) => {
                let leaf_values = &(box_leaf.0);
                let num_candidates_found = n.min(leaf_values.len());

                leaf_values
                    .iter()
                    .take(num_candidates_found)
                    .for_each(|&c| {
                        candidates.insert(c);
                    });
                num_candidates_found
            }
            Node::Inner(inner) => {
                let above = (inner)
                    .hyperplane
                    .point_is_above(query)
                    .map_or(false, |d| d);
                let (main, backup) = match above {
                    true => (&(inner.right_node), &(inner.left_node)),
                    false => (&(inner.left_node), &(inner.right_node)),
                };
                match Self::tree_result(query, n, main, candidates) {
                    k if k < n => k + Self::tree_result(query, n - k, backup, candidates),
                    k => k,
                }
            }
        }
    }

    pub fn search_approximate(
        &self,
        query: &Tensor,
        top_k: usize,
        cutoff: Option<f32>,
    ) -> candle_core::Result<Vec<(usize, f32)>> {
        let mut candidates = HashSet::new();
        // Find approximate matches
        self.trees.iter().for_each(|tree| {
            Self::tree_result(query, top_k, tree, &mut candidates);
        });

        // Create lists for return
        let mut approx = vec![];
        let mut idxs = vec![];

        candidates
            .into_iter()
            .filter_map(|i| self.values.get(i).ok().map(|t| (i, t)))
            .for_each(|(idx, t)| {
                approx.push(t);
                idxs.push(idx);
            });

        // sort by distance
        // First create a tensor of shape top_k, 1024
        let cutoff = cutoff.map_or(0., |c| c);
        let approx = Tensor::stack(&approx, 0).unwrap();
        let mut res = idxs
            .into_iter()
            .zip(query.matmul(&approx.t()?)?.squeeze(0)?.to_vec1::<f32>()?)
            .filter_map(|(idx, d)| {
                if d >= cutoff {
                    Some((self.ids[idx], d))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        res.par_sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .map_or(std::cmp::Ordering::Equal, |o| o)
        });

        Ok(res[0..top_k.min(res.len())].to_vec())
    }

    /// Reads a file and seeks (1024 * sizeof(f32)) to retrieve a tensor
    /// creates the super tensor (data) and builds index
    pub fn new_from_disk(
        data_file: &Path,
        num_trees: usize,
        max_size: usize,
    ) -> candle_core::Result<Self> {
        todo!()
    }
}

/// The store would represent data that is indexed.
/// Upon initiation it'd read (or create) a .store, text.data, embed.data file
/// In `text.data` file we'll maintain bytes of text split into embedding chunks. The start index byte, the length of the chunk and some more metadata will be maintained
/// in `struct Data`
/// In `embedding.data` file we'll maintain byte representations of the tensor, one per each segment.
/// `.store` file will maintain a bincode serialized representation of the `struct Store`
#[derive(Serialize, Deserialize)]
pub struct Store {
    #[serde(skip)]
    index: Option<ANNIndex>,
    data: Vec<Data>,
    dir: PathBuf,
    text_end: usize,
    embed_end: usize,
}

#[derive(Serialize, Deserialize)]
pub struct Data {
    file: FileKind,
    start: usize,
    length: usize,
    deleted: bool,
    indexed: bool,
}

#[derive(Serialize, Deserialize, Clone)]
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
        println!("{:?} {}", dir, dir.is_dir());
        let store = if dir.join(STORE_FILE).is_file() {
            let mut store = fs::File::open(STORE_FILE).unwrap();
            let mut buf = vec![];
            store.read_to_end(&mut buf).unwrap();

            let mut store = bincode::deserialize::<Store>(&buf).unwrap();
            // TODO: build ANN here

            store
        } else {
            fs::File::create_new(dir.join(STORE_FILE))?;
            fs::File::create_new(dir.join(EMBED_FILE))?;
            fs::File::create_new(dir.join(TEXT_FILE))?;

            let store = Self {
                index: None,
                data: vec![],
                dir: dir.into(),
                text_end: 0,
                embed_end: 0,
            };

            store.save()?;

            store
        };

        Ok(store)
    }

    pub fn insert(&mut self, data: &[(String, Tensor, FileKind)]) -> Result<()> {
        let mut text = OpenOptions::new()
            .append(true)
            .open(self.dir.join(TEXT_FILE))?;

        let mut embed = OpenOptions::new()
            .append(true)
            .open(self.dir.join(EMBED_FILE))?;

        for (txt, tensor, file) in data.iter() {
            let txtbytes = txt.as_bytes();
            text.write_all(txtbytes).unwrap();
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
            embed.write_all(&tensorbytes[..]).unwrap();

            self.embed_end += tensorbytes.len();

            self.data.push(data);
        }

        self.save()?;

        Ok(())
    }

    pub fn save(&self) -> Result<()> {
        let storebytes = bincode::serialize(&self)?;
        std::fs::write(self.dir.join(STORE_FILE), &storebytes)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use anyhow::Result;
    use candle_core::{IndexOp, Tensor};
    use serde::Deserialize;

    use crate::{embed::Embed, store::ANNIndex};

    use super::{FileKind, Store};

    const CHUNK: usize = 4;
    const NUM_CHUNKS: usize = 16;

    #[derive(Deserialize)]
    struct WikiNews {
        title: String,
        text: String,
    }

    #[test]
    fn index_and_fetch() -> Result<()> {
        let data = std::fs::read_to_string("../test-data/wiki-smoll.json").unwrap();
        let data = serde_json::from_str::<Vec<WikiNews>>(&data)
            .unwrap()
            .iter()
            .map(|d| format!("## {}\n{}", d.title, d.text))
            .collect::<Vec<_>>();

        let mut embed = Embed::new().unwrap();

        let chunks = data.chunks(CHUNK).take(NUM_CHUNKS);
        let mut all_tensors = vec![];
        for c in chunks {
            if let Ok(e) = embed.embeddings(c) {
                all_tensors.push(e);
            } else {
                continue;
            }
        }

        let tensor = Tensor::stack(&all_tensors, 0)
            .unwrap()
            .reshape((CHUNK * NUM_CHUNKS, 1024))
            .unwrap();
        println!("{:?}", tensor.shape());

        let store = ANNIndex::build_index(
            5,
            16,
            &tensor,
            &(0..CHUNK * NUM_CHUNKS).collect::<Vec<_>>()[..],
        )
        .unwrap();

        println!("Indexed!!");
        let qry = embed.query("What are the latest news about Iraq?")?;
        let res = store.search_approximate(&qry, 4, Some(0.4))?;

        println!("Found matches: {}", res.len());
        for (idx, similarity) in res.iter() {
            println!("---------------\n{}\n[{idx} {}]", data[*idx], similarity);
        }
        Ok(())
    }

    #[test]
    fn storage() -> Result<()> {
        let mut store = Store::load_from_file(Path::new("../test-data")).unwrap();

        let data = {
            let data = std::fs::read_to_string("../test-data/wiki-smoll.json")?;
            serde_json::from_str::<Vec<WikiNews>>(&data)?
        };

        let mut embed = Embed::new()?;

        let chunks = data.chunks(CHUNK).take(NUM_CHUNKS);

        for c in chunks {
            let c = c
                .iter()
                .map(|t| format!("## {}\n{}", t.title, t.text))
                .collect::<Vec<_>>();
            let tensor = if let Ok(e) = embed.embeddings(&c) {
                e
            } else {
                continue;
            };

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

            store.insert(&tosave[..])?;
        }

        Ok(())
    }
}
