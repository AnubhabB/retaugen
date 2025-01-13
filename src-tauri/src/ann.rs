// Reference for this implementation: https://fennel.ai/blog/vector-search-in-200-lines-of-rust/
// Code: https://github.com/fennel-ai/fann

use candle_core::{IndexOp, Tensor};
use dashmap::DashSet;
use rand::seq::SliceRandom;
use rayon::{
    iter::{IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

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
        // Choosing random indices
        let sample = indexes
            .choose_multiple(&mut rand::thread_rng(), 2)
            .collect::<Vec<_>>();
        // cartesian eq for hyperplane n * (x - x_0) = 0
        // n (normal vector) is the coefs x_1 to x_n
        let (a, b) = (*sample[0], *sample[1]);

        // Calculate the midpoint
        let coefficients = (all_vecs.i(b) - &all_vecs.i(a)?)?;
        let point_on_plane = ((all_vecs.i(b)? + &all_vecs.i(a)?)? / 2.)?;

        // And build the hyperplane
        let constant = coefficients
            .mul(&point_on_plane)?
            .sum(0)?
            .to_scalar::<f32>()?;

        let hyperplane = HyperPlane {
            coefficients,
            constant,
        };

        // Classify the vectors as being above or below
        let (mut above, mut below) = (vec![], vec![]);
        for &id in indexes.iter() {
            if hyperplane.point_is_above(&all_vecs.i(id)?).is_ok_and(|d| d) {
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
        // If the number of elements in a tree has reached the configurable `max size of node` then move on
        if indexes.len() <= max_size {
            return Ok(Node::Leaf(Box::new(LeafNode(indexes.to_vec()))));
        }

        // Recursively keep constructing the tree
        let (plane, above, below) = Self::build_hyperplane(indexes, data)?;
        let node_above = Self::build_a_tree(max_size, &above, data)?;
        let node_below = Self::build_a_tree(max_size, &below, data)?;

        Ok(Node::Inner(Box::new(InnerNode {
            hyperplane: plane,
            left_node: node_below,
            right_node: node_above,
        })))
    }

    fn tree_result(query: &Tensor, n: usize, tree: &Node, candidates: &DashSet<usize>) -> usize {
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
                let above = (inner).hyperplane.point_is_above(query).is_ok_and(|d| d);
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

    /// API for building the index
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

    /// API for search
    pub fn search_approximate(
        &self,
        query: &Tensor,
        top_k: usize,
        cutoff: Option<f32>,
    ) -> candle_core::Result<Vec<(usize, f32)>> {
        let candidates = DashSet::new();
        // Find approximate matches
        self.trees.par_iter().for_each(|tree| {
            Self::tree_result(query, top_k, tree, &candidates);
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
        let approx = Tensor::stack(&approx, 0)?;
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

        res.par_sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        Ok(res[0..top_k.min(res.len())].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use anyhow::Result;
    use candle_core::Tensor;
    use serde::Deserialize;

    use crate::{ann::ANNIndex, embed::Embed};

    #[derive(Deserialize)]
    struct WikiNews {
        title: String,
        text: String,
    }

    const NUM_CHUNKS: usize = 16;

    #[test]
    fn index_and_fetch() -> Result<()> {
        let data = std::fs::read_to_string("../test-data/wiki-smoll.json")?;
        let data = serde_json::from_str::<Vec<WikiNews>>(&data)?
            .iter()
            .map(|d| format!("## {}\n{}", d.title, d.text))
            .collect::<Vec<_>>();

        let mut embed = Embed::new(Path::new("../models"))?;

        let chunks = data.chunks(Embed::STELLA_MAX_BATCH).take(NUM_CHUNKS);
        let mut all_tensors = vec![];
        for c in chunks {
            if let Ok(e) = embed.embeddings(c) {
                all_tensors.push(e);
            } else {
                continue;
            }
        }

        let tensor = Tensor::stack(&all_tensors, 0)?
            .reshape((Embed::STELLA_MAX_BATCH * NUM_CHUNKS, 1024))?;

        let store = ANNIndex::build_index(
            16,
            16,
            &tensor,
            &(0..Embed::STELLA_MAX_BATCH * NUM_CHUNKS).collect::<Vec<_>>()[..],
        )?;

        println!("Indexed!!");
        let qry = embed.query(&["What are the latest news about Iraq?".to_string()])?;
        let res = store.search_approximate(&qry, 4, Some(0.35))?;

        println!("Found matches: {}", res.len());
        for (idx, similarity) in res.iter() {
            println!("---------------\n{}\n[{idx} {}]", data[*idx], similarity);
        }
        Ok(())
    }
}
