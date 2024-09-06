// Reference for this implementation: https://fennel.ai/blog/vector-search-in-200-lines-of-rust/
// Code: https://github.com/fennel-ai/fann

use std::collections::HashSet;

use candle_core::{IndexOp, Tensor};
use dashmap::DashSet;
use rand::seq::SliceRandom;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

// #[derive(Eq, PartialEq, Hash)]
// pub struct HashKey<const N: usize>([u8; N]);

struct HyperPlane {
    coefficients: Tensor,
    constant: f32,
}

impl HyperPlane {
    pub fn point_is_above(&self, point: &Tensor) -> candle_core::Result<bool> {
        Ok((self.coefficients.matmul(point)?.to_scalar::<f32>()? + self.constant) >= 0.0)
    }
}

enum Node<const N: usize> {
    Inner(Box<InnerNode<N>>),
    Leaf(Box<LeafNode<N>>),
}
struct LeafNode<const N: usize>(Vec<usize>);
struct InnerNode<const N: usize> {
    hyperplane: HyperPlane,
    left_node: Node<N>,
    right_node: Node<N>,
}
pub struct ANNIndex<const N: usize> {
    trees: Vec<Node<N>>,
    ids: Vec<u32>,
    values: Vec<Tensor>,
}

impl<const N: usize> ANNIndex<N> {
    fn build_hyperplane(
        indexes: &Vec<usize>,
        all_vecs: &[Tensor],
    ) -> candle_core::Result<(HyperPlane, Vec<usize>, Vec<usize>)> {
        let sample = indexes
            .choose_multiple(&mut rand::thread_rng(), 2)
            .collect::<Vec<_>>();
        // cartesian eq for hyperplane n * (x - x_0) = 0
        // n (normal vector) is the coefs x_1 to x_n
        let (a, b) = (*sample[0], *sample[1]);
        // let coefficients = all_vecs.i(a) -  .subtract_from(&all_vecs[b]);
        let coefficients = (&all_vecs[b] - &all_vecs[a])?;
        let point_on_plane = ((&all_vecs[b] + &all_vecs[a])? / 2.)?;
        // let constant = -coefficients.dot_product(&point_on_plane);
        let constant = coefficients.matmul(&point_on_plane)?.to_scalar::<f32>()?;
        let hyperplane = HyperPlane {
            coefficients,
            constant,
        };
        let (mut above, mut below) = (vec![], vec![]);
        for &id in indexes.iter() {
            if hyperplane
                .point_is_above(&all_vecs[id])
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
        max_size: u32,
        indexes: &Vec<usize>,
        all_vecs: &[Tensor],
    ) -> candle_core::Result<Node<N>> {
        if indexes.len() <= (max_size as usize) {
            return Ok(Node::Leaf(Box::new(LeafNode::<N>(indexes.clone()))));
        }
        let (plane, above, below) = Self::build_hyperplane(indexes, all_vecs)?;
        let node_above = Self::build_a_tree(max_size, &above, all_vecs)?;
        let node_below = Self::build_a_tree(max_size, &below, all_vecs)?;

        Ok(Node::Inner(Box::new(InnerNode::<N> {
            hyperplane: plane,
            left_node: node_below,
            right_node: node_above,
        })))
    }

    fn deduplicate(
        vectors: &Tensor,
        ids: &[u32],
        dedup_vectors: &mut Vec<Tensor>,
        ids_of_dedup_vectors: &mut Vec<u32>,
    ) -> candle_core::Result<()> {
        let mut hashes_seen = HashSet::new();

        let (b, _) = vectors.dims2()?;

        for i in 0..b {
            // let hash_key =
            let mut b = vec![];
            let tensor = vectors.i(i)?;
            tensor.write_bytes(&mut b);
            if !hashes_seen.contains(&b) {
                hashes_seen.insert(b);
                dedup_vectors.push(tensor);
                ids_of_dedup_vectors.push(ids[i]);
            }
        }

        Ok(())
    }

    pub fn build_index(
        num_trees: u32,
        max_size: u32,
        vecs: &Tensor,
        vec_ids: &[u32],
    ) -> candle_core::Result<ANNIndex<N>> {
        let (mut unique_vecs, mut ids) = (vec![], vec![]);
        Self::deduplicate(vecs, vec_ids, &mut unique_vecs, &mut ids)?;
        // Trees hold an index into the [unique_vecs] list which is not
        // necessarily its id, if duplicates existed
        let all_indexes: Vec<usize> = (0..unique_vecs.len()).collect();
        let trees = (0..num_trees)
            .into_par_iter()
            .map(|_| Self::build_a_tree(max_size, &all_indexes, &unique_vecs).unwrap())
            .collect::<Vec<_>>();
        Ok(ANNIndex::<N> {
            trees,
            ids,
            values: unique_vecs,
        })
    }

    fn tree_result(query: &Tensor, n: i32, tree: &Node<N>, candidates: &DashSet<usize>) -> i32 {
        // take everything in node, if still needed, take from alternate subtree
        match tree {
            Node::Leaf(box_leaf) => {
                let leaf_values = &(box_leaf.0);
                let num_candidates_found = (n as usize).min(leaf_values.len());

                leaf_values
                    .iter()
                    .take(num_candidates_found)
                    .for_each(|&c| {
                        candidates.insert(c);
                    });
                // for i in 0..num_candidates_found {
                //     candidates.insert(leaf_values[i]);
                // }
                num_candidates_found as i32
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

    pub fn search_approximate(&self, query: &Tensor, top_k: i32) -> Vec<(i32, f32)> {
        let candidates = DashSet::new();
        self.trees.par_iter().for_each(|tree| {
            Self::tree_result(query, top_k, tree, &candidates);
        });

        todo!()
        // candidates
        //     .into_iter()
        //     .map(|idx| (idx, self.values[idx]..sq_euc_dis(&query)))
        //     .sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        //     .take(top_k as usize)
        //     .map(|(idx, dis)| (self.ids[idx], dis))
        //     .collect()
    }
}
