use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};
use rand::{prelude::Distribution, SeedableRng};
use rayon::{iter::{IntoParallelRefIterator, ParallelIterator}, slice::{ParallelSlice, ParallelSliceMut}};

pub struct Sampler {
    temperature: f64,
    top_p: f32,
    top_k: usize,
    idx_list: Vec<usize>
}

pub struct AtomicF32 {
    storage: AtomicU32,
}
impl AtomicF32 {
    pub fn new(value: f32) -> Self {
        let as_u32 = value.to_bits();
        Self { storage: AtomicU32::new(as_u32) }
    }
    pub fn store(&self, value: f32, ordering: Ordering) {
        let as_u32 = value.to_bits();
        self.storage.store(as_u32, ordering)
    }
    pub fn load(&self, ordering: Ordering) -> f32 {
        let as_u32 = self.storage.load(ordering);
        f32::from_bits(as_u32)
    }
}

impl Sampler {
    pub fn new(temperature: f64, top_p: f32, top_k: usize, vocab_size: usize, dev: &Device) -> Self {
        let tk = Tensor::arange(0, vocab_size as u32, dev)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap()
            .par_iter().map(|&i| i as usize)
            .collect::<Vec<_>>();

        Self {
            temperature,
            top_p,
            top_k,
            idx_list: tk
        }
    }

    pub fn sample(&self, t: &Tensor) -> Result<u32> {
        let t = candle_nn::ops::softmax_last_dim(
            &(t / self.temperature)?
        )?;
        let mut idxt = self.idx_list.clone();
        // match t.device() {
        //     Device::Cpu => 
                let rest = t.to_vec1::<f32>()?;
                // Sort indices based on their corresponding values
                idxt.par_sort_unstable_by(|&a, &b| {
                    rest[b].total_cmp(&rest[a])
                });
                // idxt.as_parallel_slice().nth
                // let (indices, _, _) =
                // idxt.as_parallel_slice_mut().select_nth_unstable_by(self.top_k, |&i, &j| rest[j].total_cmp(&rest[i]));
            // }
            // Device::Cuda(_) => {}
            // Device::Metal(_) => {}
        // }
        let top_p_impl = AtomicF32::new(0.);
        let mut idxs = idxt[0 .. self.top_k].par_iter().copied()
                .take_any_while(|&t| {
                    let v = rest[t];
                    let p = top_p_impl.load(Ordering::Relaxed) + v;

                    top_p_impl.store(p, Ordering::Relaxed);

                    p <= self.top_p
                }).collect::<Vec<_>>();
        // println!("Me: {}", idxs.len());
        // A minor hack here!
        // let prs = if idxs.is_empty() {
        //     idxt[ .. 32].par_iter().map(|&i| rest[i as usize]).collect::<Vec<_>>()
        // } else {
        //     
        // };
        if idxs.is_empty() {
            idxs = idxt[.. 32].to_vec();
        }

        let prs = idxs.par_iter().map(|&i| rest[i]).collect::<Vec<_>>();
        let fidx = Self::sample_multinomial(&prs)?;
        
        Ok( idxs[fidx as usize] as u32 )
    }

    fn sample_multinomial(v: &Vec<f32>) -> Result<u32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let distr = rand::distributions::WeightedIndex::new(v)?;
        let next_token = distr.sample(&mut rng) as u32;

        Ok(next_token)
    }
}