use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::{Encoding, PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer};

use crate::utils::select_device;

// A container for our embedding related tasks
pub struct Embed {
    device: Device,
    model: crate::stella::Stella,
    tokenizer: Tokenizer,
}


impl Embed {
    pub fn new() -> Result<Self> {
        // Config straitup copied from https://huggingface.co/dunzhang/stella_en_1.5B_v5/blob/main/config.json
        let cfg = crate::stella::Config::default();

        let device = select_device()?;

        // unsafe inherited from candle_core::safetensors
        let qwen = unsafe {
            VarBuilder::from_mmaped_safetensors(&["../models/qwen2.safetensors"],
            candle_core::DType::F32,
            &device)?
        };

        let head = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &["../models/embed_head1024.safetensors"],
                candle_core::DType::F32,
                &device
            )?
        };

        let model = crate::stella::Stella::new(&cfg, qwen, head)?;
        let mut tokenizer = Tokenizer::from_file("../models/qwen_tokenizer.json").map_err(|e| anyhow!(e))?;
        let pad_id = tokenizer.token_to_id("<|endoftext|>").unwrap();

        tokenizer.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                direction: PaddingDirection::Left,
                pad_id,
                pad_token: "<|endoftext|>".to_string(),
                ..Default::default()
            }));
        

        Ok(Self {
            device,
            model,
            tokenizer,
        })
    }

    pub fn query(&mut self, query: &str) -> Result<Tensor> {
        let qry = format!("Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: {query}");
        let ids = self.tokenize(&[&qry])?;

        let tensor = Tensor::from_iter(ids[0].get_ids().to_vec(), &self.device)?
            .unsqueeze(0)?;

        let shape = tensor.dims2()?;
        let mask = Tensor::from_iter(ids[0].get_attention_mask().to_vec(), &self.device)?.unsqueeze(0)?.to_dtype(DType::U8)?;

        Ok(
            self.model.forward(&tensor, &mask, 0)?.mean(1)?
        )
    }

    pub fn embeddings(&mut self, doc_batch: &[&str]) -> Result<Vec<Tensor>> {

        // let token_batch = self.tokenize(doc_batch)?;
        // let batch = Tensor::stack(
        //     &token_batch.iter()
        //         .map(|t| Tensor::from_iter(t.get_ids().to_vec(), &self.device).unwrap())
        //         .collect::<Vec<_>>()[..],
        //     0
        // )?;

        // let mask = Tensor::stack(
        //     &token_batch.iter()
        //         .map(|t| Tensor::from_iter(t.get_attention_mask().to_vec(), &self.device).unwrap())
        //         .collect::<Vec<_>>()[..],
        //     0
        // )?;

        // println!("{:?} {:?}", batch.shape(), mask.shape());
        
        // let out = self.model.forward(&batch, 0)?;
        // println!("{out}");
        Ok(Vec::new())
    }

    fn tokenize(&self, query: &[&str]) -> Result<Vec<Encoding>> {
        if query.len() == 1 {
            Ok(
                vec![self.tokenizer.encode(query[0], false).map_err(|e| anyhow!(e))?]
            )
        } else {
            self.tokenizer
                .encode_batch(query.to_vec(), false).map_err(|e| anyhow!(e))
        }
    }
}


#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::Embed;

    #[test]
    fn embed() -> Result<()> {
        let mut embed = Embed::new()?;
        let qry = embed.query("Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: What are some ways to reduce stress?")?;
        let docs = embed.embeddings(&[
            "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.",
            "Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.",
        ])?;
        // let tk = embed.tokenizer.encode_batch(vec![
        //     "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: What are some ways to reduce stress?",
        //     "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: Hello world!"
        // ], false).map_err(|e| anyhow!(e))?;

        // for t in tk.iter() {
        //     let tensor = Tensor::from_iter(t.get_ids().to_vec(), &embed.device)?.unsqueeze(0)?;
        //     let out = embed.model.forward(&tensor, 0)?;
        //     println!("{:?}", t.get_ids());
        //     println!("{:?}", out.shape());
        // }
        Ok(())
    }
}