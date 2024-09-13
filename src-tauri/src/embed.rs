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

pub enum ForEmbed<'a> {
    Query(&'a str),
    Docs(&'a [String]),
}

impl Embed {
    pub fn new() -> Result<Self> {
        // Config straitup copied from https://huggingface.co/dunzhang/stella_en_1.5B_v5/blob/main/config.json
        let cfg = crate::stella::Config::default();

        let device = select_device()?;

        // unsafe inherited from candle_core::safetensors
        let qwen = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &["../models/qwen2.safetensors"],
                candle_core::DType::F32, // TODO: why is this giving `NaN` @ F16
                &device,
            )?
        };

        let head = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &["../models/embed_head1024.safetensors"],
                candle_core::DType::F32,
                &device,
            )?
        };

        let model = crate::stella::Stella::new(&cfg, qwen, head)?;
        let mut tokenizer =
            Tokenizer::from_file("../models/qwen_tokenizer.json").map_err(|e| anyhow!(e))?;
        let pad_id = tokenizer.token_to_id("<|endoftext|>").unwrap();

        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
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
        let tokens = self.tokenize(
            ForEmbed::Query(
                    format!("Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery:{query}").as_str()
                )
            )?;

        let ids = Tensor::from_iter(
            tokens[0]
                .get_ids()
                .get(0..tokens[0].get_ids().len().min(512))
                .unwrap()
                .to_vec(),
            &self.device,
        )?
        .unsqueeze(0)?;
        let mask = Tensor::from_iter(
            tokens[0]
                .get_attention_mask()
                .get(0..tokens[0].get_attention_mask().len().min(512))
                .unwrap()
                .to_vec(),
            &self.device,
        )?
        .unsqueeze(0)?
        .to_dtype(DType::U8)?;

        Ok(self.model.forward(&ids, &mask, 0)?)
    }

    pub fn embeddings(&mut self, doc_batch: ForEmbed<'_>) -> Result<Tensor> {
        let mut token_batch = self.tokenize(doc_batch)?;
        let mut ids = Tensor::zeros(
            (token_batch.len(), token_batch[0].get_ids().len()),
            DType::U32,
            &self.device,
        )?;
        let mut masks = Tensor::zeros(
            (token_batch.len(), token_batch[0].get_ids().len()),
            DType::U8,
            &self.device,
        )?;

        for (i, e) in token_batch.drain(..).enumerate() {
            let input_id = Tensor::from_iter(e.get_ids().to_vec(), &self.device)?.unsqueeze(0)?;
            let mask = Tensor::from_iter(e.get_attention_mask().to_vec(), &self.device)?
                .to_dtype(DType::U8)?
                .unsqueeze(0)?;

            ids = ids.slice_assign(&[i..i + 1, 0..input_id.dims2().unwrap().1], &input_id)?;
            masks = masks.slice_assign(&[i..i + 1, 0..mask.dims2().unwrap().1], &mask)?;
        }

        Ok(self.model.forward(&ids, &masks, 0)?)
    }

    // pub fn split_text(&self, doc_batch: &[String]) -> Result<Vec<>>

    fn tokenize(&self, doc: ForEmbed) -> Result<Vec<Encoding>> {
        match doc {
            ForEmbed::Query(q) => Ok(vec![self
                .tokenizer
                .encode(q, true)
                .map_err(|e| anyhow!(e))?]),
            ForEmbed::Docs(d) => self
                .tokenizer
                .encode_batch(d.to_vec(), true)
                .map_err(|e| anyhow!(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::Embed;

    #[test]
    fn basic_similarity() -> Result<()> {
        let mut embed = Embed::new()?;

        let qry = embed.query("What are some ways to reduce stress?")?; // [1, 1024]
        let docs = embed.embeddings(crate::embed::ForEmbed::Docs(&[
            "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.".to_string(),
            "Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.".to_string(),
        ]))?; // [2, 1024]

        // a matmul should do the trick
        let res = qry.matmul(&docs.t()?)?;

        println!("{res}");
        Ok(())
    }
}
