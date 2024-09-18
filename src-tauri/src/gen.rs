use std::path::Path;

use anyhow::{anyhow, Result};
use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_llama::ModelWeights,
};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

const TEMPERATURE: f64 = 0.8;
const TOP_P: f64 = 0.95;
const TOP_K: usize = 40;

const MAX_NEW_TOKENS: usize = 2048;

/// A struct to maintain a initialized Llama quantized `gguf` model and associated methods
pub struct Generator {
    device: Device,
    model: ModelWeights,
    tokenizer: Tokenizer,
    sampler: LogitsProcessor,
    stop_tokens: [u32; 2],
}

impl Generator {
    const MODEL_FILE: &'static str = "Meta-Llama-3.1-8B-Instruct.Q8_0.gguf";
    const TOKENIZER_FILE: &'static str = "llama_tokenizer.json";

    /// Initializer for new llama manager
    pub fn new(dir: &Path, device: &Device) -> Result<Self> {
        let (model, tokenizer) = Self::load_model(dir, device)?;
        let stop_tokens = [
            tokenizer.token_to_id("<|eot_id|>").unwrap(),
            tokenizer.token_to_id("<|end_of_text|>").unwrap(),
        ];

        let sampler = LogitsProcessor::from_sampling(
            42,
            Sampling::TopKThenTopP {
                k: TOP_K,
                p: TOP_P,
                temperature: TEMPERATURE,
            },
        );

        println!("Llama ready!");
        Ok(Self {
            device: device.clone(),
            model,
            tokenizer,
            sampler,
            stop_tokens,
        })
    }

    fn load_model(model_dir: &Path, device: &Device) -> Result<(ModelWeights, Tokenizer)> {
        let model_file = model_dir.join(Self::MODEL_FILE);
        let tok_file = model_dir.join(Self::TOKENIZER_FILE);

        println!("Loading gguf model @{:?}", model_file);

        let mut file = std::fs::File::open(model_file)?;
        // reading the params from file
        let model = gguf_file::Content::read(&mut file)?;

        let model = ModelWeights::from_gguf(model, &mut file, device)?;

        let tokenizer = Tokenizer::from_file(tok_file).unwrap();

        Ok((model, tokenizer))
    }
}

#[derive(Debug, Deserialize)]
pub struct QueryMore {
    #[serde(skip)]
    src: String,
    #[serde(rename = "sub_queries")]
    more: Vec<String>,
    topic: String,
}

impl Generator {
    /// Given a `query` string and a `context` returns a response
    pub fn answer(&mut self, topic: &str, query: &str, context: &str) -> Result<String> {
        let prompt = format!(
            "<|start_header_id|>system<|end_header_id|>\n\nThis is a context-based question answering system. It retrieves information from provided context to answer user queries.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nThe given context are extracted from a bunch of documents with approximate nearest neighbour search and may contain data that are not relevant to the given query. You want to find out information about \"{topic}\" only if present in the given context.\n\nContext:\n\n```\n{context}\n```\n\n<|eot_id|><|start_header_id|>system<|end_header_id|>\n\nBased on the provided documents, generate a concise and relevant response to the following query:\n\n{query}\n\nRequirements:\n\n- Answer must be supported by at least one datapoint in the context.\n- Use natural language and avoid copying from documents.\n- Truthfully answer \"Not Found\" if the given context doesn't contain the answer to the query.\n- Do not write an introduction or summary<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );

        let input = self
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| anyhow!(e))?;

        println!("Prompt:\n{prompt}\nSize: {}", input.len());

        Ok(String::new())
    }

    /// Preprocesses a query to generate `topic` and supplimental queries for `Fusion Retrieval`
    pub fn query_preproc(&mut self, query: &str, num_sub_qry: usize) -> Result<QueryMore> {
        let prompt = format!(
"<|start_header_id|>system<|end_header_id|>\n\nThis is a query generation system for Fusion Retrieval. It generates relevant sub-queries related to a given source query.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven a source query that may require additional context or specific information, generate relevant sub-queries to retrieve more accurate results. Identify a word or a very short phrase that represents the topic of the `Query`.\n\n\nSource Query:\n{query}\n\nGenerate {num_sub_qry} relevant sub-queries that:\n\n- Are closely related to the source query\n- Can be used to retrieve additional context or specific information\n- Are concise and clear\n\n\nRequirements:\n\n- Sub-queries should not repeat the source query\n- Sub-queries should be relevant to the source query's intent, purpose and context\n- use natural language\n- your answer should be a valid json of the following schema.\n\n\nSchema:\n\n
{{
  sub_queries: Array<string>,
  topic: string
}}\n\nAnswer must be a valid json.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{{\n  \"sub_queries\": [\""
        );

        let input = self
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| anyhow!(e))?;

        println!("Prompt:\n{prompt}\nSize: {}", input.len());

        let mut all_tokens = vec![];

        let mut ip = Tensor::new(input.get_ids(), &self.device)?.unsqueeze(0)?;
        let mut logits = self.model.forward(&ip, 0)?;
        let mut next = self.sampler.sample(&logits.squeeze(0)?)?;

        all_tokens.push(next);

        for i in input.len()..MAX_NEW_TOKENS {
            ip = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;

            logits = self.model.forward(&ip, i)?;
            next = self.sampler.sample(&logits.squeeze(0)?)?;

            if self.stop_tokens.contains(&next) {
                break;
            }

            all_tokens.push(next);
        }

        let tk = match self.tokenizer.decode(&all_tokens[..], false) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Error generating tokens: {e:?}");
                anyhow::bail!("Error generating tokens")
            }
        };
        let tk = format!("{{\n  \"sub_queries\": [\"{tk}");
        let mut res = serde_json::from_str::<QueryMore>(&tk)?;
        res.src = query.to_string();

        Ok(res)
    }
}


#[cfg(test)]
mod tests {
    use std::path::Path;

    use anyhow::Result;

    use crate::utils::select_device;

    use super::Generator;

    #[test]
    fn sub_query() -> Result<()> {
        let mut gen = Generator::new(Path::new("../models"), &select_device()?)?;
        let res = gen.query_preproc(
            "What are the impacts of climate change on the environment?",
            4,
        )?;
        println!("{res:#?}");
        let res = gen.query_preproc("What are the latest news about Iraq?", 4)?;
        println!("{res:#?}");
        Ok(())
    }
}
