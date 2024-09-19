use std::path::Path;

use anyhow::{anyhow, Result};
use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_llama::ModelWeights,
};
use serde::Deserialize;
use tokenizers::Tokenizer;

// Sampling constants
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

        // Initializing the sampler
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

    // A utility function to load the model and tokenizer
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

    // Utility function to run the generation loop
    fn generate(&mut self, prompt: &str) -> Result<String> {
        // Tokenize the input
        let input = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!(e))?;

        // Creating a tensor of input tokens
        let mut ip = Tensor::new(input.get_ids(), &self.device)?.unsqueeze(0)?;
        // The forward pass to the first token
        let mut logits = self.model.forward(&ip, 0)?;
        // Sampling the first token
        let mut next = self.sampler.sample(&logits.squeeze(0)?)?;
        // A container for all tokens generated
        let mut all_tokens = vec![next];

        // Forward pass - decoder loop
        for i in input.len()..MAX_NEW_TOKENS {
            ip = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;

            logits = self.model.forward(&ip, i)?;
            next = self.sampler.sample(&logits.squeeze(0)?)?;

            if self.stop_tokens.contains(&next) {
                break;
            }

            all_tokens.push(next);
        }

        // Decode tokens and return result
        Ok(match self.tokenizer.decode(&all_tokens[..], false) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Error generating tokens: {e:?}");
                anyhow::bail!("Error generating tokens")
            }
        })
    }
}

/// A struct to hold `sub queries` and a `topic`
#[derive(Debug, Deserialize)]
pub struct QueryMore {
    #[serde(skip)]
    src: String,
    #[serde(rename = "sub_queries")]
    more: Vec<String>,
    topic: String,
}

#[derive(Debug, Deserialize)]
pub struct GeneratedAnswer {
    evidence: Vec<String>,
    answer: String,
}

impl Generator {
    /// Given a `query` string and a `context` returns a response
    pub fn answer(&mut self, topic: &str, query: &str, context: &str) -> Result<GeneratedAnswer> {
        let prompt = format!(
"<|start_header_id|>system<|end_header_id|>\n\nThis is a context-based question answering system. It retrieves information from provided context to answer user queries.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nThe given context are extracted from a bunch of documents with approximate nearest neighbour search and may contain data that are not relevant to the given query. You want to find out information about \"{topic}\" only if present in the given context.\n\nContext:\n\n```\n{context}\n```\n\n<|eot_id|><|start_header_id|>system<|end_header_id|>\n\nBased on the provided documents, generate a concise and relevant response to the following query:\n\n{query}\n\n\nRequirements:\n- Answer must be supported by at least one datapoint in the context.\n- Use natural language and avoid copying from documents.\n- Extract specific and short phrases from the given context that has been used to support the ansert as evidence\n- Truthfully return empty string (\"\") for answer and empty array for evidence if the given context doesn't contain the answer to the query.\n- Do not write an introduction or summary.\n- Return the response as a valid json of the following Schema.\n\n\nSchema:
{{
  evidence: Array<string>,
  answer: string
}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{{
  \"evidence\": ["
        );

        let tk = self.generate(&prompt)?;

        serde_json::from_str(format!("{{\n  \"evidence\": [{tk}").as_str()).map_err(|e| anyhow!(e))
    }

    /// Given a set of queries and a set of documents, return a list of indices that are relevant to the queries
    pub fn find_relevant(
        &mut self,
        query: &[String],
        docs: &[(usize, String)],
    ) -> Result<Vec<usize>> {
        let docfmt = docs
            .iter()
            .map(|(idx, txt)| format!("Id: {idx}\n{txt}\n-------------"))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
"<|start_header_id|>system<|end_header_id|>\n\nThis is a document relevance identification system. It identifies relevant documents based on a set of queries.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nDocuments:\n```\n{}\n```\n\nQueries:\n```\n- {}\n```\n\n\nTask: Identify the ids of documents relevant to these queries.\n\n\nRequirements:\n- Return an array of relevant numeric ids.\n- Only include indices of documents containing relevant information.\n- If no documents are relevant, return an empty array.\n- Don't add any introduction or summary to the response.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n[",
            docfmt,
            query.join("\n- ")
        );

        let tk = self.generate(&prompt)?;

        serde_json::from_str::<Vec<usize>>(format!("[{tk}").as_str()).map_err(|e| anyhow!(e))
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

        let tk = self.generate(&prompt)?;
        let mut res =
            serde_json::from_str::<QueryMore>(format!("{{\n  \"sub_queries\": [\"{tk}").as_str())?;
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

    #[test]
    fn doc_relevance() -> Result<()> {
        let mut gen = Generator::new(Path::new("../models"), &select_device()?)?;

        let relevance = gen.find_relevant(&[
            "What are the latest news about Iraq?".to_string(),
            "What is happening in Iraq today?".to_string(),
            "Iraq latest events".to_string()
        ], &[
            (
                1,
                "## Bulk of Iraqi debt to Paris Club to be forgiven\nThe nineteen member nations of the Paris Club have agreed to forgive 80 percent of Iraq\'s debt, reports the Reuters news service. The plan will reduce Iraq\'s total debt to the member nations to $7.8 billion, from the original $38.9 billion, over a period of four years.\n\nThe plan is the culmination of a trans-Atlantic struggle over the amount of Iraqi debt to be forgiven, the United States pushing for a 90 to 95 percent reduction while France argued for much less.".to_string()),
            (
                5,
                "## Suicide car bomb attack in Baghdad\n\nOn the one year anniversary of Saddam Hussein\'s capture, 13 Iraqis were killed according to AP in a suicide car bomb attack by an al-Qaida-linked terrorist. The attack took place about 9 AM Monday, when many Iraquis were arriving to work, outside the Green Zone which is home to the Iraqi interim government and several embassies. The region is under severe protection.\n\nAuthorities said the suicide bomber detonated the car while it was in line at a checkpoint. The explosion was very loud and it could be heard throughout the city.".to_string(),
            ),
            (
                2,
                "## Lycos Europe ends its anti-spam campaign\nLycos Europe has ended its anti-spam operation: \"Make Love Not Spam.\" A company spokesperson said the objective of the time-limited campaign was to raise people\'s awareness. The reasons why it ended the campaign was variously reported and speculated in media. The operation, while fairly popular, suffered unexpected troubles and drew criticism from security experts and others from the start.\n\nNext, one of the targeted sites redirected all traffic to the Lycos\' server, making Lycos itself a target. The company had maintained that its server was immune from the attack. Lycos stopped distributing the program on December 3, 2004 and asked clients to \"stay tuned.\" The company later ended the program.".to_string()
            ),
            (
                12,
                "## Saddam Hussein profited roughly 1B by taking funds from UN program\nInvestigators said that Saddam Hussein diverted money from the Oil-for-Food Program to pay millions of dollars to families of suicide bombers from the West Bank and Gaza Strip who carried out attacks on Israeli civilians.\n\nPaul Volcker, a former American official investigating the diverted funds scandal, has taken some heat from advocates demanding that he haul United Nations personnel before the US Congress. His reason for not subjecting them to this degree of open scrutiny is that it would have the perverse effect of pushing them into refusing to cooperate with the investigation at all. He plans to release documentary evidence early next year, when his investigation is complete.".to_string()
            )
        ])?;

        assert_eq!(&[1, 5, 12], &relevance[..]);
        Ok(())
    }

    #[test]
    fn answer() -> Result<()> {
        let mut gen = Generator::new(Path::new("../models"), &select_device()?)?;
        let ans = gen.answer(
            "Iraq",
            "What are the latest news about Iraq?",
            "## Bulk of Iraqi debt to Paris Club to be forgiven\nThe nineteen member nations of the Paris Club have agreed to forgive 80 percent of Iraq\'s debt, reports the Reuters news service. The plan will reduce Iraq\'s total debt to the member nations to $7.8 billion, from the original $38.9 billion, over a period of four years.\n\nThe plan is the culmination of a trans-Atlantic struggle over the amount of Iraqi debt to be forgiven, the United States pushing for a 90 to 95 percent reduction while France argued for much less.\n\n\n## Suicide car bomb attack in Baghdad\n\nOn the one year anniversary of Saddam Hussein\'s capture, 13 Iraqis were killed according to AP in a suicide car bomb attack by an al-Qaida-linked terrorist. The attack took place about 9 AM Monday, when many Iraquis were arriving to work, outside the Green Zone which is home to the Iraqi interim government and several embassies. The region is under severe protection.\n\nAuthorities said the suicide bomber detonated the car while it was in line at a checkpoint. The explosion was very loud and it could be heard throughout the city.\n\n\n## Saddam Hussein profited roughly 1B by taking funds from UN program\nInvestigators said that Saddam Hussein diverted money from the Oil-for-Food Program to pay millions of dollars to families of suicide bombers from the West Bank and Gaza Strip who carried out attacks on Israeli civilians.\n\nPaul Volcker, a former American official investigating the diverted funds scandal, has taken some heat from advocates demanding that he haul United Nations personnel before the US Congress. His reason for not subjecting them to this degree of open scrutiny is that it would have the perverse effect of pushing them into refusing to cooperate with the investigation at all. He plans to release documentary evidence early next year, when his investigation is complete."
        )?;

        println!("{ans:?}");
        Ok(())
    }
}
