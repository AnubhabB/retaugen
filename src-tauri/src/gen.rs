use std::{path::Path, time::Instant};

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::llama::{Cache, Config, Llama, LlamaConfig},
};
use serde::Deserialize;
use tokenizers::{Encoding, Tokenizer};

// Sampling constants
const TEMPERATURE: f64 = 0.8;
const TOP_P: f64 = 0.95;
const TOP_K: usize = 32;

/// A struct to maintain a initialized Llama quantized `gguf` model and associated methods
pub struct Generator {
    cfg: Config,
    device: Device,
    model: Llama,
    tokenizer: Tokenizer,
    sampler: LogitsProcessor,
    stop_tokens: [u32; 2],
}

impl Generator {
    // Download model `safetensor` files into your project dir `models` folder
    // I'm using LLaMA3.1 8B instruct, you can use whatever you want
    const MODEL_FILES: [&'static str; 2] = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ];
    const TOKENIZER_FILE: &'static str = "llama_tokenizer.json";
    const MODEL_CONFIG_FILE: &'static str = "llama_config.json";

    /// Initializer for new llama manager
    pub fn new(dir: &Path, device: &Device) -> Result<Self> {
        let mut device = device.to_owned();
        if let candle_core::Device::Metal(mut m) = device {
            m.set_use_mlx_mm(false);
            device = Device::Metal(m);
        }

        let (model, mut cfg, tokenizer) = Self::load_model(dir, &device)?;
        let stop_tokens = [
            tokenizer.token_to_id("<|eot_id|>").unwrap(),
            tokenizer.token_to_id("<|end_of_text|>").unwrap(),
        ];
        cfg.max_position_embeddings = 4096;

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
            cfg,
            device: device.clone(),
            model,
            tokenizer,
            sampler,
            stop_tokens,
        })
    }

    // A utility function to load the model and tokenizer
    fn load_model(model_dir: &Path, device: &Device) -> Result<(Llama, Config, Tokenizer)> {
        // let model_file = model_dir.join(Self::MODEL_FILE);
        let tok_file = model_dir.join(Self::TOKENIZER_FILE);
        let cfg_file = model_dir.join(Self::MODEL_CONFIG_FILE);
        let model_files = Self::MODEL_FILES
            .iter()
            .map(|mf| model_dir.join(mf))
            .collect::<Vec<_>>();

        println!("Loading LLaMA ..");
        let start = Instant::now();
        let cfg =
            serde_json::from_slice::<LlamaConfig>(&std::fs::read(&cfg_file)?)?.into_config(false);

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, DType::BF16, device)? };
        let llama = Llama::load(vb, &cfg)?;
        println!("LLaMA loaded in {}s", (Instant::now() - start).as_secs());

        let tokenizer = Tokenizer::from_file(tok_file).unwrap();

        Ok((llama, cfg, tokenizer))
    }

    /// Utility function to run the generation loop
    pub fn generate(&mut self, prompt: &str) -> Result<String> {
        // Tokenize the input
        let input = self.tokenize(prompt)?;
        if input.len() >= self.cfg.max_position_embeddings {
            return Err(anyhow!("large input tokens!"));
        }

        let mut cache = Cache::new(true, DType::BF16, &self.cfg, &self.device)?;
        // Creating a tensor of input tokens
        let mut ip = Tensor::new(input.get_ids(), &self.device)?.unsqueeze(0)?;

        let mut start = std::time::Instant::now();

        // The forward pass to the first token
        let mut logits = self.model.forward(&ip, 0, &mut cache)?;
        // Sampling the first token
        let mut next = self.sampler.sample(&logits.squeeze(0)?)?;

        println!(
            "{} prompt tokens processed @ {}t/s",
            input.len(),
            input.len() as f32 / (std::time::Instant::now() - start).as_secs_f32()
        );
        // A container for all tokens generated
        let mut all_tokens = vec![next];

        start = std::time::Instant::now();
        // Forward pass - decoder loop
        for i in input.len()..self.cfg.max_position_embeddings {
            ip = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;

            logits = self.model.forward(&ip, i, &mut cache)?;
            next = self.sampler.sample(&logits.squeeze(0)?).unwrap();
            if self.stop_tokens.contains(&next) {
                break;
            }

            all_tokens.push(next);
        }
        println!(
            "{} tokens generated @ {}t/s",
            all_tokens.len() - 1,
            (all_tokens.len() - 1) as f32 / (std::time::Instant::now() - start).as_secs_f32()
        );

        // Decode tokens and return result
        Ok(match self.tokenizer.decode(&all_tokens[..], false) {
            Ok(t) => t,
            Err(e) => {
                println!("Error generating tokens: {e:?}");
                anyhow::bail!("Error generating tokens")
            }
        })
    }

    /// Small helper to tokenize a given text
    pub fn tokenize(&self, txt: &str) -> Result<Encoding> {
        self.tokenizer.encode(txt, true).map_err(|e| anyhow!(e))
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

impl QueryMore {
    pub fn new(src: &str) -> Self {
        Self {
            src: src.to_string(),
            more: vec![],
            topic: String::new(),
        }
    }
    pub fn source(&self) -> &str {
        &self.src
    }

    pub fn sub_queries(&self) -> &[String] {
        &self.more[..]
    }

    pub fn topic(&self) -> &str {
        &self.topic
    }

    pub fn queries(&self) -> Vec<String> {
        [&[self.source().to_string()], self.sub_queries()].concat()
    }
}

impl Generator {
    /// Preprocesses a query to generate `topic` and supplimental queries for `Fusion Retrieval`
    pub fn query_preproc(&mut self, query: &str, num_sub_qry: usize) -> Result<QueryMore> {
        let prompt = format!(
"<|start_header_id|>system<|end_header_id|>

You are a smart and intelligent AI assistant generating sub-queries and a topic for a Fusion Retrieval system based on a given source query. You always adhere to the given requirements.<|eot_id|><|start_header_id|>user<|end_header_id|>

Given a source query that may require additional context or specific information, generate relevant sub-queries to retrieve more accurate results. Identify a word or a very short phrase that represents the topic of the query.


Source Query:
{query}


Generate {num_sub_qry} relevant sub-queries that:
- Are closely related to the source query
- Can be used to retrieve additional context or specific information
- Are concise and clear


Requirements:
- Sub-queries should not repeat the source query
- Sub-queries should be relevant to the source query's intent, purpose and context
- use natural language for sub queries
- your answer should be a valid json of the following schema.


Schema:

{{
  sub_queries: Array<string>,
  topic: string
}}


Answer must be a valid json.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{
    \"sub_queries\": [\""
        );

        let tk = self.generate(&prompt)?;
        let mut res =
            serde_json::from_str::<QueryMore>(format!("{{\n  \"sub_queries\": [\"{tk}").as_str())?;
        res.src = query.to_string();

        Ok(res)
    }
}

#[derive(Debug, Deserialize)]
pub struct Evidence {
    #[serde(rename = "source_id")]
    index: usize,
    #[serde(rename = "excerpt")]
    text: String,
}

impl Evidence {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn text(&self) -> &str {
        &self.text
    }
}

#[derive(Debug, Deserialize)]
pub struct GeneratedAnswer {
    evidence: Vec<Evidence>,
    answer: String,
}

impl GeneratedAnswer {
    pub fn evidence(&self) -> &[Evidence] {
        &self.evidence[..]
    }

    pub fn answer(&self) -> &str {
        &self.answer
    }
}

impl Generator {
    /// Given a `query` string and a `context` returns a response
    pub fn answer(&mut self, topic: &str, query: &str, context: &str) -> Result<GeneratedAnswer> {
        let prompt = format!(
"<|start_header_id|>system<|end_header_id|>

You are a smart, intelligent, attentive and diligient AI. You are analysing, understanding and retrieving information from given context to answer user's query. Based on the provided context generate a informative, complete, relevant yet concise response to the query. You must follow the given requirements while writing your answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

The given Context contains information extracted from a corpus based on an approximate search. You have to find out information about \"{topic}\" and respond to the user's query only if it can be derived from the given context.

Context:
```
{context}
```

Query:
```
{query}
```

Requirements:
- Your answer must be supported by at least one source from the given context. Extract the supporting text data along with the associated source id as evidence.
- Use natural language summary for your answer and avoid copying from given context for your answer.
- Truthfully return empty string (\"\") for answer if the given context doesn't contain the answer to the query.
- Do not write an introduction or summary.
- Your response must be a valid json of the following Schema.

Schema:
{{
    evidence: Array<{{source_id: int, excerpt: string}}>,
    answer: string
}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{
    \"evidence\": ["
        );

        // println!("Answer prompt:\n{prompt}");

        let mut tk = self.generate(&prompt)?;

        if !tk.ends_with("}") {
            tk = format!("{tk}}}");
        }

        serde_json::from_str(format!("{{\n\t\"evidence\": [{tk}").as_str()).map_err(|e| {
            println!("Answer.Error:\n{tk}");
            anyhow!(e)
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct Relevant {
    relevant: Vec<DocRelevance>,
}

#[derive(Debug, Deserialize)]
pub struct DocRelevance {
    id: usize,
    score: f32,
}

impl Relevant {
    fn to_list(&self) -> Vec<(usize, f32)> {
        self.relevant
            .iter()
            .map(|r| (r.id, r.score))
            .collect::<Vec<_>>()
    }
}

impl Generator {
    /// Given a set of queries and a set of documents, return a list of indices that are relevant to the queries
    pub fn find_relevant(
        &mut self,
        query: &[String],
        docs: &[(usize, String)],
    ) -> Result<Vec<(usize, f32)>> {
        let docfmt = docs
            .iter()
            .map(|(idx, txt)| format!("Id: {idx}\n{txt}\n-------------"))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
"<|start_header_id|>system<|end_header_id|>

You are an intelligent and dilligient AI who analyses text documents to figure out if a particular document contains relevant information for answering a set of queries. You must follow the given requirements while analysing and scoring the documents for your answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

Documents:
```
{}
```

Queries:
```
- {}
```

Task:
Identify the ids of documents that are relevant for generating answers to the given queries and rate them in a scale of 1-10 where a score of 10 is most relevant.

Requirements:
- Only include ids of documents containing relevant information.
- If no documents are relevant the field \"relevant\" must be an empty array.
- Do not write any note, introduction, summary or justifications.
- Your answer must be a valid JSON of the following Schema.

Schema:
{{
    \"relevant\": Array<{{\"id\": numeric id, \"score\": numeric score}}>
}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{
    \"relevant\": [",
            docfmt,
            query.join("\n- ")
        );

        // println!("Relevance prompt:\n{prompt}");
        let tk = self.generate(&prompt)?;

        match serde_json::from_str::<Relevant>(format!("{{\n\t\"relevant\": [{tk}").as_str()) {
            Ok(d) => Ok(d.to_list()),
            Err(e) => {
                println!("Generator::find_relevant: error while deserializing: {e:?}\n{tk:?}\n");
                Err(anyhow!(e))
            }
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Summary {
    heading: String,
    summary: String,
}

impl Summary {
    pub fn summary(&self) -> &str {
        &self.summary
    }

    pub fn heading(&self) -> &str {
        &self.heading
    }
}

impl Generator {
    /// Generates summaries of given text
    pub fn summarize(&mut self, queries: &str, context: &str) -> Result<Summary> {
        let prompt = format!(
"<|start_header_id|>system<|end_header_id|>

You are a smart and intelligent AI assistant generating a heading and summary of a given data so that it can be used for answering the user queries.<|eot_id|><|start_header_id|>user<|end_header_id|>

Queries:
```
{queries}
```

Data:
```
{context}
```


Generate a short summary and a heading for the given data that:
- Reflects the essence, tone and information of the data
- Retains all key facts
- Are concise and clear
- Can be used as evidence to answer given queries


Requirements:
- Heading should reflect the topic and essence of the data
- Summary and heading should be relevant to the source data's intent, purpose and context
- use natural language for summary
- All key facts should be retained
- Summary should not be more than 350 words


Schema:
{{
    heading: string,
    summary: string
}}


Answer must be a valid json.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{
    \"heading\": \""
        );

        let tk = self.generate(&prompt)?;

        serde_json::from_str::<Summary>(format!("{{\n   \"heading\": \"{tk}").as_str())
            .map_err(|e| anyhow!(e))
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

        println!("{relevance:?}");
        // assert_eq!(&[1, 5, 12], &relevance[..]);
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

    #[test]
    fn temp_count_tokens() -> Result<()> {
        let prompt = "<|start_header_id|>system<|end_header_id|>

You are a context-based question answering AI. You retrieve information from provided context to answer user queries. Based on the provided context, generate a informative, compete, relevant yet concise response to the given query by following the given requirments.<|eot_id|><|start_header_id|>user<|end_header_id|>

Text in the given context are extracted with approximate search of a text corpus. You want to find out information about \"\" only if present in the given context.


Context:




Query:




Requirements:
- Answer must be supported by at least one datapoint in the given context, extract the supporting text along with the associated source id as evidence.
- Use natural language summary for your answer and avoid copying from given context for your answer.
- Truthfully return empty string (\"\") for answer if the given context doesn't contain the answer to the query.
- Do not write an introduction or summary.
- Your response must be a valid json of the following Schema.


Schema:

{{
    evidence: Array<{{source: int, text: string}}>,
    answer: string
}}
    
Your answer must be a valid json.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{
    \"evidence\": [{{\"source\": ";

        let tokenizer = super::Tokenizer::from_file("../models/llama_tokenizer.json").unwrap();
        let tkz = tokenizer.encode(prompt, true).unwrap();

        println!("{}", tkz.len());
        Ok(())
    }
}
