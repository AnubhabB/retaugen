use std::path::Path;

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stella_en_v5;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use text_splitter::{ChunkConfig, TextSplitter};
use tokenizers::{Encoding, PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer};

use crate::utils::select_device;

// A container for our embedding related tasks
pub struct Embed {
    device: Device,
    model: stella_en_v5::EmbeddingModel,
    tokenizer: Tokenizer,
    splitter: TextSplitter<Tokenizer>,
}

impl Embed {
    // Some constants

    // The max batch size we can pass to the stella model for generating embeddings.
    // The size `8` is based on my 16GB memory considering memory requirements for the application
    // including other models and other runtime memory requirement
    pub const STELLA_MAX_BATCH: usize = 8;
    // Split size is based on the recommended `input` tokens size
    pub const SPLIT_SIZE: usize = 512;
    // The paths to files required for the Stella_en_1.5B_v5 to run
    // Check example: https://github.com/huggingface/candle/tree/main/candle-examples/examples/stella-en-v5 for smaller model
    pub const BASE_MODEL_FILE: &'static str = "qwen2.safetensors";
    pub const HEAD_MODEL_FILE: &'static str = "embed_head1024.safetensors";
    pub const TOKENIZER_FILE: &'static str = "qwen_tokenizer.json";

    pub fn new(dir: &Path) -> Result<Self> {
        let cfg = stella_en_v5::Config::new_1_5_b_v5(stella_en_v5::EmbedDim::Dim1024);

        let device = select_device()?;

        // unsafe inherited from candle_core::safetensors
        let qwen = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join(Self::BASE_MODEL_FILE)],
                candle_core::DType::BF16,
                &device,
            )?
        };

        let head = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join(Self::HEAD_MODEL_FILE)],
                candle_core::DType::F32,
                &device,
            )?
        };

        let model = stella_en_v5::EmbeddingModel::new(&cfg, qwen, head)?;
        let mut tokenizer =
            Tokenizer::from_file(dir.join(Self::TOKENIZER_FILE)).map_err(|e| anyhow!(e))?;
        let pad_id = tokenizer.token_to_id("<|endoftext|>").unwrap();

        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Left,
            pad_id,
            pad_token: "<|endoftext|>".to_string(),
            ..Default::default()
        }));

        let splitter = TextSplitter::new(
            ChunkConfig::new(Self::SPLIT_SIZE)
                .with_sizer(tokenizer.clone())
                .with_overlap(Self::SPLIT_SIZE / 4)?,
        );

        Ok(Self {
            device,
            model,
            tokenizer,
            splitter,
        })
    }

    // Prepends `prompt` template and tokenizes a `query`
    pub fn query(&mut self, query_batch: &[String]) -> Result<Tensor> {
        let q = query_batch
            .par_iter().map(|q| format!("Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery:{q}"))
            .collect::<Vec<_>>();

        self.embeddings(&q)
    }

    // Tokenizes a doc text batch
    pub fn embeddings(&mut self, batch: &[String]) -> Result<Tensor> {
        let mut token_batch = self.tokenize(batch)?;
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

        Ok(self.model.forward(&ids, &masks)?)
    }

    fn tokenize(&self, doc: &[String]) -> Result<Vec<Encoding>> {
        self.tokenizer
            .encode_batch(doc.to_vec(), true)
            .map_err(|e| anyhow!(e))
    }

    pub fn split_text_and_encode(
        &mut self,
        doc: &str,
    ) -> Vec<(std::string::String, candle_core::Tensor)> {
        let splits = self.splitter.chunks(doc).collect::<Vec<_>>();

        splits
            .chunks(Self::STELLA_MAX_BATCH)
            .flat_map(|c| {
                let embed = self
                    .embeddings(&c.iter().map(|c| c.to_string()).collect::<Vec<_>>()[..])
                    .ok()?;
                Some(
                    c.iter()
                        .enumerate()
                        .filter_map(move |(i, &txt)| {
                            if let Ok(t) = embed.i(i) {
                                Some((txt.to_string(), t))
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .flatten()
            .collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use anyhow::Result;

    use super::Embed;

    #[test]
    fn basic_similarity() -> Result<()> {
        let mut embed = Embed::new(Path::new("../models"))?;

        let qry = embed.query(&["What are some ways to reduce stress?".to_string()])?; // [1, 1024]
        let docs = embed.embeddings(&[
            "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.".to_string(),
            "Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.".to_string(),
        ])?; // [2, 1024]

        // a matmul should do the trick
        let res = qry.matmul(&docs.t()?)?;

        println!("{res}");
        Ok(())
    }

    #[test]
    fn split_and_encode() -> Result<()> {
        let docs = &[
            "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.".to_string(),
            "## President of China lunches with Brazilian President

Brazil: Hu Jintao, the President of the People's Republic of China had lunch today with the President of Brazil, Luiz Inácio Lula da Silva, at the Granja do Torto, the President's country residence in the Brazilian Federal District. Lunch was a traditional Brazilian barbecue with different kinds of meat.
Some Brazilian ministers were present at the event: Antonio Palocci (Economy), Eduardo Campos (Science and Technology), Roberto Rodrigues (Agriculture), Luiz Fernando Furlan (Development), Celso Amorim (Exterior Relations), Dilma Rousseff (Mines and Energy). Also present were Roger Agnelli (Vale do Rio Doce company president) and Eduardo Dutra (Petrobras, government oil company, president).
This meeting is part of a new political economy agreement between Brazil and China where Brazil has recognized mainland China's market economy status, and China has promised to buy more Brazilian products.## Palestinians to elect new president on January 9
Acting president Rawhi Fattuh has announced today that Palestinian elections will be held on January 9. Futtuh, head of the Palestinian parliament, was sworn in hours after the death of Yasser Arafat on Thursday, and Palestinian Basic Law dictates that he may only serve up to two months before elections are held.
New leadership could prove to be the key to revitalizing the peace process in the Middle East, as both Israel and the United States had refused to work with Arafat.
The Haaretz had initially reported that former prime minister Mahmoud Abbas was selected by the Fatah central committee as their candidate for president, but Abbas has denied this, saying, \"the matter is still being discussed.\" There have also been conflicting reports on whether or not jailed Palestinian leader Marwan Barghouti will run.
Barghouti is currently serving five life sentences in Israel for attacks against Israelis. Nonetheless, he remains a popular figure among Palestinians for his role in the Palestinian uprising, and could potentially win the election if he decided to run.
A win by Barghouti could put Israel in an awkward spot; however an Israeli official said this week that he would not be freed, and a landslide win by Barghouti would signify to them that the Palestinians were not yet ready for peace.## Brazilian delegation returns from Arafat funeral
PalestineThe delegation representing Brazil at the funeral of Yasser Arafat returned today, November 13, 2004. The chief-minister of Civil House José Dirceu was a member of the delegation. Unfortunately they arrived too late for the funeral and the delegation watched only part of the funeral activities.
PCdoB (Brazilian communist political party) Deputy Jamil Murad, member of the delegation, said there was a \"deep mourning\" feeling. Jamil Murad had visited Yasser Arafat in April 2004, along with nine other Brazilian deputies. According to Jamil Murad: \"Yasser Arafat was a Palestinian leader who became a world projection leader\". He said Arafat had written him a letter thanking the Brazilian people for their support of the Palestinian cause and saying that he, Arafat, considered President Luiz Inácio Lula da Silva a great world leader.## Hearing begins over David Hookes death
A hearing started today over the death of Australian cricket coach David Hookes. Hookes died after an incident outside a hotel in Melbourne, Australia on the 19th of January.
Bouncer Zdravko Micevic, 22, is charged with manslaughter.".to_string()
        ];

        let mut embed = Embed::new(Path::new("../models"))?;
        docs.iter().enumerate().for_each(|(idx, d)| {
            println!("{idx} ---------------");
            let d = embed.split_text_and_encode(d);
            d.iter().enumerate().for_each(|(chunk, (s, _))| {
                println!("Chunk {chunk} ================");
                println!("{s:?}");
            });
        });
        Ok(())
    }
}
