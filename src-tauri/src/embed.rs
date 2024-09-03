use anyhow::{anyhow, Result};
use candle_core::{backend::BackendDevice, Device, MetalDevice};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use crate::stella::EmbedHead;

// A container for our embedding related tasks
pub struct Embed {
    device: Device,
    model: crate::stella::Stella,
    tokenizer: Tokenizer
}


impl Embed {
    pub fn new() -> Result<Self> {
        // Config straitup copied from https://huggingface.co/dunzhang/stella_en_1.5B_v5/blob/main/config.json
        let cfg = crate::stella::Config {
            hidden_act: candle_nn::Activation::Silu,
            vocab_size: 151646,
            hidden_size: 1536,
            intermediate_size: 8960,
            num_hidden_layers: 28,
            num_attention_heads: 12,
            num_key_value_heads: 2,
            max_position_embeddings: 131072,
            sliding_window: 131072,
            max_window_layers: 21,
            tie_word_embeddings: false,
            rope_theta: 1000000.,
            rms_norm_eps: 1e-06,
            use_sliding_window: false,
            lm_head: EmbedHead {
                in_features: 1536,
                out_features: 1024
            }
        };

        let device = candle_core::Device::Metal(MetalDevice::new(0)?);

        // unsafe inherited from candle_core::safetensors
        let qwen = unsafe {
            VarBuilder::from_mmaped_safetensors(&["../models/qwen2.safetensors"],
            candle_core::DType::F16,
            &device)?
        };

        let head = unsafe {
            VarBuilder::from_mmaped_safetensors(&["../models/embed_head1024.safetensors"],
            candle_core::DType::F16,
            &device)?
        };

        let model = crate::stella::Stella::new(&cfg, qwen, head)?;
        let tokenizer = Tokenizer::from_file("../models/qwen_tokenizer.json").map_err(|e| anyhow!(e))?;

        Ok(Self {
            device,
            model,
            tokenizer
        })
    }
}


#[cfg(test)]
mod tests {
    use anyhow::{anyhow, Result};
    use candle_core::Tensor;

    use super::Embed;

    #[test]
    fn embed() -> Result<()> {
        let mut embed = Embed::new()?;
        let tk = embed.tokenizer.encode_batch(vec![
            "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: What are some ways to reduce stress?",
            "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: Hello world!"
        ], false).map_err(|e| anyhow!(e))?;

        for t in tk.iter() {
            let tensor = Tensor::from_iter(t.get_ids().to_vec(), &embed.device)?.unsqueeze(0)?;
            let out = embed.model.forward(&tensor, 0)?;
            println!("{:?}", t.get_ids());
            println!("{:?}", out.shape());
        }
        Ok(())
    }
}