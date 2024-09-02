use candle_onnx::onnx::ModelProto;

pub struct Embed {
    model: ModelProto
}

impl Embed {
    pub fn new() -> Self {
        let model = candle_onnx::read_file("../model_int8.onnx").unwrap();
        Self {
            model
        }
    }

    fn generate_embedding(&self, data: &str) {

    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{Device, MetalDevice, Tensor};

    use super::Embed;

    #[test]
    fn embed() {
        let embd = Embed::new();
        let dev = Device::new_metal(0).unwrap();
        println!("{}", embd.model.doc_string);
        let graph = embd.model.graph.as_ref().unwrap();
        println!("{:?} {} {:?}", graph.input[0].name, graph.input[0].doc_string, graph.input[0].r#type);
        let ones = Tensor::ones((1, 512), candle_core::DType::I64, &dev).unwrap();
        let res = candle_onnx::simple_eval(
            &embd.model, 
            HashMap::from([
                    ("input_ids".to_string(), ones)
            ])).unwrap();

        println!("{res:?}");
    }
}