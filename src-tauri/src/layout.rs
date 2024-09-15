// This file is heavily copied from https://github.com/styrowolf/layoutparser-ort/blob/master/src/models/detectron2.rs Licensed Apache 2.0
// Simplified for our case

use std::fmt::Display;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use image::imageops;
use ort::{GraphOptimizationLevel, Session, SessionOutputs};
use rayon::slice::ParallelSliceMut;

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum DetectedElem {
    Text,
    Title,
    List,
    Table,
    Figure,
}

impl Display for DetectedElem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Text => "Text",
                Self::Title => "Title",
                Self::List => "List",
                Self::Table => "Table",
                Self::Figure => "Figure",
            }
        )
    }
}

#[derive(Debug)]
pub struct ElemFromImage {
    kind: DetectedElem,
    bbox: [f32; 4],
}

/// A [`Detectron2`](https://github.com/facebookresearch/detectron2)-based model.
pub struct Detectron2Model {
    model: Session,
    label_map: [DetectedElem; 5],
}

// Copied from: https://github.com/styrowolf/layoutparser-ort/blob/master/src/utils.rs
fn vec_to_bbox<T: Copy>(v: Vec<T>) -> [T; 4] {
    [v[0], v[1], v[2], v[3]]
}

impl Detectron2Model {
    /// Required input image width.
    pub const REQUIRED_WIDTH: usize = 800;
    /// Required input image height.
    pub const REQUIRED_HEIGHT: usize = 1035;
    /// Default confidence threshold for detections.
    pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.8;

    pub fn new() -> Result<Self> {
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .commit_from_file("../models/layout.onnx")?;

        println!("{:?}", model.outputs);

        Ok(Self {
            model,
            label_map: [
                DetectedElem::Text,
                DetectedElem::Title,
                DetectedElem::List,
                DetectedElem::Table,
                DetectedElem::Figure,
            ],
        })
    }

    pub fn predict(&self, page: &image::DynamicImage) -> Result<Vec<()>> {
        let (img_width, img_height, input) = self.preprocess(page)?;
        // let hm = HashMap::from([("x.1".to_string(), input)]);
        let res = self.model.run(ort::inputs!["x.1" => input]?)?;

        let elements = self.postprocess(res, img_width, img_height)?;
        println!("{elements:?}");

        Ok(vec![])
    }

    // Resizes an image to the required format!
    fn preprocess(&self, img: &image::DynamicImage) -> Result<(u32, u32, ort::Value)> {
        // TODO: re-visit this and resize smarter
        let (img_width, img_height) = (img.width(), img.height());
        let img = img.resize_exact(
            Self::REQUIRED_WIDTH as u32,
            Self::REQUIRED_HEIGHT as u32,
            imageops::FilterType::Triangle,
        );

        let img = img.to_rgb8().into_raw();
        let t = Tensor::from_vec(
            img,
            (Self::REQUIRED_HEIGHT, Self::REQUIRED_WIDTH, 3),
            &Device::Cpu,
        )?
        .to_dtype(DType::F32)?
        .permute((2, 0, 1))? // shape: [3, height, width]
        .to_vec3::<f32>()?
        .concat()
        .concat();

        println!("Before input create ..");
        let input =
            ort::Value::from_array(([3, Self::REQUIRED_HEIGHT, Self::REQUIRED_WIDTH], &t[..]))?;
        println!("After input create ..");
        Ok((img_width, img_height, input.into()))
    }

    fn postprocess(
        &self,
        outputs: SessionOutputs<'_, '_>,
        width: u32,
        height: u32,
    ) -> Result<Vec<ElemFromImage>> {
        let bboxes = &outputs[0].try_extract_tensor::<f32>()?;
        let labels = &outputs[1].try_extract_tensor::<i64>()?;
        let confidence = &outputs[3].try_extract_tensor::<f32>()?;

        let width_factor = width as f32 / Self::REQUIRED_WIDTH as f32;
        let height_factor = height as f32 / Self::REQUIRED_HEIGHT as f32;

        let mut elements = bboxes
            .rows()
            .into_iter()
            .zip(labels.iter().zip(confidence.iter()))
            .filter_map(|(bbox, (&label, &confidence))| {
                if confidence < 0.8 {
                    return None;
                }

                let label = self.label_map.get(label as usize)?;
                // We don't have any way of interpreting Figure and Table as text
                // So, we'll skip that
                if label == &DetectedElem::Figure || label == &DetectedElem::Table {
                    return None;
                }
                let [x1, y1, x2, y2] = vec_to_bbox(bbox.iter().copied().collect::<Vec<_>>());
                Some(ElemFromImage {
                    kind: *label,
                    bbox: [
                        x1 * width_factor,
                        y1 * height_factor,
                        x2 * width_factor,
                        y2 * height_factor,
                    ],
                })
            })
            .collect::<Vec<_>>();

        elements.par_sort_unstable_by(|a, b| {
            (a.bbox[1].max(a.bbox[3])).total_cmp(&(b.bbox[1].max(b.bbox[3])))
        });

        Ok(elements)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::Detectron2Model;

    #[test]
    fn single_page_layout() -> Result<()> {
        let d2model = Detectron2Model::new()?;
        let img = image::open("../test-data/paper-image.jpg")?;

        d2model.predict(&img)?;

        Ok(())
    }
}
