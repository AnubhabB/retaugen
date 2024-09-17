// This file is heavily copied from https://github.com/styrowolf/layoutparser-ort/blob/master/src/models/detectron2.rs Licensed Apache 2.0
// Simplified for our case

use std::fmt::Display;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use image::imageops;
use ort::{GraphOptimizationLevel, Session, SessionOutputs};
use rayon::slice::ParallelSliceMut;

// An emum to represent the classes of regions of interest
// detected by the `layout detection` model
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

/// This struct represents a Region of interest
#[derive(Debug)]
pub struct RegionOfInterest {
    kind: DetectedElem,
    // the bounding box - x1, y1, x2, y2 - top, left, bottom, right
    bbox: [f32; 4],
    confidence: f32,
}

impl RegionOfInterest {
    pub fn kind(&self) -> DetectedElem {
        self.kind
    }

    pub fn bbox(&self) -> [f32; 4] {
        self.bbox
    }

    pub fn confidence(&self) -> f32 {
        self.confidence
    }
}

/// A [`Detectron2`](https://github.com/facebookresearch/detectron2)-based model.
pub struct Detectron2Model {
    model: Session,
    label_map: [DetectedElem; 5],
}

// Copied from: https://github.com/styrowolf/layoutparser-ort/blob/master/src/utils.rs
/// Utility function to convert bbox to a array
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
        // Loading and initializing the model from `onnx` file
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            // We could make this a little more generic with `numcpus` crate
            .with_intra_threads(8)?
            .commit_from_file("../models/layout.onnx")?;

        // You could print the model outputs to figure out which prediction datapoints are useful
        // println!("{:?}", model.outputs);

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

    pub fn predict(&self, page: &image::DynamicImage) -> Result<Vec<RegionOfInterest>> {
        let (img_width, img_height, input) = self.preprocess(page)?;
        // let hm = HashMap::from([("x.1".to_string(), input)]);
        let res = self.model.run(ort::inputs!["x.1" => input]?)?;

        self.postprocess(res, img_width, img_height)
    }

    // 1. Resizes an image to the required format!
    // 2. Creates a tensor from the image
    // 3. Reshapes the tensor to channel first format
    // 4. Creates input ndarray for `ort` to consume
    fn preprocess(&self, img: &image::DynamicImage) -> Result<(u32, u32, ort::Value)> {
        // TODO: re-visit this and resize smarter
        let (img_width, img_height) = (img.width(), img.height());
        let img = img.resize_exact(
            Self::REQUIRED_WIDTH as u32,
            Self::REQUIRED_HEIGHT as u32,
            imageops::FilterType::Triangle,
        );

        let img = img.to_rgb8().into_raw();

        // Read the image as a tensor
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

        // Create a `ndarray` input for `ort` runtime to consume
        let input =
            ort::Value::from_array(([3, Self::REQUIRED_HEIGHT, Self::REQUIRED_WIDTH], &t[..]))?;

        Ok((img_width, img_height, input.into()))
    }

    // Reads the predictions and converts them to regions of interest
    fn postprocess(
        &self,
        outputs: SessionOutputs<'_, '_>,
        width: u32,
        height: u32,
    ) -> Result<Vec<RegionOfInterest>> {
        // Extract predictions for bounding boxes,
        // labels and confidence scores
        // Shape: [num pred, 4]
        let bboxes = &outputs[0].try_extract_tensor::<f32>()?;
        // Shape: [num pred]
        let labels = &outputs[1].try_extract_tensor::<i64>()?;
        // 3 for MASK_RCNN_X_101_32X8D_FPN_3x | 2 for FASTER_RCNN_R_50_FPN_3X
        // Shape: [num pred]
        let confidence = &outputs[3].try_extract_tensor::<f32>()?;

        // We had originally `resized` the image to fit
        // the required input dimensions,
        // we are just going to adjust the predictions to factor in the resize
        let width_factor = width as f32 / Self::REQUIRED_WIDTH as f32;
        let height_factor = height as f32 / Self::REQUIRED_HEIGHT as f32;

        // Iterate over (region bounding boxes, predicted classes/ labels, and confidence scores)
        let mut elements = bboxes
            .rows()
            .into_iter()
            .zip(labels.iter().zip(confidence.iter()))
            .filter_map(|(bbox, (&label, &confidence))| {
                // Skip everything below some confidence score we want to work with
                if confidence < Self::DEFAULT_CONFIDENCE_THRESHOLD {
                    return None;
                }

                // Getting the predicted label from the predicted index
                let label = self.label_map.get(label as usize)?;
                // We don't have any way of interpreting Figure and Table as text
                // So, we'll skip that
                if label == &DetectedElem::Figure || label == &DetectedElem::Table {
                    return None;
                }
                let [x1, y1, x2, y2] = vec_to_bbox(bbox.iter().copied().collect::<Vec<_>>());
                // Adjusting the predicted bounding box to our original image size
                Some(RegionOfInterest {
                    kind: *label,
                    bbox: [
                        x1 * width_factor,
                        y1 * height_factor,
                        x2 * width_factor,
                        y2 * height_factor,
                    ],
                    confidence,
                })
            })
            .collect::<Vec<_>>();

        // Now we sort the predictions to (kind of) visual hierarchy
        // from top left
        elements.par_sort_unstable_by(|a, b| {
            (a.bbox()[1].max(a.bbox()[3])).total_cmp(&(b.bbox()[1].max(b.bbox()[3])))
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

        let pred = d2model.predict(&img)?;
        println!("{pred:?}");

        Ok(())
    }
}
