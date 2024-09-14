// This file is heavily copied from https://github.com/styrowolf/layoutparser-ort/blob/master/src/models/detectron2.rs Licensed Apache 2.0
// Simplified for our case

use anyhow::Result;
use image::imageops;

use crate::{utils::vec_to_bbox, LayoutElement};

/// A [`Detectron2`](https://github.com/facebookresearch/detectron2)-based model.
pub struct Detectron2Model {
    label_map: Vec<(usize, &'static str)>
}

impl Detectron2Model {
    /// Required input image width.
    pub const REQUIRED_WIDTH: u32 = 800;
    /// Required input image height.
    pub const REQUIRED_HEIGHT: u32 = 1035;
    /// Default confidence threshold for detections.
    pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.8;
    
    pub fn new() -> Result<Self> {

        Ok(Self {
            label_map: ["Text", "Title", "List", "Table", "Figure"].iter().enumerate().map(|(i, &s)| (i, s)).collect::<Vec<_>>()
        })
    }
}


impl Detectron2Model {
    

    /// Construct a [`Detectron2Model`] with a pretrained model downloaded from Hugging Face.
    pub fn pretrained(p_model: Detectron2PretrainedModels) -> Result<Self> {
        let session_builder = Session::builder()?;
        let api = hf_hub::api::sync::Api::new()?;
        let filename = api
            .model(p_model.hf_repo().to_string())
            .get(p_model.hf_filename())?;

        let model = session_builder.commit_from_file(filename)?;

        Ok(Self {
            model_name: p_model.name().to_string(),
            model,
            label_map: p_model.label_map(),
            confidence_threshold: Self::DEFAULT_CONFIDENCE_THRESHOLD,
            confidence_score_index: p_model.confidence_score_index(),
        })
    }

    /// Construct a configured [`Detectron2Model`] with a pretrained model downloaded from Hugging Face.
    pub fn configure_pretrained(
        p_model: Detectron2PretrainedModels,
        confidence_threshold: f32,
        session_builder: SessionBuilder,
    ) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()?;
        let filename = api
            .model(p_model.hf_repo().to_string())
            .get(p_model.hf_filename())?;

        let model = session_builder.commit_from_file(filename)?;

        Ok(Self {
            model_name: p_model.name().to_string(),
            model,
            label_map: p_model.label_map(),
            confidence_threshold,
            confidence_score_index: p_model.confidence_score_index(),
        })
    }

    /// Construct a [`Detectron2Model`] from a model file.
    pub fn new_from_file(
        file_path: &str,
        model_name: &str,
        label_map: &[(i64, &str)],
        confidence_threshold: f32,
        confidence_score_index: usize,
        session_builder: SessionBuilder,
    ) -> Result<Self> {
        let model = session_builder.commit_from_file(file_path)?;

        Ok(Self {
            model_name: model_name.to_string(),
            model,
            label_map: label_map.iter().map(|(i, l)| (*i, l.to_string())).collect(),
            confidence_threshold,
            confidence_score_index,
        })
    }

    /// Predict [`LayoutElement`]s from the image provided.
    pub fn predict(&self, img: &image::DynamicImage) -> Result<Vec<LayoutElement>> {
        let (img_width, img_height, input) = self.preprocess(img);

        let run_result = self.model.run(ort::inputs!["x.1" => input]?);
        match run_result {
            Ok(outputs) => {
                let elements = self.postprocess(&outputs, img_width, img_height)?;
                return Ok(elements);
            }
            Err(_err) => {
                tracing::warn!(
                    "Ignoring runtime error from onnx (likely due to encountering blank page)."
                );
                return Ok(vec![]);
            }
        }
    }

    fn preprocess(
        &self,
        img: &image::DynamicImage,
    ) -> (u32, u32, ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>) {
        let (img_width, img_height) = (img.width(), img.height());
        let img = img.resize_exact(
            Self::REQUIRED_WIDTH,
            Self::REQUIRED_HEIGHT,
            imageops::FilterType::Triangle,
        );
        let img_rgb8 = img.into_rgba8();

        let mut input = Array::zeros((3, 1035, 800));

        for pixel in img_rgb8.enumerate_pixels() {
            let x = pixel.0 as _;
            let y = pixel.1 as _;
            let [r, g, b, _] = pixel.2 .0;
            input[[0, y, x]] = r as f32;
            input[[1, y, x]] = g as f32;
            input[[2, y, x]] = b as f32;
        }

        return (img_width, img_height, input);
    }

    fn postprocess<'s>(
        &self,
        outputs: &SessionOutputs<'s>,
        img_width: u32,
        img_height: u32,
    ) -> Result<Vec<LayoutElement>> {
        let bboxes = &outputs[0].try_extract_tensor::<f32>()?;
        let labels = &outputs[1].try_extract_tensor::<i64>()?;
        let confidence_scores =
            &outputs[self.confidence_score_index].try_extract_tensor::<f32>()?;

        let width_conversion = img_width as f32 / Self::REQUIRED_WIDTH as f32;
        let height_conversion = img_height as f32 / Self::REQUIRED_HEIGHT as f32;

        let mut elements = vec![];

        for (bbox, (label, confidence_score)) in bboxes
            .rows()
            .into_iter()
            .zip(labels.iter().zip(confidence_scores))
        {
            let [x1, y1, x2, y2] = vec_to_bbox(bbox.iter().copied().collect());

            let detected_label = &self
                .label_map
                .iter()
                .find(|(l_i, _)| l_i == label)
                .unwrap() // SAFETY: the model always yields one of these labels
                .1;

            if *confidence_score > self.confidence_threshold as f32 {
                elements.push(LayoutElement::new(
                    x1 * width_conversion,
                    y1 * height_conversion,
                    x2 * width_conversion,
                    y2 * height_conversion,
                    &detected_label,
                    *confidence_score,
                    &self.model_name,
                ))
            }
        }

        elements.sort_by(|a, b| a.bbox.max().y.total_cmp(&b.bbox.max().y));

        return Ok(elements);
    }
}