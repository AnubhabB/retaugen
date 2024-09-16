use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use pdfium_render::prelude::{PdfRect, PdfRenderConfig, Pdfium};

use crate::layout::{DetectedElem, Detectron2Model};

const PADDING: f32 = 1.;

pub fn files_to_text(files: &[PathBuf]) -> Result<()> {
    let mut pdfs = vec![];
    
    files.iter()
        .for_each(|f| {
            if let Some(ext) = f.extension() {
                if ext == "pdf" {
                    pdfs.push(f);
                }
            }
        });

    if !pdfs.is_empty() {
        pdf_to_text(&pdfs[..])?;
    }
    
    Ok(())
}

pub fn pdf_to_text(files: &[&PathBuf]) -> Result<()> {
    let pdfium = Pdfium::new(
        Pdfium::bind_to_library(
            Pdfium::pdfium_platform_library_name_at_path(env!("PDFIUM_DYNAMIC_LIB_PATH"))
        )?
    );

    let layout_model = Detectron2Model::new()?;

    let cfg = PdfRenderConfig::new()
        .set_target_width(Detectron2Model::REQUIRED_WIDTH as i32)
        .set_maximum_height(Detectron2Model::REQUIRED_HEIGHT as i32);

    let file_encoded = files.iter().filter_map(|&file| {
        let pdf = pdfium.load_pdf_from_file(file, None).ok()?;

        let page_data = pdf.pages()
            .iter()
            .enumerate()
            .filter_map(|(idx, page)| {
                let img = page.render_with_config(&cfg).ok()?
                    .as_image(); // Renders this page to an image::DynamicImage...

                let w_f = page.width().value / img.width() as f32;
                let h_f = page.height().value / img.height() as f32;

                let res =  layout_model.predict(&img).ok()?.iter().map(|e| {
                    let bbox = e.bbox(); // x1, y1, x2, y2
                    // But we need from bottom left coordinate
                    let top = page.height().value - bbox[1] * h_f + PADDING;
                    let bottom = page.height().value - bbox[3] * h_f - PADDING;
                    let left = bbox[0] * w_f - PADDING;
                    let right = bbox[2] * w_f + PADDING;
                    let text = page.text().unwrap().inside_rect(PdfRect::new_from_values(bottom, left, top, right)).replace("\t", " ");

                    match e.kind() {
                        DetectedElem:: Title => {
                            format!("## {}\n", text)
                        },
                        DetectedElem::Text | DetectedElem::List => text,
                        _ => unimplemented!()
                    }
                }).collect::<Vec<_>>().join("\n");

                Some(res)
            })
            .collect::<Vec<_>>();

        Some(())
    }).collect::<Vec<_>>();

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use anyhow::Result;

    use super::files_to_text;

    #[test]
    fn extract_from_pdf() -> Result<()> {
        
        let files = &[Path::new("../test-data/prehistory/archaeology.pdf").to_path_buf()];
        files_to_text(files)?;

        Ok(())
    }
}