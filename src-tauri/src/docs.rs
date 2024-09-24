use std::{
    fs::read_to_string,
    path::{Path, PathBuf},
};

use anyhow::Result;
use pdfium_render::prelude::{PdfRect, PdfRenderConfig, Pdfium};

use crate::{
    layout::{DetectedElem, Detectron2Model},
    store::FileKind,
};

// We'll pad the detected regions of interest by a little to increase chances of getting valid text.
// Often, the detected regions of interest would be too tihtly packed which might harm the text extraction
const PADDING: f32 = 1.;

// Given a list of files, group them by file types supported - `.pdf` & `.txt` for now and call their respective extraction flow
pub fn files_to_text(models_dir: &Path, files: &[PathBuf]) -> Result<Vec<Vec<(String, FileKind)>>> {
    println!("Sorting files for extraction..");
    let mut pdfs = vec![];
    let mut txts = vec![];

    files.iter().for_each(|f| {
        if let Some(ext) = f.extension() {
            if ext == "pdf" {
                pdfs.push(f);
            } else if ext == "txt" {
                txts.push(f);
            }
        }
    });

    println!("Pdfs: {} Text: {}", pdfs.len(), txts.len());

    let mut results = vec![];
    if !pdfs.is_empty() {
        results = [&results, &pdf_to_text(models_dir, &pdfs[..])?[..]].concat();
    }
    if !txts.is_empty() {
        results = [&results, &txt_to_text(&txts[..])?[..]].concat();
    }

    Ok(results)
}

// Reads text from each `.pdf` file and returns the text per page for the file
pub fn pdf_to_text(models_dir: &Path, files: &[&PathBuf]) -> Result<Vec<Vec<(String, FileKind)>>> {
    let n_doc = files.len();
    println!("Begin extraction of {n_doc} pdf files");
    
    // Initializing the layout detection model
    let layout_model = Detectron2Model::new(models_dir)?;

    // Initialize `pdfium` linking it `dynamically`
    // the env `PDFIUM_DYNAMIC_LIB_PATH` is what we had defined in `src-tauri/.cargo/config.toml`
    let pdfium = Pdfium::new(Pdfium::bind_to_library(
        Pdfium::pdfium_platform_library_name_at_path(env!("PDFIUM_DYNAMIC_LIB_PATH")),
    )?);

    // Some config for `pdfium` rendering
    let cfg = PdfRenderConfig::new()
        .set_target_width(Detectron2Model::REQUIRED_WIDTH as i32)
        .set_maximum_height(Detectron2Model::REQUIRED_HEIGHT as i32);

    // Now, for each `.pdf` file we are going to convert the pages to images
    let file_encoded = files
        .iter()
        .enumerate()
        .filter_map(|(i, &file)| {
            // Load the `.pdf` file
            let pdf = pdfium.load_pdf_from_file(file, None).ok()?;
            let n_pages = pdf.pages().len();
            println!("Doc[{i}]: got {n_pages} pages");
            // iterate over each page
            // Text and file info for each page
            let page_data = pdf
                .pages()
                .iter()
                // keep track of page index to help us give better results
                .enumerate()
                .filter_map(|(idx, page)| {
                    println!("Doc[{i}/{n_doc}]: Page[{idx}/{n_pages}]");
                    // Convert the page to an image
                    let img = page.render_with_config(&cfg).ok()?.as_image(); // Renders this page to an image::DynamicImage...

                    // Keep track of the factors by which the page and images of pages were resized to
                    // This is required to get accurate output from the predicted regions of interest
                    let w_f = page.width().value / img.width() as f32;
                    let h_f = page.height().value / img.height() as f32;

                    // send the image for prediction
                    // and for each predicted `region of interest`
                    // fetch the text inside the bounding box
                    let text = layout_model
                        .predict(&img)
                        .ok()?
                        .iter()
                        .filter_map(|e| {
                            // The bounding box for the region of interest
                            let bbox = e.bbox(); // x1, y1, x2, y2

                            // The bounding boxes for the predicted regions follow a `left-top` co-ordinate system
                            // But `pdfium` uses a bottom-left coordinate system, let's convert it
                            // We'll also factor in the original page size here
                            let top = page.height().value - bbox[1] * h_f + PADDING;
                            let bottom = page.height().value - bbox[3] * h_f - PADDING;
                            let left = bbox[0] * w_f - PADDING;
                            let right = bbox[2] * w_f + PADDING;

                            // Now, we have the `pdfium` compatible bounding boxes
                            // Let's fetch the text
                            let text = page
                                .text()
                                .ok()?
                                .inside_rect(PdfRect::new_from_values(bottom, left, top, right))
                                .replace("\t", " ")
                                .replace("\r\n", "\n");

                            Some(match e.kind() {
                                // We are using `MarkDownSplitter` for our text splitting task
                                // Here we are adding `##` to mark the generated text as title
                                DetectedElem::Title => {
                                    format!("## {}\n", text.replace("\n", "; "))
                                }
                                // Rest of the text remains as is
                                DetectedElem::Text | DetectedElem::List => text,
                                _ => unimplemented!(),
                            })
                        })
                        .collect::<Vec<_>>()
                        .join("\n");

                    Some((text, FileKind::Pdf((file.to_owned(), idx))))
                })
                .collect::<Vec<_>>();

            Some(page_data)
        })
        .collect::<Vec<_>>();

    Ok(file_encoded)
}

// Reads text from each `.pdf` file and returns the text per page for the file
pub fn txt_to_text(files: &[&PathBuf]) -> Result<Vec<Vec<(String, FileKind)>>> {
    let d = files
        .iter()
        .filter_map(|f| {
            let txt = read_to_string(f.as_path()).ok()?;

            Some(vec![(txt, FileKind::Text(f.to_path_buf()))])
        })
        .collect::<Vec<_>>();

    Ok(d)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use anyhow::Result;

    use super::files_to_text;

    #[test]
    fn extract_from_pdf() -> Result<()> {
        let files = &[Path::new("../test-data/prehistory/origins-of-agriculture.pdf").to_path_buf()];
        let results = files_to_text(Path::new("../models"), files)?;

        assert!(!results.is_empty());

        // Let's print out the first page
        for (txt, _) in results[0].iter() {
            println!("{txt}");
        }
        Ok(())
    }
}
