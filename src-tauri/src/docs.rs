use std::{
    fs::read_to_string,
    path::{Path, PathBuf},
    sync::mpsc::Sender,
};

use anyhow::{anyhow, Result};
use pdfium_render::prelude::{PdfDocument, PdfRect, PdfRenderConfig, Pdfium};

use crate::{
    layout::{DetectedElem, Detectron2Model},
    store::FileKind,
};

// We'll pad the detected regions of interest by a little to increase chances of getting valid text.
// Often, the detected regions of interest would be too tihtly packed which might harm the text extraction
const PADDING: f32 = 1.;

pub enum ExtractorEvt {
    Page,
    Estimate(usize),
    Data(Result<Vec<Vec<(String, FileKind)>>>),
}

pub struct Extractor {
    txts: Vec<PathBuf>,
    pdfproc: Option<PdfProc>,
}

pub struct PdfProc {
    pdfs: Vec<PathBuf>,
    layout: Detectron2Model,
    pdfium: Pdfium,
    pdfium_cfg: PdfRenderConfig,
}

impl PdfProc {
    pub fn new(model_path: &Path, pdfs: Vec<PathBuf>) -> Result<Self> {
        // Initializing the layout detection model
        let layout = Detectron2Model::new(model_path)?;
        // Initialize `pdfium` linking it `dynamically`
        // the env `PDFIUM_DYNAMIC_LIB_PATH` is what we had defined in `src-tauri/.cargo/config.toml`
        let pdfium = Pdfium::new(Pdfium::bind_to_library(
            Pdfium::pdfium_platform_library_name_at_path(env!("PDFIUM_DYNAMIC_LIB_PATH")),
        )?);

        // Some config for `pdfium` rendering
        let cfg = PdfRenderConfig::new()
            .set_target_width(Detectron2Model::REQUIRED_WIDTH as i32)
            .set_maximum_height(Detectron2Model::REQUIRED_HEIGHT as i32);

        Ok(Self {
            pdfs,
            layout,
            pdfium,
            pdfium_cfg: cfg,
        })
    }

    /// Returns the total number of pages to be analyzed
    pub fn estimate(&self) -> usize {
        let count: u16 = self
            .pdfs
            .iter()
            .filter_map(|pth| {
                let pdf = self.pdfium.load_pdf_from_file(pth, None).ok()?;
                Some(pdf.pages().len())
            })
            .sum();

        count as usize
    }

    /// Extract text from `pdfs`
    pub fn extract(&self, send: Sender<ExtractorEvt>) -> Result<Vec<Vec<(String, FileKind)>>> {
        // for each `.pdf` file we are going to convert the pages to images
        let file_extracted = self
            .pdfs
            .iter()
            .filter_map(|file| {
                let pdf = self.pdfium.load_pdf_from_file(&file, None).ok()?;
                println!("Comes here!");
                self.process_pages(file, pdf, send.clone())
            })
            .collect::<Vec<_>>();

        Ok(file_extracted)
    }

    /// Processes each pages
    /// - reneders page with rendering config
    /// - runs layout detection
    /// - reads text from bounding boxes detected by the model
    pub fn process_pages(
        &self,
        file: &PathBuf,
        doc: PdfDocument<'_>,
        send: Sender<ExtractorEvt>,
    ) -> Option<Vec<(String, FileKind)>> {
        Some(
            doc.pages()
                .iter()
                .enumerate()
                .filter_map(|(idx, page)| {
                    // Convert the page to an image
                    let img = page.render_with_config(&self.pdfium_cfg).ok()?.as_image(); // Renders this page to an image::DynamicImage...

                    // Keep track of the factors by which the page and images of pages were resized to
                    // This is required to get accurate output from the predicted regions of interest
                    let w_f = page.width().value / img.width() as f32;
                    let h_f = page.height().value / img.height() as f32;
                    let pg_num = idx + 1;

                    // send the image for prediction
                    // and for each predicted `region of interest`
                    // fetch the text inside the bounding box
                    let text = self
                        .layout
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

                    if let Err(e) = send.send(ExtractorEvt::Page) {
                        eprintln!("Warn: error sending page event: {e:?}");
                    }

                    Some((text, FileKind::Pdf((file.to_owned(), pg_num))))
                })
                .collect::<Vec<_>>(),
        )
    }
}

impl Extractor {
    // Given a list of files, group them by file types supported - `.pdf` & `.txt` for now and call their respective extraction flow
    pub fn new(model_dir: &Path, files: &[PathBuf]) -> Result<Self> {
        println!("Sorting files for extraction..");

        let mut pdfs = vec![];
        let mut txts = vec![];

        files.iter().for_each(|f| {
            if let Some(ext) = f.extension() {
                if ext == "pdf" {
                    pdfs.push(f.to_owned());
                } else if ext == "txt" {
                    txts.push(f.to_owned());
                }
            }
        });

        let pdfproc = if !pdfs.is_empty() {
            PdfProc::new(model_dir, pdfs).ok()
        } else {
            None
        };

        Ok(Self { txts, pdfproc })
    }

    // Computes the estimated number of pages to be processed
    pub fn estimate(&self, send: Sender<ExtractorEvt>) {
        let mut count = self.txts.len();
        if let Some(p) = &self.pdfproc {
            count += p.estimate();
        }

        if let Err(e) = send.send(ExtractorEvt::Estimate(count)) {
            eprintln!("Extractor.estimate: error trying to send: {e:?}");
        }
    }

    pub fn extract(&self, send: Sender<ExtractorEvt>) -> Result<()> {
        let mut results = vec![];
        if let Some(pdf) = &self.pdfproc {
            results = [&results, &pdf.extract(send.clone())?[..]].concat();
        }

        if !self.txts.is_empty() {
            results = [&results, &self.extract_txt(send.clone())?[..]].concat();
        }

        send.send(ExtractorEvt::Data(Ok(results)))
            .map_err(|e| anyhow!(e))
    }

    // Reads text from each `.pdf` file and returns the text per page for the file
    pub fn extract_txt(&self, send: Sender<ExtractorEvt>) -> Result<Vec<Vec<(String, FileKind)>>> {
        let d = self
            .txts
            .iter()
            .filter_map(|f| {
                let txt = read_to_string(f.as_path()).ok()?;
                send.send(ExtractorEvt::Page).ok()?;
                Some(vec![(txt, FileKind::Text(f.to_path_buf()))])
            })
            .collect::<Vec<_>>();

        Ok(d)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use anyhow::Result;

    use super::PdfProc;

    #[test]
    fn extract_from_pdf() -> Result<()> {
        let files = vec![Path::new("../test-data/prehistory/archaeology.pdf").to_path_buf()];
        let pdfproc = PdfProc::new(Path::new("../models"), files)?;

        let res = {
            // Creating a dummy channel, we are not using this in this test
            let (s, _) = std::sync::mpsc::channel();
            pdfproc.extract(s)
        }?;

        assert!(!res.is_empty());
        // Let's print out the first page
        for (txt, _) in res[0].iter() {
            println!("{txt}");
        }

        Ok(())
    }
}
