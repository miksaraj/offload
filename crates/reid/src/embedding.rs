//! OSNet appearance embeddings (Phase 4a).
//!
//! A player crop is stretched to 128×256 (W×H), the standard person-ReID
//! input, converted to RGB `[0, 1]`, normalised with ImageNet mean/std
//! (the model's first layer is a plain `Conv`; normalisation is not baked
//! into the export), and run through OSNet-x1.0 to give a 512-d feature.
//! Features are L2-normalised so cosine similarity is a dot product.

use std::path::{Path, PathBuf};

use detector::Rect;
use image::{imageops, Rgb, RgbImage};
use ndarray::{Array4, Axis};
use video_io::Frame;

use crate::{ReIdError, Result};

/// OSNet input width/height (the 256×128 H×W person-ReID standard).
pub const INPUT_WIDTH: u32 = 128;
pub const INPUT_HEIGHT: u32 = 256;
/// Feature dimension.
pub const EMBEDDING_DIM: usize = 512;
const INPUT_TENSOR_NAME: &str = "base_images";
const OUTPUT_TENSOR_NAME: &str = "features";
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// A 512-dim, L2-normalised appearance feature vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding(pub Vec<f32>);

impl Embedding {
    /// Wraps a raw feature vector, L2-normalising it. An all-zero vector
    /// stays all-zero (cosine with it is 0).
    pub fn from_raw(mut v: Vec<f32>) -> Self {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        Self(v)
    }

    /// Cosine similarity in `[-1, 1]` (OSNet's features are post-ReLU, so
    /// in practice `[0, 1]`).
    pub fn cosine(&self, other: &Embedding) -> f32 {
        self.0.iter().zip(&other.0).map(|(a, b)| a * b).sum()
    }

    /// L2-normalised mean of several embeddings (e.g. a track's appearance
    /// over a vote window). `None` for an empty slice.
    pub fn mean(embeddings: &[Embedding]) -> Option<Embedding> {
        let first = embeddings.first()?;
        let mut acc = vec![0.0f32; first.0.len()];
        for e in embeddings {
            for (a, x) in acc.iter_mut().zip(&e.0) {
                *a += x;
            }
        }
        Some(Embedding::from_raw(acc))
    }
}

/// Crops `bbox` out of `frame` (clamped to the frame; at least 1×1 px).
pub fn crop(frame: &Frame, bbox: &Rect) -> RgbImage {
    assert_eq!(
        frame.pixels.len(),
        (frame.width * frame.height * 3) as usize,
        "Frame pixel buffer length must be width * height * 3"
    );
    let x0 = (bbox.x.max(0.0) as u32).min(frame.width.saturating_sub(1));
    let y0 = (bbox.y.max(0.0) as u32).min(frame.height.saturating_sub(1));
    let x1 = ((bbox.x + bbox.width).ceil().max(0.0) as u32).min(frame.width);
    let y1 = ((bbox.y + bbox.height).ceil().max(0.0) as u32).min(frame.height);
    // Copied directly rather than via `imageops::crop_imm` over a borrowed
    // `ImageBuffer<_, &[u8]>`, whose `to_image()` result can't outlive the
    // borrow (E0521).
    let (w, h) = ((x1 - x0).max(1), (y1 - y0).max(1));
    let stride = frame.width as usize * 3;
    RgbImage::from_fn(w, h, |x, y| {
        let px = (x0 + x).min(frame.width - 1) as usize;
        let py = (y0 + y).min(frame.height - 1) as usize;
        let i = py * stride + px * 3;
        Rgb([frame.pixels[i], frame.pixels[i + 1], frame.pixels[i + 2]])
    })
}

/// Writes one crop into slot `n` of a `[N, 3, 256, 128]` batch tensor.
fn fill_input(batch: &mut Array4<f32>, n: usize, crop: &RgbImage) {
    let resized = imageops::resize(
        crop,
        INPUT_WIDTH,
        INPUT_HEIGHT,
        imageops::FilterType::Triangle,
    );
    for (x, y, px) in resized.enumerate_pixels() {
        for c in 0..3 {
            batch[[n, c, y as usize, x as usize]] =
                (f32::from(px[c]) / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
        }
    }
}

/// Preprocesses crops into the `[N, 3, 256, 128]` tensor OSNet expects.
pub fn preprocess(crops: &[RgbImage]) -> Array4<f32> {
    let mut batch =
        Array4::<f32>::zeros((crops.len(), 3, INPUT_HEIGHT as usize, INPUT_WIDTH as usize));
    for (n, c) in crops.iter().enumerate() {
        fill_input(&mut batch, n, c);
    }
    batch
}

/// Loads the OSNet ONNX model and produces appearance embeddings for crops.
pub struct ReIdModel {
    session: ort::session::Session,
    model_path: PathBuf,
}

impl ReIdModel {
    /// Loads the model and runs a warm-up inference.
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();
        if !model_path.exists() {
            return Err(ReIdError::ModelNotFound(model_path.display().to_string()));
        }
        detector::ensure_ort_initialized().map_err(|e| ReIdError::Inference(e.to_string()))?;
        let session = ort::session::Session::builder()
            .and_then(|mut b| b.commit_from_file(&model_path))
            .map_err(|e| ReIdError::Inference(e.to_string()))?;
        let mut model = Self {
            session,
            model_path,
        };
        let warmup = model.embed_crops(&[RgbImage::new(INPUT_WIDTH, INPUT_HEIGHT)])?;
        tracing::info!(
            model = %model.model_path.display(),
            dim = warmup.first().map(|e| e.0.len()),
            "reid model loaded and warmed up"
        );
        Ok(model)
    }

    /// Embeds each crop (one batched inference). Empty in → empty out.
    pub fn embed_crops(&mut self, crops: &[RgbImage]) -> Result<Vec<Embedding>> {
        if crops.is_empty() {
            return Ok(Vec::new());
        }
        let input = preprocess(crops);
        let err = |e: ort::Error| ReIdError::Inference(e.to_string());
        let outputs = self
            .session
            .run(ort::inputs![
                INPUT_TENSOR_NAME => ort::value::TensorRef::from_array_view(&input).map_err(err)?
            ])
            .map_err(err)?;
        let features = outputs[OUTPUT_TENSOR_NAME]
            .try_extract_array::<f32>()
            .map_err(err)?;
        let shape = features.shape().to_vec();
        if shape != [crops.len(), EMBEDDING_DIM] {
            return Err(ReIdError::Inference(format!(
                "unexpected reid output shape {shape:?}; expected [{}, {EMBEDDING_DIM}]",
                crops.len()
            )));
        }
        Ok(features
            .axis_iter(Axis(0))
            .map(|row| Embedding::from_raw(row.iter().copied().collect()))
            .collect())
    }

    /// Crops each box out of `frame` and embeds them in one batch.
    pub fn embed_boxes(&mut self, frame: &Frame, boxes: &[Rect]) -> Result<Vec<Embedding>> {
        let crops: Vec<RgbImage> = boxes.iter().map(|b| crop(frame, b)).collect();
        self.embed_crops(&crops)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddings_are_unit_length_and_cosine_is_a_dot_product() {
        let a = Embedding::from_raw(vec![3.0, 4.0]);
        assert!((a.0[0] - 0.6).abs() < 1e-6 && (a.0[1] - 0.8).abs() < 1e-6);
        assert!((a.cosine(&a) - 1.0).abs() < 1e-6);
        let b = Embedding::from_raw(vec![4.0, -3.0]);
        assert!(a.cosine(&b).abs() < 1e-6);
        assert_eq!(Embedding::from_raw(vec![0.0, 0.0]).cosine(&a), 0.0);
    }

    #[test]
    fn mean_is_normalised() {
        let m = Embedding::mean(&[
            Embedding::from_raw(vec![1.0, 0.0]),
            Embedding::from_raw(vec![0.0, 1.0]),
        ])
        .unwrap();
        assert!((m.0[0] - m.0[1]).abs() < 1e-6);
        assert!((m.cosine(&m) - 1.0).abs() < 1e-6);
        assert!(Embedding::mean(&[]).is_none());
    }

    #[test]
    fn crop_clamps_to_the_frame() {
        let frame = Frame {
            timestamp_ms: 0,
            frame_number: 0,
            pixels: vec![7; 100 * 50 * 3],
            width: 100,
            height: 50,
        };
        let c = crop(
            &frame,
            &Rect {
                x: -10.0,
                y: 40.0,
                width: 30.0,
                height: 30.0,
            },
        );
        assert_eq!(c.dimensions(), (20, 10));
        let c = crop(
            &frame,
            &Rect {
                x: 200.0,
                y: 200.0,
                width: 5.0,
                height: 5.0,
            },
        );
        assert_eq!(c.dimensions(), (1, 1));
    }

    #[test]
    fn preprocess_normalises_with_imagenet_stats_in_nchw() {
        let white = RgbImage::from_pixel(10, 20, Rgb([255, 255, 255]));
        let t = preprocess(&[white]);
        assert_eq!(t.shape(), &[1, 3, 256, 128]);
        for c in 0..3 {
            let expected = (1.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            assert!((t[[0, c, 100, 50]] - expected).abs() < 1e-5);
        }
    }
}
