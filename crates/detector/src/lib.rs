//! YOLOv8 player detection via ONNX Runtime.
//!
//! Phase 2a: model loading, preprocessing, and raw inference. Postprocessing
//! (NMS, confidence/class filtering, decoding the output tensor into
//! [`Detection`]s) lands in Phase 2b.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use image::{imageops, ImageBuffer, Rgb, RgbImage};
use ndarray::{Array4, ArrayD};
use thiserror::Error;
use video_io::Frame;

#[derive(Debug, Error)]
pub enum DetectorError {
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("inference error: {0}")]
    Inference(String),
}

pub type Result<T> = std::result::Result<T, DetectorError>;

impl From<ort::Error> for DetectorError {
    fn from(err: ort::Error) -> Self {
        DetectorError::Inference(err.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// Pixel coordinates in the original frame space.
    pub bbox: Rect,
    pub confidence: f32,
    /// COCO class id; 0 = person.
    pub class_id: u32,
}

#[derive(Debug, Clone)]
pub struct DetectorConfig {
    pub confidence_threshold: f32,
    pub nms_iou_threshold: f32,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.4,
            nms_iou_threshold: 0.45,
        }
    }
}

/// YOLOv8's expected square input resolution.
pub const MODEL_INPUT_SIZE: u32 = 640;
const LETTERBOX_FILL: u8 = 114;
const INPUT_TENSOR_NAME: &str = "images";
const OUTPUT_TENSOR_NAME: &str = "output0";

/// Resizes a frame to [`MODEL_INPUT_SIZE`] square, preserving aspect ratio and
/// padding the remainder with grey (114,114,114) — the standard YOLO letterbox.
fn letterbox(frame: &Frame) -> RgbImage {
    let img: ImageBuffer<Rgb<u8>, &[u8]> =
        ImageBuffer::from_raw(frame.width, frame.height, frame.pixels.as_slice())
            .expect("Frame pixel buffer length must be width * height * 3");

    let scale = (MODEL_INPUT_SIZE as f32 / frame.width as f32)
        .min(MODEL_INPUT_SIZE as f32 / frame.height as f32);
    let resized_w = ((frame.width as f32 * scale).round() as u32).max(1);
    let resized_h = ((frame.height as f32 * scale).round() as u32).max(1);
    let resized = imageops::resize(&img, resized_w, resized_h, imageops::FilterType::Triangle);

    let mut canvas: RgbImage =
        ImageBuffer::from_pixel(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, Rgb([LETTERBOX_FILL; 3]));
    let pad_x = (MODEL_INPUT_SIZE - resized_w) / 2;
    let pad_y = (MODEL_INPUT_SIZE - resized_h) / 2;
    imageops::replace(&mut canvas, &resized, pad_x as i64, pad_y as i64);
    canvas
}

/// Normalises `[0, 255] -> [0.0, 1.0]` and converts `HWC -> NCHW`.
fn hwc_to_nchw(img: &RgbImage) -> Array4<f32> {
    Array4::from_shape_fn(
        (1, 3, MODEL_INPUT_SIZE as usize, MODEL_INPUT_SIZE as usize),
        |(_, c, y, x)| img.get_pixel(x as u32, y as u32)[c] as f32 / 255.0,
    )
}

/// Preprocesses a frame into the `[1, 3, 640, 640]` float32 tensor YOLOv8 expects.
pub fn preprocess(frame: &Frame) -> Array4<f32> {
    hwc_to_nchw(&letterbox(frame))
}

/// Dynamically loads the ONNX Runtime shared library and commits a process-wide
/// `ort` environment, exactly once. The path comes from `ORT_DYLIB_PATH` since
/// `ort`'s `load-dynamic` feature (used because the default `download-binaries`
/// feature fetches a prebuilt library from a CDN this project can't assume is
/// reachable) has no built-in env var fallback — see CLAUDE.md's Gotchas.
fn ensure_ort_initialized() -> Result<()> {
    static INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        let dylib_path = std::env::var("ORT_DYLIB_PATH").map_err(|_| {
            "ORT_DYLIB_PATH not set; point it at libonnxruntime.so (see .claude/skills/run-offload/SKILL.md)".to_string()
        })?;
        ort::init_from(dylib_path)
            .map_err(|e| e.to_string())?
            .commit();
        Ok(())
    })
    .clone()
    .map_err(DetectorError::Inference)
}

/// Loads a YOLOv8 ONNX model and runs person detection on frames.
pub struct Detector {
    session: ort::session::Session,
    #[allow(dead_code)]
    config: DetectorConfig,
}

impl Detector {
    /// Loads the model at `model_path` and runs a warm-up inference so the
    /// first real `detect`/`run_inference` call isn't paying for lazy
    /// initialisation inside the timed pipeline.
    pub fn new(model_path: impl AsRef<Path>, config: DetectorConfig) -> Result<Self> {
        let model_path: PathBuf = model_path.as_ref().to_path_buf();
        if !model_path.exists() {
            return Err(DetectorError::ModelNotFound(
                model_path.display().to_string(),
            ));
        }

        ensure_ort_initialized()?;

        let mut session = ort::session::Session::builder()?.commit_from_file(&model_path)?;

        let warmup_input =
            Array4::<f32>::zeros((1, 3, MODEL_INPUT_SIZE as usize, MODEL_INPUT_SIZE as usize));
        let warmup_outputs = session.run(ort::inputs![
            INPUT_TENSOR_NAME => ort::value::TensorRef::from_array_view(&warmup_input)?
        ])?;
        let warmup_shape = warmup_outputs[OUTPUT_TENSOR_NAME]
            .try_extract_array::<f32>()?
            .shape()
            .to_vec();
        tracing::info!(
            model = %model_path.display(),
            warmup_output_shape = ?warmup_shape,
            "detector model loaded and warmed up"
        );
        drop(warmup_outputs);

        Ok(Self { session, config })
    }

    /// Preprocesses `frame`, runs raw inference, and returns the model's
    /// undecoded output tensor. Logs the tensor's shape.
    pub fn run_inference(&mut self, frame: &Frame) -> Result<ArrayD<f32>> {
        let input = preprocess(frame);
        let outputs = self.session.run(ort::inputs![
            INPUT_TENSOR_NAME => ort::value::TensorRef::from_array_view(&input)?
        ])?;
        let output = outputs[OUTPUT_TENSOR_NAME].try_extract_array::<f32>()?;
        tracing::info!(shape = ?output.shape(), "raw detector output tensor");
        Ok(output.to_owned())
    }

    pub fn detect(&mut self, frame: &Frame) -> Result<Vec<Detection>> {
        self.run_inference(frame)?;
        Err(DetectorError::Inference(
            "postprocessing not yet implemented (Phase 2b)".into(),
        ))
    }
}
