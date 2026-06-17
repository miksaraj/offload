//! YOLOv8 player detection via ONNX Runtime.
//!
//! Real inference (backed by the `ort` crate) lands in Phase 2; this crate
//! currently exposes the stable types and API surface other crates depend on.

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

/// Loads a YOLOv8 ONNX model and runs person detection on frames.
pub struct Detector {
    #[allow(dead_code)]
    config: DetectorConfig,
}

impl Detector {
    pub fn new(_model_path: impl Into<String>, config: DetectorConfig) -> Result<Self> {
        Ok(Self { config })
    }

    pub fn detect(&self, _frame: &Frame) -> Result<Vec<Detection>> {
        Err(DetectorError::Inference("not yet implemented".into()))
    }
}
