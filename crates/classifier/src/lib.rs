//! Semantic moment scoring via the local Ollama vision API.
//!
//! The async HTTP client and moment-grouping logic land in Phase 5; this
//! crate currently exposes the stable types and API surface other crates
//! depend on.

use serde::{Deserialize, Serialize};
use thiserror::Error;
use video_io::Frame;

#[derive(Debug, Error)]
pub enum ClassifierError {
    #[error("ollama request failed: {0}")]
    Request(String),
}

pub type Result<T> = std::result::Result<T, ClassifierError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MomentKind {
    Highlight,
    Lowlight,
}

#[derive(Debug, Clone)]
pub struct MomentScore {
    pub timestamp_ms: u64,
    pub score: f32,
    pub kind: MomentKind,
    pub event_label: String,
}

#[derive(Debug, Clone)]
pub struct Moment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub kind: MomentKind,
    pub peak_score: f32,
    pub label: String,
}

#[derive(Debug, Clone)]
pub struct ClassifierConfig {
    pub endpoint: String,
    pub model: String,
    pub timeout_secs: u64,
    pub sample_interval_ms: u64,
    pub min_bbox_area_fraction: f32,
    pub highlight_threshold: f32,
    pub merge_gap_ms: u64,
    pub min_moment_duration_ms: u64,
    pub jpeg_quality: u8,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:11434".into(),
            model: "llava".into(),
            timeout_secs: 30,
            sample_interval_ms: 2000,
            min_bbox_area_fraction: 0.002,
            highlight_threshold: 6.5,
            merge_gap_ms: 4000,
            min_moment_duration_ms: 1500,
            jpeg_quality: 80,
        }
    }
}

/// Scores subject-present frames for highlight/lowlight significance via Ollama.
pub struct OllamaClassifier {
    #[allow(dead_code)]
    config: ClassifierConfig,
}

impl OllamaClassifier {
    pub fn new(config: ClassifierConfig) -> Self {
        Self { config }
    }

    pub async fn score(&self, _frame: &Frame) -> Result<MomentScore> {
        Err(ClassifierError::Request("not yet implemented".into()))
    }
}
