//! Player re-identification via OSNet, with an interactive identification phase.
//!
//! Phase 4a: [`ReIdModel`] — OSNet-x1.0 appearance embeddings for player
//! crops (see [`embedding`]). Interactive identification (4b), confirmation
//! and matching (4c) and temporal smoothing (4d) are still stubs below.

pub mod embedding;

pub use embedding::{crop, Embedding, ReIdModel};
use thiserror::Error;
use video_io::Frame;

#[derive(Debug, Error)]
pub enum ReIdError {
    #[error("model not found: {0} (run models/download.sh to fetch it)")]
    ModelNotFound(String),
    #[error("reid inference error: {0}")]
    Inference(String),
    #[error("identification error: {0}")]
    Identification(String),
}

pub type Result<T> = std::result::Result<T, ReIdError>;

#[derive(Debug, Clone)]
pub struct ReIdConfig {
    pub candidate_frame_count: u32,
    pub candidate_window_secs: u32,
    pub confirmation_scan_secs: u32,
    pub max_identification_attempts: u32,
    pub lock_threshold: f32,
    pub vote_window: u32,
    pub reentry_threshold: f32,
    pub reentry_vote_window: u32,
    pub max_identification_frame: u64,
}

impl Default for ReIdConfig {
    fn default() -> Self {
        Self {
            candidate_frame_count: 3,
            candidate_window_secs: 300,
            confirmation_scan_secs: 60,
            max_identification_attempts: 5,
            lock_threshold: 0.72,
            vote_window: 10,
            reentry_threshold: 0.68,
            reentry_vote_window: 5,
            max_identification_frame: 600,
        }
    }
}

/// Drives the interactive subject identification phase (Stage 4A).
pub struct Identifier {
    #[allow(dead_code)]
    config: ReIdConfig,
}

impl Identifier {
    pub fn new(config: ReIdConfig) -> Self {
        Self { config }
    }

    pub fn select_candidate_frame(&self, _video_path: &str) -> Result<Frame> {
        Err(ReIdError::Identification("not yet implemented".into()))
    }
}

/// Matches tracks against a locked reference embedding across the full video.
pub struct ReIdMatcher {
    #[allow(dead_code)]
    reference_embedding: Embedding,
}

impl ReIdMatcher {
    pub fn new(reference_embedding: Embedding) -> Self {
        Self {
            reference_embedding,
        }
    }

    pub fn identify(&self, _candidates: Vec<(u64, Embedding)>) -> Option<u64> {
        None
    }
}
