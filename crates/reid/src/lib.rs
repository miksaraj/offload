//! Player re-identification via OSNet, with an interactive identification phase.
//!
//! Phase 4a: [`ReIdModel`] — OSNet-x1.0 appearance embeddings for player
//! crops (see [`embedding`]).
//! Phase 4b: interactive identification — candidate frames, numbered
//! selection image, viewer, prompts (see [`identify`]).
//! Confirmation and matching (4c) and temporal smoothing (4d) are still
//! stubs below.

pub mod embedding;
pub mod identify;

pub use embedding::{crop, Embedding, ReIdModel};
pub use identify::{Candidate, InteractionOptions, Selection};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReIdError {
    #[error("model not found: {0} (run models/download.sh to fetch it)")]
    ModelNotFound(String),
    #[error("reid inference error: {0}")]
    Inference(String),
    #[error("identification error: {0}")]
    Identification(String),
    #[error(
        "no players detected in the candidate frames; check the video quality, the \
         identification window (reid.candidate_window_secs / candidate_window_start_secs), \
         and detector.confidence_threshold"
    )]
    NoPlayers,
    #[error("identification aborted")]
    Aborted,
}

pub type Result<T> = std::result::Result<T, ReIdError>;

#[derive(Debug, Clone)]
pub struct ReIdConfig {
    pub candidate_frame_count: u32,
    pub candidate_window_secs: u32,
    /// Seconds into the video where the identification window starts
    /// (for a subject who appears only partway through).
    pub candidate_window_start_secs: Option<u32>,
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
            candidate_window_start_secs: None,
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
