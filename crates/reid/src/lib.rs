//! Player re-identification via OSNet, with an interactive identification phase.
//!
//! Phase 4a: [`ReIdModel`] — OSNet-x1.0 appearance embeddings for player
//! crops (see [`embedding`]).
//! Phase 4b: interactive identification — candidate frames, numbered
//! selection image, viewer, prompts (see [`identify`]).
//! Phases 4c/4d: seeding from the pick, confirmation, and offline
//! identity chaining across gaps (see [`chain`]).

pub mod chain;
pub mod embedding;
pub mod identify;

pub use chain::{SubjectTimeline, TrackLog};
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

/// `[reid]` section of `offload.toml` (deserialised directly, like
/// `video_io::SourceConfig`).
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct ReIdConfig {
    pub candidate_frame_count: u32,
    pub candidate_window_secs: u32,
    /// Seconds into the video where the identification window starts
    /// (for a subject who appears only partway through).
    pub candidate_window_start_secs: Option<u32>,
    pub confirmation_scan_secs: u32,
    pub max_identification_attempts: u32,
    /// Track appearance signatures: embed each tracked player at most
    /// this often, up to `signature_max_samples`, if its box is at least
    /// `min_crop_height_px` tall.
    pub signature_interval_ms: u64,
    pub signature_max_samples: usize,
    pub min_crop_height_px: f32,
    /// Identity chaining (see [`chain`] and SPEC Stage 4).
    pub reentry_threshold: f32,
    pub reentry_margin: f32,
    pub reentry_window_secs: u32,
    pub reentry_min_samples: usize,
    pub gallery_max_size: usize,
}

impl Default for ReIdConfig {
    fn default() -> Self {
        let c = chain::ChainConfig::default();
        Self {
            candidate_frame_count: 3,
            candidate_window_secs: 300,
            candidate_window_start_secs: None,
            confirmation_scan_secs: 60,
            max_identification_attempts: 5,
            signature_interval_ms: 1000,
            signature_max_samples: 8,
            min_crop_height_px: 40.0,
            reentry_threshold: c.reentry_threshold,
            reentry_margin: c.reentry_margin,
            reentry_window_secs: (c.reentry_window_ms / 1000) as u32,
            reentry_min_samples: c.reentry_min_samples,
            gallery_max_size: c.gallery_max_size,
        }
    }
}

impl ReIdConfig {
    pub fn chain(&self) -> chain::ChainConfig {
        chain::ChainConfig {
            reentry_threshold: self.reentry_threshold,
            reentry_margin: self.reentry_margin,
            reentry_window_ms: u64::from(self.reentry_window_secs) * 1000,
            reentry_min_samples: self.reentry_min_samples,
            gallery_max_size: self.gallery_max_size,
        }
    }

    pub fn sampler(&self) -> chain::SignatureSampler {
        chain::SignatureSampler::new(
            self.signature_interval_ms,
            self.signature_max_samples,
            self.min_crop_height_px,
        )
    }
}
