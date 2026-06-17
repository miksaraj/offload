//! ByteTrack multi-object tracking, implemented in pure Rust.
//!
//! The Kalman filter and Hungarian assignment land in Phase 3; this crate
//! currently exposes the stable types and API surface other crates depend on.

use detector::{Detection, Rect};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TrackerError {
    #[error("tracking error: {0}")]
    Update(String),
}

pub type Result<T> = std::result::Result<T, TrackerError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    Tracked,
    Lost,
    Removed,
}

#[derive(Debug, Clone)]
pub struct Track {
    pub track_id: u64,
    pub bbox: Rect,
    pub state: TrackState,
    pub age: u32,
    pub frames_since_update: u32,
}

#[derive(Debug, Clone)]
pub struct TrackerConfig {
    pub high_confidence_threshold: f32,
    pub low_confidence_threshold: f32,
    pub max_lost_frames: u32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            high_confidence_threshold: 0.6,
            low_confidence_threshold: 0.1,
            max_lost_frames: 30,
        }
    }
}

/// Runs ByteTrack association across frames, maintaining stable track IDs.
pub struct Tracker {
    #[allow(dead_code)]
    config: TrackerConfig,
    #[allow(dead_code)]
    next_track_id: u64,
}

impl Tracker {
    pub fn new(config: TrackerConfig) -> Self {
        Self {
            config,
            next_track_id: 0,
        }
    }

    pub fn update(&mut self, _detections: Vec<Detection>) -> Result<Vec<Track>> {
        Err(TrackerError::Update("not yet implemented".into()))
    }
}
