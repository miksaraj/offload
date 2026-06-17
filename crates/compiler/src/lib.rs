//! Final highlight reel assembly via FFmpeg.
//!
//! Real clip extraction, transitions, and encoding (backed by `ffmpeg-next`)
//! land in Phase 7; this crate currently exposes the stable types and API
//! surface other crates depend on.

use std::path::Path;

use classifier::Moment;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompilerError {
    #[error("encode error: {0}")]
    Encode(String),
}

pub type Result<T> = std::result::Result<T, CompilerError>;

#[derive(Debug, Clone)]
pub struct CompilerConfig {
    pub padding_ms: u64,
    pub transition_duration_ms: u64,
    pub label_display_ms: u64,
    pub output_crf: u8,
    pub output_preset: String,
    pub music_volume: f32,
    pub music_only: bool,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            padding_ms: 3000,
            transition_duration_ms: 500,
            label_display_ms: 3000,
            output_crf: 23,
            output_preset: "medium".into(),
            music_volume: 0.25,
            music_only: false,
        }
    }
}

/// Assembles the final compiled video from source footage and scored moments.
pub struct Compiler {
    #[allow(dead_code)]
    config: CompilerConfig,
}

impl Compiler {
    pub fn new(config: CompilerConfig) -> Self {
        Self { config }
    }

    pub fn compile(&self, _source: &Path, _moments: Vec<Moment>) -> Result<()> {
        Err(CompilerError::Encode("not yet implemented".into()))
    }
}
