//! Frame extraction and clip assembly via FFmpeg.
//!
//! Real decoding/encoding (backed by `ffmpeg-next`) lands in Phase 1; this
//! crate currently exposes the stable types and API surface other crates
//! depend on.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VideoIoError {
    #[error("input video not found: {0}")]
    NotFound(String),
    #[error("decode error: {0}")]
    Decode(String),
    #[error("encode error: {0}")]
    Encode(String),
}

pub type Result<T> = std::result::Result<T, VideoIoError>;

/// A single decoded video frame, normalised to the pipeline's working resolution.
#[derive(Debug, Clone)]
pub struct Frame {
    pub timestamp_ms: u64,
    pub frame_number: u64,
    /// Raw RGB pixels, row-major.
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// A clip to extract from the source video when compiling the final output.
#[derive(Debug, Clone)]
pub struct ClipSpec {
    pub start_ms: u64,
    pub end_ms: u64,
    pub label: Option<String>,
}

/// Streams decoded frames from a source video at a configured sampling rate.
pub struct FrameExtractor {
    #[allow(dead_code)]
    input_path: String,
}

impl FrameExtractor {
    pub fn new(input_path: impl Into<String>) -> Result<Self> {
        Ok(Self {
            input_path: input_path.into(),
        })
    }
}

/// Extracts, concatenates, and transcodes clip segments into a final output file.
pub struct ClipWriter {
    #[allow(dead_code)]
    output_path: String,
}

impl ClipWriter {
    pub fn new(output_path: impl Into<String>) -> Self {
        Self {
            output_path: output_path.into(),
        }
    }

    pub fn write(&self, _clips: Vec<ClipSpec>) -> Result<()> {
        Err(VideoIoError::Encode("not yet implemented".into()))
    }
}
