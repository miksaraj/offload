//! Frame extraction and clip assembly via FFmpeg.
//!
//! Frame extraction (Phase 1a) is implemented on top of `ffmpeg-next`.
//! Clip assembly (`ClipWriter`, Phase 1b) is still a stub.

use std::path::Path;

use ffmpeg_next as ffmpeg;
use ffmpeg_next::Rescale;
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

impl From<ffmpeg::Error> for VideoIoError {
    fn from(err: ffmpeg::Error) -> Self {
        VideoIoError::Decode(err.to_string())
    }
}

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

/// Streams decoded frames from a source video, one [`Frame`] per source frame.
///
/// Yields every decoded frame in the original resolution; sampling (e.g. "every
/// Nth frame" or a target detection fps) is the caller's responsibility.
pub struct FrameExtractor {
    input_ctx: ffmpeg::format::context::Input,
    decoder: ffmpeg::decoder::Video,
    scaler: ffmpeg::software::scaling::Context,
    video_stream_index: usize,
    time_base: ffmpeg::Rational,
    frame_number: u64,
    eof_sent: bool,
}

impl FrameExtractor {
    pub fn new(input_path: impl AsRef<Path>) -> Result<Self> {
        let input_path = input_path.as_ref();
        if !input_path.exists() {
            return Err(VideoIoError::NotFound(input_path.display().to_string()));
        }

        ffmpeg::init().map_err(|e| VideoIoError::Decode(e.to_string()))?;

        let input_ctx = ffmpeg::format::input(input_path)?;

        let (video_stream_index, time_base, parameters) = {
            let stream = input_ctx
                .streams()
                .best(ffmpeg::media::Type::Video)
                .ok_or_else(|| VideoIoError::Decode("no video stream found".into()))?;
            (stream.index(), stream.time_base(), stream.parameters())
        };

        let decoder = ffmpeg::codec::context::Context::from_parameters(parameters)?
            .decoder()
            .video()?;

        let scaler = ffmpeg::software::scaling::Context::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            ffmpeg::format::Pixel::RGB24,
            decoder.width(),
            decoder.height(),
            ffmpeg::software::scaling::Flags::BILINEAR,
        )?;

        Ok(Self {
            input_ctx,
            decoder,
            scaler,
            video_stream_index,
            time_base,
            frame_number: 0,
            eof_sent: false,
        })
    }

    fn build_frame(&mut self, decoded: &ffmpeg::frame::Video) -> Result<Frame> {
        let mut rgb = ffmpeg::frame::Video::empty();
        self.scaler.run(decoded, &mut rgb)?;

        let width = rgb.width();
        let height = rgb.height();
        let stride = rgb.stride(0);
        let row_bytes = width as usize * 3;
        let data = rgb.data(0);

        let mut pixels = Vec::with_capacity(row_bytes * height as usize);
        for row in 0..height as usize {
            let start = row * stride;
            pixels.extend_from_slice(&data[start..start + row_bytes]);
        }

        let pts = decoded.timestamp().unwrap_or(0).max(0);
        let timestamp_ms = pts
            .rescale(self.time_base, ffmpeg::Rational(1, 1000))
            .max(0) as u64;

        let frame_number = self.frame_number;
        self.frame_number += 1;

        Ok(Frame {
            timestamp_ms,
            frame_number,
            pixels,
            width,
            height,
        })
    }

    /// Pulls the next packet for our video stream and feeds it to the decoder.
    /// Returns `false` once the input is exhausted (and the decoder has been
    /// sent EOF so any buffered frames can still be drained).
    fn advance(&mut self) -> bool {
        loop {
            match self.input_ctx.packets().next() {
                Some((stream, packet)) => {
                    if stream.index() != self.video_stream_index {
                        continue;
                    }
                    if let Err(e) = self.decoder.send_packet(&packet) {
                        tracing::warn!(error = %e, "failed to send packet to decoder");
                        continue;
                    }
                    return true;
                }
                None => {
                    if !self.eof_sent {
                        let _ = self.decoder.send_eof();
                        self.eof_sent = true;
                    }
                    return false;
                }
            }
        }
    }
}

impl Iterator for FrameExtractor {
    type Item = Frame;

    fn next(&mut self) -> Option<Frame> {
        loop {
            let mut decoded = ffmpeg::frame::Video::empty();
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => match self.build_frame(&decoded) {
                    Ok(frame) => return Some(frame),
                    Err(e) => {
                        tracing::warn!(error = %e, "failed to convert decoded frame, stopping");
                        return None;
                    }
                },
                // Decoder needs more input; feed it the next packet (or, once
                // the input is exhausted, EOF) and try receiving again.
                Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::error::EAGAIN => {
                    self.advance();
                }
                Err(ffmpeg::Error::Eof) => return None,
                Err(e) => {
                    tracing::warn!(error = %e, "decode error, stopping frame extraction");
                    return None;
                }
            }
        }
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
