//! Frame extraction and clip assembly via FFmpeg.
//!
//! Frame extraction (Phase 1a) and clip assembly (Phase 1b) are both
//! implemented on top of `ffmpeg-next`.

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

/// Configuration for [`ClipWriter`]'s output encode.
#[derive(Debug, Clone, Default)]
pub struct ClipWriterConfig {
    /// Extra time added before each clip's start and after its end, clamped
    /// to the source video's bounds.
    pub padding_ms: u64,
    /// Output width; defaults to the source video's width if unset.
    pub output_width: Option<u32>,
    /// Output height; defaults to the source video's height if unset.
    pub output_height: Option<u32>,
    /// Output bitrate in kbps. If unset, encodes at a fixed CRF instead.
    pub bitrate_kbps: Option<u32>,
}

/// Extracts, concatenates, and transcodes clip segments into a final output file.
pub struct ClipWriter {
    output_path: String,
    config: ClipWriterConfig,
}

impl ClipWriter {
    pub fn new(output_path: impl Into<String>, config: ClipWriterConfig) -> Self {
        Self {
            output_path: output_path.into(),
            config,
        }
    }

    /// Extracts `clips` from `source_path` (applying configured padding) and
    /// concatenates them, in timestamp order, into a single H.264 output file.
    pub fn write(&self, source_path: impl AsRef<Path>, clips: &[ClipSpec]) -> Result<()> {
        let source_path = source_path.as_ref();
        if !source_path.exists() {
            return Err(VideoIoError::NotFound(source_path.display().to_string()));
        }
        if clips.is_empty() {
            return Err(VideoIoError::Encode("no clips specified".into()));
        }

        ffmpeg::init().map_err(|e| VideoIoError::Decode(e.to_string()))?;

        let mut ictx = ffmpeg::format::input(source_path)?;
        let source_duration_ms = (ictx.duration().max(0) as u64 / 1000).max(1);

        let (video_stream_index, input_time_base, frame_rate, parameters) = {
            let stream = ictx
                .streams()
                .best(ffmpeg::media::Type::Video)
                .ok_or_else(|| VideoIoError::Decode("no video stream found".into()))?;
            (
                stream.index(),
                stream.time_base(),
                stream.rate(),
                stream.parameters(),
            )
        };

        let mut decoder = ffmpeg::codec::context::Context::from_parameters(parameters)?
            .decoder()
            .video()?;

        let merged = merged_ranges(clips, self.config.padding_ms, source_duration_ms);
        if merged.is_empty() {
            return Err(VideoIoError::Encode(
                "no clip ranges fall within the source video".into(),
            ));
        }

        let out_width = self.config.output_width.unwrap_or(decoder.width());
        let out_height = self.config.output_height.unwrap_or(decoder.height());

        let mut octx = ffmpeg::format::output(&self.output_path)
            .map_err(|e| VideoIoError::Encode(e.to_string()))?;
        let global_header = octx
            .format()
            .flags()
            .contains(ffmpeg::format::Flags::GLOBAL_HEADER);

        let codec = ffmpeg::encoder::find(ffmpeg::codec::Id::H264)
            .ok_or_else(|| VideoIoError::Encode("H.264 encoder not available".into()))?;
        let mut ost = octx.add_stream(codec)?;

        let mut encoder = ffmpeg::codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()?;
        encoder.set_width(out_width);
        encoder.set_height(out_height);
        encoder.set_format(ffmpeg::format::Pixel::YUV420P);
        let frame_rate = if frame_rate.numerator() > 0 {
            frame_rate
        } else {
            ffmpeg::Rational(30, 1)
        };
        // One tick per output frame (1/fps), so sequential integer pts values
        // are exact and strictly increasing — no rounding-induced duplicate
        // or out-of-order timestamps for the encoder to reject.
        let encoder_time_base = ffmpeg::Rational(frame_rate.denominator(), frame_rate.numerator());
        encoder.set_time_base(encoder_time_base);
        encoder.set_frame_rate(Some(frame_rate));
        if global_header {
            encoder.set_flags(ffmpeg::codec::Flags::GLOBAL_HEADER);
        }

        let mut opts = ffmpeg::Dictionary::new();
        opts.set("preset", "medium");
        match self.config.bitrate_kbps {
            Some(kbps) => encoder.set_bit_rate(kbps as usize * 1000),
            None => opts.set("crf", "23"),
        }

        let opened_encoder = encoder
            .open_with(opts)
            .map_err(|e| VideoIoError::Encode(e.to_string()))?;
        ost.set_parameters(&opened_encoder);
        let mut encoder = opened_encoder;

        octx.write_header()
            .map_err(|e| VideoIoError::Encode(e.to_string()))?;
        let ost_time_base = octx.stream(0).expect("just added one stream").time_base();

        let mut scaler = ffmpeg::software::scaling::Context::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            ffmpeg::format::Pixel::YUV420P,
            out_width,
            out_height,
            ffmpeg::software::scaling::Flags::BILINEAR,
        )?;

        let mut range_idx = 0usize;
        let mut output_frame_counter: i64 = 0;
        let mut eof_sent = false;

        loop {
            let mut decoded = ffmpeg::frame::Video::empty();
            match decoder.receive_frame(&mut decoded) {
                Ok(()) => {
                    let pts = decoded.timestamp().unwrap_or(0).max(0);
                    let ts_ms = pts
                        .rescale(input_time_base, ffmpeg::Rational(1, 1000))
                        .max(0) as u64;

                    while range_idx < merged.len() && ts_ms > merged[range_idx].1 {
                        range_idx += 1;
                    }
                    let in_range = range_idx < merged.len()
                        && ts_ms >= merged[range_idx].0
                        && ts_ms <= merged[range_idx].1;

                    if in_range {
                        let mut scaled = ffmpeg::frame::Video::empty();
                        scaler.run(&decoded, &mut scaled)?;
                        scaled.set_pts(Some(output_frame_counter));
                        output_frame_counter += 1;

                        encoder
                            .send_frame(&scaled)
                            .map_err(|e| VideoIoError::Encode(e.to_string()))?;
                        drain_encoder(&mut encoder, &mut octx, encoder_time_base, ost_time_base)?;
                    }
                }
                Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::error::EAGAIN => {
                    advance(&mut ictx, &mut decoder, video_stream_index, &mut eof_sent);
                }
                Err(ffmpeg::Error::Eof) => break,
                Err(e) => return Err(e.into()),
            }
        }

        encoder
            .send_eof()
            .map_err(|e| VideoIoError::Encode(e.to_string()))?;
        drain_encoder(&mut encoder, &mut octx, encoder_time_base, ost_time_base)?;
        octx.write_trailer()
            .map_err(|e| VideoIoError::Encode(e.to_string()))?;

        Ok(())
    }
}

/// Pulls the next packet for `video_stream_index` and feeds it to `decoder`.
/// Mirrors [`FrameExtractor::advance`]; see its docs for the EAGAIN/EOF protocol.
fn advance(
    ictx: &mut ffmpeg::format::context::Input,
    decoder: &mut ffmpeg::decoder::Video,
    video_stream_index: usize,
    eof_sent: &mut bool,
) -> bool {
    loop {
        match ictx.packets().next() {
            Some((stream, packet)) => {
                if stream.index() != video_stream_index {
                    continue;
                }
                if let Err(e) = decoder.send_packet(&packet) {
                    tracing::warn!(error = %e, "failed to send packet to decoder");
                    continue;
                }
                return true;
            }
            None => {
                if !*eof_sent {
                    let _ = decoder.send_eof();
                    *eof_sent = true;
                }
                return false;
            }
        }
    }
}

/// Drains all packets the encoder currently has buffered and muxes them.
fn drain_encoder(
    encoder: &mut ffmpeg::encoder::Video,
    octx: &mut ffmpeg::format::context::Output,
    encoder_time_base: ffmpeg::Rational,
    ost_time_base: ffmpeg::Rational,
) -> Result<()> {
    let mut packet = ffmpeg::Packet::empty();
    while encoder.receive_packet(&mut packet).is_ok() {
        packet.set_stream(0);
        packet.rescale_ts(encoder_time_base, ost_time_base);
        packet
            .write_interleaved(octx)
            .map_err(|e| VideoIoError::Encode(e.to_string()))?;
    }
    Ok(())
}

/// Pads each clip's range by `padding_ms`, clamps to `[0, source_duration_ms]`,
/// then sorts and merges overlapping ranges so concatenation has no
/// duplicated or out-of-order frames.
fn merged_ranges(clips: &[ClipSpec], padding_ms: u64, source_duration_ms: u64) -> Vec<(u64, u64)> {
    let mut ranges: Vec<(u64, u64)> = clips
        .iter()
        .filter_map(|c| {
            let start = c.start_ms.saturating_sub(padding_ms);
            let end = c.end_ms.saturating_add(padding_ms).min(source_duration_ms);
            (start < end).then_some((start, end))
        })
        .collect();
    ranges.sort_by_key(|r| r.0);

    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for (start, end) in ranges {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 {
                last.1 = last.1.max(end);
                continue;
            }
        }
        merged.push((start, end));
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merged_ranges_clamps_pads_and_merges_overlaps() {
        let clips = vec![
            ClipSpec {
                start_ms: 1_000,
                end_ms: 2_000,
                label: None,
            },
            // Overlaps the first clip once padding is applied (1900..=3500).
            ClipSpec {
                start_ms: 2_400,
                end_ms: 3_000,
                label: None,
            },
            // Far enough away to stay a separate range, but clamped to duration.
            ClipSpec {
                start_ms: 9_800,
                end_ms: 10_500,
                label: None,
            },
        ];

        let merged = merged_ranges(&clips, 500, 10_000);

        assert_eq!(merged, vec![(500, 3_500), (9_300, 10_000)]);
    }

    #[test]
    fn merged_ranges_drops_degenerate_ranges() {
        let clips = vec![ClipSpec {
            start_ms: 5_000,
            end_ms: 5_000,
            label: None,
        }];

        assert_eq!(merged_ranges(&clips, 0, 10_000), Vec::new());
    }
}
