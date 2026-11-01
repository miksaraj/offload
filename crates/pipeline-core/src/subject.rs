//! Stage 4 orchestration: identification → seeding → confirmation →
//! full-video pass → identity chaining (SPEC.md Stage 4, Phases 4b–4d).
//!
//! Domain logic lives in `reid` (`identify`, `chain`); this module only
//! drives frames through detector → tracker → signature sampling and
//! wires the steps together.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use detector::Rect;
use reid::chain::{FrameRecord, TrackLog};
use reid::{ReIdModel, Selection, SubjectTimeline};
use tracker::{TrackState, Tracker};
use video_io::{Frame, FrameExtractor};

use crate::segmentation::{self, MatchSegment};
use crate::{Pipeline, PipelineError, Result};

/// Where to seed the subject during a scan: the picked box, on the first
/// frame at or after `timestamp_ms` (the pick's own frame).
#[derive(Debug, Clone, Copy)]
pub struct Seed {
    pub timestamp_ms: u64,
    pub bbox: Rect,
    pub confidence: f32,
}

impl From<&Selection> for Seed {
    fn from(s: &Selection) -> Self {
        Self {
            timestamp_ms: s.frame.timestamp_ms,
            bbox: s.bbox,
            confidence: s.confidence,
        }
    }
}

/// A scan's output: the track log and which track the seed became.
#[derive(Debug, Clone, Default)]
pub struct Scan {
    pub log: TrackLog,
    pub seed_track: Option<u64>,
}

/// The confirmed subject over the whole video.
#[derive(Debug, Clone)]
pub struct SubjectResult {
    pub selection: Selection,
    /// The match segments in force (one implicit segment by default).
    pub segments: Vec<MatchSegment>,
    pub log: TrackLog,
    pub timeline: SubjectTimeline,
}

/// Minimum IoU between the picked box and a track for that track to be
/// the seed; below it, a track is created from the pick instead.
const SEED_MIN_IOU: f32 = 0.3;

impl Pipeline {
    /// Runs detection + tracking + signature sampling over
    /// `[start_ms, end_ms)` (to the end if `None`), sampled at
    /// `video.detection_fps`. If `seed` is given, the frame at the seed's
    /// timestamp is always sampled and the subject seeded there. Without
    /// a `model`, no appearance signatures are collected (detection counts
    /// for segmentation only).
    pub fn scan(
        &self,
        source_path: &Path,
        start_ms: u64,
        end_ms: Option<u64>,
        seed: Option<Seed>,
        mut model: Option<&mut ReIdModel>,
    ) -> Result<Scan> {
        let sure = self.config.detector.confidence_threshold;
        let mut detector = detector::Detector::new(
            &self.config.models.detector,
            self.config.detector_config_for_tracking(),
        )?;
        let mut tracker = Tracker::new((&self.config.tracker).into());
        let mut sampler = self.config.reid.sampler();
        let interval_ms = 1000.0 / f64::from(self.config.video.detection_fps.max(1));
        let mut next_sample_ms = start_ms as f64;
        let mut pending_seed = seed;
        let mut scan = Scan::default();

        for frame in FrameExtractor::new(source_path)? {
            let ts = frame.timestamp_ms;
            if end_ms.is_some_and(|e| ts >= e) {
                break;
            }
            let seed_here = pending_seed.is_some_and(|s| ts >= s.timestamp_ms);
            if ts < start_ms || ((ts as f64) < next_sample_ms && !seed_here) {
                continue;
            }
            while next_sample_ms <= ts as f64 {
                next_sample_ms += interval_ms;
            }

            let detections = detector.detect(&frame)?;
            let sure_count = detections.iter().filter(|d| d.confidence >= sure).count();
            let tracks = tracker
                .update(detections)
                .map_err(|e| PipelineError::Stage(e.to_string()))?;
            let mut tracked: Vec<(u64, Rect)> = tracks
                .iter()
                .filter(|t| t.state == TrackState::Tracked)
                .map(|t| (t.track_id, t.bbox))
                .collect();

            if seed_here {
                let s = pending_seed.take().expect("seed_here implies a seed");
                let best = tracked
                    .iter()
                    .map(|(id, r)| (*id, r.iou(&s.bbox)))
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .filter(|&(_, iou)| iou >= SEED_MIN_IOU);
                let id = match best {
                    Some((id, _)) => id,
                    None => {
                        let id = tracker.add_track(s.bbox, s.confidence);
                        tracked.push((id, s.bbox));
                        id
                    }
                };
                tracing::info!(track = id, timestamp_ms = ts, "subject seeded");
                scan.seed_track = Some(id);
            }

            if let Some(model) = model.as_deref_mut() {
                let due = sampler.due(ts, &tracked);
                if !due.is_empty() {
                    let boxes: Vec<Rect> = due.iter().map(|(_, r)| *r).collect();
                    for ((id, _), e) in due.iter().zip(model.embed_boxes(&frame, &boxes)?) {
                        scan.log.signatures.entry(*id).or_default().push(e);
                    }
                }
            }
            scan.log.frames.push(FrameRecord {
                timestamp_ms: ts,
                frame_number: frame.frame_number,
                tracks: tracked,
                detections: sure_count,
            });
        }
        Ok(scan)
    }

    /// Decodes forward to the frame with `frame_number`.
    pub fn frame_at(&self, source_path: &Path, frame_number: u64) -> Result<Frame> {
        FrameExtractor::new(source_path)?
            .find(|f| f.frame_number == frame_number)
            .ok_or_else(|| PipelineError::Stage(format!("frame {frame_number} not found")))
    }

    fn resolve_timeline(
        &self,
        log: &TrackLog,
        seed_track: Option<u64>,
        selection: &Selection,
        segments: &[MatchSegment],
    ) -> SubjectTimeline {
        seed_track
            .map(|seed| {
                reid::chain::resolve_segmented(
                    log,
                    &segment_ranges(log, segments),
                    seed,
                    &selection.embedding,
                    &self.config.reid.chain(),
                )
            })
            .unwrap_or_default()
    }

    /// The identification window `[start, end)`: inside the target
    /// segment (the one containing `reid.candidate_window_start_secs`, or
    /// the first), starting there or at that start time, lasting
    /// `reid.candidate_window_secs`, clipped to the segment.
    fn identification_window(&self, segments: &[MatchSegment]) -> Result<(u64, u64)> {
        let cfg = &self.config.reid;
        let start_at = cfg.candidate_window_start_secs.map(|s| u64::from(s) * 1000);
        let target = segmentation::target_segment(segments, start_at).ok_or_else(|| {
            PipelineError::Config(
                "reid.candidate_window_start_secs falls outside every match segment".into(),
            )
        })?;
        let seg = &segments[target];
        let start = start_at.unwrap_or(seg.start_ms).max(seg.start_ms);
        let end = (start + u64::from(cfg.candidate_window_secs) * 1000).min(seg.end_ms);
        Ok((start, end))
    }

    /// Stage 4 end to end: identification (4b) → confirmation scan and
    /// `[y/N]` (4c), retrying from the next candidate frame on `n` up to
    /// `reid.max_identification_attempts` → full-video pass and identity
    /// chaining (4d).
    pub fn follow_subject(
        &self,
        source_path: &Path,
        options: &reid::InteractionOptions,
        input: &mut impl BufRead,
        out: &mut impl Write,
    ) -> Result<SubjectResult> {
        let cfg = &self.config.reid;
        let video_ms = FrameExtractor::new(source_path)?.duration_ms();
        if self.config.segmentation.auto() {
            return self.follow_subject_auto_segmented(source_path, video_ms, options, input, out);
        }
        let segments = self.config.segmentation.manual_or_implicit(video_ms)?;
        let (win_start, win_end) = self.identification_window(&segments)?;
        let mut model = ReIdModel::new(&self.config.models.reid)?;
        let candidates = self.candidates_in(source_path, win_start, win_end)?;
        let attempts = cfg.max_identification_attempts.max(1) as usize;

        let mut confirmed = None;
        for attempt in 0..attempts {
            let selection = reid::identify::identify_from(
                &candidates,
                attempt,
                &mut model,
                options,
                input,
                out,
            )?;
            let seed = Seed::from(&selection);
            // The confirmation scan stays inside the pick's own match.
            let seg_end = segments
                .iter()
                .find(|s| s.contains(seed.timestamp_ms))
                .map_or(u64::MAX, |s| s.end_ms);
            let end =
                (seed.timestamp_ms + u64::from(cfg.confirmation_scan_secs) * 1000).min(seg_end);
            let scan = self.scan(
                source_path,
                seed.timestamp_ms,
                Some(end),
                Some(seed),
                Some(&mut model),
            )?;
            let timeline = self.resolve_timeline(
                &scan.log,
                scan.seed_track,
                &selection,
                &[segmentation::whole(0)],
            );

            // Latest frame of the scan with the subject on screen: shows
            // tracking held, not just the pick again.
            if self.confirm_preview(
                source_path,
                &scan.log,
                &timeline,
                0..scan.log.frames.len(),
                options,
                input,
                out,
            )? {
                confirmed = Some(selection);
                break;
            }
        }
        let selection = confirmed.ok_or_else(|| {
            PipelineError::ReId(reid::ReIdError::Identification(format!(
                "subject not confirmed after {attempts} attempts (reid.max_identification_attempts)"
            )))
        })?;

        writeln!(out, "\nConfirmed. Following you through the whole video…")
            .and_then(|_| out.flush())
            .map_err(|e| PipelineError::Stage(e.to_string()))?;
        let scan = self.scan(
            source_path,
            0,
            None,
            Some(Seed::from(&selection)),
            Some(&mut model),
        )?;
        let timeline = self.resolve_timeline(&scan.log, scan.seed_track, &selection, &segments);
        Ok(SubjectResult {
            selection,
            segments,
            log: scan.log,
            timeline,
        })
    }

    /// Shows the latest frame in `range` where the subject is present
    /// (green) and asks `[y/N]`. `Ok(false)` also when the subject never
    /// appears in `range`.
    #[allow(clippy::too_many_arguments)]
    fn confirm_preview(
        &self,
        source_path: &Path,
        log: &TrackLog,
        timeline: &SubjectTimeline,
        range: std::ops::Range<usize>,
        options: &reid::InteractionOptions,
        input: &mut impl BufRead,
        out: &mut impl Write,
    ) -> Result<bool> {
        // Latest frame with the subject on screen: shows tracking held,
        // not just the pick again.
        let Some((idx, subject_box)) = range
            .rev()
            .find_map(|i| timeline.subject_at(log, i).map(|(_, r)| (i, *r)))
        else {
            return Ok(false);
        };
        let record = &log.frames[idx];
        let frame = self.frame_at(source_path, record.frame_number)?;
        let others: Vec<Rect> = record
            .tracks
            .iter()
            .filter(|(_, r)| *r != subject_box)
            .map(|(_, r)| *r)
            .collect();
        Ok(reid::identify::confirm_interactively(
            &frame,
            &subject_box,
            &others,
            &options.confirm_path,
            options.open_viewer,
            input,
            out,
        )?)
    }

    /// The auto-segmentation flow (`segmentation.enabled`, no manual
    /// list): segments come from the footage, so the full-video scan runs
    /// *first*. Then the proposed segments are confirmed or edited, the
    /// user picks themselves inside the target segment from tracked boxes
    /// in that scan (so the pick is exactly a track: no rescan, no
    /// seeding guesswork), and the confirmation preview and final chaining
    /// both come from the same log.
    fn follow_subject_auto_segmented(
        &self,
        source_path: &Path,
        video_ms: u64,
        options: &reid::InteractionOptions,
        input: &mut impl BufRead,
        out: &mut impl Write,
    ) -> Result<SubjectResult> {
        let cfg = &self.config.reid;
        let mut model = ReIdModel::new(&self.config.models.reid)?;
        writeln!(
            out,
            "Scanning the whole video for players and breaks between matches…"
        )
        .and_then(|_| out.flush())
        .map_err(|e| PipelineError::Stage(e.to_string()))?;
        let scan = self.scan(source_path, 0, None, None, Some(&mut model))?;
        let samples: Vec<(u64, usize)> = scan
            .log
            .frames
            .iter()
            .map(|f| (f.timestamp_ms, f.detections))
            .collect();
        let proposed = segmentation::detect_segments(&samples, video_ms, &self.config.segmentation);
        let segments = segmentation::confirm_segments(proposed, video_ms, input, out)?;

        let (win_start, win_end) = self.identification_window(&segments)?;
        let candidates = self.candidates_from_log(source_path, &scan.log, win_start, win_end)?;
        let attempts = cfg.max_identification_attempts.max(1) as usize;
        for attempt in 0..attempts {
            let selection = reid::identify::identify_from(
                &candidates,
                attempt,
                &mut model,
                options,
                input,
                out,
            )?;
            let seed_idx = scan
                .log
                .frames
                .iter()
                .position(|f| f.frame_number == selection.frame.frame_number);
            let seed_track = seed_idx.and_then(|i| {
                scan.log.frames[i]
                    .tracks
                    .iter()
                    .find(|(_, r)| *r == selection.bbox)
                    .map(|(id, _)| *id)
            });
            let timeline = self.resolve_timeline(&scan.log, seed_track, &selection, &segments);
            let from = seed_idx.unwrap_or(0);
            let until_ms =
                selection.frame.timestamp_ms + u64::from(cfg.confirmation_scan_secs) * 1000;
            let to = scan
                .log
                .frames
                .iter()
                .position(|f| f.timestamp_ms >= until_ms)
                .unwrap_or(scan.log.frames.len());
            if self.confirm_preview(
                source_path,
                &scan.log,
                &timeline,
                from..to,
                options,
                input,
                out,
            )? {
                return Ok(SubjectResult {
                    selection,
                    segments,
                    log: scan.log,
                    timeline,
                });
            }
        }
        Err(PipelineError::ReId(reid::ReIdError::Identification(
            format!(
            "subject not confirmed after {attempts} attempts (reid.max_identification_attempts)"
        ),
        )))
    }

    /// Identification candidates taken from an existing scan's log: the
    /// sampled frames nearest `reid.candidate_frame_count` timestamps in
    /// `[start, end)`, offering that frame's tracked boxes.
    fn candidates_from_log(
        &self,
        source_path: &Path,
        log: &TrackLog,
        start: u64,
        end: u64,
    ) -> Result<Vec<reid::Candidate>> {
        let targets = reid::identify::candidate_timestamps(
            start,
            end.saturating_sub(start),
            0,
            self.config.reid.candidate_frame_count,
        );
        let mut picks: Vec<usize> = targets
            .iter()
            .filter_map(|&t| {
                log.frames
                    .iter()
                    .position(|f| f.timestamp_ms >= t && f.timestamp_ms < end)
            })
            .collect();
        picks.dedup();
        let wanted: Vec<u64> = picks.iter().map(|&i| log.frames[i].frame_number).collect();
        let mut candidates = Vec::new();
        for frame in FrameExtractor::new(source_path)? {
            if let Some(pos) = wanted.iter().position(|&n| n == frame.frame_number) {
                let detections = log.frames[picks[pos]]
                    .tracks
                    .iter()
                    .map(|(_, r)| detector::Detection {
                        bbox: *r,
                        confidence: 1.0,
                        class_id: detector::PERSON_CLASS_ID,
                    })
                    .collect();
                candidates.push(reid::Candidate { frame, detections });
                if candidates.len() == wanted.len() {
                    break;
                }
            }
        }
        Ok(candidates)
    }

    /// Writes one debug frame per sampled frame to `dir`: the subject in a
    /// green box (when present), other tracked players in grey
    /// (SPEC.md §5 `debug/reid/`).
    pub fn write_subject_debug_frames(
        &self,
        source_path: &Path,
        result: &SubjectResult,
        dir: &Path,
    ) -> Result<usize> {
        std::fs::create_dir_all(dir)
            .map_err(|e| PipelineError::Output(dir.to_path_buf(), e.to_string()))?;
        let log = &result.log;
        let mut idx = 0;
        let mut written = 0;
        for frame in FrameExtractor::new(source_path)? {
            let Some(record) = log.frames.get(idx) else {
                break;
            };
            if frame.frame_number != record.frame_number {
                continue;
            }
            let subject = result.timeline.subject_at(log, idx).map(|(_, r)| *r);
            let others: Vec<Rect> = record
                .tracks
                .iter()
                .filter(|(_, r)| Some(*r) != subject)
                .map(|(_, r)| *r)
                .collect();
            let mut img = match subject {
                Some(s) => reid::identify::annotate_subject(&frame, &s, &others),
                None => {
                    let mut img =
                        image::RgbImage::from_raw(frame.width, frame.height, frame.pixels.clone())
                            .expect("frame buffer");
                    for r in &others {
                        detector::draw_rect(&mut img, r, 1, image::Rgb([170, 170, 170]));
                    }
                    img
                }
            };
            // Track IDs, so re-entry decisions in the debug log
            // (`RUST_LOG=reid=debug`) can be matched to people on screen.
            for (id, r) in &record.tracks {
                let is_subject = Some(*r) == subject;
                let (fg, bg) = if is_subject {
                    (image::Rgb([0, 0, 0]), image::Rgb([0, 230, 0]))
                } else {
                    (image::Rgb([0, 0, 0]), image::Rgb([170, 170, 170]))
                };
                detector::draw_label(
                    &mut img,
                    r.x as i64,
                    (r.y as i64 - 14).max(0),
                    &id.to_string(),
                    fg,
                    bg,
                    2,
                );
            }
            let path: PathBuf = dir.join(format!(
                "frame_{:06}_{:08}ms.jpg",
                frame.frame_number, frame.timestamp_ms
            ));
            img.save(&path)
                .map_err(|e| PipelineError::Output(path.clone(), e.to_string()))?;
            written += 1;
            idx += 1;
        }
        Ok(written)
    }
}

/// Inclusive frame-index ranges into `log` for each segment (segments
/// with no sampled frames are skipped).
pub fn segment_ranges(log: &TrackLog, segments: &[MatchSegment]) -> Vec<(usize, usize)> {
    segments
        .iter()
        .filter_map(|seg| {
            let first = log
                .frames
                .iter()
                .position(|f| seg.contains(f.timestamp_ms))?;
            let last = log
                .frames
                .iter()
                .rposition(|f| seg.contains(f.timestamp_ms))?;
            Some((first, last))
        })
        .collect()
}
