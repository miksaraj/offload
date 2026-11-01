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
    /// timestamp is always sampled and the subject seeded there.
    pub fn scan(
        &self,
        source_path: &Path,
        start_ms: u64,
        end_ms: Option<u64>,
        seed: Option<Seed>,
        model: &mut ReIdModel,
    ) -> Result<Scan> {
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

            let tracks = tracker
                .update(detector.detect(&frame)?)
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

            let due = sampler.due(ts, &tracked);
            if !due.is_empty() {
                let boxes: Vec<Rect> = due.iter().map(|(_, r)| *r).collect();
                for ((id, _), e) in due.iter().zip(model.embed_boxes(&frame, &boxes)?) {
                    scan.log.signatures.entry(*id).or_default().push(e);
                }
            }
            scan.log.frames.push(FrameRecord {
                timestamp_ms: ts,
                frame_number: frame.frame_number,
                tracks: tracked,
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

    fn resolve_timeline(&self, scan: &Scan, selection: &Selection) -> SubjectTimeline {
        scan.seed_track
            .map(|seed| {
                reid::chain::resolve(
                    &scan.log,
                    seed,
                    &selection.embedding,
                    &self.config.reid.chain(),
                )
            })
            .unwrap_or_default()
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
        let mut model = ReIdModel::new(&self.config.models.reid)?;
        let candidates = self.candidates(source_path)?;
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
            let end = seed.timestamp_ms + u64::from(cfg.confirmation_scan_secs) * 1000;
            let scan = self.scan(
                source_path,
                seed.timestamp_ms,
                Some(end),
                Some(seed),
                &mut model,
            )?;
            let timeline = self.resolve_timeline(&scan, &selection);

            // Latest frame of the scan with the subject on screen: shows
            // tracking held, not just the pick again.
            let shown = (0..scan.log.frames.len())
                .rev()
                .find_map(|i| timeline.subject_at(&scan.log, i).map(|(_, r)| (i, *r)));
            let Some((idx, subject_box)) = shown else {
                continue;
            };
            let record = &scan.log.frames[idx];
            let frame = self.frame_at(source_path, record.frame_number)?;
            let others: Vec<Rect> = record
                .tracks
                .iter()
                .filter(|(_, r)| *r != subject_box)
                .map(|(_, r)| *r)
                .collect();
            if reid::identify::confirm_interactively(
                &frame,
                &subject_box,
                &others,
                &options.confirm_path,
                options.open_viewer,
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
            &mut model,
        )?;
        let timeline = self.resolve_timeline(&scan, &selection);
        Ok(SubjectResult {
            selection,
            log: scan.log,
            timeline,
        })
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
