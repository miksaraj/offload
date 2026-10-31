//! ByteTrack multi-object tracking, implemented in pure Rust (SPEC.md
//! Stage 3, Phases 3a/3b).
//!
//! Follows the reference ByteTrack algorithm (Zhang et al., 2022):
//!
//! 1. Split detections into **high** (`≥ high_confidence_threshold`) and
//!    **low** (`low_confidence_threshold ≤ score < high`) confidence.
//! 2. Predict every confirmed track (tracked or lost) one frame ahead
//!    with its [`kalman::KalmanFilter`].
//! 3. **First association:** confirmed + lost tracks vs high detections,
//!    by IoU, solved optimally ([`assignment::assign`]). A matched lost
//!    track is re-activated with its old ID.
//! 4. **Second association:** tracks still unmatched *and currently
//!    tracked* vs the low detections, with a stricter IoU. This is
//!    ByteTrack's key idea — occluded / motion-blurred players produce
//!    low-score boxes, and using them keeps their tracks alive instead of
//!    dropping and re-creating them. Tracks unmatched after this become
//!    `Lost`.
//! 5. Unconfirmed tracks (created last frame) vs leftover high
//!    detections; unmatched unconfirmed tracks are discarded, so a
//!    single-frame false positive never becomes a track.
//! 6. Leftover high detections scoring `≥ high + NEW_TRACK_MARGIN` start
//!    new unconfirmed tracks (confirmed immediately on the first frame).
//! 7. Tracks lost for more than `max_lost_frames` are removed. IDs are
//!    monotonically increasing from 1 and never reused.

pub mod assignment;
pub mod kalman;

use detector::{draw_label, draw_rect, Detection, Rect};
use image::{Rgb, RgbImage};
use kalman::{KalmanFilter, MeasVec, StateCov, StateVec};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TrackerError {
    #[error("tracking error: {0}")]
    Update(String),
}

pub type Result<T> = std::result::Result<T, TrackerError>;

/// IoU must exceed `1 - this` in the first (high-confidence) association.
const FIRST_MATCH_COST_LIMIT: f64 = 0.8;
/// Stricter limit for the second (low-confidence) association.
const SECOND_MATCH_COST_LIMIT: f64 = 0.5;
/// Limit for confirming last frame's new tracks.
const UNCONFIRMED_MATCH_COST_LIMIT: f64 = 0.7;
/// A new track needs `score ≥ high_confidence_threshold + this`.
const NEW_TRACK_MARGIN: f32 = 0.1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    Tracked,
    Lost,
    Removed,
}

/// A confirmed track as seen after one [`Tracker::update`].
#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    /// Stable for the life of the track; never reused within a run.
    pub track_id: u64,
    /// For `Tracked`: the Kalman-filtered box for this frame. For `Lost`:
    /// the motion-predicted box (where the player is expected to be).
    pub bbox: Rect,
    pub state: TrackState,
    /// Frames since the track was created.
    pub age: u32,
    /// 0 when matched this frame; counts up while `Lost`.
    pub frames_since_update: u32,
    /// Score of the detection last matched to this track.
    pub confidence: f32,
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
            // 0.4, not the 0.6 of ByteTrack's MOT17 setup (and SPEC's
            // first draft): YOLOv8n scores distant players ~0.3–0.6, so
            // at 0.6 only ~60% of detected players ever got a track on
            // real club footage. 0.4 tracked 32% more players with no
            // drop in mean frames per ID; 0.3 started fragmenting IDs.
            // See CLAUDE.md's Phase 3 real-footage check.
            high_confidence_threshold: 0.4,
            low_confidence_threshold: 0.1,
            max_lost_frames: 30,
        }
    }
}

#[derive(Debug, Clone)]
struct STrack {
    id: u64,
    mean: StateVec,
    cov: StateCov,
    state: TrackState,
    confirmed: bool,
    confidence: f32,
    start_frame: u64,
    last_update_frame: u64,
}

impl STrack {
    fn bbox(&self) -> Rect {
        xyah_to_rect(&self.mean)
    }

    fn to_track(&self, frame: u64) -> Track {
        Track {
            track_id: self.id,
            bbox: self.bbox(),
            state: self.state,
            age: (frame - self.start_frame) as u32,
            frames_since_update: (frame - self.last_update_frame) as u32,
            confidence: self.confidence,
        }
    }
}

/// Runs ByteTrack association across frames, maintaining stable track IDs.
/// Call [`update`](Self::update) once per *sampled* frame, in order.
pub struct Tracker {
    config: TrackerConfig,
    kf: KalmanFilter,
    tracks: Vec<STrack>,
    next_track_id: u64,
    frame: u64,
}

impl Tracker {
    pub fn new(config: TrackerConfig) -> Self {
        Self {
            config,
            kf: KalmanFilter::default(),
            tracks: Vec::new(),
            next_track_id: 1,
            frame: 0,
        }
    }

    pub fn config(&self) -> &TrackerConfig {
        &self.config
    }

    /// Creates a confirmed, `Tracked` track at `bbox` on the current frame
    /// and returns its ID. Used to seed the subject from the user's pick
    /// when the picked box scored below the new-track threshold (so
    /// `update` didn't create a track for it). Call after `update` for
    /// the frame the box belongs to; the track is matched like any other
    /// from the next frame on.
    pub fn add_track(&mut self, bbox: Rect, confidence: f32) -> u64 {
        let (mean, cov) = self.kf.initiate(&rect_to_xyah(&bbox));
        let id = self.next_track_id;
        self.next_track_id += 1;
        self.tracks.push(STrack {
            id,
            mean,
            cov,
            state: TrackState::Tracked,
            confirmed: true,
            confidence,
            start_frame: self.frame,
            last_update_frame: self.frame,
        });
        id
    }

    /// Advances one frame. Returns every confirmed track that is
    /// `Tracked` or `Lost` after this frame (removed tracks are dropped),
    /// sorted by `track_id`. Detections below `low_confidence_threshold`
    /// are ignored.
    pub fn update(&mut self, detections: Vec<Detection>) -> Result<Vec<Track>> {
        self.frame += 1;
        let frame = self.frame;
        let cfg = self.config.clone();

        if let Some(bad) = detections
            .iter()
            .find(|d| !(d.bbox.width.is_finite() && d.bbox.height.is_finite()))
        {
            return Err(TrackerError::Update(format!(
                "non-finite detection box {:?}",
                bad.bbox
            )));
        }

        let (high, low): (Vec<Detection>, Vec<Detection>) = detections
            .into_iter()
            .filter(|d| {
                d.confidence >= cfg.low_confidence_threshold
                    && d.bbox.width > 0.0
                    && d.bbox.height > 0.0
            })
            .partition(|d| d.confidence >= cfg.high_confidence_threshold);

        // Predict all live tracks. Like the reference implementation,
        // a lost track's height velocity is zeroed so a coasting box
        // doesn't grow or shrink without evidence.
        for t in &mut self.tracks {
            if t.state != TrackState::Tracked {
                t.mean[7] = 0.0;
            }
            let (m, c) = self.kf.predict(&t.mean, &t.cov);
            t.mean = m;
            t.cov = c;
        }

        let confirmed_pool: Vec<usize> = (0..self.tracks.len())
            .filter(|&i| self.tracks[i].confirmed)
            .collect();
        let unconfirmed: Vec<usize> = (0..self.tracks.len())
            .filter(|&i| !self.tracks[i].confirmed)
            .collect();

        // First association: confirmed (tracked + lost) vs high.
        let a1 = self.associate(&confirmed_pool, &high, FIRST_MATCH_COST_LIMIT);
        for &(ti, di) in &a1.matches {
            self.apply_match(confirmed_pool[ti], &high[di], frame);
        }

        // Second association: still-unmatched *tracked* tracks vs low.
        let remaining_tracked: Vec<usize> = a1
            .unmatched_rows
            .iter()
            .map(|&ti| confirmed_pool[ti])
            .filter(|&i| self.tracks[i].state == TrackState::Tracked)
            .collect();
        let a2 = self.associate(&remaining_tracked, &low, SECOND_MATCH_COST_LIMIT);
        for &(ti, di) in &a2.matches {
            self.apply_match(remaining_tracked[ti], &low[di], frame);
        }
        for &ti in &a2.unmatched_rows {
            self.tracks[remaining_tracked[ti]].state = TrackState::Lost;
        }

        // Unconfirmed tracks vs leftover high detections.
        let leftover_high: Vec<Detection> = a1
            .unmatched_cols
            .iter()
            .map(|&di| high[di].clone())
            .collect();
        let a3 = self.associate(&unconfirmed, &leftover_high, UNCONFIRMED_MATCH_COST_LIMIT);
        for &(ti, di) in &a3.matches {
            let i = unconfirmed[ti];
            self.apply_match(i, &leftover_high[di], frame);
            self.tracks[i].confirmed = true;
        }
        for &ti in &a3.unmatched_rows {
            self.tracks[unconfirmed[ti]].state = TrackState::Removed;
        }

        // New tracks from what's left.
        let new_track_threshold = cfg.high_confidence_threshold + NEW_TRACK_MARGIN;
        for &di in &a3.unmatched_cols {
            let d = &leftover_high[di];
            if d.confidence < new_track_threshold {
                continue;
            }
            let (mean, cov) = self.kf.initiate(&rect_to_xyah(&d.bbox));
            let id = self.next_track_id;
            self.next_track_id += 1;
            self.tracks.push(STrack {
                id,
                mean,
                cov,
                state: TrackState::Tracked,
                // ByteTrack confirms tracks born on the very first frame
                // straight away; later ones need a second sighting.
                confirmed: frame == 1,
                confidence: d.confidence,
                start_frame: frame,
                last_update_frame: frame,
            });
        }

        // Expire long-lost tracks, then drop removed ones.
        for t in &mut self.tracks {
            if t.state == TrackState::Lost
                && frame - t.last_update_frame > u64::from(cfg.max_lost_frames)
            {
                t.state = TrackState::Removed;
            }
        }
        self.tracks.retain(|t| t.state != TrackState::Removed);

        let mut out: Vec<Track> = self
            .tracks
            .iter()
            .filter(|t| t.confirmed)
            .map(|t| t.to_track(frame))
            .collect();
        out.sort_by_key(|t| t.track_id);
        Ok(out)
    }

    fn associate(
        &self,
        track_idx: &[usize],
        dets: &[Detection],
        cost_limit: f64,
    ) -> assignment::Assignment {
        let cost: Vec<Vec<f64>> = track_idx
            .iter()
            .map(|&i| {
                let tb = self.tracks[i].bbox();
                dets.iter()
                    .map(|d| 1.0 - f64::from(tb.iou(&d.bbox)))
                    .collect()
            })
            .collect();
        assignment::assign(&cost, track_idx.len(), dets.len(), cost_limit)
    }

    fn apply_match(&mut self, i: usize, d: &Detection, frame: u64) {
        let t = &mut self.tracks[i];
        let (m, c) = self.kf.update(&t.mean, &t.cov, &rect_to_xyah(&d.bbox));
        t.mean = m;
        t.cov = c;
        t.state = TrackState::Tracked;
        t.confidence = d.confidence;
        t.last_update_frame = frame;
    }
}

fn rect_to_xyah(r: &Rect) -> MeasVec {
    let w = f64::from(r.width);
    let h = f64::from(r.height).max(1e-6);
    MeasVec::new(f64::from(r.x) + w / 2.0, f64::from(r.y) + h / 2.0, w / h, h)
}

fn xyah_to_rect(m: &StateVec) -> Rect {
    let h = m[3].max(0.0);
    let w = (m[2] * h).max(0.0);
    Rect {
        x: (m[0] - w / 2.0) as f32,
        y: (m[1] - h / 2.0) as f32,
        width: w as f32,
        height: h as f32,
    }
}

/// A stable, distinct-ish colour per track ID, so the same player keeps
/// the same colour across debug frames (an ID switch shows as a colour
/// change).
pub fn track_colour(track_id: u64) -> Rgb<u8> {
    // Golden-ratio hue stepping, full saturation/value.
    let hue = (track_id as f64 * 0.618_033_988_75).fract() * 6.0;
    let x = 1.0 - (hue % 2.0 - 1.0).abs();
    let (r, g, b) = match hue as u32 {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8])
}

/// Draws `Tracked` tracks onto `img`: a box in the track's colour with its
/// ID label above the top-left corner (SPEC.md §5 `debug/tracking/`).
/// `Lost` tracks are drawn as a thin box at their predicted position.
pub fn draw_tracks(img: &mut RgbImage, tracks: &[Track]) {
    let (w, h) = img.dimensions();
    let thickness = (w.max(h) / 400).max(2);
    let scale = (w.max(h) / 400).max(2);
    for t in tracks {
        let colour = track_colour(t.track_id);
        match t.state {
            TrackState::Tracked => {
                draw_rect(img, &t.bbox, thickness, colour);
                let label_h = 7 * i64::from(scale);
                let ly = (t.bbox.y as i64 - label_h).max(0);
                draw_label(
                    img,
                    t.bbox.x as i64,
                    ly,
                    &t.track_id.to_string(),
                    Rgb([0, 0, 0]),
                    colour,
                    scale,
                );
            }
            TrackState::Lost => draw_rect(img, &t.bbox, 1, colour),
            TrackState::Removed => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f32, y: f32, conf: f32) -> Detection {
        Detection {
            bbox: Rect {
                x,
                y,
                width: 40.0,
                height: 100.0,
            },
            confidence: conf,
            class_id: 0,
        }
    }

    fn ids(tracks: &[Track]) -> Vec<u64> {
        tracks
            .iter()
            .filter(|t| t.state == TrackState::Tracked)
            .map(|t| t.track_id)
            .collect()
    }

    #[test]
    fn box_conversions_round_trip() {
        let r = Rect {
            x: 10.0,
            y: 20.0,
            width: 30.0,
            height: 60.0,
        };
        let mut m = StateVec::zeros();
        m.fixed_rows_mut::<4>(0).copy_from(&rect_to_xyah(&r));
        let back = xyah_to_rect(&m);
        assert!((back.x - r.x).abs() < 1e-4 && (back.width - r.width).abs() < 1e-4);
        assert!((back.y - r.y).abs() < 1e-4 && (back.height - r.height).abs() < 1e-4);
    }

    #[test]
    fn keeps_ids_stable_for_moving_players() {
        let mut tr = Tracker::new(TrackerConfig::default());
        let mut first = Vec::new();
        for f in 0..100 {
            let dx = f as f32 * 3.0;
            let tracks = tr
                .update(vec![
                    det(100.0 + dx, 100.0, 0.9),
                    det(400.0 - dx, 300.0, 0.9),
                ])
                .unwrap();
            if f == 0 {
                first = ids(&tracks);
                assert_eq!(first, vec![1, 2]);
            } else {
                assert_eq!(ids(&tracks), first, "frame {f}");
            }
        }
    }

    #[test]
    fn crossing_players_keep_their_own_ids() {
        // A moves right, B moves left on a slightly different row; they
        // overlap around frame 25. The Kalman velocities should carry each
        // ID through the crossing.
        let mut tr = Tracker::new(TrackerConfig::default());
        let mut a_id = 0;
        let mut b_id = 0;
        for f in 0..50 {
            let a = det(100.0 + 8.0 * f as f32, 100.0, 0.9);
            let b = det(500.0 - 8.0 * f as f32, 130.0, 0.9);
            let tracks = tr.update(vec![a.clone(), b.clone()]).unwrap();
            let id_near = |d: &Detection| {
                tracks
                    .iter()
                    .filter(|t| t.state == TrackState::Tracked)
                    .max_by(|x, y| x.bbox.iou(&d.bbox).total_cmp(&y.bbox.iou(&d.bbox)))
                    .unwrap()
                    .track_id
            };
            if f == 0 {
                a_id = id_near(&a);
                b_id = id_near(&b);
                assert_ne!(a_id, b_id);
            } else {
                assert_eq!(id_near(&a), a_id, "frame {f}");
                assert_eq!(id_near(&b), b_id, "frame {f}");
            }
        }
    }

    #[test]
    fn short_occlusion_recovers_the_same_id() {
        let mut tr = Tracker::new(TrackerConfig::default());
        for f in 0..10 {
            tr.update(vec![det(100.0 + 5.0 * f as f32, 100.0, 0.9)])
                .unwrap();
        }
        // Gone for 8 frames: the track is Lost (predicted onwards), not removed.
        for _ in 10..18 {
            let tracks = tr.update(vec![]).unwrap();
            assert_eq!(tracks.len(), 1);
            assert_eq!(tracks[0].state, TrackState::Lost);
        }
        // Reappears where constant velocity says it should be.
        let tracks = tr
            .update(vec![det(100.0 + 5.0 * 18.0, 100.0, 0.9)])
            .unwrap();
        assert_eq!(ids(&tracks), vec![1]);
    }

    #[test]
    fn low_confidence_detections_keep_a_track_alive() {
        // The ByteTrack second association: a player whose score dips
        // below the high threshold (occlusion/blur) stays Tracked under
        // the same ID instead of going Lost.
        let mut tr = Tracker::new(TrackerConfig::default());
        tr.update(vec![det(100.0, 100.0, 0.9)]).unwrap();
        for f in 1..10 {
            let tracks = tr.update(vec![det(100.0 + f as f32, 100.0, 0.3)]).unwrap();
            assert_eq!(ids(&tracks), vec![1], "frame {f}");
        }
        // Without the low-confidence boxes (filtered out by a higher low
        // threshold), the same sequence loses the track.
        let mut strict = Tracker::new(TrackerConfig {
            low_confidence_threshold: 0.35,
            ..TrackerConfig::default()
        });
        strict.update(vec![det(100.0, 100.0, 0.9)]).unwrap();
        let tracks = strict.update(vec![det(101.0, 100.0, 0.3)]).unwrap();
        assert_eq!(tracks[0].state, TrackState::Lost);
    }

    #[test]
    fn low_confidence_detections_never_start_tracks() {
        let mut tr = Tracker::new(TrackerConfig::default());
        for _ in 0..5 {
            assert!(tr.update(vec![det(100.0, 100.0, 0.3)]).unwrap().is_empty());
        }
    }

    #[test]
    fn single_frame_false_positives_never_become_tracks() {
        let mut tr = Tracker::new(TrackerConfig::default());
        tr.update(vec![det(100.0, 100.0, 0.9)]).unwrap(); // frame 1: id 1
                                                          // Frame 2: a one-off detection elsewhere → unconfirmed, not output.
        let tracks = tr
            .update(vec![det(100.0, 100.0, 0.9), det(600.0, 400.0, 0.9)])
            .unwrap();
        assert_eq!(ids(&tracks), vec![1]);
        // Frame 3: it's gone → discarded without ever being reported.
        let tracks = tr.update(vec![det(100.0, 100.0, 0.9)]).unwrap();
        assert_eq!(ids(&tracks), vec![1]);
        // A detection that persists is confirmed on its second frame, and
        // takes the next ID after the discarded one (IDs aren't reused).
        tr.update(vec![det(100.0, 100.0, 0.9), det(600.0, 400.0, 0.9)])
            .unwrap();
        let tracks = tr
            .update(vec![det(100.0, 100.0, 0.9), det(600.0, 400.0, 0.9)])
            .unwrap();
        assert_eq!(ids(&tracks), vec![1, 3]);
    }

    #[test]
    fn long_lost_tracks_are_removed_and_ids_not_reused() {
        let cfg = TrackerConfig {
            max_lost_frames: 5,
            ..TrackerConfig::default()
        };
        let mut tr = Tracker::new(cfg);
        tr.update(vec![det(100.0, 100.0, 0.9)]).unwrap();
        for _ in 0..5 {
            assert_eq!(tr.update(vec![]).unwrap().len(), 1);
        }
        assert!(
            tr.update(vec![]).unwrap().is_empty(),
            "removed after 5 lost frames"
        );
        // Same place again: a brand-new ID, confirmed on its second frame.
        tr.update(vec![det(100.0, 100.0, 0.9)]).unwrap();
        let tracks = tr.update(vec![det(100.0, 100.0, 0.9)]).unwrap();
        assert_eq!(ids(&tracks), vec![2]);
    }

    #[test]
    fn added_tracks_are_confirmed_and_followed() {
        let mut tr = Tracker::new(TrackerConfig::default());
        tr.update(vec![]).unwrap();
        // A 0.3 box would never start a track by itself.
        let id = tr.add_track(det(100.0, 100.0, 0.3).bbox, 0.3);
        assert_eq!(id, 1);
        for f in 1..10 {
            let tracks = tr
                .update(vec![det(100.0 + 2.0 * f as f32, 100.0, 0.3)])
                .unwrap();
            assert_eq!(ids(&tracks), vec![id], "frame {f}");
        }
        // IDs keep counting from there.
        tr.update(vec![det(100.0 + 20.0, 100.0, 0.3), det(500.0, 300.0, 0.9)])
            .unwrap();
        let tracks = tr
            .update(vec![det(100.0 + 22.0, 100.0, 0.3), det(500.0, 300.0, 0.9)])
            .unwrap();
        assert_eq!(ids(&tracks), vec![1, 2]);
    }

    #[test]
    fn rejects_non_finite_boxes() {
        let mut tr = Tracker::new(TrackerConfig::default());
        let mut d = det(0.0, 0.0, 0.9);
        d.bbox.width = f32::NAN;
        assert!(tr.update(vec![d]).is_err());
    }

    #[test]
    fn draw_tracks_labels_tracked_boxes_in_their_colour() {
        let mut img = RgbImage::from_pixel(200, 200, Rgb([0, 0, 0]));
        let track = Track {
            track_id: 7,
            bbox: Rect {
                x: 50.0,
                y: 60.0,
                width: 40.0,
                height: 80.0,
            },
            state: TrackState::Tracked,
            age: 3,
            frames_since_update: 0,
            confidence: 0.9,
        };
        draw_tracks(&mut img, &[track]);
        let c = track_colour(7);
        assert_eq!(*img.get_pixel(50, 100), c); // left edge
        assert_eq!(*img.get_pixel(50, 47), c); // label background above the box
        assert_eq!(*img.get_pixel(70, 100), Rgb([0, 0, 0])); // interior untouched
        assert_ne!(track_colour(1), track_colour(2));
    }
}
