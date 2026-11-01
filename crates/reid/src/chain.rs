//! Subject identity chaining (SPEC.md Stage 4, Phases 4c/4d).
//!
//! The tracker leads: a subject track stays the subject for its whole
//! life. Appearance only bridges *gaps* — when a subject track ends, which
//! later track (if any) is the subject coming back? — and those decisions
//! are **relative**, never "cosine above X means it's you", because on
//! real footage same-kit teammates overlap heavily with the subject's own
//! appearance (see CLAUDE.md's Phase 4a measurements). A re-entry
//! candidate must clear an absolute floor, beat this video's own
//! teammate baseline (tracks seen *alongside* the subject, so known not
//! to be them) by a margin, and beat the runner-up candidate by the same
//! margin. When nothing qualifies, the subject is absent: missing a few
//! seconds is better than putting a teammate's play in the reel.
//!
//! Everything here is pure logic over a recorded [`TrackLog`], so it runs
//! offline after tracking (in both directions from the seed) and is
//! testable with synthetic sequences.

use std::collections::{HashMap, HashSet};

use detector::Rect;

use crate::Embedding;

/// The tracked (not `Lost`) players on one sampled frame.
#[derive(Debug, Clone, Default)]
pub struct FrameRecord {
    pub timestamp_ms: u64,
    pub frame_number: u64,
    pub tracks: Vec<(u64, Rect)>,
    /// "Sure" detections on this frame (at `detector.confidence_threshold`),
    /// the signal match segmentation reads.
    pub detections: usize,
}

/// Tracking output for a span of video: per-sampled-frame tracked boxes,
/// plus each track's sparse appearance signature.
#[derive(Debug, Clone, Default)]
pub struct TrackLog {
    pub frames: Vec<FrameRecord>,
    pub signatures: HashMap<u64, Vec<Embedding>>,
}

impl TrackLog {
    /// `(first, last)` frame index at which each track is `Tracked`.
    pub fn spans(&self) -> HashMap<u64, (usize, usize)> {
        spans_of(&self.frames)
    }
}

fn spans_of(frames: &[FrameRecord]) -> HashMap<u64, (usize, usize)> {
    let mut spans: HashMap<u64, (usize, usize)> = HashMap::new();
    for (i, f) in frames.iter().enumerate() {
        for &(id, _) in &f.tracks {
            spans.entry(id).and_modify(|s| s.1 = i).or_insert((i, i));
        }
    }
    spans
}

/// Decides which tracked boxes get a signature sample on a frame: the
/// first eligible frame of a track, then every `interval_ms`, up to
/// `max_samples`, only for boxes at least `min_height` px tall.
#[derive(Debug, Clone)]
pub struct SignatureSampler {
    interval_ms: u64,
    max_samples: usize,
    min_height: f32,
    last: HashMap<u64, (u64, usize)>,
}

impl SignatureSampler {
    pub fn new(interval_ms: u64, max_samples: usize, min_height: f32) -> Self {
        Self {
            interval_ms,
            max_samples,
            min_height,
            last: HashMap::new(),
        }
    }

    /// The boxes to embed on this frame (and records them as sampled).
    pub fn due(&mut self, timestamp_ms: u64, tracks: &[(u64, Rect)]) -> Vec<(u64, Rect)> {
        let mut out = Vec::new();
        for &(id, bbox) in tracks {
            if bbox.height < self.min_height {
                continue;
            }
            let ready = match self.last.get(&id) {
                None => true,
                Some(&(ts, n)) => {
                    n < self.max_samples && timestamp_ms >= ts.saturating_add(self.interval_ms)
                }
            };
            if ready {
                let n = self.last.get(&id).map_or(0, |&(_, n)| n);
                self.last.insert(id, (timestamp_ms, n + 1));
                out.push((id, bbox));
            }
        }
        out
    }
}

/// Re-entry acceptance settings (`[reid]` config; see SPEC Stage 4).
#[derive(Debug, Clone)]
pub struct ChainConfig {
    /// Absolute floor: rules out clearly different people.
    pub reentry_threshold: f32,
    /// Required lead over the teammate baseline and over the runner-up.
    pub reentry_margin: f32,
    /// Candidates must start (forward) / end (backward) within this of
    /// the subject's loss; on no acceptance the window slides on.
    pub reentry_window_ms: u64,
    /// Signature samples a candidate needs to be considered.
    pub reentry_min_samples: usize,
    pub gallery_max_size: usize,
}

impl Default for ChainConfig {
    fn default() -> Self {
        Self {
            reentry_threshold: 0.55,
            reentry_margin: 0.05,
            reentry_window_ms: 10_000,
            reentry_min_samples: 2,
            gallery_max_size: 64,
        }
    }
}

/// A stretch where `track_id` is the subject, as frame indices into the
/// [`TrackLog`] (inclusive).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubjectSpan {
    pub track_id: u64,
    pub first: usize,
    pub last: usize,
}

/// Who the subject is, over a [`TrackLog`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SubjectTimeline {
    /// Ordered by time, non-overlapping.
    pub spans: Vec<SubjectSpan>,
    /// Re-entries accepted (spans beyond the seed).
    pub reentries: usize,
    /// Candidate sets examined and rejected as ambiguous or too unlike
    /// the subject (each left the subject absent for that window).
    pub rejected_windows: usize,
}

impl SubjectTimeline {
    /// The subject's box on frame `idx`, if present.
    pub fn subject_at<'a>(&self, log: &'a TrackLog, idx: usize) -> Option<(u64, &'a Rect)> {
        let span = self
            .spans
            .iter()
            .find(|s| (s.first..=s.last).contains(&idx))?;
        log.frames
            .get(idx)?
            .tracks
            .iter()
            .find(|(id, _)| *id == span.track_id)
            .map(|(id, r)| (*id, r))
    }

    /// Share of sampled frames on which the subject is present.
    pub fn coverage(&self, log: &TrackLog) -> f64 {
        if log.frames.is_empty() {
            return 0.0;
        }
        let present = (0..log.frames.len())
            .filter(|&i| self.subject_at(log, i).is_some())
            .count();
        present as f64 / log.frames.len() as f64
    }
}

fn score(samples: &[Embedding], gallery: &[Embedding]) -> f32 {
    if samples.is_empty() || gallery.is_empty() {
        return 0.0;
    }
    let total: f32 = samples
        .iter()
        .map(|s| gallery.iter().map(|g| s.cosine(g)).fold(f32::MIN, f32::max))
        .sum();
    total / samples.len() as f32
}

fn percentile_90(mut v: Vec<f32>) -> Option<f32> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Some(v[((v.len() - 1) as f64 * 0.9).round() as usize])
}

struct Chainer<'a> {
    /// The frames being chained over: the whole log, or one match
    /// segment's slice of it (chaining never crosses a segment boundary).
    frames: &'a [FrameRecord],
    signatures: &'a HashMap<u64, Vec<Embedding>>,
    /// Teammate baseline carried in from another segment, used until this
    /// one has subject spans of its own to derive one from.
    external_baseline: Option<f32>,
    cfg: &'a ChainConfig,
    spans: HashMap<u64, (usize, usize)>,
    gallery: Vec<Embedding>,
    subject: Vec<SubjectSpan>,
    used: HashSet<u64>,
    reentries: usize,
    rejected_windows: usize,
}

impl Chainer<'_> {
    fn ts(&self, idx: usize) -> u64 {
        self.frames[idx].timestamp_ms
    }

    fn samples(&self, id: u64) -> &[Embedding] {
        self.signatures.get(&id).map_or(&[], Vec::as_slice)
    }

    fn overlaps_subject(&self, (first, last): (usize, usize)) -> bool {
        self.subject
            .iter()
            .any(|s| first <= s.last && last >= s.first)
    }

    /// 90th-percentile gallery score of tracks seen alongside the subject
    /// (known not to be them): this video's "teammates look this alike".
    fn negative_baseline(&self) -> Option<f32> {
        let scores = self
            .spans
            .iter()
            .filter(|(id, &span)| !self.used.contains(id) && self.overlaps_subject(span))
            .filter(|(id, _)| !self.samples(**id).is_empty())
            .map(|(id, _)| score(self.samples(*id), &self.gallery))
            .collect();
        percentile_90(scores)
    }

    fn accept(&mut self, id: u64) {
        let (first, last) = self.spans[&id];
        self.subject.push(SubjectSpan {
            track_id: id,
            first,
            last,
        });
        self.used.insert(id);
        let samples = self.samples(id).to_vec();
        self.gallery.extend(samples);
        let excess = self.gallery.len().saturating_sub(self.cfg.gallery_max_size);
        // Keep the pick's own embedding (index 0); drop the oldest after it.
        if excess > 0 && self.gallery.len() > 1 {
            self.gallery
                .drain(1..1 + excess.min(self.gallery.len() - 1));
        }
    }

    /// Looks for the subject's re-entry after (`forward`) or before the
    /// time `anchor_ms`, sliding the window until something qualifies or
    /// the video runs out. Returns the accepted track.
    fn search(&mut self, anchor_ms: u64, forward: bool) -> Option<u64> {
        self.search_from(anchor_ms, forward, false)
    }

    /// [`search`](Self::search); with `inclusive`, a forward search also
    /// considers tracks starting exactly at `anchor_ms` (a segment's
    /// first frame, which may be timestamp 0).
    fn search_from(
        &mut self,
        mut anchor_ms: u64,
        forward: bool,
        mut inclusive: bool,
    ) -> Option<u64> {
        let (video_start, video_end) = (self.ts(0), self.ts(self.frames.len() - 1));
        let w = self.cfg.reentry_window_ms.max(1);
        loop {
            if (forward && anchor_ms >= video_end) || (!forward && anchor_ms <= video_start) {
                return None;
            }
            let (lo, hi) = if forward {
                (anchor_ms, anchor_ms.saturating_add(w))
            } else {
                (anchor_ms.saturating_sub(w), anchor_ms)
            };
            let mut cands: Vec<(u64, f32)> = self
                .spans
                .iter()
                .filter(|(id, _)| !self.used.contains(id))
                .filter(|(_, &span)| !self.overlaps_subject(span))
                .filter(|(_, &(first, last))| {
                    let t = if forward {
                        self.ts(first)
                    } else {
                        self.ts(last)
                    };
                    if forward {
                        (t > lo || (inclusive && t == lo)) && t <= hi
                    } else {
                        t >= lo && t < hi
                    }
                })
                .filter(|(id, _)| self.samples(**id).len() >= self.cfg.reentry_min_samples)
                .map(|(id, _)| (*id, score(self.samples(*id), &self.gallery)))
                .collect();
            cands.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

            if let Some(&(best, best_score)) = cands.first() {
                let runner_up = cands.get(1).map(|c| c.1);
                let baseline = self.negative_baseline().or(self.external_baseline);
                let m = self.cfg.reentry_margin;
                let ok = best_score >= self.cfg.reentry_threshold
                    && baseline.is_none_or(|b| best_score >= b + m)
                    && runner_up.is_none_or(|r| best_score >= r + m);
                tracing::debug!(
                    best,
                    best_score,
                    ?runner_up,
                    ?baseline,
                    accepted = ok,
                    forward,
                    "re-entry decision"
                );
                if ok {
                    return Some(best);
                }
                self.rejected_windows += 1;
            }
            inclusive = false;
            anchor_ms = if forward { hi } else { lo };
        }
    }
}

impl<'a> Chainer<'a> {
    fn new(
        frames: &'a [FrameRecord],
        signatures: &'a HashMap<u64, Vec<Embedding>>,
        cfg: &'a ChainConfig,
        gallery: Vec<Embedding>,
        external_baseline: Option<f32>,
    ) -> Self {
        Self {
            frames,
            signatures,
            external_baseline,
            cfg,
            spans: spans_of(frames),
            gallery,
            subject: Vec::new(),
            used: HashSet::new(),
            reentries: 0,
            rejected_windows: 0,
        }
    }

    /// Accepts `seed` and chains forward and backward from it.
    fn chain_from(&mut self, seed: u64) {
        let seed_span = self.spans[&seed];
        self.accept(seed);
        let mut cur = seed_span;
        while let Some(next) = self.search(self.ts(cur.1), true) {
            self.accept(next);
            self.reentries += 1;
            cur = self.spans[&next];
        }
        let mut cur = seed_span;
        while let Some(prev) = self.search(self.ts(cur.0), false) {
            self.accept(prev);
            self.reentries += 1;
            cur = self.spans[&prev];
        }
    }

    /// For a segment with no seed: the first track, searching forward from
    /// the segment's start, that qualifies as the subject by appearance.
    fn find_first(&mut self) -> Option<u64> {
        if self.frames.is_empty() {
            return None;
        }
        self.search_from(self.ts(0), true, true)
    }
}

/// Resolves the subject over `log`, starting from `seed_track` (the
/// track the user's pick seeded) and the pick's embedding, chaining
/// forward and backward through re-entries. The whole log is one match.
pub fn resolve(
    log: &TrackLog,
    seed_track: u64,
    pick_embedding: &Embedding,
    cfg: &ChainConfig,
) -> SubjectTimeline {
    let all = [(0, log.frames.len().saturating_sub(1))];
    if log.frames.is_empty() {
        return SubjectTimeline::default();
    }
    resolve_segmented(log, &all, seed_track, pick_embedding, cfg)
}

/// [`resolve`] over match segments, given as inclusive frame-index ranges
/// into `log` (sorted, non-overlapping; frames outside every range are
/// never the subject). Chaining never crosses a segment boundary. The
/// seed's segment chains from the seed. Every other segment has no seed,
/// so the subject is found there by appearance alone: the first track
/// that qualifies under the same relative rule, judged against the
/// gallery and teammate baseline built so far (the seed segment's
/// baseline until the new segment has its own). If none qualifies, the
/// subject is absent for that match (a bye, or not selected).
pub fn resolve_segmented(
    log: &TrackLog,
    segments: &[(usize, usize)],
    seed_track: u64,
    pick_embedding: &Embedding,
    cfg: &ChainConfig,
) -> SubjectTimeline {
    let all_spans = log.spans();
    let Some(&(seed_first, _)) = all_spans.get(&seed_track) else {
        return SubjectTimeline::default();
    };
    let Some(seed_seg) = segments
        .iter()
        .position(|&(a, b)| (a..=b).contains(&seed_first))
    else {
        return SubjectTimeline::default();
    };

    let mut timeline = SubjectTimeline::default();
    let absorb = |c: Chainer<'_>, offset: usize, timeline: &mut SubjectTimeline| {
        timeline.spans.extend(c.subject.iter().map(|s| SubjectSpan {
            track_id: s.track_id,
            first: s.first + offset,
            last: s.last + offset,
        }));
        timeline.reentries += c.reentries;
        timeline.rejected_windows += c.rejected_windows;
        let baseline = c.negative_baseline().or(c.external_baseline);
        (c.gallery, baseline)
    };

    // The seed's own segment first: it defines the gallery and baseline.
    let (a, b) = segments[seed_seg];
    let mut c = Chainer::new(
        &log.frames[a..=b],
        &log.signatures,
        cfg,
        vec![pick_embedding.clone()],
        None,
    );
    c.chain_from(seed_track);
    let (mut gallery, mut baseline) = absorb(c, a, &mut timeline);

    for (i, &(a, b)) in segments.iter().enumerate() {
        if i == seed_seg || b < a || b >= log.frames.len() {
            continue;
        }
        let mut c = Chainer::new(
            &log.frames[a..=b],
            &log.signatures,
            cfg,
            gallery.clone(),
            baseline,
        );
        if let Some(first) = c.find_first() {
            c.chain_from(first);
            c.reentries += 1;
        }
        (gallery, baseline) = absorb(c, a, &mut timeline);
    }

    timeline.spans.sort_by_key(|s| s.first);
    timeline
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit embedding pointing mostly along `axis`, with `mix` of a
    /// shared "same kit" direction (axis 0). Cosine between two players =
    /// roughly mix² (teammates look alike); a player with itself = 1.
    fn emb(axis: usize, mix: f32) -> Embedding {
        let mut v = vec![0.0; 16];
        v[0] = mix;
        v[axis] = (1.0 - mix * mix).sqrt();
        Embedding::from_raw(v)
    }

    fn rect(x: f32) -> Rect {
        Rect {
            x,
            y: 0.0,
            width: 20.0,
            height: 60.0,
        }
    }

    /// Builds a log at 1 frame/s from `(track_id, first_sec, last_sec,
    /// signature)` entries; each track gets 3 signature samples.
    fn log(tracks: &[(u64, usize, usize, Embedding)], seconds: usize) -> TrackLog {
        let mut log = TrackLog::default();
        for s in 0..seconds {
            let mut f = FrameRecord {
                timestamp_ms: s as u64 * 1000,
                frame_number: s as u64,
                tracks: Vec::new(),
                detections: 0,
            };
            for (id, first, last, _) in tracks {
                if (*first..=*last).contains(&s) {
                    f.tracks.push((*id, rect(*id as f32 * 30.0)));
                }
            }
            log.frames.push(f);
        }
        for (id, _, _, e) in tracks {
            log.signatures.insert(*id, vec![e.clone(); 3]);
        }
        log
    }

    fn ids(t: &SubjectTimeline) -> Vec<u64> {
        t.spans.iter().map(|s| s.track_id).collect()
    }

    const KIT: f32 = 0.75; // teammates' shared-appearance component

    #[test]
    fn follows_the_seed_track_without_needing_appearance() {
        let me = emb(1, KIT);
        let l = log(&[(1, 0, 29, me.clone()), (2, 0, 29, emb(2, KIT))], 30);
        let t = resolve(&l, 1, &me, &ChainConfig::default());
        assert_eq!(ids(&t), vec![1]);
        assert_eq!(t.reentries, 0);
        assert!((t.coverage(&l) - 1.0).abs() < 1e-9);
        assert_eq!(t.subject_at(&l, 5).unwrap().0, 1);
    }

    #[test]
    fn bridges_a_gap_to_the_returning_subject_not_a_teammate() {
        let me = emb(1, KIT);
        // Subject is track 1 (0-9 s), lost, returns as track 5 at 14 s.
        // Teammate 3 was on screen with the subject (a known negative);
        // teammate 4 appears in the same window as the re-entry.
        let l = log(
            &[
                (1, 0, 9, me.clone()),
                (3, 0, 29, emb(3, KIT)),
                (4, 12, 29, emb(4, KIT)),
                (5, 14, 29, me.clone()),
            ],
            30,
        );
        let t = resolve(&l, 1, &me, &ChainConfig::default());
        assert_eq!(ids(&t), vec![1, 5]);
        assert_eq!(t.reentries, 1);
        assert!(t.subject_at(&l, 11).is_none(), "absent during the gap");
        assert_eq!(t.subject_at(&l, 20).unwrap().0, 5);
    }

    #[test]
    fn ambiguous_candidates_leave_the_subject_absent() {
        let me = emb(1, KIT);
        // Two candidates that look equally like the subject: refuse to guess.
        let half = Embedding::from_raw(
            me.0.iter()
                .zip(&emb(6, KIT).0)
                .map(|(a, b)| a + b)
                .collect(),
        );
        let l = log(
            &[
                (1, 0, 9, me.clone()),
                (5, 12, 29, half.clone()),
                (6, 13, 29, half.clone()),
            ],
            30,
        );
        let t = resolve(&l, 1, &me, &ChainConfig::default());
        assert_eq!(ids(&t), vec![1]);
        assert!(t.rejected_windows >= 1);
    }

    #[test]
    fn a_lone_teammate_is_rejected_by_the_video_baseline() {
        let me = emb(1, KIT);
        // Only one candidate after the loss, and it's a teammate: it scores
        // like the known negative (track 3) does, so no lead → absent.
        let l = log(
            &[
                (1, 0, 9, me.clone()),
                (3, 0, 9, emb(3, KIT)),
                (4, 12, 29, emb(4, KIT)),
            ],
            30,
        );
        let t = resolve(&l, 1, &me, &ChainConfig::default());
        assert_eq!(ids(&t), vec![1]);
    }

    #[test]
    fn chains_backward_from_a_mid_video_pick() {
        let me = emb(1, KIT);
        // Picked at 20 s (track 7); the subject was earlier track 2 (0-12 s).
        let l = log(
            &[
                (2, 0, 12, me.clone()),
                (3, 0, 29, emb(3, KIT)),
                (7, 16, 29, me.clone()),
            ],
            30,
        );
        let t = resolve(&l, 7, &me, &ChainConfig::default());
        assert_eq!(ids(&t), vec![2, 7], "ordered by time");
    }

    #[test]
    fn window_slides_over_long_absences() {
        let me = emb(1, KIT);
        // Subbed off at 5 s, back on at 50 s: five empty windows later.
        let l = log(&[(1, 0, 5, me.clone()), (9, 50, 59, me.clone())], 60);
        let t = resolve(&l, 1, &me, &ChainConfig::default());
        assert_eq!(ids(&t), vec![1, 9]);
    }

    #[test]
    fn candidates_need_enough_samples() {
        let me = emb(1, KIT);
        let mut l = log(&[(1, 0, 9, me.clone()), (5, 12, 29, me.clone())], 30);
        l.signatures.insert(5, vec![me.clone()]); // one glimpse only
        let t = resolve(&l, 1, &me, &ChainConfig::default());
        assert_eq!(ids(&t), vec![1]);
    }

    #[test]
    fn a_clearly_different_person_fails_the_floor() {
        let me = emb(1, 0.0);
        let l = log(&[(1, 0, 9, me.clone()), (5, 12, 29, emb(5, 0.0))], 30);
        let t = resolve(&l, 1, &me, &ChainConfig::default());
        assert_eq!(ids(&t), vec![1]);
    }

    #[test]
    fn segments_stop_chaining_at_the_boundary() {
        let me = emb(1, KIT);
        // Match 1 = 0–29 s, match 2 = 30–59 s. The subject's track 1 ends
        // at 20 s; track 5 (looks just like them) starts at 32 s, inside
        // the re-entry window but across the boundary. Unsegmented, that's
        // a re-entry; segmented, track 5 is only accepted as match 2's own
        // first appearance, not by bridging.
        let l = log(
            &[
                (1, 0, 20, me.clone()),
                (3, 0, 59, emb(3, KIT)),
                (5, 32, 59, me.clone()),
            ],
            60,
        );
        let t = resolve_segmented(&l, &[(0, 29), (30, 59)], 1, &me, &ChainConfig::default());
        assert_eq!(ids(&t), vec![1, 5]);
        // A track straddling the boundary is split: its match-2 part is
        // judged separately.
        assert!(t.subject_at(&l, 25).is_none());
    }

    #[test]
    fn a_segment_without_the_subject_is_not_an_error() {
        let me = emb(1, KIT);
        // Match 2 has only teammates: nobody clears the seed match's
        // teammate baseline, so the subject is absent there.
        let l = log(
            &[
                (1, 0, 29, me.clone()),
                (3, 0, 29, emb(3, KIT)),
                (6, 30, 59, emb(6, KIT)),
                (7, 30, 59, emb(7, KIT)),
            ],
            60,
        );
        let t = resolve_segmented(&l, &[(0, 29), (30, 59)], 1, &me, &ChainConfig::default());
        assert_eq!(ids(&t), vec![1]);
        assert!(t.subject_at(&l, 45).is_none());
    }

    #[test]
    fn subject_is_found_in_an_earlier_segment_too() {
        let me = emb(1, KIT);
        // Picked in match 2 (track 8); the subject also played in match 1
        // (track 2), alongside teammate 3.
        let l = log(
            &[
                (2, 0, 29, me.clone()),
                (3, 0, 29, emb(3, KIT)),
                (8, 30, 59, me.clone()),
                (9, 30, 59, emb(9, KIT)),
            ],
            60,
        );
        let t = resolve_segmented(&l, &[(0, 29), (30, 59)], 8, &me, &ChainConfig::default());
        assert_eq!(ids(&t), vec![2, 8]);
    }

    #[test]
    fn frames_outside_every_segment_are_never_the_subject() {
        let me = emb(1, KIT);
        let l = log(&[(1, 0, 59, me.clone())], 60);
        let t = resolve_segmented(&l, &[(0, 19), (40, 59)], 1, &me, &ChainConfig::default());
        assert!(t.subject_at(&l, 10).is_some());
        assert!(t.subject_at(&l, 30).is_none(), "between matches");
    }

    #[test]
    fn unknown_seed_gives_an_empty_timeline() {
        let me = emb(1, KIT);
        let l = log(&[(1, 0, 9, me.clone())], 10);
        let t = resolve(&l, 42, &me, &ChainConfig::default());
        assert!(t.spans.is_empty());
        assert_eq!(t.coverage(&l), 0.0);
    }

    #[test]
    fn sampler_respects_interval_cap_and_min_height() {
        let mut s = SignatureSampler::new(1000, 2, 40.0);
        let small = Rect {
            height: 20.0,
            ..rect(0.0)
        };
        let tracks = [(1, rect(0.0)), (2, small)];
        assert_eq!(s.due(0, &tracks).len(), 1, "small box skipped");
        assert!(s.due(500, &tracks).is_empty(), "too soon");
        assert_eq!(s.due(1000, &tracks).len(), 1);
        assert!(s.due(5000, &tracks).is_empty(), "capped at 2 samples");
    }

    #[test]
    fn gallery_is_capped_but_keeps_the_pick() {
        let me = emb(1, KIT);
        let mut l = log(&[(1, 0, 9, emb(2, KIT))], 10);
        l.signatures.insert(1, vec![emb(2, KIT); 10]);
        let cfg = ChainConfig {
            gallery_max_size: 4,
            ..ChainConfig::default()
        };
        let mut c = Chainer::new(&l.frames, &l.signatures, &cfg, vec![me.clone()], None);
        c.accept(1);
        assert_eq!(c.gallery.len(), 4);
        assert_eq!(c.gallery[0], me);
    }
}
