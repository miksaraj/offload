//! Match segmentation (SPEC.md Stage 3.5, Phase 4e).
//!
//! A [`MatchSegment`] is a time range treated as one self-contained
//! match. By default there's one implicit segment (the whole video).
//! Segments come from a manual list in config, or from a heuristic over
//! per-frame detection counts: a long stretch of near-empty pitch (the
//! broadcast cut away between matches) is a boundary.

use std::io::{BufRead, Write};

use serde::Deserialize;

use crate::{PipelineError, Result};

/// A time range within the source video treated as one match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchSegment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub label: Option<String>,
}

impl MatchSegment {
    pub fn contains(&self, ts: u64) -> bool {
        (self.start_ms..self.end_ms).contains(&ts)
    }

    pub fn describe(&self, index: usize) -> String {
        let label = self
            .label
            .as_deref()
            .map(|l| format!(" — {l}"))
            .unwrap_or_default();
        format!(
            "Match {} ({}–{}){label}",
            index + 1,
            hms(self.start_ms),
            hms(self.end_ms)
        )
    }
}

/// One manual entry in `segmentation.segments`.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ManualSegment {
    pub start: String,
    pub end: String,
    pub label: Option<String>,
}

/// `[segmentation]` section.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SegmentationConfig {
    /// Heuristic auto-detection (ignored if `segments` is non-empty).
    pub enabled: bool,
    pub min_gap_ms: u64,
    pub empty_pitch_threshold: usize,
    pub min_segment_duration_ms: u64,
    pub segments: Vec<ManualSegment>,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_gap_ms: 90_000,
            empty_pitch_threshold: 2,
            min_segment_duration_ms: 180_000,
            segments: Vec::new(),
        }
    }
}

impl SegmentationConfig {
    /// True when segments must be detected from the footage (auto on,
    /// no manual list).
    pub fn auto(&self) -> bool {
        self.enabled && self.segments.is_empty()
    }

    /// The manual list, validated, or the implicit whole-video segment.
    pub fn manual_or_implicit(&self, video_ms: u64) -> Result<Vec<MatchSegment>> {
        if self.segments.is_empty() {
            return Ok(vec![whole(video_ms)]);
        }
        let mut out = self
            .segments
            .iter()
            .map(|m| {
                Ok(MatchSegment {
                    start_ms: parse_hms(&m.start)?,
                    end_ms: parse_hms(&m.end)?,
                    label: m.label.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        validate(&mut out)?;
        Ok(out)
    }
}

/// The implicit segment: the whole video (`u64::MAX` end when the
/// duration is unknown).
pub fn whole(video_ms: u64) -> MatchSegment {
    MatchSegment {
        start_ms: 0,
        end_ms: if video_ms == 0 { u64::MAX } else { video_ms },
        label: None,
    }
}

fn validate(segs: &mut [MatchSegment]) -> Result<()> {
    segs.sort_by_key(|s| s.start_ms);
    for s in segs.iter() {
        if s.end_ms <= s.start_ms {
            return Err(PipelineError::Config(format!(
                "segment {}–{} ends before it starts",
                hms(s.start_ms),
                hms(s.end_ms)
            )));
        }
    }
    for w in segs.windows(2) {
        if w[1].start_ms < w[0].end_ms {
            return Err(PipelineError::Config(format!(
                "segments {}–{} and {}–{} overlap",
                hms(w[0].start_ms),
                hms(w[0].end_ms),
                hms(w[1].start_ms),
                hms(w[1].end_ms)
            )));
        }
    }
    Ok(())
}

/// `"HH:MM:SS"`, `"H:MM:SS"` or `"MM:SS"` → milliseconds.
pub fn parse_hms(s: &str) -> Result<u64> {
    let parts: Vec<&str> = s.trim().split(':').collect();
    let nums: Option<Vec<u64>> = parts.iter().map(|p| p.parse().ok()).collect();
    let bad = || PipelineError::Config(format!("bad time \"{s}\" (expected HH:MM:SS)"));
    let nums = nums.ok_or_else(bad)?;
    let secs = match nums.as_slice() {
        [h, m, s] if *m < 60 && *s < 60 => h * 3600 + m * 60 + s,
        [m, s] if *s < 60 => m * 60 + s,
        _ => return Err(bad()),
    };
    Ok(secs * 1000)
}

/// Milliseconds → `"H:MM:SS"` (or `"M:SS"` under an hour).
pub fn hms(ms: u64) -> String {
    if ms == u64::MAX {
        return "end".into();
    }
    let s = ms / 1000;
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
    } else {
        format!("{}:{:02}", s / 60, s % 60)
    }
}

/// Window (each side) of the rolling median that smooths detection
/// counts, so a brief shot of players during a break, or a momentary
/// detection dropout during play, doesn't split or hide a gap.
const SMOOTHING_HALF_WINDOW_MS: u64 = 2_500;

/// Heuristic auto-detection over `(timestamp_ms, player_count)` samples
/// of a `video_ms`-long video: stretches of at least `min_gap_ms` where
/// the smoothed count is at or below `empty_pitch_threshold` are gaps; a
/// boundary goes at each gap's midpoint. A segment whose *active* span
/// (first to last non-empty sample, i.e. not counting the half-gaps it
/// inherits at either end) is shorter than `min_segment_duration_ms` is
/// dropped as noise (a warm-up shot between matches). Returns the whole
/// video as one segment if nothing qualifies.
pub fn detect_segments(
    samples: &[(u64, usize)],
    video_ms: u64,
    cfg: &SegmentationConfig,
) -> Vec<MatchSegment> {
    let (Some(&(first_ts, _)), Some(&(last_ts, _))) = (samples.first(), samples.last()) else {
        return vec![whole(video_ms)];
    };
    let video_end = video_ms.max(last_ts + 1);

    // Rolling median over ±SMOOTHING_HALF_WINDOW_MS.
    let smoothed: Vec<usize> = samples
        .iter()
        .map(|&(ts, _)| {
            let mut w: Vec<usize> = samples
                .iter()
                .filter(|(t, _)| t.abs_diff(ts) <= SMOOTHING_HALF_WINDOW_MS)
                .map(|&(_, c)| c)
                .collect();
            w.sort_unstable();
            w[w.len() / 2]
        })
        .collect();

    // Gaps: maximal runs of "empty" samples lasting >= min_gap_ms.
    let mut boundaries = Vec::new();
    let mut run_start: Option<u64> = None;
    for (i, &(ts, _)) in samples.iter().enumerate() {
        let empty = smoothed[i] <= cfg.empty_pitch_threshold;
        match (empty, run_start) {
            (true, None) => run_start = Some(ts),
            (false, Some(start)) => {
                if ts - start >= cfg.min_gap_ms {
                    boundaries.push(start + (ts - start) / 2);
                }
                run_start = None;
            }
            _ => {}
        }
    }
    if let Some(start) = run_start {
        if video_end - start >= cfg.min_gap_ms {
            boundaries.push(start + (video_end - start) / 2);
        }
    }

    let mut edges = vec![first_ts];
    edges.extend(boundaries);
    edges.push(video_end);
    let active_span = |start: u64, end: u64| {
        let mut active = samples
            .iter()
            .zip(&smoothed)
            .filter(|((ts, _), &c)| (start..end).contains(ts) && c > cfg.empty_pitch_threshold)
            .map(|((ts, _), _)| *ts);
        match active.next() {
            Some(first) => active.next_back().unwrap_or(first) - first,
            None => 0,
        }
    };
    let segs: Vec<MatchSegment> = edges
        .windows(2)
        .filter(|w| active_span(w[0], w[1]) >= cfg.min_segment_duration_ms)
        .map(|w| MatchSegment {
            start_ms: w[0],
            end_ms: w[1],
            label: None,
        })
        .collect();
    if segs.is_empty() {
        vec![whole(video_end)]
    } else {
        segs
    }
}

/// The segment identification targets: the one containing
/// `reid.candidate_window_start_secs` if set, else the first.
pub fn target_segment(segments: &[MatchSegment], start_ms: Option<u64>) -> Option<usize> {
    match start_ms {
        Some(t) => segments.iter().position(|s| s.contains(t)),
        None => (!segments.is_empty()).then_some(0),
    }
}

fn say(out: &mut impl Write, text: &str) -> Result<()> {
    out.write_all(text.as_bytes())
        .and_then(|_| out.flush())
        .map_err(|e| PipelineError::Stage(e.to_string()))
}

fn read_line(input: &mut impl BufRead) -> Result<String> {
    let mut line = String::new();
    let n = input
        .read_line(&mut line)
        .map_err(|e| PipelineError::Stage(e.to_string()))?;
    if n == 0 {
        return Err(PipelineError::ReId(reid::ReIdError::Aborted));
    }
    Ok(line.trim().to_string())
}

/// Parses `"0:00:00-0:32:10, 0:58:40-1:30:05"` into segments.
pub fn parse_segment_list(s: &str) -> Result<Vec<MatchSegment>> {
    let mut out = s
        .split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| {
            let (a, b) = p.split_once('-').ok_or_else(|| {
                PipelineError::Config(format!("\"{}\" isn't START-END", p.trim()))
            })?;
            Ok(MatchSegment {
                start_ms: parse_hms(a)?,
                end_ms: parse_hms(b)?,
                label: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if out.is_empty() {
        return Err(PipelineError::Config("no segments given".into()));
    }
    validate(&mut out)?;
    Ok(out)
}

/// When auto-detection proposes more than one segment: show them and let
/// the user accept, edit (type a manual list), or drop segmentation.
pub fn confirm_segments(
    proposed: Vec<MatchSegment>,
    video_ms: u64,
    input: &mut impl BufRead,
    out: &mut impl Write,
) -> Result<Vec<MatchSegment>> {
    if proposed.len() <= 1 {
        return Ok(proposed);
    }
    say(
        out,
        "\n── Match Segmentation ──────────────────────────────────\n",
    )?;
    say(
        out,
        &format!(
            "Found {} matches separated by breaks in play:\n",
            proposed.len()
        ),
    )?;
    for (i, s) in proposed.iter().enumerate() {
        say(out, &format!("  {}\n", s.describe(i)))?;
    }
    loop {
        say(
            out,
            "\nUse these? [Y]es / [e]dit / [n]o segmentation (treat the video as one match): ",
        )?;
        match read_line(input)?.to_ascii_lowercase().as_str() {
            "" | "y" | "yes" => return Ok(proposed),
            "n" | "no" => return Ok(vec![whole(video_ms)]),
            "e" | "edit" => loop {
                say(
                    out,
                    "Enter matches as START-END, comma-separated (e.g. 0:00:00-0:32:10, 0:58:40-1:30:05): ",
                )?;
                match parse_segment_list(&read_line(input)?) {
                    Ok(segs) => return Ok(segs),
                    Err(e) => say(out, &format!("  {e}\n"))?,
                }
            },
            other => say(out, &format!("  \"{other}\" isn't one of the options.\n"))?,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn cfg(min_gap_s: u64, min_seg_s: u64) -> SegmentationConfig {
        SegmentationConfig {
            enabled: true,
            min_gap_ms: min_gap_s * 1000,
            empty_pitch_threshold: 2,
            min_segment_duration_ms: min_seg_s * 1000,
            segments: Vec::new(),
        }
    }

    /// 1 sample/s; `counts[i]` players at second i.
    fn samples(counts: &[usize]) -> Vec<(u64, usize)> {
        counts
            .iter()
            .enumerate()
            .map(|(i, &c)| (i as u64 * 1000, c))
            .collect()
    }

    fn spans(segs: &[MatchSegment]) -> Vec<(u64, u64)> {
        segs.iter()
            .map(|s| (s.start_ms / 1000, s.end_ms / 1000))
            .collect()
    }

    #[test]
    fn parses_and_formats_times() {
        assert_eq!(parse_hms("01:30:05").unwrap(), 5_405_000);
        assert_eq!(parse_hms("0:58:40").unwrap(), 3_520_000);
        assert_eq!(parse_hms("12:30").unwrap(), 750_000);
        assert!(parse_hms("1:75:00").is_err());
        assert!(parse_hms("noon").is_err());
        assert_eq!(hms(5_405_000), "1:30:05");
        assert_eq!(hms(750_000), "12:30");
    }

    #[test]
    fn splits_at_the_middle_of_a_long_empty_stretch() {
        // 0–39 s match, 40–59 s empty pitch, 60–99 s match.
        let mut c = vec![12; 40];
        c.extend(vec![0; 20]);
        c.extend(vec![12; 40]);
        let segs = detect_segments(&samples(&c), c.len() as u64 * 1000, &cfg(15, 30));
        assert_eq!(spans(&segs), vec![(0, 50), (50, 100)]);
    }

    #[test]
    fn short_breaks_and_blips_dont_split() {
        // A 10 s injury stoppage (below min_gap) and isolated 1-sample
        // dropouts during play.
        let mut c = vec![12; 40];
        c.extend(vec![1; 10]);
        c.extend(vec![12; 40]);
        c[5] = 0;
        c[70] = 0;
        let segs = detect_segments(&samples(&c), c.len() as u64 * 1000, &cfg(15, 30));
        assert_eq!(segs.len(), 1);
        // And a brief shot of players during a long break doesn't split it.
        let mut c = vec![12; 40];
        c.extend(vec![0; 10]);
        c.extend(vec![15; 2]);
        c.extend(vec![0; 10]);
        c.extend(vec![12; 40]);
        assert_eq!(
            detect_segments(&samples(&c), c.len() as u64 * 1000, &cfg(15, 30)).len(),
            2
        );
    }

    #[test]
    fn short_segments_are_dropped_as_noise() {
        // match, gap, 10 s of warm-up, gap, match.
        let mut c = vec![12; 40];
        c.extend(vec![0; 20]);
        c.extend(vec![10; 10]);
        c.extend(vec![0; 20]);
        c.extend(vec![12; 40]);
        let segs = detect_segments(&samples(&c), c.len() as u64 * 1000, &cfg(15, 30));
        assert_eq!(segs.len(), 2, "{:?}", spans(&segs));
    }

    #[test]
    fn no_gaps_means_one_implicit_segment() {
        let segs = detect_segments(&samples(&[10; 100]), 100_000, &cfg(15, 30));
        assert_eq!(spans(&segs), vec![(0, 100)]);
        assert_eq!(detect_segments(&[], 0, &cfg(15, 30)).len(), 1);
    }

    #[test]
    fn manual_segments_validate() {
        let mut c = SegmentationConfig::default();
        assert_eq!(c.manual_or_implicit(60_000).unwrap(), vec![whole(60_000)]);
        c.segments = vec![
            ManualSegment {
                start: "0:58:40".into(),
                end: "1:30:05".into(),
                label: Some("B".into()),
            },
            ManualSegment {
                start: "0:00:00".into(),
                end: "0:32:10".into(),
                label: None,
            },
        ];
        let segs = c.manual_or_implicit(0).unwrap();
        assert_eq!(segs[0].start_ms, 0, "sorted");
        assert_eq!(segs[1].label.as_deref(), Some("B"));
        c.segments[0].start = "0:10:00".into();
        assert!(c.manual_or_implicit(0).is_err(), "overlap rejected");
    }

    #[test]
    fn target_segment_follows_the_start_time() {
        let segs = parse_segment_list("0:00-10:00, 20:00-30:00").unwrap();
        assert_eq!(target_segment(&segs, None), Some(0));
        assert_eq!(target_segment(&segs, Some(25 * 60_000)), Some(1));
        assert_eq!(target_segment(&segs, Some(15 * 60_000)), None);
    }

    #[test]
    fn confirm_accepts_edits_or_drops() {
        let proposed = parse_segment_list("0:00-10:00, 20:00-30:00").unwrap();
        let run = |text: &str| {
            confirm_segments(
                proposed.clone(),
                1_800_000,
                &mut Cursor::new(text.to_string()),
                &mut Vec::new(),
            )
        };
        assert_eq!(run("\n").unwrap(), proposed);
        assert_eq!(run("n\n").unwrap(), vec![whole(1_800_000)]);
        let edited = run("e\nbogus\n0:00-12:00, 18:00-30:00\n").unwrap();
        assert_eq!(edited[0].end_ms, 720_000);
        assert!(matches!(
            run(""),
            Err(PipelineError::ReId(reid::ReIdError::Aborted))
        ));
        // A single proposed segment needs no prompt.
        let one = vec![whole(1000)];
        assert_eq!(
            confirm_segments(one.clone(), 1000, &mut Cursor::new(""), &mut Vec::new()).unwrap(),
            one
        );
    }
}
