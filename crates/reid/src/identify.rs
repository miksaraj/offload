//! Interactive subject identification (SPEC.md Sub-phase 4A, Phase 4b).
//!
//! The user identifies themselves by picking a numbered box on a frame
//! from their own footage — no reference photo. This module owns the
//! pieces: choosing candidate timestamps, ordering candidate frames,
//! drawing the numbered selection image, opening the system viewer, and
//! the terminal prompts. I/O goes through `BufRead`/`Write` parameters so
//! the whole flow is testable with scripted input.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use detector::{draw_label, draw_rect, Detection, Rect};
use image::{Rgb, RgbImage};
use video_io::Frame;

use crate::{Embedding, ReIdError, ReIdModel, Result};

/// A frame considered for identification, with its detections.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub frame: Frame,
    pub detections: Vec<Detection>,
}

/// The user's pick.
#[derive(Debug, Clone)]
pub struct Selection {
    /// The candidate frame the user picked from.
    pub frame: Frame,
    /// The chosen player's box, in `frame` pixels.
    pub bbox: Rect,
    pub confidence: f32,
    /// Optional free-text descriptor ("blue 4"); informational only.
    pub descriptor: Option<String>,
    /// OSNet embedding of the chosen crop: the reference appearance.
    pub embedding: Embedding,
}

/// `count` timestamps spread evenly across the identification window
/// `[start_ms, start_ms + window_ms)`, clipped to the video's duration
/// (`video_ms`; 0 = unknown, don't clip). Each sits at the centre of its
/// slice, so none lands on the very first frame (often a title card or
/// a camera still settling).
pub fn candidate_timestamps(start_ms: u64, window_ms: u64, video_ms: u64, count: u32) -> Vec<u64> {
    let mut end = start_ms.saturating_add(window_ms);
    if video_ms > 0 {
        end = end.min(video_ms);
    }
    if count == 0 || end <= start_ms {
        return Vec::new();
    }
    let span = (end - start_ms) as f64;
    (0..count)
        .map(|i| start_ms + (span * (f64::from(i) + 0.5) / f64::from(count)) as u64)
        .collect()
}

/// Candidate indices in presentation order: most detections first (most
/// players visible = easiest to find yourself), ties by time. Candidates
/// with no detections are left out.
pub fn presentation_order(candidates: &[Candidate]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..candidates.len())
        .filter(|&i| !candidates[i].detections.is_empty())
        .collect();
    order.sort_by(|&a, &b| {
        candidates[b]
            .detections
            .len()
            .cmp(&candidates[a].detections.len())
            .then(
                candidates[a]
                    .frame
                    .timestamp_ms
                    .cmp(&candidates[b].frame.timestamp_ms),
            )
    });
    order
}

/// Boxes numbered left to right (then top to bottom), which is easier to
/// scan than detector order. Returns the detections in that order; number
/// `n` on the image is index `n - 1`.
pub fn number_detections(detections: &[Detection]) -> Vec<Detection> {
    let mut d = detections.to_vec();
    d.sort_by(|a, b| {
        a.bbox
            .x
            .total_cmp(&b.bbox.x)
            .then(a.bbox.y.total_cmp(&b.bbox.y))
    });
    d
}

/// Draws every box with a large number label (1-based) at its top-left.
pub fn annotate_numbered(frame: &Frame, numbered: &[Detection]) -> RgbImage {
    let mut img: RgbImage = RgbImage::from_raw(frame.width, frame.height, frame.pixels.clone())
        .expect("Frame pixel buffer length must be width * height * 3");
    let long_side = frame.width.max(frame.height);
    let thickness = (long_side / 400).max(2);
    let scale = (long_side / 250).max(3);
    let yellow = Rgb([255, 220, 0]);
    for d in numbered {
        draw_rect(&mut img, &d.bbox, thickness, yellow);
    }
    // Labels after all boxes so no box line crosses a number.
    for (i, d) in numbered.iter().enumerate() {
        let label_h = 7 * i64::from(scale);
        let y = (d.bbox.y as i64 - label_h).max(0);
        draw_label(
            &mut img,
            d.bbox.x as i64,
            y,
            &(i + 1).to_string(),
            Rgb([0, 0, 0]),
            yellow,
            scale,
        );
    }
    img
}

/// Tries to open `path` in the system image viewer (`open` on macOS,
/// `xdg-open` elsewhere). Returns whether that looked successful; the
/// caller prints the path either way so the user can open it manually.
pub fn open_in_viewer(path: &Path) -> bool {
    let opener = if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    Command::new(opener)
        .arg(path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// What the user typed at the selection prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choice {
    /// 0-based index into the numbered detections.
    Player(usize),
    /// Show the next candidate frame instead.
    NextFrame,
}

fn read_line(input: &mut impl BufRead) -> Result<String> {
    let mut line = String::new();
    let n = input
        .read_line(&mut line)
        .map_err(|e| ReIdError::Identification(e.to_string()))?;
    if n == 0 {
        return Err(ReIdError::Aborted);
    }
    Ok(line.trim().to_string())
}

fn say(out: &mut impl Write, text: &str) -> Result<()> {
    out.write_all(text.as_bytes())
        .and_then(|_| out.flush())
        .map_err(|e| ReIdError::Identification(e.to_string()))
}

/// Asks for a box number in `1..=count` (or `n` for the next frame when
/// `can_skip`), re-prompting until valid. End of input = abort.
pub fn prompt_choice(
    count: usize,
    can_skip: bool,
    input: &mut impl BufRead,
    out: &mut impl Write,
) -> Result<Choice> {
    loop {
        let hint = if can_skip {
            format!("1–{count}, or n for another frame")
        } else {
            format!("1–{count}")
        };
        say(
            out,
            &format!("Enter the number shown on the bounding box that is you ({hint}): "),
        )?;
        let line = read_line(input)?;
        if can_skip && line.eq_ignore_ascii_case("n") {
            return Ok(Choice::NextFrame);
        }
        match line.parse::<usize>() {
            Ok(n) if (1..=count).contains(&n) => return Ok(Choice::Player(n - 1)),
            _ => say(out, &format!("  \"{line}\" isn't one of the options.\n"))?,
        }
    }
}

/// Optional jersey descriptor; empty input = none.
pub fn prompt_descriptor(input: &mut impl BufRead, out: &mut impl Write) -> Result<Option<String>> {
    say(
        out,
        "Enter a jersey descriptor to aid confirmation (e.g. \"blue 4\") [Enter to skip]: ",
    )?;
    let line = read_line(input)?;
    Ok((!line.is_empty()).then_some(line))
}

/// Where to write the selection image and whether to try a viewer.
#[derive(Debug, Clone)]
pub struct InteractionOptions {
    pub image_path: PathBuf,
    /// Where the confirmation frame goes.
    pub confirm_path: PathBuf,
    pub open_viewer: bool,
}

impl Default for InteractionOptions {
    fn default() -> Self {
        Self {
            image_path: std::env::temp_dir().join("offload_select.jpg"),
            confirm_path: std::env::temp_dir().join("offload_confirm.jpg"),
            open_viewer: true,
        }
    }
}

/// Runs the whole interactive identification over `candidates`: show the
/// best candidate frame with numbered boxes, take the user's pick (or
/// move to the next frame on `n`), ask for a descriptor, and embed the
/// chosen crop.
pub fn identify_interactively(
    candidates: &[Candidate],
    model: &mut ReIdModel,
    options: &InteractionOptions,
    input: &mut impl BufRead,
    out: &mut impl Write,
) -> Result<Selection> {
    identify_from(candidates, 0, model, options, input, out)
}

/// [`identify_interactively`], starting at the `start`-th candidate in
/// presentation order (wrapping). A rejected confirmation re-runs
/// identification from the next candidate frame.
pub fn identify_from(
    candidates: &[Candidate],
    start: usize,
    model: &mut ReIdModel,
    options: &InteractionOptions,
    input: &mut impl BufRead,
    out: &mut impl Write,
) -> Result<Selection> {
    let order = presentation_order(candidates);
    if order.is_empty() {
        return Err(ReIdError::NoPlayers);
    }
    let mut pos = start % order.len();
    loop {
        let cand = &candidates[order[pos]];
        let numbered = number_detections(&cand.detections);
        annotate_numbered(&cand.frame, &numbered)
            .save(&options.image_path)
            .map_err(|e| {
                ReIdError::Identification(format!("{}: {e}", options.image_path.display()))
            })?;
        let opened = options.open_viewer && open_in_viewer(&options.image_path);

        say(
            out,
            "\n── Subject Identification ──────────────────────────────\n",
        )?;
        say(
            out,
            &format!(
                "A candidate frame ({}) has been saved to: {}\n",
                format_ms(cand.frame.timestamp_ms),
                options.image_path.display()
            ),
        )?;
        if !opened {
            say(out, "Open it in an image viewer to continue.\n")?;
        }
        say(out, "Each detected player is labelled with a number.\n\n")?;

        let can_skip = order.len() > 1;
        match prompt_choice(numbered.len(), can_skip, input, out)? {
            Choice::NextFrame => {
                pos = (pos + 1) % order.len();
                continue;
            }
            Choice::Player(i) => {
                let picked = &numbered[i];
                let descriptor = prompt_descriptor(input, out)?;
                let embedding = model
                    .embed_boxes(&cand.frame, std::slice::from_ref(&picked.bbox))?
                    .pop()
                    .ok_or_else(|| ReIdError::Inference("no embedding returned".into()))?;
                return Ok(Selection {
                    frame: cand.frame.clone(),
                    bbox: picked.bbox,
                    confidence: picked.confidence,
                    descriptor,
                    embedding,
                });
            }
        }
    }
}

/// The confirmation image: the subject in a thick green box, everyone
/// else in a thin grey one.
pub fn annotate_subject(frame: &Frame, subject: &Rect, others: &[Rect]) -> RgbImage {
    let mut img: RgbImage = RgbImage::from_raw(frame.width, frame.height, frame.pixels.clone())
        .expect("Frame pixel buffer length must be width * height * 3");
    let thickness = (frame.width.max(frame.height) / 300).max(3);
    for r in others {
        draw_rect(&mut img, r, 1, Rgb([170, 170, 170]));
    }
    draw_rect(&mut img, subject, thickness, Rgb([0, 230, 0]));
    img
}

/// `[y/N]` prompt: `y`/`yes` (any case) is yes, anything else (including
/// empty) is no; end of input aborts.
pub fn prompt_confirm(input: &mut impl BufRead, out: &mut impl Write) -> Result<bool> {
    say(out, "Is this correct? [y/N]: ")?;
    let line = read_line(input)?;
    Ok(matches!(line.to_ascii_lowercase().as_str(), "y" | "yes"))
}

/// Shows the confirmation frame and asks whether the green box is the
/// user.
pub fn confirm_interactively(
    frame: &Frame,
    subject: &Rect,
    others: &[Rect],
    image_path: &Path,
    open_viewer: bool,
    input: &mut impl BufRead,
    out: &mut impl Write,
) -> Result<bool> {
    annotate_subject(frame, subject, others)
        .save(image_path)
        .map_err(|e| ReIdError::Identification(format!("{}: {e}", image_path.display())))?;
    let opened = open_viewer && open_in_viewer(image_path);
    say(
        out,
        "\n── Confirmation ────────────────────────────────────────\n",
    )?;
    say(
        out,
        &format!(
            "A confirmation frame ({}) has been saved to: {}\n",
            format_ms(frame.timestamp_ms),
            image_path.display()
        ),
    )?;
    if !opened {
        say(out, "Open it in an image viewer to continue.\n")?;
    }
    say(
        out,
        "The highlighted player (green box) is who Offload will track.\n\n",
    )?;
    prompt_confirm(input, out)
}

/// `m:ss.mmm`, the timestamp format the dry-run table uses.
pub fn format_ms(ms: u64) -> String {
    format!("{}:{:02}.{:03}", ms / 60_000, (ms / 1000) % 60, ms % 1000)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn det(x: f32, y: f32) -> Detection {
        Detection {
            bbox: Rect {
                x,
                y,
                width: 20.0,
                height: 40.0,
            },
            confidence: 0.9,
            class_id: 0,
        }
    }

    fn frame(ts: u64) -> Frame {
        Frame {
            timestamp_ms: ts,
            frame_number: ts,
            pixels: vec![0; 100 * 80 * 3],
            width: 100,
            height: 80,
        }
    }

    #[test]
    fn candidate_timestamps_spread_and_clip() {
        assert_eq!(
            candidate_timestamps(0, 300_000, 0, 3),
            vec![50_000, 150_000, 250_000]
        );
        // Clipped to a 30 s video.
        assert_eq!(
            candidate_timestamps(0, 300_000, 30_000, 3),
            vec![5_000, 15_000, 25_000]
        );
        // Offset start.
        assert_eq!(
            candidate_timestamps(60_000, 60_000, 0, 2),
            vec![75_000, 105_000]
        );
        // Window starting past the end of the video: nothing.
        assert!(candidate_timestamps(60_000, 60_000, 30_000, 3).is_empty());
        assert!(candidate_timestamps(0, 1000, 0, 0).is_empty());
    }

    #[test]
    fn presentation_prefers_most_players_then_earliest() {
        let c = |ts, n: usize| Candidate {
            frame: frame(ts),
            detections: (0..n).map(|i| det(i as f32 * 10.0, 0.0)).collect(),
        };
        let cands = vec![c(0, 2), c(1, 0), c(2, 5), c(3, 5)];
        assert_eq!(presentation_order(&cands), vec![2, 3, 0]);
    }

    #[test]
    fn numbering_runs_left_to_right() {
        let n = number_detections(&[det(50.0, 0.0), det(10.0, 5.0), det(30.0, 0.0)]);
        let xs: Vec<f32> = n.iter().map(|d| d.bbox.x).collect();
        assert_eq!(xs, vec![10.0, 30.0, 50.0]);
    }

    #[test]
    fn prompt_reprompts_until_valid_and_supports_next_frame() {
        let mut out = Vec::new();
        let mut input = Cursor::new("0\nfoo\n4\n2\n");
        assert_eq!(
            prompt_choice(3, false, &mut input, &mut out).unwrap(),
            Choice::Player(1)
        );
        let text = String::from_utf8(out).unwrap();
        assert_eq!(text.matches("isn't one of the options").count(), 3);

        let mut input = Cursor::new("N\n");
        assert_eq!(
            prompt_choice(3, true, &mut input, &mut Vec::new()).unwrap(),
            Choice::NextFrame
        );
        // `n` isn't accepted when there's no other frame to go to.
        let mut input = Cursor::new("n\n1\n");
        assert_eq!(
            prompt_choice(3, false, &mut input, &mut Vec::new()).unwrap(),
            Choice::Player(0)
        );
    }

    #[test]
    fn end_of_input_aborts() {
        let mut input = Cursor::new("");
        assert!(matches!(
            prompt_choice(3, false, &mut input, &mut Vec::new()),
            Err(ReIdError::Aborted)
        ));
    }

    #[test]
    fn descriptor_is_optional() {
        let mut input = Cursor::new("\n");
        assert_eq!(
            prompt_descriptor(&mut input, &mut Vec::new()).unwrap(),
            None
        );
        let mut input = Cursor::new("  blue 4 \n");
        assert_eq!(
            prompt_descriptor(&mut input, &mut Vec::new()).unwrap(),
            Some("blue 4".into())
        );
    }

    #[test]
    fn annotation_draws_numbered_boxes() {
        let f = frame(0);
        let img = annotate_numbered(&f, &[det(20.0, 30.0), det(60.0, 30.0)]);
        let yellow = Rgb([255, 220, 0]);
        assert_eq!(*img.get_pixel(20, 50), yellow); // box edge
        assert_eq!(*img.get_pixel(20, 29), yellow); // label background above it
        assert_eq!(*img.get_pixel(30, 60), Rgb([0, 0, 0])); // interior untouched
    }

    #[test]
    fn confirm_prompt_defaults_to_no() {
        for (text, want) in [
            ("y\n", true),
            ("YES\n", true),
            ("\n", false),
            ("n\n", false),
            ("maybe\n", false),
        ] {
            let mut input = Cursor::new(text);
            assert_eq!(
                prompt_confirm(&mut input, &mut Vec::new()).unwrap(),
                want,
                "{text:?}"
            );
        }
        assert!(matches!(
            prompt_confirm(&mut Cursor::new(""), &mut Vec::new()),
            Err(ReIdError::Aborted)
        ));
    }

    #[test]
    fn confirmation_image_highlights_the_subject() {
        let f = frame(0);
        let subject = det(20.0, 20.0).bbox;
        let other = det(60.0, 20.0).bbox;
        let img = annotate_subject(&f, &subject, &[other]);
        assert_eq!(*img.get_pixel(20, 40), Rgb([0, 230, 0]));
        assert_eq!(*img.get_pixel(60, 40), Rgb([170, 170, 170]));
    }

    #[test]
    fn formats_timestamps() {
        assert_eq!(format_ms(83_456), "1:23.456");
        assert_eq!(format_ms(0), "0:00.000");
    }
}
