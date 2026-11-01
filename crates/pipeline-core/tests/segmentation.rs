//! Phase 4e end to end with a known answer: an 18 s video of the
//! `bus.jpg` still with the whole frame blanked to grey from 6 s to 12 s
//! (no players: a "break between matches"), so it holds two matches.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

use pipeline_core::segmentation::ManualSegment;
use pipeline_core::{Config, Pipeline};

fn repo(p: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(p)
}

fn two_match_video(name: &str) -> PathBuf {
    let sample = repo("models/samples/bus.jpg");
    assert!(sample.exists(), "run models/download.sh first");
    let out = std::env::temp_dir().join(format!("offload_test_two_matches_{name}.mp4"));
    let status = Command::new("ffmpeg")
        .args(["-loglevel", "error", "-y", "-loop", "1", "-i"])
        .arg(&sample)
        .args([
            "-t",
            "18",
            "-r",
            "10",
            "-vf",
            "drawbox=x=0:y=0:w=iw:h=ih:color=gray:t=fill:enable='between(t,6,12)'",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&out)
        .status()
        .expect("ffmpeg CLI available");
    assert!(status.success());
    out
}

fn config() -> Config {
    let mut c = Config::default();
    c.models.detector = repo("models/yolov8n.onnx").display().to_string();
    c.models.reid = repo("models/osnet_x1_0.onnx").display().to_string();
    c.video.detection_fps = 4;
    c.reid.candidate_window_secs = 4;
    c.reid.candidate_frame_count = 1;
    c.reid.confirmation_scan_secs = 4;
    c
}

fn options(name: &str) -> reid::InteractionOptions {
    reid::InteractionOptions {
        image_path: std::env::temp_dir().join(format!("offload_test_seg_select_{name}.jpg")),
        confirm_path: std::env::temp_dir().join(format!("offload_test_seg_confirm_{name}.jpg")),
        open_viewer: false,
    }
}

#[test]
fn auto_detected_matches_keep_identity_separate_and_refind_the_subject() {
    let video = two_match_video("auto");
    let mut c = config();
    c.segmentation.enabled = true;
    c.segmentation.min_gap_ms = 4_000;
    c.segmentation.min_segment_duration_ms = 3_000;
    // Accept the proposed segments, pick #1 (the man in the light coat:
    // this flow offers the scan's *tracked* boxes, and the partly visible
    // person at the edge scores 0.42, below the 0.5 needed to start a
    // track), no descriptor, confirm.
    let mut input = Cursor::new("\n1\n\ny\n");
    let mut out = Vec::new();
    let r = Pipeline::new(c)
        .follow_subject(&video, &options("auto"), &mut input, &mut out)
        .expect("stage 4 succeeds");
    let transcript = String::from_utf8(out).unwrap();
    assert!(transcript.contains("Found 2 matches"), "{transcript}");

    // Two segments, split inside the 6–12 s break.
    assert_eq!(r.segments.len(), 2, "{:?}", r.segments);
    let boundary = r.segments[1].start_ms;
    assert!(
        (6_000..=12_000).contains(&boundary),
        "boundary at {boundary}"
    );

    // The tracker sees the same man on both sides of the break, but the
    // subject must come from two separate decisions: the seed in match 1,
    // and an appearance match in match 2.
    let ids: Vec<u64> = r.timeline.spans.iter().map(|s| s.track_id).collect();
    assert_eq!(ids.len(), 2, "{:?}", r.timeline);
    for (i, f) in r.log.frames.iter().enumerate() {
        let s = r.timeline.subject_at(&r.log, i);
        let ts = f.timestamp_ms;
        if (1_000..=5_000).contains(&ts) || (13_000..=17_000).contains(&ts) {
            let (_, b) = s.unwrap_or_else(|| panic!("subject missing at {ts} ms"));
            assert!(
                b.x > 30.0 && b.x < 80.0 && b.width > 150.0,
                "wrong person at {ts}: {b:?}"
            );
        }
        if (7_000..=11_000).contains(&ts) {
            assert!(s.is_none(), "subject during the break, at {ts} ms");
        }
    }
}

#[test]
fn manual_segments_target_identification_at_the_second_match() {
    let video = two_match_video("manual");
    let mut c = config();
    c.segmentation.segments = vec![
        ManualSegment {
            start: "0:00".into(),
            end: "0:06".into(),
            label: Some("Pool A".into()),
        },
        ManualSegment {
            start: "0:12".into(),
            end: "0:18".into(),
            label: Some("Pool B".into()),
        },
    ];
    c.reid.candidate_window_start_secs = Some(13);
    let mut input = Cursor::new("2\n\ny\n");
    let r = Pipeline::new(c)
        .follow_subject(&video, &options("manual"), &mut input, &mut Vec::new())
        .expect("stage 4 succeeds");
    assert_eq!(r.segments.len(), 2);
    assert_eq!(r.segments[1].label.as_deref(), Some("Pool B"));
    assert!(
        r.selection.frame.timestamp_ms >= 12_000,
        "identified in match 2: {} ms",
        r.selection.frame.timestamp_ms
    );
    // Found in match 1 too, by appearance, from a match-2 pick.
    let first = r.timeline.spans.first().expect("subject found");
    assert!(
        r.log.frames[first.first].timestamp_ms < 6_000,
        "{:?}",
        r.timeline
    );
}
