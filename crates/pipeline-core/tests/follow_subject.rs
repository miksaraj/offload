//! Stage 4 end to end with a known answer: a video of the `bus.jpg`
//! still in which the picked man is covered for 5 s, longer than the
//! tracker keeps a lost track, so he comes back as a *new* track ID.
//! Identity chaining must bridge that gap from real OSNet signatures,
//! and must not hand the identity to anyone else.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

use pipeline_core::{Config, Pipeline};

fn repo(p: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(p)
}

/// 12 s at 10 fps; the man in the light coat (x≈50–245, y≈398–905 in the
/// 810x1080 photo) is painted over in grey from 3 s to 8 s.
fn occluded_video(name: &str) -> PathBuf {
    let sample = repo("models/samples/bus.jpg");
    assert!(sample.exists(), "run models/download.sh first");
    let out = std::env::temp_dir().join(format!("offload_test_bus_occluded_{name}.mp4"));
    let status = Command::new("ffmpeg")
        .args(["-loglevel", "error", "-y", "-loop", "1", "-i"])
        .arg(&sample)
        .args([
            "-t",
            "12",
            "-r",
            "10",
            "-vf",
            "drawbox=x=40:y=380:w=215:h=540:color=gray:t=fill:enable='between(t,3,8)'",
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
    // Keep the test quick: 4 fps sampling, a 2 s identification window at
    // the start (before the occlusion), confirmation over 5 s.
    c.video.detection_fps = 4;
    c.reid.candidate_window_secs = 2;
    c.reid.candidate_frame_count = 1;
    c.reid.confirmation_scan_secs = 5;
    // Lost tracks expire after 2 s, well inside the 5 s occlusion.
    c.tracker.max_lost_frames = 8;
    c
}

#[test]
fn identity_survives_an_occlusion_longer_than_the_tracker_remembers() {
    let video = occluded_video("follow");
    let options = reid::InteractionOptions {
        image_path: std::env::temp_dir().join("offload_test_follow_select.jpg"),
        confirm_path: std::env::temp_dir().join("offload_test_follow_confirm.jpg"),
        open_viewer: false,
    };
    // Pick #2 (left to right: edge person, then the man in the light
    // coat), no descriptor, confirm.
    let mut input = Cursor::new("2\n\ny\n");
    let mut out = Vec::new();
    let result = Pipeline::new(config())
        .follow_subject(&video, &options, &mut input, &mut out)
        .expect("stage 4 succeeds");
    let transcript = String::from_utf8(out).unwrap();
    assert!(transcript.contains("Confirmation"), "{transcript}");
    assert!(options.confirm_path.exists());

    let (log, t) = (&result.log, &result.timeline);
    let ids: Vec<u64> = t.spans.iter().map(|s| s.track_id).collect();
    assert_eq!(ids.len(), 2, "one re-entry across the occlusion: {t:?}");
    assert_eq!(t.reentries, 1);

    // Present before and after the occlusion, absent during it, and always
    // on the same man (his box stays in the same place in a still video).
    for (i, f) in log.frames.iter().enumerate() {
        let s = t.subject_at(log, i);
        let ts = f.timestamp_ms;
        if (500..=2500).contains(&ts) || (9500..=11500).contains(&ts) {
            let (_, r) = s.unwrap_or_else(|| panic!("subject missing at {ts} ms"));
            assert!(
                r.x > 30.0 && r.x < 80.0 && r.width > 150.0,
                "wrong person at {ts} ms: {r:?}"
            );
        }
        if (3500..=7500).contains(&ts) {
            assert!(s.is_none(), "subject reported while covered, at {ts} ms");
        }
    }
    let coverage = t.coverage(log);
    assert!(coverage > 0.45 && coverage < 0.75, "coverage {coverage}");
}

#[test]
fn rejecting_the_confirmation_retries_then_gives_up() {
    let video = occluded_video("reject");
    let mut c = config();
    c.reid.max_identification_attempts = 2;
    let options = reid::InteractionOptions {
        image_path: std::env::temp_dir().join("offload_test_reject_select.jpg"),
        confirm_path: std::env::temp_dir().join("offload_test_reject_confirm.jpg"),
        open_viewer: false,
    };
    let mut input = Cursor::new("2\n\nn\n3\n\nn\n");
    let mut out = Vec::new();
    let err = Pipeline::new(c)
        .follow_subject(&video, &options, &mut input, &mut out)
        .unwrap_err();
    assert!(
        err.to_string().contains("not confirmed after 2 attempts"),
        "{err}"
    );
    let transcript = String::from_utf8(out).unwrap();
    assert_eq!(transcript.matches("Is this correct?").count(), 2);
}
