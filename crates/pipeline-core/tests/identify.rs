//! Stage 4A end to end: candidate frames from a real video, real
//! detections, numbered selection image, scripted terminal input, OSNet
//! embedding of the pick.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

use pipeline_core::{Config, Pipeline, PipelineError};

fn repo(p: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(p)
}

fn config() -> Config {
    let mut c = Config::default();
    c.models.detector = repo("models/yolov8n.onnx").display().to_string();
    c.models.reid = repo("models/osnet_x1_0.onnx").display().to_string();
    c.reid.candidate_window_secs = 2;
    c.reid.candidate_frame_count = 2;
    c
}

fn options(name: &str) -> reid::InteractionOptions {
    reid::InteractionOptions {
        image_path: std::env::temp_dir().join(format!("offload_test_select_{name}.jpg")),
        confirm_path: std::env::temp_dir().join(format!("offload_test_confirm_{name}.jpg")),
        open_viewer: false,
    }
}

/// A 2 s, 10 fps video of the `bus.jpg` still (made with the ffmpeg CLI).
/// One file per test: tests run in parallel.
fn bus_video(name: &str) -> PathBuf {
    let sample = repo("models/samples/bus.jpg");
    assert!(sample.exists(), "run models/download.sh first");
    let out = std::env::temp_dir().join(format!("offload_test_bus_{name}.mp4"));
    let status = Command::new("ffmpeg")
        .args(["-loglevel", "error", "-y", "-loop", "1", "-i"])
        .arg(&sample)
        .args(["-t", "2", "-r", "10", "-pix_fmt", "yuv420p"])
        .arg(&out)
        .status()
        .expect("ffmpeg CLI available");
    assert!(status.success());
    out
}

#[test]
fn user_picks_a_numbered_player_and_gets_a_reference_embedding() {
    let video = bus_video("pick");
    let pipeline = Pipeline::new(config());
    let opts = options("pick");
    // An invalid number first (re-prompt), then player 2, then a descriptor.
    let mut input = Cursor::new("9\n2\nblue 4\n");
    let mut out = Vec::new();
    let sel = pipeline
        .identify(&video, &opts, &mut input, &mut out)
        .expect("identification succeeds");
    let transcript = String::from_utf8(out).unwrap();
    assert!(
        transcript.contains("Subject Identification"),
        "{transcript}"
    );
    assert!(
        transcript.contains("isn't one of the options"),
        "{transcript}"
    );
    assert!(opts.image_path.exists());

    // Boxes are numbered left to right: in bus.jpg that's the partly
    // visible person at the edge (1), then the man in the light coat (2).
    assert!(sel.bbox.x > 30.0 && sel.bbox.x < 80.0, "{:?}", sel.bbox);
    assert_eq!(sel.descriptor.as_deref(), Some("blue 4"));
    assert_eq!(sel.embedding.0.len(), 512);
    // Both candidates show the same still, so the first (earlier) one is used.
    assert!(sel.frame.timestamp_ms < 1000, "{}", sel.frame.timestamp_ms);
}

#[test]
fn next_frame_cycles_candidates() {
    let video = bus_video("next");
    let pipeline = Pipeline::new(config());
    let mut input = Cursor::new("n\n1\n\n");
    let sel = pipeline
        .identify(&video, &options("next"), &mut input, &mut Vec::new())
        .unwrap();
    assert!(
        sel.frame.timestamp_ms >= 1000,
        "second candidate: {}",
        sel.frame.timestamp_ms
    );
    assert_eq!(sel.descriptor, None);
}

#[test]
fn a_video_without_people_is_a_clear_error() {
    let fixture = repo("crates/video-io/tests/fixtures/testsrc.mp4");
    let err = Pipeline::new(config())
        .identify(
            &fixture,
            &options("none"),
            &mut Cursor::new(""),
            &mut Vec::new(),
        )
        .unwrap_err();
    assert!(
        matches!(err, PipelineError::ReId(reid::ReIdError::NoPlayers)),
        "{err}"
    );
}

#[test]
fn closing_stdin_aborts() {
    let video = bus_video("abort");
    let err = Pipeline::new(config())
        .identify(
            &video,
            &options("abort"),
            &mut Cursor::new(""),
            &mut Vec::new(),
        )
        .unwrap_err();
    assert!(
        matches!(err, PipelineError::ReId(reid::ReIdError::Aborted)),
        "{err}"
    );
}
