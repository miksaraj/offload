use std::path::{Path, PathBuf};

use detector::{Detector, DetectorConfig};
use video_io::Frame;

fn model_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/yolov8n.onnx")
}

fn synthetic_frame(width: u32, height: u32) -> Frame {
    Frame {
        timestamp_ms: 0,
        frame_number: 0,
        pixels: vec![128u8; (width * height * 3) as usize],
        width,
        height,
    }
}

#[test]
fn runs_inference_on_a_single_frame_and_reports_raw_tensor_shape() {
    let model_path = model_path();
    assert!(
        model_path.exists(),
        "{} missing — run models/download.sh first",
        model_path.display()
    );

    let mut detector = Detector::new(&model_path, DetectorConfig::default())
        .expect("model should load and warm up");

    // A non-square source resolution exercises the letterbox padding path.
    let frame = synthetic_frame(1280, 720);
    let output = detector
        .run_inference(&frame)
        .expect("inference should run on a real frame");

    // YOLOv8n: 4 box coords + 80 COCO class scores, across 8400 grid cells
    // (80x80 + 40x40 + 20x20 anchor-free detection heads at 640x640 input).
    assert_eq!(output.shape(), &[1, 84, 8400]);
}

#[test]
fn detect_logs_inference_but_postpones_postprocessing_to_phase_2b() {
    let model_path = model_path();
    assert!(
        model_path.exists(),
        "{} missing — run models/download.sh first",
        model_path.display()
    );

    let mut detector = Detector::new(&model_path, DetectorConfig::default())
        .expect("model should load and warm up");
    let frame = synthetic_frame(640, 640);

    let err = detector
        .detect(&frame)
        .expect_err("postprocessing isn't implemented yet");
    assert!(err.to_string().contains("Phase 2b"));
}
