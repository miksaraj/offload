use std::path::{Path, PathBuf};

use detector::{draw_detections, Detector, DetectorConfig, PERSON_CLASS_ID};
use video_io::Frame;

fn model_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/yolov8n.onnx")
}

fn require(path: &Path) {
    assert!(
        path.exists(),
        "{} missing — run models/download.sh first",
        path.display()
    );
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

fn load_frame(path: &Path) -> Frame {
    let img = image::open(path).expect("decode sample image").to_rgb8();
    let (width, height) = img.dimensions();
    Frame {
        timestamp_ms: 0,
        frame_number: 0,
        pixels: img.into_raw(),
        width,
        height,
    }
}

#[test]
fn runs_inference_on_a_single_frame_and_reports_raw_tensor_shape() {
    let model_path = model_path();
    require(&model_path);

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
fn detects_nobody_in_a_blank_frame() {
    let model_path = model_path();
    require(&model_path);
    let mut detector = Detector::new(&model_path, DetectorConfig::default()).unwrap();
    let detections = detector.detect(&synthetic_frame(1280, 720)).unwrap();
    assert!(detections.is_empty(), "{detections:?}");
}

/// The standard ultralytics demo photo: four people in front of a bus
/// (one partly cut off at the left edge). YOLOv8n's reference output on it
/// is "4 persons, 1 bus"; only the persons should survive Offload's
/// person-only filter.
#[test]
fn detects_the_four_people_in_the_bus_sample() {
    let model_path = model_path();
    require(&model_path);
    let sample = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/samples/bus.jpg");
    require(&sample);

    let frame = load_frame(&sample);
    let mut detector = Detector::new(&model_path, DetectorConfig::default()).unwrap();
    let detections = detector.detect(&frame).unwrap();
    for d in &detections {
        eprintln!("{:.3} {:?}", d.confidence, d.bbox);
    }

    let out_dir = std::env::temp_dir().join("offload_detector_frames");
    std::fs::create_dir_all(&out_dir).unwrap();
    let out = out_dir.join("bus_annotated.jpg");
    draw_detections(&frame, &detections).save(&out).unwrap();
    eprintln!("annotated frame: {}", out.display());

    assert_eq!(detections.len(), 4, "{detections:?}");
    for d in &detections {
        assert_eq!(d.class_id, PERSON_CLASS_ID);
        assert!(d.confidence >= DetectorConfig::default().confidence_threshold);
        // Boxes are in source pixels (810x1080), inside the frame, and
        // person-shaped (taller than wide).
        assert!(d.bbox.x >= 0.0 && d.bbox.y >= 0.0);
        assert!(d.bbox.x + d.bbox.width <= frame.width as f32 + 0.5);
        assert!(d.bbox.y + d.bbox.height <= frame.height as f32 + 0.5);
        assert!(d.bbox.height > d.bbox.width, "{:?}", d.bbox);
    }
    // Sorted by confidence, and no two survivors overlap past the NMS threshold.
    assert!(detections
        .windows(2)
        .all(|w| w[0].confidence >= w[1].confidence));
    for (i, a) in detections.iter().enumerate() {
        for b in &detections[i + 1..] {
            assert!(a.bbox.iou(&b.bbox) <= DetectorConfig::default().nms_iou_threshold);
        }
    }
}
