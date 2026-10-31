//! Real OSNet inference on person crops from the ultralytics `bus.jpg`
//! sample (boxes from real YOLOv8n detections).

use std::path::{Path, PathBuf};

use detector::{Detector, DetectorConfig, Rect};
use reid::{embedding::EMBEDDING_DIM, ReIdModel};
use video_io::Frame;

fn repo(p: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(p)
}

fn require(p: &Path) {
    assert!(
        p.exists(),
        "{} missing — run models/download.sh first",
        p.display()
    );
}

fn jitter(r: &Rect, dx: f32, dy: f32, scale: f32) -> Rect {
    Rect {
        x: r.x + dx * r.width,
        y: r.y + dy * r.height,
        width: r.width * scale,
        height: r.height * scale,
    }
}

#[test]
fn same_person_under_box_jitter_is_closer_than_any_other_person() {
    let (yolo, osnet, sample) = (
        repo("models/yolov8n.onnx"),
        repo("models/osnet_x1_0.onnx"),
        repo("models/samples/bus.jpg"),
    );
    for p in [&yolo, &osnet, &sample] {
        require(p);
    }
    let img = image::open(&sample).unwrap().to_rgb8();
    let (width, height) = img.dimensions();
    let frame = Frame {
        timestamp_ms: 0,
        frame_number: 0,
        pixels: img.into_raw(),
        width,
        height,
    };
    // The three fully visible people (the fourth is cut off at the edge).
    let people: Vec<Rect> = Detector::new(&yolo, DetectorConfig::default())
        .unwrap()
        .detect(&frame)
        .unwrap()
        .into_iter()
        .filter(|d| d.confidence > 0.8)
        .map(|d| d.bbox)
        .collect();
    assert_eq!(people.len(), 3);

    let mut model = ReIdModel::new(&osnet).unwrap();
    let base = model.embed_boxes(&frame, &people).unwrap();
    // Box jitter of the kind successive detections produce: shifted ~5%
    // and rescaled.
    let jittered: Vec<Rect> = people
        .iter()
        .map(|r| jitter(r, 0.05, -0.03, 0.93))
        .collect();
    let moved = model.embed_boxes(&frame, &jittered).unwrap();

    for e in base.iter().chain(&moved) {
        assert_eq!(e.0.len(), EMBEDDING_DIM);
        assert!((e.cosine(e) - 1.0).abs() < 1e-4, "not unit length");
    }
    for i in 0..people.len() {
        let same = base[i].cosine(&moved[i]);
        for (j, other) in base.iter().enumerate() {
            if i != j {
                let diff = base[i].cosine(other);
                eprintln!("person {i}: same {same:.3} vs person {j} {diff:.3}");
                assert!(
                    same > diff + 0.1,
                    "person {i}: same {same:.3} vs {j} {diff:.3}"
                );
            }
        }
    }

    // Batching doesn't change results: one-at-a-time == batched.
    let single = model.embed_boxes(&frame, &people[..1]).unwrap();
    assert!(single[0].cosine(&base[0]) > 0.9999);
}

#[test]
fn empty_batch_is_empty() {
    let osnet = repo("models/osnet_x1_0.onnx");
    require(&osnet);
    let mut model = ReIdModel::new(&osnet).unwrap();
    assert!(model.embed_crops(&[]).unwrap().is_empty());
}
