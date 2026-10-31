//! End-to-end: real YOLOv8n detections fed through ByteTrack across a
//! simulated camera pan (a sliding crop window over the ultralytics
//! `bus.jpg` sample), so the tracker sees real detector jitter and
//! confidence noise rather than synthetic boxes.

use std::path::{Path, PathBuf};

use detector::{Detector, DetectorConfig};
use image::imageops;
use tracker::{TrackState, Tracker, TrackerConfig};
use video_io::Frame;

fn repo(p: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(p)
}

#[test]
fn ids_stay_stable_across_a_panning_shot() {
    let model = repo("models/yolov8n.onnx");
    let sample = repo("models/samples/bus.jpg");
    for p in [&model, &sample] {
        assert!(
            p.exists(),
            "{} missing — run models/download.sh first",
            p.display()
        );
    }
    let img = image::open(&sample).unwrap().to_rgb8(); // 810x1080

    let tracker_cfg = TrackerConfig::default();
    let mut detector = Detector::new(
        &model,
        DetectorConfig {
            // Emit down to the tracker's low threshold, as the pipeline does.
            confidence_threshold: tracker_cfg.low_confidence_threshold,
            ..DetectorConfig::default()
        },
    )
    .unwrap();
    let mut tracker = Tracker::new(tracker_cfg);

    // 40 frames: a 700x1000 window panning 2 px/frame right and 1 px/frame
    // down. Two people are fully in view throughout; a third, cut off at
    // the left edge, slides out of view as the window pans; a fourth (at
    // x≈671 in the full photo) enters from the right edge partway through.
    let mut solid_ids: Vec<u64> = Vec::new();
    let mut edge_id = None;
    let mut edge_kept_alive_on_low_confidence = false;
    let mut all_ids = std::collections::BTreeSet::new();
    for f in 0..40u32 {
        let view = imageops::crop_imm(&img, 2 * f, f, 700, 1000).to_image();
        let frame = Frame {
            timestamp_ms: u64::from(f) * 125,
            frame_number: u64::from(f),
            pixels: view.into_raw(),
            width: 700,
            height: 1000,
        };
        let tracks = tracker.update(detector.detect(&frame).unwrap()).unwrap();
        if std::env::var("TRACE_TRACKS").is_ok() {
            for t in &tracks {
                eprintln!(
                    "f{f} id{} {:?} x={:.0} w={:.0} conf={:.2}",
                    t.track_id, t.state, t.bbox.x, t.bbox.width, t.confidence
                );
            }
        }
        let tracked: Vec<_> = tracks
            .iter()
            .filter(|t| t.state == TrackState::Tracked)
            .collect();
        for t in &tracked {
            if all_ids.insert(t.track_id) && f > 0 {
                // Any ID born after the first frame must be someone
                // entering at the right edge, not a re-ID'd existing player.
                assert!(
                    t.bbox.x + t.bbox.width > 650.0,
                    "frame {f}: new id {} away from the entry edge: {t:?}",
                    t.track_id
                );
            }
        }
        if f == 0 {
            solid_ids = tracked
                .iter()
                .filter(|t| t.confidence > 0.8)
                .map(|t| t.track_id)
                .collect();
            assert_eq!(solid_ids.len(), 2, "two clearly visible people: {tracks:?}");
            edge_id = tracked.iter().find(|t| t.bbox.x < 10.0).map(|t| t.track_id);
            assert!(
                edge_id.is_some(),
                "edge person tracked on frame 1: {tracks:?}"
            );
        } else {
            for id in &solid_ids {
                assert!(
                    tracked.iter().any(|t| t.track_id == *id),
                    "frame {f}: lost id {id}; {tracks:?}"
                );
            }
        }
        // ByteTrack's second association on real detector output: the edge
        // person's score decays as they leave the frame, and their track
        // should stay `Tracked` on boxes below the high threshold.
        if tracked.iter().any(|t| {
            Some(t.track_id) == edge_id && t.confidence < tracker.config().high_confidence_threshold
        }) {
            edge_kept_alive_on_low_confidence = true;
        }
    }
    assert!(edge_kept_alive_on_low_confidence);
    // No ID churn: the three people present from the start plus the one
    // entering on the right — four people, four IDs.
    assert_eq!(all_ids.len(), 4, "ids {all_ids:?}");
}
