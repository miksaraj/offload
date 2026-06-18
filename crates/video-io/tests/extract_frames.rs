//! Phase 1a acceptance test: `FrameExtractor` iterates real decoded frames.
//!
//! Extracts every 30th frame from a small synthetic fixture clip and dumps
//! each as a PNG for visual verification (the fixture is `testsrc`, so each
//! dumped frame should show a moving test pattern with a visible timestamp).

use std::path::PathBuf;

use video_io::FrameExtractor;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/testsrc.mp4")
}

#[test]
fn extracts_every_30th_frame_as_png() {
    let extractor = FrameExtractor::new(fixture_path()).expect("failed to open fixture clip");

    let out_dir = std::env::temp_dir().join("offload_test_frames");
    std::fs::create_dir_all(&out_dir).expect("failed to create output dir");

    let mut dumped = 0;
    let mut last_timestamp_ms = None;

    for frame in extractor.filter(|f| f.frame_number % 30 == 0) {
        assert_eq!(frame.width, 320);
        assert_eq!(frame.height, 240);
        assert_eq!(frame.pixels.len(), 320 * 240 * 3);
        if let Some(prev) = last_timestamp_ms {
            assert!(
                frame.timestamp_ms >= prev,
                "timestamps must be non-decreasing"
            );
        }
        last_timestamp_ms = Some(frame.timestamp_ms);

        let image = image::RgbImage::from_raw(frame.width, frame.height, frame.pixels)
            .expect("pixel buffer did not match declared dimensions");
        let out_path = out_dir.join(format!("frame_{:06}.png", frame.frame_number));
        image.save(&out_path).expect("failed to save PNG");
        dumped += 1;
    }

    assert_eq!(dumped, 4, "fixture has 120 frames, expected 4 at stride 30");
    eprintln!("dumped {dumped} frames to {}", out_dir.display());
}
