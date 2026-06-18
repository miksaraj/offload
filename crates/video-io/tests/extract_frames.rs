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

/// Ad hoc verification against real footage, since no real clip is checked
/// into the repo (personal/copyrighted video, not committed). Run with:
/// `OFFLOAD_SAMPLE_CLIP=/path/to/clip.mp4 cargo test -p video-io -- --ignored`
#[test]
#[ignore = "requires OFFLOAD_SAMPLE_CLIP env var pointing at a real video file"]
fn extracts_every_30th_frame_from_real_clip() {
    let path = std::env::var("OFFLOAD_SAMPLE_CLIP")
        .expect("set OFFLOAD_SAMPLE_CLIP to a real video file path");
    let extractor = FrameExtractor::new(&path).expect("failed to open clip");

    let out_dir = std::env::temp_dir().join("offload_real_clip_frames");
    std::fs::create_dir_all(&out_dir).expect("failed to create output dir");

    let mut dumped = 0;
    for frame in extractor.filter(|f| f.frame_number % 30 == 0) {
        let image = image::RgbImage::from_raw(frame.width, frame.height, frame.pixels)
            .expect("pixel buffer did not match declared dimensions");
        let out_path = out_dir.join(format!("frame_{:06}.png", frame.frame_number));
        image.save(&out_path).expect("failed to save PNG");
        dumped += 1;
    }

    assert!(dumped > 0, "expected at least one frame from {path}");
    eprintln!("dumped {dumped} frames to {}", out_dir.display());
}
