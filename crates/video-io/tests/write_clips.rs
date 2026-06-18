//! Phase 1b acceptance test: `ClipWriter` extracts and concatenates clip
//! segments from a source video into a single output file.

use std::path::PathBuf;

use video_io::{ClipSpec, ClipWriter, ClipWriterConfig, FrameExtractor};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/testsrc.mp4")
}

/// The fixture is `ffmpeg testsrc`, 320x240 @ 30fps, 120 frames (4s).
#[test]
fn writes_concatenated_clips_with_padding_and_resize() {
    let out_path = std::env::temp_dir().join("offload_test_write_clips.mp4");

    // Two manually chosen, non-overlapping windows from the 4s fixture:
    // 0.5-1.0s and 2.5-3.0s, each padded by 200ms.
    let clips = vec![
        ClipSpec {
            start_ms: 500,
            end_ms: 1_000,
            label: Some("first".into()),
        },
        ClipSpec {
            start_ms: 2_500,
            end_ms: 3_000,
            label: Some("second".into()),
        },
    ];

    let config = ClipWriterConfig {
        padding_ms: 200,
        output_width: Some(160),
        output_height: Some(120),
        bitrate_kbps: Some(500),
    };
    let writer = ClipWriter::new(out_path.to_str().unwrap(), config);
    writer
        .write(fixture_path(), &clips)
        .expect("failed to write clips");

    assert!(out_path.exists(), "output file was not created");

    let frames: Vec<_> = FrameExtractor::new(&out_path)
        .expect("failed to open written clip")
        .collect();

    assert!(
        !frames.is_empty(),
        "expected at least one decoded frame in the output"
    );
    for frame in &frames {
        assert_eq!(frame.width, 160);
        assert_eq!(frame.height, 120);
        assert_eq!(frame.pixels.len(), 160 * 120 * 3);
    }

    // Each padded window is 1000ms (600ms clip + 200ms padding either side);
    // two of them concatenated back-to-back is ~2000ms total. Allow slack
    // for encoder GOP rounding.
    let last_ts = frames.last().unwrap().timestamp_ms;
    assert!(
        last_ts < 2_500,
        "output duration {last_ts}ms looks too long for two ~1000ms windows"
    );

    let mut prev = None;
    for frame in &frames {
        if let Some(p) = prev {
            assert!(
                frame.timestamp_ms >= p,
                "output timestamps must be non-decreasing"
            );
        }
        prev = Some(frame.timestamp_ms);
    }

    let _ = std::fs::remove_file(&out_path);
}

#[test]
fn rejects_empty_clip_list() {
    let out_path = std::env::temp_dir().join("offload_test_write_clips_empty.mp4");
    let writer = ClipWriter::new(out_path.to_str().unwrap(), ClipWriterConfig::default());
    let err = writer
        .write(fixture_path(), &[])
        .expect_err("empty clip list should error");
    assert!(err.to_string().contains("no clips specified"));
}
