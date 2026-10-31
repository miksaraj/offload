//! Stage 0 (source resolution) tests against a fake `yt-dlp`.
//!
//! The real `yt-dlp` needs network access to YouTube, which CI and the
//! dev container don't have. Each test instead writes a small shell
//! script that answers `--version`, `--dump-single-json` (with canned
//! metadata) and the download invocation (by copying the checked-in
//! fixture clip to the `--output` path), and logs every call so tests
//! can assert what was — and wasn't — run.
#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use video_io::source::{resolve_source, SourceConfig, SourceOrigin, SourceSidecar};
use video_io::{remove_cached_source, VideoIoError};

const VIDEO_ID: &str = "dQw4w9WgXcQ";

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/testsrc.mp4")
}

struct FakeYtDlp {
    dir: PathBuf,
    script: PathBuf,
}

impl FakeYtDlp {
    /// `probe` is either `Ok(json)` printed for `--dump-single-json`, or
    /// `Err(stderr)` printed to stderr with exit 1.
    fn new(name: &str, probe: Result<&str, &str>) -> Self {
        let dir =
            std::env::temp_dir().join(format!("offload_resolve_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let (probe_body, probe_exit) = match probe {
            Ok(json) => {
                fs::write(dir.join("probe.json"), json).unwrap();
                (format!("cat '{}'", dir.join("probe.json").display()), 0)
            }
            Err(stderr) => {
                fs::write(dir.join("probe.err"), stderr).unwrap();
                (format!("cat '{}' >&2", dir.join("probe.err").display()), 1)
            }
        };
        let script = dir.join("yt-dlp");
        let body = format!(
            r#"#!/bin/sh
echo "$@" >> '{log}'
case " $* " in
  *" --version "*) echo 2099.01.01; exit 0 ;;
  *" --dump-single-json "*) {probe_body}; exit {probe_exit} ;;
esac
out=""
while [ $# -gt 0 ]; do
  if [ "$1" = "--output" ]; then out="$2"; fi
  shift
done
dest=$(printf '%s' "$out" | sed 's/%(ext)s/mp4/')
echo "[download]  50.0% of 1.00MiB"
echo "[download] 100.0% of 1.00MiB"
cp '{fixture}' "$dest"
"#,
            log = dir.join("calls.log").display(),
            fixture = fixture().display(),
        );
        fs::write(&script, body).unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
        Self { dir, script }
    }

    fn config(&self) -> SourceConfig {
        SourceConfig {
            ytdlp_path: self.script.display().to_string(),
            ..SourceConfig::default()
        }
    }

    fn cache_dir(&self) -> PathBuf {
        self.dir.join("cache")
    }

    fn calls(&self) -> Vec<String> {
        fs::read_to_string(self.dir.join("calls.log"))
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    }
}

fn video_json(live_status: &str, height: u32) -> String {
    format!(
        r#"{{"_type":"video","id":"{VIDEO_ID}","extractor":"youtube","title":"Pool vs Avengers 7s",
            "uploader":"Club Channel","duration":1925.0,"live_status":"{live_status}",
            "format_id":"137+140","width":1920,"height":{height}}}"#
    )
}

#[test]
fn downloads_once_then_reuses_across_url_forms_without_running_ytdlp() {
    let json = video_json("was_live", 1080);
    let fake = FakeYtDlp::new("reuse", Ok(&json));
    let config = fake.config();
    let url = format!("https://www.youtube.com/watch?v={VIDEO_ID}&t=95s");

    let first = resolve_source(&url, &config, &fake.cache_dir()).expect("first resolve");
    let expected = fake
        .cache_dir()
        .join("sources")
        .join(format!("youtube-{VIDEO_ID}.mp4"));
    assert_eq!(first.path, expected);
    assert_eq!(
        fs::read(&first.path).unwrap(),
        fs::read(fixture()).unwrap(),
        "downloaded file should be what yt-dlp wrote"
    );
    match &first.origin {
        SourceOrigin::Url {
            extractor,
            video_id,
            title,
            ..
        } => {
            assert_eq!(extractor, "youtube");
            assert_eq!(video_id, VIDEO_ID);
            assert_eq!(title.as_deref(), Some("Pool vs Avengers 7s"));
        }
        other => panic!("expected Url origin, got {other:?}"),
    }

    let sidecar: SourceSidecar =
        serde_json::from_str(&fs::read_to_string(expected.with_extension("source.json")).unwrap())
            .unwrap();
    assert_eq!(sidecar.url, url);
    assert_eq!(sidecar.height, Some(1080));
    assert_eq!(sidecar.sha256.len(), 64);

    let calls = fake.calls();
    assert_eq!(calls.len(), 3, "version + probe + download: {calls:?}");
    let download = &calls[2];
    for arg in [
        "--no-playlist",
        "--continue",
        "--merge-output-format mp4",
        "bv*[height<=1080]+ba/b[height<=1080]",
    ] {
        assert!(
            download.contains(arg),
            "download call missing {arg}: {download}"
        );
    }
    assert!(!download.contains("--cookies-from-browser"));

    // Different URL form, same video ID: served from the cache, yt-dlp
    // never invoked (so no network).
    let second = resolve_source(
        &format!("https://youtu.be/{VIDEO_ID}"),
        &config,
        &fake.cache_dir(),
    )
    .expect("second resolve");
    assert_eq!(second.path, first.path);
    assert_eq!(fake.calls().len(), 3, "second resolve must not call yt-dlp");

    // cache --clear --input <URL> removes the download and its sidecar.
    let removed = remove_cached_source(&url, &config, &fake.cache_dir()).unwrap();
    assert_eq!(removed.len(), 2, "{removed:?}");
    assert!(!expected.exists());
}

#[test]
fn passes_cookies_and_height_cap_through() {
    let json = video_json("not_live", 720);
    let fake = FakeYtDlp::new("cookies", Ok(&json));
    let config = SourceConfig {
        cookies_from_browser: Some("firefox".into()),
        max_height: 720,
        ..fake.config()
    };
    resolve_source(
        &format!("https://youtu.be/{VIDEO_ID}"),
        &config,
        &fake.cache_dir(),
    )
    .unwrap();
    let calls = fake.calls();
    for call in &calls[1..] {
        assert!(call.contains("--cookies-from-browser firefox"), "{call}");
        assert!(call.contains("height<=720"), "{call}");
    }
}

#[test]
fn rejects_live_and_upcoming_streams_before_downloading() {
    for status in ["is_live", "is_upcoming", "post_live"] {
        let json = video_json(status, 1080);
        let fake = FakeYtDlp::new(&format!("live_{status}"), Ok(&json));
        let err = resolve_source(
            &format!("https://www.youtube.com/live/{VIDEO_ID}"),
            &fake.config(),
            &fake.cache_dir(),
        )
        .unwrap_err();
        assert!(
            matches!(err, VideoIoError::SourceLive(_)),
            "{status}: {err}"
        );
        assert_eq!(fake.calls().len(), 2, "no download attempted for {status}");
    }
}

#[test]
fn rejects_playlists() {
    let fake = FakeYtDlp::new(
        "playlist",
        Ok(r#"{"_type":"playlist","id":"PL123","extractor":"youtube:tab","title":"Season"}"#),
    );
    let err = resolve_source(
        "https://www.youtube.com/playlist?list=PL123",
        &fake.config(),
        &fake.cache_dir(),
    )
    .unwrap_err();
    assert!(matches!(err, VideoIoError::SourcePlaylist(_)), "{err}");
}

#[test]
fn reports_private_videos_with_cookie_hint() {
    let fake = FakeYtDlp::new(
        "private",
        Err("ERROR: [youtube] dQw4w9WgXcQ: Private video. Sign in if you've been granted access to this video\n"),
    );
    let err = resolve_source(
        &format!("https://youtu.be/{VIDEO_ID}"),
        &fake.config(),
        &fake.cache_dir(),
    )
    .unwrap_err();
    assert!(matches!(err, VideoIoError::SourceAuthRequired(_)), "{err}");
    assert!(err.to_string().contains("cookies_from_browser"));
}

#[test]
fn missing_ytdlp_gives_install_hint() {
    let config = SourceConfig {
        ytdlp_path: "/nonexistent/yt-dlp".into(),
        ..SourceConfig::default()
    };
    let err = resolve_source(
        &format!("https://youtu.be/{VIDEO_ID}"),
        &config,
        &std::env::temp_dir().join("offload_resolve_missing"),
    )
    .unwrap_err();
    assert!(matches!(err, VideoIoError::YtDlpNotFound(_)), "{err}");
    assert!(err.to_string().contains("source.ytdlp_path"));
}

#[test]
fn local_paths_pass_through_unchanged() {
    let path = fixture();
    let resolved = resolve_source(
        path.to_str().unwrap(),
        &SourceConfig::default(),
        Path::new("/unused"),
    )
    .unwrap();
    assert_eq!(resolved.path, path);
    assert_eq!(resolved.origin, SourceOrigin::LocalFile);
}
