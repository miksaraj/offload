//! Stage 0 — source resolution (SPEC.md §2, Phase 1c).
//!
//! Turns whatever the user passed as `--input` into a local video file.
//! A local path is passed through unchanged; an `http(s)` URL is
//! downloaded once via a `yt-dlp` subprocess into the cache's `sources/`
//! directory and reused on every later run.

use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Result, VideoIoError};

/// `[source]` section of `offload.toml`. Only consulted for URL inputs.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SourceConfig {
    /// `yt-dlp` executable; a bare name is looked up on `PATH`.
    pub ytdlp_path: String,
    /// Cap on the downloaded video height.
    pub max_height: u32,
    /// Passed through to `yt-dlp --cookies-from-browser` when set.
    pub cookies_from_browser: Option<String>,
    /// Download directory; defaults to `<cache dir>/sources/`.
    pub download_dir: Option<PathBuf>,
}

impl Default for SourceConfig {
    fn default() -> Self {
        Self {
            ytdlp_path: "yt-dlp".into(),
            max_height: 1080,
            cookies_from_browser: None,
            download_dir: None,
        }
    }
}

impl SourceConfig {
    /// Directory URL downloads live in, given the run's cache directory.
    pub fn download_dir(&self, cache_dir: &Path) -> PathBuf {
        self.download_dir
            .clone()
            .unwrap_or_else(|| cache_dir.join("sources"))
    }
}

/// Where a resolved source came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceOrigin {
    LocalFile,
    Url {
        /// As the user passed it.
        url: String,
        /// e.g. `"youtube"`.
        extractor: String,
        /// Download/cache key, stable across URL forms.
        video_id: String,
        title: Option<String>,
    },
}

/// A local video file every later stage reads, plus where it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSource {
    pub path: PathBuf,
    pub origin: SourceOrigin,
}

/// `source.json`-style sidecar written next to each download once it is
/// complete. Its presence is what marks a download as complete.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSidecar {
    pub url: String,
    pub extractor: String,
    pub video_id: String,
    pub title: Option<String>,
    pub uploader: Option<String>,
    pub duration_secs: Option<f64>,
    pub format_id: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub downloaded_at_unix: u64,
    pub sha256: String,
}

/// `true` if `input` should be treated as a URL source rather than a path.
pub fn is_url(input: &str) -> bool {
    let lower = input.trim_start().to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

/// Extracts the 11-character YouTube video ID from the URL forms Offload
/// supports, without touching the network. Returns `None` for anything
/// that isn't a recognisable single-video YouTube URL (including
/// playlist-only links).
pub fn youtube_video_id(url: &str) -> Option<String> {
    let rest = url.trim().split_once("://").map(|(_, r)| r).unwrap_or(url);
    let (host_port, path_query) = rest.split_once('/').unwrap_or((rest, ""));
    let host = host_port
        .rsplit('@')
        .next()?
        .split(':')
        .next()?
        .to_ascii_lowercase();
    let host = host.strip_prefix("www.").unwrap_or(&host);
    let host = host.strip_prefix("m.").unwrap_or(host);
    let no_fragment = path_query.split('#').next().unwrap_or("");
    let (path, query) = no_fragment.split_once('?').unwrap_or((no_fragment, ""));

    let candidate = match host {
        "youtu.be" => path.split('/').next().map(str::to_string),
        "youtube.com" | "music.youtube.com" | "youtube-nocookie.com" => {
            let mut segs = path.split('/').filter(|s| !s.is_empty());
            match segs.next() {
                Some("watch") => query
                    .split('&')
                    .find_map(|kv| kv.strip_prefix("v="))
                    .map(str::to_string),
                Some("live") | Some("shorts") | Some("embed") | Some("v") => {
                    segs.next().map(str::to_string)
                }
                _ => None,
            }
        }
        _ => None,
    }?;

    let valid = candidate.len() == 11
        && candidate
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
    valid.then_some(candidate)
}

/// Stage 0 entry point: resolve `input` (a path or a URL) to a local file.
pub fn resolve_source(
    input: &str,
    config: &SourceConfig,
    cache_dir: &Path,
) -> Result<ResolvedSource> {
    if !is_url(input) {
        let path = PathBuf::from(input);
        if !path.exists() {
            return Err(VideoIoError::NotFound(input.to_string()));
        }
        return Ok(ResolvedSource {
            path,
            origin: SourceOrigin::LocalFile,
        });
    }

    let dir = config.download_dir(cache_dir);
    if let Some(hit) = find_cached(input, &dir)? {
        tracing::info!(path = %hit.path.display(), "reusing downloaded source; no network needed");
        return Ok(hit);
    }

    let ytdlp = YtDlp::new(config)?;
    let meta = ytdlp.probe(input)?;
    let stem = format!("{}-{}", meta.extractor, meta.video_id);
    let video_path = dir.join(format!("{stem}.mp4"));

    if meta.height.is_some_and(|h| h > config.max_height) {
        tracing::warn!(
            height = meta.height,
            max_height = config.max_height,
            "no stream at or under source.max_height; downloading the smallest available instead"
        );
    }

    fs::create_dir_all(&dir).map_err(|e| io_err(&dir, e))?;
    tracing::info!(url = input, title = ?meta.title, dest = %video_path.display(), "downloading source video");
    ytdlp.download(input, &dir, &stem)?;
    if !video_path.exists() {
        return Err(VideoIoError::Source(format!(
            "yt-dlp reported success but {} was not created",
            video_path.display()
        )));
    }

    let sidecar = SourceSidecar {
        url: input.to_string(),
        extractor: meta.extractor.clone(),
        video_id: meta.video_id.clone(),
        title: meta.title.clone(),
        uploader: meta.uploader,
        duration_secs: meta.duration,
        format_id: meta.format_id,
        width: meta.width,
        height: meta.height,
        downloaded_at_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        sha256: sha256_file(&video_path)?,
    };
    let sidecar_path = sidecar_path_for(&video_path);
    let json = serde_json::to_string_pretty(&sidecar)
        .map_err(|e| VideoIoError::Source(format!("serialising sidecar: {e}")))?;
    fs::write(&sidecar_path, json).map_err(|e| io_err(&sidecar_path, e))?;

    Ok(ResolvedSource {
        path: video_path,
        origin: SourceOrigin::Url {
            url: input.to_string(),
            extractor: meta.extractor,
            video_id: meta.video_id,
            title: meta.title,
        },
    })
}

/// Deletes the cached download (video, sidecar, and any partial
/// download) for a URL input. Returns the paths removed. Local file
/// inputs are never touched.
pub fn remove_cached_source(
    input: &str,
    config: &SourceConfig,
    cache_dir: &Path,
) -> Result<Vec<PathBuf>> {
    if !is_url(input) {
        return Ok(Vec::new());
    }
    let dir = config.download_dir(cache_dir);
    let Some(stem) = cached_stem(input, &dir)? else {
        return Ok(Vec::new());
    };
    let mut removed = Vec::new();
    let Ok(entries) = fs::read_dir(&dir) else {
        return Ok(removed);
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name == stem || name.starts_with(&format!("{stem}.")) {
            fs::remove_file(&path).map_err(|e| io_err(&path, e))?;
            removed.push(path);
        }
    }
    Ok(removed)
}

fn sidecar_path_for(video_path: &Path) -> PathBuf {
    video_path.with_extension("source.json")
}

/// The `<extractor>-<video_id>` stem a URL maps to, if it can be known
/// without the network: from the URL itself for YouTube, otherwise from
/// an existing sidecar recording the same URL.
fn cached_stem(url: &str, dir: &Path) -> Result<Option<String>> {
    if let Some(id) = youtube_video_id(url) {
        return Ok(Some(format!("youtube-{id}")));
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return Ok(None);
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.to_string_lossy().ends_with(".source.json") {
            continue;
        }
        if let Some(sc) = read_sidecar(&path) {
            if sc.url == url {
                return Ok(Some(format!("{}-{}", sc.extractor, sc.video_id)));
            }
        }
    }
    Ok(None)
}

fn find_cached(url: &str, dir: &Path) -> Result<Option<ResolvedSource>> {
    let Some(stem) = cached_stem(url, dir)? else {
        return Ok(None);
    };
    let video_path = dir.join(format!("{stem}.mp4"));
    let Some(sc) = read_sidecar(&sidecar_path_for(&video_path)) else {
        return Ok(None);
    };
    if !video_path.exists() {
        return Ok(None);
    }
    Ok(Some(ResolvedSource {
        path: video_path,
        origin: SourceOrigin::Url {
            url: url.to_string(),
            extractor: sc.extractor,
            video_id: sc.video_id,
            title: sc.title,
        },
    }))
}

fn read_sidecar(path: &Path) -> Option<SourceSidecar> {
    let raw = fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).map_err(|e| io_err(path, e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf).map_err(|e| io_err(path, e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn io_err(path: &Path, e: std::io::Error) -> VideoIoError {
    VideoIoError::Source(format!("{}: {e}", path.display()))
}

/// Metadata from `yt-dlp --dump-single-json` that Stage 0 needs.
#[derive(Debug, Deserialize)]
struct ProbeJson {
    #[serde(rename = "_type")]
    kind: Option<String>,
    id: Option<String>,
    extractor: Option<String>,
    title: Option<String>,
    uploader: Option<String>,
    duration: Option<f64>,
    live_status: Option<String>,
    is_live: Option<bool>,
    format_id: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
}

#[derive(Debug)]
struct ProbeMeta {
    extractor: String,
    video_id: String,
    title: Option<String>,
    uploader: Option<String>,
    duration: Option<f64>,
    format_id: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
}

struct YtDlp<'a> {
    config: &'a SourceConfig,
}

impl<'a> YtDlp<'a> {
    /// Fails with an install hint if the configured `yt-dlp` can't run.
    fn new(config: &'a SourceConfig) -> Result<Self> {
        match Command::new(&config.ytdlp_path)
            .arg("--version")
            .stdin(Stdio::null())
            .output()
        {
            Ok(out) if out.status.success() => {
                tracing::debug!(
                    version = %String::from_utf8_lossy(&out.stdout).trim(),
                    "found yt-dlp"
                );
                Ok(Self { config })
            }
            Ok(_) | Err(_) => Err(VideoIoError::YtDlpNotFound(config.ytdlp_path.clone())),
        }
    }

    fn format_selector(&self) -> String {
        let h = self.config.max_height;
        // Best video+audio at or under the cap; else best muxed at or under
        // the cap; else the smallest available (warned about by the caller).
        format!("bv*[height<={h}]+ba/b[height<={h}]/wv*+ba/w")
    }

    fn base_command(&self) -> Command {
        let mut cmd = Command::new(&self.config.ytdlp_path);
        cmd.stdin(Stdio::null())
            .arg("--no-playlist")
            .arg("--format")
            .arg(self.format_selector());
        if let Some(browser) = &self.config.cookies_from_browser {
            cmd.arg("--cookies-from-browser").arg(browser);
        }
        cmd
    }

    fn probe(&self, url: &str) -> Result<ProbeMeta> {
        let out = self
            .base_command()
            .arg("--dump-single-json")
            .arg("--no-warnings")
            .arg("--")
            .arg(url)
            .output()
            .map_err(|e| VideoIoError::Source(format!("running yt-dlp: {e}")))?;
        if !out.status.success() {
            return Err(classify_failure(&String::from_utf8_lossy(&out.stderr)));
        }
        let json: ProbeJson = serde_json::from_slice(&out.stdout)
            .map_err(|e| VideoIoError::Source(format!("unexpected yt-dlp metadata output: {e}")))?;
        if json.kind.as_deref().is_some_and(|k| k != "video") {
            return Err(VideoIoError::SourcePlaylist(url.to_string()));
        }
        match json.live_status.as_deref() {
            Some(status @ ("is_live" | "is_upcoming" | "post_live")) => {
                return Err(VideoIoError::SourceLive(status.to_string()));
            }
            _ if json.is_live == Some(true) => {
                return Err(VideoIoError::SourceLive("is_live".into()));
            }
            _ => {}
        }
        let video_id = json
            .id
            .ok_or_else(|| VideoIoError::Source("yt-dlp metadata has no video id".to_string()))?;
        let extractor =
            sanitize(json.extractor.as_deref().unwrap_or("generic")).to_ascii_lowercase();
        Ok(ProbeMeta {
            extractor,
            video_id: sanitize(&video_id),
            title: json.title,
            uploader: json.uploader,
            duration: json.duration,
            format_id: json.format_id,
            width: json.width,
            height: json.height,
        })
    }

    /// Downloads to `<dir>/<stem>.mp4`, echoing yt-dlp's progress lines.
    fn download(&self, url: &str, dir: &Path, stem: &str) -> Result<()> {
        let template = dir.join(format!("{stem}.%(ext)s"));
        let mut child = self
            .base_command()
            .arg("--continue")
            .arg("--newline")
            .arg("--merge-output-format")
            .arg("mp4")
            .arg("--output")
            .arg(&template)
            .arg("--")
            .arg(url)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| VideoIoError::Source(format!("running yt-dlp: {e}")))?;

        let stderr = child.stderr.take().expect("stderr piped");
        let stderr_reader = std::thread::spawn(move || {
            let mut collected = String::new();
            for line in BufReader::new(stderr).lines().map_while(|l| l.ok()) {
                if line.starts_with("WARNING") {
                    tracing::warn!(target: "yt-dlp", "{line}");
                }
                collected.push_str(&line);
                collected.push('\n');
            }
            collected
        });

        let stdout = child.stdout.take().expect("stdout piped");
        for line in BufReader::new(stdout).lines().map_while(|l| l.ok()) {
            if line.starts_with("[download]") {
                eprint!("\r\x1b[2K{line}");
            } else {
                tracing::debug!(target: "yt-dlp", "{line}");
            }
        }
        eprintln!();

        let status = child
            .wait()
            .map_err(|e| VideoIoError::Source(format!("waiting for yt-dlp: {e}")))?;
        let stderr_text = stderr_reader.join().unwrap_or_default();
        if status.success() {
            Ok(())
        } else {
            Err(classify_failure(&stderr_text))
        }
    }
}

/// Keep cache file names to a safe character set. Case is preserved:
/// YouTube video IDs are case-sensitive.
fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>()
        .chars()
        .take(64)
        .collect()
}

/// Maps yt-dlp's stderr to the SPEC.md §7 error cases.
fn classify_failure(stderr: &str) -> VideoIoError {
    let lower = stderr.to_ascii_lowercase();
    let detail = stderr
        .lines()
        .filter(|l| l.starts_with("ERROR"))
        .collect::<Vec<_>>()
        .join("\n");
    let detail = if detail.is_empty() {
        stderr.trim().to_string()
    } else {
        detail
    };
    if lower.contains("unsupported url") {
        VideoIoError::SourceUnsupported(detail)
    } else if lower.contains("private video")
        || lower.contains("members-only")
        || lower.contains("join this channel")
        || lower.contains("age-restricted")
        || lower.contains("sign in to confirm")
    {
        VideoIoError::SourceAuthRequired(detail)
    } else if lower.contains("this live event will begin") || lower.contains("premieres in") {
        VideoIoError::SourceLive(detail)
    } else if lower.contains("video unavailable") || lower.contains("has been removed") {
        VideoIoError::SourceUnavailable(detail)
    } else {
        VideoIoError::Source(format!("yt-dlp failed:\n{detail}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_urls() {
        assert!(is_url("https://youtu.be/dQw4w9WgXcQ"));
        assert!(is_url("HTTP://example.com/a.mp4"));
        assert!(!is_url("match.mp4"));
        assert!(!is_url("/videos/https-match.mp4"));
        assert!(!is_url("youtube.com/watch?v=dQw4w9WgXcQ"));
    }

    #[test]
    fn extracts_youtube_ids_from_all_supported_forms() {
        let id = Some("dQw4w9WgXcQ".to_string());
        for url in [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?feature=share&v=dQw4w9WgXcQ&t=95s",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ?t=10",
            "https://www.youtube.com/live/dQw4w9WgXcQ?si=abc",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ#comments",
        ] {
            assert_eq!(youtube_video_id(url), id, "{url}");
        }
    }

    #[test]
    fn rejects_non_video_youtube_urls() {
        for url in [
            "https://www.youtube.com/playlist?list=PL123",
            "https://www.youtube.com/@somechannel",
            "https://www.youtube.com/watch?v=tooshort",
            "https://vimeo.com/123456",
            "https://notyoutube.com/watch?v=dQw4w9WgXcQ",
        ] {
            assert_eq!(youtube_video_id(url), None, "{url}");
        }
    }

    #[test]
    fn maps_ytdlp_errors() {
        assert!(matches!(
            classify_failure(
                "ERROR: [youtube] abc: Private video. Sign in if you've been granted access"
            ),
            VideoIoError::SourceAuthRequired(_)
        ));
        assert!(matches!(
            classify_failure("ERROR: Unsupported URL: https://example.com"),
            VideoIoError::SourceUnsupported(_)
        ));
        assert!(matches!(
            classify_failure("ERROR: [youtube] abc: Video unavailable"),
            VideoIoError::SourceUnavailable(_)
        ));
        assert!(matches!(
            classify_failure("ERROR: [youtube] abc: This live event will begin in 3 hours."),
            VideoIoError::SourceLive(_)
        ));
        assert!(matches!(
            classify_failure("ERROR: unable to download video data: HTTP Error 403"),
            VideoIoError::Source(_)
        ));
    }

    #[test]
    fn local_missing_file_is_not_found() {
        let err = resolve_source(
            "/definitely/not/here.mp4",
            &SourceConfig::default(),
            Path::new("/tmp"),
        )
        .unwrap_err();
        assert!(matches!(err, VideoIoError::NotFound(_)));
    }
}
