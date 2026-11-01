//! Orchestration and shared configuration for the Offload pipeline.
//!
//! Contains no domain logic itself: it loads `offload.toml`, instantiates
//! each stage crate in order, and manages the intermediate cache. The
//! `offload` binary crate calls into [`Pipeline`].

use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;
pub mod segmentation;
pub mod subject;

use video_io::{ResolvedSource, SourceConfig, VideoIoError};

pub use video_io::is_url;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("input video not found: {0}")]
    InputNotFound(PathBuf),
    #[error("config error: {0}")]
    Config(String),
    #[error("stage error: {0}")]
    Stage(String),
    #[error(transparent)]
    Detector(#[from] detector::DetectorError),
    #[error(transparent)]
    ReId(#[from] reid::ReIdError),
    #[error("writing {0}: {1}")]
    Output(PathBuf, String),
    #[error(transparent)]
    Source(VideoIoError),
}

impl From<VideoIoError> for PipelineError {
    fn from(err: VideoIoError) -> Self {
        match err {
            VideoIoError::NotFound(p) => PipelineError::InputNotFound(PathBuf::from(p)),
            other => PipelineError::Source(other),
        }
    }
}

pub type Result<T> = std::result::Result<T, PipelineError>;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ModelsConfig {
    pub detector: String,
    pub reid: String,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            detector: "models/yolov8n.onnx".into(),
            reid: "models/osnet_x1_0.onnx".into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct OllamaConfig {
    pub endpoint: String,
    pub model: String,
    pub timeout_secs: u64,
    pub required: bool,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:11434".into(),
            model: "llava".into(),
            timeout_secs: 30,
            required: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct VideoConfig {
    pub detection_fps: u32,
    pub working_width: u32,
    pub working_height: u32,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            detection_fps: 8,
            working_width: 1280,
            working_height: 720,
        }
    }
}

/// `[detector]` section. `execution_provider` from SPEC.md §6 isn't read
/// yet: inference always runs on the CPU provider.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DetectorSection {
    pub confidence_threshold: f32,
    pub nms_iou_threshold: f32,
}

impl Default for DetectorSection {
    fn default() -> Self {
        let d = detector::DetectorConfig::default();
        Self {
            confidence_threshold: d.confidence_threshold,
            nms_iou_threshold: d.nms_iou_threshold,
        }
    }
}

impl From<&DetectorSection> for detector::DetectorConfig {
    fn from(s: &DetectorSection) -> Self {
        Self {
            confidence_threshold: s.confidence_threshold,
            nms_iou_threshold: s.nms_iou_threshold,
        }
    }
}

/// `[tracker]` section (SPEC.md Stage 3).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TrackerSection {
    pub high_confidence_threshold: f32,
    pub low_confidence_threshold: f32,
    pub max_lost_frames: u32,
}

impl Default for TrackerSection {
    fn default() -> Self {
        let d = tracker::TrackerConfig::default();
        Self {
            high_confidence_threshold: d.high_confidence_threshold,
            low_confidence_threshold: d.low_confidence_threshold,
            max_lost_frames: d.max_lost_frames,
        }
    }
}

impl From<&TrackerSection> for tracker::TrackerConfig {
    fn from(s: &TrackerSection) -> Self {
        Self {
            high_confidence_threshold: s.high_confidence_threshold,
            low_confidence_threshold: s.low_confidence_threshold,
            max_lost_frames: s.max_lost_frames,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CacheConfig {
    pub enabled: bool,
    pub dir: Option<String>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dir: None,
        }
    }
}

/// Top-level deserialised form of `offload.toml`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub models: ModelsConfig,
    pub ollama: OllamaConfig,
    pub source: SourceConfig,
    pub video: VideoConfig,
    pub detector: DetectorSection,
    pub tracker: TrackerSection,
    pub segmentation: segmentation::SegmentationConfig,
    pub reid: reid::ReIdConfig,
    pub cache: CacheConfig,
}

impl Config {
    /// Detector settings for feeding the tracker. ByteTrack's second
    /// association needs the low-confidence boxes, so the detector must
    /// emit everything down to `tracker.low_confidence_threshold`, not just
    /// the "sure" boxes at `detector.confidence_threshold`.
    pub fn detector_config_for_tracking(&self) -> detector::DetectorConfig {
        detector::DetectorConfig {
            confidence_threshold: self
                .detector
                .confidence_threshold
                .min(self.tracker.low_confidence_threshold),
            nms_iou_threshold: self.detector.nms_iou_threshold,
        }
    }
}

/// What `offload inspect` processed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InspectSummary {
    pub frames_processed: usize,
    /// Detections at or above `detector.confidence_threshold` (the
    /// lower-confidence boxes passed to the tracker aren't counted).
    pub total_detections: usize,
    pub frames_without_detections: usize,
    /// Sum over frames of tracks in the `Tracked` state.
    pub total_tracked: usize,
    /// Distinct track IDs seen. Far above the number of people on screen
    /// means IDs are being dropped and re-created.
    pub unique_track_ids: usize,
    pub output_dir: PathBuf,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| PipelineError::Config(format!("{}: {e}", path.display())))?;
        toml::from_str(&raw).map_err(|e| PipelineError::Config(e.to_string()))
    }
}

/// Orchestrates the full detect -> track -> reid -> classify -> compile pipeline.
pub struct Pipeline {
    #[allow(dead_code)]
    config: Config,
}

impl Pipeline {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// `input` is a local path or an http(s) URL (SPEC.md Stage 0).
    pub fn run(&self, input: &str, _output: &Path, debug: bool) -> Result<()> {
        let source = self.resolve_source(input)?;
        tracing::info!(path = %source.path.display(), origin = ?source.origin, "source resolved");

        let stdin = std::io::stdin();
        let mut stdout = std::io::stdout();
        let result = self.follow_subject(
            &source.path,
            &reid::InteractionOptions::default(),
            &mut stdin.lock(),
            &mut stdout,
        )?;
        let coverage = result.timeline.coverage(&result.log);
        println!(
            "Subject present in {:.0}% of {} sampled frames ({} re-entries accepted, {} \
             ambiguous gaps left absent).",
            coverage * 100.0,
            result.log.frames.len(),
            result.timeline.reentries,
            result.timeline.rejected_windows,
        );
        if result.segments.len() > 1 {
            let ranges = subject::segment_ranges(&result.log, &result.segments);
            for (i, seg) in result.segments.iter().enumerate() {
                let present = ranges.get(i).map_or(0, |&(a, b)| {
                    (a..=b)
                        .filter(|&f| result.timeline.subject_at(&result.log, f).is_some())
                        .count()
                });
                let total = ranges.get(i).map_or(0, |&(a, b)| b - a + 1);
                println!(
                    "  {}: present in {present} of {total} sampled frames",
                    seg.describe(i)
                );
            }
        }
        if coverage < 0.10 {
            tracing::warn!(
                coverage,
                "subject coverage is below 10%; highlight classification will have little to work with"
            );
        }
        if debug {
            let dir = Path::new("debug").join("reid");
            let n = self.write_subject_debug_frames(&source.path, &result, &dir)?;
            println!("Wrote {n} debug frames to {}", dir.display());
        }
        Err(PipelineError::Stage(
            "subject followed; highlight classification (Phase 5) not yet implemented".into(),
        ))
    }

    /// `offload inspect --show-segments`: a detection-only scan of the
    /// whole video, then heuristic segmentation over its detection counts
    /// (regardless of `segmentation.enabled`).
    pub fn show_segments(&self, input: &str) -> Result<Vec<segmentation::MatchSegment>> {
        let source = self.resolve_source(input)?;
        let video_ms = video_io::FrameExtractor::new(&source.path)?.duration_ms();
        let scan = self.scan(&source.path, 0, None, None, None)?;
        let samples: Vec<(u64, usize)> = scan
            .log
            .frames
            .iter()
            .map(|f| (f.timestamp_ms, f.detections))
            .collect();
        Ok(segmentation::detect_segments(
            &samples,
            video_ms,
            &self.config.segmentation,
        ))
    }

    /// Stage 4A candidates over the whole video's identification window
    /// (`reid.candidate_window_start_secs` + `candidate_window_secs`).
    pub fn candidates(&self, source_path: &Path) -> Result<Vec<reid::Candidate>> {
        let cfg = &self.config.reid;
        let start = u64::from(cfg.candidate_window_start_secs.unwrap_or(0)) * 1000;
        self.candidates_in(
            source_path,
            start,
            start + u64::from(cfg.candidate_window_secs) * 1000,
        )
    }

    /// Stage 4A candidates: `reid.candidate_frame_count` frames spread
    /// across `[start_ms, end_ms)` (clipped to the video), with players
    /// detected at `detector.confidence_threshold` (only "sure" boxes are
    /// offered).
    pub fn candidates_in(
        &self,
        source_path: &Path,
        start_ms: u64,
        end_ms: u64,
    ) -> Result<Vec<reid::Candidate>> {
        let cfg = &self.config.reid;
        let mut detector =
            detector::Detector::new(&self.config.models.detector, (&self.config.detector).into())?;
        let frames = video_io::FrameExtractor::new(source_path)?;
        let targets = reid::identify::candidate_timestamps(
            start_ms,
            end_ms.saturating_sub(start_ms),
            frames.duration_ms(),
            cfg.candidate_frame_count,
        );
        let mut candidates = Vec::new();
        let mut next = targets.iter().peekable();
        for frame in frames {
            let Some(&&target) = next.peek() else {
                break;
            };
            if frame.timestamp_ms < target {
                continue;
            }
            while next.peek().is_some_and(|&&t| t <= frame.timestamp_ms) {
                next.next();
            }
            let detections = detector.detect(&frame)?;
            tracing::info!(
                timestamp_ms = frame.timestamp_ms,
                players = detections.len(),
                "identification candidate frame"
            );
            candidates.push(reid::Candidate { frame, detections });
        }
        Ok(candidates)
    }

    /// Stage 4A on its own: candidates, then the interactive pick (see
    /// [`reid::identify`]). `input`/`out` are the terminal (stdin/stdout
    /// from `run`; scripted in tests).
    pub fn identify(
        &self,
        source_path: &Path,
        options: &reid::InteractionOptions,
        input: &mut impl std::io::BufRead,
        out: &mut impl std::io::Write,
    ) -> Result<reid::Selection> {
        let candidates = self.candidates(source_path)?;
        let mut model = reid::ReIdModel::new(&self.config.models.reid)?;
        Ok(reid::identify::identify_interactively(
            &candidates,
            &mut model,
            options,
            input,
            out,
        )?)
    }

    /// `offload inspect`: resolve the source, then run detection and
    /// tracking on the first `duration_secs` of it, sampled at
    /// `video.detection_fps`, writing one annotated JPEG per sampled frame
    /// (track boxes coloured and labelled by ID) to `output_dir`.
    pub fn inspect(
        &self,
        input: &str,
        duration_secs: u32,
        output_dir: &Path,
    ) -> Result<InspectSummary> {
        let source = self.resolve_source(input)?;
        let mut detector = detector::Detector::new(
            &self.config.models.detector,
            self.config.detector_config_for_tracking(),
        )?;
        let mut tracker = tracker::Tracker::new((&self.config.tracker).into());
        let mut seen_ids = std::collections::HashSet::new();
        std::fs::create_dir_all(output_dir)
            .map_err(|e| PipelineError::Output(output_dir.to_path_buf(), e.to_string()))?;

        let end_ms = u64::from(duration_secs) * 1000;
        let interval_ms = 1000.0 / f64::from(self.config.video.detection_fps.max(1));
        let mut next_sample_ms = 0.0_f64;
        let mut summary = InspectSummary {
            output_dir: output_dir.to_path_buf(),
            ..InspectSummary::default()
        };

        for frame in video_io::FrameExtractor::new(&source.path)? {
            if frame.timestamp_ms >= end_ms {
                break;
            }
            if (frame.timestamp_ms as f64) < next_sample_ms {
                continue;
            }
            while next_sample_ms <= frame.timestamp_ms as f64 {
                next_sample_ms += interval_ms;
            }

            let detections = detector.detect(&frame)?;
            let sure = detections
                .iter()
                .filter(|d| d.confidence >= self.config.detector.confidence_threshold)
                .count();
            let tracks = tracker
                .update(detections)
                .map_err(|e| PipelineError::Stage(e.to_string()))?;
            let tracked = tracks
                .iter()
                .filter(|t| t.state == tracker::TrackState::Tracked)
                .count();
            tracing::debug!(
                frame = frame.frame_number,
                timestamp_ms = frame.timestamp_ms,
                detections = sure,
                tracked,
                "inspected frame"
            );
            let path = output_dir.join(format!(
                "frame_{:06}_{:08}ms.jpg",
                frame.frame_number, frame.timestamp_ms
            ));
            let mut img = detector::draw_detections(&frame, &[]);
            tracker::draw_tracks(&mut img, &tracks);
            img.save(&path)
                .map_err(|e| PipelineError::Output(path.clone(), e.to_string()))?;

            summary.frames_processed += 1;
            summary.total_detections += sure;
            if sure == 0 {
                summary.frames_without_detections += 1;
            }
            summary.total_tracked += tracked;
            seen_ids.extend(tracks.iter().map(|t| t.track_id));
        }
        summary.unique_track_ids = seen_ids.len();
        Ok(summary)
    }

    /// Stage 0: turn `--input` into a local file, downloading URL sources
    /// into the cache (or reusing an earlier download).
    pub fn resolve_source(&self, input: &str) -> Result<ResolvedSource> {
        Ok(video_io::resolve_source(
            input,
            &self.config.source,
            &self.cache_root(),
        )?)
    }

    /// Removes the cached download for a URL input; no-op for local
    /// paths. Returns the files removed.
    pub fn clear_cached_source(&self, input: &str) -> Result<Vec<PathBuf>> {
        Ok(video_io::remove_cached_source(
            input,
            &self.config.source,
            &self.cache_root(),
        )?)
    }

    /// Base cache directory (`cache.dir`, default `.offload_cache`).
    pub fn cache_root(&self) -> PathBuf {
        PathBuf::from(
            self.config
                .cache
                .dir
                .clone()
                .unwrap_or_else(|| ".offload_cache".to_string()),
        )
    }

    pub fn cache_dir_for(&self, input: &Path) -> PathBuf {
        self.cache_root()
            .join(input.file_name().unwrap_or_default())
    }
}
