//! Orchestration and shared configuration for the Offload pipeline.
//!
//! Contains no domain logic itself: it loads `offload.toml`, instantiates
//! each stage crate in order, and manages the intermediate cache. The
//! `offload` binary crate calls into [`Pipeline`].

use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;
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
    pub cache: CacheConfig,
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
    pub fn run(&self, input: &str, _output: &Path) -> Result<()> {
        let source = self.resolve_source(input)?;
        tracing::info!(path = %source.path.display(), origin = ?source.origin, "source resolved");
        Err(PipelineError::Stage("not yet implemented".into()))
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
