use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = "offload",
    version,
    about = "Personal rugby highlight reel compiler"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the full pipeline on a match video.
    Run {
        /// Source match video: a local path, or a YouTube URL (downloaded once to the cache).
        #[arg(short, long, value_name = "FILE|URL")]
        input: String,

        /// Output video path.
        #[arg(short, long, default_value = "highlights.mp4")]
        output: PathBuf,

        /// Config file path.
        #[arg(short, long, default_value = "offload.toml")]
        config: PathBuf,

        /// Write annotated frame dumps to ./debug/.
        #[arg(long)]
        debug: bool,

        /// Print moment list without rendering video.
        #[arg(long)]
        dry_run: bool,

        /// Ignore and overwrite any existing cache.
        #[arg(long)]
        no_cache: bool,

        /// Detect breaks between matches in the footage (overrides
        /// segmentation.enabled for this run).
        #[arg(long)]
        auto_segment: bool,
    },

    /// Run detection and tracking on the first N seconds of a video, without ReID or classification.
    Inspect {
        /// Source match video: a local path, or a YouTube URL.
        #[arg(short, long, value_name = "FILE|URL")]
        input: String,

        /// Seconds to inspect.
        #[arg(short, long, default_value_t = 30)]
        duration: u32,

        /// Directory for annotated frames.
        #[arg(short = 'o', long = "output-dir", default_value = "./inspect/")]
        output_dir: PathBuf,

        /// Config file path.
        #[arg(short, long, default_value = "offload.toml")]
        config: PathBuf,

        /// Run heuristic match segmentation over the whole video and print
        /// the proposed boundaries, without processing further.
        #[arg(long)]
        show_segments: bool,
    },

    /// Cache management.
    Cache {
        /// Clear cache (all inputs, or just --input if specified).
        #[arg(long)]
        clear: bool,

        /// Scope cache operation to this input only; for a URL this also deletes the downloaded video.
        #[arg(short, long, value_name = "FILE|URL")]
        input: Option<String>,
    },

    /// Model management.
    Models {
        /// Download default ONNX models to ./models/.
        #[arg(long)]
        download: bool,

        /// List currently installed models and their paths.
        #[arg(long)]
        list: bool,
    },
}
