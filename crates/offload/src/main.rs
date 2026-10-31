use std::path::Path;

use anyhow::Result;
use clap::Parser;
use offload::cli::{Cli, Command};
use pipeline_core::{Config, Pipeline};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Run {
            input,
            output,
            config,
            debug: debug_mode,
            dry_run,
            no_cache,
        } => {
            tracing::info!(
                ?input,
                ?output,
                ?config,
                debug_mode,
                dry_run,
                no_cache,
                "offload run"
            );
            let config = Config::load(&config).unwrap_or_default();
            let pipeline = Pipeline::new(config);
            pipeline.run(&input, &output)?;
        }
        Command::Inspect {
            input,
            duration,
            output_dir,
            config,
        } => {
            tracing::info!(?input, duration, ?output_dir, ?config, "offload inspect");
            let pipeline = Pipeline::new(Config::load(&config).unwrap_or_default());
            let s = pipeline.inspect(&input, duration, &output_dir)?;
            let mean = if s.frames_processed == 0 {
                0.0
            } else {
                s.total_detections as f64 / s.frames_processed as f64
            };
            println!(
                "Inspected {} frames: {} person detections ({mean:.1}/frame), {} frames with none. \
                 Annotated frames in {}",
                s.frames_processed,
                s.total_detections,
                s.frames_without_detections,
                s.output_dir.display()
            );
            tracing::warn!("tracking is not yet implemented; inspect shows detections only");
        }
        Command::Cache { clear, input } => {
            tracing::info!(clear, ?input, "offload cache");
            match input.as_deref() {
                Some(input) if clear && pipeline_core::is_url(input) => {
                    // `cache` has no --config flag; use the default config path.
                    let config = Config::load(Path::new("offload.toml")).unwrap_or_default();
                    let removed = Pipeline::new(config).clear_cached_source(input)?;
                    if removed.is_empty() {
                        tracing::info!("no downloaded source cached for {input}");
                    }
                    for path in removed {
                        tracing::info!(path = %path.display(), "removed cached source");
                    }
                    tracing::warn!("stage cache management is not yet implemented");
                }
                _ => tracing::warn!("cache management is not yet implemented"),
            }
        }
        Command::Models { download, list } => {
            tracing::info!(download, list, "offload models");
            tracing::warn!("model management is not yet implemented");
        }
    }

    Ok(())
}
