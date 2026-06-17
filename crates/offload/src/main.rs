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
            tracing::warn!("inspect is not yet implemented");
        }
        Command::Cache { clear, input } => {
            tracing::info!(clear, ?input, "offload cache");
            tracing::warn!("cache management is not yet implemented");
        }
        Command::Models { download, list } => {
            tracing::info!(download, list, "offload models");
            tracing::warn!("model management is not yet implemented");
        }
    }

    Ok(())
}
