use clap::Parser;
use offload::cli::{Cli, Command};

#[test]
fn parses_run_subcommand() {
    let cli = Cli::try_parse_from(["offload", "run", "--input", "game.mp4"])
        .expect("run subcommand should parse");

    match cli.command {
        Command::Run { input, output, .. } => {
            assert_eq!(input.to_str(), Some("game.mp4"));
            assert_eq!(output.to_str(), Some("highlights.mp4"));
        }
        other => panic!("expected Run, got {other:?}"),
    }
}

#[test]
fn parses_inspect_cache_and_models_subcommands() {
    assert!(Cli::try_parse_from(["offload", "inspect", "--input", "game.mp4"]).is_ok());
    assert!(Cli::try_parse_from(["offload", "cache", "--clear"]).is_ok());
    assert!(Cli::try_parse_from(["offload", "models", "--download"]).is_ok());
}

#[test]
fn rejects_run_without_input() {
    assert!(Cli::try_parse_from(["offload", "run"]).is_err());
}
