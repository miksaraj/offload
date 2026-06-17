---
name: run-offload
description: Build, run, and smoke-test the offload CLI binary. Use when asked to build offload, run the CLI, verify the pipeline skeleton still works, or check that a stage's stub is wired into the binary correctly.
---

`offload` is a Rust CLI (Cargo workspace, binary crate `crates/offload`).
There's no server or GUI to drive — exercise it directly via the
built binary. The fastest way to confirm the project still builds and
the CLI still parses/runs correctly after a change is
`.claude/skills/run-offload/smoke.sh`, which builds the workspace and
runs representative invocations of every subcommand, checking exit
codes and output.

## Prerequisites

No system packages are required for the current (Phase 0) scaffold —
`ffmpeg-next` and `ort` (ONNX Runtime) are intentionally not yet wired
in as dependencies, so there are no system FFmpeg/ONNX Runtime libs to
install. When a later phase adds them (video-io needs `libavformat`
etc., detector/reid need the ONNX Runtime shared lib), update this
section with the exact `apt-get` line and re-verify this skill.

## Build

```bash
cargo build --workspace
```

## Run (agent path)

```bash
.claude/skills/run-offload/smoke.sh
```

This builds the workspace, then runs `offload --help`, `--version`,
`run` (with a missing input, expecting the "input video not found"
error and exit 1), `inspect`, `cache --clear`, `models --download`
(all currently stubs, expecting exit 0 with a "not yet implemented"
warning logged), and `run` with no `--input` (expecting a clap usage
error, exit 2) — then runs `cargo test --workspace`. It prints
`ok`/`FAIL` per check and a pass/fail summary line; exits non-zero if
anything failed.

Direct invocation, if you just need one command:

```bash
RUST_LOG=info ./target/debug/offload <run|inspect|cache|models> [args]
```

`RUST_LOG=info` (or `debug`) makes the `tracing` output visible —
without it, only warnings/errors print by default.

## Test

```bash
cargo test --workspace
```

The only test today is `crates/offload/tests/cli_parses.rs`, asserting
the CLI parses each subcommand and rejects `run` without `--input`.

## Gotchas

- **`tracing::info!` with a field literally named `debug` fails to
  compile** with `the trait bound ... tracing::Value is not satisfied`
  — `debug` collides with `tracing::field::debug` during the macro's
  name resolution. Destructure it under another name (e.g.
  `debug: debug_mode`) before logging it. See
  `crates/offload/src/main.rs`'s `Command::Run` arm.
- **Stub stages report success (exit 0), not failure.** `inspect`,
  `cache`, and `models` are unimplemented stubs that log a `WARN` and
  return `Ok(())` — only `run` currently exercises a real error path
  (missing input file → exit 1). Don't assume a clean exit means a
  stage actually did something; check the log output too.
- **No CI workflow exists yet** (no `.github/workflows/`), so there's
  nothing to babysit through GitHub Actions yet beyond local checks.
  Run this skill's smoke script (and `cargo clippy --workspace
  --all-targets`, `cargo fmt --check`) locally before pushing.

## Troubleshooting

- **`cargo build` is slow / hangs on a fresh container**: the first
  build fetches the full dependency graph (`tokio`, `reqwest`,
  `clap`, etc.) from crates.io — this needs outbound network access to
  `index.crates.io` and `static.crates.io`. Subsequent builds are
  fast (~1s once `target/` is warm).
