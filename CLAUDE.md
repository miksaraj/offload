# CLAUDE.md

Operational notes for working on Offload. For what the project does and
how it's designed, read [README.md](./README.md), [ARCHITECTURE.md](./ARCHITECTURE.md),
and [SPEC.md](./SPEC.md) first — this file only covers things those
docs don't: current build state, conventions, and gotchas hit while
building it.

## Current state

Phase 0 (project skeleton) is complete: Cargo workspace with all seven
crates (`pipeline-core`, `video-io`, `detector`, `tracker`, `reid`,
`classifier`, `compiler`, plus the `offload` binary), the `clap` CLI
skeleton, `tracing` logging, the `justfile`, `config/offload.default.toml`,
and `models/download.sh`. Every stage crate exposes the types from
ARCHITECTURE.md §5 but its actual logic is a stub returning a
"not yet implemented" error — see SPEC.md §8 (Build Phases) for what's
next (1a: frame extraction, 2a/2b: detection, 3a/3b: tracking, etc.).

To verify the project still builds and the CLI still works after a
change, use the `/run-offload` skill (`.claude/skills/run-offload/`)
rather than re-deriving build/run commands from scratch.

## Deliberately deferred dependencies

`ffmpeg-next` and `ort` (ONNX Runtime) are **not yet** in any
`Cargo.toml`, even though ARCHITECTURE.md's dependency map lists them
for `video-io`/`compiler` and `detector`/`reid` respectively. They're
sys-binding crates that link against system FFmpeg / ONNX Runtime
libraries at build time, which aren't installed in a bare container —
adding them before they're needed would break `cargo build` for anyone
without those system libs. Add them when implementing the phase that
actually needs them (1a/7 for `ffmpeg-next`, 2a/4a for `ort`), and at
that point also add the real prerequisite `apt-get` line to
`.claude/skills/run-offload/SKILL.md`.

Similarly, `reqwest`/`tokio`/`serde_json` are already wired into
`classifier`'s `Cargo.toml` (pure-Rust, no system deps, safe to add
early) but unused until Phase 5's Ollama client lands.

## No CI yet

There's no `.github/workflows/` — SPEC.md's Phase 0 deliverable says
"CI green" but that wasn't part of the original ask for this scaffold.
Before relying on GitHub Actions for anything (status checks, branch
protection "require status checks"), a workflow needs to be added
(`cargo build --workspace`, `cargo test --workspace`, `cargo clippy
--workspace --all-targets -- -D warnings`, `cargo fmt --check`).

## Gotchas

- **`tracing` field shorthand collides with a local variable named
  `debug`.** `tracing::info!(debug, ...)` (or `?debug`) fails to
  compile with `the trait bound ... tracing::Value is not satisfied`
  because `debug` resolves to `tracing::field::debug` instead of the
  local binding. Rename the binding (e.g. destructure
  `debug: debug_mode`) before logging it. Hit in
  `crates/offload/src/main.rs`'s `Command::Run` arm — watch for the
  same trap with any other field named after a `tracing::field::*`
  helper (`debug`, `display`).
- **`cargo fmt`/`cargo clippy` will reformat files you just wrote** —
  expected; let it, don't fight it. Run `cargo fmt` before committing.
- Workspace crates default-disable `reqwest`'s native-tls in favor of
  `rustls-tls` (`Cargo.toml` workspace deps) specifically so `cargo
  build` doesn't need system OpenSSL headers. Keep that feature flag
  if you touch the `classifier` crate's HTTP client.

## Conventions

- Crate `Cargo.toml`s pull shared versions from
  `[workspace.dependencies]` in the root `Cargo.toml` via
  `foo.workspace = true` — don't pin versions per-crate.
- Each stage crate's `lib.rs` defines its own `Error` enum (via
  `thiserror`) and a crate-local `Result<T>` alias, matching
  ARCHITECTURE.md's "no domain logic in pipeline-core" split:
  `pipeline-core::PipelineError` only wraps orchestration failures,
  not stage internals.
