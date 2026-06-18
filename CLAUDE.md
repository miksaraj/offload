# CLAUDE.md

Operational notes for working on Offload. For what the project does and
how it's designed, read [README.md](./README.md), [ARCHITECTURE.md](./ARCHITECTURE.md),
and [SPEC.md](./SPEC.md) first — this file only covers things those
docs don't: current build state, conventions, and gotchas hit while
building it.

## Before every push / PR

Before pushing or opening a PR, reflect on the session's work and
update this file and any `.claude/skills/*` if they're now stale:

- Did the **build-phase state** below move on (a phase landed, a stub
  became real)? Update "Current state."
- Did a **deferred dependency** get added, or should the next one be?
  Update "Deliberately deferred dependencies."
- Did you hit a **new gotcha** — a confusing compiler error, a footgun
  in a crate API, a non-obvious fix? Add it, with the fix, to
  "Gotchas." Only add things you actually hit, not generic advice.
- Did a **convention** change or get established (new error-handling
  pattern, new crate dependency rule)? Update "Conventions."
- Did the CLI surface, build command, or test command change in a way
  that makes `.claude/skills/run-offload/SKILL.md` or `smoke.sh`
  inaccurate? Update and re-run the skill to verify it still passes
  before committing it.

Skip silently if nothing changed — don't pad these docs with no-op
edits, and don't write speculative entries for work you haven't done.

## Current state

Phase 0 (project skeleton) is complete: Cargo workspace with all seven
crates (`pipeline-core`, `video-io`, `detector`, `tracker`, `reid`,
`classifier`, `compiler`, plus the `offload` binary), the `clap` CLI
skeleton, `tracing` logging, the `justfile`, `config/offload.default.toml`,
and `models/download.sh`. Every stage crate exposes the types from
ARCHITECTURE.md §5 but its actual logic is a stub returning a
"not yet implemented" error — see SPEC.md §8 (Build Phases) for what's
next (2a/2b: detection, 3a/3b: tracking, etc.).

Phase 1a (frame extraction) is complete: `video-io`'s `FrameExtractor`
(`crates/video-io/src/lib.rs`) wraps `ffmpeg-next`, opening a video and
implementing `Iterator<Item = Frame>` — it yields every decoded frame
(RGB24, original resolution) with `timestamp_ms`/`frame_number` derived
from the stream's `best_effort_timestamp`; sampling (stride, target fps,
resolution normalisation) is left to the caller/later phases, not done
inside the extractor. `ClipWriter` (Phase 1b) is still a stub.
`crates/video-io/tests/extract_frames.rs` decodes a checked-in synthetic
fixture (`tests/fixtures/testsrc.mp4`, generated via `ffmpeg -f lavfi -i
testsrc=...`) and dumps every 30th frame as a PNG for visual
verification.

To verify the project still builds and the CLI still works after a
change, use the `/run-offload` skill (`.claude/skills/run-offload/`)
rather than re-deriving build/run commands from scratch.

## Deliberately deferred dependencies

`ffmpeg-next` landed in `video-io`'s `Cargo.toml` in Phase 1a (see
"Current state"). It links against system FFmpeg dev libraries
(`libavformat`, `libavcodec`, `libavutil`, `libswscale`,
`libswresample` — installed via the `apt-get` line in
`.claude/skills/run-offload/SKILL.md`'s "Prerequisites", and now also
in `.github/workflows/ci.yml`). `compiler` will need it too for Phase
7's output assembly, but hasn't been wired up yet.

`ort` (ONNX Runtime) is **not yet** in any `Cargo.toml`, even though
ARCHITECTURE.md's dependency map lists it for `detector`/`reid`. It's a
sys-binding crate that links against the ONNX Runtime shared library at
build time, which isn't installed in a bare container — adding it
before it's needed would break `cargo build` for anyone without that
lib. Add it when implementing Phase 2a/4a, and at that point also add
the real prerequisite to `.claude/skills/run-offload/SKILL.md` and CI.

Similarly, `reqwest`/`tokio`/`serde_json` are already wired into
`classifier`'s `Cargo.toml` (pure-Rust, no system deps, safe to add
early) but unused until Phase 5's Ollama client lands.

## CI

`.github/workflows/ci.yml` runs on every push to `main` and every PR:
`cargo fmt --all --check`, `cargo clippy --workspace --all-targets --
-D warnings`, `cargo build --workspace`, `cargo test --workspace`, in
that order, on `ubuntu-latest` with `Swatinem/rust-cache` for
dependency caching. All four checks were run locally against the
current scaffold before this workflow was added — fmt/clippy are
clean with no warnings.

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
- **`ffmpeg-next` decode loop must drive on `Error::Other { errno:
  ffmpeg::error::EAGAIN }`, not on packets running out.** The
  decoder is a push/pull state machine: `receive_frame` returns that
  `EAGAIN` variant to mean "send more input first," and only
  `decoder.send_eof()` (once, after the packet stream is exhausted)
  makes it start returning buffered frames followed by `Error::Eof`.
  Looping packet-reads first and frame-reads second (rather than
  reacting to `EAGAIN`) drops the last GOP of buffered frames. See
  `FrameExtractor::{next, advance}` in `crates/video-io/src/lib.rs`.
- **A decoded video frame's row data isn't tightly packed.**
  `Video::data(0)` returns the whole plane including row padding;
  use `Video::stride(0)` as the per-row byte offset and slice out
  only `width * 3` bytes per row when copying to a `Frame`'s flat
  `Vec<u8>` — copying `width * height * 3` bytes straight from
  `data(0)` silently shifts/garbles every row after the first.
- **Clippy's `wrong_self_convention` fires on any `fn to_*(&mut self,
  ...)`** (it expects `to_*` methods to take `&self`, like
  `to_string`). Name frame-conversion helpers something else (e.g.
  `build_frame`) instead of `to_frame`.

## Conventions

- Crate `Cargo.toml`s pull shared versions from
  `[workspace.dependencies]` in the root `Cargo.toml` via
  `foo.workspace = true` — don't pin versions per-crate.
- Each stage crate's `lib.rs` defines its own `Error` enum (via
  `thiserror`) and a crate-local `Result<T>` alias, matching
  ARCHITECTURE.md's "no domain logic in pipeline-core" split:
  `pipeline-core::PipelineError` only wraps orchestration failures,
  not stage internals.
