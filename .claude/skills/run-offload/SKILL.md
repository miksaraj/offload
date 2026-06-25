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

Phase 1a wired `ffmpeg-next` into `video-io`, which links against system
FFmpeg libraries at build time. Install the dev libs (and the `ffmpeg`
CLI, used to generate/regenerate the test fixture clip) before building:

```bash
sudo apt-get install -y libavutil-dev libavformat-dev libavfilter-dev \
  libavdevice-dev libswscale-dev libswresample-dev libavcodec-dev \
  pkg-config ffmpeg
```

Phase 2a wired `ort` (ONNX Runtime) into `detector`. It's built with the
`load-dynamic` feature (the default `download-binaries` feature fetches a
prebuilt library from `cdn.pyke.io`, which this project's egress policy
blocks) so the ONNX Runtime shared library is loaded at runtime via
`ORT_DYLIB_PATH` instead of linked at build time. Fetch it from
`microsoft/onnxruntime`'s GitHub releases and point the env var at it
before building/running/testing anything that touches `detector`:

```bash
curl -sSL -o /tmp/onnxruntime.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-linux-x64-1.24.4.tgz
tar -xzf /tmp/onnxruntime.tgz -C /tmp
export ORT_DYLIB_PATH=/tmp/onnxruntime-linux-x64-1.24.4/lib/libonnxruntime.so
```

The `detector` crate's tests also need the actual model weights —
run `models/download.sh` once to fetch `models/yolov8n.onnx`.

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

`crates/offload/tests/cli_parses.rs` asserts the CLI parses each
subcommand and rejects `run` without `--input`.
`crates/video-io/tests/extract_frames.rs` decodes the checked-in
fixture clip (`crates/video-io/tests/fixtures/testsrc.mp4`, a synthetic
`ffmpeg testsrc` pattern) via `FrameExtractor`, dumps every 30th frame
as a PNG to `$TMPDIR/offload_test_frames/`, and asserts dimensions,
pixel buffer size, and non-decreasing timestamps. The same file also
has an `#[ignore]`d test, `extracts_every_30th_frame_from_real_clip`,
for ad hoc verification against real footage that isn't checked into
the repo (personal/copyrighted video, no Git LFS configured) — run it
with `OFFLOAD_SAMPLE_CLIP=/path/to/clip.mp4 cargo test -p video-io --
--ignored --nocapture` and inspect the PNGs it dumps to
`$TMPDIR/offload_real_clip_frames/`.
`crates/video-io/tests/write_clips.rs` exercises `ClipWriter` against
the same fixture: writes two padded, non-adjacent windows at a
downscaled resolution and configured bitrate, then re-decodes the
output with `FrameExtractor` to assert dimensions and non-decreasing
timestamps; a second test asserts an empty clip list errors. `video-io`
also has a `#[cfg(test)]` unit test module covering `ClipWriter`'s
internal range-padding/clamping/merging logic directly.

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
- **`ort` v2.0.0-rc.12 fails to compile with `default-features =
  false` unless `api-24` is explicitly re-added.** Without it you get
  `no field SessionOptionsAppendExecutionProvider_VitisAI on type
  &'static OrtApi` from `ort`'s `src/ep/vitis.rs` — that module is
  gated on `feature = "load-dynamic"` alone (not the separate `vitis`
  EP feature), so enabling `load-dynamic` without `api-24` compiles a
  call to a struct field that only exists when `api-24` is on. See the
  workspace `Cargo.toml`'s `ort` dependency.
- **`ort::session::Session::run` takes `&mut self`.** `Detector::new`'s
  warm-up call therefore needs `session` to be `mut`, and the
  `SessionOutputs` it returns borrows `session` — `drop` it explicitly
  before moving `session` into the `Detector` struct, or the compiler
  rejects the move with `E0505`. See `Detector::new` in
  `crates/detector/src/lib.rs`.
- **CI (`.github/workflows/ci.yml`) runs `fmt --check`, `clippy -D
  warnings`, `build`, and `test`** on every push to `main` and every
  PR. Run this skill's smoke script (and `cargo clippy --workspace
  --all-targets -- -D warnings`, `cargo fmt --all --check`) locally
  before pushing so CI doesn't fail on something checkable upfront.
  CI's `ubuntu-latest` runner installs the same FFmpeg dev libs from
  "Prerequisites" via an explicit step before the build (they aren't
  preinstalled on the runner image).

## Troubleshooting

- **`cargo build` is slow / hangs on a fresh container**: the first
  build fetches the full dependency graph (`tokio`, `reqwest`,
  `clap`, etc.) from crates.io — this needs outbound network access to
  `index.crates.io` and `static.crates.io`. Subsequent builds are
  fast (~1s once `target/` is warm).
