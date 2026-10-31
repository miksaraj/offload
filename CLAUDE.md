# CLAUDE.md

Operational notes for working on Offload. For what the project does and
how it's designed, read [README.md](./README.md), [ARCHITECTURE.md](./ARCHITECTURE.md),
and [SPEC.md](./SPEC.md) first — this file only covers things those
docs don't: current build state, conventions, and gotchas hit while
building it.

## Before every push / PR

Before pushing or opening a PR, reflect on the session's work and
update this file, `README.md`, `SPEC.md`/`ARCHITECTURE.md`, and any
`.claude/skills/*` if they're now stale:

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
- Does **`README.md`** (status badge, roadmap table, CLI reference,
  quick start) still match reality? It's user-facing, so it drifts
  silently — nothing fails CI when it's wrong. Check it every time,
  not just when something feels off.
- Do **`SPEC.md`** and **`ARCHITECTURE.md`** still describe the system
  accurately? They're the design source of truth this file and
  `README.md` build on — if a phase changed the actual shape of a
  type, crate boundary, or pipeline stage from what they describe,
  update them too. They drift less often than `README.md` (design vs.
  status), but check both on any change that touches design, not only
  status.
- **Reflect explicitly: did anything this session involve research,
  trial-and-error, or a non-obvious fix that would otherwise be
  re-derived from scratch next time?** If so, it belongs in a skill or
  in this file *now*, before pushing — not "maybe later." The cost of
  re-discovering the same API quirk, build gotcha, or workflow twice
  is the exact waste this checklist exists to prevent.

Skip silently if nothing changed — don't pad these docs with no-op
edits, and don't write speculative entries for work you haven't done.

As skills and context grow, prefer splitting into separate, focused
Markdown files referenced from a hub file (this one, `README.md`)
rather than letting any single file grow unbounded. Not worth
restructuring preemptively while everything still fits comfortably —
but when a section here gets long enough that finding things in it
gets slow, that's the signal to split it out rather than letting it
keep growing.

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
inside the extractor. `crates/video-io/tests/extract_frames.rs` decodes
a checked-in synthetic fixture (`tests/fixtures/testsrc.mp4`, generated
via `ffmpeg -f lavfi -i testsrc=...`) and dumps every 30th frame as a
PNG for visual verification. It also decoded correctly against a real
~25s rugby clip (706x848, 60fps h264) the user sourced — confirmed via
an `#[ignore]`d test (see `.claude/skills/run-offload/SKILL.md`'s
"Test" section); that clip wasn't committed (personal footage, no Git
LFS configured), so future real-footage checks need a clip supplied
locally via `OFFLOAD_SAMPLE_CLIP`.

Phase 1b (clip assembly) is complete: `video-io`'s `ClipWriter::write`
takes a source path and a `&[ClipSpec]`, pads each clip's start/end by
`ClipWriterConfig::padding_ms` (clamped to the source's duration),
sorts and merges overlapping padded ranges, then does a single forward
decode pass over the source — re-encoding (via libx264) only the
frames that fall in a merged range — and concatenates them into one
output file with continuous, gap-free timestamps (a plain incrementing
frame counter, not wall-clock source timestamps; see Gotchas). Output
resolution (`output_width`/`output_height`, default source resolution)
and bitrate (`bitrate_kbps`, default: CRF 23 instead of a fixed
bitrate) are configurable via `ClipWriterConfig`.
`crates/video-io/tests/write_clips.rs` writes two padded, non-adjacent
windows from the synthetic `testsrc.mp4` fixture at a downscaled
resolution and configured bitrate, then re-decodes the output with
`FrameExtractor` to assert dimensions and non-decreasing timestamps.

To verify the project still builds and the CLI still works after a
change, use the `/run-offload` skill (`.claude/skills/run-offload/`)
rather than re-deriving build/run commands from scratch.

Phase 2a (ONNX Runtime setup & model loading) is complete: `detector`
(`crates/detector/src/lib.rs`) loads a YOLOv8 ONNX model via `ort`,
built with the `load-dynamic` feature so the ONNX Runtime shared
library is `dlopen`ed at runtime from `ORT_DYLIB_PATH` rather than
linked at build time (see "Deliberately deferred dependencies" and
Gotchas). `Detector::new` loads the model and runs a zeroed warm-up
inference so the first real call isn't paying for lazy init.
`preprocess` letterboxes a `Frame` to `640x640` (aspect-preserving
resize + grey padding, not a naive stretch — see SPEC.md), normalises
`u8 [0,255] -> f32 [0,1]`, and reorders `HWC -> NCHW`.
`Detector::run_inference` runs that tensor through the model and
returns the raw output undecoded. `models/download.sh` now actually
fetches `models/yolov8n.onnx` (pinned to the `ultralytics/assets` GitHub
release `v8.4.0`, SHA256-verified) instead of being a no-op stub.

Phase 2b (detection postprocessing) is complete. `decode_output` takes
the `[1, 84, N]` tensor; each anchor's best class must be person (0)
with a score of at least `confidence_threshold`, following ultralytics'
single-label behaviour. YOLOv8 has no objectness row, so the class
score is the confidence. Greedy `nms` at `nms_iou_threshold` runs in
letterbox space, keeping at most `MAX_DETECTIONS` (300). Survivors are
then mapped back through `LetterboxTransform::to_source`, which undoes
the padding and scale and clamps to the frame, so `Detection.bbox` is
in source-frame pixels (top-left `x`/`y`). `Detector::detect` chains
all of this. `draw_detections` draws boxes shaded red (low confidence)
to green (high) for debug frames.

`pipeline-core` gained a `[detector]` config section
(`DetectorSection`: thresholds only; `execution_provider` isn't read
yet, so it's always CPU) and `Pipeline::inspect`, which backs
`offload inspect`: resolve the source (URLs work), sample frames at
`video.detection_fps` by timestamp, detect, and write
`frame_<n>_<ms>ms.jpg` annotated frames. There's no tracking yet, and
frames aren't normalised to `video.working_*`, because the detector
letterboxes any resolution anyway. `run` still stops after Stage 0.

Testing: `models/download.sh` also fetches `models/samples/bus.jpg`
(ultralytics' demo photo, AGPL, so not committed; it's gitignored).
`crates/detector/tests/run_inference.rs` asserts YOLOv8n finds exactly
the reference 4 people in it, and that a blank frame produces none.
`detector`'s unit tests cover decode, NMS, IoU and letterbox mapping on
hand-built tensors.

**Real-footage check (user's ~25 s club rugby clip).** This is the same
clip as in Phase 1a: 706x848 portrait, with the actual picture in a band
from rows 265 to 583 and black bars above and below. `offload inspect`
over the whole clip: clearly visible players and the referee get tight,
confident boxes, with no false positives seen. Distant packs and the
edges of rucks are under-detected (frame 0: 2 boxes for about 20
players). Mean person boxes per frame, measured at 1 fps over the clip:

| | threshold 0.40 (original default) | threshold 0.25 (new default) |
|---|---|---|
| full frame | 7.6 | 9.6 |
| black bars cropped | 8.2 | 10.3 |

On the 3 s wide shot it went from 4 boxes to 16, and every one of the
16 at crop + 0.25 was checked by eye as a real person. As a result,
`detector.confidence_threshold` now defaults to **0.25**: the 0.4
originally written into SPEC didn't hold up on real footage, and the
user's stated rule is that a guessed value gets corrected when reality
disagrees. `bus.jpg` still gives exactly its 4 people at 0.25.
Black-bar cropping was considered and **rejected**: that clip's
portrait-with-bars format is a one-off, not representative of the
footage Offload is meant for (the user's call), so there's no
letterbox auto-crop step.

Phase 3 (tracking) is complete, in `crates/tracker/`:
- `kalman.rs` (3a): a port of ByteTrack's constant-velocity Kalman
  filter over `(cx, cy, aspect, height)`, with noise scaled by box
  height, using `nalgebra` fixed-size matrices (f64).
- `assignment.rs`: exact Hungarian assignment with a cost limit applied
  *inside* the optimisation. The matrix is padded with dummy
  rows/columns at `limit/2`, the same as `lap.lapjv(extend_cost=True)`.
  Filtering an unconstrained solution afterwards gives different, worse
  matches; a unit test covers this.
- `lib.rs` (3b): ByteTrack proper. Confirmed and lost tracks are matched
  against high-confidence boxes (IoU > 0.2); still-`Tracked` leftovers
  against low-confidence boxes (IoU > 0.5); last frame's new tracks
  against leftover high boxes (IoU > 0.3), discarding any unmatched
  one; then new tracks from boxes scoring at least high + 0.1; expiry
  after `max_lost_frames`. These IoU limits and the 0.1 margin are
  ByteTrack's reference constants, kept as named consts rather than
  config. `update` returns confirmed `Tracked`/`Lost` tracks sorted by
  ID. IDs start at 1 and are never reused. `Track` gained
  `confidence`; it does **not** carry a bbox history (ARCHITECTURE
  updated): Phase 4 can keep per-frame track lists itself.
- `draw_tracks` and `track_colour`: a stable colour per ID, with an ID
  label drawn by `detector::draw_label`, a 3×5 bitmap digit font added
  to `detector` (reusable for Phase 4b's numbered boxes; no font crate).
  `detector::draw_rect` is now public.
- `Config::detector_config_for_tracking()` lowers the detector's
  emission floor to `tracker.low_confidence_threshold` (0.1), which was
  the Phase 2b follow-up. `inspect` now detects, tracks and draws
  tracks, and reports tracked players per frame and the number of
  distinct IDs.

**`tracker.high_confidence_threshold` is now 0.4, not the 0.6 in the
SPEC draft** (ByteTrack's MOT17 value). YOLOv8n scores distant players
around 0.3–0.6, and new tracks need high + 0.1. Sweep over the user's
25 s clip (199 frames at 8 fps, 10.0 detections per frame):

| high | tracked/frame | IDs | mean frames per ID |
|---|---|---|---|
| 0.6 | 6.0 | 33 | 36 |
| 0.5 | 6.8 | 38 | 36 |
| **0.4** | **7.9** | **43** | **37** |
| 0.3 | 9.1 | 56 | 32 |

At 0.4, IDs were checked by eye over a 1.1 s stretch with a camera pan
and a ruck, and every player kept their ID. A padded goalpost is
detected and tracked as a person (a detector false positive, harmless
since the user picks themselves in Phase 4).
`crates/tracker/tests/track_real_detections.rs` runs real YOLOv8n
detections through the tracker over a simulated pan across `bus.jpg`:
four people get exactly four IDs, the two fully visible people keep
theirs for all 40 frames, and the person leaving at the left edge stays
`Tracked` on sub-0.4 boxes (the second association on real detector
output) until they're out of view.

Phase 1c (URL / YouTube source input) is complete: SPEC.md Stage 0,
in `crates/video-io/src/source.rs`. `resolve_source(input, &SourceConfig,
cache_root)` passes local paths through unchanged (missing path →
`VideoIoError::NotFound`, which `pipeline-core` maps to
`InputNotFound`). It downloads an `http(s)` input by running the
`yt-dlp` executable (`std::process::Command`, no crate) in three steps:
`--version` to check it exists, `--dump-single-json` to get metadata and
reject playlists and live/upcoming streams, then the download (merged
to MP4). The download goes to
`<cache>/sources/<extractor>-<video_id>.mp4`, and a
`<extractor>-<video_id>.source.json` sidecar is written last; its
existence is what marks a download as complete. YouTube IDs are parsed
from the URL offline (`youtube_video_id`), so a cache hit never runs
`yt-dlp`. `remove_cached_source` backs `offload cache --clear --input
<URL>`; the rest of `cache` is still a stub. `SourceConfig` derives
`Deserialize` in `video-io` and is embedded directly as
`pipeline_core::Config::source`. `--input` is now a `String` (not
`PathBuf`) on `run`/`inspect`/`cache`. `Pipeline::run` resolves the
source and then still returns "not yet implemented". `inspect`
accepts a URL but doesn't resolve it yet, since it's still a stub.
`crates/video-io/tests/resolve_source.rs` tests against a fake
`yt-dlp` shell script, because YouTube is blocked by the egress
policy (see Gotchas). The real binary was only checked for argument
parsing. An actual YouTube download hasn't been verified from this
environment.

## Deliberately deferred dependencies

`ffmpeg-next` landed in `video-io`'s `Cargo.toml` in Phase 1a (see
"Current state"). It links against system FFmpeg dev libraries
(`libavformat`, `libavcodec`, `libavutil`, `libswscale`,
`libswresample` — installed via the `apt-get` line in
`.claude/skills/run-offload/SKILL.md`'s "Prerequisites", and now also
in `.github/workflows/ci.yml`). `compiler` will need it too for Phase
7's output assembly, but hasn't been wired up yet.

`ort` (ONNX Runtime) landed in `detector`'s `Cargo.toml` in Phase 2a
(see "Current state"), with `default-features = false, features =
["std", "ndarray", "load-dynamic", "api-24"]` in the workspace
`Cargo.toml`. `load-dynamic` instead of the default
`download-binaries` is deliberate: `download-binaries` fetches a
prebuilt ONNX Runtime library from `cdn.pyke.io` at build time, and
that host is blocked by this project's egress policy. `load-dynamic`
defers loading the shared library to runtime via `dlopen`
(`ort::init_from(path)?.commit()`, called once from
`detector::ensure_ort_initialized`), reading the path from
`ORT_DYLIB_PATH` — see `.claude/skills/run-offload/SKILL.md`'s
"Prerequisites" for how to fetch that library (from
`microsoft/onnxruntime`'s GitHub releases, which isn't blocked) and
point the env var at it. `reid` will need `ort` too for Phase 4a, with
the same feature set.

`yt-dlp` (Phase 1c) is an optional *runtime* tool, not a crate. It's
only needed for URL inputs, and its path comes from `source.ytdlp_path`.
`video-io` gained `serde`/`serde_json` (sidecar and yt-dlp metadata
JSON) and `sha2` (sidecar checksum); `sha2` is a new workspace
dependency.

`nalgebra` (Phase 3, pure Rust, `default-features = false` +
`std`) is used by `tracker` for the Kalman filter's 8×8 / 4×4 matrices
and Cholesky solve. `tracker` also uses `image` for `draw_tracks`, and
takes `video-io` as a dev-dependency for its real-detection test.

Similarly, `reqwest`/`tokio`/`serde_json` are already wired into
`classifier`'s `Cargo.toml` (pure-Rust, no system deps, safe to add
early) but unused until Phase 5's Ollama client lands.

## CI

`.github/workflows/ci.yml` runs on every push to `main` and every PR:
installs FFmpeg dev libs, downloads the ONNX Runtime shared library
from `microsoft/onnxruntime`'s GitHub releases and exports
`ORT_DYLIB_PATH` (`$GITHUB_ENV`) so later steps pick it up, runs
`models/download.sh` to fetch `yolov8n.onnx`, then `cargo fmt --all
--check`, `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo build --workspace`, `cargo test --workspace`, in that order, on
`ubuntu-latest` with `Swatinem/rust-cache` for dependency caching. All
four checks were run locally against the current scaffold before this
workflow was added — fmt/clippy are clean with no warnings.

## Security advisories

GitHub's Dependabot alerts can't be read from a Claude Code cloud
session; there's no tool for them. Reproduce them locally with
`cargo install cargo-audit --locked && cargo audit`, which checks
`Cargo.lock` against the same RustSec database and usually reports
more than the alert summary shows. Fix with targeted `cargo update -p
<crate>`, which is lockfile-only when a semver-compatible patch exists.
Known remaining warning: `paste` is unmaintained (RUSTSEC-2024-0436).
It's a compile-time proc-macro pulled in through `image` → `ravif` →
`rav1e`, with no patched release, so it's accepted for now.

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
- **Assigning encoder frame pts in millisecond units (with a `1/1000`
  encoder time base) causes libx264 to reject the stream** once
  B-frames are in play: rescaling 30fps source timestamps to ms
  truncates, so two consecutive frames can round to the same
  millisecond, producing a duplicate/non-strictly-increasing pts.
  x264's frame-reordering lookahead then computes a non-monotonic dts
  and the muxer errors with `Encode("Invalid argument")`. Fix: set the
  encoder's time base to `1/fps` (`Rational(frame_rate.denominator(),
  frame_rate.numerator())`) and assign pts as a plain incrementing
  `i64` frame counter — exact integer ticks, never duplicated. See
  `ClipWriter::write` in `crates/video-io/src/lib.rs`.
- **`cdn.pyke.io` (ort's default binary CDN) is blocked by the egress
  policy.** `ort`'s default `download-binaries` feature fetches a
  prebuilt ONNX Runtime library from there at build time, which fails
  with a 403 in this environment. Use `default-features = false` with
  `load-dynamic` instead, and source the shared library separately
  (e.g. `microsoft/onnxruntime`'s GitHub releases, which isn't
  blocked) — see "Deliberately deferred dependencies."
- **`ort` v2.0.0-rc.12 with `default-features = false` won't compile
  unless `api-24` is also enabled.** `src/ep/vitis.rs` gates its FFI
  call on `feature = "load-dynamic"` alone, not the separate `vitis`
  EP feature — so `load-dynamic` without `api-24` compiles a reference
  to `OrtApi::SessionOptionsAppendExecutionProvider_VitisAI`, a field
  that only exists in the `api-24` struct bindings, producing
  `error[E0609]: no field ... on type &'static OrtApi`. Fix: include
  `api-24` alongside `load-dynamic` in the workspace `Cargo.toml`'s
  `ort` features.
- **`ort::session::Session::run` takes `&mut self`,** so any method
  that runs inference (`Detector::new`'s warm-up, `run_inference`)
  needs a `mut` binding/`&mut self`. The `SessionOutputs` a `run` call
  returns holds a borrow of the session — if you need to move the
  session afterwards (e.g. into a struct), `drop` the outputs first or
  the compiler rejects the move with `E0505: cannot move out of
  ... because it is borrowed`. See `Detector::new` in
  `crates/detector/src/lib.rs`.

- **YouTube (`www.youtube.com`, `youtu.be`) is blocked by the egress
  policy** (proxy `CONNECT` → 403), so real `yt-dlp` downloads can't
  be exercised here or in CI. Test Stage 0 with a fake `yt-dlp` shell
  script instead (see `crates/video-io/tests/resolve_source.rs`). The
  real binary is available from `yt-dlp/yt-dlp`'s GitHub releases
  (`yt-dlp_linux`, standalone, no Python needed), which aren't blocked:
  useful for checking that it accepts an argument list.
- **YouTube video IDs are case-sensitive.** An early filename
  sanitiser lowercased them, so `youtube-dQw4w9WgXcQ.mp4` was written
  as `youtube-dqw4w9wgxcq.mp4` and the offline cache lookup (which uses
  the URL's ID verbatim) never hit. Lowercase only the extractor name.
- **`yt-dlp --dump-single-json` prints `null` to stdout on failure**
  (exit 1, with `ERROR:` lines on stderr). Check the exit status before
  parsing stdout, or the real error is replaced by a JSON parse error.

## Conventions

- Crate `Cargo.toml`s pull shared versions from
  `[workspace.dependencies]` in the root `Cargo.toml` via
  `foo.workspace = true` — don't pin versions per-crate.
- Each stage crate's `lib.rs` defines its own `Error` enum (via
  `thiserror`) and a crate-local `Result<T>` alias, matching
  ARCHITECTURE.md's "no domain logic in pipeline-core" split:
  `pipeline-core::PipelineError` only wraps orchestration failures,
  not stage internals.
