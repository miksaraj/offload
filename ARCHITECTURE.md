# Offload — Architecture

This document describes the technical architecture of Offload: the workspace layout, crate responsibilities, data flow, model choices, and the key design decisions behind them.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Workspace Layout](#2-workspace-layout)
3. [Crate Responsibilities](#3-crate-responsibilities)
4. [Data Flow](#4-data-flow)
5. [Core Data Types](#5-core-data-types)
6. [Models & Inference](#6-models--inference)
7. [Intermediate Caching](#7-intermediate-caching)
8. [Concurrency Model](#8-concurrency-model)
9. [Key Design Decisions](#9-key-design-decisions)
10. [Dependency Map](#10-dependency-map)

---

## 1. System Overview

Offload is a sequential processing pipeline. A single match video enters; a compiled highlight reel exits. The pipeline has six distinct stages, each implemented as a separate Rust crate, orchestrated by a top-level binary crate.

```
┌─────────────────────────────────────────────────────────────────┐
│                        offload (binary)                         │
│                    pipeline-core orchestrator                   │
└──────┬───────────────────────────────────────────────┬──────────┘
       │                                               │
       ▼                                               ▼
┌─────────────┐   frames    ┌──────────┐  detections  ┌─────────┐
│  video-io   │ ──────────► │ detector │ ────────────► │ tracker │
│  (ffmpeg)   │             │ (YOLOv8) │              │(ByteTrack│
└─────────────┘             └──────────┘              └────┬─────┘
                                                           │ tracks
                                                           ▼
                                                     ┌──────────┐
                                                     │   reid   │
                                                     │ (OSNet)  │
                                                     └────┬─────┘
                                                          │ identity
                                                          ▼
                                                   ┌────────────┐
                                                   │ classifier │
                                                   │  (Ollama)  │
                                                   └─────┬──────┘
                                                         │ moments
                                                         ▼
                                                   ┌────────────┐
                                                   │  compiler  │
                                                   │  (ffmpeg)  │
                                                   └────────────┘
```

---

## 2. Workspace Layout

```
offload/
├── Cargo.toml                  # workspace root
├── Cargo.lock
├── justfile                    # build/run/test tasks
├── config/
│   └── offload.default.toml    # default configuration template
├── models/
│   ├── download.sh             # fetches ONNX models
│   ├── yolov8n.onnx            # (downloaded, not committed)
│   └── osnet_x1_0.onnx         # (downloaded, not committed)
├── crates/
│   ├── pipeline-core/          # orchestrator & shared config
│   ├── video-io/               # frame extraction & clip assembly
│   ├── detector/               # YOLOv8 player detection
│   ├── tracker/                # ByteTrack multi-object tracking
│   ├── reid/                   # OSNet re-identification
│   ├── classifier/             # Ollama vision moment scoring
│   └── compiler/               # final video assembly
├── tests/
│   └── integration/            # end-to-end pipeline tests
└── docs/
    ├── README.md
    ├── ARCHITECTURE.md         # this file
    └── SPEC.md
```

---

## 3. Crate Responsibilities

### `pipeline-core`

The brain of the system. Contains no domain logic itself — only orchestration.

- Reads and validates `offload.toml`
- Instantiates each stage crate in order
- Manages the intermediate cache (read/write JSON at each stage boundary)
- Propagates the `tracing` subscriber to all crates
- Exposes the `Pipeline` struct that `main.rs` calls
- Derives the `Vec<MatchSegment>` for the run — either from manual config, from a heuristic gap-detection pass over cached detection counts, or (the common case) an implicit single segment spanning the whole video — and threads it into ReID's identification windowing and into moment grouping/compilation boundary checks

### `video-io`

All interaction with video files via `ffmpeg-next`.

- `FrameExtractor`: opens a video, iterates decoded frames as `Frame` structs
- `ClipWriter`: accepts a `Vec<ClipSpec>` (start/end timestamps + optional label), extracts segments, concatenates, transcodes to output
- Handles frame rate normalisation and resolution scaling
- Provides timestamp-to-frame-number conversion utilities

### `detector`

Runs YOLOv8 inference via `ort` (ONNX Runtime Rust bindings).

- `Detector::new(model_path, config)`: loads and warms up the ONNX model
- `Detector::detect(frame) -> Vec<Detection>`: preprocesses frame, runs inference, decodes output tensor, applies NMS, filters to person class only
- Preprocessing: resize to 640×640, normalise `[0, 255] → [0.0, 1.0]`, `HWC → NCHW` layout
- Output decoding: parses `[1, 84, 8400]` tensor, extracts `xywh` boxes + confidence + class scores

### `tracker`

ByteTrack multi-object tracking, implemented in pure Rust.

- `Tracker::new(config)`: initialises with configured thresholds
- `Tracker::update(detections: Vec<Detection>) -> Vec<Track>`: runs one frame of tracking, returns all active tracks with stable IDs
- Internally: Kalman filter for motion prediction, IoU-based two-stage Hungarian assignment, track lifecycle management (Tracked / Lost / Removed states)
- Each `Track` carries its history of bounding boxes for ReID sampling

### `reid`

Player re-identification via OSNet ONNX model, with an interactive identification phase.

- `ReIdModel::new(model_path)`: loads model
- `ReIdModel::embed(crop: &RgbImage) -> Embedding`: preprocesses a bounding-box crop and produces a 512-dim feature vector

**Interactive identification (`Identifier` struct):**
- `Identifier::select_candidate_frame(video, detector, config) -> (Frame, Vec<Detection>)`: extracts `candidate_frame_count` frames from the first `candidate_window_secs` of the video, runs detection on each, returns the frame with the most detections
- `Identifier::annotate_and_prompt(frame, detections) -> usize`: saves annotated JPEG with numbered bounding boxes to `/tmp/offload_select.jpg`, opens system image viewer, reads user integer input from stdin, returns selected detection index
- `Identifier::confirm(frame, subject_track) -> bool`: saves confirmation frame to `/tmp/offload_confirm.jpg`, opens viewer, prompts `[y/N]`, returns user decision
- `Identifier::build_reference(frame, detection, model) -> Embedding`: crops the selected detection, resizes to 256×128, runs OSNet inference, returns the reference embedding

**Matching (`ReIdMatcher` struct):**
- `ReIdMatcher::new(reference_embedding, model)`: initialised from the result of `Identifier`
- `ReIdMatcher::identify(candidates: Vec<(TrackId, Embedding)>) -> Option<TrackId>`: cosine similarity comparison, returns best match above threshold
- Temporal smoother: maintains a vote buffer per track; only locks identity after N consecutive high-confidence matches
- Re-entry matching: automatic, no user interaction required after initial lock

### `classifier`

Semantic moment scoring via Ollama local HTTP API.

- `OllamaClassifier::new(config)`: configures endpoint URL, model name, prompt template
- `OllamaClassifier::score(frame: &Frame) -> MomentScore`: base64-encodes frame, calls `/api/generate`, parses JSON response `{ score, kind, event }`
- **Highlight**: positive play event — try, line break, dominant tackle, turnover won, effective kick, offload
- **Lowlight**: bad play / error event — missed tackle, high tackle, knock-on, fumble, being turned over, penalty conceded; lowlights are discrete events, not sustained passive periods
- **Neutral**: subject present but not the focus of significant action — not compiled into output
- Moment aggregation: groups consecutive high-scoring `highlight` samples into highlight moments; flags individual `lowlight` samples as lowlight moments
- Handles Ollama unavailability gracefully (falls back to neutral classification if `ollama.required = false`)

### `compiler`

Final output assembly via `ffmpeg-next`.

- `Compiler::new(config)`: sets up output encoding parameters
- `Compiler::compile(source: &Path, moments: Vec<Moment>) -> Result<()>`: extracts clip segments from source with configured padding, applies transitions, overlays event label text, concatenates, mixes optional background audio, writes output file
- Debug mode: writes annotated intermediate frames showing bounding boxes, track IDs, identity lock status, and moment scores

---

## 4. Data Flow

### Stage-by-stage

```
INPUT
  match.mp4 + offload.toml + [user interaction at Stage 4]
      │
      ▼
[video-io] Frame extraction
  → Frame { timestamp_ms, frame_number, rgb_pixels, width, height }
  → Sampled at configured rate (e.g. every 3rd frame for detection)
      │
      ▼
[detector] YOLOv8 person detection per frame
  → Vec<Detection> { bbox: Rect, confidence: f32, class_id: u32 }
  → Cached to: cache/detections.json
      │
      ▼
[tracker] ByteTrack association across frames
  → Vec<Track> { track_id, bbox, state, age }
  → Stable track_id maintained through occlusion
  → Cached to: cache/tracks.json
      │
      ▼
[pipeline-core] Match segmentation (multi-match inputs only; no-op by default)
  → Manual segments from config, or heuristic gap-detection over detection counts
  → Vec<MatchSegment> { start_ms, end_ms, label }
  → Cached to: cache/segments.json
      │
      ▼
[reid] Interactive identification (Sub-phase 4A)
  ⟳ Target segment selected (first segment, or the one containing candidate_window_start_secs)
  ⟳ Candidate frames extracted → detected → annotated JPEG saved
  ⟳ System image viewer opened → user selects bounding box number
  ⟳ Reference embedding generated from selected crop
  ⟳ Confirmation frame shown → user confirms [y/N] → retry on rejection
      │ reference_embedding locked
      ▼
[reid] Per-frame identity matching (Sub-phase 4B)
  → Cosine similarity per track per frame
  → Temporal vote buffer + lock threshold
  → Output: subject_track_id (the track that is you)
  → Cached to: cache/identity.json
      │
      ▼
[classifier] Moment scoring (only frames where subject is present)
  → Sampled at 1 frame per 2 seconds of subject presence
  → Ollama vision query per sample → {score, kind, event}
  → kind: "highlight" | "lowlight" | "neutral"
  → Highlights: positive plays (try, carry, tackle won, turnover won)
  → Lowlights: error events (missed tackle, knock-on, penalty, turnover lost)
  → Grouped into: Vec<Moment> { start_ms, end_ms, kind, label, segment }
  → Same-kind merge never bridges a MatchSegment boundary
  → Cached to: cache/moments.json
      │
      ▼
[compiler] Video assembly
  → ClipSpec per moment (with padding, clamped to the moment's segment bounds)
  → Transition effects between clips
  → Event label overlays
  → Optional background audio mix
  → Optional: one output file per segment instead of one combined file
      │
      ▼
OUTPUT
  highlights.mp4
```

### Identity re-entry

When the subject's track is lost (goes to `Lost` state — e.g. tackled to ground, leaves frame) and later a new track appears that could be the subject re-entering, the ReID matcher is re-queried. If similarity exceeds the re-entry threshold, the new track is assigned as the subject and tracking continues. This prevents the subject from "disappearing" after a tackle or ruck.

The same mechanism handles longer absences without any special-casing: a sub coming on mid-match, or a player going on/off repeatedly in 7s/touch, is just a longer-than-usual `Lost` state followed by re-entry — re-entry has no upper bound on gap length, only on similarity. The one place that *does* need special-casing is a multi-match input (see "Match segmentation" below): identity must not be assumed to carry across a match boundary by track continuity, since the subject may legitimately not appear in a later match at all.

### Match segmentation

A `MatchSegment` is a time range within the source video treated as a self-contained match. By default there is exactly one implicit segment spanning the whole video, so single-match inputs (the common case) are unaffected. For multi-match inputs (a tournament livestream, or to point identification at a specific point in a long file), `pipeline-core` derives an explicit `Vec<MatchSegment>` from manual config or from a heuristic pass over Stage 2's detection counts (looking for sustained stretches of near-zero detections — broadcast cut away from an active game).

Segments are deliberately *not* derived from a dedicated ML model (e.g. scoreboard OCR or scene-cut detection): the detection-count signal is already computed for every frame by Stage 2, so the heuristic is a few lines of analysis over existing cached data rather than a new model, new dependency, or extra inference pass. The tradeoff is that it only detects boundaries marked by a visible break in play (teams off the pitch) — a tournament feed that cuts hard from one match's final whistle straight into the next kickoff with players still milling around would not produce a clean gap. Manual `segmentation.segments` config is the fallback for those cases, and is the recommended path whenever the user already knows their team's kickoff times.

---

## 5. Core Data Types

```rust
// pipeline-core
pub struct MatchSegment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub label: Option<String>,
}

// video-io
pub struct Frame {
    pub timestamp_ms: u64,
    pub frame_number: u64,
    pub pixels: Vec<u8>,       // raw RGB, row-major
    pub width: u32,
    pub height: u32,
}

// detector
pub struct Detection {
    pub bbox: Rect,            // pixel coords in original frame space
    pub confidence: f32,
    pub class_id: u32,         // 0 = person
}

pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

// tracker
pub struct Track {
    pub track_id: u64,
    pub bbox: Rect,
    pub state: TrackState,
    pub age: u32,
    pub frames_since_update: u32,
}

pub enum TrackState {
    Tracked,
    Lost,
    Removed,
}

// reid
pub struct Embedding(pub Vec<f32>);   // 512-dim feature vector

// classifier
pub struct MomentScore {
    pub timestamp_ms: u64,
    pub score: f32,            // 0.0–10.0
    pub kind: MomentKind,
    pub event_label: String,   // e.g. "Missed tackle", "Line break", "Knock-on"
}

pub struct Moment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub kind: MomentKind,
    pub peak_score: f32,
    pub label: String,
    pub segment: Option<MatchSegment>,   // None when segmentation is inactive
}

pub enum MomentKind {
    Highlight,
    Lowlight,
}

// compiler
pub struct ClipSpec {
    pub start_ms: u64,
    pub end_ms: u64,
    pub label: Option<String>,
}
```

---

## 6. Models & Inference

### YOLOv8 (Detection)

| Property | Value |
|---|---|
| Model | YOLOv8n or YOLOv8s (configurable) |
| Format | ONNX |
| Input | `[1, 3, 640, 640]` float32, normalised `[0, 1]` |
| Output | `[1, 84, 8400]` float32 |
| Output format | 84 = 4 (xywh) + 80 (COCO class scores) |
| Source | Pre-exported from ultralytics — no Python needed |
| Runtime | `ort` crate (ONNX Runtime Rust bindings) |

Start with `yolov8n` (nano) for speed; upgrade to `yolov8s` (small) if detection quality is insufficient on typical match footage distances.

### OSNet (Re-Identification)

| Property | Value |
|---|---|
| Model | OSNet-x1.0 |
| Format | ONNX |
| Input | `[1, 3, 256, 128]` float32, ImageNet normalised |
| Output | `[1, 512]` float32 embedding vector |
| Source | Export from torchreid, or find community ONNX export |
| Runtime | `ort` crate |

The 256×128 input is the standard ReID crop size. Bounding box crops from YOLO/tracker output must be resized to this before inference.

### Ollama Vision Model (Classification)

| Property | Value |
|---|---|
| Model | LLaVA 1.6 (default) or Moondream 2 (faster, lower quality) |
| Interface | Local HTTP API at `http://localhost:11434/api/generate` |
| Input | Base64-encoded JPEG frame + text prompt |
| Output | JSON `{ "score": float, "event": string }` |
| Runtime | `reqwest` async HTTP client |

The prompt instructs the model to score the moment from the perspective of the tracked player's involvement. The model is not expected to be rugby-expert; it is used for coarse action/non-action discrimination, which vision LLMs handle adequately.

---

## 7. Intermediate Caching

Processing a full 80-minute match is expensive. Offload caches the output of each stage so that reruns (for config tuning, debugging, or after a crash) skip already-completed work.

Cache lives at `<output_dir>/.offload_cache/<input_video_hash>/`.

| File | Contents | Invalidated when |
|---|---|---|
| `detections.json` | `Vec<(frame_number, Vec<Detection>)>` | Input video or detection config changes |
| `tracks.json` | `Vec<(frame_number, Vec<Track>)>` | Detections or tracker config changes |
| `segments.json` | `Vec<MatchSegment>` | Detections or `segmentation` config changes |
| `identity.json` | `{ subject_track_id, confidence_history }` | Tracks, segments, or reference image changes |
| `moments.json` | `Vec<Moment>` | Identity, classifier config, or prompt changes |

Cache validity is checked by hashing the inputs (video file, reference image, relevant config section) and comparing against a stored manifest. Stale cache entries are automatically discarded.

Use `offload cache --clear` to wipe all cache for the current input.

---

## 8. Concurrency Model

The pipeline is **stage-sequential but internally parallel** where practical.

- Frame extraction and detection run in a **producer-consumer pattern**: a background thread extracts frames into a bounded channel; the main thread drains the channel for detection. This keeps GPU/CPU inference fed without loading the entire video into memory.
- The Ollama classifier is **async** (`tokio`): classification requests are issued concurrently up to a configured limit (default: 4 in-flight requests), respecting Ollama's single-model concurrency.
- ByteTrack and ReID are inherently sequential (each frame depends on the previous) and run single-threaded.

The pipeline-core crate owns the `tokio` runtime. Individual stage crates expose synchronous interfaces except for `classifier`, which exposes an async interface.

---

## 9. Key Design Decisions

### Why interactive identification rather than a reference photo?

A reference photo taken outside match conditions (different lighting, no kit, studio background) produces a poor ReID embedding — the OSNet model is trained on person crops extracted from surveillance and street footage, not posed portraits. Even a photo in kit suffers from differences in camera angle, compression artefacts, and distance.

A crop extracted directly from the match footage is the ideal input: same camera, same lighting, same kit at the same distance. The interactive selection phase turns the user's own recognition ability into the identification mechanism — the human is far better at recognising themselves in a crowd than any automated approach, and the cost is 30 seconds of interaction at the start of a run. This also eliminates the friction of requiring users to have a suitable reference photo available at all.

The jersey descriptor (e.g. "blue 4") is stored for display and context purposes but is not used programmatically in v1 — appearance-based ReID is sufficient, and number OCR on low-resolution jersey crops is a stretch goal.

### Why not Python for the CV components?

The explicit project constraint. Beyond preference, there are real advantages: a single statically-linked binary is trivial to distribute to teammates; Rust's type system catches preprocessing shape mismatches at compile time rather than runtime; and `ort` is mature enough to be production-viable for ONNX inference.

### Why ONNX rather than calling Python model libraries?

ONNX Runtime (`ort` crate) is the only mature Rust path to running real CV models. The alternative — reimplementing model architectures from scratch in Rust — would be a multi-month project in itself. ONNX exports of YOLOv8 and OSNet are stable, well-documented, and available without a Python environment.

### Why ByteTrack rather than DeepSORT?

ByteTrack is simpler to implement (no appearance model dependency in the tracker itself — that's handled separately by ReID), performs better on crowded scenes (rugby has 30 players), and the algorithm is compact enough to port to Rust in one or two sessions. DeepSORT requires a separate appearance CNN embedded in the tracker, which would complicate the architecture without adding value over a dedicated ReID stage.

### Why Ollama for highlight/lowlight detection rather than a hardcoded heuristic?

A heuristic (e.g. "highlight = frame where subject bbox overlaps another player's bbox") cannot distinguish between a dominant tackle and a missed one, or between a clean carry and a knock-on. A vision model can make a coarse semantic judgement — "is this player doing something positive, negative, or neutral?" — that a geometry-based heuristic cannot.

This distinction is especially important for lowlights: a lowlight is not a prolonged period of inactivity (that would just be noise), but a discrete bad play event — a missed tackle, a knock-on, a high tackle that gives away a penalty. The Ollama classifier is the only component capable of making that call from raw video without structured event tagging. Ollama makes this feasible locally without a Python environment.

### Why separate `reid` and `tracker` crates?

Tracker = geometric identity (IoU-based bbox association). ReID = appearance identity (embedding similarity). They solve different subproblems and can be tuned independently. Keeping them separate also makes it straightforward to swap either implementation — e.g. upgrading to a newer ReID model — without touching the other.

### Why is match segmentation a no-op by default rather than always-on?

The v1 design (single 80-minute XV match) doesn't need segmentation at all, and most users never will — heuristic gap-detection adds a chance of misfiring (a long lineout/injury stoppage in a single match could in theory look like a between-matches gap if thresholds are misconfigured). Making it opt-in (`segmentation.enabled` or a manual `segments` list) keeps the default path exactly as simple and predictable as it is today, and confines the new complexity — and its failure modes — to the tournament/sub use case that actually needs it.

---

## 10. Dependency Map

```
offload (binary)
└── pipeline-core
    ├── video-io
    │   └── ffmpeg-next
    ├── detector
    │   ├── ort (ONNX Runtime)
    │   └── ndarray
    ├── tracker
    │   └── (pure Rust — no external ML deps)
    ├── reid
    │   ├── ort
    │   ├── ndarray
    │   └── image (crop/resize)
    ├── classifier
    │   ├── reqwest (async HTTP)
    │   ├── tokio
    │   └── serde_json
    └── compiler
        └── ffmpeg-next

Shared across all crates:
├── tracing / tracing-subscriber
├── serde / serde_json
├── thiserror (error types)
└── anyhow (error propagation in binary)
```

External runtime dependencies (not Rust crates — must be installed on host):
- `ffmpeg` ≥ 6.0 (system library, linked by `ffmpeg-next`)
- `onnxruntime` shared library (loaded at runtime via `ORT_DYLIB_PATH`;
  see CLAUDE.md's "Deliberately deferred dependencies" for why `ort`'s
  default bundling is disabled in this project)
- `ollama` binary running locally on port 11434
