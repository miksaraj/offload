# Offload — Specification

This document defines the functional requirements, pipeline behaviour, input/output contracts, CLI interface, and configuration reference for Offload.

---

## Table of Contents

1. [Goals & Non-Goals](#1-goals--non-goals)
2. [Pipeline Specification](#2-pipeline-specification)
3. [CLI](#3-cli)
4. [Inputs](#4-inputs)
5. [Outputs](#5-outputs)
6. [Configuration Reference](#6-configuration-reference)
7. [Error Handling](#7-error-handling)
8. [Build Phases](#8-build-phases)
9. [Acceptance Criteria](#9-acceptance-criteria)

---

## 1. Goals & Non-Goals

### Goals

- Accept a single match video file as input
- Detect all players in every sampled frame using a local ONNX model
- Track all players across the full video with stable per-player IDs
- Identify the subject player through an **interactive selection phase** in which the user clicks or selects themselves from an annotated candidate frame extracted from the footage — no external reference photo required
- Confirm identification with the user before committing to a full-video tracking run
- Score moments featuring the subject player for **highlight** (good plays) and **lowlight** (bad plays) significance using a local vision model
- Assemble and export a final highlight/lowlight compilation video
- Run entirely locally with no network dependency beyond Ollama (which is also local)
- Resume gracefully from cached intermediate results after interruption
- Support a configurable debug mode that produces annotated frame dumps at every pipeline stage
- Support inputs that contain more than one discrete match (e.g. a 7s tournament livestream, or an XV/touch game where the subject is substituted on partway through) by segmenting the input — manually via config or via heuristic detection — so identification, moment grouping, and clip assembly are each scoped to the correct match

### Non-Goals

- Multi-subject tracking (identifying more than one named player per run) — deferred to stretch goals
- Real-time / live processing — batch processing of recorded footage only
- Tactical team analysis — this tool is for individual personal highlights, not coaching analytics
- Automatic camera operation or recording — input must be pre-recorded footage
- Any form of cloud sync, account system, or remote storage
- Support for multi-camera inputs in v1 — single static or panning camera only

---

## 2. Pipeline Specification

### Stage 1 — Frame Extraction

**Input:** video file path, extraction config  
**Output:** stream of `Frame` structs

- Decode video using FFmpeg
- Extract frames at the configured `detection_fps` rate (default: every 3rd frame of a 25fps video ≈ 8fps)
- Normalise all frames to a consistent internal resolution (default: 1280×720) regardless of source resolution
- Preserve original timestamps; all downstream timestamps reference the source video timeline in milliseconds
- Do not load the entire video into memory — stream frames through a bounded channel

### Stage 2 — Player Detection

**Input:** `Frame`  
**Output:** `Vec<Detection>`

- Run YOLOv8 ONNX inference on each frame
- Preprocessing: resize to 640×640 with letterbox padding, normalise pixel values to `[0.0, 1.0]`, convert `HWC u8` to `NCHW f32`
- Postprocessing: decode `[1, 84, 8400]` output tensor, apply confidence threshold (default: 0.4), apply NMS (IoU threshold default: 0.45), filter to class 0 (person) only
- Scale bounding boxes back to original frame pixel coordinates after NMS
- Frames with zero detections are recorded as empty; they do not halt the pipeline

### Stage 3 — Multi-Object Tracking

**Input:** per-frame `Vec<Detection>`  
**Output:** per-frame `Vec<Track>` with stable `track_id` values

ByteTrack algorithm:

1. For each new frame, predict next position of all existing tracks via Kalman filter
2. Compute IoU matrix between predicted track positions and new high-confidence detections (confidence ≥ `tracker.high_confidence_threshold`, default: 0.6)
3. Run Hungarian algorithm on high-confidence detections; assign matched detections to tracks
4. For unmatched tracks, attempt second-stage matching against low-confidence detections (confidence between `tracker.low_confidence_threshold` and high threshold, default: 0.1–0.6)
5. Unmatched tracks after both stages move to `Lost` state
6. Tracks in `Lost` state for more than `tracker.max_lost_frames` (default: 30) frames are removed
7. Unmatched high-confidence detections with no existing track create new `Track` objects
8. Track IDs are monotonically increasing integers; never reused within a run

### Stage 3.5 — Match Segmentation (multi-match inputs only)

**Input:** per-frame detection counts (Stage 2 output), `segmentation` config  
**Output:** `Vec<MatchSegment> { start_ms, end_ms, label: Option<String> }`

Most inputs (a single XV/touch/7s game) are one implicit segment spanning the whole video and need no configuration — this stage is a no-op by default. It only activates when `segmentation.enabled = true` or `segmentation.segments` is non-empty.

1. **Manual segments** (`segmentation.segments` in `offload.toml`, an array of `{ start, end, label }` with `start`/`end` as `"HH:MM:SS"` strings): used verbatim, no detection required. This is the reliable path for a known tournament running order.
2. **Heuristic auto-detection** (`segmentation.enabled = true`, no manual list given): scans the per-frame detection counts already produced by Stage 2 for stretches longer than `segmentation.min_gap_ms` (default: 90000ms) where detected-player count stays at or below `segmentation.empty_pitch_threshold` (default: 2) — i.e. the broadcast has cut away from an active game (studio, crowd, next-game warm-up). Boundaries are placed at the midpoint of each qualifying gap. Segments shorter than `segmentation.min_segment_duration_ms` (default: 180000ms) are discarded as noise rather than treated as a match.
3. **Confirmation:** when auto-detection produces more than one segment, print the proposed list (start/end timestamps, like the `--dry-run` moment table) and prompt the user to accept, edit (re-enter as manual `segments`), or proceed without segmentation. This reuses the existing confirm-before-committing UX pattern from Sub-phase 4B rather than introducing a new interaction style.

If segmentation is inactive, all downstream stages behave exactly as today (one implicit segment = the whole video). If active, `MatchSegment`s are threaded into Stage 4A (identification windowing), Stage 5 (moment grouping must not merge across a boundary), and Stage 6 (clip padding must clamp to segment bounds, not just video bounds).

### Stage 4 — Player Re-Identification

**Input:** per-frame track list, ReID model, user interaction  
**Output:** `subject_track_id: u64`

This stage has two sub-phases: **interactive identification** (run once before the main tracking pass) and **per-frame ReID matching** (run across the full video).

#### Sub-phase 4A — Interactive Identification

Before the full video is processed, the pipeline pauses to let the user identify themselves from the actual footage. This is the only input required to identify the subject — no external reference photo is needed.

1. Extract `reid.candidate_frame_count` (default: 3) frames spread across `reid.candidate_window_secs` (default: 300) seconds of footage, starting at the beginning of the **target segment** — the first `MatchSegment` by default, or the segment containing `reid.candidate_window_start_secs` if set (use this when the subject isn't visible until partway through, e.g. subbed on mid-game, or playing in a later game within a tournament stream). When no segmentation is active, the target segment is the whole video, matching today's behaviour.
2. Run detection on each candidate frame; select the frame with the most detections (most players visible = easiest to find yourself)
3. Save an annotated JPEG to a known temporary path (e.g. `/tmp/offload_select.jpg`) with each detected player bounding box labelled with a number (1, 2, 3 …)
4. Attempt to open the image in the system default viewer:
   - macOS: `open /tmp/offload_select.jpg`
   - Linux: `xdg-open /tmp/offload_select.jpg`
   - If neither succeeds, print the path and instruct the user to open it manually
5. Print to terminal:
   ```
   ── Subject Identification ──────────────────────────────
   A candidate frame has been saved to: /tmp/offload_select.jpg
   Each detected player is labelled with a number.

   Enter the number shown on the bounding box that is you: _
   ```
6. Read the user's integer input; validate it is within range; re-prompt on invalid input
7. Optionally prompt for a jersey descriptor (informational only, stored in cache for display):
   ```
   Enter a jersey descriptor to aid confirmation (e.g. "blue 4") [Enter to skip]: _
   ```
8. Extract the selected bounding box crop from the candidate frame; resize to 256×128; run OSNet inference to obtain the `reference_embedding`

If the selected player is not detected with sufficient confidence in the candidate frame, offer to try a different candidate frame (cycle through the other candidates extracted in step 1).

#### Sub-phase 4B — Confirmation

After the reference embedding is established, run a short pre-scan (first `reid.confirmation_scan_secs`, default: 60 seconds of footage) to find the most probable subject track:

1. Run ReID matching (as described below) on the pre-scan frames
2. Take the track with the highest cumulative cosine similarity across the pre-scan
3. Save an annotated confirmation frame showing only that track highlighted in green, to `/tmp/offload_confirm.jpg`
4. Open it in the system viewer and prompt:
   ```
   ── Confirmation ────────────────────────────────────────
   A confirmation frame has been saved to: /tmp/offload_confirm.jpg
   The highlighted player (green box) is who Offload will track.

   Is this correct? [y/N]: _
   ```
5. If `y`: commit the `reference_embedding` and proceed to full-video ReID matching
6. If `n`: discard the current reference embedding and re-run Sub-phase 4A with a different candidate frame; repeat up to `reid.max_identification_attempts` (default: 5) times before exiting with an error

#### Per-Frame ReID Matching

Once the `reference_embedding` is established:

- For each frame, for each active track: extract the bounding box crop, resize to 256×128, run OSNet inference
- Compute cosine similarity between the track's embedding and `reference_embedding`
- Maintain a sliding vote buffer of length `reid.vote_window` (default: 10 frames) per track
- A track is "locked" as the subject when its mean cosine similarity exceeds `reid.lock_threshold` (default: 0.72) for the duration of the vote window
- Once locked, the subject track ID is fixed until that track is removed

#### Re-Entry Handling

- When the subject's track moves to `Lost` state (e.g. tackled to ground, enters a ruck, leaves frame), re-entry matching is activated
- On each subsequent frame, any new track with cosine similarity above `reid.reentry_threshold` (default: 0.68) is candidate for re-assignment as the subject
- Re-entry uses a shorter vote window (`reid.reentry_vote_window`, default: 5 frames)
- Re-entry does **not** prompt the user; it is automatic

#### Cross-Segment Behaviour (multi-match inputs only)

- The `reference_embedding` from Sub-phase 4A/4B is established once and reused for every segment — it is not re-collected per match
- At the start of each subsequent segment, the lock state resets and per-frame ReID matching (and re-entry) runs fresh against that segment's tracks; no new interactive prompt is shown
- A segment in which the subject never locks (e.g. their team has a bye, or they don't feature in that game) is not an error — it simply contributes zero moments in Stage 5

#### Failure Mode

- If no track achieves lock during the confirmation pre-scan, the confirmation step is skipped and the user is asked to re-identify
- If no track achieves lock during full-video processing by frame `reid.max_identification_frame` (default: frame 600, ~75 seconds at 8fps), emit a warning — the classifier stage will have no subject frames and the output will be empty

### Stage 5 — Moment Classification

**Input:** per-frame subject track presence, `Frame` images, Ollama endpoint  
**Output:** `Vec<Moment>`

#### Sampling

- Only process frames where the subject track is `Tracked` (not `Lost`)
- Sample one frame per `classifier.sample_interval_ms` (default: 2000ms) of subject presence
- Do not sample frames where the subject bounding box is smaller than `classifier.min_bbox_area_fraction` of the frame area (default: 0.002) — subject is too far away to classify reliably

#### Definitions

A **highlight** is a positive play event in which the subject is the active participant: try scored or directly assisted, clean line break, dominant or jackal-winning tackle, turnover won, effective kick, offload out of contact.

A **lowlight** is a negative play event — a bad play or error — in which the subject is responsible: missed tackle, high tackle (penalty conceded), knock-on, fumble, being turned over, going offside, poor positional play that directly concedes ground or a score. Lowlights are **discrete events**, not sustained periods of inactivity or off-ball presence. A player standing in a lineout watching is simply neutral.

#### Ollama Query

- Encode the sampled frame as base64 JPEG at `classifier.jpeg_quality` (default: 80)
- Send to Ollama `/api/generate` with the configured vision model and prompt template
- Expected response: valid JSON `{ "score": <float 0–10>, "kind": "<highlight|lowlight|neutral>", "event": "<string>" }`
- If Ollama returns malformed JSON or times out, the sample is assigned `score: 5.0`, `kind: "neutral"`, `event: "unknown"`
- Default prompt template (configurable):

```
This is a frame from a rugby union match.
The player highlighted with a green bounding box is the subject.

Classify this moment from the subject's perspective:

"highlight" — subject is actively involved in a positive play:
  try scored or assisted, clean line break, effective ball carry with gain,
  dominant tackle, jackal turnover won, good kick, offload out of contact.

"lowlight" — subject commits an error or bad play:
  missed tackle, high tackle, knock-on, fumble, being turned over,
  giving away a penalty, clear positional error conceding ground or a score.

"neutral" — subject is present but not the focus of any significant action:
  walking, standing in a ruck, off-ball positioning, set piece participation
  without direct involvement.

Also provide a score (0–10) reflecting intensity: 10 = match-defining moment,
5 = routine involvement, 1 = irrelevant.

Return ONLY a JSON object with no preamble:
{"score": <number>, "kind": "<highlight|lowlight|neutral>", "event": "<brief label>"}
```

#### Moment Grouping

- A **highlight moment** is formed from a cluster of samples where `kind == "highlight"` and `score >= classifier.highlight_threshold` (default: 6.5)
- A **lowlight moment** is formed from a single sample or tight cluster where `kind == "lowlight"` — lowlights are point events; they do not require duration
- `kind == "neutral"` samples do not form moments
- Adjacent moments of the same kind within `classifier.merge_gap_ms` (default: 4000ms) are merged, **except** across a `MatchSegment` boundary — two same-kind samples either side of a segment boundary are never merged into one moment, regardless of gap size, since they belong to different matches
- Each moment's `label` is taken from the `event` field of the highest-scoring sample in that cluster
- Moments shorter than `classifier.min_moment_duration_ms` (default: 1500ms) after padding are discarded
- When segmentation is active, each `Moment` records the `MatchSegment` it belongs to, so dry-run output and compilation can group/label by match

### Stage 6 — Compilation

**Input:** source video path, `Vec<Moment>`, compiler config  
**Output:** compiled video file

- For each moment, define a clip: `[moment.start_ms - compiler.padding_ms, moment.end_ms + compiler.padding_ms]` (default padding: 3000ms each side), clamped to the moment's `MatchSegment` bounds when segmentation is active (otherwise clamped to video bounds, as today) — this stops padding from pulling in footage from a different match
- Extract each clip from the source video (stream copy where possible, transcode only if needed for concat compatibility)
- Apply crossfade transition of `compiler.transition_duration_ms` (default: 500ms) between clips
- Overlay event label as lower-third text for `compiler.label_display_ms` (default: 3000ms) at the start of each clip
- If `compiler.audio_track` is set, mix it in at `compiler.music_volume` (default: 0.25) under the original match audio (or `compiler.music_only: true` to replace original audio)
- Encode output at `compiler.output_crf` (default: 23), `compiler.output_preset` (default: "medium"), `compiler.output_resolution` (default: source resolution)
- When segmentation is active and `compiler.split_output_per_segment = true`, write one output file per segment that has moments (named `<output>.<segment_label_or_index>.mp4`) instead of one combined file (default: `false`, matching today's single-file behaviour)

---

## 3. CLI

### `offload run`

The primary command. Runs the full pipeline. Pauses for interactive subject identification before beginning full-video processing.

```
offload run [OPTIONS] --input <FILE>

Options:
  -i, --input <FILE>          Path to source match video
  -o, --output <FILE>         Output video path [default: highlights.mp4]
  -c, --config <FILE>         Config file path [default: offload.toml]
      --debug                 Write annotated frame dumps to ./debug/
      --dry-run               Print moment list without rendering video
      --no-cache              Ignore and overwrite any existing cache
      --auto-segment          Override segmentation.enabled for this run (heuristic match-boundary detection)
  -h, --help                  Print help
```

### `offload inspect`

Debug utility: runs detection and tracking on the first N seconds of a video and dumps annotated frames, without running ReID or classification. Useful for verifying detection quality before a full run.

```
offload inspect [OPTIONS] --input <FILE>

Options:
  -i, --input <FILE>          Path to source match video
  -d, --duration <SECS>       Seconds to inspect [default: 30]
  -o, --output-dir <DIR>      Directory for annotated frames [default: ./inspect/]
  -c, --config <FILE>         Config file path [default: offload.toml]
      --show-segments         Run heuristic match segmentation and print the proposed boundaries without processing further
```

### `offload cache`

Cache management.

```
offload cache --clear [--input <FILE>]

Options:
      --clear                 Clear cache (all inputs, or just --input if specified)
  -i, --input <FILE>          Scope cache operation to this input file only
```

### `offload models`

Model management.

```
offload models --download     Download default ONNX models to ./models/
offload models --list         List currently installed models and their paths
```

---

## 4. Inputs

### Source Video

| Property | Requirement |
|---|---|
| Format | Any format decodable by FFmpeg (MP4, MKV, MOV, AVI, etc.) |
| Resolution | Minimum 720p recommended for reliable detection |
| Frame rate | Any; internally normalised |
| Duration | Any; typical use case is 80-minute match |
| Audio | Optional; preserved in output if present |
| Camera | Single fixed or panning camera; drone footage generally not supported in v1 |

### Subject Identification

There is no reference photo input. The subject is identified interactively during the pipeline run using crops extracted directly from the match footage. This approach is more reliable than an external photo because the in-footage crop captures the exact lighting conditions, kit appearance, and camera angle of the actual game.

The interactive identification phase (Stage 4A) requires:
- A terminal capable of displaying text prompts
- A system image viewer (macOS: Preview via `open`; Linux: any viewer via `xdg-open`) to display the candidate and confirmation frames
- Approximately 30–60 seconds of wall time for the user to view and respond

---

## 5. Outputs

### Compiled Video

- Format: MP4 (H.264 video, AAC audio)
- Content: ordered sequence of highlight/lowlight clips, trimmed with padding, with event label overlays
- Duration: variable depending on match and thresholds; typical output is 5–15 minutes for an 80-minute match
- Filename: configured by `--output` flag, default `highlights.mp4`

### Dry-Run Output

When `--dry-run` is passed, prints a table to stdout:

```
Offload — Moment Summary
═══════════════════════════════════════════════════════════
 #   Kind        Start      End       Duration   Label
─────────────────────────────────────────────────────────
 1   HIGHLIGHT   12:34.200  12:41.800   7.6s     Ball carry
 2   HIGHLIGHT   23:09.100  23:15.500   6.4s     Tackle made
 3   LOWLIGHT    34:22.000  34:26.400   4.4s     Missed tackle
 4   HIGHLIGHT   41:55.300  42:03.100   7.8s     Line break
 5   LOWLIGHT    58:11.700  58:15.200   3.5s     Knock-on
...
═══════════════════════════════════════════════════════════
Total: 7 highlights, 2 lowlights | Est. output: 8m 44s
```

When segmentation is active and more than one segment contains moments, the table is grouped by segment with a per-segment subtotal:

```
Offload — Moment Summary
═══════════════════════════════════════════════════════════
── Match 1 (00:00–32:10) — Pool vs Avengers 7s ──────────────
 #   Kind        Start      End       Duration   Label
 1   HIGHLIGHT   12:34.200  12:41.800   7.6s     Ball carry
 2   LOWLIGHT    34:22.000  34:26.400   4.4s     Missed tackle
── Match 2 (58:40–90:05) — Pool vs Harlequins 7s ────────────
 3   HIGHLIGHT   61:55.300  62:03.100   7.8s     Line break
═══════════════════════════════════════════════════════════
Total: 2 highlights, 1 lowlight across 2 matches | Est. output: 6m 12s
```

### Debug Output

When `--debug` is passed, writes annotated JPEG frames to `./debug/<stage>/`:

- `debug/detection/` — bounding boxes on raw detections
- `debug/tracking/` — track IDs per bounding box, coloured by state
- `debug/reid/` — cosine similarity score per track overlaid, subject highlighted in green
- `debug/classification/` — sampled frames with score and event label overlaid

---

## 6. Configuration Reference

Default config at `config/offload.default.toml`. Copy to `offload.toml` and edit.

```toml
[models]
# Paths to ONNX model files
detector = "models/yolov8n.onnx"
reid     = "models/osnet_x1_0.onnx"

[ollama]
endpoint = "http://localhost:11434"
model    = "llava"               # or "moondream", or any installed vision model
timeout_secs = 30

[video]
# FPS to sample for detection. Lower = faster, less temporal resolution.
# At 25fps source, 8 = every 3rd frame.
detection_fps = 8
# Internal working resolution (frames rescaled to this before detection)
working_width  = 1280
working_height = 720

[detector]
# YOLOv8 inference settings
confidence_threshold = 0.4
nms_iou_threshold    = 0.45
# Execution provider: "cpu", "cuda", "coreml"
execution_provider   = "cpu"

[tracker]
high_confidence_threshold = 0.6
low_confidence_threshold  = 0.1
max_lost_frames           = 30    # frames before a lost track is removed

[segmentation]
# Enable heuristic match-boundary auto-detection (no-op if `segments` is set below)
enabled                    = false
# Minimum stretch of near-empty pitch before it's considered a gap between matches
min_gap_ms                 = 90000
# Detected-player count at or below this is considered "empty pitch"
empty_pitch_threshold      = 2
# Discard candidate segments shorter than this (treated as noise, not a match)
min_segment_duration_ms    = 180000
# Manual segments, e.g. for a known tournament running order. Overrides auto-detection.
# Times are "HH:MM:SS" against the source video timeline.
# segments = [
#   { start = "00:00:00", end = "00:32:10", label = "Pool vs Avengers 7s" },
#   { start = "00:58:40", end = "01:30:05", label = "Pool vs Harlequins 7s" },
# ]

[reid]
# How many candidate frames to extract for interactive identification
candidate_frame_count    = 3
# How many seconds into the target segment to look for candidate frames
candidate_window_secs    = 300
# Optional: timestamp (seconds into the whole video) used to pick the target
# segment for identification, for cases where the subject isn't visible until
# partway through (subbed on mid-game, or first appears in a later match of a
# tournament stream). Unset = use the first segment (or whole video).
# candidate_window_start_secs = 3520
# How many seconds to pre-scan for confirmation after identification
confirmation_scan_secs   = 60
# Max re-identification attempts before exiting with error
max_identification_attempts = 5
lock_threshold           = 0.72   # cosine similarity to lock subject identity
vote_window              = 10     # frames over which to average similarity votes
reentry_threshold        = 0.68
reentry_vote_window      = 5
max_identification_frame = 600    # warn if subject not re-locked by this frame

[classifier]
sample_interval_ms       = 2000   # how often to query Ollama per subject presence
min_bbox_area_fraction   = 0.002  # skip frames where subject bbox is too small
highlight_threshold      = 6.5    # score at or above (with kind=highlight) = highlight moment
merge_gap_ms             = 4000   # merge moments of same kind closer than this
min_moment_duration_ms   = 1500   # discard moments shorter than this after padding
jpeg_quality             = 80     # JPEG quality for frames sent to Ollama
# Override the default prompt (must request JSON {score, kind, event} format)
# prompt_template = "..."

[compiler]
padding_ms             = 3000    # padding added before and after each moment
transition_duration_ms = 500     # crossfade duration between clips
label_display_ms       = 3000    # how long to show event label overlay
output_crf             = 23      # H.264 CRF (lower = higher quality/larger file)
output_preset          = "medium"
# output_resolution = "1280x720" # defaults to source resolution if unset
# audio_track = "music.mp3"      # optional background music
music_volume           = 0.25    # background music relative volume (0.0–1.0)
music_only             = false   # replace match audio with music track
split_output_per_segment = false # write one output file per match instead of one combined file

[cache]
enabled = true
# Cache directory. Defaults to <output_dir>/.offload_cache/
# dir = ".offload_cache"
```

---

## 7. Error Handling

All errors are reported via `tracing` at the appropriate level. The CLI exits with a non-zero code on fatal errors.

| Condition | Behaviour |
|---|---|
| Input video not found | Fatal error, exit 1 |
| ONNX model file not found | Fatal error with hint to run `offload models --download` |
| No players detected in candidate frames | Fatal error; suggests checking video quality and detector confidence threshold |
| User enters invalid bounding box number | Re-prompt until valid input or Ctrl-C |
| User aborts interactive identification (Ctrl-C) | Graceful exit, exit 1 |
| Identification confirmation rejected `max_identification_attempts` times | Fatal error, exit 1 |
| System image viewer unavailable | Print path to terminal; user opens manually; pipeline waits |
| Ollama not reachable at startup | Warning; classification falls back to neutral scores if `ollama.required = false` (default) |
| Ollama returns malformed JSON | Sample assigned `score: 5.0, kind: "neutral", event: "unknown"`; logged at WARN |
| Subject not re-locked by `max_identification_frame` | Warning emitted; pipeline continues; output will likely be empty |
| Zero moments found after classification | Warning; no output video written; suggests adjusting thresholds |
| FFmpeg encoding error | Fatal error with ffmpeg stderr captured in log |
| Cache write failure | Non-fatal warning; run continues without caching |
| Heuristic segmentation finds no qualifying gaps | Treated as a single implicit segment (no warning — this is the common case for a non-tournament input) |
| User rejects the proposed segment list at confirmation | Re-prompt to edit manually, or proceed with no segmentation (single implicit segment) |

---

## 8. Build Phases

The following phases define the implementation roadmap. Each phase is designed to be completable in one afternoon/evening session, assisted by AI-assisted development tooling.

| Phase | Title | Deliverable |
|---|---|---|
| 0 | Project Skeleton | Cargo workspace, CLI parses, CI green, `justfile` tasks |
| 1a | Frame Extraction | `FrameExtractor` iterates frames, PNG dump test passes |
| 1b | Clip Assembly | `ClipWriter` produces correct clips from known timestamps |
| 2a | Detection: Inference | Single-frame ONNX inference runs, raw tensor logged |
| 2b | Detection: Postprocessing | NMS and class filtering correct, annotated debug frames pass visual check |
| 3a | Tracking: Kalman Filter | Unit tests pass on synthetic track data |
| 3b | Tracking: ByteTrack | End-to-end: track IDs stable across 100-frame sample |
| 4a | ReID: Embedding | Single crop produces 512-dim embedding via ONNX |
| 4b | ReID: Interactive Selection | Annotated frame saves, system viewer opens, user picks player, embedding generated |
| 4c | ReID: Confirmation & Matching | Confirmation prompt works; per-frame cosine similarity correct |
| 4d | ReID: Temporal Smoothing | Identity lock/re-entry logic verified on test sequence |
| 4e | Match Segmentation | Manual `segmentation.segments` respected; heuristic gap-detection proposes boundaries from detection counts; confirm/edit prompt works; Stage 4A identification windowing targets the correct segment |
| 5a | Classification: Ollama Client | Single frame scored successfully via Ollama with new 3-field response |
| 5b | Classification: Moment Grouping | Highlights and lowlights correctly grouped as discrete events; merge never bridges a segment boundary |
| 6 | Integration | Full pipeline runs end-to-end on a real 10-minute clip with interaction |
| 7a | Output Polish | Transitions, overlays, and audio mixing in compiler; clip padding clamped to segment bounds; optional per-segment output files |
| 7b | Usability | Config file, progress bars, dry-run (with per-segment grouping), README |

---

## 9. Acceptance Criteria

The v1.0 release is considered complete when all of the following hold:

- [ ] `offload run --input match.mp4` completes without errors on a full 80-minute match video
- [ ] The interactive identification phase correctly opens an annotated candidate frame and accepts user selection
- [ ] The confirmation frame clearly shows the identified subject and correctly re-runs identification on rejection
- [ ] The subject player is correctly identified for ≥ 80% of the frames in which they are visibly present (verified by manual spot-check of debug frames)
- [ ] The output video contains no clips in which the subject player is absent from the majority of the clip
- [ ] Lowlight clips contain only frames where the subject commits a visible error or bad play — not passive or off-ball frames
- [ ] The output video does not contain obvious duplicate clips of the same moment
- [ ] An interrupted run resumed from cache produces identical output to a full run (interactive phase is skipped on resume if identity is cached)
- [ ] `--dry-run` output matches the actual compiled moments
- [ ] All configuration options documented in this spec are functional
- [ ] The tool compiles and runs on macOS (Apple Silicon) and Linux (x86_64) from a clean checkout using only `cargo build --release` plus the listed system dependencies
- [ ] No network requests are made at runtime except to the local Ollama instance
- [ ] On a tournament-style input with multiple matches in one file, manual `segmentation.segments` produce correctly scoped output: no clip bridges two different matches, and identification correctly targets a segment other than the first when configured to
- [ ] On the same input, heuristic auto-detection (`segmentation.enabled = true`) proposes a segment list that a human would agree matches the actual match boundaries, and the confirm/edit prompt allows correcting it
