# Offload

**Offload** is a self-hosted, privacy-first CLI tool that automatically identifies and compiles personal highlight and lowlight reels from raw rugby match footage.

You provide a full game video. Offload pauses early in the run to let you identify yourself interactively from an annotated frame extracted from the footage — no reference photo needed. From there it follows you across all 80 minutes and produces a compilation of your key moments — tries, carries, tackles, turnovers — with no manual clipping, no cloud upload, no subscription, and no reliance on club infrastructure.

> *In rugby, an offload is a pass made while in the grip of a tackle — getting the ball out of a dangerous situation. This tool offloads the tedious editing work so you can focus on the game.*

---

## Status

🚧 **Pre-development.** Architecture and specification are complete. Build not yet started.

---

## What it does

1. Extracts frames from your match video
2. Detects all players in each frame using a local YOLOv8 ONNX model
3. Tracks all detected players across frames using ByteTrack
4. Pauses to let you **identify yourself interactively** — you pick yourself from an annotated candidate frame extracted from the footage; no reference photo needed
5. Confirms identification before committing to a full-video processing run
6. Scores moments where you are present using a local Ollama vision model, distinguishing **highlights** (positive plays: tries, carries, tackles won, turnovers) from **lowlights** (bad plays: missed tackles, knock-ons, penalties conceded)
7. Assembles a final compilation video of your highlights and lowlights

Everything runs locally. No data leaves your machine.

---

## Requirements

| Dependency | Purpose |
|---|---|
| Rust (stable, ≥ 1.75) | Build toolchain |
| FFmpeg (≥ 6.0, system install) | Video decoding & encoding |
| Ollama (local, ≥ 0.1.30) | Vision model inference for highlight classification |
| ONNX model files | YOLOv8 (detection) + OSNet (ReID) — see setup |

---

## Quick Start

```bash
# Clone and build
git clone https://github.com/yourname/offload
cd offload
cargo build --release

# Download ONNX models
just download-models

# Start Ollama with a vision model
ollama pull llava

# Run on a match video — pauses for interactive identification
./target/release/offload run --input match.mp4
```

---

## CLI Reference

```
offload run     --input <video>  [--output <video>]
                [--config <path>] [--debug] [--dry-run]

offload inspect --input <video>                        # annotated frame dump
offload cache   --clear                                # wipe intermediate results
```

Full option reference: see [SPEC.md](./SPEC.md#cli).

---

## Configuration

All thresholds and model paths are controlled via `offload.toml`. Copy the default:

```bash
cp config/offload.default.toml offload.toml
```

See [SPEC.md](./SPEC.md#configuration) for all fields.

---

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full system design, crate breakdown, data flow, and model details.

## Specification

See [SPEC.md](./SPEC.md) for functional requirements, pipeline behaviour, input/output contracts, and configuration reference.

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 0 | Project skeleton & tooling | ✅ |
| 1 | Video I/O (extraction & compilation) | ⬜ |
| 2 | Player detection (YOLOv8 + ONNX) | ⬜ |
| 3 | Multi-object tracking (ByteTrack) | ⬜ |
| 4 | Player re-identification (OSNet) + match segmentation for multi-match/tournament inputs | ⬜ |
| 5 | Highlight classification (Ollama) | ⬜ |
| 6 | Pipeline integration | ⬜ |
| 7 | Output polish & usability | ⬜ |

---

## Project Principles

- **No Python.** The entire stack is Rust. CV inference runs via ONNX Runtime (`ort` crate). Vision classification runs via local Ollama HTTP API.
- **No cloud.** All processing is local. No accounts, no uploads, no API keys.
- **No club dependency.** Works with any single-camera match recording. You don't need your club to buy hardware.
- **Resumable.** Intermediate results are cached. A crash or interrupted run doesn't mean starting over.
- **Configurable, not magic.** Every threshold that affects output is exposed in `offload.toml`.

---

## License

MIT
