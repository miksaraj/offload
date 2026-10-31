#!/usr/bin/env bash
# Downloads the ONNX models Offload needs into ./models/ (plus a sample
# image the detector tests use, into ./models/samples/).
#
# yolov8n.onnx is a pre-exported asset from the ultralytics/assets GitHub
# release — no Python/PyTorch export step needed. osnet_x1_0.onnx is
# OSNet-x1.0 (torchreid, trained on MSMT17 "combineall") as exported to
# ONNX by PINTO0309 for BoT-SORT-ONNX-TensorRT, published as a GitHub
# release asset (torchreid's own weights live on Google Drive and the
# Hugging Face mirrors are blocked by this project's egress policy).
# Input `base_images` [N,3,256,128] float32 (RGB, ImageNet-normalised by
# the caller), output `features` [N,512].
set -euo pipefail

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

YOLO_MODEL="${MODELS_DIR}/yolov8n.onnx"
YOLO_URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx"
YOLO_SHA256="b2bc52f40e8e1c532427d5bde3575a5d5b571b739fab2c6df443733ed1589cbd"

OSNET_MODEL="${MODELS_DIR}/osnet_x1_0.onnx"
OSNET_URL="https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/releases/download/onnx/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_Nx3x256x128.onnx"
OSNET_SHA256="65e4b18b8a2e6c6b27d3c7720a15c0d49cfe7b226b6a2250c223fbfd24e991c7"

# Sample image for the detector's real-inference test (Phase 2b): the
# standard ultralytics demo photo (4 people + a bus). AGPL-licensed like
# the model, so fetched rather than committed.
SAMPLES_DIR="${MODELS_DIR}/samples"
BUS_IMAGE="${SAMPLES_DIR}/bus.jpg"
BUS_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg"
BUS_SHA256="c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63"

echo "Offload model downloader"
echo "Models directory: ${MODELS_DIR}"
echo

if [[ -f "${YOLO_MODEL}" ]]; then
  echo "✓ yolov8n.onnx already present"
else
  echo "Downloading yolov8n.onnx from ${YOLO_URL}..."
  curl -sSL -o "${YOLO_MODEL}.tmp" "${YOLO_URL}"
  echo "${YOLO_SHA256}  ${YOLO_MODEL}.tmp" | sha256sum -c -
  mv "${YOLO_MODEL}.tmp" "${YOLO_MODEL}"
  echo "✓ yolov8n.onnx downloaded and verified"
fi

if [[ -f "${BUS_IMAGE}" ]]; then
  echo "✓ samples/bus.jpg already present"
else
  echo "Downloading samples/bus.jpg from ${BUS_URL}..."
  mkdir -p "${SAMPLES_DIR}"
  curl -sSL -o "${BUS_IMAGE}.tmp" "${BUS_URL}"
  echo "${BUS_SHA256}  ${BUS_IMAGE}.tmp" | sha256sum -c -
  mv "${BUS_IMAGE}.tmp" "${BUS_IMAGE}"
  echo "✓ samples/bus.jpg downloaded and verified"
fi

if [[ -f "${OSNET_MODEL}" ]]; then
  echo "✓ osnet_x1_0.onnx already present"
else
  echo "Downloading osnet_x1_0.onnx from ${OSNET_URL}..."
  curl -sSL -o "${OSNET_MODEL}.tmp" "${OSNET_URL}"
  echo "${OSNET_SHA256}  ${OSNET_MODEL}.tmp" | sha256sum -c -
  mv "${OSNET_MODEL}.tmp" "${OSNET_MODEL}"
  echo "✓ osnet_x1_0.onnx downloaded and verified"
fi

echo
echo "See ARCHITECTURE.md (section 6) for model format/source requirements."
