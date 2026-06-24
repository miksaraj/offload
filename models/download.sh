#!/usr/bin/env bash
# Downloads the ONNX models Offload needs into ./models/.
#
# yolov8n.onnx is a pre-exported asset from the ultralytics/assets GitHub
# release — no Python/PyTorch export step needed. osnet_x1_0.onnx is filled
# in during Phase 4a (no equivalent pre-exported asset exists upstream yet).
set -euo pipefail

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

YOLO_MODEL="${MODELS_DIR}/yolov8n.onnx"
YOLO_URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx"
YOLO_SHA256="b2bc52f40e8e1c532427d5bde3575a5d5b571b739fab2c6df443733ed1589cbd"

OSNET_MODEL="${MODELS_DIR}/osnet_x1_0.onnx"

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

if [[ -f "${OSNET_MODEL}" ]]; then
  echo "✓ osnet_x1_0.onnx already present"
else
  echo "✗ osnet_x1_0.onnx missing — TODO: pin a download URL for the OSNet-x1.0 ONNX export (Phase 4a)"
fi

echo
echo "See ARCHITECTURE.md (section 6) for model format/source requirements."
