#!/usr/bin/env bash
# Downloads the ONNX models Offload needs into ./models/.
#
# Stub: the real model URLs (pinned, versioned releases of the YOLOv8 and
# OSNet ONNX exports) are filled in during Phase 2/4. For now this just
# documents what's expected to land here.
set -euo pipefail

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

YOLO_MODEL="${MODELS_DIR}/yolov8n.onnx"
OSNET_MODEL="${MODELS_DIR}/osnet_x1_0.onnx"

echo "Offload model downloader (stub)"
echo "Models directory: ${MODELS_DIR}"
echo

if [[ -f "${YOLO_MODEL}" ]]; then
  echo "✓ yolov8n.onnx already present"
else
  echo "✗ yolov8n.onnx missing — TODO: pin a download URL for the YOLOv8n ONNX export"
fi

if [[ -f "${OSNET_MODEL}" ]]; then
  echo "✓ osnet_x1_0.onnx already present"
else
  echo "✗ osnet_x1_0.onnx missing — TODO: pin a download URL for the OSNet-x1.0 ONNX export"
fi

echo
echo "This script is a placeholder. See ARCHITECTURE.md (section 6) for model"
echo "format/source requirements once real download URLs are wired up."
