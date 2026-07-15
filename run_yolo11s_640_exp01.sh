#!/usr/bin/env bash
set -euo pipefail

# Required arguments:
#   1. HOST_CHECKPOINT_DIR
#   2. JFROG_IMAGE
#
# Example:
#   ./yolo11s_640_exp01.sh /home/aoms/yolo_checkpoints \
#       mycompany.jfrog.io/path/to/bank-logo-yolo11s:123

if [[ $# -lt 2 ]]; then
    echo "Usage:"
    echo "  $0 HOST_CHECKPOINT_DIR JFROG_IMAGE"
    exit 1
fi

HOST_CHECKPOINT_DIR="$1"
IMAGE_NAME="$2"

EXPERIMENT_NAME="yolo11s_640_exp01"
EPOCHS=150
IMGSZ=640
DEVICE=0
SAVE_PERIOD=10

CONTAINER_CHECKPOINT_ROOT="/checkpoints"
CONTAINER_EXPERIMENT_DIR="${CONTAINER_CHECKPOINT_ROOT}/${EXPERIMENT_NAME}"

mkdir -p "${HOST_CHECKPOINT_DIR}"
HOST_CHECKPOINT_DIR="$(realpath "${HOST_CHECKPOINT_DIR}")"

echo "============================================================"
echo "YOLO11s EXPERIMENT 1 - IMGSZ 640"
echo "============================================================"
echo "Image:               ${IMAGE_NAME}"
echo "Host checkpoint dir: ${HOST_CHECKPOINT_DIR}"
echo "Experiment name:     ${EXPERIMENT_NAME}"
echo "Epochs:              ${EPOCHS}"
echo "Image size:          ${IMGSZ}"
echo "GPU device:          ${DEVICE}"
echo "Save period:         ${SAVE_PERIOD}"
echo "============================================================"

echo
echo "[1/3] Checking NVIDIA GPU..."
nvidia-smi

echo
echo "[2/3] Pulling Docker image from JFrog..."
docker pull "${IMAGE_NAME}"

echo
echo "[3/3] Verifying CUDA inside container..."
docker run \
    --rm \
    --gpus all \
    --entrypoint python \
    "${IMAGE_NAME}" \
    -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); raise SystemExit(0 if torch.cuda.is_available() else 1)"

echo
echo "Starting training..."

docker run \
    --rm \
    --gpus all \
    --ipc=host \
    -v "${HOST_CHECKPOINT_DIR}:${CONTAINER_CHECKPOINT_ROOT}" \
    "${IMAGE_NAME}" \
    --checkpoint-dir "${CONTAINER_EXPERIMENT_DIR}" \
    --epochs "${EPOCHS}" \
    --imgsz "${IMGSZ}" \
    --device "${DEVICE}" \
    --save-period "${SAVE_PERIOD}"

echo
echo "============================================================"
echo "TRAINING FINISHED"
echo "============================================================"
echo "Results:"
echo "  ${HOST_CHECKPOINT_DIR}/${EXPERIMENT_NAME}"
echo
echo "Best model:"
echo "  ${HOST_CHECKPOINT_DIR}/${EXPERIMENT_NAME}/weights/best.pt"
