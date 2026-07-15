#!/usr/bin/env bash
set -euo pipefail


# ============================================================
# REQUIRED ARGUMENTS
# ============================================================

if [[ $# -lt 2 ]]; then
    echo "Usage:"
    echo "  $0 HOST_CHECKPOINT_DIR JFROG_IMAGE [EXPERIMENT_NAME] [EPOCHS] [IMGSZ] [DEVICE] [SAVE_PERIOD]"
    exit 1
fi


HOST_CHECKPOINT_DIR="$1"
IMAGE_NAME="$2"

EXPERIMENT_NAME="${3:-yolo11s_640_exp01}"
EPOCHS="${4:-150}"
IMGSZ="${5:-640}"
DEVICE="${6:-0}"
SAVE_PERIOD="${7:-10}"


CONTAINER_CHECKPOINT_ROOT="/checkpoints"
CONTAINER_EXPERIMENT_DIR="/checkpoints/${EXPERIMENT_NAME}"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

mkdir -p "${HOST_CHECKPOINT_DIR}"

HOST_CHECKPOINT_DIR="$(
    realpath "${HOST_CHECKPOINT_DIR}"
)"


# ============================================================
# CHECK GPU
# ============================================================

echo "Checking NVIDIA GPU..."

nvidia-smi


# ============================================================
# PULL IMAGE FROM JFROG
# ============================================================

echo "Pulling Docker image from JFrog:"

echo "  ${IMAGE_NAME}"

docker pull "${IMAGE_NAME}"


# ============================================================
# VERIFY CUDA INSIDE CONTAINER
# ============================================================

docker run \
    --rm \
    --gpus all \
    --entrypoint python \
    "${IMAGE_NAME}" \
    -c "
import torch

print(
    'CUDA available:',
    torch.cuda.is_available()
)

print(
    'GPU:',
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else 'None'
)

raise SystemExit(
    0 if torch.cuda.is_available() else 1
)
"


# ============================================================
# RUN TRAINING
# ============================================================

docker run \
    --rm \
    --gpus all \
    --ipc=host \
    -v "${HOST_CHECKPOINT_DIR}:/checkpoints" \
    "${IMAGE_NAME}" \
    --checkpoint-dir "${CONTAINER_EXPERIMENT_DIR}" \
    --epochs "${EPOCHS}" \
    --imgsz "${IMGSZ}" \
    --device "${DEVICE}" \
    --save-period "${SAVE_PERIOD}"


# ============================================================
# DONE
# ============================================================

echo
echo "Training finished."

echo "Results:"
echo "  ${HOST_CHECKPOINT_DIR}/${EXPERIMENT_NAME}"

echo "Best model:"
echo "  ${HOST_CHECKPOINT_DIR}/${EXPERIMENT_NAME}/weights/best.pt"
