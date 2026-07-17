#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage:"
    echo "  $0 DATASET_ROOT MODEL_640_BEST_PT MODEL_960_BEST_PT [OUTPUT_DIR] [DEVICE]"
    echo
    echo "Example:"
    echo "  $0 ./bank_logo_dataset \\\n     ../yolo11s_640_exp01/weights/best.pt \\\n     ../yolo11s_960_exp02/weights/best.pt \\\n     ./yolo_test_results \\\n     cpu"
    exit 1
fi

DATASET_ROOT="$1"
MODEL_640="$2"
MODEL_960="$3"
OUTPUT_DIR="${4:-./yolo_test_results}"
DEVICE="${5:-cpu}"

python evaluate_yolo_models.py \
    --dataset-root "${DATASET_ROOT}" \
    --model-640 "${MODEL_640}" \
    --model-960 "${MODEL_960}" \
    --output-dir "${OUTPUT_DIR}" \
    --device "${DEVICE}"
