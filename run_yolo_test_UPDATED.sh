#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/evaluate_yolo_models.py"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: evaluate_yolo_models.py was not found."
    echo "Expected location:"
    echo "  ${PYTHON_SCRIPT}"
    echo
    echo "Put both scripts in the same folder."
    exit 1
fi

if [[ $# -lt 3 ]]; then
    cat <<'EOF'
Usage:
  ./run_yolo_test.sh \
      DATASET_ROOT \
      MODEL_640_BEST_PT \
      MODEL_960_BEST_PT \
      [OUTPUT_DIR] \
      [DEVICE] \
      [VAL_BATCH] \
      [BENCHMARK_BATCH] \
      [BENCHMARK_REPEATS] \
      [PREDICT_CONF] \
      [PREDICT_IOU] \
      [WORKERS] \
      [WARMUP_RUNS]

CPU batch-1 example:
  SKIP_PREDICTIONS=1 ./run_yolo_test.sh \
      ../bank_logo_dataset \
      ../yolo11s_640_exp01/weights/best.pt \
      ../yolo11s_960_exp02/weights/best.pt \
      ./yolo_fps_results_cpu_b1 \
      cpu \
      4 \
      1 \
      5 \
      0.50 \
      0.45 \
      4 \
      2

Environment switches:
  SKIP_PREDICTIONS=1
      Do not save annotated prediction images.

  AGNOSTIC_NMS=1
      Enable class-agnostic NMS.
EOF
    exit 1
fi

DATASET_ROOT="$1"
MODEL_640="$2"
MODEL_960="$3"
OUTPUT_DIR="${4:-./yolo_test_results}"
DEVICE="${5:-cpu}"
VAL_BATCH="${6:-4}"
BENCHMARK_BATCH="${7:-1}"
BENCHMARK_REPEATS="${8:-5}"
PREDICT_CONF="${9:-0.50}"
PREDICT_IOU="${10:-0.45}"
WORKERS="${11:-4}"
WARMUP_RUNS="${12:-2}"

COMMAND=(
    python
    "${PYTHON_SCRIPT}"
    --dataset-root "${DATASET_ROOT}"
    --model-640 "${MODEL_640}"
    --model-960 "${MODEL_960}"
    --output-dir "${OUTPUT_DIR}"
    --device "${DEVICE}"
    --val-batch "${VAL_BATCH}"
    --benchmark-batch "${BENCHMARK_BATCH}"
    --benchmark-repeats "${BENCHMARK_REPEATS}"
    --predict-conf "${PREDICT_CONF}"
    --predict-iou "${PREDICT_IOU}"
    --workers "${WORKERS}"
    --warmup-runs "${WARMUP_RUNS}"
)

if [[ "${SKIP_PREDICTIONS:-0}" == "1" ]]; then
    COMMAND+=(--skip-predictions)
fi

if [[ "${AGNOSTIC_NMS:-0}" == "1" ]]; then
    COMMAND+=(--agnostic-nms)
fi

echo "Running YOLO evaluation and FPS benchmark"
echo "Python script:      ${PYTHON_SCRIPT}"
echo "Dataset:            ${DATASET_ROOT}"
echo "640 model:          ${MODEL_640}"
echo "960 model:          ${MODEL_960}"
echo "Output directory:   ${OUTPUT_DIR}"
echo "Device:             ${DEVICE}"
echo "Validation batch:   ${VAL_BATCH}"
echo "Benchmark batch:    ${BENCHMARK_BATCH}"
echo "Benchmark repeats:  ${BENCHMARK_REPEATS}"
echo "Confidence:         ${PREDICT_CONF}"
echo "NMS IoU:            ${PREDICT_IOU}"
echo "Workers:            ${WORKERS}"
echo "Warm-up runs:       ${WARMUP_RUNS}"
echo "Skip predictions:   ${SKIP_PREDICTIONS:-0}"
echo "Agnostic NMS:       ${AGNOSTIC_NMS:-0}"
echo

"${COMMAND[@]}"
