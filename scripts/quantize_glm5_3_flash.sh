#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

export SAFETENSORS_FAST_GPU=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

MODEL_ID="${MODEL_ID:-zai-org/GLM-5.3-Flash-BF16}"
EXPORT_DIR="${EXPORT_DIR:-/data/models/GLM-5.3-Flash-NVFP4-routed}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/GLM-5.3-Flash-NVFP4-routed}"
CALIB_CONFIG="${CALIB_CONFIG:-configs/calib_glm5_3_flash.toml}"
CALIB_DATA_DIR="${CALIB_DATA_DIR:-./data}"

mkdir -p "${OUTPUT_DIR}"

python quantize.py \
    --model glm5_3_flash \
    --model-id "${MODEL_ID}" \
    --export-dir "${EXPORT_DIR}" \
    --batch-tokens 65536 \
    --calib-config "${CALIB_CONFIG}" \
    --data-dir "${CALIB_DATA_DIR}" \
    --save-amax "${OUTPUT_DIR}/amax.safetensors" \
    --save-quantiles "${OUTPUT_DIR}/quantiles.json" \
    --streaming \
    --cpu-capacity "${CPU_CAPACITY:-100GiB}" \
    --streaming-gpu0-storage-capacity "${GPU0_STORAGE_CAPACITY:-0GiB}" \
    --floor-amaxes \
    "$@"
