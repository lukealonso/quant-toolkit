#!/bin/bash
set -e

cd "$(dirname "$0")/.."
source venv/bin/activate

export SAFETENSORS_FAST_GPU=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

python quantize.py \
    --model glm4_7 \
    --model-id /nvme/models/safetensors/GLM-4.7/ \
    --export-dir /nvme/models/temp/GLM-4.7-NVFP4 \
    --calib-config configs/calib_glm4_7.toml \
    --cpu-capacity 256GiB \
    --streaming \
    --floor-amaxes \
    --save-amax /nvme/models/temp/GLM-4.7-NVFP4/amaxes.safetensors \
    --batch-tokens 1572864 \
