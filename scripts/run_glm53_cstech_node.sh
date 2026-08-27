#!/usr/bin/env bash
set -euo pipefail

# Launch either the pinned two-host BF16 TP8 runtime or the single-host TP4
# runtime used for official-FP8 and NVFP4 capture. For two hosts, start node
# rank 1 before node rank 0. The pinned sparse-SM120 backend currently supports
# only the effective fp8_ds_mla cache layout.

IMAGE="${IMAGE:-cstechdev/vllm@sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5}"
MODEL_DIR="${MODEL_DIR:-/data/models/GLM-5.3-Flash-BF16-b1967181}"
MODEL_REVISION="${MODEL_REVISION:-b1967181a3917ae70a437f4884748f6b8e3a1f4d}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.3-Flash-BF16}"
MODEL_MOUNT_SOURCE="${MODEL_MOUNT_SOURCE:-${MODEL_DIR}}"
MODEL_MOUNT_DEST="${MODEL_MOUNT_DEST:-/model}"
MODEL_CONTAINER_PATH="${MODEL_CONTAINER_PATH:-/model}"

NODE_RANK="${NODE_RANK:?set NODE_RANK to 0 (head) or 1 (worker)}"
NNODES="${NNODES:-2}"
TP_SIZE="${TP_SIZE:-8}"
DCP_SIZE="${DCP_SIZE:-1}"
MASTER_ADDR="${MASTER_ADDR:-10.42.20.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
HEAD_HOST_IP="${HEAD_HOST_IP:-10.42.20.1}"
WORKER_HOST_IP="${WORKER_HOST_IP:-10.42.20.2}"
NCCL_IFACE="${NCCL_IFACE:-enp33s0f1np1}"

PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-10}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MTP_TOKENS="${MTP_TOKENS:-5}"
PREFIX_CACHING="${PREFIX_CACHING:-1}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-7200}"
DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS:-7200}"
CPU_DISTRIBUTED_TIMEOUT_SECONDS="${CPU_DISTRIBUTED_TIMEOUT_SECONDS:-7200}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
SAFETENSORS_LOAD_STRATEGY="${SAFETENSORS_LOAD_STRATEGY:-prefetch}"
SAFETENSORS_PREFETCH_NUM_THREADS="${SAFETENSORS_PREFETCH_NUM_THREADS:-8}"
SAFETENSORS_PREFETCH_BLOCK_SIZE="${SAFETENSORS_PREFETCH_BLOCK_SIZE:-16777216}"
CAPTURE_DIR="${CAPTURE_DIR:-}"
ROUTED_CAPTURE_DIR="${ROUTED_CAPTURE_DIR:-}"
EXPECTED_IMAGE_ID="${EXPECTED_IMAGE_ID:-}"

case "${NODE_RANK}" in
  0)
    ROLE="head"
    HOST_IP="${HEAD_HOST_IP}"
    ;;
  1)
    ROLE="worker"
    HOST_IP="${WORKER_HOST_IP}"
    ;;
  *)
    echo "NODE_RANK must be 0 or 1, got ${NODE_RANK}" >&2
    exit 2
    ;;
esac

case "${NNODES}:${TP_SIZE}" in
  2:8) ;;
  1:4)
    if [[ "${NODE_RANK}" != 0 ]]; then
      echo "The single-host TP4 runtime requires NODE_RANK=0." >&2
      exit 2
    fi
    ;;
  *)
    echo "This sealed campaign launcher requires NNODES=2/TP_SIZE=8 or NNODES=1/TP_SIZE=4." >&2
    exit 2
    ;;
esac
if ((DCP_SIZE < 1 || TP_SIZE % DCP_SIZE != 0)); then
  echo "DCP_SIZE must be a positive divisor of TP_SIZE." >&2
  exit 2
fi
if [[ "${KV_CACHE_DTYPE}" != fp8 ]]; then
  echo "The pinned sparse-SM120 backend only supports effective fp8_ds_mla; requested ${KV_CACHE_DTYPE}." >&2
  exit 2
fi
if [[ "${SAFETENSORS_LOAD_STRATEGY}" != auto && "${SAFETENSORS_LOAD_STRATEGY}" != prefetch ]]; then
  echo "SAFETENSORS_LOAD_STRATEGY must be auto or prefetch." >&2
  exit 2
fi
if ((SAFETENSORS_PREFETCH_NUM_THREADS < 1 || SAFETENSORS_PREFETCH_BLOCK_SIZE < 1)); then
  echo "Safetensors prefetch thread count and block size must be positive." >&2
  exit 2
fi
if [[ ! "${PREFIX_CACHING}" =~ ^[01]$ ]]; then
  echo "PREFIX_CACHING must be 0 or 1." >&2
  exit 2
fi
if ((MTP_TOKENS < 0)); then
  echo "MTP_TOKENS must be non-negative." >&2
  exit 2
fi
if [[ -n "${CAPTURE_DIR}" || -n "${ROUTED_CAPTURE_DIR}" ]]; then
  if [[ "${PREFIX_CACHING}" != 0 || "${MTP_TOKENS}" != 0 || "${MAX_NUM_SEQS}" != 1 ]]; then
    echo "Capture requires PREFIX_CACHING=0, MTP_TOKENS=0, and MAX_NUM_SEQS=1." >&2
    exit 2
  fi
fi
if [[ -n "${CAPTURE_DIR}" ]]; then
  mkdir -p "${CAPTURE_DIR}"
fi
if [[ -n "${ROUTED_CAPTURE_DIR}" ]]; then
  if [[ "${MAX_NUM_BATCHED_TOKENS}" != 2048 ]]; then
    echo "Routed capture requires MAX_NUM_BATCHED_TOKENS=2048 for one complete panel window per step." >&2
    exit 2
  fi
  mkdir -p "${ROUTED_CAPTURE_DIR}"
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Missing model directory: ${MODEL_DIR}" >&2
  exit 2
fi
if [[ ! -f "${MODEL_DIR}/model.safetensors.index.json" ]]; then
  echo "Missing model index under ${MODEL_DIR}." >&2
  exit 2
fi
if [[ ! -d "${MODEL_MOUNT_SOURCE}" ]]; then
  echo "Missing model mount source: ${MODEL_MOUNT_SOURCE}." >&2
  exit 2
fi

CONTAINER_NAME="${CONTAINER_NAME:-glm53-flash-bf16-tp8-${ROLE}}"
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/vllm-glm53-flash-bf16-tp8}"
mkdir -p "${CACHE_DIR}"

ACTUAL_IMAGE_ID="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
if [[ -n "${EXPECTED_IMAGE_ID}" && "${ACTUAL_IMAGE_ID}" != "${EXPECTED_IMAGE_ID}" ]]; then
  echo "Image ID mismatch: expected ${EXPECTED_IMAGE_ID}, got ${ACTUAL_IMAGE_ID}." >&2
  exit 2
fi
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

capture_args=()
if [[ -n "${CAPTURE_DIR}" ]]; then
  capture_args=(
    -e VLLM_KLD_HIDDEN_CAPTURE_DIR=/capture-hidden
    -v "${CAPTURE_DIR}:/capture-hidden:rw"
  )
fi
routed_capture_args=()
if [[ -n "${ROUTED_CAPTURE_DIR}" ]]; then
  routed_capture_args=(
    -e VLLM_GLM53_ROUTED_CAPTURE_DIR=/capture-routed
    -v "${ROUTED_CAPTURE_DIR}:/capture-routed:rw"
  )
fi
eager_args=()
if [[ -n "${CAPTURE_DIR}" || -n "${ROUTED_CAPTURE_DIR}" ]]; then
  eager_args=(--enforce-eager)
fi
prefix_args=()
if [[ "${PREFIX_CACHING}" == 1 ]]; then
  prefix_args=(--enable-prefix-caching)
else
  prefix_args=(--no-enable-prefix-caching)
fi
speculative_args=()
if ((MTP_TOKENS > 0)); then
  speculative_args=(
    --speculative-config
    "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}"
  )
fi
headless_args=()
if [[ "${NNODES}" == 2 && "${NODE_RANK}" != 0 ]]; then
  # vLLM's multi-node multiprocess executor requires every non-head TP node
  # to run without an API server.  Without this, the follower constructs an
  # EngineCore and fails during KV initialization because collective_rpc is
  # valid only on the head node.
  headless_args=(--headless)
fi
load_args=()
if [[ "${SAFETENSORS_LOAD_STRATEGY}" == prefetch ]]; then
  load_args=(
    --safetensors-load-strategy prefetch
    --safetensors-prefetch-num-threads "${SAFETENSORS_PREFETCH_NUM_THREADS}"
    --safetensors-prefetch-block-size "${SAFETENSORS_PREFETCH_BLOCK_SIZE}"
  )
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --init \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e VLLM_ENGINE_READY_TIMEOUT_S="${READY_TIMEOUT_SECONDS}" \
  -e OMP_NUM_THREADS=2 \
  -e VLLM_HOST_IP="${HOST_IP}" \
  -e GLOO_SOCKET_IFNAME="${NCCL_IFACE}" \
  -e NCCL_SOCKET_IFNAME="${NCCL_IFACE}" \
  -e NCCL_NET=Socket \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_DEBUG=INFO \
  -e NCCL_ENV_PLUGIN=none \
  -e NCCL_NET_PLUGIN=none \
  -e NCCL_RMA_PLUGIN=none \
  -e NCCL_GIN_PLUGIN=none \
  "${capture_args[@]}" \
  "${routed_capture_args[@]}" \
  -v "${MODEL_MOUNT_SOURCE}:${MODEL_MOUNT_DEST}:ro" \
  -v "${CACHE_DIR}:/root/.cache" \
  "${IMAGE}" \
  "${MODEL_CONTAINER_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --decode-context-parallel-size "${DCP_SIZE}" \
  --nnodes "${NNODES}" \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --distributed-executor-backend mp \
  --distributed-timeout-seconds "${DISTRIBUTED_TIMEOUT_SECONDS}" \
  --cpu-distributed-timeout-seconds "${CPU_DISTRIBUTED_TIMEOUT_SECONDS}" \
  "${load_args[@]}" \
  --disable-custom-all-reduce \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  "${eager_args[@]}" \
  "${prefix_args[@]}" \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --default-chat-template-kwargs '{"reasoning_effort":"max"}' \
  "${headless_args[@]}" \
  "${speculative_args[@]}"

cat <<EOF
Started ${CONTAINER_NAME} (node rank ${NODE_RANK})
Image: ${IMAGE}
Model revision: ${MODEL_REVISION}
TP=${TP_SIZE} DCP=${DCP_SIZE} effective KV=fp8_ds_mla
Safetensors load strategy: ${SAFETENSORS_LOAD_STRATEGY}
Image ID: ${ACTUAL_IMAGE_ID}
EOF
