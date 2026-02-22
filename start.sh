#!/usr/bin/env bash
set -Eeuo pipefail

# Start Qwen3-Next-80B-A3B-Instruct-NVFP4 on DGX Spark (GB10 / SM121)
#
# Usage:
#   ./start.sh              # defaults: Marlin + MTP (~59.9 tok/s, fastest)
#   ./start.sh --cutlass    # VLLM_CUTLASS MoE only (~36.4 tok/s, baseline)
#   ./start.sh --marlin     # Marlin W4A16 MoE only, no MTP (~47 tok/s)
#   ./start.sh --eager      # disable CUDA graphs (for debugging)
#
# Environment overrides:
#   PORT=8889 ./start.sh
#   MAX_MODEL_LEN=2048 ./start.sh
#   GPU_MEMORY_UTIL=0.85 ./start.sh
#   CONTAINER_NAME=my-vllm ./start.sh
#   IMAGE=dgx-vllm:v22 ./start.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source .env for IMAGE_NAME / IMAGE_VERSION if available
if [ -f "${SCRIPT_DIR}/.env" ]; then
  set -a
  source "${SCRIPT_DIR}/.env"
  set +a
fi

# Defaults
IMAGE="${IMAGE:-${IMAGE_NAME:-dgx-vllm}:v${IMAGE_VERSION:-22}}"
CONTAINER_NAME="${CONTAINER_NAME:-dgx-vllm-nvfp4}"
MODEL="${MODEL:-nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4}"
PORT="${PORT:-8888}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"

# Parse flags — default is Marlin + MTP (fastest)
MARLIN=1
MTP=1
EAGER=0
EXPLICIT_BACKEND=0
for arg in "$@"; do
  case "$arg" in
    --cutlass)    MARLIN=0; MTP=0; EXPLICIT_BACKEND=1 ;;
    --marlin)     MARLIN=1; MTP=0; EXPLICIT_BACKEND=1 ;;
    --eager)      EAGER=1 ;;
    --help|-h)
      sed -n '3,14p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown flag: $arg (try --help)"
      exit 1
      ;;
  esac
done

# Build env vars array
ENV_ARGS=(
  # Disable FlashInfer fused MoE — JIT hits missing E2M1 PTX on SM121
  -e VLLM_USE_FLASHINFER_MOE_FP4=0
  # Model and server config
  -e "MODEL=${MODEL}"
  -e "PORT=${PORT}"
  -e "GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL}"
  -e "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
  -e "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
)

# VLLM_EXTRA_ARGS pieces
EXTRA_ARGS="--attention-backend flashinfer --kv-cache-dtype fp8"

if [ "$MARLIN" -eq 1 ]; then
  ENV_ARGS+=(
    -e VLLM_TEST_FORCE_FP8_MARLIN=1
    -e VLLM_NVFP4_GEMM_BACKEND=marlin
  )
fi

if [ "$MTP" -eq 1 ]; then
  EXTRA_ARGS+=" --num-speculative-tokens 2 --speculative-model [ngram] --ngram-prompt-lookup-max 3"
fi

if [ "$EAGER" -eq 1 ]; then
  EXTRA_ARGS+=" --enforce-eager"
fi

# Reduce memory fragmentation
ENV_ARGS+=(-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)

ENV_ARGS+=(-e "VLLM_EXTRA_ARGS=${EXTRA_ARGS}")

# Stop existing container if running
if sudo docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Stopping existing container: ${CONTAINER_NAME}"
  sudo docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1
fi

echo "=== Starting Qwen3-Next-80B NVFP4 ==="
echo ""
echo "  Image:      ${IMAGE}"
echo "  Container:  ${CONTAINER_NAME}"
echo "  Model:      ${MODEL}"
echo "  Port:       ${PORT}"
echo "  Context:    ${MAX_MODEL_LEN} tokens"
echo "  GPU Util:   ${GPU_MEMORY_UTIL}"
echo "  Marlin MoE: $([ "$MARLIN" -eq 1 ] && echo "YES" || echo "no (VLLM_CUTLASS)")"
echo "  MTP Spec:   $([ "$MTP" -eq 1 ] && echo "YES (2 tokens)" || echo "no")"
echo "  CUDA Graph: $([ "$EAGER" -eq 1 ] && echo "disabled (eager)" || echo "FULL_AND_PIECEWISE")"
echo ""

sudo docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --gpus all \
  --ipc=host \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  "${ENV_ARGS[@]}" \
  "${IMAGE}" serve

echo ""
echo "Container started. Waiting for server..."
echo ""
echo "  Logs:     sudo docker logs -f ${CONTAINER_NAME}"
echo "  Test:     curl http://localhost:${PORT}/v1/models"
echo "  Stop:     sudo docker stop ${CONTAINER_NAME}"
echo "  Bench:    BENCH_PORT=${PORT} BENCH_CONTAINER=${CONTAINER_NAME} BENCH_MAX_MODEL_LEN=${MAX_MODEL_LEN} python3 sparse_fp4_kernel/bench_full.py"
echo ""
echo "Startup takes ~7 minutes (model loading + CUDA graph capture)."
