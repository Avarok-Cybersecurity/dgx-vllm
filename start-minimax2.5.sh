#!/usr/bin/env bash
set -Eeuo pipefail

# Start MiniMax-M2.5-NVFP4 (230B, 256 experts) on 2x DGX Spark with TP=2
#
# Requires two DGX Spark nodes connected via InfiniBand (10.10.10.1/2).
# Each node runs one GPU shard (~63 GB model weights per GPU).
#
# Usage:
#   ./start-minimax2.5.sh              # start with EP + Marlin (fastest, ~17 tok/s)
#   ./start-minimax2.5.sh --cutlass    # use CUTLASS MoE backend (~13.7 tok/s)
#   ./start-minimax2.5.sh --stop       # stop all containers on both nodes
#   ./start-minimax2.5.sh --status     # check cluster status
#
# Environment overrides:
#   PORT=8889 ./start-minimax2.5.sh
#   MAX_MODEL_LEN=2048 ./start-minimax2.5.sh
#   GPU_MEMORY_UTIL=0.80 ./start-minimax2.5.sh
#   HEAD_IP=10.10.10.1 WORKER_IP=10.10.10.2 ./start-minimax2.5.sh
#
# Performance (TP=2, EP + Marlin MoE):
#   ~17 tok/s decode throughput (16.1-17.5 tok/s range)
#   Baseline CUTLASS: ~13.7 tok/s
#
# Note: CUDA graphs are unstable with cross-node TP=2 (Ray compiled DAG
# hangs on NCCL all-reduce after 2-3 requests). Defaults to --enforce-eager
# which is equally fast since the bottleneck is NCCL, not kernel launches.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source .env for IMAGE_NAME / IMAGE_VERSION if available
if [ -f "${SCRIPT_DIR}/.env" ]; then
  set -a
  source "${SCRIPT_DIR}/.env"
  set +a
fi

# Defaults
IMAGE="${IMAGE:-avarok/dgx-vllm-nvfp4-kernel:v${IMAGE_VERSION:-22}}"
HEAD_CONTAINER="${HEAD_CONTAINER:-dgx-vllm-head}"
WORKER_CONTAINER="${WORKER_CONTAINER:-dgx-vllm-worker}"
MODEL="${MODEL:-lukealonso/MiniMax-M2.5-NVFP4}"
PORT="${PORT:-8888}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
HEAD_IP="${HEAD_IP:-10.10.10.1}"
WORKER_IP="${WORKER_IP:-10.10.10.2}"
REMOTE_USER="${REMOTE_USER:-$(whoami)}"

# Parse flags
EAGER=1    # default: CUDA graphs are unstable with cross-node TP=2 (NCCL hang)
MARLIN=1   # default: Marlin W4A16 dequant MoE backend (+24% over CUTLASS)
EP=1       # default: Expert Parallelism (splits 256 experts across GPUs)
for arg in "$@"; do
  case "$arg" in
    --stop)
      echo "Stopping MiniMax containers on both nodes..."
      sudo docker rm -f "${HEAD_CONTAINER}" 2>/dev/null || true
      ssh "${WORKER_IP}" "sudo docker rm -f ${WORKER_CONTAINER} 2>/dev/null || true"
      echo "Done."
      exit 0
      ;;
    --status)
      echo "=== Head Node (${HEAD_IP}) ==="
      sudo docker ps -a --filter "name=${HEAD_CONTAINER}" --format 'table {{.Names}}\t{{.Status}}'
      echo ""
      echo "=== Worker Node (${WORKER_IP}) ==="
      ssh "${WORKER_IP}" "sudo docker ps -a --filter 'name=${WORKER_CONTAINER}' --format 'table {{.Names}}\t{{.Status}}'"
      echo ""
      echo "=== Ray Cluster ==="
      sudo docker exec "${HEAD_CONTAINER}" ray status 2>/dev/null || echo "(Ray not running)"
      exit 0
      ;;
    --cutlass) MARLIN=0; EP=0 ;;  # CUTLASS MoE backend (~13.7 tok/s)
    --cuda-graph) EAGER=0 ;;  # WARNING: hangs after 2-3 requests (NCCL deadlock)
    --help|-h)
      sed -n '3,26p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown flag: $arg (try --help)"
      exit 1
      ;;
  esac
done

# Stop existing containers
echo "Cleaning up existing containers..."
sudo docker rm -f "${HEAD_CONTAINER}" 2>/dev/null || true
ssh "${WORKER_IP}" "sudo docker rm -f ${WORKER_CONTAINER} 2>/dev/null || true"

echo ""
echo "=== Starting MiniMax-M2.5 NVFP4 (TP=2) ==="
echo ""
echo "  Image:      ${IMAGE}"
echo "  Model:      ${MODEL}"
echo "  Port:       ${PORT}"
echo "  Context:    ${MAX_MODEL_LEN} tokens"
echo "  GPU Util:   ${GPU_MEMORY_UTIL}"
echo "  Head:       ${HEAD_IP} (${HEAD_CONTAINER})"
echo "  Worker:     ${WORKER_IP} (${WORKER_CONTAINER})"
echo "  Marlin MoE: $([ "$MARLIN" -eq 1 ] && echo "YES (+24% throughput)" || echo "no (CUTLASS)")"
echo "  Expert Par: $([ "$EP" -eq 1 ] && echo "YES (128 experts/GPU)" || echo "no (TP all-reduce)")"
echo "  Mode:       $([ "$EAGER" -eq 1 ] && echo "eager (stable)" || echo "CUDA graphs (may hang after 2-3 requests)")"
echo ""

# Ensure entrypoint is available on remote node
scp -q "${SCRIPT_DIR}/entrypoint.sh" "${REMOTE_USER}@${WORKER_IP}:/tmp/dgx-vllm-entrypoint.sh"

# Start Ray head node
echo "Starting Ray head on ${HEAD_IP}..."
sudo docker run -d \
  --name "${HEAD_CONTAINER}" \
  --network host \
  --gpus all \
  --ipc=host \
  -e HEAD_IP="${HEAD_IP}" \
  -e VLLM_HOST_IP="${HEAD_IP}" \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e RAY_memory_monitor_refresh_ms=0 \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${SCRIPT_DIR}/entrypoint.sh:/workspace/entrypoint.sh:ro" \
  "${IMAGE}" ray-head

echo "Waiting for Ray head to initialize..."
sleep 8

# Start Ray worker on remote node
echo "Starting Ray worker on ${WORKER_IP}..."
ssh "${REMOTE_USER}@${WORKER_IP}" "sudo docker run -d \
  --name ${WORKER_CONTAINER} \
  --network host \
  --gpus all \
  --ipc=host \
  -e HEAD_IP=${HEAD_IP} \
  -e WORKER_IP=${WORKER_IP} \
  -e VLLM_HOST_IP=${WORKER_IP} \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e RAY_memory_monitor_refresh_ms=0 \
  -v \${HOME}/.cache/huggingface:/root/.cache/huggingface \
  -v /tmp/dgx-vllm-entrypoint.sh:/workspace/entrypoint.sh:ro \
  ${IMAGE} ray-worker"

echo "Waiting for Ray worker to connect..."
sleep 10

# Verify Ray cluster
echo ""
echo "Checking Ray cluster..."
RAY_STATUS=$(sudo docker exec "${HEAD_CONTAINER}" ray status 2>&1)
GPU_COUNT=$(echo "${RAY_STATUS}" | grep -oP '\d+\.\d+/\d+\.\d+ GPU' | head -1 || echo "unknown")
echo "  GPUs: ${GPU_COUNT}"

# Launch vLLM server inside head container
echo ""
echo "Launching vLLM server with TP=2..."

EXTRA_ARGS="--attention-backend flashinfer --trust-remote-code --disable-custom-all-reduce"
if [ "$EAGER" -eq 1 ]; then
  EXTRA_ARGS+=" --enforce-eager"
fi
if [ "$EP" -eq 1 ]; then
  EXTRA_ARGS+=" --enable-expert-parallel"
fi

# Build env exports for MoE backend selection
MOE_EXPORTS="export VLLM_USE_FLASHINFER_MOE_FP4=0"
if [ "$MARLIN" -eq 1 ]; then
  MOE_EXPORTS+="
export VLLM_TEST_FORCE_FP8_MARLIN=1
export VLLM_NVFP4_GEMM_BACKEND=marlin"
fi

sudo docker exec -d "${HEAD_CONTAINER}" bash -c "
export VLLM_HOST_IP=${HEAD_IP}
export RAY_ADDRESS=${HEAD_IP}:6379
export MASTER_ADDR=${HEAD_IP}
${MOE_EXPORTS}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export SAFETENSORS_FAST_GPU=1
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export GLOO_SOCKET_IFNAME=enp1s0f0np0
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vllm serve ${MODEL} \
  --host 0.0.0.0 \
  --port ${PORT} \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --max-model-len ${MAX_MODEL_LEN} \
  --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
  --max-num-seqs ${MAX_NUM_SEQS} \
  ${EXTRA_ARGS} \
  2>&1 | tee /tmp/vllm-minimax.log
"

echo ""
echo "=== MiniMax-M2.5 Starting ==="
echo ""
echo "  Logs:     sudo docker exec ${HEAD_CONTAINER} tail -f /tmp/vllm-minimax.log"
echo "  Test:     curl http://localhost:${PORT}/v1/models"
echo "  Stop:     $0 --stop"
echo "  Status:   $0 --status"
echo ""
echo "Startup takes ~10 minutes (model loading on both nodes)."
echo "First run downloads ~126 GB of weights on each node."
