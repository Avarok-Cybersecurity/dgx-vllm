#!/usr/bin/env bash
set -Eeuo pipefail

# Entrypoint script for dgx-vllm container
# Supports multiple modes: ray-head, ray-worker, serve

MODE="${1:-serve}"

# Common environment setup
export PATH="/opt/venv/bin:/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# Auto-detect InfiniBand HCA if available
if command -v ibv_devinfo >/dev/null 2>&1; then
  HCA="$(ibv_devinfo -l 2>/dev/null | head -n1 || true)"
  if [[ -n "${HCA:-}" ]]; then
    export NCCL_IB_HCA="${HCA}"
    echo "Using InfiniBand HCA: ${HCA}"
  fi
fi

# Set node IP - for head use HEAD_IP, for worker use WORKER_IP, otherwise auto-detect
if [[ -n "${WORKER_IP:-}" ]]; then
  NODE_IP="${WORKER_IP}"
elif [[ -n "${HEAD_IP:-}" ]]; then
  NODE_IP="${HEAD_IP}"
else
  # Auto-detect IP on InfiniBand interface
  NODE_IP=$(ip -4 addr show ${NCCL_SOCKET_IFNAME:-enp1s0f0np0} 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' || echo "127.0.0.1")
fi

case "$MODE" in
  ray-head)
    echo "=== Starting Ray Head Node ==="
    echo "Node IP: ${NODE_IP}"

    # Start Ray head node
    ray start --head \
      --node-ip-address="${NODE_IP}" \
      --port=6379 \
      --dashboard-host=0.0.0.0 \
      --num-gpus="${NUM_GPUS:-1}" \
      --disable-usage-stats

    echo "Ray head started at ${NODE_IP}:6379"

    # Keep container running
    tail -f /dev/null
    ;;

  ray-worker)
    echo "=== Starting Ray Worker Node ==="
    echo "Node IP: ${NODE_IP}"
    echo "Connecting to Ray head: ${HEAD_IP}:6379"

    if [[ -z "${HEAD_IP:-}" ]]; then
      echo "ERROR: HEAD_IP environment variable must be set for ray-worker mode"
      exit 1
    fi

    # Start Ray worker node
    ray start \
      --address="${HEAD_IP}:6379" \
      --node-ip-address="${NODE_IP}" \
      --num-gpus="${NUM_GPUS:-1}" \
      --disable-usage-stats

    echo "Ray worker started, connected to ${HEAD_IP}:6379"

    # Keep container running
    tail -f /dev/null
    ;;

  serve)
    echo "=== Starting vLLM Server ==="

    # Default configuration
    MODEL="${MODEL:-}"
    PORT="${PORT:-8888}"
    HOST="${HOST:-0.0.0.0}"
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
    GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.75}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"

    if [[ -z "${MODEL}" ]]; then
      echo "ERROR: MODEL environment variable must be set for serve mode"
      echo "Example: MODEL=DevQuasar/Qwen.Qwen3-Next-80B-A3B-Instruct-FP8-Dynamic"
      exit 1
    fi

    echo "Model: ${MODEL}"
    echo "Port: ${PORT}"
    echo "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"

    # Configure for distributed execution if TP > 1
    if [[ ${TENSOR_PARALLEL_SIZE} -gt 1 ]]; then
      if [[ -z "${HEAD_IP:-}" ]]; then
        echo "ERROR: HEAD_IP must be set for distributed execution (TP > 1)"
        exit 1
      fi

      echo "Distributed execution: connecting to Ray at ${HEAD_IP}:6379"
      export RAY_ADDRESS="${HEAD_IP}:6379"
      export VLLM_HOST_IP="${HEAD_IP}"
      export MASTER_ADDR="${HEAD_IP}"
      export RAY_memory_usage_threshold=0.98

      DISTRIBUTED_ARGS="--distributed-executor-backend ray --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}"
    else
      echo "Single GPU execution"
      DISTRIBUTED_ARGS=""
    fi

    # Build vLLM command
    VLLM_CMD="vllm serve ${MODEL} \
      --host ${HOST} \
      --port ${PORT} \
      --max-model-len ${MAX_MODEL_LEN} \
      --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
      --max-num-seqs ${MAX_NUM_SEQS} \
      ${DISTRIBUTED_ARGS} \
      ${VLLM_EXTRA_ARGS:-}"

    echo "Starting vLLM..."
    echo "Command: ${VLLM_CMD}"
    echo "Note: SM_121 uses native Triton backend (integrated at build time)"

    exec ${VLLM_CMD}
    ;;

  bash)
    echo "=== Starting Interactive Bash Shell ==="
    exec /bin/bash
    ;;

  *)
    echo "Unknown mode: ${MODE}"
    echo "Valid modes: ray-head, ray-worker, serve, bash"
    exit 1
    ;;
esac
