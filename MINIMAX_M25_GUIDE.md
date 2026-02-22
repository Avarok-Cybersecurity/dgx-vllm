# MiniMax-M2.5-NVFP4 on 2x DGX Spark (TP=2)

Deploy the 230B-parameter MiniMax-M2.5 MoE model across two DGX Spark nodes using tensor parallelism with Expert Parallelism and Marlin MoE.

---

## Hardware Requirements

- **2x NVIDIA DGX Spark GB10** connected via InfiniBand (RoCE)
- **119.7 GB GPU memory per node** (~63 GB model weights per GPU)
- **InfiniBand network**: 10.10.10.1 (head) / 10.10.10.2 (worker)
- **SSH access** from head to worker node (passwordless)

## Model

| Property | Value |
|----------|-------|
| Model | [lukealonso/MiniMax-M2.5-NVFP4](https://huggingface.co/lukealonso/MiniMax-M2.5-NVFP4) |
| Parameters | 230B total |
| Architecture | MoE (256 experts, top-k routing) + Attention |
| Quantization | NVFP4 (E2M1 weights + FP8 block scales) |
| Weights | ~126 GB (26 safetensor shards) |

---

## Quick Start

```bash
# Pull the image on both nodes
docker pull avarok/dgx-vllm-nvfp4-kernel:v22

# On node 2:
ssh 10.10.10.2 "docker pull avarok/dgx-vllm-nvfp4-kernel:v22"

# Start the cluster (EP + Marlin, ~17 tok/s)
./start-minimax2.5.sh

# Check status
./start-minimax2.5.sh --status

# Stop everything
./start-minimax2.5.sh --stop
```

Startup takes ~10 minutes (model loading on both nodes). First run downloads ~126 GB of weights on each node.

### Verify

```bash
# Wait for startup, then:
curl http://localhost:8888/v1/models

curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lukealonso/MiniMax-M2.5-NVFP4",
       "messages":[{"role":"user","content":"Hello!"}],
       "max_tokens":100}' | jq -r '.choices[0].message.content'
```

---

## Performance

| Configuration | Throughput | Stability |
|--------------|-----------|-----------|
| CUTLASS + TP=2 (baseline) | 13.7 tok/s | Stable |
| Marlin + TP=2 (no EP) | 11.0-17.9 tok/s | Inconsistent |
| **EP + Marlin + TP=2 (default)** | **16.1-17.5 tok/s** | **Stable (10/10)** |
| EP + Marlin + CUDA graphs | 15.3-17.3 tok/s | Crashes after ~8 req |
| EP + Marlin + Ngram spec | 14.8-17.5 tok/s | No benefit over base |
| PP=2 (pipeline parallel) | 13.5 tok/s | Crashes on 2nd request |

### Why EP + Marlin is fastest

1. **Expert Parallelism (EP)**: Splits 256 experts across 2 GPUs (128 each) instead of tensor-splitting every expert. This replaces costly per-MoE-layer NCCL all-reduce with lighter expert dispatch/gather.
2. **Marlin MoE**: W4A16 dequantization backend, faster than CUTLASS for cross-node workloads on SM121.

Without EP, the MoE layers require all-reduce across the InfiniBand link for every layer, creating variable latency (11-18 tok/s). EP reduces cross-node traffic to expert routing only, giving consistent ~17 tok/s.

### Why CUDA graphs don't help

CUDA graphs capture NCCL operations. On replay across nodes, the compiled DAG occasionally deadlocks on NCCL all-reduce (Ray `RayChannelTimeoutError` after 2-8 requests). Since the bottleneck is NCCL latency (not kernel launches), `--enforce-eager` is equally fast and completely stable.

---

## Configuration

### Script flags

```bash
./start-minimax2.5.sh                # EP + Marlin (default, ~17 tok/s)
./start-minimax2.5.sh --cutlass      # CUTLASS MoE backend (~13.7 tok/s)
./start-minimax2.5.sh --cuda-graph   # Enable CUDA graphs (WARNING: unstable)
./start-minimax2.5.sh --stop         # Stop all containers on both nodes
./start-minimax2.5.sh --status       # Check cluster and container status
```

### Environment overrides

```bash
PORT=8889 ./start-minimax2.5.sh
MAX_MODEL_LEN=2048 ./start-minimax2.5.sh
GPU_MEMORY_UTIL=0.80 ./start-minimax2.5.sh
HEAD_IP=10.10.10.1 WORKER_IP=10.10.10.2 ./start-minimax2.5.sh
IMAGE=avarok/dgx-vllm-nvfp4-kernel:v22 ./start-minimax2.5.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE` | `avarok/dgx-vllm-nvfp4-kernel:v22` | Docker image |
| `MODEL` | `lukealonso/MiniMax-M2.5-NVFP4` | HuggingFace model ID |
| `PORT` | `8888` | API server port |
| `MAX_MODEL_LEN` | `4096` | Maximum context length |
| `GPU_MEMORY_UTIL` | `0.85` | GPU memory fraction |
| `MAX_NUM_SEQS` | `32` | Maximum concurrent sequences |
| `HEAD_IP` | `10.10.10.1` | Head node InfiniBand IP |
| `WORKER_IP` | `10.10.10.2` | Worker node InfiniBand IP |
| `REMOTE_USER` | `$(whoami)` | SSH user for worker node |

---

## Architecture

The deployment uses a Ray cluster spanning both nodes:

```
  Node 1 (10.10.10.1)                 Node 2 (10.10.10.2)
  +-----------------------+            +-----------------------+
  | dgx-vllm-head         |            | dgx-vllm-worker       |
  |                       |            |                       |
  | Ray Head + GPU 0      |<---------->| Ray Worker + GPU 1    |
  | EP rank 0             | InfiniBand | EP rank 1             |
  | 128 experts           |   (RoCE)   | 128 experts           |
  | vLLM API server :8888 |            |                       |
  +-----------------------+            +-----------------------+
```

- **Ray** manages the distributed cluster and compiled DAG communication
- **Expert Parallelism** assigns 128 of 256 experts to each GPU
- **Marlin MoE** handles W4A16 dequantization for each expert
- **VLLM_HOST_IP** is exported in the Ray daemon environment to ensure correct InfiniBand IP binding (fixes the 3-IP mismatch bug)

---

## Monitoring

```bash
# View vLLM logs
sudo docker exec dgx-vllm-head tail -f /tmp/vllm-minimax.log

# Check Ray cluster
sudo docker exec dgx-vllm-head ray status

# Check containers on both nodes
./start-minimax2.5.sh --status

# GPU memory usage
nvidia-smi
ssh 10.10.10.2 nvidia-smi
```

---

## Troubleshooting

### Server hangs after a few requests

CUDA graphs are enabled. Restart with the default configuration (eager mode):
```bash
./start-minimax2.5.sh --stop
./start-minimax2.5.sh
```

### "3 unique IP addresses" error

The `VLLM_HOST_IP` environment variable is not being set correctly in the Ray daemon. The entrypoint.sh fix exports `VLLM_HOST_IP` before `ray start`. Ensure you're using the latest entrypoint.sh from this repo.

### Ray worker won't connect

1. Check SSH access: `ssh 10.10.10.2 echo ok`
2. Check InfiniBand: `ping 10.10.10.2`
3. Check image exists on worker: `ssh 10.10.10.2 "docker images | grep dgx-vllm"`
4. Check entrypoint was copied: `ssh 10.10.10.2 "ls -la /tmp/dgx-vllm-entrypoint.sh"`

### OOM during model loading

Lower GPU memory utilization:
```bash
GPU_MEMORY_UTIL=0.80 ./start-minimax2.5.sh
```

### Slow or inconsistent throughput

If seeing 11-14 tok/s instead of 16-17 tok/s, verify EP + Marlin is active:
```bash
sudo docker exec dgx-vllm-head grep -E "MARLIN|expert_parallel" /tmp/vllm-minimax.log
```

You should see:
```
enable_expert_parallel: True
Using 'MARLIN' NvFp4 MoE backend
```

---

## Manual Deployment

If you prefer to run the containers manually instead of using the script:

### 1. Start Ray head (node 1)

```bash
sudo docker run -d \
  --name dgx-vllm-head \
  --network host --gpus all --ipc=host \
  -e HEAD_IP=10.10.10.1 \
  -e VLLM_HOST_IP=10.10.10.1 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e RAY_memory_monitor_refresh_ms=0 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  avarok/dgx-vllm-nvfp4-kernel:v22 ray-head
```

### 2. Start Ray worker (node 2)

```bash
sudo docker run -d \
  --name dgx-vllm-worker \
  --network host --gpus all --ipc=host \
  -e HEAD_IP=10.10.10.1 \
  -e WORKER_IP=10.10.10.2 \
  -e VLLM_HOST_IP=10.10.10.2 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e RAY_memory_monitor_refresh_ms=0 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  avarok/dgx-vllm-nvfp4-kernel:v22 ray-worker
```

### 3. Launch vLLM (inside head container)

```bash
sudo docker exec -d dgx-vllm-head bash -c "
export VLLM_HOST_IP=10.10.10.1
export RAY_ADDRESS=10.10.10.1:6379
export MASTER_ADDR=10.10.10.1
export VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_TEST_FORCE_FP8_MARLIN=1
export VLLM_NVFP4_GEMM_BACKEND=marlin
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export GLOO_SOCKET_IFNAME=enp1s0f0np0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vllm serve lukealonso/MiniMax-M2.5-NVFP4 \
  --host 0.0.0.0 --port 8888 \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 32 \
  --enable-expert-parallel \
  --attention-backend flashinfer \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --enforce-eager \
  2>&1 | tee /tmp/vllm-minimax.log
"
```

---

## Optimization History

All configurations tested during development:

| Approach | Result | Notes |
|----------|--------|-------|
| CUTLASS + TP=2 + eager | 13.7 tok/s | Baseline, stable |
| Marlin + TP=2 (no EP) | 11.0-17.9 tok/s | NCCL variance dominates |
| **EP + Marlin + TP=2 + eager** | **16.1-17.5 tok/s** | **Best stable config** |
| EP + Marlin + PIECEWISE graphs | 15.3-17.3 tok/s | Crashes after ~8 requests |
| EP + Marlin + FULL_AND_PIECEWISE | 14.8 tok/s (1st only) | Crashes on 2nd request |
| EP + Marlin + torch.compile | 9-13 tok/s | Slower than eager |
| EP + Marlin + Ngram spec decode | 14.8-17.5 tok/s | No improvement |
| PP=2 + Marlin | 13.5 tok/s (1 req) | Ray channel crash on 2nd |
| NCCL Ring+Simple tuning | 12.8-14.3 tok/s | Worse than auto |
| NCCL IB disabled | 13.7 tok/s | No change (IB auto-detected) |
