# DGX vLLM Docker Image

Production-ready Docker image for running vLLM on NVIDIA DGX systems with Grace Blackwell GB10 GPUs, optimized for multi-node InfiniBand deployments.

## Features

- **GB10 GPU Support**: Built with CUDA 13.0 and optimized for compute capability 12.1
- **InfiniBand/RoCE Ready**: Includes RDMA libraries and NCCL configuration for multi-GPU across nodes
- **Flexible Deployment**: Supports single-node, multi-GPU, and multi-node tensor parallelism
- **CUTLASS Support**: FP4/FP6/FP8 MoE and GEMM kernels for NVFP4 quantization
- **Auto-Updating**: Latest vLLM from main branch pulled at build time
- **Easy Scaling**: Extend with model-specific Dockerfiles for quick deployment

## Version 75 - Complete NVFP4 Integration (January 2026)

**BREAKTHROUGH PERFORMANCE: 65 tok/sec on Qwen3-30B-A3B-NVFP4**

This version delivers the **first complete FP4 integration** for vLLM on CUDA 13.0 with GB10 hardware:

### The Stack (Lowest to Highest)

1. **Hardware**: NVIDIA GB10 (Blackwell) - SM_121 compute capability
2. **CUDA 13.0.2**: Custom FP4 type implementation (nv_fp4_dummy.h) - 280+ lines
3. **CUTLASS**: FP4 GEMM and MoE kernels enabled for SM_121
4. **FlashInfer**: FP4-aware JIT compilation for attention kernels
5. **PyTorch**: Nightly with CUDA 13.0 support
6. **vLLM**: Main branch with SM_121 backend selection patches
7. **Model**: nvidia/Qwen3-30B-A3B-NVFP4 quantized model

### Performance Achievements

| Version | Format | Performance | Improvement |
|---------|--------|-------------|-------------|
| Previous NVFP4 | FP4 | 35 tok/sec | Baseline |
| FP8 Optimized | FP8 | 40 tok/sec | 1.14x |
| **v75 Complete** | **FP4** | **65 tok/sec** | **1.86x over NVFP4, 1.62x over FP8** |

### Quick Start - Best Performance

**1-liner for Qwen3-30B NVFP4 at 65 tok/sec:**

```bash
docker run -d --name vllm-qwen-nvfp4 --gpus all --network host \
  -e MODEL=nvidia/Qwen3-30B-A3B-NVFP4 -e PORT=8888 -e MAX_MODEL_LEN=4096 \
  -e GPU_MEMORY_UTIL=0.60 -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  avarok/vllm-dgx-spark:v75 serve
```

**Test inference:**
```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/Qwen3-30B-A3B-NVFP4","messages":[{"role":"user","content":"Hello!"}],"max_tokens":100}' \
  | jq -r '.choices[0].message.content'
```

### Technical Implementation

**What's New in v75:**
- Complete FP4 type system for CUDA 13.0 (3 types, 5 intrinsics, 9 operators)
- CUTLASS FP4 kernels enabled for SM_121 (GB10)
- FlashInfer JIT compilation with full FP4 operator support
- CCCL header patching for build-time compilation
- Post-install FlashInfer header patching for runtime JIT

**Key Files:**
- `nv_fp4_dummy.h` - Complete FP4 type implementation
- `patch_flashinfer_fp4.sh` - FlashInfer JIT support
- `patch_cccl_fp4.sh` - CCCL header patching
- `integrate_sm121_fp8_fix_v2.sh` - Backend selection for SM_121

**Docker Hub**: Available at `avarok/vllm-dgx-spark`

```bash
# Pull from Docker Hub
docker pull avarok/vllm-dgx-spark:v75
docker pull avarok/vllm-dgx-spark:latest
```

**Build Locally**:
```bash
cd /workspace/dgx-vllm-build
IMAGE_VERSION=75 ./build.sh
```

### Blog Article

Read the full story: [NVFP4 is Finally Here: from 40 (FP8) to 65tps Qwen3-Next!](https://blog.avarok.net/)

Previous optimization: [From 20 to 35 Tokens/Second: Optimizing NVFP4 Inference on Blackwell GB10](https://blog.avarok.net/from-20-to-35-tokens-second-optimizing-nvfp4-inference-on-blackwell-gb10-306a84bff467)

## Quick Start

### Build the Image

```bash
./build.sh
```

This will build the `dgx-vllm:latest` image locally and optionally on the remote worker node.

### Start Multi-Node Cluster (TP=2)

```bash
# On head node (10.10.10.1)
./start-cluster.sh
```

This automatically:
1. Starts Ray head node
2. Starts Ray worker on remote node (10.10.10.2)
3. Starts vLLM server with TP=2

### Test Inference

```bash
./test-inference.sh
```

### Stop Cluster

```bash
./stop-cluster.sh
```

## Manual Deployment

### Single Node (Single GPU)

```bash
docker run --rm -it --gpus all --network host \
  -e MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  -e TENSOR_PARALLEL_SIZE=1 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  dgx-vllm:latest serve
```

### Multi-Node (Distributed)

**On head node (10.10.10.1):**
```bash
# Start Ray head
docker-compose up -d ray-head

# Wait, then start vLLM server
docker-compose up -d vllm-server
```

**On worker node (10.10.10.2):**
```bash
docker-compose -f docker-compose.worker.yml up -d
```

## Configuration

### Environment Variables

**Common:**
- `MODEL`: HuggingFace model ID (required for serve mode)
- `PORT`: API port (default: 8888)
- `TENSOR_PARALLEL_SIZE`: Number of GPUs (default: 1)
- `MAX_MODEL_LEN`: Maximum sequence length (default: 131072)
- `GPU_MEMORY_UTIL`: GPU memory utilization (default: 0.75)

**Multi-Node:**
- `HEAD_IP`: Ray head node IP (required for distributed)
- `WORKER_IP`: Worker node IP (for ray-worker mode)
- `NCCL_SOCKET_IFNAME`: InfiniBand interface (default: enp1s0f0np0)

**Advanced:**
- `VLLM_EXTRA_ARGS`: Additional vLLM arguments
- `NUM_GPUS`: GPUs per node for Ray (default: 1)

### Container Modes

The entrypoint supports multiple modes:

1. **serve** (default): Start vLLM API server
2. **ray-head**: Start Ray head node only
3. **ray-worker**: Start Ray worker node only
4. **bash**: Interactive shell for debugging

## Architecture

### Multi-Node Setup

```
┌─────────────────────────────────────┐
│ DGX Node 1 (10.10.10.1)             │
│ ┌─────────────┐  ┌────────────────┐ │
│ │ Ray Head    │  │ vLLM Server    │ │
│ │ (Container) │  │ (Container)    │ │
│ │ GPU 0       │  │ TP rank 0      │ │
│ └─────────────┘  └────────────────┘ │
└─────────────────────────────────────┘
            │
            │ InfiniBand/RoCE
            │ NCCL + Gloo
            ▼
┌─────────────────────────────────────┐
│ DGX Node 2 (10.10.10.2)             │
│ ┌─────────────────────────────────┐ │
│ │ Ray Worker                      │ │
│ │ (Container)                     │ │
│ │ GPU 0 - TP rank 1               │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

### File Structure

```
dgx-vllm-build/
├── Dockerfile                  # Main image definition
├── docker-compose.yml          # Head node services
├── docker-compose.worker.yml   # Worker node service
├── entrypoint.sh              # Container entrypoint
├── vllm_patch.diff            # CMakeLists patch for GB10
├── triton_allocator.patch     # (Optional) Triton fix
├── build.sh                   # Build helper
├── start-cluster.sh           # Cluster startup
├── stop-cluster.sh            # Cluster shutdown
├── test-inference.sh          # Inference test
└── README.md                  # This file
```

## Extending the Base Image

Create model-specific images:

```dockerfile
FROM dgx-vllm:latest

# Pre-download model weights
RUN huggingface-cli download \
  DevQuasar/Qwen.Qwen3-Next-80B-A3B-Instruct-FP8-Dynamic \
  --local-dir /app/models/qwen3-next-80b

# Set default model
ENV MODEL=/app/models/qwen3-next-80b
ENV TENSOR_PARALLEL_SIZE=2

# Optional: pre-compile CUDA graphs
# RUN vllm compile --model ${MODEL} ...

CMD ["serve"]
```

## Troubleshooting

### Check Ray cluster status
```bash
docker exec dgx-vllm-head ray status
```

### View logs
```bash
# Head node
docker logs -f dgx-vllm-head
docker logs -f dgx-vllm-server

# Worker node (from head)
ssh nologik@10.10.10.2 'docker logs -f dgx-vllm-worker'
```

### Check NCCL communication
```bash
docker exec dgx-vllm-server bash -c 'printenv | grep NCCL'
```

### Verify InfiniBand
```bash
docker exec dgx-vllm-head ibv_devinfo
```

## Known Limitations

- Requires `network_mode: host` for InfiniBand access
- Requires `nvidia-docker` runtime with GPU passthrough
- Build time: ~30-60 minutes depending on hardware
- Image size: ~15-20GB

## Future Enhancements

- [ ] Add Triton allocator fix (currently commented out)
- [ ] Support for pipeline parallelism
- [ ] Pre-built model images
- [ ] Kubernetes deployment manifests
- [ ] Prometheus metrics export
- [ ] Multi-model serving

## License

Based on vLLM (Apache 2.0) and NVIDIA CUDA containers.
