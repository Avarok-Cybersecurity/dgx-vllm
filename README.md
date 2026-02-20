<p align="center">
  <h1 align="center">dgx-vllm</h1>
  <p align="center">
    <strong>NVFP4 inference on NVIDIA DGX Spark GB10 — finally faster than AWQ</strong>
  </p>
  <p align="center">
    <a href="https://hub.docker.com/r/avarok/dgx-vllm-nvfp4-kernel"><img src="https://img.shields.io/badge/Docker%20Hub-v22-blue?logo=docker" alt="Docker Hub"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License"></a>
    <a href="https://github.com/Avarok-Cybersecurity/dgx-vllm"><img src="https://img.shields.io/badge/Platform-DGX%20Spark%20GB10-76B900?logo=nvidia" alt="Platform"></a>
  </p>
</p>

---

The first open-source vLLM image to unlock full NVFP4 performance on NVIDIA DGX Spark. NVFP4 is **~20% faster than AWQ INT4** — and with MTP speculative decoding, it peaks at **111.9 tok/s** from an 80B-parameter model on a single desktop GPU.

| Configuration | Avg Decode | Peak Decode | vs AWQ |
|---|---:|---:|---:|
| AWQ INT4 (NVIDIA official) | ~34 tok/s | 38.2 tok/s | baseline |
| AWQ INT4 (Avarok) | ~36 tok/s | 39.7 tok/s | +6% |
| NVFP4 (NVIDIA official) | ~36 tok/s | 40.2 tok/s | +0% |
| **NVFP4 (Avarok)** | **~42 tok/s** | **47.1 tok/s** | **+20%** |
| **NVFP4 + MTP (Avarok)** | **~67 tok/s** | **111.9 tok/s** | **~2x** |

> Read the full benchmark write-up: **[NVFP4_BREAKTHROUGH_DGX_SPARK.md](NVFP4_BREAKTHROUGH_DGX_SPARK.md)**

---

## Table of Contents

- [Quick Start](#-quick-start)
- [Why This Exists](#-why-this-exists)
- [Performance](#-performance)
- [Architecture](#-architecture)
- [Configuration](#-configuration)
- [File Reference](#-file-reference)
- [Build History](#-build-history)
- [License](#-license)

---

## Quick Start

### Pull and run (~67 tok/s with MTP)

```bash
docker pull avarok/dgx-vllm-nvfp4-kernel:v22

docker run -d --name vllm-nvfp4 \
  --network host --gpus all --ipc=host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e MODEL=nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 \
  -e PORT=8888 -e GPU_MEMORY_UTIL=0.90 \
  -e MAX_MODEL_LEN=65536 -e MAX_NUM_SEQS=128 \
  -e VLLM_EXTRA_ARGS="--attention-backend flashinfer --kv-cache-dtype fp8" \
  avarok/dgx-vllm-nvfp4-kernel:v22 serve
```

To enable MTP speculative decoding, add to `VLLM_EXTRA_ARGS`:
```
--speculative-config '{"method":"mtp","num_speculative_tokens":2}' --no-enable-chunked-prefill
```

> **Note:** MTP requires `fix_mtp_nvfp4_exclusion.py` to be run inside the container before serving. Mount it and prepend to the entrypoint, or apply it at build time.

### Build locally

```bash
git clone https://github.com/Avarok-Cybersecurity/dgx-vllm.git
cd dgx-vllm
docker build -t dgx-vllm:v22 .
```

### Verify

```bash
# Wait ~10 min for startup (model load + torch.compile + CUDA graphs)
curl http://localhost:8888/v1/models

curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4",
       "messages":[{"role":"user","content":"Hello!"}],
       "max_tokens":100}' | jq -r '.choices[0].message.content'
```

---

## Why This Exists

The DGX Spark's GB10 GPU has native FP4 tensor cores — but its SM 12.1 compute capability is **missing a critical PTX instruction** (`cvt.rn.satfinite.e2m1x2.f32`) required for NVFP4 quantization. Without it:

```
                        NVIDIA DGX Spark GB10
                Grace Blackwell Superchip (SM_121)
           119.7 GB Unified LPDDR5X @ 273 GB/s bandwidth

    FP4 Tensor Cores    mma.sync.m16n8k64.e2m1.e2m1     [WORKS]
    FP8 Tensor Cores    mma.sync.m16n8k32.e4m3.e4m3     [WORKS]
    FP4 Convert         cvt.rn.satfinite.e2m1x2.f32     [MISSING]
```

- **Vanilla vLLM** falls back to Python — **1.1 tok/s** (unusable)
- **NVIDIA's official image** uses `flashinfer-cutlass` — **~36 tok/s** (same as AWQ, no advantage)
- **TensorRT-LLM** hits a hard ceiling — **29.6 tok/s** (CUTLASS cooperative-only scheduling)

We wrote a 15-line software E2M1 conversion function, enabled the Marlin MoE backend, and patched SM 12.1 capability routing. NVFP4 now runs at **~42 tok/s** without speculative decoding — 20% faster than AWQ. With MTP, it reaches **~67 tok/s average** and peaks at **111.9 tok/s**.

---

## Performance

### Throughput progression

| Stage | Backend | Throughput | Notes |
|-------|---------|----------:|-------|
| Vanilla vLLM (broken) | Python fallback | 1.1 tok/s | Missing PTX causes catastrophic fallback |
| + Software E2M1 | CUTLASS | 35.0 tok/s | 32x improvement from one patch |
| + Marlin MoE | Marlin | 40.2 tok/s | +15% from backend switch |
| + torch.compile + pin vLLM | Marlin | **~42 tok/s** | **v22 baseline, +20% vs AWQ** |
| + MTP (2 tokens) | Marlin | **~67 tok/s** | **Peak 111.9 tok/s** |
| Theoretical ceiling | — | ~46 tok/s | 273 GB/s bandwidth limit (single-token) |

MTP exceeds the single-token bandwidth ceiling by generating ~2.4 tokens per forward pass (63-89% draft acceptance).

### Optimization commits

| Commit | Change | Throughput |
|--------|--------|----------:|
| `9e1cd69` | Software E2M1 + CUDA graphs | 35.0 tok/s |
| `e9ff094` | Marlin MoE backend | 39.5 tok/s |
| `cf81980` | Marlin dense GEMM | 40.2 tok/s |
| `83e7f1d` | MTP speculative decoding (2 tokens) | 59.9 tok/s |
| `4481506` | **v22: pin vLLM rev, re-enable torch.compile** | **~67 tok/s avg** |

See [OPTIMIZATIONS.md](OPTIMIZATIONS.md) for detailed analysis including failed experiments.

---

## Architecture

This image bridges GB10's hardware FP4 tensor cores and vLLM's inference engine through 7 layers of patches and integrations:

```
 Layer 7  Model        Qwen3-Next-80B-A3B (MoE, 512 experts, NVFP4)
            |
 Layer 6  vLLM V1      CUDA graphs, chunked prefill, FlashInfer attention,
            |           MoE routing, MTP speculative decoding
            |
 Layer 5  Patches      Qwen3Next prefix fix, EMULATION backend fix
            |           Capability 121 -> SM_120 routing, FlashInfer JIT
            |
 Layer 4  CUTLASS      FP4 MoE GEMM (BlockScaled, Cooperative, 4 tiles)
            |           FP4/FP8 scaled_mm (SM120 kernels)
            |
 Layer 3  Software     patch_nvfp4_utils_sw_e2m1.py
            |  E2M1     15-line device function replacing missing PTX
            |
 Layer 2  CUDA 13.0    nv_fp4_dummy.h (FP4 type definitions)
            |           CCCL + FlashInfer header patches
            |
 Layer 1  Base Image   nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
            |           PyTorch 2.10+cu130, Triton 3.6.0
            |
 Layer 0  Hardware     GB10 GPU: SM_121, 119.7 GB LPDDR5X, ARM64 Grace
```

### The key fix: Software E2M1 conversion

```cpp
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1210
__device__ __forceinline__ uint8_t _sw_float_to_e2m1(float x) {
  uint8_t sign = (uint8_t)((__float_as_uint(x) >> 28) & 8u);
  float ax = fabsf(x);
  uint8_t mag;
  if      (ax <= 0.25f)  mag = 0;  // 0.0
  else if (ax <  0.75f)  mag = 1;  // 0.5
  else if (ax <= 1.25f)  mag = 2;  // 1.0
  else if (ax <  1.75f)  mag = 3;  // 1.5
  else if (ax <= 2.5f)   mag = 4;  // 2.0
  else if (ax <  3.5f)   mag = 5;  // 3.0
  else if (ax <= 5.0f)   mag = 6;  // 4.0
  else                    mag = 7;  // 6.0 (satfinite)
  return sign | mag;
}
#endif
```

IEEE 754 round-to-nearest-even for E2M1, matching hardware behavior exactly. Applied to all 5 NVFP4 kernel files via `patch_nvfp4_utils_sw_e2m1.py`.

### Hardware specs

| Spec | Value |
|------|-------|
| GPU | NVIDIA GB10 (Blackwell) |
| Compute Capability | 12.1 (SM_121) |
| Architecture | Grace Blackwell Superchip (ARM64 + GPU) |
| Memory | 119.7 GB unified LPDDR5X |
| Bandwidth | 273 GB/s |
| FP4 Tensor Cores | `mma.sync.aligned.m16n8k64.f32.e2m1.e2m1` |
| FP4 Convert | **Missing** `cvt.rn.satfinite.e2m1x2.f32` |

### Software stack

| Component | Version |
|-----------|---------|
| Base image | `nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04` |
| vLLM | v0.16.0rc2 (pinned at `3b30e6150`) |
| PyTorch | 2.10.0+cu130 |
| Triton | 3.6.0 |
| FlashInfer | Latest pre-release |
| Python | 3.12 |

### Model

| Property | Value |
|----------|-------|
| Model | Qwen3-Next-80B-A3B-Instruct-NVFP4 |
| Parameters | 80B total, ~3B active per token |
| Architecture | Hybrid (Attention + Mamba SSM), MoE (512 experts, top-10) |
| Quantization | NVFP4 (E2M1 weights + FP8 E4M3 block scales) |
| Context | Up to 128K tokens |

---

## Configuration

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | *(required)* | HuggingFace model ID |
| `PORT` | `8888` | API server port |
| `GPU_MEMORY_UTIL` | `0.75` | GPU memory fraction (0.0–1.0) |
| `MAX_MODEL_LEN` | `131072` | Maximum context length |
| `MAX_NUM_SEQS` | `128` | Maximum concurrent sequences |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs |
| `VLLM_EXTRA_ARGS` | `""` | Additional vLLM CLI arguments |
| `VLLM_USE_FLASHINFER_MOE_FP4` | `0` | Set `0` for CUTLASS/Marlin MoE backend |
| `VLLM_TEST_FORCE_FP8_MARLIN` | `0` | Set `1` to force Marlin MoE backend (+17%) |
| `VLLM_NVFP4_GEMM_BACKEND` | *(auto)* | Set `marlin` for Marlin dense GEMM |
| `PYTORCH_CUDA_ALLOC_CONF` | *(unset)* | Set `expandable_segments:True` to reduce fragmentation |

### Container modes

```bash
docker run ... avarok/dgx-vllm-nvfp4-kernel:v22 serve        # vLLM API server (default)
docker run ... avarok/dgx-vllm-nvfp4-kernel:v22 ray-head     # Ray head node
docker run ... avarok/dgx-vllm-nvfp4-kernel:v22 ray-worker   # Ray worker node
docker run ... avarok/dgx-vllm-nvfp4-kernel:v22 bash         # Interactive shell
```

---

## File Reference

### Build & packaging

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build: base, patches, compile, runtime fixes |
| `build.sh` | Build script (local or remote node) |
| `push.sh` | Push to Docker Hub |
| `entrypoint.sh` | Container entrypoint (serve / ray-head / ray-worker / bash) |

### CUDA FP4 type system

| File | Purpose |
|------|---------|
| `nv_fp4_dummy.h` | FP4 type definitions for CUDA 13.0 (3 types, 5 intrinsics, 9 operators) |
| `patch_cccl_fp4.sh` | Patches CCCL headers to include FP4 types |
| `patch_flashinfer_fp4.sh` | Patches FlashInfer headers for FP4 JIT |

### Software E2M1 (the key fix)

| File | Purpose |
|------|---------|
| `patch_nvfp4_utils_sw_e2m1.py` | Software E2M1 conversion for SM121 — the patch that makes NVFP4 work |
| `cmake_patch_gb10_nvfp4_v6_full_kernels.sh` | CMake patch to compile all 5 NVFP4 kernel files |

### SM121 kernels

| File | Purpose |
|------|---------|
| `grouped_mm_gb10_native.cu` | GB10-optimized grouped GEMM (TMA + Pingpong) |
| `grouped_mm_gb10_native_v109.cu` | V109 variant of GB10 grouped GEMM |
| `scaled_mm_sm121_fp8.cu` | SM121 FP8 scaled matmul |
| `scaled_mm_blockwise_sm121_fp8.cu` | SM121 FP8 blockwise scaled matmul |
| `scaled_mm_c3x_sm121.cu` | CUTLASS 3.x SM121 kernel |
| `cutlass_nvfp4/` | Custom CUTLASS FP4 extension (headers, kernels, tests) |

### Integration scripts

| File | Purpose |
|------|---------|
| `integrate_gb10_sm121.sh` | Integrates SM121 native kernels into vLLM |
| `integrate_gb10_native_v109.sh` | Integrates V109 GB10 grouped GEMM |
| `integrate_cuda_fp4_extension.sh` | Integrates custom CUTLASS FP4 extension |
| `integrate_sm121_fp8_fix_v2.sh` | FP8 backend selection fix |

### Runtime patches

| File | Problem | Fix |
|------|---------|-----|
| `fix_qwen3_next_prefix.py` | Doubled weight loading prefix | Remove duplicate prefix in `create_qkvz_proj` |
| `fix_nvfp4_emulation_backend.py` | Garbled EMULATION output | Fix scale format + global_scale inversion |
| `fix_capability_121_v112.py` | SM121 not routed to SM120 kernels | Route CC 121 to SM120 codepath |
| `fix_cmake_sm120_archs_v113_corrected.sh` | Wrong CMake branch for CUDA 13.0+ | Fix arch list to include `12.1f` |
| `fix_dispatcher_flag_v115.sh` | `ENABLE_SCALED_MM_SM120` undefined | Set compile definition for dispatcher |
| `fix_flashinfer_e2m1_sm121.py` | FlashInfer JIT fails on SM121 | Software E2M1 for FlashInfer runtime |
| `fix_flashinfer_nvfp4_moe_backend.py` | MoE backend returns `None` | Return `k_cls` correctly |
| `fix_mtp_nvfp4_exclusion.py` | MTP layers incorrectly quantized | Exclude `mtp.*` layers from NVFP4 |

### Runtime configuration

| File | Purpose |
|------|---------|
| `E=512,N=512,...fp8_w8a8.json` | GB10-tuned MoE Triton config (+65.7% vs default) |

---

## Build History

| Version | Change | Throughput |
|---------|--------|----------:|
| v20 | Python software FP4 quant (Qwen3-80B) | 1.1 tok/s |
| v21 | Software E2M1 in C++ + CUDA graphs | 35.0 tok/s |
| v21 + Marlin | Marlin MoE + dense GEMM backends | 40.2 tok/s |
| v21 + MTP | MTP speculative decoding (2 tokens) | 59.9 tok/s |
| **v22** | **Pin vLLM, re-enable torch.compile, 64K benchmarks** | **~67 tok/s avg** |

---

## License

Built on [vLLM](https://github.com/vllm-project/vllm) (Apache 2.0) and NVIDIA CUDA containers.

Open source at [github.com/Avarok-Cybersecurity/dgx-vllm](https://github.com/Avarok-Cybersecurity/dgx-vllm).

Built by [Avarok](https://github.com/Avarok-Cybersecurity) with [Claude Code](https://claude.ai/claude-code).
