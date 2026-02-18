# dgx-vllm: NVFP4 Inference on NVIDIA DGX Spark GB10

**59.9 tok/s** on Qwen3-Next-80B (NVFP4) — 54x faster than baseline, 2x faster than TensorRT-LLM.

### NVFP4 is faster than AWQ

The era of "just use AWQ INT4" is ending. On the same model, same hardware, same runtime — NVFP4 wins:

| Quantization | Throughput | vs AWQ |
|---|---|---|
| AWQ INT4 (group=32, Marlin dequant) | 34.9 tok/s | baseline |
| **NVFP4** (E2M1 + FP8 scales, Marlin dequant) | **40.2 tok/s** | **+15%** |
| **NVFP4 + MTP speculative decoding** | **59.9 tok/s** | **+72%** |

Both use Marlin W4A16 dequantization on the same vLLM v21 runtime. NVFP4's advantage comes from native FP4 tensor core format (no zero-points, simpler scale structure) and access to the model's built-in MTP draft head for speculative decoding. This image makes it work on GB10 — where the hardware FP4 convert instruction is missing.

```
┌─────────────────────────────────────────────────────────────────┐
│                     NVIDIA DGX Spark GB10                       │
│              Grace Blackwell Superchip (SM_121)                 │
│         119.7 GB Unified LPDDR5X @ 273 GB/s bandwidth          │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  FP4 Tensor Cores (mma.sync.m16n8k64.e2m1.e2m1)         │  │
│  │  FP8 Tensor Cores (mma.sync.m16n8k32.e4m3.e4m3)         │  │
│  │  ✗ Missing: cvt.rn.satfinite.e2m1x2.f32 (FP4 convert)  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Software Stack

This image bridges the gap between GB10's hardware FP4 tensor cores and vLLM's inference engine. Each layer solves a specific problem:

```
 Layer 7 ─ Model        Qwen3-Next-80B-A3B (MoE, 512 experts, NVFP4)
           │
 Layer 6 ─ vLLM V1      Inference engine: CUDA graphs, chunked prefill,
           │             FlashInfer attention, MoE routing
           │
 Layer 5 ─ Patches      Qwen3Next prefix fix, EMULATION backend fix
           │             Capability 121 → SM_120 kernel routing
           │             FlashInfer SM121 JIT patches
           │
 Layer 4 ─ CUTLASS      FP4 MoE GEMM (BlockScaled, Cooperative, 4 tiles)
           │             FP4 scaled_mm, FP8 scaled_mm (SM120 kernels)
           │
 Layer 3 ─ Software     ★ patch_nvfp4_utils_sw_e2m1.py ★
           │ E2M1        15-line device function replacing missing PTX
           │             Enables ALL 5 NVFP4 kernel files on SM121
           │
 Layer 2 ─ CUDA 13.0    nv_fp4_dummy.h (FP4 type definitions)
           │             CCCL header patches, FlashInfer JIT patches
           │
 Layer 1 ─ Base Image   nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
           │             PyTorch 2.10+cu130, Triton 3.6.0
           │
 Layer 0 ─ Hardware     GB10 GPU: SM_121, CC 12.1, ARM64 Grace CPU
```

### Why This Exists

GB10 has FP4 tensor cores for matrix multiplication, but is **missing the hardware instruction** (`cvt.rn.satfinite.e2m1x2.f32`) that converts activations from float32 to E2M1 format. Without this instruction, CUDA refuses to compile the quantization kernels, forcing a Python software fallback that runs at 1.1 tok/s.

Our fix: a 15-line C++ device function that performs the conversion in software, guarded by `#if __CUDA_ARCH__ == 1210`. This compiles all 5 NVFP4 kernels natively, enables CUDA graph capture (54 graphs), and delivers 59.9 tok/s with Marlin GEMM + MTP speculative decoding (84% draft acceptance rate).

---

## Quick Start

### Pull and Run

```bash
docker pull avarok/dgx-vllm-nvfp4-kernel:v21

docker run -d --name vllm-nvfp4 \
  --network host --gpus all --ipc=host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v /path/to/fix_mtp_nvfp4_exclusion.py:/tmp/fix_mtp_nvfp4_exclusion.py:ro \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash \
  avarok/dgx-vllm-nvfp4-kernel:v21 -c '
python3 /tmp/fix_mtp_nvfp4_exclusion.py && \
vllm serve nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 \
  --host 0.0.0.0 --port 8888 \
  --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --max-num-seqs 128 --attention-backend flashinfer \
  --kv-cache-dtype fp8 --no-enable-chunked-prefill \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":2}"
'
```

### Build Locally

```bash
git clone https://github.com/Avarok-Cybersecurity/dgx-vllm.git
cd dgx-vllm
docker build -t dgx-vllm:v21 .
```

### Test

```bash
# Wait ~7 min for startup (model load + torch.compile + CUDA graphs)
curl http://localhost:8888/v1/models

curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"GadflyII/Qwen3-Coder-Next-NVFP4",
       "messages":[{"role":"user","content":"Hello!"}],
       "max_tokens":100}' | jq -r '.choices[0].message.content'
```

---

## Performance

| Framework | Backend | Throughput | Notes |
|-----------|---------|-----------|-------|
| vLLM v20 (Python FP4 fallback) | — | 1.1 tok/s | `.item()` calls block CUDA graphs |
| vLLM v21 (eager, no CUDA graphs) | CUTLASS | 22 tok/s | Baseline without compilation |
| TensorRT-LLM v1.3.0rc2 | CUTLASS | 29.6 tok/s | NVIDIA's optimized runtime |
| vLLM v21 + CUDA graphs | CUTLASS | 35.0 tok/s | Software E2M1 + torch.compile |
| vLLM v21 + CUDA graphs | CUTLASS | 36.4 tok/s | + expandable_segments, 0.90 util |
| vLLM v21 + CUDA graphs | Marlin MoE | 39.5 tok/s | + Marlin MoE (W4A16 dequant) |
| vLLM v21 + CUDA graphs | Marlin (all) | 40.2 tok/s | Marlin for dense + MoE GEMM |
| vLLM v21 + MTP (1 token) | Marlin (all) | 55.4 tok/s | + MTP speculative decoding (84% accept) |
| **vLLM v21 + MTP (2 tokens)** | **Marlin (all)** | **59.9 tok/s** | **MTP with 2 spec tokens (84%/53% accept)** |
| Theoretical ceiling (single-token) | — | ~46 tok/s | 273 GB/s bandwidth limit |

Benchmarked on Qwen3-Next-80B-A3B-Instruct-NVFP4 (MoE, 512 experts, top-10 routing), single GB10 GPU, 200–500 token generations. MTP (Multi-Token Prediction) uses the model's built-in draft head to speculate future tokens, achieving ~1.84 accepted tokens per step (1 spec token) or ~2.42 tokens per step (2 spec tokens). This **exceeds the theoretical single-token memory-bandwidth ceiling** by generating multiple tokens per forward pass. Marlin dequantizes FP4 weights to FP16 at runtime, optimized for memory-bandwidth-bound batch=1 decode.

### Optimization History

| Commit | Improvement | Throughput | Spec Decode |
|--------|------------|-----------|-------------|
| `9e1cd69` | Software E2M1 + CUDA graphs | 35.0 tok/s | No |
| `e9ff094` | + Marlin MoE backend (W4A16 dequant) | 39.5 tok/s | No |
| `cf81980` | + Marlin dense GEMM | 40.2 tok/s | No |
| `83e7f1d` | + MTP speculative decoding (2 tokens) | **59.9 tok/s** | Yes (MTP, 2 tokens) |

See [OPTIMIZATIONS.md](OPTIMIZATIONS.md) for detailed analysis of each improvement and failed experiments.

---

## Stack Details

### Layer 0: Hardware — NVIDIA DGX Spark GB10

| Spec | Value |
|------|-------|
| GPU | NVIDIA GB10 (Blackwell consumer variant) |
| Compute Capability | 12.1 (SM_121) |
| Architecture | Grace Blackwell Superchip (ARM64 CPU + GPU) |
| Memory | 119.7 GB unified LPDDR5X |
| Bandwidth | 273 GB/s |
| FP4 Tensor Cores | Native `mma.sync.aligned.m16n8k64.f32.e2m1.e2m1` |
| FP4 Convert | **Missing** `cvt.rn.satfinite.e2m1x2.f32` |

### Layer 1: Base Image & PyTorch

| Component | Version |
|-----------|---------|
| Base | `nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04` |
| PyTorch | 2.10.0+cu130 (ARM64 wheel) |
| Triton | 3.6.0 (SM_121 support) |
| FlashInfer | Latest pre-release |
| XGrammar | Latest stable |
| Python | 3.12 |

### Layer 2: CUDA FP4 Type System

GB10's CUDA 13.0 CCCL headers reference `__nv_fp4_e2m1` but the type doesn't exist in the SDK. We provide:

- **`nv_fp4_dummy.h`** — Complete FP4 type implementation (280+ lines): 3 types, 5 intrinsics, 9 operators
- **`patch_cccl_fp4.sh`** — Patches CCCL headers to include the FP4 types at build time
- **`patch_flashinfer_fp4.sh`** — Patches FlashInfer headers for runtime JIT compilation

### Layer 3: Software E2M1 Conversion (The Key Innovation)

The `cvt.rn.satfinite.e2m1x2.f32` PTX instruction converts float32 to 4-bit E2M1 format. GB10 doesn't have it. Our software replacement:

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

This implements IEEE 754 round-to-nearest-even for E2M1, matching hardware behavior exactly. Applied by `patch_nvfp4_utils_sw_e2m1.py` to vLLM's `nvfp4_utils.cuh`.

**Files:**
- `patch_nvfp4_utils_sw_e2m1.py` — Patches `nvfp4_utils.cuh` with `#if __CUDA_ARCH__ == 1210` guards
- `cmake_patch_gb10_nvfp4_v6_full_kernels.sh` — CMake patch to compile all 5 NVFP4 kernel files

**NVFP4 kernel files compiled for SM121:**

| Kernel File | Purpose |
|-------------|---------|
| `nvfp4_quant_kernels.cu` | Activation quantization (BF16 → FP4) |
| `nvfp4_experts_quant.cu` | Per-expert MoE quantization |
| `activation_nvfp4_quant_fusion_kernels.cu` | SiLU + Mul + FP4 quantization |
| `nvfp4_blockwise_moe_kernel.cu` | CUTLASS FP4 MoE GEMM |
| `nvfp4_scaled_mm_sm120_kernels.cu` | CUTLASS FP4 dense GEMM |

### Layer 4: CUTLASS Kernels

Native FP4 and FP8 matrix multiplication via CUTLASS 4.x BlockScaled templates:

| Kernel | Scheduling | Tile Shapes | Pipeline |
|--------|-----------|-------------|----------|
| FP4 MoE GEMM | Cooperative (8 warps) | 256x128x128, 128x128x256, 128x256x128, 128x128x128 | 3-stage |
| FP4 scaled_mm | Cooperative | Auto-selected | 3-stage |
| FP8 scaled_mm | PyTorch fallback | `torch._scaled_mm` | N/A |

ClusterShape forced to 1x1x1 (GB10 has no multi-CTA clusters).

**Additional custom kernels:**
- `grouped_mm_gb10_native.cu` — GB10-optimized grouped GEMM with TMA and Pingpong scheduling
- `scaled_mm_sm121_fp8.cu` / `scaled_mm_blockwise_sm121_fp8.cu` — SM121 FP8 kernels

### Layer 5: vLLM Source Patches

Applied after `pip install -e .` to fix runtime issues:

| Patch | Problem | Fix |
|-------|---------|-----|
| `fix_qwen3_next_prefix.py` | Doubled `.in_proj_qkvz` prefix breaks weight loading | Remove duplicate prefix append in `create_qkvz_proj` |
| `fix_nvfp4_emulation_backend.py` | EMULATION backend produces garbled output | Fix scale format (LINEAR not swizzled) and global_scale inversion |
| `fix_capability_121_v112.py` | Capability 121 not routed to SM120 kernels | Route `>= 120 && < 130` to SM120 codepath |
| `fix_dispatcher_flag_v115.sh` | `ENABLE_SCALED_MM_SM120` undefined in dispatcher | Set compile definition for `scaled_mm_entry.cu` |
| `fix_cmake_sm120_archs_v113_corrected.sh` | Wrong CMake branch for CUDA 13.0+ | Fix line 533 (not 535) to include `12.1f` |
| `fix_flashinfer_e2m1_sm121.py` | FlashInfer JIT fails on SM121 (E2M1 convert) | Software E2M1 for FlashInfer runtime compilation |
| `fix_flashinfer_nvfp4_moe_backend.py` | FlashInfer MoE backend returns None for experts_cls | Return `k_cls` instead of `None` |
| `fix_mtp_nvfp4_exclusion.py` | MTP layers not excluded from NVFP4 quantization | Exclude all `mtp.*` layers when exclude list targets MTP |

### Layer 6: vLLM V1 Engine

Runtime configuration for NVFP4 inference:

| Feature | Status | Notes |
|---------|--------|-------|
| **MTP Spec Decode** | **Enabled (2 tokens)** | **+49% throughput (40.2→59.9), 84%/53% accept rate** |
| CUDA Graphs | 54 captured (35 piecewise + 19 full decode) | +64% throughput (22→36 tok/s) |
| torch.compile | Active (mode=VLLM_COMPILE) | vLLM Inductor backend, ~1% gain |
| FlashInfer Attention | Enabled | SM120 native attention kernels |
| Chunked Prefill | Disabled (required for spec decode) | Must use `--no-enable-chunked-prefill` with MTP |
| Prefix Caching | Supported | Via `--enable-prefix-caching` |
| MoE Backend | **Marlin** (recommended) or CUTLASS | Marlin: W4A16 dequant, +13% over CUTLASS |

### Layer 7: Model

| Property | Value |
|----------|-------|
| Model | Qwen3-Next-80B-A3B-Instruct |
| Parameters | 80B total, ~3B active per token |
| Architecture | Hybrid (Attention + Mamba SSM) |
| Experts | 512 (top-10 routing) |
| Quantization | NVFP4 (E2M1 weights + FP8 E4M3 block scales) |
| Format | `compressed-tensors` |
| Context | 131K tokens (model), limited by GPU memory |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | (required) | HuggingFace model ID |
| `PORT` | 8888 | API server port |
| `GPU_MEMORY_UTIL` | 0.75 | GPU memory fraction (0.0–1.0) |
| `MAX_MODEL_LEN` | 131072 | Maximum context length |
| `MAX_NUM_SEQS` | 128 | Maximum concurrent sequences |
| `TENSOR_PARALLEL_SIZE` | 1 | Number of GPUs |
| `VLLM_EXTRA_ARGS` | "" | Additional vLLM CLI arguments |
| `VLLM_USE_FLASHINFER_MOE_FP4` | 0 | Use FlashInfer MoE (set 0 for CUTLASS/Marlin) |
| `VLLM_TEST_FORCE_FP8_MARLIN` | 0 | Force Marlin MoE backend (+13% throughput) |
| `VLLM_NVFP4_GEMM_BACKEND` | (auto) | Set `marlin` for Marlin dense GEMM (+2% over FlashInfer CUTLASS) |
| `PYTORCH_CUDA_ALLOC_CONF` | (unset) | Set `expandable_segments:True` to reduce fragmentation |

### Container Modes

```bash
docker run ... dgx-vllm:v21 serve        # Start vLLM API server (default)
docker run ... dgx-vllm:v21 ray-head     # Start Ray head node
docker run ... dgx-vllm:v21 ray-worker   # Start Ray worker node
docker run ... dgx-vllm:v21 bash         # Interactive shell
```

### Recommended Launch with MTP Speculative Decoding (~60 tok/s)

```bash
docker run -d --name vllm-nvfp4 \
  --network host --gpus all --ipc=host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v /path/to/fix_mtp_nvfp4_exclusion.py:/tmp/fix_mtp_nvfp4_exclusion.py:ro \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash \
  avarok/dgx-vllm-nvfp4-kernel:v21 -c '
python3 /tmp/fix_mtp_nvfp4_exclusion.py && \
vllm serve nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 \
  --host 0.0.0.0 --port 8888 \
  --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --max-num-seqs 128 --attention-backend flashinfer \
  --kv-cache-dtype fp8 --no-enable-chunked-prefill \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":2}"
'
```

**Note:** MTP requires `fix_mtp_nvfp4_exclusion.py` to patch a bug where MTP layers are incorrectly quantized to NVFP4. The patch excludes all `mtp.*` layers from quantization since they are stored as BF16 in the checkpoint. Chunked prefill must be disabled for speculative decoding.

### Launch without MTP (~40 tok/s)

```bash
docker run -d --name vllm-nvfp4 \
  --network host --gpus all --ipc=host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e MODEL=nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 \
  -e PORT=8888 -e GPU_MEMORY_UTIL=0.90 \
  -e MAX_MODEL_LEN=4096 -e MAX_NUM_SEQS=128 \
  -e VLLM_EXTRA_ARGS="--attention-backend flashinfer --kv-cache-dtype fp8" \
  avarok/dgx-vllm-nvfp4-kernel:v21 serve
```

---

## File Reference

### Build Files (Layer 2–3)

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build: base → patches → compile → runtime fixes |
| `nv_fp4_dummy.h` | FP4 type definitions for CUDA 13.0 |
| `patch_cccl_fp4.sh` | Patches CCCL headers with FP4 types |
| `patch_flashinfer_fp4.sh` | Patches FlashInfer headers for FP4 JIT |
| `patch_nvfp4_utils_sw_e2m1.py` | **Software E2M1 for SM121** (the key fix) |
| `cmake_patch_gb10_nvfp4_v6_full_kernels.sh` | CMake patch for all 5 NVFP4 kernels |

### Kernel Files (Layer 4)

| File | Purpose |
|------|---------|
| `grouped_mm_gb10_native.cu` | GB10-optimized grouped GEMM (TMA + Pingpong) |
| `grouped_mm_gb10_native_v109.cu` | V109 variant of GB10 grouped GEMM |
| `scaled_mm_sm121_fp8.cu` | SM121 FP8 scaled matmul |
| `scaled_mm_blockwise_sm121_fp8.cu` | SM121 FP8 blockwise scaled matmul |
| `scaled_mm_c3x_sm121.cu` | CUTLASS 3.x SM121 kernel |
| `nvfp4_stubs.cu` | Historical: stub functions (no longer used in v21) |
| `cutlass_nvfp4/` | Custom CUTLASS FP4 extension (headers + kernels) |

### Integration Scripts (Layer 4–5)

| File | Purpose |
|------|---------|
| `integrate_gb10_sm121.sh` | Integrates SM121 native kernels into vLLM |
| `integrate_gb10_native_v109.sh` | Integrates V109 GB10 grouped GEMM |
| `integrate_cuda_fp4_extension.sh` | Integrates custom CUTLASS FP4 extension |
| `integrate_sm121_fp8_fix_v2.sh` | FP8 backend selection fix |

### Runtime Patches (Layer 5)

| File | Purpose |
|------|---------|
| `fix_qwen3_next_prefix.py` | Fix Qwen3Next weight loading prefix |
| `fix_nvfp4_emulation_backend.py` | Fix EMULATION backend dequantization |
| `fix_capability_121_v112.py` | Route CC 121 → SM120 kernels |
| `fix_cmake_sm120_archs_v113_corrected.sh` | CMake arch list fix for CUDA 13.0+ |
| `fix_dispatcher_flag_v115.sh` | Enable SM120 flag in dispatcher |
| `fix_flashinfer_e2m1_sm121.py` | FlashInfer SM121 JIT E2M1 patches |
| `fix_flashinfer_nvfp4_moe_backend.py` | FlashInfer MoE backend return fix |
| `fix_mtp_nvfp4_exclusion.py` | Exclude all MTP layers from NVFP4 quantization |

### Runtime Configuration

| File | Purpose |
|------|---------|
| `entrypoint.sh` | Container entrypoint (serve/ray-head/ray-worker/bash) |
| `E=512,N=512,...fp8_w8a8.json` | GB10-tuned MoE Triton config (+65.7%) |

---

## Build History

| Version | Change | Throughput |
|---------|--------|-----------|
| v75 | Complete NVFP4 integration (Qwen3-30B) | 65 tok/s (30B model) |
| v109 | GB10 native grouped GEMM kernel | — |
| v112–v115 | Capability 121 routing (3-part fix chain) | — |
| v118 | Meta backend for torch.compile | — |
| v134 | Disable torch.compile for NVFP4 (no-op, see below) | — |
| v20 | Python software FP4 quant (Qwen3-80B NVFP4) | 1.1 tok/s |
| v21 | Software E2M1 in C++ + CUDA graphs | 35.0 tok/s |
| v21+FlashInfer | FlashInfer SM121 JIT patches + MoE backend fix | 35.0 tok/s (FlashInfer fused MoE: garbled) |
| v21+Marlin MoE | Marlin MoE backend (W4A16 dequant) + expandable_segments | 39.5 tok/s |
| v21+Marlin (all) | Marlin for both dense + MoE GEMM | 40.2 tok/s |
| v21+MTP (1 token) | MTP speculative decoding (84% accept) | 55.4 tok/s |
| **v21+MTP (2 tokens)** | **MTP with 2 speculative tokens (84%/53% accept)** | **59.9 tok/s** |

---

## License

Built on [vLLM](https://github.com/vllm-project/vllm) (Apache 2.0) and NVIDIA CUDA containers.

Open source at [github.com/Avarok-Cybersecurity/dgx-vllm](https://github.com/Avarok-Cybersecurity/dgx-vllm).

Built by [Avarok Cybersecurity](https://github.com/Avarok-Cybersecurity) with [Claude Code](https://claude.ai/claude-code).
