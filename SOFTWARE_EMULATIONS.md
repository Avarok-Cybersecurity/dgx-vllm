# Software Emulations & Hardware Replacement Paths (SM121 / DGX Spark GB10)

SM121 (NVIDIA GB10, DGX Spark) is a desktop Blackwell chip that lacks several features
present on datacenter Blackwell (SM100/SM120). This document catalogues every software
emulation, fallback, or workaround in the `dgx-vllm` inference image and identifies
which ones could be replaced with native hardware paths.

---

## Summary Table

| Rank | Software Emulation | Impact | HW Fix Available? | Implemented in Image? | Effort | Files |
|------|-------------------|--------|--------------------|-----------------------|--------|-------|
| 1 | FP8 CUTLASS scaled_mm disabled → `torch._scaled_mm` | HIGH | YES — but cuBLAS is 7-10% faster | **No** (CUTLASS slower than cuBLAS on SM121) | N/A | `integrate_sm121_fp8_fix_v2.sh` |
| 2 | Software E2M1 float→FP4 conversion (branchless) | HIGH | MAYBE — need latest ptxas for `sm_121a` | **Yes** — branchless integer comparison sum, 32x recovery from broken state | Done | `patch_nvfp4_utils_sw_e2m1.py`, `fix_flashinfer_e2m1_sm121.py` |
| 3 | FP4 GEMV via scalar CUDA cores + LUT dequant | MED-HIGH | YES — `mma.sync` FP4 tensor cores, cuSPARSELt | **No** (v8 kernel uses CUDA cores) | High | `sparse_fp4_kernel/sparse_fp4_v8.cu` |
| 4 | NVFP4 full-emulation backend (Python dequant+matmul) | MEDIUM | YES — should never activate | **Yes** — Marlin/CUTLASS backends active by default | Low | `fix_nvfp4_emulation_backend.py` |
| 5 | `nv_fp4_dummy.h` software FP4 type system | MEDIUM | PARTIAL — awaiting NVIDIA `cuda_fp4.h` | **Yes** — dummy types injected at build | Low | `nv_fp4_dummy.h`, `patch_cccl_fp4.sh` |
| 6 | CUTLASS MoE Cooperative-only (no Pingpong scheduling) | MEDIUM | NO — requires NVIDIA upstream | **N/A** (hardcoded in CUTLASS) | Blocked | CUTLASS `sm1xx_common.inl` |
| 7 | tcgen05 env vars (misleading; mma.sync IS correct HW) | LOW-MED | N/A — mma.sync is the HW path | **Yes** — mma.sync used correctly | None | `cutlass_nvfp4/nvfp4_tcgen05_ptx_v2.cuh` |
| 8 | BF16→FP16 via FP32 intermediate (2-step) | LOW | YES — `__bfloat162half()` direct | **No** (uses `__bfloat162float` + `__float2half`) | Low | `sparse_fp4_kernel/sparse_fp4_v8.cu` |
| 9 | SiLU via `expf()` | LOW | N/A — already uses hardware SFU | **Yes** — hardware `ex2.approx.f32` | None | `sparse_fp4_kernel/sparse_fp4_v8.cu` |
| 10 | FlashInfer fused MoE TMA corruption | BLOCKED | NO — requires FlashInfer upstream | **Disabled** (`VLLM_USE_FLASHINFER_MOE_FP4=0`) | Blocked | `fix_flashinfer_e2m1_sm121.py` |
| 11 | FlashInfer `arch_condition.h` bypass | LOW | N/A — build-system fix only | **Yes** — patched at build | None | `patch_flashinfer_fp4.sh` |
| 12 | CC 121→120 routing patches | LOW | N/A — routing, not compute | **Yes** — dispatchers patched | None | `fix_capability_121_v112.py` |

---

## What the Image Already Replaces with Hardware Paths

These software fallbacks have been **successfully replaced** in the `dgx-vllm` image:

| Problem | Software Fallback | Hardware Path in Image |
|---------|------------------|----------------------|
| SM121 not recognized by vLLM | Crash / unsupported arch | CC 121→120 routing patches dispatch to SM120 CUTLASS kernels |
| FP4 type system missing (`__nv_fp4_e2m1`) | N/A (compilation failure) | `nv_fp4_dummy.h` provides software types that compile against CUTLASS headers |
| E2M1 PTX instruction missing on SM121 | Broken: 1.1 tok/s (vLLM crashed at runtime) | Software E2M1 via bit manipulation — **32x recovery** to 36.4 tok/s |
| NVFP4 MoE backend returns `None` | Python crash at model load | `fix_flashinfer_nvfp4_moe_backend.py` routes to CUTLASS MoE |
| FlashInfer arch check rejects SM121 | Compilation failure | `patch_flashinfer_fp4.sh` adds GB10 exception |
| Marlin MoE not default | Users get 36.4 tok/s (CUTLASS) | `ENV VLLM_NVFP4_GEMM_BACKEND=marlin` baked in → **59.9 tok/s** |
| NVFP4 emulation backend active | ~5 tok/s (Python path) | Marlin + CUTLASS backends handle all NVFP4 GEMM natively |
| SiLU activation | N/A | Already hardware SFU (`expf` → `ex2.approx.f32` PTX) |
| FP4 tensor core MMA | Software on SM100 tcgen05 path | `mma.sync.aligned` FP4 (correct SM121 instruction, not tcgen05) |

---

## Detailed Analysis by Rank

### Rank 1: FP8 CUTLASS scaled_mm — Disabled at Python Level

**Status**: BENCHMARKED — CUTLASS is 7-10% slower than cuBLAS on SM121. Keeping cuBLAS.

The image builds native SM121 CUTLASS FP8 kernels (`scaled_mm_sm121_fp8.cu`) with three
tile configurations optimized for GB10:

- Default: 128×256×128 (general purpose)
- Small: 128×128×128 (better occupancy for small M×N)
- Large: 128×256×128 (throughput for large problems)

These use `cutlass_3x_gemm_sm100` with `Sm100` arch tag, Cooperative scheduling,
and ClusterShape 1×1×1. The C++ dispatch path (`scaled_mm_sm121_fp8_dispatch.cuh`)
with adaptive kernel selection is fully compiled and linked.

**However**, `integrate_sm121_fp8_fix_v2.sh` patches the Python-level
`CutlassFP8ScaledMMLinearKernel.is_supported()` to return `False` for CC 121:

```python
if compute_capability is not None and compute_capability == 121:
    return False, "SM_121 (GB10) not supported by CUTLASS - using PyTorch fallback"
```

This forces all FP8 matmuls through `torch._scaled_mm` (likely cuBLAS under the hood).
The custom CUTLASS kernels with GB10-specific tile shapes and adaptive dispatch are
never executed.

**Impact**: Affects all FP8 linear layers (Mamba projections, attention QKV, embeddings).
These are 82% of decode data reads. Even a 5% kernel-level improvement from CUTLASS
vs cuBLAS would translate to measurable E2E gains.

**Investigation findings** (2026-02-21):

The C++ dispatch chain is **100% complete** and ready to use:

1. `scaled_mm_c3x_sm121.cu` routes SM121 calls to `cutlass_scaled_mm_sm100_fp8` (per-tensor)
   and `cutlass_scaled_mm_blockwise_sm100_fp8` (blockwise) — both SM100 kernels compiled
   for 12.1f gencode
2. The C++ dispatcher (`scaled_mm_entry.cu`) is compiled with `ENABLE_SCALED_MM_SM120=1`
   and routes `version_num >= 120 && < 130` to the SM120 codepath
3. Python `_custom_ops.py` already accepts CC 121 in capability sets
4. The SM121-specific dispatch headers (`scaled_mm_sm121_fp8_dispatch.cuh`) with GB10-optimized
   tile shapes (128×256×128, 128×128×128) are compiled but NOT actually called — the entry
   point goes straight to SM100 kernels instead

The **only gate** is lines 137-139 of the Dockerfile:
```dockerfile
COPY integrate_sm121_fp8_fix_v2.sh ...
RUN ... /workspace/dgx-vllm-build/integrate_sm121_fp8_fix_v2.sh
```

**Benchmark result (v23, 2026-02-21)**: CUTLASS SM100 kernels on SM121 are **7-10% slower**
than `torch._scaled_mm` (cuBLAS):

| Config | cuBLAS Decode | CUTLASS Decode | Delta |
|--------|--------------|----------------|-------|
| 128/128 | 46.9 tok/s | 42.8 tok/s | -8.7% |
| 1024/128 | 47.1 tok/s | 42.1 tok/s | -10.6% |
| 128/1024 | 40.4 tok/s | 37.6 tok/s | -6.9% |

**Root cause**: SM100 CUTLASS tile shapes (128×256×128) are not optimized for GB10's
48 SMs and 301 GB/s LPDDR5X. cuBLAS has SM121-specific auto-tuning that produces
better configurations. The CUTLASS disable is **correctly in place**.

**Future opportunity**: SM121-specific CUTLASS tile configs (smaller tiles for 48 SMs,
bandwidth-optimized scheduling) could potentially match or beat cuBLAS. Would require
custom kernel tuning.

---

### Rank 2: Software E2M1 Conversion

**Status**: Optimized — branchless integer comparison sum. Hardware instruction confirmed missing on SM121.

SM121 lacks the `cvt.rn.satfinite.e2m1x2.f32` PTX instruction for float→E2M1 (FP4)
conversion. Three separate software E2M1 implementations are deployed:

1. **vLLM `nvfp4_utils.cuh`** — Branchless comparison sum with round-to-nearest-even
   (`patch_nvfp4_utils_sw_e2m1.py`)
2. **FlashInfer `quantization_utils.cuh`** — Branchless comparison sum
   (`fix_flashinfer_e2m1_sm121.py`)
3. **Host FP4 types** — Threshold-based conversion intrinsics (`nv_fp4_dummy.h`)

The software path runs during activation quantization (every forward pass for NVFP4),
not during the GEMM itself (which uses native `mma.sync` E2M1 tensor core operations).

**Optimization applied**: The original 7-branch `if/else if` chain caused warp divergence
when threads quantized values spanning different E2M1 buckets. Replaced with a branchless
sum of boolean integer comparisons on the IEEE 754 bit pattern:

```cuda
uint32_t abits = __float_as_uint(x) & 0x7FFFFFFFu;  // abs via bit clear
uint8_t mag = (abits >  0x3E800000u)   // > 0.25
            + (abits >= 0x3F400000u)   // >= 0.75
            + (abits >  0x3FA00000u)   // > 1.25
            + (abits >= 0x3FE00000u)   // >= 1.75
            + (abits >  0x40200000u)   // > 2.5
            + (abits >= 0x40600000u)   // >= 3.5
            + (abits >  0x40A00000u);  // > 5.0
```

Each comparison generates a `SETP`+`SEL`+`IADD` instruction sequence — all executed uniformly
by every thread in a warp, with zero divergence. The alternating `>`/`>=` preserves
round-to-nearest-even at midpoints. Positive IEEE 754 floats preserve ordering as unsigned
integers, making the bit-pattern comparisons correct.

---

### Rank 3: Sparse FP4 GEMV via CUDA Cores + LUT Dequantization

**Status**: Working custom kernel (v8). Hardware tensor core path exists but unused.

The v8 sparse kernel uses:
- 16-entry constant-memory LUT for FP4→FP32 dequantization
- 256-entry constant-memory LUT for FP8 E4M3→FP32 scale dequantization
- Scalar CUDA core `float` FMA operations (not tensor cores)
- `__half2float` / `__float2half` conversions throughout

SM121 supports `mma.sync.aligned.kind::mxf4nvf4.sp::ordered_metadata.block_scale`
which replaces ~20 CUDA core instructions with 1 tensor core instruction. However,
for GEMV (M=1), padding to M=16 wastes 15/16 of tensor core compute.

**Hardware alternatives**:
- cuSPARSELt v0.8.0 (SM121 support added)
- CUTLASS Sparse Block-Scaled GEMM for BS≥2 (MTP gives effective BS=2-3)
- Native `mma.sync` FP4 for single-token (2-5% E2E gain, limited by M-padding waste)

---

### Rank 4: NVFP4 Full-Emulation Backend

**Status**: Fixed and available, but should never activate in production.

`fix_nvfp4_emulation_backend.py` patches the Python dequant+matmul path that runs when
no native backend is available. This path:
- Quantizes activations to FP4 in Python
- Dequantizes both weights and activations to output dtype
- Performs `torch.matmul(x_dq, w_dq.t())`
- Involves GPU→CPU transfers (`.item()` calls)

The image defaults to Marlin (`VLLM_NVFP4_GEMM_BACKEND=marlin`) which uses native FP16
tensor cores for W4A16 dequantization at 59.9 tok/s. The emulation path is a dead code
path in normal operation.

---

### Rank 5: `nv_fp4_dummy.h` Software FP4 Type System

**Status**: Working. Waiting for NVIDIA official `cuda_fp4.h`.

CUDA 13.0 CCCL headers reference `__nv_fp4_e2m1` but NVIDIA has not released the
official implementation. The dummy header provides:
- `__nv_fp4_e2m1` struct with software `operator float()` (switch on exponent)
- `__nv_fp4x2_storage_t` packed storage
- `__nv_cvt_float_to_fp4()` software intrinsic
- `__nv_cvt_fp4_to_halfraw()` software intrinsic

These are injected into CCCL and FlashInfer headers at build time. When NVIDIA releases
official types (expected CUDA 13.1 or 13.2), they would be a drop-in replacement.

---

### Rank 6: CUTLASS MoE Cooperative-Only Scheduling

**Status**: Blocked on NVIDIA upstream.

CUTLASS SM120 MoE kernels are hardcoded to Cooperative scheduling:
- `KernelScheduleAuto` enforced by `static_assert` in `sm1xx_common.inl`
- Auto = Cooperative (8 warps/tile, `AtomLayout<4,2,1>`)
- Pipeline stages = 3 (hardcoded)
- ClusterShape 1×1×1 forced (GB10 lacks multi-CTA clusters)

Pingpong scheduling (which overlaps compute and memory for better utilization) is
available for FP8 grouped GEMM (`grouped_mm_gb10_native_v109.cu`) but NOT for the
BlockScaled FP4 MoE GEMM kernels used by NVFP4.

---

### Rank 7: tcgen05 Environment Variables (Misleading)

**Status**: Cosmetic issue only. mma.sync IS the correct hardware path.

The Dockerfile sets:
```dockerfile
ENV ENABLE_TCGEN05_HARDWARE=1
ENV NVCC_PREPEND_FLAGS="-DENABLE_TCGEN05_HARDWARE=1"
```

But SM121 does NOT have TMEM or tcgen05. It uses extended `mma.sync` (warp-cooperative,
register-to-register). The code at `nvfp4_tcgen05_ptx_v2.cuh:172-173` confirms:

```cpp
// GB10 (sm_121a) uses standard mma instruction, NOT tcgen05
```

These env vars are harmless (the code falls back to mma.sync regardless) but misleading.

---

### Rank 8: BF16→FP16 via FP32 Intermediate

**Status**: Minor inefficiency.

`sparse_fp4_v8.cu` line 213:
```cpp
output[idx] = __float2half(__bfloat162float(input[m * K + k]));
```

This does BF16→FP32→FP16 (two conversions). SM121 supports `__bfloat162half()` for
direct BF16→FP16 conversion in a single instruction. A vectorized `__bfloat1622half2()`
version could process 2 elements per instruction.

---

### Rank 9: SiLU Activation via `expf()`

**Status**: Already hardware-accelerated. No action needed.

`expf()` maps to the SFU's `ex2.approx.f32` PTX instruction on SM121. This is a
hardware special function unit operation, not a software emulation.

---

### Rank 10: FlashInfer Fused MoE TMA Corruption

**Status**: Blocked. Disabled via `VLLM_USE_FLASHINFER_MOE_FP4=0`.

FlashInfer's CUTLASS fused MoE was patched to compile on SM121 (4 compilation fixes)
but produces garbage output due to TMA (Tensor Memory Accelerator) scheduling
differences between SM120 and SM121. Also 30-40% slower than CUTLASS MoE baseline.

Requires FlashInfer upstream to fix their SM121 TMA codepath.

---

### Rank 11–12: Build-System and Routing Fixes

These are not runtime emulations — they are build-time compatibility patches:
- `patch_flashinfer_fp4.sh`: Adds SM121 exception to FlashInfer's arch check
- `fix_capability_121_v112.py`: Routes CC 121 to SM120 kernel dispatch
- `fix_dispatcher_flag_v115.sh`: Ensures `ENABLE_SCALED_MM_SM120=1` for compilation
- `fix_cmake_sm120_archs_v113_corrected.sh`: CMake architecture list fixes

These should be upstreamed to vLLM/FlashInfer so SM121 is recognized natively.

---

## Actionable Next Steps

1. **~~Re-enable CUTLASS FP8 scaled_mm~~** (Rank 1) — TESTED, REVERTED. CUTLASS SM100
   kernels are 7-10% slower than cuBLAS on SM121. cuBLAS has better SM121-specific tuning.
   Future: custom SM121 CUTLASS tile shapes could help.

2. **~~Optimize software E2M1~~** (Rank 2) — DONE. Replaced 7-branch threshold comparison
   with branchless integer comparison sum. Zero warp divergence.

3. **CUTLASS Sparse GEMM for BS≥2** (Rank 3) — With MTP speculative decode giving
   effective batch size 2-3, tensor cores operate at full efficiency. This is the
   highest-potential kernel optimization (+50-100% at BS=2+).

4. **Direct BF16→FP16 conversion** (Rank 8) — Replace `__bfloat162float` +
   `__float2half` with `__bfloat162half()` in the sparse v8 kernel.

5. **Clean up tcgen05 env vars** (Rank 7) — Remove or rename the misleading
   `ENABLE_TCGEN05_HARDWARE` environment variable.

---

**Last Updated**: 2026-02-22
