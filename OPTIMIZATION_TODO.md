# Sparse FP4 GEMV Kernel Optimization Roadmap

Current kernel: **v8** (~47 tok/s peak decode with CUDA graphs, 35.2 GiB model memory)
Overall bandwidth utilization: **~82% of 273 GB/s peak** (214 GB/s effective)
MoE kernel bandwidth during active time: **~71%** (194 GB/s)

---

## Critical Context: Bottleneck Analysis (v9 Post-Mortem)

v9 Phase 1 optimizations (TILE_K=256, bank-conflict-free LUT, fused weighted_reduce) were
**performance-neutral** (+/-2%, within noise). This prompted a deep analysis revealing that
**MoE expert GEMV is NOT the primary bottleneck**.

### Data Read Per Decode Token (Qwen3-Next-80B)

| Component | Data | % of Total |
|-----------|------|-----------|
| Mamba projections (bf16, unquantized) | 2,100 MB | **46.0%** |
| Embedding + LM head (bf16) | 1,187 MB | 26.0% |
| MoE routed experts (10 of 512) | 810 MB | 17.7% |
| Full attention Q/K/V (bf16) | 240 MB | 5.3% |
| Out/O projections (NVFP4) | 135 MB | 3.0% |
| Router gates + misc | 96 MB | 2.0% |
| **Total** | **~4,568 MB** | |

### Performance Ceiling

| Metric | Value |
|--------|-------|
| Theoretical max (273 GB/s) | **57-60 tok/s** |
| Observed (v8 + CUDA graphs) | **47 tok/s** (82% BW util) |
| MoE kernel improvement ceiling | **+8-9% E2E** → ~50-51 tok/s |
| Mamba quantization opportunity | **+30-66% E2E** → 60-78 tok/s |

### Why Phase 1 Was Neutral

1. **AtomicAdd was not the bottleneck** — with TILE_K=256 and only 8 k-blocks, contention is minimal
2. **Bank conflicts were rare** — FP8 scale values have limited entropy per expert
3. **Fused weighted_reduce** — eliminated a ~1% kernel; savings below measurement noise
4. The "18% of 273 GB/s" figure was misleading — during active MoE time, BW is actually ~71%

---

## Phase 1: Low-Risk Micro-Optimizations — COMPLETED (v9)

### 1a. TILE_K=256 — NEUTRAL
- v9 implemented, benchmarked: no measurable improvement
- 4x fewer K-blocks, but atomicAdd was not the bottleneck

### 1b. Bank-Conflict-Free FP8 LUT — NEUTRAL
- v9 implemented: padded indexing adds overhead that offsets any bank conflict reduction

### 1c. Fuse F32-to-F16 into Weighted Reduce — NEUTRAL
- v9 implemented: eliminates one tiny kernel, below noise threshold

### 1d. 128-bit Vectorized Loads — DEFERRED
- Risk: N_PER_BLOCK=2048 → too few blocks for small N
- May revisit if combined with other restructuring

---

## Phase 2: Eliminate AtomicAdd + FP16 Accumulation (expected +5-9% E2E)

### 2a. Eliminate AtomicAdd — Single K-Block + Warp Shuffle — [3-5% E2E]
- **Status**: PLANNED (v10)
- Restructure grid: each block owns FULL K-dimension for its N-columns
- Eliminates `torch::zeros()` allocation and `.to(torch::kFloat16)` conversion pass
- Design: REDUCE_THREADS=4 per N-column, each processes K/4 elements
  - THREADS=128, N_PER_BLOCK=32, gate_up grid=(32,1,10)=320 blocks (6.7/SM)
  - Down grid=(64,1,10)=640 blocks (13.3/SM)
  - 2-step `__shfl_down_sync` reduction within each group of 4 threads
- Enables direct FP16 output writes

### 2b. FP16 Accumulation with __hfma2 — [1-3% E2E]
- **Status**: PLANNED (v10)
- Halves register pressure (32 → 16 bytes for accumulators)
- FP4 inputs (max magnitude 6) × FP8 scales × K accumulation fits FP16 range
- Benefit is register pressure, not FMA throughput (kernel is BW-bound)

### 2c. Double-Buffered A-Tile Loading — DROPPED
- A-tile is <1 KB, loaded cooperatively by 128 threads in 2-4 cycles
- Weight loads dominate by 100x; double-buffering hides negligible latency

---

## Phase 3: Advanced Approaches (potentially +10-50%)

### 3a. Tensor Core GEMV via mma.sync FP4 on SM121 — [2-5% E2E]
- **Status**: RESEARCH
- SM121 supports `mma.sync.aligned.kind::mxf4nvf4.sp::ordered_metadata.block_scale`
- Replaces ~20 CUDA core instructions with 1 mma.sync
- M-padding (1→16) wastes 15/16 of compute — limits gains on BW-bound kernel
- Reference: CUTLASS example 80b

### 3b. cuSPARSELt Integration — [2-5% E2E]
- **Status**: RESEARCH
- cuSPARSELt v0.8.0 added SM121 support
- Vendor-optimized, but designed for GEMM not GEMV (may pad M to 16)

### 3c. CUTLASS Sparse Block-Scaled GEMM for BS>=2 — [50-100% at BS=2+]
- **Status**: RESEARCH (highest potential for MTP speculative decode)
- MTP gives effective BS=2-3; tensor cores at full efficiency (no M-padding)
- SM121: 356 TFLOPS FP4 dense; sparse is 2x theoretical
- Challenge: 99 KB shared memory on SM121

### 3d. Persistent Fused Kernel (gate_up + SiLU + down) — [2-4% E2E]
- **Status**: RESEARCH
- Eliminates 2 kernel launches + 2 global memory round-trips
- Intermediate tensor ~40 KB doesn't fit shared memory; needs tiled pipeline

### 3e. Interleaved Weight Layout — [1-2% E2E]
- **Status**: RESEARCH
- Pack comp/meta/scale per K-group into contiguous bytes
- v6 tested with smaller loads (marginal); uint4 loads may differ

---

## HIGHEST IMPACT: Quantize Mamba/Non-MoE Layers (expected +27-50% E2E)

### Research Findings (2026-02-21)

#### Why NVIDIA left these layers in BF16

NVIDIA's Nemotron-3 Nano report revealed that Mamba output projections suffer **40% flushes
to zero** when quantized to NVFP4 (E2M1 min nonzero ~0.0625). Layers immediately before
attention layers are most sensitive. For Qwen3-Next, NVIDIA conservatively excluded ALL 36
Mamba `in_proj_qkvz` and `in_proj_ba` projections from NVFP4.

Academic sources confirm:
- **MambaQuant (ICLR 2025)**: Gate projections contain weight outliers; Parallel Scan amplifies
  quantization error. W8A8 achieves <1% accuracy drop with KLT rotation.
- **Q-Mamba (ICLR 2025)**: FP8 E4M3 causes perplexity to explode (8,114) due to "swamping
  effect" in SSM state accumulation. BUT this only affects activation/state accumulation —
  **weight-only FP8 storage with BF16 compute avoids swamping entirely**.

#### Validated safe: FP8 for in_proj_qkvz

The Qwen team released `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` which quantizes `in_proj_qkvz`
AND `self_attn.{q,k,v}_proj` to FP8 E4M3 (block size 128×128). Their exclude list keeps
`in_proj_ba`, `conv1d`, `A_log`, `dt_bias`, norms, and gates at full precision. This validates:
- **in_proj_qkvz [12288, 2048]: FP8-safe** (Qwen-validated)
- **self_attn Q/K/V: FP8-safe** (Qwen-validated)
- **in_proj_ba: NOT FP8-safe** (Qwen excluded it)

### Quantize Mamba Projections (in_proj_qkvz only)
- **Status**: RESEARCH COMPLETE — ready to implement
- **Risk**: LOW (Qwen-validated FP8 for these specific layers)
- 36 Mamba layers × `in_proj_qkvz` [12288, 2048] at bf16 = **~1,728 MB**
- 12 Attention layers × Q/K/V at bf16 = **~240 MB**
- FP8 saves ~984 MB/token → total **3,584 MB** → **~76 tok/s** theoretical (+27%)

### Quantize Embeddings and LM Head
- **Status**: RESEARCH COMPLETE — medium risk
- 1,187 MB (26% of total) in bf16
- FP8 quantization: saves ~593 MB → **~2,991 MB** → **~91 tok/s** theoretical
- Embedding layers are lookup tables — generally robust to FP8
- LM head has some accuracy sensitivity for top-k token selection

### Combined Quantization Target
- Mamba in_proj_qkvz + attn QKV at FP8: **3,584 MB** → **~76 tok/s** (+27%)
- Add embeddings + lm_head at FP8: **~2,991 MB** → **~91 tok/s** (+52%)
- Current baseline with Marlin + MTP: 59.9 tok/s

### Recommended Implementation: LLM Compressor Mixed Checkpoint

Use [LLM Compressor](https://docs.vllm.ai/projects/llm-compressor/) to create a mixed-precision
checkpoint with two `config_groups`:
1. **group_0 (NVFP4)**: MoE expert Linear layers (as-is from NVIDIA checkpoint)
2. **group_1 (FP8 E4M3)**: `in_proj_qkvz`, `q_proj`, `k_proj`, `v_proj`
3. **Ignore list**: `in_proj_ba`, `conv1d`, `A_log`, `dt_bias`, gates, norms

This gives native vLLM compressed-tensors support, CUDA graph compatibility, and per-layer
kernel dispatch. Requires ~120 GB RAM to load the BF16 base model for requantization.

Alternative: vLLM `--quantization fp8` does runtime FP8 casting but is all-or-nothing (cannot
selectively quantize). NVIDIA ModelOpt or torchao are other options.

---

## Other Optimizations (independent of kernel)

### MTP Draft Token Tuning
- Benchmark `num_speculative_tokens=1` vs current 2
- Monitor with `--speculative-config '{"metrics": true}'`
- Estimated impact: +5-15%

### CUTLASS SSD Kernels for Mamba Layers
- CUTLASS 4.4.0 added State Space Decomposition kernels (examples 111, 112)
- If SM120-compatible, could accelerate the 36 Mamba layers
- Estimated impact: +5-10%, high risk

### KV Cache and Memory Optimizations
- FP8 KV cache (`--kv-cache-dtype fp8`) — already active
- Prefix caching (`--enable-prefix-caching`) for repeated prompts
- UMA-aware weight offloading (`--cpu-offload-gb`)

### vLLM Feature Enablement
- Output token buffering (57% E2E gain at high concurrency)
- Async scheduling with MTP
- FULL_AND_PIECEWISE CUDA graph mode — already active
- Estimated impact: +2-5% each

---

## v9 Benchmark Results (for reference)

v9 Phase 1 was tested with CUDA graphs (FULL_AND_PIECEWISE, 35 captures):

| Config | v8 Decode | v9 Decode | Delta |
|--------|-----------|-----------|-------|
| 128/128 | 46.8 tok/s | 46.9 tok/s | +0.2% |
| 256/256 | 43.3 tok/s | 42.3 tok/s | -2.3% |
| 1024/128 | 46.9 tok/s | 47.1 tok/s | +0.4% |
| 1024/1024 | 41.2 tok/s | 40.2 tok/s | -2.4% |
| 128/1024 | 41.2 tok/s | 40.4 tok/s | -1.9% |

All deltas within measurement noise. v9 is not adopted; v8 remains the current kernel.

Note: v9 without CUDA graphs (enforce-eager) measured 32-40 tok/s — confirming that
CUDA graphs provide ~20-45% speedup for the sparse kernel path.

---

## Key Research References

- [Blackwell NVFP4 Kernel Hackathon](https://yue-zhang-2025.github.io/2025/12/02/blackwell-nvfp4-kernel-hackathon-journey.html) — 89.5x speedup on B200 GEMV
- [TFLOPS Gap: FP4 MoE on Blackwell](https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison) — kernel fusion reduced memory traffic 21.9%
- [FP4 on DGX Spark](https://forums.developer.nvidia.com/t/fp4-on-dgx-spark-why-it-doesnt-scale-like-youd-expect/360142) — SM121 FP4 efficiency 36% vs 70% for BF16
- [SM121 CUTLASS Kernel Results](https://forums.developer.nvidia.com/t/sm121-cutlass-kernel-optimization-results-nvfp4-356-tflops-moe-grouped-gemm-on-dgx-spark/359960) — 356 TFLOPS FP4 dense GEMM
- [GTC 2025: Maximize Memory Bandwidth](https://shreyansh26.github.io/post/2025-03-23_gtc25-maximize-memory-bandwidth-part-1/) — Little's Law: >40 KB bytes-in-flight per SM
- [FLUTE: LUT-Quantized LLMs](https://arxiv.org/abs/2407.10960) — LUT vectorization + duplication
- [BitDecoding: Tensor Core Low-Bit Decode](https://arxiv.org/abs/2503.18773) — 8.6x speedup via tensor core decode
- [cuSPARSELt Release Notes](https://docs.nvidia.com/cuda/cusparselt/release_notes.html) — v0.8.0 added SM121
- [CUTLASS Sparse GEMM Example 80b](https://github.com/NVIDIA/cutlass/blob/main/examples/80_blackwell_geforce_sparse_gemm/80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm.cu)
- [Backend.AI: Is DGX Spark Actually Blackwell?](https://www.backend.ai/blog/2026-02-is-dgx-spark-actually-a-blackwell) — SM121 uses extended mma.sync, NOT tcgen05

---

**Last Updated**: 2026-02-21
