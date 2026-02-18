# NVFP4 Optimization History — Qwen3-Next-80B on DGX Spark GB10

Each improvement is documented with its commit, configuration, and measured results.

---

## Improvement 1: Software E2M1 + CUDA Graphs (v21 baseline)

**Commit**: `9e1cd69` — feat: Add FlashInfer SM121 patches and MoE backend fix
**Throughput**: 1.1 → 35.0 tok/s (+3081%)
**Date**: 2026-02-17

**What changed**: Replaced Python software FP4 fallback (`.item()` calls blocking CUDA) with a 15-line C++ device function implementing IEEE 754 E2M1 conversion. This enabled CUDA graph capture (54 graphs) and torch.compile.

**Configuration**:
```bash
# No special env vars needed — default CUTLASS backend
docker run ... avarok/dgx-vllm-nvfp4-kernel:v21 serve
```

**Key insight**: The GB10 has FP4 tensor cores but is missing the `cvt.rn.satfinite.e2m1x2.f32` PTX instruction. Software emulation in C++ with `#if __CUDA_ARCH__ == 1210` guard compiles all 5 NVFP4 kernel files.

---

## Improvement 2: Marlin MoE Backend (+13%)

**Commit**: `e9ff094` — docs: Update README with Marlin MoE optimization (35 → 39.5 tok/s)
**Throughput**: 35.0 → 39.5 tok/s (+13%)
**Date**: 2026-02-17

**What changed**: Switched MoE GEMM backend from CUTLASS to Marlin. Marlin uses W4A16 dequantization — it converts FP4 weights to FP16 at runtime and uses FP16 tensor cores. On GB10 (memory-bandwidth-bound at batch=1), avoiding the FP4→compute pipeline overhead gives a significant speedup.

**Configuration**:
```bash
-e VLLM_TEST_FORCE_FP8_MARLIN=1
-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
-e GPU_MEMORY_UTIL=0.90
```

**Key insight**: GB10 lacks native FP4 compute support (missing E2M1 convert instruction), so software dequantization to FP16 (Marlin) is actually faster than CUTLASS FP4 GEMM with software E2M1.

---

## Improvement 3: Marlin for Dense GEMM (+2%)

**Commit**: `cf81980` — docs: Update README with 40.2 tok/s record (Marlin dense + MoE)
**Throughput**: 39.5 → 40.2 tok/s (+2%)
**Date**: 2026-02-17

**What changed**: Extended Marlin to also handle dense (non-MoE) GEMM — attention Q/K/V/O projections. Previously these used FlashInfer CUTLASS backend.

**Configuration**:
```bash
-e VLLM_TEST_FORCE_FP8_MARLIN=1
-e VLLM_NVFP4_GEMM_BACKEND=marlin
-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Key insight**: Marlin's W4A16 approach benefits all linear layers, not just MoE. The additional 2% comes from attention projections also using the faster dequant path.

---

## Improvement 4: MTP Speculative Decoding — 1 Token (+37%)

**Commit**: `83e7f1d` — feat: MTP speculative decoding — 59.9 tok/s (40.2 → 59.9, +49%)
**Throughput**: 40.2 → 55.4 tok/s (+37%)
**Date**: 2026-02-18

**What changed**: Enabled the model's built-in Multi-Token Prediction (MTP) draft head for speculative decoding. The NVFP4 checkpoint includes MTP weights (1 decoder layer + fc + norms), all stored as BF16. Required a patch (`fix_mtp_nvfp4_exclusion.py`) to fix a vLLM bug where MTP layers were incorrectly quantized to FP4.

**The Bug**: ModelOpt's exclude list has `mtp.layers.0*` which matches `mtp.layers.0.self_attn.q_proj` but NOT `mtp.fc`. Since all MTP weights are BF16, the fc layer gets initialized as FP4 → shape mismatch assertion failure.

**The Fix**: If any exclude pattern starts with `mtp.` and the current layer prefix also starts with `mtp.`, exclude the layer.

**Configuration**:
```bash
-e VLLM_TEST_FORCE_FP8_MARLIN=1
-e VLLM_NVFP4_GEMM_BACKEND=marlin
-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# + runtime patch and explicit vllm serve command:
python3 /tmp/fix_mtp_nvfp4_exclusion.py && \
vllm serve nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 \
  --no-enable-chunked-prefill \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

**MTP Stats**:
- Mean acceptance length: 1.84 tokens per step
- Per-position acceptance rate: 84%
- Draft acceptance rate: 84%
- MTP model uses TRITON backend for unquantized MoE (BF16)

**Key insight**: MTP uses the model's own trained draft head (not n-gram or external model), so acceptance rates are much higher (84% vs 27% for NGram on TRT-LLM). The MTP head shares embeddings and lm_head with the main model.

---

## Improvement 5: MTP 2 Speculative Tokens (+8%)

**Commit**: `83e7f1d` (same commit)
**Throughput**: 55.4 → 59.9 tok/s (+8%)
**Date**: 2026-02-18

**What changed**: Increased `num_speculative_tokens` from 1 to 2. vLLM reuses the single MTP head cyclically for multiple speculative positions.

**Configuration**:
```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

**MTP Stats**:
- Mean acceptance length: 2.42 tokens per step
- Per-position acceptance rate: 84% (pos 1), 53% (pos 2)
- Overall draft acceptance rate: ~70%
- Net throughput: 59.9 tok/s (500-token generation)

**Key insight**: Even though position 2 acceptance drops to 53%, the marginal benefit of sometimes getting an extra token still outweighs the verification cost. The model generates ~2.42 tokens per forward pass on average, exceeding the single-token bandwidth ceiling of ~46 tok/s.

---

## Summary

| Step | Configuration | Throughput | Cumulative Speedup |
|------|--------------|-----------|-------------------|
| Baseline (Python FP4) | v20 | 1.1 tok/s | 1.0x |
| + Software E2M1 + CUDA graphs | v21 | 35.0 tok/s | 31.8x |
| + Marlin MoE | v21 + env vars | 39.5 tok/s | 35.9x |
| + Marlin dense | v21 + env vars | 40.2 tok/s | 36.5x |
| + MTP (1 spec token) | v21 + patch + spec | 55.4 tok/s | 50.4x |
| **+ MTP (2 spec tokens)** | **v21 + patch + spec** | **59.9 tok/s** | **54.5x** |

**Total improvement: 54.5x from 1.1 tok/s to 59.9 tok/s.**

The theoretical single-token decode ceiling on GB10 is ~46 tok/s (273 GB/s bandwidth). MTP speculative decoding breaks through this ceiling by generating ~2.42 tokens per forward pass.

---

## Failed/Neutral Experiments

| Experiment | Result | Why |
|-----------|--------|-----|
| `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` | 38.0 tok/s (WORSE) | Async stream overlap actually helps batch=1 |
| `VLLM_MARLIN_INPUT_DTYPE=fp8` | FAILED | SM121 not in FP8 input support list |
| `VLLM_MARLIN_INPUT_DTYPE=int8` | FAILED | NVFP4 doesn't support activation quantization |
| `VLLM_NVFP4_GEMM_BACKEND=flashinfer-trtllm` | FAILED | FlashInfer TRTLLM JIT fails on SM121 |
| `--cudagraph-capture-sizes [1]` + `--max-num-seqs 1` | 39.5-39.8 (same) | No benefit from restricting graph sizes |
| `--block-size 16` + `--no-enable-chunked-prefill` | 39.9 (same) | KV cache block size doesn't matter at batch=1 |
| FlashInfer cuDNN dense GEMM | 40.2 (same) | Same speed as Marlin dense |
| CUTLASS dense GEMM | 39.9 (slightly worse) | Slightly slower than Marlin/cuDNN |
| NGram spec-decode (TRT-LLM, 19 experiments) | 20.2 tok/s (WORSE) | 73% rejection rate, SSM state overhead |
| MTP `num_speculative_tokens=3` | 58.9 tok/s (slightly worse) | 3rd position only 34% acceptance — not worth it |
| MTP `parallel_drafting=true` | FAILED | Requires `ptd_token_id` in config.json (model lacks it) |
| MTP + `--enable-prefix-caching` | FAILED | Prefix caching requires chunked prefill, incompatible with spec decode |

---

**Last Updated**: 2026-02-18
