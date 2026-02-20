# We Unlocked NVFP4 on DGX Spark -- and It's 20% Faster Than AWQ

**NVIDIA's own software stack couldn't do it. We did. Avarok's open-source vLLM image is the first to make NVFP4 outperform AWQ on GB10.**

---

*Thomas P. Braun, Avarok*

Feb 19, 2026

---

## TL;DR

The NVIDIA DGX Spark has native FP4 tensor cores. NVFP4 — the quantization format designed for those tensor cores — should be the fastest 4-bit inference path. It wasn't. On every shipping software stack, NVFP4 either didn't work at all or ran at the same speed as AWQ INT4.

We fixed that. NVFP4 on our image is now **~20% faster than AWQ** across every workload. No exceptions.

| Configuration | Avg Decode | Peak Decode | Image |
|---|---:|---:|---|
| **NVFP4 (Avarok)** | **~42 tok/s** | **47.1 tok/s** | avarok/dgx-vllm-nvfp4-kernel:v22 |
| NVFP4 (NVIDIA) | ~36 tok/s | 40.2 tok/s | nvcr.io/nvidia/vllm:26.01-py3 |
| AWQ INT4 (Avarok) | ~36 tok/s | 39.7 tok/s | avarok/dgx-vllm-nvfp4-kernel:v22 |
| AWQ INT4 (NVIDIA) | ~34 tok/s | 38.2 tok/s | nvcr.io/nvidia/vllm:26.01-py3 |

For models that ship with an MTP head, enabling speculative decoding on top of our NVFP4 stack pushes throughput to **~67 tok/s average** — but the core contribution is making NVFP4 itself faster than AWQ for the first time on this hardware.

---

## Why This Matters

At Avarok, we build open-source AI infrastructure. When the DGX Spark launched, we saw a machine with incredible hardware — 119.7 GB of unified memory, 273 GB/s bandwidth, Blackwell FP4 tensor cores — held back by incomplete software support. So we set out to fix it.

The DGX Spark's GB10 chip runs compute capability SM 12.1 — a variant unique to this hardware. The data center Blackwell chips (B200, GB200) run SM 12.0. The consumer chips (RTX 5090) run SM 12.0. SM 12.1 is the odd one out, and it lacks hardware support for the `cvt.rn.satfinite.e2m1x2.f32` PTX instruction that NVFP4 quantization requires to convert values to the E2M1 format.

The result: **NVFP4 was broken on DGX Spark.**

- **Vanilla vLLM**: NVFP4 crawled at **1.1 tok/s** — the missing instruction triggered a catastrophic fallback that made inference 30x slower than it should be. Essentially unusable.
- **NVIDIA's official vLLM image** (26.01): Fixed the PTX issue but shipped with `flashinfer-cutlass` as the only NVFP4 MoE backend. Result: **~36 tok/s — the same speed as AWQ**. No advantage to using NVFP4.
- **TensorRT-LLM**: Topped out at **29.6 tok/s** (CUTLASS kernels hardcoded to Cooperative scheduling, no alternative MoE backends available on SM 12.1).

DGX Spark owners had no reason to use NVFP4. The format that Blackwell's tensor cores were designed for was no faster than a generic INT4 quantization scheme. AWQ at ~35 tok/s was the best anyone could get.

We changed that.

---

## What We Built

Over several weeks of kernel-level debugging, CUTLASS patching, and vLLM integration work, we built a custom Docker image (`avarok/dgx-vllm-nvfp4-kernel:v22`) that solves every SM 12.1 limitation.

### 1. Software E2M1 Conversion — From Broken to Working

This was the breakthrough that made everything else possible.

The missing PTX instruction meant NVFP4 models loaded at 1.1 tok/s on vanilla vLLM. We patched CUTLASS and FlashInfer to use a **software E2M1 conversion path**: bit manipulation (`__float_as_uint`, shift, mask) instead of the missing hardware instruction. This single change took NVFP4 from **1.1 tok/s to 35+ tok/s** — a 32x improvement that made NVFP4 viable on GB10 for the first time.

### 2. Marlin MoE Backend — From Matching AWQ to Beating It

Getting NVFP4 to work was step one. Making it faster than AWQ was step two.

The default `flashinfer-cutlass` backend that ships with NVIDIA's image runs NVFP4 at ~36 tok/s — the same speed as AWQ. We enabled the **Marlin backend** (`VLLM_NVFP4_GEMM_BACKEND=marlin`), which performs W4A16 dequantization through a code path better suited to SM 12.1 hardware.

The result: NVFP4 jumped from ~36 tok/s to **~42 tok/s**. Same model, same quantization, same hardware — **17% faster** from the backend alone. This is the difference between NVFP4 matching AWQ and NVFP4 decisively beating it.

| MoE Backend | NVFP4 Throughput | vs AWQ |
|---|---:|---:|
| flashinfer-cutlass (NVIDIA ships this) | ~36 tok/s | same as AWQ |
| **Marlin (Avarok)** | **~42 tok/s** | **~20% faster** |

### 3. SM 12.1 Capability Routing

Multiple vLLM subsystems check GPU capability and route to optimized code paths. GB10 reports SM 12.1, which falls through to generic (slow) fallbacks. We patched the checks to route SM 12.1 to SM 12.0 optimized paths, ensuring FlashInfer attention, CUTLASS GEMMs, and torch.compile all work correctly.

### 4. vLLM v0.16.0rc2

Our image ships vLLM `v0.16.0rc2` (Feb 2026), three months ahead of NVIDIA's shipping `v0.13.0`. This gives us improved torch.compile graph fusion, better chunked prefill, and optimized CUDA graph capture. The version gap alone accounts for 6% higher AWQ throughput and up to 3x faster prefill latency at large context lengths — meaning even if you only run AWQ, our image is faster than NVIDIA's.

### 5. MTP Speculative Decoding

Qwen3-Next-80B ships with a built-in **Multi-Token Prediction (MTP) head** that predicts 2 tokens per step. NVIDIA's vLLM image excludes NVFP4 models from this code path. We patched that exclusion.

With MTP active, throughput jumps from 42 tok/s to **~67 tok/s average** (63-89% draft acceptance rates). This is a powerful bonus for models that support MTP — but it's separate from the core NVFP4-vs-AWQ story. NVFP4 is already 20% faster than AWQ before MTP enters the picture.

### Why 20% Faster Becomes 2x Faster with MTP

A 20% base throughput improvement sounds modest. But speculative decoding is a **multiplier** on your base decode speed — and multipliers compound.

MTP predicts 2 draft tokens per step. When a draft is accepted, you get multiple tokens for the cost of one verification pass. The effective throughput scales as:

> **effective tok/s = base tok/s × average accepted tokens per step**

Our Marlin backend produces a base speed of ~42 tok/s. MTP's average acceptance yields ~1.6 tokens per step on this model. That gives us:

| Base Backend | Base Speed | × MTP (1.6x) | vs AWQ (35 tok/s) |
|---|---:|---:|---:|
| flashinfer-cutlass (NVIDIA) | ~36 tok/s | ~58 tok/s | 1.7x |
| **Marlin (Avarok)** | **~42 tok/s** | **~67 tok/s** | **1.9x** |

The 6 tok/s gap between 36 and 42 at the base level becomes a **9 tok/s gap** after the MTP multiplier (58 vs 67). Every millisecond we shaved off per-token decode gets amplified by speculative decoding. The Marlin improvement isn't just additive — it's the foundation that MTP builds on.

And critically: the current AWQ checkpoint for this model doesn't include MTP head weights, so AWQ is stuck at ~35 tok/s with no speculative decoding path available. NVFP4 + Marlin + MTP delivers **nearly 2x the throughput of AWQ** — not because any single optimization is a 2x improvement, but because three targeted gains stack multiplicatively.

---

## The Benchmarks

We ran a comprehensive **Pareto frontier benchmark** across 14 ISL/OSL configurations, covering prefill-heavy (RAG, summarization), balanced (chat), and decode-heavy (code generation, reasoning) workloads. Every configuration ran 3 times with the median reported. All tests used 64K max context on a single GB10 GPU.

### Decode Throughput — NVFP4 (Avarok) Wins Every Row

| Workload | NVFP4+MTP | NVFP4 (Avarok) | NVFP4 (NVIDIA) | AWQ (Avarok) | AWQ (NVIDIA) |
|---|---:|---:|---:|---:|---:|
| 128/128 (peak) | **111.9** | **47.1** | 40.1 | 39.7 | 38.2 |
| 256/256 (short chat) | **86.9** | **44.0** | 38.0 | 37.2 | 35.8 |
| 1024/1024 (standard chat) | **67.1** | **41.8** | 36.3 | 35.4 | 34.1 |
| 4096/4096 (long context) | **60.4** | **40.5** | 35.5 | 34.7 | 33.4 |
| 8192/1024 (summarization) | **64.7** | **40.9** | 35.4 | 34.8 | 33.3 |
| 32768/256 (RAG prefill) | **72.1** | **41.9** | 35.4 | 36.2 | 29.8 |

Without MTP, NVFP4 on our image beats AWQ by **15-19%** on the same image, and by **21-41%** against NVIDIA's official image. With MTP, the gap becomes **1.7-2.9x**.

### Peak Performance: 111.9 tok/s from an 80B Model

At short sequence lengths (128/128), NVFP4 + MTP hits **111.9 tok/s** — that's a 9 ms per-token latency from an 80-billion-parameter model running on a single desktop GPU.

To put that in perspective: this is faster than most humans can read. At 112 tokens per second, the model produces roughly 80 words per second. A fast reader processes about 4-5 words per second. The model is generating text **16x faster than you can read it**.

Even at realistic chat workloads (1024/1024), MTP sustains **67 tok/s**. At long-context summarization (8192/1024), it holds **64.7 tok/s**. The MTP acceptance rate stays high (63-89%) across all input lengths because the Qwen3-Next architecture's MTP head was trained end-to-end with the base model — it's not a bolted-on afterthought, it's integral to the model's design.

This is only possible because of the Marlin backend. On NVIDIA's flashinfer-cutlass backend at 36 tok/s base, MTP would top out around ~58 tok/s. Our 42 tok/s base lifts the ceiling to 112.

### Per-Token Latency (TPOT) — Lower is Better

| Workload | NVFP4 (Avarok) | NVFP4 (NVIDIA) | AWQ (Avarok) | AWQ (NVIDIA) |
|---|---:|---:|---:|---:|
| 128/128 | **21.4 ms** | 25.1 ms | 25.4 ms | 26.4 ms |
| 1024/1024 | **24.0 ms** | 27.6 ms | 28.2 ms | 29.4 ms |
| 4096/4096 | **24.7 ms** | 28.2 ms | 28.8 ms | 29.9 ms |

NVFP4 (Avarok) delivers **16-19% lower per-token latency** than AWQ across all sequence lengths.

### Prefill Latency (TTFT) — The NVIDIA Image Collapses at Scale

| Input Length | NVFP4 (Avarok) | NVFP4 (NVIDIA) | AWQ (Avarok) | AWQ (NVIDIA) |
|---|---:|---:|---:|---:|
| 256 tokens | **533 ms** | 585 ms | 598 ms | 607 ms |
| 1024 tokens | **630 ms** | 685 ms | 662 ms | 692 ms |
| 4096 tokens | **1,275 ms** | 1,367 ms | 1,241 ms | **3,770 ms** |
| 8192 tokens | **2,034 ms** | 2,320 ms | 2,070 ms | **5,258 ms** |

At short inputs, all configurations are within ~100ms. At 4K+ input lengths, NVIDIA's official image takes **3x longer** to produce the first token — the difference between a responsive experience and a noticeable delay.

---

## Concurrency: It Scales

Single-user performance matters, but real deployments serve multiple users. We benchmarked NVFP4 with MTP enabled across 7 concurrency levels on the 1024/1024 (standard chat) workload:

| Concurrent Users | Aggregate Throughput | Per-User Decode |
|---:|---:|---:|
| 1 | 60 tok/s | 62 tok/s |
| 4 | 140 tok/s | 37 tok/s |
| 8 | 237 tok/s | 32 tok/s |
| 16 | 341 tok/s | 24 tok/s |
| 32 | 487 tok/s | 17 tok/s |
| 64 | 658 tok/s | 12 tok/s |

At 32 concurrent users, aggregate throughput reaches **487 tok/s** while each user still gets 17 tok/s — comparable to AWQ's single-user performance on the NVIDIA image.

---

## Reproducibility

Everything is open source. Pull our image and run:

```bash
# NVFP4 with Marlin backend (~42 tok/s, no speculative decoding)
sudo docker run -d --name dgx-vllm-nvfp4 \
  --network host --gpus all --ipc=host \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
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

To enable MTP speculative decoding (for models with an MTP head, like Qwen3-Next-80B), add to `VLLM_EXTRA_ARGS`:
```
--speculative-config '{"method":"mtp","num_speculative_tokens":2}' --no-enable-chunked-prefill
```

The image is on Docker Hub: `avarok/dgx-vllm-nvfp4-kernel:v22`

---

## What This Means for DGX Spark Owners

If you own a DGX Spark, NVFP4 should be your default 4-bit format. NVIDIA's shipping software stack gives you ~35 tok/s with AWQ. Our stack gives you **~42 tok/s with NVFP4** — a 20% throughput improvement from the same hardware, running the same model weights.

The patches are specific to SM 12.1 (GB10), but the approach generalizes: when the official toolchain doesn't fully support your hardware, sometimes the community has to step in.

We stepped in:
- **NVFP4 went from broken (1.1 tok/s) to the fastest 4-bit format on DGX Spark**
- **Decode throughput**: +15-19% vs AWQ (same image), +21-41% vs AWQ (NVIDIA image)
- **Per-token latency**: 16-19% lower TPOT
- **Prefill latency**: Up to 66% faster TTFT vs NVIDIA's image at 4K+ inputs
- **Zero errors**: 100% stability across all benchmarks

NVFP4 is the new default for 4-bit inference on DGX Spark. AWQ's reign is over.

---

## Technical Details

### Hardware
- NVIDIA DGX Spark (GB10, Compute Capability SM 12.1)
- 119.7 GB unified GPU memory
- 273 GB/s LPDDR5X bandwidth

### Software Stack
- **Avarok image**: `avarok/dgx-vllm-nvfp4-kernel:v22` — vLLM v0.16.0rc2, PyTorch nightly + CUDA 13.0, FlashInfer latest, Marlin MoE backend
- **NVIDIA image**: `nvcr.io/nvidia/vllm:26.01-py3` — vLLM v0.13.0+nv26.01, flashinfer-cutlass backend
- **NVFP4 Model**: `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` (80B params, ~3B active/token, MoE + Mamba hybrid)
- **AWQ Model**: `cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit` (W4A16, group_size=32)

### Patches Applied (Avarok v22)
| Patch | Purpose |
|---|---|
| `fix_flashinfer_e2m1_sm121.py` | Software E2M1 conversion (bypasses missing PTX) |
| `fix_flashinfer_nvfp4_moe_backend.py` | Fix upstream vLLM bug in NVFP4 MoE backend selection |
| `fix_capability_121_v112.py` | Route SM 12.1 to SM 12.0 optimized code paths |
| `fix_mtp_nvfp4_exclusion.py` | Enable MTP speculative decoding for NVFP4 models |

### Benchmark Methodology
- 14 ISL/OSL configurations across prefill-heavy, balanced, and decode-heavy regimes
- 3 runs per configuration, medians reported
- 2 warmup runs before measurement
- Streaming TTFT measurement via SSE
- TPOT calculated from API-reported usage statistics
- 64K max context length, batch size 1, single GPU

---

*All benchmark data, scripts, and raw JSON results are available in the project repository. Docker image: `avarok/dgx-vllm-nvfp4-kernel:v22`*

*Avarok — Advancing open-source AI infrastructure for everyone.*
