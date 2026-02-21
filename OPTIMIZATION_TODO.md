# Optimization TODO (Optional)

These are potential optimization paths explored but not yet implemented. All are optional and independent of each other.

## v7 Kernel Micro-Optimizations
- Vectorized 128-bit loads (`uint4`) instead of 32-bit `uint32_t` — coalesces memory into 128-byte LPDDR5X bursts
- Warp shuffle K-reduction (`__shfl_down_sync`) — eliminates the separate `.sum(1)` kernel launch and global memory round-trip
- Pre-multiply global scale into FP8 block scale LUT at expert load time — saves one multiply per inner loop iteration
- Double-buffered shared memory for A vector tiles — overlaps next tile load with current tile compute
- Estimated impact: +5-15%

## MTP Draft Token Tuning
- Benchmark `num_speculative_tokens=1` vs current 2
- If draft acceptance rate is below ~75%, 1 token may yield better throughput
- Monitor with `--speculative-config '{"metrics": true}'`
- Estimated impact: +5-15% or -10% if misconfigured

## cuSPARSELt Integration
- NVIDIA's vendor-optimized 2:4 structured sparsity library now supports SM120 with block-scaled kernels
- Auto-tuned for each GPU architecture and problem size
- Test `cusparseLt_matmul()` with expert dimensions (N=512/1024/2048, K=512/2048)
- Estimated impact: +10-20%

## Fuse Python Overhead in MoE Forward
- The `_make_v7_apply_fn` closure still goes through PyTorch for expert_map indexing, dtype casts, and `.sum(1)` reduction
- Fuse all of these into the C++ `fused_sparse_moe_v7` kernel
- Estimated impact: +3-8%

## CUTLASS SSD Kernels for Mamba Layers
- CUTLASS 4.4.0 added State Space Decomposition kernels (examples 111, 112)
- If SM120-compatible, could accelerate the 36 Mamba layers in Qwen3-Next
- Currently FLA processing is the bottleneck during MTP verification
- Estimated impact: +5-10%, high risk (SM120 compatibility unknown)

## KV Cache and Memory Optimizations
- FP8 KV cache (`--kv-cache-dtype fp8`) to halve KV cache memory
- Prefix caching (`--enable-prefix-caching`) for repeated prompts
- UMA-aware weight offloading (`--cpu-offload-gb`) — DGX Spark's NVLink-C2C makes this faster than traditional GPU systems
- Estimated impact: +2-5%, enables longer contexts

## BS>1 Dispatch to CUTLASS Sparse GEMM
- v7 kernel is optimized for BS=1 (GEMV)
- For BS>=2, dispatch to CUTLASS sparse GEMM using hardware Sparse Tensor Cores
- Unlocks multi-request serving with near-linear scaling
- Estimated impact: +2-5x at higher concurrency

## vLLM Feature Enablement
- Output token buffering (57% e2e gain at high concurrency)
- Async scheduling with MTP
- Verify FULL_AND_PIECEWISE CUDA graph mode is active (not `--enforce-eager`)
- Estimated impact: +2-5% each
