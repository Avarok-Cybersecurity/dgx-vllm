# Writing a Custom 2:4 Sparse FP4 CUDA Kernel for MoE Inference on NVIDIA GB10

## How we replaced vLLM's Marlin MoE backend with a hand-written sparse GEMV kernel -- and the 7 iterations it took to get there

---

### The Problem

We're running **Qwen3-Next-80B-A3B** on a single NVIDIA DGX Spark GB10 workstation. This is a Mixture-of-Experts model with 80 billion parameters, but only ~3 billion active per token -- routed through 10 of 512 experts per layer across 48 MoE layers.

The model uses **NVFP4** quantization: 4-bit FP (E2M1) weights with FP8 block scales. At batch size 1, inference is entirely memory-bandwidth bound. Every token requires streaming ~40 GB of expert weights through the GPU's 273 GB/s LPDDR5X bus. Anything that reduces the bytes read per token directly improves throughput.

vLLM handles this model through its **Marlin** backend -- a high-performance W4A16 (weight-4bit, activation-16bit) GEMM kernel. Marlin dequantizes FP4 weights to FP16 on the fly and uses tensor cores for the matrix multiply. It's fast, but it has a fundamental inefficiency for MoE: it pads every expert to a minimum of 8 rows. With 10 active experts at batch size 1, that's 10 experts each padded from 1 token to 8 -- an **87.5% compute waste** on padding.

We asked: what if we wrote a custom kernel that applies **2:4 structured sparsity** to the FP4 weights? This would:
1. Reduce weight data by 25% (keep 2 of every 4 values)
2. Eliminate padding waste entirely (GEMV, not GEMM -- no row padding)
3. Trade tensor core utilization for memory bandwidth efficiency

This is the story of 7 kernel iterations, several failed approaches, and the bugs we hit along the way.

---

### Background: 2:4 Structured Sparsity

NVIDIA's Ampere and later architectures support **2:4 structured sparsity** in hardware: within every group of 4 contiguous values, exactly 2 are kept and 2 are zeroed. The hardware stores only the 2 non-zero values plus a 4-bit metadata index indicating which positions they came from.

For a weight matrix of shape `[N, K]`:
- **Dense FP4**: `[N, K/2]` uint8 (2 values per byte) = `N*K/2` bytes
- **Sparse FP4**: `[N, K/4]` uint8 (compressed) + `[N, K/8]` uint8 (metadata) = `N*K/4 + N*K/8 = 3*N*K/8` bytes

That's **75% of the dense size** -- a 25% bandwidth reduction. For the down-projection of each expert (`K=2048, N=512`), this saves 128 KB per expert, or 1.28 MB across 10 active experts per layer.

The metadata encodes which 2 of 4 positions are non-zero. Each group of 4 K-elements produces 2 index values (0-3), packed into 4 bits. Two groups pack into one byte:

```
byte = (idx0_group0) | (idx1_group0 << 2) | (idx0_group1 << 4) | (idx1_group1 << 6)
```

During inference, the kernel looks up the activation values at the metadata-specified positions, multiplies by the stored weights, and accumulates.

---

### The Sparsity Selection

Before writing any kernels, we needed to validate that pruning 50% of FP4 weights wouldn't destroy model quality. We applied 2:4 sparsity offline to all 48 MoE layers using a magnitude-based selection: within each group of 4, keep the 2 values with the largest absolute magnitude.

The results were surprisingly good. FP4 weights only have 16 distinct values (`{0, +/-0.5, +/-1.0, +/-1.5, +/-2.0, +/-3.0, +/-4.0, +/-6.0}`), and the distribution is heavily concentrated around zero. In practice, ~40% of FP4 values are already zero, so "pruning" them discards very little information.

We validated output quality by comparing sparse vs. dense outputs token-by-token on reference prompts. The outputs were functionally identical -- the NVFP4 quantization noise already dominates any additional error from 2:4 sparsity.

---

### Iteration 1: The Naive Baseline

**File: `sparse_fp4_gemv.cu`**

The first kernel was deliberately simple: load the entire activation vector `A[K]` into shared memory, then each thread processes one output column, streaming through the compressed weight rows.

```cuda
// Pseudocode for v1 sparse GEMV
__shared__ float sh_A[K];  // Full activation vector
load_A_to_shared(A, sh_A, K);
__syncthreads();

float acc = 0;
for (int g = 0; g < K/4; g += 2) {
    uint8_t comp0 = B_comp[n * (K/4) + g];
    uint8_t comp1 = B_comp[n * (K/4) + g + 1];
    uint8_t meta  = Meta[n * (K/8) + g/2];
    // Decompress: look up FP4 values, gather A at metadata positions
    acc += fp4_lut[comp0 & 0xF] * sh_A[g*4 + (meta & 3)];
    acc += fp4_lut[comp0 >> 4]  * sh_A[g*4 + ((meta>>2) & 3)];
    acc += fp4_lut[comp1 & 0xF] * sh_A[g*4 + 4 + ((meta>>4) & 3)];
    acc += fp4_lut[comp1 >> 4]  * sh_A[g*4 + 4 + ((meta>>6) & 3)];
}
```

**Problem**: The weight layout was `[N, K/4]` (row-major in N). Adjacent threads read from rows that are `K/4` bytes apart -- completely uncoalesced. On GB10's LPDDR5X memory, this is catastrophic. Every 128-byte cache line fetch retrieves data for only one thread.

**Lesson**: Memory coalescing isn't optional.

---

### Iteration 2: Transpose for Coalescing

**File: `sparse_fp4_gemv_v2.cu`**

The fix was straightforward: transpose all weight tensors so the N dimension is contiguous.

```
Before: B_comp[N, K/4]   -- adjacent threads read different rows (uncoalesced)
After:  B_comp_T[K/4, N] -- adjacent threads read adjacent bytes (perfectly coalesced)
```

With the transposed layout, thread `t` reads `B_comp_T[g, n_base + t]` -- adjacent threads read adjacent bytes, achieving perfect 128-byte cache line utilization.

We also introduced **packed metadata** (`Meta_T_pk[K/8, N]`): two 4-bit group indices packed into one byte, halving metadata memory.

**Result**: ~2x speedup over v1 from coalescing alone. But we were still limited by the single-K-loop architecture requiring the full activation vector in shared memory.

---

### Iteration 3: K-Tiling and Partial Sums

**File: `sparse_fp4_v3.cu`**

Loading `K=2048` floats into shared memory (8 KB) limits occupancy. We split the K dimension into tiles of `TILE_K=64`:

```
Grid: (N_blocks, K_blocks, 1)
Each block processes: N_PER_BLOCK columns × TILE_K rows
Output: C[K_blocks, N] partial sums → reduced via .sum(0)
```

Each block only needs `TILE_K=64` floats (256 bytes) of shared memory for the A tile, plus 16 floats for the FP4 lookup table. This dramatically improved occupancy.

We also introduced `THREAD_N=4`: each thread processes 4 adjacent N-elements using `uint32` reads (4 bytes at once), extracting individual bytes with bit shifts. This amortizes instruction overhead across 4 outputs.

**Template**: `<THREADS=128, THREAD_N=4, TILE_K=64>`
- `N_PER_BLOCK = 128 * 4 = 512`
- Shared memory: ~320 bytes
- K-blocks: `2048/64 = 32` partial sums per output

**Result**: Higher occupancy, better instruction-level parallelism. The `.sum(0)` reduction in PyTorch added overhead, but the kernel itself was significantly faster.

---

### Iteration 4: cp.async Pipelining (Dead End)

**File: `sparse_fp4_v4.cu`**

We tried NVIDIA's `cp.async` instructions to overlap global memory loads with computation using a 2-stage pipeline:

```cuda
// Stage 0: Prefetch first sub-tile
cp_async_4(&sh_B[0][...], &B_comp_T[...]);
cp_async_commit();

for (sub = 0; sub < num_subs; sub++) {
    // Fetch next stage while computing current
    cp_async_4(&sh_B[next_stage][...], &B_comp_T[next_sub]);
    cp_async_commit();
    cp_async_wait_group(1);  // Wait for current stage

    compute_on(sh_B[current_stage]);
}
```

We implemented three configurations trading threadblock size for tile depth:

| Config | Threads | TILE_K | Shared Mem | Occupancy |
|--------|---------|--------|------------|-----------|
| A | 128 | 64 | 8.5 KB | 50% |
| B | 64 | 32 | 4.5 KB | 75% |
| C | 128 | 128 | 15 KB | 37.5% |

**Result**: Pipelining didn't help. On GB10, the bottleneck is DRAM bandwidth (273 GB/s), not load latency. The hardware's out-of-order execution already hides latency well enough. The extra shared memory for double-buffering actually hurt occupancy.

**Lesson**: On bandwidth-bound workloads, latency hiding through software pipelining provides minimal benefit. The hardware scheduler does a good enough job.

---

### Iteration 5: Batched Multi-Expert + 16-byte Loads

**File: `sparse_fp4_v5.cu`**

Two changes: batched processing and wider loads.

**Batching**: Instead of launching one kernel per expert, we use `blockIdx.z` to index into an array of expert IDs:

```cuda
Grid: (N_blocks, K_blocks, E_active)  // E_active = num tokens × topk
int expert_total = expert_ids[blockIdx.z];
// All weight reads offset by expert_total
```

This eliminates Python-level loops over experts and reduces kernel launch overhead.

**16-byte cp.async**: We tried `cp.async.cg.shared.global` for 16-byte (128-bit) loads, bypassing L1 cache to reduce pollution. This required `THREAD_N=16` to align with the 16-byte granularity.

**Result**: Batching was essential and carried forward to all future versions. The 16-byte loads showed marginal improvement -- again, bandwidth-bound, not latency-bound.

---

### Iteration 6: Simplification -- Direct Global Loads

**File: `sparse_fp4_v6.cu`**

We stripped out all `cp.async` complexity and went back to direct global memory reads:

```cuda
// Just read directly -- hardware handles everything
uint32_t comp0_4 = *reinterpret_cast<const uint32_t*>(&B_ptr[gi * N]);
uint32_t comp1_4 = *reinterpret_cast<const uint32_t*>(&B_ptr[(gi + 1) * N]);
uint32_t meta_4  = *reinterpret_cast<const uint32_t*>(&M_ptr[mi * N]);
```

Shared memory shrank to just ~320 bytes (16 LUT floats + TILE_K activation floats). This maximized occupancy.

We also experimented with an **interleaved format**: packing compressed weights, metadata, and scales into a single tensor `[E, K/8, 3*N]` so all three reads for a group come from adjacent memory. This showed marginal improvement since all three reads tend to hit the same DRAM page anyway.

**Result**: Matched v4/v5 performance with far simpler code. Direct loads became the standard approach.

---

### Iteration 7: FP8 Scales + Fused MoE (Production)

**File: `sparse_fp4_v7.cu`**

The real model uses FP8 E4M3FN block scales (one per 8 K-elements per output) plus a float32 global scale per expert. Previous iterations ignored scales entirely. v7 added full dequantization:

```
value = fp4_lut[nibble] * fp8_to_float(block_scale) * global_scale
```

**FP8 Decoding**: Rather than computing `fp8_to_float()` with branching bit manipulation in every thread, we precompute a 256-entry lookup table in constant memory:

```cuda
__constant__ float c_fp8_lut[256];  // Initialized once at module load
// ...
float scale = c_fp8_lut[scale_byte] * g_scale;  // Single LUT lookup
```

**The Inner Loop** (processing 8 K-elements per iteration):

```cuda
for (int gi = g_start; gi < g_end; gi += 2) {
    // 4 coalesced uint32 reads: comp0, comp1, meta, scale
    uint32_t comp0_4 = *(uint32_t*)(&B_ptr[gi * N]);
    uint32_t comp1_4 = *(uint32_t*)(&B_ptr[(gi+1) * N]);
    uint32_t meta_4  = *(uint32_t*)(&M_ptr[mi * N]);
    uint32_t scale_4 = *(uint32_t*)(&S_ptr[si * N]);

    for (int j = 0; j < 4; j++) {  // 4 N-elements per thread
        uint8_t comp0 = (comp0_4 >> (j*8)) & 0xFF;
        uint8_t comp1 = (comp1_4 >> (j*8)) & 0xFF;
        uint8_t meta  = (meta_4  >> (j*8)) & 0xFF;
        float scale   = c_fp8_lut[(scale_4 >> (j*8)) & 0xFF] * g_scale;

        // 4 multiply-adds: decompress, gather, accumulate
        float sum = fp4_lut[comp0 & 0xF] * A[k + (meta & 3)]
                  + fp4_lut[comp0 >> 4]  * A[k + ((meta>>2) & 3)]
                  + fp4_lut[comp1 & 0xF] * A[k+4 + ((meta>>4) & 3)]
                  + fp4_lut[comp1 >> 4]  * A[k+4 + ((meta>>6) & 3)];
        acc[j] += sum * scale;
    }
}
```

Each iteration reads 16 bytes (4 uint32s) and performs 16 multiply-adds. The ratio of compute to memory access is intentionally low -- we're bandwidth-bound, so we want to minimize bytes read, not maximize FLOPs.

---

### The Fused C++ MoE Forward Pass

The kernel alone only handles one GEMM. A full MoE layer requires:

1. BF16 to FP16 cast + token replication (duplicate each token for each of its topk experts)
2. GEMM1: gate+up projection (`[M*topk, K] @ [K, 2*N]`)
3. SiLU activation: `silu(gate) * up`
4. GEMM2: down projection (`[M*topk, N] @ [N, K]`)
5. Weighted reduction across experts

Originally, steps 1-5 were orchestrated in Python. Profiling showed that **72% of per-layer time was Python overhead** -- tensor allocation, dtype conversion, `torch.repeat_interleave`, etc. -- not the actual GEMM kernel.

We wrote `fused_sparse_moe_v7()` in C++ to do everything in a single call:

```cpp
torch::Tensor fused_sparse_moe_v7(
    torch::Tensor hidden_states,    // [M, K] bfloat16
    torch::Tensor topk_weights,     // [M, topk] float32
    torch::Tensor topk_ids,         // [M, topk] int32
    torch::Tensor expert_map,       // expert ID remapping
    // W13 weights (gate+up, sparse format)
    torch::Tensor w13_comp, w13_meta, w13_scale, w13_g_scales,
    // W2 weights (down, sparse format)
    torch::Tensor w2_comp, w2_meta, w2_scale, w2_g_scales,
    int inter_size,
    bool apply_router_weight_on_input
) {
    // 1. Map expert IDs
    // 2. BF16->FP16 + replicate tokens (custom kernel)
    // 3. GEMM1 via batched_sparse_v7_kernel + .sum(1)
    // 4. SiLU activation (custom kernel)
    // 5. GEMM2 via batched_sparse_v7_kernel + .sum(1)
    // 6. Weighted reduction (custom kernel)
    return output;  // [M, K] float16
}
```

Helper kernels:
- **`silu_mul_kernel`**: Fused `silu(gate) * up` -- avoids materializing intermediate tensors
- **`bf16_to_fp16_replicate_kernel`**: BF16-to-FP16 conversion + token replication in one pass
- **`weighted_reduce_fp16_kernel`**: Weighted sum across topk experts

This eliminated the Python overhead, improving from 31.8 to 33.3 tok/s in eager mode (+5%).

---

### Integrating with vLLM: The Monkey-Patch

vLLM's MoE dispatch chain is deeply layered:

```
ModelOptNvFp4FusedMoE.apply()
  -> FusedMoEModularKernel.forward()
    -> MarlinExperts.apply()
      -> fused_marlin_moe()
        -> ops.moe_wna16_marlin_gemm()  (CUDA kernel)
```

We needed to intercept `MarlinExperts.apply()` without touching vLLM's source code. The strategy:

1. **Before Marlin repacking destroys the original weights**, clone them to CPU
2. **Let Marlin's `process_weights_after_loading` run normally** -- this builds the full `FusedMoEModularKernel` with all required interface methods
3. **Convert the cloned weights to sparse format** on GPU
4. **Replace `MarlinExperts.apply` with our closure** as an instance attribute

```python
def _patched_process_weights_after_loading(self, layer):
    # 1. Clone weights before Marlin repacking destroys them
    w13_weight = layer.w13_weight.data.cpu().clone()
    w2_weight = layer.w2_weight.data.cpu().clone()
    # ... scales too

    # 2. Let Marlin do its thing
    _original_process_weights(self, layer)

    # 3. Convert to sparse format
    w13_comp, w13_meta, w13_sc, w13_g = convert_weight_batch_to_v7(w13_weight, ...)
    w2_comp, w2_meta, w2_sc, w2_g = convert_weight_batch_to_v7(w2_weight, ...)

    # 4. Replace apply() with our kernel
    experts = self.moe_mk.fused_experts
    experts.apply = _make_v7_apply_fn(w13_comp, ..., w2_comp, ...)
```

The closure captures the sparse weights and calls `fused_sparse_moe_v7()`. Since Python doesn't inject `self` for instance-attribute functions (only for class methods), the closure works directly.

---

### The Bugs

#### Bug 1: CUDA Stream Mismatch

The kernel worked perfectly with `CUDA_LAUNCH_BLOCKING=1` but crashed with `illegal memory access` without it.

**Root cause**: Our custom kernels launched on CUDA's legacy stream 0 (`<<<grid, threads, smem>>>`), but PyTorch uses per-thread default streams. Without explicit synchronization, the kernel could execute before its input tensors were ready.

**Fix**: Pass the current PyTorch stream to every kernel launch:

```cuda
auto stream = at::cuda::getCurrentCUDAStream();
kernel<<<grid, threads, smem, stream>>>(...);
```

#### Bug 2: CUDAStreamGuard for CUDA Graphs

With `max_num_seqs=128`, vLLM captures CUDA graphs for 51 different batch sizes. Our fused function called PyTorch operations (`.sum()`, `.index()`) that internally launched on the wrong stream during graph capture.

**Fix**: Wrap the entire function with a stream guard:

```cuda
auto stream = at::cuda::getCurrentCUDAStream();
c10::cuda::CUDAStreamGuard stream_guard(stream);
```

#### Bug 3: OOM from Double Weight Storage

With MTP speculative decoding, the model needs a draft model in memory. Storing both Marlin-repacked weights AND our sparse weights caused OOM (78.77 GiB vs 75.7 GiB budget).

**Fix**: After patching, replace Marlin weights with zero-strided dummy tensors that preserve shape metadata but use only 1 element of storage:

```python
# Preserve shape for moe_problem_size() but free all storage
stub = torch.zeros(1, device='cuda', dtype=param.dtype)
param.data = stub.as_strided(original_shape, [0] * ndim)
```

This freed ~43.5 GB across 48 MoE layers (906 MB each).

#### Bug 4: Read-Only Properties

We tried `delattr(experts, 'w1_scale')` but `w1_scale` is a `@property` on a parent class, not a regular attribute. The actual weights live on `layer.w13_weight` as `nn.Parameter`s. Understanding vLLM's `FusedMoEPermuteExpertsUnpermute -> MarlinExpertsBase -> MarlinExperts` class hierarchy was essential.

---

### Failed Approaches

#### Attempt: Large K Tiles (TK=2048)

To eliminate the `.sum(1)` k-block reduction entirely, we tried `TK=2048` (matching the full K dimension, so only 1 k-block). Result: **27.0 tok/s** -- worse than TK=64's 31.8 tok/s. With only 20 blocks for GEMM1, we couldn't saturate the GPU's 84 SMs.

#### Attempt: atomicAdd Accumulation

Instead of writing partial sums and reducing, we tried `atomicAdd` to accumulate directly into the output:

```cuda
atomicAdd(&output[expert_active * N + n], __float2half(acc[j]));
```

Result: **31.1 tok/s** -- worse than the `.sum()` approach (33.3 tok/s). Contention from 32 k-blocks all atomicAdd-ing to the same locations, plus the need for FP32 accumulators (2x memory), made it a net loss.

#### Attempt: cp.async Pipelining (v4)

As described above, software pipelining was complexity without payoff on this bandwidth-bound workload. We spent time implementing 3 configurations before concluding that direct global loads (v6) matched performance with far less code.

---

### Final Performance

All measurements on DGX Spark GB10, same Docker container, same model, BS=1:

| Configuration | Throughput | Memory |
|---|---|---|
| Marlin eager (no MTP) | 31.1 tok/s | 47.3 GiB |
| **V7 sparse eager (no MTP)** | **33.3 tok/s (+7%)** | 47.3 GiB |
| Marlin + MTP eager | 44.4 tok/s | 47.3 GiB |
| **V7 sparse + MTP eager** | **43.6 tok/s** | 38.3 GiB |
| Marlin + MTP + CUDA graphs | 54.9 tok/s | 47.3 GiB |
| **V7 sparse + MTP + CUDA graphs** | **54.6 tok/s** | **38.3 GiB** |

The v7 sparse kernel achieves **production parity** with Marlin at 54.6 tok/s while using **9 GiB less GPU memory** (38.3 vs 47.3 GiB). The memory savings come from freeing the Marlin-repacked weights after converting to sparse format.

In eager mode (no CUDA graphs), the sparse kernel is **7% faster** than Marlin. This advantage narrows with CUDA graphs because the non-MoE layers (attention, Mamba SSM, layer norms) dominate and benefit equally from graph replay.

---

### What We Learned

1. **Memory coalescing is non-negotiable.** Transposing the weight layout (v1 to v2) was the single largest improvement. Everything else was incremental.

2. **Software pipelining doesn't help bandwidth-bound kernels.** We spent significant effort on `cp.async` pipelining (v4) only to find that direct global loads (v6) performed identically with half the code. The hardware's out-of-order execution already hides load latency.

3. **Python overhead dominates at small batch sizes.** Moving from a Python-orchestrated MoE forward to a single C++ fused function improved throughput by 5% -- more than any kernel micro-optimization.

4. **CUDA streams are implicit but critical.** Both stream bugs (missing stream parameter, missing CUDAStreamGuard) worked fine in synchronous mode but crashed in production. Always pass the PyTorch stream explicitly.

5. **Sparsity helps most when you're already bandwidth-bound.** The 2:4 pruning removes 25% of weight reads, which matters when the memory bus is the bottleneck. At larger batch sizes where compute becomes the limiter, the benefit would shift to reduced FLOPs.

6. **Monkey-patching frameworks requires understanding their internals deeply.** vLLM's layered MoE dispatch (quantization method -> modular kernel -> fused experts -> marlin kernel) has many interception points. Choosing the right one (instance-level `.apply` override) kept us compatible with CUDA graphs, MTP speculative decoding, and future vLLM updates.

---

### Repository

The complete kernel evolution (v1-v7), vLLM integration patch, and benchmark scripts are available at: [github.com/Avarok-Cybersecurity/dgx-vllm](https://github.com/Avarok-Cybersecurity/dgx-vllm) in the `sparse_fp4_kernel/` directory.

**Key files:**
- `sparse_fp4_v7.cu` -- Production kernel (749 lines)
- `sparse_v7_moe_patch.py` -- vLLM monkey-patch
- `setup_v7.py` -- Build script (`python setup_v7.py install`)
- `sparse_fp4_gemv.cu` through `sparse_fp4_v6.cu` -- Kernel evolution history
