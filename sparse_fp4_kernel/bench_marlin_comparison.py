#!/usr/bin/env python3
"""Direct comparison: v6 sparse GEMV vs Marlin dense GEMV.

Calls the Marlin kernel through vLLM's ops interface with proper
NVFP4 weight format to get a true apples-to-apples comparison.
"""
import torch, time, sys

# Try to import our v6 kernel
try:
    import sparse_fp4_v6 as sp6
    HAS_V6 = True
except ImportError:
    HAS_V6 = False
    print("WARNING: v6 not available")

# Try to import vLLM's Marlin ops
try:
    from vllm._custom_ops import gptq_marlin_gemm, gptq_marlin_repack
    HAS_MARLIN = True
except ImportError:
    try:
        import vllm._C as vllm_C
        HAS_MARLIN = hasattr(vllm_C, 'ops')
    except ImportError:
        HAS_MARLIN = False
    if not HAS_MARLIN:
        print("WARNING: Marlin ops not available")

def bench_warm(fn, warmup=200, iters=1000):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.time() - t) / iters * 1e6

def bench_cold(fn, flush_buf, warmup=50, iters=500):
    for _ in range(warmup):
        _ = flush_buf.sum()
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = flush_buf.sum()
    torch.cuda.synchronize()
    flush_cost = (time.time() - t0) / iters * 1e6
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = flush_buf.sum()
        fn()
    torch.cuda.synchronize()
    total = (time.time() - t0) / iters * 1e6
    return total - flush_cost

flush_buf = torch.randn(64 * 1024 * 1024 // 4, device='cuda', dtype=torch.float32)

print(f"{'='*80}")
print(f"v6 Sparse GEMV vs Marlin Dense GEMV — Direct Comparison")
print(f"{'='*80}")

# Test single-expert GEMV to compare kernel quality
for name, N, K in [("gate_proj", 512, 2048), ("down_proj", 2048, 512)]:
    print(f"\n--- {name} [N={N}, K={K}] ---")

    dense_bytes = N * K // 2  # per expert, FP4 packed
    sparse_bytes = N * K // 4 + N * K // 8  # compressed + metadata

    # ---- Marlin ----
    if HAS_MARLIN:
        try:
            # Create Marlin-format weights for single GEMV
            # Marlin expects: B_q_weight [K/pack_factor, N] in specific permuted layout
            # For FP4 (num_bits=4): pack_factor = 8 (8 FP4 per int32)
            # But Marlin uses its own repack format

            # First create "original" packed weight in standard format
            # Standard FP4: [K, N] with 2 values per byte = [K/2, N] uint8
            # Marlin: [K/8, N] int32 (8 FP4 per int32, different packing)

            # Use gptq_marlin_repack to convert
            pack_factor = 8  # FP4: 8 values per int32
            B_orig = torch.randint(0, 2**32, (K // pack_factor, N),
                                   device='cuda', dtype=torch.int32)

            # Repack for Marlin layout
            # gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)
            perm = torch.empty(0, device='cuda', dtype=torch.int32)  # no act_order
            B_marlin = gptq_marlin_repack(B_orig, perm, K, N, 4)  # num_bits=4

            # Scales: [N // group_size, K] or [1, N] for channel-wise
            # For NVFP4 with group_size=K (per-channel): [1, N]
            # Actually Marlin uses [ceil(K/group_size), N] scales
            group_size = 128  # typical for NVFP4
            n_groups = (K + group_size - 1) // group_size
            scales = torch.ones(n_groups, N, device='cuda', dtype=torch.float16)

            # Workspace
            workspace = torch.zeros(N, device='cuda', dtype=torch.int32)

            # A input: [M=1, K] in FP16
            A_marlin = torch.randn(1, K, device='cuda', dtype=torch.float16)

            # Sort indices (for act_order=False, just identity)
            g_idx = torch.empty(0, device='cuda', dtype=torch.int32)
            sort_indices = torch.empty(0, device='cuda', dtype=torch.int32)

            # gptq_marlin_gemm(a, b_q_weight, b_scales, b_zeros, g_idx,
            #                  perm, workspace, b_q_type, size_m, size_n, size_k,
            #                  is_k_full, has_zp, use_fp32_reduce, is_zp_float)
            def run_marlin():
                return gptq_marlin_gemm(
                    A_marlin, B_marlin, scales,
                    torch.empty(0, device='cuda', dtype=torch.int32),  # b_zeros
                    g_idx, sort_indices, workspace,
                    8,   # b_q_type: vllm scalar type for FP4 (FE2M1f = 8)
                    1,   # size_m
                    N,   # size_n
                    K,   # size_k
                    True,  # is_k_full
                    False, # has_zp
                    False, # use_fp32_reduce
                    False, # is_zp_float
                )

            # Test it works
            try:
                C_marlin = run_marlin()
                print(f"  Marlin output: {list(C_marlin.shape)}")

                m_warm = bench_warm(run_marlin)
                m_cold = bench_cold(run_marlin, flush_buf)

                print(f"  Marlin Warm: {m_warm:6.1f} us ({dense_bytes/(m_warm*1e-6)/1e9:5.0f} GB/s)")
                print(f"  Marlin Cold: {m_cold:6.1f} us ({dense_bytes/(m_cold*1e-6)/1e9:5.0f} GB/s)")
            except Exception as e:
                print(f"  Marlin FAILED: {e}")
                HAS_MARLIN = False
        except Exception as e:
            print(f"  Marlin setup FAILED: {e}")
            HAS_MARLIN = False

    # ---- v6 ----
    if HAS_V6:
        # Single expert comparison (not batched)
        E_total = 1
        E_active = 1
        expert_ids = torch.zeros(1, device='cuda', dtype=torch.int32)
        A_v6 = torch.randn(1, K, device='cuda', dtype=torch.float16)
        B_dense_T = torch.randint(0, 256, (1, K//2, N), device='cuda', dtype=torch.uint8)
        B_comp_T = torch.randint(0, 256, (1, K//4, N), device='cuda', dtype=torch.uint8)
        Meta_T_pk = torch.randint(0, 256, (1, K//8, N), device='cuda', dtype=torch.uint8)

        d6w = bench_warm(lambda: sp6.batched_dense_v6(A_v6, B_dense_T, expert_ids))
        s6w = bench_warm(lambda: sp6.batched_sparse_v6(A_v6, B_comp_T, Meta_T_pk, expert_ids))
        d6c = bench_cold(lambda: sp6.batched_dense_v6(A_v6, B_dense_T, expert_ids), flush_buf)
        s6c = bench_cold(lambda: sp6.batched_sparse_v6(A_v6, B_comp_T, Meta_T_pk, expert_ids), flush_buf)

        print(f"  v6 Dense Warm: {d6w:6.1f} us ({dense_bytes/(d6w*1e-6)/1e9:5.0f} GB/s)")
        print(f"  v6 Dense Cold: {d6c:6.1f} us ({dense_bytes/(d6c*1e-6)/1e9:5.0f} GB/s)")
        print(f"  v6 Sparse Warm: {s6w:6.1f} us ({sparse_bytes/(s6w*1e-6)/1e9:5.0f} GB/s)")
        print(f"  v6 Sparse Cold: {s6c:6.1f} us ({sparse_bytes/(s6c*1e-6)/1e9:5.0f} GB/s)")

        del A_v6, B_dense_T, B_comp_T, Meta_T_pk, expert_ids

    # ---- Batched comparison (10 experts) ----
    if HAS_V6:
        print(f"\n  --- Batched 10/512 experts ---")
        E_total = 512
        E_active = 10
        expert_ids = torch.randperm(E_total, device='cuda')[:E_active].to(torch.int32).sort().values
        A = torch.randn(E_active, K, device='cuda', dtype=torch.float16)
        B_dense_T = torch.randint(0, 256, (E_total, K//2, N), device='cuda', dtype=torch.uint8)
        B_comp_T = torch.randint(0, 256, (E_total, K//4, N), device='cuda', dtype=torch.uint8)
        Meta_T_pk = torch.randint(0, 256, (E_total, K//8, N), device='cuda', dtype=torch.uint8)

        batched_dense_bytes = E_active * N * K // 2
        batched_sparse_bytes = E_active * (N * K // 4 + N * K // 8)

        d6bw = bench_warm(lambda: sp6.batched_dense_v6(A, B_dense_T, expert_ids))
        s6bw = bench_warm(lambda: sp6.batched_sparse_v6(A, B_comp_T, Meta_T_pk, expert_ids))
        d6bc = bench_cold(lambda: sp6.batched_dense_v6(A, B_dense_T, expert_ids), flush_buf)
        s6bc = bench_cold(lambda: sp6.batched_sparse_v6(A, B_comp_T, Meta_T_pk, expert_ids), flush_buf)

        print(f"  v6 Batched Dense Cold: {d6bc:6.1f} us ({batched_dense_bytes/(d6bc*1e-6)/1e9:5.0f} GB/s)")
        print(f"  v6 Batched Sparse Cold: {s6bc:6.1f} us ({batched_sparse_bytes/(s6bc*1e-6)/1e9:5.0f} GB/s)")
        print(f"  Sparse/Dense: {s6bc/d6bc:.3f}")

        del A, B_dense_T, B_comp_T, Meta_T_pk, expert_ids

    torch.cuda.empty_cache()

# Summary
print(f"\n{'='*80}")
print("SUMMARY: Per-projection latency for 10 experts (DRAM-cold)")
print(f"{'='*80}")
print("Note: Marlin times are per-expert (single GEMV). v6 is batched (all 10 experts).")
print("In production, Marlin also batches experts via block-cooperative dispatch.")
