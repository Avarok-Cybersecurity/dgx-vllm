#!/usr/bin/env python3
"""Benchmark v7 vs v8 sparse FP4 GEMV kernels.

Tests the raw GEMV kernel and the full fused MoE forward on synthetic data
matching Qwen3-Next-80B-A3B dimensions (E=512, inter=512, hidden=2048, topk=10).

Usage (inside container):
  python setup_v7.py build_ext --inplace
  python setup_v8.py build_ext --inplace
  python bench_v7_v8.py
"""
import torch
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Model dimensions (Qwen3-Next-80B-A3B MoE)
E_TOTAL = 512       # total experts
TOPK = 10           # experts per token
INTER = 512         # moe_intermediate_size
HIDDEN = 2048       # hidden_size
M = 1               # batch size (single token decode)
N_SCALE_GROUPS_13 = HIDDEN // 8   # one scale per 8 K-elements
N_SCALE_GROUPS_2  = INTER // 8


def make_synthetic_weights(E, N, K, n_scale_groups, device='cuda'):
    """Create synthetic v7-format sparse weights."""
    comp_T  = torch.randint(0, 256, (E, K // 4, N), dtype=torch.uint8, device=device)
    meta_T  = torch.randint(0, 256, (E, K // 8, N), dtype=torch.uint8, device=device)
    scale_T = torch.randint(1, 200, (E, n_scale_groups, N), dtype=torch.uint8, device=device)
    g_scales = torch.rand(E, dtype=torch.float32, device=device) * 0.01 + 0.001
    return comp_T, meta_T, scale_T, g_scales


def bench_gemv(fn, A, comp, meta, scale, g_scale, eids, warmup=20, iters=200):
    """Benchmark a single GEMV call."""
    for _ in range(warmup):
        fn(A, comp, meta, scale, g_scale, eids)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(A, comp, meta, scale, g_scale, eids)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms per call


def bench_fused_moe(fn, hidden, tw, ti, emap,
                    w13c, w13m, w13s, w13g,
                    w2c, w2m, w2s, w2g,
                    inter, arwoi, warmup=20, iters=200):
    """Benchmark fused MoE forward."""
    for _ in range(warmup):
        fn(hidden, tw, ti, emap,
           w13c, w13m, w13s, w13g,
           w2c, w2m, w2s, w2g, inter, arwoi)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(hidden, tw, ti, emap,
           w13c, w13m, w13s, w13g,
           w2c, w2m, w2s, w2g, inter, arwoi)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    device = 'cuda'
    print("=" * 60)
    print("Sparse FP4 GEMV Benchmark: v7 vs v8")
    print("=" * 60)
    print(f"Model dims: E={E_TOTAL}, topk={TOPK}, inter={INTER}, hidden={HIDDEN}")
    print()

    # Import both modules
    try:
        import sparse_fp4_v7 as v7
        have_v7 = True
        print("[OK] sparse_fp4_v7 loaded")
    except ImportError as e:
        print(f"[SKIP] sparse_fp4_v7 not available: {e}")
        have_v7 = False

    try:
        import sparse_fp4_v8 as v8
        have_v8 = True
        print("[OK] sparse_fp4_v8 loaded")
    except ImportError as e:
        print(f"[SKIP] sparse_fp4_v8 not available: {e}")
        have_v8 = False

    if not have_v7 and not have_v8:
        print("Neither v7 nor v8 available. Build with:")
        print("  python setup_v7.py build_ext --inplace")
        print("  python setup_v8.py build_ext --inplace")
        return

    # Create synthetic data
    E_active = M * TOPK
    expert_ids = torch.randint(0, E_TOTAL, (E_active,), dtype=torch.int32, device=device)

    # gate_up projection: A=[E_active, K=HIDDEN] -> [E_active, 2*INTER]
    A_13 = torch.randn(E_active, HIDDEN, dtype=torch.float16, device=device)
    w13_comp, w13_meta, w13_scale, w13_g = make_synthetic_weights(
        E_TOTAL, 2 * INTER, HIDDEN, N_SCALE_GROUPS_13, device)

    # down projection: A=[E_active, K=INTER] -> [E_active, HIDDEN]
    A_2 = torch.randn(E_active, INTER, dtype=torch.float16, device=device)
    w2_comp, w2_meta, w2_scale, w2_g = make_synthetic_weights(
        E_TOTAL, HIDDEN, INTER, N_SCALE_GROUPS_2, device)

    print()
    print("-" * 60)
    print("Test 1: Raw GEMV (gate_up: K=2048 -> N=1024)")
    print("-" * 60)

    if have_v7:
        t_v7 = bench_gemv(v7.batched_sparse_v7,
                          A_13, w13_comp, w13_meta, w13_scale, w13_g, expert_ids)
        print(f"  v7: {t_v7:.3f} ms")

    if have_v8:
        t_v8 = bench_gemv(v8.batched_sparse_v8,
                          A_13, w13_comp, w13_meta, w13_scale, w13_g, expert_ids)
        print(f"  v8: {t_v8:.3f} ms")

    if have_v7 and have_v8:
        speedup = t_v7 / t_v8
        print(f"  speedup: {speedup:.2f}x {'(v8 faster)' if speedup > 1 else '(v7 faster)'}")

    # Verify correctness
    if have_v7 and have_v8:
        out_v7 = v7.batched_sparse_v7(A_13, w13_comp, w13_meta, w13_scale, w13_g, expert_ids)
        out_v8 = v8.batched_sparse_v8(A_13, w13_comp, w13_meta, w13_scale, w13_g, expert_ids)
        diff = (out_v7.float() - out_v8.float()).abs().max().item()
        print(f"  max abs diff: {diff:.6f}")

    print()
    print("-" * 60)
    print("Test 2: Raw GEMV (down: K=512 -> N=2048)")
    print("-" * 60)

    if have_v7:
        t_v7 = bench_gemv(v7.batched_sparse_v7,
                          A_2, w2_comp, w2_meta, w2_scale, w2_g, expert_ids)
        print(f"  v7: {t_v7:.3f} ms")

    if have_v8:
        t_v8 = bench_gemv(v8.batched_sparse_v8,
                          A_2, w2_comp, w2_meta, w2_scale, w2_g, expert_ids)
        print(f"  v8: {t_v8:.3f} ms")

    if have_v7 and have_v8:
        speedup = t_v7 / t_v8
        print(f"  speedup: {speedup:.2f}x {'(v8 faster)' if speedup > 1 else '(v7 faster)'}")

    print()
    print("-" * 60)
    print("Test 3: Fused MoE forward (full gate_up + SiLU + down + reduce)")
    print("-" * 60)

    hidden = torch.randn(M, HIDDEN, dtype=torch.bfloat16, device=device)
    topk_weights = torch.softmax(torch.randn(M, TOPK, device=device), dim=-1)
    topk_ids = torch.randint(0, E_TOTAL, (M, TOPK), dtype=torch.int32, device=device)
    expert_map = torch.empty(0, dtype=torch.int32, device=device)

    if have_v7:
        t_v7 = bench_fused_moe(v7.fused_sparse_moe_v7, hidden, topk_weights, topk_ids,
                               expert_map, w13_comp, w13_meta, w13_scale, w13_g,
                               w2_comp, w2_meta, w2_scale, w2_g, INTER, False)
        print(f"  v7: {t_v7:.3f} ms")

    if have_v8:
        t_v8 = bench_fused_moe(v8.fused_sparse_moe_v8, hidden, topk_weights, topk_ids,
                               expert_map, w13_comp, w13_meta, w13_scale, w13_g,
                               w2_comp, w2_meta, w2_scale, w2_g, INTER, False)
        print(f"  v8: {t_v8:.3f} ms")

    if have_v7 and have_v8:
        speedup = t_v7 / t_v8
        print(f"  speedup: {speedup:.2f}x {'(v8 faster)' if speedup > 1 else '(v7 faster)'}")

    if have_v7 and have_v8:
        out_v7 = v7.fused_sparse_moe_v7(hidden, topk_weights, topk_ids, expert_map,
                                         w13_comp, w13_meta, w13_scale, w13_g,
                                         w2_comp, w2_meta, w2_scale, w2_g, INTER, False)
        out_v8 = v8.fused_sparse_moe_v8(hidden, topk_weights, topk_ids, expert_map,
                                         w13_comp, w13_meta, w13_scale, w13_g,
                                         w2_comp, w2_meta, w2_scale, w2_g, INTER, False)
        diff = (out_v7.float() - out_v8.float()).abs().max().item()
        rel_diff = diff / (out_v7.float().abs().max().item() + 1e-8)
        print(f"  max abs diff: {diff:.6f} (relative: {rel_diff:.6f})")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
