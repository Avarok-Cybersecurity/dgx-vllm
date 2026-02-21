#!/usr/bin/env python3
"""Benchmark v7 (v6 + FP8 block scales) vs MoE Marlin — real Qwen3-Next-80B dims.

Also validates correctness of FP8 scale dequantization against PyTorch reference.
"""
import torch, time, sys, traceback

try:
    import sparse_fp4_v7 as sp7
    HAS_V7 = True
except ImportError:
    HAS_V7 = False
    print("WARNING: v7 not available")

try:
    import sparse_fp4_v6 as sp6
    HAS_V6 = True
except ImportError:
    HAS_V6 = False

try:
    from vllm._custom_ops import (moe_wna16_marlin_gemm, moe_align_block_size,
                                   gptq_marlin_moe_repack)
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        nvfp4_marlin_process_scales, nvfp4_marlin_process_global_scale
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales
    )
    fp4_type = scalar_types.float4_e2m1f
    HAS_MARLIN = True
except Exception as e:
    print(f"WARNING: Marlin MoE not available: {e}")
    HAS_MARLIN = False

def bench_warm(fn, w=200, n=1000):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1e6

def bench_cold(fn, fb, w=50, n=500):
    for _ in range(w): _ = fb.sum(); fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n): _ = fb.sum()
    torch.cuda.synchronize()
    fc = (time.time() - t0) / n * 1e6
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n): _ = fb.sum(); fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1e6 - fc

fb = torch.randn(64 * 1024 * 1024 // 4, device='cuda', dtype=torch.float32)

E_total = 512
E_active = 10
hidden = 2048
moe_inter = 512
group_size = 16  # NVFP4 native
block_size = 8

# FP4 E2M1 LUT (same as kernel constant memory)
FP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                        device='cuda', dtype=torch.float32)

def validate_dense_v7(N, K, E_active=2, E_total=4):
    """Validate v7 dense dequant against PyTorch reference."""
    print(f"  Validating dense v7 [N={N}, K={K}]...")

    # Create FP4 packed weights [E_total, K/2, N] (transposed)
    # Each byte = 2 FP4 nibbles
    B_dense_T = torch.randint(0, 256, (E_total, K // 2, N), device='cuda', dtype=torch.uint8)

    # Create FP8 block scales [E_total, K/8, N] (transposed from [N, K/8])
    # Use small positive values typical of real scales
    scales_T = torch.randint(56, 120, (E_total, K // 8, N), device='cuda', dtype=torch.uint8)

    # Global scales [E_total]
    g_scales = torch.rand(E_total, device='cuda', dtype=torch.float32) * 0.001 + 0.0001

    # Expert IDs
    expert_ids = torch.arange(E_active, device='cuda', dtype=torch.int32)
    A = torch.randn(E_active, K, device='cuda', dtype=torch.float16)

    # v7 kernel result
    C_v7 = sp7.batched_dense_v7(A, B_dense_T, scales_T, g_scales, expert_ids)

    # PyTorch reference
    C_ref = torch.zeros(E_active, N, device='cuda', dtype=torch.float32)
    for ei in range(E_active):
        eid = expert_ids[ei].item()
        gs = g_scales[eid].item()
        a = A[ei].float()
        for kp in range(K // 2):
            k0 = kp * 2
            k1 = kp * 2 + 1
            sg = kp // 4  # scale group (8 K per group)
            for n in range(N):
                byte = B_dense_T[eid, kp, n].item()
                nib0 = byte & 0x0F
                nib1 = (byte >> 4) & 0x0F
                s_byte = scales_T[eid, sg, n].item()
                # Convert FP8 E4M3FN to float
                sign = (s_byte >> 7) & 1
                exp = (s_byte >> 3) & 0xF
                mant = s_byte & 0x7
                if exp == 0 and mant == 0:
                    scale = 0.0
                elif exp == 0:
                    scale = (2**-6) * (mant / 8.0)
                    if sign: scale = -scale
                else:
                    scale = (2**(exp - 7)) * (1 + mant / 8.0)
                    if sign: scale = -scale
                scale *= gs
                v0 = FP4_LUT[nib0].item() * scale
                v1 = FP4_LUT[nib1].item() * scale
                C_ref[ei, n] += v0 * a[k0].item() + v1 * a[k1].item()

    C_ref_h = C_ref.half()
    max_diff = (C_v7.float() - C_ref.float()).abs().max().item()
    rel_diff = max_diff / (C_ref.float().abs().max().item() + 1e-8)
    print(f"    max_abs_diff={max_diff:.6f}, rel_diff={rel_diff:.6f} {'PASS' if rel_diff < 0.05 else 'FAIL'}")
    return rel_diff < 0.05


def validate_sparse_v7(N, K, E_active=2, E_total=4):
    """Validate v7 sparse dequant against PyTorch reference."""
    print(f"  Validating sparse v7 [N={N}, K={K}]...")

    # Create compressed sparse weights
    # B_comp_T: [E_total, K/4, N] — 2 non-zero values per group of 4 K
    B_comp_T = torch.randint(0, 256, (E_total, K // 4, N), device='cuda', dtype=torch.uint8)

    # Meta_T_pk: [E_total, K/8, N] — 2-bit indices for each non-zero
    Meta_T_pk = torch.zeros(E_total, K // 8, N, device='cuda', dtype=torch.uint8)
    # Fill with valid metadata (each 2-bit index must be 0-3)
    for e in range(E_total):
        for m in range(K // 8):
            for n in range(N):
                # 4 non-zeros from 8 K positions, each picks 1 of 4 in its group
                i0 = torch.randint(0, 4, (1,)).item()
                i1 = torch.randint(0, 4, (1,)).item()
                i2 = torch.randint(0, 4, (1,)).item()
                i3 = torch.randint(0, 4, (1,)).item()
                Meta_T_pk[e, m, n] = i0 | (i1 << 2) | (i2 << 4) | (i3 << 6)

    # Scales
    scales_T = torch.randint(56, 120, (E_total, K // 8, N), device='cuda', dtype=torch.uint8)
    g_scales = torch.rand(E_total, device='cuda', dtype=torch.float32) * 0.001 + 0.0001

    expert_ids = torch.arange(E_active, device='cuda', dtype=torch.int32)
    A = torch.randn(E_active, K, device='cuda', dtype=torch.float16)

    # v7 kernel result
    C_v7 = sp7.batched_sparse_v7(A, B_comp_T, Meta_T_pk, scales_T, g_scales, expert_ids)

    # PyTorch reference
    C_ref = torch.zeros(E_active, N, device='cuda', dtype=torch.float32)

    def fp8_to_float(b):
        sign = (b >> 7) & 1
        exp = (b >> 3) & 0xF
        mant = b & 0x7
        if exp == 0 and mant == 0: return 0.0
        if exp == 0:
            val = (2**-6) * (mant / 8.0)
        else:
            val = (2**(exp - 7)) * (1 + mant / 8.0)
        return -val if sign else val

    for ei in range(E_active):
        eid = expert_ids[ei].item()
        gs = g_scales[eid].item()
        a = A[ei].float()
        for gi in range(0, K // 4, 2):  # pairs of compressed groups
            mi = gi // 2
            k_base = gi * 4  # original K offset
            for n in range(N):
                comp0 = B_comp_T[eid, gi, n].item()
                comp1 = B_comp_T[eid, gi + 1, n].item()
                meta = Meta_T_pk[eid, mi, n].item()
                s_byte = scales_T[eid, mi, n].item()
                scale = fp8_to_float(s_byte) * gs

                # 4 non-zero values with metadata indices
                v0 = FP4_LUT[comp0 & 0xF].item() * scale
                v1 = FP4_LUT[(comp0 >> 4) & 0xF].item() * scale
                v2 = FP4_LUT[comp1 & 0xF].item() * scale
                v3 = FP4_LUT[(comp1 >> 4) & 0xF].item() * scale

                C_ref[ei, n] += v0 * a[k_base + (meta & 3)].item()
                C_ref[ei, n] += v1 * a[k_base + ((meta >> 2) & 3)].item()
                C_ref[ei, n] += v2 * a[k_base + 4 + ((meta >> 4) & 3)].item()
                C_ref[ei, n] += v3 * a[k_base + 4 + ((meta >> 6) & 3)].item()

    max_diff = (C_v7.float() - C_ref.float()).abs().max().item()
    rel_diff = max_diff / (C_ref.float().abs().max().item() + 1e-8)
    print(f"    max_abs_diff={max_diff:.6f}, rel_diff={rel_diff:.6f} {'PASS' if rel_diff < 0.05 else 'FAIL'}")
    return rel_diff < 0.05


# =====================================================================
# Main benchmark
# =====================================================================
print("=" * 80)
print("v7 (FP8 scales) vs MoE Marlin — Qwen3-Next-80B Dimensions")
print("=" * 80)

# Correctness validation first (small sizes for speed)
if HAS_V7:
    print("\n--- Correctness Validation ---")
    # N must be >= 512 (NPB) so all threads participate in __syncthreads__
    ok_d = validate_dense_v7(512, 64, E_active=2, E_total=4)
    ok_s = validate_sparse_v7(512, 64, E_active=2, E_total=4)
    if not ok_d or not ok_s:
        print("CORRECTNESS FAILED — aborting benchmark")
        sys.exit(1)
    print("  All correctness checks PASSED")

configs = [
    ("gate_up_proj", 2 * moe_inter, hidden),   # N=1024, K=2048
    ("down_proj", hidden, moe_inter),            # N=2048, K=512
]

for name, N, K in configs:
    print(f"\n--- {name} [N={N}, K={K}] ---")
    dense_bytes = E_active * N * K // 2
    sparse_bytes_no_scale = E_active * (N * K // 4 + N * K // 8)
    sparse_bytes_with_scale = sparse_bytes_no_scale + E_active * (N * K // 8)
    dense_bytes_with_scale = dense_bytes + E_active * (N * K // 8)

    print(f"  Data per projection (10 experts):")
    print(f"    Dense (no scale): {dense_bytes // 1024} KB")
    print(f"    Dense + scale:    {dense_bytes_with_scale // 1024} KB")
    print(f"    Sparse + scale:   {sparse_bytes_with_scale // 1024} KB ({sparse_bytes_with_scale * 100 // dense_bytes_with_scale}%)")

    # === MoE Marlin ===
    moe_c = float('inf')
    if HAS_MARLIN:
        pack_factor = 8
        b_q_weight = torch.randint(0, 2**31, (E_total, K // pack_factor, N),
                                   device='cuda', dtype=torch.int32)
        perm = torch.empty(E_total, 0, device='cuda', dtype=torch.int32)

        try:
            b_marlin = gptq_marlin_moe_repack(b_q_weight, perm, K, N, 4)

            n_groups = (K + group_size - 1) // group_size
            scales_list = []
            for e in range(E_total):
                s = torch.ones(n_groups, N, device='cuda', dtype=torch.float16)
                sp = marlin_permute_scales(s, K, N, group_size)
                sp = nvfp4_marlin_process_scales(sp)
                scales_list.append(sp)
            b_scales = torch.stack(scales_list)

            g_scale = nvfp4_marlin_process_global_scale(
                torch.tensor(1.0, device='cuda', dtype=torch.float16))
            g_scales_m = g_scale.expand(E_total).contiguous()

            workspace = torch.zeros(E_total * N, device='cuda', dtype=torch.int32)

            M_total = 1
            topk = E_active
            A_input = torch.randn(M_total, K, device='cuda', dtype=torch.float16)
            topk_ids = torch.randperm(E_total, device='cuda')[:topk].unsqueeze(0).to(torch.int32)
            topk_weights_t = torch.ones(M_total, topk, device='cuda', dtype=torch.float32) / topk

            max_num_tokens_padded = E_total * block_size
            sorted_token_ids = torch.empty(max_num_tokens_padded, device='cuda', dtype=torch.int32)
            expert_ids_sorted = torch.empty(max_num_tokens_padded, device='cuda', dtype=torch.int32)
            num_tokens_past_padded = torch.empty(1, device='cuda', dtype=torch.int32)
            moe_align_block_size(topk_ids, E_total, block_size,
                                sorted_token_ids, expert_ids_sorted,
                                num_tokens_past_padded, None)
            padded_m = num_tokens_past_padded.item()
            perm_flat = torch.empty(0, device='cuda', dtype=torch.int32)

            A_padded = torch.zeros(padded_m, K, device='cuda', dtype=torch.float16)
            A_padded[:M_total] = A_input

            def run_moe():
                out = torch.zeros(padded_m * topk, N, device='cuda', dtype=torch.float16)
                return moe_wna16_marlin_gemm(
                    A_padded, out, b_marlin, None, b_scales, None, g_scales_m,
                    None, torch.empty(0, device='cuda', dtype=torch.int32), perm_flat,
                    workspace, sorted_token_ids, expert_ids_sorted, num_tokens_past_padded,
                    topk_weights_t, block_size, topk, True,
                    fp4_type, padded_m, N, K,
                    True, True, False, False)

            run_moe()  # warmup
            moe_c = bench_cold(run_moe, fb)
            print(f"  MoE Marlin: Cold={moe_c:.1f}us  ({dense_bytes_with_scale/(moe_c*1e-6)/1e9:.0f} GB/s)")

        except Exception as e:
            print(f"  MoE Marlin FAILED: {e}")
            traceback.print_exc()

        del b_q_weight
        torch.cuda.empty_cache()

    # === v7 with scales ===
    if HAS_V7:
        eid = torch.randperm(E_total, device='cuda')[:E_active].to(torch.int32).sort().values
        A = torch.randn(E_active, K, device='cuda', dtype=torch.float16)
        Bd = torch.randint(0, 256, (E_total, K // 2, N), device='cuda', dtype=torch.uint8)
        Bc = torch.randint(0, 256, (E_total, K // 4, N), device='cuda', dtype=torch.uint8)
        Mp = torch.randint(0, 256, (E_total, K // 8, N), device='cuda', dtype=torch.uint8)
        Sc = torch.randint(56, 120, (E_total, K // 8, N), device='cuda', dtype=torch.uint8)
        Gs = torch.rand(E_total, device='cuda', dtype=torch.float32) * 0.001 + 0.0001

        d7c = bench_cold(lambda: sp7.batched_dense_v7(A, Bd, Sc, Gs, eid), fb)
        s7c = bench_cold(lambda: sp7.batched_sparse_v7(A, Bc, Mp, Sc, Gs, eid), fb)

        print(f"  v7 Dense (scaled):  Cold={d7c:.1f}us  ({dense_bytes_with_scale/(d7c*1e-6)/1e9:.0f} GB/s)")
        print(f"  v7 Sparse (scaled): Cold={s7c:.1f}us  ({sparse_bytes_with_scale/(s7c*1e-6)/1e9:.0f} GB/s)  sp/d={s7c/d7c:.3f}")

        if moe_c < float('inf'):
            print(f"  --- DIRECT COMPARISON (cold DRAM, 10 experts) ---")
            print(f"  MoE Marlin:        {moe_c:.1f}us")
            print(f"  v7 dense (scaled): {d7c:.1f}us  ({moe_c/d7c:.2f}x faster)")
            print(f"  v7 sparse (scaled):{s7c:.1f}us  ({moe_c/s7c:.2f}x faster)")

        del A, Bd, Bc, Mp, Sc, Gs
    torch.cuda.empty_cache()

    # === v7 packed (interleaved comp+meta+scale) ===
    if HAS_V7:
        eid = torch.randperm(E_total, device='cuda')[:E_active].to(torch.int32).sort().values
        A = torch.randn(E_active, K, device='cuda', dtype=torch.float16)
        Bc = torch.randint(0, 256, (E_total, K // 4, N), device='cuda', dtype=torch.uint8)
        Mp = torch.randint(0, 256, (E_total, K // 8, N), device='cuda', dtype=torch.uint8)
        Sc = torch.randint(56, 120, (E_total, K // 8, N), device='cuda', dtype=torch.uint8)
        Gs = torch.rand(E_total, device='cuda', dtype=torch.float32) * 0.001 + 0.0001

        # Build packed tensor: [E_total, K/8, 4*N]
        n_pairs = K // 8
        Bp = torch.empty(E_total, n_pairs, 4 * N, device='cuda', dtype=torch.uint8)
        for pi in range(n_pairs):
            Bp[:, pi, :N] = Bc[:, 2 * pi, :]       # comp_group0
            Bp[:, pi, N:2*N] = Bc[:, 2 * pi + 1, :] # comp_group1
            Bp[:, pi, 2*N:3*N] = Mp[:, pi, :]       # meta
            Bp[:, pi, 3*N:4*N] = Sc[:, pi, :]       # scale

        s7pc = bench_cold(lambda: sp7.batched_sparse_packed_v7(A, Bp, Gs, eid, K), fb)
        print(f"  v7 Sparse PACKED:   Cold={s7pc:.1f}us  ({sparse_bytes_with_scale/(s7pc*1e-6)/1e9:.0f} GB/s)")
        if moe_c < float('inf'):
            print(f"    vs Marlin: {moe_c/s7pc:.2f}x faster")
        del A, Bc, Mp, Sc, Gs, Bp
    torch.cuda.empty_cache()

    # === v6 without scales (for regression check) ===
    if HAS_V6:
        eid = torch.randperm(E_total, device='cuda')[:E_active].to(torch.int32).sort().values
        A = torch.randn(E_active, K, device='cuda', dtype=torch.float16)
        Bc = torch.randint(0, 256, (E_total, K // 4, N), device='cuda', dtype=torch.uint8)
        Mp = torch.randint(0, 256, (E_total, K // 8, N), device='cuda', dtype=torch.uint8)

        s6c = bench_cold(lambda: sp6.batched_sparse_v6(A, Bc, Mp, eid), fb)
        print(f"  v6 Sparse (no scale): Cold={s6c:.1f}us  (reference)")
        del A, Bc, Mp
    torch.cuda.empty_cache()


# End-to-end estimates
print(f"\n{'='*80}")
print("END-TO-END ESTIMATES (48 layers × 2 projections)")
print("=" * 80)
