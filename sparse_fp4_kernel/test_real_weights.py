#!/usr/bin/env python3
"""Load real NVFP4 expert weights, convert to v7 sparse format, validate.

Tests the full pipeline:
1. Load packed NVFP4 weights + FP8 scales from checkpoint
2. Apply 2:4 sparsity to get compressed + metadata
3. Run v7 sparse kernel
4. Compare output against dense v7 kernel (same weights, no sparsity)
"""
import torch, os, glob, time, sys

try:
    import sparse_fp4_v7 as sp7
    HAS_V7 = True
except ImportError:
    HAS_V7 = False
    print("ERROR: v7 not available")
    sys.exit(1)

# FP4 E2M1 LUT (same as kernel)
FP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                        dtype=torch.float32)

def load_expert_weight(ckpt_dir, layer_idx, proj_name, expert_idx):
    """Load weight, weight_scale, weight_scale_2 for one expert from safetensors."""
    from safetensors import safe_open

    # Find the right shard
    shard_files = sorted(glob.glob(os.path.join(ckpt_dir, "model-*.safetensors")))
    prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}"
    w_key = f"{prefix}.weight"
    s_key = f"{prefix}.weight_scale"
    s2_key = f"{prefix}.weight_scale_2"

    weight = scale = scale2 = None
    for sf in shard_files:
        with safe_open(sf, framework="pt", device="cpu") as f:
            keys = f.keys()
            if w_key in keys:
                weight = f.get_tensor(w_key)
            if s_key in keys:
                scale = f.get_tensor(s_key)
            if s2_key in keys:
                scale2 = f.get_tensor(s2_key)
        if weight is not None and scale is not None and scale2 is not None:
            break

    return weight, scale, scale2


def unpack_fp4_to_nibbles(packed_uint8):
    """Unpack [N, K/2] uint8 to [N, K] nibbles (0-15)."""
    N, K_half = packed_uint8.shape
    low = packed_uint8 & 0x0F
    high = (packed_uint8 >> 4) & 0x0F
    nibbles = torch.zeros(N, K_half * 2, dtype=torch.uint8)
    nibbles[:, 0::2] = low
    nibbles[:, 1::2] = high
    return nibbles


def apply_2of4_sparsity(nibbles):
    """Apply 2:4 structured sparsity. Returns compressed, metadata.

    nibbles: [N, K] uint8 (0-15)
    Returns:
        comp: [N, K/2] uint8 (2 non-zero nibbles per byte, from each group of 4)
        meta: [N, K/8] uint8 (2-bit indices for each non-zero, packed 4 per byte)
    """
    N, K = nibbles.shape
    assert K % 4 == 0

    # Get magnitudes from FP4 LUT
    mags = FP4_LUT[nibbles.long()].abs()

    comp_list = []
    meta_list = []

    for g in range(0, K, 4):
        group = nibbles[:, g:g+4]       # [N, 4]
        group_mags = mags[:, g:g+4]     # [N, 4]

        # Keep top-2 by magnitude per row
        _, top2_idx = group_mags.topk(2, dim=1)  # [N, 2]
        top2_idx_sorted, _ = top2_idx.sort(dim=1)

        # Extract values
        v0 = group.gather(1, top2_idx_sorted[:, 0:1]).squeeze(1)  # [N]
        v1 = group.gather(1, top2_idx_sorted[:, 1:2]).squeeze(1)  # [N]

        # Packed: low nibble = v0, high nibble = v1
        packed = (v1 << 4) | v0  # [N] uint8
        comp_list.append(packed)

        # 2-bit indices
        i0 = top2_idx_sorted[:, 0]  # [N]
        i1 = top2_idx_sorted[:, 1]  # [N]
        meta_list.append((i0, i1))

    # Assemble compressed: [N, K/4] uint8
    comp = torch.stack(comp_list, dim=1).to(torch.uint8)  # [N, K/4]

    # Assemble metadata: [N, K/8] uint8 (4 two-bit indices per byte)
    # Each byte covers 2 groups (= 8 K-elements): i0_g0, i1_g0, i0_g1, i1_g1
    meta_bytes = []
    for p in range(0, len(meta_list), 2):
        i0_g0, i1_g0 = meta_list[p]
        i0_g1, i1_g1 = meta_list[p + 1]
        byte = (i0_g0 & 3) | ((i1_g0 & 3) << 2) | ((i0_g1 & 3) << 4) | ((i1_g1 & 3) << 6)
        meta_bytes.append(byte.to(torch.uint8))

    meta = torch.stack(meta_bytes, dim=1)  # [N, K/8]
    return comp, meta


def dequant_dense_ref(packed_uint8, scale, scale2, A_fp16):
    """Reference dense dequant: FP4 weight × FP8 scale × global_scale × activation."""
    N, K_half = packed_uint8.shape
    K = K_half * 2

    nibbles = unpack_fp4_to_nibbles(packed_uint8)  # [N, K]
    values = FP4_LUT[nibbles.long()]  # [N, K] float32

    # FP8 scale: [N, K/group_size] → broadcast to [N, K]
    scale_f32 = scale.float()  # [N, n_groups]
    n_groups = scale_f32.shape[1]
    group_size = K // n_groups
    scale_expanded = scale_f32.repeat_interleave(group_size, dim=1)[:, :K]  # [N, K]

    # Global scale
    gs = scale2.float().item()

    # Dequantized weight: [N, K]
    W = values * scale_expanded * gs

    # GEMV: A [1, K] × W^T [K, N] → [1, N]
    return (A_fp16.float() @ W.T).half()


# =====================================================================
# Main test
# =====================================================================

# Try both dense and sparse checkpoint dirs
ckpt_dirs = [
    "/workspace/models/Qwen3-Next-80B-A3B-Instruct-NVFP4-Sparse-2of4",
    os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--Qwen3-Next-80B-A3B-Instruct-NVFP4/snapshots"),
]

ckpt_dir = None
for d in ckpt_dirs:
    if os.path.isdir(d):
        # For HF cache, find the actual snapshot dir
        if "snapshots" in d:
            snaps = glob.glob(os.path.join(d, "*"))
            if snaps:
                ckpt_dir = snaps[0]
        else:
            ckpt_dir = d
        break

if ckpt_dir is None:
    print("ERROR: No checkpoint found")
    sys.exit(1)

print(f"Using checkpoint: {ckpt_dir}")

# Test with layer 0, expert 0, gate_proj
layer = 0
expert = 0
projections = ["gate_proj", "up_proj", "down_proj"]

print("=" * 80)
print("Real Weight Validation: v7 sparse kernel vs reference dequant")
print("=" * 80)

for proj in projections:
    print(f"\n--- Layer {layer}, Expert {expert}, {proj} ---")

    weight, scale, scale2 = load_expert_weight(ckpt_dir, layer, proj, expert)
    if weight is None:
        print(f"  Weight not found, skipping")
        continue

    N, K_half = weight.shape
    K = K_half * 2
    print(f"  Weight: [{N}, {K_half}] uint8 → N={N}, K={K}")
    print(f"  Scale: {list(scale.shape)} {scale.dtype}")
    print(f"  Scale2: {scale2.item():.6e}")

    # Random activation
    A = torch.randn(1, K, dtype=torch.float16)

    # Reference dense output
    C_ref = dequant_dense_ref(weight, scale, scale2, A)
    print(f"  Ref output: mean={C_ref.float().mean():.4f}, std={C_ref.float().std():.4f}")

    # Prepare tensors for v7 kernel
    # Dense weight: transpose to [K/2, N] for v7
    B_dense_T = weight.T.contiguous()  # [K/2, N]
    # Add batch dim for E_total=1
    B_dense_T_e = B_dense_T.unsqueeze(0).cuda()  # [1, K/2, N]

    # Scale: transpose to [K/8, N] for v7
    # scale is [N, K/8] in model format
    scales_T = scale.view(torch.uint8).T.contiguous()  # [K/8, N]
    scales_T_e = scales_T.unsqueeze(0).cuda()  # [1, K/8, N]

    # Global scale: [1] float32
    g_scales = torch.tensor([scale2.float().item()], dtype=torch.float32).cuda()

    # Expert ID and activation
    eid = torch.zeros(1, dtype=torch.int32, device='cuda')
    A_gpu = A.cuda()

    # v7 dense kernel
    C_v7d = sp7.batched_dense_v7(A_gpu, B_dense_T_e, scales_T_e, g_scales, eid)
    diff_d = (C_v7d.cpu().float() - C_ref.float()).abs()
    rdiff_d = diff_d.max() / (C_ref.float().abs().max() + 1e-8)
    print(f"  v7 Dense: max_diff={diff_d.max():.4f}, rel={rdiff_d:.6f} {'PASS' if rdiff_d < 0.02 else 'FAIL'}")

    # Apply 2:4 sparsity
    nibbles = unpack_fp4_to_nibbles(weight)  # [N, K]
    comp, meta = apply_2of4_sparsity(nibbles)  # [N, K/4], [N, K/8]
    print(f"  Sparse: comp [{list(comp.shape)}], meta [{list(meta.shape)}]")

    # Transpose for v7 kernel
    B_comp_T_e = comp.T.contiguous().unsqueeze(0).cuda()  # [1, K/4, N]
    Meta_T_pk_e = meta.T.contiguous().unsqueeze(0).cuda()  # [1, K/8, N]

    # Compute sparse reference directly from comp + meta tensors
    # This ensures we test kernel correctness, not sparsity selection consistency
    gs_val = scale2.float().item()
    scale_f32 = scale.float()  # [N, n_groups]
    n_groups = scale_f32.shape[1]
    group_sz = K // n_groups

    W_sparse = torch.zeros(N, K, dtype=torch.float32)
    for g in range(K // 4):  # compressed group index
        pair = g // 2
        within = g % 2
        for n in range(N):
            cb = comp[n, g].item()
            mb = meta[n, pair].item()
            if within == 0:
                i0 = mb & 3
                i1 = (mb >> 2) & 3
            else:
                i0 = (mb >> 4) & 3
                i1 = (mb >> 6) & 3
            k_base = g * 4
            s_idx = k_base // group_sz
            sv = scale_f32[n, s_idx].item() * gs_val
            W_sparse[n, k_base + i0] = FP4_LUT[cb & 0xF].item() * sv
            W_sparse[n, k_base + i1] = FP4_LUT[(cb >> 4) & 0xF].item() * sv
    C_sparse_ref = (A.float() @ W_sparse.T).half()

    # v7 sparse kernel
    C_v7s = sp7.batched_sparse_v7(A_gpu, B_comp_T_e, Meta_T_pk_e, scales_T_e, g_scales, eid)
    diff_s = (C_v7s.cpu().float() - C_sparse_ref.float()).abs()
    rdiff_s = diff_s.max() / (C_sparse_ref.float().abs().max() + 1e-8)
    print(f"  v7 Sparse vs comp_ref: max_diff={diff_s.max():.4f}, rel={rdiff_s:.6f} {'PASS' if rdiff_s < 0.02 else 'FAIL'}")

    # Cosine similarity of sparse vs dense (quality metric)
    cos = torch.nn.functional.cosine_similarity(
        C_v7s.cpu().float().flatten(), C_ref.float().flatten(), dim=0)
    print(f"  Cosine similarity (sparse vs dense): {cos:.6f}")

    # Packed format test skipped for now — needs group_size-aware packing
    # (scale groups != pair count when group_size > 8)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
