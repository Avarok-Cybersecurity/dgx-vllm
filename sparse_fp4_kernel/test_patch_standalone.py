#!/usr/bin/env python3
"""Standalone test of the v7 sparse MoE patch.

Tests the SparseV7Experts class with real weights in isolation,
without needing full vLLM model loading.
"""

import torch
import os
import glob
import sys
import time

try:
    import sparse_fp4_v7 as sp7
except ImportError:
    print("ERROR: sparse_fp4_v7 not installed")
    sys.exit(1)

from sparse_v7_moe_patch import (
    SparseV7Experts, convert_weight_batch_to_v7,
    unpack_fp4_nibbles, apply_2of4_sparsity_fast, FP4_LUT)

# =====================================================================
# Find checkpoint
# =====================================================================
ckpt_dirs = [
    os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--Qwen3-Next-80B-A3B-Instruct-NVFP4/snapshots"),
]
ckpt_dir = None
for d in ckpt_dirs:
    if os.path.isdir(d):
        snaps = glob.glob(os.path.join(d, "*"))
        if snaps:
            ckpt_dir = snaps[0]
            break

if ckpt_dir is None:
    print("ERROR: No checkpoint found")
    sys.exit(1)

print(f"Checkpoint: {ckpt_dir}")


def load_expert_weight(ckpt_dir, layer_idx, proj_name, expert_idx):
    from safetensors import safe_open
    shard_files = sorted(glob.glob(os.path.join(ckpt_dir, "model-*.safetensors")))
    prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}"
    w_key = f"{prefix}.weight"
    s_key = f"{prefix}.weight_scale"
    s2_key = f"{prefix}.weight_scale_2"

    weight = scale = scale2 = None
    for sf in shard_files:
        with safe_open(sf, framework="pt", device="cpu") as f:
            keys = f.keys()
            if w_key in keys: weight = f.get_tensor(w_key)
            if s_key in keys: scale = f.get_tensor(s_key)
            if s2_key in keys: scale2 = f.get_tensor(s2_key)
        if weight is not None and scale is not None and scale2 is not None:
            break
    return weight, scale, scale2


# =====================================================================
# Load experts and build v7 tensors (simulating vLLM weight loading)
# =====================================================================
layer = 0
n_experts = 10  # First 10 experts
topk = 10

print(f"\nLoading {n_experts} experts from layer {layer}...")
t0 = time.time()

gate_weights, gate_scales, gate_gs = [], [], []
up_weights, up_scales, up_gs = [], [], []
down_weights, down_scales, down_gs = [], [], []

for e in range(n_experts):
    gw, gs, gg = load_expert_weight(ckpt_dir, layer, "gate_proj", e)
    uw, us, ug = load_expert_weight(ckpt_dir, layer, "up_proj", e)
    dw, ds, dg = load_expert_weight(ckpt_dir, layer, "down_proj", e)

    gate_weights.append(gw)
    gate_scales.append(gs)
    gate_gs.append(gg.float().item())
    up_weights.append(uw)
    up_scales.append(us)
    up_gs.append(ug.float().item())
    down_weights.append(dw)
    down_scales.append(ds)
    down_gs.append(dg.float().item())

# Stack into [E, N, K/2] format (as vLLM does)
gate_w = torch.stack(gate_weights)   # [E, 512, 1024]
gate_s = torch.stack(gate_scales)    # [E, 512, 128]
up_w = torch.stack(up_weights)       # [E, 512, 1024]
up_s = torch.stack(up_scales)        # [E, 512, 128]
down_w = torch.stack(down_weights)   # [E, 2048, 256]
down_s = torch.stack(down_scales)    # [E, 2048, 32]

# Concatenate gate+up → w13 (as vLLM does)
w13_weight = torch.cat([gate_w, up_w], dim=1)  # [E, 1024, 1024]
w13_scale = torch.cat([gate_s, up_s], dim=1)    # [E, 1024, 128]
w13_gs = torch.tensor(gate_gs, dtype=torch.float32)  # [E]
# (Using gate global scale for both, as they should be same per vLLM)

w2_weight = down_w   # [E, 2048, 256]
w2_scale = down_s    # [E, 2048, 32]
w2_gs = torch.tensor(down_gs, dtype=torch.float32)

print(f"  Load time: {time.time()-t0:.1f}s")
print(f"  w13: weight {list(w13_weight.shape)}, scale {list(w13_scale.shape)}")
print(f"  w2:  weight {list(w2_weight.shape)}, scale {list(w2_scale.shape)}")

# =====================================================================
# Convert to v7 sparse format (simulating process_weights_after_loading)
# =====================================================================
print("\nConverting to v7 sparse format on GPU...")
t0 = time.time()

w13_comp, w13_meta, w13_sc, w13_g = convert_weight_batch_to_v7(
    w13_weight, w13_scale, w13_gs, device='cuda', batch_size=4)
w2_comp, w2_meta, w2_sc, w2_g = convert_weight_batch_to_v7(
    w2_weight, w2_scale, w2_gs, device='cuda', batch_size=4)

print(f"  Conversion time: {time.time()-t0:.1f}s")
print(f"  w13 comp: {list(w13_comp.shape)}, meta: {list(w13_meta.shape)}")
print(f"  w2  comp: {list(w2_comp.shape)}, meta: {list(w2_meta.shape)}")

# =====================================================================
# Create SparseV7Experts instance
# =====================================================================
inter_size = 512
hidden_size = 2048

v7 = SparseV7Experts(
    w13_comp, w13_meta, w13_sc, w13_g,
    w2_comp, w2_meta, w2_sc, w2_g,
    inter_size, hidden_size)

# =====================================================================
# Test: BS=1 (decode)
# =====================================================================
print("\n" + "=" * 70)
print("Test 1: BS=1 (decode)")
print("=" * 70)

M = 1
hidden = torch.randn(M, hidden_size, dtype=torch.float16, device='cuda')
topk_ids = torch.arange(topk, device='cuda').unsqueeze(0).long()  # [1, 10]
topk_weights_t = torch.softmax(torch.randn(M, topk, device='cuda'), dim=-1).float()
output = torch.zeros(M, hidden_size, dtype=torch.float16, device='cuda')

# Warmup
for _ in range(5):
    v7.apply(output, hidden, None, None, topk_weights_t, topk_ids,
             'silu', n_experts, None, None, None,
             torch.empty(0), torch.empty(0), None, False)
torch.cuda.synchronize()

# Benchmark
n_iters = 200
torch.cuda.synchronize()
t0 = time.time()
for _ in range(n_iters):
    v7.apply(output, hidden, None, None, topk_weights_t, topk_ids,
             'silu', n_experts, None, None, None,
             torch.empty(0), torch.empty(0), None, False)
torch.cuda.synchronize()
t_bs1 = (time.time() - t0) / n_iters * 1e6

print(f"  Output: mean={output.float().mean():.4f}, std={output.float().std():.4f}")
print(f"  Time: {t_bs1:.1f} μs per MoE layer")

# =====================================================================
# Test: BS=4 (small batch prefill)
# =====================================================================
print("\n" + "=" * 70)
print("Test 2: BS=4 (small batch)")
print("=" * 70)

M = 4
hidden4 = torch.randn(M, hidden_size, dtype=torch.float16, device='cuda')
# Each token routes to 10 experts (may overlap)
topk_ids4 = torch.randint(0, n_experts, (M, topk), device='cuda').long()
topk_weights4 = torch.softmax(torch.randn(M, topk, device='cuda'), dim=-1).float()
output4 = torch.zeros(M, hidden_size, dtype=torch.float16, device='cuda')

# Warmup
for _ in range(3):
    v7.apply(output4, hidden4, None, None, topk_weights4, topk_ids4,
             'silu', n_experts, None, None, None,
             torch.empty(0), torch.empty(0), None, False)
torch.cuda.synchronize()

# Benchmark
torch.cuda.synchronize()
t0 = time.time()
for _ in range(n_iters):
    v7.apply(output4, hidden4, None, None, topk_weights4, topk_ids4,
             'silu', n_experts, None, None, None,
             torch.empty(0), torch.empty(0), None, False)
torch.cuda.synchronize()
t_bs4 = (time.time() - t0) / n_iters * 1e6

print(f"  Output: mean={output4.float().mean():.4f}, std={output4.float().std():.4f}")
print(f"  Time: {t_bs4:.1f} μs per MoE layer")

# =====================================================================
# Correctness: compare v7 sparse output vs dense reference
# =====================================================================
print("\n" + "=" * 70)
print("Correctness: v7 sparse vs dense reference (BS=1)")
print("=" * 70)

# Dense reference using v7 dense kernel
# First, create dense weight tensors (before sparsity)
def compute_dense_ref(hidden_state, expert_idx, proj_name):
    """Compute dense FP4 dequant reference for one expert."""
    w, s, s2 = load_expert_weight(ckpt_dir, layer, proj_name, expert_idx)
    N, K_half = w.shape
    K = K_half * 2
    nibs = unpack_fp4_nibbles(w)  # [N, K]
    vals = FP4_LUT[nibs.long()]   # [N, K] float

    scale_f32 = s.float()
    n_groups = scale_f32.shape[1]
    group_size = K // n_groups
    scale_exp = scale_f32.repeat_interleave(group_size, dim=1)[:, :K]
    gs = s2.float().item()

    W = vals * scale_exp * gs
    return (hidden_state.float().cpu() @ W.T).half()

h_cpu = hidden[0].cpu()
dense_outs = []
for e in range(topk):
    gate_out = compute_dense_ref(h_cpu, e, "gate_proj")  # [1, 512]
    up_out = compute_dense_ref(h_cpu, e, "up_proj")        # [1, 512]
    inter = torch.nn.functional.silu(gate_out) * up_out
    down_out = compute_dense_ref(inter, e, "down_proj")    # [1, 2048]
    dense_outs.append(down_out)

dense_stack = torch.stack([d.squeeze(0) for d in dense_outs])  # [10, 2048]
weights_cpu = topk_weights_t[0].cpu().unsqueeze(-1)
dense_ref = (dense_stack.float() * weights_cpu).sum(0)  # [2048]

sparse_out = output[0].cpu().float()

diff = (sparse_out - dense_ref).abs()
rdiff = diff.max() / (dense_ref.abs().max() + 1e-8)
cos = torch.nn.functional.cosine_similarity(
    sparse_out.flatten(), dense_ref.flatten(), dim=0)

print(f"  Max abs diff: {diff.max():.6f}")
print(f"  Max rel diff: {rdiff:.6f}")
print(f"  Cosine sim:   {cos:.6f}")
print(f"  {'PASS' if cos > 0.90 else 'FAIL'} (cosine > 0.90 for 2:4 sparsity)")

# =====================================================================
# Summary
# =====================================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

marlin_est = 155.0  # μs per MoE layer (from bench measurements)
n_moe_layers = 48
baseline_tok_s = 59.9

moe_marlin = n_moe_layers * marlin_est
moe_v7 = n_moe_layers * t_bs1
savings = moe_marlin - moe_v7
token_time = 1e6 / baseline_tok_s
new_time = token_time - savings
new_tok_s = 1e6 / new_time

print(f"  BS=1 MoE layer: {t_bs1:.1f} μs (Marlin est: {marlin_est:.1f} μs)")
print(f"  BS=4 MoE layer: {t_bs4:.1f} μs")
print(f"  Speedup (BS=1): {marlin_est/t_bs1:.2f}x")
print(f"  E2E estimate:   {new_tok_s:.1f} tok/s (+{(new_tok_s/baseline_tok_s-1)*100:.1f}%)")
print()
