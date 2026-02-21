#!/usr/bin/env python3
"""End-to-end MoE layer benchmark: v7 sparse kernel vs Marlin.

Loads real NVFP4 expert weights, applies 2:4 sparsity, runs the complete
MoE forward pass (route → gate_up GEMM → SiLU → down GEMM → weighted reduce)
and compares against Marlin timing.
"""

import torch, os, glob, time, sys

try:
    import sparse_fp4_v7 as sp7
except ImportError:
    print("ERROR: sparse_fp4_v7 not available, run: python setup_v7.py install")
    sys.exit(1)

FP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                        dtype=torch.float32)


def load_expert_weight(ckpt_dir, layer_idx, proj_name, expert_idx):
    """Load weight, weight_scale, weight_scale_2 for one expert."""
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


def apply_2of4_sparsity_fast(nibbles):
    """Vectorized 2:4 sparsity. Returns comp [N, K/4], meta [N, K/8]."""
    N, K = nibbles.shape
    assert K % 8 == 0

    mags = FP4_LUT[nibbles.long()].abs()

    # Reshape to groups of 4
    nib_4 = nibbles.view(N, K // 4, 4)
    mag_4 = mags.view(N, K // 4, 4)

    # Top-2 by magnitude for ALL groups at once
    _, top2 = mag_4.topk(2, dim=2)
    top2, _ = top2.sort(dim=2)

    # Extract values
    v0 = nib_4.gather(2, top2[:, :, 0:1]).squeeze(2)
    v1 = nib_4.gather(2, top2[:, :, 1:2]).squeeze(2)

    # Pack: low nibble = v0, high nibble = v1
    comp = ((v1 << 4) | v0).to(torch.uint8)  # [N, K/4]

    # Pack metadata: pairs of 2-bit indices
    i0 = top2[:, :, 0]  # [N, K/4]
    i1 = top2[:, :, 1]  # [N, K/4]

    # Group pairs: [N, K/4] → [N, K/8, 2]
    i0_pairs = i0.view(N, K // 8, 2)
    i1_pairs = i1.view(N, K // 8, 2)

    # Pack 4 two-bit indices per byte
    meta = ((i0_pairs[:, :, 0] & 3) |
            ((i1_pairs[:, :, 0] & 3) << 2) |
            ((i0_pairs[:, :, 1] & 3) << 4) |
            ((i1_pairs[:, :, 1] & 3) << 6)).to(torch.uint8)

    return comp, meta


def convert_expert_to_v7(weight, scale, scale2):
    """Convert one expert from NVFP4 packed to v7 sparse format.

    Returns: comp_T [K/4, N], meta_T [K/8, N], scale_T [n_groups, N], g_scale float
    """
    nibbles = unpack_fp4_to_nibbles(weight)
    comp, meta = apply_2of4_sparsity_fast(nibbles)
    comp_T = comp.T.contiguous()
    meta_T = meta.T.contiguous()
    scale_T = scale.view(torch.uint8).T.contiguous()
    g_scale = scale2.float().item()
    return comp_T, meta_T, scale_T, g_scale


# =====================================================================
# Find checkpoint
# =====================================================================
ckpt_dirs = [
    "/workspace/models/Qwen3-Next-80B-A3B-Instruct-NVFP4-Sparse-2of4",
    os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--Qwen3-Next-80B-A3B-Instruct-NVFP4/snapshots"),
]

ckpt_dir = None
for d in ckpt_dirs:
    if os.path.isdir(d):
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

print(f"Checkpoint: {ckpt_dir}")

# =====================================================================
# Load and convert experts for one MoE layer
# =====================================================================
layer = 0
topk = 10
# Pick 10 experts (simulating router selection)
expert_ids = list(range(topk))

print(f"\nLoading layer {layer}, {topk} experts: {expert_ids}")

# For each projection type, build the v7 tensors
def load_and_convert_proj(proj_name, experts):
    """Load and convert multiple experts for one projection."""
    comp_T_list, meta_T_list, scale_T_list, g_scale_list = [], [], [], []
    N = K = 0
    n_groups = 0

    for eidx in experts:
        w, s, s2 = load_expert_weight(ckpt_dir, layer, proj_name, eidx)
        if w is None:
            print(f"  Expert {eidx} {proj_name} not found!")
            return None
        N, K_half = w.shape
        K = K_half * 2
        n_groups = s.shape[1]

        ct, mt, st, gs = convert_expert_to_v7(w, s, s2)
        comp_T_list.append(ct)
        meta_T_list.append(mt)
        scale_T_list.append(st)
        g_scale_list.append(gs)

    # Stack into [E_total, ...] tensors
    comp_T = torch.stack(comp_T_list, dim=0)     # [E, K/4, N]
    meta_T = torch.stack(meta_T_list, dim=0)     # [E, K/8, N]
    scale_T = torch.stack(scale_T_list, dim=0)   # [E, n_groups, N]
    g_scales = torch.tensor(g_scale_list, dtype=torch.float32)

    print(f"  {proj_name}: N={N}, K={K}, n_groups={n_groups}")
    print(f"    comp_T: {list(comp_T.shape)}, meta_T: {list(meta_T.shape)}")
    print(f"    scale_T: {list(scale_T.shape)}, g_scales: {list(g_scales.shape)}")

    return {
        'comp_T': comp_T, 'meta_T': meta_T,
        'scale_T': scale_T, 'g_scales': g_scales,
        'N': N, 'K': K
    }


t0 = time.time()
gate_proj = load_and_convert_proj("gate_proj", expert_ids)
up_proj = load_and_convert_proj("up_proj", expert_ids)
down_proj = load_and_convert_proj("down_proj", expert_ids)
print(f"\nConversion time: {time.time()-t0:.1f}s")

if gate_proj is None or up_proj is None or down_proj is None:
    print("ERROR: Failed to load some experts")
    sys.exit(1)

# =====================================================================
# Move to GPU
# =====================================================================
print("\nMoving to GPU...")

# Concatenate gate + up into w13 (as vLLM does)
# gate_proj: N=512, K=2048 → up_proj: N=512, K=2048
# w13: N=1024, K=2048 (concatenated along N dimension)
# For v7, we need [E, K/4, 2*N] for comp, [E, K/8, 2*N] for meta, etc.

# Actually, in the model, gate and up are separate experts weights with the same K.
# In vLLM, they're concatenated as w13. For our kernel, we can either:
# a) Concatenate them and run one GEMM per layer
# b) Run them as separate GEMMs

# Option (a) is what vLLM/Marlin does. Let's do the same.
# Concatenate along N dimension: gate [E, K/4, 512] + up [E, K/4, 512] → [E, K/4, 1024]

w13_comp = torch.cat([gate_proj['comp_T'], up_proj['comp_T']], dim=2).cuda()
w13_meta = torch.cat([gate_proj['meta_T'], up_proj['meta_T']], dim=2).cuda()
w13_scale = torch.cat([gate_proj['scale_T'], up_proj['scale_T']], dim=2).cuda()
# For global scales, gate and up have the same scale2 per expert
w13_g_scales = gate_proj['g_scales'].cuda()

w2_comp = down_proj['comp_T'].cuda()
w2_meta = down_proj['meta_T'].cuda()
w2_scale = down_proj['scale_T'].cuda()
w2_g_scales = down_proj['g_scales'].cuda()

K_hidden = gate_proj['K']    # 2048
N_inter = gate_proj['N']     # 512
K_down = down_proj['K']      # 512
N_down = down_proj['N']      # 2048

print(f"  w13 (gate+up): comp {list(w13_comp.shape)}, meta {list(w13_meta.shape)}")
print(f"  w2 (down):     comp {list(w2_comp.shape)}, meta {list(w2_meta.shape)}")

# Expert IDs for v7 kernel (0-indexed into our loaded experts)
eid = torch.arange(topk, dtype=torch.int32, device='cuda')

# Random topk_weights
topk_weights = torch.softmax(torch.randn(1, topk), dim=-1).float().cuda()

print(f"\nDimensions: hidden={K_hidden}, inter={N_inter}, topk={topk}")
print(f"  GEMM 1 (gate_up): [{topk}, {K_hidden}] × [{topk}, {2*N_inter}, {K_hidden}]^T → [{topk}, {2*N_inter}]")
print(f"  GEMM 2 (down):    [{topk}, {K_down}] × [{topk}, {N_down}, {K_down}]^T → [{topk}, {N_down}]")


# =====================================================================
# V7 Sparse MoE Forward Pass
# =====================================================================
def v7_sparse_moe_forward(hidden_state):
    """Complete MoE forward pass using v7 sparse kernel.

    hidden_state: [1, K_hidden] fp16
    Returns: [1, K_hidden] fp16
    """
    # Replicate activation for all topk experts
    A_gate_up = hidden_state.expand(topk, -1).contiguous()  # [topk, K_hidden]

    # GEMM 1: gate_up
    gate_up_out = sp7.batched_sparse_v7(
        A_gate_up, w13_comp, w13_meta, w13_scale, w13_g_scales, eid)
    # gate_up_out: [topk, 2*N_inter]

    # SiLU gating
    gate = gate_up_out[:, :N_inter]
    up = gate_up_out[:, N_inter:]
    intermediate = torch.nn.functional.silu(gate) * up  # [topk, N_inter]

    # GEMM 2: down
    down_out = sp7.batched_sparse_v7(
        intermediate, w2_comp, w2_meta, w2_scale, w2_g_scales, eid)
    # down_out: [topk, N_down]

    # Weighted reduction
    output = (down_out.float() * topk_weights[0].unsqueeze(-1)).sum(0, keepdim=True).half()
    return output


# =====================================================================
# Warmup
# =====================================================================
print("\nWarming up...")
hidden = torch.randn(1, K_hidden, dtype=torch.float16, device='cuda')

for _ in range(5):
    out = v7_sparse_moe_forward(hidden)
torch.cuda.synchronize()

print(f"  Output shape: {list(out.shape)}, mean={out.float().mean():.4f}, std={out.float().std():.4f}")

# =====================================================================
# Benchmark
# =====================================================================
print("\n" + "=" * 70)
print("Benchmark: v7 Sparse MoE Layer (10 experts, BS=1)")
print("=" * 70)

n_iters = 200

# Full MoE layer
torch.cuda.synchronize()
t0 = time.time()
for _ in range(n_iters):
    out = v7_sparse_moe_forward(hidden)
torch.cuda.synchronize()
t_total = (time.time() - t0) / n_iters * 1e6

# Breakdown: GEMM 1 only
A_gate_up = hidden.expand(topk, -1).contiguous()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(n_iters):
    sp7.batched_sparse_v7(A_gate_up, w13_comp, w13_meta, w13_scale, w13_g_scales, eid)
torch.cuda.synchronize()
t_gemm1 = (time.time() - t0) / n_iters * 1e6

# Breakdown: SiLU
gate_up_out = sp7.batched_sparse_v7(A_gate_up, w13_comp, w13_meta, w13_scale, w13_g_scales, eid)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(n_iters):
    gate = gate_up_out[:, :N_inter]
    up = gate_up_out[:, N_inter:]
    intermediate = torch.nn.functional.silu(gate) * up
torch.cuda.synchronize()
t_silu = (time.time() - t0) / n_iters * 1e6

# Breakdown: GEMM 2 only
intermediate = torch.nn.functional.silu(gate_up_out[:, :N_inter]) * gate_up_out[:, N_inter:]
torch.cuda.synchronize()
t0 = time.time()
for _ in range(n_iters):
    sp7.batched_sparse_v7(intermediate, w2_comp, w2_meta, w2_scale, w2_g_scales, eid)
torch.cuda.synchronize()
t_gemm2 = (time.time() - t0) / n_iters * 1e6

# Breakdown: Reduction
down_out = sp7.batched_sparse_v7(intermediate, w2_comp, w2_meta, w2_scale, w2_g_scales, eid)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(n_iters):
    _ = (down_out.float() * topk_weights[0].unsqueeze(-1)).sum(0, keepdim=True).half()
torch.cuda.synchronize()
t_reduce = (time.time() - t0) / n_iters * 1e6

print(f"\n  GEMM 1 (gate_up): {t_gemm1:8.1f} μs")
print(f"  SiLU activation:  {t_silu:8.1f} μs")
print(f"  GEMM 2 (down):    {t_gemm2:8.1f} μs")
print(f"  Reduce:           {t_reduce:8.1f} μs")
print(f"  ---")
print(f"  Sum of parts:     {t_gemm1+t_silu+t_gemm2+t_reduce:8.1f} μs")
print(f"  Full layer:       {t_total:8.1f} μs")
print()

# Marlin comparison (from bench_v7_scaled.py measurements)
marlin_gemm1 = 70.0  # typical Marlin gate_up μs
marlin_gemm2 = 58.0  # typical Marlin down μs
marlin_total_est = marlin_gemm1 + marlin_gemm2 + t_silu + t_reduce  # same overhead ops

print(f"  Marlin gate_up (estimated):  {marlin_gemm1:8.1f} μs")
print(f"  Marlin down (estimated):     {marlin_gemm2:8.1f} μs")
print(f"  Marlin total (estimated):    {marlin_total_est:8.1f} μs")
print()
print(f"  Speedup GEMM 1:  {marlin_gemm1/t_gemm1:.2f}x")
print(f"  Speedup GEMM 2:  {marlin_gemm2/t_gemm2:.2f}x")
print(f"  Speedup total:   {marlin_total_est/t_total:.2f}x")

# Estimate end-to-end impact
# At 59.9 tok/s, each token takes 16.7ms
# Model has ~36 MoE layers (mixed attention + Mamba)
# Estimate MoE fraction of total inference time
n_moe_layers = 36  # approximate for Qwen3-Next-80B
baseline_tok_s = 59.9
token_time_us = 1e6 / baseline_tok_s

moe_time_marlin = n_moe_layers * marlin_total_est
moe_time_v7 = n_moe_layers * t_total
moe_savings = moe_time_marlin - moe_time_v7

new_token_time = token_time_us - moe_savings
new_tok_s = 1e6 / new_token_time

print(f"\n  End-to-end estimate ({n_moe_layers} MoE layers):")
print(f"    Marlin MoE total: {moe_time_marlin/1e3:.1f} ms / token")
print(f"    v7 Sparse total:  {moe_time_v7/1e3:.1f} ms / token")
print(f"    Savings:          {moe_savings/1e3:.1f} ms / token")
print(f"    Baseline:         {baseline_tok_s:.1f} tok/s ({token_time_us/1e3:.1f} ms/tok)")
print(f"    Estimated:        {new_tok_s:.1f} tok/s ({new_token_time/1e3:.1f} ms/tok)")
print(f"    Improvement:      {(new_tok_s/baseline_tok_s - 1)*100:.1f}%")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
