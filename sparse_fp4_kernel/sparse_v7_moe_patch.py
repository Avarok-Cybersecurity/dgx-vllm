#!/usr/bin/env python3
"""Monkey-patch vLLM to use v7 sparse FP4 kernel for MoE layers.

This replaces the Marlin MoE GEMM with our v7 sparse FP4 GEMV kernel,
achieving ~2x speedup per MoE layer at BS=1 through:
1. 2:4 structured sparsity (25% less data to read)
2. Zero padding waste (Marlin pads each expert to 8 rows)

Strategy: Let the original Marlin process_weights_after_loading run fully
(building MarlinExperts with all required interface methods), then replace
only the apply() method with our v7 sparse kernel via an instance-level
closure. This preserves moe_problem_size, quant_config, etc.

Usage: Import this module before model loading.
  import sparse_v7_moe_patch  # applies patches automatically

Or explicitly:
  from sparse_v7_moe_patch import patch_vllm
  patch_vllm()
"""

import torch
import os
import sys
import time
import logging

logger = logging.getLogger(__name__)

# =====================================================================
# FP4 utilities
# =====================================================================

FP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                        dtype=torch.float32)


def unpack_fp4_nibbles(packed):
    """Unpack [N, K/2] uint8 -> [N, K] nibbles on same device."""
    N, K_half = packed.shape
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = torch.empty(N, K_half * 2, dtype=torch.uint8, device=packed.device)
    nibbles[:, 0::2] = low
    nibbles[:, 1::2] = high
    return nibbles


def apply_2of4_sparsity_fast(nibbles):
    """Vectorized 2:4 sparsity. Returns comp [N, K/4], meta [N, K/8]."""
    N, K = nibbles.shape
    lut = FP4_LUT.to(nibbles.device)
    mags = lut[nibbles.long()].abs()

    nib_4 = nibbles.view(N, K // 4, 4)
    mag_4 = mags.view(N, K // 4, 4)

    _, top2 = mag_4.topk(2, dim=2)
    top2, _ = top2.sort(dim=2)

    v0 = nib_4.gather(2, top2[:, :, 0:1]).squeeze(2)
    v1 = nib_4.gather(2, top2[:, :, 1:2]).squeeze(2)
    comp = ((v1 << 4) | v0).to(torch.uint8)

    i0 = top2[:, :, 0]
    i1 = top2[:, :, 1]
    i0_p = i0.view(N, K // 8, 2)
    i1_p = i1.view(N, K // 8, 2)

    meta = ((i0_p[:, :, 0] & 3) |
            ((i1_p[:, :, 0] & 3) << 2) |
            ((i0_p[:, :, 1] & 3) << 4) |
            ((i1_p[:, :, 1] & 3) << 6)).to(torch.uint8)

    return comp, meta


def convert_expert_to_v7_gpu(weight, scale, g_scale_val):
    """Convert one expert from NVFP4 packed to v7 sparse format on GPU.

    weight: [N, K/2] uint8
    scale: [N, n_groups] float8_e4m3fn
    g_scale_val: float

    Returns: comp_T [K/4, N], meta_T [K/8, N], scale_T [n_groups, N]
    """
    nibbles = unpack_fp4_nibbles(weight)
    comp, meta = apply_2of4_sparsity_fast(nibbles)
    comp_T = comp.T.contiguous()
    meta_T = meta.T.contiguous()
    scale_T = scale.view(torch.uint8).T.contiguous()
    return comp_T, meta_T, scale_T


def convert_weight_batch_to_v7(weights, scales, g_scales, device='cuda',
                                batch_size=16):
    """Convert batch of experts to v7 sparse format.

    weights: [E, N, K/2] uint8 (CPU)
    scales: [E, N, n_groups] float8_e4m3fn (CPU)
    g_scales: [E] float32 (CPU)

    Returns GPU tensors:
        comp_T: [E, K/4, N] uint8
        meta_T: [E, K/8, N] uint8
        scale_T: [E, n_groups, N] uint8
        g_scales: [E] float32
    """
    E = weights.shape[0]
    comp_list, meta_list, scale_list = [], [], []

    for start in range(0, E, batch_size):
        end = min(start + batch_size, E)
        batch_w = weights[start:end].to(device)
        batch_s = scales[start:end].to(device)

        for i in range(end - start):
            ct, mt, st = convert_expert_to_v7_gpu(batch_w[i], batch_s[i], 0)
            comp_list.append(ct)
            meta_list.append(mt)
            scale_list.append(st)

    comp_T = torch.stack(comp_list)    # [E, K/4, N]
    meta_T = torch.stack(meta_list)    # [E, K/8, N]
    scale_T = torch.stack(scale_list)  # [E, n_groups, N]
    g_scales_gpu = g_scales.float().to(device)

    return comp_T, meta_T, scale_T, g_scales_gpu


# =====================================================================
# V7 Sparse MoE Expert Class (for standalone testing)
# =====================================================================

class SparseV7Experts:
    """Standalone v7 sparse experts for testing outside vLLM."""

    def __init__(self, w13_comp, w13_meta, w13_scale, w13_g_scales,
                       w2_comp, w2_meta, w2_scale, w2_g_scales,
                       inter_size, hidden_size, activation='silu'):
        self.w13_comp = w13_comp
        self.w13_meta = w13_meta
        self.w13_scale = w13_scale
        self.w13_g_scales = w13_g_scales
        self.w2_comp = w2_comp
        self.w2_meta = w2_meta
        self.w2_scale = w2_scale
        self.w2_g_scales = w2_g_scales
        self.inter_size = inter_size
        self.hidden_size = hidden_size

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
              activation, global_num_experts, expert_map, a1q_scale, a2_scale,
              workspace13, workspace2, expert_tokens_meta,
              apply_router_weight_on_input):
        """V7 sparse MoE forward pass."""
        import sparse_fp4_v7 as sp7

        M, K = hidden_states.shape
        topk = topk_ids.shape[1]
        N = self.inter_size

        if expert_map is not None:
            mapped_ids = expert_map[topk_ids]
        else:
            mapped_ids = topk_ids

        A = hidden_states.unsqueeze(1).expand(-1, topk, -1).reshape(
            M * topk, K).contiguous()
        eid = mapped_ids.reshape(-1).int()

        gate_up = sp7.batched_sparse_v7(
            A, self.w13_comp, self.w13_meta,
            self.w13_scale, self.w13_g_scales, eid)

        gate = gate_up[:, :N]
        up = gate_up[:, N:]
        intermediate = torch.nn.functional.silu(gate) * up

        down = sp7.batched_sparse_v7(
            intermediate, self.w2_comp, self.w2_meta,
            self.w2_scale, self.w2_g_scales, eid)

        if apply_router_weight_on_input:
            reduced = down.view(M, topk, -1).sum(1)
        else:
            weights = topk_weights.unsqueeze(-1)
            reduced = (down.view(M, topk, -1).float() *
                       weights.float()).sum(1)

        output[:M] = reduced.to(output.dtype)
        return output


# =====================================================================
# V7 apply function factory (closure over converted weights)
# =====================================================================

def _make_v7_apply_fn(w13_comp, w13_meta, w13_scale, w13_g_scales,
                      w2_comp, w2_meta, w2_scale, w2_g_scales,
                      inter_size, hidden_size):
    """Create a v7 sparse apply function as closure over converted weights.

    The returned function is assigned as an instance attribute on MarlinExperts,
    shadowing the class method. Python does NOT inject self for instance
    attribute functions, so all weight data is captured in the closure.

    Call signature matches MarlinExperts.apply (minus self):
      apply(output, hidden_states, w1, w2, topk_weights=, topk_ids=, ...)
    """
    import sparse_fp4_v7 as sp7
    _fused = sp7.fused_sparse_moe_v7
    _empty_map = torch.empty(0, dtype=torch.int32)

    def v7_apply(output, hidden_states, w1, w2, topk_weights, topk_ids,
                 activation, global_num_experts, expert_map, a1q_scale,
                 a2_scale, workspace13, workspace2, expert_tokens_meta,
                 apply_router_weight_on_input):

        emap = expert_map if expert_map is not None else _empty_map
        result = _fused(
            hidden_states, topk_weights.float(), topk_ids,
            emap,
            w13_comp, w13_meta, w13_scale, w13_g_scales,
            w2_comp, w2_meta, w2_scale, w2_g_scales,
            inter_size, apply_router_weight_on_input)

        output[:hidden_states.shape[0]] = result.to(output.dtype)
        return output

    return v7_apply


# =====================================================================
# Monkey-patch functions
# =====================================================================

_original_process_weights = None
_patched = False


def _patched_process_weights_after_loading(self, layer):
    """Intercept weight processing to add v7 sparse kernel.

    Strategy:
    1. Clone original weights before Marlin repacking destroys them
    2. Call original process_weights_after_loading (Marlin + moe_mk setup)
    3. Convert cloned weights to v7 sparse format
    4. Replace MarlinExperts.apply with v7 closure (keeps all other methods)
    """
    t0 = time.time()

    # 1. Clone original weights to CPU before Marlin repacking
    #    (cloning on GPU would cause peak memory issues)
    w13_weight = layer.w13_weight.data.cpu().clone()
    w2_weight = layer.w2_weight.data.cpu().clone()
    w13_scale = layer.w13_weight_scale.data.cpu().clone()
    w2_scale = layer.w2_weight_scale.data.cpu().clone()
    w13_g_scale = layer.w13_weight_scale_2.data.cpu().clone()
    w2_g_scale = layer.w2_weight_scale_2.data.cpu().clone()

    # Extract dimensions
    E = w13_weight.shape[0]
    inter2 = w13_weight.shape[1]  # 2 * moe_intermediate_size
    inter_size = inter2 // 2
    hidden_half = w13_weight.shape[2]
    hidden_size = hidden_half * 2

    logger.info(f"[v7 sparse] Converting MoE layer: {E} experts, "
                f"inter={inter_size}, hidden={hidden_size}")

    # 2. Call original process_weights_after_loading
    #    This does Marlin repacking, builds FusedMoEModularKernel with
    #    MarlinExperts that has all required interface methods
    _original_process_weights(self, layer)

    # 3. Convert cloned weights to v7 sparse format
    w13_gs = w13_g_scale[:, 0] if w13_g_scale.dim() > 1 else w13_g_scale
    w2_gs = w2_g_scale[:, 0] if w2_g_scale.dim() > 1 else w2_g_scale

    w13_comp, w13_meta, w13_sc, w13_g = convert_weight_batch_to_v7(
        w13_weight, w13_scale, w13_gs, device='cuda', batch_size=16)
    w2_comp, w2_meta, w2_sc, w2_g = convert_weight_batch_to_v7(
        w2_weight, w2_scale, w2_gs, device='cuda', batch_size=16)

    t_convert = time.time() - t0
    logger.info(f"[v7 sparse] Conversion done in {t_convert:.1f}s, "
                f"w13 comp {list(w13_comp.shape)}, w2 comp {list(w2_comp.shape)}")

    # 4. Monkey-patch MarlinExperts.apply with v7 closure
    if hasattr(self, 'moe_mk') and self.moe_mk is not None:
        experts = self.moe_mk.fused_experts
        v7_fn = _make_v7_apply_fn(
            w13_comp, w13_meta, w13_sc, w13_g,
            w2_comp, w2_meta, w2_sc, w2_g,
            inter_size, hidden_size)
        # Instance attribute shadows class method — no self injection
        experts.apply = v7_fn

        # Free Marlin-repacked weights to save GPU memory for MTP.
        # The big tensors are on the layer as nn.Parameters (w13_weight,
        # w2_weight). Our v7 closure ignores the w1/w2 args passed to
        # apply(), but moe_problem_size() reads their .shape, so we
        # replace with zero-strided dummies that preserve shape metadata
        # while using only 1 element of storage.
        freed = 0
        for attr in ('w13_weight', 'w2_weight'):
            if hasattr(layer, attr):
                param = getattr(layer, attr)
                orig_shape = param.data.shape
                freed += param.data.numel() * param.data.element_size()
                stub = torch.zeros(1, device='cuda', dtype=param.dtype)
                param.data = stub.as_strided(
                    orig_shape, [0] * len(orig_shape))
        for attr in ('w13_weight_scale', 'w2_weight_scale',
                     'w13_weight_scale_2', 'w2_weight_scale_2'):
            if hasattr(layer, attr):
                param = getattr(layer, attr)
                if hasattr(param, 'data'):
                    orig_shape = param.data.shape
                    freed += param.data.numel() * param.data.element_size()
                    stub = torch.zeros(1, device='cuda', dtype=param.dtype)
                    param.data = stub.as_strided(
                        orig_shape, [0] * len(orig_shape))
        torch.cuda.empty_cache()
        logger.info(f"[v7 sparse] Freed {freed / 1e6:.1f} MB of "
                    f"Marlin-repacked weights")
        logger.info("[v7 sparse] Patched MarlinExperts.apply with v7 kernel")
    else:
        logger.warning("[v7 sparse] moe_mk not found, could not patch")


def patch_vllm():
    """Apply v7 sparse MoE patches to vLLM."""
    global _original_process_weights, _patched

    if _patched:
        logger.info("[v7 sparse] Already patched")
        return

    try:
        import sparse_fp4_v7
    except ImportError:
        logger.error("[v7 sparse] sparse_fp4_v7 not installed! "
                     "Run: python setup_v7.py install")
        return

    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4FusedMoE)

    _original_process_weights = ModelOptNvFp4FusedMoE.process_weights_after_loading

    ModelOptNvFp4FusedMoE.process_weights_after_loading = (
        _patched_process_weights_after_loading)

    _patched = True
    logger.info("[v7 sparse] vLLM MoE patches applied successfully")


# Auto-apply when imported
if os.environ.get('VLLM_NVFP4_SPARSE_V7', '0') == '1':
    patch_vllm()
