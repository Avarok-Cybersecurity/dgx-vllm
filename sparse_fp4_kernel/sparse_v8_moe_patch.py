#!/usr/bin/env python3
"""Monkey-patch vLLM to use v8 sparse FP4 kernel for MoE layers.

Same strategy as v7 patch but uses the v8 optimized kernel with:
1. Vectorized 64-bit loads (THREAD_N=8)
2. AtomicAdd accumulation (eliminates .sum(1) reduction kernel)
3. Pre-scaled FP8 LUT in shared memory (folds global_scale into LUT)

Usage: Import this module before model loading.
  import sparse_v8_moe_patch  # applies patches automatically

Or explicitly:
  from sparse_v8_moe_patch import patch_vllm
  patch_vllm()
"""

import torch
import os
import sys
import time
import logging

logger = logging.getLogger(__name__)

# Reuse all FP4 utilities and weight conversion from v7 patch
from sparse_v7_moe_patch import (
    convert_weight_batch_to_v7,
)


# =====================================================================
# V8 apply function factory (closure over converted weights)
# =====================================================================

def _make_v8_apply_fn(w13_comp, w13_meta, w13_scale, w13_g_scales,
                      w2_comp, w2_meta, w2_scale, w2_g_scales,
                      inter_size, hidden_size):
    """Create a v8 sparse apply function as closure over converted weights.

    Same interface as v7 but calls fused_sparse_moe_v8.
    """
    import sparse_fp4_v8 as sp8
    _fused = sp8.fused_sparse_moe_v8
    _empty_map = torch.empty(0, dtype=torch.int32)

    def v8_apply(output, hidden_states, w1, w2, topk_weights, topk_ids,
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

    return v8_apply


# =====================================================================
# Monkey-patch functions
# =====================================================================

_original_process_weights = None
_patched = False


def _patched_process_weights_after_loading(self, layer):
    """Intercept weight processing to add v8 sparse kernel.

    Same strategy as v7:
    1. Clone original weights before Marlin repacking destroys them
    2. Call original process_weights_after_loading (Marlin + moe_mk setup)
    3. Convert cloned weights to sparse format (same as v7)
    4. Replace MarlinExperts.apply with v8 closure
    """
    t0 = time.time()

    # 1. Clone original weights to CPU before Marlin repacking
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

    logger.info(f"[v8 sparse] Converting MoE layer: {E} experts, "
                f"inter={inter_size}, hidden={hidden_size}")

    # 2. Call original process_weights_after_loading
    _original_process_weights(self, layer)

    # 3. Convert cloned weights to sparse format (reuses v7 conversion)
    w13_gs = w13_g_scale[:, 0] if w13_g_scale.dim() > 1 else w13_g_scale
    w2_gs = w2_g_scale[:, 0] if w2_g_scale.dim() > 1 else w2_g_scale

    w13_comp, w13_meta, w13_sc, w13_g = convert_weight_batch_to_v7(
        w13_weight, w13_scale, w13_gs, device='cuda', batch_size=16)
    w2_comp, w2_meta, w2_sc, w2_g = convert_weight_batch_to_v7(
        w2_weight, w2_scale, w2_gs, device='cuda', batch_size=16)

    t_convert = time.time() - t0
    logger.info(f"[v8 sparse] Conversion done in {t_convert:.1f}s, "
                f"w13 comp {list(w13_comp.shape)}, w2 comp {list(w2_comp.shape)}")

    # 4. Monkey-patch MarlinExperts.apply with v8 closure
    if hasattr(self, 'moe_mk') and self.moe_mk is not None:
        experts = self.moe_mk.fused_experts
        v8_fn = _make_v8_apply_fn(
            w13_comp, w13_meta, w13_sc, w13_g,
            w2_comp, w2_meta, w2_sc, w2_g,
            inter_size, hidden_size)
        experts.apply = v8_fn

        # Free Marlin-repacked weights (same as v7)
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
        logger.info(f"[v8 sparse] Freed {freed / 1e6:.1f} MB of "
                    f"Marlin-repacked weights")
        logger.info("[v8 sparse] Patched MarlinExperts.apply with v8 kernel")
    else:
        logger.warning("[v8 sparse] moe_mk not found, could not patch")


def patch_vllm():
    """Apply v8 sparse MoE patches to vLLM."""
    global _original_process_weights, _patched

    if _patched:
        logger.info("[v8 sparse] Already patched")
        return

    try:
        import sparse_fp4_v8
    except ImportError:
        logger.error("[v8 sparse] sparse_fp4_v8 not installed! "
                     "Run: python setup_v8.py install")
        return

    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4FusedMoE)

    _original_process_weights = ModelOptNvFp4FusedMoE.process_weights_after_loading

    ModelOptNvFp4FusedMoE.process_weights_after_loading = (
        _patched_process_weights_after_loading)

    _patched = True
    logger.info("[v8 sparse] vLLM MoE patches applied successfully")


# Auto-apply when imported
if os.environ.get('VLLM_NVFP4_SPARSE_V8', '0') == '1':
    patch_vllm()
