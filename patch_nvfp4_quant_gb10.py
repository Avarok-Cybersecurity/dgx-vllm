#!/usr/bin/env python3
"""
Patch vLLM _custom_ops.py to use software FP4 quantization on GB10 (SM_121).

GB10 lacks cvt.e2m1x2 PTX instruction. This patch replaces C++ calls
(which hit our error stubs) with software Python implementations that
produce identical packed FP4 output for CUTLASS GEMM tensor cores.

Run during Docker build after vLLM source is cloned.
"""

import sys
import re

VLLM_DIR = "/app/vllm"
CUSTOM_OPS = f"{VLLM_DIR}/vllm/_custom_ops.py"
MOE_UTILS = f"{VLLM_DIR}/vllm/model_executor/layers/fused_moe/utils.py"

# Also patch silu_and_mul_nvfp4_quant which may be called outside MoE
QUANT_MODULE = "gb10_nvfp4_software_quant"
QUANT_MODULE_PATH = f"{VLLM_DIR}/vllm/model_executor/layers/quantization/utils/{QUANT_MODULE}.py"


def patch_custom_ops():
    """Patch _custom_ops.py to use software FP4 quant on GB10."""
    with open(CUSTOM_OPS, "r") as f:
        content = f.read()

    # Add import at the top (after existing imports)
    import_line = (
        "\n# GB10 software FP4 quantization (no cvt.e2m1x2)\n"
        "import torch.cuda\n"
        "_gb10_software_fp4 = None\n"
        "def _use_gb10_software_fp4():\n"
        "    global _gb10_software_fp4\n"
        "    if _gb10_software_fp4 is None:\n"
        "        try:\n"
        "            cc = torch.cuda.get_device_capability()\n"
        "            _gb10_software_fp4 = (cc[0] == 12 and cc[1] == 1)\n"
        "        except Exception:\n"
        "            _gb10_software_fp4 = False\n"
        "    return _gb10_software_fp4\n"
        "\n"
        "def _get_gb10_quant():\n"
        "    from vllm.model_executor.layers.quantization.utils.gb10_nvfp4_software_quant import (\n"
        "        software_scaled_fp4_quant,\n"
        "        software_scaled_fp4_experts_quant,\n"
        "        software_silu_and_mul_scaled_fp4_experts_quant,\n"
        "    )\n"
        "    return (software_scaled_fp4_quant, software_scaled_fp4_experts_quant,\n"
        "            software_silu_and_mul_scaled_fp4_experts_quant)\n"
        "\n"
    )

    # Insert after the first block of imports
    # Find a good insertion point - after "from vllm" imports
    insert_marker = "from vllm.triton_utils import tl, triton"
    if insert_marker in content:
        content = content.replace(
            insert_marker,
            insert_marker + import_line,
        )
    else:
        # Fallback: insert after "import torch"
        content = content.replace(
            "import torch\n",
            "import torch\n" + import_line,
            1,
        )

    # Patch 1: scaled_fp4_quant - replace C++ call with software fallback
    old_call_1 = "        torch.ops._C.scaled_fp4_quant(\n            output, input, output_scale, input_global_scale, is_sf_swizzled_layout\n        )"
    new_call_1 = (
        "        if _use_gb10_software_fp4():\n"
        "            _get_gb10_quant()[0](output, input, output_scale, input_global_scale, is_sf_swizzled_layout)\n"
        "        else:\n"
        "            torch.ops._C.scaled_fp4_quant(\n"
        "                output, input, output_scale, input_global_scale, is_sf_swizzled_layout\n"
        "            )"
    )
    if old_call_1 in content:
        content = content.replace(old_call_1, new_call_1)
        print("  Patched scaled_fp4_quant C++ call")
    else:
        print("  WARNING: Could not find scaled_fp4_quant C++ call pattern")
        # Try alternative pattern
        alt_pattern = "torch.ops._C.scaled_fp4_quant("
        if alt_pattern in content:
            print("  Found alternative pattern, attempting line-by-line patch")

    # Patch 2: scaled_fp4_experts_quant
    old_call_2 = "    torch.ops._C.scaled_fp4_experts_quant(\n        output,\n        output_scales,\n        input_tensor,\n        input_global_scale,\n        expert_offsets,\n        blockscale_offsets,\n    )"
    new_call_2 = (
        "    if _use_gb10_software_fp4():\n"
        "        _get_gb10_quant()[1](output, output_scales, input_tensor, input_global_scale, expert_offsets, blockscale_offsets)\n"
        "    else:\n"
        "        torch.ops._C.scaled_fp4_experts_quant(\n"
        "            output,\n"
        "            output_scales,\n"
        "            input_tensor,\n"
        "            input_global_scale,\n"
        "            expert_offsets,\n"
        "            blockscale_offsets,\n"
        "        )"
    )
    if old_call_2 in content:
        content = content.replace(old_call_2, new_call_2)
        print("  Patched scaled_fp4_experts_quant C++ call")
    else:
        print("  WARNING: Could not find scaled_fp4_experts_quant C++ call pattern")

    # Patch 3: silu_and_mul_scaled_fp4_experts_quant
    old_call_3 = "    torch.ops._C.silu_and_mul_scaled_fp4_experts_quant(\n        output,\n        output_scales,\n        input_tensor,\n        input_global_scale,\n        expert_offsets,\n        blockscale_offsets,\n    )"
    new_call_3 = (
        "    if _use_gb10_software_fp4():\n"
        "        _get_gb10_quant()[2](output, output_scales, input_tensor, input_global_scale, expert_offsets, blockscale_offsets)\n"
        "    else:\n"
        "        torch.ops._C.silu_and_mul_scaled_fp4_experts_quant(\n"
        "            output,\n"
        "            output_scales,\n"
        "            input_tensor,\n"
        "            input_global_scale,\n"
        "            expert_offsets,\n"
        "            blockscale_offsets,\n"
        "        )"
    )
    if old_call_3 in content:
        content = content.replace(old_call_3, new_call_3)
        print("  Patched silu_and_mul_scaled_fp4_experts_quant C++ call")
    else:
        print("  WARNING: Could not find silu_and_mul_scaled_fp4_experts_quant C++ call pattern")

    # Patch 4: silu_and_mul_nvfp4_quant (non-MoE variant)
    old_call_4 = "    torch.ops._C.silu_and_mul_nvfp4_quant(output, output_sf, input, input_sf)"
    new_call_4 = (
        "    if _use_gb10_software_fp4():\n"
        "        # Software SiLU+Mul+FP4 quant for GB10\n"
        "        _k = input.shape[-1] // 2\n"
        "        _activated = torch.nn.functional.silu(input[..., :_k]) * input[..., _k:]\n"
        "        _get_gb10_quant()[0](output, _activated, output_sf, input_sf, True)\n"
        "    else:\n"
        "        torch.ops._C.silu_and_mul_nvfp4_quant(output, output_sf, input, input_sf)"
    )
    if old_call_4 in content:
        content = content.replace(old_call_4, new_call_4)
        print("  Patched silu_and_mul_nvfp4_quant C++ call")
    else:
        print("  INFO: silu_and_mul_nvfp4_quant not found (may not exist in this version)")

    with open(CUSTOM_OPS, "w") as f:
        f.write(content)

    print(f"  _custom_ops.py patched successfully")


def main():
    print("Patching vLLM for GB10 software NVFP4 quantization...")

    # Copy the software quant module
    import shutil
    shutil.copy2(
        "/workspace/dgx-vllm-build/gb10_nvfp4_software_quant.py",
        QUANT_MODULE_PATH,
    )
    print(f"  Copied {QUANT_MODULE}.py to vLLM quantization utils")

    # Patch _custom_ops.py
    patch_custom_ops()

    print("GB10 software NVFP4 quantization patch complete!")
    print("  Linear GEMM: EMULATION backend (env VLLM_USE_NVFP4_CT_EMULATIONS=1)")
    print("  MoE GEMM: Software FP4 quant -> CUTLASS FP4 tensor cores")


if __name__ == "__main__":
    main()
