#!/usr/bin/env python3
"""Force swizzle=False in run_nvfp4_emulations() for the EMULATION-backend
weight dequant call.

EMULATION-backend NVFP4 checkpoints store weight_scale_swizzled in LINEAR
format (compressed-tensors' convert_to_nvfp4_linear_kernel_format has no
EMULATION case, so scales are passed through from safetensors unchanged
despite the tensor name). Calling dequantize_to_dtype with swizzle=True
(the default) routes through convert_swizzled_to_linear, which scrambles
the already-linear scales and corrupts the model output.

vllm v0.21.0 added a Triton fast-path in dequantize_to_dtype when
swizzle=False (csrc-equivalent path: _triton_dequantize_nvfp4). That path
reads scales directly as linear and applies the standard sf * global_scale
formula, which is the correct dequant given that compressed-tensors stores
weight_global_scale as 1/actual_gs.

Earlier ports replaced the whole run_nvfp4_emulations body with a manual
PyTorch dequant. With v0.21.0 we only need to flip swizzle=swizzle to
swizzle=False on the weight dequant call site — the Triton fast path does
the rest, faster.
"""

path = "/app/vllm/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py"
with open(path) as f:
    content = f.read()

# Anchor on the full multi-line dequantize_to_dtype call inside
# run_nvfp4_emulations so we don't accidentally match a different caller.
OLD = '''    w_dq = dequantize_to_dtype(
        w_fp4,
        weight_scale_swizzled.data,
        weight_global_scale,
        output_dtype,
        group_size,
        swizzle=swizzle,
    )'''

NEW = '''    w_dq = dequantize_to_dtype(
        w_fp4,
        weight_scale_swizzled.data,
        weight_global_scale,
        output_dtype,
        group_size,
        # GB10/EMULATION patch: scales are stored LINEAR for
        # compressed-tensors EMULATION checkpoints despite the
        # "_swizzled" tensor name. Force swizzle=False to take the
        # Triton fast path that treats scales as linear.
        swizzle=False,
    )'''

if NEW in content:
    print("Already patched (swizzle=False sentinel present)")
elif OLD in content:
    content = content.replace(OLD, NEW)
    with open(path, "w") as f:
        f.write(content)
    print("Fix applied: run_nvfp4_emulations weight dequant now uses swizzle=False")
else:
    print("ERROR: anchor block not found in nvfp4_emulation_utils.py")
    idx = content.find("def run_nvfp4_emulations")
    if idx >= 0:
        print("Current run_nvfp4_emulations:")
        print(content[idx:idx + 800])
    exit(1)
