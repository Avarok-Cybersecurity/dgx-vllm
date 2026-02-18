#!/usr/bin/env python3
"""
Fix FlashInfer CUTLASS E2M1 conversion PTX on SM121 (GB10).

Bug: FlashInfer's bundled CUTLASS float_subbyte.h defines CUDA_PTX_FP4FP6_CVT_ENABLED
for SM121A, but SM121 doesn't support the `cvt.rn.satfinite.e2m1x2.f32` and
`cvt.rn.f16x2.e2m1x2` PTX instructions. This causes JIT compilation failure when
building the fused MoE CUTLASS kernels.

Fix: Remove SM121A and SM121F from the CUDA_PTX_FP4FP6_CVT_ENABLED condition,
so CUTLASS uses its software fallback for FP4 E2M1 conversions on SM121.
The tensor core MMA operations (tcgen05.mma) still work natively on SM121 — only
the data format conversion PTX is missing.

Also patches quantization_utils.cuh (TRT-LLM internal) to add software E2M1
fallback for SM121 in case other FlashInfer operations need it.

Also fixes the custom cuda_fp4.h header to guard __device__ functions with
#ifdef __CUDACC__ so it compiles when included from host C++ code.
"""

import sys
import os
import shutil


def patch_float_subbyte():
    """Remove SM121 from CUDA_PTX_FP4FP6_CVT_ENABLED in CUTLASS float_subbyte.h"""
    target = (
        "/opt/venv/lib/python3.12/site-packages/flashinfer/data/cutlass/"
        "include/cutlass/float_subbyte.h"
    )

    if not os.path.exists(target):
        print(f"ERROR: {target} not found")
        return False

    with open(target, "r") as f:
        content = f.read()

    # Pattern 1: SM*A_ENABLED block
    old_a = (
        '#if (defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) || '
        'defined(CUTLASS_ARCH_MMA_SM101A_ENABLED) ||\\\n'
        '     defined(CUTLASS_ARCH_MMA_SM103A_ENABLED) || '
        'defined(CUTLASS_ARCH_MMA_SM110A_ENABLED) ||\\\n'
        '     defined(CUTLASS_ARCH_MMA_SM120A_ENABLED) || '
        'defined(CUTLASS_ARCH_MMA_SM121A_ENABLED))\n'
        '#  define CUDA_PTX_FP4FP6_CVT_ENABLED 1'
    )
    new_a = (
        '/* SM121 removed: cvt.rn.satfinite.e2m1x2.f32 not supported on GB10 */\n'
        '#if (defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) || '
        'defined(CUTLASS_ARCH_MMA_SM101A_ENABLED) ||\\\n'
        '     defined(CUTLASS_ARCH_MMA_SM103A_ENABLED) || '
        'defined(CUTLASS_ARCH_MMA_SM110A_ENABLED) ||\\\n'
        '     defined(CUTLASS_ARCH_MMA_SM120A_ENABLED))\n'
        '#  define CUDA_PTX_FP4FP6_CVT_ENABLED 1'
    )

    # Pattern 2: SM*F_ENABLED block
    old_f = (
        '#if (defined(CUTLASS_ARCH_MMA_SM100F_ENABLED) || '
        'defined(CUTLASS_ARCH_MMA_SM101F_ENABLED) ||\\\n'
        '     defined(CUTLASS_ARCH_MMA_SM103F_ENABLED) || '
        'defined(CUTLASS_ARCH_MMA_SM110F_ENABLED) ||\\\n'
        '     defined(CUTLASS_ARCH_MMA_SM120F_ENABLED) || '
        'defined(CUTLASS_ARCH_MMA_SM121F_ENABLED))\n'
        '#  define CUDA_PTX_FP4FP6_CVT_ENABLED 1'
    )
    new_f = (
        '/* SM121 removed: cvt.rn.satfinite.e2m1x2.f32 not supported on GB10 */\n'
        '#if (defined(CUTLASS_ARCH_MMA_SM100F_ENABLED) || '
        'defined(CUTLASS_ARCH_MMA_SM101F_ENABLED) ||\\\n'
        '     defined(CUTLASS_ARCH_MMA_SM103F_ENABLED) || '
        'defined(CUTLASS_ARCH_MMA_SM110F_ENABLED) ||\\\n'
        '     defined(CUTLASS_ARCH_MMA_SM120F_ENABLED))\n'
        '#  define CUDA_PTX_FP4FP6_CVT_ENABLED 1'
    )

    patched = False

    if old_a in content:
        content = content.replace(old_a, new_a)
        patched = True
        print(f"  Patched SM*A block: removed SM121A from CUDA_PTX_FP4FP6_CVT_ENABLED")
    elif 'SM121 removed' in content and 'SM121A' not in content.split('CUDA_PTX_FP4FP6_CVT_ENABLED')[1][:200]:
        print(f"  SM*A block: already patched")
    else:
        print(f"  WARNING: Could not find SM*A pattern in float_subbyte.h")

    if old_f in content:
        content = content.replace(old_f, new_f)
        patched = True
        print(f"  Patched SM*F block: removed SM121F from CUDA_PTX_FP4FP6_CVT_ENABLED")
    elif 'SM121 removed' in content:
        print(f"  SM*F block: already patched")
    else:
        print(f"  WARNING: Could not find SM*F pattern in float_subbyte.h")

    if patched:
        with open(target, "w") as f:
            f.write(content)
        print(f"  Written: {target}")

    return True


def patch_quantization_utils():
    """Add software E2M1 fallback in TRT-LLM quantization_utils.cuh for SM121."""
    target = (
        "/opt/venv/lib/python3.12/site-packages/flashinfer/data/csrc/"
        "nv_internal/tensorrt_llm/kernels/quantization_utils.cuh"
    )

    if not os.path.exists(target):
        print(f"  SKIP: {target} not found")
        return True

    with open(target, "r") as f:
        content = f.read()

    # The functions use __CUDA_ARCH__ >= 1000 but SM121 (__CUDA_ARCH__ == 1210)
    # doesn't support cvt.rn.satfinite.e2m1x2.f32.
    # Replace the guard to exclude SM121.

    old_guard = '#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)'
    new_guard = '#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ != 1210)'

    if old_guard in content:
        content = content.replace(old_guard, new_guard)

        # Also add a software E2M1 helper before the first function that uses it.
        # The existing #else branches return 0 which is wrong for SM121.
        # Add a proper software E2M1 conversion.
        sw_e2m1_helper = '''
// Software E2M1 conversion for SM121 (GB10) which lacks cvt.rn.satfinite.e2m1x2.f32
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1210)
inline __device__ uint8_t _sw_float_to_e2m1_single(float v) {
    // E2M1: 1 sign + 2 exponent + 1 mantissa, bias=1
    // Values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    uint32_t bits = __float_as_uint(v);
    uint8_t sign = (bits >> 31) & 1;
    float av = fabsf(v);
    uint8_t e2m1;
    if (av < 0.25f)       e2m1 = 0;         // 0
    else if (av < 0.75f)  e2m1 = 1;         // 0.5
    else if (av < 1.25f)  e2m1 = 2;         // 1.0
    else if (av < 1.75f)  e2m1 = 3;         // 1.5
    else if (av < 2.5f)   e2m1 = 4;         // 2.0
    else if (av < 3.5f)   e2m1 = 5;         // 3.0
    else if (av < 5.0f)   e2m1 = 6;         // 4.0
    else                  e2m1 = 7;         // 6.0 (satfinite)
    return (sign << 3) | e2m1;
}

inline __device__ uint8_t _sw_float2_to_e2m1x2(float lo, float hi) {
    return (_sw_float_to_e2m1_single(lo) & 0xF) |
           ((_sw_float_to_e2m1_single(hi) & 0xF) << 4);
}

inline __device__ uint32_t _sw_fp32x8_to_e2m1x8(float* arr) {
    uint8_t b0 = _sw_float2_to_e2m1x2(arr[0], arr[1]);
    uint8_t b1 = _sw_float2_to_e2m1x2(arr[2], arr[3]);
    uint8_t b2 = _sw_float2_to_e2m1x2(arr[4], arr[5]);
    uint8_t b3 = _sw_float2_to_e2m1x2(arr[6], arr[7]);
    return (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
}

inline __device__ uint32_t _sw_fp32x8_to_e2m1x8_f2(float2* arr) {
    float flat[8] = {arr[0].x, arr[0].y, arr[1].x, arr[1].y,
                     arr[2].x, arr[2].y, arr[3].x, arr[3].y};
    return _sw_fp32x8_to_e2m1x8(flat);
}

inline __device__ uint64_t _sw_fp32x16_to_e2m1x16_f2(float2* arr) {
    float flat0[8] = {arr[0].x, arr[0].y, arr[1].x, arr[1].y,
                      arr[2].x, arr[2].y, arr[3].x, arr[3].y};
    float flat1[8] = {arr[4].x, arr[4].y, arr[5].x, arr[5].y,
                      arr[6].x, arr[6].y, arr[7].x, arr[7].y};
    uint32_t lo = _sw_fp32x8_to_e2m1x8(flat0);
    uint32_t hi = _sw_fp32x8_to_e2m1x8(flat1);
    return (uint64_t)lo | ((uint64_t)hi << 32);
}
#endif  // SM121 software E2M1
'''

        # Insert the helper before the first fp32_vec_to_e2m1 function
        insert_marker = '// Convert 8 float32 values into 8 e2m1 values'
        if insert_marker in content:
            content = content.replace(insert_marker, sw_e2m1_helper + '\n' + insert_marker)

        # Now fix the #else branches that return 0 to use software conversion
        # Pattern: the fp32_vec_to_e2m1(float (&array)[8]) #else
        content = content.replace(
            '#else\n  // static_assert(false, "not supported.");\n  return 0;\n#endif\n}\n\n'
            '// Convert 4 float2 values into 8 e2m1 values',
            '#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1210)\n'
            '  return _sw_fp32x8_to_e2m1x8(array);\n'
            '#else\n  // static_assert(false, "not supported.");\n  return 0;\n#endif\n}\n\n'
            '// Convert 4 float2 values into 8 e2m1 values'
        )

        # Fix fp32_vec_to_e2m1(float2 (&array)[4]) #else
        content = content.replace(
            '#else\n  // static_assert(false, "not supported.");\n  return 0;\n#endif\n}\n\n'
            '// Convert 8 float2 values into 16 e2m1 values',
            '#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1210)\n'
            '  return _sw_fp32x8_to_e2m1x8_f2(array);\n'
            '#else\n  // static_assert(false, "not supported.");\n  return 0;\n#endif\n}\n\n'
            '// Convert 8 float2 values into 16 e2m1 values'
        )

        # Fix fp32_vec_to_e2m1(float2 (&array)[8]) #else for 64-bit return
        content = content.replace(
            '#else\n  // static_assert(false, "not supported.");\n  return 0;\n#endif\n}\n\n'
            '// Dequantize e2m1 to bf16',
            '#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1210)\n'
            '  return _sw_fp32x16_to_e2m1x16_f2(array);\n'
            '#else\n  // static_assert(false, "not supported.");\n  return 0;\n#endif\n}\n\n'
            '// Dequantize e2m1 to bf16'
        )

        with open(target, "w") as f:
            f.write(content)
        print(f"  Patched: {target}")
        print(f"    - Excluded SM121 from PTX E2M1 path")
        print(f"    - Added software E2M1 fallback functions")
    elif '__CUDA_ARCH__ != 1210' in content:
        print(f"  quantization_utils.cuh: already patched")
    else:
        print(f"  WARNING: Could not find __CUDA_ARCH__ >= 1000 pattern")

    return True


def patch_cuda_fp4_header():
    """Fix custom cuda_fp4.h for both nvcc and host C++ compilation.

    Issues with the original:
    1. extern "C" block wraps C++ code, causing CCCL template linkage errors
       when compiled with c++ (host compiler) by FlashInfer JIT
    2. __device__ functions aren't guarded by __CUDACC__

    Fix: Remove extern "C", guard device functions with __CUDACC__.
    FlashInfer's vec_dtypes.cuh needs this file so we can't remove it.
    """
    target = "/usr/local/cuda/include/cuda_fp4.h"

    # If file was renamed by a previous version of this script, restore it
    backup = "/usr/local/cuda/include/cuda_fp4_custom.h"
    if os.path.exists(backup) and not os.path.exists(target):
        os.rename(backup, target)
        print(f"  Restored: {backup} -> {target}")

    if not os.path.exists(target):
        print(f"  SKIP: {target} not found")
        return True

    with open(target, "r") as f:
        content = f.read()

    if "We're MAKING FP4 support" not in content:
        print(f"  SKIP: {target} is not the custom header")
        return True

    if '/* Fixed for FlashInfer JIT */' in content:
        print(f"  cuda_fp4.h: already patched")
        return True

    # Remove extern "C" { ... } entirely - it causes CCCL template linkage errors
    content = content.replace(
        '#ifdef __cplusplus\nextern "C" {\n#endif',
        '/* Fixed for FlashInfer JIT */'
    )
    content = content.replace(
        '#ifdef __cplusplus\n}\n#endif\n\n#endif',
        '#endif'
    )

    # Guard device functions with __CUDACC__
    content = content.replace(
        '// ============================================================================\n'
        '// DEVICE FUNCTIONS - Conversions\n'
        '// ============================================================================',
        '#ifdef __CUDACC__\n'
        '// ============================================================================\n'
        '// DEVICE FUNCTIONS - Conversions\n'
        '// ============================================================================'
    )

    # Close the __CUDACC__ guard before HOST FUNCTIONS section
    content = content.replace(
        '// ============================================================================\n'
        '// HOST FUNCTIONS (CPU-side conversions)\n'
        '// ============================================================================',
        '#endif  /* __CUDACC__ */\n\n'
        '// ============================================================================\n'
        '// HOST FUNCTIONS (CPU-side conversions)\n'
        '// ============================================================================'
    )

    with open(target, "w") as f:
        f.write(content)
    print(f"  Patched: {target}")
    print(f"    - Removed extern 'C' block")
    print(f"    - Guarded __device__ functions with __CUDACC__")

    return True


def clear_jit_cache():
    """Clear FlashInfer JIT cache to force recompilation with patched headers."""
    cache_dir = "/root/.cache/flashinfer"
    if os.path.exists(cache_dir):
        fused_moe_cache = os.path.join(cache_dir, "0.6.3", "121a", "cached_ops", "fused_moe_120")
        if os.path.exists(fused_moe_cache):
            shutil.rmtree(fused_moe_cache)
            print(f"  Cleared JIT cache: {fused_moe_cache}")
        else:
            # Try to find any fused_moe cache
            for root, dirs, files in os.walk(cache_dir):
                if 'fused_moe' in os.path.basename(root):
                    shutil.rmtree(root)
                    print(f"  Cleared JIT cache: {root}")
    else:
        print(f"  No JIT cache found (first run)")


def main():
    print("=== FlashInfer E2M1 SM121 Fix ===")
    print()

    print("[1/4] Patching CUTLASS float_subbyte.h...")
    if not patch_float_subbyte():
        sys.exit(1)
    print()

    print("[2/4] Patching TRT-LLM quantization_utils.cuh...")
    patch_quantization_utils()
    print()

    print("[3/4] Fixing custom cuda_fp4.h for host compilation...")
    patch_cuda_fp4_header()
    print()

    print("[4/4] Clearing FlashInfer JIT cache...")
    clear_jit_cache()
    print()

    print("Done. FlashInfer will use software E2M1 conversion on SM121.")


if __name__ == "__main__":
    main()
