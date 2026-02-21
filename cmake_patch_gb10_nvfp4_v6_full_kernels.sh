#!/bin/bash
# Selective NVFP4 Compilation for GB10 - v6 FULL KERNELS (no stubs needed!)
#
# v6 changes from v5:
# - NO stubs appended to any entry file
# - COMPILES ALL kernel files for sm_121:
#   1. nvfp4_quant_kernels.cu - scaled_fp4_quant (activation quant)
#   2. nvfp4_experts_quant.cu - per-expert FP4 quant + silu_mul variant
#   3. activation_nvfp4_quant_fusion_kernels.cu - silu_and_mul_nvfp4_quant
#   4. nvfp4_blockwise_moe_kernel.cu - CUTLASS FP4 MoE GEMM
#   5. nvfp4_scaled_mm_sm120_kernels.cu - CUTLASS FP4 GEMM (non-MoE)
#
# This is possible because nvfp4_utils.cuh has been patched with software
# E2M1 conversion that replaces cvt.rn.satfinite.e2m1x2.f32 PTX on SM121.
#
# With real C++ quant kernels:
# - No Python software fallback needed (eliminating .item() calls)
# - CUDA graph capture becomes possible (no GPU→CPU transfers)
# - Expected 10-27x speedup from eliminating Python per-op overhead

set -e

VLLM_DIR="/app/vllm"
cd "$VLLM_DIR"

echo "Patching CMakeLists.txt for FULL NVFP4 kernel compilation on GB10..."

# NOTE: No stubs appended! All symbols are provided by real compiled kernels.

cat >> CMakeLists.txt << 'CMAKE_PATCH'

# ============================================================================
# CUSTOM: GB10 Full NVFP4 Compilation v6 (ALL KERNELS - no stubs!)
# ============================================================================
# GB10 (sm_121) now has software E2M1 conversion in nvfp4_utils.cuh.
# All quant kernels compile for SM121 using this software path.
# MoE GEMM and scaled_mm kernels use CUTLASS mma.e2m1 (always worked).
#
# This eliminates:
# - Python software FP4 quantization fallback (gb10_nvfp4_software_quant.py)
# - Quant function stubs (nvfp4_stubs.cu)
# - .item() calls in Python per-expert loop
# - CUDA graph capture failures (cudaErrorStreamCaptureUnsupported)
# ============================================================================

if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
  message(STATUS "GB10 Custom v6: Compiling ALL NVFP4 kernels for sm_121")

  # Entry files are ALREADY in VLLM_EXT_SRC (added at line ~344).
  set(GB10_NVFP4_ENTRY_FILES
    "csrc/quantization/fp4/nvfp4_quant_entry.cu"
    "csrc/quantization/fp4/nvfp4_scaled_mm_entry.cu"
  )

  # ALL kernel files - none need stubs anymore!
  # Quant kernels now compile because nvfp4_utils.cuh has software E2M1.
  # MoE GEMM and scaled_mm use CUTLASS BlockScaled MMA (always worked).
  set(GB10_NVFP4_KERNEL_FILES
    "csrc/quantization/fp4/nvfp4_quant_kernels.cu"
    "csrc/quantization/fp4/nvfp4_experts_quant.cu"
    "csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu"
    "csrc/quantization/fp4/nvfp4_blockwise_moe_kernel.cu"
    "csrc/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu"
  )

  # Combine for setting properties
  set(GB10_NVFP4_ALL_FILES ${GB10_NVFP4_ENTRY_FILES} ${GB10_NVFP4_KERNEL_FILES})

  # Set arch to sm_121 for all files
  set_gencode_flags_for_srcs(
    SRCS "${GB10_NVFP4_ALL_FILES}"
    CUDA_ARCHS "12.1"
  )

  # Set compile definition on all files
  set_source_files_properties(
    ${GB10_NVFP4_ALL_FILES}
    PROPERTIES
    COMPILE_DEFINITIONS "ENABLE_NVFP4_SM120=1"
  )

  # Add kernel files to the _C target directly (target already exists)
  target_sources(_C PRIVATE ${GB10_NVFP4_KERNEL_FILES})

  message(STATUS "GB10 Custom v6: ALL kernel files compiled with sm_121 + ENABLE_NVFP4_SM120")
  message(STATUS "GB10 Custom v6: No stubs needed - software E2M1 in nvfp4_utils.cuh")
endif()

# ============================================================================

CMAKE_PATCH

echo "CMakeLists.txt patched for full NVFP4 kernel compilation!"
echo "  nvfp4_quant_entry.cu: vLLM original (NO stubs)"
echo "  nvfp4_scaled_mm_entry.cu: vLLM original (NO stubs)"
echo "  nvfp4_quant_kernels.cu: Activation FP4 quant (software E2M1)"
echo "  nvfp4_experts_quant.cu: Per-expert FP4 quant (software E2M1)"
echo "  activation_nvfp4_quant_fusion_kernels.cu: SiLU+Mul+FP4 quant"
echo "  nvfp4_blockwise_moe_kernel.cu: CUTLASS FP4 MoE GEMM"
echo "  nvfp4_scaled_mm_sm120_kernels.cu: CUTLASS FP4 GEMM"
echo "  Flag: ENABLE_NVFP4_SM120=1"
