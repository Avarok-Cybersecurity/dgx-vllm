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

# Detect csrc layout. v0.20.0 moved csrc/quantization/fp4/* into
# csrc/libtorch_stable/quantization/fp4/* and renamed the build target
# from _C to _C_stable_libtorch. Older versions kept everything in _C.
if [ -d "csrc/libtorch_stable/quantization/fp4" ]; then
    NVFP4_DIR="csrc/libtorch_stable/quantization/fp4"
    NVFP4_TARGET="_C_stable_libtorch"
    echo "Using libtorch_stable structure (vLLM >= v0.20.0), target=${NVFP4_TARGET}"
elif [ -d "csrc/quantization/fp4" ]; then
    NVFP4_DIR="csrc/quantization/fp4"
    NVFP4_TARGET="_C"
    echo "Using legacy structure (vLLM <= v0.19.x), target=${NVFP4_TARGET}"
else
    echo "ERROR: Neither csrc/libtorch_stable/quantization/fp4 nor csrc/quantization/fp4 found"
    exit 1
fi

echo "Patching CMakeLists.txt for FULL NVFP4 kernel compilation on GB10..."

# Use unquoted heredoc so $NVFP4_DIR and $NVFP4_TARGET expand at script time.
# CMake variables (\${VAR}) are escaped to keep them as CMake-time substitution.
cat >> CMakeLists.txt << CMAKE_PATCH

# ============================================================================
# CUSTOM: GB10 Full NVFP4 Compilation v6 (ALL KERNELS - no stubs!)
# ============================================================================
# GB10 (sm_121) now has software E2M1 conversion in nvfp4_utils.cuh.
# All quant kernels compile for SM121 using this software path.
# MoE GEMM and scaled_mm kernels use CUTLASS mma.e2m1 (always worked).
#
# In vLLM v0.20.0+, the kernel files moved into csrc/libtorch_stable/
# and got built into target _C_stable_libtorch instead of _C. This script
# detects the layout at run time so it works against either tag.
# ============================================================================

if(\${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
  message(STATUS "GB10 Custom v6: Compiling ALL NVFP4 kernels for sm_121 (target: ${NVFP4_TARGET})")

  # Entry files are already added to the target's source list upstream.
  set(GB10_NVFP4_ENTRY_FILES
    "${NVFP4_DIR}/nvfp4_quant_entry.cu"
    "${NVFP4_DIR}/nvfp4_scaled_mm_entry.cu"
  )

  # Kernel files compile for sm_121 thanks to the software-E2M1 patch in
  # nvfp4_utils.cuh. MoE GEMM and scaled_mm use CUTLASS BlockScaled MMA.
  set(GB10_NVFP4_KERNEL_FILES
    "${NVFP4_DIR}/nvfp4_quant_kernels.cu"
    "${NVFP4_DIR}/nvfp4_experts_quant.cu"
    "${NVFP4_DIR}/activation_nvfp4_quant_fusion_kernels.cu"
    "${NVFP4_DIR}/nvfp4_blockwise_moe_kernel.cu"
    "${NVFP4_DIR}/nvfp4_scaled_mm_sm120_kernels.cu"
  )

  set(GB10_NVFP4_ALL_FILES \${GB10_NVFP4_ENTRY_FILES} \${GB10_NVFP4_KERNEL_FILES})

  set_gencode_flags_for_srcs(
    SRCS "\${GB10_NVFP4_ALL_FILES}"
    CUDA_ARCHS "12.1"
  )

  set_source_files_properties(
    \${GB10_NVFP4_ALL_FILES}
    PROPERTIES
    COMPILE_DEFINITIONS "ENABLE_NVFP4_SM120=1"
  )

  # Add kernel files to the actual build target (already defined at this point).
  if(TARGET ${NVFP4_TARGET})
    target_sources(${NVFP4_TARGET} PRIVATE \${GB10_NVFP4_KERNEL_FILES})
    target_compile_definitions(${NVFP4_TARGET} PRIVATE ENABLE_NVFP4_SM120=1)
  else()
    message(WARNING "GB10 Custom v6: target ${NVFP4_TARGET} not defined, skipping")
  endif()

  message(STATUS "GB10 Custom v6: ALL kernel files compiled with sm_121 + ENABLE_NVFP4_SM120")
endif()

# ============================================================================

CMAKE_PATCH

echo "CMakeLists.txt patched for full NVFP4 kernel compilation!"
echo "  Source dir: ${NVFP4_DIR}"
echo "  Build target: ${NVFP4_TARGET}"
echo "  Flag: ENABLE_NVFP4_SM120=1"
