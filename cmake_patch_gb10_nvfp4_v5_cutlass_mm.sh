#!/bin/bash
# Selective NVFP4 Compilation for GB10 - v5 CUTLASS SCALED_MM + MOE GEMM
# CRITICAL: Append stubs to entry files (which ARE compiled)
# Quant stubs -> nvfp4_quant_entry.cu (still needed - cvt.e2m1x2 missing)
# Scaled MM -> COMPILE THE REAL KERNEL (same CUTLASS templates as MoE!)
# Also compile MoE GEMM kernel (uses mma.e2m1, NOT cvt.e2m1x2)
#
# v5 changes from v4:
# - NO LONGER appends scaled_mm stubs to nvfp4_scaled_mm_entry.cu
# - COMPILES nvfp4_scaled_mm_sm120_kernels.cu for sm_121
#   (uses identical CUTLASS Sm120 BlockScaledTensorOp templates as MoE kernel)
# - This enables CUTLASS FP4 GEMM for non-MoE layers (5-15x faster than EMULATION)

set -e

VLLM_DIR="/app/vllm"
cd "$VLLM_DIR"

echo "Patching CMakeLists.txt + APPENDING stubs to NVFP4 entry files..."

# Append quant stubs (1-4) to nvfp4_quant_entry.cu
# These are still needed because GB10 lacks cvt.e2m1x2 for quantization
echo "Appending quant stubs to nvfp4_quant_entry.cu..."
cat /workspace/dgx-vllm-build/nvfp4_stubs.cu >> "${VLLM_DIR}/csrc/quantization/fp4/nvfp4_quant_entry.cu"

# NOTE: We do NOT append scaled_mm stubs anymore!
# Instead, we compile the real nvfp4_scaled_mm_sm120_kernels.cu kernel.
echo "NOT appending scaled_mm stubs - will compile real kernel instead..."

cat >> CMakeLists.txt << 'CMAKE_PATCH'

# ============================================================================
# CUSTOM: GB10 Selective NVFP4 Compilation v5 (CUTLASS SCALED_MM + MOE GEMM)
# ============================================================================
# GB10 (sm_121) has mma.e2m1 but lacks cvt.e2m1x2
# Compile: Entry files + MoE GEMM kernel + SCALED_MM kernel
# Skip: Quantization kernels (use cvt.e2m1x2)
#
# v5: Also compiles nvfp4_scaled_mm_sm120_kernels.cu!
# This kernel uses the same CUTLASS templates as the MoE kernel:
#   - ArchTag: cutlass::arch::Sm120
#   - OperatorClass: cutlass::arch::OpClassBlockScaledTensorOp
#   - ElementA/B: cutlass::nv_float4_t<cutlass::float_e2m1_t>
#   - ClusterShape: 1x1x1
#   - KernelScheduleAuto
# All confirmed working on sm_121 via the MoE kernel.
# ============================================================================

if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
  message(STATUS "GB10 Custom v5: Compiling NVFP4 entry files + MoE GEMM + SCALED_MM kernel for sm_121")

  # Entry files are ALREADY in VLLM_EXT_SRC (added at line ~344).
  # We only need to set their gencode flags and compile definitions.
  set(GB10_NVFP4_ENTRY_FILES
    "csrc/quantization/fp4/nvfp4_quant_entry.cu"
    "csrc/quantization/fp4/nvfp4_scaled_mm_entry.cu"
  )

  # MoE GEMM kernel - uses CUTLASS BlockScaled MMA (mma.e2m1), NOT cvt.e2m1x2
  # Scaled MM kernel - uses IDENTICAL CUTLASS BlockScaled MMA templates
  # Both files are NOT in the initial VLLM_EXT_SRC because FP4_ARCHS is empty
  # for sm_121 (intersection("12.0f", [12.1]) = empty).
  # Must use target_sources since _C target is already created.
  set(GB10_NVFP4_KERNEL_FILES
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

  message(STATUS "GB10 Custom v5: Entry stubs + MoE GEMM + SCALED_MM kernel compiled with sm_121 + ENABLE_NVFP4_SM120")
  message(STATUS "GB10 Custom v5: Kernel files added via target_sources(_C)")
  message(STATUS "GB10 Custom v5: Skipping quantization kernels (cvt.e2m1x2 not supported)")
endif()

# ============================================================================

CMAKE_PATCH

echo "CMakeLists.txt patched + stubs appended!"
echo "  nvfp4_quant_entry.cu: vLLM original + 4 quant stubs"
echo "  nvfp4_scaled_mm_entry.cu: vLLM original (NO stubs - real kernel compiled)"
echo "  nvfp4_blockwise_moe_kernel.cu: CUTLASS FP4 MoE GEMM (mma.e2m1)"
echo "  nvfp4_scaled_mm_sm120_kernels.cu: CUTLASS FP4 GEMM (mma.e2m1)"
echo "  Flag: ENABLE_NVFP4_SM120=1"
echo "  Skipping: nvfp4_quant_kernels.cu, nvfp4_experts_quant.cu (cvt.e2m1x2)"
