#include "c3x/scaled_mm_helper.hpp"
#include "c3x/scaled_mm_kernels.hpp"

/**
 * This file defines quantized GEMM operations using the CUTLASS 3.x API,
 * for NVIDIA GB10 with SM_121 (DGX Spark Blackwell).
 *
 * Key differences from SM120 (GB100 data center):
 * - SM_121 vs SM_120: Consumer/workstation variant
 * - 301 GB/s LPDDR5X vs 8 TB/s HBM3e
 * - Unified CPU/GPU memory vs discrete GPU
 * - No cluster multicast support (1x1x1 only)
 */

#if defined ENABLE_SCALED_MM_SM121 && ENABLE_SCALED_MM_SM121

void cutlass_scaled_mm_sm121(torch::Tensor& c, torch::Tensor const& a,
                             torch::Tensor const& b,
                             torch::Tensor const& a_scales,
                             torch::Tensor const& b_scales,
                             std::optional<torch::Tensor> const& bias) {
  // GB10 (SM_121) is Blackwell family - use SM100 kernels which are compatible
  // SM100 has 1x1x1 cluster shape configs that work on GB10's hardware constraints
  dispatch_scaled_mm(c, a, b, a_scales, b_scales, bias,
                     vllm::cutlass_scaled_mm_sm100_fp8,
                     nullptr,  // int8 not supported on SM121
                     vllm::cutlass_scaled_mm_blockwise_sm100_fp8);
}

#endif
