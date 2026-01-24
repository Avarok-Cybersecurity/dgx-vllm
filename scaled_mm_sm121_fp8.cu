#include "scaled_mm_kernels.hpp"
#include "scaled_mm_sm121_fp8_dispatch.cuh"
#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"

/**
 * Native FP8 Scaled Matrix Multiplication for SM_121 (NVIDIA GB10)
 *
 * This kernel is specifically optimized for GB10 (DGX Spark) hardware:
 * - Compute Capability 12.1 (SM_121)
 * - 301 GB/s LPDDR5X unified memory
 * - No cluster multicast support (1x1x1 only)
 * - 5th generation Tensor Cores
 */

namespace vllm {

void cutlass_scaled_mm_sm121_fp8(torch::Tensor& out, torch::Tensor const& a,
                                 torch::Tensor const& b,
                                 torch::Tensor const& a_scales,
                                 torch::Tensor const& b_scales,
                                 std::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  if (bias) {
    TORCH_CHECK(bias->dtype() == out.dtype(),
                "currently bias dtype must match output dtype ", out.dtype());
    return cutlass_scaled_mm_sm121_fp8_epilogue<c3x::ScaledEpilogueBias>(
        out, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm121_fp8_epilogue<c3x::ScaledEpilogue>(
        out, a, b, a_scales, b_scales);
  }
}

}  // namespace vllm
