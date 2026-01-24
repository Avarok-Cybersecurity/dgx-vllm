#pragma once

#include "scaled_mm.cuh"
#include "cutlass_gemm_caller.cuh"

/**
 * Blockwise scaled GEMM for SM121 (GB10)
 *
 * NOTE: Blockwise quantization for SM_121 is currently not implemented.
 * The SM100/SM120 blockwise implementations use architecture-specific
 * ScaleConfig features that require significant additional work to adapt.
 *
 * For now, blockwise calls will throw an error directing users to use
 * per-tensor quantization instead.
 */

namespace vllm {

template <typename OutType>
void cutlass_gemm_blockwise_sm121_fp8_dispatch(torch::Tensor& out,
                                               torch::Tensor const& a,
                                               torch::Tensor const& b,
                                               torch::Tensor const& a_scales,
                                               torch::Tensor const& b_scales) {
  TORCH_CHECK(false,
              "Blockwise quantization is not yet implemented for SM_121 (GB10). "
              "Please use per-tensor quantization instead. "
              "This is a known limitation and will be addressed in a future release.");
}

}  // namespace vllm
