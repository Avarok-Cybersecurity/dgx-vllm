#pragma once

#include "scaled_mm.cuh"
#include "cutlass_gemm_caller.cuh"

/**
 * This file defines Gemm kernel configurations for SM121 (GB10 - fp8)
 * Optimized for NVIDIA GB10 (DGX Spark) hardware characteristics:
 * - ClusterShape 1x1x1 (no multicast support)
 * - 301 GB/s LPDDR5X unified memory
 * - Simpler scheduling than SM100/SM120 data center variants
 */

namespace vllm {

using c3x::cutlass_gemm_caller;

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm121_fp8_config_default {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());

  // GB10-specific configuration matching SM100 that works
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;

  // Optimized tile shape for GB10's 301 GB/s bandwidth
  using TileShape = Shape<_128, _256, _128>;

  // GB10 CRITICAL: Only 1x1x1 cluster shape supported (no multicast)
  using ClusterShape = Shape<_1, _1, _1>;

  // Use Sm100 arch tag for Blackwell generation (works with 12.1f gencode)
  using Cutlass3xGemm =
      cutlass_3x_gemm_sm100<InType, OutType, Epilogue, TileShape, ClusterShape,
                            KernelSchedule, EpilogueSchedule>;
};

// Optimized configuration for small batches
template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm121_fp8_config_small {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());

  // Use Auto schedules matching SM100
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;

  // Smaller tile for better occupancy on small problems
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;

  using Cutlass3xGemm =
      cutlass_3x_gemm_sm100<InType, OutType, Epilogue, TileShape, ClusterShape,
                            KernelSchedule, EpilogueSchedule>;
};

// Optimized configuration for large batches
template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm121_fp8_config_large {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());

  // Use Auto schedules matching SM100
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;

  // Larger tile for throughput on large problems
  using TileShape = Shape<_128, _256, _128>;
  using ClusterShape = Shape<_1, _1, _1>;

  using Cutlass3xGemm =
      cutlass_3x_gemm_sm100<InType, OutType, Epilogue, TileShape, ClusterShape,
                            KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
inline void cutlass_gemm_sm121_fp8_dispatch(torch::Tensor& out,
                                            torch::Tensor const& a,
                                            torch::Tensor const& b,
                                            EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

  using Cutlass3xGemmDefault =
      typename sm121_fp8_config_default<InType, OutType,
                                        Epilogue>::Cutlass3xGemm;
  return cutlass_gemm_caller<Cutlass3xGemmDefault>(
      out, a, b, std::forward<EpilogueArgs>(args)...);
}

template <template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_mm_sm121_fp8_epilogue(torch::Tensor& out,
                                          torch::Tensor const& a,
                                          torch::Tensor const& b,
                                          EpilogueArgs&&... epilogue_args) {
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

  if (out.dtype() == torch::kBFloat16) {
    return cutlass_gemm_sm121_fp8_dispatch<cutlass::float_e4m3_t,
                                           cutlass::bfloat16_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    return cutlass_gemm_sm121_fp8_dispatch<cutlass::float_e4m3_t,
                                           cutlass::half_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

// Simplified adaptive selection based on problem size
template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
inline void cutlass_gemm_sm121_fp8_dispatch_adaptive(torch::Tensor& out,
                                            torch::Tensor const& a,
                                            torch::Tensor const& b,
                                            EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

  // Adaptive kernel selection based on problem size
  uint32_t const m = a.size(0);
  uint32_t const n = b.size(1);

  using Cutlass3xGemmDefault =
      typename sm121_fp8_config_default<InType, OutType,
                                        Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmSmall =
      typename sm121_fp8_config_small<InType, OutType,
                                      Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmLarge =
      typename sm121_fp8_config_large<InType, OutType,
                                      Epilogue>::Cutlass3xGemm;

  // Heuristic selection
  if (m * n < 65536) {
    // Small problem: use smaller tiles for better occupancy
    return cutlass_gemm_caller<Cutlass3xGemmSmall>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (m * n > 1048576) {
    // Large problem: use larger tiles for throughput
    return cutlass_gemm_caller<Cutlass3xGemmLarge>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else {
    // Default configuration
    return cutlass_gemm_caller<Cutlass3xGemmDefault>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  }
}

}  // namespace vllm
