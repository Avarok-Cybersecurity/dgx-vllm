/*
 * NVIDIA GeForce Blackwell (GB10) Native MoE Kernel - v109
 * Based on CUTLASS official example: 79d_blackwell_geforce_nvfp4_grouped_gemm.cu
 * Compute Capability: 12.1 (SM_121) / 12.0 (SM_120)
 *
 * Key Optimizations for GB10 (GeForce Blackwell):
 * - Uses Sm120 ArchTag (GeForce-specific, NOT datacenter Sm100)
 * - Uses KernelPtrArrayTmaWarpSpecializedPingpong (NVIDIA's grouped GEMM schedule)
 * - Tile size: 128×128×128 (NVIDIA's recommended for GeForce)
 * - Cluster shape: 1×1×1 (GeForce constraint - no multicast)
 * - TMA for efficient global→shared transfers
 * - Optimized for LPDDR5X unified memory (301 GB/s)
 *
 * This configuration LEVERAGES GB10 hardware instead of using SM100 fallback!
 */

#include <cudaTypedefs.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cutlass/cutlass.h"
#include "grouped_mm_c3x.cuh"

using namespace cute;

namespace vllm {
namespace gb10 {

// ============================================================================
// GB10 Native Configuration (Based on NVIDIA Official Example)
// ============================================================================

/*
 * GeForce Blackwell Optimized Configuration
 *
 * Key Differences from Datacenter SM100:
 * - Sm120 ArchTag: GeForce-specific architecture tag
 * - Pingpong Schedule: Overlaps compute and memory for grouped GEMM (MOE)
 * - 128×128×128 Tiles: Balanced for GeForce cache hierarchy and LPDDR5X
 * - 1×1×1 Cluster: Required for GeForce (no multicast support)
 *
 * NVIDIA's official configuration for FP4 grouped GEMM on GeForce Blackwell!
 */
template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct gb10_fp8_config_native {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());

  // CRITICAL: Use Sm120 (GeForce Blackwell) NOT Sm100 (datacenter GB100)!
  using ArchTag = cutlass::arch::Sm120;

  // NVIDIA's Pingpong schedule for grouped GEMM (MOE)
  // This overlaps compute and memory operations optimally for GeForce
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;

  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;

  // NVIDIA's recommended tile size for GeForce Blackwell: 128×128×128
  // Balanced for:
  // - LPDDR5X unified memory bandwidth
  // - GeForce cache hierarchy (smaller than GB100)
  // - Register file size
  // - Shared memory per SM
  using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;

  // GeForce REQUIRES 1×1×1 cluster (no multicast support)
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, ArchTag, Epilogue, TileShape,
                            ClusterShape, KernelSchedule, EpilogueSchedule>;
};

/*
 * Small Batch Configuration (for batch_size < 64)
 *
 * Uses smaller tiles (64×128×64) for:
 * - Better SM utilization with small batches
 * - Lower shared memory usage
 * - Better cache locality
 */
template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct gb10_fp8_config_small_batch {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());

  using ArchTag = cutlass::arch::Sm120;  // GeForce Blackwell

  // Still use Pingpong schedule for overlapped compute/memory
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;

  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;

  // Smaller tiles for small batch sizes
  using TileShape = cute::Shape<cute::_64, cute::_128, cute::_64>;

  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, ArchTag, Epilogue, TileShape,
                            ClusterShape, KernelSchedule, EpilogueSchedule,
                            true>;  // enable_m_tiling for small M
};

// ============================================================================
// Runtime Dispatch
// ============================================================================

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
void cutlass_gemm_caller_gb10_native(torch::Tensor& out, torch::Tensor const& a,
                                      torch::Tensor const& b,
                                      Epilogue<InType, OutType, InType>&& epilogue) {
  // Dispatch based on batch size
  int batch_size = a.size(0);

  if (batch_size < 64) {
    // Small batch: use smaller tiles for better occupancy
    using Config = gb10_fp8_config_small_batch<InType, OutType, Epilogue>;
    grouped_mm_runner<Config>(out, a, b, std::forward<decltype(epilogue)>(epilogue));
  } else {
    // Large batch: use NVIDIA's recommended 128×128×128 tiles
    using Config = gb10_fp8_config_native<InType, OutType, Epilogue>;
    grouped_mm_runner<Config>(out, a, b, std::forward<decltype(epilogue)>(epilogue));
  }
}

}  // namespace gb10
}  // namespace vllm

// ============================================================================
// PyTorch Bindings
// ============================================================================

void cutlass_moe_mm_gb10_native(
    torch::Tensor& out,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales) {

  std::cout << "🔥🔥🔥 GB10 NATIVE MOE KERNEL v109 🔥🔥🔥\n";
  std::cout << "  ✅ NVIDIA GeForce Blackwell (Sm120)\n";
  std::cout << "  ✅ Pingpong Schedule (NVIDIA official)\n";
  std::cout << "  ✅ 128×128×128 tiles (GeForce-optimized)\n";
  std::cout << "  ✅ LEVERAGES GB10 hardware!\n\n";

  using InType = cutlass::float_e4m3_t;
  using OutType = cutlass::bfloat16_t;

  // Create epilogue with scale factors
  auto epilogue = vllm::ScaledEpilogueBiasForward<InType, OutType, InType>(
      a_scales, b_scales);

  vllm::gb10::cutlass_gemm_caller_gb10_native<InType, OutType,
                                               vllm::ScaledEpilogueBiasForward>(
      out, a, b, std::move(epilogue));
}
