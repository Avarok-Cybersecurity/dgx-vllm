/*
 * Native CUTLASS MoE Kernel for NVIDIA GeForce Blackwell (GB10)
 * Compute Capability: 12.1 (SM_121)
 *
 * Key GB10 Constraints (vs Data Center GB100):
 * - Cluster Shape: MUST be 1x1x1 (no multicast support)
 * - Layout: MUST be TN (transposed-normal only)
 * - Max Warps/SM: 48 (vs 64 on GB100)
 * - Memory Bandwidth: 273 GB/s (vs 8 TB/s on GB100)
 * - Kernel Schedule: 1SM Pingpong or Cooperative (NO 2SM variants)
 *
 * Optimizations for GB10:
 * - Smaller tile sizes for bandwidth constraints (128x256x128)
 * - Leverages TMEM (256 KB/SM) for accumulation
 * - Uses TMA for efficient global->shared transfers
 * - Optimized for unified memory architecture
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
// GB10-Specific CUTLASS MoE Kernel Configurations
// ============================================================================

/*
 * Default Configuration: 128x256x128 tile with 1SM Pingpong schedule
 *
 * This configuration is optimized for GB10's memory bandwidth constraints:
 * - Smaller tiles reduce memory pressure
 * - Pingpong schedule overlaps compute and memory ops
 * - 1x1x1 cluster required for GeForce (no multicast)
 * - TN layout required for SM120
 */
template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct gb10_fp8_config_default {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());

  // GB10 kernel schedule - use Sm100 but with conservative parameters
  // Runtime "Error Internal" might be from tile size or other config, not schedule
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;

  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;

  // CONSERVATIVE tile size - start small to avoid shared memory/register issues
  // GB10 has different limits than GB100, try minimal 64×128×64 first
  using TileShape = cute::Shape<cute::_64, cute::_128, cute::_64>;

  // GB10 REQUIRES 1x1x1 cluster (no multicast support in GeForce)
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  // Keep Sm100 arch tag since schedule requires it
  using ArchTag = cutlass::arch::Sm100;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, ArchTag, Epilogue, TileShape,
                            ClusterShape, KernelSchedule, EpilogueSchedule>;
};

/*
 * Small Batch Configuration: 128x128x128 tile
 *
 * For batch sizes 1-64 or when memory is constrained:
 * - Even smaller tiles for better cache locality
 * - Higher SM utilization with more thread blocks
 * - Lower shared memory usage (132 KB vs 196 KB)
 */
template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct gb10_fp8_config_small_batch {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());

  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;

  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;

  // Small batch: use VERY conservative 64×64×64 tiles
  using TileShape = cute::Shape<cute::_64, cute::_64, cute::_64>;

  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using ArchTag = cutlass::arch::Sm100;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, ArchTag, Epilogue, TileShape,
                            ClusterShape, KernelSchedule, EpilogueSchedule,
                            true>;  // enable_m_tiling for small M
};

// Cooperative configuration removed - KernelPtrArrayTmaCooperative not available in this CUTLASS version

// ============================================================================
// GB10 MoE Kernel Runtime Implementation
// ============================================================================

/*
 * Main kernel execution function with adaptive configuration selection
 *
 * Selects optimal kernel configuration based on problem size:
 * - M <= 64: Use small batch config (128x128x128)
 * - M > 64: Use default WarpSpecialized config (128x256x128)
 */
template <typename InType, typename OutType>
void run_gb10_moe_mm(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_token, bool per_out_ch) {

  // Input validation
  TORCH_CHECK(a_tensors.size(0) > 0, "No input A tensors provided.");
  TORCH_CHECK(b_tensors.size(0) > 0, "No input B tensors provided.");
  TORCH_CHECK(out_tensors.size(0) > 0, "No output tensors provided.");

  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn,
              "A tensors must be of type float8_e4m3fn.");
  TORCH_CHECK(b_tensors.dtype() == torch::kFloat8_e4m3fn,
              "B tensors must be of type float8_e4m3fn.");

  // Define kernel types for each configuration
  using Cutlass3xGemmDefault = typename gb10_fp8_config_default<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;

  using Cutlass3xGemmSmallBatch = typename gb10_fp8_config_small_batch<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;

  uint32_t const m = a_tensors.size(0);

  // Adaptive kernel selection based on problem size
  if (m <= 64) {
    // Small batch: Use 128x128x128 for better SM utilization
    cutlass_group_gemm_caller<Cutlass3xGemmSmallBatch>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  } else {
    // Default: WarpSpecialized schedule with 128x256x128 tiles
    cutlass_group_gemm_caller<Cutlass3xGemmDefault>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  }
}

/*
 * Type dispatcher - selects appropriate instantiation based on output type
 */
void dispatch_gb10_moe_mm(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_token, bool per_out_ch) {

  if (out_tensors.dtype() == torch::kBFloat16) {
    run_gb10_moe_mm<cutlass::float_e4m3_t, cutlass::bfloat16_t>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  } else {
    run_gb10_moe_mm<cutlass::float_e4m3_t, cutlass::half_t>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  }
}

} // namespace gb10
} // namespace vllm

// ============================================================================
// Public API - Exported to PyTorch
// ============================================================================

/*
 * Main entry point for GB10 MoE matrix multiplication
 *
 * This function is called from vLLM's Python layer through PyTorch's
 * custom ops system.
 */
extern "C" void cutlass_moe_mm_gb10(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_token, bool per_out_ch) {

  vllm::gb10::dispatch_gb10_moe_mm(
      out_tensors, a_tensors, b_tensors, a_scales, b_scales,
      expert_offsets, problem_sizes, a_strides, b_strides,
      c_strides, per_act_token, per_out_ch);
}

// Cooperative variant removed - not supported in this CUTLASS version
