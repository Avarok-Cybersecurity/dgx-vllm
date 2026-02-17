// SPDX-License-Identifier: Apache-2.0
/**
 * @file nvfp4_stubs.cu
 * @brief GB10 stub implementations for NVFP4 quantization functions
 *
 * GB10 (sm_121a) lacks cvt.e2m1x2 instruction for FP4 quantization.
 * These stubs match the exact function signatures from vLLM 0.16.x
 * nvfp4_quant_entry.cu declarations (guarded by ENABLE_NVFP4_SM120).
 *
 * Appended to nvfp4_quant_entry.cu only.
 */

// 1. scaled_fp4_quant_sm1xxa
void scaled_fp4_quant_sm1xxa(torch::Tensor const& output,
                              torch::Tensor const& input,
                              torch::Tensor const& output_sf,
                              torch::Tensor const& input_sf,
                              bool is_sf_swizzled_layout) {
  TORCH_CHECK(false, "scaled_fp4_quant_sm1xxa not supported on GB10 (missing cvt.e2m1x2)");
}

// 2. scaled_fp4_experts_quant_sm1xxa
void scaled_fp4_experts_quant_sm1xxa(
    torch::Tensor& output, torch::Tensor& output_scale,
    torch::Tensor const& input, torch::Tensor const& input_global_scale,
    torch::Tensor const& input_offset_by_experts,
    torch::Tensor const& output_scale_offset_by_experts) {
  TORCH_CHECK(false, "scaled_fp4_experts_quant_sm1xxa not supported on GB10 (missing cvt.e2m1x2)");
}

// 3. silu_and_mul_nvfp4_quant_sm1xxa
void silu_and_mul_nvfp4_quant_sm1xxa(torch::Tensor& output,
                                      torch::Tensor& output_sf,
                                      torch::Tensor& input,
                                      torch::Tensor& input_sf) {
  TORCH_CHECK(false, "silu_and_mul_nvfp4_quant_sm1xxa not supported on GB10 (missing cvt.e2m1x2)");
}

// 4. silu_and_mul_scaled_fp4_experts_quant_sm1xxa
void silu_and_mul_scaled_fp4_experts_quant_sm1xxa(
    torch::Tensor& output, torch::Tensor& output_scale,
    torch::Tensor const& input, torch::Tensor const& input_global_scale,
    torch::Tensor const& input_offset_by_experts,
    torch::Tensor const& output_scale_offset_by_experts) {
  TORCH_CHECK(false, "silu_and_mul_scaled_fp4_experts_quant_sm1xxa not supported on GB10 (missing cvt.e2m1x2)");
}
