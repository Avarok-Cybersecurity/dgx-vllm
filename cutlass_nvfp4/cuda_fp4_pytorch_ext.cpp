// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda_fp4_pytorch_ext.cpp
 * @brief PyTorch C++ Extension for CUDA FP4
 *
 * This makes our CUDA FP4 extension available in Python/PyTorch!
 *
 * Usage in Python:
 *   import torch
 *   import cuda_fp4_ext
 *
 *   fp4_data = cuda_fp4_ext.quantize(fp32_tensor, group_size=16)
 *   output = cuda_fp4_ext.gemm(fp4_A, fp4_B, scales_A, scales_B)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declarations of CUDA kernels
void launch_fp4_quantize(
    void* dst_data,
    void* dst_scales,
    float* dst_scale_global,
    const float* src,
    int rows, int cols, int group_size,
    cudaStream_t stream
);

void launch_fp4_gemm(
    const void* A, const void* A_scales, float A_scale_global,
    const void* B, const void* B_scales, float B_scale_global,
    float* C,
    int M, int N, int K,
    int group_size,
    cudaStream_t stream
);

void launch_fp4_dequantize(
    float* dst,
    const void* src_data,
    const void* src_scales,
    float src_scale_global,
    int rows, int cols, int group_size,
    cudaStream_t stream
);

// ============================================================================
// PyTorch-Compatible Quantization
// ============================================================================

/**
 * @brief Quantize FP32 tensor to FP4 with block scales
 *
 * Returns tuple of (data, scales, global_scale)
 */
std::tuple<torch::Tensor, torch::Tensor, float> quantize_fp4(
    torch::Tensor input,
    int64_t group_size = 16
) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be FP32");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D (M x N)");

    int64_t rows = input.size(0);
    int64_t cols = input.size(1);

    // Output: packed FP4 (uint8), each byte holds 2 FP4 values
    auto data = torch::empty({rows, (cols + 1) / 2},
                             torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));

    // Scales: FP8 per group
    int64_t num_groups = (cols + group_size - 1) / group_size;
    auto scales = torch::empty({rows, num_groups},
                               torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));

    // Global scale (will be computed on device, returned on host)
    float global_scale = 1.0f;

    // Launch CUDA kernel
    launch_fp4_quantize(
        data.data_ptr(),
        scales.data_ptr(),
        &global_scale,
        input.data_ptr<float>(),
        rows, cols, group_size,
        c10::cuda::getCurrentCUDAStream()
    );

    return std::make_tuple(data, scales, global_scale);
}

/**
 * @brief Dequantize FP4 tensor back to FP32
 */
torch::Tensor dequantize_fp4(
    torch::Tensor data,
    torch::Tensor scales,
    float global_scale,
    int64_t rows,
    int64_t cols,
    int64_t group_size = 16
) {
    TORCH_CHECK(data.is_cuda(), "Data must be CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "Scales must be CUDA tensor");
    TORCH_CHECK(data.dtype() == torch::kUInt8, "Data must be uint8");
    TORCH_CHECK(scales.dtype() == torch::kUInt8, "Scales must be uint8 (FP8)");

    // Output: FP32
    auto output = torch::empty({rows, cols},
                               torch::TensorOptions().dtype(torch::kFloat32).device(data.device()));

    // Launch CUDA kernel
    launch_fp4_dequantize(
        output.data_ptr<float>(),
        data.data_ptr(),
        scales.data_ptr(),
        global_scale,
        rows, cols, group_size,
        c10::cuda::getCurrentCUDAStream()
    );

    return output;
}

// ============================================================================
// FP4 GEMM
// ============================================================================

/**
 * @brief FP4 matrix multiplication
 *
 * C[M×N] = A[M×K] @ B[K×N]
 *
 * All inputs are FP4 with block scales
 */
torch::Tensor gemm_fp4(
    torch::Tensor A_data,
    torch::Tensor A_scales,
    float A_scale_global,
    torch::Tensor B_data,
    torch::Tensor B_scales,
    float B_scale_global,
    int64_t M, int64_t N, int64_t K,
    int64_t group_size = 16
) {
    TORCH_CHECK(A_data.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B_data.is_cuda(), "B must be CUDA tensor");
    TORCH_CHECK(A_data.dtype() == torch::kUInt8, "A must be uint8");
    TORCH_CHECK(B_data.dtype() == torch::kUInt8, "B must be uint8");

    // Output: FP32
    auto C = torch::empty({M, N},
                         torch::TensorOptions().dtype(torch::kFloat32).device(A_data.device()));

    // Launch CUDA kernel
    launch_fp4_gemm(
        A_data.data_ptr(),
        A_scales.data_ptr(),
        A_scale_global,
        B_data.data_ptr(),
        B_scales.data_ptr(),
        B_scale_global,
        C.data_ptr<float>(),
        M, N, K,
        group_size,
        c10::cuda::getCurrentCUDAStream()
    );

    return C;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Get FP4 extension version
 */
std::string get_version() {
    return "1.0.0";
}

/**
 * @brief Check if hardware acceleration available
 */
bool has_hardware_acceleration() {
#ifdef ENABLE_TCGEN05_HARDWARE
    return true;
#else
    return false;
#endif
}

/**
 * @brief Get device compute capability
 */
std::tuple<int, int> get_compute_capability() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    return std::make_tuple(prop.major, prop.minor);
}

// ============================================================================
// PyTorch Bindings
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA FP4 Extension for PyTorch - Community Built!";

    // Quantization functions
    m.def("quantize", &quantize_fp4,
          "Quantize FP32 tensor to FP4 with block scales",
          py::arg("input"),
          py::arg("group_size") = 16);

    m.def("dequantize", &dequantize_fp4,
          "Dequantize FP4 tensor to FP32",
          py::arg("data"),
          py::arg("scales"),
          py::arg("global_scale"),
          py::arg("rows"),
          py::arg("cols"),
          py::arg("group_size") = 16);

    // GEMM function
    m.def("gemm", &gemm_fp4,
          "FP4 matrix multiplication with block scales",
          py::arg("A_data"),
          py::arg("A_scales"),
          py::arg("A_scale_global"),
          py::arg("B_data"),
          py::arg("B_scales"),
          py::arg("B_scale_global"),
          py::arg("M"),
          py::arg("N"),
          py::arg("K"),
          py::arg("group_size") = 16);

    // Utility functions
    m.def("version", &get_version, "Get extension version");
    m.def("has_hardware_acceleration", &has_hardware_acceleration,
          "Check if hardware tensor cores available");
    m.def("get_compute_capability", &get_compute_capability,
          "Get device compute capability (major, minor)");
}
