// SPDX-License-Identifier: Apache-2.0
/**
 * @file test_nvfp4_gemm_hardware.cu
 * @brief Unit tests for hardware tensor core NVFP4 GEMM
 *
 * Tests the tensor core accelerated kernel
 */

#include "nvfp4_gemm_simple_hw.cuh"
#include "nvfp4_gemm_kernel.cuh"  // For reference implementation
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>

using namespace cutlass_nvfp4;

/**
 * @brief Test hardware kernel against software reference
 */
bool test_hardware_vs_software() {
    std::cout << "\n=== Testing Hardware Tensor Core Kernel vs Software ===" << std::endl;

    const int M = 128;
    const int N = 128;
    const int K = 64;
    const int GROUP_SIZE = 16;

    // Allocate host memory
    std::vector<nvfp4x2_t> h_A(M * K / 2);
    std::vector<__nv_fp8_e4m3> h_A_scales(M * K / GROUP_SIZE);
    std::vector<nvfp4x2_t> h_B(N * K / 2);
    std::vector<__nv_fp8_e4m3> h_B_scales(N * K / GROUP_SIZE);
    std::vector<float> h_C_hardware(M * N);
    std::vector<float> h_C_software(M * N);

    // Initialize with random values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    for (auto& val : h_A) {
        float v1 = dist(rng);
        float v2 = dist(rng);
        val = nvfp4x2_t::from_floats(v1, v2);
    }

    for (auto& val : h_B) {
        float v1 = dist(rng);
        float v2 = dist(rng);
        val = nvfp4x2_t::from_floats(v1, v2);
    }

    // Simple scales for validation
    for (auto& scale : h_A_scales) scale = __nv_fp8_e4m3(1.0f);
    for (auto& scale : h_B_scales) scale = __nv_fp8_e4m3(1.0f);

    float A_scale_global = 1.0f;
    float B_scale_global = 1.0f;

    // Allocate device memory
    nvfp4x2_t* d_A;
    __nv_fp8_e4m3* d_A_scales;
    nvfp4x2_t* d_B;
    __nv_fp8_e4m3* d_B_scales;
    float* d_C_hardware;
    float* d_C_software;

    cudaMalloc(&d_A, h_A.size() * sizeof(nvfp4x2_t));
    cudaMalloc(&d_A_scales, h_A_scales.size() * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_B, h_B.size() * sizeof(nvfp4x2_t));
    cudaMalloc(&d_B_scales, h_B_scales.size() * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_C_hardware, h_C_hardware.size() * sizeof(float));
    cudaMalloc(&d_C_software, h_C_software.size() * sizeof(float));

    // Copy to device
    cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(nvfp4x2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_scales, h_A_scales.data(), h_A_scales.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(nvfp4x2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_scales, h_B_scales.data(), h_B_scales.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);

    // Launch HARDWARE kernel
    cudaEvent_t hw_start, hw_stop;
    cudaEventCreate(&hw_start);
    cudaEventCreate(&hw_stop);

    cudaEventRecord(hw_start);
    launch_nvfp4_gemm_simple_hw(
        d_A, d_A_scales, A_scale_global,
        d_B, d_B_scales, B_scale_global,
        d_C_hardware, M, N, K
    );
    cudaEventRecord(hw_stop);
    cudaDeviceSynchronize();

    float hw_time = 0;
    cudaEventElapsedTime(&hw_time, hw_start, hw_stop);

    // Launch SOFTWARE kernel (reference)
    cudaEvent_t sw_start, sw_stop;
    cudaEventCreate(&sw_start);
    cudaEventCreate(&sw_stop);

    cudaEventRecord(sw_start);
    launch_nvfp4_gemm(
        d_A, d_A_scales, A_scale_global,
        d_B, d_B_scales, B_scale_global,
        d_C_software, M, N, K
    );
    cudaEventRecord(sw_stop);
    cudaDeviceSynchronize();

    float sw_time = 0;
    cudaEventElapsedTime(&sw_time, sw_start, sw_stop);

    // Copy results back
    cudaMemcpy(h_C_hardware.data(), d_C_hardware, h_C_hardware.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_software.data(), d_C_software, h_C_software.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare results
    bool passed = true;
    float max_error = 0.0f;
    int errors = 0;

    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C_hardware[i] - h_C_software[i]);
        max_error = std::max(max_error, error);

        if (error > 0.5f) {  // Tolerance for FP4 quantization
            if (errors < 5) {
                int m = i / N;
                int n = i % N;
                std::cout << "  Mismatch at (" << m << ", " << n << "): "
                          << "HW=" << h_C_hardware[i] << ", "
                          << "SW=" << h_C_software[i] << ", "
                          << "Error=" << error << std::endl;
            }
            errors++;
            passed = false;
        }
    }

    std::cout << "  Max error: " << max_error << std::endl;
    std::cout << "  Total errors: " << errors << " / " << M * N << std::endl;
    std::cout << "  Hardware time: " << hw_time << " ms" << std::endl;
    std::cout << "  Software time: " << sw_time << " ms" << std::endl;
    if (hw_time > 0) {
        std::cout << "  Speedup: " << (sw_time / hw_time) << "x" << std::endl;
    }
    std::cout << "  Result: " << (passed ? "✓ PASSED" : "✗ FAILED") << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_A_scales);
    cudaFree(d_B);
    cudaFree(d_B_scales);
    cudaFree(d_C_hardware);
    cudaFree(d_C_software);
    cudaEventDestroy(hw_start);
    cudaEventDestroy(hw_stop);
    cudaEventDestroy(sw_start);
    cudaEventDestroy(sw_stop);

    return passed;
}

/**
 * @brief Main test runner
 */
int main() {
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║      NVFP4 Hardware Tensor Core GEMM Tests               ║" << std::endl;
    std::cout << "║      GB10 5th-Gen Tensor Cores                            ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;

    bool all_tests_passed = true;

    // Run tests
    all_tests_passed &= test_hardware_vs_software();

    // Summary
    std::cout << "\n╔════════════════════════════════════════════════════════════╗" << std::endl;
    if (all_tests_passed) {
        std::cout << "║              ✅ ALL TESTS PASSED!                         ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
        std::cout << "\n🚀 Hardware tensor core kernel ready for optimization!" << std::endl;
        return 0;
    } else {
        std::cout << "║              ❌ SOME TESTS FAILED                         ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
        return 1;
    }
}
