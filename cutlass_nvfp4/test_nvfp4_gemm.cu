// SPDX-License-Identifier: Apache-2.0
/**
 * @file test_nvfp4_gemm.cu
 * @brief Unit tests for NVFP4 GEMM kernel
 *
 * Tests block-scaled matrix multiplication with FP4 inputs
 */

#include "nvfp4_gemm_kernel.cuh"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <vector>
#include <random>

using namespace cutlass_nvfp4;

/**
 * @brief CPU reference implementation for validation
 */
void reference_gemm_fp4(
    const nvfp4x2_t* A_packed,
    const __nv_fp8_e4m3* A_scales,
    float A_scale_global,
    const nvfp4x2_t* B_packed,
    const __nv_fp8_e4m3* B_scales,
    float B_scale_global,
    float* C,
    int M,
    int N,
    int K
) {
    constexpr int GROUP_SIZE = 16;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                // Get packed values
                nvfp4x2_t A_pack = A_packed[m * (K / 2) + k / 2];
                nvfp4x2_t B_pack = B_packed[n * (K / 2) + k / 2];

                // Unpack FP4
                nvfp4_t A_fp4 = (k % 2 == 0) ? A_pack.lo() : A_pack.hi();
                nvfp4_t B_fp4 = (k % 2 == 0) ? B_pack.lo() : B_pack.hi();

                // Convert to float
                float A_val = A_fp4.to_float();
                float B_val = B_fp4.to_float();

                // Get block scales
                int k_scale_idx = k / GROUP_SIZE;
                float A_scale = float(A_scales[m * (K / GROUP_SIZE) + k_scale_idx]);
                float B_scale = float(B_scales[n * (K / GROUP_SIZE) + k_scale_idx]);

                // Apply all scales and accumulate
                float scaled = A_val * B_val * A_scale * B_scale * A_scale_global * B_scale_global;
                sum += scaled;
            }

            C[m * N + n] = sum;
        }
    }
}

/**
 * @brief Test small matrix multiplication
 */
bool test_small_gemm() {
    std::cout << "\n=== Testing Small GEMM (32x32x32) ===" << std::endl;

    const int M = 32;
    const int N = 32;
    const int K = 32;
    const int GROUP_SIZE = 16;

    // Allocate host memory
    std::vector<nvfp4x2_t> h_A(M * K / 2);
    std::vector<__nv_fp8_e4m3> h_A_scales(M * K / GROUP_SIZE);
    std::vector<nvfp4x2_t> h_B(N * K / 2);
    std::vector<__nv_fp8_e4m3> h_B_scales(N * K / GROUP_SIZE);
    std::vector<float> h_C_gpu(M * N);
    std::vector<float> h_C_cpu(M * N);

    // Initialize with simple values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; k += 2) {
            float val1 = dist(rng);
            float val2 = dist(rng);
            h_A[m * K / 2 + k / 2] = nvfp4x2_t::from_floats(val1, val2);
        }
    }

    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; k += 2) {
            float val1 = dist(rng);
            float val2 = dist(rng);
            h_B[n * K / 2 + k / 2] = nvfp4x2_t::from_floats(val1, val2);
        }
    }

    // Initialize scales (all 1.0 for simplicity)
    for (auto& scale : h_A_scales) scale = __nv_fp8_e4m3(1.0f);
    for (auto& scale : h_B_scales) scale = __nv_fp8_e4m3(1.0f);

    float A_scale_global = 1.0f;
    float B_scale_global = 1.0f;

    // Allocate device memory
    nvfp4x2_t* d_A;
    __nv_fp8_e4m3* d_A_scales;
    nvfp4x2_t* d_B;
    __nv_fp8_e4m3* d_B_scales;
    float* d_C;

    cudaMalloc(&d_A, h_A.size() * sizeof(nvfp4x2_t));
    cudaMalloc(&d_A_scales, h_A_scales.size() * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_B, h_B.size() * sizeof(nvfp4x2_t));
    cudaMalloc(&d_B_scales, h_B_scales.size() * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_C, h_C_gpu.size() * sizeof(float));

    // Copy to device
    cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(nvfp4x2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_scales, h_A_scales.data(), h_A_scales.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(nvfp4x2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_scales, h_B_scales.data(), h_B_scales.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);

    // Launch GPU kernel
    launch_nvfp4_gemm(
        d_A, d_A_scales, A_scale_global,
        d_B, d_B_scales, B_scale_global,
        d_C, M, N, K
    );

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C_gpu.data(), d_C, h_C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute CPU reference
    reference_gemm_fp4(
        h_A.data(), h_A_scales.data(), A_scale_global,
        h_B.data(), h_B_scales.data(), B_scale_global,
        h_C_cpu.data(), M, N, K
    );

    // Validate results
    bool passed = true;
    float max_error = 0.0f;
    int errors = 0;

    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C_gpu[i] - h_C_cpu[i]);
        max_error = std::max(max_error, error);

        // Allow some tolerance due to FP4 quantization
        if (error > 0.1f) {
            if (errors < 5) {  // Print first 5 errors
                int m = i / N;
                int n = i % N;
                std::cout << "  Mismatch at (" << m << ", " << n << "): "
                          << "GPU=" << h_C_gpu[i] << ", "
                          << "CPU=" << h_C_cpu[i] << ", "
                          << "Error=" << error << std::endl;
            }
            errors++;
            passed = false;
        }
    }

    std::cout << "  Max error: " << max_error << std::endl;
    std::cout << "  Total errors: " << errors << " / " << M * N << std::endl;
    std::cout << "  Result: " << (passed ? "✓ PASSED" : "✗ FAILED") << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_A_scales);
    cudaFree(d_B);
    cudaFree(d_B_scales);
    cudaFree(d_C);

    return passed;
}

/**
 * @brief Test with block scales
 */
bool test_with_block_scales() {
    std::cout << "\n=== Testing GEMM with Block Scales (64x64x64) ===" << std::endl;

    const int M = 64;
    const int N = 64;
    const int K = 64;
    const int GROUP_SIZE = 16;

    // Allocate host memory
    std::vector<nvfp4x2_t> h_A(M * K / 2);
    std::vector<__nv_fp8_e4m3> h_A_scales(M * K / GROUP_SIZE);
    std::vector<nvfp4x2_t> h_B(N * K / 2);
    std::vector<__nv_fp8_e4m3> h_B_scales(N * K / GROUP_SIZE);
    std::vector<float> h_C_gpu(M * N);
    std::vector<float> h_C_cpu(M * N);

    // Initialize with varying values
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    std::uniform_real_distribution<float> scale_dist(0.5f, 2.0f);

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

    // Initialize with varying block scales
    for (auto& scale : h_A_scales) {
        scale = __nv_fp8_e4m3(scale_dist(rng));
    }
    for (auto& scale : h_B_scales) {
        scale = __nv_fp8_e4m3(scale_dist(rng));
    }

    float A_scale_global = 1.25f;
    float B_scale_global = 0.8f;

    // Allocate device memory
    nvfp4x2_t* d_A;
    __nv_fp8_e4m3* d_A_scales;
    nvfp4x2_t* d_B;
    __nv_fp8_e4m3* d_B_scales;
    float* d_C;

    cudaMalloc(&d_A, h_A.size() * sizeof(nvfp4x2_t));
    cudaMalloc(&d_A_scales, h_A_scales.size() * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_B, h_B.size() * sizeof(nvfp4x2_t));
    cudaMalloc(&d_B_scales, h_B_scales.size() * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_C, h_C_gpu.size() * sizeof(float));

    // Copy to device
    cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(nvfp4x2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_scales, h_A_scales.data(), h_A_scales.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(nvfp4x2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_scales, h_B_scales.data(), h_B_scales.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);

    // Launch GPU kernel
    launch_nvfp4_gemm(
        d_A, d_A_scales, A_scale_global,
        d_B, d_B_scales, B_scale_global,
        d_C, M, N, K
    );

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C_gpu.data(), d_C, h_C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute CPU reference
    reference_gemm_fp4(
        h_A.data(), h_A_scales.data(), A_scale_global,
        h_B.data(), h_B_scales.data(), B_scale_global,
        h_C_cpu.data(), M, N, K
    );

    // Validate results
    bool passed = true;
    float max_error = 0.0f;
    int errors = 0;

    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C_gpu[i] - h_C_cpu[i]);
        max_error = std::max(max_error, error);

        if (error > 0.5f) {  // Higher tolerance with scales
            if (errors < 5) {
                int m = i / N;
                int n = i % N;
                std::cout << "  Mismatch at (" << m << ", " << n << "): "
                          << "GPU=" << h_C_gpu[i] << ", "
                          << "CPU=" << h_C_cpu[i] << ", "
                          << "Error=" << error << std::endl;
            }
            errors++;
            passed = false;
        }
    }

    std::cout << "  Max error: " << max_error << std::endl;
    std::cout << "  Total errors: " << errors << " / " << M * N << std::endl;
    std::cout << "  Result: " << (passed ? "✓ PASSED" : "✗ FAILED") << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_A_scales);
    cudaFree(d_B);
    cudaFree(d_B_scales);
    cudaFree(d_C);

    return passed;
}

/**
 * @brief Test with large matrices
 */
bool test_large_gemm() {
    std::cout << "\n=== Testing Large GEMM (256x256x256) ===" << std::endl;

    const int M = 256;
    const int N = 256;
    const int K = 256;
    const int GROUP_SIZE = 16;

    // Allocate host memory
    std::vector<nvfp4x2_t> h_A(M * K / 2);
    std::vector<__nv_fp8_e4m3> h_A_scales(M * K / GROUP_SIZE);
    std::vector<nvfp4x2_t> h_B(N * K / 2);
    std::vector<__nv_fp8_e4m3> h_B_scales(N * K / GROUP_SIZE);
    std::vector<float> h_C_gpu(M * N);

    // Initialize with simple pattern
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; k += 2) {
            float val1 = ((m + k) % 8 == 0) ? 1.0f : 0.25f;
            float val2 = ((m + k + 1) % 8 == 0) ? 1.0f : 0.25f;
            h_A[m * K / 2 + k / 2] = nvfp4x2_t::from_floats(val1, val2);
        }
    }

    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; k += 2) {
            float val1 = ((n + k) % 8 == 0) ? 1.0f : 0.25f;
            float val2 = ((n + k + 1) % 8 == 0) ? 1.0f : 0.25f;
            h_B[n * K / 2 + k / 2] = nvfp4x2_t::from_floats(val1, val2);
        }
    }

    // Simple scales
    for (auto& scale : h_A_scales) scale = __nv_fp8_e4m3(1.0f);
    for (auto& scale : h_B_scales) scale = __nv_fp8_e4m3(1.0f);

    float A_scale_global = 1.0f;
    float B_scale_global = 1.0f;

    // Allocate device memory
    nvfp4x2_t* d_A;
    __nv_fp8_e4m3* d_A_scales;
    nvfp4x2_t* d_B;
    __nv_fp8_e4m3* d_B_scales;
    float* d_C;

    cudaMalloc(&d_A, h_A.size() * sizeof(nvfp4x2_t));
    cudaMalloc(&d_A_scales, h_A_scales.size() * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_B, h_B.size() * sizeof(nvfp4x2_t));
    cudaMalloc(&d_B_scales, h_B_scales.size() * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_C, h_C_gpu.size() * sizeof(float));

    // Copy to device
    cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(nvfp4x2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_scales, h_A_scales.data(), h_A_scales.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(nvfp4x2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_scales, h_B_scales.data(), h_B_scales.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);

    // Launch GPU kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    launch_nvfp4_gemm(
        d_A, d_A_scales, A_scale_global,
        d_B, d_B_scales, B_scale_global,
        d_C, M, N, K
    );
    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back
    cudaMemcpy(h_C_gpu.data(), d_C, h_C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Basic sanity check (just verify it ran without errors)
    bool passed = true;
    for (int i = 0; i < M * N; ++i) {
        if (std::isnan(h_C_gpu[i]) || std::isinf(h_C_gpu[i])) {
            std::cout << "  Found NaN/Inf at index " << i << std::endl;
            passed = false;
            break;
        }
    }

    std::cout << "  Kernel time: " << milliseconds << " ms" << std::endl;
    std::cout << "  Result: " << (passed ? "✓ PASSED (no NaN/Inf)" : "✗ FAILED") << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_A_scales);
    cudaFree(d_B);
    cudaFree(d_B_scales);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return passed;
}

/**
 * @brief Main test runner
 */
int main() {
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║           NVFP4 GEMM Kernel Unit Tests                    ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;

    bool all_tests_passed = true;

    // Run tests
    all_tests_passed &= test_small_gemm();
    all_tests_passed &= test_with_block_scales();
    all_tests_passed &= test_large_gemm();

    // Summary
    std::cout << "\n╔════════════════════════════════════════════════════════════╗" << std::endl;
    if (all_tests_passed) {
        std::cout << "║              ✅ ALL TESTS PASSED!                         ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
        std::cout << "\nNVFP4 GEMM kernel is ready for optimization (Phase 3)." << std::endl;
        return 0;
    } else {
        std::cout << "║              ❌ SOME TESTS FAILED                         ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
        return 1;
    }
}
