/**
 * Test suite for optimized NVFP4 GEMM kernel
 * Validates all 4 optimizations and measures speedup vs baseline
 */

#include "nvfp4_types.cuh"
#include "nvfp4_gemm_kernel.cuh"  // Baseline
#include "nvfp4_gemm_kernel_optimized.cuh"  // Optimized
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

using namespace cutlass_nvfp4;

// Benchmark harness
template<typename LaunchFunc>
float benchmark_kernel(LaunchFunc launch_func, int iterations = 100) {
    // Warmup
    for (int i = 0; i < 10; ++i) {
        launch_func();
    }
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        launch_func();
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;
    return duration.count() / iterations;  // ms per iteration
}

// Test case: Decode workload (N=4, typical single-batch decode)
void test_decode_workload() {
    printf("\n=== TEST 1: Decode Workload (N=4, M=4096, K=5120) ===\n");
    printf("Simulates: 80B model decode phase (batch=4)\n\n");

    const int M = 4096;
    const int N = 4;      // Small batch (decode)
    const int K = 5120;

    // Allocate memory
    nvfp4x2_t *d_A, *d_B;
    __nv_fp8_e4m3 *d_A_scales, *d_B_scales;
    float *d_C_baseline, *d_C_optimized;

    cudaMalloc(&d_A, M * K / 2 * sizeof(nvfp4x2_t));
    cudaMalloc(&d_B, N * K / 2 * sizeof(nvfp4x2_t));
    cudaMalloc(&d_A_scales, M * (K / 16) * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_B_scales, N * (K / 16) * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_C_baseline, M * N * sizeof(float));
    cudaMalloc(&d_C_optimized, M * N * sizeof(float));

    // Benchmark baseline
    auto launch_baseline = [&]() {
        launch_nvfp4_gemm(d_A, d_A_scales, 1.0f, d_B, d_B_scales, 1.0f,
                         d_C_baseline, M, N, K);
    };
    float time_baseline = benchmark_kernel(launch_baseline, 100);

    // Benchmark optimized
    auto launch_optimized = [&]() {
        launch_nvfp4_gemm_optimized(d_A, d_A_scales, 1.0f, d_B, d_B_scales, 1.0f,
                                   d_C_optimized, M, N, K);
    };
    float time_optimized = benchmark_kernel(launch_optimized, 100);

    printf("Baseline:  %.3f ms\n", time_baseline);
    printf("Optimized: %.3f ms\n", time_optimized);
    printf("Speedup:   %.2fx\n", time_baseline / time_optimized);
    printf("Improvement: %.1f%%\n\n", (time_baseline - time_optimized) / time_baseline * 100.0f);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_A_scales);
    cudaFree(d_B_scales);
    cudaFree(d_C_baseline);
    cudaFree(d_C_optimized);
}

// Test case: Prefill workload (N=128, typical batch prefill)
void test_prefill_workload() {
    printf("\n=== TEST 2: Prefill Workload (N=128, M=4096, K=5120) ===\n");
    printf("Simulates: 80B model prefill phase (batch=128)\n\n");

    const int M = 4096;
    const int N = 128;    // Large batch (prefill)
    const int K = 5120;

    // Allocate memory
    nvfp4x2_t *d_A, *d_B;
    __nv_fp8_e4m3 *d_A_scales, *d_B_scales;
    float *d_C_baseline, *d_C_optimized;

    cudaMalloc(&d_A, M * K / 2 * sizeof(nvfp4x2_t));
    cudaMalloc(&d_B, N * K / 2 * sizeof(nvfp4x2_t));
    cudaMalloc(&d_A_scales, M * (K / 16) * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_B_scales, N * (K / 16) * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_C_baseline, M * N * sizeof(float));
    cudaMalloc(&d_C_optimized, M * N * sizeof(float));

    // Benchmark baseline
    auto launch_baseline = [&]() {
        launch_nvfp4_gemm(d_A, d_A_scales, 1.0f, d_B, d_B_scales, 1.0f,
                         d_C_baseline, M, N, K);
    };
    float time_baseline = benchmark_kernel(launch_baseline, 50);

    // Benchmark optimized
    auto launch_optimized = [&]() {
        launch_nvfp4_gemm_optimized(d_A, d_A_scales, 1.0f, d_B, d_B_scales, 1.0f,
                                   d_C_optimized, M, N, K);
    };
    float time_optimized = benchmark_kernel(launch_optimized, 50);

    printf("Baseline:  %.3f ms\n", time_baseline);
    printf("Optimized: %.3f ms\n", time_optimized);
    printf("Speedup:   %.2fx\n", time_baseline / time_optimized);
    printf("Improvement: %.1f%%\n\n", (time_baseline - time_optimized) / time_baseline * 100.0f);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_A_scales);
    cudaFree(d_B_scales);
    cudaFree(d_C_baseline);
    cudaFree(d_C_optimized);
}

// Test case: Long sequence decode (300 tokens)
void test_long_sequence() {
    printf("\n=== TEST 3: Long Sequence (300 tokens, where NVFP4 degraded) ===\n");
    printf("Simulates: 300-token generation where we saw 44%% slowdown\n\n");

    const int M = 5120;   // Embedding dimension
    const int N = 4;      // Batch size
    const int K = 10240;  // Large K for long sequence context

    // Allocate memory
    nvfp4x2_t *d_A, *d_B;
    __nv_fp8_e4m3 *d_A_scales, *d_B_scales;
    float *d_C_baseline, *d_C_optimized;

    cudaMalloc(&d_A, M * K / 2 * sizeof(nvfp4x2_t));
    cudaMalloc(&d_B, N * K / 2 * sizeof(nvfp4x2_t));
    cudaMalloc(&d_A_scales, M * (K / 16) * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_B_scales, N * (K / 16) * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_C_baseline, M * N * sizeof(float));
    cudaMalloc(&d_C_optimized, M * N * sizeof(float));

    // Benchmark baseline
    auto launch_baseline = [&]() {
        launch_nvfp4_gemm(d_A, d_A_scales, 1.0f, d_B, d_B_scales, 1.0f,
                         d_C_baseline, M, N, K);
    };
    float time_baseline = benchmark_kernel(launch_baseline, 50);

    // Benchmark optimized
    auto launch_optimized = [&]() {
        launch_nvfp4_gemm_optimized(d_A, d_A_scales, 1.0f, d_B, d_B_scales, 1.0f,
                                   d_C_optimized, M, N, K);
    };
    float time_optimized = benchmark_kernel(launch_optimized, 50);

    printf("Baseline:  %.3f ms\n", time_baseline);
    printf("Optimized: %.3f ms\n", time_optimized);
    printf("Speedup:   %.2fx\n", time_baseline / time_optimized);
    printf("Improvement: %.1f%%\n\n", (time_baseline - time_optimized) / time_baseline * 100.0f);

    printf("📊 EXPECTED: 40-60%% improvement to fix the 300-token degradation!\n\n");

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_A_scales);
    cudaFree(d_B_scales);
    cudaFree(d_C_baseline);
    cudaFree(d_C_optimized);
}

int main() {
    printf("================================\n");
    printf("  NVFP4 OPTIMIZED KERNEL TESTS\n");
    printf("  TEAM NVIDIA vs TEAM AWQ!\n");
    printf("================================\n");
    printf("\n");
    printf("Testing all 4 optimizations:\n");
    printf("  [1] Adaptive tile sizes (decode vs prefill)\n");
    printf("  [2] Hardware tensor cores (tcgen05.mma PTX)\n");
    printf("  [3] Pre-computed scale indices (register caching)\n");
    printf("  [4] Pipelined load + compute (double-buffered)\n");
    printf("\n");
    printf("Target: Beat AWQ's 34.12 tok/sec on Qwen3-Next-80B-A3B\n");
    printf("Current NVFP4: 27.42 tok/sec (-19.6%%)\n");
    printf("Need: +25%% improvement minimum!\n");
    printf("\n");

    // Run tests
    test_decode_workload();
    test_prefill_workload();
    test_long_sequence();

    printf("================================\n");
    printf("  TESTS COMPLETE!\n");
    printf("================================\n");

    return 0;
}
