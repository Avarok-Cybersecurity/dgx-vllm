// SPDX-License-Identifier: Apache-2.0
/**
 * @file nvfp4_gemm_kernel.cuh
 * @brief NVFP4 Block-Scaled GEMM Kernel for GB10 (SM_121)
 *
 * Implements C = A @ B with:
 * - A, B: NVFP4 (e2m1) matrices, packed 2 values per byte
 * - Block scales: FP8 (per GROUP_SIZE elements)
 * - Global scales: FP32 (per-matrix)
 * - Accumulation: FP32 for numerical accuracy
 * - Tensor cores: GB10 5th-gen tensor cores (16x8x16 for e2m1)
 *
 * Target: NVIDIA GB10 (SM_121, Blackwell architecture)
 */

#pragma once

#include "nvfp4_types.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <cstdio>

namespace cutlass_nvfp4 {

/**
 * @brief GEMM configuration for GB10
 *
 * Optimized for unified LPDDR5X memory (301 GB/s)
 */
struct GemmConfigGB10 {
    // Tile dimensions
    static constexpr int TILE_M = 128;       // M dimension of thread block tile
    static constexpr int TILE_N = 256;       // N dimension (wider for bandwidth)
    static constexpr int TILE_K = 128;       // K dimension (fits in cache)

    // Warp configuration
    static constexpr int WARP_SIZE = 32;
    static constexpr int WARP_COUNT = 8;     // Total warps per block
    static constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARP_COUNT;

    // MMA dimensions (GB10 tensor core for e2m1)
    static constexpr int MMA_M = 16;         // MMA instruction shape
    static constexpr int MMA_N = 8;
    static constexpr int MMA_K = 16;

    // Pipeline stages
    static constexpr int STAGES = 3;

    // Block quantization
    static constexpr int GROUP_SIZE = 16;    // FP4 values per block scale
};

/**
 * @brief Shared memory layout for GEMM kernel
 */
template <typename Config = GemmConfigGB10>
struct SharedMemoryLayout {
    // A tile: [TILE_M][TILE_K/2] packed FP4
    __align__(128) nvfp4x2_t A_tile[Config::TILE_M][Config::TILE_K / 2];

    // B tile: [TILE_N][TILE_K/2] packed FP4
    __align__(128) nvfp4x2_t B_tile[Config::TILE_N][Config::TILE_K / 2];

    // A block scales: [TILE_M][TILE_K/GROUP_SIZE] FP8
    __align__(16) __nv_fp8_e4m3 A_scales[Config::TILE_M][Config::TILE_K / Config::GROUP_SIZE];

    // B block scales: [TILE_N][TILE_K/GROUP_SIZE] FP8
    __align__(16) __nv_fp8_e4m3 B_scales[Config::TILE_N][Config::TILE_K / Config::GROUP_SIZE];
};

/**
 * @brief Load A tile from global memory to shared memory
 *
 * Loads a TILE_M x TILE_K tile of matrix A (packed FP4) and its block scales
 */
template <typename Config>
__device__ __forceinline__
void load_A_tile(
    const nvfp4x2_t* __restrict__ A_global,          // [M, K/2] packed
    const __nv_fp8_e4m3* __restrict__ A_scales_global, // [M, K/GROUP_SIZE] FP8
    SharedMemoryLayout<Config>& smem,
    int block_m,
    int k_offset,
    int M,
    int K
) {
    const int tid = threadIdx.x;
    const int num_threads = Config::THREADS_PER_BLOCK;

    // Number of packed elements to load
    const int packed_K = Config::TILE_K / 2;
    const int total_elements = Config::TILE_M * packed_K;

    // Coalesced loading - each thread loads multiple elements
    for (int idx = tid; idx < total_elements; idx += num_threads) {
        int m = idx / packed_K;
        int k_packed = idx % packed_K;

        int global_m = block_m * Config::TILE_M + m;
        int global_k_packed = k_offset / 2 + k_packed;

        if (global_m < M && global_k_packed < K / 2) {
            smem.A_tile[m][k_packed] = A_global[global_m * (K / 2) + global_k_packed];
        } else {
            smem.A_tile[m][k_packed] = nvfp4x2_t(0);  // Zero padding
        }
    }

    // Load block scales
    const int scale_K = Config::TILE_K / Config::GROUP_SIZE;
    const int total_scales = Config::TILE_M * scale_K;

    for (int idx = tid; idx < total_scales; idx += num_threads) {
        int m = idx / scale_K;
        int k_scale = idx % scale_K;

        int global_m = block_m * Config::TILE_M + m;
        int global_k_scale = k_offset / Config::GROUP_SIZE + k_scale;

        if (global_m < M && global_k_scale < K / Config::GROUP_SIZE) {
            smem.A_scales[m][k_scale] = A_scales_global[global_m * (K / Config::GROUP_SIZE) + global_k_scale];
        } else {
            smem.A_scales[m][k_scale] = __nv_fp8_e4m3(1.0f);  // Neutral scale
        }
    }
}

/**
 * @brief Load B tile from global memory to shared memory
 *
 * Loads a TILE_N x TILE_K tile of matrix B (packed FP4) and its block scales
 */
template <typename Config>
__device__ __forceinline__
void load_B_tile(
    const nvfp4x2_t* __restrict__ B_global,          // [N, K/2] packed
    const __nv_fp8_e4m3* __restrict__ B_scales_global, // [N, K/GROUP_SIZE] FP8
    SharedMemoryLayout<Config>& smem,
    int block_n,
    int k_offset,
    int N,
    int K
) {
    const int tid = threadIdx.x;
    const int num_threads = Config::THREADS_PER_BLOCK;

    // Number of packed elements to load
    const int packed_K = Config::TILE_K / 2;
    const int total_elements = Config::TILE_N * packed_K;

    // Coalesced loading
    for (int idx = tid; idx < total_elements; idx += num_threads) {
        int n = idx / packed_K;
        int k_packed = idx % packed_K;

        int global_n = block_n * Config::TILE_N + n;
        int global_k_packed = k_offset / 2 + k_packed;

        if (global_n < N && global_k_packed < K / 2) {
            smem.B_tile[n][k_packed] = B_global[global_n * (K / 2) + global_k_packed];
        } else {
            smem.B_tile[n][k_packed] = nvfp4x2_t(0);  // Zero padding
        }
    }

    // Load block scales
    const int scale_K = Config::TILE_K / Config::GROUP_SIZE;
    const int total_scales = Config::TILE_N * scale_K;

    for (int idx = tid; idx < total_scales; idx += num_threads) {
        int n = idx / scale_K;
        int k_scale = idx % scale_K;

        int global_n = block_n * Config::TILE_N + n;
        int global_k_scale = k_offset / Config::GROUP_SIZE + k_scale;

        if (global_n < N && global_k_scale < K / Config::GROUP_SIZE) {
            smem.B_scales[n][k_scale] = B_scales_global[global_n * (K / Config::GROUP_SIZE) + global_k_scale];
        } else {
            smem.B_scales[n][k_scale] = __nv_fp8_e4m3(1.0f);  // Neutral scale
        }
    }
}

/**
 * @brief Compute tile product using software emulation
 *
 * NOTE: This is a software implementation for initial testing.
 * Phase 3 will replace this with direct PTX tensor core instructions.
 *
 * For now, we unpack FP4 to float and use standard FP32 math.
 * Each thread computes a subset of the output elements.
 */
template <typename Config>
__device__ __forceinline__
void compute_tile_product_software(
    SharedMemoryLayout<Config>& smem,
    float* accumulator,  // [elements_per_thread] flat array
    float A_scale_global,
    float B_scale_global,
    int tid
) {
    // Simplified approach: Each thread computes a strided subset of output elements
    const int total_outputs = Config::TILE_M * Config::TILE_N;
    const int num_threads = Config::THREADS_PER_BLOCK;

    // Each thread handles multiple output elements
    int elements_per_thread = (total_outputs + num_threads - 1) / num_threads;

    for (int elem_local = 0; elem_local < elements_per_thread; ++elem_local) {
        int elem_idx = tid + elem_local * num_threads;
        if (elem_idx >= total_outputs) break;

        int m_idx = elem_idx / Config::TILE_N;
        int n_idx = elem_idx % Config::TILE_N;

        float sum = 0.0f;

        // Compute dot product for this output element
        for (int k = 0; k < Config::TILE_K; ++k) {
            // Get block scales for this K position
            int k_scale_idx = k / Config::GROUP_SIZE;
            float A_scale = float(smem.A_scales[m_idx][k_scale_idx]);
            float B_scale = float(smem.B_scales[n_idx][k_scale_idx]);
            float combined_scale = A_scale * B_scale * A_scale_global * B_scale_global;

            // Unpack FP4 values
            nvfp4x2_t A_packed = smem.A_tile[m_idx][k / 2];
            nvfp4x2_t B_packed = smem.B_tile[n_idx][k / 2];

            nvfp4_t A_fp4 = (k % 2 == 0) ? A_packed.lo() : A_packed.hi();
            nvfp4_t B_fp4 = (k % 2 == 0) ? B_packed.lo() : B_packed.hi();

            float A_val = A_fp4.to_float();
            float B_val = B_fp4.to_float();

            // Accumulate scaled product
            sum += A_val * B_val * combined_scale;
        }

        accumulator[elem_local] = sum;
    }
}

/**
 * @brief Store results from accumulator to global memory
 */
template <typename Config>
__device__ __forceinline__
void store_results(
    float* __restrict__ C_global,  // [M, N] output
    const float* accumulator,      // Local accumulator [elements_per_thread]
    int block_m,
    int block_n,
    int tid,
    int M,
    int N
) {
    const int total_outputs = Config::TILE_M * Config::TILE_N;
    const int num_threads = Config::THREADS_PER_BLOCK;
    int elements_per_thread = (total_outputs + num_threads - 1) / num_threads;

    // Store results
    for (int elem_local = 0; elem_local < elements_per_thread; ++elem_local) {
        int elem_idx = tid + elem_local * num_threads;
        if (elem_idx >= total_outputs) break;

        int m_local = elem_idx / Config::TILE_N;
        int n_local = elem_idx % Config::TILE_N;

        int m_global = block_m * Config::TILE_M + m_local;
        int n_global = block_n * Config::TILE_N + n_local;

        if (m_global < M && n_global < N) {
            C_global[m_global * N + n_global] = accumulator[elem_local];
        }
    }
}

/**
 * @brief Main GEMM kernel: C = A @ B
 *
 * @param A         Input matrix A [M, K/2] packed FP4
 * @param A_scales  Block scales for A [M, K/GROUP_SIZE] FP8
 * @param A_scale_global  Global scale for A (FP32)
 * @param B         Input matrix B [N, K/2] packed FP4
 * @param B_scales  Block scales for B [N, K/GROUP_SIZE] FP8
 * @param B_scale_global  Global scale for B (FP32)
 * @param C         Output matrix C [M, N] FP32
 * @param M         Number of rows in A
 * @param N         Number of columns in B
 * @param K         Shared dimension (must be multiple of 16)
 */
template <typename Config = GemmConfigGB10>
__global__ void nvfp4_gemm_kernel(
    const nvfp4x2_t* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ A_scales,
    float A_scale_global,
    const nvfp4x2_t* __restrict__ B,
    const __nv_fp8_e4m3* __restrict__ B_scales,
    float B_scale_global,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    // Thread and block indices
    int block_m = blockIdx.y;
    int block_n = blockIdx.x;
    int tid = threadIdx.x;

    // Allocate shared memory
    __shared__ SharedMemoryLayout<Config> smem;

    // Per-thread accumulator (FP32 for accuracy)
    const int total_outputs = Config::TILE_M * Config::TILE_N;
    const int elements_per_thread = (total_outputs + Config::THREADS_PER_BLOCK - 1) / Config::THREADS_PER_BLOCK;
    float accumulator[elements_per_thread];

    #pragma unroll
    for (int i = 0; i < elements_per_thread; ++i) {
        accumulator[i] = 0.0f;
    }

    // Main loop over K dimension
    for (int k_tile = 0; k_tile < K; k_tile += Config::TILE_K) {
        // Load A and B tiles to shared memory
        load_A_tile<Config>(A, A_scales, smem, block_m, k_tile, M, K);
        load_B_tile<Config>(B, B_scales, smem, block_n, k_tile, N, K);

        __syncthreads();

        // Compute tile product
        compute_tile_product_software<Config>(
            smem, accumulator, A_scale_global, B_scale_global, tid
        );

        __syncthreads();
    }

    // Store results to global memory
    store_results<Config>(C, accumulator, block_m, block_n, tid, M, N);
}

/**
 * @brief Host-side launcher for NVFP4 GEMM
 */
inline void launch_nvfp4_gemm(
    const nvfp4x2_t* A,
    const __nv_fp8_e4m3* A_scales,
    float A_scale_global,
    const nvfp4x2_t* B,
    const __nv_fp8_e4m3* B_scales,
    float B_scale_global,
    float* C,
    int M,
    int N,
    int K,
    cudaStream_t stream = 0
) {
    using Config = GemmConfigGB10;

    // Grid dimensions
    dim3 grid(
        (N + Config::TILE_N - 1) / Config::TILE_N,
        (M + Config::TILE_M - 1) / Config::TILE_M
    );

    // Block dimensions
    dim3 block(Config::THREADS_PER_BLOCK);

    // Launch kernel
    nvfp4_gemm_kernel<Config><<<grid, block, 0, stream>>>(
        A, A_scales, A_scale_global,
        B, B_scales, B_scale_global,
        C, M, N, K
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

} // namespace cutlass_nvfp4
