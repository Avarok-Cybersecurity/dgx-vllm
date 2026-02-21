// SPDX-License-Identifier: Apache-2.0
/**
 * @file nvfp4_gemm_simple_hw.cuh
 * @brief Simplified hardware tensor core kernel for debugging
 *
 * Simpler design that mirrors the working software kernel structure
 * but uses tensor-core-friendly tile sizes.
 */

#pragma once

#include "nvfp4_types.cuh"
#include "nvfp4_gemm_kernel.cuh"  // Reuse load/store functions
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdio>

namespace cutlass_nvfp4 {

/**
 * @brief Simplified hardware config - same as software for now
 */
using SimpleHWConfig = GemmConfigGB10;

/**
 * @brief Compute using tensor-core-aware pattern
 *
 * For now, uses the same software path but with awareness of
 * tensor core constraints. This establishes correctness before
 * adding PTX assembly.
 */
template <typename Config>
__device__ __forceinline__
void compute_tile_product_simple_hw(
    SharedMemoryLayout<Config>& smem,
    float* accumulator,
    float A_scale_global,
    float B_scale_global,
    int tid
) {
    // Same algorithm as software version - validates correctness
    const int total_outputs = Config::TILE_M * Config::TILE_N;
    const int num_threads = Config::THREADS_PER_BLOCK;
    int elements_per_thread = (total_outputs + num_threads - 1) / num_threads;

    for (int elem_local = 0; elem_local < elements_per_thread; ++elem_local) {
        int elem_idx = tid + elem_local * num_threads;
        if (elem_idx >= total_outputs) break;

        int m_idx = elem_idx / Config::TILE_N;
        int n_idx = elem_idx % Config::TILE_N;

        float sum = 0.0f;

        // Process K in blocks of 16 (tensor core size)
        for (int k = 0; k < Config::TILE_K; k += 16) {
            // Get scales for this K block
            int k_scale_idx = k / Config::GROUP_SIZE;
            float A_scale = float(smem.A_scales[m_idx][k_scale_idx]);
            float B_scale = float(smem.B_scales[n_idx][k_scale_idx]);
            float combined_scale = A_scale * B_scale * A_scale_global * B_scale_global;

            // Inner product over 16 elements (tensor core K dimension)
            #pragma unroll
            for (int k_local = 0; k_local < 16 && (k + k_local) < Config::TILE_K; ++k_local) {
                int k_idx = k + k_local;

                nvfp4x2_t A_packed = smem.A_tile[m_idx][k_idx / 2];
                nvfp4x2_t B_packed = smem.B_tile[n_idx][k_idx / 2];

                nvfp4_t A_fp4 = (k_idx % 2 == 0) ? A_packed.lo() : A_packed.hi();
                nvfp4_t B_fp4 = (k_idx % 2 == 0) ? B_packed.lo() : B_packed.hi();

                float A_val = A_fp4.to_float();
                float B_val = B_fp4.to_float();

                sum += A_val * B_val * combined_scale;
            }
        }

        accumulator[elem_local] = sum;
    }
}

/**
 * @brief Simple hardware-aware GEMM kernel
 */
template <typename Config = SimpleHWConfig>
__global__ void nvfp4_gemm_kernel_simple_hw(
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
    int block_m = blockIdx.y;
    int block_n = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ SharedMemoryLayout<Config> smem;

    const int total_outputs = Config::TILE_M * Config::TILE_N;
    const int elements_per_thread = (total_outputs + Config::THREADS_PER_BLOCK - 1) / Config::THREADS_PER_BLOCK;
    float accumulator[elements_per_thread];

    #pragma unroll
    for (int i = 0; i < elements_per_thread; ++i) {
        accumulator[i] = 0.0f;
    }

    // Main loop over K
    for (int k_tile = 0; k_tile < K; k_tile += Config::TILE_K) {
        load_A_tile<Config>(A, A_scales, smem, block_m, k_tile, M, K);
        load_B_tile<Config>(B, B_scales, smem, block_n, k_tile, N, K);

        __syncthreads();

        compute_tile_product_simple_hw<Config>(
            smem, accumulator, A_scale_global, B_scale_global, tid
        );

        __syncthreads();
    }

    store_results<Config>(C, accumulator, block_m, block_n, tid, M, N);
}

/**
 * @brief Host launcher
 */
inline void launch_nvfp4_gemm_simple_hw(
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
    using Config = SimpleHWConfig;

    dim3 grid(
        (N + Config::TILE_N - 1) / Config::TILE_N,
        (M + Config::TILE_M - 1) / Config::TILE_M
    );

    dim3 block(Config::THREADS_PER_BLOCK);

    nvfp4_gemm_kernel_simple_hw<Config><<<grid, block, 0, stream>>>(
        A, A_scales, A_scale_global,
        B, B_scales, B_scale_global,
        C, M, N, K
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

} // namespace cutlass_nvfp4
