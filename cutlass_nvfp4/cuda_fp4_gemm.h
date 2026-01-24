// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda_fp4_gemm.h
 * @brief HIGH-PERFORMANCE FP4 GEMM Library for CUDA 13.1+
 *
 * OFFICIAL-STYLE GEMM API for FP4 matrices!
 *
 * Provides cuBLAS-style API for FP4 matrix multiplication:
 * - C = alpha * (A @ B) + beta * C
 * - Block-scaled quantization support
 * - Optimized for GB10 Blackwell
 * - Drop-in replacement for cuBLAS when FP4 needed
 *
 * Performance:
 * - Current: Optimized software (production-ready)
 * - Future: Hardware tensor cores (20-60x when available)
 */

#ifndef __CUDA_FP4_GEMM_H__
#define __CUDA_FP4_GEMM_H__

#include "cuda_fp4.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ERROR CODES (cuBLAS-style)
// ============================================================================

typedef enum {
    CUDA_FP4_STATUS_SUCCESS = 0,
    CUDA_FP4_STATUS_NOT_INITIALIZED = 1,
    CUDA_FP4_STATUS_ALLOC_FAILED = 3,
    CUDA_FP4_STATUS_INVALID_VALUE = 7,
    CUDA_FP4_STATUS_ARCH_MISMATCH = 8,
    CUDA_FP4_STATUS_EXECUTION_FAILED = 13,
    CUDA_FP4_STATUS_INTERNAL_ERROR = 14,
    CUDA_FP4_STATUS_NOT_SUPPORTED = 15
} cudaFP4Status_t;

// ============================================================================
// HANDLE (cuBLAS-style)
// ============================================================================

typedef struct cudaFP4Handle_st* cudaFP4Handle_t;

/**
 * @brief Create FP4 GEMM handle
 */
cudaFP4Status_t cudaFP4Create(cudaFP4Handle_t* handle);

/**
 * @brief Destroy FP4 GEMM handle
 */
cudaFP4Status_t cudaFP4Destroy(cudaFP4Handle_t handle);

/**
 * @brief Set CUDA stream for operations
 */
cudaFP4Status_t cudaFP4SetStream(cudaFP4Handle_t handle, cudaStream_t stream);

// ============================================================================
// GEMM OPERATIONS
// ============================================================================

/**
 * @brief FP4 GEMM with block-scaled quantization
 *
 * Computes: C[M×N] = A[M×K] @ B[K×N]
 *
 * With hierarchical scaling:
 * - A_scales: [M × (K/group_size)] FP8 block scales
 * - B_scales: [N × (K/group_size)] FP8 block scales
 * - A_scale_global, B_scale_global: FP32 global scales
 *
 * @param handle FP4 handle
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param A Device pointer to A matrix (packed FP4x2)
 * @param A_scales Device pointer to A block scales (FP8)
 * @param A_scale_global Global scale for A (FP32)
 * @param B Device pointer to B matrix (packed FP4x2)
 * @param B_scales Device pointer to B block scales (FP8)
 * @param B_scale_global Global scale for B (FP32)
 * @param C Device pointer to C matrix (FP32 output)
 * @param group_size Quantization group size (default: 16)
 *
 * @return Status code
 */
cudaFP4Status_t cudaFP4GemmEx(
    cudaFP4Handle_t handle,
    int M, int N, int K,
    const cuda_fp4x2_t* A,
    const __nv_fp8_e4m3* A_scales,
    float A_scale_global,
    const cuda_fp4x2_t* B,
    const __nv_fp8_e4m3* B_scales,
    float B_scale_global,
    float* C,
    int group_size
);

/**
 * @brief Simple FP4 GEMM (uniform scaling)
 *
 * Computes: C = alpha * (A @ B)
 * With uniform scaling (no block scales)
 */
cudaFP4Status_t cudaFP4Gemm(
    cudaFP4Handle_t handle,
    int M, int N, int K,
    float alpha,
    const cuda_fp4x2_t* A,
    const cuda_fp4x2_t* B,
    float* C
);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Get version string
 */
const char* cudaFP4GetVersionString();

/**
 * @brief Get version number
 */
int cudaFP4GetVersion();

/**
 * @brief Check if hardware acceleration available
 *
 * @return 1 if tcgen05.mma available, 0 if software fallback
 */
int cudaFP4HasHardwareAcceleration();

/**
 * @brief Get error string from status code
 */
const char* cudaFP4GetErrorString(cudaFP4Status_t status);

// ============================================================================
// CONVERSION UTILITIES
// ============================================================================

/**
 * @brief Convert FP32 matrix to packed FP4x2
 *
 * @param dst Device pointer to output (packed FP4x2)
 * @param src Device pointer to input (FP32)
 * @param rows Number of rows
 * @param cols Number of columns
 * @param stream CUDA stream (or NULL for default)
 */
cudaFP4Status_t cudaFP4ConvertFromFP32(
    cuda_fp4x2_t* dst,
    const float* src,
    int rows,
    int cols,
    cudaStream_t stream
);

/**
 * @brief Convert packed FP4x2 to FP32 matrix
 */
cudaFP4Status_t cudaFP4ConvertToFP32(
    float* dst,
    const cuda_fp4x2_t* src,
    int rows,
    int cols,
    cudaStream_t stream
);

/**
 * @brief Quantize FP32 matrix to FP4 with block scales
 *
 * Computes optimal block scales and quantizes to FP4
 *
 * @param dst_data Device pointer to output FP4 data
 * @param dst_scales Device pointer to output FP8 block scales
 * @param dst_scale_global Host pointer to output FP32 global scale
 * @param src Device pointer to input FP32 data
 * @param rows Number of rows
 * @param cols Number of columns
 * @param group_size Quantization group size
 * @param stream CUDA stream
 */
cudaFP4Status_t cudaFP4Quantize(
    cuda_fp4x2_t* dst_data,
    __nv_fp8_e4m3* dst_scales,
    float* dst_scale_global,
    const float* src,
    int rows,
    int cols,
    int group_size,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // __CUDA_FP4_GEMM_H__
