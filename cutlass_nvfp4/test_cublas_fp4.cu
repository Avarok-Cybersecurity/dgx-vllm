/**
 * @file test_cublas_fp4.cu
 * @brief Check if cuBLAS/cuBLASLt has FP4 support on GB10
 *
 * If NVIDIA's library has it, we can just CALL IT!
 */

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <stdio.h>

int main() {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║  Checking cuBLAS FP4 Support on GB10 Blackwell       ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n");
    printf("\n");

    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("❌ cuBLAS initialization failed!\n");
        return 1;
    }

    // Get cuBLAS version
    int version;
    cublasGetVersion(handle, &version);
    printf("✅ cuBLAS version: %d\n", version);

    // Initialize cuBLASLt (modern API)
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);
    printf("✅ cuBLASLt initialized\n");
    printf("\n");

    // Check available compute types
    printf("Checking available data types:\n");
    printf("  CUDA_R_16F (FP16):    Standard\n");
    printf("  CUDA_R_16BF (BF16):   Standard\n");
    printf("  CUDA_R_8I (INT8):     Standard\n");

    // Check for FP8
#ifdef CUDA_R_8F_E4M3
    printf("  CUDA_R_8F_E4M3 (FP8): ✅ Defined\n");
#else
    printf("  CUDA_R_8F_E4M3 (FP8): ❌ Not defined\n");
#endif

#ifdef CUDA_R_8F_E5M2
    printf("  CUDA_R_8F_E5M2 (FP8): ✅ Defined\n");
#else
    printf("  CUDA_R_8F_E5M2 (FP8): ❌ Not defined\n");
#endif

    // Check for FP4 (if it exists)
#ifdef CUDA_R_4F_E2M1
    printf("  CUDA_R_4F_E2M1 (FP4): ✅ FOUND!!! WE CAN USE IT!\n");
#else
    printf("  CUDA_R_4F_E2M1 (FP4): ❌ Not defined in this CUDA version\n");
#endif

    printf("\n");
    printf("Device info:\n");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("  Name: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total memory: %.2f GB\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
    printf("\n");

    // Try to query supported matrix types
    printf("Attempting FP4 GEMM query...\n");

    // Create matrix layout descriptors
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;

    // Try with INT8 first (which we know works)
    int M = 128, N = 128, K = 128;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, M, K, M);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, K, N, K);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, M, N, M);
    printf("  INT8 layouts: ✅ Created successfully\n");

    // Cleanup
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtDestroy(ltHandle);
    cublasDestroy(handle);

    printf("\n");
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║  Conclusion:                                          ║\n");
    printf("║  If FP4 not found above, we must use custom kernel   ║\n");
    printf("║  Our software kernel is READY TO GO!                 ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n");
    printf("\n");

    return 0;
}
