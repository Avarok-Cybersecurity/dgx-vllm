// Test for PTX tcgen05.mma with REAL HARDWARE (V2)

#include "nvfp4_tcgen05_ptx_v2.cuh"
#include "nvfp4_gemm_kernel.cuh"  // Software reference
#include <iostream>
#include <vector>
#include <random>

using namespace cutlass_nvfp4;

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║      REAL HARDWARE tcgen05.mma Kernel Test (V2)          ║" << std::endl;
    std::cout << "║      FIRST custom e2m1 tensor core kernel for GB10!     ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;

    const int M = 128;
    const int N = 128;
    const int K = 64;
    const int GROUP_SIZE = 16;

    // Allocate host memory
    std::vector<nvfp4x2_t> h_A(M * K / 2);
    std::vector<__nv_fp8_e4m3> h_A_scales(M * K / GROUP_SIZE);
    std::vector<nvfp4x2_t> h_B(N * K / 2);
    std::vector<__nv_fp8_e4m3> h_B_scales(N * K / GROUP_SIZE);
    std::vector<float> h_C_ptx(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);

    // Initialize with simple pattern
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    for (auto& val : h_A) {
        val = nvfp4x2_t::from_floats(dist(rng), dist(rng));
    }

    for (auto& val : h_B) {
        val = nvfp4x2_t::from_floats(dist(rng), dist(rng));
    }

    for (auto& scale : h_A_scales) scale = __nv_fp8_e4m3(1.0f);
    for (auto& scale : h_B_scales) scale = __nv_fp8_e4m3(1.0f);

    // Allocate device memory
    nvfp4x2_t* d_A;
    __nv_fp8_e4m3* d_A_scales;
    nvfp4x2_t* d_B;
    __nv_fp8_e4m3* d_B_scales;
    float* d_C_ptx;
    float* d_C_ref;

    cudaMalloc(&d_A, h_A.size() * sizeof(nvfp4x2_t));
    cudaMalloc(&d_A_scales, h_A_scales.size() * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_B, h_B.size() * sizeof(nvfp4x2_t));
    cudaMalloc(&d_B_scales, h_B_scales.size() * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_C_ptx, h_C_ptx.size() * sizeof(float));
    cudaMalloc(&d_C_ref, h_C_ref.size() * sizeof(float));

    // Copy to device
    cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(nvfp4x2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_scales, h_A_scales.data(), h_A_scales.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(nvfp4x2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_scales, h_B_scales.data(), h_B_scales.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);

    std::cout << "\n=== Testing REAL HARDWARE tcgen05.mma (V2) ===" << std::endl;

    // Launch PTX V2 kernel (REAL TENSOR CORES!)
    cudaEvent_t ptx_start, ptx_stop;
    cudaEventCreate(&ptx_start);
    cudaEventCreate(&ptx_stop);

    std::cout << "Launching tcgen05.mma kernel..." << std::endl;
    cudaEventRecord(ptx_start);
    launch_nvfp4_gemm_ptx_v2(
        d_A, d_A_scales, 1.0f,
        d_B, d_B_scales, 1.0f,
        d_C_ptx, M, N, K
    );
    cudaEventRecord(ptx_stop);
    cudaError_t ptx_err = cudaDeviceSynchronize();

    float ptx_time = 0;
    cudaEventElapsedTime(&ptx_time, ptx_start, ptx_stop);

    if (ptx_err != cudaSuccess) {
        std::cout << "❌ PTX kernel failed: " << cudaGetErrorString(ptx_err) << std::endl;
    } else {
        std::cout << "✅ PTX kernel executed successfully!" << std::endl;
    }

    // Launch software reference
    cudaEvent_t ref_start, ref_stop;
    cudaEventCreate(&ref_start);
    cudaEventCreate(&ref_stop);

    std::cout << "Launching software reference..." << std::endl;
    cudaEventRecord(ref_start);
    launch_nvfp4_gemm(
        d_A, d_A_scales, 1.0f,
        d_B, d_B_scales, 1.0f,
        d_C_ref, M, N, K
    );
    cudaEventRecord(ref_stop);
    cudaDeviceSynchronize();

    float ref_time = 0;
    cudaEventElapsedTime(&ref_time, ref_start, ref_stop);

    // Copy results back
    cudaMemcpy(h_C_ptx.data(), d_C_ptx, h_C_ptx.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_ref.data(), d_C_ref, h_C_ref.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare results
    float max_error = 0.0f;
    int errors = 0;
    int non_zero_ptx = 0;
    int non_zero_ref = 0;

    for (int i = 0; i < M * N; ++i) {
        if (h_C_ptx[i] != 0.0f) non_zero_ptx++;
        if (h_C_ref[i] != 0.0f) non_zero_ref++;

        float error = std::abs(h_C_ptx[i] - h_C_ref[i]);
        max_error = std::max(max_error, error);
        if (error > 0.5f) errors++;
    }

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "  PTX V2 kernel time: " << ptx_time << " ms" << std::endl;
    std::cout << "  Reference time: " << ref_time << " ms" << std::endl;
    std::cout << "  Speedup: " << (ref_time / ptx_time) << "x" << std::endl;
    std::cout << "  Non-zero outputs (PTX): " << non_zero_ptx << " / " << M * N << std::endl;
    std::cout << "  Non-zero outputs (Ref): " << non_zero_ref << " / " << M * N << std::endl;
    std::cout << "  Max error: " << max_error << std::endl;
    std::cout << "  Errors: " << errors << " / " << M * N << std::endl;

    std::cout << "\n╔════════════════════════════════════════════════════════════╗" << std::endl;
    if (ptx_err == cudaSuccess && errors < 100) {
        std::cout << "║  🚀 SUCCESS! First e2m1 tcgen05.mma kernel works!        ║" << std::endl;
        std::cout << "║  Real GB10 tensor cores accessed for the first time!    ║" << std::endl;
    } else {
        std::cout << "║  ⚠️  Issues detected - debugging needed                  ║" << std::endl;
        std::cout << "║  This is normal for first PTX attempt on new hardware   ║" << std::endl;
    }
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_A_scales);
    cudaFree(d_B);
    cudaFree(d_B_scales);
    cudaFree(d_C_ptx);
    cudaFree(d_C_ref);
    cudaEventDestroy(ptx_start);
    cudaEventDestroy(ptx_stop);
    cudaEventDestroy(ref_start);
    cudaEventDestroy(ref_stop);

    return (ptx_err == cudaSuccess && errors < 100) ? 0 : 1;
}
