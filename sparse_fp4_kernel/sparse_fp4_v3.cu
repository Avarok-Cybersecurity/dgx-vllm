/*
 * Sparse FP4 GEMV/GEMM v3 — Bandwidth-optimized for SM121 (GB10)
 *
 * v3b: Partials-buffer reduction (no atomicAdd)
 *
 * Weight layouts (transposed for coalesced N-access):
 *   Dense:  B_dense_T  [K/2, N] uint8   — 2 packed FP4 per byte
 *   Sparse: B_comp_T   [K/4, N] uint8   — 2 non-zero FP4 per byte
 *           Meta_T_pk  [K/8, N] uint8   — 4-bit packed metadata per group pair
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define TILE_K_GEMV 64
#define TILE_K_GEMM 128

// FP4 E2M1 lookup table
__constant__ float c_fp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// =====================================================================
// Dense FP4 GEMV (M=1) — writes to partials buffer C[blockIdx.y, :]
// =====================================================================
template <int THREADS = 128, int THREAD_N = 4, int TILE_K = TILE_K_GEMV>
__global__ void dense_fp4_gemv_v3_kernel(
    const half* __restrict__ A,            // [1, K]
    const uint8_t* __restrict__ B_dense_T, // [K/2, N]
    half* __restrict__ C,                  // [K_blocks, N] partials (half)
    int N, int K
) {
    const int tid = threadIdx.x;
    const int n_offset = blockIdx.x * THREADS * THREAD_N + tid * THREAD_N;
    if (n_offset + THREAD_N > N) return;

    int k_start = blockIdx.y * TILE_K;
    int k_end = k_start + TILE_K;
    if (k_end > K) k_end = K;

    extern __shared__ char smem[];
    float* sh_lut = reinterpret_cast<float*>(smem);
    float* sh_A = sh_lut + 16;

    if (tid < 16) sh_lut[tid] = c_fp4_lut[tid];

    int tile_k = k_end - k_start;
    for (int i = tid; i < tile_k; i += THREADS)
        sh_A[i] = __half2float(A[k_start + i]);
    __syncthreads();

    float acc[THREAD_N] = {};

    int kp_start = k_start / 2;
    int kp_end = k_end / 2;

    for (int kp = kp_start; kp < kp_end; kp++) {
        int k_local = (kp - kp_start) * 2;
        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(
            &B_dense_T[kp * N + n_offset]);
        float a0 = sh_A[k_local];
        float a1 = sh_A[k_local + 1];

        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            uint8_t byte = (packed4 >> (j * 8)) & 0xFF;
            acc[j] += sh_lut[byte & 0x0F] * a0 + sh_lut[(byte >> 4) & 0x0F] * a1;
        }
    }

    // Write partials (no atomicAdd — each K-block writes to its own row)
    int out_offset = blockIdx.y * N + n_offset;
    #pragma unroll
    for (int j = 0; j < THREAD_N; j++)
        C[out_offset + j] = __float2half(acc[j]);
}


// =====================================================================
// Sparse FP4 GEMV (M=1) — writes to partials buffer
// =====================================================================
template <int THREADS = 128, int THREAD_N = 4, int TILE_K = TILE_K_GEMV>
__global__ void sparse_fp4_gemv_v3_kernel(
    const half* __restrict__ A,               // [1, K]
    const uint8_t* __restrict__ B_comp_T,     // [K/4, N]
    const uint8_t* __restrict__ Meta_T_pk,    // [K/8, N]
    half* __restrict__ C,                     // [K_blocks, N] partials (half)
    int N, int K
) {
    const int tid = threadIdx.x;
    const int n_offset = blockIdx.x * THREADS * THREAD_N + tid * THREAD_N;
    if (n_offset + THREAD_N > N) return;

    int k_start = blockIdx.y * TILE_K;
    int k_end = k_start + TILE_K;
    if (k_end > K) k_end = K;

    extern __shared__ char smem[];
    float* sh_lut = reinterpret_cast<float*>(smem);
    float* sh_A = sh_lut + 16;

    if (tid < 16) sh_lut[tid] = c_fp4_lut[tid];

    int tile_k = k_end - k_start;
    for (int i = tid; i < tile_k; i += THREADS)
        sh_A[i] = __half2float(A[k_start + i]);
    __syncthreads();

    float acc[THREAD_N] = {};

    int g_start = k_start / 4;
    int g_end = k_end / 4;

    for (int g = g_start; g < g_end; g += 2) {
        int k_local = (g - g_start) * 4;
        int meta_row = g / 2;

        uint32_t comp0_4 = *reinterpret_cast<const uint32_t*>(
            &B_comp_T[g * N + n_offset]);
        uint32_t comp1_4 = *reinterpret_cast<const uint32_t*>(
            &B_comp_T[(g + 1) * N + n_offset]);
        uint32_t meta_4 = *reinterpret_cast<const uint32_t*>(
            &Meta_T_pk[meta_row * N + n_offset]);

        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            uint8_t comp0 = (comp0_4 >> (j * 8)) & 0xFF;
            uint8_t comp1 = (comp1_4 >> (j * 8)) & 0xFF;
            uint8_t meta  = (meta_4  >> (j * 8)) & 0xFF;

            // Direct shared memory access (avoids register array branch overhead)
            acc[j] += sh_lut[comp0 & 0x0F] * sh_A[k_local + (meta & 3)];
            acc[j] += sh_lut[(comp0 >> 4) & 0x0F] * sh_A[k_local + ((meta >> 2) & 3)];
            acc[j] += sh_lut[comp1 & 0x0F] * sh_A[k_local + 4 + ((meta >> 4) & 3)];
            acc[j] += sh_lut[(comp1 >> 4) & 0x0F] * sh_A[k_local + 4 + ((meta >> 6) & 3)];
        }
    }

    int out_offset = blockIdx.y * N + n_offset;
    #pragma unroll
    for (int j = 0; j < THREAD_N; j++)
        C[out_offset + j] = __float2half(acc[j]);
}


// =====================================================================
// Dense FP4 GEMM (small M) — writes to partials buffer
// =====================================================================
template <int THREADS = 128, int THREAD_N = 4, int TILE_K = TILE_K_GEMM>
__global__ void dense_fp4_gemm_v3_kernel(
    const half* __restrict__ A,            // [M, K]
    const uint8_t* __restrict__ B_dense_T, // [K/2, N]
    half* __restrict__ C,                  // [K_blocks, M, N] partials
    int M, int N, int K
) {
    const int tid = threadIdx.x;
    const int n_offset = blockIdx.x * THREADS * THREAD_N + tid * THREAD_N;
    if (n_offset + THREAD_N > N) return;

    int k_start = blockIdx.y * TILE_K;
    int k_end = k_start + TILE_K;
    if (k_end > K) k_end = K;
    int tile_k = k_end - k_start;

    extern __shared__ char smem[];
    float* sh_lut = reinterpret_cast<float*>(smem);
    float* sh_A = sh_lut + 16;

    if (tid < 16) sh_lut[tid] = c_fp4_lut[tid];

    for (int m = 0; m < M; m++)
        for (int i = tid; i < tile_k; i += THREADS)
            sh_A[m * tile_k + i] = __half2float(A[m * K + k_start + i]);
    __syncthreads();

    float acc[16][4];
    for (int m = 0; m < M && m < 16; m++)
        for (int j = 0; j < THREAD_N; j++)
            acc[m][j] = 0.0f;

    int kp_start = k_start / 2;
    int kp_end = k_end / 2;

    for (int kp = kp_start; kp < kp_end; kp++) {
        int k_local = (kp - kp_start) * 2;
        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(
            &B_dense_T[kp * N + n_offset]);

        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            uint8_t byte = (packed4 >> (j * 8)) & 0xFF;
            float w0 = sh_lut[byte & 0x0F];
            float w1 = sh_lut[(byte >> 4) & 0x0F];

            for (int m = 0; m < M && m < 16; m++) {
                acc[m][j] += w0 * sh_A[m * tile_k + k_local]
                           + w1 * sh_A[m * tile_k + k_local + 1];
            }
        }
    }

    for (int m = 0; m < M && m < 16; m++) {
        int out_offset = blockIdx.y * M * N + m * N + n_offset;
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++)
            C[out_offset + j] = __float2half(acc[m][j]);
    }
}


// =====================================================================
// Sparse FP4 GEMM (small M)
// =====================================================================
template <int THREADS = 128, int THREAD_N = 4, int TILE_K = TILE_K_GEMM>
__global__ void sparse_fp4_gemm_v3_kernel(
    const half* __restrict__ A,              // [M, K]
    const uint8_t* __restrict__ B_comp_T,    // [K/4, N]
    const uint8_t* __restrict__ Meta_T_pk,   // [K/8, N]
    half* __restrict__ C,                    // [K_blocks, M, N] partials
    int M, int N, int K
) {
    const int tid = threadIdx.x;
    const int n_offset = blockIdx.x * THREADS * THREAD_N + tid * THREAD_N;
    if (n_offset + THREAD_N > N) return;

    int k_start = blockIdx.y * TILE_K;
    int k_end = k_start + TILE_K;
    if (k_end > K) k_end = K;
    int tile_k = k_end - k_start;

    extern __shared__ char smem[];
    float* sh_lut = reinterpret_cast<float*>(smem);
    float* sh_A = sh_lut + 16;

    if (tid < 16) sh_lut[tid] = c_fp4_lut[tid];

    for (int m = 0; m < M; m++)
        for (int i = tid; i < tile_k; i += THREADS)
            sh_A[m * tile_k + i] = __half2float(A[m * K + k_start + i]);
    __syncthreads();

    float acc[16][4];
    for (int m = 0; m < M && m < 16; m++)
        for (int j = 0; j < THREAD_N; j++)
            acc[m][j] = 0.0f;

    int g_start = k_start / 4;
    int g_end = k_end / 4;

    for (int g = g_start; g < g_end; g += 2) {
        int k_local = (g - g_start) * 4;
        int meta_row = g / 2;

        uint32_t comp0_4 = *reinterpret_cast<const uint32_t*>(
            &B_comp_T[g * N + n_offset]);
        uint32_t comp1_4 = *reinterpret_cast<const uint32_t*>(
            &B_comp_T[(g + 1) * N + n_offset]);
        uint32_t meta_4 = *reinterpret_cast<const uint32_t*>(
            &Meta_T_pk[meta_row * N + n_offset]);

        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            uint8_t comp0 = (comp0_4 >> (j * 8)) & 0xFF;
            uint8_t comp1 = (comp1_4 >> (j * 8)) & 0xFF;
            uint8_t meta  = (meta_4  >> (j * 8)) & 0xFF;

            float w0_g0 = sh_lut[comp0 & 0x0F];
            float w1_g0 = sh_lut[(comp0 >> 4) & 0x0F];
            int p0_g0 = meta & 3, p1_g0 = (meta >> 2) & 3;

            float w0_g1 = sh_lut[comp1 & 0x0F];
            float w1_g1 = sh_lut[(comp1 >> 4) & 0x0F];
            int p0_g1 = (meta >> 4) & 3, p1_g1 = (meta >> 6) & 3;

            for (int m = 0; m < M && m < 16; m++) {
                float* a_row = &sh_A[m * tile_k + k_local];
                acc[m][j] += w0_g0 * a_row[p0_g0] + w1_g0 * a_row[p1_g0];
                acc[m][j] += w0_g1 * a_row[4 + p0_g1] + w1_g1 * a_row[4 + p1_g1];
            }
        }
    }

    for (int m = 0; m < M && m < 16; m++) {
        int out_offset = blockIdx.y * M * N + m * N + n_offset;
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++)
            C[out_offset + j] = __float2half(acc[m][j]);
    }
}


// =====================================================================
// Python wrappers — partials buffer + sum reduction
// =====================================================================

constexpr int THREADS = 128;
constexpr int THREAD_N = 4;
constexpr int TILE_N = THREADS * THREAD_N;

torch::Tensor dense_fp4_gemv_v3(torch::Tensor A, torch::Tensor B_dense_T) {
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1);
    int K = A.size(1);
    int N = B_dense_T.size(1);

    int n_blocks = (N + TILE_N - 1) / TILE_N;
    int k_blocks = (K + TILE_K_GEMV - 1) / TILE_K_GEMV;
    dim3 grid(n_blocks, k_blocks);
    int smem_bytes = (16 + TILE_K_GEMV) * sizeof(float);

    auto C_partials = torch::empty({k_blocks, N},
        torch::dtype(torch::kFloat16).device(A.device()));

    dense_fp4_gemv_v3_kernel<THREADS, THREAD_N, TILE_K_GEMV>
        <<<grid, THREADS, smem_bytes>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_dense_T.data_ptr<uint8_t>(),
            reinterpret_cast<half*>(C_partials.data_ptr<at::Half>()),
            N, K);

    // Sum partials along K-block dimension
    if (k_blocks == 1)
        return C_partials;  // already [1, N]
    return C_partials.sum(0, /*keepdim=*/true);  // [1, N]
}

torch::Tensor sparse_fp4_gemv_v3(
    torch::Tensor A, torch::Tensor B_comp_T, torch::Tensor Meta_T_pk) {
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1);
    int K = A.size(1);
    int N = B_comp_T.size(1);
    TORCH_CHECK(K % 8 == 0);

    int n_blocks = (N + TILE_N - 1) / TILE_N;
    int k_blocks = (K + TILE_K_GEMV - 1) / TILE_K_GEMV;
    dim3 grid(n_blocks, k_blocks);
    int smem_bytes = (16 + TILE_K_GEMV) * sizeof(float);

    auto C_partials = torch::empty({k_blocks, N},
        torch::dtype(torch::kFloat16).device(A.device()));

    sparse_fp4_gemv_v3_kernel<THREADS, THREAD_N, TILE_K_GEMV>
        <<<grid, THREADS, smem_bytes>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_comp_T.data_ptr<uint8_t>(),
            Meta_T_pk.data_ptr<uint8_t>(),
            reinterpret_cast<half*>(C_partials.data_ptr<at::Half>()),
            N, K);

    if (k_blocks == 1)
        return C_partials;
    return C_partials.sum(0, /*keepdim=*/true);
}

torch::Tensor dense_fp4_gemm_v3(torch::Tensor A, torch::Tensor B_dense_T) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B_dense_T.size(1);
    TORCH_CHECK(M <= 16);

    int n_blocks = (N + TILE_N - 1) / TILE_N;
    int k_blocks = (K + TILE_K_GEMM - 1) / TILE_K_GEMM;
    dim3 grid(n_blocks, k_blocks);
    int smem_bytes = (16 + M * TILE_K_GEMM) * sizeof(float);

    auto C_partials = torch::empty({k_blocks, M, N},
        torch::dtype(torch::kFloat16).device(A.device()));

    dense_fp4_gemm_v3_kernel<THREADS, THREAD_N, TILE_K_GEMM>
        <<<grid, THREADS, smem_bytes>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_dense_T.data_ptr<uint8_t>(),
            reinterpret_cast<half*>(C_partials.data_ptr<at::Half>()),
            M, N, K);

    if (k_blocks == 1)
        return C_partials.squeeze(0);
    return C_partials.sum(0);
}

torch::Tensor sparse_fp4_gemm_v3(
    torch::Tensor A, torch::Tensor B_comp_T, torch::Tensor Meta_T_pk) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B_comp_T.size(1);
    TORCH_CHECK(M <= 16);
    TORCH_CHECK(K % 8 == 0);

    int n_blocks = (N + TILE_N - 1) / TILE_N;
    int k_blocks = (K + TILE_K_GEMM - 1) / TILE_K_GEMM;
    dim3 grid(n_blocks, k_blocks);
    int smem_bytes = (16 + M * TILE_K_GEMM) * sizeof(float);

    auto C_partials = torch::empty({k_blocks, M, N},
        torch::dtype(torch::kFloat16).device(A.device()));

    sparse_fp4_gemm_v3_kernel<THREADS, THREAD_N, TILE_K_GEMM>
        <<<grid, THREADS, smem_bytes>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_comp_T.data_ptr<uint8_t>(),
            Meta_T_pk.data_ptr<uint8_t>(),
            reinterpret_cast<half*>(C_partials.data_ptr<at::Half>()),
            M, N, K);

    if (k_blocks == 1)
        return C_partials.squeeze(0);
    return C_partials.sum(0);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dense_fp4_gemv_v3", &dense_fp4_gemv_v3);
    m.def("sparse_fp4_gemv_v3", &sparse_fp4_gemv_v3);
    m.def("dense_fp4_gemm_v3", &dense_fp4_gemm_v3);
    m.def("sparse_fp4_gemm_v3", &sparse_fp4_gemm_v3);
}
