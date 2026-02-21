/*
 * Sparse FP4 GEMV v6 — Direct global loads + interleaved sparse format
 *
 * Key changes from v4:
 * 1. Direct global loads (no cp_async, no shared memory for B)
 *    → Eliminates pipeline overhead, lets hardware schedule loads naturally
 * 2. Interleaved sparse format: B_comp and Meta packed together
 *    → Same DRAM pages, fewer TLB misses, better page locality
 *
 * Thread model: THREADS=128, THREAD_N=4 (same as v4)
 * Shared memory: Only LUT (64B) + A vector (256-1024B) — tiny footprint
 * → Higher occupancy than v4 (which used ~8.5KB shared for B pipeline)
 *
 * Batched: processes multiple experts in a single launch.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__constant__ float c_fp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};


// =====================================================================
// Batched Dense FP4 GEMV — Direct global loads
// =====================================================================
template <int THREADS, int THREAD_N, int TILE_K>
__global__ void batched_dense_v6_kernel(
    const half* __restrict__ A,            // [E_active, K]
    const uint8_t* __restrict__ B_dense_T, // [E_total, K/2, N]
    half* __restrict__ C,                  // [E_active, K_blocks, N] partials
    const int* __restrict__ expert_ids,
    int N, int K, int E_active
) {
    constexpr int N_PER_BLOCK = THREADS * THREAD_N;

    const int expert_active = blockIdx.z;
    if (expert_active >= E_active) return;
    const int expert_total = expert_ids[expert_active];
    const int tid = threadIdx.x;
    const int n_base = blockIdx.x * N_PER_BLOCK + tid * THREAD_N;
    if (n_base + THREAD_N > N) return;

    const int k_start = blockIdx.y * TILE_K;
    const int k_end = min(k_start + TILE_K, K);

    const long b_off = (long)expert_total * (K / 2) * N;
    const half* A_row = A + expert_active * K;

    // Only shared memory for LUT + A vector (tiny: ~320 bytes for TILE_K=64)
    extern __shared__ float sh_data[];
    float* sh_lut = sh_data;
    float* sh_A = sh_data + 16;

    if (tid < 16) sh_lut[tid] = c_fp4_lut[tid];
    for (int i = tid; i < k_end - k_start; i += THREADS)
        sh_A[i] = __half2float(A_row[k_start + i]);
    __syncthreads();

    float acc[THREAD_N] = {};
    const uint8_t* B_ptr = B_dense_T + b_off + n_base;

    // Direct global loads — hardware schedules outstanding requests
    for (int kp = k_start / 2; kp < k_end / 2; kp++) {
        int k_local = (kp - k_start / 2) * 2;
        float a0 = sh_A[k_local];
        float a1 = sh_A[k_local + 1];

        // 4-byte coalesced read: 128 threads × 4 bytes = 512 bytes per kp row
        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(&B_ptr[kp * N]);

        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            uint8_t byte = (packed4 >> (j * 8)) & 0xFF;
            acc[j] += sh_lut[byte & 0x0F] * a0 + sh_lut[(byte >> 4) & 0x0F] * a1;
        }
    }

    // Write partials
    long out_base = (long)expert_active * ((K + TILE_K - 1) / TILE_K) * N;
    int out_offset = blockIdx.y * N + n_base;
    #pragma unroll
    for (int j = 0; j < THREAD_N; j++)
        C[out_base + out_offset + j] = __float2half(acc[j]);
}


// =====================================================================
// Batched Sparse FP4 GEMV — Direct global loads, separate tensors
// =====================================================================
template <int THREADS, int THREAD_N, int TILE_K>
__global__ void batched_sparse_v6_kernel(
    const half* __restrict__ A,               // [E_active, K]
    const uint8_t* __restrict__ B_comp_T,     // [E_total, K/4, N]
    const uint8_t* __restrict__ Meta_T_pk,    // [E_total, K/8, N]
    half* __restrict__ C,                     // [E_active, K_blocks, N] partials
    const int* __restrict__ expert_ids,
    int N, int K, int E_active
) {
    constexpr int N_PER_BLOCK = THREADS * THREAD_N;

    const int expert_active = blockIdx.z;
    if (expert_active >= E_active) return;
    const int expert_total = expert_ids[expert_active];
    const int tid = threadIdx.x;
    const int n_base = blockIdx.x * N_PER_BLOCK + tid * THREAD_N;
    if (n_base + THREAD_N > N) return;

    const int k_start = blockIdx.y * TILE_K;
    const int k_end = min(k_start + TILE_K, K);

    const long b_comp_off = (long)expert_total * (K / 4) * N;
    const long meta_off = (long)expert_total * (K / 8) * N;
    const half* A_row = A + expert_active * K;

    extern __shared__ float sh_data[];
    float* sh_lut = sh_data;
    float* sh_A = sh_data + 16;

    if (tid < 16) sh_lut[tid] = c_fp4_lut[tid];
    for (int i = tid; i < k_end - k_start; i += THREADS)
        sh_A[i] = __half2float(A_row[k_start + i]);
    __syncthreads();

    float acc[THREAD_N] = {};
    const uint8_t* B_ptr = B_comp_T + b_comp_off + n_base;
    const uint8_t* M_ptr = Meta_T_pk + meta_off + n_base;

    // Process 2 groups (8 K-elements) per iteration
    int g_start = k_start / 4;
    int g_end = k_end / 4;
    int m_start = k_start / 8;

    for (int gi = g_start; gi < g_end; gi += 2) {
        int k_local = (gi - g_start) * 4;
        int mi = m_start + (gi - g_start) / 2;

        // 3 coalesced 4-byte reads from global
        uint32_t comp0_4 = *reinterpret_cast<const uint32_t*>(&B_ptr[gi * N]);
        uint32_t comp1_4 = *reinterpret_cast<const uint32_t*>(&B_ptr[(gi + 1) * N]);
        uint32_t meta_4  = *reinterpret_cast<const uint32_t*>(&M_ptr[mi * N]);

        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            uint8_t comp0 = (comp0_4 >> (j * 8)) & 0xFF;
            uint8_t comp1 = (comp1_4 >> (j * 8)) & 0xFF;
            uint8_t meta  = (meta_4  >> (j * 8)) & 0xFF;

            acc[j] += sh_lut[comp0 & 0x0F] * sh_A[k_local + (meta & 3)];
            acc[j] += sh_lut[(comp0 >> 4) & 0x0F] * sh_A[k_local + ((meta >> 2) & 3)];
            acc[j] += sh_lut[comp1 & 0x0F] * sh_A[k_local + 4 + ((meta >> 4) & 3)];
            acc[j] += sh_lut[(comp1 >> 4) & 0x0F] * sh_A[k_local + 4 + ((meta >> 6) & 3)];
        }
    }

    long out_base = (long)expert_active * ((K + TILE_K - 1) / TILE_K) * N;
    int out_offset = blockIdx.y * N + n_base;
    #pragma unroll
    for (int j = 0; j < THREAD_N; j++)
        C[out_base + out_offset + j] = __float2half(acc[j]);
}


// =====================================================================
// Batched Sparse FP4 GEMV — Interleaved format for DRAM locality
//
// B_interleaved_T layout: [E_total, K/8, 3*N] uint8
// For each pair of 2:4 groups (8 K-elements):
//   Row i: [comp_group0[N] | comp_group1[N] | meta_pair[N]]
// All 3 arrays contiguous in memory → same DRAM pages
// =====================================================================
template <int THREADS, int THREAD_N, int TILE_K>
__global__ void batched_sparse_interleaved_v6_kernel(
    const half* __restrict__ A,                  // [E_active, K]
    const uint8_t* __restrict__ B_interleaved_T, // [E_total, K/8, 3*N]
    half* __restrict__ C,                        // [E_active, K_blocks, N] partials
    const int* __restrict__ expert_ids,
    int N, int K, int E_active
) {
    constexpr int N_PER_BLOCK = THREADS * THREAD_N;

    const int expert_active = blockIdx.z;
    if (expert_active >= E_active) return;
    const int expert_total = expert_ids[expert_active];
    const int tid = threadIdx.x;
    const int n_base = blockIdx.x * N_PER_BLOCK + tid * THREAD_N;
    if (n_base + THREAD_N > N) return;

    const int k_start = blockIdx.y * TILE_K;
    const int k_end = min(k_start + TILE_K, K);

    // Interleaved: each "row" is 3*N bytes (comp0, comp1, meta)
    const int row_stride = 3 * N;
    const long b_off = (long)expert_total * (K / 8) * row_stride;
    const half* A_row = A + expert_active * K;

    extern __shared__ float sh_data[];
    float* sh_lut = sh_data;
    float* sh_A = sh_data + 16;

    if (tid < 16) sh_lut[tid] = c_fp4_lut[tid];
    for (int i = tid; i < k_end - k_start; i += THREADS)
        sh_A[i] = __half2float(A_row[k_start + i]);
    __syncthreads();

    float acc[THREAD_N] = {};
    const uint8_t* B_ptr = B_interleaved_T + b_off + n_base;

    int pair_start = k_start / 8;
    int pair_end = k_end / 8;

    for (int pi = pair_start; pi < pair_end; pi++) {
        int k_local = (pi - pair_start) * 8;
        int row_off = pi * row_stride;

        // 3 reads from ADJACENT memory (same DRAM page!)
        uint32_t comp0_4 = *reinterpret_cast<const uint32_t*>(&B_ptr[row_off]);
        uint32_t comp1_4 = *reinterpret_cast<const uint32_t*>(&B_ptr[row_off + N]);
        uint32_t meta_4  = *reinterpret_cast<const uint32_t*>(&B_ptr[row_off + 2 * N]);

        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            uint8_t comp0 = (comp0_4 >> (j * 8)) & 0xFF;
            uint8_t comp1 = (comp1_4 >> (j * 8)) & 0xFF;
            uint8_t meta  = (meta_4  >> (j * 8)) & 0xFF;

            acc[j] += sh_lut[comp0 & 0x0F] * sh_A[k_local + (meta & 3)];
            acc[j] += sh_lut[(comp0 >> 4) & 0x0F] * sh_A[k_local + ((meta >> 2) & 3)];
            acc[j] += sh_lut[comp1 & 0x0F] * sh_A[k_local + 4 + ((meta >> 4) & 3)];
            acc[j] += sh_lut[(comp1 >> 4) & 0x0F] * sh_A[k_local + 4 + ((meta >> 6) & 3)];
        }
    }

    long out_base = (long)expert_active * ((K + TILE_K - 1) / TILE_K) * N;
    int out_offset = blockIdx.y * N + n_base;
    #pragma unroll
    for (int j = 0; j < THREAD_N; j++)
        C[out_base + out_offset + j] = __float2half(acc[j]);
}


// =====================================================================
// Python wrappers
// =====================================================================

constexpr int T = 128;
constexpr int TN = 4;
constexpr int TK = 64;
constexpr int NPB = T * TN;  // 512

static int smem_v6(int tile_k) {
    return (16 + tile_k) * sizeof(float);
}

torch::Tensor batched_dense_v6(
    torch::Tensor A, torch::Tensor B_dense_T, torch::Tensor expert_ids
) {
    int E_active = A.size(0);
    int K = A.size(1);
    int N = B_dense_T.size(2);

    int n_blocks = (N + NPB - 1) / NPB;
    int k_blocks = (K + TK - 1) / TK;
    dim3 grid(n_blocks, k_blocks, E_active);

    auto C = torch::empty({E_active, k_blocks, N},
        torch::dtype(torch::kFloat16).device(A.device()));

    batched_dense_v6_kernel<T, TN, TK>
        <<<grid, T, smem_v6(TK)>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_dense_T.data_ptr<uint8_t>(),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            expert_ids.data_ptr<int>(),
            N, K, E_active);

    return k_blocks == 1 ? C.squeeze(1) : C.sum(1);
}

torch::Tensor batched_sparse_v6(
    torch::Tensor A, torch::Tensor B_comp_T, torch::Tensor Meta_T_pk,
    torch::Tensor expert_ids
) {
    int E_active = A.size(0);
    int K = A.size(1);
    int N = B_comp_T.size(2);

    int n_blocks = (N + NPB - 1) / NPB;
    int k_blocks = (K + TK - 1) / TK;
    dim3 grid(n_blocks, k_blocks, E_active);

    auto C = torch::empty({E_active, k_blocks, N},
        torch::dtype(torch::kFloat16).device(A.device()));

    batched_sparse_v6_kernel<T, TN, TK>
        <<<grid, T, smem_v6(TK)>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_comp_T.data_ptr<uint8_t>(),
            Meta_T_pk.data_ptr<uint8_t>(),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            expert_ids.data_ptr<int>(),
            N, K, E_active);

    return k_blocks == 1 ? C.squeeze(1) : C.sum(1);
}

torch::Tensor batched_sparse_interleaved_v6(
    torch::Tensor A, torch::Tensor B_interleaved_T,
    torch::Tensor expert_ids, int orig_K
) {
    int E_active = A.size(0);
    int K = orig_K;
    int N = B_interleaved_T.size(2) / 3;  // row is 3*N bytes

    int n_blocks = (N + NPB - 1) / NPB;
    int k_blocks = (K + TK - 1) / TK;
    dim3 grid(n_blocks, k_blocks, E_active);

    auto C = torch::empty({E_active, k_blocks, N},
        torch::dtype(torch::kFloat16).device(A.device()));

    batched_sparse_interleaved_v6_kernel<T, TN, TK>
        <<<grid, T, smem_v6(TK)>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_interleaved_T.data_ptr<uint8_t>(),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            expert_ids.data_ptr<int>(),
            N, K, E_active);

    return k_blocks == 1 ? C.squeeze(1) : C.sum(1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_dense_v6", &batched_dense_v6);
    m.def("batched_sparse_v6", &batched_sparse_v6);
    m.def("batched_sparse_interleaved_v6", &batched_sparse_interleaved_v6);
}
