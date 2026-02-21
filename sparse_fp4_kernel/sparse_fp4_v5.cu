/*
 * Sparse FP4 GEMV v5 — 16-byte cp.async.cg loads for DRAM efficiency
 *
 * Key improvement: Uses cp.async.cg.shared.global with 16-byte copies
 * (same as Marlin) for maximum DRAM streaming bandwidth.
 *
 * Thread model: THREADS=32 (1 warp), THREAD_N=16 (16 N-elements per thread)
 * Each warp loads 32 × 16 = 512 bytes per cp_async round = one full K-pair row
 * for N=512. Fully coalesced 128-byte cache line transactions.
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

// 16-byte cp.async with cache-global policy (bypasses L1, coalesces in L2)
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* glob_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(glob_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n");
}

__device__ __forceinline__ void cp_async_wait_one() {
    asm volatile("cp.async.wait_group 1;\n");
}


// =====================================================================
// Batched Dense FP4 GEMV — 16-byte loads
// =====================================================================
template <int THREADS, int TILE_K, int SUB_K>
__global__ void batched_dense_v5_kernel(
    const half* __restrict__ A,            // [E_active, K]
    const uint8_t* __restrict__ B_dense_T, // [E_total, K/2, N]
    half* __restrict__ C,                  // [E_active, K_blocks, N] partials
    const int* __restrict__ expert_ids,
    int N, int K, int E_active
) {
    static_assert(TILE_K % SUB_K == 0);
    constexpr int THREAD_N = 16;  // Fixed: 16 bytes per thread via cp_async_16
    constexpr int N_PER_BLOCK = THREADS * THREAD_N;
    constexpr int KP_PER_SUB = SUB_K / 2;
    constexpr int B_STAGE_BYTES = KP_PER_SUB * N_PER_BLOCK;

    const int expert_active = blockIdx.z;
    if (expert_active >= E_active) return;
    const int expert_total = expert_ids[expert_active];
    const int tid = threadIdx.x;
    const int n_base = blockIdx.x * N_PER_BLOCK + tid * THREAD_N;
    if (n_base + THREAD_N > N) return;

    const int k_start = blockIdx.y * TILE_K;
    const int k_end = min(k_start + TILE_K, K);
    const int tile_k = k_end - k_start;
    const int n_subs = tile_k / SUB_K;

    const long b_expert_off = (long)expert_total * (K / 2) * N;
    const half* A_row = A + expert_active * K;

    extern __shared__ char smem[];
    float* sh_lut = reinterpret_cast<float*>(smem);
    float* sh_A = sh_lut + 16;
    uint8_t* sh_B = reinterpret_cast<uint8_t*>(sh_A + TILE_K);

    if (tid < 16) sh_lut[tid] = c_fp4_lut[tid];
    for (int i = tid; i < tile_k; i += THREADS)
        sh_A[i] = __half2float(A_row[k_start + i]);
    __syncthreads();

    float acc[THREAD_N] = {};
    const int kp_start = k_start / 2;
    const int b_thread_off = tid * THREAD_N;

    // Pipeline startup: fetch sub-tile 0
    if (n_subs > 0) {
        int base_kp = kp_start;
        #pragma unroll
        for (int s = 0; s < KP_PER_SUB; s++) {
            cp_async_16(&sh_B[0 * B_STAGE_BYTES + s * N_PER_BLOCK + b_thread_off],
                        &B_dense_T[b_expert_off + (base_kp + s) * N + n_base]);
        }
        cp_async_commit();
    }

    // Main pipeline
    for (int sub = 0; sub < n_subs; sub++) {
        int cur_stage = sub & 1;
        int next_stage = (sub + 1) & 1;

        if (sub + 1 < n_subs) {
            int base_kp = kp_start + (sub + 1) * KP_PER_SUB;
            #pragma unroll
            for (int s = 0; s < KP_PER_SUB; s++) {
                cp_async_16(&sh_B[next_stage * B_STAGE_BYTES + s * N_PER_BLOCK + b_thread_off],
                            &B_dense_T[b_expert_off + (base_kp + s) * N + n_base]);
            }
            cp_async_commit();
        }

        if (sub + 1 < n_subs) cp_async_wait_one();
        else cp_async_wait_all();
        __syncthreads();

        int k_local_base = sub * SUB_K;
        #pragma unroll
        for (int s = 0; s < KP_PER_SUB; s++) {
            int k_local = k_local_base + s * 2;
            // Load 16 bytes from shared memory as uint4
            uint4 data = *reinterpret_cast<uint4*>(
                &sh_B[cur_stage * B_STAGE_BYTES + s * N_PER_BLOCK + b_thread_off]);
            float a0 = sh_A[k_local];
            float a1 = sh_A[k_local + 1];

            // Process 4 uint32 words, each covering 4 N-elements
            uint32_t w[4] = {data.x, data.y, data.z, data.w};
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    uint8_t byte = (w[q] >> (j * 8)) & 0xFF;
                    acc[q * 4 + j] += sh_lut[byte & 0x0F] * a0 + sh_lut[(byte >> 4) & 0x0F] * a1;
                }
            }
        }
        __syncthreads();
    }

    // Write partials
    long out_base = (long)expert_active * ((K + TILE_K - 1) / TILE_K) * N;
    int out_offset = blockIdx.y * N + n_base;
    #pragma unroll
    for (int j = 0; j < THREAD_N; j++)
        C[out_base + out_offset + j] = __float2half(acc[j]);
}


// =====================================================================
// Batched Sparse FP4 GEMV — 16-byte loads
// =====================================================================
template <int THREADS, int TILE_K, int SUB_K>
__global__ void batched_sparse_v5_kernel(
    const half* __restrict__ A,               // [E_active, K]
    const uint8_t* __restrict__ B_comp_T,     // [E_total, K/4, N]
    const uint8_t* __restrict__ Meta_T_pk,    // [E_total, K/8, N]
    half* __restrict__ C,                     // [E_active, K_blocks, N] partials
    const int* __restrict__ expert_ids,
    int N, int K, int E_active
) {
    static_assert(TILE_K % SUB_K == 0);
    static_assert(SUB_K % 8 == 0);
    constexpr int THREAD_N = 16;
    constexpr int N_PER_BLOCK = THREADS * THREAD_N;
    constexpr int GROUPS_PER_SUB = SUB_K / 4;
    constexpr int META_ROWS_PER_SUB = SUB_K / 8;
    constexpr int B_COMP_STAGE = GROUPS_PER_SUB * N_PER_BLOCK;
    constexpr int META_STAGE = META_ROWS_PER_SUB * N_PER_BLOCK;
    constexpr int TOTAL_STAGE = B_COMP_STAGE + META_STAGE;

    const int expert_active = blockIdx.z;
    if (expert_active >= E_active) return;
    const int expert_total = expert_ids[expert_active];
    const int tid = threadIdx.x;
    const int n_base = blockIdx.x * N_PER_BLOCK + tid * THREAD_N;
    if (n_base + THREAD_N > N) return;

    const int k_start = blockIdx.y * TILE_K;
    const int k_end = min(k_start + TILE_K, K);
    const int tile_k = k_end - k_start;
    const int n_subs = tile_k / SUB_K;

    const long b_comp_off = (long)expert_total * (K / 4) * N;
    const long meta_off = (long)expert_total * (K / 8) * N;
    const half* A_row = A + expert_active * K;

    extern __shared__ char smem[];
    float* sh_lut = reinterpret_cast<float*>(smem);
    float* sh_A = sh_lut + 16;
    uint8_t* sh_data = reinterpret_cast<uint8_t*>(sh_A + TILE_K);

    if (tid < 16) sh_lut[tid] = c_fp4_lut[tid];
    for (int i = tid; i < tile_k; i += THREADS)
        sh_A[i] = __half2float(A_row[k_start + i]);
    __syncthreads();

    float acc[THREAD_N] = {};
    const int g_start = k_start / 4;
    const int meta_start = k_start / 8;
    const int b_thread_off = tid * THREAD_N;

    auto fetch_sub = [&](int sub_idx, int stage) {
        int base_g = g_start + sub_idx * GROUPS_PER_SUB;
        int base_meta = meta_start + sub_idx * META_ROWS_PER_SUB;
        uint8_t* dst_b = &sh_data[stage * TOTAL_STAGE];
        uint8_t* dst_m = &sh_data[stage * TOTAL_STAGE + B_COMP_STAGE];

        #pragma unroll
        for (int s = 0; s < GROUPS_PER_SUB; s++) {
            cp_async_16(&dst_b[s * N_PER_BLOCK + b_thread_off],
                        &B_comp_T[b_comp_off + (base_g + s) * N + n_base]);
        }
        #pragma unroll
        for (int s = 0; s < META_ROWS_PER_SUB; s++) {
            cp_async_16(&dst_m[s * N_PER_BLOCK + b_thread_off],
                        &Meta_T_pk[meta_off + (base_meta + s) * N + n_base]);
        }
        cp_async_commit();
    };

    if (n_subs > 0) fetch_sub(0, 0);

    for (int sub = 0; sub < n_subs; sub++) {
        int cur_stage = sub & 1;

        if (sub + 1 < n_subs) fetch_sub(sub + 1, (sub + 1) & 1);

        if (sub + 1 < n_subs) cp_async_wait_one();
        else cp_async_wait_all();
        __syncthreads();

        uint8_t* cur_b = &sh_data[cur_stage * TOTAL_STAGE];
        uint8_t* cur_m = &sh_data[cur_stage * TOTAL_STAGE + B_COMP_STAGE];
        int k_local_base = sub * SUB_K;

        #pragma unroll
        for (int gi = 0; gi < GROUPS_PER_SUB; gi += 2) {
            int k_local = k_local_base + gi * 4;
            int meta_idx = gi / 2;

            // Load B_comp data: 16 bytes per group row
            uint4 comp0_data = *reinterpret_cast<uint4*>(
                &cur_b[gi * N_PER_BLOCK + b_thread_off]);
            uint4 comp1_data = *reinterpret_cast<uint4*>(
                &cur_b[(gi + 1) * N_PER_BLOCK + b_thread_off]);
            uint4 meta_data = *reinterpret_cast<uint4*>(
                &cur_m[meta_idx * N_PER_BLOCK + b_thread_off]);

            // Process 4 uint32 words, each covering 4 N-elements
            uint32_t c0[4] = {comp0_data.x, comp0_data.y, comp0_data.z, comp0_data.w};
            uint32_t c1[4] = {comp1_data.x, comp1_data.y, comp1_data.z, comp1_data.w};
            uint32_t mt[4] = {meta_data.x, meta_data.y, meta_data.z, meta_data.w};

            #pragma unroll
            for (int q = 0; q < 4; q++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int idx = q * 4 + j;
                    uint8_t comp0 = (c0[q] >> (j * 8)) & 0xFF;
                    uint8_t comp1 = (c1[q] >> (j * 8)) & 0xFF;
                    uint8_t meta  = (mt[q] >> (j * 8)) & 0xFF;

                    acc[idx] += sh_lut[comp0 & 0x0F] * sh_A[k_local + (meta & 3)];
                    acc[idx] += sh_lut[(comp0 >> 4) & 0x0F] * sh_A[k_local + ((meta >> 2) & 3)];
                    acc[idx] += sh_lut[comp1 & 0x0F] * sh_A[k_local + 4 + ((meta >> 4) & 3)];
                    acc[idx] += sh_lut[(comp1 >> 4) & 0x0F] * sh_A[k_local + 4 + ((meta >> 6) & 3)];
                }
            }
        }
        __syncthreads();
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

// THREADS=32 (1 warp, for N=512 scenarios)
constexpr int T32 = 32;
constexpr int TN = 16;  // always 16 for 16-byte loads
constexpr int TK = 32;
constexpr int SK = 16;
constexpr int NPB_32 = T32 * TN;  // 512

static int smem_dense_v5(int tile_k) {
    int kp_per_sub = SK / 2;
    int b_stage = kp_per_sub * NPB_32;
    return (16 + tile_k) * sizeof(float) + 2 * b_stage;
}

static int smem_sparse_v5(int tile_k) {
    int groups_per_sub = SK / 4;
    int meta_per_sub = SK / 8;
    int total_stage = (groups_per_sub + meta_per_sub) * NPB_32;
    return (16 + tile_k) * sizeof(float) + 2 * total_stage;
}

torch::Tensor batched_dense_v5(
    torch::Tensor A, torch::Tensor B_dense_T, torch::Tensor expert_ids) {
    int E_active = A.size(0);
    int K = A.size(1);
    int N = B_dense_T.size(2);

    int n_blocks = (N + NPB_32 - 1) / NPB_32;
    int k_blocks = (K + TK - 1) / TK;
    dim3 grid(n_blocks, k_blocks, E_active);

    auto C = torch::empty({E_active, k_blocks, N},
        torch::dtype(torch::kFloat16).device(A.device()));

    batched_dense_v5_kernel<T32, TK, SK>
        <<<grid, T32, smem_dense_v5(TK)>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_dense_T.data_ptr<uint8_t>(),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            expert_ids.data_ptr<int>(),
            N, K, E_active);

    return k_blocks == 1 ? C.squeeze(1) : C.sum(1);
}

torch::Tensor batched_sparse_v5(
    torch::Tensor A, torch::Tensor B_comp_T, torch::Tensor Meta_T_pk,
    torch::Tensor expert_ids) {
    int E_active = A.size(0);
    int K = A.size(1);
    int N = B_comp_T.size(2);

    int n_blocks = (N + NPB_32 - 1) / NPB_32;
    int k_blocks = (K + TK - 1) / TK;
    dim3 grid(n_blocks, k_blocks, E_active);

    auto C = torch::empty({E_active, k_blocks, N},
        torch::dtype(torch::kFloat16).device(A.device()));

    batched_sparse_v5_kernel<T32, TK, SK>
        <<<grid, T32, smem_sparse_v5(TK)>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_comp_T.data_ptr<uint8_t>(),
            Meta_T_pk.data_ptr<uint8_t>(),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            expert_ids.data_ptr<int>(),
            N, K, E_active);

    return k_blocks == 1 ? C.squeeze(1) : C.sum(1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_dense_v5", &batched_dense_v5);
    m.def("batched_sparse_v5", &batched_sparse_v5);
}
