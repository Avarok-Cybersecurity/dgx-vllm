/*
 * Batched Sparse FP4 GEMV — processes multiple experts in a single launch
 *
 * For MoE inference: instead of one kernel per expert (launch overhead kills
 * performance), dispatch all active experts as a batched GEMV.
 *
 * Grid: dim3(n_blocks, k_blocks, n_experts)
 * Each threadblock processes one K-tile of one expert.
 *
 * Weight layouts (transposed for coalesced N-access, stacked per expert):
 *   Dense:  B_dense_T  [E, K/2, N] uint8   — 2 packed FP4 per byte
 *   Sparse: B_comp_T   [E, K/4, N] uint8   — 2 non-zero FP4 per byte
 *           Meta_T_pk  [E, K/8, N] uint8   — 4-bit packed metadata per group pair
 *
 * Uses cp_async pipelining (same as v4) for bandwidth optimization.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__constant__ float c_fp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ void cp_async_4(void* smem_ptr, const void* glob_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
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
// Batched Dense FP4 GEMV
// =====================================================================
template <int THREADS = 128, int THREAD_N = 4, int TILE_K = 64, int SUB_K = 16>
__global__ void batched_dense_fp4_gemv_kernel(
    const half* __restrict__ A,            // [E_active, K] — one A row per expert
    const uint8_t* __restrict__ B_dense_T, // [E_total, K/2, N]
    half* __restrict__ C,                  // [E_active, K_blocks, N] partials
    const int* __restrict__ expert_ids,    // [E_active] — maps active index → total index
    int N, int K, int E_active
) {
    static_assert(TILE_K % SUB_K == 0);

    constexpr int N_PER_BLOCK = THREADS * THREAD_N;
    constexpr int KP_PER_SUB = SUB_K / 2;
    constexpr int B_STAGE_ELEMS = KP_PER_SUB * N_PER_BLOCK;

    const int expert_active = blockIdx.z;
    if (expert_active >= E_active) return;

    const int expert_total = expert_ids[expert_active];
    const int tid = threadIdx.x;
    const int n_offset = blockIdx.x * N_PER_BLOCK + tid * THREAD_N;
    if (n_offset + THREAD_N > N) return;

    const int k_start = blockIdx.y * TILE_K;
    const int k_end = min(k_start + TILE_K, K);
    const int tile_k = k_end - k_start;
    const int n_subs = tile_k / SUB_K;

    // Expert offsets
    const long b_expert_offset = (long)expert_total * (K / 2) * N;  // B_dense_T stride
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

    // Pipeline startup
    if (n_subs > 0) {
        int base_kp = kp_start;
        #pragma unroll
        for (int s = 0; s < KP_PER_SUB; s++) {
            cp_async_4(&sh_B[0 * B_STAGE_ELEMS + s * N_PER_BLOCK + b_thread_off],
                       &B_dense_T[b_expert_offset + (base_kp + s) * N + n_offset]);
        }
        cp_async_commit();
    }

    for (int sub = 0; sub < n_subs; sub++) {
        int cur_stage = sub & 1;
        int next_stage = (sub + 1) & 1;

        if (sub + 1 < n_subs) {
            int base_kp = kp_start + (sub + 1) * KP_PER_SUB;
            #pragma unroll
            for (int s = 0; s < KP_PER_SUB; s++) {
                cp_async_4(&sh_B[next_stage * B_STAGE_ELEMS + s * N_PER_BLOCK + b_thread_off],
                           &B_dense_T[b_expert_offset + (base_kp + s) * N + n_offset]);
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
            uint32_t packed4 = *reinterpret_cast<uint32_t*>(
                &sh_B[cur_stage * B_STAGE_ELEMS + s * N_PER_BLOCK + b_thread_off]);
            float a0 = sh_A[k_local];
            float a1 = sh_A[k_local + 1];

            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) {
                uint8_t byte = (packed4 >> (j * 8)) & 0xFF;
                acc[j] += sh_lut[byte & 0x0F] * a0 + sh_lut[(byte >> 4) & 0x0F] * a1;
            }
        }
        __syncthreads();
    }

    // Write partials: C[expert_active, blockIdx.y, n_offset]
    long out_base = (long)expert_active * ((K + TILE_K - 1) / TILE_K) * N;
    int out_offset = blockIdx.y * N + n_offset;
    #pragma unroll
    for (int j = 0; j < THREAD_N; j++)
        C[out_base + out_offset + j] = __float2half(acc[j]);
}


// =====================================================================
// Batched Sparse FP4 GEMV
// =====================================================================
template <int THREADS = 128, int THREAD_N = 4, int TILE_K = 64, int SUB_K = 16>
__global__ void batched_sparse_fp4_gemv_kernel(
    const half* __restrict__ A,               // [E_active, K]
    const uint8_t* __restrict__ B_comp_T,     // [E_total, K/4, N]
    const uint8_t* __restrict__ Meta_T_pk,    // [E_total, K/8, N]
    half* __restrict__ C,                     // [E_active, K_blocks, N] partials
    const int* __restrict__ expert_ids,       // [E_active]
    int N, int K, int E_active
) {
    static_assert(TILE_K % SUB_K == 0);
    static_assert(SUB_K % 8 == 0);

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
    const int n_offset = blockIdx.x * N_PER_BLOCK + tid * THREAD_N;
    if (n_offset + THREAD_N > N) return;

    const int k_start = blockIdx.y * TILE_K;
    const int k_end = min(k_start + TILE_K, K);
    const int tile_k = k_end - k_start;
    const int n_subs = tile_k / SUB_K;

    const long b_comp_expert_off = (long)expert_total * (K / 4) * N;
    const long meta_expert_off = (long)expert_total * (K / 8) * N;
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

    // Fetch helper
    auto fetch_sub = [&](int sub_idx, int stage) {
        int base_g = g_start + sub_idx * GROUPS_PER_SUB;
        int base_meta = meta_start + sub_idx * META_ROWS_PER_SUB;
        uint8_t* dst_b = &sh_data[stage * TOTAL_STAGE];
        uint8_t* dst_m = &sh_data[stage * TOTAL_STAGE + B_COMP_STAGE];

        #pragma unroll
        for (int s = 0; s < GROUPS_PER_SUB; s++) {
            cp_async_4(&dst_b[s * N_PER_BLOCK + b_thread_off],
                       &B_comp_T[b_comp_expert_off + (base_g + s) * N + n_offset]);
        }
        #pragma unroll
        for (int s = 0; s < META_ROWS_PER_SUB; s++) {
            cp_async_4(&dst_m[s * N_PER_BLOCK + b_thread_off],
                       &Meta_T_pk[meta_expert_off + (base_meta + s) * N + n_offset]);
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

            uint32_t comp0_4 = *reinterpret_cast<uint32_t*>(
                &cur_b[gi * N_PER_BLOCK + b_thread_off]);
            uint32_t comp1_4 = *reinterpret_cast<uint32_t*>(
                &cur_b[(gi + 1) * N_PER_BLOCK + b_thread_off]);
            uint32_t meta_4 = *reinterpret_cast<uint32_t*>(
                &cur_m[meta_idx * N_PER_BLOCK + b_thread_off]);

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
        __syncthreads();
    }

    long out_base = (long)expert_active * ((K + TILE_K - 1) / TILE_K) * N;
    int out_offset = blockIdx.y * N + n_offset;
    #pragma unroll
    for (int j = 0; j < THREAD_N; j++)
        C[out_base + out_offset + j] = __float2half(acc[j]);
}


// =====================================================================
// Python wrappers
// =====================================================================

constexpr int THREADS = 128;
constexpr int THREAD_N = 4;
constexpr int TILE_K = 64;
constexpr int SUB_K = 16;
constexpr int N_PER_BLOCK = THREADS * THREAD_N;

static int smem_dense() {
    int kp_per_sub = SUB_K / 2;
    int b_stage = kp_per_sub * N_PER_BLOCK;
    return (16 + TILE_K) * sizeof(float) + 2 * b_stage;
}

static int smem_sparse() {
    int groups_per_sub = SUB_K / 4;
    int meta_per_sub = SUB_K / 8;
    int b_comp_stage = groups_per_sub * N_PER_BLOCK;
    int meta_stage = meta_per_sub * N_PER_BLOCK;
    int total_stage = b_comp_stage + meta_stage;
    return (16 + TILE_K) * sizeof(float) + 2 * total_stage;
}

// Batched dense: A[E_active, K], B[E_total, K/2, N], expert_ids[E_active]
// Returns C[E_active, N] (reduced)
torch::Tensor batched_dense_fp4_gemv(
    torch::Tensor A,           // [E_active, K]
    torch::Tensor B_dense_T,   // [E_total, K/2, N]
    torch::Tensor expert_ids   // [E_active] int32
) {
    int E_active = A.size(0);
    int K = A.size(1);
    int N = B_dense_T.size(2);

    int n_blocks = (N + N_PER_BLOCK - 1) / N_PER_BLOCK;
    int k_blocks = (K + TILE_K - 1) / TILE_K;
    dim3 grid(n_blocks, k_blocks, E_active);

    auto C_partials = torch::empty({E_active, k_blocks, N},
        torch::dtype(torch::kFloat16).device(A.device()));

    batched_dense_fp4_gemv_kernel<THREADS, THREAD_N, TILE_K, SUB_K>
        <<<grid, THREADS, smem_dense()>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_dense_T.data_ptr<uint8_t>(),
            reinterpret_cast<half*>(C_partials.data_ptr<at::Half>()),
            expert_ids.data_ptr<int>(),
            N, K, E_active);

    if (k_blocks == 1)
        return C_partials.squeeze(1);  // [E_active, N]
    return C_partials.sum(1);  // [E_active, N]
}

// Batched sparse: A[E_active, K], B_comp[E_total, K/4, N], Meta[E_total, K/8, N]
torch::Tensor batched_sparse_fp4_gemv(
    torch::Tensor A,           // [E_active, K]
    torch::Tensor B_comp_T,    // [E_total, K/4, N]
    torch::Tensor Meta_T_pk,   // [E_total, K/8, N]
    torch::Tensor expert_ids   // [E_active] int32
) {
    int E_active = A.size(0);
    int K = A.size(1);
    int N = B_comp_T.size(2);

    int n_blocks = (N + N_PER_BLOCK - 1) / N_PER_BLOCK;
    int k_blocks = (K + TILE_K - 1) / TILE_K;
    dim3 grid(n_blocks, k_blocks, E_active);

    auto C_partials = torch::empty({E_active, k_blocks, N},
        torch::dtype(torch::kFloat16).device(A.device()));

    batched_sparse_fp4_gemv_kernel<THREADS, THREAD_N, TILE_K, SUB_K>
        <<<grid, THREADS, smem_sparse()>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_comp_T.data_ptr<uint8_t>(),
            Meta_T_pk.data_ptr<uint8_t>(),
            reinterpret_cast<half*>(C_partials.data_ptr<at::Half>()),
            expert_ids.data_ptr<int>(),
            N, K, E_active);

    if (k_blocks == 1)
        return C_partials.squeeze(1);
    return C_partials.sum(1);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_dense_fp4_gemv", &batched_dense_fp4_gemv);
    m.def("batched_sparse_fp4_gemv", &batched_sparse_fp4_gemv);
}
