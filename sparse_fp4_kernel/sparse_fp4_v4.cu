/*
 * Sparse FP4 GEMV v4 — cp_async pipelined + high-occupancy
 *
 * Key improvements over v3:
 *   1. cp_async: B weights loaded asynchronously from global → shared memory
 *      using cp.async.ca.shared.global (bypasses register file, enables overlap)
 *   2. 2-stage pipeline: compute on stage[i] while loading stage[i+1]
 *   3. Configurable tile sizes for occupancy tuning
 *   4. All compute reads from shared memory (LUT + A + B)
 *
 * Weight layouts (same as v3, transposed for coalesced N-access):
 *   Dense:  B_dense_T  [K/2, N] uint8   — 2 packed FP4 per byte
 *   Sparse: B_comp_T   [K/4, N] uint8   — 2 non-zero FP4 per byte
 *           Meta_T_pk  [K/8, N] uint8   — 4-bit packed metadata per group pair
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// FP4 E2M1 lookup table
__constant__ float c_fp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ── cp_async intrinsics ──────────────────────────────────────────────
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

__device__ __forceinline__ void cp_async_wait_group(int n) {
    if (n == 0) {
        asm volatile("cp.async.wait_group 0;\n");
    } else {
        asm volatile("cp.async.wait_group 1;\n");
    }
}

// =====================================================================
// Dense FP4 GEMV v4 — cp_async pipelined
// =====================================================================
//
// Pipeline structure (2-stage):
//   - Each TILE_K is divided into SUB_K sub-tiles
//   - Stage alternates between 0 and 1
//   - Fetch stage[s+1] while computing stage[s]
//
// Shared memory layout:
//   [LUT: 16 floats]
//   [A_tile: TILE_K floats]
//   [B_stage0: (SUB_K/2) * N_PER_BLOCK bytes]
//   [B_stage1: (SUB_K/2) * N_PER_BLOCK bytes]
//
template <int THREADS, int THREAD_N, int TILE_K, int SUB_K>
__global__ void dense_fp4_gemv_v4_kernel(
    const half* __restrict__ A,            // [1, K]
    const uint8_t* __restrict__ B_dense_T, // [K/2, N]
    half* __restrict__ C,                  // [K_blocks, N] partials
    int N, int K
) {
    static_assert(TILE_K % SUB_K == 0, "TILE_K must be divisible by SUB_K");
    static_assert(SUB_K % 2 == 0, "SUB_K must be even");

    constexpr int N_PER_BLOCK = THREADS * THREAD_N;
    constexpr int KP_PER_SUB = SUB_K / 2;  // K-pairs per sub-tile
    constexpr int B_STAGE_ELEMS = KP_PER_SUB * N_PER_BLOCK;  // bytes per B stage

    const int tid = threadIdx.x;
    const int n_offset = blockIdx.x * N_PER_BLOCK + tid * THREAD_N;
    if (n_offset + THREAD_N > N) return;

    const int k_start = blockIdx.y * TILE_K;
    const int k_end = min(k_start + TILE_K, K);
    const int tile_k = k_end - k_start;
    const int n_subs = tile_k / SUB_K;  // number of sub-tiles (complete only)

    // ── Shared memory ──
    extern __shared__ char smem[];
    float* sh_lut = reinterpret_cast<float*>(smem);
    float* sh_A = sh_lut + 16;
    uint8_t* sh_B = reinterpret_cast<uint8_t*>(sh_A + TILE_K);

    // Load LUT
    if (tid < 16) sh_lut[tid] = c_fp4_lut[tid];

    // Load A tile to shared memory
    for (int i = tid; i < tile_k; i += THREADS)
        sh_A[i] = __half2float(A[k_start + i]);
    __syncthreads();

    float acc[THREAD_N] = {};

    const int kp_start = k_start / 2;
    const int b_thread_off = tid * THREAD_N;  // thread's offset within N_PER_BLOCK

    // ── Pipeline startup: fetch sub-tile 0 ──
    if (n_subs > 0) {
        int base_kp = kp_start;
        #pragma unroll
        for (int s = 0; s < KP_PER_SUB; s++) {
            cp_async_4(&sh_B[0 * B_STAGE_ELEMS + s * N_PER_BLOCK + b_thread_off],
                       &B_dense_T[(base_kp + s) * N + n_offset]);
        }
        cp_async_commit();
    }

    // ── Main pipeline loop ──
    for (int sub = 0; sub < n_subs; sub++) {
        int cur_stage = sub & 1;
        int next_stage = (sub + 1) & 1;

        // Fetch next sub-tile (if any)
        if (sub + 1 < n_subs) {
            int base_kp = kp_start + (sub + 1) * KP_PER_SUB;
            #pragma unroll
            for (int s = 0; s < KP_PER_SUB; s++) {
                cp_async_4(&sh_B[next_stage * B_STAGE_ELEMS + s * N_PER_BLOCK + b_thread_off],
                           &B_dense_T[(base_kp + s) * N + n_offset]);
            }
            cp_async_commit();
        }

        // Wait for current stage
        if (sub + 1 < n_subs) {
            cp_async_wait_group(1);  // keep 1 group in flight
        } else {
            cp_async_wait_group(0);  // wait for all
        }
        __syncthreads();

        // ── Compute on current stage ──
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

    // Write partials
    int out_offset = blockIdx.y * N + n_offset;
    #pragma unroll
    for (int j = 0; j < THREAD_N; j++)
        C[out_offset + j] = __float2half(acc[j]);
}


// =====================================================================
// Sparse FP4 GEMV v4 — cp_async pipelined
// =====================================================================
//
// Shared memory layout:
//   [LUT: 16 floats]
//   [A_tile: TILE_K floats]
//   [B_comp_stage0: (SUB_K/4) * N_PER_BLOCK bytes]
//   [B_comp_stage1: (SUB_K/4) * N_PER_BLOCK bytes]
//   [Meta_stage0: (SUB_K/8) * N_PER_BLOCK bytes]
//   [Meta_stage1: (SUB_K/8) * N_PER_BLOCK bytes]
//
template <int THREADS, int THREAD_N, int TILE_K, int SUB_K>
__global__ void sparse_fp4_gemv_v4_kernel(
    const half* __restrict__ A,               // [1, K]
    const uint8_t* __restrict__ B_comp_T,     // [K/4, N]
    const uint8_t* __restrict__ Meta_T_pk,    // [K/8, N]
    half* __restrict__ C,                     // [K_blocks, N] partials
    int N, int K
) {
    static_assert(TILE_K % SUB_K == 0, "TILE_K must be divisible by SUB_K");
    static_assert(SUB_K % 8 == 0, "SUB_K must be divisible by 8");

    constexpr int N_PER_BLOCK = THREADS * THREAD_N;
    constexpr int GROUPS_PER_SUB = SUB_K / 4;       // groups of 4 K-elements per sub-tile
    constexpr int META_ROWS_PER_SUB = SUB_K / 8;    // packed meta rows per sub-tile
    constexpr int B_COMP_STAGE = GROUPS_PER_SUB * N_PER_BLOCK;  // bytes per stage
    constexpr int META_STAGE = META_ROWS_PER_SUB * N_PER_BLOCK;
    constexpr int TOTAL_STAGE = B_COMP_STAGE + META_STAGE;

    const int tid = threadIdx.x;
    const int n_offset = blockIdx.x * N_PER_BLOCK + tid * THREAD_N;
    if (n_offset + THREAD_N > N) return;

    const int k_start = blockIdx.y * TILE_K;
    const int k_end = min(k_start + TILE_K, K);
    const int tile_k = k_end - k_start;
    const int n_subs = tile_k / SUB_K;

    // ── Shared memory ──
    extern __shared__ char smem[];
    float* sh_lut = reinterpret_cast<float*>(smem);
    float* sh_A = sh_lut + 16;
    uint8_t* sh_data = reinterpret_cast<uint8_t*>(sh_A + TILE_K);
    // sh_data layout: [stage0: B_comp, Meta] [stage1: B_comp, Meta]

    if (tid < 16) sh_lut[tid] = c_fp4_lut[tid];

    for (int i = tid; i < tile_k; i += THREADS)
        sh_A[i] = __half2float(A[k_start + i]);
    __syncthreads();

    float acc[THREAD_N] = {};

    const int g_start = k_start / 4;     // first group index
    const int meta_start = k_start / 8;  // first packed meta row
    const int b_thread_off = tid * THREAD_N;

    // Helper to get stage base pointers
    auto get_b_comp = [&](int stage) -> uint8_t* {
        return &sh_data[stage * TOTAL_STAGE];
    };
    auto get_meta = [&](int stage) -> uint8_t* {
        return &sh_data[stage * TOTAL_STAGE + B_COMP_STAGE];
    };

    // ── Fetch one sub-tile via cp_async ──
    auto fetch_sub = [&](int sub_idx, int stage) {
        int base_g = g_start + sub_idx * GROUPS_PER_SUB;
        int base_meta = meta_start + sub_idx * META_ROWS_PER_SUB;

        uint8_t* dst_b = get_b_comp(stage);
        uint8_t* dst_m = get_meta(stage);

        // Load compressed B (one row per group)
        #pragma unroll
        for (int s = 0; s < GROUPS_PER_SUB; s++) {
            cp_async_4(&dst_b[s * N_PER_BLOCK + b_thread_off],
                       &B_comp_T[(base_g + s) * N + n_offset]);
        }
        // Load metadata (one row per 2 groups)
        #pragma unroll
        for (int s = 0; s < META_ROWS_PER_SUB; s++) {
            cp_async_4(&dst_m[s * N_PER_BLOCK + b_thread_off],
                       &Meta_T_pk[(base_meta + s) * N + n_offset]);
        }
        cp_async_commit();
    };

    // ── Pipeline startup ──
    if (n_subs > 0) {
        fetch_sub(0, 0);
    }

    // ── Main pipeline loop ──
    for (int sub = 0; sub < n_subs; sub++) {
        int cur_stage = sub & 1;

        // Fetch next sub-tile
        if (sub + 1 < n_subs) {
            fetch_sub(sub + 1, (sub + 1) & 1);
        }

        // Wait for current stage
        if (sub + 1 < n_subs) {
            cp_async_wait_group(1);
        } else {
            cp_async_wait_group(0);
        }
        __syncthreads();

        // ── Compute on current stage ──
        uint8_t* cur_b = get_b_comp(cur_stage);
        uint8_t* cur_m = get_meta(cur_stage);
        int k_local_base = sub * SUB_K;

        // Process groups in pairs (2 groups = 8 K-elements per iteration)
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

    int out_offset = blockIdx.y * N + n_offset;
    #pragma unroll
    for (int j = 0; j < THREAD_N; j++)
        C[out_offset + j] = __float2half(acc[j]);
}


// =====================================================================
// Python wrappers
// =====================================================================

// Configuration constants — tuned via benchmarking
// All configs use THREAD_N=4 (cp_async requires 4-byte alignment)

// Config A: Baseline — same tile as v3 but with cp_async
constexpr int THREADS_A = 128;
constexpr int THREAD_N_A = 4;
constexpr int TILE_K_A = 64;
constexpr int SUB_K_A = 16;
constexpr int N_PER_BLOCK_A = THREADS_A * THREAD_N_A;

// Config B: More blocks — smaller threadblock (256 N/block vs 512)
constexpr int THREADS_B = 64;
constexpr int THREAD_N_B = 4;
constexpr int TILE_K_B = 32;
constexpr int SUB_K_B = 16;
constexpr int N_PER_BLOCK_B = THREADS_B * THREAD_N_B;

// Config C: Deep pipeline — larger tile for more pipeline overlap
constexpr int THREADS_C = 128;
constexpr int THREAD_N_C = 4;
constexpr int TILE_K_C = 128;
constexpr int SUB_K_C = 16;
constexpr int N_PER_BLOCK_C = THREADS_C * THREAD_N_C;


static int smem_dense(int tile_k, int sub_k, int n_per_block, int threads, int thread_n) {
    int kp_per_sub = sub_k / 2;
    int b_stage = kp_per_sub * n_per_block;
    return (16 + tile_k) * sizeof(float) + 2 * b_stage;
}

static int smem_sparse(int tile_k, int sub_k, int n_per_block) {
    int groups_per_sub = sub_k / 4;
    int meta_per_sub = sub_k / 8;
    int b_comp_stage = groups_per_sub * n_per_block;
    int meta_stage = meta_per_sub * n_per_block;
    int total_stage = b_comp_stage + meta_stage;
    return (16 + tile_k) * sizeof(float) + 2 * total_stage;
}


// ── Dense GEMV ──

torch::Tensor dense_fp4_gemv_v4_A(torch::Tensor A, torch::Tensor B_dense_T) {
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1);
    int K = A.size(1);
    int N = B_dense_T.size(1);
    int n_blocks = (N + N_PER_BLOCK_A - 1) / N_PER_BLOCK_A;
    int k_blocks = (K + TILE_K_A - 1) / TILE_K_A;
    int smem = smem_dense(TILE_K_A, SUB_K_A, N_PER_BLOCK_A, THREADS_A, THREAD_N_A);

    auto C = torch::empty({k_blocks, N}, torch::dtype(torch::kFloat16).device(A.device()));
    dense_fp4_gemv_v4_kernel<THREADS_A, THREAD_N_A, TILE_K_A, SUB_K_A>
        <<<dim3(n_blocks, k_blocks), THREADS_A, smem>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_dense_T.data_ptr<uint8_t>(),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            N, K);
    return k_blocks == 1 ? C : C.sum(0, true);
}

torch::Tensor dense_fp4_gemv_v4_B(torch::Tensor A, torch::Tensor B_dense_T) {
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1);
    int K = A.size(1);
    int N = B_dense_T.size(1);
    int n_blocks = (N + N_PER_BLOCK_B - 1) / N_PER_BLOCK_B;
    int k_blocks = (K + TILE_K_B - 1) / TILE_K_B;
    int smem = smem_dense(TILE_K_B, SUB_K_B, N_PER_BLOCK_B, THREADS_B, THREAD_N_B);

    auto C = torch::empty({k_blocks, N}, torch::dtype(torch::kFloat16).device(A.device()));
    dense_fp4_gemv_v4_kernel<THREADS_B, THREAD_N_B, TILE_K_B, SUB_K_B>
        <<<dim3(n_blocks, k_blocks), THREADS_B, smem>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_dense_T.data_ptr<uint8_t>(),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            N, K);
    return k_blocks == 1 ? C : C.sum(0, true);
}

torch::Tensor dense_fp4_gemv_v4_C(torch::Tensor A, torch::Tensor B_dense_T) {
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1);
    int K = A.size(1);
    int N = B_dense_T.size(1);
    int n_blocks = (N + N_PER_BLOCK_C - 1) / N_PER_BLOCK_C;
    int k_blocks = (K + TILE_K_C - 1) / TILE_K_C;
    int smem = smem_dense(TILE_K_C, SUB_K_C, N_PER_BLOCK_C, THREADS_C, THREAD_N_C);

    auto C = torch::empty({k_blocks, N}, torch::dtype(torch::kFloat16).device(A.device()));
    dense_fp4_gemv_v4_kernel<THREADS_C, THREAD_N_C, TILE_K_C, SUB_K_C>
        <<<dim3(n_blocks, k_blocks), THREADS_C, smem>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_dense_T.data_ptr<uint8_t>(),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            N, K);
    return k_blocks == 1 ? C : C.sum(0, true);
}


// ── Sparse GEMV ──

torch::Tensor sparse_fp4_gemv_v4_A(
    torch::Tensor A, torch::Tensor B_comp_T, torch::Tensor Meta_T_pk) {
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1);
    int K = A.size(1);
    int N = B_comp_T.size(1);
    int n_blocks = (N + N_PER_BLOCK_A - 1) / N_PER_BLOCK_A;
    int k_blocks = (K + TILE_K_A - 1) / TILE_K_A;
    int smem = smem_sparse(TILE_K_A, SUB_K_A, N_PER_BLOCK_A);

    auto C = torch::empty({k_blocks, N}, torch::dtype(torch::kFloat16).device(A.device()));
    sparse_fp4_gemv_v4_kernel<THREADS_A, THREAD_N_A, TILE_K_A, SUB_K_A>
        <<<dim3(n_blocks, k_blocks), THREADS_A, smem>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_comp_T.data_ptr<uint8_t>(),
            Meta_T_pk.data_ptr<uint8_t>(),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            N, K);
    return k_blocks == 1 ? C : C.sum(0, true);
}

torch::Tensor sparse_fp4_gemv_v4_B(
    torch::Tensor A, torch::Tensor B_comp_T, torch::Tensor Meta_T_pk) {
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1);
    int K = A.size(1);
    int N = B_comp_T.size(1);
    int n_blocks = (N + N_PER_BLOCK_B - 1) / N_PER_BLOCK_B;
    int k_blocks = (K + TILE_K_B - 1) / TILE_K_B;
    int smem = smem_sparse(TILE_K_B, SUB_K_B, N_PER_BLOCK_B);

    auto C = torch::empty({k_blocks, N}, torch::dtype(torch::kFloat16).device(A.device()));
    sparse_fp4_gemv_v4_kernel<THREADS_B, THREAD_N_B, TILE_K_B, SUB_K_B>
        <<<dim3(n_blocks, k_blocks), THREADS_B, smem>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_comp_T.data_ptr<uint8_t>(),
            Meta_T_pk.data_ptr<uint8_t>(),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            N, K);
    return k_blocks == 1 ? C : C.sum(0, true);
}

torch::Tensor sparse_fp4_gemv_v4_C(
    torch::Tensor A, torch::Tensor B_comp_T, torch::Tensor Meta_T_pk) {
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1);
    int K = A.size(1);
    int N = B_comp_T.size(1);
    int n_blocks = (N + N_PER_BLOCK_C - 1) / N_PER_BLOCK_C;
    int k_blocks = (K + TILE_K_C - 1) / TILE_K_C;
    int smem = smem_sparse(TILE_K_C, SUB_K_C, N_PER_BLOCK_C);

    auto C = torch::empty({k_blocks, N}, torch::dtype(torch::kFloat16).device(A.device()));
    sparse_fp4_gemv_v4_kernel<THREADS_C, THREAD_N_C, TILE_K_C, SUB_K_C>
        <<<dim3(n_blocks, k_blocks), THREADS_C, smem>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_comp_T.data_ptr<uint8_t>(),
            Meta_T_pk.data_ptr<uint8_t>(),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            N, K);
    return k_blocks == 1 ? C : C.sum(0, true);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Config A: THREADS=128, THREAD_N=4, TILE_K=64, SUB_K=16
    m.def("dense_fp4_gemv_v4_A", &dense_fp4_gemv_v4_A);
    m.def("sparse_fp4_gemv_v4_A", &sparse_fp4_gemv_v4_A);
    // Config B: THREADS=128, THREAD_N=2, TILE_K=32, SUB_K=16
    m.def("dense_fp4_gemv_v4_B", &dense_fp4_gemv_v4_B);
    m.def("sparse_fp4_gemv_v4_B", &sparse_fp4_gemv_v4_B);
    // Config C: THREADS=64, THREAD_N=2, TILE_K=16, SUB_K=16
    m.def("dense_fp4_gemv_v4_C", &dense_fp4_gemv_v4_C);
    m.def("sparse_fp4_gemv_v4_C", &sparse_fp4_gemv_v4_C);
}
