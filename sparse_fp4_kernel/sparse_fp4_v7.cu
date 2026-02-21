/*
 * Sparse FP4 GEMV v7 — v6 + FP8 block scale + global scale dequantization
 *
 * Changes from v6:
 * 1. FP8 E4M3FN block scales: one scale per output × 8-K-element group
 *    Tensor layout: [E_total, K/8, N] uint8 (transposed from model's [N, K/8])
 * 2. FP32 global scale: one scalar per expert
 * 3. Dequant: value = fp4_lut[nibble] * fp8_to_float(block_scale) * global_scale
 * 4. FP8→float via 256-entry constant memory LUT (no branching)
 *
 * Bandwidth analysis (sparse with scales, 10 experts, gate_up N=1024 K=2048):
 *   Compressed weights: 10 × 512 KB = 5.0 MB
 *   Metadata:           10 × 256 KB = 2.5 MB
 *   Scales:             10 × 256 KB = 2.5 MB
 *   Total:              10.0 MB (80% of dense-with-scales 12.5 MB)
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

// FP4 E2M1 LUT (16 entries)
__constant__ float c_fp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// FP8 E4M3FN → float32 LUT (256 entries, initialized at module load)
__constant__ float c_fp8_lut[256];

// Host-side FP8 E4M3FN to float conversion for LUT initialization
static float fp8_e4m3fn_to_float_host(uint8_t x) {
    uint32_t sign = (x >> 7) & 1;
    uint32_t exp = (x >> 3) & 0xF;
    uint32_t mant = x & 0x7;

    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;

    float val;
    if (exp == 0) {
        // Subnormal: 2^(-6) * (mant/8)
        val = ldexpf((float)mant / 8.0f, -6);
    } else {
        // Normal: 2^(exp-7) * (1 + mant/8)
        val = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    }
    return sign ? -val : val;
}

// Initialize FP8 LUT on first use
static bool fp8_lut_initialized = false;
static void init_fp8_lut() {
    if (fp8_lut_initialized) return;
    float lut[256];
    for (int i = 0; i < 256; i++)
        lut[i] = fp8_e4m3fn_to_float_host((uint8_t)i);
    cudaMemcpyToSymbol(c_fp8_lut, lut, 256 * sizeof(float));
    fp8_lut_initialized = true;
}


// =====================================================================
// Batched Dense FP4 GEMV with FP8 block scales
// =====================================================================
template <int THREADS, int THREAD_N, int TILE_K>
__global__ void batched_dense_v7_kernel(
    const half* __restrict__ A,            // [E_active, K]
    const uint8_t* __restrict__ B_dense_T, // [E_total, K/2, N]
    const uint8_t* __restrict__ scales_T,  // [E_total, n_scale_groups, N] FP8 E4M3FN
    const float* __restrict__ g_scales,    // [E_total] global scales
    half* __restrict__ C,                  // [E_active, K_blocks, N] partials
    const int* __restrict__ expert_ids,
    int N, int K, int n_scale_groups, int E_active
) {
    constexpr int N_PER_BLOCK = THREADS * THREAD_N;

    const int expert_active = blockIdx.z;
    if (expert_active >= E_active) return;
    const int expert_total = expert_ids[expert_active];
    const int tid = threadIdx.x;
    const int n_base = blockIdx.x * N_PER_BLOCK + tid * THREAD_N;
    const bool valid_n = (n_base + THREAD_N <= N);

    const int k_start = blockIdx.y * TILE_K;
    const int k_end = min(k_start + TILE_K, K);

    const long b_off = (long)expert_total * (K / 2) * N;
    const long s_off = (long)expert_total * (long)n_scale_groups * N;
    const half* A_row = A + expert_active * K;
    const float g_scale = g_scales[expert_total];
    // How many kp iterations (2 K-elements each) per scale group
    const int kp_per_scale = K / (2 * n_scale_groups);

    extern __shared__ float sh_data[];
    float* sh_lut = sh_data;
    float* sh_A = sh_data + 16;

    if (tid < 16) sh_lut[tid] = c_fp4_lut[tid];
    for (int i = tid; i < k_end - k_start; i += THREADS)
        sh_A[i] = __half2float(A_row[k_start + i]);
    __syncthreads();

    if (!valid_n) return;

    float acc[THREAD_N] = {};
    const uint8_t* B_ptr = B_dense_T + b_off + n_base;
    const uint8_t* S_ptr = scales_T + s_off + n_base;

    int kp_start = k_start / 2;
    int kp_end = k_end / 2;

    float cur_scale[THREAD_N];
    int prev_sg = -1;

    for (int kp = kp_start; kp < kp_end; kp++) {
        int k_local = (kp - kp_start) * 2;

        // Load new scale when crossing scale group boundary
        int sg = kp / kp_per_scale;
        if (sg != prev_sg) {
            prev_sg = sg;
            uint32_t s4 = *reinterpret_cast<const uint32_t*>(&S_ptr[sg * N]);
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++)
                cur_scale[j] = c_fp8_lut[(s4 >> (j * 8)) & 0xFF] * g_scale;
        }

        float a0 = sh_A[k_local];
        float a1 = sh_A[k_local + 1];

        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(&B_ptr[kp * N]);

        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            uint8_t byte = (packed4 >> (j * 8)) & 0xFF;
            float v0 = sh_lut[byte & 0x0F] * cur_scale[j];
            float v1 = sh_lut[(byte >> 4) & 0x0F] * cur_scale[j];
            acc[j] += v0 * a0 + v1 * a1;
        }
    }

    long out_base = (long)expert_active * ((K + TILE_K - 1) / TILE_K) * N;
    int out_offset = blockIdx.y * N + n_base;
    #pragma unroll
    for (int j = 0; j < THREAD_N; j++)
        C[out_base + out_offset + j] = __float2half(acc[j]);
}


// =====================================================================
// Batched Sparse FP4 GEMV with FP8 block scales
//
// Scale alignment: each pair (8 original K-elements) = 1 scale group.
// Perfect 1:1 mapping — one scale load per pair iteration.
// =====================================================================
template <int THREADS, int THREAD_N, int TILE_K>
__global__ void batched_sparse_v7_kernel(
    const half* __restrict__ A,               // [E_active, K]
    const uint8_t* __restrict__ B_comp_T,     // [E_total, K/4, N]
    const uint8_t* __restrict__ Meta_T_pk,    // [E_total, K/8, N]
    const uint8_t* __restrict__ scales_T,     // [E_total, n_scale_groups, N] FP8 E4M3FN
    const float* __restrict__ g_scales,       // [E_total] global scales
    half* __restrict__ C,                     // [E_active, K_blocks, N] partials
    const int* __restrict__ expert_ids,
    int N, int K, int n_scale_groups, int E_active
) {
    constexpr int N_PER_BLOCK = THREADS * THREAD_N;

    const int expert_active = blockIdx.z;
    if (expert_active >= E_active) return;
    const int expert_total = expert_ids[expert_active];
    const int tid = threadIdx.x;
    const int n_base = blockIdx.x * N_PER_BLOCK + tid * THREAD_N;
    const bool valid_n = (n_base + THREAD_N <= N);

    const int k_start = blockIdx.y * TILE_K;
    const int k_end = min(k_start + TILE_K, K);

    const long b_comp_off = (long)expert_total * (K / 4) * N;
    const long meta_off = (long)expert_total * (K / 8) * N;
    const long scale_off = (long)expert_total * (long)n_scale_groups * N;
    const half* A_row = A + expert_active * K;
    const float g_scale = g_scales[expert_total];
    // How many pairs (8 K-elements each) per scale group
    const int pairs_per_scale = (K / 8) / n_scale_groups;

    extern __shared__ float sh_data[];
    float* sh_lut = sh_data;
    float* sh_A = sh_data + 16;

    if (tid < 16) sh_lut[tid] = c_fp4_lut[tid];
    for (int i = tid; i < k_end - k_start; i += THREADS)
        sh_A[i] = __half2float(A_row[k_start + i]);
    __syncthreads();

    if (!valid_n) return;

    float acc[THREAD_N] = {};
    const uint8_t* B_ptr = B_comp_T + b_comp_off + n_base;
    const uint8_t* M_ptr = Meta_T_pk + meta_off + n_base;
    const uint8_t* S_ptr = scales_T + scale_off + n_base;

    // Process 2 compressed groups (8 original K-elements) per iteration
    int g_start = k_start / 4;
    int g_end = k_end / 4;
    int m_start = k_start / 8;

    for (int gi = g_start; gi < g_end; gi += 2) {
        int k_local = (gi - g_start) * 4;
        int mi = m_start + (gi - g_start) / 2;
        // Scale index: map pair index to scale group
        int si = (k_start / 8 + (gi - g_start) / 2) / pairs_per_scale;

        // 4 coalesced 4-byte reads: comp0, comp1, meta, scale
        uint32_t comp0_4 = *reinterpret_cast<const uint32_t*>(&B_ptr[gi * N]);
        uint32_t comp1_4 = *reinterpret_cast<const uint32_t*>(&B_ptr[(gi + 1) * N]);
        uint32_t meta_4  = *reinterpret_cast<const uint32_t*>(&M_ptr[mi * N]);
        uint32_t scale_4 = *reinterpret_cast<const uint32_t*>(&S_ptr[si * N]);

        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            uint8_t comp0 = (comp0_4 >> (j * 8)) & 0xFF;
            uint8_t comp1 = (comp1_4 >> (j * 8)) & 0xFF;
            uint8_t meta  = (meta_4  >> (j * 8)) & 0xFF;
            float scale = c_fp8_lut[(scale_4 >> (j * 8)) & 0xFF] * g_scale;

            float sum = sh_lut[comp0 & 0x0F] * sh_A[k_local + (meta & 3)]
                      + sh_lut[(comp0 >> 4) & 0x0F] * sh_A[k_local + ((meta >> 2) & 3)]
                      + sh_lut[comp1 & 0x0F] * sh_A[k_local + 4 + ((meta >> 4) & 3)]
                      + sh_lut[(comp1 >> 4) & 0x0F] * sh_A[k_local + 4 + ((meta >> 6) & 3)];
            acc[j] += sum * scale;
        }
    }

    long out_base = (long)expert_active * ((K + TILE_K - 1) / TILE_K) * N;
    int out_offset = blockIdx.y * N + n_base;
    #pragma unroll
    for (int j = 0; j < THREAD_N; j++)
        C[out_base + out_offset + j] = __float2half(acc[j]);
}


// =====================================================================
// Batched Sparse FP4 GEMV — PACKED format (comp+meta+scale in one tensor)
//
// B_packed_T: [E_total, K/8, 4*N] uint8
//   Row layout: [comp_group0(N) | comp_group1(N) | meta(N) | scale(N)]
//   All 4 reads from adjacent memory → same DRAM page → fewer page misses
// =====================================================================
template <int THREADS, int THREAD_N, int TILE_K>
__global__ void batched_sparse_packed_v7_kernel(
    const half* __restrict__ A,              // [E_active, K]
    const uint8_t* __restrict__ B_packed_T,  // [E_total, K/8, 4*N]
    const float* __restrict__ g_scales,      // [E_total] global scales
    half* __restrict__ C,                    // [E_active, K_blocks, N] partials
    const int* __restrict__ expert_ids,
    int N, int K, int E_active
) {
    constexpr int N_PER_BLOCK = THREADS * THREAD_N;

    const int expert_active = blockIdx.z;
    if (expert_active >= E_active) return;
    const int expert_total = expert_ids[expert_active];
    const int tid = threadIdx.x;
    const int n_base = blockIdx.x * N_PER_BLOCK + tid * THREAD_N;
    const bool valid_n = (n_base + THREAD_N <= N);

    const int k_start = blockIdx.y * TILE_K;
    const int k_end = min(k_start + TILE_K, K);

    const int row_stride = 4 * N;  // comp0 + comp1 + meta + scale
    const long b_off = (long)expert_total * (K / 8) * row_stride;
    const half* A_row = A + expert_active * K;
    const float g_scale = g_scales[expert_total];

    extern __shared__ float sh_data[];
    float* sh_lut = sh_data;
    float* sh_A = sh_data + 16;

    if (tid < 16) sh_lut[tid] = c_fp4_lut[tid];
    for (int i = tid; i < k_end - k_start; i += THREADS)
        sh_A[i] = __half2float(A_row[k_start + i]);
    __syncthreads();

    if (!valid_n) return;

    float acc[THREAD_N] = {};
    const uint8_t* B_ptr = B_packed_T + b_off + n_base;

    int pair_start = k_start / 8;
    int pair_end = k_end / 8;

    for (int pi = pair_start; pi < pair_end; pi++) {
        int k_local = (pi - pair_start) * 8;
        int row_off = pi * row_stride;

        // 4 reads from adjacent memory (same DRAM page for N <= 1024)
        uint32_t comp0_4 = *reinterpret_cast<const uint32_t*>(&B_ptr[row_off]);
        uint32_t comp1_4 = *reinterpret_cast<const uint32_t*>(&B_ptr[row_off + N]);
        uint32_t meta_4  = *reinterpret_cast<const uint32_t*>(&B_ptr[row_off + 2 * N]);
        uint32_t scale_4 = *reinterpret_cast<const uint32_t*>(&B_ptr[row_off + 3 * N]);

        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            uint8_t comp0 = (comp0_4 >> (j * 8)) & 0xFF;
            uint8_t comp1 = (comp1_4 >> (j * 8)) & 0xFF;
            uint8_t meta  = (meta_4  >> (j * 8)) & 0xFF;
            float scale = c_fp8_lut[(scale_4 >> (j * 8)) & 0xFF] * g_scale;

            float sum = sh_lut[comp0 & 0x0F] * sh_A[k_local + (meta & 3)]
                      + sh_lut[(comp0 >> 4) & 0x0F] * sh_A[k_local + ((meta >> 2) & 3)]
                      + sh_lut[comp1 & 0x0F] * sh_A[k_local + 4 + ((meta >> 4) & 3)]
                      + sh_lut[(comp1 >> 4) & 0x0F] * sh_A[k_local + 4 + ((meta >> 6) & 3)];
            acc[j] += sum * scale;
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

static int smem_v7(int tile_k) {
    return (16 + tile_k) * sizeof(float);
}

torch::Tensor batched_dense_v7(
    torch::Tensor A, torch::Tensor B_dense_T,
    torch::Tensor scales_T, torch::Tensor g_scales,
    torch::Tensor expert_ids
) {
    init_fp8_lut();

    int E_active = A.size(0);
    int K = A.size(1);
    int N = B_dense_T.size(2);
    int n_scale_groups = scales_T.size(1);

    int n_blocks = (N + NPB - 1) / NPB;
    int k_blocks = (K + TK - 1) / TK;
    dim3 grid(n_blocks, k_blocks, E_active);

    auto C = torch::empty({E_active, k_blocks, N},
        torch::dtype(torch::kFloat16).device(A.device()));

    auto stream = at::cuda::getCurrentCUDAStream();
    batched_dense_v7_kernel<T, TN, TK>
        <<<grid, T, smem_v7(TK), stream>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_dense_T.data_ptr<uint8_t>(),
            scales_T.data_ptr<uint8_t>(),
            g_scales.data_ptr<float>(),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            expert_ids.data_ptr<int>(),
            N, K, n_scale_groups, E_active);

    return k_blocks == 1 ? C.squeeze(1) : C.sum(1);
}

torch::Tensor batched_sparse_v7(
    torch::Tensor A, torch::Tensor B_comp_T, torch::Tensor Meta_T_pk,
    torch::Tensor scales_T, torch::Tensor g_scales,
    torch::Tensor expert_ids
) {
    init_fp8_lut();

    int E_active = A.size(0);
    int K = A.size(1);
    int N = B_comp_T.size(2);
    int n_scale_groups = scales_T.size(1);

    int n_blocks = (N + NPB - 1) / NPB;
    int k_blocks = (K + TK - 1) / TK;
    dim3 grid(n_blocks, k_blocks, E_active);

    auto C = torch::empty({E_active, k_blocks, N},
        torch::dtype(torch::kFloat16).device(A.device()));

    auto stream = at::cuda::getCurrentCUDAStream();
    batched_sparse_v7_kernel<T, TN, TK>
        <<<grid, T, smem_v7(TK), stream>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_comp_T.data_ptr<uint8_t>(),
            Meta_T_pk.data_ptr<uint8_t>(),
            scales_T.data_ptr<uint8_t>(),
            g_scales.data_ptr<float>(),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            expert_ids.data_ptr<int>(),
            N, K, n_scale_groups, E_active);

    return k_blocks == 1 ? C.squeeze(1) : C.sum(1);
}

torch::Tensor batched_sparse_packed_v7(
    torch::Tensor A, torch::Tensor B_packed_T,
    torch::Tensor g_scales, torch::Tensor expert_ids, int orig_K
) {
    init_fp8_lut();

    int E_active = A.size(0);
    int K = orig_K;
    int N = B_packed_T.size(2) / 4;  // row is 4*N bytes

    int n_blocks = (N + NPB - 1) / NPB;
    int k_blocks = (K + TK - 1) / TK;
    dim3 grid(n_blocks, k_blocks, E_active);

    auto C = torch::empty({E_active, k_blocks, N},
        torch::dtype(torch::kFloat16).device(A.device()));

    auto stream = at::cuda::getCurrentCUDAStream();
    batched_sparse_packed_v7_kernel<T, TN, TK>
        <<<grid, T, smem_v7(TK), stream>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_packed_T.data_ptr<uint8_t>(),
            g_scales.data_ptr<float>(),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            expert_ids.data_ptr<int>(),
            N, K, E_active);

    return k_blocks == 1 ? C.squeeze(1) : C.sum(1);
}

// =====================================================================
// Batched Sparse FP4 GEMV — atomicAdd variant (no k-block reduction)
//
// Writes directly to [E_active, N] output using atomicAdd across k-blocks.
// Output must be zero-initialized before launch.
// =====================================================================
template <int THREADS, int THREAD_N, int TILE_K>
__global__ void batched_sparse_v7_atomic_kernel(
    const half* __restrict__ A,               // [E_active, K]
    const uint8_t* __restrict__ B_comp_T,     // [E_total, K/4, N]
    const uint8_t* __restrict__ Meta_T_pk,    // [E_total, K/8, N]
    const uint8_t* __restrict__ scales_T,     // [E_total, n_scale_groups, N] FP8 E4M3FN
    const float* __restrict__ g_scales,       // [E_total] global scales
    float* __restrict__ C,                    // [E_active, N] float32 accumulator
    const int* __restrict__ expert_ids,
    int N, int K, int n_scale_groups, int E_active
) {
    constexpr int N_PER_BLOCK = THREADS * THREAD_N;

    const int expert_active = blockIdx.z;
    if (expert_active >= E_active) return;
    const int expert_total = expert_ids[expert_active];
    const int tid = threadIdx.x;
    const int n_base = blockIdx.x * N_PER_BLOCK + tid * THREAD_N;
    const bool valid_n = (n_base + THREAD_N <= N);

    const int k_start = blockIdx.y * TILE_K;
    const int k_end = min(k_start + TILE_K, K);

    const long b_comp_off = (long)expert_total * (K / 4) * N;
    const long meta_off = (long)expert_total * (K / 8) * N;
    const long scale_off = (long)expert_total * (long)n_scale_groups * N;
    const half* A_row = A + expert_active * K;
    const float g_scale = g_scales[expert_total];
    const int pairs_per_scale = (K / 8) / n_scale_groups;

    extern __shared__ float sh_data[];
    float* sh_lut = sh_data;
    float* sh_A = sh_data + 16;

    if (tid < 16) sh_lut[tid] = c_fp4_lut[tid];
    for (int i = tid; i < k_end - k_start; i += THREADS)
        sh_A[i] = __half2float(A_row[k_start + i]);
    __syncthreads();

    if (!valid_n) return;

    float acc[THREAD_N] = {};
    const uint8_t* B_ptr = B_comp_T + b_comp_off + n_base;
    const uint8_t* M_ptr = Meta_T_pk + meta_off + n_base;
    const uint8_t* S_ptr = scales_T + scale_off + n_base;

    int g_start = k_start / 4;
    int g_end = k_end / 4;
    int m_start = k_start / 8;

    for (int gi = g_start; gi < g_end; gi += 2) {
        int k_local = (gi - g_start) * 4;
        int mi = m_start + (gi - g_start) / 2;
        int si = (k_start / 8 + (gi - g_start) / 2) / pairs_per_scale;

        uint32_t comp0_4 = *reinterpret_cast<const uint32_t*>(&B_ptr[gi * N]);
        uint32_t comp1_4 = *reinterpret_cast<const uint32_t*>(&B_ptr[(gi + 1) * N]);
        uint32_t meta_4  = *reinterpret_cast<const uint32_t*>(&M_ptr[mi * N]);
        uint32_t scale_4 = *reinterpret_cast<const uint32_t*>(&S_ptr[si * N]);

        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            uint8_t comp0 = (comp0_4 >> (j * 8)) & 0xFF;
            uint8_t comp1 = (comp1_4 >> (j * 8)) & 0xFF;
            uint8_t meta  = (meta_4  >> (j * 8)) & 0xFF;
            float scale = c_fp8_lut[(scale_4 >> (j * 8)) & 0xFF] * g_scale;

            float sum = sh_lut[comp0 & 0x0F] * sh_A[k_local + (meta & 3)]
                      + sh_lut[(comp0 >> 4) & 0x0F] * sh_A[k_local + ((meta >> 2) & 3)]
                      + sh_lut[comp1 & 0x0F] * sh_A[k_local + 4 + ((meta >> 4) & 3)]
                      + sh_lut[(comp1 >> 4) & 0x0F] * sh_A[k_local + 4 + ((meta >> 6) & 3)];
            acc[j] += sum * scale;
        }
    }

    // AtomicAdd to accumulate across k-blocks directly
    long out_base = (long)expert_active * N + n_base;
    #pragma unroll
    for (int j = 0; j < THREAD_N; j++)
        atomicAdd(&C[out_base + j], acc[j]);
}

// FP32 → FP16 conversion kernel (for atomic accumulator → FP16 output)
__global__ void f32_to_f16_kernel(
    const float* __restrict__ input,
    half* __restrict__ output,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    output[idx] = __float2half(input[idx]);
}

// =====================================================================
// SiLU + multiply kernel: out = silu(gate) * up
// gate = input[:, :N], up = input[:, N:]  (input is [M, 2*N])
// =====================================================================
__global__ void silu_mul_kernel(
    const half* __restrict__ input,  // [M, 2*N]
    half* __restrict__ output,       // [M, N]
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int m = idx / N;
    int n = idx % N;
    float gate = __half2float(input[m * 2 * N + n]);
    float up = __half2float(input[m * 2 * N + N + n]);
    float sig = 1.0f / (1.0f + expf(-gate));
    output[idx] = __float2half(gate * sig * up);
}

// =====================================================================
// BF16 → FP16 conversion + token replication kernel
// Input: [M, K] bfloat16, Output: [M*topk, K] float16
// =====================================================================
__global__ void bf16_to_fp16_replicate_kernel(
    const __nv_bfloat16* __restrict__ input,  // [M, K]
    half* __restrict__ output,                // [M*topk, K]
    int M, int K, int topk
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * topk * K;
    if (idx >= total) return;
    int mk = idx / K;          // which (m, topk) pair
    int k = idx % K;           // which K element
    int m = mk / topk;         // original token index
    float val = __bfloat162float(input[m * K + k]);
    output[idx] = __float2half(val);
}

// =====================================================================
// Weighted reduction kernel (FP16 input): output[m] = sum_t(down[m*topk+t] * weight[m][t])
// =====================================================================
__global__ void weighted_reduce_fp16_kernel(
    const half* __restrict__ down,       // [M*topk, K] float16
    const float* __restrict__ weights,   // [M, topk]
    half* __restrict__ output,           // [M, K]
    int M, int K, int topk,
    bool apply_router_weight_on_input    // if true, just sum (no weighting)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;
    int m = idx / K;
    int k = idx % K;
    float acc = 0.0f;
    for (int t = 0; t < topk; t++) {
        float val = __half2float(down[(m * topk + t) * K + k]);
        if (apply_router_weight_on_input) {
            acc += val;
        } else {
            acc += val * weights[m * topk + t];
        }
    }
    output[idx] = __float2half(acc);
}

// =====================================================================
// Fused sparse MoE forward: single C++ call for full MoE layer
//
// Does: BF16→FP16 cast + token replication + GEMM1 + SiLU + GEMM2 +
//       weighted reduction. Eliminates all Python tensor overhead.
// =====================================================================
torch::Tensor fused_sparse_moe_v7(
    torch::Tensor hidden_states,       // [M, K] bfloat16 or float16
    torch::Tensor topk_weights,        // [M, topk] float32
    torch::Tensor topk_ids,            // [M, topk] int32
    torch::Tensor expert_map,          // [global_E] int32 or empty
    // W13 (gate+up) weights
    torch::Tensor w13_comp,            // [E_total, K/4, 2*N] uint8
    torch::Tensor w13_meta,            // [E_total, K/8, 2*N] uint8 (packed)
    torch::Tensor w13_scale,           // [E_total, n_groups, 2*N] uint8
    torch::Tensor w13_g_scales,        // [E_total] float32
    // W2 (down) weights
    torch::Tensor w2_comp,             // [E_total, N/4, K] uint8
    torch::Tensor w2_meta,             // [E_total, N/8, K] uint8 (packed)
    torch::Tensor w2_scale,            // [E_total, n_groups2, K] uint8
    torch::Tensor w2_g_scales,         // [E_total] float32
    int inter_size,                    // N (moe_intermediate_size)
    bool apply_router_weight_on_input
) {
    init_fp8_lut();

    int M = hidden_states.size(0);
    int K = hidden_states.size(1);
    int topk = topk_ids.size(1);
    int N = inter_size;
    int E_active = M * topk;

    auto stream = at::cuda::getCurrentCUDAStream();
    // Ensure all PyTorch ops (sum, index, etc.) use the same stream —
    // critical during CUDA graph capture with max_num_seqs > 1.
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    auto opts_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(hidden_states.device());

    // 1. Map expert IDs
    torch::Tensor mapped_ids;
    if (expert_map.numel() > 0) {
        mapped_ids = expert_map.index({topk_ids}).reshape(-1).to(torch::kInt32);
    } else {
        mapped_ids = topk_ids.reshape(-1).to(torch::kInt32);
    }

    // 2. BF16→FP16 + token replication (fused kernel)
    auto A = torch::empty({E_active, K}, opts_fp16);
    if (hidden_states.scalar_type() == torch::kBFloat16) {
        int total = E_active * K;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        bf16_to_fp16_replicate_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr()),
            reinterpret_cast<half*>(A.data_ptr<at::Half>()),
            M, K, topk);
    } else {
        // Already FP16, just replicate
        A = hidden_states.repeat_interleave(topk, 0);
    }

    // 3. GEMM1: gate_up projection [E_active, 2*N]
    int N2 = 2 * N;
    int n_scale_groups_13 = w13_scale.size(1);
    int n_blocks_13 = (N2 + NPB - 1) / NPB;
    int k_blocks_13 = (K + TK - 1) / TK;
    dim3 grid_13(n_blocks_13, k_blocks_13, E_active);

    auto C13 = torch::empty({E_active, k_blocks_13, N2}, opts_fp16);
    batched_sparse_v7_kernel<T, TN, TK>
        <<<grid_13, T, smem_v7(TK), stream>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            w13_comp.data_ptr<uint8_t>(),
            w13_meta.data_ptr<uint8_t>(),
            w13_scale.data_ptr<uint8_t>(),
            w13_g_scales.data_ptr<float>(),
            reinterpret_cast<half*>(C13.data_ptr<at::Half>()),
            mapped_ids.data_ptr<int>(),
            N2, K, n_scale_groups_13, E_active);

    // Reduce k-blocks if needed
    torch::Tensor gate_up;
    if (k_blocks_13 == 1) {
        gate_up = C13.squeeze(1);
    } else {
        gate_up = C13.sum(1);
    }

    // 4. SiLU activation: intermediate = silu(gate) * up
    auto intermediate = torch::empty({E_active, N}, opts_fp16);
    {
        int total = E_active * N;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        silu_mul_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const half*>(gate_up.data_ptr<at::Half>()),
            reinterpret_cast<half*>(intermediate.data_ptr<at::Half>()),
            E_active, N);
    }

    // 5. GEMM2: down projection [E_active, K]
    int K2 = N;  // Input to w2 is inter_size
    int N2_down = K;  // Output of w2 is hidden_size
    int n_scale_groups_2 = w2_scale.size(1);
    int n_blocks_2 = (N2_down + NPB - 1) / NPB;
    int k_blocks_2 = (K2 + TK - 1) / TK;
    dim3 grid_2(n_blocks_2, k_blocks_2, E_active);

    auto C2 = torch::empty({E_active, k_blocks_2, N2_down}, opts_fp16);
    batched_sparse_v7_kernel<T, TN, TK>
        <<<grid_2, T, smem_v7(TK), stream>>>(
            reinterpret_cast<const half*>(intermediate.data_ptr<at::Half>()),
            w2_comp.data_ptr<uint8_t>(),
            w2_meta.data_ptr<uint8_t>(),
            w2_scale.data_ptr<uint8_t>(),
            w2_g_scales.data_ptr<float>(),
            reinterpret_cast<half*>(C2.data_ptr<at::Half>()),
            mapped_ids.data_ptr<int>(),
            N2_down, K2, n_scale_groups_2, E_active);

    torch::Tensor down;
    if (k_blocks_2 == 1) {
        down = C2.squeeze(1);
    } else {
        down = C2.sum(1);
    }

    // 6. Weighted reduction [M, K]
    auto output = torch::empty({M, K}, opts_fp16);
    {
        int total = M * K;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        weighted_reduce_fp16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const half*>(down.data_ptr<at::Half>()),
            topk_weights.data_ptr<float>(),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            M, K, topk, apply_router_weight_on_input);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_dense_v7", &batched_dense_v7);
    m.def("batched_sparse_v7", &batched_sparse_v7);
    m.def("batched_sparse_packed_v7", &batched_sparse_packed_v7);
    m.def("fused_sparse_moe_v7", &fused_sparse_moe_v7);
}
