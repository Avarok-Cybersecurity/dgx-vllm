/*
 * Sparse FP4 GEMV v8 — v7 + three micro-optimizations
 *
 * Changes from v7:
 * 1. THREAD_N=8 with uint2 (64-bit) vectorized loads — fewer load instructions
 * 2. AtomicAdd accumulation — eliminates k-block dim + separate .sum(1) kernel
 * 3. Pre-scaled FP8 LUT in shared memory — folds global_scale into LUT once,
 *    also moves divergent constant-memory reads to shared memory (parallel banks)
 *
 * Bandwidth: same total data as v7, but eliminates the k-block reduction pass
 * (~1.3 MB saved per GEMM at K=2048, N=1024, 10 experts).
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

// FP8 E4M3FN -> float32 LUT (256 entries, initialized at module load)
__constant__ float c_fp8_lut[256];

static float fp8_e4m3fn_to_float_host(uint8_t x) {
    uint32_t sign = (x >> 7) & 1;
    uint32_t exp = (x >> 3) & 0xF;
    uint32_t mant = x & 0x7;
    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;
    float val;
    if (exp == 0) {
        val = ldexpf((float)mant / 8.0f, -6);
    } else {
        val = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    }
    return sign ? -val : val;
}

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
// v8 kernel: vectorized loads + atomic accumulation + pre-scaled LUT
// =====================================================================
template <int THREADS, int THREAD_N, int TILE_K>
__global__ void batched_sparse_v8_kernel(
    const half* __restrict__ A,               // [E_active, K]
    const uint8_t* __restrict__ B_comp_T,     // [E_total, K/4, N]
    const uint8_t* __restrict__ Meta_T_pk,    // [E_total, K/8, N]
    const uint8_t* __restrict__ scales_T,     // [E_total, n_scale_groups, N] FP8 E4M3FN
    const float* __restrict__ g_scales,       // [E_total] global scales
    float* __restrict__ C,                    // [E_active, N] float32 (atomicAdd target)
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

    // Shared memory layout: [fp4_lut(16)] [fp8_lut_scaled(256)] [A_tile(TILE_K)]
    extern __shared__ float sh_data[];
    float* sh_lut = sh_data;
    float* sh_fp8 = sh_data + 16;
    float* sh_A   = sh_data + 16 + 256;

    // FP4 LUT -> shared memory
    if (tid < 16) sh_lut[tid] = c_fp4_lut[tid];

    // Pre-scale FP8 LUT with global scale (optimization #3)
    // Moves from divergent constant-memory reads to shared-memory parallel reads
    for (int i = tid; i < 256; i += THREADS)
        sh_fp8[i] = c_fp8_lut[i] * g_scale;

    // Load A tile into shared memory
    for (int i = tid; i < k_end - k_start; i += THREADS)
        sh_A[i] = __half2float(A_row[k_start + i]);
    __syncthreads();

    if (!valid_n) return;

    float acc[THREAD_N] = {};
    const uint8_t* B_ptr = B_comp_T + b_comp_off + n_base;
    const uint8_t* M_ptr = Meta_T_pk + meta_off + n_base;
    const uint8_t* S_ptr = scales_T + scale_off + n_base;

    int g_start = k_start / 4;
    int g_end   = k_end / 4;
    int m_start = k_start / 8;

    for (int gi = g_start; gi < g_end; gi += 2) {
        int k_local = (gi - g_start) * 4;
        int mi = m_start + (gi - g_start) / 2;
        int si = (k_start / 8 + (gi - g_start) / 2) / pairs_per_scale;

        // 64-bit vectorized loads (optimization #1)
        uint2 comp0_8 = *reinterpret_cast<const uint2*>(&B_ptr[gi * N]);
        uint2 comp1_8 = *reinterpret_cast<const uint2*>(&B_ptr[(gi + 1) * N]);
        uint2 meta_8  = *reinterpret_cast<const uint2*>(&M_ptr[mi * N]);
        uint2 scale_8 = *reinterpret_cast<const uint2*>(&S_ptr[si * N]);

        // First 4 columns (from .x words)
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint8_t comp0 = (comp0_8.x >> (j * 8)) & 0xFF;
            uint8_t comp1 = (comp1_8.x >> (j * 8)) & 0xFF;
            uint8_t meta  = (meta_8.x  >> (j * 8)) & 0xFF;
            float scale   = sh_fp8[(scale_8.x >> (j * 8)) & 0xFF];

            float sum = sh_lut[comp0 & 0x0F]        * sh_A[k_local     + (meta & 3)]
                      + sh_lut[(comp0 >> 4) & 0x0F]  * sh_A[k_local     + ((meta >> 2) & 3)]
                      + sh_lut[comp1 & 0x0F]          * sh_A[k_local + 4 + ((meta >> 4) & 3)]
                      + sh_lut[(comp1 >> 4) & 0x0F]    * sh_A[k_local + 4 + ((meta >> 6) & 3)];
            acc[j] += sum * scale;
        }

        // Second 4 columns (from .y words)
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint8_t comp0 = (comp0_8.y >> (j * 8)) & 0xFF;
            uint8_t comp1 = (comp1_8.y >> (j * 8)) & 0xFF;
            uint8_t meta  = (meta_8.y  >> (j * 8)) & 0xFF;
            float scale   = sh_fp8[(scale_8.y >> (j * 8)) & 0xFF];

            float sum = sh_lut[comp0 & 0x0F]        * sh_A[k_local     + (meta & 3)]
                      + sh_lut[(comp0 >> 4) & 0x0F]  * sh_A[k_local     + ((meta >> 2) & 3)]
                      + sh_lut[comp1 & 0x0F]          * sh_A[k_local + 4 + ((meta >> 4) & 3)]
                      + sh_lut[(comp1 >> 4) & 0x0F]    * sh_A[k_local + 4 + ((meta >> 6) & 3)];
            acc[4 + j] += sum * scale;
        }
    }

    // AtomicAdd directly to [E_active, N] output (optimization #2)
    long out_base = (long)expert_active * N + n_base;
    #pragma unroll
    for (int j = 0; j < THREAD_N; j++)
        atomicAdd(&C[out_base + j], acc[j]);
}


// =====================================================================
// Helper kernels
// =====================================================================

__global__ void f32_to_f16_kernel(const float* __restrict__ input,
                                   half* __restrict__ output, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    output[idx] = __float2half(input[idx]);
}

__global__ void silu_mul_kernel(const half* __restrict__ input,
                                half* __restrict__ output, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int m = idx / N;
    int n = idx % N;
    float gate = __half2float(input[m * 2 * N + n]);
    float up   = __half2float(input[m * 2 * N + N + n]);
    float sig  = 1.0f / (1.0f + expf(-gate));
    output[idx] = __float2half(gate * sig * up);
}

// Fused SiLU: reads float32 input directly, writes FP16 output
// Saves one f32->f16 conversion pass for the gate_up GEMM output
__global__ void silu_mul_f32_kernel(const float* __restrict__ input,
                                     half* __restrict__ output, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int m = idx / N;
    int n = idx % N;
    float gate = input[m * 2 * N + n];
    float up   = input[m * 2 * N + N + n];
    float sig  = 1.0f / (1.0f + expf(-gate));
    output[idx] = __float2half(gate * sig * up);
}

__global__ void bf16_to_fp16_replicate_kernel(
    const __nv_bfloat16* __restrict__ input, half* __restrict__ output,
    int M, int K, int topk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * topk * K;
    if (idx >= total) return;
    int mk = idx / K;
    int k  = idx % K;
    int m  = mk / topk;
    output[idx] = __float2half(__bfloat162float(input[m * K + k]));
}

__global__ void weighted_reduce_fp16_kernel(
    const half* __restrict__ down, const float* __restrict__ weights,
    half* __restrict__ output, int M, int K, int topk,
    bool apply_router_weight_on_input) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;
    int m = idx / K;
    int k = idx % K;
    float acc = 0.0f;
    for (int t = 0; t < topk; t++) {
        float val = __half2float(down[(m * topk + t) * K + k]);
        acc += apply_router_weight_on_input ? val : val * weights[m * topk + t];
    }
    output[idx] = __float2half(acc);
}


// =====================================================================
// Python wrappers
// =====================================================================

constexpr int T8  = 128;    // threads per block
constexpr int TN8 = 8;      // N columns per thread (was 4 in v7)
constexpr int TK8 = 64;     // K tile size
constexpr int NPB8 = T8 * TN8;  // 1024 N per block (was 512 in v7)

static int smem_v8(int tile_k) {
    // fp4_lut(16) + fp8_scaled(256) + A_tile(tile_k) — all float32
    return (16 + 256 + tile_k) * sizeof(float);
}

torch::Tensor batched_sparse_v8(
    torch::Tensor A, torch::Tensor B_comp_T, torch::Tensor Meta_T_pk,
    torch::Tensor scales_T, torch::Tensor g_scales,
    torch::Tensor expert_ids
) {
    init_fp8_lut();

    int E_active = A.size(0);
    int K = A.size(1);
    int N = B_comp_T.size(2);
    int n_scale_groups = scales_T.size(1);

    int n_blocks = (N + NPB8 - 1) / NPB8;
    int k_blocks = (K + TK8 - 1) / TK8;
    dim3 grid(n_blocks, k_blocks, E_active);

    // Float32 output for atomicAdd
    auto C_f32 = torch::zeros({E_active, N},
        torch::dtype(torch::kFloat32).device(A.device()));

    auto stream = at::cuda::getCurrentCUDAStream();
    batched_sparse_v8_kernel<T8, TN8, TK8>
        <<<grid, T8, smem_v8(TK8), stream>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_comp_T.data_ptr<uint8_t>(),
            Meta_T_pk.data_ptr<uint8_t>(),
            scales_T.data_ptr<uint8_t>(),
            g_scales.data_ptr<float>(),
            C_f32.data_ptr<float>(),
            expert_ids.data_ptr<int>(),
            N, K, n_scale_groups, E_active);

    return C_f32.to(torch::kFloat16);
}


// =====================================================================
// Fused sparse MoE v8: full MoE layer in one C++ call
// =====================================================================
torch::Tensor fused_sparse_moe_v8(
    torch::Tensor hidden_states,       // [M, K] bfloat16 or float16
    torch::Tensor topk_weights,        // [M, topk] float32
    torch::Tensor topk_ids,            // [M, topk] int32
    torch::Tensor expert_map,          // [global_E] int32 or empty
    torch::Tensor w13_comp,            // [E_total, K/4, 2*N] uint8
    torch::Tensor w13_meta,            // [E_total, K/8, 2*N] uint8
    torch::Tensor w13_scale,           // [E_total, n_groups, 2*N] uint8
    torch::Tensor w13_g_scales,        // [E_total] float32
    torch::Tensor w2_comp,             // [E_total, N/4, K] uint8
    torch::Tensor w2_meta,             // [E_total, N/8, K] uint8
    torch::Tensor w2_scale,            // [E_total, n_groups2, K] uint8
    torch::Tensor w2_g_scales,         // [E_total] float32
    int inter_size,
    bool apply_router_weight_on_input
) {
    init_fp8_lut();

    int M = hidden_states.size(0);
    int K = hidden_states.size(1);
    int topk = topk_ids.size(1);
    int N = inter_size;
    int E_active = M * topk;

    auto stream = at::cuda::getCurrentCUDAStream();
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    auto opts_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(hidden_states.device());
    auto opts_f32  = torch::TensorOptions().dtype(torch::kFloat32).device(hidden_states.device());

    // 1. Map expert IDs
    torch::Tensor mapped_ids;
    if (expert_map.numel() > 0) {
        mapped_ids = expert_map.index({topk_ids}).reshape(-1).to(torch::kInt32);
    } else {
        mapped_ids = topk_ids.reshape(-1).to(torch::kInt32);
    }

    // 2. BF16->FP16 + token replication
    auto A = torch::empty({E_active, K}, opts_fp16);
    if (hidden_states.scalar_type() == torch::kBFloat16) {
        int total = E_active * K;
        bf16_to_fp16_replicate_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr()),
            reinterpret_cast<half*>(A.data_ptr<at::Half>()),
            M, K, topk);
    } else {
        A = hidden_states.repeat_interleave(topk, 0);
    }

    // 3. GEMM1: gate_up [E_active, 2*N] via atomic accumulation
    int N2 = 2 * N;
    int n_scale_groups_13 = w13_scale.size(1);
    int n_blocks_13 = (N2 + NPB8 - 1) / NPB8;
    int k_blocks_13 = (K + TK8 - 1) / TK8;
    dim3 grid_13(n_blocks_13, k_blocks_13, E_active);

    auto C13_f32 = torch::zeros({E_active, N2}, opts_f32);
    batched_sparse_v8_kernel<T8, TN8, TK8>
        <<<grid_13, T8, smem_v8(TK8), stream>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            w13_comp.data_ptr<uint8_t>(),
            w13_meta.data_ptr<uint8_t>(),
            w13_scale.data_ptr<uint8_t>(),
            w13_g_scales.data_ptr<float>(),
            C13_f32.data_ptr<float>(),
            mapped_ids.data_ptr<int>(),
            N2, K, n_scale_groups_13, E_active);

    // 4. SiLU directly from float32 accumulator (saves f32->f16 pass)
    auto intermediate = torch::empty({E_active, N}, opts_fp16);
    {
        int total = E_active * N;
        silu_mul_f32_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            C13_f32.data_ptr<float>(),
            reinterpret_cast<half*>(intermediate.data_ptr<at::Half>()),
            E_active, N);
    }

    // 5. GEMM2: down [E_active, K] via atomic accumulation
    int K2 = N;
    int N2_down = K;
    int n_scale_groups_2 = w2_scale.size(1);
    int n_blocks_2  = (N2_down + NPB8 - 1) / NPB8;
    int k_blocks_2  = (K2 + TK8 - 1) / TK8;
    dim3 grid_2(n_blocks_2, k_blocks_2, E_active);

    auto C2_f32 = torch::zeros({E_active, N2_down}, opts_f32);
    batched_sparse_v8_kernel<T8, TN8, TK8>
        <<<grid_2, T8, smem_v8(TK8), stream>>>(
            reinterpret_cast<const half*>(intermediate.data_ptr<at::Half>()),
            w2_comp.data_ptr<uint8_t>(),
            w2_meta.data_ptr<uint8_t>(),
            w2_scale.data_ptr<uint8_t>(),
            w2_g_scales.data_ptr<float>(),
            C2_f32.data_ptr<float>(),
            mapped_ids.data_ptr<int>(),
            N2_down, K2, n_scale_groups_2, E_active);

    // 6. Convert down to FP16
    auto down = torch::empty({E_active, N2_down}, opts_fp16);
    {
        int total = E_active * N2_down;
        f32_to_f16_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            C2_f32.data_ptr<float>(),
            reinterpret_cast<half*>(down.data_ptr<at::Half>()),
            total);
    }

    // 7. Weighted reduction [M, K]
    auto output = torch::empty({M, K}, opts_fp16);
    {
        int total = M * K;
        weighted_reduce_fp16_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<const half*>(down.data_ptr<at::Half>()),
            topk_weights.data_ptr<float>(),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            M, K, topk, apply_router_weight_on_input);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_sparse_v8", &batched_sparse_v8);
    m.def("fused_sparse_moe_v8", &fused_sparse_moe_v8);
}
