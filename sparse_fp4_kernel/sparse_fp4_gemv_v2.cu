/*
 * Optimized Sparse FP4 GEMV kernel for SM121 (GB10).
 *
 * Key optimization: TRANSPOSED weight layout for coalesced memory access.
 * Dense B: [K/2, N] uint8 (columns of N are contiguous — coalesced)
 * Sparse B: [K/4, N] uint8 compressed + [K/4, N] uint8 metadata
 *
 * For GEMV (M=1), this kernel reads one column of activations
 * and streams through weight rows with perfect coalescing.
 *
 * Weight format (transposed, row-major in K-group dimension):
 *   B_comp_T: [K/4, N] uint8 — 2 FP4 non-zero values per group, packed
 *   Meta_T:   [K/4, N] uint8 — 1 byte per group: bits[1:0]=pos0, bits[3:2]=pos1
 *   (Using 1-byte metadata per group for simplicity and alignment)
 *
 * Memory per row of N: K/4 (compressed) + K/4 (metadata) = K/2 bytes
 * Dense baseline: K/2 bytes per row of N → SAME size but different layout
 *
 * Wait — with 1-byte metadata per group, sparse is NOT smaller than dense!
 * With 4-bit packed metadata (2 groups/byte): K/4 + K/8 = 3K/8 per row → 75%
 *
 * Let's use BOTH formats:
 * v2a: 1-byte metadata (same size as dense, tests decompression overhead)
 * v2b: 4-bit packed metadata (75% of dense, actual savings)
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

// FP4 E2M1 dequantization lookup table
__constant__ float c_fp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};


// ─── Dense FP4 GEMV with transposed weights ──────────────────

// B_dense_T: [K/2, N] uint8 — transposed for coalesced access
// Each row g has N bytes: B_dense_T[g, n] = packed byte containing 2 FP4s
// that map to logical K indices 2g and 2g+1
template <int BLOCK_N>
__global__ void dense_fp4_gemv_T_kernel(
    const half* __restrict__ A,           // [1, K]
    const uint8_t* __restrict__ B_dense_T, // [K/2, N] transposed packed FP4
    half* __restrict__ C,                  // [1, N]
    const int N,
    const int K
) {
    extern __shared__ half sh_A[];  // [K]

    const int K2 = K / 2;

    // Cooperative load of A into shared memory
    for (int i = threadIdx.x; i < K; i += BLOCK_N)
        sh_A[i] = A[i];
    __syncthreads();

    const int n = blockIdx.x * BLOCK_N + threadIdx.x;
    if (n >= N) return;

    float acc = 0.0f;

    for (int g = 0; g < K2; g++) {
        // Coalesced read: adjacent threads read adjacent N elements
        const uint8_t packed = B_dense_T[g * N + n];
        const float w0 = c_fp4_lut[packed & 0x0F];
        const float w1 = c_fp4_lut[(packed >> 4) & 0x0F];

        const int base = g * 2;
        const float a0 = __half2float(sh_A[base]);
        const float a1 = __half2float(sh_A[base + 1]);

        acc += w0 * a0 + w1 * a1;
    }

    C[n] = __float2half(acc);
}


// ─── Sparse FP4 GEMV with transposed weights (1-byte metadata) ─

// B_comp_T: [K/4, N] uint8 — compressed (2 non-zero FP4 per group of 4)
// Meta_T:   [K/4, N] uint8 — metadata (1 byte per group: pos0 | pos1<<2)
template <int BLOCK_N>
__global__ void sparse_fp4_gemv_T_kernel(
    const half* __restrict__ A,            // [1, K]
    const uint8_t* __restrict__ B_comp_T,  // [K/4, N]
    const uint8_t* __restrict__ Meta_T,    // [K/4, N]
    half* __restrict__ C,                  // [1, N]
    const int N,
    const int K
) {
    extern __shared__ half sh_A[];

    const int K4 = K / 4;

    for (int i = threadIdx.x; i < K; i += BLOCK_N)
        sh_A[i] = A[i];
    __syncthreads();

    const int n = blockIdx.x * BLOCK_N + threadIdx.x;
    if (n >= N) return;

    float acc = 0.0f;

    for (int g = 0; g < K4; g++) {
        // Coalesced reads: adjacent threads read adjacent N elements
        const uint8_t comp = B_comp_T[g * N + n];
        const uint8_t meta = Meta_T[g * N + n];

        const float w0 = c_fp4_lut[comp & 0x0F];
        const float w1 = c_fp4_lut[(comp >> 4) & 0x0F];

        const int pos0 = meta & 3;
        const int pos1 = (meta >> 2) & 3;

        const int base = g * 4;
        const float a0 = __half2float(sh_A[base + pos0]);
        const float a1 = __half2float(sh_A[base + pos1]);

        acc += w0 * a0 + w1 * a1;
    }

    C[n] = __float2half(acc);
}


// ─── Sparse FP4 GEMV with 4-bit packed metadata (actual savings) ─

// B_comp_T: [K/4, N] uint8
// Meta_T:   [K/8, N] uint8 — 2 groups per byte (low nibble=even group, high=odd)
template <int BLOCK_N>
__global__ void sparse_fp4_gemv_T_packed_kernel(
    const half* __restrict__ A,
    const uint8_t* __restrict__ B_comp_T,  // [K/4, N]
    const uint8_t* __restrict__ Meta_T,    // [K/8, N]
    half* __restrict__ C,
    const int N,
    const int K
) {
    extern __shared__ half sh_A[];

    const int K4 = K / 4;
    const int K8 = K / 8;

    for (int i = threadIdx.x; i < K; i += BLOCK_N)
        sh_A[i] = A[i];
    __syncthreads();

    const int n = blockIdx.x * BLOCK_N + threadIdx.x;
    if (n >= N) return;

    float acc = 0.0f;

    // Process 2 groups at a time (matching metadata byte packing)
    for (int g_pair = 0; g_pair < K8; g_pair++) {
        // One metadata byte covers 2 groups
        const uint8_t meta_byte = Meta_T[g_pair * N + n];

        // Group 2*g_pair (even)
        {
            const int g = g_pair * 2;
            const uint8_t comp = B_comp_T[g * N + n];
            const float w0 = c_fp4_lut[comp & 0x0F];
            const float w1 = c_fp4_lut[(comp >> 4) & 0x0F];

            const uint8_t meta = meta_byte & 0x0F;
            const int pos0 = meta & 3;
            const int pos1 = (meta >> 2) & 3;

            const int base = g * 4;
            acc += w0 * __half2float(sh_A[base + pos0])
                 + w1 * __half2float(sh_A[base + pos1]);
        }

        // Group 2*g_pair+1 (odd)
        {
            const int g = g_pair * 2 + 1;
            if (g < K4) {
                const uint8_t comp = B_comp_T[g * N + n];
                const float w0 = c_fp4_lut[comp & 0x0F];
                const float w1 = c_fp4_lut[(comp >> 4) & 0x0F];

                const uint8_t meta = (meta_byte >> 4) & 0x0F;
                const int pos0 = meta & 3;
                const int pos1 = (meta >> 2) & 3;

                const int base = g * 4;
                acc += w0 * __half2float(sh_A[base + pos0])
                     + w1 * __half2float(sh_A[base + pos1]);
            }
        }
    }

    C[n] = __float2half(acc);
}


// ─── GEMM variants (small M) ─────────────────────────────────

template <int BLOCK_N, int BLOCK_M>
__global__ void dense_fp4_gemm_T_kernel(
    const half* __restrict__ A,
    const uint8_t* __restrict__ B_dense_T,
    half* __restrict__ C,
    const int M, const int N, const int K,
    const int stride_am, const int stride_cm
) {
    extern __shared__ half sh_A[];  // [BLOCK_M * K]

    const int K2 = K / 2;
    const int bm = blockIdx.y * BLOCK_M;

    // Load BLOCK_M rows of A into shared memory
    for (int i = threadIdx.x; i < BLOCK_M * K; i += BLOCK_N) {
        int row = i / K;
        int col = i % K;
        int m = bm + row;
        sh_A[i] = (m < M) ? A[m * stride_am + col] : __float2half(0.0f);
    }
    __syncthreads();

    const int n = blockIdx.x * BLOCK_N + threadIdx.x;
    if (n >= N) return;

    float acc[BLOCK_M];
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++) acc[i] = 0.0f;

    for (int g = 0; g < K2; g++) {
        const uint8_t packed = B_dense_T[g * N + n];
        const float w0 = c_fp4_lut[packed & 0x0F];
        const float w1 = c_fp4_lut[(packed >> 4) & 0x0F];
        const int base = g * 2;

        #pragma unroll
        for (int i = 0; i < BLOCK_M; i++) {
            acc[i] += w0 * __half2float(sh_A[i * K + base])
                    + w1 * __half2float(sh_A[i * K + base + 1]);
        }
    }

    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++) {
        int m = bm + i;
        if (m < M) C[m * stride_cm + n] = __float2half(acc[i]);
    }
}


template <int BLOCK_N, int BLOCK_M>
__global__ void sparse_fp4_gemm_T_packed_kernel(
    const half* __restrict__ A,
    const uint8_t* __restrict__ B_comp_T,
    const uint8_t* __restrict__ Meta_T,
    half* __restrict__ C,
    const int M, const int N, const int K,
    const int stride_am, const int stride_cm
) {
    extern __shared__ half sh_A[];

    const int K4 = K / 4;
    const int K8 = K / 8;
    const int bm = blockIdx.y * BLOCK_M;

    for (int i = threadIdx.x; i < BLOCK_M * K; i += BLOCK_N) {
        int row = i / K;
        int col = i % K;
        int m = bm + row;
        sh_A[i] = (m < M) ? A[m * stride_am + col] : __float2half(0.0f);
    }
    __syncthreads();

    const int n = blockIdx.x * BLOCK_N + threadIdx.x;
    if (n >= N) return;

    float acc[BLOCK_M];
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++) acc[i] = 0.0f;

    for (int g_pair = 0; g_pair < K8; g_pair++) {
        const uint8_t meta_byte = Meta_T[g_pair * N + n];

        // Even group
        {
            const int g = g_pair * 2;
            const uint8_t comp = B_comp_T[g * N + n];
            const float w0 = c_fp4_lut[comp & 0x0F];
            const float w1 = c_fp4_lut[(comp >> 4) & 0x0F];
            const uint8_t meta = meta_byte & 0x0F;
            const int pos0 = meta & 3;
            const int pos1 = (meta >> 2) & 3;
            const int base = g * 4;

            #pragma unroll
            for (int i = 0; i < BLOCK_M; i++) {
                acc[i] += w0 * __half2float(sh_A[i * K + base + pos0])
                        + w1 * __half2float(sh_A[i * K + base + pos1]);
            }
        }

        // Odd group
        {
            const int g = g_pair * 2 + 1;
            if (g < K4) {
                const uint8_t comp = B_comp_T[g * N + n];
                const float w0 = c_fp4_lut[comp & 0x0F];
                const float w1 = c_fp4_lut[(comp >> 4) & 0x0F];
                const uint8_t meta = (meta_byte >> 4) & 0x0F;
                const int pos0 = meta & 3;
                const int pos1 = (meta >> 2) & 3;
                const int base = g * 4;

                #pragma unroll
                for (int i = 0; i < BLOCK_M; i++) {
                    acc[i] += w0 * __half2float(sh_A[i * K + base + pos0])
                            + w1 * __half2float(sh_A[i * K + base + pos1]);
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++) {
        int m = bm + i;
        if (m < M) C[m * stride_cm + n] = __float2half(acc[i]);
    }
}


// ─── Python bindings ──────────────────────────────────────────

torch::Tensor dense_fp4_gemv_T(torch::Tensor A, torch::Tensor B_dense_T) {
    const int K = A.size(1);
    const int N = B_dense_T.size(1);
    auto C = torch::empty({1, N}, A.options());

    constexpr int BN = 256;
    const int grid = (N + BN - 1) / BN;
    dense_fp4_gemv_T_kernel<BN><<<grid, BN, K * sizeof(half)>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        B_dense_T.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        N, K);
    return C;
}

torch::Tensor sparse_fp4_gemv_T(torch::Tensor A, torch::Tensor B_comp_T, torch::Tensor Meta_T) {
    const int K = A.size(1);
    const int N = B_comp_T.size(1);
    auto C = torch::empty({1, N}, A.options());

    constexpr int BN = 256;
    const int grid = (N + BN - 1) / BN;
    sparse_fp4_gemv_T_kernel<BN><<<grid, BN, K * sizeof(half)>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        B_comp_T.data_ptr<uint8_t>(),
        Meta_T.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        N, K);
    return C;
}

torch::Tensor sparse_fp4_gemv_T_packed(torch::Tensor A, torch::Tensor B_comp_T, torch::Tensor Meta_T_packed) {
    const int K = A.size(1);
    const int N = B_comp_T.size(1);
    auto C = torch::empty({1, N}, A.options());

    constexpr int BN = 256;
    const int grid = (N + BN - 1) / BN;
    sparse_fp4_gemv_T_packed_kernel<BN><<<grid, BN, K * sizeof(half)>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        B_comp_T.data_ptr<uint8_t>(),
        Meta_T_packed.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        N, K);
    return C;
}

torch::Tensor dense_fp4_gemm_T(torch::Tensor A, torch::Tensor B_dense_T) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B_dense_T.size(1);

    if (M == 1) return dense_fp4_gemv_T(A, B_dense_T);

    auto C = torch::empty({M, N}, A.options());
    constexpr int BN = 256;
    constexpr int BM = 4;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dense_fp4_gemm_T_kernel<BN, BM><<<grid, BN, BM * K * sizeof(half)>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        B_dense_T.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K, (int)A.stride(0), (int)C.stride(0));
    return C;
}

torch::Tensor sparse_fp4_gemm_T_packed(torch::Tensor A, torch::Tensor B_comp_T, torch::Tensor Meta_T_packed) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B_comp_T.size(1);

    if (M == 1) return sparse_fp4_gemv_T_packed(A, B_comp_T, Meta_T_packed);

    auto C = torch::empty({M, N}, A.options());
    constexpr int BN = 256;
    constexpr int BM = 4;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sparse_fp4_gemm_T_packed_kernel<BN, BM><<<grid, BN, BM * K * sizeof(half)>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        B_comp_T.data_ptr<uint8_t>(),
        Meta_T_packed.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K, (int)A.stride(0), (int)C.stride(0));
    return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dense_fp4_gemv_T", &dense_fp4_gemv_T, "Dense FP4 GEMV (transposed weights)");
    m.def("sparse_fp4_gemv_T", &sparse_fp4_gemv_T, "Sparse FP4 GEMV (1-byte meta)");
    m.def("sparse_fp4_gemv_T_packed", &sparse_fp4_gemv_T_packed, "Sparse FP4 GEMV (4-bit meta)");
    m.def("dense_fp4_gemm_T", &dense_fp4_gemm_T, "Dense FP4 GEMM (transposed weights)");
    m.def("sparse_fp4_gemm_T_packed", &sparse_fp4_gemm_T_packed, "Sparse FP4 GEMM (4-bit meta)");
}
