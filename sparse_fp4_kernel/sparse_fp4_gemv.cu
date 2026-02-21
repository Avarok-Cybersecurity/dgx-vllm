/*
 * Sparse FP4 GEMV/GEMM kernel for SM121 (GB10).
 *
 * Software-based 2:4 sparse decompression: reads compressed FP4 weights
 * (50% of dense) plus metadata (12.5% of dense), decompresses in registers,
 * and accumulates with FP32 precision.
 *
 * Total memory read: 75% of dense FP4 → 25% bandwidth savings.
 *
 * Weight format:
 *   Compressed: [N, K_logical/4] uint8 — 2 FP4 values packed per byte
 *   Metadata:   [N, K_logical/8] uint8 — 2-bit positions, 2 groups per byte
 *
 * For each group of 4 FP4 elements along K:
 *   - 2 non-zero values stored (packed in 1 compressed byte)
 *   - Their positions (0-3) stored as 2-bit indices in metadata
 *   - Metadata nibble: bits[1:0] = pos0, bits[3:2] = pos1
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

// FP4 E2M1 dequantization lookup table (constant memory)
__constant__ float c_fp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};


// ─── Sparse FP4 GEMV: C[1,N] = A[1,K] @ decompress(B,Meta).T ─────────

template <int BLOCK_N>
__global__ void sparse_fp4_gemv_kernel(
    const half* __restrict__ A,          // [1, K]
    const uint8_t* __restrict__ B_comp,  // [N, K/4] compressed FP4
    const uint8_t* __restrict__ Meta,    // [N, K/8] packed metadata
    half* __restrict__ C,                // [1, N]
    const int N,
    const int K
) {
    // Load activation vector into shared memory
    extern __shared__ half sh_A[];

    const int K4 = K / 4;  // groups of 4 FP4
    const int K8 = K / 8;  // metadata bytes

    for (int i = threadIdx.x; i < K; i += BLOCK_N)
        sh_A[i] = A[i];
    __syncthreads();

    const int n = blockIdx.x * BLOCK_N + threadIdx.x;
    if (n >= N) return;

    float acc = 0.0f;

    // Pointers for this row
    const uint8_t* b_row = B_comp + n * K4;
    const uint8_t* m_row = Meta + n * K8;

    for (int g = 0; g < K4; g++) {
        // Load compressed byte: 2 FP4 values
        const uint8_t comp = b_row[g];
        const float w0 = c_fp4_lut[comp & 0x0F];
        const float w1 = c_fp4_lut[(comp >> 4) & 0x0F];

        // Load metadata: 4 bits per group, 2 groups per byte
        const uint8_t meta_byte = m_row[g >> 1];
        const uint8_t meta_nibble = (g & 1) ? (meta_byte >> 4) : (meta_byte & 0x0F);
        const int pos0 = meta_nibble & 3;
        const int pos1 = (meta_nibble >> 2) & 3;

        // Gather activation values from shared memory
        const int base = g * 4;
        const float a0 = __half2float(sh_A[base + pos0]);
        const float a1 = __half2float(sh_A[base + pos1]);

        acc += w0 * a0 + w1 * a1;
    }

    C[n] = __float2half(acc);
}


// ─── Dense FP4 GEMV (reference): C[1,N] = A[1,K] @ dequant(B_dense).T ─

template <int BLOCK_N>
__global__ void dense_fp4_gemv_kernel(
    const half* __restrict__ A,          // [1, K]
    const uint8_t* __restrict__ B_dense, // [N, K/2] packed FP4
    half* __restrict__ C,                // [1, N]
    const int N,
    const int K
) {
    extern __shared__ half sh_A[];
    const int K2 = K / 2;

    for (int i = threadIdx.x; i < K; i += BLOCK_N)
        sh_A[i] = A[i];
    __syncthreads();

    const int n = blockIdx.x * BLOCK_N + threadIdx.x;
    if (n >= N) return;

    float acc = 0.0f;
    const uint8_t* b_row = B_dense + n * K2;

    for (int i = 0; i < K2; i++) {
        const uint8_t packed = b_row[i];
        const float w0 = c_fp4_lut[packed & 0x0F];
        const float w1 = c_fp4_lut[(packed >> 4) & 0x0F];

        const int base = i * 2;
        const float a0 = __half2float(sh_A[base]);
        const float a1 = __half2float(sh_A[base + 1]);

        acc += w0 * a0 + w1 * a1;
    }

    C[n] = __float2half(acc);
}


// ─── Sparse FP4 GEMM: C[M,N] = A[M,K] @ decompress(B,Meta).T ─────────

template <int BLOCK_M, int BLOCK_N, int TILE_K>
__global__ void sparse_fp4_gemm_kernel(
    const half* __restrict__ A,          // [M, K]
    const uint8_t* __restrict__ B_comp,  // [N, K/4]
    const uint8_t* __restrict__ Meta,    // [N, K/8]
    half* __restrict__ C,                // [M, N]
    const int M, const int N, const int K,
    const int stride_am, const int stride_cm
) {
    const int K4 = K / 4;
    const int K8 = K / 8;

    // Thread (tx, ty) handles output C[bm + ty, bn + tx]
    const int bn = blockIdx.x * BLOCK_N + threadIdx.x;
    const int bm = blockIdx.y * BLOCK_M;

    if (bn >= N) return;

    // Each thread accumulates BLOCK_M output values
    float acc[BLOCK_M];
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++) acc[i] = 0.0f;

    const uint8_t* b_row = B_comp + bn * K4;
    const uint8_t* m_row = Meta + bn * K8;

    for (int g = 0; g < K4; g++) {
        const uint8_t comp = b_row[g];
        const float w0 = c_fp4_lut[comp & 0x0F];
        const float w1 = c_fp4_lut[(comp >> 4) & 0x0F];

        const uint8_t meta_byte = m_row[g >> 1];
        const uint8_t meta_nibble = (g & 1) ? (meta_byte >> 4) : (meta_byte & 0x0F);
        const int pos0 = meta_nibble & 3;
        const int pos1 = (meta_nibble >> 2) & 3;

        const int base = g * 4;
        #pragma unroll
        for (int i = 0; i < BLOCK_M; i++) {
            const int m = bm + i;
            if (m < M) {
                const float a0 = __half2float(A[m * stride_am + base + pos0]);
                const float a1 = __half2float(A[m * stride_am + base + pos1]);
                acc[i] += w0 * a0 + w1 * a1;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++) {
        const int m = bm + i;
        if (m < M) {
            C[m * stride_cm + bn] = __float2half(acc[i]);
        }
    }
}


// ─── Dense FP4 GEMM (reference) ───────────────────────────────

template <int BLOCK_M, int BLOCK_N>
__global__ void dense_fp4_gemm_kernel(
    const half* __restrict__ A,
    const uint8_t* __restrict__ B_dense,
    half* __restrict__ C,
    const int M, const int N, const int K,
    const int stride_am, const int stride_cm
) {
    const int K2 = K / 2;
    const int bn = blockIdx.x * BLOCK_N + threadIdx.x;
    const int bm = blockIdx.y * BLOCK_M;

    if (bn >= N) return;

    float acc[BLOCK_M];
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++) acc[i] = 0.0f;

    const uint8_t* b_row = B_dense + bn * K2;

    for (int i = 0; i < K2; i++) {
        const uint8_t packed = b_row[i];
        const float w0 = c_fp4_lut[packed & 0x0F];
        const float w1 = c_fp4_lut[(packed >> 4) & 0x0F];

        const int base = i * 2;
        #pragma unroll
        for (int j = 0; j < BLOCK_M; j++) {
            const int m = bm + j;
            if (m < M) {
                const float a0 = __half2float(A[m * stride_am + base]);
                const float a1 = __half2float(A[m * stride_am + base + 1]);
                acc[j] += w0 * a0 + w1 * a1;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++) {
        const int m = bm + i;
        if (m < M) {
            C[m * stride_cm + bn] = __float2half(acc[i]);
        }
    }
}


// ─── Python bindings ──────────────────────────────────────────

torch::Tensor sparse_fp4_gemv(
    torch::Tensor A,        // [1, K] float16
    torch::Tensor B_comp,   // [N, K/4] uint8
    torch::Tensor Meta      // [N, K/8] uint8
) {
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1, "A must be [1, K]");
    TORCH_CHECK(B_comp.dim() == 2 && Meta.dim() == 2, "B_comp and Meta must be 2D");

    const int K = A.size(1);
    const int N = B_comp.size(0);

    TORCH_CHECK(B_comp.size(1) == K / 4, "B_comp must be [N, K/4]");
    TORCH_CHECK(Meta.size(1) == K / 8, "Meta must be [N, K/8]");

    auto C = torch::empty({1, N}, A.options());

    constexpr int BLOCK_N = 256;
    const int grid = (N + BLOCK_N - 1) / BLOCK_N;
    const int smem = K * sizeof(half);

    sparse_fp4_gemv_kernel<BLOCK_N><<<grid, BLOCK_N, smem>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        B_comp.data_ptr<uint8_t>(),
        Meta.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        N, K
    );
    return C;
}


torch::Tensor dense_fp4_gemv(
    torch::Tensor A,        // [1, K] float16
    torch::Tensor B_dense   // [N, K/2] uint8
) {
    const int K = A.size(1);
    const int N = B_dense.size(0);

    auto C = torch::empty({1, N}, A.options());

    constexpr int BLOCK_N = 256;
    const int grid = (N + BLOCK_N - 1) / BLOCK_N;
    const int smem = K * sizeof(half);

    dense_fp4_gemv_kernel<BLOCK_N><<<grid, BLOCK_N, smem>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        B_dense.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        N, K
    );
    return C;
}


torch::Tensor sparse_fp4_gemm(
    torch::Tensor A,        // [M, K] float16
    torch::Tensor B_comp,   // [N, K/4] uint8
    torch::Tensor Meta      // [N, K/8] uint8
) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B_comp.size(0);

    auto C = torch::empty({M, N}, A.options());

    if (M == 1) {
        return sparse_fp4_gemv(A, B_comp, Meta);
    }

    constexpr int BLOCK_N = 256;
    constexpr int BLOCK_M = 4;
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    sparse_fp4_gemm_kernel<BLOCK_M, BLOCK_N, 64><<<grid, BLOCK_N>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        B_comp.data_ptr<uint8_t>(),
        Meta.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K,
        (int)A.stride(0), (int)C.stride(0)
    );
    return C;
}


torch::Tensor dense_fp4_gemm(
    torch::Tensor A,        // [M, K] float16
    torch::Tensor B_dense   // [N, K/2] uint8
) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B_dense.size(0);

    auto C = torch::empty({M, N}, A.options());

    if (M == 1) {
        return dense_fp4_gemv(A, B_dense);
    }

    constexpr int BLOCK_N = 256;
    constexpr int BLOCK_M = 4;
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    dense_fp4_gemm_kernel<BLOCK_M, BLOCK_N><<<grid, BLOCK_N>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        B_dense.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K,
        (int)A.stride(0), (int)C.stride(0)
    );
    return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_fp4_gemv", &sparse_fp4_gemv, "Sparse FP4 GEMV (2:4)");
    m.def("dense_fp4_gemv", &dense_fp4_gemv, "Dense FP4 GEMV (baseline)");
    m.def("sparse_fp4_gemm", &sparse_fp4_gemm, "Sparse FP4 GEMM (2:4)");
    m.def("dense_fp4_gemm", &dense_fp4_gemm, "Dense FP4 GEMM (baseline)");
}
