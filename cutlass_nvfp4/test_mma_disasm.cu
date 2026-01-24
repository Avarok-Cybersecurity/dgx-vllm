// Compile a WORKING tensor core instruction to study encoding
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Use WMMA (which DOES compile) to study instruction encoding
__global__ void wmma_reference_kernel(half* output) {
    __shared__ half A[16*16];
    __shared__ half B[16*16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    // Initialize shared memory
    int tid = threadIdx.x;
    if (tid < 256) {
        A[tid] = __float2half(1.0f);
        B[tid] = __float2half(1.0f);
    }
    __syncthreads();

    // Load from shared memory
    wmma::load_matrix_sync(a_frag, A, 16);
    wmma::load_matrix_sync(b_frag, B, 16);
    wmma::fill_fragment(c_frag, __float2half(0.0f));

    // THIS generates the actual mma.sync instruction we can study!
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Force output so optimizer doesn't remove it
    wmma::store_matrix_sync(output, c_frag, 16, wmma::mem_row_major);
}

int main() {
    half* d_output;
    cudaMalloc(&d_output, 16*16*sizeof(half));
    wmma_reference_kernel<<<1, 32>>>(d_output);
    cudaDeviceSynchronize();
    cudaFree(d_output);
    return 0;
}
