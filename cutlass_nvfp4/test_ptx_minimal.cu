// Minimal PTX test to diagnose tcgen05.mma support

#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>

__global__ void test_ptx_instruction() {
    uint32_t tmem_addr = 0;
    uint64_t a_desc = 0;
    uint64_t b_desc = 0;
    uint32_t idesc = 0;

    // Test if tcgen05.mma instruction is recognized
    asm volatile(
        "{\n"
        "  .reg .b32 tmem_reg;\n"
        "  .reg .b64 a_reg, b_reg;\n"
        "  .reg .b32 i_reg;\n"
        "  mov.b32 tmem_reg, %0;\n"
        "  mov.b64 a_reg, %1;\n"
        "  mov.b64 b_reg, %2;\n"
        "  mov.b32 i_reg, %3;\n"
        "  tcgen05.mma.ss.kind:f8f6f4 [tmem_reg], a_reg, b_reg, i_reg;\n"
        "}\n"
        :
        : "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc)
        : "memory"
    );
}

int main() {
    std::cout << "Testing minimal tcgen05 PTX support..." << std::endl;

    test_ptx_instruction<<<1, 32>>>();

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Success!" << std::endl;
    return 0;
}
