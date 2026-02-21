// SPDX-License-Identifier: Apache-2.0
/**
 * @file test_nvfp4_types.cu
 * @brief Unit tests for NVFP4 data type implementation
 *
 * Tests FP4 conversion accuracy, packing/unpacking, and edge cases.
 */

#include "nvfp4_types.cuh"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>

using namespace cutlass_nvfp4;

/// Test structure for FP4 value validation
struct FP4TestCase {
    uint8_t fp4_bits;
    float expected_float;
    const char* description;
};

/// All 16 possible FP4 values with expected conversions
constexpr FP4TestCase test_cases[] = {
    // Positive values
    {0x0, 0.0f,   "Zero"},
    {0x1, 0.25f,  "Subnormal: 0.25"},
    {0x2, 1.0f,   "Normal: 1.0"},
    {0x3, 1.5f,   "Normal: 1.5"},
    {0x4, 2.0f,   "Normal: 2.0"},
    {0x5, 3.0f,   "Normal: 3.0"},
    {0x6, 4.0f,   "Normal: 4.0"},
    {0x7, 6.0f,   "Normal: 6.0 (max)"},

    // Negative values
    {0x8, -0.0f,  "Negative zero"},
    {0x9, -0.25f, "Subnormal: -0.25"},
    {0xA, -1.0f,  "Normal: -1.0"},
    {0xB, -1.5f,  "Normal: -1.5"},
    {0xC, -2.0f,  "Normal: -2.0"},
    {0xD, -3.0f,  "Normal: -3.0"},
    {0xE, -4.0f,  "Normal: -4.0"},
    {0xF, -6.0f,  "Normal: -6.0 (min)"},
};

/**
 * @brief Test all FP4 → float conversions
 */
bool test_fp4_to_float_conversions() {
    std::cout << "\n=== Testing FP4 → Float Conversions ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    bool all_passed = true;

    for (const auto& test : test_cases) {
        nvfp4_t fp4(test.fp4_bits);
        float result = fp4.to_float();

        bool passed = (std::abs(result - test.expected_float) < 1e-6f);

        std::cout << "  0x" << std::hex << std::setw(1) << (int)test.fp4_bits
                  << std::dec << " (" << test.description << "): "
                  << "Expected " << std::setw(6) << test.expected_float
                  << ", Got " << std::setw(6) << result
                  << " ... " << (passed ? "✓" : "✗") << std::endl;

        if (!passed) {
            all_passed = false;
        }
    }

    return all_passed;
}

/**
 * @brief Test float → FP4 quantization
 */
bool test_float_to_fp4_quantization() {
    std::cout << "\n=== Testing Float → FP4 Quantization ===" << std::endl;

    struct QuantTestCase {
        float input;
        float expected_output;
        const char* description;
    };

    QuantTestCase quant_tests[] = {
        {0.0f, 0.0f, "Exact zero"},
        {0.1f, 0.0f, "Round down to zero"},
        {0.2f, 0.25f, "Round up to 0.25"},
        {0.25f, 0.25f, "Exact 0.25"},
        {0.9f, 1.0f, "Round to 1.0"},
        {1.0f, 1.0f, "Exact 1.0"},
        {1.3f, 1.5f, "Round to 1.5"},
        {1.5f, 1.5f, "Exact 1.5"},
        {1.8f, 2.0f, "Round to 2.0"},
        {2.8f, 3.0f, "Round to 3.0"},
        {5.0f, 6.0f, "Round to 6.0"},
        {7.0f, 6.0f, "Clamp to max (6.0)"},
        {100.0f, 6.0f, "Clamp large value"},
        {-1.0f, -1.0f, "Exact -1.0"},
        {-5.0f, -6.0f, "Round to -6.0"},
        {-100.0f, -6.0f, "Clamp to min (-6.0)"},
    };

    bool all_passed = true;

    for (const auto& test : quant_tests) {
        nvfp4_t fp4 = nvfp4_t::from_float(test.input);
        float result = fp4.to_float();

        bool passed = (std::abs(result - test.expected_output) < 1e-6f);

        std::cout << "  " << std::setw(8) << test.input
                  << " → FP4 → " << std::setw(6) << result
                  << " (expected " << std::setw(6) << test.expected_output << ") "
                  << " [" << test.description << "] ... "
                  << (passed ? "✓" : "✗") << std::endl;

        if (!passed) {
            all_passed = false;
        }
    }

    return all_passed;
}

/**
 * @brief Test packed FP4 operations
 */
bool test_packed_fp4() {
    std::cout << "\n=== Testing Packed FP4 Operations ===" << std::endl;

    bool all_passed = true;

    // Test packing and unpacking
    {
        nvfp4_t lo = nvfp4_t::from_float(1.0f);
        nvfp4_t hi = nvfp4_t::from_float(2.0f);

        nvfp4x2_t packed(lo, hi);

        nvfp4_t unpacked_lo = packed.lo();
        nvfp4_t unpacked_hi = packed.hi();

        float result_lo = unpacked_lo.to_float();
        float result_hi = unpacked_hi.to_float();

        bool passed = (std::abs(result_lo - 1.0f) < 1e-6f &&
                       std::abs(result_hi - 2.0f) < 1e-6f);

        std::cout << "  Pack(1.0, 2.0) → Unpack → ("
                  << result_lo << ", " << result_hi << ") ... "
                  << (passed ? "✓" : "✗") << std::endl;

        if (!passed) all_passed = false;
    }

    // Test from_floats
    {
        nvfp4x2_t packed = nvfp4x2_t::from_floats(1.5f, 3.0f);
        float lo_val, hi_val;
        packed.to_floats(lo_val, hi_val);

        bool passed = (std::abs(lo_val - 1.5f) < 1e-6f &&
                       std::abs(hi_val - 3.0f) < 1e-6f);

        std::cout << "  from_floats(1.5, 3.0) → to_floats → ("
                  << lo_val << ", " << hi_val << ") ... "
                  << (passed ? "✓" : "✗") << std::endl;

        if (!passed) all_passed = false;
    }

    // Test byte-level packing
    {
        nvfp4x2_t packed = nvfp4x2_t::from_floats(1.0f, 2.0f);
        // 1.0 = 0x2, 2.0 = 0x4, packed = 0x42
        uint8_t expected_byte = 0x42;

        bool passed = (packed.data == expected_byte);

        std::cout << "  Byte packing: 0x" << std::hex << (int)packed.data
                  << " (expected 0x" << (int)expected_byte << ")"
                  << std::dec << " ... "
                  << (passed ? "✓" : "✗") << std::endl;

        if (!passed) all_passed = false;
    }

    return all_passed;
}

/**
 * @brief Test lookup table conversion
 */
bool test_lookup_table() {
    std::cout << "\n=== Testing Lookup Table ===" << std::endl;

    bool all_passed = true;

    for (const auto& test : test_cases) {
        float result = NVFP4LookupTable::convert(test.fp4_bits);
        bool passed = (std::abs(result - test.expected_float) < 1e-6f);

        if (!passed) {
            std::cout << "  Lookup 0x" << std::hex << (int)test.fp4_bits << std::dec
                      << ": Expected " << test.expected_float
                      << ", Got " << result << " ... ✗" << std::endl;
            all_passed = false;
        }
    }

    if (all_passed) {
        std::cout << "  All 16 lookup table entries correct ✓" << std::endl;
    }

    return all_passed;
}

/**
 * @brief Test GPU kernels
 */
bool test_gpu_kernels() {
    std::cout << "\n=== Testing GPU Kernels ===" << std::endl;

    const int N = 16;  // Test with 16 floats (8 packed)
    const int N_packed = N / 2;

    // Allocate host memory
    float* h_input = new float[N];
    float* h_output = new float[N];

    // Create test data
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)(i - 8);  // -8 to +7
    }

    // Allocate device memory
    float* d_input;
    nvfp4x2_t* d_packed;
    float* d_output;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_packed, N_packed * sizeof(nvfp4x2_t));
    cudaMalloc(&d_output, N * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Pack float → FP4
    int threads = 256;
    int blocks = (N_packed + threads - 1) / threads;
    pack_float_to_nvfp4_kernel<<<blocks, threads>>>(d_input, d_packed, N);
    cudaDeviceSynchronize();

    // Unpack FP4 → float
    unpack_nvfp4_to_float_kernel<<<blocks, threads>>>(d_packed, d_output, N_packed);
    cudaDeviceSynchronize();

    // Copy output back
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results (FP4 has limited precision, expect quantization error)
    bool all_passed = true;
    for (int i = 0; i < N; ++i) {
        nvfp4_t fp4_expected = nvfp4_t::from_float(h_input[i]);
        float expected = fp4_expected.to_float();
        float error = std::abs(h_output[i] - expected);

        if (error > 1e-6f) {
            std::cout << "  Index " << i << ": Input " << h_input[i]
                      << " → Expected " << expected
                      << ", Got " << h_output[i] << " ... ✗" << std::endl;
            all_passed = false;
        }
    }

    if (all_passed) {
        std::cout << "  GPU pack/unpack kernels work correctly ✓" << std::endl;
    }

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_packed);
    cudaFree(d_output);

    return all_passed;
}

/**
 * @brief Main test runner
 */
int main() {
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║           NVFP4 Data Type Unit Tests                      ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;

    bool all_tests_passed = true;

    // Run all tests
    all_tests_passed &= test_fp4_to_float_conversions();
    all_tests_passed &= test_float_to_fp4_quantization();
    all_tests_passed &= test_packed_fp4();
    all_tests_passed &= test_lookup_table();
    all_tests_passed &= test_gpu_kernels();

    // Summary
    std::cout << "\n╔════════════════════════════════════════════════════════════╗" << std::endl;
    if (all_tests_passed) {
        std::cout << "║              ✅ ALL TESTS PASSED!                         ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
        std::cout << "\nNVFP4 data type is ready for GEMM kernel implementation." << std::endl;
        return 0;
    } else {
        std::cout << "║              ❌ SOME TESTS FAILED                         ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
        return 1;
    }
}
