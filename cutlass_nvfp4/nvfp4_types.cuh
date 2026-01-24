// SPDX-License-Identifier: Apache-2.0
/**
 * @file nvfp4_types.cuh
 * @brief NVFP4 (e2m1) data type implementation for GB10 Blackwell GPUs
 *
 * Implements 4-bit floating point format (NVFP4/e2m1):
 * - 1 sign bit
 * - 2 exponent bits
 * - 1 mantissa bit
 *
 * Format: [sign][exp1][exp0][mantissa]
 *         [  1  ][  1 ][  1 ][    1   ]
 *
 * Packed representation: 2 FP4 values per uint8 byte
 *   Byte layout: [FP4_hi (bits 7-4)][FP4_lo (bits 3-0)]
 *
 * Value ranges:
 *   Normalized: ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
 *   Subnormal: ±0.0, ±0.25
 *   Total: 16 distinct values (4 bits)
 *
 * Target: NVIDIA GB10 (SM_121, Blackwell architecture)
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <cmath>

namespace cutlass_nvfp4 {

/**
 * @brief Single NVFP4 value (4 bits)
 *
 * e2m1 format encoding:
 * - Exponent bias: 1
 * - Normal: (-1)^sign * 2^(exp-1) * (1 + mantissa/2)
 * - Subnormal (exp==0): (-1)^sign * 2^(-1) * (mantissa/2)
 */
struct __align__(1) nvfp4_t {
    uint8_t data : 4;  // 4-bit storage

    /// Default constructor
    __host__ __device__ __forceinline__
    nvfp4_t() : data(0) {}

    /// Construct from uint8_t
    __host__ __device__ __forceinline__
    explicit nvfp4_t(uint8_t val) : data(val & 0x0F) {}

    /**
     * @brief Convert FP4 to float
     *
     * Decoding table for all 16 possible values:
     *   0000 (0):  0.0      1000 (8): -0.0
     *   0001 (1):  0.25     1001 (9): -0.25
     *   0010 (2):  1.0      1010 (10): -1.0
     *   0011 (3):  1.5      1011 (11): -1.5
     *   0100 (4):  2.0      1100 (12): -2.0
     *   0101 (5):  3.0      1101 (13): -3.0
     *   0110 (6):  4.0      1110 (14): -4.0
     *   0111 (7):  6.0      1111 (15): -6.0
     */
    __host__ __device__ __forceinline__
    float to_float() const {
        // Extract bit fields
        uint8_t sign = (data >> 3) & 0x1;        // Bit 3
        uint8_t exp = (data >> 1) & 0x3;         // Bits 2-1
        uint8_t mantissa = data & 0x1;           // Bit 0

        float value;

        if (exp == 0) {
            // Subnormal or zero (exp == 0)
            if (mantissa == 0) {
                value = 0.0f;  // Zero
            } else {
                // Subnormal: 2^(-1) * (0 + mantissa/2) = 2^(-1) * 0.5 = 0.25
                value = 0.25f;
            }
        } else {
            // Normalized values
            // Formula: 2^(exp-1) * (1 + mantissa/2)

            float base = 1.0f + mantissa * 0.5f;  // 1.0 or 1.5

            // Exponent: 2^(exp-1)
            //   exp=1: 2^0 = 1.0
            //   exp=2: 2^1 = 2.0
            //   exp=3: 2^2 = 4.0
            float exponent_scale;
            switch (exp) {
                case 1: exponent_scale = 1.0f; break;  // base * 1 = {1.0, 1.5}
                case 2: exponent_scale = 2.0f; break;  // base * 2 = {2.0, 3.0}
                case 3: exponent_scale = 4.0f; break;  // base * 4 = {4.0, 6.0}
                default: exponent_scale = 1.0f; break; // Should never happen
            }

            value = base * exponent_scale;
        }

        // Apply sign
        return sign ? -value : value;
    }

    /**
     * @brief Convert FP4 to half precision (fp16)
     */
    __host__ __device__ __forceinline__
    __half to_half() const {
        return __float2half(to_float());
    }

    /**
     * @brief Quantize float to FP4 (nearest rounding)
     *
     * Quantization strategy:
     * - Find closest representable FP4 value
     * - Clamp to FP4 range [-6.0, 6.0]
     * - Use round-to-nearest-even
     */
    __host__ __device__ __forceinline__
    static nvfp4_t from_float(float val) {
        // Handle sign
        bool is_negative = (val < 0.0f);
        float abs_val = is_negative ? -val : val;

        uint8_t result_data;

        // Clamp to FP4 range [0, 6.0]
        if (abs_val >= 6.0f) {
            result_data = 0x7;  // ±6.0 (maximum)
        } else if (abs_val >= 4.5f) {
            result_data = 0x7;  // 6.0
        } else if (abs_val >= 3.5f) {
            result_data = 0x6;  // 4.0
        } else if (abs_val >= 2.5f) {
            result_data = 0x5;  // 3.0
        } else if (abs_val >= 1.75f) {
            result_data = 0x4;  // 2.0
        } else if (abs_val >= 1.25f) {
            result_data = 0x3;  // 1.5
        } else if (abs_val >= 0.75f) {
            result_data = 0x2;  // 1.0
        } else if (abs_val >= 0.125f) {
            result_data = 0x1;  // 0.25 (subnormal)
        } else {
            result_data = 0x0;  // 0.0
        }

        // Apply sign bit
        if (is_negative) {
            result_data |= 0x8;  // Set sign bit
        }

        return nvfp4_t(result_data);
    }

    /**
     * @brief Quantize half to FP4
     */
    __host__ __device__ __forceinline__
    static nvfp4_t from_half(__half val) {
        return from_float(__half2float(val));
    }
};

/**
 * @brief Packed FP4 pair (2 values in 1 byte)
 *
 * Layout: [hi (bits 7-4)][lo (bits 3-0)]
 *
 * This is the primary storage format for FP4 tensors,
 * achieving 2x compression vs FP8.
 */
struct __align__(1) nvfp4x2_t {
    uint8_t data;  // 8-bit storage for 2 FP4 values

    /// Default constructor
    __host__ __device__ __forceinline__
    nvfp4x2_t() : data(0) {}

    /// Construct from uint8_t
    __host__ __device__ __forceinline__
    explicit nvfp4x2_t(uint8_t val) : data(val) {}

    /// Construct from two FP4 values
    __host__ __device__ __forceinline__
    nvfp4x2_t(nvfp4_t lo, nvfp4_t hi)
        : data((hi.data << 4) | (lo.data & 0x0F)) {}

    /**
     * @brief Extract low FP4 value (bits 3-0)
     */
    __host__ __device__ __forceinline__
    nvfp4_t lo() const {
        return nvfp4_t(data & 0x0F);
    }

    /**
     * @brief Extract high FP4 value (bits 7-4)
     */
    __host__ __device__ __forceinline__
    nvfp4_t hi() const {
        return nvfp4_t((data >> 4) & 0x0F);
    }

    /**
     * @brief Set low FP4 value
     */
    __host__ __device__ __forceinline__
    void set_lo(nvfp4_t val) {
        data = (data & 0xF0) | (val.data & 0x0F);
    }

    /**
     * @brief Set high FP4 value
     */
    __host__ __device__ __forceinline__
    void set_hi(nvfp4_t val) {
        data = (data & 0x0F) | ((val.data & 0x0F) << 4);
    }

    /**
     * @brief Pack two floats into FP4x2
     */
    __host__ __device__ __forceinline__
    static nvfp4x2_t from_floats(float lo_val, float hi_val) {
        nvfp4_t lo = nvfp4_t::from_float(lo_val);
        nvfp4_t hi = nvfp4_t::from_float(hi_val);
        return nvfp4x2_t(lo, hi);
    }

    /**
     * @brief Unpack to two floats
     */
    __host__ __device__ __forceinline__
    void to_floats(float& lo_val, float& hi_val) const {
        lo_val = lo().to_float();
        hi_val = hi().to_float();
    }
};

/**
 * @brief Lookup table for fast FP4 → float conversion
 *
 * Precomputed for all 16 possible FP4 values.
 * Can be used for vectorized unpacking.
 */
struct NVFP4LookupTable {
    /**
     * @brief Fast FP4 → float conversion using lookup
     */
    __host__ __device__ __forceinline__
    static float convert(uint8_t fp4_data) {
        // Lookup table stored as immediate values (compiler will optimize)
        constexpr float values[16] = {
            0.0f,   // 0000
            0.25f,  // 0001
            1.0f,   // 0010
            1.5f,   // 0011
            2.0f,   // 0100
            3.0f,   // 0101
            4.0f,   // 0110
            6.0f,   // 0111
            -0.0f,  // 1000
            -0.25f, // 1001
            -1.0f,  // 1010
            -1.5f,  // 1011
            -2.0f,  // 1100
            -3.0f,  // 1101
            -4.0f,  // 1110
            -6.0f   // 1111
        };
        return values[fp4_data & 0x0F];
    }
};

/**
 * @brief Helper functions for tensor operations
 */

/**
 * @brief Unpack uint8 array to float array (2x expansion)
 *
 * @param packed Input array of packed FP4 values (nvfp4x2_t)
 * @param unpacked Output array of floats (2x size of packed)
 * @param count Number of packed elements
 */
__global__ void unpack_nvfp4_to_float_kernel(
    const nvfp4x2_t* __restrict__ packed,
    float* __restrict__ unpacked,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        nvfp4x2_t packed_val = packed[idx];
        unpacked[idx * 2] = packed_val.lo().to_float();
        unpacked[idx * 2 + 1] = packed_val.hi().to_float();
    }
}

/**
 * @brief Pack float array to uint8 array (2x compression)
 *
 * @param unpacked Input array of floats
 * @param packed Output array of packed FP4 values
 * @param count Number of unpacked elements (must be even)
 */
__global__ void pack_float_to_nvfp4_kernel(
    const float* __restrict__ unpacked,
    nvfp4x2_t* __restrict__ packed,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int packed_idx = idx;

    if (idx * 2 < count) {
        float lo_val = unpacked[idx * 2];
        float hi_val = (idx * 2 + 1 < count) ? unpacked[idx * 2 + 1] : 0.0f;
        packed[packed_idx] = nvfp4x2_t::from_floats(lo_val, hi_val);
    }
}

} // namespace cutlass_nvfp4
