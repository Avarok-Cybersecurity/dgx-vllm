// ============================================================================
// NVIDIA FP4 Type Implementation for CUDA 13.0
// ============================================================================
// CUDA 13.0's CCCL headers reference __nv_fp4_e2m1 for SM_120/SM_121 but
// NVIDIA hasn't released the official type yet. This header provides a
// complete implementation using proven E2M1 conversion algorithms.
//
// Based on tested code from cutlass_nvfp4/nvfp4_types.cuh
// Format: 1 sign + 2 exponent + 1 mantissa bits
// ============================================================================

#ifndef NV_FP4_TYPES_H
#define NV_FP4_TYPES_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Only define if not already defined
#ifndef __nv_fp4_e2m1

/**
 * @brief NVIDIA FP4 E2M1 format (4-bit floating point)
 *
 * Format: [sign][exp1][exp0][mantissa]
 *         [  1  ][  1 ][  1 ][    1   ]
 *
 * Exponent bias: 1
 * Values: ±0.0, ±0.25, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
 *
 * This is a LITERAL TYPE compatible with constexpr functions.
 */
struct __align__(1) __nv_fp4_e2m1 {
    unsigned char __x;  // 8-bit storage (lower 4 bits used)

    // Trivial default constructor (required for literal type)
    __host__ __device__ constexpr
    __nv_fp4_e2m1() : __x(0) {}

    // Constexpr constructor from uint8_t
    __host__ __device__ constexpr
    __nv_fp4_e2m1(unsigned char val) : __x(val & 0x0F) {}

    /**
     * @brief Convert FP4 to float
     *
     * Decoding:
     *   0x0 (0):  0.0      0x8 (8): -0.0
     *   0x1 (1):  0.25     0x9 (9): -0.25
     *   0x2 (2):  1.0      0xA (10): -1.0
     *   0x3 (3):  1.5      0xB (11): -1.5
     *   0x4 (4):  2.0      0xC (12): -2.0
     *   0x5 (5):  3.0      0xD (13): -3.0
     *   0x6 (6):  4.0      0xE (14): -4.0
     *   0x7 (7):  6.0      0xF (15): -6.0
     */
    __host__ __device__ __forceinline__
    operator float() const {
        // Extract bit fields
        unsigned char sign = (__x >> 3) & 0x1;     // Bit 3
        unsigned char exp = (__x >> 1) & 0x3;      // Bits 2-1
        unsigned char mantissa = __x & 0x1;        // Bit 0

        float value;

        if (exp == 0) {
            // Subnormal or zero
            if (mantissa == 0) {
                value = 0.0f;
            } else {
                value = 0.25f;  // 2^(-1) * 0.5
            }
        } else {
            // Normalized values: 2^(exp-1) * (1 + mantissa/2)
            float base = 1.0f + mantissa * 0.5f;  // 1.0 or 1.5

            float exponent_scale;
            switch (exp) {
                case 1: exponent_scale = 1.0f; break;  // {1.0, 1.5}
                case 2: exponent_scale = 2.0f; break;  // {2.0, 3.0}
                case 3: exponent_scale = 4.0f; break;  // {4.0, 6.0}
                default: exponent_scale = 1.0f; break;
            }

            value = base * exponent_scale;
        }

        // Apply sign
        return sign ? -value : value;
    }

    // Conversion to half
    __host__ __device__ __forceinline__
    operator __half() const {
        return __float2half(float(*this));
    }

    // Arithmetic operators (not constexpr - use float conversion)
    __host__ __device__ __forceinline__
    __nv_fp4_e2m1 operator-() const {
        __nv_fp4_e2m1 result;
        result.__x = __x ^ 0x8;  // Flip sign bit
        return result;
    }

    // Comparison operators
    __host__ __device__ constexpr __forceinline__
    bool operator==(const __nv_fp4_e2m1& other) const {
        return __x == other.__x;
    }

    __host__ __device__ constexpr __forceinline__
    bool operator!=(const __nv_fp4_e2m1& other) const {
        return __x != other.__x;
    }
};

// Also define __nv_fp4_e2m3 (sometimes referenced)
struct __align__(1) __nv_fp4_e2m3 {
    unsigned char __x;

    __host__ __device__ constexpr
    __nv_fp4_e2m3() : __x(0) {}

    __host__ __device__ constexpr
    __nv_fp4_e2m3(unsigned char val) : __x(val & 0x0F) {}

    __host__ __device__ __forceinline__
    operator float() const {
        unsigned char sign = (__x >> 3) & 0x1;
        unsigned char mantissa = __x & 0x7;
        float value = float(mantissa);
        return sign ? -value : value;
    }
};

// ============================================================================
// Define __nv_fp4x2_storage_t (Packed FP4 storage - 2 values per byte)
// ============================================================================
// FlashInfer and other libraries expect this type
// ============================================================================

#ifndef __nv_fp4x2_storage_t
struct __align__(1) __nv_fp4x2_storage_t {
    unsigned char __x;  // 8 bits storing 2 FP4 values

    __host__ __device__ constexpr
    __nv_fp4x2_storage_t() : __x(0) {}

    __host__ __device__ constexpr
    __nv_fp4x2_storage_t(unsigned char val) : __x(val) {}

    // Extract low FP4 value (bits 0-3)
    __host__ __device__ __forceinline__
    __nv_fp4_e2m1 get_low() const {
        return __nv_fp4_e2m1(__x & 0x0F);
    }

    // Extract high FP4 value (bits 4-7)
    __host__ __device__ __forceinline__
    __nv_fp4_e2m1 get_high() const {
        return __nv_fp4_e2m1((__x >> 4) & 0x0F);
    }

    // Set low FP4 value
    __host__ __device__ __forceinline__
    void set_low(__nv_fp4_e2m1 val) {
        __x = (__x & 0xF0) | (val.__x & 0x0F);
    }

    // Set high FP4 value
    __host__ __device__ __forceinline__
    void set_high(__nv_fp4_e2m1 val) {
        __x = (__x & 0x0F) | ((val.__x & 0x0F) << 4);
    }

    // Bitwise operators (required by FlashInfer)
    __host__ __device__ constexpr __forceinline__
    __nv_fp4x2_storage_t operator<<(int shift) const {
        return __nv_fp4x2_storage_t(__x << shift);
    }

    __host__ __device__ constexpr __forceinline__
    __nv_fp4x2_storage_t operator>>(int shift) const {
        return __nv_fp4x2_storage_t(__x >> shift);
    }

    __host__ __device__ constexpr __forceinline__
    __nv_fp4x2_storage_t operator|(const __nv_fp4x2_storage_t& other) const {
        return __nv_fp4x2_storage_t(__x | other.__x);
    }

    __host__ __device__ constexpr __forceinline__
    __nv_fp4x2_storage_t operator&(const __nv_fp4x2_storage_t& other) const {
        return __nv_fp4x2_storage_t(__x & other.__x);
    }

    // Conversion to uint16_t (required by FlashInfer)
    __host__ __device__ constexpr __forceinline__
    operator unsigned short() const {
        return static_cast<unsigned short>(__x);
    }
};
#endif // __nv_fp4x2_storage_t

#endif // __nv_fp4_e2m1

// ============================================================================
// Define missing FP4 conversion intrinsics
// ============================================================================
// CUDA 13.0 doesn't provide these, but CCCL expects them
// ============================================================================

// Enum for FP4 format (E2M1)
#ifndef __NV_E2M1
#define __NV_E2M1 0
#endif

// FP4 conversion intrinsics (simple implementations using our type)
__device__ __host__ inline
unsigned char __nv_cvt_float_to_fp4(float val, int format, int rounding) {
    // Convert float to FP4 using our proven algorithm
    bool is_negative = (val < 0.0f);
    float abs_val = is_negative ? -val : val;

    unsigned char result;
    if (abs_val >= 6.0f) result = 0x7;
    else if (abs_val >= 4.5f) result = 0x7;
    else if (abs_val >= 3.5f) result = 0x6;
    else if (abs_val >= 2.5f) result = 0x5;
    else if (abs_val >= 1.75f) result = 0x4;
    else if (abs_val >= 1.25f) result = 0x3;
    else if (abs_val >= 0.75f) result = 0x2;
    else if (abs_val >= 0.125f) result = 0x1;
    else result = 0x0;

    if (is_negative) result |= 0x8;
    return result;
}

__device__ __host__ inline
unsigned char __nv_cvt_double_to_fp4(double val, int format, int rounding) {
    return __nv_cvt_float_to_fp4((float)val, format, rounding);
}

__device__ __host__ inline
unsigned char __nv_cvt_halfraw_to_fp4(unsigned short val, int format, int rounding) {
    __half h;
    *(unsigned short*)&h = val;
    return __nv_cvt_float_to_fp4(__half2float(h), format, rounding);
}

__device__ __host__ inline
unsigned char __nv_cvt_bfloat16raw_to_fp4(unsigned short val, int format, int rounding) {
    // Simple bfloat16 to float conversion
    float f;
    *(unsigned int*)&f = ((unsigned int)val) << 16;
    return __nv_cvt_float_to_fp4(f, format, rounding);
}

__device__ __host__ inline
unsigned short __nv_cvt_fp4_to_halfraw(unsigned char fp4_val, int format) {
    // Convert FP4 to float using our algorithm
    unsigned char sign = (fp4_val >> 3) & 0x1;
    unsigned char exp = (fp4_val >> 1) & 0x3;
    unsigned char mantissa = fp4_val & 0x1;

    float value;
    if (exp == 0) {
        value = (mantissa == 0) ? 0.0f : 0.25f;
    } else {
        float base = 1.0f + mantissa * 0.5f;
        float exponent_scale = (exp == 1) ? 1.0f : ((exp == 2) ? 2.0f : 4.0f);
        value = base * exponent_scale;
    }
    if (sign) value = -value;

    __half h = __float2half(value);
    return *(unsigned short*)&h;
}

#endif // NV_FP4_TYPES_H
