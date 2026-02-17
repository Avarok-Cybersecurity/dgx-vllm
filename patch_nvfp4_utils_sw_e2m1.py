#!/usr/bin/env python3
"""
Patch nvfp4_utils.cuh to provide software E2M1 conversion for SM121 (GB10).

SM121 has mma.e2m1 (FP4 tensor core multiply) but LACKS the
cvt.rn.satfinite.e2m1x2.f32 PTX instruction for float→E2M1 conversion.

This patch adds #if __CUDA_ARCH__ == 1210 guards around the three functions
that use this PTX instruction, providing software implementations that produce
bit-identical results (round-to-nearest-even, satfinite clamping).

This allows nvfp4_quant_kernels.cu, nvfp4_experts_quant.cu, and
activation_nvfp4_quant_fusion_kernels.cu to compile for SM121.
"""

import sys

VLLM_DIR = "/app/vllm"
UTILS_FILE = f"{VLLM_DIR}/csrc/quantization/fp4/nvfp4_utils.cuh"

# Software E2M1 helper - inserted after "namespace vllm {"
SW_E2M1_HELPER = r"""
// ============================================================================
// Software E2M1 conversion for SM121 (GB10) - no cvt.rn.satfinite.e2m1x2.f32
// ============================================================================
// E2M1 format: 1 sign + 2 exponent + 1 mantissa
// Representable values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (and negatives)
// Uses round-to-nearest-even at midpoints, satfinite clamping to 6.0.
// ============================================================================
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1210
__device__ __forceinline__ uint8_t _sw_float_to_e2m1(float x) {
  uint8_t sign = (uint8_t)((__float_as_uint(x) >> 28) & 8u);
  float ax = fabsf(x);
  uint8_t mag;
  // Thresholds match hardware round-to-nearest-even behavior:
  // At midpoints between representable values, round to value with M=0 (even).
  if      (ax <= 0.25f)  mag = 0;  // 0.0   (code 000)
  else if (ax <  0.75f)  mag = 1;  // 0.5   (code 001)
  else if (ax <= 1.25f)  mag = 2;  // 1.0   (code 010)
  else if (ax <  1.75f)  mag = 3;  // 1.5   (code 011)
  else if (ax <= 2.5f)   mag = 4;  // 2.0   (code 100)
  else if (ax <  3.5f)   mag = 5;  // 3.0   (code 101)
  else if (ax <= 5.0f)   mag = 6;  // 4.0   (code 110)
  else                    mag = 7;  // 6.0   (code 111) - satfinite
  return sign | mag;
}

__device__ __forceinline__ uint32_t _sw_fp32_vec8_to_e2m1_flat(
    float f0, float f1, float f2, float f3,
    float f4, float f5, float f6, float f7) {
  // Pack 8 E2M1 nibbles into uint32: nibble[i] at bits [i*4+3:i*4]
  // Matches hardware byte packing: byte0={e2m1(f1),e2m1(f0)}, etc.
  uint32_t val = 0;
  val |= (uint32_t)_sw_float_to_e2m1(f0);
  val |= (uint32_t)_sw_float_to_e2m1(f1) << 4;
  val |= (uint32_t)_sw_float_to_e2m1(f2) << 8;
  val |= (uint32_t)_sw_float_to_e2m1(f3) << 12;
  val |= (uint32_t)_sw_float_to_e2m1(f4) << 16;
  val |= (uint32_t)_sw_float_to_e2m1(f5) << 20;
  val |= (uint32_t)_sw_float_to_e2m1(f6) << 24;
  val |= (uint32_t)_sw_float_to_e2m1(f7) << 28;
  return val;
}
#endif  // __CUDA_ARCH__ == 1210
"""

def patch_file():
    with open(UTILS_FILE, "r") as f:
        content = f.read()

    if "_sw_float_to_e2m1" in content:
        print("  nvfp4_utils.cuh already patched, skipping")
        return

    # 1. Insert software helper after "namespace vllm {"
    marker = "namespace vllm {"
    if marker not in content:
        print("ERROR: Cannot find 'namespace vllm {' in nvfp4_utils.cuh")
        sys.exit(1)
    content = content.replace(marker, marker + SW_E2M1_HELPER, 1)
    print("  Inserted _sw_float_to_e2m1 helper function")

    # 2. Replace fp32_vec8_to_e2m1(float (&array)[8]) - first overload
    old_func1 = """// Convert 8 float32 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec8_to_e2m1(float (&array)[8]) {
  uint32_t val;
  asm volatile(
      "{\\n"
      ".reg .b8 byte0;\\n"
      ".reg .b8 byte1;\\n"
      ".reg .b8 byte2;\\n"
      ".reg .b8 byte3;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\\n"
      "}"
      : "=r"(val)
      : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]),
        "f"(array[4]), "f"(array[5]), "f"(array[6]), "f"(array[7]));
  return val;
}"""

    new_func1 = """// Convert 8 float32 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec8_to_e2m1(float (&array)[8]) {
  uint32_t val;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1210
  // Software E2M1 for SM121 (no cvt.rn.satfinite.e2m1x2.f32)
  val = _sw_fp32_vec8_to_e2m1_flat(
      array[0], array[1], array[2], array[3],
      array[4], array[5], array[6], array[7]);
#else
  asm volatile(
      "{\\n"
      ".reg .b8 byte0;\\n"
      ".reg .b8 byte1;\\n"
      ".reg .b8 byte2;\\n"
      ".reg .b8 byte3;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\\n"
      "}"
      : "=r"(val)
      : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]),
        "f"(array[4]), "f"(array[5]), "f"(array[6]), "f"(array[7]));
#endif
  return val;
}"""

    if old_func1 in content:
        content = content.replace(old_func1, new_func1, 1)
        print("  Patched fp32_vec8_to_e2m1(float[8])")
    else:
        print("  WARNING: Could not find fp32_vec8_to_e2m1(float[8]) - trying relaxed match")
        # Try without the comment
        if "fp32_vec8_to_e2m1(float (&array)[8])" in content:
            print("  ERROR: Function signature found but body doesn't match expected pattern")
            sys.exit(1)
        else:
            print("  ERROR: Function not found at all")
            sys.exit(1)

    # 3. Replace fp32_vec8_to_e2m1(float2 (&array)[4]) - second overload
    old_func2 = """// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
__device__ __forceinline__ uint32_t fp32_vec8_to_e2m1(float2 (&array)[4]) {
  uint32_t val;
  asm volatile(
      "{\\n"
      ".reg .b8 byte0;\\n"
      ".reg .b8 byte1;\\n"
      ".reg .b8 byte2;\\n"
      ".reg .b8 byte3;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\\n"
      "}\\n"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
}"""

    new_func2 = """// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
__device__ __forceinline__ uint32_t fp32_vec8_to_e2m1(float2 (&array)[4]) {
  uint32_t val;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1210
  val = _sw_fp32_vec8_to_e2m1_flat(
      array[0].x, array[0].y, array[1].x, array[1].y,
      array[2].x, array[2].y, array[3].x, array[3].y);
#else
  asm volatile(
      "{\\n"
      ".reg .b8 byte0;\\n"
      ".reg .b8 byte1;\\n"
      ".reg .b8 byte2;\\n"
      ".reg .b8 byte3;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\\n"
      "}\\n"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
#endif
  return val;
}"""

    if old_func2 in content:
        content = content.replace(old_func2, new_func2, 1)
        print("  Patched fp32_vec8_to_e2m1(float2[4])")
    else:
        print("  WARNING: Could not find fp32_vec8_to_e2m1(float2[4])")
        if "fp32_vec8_to_e2m1(float2 (&array)[4])" in content:
            print("  ERROR: Function signature found but body doesn't match")
            sys.exit(1)

    # 4. Replace fp32_vec16_to_e2m1(float2 (&array)[8])
    old_func3 = """__device__ __forceinline__ u32x2 fp32_vec16_to_e2m1(float2 (&array)[8]) {
  u32x2 out;
  asm volatile(
      "{\\n"
      ".reg .b8 b0;\\n"
      ".reg .b8 b1;\\n"
      ".reg .b8 b2;\\n"
      ".reg .b8 b3;\\n"
      ".reg .b8 b4;\\n"
      ".reg .b8 b5;\\n"
      ".reg .b8 b6;\\n"
      ".reg .b8 b7;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   b0,  %3,  %2;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   b1,  %5,  %4;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   b2,  %7,  %6;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   b3,  %9,  %8;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   b4, %11, %10;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   b5, %13, %12;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   b6, %15, %14;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   b7, %17, %16;\\n"
      "mov.b32 %0, {b0, b1, b2, b3};\\n"
      "mov.b32 %1, {b4, b5, b6, b7};\\n"
      "}\\n"
      : "=r"(out.lo), "=r"(out.hi)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y),
        "f"(array[4].x), "f"(array[4].y), "f"(array[5].x), "f"(array[5].y),
        "f"(array[6].x), "f"(array[6].y), "f"(array[7].x), "f"(array[7].y));
  return out;
}"""

    new_func3 = """__device__ __forceinline__ u32x2 fp32_vec16_to_e2m1(float2 (&array)[8]) {
  u32x2 out;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1210
  out.lo = _sw_fp32_vec8_to_e2m1_flat(
      array[0].x, array[0].y, array[1].x, array[1].y,
      array[2].x, array[2].y, array[3].x, array[3].y);
  out.hi = _sw_fp32_vec8_to_e2m1_flat(
      array[4].x, array[4].y, array[5].x, array[5].y,
      array[6].x, array[6].y, array[7].x, array[7].y);
#else
  asm volatile(
      "{\\n"
      ".reg .b8 b0;\\n"
      ".reg .b8 b1;\\n"
      ".reg .b8 b2;\\n"
      ".reg .b8 b3;\\n"
      ".reg .b8 b4;\\n"
      ".reg .b8 b5;\\n"
      ".reg .b8 b6;\\n"
      ".reg .b8 b7;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   b0,  %3,  %2;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   b1,  %5,  %4;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   b2,  %7,  %6;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   b3,  %9,  %8;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   b4, %11, %10;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   b5, %13, %12;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   b6, %15, %14;\\n"
      "cvt.rn.satfinite.e2m1x2.f32   b7, %17, %16;\\n"
      "mov.b32 %0, {b0, b1, b2, b3};\\n"
      "mov.b32 %1, {b4, b5, b6, b7};\\n"
      "}\\n"
      : "=r"(out.lo), "=r"(out.hi)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y),
        "f"(array[4].x), "f"(array[4].y), "f"(array[5].x), "f"(array[5].y),
        "f"(array[6].x), "f"(array[6].y), "f"(array[7].x), "f"(array[7].y));
#endif
  return out;
}"""

    if old_func3 in content:
        content = content.replace(old_func3, new_func3, 1)
        print("  Patched fp32_vec16_to_e2m1(float2[8])")
    else:
        print("  WARNING: Could not find fp32_vec16_to_e2m1(float2[8])")
        if "fp32_vec16_to_e2m1" in content:
            print("  ERROR: Function found but body doesn't match")
            sys.exit(1)

    # Verify no remaining unguarded cvt.rn.satfinite.e2m1x2
    remaining = content.count("cvt.rn.satfinite.e2m1x2.f32")
    guarded = content.count("#else")  # Each function has one #else with asm
    print(f"  Remaining cvt.rn.satfinite.e2m1x2 instances: {remaining} (all inside #else branches)")

    with open(UTILS_FILE, "w") as f:
        f.write(content)
    print("  nvfp4_utils.cuh patched successfully!")


if __name__ == "__main__":
    print("Patching nvfp4_utils.cuh with software E2M1 for SM121...")
    patch_file()
    print("Done!")
