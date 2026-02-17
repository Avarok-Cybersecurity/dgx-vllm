# SPDX-License-Identifier: Apache-2.0
"""
Software NVFP4 (E2M1) quantization for GB10 (SM_121).

GB10 lacks the cvt.rn.satfinite.e2m1x2.f32 PTX instruction for hardware
FP4 conversion. This module provides software implementations using PyTorch
ops that produce identical packed FP4 output suitable for CUTLASS FP4 GEMM
tensor cores (mma.e2m1, which GB10 DOES support).

Replaces the C++ stubs in nvfp4_quant_entry.cu for runtime execution.
"""

import torch

FLOAT4_E2M1_MAX = 6.0


def _float_to_e2m1_nibble(x: torch.Tensor) -> torch.Tensor:
    """Convert float32 values (pre-scaled, in [-6,6]) to E2M1 4-bit nibbles.

    E2M1 format (4-bit): 1 sign + 2 exponent + 1 mantissa
    Representable values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (and negatives)

    Uses round-to-nearest-even (matching hardware cvt.e2m1x2 behavior).
    Returns uint8 tensor with values 0-15 (4-bit nibbles).
    """
    ax = x.abs()
    sign = (x < 0).to(torch.uint8) * 8  # bit 3 = sign

    # Threshold-based nearest rounding (matches nvfp4_emulation_utils.cast_to_fp4)
    # Round-to-even at midpoints: 0.25->0, 0.75->1.0, 1.25->1.0, 1.75->2.0, etc.
    mag = torch.zeros_like(ax, dtype=torch.uint8)
    mag[(ax > 0.25) & (ax < 0.75)] = 1    # -> 0.5
    mag[(ax >= 0.75) & (ax <= 1.25)] = 2   # -> 1.0
    mag[(ax > 1.25) & (ax < 1.75)] = 3     # -> 1.5
    mag[(ax >= 1.75) & (ax <= 2.5)] = 4    # -> 2.0
    mag[(ax > 2.5) & (ax < 3.5)] = 5       # -> 3.0
    mag[(ax >= 3.5) & (ax <= 5.0)] = 6     # -> 4.0
    mag[ax > 5.0] = 7                       # -> 6.0

    return sign + mag


def _pack_fp4_pairs(nibbles: torch.Tensor) -> torch.Tensor:
    """Pack pairs of E2M1 nibbles into uint8 bytes.

    Input:  (..., N) uint8 with values 0-15
    Output: (..., N//2) uint8 with low nibble = even index, high nibble = odd
    """
    n = nibbles.shape[-1]
    assert n % 2 == 0
    paired = nibbles.reshape(*nibbles.shape[:-1], n // 2, 2)
    return paired[..., 0] | (paired[..., 1] << 4)


def _convert_linear_to_swizzled(
    scales_fp8: torch.Tensor, m: int, k: int, block_size: int = 16
) -> torch.Tensor:
    """Convert linear scale factor layout to NVIDIA tensor core swizzled layout.

    See: https://docs.nvidia.com/cuda/parallel-thread-execution/
         #tcgen05-mma-scale-factor-b-layout-4x

    Args:
        scales_fp8: (m, k // block_size) FP8 E4M3 scale factors
        m, k: original input dimensions
        block_size: quantization block size (16)

    Returns:
        (rounded_m, rounded_n // 4) int32 tensor (4 packed FP8 values each)
    """
    scale_n = k // block_size

    round_up = lambda x, y: (x + y - 1) // y * y
    rounded_m = round_up(m, 128)
    rounded_n = round_up(scale_n, 4)

    m_tiles = rounded_m // 128
    f = block_size * 4  # 64
    k_tiles = (k + f - 1) // f

    # Pad to (rounded_m, rounded_n)
    padded = torch.zeros(
        rounded_m, rounded_n,
        dtype=scales_fp8.dtype, device=scales_fp8.device,
    )
    padded[:m, :scale_n] = scales_fp8

    # Reshape to linear tile layout: (1, m_tiles, 4, 32, k_tiles, 4)
    linear = padded.reshape(1, m_tiles, 4, 32, k_tiles, 4)

    # Apply swizzle permutation (self-inverse: dims 2 and 4 swap)
    swizzled = linear.permute(0, 1, 4, 3, 2, 5).contiguous()

    # Pack as int32 (4 FP8 values per int32)
    swizzled_flat = swizzled.reshape(rounded_m, rounded_n)
    return swizzled_flat.view(torch.uint8).view(torch.int32).reshape(
        rounded_m, rounded_n // 4
    )


def _quantize_block_fp4(
    x: torch.Tensor, global_scale: torch.Tensor, block_size: int = 16
):
    """Core FP4 quantization: converts BF16/FP16 input to packed FP4 + FP8 scales.

    Algorithm (matches nvfp4_quant_kernels.cu):
    1. Reshape to blocks of `block_size` elements
    2. Compute per-block max absolute value
    3. Compute block scale = global_scale * (max / 6.0)
    4. Cast scale to FP8 E4M3
    5. Quantize: input * (global_scale / scale_fp8), clamp to [-6, 6]
    6. Round to nearest E2M1 value
    7. Pack pairs into uint8

    Args:
        x: (m, n) BF16/FP16 input tensor
        global_scale: scalar tensor (global scale inverse, typically 1/GS)
        block_size: quantization block size (16)

    Returns:
        (packed_fp4, scales_fp8) where:
        - packed_fp4: (m, n//2) uint8 tensor
        - scales_fp8: (m, n//block_size) FP8 E4M3 tensor
    """
    m, n = x.shape
    assert n % block_size == 0

    x_f32 = x.to(torch.float32).reshape(m, n // block_size, block_size)

    # Ensure global_scale is a proper scalar for broadcasting.
    # The C++ kernel extracts a single float (data_ptr<float>()[0]).
    # For per-expert scales (shape (n_experts,)), use first element.
    gs_flat = global_scale.to(torch.float32).flatten()
    gs = gs_flat[0:1]  # shape (1,) - broadcasts with any shape

    # Per-block max
    vec_max = x_f32.abs().amax(dim=-1, keepdim=True)  # (m, n//16, 1)

    # Block scale = global_scale * max / FP4_MAX
    scale = gs * (vec_max / FLOAT4_E2M1_MAX)  # (m, n//16, 1)
    scale = scale.clamp(min=-448, max=448)
    scale_fp8 = scale.to(torch.float8_e4m3fn)
    scale_f32 = scale_fp8.to(torch.float32)

    # Reciprocal scale for quantization: global_scale / scale
    recip = torch.where(
        scale_f32 == 0,
        torch.zeros_like(scale_f32),
        gs / scale_f32,
    )  # (m, n//16, 1)

    # Scale and clamp
    scaled_x = (x_f32 * recip).clamp(-6.0, 6.0).reshape(m, n)

    # Convert to E2M1 nibbles and pack
    nibbles = _float_to_e2m1_nibble(scaled_x)
    packed = _pack_fp4_pairs(nibbles)

    return packed, scale_fp8.squeeze(-1)


def software_scaled_fp4_quant(
    output: torch.Tensor,
    input: torch.Tensor,
    output_sf: torch.Tensor,
    input_global_scale: torch.Tensor,
    is_sf_swizzled_layout: bool,
) -> None:
    """Software replacement for torch.ops._C.scaled_fp4_quant on GB10.

    Fills pre-allocated output and output_sf tensors in-place.
    """
    m, n = input.shape
    packed, scales_fp8 = _quantize_block_fp4(input, input_global_scale)

    output.copy_(packed)

    if is_sf_swizzled_layout:
        swizzled = _convert_linear_to_swizzled(scales_fp8, m, n)
        # output_sf is (rounded_m, rounded_n // 4) int32
        output_sf.copy_(swizzled[:output_sf.shape[0], :output_sf.shape[1]])
    else:
        output_sf.copy_(scales_fp8.view(torch.uint8)[:output_sf.shape[0], :output_sf.shape[1]])


def software_scaled_fp4_experts_quant(
    output: torch.Tensor,
    output_scales: torch.Tensor,
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
) -> None:
    """Software replacement for torch.ops._C.scaled_fp4_experts_quant on GB10.

    Quantizes per-expert slices with per-expert global scales.
    """
    m_numtopk, k = input_tensor.shape
    n_experts = expert_offsets.shape[0] - 1

    for i in range(n_experts):
        start = expert_offsets[i].item()
        end = expert_offsets[i + 1].item()
        if start >= end:
            continue

        expert_input = input_tensor[start:end]

        # Get per-expert global scale (scalar)
        if input_global_scale.ndim >= 1 and input_global_scale.numel() > 1:
            expert_scale = input_global_scale[i].reshape(1)
        else:
            expert_scale = input_global_scale.reshape(1)

        packed, scales_fp8 = _quantize_block_fp4(expert_input, expert_scale)
        output[start:end].copy_(packed)

        # Write scales at blockscale offsets
        bs_start = blockscale_offsets[i].item()
        bs_end = blockscale_offsets[i + 1].item()
        num_tokens = end - start

        # Swizzle per-expert scales
        swizzled = _convert_linear_to_swizzled(scales_fp8, num_tokens, k)
        # Write into output_scales at the correct offset
        out_rows = min(swizzled.shape[0], bs_end - bs_start)
        out_cols = min(swizzled.shape[1], output_scales.shape[1])
        output_scales[bs_start:bs_start + out_rows, :out_cols].copy_(
            swizzled[:out_rows, :out_cols]
        )


def software_silu_and_mul_scaled_fp4_experts_quant(
    output: torch.Tensor,
    output_scales: torch.Tensor,
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
) -> None:
    """Software replacement for torch.ops._C.silu_and_mul_scaled_fp4_experts_quant.

    Fuses SiLU activation + multiply + FP4 quantization for MoE gate||up layout.
    Input: (m_numtopk, k*2) with gate||up concatenated
    Output: (m_numtopk, k//2) packed FP4
    """
    m_numtopk, k_times_2 = input_tensor.shape
    k = k_times_2 // 2

    # Split gate and up projections
    gate = input_tensor[:, :k]
    up = input_tensor[:, k:]

    # SiLU(gate) * up
    activated = torch.nn.functional.silu(gate) * up

    # Quantize the activated output
    software_scaled_fp4_experts_quant(
        output, output_scales, activated,
        input_global_scale, expert_offsets, blockscale_offsets,
    )
