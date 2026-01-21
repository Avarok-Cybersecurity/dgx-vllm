"""
Pure Triton backend for FP8 scaled matrix multiplication on SM_121 (GB10).

This module provides a CUTLASS-free implementation using Triton kernels,
specifically for NVIDIA GB10 hardware which lacks CUTLASS runtime support.
"""

from typing import Optional

import torch

from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
    ScaledMMLinearKernel,
)


class TritonPureScaledMMLinearKernel(ScaledMMLinearKernel):
    """
    Pure Triton implementation of scaled matrix multiplication for FP8.

    This kernel uses vLLM's Triton-based FP8 implementation instead of CUTLASS,
    making it suitable for hardware without CUTLASS support (e.g., GB10/SM_121).

    Features:
    - FP8 (e4m3fn) support with per-tensor and per-channel scaling
    - Auto-tuned block sizes for optimal performance
    - No CUTLASS dependency
    - Tested and working on ROCm (adapted for CUDA SM_121)
    """

    @classmethod
    def get_min_capability(cls) -> int:
        """Minimum compute capability required (7.5 for Triton)."""
        return 75

    @classmethod
    def can_implement(
        cls,
        c: "CompressedTensorsConfig",  # type: ignore
        capability: Optional[int] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if this kernel can implement the given configuration.

        Args:
            c: Compressed tensors configuration
            capability: Optional compute capability override

        Returns:
            (can_implement, reason) tuple
        """
        # Check base requirements
        can_impl, reason = super().can_implement(c, capability)
        if not can_impl:
            return False, reason

        # Triton supports FP8 with both static and dynamic input schemes
        if c.quantization_config.activation_scheme not in [
            "static",
            "dynamic",
        ]:
            return False, (
                f"Activation scheme {c.quantization_config.activation_scheme} "
                "not supported by Triton backend"
            )

        return True, None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply FP8 quantized weights using Triton kernels.

        Args:
            layer: The linear layer with quantized weights
            x: Input tensor (potentially FP8 quantized)
            bias: Optional bias tensor

        Returns:
            Output tensor after scaled matrix multiplication
        """
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
            triton_scaled_mm,
        )

        # Get quantized weight and scale
        weight = getattr(layer, self.w_q_name)
        w_scale = getattr(layer, self.w_s_name)

        # Get input scale (None for dynamic quantization)
        input_scale = None
        if self.config.is_static_input_scheme:
            input_scale = getattr(layer, self.i_s_name)

        # Call Triton scaled_mm implementation
        # This completely bypasses CUTLASS and uses pure Triton kernels
        return triton_scaled_mm(
            input=x,
            weight=weight,
            scale_a=input_scale,
            scale_b=w_scale,
            out_dtype=x.dtype,
            bias=bias,
            use_heuristic=True,  # Auto-tune block sizes
        )
