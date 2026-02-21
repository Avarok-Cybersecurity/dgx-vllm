#!/usr/bin/env python3
"""
vLLM Quantization Backend for NVFP4

This integrates our CUDA FP4 extension with vLLM!

Usage in vLLM config:
    quantization_config = {
        "format": "nvfp4",
        "group_size": 16
    }
"""

from typing import Any, Dict, List, Optional
import torch
from torch.nn import Parameter

try:
    import cuda_fp4_ext
    FP4_AVAILABLE = True
except ImportError:
    FP4_AVAILABLE = False
    print("⚠️ cuda_fp4_ext not available. Install with: pip install -e /workspace/dgx-vllm-build/cutlass_nvfp4")


class NVFP4Config:
    """Configuration for NVFP4 quantization"""

    def __init__(
        self,
        group_size: int = 16,
        **kwargs
    ):
        self.group_size = group_size
        self.quant_method = "nvfp4"

    def __repr__(self):
        return f"NVFP4Config(group_size={self.group_size})"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NVFP4Config":
        """Create from model config dict"""
        return cls(
            group_size=config.get("group_size", 16)
        )


class NVFP4LinearMethod:
    """NVFP4 quantization method for linear layers"""

    def __init__(self, config: NVFP4Config):
        self.config = config

        if not FP4_AVAILABLE:
            raise RuntimeError(
                "cuda_fp4_ext not available. Install with:\n"
                "  pip install -e /workspace/dgx-vllm-build/cutlass_nvfp4"
            )

        # Print info
        major, minor = cuda_fp4_ext.get_compute_capability()
        has_hw = cuda_fp4_ext.has_hardware_acceleration()

        print(f"✅ NVFP4 Quantization Backend Initialized")
        print(f"   Version: {cuda_fp4_ext.version()}")
        print(f"   Compute Capability: {major}.{minor}")
        print(f"   Group Size: {self.config.group_size}")
        print(f"   Hardware Acceleration: {'✅ Enabled' if has_hw else '⏳ Software (waiting for NVIDIA)'}")

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs
    ):
        """
        Create quantized weight tensors for a linear layer

        Replaces FP32 weight with:
        - weight_data: Packed FP4 values (uint8)
        - weight_scales: Per-group FP8 scales (uint8)
        - weight_scale_global: Global FP32 scale (scalar)
        """

        # Weight data: packed FP4 (2 values per byte)
        weight_data = Parameter(
            torch.empty(
                output_size,
                (input_size + 1) // 2,  # Packed
                dtype=torch.uint8,
                device="cuda"
            ),
            requires_grad=False
        )

        # Weight scales: FP8 per group
        num_groups = (input_size + self.config.group_size - 1) // self.config.group_size
        weight_scales = Parameter(
            torch.empty(
                output_size,
                num_groups,
                dtype=torch.uint8,  # FP8 stored as uint8
                device="cuda"
            ),
            requires_grad=False
        )

        # Global scale: FP32 scalar
        layer.register_buffer(
            "weight_scale_global",
            torch.tensor(1.0, dtype=torch.float32, device="cuda")
        )

        # Register quantized weights
        layer.register_parameter("weight_data", weight_data)
        layer.register_parameter("weight_scales", weight_scales)

        # Mark layer as quantized
        layer.is_quantized = True
        layer.quant_method = "nvfp4"

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply quantized weights to input

        Computes: y = x @ W^T + bias (where W is FP4 quantized)
        """

        # Get quantized weights
        weight_data = layer.weight_data
        weight_scales = layer.weight_scales
        weight_scale_global = layer.weight_scale_global

        # Input: [batch, seq_len, in_features]
        # Weight: [out_features, in_features] (quantized)
        # Output: [batch, seq_len, out_features]

        batch_size, seq_len, in_features = x.shape
        out_features = weight_data.size(0)

        # Reshape input for GEMM: [batch * seq_len, in_features]
        x_2d = x.view(-1, in_features)

        # Perform FP4 GEMM: C = x @ W^T
        # Note: Our GEMM does C = A @ B, so we need to transpose
        # x_2d: [M, K], W: [N, K] → C: [M, N]
        output = cuda_fp4_ext.gemm(
            weight_data,  # A: quantized weights [out_features, in_features]
            weight_scales,
            weight_scale_global.item(),
            x_2d.contiguous(),  # B: input (we'll need to quantize this too!)
            torch.zeros_like(weight_scales),  # Dummy scales for input
            1.0,  # Input not quantized (yet)
            out_features,  # M
            x_2d.size(0),  # N
            in_features,  # K
            group_size=self.config.group_size
        )

        # Note: This is a simplified version
        # For full performance, we should also quantize the input x
        # For now, we'll use a hybrid approach

        # Actually, let's dequantize weights and use PyTorch matmul
        # (More compatible, still saves memory)
        weight_fp32 = cuda_fp4_ext.dequantize(
            weight_data,
            weight_scales,
            weight_scale_global.item(),
            out_features,
            in_features,
            group_size=self.config.group_size
        )

        # Standard linear: y = x @ W^T
        output = torch.nn.functional.linear(x, weight_fp32, bias)

        return output


class NVFP4LinearLayer(torch.nn.Module):
    """
    Drop-in replacement for torch.nn.Linear with NVFP4 quantization

    Example:
        layer = NVFP4LinearLayer(in_features=4096, out_features=4096)
        output = layer(input)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 16,
        device: str = "cuda"
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # Config
        config = NVFP4Config(group_size=group_size)
        self.method = NVFP4LinearMethod(config)

        # Create quantized weights
        self.method.create_weights(
            self,
            in_features,
            out_features,
            torch.float32
        )

        # Bias
        if bias:
            self.bias = Parameter(torch.zeros(out_features, device=device))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with FP4 quantized weights"""
        return self.method.apply_weights(self, x, self.bias)

    def from_float(self, fp32_linear: torch.nn.Linear):
        """
        Convert a FP32 linear layer to FP4

        Args:
            fp32_linear: Standard torch.nn.Linear layer

        Returns:
            Self (for chaining)
        """

        # Quantize weights
        with torch.no_grad():
            weight_fp32 = fp32_linear.weight.data  # [out_features, in_features]

            data, scales, global_scale = cuda_fp4_ext.quantize(
                weight_fp32,
                group_size=self.group_size
            )

            self.weight_data.copy_(data)
            self.weight_scales.copy_(scales)
            self.weight_scale_global.copy_(torch.tensor(global_scale))

            # Copy bias if exists
            if fp32_linear.bias is not None and self.bias is not None:
                self.bias.copy_(fp32_linear.bias.data)

        return self

    def __repr__(self):
        return (
            f"NVFP4LinearLayer(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"group_size={self.group_size}, "
            f"bias={self.bias is not None})"
        )


def quantize_model_to_nvfp4(
    model: torch.nn.Module,
    group_size: int = 16,
    replace_linear: bool = True
) -> torch.nn.Module:
    """
    Quantize all linear layers in a model to NVFP4

    Args:
        model: PyTorch model to quantize
        group_size: Quantization group size
        replace_linear: If True, replace torch.nn.Linear with NVFP4LinearLayer

    Returns:
        Quantized model
    """

    print(f"Quantizing model to NVFP4 (group_size={group_size})...")

    total_params = 0
    quantized_params = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Count parameters
            total_params += module.weight.numel()

            if replace_linear:
                # Create FP4 layer
                fp4_layer = NVFP4LinearLayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    group_size=group_size
                )

                # Quantize
                fp4_layer.from_float(module)

                # Replace in model
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model

                setattr(parent, child_name, fp4_layer)

                quantized_params += module.weight.numel()

    print(f"✅ Quantized {quantized_params:,} / {total_params:,} parameters")
    print(f"   Compression: {total_params * 4 / (quantized_params * 0.5 + total_params * 4 - quantized_params * 4):.2f}x")

    return model


# ============================================================================
# vLLM Integration
# ============================================================================

def get_quant_method(config: Dict[str, Any]):
    """vLLM entry point for NVFP4 quantization"""
    return NVFP4LinearMethod(NVFP4Config.from_config(config))


if __name__ == "__main__":
    print("=" * 60)
    print("  vLLM NVFP4 Quantization Backend")
    print("=" * 60)
    print()

    if not FP4_AVAILABLE:
        print("❌ cuda_fp4_ext not available")
        print("   Install with: pip install -e /workspace/dgx-vllm-build/cutlass_nvfp4")
    else:
        print("✅ cuda_fp4_ext available")
        print(f"   Version: {cuda_fp4_ext.version()}")
        major, minor = cuda_fp4_ext.get_compute_capability()
        print(f"   Compute: SM_{major}{minor}")
        print()
        print("Ready for vLLM integration! 🚀")
