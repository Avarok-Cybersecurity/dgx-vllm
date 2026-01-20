# DGX-vLLM: Optimized vLLM for NVIDIA DGX Spark GB10

High-performance vLLM Docker image optimized for NVIDIA DGX Spark GB10 (Grace Blackwell Superchip) hardware.

## 🚀 Performance Achievements

- **65.7% throughput improvement** over default vLLM configuration
- **Complete SM_121 (Blackwell GB10) FP8 support** via custom backend routing
- **Production-tested** with Qwen3-Next-80B-A3B-Instruct (39.36 tok/s)
- **Unified memory optimizations** tailored for GB10's 301 GB/s LPDDR5X architecture

## 🎯 Key Features

### Complete SM_121 Support
- **TORCH backend** for FP8 linear layers (torch._scaled_mm)
- **TRITON backend** for FP8 MoE layers (bypasses CUTLASS runtime incompatibility)
- Source-level patching ensures compatibility with torch.compile

### GB10-Optimized MoE Configuration
- Custom Triton kernel configs tuned for unified memory architecture
- Reduced BLOCK_SIZE_K (64-128) to minimize memory traffic
- Increased num_stages (4-5) to hide memory latency
- Optimized GROUP_SIZE_M (1-16) for unified memory access patterns

### Production-Ready Stack
- vLLM latest from main branch (auto-updated at build time)
- PyTorch stable with CUDA 13.0 (ARM64 compatible)
- FlashInfer latest pre-release
- XGrammar latest stable
- InfiniBand/RoCE multi-node support

## 📋 Hardware Requirements

- **GPU**: NVIDIA GB10 (Compute Capability 12.1)
- **System**: NVIDIA DGX Spark (Grace Blackwell Superchip)
- **Memory**: 119.7 GB GPU memory
- **Network**: InfiniBand RoCE (optional, for multi-node)

## 🔨 Building the Image

### Prerequisites
- Docker with NVIDIA Container Runtime
- CUDA 13.0+ drivers
- ~60 minutes build time on DGX hardware

### Build Command

```bash
docker build -t dgx-vllm:latest .
```

Or use the provided build script:

```bash
./build.sh
```

## 🚢 Usage

### Quick Start

**Single GPU deployment:**
```bash
docker run -d \
  --name vllm-server \
  --network host \
  --gpus all \
  --ipc=host \
  -e MODEL="DevQuasar/Qwen.Qwen3-Next-80B-A3B-Instruct-FP8-Dynamic" \
  -e PORT=8888 \
  -e TENSOR_PARALLEL_SIZE=1 \
  -e MAX_MODEL_LEN=65536 \
  -e GPU_MEMORY_UTIL=0.88 \
  -e VLLM_EXTRA_ARGS="--kv-cache-dtype fp8 --swap-space 32 --tool-call-parser hermes --reasoning-parser qwen3" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  dgx-vllm:latest serve
```

### API Testing

```bash
# Test inference
curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DevQuasar/Qwen.Qwen3-Next-80B-A3B-Instruct-FP8-Dynamic",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'

# Check available models
curl http://localhost:8888/v1/models
```

## 🏗️ Technical Architecture

### SM_121 Backend Strategy

**Problem**: CUTLASS FP8 kernels compile for SM_121 but fail at runtime with:
```
RuntimeError: No suitable CUTLASS kernel found for SM_121
```

**Solution**: Dual-backend routing patched at source level

#### 1. Regular FP8 Linear Layers
**File**: `vllm/model_executor/layers/quantization/utils/w8a8_utils.py`

**Patch**: Detect SM_121 and route to `torch._scaled_mm` backend
```python
capability_tuple = current_platform.get_device_capability()
if capability_tuple is not None and capability_tuple.to_int() == 121:
    print("[vLLM SM_121] Detected GB10 (SM_121) - using torch backend for FP8 linear layers")
    self.preferred_backend = "torch"
```

#### 2. MoE FP8 Layers
**File**: `vllm/model_executor/layers/fused_moe/oracle/fp8.py`

**Patch**: Detect SM_121 and route to Triton backend
```python
from vllm.platforms import current_platform
capability_tuple = current_platform.get_device_capability()
if capability_tuple is not None and capability_tuple.to_int() == 121:
    print("[vLLM SM_121] Detected GB10 (SM_121) - using Triton backend for MOE layers")
    return Fp8MoeBackend.TRITON
```

### GB10 MoE Configuration

**File**: `E=512,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json`

Optimized Triton kernel parameters for GB10's unified memory:

| Parameter | B200 (8 TB/s HBM3e) | GB10 (301 GB/s LPDDR5X) | Reason |
|-----------|---------------------|--------------------------|--------|
| BLOCK_SIZE_K | 128-256 | 64-128 | Reduce memory traffic |
| num_stages | 3-4 | 4-5 | Hide memory latency |
| GROUP_SIZE_M | 1-64 | 1-16 | Unified memory access pattern |

**Result**: 65.7% performance improvement (23.75 → 39.36 tok/s)

## 📊 Benchmarks

**Test Model**: Qwen3-Next-80B-A3B-Instruct-FP8-Dynamic
**Context Length**: 64k tokens
**Hardware**: Single NVIDIA GB10 (119.7 GB)

| Configuration | Throughput | Status |
|---------------|-----------|---------|
| Default vLLM | 23.75 tok/s | ⚠️ Sub-optimal |
| v48 (GB10-optimized) | **39.36 tok/s** | ✅ **+65.7%** |
| v48 + Speculative | 18.12 tok/s | ❌ -54% (not recommended) |

### Why Speculative Decoding Fails on GB10

- **Acceptance rate**: 0-1.4% (vs 70-80% on H100/B200)
- **Root cause**: Memory bandwidth bottleneck overwhelms MTP head
- **Recommendation**: Avoid speculative decoding on GB10
- See: [SPECULATIVE_DECODING_RESULTS.md](../SPECULATIVE_DECODING_RESULTS.md)

## 🔍 Build Process Details

### CMakeLists.txt Modifications
Adds SM_121 (12.1) to CUDA architecture lists:
- `CUDA_SUPPORTED_ARCHS`
- `SCALED_MM_ARCHS` (3 locations)
- `FP4_ARCHS`
- `MLA_ARCHS`
- `CUTLASS_MOE_DATA_ARCHS`

### Source Code Patches
Applied via `integrate_sm121_fp8_fix.sh` before compilation:
1. **w8a8_utils.py**: Regular FP8 backend selection
2. **oracle/fp8.py**: MoE FP8 backend selection

### MoE Config Installation
Custom GB10 config copied to:
```
/app/vllm/vllm/model_executor/layers/fused_moe/configs/
```

## 🐛 Troubleshooting

### Build Failures

**Issue**: CUTLASS compilation errors
```
error: no suitable kernel for SM_121
```

**Solution**: Ensure `TORCH_CUDA_ARCH_LIST="12.0f;12.1f"` is set and CMakeLists.txt patches applied

### Runtime Errors

**Issue**: "No CUTLASS kernel found" errors
```
RuntimeError: No suitable CUTLASS kernel found for SM_121
```

**Solution**: Verify source patches were applied:
```bash
docker logs <container> 2>&1 | grep "\[vLLM SM_121\]"
```

Expected output:
```
[vLLM SM_121] Detected GB10 (SM_121) - using torch backend for FP8 linear layers
[vLLM SM_121] Detected GB10 (SM_121) - using Triton backend for MOE layers
```

### Performance Issues

**Issue**: Sub-optimal performance warning
```
vLLM WARNING: Using default MoE config (sub-optimal for GB10)
```

**Solution**: Verify GB10 MoE config file is present:
```bash
docker exec <container> ls -la /app/vllm/vllm/model_executor/layers/fused_moe/configs/ | grep GB10
```

## 📚 Development History

This image is the result of **28 iterations** (v22-v48) debugging and optimizing vLLM for GB10 hardware:

- **v22-v45**: Attempted CUTLASS monkeypatching (failed - wrong code path)
- **v46**: Partial fix - regular FP8 layers work, MoE crashes
- **v47**: Complete SM_121 support via dual backend routing
- **v48**: GB10 MoE optimization (+65.7% performance)

See our [Medium article](../MEDIUM_ARTICLE.md) for the complete technical journey.

## 🤝 Contributing

This project documents production optimizations for GB10 hardware. Contributions welcome:

- Bug reports and fixes
- Performance optimizations
- Documentation improvements
- Support for additional models

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🔗 Related Resources

- [vLLM Project](https://github.com/vllm-project/vllm)
- [NVIDIA DGX Documentation](https://docs.nvidia.com/dgx/)
- [CUTLASS Library](https://github.com/NVIDIA/cutlass)
- [Triton Language](https://github.com/openai/triton)

## 📝 Citation

If you use this work in your research or production deployments:

```bibtex
@software{dgx_vllm_2026,
  title = {DGX-vLLM: Optimized vLLM for NVIDIA DGX Spark GB10},
  author = {Avarok Cybersecurity},
  year = {2026},
  url = {https://github.com/Avarok-Cybersecurity/dgx-vllm}
}
```

## 🎯 Acknowledgments

- vLLM team for the excellent inference framework
- NVIDIA for GB10 hardware and CUDA ecosystem
- Community contributors for debugging insights

---

**Version**: 48
**Last Updated**: 2026-01-20
**Maintained by**: Avarok Cybersecurity
**Status**: Production Ready ✅
