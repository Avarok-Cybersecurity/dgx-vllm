FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

# ============================================================================
# vLLM Docker Image for DGX Spark GB10
# ============================================================================
# Features:
# - vLLM latest from main (auto-updated at build time)
# - PyTorch stable with CUDA 13.0 (ARM64 compatible)
# - Triton 3.6.0 with SM_121 support
# - FlashInfer latest pre-release (patched for sm_121a)
# - CUDA FP4 extension (custom headers + kernels for GB10)
# - NVFP4 full compilation (software E2M1 for SM121, no stubs needed)
# - GB10-optimized MoE Triton config (+65.7% throughput)
# - SM_121 capability routing to SM_120 kernels
# - CUTLASS FP8 disabled for SM_121 (PyTorch fallback)
# - torch.compile RE-ENABLED for NVFP4 (v22: Marlin backend bypasses AutogradCUDA issue)
#
# Build time: 30-60 minutes
# Target: NVIDIA GB10 (sm_121, Compute Capability 12.1)
# ============================================================================

# Install essentials, InfiniBand/RDMA libraries, and network utilities
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3-pip git wget patch \
    cmake build-essential ninja-build \
    libibverbs1 libibverbs-dev ibverbs-providers rdma-core perftest \
    libnuma-dev \
    iproute2 iputils-ping net-tools curl openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create virtual env
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA 13.0 support (stable release for ARM64 compatibility)
# Note: PyTorch depends on triton, so it will be installed automatically
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install xgrammar from PyPI (not in cu130 index)
RUN pip install xgrammar

# Install flashinfer using --pre flag for pre-release versions
RUN pip install flashinfer-python --pre

# Pin PyTorch CUDA version - flashinfer and vLLM pip install pull torch from
# PyPI (CPU-only). Setting extra-index-url ensures all subsequent pip installs
# resolve torch from the cu130 index instead of defaulting to CPU.
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu130

# Reinstall PyTorch CUDA after flashinfer (which just downgraded it)
RUN pip install torch==2.10.0+cu130 torchvision==0.25.0+cu130 torchaudio==2.10.0+cu130

# Clone vLLM (pinned to known-good revision for reproducible builds)
RUN git clone https://github.com/vllm-project/vllm.git && \
    cd vllm && git checkout 3b30e6150777de549b11f67dde3ecc0d3b1f3f50
WORKDIR /app/vllm

# Prepare for existing torch
RUN python3 use_existing_torch.py

# Install build requirements
RUN pip install -r requirements/build.txt

# ============================================================================
# Install FP4 Type Definitions for CUDA 13.0
# ============================================================================
# CUDA 13.0's CCCL headers reference __nv_fp4_e2m1 for SM_120/SM_121 but
# the type doesn't exist. Install our proven FP4 implementation and patch
# CCCL headers to include it.
# ============================================================================
COPY nv_fp4_dummy.h /usr/local/cuda/include/nv_fp4_dummy.h
COPY patch_cccl_fp4.sh /tmp/patch_cccl_fp4.sh
RUN chmod +x /tmp/patch_cccl_fp4.sh && /tmp/patch_cccl_fp4.sh

# ============================================================================
# Apply CUTLASS Blackwell Support for GB10
# ============================================================================
# Enables CUTLASS kernels for GB10 (12.1) - adds SM_121 support
# Must add 12.0f and 12.1f to multiple architecture lists:
# 1. CUDA_SUPPORTED_ARCHS - filters all architectures
# 2. SCALED_MM_ARCHS - FP8 quantization kernels (3 locations)
# 3. FP4_ARCHS - ENABLED (12.1f - uses our __nv_fp4_e2m1 implementation!)
# 4. NVFP4_ARCHS - ENABLED (12.1f - uses our complete FP4 intrinsics!)
# 5. MLA_ARCHS - Multi-head latent attention
# 6. CUTLASS_MOE_DATA_ARCHS - MoE data handling
#
# NOTE: DUAL FP4 support - CUTLASS kernels + custom extension (cutlass_nvfp4/)
# Using sed to be resilient to vLLM version changes
# ============================================================================
RUN if [ -f CMakeLists.txt ]; then \
    # Add 12.1 to CUDA_SUPPORTED_ARCHS for CUDA 13.0+ \
    sed -i 's/set(CUDA_SUPPORTED_ARCHS "7\.5;8\.0;8\.6;8\.7;8\.9;9\.0;10\.0;11\.0;12\.0")/set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0;12.1")/g' CMakeLists.txt && \
    # Add 12.0f and 12.1f to SCALED_MM_ARCHS (SM100 kernels) - 3 instances \
    sed -i 's/cuda_archs_loose_intersection(SCALED_MM_ARCHS "10\.0f;11\.0f"/cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f;12.0f;12.1f"/g' CMakeLists.txt && \
    # NOTE: FP4_ARCHS for SM_120 is NOT modified - nvfp4_quant_kernels.cu uses \
    # cvt.e2m1x2 instruction which doesn't exist on GB10 (sm_121). \
    # Missing symbols (scaled_fp4_quant_sm1xxa etc) are handled by nvfp4_stubs.cu \
    # Add 12.1f to MLA_ARCHS (multi-head latent attention) \
    sed -i 's/cuda_archs_loose_intersection(MLA_ARCHS "10\.0f;11\.0f;12\.0f"/cuda_archs_loose_intersection(MLA_ARCHS "10.0f;11.0f;12.0f;12.1f"/g' CMakeLists.txt && \
    # Add 12.1f to CUTLASS_MOE_DATA_ARCHS (MoE data handling) \
    sed -i 's/cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9\.0a;10\.0f;11\.0f;12\.0f"/cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0f;11.0f;12.0f;12.1f"/g' CMakeLists.txt; \
fi

# ============================================================================
# Integrate Native SM_121 Kernels for GB10 (NO FALLBACKS)
# ============================================================================
# Adds GB10-specific CUTLASS kernels optimized for SM_121:
# - Native MoE kernel: grouped_mm_gb10_native.cu
# - Native scaled_mm kernels: scaled_mm_sm121_fp8.cu + blockwise variant
# - 1x1x1 cluster shape (no multicast support)
# - 301 GB/s LPDDR5X unified memory optimizations
# - Optimized tile sizes and scheduling for GB10 hardware
# ============================================================================
COPY grouped_mm_gb10_native.cu /workspace/dgx-vllm-build/grouped_mm_gb10_native.cu
COPY scaled_mm_sm121_fp8.cu /workspace/dgx-vllm-build/scaled_mm_sm121_fp8.cu
COPY scaled_mm_blockwise_sm121_fp8.cu /workspace/dgx-vllm-build/scaled_mm_blockwise_sm121_fp8.cu
COPY scaled_mm_sm121_fp8_dispatch.cuh /workspace/dgx-vllm-build/scaled_mm_sm121_fp8_dispatch.cuh
COPY scaled_mm_blockwise_sm121_fp8_dispatch.cuh /workspace/dgx-vllm-build/scaled_mm_blockwise_sm121_fp8_dispatch.cuh
COPY scaled_mm_c3x_sm121.cu /workspace/dgx-vllm-build/scaled_mm_c3x_sm121.cu
COPY fix_dispatcher_v2.sh /workspace/dgx-vllm-build/fix_dispatcher_v2.sh
COPY integrate_gb10_sm121.sh .
RUN chmod +x integrate_gb10_sm121.sh && ./integrate_gb10_sm121.sh

# ============================================================================
# Integrate SM_121 FP8 Backend Fix
# ============================================================================
# CRITICAL: Modify vLLM source BEFORE compilation
# - Patches CUTLASS backend to return False for SM_121
# - Forces fallback to PyTorch (torch._scaled_mm) which works on SM_121
# - Updated for new vLLM scaled_mm architecture
# ============================================================================
COPY integrate_sm121_fp8_fix_v2.sh /workspace/dgx-vllm-build/integrate_sm121_fp8_fix_v2.sh
RUN chmod +x /workspace/dgx-vllm-build/integrate_sm121_fp8_fix_v2.sh && \
    /workspace/dgx-vllm-build/integrate_sm121_fp8_fix_v2.sh

# ============================================================================
# v114 FIX 1: CMAKE - Force SM_120 Kernel Build for GB10 (CORRECTED!)
# ============================================================================
# ROOT CAUSE (CORRECTED - found in v113 failure):
#   Lines 532-536: CUDA version check determines arch list!
#
#   if CUDA >= 13.0:
#     Line 533: cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0f" ...)
#     → Only 12.0f! NOT 12.1f!
#   else:
#     Line 535: cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0a" ...)
#
#   We use CUDA 13.0.88 → Uses LINE 533 branch!
#   v113 fixed line 535 → WRONG BRANCH! Still failed!
#
# Solution (v114 Fix 1 - CORRECTED):
#   Fix LINE 533: "12.0f" → "12.0f;12.1f"  ← The branch we actually use!
#   → CMake will build SM_120 kernels for GB10
#   → ENABLE_SCALED_MM_SM120=1 will be defined
# ============================================================================
COPY fix_cmake_sm120_archs_v113_corrected.sh /workspace/dgx-vllm-build/fix_cmake_sm120_archs_v113_corrected.sh
RUN chmod +x /workspace/dgx-vllm-build/fix_cmake_sm120_archs_v113_corrected.sh && \
    /workspace/dgx-vllm-build/fix_cmake_sm120_archs_v113_corrected.sh

# ============================================================================
# v114 FIX 2: ROUTING - Capability 121 to SM_120
# ============================================================================
# v112 Achievement:
#   ✅ Modified routing code (line 199) to catch 121
#   ❌ But code was never compiled (ENABLE_SCALED_MM_SM120 undefined)
#
# v113 Achievement:
#   ❌ Fixed wrong CMake branch (line 535 instead of 533)
#   ❌ SM_120 kernels still not built
#
# Solution (v114 Fix 2):
#   Same as v112 - modify routing to >= 120 && < 130
#   BUT with v114 Fix 1 corrected, it will finally be compiled!
#
# Combined Result:
#   Fix 1: SM_120 kernels built (correct branch!) → routing code exists
#   Fix 2: Routing catches 121 → routes to SM_120 kernels
#   SUCCESS!
# ============================================================================
COPY fix_capability_121_v112.py /workspace/dgx-vllm-build/fix_capability_121_v112.py
RUN chmod +x /workspace/dgx-vllm-build/fix_capability_121_v112.py && \
    python3 /workspace/dgx-vllm-build/fix_capability_121_v112.py /app/vllm

# ============================================================================
# v115 FIX 3: DISPATCHER FLAG - THE FINAL PIECE!
# ============================================================================
# ROOT CAUSE (discovered in v114 failure):
#   scaled_mm_entry.cu (dispatcher) compiled WITHOUT ENABLE_SCALED_MM_SM120=1
#   SM_120 kernels compiled WITH the flag
#   Result: Kernels exist, but #ifdef evaluates to FALSE in dispatcher!
#
# v114 Evidence:
#   - CMake: "Building scaled_mm_c3x_sm120 for archs: 12.1f" ✅
#   - Kernels 47, 51, 55: SM_120 kernels compiled ✅
#   - Runtime: "No compiled cutlass_scaled_mm for capability: 121" ❌
#   - Extracted code shows: #ifdef ENABLE_SCALED_MM_SM120 block exists
#   - But: #ifdef evaluates to FALSE → flag not defined for dispatcher!
#
# Solution (v115 Fix 3):
#   Explicitly set ENABLE_SCALED_MM_SM120=1 for scaled_mm_entry.cu
#   using set_source_files_properties() after SM_120 section
#   → Ensures dispatcher knows about SM_120 kernels!
#
# Complete Fix Chain:
#   Fix 1 (v114): Build SM_120 kernels → kernels exist
#   Fix 2 (v112): Update routing code → routing logic correct
#   Fix 3 (v115): Flag for dispatcher → #ifdef passes!
#   RESULT: Capability 121 → routes to SM_120 kernels → SUCCESS!
# ============================================================================
COPY fix_dispatcher_flag_v115.sh /workspace/dgx-vllm-build/fix_dispatcher_flag_v115.sh
RUN chmod +x /workspace/dgx-vllm-build/fix_dispatcher_flag_v115.sh && \
    /workspace/dgx-vllm-build/fix_dispatcher_flag_v115.sh

# ============================================================================
# v126 FIX: Register cutlass_fp4_group_mm for Plain CUDA Dispatch (CORRECT FIX!)
# NOTE: v126 CUDA dispatch fix (AutogradCUDA) was replaced by v134 (disable torch.compile for NVFP4)
# v134 fix is applied AFTER pip install (see below)

# ============================================================================
# GB10 Native MoE Kernel v109 (GeForce Blackwell Optimized)
# ============================================================================
COPY grouped_mm_gb10_native_v109.cu /workspace/dgx-vllm-build/grouped_mm_gb10_native_v109.cu
COPY integrate_gb10_native_v109.sh .
RUN chmod +x integrate_gb10_native_v109.sh && ./integrate_gb10_native_v109.sh

# ============================================================================
# CUDA FP4 Extension - Custom headers and test binaries for GB10
# ============================================================================
COPY cutlass_nvfp4 /workspace/dgx-vllm-build/cutlass_nvfp4
COPY integrate_cuda_fp4_extension.sh /workspace/dgx-vllm-build/integrate_cuda_fp4_extension.sh
RUN chmod +x /workspace/dgx-vllm-build/integrate_cuda_fp4_extension.sh && \
    /workspace/dgx-vllm-build/integrate_cuda_fp4_extension.sh

# FP4 tensor core env flags
ENV ENABLE_TCGEN05_HARDWARE=1
ENV NVCC_PREPEND_FLAGS="-DENABLE_TCGEN05_HARDWARE=1"

# ============================================================================
# GB10 Full NVFP4 Compilation v6 - ALL KERNELS (no stubs!)
# ============================================================================
# v6: Software E2M1 conversion in nvfp4_utils.cuh enables ALL quant kernels
# to compile for SM121, eliminating the need for Python software fallback.
#
# Step 1: Patch nvfp4_utils.cuh with software E2M1 for SM121
#   - Adds #if __CUDA_ARCH__ == 1210 guards around cvt.rn.satfinite.e2m1x2
#   - Software implementation matches hardware round-to-nearest-even behavior
#
# Step 2: CMake patch to compile ALL kernel files for sm_121:
#   - nvfp4_quant_kernels.cu (activation quant - uses software E2M1)
#   - nvfp4_experts_quant.cu (per-expert quant - uses software E2M1)
#   - activation_nvfp4_quant_fusion_kernels.cu (SiLU+Mul+quant)
#   - nvfp4_blockwise_moe_kernel.cu (CUTLASS MoE GEMM - mma.e2m1)
#   - nvfp4_scaled_mm_sm120_kernels.cu (CUTLASS FP4 GEMM - mma.e2m1)
#
# This eliminates:
#   - Python software FP4 quant fallback (.item() calls = ~1 tok/s)
#   - Quant function stubs (nvfp4_stubs.cu)
#   - CUDA graph capture failure (cudaErrorStreamCaptureUnsupported)
# ============================================================================
COPY patch_nvfp4_utils_sw_e2m1.py /workspace/dgx-vllm-build/patch_nvfp4_utils_sw_e2m1.py
RUN python3 /workspace/dgx-vllm-build/patch_nvfp4_utils_sw_e2m1.py

COPY cmake_patch_gb10_nvfp4_v6_full_kernels.sh /workspace/dgx-vllm-build/cmake_patch_gb10_nvfp4_v6_full_kernels.sh
RUN chmod +x /workspace/dgx-vllm-build/cmake_patch_gb10_nvfp4_v6_full_kernels.sh && \
    /workspace/dgx-vllm-build/cmake_patch_gb10_nvfp4_v6_full_kernels.sh

# ============================================================================
# Build Configuration for GB10 Blackwell
# ============================================================================
# Use 12.1a for architecture-specific features (tcgen05.mma FP4 tensor cores)
# sm_121a enables Blackwell-specific PTX instructions including tcgen05
# CUDA 13.0 PTX ISA 8.8 adds support for sm_121a target architecture
# ============================================================================
ENV TORCH_CUDA_ARCH_LIST="12.1a"
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV NVCC_PREPEND_FLAGS="-arch=sm_121a ${NVCC_PREPEND_FLAGS}"
ENV TIKTOKEN_ENCODINGS_BASE=/app/tiktoken_encodings

# NCCL configuration for InfiniBand/RoCE multi-GPU
ENV NCCL_SOCKET_IFNAME=enp1s0f0np0
ENV NCCL_IB_DISABLE=0
ENV NCCL_DEBUG=WARN
ENV NCCL_NET_GDR_LEVEL=2
ENV NCCL_IB_TIMEOUT=23
ENV NCCL_IB_GID_INDEX=0
ENV NCCL_ASYNC_ERROR_HANDLING=1
ENV TORCH_NCCL_BLOCKING_WAIT=1

# ============================================================================
# Native FP4 Tensor Cores: sm_121a compilation for tcgen05.mma
# ============================================================================
# CRITICAL: Use sm_121a (not sm_121) to enable architecture-specific instructions
# - sm_121a unlocks tcgen05.mma.ss.kind::f8f6f4 (native FP4 tensor cores)
# - Requires PTX ISA 8.8 (CUDA 13.0+) and GB10 hardware
# - Expected: 4x compute throughput vs BF16, 2.5-3.5x total speedup
# - Patched in integrate_optimized_kernel.sh via cmake/utils.cmake modification
# ============================================================================

# Install vLLM with local build (this takes a while)
# Pin torch to cu130 via constraints to prevent pip from downgrading to CPU version.
# Without this, vLLM's dependency resolution replaces torch+cu130 with torch (CPU).
RUN echo "torch==2.10.0+cu130" > /tmp/constraints.txt && \
    echo "torchvision==0.25.0+cu130" >> /tmp/constraints.txt && \
    echo "torchaudio==2.10.0+cu130" >> /tmp/constraints.txt && \
    PIP_CONSTRAINT=/tmp/constraints.txt pip install --no-build-isolation -e . -v --pre

# Fix PyTorch CUDA: vLLM pip install pulls torch from PyPI (CPU-only) despite
# PIP_EXTRA_INDEX_URL. Force reinstall cu130 version. The CUDA extensions were
# already compiled against CUDA torch (from our pre-build install), so the .so
# files are compatible - only the Python package metadata needs fixing.
RUN pip install torch==2.10.0+cu130 torchvision==0.25.0+cu130 torchaudio==2.10.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130 --force-reinstall --no-deps

# ============================================================================
# Patch FlashInfer Headers for FP4 JIT Compilation
# ============================================================================
# CRITICAL: Patch AFTER vLLM installation (when FlashInfer is installed)
# FlashInfer JIT-compiles kernels at runtime and needs FP4 types in headers
# ============================================================================
COPY patch_flashinfer_fp4.sh /tmp/patch_flashinfer_fp4.sh
COPY nv_fp4_dummy.h /workspace/dgx-vllm-build/nv_fp4_dummy.h
RUN chmod +x /tmp/patch_flashinfer_fp4.sh && /tmp/patch_flashinfer_fp4.sh

# ============================================================================
# v134: torch.compile for NVFP4 — REMOVED in v22
# ============================================================================
# Previously disabled torch.compile due to AutogradCUDA dispatch key error
# on cutlass_fp4_group_mm. With Marlin backend (VLLM_TEST_FORCE_FP8_MARLIN=1),
# CUTLASS FP4 GEMM ops are never called at runtime, so torch.compile should
# work. If it crashes, fallback: add cutlass_fp4_group_mm to splitting_ops.
# ============================================================================

# NOTE: Python software FP4 quantization (gb10_nvfp4_software_quant.py) is
# NO LONGER NEEDED in v21. The C++ quant kernels now compile for SM121 with
# software E2M1 conversion in nvfp4_utils.cuh. The compiled CUDA kernels
# are CUDA-graph-capturable (no .item() calls, no GPU→CPU transfers).

# ============================================================================
# Fix Qwen3Next doubled prefix in create_qkvz_proj (MUST be AFTER pip install)
# ============================================================================
# Bug: Both caller and create_qkvz_proj append '.in_proj_qkvz' to prefix,
# creating 'model.layers.X.linear_attn.in_proj_qkvz.in_proj_qkvz' which
# doesn't match the quantization ignore list, causing weight loading failures.
# ============================================================================
COPY fix_qwen3_next_prefix.py /workspace/dgx-vllm-build/fix_qwen3_next_prefix.py
RUN python3 /workspace/dgx-vllm-build/fix_qwen3_next_prefix.py

# ============================================================================
# Fix NVFP4 EMULATION backend dequantization (MUST be AFTER pip install)
# ============================================================================
# Two bugs in run_nvfp4_emulations():
# 1. Weight scales in LINEAR format but dequantize_to_dtype assumes swizzled
# 2. weight_global_scale inverted (1/actual_gs) but code divides by it
# Result: Garbled output from all NVFP4 models using EMULATION backend
# ============================================================================
COPY fix_nvfp4_emulation_backend.py /workspace/dgx-vllm-build/fix_nvfp4_emulation_backend.py
RUN python3 /workspace/dgx-vllm-build/fix_nvfp4_emulation_backend.py

# NOTE: Triton 3.6.0 is already installed by PyTorch cu130 wheel.
# No separate install step needed (removed in v17 cleanup).

# ============================================================================
# Install GB10-Optimized MoE Configuration
# ============================================================================
# Custom Triton kernel config tuned for GB10's unified memory (301 GB/s)
# Provides 65.7% throughput improvement vs default config
# - Smaller BLOCK_SIZE_K (64-128) reduces memory traffic
# - More num_stages (4-5) hides memory latency
# - Smaller GROUP_SIZE_M (1-16) optimized for unified memory
# ============================================================================
COPY E=512,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json /app/vllm/vllm/model_executor/layers/fused_moe/configs/

# Download tiktoken encodings
WORKDIR /app
RUN mkdir -p tiktoken_encodings && \
    wget -O tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -O tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set working directory back to vllm
WORKDIR /app/vllm

# Expose ports (vLLM API and Ray)
EXPOSE 8888 6379

# Version metadata
LABEL version="22"
LABEL build_date="2026-02-18"
LABEL vllm_source="3b30e6150-patched"
LABEL pytorch_version="stable-cu130"
LABEL compute_capability="12.1a-gb10"
LABEL quantization_support="fp8-nvfp4"
LABEL sm121_fp8_backend="torch-scaled-mm-fallback"
LABEL moe_config="gb10-custom-tuned"
LABEL maintainer="avarok"

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8888/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["serve"]
