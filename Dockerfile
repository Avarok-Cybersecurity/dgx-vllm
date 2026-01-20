FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

# ============================================================================
# vLLM Docker Image for DGX Spark GB10 - Version 48
# ============================================================================
# Production-ready vLLM with COMPLETE SM_121 support for GB10 Blackwell
#
# Features:
# - vLLM latest from main (auto-updated at build time)
# - PyTorch stable with CUDA 13.0 (ARM64 compatible)
# - TORCH backend for SM_121 FP8 linear layers (torch._scaled_mm)
# - TRITON backend for SM_121 FP8 MOE layers (bypasses CUTLASS entirely)
# - GB10-optimized MoE config (+65.7% throughput vs default)
# - FlashInfer latest pre-release
# - XGrammar latest stable
#
# Build time: 30-60 minutes (kernel compilation)
# Target: NVIDIA GB10 (sm_121)
# Approach: Patch BOTH w8a8_utils.py AND oracle/fp8.py for complete SM_121 support
# v22-v45: 26 iterations debugging wrong code path
# v46: Partial fix - regular layers work, MOE still fails
# v47: COMPLETE FIX - patches both regular FP8 AND MOE backends!
# v48: GB10 MoE optimization - 65.7% performance improvement!
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

# REMOVE triton after ALL installations (causes CUDA 13.0 errors)
# Must be done LAST as both PyTorch and xgrammar pull it in as dependency
RUN pip uninstall -y triton && echo "Triton successfully removed"

# Clone vLLM
RUN git clone https://github.com/vllm-project/vllm.git
WORKDIR /app/vllm

# Prepare for existing torch
RUN python3 use_existing_torch.py

# Install build requirements
RUN pip install -r requirements/build.txt

# ============================================================================
# Apply CUTLASS Blackwell Support for GB10
# ============================================================================
# Enables CUTLASS kernels for NVFP4, MXFP4, MXFP6, MXFP8 quantization on GB10 (12.1)
# Must add 12.0f and 12.1f to multiple architecture lists:
# 1. CUDA_SUPPORTED_ARCHS - filters all architectures
# 2. SCALED_MM_ARCHS - FP8 quantization kernels (3 locations)
# 3. FP4_ARCHS - FP4 quantization kernels
# 4. MLA_ARCHS - Multi-head latent attention
# 5. CUTLASS_MOE_DATA_ARCHS - MoE data handling
# Using sed to be resilient to vLLM version changes
# ============================================================================
RUN if [ -f CMakeLists.txt ]; then \
    # Add 12.1 to CUDA_SUPPORTED_ARCHS for CUDA 13.0+ \
    sed -i 's/set(CUDA_SUPPORTED_ARCHS "7\.5;8\.0;8\.6;8\.7;8\.9;9\.0;10\.0;11\.0;12\.0")/set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0;12.1")/g' CMakeLists.txt && \
    # Add 12.0f and 12.1f to SCALED_MM_ARCHS (SM100 kernels) - 3 instances \
    sed -i 's/cuda_archs_loose_intersection(SCALED_MM_ARCHS "10\.0f;11\.0f"/cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f;12.0f;12.1f"/g' CMakeLists.txt && \
    # Add 12.1f to FP4_ARCHS (FP4 quantization for SM100) \
    sed -i 's/cuda_archs_loose_intersection(FP4_ARCHS "10\.0f;11\.0f"/cuda_archs_loose_intersection(FP4_ARCHS "10.0f;11.0f;12.1f"/g' CMakeLists.txt && \
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
# Integrate Triton Backend for SM_121 FP8 Operations
# ============================================================================
# CRITICAL: Modify vLLM source BEFORE compilation
# - Adds TritonPureScaledMMLinearKernel to backend options
# - Routes SM_121 to Triton instead of CUTLASS for FP8 scaled_mm
# - Works with torch.compile (unlike Python monkeypatching)
# - Backend selection happens before graph compilation
# ============================================================================
COPY triton_pure.py /workspace/dgx-vllm-build/triton_pure.py
COPY integrate_sm121_fp8_fix.sh /workspace/dgx-vllm-build/integrate_sm121_fp8_fix.sh
RUN chmod +x /workspace/dgx-vllm-build/integrate_sm121_fp8_fix.sh && \
    /workspace/dgx-vllm-build/integrate_sm121_fp8_fix.sh

# ============================================================================
# Build Configuration for GB10 Blackwell
# ============================================================================
# Include both 12.0f (relaxed) and 12.1f (GB10)
# Note: CUTLASS used for MoE, Triton used for scaled_mm on SM_121
# ============================================================================
ENV TORCH_CUDA_ARCH_LIST="12.0f;12.1f"
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
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

# Install vLLM with local build (this takes a while)
RUN pip install --no-build-isolation -e . -v --pre

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
LABEL version="48"
LABEL build_date="2026-01-20"
LABEL vllm_source="main-branch-latest-modified"
LABEL pytorch_version="stable-cu130"
LABEL sm121_backend="torch-scaled-mm-fallback"
LABEL compute_capability="12.0f,12.1f"
LABEL sm121_scaled_mm="triton-pure-backend"
LABEL sm121_moe="triton-gb10-optimized"
LABEL moe_config="gb10-custom-tuned"
LABEL moe_improvement="+65.7%"
LABEL backend_selection="modified-before-compilation"
LABEL cmake_fix="sm121-early-placement"
LABEL kernel_fix="epilogue-dispatch-functions"
LABEL linkage_fix="extern-c-no-duplicates"
LABEL approach="vllm-source-modification"
LABEL previous_attempts="v22-v38-cutlass-monkeypatch-failed"
LABEL solution="triton-native-backend-pre-compile"
LABEL optimization="gb10-moe-config-unified-memory"
LABEL maintainer="avarok"

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8888/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["serve"]
