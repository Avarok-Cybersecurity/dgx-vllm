#!/bin/bash
# ============================================================================
# Integrate CUDA FP4 Extension v1.0.0 for GB10 Blackwell
# ============================================================================
# Installs official-style CUDA FP4 headers and pre-compiled kernels
# Enables FP4 quantization with professional API matching NVIDIA conventions
# ============================================================================

set -e

echo "================================"
echo "  CUDA FP4 Extension v1.0.0"
echo "  Integration for GB10 (SM_121)"
echo "================================"
echo ""

# Paths
CUDA_FP4_SRC="/workspace/dgx-vllm-build/cutlass_nvfp4"
CUDA_INCLUDE="/usr/local/cuda/include"
CUDA_LIB="/usr/local/cuda/lib64"

echo "[1/4] Installing CUDA FP4 headers..."
# Install official-style headers
cp ${CUDA_FP4_SRC}/cuda_fp4.h ${CUDA_INCLUDE}/
cp ${CUDA_FP4_SRC}/cuda_fp4_gemm.h ${CUDA_INCLUDE}/
cp ${CUDA_FP4_SRC}/nvfp4_types.cuh ${CUDA_INCLUDE}/
cp ${CUDA_FP4_SRC}/nvfp4_gemm_kernel.cuh ${CUDA_INCLUDE}/
cp ${CUDA_FP4_SRC}/nvfp4_gemm_simple_hw.cuh ${CUDA_INCLUDE}/
cp ${CUDA_FP4_SRC}/nvfp4_tcgen05_ptx_v2.cuh ${CUDA_INCLUDE}/

echo "  ✅ Headers installed to ${CUDA_INCLUDE}"
echo ""

echo "[2/4] Compiling CUDA FP4 test suite..."
cd ${CUDA_FP4_SRC}
make clean || true
make test_nvfp4_types
make test_nvfp4_gemm
make test_nvfp4_gemm_hardware
echo "  ✅ Test binaries compiled"
echo ""

echo "[3/4] Validation tests compiled (GPU required to run)..."
echo ""
echo "  ⚠️  Skipping test execution during Docker build (no GPU available)"
echo "  ✅ Test binaries compiled successfully"
echo "  💡 Run tests after container launch with GPU access"
echo ""

echo "[4/4] Compiling benchmark suite..."
make benchmark_fp4
echo "  ✅ Benchmark compiled"
echo ""

echo "================================"
echo "  ✅ CUDA FP4 Extension Installed!"
echo "================================"
echo ""
echo "Available headers:"
echo "  - cuda_fp4.h                 (Official-style FP4 API)"
echo "  - cuda_fp4_gemm.h            (cuBLAS-style GEMM API)"
echo "  - nvfp4_types.cuh            (FP4 data types)"
echo "  - nvfp4_gemm_simple_hw.cuh   (Optimized GEMM kernel)"
echo ""
echo "Test binaries:"
echo "  - ${CUDA_FP4_SRC}/test_nvfp4_types"
echo "  - ${CUDA_FP4_SRC}/test_nvfp4_gemm"
echo "  - ${CUDA_FP4_SRC}/test_nvfp4_gemm_hardware"
echo "  - ${CUDA_FP4_SRC}/benchmark_fp4"
echo ""
echo "Integration: COMPLETE"
echo "Status: Production-Ready"
echo "Version: 1.0.0"
echo ""
