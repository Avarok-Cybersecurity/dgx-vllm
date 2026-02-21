#!/bin/bash
set -e

echo "================================"
echo "  Manual CUDA FP4 Build (NVCC)"
echo "================================"
echo ""

CUDA_HOME=/usr/local/cuda
TORCH_PATH=$(python3 -c "import torch; print(torch.__path__[0])")

echo "CUDA_HOME: $CUDA_HOME"
echo "Torch path: $TORCH_PATH"
echo ""

# Compile CUDA kernels to object file
echo "Compiling CUDA kernels..."
$CUDA_HOME/bin/nvcc \
    -c cuda_fp4_kernels.cu \
    -o cuda_fp4_kernels.o \
    -std=c++17 \
    -arch=sm_121 \
    -O3 \
    -use_fast_math \
    --expt-relaxed-constexpr \
    -Xcompiler -fPIC \
    -I. \
    -I/workspace/cutlass/include \
    -I$CUDA_HOME/include \
    -I$TORCH_PATH/include \
    -I$TORCH_PATH/include/torch/csrc/api/include

echo "✅ CUDA kernels compiled!"
echo ""

# Compile C++ extension to object file
echo "Compiling C++ extension..."
g++ \
    -c cuda_fp4_pytorch_ext.cpp \
    -o cuda_fp4_pytorch_ext.o \
    -std=c++17 \
    -O3 \
    -fPIC \
    -I. \
    -I$CUDA_HOME/include \
    -I$TORCH_PATH/include \
    -I$TORCH_PATH/include/torch/csrc/api/include \
    -I$(python3 -c "import pybind11; print(pybind11.get_include())")

echo "✅ C++ extension compiled!"
echo ""

# Link into shared library
echo "Linking shared library..."
g++ \
    -shared \
    -o cuda_fp4_ext.so \
    cuda_fp4_kernels.o \
    cuda_fp4_pytorch_ext.o \
    -L$CUDA_HOME/lib64 \
    -L$TORCH_PATH/lib \
    -lcudart \
    -ltorch \
    -ltorch_python \
    -lc10 \
    -lc10_cuda

echo "✅ Shared library created!"
echo ""

echo "================================"
echo "  ✅ BUILD SUCCESSFUL!"
echo "================================"
echo ""
echo "Test with:"
echo "  python3 -c 'import cuda_fp4_ext; print(cuda_fp4_ext.version())'"
echo "  python3 test_pytorch_extension.py"
