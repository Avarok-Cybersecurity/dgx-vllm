#!/bin/bash
set -e

echo "================================"
echo "  Building CUDA FP4 Extension"
echo "================================"
echo ""

# Set CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA_HOME: $CUDA_HOME"
echo "NVCC: $(which nvcc)"
nvcc --version | head -3
echo ""

# Build
echo "Building PyTorch extension..."
python3 setup.py build_ext --inplace

echo ""
echo "✅ Build complete!"
echo ""
echo "Test with:"
echo "  python3 test_pytorch_extension.py"
