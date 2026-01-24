#!/bin/bash
# ============================================================================
# Patch FlashInfer vec_dtypes.cuh for FP4 JIT Compilation
# ============================================================================
# FlashInfer JIT-compiles kernels at runtime and needs FP4 types visible
# in its own headers. We inject our nv_fp4_dummy.h include.
# ============================================================================

set -e

echo "=== Patching FlashInfer Headers for FP4 JIT Compilation ==="

# Find FlashInfer vec_dtypes.cuh
FLASHINFER_HEADER="/opt/venv/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/vec_dtypes.cuh"

if [ ! -f "$FLASHINFER_HEADER" ]; then
    echo "ERROR: FlashInfer vec_dtypes.cuh not found at $FLASHINFER_HEADER"
    exit 1
fi

echo "Found FlashInfer header: $FLASHINFER_HEADER"

# Copy our FP4 types header to FlashInfer include directory
FLASHINFER_INCLUDE_DIR=$(dirname "$FLASHINFER_HEADER")
cp /workspace/dgx-vllm-build/nv_fp4_dummy.h "$FLASHINFER_INCLUDE_DIR/"
echo "Copied nv_fp4_dummy.h to $FLASHINFER_INCLUDE_DIR/"

# Add include directive at the top of vec_dtypes.cuh (after existing includes)
# Find line with "#pragma once" or first #include and add after it
if grep -q "#pragma once" "$FLASHINFER_HEADER"; then
    # Add after #pragma once
    sed -i '/#pragma once/a \
\
// FP4 types for CUDA 13.0 (GB10 support)\
#include "nv_fp4_dummy.h"' "$FLASHINFER_HEADER"
    echo "Added nv_fp4_dummy.h include after #pragma once"
elif grep -q "#include" "$FLASHINFER_HEADER"; then
    # Add after first #include
    sed -i '0,/#include/a \
\
// FP4 types for CUDA 13.0 (GB10 support)\
#include "nv_fp4_dummy.h"' "$FLASHINFER_HEADER"
    echo "Added nv_fp4_dummy.h include after first #include"
else
    # Add at the very top
    sed -i '1i // FP4 types for CUDA 13.0 (GB10 support)\
#include "nv_fp4_dummy.h"\
' "$FLASHINFER_HEADER"
    echo "Added nv_fp4_dummy.h include at top of file"
fi

# Verify the patch
if grep -q "nv_fp4_dummy.h" "$FLASHINFER_HEADER"; then
    echo "✅ FlashInfer vec_dtypes.cuh successfully patched for FP4 JIT"
else
    echo "❌ ERROR: Failed to patch FlashInfer header"
    exit 1
fi

echo "=== FlashInfer FP4 Patching Complete ==="
