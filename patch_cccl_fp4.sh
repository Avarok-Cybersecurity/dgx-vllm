#!/bin/bash
# ============================================================================
# Patch CUDA CCCL Headers to Include FP4 Type Definitions
# ============================================================================
# CUDA 13.0's CCCL headers reference __nv_fp4_e2m1 but don't define it.
# This script injects our FP4 type definitions into the CCCL header.
# ============================================================================

set -e

echo "================================"
echo "  Patching CCCL for FP4 Support"
echo "================================"

CCCL_HEADER="/usr/local/cuda/include/cccl/cuda/std/__type_traits/is_extended_floating_point.h"

if [ ! -f "$CCCL_HEADER" ]; then
    echo "❌ CCCL header not found: $CCCL_HEADER"
    exit 1
fi

echo "Found CCCL header: $CCCL_HEADER"

# Check if already patched
if grep -q "nv_fp4_dummy.h" "$CCCL_HEADER"; then
    echo "✅ Already patched, skipping"
    exit 0
fi

echo "Adding FP4 type definitions..."

# Insert include at the top of the file (after the first #ifndef)
sed -i '0,/#ifndef/a\
// FP4 type definitions for CUDA 13.0 compatibility\
#include <nv_fp4_dummy.h>\
' "$CCCL_HEADER"

echo "✅ CCCL header patched successfully"
echo ""
