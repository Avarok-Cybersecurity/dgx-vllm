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

# ============================================================================
# Patch FlashInfer arch_condition.h for GB10 (sm_121a) Support
# ============================================================================
# FlashInfer's arch_condition.h enforces arch-family targets for SM90+
# but doesn't recognize GB10's sm_121a. We add an exception for GB10.
# ============================================================================

echo "=== Patching FlashInfer arch_condition.h for GB10 (sm_121a) ==="

ARCH_COND_HEADER="/opt/venv/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/arch_condition.h"

if [ ! -f "$ARCH_COND_HEADER" ]; then
    echo "WARNING: arch_condition.h not found at $ARCH_COND_HEADER"
    echo "FlashInfer may not be installed yet - continuing without patching"
else
    echo "Found FlashInfer arch_condition.h: $ARCH_COND_HEADER"

    # Backup original
    cp "$ARCH_COND_HEADER" "${ARCH_COND_HEADER}.original"

    # Check if already patched
    if grep -q "GB10.*sm_121a" "$ARCH_COND_HEADER"; then
        echo "✅ arch_condition.h already patched for GB10"
    else
        # Use Python to do the patching - more reliable than sed for complex multiline
        python3 << 'PYTHON_PATCH'
import re

file_path = "/opt/venv/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/arch_condition.h"

with open(file_path, 'r') as f:
    content = f.read()

# Find the #error line
error_pattern = r'(#if.*\n.*\n.*#error.*"Compiling for SM90 or newer.*arch.*target")'

# Replacement: add GB10 exception before the #error block
replacement = r'''// GB10 (sm_121a) Exception - allow sm_121a without arch-family target
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1210
  // GB10 detected - sm_121a is architecture-specific, bypass check
#else
\1
#endif  // GB10 check'''

# Apply the patch
new_content = re.sub(error_pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

if new_content != content:
    with open(file_path, 'w') as f:
        f.write(new_content)
    print("✅ Patched arch_condition.h with GB10 exception")
else:
    print("⚠️  Pattern not found, trying alternative approach...")
    # Fallback: just comment out the #error
    new_content = re.sub(r'(#error.*"Compiling for SM90)', r'// GB10: Commented out\n  // \1', content)
    with open(file_path, 'w') as f:
        f.write(new_content)
    print("✅ Commented out arch check error")
PYTHON_PATCH

        # Verify the patch
        if grep -q "GB10" "$ARCH_COND_HEADER"; then
            echo "✅ arch_condition.h successfully patched for GB10"
            echo "   GB10 (sm_121a) will bypass arch-family requirement"
            # Show the patched section
            echo ""
            echo "Patched section:"
            grep -B2 -A5 "GB10" "$ARCH_COND_HEADER" || true
        else
            echo "⚠️  WARNING: GB10 marker not found, but error may be commented out"
            # Check if error is commented
            if grep -q "// GB10: Commented out" "$ARCH_COND_HEADER"; then
                echo "✅ Error line successfully commented out"
            fi
        fi
    fi
fi

echo "=== FlashInfer FP4 Patching Complete ==="
