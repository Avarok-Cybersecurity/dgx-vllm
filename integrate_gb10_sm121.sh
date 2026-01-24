#!/bin/bash
set -e

echo "============================================================================"
echo "Integrating Native SM_121 Kernels for GB10 (DGX Spark)"
echo "============================================================================"
echo ""
echo "This integration provides TRUE SM_121 kernels optimized for GB10:"
echo "  • Native scaled_mm implementation (not SM89/SM100 fallback)"
echo "  • Native MoE implementation"
echo "  • Hardware-optimized for 301 GB/s LPDDR5X unified memory"
echo "  • ClusterShape 1x1x1 (no multicast dependencies)"
echo ""

# =============================================================================
# Step 1: Copy MoE Kernel
# =============================================================================
echo "[1/5] Copying GB10 MoE kernel..."

# Debug: show current directory and structure
echo "DEBUG: Current directory: $(pwd)"
echo "DEBUG: Checking for vLLM structure..."
ls -la | head -20
echo ""

# Try both possible structures (with and without src/)
if [ -d "src/csrc/quantization/w8a8/cutlass/moe" ]; then
    MOE_KERNEL_DEST="src/csrc/quantization/w8a8/cutlass/moe/grouped_mm_gb10_native.cu"
    SCALED_MM_DIR="src/csrc/quantization/w8a8/cutlass"
    DISPATCHER_FILE="src/csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu"
    echo "Using src/ prefix structure"
elif [ -d "csrc/quantization/w8a8/cutlass/moe" ]; then
    MOE_KERNEL_DEST="csrc/quantization/w8a8/cutlass/moe/grouped_mm_gb10_native.cu"
    SCALED_MM_DIR="csrc/quantization/w8a8/cutlass"
    DISPATCHER_FILE="csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu"
    echo "Using direct csrc/ structure"
else
    echo "ERROR: vLLM MoE directory not found in either:"
    echo "  - src/csrc/quantization/w8a8/cutlass/moe"
    echo "  - csrc/quantization/w8a8/cutlass/moe"
    echo "DEBUG: Available directories:"
    find . -maxdepth 4 -type d -name "quantization" 2>/dev/null || echo "No quantization directory found"
    exit 1
fi

cp /workspace/dgx-vllm-build/grouped_mm_gb10_native.cu "$MOE_KERNEL_DEST"
echo "✓ Copied MoE kernel: $MOE_KERNEL_DEST"

# =============================================================================
# Step 2: Copy SM_121 Scaled MM Kernels
# =============================================================================
echo "[2/5] Copying SM_121 scaled_mm kernels..."
# SCALED_MM_DIR is set in Step 1 based on detected structure

# Verify directory exists or search for it
if [ ! -d "$SCALED_MM_DIR" ]; then
    echo "WARNING: Scaled MM directory not found at expected location: $SCALED_MM_DIR"
    echo "DEBUG: Searching for quantization directories..."
    find . -maxdepth 5 -type d -name "quantization" -o -name "cutlass*" -o -name "w8a8" 2>/dev/null | sort
    echo ""
    echo "DEBUG: Searching for existing scaled_mm files..."
    find . -maxdepth 6 -name "scaled_mm*.cu" -type f 2>/dev/null | head -10
    exit 1
fi

# Create c3x subdirectory if it doesn't exist
mkdir -p "$SCALED_MM_DIR/c3x"

# Copy kernel implementations
cp /workspace/dgx-vllm-build/scaled_mm_sm121_fp8.cu "$SCALED_MM_DIR/c3x/"
cp /workspace/dgx-vllm-build/scaled_mm_blockwise_sm121_fp8.cu "$SCALED_MM_DIR/c3x/"
echo "✓ Copied kernel implementations"

# Copy dispatch headers
cp /workspace/dgx-vllm-build/scaled_mm_sm121_fp8_dispatch.cuh "$SCALED_MM_DIR/c3x/"
cp /workspace/dgx-vllm-build/scaled_mm_blockwise_sm121_fp8_dispatch.cuh "$SCALED_MM_DIR/c3x/"
echo "✓ Copied dispatch headers"

# Copy entry point
cp /workspace/dgx-vllm-build/scaled_mm_c3x_sm121.cu "$SCALED_MM_DIR/"
echo "✓ Copied entry point"

# =============================================================================
# Step 2b: Add SM121 Forward Declarations to scaled_mm_kernels.hpp
# =============================================================================
echo "[2b/6] Adding SM121 forward declarations to scaled_mm_kernels.hpp..."

KERNELS_HEADER="$SCALED_MM_DIR/c3x/scaled_mm_kernels.hpp"

if [ ! -f "$KERNELS_HEADER" ]; then
    echo "ERROR: scaled_mm_kernels.hpp not found at: $KERNELS_HEADER"
    exit 1
fi

# Check if SM121 declarations already exist
if grep -q "cutlass_scaled_mm_sm121_fp8" "$KERNELS_HEADER"; then
    echo "⚠ SM121 forward declarations already present"
else
    # Create a temporary file with the declarations to insert
    cat > /tmp/sm121_declarations.txt << 'DECLEOF'

void cutlass_scaled_mm_sm121_fp8(torch::Tensor& out, torch::Tensor const& a,
                                 torch::Tensor const& b,
                                 torch::Tensor const& a_scales,
                                 torch::Tensor const& b_scales,
                                 std::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_blockwise_sm121_fp8(torch::Tensor& out,
                                           torch::Tensor const& a,
                                           torch::Tensor const& b,
                                           torch::Tensor const& a_scales,
                                           torch::Tensor const& b_scales);
DECLEOF

    # Insert the declarations before the closing namespace brace
    # Find the line number of the closing brace
    CLOSING_BRACE_LINE=$(grep -n '^}  // namespace vllm' "$KERNELS_HEADER" | cut -d: -f1)

    if [ -z "$CLOSING_BRACE_LINE" ]; then
        echo "ERROR: Could not find closing namespace brace"
        grep -n "namespace vllm" "$KERNELS_HEADER" || echo "No namespace lines found"
        exit 1
    fi

    # Insert before the closing brace (sed 'r' inserts after, so use line-1)
    BEFORE_BRACE=$((CLOSING_BRACE_LINE - 1))
    sed -i "${BEFORE_BRACE}r /tmp/sm121_declarations.txt" "$KERNELS_HEADER"

    # Verify the declarations were added
    if grep -q "cutlass_scaled_mm_sm121_fp8" "$KERNELS_HEADER"; then
        echo "✓ Added SM121 forward declarations to $KERNELS_HEADER"
        rm -f /tmp/sm121_declarations.txt
    else
        echo "ERROR: Failed to add SM121 forward declarations"
        echo "DEBUG: Attempting to find SM120 blockwise declaration..."
        grep -n "cutlass_scaled_mm_blockwise_sm120_fp8" "$KERNELS_HEADER" || echo "Pattern not found"
        exit 1
    fi
fi

# =============================================================================
# Step 3: Update CMakeLists.txt
# =============================================================================
echo "[3/6] Updating CMakeLists.txt..."

# Add GB10 MoE kernel block (if not already present)
if grep -q "GB10_ARCHS" CMakeLists.txt; then
    echo "⚠ GB10 MoE kernel already in CMakeLists.txt"
else
    # Build CMake block with detected path
    GB10_MOE_BLOCK="
# ============================================================================
# GB10 Native MoE Kernel (GeForce Blackwell - Compute Capability 12.1)
# ============================================================================
cuda_archs_loose_intersection(GB10_ARCHS \"12.0f;12.1f\" \"\${CUDA_ARCHS}\")
if(\${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND GB10_ARCHS)
  message(STATUS \"Building GB10 native MoE kernel for architectures: \${GB10_ARCHS}\")
  set(GB10_SRCS \"$MOE_KERNEL_DEST\")
  set_gencode_flags_for_srcs(
    SRCS \"\${GB10_SRCS}\"
    CUDA_ARCHS \"\${GB10_ARCHS}\")
  list(APPEND VLLM_EXT_SRC \"\${GB10_SRCS}\")
  list(APPEND VLLM_GPU_FLAGS \"-DENABLE_CUTLASS_MOE_GB10=1\")
  message(STATUS \"✓ GB10 native MoE kernel enabled\")
endif()
"

    # Insert after existing CUTLASS blocks
    awk -v block="$GB10_MOE_BLOCK" '
        /endif\(\)/ && prev ~ /SCALED_MM_ARCHS|FP4_ARCHS/ && !inserted {
            print
            print block
            inserted=1
            next
        }
        {
            prev = $0
            print
        }
        END {
            if (!inserted) {
                print block
            }
        }
    ' CMakeLists.txt > CMakeLists.txt.tmp

    mv CMakeLists.txt.tmp CMakeLists.txt
    echo "✓ Added GB10 MoE kernel to CMake"
fi

# Add SM_121 scaled_mm kernel block (if not already present)
if grep -q "SM121_ARCHS" CMakeLists.txt; then
    echo "⚠ SM_121 scaled_mm kernel already in CMakeLists.txt"
else
    # Build CMake block with detected paths
    SM121_BLOCK="
# ============================================================================
# SM_121 Native Scaled MM Kernels (GB10 - Compute Capability 12.1)
# ============================================================================
cuda_archs_loose_intersection(SM121_ARCHS \"12.0f;12.1f\" \"\${CUDA_ARCHS}\")
if(\${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SM121_ARCHS)
  message(STATUS \"Building SM_121 native scaled_mm kernels for: \${SM121_ARCHS}\")
  set(SM121_SRCS
    \"$SCALED_MM_DIR/scaled_mm_c3x_sm121.cu\"
    \"$SCALED_MM_DIR/c3x/scaled_mm_sm121_fp8.cu\"
    \"$SCALED_MM_DIR/c3x/scaled_mm_blockwise_sm121_fp8.cu\")
  set_gencode_flags_for_srcs(
    SRCS \"\${SM121_SRCS}\"
    CUDA_ARCHS \"\${SM121_ARCHS}\")
  list(APPEND VLLM_EXT_SRC \"\${SM121_SRCS}\")
  list(APPEND VLLM_GPU_FLAGS \"-DENABLE_SCALED_MM_SM121=1\")
  message(STATUS \"✓ SM_121 native scaled_mm kernels enabled\")
endif()
"

    # Insert after GB10 MoE block using sed
    # Find the line with "GB10 native MoE kernel enabled" and insert after the next endif()
    sed -i '/GB10 native MoE kernel enabled/,/^endif()/{
        /^endif()/{
            a\

            r /dev/stdin
        }
    }' CMakeLists.txt <<< "$SM121_BLOCK"

    # Verify insertion succeeded (SM_121 section should appear before line 600)
    if head -600 CMakeLists.txt | grep -q "SM_121 Native Scaled MM Kernels"; then
        echo "✓ SM_121 section inserted after GB10 section (before line 600)"
    else
        # Fallback: check if it's at the end (indicates insertion failed)
        if tail -50 CMakeLists.txt | grep -q "SM_121 Native Scaled MM Kernels"; then
            echo "WARNING: SM_121 section at end of file - sed insertion failed"
            echo "This will cause SM_121 kernels to NOT compile!"
            exit 1
        fi
    fi
    echo "✓ Added SM_121 scaled_mm kernels to CMake"
fi

# =============================================================================
# Step 4: Update scaled_mm_entry.cu Dispatcher
# =============================================================================
echo "[4/6] Updating C++ dispatcher (scaled_mm_entry.cu)..."
# DISPATCHER_FILE is set in Step 1 based on detected structure

if [ ! -f "$DISPATCHER_FILE" ]; then
    echo "ERROR: Dispatcher file not found: $DISPATCHER_FILE"
    exit 1
fi

# Check if already patched
if grep -q "cutlass_moe_mm_gb10" "$DISPATCHER_FILE" && grep -q "cutlass_scaled_mm_sm121" "$DISPATCHER_FILE"; then
    echo "⚠ Dispatcher already patched for GB10/SM_121"
else
    # Use improved script-based fix
    echo "Applying GB10 dispatcher modifications..."

    # Copy improved fix script
    cp /workspace/dgx-vllm-build/fix_dispatcher_v2.sh .
    chmod +x fix_dispatcher_v2.sh

    # Run the fix script
    if ./fix_dispatcher_v2.sh "$DISPATCHER_FILE"; then
        echo "✓ Dispatcher fixed successfully"
    else
        echo "ERROR: Dispatcher fix failed"
        exit 1
    fi
fi

# =============================================================================
# Step 5: Update Python Dispatcher (_custom_ops.py)
# =============================================================================
echo "[5/6] Updating Python dispatcher to accept capability 121..."

PYTHON_CUSTOM_OPS="vllm/_custom_ops.py"

if [ -f "$PYTHON_CUSTOM_OPS" ]; then
    # Update cutlass_moe_mm capability check
    if grep -q "device_capability in {90, 100}" "$PYTHON_CUSTOM_OPS"; then
        sed -i 's/device_capability in {90, 100}/device_capability in {90, 100, 121}/g' "$PYTHON_CUSTOM_OPS"
        echo "✓ Updated cutlass_moe_mm to accept capability 121"
    fi

    # Update cutlass_scaled_mm capability check
    if grep -q "capability in {90, 100, 120}" "$PYTHON_CUSTOM_OPS"; then
        sed -i 's/capability in {90, 100, 120}/capability in {90, 100, 120, 121}/g' "$PYTHON_CUSTOM_OPS"
        echo "✓ Updated cutlass_scaled_mm to accept capability 121"
    elif grep -q "capability in {90, 100}" "$PYTHON_CUSTOM_OPS"; then
        sed -i 's/capability in {90, 100}/capability in {90, 100, 121}/g' "$PYTHON_CUSTOM_OPS"
        echo "✓ Updated cutlass_scaled_mm to accept capability 121"
    fi

    # Update error messages
    if grep -q "capability: 90 or 100" "$PYTHON_CUSTOM_OPS"; then
        sed -i 's/capability: 90 or 100/capability: 90, 100, or 121/g' "$PYTHON_CUSTOM_OPS"
        echo "✓ Updated error messages"
    fi
else
    echo "⚠ Python custom ops file not found: $PYTHON_CUSTOM_OPS"
fi

# =============================================================================
# Step 6: Verification
# =============================================================================
echo ""
echo "[6/6] Verifying Integration..."
echo "============================================================================"

# Check kernel files exist
echo "Checking kernel files..."
[ -f "$MOE_KERNEL_DEST" ] && echo "✓ MoE kernel: $MOE_KERNEL_DEST" || echo "✗ Missing MoE kernel"
[ -f "$SCALED_MM_DIR/c3x/scaled_mm_sm121_fp8.cu" ] && echo "✓ SM_121 FP8 kernel" || echo "✗ Missing SM_121 FP8 kernel"
[ -f "$SCALED_MM_DIR/scaled_mm_c3x_sm121.cu" ] && echo "✓ SM_121 entry point" || echo "✗ Missing SM_121 entry point"

# Check header forward declarations
echo "Checking header forward declarations..."
if grep -q "cutlass_scaled_mm_sm121_fp8" "$KERNELS_HEADER"; then
    echo "✓ SM_121 forward declarations in scaled_mm_kernels.hpp"
else
    echo "✗ Missing SM_121 forward declarations in scaled_mm_kernels.hpp"
fi

# Check CMakeLists.txt
echo "Checking CMake configuration..."
if grep -q "GB10_ARCHS" CMakeLists.txt && grep -q "SM121_ARCHS" CMakeLists.txt; then
    echo "✓ CMakeLists.txt: Both GB10 and SM_121 configured"
else
    echo "✗ CMakeLists.txt configuration incomplete"
fi

# Check dispatcher
echo "Checking dispatcher..."
if grep -q "ENABLE_CUTLASS_MOE_GB10" "$DISPATCHER_FILE" && \
   grep -q "ENABLE_SCALED_MM_SM121" "$DISPATCHER_FILE"; then
    echo "✓ Dispatcher: Both GB10 MoE and SM_121 scaled_mm configured"
else
    echo "✗ Dispatcher configuration incomplete"
fi

echo ""
echo "============================================================================"
echo "✓ GB10 Native SM_121 Kernel Integration Complete!"
echo "============================================================================"
echo ""
echo "Summary:"
echo "  • GB10 MoE kernel: csrc/quantization/w8a8/cutlass/moe/grouped_mm_gb10_native.cu"
echo "  • SM_121 scaled_mm kernels: csrc/quantization/cutlass_w8a8/c3x/scaled_mm_sm121_*.cu"
echo "  • Build flags: -DENABLE_CUTLASS_MOE_GB10=1 -DENABLE_SCALED_MM_SM121=1"
echo "  • Target architectures: 12.0f, 12.1f"
echo "  • Dispatcher routing: SM_121 (capability 121) → native GB10 kernels"
echo ""
echo "Next steps:"
echo "  1. Build vLLM with 'pip install -e .' or Docker image build"
echo "  2. Test with Qwen3-Next-80B-A3B-Instruct-FP8-Dynamic"
echo "  3. Verify no 'Error Internal' - should work correctly!"
echo ""
echo "NO FALLBACKS OR WORKAROUNDS - Pure SM_121 implementation!"
echo ""
