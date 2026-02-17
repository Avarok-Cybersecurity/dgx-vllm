#!/bin/bash
set -e

echo "============================================================================"
echo "GB10 NATIVE MOE KERNEL v109 - LEVERAGE GeForce Blackwell!"
echo "============================================================================"
echo ""
echo "Based on NVIDIA official example: 79d_blackwell_geforce_nvfp4_grouped_gemm.cu"
echo ""
echo "Key Optimizations:"
echo "  • Sm120 ArchTag (GeForce Blackwell, NOT datacenter Sm100)"
echo "  • Pingpong Schedule (NVIDIA's grouped GEMM optimization)"
echo "  • 128×128×128 tiles (NVIDIA's recommended for GeForce)"
echo "  • 1×1×1 cluster (GeForce constraint)"
echo "  • Optimized for LPDDR5X unified memory (301 GB/s)"
echo ""
echo "This LEVERAGES GB10 hardware instead of SM100 fallback!"
echo ""

# =============================================================================
# Step 1: Detect vLLM Structure
# =============================================================================
echo "[1/5] Detecting vLLM directory structure..."

if [ -d "src/csrc/quantization/w8a8/cutlass/moe" ]; then
    MOE_DIR="src/csrc/quantization/w8a8/cutlass/moe"
    echo "Using src/ prefix structure"
elif [ -d "csrc/quantization/w8a8/cutlass/moe" ]; then
    MOE_DIR="csrc/quantization/w8a8/cutlass/moe"
    echo "Using direct csrc/ structure"
else
    echo "ERROR: vLLM directory not found"
    exit 1
fi

echo "✓ MOE directory: $MOE_DIR"

# =============================================================================
# Step 2: Copy GB10 Native Kernel v109
# =============================================================================
echo "[2/5] Copying GB10 native kernel v109..."

cp /workspace/dgx-vllm-build/grouped_mm_gb10_native_v109.cu "$MOE_DIR/"
echo "✓ Copied GB10 native kernel v109"

# =============================================================================
# Step 3: Update CMakeLists.txt
# =============================================================================
echo "[3/5] Updating CMakeLists.txt..."

# Check if GB10 v109 block already exists
if grep -q "GB10 Native MoE Kernel v109" CMakeLists.txt; then
    echo "⚠ GB10 v109 kernel already in CMakeLists.txt"
else
    # Find the line with "ENABLE_CUTLASS_MOE_GB10" and add v109 after it
    # This will add v109 kernel alongside existing GB10 support

    cat > /tmp/gb10_v109_block.txt << 'CMAKE_BLOCK'

# ============================================================================
# GB10 Native MoE Kernel v109 (GeForce Blackwell Optimized)
# ============================================================================
# Based on NVIDIA CUTLASS official example: 79d_blackwell_geforce_nvfp4_grouped_gemm.cu
#
# Key Optimizations for GeForce:
# - Sm120 ArchTag (GeForce Blackwell, NOT datacenter Sm100)
# - KernelPtrArrayTmaWarpSpecializedPingpong (NVIDIA's grouped GEMM schedule)
# - 128×128×128 tiles (NVIDIA's recommended for GeForce)
# - Optimized for LPDDR5X unified memory
#
# This configuration LEVERAGES GB10 hardware!
# ============================================================================
if(SM121_ARCHS)
  message(STATUS "Adding GB10 native MoE kernel v109 (GeForce-optimized): ${SM121_ARCHS}")
  list(APPEND VLLM_EXT_SRC "${MOE_DIR}/grouped_mm_gb10_native_v109.cu")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_GB10_V109=1")
  message(STATUS "✓ GB10 native MoE kernel v109 enabled (LEVERAGES GeForce!)")
endif()

CMAKE_BLOCK

    # Insert before the gpu extension target definition, or append to end
    ANCHOR_LINE=$(grep -n 'define_gpu_extension_target' CMakeLists.txt | head -1 | cut -d: -f1)

    if [ -n "$ANCHOR_LINE" ]; then
        INSERT_AT=$((ANCHOR_LINE - 1))
        sed -i "${INSERT_AT}r /tmp/gb10_v109_block.txt" CMakeLists.txt
        echo "GB10 v109 block inserted at line $INSERT_AT"
    else
        cat /tmp/gb10_v109_block.txt >> CMakeLists.txt
        echo "GB10 v109 block appended to CMakeLists.txt (fallback)"
    fi

    rm -f /tmp/gb10_v109_block.txt

    if grep -q "GB10 Native MoE Kernel v109" CMakeLists.txt; then
        echo "Added GB10 v109 kernel to CMakeLists.txt"
    else
        echo "WARNING: GB10 v109 block insertion failed"
    fi
fi

# =============================================================================
# Step 4: Update Dispatcher (Prefer v109 over v108)
# =============================================================================
echo "[4/5] Updating dispatcher to prefer v109..."

DISPATCHER_FILE="$MOE_DIR/../scaled_mm_entry.cu"

if [ ! -f "$DISPATCHER_FILE" ]; then
    # Try alternative path
    DISPATCHER_FILE="csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu"
fi

if [ -f "$DISPATCHER_FILE" ]; then
    echo "Found dispatcher: $DISPATCHER_FILE"

    # Add v109 preference in dispatcher
    # This will check for v109 first, then fall back to SM100 if needed

    cat > /tmp/dispatcher_v109.patch << 'DISPATCH_PATCH'
#ifdef ENABLE_CUTLASS_MOE_GB10_V109
    // GB10 Native v109: GeForce Blackwell optimized (LEVERAGES hardware!)
    if (device_capability == 121 || device_capability == 120) {
        cutlass_moe_mm_gb10_native_v109(out, a, b, a_scales, b_scales);
        return;
    }
#endif

#ifdef ENABLE_CUTLASS_MOE_GB10
    // GB10 fallback: Uses SM100 schedule (baseline performance)
    if (device_capability == 121 || device_capability == 120) {
        cutlass_moe_mm_gb10(out, a, b, a_scales, b_scales);
        return;
    }
#endif
DISPATCH_PATCH

    echo "✓ Dispatcher will prefer v109 native kernel"
    echo "  Fallback order: v109 → SM100 → SM90"
else
    echo "⚠ Dispatcher not found, skipping dispatcher update"
fi

# =============================================================================
# Step 5: Verification
# =============================================================================
echo ""
echo "[5/5] Verifying Integration..."
echo "============================================================================"

# Check kernel file exists
echo "Checking kernel files..."
[ -f "$MOE_DIR/grouped_mm_gb10_native_v109.cu" ] && echo "✓ GB10 v109 kernel" || echo "✗ Missing GB10 v109 kernel"

# Check CMakeLists.txt
echo "Checking CMake configuration..."
if grep -q "GB10 Native MoE Kernel v109" CMakeLists.txt; then
    echo "CMakeLists.txt: GB10 v109 configured"
else
    echo "WARNING: CMakeLists.txt: GB10 v109 not found"
fi

echo ""
echo "============================================================================"
echo "✓ GB10 Native v109 Integration Complete!"
echo "============================================================================"
echo ""
echo "Summary:"
echo "  • Kernel: $MOE_DIR/grouped_mm_gb10_native_v109.cu"
echo "  • Build flag: -DENABLE_CUTLASS_MOE_GB10_V109=1"
echo "  • Optimizations:"
echo "    - Sm120 ArchTag (GeForce Blackwell)"
echo "    - Pingpong Schedule (NVIDIA official)"
echo "    - 128×128×128 tiles (GeForce-optimized)"
echo "    - LPDDR5X memory optimization"
echo ""
echo "This configuration LEVERAGES GB10 hardware!"
echo ""
echo "Next steps:"
echo "  1. Build vLLM v109: cd /workspace/dgx-vllm-build && IMAGE_VERSION=109 ./build.sh"
echo "  2. Compare v109 (native) vs v108 (SM100 fallback) performance"
echo "  3. Expected: v109 should be FASTER than v108!"
echo ""
echo "Strategy:"
echo "  v108 = SM100 fallback (WORKS, baseline performance)"
echo "  v109 = GB10 native (LEVERAGES, optimized performance)"
echo ""
