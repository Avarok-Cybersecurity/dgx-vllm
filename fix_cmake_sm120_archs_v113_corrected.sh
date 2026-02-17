#!/bin/bash
set -e

echo "════════════════════════════════════════════════════════════"
echo "CMAKE SM_120 ARCHITECTURE FIX (Pattern-based, v17)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "PURPOSE:"
echo "  The SCALED_MM_ARCHS for SM_120 only lists \"12.0f\" (CUDA >= 13.0 branch)"
echo "  GB10 is compute 12.1 so needs \"12.1f\" added to be included."
echo ""
echo "  Uses PATTERN-BASED sed (not hardcoded line numbers) for resilience."
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

CMAKELISTS="/app/vllm/CMakeLists.txt"

if [ ! -f "$CMAKELISTS" ]; then
    echo "ERROR: CMakeLists.txt not found at $CMAKELISTS"
    exit 1
fi

echo "Fixing CMakeLists.txt..."
echo ""

# Create backup
cp "$CMAKELISTS" "${CMAKELISTS}.sm120_arch.backup"
echo "Backup created: ${CMAKELISTS}.sm120_arch.backup"

# ============================================================================
# CUDA >= 13.0 branch: "12.0f" -> "12.0f;12.1f"
# Pattern: cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0f"
# This is the branch we use with CUDA 13.0.x
# ============================================================================
BEFORE_COUNT=$(grep -c 'SCALED_MM_ARCHS "12\.0f"' "$CMAKELISTS" || true)

if [ "$BEFORE_COUNT" -gt 0 ]; then
    # Only replace instances that have exactly "12.0f" (not already "12.0f;12.1f")
    sed -i 's/cuda_archs_loose_intersection(SCALED_MM_ARCHS "12\.0f"/cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0f;12.1f"/g' "$CMAKELISTS"
    AFTER_COUNT=$(grep -c 'SCALED_MM_ARCHS "12\.0f;12\.1f"' "$CMAKELISTS" || true)
    echo "CUDA >= 13.0 branch: Changed $AFTER_COUNT instance(s) from \"12.0f\" to \"12.0f;12.1f\""
else
    # Check if already patched
    ALREADY=$(grep -c 'SCALED_MM_ARCHS "12\.0f;12\.1f"' "$CMAKELISTS" || true)
    if [ "$ALREADY" -gt 0 ]; then
        echo "CUDA >= 13.0 branch: Already contains 12.1f ($ALREADY instance(s))"
    else
        echo "WARNING: Could not find SCALED_MM_ARCHS with \"12.0f\" pattern"
        echo "  Upstream may have changed the arch list format"
        grep -n "SCALED_MM_ARCHS" "$CMAKELISTS" | head -5
    fi
fi

# ============================================================================
# CUDA < 13.0 branch: "12.0a" -> "12.0a;12.1a" (for completeness)
# ============================================================================
BEFORE_COUNT_A=$(grep -c 'SCALED_MM_ARCHS "12\.0a"' "$CMAKELISTS" || true)

if [ "$BEFORE_COUNT_A" -gt 0 ]; then
    sed -i 's/cuda_archs_loose_intersection(SCALED_MM_ARCHS "12\.0a"/cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0a;12.1a"/g' "$CMAKELISTS"
    AFTER_COUNT_A=$(grep -c 'SCALED_MM_ARCHS "12\.0a;12\.1a"' "$CMAKELISTS" || true)
    echo "CUDA < 13.0 branch:  Changed $AFTER_COUNT_A instance(s) from \"12.0a\" to \"12.0a;12.1a\""
else
    ALREADY_A=$(grep -c 'SCALED_MM_ARCHS "12\.0a;12\.1a"' "$CMAKELISTS" || true)
    if [ "$ALREADY_A" -gt 0 ]; then
        echo "CUDA < 13.0 branch:  Already contains 12.1a ($ALREADY_A instance(s))"
    else
        echo "INFO: No SCALED_MM_ARCHS with \"12.0a\" pattern found (may not exist in this vLLM version)"
    fi
fi

echo ""

# Verify the changes
echo "Verification:"
if grep -q 'SCALED_MM_ARCHS "12\.0f;12\.1f"' "$CMAKELISTS"; then
    echo "  PASS: 12.1f present in SCALED_MM_ARCHS (CUDA >= 13.0 branch)"
else
    echo "  FAIL: 12.1f NOT found in SCALED_MM_ARCHS!"
    exit 1
fi

if grep -q 'SCALED_MM_ARCHS "12\.0a;12\.1a"' "$CMAKELISTS"; then
    echo "  PASS: 12.1a present in SCALED_MM_ARCHS (CUDA < 13.0 branch)"
else
    echo "  INFO: 12.1a not found (may not exist in this vLLM version, non-critical)"
fi

echo ""
echo "Expected CMake output:"
echo "  'Building scaled_mm_c3x_sm120 for archs: 12.1f'"
echo ""
echo "════════════════════════════════════════════════════════════"
