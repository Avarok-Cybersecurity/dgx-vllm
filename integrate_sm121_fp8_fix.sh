#!/bin/bash
# SM_121 FP8 Complete Fix - Patch both regular FP8 and MOE backends
# Patches:
#   1. w8a8_utils.py - Regular FP8 layers use torch backend
#   2. oracle/fp8.py - MOE layers use Triton backend

set -e

echo "=== SM_121 FP8 Complete Backend Fix ==="

# =============================================================================
# PATCH 1: Regular FP8 Linear Layers (w8a8_utils.py)
# =============================================================================
echo ""
echo "[1/2] Patching regular FP8 layers (w8a8_utils.py)..."

W8A8_FILE="/app/vllm/vllm/model_executor/layers/quantization/utils/w8a8_utils.py"

if [ ! -f "${W8A8_FILE}" ]; then
    echo "ERROR: ${W8A8_FILE} not found!"
    exit 1
fi

PATCH_LINE=$(grep -n "elif current_platform.is_cuda() and cutlass_fp8_supported():" "${W8A8_FILE}" | cut -d: -f1 | head -1)

if [ -z "${PATCH_LINE}" ]; then
    echo "ERROR: Could not find backend selection code in ${W8A8_FILE}"
    exit 1
fi

cat > /tmp/sm121_fp8_patch.txt << 'EOFPATCH'
        # SM_121 (GB10) special case: CUTLASS has runtime incompatibility
        # Use torch._scaled_mm fallback which works correctly on SM_121
        capability_tuple = current_platform.get_device_capability()
        if capability_tuple is not None and capability_tuple.to_int() == 121:
            print("[vLLM SM_121] Detected GB10 (SM_121) - using torch backend for FP8 linear layers", flush=True)
            self.preferred_backend = "torch"
        elif current_platform.is_cuda() and cutlass_fp8_supported():
EOFPATCH

awk -v line="${PATCH_LINE}" -v patch="$(cat /tmp/sm121_fp8_patch.txt)" '
    NR == line {
        print patch
        next
    }
    { print }
' "${W8A8_FILE}" > "${W8A8_FILE}.new"

mv "${W8A8_FILE}.new" "${W8A8_FILE}"
echo "✓ Patched ${W8A8_FILE} at line ${PATCH_LINE}"

# =============================================================================
# PATCH 2: MOE FP8 Backend Selection (oracle/fp8.py)
# =============================================================================
echo ""
echo "[2/2] Patching MOE FP8 backend selection (oracle/fp8.py)..."

ORACLE_FILE="/app/vllm/vllm/model_executor/layers/fused_moe/oracle/fp8.py"

if [ ! -f "${ORACLE_FILE}" ]; then
    echo "ERROR: ${ORACLE_FILE} not found!"
    exit 1
fi

# Find the select_fp8_moe_backend function start
FUNC_LINE=$(grep -n "^def select_fp8_moe_backend" "${ORACLE_FILE}" | cut -d: -f1 | head -1)

if [ -z "${FUNC_LINE}" ]; then
    echo "ERROR: Could not find select_fp8_moe_backend function"
    exit 1
fi

# Find the first line after the docstring (look for first non-comment, non-docstring code)
# The function has parameters, docstring, then starts with "if with_lora_support:"
INSERT_LINE=$(awk -v funcline="${FUNC_LINE}" '
    NR >= funcline {
        # Skip the function definition lines and docstring
        if (/^    if with_lora_support:/) {
            print NR
            exit
        }
    }
' "${ORACLE_FILE}")

if [ -z "${INSERT_LINE}" ]; then
    echo "ERROR: Could not find insertion point in oracle/fp8.py"
    exit 1
fi

# Create the MOE patch
cat > /tmp/sm121_moe_patch.txt << 'EOFPATCH'
    # SM_121 (GB10) special case: CUTLASS MOE has runtime incompatibility
    # Use Triton fallback which works correctly on SM_121
    from vllm.platforms import current_platform
    capability_tuple = current_platform.get_device_capability()
    if capability_tuple is not None and capability_tuple.to_int() == 121:
        print("[vLLM SM_121] Detected GB10 (SM_121) - using Triton backend for MOE layers", flush=True)
        return Fp8MoeBackend.TRITON

EOFPATCH

# Insert the patch BEFORE the first if statement
awk -v line="${INSERT_LINE}" -v patch="$(cat /tmp/sm121_moe_patch.txt)" '
    NR == line {
        print patch
    }
    { print }
' "${ORACLE_FILE}" > "${ORACLE_FILE}.new"

mv "${ORACLE_FILE}.new" "${ORACLE_FILE}"
echo "✓ Patched ${ORACLE_FILE} at line ${INSERT_LINE}"

echo ""
echo "=== SM_121 FP8 Complete Fix Applied Successfully ==="
echo "  - Regular FP8 layers: torch._scaled_mm backend"
echo "  - MOE FP8 layers: Triton backend"
echo ""
