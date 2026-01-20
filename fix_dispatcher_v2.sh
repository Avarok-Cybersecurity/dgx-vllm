#!/bin/bash
# Improved dispatcher fix script with better error handling
# v3: Prevents duplicate declarations inside function bodies

set -e

DISPATCHER_FILE="$1"

if [ ! -f "$DISPATCHER_FILE" ]; then
    echo "ERROR: File not found: $DISPATCHER_FILE"
    exit 1
fi

echo "Fixing dispatcher: $DISPATCHER_FILE"

# Create backup
cp "$DISPATCHER_FILE" "${DISPATCHER_FILE}.backup"

# ============================================================================
# Step 0: Remove any existing GB10/SM121 declarations (clean slate)
# ============================================================================
# This handles cases where the script ran partially before
sed -i '/^\/\/ GB10 Native MoE Kernel (SM_121)/,/^#endif/{
/^#if defined ENABLE_CUTLASS_MOE_GB10/,/^#endif/d
/^\/\/ GB10 Native MoE Kernel (SM_121)/d
}' "$DISPATCHER_FILE"

sed -i '/^\/\/ SM_121 Native Scaled MM Kernel (GB10)/,/^#endif/{
/^#if defined ENABLE_SCALED_MM_SM121/,/^#endif/d
/^\/\/ SM_121 Native Scaled MM Kernel (GB10)/d
}' "$DISPATCHER_FILE"

# ============================================================================
# Step 1: Add GB10 MoE forward declaration at file level only
# ============================================================================
# Strategy: Find the SM100 MoE declaration block (has semicolon), not the routing code
# We look for "void cutlass_moe_mm_sm100(" followed by parameters ending with ");"

# Find line number of SM100 MoE forward declaration
SM100_DECL_LINE=$(grep -n "^void cutlass_moe_mm_sm100(" "$DISPATCHER_FILE" | head -1 | cut -d: -f1)

if [ -z "$SM100_DECL_LINE" ]; then
    echo "ERROR: Cannot find SM100 MoE forward declaration"
    exit 1
fi

# Find the #endif after this declaration
ENDIF_LINE=$(awk -v start=$SM100_DECL_LINE 'NR > start && /^#endif/ {print NR; exit}' "$DISPATCHER_FILE")

if [ -z "$ENDIF_LINE" ]; then
    echo "ERROR: Cannot find #endif after SM100 declaration"
    exit 1
fi

# Insert GB10 declaration after this #endif
GB10_DECL='
// GB10 Native MoE Kernel (SM_121)
#if defined ENABLE_CUTLASS_MOE_GB10 && ENABLE_CUTLASS_MOE_GB10
extern "C" void cutlass_moe_mm_gb10(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_token, bool per_out_ch);
#endif'

# Use awk to insert at specific line
awk -v line=$ENDIF_LINE -v decl="$GB10_DECL" '
NR == line { print; print decl; next }
{ print }
' "$DISPATCHER_FILE" > "${DISPATCHER_FILE}.tmp"
mv "${DISPATCHER_FILE}.tmp" "$DISPATCHER_FILE"

# ============================================================================
# Step 1b: Add SM_121 scaled_mm forward declaration at file level only
# ============================================================================
# Same strategy: Find the SM120 declaration (not routing code)

SM120_DECL_LINE=$(grep -n "^void cutlass_scaled_mm_sm120(" "$DISPATCHER_FILE" | head -1 | cut -d: -f1)

if [ -z "$SM120_DECL_LINE" ]; then
    echo "ERROR: Cannot find SM120 scaled_mm forward declaration"
    exit 1
fi

# Find the #endif after this declaration
SM120_ENDIF_LINE=$(awk -v start=$SM120_DECL_LINE 'NR > start && /^#endif/ {print NR; exit}' "$DISPATCHER_FILE")

if [ -z "$SM120_ENDIF_LINE" ]; then
    echo "ERROR: Cannot find #endif after SM120 declaration"
    exit 1
fi

# Insert SM_121 declaration after this #endif
SM121_DECL='
// SM_121 Native Scaled MM Kernel (GB10)
#if defined ENABLE_SCALED_MM_SM121 && ENABLE_SCALED_MM_SM121
void cutlass_scaled_mm_sm121(torch::Tensor& c, torch::Tensor const& a,
                             torch::Tensor const& b,
                             torch::Tensor const& a_scales,
                             torch::Tensor const& b_scales,
                             std::optional<torch::Tensor> const& bias);
#endif'

awk -v line=$SM120_ENDIF_LINE -v decl="$SM121_DECL" '
NR == line { print; print decl; next }
{ print }
' "$DISPATCHER_FILE" > "${DISPATCHER_FILE}.tmp"
mv "${DISPATCHER_FILE}.tmp" "$DISPATCHER_FILE"

echo "  - Function declarations added at file level"

# ============================================================================
# Step 2: Add GB10 MoE routing in cutlass_moe_mm function
# ============================================================================

# Insert GB10 routing right after "int32_t version_num = get_sm_version_num();"
# in the cutlass_moe_mm function (NOT cutlass_scaled_mm)
sed -i '/^void cutlass_moe_mm(/,/^}$/{
/int32_t version_num = get_sm_version_num();/a\
  // Route SM_121 (GB10) to native MoE kernel\
#if defined ENABLE_CUTLASS_MOE_GB10 && ENABLE_CUTLASS_MOE_GB10\
  if (version_num >= 121 && version_num < 130) {\
    cutlass_moe_mm_gb10(out_tensors, a_tensors, b_tensors, a_scales, b_scales,\
                        expert_offsets, problem_sizes, a_strides, b_strides,\
                        c_strides, per_act_token, per_out_ch);\
    return;\
  }\
#endif
}' "$DISPATCHER_FILE"

echo "  - GB10 MoE routing added"

# ============================================================================
# Step 3: Add SM_121 scaled_mm routing in cutlass_scaled_mm function
# ============================================================================

# Insert SM_121 routing right after "int32_t version_num = get_sm_version_num();"
# in the cutlass_scaled_mm function
# Need to be more specific to target only cutlass_scaled_mm (not cutlass_scaled_mm_azp)
sed -i '/^void cutlass_scaled_mm(torch::Tensor& c, torch::Tensor const& a,$/,/^}$/{
/int32_t version_num = get_sm_version_num();/{
n
a\
\
  // Route SM_121 (GB10) to native scaled_mm kernel\
#if defined ENABLE_SCALED_MM_SM121 && ENABLE_SCALED_MM_SM121\
  if (version_num >= 121 && version_num < 130) {\
    cutlass_scaled_mm_sm121(c, a, b, a_scales, b_scales, bias);\
    return;\
  }\
#endif
}
}' "$DISPATCHER_FILE"

echo "  - SM_121 scaled_mm routing added"

# ============================================================================
# Step 4: Update ENABLE_CUTLASS_MOE_* condition to include GB10
# ============================================================================
sed -i 's/\(#if (defined(ENABLE_CUTLASS_MOE_SM90) && ENABLE_CUTLASS_MOE_SM90) ||   \\\)/\1\n    (defined(ENABLE_CUTLASS_MOE_GB10) \&\& ENABLE_CUTLASS_MOE_GB10) ||    \\/g' "$DISPATCHER_FILE"

echo "  - Preprocessor conditions updated"

# Verify changes
echo ""
echo "Verifying modifications..."
if grep -q "ENABLE_CUTLASS_MOE_GB10" "$DISPATCHER_FILE" && \
   grep -q "ENABLE_SCALED_MM_SM121" "$DISPATCHER_FILE" && \
   grep -q "cutlass_moe_mm_gb10" "$DISPATCHER_FILE" && \
   grep -q "cutlass_scaled_mm_sm121" "$DISPATCHER_FILE"; then
    echo "✓ All modifications verified"
    rm -f "${DISPATCHER_FILE}.backup"
    exit 0
else
    echo "✗ ERROR: Some modifications missing"
    echo "Restoring backup..."
    mv "${DISPATCHER_FILE}.backup" "$DISPATCHER_FILE"
    exit 1
fi
