#!/bin/bash
set -e

echo "════════════════════════════════════════════════════════════"
echo "v115: FIX DISPATCHER FLAG - THE FINAL PIECE!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "ROOT CAUSE (discovered in v114):"
echo "  scaled_mm_entry.cu compiled WITHOUT -DENABLE_SCALED_MM_SM120=1"
echo "  SM_120 kernels compiled WITH the flag"
echo "  Result: Kernels exist, but #ifdef fails in dispatcher!"
echo ""
echo "SOLUTION:"
echo "  Ensure scaled_mm_entry.cu gets compiled with the flag"
echo "  by explicitly setting compile definitions for this file"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

CMAKELISTS="/app/vllm/CMakeLists.txt"

if [ ! -f "$CMAKELISTS" ]; then
    echo "❌ ERROR: CMakeLists.txt not found"
    exit 1
fi

# Backup
cp "$CMAKELISTS" "${CMAKELISTS}.v115.backup"
echo "✓ Backup created"

# Find where scaled_mm_entry.cu is added (before SM_120 section)
# We need to add a separate block that sets the flag for dispatcher files

# Add this AFTER the SM_120 section (after line ~550)
# This ensures that when SM_120 is built, the dispatcher also gets the flag

cat > /tmp/v115_dispatcher_fix.cmake << 'CMAKE_FIX'

# ============================================================================
# v115 FIX: Ensure dispatcher knows about SM_120 kernels
# ============================================================================
# Problem: VLLM_GPU_FLAGS is set, but scaled_mm_entry.cu might be compiled
# before the flag is applied. Solution: Explicitly set compile definitions
# for the dispatcher file after SM_120 section.
# ============================================================================
if(ENABLE_SCALED_MM_SM120 OR TARGET_SM120_BUILT)
  # Find scaled_mm_entry.cu in the sources and add compile definition
  set(DISPATCHER_FILE "csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu")

  # Add compile definition specifically for this file
  set_source_files_properties(
    ${DISPATCHER_FILE}
    PROPERTIES
    COMPILE_DEFINITIONS "ENABLE_SCALED_MM_SM120=1"
  )

  message(STATUS "v115: Set ENABLE_SCALED_MM_SM120=1 for ${DISPATCHER_FILE}")
endif()

CMAKE_FIX

# Find the SM_120 section and insert after its endif()
line_num=$(grep -n "Building scaled_mm_c3x_sm120 for archs" "$CMAKELISTS" | cut -d: -f1)

if [ -z "$line_num" ]; then
    # Fallback: look for ENABLE_SCALED_MM_SM120
    line_num=$(grep -n "ENABLE_SCALED_MM_SM120" "$CMAKELISTS" | tail -1 | cut -d: -f1)
fi

if [ -z "$line_num" ]; then
    echo "WARNING: Could not find SM_120 section - appending to end"
    cat /tmp/v115_dispatcher_fix.cmake >> "$CMAKELISTS"
else
    # Find the next endif() after the SM_120 message line
    insert_line=$(tail -n "+${line_num}" "$CMAKELISTS" | grep -n "^endif()" | head -1 | cut -d: -f1)
    if [ -n "$insert_line" ]; then
        actual_line=$((line_num + insert_line - 1))
        sed -i "${actual_line}r /tmp/v115_dispatcher_fix.cmake" "$CMAKELISTS"
        echo "Added dispatcher fix after endif() at line $actual_line"
    else
        # Fallback: insert 10 lines after the message
        insert_line=$((line_num + 10))
        sed -i "${insert_line}r /tmp/v115_dispatcher_fix.cmake" "$CMAKELISTS"
        echo "Added dispatcher fix at line $insert_line (offset fallback)"
    fi
fi
echo ""
echo "What this does:"
echo "  1. After SM_120 section completes"
echo "  2. Explicitly set ENABLE_SCALED_MM_SM120=1 for scaled_mm_entry.cu"
echo "  3. Ensures #ifdef block in dispatcher evaluates to TRUE"
echo ""
echo "Expected result:"
echo "  Dispatcher compiled WITH flag → #ifdef passes → routes 121 to SM_120!"
echo ""
echo "════════════════════════════════════════════════════════════"
