#!/usr/bin/env python3
"""
v112 (simplified v17): Route capability 121 to SM_120 kernels.

Upstream vLLM 0.16.0 already uses >= 120 range checks in most files.
Only scaled_mm_entry.cu needs a minor fix: add upper bound to >= 120 check.

Changes:
  - version_num >= 120 -> version_num >= 120 && version_num < 130
  - Error message update to mention 120
"""

import re
import os
import sys


def fix_scaled_mm_entry(file_path):
    """Fix the primary dispatcher to accept capability 121."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False

    print(f"Patching: {file_path}")

    with open(file_path, 'r') as f:
        content = f.read()

    original_content = content
    modifications = 0

    # Pattern 1: version_num == 120 -> range check
    pattern1 = r'version_num\s*==\s*120\b'
    replacement1 = r'(version_num >= 120 && version_num < 130)'
    content, count1 = re.subn(pattern1, replacement1, content)
    if count1 > 0:
        print(f"  Changed {count1} exact equality check(s) (== 120)")
        modifications += count1

    # Pattern 2: version_num >= 120 without upper bound -> add < 130
    pattern2 = r'(\bversion_num\s*>=\s*120\b)(?!\s*&&\s*version_num\s*<)'
    replacement2 = r'\1 && version_num < 130'
    content, count2 = re.subn(pattern2, replacement2, content)
    if count2 > 0:
        print(f"  Added upper bound to {count2} >= 120 check(s)")
        modifications += count2

    # Pattern 3: Update error message
    error_pattern = r'(Required capability:.*?)90 or 100'
    error_replacement = r'\g<1>90, 100, or 120'
    content, count3 = re.subn(error_pattern, error_replacement, content)
    if count3 > 0:
        print(f"  Updated {count3} error message(s)")
        modifications += count3

    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"  Applied {modifications} modification(s)")
        return True
    else:
        print("  No changes needed (already patched)")
        return False


if __name__ == '__main__':
    vllm_root = sys.argv[1] if len(sys.argv) > 1 else '/app/vllm'

    print("=" * 70)
    print("Capability 121 Routing Fix (v17 simplified)")
    print("=" * 70)
    print(f"vLLM root: {vllm_root}")
    print()

    # Only patch the primary dispatcher - other files already handle 121 upstream
    primary = os.path.join(vllm_root, 'csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu')

    if os.path.exists(primary):
        fixed = fix_scaled_mm_entry(primary)
        if fixed:
            print("\nCapability 121 will now route to SM_120 kernels.")
        else:
            print("\nAlready patched or no changes needed.")
    else:
        print(f"ERROR: Primary dispatcher not found: {primary}")
        sys.exit(1)

    print("=" * 70)
