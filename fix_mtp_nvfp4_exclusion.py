#!/usr/bin/env python3
"""
Fix MTP layer exclusion for ModelOpt NVFP4 quantization.

Bug: The NVFP4 checkpoint exclude list has 'mtp.layers.0*' to skip MTP decoder
layers, but ALL MTP weights (including mtp.fc, mtp.layers.0.self_attn.*, etc.)
are stored as BF16 in the checkpoint. The pattern 'mtp.layers.0*' does NOT
match 'mtp.fc', so the fc layer gets initialized as FP4 quantized while the
checkpoint has BF16 weights — causing a shape mismatch assertion failure.

Fix: In is_layer_excluded(), if any exclude pattern starts with 'mtp.' and the
current prefix also starts with 'mtp.', exclude the entire MTP module. All MTP
weights in NVFP4 checkpoints are BF16 (unquantized).
"""

import sys
import os


def patch_modelopt():
    target = "/app/vllm/vllm/model_executor/layers/quantization/modelopt.py"

    if not os.path.exists(target):
        print(f"ERROR: {target} not found")
        sys.exit(1)

    with open(target, "r") as f:
        content = f.read()

    # The bug is in is_layer_excluded(). The wildcard 'mtp.layers.0*' only
    # matches mtp.layers.0.XXX but misses mtp.fc and other MTP layers.
    # Since ALL MTP weights are BF16, we exclude the entire mtp.* prefix.
    old = """        # modelopt exclude modules are not simple strings, they are wildcards
        for wildcard_pattern in self.exclude_modules:
            if fnmatch(prefix, wildcard_pattern):
                return True

        return False"""

    new = """        # modelopt exclude modules are not simple strings, they are wildcards
        for wildcard_pattern in self.exclude_modules:
            if fnmatch(prefix, wildcard_pattern):
                return True

        # All MTP weights in NVFP4 checkpoints are BF16 (unquantized).
        # The exclude list only has 'mtp.layers.0*' which misses 'mtp.fc'.
        # If any exclude pattern targets mtp.*, exclude all mtp.* layers.
        if prefix.startswith("mtp."):
            for wildcard_pattern in self.exclude_modules:
                if wildcard_pattern.startswith("mtp."):
                    return True

        return False"""

    if old not in content:
        if "prefix.startswith(\"mtp.\")" in content:
            print("Already patched")
            return
        print("ERROR: Could not find target pattern in modelopt.py")
        print("Looking for:")
        print(old[:200])
        sys.exit(1)

    content = content.replace(old, new)

    with open(target, "w") as f:
        f.write(content)

    print(f"Patched {target}: ALL MTP layers now excluded from NVFP4")


if __name__ == "__main__":
    patch_modelopt()
