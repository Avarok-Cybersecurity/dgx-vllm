#!/usr/bin/env python3
"""
Fix FlashInfer NVFP4 MoE backend auto-selection returning None for experts_cls.

Bug: In select_nvfp4_moe_backend(), the FlashInfer iteration path computes
k_cls = backend_to_kernel_cls(backend) but then returns (backend, None)
instead of (backend, k_cls). This causes an assertion failure in
CompressedTensorsW4A4Nvfp4MoEMethod.process_weights_after_loading()
when using VLLM_USE_FLASHINFER_MOE_FP4=1.

Fix: Return k_cls instead of None for non-TRTLLM FlashInfer backends.
"""

import sys
import os

def patch_nvfp4_oracle():
    # Find the file
    vllm_root = "/app/vllm"
    target = os.path.join(
        vllm_root,
        "vllm/model_executor/layers/fused_moe/oracle/nvfp4.py"
    )

    if not os.path.exists(target):
        print(f"ERROR: {target} not found")
        sys.exit(1)

    with open(target, "r") as f:
        content = f.read()

    # The bug is in the FlashInfer iteration path.
    # Current code returns (backend, None) for ALL FlashInfer backends,
    # but non-TRTLLM backends need their k_cls returned.
    #
    # Find the pattern:
    #   if supported:
    #       logger.info_once(_make_log_backend(backend), scope="local")
    #       return backend, None
    #
    # Replace with:
    #   if supported:
    #       logger.info_once(_make_log_backend(backend), scope="local")
    #       return backend, k_cls

    old = """                if supported:
                    logger.info_once(_make_log_backend(backend), scope="local")
                    return backend, None
                else:
                    logger.debug_once(
                        _make_log_unsupported(backend, reason), scope="local"
                    )"""

    new = """                if supported:
                    logger.info_once(_make_log_backend(backend), scope="local")
                    return backend, k_cls
                else:
                    logger.debug_once(
                        _make_log_unsupported(backend, reason), scope="local"
                    )"""

    if old not in content:
        # Check if already patched
        if "return backend, k_cls" in content:
            print("Already patched")
            return
        print("ERROR: Could not find target pattern in nvfp4.py")
        print("Looking for:")
        print(old[:200])
        sys.exit(1)

    content = content.replace(old, new)

    with open(target, "w") as f:
        f.write(content)

    print(f"Patched {target}: FlashInfer NVFP4 MoE backend now returns k_cls")

if __name__ == "__main__":
    patch_nvfp4_oracle()
