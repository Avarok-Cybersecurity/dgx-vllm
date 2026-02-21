#!/usr/bin/env python3
"""vLLM serve wrapper that applies v7 sparse MoE patch before model loading.

Usage:
  python vllm_serve_v7.py serve nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 \
    --port 8888 --max-model-len 4096 ...

All arguments after the script name are passed to vLLM CLI.
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v7_sparse")

# Ensure our kernel module is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set env to trigger auto-patching
os.environ['VLLM_NVFP4_SPARSE_V7'] = '1'
# Use Marlin backend as base (we'll replace fused_experts after weight loading)
os.environ['VLLM_TEST_FORCE_FP8_MARLIN'] = '1'
os.environ['VLLM_NVFP4_GEMM_BACKEND'] = 'marlin'
# Performance settings
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

logger.info("Importing and applying v7 sparse MoE patch...")
import sparse_v7_moe_patch
sparse_v7_moe_patch.patch_vllm()
logger.info("Patch applied. Starting vLLM...")

# Pass remaining args to vLLM CLI
from vllm.scripts import main
main()
