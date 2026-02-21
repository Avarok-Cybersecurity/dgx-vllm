#!/usr/bin/env python3
"""Test v7 sparse MoE with MTP speculative decoding."""

import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
logger = logging.getLogger("v7_mtp")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['VLLM_TEST_FORCE_FP8_MARLIN'] = '1'
os.environ['VLLM_NVFP4_GEMM_BACKEND'] = 'marlin'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['VLLM_NVFP4_SPARSE_V7'] = '1'

import shutil
site_dir = '/opt/venv/lib/python3.12/site-packages'
shutil.copy('/workspace/sparse_fp4_kernel/sparse_v7_moe_patch.py', site_dir)

usercustomize = os.path.join(site_dir, 'usercustomize.py')
with open(usercustomize, 'w') as f:
    f.write("import os\n")
    f.write("if os.environ.get('VLLM_NVFP4_SPARSE_V7', '0') == '1':\n")
    f.write("    try:\n")
    f.write("        import sparse_v7_moe_patch\n")
    f.write("        sparse_v7_moe_patch.patch_vllm()\n")
    f.write("    except Exception as e:\n")
    f.write("        print(f'[v7 sparse] usercustomize patch failed: {e}')\n")

import sparse_v7_moe_patch
sparse_v7_moe_patch.patch_vllm()


def main():
    from vllm import LLM, SamplingParams

    logger.info("Loading model with v7 sparse MoE + MTP...")
    t0 = time.time()

    llm = LLM(
        model="nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        max_num_seqs=1,
        kv_cache_dtype="fp8",
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 1,
        },
    )

    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")

    # Warmup
    sampling_params = SamplingParams(max_tokens=20, temperature=0.0)
    llm.generate(["Hello"], sampling_params)

    # Benchmark
    logger.info("\nThroughput benchmark (5 requests, 100 tokens each)")
    sampling_params = SamplingParams(max_tokens=100, temperature=0.7)
    prompt = "Explain the theory of relativity in simple terms."

    times = []
    for i in range(5):
        t0 = time.time()
        outputs = llm.generate([prompt], sampling_params)
        gen_time = time.time() - t0
        n_tokens = len(outputs[0].outputs[0].token_ids)
        tok_s = n_tokens / gen_time
        times.append(tok_s)
        logger.info(f"  Run {i+1}: {tok_s:.1f} tok/s ({n_tokens} tokens in {gen_time:.2f}s)")

    avg_tok_s = sum(times) / len(times)
    logger.info(f"\n  V7 sparse + MTP average: {avg_tok_s:.1f} tok/s")
    logger.info(f"  Baseline (Marlin+MTP): 59.9 tok/s")
    logger.info(f"  Baseline (Marlin eager, same container): 31.1 tok/s")
    logger.info(f"  V7 sparse eager: 33.3 tok/s")
    logger.info("DONE")


if __name__ == '__main__':
    main()
