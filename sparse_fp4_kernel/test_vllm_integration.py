#!/usr/bin/env python3
"""Test v7 sparse MoE patch with full vLLM model loading.

Uses vLLM's offline inference to load the model and run a single generation,
verifying the patch works end-to-end.
"""

import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
logger = logging.getLogger("v7_test")

# Setup environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['VLLM_TEST_FORCE_FP8_MARLIN'] = '1'
os.environ['VLLM_NVFP4_GEMM_BACKEND'] = 'marlin'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['VLLM_NVFP4_SPARSE_V7'] = '1'  # Trigger auto-patch in child processes

# Apply v7 sparse patch
logger.info("Applying v7 sparse MoE patch...")

# Install patch into site-packages so child processes can import it
import shutil
site_dir = '/opt/venv/lib/python3.12/site-packages'
shutil.copy('/workspace/sparse_fp4_kernel/sparse_v7_moe_patch.py', site_dir)

# Create usercustomize.py to auto-apply patch in child processes
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
    # Now import vLLM (after patching)
    from vllm import LLM, SamplingParams

    logger.info("Loading model with v7 sparse MoE...")
    t0 = time.time()

    llm = LLM(
        model="nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4",
        max_model_len=4096,
        gpu_memory_utilization=0.70,  # Lower to accommodate v7 sparse weights
        enforce_eager=True,  # No CUDA graphs for initial test
        max_num_seqs=1,
    )

    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")

    # Test generation
    logger.info("Running test generation...")
    sampling_params = SamplingParams(max_tokens=50, temperature=0.0)

    prompts = [
        "What is the capital of France?",
        "Count from 1 to 10:",
    ]

    for prompt in prompts:
        logger.info(f"\nPrompt: {prompt}")
        t0 = time.time()
        outputs = llm.generate([prompt], sampling_params)
        gen_time = time.time() - t0

        output = outputs[0]
        text = output.outputs[0].text
        n_tokens = len(output.outputs[0].token_ids)
        tok_s = n_tokens / gen_time if gen_time > 0 else 0

        logger.info(f"Output ({n_tokens} tokens, {tok_s:.1f} tok/s): {text[:200]}")

    # Throughput benchmark
    logger.info("\n" + "=" * 70)
    logger.info("Throughput benchmark (5 requests, 100 tokens each)")
    logger.info("=" * 70)

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
    logger.info(f"\n  Average: {avg_tok_s:.1f} tok/s")
    logger.info(f"  Baseline (Marlin+MTP): 59.9 tok/s")
    logger.info(f"  Improvement: {(avg_tok_s/59.9 - 1)*100:+.1f}%")

    logger.info("\nDONE")


if __name__ == '__main__':
    main()
