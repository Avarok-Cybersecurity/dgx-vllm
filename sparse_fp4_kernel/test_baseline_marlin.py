#!/usr/bin/env python3
"""Baseline Marlin benchmark (no v7 sparse patch) for comparison."""

import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
logger = logging.getLogger("baseline")

# No v7 sparse patch — use stock Marlin
os.environ['VLLM_TEST_FORCE_FP8_MARLIN'] = '1'
os.environ['VLLM_NVFP4_GEMM_BACKEND'] = 'marlin'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def main():
    from vllm import LLM, SamplingParams

    logger.info("Loading model with stock Marlin (no sparse)...")
    t0 = time.time()

    llm = LLM(
        model="nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4",
        max_model_len=4096,
        gpu_memory_utilization=0.70,
        enforce_eager=True,
        max_num_seqs=1,
    )

    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")

    # Warmup
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)
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
    logger.info(f"\n  Marlin baseline average: {avg_tok_s:.1f} tok/s")
    logger.info("DONE")


if __name__ == '__main__':
    main()
