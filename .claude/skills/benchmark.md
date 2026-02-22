# Benchmark Skill

## Trigger

Only activate this skill when the user explicitly says one of the following:
- "begin benchmarking"
- "run benchmarks"
- "benchmark this"
- "run the benchmark suite"

This is NOT a slash command. Do not trigger on general discussion about benchmarks or performance. Only trigger when the user gives a direct instruction to execute benchmarks.

## Methodology

The benchmark suite uses `sparse_fp4_kernel/bench_full.py` (located at `/workspace/dgx-vllm/sparse_fp4_kernel/bench_full.py`).

### How it works

- Sends streaming HTTP requests to the vLLM/TRT-LLM OpenAI-compatible API with `temperature=0.0` (deterministic)
- Performs **2 warmup runs** followed by **3 bench runs** per ISL/OSL configuration
- Measures via Server-Sent Events (SSE) streaming:
  - **TTFT** (Time To First Token): wall-clock time from request send to first content delta in the SSE stream
  - **TPOT** (Time Per Output Token): `decode_time / (completion_tokens - 1)`
  - **Decode throughput** (tok/s): `completion_tokens / decode_time`
  - **E2E throughput** (tok/s): `completion_tokens / total_time` (includes prefill)
- Uses synthetic prompts of varying lengths to hit target ISL (Input Sequence Length)
- Tests across a Pareto frontier: prefill-heavy, balanced, and decode-heavy workloads
- Auto-detects the model name from the `/v1/models` endpoint
- Extracts CUDA graph capture and speculative decoding metrics from container logs

## Standard Test Matrix (ISL/OSL pairs)

### Core configs (always run):

| ISL | OSL | Category | Label |
|-----|-----|----------|-------|
| 128 | 128 | Balanced | micro-benchmark baseline (peak decode speed) |
| 256 | 256 | Balanced | short balanced |
| 1024 | 1024 | Balanced | standard chat --- **primary comparison point** |
| 1024 | 128 | Prefill-heavy | vLLM default |
| 128 | 1024 | Decode-heavy | code generation |

### Extended configs (run when MAX_MODEL_LEN allows):

| ISL | OSL | Category | Label |
|-----|-----|----------|-------|
| 4096 | 256 | Prefill-heavy | 4k short summary |
| 4096 | 4096 | Balanced | 4k balanced |
| 8192 | 1024 | Prefill-heavy | 8k summarization (SemiAnalysis) |
| 8192 | 8192 | Balanced | 8k balanced |
| 16384 | 512 | Prefill-heavy | 16k RAG / document |
| 256 | 4096 | Decode-heavy | long-form reasoning |
| 1024 | 8192 | Decode-heavy | 1k to 8k decode (SemiAnalysis) |
| 32768 | 256 | Prefill-heavy | 32k prefill stress |
| 256 | 16384 | Decode-heavy | max decode stress |
| 65536 | 256 | Prefill-heavy | 64k prefill stress |
| 256 | 65536 | Decode-heavy | 64k decode stress |

Configs where ISL + OSL exceeds MAX_MODEL_LEN are automatically skipped by the script.

## Required Columns in All Benchmark Reports

Every benchmark report or comparison table MUST include ALL of the following columns. Do not omit any.

| Column | Source | Unit |
|--------|--------|------|
| Decode throughput | bench_full.py output | tok/s |
| TPOT | bench_full.py output | ms |
| Model memory | Container logs: `"Model loading took X GiB"` | GiB |
| Available KV cache | Container logs: `"Available KV cache memory: X GiB"` | GiB |
| KV cache tokens | Container logs: `"GPU KV cache size: X tokens"` | tokens |

To extract memory metrics from container logs:

```bash
sudo docker logs <CONTAINER_NAME> 2>&1 | grep -E "Model loading took|Available KV cache memory|GPU KV cache size"
```

## How to Run

### Step 1: Set environment variables

```bash
export BENCH_CONTAINER=<container_name>       # e.g., dgx-vllm-nvfp4
export BENCH_RESULTS_FILE=/workspace/dgx-vllm/sparse_fp4_kernel/bench_<name>_results.json
export BENCH_MAX_MODEL_LEN=<match container>  # e.g., 4096, 131072
export BENCH_PORT=8888                        # default is 8888
export PYTHONUNBUFFERED=1                     # real-time output
```

### Step 2: Verify the container is ready

```bash
curl -s http://localhost:${BENCH_PORT:-8888}/v1/models | python3 -m json.tool
```

The container must show "Application startup complete" in its logs before benchmarks will succeed. The bench script will wait up to 10 minutes for the server, but it is better to verify readiness first.

### Step 3: Run the benchmark

```bash
python3 /workspace/dgx-vllm/sparse_fp4_kernel/bench_full.py
```

### Step 4: Extract memory metrics from container logs

```bash
sudo docker logs ${BENCH_CONTAINER} 2>&1 | grep -E "Model loading took|Available KV cache memory|GPU KV cache size"
```

### Step 5: Save and report results

The script automatically saves a JSON file to the path specified by BENCH_RESULTS_FILE. Include the memory metrics from Step 4 in any summary table or report.

## Baseline Reference Data

Use these as comparison points when reporting results:

| Configuration | Decode (tok/s) @ 1024/1024 | Model Memory (GiB) | KV Cache (GiB) | Notes |
|--------------|---------------------------|--------------------|--------------------|-------|
| v22 Marlin (no MTP) | 41.8 | 44.2 | 62.2 | NVFP4 via Marlin W4A16 dequant |
| v22 Marlin + MTP (2 tokens) | ~67 | 44.2 | 62.2 | MTP speculative decoding |
| v8 Sparse (no MTP) | 41.2 | 35.2 | 70.1 | Sparse FP4 kernel |
| TRT-LLM NVFP4 v1.3.0rc2 | 29.6 | N/A | N/A | CUTLASS SM120 ceiling |
| vLLM FP8 (Qwen3-Next) | ~34 | ~102 | N/A | Different quantization |

## Important Notes

1. **Apples-to-apples comparisons only**: When comparing two configurations, they MUST use the same MAX_MODEL_LEN, same GPU_MEMORY_UTIL, and the same benchmark script. Different MAX_MODEL_LEN values change KV cache allocation and affect throughput.

2. **Memory metrics are as important as throughput**: A configuration that uses less model memory leaves more room for KV cache, which means more concurrent sequences and higher batch throughput. Always include memory columns.

3. **Use PYTHONUNBUFFERED=1**: Without this, Python buffers stdout and you will not see real-time benchmark progress.

4. **Save results JSON files in the repo**: All benchmark result JSON files should be saved under `/workspace/dgx-vllm/sparse_fp4_kernel/` for future reference and regression tracking.

5. **Container must be fully started**: The server must report "Application startup complete" in its logs. Model loading (safetensor shards), KV cache initialization, and CUDA graph capture must all finish before benchmarks produce valid results.

6. **First request after startup may be slow**: CUDA graph compilation can happen on the first real request. The 2 warmup runs in the script handle this, but be aware if running manual curl tests first.

7. **temperature=0.0 is mandatory**: The benchmark uses deterministic decoding. Do not change this, as it ensures reproducible results across runs.
