#!/bin/bash
# Benchmark vLLM NVFP4 with FlashInfer CUTLASS MoE backend
set -euo pipefail

MODEL="nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4"
PORT=8888
API="http://localhost:${PORT}/v1/chat/completions"
RESULTS="/workspace/vllm-nvfp4-cutlass-results.log"

echo "=== vLLM NVFP4 FlashInfer CUTLASS Benchmark ===" | tee "$RESULTS"
echo "Date: $(date)" | tee -a "$RESULTS"
echo "Model: $MODEL" | tee -a "$RESULTS"
echo "Backend: FlashInfer CUTLASS MoE + FLASHINFER attention" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# Verify server is up
if ! curl -s http://localhost:${PORT}/v1/models > /dev/null 2>&1; then
    echo "ERROR: Server not ready at port ${PORT}"
    exit 1
fi

# Warm-up
echo "=== Warm-up (3 runs) ===" | tee -a "$RESULTS"
for i in 1 2 3; do
    start=$(date +%s%N)
    resp=$(curl -s "$API" -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 5.\"}],\"max_tokens\":30,\"temperature\":0.1}")
    end=$(date +%s%N)
    ms=$(( (end - start) / 1000000 ))
    toks=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo "0")
    echo "  Warmup $i: ${toks} tokens in ${ms}ms" | tee -a "$RESULTS"
done

sleep 3

# Benchmark: 200 tokens x 8 runs
echo "" | tee -a "$RESULTS"
echo "=== Benchmark: 200 tokens x 8 runs ===" | tee -a "$RESULTS"
total_toks=0
total_ms=0

for i in $(seq 1 8); do
    start=$(date +%s%N)
    resp=$(curl -s "$API" -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed technical explanation of how transformer neural networks process natural language, including attention mechanisms and positional encoding.\"}],\"max_tokens\":200,\"temperature\":0.1}")
    end=$(date +%s%N)
    ms=$(( (end - start) / 1000000 ))
    toks=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo "0")

    if [ "$toks" -gt 0 ] && [ "$ms" -gt 0 ]; then
        tps=$(python3 -c "print(f'{$toks / ($ms / 1000.0):.2f}')")
        echo "  Run $i: ${toks} tokens in ${ms}ms = ${tps} tok/s" | tee -a "$RESULTS"
        total_toks=$((total_toks + toks))
        total_ms=$((total_ms + ms))
    else
        echo "  Run $i: FAILED - response: $(echo "$resp" | head -c 200)" | tee -a "$RESULTS"
    fi
done

# Extended benchmark: 500 tokens x 3 runs
echo "" | tee -a "$RESULTS"
echo "=== Benchmark: 500 tokens x 3 runs ===" | tee -a "$RESULTS"
for i in 1 2 3; do
    start=$(date +%s%N)
    resp=$(curl -s "$API" -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a comprehensive essay on the history of computing, from early mechanical calculators through the invention of the transistor, integrated circuits, and modern microprocessors. Include technical details about how each advancement improved computing capability.\"}],\"max_tokens\":500,\"temperature\":0.1}")
    end=$(date +%s%N)
    ms=$(( (end - start) / 1000000 ))
    toks=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo "0")

    if [ "$toks" -gt 0 ] && [ "$ms" -gt 0 ]; then
        tps=$(python3 -c "print(f'{$toks / ($ms / 1000.0):.2f}')")
        echo "  Run $i: ${toks} tokens in ${ms}ms = ${tps} tok/s" | tee -a "$RESULTS"
        total_toks=$((total_toks + toks))
        total_ms=$((total_ms + ms))
    else
        echo "  Run $i: FAILED" | tee -a "$RESULTS"
    fi
done

if [ "$total_toks" -gt 0 ] && [ "$total_ms" -gt 0 ]; then
    avg_tps=$(python3 -c "print(f'{$total_toks / ($total_ms / 1000.0):.2f}')")
    echo "" | tee -a "$RESULTS"
    echo "============================================" | tee -a "$RESULTS"
    echo "RESULT: ${avg_tps} tok/s (avg over all runs)" | tee -a "$RESULTS"
    echo "Total: ${total_toks} tokens in ${total_ms}ms" | tee -a "$RESULTS"
    echo "" | tee -a "$RESULTS"
    echo "COMPARISON:" | tee -a "$RESULTS"
    echo "  vLLM FP8 baseline:        ~34.4 tok/s" | tee -a "$RESULTS"
    echo "  vLLM FP8 best (v48):      ~39.4 tok/s" | tee -a "$RESULTS"
    echo "  TRT-LLM NVFP4 baseline:   ~28.7 tok/s" | tee -a "$RESULTS"
    echo "  vLLM NVFP4 CUTLASS (this): ${avg_tps} tok/s" | tee -a "$RESULTS"

    improvement_v14=$(python3 -c "print(f'{(($total_toks / ($total_ms / 1000.0)) / 34.4 - 1) * 100:.1f}')")
    improvement_trtllm=$(python3 -c "print(f'{(($total_toks / ($total_ms / 1000.0)) / 28.7 - 1) * 100:.1f}')")
    echo "  vs vLLM FP8: ${improvement_v14}%" | tee -a "$RESULTS"
    echo "  vs TRT-LLM NVFP4: ${improvement_trtllm}%" | tee -a "$RESULTS"
    echo "============================================" | tee -a "$RESULTS"
fi
