#!/usr/bin/env python3
"""
Full NVFP4 Benchmark Suite — TTFT, TPOT, Throughput, CUDA Graph Info

Tests a 2D matrix of (ISL, OSL) configurations along the Pareto frontier:
  - Prefill-heavy:  high input, low output  (RAG, summarization)
  - Balanced:        similar input/output    (chat, translation)
  - Decode-heavy:    low input, high output  (code gen, reasoning)

Measures TTFT via streaming, token counts via API usage field.
Extracts CUDA graph capture and compilation details from container logs.

Industry-standard configs based on NVIDIA NIM, MLPerf, vLLM, SGLang,
SemiAnalysis InferenceMAX, and Artificial Analysis methodologies.
"""

import json
import re
import time
import sys
import subprocess
import urllib.request

import os

PORT = int(os.environ.get("BENCH_PORT", "8888"))
URL_BASE = f"http://localhost:{PORT}"
URL_CHAT = f"{URL_BASE}/v1/chat/completions"
MODEL = None

# ---- Configuration ----
WARMUP_RUNS = 2
BENCH_RUNS = 3
MAX_MODEL_LEN = int(os.environ.get("BENCH_MAX_MODEL_LEN", "131072"))
CONTAINER_NAME = os.environ.get("BENCH_CONTAINER", "bench-nvfp4")
RESULTS_FILE = os.environ.get("BENCH_RESULTS_FILE", "/workspace/bench-nvfp4-results.json")

# ---- Pareto Frontier Test Matrix ----
# (ISL, OSL, label) — ISL + OSL must fit within MAX_MODEL_LEN (128k)
# Ordered by regime: prefill-heavy → balanced → decode-heavy
# Based on NVIDIA NIM, MLPerf, vLLM, SGLang, SemiAnalysis methodologies
TEST_MATRIX = [
    # Prefill-heavy (ISL >> OSL)
    (65536,  256, "64k prefill stress"),
    (32768,  256, "32k prefill stress"),
    (16384,  512, "16k RAG / document"),
    (8192,  1024, "8k summarization (SemiAnalysis)"),
    (4096,   256, "4k short summary"),
    (1024,   128, "vLLM default"),

    # Balanced (ISL ~ OSL)
    (128,    128, "micro-benchmark baseline"),
    (256,    256, "short balanced"),
    (1024,  1024, "standard chat (vLLM/SGLang)"),
    (4096,  4096, "4k balanced"),
    (8192,  8192, "8k balanced"),

    # Decode-heavy (ISL << OSL)
    (128,   1024, "code generation"),
    (256,   4096, "long-form reasoning"),
    (1024,  8192, "1k→8k decode (SemiAnalysis)"),
    (256,  16384, "max decode stress"),
    (256,  65536, "64k decode stress"),
]

# Filler text (~4 chars per token average)
FILLER_WORD = "The quick brown fox jumps over the lazy dog. "
PROMPT_SUFFIX = "\n\nProvide a very detailed and comprehensive analysis. Do not stop early. Cover every aspect in depth."


def detect_model():
    """Auto-detect the model name from the server."""
    global MODEL
    for i in range(120):
        try:
            with urllib.request.urlopen(f"{URL_BASE}/v1/models", timeout=2) as r:
                data = json.loads(r.read())
                MODEL = data["data"][0]["id"]
                return True
        except Exception:
            if i % 12 == 0:
                print(f"  Waiting for server... ({i*5}s)")
            time.sleep(5)
    return False


def get_cuda_graph_info() -> dict:
    """Extract CUDA graph capture details from container logs."""
    info = {
        "cudagraph_mode_requested": None,
        "cudagraph_mode_actual": None,
        "graphs_captured": 0,
        "capture_sizes": [],
        "compilation_mode": None,
        "spec_decode_info": [],
        "raw_lines": [],
    }
    try:
        result = subprocess.run(
            ["sudo", "docker", "logs", CONTAINER_NAME],
            capture_output=True, text=True, timeout=30
        )
        logs = result.stderr + result.stdout
    except Exception:
        return info

    for line in logs.splitlines():
        ll = line.lower().strip()

        if "compilation_config" in ll and "cudagraph_mode" in ll:
            m = re.search(r"cudagraph_mode.*?<CUDAGraphMode\.(\w+):", line)
            if m:
                info["cudagraph_mode_requested"] = m.group(1)
            m = re.search(r"cudagraph_capture_sizes.*?\[([\d, ]+)\]", line)
            if m:
                info["capture_sizes"] = [int(x.strip()) for x in m.group(1).split(",")]
                info["graphs_captured"] = len(info["capture_sizes"])
            m = re.search(r"mode.*?<CompilationMode\.(\w+):", line)
            if m:
                info["compilation_mode"] = m.group(1)

        if "setting cudagraph_mode" in ll:
            m = re.search(r"cudagraph_mode=(\w+)", line)
            if m:
                info["cudagraph_mode_actual"] = m.group(1)
            info["raw_lines"].append(line.strip())

        if "capturing cuda graphs" in ll:
            info["raw_lines"].append(line.strip())

        if "specdecoding metrics" in ll:
            info["spec_decode_info"].append(line.strip())

    info["spec_decode_info"] = info["spec_decode_info"][-5:]
    capture_lines = [l for l in info["raw_lines"] if "capturing" in l.lower()]
    other_lines = [l for l in info["raw_lines"] if "capturing" not in l.lower()]
    if len(capture_lines) > 2:
        info["raw_lines"] = other_lines + [capture_lines[0], f"  ... ({len(capture_lines)-2} more)", capture_lines[-1]]

    return info


def make_prompt(target_tokens: int) -> str:
    """Generate a prompt that's approximately target_tokens long."""
    chars_needed = target_tokens * 4
    repeats = max(1, chars_needed // len(FILLER_WORD))
    filler = (FILLER_WORD * repeats)[:chars_needed]
    return f"Analyze the following text thoroughly:\n\n{filler}{PROMPT_SUFFIX}"


def bench_request(prompt: str, max_tokens: int) -> dict:
    """
    Send a streaming request with usage reporting.
    TTFT from first content event, token counts from API usage field.
    """
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()

    req = urllib.request.Request(
        URL_CHAT,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t_start = time.perf_counter()
    t_first_token = None
    completion_tokens = 0
    prompt_tokens = 0

    resp = urllib.request.urlopen(req, timeout=600)

    buffer = b""
    while True:
        chunk = resp.read(4096)
        if not chunk:
            break
        buffer += chunk

        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.decode("utf-8", errors="replace").strip()

            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break

            try:
                event = json.loads(data_str)

                choices = event.get("choices", [])
                if choices and t_first_token is None:
                    delta = choices[0].get("delta", {})
                    if delta.get("content"):
                        t_first_token = time.perf_counter()

                usage = event.get("usage")
                if usage:
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            except json.JSONDecodeError:
                pass

    t_end = time.perf_counter()
    resp.close()

    total_time = t_end - t_start
    ttft = (t_first_token - t_start) if t_first_token else total_time
    decode_time = (t_end - t_first_token) if t_first_token else total_time

    if completion_tokens > 1:
        tpot = decode_time / (completion_tokens - 1)
    else:
        tpot = 0.0

    e2e_throughput = completion_tokens / total_time if total_time > 0 else 0.0
    decode_throughput = completion_tokens / decode_time if decode_time > 0 else 0.0

    return {
        "ttft_ms": ttft * 1000,
        "tpot_ms": tpot * 1000,
        "e2e_throughput": e2e_throughput,
        "decode_throughput": decode_throughput,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_time_s": total_time,
        "decode_time_s": decode_time,
    }


def main():
    print("=" * 80)
    print("  NVFP4 Pareto Frontier Benchmark — TTFT / TPOT / Throughput")
    print("=" * 80)
    print()

    # ---- Wait for server ----
    print("Waiting for server...")
    if not detect_model():
        print("ERROR: Server not ready after 10 minutes")
        sys.exit(1)

    print(f"\nModel: {MODEL}")

    # ---- CUDA Graph Info ----
    print("\n--- CUDA Graph & Compilation ---")
    graph_info = get_cuda_graph_info()
    print(f"  Compilation mode:    {graph_info['compilation_mode'] or '(unknown)'}")
    print(f"  CUDAGraph requested: {graph_info['cudagraph_mode_requested'] or '(unknown)'}")
    print(f"  CUDAGraph actual:    {graph_info['cudagraph_mode_actual'] or graph_info['cudagraph_mode_requested'] or '(unknown)'}")
    print(f"  Capture sizes:       {graph_info['graphs_captured']} graphs ({graph_info['capture_sizes'][:5]}{'...' if len(graph_info['capture_sizes']) > 5 else ''})")
    if graph_info["raw_lines"]:
        for line in graph_info["raw_lines"]:
            print(f"  > {line}")
    if graph_info["spec_decode_info"]:
        print(f"\n--- Recent Speculative Decoding Metrics ---")
        for line in graph_info["spec_decode_info"][-3:]:
            idx = line.find("SpecDecoding metrics:")
            if idx >= 0:
                print(f"  {line[idx:]}")

    # ---- Validate test matrix ----
    valid_configs = []
    skipped = []
    for isl, osl, label in TEST_MATRIX:
        if isl + osl > MAX_MODEL_LEN:
            skipped.append((isl, osl, label, f"ISL+OSL={isl+osl} > max_model_len={MAX_MODEL_LEN}"))
        else:
            valid_configs.append((isl, osl, label))

    print(f"\n--- Benchmark Configuration ---")
    print(f"  Max model len:  {MAX_MODEL_LEN}")
    print(f"  Warmup runs:    {WARMUP_RUNS}")
    print(f"  Bench runs:     {BENCH_RUNS}")
    print(f"  Test configs:   {len(valid_configs)} valid, {len(skipped)} skipped")
    print()
    print(f"  {'ISL':>5} {'OSL':>5}  {'Category'}")
    print(f"  {'---':>5} {'---':>5}  {'--------'}")
    for isl, osl, label in valid_configs:
        regime = "PREFILL" if isl > osl * 2 else ("DECODE" if osl > isl * 2 else "BALANCED")
        print(f"  {isl:>5} {osl:>5}  {label} [{regime}]")
    for isl, osl, label, reason in skipped:
        print(f"  {isl:>5} {osl:>5}  {label} [SKIPPED: {reason}]")
    print()

    # ---- Warmup ----
    print("Warming up...")
    for i in range(WARMUP_RUNS):
        try:
            r = bench_request("Hello, how are you? Please respond in detail.", 50)
            print(f"  warmup {i+1}: {r['e2e_throughput']:.1f} tok/s, {r['completion_tokens']} tokens, TTFT={r['ttft_ms']:.0f}ms")
        except Exception as e:
            print(f"  warmup {i+1}: failed ({e})")
    print()

    # ---- Benchmark by regime ----
    all_results = []

    # Group by regime for display
    prefill_configs = [(i, o, l) for i, o, l in valid_configs if i > o * 2]
    balanced_configs = [(i, o, l) for i, o, l in valid_configs if not (i > o * 2) and not (o > i * 2)]
    decode_configs = [(i, o, l) for i, o, l in valid_configs if o > i * 2]

    for regime_name, configs in [("PREFILL-HEAVY", prefill_configs), ("BALANCED", balanced_configs), ("DECODE-HEAVY", decode_configs)]:
        if not configs:
            continue

        print(f"  === {regime_name} ===")
        hdr = f"  {'ISL':>5} {'OSL':>5} | {'TTFT(ms)':>9} {'TPOT(ms)':>9} | {'Dec':>7} {'E2E':>7} | {'OutTok':>6} {'InTok':>6}  {'Label'}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for isl, osl, label in configs:
            prompt = make_prompt(isl)
            run_results = []

            for run in range(BENCH_RUNS):
                try:
                    r = bench_request(prompt, osl)
                    run_results.append(r)
                except Exception as e:
                    print(f"  [ERROR {isl}/{osl} run {run+1}]: {e}")

            if not run_results:
                print(f"  {isl:>5} {osl:>5} | {'FAILED':>9}")
                continue

            n = len(run_results)
            avg_f = lambda key: sum(r[key] for r in run_results) / n
            min_f = lambda key: min(r[key] for r in run_results)
            max_f = lambda key: max(r[key] for r in run_results)

            a_ttft = avg_f("ttft_ms")
            a_tpot = avg_f("tpot_ms")
            a_dtput = avg_f("decode_throughput")
            a_etput = avg_f("e2e_throughput")
            a_ctok = avg_f("completion_tokens")
            a_ptok = avg_f("prompt_tokens")

            print(f"  {isl:>5} {osl:>5} | {a_ttft:>8.0f}ms {a_tpot:>8.2f}ms | {a_dtput:>6.1f}/s {a_etput:>6.1f}/s | {a_ctok:>6.0f} {a_ptok:>6.0f}  {label}")

            all_results.append({
                "regime": regime_name.lower().replace("-", "_"),
                "label": label,
                "isl_target": isl,
                "osl_target": osl,
                "isl_actual": round(a_ptok),
                "osl_actual": round(a_ctok),
                "ttft_ms": {"avg": round(a_ttft, 1), "min": round(min_f("ttft_ms"), 1), "max": round(max_f("ttft_ms"), 1)},
                "tpot_ms": {"avg": round(a_tpot, 2), "min": round(min_f("tpot_ms"), 2), "max": round(max_f("tpot_ms"), 2)},
                "decode_throughput": {"avg": round(a_dtput, 1), "min": round(min_f("decode_throughput"), 1), "max": round(max_f("decode_throughput"), 1)},
                "e2e_throughput": {"avg": round(a_etput, 1), "min": round(min_f("e2e_throughput"), 1), "max": round(max_f("e2e_throughput"), 1)},
            })

        print()

    # ---- Summary ----
    print("=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n  {'ISL':>5}/{'OSL':<5} | {'TTFT':>8} | {'TPOT':>8} | {'Decode':>8} | {'E2E':>8} | {'Regime'}")
    print(f"  {'':>5} {'':>5} | {'(ms)':>8} | {'(ms)':>8} | {'(tok/s)':>8} | {'(tok/s)':>8} |")
    print("  " + "-" * 70)
    for r in all_results:
        print(f"  {r['isl_target']:>5}/{r['osl_target']:<5} | {r['ttft_ms']['avg']:>8.0f} | {r['tpot_ms']['avg']:>8.2f} | {r['decode_throughput']['avg']:>8.1f} | {r['e2e_throughput']['avg']:>8.1f} | {r['regime']}")
    print()

    # ---- Key insights ----
    if all_results:
        best_decode = max(all_results, key=lambda r: r["decode_throughput"]["avg"])
        best_e2e = max(all_results, key=lambda r: r["e2e_throughput"]["avg"])
        best_ttft = min(all_results, key=lambda r: r["ttft_ms"]["avg"])
        best_tpot = min(all_results, key=lambda r: r["tpot_ms"]["avg"])

        print("  Key findings:")
        print(f"    Best decode throughput: {best_decode['decode_throughput']['avg']:.1f} tok/s ({best_decode['isl_target']}/{best_decode['osl_target']} — {best_decode['label']})")
        print(f"    Best E2E throughput:    {best_e2e['e2e_throughput']['avg']:.1f} tok/s ({best_e2e['isl_target']}/{best_e2e['osl_target']} — {best_e2e['label']})")
        print(f"    Best TTFT:             {best_ttft['ttft_ms']['avg']:.0f}ms ({best_ttft['isl_target']}/{best_ttft['osl_target']} — {best_ttft['label']})")
        print(f"    Best TPOT:             {best_tpot['tpot_ms']['avg']:.2f}ms ({best_tpot['isl_target']}/{best_tpot['osl_target']} — {best_tpot['label']})")
        print()

    # ---- Final spec decode metrics ----
    print("--- Final Speculative Decoding Metrics ---")
    try:
        result = subprocess.run(
            ["sudo", "docker", "logs", CONTAINER_NAME],
            capture_output=True, text=True, timeout=30
        )
        logs = result.stderr + result.stdout
        spec_lines = [l.strip() for l in logs.splitlines() if "SpecDecoding metrics" in l]
        for line in spec_lines[-3:]:
            idx = line.find("SpecDecoding metrics:")
            if idx >= 0:
                print(f"  {line[idx:]}")
    except Exception:
        print("  (could not read spec decode metrics)")
    print()

    # ---- Save JSON ----
    outfile = RESULTS_FILE
    output = {
        "model": MODEL,
        "max_model_len": MAX_MODEL_LEN,
        "bench_runs": BENCH_RUNS,
        "cuda_graph_info": {
            "compilation_mode": graph_info.get("compilation_mode"),
            "cudagraph_mode_requested": graph_info.get("cudagraph_mode_requested"),
            "cudagraph_mode_actual": graph_info.get("cudagraph_mode_actual"),
            "capture_sizes": graph_info.get("capture_sizes", []),
            "num_graphs": graph_info.get("graphs_captured", 0),
        },
        "test_matrix": [{"isl": i, "osl": o, "label": l} for i, o, l in valid_configs],
        "results": all_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Full results saved to {outfile}")


if __name__ == "__main__":
    main()
