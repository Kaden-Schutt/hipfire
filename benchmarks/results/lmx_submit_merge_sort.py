#!/usr/bin/env python3
"""Submit a Qwen3.5 DFlash merge_sort thinking-OFF bench to localmaxxing.

Generalized version of lmx_submit_27b_merge_sort.py — takes --size {9b|27b}
(default 27b) and submits the matching `<size>-merge-sort-bench-*.json`
with full metrics (decode tok/s + τ + accept_rate + ttft_ms + prefill_tok_s
+ peakVramGb).

Usage:
  LMX_API_KEY=... python3 lmx_submit_merge_sort.py --size 9b
  LMX_API_KEY=... python3 lmx_submit_merge_sort.py --size 27b
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path("/home/kaden/ClaudeCode/autorocm/hipfire/.worktrees/dflash")
BENCH_ROOT = ROOT / "benchmarks" / "results"
PROMPT_FILE = ROOT / "benchmarks/prompts/merge_sort_thinking_off.txt"
API_URL = "https://www.localmaxxing.com/api/benchmarks"

HARDWARE = {
    "hwClass": "DISCRETE_GPU", "gpuName": "RX 7900 XTX", "gpuCount": 1,
    "vramGb": 24, "cpu": "AMD Ryzen 9 3900X", "ramGb": 64, "os": "Ubuntu 24.04",
}

SIZE_TO_HFID = {
    "9b":  "Qwen/Qwen3.5-9B",
    "27b": "Qwen/Qwen3.5-27B",
}
SIZE_TO_DRAFT = {
    "9b":  "qwen3.5-9b-mq4.dflash.hfq",
    "27b": "qwen3.5-27b-mq4.dflash.hfq",
}
SIZE_TO_TARGET = {
    "9b":  "qwen3.5-9b-mq4.hfq",
    "27b": "qwen3.5-27b-mq4.hfq",
}


def submit(payload, api_key):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode() if e.fp else str(e)
    except Exception as e:
        return 0, f"transport: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["9b", "27b"], default="27b",
                    help="model size (picks both bench JSON glob + LMX hfId)")
    args = ap.parse_args()

    api_key = os.environ["LMX_API_KEY"]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log_path = BENCH_ROOT / f"lmx-{args.size}-merge-sort-{stamp}.run.txt"
    submits_path = BENCH_ROOT / f"lmx-{args.size}-merge-sort-{stamp}-submission.json"

    candidates = sorted(BENCH_ROOT.glob(f"{args.size}-merge-sort-bench-*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"No {args.size}-merge-sort-bench-*.json found")
    source = candidates[0]

    engine_ver = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True).strip()

    log_path.write_text("")
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    bench = json.loads(source.read_text())
    median = bench["median"]
    runs = bench["runs"]
    all_tok_s = [r["decode_tok_s"] for r in runs]
    prompt_md5 = bench.get("prompt_md5")

    log(f"=== LMX {args.size}-3.5 merge_sort submission  engine={engine_ver} ===")
    log(f"source: {source.name}")
    log(f"median tok/s: {median['decode_tok_s']:.1f}  τ: {median['decode_tau']:.3f}  ttft: {median['ttft_ms']:.1f}ms")

    notes = "\n".join([
        f"hipfire @ {engine_ver} (master post-PR #51 + #52 series)",
        f"prompt: merge_sort thinking-OFF (md5={prompt_md5})",
        f"  chatml-wrapped + explicit empty <think></think> for thinking-off",
        f"  prompt file: benchmarks/prompts/merge_sort_thinking_off.txt",
        f"runs: {len(runs)} (median reported); range {bench['min_tok_s']:.1f}–{bench['max_tok_s']:.1f}",
        f"  per-run tok/s: {all_tok_s}",
        "decode mode: DFlash speculative (block_size=16)",
        "kv_cache: asym3",
        "prompt_normalize: true (default since 2026-04-26)",
        f"τ (median): {median['decode_tau']:.3f}",
        f"accept_rate (median): {median['decode_accept_rate']:.3f}",
        f"prefill: {median['prefill_secs']*1000:.1f}ms ({median['prefill_tok_s']:.1f} tok/s)",
        f"ttft (excl warmup): {median['ttft_ms']:.1f}ms = prefill + first cycle",
        f"vram: {int(median['vram_used_mb'])} MB used / {int(median['vram_total_mb'])} MB total",
        f"natural EOS at {int(median['decode_tokens_emitted'])} tokens — production-shape bounded code (no loop)",
    ])

    payload = {
        "hfId": SIZE_TO_HFID[args.size],
        "hardware": HARDWARE,
        "engineName": "hipfire",
        "engineVersion": f"0.1.8-alpha+{engine_ver}",
        "backend": "rocm",
        "quantization": "MQ4",
        "tokSOut": median["decode_tok_s"],
        "ttftMs": median["ttft_ms"],
        "promptTokens": int(median["prompt_tokens"]),
        "outputTokens": int(median["decode_tokens_emitted"]),
        "contextLength": 4096,
        "batchSize": 1,
        "peakVramGb": median["vram_used_mb"] / 1024.0,
        "notes": notes[:2000],
        "engineFlags": {
            "kvCache": "asym3",
            "wmma": True,
            "specDecoding": True,
            "specMethod": "DFlash",
            "specBlockSize": 16,
            "specModel": f"hipfire-qwen3.5-{args.size}-mq4.dflash",
            "promptNormalize": True,
            "thinking": "off",
            "prefillTokSPerSec": median["prefill_tok_s"],
            "acceptRate": median["decode_accept_rate"],
            "tau": median["decode_tau"],
            "commandSnippet": (
                "./target/release/examples/dflash_spec_demo "
                f"--target {SIZE_TO_TARGET[args.size]} --draft {SIZE_TO_DRAFT[args.size]} "
                "--prompt $(cat benchmarks/prompts/merge_sort_thinking_off.txt) "
                "--max 256 --no-chatml --kv-mode asym3"
            ),
        },
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    log(f"\n=== submitting ===")
    log(f"  hfId={payload['hfId']}  tokSOut={payload['tokSOut']:.1f}  ttftMs={payload['ttftMs']:.1f}  vramGb={payload['peakVramGb']:.2f}")
    status, body = submit(payload, api_key)
    log(f"  HTTP {status}")
    log(f"  body: {body[:400]}")
    submits_path.write_text(json.dumps({
        "request": payload, "status": status, "response": body,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    log(f"\nsaved: {submits_path}")


if __name__ == "__main__":
    main()
