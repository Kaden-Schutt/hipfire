#!/usr/bin/env python3
"""Long-ctx + sidecar AR bench sweep — the realistic deployment config.

5 models × AR @ max_seq=32768, sidecar on (where available), budget=2048.
Output written to benchmarks/results/<stamp>-longctx.json.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PER_MODEL_CFG = Path.home() / ".hipfire" / "per_model_config.json"
PER_MODEL_BAK = Path("/tmp/per_model_config.before_longctx.bak.json")
STAMP = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
BENCH_ROOT = Path("/home/kaden/ClaudeCode/autorocm/hipfire/.worktrees/dflash/benchmarks/results")
RESULTS_PATH = BENCH_ROOT / f"lmx-longctx-{STAMP}.json"
LOG_PATH = BENCH_ROOT / f"lmx-longctx-{STAMP}.run.txt"

PROMPT = "Explain the theory of general relativity in simple terms."
RUNS = 3
MAX_SEQ = 32768
BUDGET = 2048
BETA = 128

MODELS = [
    {
        "tag": "qwen3.5:9b",
        "hfid": "Qwen/Qwen3.5-9B",
        "sidecar": "/home/kaden/.hipfire/models/qwen3.5-9b-mq4.triattn.hfq",
    },
    {
        "tag": "qwen3.5:27b",
        "hfid": "Qwen/Qwen3.5-27B",
        "sidecar": "/home/kaden/.hipfire/models/qwen3.5-27b-mq4.triattn.hfq",
    },
    {
        "tag": "qwen3.5:35b-a3b",
        "hfid": "Qwen/Qwen3.5-35B-A3B",
        "sidecar": "/home/kaden/.hipfire/models/qwen3.5-35b-a3b-mq4.triattn.hfq",
    },
    {
        "tag": "qwen3.6:27b",
        "hfid": "Qwen/Qwen3.6-27B",
        "sidecar": None,  # no sidecar shipped for this model yet
    },
    {
        "tag": "qwen3.6:35b-a3b",
        "hfid": "Qwen/Qwen3.6-35B-A3B",
        "sidecar": "/home/kaden/.hipfire/models/qwen3.6-35b-a3b-mq4.hfq.hermes.triattn.bin",
    },
]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def patch_per_model(tag, sidecar):
    cfg = json.loads(PER_MODEL_CFG.read_text()) if PER_MODEL_CFG.exists() else {}
    cfg[tag] = {
        "cask_sidecar": sidecar or "",
        "cask": False,  # plain TriAttention drop-eviction (not CASK m-folding)
        "cask_budget": BUDGET,
        "cask_beta": BETA,
        "max_seq": MAX_SEQ,
        "dflash_mode": "off",
        "thinking": "off",
    }
    PER_MODEL_CFG.write_text(json.dumps(cfg, indent=2))


def parse_bench_output(out):
    def m(p):
        x = re.search(p, out)
        return float(x.group(1)) if x else None

    def mi(p):
        x = re.search(p, out)
        return int(x.group(1)) if x else None

    return {
        "decode_tokS": m(r"Decode\s+tok/s\s+([\d.]+)"),
        "wall_tokS": m(r"Wall\s+tok/s\s+([\d.]+)"),
        "ttft_ms": m(r"TTFT\s+ms\s+([\d.]+)"),
        "prefill_user_tokS": m(r"Prefill\s+tok/s\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\(user prompt"),
        "pp128_tokS": m(r"pp128\s*=?\s*([\d.]+)"),
        "pp512_tokS": m(r"pp512\s*=?\s*([\d.]+)"),
        "pp1024_tokS": m(r"pp1024\s*=?\s*([\d.]+)"),
        "pp2048_tokS": m(r"pp2048\s*=?\s*([\d.]+)"),
        "vram_loaded_mb": mi(r"vram:\s+(\d+)\s+MB loaded"),
        "vram_free_mb": mi(r"vram:\s+\d+\s+MB loaded\s+\((\d+)/\d+"),
        "max_seq": mi(r"max_seq:\s+(\d+)"),
        "physical_cap": mi(r"physical_cap=(\d+)"),
        "raw_output_tail": out[-2000:],
    }


def run_bench(tag):
    env = os.environ.copy()
    env["PATH"] = f"{Path.home() / '.hipfire' / 'bin'}:" + env.get("PATH", "")
    cmd = ["hipfire", "bench", tag, "--runs", str(RUNS), PROMPT]
    log(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600, check=False)
    return proc.stdout + proc.stderr, proc.returncode


def main():
    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("")
    log(f"=== LongCtx + Sidecar AR Bench Sweep ===")
    log(f"max_seq={MAX_SEQ} budget={BUDGET} beta={BETA} runs={RUNS}")
    if PER_MODEL_CFG.exists():
        PER_MODEL_BAK.write_text(PER_MODEL_CFG.read_text())
        log(f"backed up per_model_config to {PER_MODEL_BAK}")

    results = []
    try:
        for i, m in enumerate(MODELS, 1):
            log(f"\n=== [{i}/{len(MODELS)}] {m['tag']} sidecar={'on' if m['sidecar'] else 'off (none avail)'} ===")
            patch_per_model(m["tag"], m["sidecar"])
            t0 = time.time()
            out, rc = run_bench(m["tag"])
            dt = time.time() - t0
            p = parse_bench_output(out)
            log(f"  rc={rc} dt={dt:.1f}s")
            log(f"  decode={p.get('decode_tokS')} ttft={p.get('ttft_ms')}ms wall={p.get('wall_tokS')}")
            log(f"  max_seq={p.get('max_seq')} physical_cap={p.get('physical_cap')} vram_loaded={p.get('vram_loaded_mb')}MB")
            results.append({**m, "rc": rc, "dt_s": dt, "parsed": p})
            RESULTS_PATH.write_text(json.dumps(results, indent=2))
        log(f"\nWrote results → {RESULTS_PATH}")
    finally:
        if PER_MODEL_BAK.exists():
            PER_MODEL_CFG.write_text(PER_MODEL_BAK.read_text())
            log("per_model_config restored")


if __name__ == "__main__":
    main()
