#!/usr/bin/env python3
"""max_seq=131072 (full Qwen3 native ctx) AR bench sweep.

Models with sidecar → max_seq=131072 (sidecar caps physical_cap so KV
fits on 24GB). qwen3.6:27b has no sidecar locally — held at max_seq=
32768 for this run (full ring at 131072 would need ~25GB KV alone).
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path

PER_MODEL_CFG = Path.home() / ".hipfire" / "per_model_config.json"
PER_MODEL_BAK = Path("/tmp/per_model_config.before_131k.bak.json")
STAMP = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
BENCH_ROOT = Path("/home/kaden/ClaudeCode/autorocm/hipfire/.worktrees/dflash/benchmarks/results")
RESULTS_PATH = BENCH_ROOT / f"lmx-longctx-131k-{STAMP}.json"
LOG_PATH = BENCH_ROOT / f"lmx-longctx-131k-{STAMP}.run.txt"

PROMPT = "Explain the theory of general relativity in simple terms."
RUNS = 3
BUDGET = 2048
BETA = 128

MODELS = [
    {
        "tag": "qwen3.5:9b",
        "hfid": "Qwen/Qwen3.5-9B",
        "sidecar": "/home/kaden/.hipfire/models/qwen3.5-9b-mq4.triattn.hfq",
        "max_seq": 131072,
    },
    {
        "tag": "qwen3.5:27b",
        "hfid": "Qwen/Qwen3.5-27B",
        "sidecar": "/home/kaden/.hipfire/models/qwen3.5-27b-mq4.triattn.hfq",
        "max_seq": 131072,
    },
    {
        "tag": "qwen3.5:35b-a3b",
        "hfid": "Qwen/Qwen3.5-35B-A3B",
        "sidecar": "/home/kaden/.hipfire/models/qwen3.5-35b-a3b-mq4.triattn.hfq",
        "max_seq": 131072,
    },
    {
        "tag": "qwen3.6:27b",
        "hfid": "Qwen/Qwen3.6-27B",
        "sidecar": None,  # no sidecar shipped — full-ring physical_cap caps at max_seq
        "max_seq": 32768,  # 131072 would need ~25GB KV; doesn't fit 24GB without sidecar
    },
    {
        "tag": "qwen3.6:35b-a3b",
        "hfid": "Qwen/Qwen3.6-35B-A3B",
        "sidecar": "/home/kaden/.hipfire/models/qwen3.6-35b-a3b-mq4.hfq.hermes.triattn.bin",
        "max_seq": 131072,
    },
]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def patch_per_model(tag, sidecar, max_seq):
    cfg = json.loads(PER_MODEL_CFG.read_text()) if PER_MODEL_CFG.exists() else {}
    cfg[tag] = {
        "cask_sidecar": sidecar or "",
        "cask": False,
        "cask_budget": BUDGET,
        "cask_beta": BETA,
        "max_seq": max_seq,
        "dflash_mode": "off",
        "thinking": "off",
    }
    PER_MODEL_CFG.write_text(json.dumps(cfg, indent=2))


def parse(out):
    def f(p):
        x = re.search(p, out)
        return float(x.group(1)) if x else None
    def i(p):
        x = re.search(p, out)
        return int(x.group(1)) if x else None
    return {
        "decode_tokS": f(r"Decode\s+tok/s\s+([\d.]+)"),
        "wall_tokS": f(r"Wall\s+tok/s\s+([\d.]+)"),
        "ttft_ms": f(r"TTFT\s+ms\s+([\d.]+)"),
        "prefill_user_tokS": f(r"Prefill\s+tok/s\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\(user prompt"),
        "pp128_tokS": f(r"pp128\s*=?\s*([\d.]+)"),
        "pp512_tokS": f(r"pp512\s*=?\s*([\d.]+)"),
        "pp1024_tokS": f(r"pp1024\s*=?\s*([\d.]+)"),
        "pp2048_tokS": f(r"pp2048\s*=?\s*([\d.]+)"),
        "vram_loaded_mb": i(r"vram:\s+(\d+)\s+MB loaded"),
        "vram_free_mb": i(r"vram:\s+\d+\s+MB loaded\s+\((\d+)/\d+"),
        "max_seq": i(r"max_seq:\s+(\d+)"),
        "physical_cap": i(r"physical_cap=(\d+)"),
        "raw_output_tail": out[-2000:],
    }


def run_bench(tag):
    env = os.environ.copy()
    env["PATH"] = f"{Path.home() / '.hipfire' / 'bin'}:" + env.get("PATH", "")
    cmd = ["hipfire", "bench", tag, "--runs", str(RUNS), PROMPT]
    log(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    return proc.stdout + proc.stderr, proc.returncode


def main():
    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("")
    log(f"=== max_seq=131072 (Qwen3 native) AR bench sweep ===")
    log(f"budget={BUDGET} beta={BETA} runs={RUNS} prompt=\"{PROMPT}\"")
    if PER_MODEL_CFG.exists():
        PER_MODEL_BAK.write_text(PER_MODEL_CFG.read_text())
        log(f"backed up per_model_config to {PER_MODEL_BAK}")

    results = []
    try:
        for i, m in enumerate(MODELS, 1):
            label = f"sidecar={'on' if m['sidecar'] else 'off'} max_seq={m['max_seq']}"
            log(f"\n=== [{i}/{len(MODELS)}] {m['tag']} {label} ===")
            patch_per_model(m["tag"], m["sidecar"], m["max_seq"])
            t0 = time.time()
            out, rc = run_bench(m["tag"])
            dt = time.time() - t0
            p = parse(out)
            log(f"  rc={rc} dt={dt:.1f}s")
            log(f"  decode={p.get('decode_tokS')} ttft={p.get('ttft_ms')}ms wall={p.get('wall_tokS')}")
            log(f"  max_seq={p.get('max_seq')} physical_cap={p.get('physical_cap')} vram_loaded={p.get('vram_loaded_mb')}MB")
            if rc != 0:
                log(f"  WARN rc={rc} (likely OOM); raw tail: {p['raw_output_tail'][-500:]}")
            results.append({**m, "rc": rc, "dt_s": dt, "parsed": p})
            RESULTS_PATH.write_text(json.dumps(results, indent=2))
        log(f"\nWrote results → {RESULTS_PATH}")
    finally:
        if PER_MODEL_BAK.exists():
            PER_MODEL_CFG.write_text(PER_MODEL_BAK.read_text())
            log("per_model_config restored")


if __name__ == "__main__":
    main()
