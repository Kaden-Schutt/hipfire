#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# bench_humaneval_dflash.py — run dflash_spec_demo across a HumanEval sample.
# HumanEval prompts are code-continuation (signature + docstring) so τ stays
# in code-token regime. Matches Lucebox methodology.
#
# Usage:
#   python3 scripts/bench_humaneval_dflash.py            # default: 33 prompts, DFlash mode
#   python3 scripts/bench_humaneval_dflash.py --ar       # AR baseline mode (no draft used)
#   python3 scripts/bench_humaneval_dflash.py --n 25     # sample size
#   python3 scripts/bench_humaneval_dflash.py --jsonl PATH --max 128

import argparse, json, os, re, subprocess, sys, time

def pct(sorted_arr, p):
    if not sorted_arr:
        return float("nan")
    idx = int(round((len(sorted_arr) - 1) * p / 100))
    return sorted_arr[idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/tmp/humaneval/HumanEval.jsonl")
    ap.add_argument("--target", default=os.path.expanduser("~/.hipfire/models/qwen3.5-27b.mq4"))
    ap.add_argument("--draft", default=os.path.expanduser("~/.hipfire/models/qwen35-27b-dflash-mq4.hfq"))
    ap.add_argument("--demo", default="./target/release/examples/dflash_spec_demo")
    ap.add_argument("--max", type=int, default=128, help="max tokens per prompt")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--n", type=int, default=33, help="sample size (0..164)")
    ap.add_argument("--kv-mode", default="q8", help="q8 | fwht4 | fwht3 | fwht2")
    ap.add_argument("--ar", action="store_true", help="AR baseline mode (passes --ar-baseline; no spec decode)")
    ap.add_argument("--ddtree-batched", action="store_true", help="enable --ddtree-batched")
    ap.add_argument("--ddtree-budget", type=int, default=None)
    ap.add_argument("--ddtree-topk", type=int, default=None)
    ap.add_argument("--label", default="", help="label printed in the summary (e.g. 'linear-q8')")
    args = ap.parse_args()
    if args.kv_mode.lower() not in ("q8", "fwht4", "fwht3", "fwht2"):
        print("--kv-mode must be q8, fwht4, fwht3, or fwht2 for DFlash perf runs", file=sys.stderr)
        sys.exit(2)
    if args.ar and (args.ddtree_batched or args.ddtree_budget is not None or args.ddtree_topk is not None):
        print("--ar is incompatible with --ddtree-* flags", file=sys.stderr); sys.exit(2)

    with open(args.jsonl) as f:
        tasks = [json.loads(l) for l in f]
    if not tasks:
        print("empty jsonl", file=sys.stderr); sys.exit(1)

    stride = max(1, len(tasks) // args.n)
    sampled = tasks[::stride][: args.n]
    print(f"# sampling {len(sampled)} of {len(tasks)} HumanEval prompts (stride={stride})")
    print(f"# target: {args.target}")
    print(f"# draft:  {args.draft}")
    extra_args = ["--kv-mode", args.kv_mode]
    if args.ar:
        extra_args.append("--ar-baseline")
    if args.ddtree_batched:
        extra_args.append("--ddtree-batched")
    if args.ddtree_budget is not None:
        extra_args += ["--ddtree-budget", str(args.ddtree_budget)]
    if args.ddtree_topk is not None:
        extra_args += ["--ddtree-topk", str(args.ddtree_topk)]
    label = args.label or ("ar" if args.ar else "ddtree" if args.ddtree_batched else "linear")
    print(f"# label={label}  max={args.max} ctx={args.ctx} --no-chatml kv={args.kv_mode}  "
          f"extra={' '.join(extra_args)}")
    print()
    print(f"{'task_id':<16} {'tok/s':>8} {'tau':>7} {'emitted':>8} {'cyc':>5} {'acc':>5} {'run_s':>6}")
    print("-" * 64)

    results = []
    for task in sampled:
        tid = task["task_id"]
        prompt = task["prompt"]
        t0 = time.time()
        try:
            proc = subprocess.run(
                [args.demo, "--target", args.target, "--draft", args.draft,
                 "--prompt", prompt, "--max", str(args.max), "--ctx", str(args.ctx),
                 "--no-chatml", *extra_args],
                capture_output=True, text=True, timeout=240)
            out = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired:
            print(f"{tid:<16} TIMEOUT")
            continue
        dt = time.time() - t0

        m_toks = re.search(r"emitted: (\d+) tokens in ([\d.]+)s\s+\(([\d.]+) tok/s\)", out)
        m_tau  = re.search(r"\xcf\x84=([\d.]+)|τ=([\d.]+)", out)
        m_cyc  = re.search(r"cycles: (\d+)", out)
        m_acc  = re.search(r"accepted: (\d+)", out)

        # In AR mode there is no τ / cycles / accepted — only emitted+tok_s are required.
        if not m_toks or (not args.ar and not m_tau):
            print(f"{tid:<16} PARSE_FAIL  run_s={dt:.1f}")
            results.append((tid, None, None, None, None, None, dt))
            continue

        emitted = int(m_toks.group(1))
        tok_s = float(m_toks.group(3))
        tau = float(m_tau.group(1) or m_tau.group(2)) if m_tau else float("nan")
        cyc = int(m_cyc.group(1)) if m_cyc else -1
        acc = int(m_acc.group(1)) if m_acc else -1

        print(f"{tid:<16} {tok_s:8.2f} {tau:7.3f} {emitted:8d} {cyc:5d} {acc:5d} {dt:6.1f}")
        results.append((tid, tok_s, tau, emitted, cyc, acc, dt))

    # summary
    valid = [r for r in results if r[1] is not None]
    if not valid:
        print("\nno valid results"); sys.exit(2)

    tok_s = sorted([r[1] for r in valid])
    print()
    print(f"=== SUMMARY (n={len(valid)}/{len(sampled)}) ===")
    print(f"tok/s  mean={sum(tok_s)/len(tok_s):7.2f}  median={pct(tok_s,50):7.2f}  "
          f"p10={pct(tok_s,10):7.2f}  p90={pct(tok_s,90):7.2f}  "
          f"min={tok_s[0]:7.2f}  max={tok_s[-1]:7.2f}")
    if not args.ar:
        taus = sorted([r[2] for r in valid])
        print(f"tau    mean={sum(taus)/len(taus):7.3f}  median={pct(taus,50):7.3f}  "
              f"p10={pct(taus,10):7.3f}  p90={pct(taus,90):7.3f}  "
              f"min={taus[0]:7.3f}  max={taus[-1]:7.3f}")

if __name__ == "__main__":
    main()
