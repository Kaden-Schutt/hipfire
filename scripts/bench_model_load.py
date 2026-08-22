#!/usr/bin/env python3
"""Model-load wall-clock benchmark: hipfire daemon vs llama.cpp (ROCmFPX).

Measures one thing per engine and reports medians of identical windows:

  hipfire : daemon process spawn → {"type":"loaded"} protocol response.
            Includes process init + GPU context + full weight upload.
  rocmfpx : process spawn → engine-reported `load time` (print_timings),
            cross-checked with wall-clock to end-of-load log line.

GPU access MUST be serialized externally through /home/ghazni/gpu-coord/
(e.g. gpu-ctl run ... -- python3 scripts/bench_model_load.py ...).
This script never touches the lock itself so the caller owns acquisition.

Usage:
  python3 scripts/bench_model_load.py hipfire  --model M.hfq --runs 5 [--daemon PATH]
  python3 scripts/bench_model_load.py rocmfpx --model M.gguf --llama-bin DIR --runs 5
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import threading
import time


def median(xs):
    return statistics.median(xs)


def spread_pct(xs):
    if len(xs) < 2:
        return 0.0
    m = median(xs)
    return (max(xs) - min(xs)) / m * 100.0


def bench_hipfire(model: str, daemon_bin: str, runs: int) -> list[float]:
    results = []
    for i in range(runs):
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            [daemon_bin],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            proc.stdin.write(
                json.dumps({"type": "load", "model": model, "params": {"max_seq": 4096}})
                + "\n"
            )
            proc.stdin.flush()
            loaded = False
            while not loaded:
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError(f"run {i}: daemon closed stdout before 'loaded'")
                msg = line.strip()
                if not msg:
                    continue
                try:
                    obj = json.loads(msg)
                except json.JSONDecodeError:
                    continue  # non-protocol log noise
                if obj.get("type") == "loaded":
                    loaded = True
                    dt = time.perf_counter() - t0
                    results.append(dt)
                    print(f"  run {i+1}/{runs}: {dt*1000:.1f} ms  arch={obj.get('arch')}")
                elif obj.get("type") == "error":
                    raise RuntimeError(f"run {i}: daemon error: {msg[:400]}")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        time.sleep(0.3)  # let the pid-file flock release before next spawn
    return results




READY_MARKER_RE = re.compile(r"main: llama threadpool init")


def bench_rocmfpx(model: str, llama_bin_dir: str, runs: int) -> list[float]:
    """Wall-clock process spawn → end-of-model-load marker.

    The fork suppresses per-tensor load logs and its engine-reported
    `load time` is clobbered by llama_perf_context_reset, so the reliable
    ready marker is the first log line emitted after common_init_result
    returns (`main: llama threadpool init`). The process is killed as soon
    as the marker fires, so the window excludes prompt eval / generation.
    """
    results = []
    bin_path = f"{llama_bin_dir}/llama-completion"
    for i in range(runs):
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            [
                bin_path,
                "-m",
                model,
                "-ngl",
                "99",
                "-p",
                "hi",
                "-n",
                "1",
                "-t",
                "16",
                "--no-warmup",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env={**dict(__import__("os").environ), "LD_LIBRARY_PATH": llama_bin_dir},
        )
        ready = None
        assert proc.stderr is not None

        def pump():
            nonlocal ready
            for line in proc.stderr:
                if READY_MARKER_RE.search(line):
                    ready = time.perf_counter() - t0
                    return  # stop consuming; caller terminates the process

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        # pump() returns on the marker or at stderr EOF (process exit).
        reader.join(timeout=600)
        proc.kill()
        proc.wait(timeout=10)
        reader.join(timeout=5)
        if ready is None:
            raise RuntimeError(f"run {i}: ready marker never seen")
        results.append(ready)
        print(f"  run {i+1}/{runs}: {ready*1000:.1f} ms")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["hipfire", "rocmfpx"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--daemon", default="target/release/daemon")
    ap.add_argument("--llama-bin", default="/home/ghazni/projects/ROCmFPX/build-opt/bin")
    args = ap.parse_args()

    print(f"== {args.engine} load benchmark: {args.runs} fresh-process runs ==")
    if args.engine == "hipfire":
        res = bench_hipfire(args.model, args.daemon, args.runs)
    else:
        res = bench_rocmfpx(args.model, args.llama_bin, args.runs)

    print(f"MEDIAN_LOAD_S {median(res):.3f}")
    print(f"SPREAD_PCT {spread_pct(res):.2f}")
    print(f"SAMPLES_S {[round(x, 3) for x in res]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
