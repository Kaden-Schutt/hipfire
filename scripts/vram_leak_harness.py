#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Load/unload VRAM-delta oracle.

## Why this exists

A leaked GPU allocation is *silent*. The model still loads, still generates
correct text, and every functional smoke test passes — you only find out when
the fifth reload OOMs. hipfire has already shipped this bug once: the Glimmer
teardown freed weights but not state, leaking roughly 1.3 GB over five cycles
(PR #566), and it was invisible to every gate in the tree.

Anything that touches `unload_model`, an `ArchModel::free_gpu` impl, or the
coupled multi-GPU fields `skeleton_pp` sets together needs this oracle. That
comment in `hipfire-loader/src/lib.rs` says it outright: "a dropped
`pp_scratch_set` is a silent VRAM leak".

## What it measures

Drives the daemon over its JSONL stdio protocol:

    -> {"type":"load","model":"<path>","params":{"max_seq":N}}
    <- {"type":"loaded",...}
    -> {"type":"unload"}
    <- {"type":"unloaded"}

sampling free VRAM before the first load and after every unload. A correct
teardown returns to the baseline; a leak shows up as monotonic drift across
cycles.

## Reading the result

Judge the SLOPE across cycles, not a single delta. Allocator behaviour,
fragmentation and driver caching make one cycle noisy; a real leak grows every
time. The harness reports per-cycle deltas and a linear fit, and fails on
sustained growth rather than on any single sample.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

AMD_SMI = "/opt/rocm/core-7.14/bin/amd-smi"


def vram_used_total_mib(devices: list[int]) -> int | None:
    """Summed used VRAM across devices.

    Pipeline parallelism spreads one model over several GPUs, so a per-device
    sample can look flat while another card leaks. `skeleton_pp` sets four
    multi-GPU fields as a unit precisely because dropping one leaks silently,
    and a single-device probe would not see it.
    """
    total = 0
    for d in devices:
        v = vram_used_mib(d)
        if v is None:
            return None
        total += v
    return total


def vram_used_mib(device: int = 0) -> int | None:
    """Used VRAM in MiB, or None when no probe is available.

    Tries amd-smi first (hiptrx has no rocm-smi), then rocm-smi. Returning None
    rather than raising keeps the harness usable on a box without either — the
    caller degrades to reporting nothing instead of dying.
    """
    for cmd in ([AMD_SMI, "metric", "-g", str(device), "--mem"], ["rocm-smi", "--showmemuse"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        # amd-smi prints "USED_VRAM: 1234 MB"; rocm-smi a percentage plus bytes.
        m = re.search(r"USED_VRAM[^0-9]*([0-9]+)", out) or re.search(
            r"VRAM Total Used Memory \(B\)[^0-9]*([0-9]+)", out
        )
        if m:
            value = int(m.group(1))
            # Heuristic: the byte-valued field is orders of magnitude larger.
            return value // (1024 * 1024) if value > 10**7 else value
    return None


def cycle(
    daemon: str, model: str, max_seq: int, cycles: int, devices: list[int], pp: int
) -> int:
    env = dict(os.environ)
    env["HOME"] = env.get("VRAM_HARNESS_HOME", "/home/kaden/.vram-harness")
    os.makedirs(os.path.join(env["HOME"], ".hipfire"), exist_ok=True)
    env["HIPFIRE_NO_REGISTRY_FETCH"] = "1"

    proc = subprocess.Popen(
        [daemon], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env,
    )

    def send(obj: dict) -> None:
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()

    def await_kind(kind: str, timeout: float = 900.0) -> dict | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                return None
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == kind:
                return msg
            if msg.get("type") in ("error", "fatal"):
                print(f"  daemon error: {str(msg)[:160]}")
                return None
        return None

    baseline = vram_used_total_mib(devices)
    print(f"  baseline used VRAM: {baseline if baseline is not None else 'unavailable'} MiB")
    deltas: list[int] = []
    rc = 0

    for i in range(1, cycles + 1):
        params: dict = {"max_seq": max_seq}
        if pp > 1:
            # pp is reachable only through the daemon's load params -- there is
            # no `serve --pp` flag -- and is Qwen3.5 dense/MoE only.
            params["pp"] = pp
        send({"type": "load", "model": model, "params": params})
        if await_kind("loaded") is None:
            print(f"  cycle {i}: load FAILED")
            rc = 2
            break
        send({"type": "unload"})
        if await_kind("unloaded") is None:
            print(f"  cycle {i}: unload FAILED")
            rc = 2
            break
        time.sleep(2.0)  # let the allocator settle before sampling
        after = vram_used_total_mib(devices)
        if baseline is None or after is None:
            print(f"  cycle {i}: ok (no VRAM probe)")
            continue
        d = after - baseline
        deltas.append(d)
        print(f"  cycle {i}: used {after} MiB, delta vs baseline {d:+d} MiB")

    send({"type": "shutdown"})
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()

    if len(deltas) >= 3:
        # Least-squares slope over cycle index. A clean teardown is flat; a leak
        # climbs by roughly the leaked amount every cycle.
        n = len(deltas)
        xs = list(range(1, n + 1))
        mx, my = sum(xs) / n, sum(deltas) / n
        denom = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, deltas)) / denom if denom else 0.0
        # Magnitude alone is the wrong test. Measured on gfx1201, lfm2moe
        # grows +6 MiB per cycle EXACTLY -- five samples, zero scatter. That is
        # far under any sane magnitude threshold and yet unmistakably a leak,
        # because real allocator jitter is noisy and a leak is not. So score
        # linearity too: R^2 near 1.0 on a positive slope means something is
        # retained every cycle, however small.
        ss_tot = sum((y - my) ** 2 for y in deltas)
        ss_res = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(xs, deltas))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 1.0
        print(f"  slope: {slope:+.1f} MiB/cycle over {n} cycles (R^2 {r2:.3f})")
        # 32 MiB/cycle is well under the ~260 MiB/cycle the Glimmer bug leaked.
        if slope > 32.0:
            print(f"  LEAK: VRAM grows {slope:.1f} MiB per cycle")
            rc = 1
        elif slope > 1.0 and r2 > 0.95:
            print(
                f"  SUSPECT: {slope:.1f} MiB/cycle with R^2 {r2:.3f} -- too linear for"
                " allocator noise. Small, but it never comes back."
            )
            rc = 1
        else:
            print("  no sustained growth")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--daemon", default="target/release/daemon")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--devices", default="0", help="comma-separated, e.g. 0,1,2,3")
    ap.add_argument("--pp", type=int, default=1, help="pipeline-parallel degree")
    a = ap.parse_args()
    if not os.path.exists(a.daemon):
        print(f"daemon not found: {a.daemon}")
        return 2
    devices = [int(x) for x in a.devices.split(",") if x.strip() != ""]
    print(f"  model: {a.model}  pp={a.pp}  devices={devices}")
    return cycle(a.daemon, a.model, a.max_seq, a.cycles, devices, a.pp)


if __name__ == "__main__":
    sys.exit(main())
