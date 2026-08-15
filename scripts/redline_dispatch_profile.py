#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Collect steady-state, exactly-once retained-PM4 dispatch timings."""

import argparse
import datetime
import hashlib
import json
import math
import os
import select
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1


class Daemon:
    def __init__(self, binary, log_path, timeout_s, kv_mode):
        self.timeout_s = timeout_s
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("w")
        env = dict(os.environ)
        env.update(
            HIPFIRE_REPLAY_BACKEND="shadow",
            HIPFIRE_REPLAY_MANUAL_CAPTURE="1",
            HIPFIRE_KV_MODE=kv_mode,
            HIPFIRE_CASK_OFF="1",
            HIPFIRE_AR_GRAPH="0",
            HIPFIRE_GRAPH="0",
            HIPFIRE_REDLINE_DISPATCH_PROFILE="0",
        )
        self.proc = subprocess.Popen(
            [str(binary)],
            cwd=REPO,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

    def request(self, message):
        if self.proc.poll() is not None:
            raise RuntimeError(f"daemon exited with code {self.proc.returncode}")
        self.proc.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        ready, _, _ = select.select([self.proc.stdout], [], [], self.timeout_s)
        if not ready:
            raise TimeoutError(f"daemon timed out handling {message['type']}")
        response = json.loads(self.proc.stdout.readline())
        if response.get("type") == "error":
            raise RuntimeError(response.get("message", "daemon error"))
        return response

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.log.close()


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def statistics_for(values):
    mean = statistics.fmean(values)
    return {
        "min_ns": min(values),
        "median_ns": statistics.median(values),
        "p90_ns": percentile(values, 0.90),
        "max_ns": max(values),
        "cv": statistics.pstdev(values) / mean if len(values) > 1 and mean else 0.0,
    }


def validate_profile(profile):
    if profile.get("type") != "redline_dispatch_profile":
        raise ValueError("unexpected RPC response type")
    if profile.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported RPC schema version")
    if not profile.get("steady_state") or not profile.get("exactly_once_per_sample"):
        raise ValueError("RPC did not guarantee steady-state exactly-once samples")

    dispatches = profile.get("dispatches")
    samples = profile.get("samples")
    if not dispatches or not samples:
        raise ValueError("profile has no dispatches or samples")
    if profile.get("sample_replays") != len(samples):
        raise ValueError("sample count mismatch")
    if profile["route"].get("launches") != len(dispatches):
        raise ValueError("route/dispatch count mismatch")
    if profile["route"].get("timestamp_slots") != len(dispatches) + 1:
        raise ValueError("timestamp slot count mismatch")
    if [row.get("index") for row in dispatches] != list(range(len(dispatches))):
        raise ValueError("dispatch indices are not contiguous")

    for index, sample in enumerate(samples):
        spans = sample.get("spans_ns")
        if sample.get("sample") != index:
            raise ValueError("sample indices are not contiguous")
        if not isinstance(spans, list) or len(spans) != len(dispatches):
            raise ValueError(f"sample {index} span length mismatch")
        if any(not isinstance(value, int) or value < 0 for value in spans):
            raise ValueError(f"sample {index} contains an invalid span")
        total = sample.get("total_gpu_ns")
        if not isinstance(total, int) or abs(total - sum(spans)) > len(spans):
            raise ValueError(f"sample {index} total does not match its spans")


def validate_capture(capture, profile):
    expected = {
        "launches": profile["route"]["launches"],
        "unique_kernels": profile["route"]["unique_kernels"],
        "sequence_hash": profile["route"]["sequence_hash"],
    }
    if {key: capture.get(key) for key in expected} != expected:
        raise ValueError("capture/profile route mismatch")


def analyze(profile):
    validate_profile(profile)
    samples = profile["samples"]
    columns = zip(*(sample["spans_ns"] for sample in samples))
    dispatches = []
    for metadata, values in zip(profile["dispatches"], columns):
        row = dict(metadata)
        row["samples_ns"] = list(values)
        row.update(statistics_for(row["samples_ns"]))
        dispatches.append(row)

    dispatch_total = sum(row["median_ns"] for row in dispatches)
    ranked = sorted(dispatches, key=lambda row: row["median_ns"], reverse=True)
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
        row["share_pct"] = 100.0 * row["median_ns"] / dispatch_total

    grouped = defaultdict(list)
    for row in dispatches:
        grouped[(row["kernel"], tuple(row["grid"]), tuple(row["block"]))].append(row)
    kernels = []
    for (kernel, grid, block), rows in grouped.items():
        total = sum(row["median_ns"] for row in rows)
        kernels.append(
            {
                "kernel": kernel,
                "grid": list(grid),
                "block": list(block),
                "dispatch_indices": [row["index"] for row in rows],
                "total_median_ns": total,
                "share_pct": 100.0 * total / dispatch_total,
            }
        )
    kernels.sort(key=lambda row: row["total_median_ns"], reverse=True)

    medians = [row["median_ns"] for row in dispatches]
    tail_count = max(1, math.ceil(len(medians) * 0.05))
    return {
        "dispatches": dispatches,
        "kernels": kernels,
        "summary": {
            "gpu_total": statistics_for(
                [sample["total_gpu_ns"] for sample in samples]
            ),
            "host_total": statistics_for([sample["host_ns"] for sample in samples]),
            "dispatch_p50_ns": percentile(medians, 0.50),
            "dispatch_p90_ns": percentile(medians, 0.90),
            "dispatch_p99_ns": percentile(medians, 0.99),
            "slowest_5pct_share_pct": (
                100.0 * sum(sorted(medians)[-tail_count:]) / dispatch_total
            ),
        },
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_summary(report, top):
    route = report["route"]
    summary = report["summary"]
    gpu = summary["gpu_total"]
    print(
        f"route={route['sequence_hash']} launches={route['launches']} "
        f"samples={report['request']['sample_replays']} "
        f"gpu_median={gpu['median_ns'] / 1_000:.3f}us "
        f"slowest5%={summary['slowest_5pct_share_pct']:.1f}%"
    )
    correctness = report["correctness"]
    if correctness.get("performed"):
        print(f"instrumented-shadow bit_exact={correctness.get('bit_exact')}")
    print("rank  idx  median_us  p90_us  share%  boundary  kernel")
    for row in sorted(report["dispatches"], key=lambda item: item["rank"])[:top]:
        boundary = row["boundary"]
        flags = "".join(
            (
                "E" if boundary.get("entry_acquire") else "-",
                "W" if boundary["wait_compute_idle"] else "-",
                "A" if boundary["acquire_inter_node"] else "-",
                "V" if boundary["acquire_vmem"] else "-",
            )
        )
        print(
            f"{row['rank']:4d}  {row['index']:3d}  "
            f"{row['median_ns'] / 1_000:9.3f}  "
            f"{row['p90_ns'] / 1_000:7.3f}  "
            f"{row['share_pct']:6.2f}  {flags:8s}  {row['kernel']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="steady-state exactly-once retained-PM4 dispatch profiler"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--daemon", default=str(REPO / "target/release/examples/daemon")
    )
    parser.add_argument(
        "--out", default=str(REPO / ".redline-work/dispatch-profile.json")
    )
    parser.add_argument(
        "--log", default=str(REPO / ".redline-work/dispatch-profile.log")
    )
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--warmup-replays", type=int, default=10)
    parser.add_argument("--sample-replays", type=int, default=20)
    parser.add_argument("--max-seq", type=int, default=2048)
    parser.add_argument("--kv-mode", default="q8")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()
    if args.context <= 0 or args.warmup_replays < 0 or args.sample_replays <= 0:
        parser.error("invalid context/warmup/sample count")

    model = Path(args.model).expanduser().resolve()
    daemon_path = Path(args.daemon).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    log_path = Path(args.log).expanduser().resolve()
    if not model.is_file() or not daemon_path.is_file():
        parser.error("model and daemon must exist")

    daemon = Daemon(daemon_path, log_path, args.timeout, args.kv_mode)
    try:
        daemon.request(
            {
                "type": "load",
                "model": str(model),
                "params": {
                    "max_seq": args.max_seq,
                    "kv_mode": args.kv_mode,
                    "dflash_mode": "off",
                },
            }
        )
        capture = daemon.request(
            {
                "type": "bench_decode",
                "context_tokens": args.context,
                "iterations": 1,
                "redline_capture": True,
                "redline_detail": True,
            }
        )["redline_capture"]
        profile = daemon.request(
            {
                "type": "redline_dispatch_profile",
                "context_tokens": args.context,
                "warmup_replays": args.warmup_replays,
                "sample_replays": args.sample_replays,
                "validate_correctness": not args.skip_correctness,
            }
        )
    finally:
        daemon.close()

    validate_capture(capture, profile)
    analysis = analyze(profile)
    report = {
        "schema_version": SCHEMA_VERSION,
        "type": "hipfire_redline_dispatch_profile_report",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": {
            "path": str(model),
            "bytes": model.stat().st_size,
            "sha256": sha256_file(model),
        },
        "request": {
            "context_tokens": args.context,
            "warmup_replays": args.warmup_replays,
            "sample_replays": args.sample_replays,
            "max_seq": args.max_seq,
            "kv_mode": args.kv_mode,
        },
        "route": profile["route"],
        "timestamp_semantics": profile["timestamp_semantics"],
        "correctness": profile["correctness"],
        "samples": profile["samples"],
        **analysis,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print_summary(report, args.top)
    print(f"report={output}")
    print(f"daemon_log={log_path}")
    if report["correctness"].get("bit_exact") is False:
        print("FAIL: instrumented PM4 shadow is not bit-exact", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
