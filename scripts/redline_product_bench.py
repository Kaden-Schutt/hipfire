#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Resident HipGraph-vs-Redline product decode benchmark.

Unlike redline_daemon_harness.py (manual shadow capture), this drives the real
default-off product lifecycle: HIPFIRE_REPLAY_BACKEND=auto records one ordinary
AR forward and routes later forwards through the prepared AQL replay. The HIP
arm leaves the existing AR HipGraph enabled. Models stay resident within each
arm, clocks are never modified, and every row uses the daemon's full Qwen reset
and prefill-prime path.
"""

import argparse
import json
import os
import select
import signal
import statistics
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


class Daemon:
    def __init__(
        self,
        binary: Path,
        backend: str,
        transport: str,
        log_path: Path,
        timeout: float,
        kv_mode: str,
    ):
        self.timeout = timeout
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("w")
        env = dict(os.environ)
        env.update(
            HIPFIRE_REPLAY_BACKEND=backend,
            HIPFIRE_REPLAY_TRANSPORT=transport,
            HIPFIRE_KV_MODE=kv_mode,
            HIPFIRE_CASK_OFF="1",
            HIPFIRE_AR_GRAPH="1",
            HIPFIRE_GRAPH="1",
        )
        env.pop("HIPFIRE_REPLAY_MANUAL_CAPTURE", None)
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
            raise RuntimeError(f"daemon exited with {self.proc.returncode}")
        self.proc.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        ready, _, _ = select.select([self.proc.stdout], [], [], self.timeout)
        if not ready:
            raise TimeoutError(f"daemon timed out on {message['type']}")
        response = json.loads(self.proc.stdout.readline())
        if response.get("type") == "error":
            raise RuntimeError(response.get("message", "daemon error"))
        return response

    def close(self):
        if self.proc.poll() is None:
            try:
                self.request({"type": "unload"})
            except Exception:
                pass
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=5)
            except Exception:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=5)
        self.log.close()


def run_arm(args, backend):
    daemon = Daemon(
        Path(args.daemon).resolve(),
        backend,
        args.transport,
        Path(args.work_dir) / f"product-{backend}.log",
        args.timeout,
        args.kv_mode,
    )
    try:
        loaded = daemon.request(
            {
                "type": "load",
                "model": str(Path(args.model).expanduser().resolve()),
                "params": {
                    "max_seq": args.max_seq,
                    "kv_mode": args.kv_mode,
                    "dflash_mode": "off",
                },
            }
        )
        request = {
            "type": "bench_decode",
            "context_tokens": args.context,
            "iterations": args.iterations,
        }
        warmups = [daemon.request(request) for _ in range(args.warmups)]
        rows = [daemon.request(request) for _ in range(args.runs)]
        values = [row["tok_s"] for row in rows]
        return {
            "loaded": loaded,
            "warmups": warmups,
            "rows": rows,
            "tok_s": {
                "min": min(values),
                "median": statistics.median(values),
                "max": max(values),
            },
        }
    finally:
        daemon.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--daemon", default=str(REPO / "target/release/examples/daemon")
    )
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--transport", choices=("aql", "pm4"), default="aql")
    parser.add_argument(
        "--kv-mode",
        choices=("q8", "fwht2", "fwht3", "fwht4"),
        default="q8",
        help="KV layout used by both the HipGraph and retained-replay arms",
    )
    parser.add_argument("--max-seq", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--work-dir", default=str(REPO / ".redline-work"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = {
        "model": str(Path(args.model).expanduser().resolve()),
        "automatic_clocks": True,
        "context": args.context,
        "iterations": args.iterations,
        "warmups": args.warmups,
        "runs": args.runs,
        "transport": args.transport,
        "kv_mode": args.kv_mode,
        "hip": run_arm(args, "hip"),
        "auto": run_arm(args, "auto"),
    }
    hip = report["hip"]["tok_s"]["median"]
    auto = report["auto"]["tok_s"]["median"]
    report["speedup"] = auto / hip
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"hip={hip:.3f} tok/s auto={auto:.3f} tok/s "
        f"speedup={report['speedup']:.5f} report={output}"
    )


if __name__ == "__main__":
    main()
