#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Resident HIP-vs-Redline product decode benchmark.

Unlike redline_daemon_harness.py (manual shadow capture), this drives the real
default-off product lifecycle. The default ``ar`` workload records one ordinary
AR forward and routes later forwards through retained replay. ``mtp-greedy``
drives the production daemon MTP generator with a byte-pinned prompt and applies
the same cache warmup and stationarity contract to its reported decode rate.
Models stay resident within each arm and clocks are never modified.
"""

import argparse
import hashlib
import json
import os
import select
import signal
import socket
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
DEFAULT_MTP_PROMPT = REPO / "benchmarks/prompts/redline_mtp_greedy_smoke.txt"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def window_statistics(values, window, max_slope_pct, max_spread_pct):
    if window < 3 or len(values) < window:
        return None
    sample = values[-window:]
    median = statistics.median(sample)
    scale = abs(median)
    if scale == 0.0:
        slope_pct = float("inf")
        spread_pct = float("inf")
    else:
        x_mid = (window - 1) / 2.0
        denominator = sum((i - x_mid) ** 2 for i in range(window))
        slope = sum(
            (i - x_mid) * (value - statistics.mean(sample))
            for i, value in enumerate(sample)
        ) / denominator
        slope_pct = 100.0 * slope / scale
        spread_pct = 100.0 * (max(sample) - min(sample)) / scale
    return {
        "window": window,
        "first_row": len(values) - window + 1,
        "last_row": len(values),
        "min": min(sample),
        "median": median,
        "max": max(sample),
        "slope_pct_per_row": slope_pct,
        "spread_pct": spread_pct,
        "stable": abs(slope_pct) <= max_slope_pct
        and spread_pct <= max_spread_pct,
    }


def analyze_stationarity(
    values,
    *,
    window,
    min_runs,
    confirmation_runs,
    max_slope_pct,
    max_spread_pct,
    max_median_drift_pct,
):
    candidate = None
    rejections = []
    latest = None
    for end in range(1, len(values) + 1):
        latest = window_statistics(
            values[:end], window, max_slope_pct, max_spread_pct
        )
        if latest is None or end < min_runs:
            continue

        if candidate is not None:
            drift_pct = (
                100.0
                * abs(latest["median"] - candidate["window"]["median"])
                / abs(candidate["window"]["median"])
            )
            rejection = None
            if not latest["stable"]:
                rejection = "confirmation_window_unstable"
            elif drift_pct > max_median_drift_pct:
                rejection = "confirmation_median_drift"
            if rejection is not None:
                rejections.append(
                    {
                        "candidate_row": candidate["at_row"],
                        "rejected_at_row": end,
                        "reason": rejection,
                        "median_drift_pct": drift_pct,
                    }
                )
                candidate = None

        if candidate is None and latest["stable"]:
            candidate = {"at_row": end, "window": dict(latest)}

        if (
            candidate is not None
            and end - candidate["at_row"] >= confirmation_runs
        ):
            drift_pct = (
                100.0
                * abs(latest["median"] - candidate["window"]["median"])
                / abs(candidate["window"]["median"])
            )
            return {
                "stationary": True,
                "candidate": candidate,
                "confirmed_at_row": end,
                "confirmed_window": dict(latest),
                "median_drift_pct": drift_pct,
                "rejections": rejections,
            }

    return {
        "stationary": False,
        "candidate": candidate,
        "confirmed_at_row": None,
        "confirmed_window": latest,
        "median_drift_pct": None,
        "rejections": rejections,
    }


def stationarity_kwargs(args):
    return {
        "window": args.settle_window,
        "min_runs": args.settle_min_runs,
        "confirmation_runs": args.settle_confirmation_runs,
        "max_slope_pct": args.settle_max_slope_pct,
        "max_spread_pct": args.settle_max_spread_pct,
        "max_median_drift_pct": args.settle_max_median_drift_pct,
    }


def validate_measurement(values, settlement, args):
    window = min(args.settle_window, len(values))
    stats = window_statistics(
        values,
        window,
        args.settle_max_slope_pct,
        args.settle_max_spread_pct,
    )
    reference = settlement["confirmed_window"]["median"]
    measured = statistics.median(values)
    drift_pct = 100.0 * abs(measured - reference) / abs(reference)
    enough_rows = len(values) >= 5
    return {
        "valid": enough_rows
        and stats is not None
        and stats["stable"]
        and drift_pct <= args.settle_max_median_drift_pct,
        "enough_rows": enough_rows,
        "median_drift_from_settlement_pct": drift_pct,
        "window": stats,
    }


class Daemon:
    def __init__(
        self,
        binary: Path,
        backend: str,
        transport: str,
        log_path: Path,
        timeout: float,
        kv_mode: str,
        dpm_warmup_secs: float,
        workload: str,
        mtp_k: int,
        mtp_p_min: float,
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
            HIPFIRE_DPM_WARMUP_SECS=str(dpm_warmup_secs),
        )
        if workload == "mtp-greedy":
            env.update(
                HIPFIRE_QWEN_MTP="1",
                HIPFIRE_MTP_K=str(mtp_k),
                HIPFIRE_MTP_P_MIN=str(mtp_p_min),
                # Fingerprint committed IDs, not visible tokenizer chunks.
                # One MTP cycle can commit several IDs while EosFilter emits
                # fewer text events.
                HIPFIRE_EMIT_TOKEN_IDS="1",
                # Keep HipGraph out of the MTP comparison. The auto arm will
                # become the retained-PM4 arm as MTP tape support lands.
                HIPFIRE_MTP_PROPOSAL_GRAPH="0",
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

    def generate(self, message):
        """Consume a streamed generate response and return its final envelope."""
        if self.proc.poll() is not None:
            raise RuntimeError(f"daemon exited with {self.proc.returncode}")
        self.proc.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        deadline = time.monotonic() + self.timeout
        committed = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"daemon timed out on {message['type']}")
            ready, _, _ = select.select([self.proc.stdout], [], [], remaining)
            if not ready:
                raise TimeoutError(f"daemon timed out on {message['type']}")
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("daemon closed during generate")
            response = json.loads(line)
            if response.get("type") == "error":
                raise RuntimeError(response.get("message", "daemon error"))
            if response.get("type") == "committed":
                committed.append(int(response["tok_id"]))
                continue
            if response.get("type") != "done":
                continue
            digest = hashlib.sha256()
            for token in committed:
                digest.update(token.to_bytes(4, "little", signed=False))
            response["committed_events"] = len(committed)
            response["token_ids_sha256"] = digest.hexdigest()
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


def metric_key(args):
    return "decode_tok_s" if args.workload == "mtp-greedy" else "tok_s"


def validate_workload_row(args, row):
    if args.workload == "mtp-greedy":
        if row.get("mtp") is not True:
            raise RuntimeError("greedy MTP smoke fell through to a non-MTP path")
        if row.get("cycles", 0) <= 0 or row.get("tau", 0.0) <= 0.0:
            raise RuntimeError(f"greedy MTP smoke returned invalid cycle data: {row}")
        if row.get("committed_events") != row.get("tokens"):
            raise RuntimeError(f"greedy MTP stream/final token mismatch: {row}")
    return row


def run_arm(args, backend, prompt):
    daemon = Daemon(
        Path(args.daemon).resolve(),
        backend,
        args.transport,
        Path(args.work_dir) / f"product-{backend}.log",
        args.timeout,
        args.kv_mode,
        args.dpm_warmup_secs,
        args.workload,
        args.mtp_k,
        args.mtp_p_min,
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
                    "mtp_mode": "on" if args.workload == "mtp-greedy" else "off",
                    "mtp_k": args.mtp_k,
                },
            }
        )
        if args.workload == "mtp-greedy":
            request_serial = 0

            def mtp_request(max_tokens):
                nonlocal request_serial
                request_serial += 1
                return {
                    "type": "generate",
                    "id": f"redline-mtp-{backend}-{request_serial}",
                    "prompt": prompt,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": max_tokens,
                    "max_think_tokens": 1,
                    "assistant_prefix": "closed_think",
                }

            def issue(max_tokens):
                return validate_workload_row(args, daemon.generate(mtp_request(max_tokens)))

        else:

            def issue(iterations):
                return daemon.request(
                    {
                        "type": "bench_decode",
                        "context_tokens": args.context,
                        "iterations": iterations,
                    }
                )

        warmup_started = time.monotonic()
        warmups = [issue(args.warmup_iterations) for _ in range(args.warmups)]
        warmup_seconds = time.monotonic() - warmup_started
        if warmups:
            print(
                f"{backend}: warming caches... took {warmup_seconds:.2f}s",
                flush=True,
            )

        settling_started = time.monotonic()
        settling_rows = []
        settlement = None
        for _ in range(args.settle_max_runs):
            settling_rows.append(issue(args.iterations))
            settlement = analyze_stationarity(
                [row[metric_key(args)] for row in settling_rows],
                **stationarity_kwargs(args),
            )
            if settlement["stationary"]:
                break
        settling_seconds = time.monotonic() - settling_started
        if settlement is None or not settlement["stationary"]:
            latest = settlement["confirmed_window"] if settlement else None
            raise RuntimeError(
                f"{backend} failed to become stationary after "
                f"{len(settling_rows)} full-tg rows; latest={latest}"
            )
        settled = settlement["confirmed_window"]
        print(
            f"{backend}: stationary after {len(settling_rows)} full-tg rows "
            f"({settling_seconds:.2f}s, median={settled['median']:.3f}, "
            f"slope={settled['slope_pct_per_row']:+.4f}%/row, "
            f"spread={settled['spread_pct']:.3f}%)",
            flush=True,
        )

        rows = [issue(args.iterations) for _ in range(args.runs)]
        values = [row[metric_key(args)] for row in rows]
        measurement_validation = validate_measurement(values, settlement, args)
        output_hashes = sorted({row.get("token_ids_sha256") for row in rows} - {None})
        output_stable = len(output_hashes) <= 1
        tau_values = [row["tau"] for row in rows if "tau" in row]
        if args.workload == "mtp-greedy":
            measurement_validation["output_stable"] = output_stable
            measurement_validation["valid"] &= output_stable
        print(
            f"{backend}: measured median={statistics.median(values):.3f} tok/s "
            f"valid={measurement_validation['valid']}",
            flush=True,
        )
        return {
            "loaded": loaded,
            "warmups": warmups,
            "warmup_seconds": warmup_seconds,
            "settling": {
                "rows": settling_rows,
                "seconds": settling_seconds,
                "decision": settlement,
            },
            "rows": rows,
            "tok_s": {
                "min": min(values),
                "median": statistics.median(values),
                "max": max(values),
            },
            "metric": metric_key(args),
            "output_hashes": output_hashes,
            "tau": (
                {
                    "min": min(tau_values),
                    "median": statistics.median(tau_values),
                    "max": max(tau_values),
                }
                if tau_values
                else None
            ),
            "measurement_validation": measurement_validation,
        }
    finally:
        daemon.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--workload",
        choices=("ar", "mtp-greedy"),
        default="ar",
        help="stationary decode workload (default: ar)",
    )
    parser.add_argument(
        "--daemon", default=str(REPO / "target/release/examples/daemon")
    )
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--prompt-file",
        default=str(DEFAULT_MTP_PROMPT),
        help="byte-pinned plain-text prompt used by mtp-greedy",
    )
    parser.add_argument(
        "--mtp-k",
        type=int,
        default=3,
        help="MTP draft depth; 3 matches the user-facing serve default",
    )
    parser.add_argument("--mtp-p-min", type=float, default=0.0)
    parser.add_argument(
        "--warmups",
        type=int,
        default=10,
        help="number of short replay warmup requests (default: 10)",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=32,
        help="decode iterations per replay warmup request (default: 32)",
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--settle-window",
        type=int,
        default=10,
        help="rolling full-tg rows used for stationarity (default: 10)",
    )
    parser.add_argument(
        "--settle-min-runs",
        type=int,
        default=10,
        help="minimum full-tg settling rows before a candidate (default: 10)",
    )
    parser.add_argument(
        "--settle-confirmation-runs",
        type=int,
        default=10,
        help="consecutive stable rows required after a candidate (default: 10)",
    )
    parser.add_argument(
        "--settle-max-runs",
        type=int,
        default=60,
        help="fail instead of reporting if stationarity is absent (default: 60)",
    )
    parser.add_argument(
        "--settle-max-slope-pct",
        type=float,
        default=0.05,
        help="maximum absolute rolling slope in percent per row (default: 0.05)",
    )
    parser.add_argument(
        "--settle-max-spread-pct",
        type=float,
        default=1.0,
        help="maximum rolling min/max spread in percent (default: 1.0)",
    )
    parser.add_argument(
        "--settle-max-median-drift-pct",
        type=float,
        default=0.5,
        help="maximum confirmation/measurement median drift percent (default: 0.5)",
    )
    parser.add_argument("--transport", choices=("aql", "pm4"), default="aql")
    parser.add_argument(
        "--kv-mode",
        choices=("q8", "fwht2", "fwht3", "fwht4"),
        default="q8",
        help="KV layout used by both the HipGraph and retained-replay arms",
    )
    parser.add_argument("--max-seq", type=int, default=2048)
    parser.add_argument(
        "--dpm-warmup-secs",
        type=float,
        default=0.0,
        help="optional legacy memset warmup per daemon arm (default: 0)",
    )
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--work-dir", default=str(REPO / ".redline-work"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.settle_window < 3:
        parser.error("--settle-window must be at least 3")
    if args.settle_min_runs < args.settle_window:
        parser.error("--settle-min-runs must be at least --settle-window")
    if args.settle_confirmation_runs < 1:
        parser.error("--settle-confirmation-runs must be positive")
    if args.settle_max_runs < args.settle_min_runs + args.settle_confirmation_runs:
        parser.error(
            "--settle-max-runs must cover the minimum plus confirmation rows"
        )
    if args.runs < 5:
        parser.error("--runs must be at least 5 for measurement validation")
    if not 1 <= args.mtp_k <= 8:
        parser.error("--mtp-k must be in [1, 8]")
    if not 0.0 <= args.mtp_p_min <= 1.0:
        parser.error("--mtp-p-min must be in [0, 1]")
    if args.workload == "mtp-greedy" and args.kv_mode != "q8":
        parser.error("mtp-greedy currently requires --kv-mode q8")

    model = Path(args.model).expanduser().resolve()
    daemon = Path(args.daemon).expanduser().resolve()
    prompt_path = Path(args.prompt_file).expanduser().resolve()
    prompt = prompt_path.read_text() if args.workload == "mtp-greedy" else ""
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "git_commit": git_head(),
        "model": str(model),
        "model_bytes": model.stat().st_size,
        "daemon": str(daemon),
        "daemon_sha256": sha256_file(daemon),
        "device_visibility": {
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        },
        "automatic_clocks": True,
        "workload": args.workload,
        "context": args.context,
        "iterations": args.iterations,
        "warmups": args.warmups,
        "warmup_iterations": args.warmup_iterations,
        "stationarity": stationarity_kwargs(args)
        | {"max_runs": args.settle_max_runs},
        "dpm_warmup_secs": args.dpm_warmup_secs,
        "runs": args.runs,
        "transport": args.transport,
        "kv_mode": args.kv_mode,
        "mtp": {
            "k": args.mtp_k,
            "p_min": args.mtp_p_min,
            "temperature": 0.0,
            "prompt_file": str(prompt_path) if prompt else None,
            "prompt_md5": hashlib.md5(prompt.encode()).hexdigest() if prompt else None,
            "prompt_bytes": len(prompt.encode()) if prompt else None,
        },
        "hip": run_arm(args, "hip", prompt),
        "auto": run_arm(args, "auto", prompt),
    }
    hip = report["hip"]["tok_s"]["median"]
    auto = report["auto"]["tok_s"]["median"]
    report["speedup"] = auto / hip
    report["valid"] = report["hip"]["measurement_validation"]["valid"] and report[
        "auto"
    ]["measurement_validation"]["valid"]
    if args.workload == "mtp-greedy":
        report["cross_arm_output_equal"] = (
            report["hip"]["output_hashes"] == report["auto"]["output_hashes"]
        )
        report["valid"] &= report["cross_arm_output_equal"]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"hip={hip:.3f} tok/s auto={auto:.3f} tok/s "
        f"speedup={report['speedup']:.5f} valid={report['valid']} report={output}"
    )
    if not report["valid"]:
        raise SystemExit("benchmark samples drifted after settlement; report is invalid")


if __name__ == "__main__":
    main()
