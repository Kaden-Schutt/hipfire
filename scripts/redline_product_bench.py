#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Resident HipGraph-vs-Redline product decode benchmark.

Unlike redline_daemon_harness.py (manual shadow capture), this drives the real
default-off product lifecycle: the explicit ``redline`` backend records one
ordinary AR forward and routes later forwards through the prepared replay. The
HIP arm leaves the existing AR HipGraph enabled. Models stay resident within
each arm, clocks are never modified, and every row uses the daemon's full Qwen
reset and prefill-prime path.
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

# Product certification must not inherit an old PM4 experiment from the
# caller's environment or ~/.hipfire/config.toml. In particular, fully
# stateful gfx12 register elision depends on compiler-produced descriptor
# equality and has produced shorter, slower IBs on otherwise identical
# gfx1201 tapes. Static mode retains only queue-global invariant registers and
# re-emits program/resource/workgroup/user-data state for every dispatch.
CERTIFIED_PM4_POLICY = {
    "HIPFIRE_REPLAY_PM4_QUEUES": "1",
    "HIPFIRE_REPLAY_PM4_STATEFUL": "static",
    "HIPFIRE_REPLAY_PM4_WAIT_POLICY": "resource",
    "HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY": "required-only",
    "HIPFIRE_REPLAY_PM4_GCR_TRIM": "1",
    "HIPFIRE_REPLAY_PM4_NATIVE_PHASES": "0",
    "HIPFIRE_REPLAY_PM4_DYNAMIC_GRID": "0",
}


def backend_config_value(backend):
    """Map report-arm vocabulary to the typed replay config vocabulary."""
    return "redline" if backend == "auto" else backend


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


def validate_route_proof(
    rows, backend, transport, require_complete_replay=False
):
    errors = []
    proofs = []
    identities = set()
    sequences = set()
    observed_positions = set()
    retained_rows = 0

    for index, row in enumerate(rows):
        proof = row.get("redline_route")
        if not isinstance(proof, dict):
            errors.append(f"row {index}: missing redline_route")
            continue
        proofs.append(proof)
        if proof.get("requested_backend") != backend:
            errors.append(
                f"row {index}: backend={proof.get('requested_backend')!r}, expected {backend!r}"
            )
        if proof.get("transport") != transport:
            errors.append(
                f"row {index}: transport={proof.get('transport')!r}, expected {transport!r}"
            )
        if proof.get("fallback_reason") is not None:
            errors.append(f"row {index}: fallback={proof['fallback_reason']!r}")

        observed = proof.get("observed") or {}
        delta = observed.get("count_delta")
        if not isinstance(delta, int) or delta < 0:
            errors.append(f"row {index}: invalid observed count delta {delta!r}")
            delta = 0
        for key in ("first_position", "last_position"):
            position = observed.get(key)
            if isinstance(position, int):
                observed_positions.add(position)

        prepared = proof.get("prepared")
        sequence = proof.get("sequence") or {}
        if backend == "hip":
            if proof.get("state") != "hip":
                errors.append(f"row {index}: HIP baseline state={proof.get('state')!r}")
            if proof.get("retained_replay_observed") or delta:
                errors.append(f"row {index}: HIP baseline observed retained replay")
            if prepared is not None:
                errors.append(f"row {index}: HIP baseline owns a prepared route")
            continue

        if proof.get("state") != "ready":
            errors.append(f"row {index}: automatic route state={proof.get('state')!r}")
        if proof.get("execution_mode") != "plain_ar":
            errors.append(f"row {index}: automatic route was not plain AR")
        if require_complete_replay:
            iterations = row.get("iterations")
            if not proof.get("retained_replay_observed"):
                errors.append(f"row {index}: timed row observed no retained replay")
            if not isinstance(iterations, int) or iterations <= 0:
                errors.append(
                    f"row {index}: invalid timed iteration count {iterations!r}"
                )
            elif delta != iterations:
                errors.append(
                    f"row {index}: observed {delta} retained replays for "
                    f"{iterations} timed iterations"
                )
            context = row.get("context_tokens")
            if not isinstance(context, int) or context <= 0:
                errors.append(
                    f"row {index}: invalid timed context position {context!r}"
                )
            elif isinstance(iterations, int) and iterations > 0:
                first_position = observed.get("first_position")
                last_position = observed.get("last_position")
                expected_last = context + iterations - 1
                if first_position != context:
                    errors.append(
                        f"row {index}: first replay position {first_position!r} "
                        f"!= {context}"
                    )
                if last_position != expected_last:
                    errors.append(
                        f"row {index}: last replay position {last_position!r} "
                        f"!= {expected_last}"
                    )
        if not isinstance(prepared, dict):
            errors.append(f"row {index}: automatic route has no prepared identity")
            continue
        packets = prepared.get("packets")
        if not isinstance(packets, int) or packets <= 0:
            errors.append(f"row {index}: packet identity unavailable")
        dispatches = prepared.get("dispatches")
        if not isinstance(dispatches, int) or dispatches <= 0:
            errors.append(f"row {index}: invalid dispatch count {dispatches!r}")
        command_dwords = prepared.get("command_dwords")
        if transport == "pm4" and (
            not isinstance(command_dwords, int) or command_dwords <= 0
        ):
            errors.append(f"row {index}: PM4 command identity unavailable")
        if transport == "aql" and command_dwords is not None:
            errors.append(f"row {index}: AQL row unexpectedly reports PM4 commands")
        identities.add(
            (
                dispatches,
                packets,
                prepared.get("queue_id"),
                command_dwords,
            )
        )
        launches = sequence.get("launches")
        sequence_hash = sequence.get("hash")
        if launches != dispatches:
            errors.append(
                f"row {index}: sequence launches {launches!r} != dispatches {dispatches!r}"
            )
        if not isinstance(sequence_hash, str) or sequence_hash == "0000000000000000":
            errors.append(f"row {index}: invalid sequence hash {sequence_hash!r}")
        sequences.add(
            (launches, sequence.get("unique_kernels"), sequence_hash)
        )
        if proof.get("retained_replay_observed") and delta > 0:
            retained_rows += 1

    if not proofs:
        errors.append("no route-proof rows")
    if backend == "auto":
        if retained_rows == 0:
            errors.append("automatic arm observed no successful retained replay")
        if len(observed_positions) < 2:
            errors.append("automatic arm did not observe multiple replay positions")
        if len(identities) != 1:
            errors.append(f"prepared identity changed across rows: {sorted(identities)!r}")
        if len(sequences) != 1:
            errors.append(f"sequence identity changed across rows: {sorted(sequences)!r}")

    return {
        "valid": not errors,
        "backend": backend,
        "transport": transport,
        "rows": len(proofs),
        "require_complete_replay": require_complete_replay,
        "retained_rows": retained_rows,
        "observed_positions": sorted(observed_positions),
        "prepared_identities": [list(identity) for identity in sorted(identities)],
        "sequences": [list(sequence) for sequence in sorted(sequences)],
        "errors": errors,
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
    ):
        self.timeout = timeout
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("w")
        env = dict(os.environ)
        # ``auto`` is the report-arm name and ReplayBackendRequest value. In
        # the typed config surface it means "follow immutable model admission";
        # ``redline`` is the explicit opt-in required to certify an unadmitted
        # architecture without changing product defaults first.
        configured_backend = backend_config_value(backend)
        env.update(
            HIPFIRE_REPLAY_BACKEND=configured_backend,
            HIPFIRE_REPLAY_TRANSPORT=transport,
            HIPFIRE_KV_MODE=kv_mode,
            HIPFIRE_CASK_OFF="1",
            HIPFIRE_AR_GRAPH="1",
            HIPFIRE_GRAPH="1",
            HIPFIRE_DPM_WARMUP_SECS=str(dpm_warmup_secs),
        )
        env.update(CERTIFIED_PM4_POLICY)
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
        args.dpm_warmup_secs,
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
        warmup_request = {
            "type": "bench_decode",
            "context_tokens": args.context,
            "iterations": args.warmup_iterations,
            "redline_product_route": True,
        }
        request = {
            "type": "bench_decode",
            "context_tokens": args.context,
            "iterations": args.iterations,
            "redline_product_route": True,
        }
        warmup_started = time.monotonic()
        warmups = [daemon.request(warmup_request) for _ in range(args.warmups)]
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
            settling_rows.append(daemon.request(request))
            settlement = analyze_stationarity(
                [row["tok_s"] for row in settling_rows],
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

        rows = [daemon.request(request) for _ in range(args.runs)]
        values = [row["tok_s"] for row in rows]
        measurement_validation = validate_measurement(values, settlement, args)
        print(
            f"{backend}: measured median={statistics.median(values):.3f} tok/s "
            f"valid={measurement_validation['valid']}",
            flush=True,
        )
        lifecycle_route_proof = validate_route_proof(
            warmups + settling_rows + rows, backend, args.transport
        )
        route_proof = validate_route_proof(
            rows,
            backend,
            args.transport,
            require_complete_replay=backend == "auto",
        )
        print(
            f"{backend}: timed route proof valid={route_proof['valid']} "
            f"retained_rows={route_proof['retained_rows']} "
            f"positions={route_proof['observed_positions']} "
            f"lifecycle_valid={lifecycle_route_proof['valid']}",
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
            "measurement_validation": measurement_validation,
            "lifecycle_route_proof": lifecycle_route_proof,
            "route_proof": route_proof,
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
        default=120,
        help="fail instead of reporting if stationarity is absent (default: 120)",
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
    parser.add_argument(
        "--expected-model-sha256",
        help="fail before loading the GPU when the model digest differs",
    )
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

    model = Path(args.model).expanduser().resolve()
    daemon = Path(args.daemon).expanduser().resolve()
    model_sha256 = sha256_file(model)
    if (
        args.expected_model_sha256 is not None
        and model_sha256.lower() != args.expected_model_sha256.lower()
    ):
        raise SystemExit(
            "model SHA-256 mismatch: "
            f"expected {args.expected_model_sha256.lower()}, got {model_sha256}"
        )
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "git_commit": git_head(),
        "model": str(model),
        "model_bytes": model.stat().st_size,
        "model_sha256": model_sha256,
        "daemon": str(daemon),
        "daemon_sha256": sha256_file(daemon),
        "device_visibility": {
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        },
        "automatic_clocks": True,
        "context": args.context,
        "iterations": args.iterations,
        "warmups": args.warmups,
        "warmup_iterations": args.warmup_iterations,
        "stationarity": stationarity_kwargs(args)
        | {"max_runs": args.settle_max_runs},
        "dpm_warmup_secs": args.dpm_warmup_secs,
        "runs": args.runs,
        "transport": args.transport,
        "pm4_policy": dict(CERTIFIED_PM4_POLICY),
        "kv_mode": args.kv_mode,
        "hip": run_arm(args, "hip"),
        "auto": run_arm(args, "auto"),
    }
    hip = report["hip"]["tok_s"]["median"]
    auto = report["auto"]["tok_s"]["median"]
    report["speedup"] = auto / hip
    report["valid"] = (
        report["hip"]["measurement_validation"]["valid"]
        and report["auto"]["measurement_validation"]["valid"]
        and report["hip"]["route_proof"]["valid"]
        and report["auto"]["route_proof"]["valid"]
        and report["hip"]["lifecycle_route_proof"]["valid"]
        and report["auto"]["lifecycle_route_proof"]["valid"]
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"hip={hip:.3f} tok/s auto={auto:.3f} tok/s "
        f"speedup={report['speedup']:.5f} valid={report['valid']} report={output}"
    )
    if not report["valid"]:
        raise SystemExit("benchmark samples or route proof are invalid")


if __name__ == "__main__":
    main()
