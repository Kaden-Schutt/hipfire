#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Resident product decode benchmark with retained-route proof telemetry.

Unlike redline_daemon_harness.py (manual shadow capture), this drives the real
automatic lifecycle: ``auto`` records one eligible plain-AR forward and later
forwards use the prepared AQL or PM4 route. Every timed request asks the daemon
for controller state, prepared identity, and successful replay observations;
the benchmark fails closed if the automatic arm cannot prove multiple retained
positions. Models stay resident within each arm and clocks are never modified.
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
import struct
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
FRESH_ABBA_CYCLE = ("hip", "auto", "auto", "hip")
FRESH_MINIMUM_UPLIFT_PCT = 5.0

PM4_POLICY_DEFAULTS = {
    "HIPFIRE_REPLAY_PM4_QUEUES": "1",
    "HIPFIRE_REPLAY_PM4_WAIT_POLICY": "conservative",
    "HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY": "conservative",
    "HIPFIRE_REPLAY_PM4_STATEFUL": "legacy",
    "HIPFIRE_REPLAY_PM4_GCR_TRIM": "1",
}


def effective_pm4_policy(source):
    return {
        name: source.get(name, default)
        for name, default in PM4_POLICY_DEFAULTS.items()
    }


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


def md5_file(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bench_decode_input_identity(context, iterations):
    if context < 0 or iterations < 0:
        raise ValueError("bench_decode token counts must be nonnegative")

    prime = bytearray(context * 4)
    decode = bytearray(iterations * 4)
    for index in range(context):
        struct.pack_into("<I", prime, index * 4, 10 + (index % 1000))
    for index in range(iterations):
        struct.pack_into("<I", decode, index * 4, 101 + (index % 1000))

    combined = hashlib.md5()
    combined.update(prime)
    combined.update(decode)
    return {
        "prime_md5": hashlib.md5(prime).hexdigest(),
        "decode_md5": hashlib.md5(decode).hexdigest(),
        "combined_md5": combined.hexdigest(),
    }


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


def fresh_abba_order(cycles):
    if cycles < 0:
        raise ValueError("fresh ABBA cycles must be nonnegative")
    return list(FRESH_ABBA_CYCLE) * cycles


def _invalid_fresh_route_proof(backend, transport, error):
    return {
        "valid": False,
        "backend": backend,
        "transport": transport,
        "rows": 0,
        "require_complete_replay": backend == "auto",
        "retained_rows": 0,
        "observed_positions": [],
        "prepared_identities": [],
        "sequences": [],
        "errors": [error],
    }


def _fresh_route_proofs(sample, backend, transport):
    warmups = sample.get("warmups")
    measured = sample.get("measured_row")
    if not isinstance(warmups, list) or not all(
        isinstance(row, dict) for row in warmups
    ):
        invalid = _invalid_fresh_route_proof(
            backend, transport, "warmup responses are missing or malformed"
        )
        return invalid, invalid
    if not isinstance(measured, dict):
        invalid = _invalid_fresh_route_proof(
            backend, transport, "measured response is missing or malformed"
        )
        return invalid, invalid
    lifecycle = validate_route_proof(
        warmups + [measured], backend, transport
    )
    timed = validate_route_proof(
        [measured],
        backend,
        transport,
        require_complete_replay=backend == "auto",
    )
    return lifecycle, timed


def _fresh_sample_decision(sample, index, expected_identity):
    expected_backend = FRESH_ABBA_CYCLE[index % len(FRESH_ABBA_CYCLE)]
    expected_ordinal = index + 1
    expected_cycle = index // len(FRESH_ABBA_CYCLE) + 1
    expected_slot = index % len(FRESH_ABBA_CYCLE) + 1
    errors = [f"runtime: {error}" for error in sample.get("errors", [])]

    if sample.get("process_ordinal") != expected_ordinal:
        errors.append(
            f"process ordinal {sample.get('process_ordinal')!r} "
            f"!= {expected_ordinal}"
        )
    if sample.get("cycle") != expected_cycle:
        errors.append(f"cycle {sample.get('cycle')!r} != {expected_cycle}")
    if sample.get("cycle_slot") != expected_slot:
        errors.append(
            f"cycle slot {sample.get('cycle_slot')!r} != {expected_slot}"
        )
    if sample.get("backend") != expected_backend:
        errors.append(
            f"backend {sample.get('backend')!r} != {expected_backend!r}"
        )

    pid = sample.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        errors.append(f"invalid daemon pid {pid!r}")
    if sample.get("daemon_process_started") is not True:
        errors.append("sample did not prove a newly started daemon process")
    if sample.get("daemon_process_closed") is not True:
        errors.append("sample daemon process was not closed")
    if sample.get("lifecycle") != "completed":
        errors.append(f"sample lifecycle is {sample.get('lifecycle')!r}")
    if sample.get("identity") != expected_identity:
        errors.append("sample binary/model/settings identity mismatch")

    expected_inputs = {
        "measurement_combined_md5": expected_identity[
            "measurement_input_identity"
        ]["combined_md5"],
        "warmup_combined_md5": expected_identity[
            "warmup_input_identity"
        ]["combined_md5"],
    }
    if sample.get("input_identity") != expected_inputs:
        errors.append("sample bench_decode input identity mismatch")

    loaded = sample.get("loaded")
    if not isinstance(loaded, dict) or loaded.get("type") == "error":
        errors.append("sample has no successful loaded response")
    warmups = sample.get("warmups")
    if not isinstance(warmups, list) or len(warmups) != expected_identity["warmups"]:
        count = len(warmups) if isinstance(warmups, list) else None
        errors.append(
            f"sample has {count!r} warmup responses, "
            f"expected {expected_identity['warmups']}"
        )
    measured = sample.get("measured_row")
    tok_s = sample.get("tok_s")
    if (
        not isinstance(tok_s, (int, float))
        or isinstance(tok_s, bool)
        or tok_s <= 0.0
    ):
        errors.append(f"invalid measured tok/s {tok_s!r}")
    if not isinstance(measured, dict) or measured.get("tok_s") != tok_s:
        errors.append("measured row and archived tok/s do not match")
    if not isinstance(sample.get("log_path"), str) or not sample["log_path"]:
        errors.append("sample log path is missing")

    lifecycle_proof, timed_proof = _fresh_route_proofs(
        sample, expected_backend, expected_identity["transport"]
    )
    if not lifecycle_proof["valid"]:
        errors.extend(
            f"lifecycle route proof: {error}"
            for error in lifecycle_proof["errors"]
        )
    if not timed_proof["valid"]:
        errors.extend(
            f"timed route proof: {error}" for error in timed_proof["errors"]
        )
    return {
        "process_ordinal": expected_ordinal,
        "backend": expected_backend,
        "valid": not errors,
        "lifecycle_route_proof": lifecycle_proof,
        "timed_route_proof": timed_proof,
        "errors": errors,
    }


def evaluate_fresh_samples(
    samples, completed_cycles, expected_identity, stationarity
):
    expected_order = fresh_abba_order(completed_cycles)
    raw_order = [sample.get("backend") for sample in samples]
    sample_decisions = [
        _fresh_sample_decision(sample, index, expected_identity)
        for index, sample in enumerate(samples)
    ]
    complete_cycles_only = (
        len(samples) == len(expected_order) and raw_order == expected_order
    )

    values = {"hip": [], "auto": []}
    for sample in samples:
        backend = sample.get("backend")
        tok_s = sample.get("tok_s")
        if (
            backend in values
            and isinstance(tok_s, (int, float))
            and not isinstance(tok_s, bool)
            and tok_s > 0.0
        ):
            values[backend].append(tok_s)
    stationarity_decisions = {
        backend: analyze_stationarity(arm_values, **stationarity)
        for backend, arm_values in values.items()
    }

    arm_statistics = {}
    for backend, arm_values in values.items():
        arm_statistics[backend] = (
            {
                "samples": len(arm_values),
                "min": min(arm_values),
                "median": statistics.median(arm_values),
                "max": max(arm_values),
            }
            if arm_values
            else None
        )

    all_samples_valid = bool(sample_decisions) and all(
        decision["valid"] for decision in sample_decisions
    )
    both_stationary = all(
        decision["stationary"]
        for decision in stationarity_decisions.values()
    )
    comparison = {
        "estimator": "confirmed_final_window_median_ratio",
        "minimum_uplift_pct": FRESH_MINIMUM_UPLIFT_PCT,
        "hip_confirmed_median": None,
        "auto_confirmed_median": None,
        "speedup": None,
        "uplift_pct": None,
        "minimum_uplift_met": False,
    }
    if both_stationary:
        hip_median = stationarity_decisions["hip"]["confirmed_window"]["median"]
        auto_median = stationarity_decisions["auto"]["confirmed_window"]["median"]
        comparison["hip_confirmed_median"] = hip_median
        comparison["auto_confirmed_median"] = auto_median
        if hip_median:
            speedup = auto_median / hip_median
            uplift_pct = 100.0 * (speedup - 1.0)
            comparison["speedup"] = speedup
            comparison["uplift_pct"] = uplift_pct
            comparison["minimum_uplift_met"] = (
                uplift_pct >= FRESH_MINIMUM_UPLIFT_PCT
            )
    valid = (
        completed_cycles > 0
        and complete_cycles_only
        and all_samples_valid
        and both_stationary
        and comparison["minimum_uplift_met"]
    )
    errors = []
    if not complete_cycles_only:
        errors.append(
            f"raw order/sample count does not equal {completed_cycles} "
            "completed ABBA cycles"
        )
    errors.extend(
        f"sample {decision['process_ordinal']}: {error}"
        for decision in sample_decisions
        for error in decision["errors"]
    )
    for backend, decision in stationarity_decisions.items():
        if not decision["stationary"]:
            errors.append(f"{backend} arm is not independently stationary")
    if both_stationary and not comparison["minimum_uplift_met"]:
        errors.append(
            f"confirmed-window uplift {comparison['uplift_pct']!r}% is below "
            f"the predeclared {FRESH_MINIMUM_UPLIFT_PCT:.1f}% minimum"
        )

    return {
        "valid": valid,
        "complete_cycles_only": complete_cycles_only,
        "raw_order": raw_order,
        "sample_decisions": sample_decisions,
        "stationarity_decisions": stationarity_decisions,
        "comparison": comparison,
        "descriptive_statistics": {
            "promotable": valid,
            "arms": arm_statistics,
            "speedup": comparison["speedup"],
        },
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
        env.update(
            HIPFIRE_REPLAY_BACKEND=backend,
            HIPFIRE_REPLAY_TRANSPORT=transport,
            HIPFIRE_KV_MODE=kv_mode,
            HIPFIRE_CASK_OFF="1",
            HIPFIRE_AR_GRAPH="1",
            HIPFIRE_GRAPH="1",
            HIPFIRE_DPM_WARMUP_SECS=str(dpm_warmup_secs),
            HIPFIRE_LFM2_GRAPH="0",
            HIPFIRE_LFM2_DECODE_FUSION="0",
            HIPFIRE_LFM2_350M_MQ4_DECODE_FUSION="0",
            HIPFIRE_FORWARD_LOWERED="1",
            HIPFIRE_ATTN_FLASH="always",
            HIPFIRE_SPECULATION="off",
        )
        env.update(effective_pm4_policy(env))
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


def run_fresh_sample(
    args,
    backend,
    process_ordinal,
    cycle,
    cycle_slot,
    identity,
    *,
    daemon_factory=Daemon,
):
    log_path = (
        Path(args.work_dir)
        / "fresh-abba-logs"
        / f"sample-{process_ordinal:03d}-{backend}.log"
    ).resolve()
    sample = {
        "process_ordinal": process_ordinal,
        "cycle": cycle,
        "cycle_slot": cycle_slot,
        "pid": None,
        "backend": backend,
        "daemon_process_started": False,
        "daemon_process_closed": False,
        "lifecycle": "starting",
        "identity": dict(identity),
        "input_identity": {
            "measurement_combined_md5": identity[
                "measurement_input_identity"
            ]["combined_md5"],
            "warmup_combined_md5": identity[
                "warmup_input_identity"
            ]["combined_md5"],
        },
        "loaded": None,
        "warmups": [],
        "warmup_seconds": None,
        "measured_row": None,
        "measurement_seconds": None,
        "tok_s": None,
        "lifecycle_route_proof": None,
        "timed_route_proof": None,
        "log_path": str(log_path),
        "errors": [],
    }
    daemon = None
    try:
        daemon = daemon_factory(
            Path(args.daemon).expanduser().resolve(),
            backend,
            args.transport,
            log_path,
            args.timeout,
            args.kv_mode,
            args.dpm_warmup_secs,
        )
        sample["pid"] = daemon.proc.pid
        sample["daemon_process_started"] = True
        sample["lifecycle"] = "loading"
        sample["loaded"] = daemon.request(
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
        sample["lifecycle"] = "warming"
        warmup_started = time.monotonic()
        for _ in range(args.warmups):
            sample["warmups"].append(daemon.request(warmup_request))
        sample["warmup_seconds"] = time.monotonic() - warmup_started

        measured_request = {
            "type": "bench_decode",
            "context_tokens": args.context,
            "iterations": args.iterations,
            "redline_product_route": True,
        }
        sample["lifecycle"] = "measuring"
        measured_started = time.monotonic()
        sample["measured_row"] = daemon.request(measured_request)
        sample["measurement_seconds"] = time.monotonic() - measured_started
        sample["tok_s"] = sample["measured_row"].get("tok_s")
    except Exception as error:
        sample["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        if daemon is not None:
            try:
                daemon.close()
                sample["daemon_process_closed"] = True
            except Exception as error:
                sample["errors"].append(
                    f"daemon close {type(error).__name__}: {error}"
                )

    lifecycle_proof, timed_proof = _fresh_route_proofs(
        sample, backend, args.transport
    )
    sample["lifecycle_route_proof"] = lifecycle_proof
    sample["timed_route_proof"] = timed_proof
    sample["lifecycle"] = "completed" if not sample["errors"] else "failed"
    decision = _fresh_sample_decision(
        sample, process_ordinal - 1, identity
    )
    sample["valid"] = decision["valid"]
    sample["validation_errors"] = decision["errors"]
    return sample
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


def _fresh_identity(args, model, daemon):
    return {
        "model": str(model),
        "model_bytes": model.stat().st_size,
        "model_md5": md5_file(model),
        "daemon": str(daemon),
        "daemon_md5": md5_file(daemon),
        "daemon_sha256": sha256_file(daemon),
        "context": args.context,
        "iterations": args.iterations,
        "warmups": args.warmups,
        "warmup_iterations": args.warmup_iterations,
        "max_seq": args.max_seq,
        "kv_mode": args.kv_mode,
        "transport": args.transport,
        "dpm_warmup_secs": args.dpm_warmup_secs,
        "decode_mode": "plain_ar",
        "pp": 1,
        "tp": 1,
        "automatic_clocks": True,
        "pm4_policy": effective_pm4_policy(os.environ),
        "measurement_input_identity": bench_decode_input_identity(
            args.context, args.iterations
        ),
        "warmup_input_identity": bench_decode_input_identity(
            args.context, args.warmup_iterations
        ),
    }


def _write_report_atomic(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def _apply_fresh_evaluation(report, decision):
    report["raw_order"] = decision["raw_order"]
    report["complete_cycles_only"] = decision["complete_cycles_only"]
    report["sample_decisions"] = decision["sample_decisions"]
    report["stationarity_decisions"] = decision[
        "stationarity_decisions"
    ]
    report["descriptive_statistics"] = decision[
        "descriptive_statistics"
    ]
    report["comparison"] = decision["comparison"]
    report["validation_errors"] = decision["errors"]
    report["valid"] = decision["valid"]


def run_fresh_abba(args):
    output = Path(args.out)
    model = Path(args.model).expanduser().resolve()
    daemon = Path(args.daemon).expanduser().resolve()
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_at_utc": None,
        "host": socket.gethostname(),
        "git_commit": None,
        "model": str(model),
        "model_bytes": None,
        "model_md5": None,
        "daemon": str(daemon),
        "daemon_md5": None,
        "daemon_sha256": None,
        "device_visibility": {
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        },
        "automatic_clocks": True,
        "sampling_policy": "fresh_process",
        "one_daemon_process_per_sample": True,
        "order_cycle": list(FRESH_ABBA_CYCLE),
        "planned_order": fresh_abba_order(args.fresh_abba_cycles),
        "requested_cycles": args.fresh_abba_cycles,
        "completed_cycles": 0,
        "maximum_samples_per_arm": args.fresh_abba_cycles * 2,
        "stationarity": stationarity_kwargs(args),
        "binary_model_settings_identity": None,
        "measurement_input_identity": None,
        "warmup_input_identity": None,
        "raw_order": [],
        "samples": [],
        "sample_decisions": [],
        "stationarity_decisions": {},
        "descriptive_statistics": {
            "promotable": False,
            "arms": {"hip": None, "auto": None},
            "speedup": None,
        },
        "comparison": {
            "minimum_uplift_pct": FRESH_MINIMUM_UPLIFT_PCT,
            "uplift_pct": None,
            "minimum_uplift_met": False,
        },
        "complete_cycles_only": False,
        "validation_errors": [],
        "campaign_errors": [],
        "disposition": "in_progress",
        "valid": False,
    }
    _write_report_atomic(report, output)
    identity = None
    try:
        identity = _fresh_identity(args, model, daemon)
        report["git_commit"] = git_head()
        report["model_bytes"] = identity["model_bytes"]
        report["model_md5"] = identity["model_md5"]
        report["daemon_md5"] = identity["daemon_md5"]
        report["daemon_sha256"] = identity["daemon_sha256"]
        report["binary_model_settings_identity"] = identity
        report["measurement_input_identity"] = identity[
            "measurement_input_identity"
        ]
        report["warmup_input_identity"] = identity[
            "warmup_input_identity"
        ]
        _write_report_atomic(report, output)

        process_ordinal = 0
        for cycle in range(1, args.fresh_abba_cycles + 1):
            for cycle_slot, backend in enumerate(FRESH_ABBA_CYCLE, start=1):
                process_ordinal += 1
                sample = run_fresh_sample(
                    args,
                    backend,
                    process_ordinal,
                    cycle,
                    cycle_slot,
                    identity,
                )
                report["samples"].append(sample)
                decision = evaluate_fresh_samples(
                    report["samples"],
                    report["completed_cycles"],
                    identity,
                    stationarity_kwargs(args),
                )
                _apply_fresh_evaluation(report, decision)
                _write_report_atomic(report, output)
                if not sample["valid"]:
                    raise RuntimeError(
                        f"fresh sample {process_ordinal} failed validation: "
                        f"{sample.get('validation_errors', [])}"
                    )

            report["completed_cycles"] = cycle
            decision = evaluate_fresh_samples(
                report["samples"],
                report["completed_cycles"],
                identity,
                stationarity_kwargs(args),
            )
            _apply_fresh_evaluation(report, decision)
            if report["valid"]:
                report["disposition"] = "accepted"
                report["finished_at_utc"] = datetime.now(
                    timezone.utc
                ).isoformat()
                _write_report_atomic(report, output)
                speedup = report["descriptive_statistics"]["speedup"]
                print(
                    f"fresh ABBA stationary after {cycle} cycles; "
                    f"speedup={speedup:.5f} report={output}"
                )
                return
            _write_report_atomic(report, output)
    except Exception as error:
        report["campaign_errors"].append(
            f"{type(error).__name__}: {error}"
        )
        if identity is not None:
            decision = evaluate_fresh_samples(
                report["samples"],
                report["completed_cycles"],
                identity,
                stationarity_kwargs(args),
            )
            _apply_fresh_evaluation(report, decision)
        report["valid"] = False
        report["descriptive_statistics"]["promotable"] = False
        report["disposition"] = "rejected_null"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_report_atomic(report, output)
        raise SystemExit(
            f"fresh ABBA campaign failed; rejected/null report={output}"
        ) from error

    report["valid"] = False
    report["descriptive_statistics"]["promotable"] = False
    report["disposition"] = "rejected_null"
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_report_atomic(report, output)
    raise SystemExit(
        f"fresh ABBA campaign did not become valid after "
        f"{args.fresh_abba_cycles} cycles; rejected/null report={output}"
    )
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--daemon", default=str(REPO / "target/release/examples/daemon")
    )
    parser.add_argument(
        "--fresh-abba-cycles",
        type=int,
        default=0,
        help=(
            "opt into fresh-process hip/auto/auto/hip sampling for at most "
            "N completed cycles (default: 0, resident mode)"
        ),
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

    if args.fresh_abba_cycles < 0:
        parser.error("--fresh-abba-cycles must be nonnegative")

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

    if args.fresh_abba_cycles:
        run_fresh_abba(args)
        return

    model = Path(args.model).expanduser().resolve()
    daemon = Path(args.daemon).expanduser().resolve()
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
        "pm4_policy": effective_pm4_policy(os.environ),
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
    )
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
