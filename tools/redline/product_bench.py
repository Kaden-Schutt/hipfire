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
import errno
import hashlib
import json
import os
import re
import secrets
import tempfile
import select
import signal
import socket
import statistics
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]

# One-turn Flagstaff smoke through scripts/serve_harness.py (CLI/serve path).
# Reserve a full 512-token visible-answer window beyond the concrete think cap;
# gfx1100 can otherwise produce a coherent answer but hit finish=length at 768.
COHERENCE_PROMPT = "What is the origin of Flagstaff, Arizona's name?"
COHERENCE_THINKING = "low"
COHERENCE_THINKING_CAP_TOKENS = 512
COHERENCE_MAX_TOKENS = 1024
COHERENCE_SEED = 1
COHERENCE_SAMPLING = "registry"
COHERENCE_MODE = "battery"
COHERENCE_MTP = "off"
assert COHERENCE_MAX_TOKENS > COHERENCE_THINKING_CAP_TOKENS

# Custom coherence maps thinking budget to concrete cap (mirrors serve_harness).
COHERENCE_THINKING_BUDGET = {
    "off": 1,
    "low": 512,
    "med": 2048,
    "high": 8192,
    "xhigh": 24576,
    "max": 32768,
    "uncapped": 0,
}

# Opt-in retained-replay proof marker. Product coherence enables it via temporary
# serve_harness TOML (`diagnostic.replay.route_proof_log`); the runtime still lowers
# through HIPFIRE_REPLAY_ROUTE_PROOF_LOG / process_value.
ROUTE_PROOF_MARKER = "HIPFIRE_REPLAY_ROUTE_PROOF"
# Current well-formed marker: transport, position, request_id, replays (all required).
_ROUTE_PROOF_CURRENT_RE = re.compile(
    r"HIPFIRE_REPLAY_ROUTE_PROOF\s+transport=([A-Za-z0-9_]+)\s+position=(\d+)\b"
    r"\s+request_id=([A-Za-z0-9_.:-]+)\s+replays=(\d+)\b"
)
# Legacy/unscoped form (transport+position only). Kept for non-cert parse compatibility;
# certification validators reject these via full literal-vs-valid accounting.
_ROUTE_PROOF_LEGACY_RE = re.compile(
    r"HIPFIRE_REPLAY_ROUTE_PROOF\s+transport=([A-Za-z0-9_]+)\s+position=(\d+)\b"
)
# Back-compat alias: optional request_id/replays (first match only was the old contract).
_ROUTE_PROOF_LINE_RE = re.compile(
    r"HIPFIRE_REPLAY_ROUTE_PROOF\s+transport=([A-Za-z0-9_]+)\s+position=(\d+)\b"
    r"(?:\s+request_id=([A-Za-z0-9_.:-]+)\s+replays=(\d+)\b)?"
)

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


def pm4_policy_with_overrides(overrides):
    """Apply explicit, reportable experiment overrides to certified PM4 policy."""
    policy = dict(CERTIFIED_PM4_POLICY)
    forbidden = {
        "HIPFIRE_REPLAY_BACKEND",
        "HIPFIRE_REPLAY_MANUAL_CAPTURE",
        "HIPFIRE_REPLAY_TRANSPORT",
    }
    for item in overrides:
        key, separator, value = item.partition("=")
        if (
            not separator
            or not key.startswith("HIPFIRE_REPLAY_PM4_")
            or key in forbidden
            or not value
        ):
            raise ValueError(
                "--pm4-policy-override expects "
                "HIPFIRE_REPLAY_PM4_<NAME>=<VALUE>; backend, transport, and "
                "manual-capture controls are not policy overrides"
            )
        policy[key] = value
    return policy


def pm4_policy_for(args):
    """Resolve a parsed candidate policy, retaining library-call compatibility."""
    return getattr(args, "pm4_policy", CERTIFIED_PM4_POLICY)


def backend_config_value(backend):
    """Map report-arm vocabulary to the typed replay config vocabulary."""
    return "redline" if backend == "auto" else backend


def _kill_serve_process_group(serve_pid_path):
    """Timeout cleanup: killpg the known session PGID from the PID file.

    With Popen(start_new_session=True) the CLI leader PID equals the PGID.
    Call os.killpg(pid, SIGKILL) directly — do not gate on os.getpgid(leader),
    which returns ESRCH after the leader exits while descendants may remain.
    Missing/invalid PID values and unexpected kill errors are cleanup failures;
    ESRCH / ProcessLookupError from killpg are benign.
    """
    cleanup_errors = []
    try:
        raw = Path(serve_pid_path).read_text(encoding="utf-8").strip()
    except OSError as read_error:
        cleanup_errors.append(
            f"serve PID file unreadable ({serve_pid_path}): {read_error}"
        )
        return cleanup_errors
    try:
        pid = int(raw)
    except ValueError:
        cleanup_errors.append(
            f"serve PID file invalid contents {raw!r} ({serve_pid_path})"
        )
        return cleanup_errors
    if pid <= 0:
        cleanup_errors.append(
            f"serve PID not positive: {pid} ({serve_pid_path})"
        )
        return cleanup_errors
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as kill_error:
        # ESRCH is benign (race with natural exit / empty group).
        if kill_error.errno != errno.ESRCH:
            cleanup_errors.append(f"killpg({pid}) failed: {kill_error}")
    return cleanup_errors


def _flagstaff_answer_errors(content):
    """Semantic checks for the one-turn Flagstaff coherence answer.

    Require a real flag-object concept plus a naming/history cue.
    Accept standalone ``flag``, ``flagpole``, ``flag pole``, ``flag-staff``,
    or ``flagstaff`` only when explicitly accompanied by ``pole``/``mast``.
    Reject bare city facts and pine-tree naming without a real flag object.
    """
    errors = []
    lowered = content.lower()

    # Word-ish tokens: letters with internal hyphen/apostrophe.
    tokens = re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", lowered)
    token_set = set(tokens)

    has_standalone_flag = "flag" in token_set
    has_flagpole_word = "flagpole" in token_set or "flag-staff" in token_set
    # Multi-word "flag pole" / "flag staff" as adjacent tokens.
    has_flag_pole_phrase = any(
        tokens[i] == "flag" and tokens[i + 1] in ("pole", "staff")
        for i in range(len(tokens) - 1)
    )
    # Bare "flagstaff" (the city name) only counts with an explicit pole/mast.
    has_flagstaff_with_support = "flagstaff" in token_set and (
        "pole" in token_set or "mast" in token_set
    )
    has_flag_object = (
        has_standalone_flag
        or has_flagpole_word
        or has_flag_pole_phrase
        or has_flagstaff_with_support
    )
    if not has_flag_object:
        errors.append(
            "answer missing real flag object "
            "(standalone flag / flagpole / flag pole / flag-staff / "
            "flagstaff with pole or mast)"
        )

    naming_cues = (
        "named",
        "name",
        "origin",
        "1876",
        "centennial",
        "boston party",
        "fourth of july",
        "4th of july",
    )
    if not any(cue in lowered for cue in naming_cues):
        errors.append(
            "answer missing naming/history cue "
            "(named/name/origin/1876/centennial/boston party/fourth of july/4th of july)"
        )
    return errors


def _coherence_thinking_cap(thinking: str) -> int:
    """Concretem thinking cap for a budget string (mirrors serve_harness)."""
    return COHERENCE_THINKING_BUDGET.get(thinking, COHERENCE_THINKING_CAP_TOKENS)


def _custom_coherence_answer_errors(content: str, expected_substrings) -> list[str]:
    """Validate custom prompt answer: non-empty plus all expected substrings."""
    errors: list[str] = []
    if not isinstance(content, str) or not content.strip():
        errors.append("assistant_content must be a nonempty string")
        return errors
    lowered = content.lower()
    if expected_substrings:
        for sub in expected_substrings:
            if not isinstance(sub, str) or not sub:
                errors.append(f"invalid expected substring {sub!r}")
                continue
            if sub.lower() not in lowered:
                errors.append(
                    f"answer missing expected substring {sub!r} (case-insensitive)"
                )
    return errors

def sampled_output_parity_errors(hip_rows: object, replay_rows: object, *, label: str) -> list[str]:
    """Pure helper: exact sampled-output parity for paired rows.

    Compares paired dict rows with exact Python equality for
    ``assistant_content``, ``ctx``, and ``gen``. Rejects non-list,
    count, type, and missing-field mismatches with deterministic
    actionable errors. Performs no normalization, case folding, or
    substring matching; byte-identical rows pass.
    """
    errors: list[str] = []
    prefix = f"{label} " if label else ""
    prefix_colon = f"{label}: " if label else ""
    if not isinstance(hip_rows, list):
        errors.append(f"{prefix_colon}hip_rows must be a list, got {type(hip_rows).__name__}")
        if not isinstance(replay_rows, list):
            errors.append(f"{prefix_colon}replay_rows must be a list, got {type(replay_rows).__name__}")
        return errors
    if not isinstance(replay_rows, list):
        errors.append(f"{prefix_colon}replay_rows must be a list, got {type(replay_rows).__name__}")
        return errors
    if len(hip_rows) != len(replay_rows):
        errors.append(
            f"{prefix_colon}sampled-output row count differs: hip {len(hip_rows)} vs replay {len(replay_rows)}"
        )
        return errors
    for idx, (hip_row, replay_row) in enumerate(zip(hip_rows, replay_rows), 1):
        if not isinstance(hip_row, dict):
            errors.append(f"{prefix}turn {idx}: hip row must be an object, got {type(hip_row).__name__}")
            continue
        if not isinstance(replay_row, dict):
            errors.append(f"{prefix}turn {idx}: replay row must be an object, got {type(replay_row).__name__}")
            continue
        for field in ("assistant_content", "ctx", "gen"):
            hip_has = field in hip_row
            replay_has = field in replay_row
            if not hip_has or not replay_has:
                if not hip_has and not replay_has:
                    errors.append(f"{prefix}turn {idx}: {field} is missing from both rows")
                elif not hip_has:
                    errors.append(f"{prefix}turn {idx}: hip {field} is missing")
                else:
                    errors.append(f"{prefix}turn {idx}: replay {field} is missing")
                continue
            hip_val = hip_row[field]
            replay_val = replay_row[field]
            if hip_val != replay_val:
                if field == "assistant_content":
                    errors.append(f"{prefix}turn {idx}: sampled output differs between HIP and PM4")
                elif field == "ctx":
                    errors.append(
                        f"{prefix}turn {idx}: prompt-token count differs between HIP and PM4 (ctx hip={hip_val!r} vs replay={replay_val!r})"
                    )
                else:  # gen
                    errors.append(
                        f"{prefix}turn {idx}: generated-token count differs between HIP and PM4 (gen hip={hip_val!r} vs replay={replay_val!r})"
                    )
    return errors


def _is_coherence_custom(args) -> bool:
    return getattr(args, "coherence_prompt_file", None) is not None


def _coherence_custom_files(args):
    """Return resolved custom prompt file, bytes, md5, sha256, text."""
    raw = getattr(args, "coherence_prompt_file", None)
    if raw is None:
        return None
    path = Path(raw).expanduser().resolve()
    data = path.read_bytes()
    # Hash exact byte sequence (no newline normalization).
    md5 = hashlib.md5(data).hexdigest()
    sha256 = hashlib.sha256(data).hexdigest()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"coherence prompt file {path} must be valid UTF-8: {exc}"
        ) from exc
    if not text.strip():
        raise RuntimeError(
            f"coherence prompt file {path} must be nonempty"
        )
    return {
        "path": path,
        "bytes": data,
        "text": text,
        "md5": md5,
        "sha256": sha256,
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

def _is_positive_int(value):
    """Reject bool (bool subclasses int) and non-positive values."""
    return type(value) is int and value > 0


def _is_non_negative_int(value):
    return type(value) is int and value >= 0



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
        "valid": bool(
            enough_rows
            and stats is not None
            and stats["stable"]
            and drift_pct <= args.settle_max_median_drift_pct
        ),
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
        if not _is_non_negative_int(delta):
            errors.append(f"row {index}: invalid observed count delta {delta!r}")
            delta = 0
        for key in ("first_position", "last_position"):
            position = observed.get(key)
            if type(position) is int:
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
            if not _is_positive_int(iterations):
                errors.append(
                    f"row {index}: invalid timed iteration count {iterations!r}"
                )
            elif delta != iterations:
                errors.append(
                    f"row {index}: observed {delta} retained replays for "
                    f"{iterations} timed iterations"
                )
            context = row.get("context_tokens")
            if not _is_positive_int(context):
                errors.append(
                    f"row {index}: invalid timed context position {context!r}"
                )
            elif _is_positive_int(iterations):
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
        packets_ok = _is_positive_int(packets)
        if not packets_ok:
            errors.append(f"row {index}: packet identity unavailable")
        dispatches = prepared.get("dispatches")
        dispatches_ok = _is_positive_int(dispatches)
        if not dispatches_ok:
            errors.append(f"row {index}: invalid dispatch count {dispatches!r}")
        # prepared identity tuple index 2 is queue_id (a queue identifier), never a phase count
        queue_id = prepared.get("queue_id")
        queue_id_ok = _is_positive_int(queue_id)
        if not queue_id_ok:
            errors.append(f"row {index}: invalid queue_id {queue_id!r}")
        queues = prepared.get("queues")
        queues_ok = _is_positive_int(queues)
        if not queues_ok:
            errors.append(f"row {index}: invalid queues {queues!r}")
        phases = prepared.get("phases")
        phases_ok = _is_positive_int(phases)
        if not phases_ok:
            errors.append(f"row {index}: invalid phases {phases!r}")
        command_dwords = prepared.get("command_dwords")
        command_dwords_ok = True
        if transport == "pm4":
            command_dwords_ok = _is_positive_int(command_dwords)
            if not command_dwords_ok:
                errors.append(f"row {index}: PM4 command identity unavailable")
        elif transport == "aql" and command_dwords is not None:
            command_dwords_ok = False
            errors.append(f"row {index}: AQL row unexpectedly reports PM4 commands")
        # Report shape:
        # [dispatches, packets, queue_id, command_dwords, queues, phases]
        # Only record fully-valid tuples so sorted() never TypeErrors on None.
        if (
            packets_ok
            and dispatches_ok
            and queue_id_ok
            and command_dwords_ok
            and queues_ok
            and phases_ok
        ):
            identities.add(
                (
                    dispatches,
                    packets,
                    queue_id,
                    command_dwords,
                    queues,
                    phases,
                )
            )
        launches = sequence.get("launches")
        unique_kernels = sequence.get("unique_kernels")
        sequence_hash = sequence.get("hash")
        launches_ok = _is_positive_int(launches)
        if not launches_ok:
            errors.append(f"row {index}: invalid sequence launches {launches!r}")
        elif dispatches_ok and launches != dispatches:
            launches_ok = False
            errors.append(
                f"row {index}: sequence launches {launches!r} != dispatches {dispatches!r}"
            )
        unique_kernels_ok = _is_positive_int(unique_kernels)
        if not unique_kernels_ok:
            errors.append(
                f"row {index}: invalid unique_kernels {unique_kernels!r}"
            )
        hash_ok = (
            isinstance(sequence_hash, str)
            and len(sequence_hash) == 16
            and all(ch in "0123456789abcdefABCDEF" for ch in sequence_hash)
            and sequence_hash != "0000000000000000"
        )
        if not hash_ok:
            errors.append(f"row {index}: invalid sequence hash {sequence_hash!r}")
        if launches_ok and unique_kernels_ok and hash_ok:
            sequences.add(
                (launches, unique_kernels, sequence_hash)
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


def require_retained_pm4(row):
    proof = validate_route_proof([row], "auto", "pm4")
    if proof["valid"]:
        return
    route = row.get("redline_route") or {}
    fallback = route.get("fallback_reason")
    detail = f": {fallback}" if fallback else ""
    raise RuntimeError(
        "Redline PM4 did not engage on the first automatic decode row"
        f"{detail}; route proof: {'; '.join(proof['errors'])}"
    )


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
        pm4_policy,
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
        env.update(pm4_policy)
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


def run_pm4_preflight(args):
    daemon = Daemon(
        Path(args.daemon).resolve(),
        "auto",
        "pm4",
        Path(args.work_dir) / "product-pm4-preflight.log",
        args.timeout,
        args.kv_mode,
        0.0,
        pm4_policy_for(args),
    )
    started = time.monotonic()
    try:
        loaded = daemon.request(
            {
                "type": "load",
                "model": str(Path(args.model).expanduser().resolve()),
                "params": {
                    "max_seq": args.max_seq,
                    "kv_mode": args.kv_mode,
                    "dflash_mode": "off",
                    "dspark_mode": "off",
                },
            }
        )
        row = daemon.request(
            {
                "type": "bench_decode",
                "context_tokens": args.context,
                # Dense LFM uses its first decode for allocation warmup, its
                # second to record/prepare the retained tape, and therefore
                # needs two further iterations to prove replay at distinct
                # positions before benchmark warmup.
                "iterations": 4,
                "redline_product_route": True,
            }
        )
        require_retained_pm4(row)
        route_proof = validate_route_proof([row], "auto", "pm4")
        return {
            "seconds": time.monotonic() - started,
            "loaded": loaded,
            "smoke": row,
            "redline_route": row.get("redline_route"),
            "route_proof": route_proof,
        }
    finally:
        daemon.close()



def _unique_coherence_daemon(src: Path, work_dir: Path, backend: str) -> Path:
    """Hard-link (or copy) daemon under a unique basename for orphan-safe pkill."""
    work_dir.mkdir(parents=True, exist_ok=True)
    dest = work_dir / f"daemon-product-{backend}-coherence-{os.getpid()}"
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)
        mode = dest.stat().st_mode
        dest.chmod(mode | 0o111)
    return dest


def _allocate_loopback_port():
    """Reserve an ephemeral loopback TCP port (OS-assigned, released after bind)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def parse_route_proof_log(text):
    """Extract HIPFIRE_REPLAY_ROUTE_PROOF markers with full literal accounting.

    Every literal ``HIPFIRE_REPLAY_ROUTE_PROOF`` occurrence is counted. A hit is
    emitted only for a well-formed single-marker line (current form preferred;
    legacy transport+position still parses for non-cert consumers). Lines with
    two+ markers, or a marker that fails both forms, are recorded as malformed
    so validators cannot ignore unparsed evidence.
    """
    if not text:
        return {
            "hits": [],
            "occurrences": [],
            "malformed": [],
            "literal_count": 0,
            "lines": [],
        }
    hits = []
    occurrences = []
    malformed = []
    lines = []
    for raw in text.splitlines():
        if ROUTE_PROOF_MARKER not in raw:
            continue
        stripped = raw.strip()
        # Count every literal marker token on the line (not regex .search once).
        start = 0
        line_occ_indices = []
        while True:
            idx = raw.find(ROUTE_PROOF_MARKER, start)
            if idx < 0:
                break
            line_occ_indices.append(idx)
            start = idx + len(ROUTE_PROOF_MARKER)
        if not line_occ_indices:
            continue
        lines.append(stripped)
        for occ_idx in line_occ_indices:
            occurrences.append(
                {
                    "line": stripped,
                    "offset": occ_idx,
                }
            )
        # Two+ markers on one line: none are valid certification hits.
        if len(line_occ_indices) != 1:
            malformed.append(
                {
                    "line": stripped,
                    "reason": "multiple_markers_on_line",
                    "marker_count": len(line_occ_indices),
                }
            )
            continue
        current = _ROUTE_PROOF_CURRENT_RE.search(raw)
        if current is not None:
            # Reject if the match does not cover the sole marker token, or if
            # extra marker-shaped garbage remains after the matched span.
            if current.start() > line_occ_indices[0]:
                malformed.append(
                    {
                        "line": stripped,
                        "reason": "malformed_marker",
                    }
                )
                continue
            trailing = raw[current.end() :]
            if ROUTE_PROOF_MARKER in trailing:
                malformed.append(
                    {
                        "line": stripped,
                        "reason": "multiple_markers_on_line",
                        "marker_count": 2,
                    }
                )
                continue
            hits.append(
                {
                    "line": stripped,
                    "transport": current.group(1),
                    "position": int(current.group(2)),
                    "request_id": current.group(3),
                    "replays": int(current.group(4)),
                }
            )
            continue
        legacy = _ROUTE_PROOF_LEGACY_RE.search(raw)
        if legacy is not None and legacy.start() <= line_occ_indices[0]:
            trailing = raw[legacy.end() :]
            # Current-form fields after legacy prefix are handled above; leftover
            # non-marker text is allowed on legacy lines for parse compatibility.
            if ROUTE_PROOF_MARKER in trailing:
                malformed.append(
                    {
                        "line": stripped,
                        "reason": "multiple_markers_on_line",
                        "marker_count": 2,
                    }
                )
                continue
            # If the line has request_id/replays-shaped text that failed CURRENT,
            # treat as malformed rather than silently dropping fields.
            if "request_id=" in raw[legacy.end() :] or "replays=" in raw[legacy.end() :]:
                malformed.append(
                    {
                        "line": stripped,
                        "reason": "malformed_marker",
                    }
                )
                continue
            hits.append(
                {
                    "line": stripped,
                    "transport": legacy.group(1),
                    "position": int(legacy.group(2)),
                    "request_id": None,
                    "replays": None,
                }
            )
            continue
        malformed.append(
            {
                "line": stripped,
                "reason": "malformed_marker",
            }
        )
    return {
        "hits": hits,
        "occurrences": occurrences,
        "malformed": malformed,
        "literal_count": len(occurrences),
        "lines": lines,
    }


def collect_route_proof_evidence(serve_log_path, stdout="", stderr=""):
    """Parse serve.log plus harness stdout/stderr for retained-route proof lines.

    Preserves every literal marker occurrence and malformed line so validators
    cannot ignore unparsed evidence. ``hits`` remains the list of successfully
    parsed markers (legacy included); ``count`` is ``len(hits)`` for back-compat.
    """
    chunks = []
    path = Path(serve_log_path) if serve_log_path is not None else None
    if path is not None and path.is_file():
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
    if stdout:
        chunks.append(stdout)
    if stderr:
        chunks.append(stderr)
    parsed = parse_route_proof_log("\n".join(chunks))
    hits = parsed["hits"]
    first = hits[0] if hits else None
    literal_count = parsed["literal_count"]
    return {
        "observed": bool(hits) or literal_count > 0 or bool(parsed["malformed"]),
        "transport": None if first is None else first["transport"],
        "position": None if first is None else first["position"],
        "marker": None if first is None else first["line"],
        "lines": list(parsed["lines"]),
        "count": len(hits),
        "hits": hits,
        "occurrences": list(parsed["occurrences"]),
        "malformed": list(parsed["malformed"]),
        "literal_count": literal_count,
    }


def validate_coherence_route_evidence(backend, transport, evidence, rows=None):
    """Gate CLI/serve coherence on request-bound retained-replay proof markers.

    Successful harness rows bind markers by nonempty ``request_id``. Auto/PM4
    requires exactly one well-formed current PM4 marker per successful row
    (positive ``replays``), rejects legacy/unscoped markers, extras, malformed
    literal occurrences, two markers on one line, and keeps HIP at zero literal
    marker occurrences.
    """
    errors = []
    required = backend == "auto" and transport == "pm4"
    hits = evidence.get("hits")
    if not isinstance(hits, list):
        hits = []
    occurrences = evidence.get("occurrences")
    if not isinstance(occurrences, list):
        occurrences = []
    malformed = evidence.get("malformed")
    if not isinstance(malformed, list):
        malformed = []
    literal_count = evidence.get("literal_count")
    if not isinstance(literal_count, int) or isinstance(literal_count, bool):
        # Fall back: count from occurrences, else from hit lines (legacy evidence).
        if occurrences:
            literal_count = len(occurrences)
        else:
            # Reconstruct literal count from lines/hits when older evidence lacks
            # occurrence accounting (still reject if hits alone look incomplete).
            lines = evidence.get("lines") or []
            if lines:
                literal_count = sum(
                    str(line).count(ROUTE_PROOF_MARKER) for line in lines
                )
            else:
                literal_count = sum(
                    str(hit.get("line") or "").count(ROUTE_PROOF_MARKER) for hit in hits
                )
                if literal_count == 0 and hits:
                    literal_count = len(hits)

    observed = (
        bool(hits)
        or literal_count > 0
        or bool(malformed)
        or bool(evidence.get("observed"))
    )
    got_transport = evidence.get("transport")
    if hits and got_transport is None:
        got_transport = hits[0].get("transport")

    successful_rows = []
    if isinstance(rows, list):
        for index, row in enumerate(rows, 1):
            if isinstance(row, dict):
                successful_rows.append((index, row))

    # Full accounting: every literal marker must parse to exactly one hit.
    accounting_mismatch = literal_count != len(hits) or bool(malformed)

    if backend == "hip":
        if hits or observed or literal_count > 0 or malformed:
            errors.append(
                "HIP coherence must not emit retained route proof "
                f"(got transport={got_transport!r} position={evidence.get('position')!r}"
                f", literal_count={literal_count}, malformed={len(malformed)})"
            )
    elif required:
        total_hits = len(hits)
        if not successful_rows:
            if total_hits == 0 and literal_count == 0 and not observed:
                errors.append(
                    "auto PM4 coherence requires HIPFIRE_REPLAY_ROUTE_PROOF "
                    "marker from the serve daemon"
                )
            else:
                # Markers alone cannot certify without successful harness rows.
                errors.append(
                    "auto PM4 coherence requires successful harness row(s) with "
                    "nonempty request_id to bind route proof markers"
                )
                if accounting_mismatch:
                    errors.append(
                        "auto PM4 coherence rejects unparsed/malformed route proof "
                        f"marker occurrence(s) (literal_count={literal_count}, "
                        f"parsed_hits={total_hits}, malformed={len(malformed)})"
                    )
        else:
            expected = len(successful_rows)
            if literal_count == 0 and total_hits == 0:
                errors.append(
                    "auto PM4 coherence requires HIPFIRE_REPLAY_ROUTE_PROOF "
                    "marker from the serve daemon"
                )
            else:
                if accounting_mismatch:
                    errors.append(
                        "auto PM4 coherence rejects unparsed/malformed route proof "
                        f"marker occurrence(s) (literal_count={literal_count}, "
                        f"parsed_hits={total_hits}, malformed={len(malformed)})"
                    )

                # Count every parsed hit before transport filtering; also require
                # literal occurrence count to match expected row count.
                if total_hits != expected:
                    errors.append(
                        f"auto PM4 coherence expected exactly {expected} route "
                        f"proof marker(s) total, got {total_hits}"
                    )
                elif literal_count != expected:
                    errors.append(
                        f"auto PM4 coherence expected exactly {expected} route "
                        f"proof marker(s) total, got {literal_count} literal "
                        f"occurrence(s) (parsed_hits={total_hits})"
                    )

                legacy_hits = [
                    hit
                    for hit in hits
                    if hit.get("request_id") is None or hit.get("replays") is None
                ]
                if legacy_hits:
                    errors.append(
                        "auto PM4 coherence rejects legacy/unscoped route proof "
                        f"marker(s) (legacy_hits={len(legacy_hits)}, "
                        f"total_hits={total_hits})"
                    )

                non_pm4 = [hit for hit in hits if hit.get("transport") != "pm4"]
                if non_pm4:
                    transports = sorted(
                        {
                            str(hit.get("transport"))
                            for hit in non_pm4
                            if hit.get("transport") is not None
                        }
                    )
                    errors.append(
                        "auto PM4 coherence rejects non-PM4 route proof markers "
                        f"(found transports={transports!r}, total_hits={total_hits})"
                    )

                pm4_hits = [hit for hit in hits if hit.get("transport") == "pm4"]
                seen_request_ids = set()
                for index, row in successful_rows:
                    request_id = row.get("request_id")
                    label = f"row {index}" if expected > 1 else "row"
                    if not isinstance(request_id, str) or not request_id:
                        errors.append(
                            f"auto PM4 coherence {label}: request_id must be nonempty"
                        )
                        continue
                    if request_id in seen_request_ids:
                        errors.append(
                            f"auto PM4 coherence {label}: duplicate harness "
                            f"request_id {request_id!r}"
                        )
                    seen_request_ids.add(request_id)

                    matches = [
                        hit for hit in pm4_hits if hit.get("request_id") == request_id
                    ]
                    if len(matches) != 1:
                        errors.append(
                            f"auto PM4 coherence {label}: expected exactly one PM4 "
                            f"route proof marker for request_id {request_id!r}, "
                            f"got {len(matches)}"
                        )
                        continue

                    replays = matches[0].get("replays")
                    if (
                        not isinstance(replays, int)
                        or isinstance(replays, bool)
                        or replays <= 0
                    ):
                        errors.append(
                            f"auto PM4 coherence {label}: route proof marker for "
                            f"request_id {request_id!r} requires positive replays, "
                            f"got {replays!r}"
                        )

    return {
        "required": required,
        "observed": observed,
        "transport": got_transport,
        "position": evidence.get("position"),
        "marker": evidence.get("marker"),
        "lines": list(evidence.get("lines") or []),
        "hits": list(hits),
        "occurrences": list(occurrences),
        "malformed": list(malformed),
        "literal_count": literal_count,
        "valid": not errors,
        "errors": errors,
    }


def _unique_smoke_dir(work_root: Path, label: str) -> Path:
    """Create a unique smoke work directory under work_root (PID + secure suffix)."""
    work_root.mkdir(parents=True, exist_ok=True)
    suffix = secrets.token_hex(4)
    path = Path(
        tempfile.mkdtemp(
            prefix=f"product-{label}-{os.getpid()}-",
            suffix=f"-{suffix}",
            dir=str(work_root),
        )
    )
    return path


def run_coherence_smoke(args, backend):
    """One coherence turn via scripts/serve_harness.py; quality gate only (no tok/s)."""
    if backend not in ("hip", "auto"):
        raise ValueError(f"unsupported coherence backend {backend!r}")

    # Resolve custom prompt (byte-identical for both arms) vs default Flagstaff.
    is_custom = _is_coherence_custom(args)
    if is_custom:
        info = _coherence_custom_files(args)
        if info is None:
            raise RuntimeError("coherence custom prompt file missing")
        prompt_text = info["text"]
        prompt_md5 = info["md5"]
        prompt_sha256 = info["sha256"]
        prompt_resolved = str(info["path"])
        expected = getattr(args, "coherence_expected_substring", None)
        if expected is None:
            expected = getattr(args, "coherence_expected_substrings", None)
        if expected is None:
            expected = []
        # Normalize to list of strings.
        if isinstance(expected, str):
            expected = [expected]
        expected = [str(s) for s in expected if isinstance(s, str) and s]
        thinking = getattr(args, "coherence_thinking", None) or COHERENCE_THINKING
        raw_max = getattr(args, "coherence_max_tokens", None)
        max_tokens = int(raw_max) if raw_max is not None else COHERENCE_MAX_TOKENS
        sampling = getattr(args, "coherence_sampling", None) or COHERENCE_SAMPLING
        cap = _coherence_thinking_cap(thinking)
        if cap != 0 and max_tokens <= cap:
            raise RuntimeError(
                f"coherence max_tokens ({max_tokens}) must exceed "
                f"thinking cap ({cap}) for thinking={thinking!r}"
            )
    else:
        if COHERENCE_MAX_TOKENS <= COHERENCE_THINKING_CAP_TOKENS:
            raise RuntimeError(
                f"coherence max_tokens ({COHERENCE_MAX_TOKENS}) must exceed "
                f"thinking cap ({COHERENCE_THINKING_CAP_TOKENS})"
            )
        prompt_text = COHERENCE_PROMPT
        prompt_md5 = None
        prompt_sha256 = None
        prompt_resolved = None
        expected = None
        thinking = COHERENCE_THINKING
        cap = COHERENCE_THINKING_CAP_TOKENS
        max_tokens = COHERENCE_MAX_TOKENS
        sampling = COHERENCE_SAMPLING

    model = Path(args.model).expanduser().resolve()
    daemon_src = Path(args.daemon).expanduser().resolve()
    cli = Path(getattr(args, "cli", REPO / "target/release/hipfire")).expanduser().resolve()
    work_root = Path(args.work_dir)
    smoke_dir = _unique_smoke_dir(work_root, f"{backend}-coherence")

    prompts_path = smoke_dir / "prompts.json"
    out_path = smoke_dir / "harness.json"
    home_path = smoke_dir / "home"
    serve_log = smoke_dir / "serve.log"
    # Prompts file: identical byte-identical prompt text for both HIP and auto.
    prompts_path.write_text(
        json.dumps([{"genre": "custom" if is_custom else "factual", "prompt": prompt_text}], indent=2) + "\n"
    )
    if out_path.exists():
        out_path.unlink()

    unique_daemon = _unique_coherence_daemon(daemon_src, smoke_dir, backend)
    configured_backend = backend_config_value(backend)
    # Ephemeral loopback port so concurrent CLI/serve gates never collide.
    port = _allocate_loopback_port()
    argv = [
        sys.executable,
        str(REPO / "scripts" / "serve_harness.py"),
        "--model",
        str(model),
        "--kv",
        args.kv_mode,
        "--mtp",
        COHERENCE_MTP,
        "--thinking",
        thinking,
        "--max-tokens",
        str(max_tokens),
        "--max-seq",
        str(args.max_seq),
        "--sampling",
        sampling,
        "--mode",
        COHERENCE_MODE,
        "--seed",
        str(COHERENCE_SEED),
        "--prompts-file",
        str(prompts_path),
        "--port",
        str(port),
        "--home",
        str(home_path),
        "--serve-log",
        str(serve_log),
        "--out",
        str(out_path),
        # Temporary TOML path: serve_harness writes
        # diagnostic.replay.route_proof_log=true under the isolated HIPFIRE_HOME.
        "--replay-route-proof-log",
    ]

    env = dict(os.environ)
    env.update(pm4_policy_for(args))
    env.update(
        HIPFIRE_CLI_BIN=str(cli),
        HIPFIRE_DAEMON_BIN=str(unique_daemon),
        HIPFIRE_REPLAY_BACKEND=configured_backend,
        HIPFIRE_REPLAY_TRANSPORT=args.transport,
        HIPFIRE_KV_MODE=args.kv_mode,
        HIPFIRE_CASK_OFF="1",
        HIPFIRE_AR_GRAPH="1",
        HIPFIRE_GRAPH="1",
    )
    env.pop("HIPFIRE_REPLAY_MANUAL_CAPTURE", None)
    env.pop("HIPFIRE_REPLAY_ROUTE_PROOF_LOG", None)
    env.pop("HIPFIRE_HOME", None)

    # Cross-file contract with serve_harness.py: product owns the path; harness
    # writes the CLI session leader PID after Popen(start_new_session=True).
    serve_pid_path = smoke_dir / f"serve-{backend}.pid"
    if serve_pid_path.exists():
        serve_pid_path.unlink()
    env["HIPFIRE_SERVE_HARNESS_PID_FILE"] = str(serve_pid_path)

    config = {
        "backend": backend,
        "configured_backend": configured_backend,
        "transport": args.transport,
        "thinking": thinking,
        "thinking_cap_tokens": cap,
        "max_tokens": max_tokens,
        "seed": COHERENCE_SEED,
        "sampling": sampling,
        "mode": COHERENCE_MODE,
        "mtp": COHERENCE_MTP,
        "kv_mode": args.kv_mode,
        "max_seq": args.max_seq,
        "prompt": prompt_text,
        "prompt_file": prompt_resolved,
        "prompt_md5": prompt_md5,
        "prompt_sha256": prompt_sha256,
        "expected_substrings": list(expected) if is_custom else None,
        "coherence_mode": "custom" if is_custom else "flagstaff",
        "port": port,
        "smoke_dir": str(smoke_dir),
        "home": str(home_path),
        "serve_log": str(serve_log),
        "serve_pid_file": str(serve_pid_path),
        "daemon_basename": unique_daemon.name,
        "cli": str(cli),
        # Requested via temporary serve_harness config.toml, not ambient env.
        "replay_route_proof_log": True,
        "diagnostic_replay_route_proof_log": "diagnostic.replay.route_proof_log",
    }

    started = time.monotonic()
    proc = None
    try:
        proc = subprocess.run(
            argv,
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired as error:
        cleanup_errors = _kill_serve_process_group(serve_pid_path)
        detail = f"{backend} coherence smoke timed out after {args.timeout}s"
        if cleanup_errors:
            detail = f"{detail}; cleanup failed: {'; '.join(cleanup_errors)}"
        raise RuntimeError(detail) from error
    finally:
        try:
            if unique_daemon.exists():
                unique_daemon.unlink()
        except OSError:
            pass

    assert proc is not None
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    errors = []
    rows = None
    row = None
    if proc.returncode != 0:
        errors.append(f"serve_harness exited {proc.returncode}")
    if out_path.is_file():
        try:
            rows = json.loads(out_path.read_text())
        except json.JSONDecodeError as error:
            errors.append(f"harness JSON unreadable: {error}")
    else:
        errors.append(f"missing harness output {out_path}")

    if isinstance(rows, list):
        if len(rows) != 1:
            errors.append(f"expected exactly one harness row, got {len(rows)}")
        elif rows:
            row = rows[0]
    elif rows is not None:
        errors.append(f"harness JSON must be a list, got {type(rows).__name__}")

    if row is not None:
        if not isinstance(row, dict):
            errors.append(f"harness row must be a dict, got {type(row).__name__}")
        else:
            finish = row.get("finish")
            if finish != "stop":
                errors.append(f"finish must be 'stop', got {finish!r}")
            content = row.get("assistant_content")
            if not isinstance(content, str) or not content.strip():
                errors.append("assistant_content must be a nonempty string")
            else:
                if is_custom:
                    errors.extend(_custom_coherence_answer_errors(content, expected))
                else:
                    errors.extend(_flagstaff_answer_errors(content))
            if row.get("empty"):
                errors.append("empty generation")
            if row.get("runaway"):
                errors.append("runaway generation (finish=length)")
            if row.get("attractor"):
                errors.append("attractor generation")

    raw_evidence = collect_route_proof_evidence(serve_log, stdout=stdout, stderr=stderr)
    # Bind markers only through successful harness row request_id(s).
    route_rows = [row] if isinstance(row, dict) else None
    route_evidence = validate_coherence_route_evidence(
        backend, args.transport, raw_evidence, rows=route_rows
    )
    errors.extend(route_evidence["errors"])

    report = {
        "backend": backend,
        "seconds": time.monotonic() - started,
        "valid": not errors,
        "errors": errors,
        "speed_checked": False,
        "prompt": prompt_text,
        "prompt_file": prompt_resolved,
        "prompt_md5": prompt_md5,
        "prompt_sha256": prompt_sha256,
        "expected_substrings": list(expected) if is_custom else None,
        "coherence_mode": "custom" if is_custom else "flagstaff",
        "thinking": thinking,
        "max_tokens": max_tokens,
        "sampling": sampling,
        "config": config,
        "row": row,
        "rows": rows,
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "out_path": str(out_path),
        "serve_log": str(serve_log),
        "smoke_dir": str(smoke_dir),
        "port": port,
        "daemon_basename": unique_daemon.name,
        "route_evidence": {
            "required": route_evidence["required"],
            "observed": route_evidence["observed"],
            "transport": route_evidence["transport"],
            "position": route_evidence["position"],
            "marker": route_evidence["marker"],
            "lines": route_evidence["lines"],
            "hits": route_evidence["hits"],
            "occurrences": route_evidence.get("occurrences", []),
            "malformed": route_evidence.get("malformed", []),
            "literal_count": route_evidence.get("literal_count", 0),
            "valid": route_evidence["valid"],
            "errors": list(route_evidence["errors"]),
        },
    }
    if errors:
        detail = "; ".join(errors)
        tail = (stderr or stdout)[-1500:]
        raise RuntimeError(
            f"{backend} coherence smoke failed: {detail}"
            + (f"\n{tail}" if tail else "")
        )
    return report


def load_pm4_multiturn_session(path):
    """Parse and validate session JSON before any GPU / harness work.

    Each turn must be a dict with nonempty string ``content``. ``expect`` is
    optional (setup turns may omit it), but when present must be a nonempty
    list of nonempty strings — no coercion. At least one turn in the session
    must declare expectations.
    """
    session_path = Path(path).expanduser().resolve()
    if not session_path.is_file():
        raise RuntimeError(f"multiturn session not found: {session_path}")
    try:
        turns = json.loads(session_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"multiturn session JSON unreadable: {error}") from error
    if not isinstance(turns, list) or not turns:
        raise RuntimeError("multiturn session must be a nonempty list")

    has_expectation = False
    for index, turn in enumerate(turns):
        label = f"turn {index}"
        if not isinstance(turn, dict):
            raise RuntimeError(f"{label}: must be a dict, got {type(turn).__name__}")
        content = turn.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"{label}: content must be a nonempty string")
        if "expect" not in turn:
            continue
        expect = turn["expect"]
        if not isinstance(expect, list):
            raise RuntimeError(
                f"{label}: expect must be a nonempty list of nonempty strings, "
                f"got {type(expect).__name__}"
            )
        if not expect:
            raise RuntimeError(
                f"{label}: expect must be a nonempty list of nonempty strings"
            )
        for needle_index, needle in enumerate(expect):
            if not isinstance(needle, str) or not needle:
                raise RuntimeError(
                    f"{label}: expect[{needle_index}] must be a nonempty string "
                    f"(no coercion), got {needle!r}"
                )
        has_expectation = True

    if not has_expectation:
        raise RuntimeError(
            "multiturn session must declare at least one nonempty expect list"
        )
    return session_path, turns


def run_pm4_multiturn_session(args):
    """Optional auto/redline PM4 multi-turn quality gate via serve_harness session mode.

    Runs only when requested; quality only (never tok/s). Uses the same certified
    PM4 env and PID-file process-group timeout cleanup as one-turn coherence.
    """
    if getattr(args, "transport", None) != "pm4":
        raise RuntimeError("--pm4-multiturn-session requires --transport pm4")
    if COHERENCE_MAX_TOKENS <= COHERENCE_THINKING_CAP_TOKENS:
        raise RuntimeError(
            f"coherence max_tokens ({COHERENCE_MAX_TOKENS}) must exceed "
            f"thinking cap ({COHERENCE_THINKING_CAP_TOKENS})"
        )

    session_path, turns = load_pm4_multiturn_session(args.pm4_multiturn_session)

    backend = "auto"
    model = Path(args.model).expanduser().resolve()
    daemon_src = Path(args.daemon).expanduser().resolve()
    cli = Path(getattr(args, "cli", REPO / "target/release/hipfire")).expanduser().resolve()
    work_root = Path(args.work_dir)
    smoke_dir = _unique_smoke_dir(work_root, "auto-multiturn")

    out_path = smoke_dir / "harness.json"
    home_path = smoke_dir / "home"
    serve_log = smoke_dir / "serve.log"
    if out_path.exists():
        out_path.unlink()

    unique_daemon = _unique_coherence_daemon(daemon_src, smoke_dir, "auto-multiturn")

    configured_backend = backend_config_value(backend)
    port = _allocate_loopback_port()
    argv = [
        sys.executable,
        str(REPO / "scripts" / "serve_harness.py"),
        "--model",
        str(model),
        "--kv",
        args.kv_mode,
        "--mtp",
        COHERENCE_MTP,
        "--thinking",
        COHERENCE_THINKING,
        "--max-tokens",
        str(COHERENCE_MAX_TOKENS),
        "--max-seq",
        str(args.max_seq),
        "--sampling",
        COHERENCE_SAMPLING,
        "--mode",
        "session",
        "--session",
        str(session_path),
        "--seed",
        str(COHERENCE_SEED),
        "--port",
        str(port),
        "--home",
        str(home_path),
        "--serve-log",
        str(serve_log),
        "--out",
        str(out_path),
        # Temporary TOML path: serve_harness writes
        # diagnostic.replay.route_proof_log=true under the isolated HIPFIRE_HOME.
        "--replay-route-proof-log",
    ]

    env = dict(os.environ)
    env.update(pm4_policy_for(args))
    env.update(
        HIPFIRE_CLI_BIN=str(cli),
        HIPFIRE_DAEMON_BIN=str(unique_daemon),
        HIPFIRE_REPLAY_BACKEND=configured_backend,
        HIPFIRE_REPLAY_TRANSPORT="pm4",
        HIPFIRE_KV_MODE=args.kv_mode,
        HIPFIRE_CASK_OFF="1",
        HIPFIRE_AR_GRAPH="1",
        HIPFIRE_GRAPH="1",
    )
    env.pop("HIPFIRE_REPLAY_MANUAL_CAPTURE", None)
    env.pop("HIPFIRE_REPLAY_ROUTE_PROOF_LOG", None)
    env.pop("HIPFIRE_HOME", None)

    serve_pid_path = smoke_dir / "serve-auto-multiturn.pid"
    if serve_pid_path.exists():
        serve_pid_path.unlink()
    env["HIPFIRE_SERVE_HARNESS_PID_FILE"] = str(serve_pid_path)

    config = {
        "backend": backend,
        "configured_backend": configured_backend,
        "transport": "pm4",
        "thinking": COHERENCE_THINKING,
        "thinking_cap_tokens": COHERENCE_THINKING_CAP_TOKENS,
        "max_tokens": COHERENCE_MAX_TOKENS,
        "seed": COHERENCE_SEED,
        "sampling": COHERENCE_SAMPLING,
        "mode": "session",
        "mtp": COHERENCE_MTP,
        "kv_mode": args.kv_mode,
        "max_seq": args.max_seq,
        "session": str(session_path),
        "turns": len(turns),
        "port": port,
        "smoke_dir": str(smoke_dir),
        "cli": str(cli),
        "daemon_bin": str(unique_daemon),
        "daemon_basename": unique_daemon.name,
        "serve_pid_file": str(serve_pid_path),
        "command": argv,
        # Requested via temporary serve_harness config.toml, not ambient env.
        "replay_route_proof_log": True,
        "diagnostic_replay_route_proof_log": "diagnostic.replay.route_proof_log",
    }

    started = time.monotonic()
    proc = None
    try:
        proc = subprocess.run(
            argv,
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired as error:
        cleanup_errors = _kill_serve_process_group(serve_pid_path)
        detail = f"auto multiturn session timed out after {args.timeout}s"
        if cleanup_errors:
            detail = f"{detail}; cleanup failed: {'; '.join(cleanup_errors)}"
        raise RuntimeError(detail) from error
    finally:
        try:
            if unique_daemon.exists():
                unique_daemon.unlink()
        except OSError:
            pass

    assert proc is not None
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    errors = []
    rows = None
    if proc.returncode != 0:
        errors.append(f"serve_harness exited {proc.returncode}")
    if out_path.is_file():
        try:
            rows = json.loads(out_path.read_text())
        except json.JSONDecodeError as error:
            errors.append(f"harness JSON unreadable: {error}")
    else:
        errors.append(f"missing harness output {out_path}")

    if isinstance(rows, list):
        if len(rows) != len(turns):
            errors.append(
                f"expected {len(turns)} harness row(s) (one per turn), got {len(rows)}"
            )
    elif rows is not None:
        errors.append(f"harness JSON must be a list, got {type(rows).__name__}")

    if isinstance(rows, list) and len(rows) == len(turns):
        for index, (turn, row) in enumerate(zip(turns, rows)):
            label = f"turn {index}"
            if not isinstance(row, dict):
                errors.append(f"{label}: harness row must be a dict, got {type(row).__name__}")
                continue
            finish = row.get("finish")
            if finish != "stop":
                errors.append(f"{label}: finish must be 'stop', got {finish!r}")
            content = row.get("assistant_content")
            if not isinstance(content, str) or not content.strip():
                errors.append(f"{label}: assistant_content must be a nonempty string")
                content = ""
            if row.get("empty"):
                errors.append(f"{label}: empty generation")
            if row.get("runaway"):
                errors.append(f"{label}: runaway generation (finish=length)")
            if row.get("attractor"):
                errors.append(f"{label}: attractor generation")
            expect = turn.get("expect") if isinstance(turn, dict) else None
            if expect is not None:
                # Loader already validated shape; enforce every declared needle.
                lowered = content.lower()
                for needle in expect:
                    if needle.lower() not in lowered:
                        errors.append(
                            f"{label}: answer missing expected substring {needle!r}"
                        )

    # Successful dict rows only — bind markers by nonempty request_id exactly
    # as one-turn coherence does (exact validator, full literal accounting).
    route_rows = None
    if isinstance(rows, list) and len(rows) == len(turns):
        route_rows = [row for row in rows if isinstance(row, dict)]
        if len(route_rows) != len(rows):
            route_rows = None

    raw_evidence = collect_route_proof_evidence(serve_log, stdout=stdout, stderr=stderr)
    route_evidence = validate_coherence_route_evidence(
        backend, "pm4", raw_evidence, rows=route_rows
    )
    errors.extend(route_evidence["errors"])

    report = {
        "backend": backend,
        "seconds": time.monotonic() - started,
        "valid": not errors,
        "errors": errors,
        "speed_checked": False,
        "session": str(session_path),
        "turns": len(turns),
        "config": config,
        "rows": rows,
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "out_path": str(out_path),
        "serve_log": str(serve_log),
        "smoke_dir": str(smoke_dir),
        "port": port,
        "daemon_basename": unique_daemon.name,
        "route_evidence": {
            "required": route_evidence["required"],
            "observed": route_evidence["observed"],
            "transport": route_evidence["transport"],
            "position": route_evidence["position"],
            "marker": route_evidence["marker"],
            "lines": route_evidence["lines"],
            "hits": route_evidence["hits"],
            "occurrences": route_evidence.get("occurrences", []),
            "malformed": route_evidence.get("malformed", []),
            "literal_count": route_evidence.get("literal_count", 0),
            "valid": route_evidence["valid"],
            "errors": list(route_evidence["errors"]),
        },
    }
    if errors:
        detail = "; ".join(errors)
        tail = (stderr or stdout)[-1500:]
        raise RuntimeError(
            f"auto multiturn session failed: {detail}"
            + (f"\n{tail}" if tail else "")
        )
    return report



def run_arm(args, backend):
    daemon = Daemon(
        Path(args.daemon).resolve(),
        backend,
        args.transport,
        Path(args.work_dir) / f"product-{backend}.log",
        args.timeout,
        args.kv_mode,
        args.dpm_warmup_secs,
        pm4_policy_for(args),
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
                    "dspark_mode": "off",
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
        pm4_checked = False

        def request_decode(payload):
            nonlocal pm4_checked
            row = daemon.request(payload)
            if backend == "auto" and args.transport == "pm4" and not pm4_checked:
                require_retained_pm4(row)
                pm4_checked = True
            return row

        warmup_started = time.monotonic()
        warmups = [request_decode(warmup_request) for _ in range(args.warmups)]
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
            settling_rows.append(request_decode(request))
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

        rows = [request_decode(request) for _ in range(args.runs)]
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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python3 -m tools.redline bench")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--daemon", default=str(REPO / "target/release/daemon")
    )
    parser.add_argument(
        "--cli",
        default=str(REPO / "target/release/hipfire"),
        help="native hipfire CLI used by the serve_harness coherence smoke",
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
        "--skip-coherence",
        action="store_true",
        help=(
            "skip the pre-warmup Flagstaff quality gates and go straight to the timed "
            "arms. The gate is quality-only and never contributes tok/s, but it aborts "
            "the whole run on models whose capability cannot satisfy it (e.g. the "
            "qwen3.5:0.8b reference fixture). Reports produced this way are marked "
            "coherence_skipped and can never be `valid`, so they are diagnostics only."
        ),
    )
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
    parser.add_argument(
        "--pm4-policy-override",
        action="append",
        default=[],
        metavar="HIPFIRE_REPLAY_PM4_NAME=VALUE",
        help=(
            "explicit candidate-only override layered onto the certified PM4 "
            "policy; repeat for multiple settings and inspect pm4_policy in "
            "the output report"
        ),
    )
    parser.add_argument(
        "--pm4-multiturn-session",
        default=None,
        help=(
            "optional multi-turn session JSON run after both performance arms "
            "on auto/redline PM4 via serve_harness --mode session (requires --transport pm4)"
        ),
    )
    parser.add_argument(
        "--coherence-prompt-file",
        default=None,
        help=(
            "opt-in capability-appropriate prompt file (UTF-8 text) that replaces "
            "the default Flagstaff prompt for both HIP and auto coherence smokes. "
            "Requires --coherence-expected-substring and is byte-identical for both arms; "
            "MD5/SHA256 and resolved path are recorded in the report"
        ),
    )
    parser.add_argument(
        "--coherence-expected-substring",
        action="append",
        dest="coherence_expected_substring",
        default=None,
        help=(
            "repeatable case-insensitive substring that must appear in the custom "
            "prompt's assistant answer (all values required); requires "
            "--coherence-prompt-file"
        ),
    )
    parser.add_argument(
        "--coherence-thinking",
        choices=list(COHERENCE_THINKING_BUDGET.keys()),
        default=None,
        help=(
            "thinking budget for the custom coherence prompt (default low); "
            "only valid with --coherence-prompt-file; exposed for 0.8B/LFM"
        ),
    )
    parser.add_argument(
        "--coherence-max-tokens",
        type=int,
        default=None,
        help=(
            "max_tokens for the custom coherence prompt (default 1024, must exceed "
            "thinking cap); only valid with --coherence-prompt-file"
        ),
    )
    parser.add_argument(
        "--coherence-sampling",
        default=None,
        help=(
            "serve_harness sampling spec pinned identically on both custom HIP and auto "
            "coherence smokes (default registry); only valid with --coherence-prompt-file; "
            "accepts the same nonempty specs as serve_harness --sampling "
            "(registry | registry:… | greedy | recipe:… | json:{…})"
        ),
    )
    args = parser.parse_args(argv)
    try:
        args.pm4_policy = pm4_policy_with_overrides(args.pm4_policy_override)
    except ValueError as error:
        parser.error(str(error))

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
    if args.pm4_multiturn_session is not None and args.transport != "pm4":
        parser.error("--pm4-multiturn-session requires --transport pm4")
    # Custom coherence: require paired file + expected substrings; reject partial config
    # before any GPU work and keep default Flagstaff bit-for-bit when absent.
    has_custom_file = args.coherence_prompt_file is not None
    has_custom_expected = args.coherence_expected_substring is not None
    has_custom_thinking = args.coherence_thinking is not None
    has_custom_max = args.coherence_max_tokens is not None
    has_custom_sampling = args.coherence_sampling is not None
    if (
        has_custom_file
        or has_custom_expected
        or has_custom_thinking
        or has_custom_max
        or has_custom_sampling
    ):
        if args.skip_coherence:
            parser.error("cannot combine --skip-coherence with custom coherence prompt")
        if not has_custom_file:
            parser.error(
                "--coherence-prompt-file is required when "
                "--coherence-expected-substring/--coherence-thinking/"
                "--coherence-max-tokens/--coherence-sampling is used"
            )
        if not has_custom_expected:
            parser.error(
                "--coherence-expected-substring is required when "
                "--coherence-prompt-file is used (repeatable, case-insensitive)"
            )
        for sub in args.coherence_expected_substring:
            if not isinstance(sub, str) or not sub.strip():
                parser.error(
                    "--coherence-expected-substring must be a nonempty string (no coercion)"
                )
        if args.coherence_max_tokens is not None and args.coherence_max_tokens <= 0:
            parser.error("--coherence-max-tokens must be positive")
        if has_custom_sampling and (
            not isinstance(args.coherence_sampling, str)
            or not args.coherence_sampling.strip()
        ):
            parser.error(
                "--coherence-sampling must be a nonempty serve_harness sampling spec"
            )
    # Thinking/max_tokens/sampling alone without a file is already rejected above; no extra
    # branching needed. The file existence / UTF-8 / hash validation happens after
    # model checks but before any GPU work (same order as multiturn session).

    model = Path(args.model).expanduser().resolve()
    daemon = Path(args.daemon).expanduser().resolve()
    cli = Path(args.cli).expanduser().resolve()
    if not model.is_file():
        raise SystemExit(f"model not found: {model}")
    if not daemon.is_file():
        raise SystemExit(f"daemon not found: {daemon}")
    if not cli.is_file():
        raise SystemExit(f"cli not found: {cli}")
    model_sha256 = sha256_file(model)
    if (
        args.expected_model_sha256 is not None
        and model_sha256.lower() != args.expected_model_sha256.lower()
    ):
        raise SystemExit(
            "model SHA-256 mismatch: "
            f"expected {args.expected_model_sha256.lower()}, got {model_sha256}"
        )
    # Validate custom coherence prompt file bytes/hashes before any GPU work.
    if has_custom_file:
        try:
            info = _coherence_custom_files(args)
        except Exception as error:
            raise SystemExit(f"coherence prompt file invalid before GPU work: {error}")
        # Validate thinking cap vs max_tokens before GPU.
        thinking = args.coherence_thinking or COHERENCE_THINKING
        cap = _coherence_thinking_cap(thinking)
        max_tok = (
            args.coherence_max_tokens
            if args.coherence_max_tokens is not None
            else COHERENCE_MAX_TOKENS
        )
        if cap != 0 and max_tok <= cap:
            raise SystemExit(
                f"coherence max_tokens ({max_tok}) must exceed "
                f"thinking cap ({cap}) for thinking={thinking!r} before GPU work"
            )
        # Stash resolved values for identical HIP/auto smokes and report.
        args.coherence_prompt_text = info["text"]
        args.coherence_prompt_md5 = info["md5"]
        args.coherence_prompt_sha256 = info["sha256"]
        args.coherence_prompt_resolved = str(info["path"])
        # Resolve sampling once so both arms and the report share one pin.
        if args.coherence_sampling is None:
            args.coherence_sampling = COHERENCE_SAMPLING
    # Validate optional multiturn session JSON before any GPU work.
    if args.pm4_multiturn_session is not None:
        try:
            load_pm4_multiturn_session(args.pm4_multiturn_session)
        except Exception as error:
            raise SystemExit(f"PM4 multiturn session invalid before GPU work: {error}")
    pm4_preflight = None
    if args.transport == "pm4":
        print("pm4: preflighting retained replay before benchmark warmup...", flush=True)
        try:
            pm4_preflight = run_pm4_preflight(args)
        except Exception as error:
            raise SystemExit(f"PM4 preflight failed before benchmark warmup: {error}")
        route = pm4_preflight["redline_route"] or {}
        prepared = route.get("prepared") or {}
        print(
            "pm4: preflight passed "
            f"({prepared.get('dispatches', 0)} dispatches, "
            f"{pm4_preflight['seconds']:.2f}s)",
            flush=True,
        )

    if args.skip_coherence:
        # Explicitly opted out: record the skip instead of a pass so no downstream
        # consumer can mistake this for a certified run (`valid` requires `is True`).
        skipped = {"skipped": True, "valid": None, "seconds": 0.0}
        coherence_hip = dict(skipped)
        coherence_auto = dict(skipped)
        print(
            "coherence: SKIPPED via --skip-coherence (report cannot be `valid`)",
            flush=True,
        )
    else:
        print("coherence: HIP CLI/serve Flagstaff smoke...", flush=True)
        try:
            coherence_hip = run_coherence_smoke(args, "hip")
        except Exception as error:
            raise SystemExit(f"HIP coherence smoke failed before benchmark warmup: {error}")
        print(
            f"coherence: HIP passed ({coherence_hip['seconds']:.2f}s)",
            flush=True,
        )
        print("coherence: auto CLI/serve Flagstaff smoke...", flush=True)
        try:
            coherence_auto = run_coherence_smoke(args, "auto")
        except Exception as error:
            raise SystemExit(
                f"auto coherence smoke failed before benchmark warmup: {error}"
            )
        print(
            f"coherence: auto passed ({coherence_auto['seconds']:.2f}s)",
            flush=True,
        )
        # Sampled-output parity is exact; substring/semantic checks are health hints only.
        hip_rows = coherence_hip.get("rows")
        auto_rows = coherence_auto.get("rows")
        if hip_rows is not None or auto_rows is not None:
            parity_errors = sampled_output_parity_errors(
                hip_rows,
                auto_rows,
                label="coherence",
            )
            if parity_errors:
                raise SystemExit(
                    "coherence sampled outputs differ between HIP and replay: "
                    + "; ".join(parity_errors)
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
        "cli": str(cli),
        "cli_sha256": sha256_file(cli),
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
        "pm4_policy": dict(args.pm4_policy),
        "kv_mode": args.kv_mode,
        "pm4_preflight": pm4_preflight,
        "coherence": {
            "mode": (
                "skipped"
                if args.skip_coherence
                else "custom" if has_custom_file else "flagstaff"
            ),
            "hip": coherence_hip,
            "auto": coherence_auto,
            "multiturn": None,
            "prompt_file": str(Path(args.coherence_prompt_file).expanduser().resolve())
            if has_custom_file
            else None,
            "prompt_md5": getattr(args, "coherence_prompt_md5", None)
            if has_custom_file
            else None,
            "prompt_sha256": getattr(args, "coherence_prompt_sha256", None)
            if has_custom_file
            else None,
            "expected_substrings": list(args.coherence_expected_substring)
            if has_custom_file and args.coherence_expected_substring is not None
            else None,
            "thinking": args.coherence_thinking or COHERENCE_THINKING
            if has_custom_file
            else COHERENCE_THINKING,
            "max_tokens": args.coherence_max_tokens
            if has_custom_file and args.coherence_max_tokens is not None
            else COHERENCE_MAX_TOKENS,
            "sampling": args.coherence_sampling
            if has_custom_file
            else COHERENCE_SAMPLING,
            "default_prompt": COHERENCE_PROMPT,
        },
        "hip": run_arm(args, "hip"),
        "auto": run_arm(args, "auto"),
    }
    if args.pm4_multiturn_session is not None:
        print(
            "coherence: auto/redline PM4 multiturn session via serve_harness...",
            flush=True,
        )
        try:
            report["coherence"]["multiturn"] = run_pm4_multiturn_session(args)
        except Exception as error:
            raise SystemExit(f"PM4 multiturn session failed after performance arms: {error}")
        print(
            "coherence: multiturn passed "
            f"({report['coherence']['multiturn']['seconds']:.2f}s, "
            f"{report['coherence']['multiturn']['turns']} turns)",
            flush=True,
        )
    hip = report["hip"]["tok_s"]["median"]
    auto = report["auto"]["tok_s"]["median"]
    report["speedup"] = auto / hip
    report["valid"] = (
        report["coherence"]["hip"]["valid"] is True
        and report["coherence"]["auto"]["valid"] is True
        and report["hip"]["measurement_validation"]["valid"] is True
        and report["auto"]["measurement_validation"]["valid"] is True
        and report["hip"]["route_proof"]["valid"] is True
        and report["auto"]["route_proof"]["valid"] is True
        and report["hip"]["lifecycle_route_proof"]["valid"] is True
        and report["auto"]["lifecycle_route_proof"]["valid"] is True
    )
    if report["coherence"]["multiturn"] is not None:
        report["valid"] = (
            report["valid"] is True
            and report["coherence"]["multiturn"]["valid"] is True
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
