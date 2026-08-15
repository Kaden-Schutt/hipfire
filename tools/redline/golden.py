#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Reproduce a sealed Redline MQ4R route on the selected AMD GPU.

This is developer orchestration, not a second user-facing control plane. It
drives the canonical product benchmark and delegates persistent configuration
to the native ``hipfire`` CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.redline.product_bench import sampled_output_parity_errors


REPO = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO / "registry" / "redline-golden-v1.json"
DEFAULT_MODEL = Path("~/.hipfire/models/qwen3.6-35b-a3b.mq4r").expanduser()
DEFAULT_DAEMON = REPO / "target" / "release" / "daemon"
DEFAULT_HIPFIRE = REPO / "target" / "release" / "hipfire"
ARCH_RE = re.compile(r"\bgfx(?:10|11|12)\d{2}\b")


class GoldenError(RuntimeError):
    """A fail-closed fixture or reproduction error."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(path: Path) -> dict[str, Any]:
    try:
        registry = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GoldenError(f"cannot load golden registry {path}: {exc}") from exc
    if registry.get("schema_version") != 1:
        raise GoldenError("golden registry schema_version must be 1")
    fixtures = registry.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise GoldenError("golden registry must contain at least one fixture")
    ids = [fixture.get("id") for fixture in fixtures]
    arches = [fixture.get("architecture") for fixture in fixtures]
    if any(not isinstance(item, str) or not item for item in ids):
        raise GoldenError("every golden fixture needs a non-empty id")
    if len(ids) != len(set(ids)):
        raise GoldenError("golden fixture ids must be unique")
    if len(arches) != len(set(arches)):
        raise GoldenError("golden registry must have one fixture per architecture")
    for fixture in fixtures:
        for key in ("reference", "acceptance", "benchmark", "route"):
            if not isinstance(fixture.get(key), dict):
                raise GoldenError(f"{fixture['id']}: missing {key} record")
    return registry


def validate_model_registry_card(golden: dict[str, Any]) -> None:
    registry_path = REPO / "registry" / "v1.json"
    registry = json.loads(registry_path.read_text())
    model = golden["model"]
    card = registry.get("models", {}).get(model["tag"])
    if card is None:
        raise GoldenError(f"{model['tag']} is absent from {registry_path}")
    card_hash = canonical_sha256(card)
    if card_hash != model["registry_card_sha256"]:
        raise GoldenError(
            "MQ4R registry card changed: "
            f"expected {model['registry_card_sha256']}, got {card_hash}"
        )
    sampling = card.get("sampling_profiles", {}).get(model["sampling_profile"])
    if sampling != model["sampling"]:
        raise GoldenError(
            f"registry sampling profile {model['sampling_profile']!r} drifted"
        )
    if card.get("recommended_settings") != model["sampling"]:
        raise GoldenError("native registry sampling defaults drifted")
    if canonical_sha256(sampling) != model["sampling_sha256"]:
        raise GoldenError("registry sampling profile hash drifted")
    for key in ("sha256", "size_bytes", "file"):
        if card.get(key) != model[key]:
            raise GoldenError(
                f"registry model {key} drifted: expected {model[key]!r}, "
                f"got {card.get(key)!r}"
            )


def visible_environment(device: int) -> dict[str, str]:
    env = os.environ.copy()
    # ROCr selects the requested physical device. HIP then sees that filtered
    # device as logical device zero; this is the same synchronization contract
    # used by hardware.devices in the native TOML control plane.
    env["ROCR_VISIBLE_DEVICES"] = str(device)
    env["HIP_VISIBLE_DEVICES"] = "0"
    return env


def detect_architecture_from_kfd(
    device: int,
    topology_root: Path = Path("/sys/class/kfd/kfd/topology/nodes"),
) -> str:
    try:
        nodes = sorted(
            (path for path in topology_root.iterdir() if path.name.isdigit()),
            key=lambda path: int(path.name),
        )
    except OSError as error:
        raise GoldenError(
            "automatic architecture detection requires rocminfo or readable "
            f"KFD topology: {error}"
        ) from error

    gpu_arches = []
    for node in nodes:
        try:
            properties = {
                key: value
                for line in (node / "properties").read_text().splitlines()
                if len(parts := line.split(maxsplit=1)) == 2
                for key, value in (parts,)
            }
        except OSError:
            continue
        if properties.get("cpu_cores_count") != "0":
            continue
        try:
            target = int(properties["gfx_target_version"])
        except (KeyError, ValueError):
            continue
        if target <= 0:
            continue
        major = target // 10_000
        minor = (target // 100) % 100
        stepping = target % 100
        gpu_arches.append(f"gfx{major}{minor}{stepping}")

    if device < 0 or device >= len(gpu_arches):
        raise GoldenError(
            f"physical device {device} is outside the KFD GPU inventory "
            f"({len(gpu_arches)} devices)"
        )
    return gpu_arches[device]


def detect_architecture(device: int) -> str:
    if shutil.which("rocminfo") is None:
        return detect_architecture_from_kfd(device)
    proc = subprocess.run(
        ["rocminfo"],
        env=visible_environment(device),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise GoldenError(f"rocminfo failed: {proc.stderr.strip()}")
    arches = sorted(set(ARCH_RE.findall(proc.stdout + "\n" + proc.stderr)))
    if len(arches) != 1:
        raise GoldenError(
            f"device {device} did not resolve to one supported architecture: {arches}"
        )
    return arches[0]


def select_fixture(
    registry: dict[str, Any],
    *,
    fixture_id: str | None,
    architecture: str | None,
    device: int,
) -> dict[str, Any]:
    fixtures = registry["fixtures"]
    if fixture_id is not None:
        matches = [fixture for fixture in fixtures if fixture["id"] == fixture_id]
        if not matches:
            raise GoldenError(f"unknown fixture {fixture_id!r}")
        selected = matches[0]
        if architecture is not None and selected["architecture"] != architecture:
            raise GoldenError("--fixture and --arch select different architectures")
        return selected
    detected = architecture or detect_architecture(device)
    matches = [fixture for fixture in fixtures if fixture["architecture"] == detected]
    if not matches:
        raise GoldenError(f"no golden fixture exists for {detected}")
    return matches[0]


def find_hipfire(explicit: str | None, *, required: bool = True) -> Path:
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.is_file():
            return candidate
        if required:
            raise GoldenError(f"hipfire CLI not found at {candidate}")
        return candidate
    local = REPO / "target" / "release" / "hipfire"
    if local.is_file():
        return local
    installed = shutil.which("hipfire")
    if installed:
        return Path(installed).resolve()
    if required:
        raise GoldenError("hipfire CLI not found; build or install the native binary")
    return local


def ensure_model(
    model: Path,
    golden: dict[str, Any],
    *,
    pull: bool,
    assume_yes: bool,
    hipfire_path: str | None,
) -> None:
    if model.is_file():
        return
    should_pull = pull
    if not should_pull and sys.stdin.isatty():
        answer = "y" if assume_yes else input(
            f"{model} is missing. Pull the 18.7 GB golden fixture now? [y/N] "
        )
        should_pull = answer.strip().lower() in {"y", "yes"}
    if not should_pull:
        raise GoldenError(
            f"model is missing: {model}; run "
            f"`hipfire pull {golden['model']['tag']}` or pass --pull"
        )
    cli = find_hipfire(hipfire_path)
    subprocess.run(
        [str(cli), "pull", golden["model"]["tag"]],
        cwd=REPO,
        check=True,
    )
    if not model.is_file():
        raise GoldenError(f"pull completed but {model} is still absent")


def ensure_binaries(daemon: Path, cli: Path, *, build: bool) -> None:
    """Ensure the release daemon and hipfire CLI are present.

    Builds each artifact with an explicit cargo target. ``--no-build`` fails
    clearly if either binary is absent.
    """
    missing = [path for path in (daemon, cli) if not path.is_file()]
    if not missing:
        return
    if not build:
        detail = ", ".join(str(path) for path in missing)
        raise GoldenError(
            f"required binary missing: {detail}; omit --no-build to build "
            "target/release/daemon and target/release/hipfire"
        )
    if not daemon.is_file():
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "-p",
                "hipfire-daemon",
            ],
            cwd=REPO,
            check=True,
        )
    if not cli.is_file():
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--bin",
                "hipfire",
                "-p",
                "hipfire-cli",
            ],
            cwd=REPO,
            check=True,
        )
    still_missing = [path for path in (daemon, cli) if not path.is_file()]
    if still_missing:
        detail = ", ".join(str(path) for path in still_missing)
        raise GoldenError(f"cargo build completed but still absent: {detail}")


def product_bench_argv(
    fixture: dict[str, Any],
    golden: dict[str, Any],
    *,
    model: Path,
    daemon: Path,
    cli: Path,
    work_dir: Path,
    output: Path,
    timeout: float,
) -> list[str]:
    """CLI argv for ``python3 -m tools.redline bench`` (no executable prefix)."""
    bench = fixture["benchmark"]
    return [
        "--model",
        str(model),
        "--daemon",
        str(daemon),
        "--cli",
        str(cli),
        "--context",
        str(bench["context"]),
        "--iterations",
        str(bench["iterations"]),
        "--warmups",
        str(bench["warmups"]),
        "--warmup-iterations",
        str(bench["warmup_iterations"]),
        "--runs",
        str(bench["runs"]),
        "--settle-window",
        str(bench["settle_window"]),
        "--settle-min-runs",
        str(bench["settle_min_runs"]),
        "--settle-confirmation-runs",
        str(bench["settle_confirmation_runs"]),
        "--settle-max-runs",
        str(bench["settle_max_runs"]),
        "--settle-max-slope-pct",
        str(bench["settle_max_slope_pct"]),
        "--settle-max-spread-pct",
        str(bench["settle_max_spread_pct"]),
        "--settle-max-median-drift-pct",
        str(bench["settle_max_median_drift_pct"]),
        "--transport",
        bench["transport"],
        "--kv-mode",
        bench["kv_mode"],
        "--max-seq",
        str(bench["max_seq"]),
        "--timeout",
        str(timeout),
        "--work-dir",
        str(work_dir),
        "--out",
        str(output),
        "--expected-model-sha256",
        golden["model"]["sha256"],
    ]


def product_command(
    fixture: dict[str, Any],
    golden: dict[str, Any],
    *,
    model: Path,
    daemon: Path,
    cli: Path,
    work_dir: Path,
    output: Path,
    timeout: float,
) -> list[str]:
    """Full shell-style command for dry-run display and logs."""
    return [
        sys.executable,
        "-m",
        "tools.redline",
        "bench",
        *product_bench_argv(
            fixture,
            golden,
            model=model,
            daemon=daemon,
            cli=cli,
            work_dir=work_dir,
            output=output,
            timeout=timeout,
        ),
    ]


def run_product_bench(
    fixture: dict[str, Any],
    golden: dict[str, Any],
    *,
    model: Path,
    daemon: Path,
    cli: Path,
    work_dir: Path,
    output: Path,
    timeout: float,
    env: dict[str, str],
) -> None:
    """Invoke the product benchmark in-process with a temporary environment."""
    from tools.redline import product_bench

    argv = product_bench_argv(
        fixture,
        golden,
        model=model,
        daemon=daemon,
        cli=cli,
        work_dir=work_dir,
        output=output,
        timeout=timeout,
    )
    saved = os.environ.copy()
    try:
        os.environ.clear()
        os.environ.update(env)
        # product_bench.main raises SystemExit on failure; let it propagate.
        product_bench.main(argv)
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _is_literal_true(value: Any) -> bool:
    """Certification markers must be JSON boolean true, not truthy junk."""
    return value is True


def _is_positive_int(value: Any) -> bool:
    """Reject bool (bool subclasses int) and non-positive values."""
    return type(value) is int and value > 0


def _is_non_negative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _require_mapping(value: Any, label: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{label} is missing or not an object")
        return {}
    return value



def _validate_measurement_record(
    arm: dict[str, Any],
    *,
    arm_name: str,
    errors: list[str],
) -> None:
    measurement = arm.get("measurement_validation")
    if not isinstance(measurement, dict):
        errors.append(f"{arm_name} measurement_validation is missing")
        return
    if not _is_literal_true(measurement.get("valid")):
        errors.append(f"{arm_name} measurement_validation is invalid")


def _validate_route_proof_record(
    proof: Any,
    *,
    label: str,
    expected_backend: str,
    expected_transport: str,
    require_retained: bool,
    errors: list[str],
) -> dict[str, Any]:
    if not isinstance(proof, dict):
        errors.append(f"{label} is missing")
        return {}
    if not _is_literal_true(proof.get("valid")):
        errors.append(f"{label} is invalid: {proof.get('errors')}")
    if proof.get("backend") != expected_backend:
        errors.append(
            f"{label} backend={proof.get('backend')!r}, expected {expected_backend!r}"
        )
    if proof.get("transport") != expected_transport:
        errors.append(
            f"{label} transport={proof.get('transport')!r}, "
            f"expected {expected_transport!r}"
        )
    retained_rows = proof.get("retained_rows")
    if require_retained:
        if not _is_positive_int(retained_rows):
            errors.append(
                f"{label} retained_rows={retained_rows!r}, expected positive retained rows"
            )
    elif retained_rows is not None and not (
        type(retained_rows) is int and retained_rows == 0
    ):
        errors.append(
            f"{label} retained_rows={retained_rows!r}, expected 0 for non-retained arm"
        )
    return proof


def _match_prepared_identity(
    proof: dict[str, Any],
    route: dict[str, Any],
    *,
    label: str,
    errors: list[str],
) -> None:
    """Require fixture dispatches/packets/queues/phases/command_dwords.

    Prepared identity shape is
    [dispatches, packets, queue_id, command_dwords, queues, phases].
    queue_id is a positive queue identifier and is never compared to phases.
    When the fixture records queue_id, it is compared like-for-like.
    """
    expected_dispatches = route["dispatches"]
    expected_packets = route["packets"]
    expected_command_dwords = route["command_dwords"]
    expected_queues = route["queues"]
    expected_phases = route["phases"]
    expected_queue_id = route.get("queue_id")
    prepared_identities = proof.get("prepared_identities") or []
    if not prepared_identities:
        errors.append(f"{label} prepared_identities is empty")
        return
    matched = False
    for identity in prepared_identities:
        if not isinstance(identity, (list, tuple)) or len(identity) != 6:
            errors.append(
                f"{label} prepared identity {identity!r} is not "
                "[dispatches, packets, queue_id, command_dwords, queues, phases]"
            )
            continue
        (
            dispatches,
            packets,
            queue_id,
            command_dwords,
            queues,
            phases,
        ) = identity
        if not _is_positive_int(queue_id):
            errors.append(
                f"{label} prepared identity queue_id={queue_id!r} must be a positive int"
            )
            continue
        if not _is_positive_int(queues):
            errors.append(
                f"{label} prepared identity queues={queues!r} must be a positive int"
            )
            continue
        if not _is_positive_int(phases):
            errors.append(
                f"{label} prepared identity phases={phases!r} must be a positive int"
            )
            continue
        queue_id_matches = expected_queue_id is None or (
            type(expected_queue_id) is int and queue_id == expected_queue_id
        )
        if (
            type(dispatches) is int
            and type(packets) is int
            and (command_dwords is None or type(command_dwords) is int)
            and dispatches == expected_dispatches
            and packets == expected_packets
            and command_dwords == expected_command_dwords
            and queues == expected_queues
            and phases == expected_phases
            and queue_id_matches
        ):
            matched = True
    if not matched:
        expected_queue = (
            f", queue_id={expected_queue_id}"
            if expected_queue_id is not None
            else " with a positive queue_id"
        )
        errors.append(
            f"{label} prepared identity {prepared_identities} does not contain "
            f"dispatches={expected_dispatches}, packets={expected_packets}, "
            f"queues={expected_queues}, phases={expected_phases}, "
            f"command_dwords={expected_command_dwords}{expected_queue}"
        )


def _match_tape_identity(
    proof: dict[str, Any],
    route: dict[str, Any],
    *,
    label: str,
    errors: list[str],
) -> None:
    expected_sequence = [
        route["dispatches"],
        route["unique_kernels"],
        route["sequence_hash"],
    ]
    sequences = proof.get("sequences") or []
    if expected_sequence not in sequences:
        errors.append(
            f"{label} tape identity {sequences} does not contain {expected_sequence}"
        )


def _validate_coherence_arm(
    arm: Any,
    *,
    arm_name: str,
    expected_backend: str,
    require_retained_route: bool,
    expected_transport: str,
    errors: list[str],
) -> None:
    if not isinstance(arm, dict):
        errors.append(f"coherence.{arm_name} is missing")
        return
    if not _is_literal_true(arm.get("valid")):
        errors.append(f"coherence.{arm_name} is invalid: {arm.get('errors')}")
    if arm.get("backend") != expected_backend:
        errors.append(
            f"coherence.{arm_name} backend={arm.get('backend')!r}, "
            f"expected {expected_backend!r}"
        )
    evidence = arm.get("route_evidence")
    if not isinstance(evidence, dict):
        errors.append(f"coherence.{arm_name} route_evidence is missing")
        return
    if require_retained_route:
        if not evidence.get("observed"):
            errors.append(
                f"coherence.{arm_name} did not observe retained replay route evidence"
            )
        if evidence.get("transport") != expected_transport:
            errors.append(
                f"coherence.{arm_name} route_evidence transport="
                f"{evidence.get('transport')!r}, expected {expected_transport!r}"
            )
    else:
        if evidence.get("observed"):
            errors.append(
                f"coherence.{arm_name} unexpectedly observed retained replay"
            )
        if evidence.get("marker") not in (None, ""):
            errors.append(
                f"coherence.{arm_name} unexpectedly reported route marker "
                f"{evidence.get('marker')!r}"
            )


def validate_report(
    report: dict[str, Any],
    fixture: dict[str, Any],
    golden: dict[str, Any],
    *,
    strict_binary: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    model = golden["model"]
    bench = fixture["benchmark"]
    route = fixture["route"]
    reference = fixture["reference"]
    acceptance = fixture["acceptance"]
    transport = bench["transport"]

    if report.get("model_sha256") != model["sha256"]:
        errors.append(
            f"model sha256={report.get('model_sha256')!r}, expected {model['sha256']}"
        )
    if report.get("model_bytes") != model["size_bytes"]:
        errors.append(
            f"model bytes={report.get('model_bytes')!r}, expected {model['size_bytes']}"
        )
    for report_key, fixture_key in (
        ("context", "context"),
        ("iterations", "iterations"),
        ("warmups", "warmups"),
        ("warmup_iterations", "warmup_iterations"),
        ("runs", "runs"),
        ("transport", "transport"),
        ("kv_mode", "kv_mode"),
    ):
        if report.get(report_key) != bench[fixture_key]:
            errors.append(
                f"{report_key}={report.get(report_key)!r}, "
                f"expected {bench[fixture_key]!r}"
            )
    expected_stationarity = {
        "window": bench["settle_window"],
        "min_runs": bench["settle_min_runs"],
        "confirmation_runs": bench["settle_confirmation_runs"],
        "max_slope_pct": bench["settle_max_slope_pct"],
        "max_spread_pct": bench["settle_max_spread_pct"],
        "max_median_drift_pct": bench["settle_max_median_drift_pct"],
    }
    reported_stationarity = dict(report.get("stationarity") or {})
    reported_max_runs = reported_stationarity.pop("max_runs", None)
    if reported_stationarity != expected_stationarity:
        errors.append("stationarity contract differs from the golden fixture")
    minimum_settle_budget = (
        bench["settle_min_runs"] + bench["settle_confirmation_runs"]
    )
    if not _is_non_negative_int(reported_max_runs) or reported_max_runs < minimum_settle_budget:
        errors.append(
            f"stationarity max_runs={reported_max_runs!r} does not cover "
            f"{minimum_settle_budget} required rows"
        )
    if report.get("pm4_policy") != golden["pm4_policy"]:
        errors.append("PM4 policy differs from the sealed static policy")
    if not _is_literal_true(report.get("valid")):
        errors.append("product benchmark or route proof is invalid")

    preflight = report.get("pm4_preflight")
    if transport == "pm4":
        if not isinstance(preflight, dict):
            errors.append("pm4_preflight is missing")
        else:
            preflight_proof = _validate_route_proof_record(
                preflight.get("route_proof"),
                label="pm4_preflight route_proof",
                expected_backend="auto",
                expected_transport="pm4",
                require_retained=True,
                errors=errors,
            )
            if preflight_proof:
                _match_prepared_identity(
                    preflight_proof,
                    route,
                    label="pm4_preflight route_proof",
                    errors=errors,
                )
                _match_tape_identity(
                    preflight_proof,
                    route,
                    label="pm4_preflight route_proof",
                    errors=errors,
                )

    coherence = _require_mapping(report.get("coherence"), "coherence", errors)
    _validate_coherence_arm(
        coherence.get("hip"),
        arm_name="hip",
        expected_backend="hip",
        require_retained_route=False,
        expected_transport=transport,
        errors=errors,
    )
    _validate_coherence_arm(
        coherence.get("auto"),
        arm_name="auto",
        expected_backend="auto",
        require_retained_route=transport == "pm4",
        expected_transport=transport,
        errors=errors,
    )
    # Exact sampled-output parity for the one-turn coherence arms; substring/semantic checks are health hints only.
    if isinstance(coherence, dict):
        hip_coh = coherence.get("hip")
        auto_coh = coherence.get("auto")
        if isinstance(hip_coh, dict) and isinstance(auto_coh, dict):
            if not hip_coh.get("skipped") and not auto_coh.get("skipped"):
                hip_rows = hip_coh.get("rows")
                auto_rows = auto_coh.get("rows")
                # Legacy reports without rows (pre-parity) are back-compat; enforce when either side carries rows.
                if hip_rows is not None or auto_rows is not None:
                    parity_errors = sampled_output_parity_errors(hip_rows, auto_rows, label="coherence")
                    errors.extend(parity_errors)

    hip_arm = _require_mapping(report.get("hip"), "hip", errors)
    auto_arm = _require_mapping(report.get("auto"), "auto", errors)
    _validate_measurement_record(hip_arm, arm_name="hip", errors=errors)
    _validate_measurement_record(auto_arm, arm_name="auto", errors=errors)

    hip_lifecycle = _validate_route_proof_record(
        hip_arm.get("lifecycle_route_proof"),
        label="hip lifecycle_route_proof",
        expected_backend="hip",
        expected_transport=transport,
        require_retained=False,
        errors=errors,
    )
    auto_lifecycle = _validate_route_proof_record(
        auto_arm.get("lifecycle_route_proof"),
        label="auto lifecycle_route_proof",
        expected_backend="auto",
        expected_transport=transport,
        require_retained=True,
        errors=errors,
    )
    _validate_route_proof_record(
        hip_arm.get("route_proof"),
        label="hip route_proof",
        expected_backend="hip",
        expected_transport=transport,
        require_retained=False,
        errors=errors,
    )
    auto_timed = _validate_route_proof_record(
        auto_arm.get("route_proof"),
        label="auto route_proof",
        expected_backend="auto",
        expected_transport=transport,
        require_retained=True,
        errors=errors,
    )

    # Prepared identity: dispatches + packets + queue_id + command_dwords +
    # queues + phases. queue_id is never compared to fixture route.phases;
    # queues/phases are compared like-for-like with the fixture route.
    if auto_timed:
        _match_prepared_identity(
            auto_timed,
            route,
            label="auto route_proof",
            errors=errors,
        )
        _match_tape_identity(
            auto_timed,
            route,
            label="auto route_proof",
            errors=errors,
        )
    observed = set(auto_timed.get("observed_positions") or [])
    if not set(route["observed_positions"]).issubset(observed):
        errors.append(
            f"observed positions {sorted(observed)} do not cover "
            f"{route['observed_positions']}"
        )
    if auto_timed.get("retained_rows") != bench["runs"]:
        errors.append(
            f"auto timed retained rows={auto_timed.get('retained_rows')!r}, "
            f"expected {bench['runs']}"
        )
    if auto_lifecycle and type(auto_lifecycle.get("retained_rows")) is int:
        if auto_lifecycle["retained_rows"] <= 0:
            errors.append("auto lifecycle_route_proof observed no retained rows")

    hip_median = hip_arm.get("tok_s", {}).get("median") if hip_arm else None
    pm4_median = auto_arm.get("tok_s", {}).get("median") if auto_arm else None
    speedup = report.get("speedup")
    if not isinstance(pm4_median, (int, float)):
        errors.append("PM4 median is missing")
    elif pm4_median < acceptance["minimum_pm4_tok_s"]:
        errors.append(
            f"PM4 median {pm4_median:.3f} < "
            f"{acceptance['minimum_pm4_tok_s']:.3f} tok/s"
        )
    if not isinstance(speedup, (int, float)):
        errors.append("speedup is missing")
    elif speedup < acceptance["minimum_speedup"]:
        errors.append(
            f"speedup {speedup:.5f} < {acceptance['minimum_speedup']:.5f}"
        )

    if report.get("git_commit") != reference["source_commit"]:
        warnings.append(
            f"source commit {report.get('git_commit')} differs from "
            f"reference {reference['source_commit']}"
        )
    if report.get("daemon_sha256") != reference["daemon_sha256"]:
        warnings.append(
            f"daemon sha256 {report.get('daemon_sha256')} differs from "
            f"reference {reference['daemon_sha256']}"
        )
    if strict_binary:
        errors.extend(warnings)
        warnings = []

    exact_reference_binary = (
        report.get("git_commit") == reference["source_commit"]
        and report.get("daemon_sha256") == reference["daemon_sha256"]
    )
    return {
        "valid": not errors,
        "classification": (
            "exact-reference-binary"
            if exact_reference_binary and not errors
            else "route-compatible-reproduction"
            if not errors
            else "failed"
        ),
        "fixture_id": fixture["id"],
        "architecture": fixture["architecture"],
        "hip_median_tok_s": hip_median,
        "pm4_median_tok_s": pm4_median,
        "speedup": speedup,
        "errors": errors,
        "warnings": warnings,
    }




def configure_default(
    golden: dict[str, Any],
    *,
    hipfire_path: str | None,
) -> None:
    cli = find_hipfire(hipfire_path)
    tag = golden["model"]["tag"]
    # A global generation override intentionally wins over registry defaults.
    # Pin the validated profile as a per-model layer so choosing the golden
    # default is deterministic without deleting unrelated global preferences.
    values = dict(golden["model"]["sampling"])
    values["kv_cache"] = "q8"
    for key, value in values.items():
        subprocess.run(
            [str(cli), "config", tag, "set", key, str(value)],
            cwd=REPO,
            check=True,
        )
    subprocess.run(
        [str(cli), "config", "set", "serve.default_model", tag],
        cwd=REPO,
        check=True,
    )
    print(
        f"{tag} is now the serve default with its pinned registry sampling "
        "profile and Q8 KV."
    )


def print_fixtures(registry: dict[str, Any]) -> None:
    print("Golden Redline fixtures:")
    for fixture in registry["fixtures"]:
        reference = fixture["reference"]
        acceptance = fixture["acceptance"]
        print(
            f"  {fixture['id']}\n"
            f"    arch={fixture['architecture']} "
            f"reference={reference['pm4_median_tok_s']:.3f} tok/s "
            f"floor={acceptance['minimum_pm4_tok_s']:.3f} tok/s"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 -m tools.redline golden",
        description="One-command sealed Redline MQ4R reproduction",
    )
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--fixture")
    parser.add_argument("--arch")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--daemon", default=str(DEFAULT_DAEMON))
    parser.add_argument("--hipfire")
    parser.add_argument("--report", help="validate an existing product report")
    parser.add_argument("--out")
    parser.add_argument("--work-dir")
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--strict-binary", action="store_true")
    parser.add_argument("--set-default", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--validate-registry", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registry_path = Path(args.registry).expanduser().resolve()
    registry = load_registry(registry_path)
    validate_model_registry_card(registry)
    if args.list:
        print_fixtures(registry)
        return 0
    if args.validate_registry:
        print(f"valid golden registry: {registry_path}")
        return 0

    fixture = select_fixture(
        registry,
        fixture_id=args.fixture,
        architecture=args.arch,
        device=args.device,
    )
    model = Path(args.model).expanduser().resolve()
    daemon = Path(args.daemon).expanduser().resolve()
    # Identify the intended release CLI path without requiring a pre-built binary.
    # Report-only never needs the CLI; live runs build/ensure then require it.
    cli = find_hipfire(args.hipfire, required=False)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        Path(args.out).expanduser().resolve()
        if args.out
        else REPO
        / ".redline-work"
        / "golden"
        / f"{fixture['id']}-{stamp}.json"
    )
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else output.parent / f"{output.stem}-work"
    )
    command = product_command(
        fixture,
        registry,
        model=model,
        daemon=daemon,
        cli=cli,
        work_dir=work_dir,
        output=output,
        timeout=args.timeout,
    )
    print(f"fixture: {fixture['id']}")
    print(f"device: physical {args.device} -> ROCr {args.device}, HIP logical 0")
    print("command:", " ".join(command))
    if args.dry_run:
        return 0
    detected_arch = detect_architecture(args.device)
    if detected_arch != fixture["architecture"]:
        raise GoldenError(
            f"device {args.device} is {detected_arch}, but the selected fixture "
            f"is for {fixture['architecture']}"
        )

    expected_model_sha = registry["model"]["sha256"]

    if args.report:
        output = Path(args.report).expanduser().resolve()
        # Report-only validation must not require or build a hipfire CLI.
        ensure_model(
            model,
            registry,
            pull=args.pull,
            assume_yes=args.yes,
            hipfire_path=args.hipfire,
        )
        actual_model_sha = sha256_file(model)
        if actual_model_sha != expected_model_sha:
            raise GoldenError(
                f"model SHA-256 mismatch: expected {expected_model_sha}, "
                f"got {actual_model_sha}"
            )
    else:
        ensure_binaries(daemon, cli, build=not args.no_build)
        # After build/ensure, require a real CLI (local release or installed).
        cli = find_hipfire(args.hipfire, required=True)
        ensure_model(
            model,
            registry,
            pull=args.pull,
            assume_yes=args.yes,
            hipfire_path=args.hipfire,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        run_product_bench(
            fixture,
            registry,
            model=model,
            daemon=daemon,
            cli=cli,
            work_dir=work_dir,
            output=output,
            timeout=args.timeout,
            env=visible_environment(args.device),
        )
    report = json.loads(output.read_text())
    if "model_sha256" not in report:
        raise GoldenError(
            "product report is missing model_sha256; historical reports are "
            "registry evidence only and cannot be certified via --report"
        )
    validation = validate_report(
        report,
        fixture,
        registry,
        strict_binary=args.strict_binary,
    )
    attestation = {
        "schema": "hipfire.redline.golden-reproduction.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_registry": str(registry_path),
        "fixture_registry_sha256": sha256_file(registry_path),
        "fixture": fixture,
        "product_report": str(output),
        "product_report_sha256": sha256_file(output),
        "validation": validation,
    }
    attestation_path = output.with_suffix(".golden.json")
    attestation_path.write_text(json.dumps(attestation, indent=2) + "\n")

    print(
        f"result: {validation['classification']} "
        f"HIP={validation['hip_median_tok_s']:.3f} "
        f"PM4={validation['pm4_median_tok_s']:.3f} "
        f"speedup={validation['speedup']:.5f}"
    )
    for warning in validation["warnings"]:
        print(f"warning: {warning}")
    if not validation["valid"]:
        for error in validation["errors"]:
            print(f"error: {error}", file=sys.stderr)
        print(f"attestation: {attestation_path}", file=sys.stderr)
        return 1
    print(f"attestation: {attestation_path}")

    configure = args.set_default
    if not configure and not args.no_prompt and sys.stdin.isatty():
        answer = "y" if args.yes else input(
            "Set this model as the hipfire default with its pinned registry "
            "sampling profile and Q8 KV? [y/N] "
        )
        configure = answer.strip().lower() in {"y", "yes"}
    if configure:
        configure_default(registry, hipfire_path=args.hipfire)
    print(
        "OpenAI endpoint: http://127.0.0.1:11435/v1 "
        f"(model {registry['model']['tag']}); see docs/GOLDEN-REDLINE.md"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GoldenError as exc:
        print(f"golden-redline: {exc}", file=sys.stderr)
        raise SystemExit(2)
