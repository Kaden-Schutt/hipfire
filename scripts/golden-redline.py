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


REPO = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO / "registry" / "redline-golden-v1.json"
PRODUCT_BENCH = REPO / "scripts" / "redline_product_bench.py"
DEFAULT_MODEL = Path("~/.hipfire/models/qwen3.6-35b-a3b.mq4r").expanduser()
DEFAULT_DAEMON = REPO / "target" / "release" / "examples" / "daemon"
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


def detect_architecture(device: int) -> str:
    if shutil.which("rocminfo") is None:
        raise GoldenError("rocminfo is required for automatic architecture detection")
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


def find_hipfire(explicit: str | None) -> Path:
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise GoldenError(f"hipfire CLI not found at {candidate}")
    local = REPO / "target" / "release" / "hipfire"
    if local.is_file():
        return local
    installed = shutil.which("hipfire")
    if installed:
        return Path(installed).resolve()
    raise GoldenError("hipfire CLI not found; build or install the native binary")


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


def ensure_daemon(daemon: Path, *, build: bool) -> None:
    if daemon.is_file():
        return
    if not build:
        raise GoldenError(f"daemon is missing: {daemon}; omit --no-build to build it")
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--example",
            "daemon",
            "-p",
            "hipfire-runtime",
        ],
        cwd=REPO,
        check=True,
    )
    if not daemon.is_file():
        raise GoldenError(f"cargo build completed but {daemon} is absent")


def product_command(
    fixture: dict[str, Any],
    golden: dict[str, Any],
    *,
    model: Path,
    daemon: Path,
    work_dir: Path,
    output: Path,
    timeout: float,
) -> list[str]:
    bench = fixture["benchmark"]
    args = [
        sys.executable,
        str(PRODUCT_BENCH),
        "--model",
        str(model),
        "--daemon",
        str(daemon),
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
    return args


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
    if (
        not isinstance(reported_max_runs, int)
        or reported_max_runs < minimum_settle_budget
    ):
        errors.append(
            f"stationarity max_runs={reported_max_runs!r} does not cover "
            f"{minimum_settle_budget} required rows"
        )
    if report.get("pm4_policy") != golden["pm4_policy"]:
        errors.append("PM4 policy differs from the sealed static policy")
    if not report.get("valid"):
        errors.append("product benchmark or route proof is invalid")

    proof = report.get("auto", {}).get("route_proof", {})
    expected_prepared = [
        route["dispatches"],
        route["packets"],
        route["phases"],
        route["command_dwords"],
    ]
    expected_sequence = [
        route["dispatches"],
        route["unique_kernels"],
        route["sequence_hash"],
    ]
    if not proof.get("valid"):
        errors.append(f"retained route proof is invalid: {proof.get('errors')}")
    if expected_prepared not in proof.get("prepared_identities", []):
        errors.append(
            f"prepared identity {proof.get('prepared_identities')} "
            f"does not contain {expected_prepared}"
        )
    if expected_sequence not in proof.get("sequences", []):
        errors.append(
            f"tape identity {proof.get('sequences')} does not contain {expected_sequence}"
        )
    observed = set(proof.get("observed_positions", []))
    if not set(route["observed_positions"]).issubset(observed):
        errors.append(
            f"observed positions {sorted(observed)} do not cover "
            f"{route['observed_positions']}"
        )
    if proof.get("retained_rows") != bench["runs"]:
        errors.append(
            f"retained rows={proof.get('retained_rows')!r}, expected {bench['runs']}"
        )

    hip_median = report.get("hip", {}).get("tok_s", {}).get("median")
    pm4_median = report.get("auto", {}).get("tok_s", {}).get("median")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command sealed Redline MQ4R reproduction"
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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

    ensure_model(
        model,
        registry,
        pull=args.pull,
        assume_yes=args.yes,
        hipfire_path=args.hipfire,
    )
    expected_model_sha = registry["model"]["sha256"]

    if args.report:
        output = Path(args.report).expanduser().resolve()
        actual_model_sha = sha256_file(model)
        if actual_model_sha != expected_model_sha:
            raise GoldenError(
                f"model SHA-256 mismatch: expected {expected_model_sha}, "
                f"got {actual_model_sha}"
            )
    else:
        ensure_daemon(daemon, build=not args.no_build)
        output.parent.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            command,
            cwd=REPO,
            env=visible_environment(args.device),
            check=True,
        )
    report = json.loads(output.read_text())
    actual_model_sha = report.get("model_sha256", expected_model_sha)
    # Older reports can still be audited when the caller supplies the exact
    # model bytes; current product reports always carry this field themselves.
    report.setdefault("model_sha256", actual_model_sha)
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
