#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""lmx_redline_campaign — multi-process certified Redline product-point driver.

Developer-only orchestration around ``python3 -m tools.redline bench``.
Each campaign repetition is an independent fresh process with unique work,
report, and log paths. Product bench remains the certification authority;
this wrapper validates child reports and never weakens their valid/route/
coherence/measurement gates.

Report schema: lmx_redline_campaign/1. Atomic final JSON. Signal-safe
process-group cleanup only.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Schema / constants
# ---------------------------------------------------------------------------
SCHEMA = "lmx_redline_campaign/1"
SCHEMA_VERSION = "1"

REPO = Path(__file__).resolve().parent.parent

# Thinking off → product_bench COHERENCE_THINKING_BUDGET["off"] = 1.
# Coherent max_tokens must exceed that cap (product_bench rejects otherwise).
COHERENCE_THINKING = "off"
COHERENCE_THINKING_CAP = 1
COHERENCE_MAX_TOKENS = 1024
assert COHERENCE_MAX_TOKENS > COHERENCE_THINKING_CAP

# Env keys recorded as the effective allowlist (child + parent visibility).
ENV_ALLOWLIST_KEYS = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "PATH",
    "HOME",
    "USER",
    "TMPDIR",
    "TMP",
    "TEMP",
    "LANG",
    "LC_ALL",
    "PYTHONPATH",
    "PYTHONNOUSERSITE",
    "HIPFIRE_HOME",
    "HIPFIRE_DAEMON_BIN",
    "HIPFIRE_CLI_BIN",
    "HIPFIRE_KERNEL_CACHE",
    "HSA_OVERRIDE_GFX_VERSION",
    "ROCM_PATH",
    "HIP_PATH",
    "LD_LIBRARY_PATH",
)

# Global current pgid for signal cleanup — only the pgid this script started.
_current_pgid: Optional[int] = None
_current_proc: Optional[subprocess.Popen] = None
_shutdown = False


def _set_current_pgid(
    pgid: Optional[int], proc: Optional[subprocess.Popen] = None
) -> None:
    global _current_pgid, _current_proc
    _current_pgid = pgid
    _current_proc = proc


def _kill_pgid(
    pgid: Optional[int],
    proc: Optional[subprocess.Popen],
) -> None:
    if pgid is None or pgid <= 1:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pgid = None
    except PermissionError:
        pass
    if proc is not None:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    if pgid is not None:
        time.sleep(0.5)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
    if proc is not None:
        try:
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True
    pgid = _current_pgid
    if pgid is not None and pgid > 1:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
        time.sleep(1.0)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass
    sys.stderr.write(
        f"\nlmx_redline_campaign: signal {signum} — stopped pgid {pgid}\n"
    )
    sys.stderr.flush()
    raise SystemExit(128 + signum)


for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    try:
        signal.signal(_sig, _signal_handler)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _file_size_sha256(path: Path) -> Tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            size += len(chunk)
            h.update(chunk)
    return size, h.hexdigest()


def _file_md5_sha256(path: Path) -> Tuple[str, str, int]:
    h_md5 = hashlib.md5()
    h_sha = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            size += len(chunk)
            h_md5.update(chunk)
            h_sha.update(chunk)
    return h_md5.hexdigest(), h_sha.hexdigest(), size


def _git_info() -> Dict[str, Any]:
    head = None
    dirty = None
    head_short = None
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            head = r.stdout.strip()
            head_short = head[:12] if head else None
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            dirty = bool(r.stdout.strip())
        else:
            dirty = None
    except Exception:
        pass
    return {"head": head, "head_short": head_short, "dirty": dirty}


def _path_info(p: str) -> Dict[str, Any]:
    pp = Path(p).expanduser()
    if not pp.is_absolute():
        cand = REPO / p
        if cand.exists():
            pp = cand
    try:
        resolved = str(pp.resolve())
    except Exception:
        resolved = str(pp)
    info: Dict[str, Any] = {"path": str(p), "resolved": resolved}
    rp = Path(resolved)
    if rp.is_file():
        try:
            sz, sha = _file_size_sha256(rp)
            info["size"] = sz
            info["sha256"] = sha
            info["exists"] = True
        except Exception as e:
            info["exists"] = True
            info["error"] = str(e)
            info["size"] = None
            info["sha256"] = None
    else:
        info["exists"] = False
        info["size"] = None
        info["sha256"] = None
    return info


def _median(vals: Sequence[float]) -> Optional[float]:
    if not vals:
        return None
    return float(statistics.median(vals))


def _parse_env_kv(items: Optional[Sequence[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not items:
        return out
    for item in items:
        if not isinstance(item, str) or "=" not in item:
            raise ValueError(
                f"--env expects KEY=VALUE (got {item!r})"
            )
        key, _, value = item.partition("=")
        if not key or not key.isidentifier() and not all(
            c.isalnum() or c == "_" for c in key
        ):
            # Accept typical env key charset: [A-Za-z_][A-Za-z0-9_]*
            if not key or not (
                (key[0].isalpha() or key[0] == "_")
                and all(c.isalnum() or c == "_" for c in key)
            ):
                raise ValueError(f"--env invalid KEY in {item!r}")
        if key in out:
            raise ValueError(f"--env duplicate KEY {key!r}")
        out[key] = value
    return out


def _parse_device(device: str) -> str:
    """Accept a single non-negative integer physical index as a string."""
    s = str(device).strip()
    if not s:
        raise ValueError("--device must be a non-negative integer index")
    # Reject lists / ranges / empties — campaign pins one physical GPU.
    if not s.isdigit():
        raise ValueError(
            f"--device must be a single non-negative integer index (got {device!r})"
        )
    # int(s) is fine; leading zeros ok as digit string of a single index.
    return str(int(s))


def _effective_env_allowlist(env: Dict[str, str]) -> Dict[str, Optional[str]]:
    allow: Dict[str, Optional[str]] = {}
    for k in ENV_ALLOWLIST_KEYS:
        allow[k] = env.get(k)
    # Also surface any HIPFIRE_REPLAY_PM4_* present (policy identity).
    for k in sorted(env):
        if k.startswith("HIPFIRE_REPLAY_PM4_"):
            allow[k] = env[k]
    return allow


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent),
        prefix="." + path.name + ".tmp.",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=False, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass


def _deep_get(d: Any, *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _is_true(v: Any) -> bool:
    return v is True


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lmx_redline_campaign.py",
        description=(
            "Fresh-process multi-run wrapper around "
            "`python3 -m tools.redline bench`. Product bench remains the "
            "certification authority; this driver validates child reports "
            "and emits one atomic lmx_redline_campaign/1 seal report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 scripts/lmx_redline_campaign.py \\\n"
            "      --model /path/model.mq4r --prompt-file prompt.txt \\\n"
            "      --expected-substring Flagstaff --device 0 \\\n"
            "      --process-runs 3 --bench-runs 5 --out campaign.json\n"
            "\n"
            "Notes:\n"
            "  --process-runs must be >= 3 (independent fresh processes)\n"
            "  --bench-runs must be >= 5 (product_bench measurement floor)\n"
            "  Each child is `python3 -m tools.redline bench` in its own "
            "process group\n"
            "  Atomic output write; signal-safe pgid cleanup only\n"
        ),
    )
    p.add_argument("--model", required=True, help="Model file path")
    p.add_argument(
        "--prompt-file",
        required=True,
        help="UTF-8 custom coherence prompt file (passed as --coherence-prompt-file)",
    )
    p.add_argument(
        "--expected-substring",
        action="append",
        dest="expected_substring",
        default=None,
        required=True,
        help=(
            "Repeatable case-insensitive substring required in the custom "
            "coherence answer (maps to --coherence-expected-substring)"
        ),
    )
    p.add_argument(
        "--device",
        default=None,
        help=(
            "Physical GPU index pinned via HIP_VISIBLE_DEVICES and "
            "ROCR_VISIBLE_DEVICES (default: env HIP_VISIBLE_DEVICES or 0)"
        ),
    )
    p.add_argument(
        "--process-runs",
        type=int,
        default=3,
        dest="process_runs",
        help="Number of independent fresh bench processes (default/minimum: 3)",
    )
    p.add_argument(
        "--bench-runs",
        type=int,
        default=5,
        dest="bench_runs",
        help="Product-bench --runs per process (default/minimum: 5)",
    )
    p.add_argument(
        "--daemon",
        default=str(REPO / "target/release/examples/daemon"),
        help="Daemon binary path (default: target/release/examples/daemon)",
    )
    p.add_argument(
        "--cli",
        default=str(REPO / "target/release/hipfire"),
        help="hipfire CLI path (default: target/release/hipfire)",
    )
    p.add_argument(
        "--transport",
        choices=("aql", "pm4"),
        default="pm4",
        help="Redline transport (default: pm4)",
    )
    p.add_argument(
        "--kv-mode",
        choices=("q8", "fwht2", "fwht3", "fwht4"),
        default="q8",
        dest="kv_mode",
        help="KV layout (default: q8)",
    )
    p.add_argument("--context", type=int, default=128, help="Bench context tokens (default: 128)")
    p.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Bench decode iterations (default: 100)",
    )
    p.add_argument(
        "--max-seq",
        type=int,
        default=2048,
        dest="max_seq",
        help="Max sequence length (default: 2048)",
    )
    p.add_argument(
        "--work-dir",
        default=None,
        dest="work_dir",
        help="Root work directory for per-process unique work dirs "
        "(default: <repo>/.redline-work/lmx-campaign-<pid>)",
    )
    p.add_argument(
        "--log-dir",
        default=None,
        dest="log_dir",
        help="Directory for per-process wrapper logs "
        "(default: <work-dir>/logs)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Campaign report JSON path (atomically written)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Per-process wall timeout seconds (default: 3600)",
    )
    p.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Optional extra env KEY=VALUE forwarded to every child (repeatable)",
    )
    p.add_argument(
        "--pm4-policy-override",
        action="append",
        default=[],
        dest="pm4_policy_override",
        metavar="HIPFIRE_REPLAY_PM4_NAME=VALUE",
        help=(
            "Forwarded to product bench --pm4-policy-override (repeatable); "
            "must be HIPFIRE_REPLAY_PM4_<NAME>=<VALUE>"
        ),
    )
    return p


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.process_runs < 3:
        parser.error("--process-runs must be >= 3")
    if args.bench_runs < 5:
        parser.error("--bench-runs must be >= 5 (product_bench measurement floor)")
    if args.context < 1:
        parser.error("--context must be >= 1")
    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.max_seq < 512 or args.max_seq > 1048576:
        parser.error("--max-seq must be between 512 and 1048576")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")

    if args.expected_substring is None or len(args.expected_substring) < 1:
        parser.error("--expected-substring is required (repeatable)")
    for sub in args.expected_substring:
        if not isinstance(sub, str) or not sub.strip():
            parser.error(
                "--expected-substring must be a nonempty string (no coercion)"
            )

    # Device resolution + validation before any subprocess.
    if args.device is None:
        raw = os.environ.get("HIP_VISIBLE_DEVICES", "0")
        # If ambient is a list, take nothing silently — require explicit pin.
        try:
            args.device = _parse_device(raw.split(",")[0] if raw else "0")
        except ValueError as e:
            parser.error(str(e))
    else:
        try:
            args.device = _parse_device(args.device)
        except ValueError as e:
            parser.error(str(e))

    try:
        args.extra_env = _parse_env_kv(args.env)
    except ValueError as e:
        parser.error(str(e))

    # Validate PM4 override shape early (same contract as product_bench).
    forbidden = {
        "HIPFIRE_REPLAY_BACKEND",
        "HIPFIRE_REPLAY_MANUAL_CAPTURE",
        "HIPFIRE_REPLAY_TRANSPORT",
    }
    for item in args.pm4_policy_override or []:
        key, separator, value = item.partition("=")
        if (
            not separator
            or not key.startswith("HIPFIRE_REPLAY_PM4_")
            or key in forbidden
            or not value
        ):
            parser.error(
                "--pm4-policy-override expects "
                "HIPFIRE_REPLAY_PM4_<NAME>=<VALUE>; backend, transport, and "
                "manual-capture controls are not policy overrides"
            )

    prompt = Path(args.prompt_file).expanduser()
    if not prompt.is_file():
        parser.error(f"--prompt-file not found: {args.prompt_file}")

    model = Path(args.model).expanduser()
    if not model.is_file() and not (REPO / args.model).is_file():
        # Soft: still allow relative that resolves later; hard-fail missing absolute.
        if model.is_absolute():
            parser.error(f"--model not found: {args.model}")

    return args


# ---------------------------------------------------------------------------
# Child validation
# ---------------------------------------------------------------------------


def _validate_child_report(
    report: Dict[str, Any],
    *,
    expected_model_sha256: str,
    expected_daemon_sha256: str,
    expected_cli_sha256: str,
    prompt_md5: str,
    prompt_sha256: str,
    expected_substrings: List[str],
    transport: str,
    process_index: int,
) -> List[str]:
    """Validate product_bench gates without weakening them. Return errors."""
    errors: List[str] = []
    prefix = f"process[{process_index}]"

    if not _is_true(report.get("valid")):
        errors.append(f"{prefix}: report.valid is not true (got {report.get('valid')!r})")

    # Model / daemon / CLI identity stability.
    got_model = report.get("model_sha256")
    if not isinstance(got_model, str) or got_model.lower() != expected_model_sha256.lower():
        errors.append(
            f"{prefix}: model_sha256 mismatch: expected {expected_model_sha256}, got {got_model!r}"
        )
    got_daemon = report.get("daemon_sha256")
    if (
        not isinstance(got_daemon, str)
        or got_daemon.lower() != expected_daemon_sha256.lower()
    ):
        errors.append(
            f"{prefix}: daemon_sha256 mismatch: expected {expected_daemon_sha256}, got {got_daemon!r}"
        )
    got_cli = report.get("cli_sha256")
    if not isinstance(got_cli, str) or got_cli.lower() != expected_cli_sha256.lower():
        errors.append(
            f"{prefix}: cli_sha256 mismatch: expected {expected_cli_sha256}, got {got_cli!r}"
        )

    # Custom coherence hashes must match the campaign input.
    coh = report.get("coherence") if isinstance(report.get("coherence"), dict) else {}
    mode = coh.get("mode")
    if mode != "custom":
        errors.append(f"{prefix}: coherence.mode must be 'custom' (got {mode!r})")
    got_md5 = coh.get("prompt_md5")
    got_sha = coh.get("prompt_sha256")
    if got_md5 != prompt_md5:
        errors.append(
            f"{prefix}: coherence.prompt_md5 mismatch: expected {prompt_md5}, got {got_md5!r}"
        )
    if got_sha != prompt_sha256:
        errors.append(
            f"{prefix}: coherence.prompt_sha256 mismatch: expected {prompt_sha256}, got {got_sha!r}"
        )
    got_subs = coh.get("expected_substrings")
    if not isinstance(got_subs, list) or [str(s) for s in got_subs] != list(
        expected_substrings
    ):
        errors.append(
            f"{prefix}: coherence.expected_substrings mismatch: "
            f"expected {list(expected_substrings)!r}, got {got_subs!r}"
        )
    if coh.get("thinking") != COHERENCE_THINKING:
        errors.append(
            f"{prefix}: coherence.thinking must be {COHERENCE_THINKING!r} "
            f"(got {coh.get('thinking')!r})"
        )
    if coh.get("max_tokens") != COHERENCE_MAX_TOKENS:
        errors.append(
            f"{prefix}: coherence.max_tokens must be {COHERENCE_MAX_TOKENS} "
            f"(got {coh.get('max_tokens')!r})"
        )

    for arm in ("hip", "auto"):
        arm_coh = coh.get(arm) if isinstance(coh.get(arm), dict) else {}
        if not _is_true(arm_coh.get("valid")):
            errors.append(
                f"{prefix}: coherence.{arm}.valid is not true (got {arm_coh.get('valid')!r})"
            )

    # Transport identity.
    if report.get("transport") != transport:
        errors.append(
            f"{prefix}: transport mismatch: expected {transport!r}, got {report.get('transport')!r}"
        )

    # Route + measurement gates (product_bench vocabulary).
    for arm in ("hip", "auto"):
        arm_rep = report.get(arm) if isinstance(report.get(arm), dict) else {}
        mv = arm_rep.get("measurement_validation")
        if not isinstance(mv, dict) or not _is_true(mv.get("valid")):
            errors.append(
                f"{prefix}: {arm}.measurement_validation.valid is not true "
                f"(got {None if not isinstance(mv, dict) else mv.get('valid')!r})"
            )
        rp = arm_rep.get("route_proof")
        if not isinstance(rp, dict) or not _is_true(rp.get("valid")):
            errors.append(
                f"{prefix}: {arm}.route_proof.valid is not true "
                f"(got {None if not isinstance(rp, dict) else rp.get('valid')!r})"
            )
        # PM4 route gate: auto arm must show retained route when transport=pm4.
        if transport == "pm4" and arm == "auto" and isinstance(rp, dict):
            if rp.get("transport") not in (None, "pm4") and rp.get("transport") != "pm4":
                errors.append(
                    f"{prefix}: auto.route_proof.transport expected pm4, got {rp.get('transport')!r}"
                )
            # Prefer explicit fields product_bench emits.
            if "transport" in rp and rp.get("transport") != "pm4":
                errors.append(
                    f"{prefix}: auto.route_proof.transport must be 'pm4' "
                    f"(got {rp.get('transport')!r})"
                )
        lrp = arm_rep.get("lifecycle_route_proof")
        if not isinstance(lrp, dict) or not _is_true(lrp.get("valid")):
            errors.append(
                f"{prefix}: {arm}.lifecycle_route_proof.valid is not true "
                f"(got {None if not isinstance(lrp, dict) else lrp.get('valid')!r})"
            )

        # Requested/executed PM4 route presence for auto+pm4.
        if transport == "pm4" and arm == "auto":
            rows = arm_rep.get("rows")
            if isinstance(rows, list) and rows:
                for i, row in enumerate(rows):
                    if not isinstance(row, dict):
                        errors.append(f"{prefix}: auto.rows[{i}] is not an object")
                        continue
                    # Product bench rows carry redline_route / retained_replay_observed.
                    rr = row.get("redline_route")
                    if isinstance(rr, dict):
                        state = rr.get("state")
                        if state is not None and state != "ready":
                            errors.append(
                                f"{prefix}: auto.rows[{i}].redline_route.state "
                                f"expected ready, got {state!r}"
                            )
                    if "retained_replay_observed" in row and not _is_true(
                        row.get("retained_replay_observed")
                    ):
                        errors.append(
                            f"{prefix}: auto.rows[{i}].retained_replay_observed is not true"
                        )

    return errors


def _extract_medians(report: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    if not isinstance(report, dict):
        return {"hip": None, "auto": None, "speedup": None}
    hip = _deep_get(report, "hip", "tok_s", "median")
    auto = _deep_get(report, "auto", "tok_s", "median")
    speedup = report.get("speedup")
    def _f(v):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if math.isfinite(float(v)):
                return float(v)
        return None
    return {"hip": _f(hip), "auto": _f(auto), "speedup": _f(speedup)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    args = parse_args(argv)

    prompt_path = Path(args.prompt_file).expanduser().resolve()
    prompt_md5, prompt_sha256, prompt_bytes = _file_md5_sha256(prompt_path)

    model_info = _path_info(args.model)
    cli_info = _path_info(args.cli)
    daemon_info = _path_info(args.daemon)

    if not model_info.get("exists") or not model_info.get("sha256"):
        sys.stderr.write(f"lmx_redline_campaign: model not found: {args.model}\n")
        return 2
    if not cli_info.get("exists") or not cli_info.get("sha256"):
        sys.stderr.write(f"lmx_redline_campaign: cli not found: {args.cli}\n")
        return 2
    if not daemon_info.get("exists") or not daemon_info.get("sha256"):
        sys.stderr.write(f"lmx_redline_campaign: daemon not found: {args.daemon}\n")
        return 2

    model_sha = model_info["sha256"]
    cli_sha = cli_info["sha256"]
    daemon_sha = daemon_info["sha256"]
    model_resolved = model_info["resolved"]
    cli_resolved = cli_info["resolved"]
    daemon_resolved = daemon_info["resolved"]

    device_str = str(args.device)

    # Work / log roots.
    if args.work_dir:
        work_root = Path(args.work_dir).expanduser().resolve()
    else:
        work_root = (REPO / ".redline-work" / f"lmx-campaign-{os.getpid()}").resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    if args.log_dir:
        log_root = Path(args.log_dir).expanduser().resolve()
    else:
        log_root = work_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    git_info = _git_info()
    host = socket.gethostname()
    platform_info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }
    utc_now = (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    campaign_argv = list(sys.argv if argv is None else [sys.argv[0], *list(argv)])

    # Base child env: pin device visibility unless caller override is consistent.
    base_env = dict(os.environ)
    # Apply optional --env first so explicit HIP/ROCR overrides can be checked.
    for k, v in args.extra_env.items():
        base_env[k] = v

    # Device visibility: set both unless overrides are present and consistent.
    hip_override = "HIP_VISIBLE_DEVICES" in args.extra_env
    rocr_override = "ROCR_VISIBLE_DEVICES" in args.extra_env
    if hip_override or rocr_override:
        hip_v = args.extra_env.get("HIP_VISIBLE_DEVICES", device_str)
        rocr_v = args.extra_env.get("ROCR_VISIBLE_DEVICES", device_str)
        if hip_v != rocr_v:
            sys.stderr.write(
                "lmx_redline_campaign: HIP_VISIBLE_DEVICES and "
                "ROCR_VISIBLE_DEVICES overrides must match "
                f"(got HIP={hip_v!r} ROCR={rocr_v!r})\n"
            )
            return 2
        if hip_v != device_str:
            # Explicit override of device pin via --env: require it equals --device
            # after parse, or treat as inconsistent.
            try:
                pinned = _parse_device(str(hip_v).split(",")[0])
            except ValueError as e:
                sys.stderr.write(f"lmx_redline_campaign: {e}\n")
                return 2
            if pinned != device_str:
                sys.stderr.write(
                    "lmx_redline_campaign: --env device visibility "
                    f"{hip_v!r} is inconsistent with --device {device_str!r}\n"
                )
                return 2
        base_env["HIP_VISIBLE_DEVICES"] = str(hip_v)
        base_env["ROCR_VISIBLE_DEVICES"] = str(rocr_v)
    else:
        base_env["HIP_VISIBLE_DEVICES"] = device_str
        base_env["ROCR_VISIBLE_DEVICES"] = device_str

    processes: List[Dict[str, Any]] = []
    overall_errors: List[str] = []
    hip_medians: List[float] = []
    auto_medians: List[float] = []
    speedups: List[float] = []

    python_exe = sys.executable

    for proc_idx in range(args.process_runs):
        if _shutdown:
            overall_errors.append("campaign interrupted by signal")
            break

        stamp = f"{int(time.time())}_{os.getpid()}_{proc_idx}"
        proc_work = work_root / f"proc_{proc_idx}_{stamp}"
        proc_work.mkdir(parents=True, exist_ok=True)
        report_path = proc_work / "product_bench.json"
        log_path = log_root / f"proc_{proc_idx}_{stamp}.log"

        child_argv = [
            python_exe,
            "-m",
            "tools.redline",
            "bench",
            "--model",
            model_resolved,
            "--daemon",
            daemon_resolved,
            "--cli",
            cli_resolved,
            "--context",
            str(args.context),
            "--iterations",
            str(args.iterations),
            "--runs",
            str(args.bench_runs),
            "--transport",
            args.transport,
            "--kv-mode",
            args.kv_mode,
            "--max-seq",
            str(args.max_seq),
            "--work-dir",
            str(proc_work),
            "--out",
            str(report_path),
            "--expected-model-sha256",
            model_sha,
            "--coherence-prompt-file",
            str(prompt_path),
            "--coherence-thinking",
            COHERENCE_THINKING,
            "--coherence-max-tokens",
            str(COHERENCE_MAX_TOKENS),
        ]
        for sub in args.expected_substring:
            child_argv.extend(["--coherence-expected-substring", sub])
        for ov in args.pm4_policy_override or []:
            child_argv.extend(["--pm4-policy-override", ov])

        child_env = dict(base_env)
        effective_env = _effective_env_allowlist(child_env)

        proc_record: Dict[str, Any] = {
            "process_index": proc_idx,
            "work_dir": str(proc_work),
            "report_path": str(report_path),
            "log_path": str(log_path),
            "argv": list(child_argv),
            "command": list(child_argv),
            "effective_env": effective_env,
            "device_visibility": {
                "HIP_VISIBLE_DEVICES": child_env.get("HIP_VISIBLE_DEVICES"),
                "ROCR_VISIBLE_DEVICES": child_env.get("ROCR_VISIBLE_DEVICES"),
            },
            "exit_code": None,
            "timed_out": False,
            "started_at_utc": None,
            "finished_at_utc": None,
            "seconds": None,
            "report": None,
            "medians": {"hip": None, "auto": None, "speedup": None},
            "validation": {"passed": False, "errors": []},
        }

        started = time.monotonic()
        proc_record["started_at_utc"] = (
            datetime.datetime.now(datetime.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

        log_file = open(log_path, "wb")
        proc: Optional[subprocess.Popen] = None
        pgid: Optional[int] = None
        exit_code: Optional[int] = None
        timed_out = False
        try:
            proc = subprocess.Popen(
                child_argv,
                cwd=str(REPO),
                env=child_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            pgid = proc.pid
            _set_current_pgid(pgid, proc)
            try:
                exit_code = proc.wait(timeout=float(args.timeout))
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_pgid(pgid, proc)
                exit_code = proc.poll()
                if exit_code is None:
                    exit_code = -9
                overall_errors.append(
                    f"process[{proc_idx}]: timed out after {args.timeout}s"
                )
        except Exception as e:
            overall_errors.append(f"process[{proc_idx}]: launch/wait failed: {e}")
            exit_code = -1
            if pgid is not None:
                _kill_pgid(pgid, proc)
        finally:
            try:
                log_file.flush()
            except Exception:
                pass
            try:
                log_file.close()
            except Exception:
                pass
            if _current_pgid is not None and pgid is not None and _current_pgid == pgid:
                # Ensure group is gone even on clean exit (bench should exit itself).
                if proc is not None and proc.poll() is None:
                    _kill_pgid(pgid, proc)
                _set_current_pgid(None, None)

        finished = time.monotonic()
        proc_record["finished_at_utc"] = (
            datetime.datetime.now(datetime.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        proc_record["seconds"] = finished - started
        proc_record["exit_code"] = exit_code
        proc_record["timed_out"] = timed_out

        child_errors: List[str] = []
        if exit_code != 0:
            child_errors.append(
                f"process[{proc_idx}]: child exit_code {exit_code} (expected 0)"
            )

        child_report: Optional[Dict[str, Any]] = None
        if report_path.is_file():
            try:
                with open(report_path, "r", encoding="utf-8") as rf:
                    child_report = json.load(rf)
            except Exception as e:
                child_errors.append(
                    f"process[{proc_idx}]: failed to parse report {report_path}: {e}"
                )
        else:
            child_errors.append(
                f"process[{proc_idx}]: missing report at {report_path}"
            )

        # Never suppress failed artifacts — keep raw report object when present.
        proc_record["report"] = child_report
        medians = _extract_medians(child_report)
        proc_record["medians"] = medians

        if isinstance(child_report, dict):
            child_errors.extend(
                _validate_child_report(
                    child_report,
                    expected_model_sha256=model_sha,
                    expected_daemon_sha256=daemon_sha,
                    expected_cli_sha256=cli_sha,
                    prompt_md5=prompt_md5,
                    prompt_sha256=prompt_sha256,
                    expected_substrings=list(args.expected_substring),
                    transport=args.transport,
                    process_index=proc_idx,
                )
            )

        # Cross-process identity already checked per child; collect medians only
        # when the child itself claims valid and we found no validation errors.
        passed = len(child_errors) == 0
        proc_record["validation"] = {"passed": passed, "errors": list(child_errors)}
        if not passed:
            overall_errors.extend(child_errors)
        else:
            if medians["hip"] is not None:
                hip_medians.append(medians["hip"])
            if medians["auto"] is not None:
                auto_medians.append(medians["auto"])
            if medians["speedup"] is not None:
                speedups.append(medians["speedup"])

        processes.append(proc_record)

        sys.stderr.write(
            f"lmx_redline_campaign: process {proc_idx + 1}/{args.process_runs} "
            f"exit={exit_code} valid={passed} "
            f"hip={medians['hip']} auto={medians['auto']} "
            f"report={report_path}\n"
        )
        sys.stderr.flush()

    # Campaign-level identity stability across processes that produced reports.
    sha_fields = [
        ("model_sha256", model_sha),
        ("daemon_sha256", daemon_sha),
        ("cli_sha256", cli_sha),
    ]
    for field, expected in sha_fields:
        seen = set()
        for pr in processes:
            rep = pr.get("report")
            if isinstance(rep, dict) and isinstance(rep.get(field), str):
                seen.add(rep[field].lower())
        if seen and (len(seen) != 1 or next(iter(seen)) != expected.lower()):
            overall_errors.append(
                f"campaign: unstable {field} across processes: {sorted(seen)} "
                f"(expected {expected})"
            )

    # Aggregate stats over per-process medians (only those that passed gates).
    def _agg(vals: List[float]) -> Dict[str, Optional[float]]:
        if not vals:
            return {"median": None, "min": None, "max": None, "mean": None, "n": 0}
        return {
            "median": _median(vals),
            "min": float(min(vals)),
            "max": float(max(vals)),
            "mean": float(sum(vals) / len(vals)),
            "n": len(vals),
        }

    hip_agg = _agg(hip_medians)
    auto_agg = _agg(auto_medians)
    speedup_agg = _agg(speedups)
    # Campaign speedup from campaign medians when both arms present.
    campaign_speedup = None
    if hip_agg["median"] is not None and auto_agg["median"] is not None and hip_agg["median"] != 0:
        campaign_speedup = float(auto_agg["median"] / hip_agg["median"])

    # Require every process to have passed for campaign valid.
    if len(processes) != args.process_runs:
        overall_errors.append(
            f"campaign: completed {len(processes)} of {args.process_runs} process runs"
        )
    for pr in processes:
        if not pr.get("validation", {}).get("passed"):
            # already recorded
            pass

    campaign_valid = len(overall_errors) == 0 and all(
        pr.get("validation", {}).get("passed") is True for pr in processes
    ) and len(processes) == args.process_runs

    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generated_utc": utc_now,
        "host": host,
        "platform": platform_info,
        "git": git_info,
        "argv": campaign_argv,
        "command": campaign_argv,
        "identities": {
            "model": model_info,
            "cli": cli_info,
            "daemon": daemon_info,
        },
        "model": model_info,
        "cli": cli_info,
        "daemon": daemon_info,
        "prompt": {
            "path": str(prompt_path),
            "md5": prompt_md5,
            "sha256": prompt_sha256,
            "bytes": prompt_bytes,
        },
        "device": device_str,
        "device_visibility": {
            "HIP_VISIBLE_DEVICES": base_env.get("HIP_VISIBLE_DEVICES"),
            "ROCR_VISIBLE_DEVICES": base_env.get("ROCR_VISIBLE_DEVICES"),
        },
        "effective_env": _effective_env_allowlist(base_env),
        "extra_env": dict(args.extra_env),
        "pm4_policy_override": list(args.pm4_policy_override or []),
        "args": {
            "model": args.model,
            "prompt_file": str(prompt_path),
            "expected_substring": list(args.expected_substring),
            "device": device_str,
            "process_runs": args.process_runs,
            "bench_runs": args.bench_runs,
            "daemon": args.daemon,
            "cli": args.cli,
            "transport": args.transport,
            "kv_mode": args.kv_mode,
            "context": args.context,
            "iterations": args.iterations,
            "max_seq": args.max_seq,
            "work_dir": str(work_root),
            "log_dir": str(log_root),
            "out": args.out,
            "timeout": args.timeout,
            "coherence_thinking": COHERENCE_THINKING,
            "coherence_max_tokens": COHERENCE_MAX_TOKENS,
        },
        "work_dir": str(work_root),
        "log_dir": str(log_root),
        "process_runs": args.process_runs,
        "bench_runs": args.bench_runs,
        "transport": args.transport,
        "kv_mode": args.kv_mode,
        "processes": processes,
        # Raw path inventory for seal retention.
        "child_report_paths": [pr["report_path"] for pr in processes],
        "child_log_paths": [pr["log_path"] for pr in processes],
        "child_work_dirs": [pr["work_dir"] for pr in processes],
        "child_commands": [pr["argv"] for pr in processes],
        "child_effective_envs": [pr["effective_env"] for pr in processes],
        "per_process_medians": {
            "hip": [pr["medians"]["hip"] for pr in processes],
            "auto": [pr["medians"]["auto"] for pr in processes],
            "speedup": [pr["medians"]["speedup"] for pr in processes],
        },
        "hip": hip_agg,
        "auto": auto_agg,
        "speedup": {
            **speedup_agg,
            "from_campaign_medians": campaign_speedup,
        },
        "campaign_median": {
            "hip_tok_s": hip_agg["median"],
            "auto_tok_s": auto_agg["median"],
            "speedup": campaign_speedup
            if campaign_speedup is not None
            else speedup_agg["median"],
        },
        "campaign_min": {
            "hip_tok_s": hip_agg["min"],
            "auto_tok_s": auto_agg["min"],
            "speedup": speedup_agg["min"],
        },
        "campaign_max": {
            "hip_tok_s": hip_agg["max"],
            "auto_tok_s": auto_agg["max"],
            "speedup": speedup_agg["max"],
        },
        "validation": {
            "errors": overall_errors,
            "valid": campaign_valid,
        },
        "valid": campaign_valid,
        "validation_errors": overall_errors,
    }

    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    else:
        out_path = out_path.resolve()

    _atomic_write_json(out_path, report)

    if not campaign_valid:
        sys.stderr.write(
            f"lmx_redline_campaign: validation FAILED — "
            f"{len(overall_errors)} errors; report written to {out_path}\n"
        )
        for e in overall_errors[:40]:
            sys.stderr.write(f"  - {e}\n")
        sys.stderr.flush()
        return 2

    sys.stderr.write(
        f"lmx_redline_campaign: validation PASSED — report written to {out_path}\n"
    )
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        if _current_pgid is not None:
            try:
                _kill_pgid(_current_pgid, _current_proc)
            except Exception:
                pass
            _set_current_pgid(None, None)
        raise
    except BaseException:
        if _current_pgid is not None:
            try:
                _kill_pgid(_current_pgid, _current_proc)
            except Exception:
                pass
            _set_current_pgid(None, None)
        raise
