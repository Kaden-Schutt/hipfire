# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Bridge from change_gate to the Rust hipfire-detect DetectorBank.

Python MUST NOT grow its own detector. The single source of truth for
thresholds, severity, and report shape is crates/hipfire-detect/ (attractor,
ngram, special_leak, think, toolcall, whitespace_only, eos_immediate). The
old bash coherence-gate drifted from CLAUDE.md by re-encoding those rules in
shell regex; this module only locates the binary, feeds it text/JSONL, and
returns its JSON report.

Public entry:
    analyse(text, *, jsonl=None, binary=None) -> dict
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# Repo root: tools/change_gate/detect.py -> parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]

_BIN_NAME = "hipfire-detect"


def _explicit_bin(explicit: str | None) -> tuple[str, str] | None:
    """An operator-named binary: the `binary=` arg, else HIPFIRE_DETECT_BIN.

    Returned as (path, source) so a bad one can be reported precisely.
    """
    if explicit:
        return explicit, "binary= argument"
    env = os.environ.get("HIPFIRE_DETECT_BIN")
    if env:
        return env, "HIPFIRE_DETECT_BIN"
    return None


def _discovered_bins() -> list[Path]:
    """Ordered auto-discovery list, used only when nothing was named."""
    out: list[Path] = [
        _REPO_ROOT / "target" / "release" / _BIN_NAME,
        _REPO_ROOT / "target" / "debug" / _BIN_NAME,
    ]
    which = shutil.which(_BIN_NAME)
    if which:
        out.append(Path(which))
    return out


def resolve_binary(binary: str | None = None) -> tuple[list[str] | None, str | None]:
    """Return (argv_prefix, detail).

    argv_prefix is either [path] for a built binary, or a cargo-run argv
    when nothing is built. detail explains the choice / failure.

    An **explicitly named** binary (`binary=` or ``HIPFIRE_DETECT_BIN``) is
    honoured or it fails — never silently substituted. Falling through to a
    different detector than the operator asked for would mean the report
    names one binary while the verdict came from another, which is the same
    class of dishonesty as synthesising a pass.
    """
    named = _explicit_bin(binary)
    if named is not None:
        path, source = named
        p = Path(path)
        if p.is_file() and os.access(p, os.X_OK):
            return [str(p)], f"binary={p} (from {source})"
        return None, (
            f"{source} points at {path!r} which is not an executable file; "
            "refusing to fall back to a different detector"
        )
    for p in _discovered_bins():
        if p.is_file() and os.access(p, os.X_OK):
            return [str(p)], f"binary={p}"
    # Last resort: cargo run (slow; only if cargo is on PATH).
    if shutil.which("cargo"):
        return (
            [
                "cargo",
                "run",
                "-q",
                "-p",
                "hipfire-detect",
                "--bin",
                _BIN_NAME,
                "--",
            ],
            "cargo-run-fallback",
        )
    return None, "hipfire-detect binary not found and cargo unavailable"


def _unavailable(detail: str, raw: Any = None) -> dict[str, Any]:
    return {
        "available": False,
        "verdict": "unknown",
        "findings": [],
        "raw": raw if raw is not None else {"error": detail},
        "detail": detail,
    }


def _map_verdict(report: dict[str, Any]) -> str:
    """Map Report hard_fails / soft_warns to gate verdict labels."""
    hard = int(report.get("hard_fails") or 0)
    soft = int(report.get("soft_warns") or 0)
    if hard > 0:
        return "fail"
    if soft > 0:
        return "flag"
    return "pass"


def _findings_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("rows") or []
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(row)
    return out


def analyse(
    text: str | None,
    *,
    jsonl: str | None = None,
    binary: str | None = None,
) -> dict[str, Any]:
    """Run hipfire-detect on generated text and/or daemon JSONL.

    Prefer ``jsonl`` when both are supplied (token-id detectors need it).
    Never synthesises a pass: missing/unrunnable binary → available=False,
    verdict=\"unknown\".
    """
    argv_prefix, locate_detail = resolve_binary(binary)
    if argv_prefix is None:
        return _unavailable(locate_detail or "binary not found")

    payload: str
    use_jsonl: bool
    if jsonl is not None:
        payload = jsonl
        use_jsonl = True
    elif text is not None:
        payload = text
        use_jsonl = False
    else:
        return _unavailable("analyse requires text or jsonl")

    # Write payload to a temp file so cargo-run and large stdin both work
    # without fighting subprocess pipe buffering on the cargo wrapper.
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".jsonl" if use_jsonl else ".txt",
            delete=False,
        ) as tmp:
            tmp.write(payload)
            tmp_path = tmp.name
    except OSError as exc:
        return _unavailable(f"tempfile: {exc}")

    cmd = list(argv_prefix)
    if use_jsonl:
        cmd.append("--jsonl")
    cmd.extend(["--input", tmp_path])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            timeout=600,
            check=False,
        )
    except FileNotFoundError as exc:
        return _unavailable(f"spawn failed: {exc}")
    except subprocess.TimeoutExpired:
        return _unavailable("hipfire-detect timed out")
    except OSError as exc:
        return _unavailable(f"spawn failed: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    # Exit 2 = usage/I/O error from the bin; treat as unavailable.
    if proc.returncode == 2 or (proc.returncode not in (0, 1) and not stdout):
        detail = stderr or stdout or f"exit {proc.returncode}"
        return _unavailable(
            f"hipfire-detect failed ({locate_detail}): {detail}",
            raw={"exit": proc.returncode, "stdout": stdout, "stderr": stderr},
        )

    if not stdout:
        return _unavailable(
            f"empty stdout from hipfire-detect ({locate_detail})",
            raw={"exit": proc.returncode, "stderr": stderr},
        )

    try:
        report = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return _unavailable(
            f"invalid JSON from hipfire-detect: {exc}",
            raw={"exit": proc.returncode, "stdout": stdout, "stderr": stderr},
        )

    if not isinstance(report, dict):
        return _unavailable(
            "hipfire-detect JSON was not an object",
            raw=report,
        )

    return {
        "available": True,
        "verdict": _map_verdict(report),
        "findings": _findings_from_report(report),
        "raw": report,
        "detail": locate_detail,
        "exit": proc.returncode,
    }
