# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Thin delegation: lower kernel → radiowave; lower pm4 → redline_daemon_harness --pm4."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "redline_daemon_harness.py"


def _err(msg: str) -> None:
    print(f"tools.redline.lower: {msg}", file=sys.stderr)


def _run(argv: list[str]) -> int:
    try:
        completed = subprocess.run(
            argv,
            cwd=REPO,
            env=os.environ,
            check=False,
        )
    except OSError as exc:
        _err(f"failed to start {argv!r}: {exc}")
        return 2
    return int(completed.returncode)


def resolve_radiowave(args: list[str]) -> tuple[list[str], list[str]] | int:
    """Return (prefix, forwarded_args) or exit code 2 on resolution error."""
    forwarded = list(args)
    if forwarded[:1] == ["--radiowave"]:
        if len(forwarded) < 2:
            _err("--radiowave requires PATH; attempted: --radiowave <missing>")
            return 2
        raw = forwarded[1]
        forwarded = forwarded[2:]
        # Resolve relative overrides against invocation cwd before cwd=REPO spawn.
        path = Path(raw)
        if not path.is_absolute():
            path = (Path(os.getcwd()) / path).resolve()
        else:
            path = path.resolve()
        if not path.is_file():
            _err(
                f"radiowave not found at --radiowave {raw}; "
                f"attempted: {path}"
            )
            return 2
        return [str(path)], forwarded


    release = REPO / "target" / "release" / "radiowave"
    debug = REPO / "target" / "debug" / "radiowave"
    if release.is_file():
        return [str(release)], forwarded
    if debug.is_file():
        return [str(debug)], forwarded
    return ["cargo", "run", "-q", "-p", "radiowave", "--"], forwarded


def run_kernel(args: list[str]) -> int:
    resolved = resolve_radiowave(args)
    if isinstance(resolved, int):
        return resolved
    prefix, forwarded = resolved
    argv = prefix + forwarded
    return _run(argv)


def run_pm4(args: list[str]) -> int:
    harness = REPO / "scripts" / "redline_daemon_harness.py"
    if not harness.is_file():
        _err(
            f"missing scripts/redline_daemon_harness.py; attempted: "
            f"{sys.executable} {harness} --pm4 {' '.join(args)}"
        )
        return 2
    forwarded = list(args)
    if "--pm4" in forwarded:
        # Keep caller non-flag order; enforce exactly one --pm4 after script.
        without = [a for a in forwarded if a != "--pm4"]
        child_args = ["--pm4", *without]
    else:
        child_args = ["--pm4", *forwarded]
    argv = [sys.executable, str(harness), *child_args]
    return _run(argv)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(
            "usage: python3 -m tools.redline lower {kernel|pm4} ...\n"
            "  kernel [Radiowave args...]  (optional: --radiowave PATH immediately after kernel)\n"
            "  pm4 [Redline harness args...]  (--pm4 enforced)",
            file=sys.stderr,
        )
        # bare lower / missing mode → 2; explicit -h on lower itself → still document modes.
        if args and args[0] in {"-h", "--help"}:
            return 0
        return 2

    mode, rest = args[0], args[1:]
    if mode == "kernel":
        return run_kernel(rest)
    if mode == "pm4":
        return run_pm4(rest)
    _err(f"unknown mode {mode!r} (expected kernel or pm4)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
