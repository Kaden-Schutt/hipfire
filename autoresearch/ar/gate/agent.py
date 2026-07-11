# Copyright (c) Kaden Schutt
"""Bounded, gate-local Codex executor for bespoke behavior probes.

The autoresearch loop intentionally lets an agent keep exploring until the agent
exits. A merge gate has a different contract: once a valid structured verdict
exists, further exploration only holds a scarce GPU runner. This executor stops
the whole probe process group at that boundary and fails closed on a wall timeout.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

DEFAULT_TIMEOUT_SECONDS = 900.0
DEFAULT_POLL_SECONDS = 0.25


def _valid_verdict(path: str) -> bool:
    try:
        with open(path) as fh:
            verdict = json.load(fh)
        return (
            isinstance(verdict.get("passed"), bool)
            and isinstance(verdict.get("detail"), str)
            and bool(verdict["detail"].strip())
        )
    except (OSError, ValueError, AttributeError):
        return False


def _stop_process_group(proc: subprocess.Popen, *, grace_seconds: float = 5.0) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait()
        return
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    # The Codex parent can exit before a spawned daemon. Kill any remaining
    # members even when wait() already reaped the parent.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if proc.poll() is None:
        try:
            proc.wait()
        except ChildProcessError:
            pass


def run_codex_probe(
    *,
    harness: str,
    model: str,
    effort: str,
    prompt: str,
    cwd: str,
    verdict_path: str,
    timeout_seconds: float | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
) -> int:
    """Run one Codex behavior probe and return 0 once its verdict is complete.

    The child gets its own process group so any daemon/build children are cleaned
    up with Codex. Missing/malformed verdicts remain failures in ``dispatch.py``;
    a timeout returns 124. This deliberately has no autoresearch retry/fallback
    policy: merge-gate evidence must come from the exact dispatched harness/model.
    """
    if (harness or "codex").lower() != "codex":
        raise ValueError("GPU gate behavior probes require harness='codex'")
    try:
        os.unlink(verdict_path)
    except FileNotFoundError:
        pass
    timeout = float(
        timeout_seconds
        if timeout_seconds is not None
        else os.environ.get("GATE_CODEX_TIMEOUT_SECS", DEFAULT_TIMEOUT_SECONDS)
    )
    argv = [
        "codex", "exec", "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check", "-m", model,
        "-c", f'model_reasoning_effort="{effort}"', "-C", cwd, "-",
    ]
    proc = subprocess.Popen(
        argv, cwd=cwd, stdin=subprocess.PIPE, stdout=sys.stderr, stderr=sys.stderr,
        text=True, start_new_session=True,
    )
    try:
        assert proc.stdin is not None
        proc.stdin.write(prompt)
        proc.stdin.close()
        deadline = time.monotonic() + timeout
        while True:
            if _valid_verdict(verdict_path):
                _stop_process_group(proc)
                return 0
            rc = proc.poll()
            if rc is not None:
                _stop_process_group(proc)
                return rc
            if time.monotonic() >= deadline:
                _stop_process_group(proc)
                return 124
            time.sleep(poll_seconds)
    except BaseException:
        _stop_process_group(proc)
        raise
