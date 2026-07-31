#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Deterministic GPU fault-injection parity harness for retry-eligible routes.

Compares a clean greedy generate against a fault-after-prefill → production
rollback → reset → retry path on independent fresh daemon processes.

Requires a daemon built with feature ``serve-fault-inject`` for the fault arm
and ``test_state_snapshot`` command. Without that feature the fault field is
ignored and the snapshot command is unsupported.

Protocol is stdout JSONL only; stderr is never parsed as protocol.

Parity compares model/runtime state categories (tokens, KV/recurrent
hashes+bytes, graph/replay/drafter/adaptive/cache flags, seq/conversation).
``state_epoch`` is validated for presence and reset-ack monotonicity on the
fault process only — it is NOT required to match across clean vs faulted
processes (fault path performs an explicit reset; clean does not).
"""

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parent.parent

# Frozen Task-14 promotion matrix: only qwen35 local GPU fixtures are
# retry-eligible today. Snapshot eligible_routes must match exactly.
EXPECTED_ELIGIBLE_ROUTES = ["qwen_ar", "qwen_dflash"]

# Events that must not appear before the injected transient error.
FORBIDDEN_PRE_ERROR = frozenset(
    {"token", "reasoning", "tool_calls", "commit_ready", "done"}
)

# Snapshot fields compared for clean-vs-retry parity after success. Full KV
# allocation hashes remain diagnostic-only: identical clean fresh-process runs
# have different hashes because kernels leave inactive/padding bytes undefined.
# Generated output plus recurrent state and logical lengths are the parity gate.
# state_epoch is intentionally excluded (see module docstring).
SNAPSHOT_COMPARE_KEYS = (
    "seq_pos",
    "conversation_len",
    "kv_bytes",
    "recurrent_hash",
    "recurrent_bytes",
    "graph_clean",
    "replay_clean",
    "drafter_reset",
    "checkpoint_empty",
    "adaptive_clean",
    "asst_cache_empty",
    "prefix_cache_clean",
)

# Post-rollback / pre-retry snapshot must be cold (graph/replay clean after
# attested rollback). Missing cleanliness fails closed.
COLD_SNAPSHOT_EXPECT = {
    "seq_pos": 0,
    "conversation_len": 0,
    "graph_clean": True,
    "replay_clean": True,
    "drafter_reset": True,
    "checkpoint_empty": True,
    "adaptive_clean": True,
    "asst_cache_empty": True,
    "prefix_cache_clean": True,
}

FAULT_ERROR_MESSAGE = "injected fault after prefill"
FAULT_ERROR_CLASS = "gpu"


class HarnessError(RuntimeError):
    """Fail-closed harness failure."""


class DaemonSession:
    """One fresh daemon process; stdout JSONL only (stderr drained, not protocol)."""

    def __init__(
        self,
        binary: Path,
        *,
        timeout_s: float,
        env: Optional[Dict[str, str]] = None,
        log_path: Optional[Path] = None,
    ) -> None:
        self.timeout_s = timeout_s
        self.binary = Path(binary)
        if not self.binary.is_file():
            raise HarnessError(f"daemon binary not found: {self.binary}")

        self._stderr_fh = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._stderr_fh = log_path.open("wb")

        run_env = dict(os.environ)
        if env:
            run_env.update(env)
        # Graph paths must stay enabled so the harness can prove the
        # graph-enabled route was exercised before rollback (not an
        # always-clean graph-off run). Cask stays off for determinism.
        run_env["HIPFIRE_AR_GRAPH"] = "1"
        run_env["HIPFIRE_GRAPH"] = "1"
        run_env.setdefault("HIPFIRE_CASK_OFF", "1")

        self.proc = subprocess.Popen(
            [str(self.binary)],
            cwd=str(REPO),
            env=run_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_fh if self._stderr_fh is not None else subprocess.PIPE,
            text=False,
            bufsize=0,
            start_new_session=True,
        )
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self._stdout = self.proc.stdout
        self._stdin = self.proc.stdin
        if self._stderr_fh is None and self.proc.stderr is not None:

            def _drain() -> None:
                try:
                    while True:
                        chunk = self.proc.stderr.read(4096)  # type: ignore[union-attr]
                        if not chunk:
                            break
                except Exception:
                    pass

            threading.Thread(target=_drain, daemon=True).start()

        # Per-connection epoch tracker for fault-process monotonic checks only.
        self.last_state_epoch: Optional[int] = None

    def _ensure_alive(self, context: str) -> None:
        code = self.proc.poll()
        if code is not None:
            raise HarnessError(f"daemon exited early (code={code}) during {context}")

    def send(self, message: Dict[str, Any]) -> None:
        self._ensure_alive(f"send {message.get('type')}")
        line = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        self._stdin.write(line + b"\n")
        self._stdin.flush()

    def recv_line(
        self, *, timeout_s: Optional[float] = None, context: str = "recv"
    ) -> Dict[str, Any]:
        """Read one JSONL object from stdout with timeout. Skips blank lines."""
        deadline = time.monotonic() + (
            self.timeout_s if timeout_s is None else timeout_s
        )
        buf = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HarnessError(f"timeout after waiting for stdout during {context}")
            self._ensure_alive(context)
            ready, _, _ = select.select([self._stdout], [], [], min(remaining, 0.5))
            if not ready:
                continue
            ch = self._stdout.read(1)
            if not ch:
                self._ensure_alive(context)
                raise HarnessError(f"daemon closed stdout during {context}")
            if ch == b"\n":
                if not buf:
                    continue
                raw = bytes(buf).decode("utf-8", errors="replace").strip()
                buf.clear()
                if not raw:
                    continue
                try:
                    return json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise HarnessError(
                        f"non-JSON stdout line during {context}: {raw[:200]!r} ({exc})"
                    ) from exc
            buf.extend(ch)
            if len(buf) > 16 * 1024 * 1024:
                raise HarnessError(f"stdout line exceeded 16MiB during {context}")

    def request_until(
        self,
        message: Dict[str, Any],
        terminal_types: Sequence[str],
        *,
        context: str,
        timeout_s: Optional[float] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        auto_commit: bool = True,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Send message and collect events until a terminal type is seen."""
        self.send(message)
        events: List[Dict[str, Any]] = []
        terminals = set(terminal_types)
        req_id = message.get("id")
        attempt_id = message.get("attempt_id")
        deadline = time.monotonic() + (
            self.timeout_s if timeout_s is None else timeout_s
        )

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HarnessError(f"timeout collecting events during {context}")
            ev = self.recv_line(timeout_s=remaining, context=context)
            events.append(ev)
            if on_event is not None:
                on_event(ev)

            ty = ev.get("type")
            if ty == "commit_ready" and auto_commit:
                if req_id is None or attempt_id is None:
                    raise HarnessError(
                        f"commit_ready without request id/attempt during {context}: {ev}"
                    )
                self.send(
                    {
                        "type": "commit",
                        "id": req_id,
                        "attempt_id": attempt_id,
                    }
                )
                continue

            if ty in terminals:
                return ev, events

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send({"type": "unload"})
                try:
                    self.recv_line(timeout_s=5.0, context="unload")
                except Exception:
                    pass
            except Exception:
                pass
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except Exception:
                try:
                    self.proc.terminate()
                except Exception:
                    pass
            try:
                self.proc.wait(timeout=8)
            except Exception:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
                try:
                    self.proc.wait(timeout=3)
                except Exception:
                    pass
        if self._stderr_fh is not None:
            try:
                self._stderr_fh.close()
            except Exception:
                pass


def load_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict) and isinstance(obj.get("prompt"), str):
                return obj["prompt"]
        except json.JSONDecodeError:
            pass
    if not text.strip():
        raise HarnessError(f"empty prompt file: {path}")
    return text


def assert_eligible_routes(routes: Any, where: str) -> None:
    if routes != EXPECTED_ELIGIBLE_ROUTES:
        raise HarnessError(
            f"{where}: eligible_routes must equal {EXPECTED_ELIGIBLE_ROUTES!r}, "
            f"got {routes!r}"
        )


def assert_retry_eligible_true(value: Any, where: str) -> None:
    if value is not True:
        raise HarnessError(f"{where}: retry_reset_eligible must be true, got {value!r}")


def validate_reset_ack(
    session: DaemonSession, ack: Dict[str, Any], attempt_id: int, where: str
) -> None:
    if ack.get("type") != "reset":
        raise HarnessError(f"{where}: expected type=reset, got {ack!r}")
    if ack.get("rolled_back") is not True:
        raise HarnessError(f"{where}: reset ack requires rolled_back=true, got {ack!r}")
    if ack.get("seq_pos") != 0:
        raise HarnessError(f"{where}: reset ack requires seq_pos=0, got {ack!r}")
    if ack.get("conversation_len") != 0:
        raise HarnessError(
            f"{where}: reset ack requires conversation_len=0, got {ack!r}"
        )
    if ack.get("attempt_id") != attempt_id:
        raise HarnessError(
            f"{where}: reset ack attempt_id mismatch expected={attempt_id} "
            f"got={ack.get('attempt_id')!r}"
        )
    epoch = ack.get("state_epoch")
    if type(epoch) is not int:
        raise HarnessError(f"{where}: reset ack missing/invalid state_epoch: {ack!r}")
    if session.last_state_epoch is not None and epoch <= session.last_state_epoch:
        raise HarnessError(
            f"{where}: reset ack state_epoch not strictly increasing: "
            f"prev={session.last_state_epoch} got={epoch}"
        )
    session.last_state_epoch = int(epoch)
    assert_retry_eligible_true(ack.get("retry_reset_eligible"), f"{where} reset ack")


def validate_snapshot_schema(snap: Dict[str, Any], where: str) -> None:
    if snap.get("type") != "test_state_snapshot":
        raise HarnessError(
            f"{where}: expected type=test_state_snapshot, got {snap.get('type')!r}"
        )
    if snap.get("schema_version") != 1:
        raise HarnessError(
            f"{where}: schema_version must be 1, got {snap.get('schema_version')!r}"
        )
    assert_eligible_routes(snap.get("eligible_routes"), where)
    for key in (
        "arch",
        "state_epoch",
        "seq_pos",
        "conversation_len",
        "kv_hash",
        "kv_bytes",
        "recurrent_hash",
        "recurrent_bytes",
        "graph_clean",
        "replay_clean",
        "drafter_reset",
        "checkpoint_empty",
        "adaptive_clean",
        "asst_cache_empty",
        "prefix_cache_clean",
    ):
        if key not in snap:
            raise HarnessError(f"{where}: snapshot missing field {key!r}")
    if type(snap.get("state_epoch")) is not int:
        raise HarnessError(
            f"{where}: snapshot state_epoch must be int, got {snap.get('state_epoch')!r}"
        )


def assert_cold_snapshot(snap: Dict[str, Any], where: str) -> List[str]:
    """Return mismatch strings (empty if cold)."""
    mismatches: List[str] = []
    for key, expected in COLD_SNAPSHOT_EXPECT.items():
        got = snap.get(key)
        if got != expected:
            mismatches.append(f"{where}.{key}: expected {expected!r}, got {got!r}")
    return mismatches


def load_model(
    session: DaemonSession,
    *,
    target: Path,
    draft: Optional[Path],
    max_seq: int,
    context: str,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"max_seq": max_seq}
    if draft is not None:
        params["draft"] = str(draft)
        params["dflash_mode"] = "on"
    else:
        params["dflash_mode"] = "off"
    session.send({"type": "load", "model": str(target), "params": params})
    deadline = time.monotonic() + session.timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise HarnessError(f"timeout waiting for loaded during {context}")
        ev = session.recv_line(timeout_s=remaining, context=context)
        ty = ev.get("type")
        if ty == "error":
            raise HarnessError(f"{context}: load error: {ev}")
        if ty == "loaded":
            assert_retry_eligible_true(
                ev.get("retry_reset_eligible"), f"{context} loaded"
            )
            return ev


def snapshot(session: DaemonSession, *, context: str) -> Dict[str, Any]:
    term, _events = session.request_until(
        {"type": "test_state_snapshot"},
        terminal_types=("test_state_snapshot", "error"),
        context=context,
        auto_commit=False,
    )
    if term.get("type") == "error":
        raise HarnessError(
            f"{context}: test_state_snapshot unsupported or failed: {term} "
            f"(daemon must be built with feature serve-fault-inject)"
        )
    validate_snapshot_schema(term, context)
    return term


def generate_clean(
    session: DaemonSession,
    *,
    req_id: str,
    attempt_id: int,
    prompt: str,
    max_tokens: int,
    context: str,
) -> Dict[str, Any]:
    req = {
        "type": "generate",
        "id": req_id,
        "attempt_id": attempt_id,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "max_think_tokens": 1,
        "assistant_prefix": "closed_think",
    }
    token_text: List[str] = []
    reasoning_text: List[str] = []
    saw_tool = False

    def on_event(ev: Dict[str, Any]) -> None:
        nonlocal saw_tool
        ty = ev.get("type")
        if ty == "token":
            token_text.append(str(ev.get("text") or ""))
        elif ty == "reasoning":
            reasoning_text.append(str(ev.get("text") or ""))
        elif ty == "tool_calls":
            saw_tool = True

    term, events = session.request_until(
        req,
        terminal_types=("done", "error"),
        context=context,
        on_event=on_event,
        auto_commit=True,
    )
    if term.get("type") == "error":
        raise HarnessError(f"{context}: unexpected error on clean generate: {term}")
    if term.get("type") != "done":
        raise HarnessError(f"{context}: expected done, got {term}")
    if term.get("attempt_id") != attempt_id:
        raise HarnessError(
            f"{context}: done attempt_id mismatch expected={attempt_id} "
            f"got={term.get('attempt_id')}"
        )
    if saw_tool:
        raise HarnessError(f"{context}: unexpected tool_calls on greedy parity prompt")

    return {
        "done": term,
        "events": events,
        "token_text": "".join(token_text),
        "reasoning_text": "".join(reasoning_text),
        "finish_reason": term.get("finish_reason"),
        "tokens": term.get("tokens"),
    }


def generate_fault(
    session: DaemonSession,
    *,
    req_id: str,
    attempt_id: int,
    prompt: str,
    max_tokens: int,
    context: str,
) -> Dict[str, Any]:
    """Inject post-prefill fault; require typed retryable error and no visible output."""
    req = {
        "type": "generate",
        "id": req_id,
        "attempt_id": attempt_id,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "max_think_tokens": 1,
        "assistant_prefix": "closed_think",
        "test_fault_after_prefill": True,
    }

    def on_event(ev: Dict[str, Any]) -> None:
        ty = ev.get("type")
        if ty in FORBIDDEN_PRE_ERROR:
            raise HarnessError(
                f"{context}: forbidden event {ty!r} before injected error: {ev}"
            )

    term, events = session.request_until(
        req,
        terminal_types=("error", "done"),
        context=context,
        on_event=on_event,
        auto_commit=True,
    )
    if term.get("type") == "done":
        raise HarnessError(
            f"{context}: fault generate emitted done "
            f"(expected typed error, no terminal done): {term}"
        )
    if term.get("type") != "error":
        raise HarnessError(f"{context}: expected error terminal, got {term}")

    if term.get("retryable") is not True:
        raise HarnessError(f"{context}: injected error retryable must be true: {term}")
    if term.get("attempt_id") != attempt_id:
        raise HarnessError(
            f"{context}: error attempt_id mismatch expected={attempt_id} "
            f"got={term.get('attempt_id')}"
        )
    err_id = term.get("id")
    if err_id is not None and err_id != req_id:
        raise HarnessError(
            f"{context}: error id mismatch expected={req_id} got={err_id}"
        )
    if "rolled_back" not in term:
        raise HarnessError(
            f"{context}: injected error missing rolled_back attestation: {term}"
        )
    if term.get("rolled_back") is not True:
        raise HarnessError(
            f"{context}: injected error rolled_back must be true for parity retry: {term}"
        )
    if term.get("class") != FAULT_ERROR_CLASS:
        raise HarnessError(
            f"{context}: error class must be {FAULT_ERROR_CLASS!r}, "
            f"got {term.get('class')!r}: {term}"
        )
    if term.get("message") != FAULT_ERROR_MESSAGE:
        raise HarnessError(
            f"{context}: error message must be {FAULT_ERROR_MESSAGE!r}, "
            f"got {term.get('message')!r}: {term}"
        )

    for ev in events:
        if ev is term:
            continue
        if ev.get("type") in FORBIDDEN_PRE_ERROR:
            raise HarnessError(
                f"{context}: forbidden pre-error event in transcript: {ev}"
            )

    return {"error": term, "events": events}


def compare_outputs(
    clean: Dict[str, Any], retry: Dict[str, Any], route: str
) -> List[str]:
    mismatches: List[str] = []
    for key in ("token_text", "reasoning_text", "finish_reason", "tokens"):
        c, r = clean.get(key), retry.get(key)
        if c != r:
            mismatches.append(f"{route}.output.{key}: clean={c!r} retry={r!r}")
    return mismatches


def compare_snapshots(
    clean: Dict[str, Any], retry: Dict[str, Any], route: str
) -> List[str]:
    """Compare runtime/model state. state_epoch is NOT compared across processes."""
    mismatches: List[str] = []
    for key in SNAPSHOT_COMPARE_KEYS:
        c, r = clean.get(key), retry.get(key)
        if c != r:
            mismatches.append(f"{route}.snapshot.{key}: clean={c!r} retry={r!r}")
    return mismatches


def _snap_view(snap: Dict[str, Any]) -> Dict[str, Any]:
    keys = list(SNAPSHOT_COMPARE_KEYS) + [
        "kv_hash",
        "type",
        "schema_version",
        "arch",
        "eligible_routes",
        "state_epoch",
    ]
    return {k: snap.get(k) for k in keys}

def assert_graph_path_exercised(snap: Dict[str, Any], where: str, *, route: str) -> List[str]:
    """Fail closed unless the graph-enabled path left non-clean evidence.

    Always-clean snapshots under HIPFIRE_GRAPH=1 mean the graph path was not
    exercised (or capture never ran). Missing fields also fail closed.
    """
    mismatches: List[str] = []
    if "graph_clean" not in snap:
        mismatches.append(f"{where}: missing graph_clean evidence field")
        return mismatches
    if "replay_clean" not in snap:
        mismatches.append(f"{where}: missing replay_clean evidence field")
        return mismatches
    graph_clean = snap.get("graph_clean")
    replay_clean = snap.get("replay_clean")
    # Graph path exercised ⇒ at least one of graph/replay is non-clean.
    if graph_clean is True and replay_clean is True:
        mismatches.append(
            f"{where}: graph path not exercised before rollback "
            f"(graph_clean=True and replay_clean=True under HIPFIRE_GRAPH=1; "
            f"route={route})"
        )
    elif graph_clean is not False and graph_clean is not True:
        mismatches.append(
            f"{where}: graph_clean must be bool evidence, got {graph_clean!r}"
        )
    elif replay_clean is not False and replay_clean is not True:
        mismatches.append(
            f"{where}: replay_clean must be bool evidence, got {replay_clean!r}"
        )
    return mismatches


def assert_dflash_speculator_evidence(snap: Dict[str, Any], where: str) -> List[str]:
    """Require live DFlash speculator reset evidence fields (fail-closed)."""
    mismatches: List[str] = []
    for key in ("drafter_reset", "checkpoint_empty"):
        if key not in snap:
            mismatches.append(f"{where}: missing live speculator field {key!r}")
            continue
        val = snap.get(key)
        if val is not True:
            mismatches.append(
                f"{where}.{key}: expected True from live Speculator evidence, "
                f"got {val!r}"
            )
    return mismatches


def assert_post_reset_graph_clean(snap: Dict[str, Any], where: str) -> List[str]:
    """After attested rollback, graph/replay must be clean."""
    mismatches: List[str] = []
    for key, expected in (("graph_clean", True), ("replay_clean", True)):
        if key not in snap:
            mismatches.append(f"{where}: missing {key} after rollback")
            continue
        got = snap.get(key)
        if got is not expected:
            mismatches.append(
                f"{where}.{key}: expected {expected!r} after rollback, got {got!r}"
            )
    return mismatches


def aggregate_route_coverage(route_results: Sequence[Dict[str, Any]]) -> List[str]:
    """Overall ok fails on missing, extra, or duplicate route records."""
    mismatches: List[str] = []
    seen = [r.get("route") for r in route_results]
    expected = list(EXPECTED_ELIGIBLE_ROUTES)
    if len(seen) != len(expected):
        mismatches.append(
            f"route coverage: expected exactly {len(expected)} records "
            f"{expected!r}, got {len(seen)}: {seen!r}"
        )
    if len(seen) != len(set(seen)):
        mismatches.append(f"route coverage: duplicate route records: {seen!r}")
    missing = [r for r in expected if r not in seen]
    extra = [r for r in seen if r not in expected]
    if missing:
        mismatches.append(f"route coverage: missing required routes {missing!r}")
    if extra:
        mismatches.append(f"route coverage: unexpected extra routes {extra!r}")
    # Prefer stable order matching EXPECTED_ELIGIBLE_ROUTES when present.
    if not mismatches and seen != expected:
        mismatches.append(
            f"route coverage: order must be {expected!r}, got {seen!r}"
        )
    return mismatches



def run_route(
    *,
    daemon: Path,
    target: Path,
    draft: Optional[Path],
    prompt: str,
    max_tokens: int,
    max_seq: int,
    timeout_s: float,
    route: str,
    log_dir: Optional[Path],
) -> Dict[str, Any]:
    """Run clean process vs fault→reset→retry process for one route."""
    result: Dict[str, Any] = {
        "route": route,
        "ok": False,
        "mismatches": [],
        "clean": {},
        "fault": {},
        "state_epoch_note": (
            "state_epoch is validated for presence and reset-ack monotonicity "
            "on the fault process only; clean vs fault epoch equality is not required"
        ),
    }
    mismatches: List[str] = []

    if draft is None:
        raise HarnessError("run_route requires --draft for enforced dual-route parity")
    if route not in EXPECTED_ELIGIBLE_ROUTES:
        raise HarnessError(
            f"route {route!r} is not in frozen eligible set {EXPECTED_ELIGIBLE_ROUTES}"
        )
    use_draft = draft if route == "qwen_dflash" else None

    # ── Clean fresh process ──────────────────────────────────────────
    clean_log = (log_dir / f"{route}-clean.stderr.log") if log_dir else None
    clean = DaemonSession(daemon, timeout_s=timeout_s, log_path=clean_log)
    try:
        loaded = load_model(
            clean,
            target=target,
            draft=use_draft,
            max_seq=max_seq,
            context=f"{route}.clean.load",
        )
        result["clean"]["loaded"] = {
            "arch": loaded.get("arch"),
            "retry_reset_eligible": loaded.get("retry_reset_eligible"),
        }
        gen = generate_clean(
            clean,
            req_id=f"{route}-clean",
            attempt_id=1,
            prompt=prompt,
            max_tokens=max_tokens,
            context=f"{route}.clean.generate",
        )
        snap = snapshot(clean, context=f"{route}.clean.snapshot")
        result["clean"]["output"] = {
            "token_text": gen["token_text"],
            "reasoning_text": gen["reasoning_text"],
            "finish_reason": gen["finish_reason"],
            "tokens": gen["tokens"],
        }
        result["clean"]["snapshot"] = _snap_view(snap)
        result["clean"]["state_epoch"] = snap.get("state_epoch")
    finally:
        clean.close()

    # ── Fault → reset → retry fresh process ──────────────────────────
    fault_log = (log_dir / f"{route}-fault.stderr.log") if log_dir else None
    fault = DaemonSession(daemon, timeout_s=timeout_s, log_path=fault_log)
    try:
        loaded_f = load_model(
            fault,
            target=target,
            draft=use_draft,
            max_seq=max_seq,
            context=f"{route}.fault.load",
        )
        result["fault"]["loaded"] = {
            "arch": loaded_f.get("arch"),
            "retry_reset_eligible": loaded_f.get("retry_reset_eligible"),
        }
        assert_retry_eligible_true(
            loaded_f.get("retry_reset_eligible"), f"{route}.fault.loaded"
        )

        # Warm the graph-enabled path before fault injection so rollback
        # evidence is not an always-clean graph-off snapshot.
        warmup = generate_clean(
            fault,
            req_id=f"{route}-fault-warmup",
            attempt_id=9,
            prompt=prompt,
            max_tokens=max_tokens,
            context=f"{route}.fault.graph_warmup",
        )
        pre_rb = snapshot(fault, context=f"{route}.fault.pre_rollback_snapshot")
        result["fault"]["pre_rollback_snapshot"] = _snap_view(pre_rb)
        result["fault"]["graph_warmup_output"] = {
            "token_text": warmup["token_text"],
            "finish_reason": warmup["finish_reason"],
            "tokens": warmup["tokens"],
        }
        result["fault"]["graph_env"] = {
            "HIPFIRE_GRAPH": "1",
            "HIPFIRE_AR_GRAPH": "1",
        }
        if route == "qwen_dflash":
            mismatches.extend(
                assert_graph_path_exercised(
                    pre_rb, f"{route}.fault.pre_rollback_snapshot", route=route
                )
            )
        else:
            # AR still runs with graph env enabled; record env evidence only.
            if pre_rb.get("graph_clean") is None or pre_rb.get("replay_clean") is None:
                mismatches.append(
                    f"{route}.fault.pre_rollback_snapshot: missing graph_clean/"
                    f"replay_clean fields under HIPFIRE_GRAPH=1"
                )

        # Return to cold before the injected fault arm so fault→reset is clean.
        warm_reset_attempt = 9
        fault.send({"type": "reset", "attempt_id": warm_reset_attempt})
        warm_ack = fault.recv_line(context=f"{route}.fault.warmup_reset")
        while warm_ack.get("type") not in ("reset", "error"):
            warm_ack = fault.recv_line(context=f"{route}.fault.warmup_reset.drain")
        if warm_ack.get("type") == "error":
            raise HarnessError(f"{route}.fault.warmup_reset error: {warm_ack}")
        validate_reset_ack(
            fault, warm_ack, warm_reset_attempt, f"{route}.fault.warmup_reset"
        )

        faulted = generate_fault(
            fault,
            req_id=f"{route}-fault",
            attempt_id=10,
            prompt=prompt,
            max_tokens=max_tokens,
            context=f"{route}.fault.generate",
        )
        result["fault"]["injected_error"] = {
            "type": faulted["error"].get("type"),
            "class": faulted["error"].get("class"),
            "retryable": faulted["error"].get("retryable"),
            "rolled_back": faulted["error"].get("rolled_back"),
            "attempt_id": faulted["error"].get("attempt_id"),
            "id": faulted["error"].get("id"),
            "message": faulted["error"].get("message"),
        }

        reset_attempt = 11
        fault.send({"type": "reset", "attempt_id": reset_attempt})
        reset_ack = fault.recv_line(context=f"{route}.fault.reset")
        while reset_ack.get("type") not in ("reset", "error"):
            reset_ack = fault.recv_line(context=f"{route}.fault.reset.drain")
        if reset_ack.get("type") == "error":
            raise HarnessError(f"{route}.fault.reset error: {reset_ack}")
        validate_reset_ack(fault, reset_ack, reset_attempt, f"{route}.fault.reset")
        result["fault"]["reset"] = {
            "rolled_back": reset_ack.get("rolled_back"),
            "state_epoch": reset_ack.get("state_epoch"),
            "seq_pos": reset_ack.get("seq_pos"),
            "conversation_len": reset_ack.get("conversation_len"),
            "attempt_id": reset_ack.get("attempt_id"),
            "retry_reset_eligible": reset_ack.get("retry_reset_eligible"),
        }

        cold = snapshot(fault, context=f"{route}.fault.cold_snapshot")
        result["fault"]["cold_snapshot"] = _snap_view(cold)
        mismatches.extend(assert_cold_snapshot(cold, f"{route}.fault.cold_snapshot"))
        mismatches.extend(
            assert_post_reset_graph_clean(cold, f"{route}.fault.cold_snapshot")
        )
        if route == "qwen_dflash":
            mismatches.extend(
                assert_dflash_speculator_evidence(
                    cold, f"{route}.fault.cold_snapshot"
                )
            )

        # Same-process: cold snapshot epoch should match reset ack epoch.
        if cold.get("state_epoch") != reset_ack.get("state_epoch"):
            mismatches.append(
                f"{route}.fault.cold_snapshot.state_epoch: "
                f"expected reset-ack epoch {reset_ack.get('state_epoch')!r}, "
                f"got {cold.get('state_epoch')!r}"
            )

        retry_attempt = 12
        retry_gen = generate_clean(
            fault,
            req_id=f"{route}-retry",
            attempt_id=retry_attempt,
            prompt=prompt,
            max_tokens=max_tokens,
            context=f"{route}.fault.retry_generate",
        )
        retry_snap = snapshot(fault, context=f"{route}.fault.retry_snapshot")
        result["fault"]["retry_output"] = {
            "token_text": retry_gen["token_text"],
            "reasoning_text": retry_gen["reasoning_text"],
            "finish_reason": retry_gen["finish_reason"],
            "tokens": retry_gen["tokens"],
        }
        result["fault"]["retry_snapshot"] = _snap_view(retry_snap)
        result["fault"]["state_epoch"] = {
            "reset_ack": reset_ack.get("state_epoch"),
            "cold_snapshot": cold.get("state_epoch"),
            "retry_snapshot": retry_snap.get("state_epoch"),
        }
    finally:
        fault.close()

    # Cross-process parity (no state_epoch equality requirement).
    mismatches.extend(
        compare_outputs(
            result["clean"]["output"], result["fault"]["retry_output"], route
        )
    )
    mismatches.extend(
        compare_snapshots(
            result["clean"]["snapshot"], result["fault"]["retry_snapshot"], route
        )
    )

    for label, snap in (
        ("clean.snapshot", result["clean"]["snapshot"]),
        ("fault.cold_snapshot", result["fault"]["cold_snapshot"]),
        ("fault.retry_snapshot", result["fault"]["retry_snapshot"]),
    ):
        try:
            assert_eligible_routes(snap.get("eligible_routes"), f"{route}.{label}")
        except HarnessError as exc:
            mismatches.append(str(exc))

    result["mismatches"] = mismatches
    result["ok"] = not mismatches
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="serve_retry_gpu_parity.py",
        description=(
            "Fresh-process GPU parity harness: clean greedy vs "
            "fault-after-prefill → rollback → reset → retry for both "
            "enforced routes qwen_ar and qwen_dflash (draft required)."
        ),
    )
    p.add_argument(
        "--daemon",
        required=True,
        type=Path,
        help="Path to daemon binary built with --features serve-fault-inject",
    )
    p.add_argument(
        "--target",
        required=True,
        type=Path,
        help="Path to qwen35 target model (.hfq/.mq4/…)",
    )
    p.add_argument(
        "--draft",
        type=Path,
        required=True,
        help="Required DFlash draft model path (enforced qwen_dflash parity)",
    )
    p.add_argument(
        "--prompt-file",
        required=True,
        type=Path,
        help="Prompt text file (or JSON object with a prompt string field)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Greedy generation budget (default: 32)",
    )
    p.add_argument(
        "--max-seq",
        type=int,
        default=2048,
        help="Load-time max_seq / context capacity (default: 2048)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-operation timeout seconds (default: 600)",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Optional directory for daemon stderr logs (not protocol)",
    )
    p.add_argument(
        "--routes",
        default="auto",
        help=(
            "Must be 'auto' or exactly qwen_ar,qwen_dflash "
            "(both enforced; subsets rejected)"
        ),
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic harness self-tests and exit (no daemon)",
    )
    return p


def select_routes(args: argparse.Namespace) -> List[str]:
    """Always execute both enforced eligible routes; reject subsets."""
    if args.draft is None:
        raise HarnessError("--draft is required (both qwen_ar and qwen_dflash enforced)")
    enforced = list(EXPECTED_ELIGIBLE_ROUTES)
    if args.routes == "auto":
        return enforced
    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    if not routes:
        raise HarnessError("--routes produced an empty list")
    # Reject duplicates and unknown names first.
    if len(routes) != len(set(routes)):
        raise HarnessError(f"--routes must not contain duplicates: {routes!r}")
    for r in routes:
        if r not in EXPECTED_ELIGIBLE_ROUTES:
            raise HarnessError(
                f"route {r!r} is not in frozen eligible set {EXPECTED_ELIGIBLE_ROUTES}"
            )
    # Exact set required — no AR-only / subset PASS.
    if sorted(routes) != sorted(enforced):
        raise HarnessError(
            f"--routes must be exactly both enforced routes {enforced!r}, "
            f"got {routes!r}"
        )
    # Preserve canonical order regardless of CLI order.
    return enforced



def main(argv: Optional[Sequence[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # Fail closed before launch when --self-test is requested (no daemon).
    if "--self-test" in raw:
        return run_self_tests()

    parser = build_parser()
    args = parser.parse_args(argv)

    report: Dict[str, Any] = {
        "type": "serve_retry_gpu_parity_report",
        "schema_version": 1,
        "ok": False,
        "expected_eligible_routes": list(EXPECTED_ELIGIBLE_ROUTES),
        "daemon": str(args.daemon),
        "target": str(args.target),
        "draft": str(args.draft) if args.draft else None,
        "max_tokens": args.max_tokens,
        "max_seq": args.max_seq,
        "routes": [],
        "mismatches": [],
        "error": None,
        "graph_policy": (
            "HIPFIRE_GRAPH=1 and HIPFIRE_AR_GRAPH=1; pre-rollback snapshot must "
            "prove graph path exercised; post-rollback requires graph_clean and "
            "replay_clean true; qwen_dflash requires live drafter_reset+"
            "checkpoint_empty"
        ),
        "state_epoch_policy": (
            "validated for presence + reset-ack monotonicity on fault process; "
            "not compared for equality across clean vs faulted processes"
        ),
        "kv_hash_policy": (
            "kv_hash is diagnostic-only; full-allocation KV hash is not a "
            "parity gate (inactive/padding bytes undefined across processes)"
        ),
    }

    try:
        if args.draft is None:
            raise HarnessError("--draft is required before launch")
        if not args.draft.is_file():
            raise HarnessError(f"--draft not found: {args.draft}")
        if not args.target.is_file():
            raise HarnessError(f"--target not found: {args.target}")
        if args.max_tokens < 1:
            raise HarnessError("--max-tokens must be >= 1")
        if args.max_seq < 16:
            raise HarnessError("--max-seq must be >= 16")

        prompt = load_prompt(args.prompt_file)
        routes = select_routes(args)
        report["routes_planned"] = routes

        if args.log_dir is not None:
            args.log_dir.mkdir(parents=True, exist_ok=True)

        all_mismatches: List[str] = []
        route_results: List[Dict[str, Any]] = []
        for route in routes:
            rr = run_route(
                daemon=args.daemon,
                target=args.target,
                draft=args.draft,
                prompt=prompt,
                max_tokens=args.max_tokens,
                max_seq=args.max_seq,
                timeout_s=args.timeout,
                route=route,
                log_dir=args.log_dir,
            )
            route_results.append(rr)
            all_mismatches.extend(rr.get("mismatches") or [])

        coverage = aggregate_route_coverage(route_results)
        all_mismatches.extend(coverage)
        report["routes"] = route_results
        report["mismatches"] = all_mismatches
        report["ok"] = (
            not all_mismatches
            and all(bool(r.get("ok")) for r in route_results)
            and len(route_results) == len(EXPECTED_ELIGIBLE_ROUTES)
        )
    except HarnessError as exc:
        report["error"] = str(exc)
        report["ok"] = False
    except Exception as exc:  # noqa: BLE001 — surface unexpected failures in JSON
        report["error"] = f"internal harness failure: {exc}"
        report["ok"] = False

    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report.get("ok") else 1


def run_self_tests() -> int:
    """Deterministic pure-Python checks (no daemon / GPU)."""
    failures: List[str] = []

    # 1) no-draft select_routes fails closed
    class _A:
        draft = None
        routes = "auto"

    try:
        select_routes(_A())  # type: ignore[arg-type]
        failures.append("select_routes(no draft) should raise")
    except HarnessError:
        pass

    # 2) AR-only subset rejected
    class _B:
        draft = Path("/tmp/draft.fake")
        routes = "qwen_ar"

    try:
        select_routes(_B())  # type: ignore[arg-type]
        failures.append("select_routes(AR-only) should raise")
    except HarnessError:
        pass

    # 3) auto with draft => both enforced routes
    class _C:
        draft = Path("/tmp/draft.fake")
        routes = "auto"

    got = select_routes(_C())  # type: ignore[arg-type]
    if got != EXPECTED_ELIGIBLE_ROUTES:
        failures.append(f"auto routes expected {EXPECTED_ELIGIBLE_ROUTES}, got {got}")

    # 4) duplicate / missing coverage fails
    cov = aggregate_route_coverage([{"route": "qwen_ar", "ok": True}])
    if not cov:
        failures.append("aggregate_route_coverage missing route should fail")
    cov2 = aggregate_route_coverage(
        [
            {"route": "qwen_ar", "ok": True},
            {"route": "qwen_dflash", "ok": True},
            {"route": "qwen_ar", "ok": True},
        ]
    )
    if not cov2:
        failures.append("aggregate_route_coverage duplicate should fail")
    cov_ok = aggregate_route_coverage(
        [
            {"route": "qwen_ar", "ok": True},
            {"route": "qwen_dflash", "ok": True},
        ]
    )
    if cov_ok:
        failures.append(f"aggregate exact dual routes should pass, got {cov_ok}")

    # 5) always-clean pre-rollback fails closed
    ge = assert_graph_path_exercised(
        {"graph_clean": True, "replay_clean": True},
        "t.pre",
        route="qwen_ar",
    )
    if not ge:
        failures.append("always-clean graph evidence must fail")
    ge_ok = assert_graph_path_exercised(
        {"graph_clean": False, "replay_clean": True},
        "t.pre",
        route="qwen_ar",
    )
    if ge_ok:
        failures.append(f"non-clean graph evidence should pass, got {ge_ok}")

    # 6) post-reset cleanliness
    cold_bad = assert_post_reset_graph_clean(
        {"graph_clean": False, "replay_clean": True}, "t.cold"
    )
    if not cold_bad:
        failures.append("post-reset dirty graph must fail")
    cold_ok = assert_post_reset_graph_clean(
        {"graph_clean": True, "replay_clean": True}, "t.cold"
    )
    if cold_ok:
        failures.append(f"post-reset clean should pass, got {cold_ok}")

    # 7) dflash live speculator evidence
    df = assert_dflash_speculator_evidence(
        {"drafter_reset": False, "checkpoint_empty": True}, "t.df"
    )
    if not df:
        failures.append("drafter_reset=False must fail closed")
    df_ok = assert_dflash_speculator_evidence(
        {"drafter_reset": True, "checkpoint_empty": True}, "t.df"
    )
    if df_ok:
        failures.append(f"live speculator true should pass, got {df_ok}")

    # 8) cold snapshot includes replay_clean
    if "replay_clean" not in COLD_SNAPSHOT_EXPECT:
        failures.append("COLD_SNAPSHOT_EXPECT must require replay_clean")
    if COLD_SNAPSHOT_EXPECT.get("replay_clean") is not True:
        failures.append("COLD_SNAPSHOT_EXPECT.replay_clean must be True")

    # 9) argparse rejects missing --draft before launch
    parser = build_parser()
    try:
        parser.parse_args(
            [
                "--daemon",
                "/tmp/d",
                "--target",
                "/tmp/t",
                "--prompt-file",
                "/tmp/p",
            ]
        )
        failures.append("argparse must require --draft")
    except SystemExit as exc:
        if exc.code in (0, None):
            failures.append("missing --draft should nonzero-exit argparse")

    if failures:
        sys.stderr.write("self-test FAILURES:\n")
        for f in failures:
            sys.stderr.write(f"  - {f}\n")
        return 1
    sys.stdout.write(
        json.dumps(
            {
                "type": "serve_retry_gpu_parity_self_test",
                "ok": True,
                "checks": 9,
                "expected_eligible_routes": list(EXPECTED_ELIGIBLE_ROUTES),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
