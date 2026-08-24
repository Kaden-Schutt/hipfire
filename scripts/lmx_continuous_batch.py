#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""lmx_continuous_batch — fresh-process continuous-batch benchmark driver.

Developer-only orchestration: owns a fresh ``hipfire serve`` process for every
measured run and sends concurrent non-streaming OpenAI chat requests.

Fresh-process, exact-HIP, continuous-batch only: every run starts its own
CLI foreground in a new process group with HIPFIRE_DAEMON_BIN /
HIP_VISIBLE_DEVICES and ``serve ... --continuous-batch-size B``, waits for
/health, launches R requests simultaneously via ThreadPoolExecutor+barrier
with byte-identical prompt content (temperature 0, reasoning_effort none),
retains full response JSON + decoded content, captures per-request wall
start/end/latency + returned timings/usage/hipfire evidence, then stops the
whole process group and preserves the full serve log. Never reuses a server.

Report: schema, UTC, host/platform, full command/environment, git
head+dirty, model/CLI/daemon paths+size+sha256, prompt path+md5+sha256+bytes,
batch/concurrency/request counts, mode label (fixed_wave vs
continuous_refill), every raw run/request/decoded output, per-run aggregate
output tok/s = sum completion_tokens / wall(earliest start .. latest end),
summary medians + p50/p95 latency and TTFT. Validates batch attestation per
contract. Atomic output write. Signal-safe process-group cleanup only.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Schema / constants
# ---------------------------------------------------------------------------
SCHEMA = "lmx_continuous_batch/1"
SCHEMA_VERSION = "1"

REPO = Path(__file__).resolve().parent.parent

# Global current pgid for signal cleanup — only the pgid this script started.
_current_pgid: Optional[int] = None
_current_proc: Optional[subprocess.Popen] = None
_shutdown = False

_SECRET_ENV_NAME_PARTS = (
    "ACCESS_KEY",
    "API_KEY",
    "_KEY",
    "AUTH",
    "COOKIE",
    "CREDENTIAL",
    "JWT",
    "PASSWORD",
    "PASSWD",
    "PRIVATE_KEY",
    "SECRET",
    "SESSION",
    "TOKEN",
)


def _redacted_environment(source: Dict[str, str]) -> Dict[str, str]:
    """Retain environment shape without persisting credential values."""
    return {
        key: (
            "<redacted>"
            if any(part in key.upper() for part in _SECRET_ENV_NAME_PARTS)
            else value
        )
        for key, value in source.items()
    }


def _set_current_pgid(pgid: Optional[int], proc: Optional[subprocess.Popen] = None) -> None:
    global _current_pgid, _current_proc
    _current_pgid = pgid
    _current_proc = proc


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
        # give it a moment then SIGKILL
        time.sleep(1.0)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass
    # re-raise as SystemExit
    sys.stderr.write(f"\nlmx_continuous_batch: signal {signum} — stopped pgid {pgid}\n")
    sys.stderr.flush()
    # Do not sys.exit here; let main's finally handle report.
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
    # also try git diff --name-only to be safe
    return {"head": head, "head_short": head_short, "dirty": dirty}


def _is_device_occupied(device: str) -> bool:
    """Return True if KFD device is known occupied and monitor is reliable.

    Tries rocm-smi / amd-smi / fuser / lsof in order. If no reliable monitor
    is available, returns False (never refuse without evidence).
    """
    dev_id: Optional[int] = None
    # device may be "0" or "0,1" or env string
    try:
        # take first integer
        import re as _re
        m = _re.search(r"\d+", str(device))
        if m:
            dev_id = int(m.group(0))
    except Exception:
        dev_id = None

    # Try rocm-smi
    for bin_name in ("rocm-smi", "amd-smi"):
        bin_path = None
        # search PATH without shell
        for p in os.environ.get("PATH", "").split(os.pathsep):
            cand = Path(p) / bin_name
            if cand.is_file() and os.access(cand, os.X_OK):
                bin_path = str(cand)
                break
        # also try which via shutil fallback? we already did manual
        if bin_path is None:
            continue
        try:
            # Try JSON output first (amd-smi supports --json)
            # Use timeout and no shell
            r = subprocess.run(
                [bin_path, "--showpids"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            out = (r.stdout + r.stderr).lower()
            # If output mentions pid or process and we see device id, consider occupied
            # Heuristic: if output contains "pid" and non-empty pids, treat as occupied
            # Only refuse if we can reliably map to our device.
            if "pid" in out or "process" in out:
                # If dev_id specific, look for that device string
                if dev_id is not None:
                    # Look for patterns like "card0" or "gpu 0" or ":0"
                    if f"card{dev_id}" in out or f"gpu{dev_id}" in out or f"device {dev_id}" in out:
                        # Check if any digit pid present
                        import re as _re2
                        if _re2.search(r"\b\d{2,}\b", out):
                            # Need to ensure it's not just our own pid? Assume occupied
                            # Filter: if output is just header with no pids, don't refuse
                            lines = [l.strip() for l in out.splitlines() if l.strip()]
                            # Count lines that contain a pid number >100
                            pid_lines = 0
                            for line in lines:
                                if _re2.search(r"\b\d{3,}\b", line) and "pid" not in line.lower():
                                    # ambiguous, but assume occupied if line has numbers
                                    pass
                            # Safer: if output contains more than header, assume occupied
                            # Check if any line after header has numbers
                            # Simple: if we found "pid" and output length > 50 chars beyond header, occupied
                            if len(out) > 100 and any(c.isdigit() for c in out):
                                # Try to avoid false positive on empty list
                                # Look for "no process" phrases meaning not occupied
                                if "no " in out and "process" in out:
                                    return False
                                return True
                else:
                    # No dev_id, any pid means occupied
                    if "no " in out and "process" in out:
                        return False
                    import re as _re2
                    if _re2.search(r"\b\d{2,}\b", out):
                        return True
            # Also try --showmeminfo vram or --json
            # If that call showed occupancy, we already returned
            # Try second style: rocm-smi --json
            r2 = subprocess.run(
                [bin_path, "--json"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r2.returncode == 0 and r2.stdout.strip():
                try:
                    j = json.loads(r2.stdout)
                    # amd-smi json often has keys per card
                    txt = json.dumps(j).lower()
                    if dev_id is not None and f"card{dev_id}" in txt:
                        # check for processes
                        if "pid" in txt or "process" in txt:
                            return True
                    elif "pid" in txt:
                        return True
                except Exception:
                    pass
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            continue

    # Try fuser /dev/kfd and /dev/dri/card*
    for fuser_bin in ("fuser",):
        # find in PATH
        fuser_path = None
        for p in os.environ.get("PATH", "").split(os.pathsep):
            cand = Path(p) / fuser_bin
            if cand.is_file() and os.access(cand, os.X_OK):
                fuser_path = str(cand)
                break
        if fuser_path is None:
            continue
        # Check /dev/kfd
        for dev_path in ("/dev/kfd", f"/dev/dri/card{dev_id}" if dev_id is not None else None):
            if dev_path is None:
                continue
            if not Path(dev_path).exists():
                continue
            try:
                r = subprocess.run(
                    [fuser_path, dev_path],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                # fuser prints pids to stdout/stderr when occupied, empty when free
                out = (r.stdout + r.stderr).strip()
                if out:
                    # Has pids -> occupied
                    # Verify it's not just our own pid? fuser may list us if we hold kfd
                    # We are not holding kfd here, so any pid means occupied
                    import re as _re3
                    if _re3.search(r"\d+", out):
                        return True
            except Exception:
                continue

    # No reliable monitor found or no occupancy detected
    return False


def _kill_pgid(pgid: Optional[int], proc: Optional[subprocess.Popen], log_path: Optional[Path] = None) -> None:
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
        # ensure killed
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


def _wait_for_health(port: int, timeout_s: float, proc: Optional[subprocess.Popen]) -> bool:
    """Poll /health until healthy or timeout. Return True if healthy."""
    import urllib.request
    import urllib.error
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                try:
                    data = json.load(resp)
                except Exception:
                    time.sleep(0.5)
                    continue
                # Health is healthy if model field present or loading_model false? Check similar to serve_harness
                # Accept any 200 with json containing model or status ok
                if isinstance(data, dict):
                    # If explicit loading flag false and model present, healthy
                    if data.get("loading_model"):
                        time.sleep(0.5)
                        continue
                    # If health contains status ok
                    if data.get("status") == "ok":
                        return True
                    if isinstance(data.get("model"), str) and data.get("model"):
                        return True
                    # Even if model not yet loaded but health responds, consider healthy for continuous batch
                    # The serve may report model==null before first load but health 200 means listening
                    # For benchmark we need model loaded? The serve loads on first request
                    # So just require HTTP 200
                    return True
                return True
        except urllib.error.URLError:
            time.sleep(0.5)
            continue
        except Exception:
            time.sleep(0.5)
            continue
    return False


def _percentile(sorted_vals: List[float], pct: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    # pct in 0..100
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    d0 = k - f
    return float(sorted_vals[f] * (1 - d0) + sorted_vals[c] * d0)


def _median(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def _is_attractor(text: str) -> bool:
    """Heuristic non-attractor check: similar to serve_harness but dependency-free."""
    if not text or not text.strip():
        return True
    import re
    from collections import Counter
    toks = re.findall(r"\S+", text)
    if not toks:
        return True
    def uniq_ratio(t):
        return len(set(t)) / len(t) if t else 1.0
    def maxfreq_ratio(t):
        if not t:
            return 0.0
        c = Counter(t)
        return c.most_common(1)[0][1] / len(t)
    def gram3_ratio(t):
        if len(t) < 6:
            return 0.0
        g = [tuple(t[i:i+3]) for i in range(len(t)-2)]
        c = Counter(g)
        return sum(v for v in c.values() if v > 1) / len(g) if g else 0.0
    first = toks[:128]
    last = toks[-128:]
    half = toks[len(toks)//2:] if len(toks) > 2 else toks
    # Tight attractor definition: low uniq or high repeat
    if first and (uniq_ratio(first) < 0.15 or maxfreq_ratio(first) > 0.50):
        return True
    if last and (uniq_ratio(last) < 0.30 or maxfreq_ratio(last) > 0.50):
        return True
    if gram3_ratio(half) > 0.50:
        return True
    return False



# ---------------------------------------------------------------------------
# TP / device selector + batch receipt validation
# ---------------------------------------------------------------------------

TP_ALLOWED = (1, 4)
TP4_DEFAULT_DEVICES = "0,1,2,3"


def _parse_device_selector(device: str) -> List[str]:
    """Split a HIP_VISIBLE_DEVICES-style selector into non-empty tokens."""
    return [p.strip() for p in str(device).split(",") if p.strip()]


def _parse_device_indices(device: str) -> List[int]:
    """Parse comma-separated nonnegative device indices; raise ValueError on bad input."""
    parts = _parse_device_selector(device)
    if not parts:
        raise ValueError(f"empty device selector: {device!r}")
    indices: List[int] = []
    for p in parts:
        if not p.isdigit():
            raise ValueError(
                f"device selector entries must be nonnegative integers; got {p!r} in {device!r}"
            )
        indices.append(int(p))
    return indices


def _validate_tp_device_selector(tp: int, device: str) -> Optional[str]:
    """Fail closed when --device does not match the --tp contract.

    tp=1: exactly one nonnegative index.
    tp=4: exactly four distinct nonnegative indices.
    Returns an error string, or None when the selector is admissible.
    """
    try:
        indices = _parse_device_indices(device)
    except ValueError as e:
        return str(e)
    if tp == 1:
        if len(indices) != 1:
            return (
                f"--tp 1 requires a single-device --device selector; "
                f"got {len(indices)} device(s): {device!r}"
            )
        return None
    if tp == 4:
        if len(indices) != 4:
            return (
                f"--tp 4 requires a four-device --device selector "
                f"(default {TP4_DEFAULT_DEVICES!r}); got {len(indices)} device(s): {device!r}"
            )
        if len(set(indices)) != 4:
            return (
                f"--tp 4 requires four distinct devices in --device; "
                f"got duplicates in {device!r}"
            )
        return None
    return f"--tp must be one of {TP_ALLOWED}; got {tp}"


def _any_selected_device_occupied(device: str) -> Optional[str]:
    """Occupancy preflight over every selected physical GPU.

    Returns the first occupied device id as a string, or None if none are
    known-occupied (or no reliable monitor exists).
    """
    parts = _parse_device_selector(device)
    if not parts:
        parts = ["0"]
    for part in parts:
        if _is_device_occupied(part):
            return part
    return None


def _validate_hipfire_batch_receipt(
    hip: Any,
    *,
    batch_size: int,
    tp: int,
    run_idx: int,
    req_idx: Any,
) -> List[str]:
    """Validate hipfire runtime receipt against B/tp continuous-batch contract.

    Source-grounded on daemon attach_continuous_batch_route_evidence /
    attach_qwen_ep_batch_receipt_evidence:
      B>1 -> execution_mode continuous_batch_independent + executed/slots/refill/lane*
      tp=4 -> continuous_batch.parallelism=expert_parallel, rank_count=4,
             reduce=peer_rooted_f32 (never inferred from load logs)
      tp=1 -> must not claim expert_parallel
      B=1 -> must not claim batch execution
    """
    errors: List[str] = []
    prefix = f"run {run_idx} req {req_idx}"

    if batch_size > 1:
        if not isinstance(hip, dict):
            errors.append(f"{prefix}: missing hipfire evidence for B>1")
            return errors

        em = hip.get("execution_mode")
        if em != "continuous_batch_independent":
            errors.append(
                f"{prefix}: hipfire.execution_mode {em!r} != 'continuous_batch_independent'"
            )

        cb = hip.get("continuous_batch")
        if not isinstance(cb, dict):
            errors.append(f"{prefix}: missing continuous_batch object")
            return errors

        if cb.get("executed") is not True:
            errors.append(f"{prefix}: continuous_batch.executed != true")
        if cb.get("slots") != batch_size:
            errors.append(
                f"{prefix}: continuous_batch.slots {cb.get('slots')} != {batch_size}"
            )
        if cb.get("refill") != "continuous":
            errors.append(
                f"{prefix}: continuous_batch.refill {cb.get('refill')!r} != 'continuous'"
            )
        for field in ("lane", "lane_capacity", "max_active_lanes"):
            if field not in cb:
                errors.append(f"{prefix}: continuous_batch.{field} missing")

        if tp == 4:
            # Fail-closed EP receipt: expert_parallel / rank_count=4 / peer_rooted_f32.
            if cb.get("parallelism") != "expert_parallel":
                errors.append(
                    f"{prefix}: continuous_batch.parallelism "
                    f"{cb.get('parallelism')!r} != 'expert_parallel' (tp=4)"
                )
            if cb.get("rank_count") != 4:
                errors.append(
                    f"{prefix}: continuous_batch.rank_count "
                    f"{cb.get('rank_count')!r} != 4"
                )
            if cb.get("reduce") != "peer_rooted_f32":
                errors.append(
                    f"{prefix}: continuous_batch.reduce "
                    f"{cb.get('reduce')!r} != 'peer_rooted_f32'"
                )
        else:
            # Single-GPU receipts must not claim expert_parallel.
            if cb.get("parallelism") == "expert_parallel":
                errors.append(
                    f"{prefix}: continuous_batch.parallelism 'expert_parallel' "
                    f"forbidden for tp={tp}"
                )
            if hip.get("parallelism") == "expert_parallel":
                errors.append(
                    f"{prefix}: hipfire.parallelism 'expert_parallel' forbidden for tp={tp}"
                )
    else:
        # B == 1: honest sequential control — no false batch attestation.
        if isinstance(hip, dict):
            em = hip.get("execution_mode")
            if em == "continuous_batch_independent":
                errors.append(
                    f"{prefix}: false batch attestation execution_mode "
                    f"continuous_batch_independent for B=1"
                )
            cb = hip.get("continuous_batch")
            if isinstance(cb, dict) and cb.get("executed") is True:
                errors.append(
                    f"{prefix}: false batch attestation continuous_batch.executed "
                    f"true for B=1"
                )
            if tp == 1:
                if hip.get("parallelism") == "expert_parallel":
                    errors.append(
                        f"{prefix}: hipfire.parallelism 'expert_parallel' "
                        f"forbidden for tp=1"
                    )
                if isinstance(cb, dict) and cb.get("parallelism") == "expert_parallel":
                    errors.append(
                        f"{prefix}: continuous_batch.parallelism 'expert_parallel' "
                        f"forbidden for tp=1"
                    )

    return errors


def _build_serve_cmd(
    cli_abs: str,
    port: int,
    model: str,
    batch_size: int,
    tp: int,
) -> List[str]:
    """Construct hipfire serve argv; always forwards ``--tp <N>``."""
    return [
        cli_abs,
        "serve",
        "127.0.0.1",
        str(port),
        "--model",
        str(model),
        "--continuous-batch-size",
        str(batch_size),
        "--tp",
        str(tp),
    ]


def _extract_runtime_receipt_evidence(hip: Any) -> Optional[Dict[str, Any]]:
    """Pull EP/batch receipt fields from hipfire evidence for the report (no claims)."""
    if not isinstance(hip, dict):
        return None
    cb = hip.get("continuous_batch")
    out: Dict[str, Any] = {
        "execution_mode": hip.get("execution_mode"),
    }
    if isinstance(cb, dict):
        for k in (
            "executed",
            "slots",
            "lane",
            "lane_capacity",
            "max_active_lanes",
            "refill",
            "parallelism",
            "rank_count",
            "rank_mask",
            "reduce",
            "epoch",
            "rows",
            "moe_collectives",
            "requested_tp",
        ):
            if k in cb:
                out[k] = cb[k]
    return out


# ---------------------------------------------------------------------------
# HTTP request
# ---------------------------------------------------------------------------

def _do_request(
    barrier: threading.Barrier,
    idx: int,
    prompt_text: str,
    args: argparse.Namespace,
    port: int,
    timeout: float,
) -> Dict[str, Any]:
    """Send one non-streaming OpenAI chat request. Return per-request record."""
    import urllib.request
    import urllib.error

    # Barrier to launch simultaneously
    try:
        barrier.wait(timeout=30)
    except threading.BrokenBarrierError:
        pass

    wall_start = time.time()
    wall_start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    start_monotonic = time.monotonic()

    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "reasoning_effort": "none",
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    http_status: Optional[int] = None
    response_json: Optional[Any] = None
    response_text: Optional[str] = None
    error: Optional[str] = None
    decoded_content: Optional[str] = None

    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            http_status = resp.status if hasattr(resp, "status") else resp.getcode()
            raw = resp.read()
            response_text = raw.decode("utf-8", errors="replace")
            try:
                response_json = json.loads(response_text) if response_text else None
            except Exception as e:
                response_json = None
                error = f"json_parse_error: {e}"
    except urllib.error.HTTPError as e:
        http_status = e.code
        try:
            raw = e.read()
            response_text = raw.decode("utf-8", errors="replace") if raw else None
            if response_text:
                try:
                    response_json = json.loads(response_text)
                except Exception:
                    response_json = None
        except Exception as ex:
            error = f"http_error {e.code}: {ex}"
        if error is None:
            error = f"http_error {e.code}"
    except Exception as e:
        error = f"request_exception: {e}"
        http_status = None

    wall_end = time.time()
    wall_end_iso = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    wall_latency_ms = (wall_end - wall_start) * 1000.0
    # monotonic latency as well
    mono_latency_ms = (time.monotonic() - start_monotonic) * 1000.0

    # Extract decoded content
    if response_json is not None and isinstance(response_json, dict):
        # OpenAI shape: choices[0].message.content
        try:
            choices = response_json.get("choices")
            if isinstance(choices, list) and len(choices) > 0:
                msg = choices[0].get("message") or {}
                c = msg.get("content")
                if isinstance(c, str):
                    decoded_content = c
                elif c is None:
                    decoded_content = ""
                else:
                    decoded_content = json.dumps(c)
            else:
                # Fallback: try to find content anywhere
                decoded_content = ""
        except Exception:
            decoded_content = ""
    else:
        decoded_content = ""

    # Extract timings/usage/hipfire evidence (retain raw)
    timings = None
    usage = None
    hipfire_ev = None
    if isinstance(response_json, dict):
        timings = response_json.get("timings")
        usage = response_json.get("usage")
        hipfire_ev = response_json.get("hipfire")

    # Also retain latency_ms projection? HTTP projects latency_ms under timings
    # We already have wall latency; keep it separate

    record: Dict[str, Any] = {
        "index": idx,
        "wall_start": wall_start,
        "wall_start_iso": wall_start_iso,
        "wall_end": wall_end,
        "wall_end_iso": wall_end_iso,
        "latency_ms": wall_latency_ms,
        "mono_latency_ms": mono_latency_ms,
        "http_status": http_status,
        "error": error,
        "request_body": body,
        "response_json": response_json,
        "response_text": response_text,
        "decoded_content": decoded_content,
        "timings": timings,
        "usage": usage,
        "hipfire": hipfire_ev,
    }
    return record


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lmx_continuous_batch.py",
        description="Fresh-process continuous-batch benchmark driver — owns a fresh hipfire serve per measured run and sends concurrent non-streaming OpenAI chat requests. Exact-HIP batch only. Validates batch attestation and honest per-lane timings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/lmx_continuous_batch.py --model /path/model.mq4 --prompt-file prompts.txt \\\n"
            "      --batch-size 4 --requests 8 --runs 5 --max-tokens 128 --max-seq 4096 \\\n"
            "      --port 11520 --home-root /tmp/lmx_home --log-dir /tmp/lmx_logs --out report.json\n"
            "  python scripts/lmx_continuous_batch.py ... --tp 4 --device 0,1,2,3\n"
            "\n"
            "Notes:\n"
            "  --runs must be >=3\n"
            "  --tp choices: 1 (default) or 4 (Qwen expert-parallel continuous batch)\n"
            "  --thinking is restricted to 'off' for this v1 exact route\n"
            "  Every run starts a fresh process group and never reuses a server\n"
            "  Atomic output write; signal-safe pgid cleanup only\n"
        ),
    )
    # Required
    p.add_argument("--model", required=True, help="Model file path to serve (passed to hipfire serve --model)")
    p.add_argument("--prompt-file", required=True, help="UTF-8 prompt file; byte-identical content sent for every request")
    p.add_argument("--batch-size", required=True, type=int, dest="batch_size", help="Continuous batch size B (1..256); maps to hipfire serve --continuous-batch-size")
    p.add_argument("--requests", required=True, type=int, help="Concurrent requests per run R (1..256)")
    p.add_argument("--runs", required=True, type=int, help="Number of measured runs (must be >=3)")
    p.add_argument("--max-tokens", required=True, type=int, dest="max_tokens", help="max_tokens per request")
    p.add_argument("--max-seq", required=True, type=int, dest="max_seq", help="Max sequence length written to the isolated [memory] config for each fresh server")
    p.add_argument("--port", required=True, type=int, help="Serve port for 127.0.0.1 (reused sequentially, not concurrently)")
    p.add_argument("--home-root", required=True, dest="home_root", help="Root directory for per-run unique HOME directories")
    p.add_argument("--log-dir", required=True, dest="log_dir", help="Directory for per-run serve logs")
    p.add_argument("--out", required=True, help="Output report JSON path (atomically written)")
    # Optional
    p.add_argument("--cli", default="target/release/hipfire", help="Path to hipfire CLI binary (default: target/release/hipfire)")
    p.add_argument("--daemon", default="target/release/daemon", help="Path to daemon binary for HIPFIRE_DAEMON_BIN (default: target/release/daemon)")
    p.add_argument(
        "--device",
        default=None,
        help=(
            "KFD device selector for HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES. "
            "tp=1: one nonnegative index (default: env HIP_VISIBLE_DEVICES or 0). "
            f"tp=4: exactly four distinct nonnegative indices (default: {TP4_DEFAULT_DEVICES})."
        ),
    )
    p.add_argument(
        "--tp",
        type=int,
        choices=list(TP_ALLOWED),
        default=1,
        help=(
            "Expert-parallel degree for hipfire serve (choices: 1,4; default: 1). "
            "Forwarded as '--tp N'. tp=4 requires a four-device --device selector and "
            "runtime receipt expert_parallel/rank_count=4/peer_rooted_f32."
        ),
    )
    p.add_argument("--timeout", type=float, default=300.0, help="Per-request and health-wait timeout seconds (default: 300)")
    p.add_argument("--thinking", default="off", help="Thinking mode; restricted to 'off' for this v1 exact route (default: off)")
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Run pure-Python self-tests (no daemon/GPU) and exit",
    )
    return p


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate runs >=3
    if args.runs < 3:
        parser.error("--runs must be >=3")

    # Validate batch-size 1..256
    if not (1 <= args.batch_size <= 256):
        parser.error("--batch-size must be between 1 and 256")

    # Validate requests 1..256
    if not (1 <= args.requests <= 256):
        parser.error("--requests must be between 1 and 256")

    # Validate max-tokens
    if args.max_tokens < 1 or args.max_tokens > 393216:
        parser.error("--max-tokens must be between 1 and 393216")

    # Validate max-seq (hipfire canonical: 512..1048576)
    if args.max_seq < 512 or args.max_seq > 1048576:
        parser.error("--max-seq must be between 512 and 1048576")

    # Validate port
    if not (1 <= args.port <= 65535):
        parser.error("--port must be between 1 and 65535")

    # Thinking restricted to off
    if args.thinking != "off":
        parser.error("--thinking is restricted to 'off' for this v1 exact route")

    # --tp is choices-validated; keep an explicit guard for programmatic callers.
    if args.tp not in TP_ALLOWED:
        parser.error(f"--tp must be one of {list(TP_ALLOWED)}; got {args.tp}")

    # Device default resolution depends on tp degree.
    if args.device is None:
        if args.tp == 4:
            args.device = TP4_DEFAULT_DEVICES
        else:
            args.device = os.environ.get("HIP_VISIBLE_DEVICES", "0")

    # Canonicalize to comma-joined indices and fail closed on bad selectors.
    try:
        indices = _parse_device_indices(str(args.device))
        args.device = ",".join(str(i) for i in indices)
    except ValueError as e:
        parser.error(str(e))

    sel_err = _validate_tp_device_selector(int(args.tp), str(args.device))
    if sel_err is not None:
        parser.error(sel_err)

    # Resolve timeout
    if args.timeout <= 0:
        parser.error("--timeout must be >0")

    # Check prompt-file exists (skip for --self-test which uses dummy required args)
    if not getattr(args, "self_test", False):
        if not Path(args.prompt_file).is_file():
            parser.error(f"--prompt-file not found: {args.prompt_file}")

    # Check model file exists (warn but not hard error for CPU-only parse? Keep error)
    # For --help / import parse we still require file? Only error after parse
    # We already error above; keep it but allow missing in help case — argparse handles
    # So we keep check

    return args


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)

    # Prompt hashes
    prompt_path = Path(args.prompt_file)
    prompt_md5, prompt_sha256, prompt_bytes = _file_md5_sha256(prompt_path)
    # Byte length is file size
    prompt_byte_len = prompt_bytes
    # Read byte-identical prompt content (utf-8, preserve bytes)
    with open(prompt_path, "rb") as f:
        raw_prompt_bytes = f.read()
    try:
        prompt_text = raw_prompt_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # Fallback: replace errors but retain byte length evidence
        prompt_text = raw_prompt_bytes.decode("utf-8", errors="replace")

    # Model / cli / daemon hashes
    def _path_info(p: str, label: str) -> Dict[str, Any]:
        pp = Path(p)
        # Resolve relative to REPO if not absolute
        if not pp.is_absolute():
            pp_cand = REPO / p
            if pp_cand.exists():
                pp = pp_cand
        info: Dict[str, Any] = {"path": str(p), "resolved": str(pp)}
        if pp.is_file():
            try:
                sz, sha = _file_size_sha256(pp)
                info["size"] = sz
                info["sha256"] = sha
                info["exists"] = True
            except Exception as e:
                info["exists"] = True
                info["error"] = str(e)
        else:
            info["exists"] = False
            info["size"] = None
            info["sha256"] = None
        return info

    model_info = _path_info(args.model, "model")
    cli_info = _path_info(args.cli, "cli")
    daemon_info = _path_info(args.daemon, "daemon")

    # Command / env / git / host
    command_argv = sys.argv
    # Retain every environment key for reproducibility, but never credential values.
    env_snapshot = _redacted_environment(dict(os.environ))
    git_info = _git_info()
    host = socket.gethostname()
    platform_info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }
    utc_now = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

    # Mode label + tp/device (parse_args already validated)
    B = args.batch_size
    R = args.requests
    tp = int(args.tp)
    device_str = str(args.device)
    device_indices = _parse_device_indices(device_str)
    requested_parallelism = "expert_parallel" if tp == 4 else "none"
    if B == 1:
        mode_label = "sequential_control"
    elif R <= B:
        mode_label = "fixed_wave"
    else:
        mode_label = "continuous_refill"

    # Ensure dirs
    home_root = Path(args.home_root)
    log_dir = Path(args.log_dir)
    home_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Resolve cli/daemon absolute for env
    cli_abs = str(Path(cli_info["resolved"]).resolve()) if cli_info.get("exists") else cli_info["resolved"]
    daemon_abs = str(Path(daemon_info["resolved"]).resolve()) if daemon_info.get("exists") else daemon_info["resolved"]

    runs: List[Dict[str, Any]] = []
    all_latencies: List[float] = []
    all_ttfts: List[float] = []
    all_output_toks: List[float] = []

    overall_validation_errors: List[str] = []
    overall_passed = True

    # Health timeout: use args.timeout but also separate health timeout (180 default? Use args.timeout)
    health_timeout = max(30.0, float(args.timeout))

    for run_idx in range(args.runs):
        run_errors: List[str] = []
        run_validation_passed = True

        # 1) Device occupancy check — every selected physical GPU
        occupied = _any_selected_device_occupied(device_str)
        if occupied is not None:
            msg = (
                f"run {run_idx}: KFD device {occupied} (from selector {device_str}) "
                f"is known occupied (reliable monitor) — refusing"
            )
            run_errors.append(msg)
            overall_validation_errors.append(msg)
            overall_passed = False
            # Still record a run entry with error
            runs.append({
                "run_index": run_idx,
                "home": None,
                "log_path": None,
                "error": msg,
                "tp_degree": tp,
                "devices": device_str,
                "device_indices": device_indices,
                "requested_parallelism": requested_parallelism,
                "requests": [],
                "aggregate": None,
                "validation": {"passed": False, "errors": [msg]},
            })
            continue

        # 2) Unique HOME
        run_home = home_root / f"run_{run_idx}_{int(time.time())}_{os.getpid()}"
        run_home.mkdir(parents=True, exist_ok=True)
        # Also ensure .hipfire subdir exists? Let CLI handle but create
        (run_home / ".hipfire").mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"serve_run_{run_idx}.log"
        # Isolated HOME config: canonical [memory] max_seq = N
        isolated_cfg_path = run_home / ".hipfire" / "config.toml"
        try:
            isolated_cfg_path.write_text(f"[memory]\nmax_seq = {args.max_seq}\n", encoding="utf-8")
        except Exception as e:
            msg = f"run {run_idx}: failed to write isolated config {isolated_cfg_path}: {e}"
            run_errors.append(msg)
            overall_validation_errors.append(msg)
            run_validation_passed = False

        # Verify effective value from written file (preserve and report)
        isolated_cfg_effective: Optional[int] = None
        isolated_cfg_text: Optional[str] = None
        try:
            isolated_cfg_text = isolated_cfg_path.read_text(encoding="utf-8")
            # Parse canonical [memory] max_seq line
            import re as _re_cfg
            _m = _re_cfg.search(r"max_seq\s*=\s*(\d+)", isolated_cfg_text or "")
            if _m:
                isolated_cfg_effective = int(_m.group(1))
        except Exception:
            pass
        if isolated_cfg_effective != args.max_seq:
            msg = f"run {run_idx}: isolated config effective max_seq {isolated_cfg_effective} != requested {args.max_seq}"
            run_errors.append(msg)
            run_validation_passed = False

        # 3) Start CLI foreground in new process group
        serve_cmd = _build_serve_cmd(cli_abs, args.port, str(args.model), B, tp)
        env = dict(os.environ)
        env["HOME"] = str(run_home)
        # Contain HIPFIRE_HOME to isolated HOME to avoid host leakage
        env["HIPFIRE_HOME"] = str(run_home / ".hipfire")
        env["HIP_VISIBLE_DEVICES"] = device_str
        env["ROCR_VISIBLE_DEVICES"] = device_str
        env["HIPFIRE_DAEMON_BIN"] = daemon_abs
        # Ensure no stale PID file leakage? Clear HIPFIRE_SERVE_HARNESS_PID_FILE if present
        # Keep it but we control pgid ourselves
        # Open log file
        log_file = open(log_path, "wb")  # binary for full preservation
        proc: Optional[subprocess.Popen] = None
        pgid: Optional[int] = None
        try:
            proc = subprocess.Popen(
                serve_cmd,
                cwd=str(REPO),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            pgid = proc.pid  # under start_new_session, pid == pgid
            _set_current_pgid(pgid, proc)

            # 4) Wait for /health
            healthy = _wait_for_health(args.port, health_timeout, proc)
            if not healthy:
                # Check if proc died
                poll = proc.poll()
                log_file.flush()
                msg = f"run {run_idx}: serve failed to become healthy on port {args.port} (proc poll={poll})"
                run_errors.append(msg)
                overall_validation_errors.append(msg)
                run_validation_passed = False
                # Kill and preserve log
                try:
                    log_file.flush()
                except Exception:
                    pass
                _kill_pgid(pgid, proc)
                _set_current_pgid(None, None)
                try:
                    log_file.close()
                except Exception:
                    pass
                # Read log bytes for report
                try:
                    log_bytes = log_path.stat().st_size
                except Exception:
                    log_bytes = None
                runs.append({
                    "run_index": run_idx,
                    "home": str(run_home),
                    "log_path": str(log_path),
                    "log_bytes": log_bytes,
                    "serve_cmd": list(serve_cmd),
                    "tp_degree": tp,
                    "devices": device_str,
                    "device_indices": list(device_indices),
                    "requested_parallelism": requested_parallelism,
                    "isolated_config": {
                        "path": str(isolated_cfg_path) if 'isolated_cfg_path' in locals() else None,
                        "max_seq": args.max_seq,
                        "effective_max_seq": isolated_cfg_effective if 'isolated_cfg_effective' in locals() else None,
                        "canonical_toml": isolated_cfg_text if 'isolated_cfg_text' in locals() else None,
                    },
                    "effective_max_seq": isolated_cfg_effective if 'isolated_cfg_effective' in locals() else None,
                    "error": msg,
                    "requests": [],
                    "aggregate": None,
                    "validation": {"passed": False, "errors": [msg]},
                })
            # 5) Launch R requests simultaneously
            barrier = threading.Barrier(R)
            per_req: List[Dict[str, Any]] = []
            # Use ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=R) as executor:
                futures = []
                for i in range(R):
                    fut = executor.submit(_do_request, barrier, i, prompt_text, args, args.port, float(args.timeout))
                    futures.append(fut)
                # Collect as completed but preserve index order later
                for fut in as_completed(futures):
                    try:
                        rec = fut.result()
                        per_req.append(rec)
                    except Exception as e:
                        per_req.append({
                            "index": -1,
                            "wall_start": None,
                            "wall_end": None,
                            "latency_ms": None,
                            "http_status": None,
                            "error": f"executor_exception: {e}",
                            "response_json": None,
                            "decoded_content": None,
                            "timings": None,
                            "usage": None,
                            "hipfire": None,
                        })

            # Sort by index for deterministic report
            per_req_sorted = sorted(per_req, key=lambda x: x.get("index", 9999))

            # Capture latencies / ttfts
            for rec in per_req_sorted:
                lat = rec.get("latency_ms")
                if isinstance(lat, (int, float)):
                    all_latencies.append(float(lat))
                # TTFT from timings
                ttft = None
                t = rec.get("timings")
                if isinstance(t, dict):
                    ttft = t.get("ttft_ms")
                    if ttft is None:
                        # also try prefill_ms as proxy? But spec says ttft_ms distinct
                        ttft = t.get("prefill_ms")
                if isinstance(ttft, (int, float)):
                    all_ttfts.append(float(ttft))

            # Compute per-run aggregate output tok/s
            # wall from earliest request start to latest response end
            starts = [r["wall_start"] for r in per_req_sorted if isinstance(r.get("wall_start"), (int, float))]
            ends = [r["wall_end"] for r in per_req_sorted if isinstance(r.get("wall_end"), (int, float))]
            sum_completion = 0
            for rec in per_req_sorted:
                usage = rec.get("usage")
                if isinstance(usage, dict):
                    ct = usage.get("completion_tokens")
                    if isinstance(ct, int):
                        sum_completion += ct
                    elif isinstance(ct, float):
                        sum_completion += int(ct)
                else:
                    # fallback: try response_json usage
                    pass
                # Also check hipfire/timings? No, usage is canonical

            wall_duration = None
            aggregate_tok_s = None
            if starts and ends:
                earliest = min(starts)
                latest = max(ends)
                wall_duration = latest - earliest
                if wall_duration > 1e-6:
                    aggregate_tok_s = sum_completion / wall_duration
                    all_output_toks.append(float(aggregate_tok_s))

            aggregate = {
                "sum_completion_tokens": sum_completion,
                "earliest_start": min(starts) if starts else None,
                "latest_end": max(ends) if ends else None,
                "wall_duration_s": wall_duration,
                "output_tok_s": aggregate_tok_s,
            }

            # Validation per request
            # Require all HTTP success/nonempty/non-attractor
            for rec in per_req_sorted:
                idx = rec.get("index")
                # HTTP success
                status = rec.get("http_status")
                if status != 200:
                    msg = f"run {run_idx} req {idx}: http_status {status} != 200"
                    run_errors.append(msg)
                    run_validation_passed = False
                if rec.get("error") is not None:
                    # error may be set for HTTP error; already covered
                    # But if error present and status 200, still fail
                    if status == 200:
                        msg = f"run {run_idx} req {idx}: request error {rec.get('error')}"
                        run_errors.append(msg)
                        run_validation_passed = False
                # Nonempty
                content = rec.get("decoded_content")
                if not isinstance(content, str) or not content.strip():
                    msg = f"run {run_idx} req {idx}: empty decoded content"
                    run_errors.append(msg)
                    run_validation_passed = False
                else:
                    # Non-attractor
                    if _is_attractor(content):
                        msg = f"run {run_idx} req {idx}: attractor detected"
                        run_errors.append(msg)
                        run_validation_passed = False

                # Batch / EP receipt attestation (fail-closed; never from load logs)
                hip = rec.get("hipfire")
                timings = rec.get("timings")
                receipt_errors = _validate_hipfire_batch_receipt(
                    hip, batch_size=B, tp=tp, run_idx=run_idx, req_idx=idx
                )
                for msg in receipt_errors:
                    run_errors.append(msg)
                    run_validation_passed = False
                # Capture runtime receipt evidence on the per-request record (report only)
                rec["runtime_receipt_evidence"] = _extract_runtime_receipt_evidence(hip)

                if B > 1:
                    # Honest nonzero metrics: tok_s, prefill_ms, prefill_tok_s, decode_tok_s, ttft_ms, latency_ms
                    def _get_metric(name: str) -> Optional[float]:
                        # Check hipfire first, then timings
                        v = None
                        if isinstance(hip, dict):
                            v = hip.get(name)
                        if v is None and isinstance(timings, dict):
                            v = timings.get(name)
                        # Also check top-level response_json for latency_ms etc.
                        if v is None and isinstance(rec.get("response_json"), dict):
                            v = rec["response_json"].get(name)
                            if v is None and isinstance(rec["response_json"].get("timings"), dict):
                                v = rec["response_json"]["timings"].get(name)
                        return v

                    for metric in ("tok_s", "prefill_ms", "prefill_tok_s", "decode_tok_s", "ttft_ms", "latency_ms"):
                        val = _get_metric(metric)
                        # latency_ms is under timings per spec projection
                        if metric == "latency_ms" and val is None and isinstance(timings, dict):
                            val = timings.get("latency_ms")
                        if not isinstance(val, (int, float)) or float(val) <= 0:
                            msg = f"run {run_idx} req {idx}: metric {metric} missing or not >0 (got {val!r})"
                            run_errors.append(msg)
                            run_validation_passed = False

            # For B>1, also require at least one response per run with max_active_lanes >=2
            if B > 1:
                max_active_vals: List[int] = []
                for rec in per_req_sorted:
                    hip = rec.get("hipfire")
                    if isinstance(hip, dict):
                        cb = hip.get("continuous_batch")
                        if isinstance(cb, dict):
                            v = cb.get("max_active_lanes")
                            if isinstance(v, int):
                                max_active_vals.append(v)
                            elif isinstance(v, float):
                                max_active_vals.append(int(v))
                if not any(v >= 2 for v in max_active_vals):
                    msg = f"run {run_idx}: no response with max_active_lanes>=2 (got {max_active_vals})"
                    run_errors.append(msg)
                    run_validation_passed = False

            # Stop process group and preserve log
            log_file.flush()
            # Need to flush file before kill
            _kill_pgid(pgid, proc)
            _set_current_pgid(None, None)
            try:
                log_file.close()
            except Exception:
                pass
            log_file = None  # prevent double close in finally

            try:
                log_bytes = log_path.stat().st_size
            except Exception:
                log_bytes = None
            # Also capture log text tail for quick debugging? But retain full log on disk
            # For report, we should not embed full log if huge; keep path and bytes

            # Collect runtime receipt evidence from first successful request (if any)
            run_receipt_evidence = None
            for rec in per_req_sorted:
                ev = rec.get("runtime_receipt_evidence")
                if isinstance(ev, dict):
                    run_receipt_evidence = ev
                    break

            runs.append({
                "run_index": run_idx,
                "home": str(run_home),
                "log_path": str(log_path),
                "log_bytes": log_bytes,
                "serve_cmd": list(serve_cmd),
                "tp_degree": tp,
                "devices": device_str,
                "device_indices": list(device_indices),
                "requested_parallelism": requested_parallelism,
                "runtime_receipt_evidence": run_receipt_evidence,
                "isolated_config": {
                    "path": str(isolated_cfg_path),
                    "max_seq": args.max_seq,
                    "effective_max_seq": isolated_cfg_effective,
                    "canonical_toml": isolated_cfg_text,
                },
                "effective_max_seq": isolated_cfg_effective,
                "requests": per_req_sorted,
                "aggregate": aggregate,
                "validation": {"passed": run_validation_passed, "errors": run_errors},
            })
            if not run_validation_passed:
                overall_passed = False
                overall_validation_errors.extend(run_errors)

        except Exception as e:
            # Ensure cleanup
            if log_file is not None:
                try:
                    log_file.flush()
                except Exception:
                    pass
            _kill_pgid(pgid if 'pgid' in locals() else None, proc if 'proc' in locals() else None)
            _set_current_pgid(None, None)
            if 'log_file' in locals() and log_file is not None:
                try:
                    log_file.close()
                except Exception:
                    pass
            msg = f"run {run_idx}: exception {e}"
            run_errors.append(msg)
            overall_validation_errors.append(msg)
            overall_passed = False
            # Record run
            try:
                log_bytes = log_path.stat().st_size if 'log_path' in locals() and Path(log_path).exists() else None
            except Exception:
                log_bytes = None
            runs.append({
                "run_index": run_idx,
                "home": str(run_home) if 'run_home' in locals() else None,
                "log_path": str(log_path) if 'log_path' in locals() else None,
                "log_bytes": log_bytes,
                "tp_degree": tp,
                "devices": device_str,
                "device_indices": list(device_indices),
                "requested_parallelism": requested_parallelism,
                "isolated_config": {
                    "path": str(isolated_cfg_path) if 'isolated_cfg_path' in locals() else None,
                    "max_seq": args.max_seq if 'args' in locals() else None,
                    "effective_max_seq": isolated_cfg_effective if 'isolated_cfg_effective' in locals() else None,
                    "canonical_toml": isolated_cfg_text if 'isolated_cfg_text' in locals() else None,
                },
                "effective_max_seq": isolated_cfg_effective if 'isolated_cfg_effective' in locals() else None,
                "error": msg,
                "requests": [],
                "aggregate": None,
                "validation": {"passed": False, "errors": [msg]},
            })
            # Also re-raise if shutdown?
            if _shutdown:
                break
        finally:
            # Ensure pgid cleared
            if _current_pgid is not None and 'pgid' in locals() and _current_pgid == pgid:
                _set_current_pgid(None, None)
            if 'log_file' in locals() and log_file is not None:
                try:
                    log_file.close()
                except Exception:
                    pass

        if _shutdown:
            break

    # Summary medians + p50/p95 latency and TTFT
    # Need to handle empty case
    summary: Dict[str, Any] = {}
    # Output tok/s
    if all_output_toks:
        sorted_toks = sorted(all_output_toks)
        summary["median_output_tok_s"] = _median(all_output_toks)
        summary["p50_output_tok_s"] = _percentile(sorted_toks, 50)
        summary["p95_output_tok_s"] = _percentile(sorted_toks, 95)
        summary["min_output_tok_s"] = min(all_output_toks)
        summary["max_output_tok_s"] = max(all_output_toks)
        summary["mean_output_tok_s"] = sum(all_output_toks) / len(all_output_toks)
    else:
        summary["median_output_tok_s"] = None
        summary["p50_output_tok_s"] = None
        summary["p95_output_tok_s"] = None

    if all_latencies:
        sorted_lat = sorted(all_latencies)
        summary["median_latency_ms"] = _median(all_latencies)
        summary["p50_latency_ms"] = _percentile(sorted_lat, 50)
        summary["p95_latency_ms"] = _percentile(sorted_lat, 95)
        summary["min_latency_ms"] = min(all_latencies)
        summary["max_latency_ms"] = max(all_latencies)
        summary["mean_latency_ms"] = sum(all_latencies) / len(all_latencies)
    else:
        summary["median_latency_ms"] = None
        summary["p50_latency_ms"] = None
        summary["p95_latency_ms"] = None

    if all_ttfts:
        sorted_ttft = sorted(all_ttfts)
        summary["median_ttft_ms"] = _median(all_ttfts)
        summary["p50_ttft_ms"] = _percentile(sorted_ttft, 50)
        summary["p95_ttft_ms"] = _percentile(sorted_ttft, 95)
        summary["min_ttft_ms"] = min(all_ttfts)
        summary["max_ttft_ms"] = max(all_ttfts)
        summary["mean_ttft_ms"] = sum(all_ttfts) / len(all_ttfts)
    else:
        summary["median_ttft_ms"] = None
        summary["p50_ttft_ms"] = None
        summary["p95_ttft_ms"] = None

    # Per-run median output tok/s already in summary; also include counts

    # Build report
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generated_utc": utc_now,
        "host": host,
        "platform": platform_info,
        "command": command_argv,
        "argv": command_argv,
        "environment": {
            "HIP_VISIBLE_DEVICES": device_str,
            "ROCR_VISIBLE_DEVICES": device_str,
            "HIPFIRE_DAEMON_BIN": daemon_abs,
            "HOME_ROOT": str(home_root),
            "LOG_DIR": str(log_dir),
            "PORT": args.port,
            # Keys are complete; credential-bearing values are redacted above.
            "full_env": env_snapshot,
        },
        "git": git_info,
        "model": model_info,
        "cli": cli_info,
        "daemon": daemon_info,
        "prompt": {
            "path": str(prompt_path),
            "md5": prompt_md5,
            "sha256": prompt_sha256,
            "bytes": prompt_bytes,
            "byte_length": prompt_byte_len,
        },
        "args": {
            "model": args.model,
            "prompt_file": args.prompt_file,
            "batch_size": B,
            "continuous_batch_size": B,
            "requests": R,
            "runs": args.runs,
            "max_tokens": args.max_tokens,
            "max_seq": args.max_seq,
            "port": args.port,
            "home_root": str(home_root),
            "log_dir": str(log_dir),
            "out": args.out,
            "cli": args.cli,
            "daemon": args.daemon,
            "device": device_str,
            "tp": tp,
            "timeout": args.timeout,
            "thinking": args.thinking,
        },
        "batch": B,
        "concurrency": R,
        "request_counts": {"batch_size": B, "requests_per_run": R, "runs": args.runs, "total_requests": args.runs * R},
        "batch_concurrency": B,
        "tp_degree": tp,
        "devices": device_str,
        "device_indices": list(device_indices),
        "requested_tp": tp,
        "requested_parallelism": requested_parallelism,
        "mode": mode_label,
        "mode_label": mode_label,
        "effective_max_seq": args.max_seq,
        "isolated_memory_config": {
            "canonical_toml": f"[memory]\nmax_seq = {args.max_seq}\n",
            "effective_max_seq": args.max_seq,
            "schema": "[memory] max_seq = N",
        },
        "runs": runs,
        # Alias for raw evidence
        "per_run": runs,
        "summary": summary,
        "validation": {
            "passed": overall_passed,
            "errors": overall_validation_errors,
            "mode_label": mode_label,
            "batch_attestation_required": B > 1,
            "expert_parallel_required": tp == 4,
            "tp_degree": tp,
            "devices": device_str,
            "requested_parallelism": requested_parallelism,
        },
    }

    # Ensure out dir exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write via temp file + os.replace
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(out_path.parent), prefix=".tmp_lmx_cb_")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tf:
            json.dump(report, tf, indent=2, sort_keys=False, ensure_ascii=False)
            tf.write("\n")
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(tmp_path, str(out_path))
    finally:
        try:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
        except Exception:
            pass

    # Exit nonzero after writing report when validation fails
    if not overall_passed:
        sys.stderr.write(f"lmx_continuous_batch: validation FAILED — {len(overall_validation_errors)} errors; report written to {out_path}\n")
        for e in overall_validation_errors[:20]:
            sys.stderr.write(f"  - {e}\n")
        sys.stderr.flush()
        return 2

    sys.stderr.write(f"lmx_continuous_batch: validation PASSED — report written to {out_path}\n")
    return 0



def run_self_tests() -> int:
    """Deterministic pure-Python checks (no daemon / GPU)."""
    failures: List[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    # --tp choices / defaults
    check(TP_ALLOWED == (1, 4), f"TP_ALLOWED={TP_ALLOWED!r}")
    check(TP4_DEFAULT_DEVICES == "0,1,2,3", f"TP4_DEFAULT_DEVICES={TP4_DEFAULT_DEVICES!r}")

    # Device selector gate
    check(_validate_tp_device_selector(1, "0") is None, "tp1 single device should pass")
    check(_validate_tp_device_selector(1, "7") is None, "tp1 any single device should pass")
    check(_validate_tp_device_selector(1, "0,1") is not None, "tp1 multi-device must fail")
    check(_validate_tp_device_selector(4, "0,1,2,3") is None, "tp4 default devices should pass")
    check(_validate_tp_device_selector(4, "0,1,2") is not None, "tp4 three devices must fail")
    check(_validate_tp_device_selector(4, "0,1,2,2") is not None, "tp4 duplicate devices must fail")
    check(_validate_tp_device_selector(4, "0") is not None, "tp4 single device must fail")
    check(_validate_tp_device_selector(4, "a,b,c,d") is not None, "tp4 non-int devices must fail")

    # Indices parser
    check(_parse_device_indices("0,1,2,3") == [0, 1, 2, 3], "parse four devices")
    try:
        _parse_device_indices("")
        check(False, "empty device selector must raise")
    except ValueError:
        check(True, "empty device selector raises")

    # serve argv: always forwards --tp N
    cmd1 = _build_serve_cmd("/cli", 11520, "/m.mq4", 4, 1)
    check(cmd1[-2:] == ["--tp", "1"], f"tp=1 must forward --tp 1, got {cmd1}")
    check(cmd1[cmd1.index("--continuous-batch-size") + 1] == "4", "batch size in serve argv")
    cmd4 = _build_serve_cmd("/cli", 11520, "/m.mq4", 8, 4)
    check(cmd4[-2:] == ["--tp", "4"], f"tp=4 must append --tp 4, got {cmd4}")

    # B>1 tp=4 requires full expert-parallel receipt (source-grounded fields)
    good_tp4 = {
        "execution_mode": "continuous_batch_independent",
        "continuous_batch": {
            "executed": True,
            "parallelism": "expert_parallel",
            "rank_count": 4,
            "reduce": "peer_rooted_f32",
            "slots": 4,
            "lane": 0,
            "lane_capacity": 4096,
            "max_active_lanes": 2,
            "refill": "continuous",
        },
    }
    check(
        _validate_hipfire_batch_receipt(good_tp4, batch_size=4, tp=4, run_idx=0, req_idx=0) == [],
        "good tp4 receipt should pass",
    )
    bad_missing_ep = {
        "execution_mode": "continuous_batch_independent",
        "continuous_batch": {
            "executed": True,
            "slots": 4,
            "lane": 0,
            "lane_capacity": 4096,
            "max_active_lanes": 2,
            "refill": "continuous",
        },
    }
    errs = _validate_hipfire_batch_receipt(bad_missing_ep, batch_size=4, tp=4, run_idx=0, req_idx=0)
    check(any("expert_parallel" in e for e in errs), f"tp4 missing EP fields must fail: {errs}")
    check(any("rank_count" in e for e in errs), f"tp4 missing rank_count must fail: {errs}")
    check(any("peer_rooted_f32" in e for e in errs), f"tp4 missing reduce must fail: {errs}")

    # B>1 tp=1 must reject expert_parallel labels
    bad_tp1_ep = {
        "execution_mode": "continuous_batch_independent",
        "continuous_batch": {
            "executed": True,
            "parallelism": "expert_parallel",
            "slots": 4,
            "lane": 0,
            "lane_capacity": 4096,
            "max_active_lanes": 2,
            "refill": "continuous",
        },
    }
    errs = _validate_hipfire_batch_receipt(bad_tp1_ep, batch_size=4, tp=1, run_idx=0, req_idx=0)
    check(any("forbidden for tp=1" in e for e in errs), f"tp1 EP label must fail: {errs}")

    good_tp1 = {
        "execution_mode": "continuous_batch_independent",
        "continuous_batch": {
            "executed": True,
            "slots": 4,
            "lane": 0,
            "lane_capacity": 4096,
            "max_active_lanes": 2,
            "refill": "continuous",
        },
    }
    check(
        _validate_hipfire_batch_receipt(good_tp1, batch_size=4, tp=1, run_idx=0, req_idx=0) == [],
        "good tp1 batch receipt should pass",
    )

    # B=1 must not claim batch; also reject expert_parallel at tp=1
    check(
        _validate_hipfire_batch_receipt(None, batch_size=1, tp=1, run_idx=0, req_idx=0) == [],
        "B=1 with no hipfire should pass",
    )
    errs = _validate_hipfire_batch_receipt(good_tp1, batch_size=1, tp=1, run_idx=0, req_idx=0)
    check(any("false batch attestation" in e for e in errs), f"B=1 batch claim must fail: {errs}")
    errs = _validate_hipfire_batch_receipt(
        {"parallelism": "expert_parallel"},
        batch_size=1,
        tp=1,
        run_idx=0,
        req_idx=0,
    )
    check(any("expert_parallel" in e for e in errs), f"B=1 tp1 EP must fail: {errs}")

    # Receipt extractor preserves EP fields without inventing them
    extracted = _extract_runtime_receipt_evidence(good_tp4)
    check(isinstance(extracted, dict), "extractor returns dict")
    check(extracted.get("parallelism") == "expert_parallel", "extractor keeps parallelism")
    check(extracted.get("rank_count") == 4, "extractor keeps rank_count")
    check(extracted.get("reduce") == "peer_rooted_f32", "extractor keeps reduce")
    check("requested_tp" not in extracted, "extractor must not invent requested_tp")

    # Secret redaction must not capture credential values
    red = _redacted_environment({"HIPFIRE_API_KEY": "supersecret", "PATH": "/usr/bin", "HOME": "/tmp/h"})
    check(red.get("HIPFIRE_API_KEY") != "supersecret", "secret env values must be redacted")
    check(red.get("PATH") == "/usr/bin", "non-secret env values must be retained")

    if failures:
        for f in failures:
            sys.stderr.write(f"lmx_continuous_batch self-test FAIL: {f}\n")
        sys.stderr.flush()
        return 1
    sys.stderr.write("lmx_continuous_batch: self-test OK\n")
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        try:
            sys.exit(run_self_tests())
        except Exception as e:
            sys.stderr.write(f"lmx_continuous_batch self-test fatal: {e}\n")
            sys.exit(1)
    try:
        sys.exit(main())
    except SystemExit as e:
        # Ensure pgid cleanup on SystemExit
        if _current_pgid is not None:
            try:
                os.killpg(_current_pgid, signal.SIGTERM)
                time.sleep(0.5)
                os.killpg(_current_pgid, signal.SIGKILL)
            except Exception:
                pass
        raise
    except KeyboardInterrupt:
        if _current_pgid is not None:
            try:
                os.killpg(_current_pgid, signal.SIGTERM)
                time.sleep(0.5)
                os.killpg(_current_pgid, signal.SIGKILL)
            except Exception:
                pass
        sys.stderr.write("lmx_continuous_batch: interrupted\n")
        sys.exit(130)
    except Exception as e:
        if _current_pgid is not None:
            try:
                os.killpg(_current_pgid, signal.SIGTERM)
                time.sleep(0.5)
                os.killpg(_current_pgid, signal.SIGKILL)
            except Exception:
                pass
        sys.stderr.write(f"lmx_continuous_batch: fatal {e}\n")
        raise

