#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DAEMON="${DAEMON:-$ROOT/target/release/examples/daemon}"
MODEL="${MODEL:-$HOME/.hipfire/models/qwen3.5-0.8b-mq4.hfq}"
MAX_SEQ="${MAX_SEQ:-512}"
EXPECTED_DAEMON_PREFILL_BACKEND="${EXPECTED_DAEMON_PREFILL_BACKEND:-fused_dense}"
SERVER_SMOKE_LOCK="${HIPFIRE_SERVER_SMOKE_LOCK:-${TMPDIR:-/tmp}/hipfire-server-smoke.lock}"
SERVER_SMOKE_LOCK_WAIT="${HIPFIRE_SERVER_SMOKE_LOCK_WAIT:-300}"

exec 9>"$SERVER_SMOKE_LOCK"
if ! flock -w "$SERVER_SMOKE_LOCK_WAIT" 9; then
  echo "timed out waiting for server smoke lock: $SERVER_SMOKE_LOCK" >&2
  exit 2
fi

if [[ ! -x "$DAEMON" ]]; then
  echo "missing daemon binary: $DAEMON" >&2
  echo "build it with: cargo build --release -p hipfire-runtime --example daemon" >&2
  exit 2
fi

if [[ ! -f "$MODEL" ]]; then
  echo "missing model: $MODEL" >&2
  exit 2
fi

python3 - "$ROOT" "$DAEMON" "$MODEL" "$MAX_SEQ" "$EXPECTED_DAEMON_PREFILL_BACKEND" <<'PY'
import concurrent.futures
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from typing import Any

root, daemon, model, max_seq_s, expected_daemon_prefill_backend = sys.argv[1:]
max_seq = int(max_seq_s)


def pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def fetch_json(url: str, body: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def wait_health(base_url: str, proc: subprocess.Popen[str], log_path: str) -> dict[str, Any]:
    deadline = time.time() + 120.0
    last_err: Exception | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}; log={log_path}")
        try:
            health = fetch_json(f"{base_url}/health", timeout=2.0)
            if health.get("status") == "ok":
                return health
        except Exception as err:
            last_err = err
        time.sleep(0.25)
    raise RuntimeError(f"server did not become healthy; last_err={last_err}; log={log_path}")


def chat_request(base_url: str, label: str) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": f"Return one short word for {label}."},
        ],
        "stream": False,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 1,
    }
    try:
        out = fetch_json(f"{base_url}/v1/chat/completions", body, timeout=120.0)
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{label}: HTTP {err.code}: {detail}") from err
    if "error" in out:
        raise RuntimeError(f"{label}: response error: {out['error']}")
    choices = out.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"{label}: malformed response: {out}")
    return out


def streaming_chat_request(base_url: str, label: str) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": f"Return one short word for streaming {label}."},
        ],
        "stream": True,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 1,
    }
    data = json.dumps(body, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{label}: streaming HTTP {err.code}: {detail}") from err
    if "data:" not in raw:
        raise RuntimeError(f"{label}: malformed streaming response: {raw!r}")
    return raw


port = pick_port()
base_url = f"http://127.0.0.1:{port}"
log_file = tempfile.NamedTemporaryFile("w", prefix="hipfire-server-prefill-", suffix=".log", delete=False)
log_path = log_file.name

env = os.environ.copy()
env.update({
    "HIPFIRE_DAEMON_BIN": daemon,
    "HIPFIRE_MODEL": model,
    "HIPFIRE_KV_MODE": "q8",
    "HIPFIRE_NO_PID_FILE": "1",
    "HIPFIRE_SERVER_PREFILL_BATCH": "1",
    "HIPFIRE_SERVER_PREFILL_BATCH_MAX": "2",
    "HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS": "250",
    "HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE": "250",
    "HIPFIRE_MAX_SEQ": str(max_seq),
    "HIPFIRE_DFLASH_DRAFT": "",
})

proc = subprocess.Popen(
    ["bun", os.path.join(root, "cli", "index.ts"), "serve", "127.0.0.1", str(port)],
    cwd=root,
    stdin=subprocess.DEVNULL,
    stdout=log_file,
    stderr=log_file,
    text=True,
    env=env,
)

try:
    initial = wait_health(base_url, proc, log_path)
    prefill = initial.get("prefill_batch", {})
    if prefill.get("generate_batch_prefill_capability") != "supported":
        raise RuntimeError(f"server prefill capability not supported after warmup: {prefill}; log={log_path}")

    single_response = chat_request(base_url, "single-timeout")
    single_health = fetch_json(f"{base_url}/health", timeout=10.0)
    single_prefill = single_health.get("prefill_batch", {})
    if single_prefill.get("runtime_dispatch_skipped_reason") != "not_skipped":
        raise RuntimeError(
            "single request did not flush through daemon prefill: "
            f"{single_prefill}; log={log_path}"
        )
    if int(single_prefill.get("pending_requests") or 0) != 0:
        raise RuntimeError(
            "single request left pending prefill waiters after timeout: "
            f"{single_prefill}; log={log_path}"
        )

    streaming_chat_request(base_url, "mixed-mode-stream")
    stream_health = fetch_json(f"{base_url}/health", timeout=10.0)
    stream_prefill = stream_health.get("prefill_batch", {})
    if int(stream_prefill.get("pending_requests") or 0) != 0:
        raise RuntimeError(
            "streaming request was incorrectly left in the prefill wait queue: "
            f"{stream_prefill}; log={log_path}"
        )
    if int(stream_prefill.get("resident_runtime_sessions") or 0) != 0:
        raise RuntimeError(
            "streaming request created a resident runtime prefill session: "
            f"{stream_prefill}; log={log_path}"
        )

    start_barrier = threading.Barrier(3)

    def synchronized_chat_request(label: str) -> dict[str, Any]:
        start_barrier.wait(timeout=10.0)
        return chat_request(base_url, label)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(synchronized_chat_request, "request-a"),
            pool.submit(synchronized_chat_request, "request-b"),
        ]
        start_barrier.wait(timeout=10.0)
        responses = [future.result() for future in futures]

    health = fetch_json(f"{base_url}/health", timeout=10.0)
    prefill = health.get("prefill_batch", {})
    checks = {
        "runtime_dispatch_skipped_reason": prefill.get("runtime_dispatch_skipped_reason"),
        "selected_batch_size": prefill.get("selected_batch_size"),
        "daemon_prefill_backend": prefill.get("daemon_prefill_backend"),
        "daemon_prefill_plan": prefill.get("daemon_prefill_plan"),
        "queue_size": prefill.get("queue_size"),
        "total_batches": prefill.get("total_batches"),
        "fused_batches": prefill.get("fused_batches"),
        "last_prefill_tok_s": prefill.get("last_prefill_tok_s"),
        "pending_requests": prefill.get("pending_requests"),
        "resident_runtime_sessions": prefill.get("resident_runtime_sessions"),
    }
    if checks["runtime_dispatch_skipped_reason"] != "not_skipped":
        raise RuntimeError(f"server prefill did not dispatch: {checks}; log={log_path}")
    if checks["selected_batch_size"] != 2:
        raise RuntimeError(f"server prefill did not select a 2-request batch: {checks}; log={log_path}")
    if checks["daemon_prefill_backend"] != expected_daemon_prefill_backend:
        raise RuntimeError(f"unexpected daemon prefill backend: {checks}; log={log_path}")
    if checks["daemon_prefill_plan"] != "fused_dense_qwen35_candidate":
        raise RuntimeError(f"unexpected daemon prefill plan: {checks}; log={log_path}")
    if int(checks["total_batches"] or 0) < 1:
        raise RuntimeError(f"server prefill did not record batch telemetry: {checks}; log={log_path}")
    if int(checks["fused_batches"] or 0) < 1:
        raise RuntimeError(f"server prefill did not record fused batch telemetry: {checks}; log={log_path}")
    if float(checks["last_prefill_tok_s"] or 0) <= 0:
        raise RuntimeError(f"server prefill did not record positive prefill tok/s: {checks}; log={log_path}")
    if int(checks["pending_requests"] or 0) != 0:
        raise RuntimeError(f"server prefill left pending requests behind: {checks}; log={log_path}")
    if int(checks["resident_runtime_sessions"] or 0) != 0:
        raise RuntimeError(f"server prefill left resident runtime sessions behind: {checks}; log={log_path}")

    print(
        "server prefill coalescing smoke passed: "
        f"responses={len(responses) + 2} selected_batch_size={checks['selected_batch_size']} "
        f"backend={checks['daemon_prefill_backend']} plan={checks['daemon_prefill_plan']}"
    )
finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log_file.close()
PY
