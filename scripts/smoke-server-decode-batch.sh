#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DAEMON="${DAEMON:-$ROOT/target/release/examples/daemon}"
MODEL="${MODEL:-$HOME/.hipfire/models/qwen3.5-0.8b.mq4.hfq}"
MAX_SEQ="${MAX_SEQ:-512}"
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

python3 - "$ROOT" "$DAEMON" "$MODEL" "$MAX_SEQ" <<'PY'
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

root, daemon, model, max_seq_s = sys.argv[1:]
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
            {"role": "system", "content": "Answer with only one short lowercase word."},
            {"role": "user", "content": f"Return a common color word for {label}."},
        ],
        "stream": False,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 2,
        "chat_template_kwargs": {"enable_thinking": False},
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
    content = choices[0].get("message", {}).get("content")
    if content is None:
        raise RuntimeError(f"{label}: missing assistant content: {out}")
    return out


port = pick_port()
base_url = f"http://127.0.0.1:{port}"
log_file = tempfile.NamedTemporaryFile("w", prefix="hipfire-server-decode-batch-", suffix=".log", delete=False)
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
    "HIPFIRE_QWEN35_PREFILL_SESSION_BATCH": "serial",
    "HIPFIRE_QWEN35_DECODE_BATCH": "serial",
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
    decode = health.get("decode_batch", {})
    prefill = health.get("prefill_batch", {})
    checks = {
        "decode_total_batches": decode.get("total_batches"),
        "decode_serial_batches": decode.get("serial_batches"),
        "decode_selected_batch_size": decode.get("selected_batch_size"),
        "decode_last_backend": decode.get("last_backend"),
        "decode_last_decode_ms": decode.get("last_decode_ms"),
        "decode_last_skipped_reason": decode.get("last_skipped_reason"),
        "decode_active_sessions": decode.get("active_sessions"),
        "prefill_selected_batch_size": prefill.get("selected_batch_size"),
        "resident_runtime_sessions": prefill.get("resident_runtime_sessions"),
        "resident_decode_sessions": prefill.get("resident_decode_sessions"),
        "pending_requests": prefill.get("pending_requests"),
    }
    if int(checks["decode_total_batches"] or 0) < 1:
        raise RuntimeError(f"server decode did not record batch telemetry: {checks}; log={log_path}")
    if int(checks["decode_serial_batches"] or 0) < 1:
        raise RuntimeError(f"server decode did not record serial batch telemetry: {checks}; log={log_path}")
    if checks["decode_selected_batch_size"] != 2:
        raise RuntimeError(f"server decode did not select a 2-request batch: {checks}; log={log_path}")
    if checks["decode_last_backend"] != "serial_reference":
        raise RuntimeError(f"unexpected decode backend: {checks}; log={log_path}")
    if float(checks["decode_last_decode_ms"] or 0) <= 0:
        raise RuntimeError(f"server decode did not record positive decode latency: {checks}; log={log_path}")
    if checks["prefill_selected_batch_size"] != 2:
        raise RuntimeError(f"server prefill did not coalesce setup requests: {checks}; log={log_path}")
    if int(checks["decode_active_sessions"] or 0) != 0:
        raise RuntimeError(f"server decode left active pending sessions: {checks}; log={log_path}")
    if int(checks["pending_requests"] or 0) != 0:
        raise RuntimeError(f"server prefill left pending requests behind: {checks}; log={log_path}")
    if int(checks["resident_runtime_sessions"] or 0) != 0:
        raise RuntimeError(f"server prefill left resident runtime sessions behind: {checks}; log={log_path}")
    if int(checks["resident_decode_sessions"] or 0) != 0:
        raise RuntimeError(f"server decode left resident decode sessions behind: {checks}; log={log_path}")

    print(
        "server decode batching smoke passed: "
        f"responses={len(responses)} selected_batch_size={checks['decode_selected_batch_size']} "
        f"backend={checks['decode_last_backend']} log={log_path}"
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
