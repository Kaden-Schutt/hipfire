#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DAEMON="${DAEMON:-$ROOT/target/release/examples/daemon}"
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

python3 - "$ROOT" "$DAEMON" <<'PY'
import concurrent.futures
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any

root, daemon = sys.argv[1:]
model = "hipfire:dummy"


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


def chat_request(base_url: str, label: str, max_tokens: int = 2) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": f"dummy prompt for {label}"}],
        "stream": False,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
    }
    try:
        out = fetch_json(f"{base_url}/v1/chat/completions", body, timeout=120.0)
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{label}: HTTP {err.code}: {detail}") from err
    if "error" in out:
        raise RuntimeError(f"{label}: response error: {out['error']}")
    content = out.get("choices", [{}])[0].get("message", {}).get("content", "")
    if "dummy:" not in content:
        raise RuntimeError(f"{label}: missing dummy counter tokens in response: {out}")
    return out


def streaming_chat_request(base_url: str, label: str) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": f"dummy streaming prompt for {label}"}],
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
    if "data:" not in raw or "dummy:" not in raw:
        raise RuntimeError(f"{label}: malformed streaming dummy response: {raw!r}")
    return raw


port = pick_port()
base_url = f"http://127.0.0.1:{port}"
log_file = tempfile.NamedTemporaryFile("w", prefix="hipfire-dummy-prefill-", suffix=".log", delete=False)
log_path = log_file.name

env = os.environ.copy()
env.update({
    "HIPFIRE_DAEMON_BIN": daemon,
    "HIPFIRE_MODEL": model,
    "HIPFIRE_NO_PID_FILE": "1",
    "HIPFIRE_SERVER_PREFILL_BATCH": "1",
    "HIPFIRE_SERVER_PREFILL_BATCH_MAX": "2",
    "HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS": "250",
    "HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE": "250",
    "HIPFIRE_DUMMY_PREFILL_DELAY_MS": "50",
    "HIPFIRE_DUMMY_GENERATE_DELAY_MS": "8",
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
        raise RuntimeError(f"dummy server prefill capability not supported after warmup: {prefill}; log={log_path}")

    chat_request(base_url, "single-timeout", max_tokens=1)
    single_health = fetch_json(f"{base_url}/health", timeout=10.0)
    single_prefill = single_health.get("prefill_batch", {})
    if single_prefill.get("runtime_dispatch_skipped_reason") != "daemon_serial_prefill_timeout":
        raise RuntimeError(f"single request did not exercise timeout fallback: {single_prefill}; log={log_path}")
    if int(single_prefill.get("pending_requests") or 0) != 0:
        raise RuntimeError(f"single request left pending requests: {single_prefill}; log={log_path}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(chat_request, base_url, "request-a", 2),
            pool.submit(chat_request, base_url, "request-b", 2),
        ]
        responses = [future.result() for future in futures]

    for response in responses:
        content = response["choices"][0]["message"]["content"]
        if content.count("dummy:") < 2:
            raise RuntimeError(f"dummy response did not contain two counter tokens: {response}")

    health = fetch_json(f"{base_url}/health", timeout=10.0)
    prefill = health.get("prefill_batch", {})
    checks = {
        "runtime_dispatch_skipped_reason": prefill.get("runtime_dispatch_skipped_reason"),
        "selected_batch_size": prefill.get("selected_batch_size"),
        "daemon_prefill_backend": prefill.get("daemon_prefill_backend"),
        "daemon_prefill_plan": prefill.get("daemon_prefill_plan"),
        "pending_requests": prefill.get("pending_requests"),
        "resident_runtime_sessions": prefill.get("resident_runtime_sessions"),
        "resident_state_limit": prefill.get("resident_state_limit"),
        "spillable_batch_max": prefill.get("spillable_batch_max"),
        "spillable_sessions": prefill.get("spillable_sessions"),
        "state_cache_disk": prefill.get("state_cache_disk"),
        "disk_spill_allowed": prefill.get("disk_spill_allowed"),
    }
    if checks["runtime_dispatch_skipped_reason"] != "not_skipped":
        raise RuntimeError(f"dummy prefill did not dispatch: {checks}; log={log_path}")
    if checks["selected_batch_size"] != 2:
        raise RuntimeError(f"dummy prefill did not select a 2-request batch: {checks}; log={log_path}")
    if checks["daemon_prefill_backend"] != "dummy_delay":
        raise RuntimeError(f"unexpected dummy backend: {checks}; log={log_path}")
    if checks["daemon_prefill_plan"] != "dummy_counter":
        raise RuntimeError(f"unexpected dummy plan: {checks}; log={log_path}")
    if int(checks["pending_requests"] or 0) != 0:
        raise RuntimeError(f"dummy prefill left pending requests: {checks}; log={log_path}")
    if int(checks["resident_runtime_sessions"] or 0) != 0:
        raise RuntimeError(f"dummy prefill left resident runtime sessions: {checks}; log={log_path}")
    if int(checks["resident_state_limit"] or 0) != 2:
        raise RuntimeError(f"dummy prefill reported wrong resident state limit: {checks}; log={log_path}")
    if int(checks["spillable_batch_max"] or 0) != 2:
        raise RuntimeError(f"dummy prefill reported wrong spillable batch max: {checks}; log={log_path}")
    if int(checks["spillable_sessions"] or 0) != 0:
        raise RuntimeError(f"dummy prefill reported spillable sessions by default: {checks}; log={log_path}")
    if checks["state_cache_disk"] is not False or checks["disk_spill_allowed"] is not False:
        raise RuntimeError(f"dummy prefill enabled disk spill by default: {checks}; log={log_path}")

    streaming_chat_request(base_url, "stream")
    stream_health = fetch_json(f"{base_url}/health", timeout=10.0)
    stream_prefill = stream_health.get("prefill_batch", {})
    if int(stream_prefill.get("pending_requests") or 0) != 0:
        raise RuntimeError(f"streaming request left pending requests: {stream_prefill}; log={log_path}")
    if int(stream_prefill.get("resident_runtime_sessions") or 0) != 0:
        raise RuntimeError(f"streaming request left resident sessions: {stream_prefill}; log={log_path}")

    print("dummy server prefill smoke passed")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log_file.close()
PY
