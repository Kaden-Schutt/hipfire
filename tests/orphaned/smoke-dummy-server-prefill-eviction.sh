#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${HIPFIRE_DUMMY_EVICTION_PORT:-11438}"
LOG_PATH="${TMPDIR:-/tmp}/hipfire-dummy-prefill-eviction-${PORT}.log"
SERVER_SMOKE_LOCK="${HIPFIRE_SERVER_SMOKE_LOCK:-${TMPDIR:-/tmp}/hipfire-server-smoke.lock}"
SERVER_SMOKE_LOCK_WAIT="${HIPFIRE_SERVER_SMOKE_LOCK_WAIT:-300}"

exec 9>"$SERVER_SMOKE_LOCK"
if ! flock -w "$SERVER_SMOKE_LOCK_WAIT" 9; then
    echo "timed out waiting for server smoke lock: $SERVER_SMOKE_LOCK" >&2
    exit 2
fi

python3 - "$ROOT" "$PORT" "$LOG_PATH" <<'PY'
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

root, port, log_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
base_url = f"http://127.0.0.1:{port}"


def fetch_json(url, *, data=None, headers=None, timeout=5.0):
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers or {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_health(proc):
    deadline = time.time() + 20
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with {proc.returncode}; log={log_path}")
        try:
            return fetch_json(f"{base_url}/health", timeout=1.0)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.1)
    raise RuntimeError(f"server did not become healthy; log={log_path}")


def chat_request(content):
    body = {
        "model": "hipfire:dummy",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2,
        "temperature": 0,
    }
    return fetch_json(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        timeout=20.0,
    )


env = os.environ.copy()
env.update({
    "HIPFIRE_MODEL": "hipfire:dummy",
    "HIPFIRE_NO_PID_FILE": "1",
    "HIPFIRE_SERVER_PREFILL_BATCH": "1",
    "HIPFIRE_SERVER_PREFILL_BATCH_MAX": "2",
    "HIPFIRE_SCHED_PREFILL_BATCH_MAX": "2",
    "HIPFIRE_SCHED_RESIDENT_STATE_MAX": "1",
    "HIPFIRE_SCHED_SPILLABLE_BATCH_MAX": "2",
    "HIPFIRE_SCHED_STATE_CACHE_DISK": "1",
    "HIPFIRE_SCHED_STATE_CACHE_DISK_MIN_PRIORITY": "64",
    "HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS": "250",
    "HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE": "250",
    "HIPFIRE_DUMMY_PREFILL_DELAY_MS": "50",
    "HIPFIRE_DUMMY_GENERATE_DELAY_MS": "8",
})

log_file = open(log_path, "w", encoding="utf-8")
proc = subprocess.Popen(
    ["cargo", "run", "-q", "-p", "hipfire-cli", "--", "serve", "--host", "127.0.0.1", "--port", str(port)],
    cwd=root,
    stdin=subprocess.DEVNULL,
    stdout=log_file,
    stderr=log_file,
    text=True,
    env=env,
)

try:
    wait_health(proc)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(chat_request, ["evict-a", "evict-b"]))

    for response in responses:
        content = response["choices"][0]["message"]["content"]
        if content.count("dummy:") < 2:
            raise RuntimeError(f"dummy response did not contain two counter tokens: {response}")

    health = fetch_json(f"{base_url}/health", timeout=10.0)
    prefill = health.get("prefill_batch", {})
    state_cache = health.get("state_cache", {})
    checks = {
        "selected_batch_size": prefill.get("selected_batch_size"),
        "resident_state_limit": prefill.get("resident_state_limit"),
        "spillable_batch_max": prefill.get("spillable_batch_max"),
        "spillable_sessions": prefill.get("spillable_sessions"),
        "state_cache_disk": prefill.get("state_cache_disk"),
        "disk_spill_allowed": prefill.get("disk_spill_allowed"),
        "evictions": prefill.get("state_cache_evictions_total"),
        "recompute": prefill.get("state_cache_recompute_required_total"),
        "state_cache_evictions": state_cache.get("evictions_total"),
        "state_cache_recompute": state_cache.get("recompute_required_total"),
        "pending_requests": prefill.get("pending_requests"),
        "resident_runtime_sessions": prefill.get("resident_runtime_sessions"),
    }
    if checks["selected_batch_size"] != 2:
        raise RuntimeError(f"eviction smoke did not select a 2-request batch: {checks}; log={log_path}")
    if checks["resident_state_limit"] != 1 or checks["spillable_batch_max"] != 2:
        raise RuntimeError(f"eviction smoke reported wrong state limits: {checks}; log={log_path}")
    if checks["spillable_sessions"] != 1:
        raise RuntimeError(f"eviction smoke did not report one spillable session: {checks}; log={log_path}")
    if checks["state_cache_disk"] is not True or checks["disk_spill_allowed"] is not True:
        raise RuntimeError(f"eviction smoke did not enable disk-spill policy: {checks}; log={log_path}")
    if checks["evictions"] != 1 or checks["recompute"] != 1:
        raise RuntimeError(f"eviction smoke reported wrong prefill eviction counters: {checks}; log={log_path}")
    if checks["state_cache_evictions"] != 1 or checks["state_cache_recompute"] != 1:
        raise RuntimeError(f"eviction smoke reported wrong state-cache counters: {checks}; log={log_path}")
    if int(checks["pending_requests"] or 0) != 0:
        raise RuntimeError(f"eviction smoke left pending requests: {checks}; log={log_path}")
    if int(checks["resident_runtime_sessions"] or 0) != 0:
        raise RuntimeError(f"eviction smoke left resident runtime sessions: {checks}; log={log_path}")

    print("dummy server prefill eviction smoke passed")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log_file.close()
PY
