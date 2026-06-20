#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DAEMON="${DAEMON:-$ROOT/target/release/hipfire-daemon}"
MODEL="${MODEL:-$HOME/.hipfire/models/qwen3.5-0.8b-mq4.hfq}"
REQUESTS="${HIPFIRE_SHARED_PREFIX_REQUESTS:-2}"
MAX_SEQ="${MAX_SEQ:-768}"
SERVER_SMOKE_LOCK="${HIPFIRE_SERVER_SMOKE_LOCK:-${TMPDIR:-/tmp}/hipfire-server-smoke.lock}"
SERVER_SMOKE_LOCK_WAIT="${HIPFIRE_SERVER_SMOKE_LOCK_WAIT:-300}"

exec 9>"$SERVER_SMOKE_LOCK"
if ! flock -w "$SERVER_SMOKE_LOCK_WAIT" 9; then
  echo "timed out waiting for server smoke lock: $SERVER_SMOKE_LOCK" >&2
  exit 2
fi

if [[ ! -x "$DAEMON" ]]; then
  echo "missing daemon binary: $DAEMON" >&2
  echo "build it with: cargo build --release -p hipfire-daemon --bin hipfire-daemon" >&2
  exit 2
fi

if [[ ! -f "$MODEL" ]]; then
  echo "missing model: $MODEL" >&2
  exit 2
fi

python3 - "$ROOT" "$DAEMON" "$MODEL" "$REQUESTS" "$MAX_SEQ" <<'PY'
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

root, daemon, model, requests_s, max_seq_s = sys.argv[1:]
requests = int(requests_s)
max_seq = int(max_seq_s)
if requests < 2:
    raise RuntimeError("HIPFIRE_SHARED_PREFIX_REQUESTS must be >= 2")


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
        return json.loads(resp.read().decode("utf-8"))


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


def start_server() -> tuple[str, subprocess.Popen[str], Any, str]:
    port = pick_port()
    base_url = f"http://127.0.0.1:{port}"
    log_file = tempfile.NamedTemporaryFile("w", prefix="hipfire-server-shared-prefix-", suffix=".log", delete=False)
    log_path = log_file.name
    env = os.environ.copy()
    env.update({
        "HIPFIRE_DAEMON_BIN": daemon,
        "HIPFIRE_MODEL": model,
        "HIPFIRE_KV_MODE": "q8",
        "HIPFIRE_NO_PID_FILE": "1",
        "HIPFIRE_SERVER_PREFILL_BATCH": "1",
        "HIPFIRE_SERVER_PREFILL_BATCH_MAX": str(requests),
        "HIPFIRE_SCHED_PREFILL_BATCH_MAX": str(requests),
        "HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE": "150",
        "HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS": "150",
        "HIPFIRE_SERVER_PREFILL_STATE_CACHE": "1",
        "HIPFIRE_STATE_CACHE_MAX_CHECKPOINTS": str(max(8, requests + 2)),
        "HIPFIRE_MAX_SEQ": str(max_seq),
        "HIPFIRE_DFLASH_DRAFT": "",
        "HIPFIRE_SERVER_PREFILL_SHARED_PREFIX_FANOUT": "1",
    })
    proc = subprocess.Popen(
        ["cargo", "run", "-q", "-p", "hipfire-cli", "--", "serve", "--host", "127.0.0.1", "--port", str(port)],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=log_file,
        text=True,
        env=env,
    )
    return base_url, proc, log_file, log_path


def stop_server(proc: subprocess.Popen[str], log_file: Any) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log_file.close()


def chat_request(base_url: str, idx: int) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Document: The access code is blue. The launch window is dawn. "
                    "The fallback contact is Mira. Return only the access code color."
                ),
            },
        ],
        "stream": False,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 1,
        "prompt_cache_retention": "in_memory",
    }
    try:
        out = fetch_json(f"{base_url}/v1/chat/completions", body, timeout=180.0)
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"request {idx}: HTTP {err.code}: {detail}") from err
    if "error" in out:
        raise RuntimeError(f"request {idx}: response error: {out['error']}")
    if out.get("choices", [{}])[0].get("message", {}).get("content") is None:
        raise RuntimeError(f"request {idx}: missing content: {out}")
    return out


base_url, proc, log_file, log_path = start_server()
try:
    initial = wait_health(base_url, proc, log_path)
    prefill = initial.get("prefill_batch", {})
    if prefill.get("generate_batch_prefill_capability") != "supported":
        raise RuntimeError(f"server prefill capability not supported after warmup: {prefill}; log={log_path}")

    start = time.perf_counter()
    start_barrier = None
    if requests == 2:
        import threading
        start_barrier = threading.Barrier(3)

    def synchronized_request(idx: int) -> dict[str, Any]:
        if start_barrier is not None:
            start_barrier.wait(timeout=10.0)
        return chat_request(base_url, idx)

    with concurrent.futures.ThreadPoolExecutor(max_workers=requests) as pool:
        futures = [pool.submit(synchronized_request, idx) for idx in range(requests)]
        if start_barrier is not None:
            start_barrier.wait(timeout=10.0)
        responses = [future.result() for future in futures]
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    health = fetch_json(f"{base_url}/health", timeout=10.0)
    state_cache = health.get("state_cache", {})
    prefill = health.get("prefill_batch", {})
    followers = int(state_cache.get("shared_prefix_fanout_followers") or 0)
    groups = int(state_cache.get("shared_prefix_fanout_groups") or 0)
    runtime_hits = int(prefill.get("runtime_cache_hits") or 0)
    if groups < 1:
        raise RuntimeError(f"shared-prefix fanout group did not run: {state_cache}; log={log_path}")
    if followers < 1:
        raise RuntimeError(f"shared-prefix fanout did not attach any followers: state_cache={state_cache}; log={log_path}")
    if runtime_hits < followers:
        raise RuntimeError(f"runtime cache hit telemetry too low after fanout: {prefill}; log={log_path}")
    if int(prefill.get("resident_decode_sessions") or 0) != 0:
        raise RuntimeError(f"decode sessions leaked after fanout smoke: {prefill}; log={log_path}")
    print(
        "server shared-prefix fanout smoke passed: "
        f"requests={len(responses)} elapsed_ms={elapsed_ms:.1f} "
        f"fanout_groups={groups} fanout_followers={followers} "
        f"runtime_cache_hits={runtime_hits} log={log_path}"
    )
finally:
    stop_server(proc, log_file)
PY
