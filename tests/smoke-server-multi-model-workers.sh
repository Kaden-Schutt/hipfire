#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DAEMON="${DAEMON:-$ROOT/target/release/hipfire-daemon}"
MODEL_A="${MODEL_A:-$HOME/.hipfire/models/qwen3.5-0.8b-mq6.hfq}"
MODEL_B="${MODEL_B:-$HOME/.hipfire/models/qwen3.5-0.8b-mq4.hfq}"
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
  echo "build it with: cargo build --release -p hipfire-daemon --bin hipfire-daemon" >&2
  exit 2
fi

if [[ ! -f "$MODEL_A" || ! -f "$MODEL_B" ]]; then
  echo "missing MODEL_A or MODEL_B: $MODEL_A / $MODEL_B" >&2
  exit 2
fi

python3 - "$ROOT" "$DAEMON" "$MODEL_A" "$MODEL_B" "$MAX_SEQ" <<'PY'
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from typing import Any

root, daemon, model_a, model_b, max_seq_s = sys.argv[1:]
max_seq = int(max_seq_s)


def pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def fetch_json(url: str, body: dict[str, Any] | None = None, timeout: float = 60.0) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_health(base_url: str, proc: subprocess.Popen[str], log_path: str) -> dict[str, Any]:
    deadline = time.time() + 120.0
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}; log={log_path}")
        try:
            health = fetch_json(f"{base_url}/health", timeout=2.0)
            if health.get("status") == "ok":
                return health
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError(f"server did not become healthy; log={log_path}")


def chat(base_url: str, model: str, label: str) -> None:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer with one lowercase word."},
            {"role": "user", "content": f"Say a common color for {label}."},
        ],
        "stream": False,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    out = fetch_json(f"{base_url}/v1/chat/completions", body, timeout=120.0)
    if "error" in out:
        raise RuntimeError(f"{label}: response error: {out['error']}")


def run_server_case(max_workers: int, requests: list[tuple[str, str]], expected_paths: set[str]) -> str:
    port = pick_port()
    base_url = f"http://127.0.0.1:{port}"
    log_file = tempfile.NamedTemporaryFile("w", prefix="hipfire-server-multi-model-workers-", suffix=".log", delete=False)
    log_path = log_file.name
    env = os.environ.copy()
    env.update({
        "HIPFIRE_DAEMON_BIN": daemon,
        "HIPFIRE_MODEL": model_a,
        "HIPFIRE_NO_PID_FILE": "1",
        "HIPFIRE_MAX_RESIDENT_WORKERS": str(max_workers),
        "HIPFIRE_SERVER_PREFILL_BATCH": "0",
        "HIPFIRE_KV_MODE": "q8",
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
        wait_health(base_url, proc, log_path)
        for model, label in requests:
            chat(base_url, model, label)
        health = fetch_json(f"{base_url}/health", timeout=10.0)
        workers = health.get("runtime_workers", {})
        worker_list = workers.get("workers", [])
        if workers.get("resident_workers") != len(expected_paths) or len(worker_list) != len(expected_paths):
            raise RuntimeError(f"expected {len(expected_paths)} resident workers: runtime_workers={workers}; log={log_path}")
        paths = {worker.get("model_path") for worker in worker_list}
        if paths != expected_paths:
            raise RuntimeError(f"resident worker paths mismatch: paths={paths}; expected={expected_paths}; log={log_path}")
        if workers.get("max_resident_workers") != max_workers:
            raise RuntimeError(f"max resident worker telemetry mismatch: {workers}; log={log_path}")
        by_path = {worker.get("model_path"): worker for worker in worker_list}
        for path in expected_paths:
            worker = by_path.get(path, {})
            if int(worker.get("model_weight_bytes") or 0) <= 0:
                raise RuntimeError(f"worker missing nonzero model_weight_bytes for {path}: {worker}; log={log_path}")
            if int(worker.get("total_resident_bytes") or 0) < int(worker.get("model_weight_bytes") or 0):
                raise RuntimeError(f"worker total_resident_bytes below model_weight_bytes for {path}: {worker}; log={log_path}")
        aggregate_model_bytes = int(workers.get("total_model_weight_bytes") or 0)
        summed_model_bytes = sum(int(worker.get("model_weight_bytes") or 0) for worker in worker_list)
        if aggregate_model_bytes != summed_model_bytes:
            raise RuntimeError(f"aggregate model bytes mismatch: aggregate={aggregate_model_bytes} sum={summed_model_bytes}; log={log_path}")
        return log_path
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()


multi_log = run_server_case(
    2,
    [(model_a, "model-a-first"), (model_b, "model-b"), (model_a, "model-a-second")],
    {model_a, model_b},
)
evict_log = run_server_case(
    1,
    [(model_a, "evict-model-a-first"), (model_b, "evict-model-b")],
    {model_b},
)
print(f"server multi-model worker smoke passed: resident_workers=2 log={multi_log}; cap1_eviction_log={evict_log}")
PY
