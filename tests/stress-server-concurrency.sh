#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DAEMON="${DAEMON:-$ROOT/target/release/hipfire-daemon}"
MODEL_A="${MODEL_A:-$HOME/.hipfire/models/qwen3.5-0.8b-mq4.hfq}"
MODEL_B="${MODEL_B:-}"
REQUESTS="${HIPFIRE_STRESS_REQUESTS:-128}"
CONCURRENCY="${HIPFIRE_STRESS_CONCURRENCY:-$REQUESTS}"
MAX_TOKENS="${HIPFIRE_STRESS_MAX_TOKENS:-2}"
MAX_SEQ="${MAX_SEQ:-1536}"
DECODE_BACKEND="${HIPFIRE_QWEN35_DECODE_BATCH:-serial}"
SERVER_STRESS_LOCK="${HIPFIRE_SERVER_STRESS_LOCK:-${TMPDIR:-/tmp}/hipfire-server-stress.lock}"
SERVER_STRESS_LOCK_WAIT="${HIPFIRE_SERVER_STRESS_LOCK_WAIT:-300}"

exec 9>"$SERVER_STRESS_LOCK"
if ! flock -w "$SERVER_STRESS_LOCK_WAIT" 9; then
  echo "timed out waiting for server stress lock: $SERVER_STRESS_LOCK" >&2
  exit 2
fi

if [[ ! -x "$DAEMON" ]]; then
  echo "missing daemon binary: $DAEMON" >&2
  echo "build it with: cargo build --release -p hipfire-daemon --bin hipfire-daemon" >&2
  exit 2
fi

if [[ ! -f "$MODEL_A" ]]; then
  echo "missing MODEL_A: $MODEL_A" >&2
  exit 2
fi

if [[ -n "$MODEL_B" && ! -f "$MODEL_B" ]]; then
  echo "missing MODEL_B: $MODEL_B" >&2
  exit 2
fi

python3 - "$ROOT" "$DAEMON" "$MODEL_A" "$MODEL_B" "$REQUESTS" "$CONCURRENCY" "$MAX_TOKENS" "$MAX_SEQ" "$DECODE_BACKEND" <<'PY'
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

root, daemon, model_a, model_b, requests_s, concurrency_s, max_tokens_s, max_seq_s, decode_backend = sys.argv[1:]
request_count = int(requests_s)
concurrency = max(1, int(concurrency_s))
max_tokens = max(1, int(max_tokens_s))
max_seq = int(max_seq_s)
models = [model_a] if not model_b else [model_a, model_b]
dense_fused = decode_backend in {"auto", "fused", "fused_dense", "fused_dense_layer_chunked"}


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


def chat_request(base_url: str, index: int, barrier: threading.Barrier) -> dict[str, Any]:
    model = models[index % len(models)]
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer with one short lowercase word."},
            {"role": "user", "content": f"Return a common color word. Request {index}."},
        ],
        "stream": False,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    barrier.wait(timeout=30.0)
    try:
        out = fetch_json(f"{base_url}/v1/chat/completions", body, timeout=240.0)
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": err.code, "error": detail, "index": index, "model": model}
    except Exception as err:
        return {"ok": False, "status": 0, "error": str(err), "index": index, "model": model}
    if "error" in out:
        return {"ok": False, "status": 200, "error": str(out["error"]), "index": index, "model": model}
    choices = out.get("choices")
    content = choices[0].get("message", {}).get("content") if isinstance(choices, list) and choices else None
    if content is None:
        return {"ok": False, "status": 200, "error": f"missing content: {out}", "index": index, "model": model}
    return {"ok": True, "status": 200, "content": content, "index": index, "model": model}


port = pick_port()
base_url = f"http://127.0.0.1:{port}"
log_file = tempfile.NamedTemporaryFile("w", prefix="hipfire-server-concurrency-", suffix=".log", delete=False)
log_path = log_file.name

env = os.environ.copy()
env.update({
    "HIPFIRE_DAEMON_BIN": daemon,
    "HIPFIRE_MODEL": model_a,
    "HIPFIRE_NO_PID_FILE": "1",
    "HIPFIRE_SERVER_PREFILL_BATCH": "1",
    "HIPFIRE_SERVER_PREFILL_BATCH_MAX": env.get("HIPFIRE_SERVER_PREFILL_BATCH_MAX", "8"),
    "HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS": env.get("HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS", "250"),
    "HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE": env.get("HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE", "250"),
    "HIPFIRE_SCHED_PREFILL_MAX_QUEUED": env.get("HIPFIRE_SCHED_PREFILL_MAX_QUEUED", "256"),
    "HIPFIRE_SCHED_DECODE_MAX_ACTIVE": env.get("HIPFIRE_SCHED_DECODE_MAX_ACTIVE", "256"),
    "HIPFIRE_MAX_SEQ": str(max_seq),
    "HIPFIRE_DFLASH_DRAFT": "",
    "HIPFIRE_SERVER_PREFILL_STATE_CACHE": env.get("HIPFIRE_STRESS_STATE_CACHE", "0"),
    "HIPFIRE_SCHED_STATE_CACHE_RESIDENT": env.get("HIPFIRE_STRESS_STATE_CACHE", "0"),
    "HIPFIRE_QWEN35_PREFILL_SESSION_BATCH": env.get("HIPFIRE_QWEN35_PREFILL_SESSION_BATCH", "serial"),
    "HIPFIRE_QWEN35_DECODE_BATCH": decode_backend,
    "HIPFIRE_KV_MODE": "fp32" if dense_fused else env.get("HIPFIRE_KV_MODE", "q8"),
    "HIPFIRE_QWEN35_STATE_QUANT": "fp32" if dense_fused else env.get("HIPFIRE_QWEN35_STATE_QUANT", "q8"),
})
if "HIPFIRE_STRESS_BOUNDARY_CHECKPOINTS" in os.environ:
    env["HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS"] = os.environ["HIPFIRE_STRESS_BOUNDARY_CHECKPOINTS"]
if model_b:
    env["HIPFIRE_MAX_RESIDENT_WORKERS"] = env.get("HIPFIRE_MAX_RESIDENT_WORKERS", "2")

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
    barrier = threading.Barrier(request_count + 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(chat_request, base_url, i, barrier) for i in range(request_count)]
        barrier.wait(timeout=30.0)
        results = [future.result(timeout=300.0) for future in futures]

    def is_backpressure(result: dict[str, Any]) -> bool:
        error = str(result.get("error", "")).lower()
        return (
            result.get("status") == 503
            and ("backpressure" in error or "memory pressure" in error)
        )

    failed = [result for result in results if not result.get("ok") and not is_backpressure(result)]
    backpressured = [result for result in results if is_backpressure(result)]
    succeeded = [result for result in results if result.get("ok")]
    if failed:
        sample = failed[:5]
        raise RuntimeError(f"{len(failed)} requests failed; sample={sample}; log={log_path}")
    if not succeeded:
        raise RuntimeError(f"all requests were backpressured; sample={backpressured[:5]}; log={log_path}")

    deadline = time.time() + 30.0
    health: dict[str, Any] = {}
    while time.time() < deadline:
        health = fetch_json(f"{base_url}/health", timeout=10.0)
        prefill = health.get("prefill_batch", {})
        decode = health.get("decode_batch", {})
        if int(prefill.get("pending_requests") or 0) == 0 and int(decode.get("active_sessions") or 0) == 0:
            break
        time.sleep(0.25)

    prefill = health.get("prefill_batch", {})
    decode = health.get("decode_batch", {})
    runtime_workers = health.get("runtime_workers", {})
    if int(prefill.get("pending_requests") or 0) != 0:
        raise RuntimeError(f"pending prefill requests leaked: {prefill}; log={log_path}")
    if int(decode.get("active_sessions") or 0) != 0:
        raise RuntimeError(f"active decode sessions leaked: {decode}; log={log_path}")
    if int(prefill.get("resident_decode_sessions") or 0) != 0:
        raise RuntimeError(f"resident decode sessions leaked: {prefill}; log={log_path}")
    if model_b and int(runtime_workers.get("resident_workers") or 0) < 2:
        raise RuntimeError(f"expected two resident workers: {runtime_workers}; log={log_path}")
    if int(runtime_workers.get("total_resident_bytes") or 0) <= 0:
        raise RuntimeError(f"runtime worker memory metrics missing: {runtime_workers}; log={log_path}")

    print(
        "server concurrency stress passed: "
        f"requests={request_count} succeeded={len(succeeded)} backpressure={len(backpressured)} "
        f"concurrency={concurrency} models={len(models)} "
        f"decode_backend={decode.get('last_backend')} batches={decode.get('total_batches')} "
        f"workers={runtime_workers.get('resident_workers')} log={log_path}"
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
