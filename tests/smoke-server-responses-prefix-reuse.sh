#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DAEMON="${DAEMON:-$ROOT/target/release/examples/daemon}"
MODEL="${MODEL:-$HOME/.hipfire/models/qwen3.5-0.8b-mq4.hfq}"
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
  echo "build it with: cargo build --release -p hipfire-runtime --example daemon" >&2
  exit 2
fi

if [[ ! -f "$MODEL" ]]; then
  echo "missing model: $MODEL" >&2
  exit 2
fi

python3 - "$ROOT" "$DAEMON" "$MODEL" "$MAX_SEQ" <<'PY'
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
    log_file = tempfile.NamedTemporaryFile("w", prefix="hipfire-server-responses-prefix-", suffix=".log", delete=False)
    log_path = log_file.name
    env = os.environ.copy()
    env.update({
        "HIPFIRE_DAEMON_BIN": daemon,
        "HIPFIRE_MODEL": model,
        "HIPFIRE_KV_MODE": "q8",
        "HIPFIRE_NO_PID_FILE": "1",
        "HIPFIRE_SERVER_PREFILL_BATCH": "1",
        "HIPFIRE_SERVER_PREFILL_BATCH_MAX": "1",
        "HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS": "0",
        "HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE": "0",
        "HIPFIRE_SERVER_PREFILL_STATE_CACHE": "1",
        "HIPFIRE_STATE_CACHE_MAX_CHECKPOINTS": "8",
        "HIPFIRE_MAX_SEQ": str(max_seq),
        "HIPFIRE_DFLASH_DRAFT": "",
        "HIPFIRE_QWEN35_PREFILL_SESSION_BATCH": "serial",
        "HIPFIRE_JINJA_CHAT": "1",
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
    return base_url, proc, log_file, log_path


def stop_server(proc: subprocess.Popen[str], log_file: Any) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log_file.close()


def responses_request(base_url: str, label: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        out = fetch_json(f"{base_url}/v1/responses", body, timeout=120.0)
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{label}: HTTP {err.code}: {detail}") from err
    if "error" in out:
        raise RuntimeError(f"{label}: response error: {out['error']}")
    if not out.get("id", "").startswith("resp_"):
        raise RuntimeError(f"{label}: missing response id: {out}")
    if out.get("output", [{}])[0].get("content", [{}])[0].get("text") is None:
        raise RuntimeError(f"{label}: missing output text: {out}")
    return out


base_url, proc, log_file, log_path = start_server()
try:
    initial = wait_health(base_url, proc, log_path)
    prefill = initial.get("prefill_batch", {})
    if prefill.get("generate_batch_prefill_capability") != "supported":
        raise RuntimeError(f"server prefill capability not supported after warmup: {prefill}; log={log_path}")

    document = (
        "Document: The launch window opens at dawn. The access code is blue. "
        "The fallback contact is Mira. Answer later questions with one word."
    )
    first = responses_request(base_url, "responses first", {
        "model": model,
        "input": [
            {"role": "system", "content": "Answer with one lowercase word."},
            {"role": "user", "content": [{"type": "input_text", "text": document}]},
        ],
        "temperature": 0,
        "top_p": 1,
        "max_output_tokens": 1,
        "prompt_cache_retention": "in_memory",
    })
    first_health = fetch_json(f"{base_url}/health", timeout=10.0)
    first_state_cache = first_health.get("state_cache", {})
    if int(first_state_cache.get("resident_checkpoints") or 0) < 1:
        raise RuntimeError(f"first response did not leave a resident checkpoint: {first_state_cache}; log={log_path}")
    if int(first_state_cache.get("responses_stored_contexts") or 0) < 1:
        raise RuntimeError(f"first response did not store response context: {first_state_cache}; log={log_path}")

    second = responses_request(base_url, "responses second", {
        "model": model,
        "previous_response_id": first["id"],
        "input": [{"role": "user", "content": "What is the access code?"}],
        "temperature": 0,
        "top_p": 1,
        "max_output_tokens": 1,
    })
    second_health = fetch_json(f"{base_url}/health", timeout=10.0)
    second_prefill = second_health.get("prefill_batch", {})
    second_state_cache = second_health.get("state_cache", {})
    if int(second_state_cache.get("responses_previous_response_hits") or 0) < 1:
        raise RuntimeError(f"previous_response_id was not recorded as a hit: {second_state_cache}; log={log_path}")
    if int(second_prefill.get("runtime_cache_hits") or 0) < 1:
        raise RuntimeError(f"responses follow-up did not attach a resident checkpoint: {second_prefill}; log={log_path}")
    if int(second_state_cache.get("prefix_hash_preflight_matches") or 0) < 1:
        raise RuntimeError(f"responses follow-up did not match daemon prefix hash: {second_state_cache}; log={log_path}")
    if int(second_prefill.get("resident_decode_sessions") or 0) != 0:
        raise RuntimeError(f"decode session leaked after responses cached request: {second_prefill}; log={log_path}")
    print(
        "server responses prefix reuse smoke passed: "
        f"previous_response_hits={second_state_cache.get('responses_previous_response_hits')} "
        f"runtime_cache_hits={second_prefill.get('runtime_cache_hits')} "
        f"response_id={second.get('id')} "
        f"log={log_path}"
    )
finally:
    stop_server(proc, log_file)
PY
