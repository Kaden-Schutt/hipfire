#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DAEMON="${DAEMON:-$ROOT/target/release/hipfire-daemon}"
MODEL="${MODEL:-$HOME/.hipfire/models/qwen3.5-0.8b-mq4.hfq}"
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
require_prefix_preflight = os.environ.get("HIPFIRE_REQUIRE_PREFIX_PREFLIGHT") == "1"


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


def chat_request(base_url: str, label: str, user_prompt: str | None = None) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer with one lowercase color word."},
            {"role": "user", "content": user_prompt or "Return the color of a clear daytime sky."},
        ],
        "stream": False,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 1,
        "prompt_cache_retention": "in_memory",
    }
    try:
        out = fetch_json(f"{base_url}/v1/chat/completions", body, timeout=120.0)
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{label}: HTTP {err.code}: {detail}") from err
    if "error" in out:
        raise RuntimeError(f"{label}: response error: {out['error']}")
    return out


def start_server(
    corrupt_prefix_hash_once: bool = False,
    max_checkpoints: str = "4",
) -> tuple[str, subprocess.Popen[str], Any, str]:
    port = pick_port()
    base_url = f"http://127.0.0.1:{port}"
    log_file = tempfile.NamedTemporaryFile("w", prefix="hipfire-server-prefix-cache-", suffix=".log", delete=False)
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
        "HIPFIRE_STATE_CACHE_MAX_CHECKPOINTS": max_checkpoints,
        "HIPFIRE_MAX_SEQ": str(max_seq),
        "HIPFIRE_DFLASH_DRAFT": "",
        "HIPFIRE_QWEN35_PREFILL_SESSION_BATCH": "serial",
    })
    if corrupt_prefix_hash_once:
        env["HIPFIRE_DEBUG_CORRUPT_PREFIX_HASH_ONCE"] = "1"

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


def assert_response_content(label: str, response: dict[str, Any]) -> None:
    if response.get("choices", [{}])[0].get("message", {}).get("content") is None:
        raise RuntimeError(f"{label} response missing content: {response}")


def run_reuse_scenario() -> str:
    base_url, proc, log_file, log_path = start_server()
    try:
        initial = wait_health(base_url, proc, log_path)
        prefill = initial.get("prefill_batch", {})
        if prefill.get("generate_batch_prefill_capability") != "supported":
            raise RuntimeError(f"server prefill capability not supported after warmup: {prefill}; log={log_path}")

        first = chat_request(base_url, "reuse first")
        first_health = fetch_json(f"{base_url}/health", timeout=10.0)
        first_prefill = first_health.get("prefill_batch", {})
        first_state_cache = first_health.get("state_cache", {})
        if int(first_prefill.get("resident_checkpoints") or 0) < 1:
            raise RuntimeError(f"first request did not leave a resident checkpoint: {first_prefill}; log={log_path}")
        if int(first_prefill.get("runtime_cache_hits") or 0) != 0:
            raise RuntimeError(f"first request unexpectedly reported a runtime cache hit: {first_prefill}; log={log_path}")
        if first_state_cache.get("daemon_prefix_hash") is not True:
            raise RuntimeError(f"first request did not expose daemon prefix hash telemetry: {first_state_cache}; log={log_path}")

        second = chat_request(base_url, "reuse second")
        second_health = fetch_json(f"{base_url}/health", timeout=10.0)
        second_prefill = second_health.get("prefill_batch", {})
        second_state_cache = second_health.get("state_cache", {})
        if int(second_prefill.get("runtime_cache_hits") or 0) < 1:
            raise RuntimeError(f"second request did not attach resident checkpoint: {second_prefill}; log={log_path}")
        if require_prefix_preflight:
            if int(second_state_cache.get("prefix_hash_preflight_requests") or 0) < 1:
                raise RuntimeError(f"second request did not run prefix hash preflight: {second_state_cache}; log={log_path}")
            if int(second_state_cache.get("prefix_hash_preflight_matches") or 0) < 1:
                raise RuntimeError(f"second request did not match through prefix hash preflight: {second_state_cache}; log={log_path}")
        if int(second_prefill.get("resident_decode_sessions") or 0) != 0:
            raise RuntimeError(f"decode session leaked after cached request: {second_prefill}; log={log_path}")
        assert_response_content("reuse first", first)
        assert_response_content("reuse second", second)
        return (
            "reuse: "
            f"runtime_cache_hits={second_prefill.get('runtime_cache_hits')} "
            f"resident_checkpoints={second_prefill.get('resident_checkpoints')} "
            f"log={log_path}"
        )
    finally:
        stop_server(proc, log_file)


def run_mismatch_scenario() -> str:
    base_url, proc, log_file, log_path = start_server(corrupt_prefix_hash_once=True)
    try:
        initial = wait_health(base_url, proc, log_path)
        prefill = initial.get("prefill_batch", {})
        if prefill.get("generate_batch_prefill_capability") != "supported":
            raise RuntimeError(f"server prefill capability not supported after warmup: {prefill}; log={log_path}")

        first = chat_request(base_url, "mismatch first")
        first_health = fetch_json(f"{base_url}/health", timeout=10.0)
        first_prefill = first_health.get("prefill_batch", {})
        first_state_cache = first_health.get("state_cache", {})
        if int(first_prefill.get("resident_checkpoints") or 0) < 1:
            raise RuntimeError(f"mismatch setup did not leave a resident checkpoint: {first_prefill}; log={log_path}")
        if first_state_cache.get("daemon_prefix_hash") is not True:
            raise RuntimeError(f"mismatch setup did not expose daemon prefix hash telemetry: {first_state_cache}; log={log_path}")

        second = chat_request(base_url, "mismatch second")
        second_health = fetch_json(f"{base_url}/health", timeout=10.0)
        second_prefill = second_health.get("prefill_batch", {})
        second_state_cache = second_health.get("state_cache", {})
        hits_after_mismatch = int(second_prefill.get("runtime_cache_hits") or 0)
        evictions_after_mismatch = int(second_state_cache.get("evictions_total") or 0)
        if hits_after_mismatch < 1:
            raise RuntimeError(f"corrupted hash did not attempt a runtime attach: {second_prefill}; log={log_path}")
        if evictions_after_mismatch < 1:
            raise RuntimeError(f"corrupted hash did not invalidate the cached manifest: {second_state_cache}; log={log_path}")

        third = chat_request(base_url, "mismatch third")
        third_health = fetch_json(f"{base_url}/health", timeout=10.0)
        third_prefill = third_health.get("prefill_batch", {})
        hits_after_third = int(third_prefill.get("runtime_cache_hits") or 0)
        if hits_after_third != hits_after_mismatch:
            raise RuntimeError(
                "stale corrupted manifest was retried: "
                f"hits_after_mismatch={hits_after_mismatch} hits_after_third={hits_after_third}; "
                f"prefill={third_prefill}; log={log_path}"
            )
        assert_response_content("mismatch first", first)
        assert_response_content("mismatch second", second)
        assert_response_content("mismatch third", third)
        return (
            "mismatch: "
            f"runtime_cache_hits={hits_after_third} "
            f"evictions_total={evictions_after_mismatch} "
            f"log={log_path}"
        )
    finally:
        stop_server(proc, log_file)


def run_cap_scenario() -> str:
    base_url, proc, log_file, log_path = start_server(max_checkpoints="1")
    try:
        initial = wait_health(base_url, proc, log_path)
        prefill = initial.get("prefill_batch", {})
        if prefill.get("generate_batch_prefill_capability") != "supported":
            raise RuntimeError(f"server prefill capability not supported after warmup: {prefill}; log={log_path}")

        first = chat_request(base_url, "cap first", "Return the color of a clear daytime sky.")
        first_health = fetch_json(f"{base_url}/health", timeout=10.0)
        first_state_cache = first_health.get("state_cache", {})
        if int(first_state_cache.get("resident_checkpoints") or 0) != 1:
            raise RuntimeError(f"cap setup did not leave exactly one resident checkpoint: {first_state_cache}; log={log_path}")

        second = chat_request(base_url, "cap second", "Return the color of healthy grass.")
        second_health = fetch_json(f"{base_url}/health", timeout=10.0)
        second_state_cache = second_health.get("state_cache", {})
        resident = int(second_state_cache.get("resident_checkpoints") or 0)
        evictions = int(second_state_cache.get("evictions_total") or 0)
        recompute = int(second_state_cache.get("recompute_required_total") or 0)
        if int(second_state_cache.get("resident_checkpoint_max") or 0) != 1:
            raise RuntimeError(f"cap scenario did not report resident checkpoint max=1: {second_state_cache}; log={log_path}")
        if resident > 1:
            raise RuntimeError(f"checkpoint cap did not bound resident checkpoints: {second_state_cache}; log={log_path}")
        if evictions < 1 or recompute < 1:
            raise RuntimeError(f"checkpoint cap did not report eviction/recompute telemetry: {second_state_cache}; log={log_path}")
        assert_response_content("cap first", first)
        assert_response_content("cap second", second)
        return (
            "cap: "
            f"resident_checkpoints={resident} "
            f"evictions_total={evictions} "
            f"recompute_required_total={recompute} "
            f"log={log_path}"
        )
    finally:
        stop_server(proc, log_file)


reuse_summary = run_reuse_scenario()
mismatch_summary = run_mismatch_scenario()
cap_summary = run_cap_scenario()
print(f"server prefix checkpoint reuse smoke passed: {reuse_summary}; {mismatch_summary}; {cap_summary}")
PY
