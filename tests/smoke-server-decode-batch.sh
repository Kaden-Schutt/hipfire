#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DAEMON="${DAEMON:-$ROOT/target/release/hipfire-daemon}"
MODEL="${MODEL:-$HOME/.hipfire/models/qwen3.5-0.8b-mq4.hfq}"
MAX_SEQ="${MAX_SEQ:-512}"
DECODE_BACKEND="${HIPFIRE_QWEN35_DECODE_BATCH:-serial}"
EXPECTED_DECODE_BACKEND="${EXPECTED_DECODE_BACKEND:-}"
SERVER_SMOKE_LOCK="${HIPFIRE_SERVER_SMOKE_LOCK:-${TMPDIR:-/tmp}/hipfire-server-smoke.lock}"
SERVER_SMOKE_LOCK_WAIT="${HIPFIRE_SERVER_SMOKE_LOCK_WAIT:-300}"

# Set HIPFIRE_DECODE_BATCH_GROUPED_PARITY_MATRIX=1 with
# HIPFIRE_QWEN35_DECODE_BATCH=fused_grouped_moe to compare serial_reference
# against native grouped-MoE decode at B=2/4/8. B=4 and B=8 force chunk size
# 2 so the smoke covers multi-chunk native grouped-MoE advancement.
# Set HIPFIRE_DECODE_BATCH_GROUPED_PARITY_CHUNK_SIZE to override the matrix
# chunk size for latency experiments; the default stays 2 for coverage.

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

python3 - "$ROOT" "$DAEMON" "$MODEL" "$MAX_SEQ" "$DECODE_BACKEND" "$EXPECTED_DECODE_BACKEND" <<'PY'
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
from math import ceil
from typing import Any

root, daemon, model, max_seq_s, decode_backend, expected_decode_backend = sys.argv[1:]
max_seq = int(max_seq_s)
if not expected_decode_backend:
    if decode_backend in {"auto", "fused", "fused_dense", "fused_dense_layer_chunked"}:
        expected_decode_backend = "fused_dense_layer_chunked"
    elif decode_backend in {"fused_grouped_moe", "grouped_moe", "fused_grouped_moe_layer_chunked"}:
        expected_decode_backend = "fused_grouped_moe_layer_chunked"
    else:
        expected_decode_backend = "serial_reference"
default_request_max_tokens = "1" if expected_decode_backend in {"fused_dense_layer_chunked", "fused_grouped_moe_layer_chunked"} else "2"
request_max_tokens = int(os.environ.get("HIPFIRE_DECODE_BATCH_MAX_TOKENS", default_request_max_tokens))
requested_decode_chunk_size = int(os.environ.get("HIPFIRE_QWEN35_DECODE_BATCH_MAX", "0") or "0")
request_count = int(os.environ.get("HIPFIRE_DECODE_BATCH_REQUESTS", "2"))
if request_count < 2:
    raise RuntimeError("HIPFIRE_DECODE_BATCH_REQUESTS must be >= 2")
dense_fused_mode = expected_decode_backend == "fused_dense_layer_chunked"
native_multirow_enabled = os.environ.get("HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW", "").lower() in {"1", "true", "yes", "on"}


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
        "max_tokens": request_max_tokens,
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


def response_contents(responses: list[dict[str, Any]]) -> list[str]:
    return [
        str(response["choices"][0]["message"]["content"])
        for response in responses
    ]


def run_scenario(run_backend: str, run_expected_backend: str, log_prefix: str) -> dict[str, Any]:
    port = pick_port()
    base_url = f"http://127.0.0.1:{port}"
    log_file = tempfile.NamedTemporaryFile("w", prefix=log_prefix, suffix=".log", delete=False)
    log_path = log_file.name

    env = os.environ.copy()
    env.update({
        "HIPFIRE_DAEMON_BIN": daemon,
        "HIPFIRE_MODEL": model,
        "HIPFIRE_KV_MODE": "fp32" if dense_fused_mode else "q8",
        "HIPFIRE_QWEN35_STATE_QUANT": "fp32" if dense_fused_mode else "q8",
        "HIPFIRE_NO_PID_FILE": "1",
        "HIPFIRE_SERVER_PREFILL_BATCH": "1",
        "HIPFIRE_SERVER_PREFILL_BATCH_MAX": str(request_count),
        "HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS": "250",
        "HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE": "250",
        "HIPFIRE_MAX_SEQ": str(max_seq),
        "HIPFIRE_DFLASH_DRAFT": "",
        "HIPFIRE_QWEN35_PREFILL_SESSION_BATCH": "serial",
        "HIPFIRE_QWEN35_DECODE_BATCH": run_backend,
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

    try:
        initial = wait_health(base_url, proc, log_path)
        prefill = initial.get("prefill_batch", {})
        if prefill.get("generate_batch_prefill_capability") != "supported":
            raise RuntimeError(f"server prefill capability not supported after warmup: {prefill}; log={log_path}")

        start_barrier = threading.Barrier(request_count + 1)

        def synchronized_chat_request(label: str) -> dict[str, Any]:
            start_barrier.wait(timeout=10.0)
            return chat_request(base_url, label)

        with concurrent.futures.ThreadPoolExecutor(max_workers=request_count) as pool:
            futures = [
                pool.submit(synchronized_chat_request, f"request-{idx}")
                for idx in range(request_count)
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
            "decode_last_chunk_count": decode.get("last_chunk_count"),
            "decode_last_chunk_size": decode.get("last_chunk_size"),
            "decode_last_decode_ms": decode.get("last_decode_ms"),
            "decode_last_skipped_reason": decode.get("last_skipped_reason"),
            "decode_compatible_state_kinds": decode.get("compatible_state_kinds"),
            "decode_cached_prefix_tokens": decode.get("cached_prefix_tokens"),
            "decode_fallback_reason": decode.get("fallback_reason"),
            "decode_active_sessions": decode.get("active_sessions"),
            "prefill_selected_batch_size": prefill.get("selected_batch_size"),
            "resident_runtime_sessions": prefill.get("resident_runtime_sessions"),
            "resident_decode_sessions": prefill.get("resident_decode_sessions"),
            "pending_requests": prefill.get("pending_requests"),
        }
        if int(checks["decode_total_batches"] or 0) < 1:
            raise RuntimeError(f"server decode did not record batch telemetry: {checks}; log={log_path}")
        if run_expected_backend == "serial_reference" and int(checks["decode_serial_batches"] or 0) < 1:
            raise RuntimeError(f"server decode did not record serial batch telemetry: {checks}; log={log_path}")
        if checks["decode_selected_batch_size"] != request_count:
            raise RuntimeError(f"server decode did not select a {request_count}-request batch: {checks}; log={log_path}")
        if checks["decode_last_backend"] != run_expected_backend:
            raise RuntimeError(f"unexpected decode backend: {checks}; log={log_path}")
        if run_expected_backend in {"fused_dense_layer_chunked", "fused_grouped_moe_layer_chunked"}:
            chunk_count = int(checks["decode_last_chunk_count"] or 0)
            chunk_size = int(checks["decode_last_chunk_size"] or 0)
            selected_size = int(checks["decode_selected_batch_size"] or 0)
            if chunk_count < 1 or chunk_size < 1:
                raise RuntimeError(f"fused decode did not record chunk telemetry: {checks}; log={log_path}")
            if requested_decode_chunk_size > 0:
                expected_chunk_size = min(requested_decode_chunk_size, selected_size)
                if run_expected_backend == "fused_dense_layer_chunked" and not native_multirow_enabled:
                    expected_chunk_size = 1
                expected_chunk_count = ceil(selected_size / expected_chunk_size)
                if chunk_size != expected_chunk_size or chunk_count != expected_chunk_count:
                    raise RuntimeError(
                        "fused decode chunk telemetry did not match HIPFIRE_QWEN35_DECODE_BATCH_MAX: "
                        f"expected_size={expected_chunk_size} expected_count={expected_chunk_count} "
                        f"checks={checks}; log={log_path}"
                    )
        if float(checks["decode_last_decode_ms"] or 0) <= 0:
            raise RuntimeError(f"server decode did not record positive decode latency: {checks}; log={log_path}")
        if checks["decode_compatible_state_kinds"] != ["attention_kv", "deltanet_recurrent"]:
            raise RuntimeError(f"server decode did not expose compatible state kinds: {checks}; log={log_path}")
        if checks["decode_cached_prefix_tokens"] is None:
            raise RuntimeError(f"server decode did not expose cached prefix token metadata: {checks}; log={log_path}")
        if int(checks["decode_cached_prefix_tokens"] or 0) <= 0:
            raise RuntimeError(f"server decode did not preserve cached prefix token metadata: {checks}; log={log_path}")
        if not isinstance(checks["decode_fallback_reason"], str) or not checks["decode_fallback_reason"]:
            raise RuntimeError(f"server decode did not expose fallback reason metadata: {checks}; log={log_path}")
        if checks["prefill_selected_batch_size"] != request_count:
            raise RuntimeError(f"server prefill did not coalesce setup requests: {checks}; log={log_path}")
        if int(checks["decode_active_sessions"] or 0) != 0:
            raise RuntimeError(f"server decode left active pending sessions: {checks}; log={log_path}")
        if int(checks["pending_requests"] or 0) != 0:
            raise RuntimeError(f"server prefill left pending requests behind: {checks}; log={log_path}")
        if int(checks["resident_runtime_sessions"] or 0) != 0:
            raise RuntimeError(f"server prefill left resident runtime sessions behind: {checks}; log={log_path}")
        if int(checks["resident_decode_sessions"] or 0) != 0:
            raise RuntimeError(f"server decode left resident decode sessions behind: {checks}; log={log_path}")

        return {
            "responses": responses,
            "contents": response_contents(responses),
            "checks": checks,
            "log_path": log_path,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()


parity_enabled = os.environ.get("HIPFIRE_DECODE_BATCH_PARITY", "").lower() in {"1", "true", "yes"}


def run_parity_pair(batch_size: int, chunk_size: int | None = None) -> dict[str, Any]:
    global request_count, requested_decode_chunk_size

    old_request_count = request_count
    old_requested_chunk_size = requested_decode_chunk_size
    old_chunk_env = os.environ.get("HIPFIRE_QWEN35_DECODE_BATCH_MAX")
    request_count = batch_size
    if chunk_size is None:
        requested_decode_chunk_size = old_requested_chunk_size
        if old_chunk_env is None:
            os.environ.pop("HIPFIRE_QWEN35_DECODE_BATCH_MAX", None)
        else:
            os.environ["HIPFIRE_QWEN35_DECODE_BATCH_MAX"] = old_chunk_env
    else:
        requested_decode_chunk_size = chunk_size
        os.environ["HIPFIRE_QWEN35_DECODE_BATCH_MAX"] = str(chunk_size)
    try:
        serial = run_scenario("serial", "serial_reference", f"hipfire-server-decode-batch-parity-b{batch_size}-serial-")
        fused = run_scenario(decode_backend, expected_decode_backend, f"hipfire-server-decode-batch-parity-b{batch_size}-fused-")
        if serial["contents"] != fused["contents"]:
            raise RuntimeError(
                "serial/fused decode response parity mismatch: "
                f"batch_size={batch_size} chunk_size={chunk_size} "
                f"serial={serial['contents']} fused={fused['contents']} "
                f"serial_log={serial['log_path']} fused_log={fused['log_path']}"
            )
        return {
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "serial_log": serial["log_path"],
            "fused_log": fused["log_path"],
            "log_path": fused["log_path"],
            "contents": fused["contents"],
            "serial_checks": serial["checks"],
            "checks": fused["checks"],
            "responses": fused["responses"],
        }
    finally:
        request_count = old_request_count
        requested_decode_chunk_size = old_requested_chunk_size
        if old_chunk_env is None:
            os.environ.pop("HIPFIRE_QWEN35_DECODE_BATCH_MAX", None)
        else:
            os.environ["HIPFIRE_QWEN35_DECODE_BATCH_MAX"] = old_chunk_env


matrix_enabled = os.environ.get("HIPFIRE_DECODE_BATCH_GROUPED_PARITY_MATRIX", "").lower() in {"1", "true", "yes"}
internal_parity_enabled = os.environ.get("HIPFIRE_QWEN35_DECODE_INTERNAL_PARITY", "").lower() in {"1", "true", "yes", "on"}
if matrix_enabled:
    if expected_decode_backend != "fused_grouped_moe_layer_chunked":
        raise RuntimeError(
            "HIPFIRE_DECODE_BATCH_GROUPED_PARITY_MATRIX requires "
            "HIPFIRE_QWEN35_DECODE_BATCH=fused_grouped_moe"
        )
    matrix_chunk_size = int(os.environ.get("HIPFIRE_DECODE_BATCH_GROUPED_PARITY_CHUNK_SIZE", "2") or "2")
    if matrix_chunk_size < 1:
        raise RuntimeError("HIPFIRE_DECODE_BATCH_GROUPED_PARITY_CHUNK_SIZE must be >= 1")
    matrix = [
        run_parity_pair(2, matrix_chunk_size),
        run_parity_pair(4, matrix_chunk_size),
        run_parity_pair(8, matrix_chunk_size),
    ]
    result = matrix[-1]
    print(
        "server grouped-MoE decode parity matrix passed: "
        f"internal_parity={internal_parity_enabled} "
        + " ".join(
            f"B={entry['batch_size']} chunks={entry['checks'].get('decode_last_chunk_count')}/{entry['checks'].get('decode_last_chunk_size')} "
            f"serial_ms={float(entry['serial_checks'].get('decode_last_decode_ms') or 0):.3f} "
            f"native_ms={float(entry['checks'].get('decode_last_decode_ms') or 0):.3f}"
            for entry in matrix
        )
    )
elif parity_enabled and expected_decode_backend in {"fused_dense_layer_chunked", "fused_grouped_moe_layer_chunked"}:
    result = run_parity_pair(request_count)
    print(
        "server decode batching parity passed: "
        f"contents={result['contents']} serial_log={result['serial_log']} fused_log={result['fused_log']}"
    )
else:
    result = run_scenario(decode_backend, expected_decode_backend, "hipfire-server-decode-batch-")

checks = result["checks"]
print(
    "server decode batching smoke passed: "
    f"responses={len(result['responses'])} selected_batch_size={checks['decode_selected_batch_size']} "
    f"backend={checks['decode_last_backend']} chunks={checks.get('decode_last_chunk_count')}/{checks.get('decode_last_chunk_size')} "
    f"log={result['log_path']}"
)
PY
