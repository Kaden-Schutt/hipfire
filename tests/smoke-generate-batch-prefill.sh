#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DAEMON="${DAEMON:-$ROOT/target/release/hipfire-daemon}"
MODEL="${MODEL:-$HOME/.hipfire/models/qwen3.5-0.8b-mq4.hfq}"
MOE_MODEL="${MOE_MODEL:-$HOME/.hipfire/models/qwen3.6-35b-a3b-mq4.hfq}"
UNSUPPORTED_MODEL="${UNSUPPORTED_MODEL:-$HOME/.hipfire/models/llama-3.2-1b-instruct-mq4.hfq}"
MAX_SEQ="${MAX_SEQ:-512}"

if [[ ! -x "$DAEMON" ]]; then
  echo "missing daemon binary: $DAEMON" >&2
  echo "build it with: cargo build --release -p hipfire-daemon --bin hipfire-daemon" >&2
  exit 2
fi

if [[ ! -f "$MODEL" ]]; then
  echo "missing model: $MODEL" >&2
  exit 2
fi

python3 - "$DAEMON" "$MODEL" "$MOE_MODEL" "$UNSUPPORTED_MODEL" "$MAX_SEQ" <<'PY'
import json
import os
import subprocess
import sys
from typing import Any

daemon, model, moe_model, unsupported_model, max_seq_s = sys.argv[1:]
max_seq = int(max_seq_s)
worker_id = "qwen35-smoke-worker"


def start_daemon(env: dict[str, str] | None = None) -> subprocess.Popen[str]:
    proc_env = os.environ.copy()
    proc_env["HIPFIRE_EMIT_TOKEN_IDS"] = "1"
    proc_env["HIPFIRE_GENERATE_BATCH_PREFILL_DEBUG_SAMPLE"] = "1"
    if env:
        proc_env.update(env)
    return subprocess.Popen(
        [daemon],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        encoding="utf-8",
        env=proc_env,
        bufsize=1,
    )


def send(proc: subprocess.Popen[str], msg: dict[str, Any]) -> None:
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
    proc.stdin.flush()


def recv_json(proc: subprocess.Popen[str]) -> dict[str, Any]:
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if line == "":
            raise RuntimeError("daemon exited before expected event")
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue


def load(proc: subprocess.Popen[str], model_path: str = model) -> None:
    send(proc, {
        "type": "load",
        "model": model_path,
        "worker_key_id": worker_id,
        "params": {
            "max_seq": max_seq,
            "kv_mode": "q8",
        },
    })
    while True:
        event = recv_json(proc)
        if event.get("type") == "loaded":
            return
        if event.get("type") == "error":
            raise RuntimeError(f"load failed: {event.get('message')}")


def probe_generate_batch_prefill(proc: subprocess.Popen[str]) -> dict[str, Any]:
    send(proc, {
        "type": "generate_batch_prefill",
        "id": "prefill-batch-probe",
        "batch_id": "prefill-batch-probe",
        "worker_key_id": worker_id,
        "sessions": [{
            "id": "probe-session",
            "suffix_tokens": [1],
            "state_handle": {
                "state_kinds": ["attention_kv"],
                "logical_position": 0,
                "cached_prefix_tokens": 0,
            },
            "params": {
                "max_tokens": 1,
                "temperature": 0,
            },
        }],
    })
    while True:
        event = recv_json(proc)
        typ = event.get("type")
        if typ in ("generate_batch_prefill_ready", "generate_batch_prefill_unsupported", "error"):
            return event


def assert_prefill_probe(proc: subprocess.Popen[str], model_path: str, supported: bool, label: str) -> None:
    load(proc, model_path)
    event = probe_generate_batch_prefill(proc)
    if supported:
        if event.get("type") != "generate_batch_prefill_ready" or event.get("supported") is not True:
            raise RuntimeError(f"{label}: expected supported prefill probe, got {event}")
        print(f"{label}: generate_batch_prefill probe supported")
    else:
        if event.get("type") != "generate_batch_prefill_unsupported" or event.get("supported") is not False:
            raise RuntimeError(f"{label}: expected unsupported prefill probe, got {event}")
        print(f"{label}: generate_batch_prefill probe unsupported ({event.get('reason')})")


def reset(proc: subprocess.Popen[str]) -> None:
    send(proc, {"type": "reset", "worker_key_id": worker_id})
    while True:
        event = recv_json(proc)
        if event.get("type") == "reset":
            return
        if event.get("type") == "error":
            raise RuntimeError(f"reset failed: {event.get('message')}")


def prefill_batch(
    proc: subprocess.Popen[str],
    batch_id: str,
    sessions: list[dict[str, Any]],
    expect_backend: str,
    expect_plan: str = "fused_dense_qwen35_candidate",
) -> tuple[dict[str, int], dict[str, int], dict[str, Any]]:
    send(proc, {
        "type": "generate_batch_prefill",
        "id": batch_id,
        "batch_id": batch_id,
        "worker_key_id": worker_id,
        "sessions": sessions,
    })
    logical_positions: dict[str, int] = {}
    debug_samples: dict[str, int] = {}
    started = False
    while True:
        event = recv_json(proc)
        typ = event.get("type")
        if typ == "generate_batch_prefill_started":
            started = True
            if event.get("plan") != expect_plan:
                raise RuntimeError(f"{batch_id}: unexpected plan in started event: {event}")
            if event.get("backend") != expect_backend:
                raise RuntimeError(f"{batch_id}: unexpected backend in started event: {event}")
        elif typ == "generate_batch_prefill_session_done":
            sid = event.get("session_id")
            pos = event.get("logical_position")
            if not isinstance(sid, str) or not isinstance(pos, int):
                raise RuntimeError(f"{batch_id}: malformed session_done event: {event}")
            logical_positions[sid] = pos
            debug_sample = event.get("debug_sample_token")
            if isinstance(debug_sample, int):
                debug_samples[sid] = debug_sample
            if int(event.get("prefill_tokens", 0)) <= 0:
                raise RuntimeError(f"{batch_id}: no prefill tokens for session {sid}")
        elif typ == "generate_batch_prefill_done":
            if not started:
                raise RuntimeError(f"{batch_id}: done before started")
            if event.get("sessions") != len(sessions):
                raise RuntimeError(f"{batch_id}: wrong session count in done event: {event}")
            if event.get("backend") != expect_backend:
                raise RuntimeError(f"{batch_id}: unexpected backend in done event: {event}")
            if int(event.get("prefill_tokens", 0)) <= 0:
                raise RuntimeError(f"{batch_id}: no total prefill tokens")
            missing = {s["id"] for s in sessions} - set(logical_positions)
            if missing:
                raise RuntimeError(f"{batch_id}: missing session_done for {sorted(missing)}")
            return logical_positions, debug_samples, event
        elif typ == "error":
            raise RuntimeError(f"{batch_id}: daemon error: {event.get('message')}")


def prefill_boundary_batch(
    proc: subprocess.Popen[str],
    batch_id: str,
    sessions: list[dict[str, Any]],
    expect_backend: str,
    expect_plan: str = "fused_dense_qwen35_candidate",
) -> tuple[dict[str, int], str]:
    send(proc, {
        "type": "generate_batch_prefill",
        "id": batch_id,
        "batch_id": batch_id,
        "worker_key_id": worker_id,
        "sessions": sessions,
    })
    prefix_checkpoint_counts: dict[str, int] = {}
    first_checkpoint_handle: str | None = None
    started = False
    while True:
        event = recv_json(proc)
        typ = event.get("type")
        if typ == "generate_batch_prefill_started":
            started = True
            if event.get("plan") != expect_plan:
                raise RuntimeError(f"{batch_id}: unexpected plan in started event: {event}")
            if event.get("backend") != expect_backend:
                raise RuntimeError(f"{batch_id}: unexpected backend in started event: {event}")
        elif typ == "generate_batch_prefill_session_done":
            sid = event.get("session_id")
            state_handle = event.get("state_handle")
            if not isinstance(sid, str) or not isinstance(state_handle, dict):
                raise RuntimeError(f"{batch_id}: malformed session_done event: {event}")
            prefix_checkpoints = state_handle.get("prefix_checkpoints")
            if not isinstance(prefix_checkpoints, list) or not prefix_checkpoints:
                raise RuntimeError(f"{batch_id}: missing prefix checkpoints for {sid}: {event}")
            for checkpoint in prefix_checkpoints:
                if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_runtime_state") != "attachable":
                    raise RuntimeError(f"{batch_id}: malformed prefix checkpoint for {sid}: {checkpoint}")
                handle = checkpoint.get("checkpoint_id")
                if not isinstance(handle, str) or not handle:
                    raise RuntimeError(f"{batch_id}: malformed prefix checkpoint handle for {sid}: {checkpoint}")
                if first_checkpoint_handle is None:
                    first_checkpoint_handle = handle
            prefix_checkpoint_counts[sid] = len(prefix_checkpoints)
        elif typ == "generate_batch_prefill_done":
            if not started:
                raise RuntimeError(f"{batch_id}: done before started")
            if event.get("sessions") != len(sessions):
                raise RuntimeError(f"{batch_id}: wrong session count in done event: {event}")
            if event.get("backend") != expect_backend:
                raise RuntimeError(f"{batch_id}: unexpected backend in done event: {event}")
            missing = {s["id"] for s in sessions} - set(prefix_checkpoint_counts)
            if missing:
                raise RuntimeError(f"{batch_id}: missing boundary checkpoint event for {sorted(missing)}")
            if first_checkpoint_handle is None:
                raise RuntimeError(f"{batch_id}: no attachable checkpoint handle captured")
            return prefix_checkpoint_counts, first_checkpoint_handle
        elif typ == "error":
            raise RuntimeError(f"{batch_id}: daemon error: {event.get('message')}")


def release_sessions(proc: subprocess.Popen[str], handles: list[str], expect_released: int, label: str) -> None:
    send(proc, {
        "type": "release_sessions",
        "id": f"{label}-release",
        "worker_key_id": worker_id,
        "sessions": handles,
    })
    while True:
        event = recv_json(proc)
        typ = event.get("type")
        if typ == "release_sessions_done":
            released = event.get("released")
            if released != expect_released:
                raise RuntimeError(
                    f"{label}: expected release count {expect_released}, got {released}: {event}"
                )
            return
        if typ == "error":
            raise RuntimeError(f"{label}: release_sessions error: {event.get('message')}")


def decode_one_token_after_prefill(
    proc: subprocess.Popen[str],
    session: dict[str, Any],
    prompt_override: str | None = None,
) -> dict[str, Any]:
    sid = str(session["id"])
    send(proc, {
        "type": "generate",
        "id": f"{sid}-decode",
        "worker_key_id": worker_id,
        "prompt": prompt_override if prompt_override is not None else session["prompt"],
        "session_id": sid,
        "prefill_already_done": True,
        "temperature": 0,
        "top_p": 1,
        "repeat_penalty": 1,
        "repeat_window": 0,
        "max_tokens": 1,
        "assistant_prefix": "plain",
    })
    token_text = ""
    committed_ids: list[int] = []
    saw_done = False
    while True:
        event = recv_json(proc)
        typ = event.get("type")
        if typ == "committed":
            tok_id = event.get("tok_id")
            if not isinstance(tok_id, int):
                raise RuntimeError(f"{sid}: malformed committed event: {event}")
            committed_ids.append(tok_id)
        elif typ == "token":
            token_text += str(event.get("text", ""))
        elif typ == "done":
            saw_done = True
            if int(event.get("tokens", 0)) <= 0:
                raise RuntimeError(f"{sid}: decode emitted no tokens")
            break
        elif typ == "error":
            raise RuntimeError(f"{sid}: decode error: {event.get('message')}")
    if not saw_done:
        raise RuntimeError(f"{sid}: decode did not complete")
    if len(committed_ids) != 1:
        raise RuntimeError(f"{sid}: decode expected one committed token, got {committed_ids!r}")
    return {"text": token_text, "tok_id": committed_ids[0]}


def generate_serial_tokens(
    proc: subprocess.Popen[str],
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    reset(proc)
    send(proc, {
        "type": "generate",
        "id": "serial-reference",
        "worker_key_id": worker_id,
        "prompt": prompt,
        "temperature": 0,
        "top_p": 1,
        "repeat_penalty": 1,
        "repeat_window": 0,
        "max_tokens": max_tokens,
        "assistant_prefix": "plain",
    })
    token_text = ""
    committed_ids: list[int] = []
    while True:
        event = recv_json(proc)
        typ = event.get("type")
        if typ == "committed":
            tok_id = event.get("tok_id")
            if not isinstance(tok_id, int):
                raise RuntimeError(f"serial generate emitted malformed committed event: {event}")
            committed_ids.append(tok_id)
            continue
        if typ == "token":
            token_text += str(event.get("text", ""))
        elif typ == "done":
            if int(event.get("tokens", 0)) <= 0:
                raise RuntimeError(f"serial generate emitted no tokens for prompt {prompt!r}")
            if len(committed_ids) != max_tokens:
                raise RuntimeError(
                    f"serial generate expected {max_tokens} committed tokens for prompt {prompt!r}, "
                    f"got {committed_ids!r}"
                )
            return {"text": token_text, "tok_ids": committed_ids}
        elif typ == "error":
            raise RuntimeError(f"serial generate error for prompt {prompt!r}: {event.get('message')}")


def generate_one_token_serial_with_id(proc: subprocess.Popen[str], prompt: str) -> dict[str, Any]:
    generated = generate_serial_tokens(proc, prompt, 1)
    return {"text": generated["text"], "tok_id": generated["tok_ids"][0]}


def generate_one_token_serial(proc: subprocess.Popen[str], prompt: str) -> str:
    return str(generate_one_token_serial_with_id(proc, prompt)["text"])


def prompt_sessions(batch_size: int, label: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{label}-session-{i}",
            "prompt": f"hello {label} {i}",
            "state_handle": {
                "state_kinds": ["attention_kv", "deltanet_recurrent"],
                "logical_position": 0,
                "cached_prefix_tokens": 0,
            },
            "params": {
                "max_tokens": 1,
                "temperature": 0,
            },
        }
        for i in range(batch_size)
    ]


def boundary_prompt_sessions(batch_size: int, label: str) -> list[dict[str, Any]]:
    sessions = prompt_sessions(batch_size, label)
    for session in sessions:
        params = dict(session["params"])
        params["semantic_boundary_checkpoints"] = True
        session["params"] = params
    return sessions


def suffix_sessions(
    logical_positions: dict[str, int],
    suffix_token_ids: dict[str, int],
) -> list[dict[str, Any]]:
    return [
        {
            "id": sid,
            "suffix_tokens": [suffix_token_ids[sid]],
            "state_handle": {
                "state_kinds": ["attention_kv", "deltanet_recurrent"],
                "logical_position": pos,
                "cached_prefix_tokens": pos,
            },
            "params": {
                "max_tokens": 1,
                "temperature": 0,
            },
        }
        for sid, pos in sorted(logical_positions.items())
    ]


proc = start_daemon()
try:
    load(proc)
    dense_probe = probe_generate_batch_prefill(proc)
    if dense_probe.get("type") != "generate_batch_prefill_ready" or dense_probe.get("supported") is not True:
        raise RuntimeError(f"dense qwen35 probe did not report supported: {dense_probe}")
    print("dense qwen35 probe: supported")
    for batch_size in (2, 4, 8):
        label = f"b{batch_size}"
        prompt_batch = prompt_sessions(batch_size, label)
        first_tokens = {
            str(session["id"]): generate_one_token_serial_with_id(proc, str(session["prompt"]))
            for session in prompt_batch
        }
        expected = {sid: str(token["text"]) for sid, token in first_tokens.items()}
        reset(proc)
        positions, _, prompt_done = prefill_batch(
            proc,
            f"{label}-prompt",
            prompt_batch,
            "fused_dense",
        )
        decoded = {
            str(session["id"]): str(decode_one_token_after_prefill(proc, session)["text"])
            for session in prompt_batch
        }
        if decoded != expected:
            raise RuntimeError(
                f"{label}: first-token mismatch after batch-prefill: "
                f"decoded={decoded!r} expected={expected!r}"
            )
        suffix_label = f"{label}-suffix-base"
        suffix_prompt_batch = prompt_sessions(batch_size, suffix_label)
        suffix_first_tokens = {
            str(session["id"]): generate_one_token_serial_with_id(proc, str(session["prompt"]))
            for session in suffix_prompt_batch
        }
        suffix_positions, _, _ = prefill_batch(
            proc,
            f"{suffix_label}-prompt",
            suffix_prompt_batch,
            "fused_dense",
        )
        _, suffix_debug_samples, _ = prefill_batch(
            proc,
            f"{label}-suffix",
            suffix_sessions(
                suffix_positions,
                {sid: int(token["tok_id"]) for sid, token in suffix_first_tokens.items()},
            ),
            "fused_dense",
        )
        if set(suffix_debug_samples) != {str(session["id"]) for session in suffix_prompt_batch}:
            raise RuntimeError(
                f"{label}: missing suffix debug samples: {suffix_debug_samples!r}"
            )
        suffix_expected = suffix_debug_samples
        suffix_decoded = {
            str(session["id"]): int(decode_one_token_after_prefill(
                proc,
                session,
                str(session["prompt"]) + str(suffix_first_tokens[str(session["id"])]["text"]),
            )["tok_id"])
            for session in suffix_prompt_batch
        }
        if suffix_decoded != suffix_expected:
            suffix_replayed = {
                sid: int(token["tok_id"])
                for sid, token in suffix_first_tokens.items()
            }
            raise RuntimeError(
                f"{label}: suffix-token mismatch after batch-prefill: "
                f"decoded={suffix_decoded!r} expected={suffix_expected!r} "
                f"replayed_suffix={suffix_replayed!r}"
            )
        print(
            f"default dense batch-prefill size={batch_size}: "
            f"plan={prompt_done['plan']} backend={prompt_done['backend']} "
            f"decoded={len(decoded)} suffix_sessions={len(positions)} ok"
        )
finally:
    if proc.stdin is not None:
        proc.stdin.close()
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


fused_proc = start_daemon({
    "HIPFIRE_QWEN35_PREFILL_SESSION_BATCH": "fused",
    "HIPFIRE_JINJA_CHAT": "1",
})
try:
    load(fused_proc)
    reset(fused_proc)
    boundary_counts, boundary_checkpoint_handle = prefill_boundary_batch(
        fused_proc,
        "fused-boundary-b2",
        boundary_prompt_sessions(2, "fused-boundary"),
        "fused_dense",
    )
    release_sessions(fused_proc, [boundary_checkpoint_handle], 1, "fused-boundary-checkpoint")
    release_sessions(fused_proc, [boundary_checkpoint_handle], 0, "fused-boundary-checkpoint-stale")
    print(
        "explicit fused boundary checkpoints: "
        f"sessions={len(boundary_counts)} "
        f"checkpoints={sum(boundary_counts.values())} "
        f"released=1 stale_release=0 ok"
    )
    for batch_size in (2, 4, 8):
        label = f"fused-b{batch_size}"
        fused_batch = prompt_sessions(batch_size, label)
        fused_first_tokens = {
            str(session["id"]): generate_one_token_serial_with_id(fused_proc, str(session["prompt"]))
            for session in fused_batch
        }
        fused_expected = {sid: str(token["text"]) for sid, token in fused_first_tokens.items()}
        reset(fused_proc)
        fused_positions, _, fused_done = prefill_batch(
            fused_proc,
            f"{label}-prompt",
            fused_batch,
            "fused_dense",
        )
        fused_decoded = {
            str(session["id"]): str(decode_one_token_after_prefill(fused_proc, session)["text"])
            for session in fused_batch
        }
        if fused_decoded != fused_expected:
            raise RuntimeError(
                f"{label}: explicit fused first-token mismatch: decoded={fused_decoded!r} "
                f"expected={fused_expected!r}"
            )

        suffix_label = f"{label}-suffix-base"
        suffix_prompt_batch = prompt_sessions(batch_size, suffix_label)
        suffix_first_tokens = {
            str(session["id"]): generate_one_token_serial_with_id(
                fused_proc,
                str(session["prompt"]),
            )
            for session in suffix_prompt_batch
        }
        suffix_positions, _, _ = prefill_batch(
            fused_proc,
            f"{suffix_label}-prompt",
            suffix_prompt_batch,
            "fused_dense",
        )
        _, fused_suffix_debug_samples, _ = prefill_batch(
            fused_proc,
            f"{label}-suffix",
            suffix_sessions(
                suffix_positions,
                {sid: int(token["tok_id"]) for sid, token in suffix_first_tokens.items()},
            ),
            "fused_dense",
        )
        if set(fused_suffix_debug_samples) != {str(session["id"]) for session in suffix_prompt_batch}:
            raise RuntimeError(
                f"{label}: missing fused suffix debug samples: {fused_suffix_debug_samples!r}"
            )
        fused_suffix_decoded = {
            str(session["id"]): int(decode_one_token_after_prefill(
                fused_proc,
                session,
                str(session["prompt"]) + str(suffix_first_tokens[str(session["id"])]["text"]),
            )["tok_id"])
            for session in suffix_prompt_batch
        }
        if fused_suffix_decoded != fused_suffix_debug_samples:
            suffix_replayed = {
                sid: int(token["tok_id"])
                for sid, token in suffix_first_tokens.items()
            }
            raise RuntimeError(
                f"{label}: fused suffix-token mismatch after batch-prefill: "
                f"decoded={fused_suffix_decoded!r} expected={fused_suffix_debug_samples!r} "
                f"replayed_suffix={suffix_replayed!r}"
            )
        print(
            "explicit fused backend: "
            f"size={batch_size} plan={fused_done['plan']} backend={fused_done['backend']} "
            f"decoded={len(fused_decoded)} suffix_sessions={len(fused_positions)} ok"
        )
finally:
    if fused_proc.stdin is not None:
        fused_proc.stdin.close()
    fused_proc.terminate()
    try:
        fused_proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        fused_proc.kill()
        fused_proc.wait()


def run_loaded_probe(model_path: str, supported: bool, label: str) -> None:
    if not model_path or not os.path.isfile(model_path):
        print(f"{label}: skipped missing model {model_path!r}")
        return
    proc = start_daemon({"HIPFIRE_QWEN35_PREFILL_SESSION_BATCH": "serial"})
    try:
        assert_prefill_probe(proc, model_path, supported, label)
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_moe_grouped_plan_smoke(model_path: str) -> None:
    if not model_path or not os.path.isfile(model_path):
        print(f"qwen35-moe/a3b grouped plan smoke: skipped missing model {model_path!r}")
        return
    proc = start_daemon({"HIPFIRE_QWEN35_PREFILL_SESSION_BATCH": "serial"})
    try:
        load(proc, model_path)
        for batch_size in (2, 4, 8):
            label = f"moe-grouped-b{batch_size}"
            prompt_batch = prompt_sessions(batch_size, label)
            first_tokens = {
                str(session["id"]): generate_one_token_serial_with_id(proc, str(session["prompt"]))
                for session in prompt_batch
            }
            expected = {sid: str(token["text"]) for sid, token in first_tokens.items()}
            reset(proc)
            _, _, done = prefill_batch(
                proc,
                f"{label}-candidate",
                prompt_batch,
                "serial_reference",
                "grouped_moe_qwen35_candidate",
            )
            decoded = {
                str(session["id"]): str(decode_one_token_after_prefill(proc, session)["text"])
                for session in prompt_batch
            }
            if decoded != expected:
                raise RuntimeError(
                    "qwen35-moe/a3b serial grouped first-token mismatch: "
                    f"decoded={decoded!r} expected={expected!r}"
                )

            suffix_label = f"{label}-suffix-base"
            suffix_prompt_batch = prompt_sessions(batch_size, suffix_label)
            suffix_first_tokens = {
                str(session["id"]): generate_one_token_serial_with_id(proc, str(session["prompt"]))
                for session in suffix_prompt_batch
            }
            suffix_positions, _, _ = prefill_batch(
                proc,
                f"{suffix_label}-prompt",
                suffix_prompt_batch,
                "serial_reference",
                "grouped_moe_qwen35_candidate",
            )
            _, suffix_debug_samples, _ = prefill_batch(
                proc,
                f"{label}-suffix",
                suffix_sessions(
                    suffix_positions,
                    {sid: int(token["tok_id"]) for sid, token in suffix_first_tokens.items()},
                ),
                "serial_reference",
                "grouped_moe_qwen35_candidate",
            )
            if set(suffix_debug_samples) != {str(session["id"]) for session in suffix_prompt_batch}:
                raise RuntimeError(
                    f"{label}: missing grouped MoE suffix debug samples: {suffix_debug_samples!r}"
                )
            suffix_decoded = {
                str(session["id"]): int(decode_one_token_after_prefill(
                    proc,
                    session,
                    str(session["prompt"]) + str(suffix_first_tokens[str(session["id"])]["text"]),
                )["tok_id"])
                for session in suffix_prompt_batch
            }
            if suffix_decoded != suffix_debug_samples:
                raise RuntimeError(
                    f"{label}: grouped MoE suffix-token mismatch after batch-prefill: "
                    f"decoded={suffix_decoded!r} expected={suffix_debug_samples!r}"
                )
            print(
                "qwen35-moe/a3b grouped candidate: "
                f"size={batch_size} plan={done['plan']} backend={done['backend']} "
                f"decoded={len(decoded)} suffix_sessions={len(suffix_positions)} ok"
            )
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_moe_fused_grouped_smoke(model_path: str) -> None:
    if not model_path or not os.path.isfile(model_path):
        print(f"qwen35-moe/a3b fused grouped smoke: skipped missing model {model_path!r}")
        return
    proc = start_daemon({"HIPFIRE_QWEN35_PREFILL_SESSION_BATCH": "fused_moe"})
    try:
        load(proc, model_path)
        for batch_size in (2, 4, 8):
            label = f"moe-fused-grouped-b{batch_size}"
            prompt_batch = prompt_sessions(batch_size, label)
            first_tokens = {
                str(session["id"]): generate_one_token_serial_with_id(proc, str(session["prompt"]))
                for session in prompt_batch
            }
            expected = {sid: str(token["text"]) for sid, token in first_tokens.items()}
            reset(proc)
            _, _, done = prefill_batch(
                proc,
                f"{label}-prompt",
                prompt_batch,
                "fused_grouped_moe",
                "grouped_moe_qwen35_candidate",
            )
            decoded = {
                str(session["id"]): str(decode_one_token_after_prefill(proc, session)["text"])
                for session in prompt_batch
            }
            if decoded != expected:
                raise RuntimeError(
                    "qwen35-moe/a3b fused grouped first-token mismatch: "
                    f"decoded={decoded!r} expected={expected!r}"
                )

            suffix_label = f"{label}-suffix-base"
            suffix_prompt_batch = prompt_sessions(batch_size, suffix_label)
            suffix_first_tokens = {
                str(session["id"]): generate_one_token_serial_with_id(proc, str(session["prompt"]))
                for session in suffix_prompt_batch
            }
            suffix_positions, _, _ = prefill_batch(
                proc,
                f"{suffix_label}-prompt",
                suffix_prompt_batch,
                "fused_grouped_moe",
                "grouped_moe_qwen35_candidate",
            )
            _, fused_suffix_debug_samples, _ = prefill_batch(
                proc,
                f"{label}-suffix",
                suffix_sessions(
                    suffix_positions,
                    {sid: int(token["tok_id"]) for sid, token in suffix_first_tokens.items()},
                ),
                "fused_grouped_moe",
                "grouped_moe_qwen35_candidate",
            )
            if set(fused_suffix_debug_samples) != {str(session["id"]) for session in suffix_prompt_batch}:
                raise RuntimeError(
                    f"{label}: missing fused grouped MoE suffix debug samples: {fused_suffix_debug_samples!r}"
                )
            fused_suffix_decoded = {
                str(session["id"]): int(decode_one_token_after_prefill(
                    proc,
                    session,
                    str(session["prompt"]) + str(suffix_first_tokens[str(session["id"])]["text"]),
                )["tok_id"])
                for session in suffix_prompt_batch
            }
            if fused_suffix_decoded != fused_suffix_debug_samples:
                raise RuntimeError(
                    f"{label}: fused grouped MoE suffix-token mismatch after batch-prefill: "
                    f"decoded={fused_suffix_decoded!r} expected={fused_suffix_debug_samples!r}"
                )
            print(
                "qwen35-moe/a3b fused grouped: "
                f"size={batch_size} plan={done['plan']} backend={done['backend']} "
                f"decoded={len(decoded)} suffix_sessions={len(suffix_positions)} ok"
            )
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


run_loaded_probe(moe_model, True, "qwen35-moe/a3b")
run_moe_grouped_plan_smoke(moe_model)
run_moe_fused_grouped_smoke(moe_model)
run_loaded_probe(unsupported_model, False, "unsupported-arch")

print("generate_batch_prefill smoke passed")
PY
