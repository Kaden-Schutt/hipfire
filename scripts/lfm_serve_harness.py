#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Single-turn LFM serve smoke with explicit thinking and combined-output checks."""

import argparse
import atexit
import json
import os
import re
import signal
import secrets
import shutil
import socket
import subprocess
import sys
import time
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = "What is the capital of France? Reply in one short sentence."
SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repeat_penalty",
    "system_prompt",
)
EFFORTS = ("minimal", "low", "medium", "high", "xhigh")
FATAL_WARMUP = re.compile(
    r"out of memory|error loading|panic|model not found|pre-warm(?: load)? failed",
    re.IGNORECASE,
)
_proc = None
_daemon_link = None


def build_sampling(registry_path, tag, effort):
    if not tag.startswith("lfm2.5:"):
        raise ValueError(f"not an LFM2.5 registry tag: {tag}")
    models = json.loads(Path(registry_path).read_text())["models"]
    entry = models.get(tag)
    if not entry:
        raise ValueError(f"registry tag not found: {tag}")
    recommended = entry.get("recommended_settings") or {}
    sampling = {key: recommended[key] for key in SAMPLING_KEYS if key in recommended}
    if not sampling:
        raise ValueError(f"registry tag has no recommended_settings: {tag}")
    sampling["reasoning_effort"] = effort
    return sampling


def _max_frequency(tokens):
    if not tokens:
        return 0.0
    return Counter(tokens).most_common(1)[0][1] / len(tokens)


def _unique_ratio(tokens):
    return len(set(tokens)) / len(tokens) if tokens else 1.0


def _repeated_trigram_ratio(tokens):
    if len(tokens) < 6:
        return 0.0
    grams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
    counts = Counter(grams)
    return sum(count for count in counts.values() if count > 1) / len(grams)


def is_attractor(text):
    tokens = re.findall(r"\S+", text)
    if len(tokens) < 6:
        return False
    first = tokens[:128]
    last = tokens[-128:]
    middle = tokens[len(tokens) // 2:]
    return (
        bool(first) and (_unique_ratio(first) < 0.15 or _max_frequency(first) > 0.50)
    ) or (
        bool(last) and (_unique_ratio(last) < 0.30 or _max_frequency(last) > 0.50)
    ) or _repeated_trigram_ratio(middle) > 0.50


def validate_turn(row, expected_text, forbid_reasoning=False):
    errors = []
    if row.get("finish") != "stop":
        errors.append(f"finish_reason={row.get('finish')}")
    content = row.get("content", "")
    reasoning = row.get("reasoning_content", "")
    combined = row.get("combined_content", "")
    if not combined.strip():
        errors.append("empty reasoning_content + content")
    if row.get("attractor"):
        errors.append("token attractor detected")
    # Framing contract (UNCONDITIONAL): the answer must reach message.content,
    # never be stranded in reasoning_content with content left empty (the
    # daemon-authoritative think-framing bug). Checked on the content channel,
    # not combined, so a reasoning-only completion is rejected on every run.
    if expected_text and expected_text.lower() not in content.lower():
        errors.append(f"missing expected text: {expected_text}")
    # --nothink: thinking was disabled at the request, so the reasoning row must
    # be gone. (Thinking-on runs don't force reasoning present — the content
    # guard above already catches the stranding bug.)
    if forbid_reasoning and reasoning.strip():
        errors.append("unexpected reasoning_content (--nothink)")
    return errors


def load_prompt(path, direct_prompt):
    if direct_prompt:
        return direct_prompt
    if not path:
        return DEFAULT_PROMPT
    text = Path(path).read_text()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]["prompt"]
    if isinstance(payload, dict) and isinstance(payload.get("prompt"), str):
        return payload["prompt"]
    raise ValueError(f"prompt file must be text, a prompt object, or a non-empty prompt list: {path}")


def allocate_runtime_defaults(home, serve_log, port):
    root = None
    if home is None or serve_log is None:
        root = Path(tempfile.mkdtemp(prefix="lfm-serve-harness-"))
    resolved_home = home or str(root / "home")
    resolved_log = serve_log or str(root / "serve.log")
    resolved_port = port
    if resolved_port == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            resolved_port = sock.getsockname()[1]
    return resolved_home, resolved_log, resolved_port


def _prepare_daemon(daemon, directory):
    global _daemon_link
    directory.mkdir(parents=True, exist_ok=True)
    name = f"lfm{os.getpid() % 100000:05d}{secrets.token_hex(2)}"
    _daemon_link = directory / name
    try:
        os.link(daemon, _daemon_link)
    except OSError:
        shutil.copy2(daemon, _daemon_link)
    _daemon_link.chmod(0o755)
    return _daemon_link


def _health_ready(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.25) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _kill_server():
    global _proc, _daemon_link
    if _proc is not None:
        try:
            os.killpg(os.getpgid(_proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                _proc.kill()
            except OSError:
                pass
    _proc = None
    if _daemon_link is not None:
        try:
            _daemon_link.unlink()
        except OSError:
            pass
    _daemon_link = None


def _log_tail(path, lines=60):
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "<serve log unavailable>"


def spawn_server(args):
    global _proc
    model = Path(args.model).expanduser().resolve()
    daemon = Path(args.daemon_bin).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(f"model does not exist: {model}")
    if not daemon.is_file():
        raise FileNotFoundError(f"daemon does not exist: {daemon}")

    home = Path(args.home).expanduser().resolve()
    daemon_exec = _prepare_daemon(daemon, home.parent)
    hipfire_home = home / ".hipfire"
    hipfire_home.mkdir(parents=True, exist_ok=True)
    (hipfire_home / "config.json").write_text(json.dumps({
        "max_seq": args.max_seq,
        "dflash_mode": "off",
        "mtp_mode": "off",
        "ngram_mode": "off",
        "max_tokens": max(args.max_tokens, 4096),
        "thinking_budget": args.thinking_budget,
        "kv_mode": "q8",
    }))

    log_path = Path(args.serve_log)
    log_path.write_text("")
    env = dict(
        os.environ,
        HOME=str(home),
        HIP_VISIBLE_DEVICES=str(args.device),
        HIPFIRE_DAEMON_BIN=str(daemon_exec),
        HIPFIRE_DAEMON_NAME=daemon_exec.name,
        HIPFIRE_KV_MODE="q8",
        HIPFIRE_CASK_OFF="1",
        HIPFIRE_MODEL=str(model),
    )
    log_handle = log_path.open("a")
    _proc = subprocess.Popen(
        [args.bun, "cli/index.ts", "serve", "127.0.0.1", str(args.port)],
        cwd=REPO,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    atexit.register(_kill_server)

    deadline = time.monotonic() + args.warm_timeout
    warmed = False
    while time.monotonic() < deadline:
        text = log_path.read_text(errors="replace")
        warmed = warmed or "warm-up complete" in text
        if _proc.poll() is not None:
            raise RuntimeError(f"serve exited with code {_proc.returncode}\n{_log_tail(log_path)}")
        if FATAL_WARMUP.search(text):
            raise RuntimeError(f"serve failed during warm-up\n{_log_tail(log_path)}")
        if warmed and _health_ready(args.port):
            return str(model)
        time.sleep(0.5)
    raise TimeoutError(f"serve did not warm within {args.warm_timeout}s\n{_log_tail(log_path)}")


def request_turn(args, model, prompt, sampling, stream=True):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "stream": stream,
        **sampling,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    if args.seed is not None:
        body["seed"] = args.seed

    request = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    first_token = None
    reasoning = []
    content = []
    tools = []
    usage = {}
    timings = {}
    finish = None
    try:
        response = urllib.request.urlopen(request, timeout=args.request_timeout)
        if stream:
            for raw in response:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if chunk.get("timings"):
                    timings = chunk["timings"]
                choice = (chunk.get("choices") or [{}])[0]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                delta = choice.get("delta") or {}
                for key, sink in (("reasoning_content", reasoning), ("content", content)):
                    value = delta.get(key)
                    if isinstance(value, str):
                        if value and first_token is None:
                            first_token = time.time() - started
                        sink.append(value)
                if delta.get("tool_calls"):
                    tools.append(json.dumps(delta["tool_calls"]))
        else:
            resp_json = json.loads(response.read())
            if resp_json.get("usage"):
                usage = resp_json["usage"]
            if resp_json.get("timings"):
                timings = resp_json["timings"]
            choice = (resp_json.get("choices") or [{}])[0]
            finish = choice.get("finish_reason")
            message = choice.get("message") or {}
            rc = message.get("reasoning_content")
            if isinstance(rc, str):
                reasoning.append(rc)
            cc = message.get("content")
            if isinstance(cc, str):
                content.append(cc)
            if message.get("tool_calls"):
                tools.append(json.dumps(message["tool_calls"]))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error

    wall = time.time() - started
    reasoning_text = "".join(reasoning)
    content_text = "".join(content)
    tool_text = "".join(tools)
    combined = "\n".join(part for part in (reasoning_text, content_text, tool_text) if part).strip()
    completion_tokens = usage.get("completion_tokens", 0)
    decode_tok_s = timings.get("decode_tok_s")
    estimated = False
    if decode_tok_s is None and completion_tokens > 1 and first_token is not None and wall > first_token:
        decode_tok_s = round((completion_tokens - 1) / (wall - first_token), 1)
        estimated = True
    return {
        "finish": finish,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": completion_tokens,
        "reasoning_words": len(re.findall(r"\S+", reasoning_text)),
        "content_words": len(re.findall(r"\S+", content_text)),
        "prefill_ms": timings.get("prefill_ms"),
        "decode_tok_s": decode_tok_s,
        "decode_estimated": estimated,
        "ttft_s": round(first_token or 0, 3),
        "wall_s": round(wall, 3),
        "attractor": is_attractor(combined),
        "reasoning_content": reasoning_text,
        "content": content_text,
        "tool_calls": tool_text,
        "combined_content": combined,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="full local LFM model path")
    parser.add_argument("--tag", required=True, help="LFM registry tag")
    parser.add_argument("--registry", default=str(REPO / "cli/registry.json"))
    parser.add_argument("--device", default=os.environ.get("HIP_VISIBLE_DEVICES", "0"))
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--expect", default="Paris", help="case-insensitive expected text; empty disables")
    parser.add_argument("--reasoning-effort", default="medium", choices=EFFORTS)
    parser.add_argument("--thinking-budget", default="med")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-seq", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--port", type=int, default=0, help="serve port; 0 allocates a per-run port")
    parser.add_argument("--warm-timeout", type=float, default=120)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--home", default=None, help="isolated HOME; default allocates a per-run directory")
    parser.add_argument("--serve-log", default=None, help="serve log; default allocates a per-run path")
    parser.add_argument("--out", default=None)
    parser.add_argument("--bun", default="/home/kaden/.bun/bin/bun")
    parser.add_argument("--daemon-bin", default=str(REPO / "target/release/examples/daemon"))
    parser.add_argument("--nothink", action="store_true",
                        help="disable thinking: sends reasoning_effort='none' (index.ts maps that "
                             "to enable_thinking=false) and asserts the reasoning row is absent")
    parser.add_argument("--no-stream", dest="stream", action="store_false",
                        help="use the non-streaming /v1/chat/completions path")
    parser.set_defaults(stream=True)
    args = parser.parse_args()
    args.home, args.serve_log, args.port = allocate_runtime_defaults(
        args.home, args.serve_log, args.port
    )

    effort = "none" if args.nothink else args.reasoning_effort
    sampling = build_sampling(args.registry, args.tag, effort)
    prompt = load_prompt(args.prompt_file, args.prompt)
    print(f"LFM smoke: tag={args.tag} model={args.model} gpu={args.device} kv=q8", flush=True)
    print(f"runtime: port={args.port} home={args.home} log={args.serve_log}", flush=True)
    print(f"stream={args.stream} nothink={args.nothink} reasoning_effort={effort} sampling={sampling}", flush=True)
    try:
        model = spawn_server(args)
        row = request_turn(args, model, prompt, sampling, stream=args.stream)
    finally:
        _kill_server()

    errors = validate_turn(row, args.expect, forbid_reasoning=args.nothink)
    row["errors"] = errors
    print(
        f"finish={row['finish']} gen={row['completion_tokens']} "
        f"reasoning={row['reasoning_words']}w content={row['content_words']}w "
        f"decode={row['decode_tok_s']} tok/s attractor={row['attractor']}",
        flush=True,
    )
    print(f"reasoning: {row['reasoning_content'][:180]!r}", flush=True)
    print(f"content:   {row['content'][:180]!r}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(row, indent=2))
    if errors:
        print("LFM SMOKE FAIL: " + "; ".join(errors), file=sys.stderr)
        return 1
    print("LFM SMOKE PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
