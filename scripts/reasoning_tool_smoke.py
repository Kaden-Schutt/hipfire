#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""
reasoning_tool_smoke — 10-turn Qwen3.8 tool-only smoke for typed reasoning contracts.

Exactly 10 sequential POSTs to /v1/chat/completions, each forcing a single
state_lookup tool call, locally executing the stub, and appending assistant
tool_calls + tool result to history. Validates hipfire.reasoning contract,
mode/effort/cap/cap_source, config_warnings, monotonic prompt_tokens and
history growth, arbitrary max_tokens, and an invalid thinking_budget drop.
Requires --serve-log and checks each config_warning is request-correlated
in that log as "[WARN: INVALID CONFIG] <exact text>".

Stdlib only (urllib, argparse, json, time, pathlib). No retries.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import urllib.parse
from pathlib import Path


TURN_MATRIX = [
    {
        "turn": 1,
        "key": "alpha",
        "prompt": "Call state_lookup with key='alpha'. You MUST call exactly one function; no text before.",
        "max_tokens": 256,
        "reasoning_effort": "xhigh",
        "max_think_tokens": None,
        "thinking_budget": None,
        "enable_thinking": False,
        "expected": {
            "contract": "qwen_jinja",
            "mode": "disabled",
            "effort": None,
            "cap": None,
            "cap_source": "none",
            "warnings_nonempty": True,
            "warning_substrings": ["dropped", "thinking disabled"],
        },
    },
    {
        "turn": 2,
        "key": "beta",
        "prompt": "Call state_lookup with key='beta'. Previous tool result was {value:'ok:alpha'}. Continue exactly one call.",
        "max_tokens": None,
        "reasoning_effort": "low",
        "max_think_tokens": None,
        "thinking_budget": None,
        "enable_thinking": None,
        "expected": {
            "contract": "qwen_jinja",
            "mode": "enabled",
            "effort": "low",
            "cap": None,
            "cap_source": "none",
            "warnings_nonempty": False,
        },
    },
    {
        "turn": 3,
        "key": "gamma",
        "prompt": "Call state_lookup with key='gamma'.",
        "max_tokens": None,
        "reasoning_effort": "medium",
        "max_think_tokens": None,
        "thinking_budget": None,
        "enable_thinking": None,
        "expected": {
            "contract": "qwen_jinja",
            "mode": "enabled",
            "effort": "medium",
            "cap": None,
            "cap_source": "none",
            "warnings_nonempty": False,
        },
    },
    {
        "turn": 4,
        "key": "delta",
        "prompt": "Call state_lookup with key='delta'.",
        "max_tokens": None,
        "reasoning_effort": "xhigh",
        "max_think_tokens": None,
        "thinking_budget": None,
        "enable_thinking": None,
        "expected": {
            "contract": "qwen_jinja",
            "mode": "enabled",
            "effort": "xhigh",
            "cap": None,
            "cap_source": "none",
            "warnings_nonempty": False,
        },
    },
    {
        "turn": 5,
        "key": "epsilon",
        "prompt": "Call state_lookup with key='epsilon'.",
        "max_tokens": 256,
        "reasoning_effort": "low",
        "max_think_tokens": 96,
        "thinking_budget": None,
        "enable_thinking": None,
        "expected": {
            "contract": "qwen_jinja",
            "mode": "enabled",
            "effort": "low",
            "cap": 96,
            "cap_source": "explicit:body:max_think_tokens",
            "warnings_nonempty": False,
        },
    },
    {
        "turn": 6,
        "key": "zeta",
        "prompt": "Call state_lookup with key='zeta'.",
        "max_tokens": 512,
        "reasoning_effort": "medium",
        "max_think_tokens": 256,
        "thinking_budget": None,
        "enable_thinking": None,
        "expected": {
            "contract": "qwen_jinja",
            "mode": "enabled",
            "effort": "medium",
            "cap": 256,
            "cap_source": "explicit:body:max_think_tokens",
            "warnings_nonempty": False,
        },
    },
    {
        "turn": 7,
        "key": "eta",
        "prompt": "Call state_lookup with key='eta'.",
        "max_tokens": 2048,
        "reasoning_effort": "xhigh",
        "max_think_tokens": 1024,
        "thinking_budget": None,
        "enable_thinking": None,
        "expected": {
            "contract": "qwen_jinja",
            "mode": "enabled",
            "effort": "xhigh",
            "cap": 1024,
            "cap_source": "explicit:body:max_think_tokens",
            "warnings_nonempty": False,
        },
    },
    {
        "turn": 8,
        "key": "theta",
        "prompt": "Call state_lookup with key='theta'.",
        "max_tokens": 256,
        "reasoning_effort": "low",
        "max_think_tokens": None,
        "thinking_budget": "high",
        "enable_thinking": None,
        "expected": {
            "contract": "qwen_jinja",
            "mode": "enabled",
            "effort": "low",
            "cap": None,
            "cap_source": "none",
            "warnings_nonempty": True,
            "warning_substrings": ["thinking_budget", "qwen_jinja", "max_think_tokens"],
        },
    },
    {
        "turn": 9,
        "key": "iota",
        "prompt": "Summarize keys seen: alpha,beta,gamma,delta,epsilon,zeta,eta,theta in order, then call state_lookup with key='iota'. Keep history; do not omit prior tool results.",
        "max_tokens": 4096,
        "reasoning_effort": "xhigh",
        "max_think_tokens": None,
        "thinking_budget": None,
        "enable_thinking": None,
        "expected": {
            "contract": "qwen_jinja",
            "mode": "enabled",
            "effort": "xhigh",
            "cap": None,
            "cap_source": "none",
            "warnings_nonempty": False,
        },
    },
    {
        "turn": 10,
        "key": "kappa",
        "prompt": "Final: confirm all 9 prior keys were ok, then call state_lookup with key='kappa'. Use arbitrary token limit.",
        "max_tokens": 3217,
        "reasoning_effort": "medium",
        "max_think_tokens": 512,
        "thinking_budget": None,
        "enable_thinking": None,
        "expected": {
            "contract": "qwen_jinja",
            "mode": "enabled",
            "effort": "medium",
            "cap": 512,
            "cap_source": "explicit:body:max_think_tokens",
            "warnings_nonempty": False,
        },
    },
]


def execute_state_lookup(key: str) -> str:
    return json.dumps({"value": f"ok:{key}"})


def http_post_json(url: str, body: dict, timeout: float):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
            text = raw.decode("utf-8", errors="replace")
            try:
                j = json.loads(text)
            except Exception:
                j = None
            return status, j, text
    except urllib.error.HTTPError as e:
        raw = e.read()
        text = raw.decode("utf-8", errors="replace") if raw else ""
        try:
            j = json.loads(text) if text else None
        except Exception:
            j = None
        return e.code, j, text
    except Exception as e:
        return None, None, str(e)


def http_get_json(url: str, timeout: float):
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1/chat/completions"):
        return endpoint
    if endpoint.endswith("/v1"):
        return endpoint + "/chat/completions"
    return endpoint + "/v1/chat/completions"


def main():
    parser = argparse.ArgumentParser(description="Qwen3.8 reasoning_tool_smoke 10-turn harness")
    parser.add_argument("--endpoint", dest="endpoint", default="http://127.0.0.1:11435", help="Base URL or full chat completions endpoint")
    parser.add_argument("--base-url", dest="endpoint", help=argparse.SUPPRESS)
    parser.add_argument("--model", dest="model", default="qwen3.8:27b", help="Model name")
    parser.add_argument("--out", dest="out", default="reasoning-tool-smoke.json", help="Output JSON report path")
    parser.add_argument("--output", dest="out", help=argparse.SUPPRESS)
    parser.add_argument("--timeout", dest="timeout", type=float, default=60.0, help="HTTP timeout seconds")
    parser.add_argument("--padding", dest="padding", type=int, default=0, help="Extra padding tokens (filler words) appended to each prompt")
    parser.add_argument("--serve-log", dest="serve_log", required=True, help="Path to serve.log for per-request warning verification")
    args = parser.parse_args()

    endpoint = normalize_endpoint(args.endpoint)
    base_for_health = endpoint.split("/v1/")[0]
    model = args.model
    out_path = Path(args.out)
    timeout = float(args.timeout)
    padding = int(args.padding)
    serve_log_path = Path(args.serve_log)


    started = time.time()
    meta = {
        "model": model,
        "endpoint": endpoint,
        "base_url": base_for_health,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "padding": padding,
        "timeout": timeout,
        "serve_log": str(serve_log_path),
    }

    # Require a readable serve log before any generation turn.
    try:
        with open(serve_log_path, "rb") as lf:
            lf.seek(0, 2)
            log_offset = lf.tell()
    except OSError as e:
        msg = f"serve-log unreadable before turn 1: {serve_log_path}: {e}"
        print(f"FAIL: {msg}", file=sys.stderr)
        report = {
            "meta": meta,
            "results": [],
            "summary": {"passed": 0, "failed": 10, "total": 10, "error": msg},
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        sys.exit(2)


    health = http_get_json(base_for_health + "/health", timeout=timeout)
    if health is None:
        msg = "hipfire /health endpoint is unreachable"
        print(f"FAIL: {msg}", file=sys.stderr)
        report = {
            "meta": meta,
            "results": [],
            "summary": {"passed": 0, "failed": 10, "total": 10, "error": msg},
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        sys.exit(2)
    contract = health.get("reasoning_contract") or health.get("current_reasoning_contract")
    if contract is not None and contract != "qwen_jinja":
        msg = f"health reasoning_contract={contract!r} expected qwen_jinja"
        print(f"FAIL: {msg}", file=sys.stderr)
        report = {
            "meta": meta,
            "results": [],
            "summary": {"passed": 0, "failed": 10, "total": 10, "error": msg},
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        sys.exit(2)



    system_msg = {"role": "system", "content": "You are tool-only. Always call exactly one state_lookup."}
    messages = [system_msg]
    results = []
    prev_prompt_tokens = -1
    failed = 0
    passed = 0
    padding_filler = (" pad" * padding).strip() if padding > 0 else ""

    for idx, turn in enumerate(TURN_MATRIX):
        key = turn["key"]
        expected = turn["expected"]
        prompt = turn["prompt"]
        if padding_filler:
            prompt = prompt + " " + padding_filler

        # Build request messages: history + current user
        user_msg = {"role": "user", "content": prompt}
        request_messages = messages + [user_msg]

        expected_history_len = 1 + 3 * idx
        if len(messages) != expected_history_len:
            msg = (
                f"turn {turn['turn']}: history length {len(messages)} "
                f"!= expected {expected_history_len}"
            )
            print(f"FAIL: {msg}", file=sys.stderr)
            results.append(
                {
                    "turn": turn["turn"],
                    "key": key,
                    "error": msg,
                    "request": {"messages": request_messages},
                }
            )
            failed += 1
            break

        # Tools: enum-singleton forces deterministic arg
        tool_def = {
            "type": "function",
            "function": {
                "name": "state_lookup",
                "description": "Lookup state by key",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string", "enum": [key]}},
                    "required": ["key"],
                },
            },
        }
        body = {
            "model": model,
            "messages": request_messages,
            "tools": [tool_def],
            "tool_choice": {"type": "function", "function": {"name": "state_lookup"}},
            "temperature": 0,
            "stream": False,
        }
        if turn.get("max_tokens") is not None:
            body["max_tokens"] = turn["max_tokens"]
        # Reasoning controls
        if turn.get("reasoning_effort") is not None:
            body["reasoning_effort"] = turn["reasoning_effort"]
        if turn.get("max_think_tokens") is not None:
            body["max_think_tokens"] = turn["max_think_tokens"]
        if turn.get("thinking_budget") is not None:
            body["thinking_budget"] = turn["thinking_budget"]
        if turn.get("enable_thinking") is not None:
            body["chat_template_kwargs"] = {"enable_thinking": turn["enable_thinking"]}

        # Capture serve.log byte offset immediately before this request.
        try:
            with open(serve_log_path, "rb") as lf:
                lf.seek(0, 2)
                log_offset = lf.tell()
        except OSError as e:
            msg = f"turn {turn['turn']} serve-log unreadable before request: {serve_log_path}: {e}"
            print(f"FAIL: {msg}", file=sys.stderr)
            results.append(
                {
                    "turn": turn["turn"],
                    "key": key,
                    "error": msg,
                    "request": body,
                }
            )
            failed += 1
            break

        # POST
        status, resp_json, raw_text = http_post_json(endpoint, body, timeout)

        record = {
            "turn": turn["turn"],
            "key": key,
            "request": body,
            "response_raw": raw_text[:8000] if isinstance(raw_text, str) else str(raw_text)[:8000],
        }

        if status is None:
            msg = f"turn {turn['turn']} infra error: {raw_text}"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            record["status"] = None
            results.append(record)
            failed += 1
            break

        record["status"] = status
        record["response"] = resp_json

        if status != 200:
            msg = f"turn {turn['turn']} HTTP {status} !=200 body={raw_text[:500]}"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break

        if not isinstance(resp_json, dict):
            msg = f"turn {turn['turn']} response not JSON object"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break

        if "error" in resp_json and resp_json["error"] is not None:
            msg = f"turn {turn['turn']} OpenAI error object: {resp_json['error']}"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break

        choices = resp_json.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            msg = f"turn {turn['turn']} missing choices"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break

        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if finish_reason != "tool_calls":
            msg = f"turn {turn['turn']} finish_reason={finish_reason!r} != tool_calls; choice={json.dumps(choice)[:500]}"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break

        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            msg = f"turn {turn['turn']} tool_calls len {len(tool_calls) if isinstance(tool_calls, list) else type(tool_calls)} !=1 ; message={json.dumps(message)[:800]}"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break

        tc = tool_calls[0]
        if tc.get("type") != "function":
            msg = f"turn {turn['turn']} tool_call type {tc.get('type')!r} != function"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break

        func = tc.get("function") or {}
        name = func.get("name")
        if name != "state_lookup":
            msg = f"turn {turn['turn']} tool name {name!r} != state_lookup"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break

        args_str = func.get("arguments")
        if not isinstance(args_str, str):
            msg = f"turn {turn['turn']} arguments not string: {args_str!r}"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break

        try:
            parsed_args = json.loads(args_str)
        except Exception as e:
            msg = f"turn {turn['turn']} arguments not valid JSON: {args_str!r} err={e}"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break

        if parsed_args != {"key": key}:
            msg = f"turn {turn['turn']} arguments {parsed_args!r} != {{'key':{key!r}}}"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break

        # No prose substitution: ensure we did not rely on content text; tool_calls must be sole source.
        # If model also returned content text alongside tool_calls, that's okay, but we must not have accepted prose without tool_calls (already enforced).

        # hipfire checks
        hipfire = resp_json.get("hipfire")
        if not isinstance(hipfire, dict):
            msg = f"turn {turn['turn']} missing hipfire object"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break

        reasoning = hipfire.get("reasoning")
        if reasoning is None or reasoning is False:
            msg = f"turn {turn['turn']} hipfire.reasoning missing/null"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break

        contract = reasoning.get("contract")
        if contract != expected["contract"]:
            msg = f"turn {turn['turn']} contract {contract!r} != {expected['contract']!r}"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break

        mode = reasoning.get("mode")
        if mode != expected["mode"]:
            msg = f"turn {turn['turn']} mode {mode!r} != {expected['mode']!r}"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break

        effort = reasoning.get("effort")
        # JSON null maps to Python None
        if effort != expected["effort"]:
            msg = f"turn {turn['turn']} effort {effort!r} != {expected['effort']!r}"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break

        cap = reasoning.get("max_think_tokens")
        if cap != expected["cap"]:
            msg = f"turn {turn['turn']} max_think_tokens {cap!r} != {expected['cap']!r}"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break

        cap_source = reasoning.get("cap_source")
        if cap_source != expected["cap_source"]:
            msg = f"turn {turn['turn']} cap_source {cap_source!r} != {expected['cap_source']!r}"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break

        warnings = hipfire.get("config_warnings")
        if not isinstance(warnings, list):
            msg = f"turn {turn['turn']} config_warnings not list: {warnings!r}"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break

        if expected.get("warnings_nonempty"):
            if len(warnings) == 0:
                msg = f"turn {turn['turn']} expected warnings non-empty but got []"
                print(f"FAIL: {msg}", file=sys.stderr)
                record["error"] = msg
                results.append(record)
                failed += 1
                break
            want = expected.get("warning_substrings") or []
            warnings_text = " ".join(str(w) for w in warnings)
            for sub in want:
                if sub not in warnings_text:
                    msg = f"turn {turn['turn']} warning missing substring {sub!r} in {warnings_text!r}"
                    print(f"FAIL: {msg}", file=sys.stderr)
                    record["error"] = msg
                    results.append(record)
                    failed += 1
                    break
            if failed > len(results):
                break
            # For turn 8 also check that warning contains invalid config hint; keep generic
        else:
            if len(warnings) != 0:
                msg = f"turn {turn['turn']} expected warnings [] but got {warnings!r}"
                print(f"FAIL: {msg}", file=sys.stderr)
                record["error"] = msg
                results.append(record)
                failed += 1
                break

        if record.get("error"):
            break

        # Request-correlated serve.log proof: only bytes appended for this request.
        before_offset = log_offset
        try:
            with open(serve_log_path, "rb") as lf:
                lf.seek(0, 2)
                end = lf.tell()
                if end < before_offset:
                    raise OSError(
                        f"serve.log truncated during turn {turn['turn']}: "
                        f"size {end} < offset {before_offset}"
                    )
                lf.seek(before_offset)
                delta = lf.read(end - before_offset)
            log_slice = delta.decode("utf-8", errors="replace")
            log_offset = end
        except OSError as e:
            msg = f"turn {turn['turn']} serve-log read error: {e}"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break
        record["log_offset_before"] = before_offset
        record["log_slice"] = log_slice[-4000:]
        log_ok = True
        for w in warnings:
            if not isinstance(w, str):
                msg = (
                    f"turn {turn['turn']} config_warnings entry not string: {w!r}"
                )
                print(f"FAIL: {msg}", file=sys.stderr)
                record["error"] = msg
                results.append(record)
                failed += 1
                log_ok = False
                break
            marker = f"[WARN: INVALID CONFIG] {w}"
            if marker not in log_slice:
                msg = (
                    f"turn {turn['turn']} serve.log missing request-correlated "
                    f"{marker!r} in +{len(delta)}B slice"
                )
                print(f"FAIL: {msg}", file=sys.stderr)
                record["error"] = msg
                results.append(record)
                failed += 1
                log_ok = False
                break
        marker_prefix = "[WARN: INVALID CONFIG] "
        logged_warnings = [
            line.split(marker_prefix, 1)[1].strip()
            for line in log_slice.splitlines()
            if marker_prefix in line
        ]
        unexpected_logged = [warning for warning in logged_warnings if warning not in warnings]
        if unexpected_logged:
            msg = (
                f"turn {turn['turn']} serve.log has config warnings absent from "
                f"response metadata: {unexpected_logged!r}"
            )
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            log_ok = False
        if not log_ok:
            break



        # Usage monotonic
        usage = resp_json.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        if not isinstance(prompt_tokens, int):
            # Try nested or alternative name
            prompt_tokens = usage.get("prompt_tokens")
        if isinstance(prompt_tokens, int):
            if prompt_tokens <= prev_prompt_tokens:
                msg = f"turn {turn['turn']} prompt_tokens {prompt_tokens} not strictly increasing vs {prev_prompt_tokens}"
                print(f"FAIL: {msg}", file=sys.stderr)
                record["error"] = msg
                results.append(record)
                failed += 1
                break
            prev_prompt_tokens = prompt_tokens
            record["usage"] = usage
        else:
            # If usage missing, treat as failure per spec says verify daemon reports usage
            msg = f"turn {turn['turn']} missing usage.prompt_tokens"
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break

        # Local execution
        tool_result_str = execute_state_lookup(key)
        # Validate arguments before execution already done
        tool_call_id = tc.get("id") or f"call_{idx}"
        # Append assistant tool_calls + tool response to history (exact)
        assistant_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                }
            ],
        }
        tool_msg = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": tool_result_str,
        }
        messages.append(user_msg)
        messages.append(assistant_msg)
        messages.append(tool_msg)
        expected_completed_len = 1 + 3 * (idx + 1)
        if len(messages) != expected_completed_len:
            msg = (
                f"turn {turn['turn']}: completed history length {len(messages)} "
                f"!= expected {expected_completed_len}"
            )
            print(f"FAIL: {msg}", file=sys.stderr)
            record["error"] = msg
            results.append(record)
            failed += 1
            break

        # Capture hipfire for record
        record["hipfire"] = hipfire
        record["tool_call"] = tc
        record["tool_result"] = tool_result_str
        # serve.log correlation already enforced above from the per-request slice

        results.append(record)
        passed += 1

        # Ensure exactly 10 turns; if we broke early, failed handled
        # Continue loop

    ended = time.time()
    meta["ended"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ended))
    meta["duration_s"] = round(ended - started, 3)

    # Final history length check: should be 1 + 3*passed (system + 3 per completed turn)
    # Spec expects monotonically growing prompt context; we already checked prompt_tokens.

    summary = {"passed": passed, "failed": failed, "total": len(TURN_MATRIX)}
    if passed == len(TURN_MATRIX) and failed == 0:
        summary["status"] = "pass"
    else:
        summary["status"] = "fail"
        if failed == 0 and passed != len(TURN_MATRIX):
            summary["failed"] = len(TURN_MATRIX) - passed
            failed = summary["failed"]

    report = {"meta": meta, "results": results, "summary": summary}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))

    # Nonzero on any mismatch
    if summary["status"] != "pass":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
