#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Fake omp for tests: emits realistic JSONL per /tmp/omp-probe.out."""

import json
import os
import sys
import time
from pathlib import Path

def _emit(assistant_text: str):
    events = [
        {"type": "session", "version": 3, "id": "test-session", "timestamp": "2026-09-02T00:00:00Z", "cwd": "/tmp"},
        {"type": "agent_start"},
        {"type": "turn_start"},
        {"type": "message_start", "message": {"role": "user", "content": [{"type": "text", "text": "prompt"}], "attribution": "user", "timestamp": 0}},
        {"type": "message_end", "message": {"role": "user", "content": [{"type": "text", "text": "prompt"}], "attribution": "user", "timestamp": 0}},
        {"type": "message_start", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "thinking...", "thinkingSignature": "{}"}, {"type": "text", "text": assistant_text, "textSignature": "{}"}], "api": "openai-responses", "provider": "fake", "model": "fake-model", "usage": {"input": 0, "output": 0}}},
        {"type": "message_end", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "thinking...", "thinkingSignature": "{}"}, {"type": "text", "text": assistant_text, "textSignature": "{}"}], "api": "openai-responses", "provider": "fake", "model": "fake-model", "usage": {"input": 0, "output": 0}}},
        {"type": "turn_end", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "thinking..."}, {"type": "text", "text": assistant_text}]}},
        {"type": "agent_end", "messages": []},
    ]
    for ev in events:
        sys.stdout.write(json.dumps(ev) + "\n")

def main():
    args = sys.argv[1:]
    # Real omp chdirs to `--cwd` relative to its own working directory and exits 1
    # if the target does not exist. Mirror that so a doubled relative path
    # (`pr/pr`) fails here exactly as it failed on the runner.
    if "--cwd" in args:
        target = args[args.index("--cwd") + 1]
        try:
            os.chdir(target)
        except OSError as e:
            sys.stderr.write(f"Error: Cannot change working directory to {target}: {e.strerror}: chdir '{os.getcwd()}' -> '{os.path.join(os.getcwd(), target)}'\n")
            sys.exit(1)
    # Dump env keys if requested (for investigate mode tests)
    env_dump_path = os.environ.get("FAKE_OMP_ENV_DUMP")
    if env_dump_path:
        try:
            Path(env_dump_path).parent.mkdir(parents=True, exist_ok=True)
            with open(env_dump_path, "w", encoding="utf-8") as f:
                json.dump({"keys": sorted(os.environ.keys()), "env": dict(os.environ)}, f, indent=2, sort_keys=True)
        except Exception:
            pass
    # Simulate sleep for timeout tests
    sleep_val = os.environ.get("FAKE_OMP_SLEEP")
    if sleep_val:
        try:
            time.sleep(float(sleep_val))
        except Exception:
            pass
    log_path = os.environ.get("FAKE_OMP_LOG")
    if log_path:
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                json.dump({"args": args}, f)
                f.write("\n")
        except Exception:
            pass

    count_path = os.environ.get("FAKE_OMP_CALL_COUNT")
    call_idx = 0
    if count_path and Path(count_path).is_file():
        try:
            call_idx = int(Path(count_path).read_text().strip() or "0")
        except Exception:
            call_idx = 0
    if count_path:
        try:
            Path(count_path).parent.mkdir(parents=True, exist_ok=True)
            Path(count_path).write_text(str(call_idx + 1))
        except Exception:
            pass

    if os.environ.get("FAKE_OMP_GARBAGE") == "1":
        _emit("this is not json at all — garbage !@#")
        sys.exit(0)

    responses_path = os.environ.get("FAKE_OMP_RESPONSES")
    if responses_path and Path(responses_path).is_file():
        try:
            responses = json.loads(Path(responses_path).read_text())
            if call_idx < len(responses):
                entry = responses[call_idx]
            else:
                entry = responses[-1] if responses else {}
            if "json" in entry:
                text = json.dumps(entry["json"])
                if entry.get("fenced"):
                    text = "Here is the result:\n```json\n" + text + "\n```\n"
                elif entry.get("prose"):
                    text = "Sure, here it is: " + text + " hope that helps."
                _emit(text)
                exit_code_val2 = os.environ.get("FAKE_OMP_EXIT_CODE")
                if exit_code_val2 not in (None, ""):
                    try:
                        sys.exit(int(exit_code_val2))
                    except ValueError:
                        sys.exit(1)
                sys.exit(0)
            elif "text" in entry:
                _emit(entry["text"])
                exit_code_val2 = os.environ.get("FAKE_OMP_EXIT_CODE")
                if exit_code_val2 not in (None, ""):
                    try:
                        sys.exit(int(exit_code_val2))
                    except ValueError:
                        sys.exit(1)
                sys.exit(0)
        except Exception as e:
            sys.stderr.write(f"fake_omp response load failed: {e}\n")

    # Default canned responses: determine phase by call_idx and model arg
    # Try to infer model from args
    model = ""
    for i, a in enumerate(args):
        if a == "--model" and i+1 < len(args):
            model = args[i+1]
    if "fable" in model or "decide" in " ".join(args):
        # fable decide default
        text = json.dumps({
            "phase": "decide",
            "decision": "merge-staging",
            "agrees_with_sol": True,
            "override": None,
            "regressions": [],
            "further_evidence_wanted": [],
            "rationale": "fable rationale",
            "announcement": "Fable merges to staging."
        })
    elif call_idx == 0:
        text = json.dumps({
            "phase": "prelim",
            "summary": "test change",
            "surfaces": ["load"],
            "suspected_regressions": [],
            "run_hardware": True,
            "run_hardware_reasons": ["safe"],
            "routes": [{"mode": "battery", "tag": "qwen3.6:27b", "source": "sol", "why": "test"}],
            "unavailable_routes": [],
            "claim_assessment": "no claim",
            "questions_for_author": []
        })
    else:
        text = json.dumps({
            "phase": "verdict",
            "decision": "greenlight",
            "confidence": 0.9,
            "regressions": [],
            "coverage": {"surfaces_touched": ["load"], "surfaces_evidenced": ["load"], "gaps": []},
            "claim_verdict": "no-claim",
            "eyeball": [],
            "rationale": "all evidenced"
        })
        override = os.environ.get("FAKE_OMP_VERDICT_DECISION")
        if override:
            obj = json.loads(text)
            obj["decision"] = override
            text = json.dumps(obj)
    _emit(text)
    # Honor explicit exit code after emitting (so investigation can be parsed even on failure)
    exit_code_val = os.environ.get("FAKE_OMP_EXIT_CODE")
    if exit_code_val not in (None, ""):
        try:
            sys.exit(int(exit_code_val))
        except ValueError:
            sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
