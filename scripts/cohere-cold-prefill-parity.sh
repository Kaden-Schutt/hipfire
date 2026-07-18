#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Compare cold-start and warm-prefix batched prefills in one daemon process,
# then replay the warm-prefix request in a fresh process. The reset between
# requests must reproduce the first model turn exactly, while the warm request
# must be parity-identical to a cold process and report cold_start=false.

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
EXE="$ROOT/target/release/examples/daemon"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-${HIPFIRE_DIR:-$HOME/.hipfire}/models}"
MODEL="${HIPFIRE_COHERE_MODEL:-$MODELS_DIR/north-mini-code.mq4.hfq}"

if [[ ! -x "$EXE" || ! -f "$MODEL" ]]; then
    echo "cohere-cold-prefill-parity: missing daemon or Cohere fixture" >&2
    exit 2
fi

evidence_root="${HIPFIRE_EVIDENCE_DIR:-${TMPDIR:-/tmp}/hipfire-cor-002}"
mkdir -p "$evidence_root"
tmp=$(mktemp -d "$evidence_root/cohere-XXXXXX")
prompt='Write one short sentence explaining why cold starts should be deterministic.'

python3 - "$tmp" "$EXE" "$MODEL" "$prompt" <<'PY'
import json
import os
import subprocess
import sys

root, exe, model, prompt = sys.argv[1:]

def run_process(requests, stdout_path, stderr_path):
    p = subprocess.Popen(
        [exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    lines = []
    responses = {}
    stderr = ""
    try:
        for request in requests:
            p.stdin.write(json.dumps(request) + "\n")
            p.stdin.flush()
            response = []
            while True:
                line = p.stdout.readline()
                if not line:
                    raise SystemExit("Cohere daemon exited before terminal response")
                lines.append(line)
                response.append(line)
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                terminal = (
                    value.get("type") == "loaded" and request.get("type") == "load"
                ) or (
                    value.get("type") == "reset" and request.get("type") == "reset"
                ) or (
                    value.get("type") == "done" and value.get("id") == request.get("id")
                )
                if terminal:
                    responses[request.get("id", "")] = response
                    break
        p.stdin.write(json.dumps({"type": "unload"}) + "\n")
        p.stdin.flush()
        p.stdin.close()
        p.wait(timeout=30)
        stderr = p.stderr.read()
    finally:
        if p.poll() is None:
            p.kill()
    with open(stdout_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    with open(stderr_path, "w", encoding="utf-8") as f:
        f.write(stderr)
    return responses, stderr

def token_text(response):
    out = []
    for line in response:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("type") == "token":
            out.append(value.get("text", ""))
    return "".join(out)

load = {"type": "load", "model": model, "params": {"max_seq": 2048}}
first_request = {
    "type": "generate", "id": "r1", "prompt": prompt,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.0, "max_tokens": 32,
}

first_responses, first_stderr = run_process(
    [load, first_request, {"type": "reset"}, {**first_request, "id": "r2"}],
    os.path.join(root, "daemon.stdout"), os.path.join(root, "daemon.stderr"),
)
first = token_text(first_responses["r1"])
second = token_text(first_responses["r2"])
if not first or first != second:
    raise SystemExit("Cohere cold-prefill reset parity failed")

warm_request = {
    "type": "generate", "id": "warm",
    "prompt": "Now explain the same point using pears.",
    "messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": first},
        {"role": "user", "content": "Now explain the same point using pears."},
    ],
    # One deterministic token keeps the warm-prefix parity fixture focused on
    # the first target result, before the known Cohere marker/parser tail can
    # make a resident suffix differ from a fresh full-prefill suffix.
    "temperature": 0.0, "max_tokens": 1,
}
resident_warm_responses, resident_warm_stderr = run_process(
    [load, first_request, warm_request],
    os.path.join(root, "warm-resident.stdout"), os.path.join(root, "warm-resident.stderr"),
)
fresh_warm_responses, fresh_warm_stderr = run_process(
    [load, warm_request],
    os.path.join(root, "warm-cold.stdout"), os.path.join(root, "warm-cold.stderr"),
)
warm = token_text(resident_warm_responses["warm"])
fresh_warm = token_text(fresh_warm_responses["warm"])
if "mode=batched cold_start=true" not in first_stderr:
    raise SystemExit("cold batched prefill evidence missing")
if "cold_start=false" not in resident_warm_stderr:
    raise SystemExit("warm-prefix prefill evidence missing")
if "mode=batched cold_start=true" not in fresh_warm_stderr:
    raise SystemExit("fresh warm-prefix replay evidence missing")
if not warm or warm != fresh_warm:
    raise SystemExit("Cohere warm-prefix produced no output")

with open(os.path.join(root, "warm-request.json"), "w", encoding="utf-8") as f:
    json.dump(warm_request, f, indent=2)
print("Cohere cold + warm-prefix parity: PASS")
print(f"evidence={root}")
PY
