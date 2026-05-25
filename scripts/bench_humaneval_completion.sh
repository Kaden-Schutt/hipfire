#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kevin Read
# hipfire — see LICENSE and NOTICE in the project root.

# bench_humaneval_completion.sh — completion-capture harness for HumanEval prompts.
#
# Phase A Step 0 deliverable: per-format completion capture on a small
# HumanEval sample. Spawns the daemon on a quantized model, prompts with
# each in-tree humaneval prompt, captures the completion at temp=0, writes
# a JSONL file with {prompt_file, completion, tokens_used, finish_reason}
# per prompt.
#
# **Not pass@1 scoring yet** — that's a Step 0.5 follow-up that needs
# subprocess-isolated Python eval. This harness produces the completion
# strings; scoring is a separate concern.
#
# Usage:
#   scripts/bench_humaneval_completion.sh <model.hfq> [out.jsonl]
#
# Output JSONL (one row per prompt):
#   {"prompt_file": "...", "completion": "...", "tokens_used": N,
#    "finish_reason": "stop"|"length", "wall_ms": N}
#
# Used by scripts/quant_cohort.sh to drive the {MSE, KLD, PPL, HumanEval}
# multi-metric Phase A bench.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
    echo "usage: $0 <model.hfq> [out.jsonl]"
    echo "       prompts: benchmarks/prompts/humaneval_*.txt"
    exit 2
fi

HFQ_PATH="$1"
OUT="${2:-benchmarks/results/humaneval_$(basename "$HFQ_PATH" .hfq)_$(date -u +%Y%m%dT%H%M%SZ).jsonl}"

if [ ! -e "$HFQ_PATH" ]; then
    echo "error: hfq file not found: $HFQ_PATH"
    exit 1
fi

PROMPTS=( $(ls benchmarks/prompts/humaneval_*.txt 2>/dev/null) )
if [ ${#PROMPTS[@]} -eq 0 ]; then
    echo "error: no humaneval prompts in benchmarks/prompts/"
    exit 1
fi

if ! command -v hipfire >/dev/null 2>&1; then
    echo "error: hipfire CLI not on PATH"
    exit 1
fi

mkdir -p "$(dirname "$OUT")"
: > "$OUT"

# Stop any previous daemon, then start fresh.
hipfire stop 2>&1 | head -1 || true
sleep 2
echo "Starting daemon with model: $HFQ_PATH"
HIPFIRE_MODEL="$HFQ_PATH" hipfire serve 8080 -d 2>&1 | tail -2

# Wait until /v1/models reports the requested model is registered. The CLI's
# own `serve -d` polls /health before returning, but /v1/models is the
# authoritative "right model is loaded" gate. Replaces the legacy
# `tail -1 serve.log | grep "warm-up complete"` check, which broke when
# serve.log had a stale "warm-up complete" line from the previous session
# (the file is opened O_APPEND), and the `pgrep -af "examples/daemon"`
# fallback that fired falsely because the CLI no longer spawns a process
# named "examples/daemon" (it's `bun ... serve <port>` now).
want=$(basename "$HFQ_PATH")
warmup_start=$(date +%s)
ready=0
tmp=$(mktemp)
while [ $(( $(date +%s) - warmup_start )) -lt 300 ]; do
    if curl -sS --max-time 3 -o "$tmp" http://127.0.0.1:8080/v1/models 2>/dev/null; then
        if python3 -c "
import sys, json
try:
    with open('$tmp') as f:
        d = json.load(f)
    sys.exit(0 if any(m.get('id','').endswith('$want') for m in d.get('data', [])) else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
            ready=1
            break
        fi
    fi
    sleep 2
done
rm -f "$tmp"
if [ "$ready" != "1" ]; then
    echo "error: daemon /v1/models did not list '$want' within 300s"
    hipfire stop || true
    exit 1
fi

MODEL_ID=$(curl -sS http://127.0.0.1:8080/v1/models 2>/dev/null \
    | python3 -c "import sys,json; ms=json.load(sys.stdin)['data']; n='$(basename "$HFQ_PATH")'; [print(m['id']) for m in ms if m['id'].endswith(n)]" \
    | head -1)
if [ -z "$MODEL_ID" ]; then
    MODEL_ID="$(basename "$HFQ_PATH")"
fi
echo "Model id: $MODEL_ID"
echo

for prompt_file in "${PROMPTS[@]}"; do
    prompt_name=$(basename "$prompt_file" .txt)
    echo "  $prompt_name ..."

    # Read prompt; json-escape via python.
    body=$(python3 -c "
import json, sys
with open('$prompt_file') as f:
    p = f.read()
print(json.dumps({
    'model': '$MODEL_ID',
    'messages': [{'role':'user','content': p}],
    'temperature': 0,
    'max_tokens': 512,
}))
")

    t0=$(date +%s%3N)
    resp=$(timeout 300 curl -sS http://127.0.0.1:8080/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d "$body" 2>&1 || echo '{"error":"timeout-or-curl-failure"}')
    t1=$(date +%s%3N)
    wall_ms=$(( t1 - t0 ))

    python3 -c "
import json
prompt_file = '$prompt_file'
wall_ms = $wall_ms
raw = '''$resp'''
try:
    d = json.loads(raw)
except Exception as e:
    print(json.dumps({'prompt_file': prompt_file, 'error': f'parse: {e}', 'wall_ms': wall_ms}))
    raise SystemExit

if 'error' in d:
    print(json.dumps({'prompt_file': prompt_file, 'error': str(d['error']), 'wall_ms': wall_ms}))
    raise SystemExit

c = d['choices'][0]
print(json.dumps({
    'prompt_file': prompt_file,
    'completion': c['message']['content'],
    'tokens_used': d['usage']['completion_tokens'],
    'finish_reason': c['finish_reason'],
    'wall_ms': wall_ms,
}))
" >> "$OUT"
done

hipfire stop 2>&1 | head -1 || true

echo
echo "Wrote: $OUT"
echo
echo "=== Summary ==="
python3 -c "
import json
rows = [json.loads(l) for l in open('$OUT')]
print(f'{\"prompt\":<40s} {\"tokens\":>7s} {\"wall_ms\":>8s} {\"finish_reason\":>14s}')
for r in rows:
    if 'error' in r:
        print(f'  {r[\"prompt_file\"]}: ERROR: {r[\"error\"]}')
        continue
    name = r['prompt_file'].split('/')[-1].replace('.txt', '')
    print(f'{name:<40s} {r[\"tokens_used\"]:>7d} {r[\"wall_ms\"]:>8d} {r[\"finish_reason\"]:>14s}')
"
