#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kevin Read
# hipfire — see LICENSE and NOTICE in the project root.

# Standing benchmark for hipfire quant quality.
#
# Triples MSE (per-tensor) + reasoning-smoke (spiral or coherent) into a
# single results file. KLD (vs BF16 reference logits) is on the roadmap
# but currently requires external infrastructure (a vLLM/transformers
# reference run on the same prompt); see docs/plans/qwen35-mq4-quality-gap.md
# for the planned wiring.
#
# Usage:
#   scripts/bench_quant_quality.sh <safetensors_dir_or_file> <model.hfq> [out.md]
#
# Example:
#   scripts/bench_quant_quality.sh \
#       ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/SNAP \
#       /local/hipfire/qwen3.6-35b-a3b.mq4 \
#       benchmarks/results/quant_quality_a3b_baseline.md
#
# Output is a markdown report with:
# - Per-tensor MSE table (top 50 by MSE descending)
# - Aggregate MSE stats by quant type (qt → mean / p99 / max)
# - Train-pursuit reasoning smoke test result
#
# When iterating on quantizer formats / scale-search algorithms, run this
# before and after the change. The MSE delta on attention/FFN tensors
# predicts the KLD impact; the reasoning smoke test catches attractor
# regressions that aggregate MSE doesn't.

set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
    echo "usage: $0 <safetensors_dir_or_file> <model.hfq> [out.md]"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi
if [ $# -lt 2 ]; then
    usage
    exit 2
fi

ST_DIR="$1"
HFQ_PATH="$2"
OUT="${3:-benchmarks/results/quant_quality_$(basename "$HFQ_PATH" .hfq)_$(date -u +%Y%m%dT%H%M%SZ).md}"

if [ ! -e "$ST_DIR" ]; then
    echo "error: safetensors dir/file not found: $ST_DIR"
    exit 1
fi
if [ ! -e "$HFQ_PATH" ]; then
    echo "error: hfq file not found: $HFQ_PATH"
    exit 1
fi

wait_for_model_ready() {
    local hfq_path="$1"; local timeout="${2:-120}"
    local want; want=$(basename "$hfq_path")
    local start; start=$(date +%s)
    local tmp; tmp=$(mktemp)
    while [ $(( $(date +%s) - start )) -lt "$timeout" ]; do
        if curl -sS --max-time 3 -o "$tmp" http://127.0.0.1:8080/v1/models 2>/dev/null; then
            if python3 - "$tmp" "$want" <<'PY' 2>/dev/null; then
import json
import sys

path, want = sys.argv[1], sys.argv[2]
try:
    with open(path) as f:
        payload = json.load(f)
except Exception:
    sys.exit(1)
sys.exit(0 if any(m.get("id", "").endswith(want) for m in payload.get("data", [])) else 1)
PY
                rm -f "$tmp"; return 0
            fi
        fi
        sleep 2
    done
    rm -f "$tmp"; return 1
}

model_id_for_path() {
    local hfq_path="$1"
    local want; want=$(basename "$hfq_path")
    local tmp; tmp=$(mktemp)
    if ! curl -sS --max-time 3 -o "$tmp" http://127.0.0.1:8080/v1/models 2>/dev/null; then
        rm -f "$tmp"
        return 0
    fi
    python3 - "$tmp" "$want" <<'PY' | head -1
import json
import sys

path, want = sys.argv[1], sys.argv[2]
try:
    with open(path) as f:
        models = json.load(f).get("data", [])
except Exception:
    models = []
for model in models:
    model_id = model.get("id", "")
    if model_id.endswith(want):
        print(model_id)
        break
PY
    rm -f "$tmp"
}

mkdir -p "$(dirname "$OUT")"

# Build prerequisites if not already
echo "Building prerequisites..."
cargo build --release --example quant_quality_mse --quiet 2>&1 | tail -5
cargo build --release --example dump_norms --quiet 2>&1 | tail -5

PROMPT='A train leaves Station A traveling at 60 km/h. Two hours later, a second train leaves Station A on the same track traveling at 90 km/h. How long after the second train departs will it catch up to the first? Show your reasoning step by step.'

{
    echo "# Quant quality bench: $(basename "$HFQ_PATH")"
    echo
    echo "**Date:** $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "**Source safetensors:** \`$ST_DIR\`"
    echo "**Quantized file:** \`$HFQ_PATH\`"
    echo "**Size:** $(ls -la "$HFQ_PATH" | awk '{print $5}') bytes"
    echo
    echo "## Per-tensor MSE vs BF16 reference"
    echo
    echo '```'
} > "$OUT"

echo "Running per-tensor MSE..."
./target/release/examples/quant_quality_mse "$ST_DIR" "$HFQ_PATH" 2>&1 \
    | tee -a /tmp/_quant_mse.log >> "$OUT"

{
    echo '```'
    echo
    echo "## Final norm sanity"
    echo
    echo '```'
} >> "$OUT"

./target/release/examples/dump_norms "$HFQ_PATH" "language_model.norm.weight" 2>&1 \
    | tail -5 >> "$OUT"

{
    echo '```'
    echo
    echo "## Reasoning smoke test (train pursuit, temp=0, max_tokens=400)"
    echo
} >> "$OUT"

# Reasoning smoke test — only run if a daemon can be started
if ! command -v hipfire >/dev/null 2>&1; then
    echo "  (skip: hipfire CLI not on PATH)" >> "$OUT"
    echo "Skipping reasoning smoke test (no hipfire CLI)"
else
    hipfire stop 2>&1 | head -1 || true
    sleep 2
    HIPFIRE_MODEL="$HFQ_PATH" hipfire serve 8080 -d 2>&1 | tail -2
    if ! wait_for_model_ready "$HFQ_PATH" 300; then
        echo "  daemon failed to list requested model: $(basename "$HFQ_PATH")" | tee -a "$OUT"
        hipfire stop 2>&1 | head -1 || true
        exit 1
    fi

    # Find the model id (it should be the basename of the hfq)
    MODEL_ID=$(model_id_for_path "$HFQ_PATH")
    if [ -z "$MODEL_ID" ]; then
        # Fallback: assume basename
        MODEL_ID="$(basename "$HFQ_PATH")"
    fi
    echo "Model id: $MODEL_ID"

    {
        echo "Model: \`$MODEL_ID\`"
        echo
        echo '```'
    } >> "$OUT"

    timeout 240 curl -sS http://127.0.0.1:8080/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d "$(python3 -c "
import json
print(json.dumps({
  'model': '$MODEL_ID',
  'messages': [{'role':'user','content':'''$PROMPT'''}],
  'temperature': 0,
  'max_tokens': 400,
}))
")" > /tmp/_smoke_default.json 2>&1 || true

    python3 -c "
import json
try:
    d=json.load(open('/tmp/_smoke_default.json'))
    c=d['choices'][0]
    print('finish_reason:', c['finish_reason'])
    print('completion_tokens:', d['usage']['completion_tokens'])
    print('content_len:', len(c['message']['content']))
    if len(c['message']['content']) == 0:
        print('VERDICT: SPIRAL (empty content after <think> strip)')
    elif len(c['message']['content']) > 800:
        print('VERDICT: COHERENT (' + str(len(c['message']['content'])) + ' chars)')
    else:
        print('VERDICT: PARTIAL (' + str(len(c['message']['content'])) + ' chars)')
    print()
    print('--- preview (first 400 chars) ---')
    print(c['message']['content'][:400])
except Exception as e:
    print('ERROR:', e)
" >> "$OUT"

    echo '```' >> "$OUT"

    hipfire stop 2>&1 | head -1 || true
fi

echo
echo "Wrote: $OUT"
echo
echo "=== Summary ==="
grep -E "^(VERDICT|MQ4|Q8_0|F16|F32|mean|completion_tokens|tensor)" "$OUT" | head -30
