#!/usr/bin/env bash
# Reproducible E2B/E4B quality and throughput A/B.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DAEMON="${DAEMON:-$ROOT/target/release/daemon}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/.hipfire/models/gemma4-eseries}"
E2B="${E2B:-$MODEL_ROOT/gemma4-e2b-it-pr439-q8.hfq}"
E4B="${E4B:-$MODEL_ROOT/gemma4-e4b-it-pr439-q8.hfq}"
DATASET="${DATASET:-$HOME/.hipfire/datasets/longbench-v2/longbench-hard30-pp32k.jsonl}"
MANIFEST="${MANIFEST:-$HOME/.hipfire/datasets/longbench-v2/longbench-hard30-pp32k.manifest.json}"
TASKS="${TASKS:-$ROOT/benchmarks/gemma4_eseries_long_decode.json}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/target/validation/gemma4-eseries-quality/$RUN_ID}"
GPU_ID="${GPU_ID:-1}"
LONG_LIMIT="${LONG_LIMIT:-0}"

for file in "$DAEMON" "$E2B" "$E4B" "$DATASET" "$MANIFEST" "$TASKS"; do
    [[ -f "$file" ]] || { echo "missing required file: $file" >&2; exit 2; }
done

run_model() {
    local label="$1" model="$2"
    python3 "$ROOT/scripts/eval_gemma4_eseries.py" \
        --daemon "$DAEMON" --model "$model" --model-label "$label" \
        --suite longbench --dataset "$DATASET" --manifest "$MANIFEST" \
        --out-dir "$OUT_ROOT/$label/longbench-hard30" --physical-gpu "$GPU_ID" \
        --max-seq 32768 --max-tokens 96 --limit "$LONG_LIMIT" --timeout 1800
    sleep 15
    python3 "$ROOT/scripts/eval_gemma4_eseries.py" \
        --daemon "$DAEMON" --model "$model" --model-label "$label" \
        --suite longdecode --tasks "$TASKS" \
        --out-dir "$OUT_ROOT/$label/longdecode" --physical-gpu "$GPU_ID" \
        --max-seq 8192 --max-tokens 2048 --timeout 1800
    sleep 15
}

mkdir -p "$OUT_ROOT"
run_model gemma4-e2b "$E2B"
run_model gemma4-e4b "$E4B"
echo "Gemma4 E-series quality A/B complete: $OUT_ROOT"
