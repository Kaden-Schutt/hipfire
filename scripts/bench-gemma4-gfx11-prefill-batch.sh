#!/usr/bin/env bash
# Reproducible Gemma 4 E-series prefill-batch sweep on one gfx1100 GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DAEMON="${DAEMON:-$ROOT/target/release/examples/daemon}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/.hipfire/models/gemma4-eseries}"
E2B="${E2B:-$MODEL_ROOT/gemma4-e2b-it-pr439-q8.hfq}"
E4B="${E4B:-$MODEL_ROOT/gemma4-e4b-it-pr439-q8.hfq}"
DATASET="${DATASET:-$ROOT/target/validation/gemma4-eseries/gsm8k/test.jsonl}"
GPU_ID="${GPU_ID:-0}"
BATCHES="${BATCHES:-8 16 32 64}"
LIMIT="${LIMIT:-10}"
MAX_TOKENS="${MAX_TOKENS:-16}"
COOLDOWN="${COOLDOWN:-10}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/target/validation/gemma4-gfx11-prefill-batch/$RUN_ID}"

for file in "$DAEMON" "$E2B" "$E4B" "$DATASET"; do
    [[ -f "$file" ]] || { echo "missing required file: $file" >&2; exit 2; }
done

mkdir -p "$OUT_ROOT"
for model in e2b e4b; do
    if [[ "$model" == e2b ]]; then artifact="$E2B"; else artifact="$E4B"; fi
    for batch in $BATCHES; do
        python3 "$ROOT/scripts/eval_gemma4_eseries.py" \
            --daemon "$DAEMON" --model "$artifact" \
            --model-label "gemma4-${model}-gfx11-b${batch}" \
            --suite gsm8k --dataset "$DATASET" \
            --out-dir "$OUT_ROOT/$model/b${batch}" --physical-gpu "$GPU_ID" \
            --runtime-home "/tmp/hipfire-gemma4-${model}-gfx11-b${batch}" \
            --max-seq 8192 --max-tokens "$MAX_TOKENS" --limit "$LIMIT" \
            --prefill-batch "$batch" --timeout 1800
        sleep "$COOLDOWN"
    done
done

echo "Gemma 4 gfx1100 prefill-batch sweep complete: $OUT_ROOT"
