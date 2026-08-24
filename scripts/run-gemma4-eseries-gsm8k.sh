#!/usr/bin/env bash
# Reproducible Gemma4 E-series GSM8K 8-shot CoT evaluation.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DAEMON="${DAEMON:-$ROOT/target/release/daemon}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/.hipfire/models/gemma4-eseries}"
E2B="${E2B:-$MODEL_ROOT/gemma4-e2b-it-pr439-q8.hfq}"
E4B="${E4B:-$MODEL_ROOT/gemma4-e4b-it-pr439-q8.hfq}"
DATASET="${DATASET:-$ROOT/target/validation/gemma4-eseries/gsm8k/test.jsonl}"
EXPECTED_DATASET_SHA256="3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/target/validation/gemma4-eseries-gsm8k/$RUN_ID}"
GPU_ID="${GPU_ID:-1}"
LIMIT="${LIMIT:-100}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_SEQ="${MAX_SEQ:-8192}"
PREFILL_BATCH="${PREFILL_BATCH:-8}"
RUNTIME_HOME="${RUNTIME_HOME:-/tmp/hipfire-gsm8k-gpu${GPU_ID}}"

for file in "$DAEMON" "$E2B" "$E4B" "$DATASET"; do
    [[ -f "$file" ]] || { echo "missing required file: $file" >&2; exit 2; }
done

actual_sha256="$(sha256sum "$DATASET" | awk '{print $1}')"
if [[ "$actual_sha256" != "$EXPECTED_DATASET_SHA256" ]]; then
    echo "GSM8K dataset SHA256 mismatch: $actual_sha256" >&2
    exit 2
fi

run_model() {
    local label="$1" model="$2"
    python3 "$ROOT/scripts/eval_gemma4_eseries.py" \
        --daemon "$DAEMON" --model "$model" --model-label "$label" \
        --suite gsm8k --dataset "$DATASET" \
        --out-dir "$OUT_ROOT/$label" --physical-gpu "$GPU_ID" \
        --max-seq "$MAX_SEQ" --max-tokens "$MAX_TOKENS" --limit "$LIMIT" \
        --prefill-batch "$PREFILL_BATCH" --runtime-home "$RUNTIME_HOME" --timeout 1800
}

mkdir -p "$OUT_ROOT"
run_model gemma4-e2b "$E2B"
sleep 15
run_model gemma4-e4b "$E4B"
echo "Gemma4 E-series GSM8K pilot complete: $OUT_ROOT"
