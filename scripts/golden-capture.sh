#!/usr/bin/env bash
# Capture committed token-id streams for the dispatch golden matrix.
# Run ONCE against the legacy (new-dispatch OFF) build to create references,
# then re-run against new builds and diff. Output: one .jsonl per case.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-crates/hipfire-dispatch-tests/fixtures/golden}"
MODELS="${HIPFIRE_MODELS_DIR:-${HIPFIRE_DIR:-$HOME/.hipfire}/models}"
PROMPT_FILE="benchmarks/prompts/lru_cache_pep8_strict.txt"
MAX_TOKENS=128
mkdir -p "$OUT"

md5sum "$PROMPT_FILE" | tee "$OUT/prompt.md5"

# case label -> model file (edit paths to match local model dir)
declare -A CASES=(
  [mq4]="qwen3.5-9b.mq4"
  [mq6]="qwen3.5-9b.mq6"
  [paro]="Qwen3.6-35B-A3B-PARO.mq4"
  [q8hfq]="qwen3.5-9b.q8hfq"
  [ds4_mq2lloyd]="deepseek4.mq2lloyd"
)

EXE="./target/release/examples/coherence_probe"
for label in "${!CASES[@]}"; do
  model="$MODELS/${CASES[$label]}"
  if [ ! -e "$model" ]; then echo "SKIP $label ($model missing)"; continue; fi
  echo "CAPTURE $label"
  HIPFIRE_EMIT_TOKEN_IDS=1 "$EXE" \
    --model "$model" --prompt-file "$PROMPT_FILE" \
    --max-tokens "$MAX_TOKENS" --temperature 0.0 \
    --emit-committed-jsonl "$OUT/$label.committed.jsonl"
done
echo "done -> $OUT"
