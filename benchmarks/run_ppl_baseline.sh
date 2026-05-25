#!/usr/bin/env bash
# Lloyd-Max baseline sweep: 9B/4B/0.8B × MQ4/MQ3 on wikitext2-test.
# Single-window, ctx=2048, warmup=8, offset=0.
# Output: stdout summary + per-model section.
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/gpu-lock.sh

CTX="${CTX:-2048}"
WARMUP="${WARMUP:-8}"
OFFSET="${OFFSET:-0}"
RESULTS="benchmarks/results/ppl_baseline_$(date -u +%Y%m%dT%H%M%SZ).md"

MODELS=(
  "/home/kaden/.hipfire/models/qwen3.5-0.8b.mq4"
  "/home/kaden/.hipfire/models/qwen3.5-0.8b.mq3"
  "/home/kaden/.hipfire/models/qwen3.5-4b.mq4"
  "/home/kaden/.hipfire/models/qwen3.5-4b.mq3"
  "/home/kaden/.hipfire/models/qwen3.5-9b.mq4"
  "/home/kaden/.hipfire/models/qwen3.5-9b.mq3"
)
CORPUS="dev/bench/data/wikitext2-test.txt"

{
  echo "# PPL baseline ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  echo
  echo "ctx=$CTX warmup=$WARMUP offset=$OFFSET corpus=$CORPUS"
  echo
  echo "| model | scored | NLL/tok | PPL |"
  echo "|---|---:|---:|---:|"
} > "$RESULTS"

for model in "${MODELS[@]}"; do
  if [[ ! -f "$model" ]]; then
    echo "(skip) missing: $model"
    continue
  fi
  echo "=== $model ==="
  gpu_acquire "ppl-$(basename "$model")"
  ./target/release/examples/perplexity "$model" "$CORPUS" \
    --ctx "$CTX" --warmup "$WARMUP" --offset "$OFFSET" 2>&1 | tee "/tmp/_ppl_$(basename "$model").log" || {
    echo "  ERROR: $model"
    gpu_release
    continue
  }
  gpu_release
  scored=$(grep -E "^Scored:" "/tmp/_ppl_$(basename "$model").log" | awk '{print $2}')
  nll=$(grep -E "^NLL/tok:" "/tmp/_ppl_$(basename "$model").log" | awk '{print $2}')
  ppl=$(grep -E "^PPL:" "/tmp/_ppl_$(basename "$model").log" | awk '{print $2}')
  echo "| $(basename "$model") | $scored | $nll | $ppl |" >> "$RESULTS"
done

echo "DONE -> $RESULTS"
cat "$RESULTS"
