#!/usr/bin/env bash
# Lloyd-Max vs uniform MQ2 vs MQ3 comparison.
# Models prepared in advance — this script just runs ppl on each and tabulates.
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/gpu-lock.sh

CTX="${CTX:-2048}"
WARMUP="${WARMUP:-8}"
RESULTS="benchmarks/results/ppl_lloyd_compare_$(date -u +%Y%m%dT%H%M%SZ).md"
CORPUS="dev/bench/data/wikitext2-test.txt"

MODELS=(
  "/home/kaden/.hipfire/models/qwen3.5-0.8b.mq4"
  "/home/kaden/.hipfire/models/qwen3.5-0.8b.mq3"
  "/home/kaden/.hipfire/models/qwen3.5-0.8b.mq2"
  "/home/kaden/.hipfire/models/qwen3.5-0.8b.mq2-lloyd"
  "/home/kaden/.hipfire/models/qwen3.5-4b.mq4"
  "/home/kaden/.hipfire/models/qwen3.5-4b.mq3"
  "/home/kaden/.hipfire/models/qwen3.5-4b.mq2"
  "/home/kaden/.hipfire/models/qwen3.5-4b.mq2-lloyd"
  "/home/kaden/.hipfire/models/qwen3.5-9b.mq4"
  "/home/kaden/.hipfire/models/qwen3.5-9b.mq3"
  "/home/kaden/.hipfire/models/qwen3.5-9b.mq2"
  "/home/kaden/.hipfire/models/qwen3.5-9b.mq2-lloyd"
)

{
  echo "# Lloyd-Max comparison ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  echo
  echo "ctx=$CTX warmup=$WARMUP corpus=$CORPUS"
  echo
  echo "| model | size | scored | NLL/tok | PPL |"
  echo "|---|---|---:|---:|---:|"
} > "$RESULTS"

for model in "${MODELS[@]}"; do
  if [[ ! -f "$model" ]]; then
    echo "(skip) missing: $model"
    continue
  fi
  echo "=== $model ==="
  size_bytes=$(stat -c%s "$model")
  size_mb=$(printf "%.1f" "$(echo "$size_bytes/1024/1024" | bc -l)")
  gpu_acquire "ppl-$(basename "$model")"
  log="/tmp/_ppl_$(basename "$model").log"
  ./target/release/examples/perplexity "$model" "$CORPUS" --ctx "$CTX" --warmup "$WARMUP" 2>&1 | tee "$log" || {
    echo "  ERROR: $model"
    gpu_release
    continue
  }
  gpu_release
  scored=$(grep -E "^Scored:" "$log" | awk '{print $2}')
  nll=$(grep -E "^NLL/tok:" "$log" | awk '{print $2}')
  ppl=$(grep -E "^PPL:" "$log" | awk '{print $2}')
  echo "| $(basename "$model") | ${size_mb}MB | $scored | $nll | $ppl |" >> "$RESULTS"
done

echo
echo "DONE -> $RESULTS"
echo
cat "$RESULTS"
