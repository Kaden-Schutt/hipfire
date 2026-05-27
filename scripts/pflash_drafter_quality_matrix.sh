#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kevin Read
#
# Overnight PFlash drafter-quality × keep-ratio matrix for the
# fix/q8-batched-masked-no-lds-cap investigation (2026-05-27).
#
# Question: at keep=0.30 the 0.8B-MQ4 drafter recovers only 2/3 multi-needle
# (misses depth-0.25), FLAT across keep ratios. Is the limiter (a) drafter
# CAPACITY (0.8B vs 4B), (b) drafter QUANT quality (MQ4 vs near-lossless Q8),
# or (c) inherent to the model? This matrix isolates each axis.
#
# Axes:
#   drafters: 0.8B-MQ4 (cap- + quant-limited, baseline)
#             0.8B-Q8  (quant control: same cap, near-lossless quant)
#             4B-MQ4   (capacity control: bigger, still MQ4)
#   keep:     0.30 0.20 0.15 0.10
#   fixture:  niah_multi_64k (3 needles @ depth .25/.5/.75) — the hard test
#
# Target: qwen3.5-9b.q8. Drafter KV: fwht3 (long-ctx default). q8 target KV.
# Each cell: full compress→prefill→decode; records kept tokens, needles
# recovered (n/3 + which), wall, PASS/FAIL. Results → TSV + full logs.
#
# Run:  nohup ./scripts/pflash_drafter_quality_matrix.sh >/tmp/pfmatrix.out 2>&1 &
set -u

cd "$(dirname "$0")/.." || exit 1
export HIPFIRE_MODELS_DIR=/local/hipfire
export HIPFIRE_PFLASH_DRAFTER_KV=fwht3

BENCH=./target/release/examples/pflash_niah_bench
TARGET=/local/hipfire/qwen3.5-9b.q8
FIXTURE=benchmarks/longctx/niah/niah_multi_64k.jsonl
SINGLE=benchmarks/longctx/niah/niah_32k.jsonl   # 21551-tok single-needle sanity

STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR=/local/hipfire/pfmatrix_$STAMP
mkdir -p "$OUTDIR"
TSV="$OUTDIR/results.tsv"
SUMMARY="$OUTDIR/SUMMARY.md"
echo -e "drafter\tdrafter_path\tkeep\tfixture\tsrc_tok\tkept_tok\trecovered\tverdict\twall_ms\tfound\tmissed" > "$TSV"

echo "=== PFlash drafter-quality matrix $STAMP ===" | tee "$SUMMARY"
echo "target=$TARGET  drafter_kv=fwht3  fixture=multi_64k(3-needle)" | tee -a "$SUMMARY"
echo "outdir=$OUTDIR" | tee -a "$SUMMARY"

# Drafter list: label|path  (skip any whose file is missing)
DRAFTERS=(
  "0.8B-MQ4|/local/hipfire/qwen3.5-0.8b.mq4"
  "0.8B-Q8|/local/hipfire/qwen3.5-0.8b.q8"
  "4B-MQ4|/local/hipfire/qwen3.5-4b.mq4"
)
KEEPS=(0.30 0.20 0.15 0.10)

# Build the bench if missing.
if [ ! -x "$BENCH" ]; then
  echo "[matrix] building pflash_niah_bench..." | tee -a "$SUMMARY"
  cargo build --release --features deltanet --example pflash_niah_bench 2>&1 | tail -2 | tee -a "$SUMMARY"
fi

source scripts/gpu-lock.sh

run_cell() {
  local label="$1" dpath="$2" keep="$3" fixture="$4" tag="$5"
  local log="$OUTDIR/${label//\//_}_keep${keep}_${tag}.log"
  if [ ! -e "$dpath" ]; then
    echo "[matrix] SKIP $label keep=$keep — drafter missing: $dpath" | tee -a "$SUMMARY"
    echo -e "$label\t$dpath\t$keep\t$tag\tNA\tNA\tSKIP\tMISSING\tNA\t\t" >> "$TSV"
    return
  fi
  echo "[matrix] >>> $label keep=$keep fixture=$tag" | tee -a "$SUMMARY"
  local t0 t1 wall
  t0=$(date +%s%3N)
  # maxgen 80 to fit 3 needle answers; gpu-lock handled internally by us.
  timeout 3600 "$BENCH" "$TARGET" "$fixture" \
      --maxgen 80 --q8kv --pflash "$dpath" --keep-ratio "$keep" \
      > "$log" 2>&1
  local rc=$?
  t1=$(date +%s%3N); wall=$((t1 - t0))
  # Parse structured fields from the log.
  local src kept rec verdict found missed
  src=$(grep -oE "tokenize:.*\(([0-9]+) tokens\)" "$log" | grep -oE "[0-9]+ tokens" | grep -oE "[0-9]+" | head -1)
  [ -z "$src" ] && src=$(grep -oE "source tokens.*\(([0-9]+)" "$log" | grep -oE "[0-9]+" | tail -1)
  kept=$(grep -oE "compressed:.*-> ([0-9]+) tokens" "$log" | grep -oE "> [0-9]+" | grep -oE "[0-9]+" | head -1)
  rec=$(grep -oE "recovered: [0-9]+ / [0-9]+" "$log" | head -1)
  # Verdict only from the real VERDICT-section lines (substring wording),
  # not early setup FAILs.
  if grep -qE "^PASS: [0-9]+ substring" "$log"; then verdict=PASS
  elif grep -qE "^FAIL: [0-9]+ of [0-9]+ substring" "$log"; then verdict=FAIL
  elif [ $rc -eq 124 ]; then verdict=TIMEOUT
  else verdict=ERR; fi
  found=$(grep -oE '\[\+\] "[^"]+"' "$log" | grep -oE '"[^"]+"' | tr -d '"' | paste -sd, -)
  missed=$(grep -oE '\[-\] "[^"]+"' "$log" | grep -oE '"[^"]+"' | tr -d '"' | paste -sd, -)
  echo -e "$label\t$dpath\t$keep\t$tag\t${src:-?}\t${kept:-?}\t${rec:-?}\t${verdict:-?}\t$wall\t$found\t$missed" >> "$TSV"
  echo "    -> kept=${kept:-?} $rec verdict=${verdict:-?} wall=${wall}ms  missed=[$missed]" | tee -a "$SUMMARY"
}

# ── Main matrix: every drafter × every keep, on the 3-needle fixture. ──
for d in "${DRAFTERS[@]}"; do
  label="${d%%|*}"; dpath="${d##*|}"
  for keep in "${KEEPS[@]}"; do
    gpu_acquire "pfmatrix-$label-$keep" 2>/dev/null
    run_cell "$label" "$dpath" "$keep" "$FIXTURE" "multi64k"
    gpu_release 2>/dev/null
  done
done

# ── Single-needle sanity at keep=0.15 for each drafter (cheap regression). ──
for d in "${DRAFTERS[@]}"; do
  label="${d%%|*}"; dpath="${d##*|}"
  gpu_acquire "pfmatrix-$label-single" 2>/dev/null
  run_cell "$label" "$dpath" "0.15" "$SINGLE" "single32k"
  gpu_release 2>/dev/null
done

echo "" | tee -a "$SUMMARY"
echo "=== MATRIX COMPLETE $(date +%H:%M:%S) ===" | tee -a "$SUMMARY"
echo "TSV: $TSV" | tee -a "$SUMMARY"
column -t -s$'\t' "$TSV" | tee -a "$SUMMARY"
