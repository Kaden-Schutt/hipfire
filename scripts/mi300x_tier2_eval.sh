#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# Tier 2 — evaluate the AWQ alpha sweep against the WT2 (out-of-distribution)
# KLD reference, then re-evaluate the winner on both references.
#
# Selection happens on WT2 on purpose. The AG reference is the held-out tail of
# the same bartowski_v5 corpus the Hessians were collected from, so picking a
# recipe by AG score rewards calibration overfit — a recipe can look better on AG
# while being flat or worse on genuinely unseen text. WT2 is unrelated to the
# calibration corpus, so it is the honest selector. AG is then reported for the
# winner as the deployment-representative number, and the pair is what makes an
# overfit visible (AG gains with flat WT2 = overfit, not improvement).
#
# Serial by necessity: each eval loads a ~15 GB model, and four concurrent evals
# already OOM'd this card once while calibration held the rest of VRAM.

set -uo pipefail

WORK=/scratch/work
E=./target/release/examples/eval_hipfire
REF_WT2=$WORK/qwen3.8-27b.ref_wt2.bin
REF_AG=$WORK/qwen3.8-27b.ref_ag.bin

cd /scratch/hipfire || exit 1
for p in "$E" "$REF_WT2" "$REF_AG"; do
  [ -e "$p" ] || { echo "MISSING: $p"; exit 1; }
done

run_eval() {           # model_path ref_path tag
  local model="$1" ref="$2" tag="$3"
  local log="$WORK/eval_$tag.log"
  [ -f "$model" ] || { echo "SKIP $tag (no model)"; return 0; }
  if grep -qa "slice-mean KLD" "$log" 2>/dev/null; then
    echo "SKIP $tag (already scored)"; return 0
  fi
  "$E" --model "$model" --ref "$ref" --output "$WORK/eval_$tag.json" \
       --scoring-mode prefill --kv-mode q8 > "$log" 2>&1
  local line
  line=$(grep -ahoE "slice-mean KLD = [0-9.]+ +mean NLL = [0-9.]+ +PPL = [0-9.]+" "$log" | tail -1)
  if [ -n "$line" ]; then echo "  $tag  $line"; else echo "  $tag  FAILED"; tail -3 "$log"; fi
}

echo "=== Tier 2: AWQ alpha sweep vs WT2 (out-of-distribution selector) ==="
echo "oracle WT2: $(grep -ah ORACLE "$WORK/kldref_wt2.log" | tail -1)"
for f in "$WORK"/qwen3.8-27b.awq_a*.mq4; do
  [ -e "$f" ] || continue
  b=$(basename "$f" .mq4); a=${b##*.}          # e.g. awq_a0p55
  run_eval "$f" "$REF_WT2" "${a}_wt2"
done

# Baseline and the default-alpha arm anchor the curve; both are already scored,
# so run_eval skips them and they are reported from their existing logs.
run_eval "$WORK/qwen3.8-27b.base.mq4" "$REF_WT2" "base_wt2"
run_eval "$WORK/qwen3.8-27b.awq.mq4"  "$REF_AG"  "awq_ag"

echo
echo "=== WT2 summary (sorted by KLD, lower is better) ==="
for f in "$WORK"/eval_*_wt2.log; do
  [ -e "$f" ] || continue
  k=$(grep -ahoE "slice-mean KLD = [0-9.]+" "$f" | tail -1 | grep -oE "[0-9.]+")
  p=$(grep -ahoE "PPL = [0-9.]+" "$f" | tail -1 | grep -oE "[0-9.]+")
  [ -n "$k" ] && printf "%12s  KLD=%-12s PPL=%s\n" "$(basename "$f" .log)" "$k" "$p"
done | sort -t= -k2 -g

echo "=== tier2 eval pass complete ==="
