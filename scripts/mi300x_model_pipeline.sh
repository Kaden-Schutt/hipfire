#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# End-to-end Tier-1 pipeline for one dense model:
#   BF16 teacher + calibration (already built) -> 2 KLD references
#   -> 4 MQ4G256 arms -> scored on both references.
#
# Usage: mi300x_model_pipeline.sh <model-stem> [alpha]
#   e.g. mi300x_model_pipeline.sh lfm2.5-350m 0.05
#
# Arms, all at a fixed MQ4G256 wire format — only the recipe changes:
#   base : no calibration at all (control)
#   awq  : imatrix-driven AWQ pre-scaling at <alpha>
#   ldlq : Hessian-driven LDLQ/GPTQ error feedback, no AWQ
#   both : AWQ + LDLQ
# Splitting awq from ldlq is deliberate: the .calib.hfq carries BOTH
# `<w>.imatrix` (feeds AWQ) and `<w>.hessian` (feeds LDLQ), so running them
# separately attributes a KLD change to the right mechanism.
#
# Run smallest model first. GPTQ cost scales badly (measured ~15 tensors/hour on
# qwen3.8-27B, K up to 17408), so the small LFM2 models are the cheap way to
# learn whether LDLQ is worth spending a day of GPU on for the 27B.

set -uo pipefail

STEM="${1:?usage: $0 <model-stem> [alpha]}"
ALPHA="${2:-0.05}"

WORK=/scratch/work
CORPUS_WT2=/scratch/corpus/wikitext2-1024s-2048ctx.txt
CORPUS_AG=/scratch/corpus/bartowski_v5.eval.txt
Q=./target/release/hipfire-quantize
E=./target/release/examples/eval_hipfire
B=./target/release/examples/build_kld_ref_native

TEACHER=$WORK/$STEM.teacher.bf16.hfq
CALIB=$WORK/$STEM.calib.hfq
PARENT=/scratch/parents/$STEM

cd /scratch/hipfire || exit 1
for p in "$TEACHER" "$CALIB" "$PARENT" "$Q" "$E" "$B"; do
  [ -e "$p" ] || { echo "MISSING: $p"; exit 1; }
done

echo "### pipeline $STEM (alpha=$ALPHA)"
echo "calib md5 $(md5sum "$CALIB" | cut -d' ' -f1)"

# ---- references (24 chunks each so WT2 and AG are directly comparable) ----
for pair in "wt2:$CORPUS_WT2" "ag:$CORPUS_AG"; do
  tag=${pair%%:*}; slice=${pair#*:}
  ref=$WORK/$STEM.ref_$tag.bin
  if [ -s "$ref" ]; then echo "SKIP ref_$tag (exists)"; continue; fi
  HIPFIRE_CALIB_BF16=1 HIPFIRE_NORMALIZE_PROMPT=0 \
    "$B" --model "$TEACHER" --slice "$slice" --output "$ref" \
         --top-k 256 --n-ctx 2048 --max-chunks 24 > "$WORK/kldref_${STEM}_$tag.log" 2>&1
  echo "ref_$tag: $(grep -ah ORACLE "$WORK/kldref_${STEM}_$tag.log" | tail -1)"
done

# ---- arms ----
quant() {                       # tag, extra flags...
  local tag="$1"; shift
  local out=$WORK/$STEM.$tag.mq4
  [ -f "$out" ] && { echo "SKIP arm $tag (exists)"; return 0; }
  local t0=$SECONDS
  "$Q" --input "$PARENT" --format mq4 --output "$out" "$@" > "$WORK/arm_${STEM}_$tag.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -f "$out" ]; then
    echo "arm $tag: $((SECONDS-t0))s md5=$(md5sum "$out" | cut -c1-12)"
  else
    echo "arm $tag: FAIL rc=$rc"; tail -3 "$WORK/arm_${STEM}_$tag.log"
  fi
}

quant base
quant awq  --hessian "$CALIB" --awq-alpha "$ALPHA"
quant ldlq --hessian "$CALIB" --ldlq
quant both --hessian "$CALIB" --awq-alpha "$ALPHA" --ldlq

# GPTQ silently degrades to RTN when a Hessian key does not resolve, which is
# indistinguishable from "GPTQ ran and changed little" unless the counter is read.
for tag in ldlq both; do
  g=$(grep -ahoE "GPTQ-MQ4: [0-9]+ tensors GPTQ'd, [0-9]+ fell back" "$WORK/arm_${STEM}_$tag.log" 2>/dev/null | tail -1)
  [ -n "$g" ] && echo "  $tag gptq: $g"
done

# ---- scoring (serial: each eval resides a full model on the card) ----
echo "--- scores (oracle-relative KLD; lower is better) ---"
for tag in base awq ldlq both; do
  m=$WORK/$STEM.$tag.mq4
  [ -f "$m" ] || continue
  for ref in wt2 ag; do
    log=$WORK/eval_${STEM}_${tag}_$ref.log
    if ! grep -qa "slice-mean KLD" "$log" 2>/dev/null; then
      "$E" --model "$m" --ref "$WORK/$STEM.ref_$ref.bin" \
           --output "$WORK/eval_${STEM}_${tag}_$ref.json" \
           --scoring-mode prefill --kv-mode q8 > "$log" 2>&1
    fi
    printf "  %-6s %-4s %s\n" "$tag" "$ref" \
      "$(grep -ahoE 'slice-mean KLD = [0-9.]+ +mean NLL = [0-9.]+ +PPL = [0-9.]+' "$log" | tail -1)"
  done
done
echo "### pipeline $STEM complete"
