#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# F1-scope vs F2-scope AWQ for qwen3.8-27b, quantized and scored on gfx1201.
#
# F1 / F2 is a WHITELIST SCOPE, not an alpha (main.rs:6415 awq_eligible):
#   F1 = input-side projections only  (q/k/v/qkv, gate/up, in_proj_*, router)
#   F2 = F1 + output-side             (o_proj, out_proj, down_proj, w_down)
# F2 is only sound when the runtime has AWQ-aware kernels on the output-side
# path (rotate_x_mq_awq, fused_silu_mul_mq_rotate_awq) to apply the compensating
# divide. Without them `(W*s)*x != W*x`; main.rs:6403-6407 records the measured
# consequence on 0.8B Qwen3.5: KLD 0.6721 -> 13.4893.
#
# Motivation: on gfx1201, F2 at the shipped default alpha=0.55 measured WORSE
# than no calibration at all (0.083322 vs 0.066668 uncalibrated). If F1 lands at
# or below the uncalibrated baseline, the damage is specifically the output-side
# pre-scaling rather than AWQ as such.
#
# Inputs are local to hiptrx: the 18/18-shard parent and the 13.6 MB imatrix
# extract. AWQ reads only <name>.imatrix; the K x K Hessian is --ldlq-only and
# 2308x larger, so it is deliberately not required here.
#
# Disk hygiene: each arm is ~15 GB and is disposable once scored. Set
# KEEP_ARMS=1 to retain them; default removes each arm after its KLD is
# recorded, because the number and the md5 are the durable artifacts.

set -uo pipefail

WORKTREE=${WORKTREE:-/home/kaden/hipfire-quantcal}
PARENT=${PARENT:-/home/kaden/qcal/parents/qwen3.8-27b}
IMATRIX=${IMATRIX:-/home/kaden/qcal/qwen3.8-27b.imatrix.hfq}
REF=${REF:-/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin}
OUTDIR=${OUTDIR:-/home/kaden/.hipfire/models}
RESULTS=${RESULTS:-/home/kaden/f1_results.txt}
GPU=${GPU:-0}
KEEP_ARMS=${KEEP_ARMS:-0}

cd "$WORKTREE" || exit 1
Q=./target/release/hipfire-quantize
E=./target/release/examples/eval_hipfire

for p in "$Q" "$E" "$PARENT" "$IMATRIX" "$REF"; do
  [ -e "$p" ] || { echo "MISSING: $p"; exit 1; }
done

: > "$RESULTS"
echo "imatrix md5 $(md5sum "$IMATRIX" | cut -d' ' -f1)" >> "$RESULTS"
echo "quantize md5 $(md5sum "$Q" | cut -d' ' -f1)" >> "$RESULTS"

for a in 0.05 0.55; do
  tag="f1_a${a/./p}"
  out="$OUTDIR/qwen3.8-27b.$tag.mq4"

  if [ ! -f "$out" ]; then
    t0=$SECONDS
    HIPFIRE_AWQ_F1_ONLY=1 "$Q" --input "$PARENT" --format mq4 --output "$out" \
      --hessian "$IMATRIX" --awq-alpha "$a" --threads 24 \
      > "/home/kaden/$tag.quant.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f "$out" ]; then
      echo "$tag QUANT_FAIL rc=$rc" >> "$RESULTS"
      tail -3 "/home/kaden/$tag.quant.log" >> "$RESULTS"
      continue
    fi
    echo "$tag quantized in $((SECONDS - t0))s md5=$(md5sum "$out" | cut -c1-12)" >> "$RESULTS"
  fi

  HIP_VISIBLE_DEVICES="$GPU" HIPFIRE_KERNEL_CACHE="/home/kaden/kc_$tag" "$E" \
    --model "$out" --ref "$REF" --output "/home/kaden/$tag.json" \
    --scoring-mode prefill --kv-mode q8 > "/home/kaden/$tag.log" 2>&1

  line=$(grep -ahoE 'slice-mean KLD = [0-9.]+ +mean NLL = [0-9.]+ +PPL = [0-9.]+' \
         "/home/kaden/$tag.log" | tail -1)
  echo "$tag ${line:-SCORE_FAIL}" >> "$RESULTS"

  # Reclaim ~15 GB per arm; the KLD line above is the durable result.
  if [ "$KEEP_ARMS" != "1" ] && [ -n "$line" ]; then
    rm -f "$out"
    echo "$tag arm removed (KEEP_ARMS=1 to retain)" >> "$RESULTS"
  fi
done

echo "F1_DONE" >> "$RESULTS"
cat "$RESULTS"
