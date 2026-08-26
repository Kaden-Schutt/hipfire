#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# Tier 2 — AWQ alpha sweep for qwen3.8-27b at FIXED MQ4G256 format.
#
# The wire format does not move: MQ4G256, 136 B/group, bit-identical layout.
# The only thing varying is the AWQ smoothing exponent:
#
#     s[j] = (RMS_act[j])^alpha,  geo-mean normalized to 1
#     (main.rs:6303 compute_awq_scales, RMS_act from the calibration in_sum2)
#
# alpha=0 collapses every scale to 1.0 — i.e. plain RTN, which on this model
# produces a 2-token attractor (KLD 12.92, PPL 2.6e6). alpha=0.55 is the shipped
# default and measures KLD 0.0833 / PPL 6.4632 against a 6.2385 oracle. This
# sweep locates the interior optimum and, just as importantly, shows whether the
# curve is flat near 0.55 (default already good) or steep (default is luck).
#
# Runs at reduced --threads on purpose: the Tier-1 GPTQ arms are holding ~18 of
# 20 cores for several hours, and starving them to finish this sweep sooner
# would trade a required Tier-1 deliverable for an optional Tier-2 data point.
# Quantize only — evaluation is GPU work and is queued separately, because the
# card is fully committed to calibration.

set -uo pipefail

WORK=/scratch/work
PARENT=/scratch/parents/qwen3.8-27b
CALIB=$WORK/qwen3.8-27b.calib.hfq
Q=./target/release/hipfire-quantize
THREADS=${THREADS:-3}

cd /scratch/hipfire || exit 1
for p in "$PARENT" "$CALIB" "$Q"; do
  [ -e "$p" ] || { echo "MISSING: $p"; exit 1; }
done

echo "tier2 awq-alpha sweep: parent=$PARENT calib=$CALIB threads=$THREADS"
echo "calib md5: $(md5sum "$CALIB" | cut -d' ' -f1)"
echo "quantize md5: $(md5sum "$Q" | cut -d' ' -f1)"

# 0.55 is the shipped default and is already measured; it is included anyway so
# the sweep is self-contained and reproducible from one script run.
for a in 0.20 0.35 0.45 0.55 0.65 0.75 0.90; do
  tag="a${a/./p}"
  out="$WORK/qwen3.8-27b.awq_$tag.mq4"
  log="$WORK/arm_awq_$tag.log"
  if [ -f "$out" ]; then echo "SKIP alpha=$a (exists)"; continue; fi
  t0=$SECONDS
  "$Q" --input "$PARENT" --format mq4 --output "$out" \
       --hessian "$CALIB" --awq-alpha "$a" --threads "$THREADS" > "$log" 2>&1
  rc=$?
  dt=$((SECONDS - t0))
  if [ $rc -eq 0 ] && [ -f "$out" ]; then
    echo "DONE alpha=$a ${dt}s md5=$(md5sum "$out" | cut -c1-12) bytes=$(stat -c%s "$out")"
  else
    echo "FAIL alpha=$a rc=$rc ${dt}s"; tail -3 "$log"
  fi
done

echo "=== tier2 alpha sweep quantize complete ==="
ls -l "$WORK"/qwen3.8-27b.awq_a*.mq4 2>/dev/null | awk '{printf "  %s  %.2f GB\n", $9, $5/1e9}'
