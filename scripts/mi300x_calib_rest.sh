#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# BF16-teacher calibration for the remaining dense models, run serially.
#
# Serial by necessity, not choice: a 27B calibration peaks near 90 GB (teacher
# resident + ~31 GB of Hessian accumulators), and two large models at once
# exhausted the 205.8 GB card earlier in this campaign. Smallest first so a
# systematic arch bug surfaces in minutes on a 0.7 GB model instead of an hour
# into glimmer.
#
# Every run is strict BF16 (HIPFIRE_CALIB_BF16=1 -> DType::BF16 ->
# KernelKey::GemmBf16Mfma). A silent fall back to a scalar F32 GEMM or per-token
# GEMV is a bug, not a slow result — calib_sweep WARNs loudly if that happens.

set -uo pipefail

WORK=/scratch/work
CORPUS=/scratch/corpus/bartowski_v5.calib.txt
SWEEP=./target/release/examples/calib_sweep

cd /scratch/hipfire || exit 1

export HIPFIRE_NORMALIZE_PROMPT=0
export HIPFIRE_GRAPH=0
export HIPFIRE_GRAPH_MOE=0
export HIPFIRE_CALIB_BF16=1
# gemma4 gates its batched/WMMA prefill behind these; without them it silently
# drops to per-token GEMV (lowered.rs:42-53). Harmless for the other arches.
export HIPFIRE_BATCHED_PREFILL=1
export HIPFIRE_WMMA_PREFILL=1

for m in lfm2.5-350m lfm2.5-1.2b gemma4-12b muse-glimmer; do
  T="$WORK/$m.teacher.bf16.hfq"
  OUT="$WORK/$m.calib.hfq"
  LOG="$WORK/calib_$m.log"

  [ -f "$T" ]   || { echo "SKIP $m (no teacher)"; continue; }
  [ -f "$OUT" ] && { echo "SKIP $m (calib exists)"; continue; }

  echo "=== START $m ==="
  t0=$SECONDS
  "$SWEEP" --model "$T" --corpus "$CORPUS" --output "$OUT" \
    --max-tokens 262144 --seq-len 2048 --layers-per-pass 64 > "$LOG" 2>&1
  rc=$?
  dt=$((SECONDS - t0))

  if [ $rc -eq 0 ] && [ -f "$OUT" ]; then
    echo "DONE  $m  ${dt}s  $(stat -c%s "$OUT") bytes"
    grep -aE "calib prefix|coverage " "$LOG" | tail -2
  else
    # Do not continue silently past a failure — a missing calibration would
    # otherwise be discovered only when its quantize arms come out identical.
    echo "FAIL  $m  rc=$rc  ${dt}s"
    grep -aE "panicked|FATAL|error|WARN" "$LOG" | tail -4
  fi
done

echo "=== remaining-model calibration finished ==="
ls -l "$WORK"/*.calib.hfq 2>/dev/null | awk '{printf "  %s  %.1f GB\n", $9, $5/1e9}'
