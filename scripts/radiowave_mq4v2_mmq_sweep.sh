#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Candidate radiowave sweep for MQ4V2 MMQ residual (gfx1151).
# Covers the single production TU and the candidate K-variant TU.
# Produces 5 scheduler profiles × 4 unroll settings = 20 code objects per TU.
# All gfx1151 execution via HipxMeasure — no NPU traffic during GPU timing.
#
# Usage on hipx (HIP_VISIBLE_DEVICES=1 = gfx1151 8060S):
#   bash scripts/radiowave_mq4v2_mmq_sweep.sh
# Outputs hsaco + manifest + inspection under /tmp/radiowave_mq4v2_mmq/
# Then run oracle correctness + microbench timing via bench_mq4v2_mmq_kvariants.

set -euo pipefail

ARCH="${ARCH:-gfx1151}"
HIPCC="${HIPCC:-/opt/rocm/core/bin/hipcc}"
OUTDIR="${OUTDIR:-/tmp/radiowave_mq4v2_mmq}"
SRC_PROD="kernels/src/gemm_mq4g256v2_residual_mmq.hip"
SRC_CAND="kernels/src/gemm_mq4g256v2_residual_mmq_kvariants.hip"

mkdir -p "$OUTDIR"

PROFILES=(default max-ilp iterative-ilp memory-clause pipeline-ilp)
UNROLLS=(1 2 4 8)

echo "== Radiowave sweep: MQ4V2 MMQ TU(s) arch=$ARCH hipcc=$HIPCC =="

for tu in prod cand; do
  if [[ "$tu" == "prod" ]]; then
    SRC="$SRC_PROD"
    PREFIX="mq4v2_mmq_prod"
  else
    SRC="$SRC_CAND"
    PREFIX="mq4v2_mmq_kvariants"
  fi
  for prof in "${PROFILES[@]}"; do
    for u in "${UNROLLS[@]}"; do
      out="$OUTDIR/${PREFIX}_${prof}_u${u}.hsaco"
      manifest="$OUTDIR/${PREFIX}_${prof}_u${u}.radiowave.json"
      echo "-> compile $tu $prof unroll=$u"
      cargo run -p radiowave -- compile \
        --source "$SRC" \
        --output "$out" \
        --arch "$ARCH" \
        --wave32 \
        --scheduler-profile "$prof" \
        --define "HIPFIRE_SWEEP_UNROLL=$u" \
        --manifest "$manifest"
      # inspect + oracle correctness gate (requires a workgroup probe; use 256-thread block [32,8,1] -> 256)
      cargo run -p radiowave -- inspect --input "$out" --arch "$ARCH" | tee "$out.inspect.json"
    done
  done
done

echo "== Sweep complete: $OUTDIR =="
echo "Next: run oracle compare + campaign ledger ingest, then bench on gfx1151:"
echo "  HIP_VISIBLE_DEVICES=1 cargo run --release -p rdna-compute --example bench_mq4v2_mmq_kvariants_gfx11 2>&1 | tee /tmp/mq4v2_kvariants_run1.jsonl"
echo "Repeat 3x fresh-process for median; see AGENTS.md measurement rules."
echo "Kill bars: <+10% kernel-level (port) and <+5% radiowave => axis DEAD (this wave)."
