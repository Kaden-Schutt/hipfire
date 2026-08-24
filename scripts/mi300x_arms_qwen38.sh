#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# Tier-1 MQ4 arm sweep for qwen3.8-27b against the BF16-teacher calibration.
#
# Four arms, all MQ4G256 at a fixed 136 B/group wire format — the format never
# changes, only the recipe that chooses the codes:
#   base  : no calibration at all (the control)
#   awq   : imatrix-driven AWQ pre-scaling
#   ldlq  : Hessian-driven LDLQ/GPTQ error feedback
#   both  : AWQ + LDLQ
# Separating awq from ldlq is the point: `.calib.hfq` carries BOTH `<w>.imatrix`
# (feeds AWQ) and `<w>.hessian` (feeds LDLQ), so running them independently
# attributes any KLD win to the right mechanism instead of to "calibration".
#
# Two lanes, not four: each arm holds the parent plus a slice of the 31.5 GB
# Hessian package resident, and four at once overruns 235 GB of host RAM.

set -uo pipefail

WORK=/scratch/work
PARENT=/scratch/parents/qwen3.8-27b
CALIB=$WORK/qwen3.8-27b.calib.hfq
Q=./target/release/hipfire-quantize

cd /scratch/hipfire || exit 1

for p in "$PARENT" "$CALIB" "$Q"; do
  [ -e "$p" ] || { echo "MISSING: $p"; exit 1; }
done

run_arm() {
  local name="$1"; shift
  local out="$WORK/qwen3.8-27b.$name.mq4"
  local log="$WORK/arm_$name.log"
  if [ -f "$out" ]; then echo "SKIP $name (exists)"; return 0; fi
  echo "START $name -> $out"
  local t0=$SECONDS
  "$Q" --input "$PARENT" --format mq4 --output "$out" "$@" > "$log" 2>&1
  local rc=$?
  local dt=$((SECONDS - t0))
  if [ $rc -eq 0 ] && [ -f "$out" ]; then
    echo "DONE  $name  ${dt}s  $(stat -c%s "$out") bytes  md5=$(md5sum "$out" | cut -c1-12)"
  else
    echo "FAIL  $name  rc=$rc  ${dt}s  (tail below)"
    tail -5 "$log"
  fi
}

# Lane A: control first (no Hessian load, fastest), then the combined arm.
(
  run_arm base
  run_arm both --hessian "$CALIB" --awq --ldlq
) > "$WORK/laneA.log" 2>&1 &
LA=$!

# Lane B: the two single-mechanism arms.
(
  run_arm awq  --hessian "$CALIB" --awq
  run_arm ldlq --hessian "$CALIB" --ldlq
) > "$WORK/laneB.log" 2>&1 &
LB=$!

wait $LA $LB
echo "=== all arms finished ==="
cat "$WORK/laneA.log" "$WORK/laneB.log"
ls -l "$WORK"/qwen3.8-27b.*.mq4 2>/dev/null | awk '{printf "  %s  %.2f GB\n", $9, $5/1e9}'
