#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
#
# Diagnostic only: does the parent share production's weak-routed-branch
# problem? Capture parent plogs at HIPFIRE_PARENT_ROUTE_SCALE=1.8 and 2.2,
# report PPL and KLD vs ref_fp8 against the baseline at checkpoint 1.5.
#
# Does NOT change the parent default (must stay at checkpoint 1.5).
# Requires the effective_parent_route_scale() override in parent/moe.rs.
#
# Usage:
#   ds4_parent_route_scale_probe.sh

set -u
export PATH="${PATH}:/root/.cargo/bin:/opt/rocm/core-7.14/bin"

ROOT="${HIPFIRE_ROOT:-/root/hipfire-work/ds4-parent-gate}"
TEACHER="${TEACHER_DIR:-/mnt/scratch/quantization/deepseek-v4-flash-0731-teacher}"
OUT="${PROBE_OUT:-${TEACHER}/gate6/parent_route_probe}"
BASELINE="${BASELINE_DIR:-/mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline}"
TOKENS="${TOKENS:-${BASELINE}/tokens.bin}"
# Prefer post-combfix parent plog if present as the 1.5 baseline reference.
BASE_PLOG_CANDIDATES=(
  "${BASELINE}/combfix/parent_1024_combfix.plog"
  "${BASELINE}/parent_1024.plog"
)
REF_FP8="${TEACHER}/ref_fp8_1024.plog"
MODEL_DIR="${PARENT_MODEL:-/mnt/scratch/models/DeepSeek-V4-Flash-0731}"

BIN_GATE="${ROOT}/target/release/examples/ds4_parent_forward_gate"
BIN_KLD="${ROOT}/target/release/examples/ds4_parent_kld"

mkdir -p "$OUT"
cd "$ROOT" || exit 1

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$OUT/probe.log"; }

BASE_PLOG=""
for c in "${BASE_PLOG_CANDIDATES[@]}"; do
  if [[ -f "$c" ]]; then BASE_PLOG="$c"; break; fi
done
if [[ -z "$BASE_PLOG" ]]; then
  log "FAIL: no baseline parent plog found"
  exit 1
fi
log "baseline parent plog: $BASE_PLOG"
log "bin_gate=$(sha256sum "$BIN_GATE" | awk '{print $1}')"
log "bin_kld=$(sha256sum "$BIN_KLD" | awk '{print $1}')"

capture_parent() {
  local scale="$1"
  local tag="parent_1024_rs$(echo "$scale" | tr . p)"
  local plog="${OUT}/${tag}.plog"
  local runlog="${OUT}/${tag}.log"
  log "START parent capture scale=${scale} tag=${tag}"
  local t0 t1
  t0=$(date +%s)
  env HIPFIRE_PARENT_ROUTE_SCALE="$scale" \
    "$BIN_GATE" \
    --model "$MODEL_DIR" \
    --token-ids "$TOKENS" \
    --tokens 1024 \
    --plog "$plog" \
    --skip-shard-hashes \
    >"$runlog" 2>&1
  local rc=$?
  t1=$(date +%s)
  log "END parent capture tag=${tag} rc=${rc} wall_s=$((t1 - t0))"
  grep -E "HIPFIRE_PARENT_ROUTE_SCALE|route_scale|PPL|coherence|FAIL|PASS" "$runlog" \
    | head -40 | tee -a "$OUT/probe.log" || true
  if [[ ! -f "$plog" ]]; then
    log "FAIL: no plog for ${tag}"
    tail -40 "$runlog" | tee -a "$OUT/probe.log"
    return 1
  fi
  echo "$tag"
}

run_kld() {
  local ref="$1" cand="$2" outtxt="$3" label="$4"
  log "KLD ${label}"
  "$BIN_KLD" --reference "$ref" --candidate "$cand" --tokens "$TOKENS" \
    >"$outtxt" 2>&1 || true
  tee -a "$OUT/probe.log" <"$outtxt" >/dev/null
}

# Baseline KLD at checkpoint 1.5 (existing plog — no re-capture required)
run_kld "$REF_FP8" "$BASE_PLOG" "$OUT/kld_parent_baseline_vs_fp8.txt" "parent_baseline_1.5_vs_fp8"

for s in 1.8 2.2; do
  tag=$(capture_parent "$s") || continue
  run_kld "$REF_FP8" "${OUT}/${tag}.plog" "$OUT/kld_${tag}_vs_fp8.txt" "${tag}_vs_fp8"
done

{
  echo "Parent route_scale diagnostic probe"
  echo "Produced: $(ts)"
  echo "Baseline plog: $BASE_PLOG"
  echo "Default PARENT_ROUTE_SCALE remains 1.5 (checkpoint). Override is env-only."
  echo
  echo "=== baseline (checkpoint 1.5) vs ref_fp8 ==="
  grep -E 'mean:|p50:|p95:|max:|top-1|top-5|PPL ' "$OUT/kld_parent_baseline_vs_fp8.txt" || true
  for s in 1.8 2.2; do
    tag="parent_1024_rs$(echo "$s" | tr . p)"
    echo
    echo "=== scale ${s} vs ref_fp8 ==="
    grep -E 'mean:|p50:|p95:|max:|top-1|top-5|PPL ' "$OUT/kld_${tag}_vs_fp8.txt" 2>/dev/null || echo "(missing)"
  done
  echo
  echo "Interpretation: if elevated scale barely moves PPL/KLD, parent does not share"
  echo "production's weak-routed-branch symptom. If PPL drops sharply, L0/L2 moe_out"
  echo "floor was unrepresentative and the remaining 12.7x has a MoE lead."
} | tee "$OUT/PROBE_SUMMARY.txt" | tee -a "$OUT/probe.log"

log "DONE -> $OUT/PROBE_SUMMARY.txt"
