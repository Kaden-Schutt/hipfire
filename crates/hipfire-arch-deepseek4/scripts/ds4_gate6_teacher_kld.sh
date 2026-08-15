#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
#
# Gate 6 teacher-baseline KLD campaign.
# Captures MQ2R and MQ2-Lloyd plogs at shipped route_scale defaults against
# tokens.bin, KLD vs teacher ref_fp8 / ref_exact, route_scale sweeps vs teacher
# KLD, and a side-by-side position scan. Writes BASELINE_SUMMARY.txt into the
# teacher directory.
#
# Fresh process per measurement. Does not change shipped defaults.
# effective_route_scale is always read from the run log, never assumed.
#
# Usage:
#   ds4_gate6_teacher_kld.sh              # full campaign
#   ds4_gate6_teacher_kld.sh floor        # teacher self-consistency only (CPU)
#   ds4_gate6_teacher_kld.sh capture      # default-scale quant captures only
#   ds4_gate6_teacher_kld.sh kld          # KLD table on existing plogs (CPU)
#   ds4_gate6_teacher_kld.sh sweep        # route_scale sweeps only
#   ds4_gate6_teacher_kld.sh scan         # position scan only (CPU)
#   ds4_gate6_teacher_kld.sh summary      # rewrite BASELINE_SUMMARY.txt

set -u
export PATH="${PATH}:/root/.cargo/bin:/opt/rocm/core-7.14/bin"

ROOT="${HIPFIRE_ROOT:-/root/hipfire-work/ds4-parent-gate}"
TEACHER="${TEACHER_DIR:-/mnt/scratch/quantization/deepseek-v4-flash-0731-teacher}"
OUT="${GATE6_OUT:-${TEACHER}/gate6}"
BASELINE="${BASELINE_DIR:-/mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline}"
TOKENS="${TOKENS:-${BASELINE}/tokens.bin}"
TOKENS_SHA="${TOKENS_SHA:-48b0f834a7a60656db79b3784c96e1ed131c501ae5b0d35c2066f45fb5bb86dd}"

MQ2R_MODEL="${MQ2R_MODEL:-/mnt/scratch/quantization/deepseek-v4-flash-0731-mq2r-p3/artifacts/deepseek-v4-flash-0731.mq2r}"
MQ2R_SHA="${MQ2R_SHA:-cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce}"
LLOYD_MODEL="${LLOYD_MODEL:-/mnt/scratch/quantization/deepseek-v4-flash-0731-mq2lloyd/devices/deepseek-v4-flash-0731.mq2lloyd}"
LLOYD_SHA="${LLOYD_SHA:-a6195336fc5505e5612678d16dde68ec66cc0b4011e0526d491f774613f9e075}"

REF_FP8="${TEACHER}/ref_fp8_1024.plog"
REF_EXACT="${TEACHER}/ref_exact_1024.plog"
REF_FP8_SHA="${REF_FP8_SHA:-deb6f4b4778a05f8cd22736df05d1125b92543efbb346483dd23dd321e402eef}"
REF_EXACT_SHA="${REF_EXACT_SHA:-1fc79a569a13a98e6cc53b029ff0bb84bd9ae339c033ee67c35bb1a6b0b37749}"

BIN_PLOG="${ROOT}/target/release/examples/ds4_quant_plog"
BIN_KLD="${ROOT}/target/release/examples/ds4_parent_kld"
FINE_SCAN="${FINE_SCAN:-${ROOT}/crates/hipfire-arch-deepseek4/scripts/plog_fine_scan.py}"
if [[ ! -f "$FINE_SCAN" ]]; then
  FINE_SCAN="/root/plog_fine_scan.py"
fi

MQ2R_SWEEP=(1.5 1.8 2.0 2.2)
LLOYD_SWEEP=(1.5 1.8 2.0 2.2 2.4)

mkdir -p "$OUT"
cd "$ROOT" || exit 1

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$OUT/campaign.log"; }

require_file() {
  local p="$1"
  [[ -f "$p" ]] || { log "FAIL: missing $p"; return 1; }
}

sha_of() { sha256sum "$1" | awk '{print $1}'; }

verify_sha() {
  local path="$1" want="$2" label="$3"
  require_file "$path" || return 1
  local got
  got=$(sha_of "$path")
  if [[ "$got" != "$want" ]]; then
    log "FAIL: ${label} sha256 mismatch got=${got} want=${want} path=${path}"
    return 1
  fi
  log "sha256 OK ${label}: ${got}"
}

read_effective_route() {
  local runlog="$1"
  local v
  v=$(grep -E '^effective_route_scale:' "$runlog" | tail -1 | awk '{print $2}')
  if [[ -z "$v" ]]; then
    v=$(grep -E 'route_scale: effective=' "$runlog" | tail -1 | sed -E 's/.*effective=([0-9.]+).*/\1/')
  fi
  if [[ -z "$v" ]]; then
    v=$(grep -E 'deepseek4: route_scale=' "$runlog" | tail -1 | sed -E 's/.*route_scale=([0-9.]+).*/\1/')
  fi
  printf '%s' "$v"
}

expect_route() {
  local got="$1" want="$2" tag="$3"
  if [[ -z "$got" ]]; then
    log "FAIL: ${tag} missing effective_route_scale in log"
    return 1
  fi
  local ok
  ok=$(awk -v g="$got" -v w="$want" 'BEGIN{ d=(g>w?g-w:w-g); print (d<1e-4)?1:0 }')
  if [[ "$ok" != "1" ]]; then
    log "FAIL: ${tag} effective_route_scale=${got} expected ${want} — artifact gate not firing as intended; STOP"
    return 1
  fi
  log "${tag} effective_route_scale=${got} (matches expected ${want})"
}

run_kld() {
  local ref="$1" cand="$2" outtxt="$3" label="$4"
  require_file "$ref" || return 1
  require_file "$cand" || return 1
  require_file "$TOKENS" || return 1
  log "KLD ${label}: ref=$(basename "$ref") cand=$(basename "$cand")"
  "$BIN_KLD" --reference "$ref" --candidate "$cand" --tokens "$TOKENS" \
    >"$outtxt" 2>&1 || true
  tee -a "$OUT/campaign.log" <"$outtxt" >/dev/null
  log "KLD ${label} -> ${outtxt}"
  return 0
}

scale_tag() {
  awk -v s="$1" 'BEGIN{
    n=split(s,a,".");
    if(n==1) printf "%sp0", a[1];
    else printf "%sp%s", a[1], a[2];
  }'
}

capture_plog() {
  local model="$1" expect="$2" tag="$3"
  local scale="${4:-}"
  local plog="${OUT}/${tag}.plog"
  local man="${OUT}/${tag}.manifest.json"
  local runlog="${OUT}/${tag}.log"
  # One verified digest per model path per campaign. Subsequent loads of the
  # same path+expect skip the 82 GB re-hash via --trust-sha256.
  local verified_key trust_flag=""
  verified_key=$(printf '%s\t%s' "$model" "$expect")
  mkdir -p "$OUT"
  touch "$OUT/verified_models.tsv"
  if grep -Fxq "$verified_key" "$OUT/verified_models.tsv" 2>/dev/null; then
    trust_flag="--trust-sha256"
    log "sha256 already verified for this model earlier in campaign; --trust-sha256"
  fi
  log "START capture tag=${tag} scale=${scale:-default} trust=${trust_flag:-hash}"
  local t0 t1 wall rc
  t0=$(date +%s)
  if [[ -n "$scale" ]]; then
    env -u HIPFIRE_DEEPSEEK4_ROUTE_SCALE \
      "$BIN_PLOG" \
      --model "$model" \
      --expect-sha256 "$expect" \
      --token-ids "$TOKENS" \
      --plog "$plog" \
      --manifest "$man" \
      --route-scale "$scale" \
      $trust_flag \
      >"$runlog" 2>&1
    rc=$?
  else
    env -u HIPFIRE_DEEPSEEK4_ROUTE_SCALE \
      "$BIN_PLOG" \
      --model "$model" \
      --expect-sha256 "$expect" \
      --token-ids "$TOKENS" \
      --plog "$plog" \
      --manifest "$man" \
      $trust_flag \
      >"$runlog" 2>&1
    rc=$?
  fi
  t1=$(date +%s)
  wall=$((t1 - t0))
  log "END capture tag=${tag} rc=${rc} wall_s=${wall}"
  if [[ ! -f "$plog" ]]; then
    log "FAIL: no plog for ${tag}; last 40 log lines:"
    tail -40 "$runlog" | tee -a "$OUT/campaign.log"
    return 1
  fi
  # Record verified digest after a successful hash path (not trust-only).
  if [[ -z "$trust_flag" ]] && grep -q "sha256: OK" "$runlog"; then
    if ! grep -Fxq "$verified_key" "$OUT/verified_models.tsv" 2>/dev/null; then
      printf '%s\n' "$verified_key" >>"$OUT/verified_models.tsv"
      log "recorded verified model sha256=${expect} path=${model}"
    fi
  fi
  local psha ers
  psha=$(sha_of "$plog")
  ers=$(read_effective_route "$runlog")
  log "plog ${tag}: sha256=${psha} bytes=$(stat -c%s "$plog") effective_route_scale=${ers}"
  grep -E "route_scale|effective_route|HIPFIRE_DEEPSEEK4_ROUTE_SCALE|deepseek4: route_scale|coherence|top-5|PPL|ppl|FAIL|trust-sha256|sha256" "$runlog" \
    | tee -a "$OUT/campaign.log" || true
  printf '%s\t%s\t%s\t%s\t%s\n' "$tag" "${ers:-NA}" "$psha" "$wall" "$rc" >>"$OUT/captures.tsv"
  if grep -q "coherence" "$runlog"; then
    grep -A20 "=== coherence" "$runlog" | head -25 | tee -a "$OUT/campaign.log" || true
  fi
  return "$rc"
}

step_floor() {
  log "=== Step 1: teacher self-consistency floor ==="
  verify_sha "$REF_FP8" "$REF_FP8_SHA" "ref_fp8_1024.plog" || return 1
  verify_sha "$REF_EXACT" "$REF_EXACT_SHA" "ref_exact_1024.plog" || return 1
  verify_sha "$TOKENS" "$TOKENS_SHA" "tokens.bin" || return 1
  run_kld "$REF_FP8" "$REF_EXACT" "$OUT/teacher_self_kld.txt" "teacher_fp8_vs_exact"
}

step_capture_defaults() {
  log "=== Step 2: capture quants at shipped defaults ==="
  require_file "$BIN_PLOG" || return 1
  echo -e "tag\teffective_route_scale\tplog_sha256\twall_s\trc" >"$OUT/captures.tsv"

  capture_plog "$MQ2R_MODEL" "$MQ2R_SHA" "mq2r_1024_default" || return 1
  local ers
  ers=$(read_effective_route "$OUT/mq2r_1024_default.log")
  expect_route "$ers" "1.8" "mq2r_default" || return 1

  capture_plog "$LLOYD_MODEL" "$LLOYD_SHA" "lloyd_1024_default" || return 1
  ers=$(read_effective_route "$OUT/lloyd_1024_default.log")
  expect_route "$ers" "2.2" "lloyd_default" || return 1
}

step_kld_table() {
  log "=== Step 3: Gate 6 KLD table ==="
  require_file "$BIN_KLD" || return 1
  require_file "$OUT/mq2r_1024_default.plog" || return 1
  require_file "$OUT/lloyd_1024_default.plog" || return 1

  run_kld "$REF_FP8" "$OUT/mq2r_1024_default.plog"  "$OUT/kld_mq2r_vs_fp8.txt"   "mq2r_vs_fp8"
  run_kld "$REF_FP8" "$OUT/lloyd_1024_default.plog" "$OUT/kld_lloyd_vs_fp8.txt"  "lloyd_vs_fp8"
  run_kld "$REF_EXACT" "$OUT/mq2r_1024_default.plog"  "$OUT/kld_mq2r_vs_exact.txt"  "mq2r_vs_exact"
  run_kld "$REF_EXACT" "$OUT/lloyd_1024_default.plog" "$OUT/kld_lloyd_vs_exact.txt" "lloyd_vs_exact"
}

step_scan() {
  log "=== Step 3b: per-position fine scan (width=64) ==="
  require_file "$FINE_SCAN" || return 1
  python3 "$FINE_SCAN" \
    "$REF_FP8" \
    "$REF_EXACT" \
    "$OUT/mq2r_1024_default.plog" \
    "$OUT/lloyd_1024_default.plog" \
    "$TOKENS" \
    --width 64 \
    >"$OUT/pos_scan_width64.txt" 2>&1 || true
  log "fine_scan -> $OUT/pos_scan_width64.txt"
  tee -a "$OUT/campaign.log" <"$OUT/pos_scan_width64.txt" >/dev/null
  return 0
}

record_sweep_row() {
  local build="$1" scale="$2" tag="$3"
  local ers psha kldtxt ppl kldmean top1 wall
  ers=$(read_effective_route "$OUT/${tag}.log")
  expect_route "$ers" "$scale" "$tag" || log "WARN: ${tag} scale mismatch (continuing for evidence)"
  kldtxt="$OUT/kld_${tag}_vs_fp8.txt"
  run_kld "$REF_FP8" "$OUT/${tag}.plog" "$kldtxt" "${tag}_vs_fp8"
  ppl=$(grep -E 'PPL candidate:' "$kldtxt" | awk '{print $3}')
  kldmean=$(grep -E '^  mean:' "$kldtxt" | head -1 | awk '{print $2}')
  if [[ -z "$kldmean" ]]; then
    kldmean=$(grep -E 'mean:' "$kldtxt" | head -1 | awk '{print $2}')
  fi
  top1=$(grep -E 'top-1 agreement:' "$kldtxt" | awk '{print $3}')
  psha=$(sha_of "$OUT/${tag}.plog")
  wall=$(grep -E "END capture tag=${tag} " "$OUT/campaign.log" | tail -1 | sed -E 's/.*wall_s=([0-9]+).*/\1/')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$build" "$scale" "${ers:-NA}" "${ppl:-NA}" "${kldmean:-NA}" "${top1:-NA}" "${wall:-NA}" "$psha" \
    >>"$OUT/sweep_results.tsv"
}

step_sweep() {
  log "=== Step 4: route_scale vs teacher KLD sweeps ==="
  echo -e "build\tscale\teffective_route_scale\tppl\tkld_mean\ttop1\twall_s\tplog_sha256" >"$OUT/sweep_results.tsv"

  local s tag st
  for s in "${MQ2R_SWEEP[@]}"; do
    st=$(scale_tag "$s")
    tag="mq2r_1024_rs${st}"
    if [[ "$s" == "1.8" && -f "$OUT/mq2r_1024_default.plog" ]]; then
      cp -f "$OUT/mq2r_1024_default.plog" "$OUT/${tag}.plog"
      cp -f "$OUT/mq2r_1024_default.log" "$OUT/${tag}.log" 2>/dev/null || true
      cp -f "$OUT/mq2r_1024_default.manifest.json" "$OUT/${tag}.manifest.json" 2>/dev/null || true
      log "reuse default mq2r plog as ${tag}"
    else
      capture_plog "$MQ2R_MODEL" "$MQ2R_SHA" "$tag" "$s" || return 1
    fi
    record_sweep_row "mq2r" "$s" "$tag"
  done

  for s in "${LLOYD_SWEEP[@]}"; do
    st=$(scale_tag "$s")
    tag="lloyd_1024_rs${st}"
    if [[ "$s" == "2.2" && -f "$OUT/lloyd_1024_default.plog" ]]; then
      cp -f "$OUT/lloyd_1024_default.plog" "$OUT/${tag}.plog"
      cp -f "$OUT/lloyd_1024_default.log" "$OUT/${tag}.log" 2>/dev/null || true
      cp -f "$OUT/lloyd_1024_default.manifest.json" "$OUT/${tag}.manifest.json" 2>/dev/null || true
      log "reuse default lloyd plog as ${tag}"
    else
      capture_plog "$LLOYD_MODEL" "$LLOYD_SHA" "$tag" "$s" || return 1
    fi
    record_sweep_row "lloyd" "$s" "$tag"
  done
  log "sweep_results.tsv written"
  cat "$OUT/sweep_results.tsv" | tee -a "$OUT/campaign.log"
}

extract_kld_block() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "(missing $f)"
    return
  fi
  grep -E 'mean:|p50:|p95:|max:|top-1|top-5|PPL ' "$f" || cat "$f"
}

step_summary() {
  log "=== Step 5: write BASELINE_SUMMARY.txt ==="
  local bin_plog_sha bin_kld_sha engine_commit finished
  bin_plog_sha=$(sha_of "$BIN_PLOG" 2>/dev/null || echo NA)
  bin_kld_sha=$(sha_of "$BIN_KLD" 2>/dev/null || echo NA)
  engine_commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo NA)
  finished=$(ts)
  export GATE6_OUT="$OUT"
  {
    cat <<EOF
Gate 6 teacher baseline — CANONICAL KLD reference
=================================================
Produced: ${finished}
Campaign log: ${OUT}/campaign.log
Engine commit (remote checkout pin): ${engine_commit}
Host: $(hostname)
Binary ds4_quant_plog sha256: ${bin_plog_sha}
Binary ds4_parent_kld sha256: ${bin_kld_sha}

THE TEACHER
-----------
DeepSeek V4 reference model.py under crates/hipfire-arch-deepseek4/reference_oracle/,
run verbatim under PyTorch. NOT the Rust parent/* backend (parent scores PPL ~59
and is marked NOT A CALIBRATION REFERENCE at the top of src/parent/mod.rs).

Teacher artifacts (sha256-verified):
  ref_fp8_1024.plog    sha256=${REF_FP8_SHA}
  ref_exact_1024.plog  sha256=${REF_EXACT_SHA}
  tokens.bin           sha256=${TOKENS_SHA}  (from parent-baseline; 1024 ids)

Teacher self-report (from reference_oracle bring-up):
  reference, fp8 shim (the recipe)   PPL 4.693   top-1 0.640
  reference, exact f32               PPL 4.624   top-1 0.649

STEP 1 — TEACHER SELF-CONSISTENCY FLOOR
---------------------------------------
ref_fp8_1024.plog (reference) vs ref_exact_1024.plog (candidate).
This is the floor every later KLD is read against. Any candidate KLD not
comfortably above this is at floor.
EOF
    extract_kld_block "$OUT/teacher_self_kld.txt"
    cat <<EOF

FLOOR: KLD mean ~0.040 is the teacher disagreeing with itself under the only
arithmetic choice available (fp8 recipe vs exact f32). Use it the way the old
quant-vs-quant 0.106 control was used, but honestly this time.

ROUTE_SCALE DEFAULTS (baked into binary; single source effective_route_scale())
------------------------------------------------------------------------------
  .mq2r artifacts default to 1.8 (measured optimum at ctx2048)
  every other DeepSeek V4 artifact defaults to 2.2
  checkpoint routed_scaling_factor 1.5 is NEVER used on this path
  HIPFIRE_DEEPSEEK4_ROUTE_SCALE still overrides; every deviation is logged
  ALWAYS read effective_route_scale from the run log

Quant artifacts (sha256-verified before load):
  MQ2R  ${MQ2R_MODEL}
        sha256=${MQ2R_SHA}
  Lloyd ${LLOYD_MODEL}
        sha256=${LLOYD_SHA}

STEP 2 — DEFAULT CAPTURES (fresh process each; no env force)
------------------------------------------------------------
EOF
    if [[ -f "$OUT/captures.tsv" ]]; then
      cat "$OUT/captures.tsv"
    fi
    echo
    echo "mq2r_1024_default effective_route_scale: $(read_effective_route "$OUT/mq2r_1024_default.log" 2>/dev/null || echo NA)"
    echo "lloyd_1024_default effective_route_scale: $(read_effective_route "$OUT/lloyd_1024_default.log" 2>/dev/null || echo NA)"
    echo
    echo "Coherence eyeball (from capture logs):"
    for tag in mq2r_1024_default lloyd_1024_default; do
      echo "--- ${tag} ---"
      if [[ -f "$OUT/${tag}.log" ]]; then
        grep -A30 "=== coherence" "$OUT/${tag}.log" | head -30 || true
      fi
    done
    cat <<EOF

STEP 3 — GATE 6 TABLE (n=1024, same tokens.bin)
-----------------------------------------------
For reference: Rust parent scored KLD mean 2.718 / top-1 0.482 against teacher.
Quants should be far better.

### MQ2R vs ref_fp8_1024.plog
EOF
    extract_kld_block "$OUT/kld_mq2r_vs_fp8.txt"
    cat <<EOF

### Lloyd vs ref_fp8_1024.plog
EOF
    extract_kld_block "$OUT/kld_lloyd_vs_fp8.txt"
    cat <<EOF

### MQ2R vs ref_exact_1024.plog
EOF
    extract_kld_block "$OUT/kld_mq2r_vs_exact.txt"
    cat <<EOF

### Lloyd vs ref_exact_1024.plog
EOF
    extract_kld_block "$OUT/kld_lloyd_vs_exact.txt"
    cat <<EOF

PER-POSITION SCAN (plog_fine_scan.py --width 64)
------------------------------------------------
Teacher + both quants side by side. Full file: ${OUT}/pos_scan_width64.txt
EOF
    if [[ -f "$OUT/pos_scan_width64.txt" ]]; then
      cat "$OUT/pos_scan_width64.txt"
    fi
    cat <<EOF

STEP 4 — ROUTE_SCALE vs TEACHER KLD (not standalone PPL)
--------------------------------------------------------
Every prior route_scale number optimised each artifact's own perplexity. That
is not the same objective as agreement with the teacher. This sweep reports
both so the two objectives can be compared directly.

Columns: build  scale  effective_route_scale  PPL  KLD_mean  top1  wall_s  plog_sha256
EOF
    if [[ -f "$OUT/sweep_results.tsv" ]]; then
      cat "$OUT/sweep_results.tsv"
    fi
    cat <<EOF

Detailed KLD reports per sweep point live under ${OUT}/kld_*_vs_fp8.txt.

VERDICT — does KLD-optimal match PPL-optimal?
---------------------------------------------
EOF
    python3 - <<'PY'
import csv, os
from pathlib import Path
p = Path(os.environ.get("GATE6_OUT", "/mnt/scratch/quantization/deepseek-v4-flash-0731-teacher/gate6"))
tsv = p / "sweep_results.tsv"
if not tsv.is_file():
    print("(no sweep_results.tsv)")
    raise SystemExit(0)
rows = list(csv.DictReader(tsv.open(), delimiter="\t"))
by = {}
for r in rows:
    by.setdefault(r["build"], []).append(r)

def fnum(x):
    try:
        return float(x)
    except Exception:
        return float("inf")

print()
for build, rs in by.items():
    valid = [r for r in rs if r.get("kld_mean") not in (None, "", "NA")]
    if not valid:
        print(f"{build}: no valid KLD rows")
        continue
    best_kld = min(valid, key=lambda r: fnum(r["kld_mean"]))
    best_ppl = min(valid, key=lambda r: fnum(r["ppl"]))
    print(f"{build}:")
    print(f"  KLD-optimal scale = {best_kld['scale']}  (KLD mean {best_kld['kld_mean']}, PPL {best_kld['ppl']}, top1 {best_kld['top1']})")
    print(f"  PPL-optimal scale = {best_ppl['scale']}  (PPL {best_ppl['ppl']}, KLD mean {best_ppl['kld_mean']}, top1 {best_ppl['top1']})")
    if best_kld["scale"] == best_ppl["scale"]:
        print(f"  VERDICT: KLD-optimal and PPL-optimal AGREE at scale {best_kld['scale']}.")
        print(f"  Ship the agreed value ({best_kld['scale']}).")
    else:
        print(f"  VERDICT: KLD-optimal ({best_kld['scale']}) DIFFERS from PPL-optimal ({best_ppl['scale']}).")
        print("  Teacher-agreement is the better objective for a model whose job is to be distilled from;")
        print("  PPL is what users feel. Prefer KLD-optimal for calibration/distillation work;")
        print(f"  note the PPL tradeoff ({best_kld['ppl']} at KLD-opt vs best PPL {best_ppl['ppl']} at {best_ppl['scale']})")
        print("  before changing the shipped default.")
    print()
PY
    cat <<EOF

ARTIFACT INDEX
--------------
  ${OUT}/teacher_self_kld.txt
  ${OUT}/mq2r_1024_default.plog + .log + .manifest.json
  ${OUT}/lloyd_1024_default.plog + .log + .manifest.json
  ${OUT}/kld_mq2r_vs_fp8.txt  ${OUT}/kld_lloyd_vs_fp8.txt
  ${OUT}/kld_mq2r_vs_exact.txt  ${OUT}/kld_lloyd_vs_exact.txt
  ${OUT}/pos_scan_width64.txt
  ${OUT}/sweep_results.tsv
  ${OUT}/captures.tsv
  ${OUT}/campaign.log

END OF GATE 6 TEACHER BASELINE SUMMARY
EOF
  } >"${TEACHER}/BASELINE_SUMMARY.txt"
  log "wrote ${TEACHER}/BASELINE_SUMMARY.txt ($(wc -c <"${TEACHER}/BASELINE_SUMMARY.txt") bytes)"
  cp -f "${TEACHER}/BASELINE_SUMMARY.txt" "$OUT/BASELINE_SUMMARY.txt"
}

mode="${1:-all}"
log "Gate 6 teacher-KLD campaign mode=${mode} ROOT=${ROOT} OUT=${OUT}"
log "bin_plog=$(sha_of "$BIN_PLOG" 2>/dev/null || echo missing)"
log "bin_kld=$(sha_of "$BIN_KLD" 2>/dev/null || echo missing)"

case "$mode" in
  floor)   step_floor ;;
  capture) step_capture_defaults ;;
  kld)     step_kld_table ;;
  scan)    step_scan ;;
  sweep)   step_sweep ;;
  summary) step_summary ;;
  all)
    T0=$(date +%s)
    step_floor || exit 1
    step_capture_defaults || exit 1
    step_kld_table || exit 1
    step_scan || true
    step_sweep || exit 1
    step_summary || exit 1
    T1=$(date +%s)
    log "ALL DONE wall_s=$((T1 - T0))"
    ;;
  *)
    echo "usage: $0 [all|floor|capture|kld|scan|sweep|summary]" >&2
    exit 2
    ;;
esac
