#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
#
# MQ2R long-context PPL curve + route_scale sweep at 2048 and max length.
# Fresh process per measurement. Does not change shipped defaults.
#
# Usage:
#   ds4_mq2r_longctx_scan.sh              # full campaign
#   ds4_mq2r_longctx_scan.sh curve        # default-scale curve only
#   ds4_mq2r_longctx_scan.sh sweep        # route_scale sweeps only
#   ds4_mq2r_longctx_scan.sh gen          # long-ctx greedy continuation only
#   ds4_mq2r_longctx_scan.sh analyze      # PPL + pos-scan on existing plogs
set -u

export PATH="${PATH}:/root/.cargo/bin:/opt/rocm/core-7.14/bin"

ROOT="${HIPFIRE_ROOT:-/root/hipfire-work/ds4-parent-gate}"
OUT="${LONGCTX_OUT:-/mnt/scratch/quantization/deepseek-v4-flash-0731-mq2r-longctx}"
MODEL="${MQ2R_MODEL:-/mnt/scratch/quantization/deepseek-v4-flash-0731-mq2r-p3/artifacts/deepseek-v4-flash-0731.mq2r}"
EXPECT_SHA="${MQ2R_SHA:-cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce}"
BIN_PLOG="${ROOT}/target/release/examples/ds4_quant_plog"
BIN_CHAT="${ROOT}/target/release/examples/deepseek4_chat"
BIN_KLD="${ROOT}/target/release/examples/ds4_parent_kld"
PPL_PY="${OUT}/plog_ppl.py"
POS_PY="${POS_SCAN:-/root/plog_fine_scan.py}"
if [[ ! -f "$PPL_PY" && -f "${ROOT}/crates/hipfire-arch-deepseek4/scripts/plog_ppl.py" ]]; then
  PPL_PY="${ROOT}/crates/hipfire-arch-deepseek4/scripts/plog_ppl.py"
fi

LENGTHS=(1024 2048 4096 8192)
SWEEP_SCALES=(1.5 1.8 2.0 2.2 2.4)

mkdir -p "$OUT"
cd "$ROOT" || exit 1

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$OUT/campaign.log"; }

record_binary() {
  local bin="$1"
  local tag="$2"
  if [[ ! -x "$bin" ]]; then
    log "FAIL: binary missing: $bin"
    return 1
  fi
  local sha size
  sha=$(sha256sum "$bin" | awk '{print $1}')
  size=$(stat -c%s "$bin")
  echo "${tag}_bin=${bin}" >>"$OUT/provenance.txt"
  echo "${tag}_sha256=${sha}" >>"$OUT/provenance.txt"
  echo "${tag}_bytes=${size}" >>"$OUT/provenance.txt"
  log "binary ${tag}: sha256=${sha} bytes=${size} path=${bin}"
}

run_plog() {
  # Fresh process. Args: length [route_scale_or_empty] [tag_suffix]
  local n="$1"
  local scale="${2:-}"
  local suffix="${3:-}"
  local tok="${OUT}/tokens_${n}.bin"
  local tag="mq2r_${n}"
  if [[ -n "$suffix" ]]; then
    tag="mq2r_${n}_${suffix}"
  fi
  local plog="${OUT}/${tag}.plog"
  local man="${OUT}/${tag}.manifest.json"
  local runlog="${OUT}/${tag}.log"

  if [[ ! -f "$tok" ]]; then
    log "FAIL: missing tokens ${tok}"
    return 1
  fi

  log "START plog n=${n} scale=${scale:-default} tag=${tag}"
  local t0 rc t1 wall
  t0=$(date +%s)
  # Intentionally not set -e around the binary.
  if [[ -n "$scale" ]]; then
    env "HIPFIRE_DEEPSEEK4_ROUTE_SCALE=${scale}" \
      "$BIN_PLOG" \
      --model "$MODEL" \
      --expect-sha256 "$EXPECT_SHA" \
      --token-ids "$tok" \
      --plog "$plog" \
      --manifest "$man" \
      >"$runlog" 2>&1
    rc=$?
  else
    env -u HIPFIRE_DEEPSEEK4_ROUTE_SCALE \
      "$BIN_PLOG" \
      --model "$MODEL" \
      --expect-sha256 "$EXPECT_SHA" \
      --token-ids "$tok" \
      --plog "$plog" \
      --manifest "$man" \
      >"$runlog" 2>&1
    rc=$?
  fi
  t1=$(date +%s)
  wall=$((t1 - t0))
  log "END plog tag=${tag} rc=${rc} wall_s=${wall}"

  if [[ ! -f "$plog" ]]; then
    log "FAIL: no plog produced for ${tag}; last 40 log lines:"
    tail -40 "$runlog" | tee -a "$OUT/campaign.log"
    return 1
  fi

  local psha
  psha=$(sha256sum "$plog" | awk '{print $1}')
  log "plog ${tag}: sha256=${psha} bytes=$(stat -c%s "$plog")"
  grep -E "route_scale|effective_route|HIPFIRE_DEEPSEEK4_ROUTE_SCALE|deepseek4: route_scale" "$runlog" \
    | tee -a "$OUT/campaign.log" || true
  grep -AE8 "coherence eyeball|greedy continuation|joined:" "$runlog" \
    | tee -a "$OUT/campaign.log" || true
  # Prefer the forward-path log line (config_cache) over the example's estimate.
  if grep -q "deepseek4: route_scale=" "$runlog"; then
    grep "deepseek4: route_scale=" "$runlog" | head -1 | tee -a "$OUT/campaign.log"
  fi
  echo "${tag}_plog_sha256=${psha}" >>"$OUT/provenance.txt"
  echo "${tag}_wall_s=${wall}" >>"$OUT/provenance.txt"
  echo "${tag}_rc=${rc}" >>"$OUT/provenance.txt"
  return 0
}

analyze_one() {
  local n="$1"
  local tag="$2"
  local plog="${OUT}/${tag}.plog"
  local tok="${OUT}/tokens_${n}.bin"
  local out="${OUT}/${tag}_ppl.txt"
  local pos="${OUT}/${tag}_posscan.txt"
  if [[ ! -f "$plog" ]]; then
    log "SKIP analyze ${tag}: no plog"
    return 1
  fi
  log "PPL ${tag}"
  # Self-compare via ds4_parent_kld: ppl_reference == ppl_candidate == solo PPL.
  # Validated against known mq2r_1024 @ scale 1.5 → 14.702954.
  if [[ -x "$BIN_KLD" ]]; then
    "$BIN_KLD" --reference "$plog" --candidate "$plog" --tokens "$tok" >"${out}.raw" 2>&1
    {
      echo "plog:       $plog"
      echo "tokens:     $tok"
      grep -E "^n_tokens:|^vocab:|^PPL reference:|^PPL candidate:|^top-1" "${out}.raw" || true
      # Canonical line for greps in this driver:
      awk '/^PPL candidate:/{printf "PPL:        %s\n", $3}' "${out}.raw"
      awk '/^PPL candidate:/{printf "NLL/tok:    (see raw; ppl=exp(nll))\n"}' "${out}.raw"
    } | tee "$out" | tee -a "$OUT/campaign.log"
  else
    python3 "$PPL_PY" "$plog" "$tok" | tee "$out" | tee -a "$OUT/campaign.log"
  fi
  if [[ -f "$POS_PY" ]]; then
    log "POS-SCAN ${tag}"
    python3 "$POS_PY" "$plog" "$tok" >"$pos" 2>&1
    tee -a "$OUT/campaign.log" <"$pos"
  else
    log "WARN: pos scan script missing at ${POS_PY}"
  fi
}

run_curve() {
  log "=== default-scale PPL curve (shipped mq2r route_scale) ==="
  record_binary "$BIN_PLOG" "ds4_quant_plog"
  record_binary "$BIN_KLD" "ds4_parent_kld"
  : >"$OUT/curve_summary.txt"
  local n
  for n in "${LENGTHS[@]}"; do
    if ! run_plog "$n" "" ""; then
      log "CEILING: length ${n} failed; stopping curve here"
      echo "ceiling_failed_at=${n}" >>"$OUT/curve_summary.txt"
      break
    fi
    analyze_one "$n" "mq2r_${n}"
    local ppl
    ppl=$(awk '/^PPL:/{print $2}' "${OUT}/mq2r_${n}_ppl.txt" | tail -1)
    echo "n=${n} ppl=${ppl} plog_sha=$(awk -F= '/^mq2r_'"${n}"'_plog_sha256=/{print $2}' "$OUT/provenance.txt" | tail -1)" \
      | tee -a "$OUT/curve_summary.txt"
  done
}

run_sweep() {
  log "=== route_scale sweep ==="
  : >"$OUT/sweep_summary.txt"
  local lengths=()
  local n
  for n in 2048 4096 8192; do
    if [[ -f "${OUT}/mq2r_${n}.plog" ]]; then
      lengths+=("$n")
    fi
  done
  if ((${#lengths[@]} == 0)); then
    # Fall back: try 1024 if nothing longer worked (still useful data)
    if [[ -f "${OUT}/mq2r_1024.plog" ]]; then
      lengths=(1024)
    else
      log "FAIL: no default plogs for sweep"
      return 1
    fi
  fi
  local sweep_ns=()
  for n in "${lengths[@]}"; do
    if (( n == 2048 )); then sweep_ns+=(2048); fi
  done
  sweep_ns+=("${lengths[-1]}")
  # dedupe preserving order
  local uniq=() x u seen
  for x in "${sweep_ns[@]}"; do
    seen=0
    for u in "${uniq[@]+"${uniq[@]}"}"; do
      if [[ "$u" == "$x" ]]; then seen=1; break; fi
    done
    if (( seen == 0 )); then uniq+=("$x"); fi
  done
  sweep_ns=("${uniq[@]}")
  log "sweep lengths: ${sweep_ns[*]}"

  local s tag_suffix ppl
  for n in "${sweep_ns[@]}"; do
    for s in "${SWEEP_SCALES[@]}"; do
      tag_suffix="rs$(echo "$s" | tr '.' 'p')"
      if ! run_plog "$n" "$s" "$tag_suffix"; then
        log "WARN: sweep n=${n} scale=${s} failed"
        echo "n=${n} scale=${s} FAIL" | tee -a "$OUT/sweep_summary.txt"
        continue
      fi
      analyze_one "$n" "mq2r_${n}_${tag_suffix}"
      ppl=$(awk '/^PPL:/{print $2}' "${OUT}/mq2r_${n}_${tag_suffix}_ppl.txt" | tail -1)
      echo "n=${n} scale=${s} ppl=${ppl}" | tee -a "$OUT/sweep_summary.txt"
    done
  done
}

run_gen() {
  log "=== greedy continuation (chat binary, shipped route_scale) ==="
  local n=""
  local L
  for L in 8192 4096 2048 1024; do
    if [[ -f "${OUT}/mq2r_${L}.plog" ]]; then n=$L; break; fi
  done
  if [[ -z "$n" ]]; then
    log "FAIL: no plog to size the gen run"
    return 1
  fi
  log "longest working plog length marker n=${n}"
  record_binary "$BIN_CHAT" "deepseek4_chat"
  local genlog="${OUT}/gen_longctx.log"
  local t0 t1 rc
  t0=$(date +%s)
  env -u HIPFIRE_DEEPSEEK4_ROUTE_SCALE \
    HIPFIRE_DEEPSEEK4_MODEL="$MODEL" \
    HIPFIRE_DEEPSEEK4_CHAT_RAW=1 \
    HIPFIRE_DEEPSEEK4_TEMP=0 \
    HIPFIRE_DEEPSEEK4_TOP_K=0 \
    HIPFIRE_DEEPSEEK4_GEN_TOKENS=64 \
    "$BIN_CHAT" \
    >"$genlog" 2>&1 <<'EOF'
The capital of France is
EOF
  rc=$?
  t1=$(date +%s)
  log "gen rc=${rc} wall_s=$((t1 - t0))"
  tee -a "$OUT/campaign.log" <"$genlog"
}

cmd="${1:-all}"
echo "campaign_start=$(ts)" >"$OUT/provenance.txt"
echo "model=${MODEL}" >>"$OUT/provenance.txt"
echo "expect_sha=${EXPECT_SHA}" >>"$OUT/provenance.txt"
echo "host=$(hostname)" >>"$OUT/provenance.txt"
model_sha=$(sha256sum "$MODEL" | awk '{print $1}')
echo "model_sha256_verified=${model_sha}" >>"$OUT/provenance.txt"
if [[ "$model_sha" != "$EXPECT_SHA" ]]; then
  log "FAIL: model sha mismatch got=${model_sha} want=${EXPECT_SHA}"
  exit 1
fi
log "model sha OK ${model_sha}"

for n in "${LENGTHS[@]}"; do
  if [[ -f "${OUT}/tokens_${n}.bin" ]]; then
    echo "tokens_${n}_sha256=$(sha256sum "${OUT}/tokens_${n}.bin" | awk '{print $1}')" >>"$OUT/provenance.txt"
  fi
done
if [[ -f "${OUT}/wikitext2-1024s-2048ctx.txt" ]]; then
  echo "corpus_md5=$(md5sum "${OUT}/wikitext2-1024s-2048ctx.txt" | awk '{print $1}')" >>"$OUT/provenance.txt"
fi

WALL0=$(date +%s)
case "$cmd" in
  curve) run_curve ;;
  sweep) run_sweep ;;
  gen) run_gen ;;
  analyze)
    for n in "${LENGTHS[@]}"; do
      [[ -f "${OUT}/mq2r_${n}.plog" ]] && analyze_one "$n" "mq2r_${n}"
    done
    ;;
  all)
    run_curve
    run_sweep
    run_gen
    ;;
  *)
    echo "unknown cmd: $cmd" >&2
    exit 2
    ;;
esac
WALL1=$(date +%s)
log "campaign_cmd=${cmd} total_wall_s=$((WALL1 - WALL0))"
echo "campaign_end=$(ts)" >>"$OUT/provenance.txt"
echo "total_wall_s=$((WALL1 - WALL0))" >>"$OUT/provenance.txt"
