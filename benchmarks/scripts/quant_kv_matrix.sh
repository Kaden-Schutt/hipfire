#!/usr/bin/env bash
# quant_kv_matrix.sh — quantize each model in ~/.hipfire/models with each
# supported quant system, then speed-bench every artifact across the KV
# systems its arch supports, using `hipfire eval --battery speed`.
#
# Constraints on this box: ~74GB free disk, 46GB RAM. The deliverable is the
# speed numbers, so each artifact is PRUNED right after its KV benches finish
# to keep disk bounded to ~one artifact at a time. Hessian-requiring formats
# (oq4+/oq4++/oq8+/oq8++) run LAST, after a one-time collect-artifacts per model.
#
# Resumable: a (model,format,kv) job is skipped if its row is already in the CSV.
set -u

REPO=/home/sadara/hipfire
export HIPFIRE_DAEMON_BIN="$REPO/target/release/hipfire-daemon"
QZ="$REPO/target/release/hipfire-quantize"
EVAL="$REPO/target/release/hipfire"
MODELS_DIR=/home/sadara/.hipfire/models
ARTDIR="$MODELS_DIR/matrix"
TS="${MATRIX_TS:-20260626T2120Z}"
OUT="$REPO/benchmarks/results/quant-kv-matrix-$TS"
CSV="$OUT/results.csv"
LOG="$OUT/run.log"
EVDIR="$OUT/eval"
mkdir -p "$ARTDIR" "$OUT" "$EVDIR"

# model | arch | space-separated KV set
MODELS=(
  "LFM2.5-350M.bf16.hfq|lfm2|q8"
  "qwen3.5-0.8b.bf16.hfq|qwen|q8 asym4 asym3 asym2"
  "llama-3.2-1b-instruct.bf16.hfq|llama|q8 fp32"
  "qwen3.5-2b.bf16.hfq|qwen|q8 asym4 asym3 asym2"
  "qwen3.6-35b-a3b.bf16.hfq|qwen|q8 asym4 asym3 asym2"
  "qwen3.5-4b.bf16.hfq|qwen|q8 asym4 asym3 asym2"
  "qwen3.5-9b.bf16.hfq|qwen|q8 asym4 asym3 asym2"
  "qwen3.6-27b.bf16.hfq|qwen|q8 asym4 asym3 asym2"
)
NONHESS=(q8f16 hfq4 hfq6 mq3 mq4 mq6 oq4 oq8)
HESS=(oq4+ oq4++ oq8+ oq8++)

# Optional smoke filters: ONLY_MODELS / ONLY_FMTS / ONLY_KV (space-sep substrings)
if [ -n "${ONLY_FMTS:-}" ]; then read -ra NONHESS <<< "$ONLY_FMTS"; read -ra HESS <<< ""; fi
keep_model(){ [ -z "${ONLY_MODELS:-}" ] && return 0; case " ${ONLY_MODELS} " in *" $1 "*) return 0;; *) for t in $ONLY_MODELS; do [[ "$1" == *"$t"* ]] && return 0; done; return 1;; esac; }
filter_kv(){ [ -z "${ONLY_KV:-}" ] && { echo "$@"; return; }; local o=""; for k in "$@"; do case " $ONLY_KV " in *" $k "*) o="$o $k";; esac; done; echo "$o"; }

MIN_FREE_GB=20   # refuse to quantize if free disk would drop below this

log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ ! -f "$CSV" ]; then
  echo "model,arch,format,kv,tok_s,decode_tok_s,gen_tok_s,prefill_tok_s,prefill_ms,ttft_ms,artifact_bytes,status,reason" > "$CSV"
fi

free_gb(){ df -BG --output=avail "$MODELS_DIR" | tail -1 | tr -dc '0-9'; }
row_done(){ # model fmt kv  -> 0 if already in CSV (exact field match; fmt may contain '+')
  awk -F, -v m="$1" -v f="$2" -v k="$3" \
    'NR>1 && $1==m && $3==f && $4==k{found=1} END{exit !found}' "$CSV"
}

# Bench one artifact across its KV set; harvest the warm (reset) row metrics.
bench_artifact(){
  local model="$1" arch="$2" fmt="$3" art="$4"; shift 4
  local kvs=("$@")
  local bytes; bytes=$(stat -c %s "$art" 2>/dev/null || echo 0)
  local kv
  for kv in "${kvs[@]}"; do
    if row_done "$model" "$fmt" "$kv"; then log "  skip (done): $model $fmt $kv"; continue; fi
    local jobout="$EVDIR/${model%.bf16.hfq}.${fmt}.${kv}"
    log "  bench: $fmt kv=$kv"
    timeout 1200 "$EVAL" eval "$art" --battery speed --kv-mode "$kv" --force --out "$jobout" \
      > "$jobout.stdout" 2> "$jobout.stderr"
    local rj="$jobout/results.jsonl"
    local toks dec gen pre pms ttft status reason
    if [ -f "$rj" ] && grep -q daemon_prefill_decode_reset "$rj"; then
      # one field per line; mapfile preserves empty fields (read collapses tabs).
      local M; mapfile -t M < <(jq -rs '
        (map(select(.case_id=="daemon_prefill_decode_reset"))[0]) as $r |
        ($r.metrics.tok_s//""),($r.metrics.decode_tok_s//""),($r.metrics.gen_tok_s//""),
        ($r.metrics.prefill_tok_s//""),($r.metrics.prefill_ms//""),($r.metrics.ttft_ms//""),
        ($r.status//"")' "$rj")
      toks="${M[0]}"; dec="${M[1]}"; gen="${M[2]}"; pre="${M[3]}"; pms="${M[4]}"; ttft="${M[5]}"; status="${M[6]}"
      reason=$(jq -rs 'map(select(.case_id=="daemon_prefill_decode_reset"))[0].reason//""' "$rj" | tr ',\n' ';;')
    else
      toks=""; dec=""; gen=""; pre=""; pms=""; ttft=""; status="fail"
      reason=$(tail -1 "$jobout.stderr" 2>/dev/null | tr ',\n' ';;' | cut -c1-200)
    fi
    echo "$model,$arch,$fmt,$kv,$toks,$dec,$gen,$pre,$pms,$ttft,$bytes,$status,$reason" >> "$CSV"
    log "    -> status=$status tok_s=$toks decode=$dec prefill_tok_s=$pre"
  done
}

quant_and_bench(){
  local model="$1" arch="$2" fmt="$3" extra="$4"; shift 4
  local kvs=("$@")
  local src="$MODELS_DIR/$model"
  local base="${model%.bf16.hfq}"
  local art="$ARTDIR/$base.$fmt.hfq"

  # all kv rows already present? skip everything.
  local need=0 kv
  for kv in "${kvs[@]}"; do row_done "$model" "$fmt" "$kv" || need=1; done
  if [ "$need" = 0 ]; then log "skip model=$model fmt=$fmt (all kv done)"; return; fi

  if [ ! -f "$art" ]; then
    local fg; fg=$(free_gb)
    if [ "$fg" -lt "$MIN_FREE_GB" ]; then
      log "DISK GUARD: only ${fg}GB free, skipping quantize $model $fmt"
      for kv in "${kvs[@]}"; do row_done "$model" "$fmt" "$kv" || \
        echo "$model,$arch,$fmt,$kv,,,,,,,0,skip,disk_guard_${fg}GB" >> "$CSV"; done
      return
    fi
    log "quantize: $model -> $fmt (free=${fg}GB)"
    if ! timeout 5400 $QZ --input "$src" --output "$art" --format "$fmt" $extra \
         > "$EVDIR/$base.$fmt.quant.log" 2>&1; then
      log "QUANT FAIL: $model $fmt (see $base.$fmt.quant.log)"
      local r; r=$(tail -1 "$EVDIR/$base.$fmt.quant.log" | tr ',\n' ';;' | cut -c1-200)
      for kv in "${kvs[@]}"; do row_done "$model" "$fmt" "$kv" || \
        echo "$model,$arch,$fmt,$kv,,,,,,,0,quant_fail,$r" >> "$CSV"; done
      rm -f "$art"
      return
    fi
  fi
  bench_artifact "$model" "$arch" "$fmt" "$art" "${kvs[@]}"
  log "prune: $art ($(du -h "$art" 2>/dev/null | cut -f1))"
  rm -f "$art"
}

log "=== PHASE 1: non-Hessian formats ==="
for entry in "${MODELS[@]}"; do
  IFS='|' read -r model arch kvset <<< "$entry"
  keep_model "$model" || continue
  read -ra kvs <<< "$(filter_kv $kvset)"
  [ ${#kvs[@]} -eq 0 ] && continue
  log ">>> MODEL $model (arch=$arch kv='$kvset')"
  for fmt in "${NONHESS[@]}"; do
    quant_and_bench "$model" "$arch" "$fmt" "" "${kvs[@]}"
  done
done

log "=== PHASE 2: Hessian formats (collect-artifacts first) ==="
for entry in "${MODELS[@]}"; do
  IFS='|' read -r model arch kvset <<< "$entry"
  keep_model "$model" || continue
  read -ra kvs <<< "$(filter_kv $kvset)"
  [ ${#kvs[@]} -eq 0 ] && continue
  base="${model%.bf16.hfq}"
  calib="$ARTDIR/$base.calib.hfq"

  # need any hess row?
  need=0
  for fmt in "${HESS[@]}"; do for kv in "${kvs[@]}"; do row_done "$model" "$fmt" "$kv" || need=1; done; done
  if [ "$need" = 0 ]; then log "skip Hessian for $model (all done)"; continue; fi

  log ">>> HESSIAN MODEL $model"
  if [ ! -f "$calib" ]; then
    fg=$(free_gb)
    if [ "$fg" -lt "$MIN_FREE_GB" ]; then log "DISK GUARD skip collect-artifacts $model (${fg}GB)"; continue; fi
    log "collect-artifacts: $model -> $base.calib.hfq"
    if ! timeout 7200 "$EVAL" collect-artifacts --model "$MODELS_DIR/$model" \
         --corpus "$REPO/benchmarks/calib/calib-1m.txt" --output "$calib" \
         > "$EVDIR/$base.calib.log" 2>&1; then
      log "COLLECT FAIL: $model (see $base.calib.log)"
      r=$(tail -1 "$EVDIR/$base.calib.log" | tr ',\n' ';;' | cut -c1-200)
      for fmt in "${HESS[@]}"; do for kv in "${kvs[@]}"; do row_done "$model" "$fmt" "$kv" || \
        echo "$model,$arch,$fmt,$kv,,,,,,,0,collect_fail,$r" >> "$CSV"; done; done
      rm -f "$calib"
      continue
    fi
  fi
  for fmt in "${HESS[@]}"; do
    quant_and_bench "$model" "$arch" "$fmt" "--hessian $calib" "${kvs[@]}"
  done
  log "prune calib: $calib"
  rm -f "$calib"
done

log "=== MATRIX COMPLETE ==="
log "rows: $(($(wc -l < "$CSV")-1))  csv=$CSV"
