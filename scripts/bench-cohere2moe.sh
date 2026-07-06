#!/usr/bin/env bash
# Cohere2-MoE (North-Mini-Code-1.0, arch_id 12) prefill/decode throughput matrix.
# Drives the `infer` example over each quant tier x context length, capturing
# prefill tok/s (batched for MQ4/MQ6, per-token for pure-Q8) and decode tok/s.
# Numbers reported in PR #446. gfx1151 (Strix Halo, RDNA3.5), greedy.
#
# Methodology (docs/methodology/perf-benchmarking.md):
#   * Each measured cell runs an in-process DPM warmup (HIPFIRE_DPM_WARMUP_SECS,
#     memset clock pin) before the timed window — infer is not a canonical bench
#     tool, so without it numbers read ~5-10% low at idle clocks.
#   * One throwaway "warmup" run per tier compiles that tier's dtype GEMM +
#     flash-attention kernels into .hipfire_kernels (DPM warmup does NOT compile
#     kernels), so JIT never lands inside a timed window. Lengths are exact
#     multiples of both infer's 256-token driver chunk and forward_batch's ≤64
#     internal GEMM sub-batch, so every prefill GEMM is a full-B shape → JIT is
#     per-tier-dtype, not per-len.
#   * Prompts are deterministic exact-length token files (throughput is
#     content-independent); their md5s are logged for byte-identical re-runs.
#
# Reproduce (REPO is derived from this script's location; point MODELS at the
# directory holding the north-mini-code.{q8,mq6,mq4}.hfq files if not /data):
#   cargo build --release --example infer -p hipfire-arch-cohere2moe
#   MODELS=/path/to/hfq-dir scripts/bench-cohere2moe.sh
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODELS="${MODELS:-/data/hipfire-models}"
INFER="$REPO/target/release/examples/infer"
OUT="${1:-$REPO/benchmarks/cohere2moe-perf}"
DECODE="${DECODE:-128}"
export HIPFIRE_DPM_WARMUP_SECS="${HIPFIRE_DPM_WARMUP_SECS:-10}"
LENS=(512 2048 8192 16384 32768)
WARM_LEN=512
ORDER=(mq4 mq6 q8)
declare -A TIER_FILE=( [mq4]="north-mini-code.mq4.hfq" [mq6]="north-mini-code.mq6.hfq" [q8]="north-mini-code.q8.hfq" )

mkdir -p "$OUT/tokens" "$OUT/logs"
RESULTS="$OUT/results.csv"
PROGRESS="$OUT/progress.log"
echo "tier,ctx,prefill_tok,prefill_s,prefill_tps,prefill_mode,decode_tok,decode_s,decode_tps,status" > "$RESULTS"
: > "$PROGRESS"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$PROGRESS"; }

gen_tokens() { # $1=N  $2=outfile  — deterministic exact-length, no python dep
  awk -v n="$1" 'BEGIN{ printf "["; for(i=0;i<n;i++){ id=100+(i*7919)%40000; printf (i?",%d":"%d"), id } printf "]" }' > "$2"
}
for L in "${LENS[@]}"; do
  f="$OUT/tokens/toks_$L.json"
  [ -s "$f" ] || gen_tokens "$L" "$f"
  log "tokens $L: md5=$(md5sum "$f" | awk '{print $1}')"
done

[ -x "$INFER" ] || { log "FATAL: $INFER missing — build it first"; exit 1; }

# Parse one infer log -> appends a CSV row. $1=tier $2=L $3=logfile $4=rc
record() {
  local tier="$1" L="$2" cell="$3" rc="$4" pline dline p_tok p_s p_mode p_tps d_tok d_s d_tps status
  pline=$(grep -oE 'prefill [0-9]+ tok in [0-9.]+s \[[a-z-]+\]' "$cell" | tail -1)
  dline=$(grep -oE 'decoded [0-9]+ tok in [0-9.]+s \([0-9.]+ tok/s\)' "$cell" | tail -1)
  if [ -n "$pline" ]; then
    p_tok=$(awk '{print $2}' <<<"$pline"); p_s=$(awk '{print $5}' <<<"$pline" | tr -d 's')
    p_mode=$(awk '{print $6}' <<<"$pline" | tr -d '[]')
    p_tps=$(awk -v t="$p_tok" -v s="$p_s" 'BEGIN{ if(s+0>0) printf "%.1f", t/s; else printf "NA" }')
  else p_tok=NA; p_s=NA; p_mode=NA; p_tps=NA; fi
  if [ -n "$dline" ]; then
    d_tok=$(awk '{print $2}' <<<"$dline"); d_s=$(awk '{print $5}' <<<"$dline" | tr -d 's')
    d_tps=$(awk '{print $6}' <<<"$dline" | tr -d '(')
  else d_tok=NA; d_s=NA; d_tps=NA; fi
  status=OK; [ "$rc" -ne 0 ] && status="EXIT$rc"; [ -z "$pline" ] && status="FAIL(rc=$rc)"
  echo "$tier,$L,$p_tok,$p_s,$p_tps,$p_mode,$d_tok,$d_s,$d_tps,$status" >> "$RESULTS"
  log "    $tier @ $L -> prefill ${p_tps} tok/s [$p_mode] | decode ${d_tps} tok/s | $status"
}

log "=== matrix start: tiers=[${ORDER[*]}] lens=[${LENS[*]}] decode=$DECODE warmup=${HIPFIRE_DPM_WARMUP_SECS}s ==="
for tier in "${ORDER[@]}"; do
  model="$MODELS/${TIER_FILE[$tier]}"
  [ -s "$model" ] || { log "SKIP $tier: $model not found"; continue; }

  # JIT-compile this tier's kernels (throwaway, discarded)
  log ">>> warmup $tier @ $WARM_LEN (compile kernels)"
  # No DPM warmup on this throwaway compile pass — it only needs to populate the
  # kernel cache, not measure, so skip the 10s clock-pin.
  HIPFIRE_DPM_WARMUP_SECS=0 timeout 600 "$INFER" --model "$model" --tokens "$OUT/tokens/toks_$WARM_LEN.json" --max 4 \
      >"$OUT/logs/${tier}_warmup.log" 2>&1 || log "    (warmup rc=$? — continuing)"

  for L in "${LENS[@]}"; do
    cell="$OUT/logs/${tier}_${L}.log"
    log ">>> measure $tier @ ctx=$L"
    timeout 1800 "$INFER" --model "$model" --tokens "$OUT/tokens/toks_$L.json" --max "$DECODE" >"$cell" 2>&1
    record "$tier" "$L" "$cell" "$?"
  done
done
log "=== matrix done -> $RESULTS ==="
