#!/usr/bin/env bash
# RoughQuant Phase-1 sweep: protect_frac × bulk_bits on Qwen3.5-0.8B.
# For each config: quantize (CPU, roughquant-sim) -> PPL (GPU) -> delete .hfq.
# Records a TSV row per config. Byte-identical corpus; gpu-lock coordinated.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=/srv/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
export HIPFIRE_QTIP_HESSIAN="$HOME/.hipfire/hessians/qwen3.5-0.8b.hessian.bin"
CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
Q=./target/release/hipfire-quantize
PPL=./target/release/examples/perplexity
TMP="$HOME/.hipfire/models/_rq_sweep.hfq"
OUT="${1:-/tmp/roughquant_phase1.tsv}"

# config list: "bulk_bits protect_frac"  (group fixed at 256)
CONFIGS=(
  "4 0.0" "4 0.015"
  "3 0.0" "3 0.015" "3 0.06"
  "2 0.0" "2 0.005" "2 0.015" "2 0.03" "2 0.06"
)

source scripts/gpu-lock.sh 2>/dev/null || true

echo -e "bulk_bits\tprotect_frac\tavg_bits_est\tnll_tok\tppl" | tee "$OUT"
for cfg in "${CONFIGS[@]}"; do
  read -r BITS FRAC <<<"$cfg"
  # est avg bits: protected cols at 16 (bf16) + bulk cols at BITS, + ~0.06 scale overhead/group
  AVGBITS=$(python3 -c "f=$FRAC;b=$BITS;print(round(f*16+(1-f)*(b+16.0/256),3))")
  HIPFIRE_RQ_BULK_BITS=$BITS HIPFIRE_RQ_PROTECT_FRAC=$FRAC HIPFIRE_RQ_GROUP=256 \
    "$Q" --input "$MODEL" --output "$TMP" --format roughquant-sim >/tmp/rq_quant.log 2>&1 || {
      echo "QUANTIZE FAILED bits=$BITS frac=$FRAC; tail:"; tail -20 /tmp/rq_quant.log; exit 1; }
  gpu_acquire "roughquant-sweep-b${BITS}-f${FRAC}" 2>/dev/null || true
  RES=$("$PPL" "$TMP" "$CORPUS" --ctx 2048 --warmup 8 --offset 0 2>/dev/null | grep -E "NLL/tok|PPL")
  gpu_release 2>/dev/null || true
  NLL=$(echo "$RES" | grep NLL | awk '{print $2}')
  PPLV=$(echo "$RES" | grep PPL | awk '{print $2}')
  echo -e "${BITS}\t${FRAC}\t${AVGBITS}\t${NLL}\t${PPLV}" | tee -a "$OUT"
  rm -f "$TMP"
done
echo "sweep done -> $OUT"
