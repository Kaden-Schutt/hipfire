#!/usr/bin/env bash
# RoughQuant Phase-2 sweep: PCA rotation + protect_frac × bulk_bits on
# Qwen3.5-0.8B. quantize (CPU, roughquant2-sim) -> PPL (GPU) -> delete .hfq.
# Byte-identical corpus; gpu-lock coordinated. Compare to mq4 gate 29.08.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-/srv/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17}
# Hessian is now the unified `.calib.hfq` (HFQM) from the native collector
# (`hipfire collect-artifacts`); the legacy HFHS `.hessian.bin` was retired.
export HIPFIRE_QTIP_HESSIAN="${HESS:-$HOME/.hipfire/calib/qwen3.5-0.8b.calib.hfq}"
CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
Q=./target/release/hipfire-quantize
# Per-config PPL via the `perplexity` example directly (tight sweep loop). For
# one-off / standalone PPL+KLD on a model, prefer the canonical harness path:
#   hipfire-eval --model <m> --battery perplexity [--kldref <ref>]
PPL=./target/release/examples/perplexity
TMP="$HOME/.hipfire/models/_rq2_sweep.hfq"
OUT="${1:-/tmp/roughquant_phase2.tsv}"

# "bulk_bits protect_frac"
CONFIGS=(
  "3 0.0" "3 0.015" "3 0.03" "3 0.06"
  "2 0.0" "2 0.015" "2 0.03" "2 0.06" "2 0.12"
)

source scripts/gpu-lock.sh 2>/dev/null || true

echo -e "bulk_bits\tprotect_frac\tavg_bits_est\tnll_tok\tppl" | tee "$OUT"
for cfg in "${CONFIGS[@]}"; do
  read -r BITS FRAC <<<"$cfg"
  # est avg bits: protected cols ~16 (bf16) + bulk cols at QTIP (~bits+0.13 trellis overhead)
  AVGBITS=$(python3 -c "f=$FRAC;b=$BITS;print(round(f*16+(1-f)*(b+0.13),3))")
  HIPFIRE_RQ2_BULK_BITS=$BITS HIPFIRE_RQ2_PROTECT_FRAC=$FRAC HIPFIRE_RQ2_DAMP=0.01 \
    "$Q" --input "$MODEL" --output "$TMP" --format roughquant2-sim >/tmp/rq2_quant.log 2>&1 || {
      echo "QUANTIZE FAILED bits=$BITS frac=$FRAC; tail:"; tail -20 /tmp/rq2_quant.log; exit 1; }
  gpu_acquire "roughquant2-sweep-b${BITS}-f${FRAC}" 2>/dev/null || true
  RES=$("$PPL" "$TMP" "$CORPUS" --ctx 2048 --warmup 8 --offset 0 2>/dev/null | grep -E "NLL/tok|PPL")
  gpu_release 2>/dev/null || true
  NLL=$(echo "$RES" | grep NLL | awk '{print $2}')
  PPLV=$(echo "$RES" | grep PPL | awk '{print $2}')
  echo -e "${BITS}\t${FRAC}\t${AVGBITS}\t${NLL}\t${PPLV}" | tee -a "$OUT"
  rm -f "$TMP"
done
echo "sweep done -> $OUT"
