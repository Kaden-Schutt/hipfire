#!/usr/bin/env bash
# Ablation oracle: TRUE per-channel importance = KLD damage of voiding exactly
# that one residual channel (all others exact bf16). Compares against the diag(H)
# ranking to test whether diag is near-optimal as the production selector and to
# bound the achievable tail gain. Deterministic teacher-forced eval => cross-channel
# ranking is apples-to-apples (no run-to-run noise). Short ctx for speed.
#
# Parameterized so the SAME oracle runs any model (generality check). Override via
# env; defaults reproduce the original 0.8B run. The Hessian is now a unified
# `.calib.hfq` (HFQM) from the native collector — the legacy HFHS `.hessian.bin`
# sidecar was retired. The bf16 KLD ref and the diag rank-map are auto-generated
# from BF16_MODEL + HESS when absent.
#
# Example (9B generality):
#   MODEL=/srv/huggingface/models--Qwen--Qwen3.5-9B/snapshots/<snap> \
#   BF16_MODEL=~/.hipfire/models/qwen3.5-9b-bf16.hfq \
#   HESS=~/.hipfire/calib/qwen3.5-9b.calib.hfq \
#   DMODEL=4096 CTX=512 RANKS="0 1 2 4 8 16 32 64 128 512 2048 4000 4090 4095" \
#   scripts/roughquant_ablation_oracle.sh
set -uo pipefail
cd /home/sadara/.hipfire/src

MODEL=${MODEL:-/srv/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17}
BF16_MODEL=${BF16_MODEL:-$HOME/.hipfire/models/qwen3.5-0.8b-bf16.hfq}
export HIPFIRE_QTIP_HESSIAN="${HESS:-$HOME/.hipfire/calib/qwen3.5-0.8b.calib.hfq}"
CORPUS=${CORPUS:-benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt}
REF=${REF:-/tmp/bf16.pkld}
BIN=./target/release/examples/perplexity
QBIN=./target/release/hipfire-quantize
RANKMAP=${RANKMAP:-/tmp/diag_rank.tsv}   # RANK<tab>rank<tab>channel<tab>energy (DUMP_RANK)
OUT=${OUT:-/tmp/ablation_oracle.tsv}
CTX=${CTX:-1024}
DMODEL=${DMODEL:-1024}

# Default sampled ranks span head→tail of a DMODEL-channel residual: dense at the
# head, log-ish through the middle, dense at the tail. Override RANKS for other dims.
if [ -z "${RANKS:-}" ]; then
  RANKS="0 1 2 3 4 5 6 8 10 13 16 20 26 32 42 54 68 87 111 142 181 231 294 375 478 609 776"
  last=$((DMODEL-1))
  RANKS="$RANKS $((DMODEL*83/100)) $((DMODEL*90/100)) $((DMODEL*95/100)) $((DMODEL-24)) $((DMODEL-14)) $((DMODEL-9)) $((DMODEL-6)) $((DMODEL-4)) $((DMODEL-3)) $((DMODEL-2)) $last"
fi

source scripts/gpu-lock.sh 2>/dev/null || true
gpu_acquire "ablation-oracle" 2>/dev/null || true

# --- bf16 KLD reference (teacher-forced top-K logprobs of the bf16 model) ---
if [ ! -f "$REF" ]; then
  echo "=== generating bf16 KLD ref -> $REF (from $BF16_MODEL) ===" >&2
  "$BIN" "$BF16_MODEL" "$CORPUS" --ctx "$CTX" --warmup 8 --offset 0 --dump-ref "$REF" >/dev/null 2>&1
fi

# --- diag(H) saliency rank-map (cheap: DUMP_RANK computes resid_energy + exits) ---
if [ ! -f "$RANKMAP" ]; then
  echo "=== generating diag rank-map -> $RANKMAP (DUMP_RANK) ===" >&2
  HIPFIRE_RQ4_DUMP_RANK=1 "$QBIN" --input "$MODEL" --output /tmp/_rankdump.hfq \
    --format roughquant4-sim 2>/dev/null | grep '^RANK' > "$RANKMAP"
  rm -f /tmp/_rankdump.hfq
fi

echo -e "rank\tchannel\tdiag_energy\tablation_kld" > "$OUT"
for r in $RANKS; do
  line=$(awk -F'\t' -v r="$r" '$2==r {print; exit}' "$RANKMAP")
  ch=$(echo "$line" | cut -f3)
  en=$(echo "$line" | cut -f4)
  [ -z "$ch" ] && { echo "rank $r: no channel, skip" >&2; continue; }
  HIPFIRE_RQ4_BULK=void HIPFIRE_RQ4_VOID_ONLY="$ch" \
    "$QBIN" --input "$MODEL" --output /tmp/abo.hfq --format roughquant4-sim >/dev/null 2>&1
  K=$("$BIN" /tmp/abo.hfq "$CORPUS" --ctx "$CTX" --warmup 8 --offset 0 --kld-ref "$REF" 2>/dev/null \
        | grep 'KLD/tok' | awk '{print $2}')
  rm -f /tmp/abo.hfq
  echo -e "${r}\t${ch}\t${en}\t${K}" | tee -a "$OUT"
done
gpu_release 2>/dev/null || true
echo "=== done -> $OUT ==="
