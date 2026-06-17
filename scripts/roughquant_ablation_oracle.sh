#!/usr/bin/env bash
# Ablation oracle: TRUE per-channel importance = KLD damage of voiding exactly
# that one residual channel (all others exact bf16). Compares against the diag(H)
# ranking to test whether diag is near-optimal as the production selector and to
# bound the achievable tail gain. Deterministic teacher-forced eval => cross-channel
# ranking is apples-to-apples (no run-to-run noise). Short ctx for speed.
set -uo pipefail
cd /home/sadara/.hipfire/src
MODEL=/srv/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
export HIPFIRE_QTIP_HESSIAN="$HOME/.hipfire/hessians/qwen3.5-0.8b.hessian.bin"
CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
REF=/tmp/bf16.pkld
BIN=./target/release/examples/perplexity
QBIN=./target/release/hipfire-quantize
RANKMAP=/tmp/diag_rank.tsv     # RANK<tab>rank<tab>channel<tab>energy (from DUMP_RANK)
OUT=/tmp/ablation_oracle.tsv
CTX=${CTX:-1024}

source scripts/gpu-lock.sh 2>/dev/null || true

# Sampled ranks: dense at the head, log-ish through the middle, dense at the tail.
RANKS="0 1 2 3 4 5 6 8 10 13 16 20 26 32 42 54 68 87 111 142 181 231 294 375 478 609 776 850 920 970 1000 1010 1015 1018 1020 1021 1022 1023"

echo -e "rank\tchannel\tdiag_energy\tablation_kld" > "$OUT"
gpu_acquire "ablation-oracle" 2>/dev/null || true
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
