#!/usr/bin/env bash
# Full vs REAP-pruned ds4 PPL + KLD comparison (same corpus/offset/ctx/warmup).
#   usage: scripts/reap/run_ppl_kld.sh [CTX=1024] [WARMUP=8]
# Requires: target/release/examples/deepseek4_perplexity built, and the keep-map
# sidecar at $KEEPMAP (see build_reap_keepmap.py). Edit MODEL/CORPUS/KEEPMAP for
# your box.
set -euo pipefail
cd "$(dirname "$0")/../.."
CTX="${1:-1024}"; WARMUP="${2:-8}"
MODEL="${MODEL:-/data/hipfire-models/deepseek-v4-flash.mq2lloyd}"
CORPUS="${CORPUS:-benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt}"
KEEPMAP="${KEEPMAP:-/data/hipfire-models/reap_keepmap_162B_k144}"
BIN=./target/release/examples/deepseek4_perplexity
OUT="${OUT:-/data/hipfire-models/reap_ppl_kld_ctx${CTX}}"
mkdir -p "$OUT"
FILT='pre-compiled blob|recompiling'

echo "===== FULL-256 (keep-map OFF) ctx=$CTX ====="
"$BIN" "$MODEL" "$CORPUS" --ctx "$CTX" --warmup "$WARMUP" \
    --dump-logits "$OUT/full.logits" 2>&1 | grep -vE "$FILT" | tee "$OUT/full.log"

echo "===== PRUNED-144 (keep-map ON) ctx=$CTX ====="
HIPFIRE_DEEPSEEK4_REAP_KEEPMAP="$KEEPMAP" "$BIN" "$MODEL" "$CORPUS" --ctx "$CTX" --warmup "$WARMUP" \
    --dump-logits "$OUT/pruned.logits" 2>&1 | grep -vE "$FILT" | tee "$OUT/pruned.log"

echo "===== KLD (full vs pruned) ====="
python3 scripts/reap/kld_compare.py "$OUT/full.logits" "$OUT/pruned.logits" | tee "$OUT/kld.txt"

echo "===== SUMMARY (ctx=$CTX) ====="
fp=$(grep '^PPL:' "$OUT/full.log" | awk '{print $2}')
pp=$(grep '^PPL:' "$OUT/pruned.log" | awk '{print $2}')
echo "full-256 PPL=$fp   pruned-144 PPL=$pp"
# Free the large logit dumps once KLD is computed (keep logs + kld.txt).
rm -f "$OUT/full.logits" "$OUT/pruned.logits"
echo "(logit dumps removed; logs + kld.txt kept in $OUT)"
