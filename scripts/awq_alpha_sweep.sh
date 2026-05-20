#!/usr/bin/env bash
# awq_alpha_sweep.sh — AWQ alpha grid-search for Qwen3.5-9B MQ4 on gfx906.
#
# Per docs/plans/qwen35-mq4-quality-gap.md §F1. For each alpha:
#   1. Delete prior AWQ quant (space-bounded — single slot)
#   2. Quantize 9B BF16 safetensors → MQ4G256 + AWQ pre-scaling at this alpha
#   3. Run eval_hipfire against the BF16 kldref, capture KLD .kldseq
#   4. Aggregate per-variant KLD/PPL into a CSV-style log
#
# Per-variant cost on gfx906 (estimated):
#   quantize: ~10-15 min (CPU-bound rayon)
#   eval (n=50 chunks): ~8-12 min  (extrapolated from n=256 ~1.4h)
#   total: ~25 min/alpha
# Full 7-alpha sweep: ~3 hours wall.
#
# Usage:
#   scripts/awq_alpha_sweep.sh <alpha> [<alpha> ...]
# Examples:
#   scripts/awq_alpha_sweep.sh 0.5                          # smoke
#   scripts/awq_alpha_sweep.sh 0.0 0.25 0.4 0.5 0.6 0.75 1.0  # full sweep

set -euo pipefail
cd "$(dirname "$0")/.."

# ── paths ─────────────────────────────────────────────────────────────
BF16_DIR=/local/hipfire/Qwen3.5-9B-BF16-st
IMATRIX=benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.imatrix.gguf
KLDREF=/data/hipfire/qwen3.5-9b-bf16.kldref.bin
QUANT_BIN=target/release/hipfire-quantize
EVAL_BIN=target/release/examples/eval_hipfire

QUANT_SLOT=/local/hipfire/qwen3.5-9b.mq4-awq-current

# ── sweep params ──────────────────────────────────────────────────────
MAX_CHUNKS=${MAX_CHUNKS:-50}
KV_MODE=${KV_MODE:-asym3}
SCORING=${SCORING:-prefill}
RESULTS_LABEL=${RESULTS_LABEL:-2026-05-14-awq-alpha-sweep-9b-gfx906}
RESULTS_DIR=benchmarks/quality-baselines/results/$RESULTS_LABEL
mkdir -p "$RESULTS_DIR/per-variant"
SUMMARY="$RESULTS_DIR/summary.tsv"
if [ ! -f "$SUMMARY" ]; then
    printf "alpha\tquantize_sec\teval_sec\tkldseq_path\n" > "$SUMMARY"
fi

# ── pre-flight ────────────────────────────────────────────────────────
for f in "$BF16_DIR" "$IMATRIX" "$KLDREF" "$QUANT_BIN" "$EVAL_BIN"; do
    if [ ! -e "$f" ]; then
        echo "FATAL: missing $f" >&2
        exit 2
    fi
done

if [ $# -eq 0 ]; then
    echo "usage: $0 <alpha> [<alpha> ...]" >&2
    exit 2
fi

# ── per-alpha loop ────────────────────────────────────────────────────
for ALPHA in "$@"; do
    TAG=$(echo "$ALPHA" | tr '.' '_')
    LOG_DIR="$RESULTS_DIR/per-variant/a${TAG}"
    mkdir -p "$LOG_DIR"
    QUANT_LOG="$LOG_DIR/quantize.log"
    EVAL_LOG="$LOG_DIR/eval.log"
    KLDSEQ="$LOG_DIR/awq-a${TAG}.kldseq"

    echo "==== alpha=$ALPHA (tag=$TAG) ===="

    # 1. Delete prior slot
    if [ -e "$QUANT_SLOT" ]; then
        echo "  rm prior: $QUANT_SLOT"
        rm -f "$QUANT_SLOT"
    fi

    # 2. Quantize
    echo "  quantize → $QUANT_SLOT  (alpha=$ALPHA)"
    QSTART=$SECONDS
    "$QUANT_BIN" \
        --input "$BF16_DIR" \
        --output "$QUANT_SLOT" \
        --format mq4g256 \
        --imatrix "$IMATRIX" \
        --awq-alpha "$ALPHA" \
        > "$QUANT_LOG" 2>&1 \
    || { echo "  QUANTIZE FAILED — see $QUANT_LOG" >&2; tail -30 "$QUANT_LOG" >&2; exit 1; }
    QSEC=$((SECONDS - QSTART))
    echo "  quantize done in ${QSEC}s; size=$(du -h "$QUANT_SLOT" | cut -f1)"

    # 2b. Self-check: confirm AWQ + L5c imatrix + Q8 conv1d all fired.
    # If the quantize binary is stale (cargo missed a rebuild) it will
    # silently produce a plain MQ4 quant without any of these features.
    # Catching this here prevents wasted eval time on broken quants.
    if ! grep -q "^AWQ pre-scaling: ENABLED" "$QUANT_LOG"; then
        echo "  FATAL: 'AWQ pre-scaling: ENABLED' not in quantize log — rebuild quantize binary." >&2
        head -10 "$QUANT_LOG" >&2
        exit 1
    fi
    if ! grep -q "^L5c activation-weighted LS: ENABLED" "$QUANT_LOG"; then
        echo "  FATAL: 'L5c activation-weighted LS: ENABLED' not in quantize log — imatrix did not load." >&2
        head -10 "$QUANT_LOG" >&2
        exit 1
    fi
    if ! grep -q "Q8_F16: .*conv1d.weight" "$QUANT_LOG"; then
        echo "  FATAL: conv1d weights are not Q8 — quantizer default missing or override broken." >&2
        grep -i conv1d "$QUANT_LOG" | head -3 >&2
        exit 1
    fi
    AWQ_SIDECAR_COUNT=$(grep -c "^    AWQ:    " "$QUANT_LOG")
    echo "  self-check OK: AWQ sidecars=$AWQ_SIDECAR_COUNT (expect 248 for 9B F2; 184 with HIPFIRE_AWQ_F1_ONLY=1), conv1d=Q8, imatrix loaded"

    # 3. Eval
    echo "  eval (n=$MAX_CHUNKS, kv=$KV_MODE, $SCORING)"
    ESTART=$SECONDS
    "$EVAL_BIN" \
        --model "$QUANT_SLOT" \
        --ref "$KLDREF" \
        --output "$KLDSEQ" \
        --kv-mode "$KV_MODE" \
        --scoring-mode "$SCORING" \
        --max-chunks "$MAX_CHUNKS" \
        > "$EVAL_LOG" 2>&1 \
    || { echo "  EVAL FAILED — see $EVAL_LOG" >&2; tail -30 "$EVAL_LOG" >&2; exit 1; }
    ESEC=$((SECONDS - ESTART))
    echo "  eval done in ${ESEC}s; kldseq=$KLDSEQ"

    # 4. Record
    printf "%s\t%d\t%d\t%s\n" "$ALPHA" "$QSEC" "$ESEC" "$KLDSEQ" >> "$SUMMARY"

    echo ""
done

echo "==== SWEEP DONE ===="
echo "summary: $SUMMARY"
cat "$SUMMARY"
