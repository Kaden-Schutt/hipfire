#!/usr/bin/env bash
# bench-eval-all.sh — Generate missing models, then hipfire eval (speed+quality)
# over all Qwen3.5 sizes in MODELS_TO_TEST.md, smallest to largest.
#
# Nohup-safe: no stdin required, timestamped report path printed at start.
# Usage: nohup bash scripts/bench-eval-all.sh > /tmp/bench-eval-all.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

TIER=${1:-medium}
REPORT=/tmp/bench-eval-all-$(date +%Y%m%d-%H%M%S).md
MODELS=~/.hipfire/models

log()     { echo "$(date '+%H:%M:%S') $*" | tee -a "$REPORT"; }
section() { echo "" | tee -a "$REPORT"; echo "## $*" | tee -a "$REPORT"; }

log "bench-eval-all start — report: $REPORT"
echo "REPORT: $REPORT"

# ── GPU lock ──────────────────────────────────────────────────────────────────
source scripts/gpu-lock.sh
gpu_acquire "bench-eval-all"
trap 'gpu_release' EXIT

# ── Build ─────────────────────────────────────────────────────────────────────
section "Build"
log "cargo build --release hipfire-quantize hipfire-eval"
cargo build -p hipfire-quantize --release 2>&1 | tail -3 | tee -a "$REPORT"
cargo build -p hipfire-runtime --bin hipfire-eval --release 2>&1 | tail -3 | tee -a "$REPORT"

# ── Phase 1: generate missing 4B models ──────────────────────────────────────
section "Generate missing 4B models"

HF_4B=/srv/huggingface/models--Qwen--Qwen3.5-4B
if [ ! -d "$HF_4B" ]; then
    log "ERROR: NFS path not found: $HF_4B — skipping 4B generation"
else
    SNAP=$(cat "$HF_4B/refs/main" 2>/dev/null || ls -1 "$HF_4B/snapshots/" | head -1 || true)
    if [ -z "$SNAP" ]; then
        log "ERROR: could not determine snapshot hash under $HF_4B/snapshots/ — skipping"
    else
        INPUT_4B="$HF_4B/snapshots/$SNAP"
        log "4B source: $INPUT_4B"

        if [ -f "$MODELS/qwen3.5-4b-bf16.hfq" ]; then
            log "  skip (exists): qwen3.5-4b-bf16.hfq"
        else
            log "  generating qwen3.5-4b-bf16.hfq ..."
            ./target/release/hipfire-quantize \
                --input "$INPUT_4B" \
                --output "$MODELS/qwen3.5-4b-bf16.hfq" \
                --format bf16 2>&1 | tee -a "$REPORT" \
                || log "  qwen3.5-4b-bf16 generation FAILED"
        fi

        if [ -f "$MODELS/qwen3.5-4b-bf16.hfq" ]; then
            if [ -f "$MODELS/qwen3.5-4b-mq4.hfq" ]; then
                log "  skip (exists): qwen3.5-4b-mq4.hfq"
            else
                log "  generating qwen3.5-4b-mq4.hfq ..."
                ./target/release/hipfire-quantize \
                    --input "$MODELS/qwen3.5-4b-bf16.hfq" \
                    --output "$MODELS/qwen3.5-4b-mq4.hfq" \
                    --format mq4 2>&1 | tee -a "$REPORT" \
                    || log "  qwen3.5-4b-mq4 generation FAILED"
            fi

            if [ -f "$MODELS/qwen3.5-4b-mq6.hfq" ]; then
                log "  skip (exists): qwen3.5-4b-mq6.hfq"
            else
                log "  generating qwen3.5-4b-mq6.hfq ..."
                ./target/release/hipfire-quantize \
                    --input "$MODELS/qwen3.5-4b-bf16.hfq" \
                    --output "$MODELS/qwen3.5-4b-mq6.hfq" \
                    --format mq6 2>&1 | tee -a "$REPORT" \
                    || log "  qwen3.5-4b-mq6 generation FAILED"
            fi
        else
            log "  skipping mq4/mq6 — bf16 source not available"
        fi
    fi
fi

# ── Phase 2: hipfire eval sweep ───────────────────────────────────────────────
# Order: smallest to largest. Models that don't exist are silently skipped.
# 27b-bf16 is omitted (~42GB GTT, noted in MODELS_TO_TEST.md).

KLDREFS=~/.hipfire/kldrefs

eval_model() {
    local label="$1" model="$2"
    if [ ! -f "$model" ]; then
        log "  skip (missing): $label"
        return
    fi
    # Derive kldref from model stem: strip quant suffix, replace with bf16
    local stem
    stem=$(basename "$model" .hfq | sed 's/-mq[0-9]*$/-bf16/ ; s/-bf16$/-bf16/')
    local kldref="$KLDREFS/${stem}.kldref.hfq"
    local kldref_arg=""
    if [ -f "$kldref" ]; then
        kldref_arg="--kldref $kldref"
        log "  kldref: $kldref"
    else
        log "  kldref not found: $kldref (quality will skip)"
    fi
    section "Eval — $label"
    log "model: $model"
    LD_LIBRARY_PATH=/opt/rocm/lib ROCM_PATH=/opt/rocm \
        ./target/release/hipfire-eval --model "$model" --battery speed,quality --tier "$TIER" \
        $kldref_arg \
        2>&1 | tee -a "$REPORT" \
        || log "  eval FAILED (exit $?) — continuing"
}

eval_model "qwen3.5-0.8b bf16"  "$MODELS/qwen3.5-0.8b-bf16.hfq"
eval_model "qwen3.5-0.8b mq4"   "$MODELS/qwen3.5-0.8b-mq4.hfq"
eval_model "qwen3.5-0.8b mq6"   "$MODELS/qwen3.5-0.8b-mq6.hfq"

eval_model "qwen3.5-2b bf16"    "$MODELS/qwen3.5-2b-bf16.hfq"
eval_model "qwen3.5-2b mq4"     "$MODELS/qwen3.5-2b-mq4.hfq"
eval_model "qwen3.5-2b mq6"     "$MODELS/qwen3.5-2b-mq6.hfq"

eval_model "qwen3.5-4b bf16"    "$MODELS/qwen3.5-4b-bf16.hfq"
eval_model "qwen3.5-4b mq4"     "$MODELS/qwen3.5-4b-mq4.hfq"
eval_model "qwen3.5-4b mq6"     "$MODELS/qwen3.5-4b-mq6.hfq"

eval_model "qwen3.5-9b bf16"    "$MODELS/qwen3.5-9b-bf16.hfq"
eval_model "qwen3.5-9b mq4"     "$MODELS/qwen3.5-9b-mq4.hfq"
eval_model "qwen3.5-9b mq6"     "$MODELS/qwen3.5-9b-mq6.hfq"
eval_model "qwen3.5-9b mq3"     "$MODELS/qwen3.5-9b-mq3.hfq"

eval_model "qwen3.6-27b mq4"    "$MODELS/qwen3.6-27b-mq4.hfq"
eval_model "qwen3.6-27b mq6"    "$MODELS/qwen3.6-27b-mq6.hfq"
eval_model "qwen3.6-27b mq3"    "$MODELS/qwen3.6-27b-mq3.hfq"

# ── Done ─────────────────────────────────────────────────────────────────────
section "Summary"
log "bench-eval-all complete"
log "report: $REPORT"
echo ""
echo "========================================"
echo "REPORT: $REPORT"
echo "========================================"
