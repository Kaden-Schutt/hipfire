#!/usr/bin/env bash
# stage_models.sh — pull + quantize the full MQ4/MQ6 model matrix from
# HuggingFace on a fresh pod. Uses the pod's HF DL link (~270 MB/s on
# a datacenter box) to download safetensors, then runs `hipfire-quantize`
# per (model × format) pair. Output land in $MODELS_DIR.
#
# Way faster than scp'ing pre-quantized files from a residential link
# (~7× net DL advantage). Quantization itself is ~30-60s per model.
#
# Usage:
#   bash stage_models.sh [--dir /root/models] [--no-mq6]
#
# Models staged by default (covering tonight's hermes-cal sweep):
#   - Qwen3.5-4B       → mq4, mq6  (small)
#   - Qwen3.5-9B       → mq4, mq6
#   - Qwen3.5-27B      → mq4, mq6
#   - Qwen3.5-35B-A3B  → mq4  (MoE, no mq6 currently)
#   - Qwen3.6-35B-A3B  → mq4  (MoE)
#
# Carnice is not on the public Qwen org — left to user to scp or pull
# from schuttdev/hipfire-carnice-* (already MQ4-quantized, no HF
# safetensors path). Same for local-only finetunes.

set -uo pipefail

MODELS_DIR="/root/models"
INCLUDE_MQ6=1
QUANT_BIN="${HIPFIRE_QUANTIZE:-/root/hipfire/target/release/hipfire-quantize}"

while [ $# -gt 0 ]; do
    case "$1" in
        --dir) MODELS_DIR="$2"; shift 2 ;;
        --no-mq6) INCLUDE_MQ6=0; shift ;;
        --bin) QUANT_BIN="$2"; shift 2 ;;
        --help|-h) sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

if [ ! -x "$QUANT_BIN" ]; then
    echo "ERROR: hipfire-quantize not found at $QUANT_BIN (run amd_quickdeploy.sh first)" >&2
    exit 2
fi

mkdir -p "$MODELS_DIR"
log() { printf '[stage-models] %s\n' "$*"; }

# MODEL = HF repo ID, OUT_STEM = filename stem in MODELS_DIR (the quant
# format will append .mq4 / .mq6 to produce the final filename).
#
# Format syntax: hipfire-quantize auto-downloads when --input is
# "org/name". Output filename convention is <stem>.<format>.
MATRIX=(
    "Qwen/Qwen3.5-4B:qwen3.5-4b:mq4"
    "Qwen/Qwen3.5-9B:qwen3.5-9b:mq4"
    "Qwen/Qwen3.5-27B:qwen3.5-27b:mq4"
    "Qwen/Qwen3.5-35B-A3B:qwen3.5-35b-a3b:mq4"
    "Qwen/Qwen3.6-35B-A3B:qwen3.6-35b-a3b:mq4"
)
if [ "$INCLUDE_MQ6" -eq 1 ]; then
    MATRIX+=(
        "Qwen/Qwen3.5-4B:qwen3.5-4b:mq6"
        "Qwen/Qwen3.5-9B:qwen3.5-9b:mq6"
        "Qwen/Qwen3.5-27B:qwen3.5-27b:mq6"
    )
fi

log "staging ${#MATRIX[@]} (model × format) pairs into $MODELS_DIR"
log "hipfire-quantize: $QUANT_BIN"
log ""

FAILED=()
for spec in "${MATRIX[@]}"; do
    IFS=':' read -r hf_id stem fmt <<< "$spec"
    out="${MODELS_DIR}/${stem}.${fmt}"
    if [ -f "$out" ]; then
        log "skip  $out (already exists, $(du -h "$out" | cut -f1))"
        continue
    fi
    log "→ $hf_id → $out [$fmt]"
    t0=$(date +%s)
    if "$QUANT_BIN" --input "$hf_id" --output "$out" --format "$fmt" 2>&1 | tail -8; then
        t1=$(date +%s)
        log "   done in $((t1-t0))s ($(du -h "$out" | cut -f1))"
    else
        rc=$?
        log "   FAILED rc=$rc for $hf_id [$fmt]"
        FAILED+=("$hf_id/$fmt")
    fi
    echo ""
done

log "────────────────────────────────────────────"
if [ ${#FAILED[@]} -eq 0 ]; then
    log "all ${#MATRIX[@]} model/format pairs staged OK"
    log "next: bash calibrate_multigpu.sh --models <comma-separated> --corpus <corpus.txt>"
else
    log "FAILURES: ${FAILED[*]}"
    exit 1
fi
