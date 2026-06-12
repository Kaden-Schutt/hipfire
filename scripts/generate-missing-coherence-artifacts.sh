#!/usr/bin/env bash

# Generate the straightforward source-only artifacts that currently skip in
# coherence-gate.sh. This intentionally excludes AWQ/GPTQ rows because those
# need an imatrix/calibration policy decision.

set -u
cd "$(dirname "$0")/.."

QUANT_BIN="${HIPFIRE_QUANTIZE:-$PWD/target/release/hipfire-quantize}"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-${HIPFIRE_DIR:-$HOME/.hipfire}/models}"
export HIPFIRE_QUANT_THREADS="${HIPFIRE_QUANT_THREADS:-8}"

if [ ! -x "$QUANT_BIN" ]; then
    echo "error: hipfire-quantize not found: $QUANT_BIN" >&2
    echo "build it with: cargo build --release -p hipfire-quantize" >&2
    exit 2
fi
mkdir -p "$MODELS_DIR"

run_quant() {
    local src="$1"
    local fmt="$2"
    local out="$3"
    local extra="${4:-}"
    local tmp="${out}.tmp"
    local qlog="${out}.quantize.log"
    local start end rc

    echo "[$(date -Iseconds)] START fmt=$fmt out=$out src=$src extra=$extra log=$qlog"
    if [ -f "$out" ]; then
        echo "[$(date -Iseconds)] SKIP exists $out ($(du -h "$out" | cut -f1))"
        return 0
    fi
    if [ ! -d "$src" ] && [ ! -f "$src" ]; then
        echo "[$(date -Iseconds)] SKIP missing source $src"
        return 0
    fi

    rm -f "$tmp"
    start=$(date +%s)
    if [ -n "$extra" ]; then
        # shellcheck disable=SC2086
        "$QUANT_BIN" --input "$src" --output "$tmp" --format "$fmt" $extra > "$qlog" 2>&1
    else
        "$QUANT_BIN" --input "$src" --output "$tmp" --format "$fmt" > "$qlog" 2>&1
    fi
    rc=$?
    end=$(date +%s)

    if [ "$rc" -eq 0 ] && [ -f "$tmp" ]; then
        mv "$tmp" "$out"
        echo "[$(date -Iseconds)] DONE  $out ($(du -h "$out" | cut -f1)) elapsed=$((end-start))s"
    else
        echo "[$(date -Iseconds)] FAIL  rc=$rc out=$out elapsed=$((end-start))s"
        rm -f "$tmp"
    fi
}

Q35_4B="/home/sadara/Models/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
Q35_9B="/home/sadara/Models/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
Q35_27B="/srv/huggingface/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"
Q35_A3B="/srv/huggingface/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754"
Q36_27B="/home/sadara/Models/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
Q36_A3B="/home/sadara/Models/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"

run_quant "$Q35_4B" "mq3" "$MODELS_DIR/qwen3.5-4b-mq3.hfq"
run_quant "$Q35_4B" "lloyd-mq3" "$MODELS_DIR/qwen3.5-4b-lloyd-mq3.hfq" "--allow-mq3-lloyd"
run_quant "$Q35_9B" "mq3" "$MODELS_DIR/qwen3.5-9b-mq3.hfq"
run_quant "$Q35_9B" "lloyd-mq3" "$MODELS_DIR/qwen3.5-9b-lloyd-mq3.hfq" "--allow-mq3-lloyd"
run_quant "$Q35_9B" "q8f16" "$MODELS_DIR/qwen3.5-9b-q8f16.hfq"
run_quant "$Q35_27B" "mq3" "$MODELS_DIR/qwen3.5-27b-mq3.hfq"
run_quant "$Q35_27B" "mq6" "$MODELS_DIR/qwen3.5-27b-mq6.hfq"
run_quant "$Q36_27B" "mq4" "$MODELS_DIR/qwen3.6-27b-mq4.hfq"
run_quant "$Q35_A3B" "mq4" "$MODELS_DIR/qwen3.5-35b-a3b-mq4.hfq"
run_quant "$Q35_A3B" "mq6" "$MODELS_DIR/qwen3.5-35b-a3b-mq6.hfq"
run_quant "$Q35_A3B" "mq3" "$MODELS_DIR/qwen3.5-35b-a3b-mq3.hfq"
run_quant "$Q36_A3B" "mq3" "$MODELS_DIR/qwen3.6-35b-a3b-mq3.hfq"

echo "[$(date -Iseconds)] QUEUE COMPLETE"
