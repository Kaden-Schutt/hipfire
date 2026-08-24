#!/usr/bin/env bash
# Isolated Gemma 4 E-series prefill-route A/B on one gfx1201 GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-$ROOT/target/validation/gemma4-eseries/gsm8k/test.jsonl}"
LIMIT="${LIMIT:-10}"
REPEATS="${REPEATS:-3}"
MAX_TOKENS="${MAX_TOKENS:-16}"
COOLDOWN="${COOLDOWN:-10}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/target/validation/gemma4-gfx12-prefill-routes/$RUN_ID}"

run_mode() {
    local mode="$1"
    local embedding=off branch=off activation=off
    case "$mode" in
        baseline) ;;
        embedding) embedding=on ;;
        ple_branch) branch=on ;;
        ple_activation) activation=on ;;
        *) echo "unknown mode: $mode" >&2; exit 2 ;;
    esac

    OUT_ROOT="$OUT_ROOT/$mode" RUN_ID="$mode" GPU_ID="$GPU_ID" ARCH_LABEL=gfx1201 \
    EXPECTED_ARCH=gfx1201 \
    DATASET="$DATASET" BATCHES=64 LIMIT="$LIMIT" REPEATS="$REPEATS" \
    MAX_TOKENS="$MAX_TOKENS" COOLDOWN="$COOLDOWN" \
    FUSED_Q8_PREFILL=off PLE_BATCHED_PREFILL=off \
    BATCHED_EMBEDDING_PREFILL="$embedding" \
    PLE_BRANCH_BATCHED_PREFILL="$branch" \
    PLE_ACTIVATION_FUSED_PREFILL="$activation" \
        "$ROOT/scripts/bench-gemma4-gfx11-prefill-batch.sh"
}

mkdir -p "$OUT_ROOT"
for mode in baseline embedding ple_branch ple_activation; do
    run_mode "$mode"
    sleep "$COOLDOWN"
done

echo "Gemma 4 gfx1201 route A/B complete: $OUT_ROOT"
