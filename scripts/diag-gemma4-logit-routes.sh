#!/usr/bin/env bash
# Isolate Gemma4 prefill routes with opt-in per-step logit traces.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DAEMON="${DAEMON:-$ROOT/target/release/examples/daemon}"
MODEL="${MODEL:?set MODEL to a Gemma4 HFQ artifact}"
DATA_ROOT="${DATA_ROOT:-$HOME/.hipfire/datasets/longbench-v2}"
DATASET="${DATASET:-$DATA_ROOT/longbench-hard30-pp32k.jsonl}"
MANIFEST="${MANIFEST:-$DATA_ROOT/longbench-hard30-pp32k.manifest.json}"
TASK_ID="${TASK_ID:-longbench-01}"
GPU_ID="${GPU_ID:-0}"
EXPECTED_ARCH="${EXPECTED_ARCH:?set EXPECTED_ARCH to gfx1100 or gfx1201}"
MAX_SEQ="${MAX_SEQ:-65536}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TRACE_STEPS="${TRACE_STEPS:-256}"
FULL_STEPS="${FULL_STEPS:-}"
COOLDOWN="${COOLDOWN:-5}"
MODES="${MODES:-baseline embedding ple_branch ple_activation auto}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/target/validation/gemma4-logit-routes/$EXPECTED_ARCH/$RUN_ID}"

# This is a deterministic correctness diagnostic, not a timing benchmark.
# eval_gemma4_eseries fixes temperature=0 and disables Gemma4 EAGLE/graph.

run_mode() {
    local mode="$1" embedding=false branch=false activation=false
    case "$mode" in
        baseline) ;;
        embedding) embedding=true ;;
        ple_branch) branch=true ;;
        ple_activation) activation=true ;;
        auto) embedding=true; branch=true; activation=true ;;
        *) echo "unknown mode: $mode" >&2; exit 2 ;;
    esac
    local out="$OUT_ROOT/$mode"
    local embedding_arg="--no-batched-embedding-prefill"
    local branch_arg="--no-ple-branch-batched-prefill"
    local activation_arg="--no-ple-activation-fused-prefill"
    [[ "$embedding" == true ]] && embedding_arg="--batched-embedding-prefill"
    [[ "$branch" == true ]] && branch_arg="--ple-branch-batched-prefill"
    [[ "$activation" == true ]] && activation_arg="--ple-activation-fused-prefill"
    mkdir -p "$out/logits"
    HIPFIRE_GEMMA4_LOGIT_TRACE_DIR="$out/logits" \
    HIPFIRE_GEMMA4_LOGIT_TRACE_MAX_STEPS="$TRACE_STEPS" \
    HIPFIRE_GEMMA4_LOGIT_TRACE_FULL_STEPS="$FULL_STEPS" \
    python3 "$ROOT/scripts/eval_gemma4_eseries.py" \
        --daemon "$DAEMON" --model "$MODEL" \
        --model-label "gemma4-logit-${EXPECTED_ARCH}-${TASK_ID}-${mode}" \
        --suite longbench --dataset "$DATASET" --manifest "$MANIFEST" \
        --task-id "$TASK_ID" --out-dir "$out" --physical-gpu "$GPU_ID" \
        --expected-arch "$EXPECTED_ARCH" \
        --runtime-home "/tmp/hipfire-gemma4-logit-${EXPECTED_ARCH}-${mode}" \
        --max-seq "$MAX_SEQ" --max-tokens "$MAX_TOKENS" --prefill-batch 64 \
        --timeout 3600 --no-q8-fused-prefill --no-ple-batched-prefill \
        "$embedding_arg" "$branch_arg" "$activation_arg"
    local task_trace
    task_trace="$(find "$out/logits" -maxdepth 1 -type f -name "${TASK_ID}-*.jsonl" -size +0c -print -quit)"
    if [[ -z "$task_trace" ]]; then
        echo "missing non-empty task logit trace for $mode: $out/logits" >&2
        exit 1
    fi
}

if [[ -e "$OUT_ROOT" ]]; then
    echo "refusing existing OUT_ROOT: $OUT_ROOT" >&2
    exit 2
fi
for mode in $MODES; do
    run_mode "$mode"
    sleep "$COOLDOWN"
done
echo "Gemma4 logit route diagnostics complete: $OUT_ROOT"
