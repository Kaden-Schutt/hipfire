#!/usr/bin/env bash
# Reproducible Gemma 4 E-series prefill-batch sweep on one gfx1100 GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DAEMON="${DAEMON:-$ROOT/target/release/examples/daemon}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/.hipfire/models/gemma4-eseries}"
E2B="${E2B:-$MODEL_ROOT/gemma4-e2b-it-pr439-q8.hfq}"
E4B="${E4B:-$MODEL_ROOT/gemma4-e4b-it-pr439-q8.hfq}"
DATASET="${DATASET:-$ROOT/target/validation/gemma4-eseries/gsm8k/test.jsonl}"
GPU_ID="${GPU_ID:-0}"
ARCH_LABEL="${ARCH_LABEL:-gfx1100}"
EXPECTED_ARCH="${EXPECTED_ARCH:-$ARCH_LABEL}"
BATCHES="${BATCHES:-8 16 32 64}"
LIMIT="${LIMIT:-10}"
MAX_TOKENS="${MAX_TOKENS:-16}"
REPEATS="${REPEATS:-1}"
# Feature policies accept auto/on/off; 1/0 and true/false remain aliases.
FUSED_Q8_PREFILL="${FUSED_Q8_PREFILL:-auto}"
BATCHED_EMBEDDING_PREFILL="${BATCHED_EMBEDDING_PREFILL:-auto}"
PLE_BATCHED_PREFILL="${PLE_BATCHED_PREFILL:-auto}"
PLE_BRANCH_BATCHED_PREFILL="${PLE_BRANCH_BATCHED_PREFILL:-auto}"
PLE_ACTIVATION_FUSED_PREFILL="${PLE_ACTIVATION_FUSED_PREFILL:-auto}"
COOLDOWN="${COOLDOWN:-10}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/target/validation/gemma4-gfx11-prefill-batch/$RUN_ID}"

for file in "$DAEMON" "$E2B" "$E4B" "$DATASET"; do
    [[ -f "$file" ]] || { echo "missing required file: $file" >&2; exit 2; }
done

mkdir -p "$OUT_ROOT"
fused_args=()
append_policy_arg() {
    local value="$1" on_arg="$2"
    case "$value" in
        auto) ;;
        1|on|true) fused_args+=("--$on_arg") ;;
        0|off|false) fused_args+=("--no-$on_arg") ;;
        *) echo "invalid $on_arg policy: $value (expected auto/on/off or 1/0)" >&2; exit 2 ;;
    esac
}
append_policy_arg "$FUSED_Q8_PREFILL" q8-fused-prefill
append_policy_arg "$BATCHED_EMBEDDING_PREFILL" batched-embedding-prefill
append_policy_arg "$PLE_BATCHED_PREFILL" ple-batched-prefill
append_policy_arg "$PLE_BRANCH_BATCHED_PREFILL" ple-branch-batched-prefill
append_policy_arg "$PLE_ACTIVATION_FUSED_PREFILL" ple-activation-fused-prefill
for model in e2b e4b; do
    if [[ "$model" == e2b ]]; then artifact="$E2B"; else artifact="$E4B"; fi
    for batch in $BATCHES; do
        python3 "$ROOT/scripts/eval_gemma4_eseries.py" \
            --daemon "$DAEMON" --model "$artifact" \
            --model-label "gemma4-${model}-${ARCH_LABEL}-b${batch}" \
            --suite gsm8k --dataset "$DATASET" \
            --out-dir "$OUT_ROOT/$model/b${batch}" --physical-gpu "$GPU_ID" \
            --expected-arch "$EXPECTED_ARCH" \
            --runtime-home "/tmp/hipfire-gemma4-${model}-${ARCH_LABEL}-b${batch}" \
            --max-seq 8192 --max-tokens "$MAX_TOKENS" --limit "$LIMIT" \
            --prefill-batch "$batch" --repeats "$REPEATS" --timeout 1800 \
            "${fused_args[@]}"
        sleep "$COOLDOWN"
    done
done

echo "Gemma 4 gfx1100 prefill-batch sweep complete: $OUT_ROOT"
