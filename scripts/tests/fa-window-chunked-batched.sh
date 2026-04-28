#!/usr/bin/env bash
# fa-window-chunked-batched.sh — exercise the long-ctx batched verify chunking
# regime where the Codex-caught row-index bug manifests.
#
# The bug: in `attention_flash_asym3_tile_batched`, tree-mode `q_pos` used
# `local_bid` (chunk-local 0..chunk-1) instead of `global_bid` (= batch_offset
# + local_bid). When `launch_asym_flash_batched` chunks the verify because
# the partials buffer can't hold all B rows at once, every chunk past the
# first computes kv_start from the wrong row position.
#
# Trigger conditions:
#   1. DFlash spec-decode active (target + draft → batched verify with B=16)
#   2. Active sliding window (HIPFIRE_FA_WINDOW > 0)
#   3. Long enough context that partials buffer forces sub_batch < B=16
#      (typically kicks in around ctx=8K+ on 27B asym3)
#
# Failure mode:
#   - Past chunk 0, kv_start is computed from local_bid not global_bid.
#   - Window mask gets applied to the wrong rows.
#   - Softmax weights corrupt → output garbage / attractor / panic.
#
# This test runs DFlash at ctx≈6K with window=2048 and checks the output
# stays coherent (unique_ratio > 0.3, no single-token > 50%). With the
# fix it passes; without the fix the second-half rows produce corrupted
# logits and the detector flags it.

set -e
cd "$(dirname "$0")/../.."

DFLASH_EXE="./target/release/examples/dflash_spec_demo"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

TARGET="$MODELS_DIR/qwen3.6-27b.mq4"
DRAFT="$MODELS_DIR/qwen36-27b-dflash-mq4.hfq"

if [ ! -x "$DFLASH_EXE" ]; then
    echo "ERROR: $DFLASH_EXE not built." >&2
    exit 2
fi
if [ ! -f "$TARGET" ] || [ ! -f "$DRAFT" ]; then
    echo "SKIP — target or draft missing (target=$TARGET draft=$DRAFT)"
    exit 0
fi

source "$LOCK_SCRIPT"
gpu_acquire "fa-window-chunked-batched" || exit 2
trap 'gpu_release 2>/dev/null || true' EXIT

# Build a long prompt that gets us into the chunked regime.
# 60K chars of wikitext ≈ 14-16K tokens, exercises chunking on 27B.
# Cap kv at 17K so prefill fits VRAM.
PROMPT_FILE="/tmp/fa-chunked-prompt.txt"
head -c 28000 dev/bench/data/wikitext2-test.txt > "$PROMPT_FILE"
printf '\n\nQuestion: Summarize the above text in two clear sentences:' >> "$PROMPT_FILE"
PROMPT="$(cat "$PROMPT_FILE")"

run_dflash() {
    local window="$1"
    env HIPFIRE_FA_WINDOW="$window" timeout 240 "$DFLASH_EXE" \
        --target "$TARGET" --draft "$DRAFT" --prompt "$PROMPT" \
        --max 80 --ctx 8500 --kv-mode asym3 --no-chatml 2>&1
}

# Detector: extract the DFlash tokens dump line and compute attractor metrics.
analyze() {
    local out="$1"
    if echo "$out" | grep -qE "panicked|FATAL|error: "; then
        echo "PANIC"
        return
    fi
    # Token list from dflash_spec_demo: "DFlash tokens: [12, 34, ...]"
    local tokens
    tokens=$(echo "$out" | sed -nE 's/^DFlash tokens: \[([^]]+)\].*/\1/p' | tr ',' ' ')
    if [ -z "$tokens" ]; then
        echo "ZERO_TOKENS"
        return
    fi
    # Compute unique ratio + max token frequency in python.
    python3 - <<PY
import sys
tokens = [int(t) for t in "$tokens".split() if t.strip()]
if not tokens:
    print("ZERO_TOKENS")
    sys.exit(0)
n = len(tokens)
unique = len(set(tokens))
unique_ratio = unique / n
from collections import Counter
counts = Counter(tokens)
max_token, max_count = counts.most_common(1)[0]
max_freq = max_count / n
print(f"n={n} unique_ratio={unique_ratio:.3f} max_freq={max_freq:.3f}")
# Attractor thresholds (from feedback_dflash_coherence_gate_mandatory.md)
if unique_ratio < 0.30:
    print("ATTRACTOR — unique_ratio < 0.30")
    sys.exit(1)
if max_freq > 0.50:
    print(f"ATTRACTOR — max_freq > 0.50 (token {max_token})")
    sys.exit(1)
print("OK")
PY
}

echo "=== Long-ctx chunked batched verify regression ==="
echo "  target: $TARGET"
echo "  draft:  $DRAFT"
echo

for window in 0 2048; do
    label=$( [ "$window" = "0" ] && echo "DEFAULT" || echo "WIN-$window" )
    printf "  %-12s " "$label"
    out=$(run_dflash "$window")
    result=$(analyze "$out")
    if echo "$result" | grep -qE "PANIC|ATTRACTOR|ZERO_TOKENS"; then
        echo "FAIL"
        echo "$result"
        echo "$out" | tail -20
        exit 1
    fi
    metrics=$(echo "$result" | head -1)
    tps=$(echo "$out" | sed -nE 's/.*emitted: [0-9]+ tokens in [0-9.]+s\s+\(([0-9.]+) tok\/s\).*/\1/p' | tail -1)
    tau=$(echo "$out" | sed -nE 's/.*τ=([0-9.]+).*/\1/p' | tail -1)
    printf "PASS  %s  tok/s=%s τ=%s\n" "$metrics" "${tps:-?}" "${tau:-?}"
done

echo
echo "FA-WINDOW CHUNKED BATCHED: clean (chunked verify with active window does not corrupt)"
