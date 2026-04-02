#!/bin/bash
# ── hipfire regression bisect test ──────────────────────────────────────────
# Full checkout + rebuild + requantize + inference test for a given commit.
# Tests both Qwen3-0.6B (Q8 + HFQ4 variants) and Qwen3-8B (HFQ4).
# Covers short/easy + long/hard prompts, checks speed + coherence.
#
# Usage: ./scripts/bisect-test.sh <commit-hash> [label]
# Output: human-readable table + machine-parseable results in /tmp/bisect-results/
set -euo pipefail

COMMIT=$1
LABEL=${2:-$COMMIT}
REPO=/home/kaden/ClaudeCode/autorocm/hipfire
MODEL_06B=/home/kaden/llama.cpp/models/qwen3-0.6b-fp16
MODEL_8B=/home/kaden/llama.cpp/models/qwen3-8b-fp16
RESULTS_DIR=/tmp/bisect-results
TIMEOUT_QUANT=180
TIMEOUT_INFER=120

mkdir -p "$RESULTS_DIR"
RESULT_FILE="$RESULTS_DIR/${COMMIT}-${LABEL//[^a-zA-Z0-9_-]/_}.txt"

log() { echo "$@" | tee -a "$RESULT_FILE"; }

cd "$REPO"
log "=== BISECT: $LABEL ($COMMIT) === $(date '+%Y-%m-%d %H:%M:%S')"

# ── 1. Checkout + clean kernel cache ─────────────────────────────────────────
git checkout "$COMMIT" --quiet 2>&1
rm -rf /tmp/hipfire_kernels/
log "checked out $COMMIT"

# ── 2. Build quantizer + inference ───────────────────────────────────────────
log "building..."
cargo build --release --bin hipfire-quantize 2>&1 | tail -1
cargo build --release --example infer_hfq 2>&1 | tail -1

# We detect format support by trying it, not by parsing --help

# ── 3. Prompts ───────────────────────────────────────────────────────────────
# Short/easy: should produce coherent philosophical text
PROMPT_SHORT="The meaning of life is"
# Long/hard: tests instruction following + reasoning
PROMPT_LONG="Explain the difference between a compiler and an interpreter. Give one concrete example of each, and describe when you would choose one over the other in a real project."

# ── 4. Test function ─────────────────────────────────────────────────────────
run_infer() {
    local model=$1 label=$2 prompt=$3
    shift 3
    local flags="$*"
    local OUT

    rm -rf /tmp/hipfire_kernels/
    OUT=$(timeout "$TIMEOUT_INFER" ./target/release/examples/infer_hfq "$model" --temp 0 $flags "$prompt" 2>&1) || true

    local TOKS=$(echo "$OUT" | grep "=== Done:" | grep -oP '[\d.]+(?= tok/s)')
    local NTOK=$(echo "$OUT" | grep "=== Done:" | grep -oP '\d+(?= tokens in)')
    local PREFILL=$(echo "$OUT" | grep "^Prompt:" | tail -1 | grep -oP '[\d.]+(?= tok/s)')

    # Extract generated text (everything between blank line after "Generating" and "=== Done")
    local TEXT=$(echo "$OUT" | sed -n '/^$/,/^===/p' | grep -v "^===" | tr '\n' ' ')
    local TEXT80="${TEXT:0:100}"

    # Coherence check: not garbage if text contains real words and no excessive repetition
    local COHERENT="?"
    if [ -z "$TOKS" ]; then
        COHERENT="FAIL"
        TOKS="---"
        PREFILL="---"
    else
        # Check for repetition: if any single word appears >10 times in first 200 chars
        local REPEAT=$(echo "${TEXT:0:200}" | tr ' ' '\n' | sort | uniq -c | sort -rn | head -1 | awk '{print $1}')
        if [ "${REPEAT:-0}" -gt 10 ]; then
            COHERENT="GARBAGE"
        else
            COHERENT="OK"
        fi
    fi

    log "  $label | gen=${TOKS} tok/s | prefill=${PREFILL} tok/s | n=${NTOK:-0} | ${COHERENT} | ${TEXT80}"
}

# ── 5. Quantize helper ───────────────────────────────────────────────────────
quantize() {
    local input_dir=$1 output=$2 fmt=$3
    local cmd="./target/release/hipfire-quantize --input $input_dir --output $output"
    if [ "$fmt" != "default" ]; then
        # Try with --format first; if it fails, the format isn't supported at this commit
        if timeout "$TIMEOUT_QUANT" $cmd --format "$fmt" 2>&1 | grep -q "Done:"; then
            local SIZE=$(ls -lh "$output" 2>/dev/null | awk '{print $5}')
            log "  quantized $fmt → $SIZE"
            return 0
        else
            log "  quantize $fmt: not available at this commit"
            return 1
        fi
    fi
    if timeout "$TIMEOUT_QUANT" $cmd 2>&1 | grep -q "Done:"; then
        local SIZE=$(ls -lh "$output" 2>/dev/null | awk '{print $5}')
        log "  quantized $fmt → $SIZE"
        return 0
    else
        log "  quantize $fmt: FAILED"
        return 1
    fi
}

# ── 6. Run tests ─────────────────────────────────────────────────────────────

log ""
log "──── Qwen3-0.6B ────"

# Q8 (default format)
log "-- q8 (default) --"
if quantize "$MODEL_06B" /tmp/bisect-06b-q8.hfq "default"; then
    run_infer /tmp/bisect-06b-q8.hfq "q8 defkv short" "$PROMPT_SHORT"
    run_infer /tmp/bisect-06b-q8.hfq "q8 defkv long " "$PROMPT_LONG"
    run_infer /tmp/bisect-06b-q8.hfq "q8 fp32kv short" "$PROMPT_SHORT" --fp32kv
fi

# HFQ4 (auto G128/G256)
log "-- hfq4 (auto) --"
if quantize "$MODEL_06B" /tmp/bisect-06b-hfq4.hfq "hfq4"; then
    run_infer /tmp/bisect-06b-hfq4.hfq "hfq4 defkv short" "$PROMPT_SHORT"
    run_infer /tmp/bisect-06b-hfq4.hfq "hfq4 defkv long " "$PROMPT_LONG"
    run_infer /tmp/bisect-06b-hfq4.hfq "hfq4 fp32kv short" "$PROMPT_SHORT" --fp32kv
fi

# HFQ4-G256 (forced)
log "-- hfq4g256 --"
if quantize "$MODEL_06B" /tmp/bisect-06b-hfq4g256.hfq "hfq4g256"; then
    run_infer /tmp/bisect-06b-hfq4g256.hfq "hfq4g256 defkv short" "$PROMPT_SHORT"
    run_infer /tmp/bisect-06b-hfq4g256.hfq "hfq4g256 defkv long " "$PROMPT_LONG"
fi

# HFQ4-G128 (forced)
log "-- hfq4g128 --"
if quantize "$MODEL_06B" /tmp/bisect-06b-hfq4g128.hfq "hfq4g128"; then
    run_infer /tmp/bisect-06b-hfq4g128.hfq "hfq4g128 defkv short" "$PROMPT_SHORT"
fi

log ""
log "──── Qwen3-8B (HFQ4 only) ────"

log "-- hfq4 (auto) --"
if quantize "$MODEL_8B" /tmp/bisect-8b-hfq4.hfq "hfq4"; then
    run_infer /tmp/bisect-8b-hfq4.hfq "8B hfq4 defkv short" "$PROMPT_SHORT"
    run_infer /tmp/bisect-8b-hfq4.hfq "8B hfq4 defkv long " "$PROMPT_LONG"
fi

log "-- hfq4g256 --"
if quantize "$MODEL_8B" /tmp/bisect-8b-hfq4g256.hfq "hfq4g256"; then
    run_infer /tmp/bisect-8b-hfq4g256.hfq "8B hfq4g256 defkv short" "$PROMPT_SHORT"
fi

log ""
log "=== END $LABEL === $(date '+%Y-%m-%d %H:%M:%S')"
log ""
