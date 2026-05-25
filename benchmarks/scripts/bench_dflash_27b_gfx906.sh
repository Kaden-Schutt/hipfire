#!/usr/bin/env bash
# Canonical DFlash 27B bench for the gfx906 MMQ kernel.
# Pinned harness so DFlash claims in PR #158 are reproducible byte-for-byte.
#
# Exercises 3 (target, drafter, prompt) tuples and records per-row
# (tok/s, τ, mean_accept_len, mean_B, runtime). Optionally A/Bs the
# `min_batch` cutover (default 8 since 2026-05-06; previous 16) by
# setting HIPFIRE_BENCH_AB=1.
#
# Prompt files are committed under benchmarks/prompts/ — md5sum recorded
# alongside results to satisfy the CLAUDE.md prompt-md5 rule.
#
# Usage:
#   ./benchmarks/scripts/bench_dflash_27b_gfx906.sh
#   HIPFIRE_BENCH_AB=1 ./benchmarks/scripts/bench_dflash_27b_gfx906.sh
#
# Prerequisites: 27B target + drafter present at $HOME/.hipfire/models/.

set -u
cd "$(dirname "$0")/../.."

EXE=./target/release/examples/dflash_spec_demo
if [ ! -x "$EXE" ]; then
    echo "building dflash_spec_demo..." >&2
    cargo build --release -p hipfire-runtime --example dflash_spec_demo \
        --features deltanet 2>&1 | tail -3
fi

MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
TARGET_35="$MODELS_DIR/qwen3.5-27b.mq4"
TARGET_36="$MODELS_DIR/qwen3.6-27b.mq4"
DRAFT_35="$MODELS_DIR/qwen35-27b-dflash-mq4.hfq"
DRAFT_36="$MODELS_DIR/qwen36-27b-dflash-mq4.hfq"

PROMPT_LRU=benchmarks/prompts/lru_cache_pep8_strict.txt
PROMPT_HE0=benchmarks/prompts/humaneval_0_has_close_elements.txt

# ── Validation ───────────────────────────────────────────────────
missing=0
for f in "$TARGET_35" "$DRAFT_35" "$PROMPT_LRU" "$PROMPT_HE0"; do
    if [ ! -f "$f" ]; then echo "MISSING: $f" >&2; missing=1; fi
done
[ -f "$TARGET_36" ] || echo "WARN: $TARGET_36 absent (3.6 rows will skip)" >&2
[ -f "$DRAFT_36" ]  || echo "WARN: $DRAFT_36 absent (3.6 rows will skip)" >&2
[ "$missing" -eq 1 ] && exit 2

# ── Reproducibility metadata ─────────────────────────────────────
echo "## Reproducibility metadata"
echo "- commit:       $(git rev-parse HEAD)"
echo "- binary md5:   $(md5sum "$EXE" | awk '{print $1}')"
echo "- target 3.5:   $(md5sum "$TARGET_35" | awk '{print $1}')  qwen3.5-27b.mq4"
echo "- draft 3.5:    $(md5sum "$DRAFT_35"  | awk '{print $1}')  qwen35-27b-dflash-mq4"
[ -f "$TARGET_36" ] && \
echo "- target 3.6:   $(md5sum "$TARGET_36" | awk '{print $1}')  qwen3.6-27b.mq4"
[ -f "$DRAFT_36" ] && \
echo "- draft 3.6:    $(md5sum "$DRAFT_36"  | awk '{print $1}')  qwen36-27b-dflash-mq4"
echo "- prompt md5:   $(md5sum "$PROMPT_LRU" | awk '{print $1}')  $(basename $PROMPT_LRU)"
echo "- prompt md5:   $(md5sum "$PROMPT_HE0" | awk '{print $1}')  $(basename $PROMPT_HE0)"
echo "- arch:         $(rocminfo 2>/dev/null | grep -E '^\s+Name:\s+gfx' | head -1 | awk '{print $2}')"
echo

# ── Bench runner ─────────────────────────────────────────────────
# Captures: emitted tokens, wallclock, tok/s, τ, mean_committed.
# Each test runs 3 times; we report the median tok/s.
run_one() {
    local label="$1" target="$2" draft="$3" prompt_file="$4" max="$5"
    local prompt log
    prompt=$(cat "$prompt_file")
    local rows=()
    for run in 1 2 3; do
        log=$(HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
            "$EXE" --target "$target" --draft "$draft" \
                   --prompt "$prompt" --max "$max" --ctx 2048 --no-chatml \
            2>&1)
        local toks  # parse "emitted: <T> tokens in <S>s  (<X> tok/s)"
        toks=$(echo "$log" | grep -oE 'emitted: [0-9]+ tokens in [0-9.]+s\s+\([0-9.]+ tok/s\)' | tail -1)
        local tau
        tau=$(echo "$log" | grep -oE 'τ=[0-9]+\.[0-9]+' | tail -1)
        local mean_b
        mean_b=$(echo "$log" | grep -oE 'mean_B=[0-9.]+' | tail -1)
        rows+=("$toks $tau $mean_b")
    done
    # median by tok/s value
    local sorted
    sorted=$(printf '%s\n' "${rows[@]}" | sort -t'(' -k2,2g)
    local mid
    mid=$(echo "$sorted" | sed -n 2p)
    printf "| %-26s | %-50s |\n" "$label" "$mid"
}

# ── Main ──────────────────────────────────────────────────────────
declare -a TESTS=(
    "27B-3.5/lru_cache    |$TARGET_35|$DRAFT_35|$PROMPT_LRU|120"
    "27B-3.5/humaneval_0  |$TARGET_35|$DRAFT_35|$PROMPT_HE0|120"
)
[ -f "$TARGET_36" ] && [ -f "$DRAFT_36" ] && TESTS+=(
    "27B-3.6/lru_cache    |$TARGET_36|$DRAFT_36|$PROMPT_LRU|120"
    "27B-3.6/humaneval_0  |$TARGET_36|$DRAFT_36|$PROMPT_HE0|120"
)

run_battery() {
    local title="$1"
    echo "## $title"
    echo
    echo "| test                       | result (3-run median: emitted / wall / tok/s, τ, mean_B) |"
    echo "|----------------------------|----------------------------------------------------------|"
    for t in "${TESTS[@]}"; do
        IFS='|' read -r label tgt drf p maxn <<< "$t"
        run_one "$(echo "$label" | xargs)" "$tgt" "$drf" "$p" "$maxn"
    done
    echo
}

if [ "${HIPFIRE_BENCH_AB:-0}" = "1" ]; then
    HIPFIRE_MMQ_MIN_BATCH=16 \
        run_battery "Battery — min_batch=16 (pre-2026-05-06 default)"
    HIPFIRE_MMQ_MIN_BATCH=8 \
        run_battery "Battery — min_batch=8  (2026-05-06+ default)"
else
    run_battery "Battery (default min_batch — 2026-05-06+ = 8)"
fi
