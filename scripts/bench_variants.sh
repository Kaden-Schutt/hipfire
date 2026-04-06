#!/bin/bash
# bench_variants.sh — Benchmark all 5 RDNA2 GEMV kernel variants back-to-back.
# Usage: ./scripts/bench_variants.sh <model.hfq> [prompt] [extra-flags...]
#
# Runs each HIPFIRE_RDNA2_VARIANT (1-5) on the same model and prompt,
# reports decode tok/s and prefill tok/s for comparison.
# Requires a pre-built infer_hfq binary (cargo build --release --example infer_hfq).
set -euo pipefail

MODEL=${1:?Usage: ./scripts/bench_variants.sh <model.hfq> [prompt] [extra-flags...]}
shift
PROMPT=${1:-"Explain the theory of general relativity in simple terms."}
if [[ "$PROMPT" != -* && $# -gt 0 ]]; then shift; fi
EXTRA_FLAGS=("$@")

REPO=$(cd "$(dirname "$0")/.." && pwd)
INFER=${INFER:-$REPO/target/release/examples/infer_hfq}
TIMEOUT=${TIMEOUT:-120}
WARMUP_PROMPT="Hello"

if [[ ! -x "$INFER" ]]; then
    echo "ERROR: infer_hfq not found at $INFER"
    echo "Build it first: cargo build --release --example infer_hfq"
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model file not found: $MODEL"
    exit 1
fi

# Clear kernel cache so each variant compiles fresh
rm -rf /tmp/hipfire_kernels/

VARIANT_NAMES=(
    ""
    "v1: baseline-rdna2 (32,16) 2x-unroll"
    "v2: high-occupancy (32,20) 2x-unroll"
    "v3: wide-unroll    (32,12) 4x-unroll"
    "v4: dp4a-packed    (32,16) dp4a+factored"
    "v5: cache-aggressive (32,16) 2x+packed+factored"
)

parse_tok_s() {
    local output=$1
    local gen prefill
    gen=$(printf '%s\n' "$output" | sed -n 's/^=== Done: .* (\([0-9.][0-9.]*\) tok\/s) ===$/\1/p' | tail -1)
    prefill=$(printf '%s\n' "$output" | sed -n 's/^Prompt: .* (.* tokens, \([0-9.][0-9.]*\) tok\/s).*$/\1/p' | tail -1)
    printf '%s %s' "${gen:-NA}" "${prefill:-NA}"
}

printf '\n'
printf '╔═══════════════════════════════════════════════════════════════════╗\n'
printf '║         RDNA2 HFQ4-G256 GEMV Kernel Variant Benchmark           ║\n'
printf '╠═══════════════════════════════════════════════════════════════════╣\n'
printf '║ Model: %-57s ║\n' "$(basename "$MODEL")"
printf '║ Prompt: %-56s ║\n' "${PROMPT:0:56}"
printf '╚═══════════════════════════════════════════════════════════════════╝\n'
printf '\n'

# Warmup run (variant 1) to load model and stabilize clocks
printf 'Warming up...\n'
HIPFIRE_RDNA2_VARIANT=1 timeout "$TIMEOUT" "$INFER" "$MODEL" --temp 0 "${EXTRA_FLAGS[@]}" "$WARMUP_PROMPT" >/dev/null 2>&1 || true
printf '\n'

declare -a RESULTS
BEST_TOK=0
BEST_V=0

for V in 1 2 3 4 5; do
    # Clear kernel cache between variants so each compiles its own source
    rm -rf /tmp/hipfire_kernels/

    printf '─── %s ───\n' "${VARIANT_NAMES[$V]}"

    set +e
    OUTPUT=$(HIPFIRE_RDNA2_VARIANT=$V timeout "$TIMEOUT" "$INFER" "$MODEL" --temp 0 "${EXTRA_FLAGS[@]}" "$PROMPT" 2>&1)
    RC=$?
    set -e

    if [[ $RC -ne 0 ]]; then
        if [[ $RC -eq 124 ]]; then
            printf '  RESULT: TIMEOUT (>%ss)\n\n' "$TIMEOUT"
        else
            printf '  RESULT: FAILED (rc=%d)\n' "$RC"
            printf '  stderr: %s\n\n' "$(printf '%s' "$OUTPUT" | tail -3)"
        fi
        RESULTS[$V]="FAIL"
        continue
    fi

    read -r GEN PREFILL <<< "$(parse_tok_s "$OUTPUT")"
    printf '  decode:  %s tok/s\n' "$GEN"
    printf '  prefill: %s tok/s\n\n' "$PREFILL"
    RESULTS[$V]="gen=${GEN} prefill=${PREFILL}"

    # Track best
    if [[ "$GEN" != "NA" ]]; then
        if (( $(echo "$GEN > $BEST_TOK" | bc -l 2>/dev/null || echo 0) )); then
            BEST_TOK=$GEN
            BEST_V=$V
        fi
    fi
done

# Summary table
printf '╔═════╤══════════════════════════════════╤══════════╤══════════╗\n'
printf '║  V  │ Name                             │ Decode   │ Prefill  ║\n'
printf '╠═════╪══════════════════════════════════╪══════════╪══════════╣\n'
for V in 1 2 3 4 5; do
    R=${RESULTS[$V]:-FAIL}
    if [[ "$R" == "FAIL" ]]; then
        printf '║  %d  │ %-32s │  FAIL    │  FAIL    ║\n' "$V" "${VARIANT_NAMES[$V]:4:32}"
    else
        GEN=$(echo "$R" | sed 's/gen=\([^ ]*\).*/\1/')
        PRE=$(echo "$R" | sed 's/.*prefill=\(.*\)/\1/')
        MARKER=""
        if [[ $V -eq $BEST_V ]]; then MARKER=" *"; fi
        printf '║  %d  │ %-32s │ %6s%s │ %6s   ║\n' "$V" "${VARIANT_NAMES[$V]:4:32}" "$GEN" "$MARKER" "$PRE"
    fi
done
printf '╚═════╧══════════════════════════════════╧══════════╧══════════╝\n'

if [[ $BEST_V -gt 0 ]]; then
    printf '\nBest: v%d (%s) at %s tok/s\n' "$BEST_V" "${VARIANT_NAMES[$BEST_V]:4}" "$BEST_TOK"
    printf 'Set as default: export HIPFIRE_RDNA2_VARIANT=%d\n' "$BEST_V"
fi
