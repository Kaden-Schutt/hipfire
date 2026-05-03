#!/usr/bin/env bash
# PFlash regression gate.
#
# Runs the 6 Phase 5 NIAH fixtures (8K, 16K, multi-16K, longcode,
# longprose, 32K) under both baseline and PFlash modes against
# qwen3.5-27b.mq3 + qwen3.5-0.8b.mq4 drafter, then asserts:
#   1. Verdict matches the recorded baseline (PASS stays PASS,
#      FAIL stays FAIL; flips are quality regressions).
#   2. Total wall-clock within ±10% of the recorded median.
#
# Baseline JSON: scripts/pflash-baselines/<arch>-<date>.json
# Default: scripts/pflash-baselines/gfx1100-2026-05-02.json
#
# Exit codes:
#   0  every fixture matches verdict and stays inside ±10%
#   1  one or more fixtures regressed (verdict flip OR drift > 10%)
#   2  build / environment error
#
# Usage:
#   ./scripts/pflash-gate.sh                # default baseline
#   ./scripts/pflash-gate.sh --baseline scripts/pflash-baselines/<file>.json
#   ./scripts/pflash-gate.sh --tolerance 15 # widen to ±15%
#
# Hooks into scripts/coherence-gate.sh as a follow-up stage.

set -u
cd "$(dirname "$0")/.."

BASELINE="scripts/pflash-baselines/gfx1100-2026-05-02.json"
TOLERANCE_PCT=10
MAXGEN_NIAH=32
MAXGEN_LONG=64
MAXGEN_MULTI=80

while [ $# -gt 0 ]; do
    case "$1" in
        --baseline) BASELINE="$2"; shift 2 ;;
        --tolerance) TOLERANCE_PCT="$2"; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ ! -f "$BASELINE" ]; then
    echo "pflash-gate: baseline not found: $BASELINE" >&2
    exit 2
fi

EXE="./target/release/examples/pflash_niah_bench"

# Rebuild if the binary is missing OR any relevant source is newer than it.
# Without this freshness check, the gate would run against a stale binary
# whenever someone forgot to rebuild after editing pflash.rs / qwen35.rs /
# llama.rs / the bench itself / the score kernel and silently report a
# pass on outdated code. Mirror the pattern from scripts/coherence-gate.sh.
#
# Coverage extends through the full PFlash kernel wiring chain so a change
# anywhere in the dispatch path forces a rebuild:
#   - engine logic (pflash.rs, qwen35.rs, llama.rs, hfq.rs, tokenizer.rs)
#   - bench harness (pflash_niah_bench.rs)
#   - rdna-compute layer:
#       lib.rs       (Gpu struct, top-level exports)
#       kernels.rs   (include_str! site for every kernel HIP source)
#       dispatch.rs  (the `pflash_score_q8_kv` Rust wrapper)
#       compiler.rs  (builds .hsaco from .hip source)
#       pool.rs      (GPU memory pool the kernel allocates from)
#   - hip-bridge layer (the actual GPU launch primitives the wrapper calls):
#       lib.rs       (top-level FFI exports)
#       ffi.rs       (launch_kernel + launch_kernel_blob primitives)
#       kernarg.rs   (kernel-argument packing for the launch)
#       error.rs     (HipResult / HipError types in every signature)
#   - the .hip kernel source itself
rebuild=0
if [ ! -x "$EXE" ]; then
    rebuild=1
else
    for src in \
        crates/engine/src/pflash.rs \
        crates/engine/src/qwen35.rs \
        crates/engine/src/llama.rs \
        crates/engine/src/hfq.rs \
        crates/engine/src/tokenizer.rs \
        crates/engine/examples/pflash_niah_bench.rs \
        crates/rdna-compute/src/lib.rs \
        crates/rdna-compute/src/kernels.rs \
        crates/rdna-compute/src/dispatch.rs \
        crates/rdna-compute/src/compiler.rs \
        crates/rdna-compute/src/pool.rs \
        crates/hip-bridge/src/lib.rs \
        crates/hip-bridge/src/ffi.rs \
        crates/hip-bridge/src/kernarg.rs \
        crates/hip-bridge/src/error.rs \
        kernels/src/pflash_score_q8_kv.hip \
    ; do
        if [ -f "$src" ] && [ "$src" -nt "$EXE" ]; then
            rebuild=1
            break
        fi
    done
fi
if [ "$rebuild" -eq 1 ]; then
    echo "pflash-gate: rebuilding pflash_niah_bench (binary missing or stale)..."
    if ! cargo build --release --features deltanet --example pflash_niah_bench -p engine >&2; then
        echo "pflash-gate: build failed" >&2
        exit 2
    fi
fi

TARGET="${HIPFIRE_PFLASH_TARGET:-$HOME/.hipfire/models/qwen3.5-27b.mq3}"
DRAFTER="${HIPFIRE_PFLASH_DRAFTER:-$HOME/.hipfire/models/qwen3.5-0.8b.mq4}"
if [ ! -f "$TARGET" ] || [ ! -f "$DRAFTER" ]; then
    echo "pflash-gate: target or drafter not present" >&2
    echo "  target  = $TARGET" >&2
    echo "  drafter = $DRAFTER" >&2
    exit 2
fi

LOCK="./scripts/gpu-lock.sh"
if [ -r "$LOCK" ]; then
    # shellcheck disable=SC1090
    . "$LOCK"
    gpu_acquire "pflash-gate" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

echo "pflash-gate: baseline=$BASELINE tolerance=±${TOLERANCE_PCT}%"
echo "pflash-gate: target=$(basename "$TARGET") drafter=$(basename "$DRAFTER")"
echo

# Read baseline rows. We use python for robust JSON without adding deps.
ROWS_JSON=$(python3 -c "
import json, sys
b = json.load(open(sys.argv[1]))
for r in b['fixtures']:
    print(f\"{r['label']}|{r['fixture']}|{r['mode']}|{r['total_ms']}|{r['verdict']}\")
" "$BASELINE")

regressions=0
total_rows=0

run_fixture() {
    local fixture="$1" mode="$2"
    local maxgen="$MAXGEN_NIAH"
    case "$fixture" in
        *multi*)     maxgen="$MAXGEN_MULTI" ;;
        *longcode*|*longprose*) maxgen="$MAXGEN_LONG" ;;
    esac
    local extra="--pretok"
    if [ "$mode" = "pflash" ]; then
        extra="$extra --pflash $DRAFTER --keep-ratio 0.30 --block-size 64"
    fi
    "$EXE" "$TARGET" "$fixture" --maxgen "$maxgen" --asym3 $extra 2>&1
}

extract_total() {
    grep '^total:' <<< "$1" | head -1 | awk '{print $2}'
}
extract_verdict() {
    if grep -q '^PASS:' <<< "$1"; then
        echo "PASS"
    elif grep -q '^FAIL:' <<< "$1"; then
        echo "FAIL"
    else
        echo "UNKNOWN"
    fi
}

while IFS='|' read -r label fixture mode baseline_total_ms baseline_verdict; do
    [ -z "$label" ] && continue
    total_rows=$((total_rows + 1))
    out=$(run_fixture "$fixture" "$mode")
    actual_total=$(extract_total "$out")
    actual_verdict=$(extract_verdict "$out")

    if [ -z "$actual_total" ] || [ "$actual_verdict" = "UNKNOWN" ]; then
        echo "  [REGRESS] $label: bench failed to produce timing or verdict"
        regressions=$((regressions + 1))
        continue
    fi

    drift_pct=$(awk "BEGIN { printf \"%.1f\", ($actual_total - $baseline_total_ms) / $baseline_total_ms * 100 }")
    abs_drift=$(awk "BEGIN { printf \"%.1f\", ($actual_total - $baseline_total_ms < 0 ? -1 : 1) * ($actual_total - $baseline_total_ms) / $baseline_total_ms * 100 }")
    verdict_match="ok"
    [ "$actual_verdict" != "$baseline_verdict" ] && verdict_match="REGRESS"
    drift_match="ok"
    if awk "BEGIN { exit !($abs_drift > $TOLERANCE_PCT) }"; then
        drift_match="REGRESS"
    fi

    if [ "$verdict_match" = "REGRESS" ] || [ "$drift_match" = "REGRESS" ]; then
        regressions=$((regressions + 1))
        printf "  [REGRESS] %-30s baseline=%6sms,%s actual=%6sms,%s drift=%s%%\n" \
            "$label" "$baseline_total_ms" "$baseline_verdict" \
            "$actual_total" "$actual_verdict" "$drift_pct"
    else
        printf "  [ok]      %-30s baseline=%6sms,%s actual=%6sms,%s drift=%s%%\n" \
            "$label" "$baseline_total_ms" "$baseline_verdict" \
            "$actual_total" "$actual_verdict" "$drift_pct"
    fi
done <<< "$ROWS_JSON"

echo
echo "pflash-gate: $((total_rows - regressions))/$total_rows rows clean"
if [ "$regressions" -gt 0 ]; then
    echo "pflash-gate: FAIL ($regressions regression(s))"
    exit 1
fi
echo "pflash-gate: PASS"
exit 0
