#!/usr/bin/env bash
# 3-fresh-process NIAH wrapper for PFlash claims (PRD §6 Phase 5).
#
# The bench inside a single process is single-shot (no internal repeat),
# so within-process A/B is exposed to DPM ramp + first-call HIP module
# load. This wrapper drives N fresh processes per (target, drafter,
# fixture, mode) point and reports min / median / max + spread.
#
# Use it whenever you publish a TTFT or compress-time number for
# PFlash. A single run is OK for pass/fail (needle survives) but not
# for performance claims.
#
# Usage:
#   ./scripts/pflash-niah-bench.sh <target.hfq> <fixture.jsonl> \
#         [--drafter <drafter.hfq>] [--keep-ratio K] [--block-size B] \
#         [--maxgen N] [--asym3|--q8kv] [--pretok] [--runs N] \
#         [--label tag]
#
# Examples:
#   # Baseline 16K NIAH, 3 fresh runs:
#   ./scripts/pflash-niah-bench.sh \
#       ~/.hipfire/models/qwen3.5-4b.mq4 \
#       benchmarks/longctx/niah/niah_16k.jsonl \
#       --pretok --runs 3 --label baseline
#
#   # PFlash compressed 16K NIAH, 3 fresh runs:
#   ./scripts/pflash-niah-bench.sh \
#       ~/.hipfire/models/qwen3.5-4b.mq4 \
#       benchmarks/longctx/niah/niah_16k.jsonl \
#       --drafter ~/.hipfire/models/qwen3.5-0.8b.mq4 \
#       --keep-ratio 0.30 --pretok --runs 3 --label pflash30
#
# Reports per (mode) point:
#   compress_ms    min / median / max  (only when --drafter set)
#   prefill_ms     min / median / max
#   decode_ms      min / median / max
#   ttft_ms        min / median / max
#   total_ms       min / median / max
#   PASS/FAIL count (needle survival)
#
# Spread above ~5 % on any metric → run was contaminated (other GPU
# users, thermal). Re-run from a quiet box.

set -u
cd "$(dirname "$0")/.."

EXE="./target/release/examples/pflash_niah_bench"
TARGET=""
FIXTURE=""
DRAFTER=""
KEEP_RATIO="0.30"
BLOCK_SIZE="64"
MAXGEN="32"
KV_MODE="--asym3"
PRETOK=0
RUNS=3
LABEL=""

while [ $# -gt 0 ]; do
    case "$1" in
        --drafter) DRAFTER="$2"; shift 2 ;;
        --keep-ratio) KEEP_RATIO="$2"; shift 2 ;;
        --block-size) BLOCK_SIZE="$2"; shift 2 ;;
        --maxgen) MAXGEN="$2"; shift 2 ;;
        --asym3|--q8kv) KV_MODE="$1"; shift ;;
        --pretok) PRETOK=1; shift ;;
        --runs) RUNS="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *)
            if [ -z "$TARGET" ]; then TARGET="$1"; shift; continue; fi
            if [ -z "$FIXTURE" ]; then FIXTURE="$1"; shift; continue; fi
            echo "unknown arg: $1" >&2; exit 2
            ;;
    esac
done

if [ -z "$TARGET" ] || [ ! -f "$TARGET" ]; then
    echo "ERR: missing or invalid target path" >&2; exit 2
fi
if [ -z "$FIXTURE" ] || [ ! -f "$FIXTURE" ]; then
    echo "ERR: missing or invalid fixture path" >&2; exit 2
fi
if [ ! -x "$EXE" ]; then
    echo "ERR: bench binary missing -- build with:" >&2
    echo "  cargo build --release --features deltanet -p engine --example pflash_niah_bench" >&2
    exit 2
fi

LOCK="./scripts/gpu-lock.sh"
if [ -r "$LOCK" ]; then
    # shellcheck disable=SC1090
    . "$LOCK"
    gpu_acquire "pflash-niah-bench${LABEL:+-$LABEL}" \
        || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

stat_line() {
    sort -n | awk '
    { a[NR] = $1 }
    END {
        n = NR
        if (n == 0) { print "?,?,?,?,?"; exit }
        if (n % 2) med = a[(n+1)/2]
        else med = (a[n/2] + a[n/2+1]) / 2
        printf "%.0f %.0f %.0f", a[1], med, a[n]
    }'
}

run_once() {
    local extra=""
    [ "$PRETOK" = "1" ] && extra="$extra --pretok"
    if [ -n "$DRAFTER" ]; then
        extra="$extra --pflash $DRAFTER --keep-ratio $KEEP_RATIO --block-size $BLOCK_SIZE"
    fi
    local out
    out=$("$EXE" "$TARGET" "$FIXTURE" --maxgen "$MAXGEN" $KV_MODE $extra 2>&1)
    local pass=0
    grep -q "^PASS:" <<< "$out" && pass=1
    local compress prefill decode ttft total
    compress=$(grep "^compress:" <<< "$out" | head -1 | grep -oE '[0-9]+ ms' | head -1 | awk '{print $1}')
    prefill=$(grep "^prefill:" <<< "$out" | head -1 | grep -oE '[0-9]+ ms' | head -1 | awk '{print $1}')
    decode=$(grep "^decode:" <<< "$out" | head -1 | grep -oE '[0-9]+ ms' | head -1 | awk '{print $1}')
    ttft=$(grep "^ttft:" <<< "$out" | head -1 | awk '{print $2}')
    total=$(grep "^total:" <<< "$out" | head -1 | awk '{print $2}')
    echo "${compress:-0} ${prefill:-0} ${decode:-0} ${ttft:-0} ${total:-0} $pass"
}

mode_label="baseline"
[ -n "$DRAFTER" ] && mode_label="pflash"

printf "target=%s  fixture=%s  mode=%s  label=%s  runs=%d\n" \
    "$(basename "$TARGET")" "$(basename "$FIXTURE")" "$mode_label" "$LABEL" "$RUNS"

c_samples=(); p_samples=(); d_samples=(); t_samples=(); tot_samples=()
pass_count=0
for run in $(seq 1 "$RUNS"); do
    r=$(run_once)
    read -r c p d t tot pass <<< "$r"
    [ "$pass" = "1" ] && pass_count=$((pass_count + 1))
    c_samples+=("$c"); p_samples+=("$p")
    d_samples+=("$d"); t_samples+=("$t"); tot_samples+=("$tot")
    printf "  run %d: compress=%-5sms prefill=%-5sms decode=%-4sms ttft=%-6sms total=%-6sms %s\n" \
        "$run" "$c" "$p" "$d" "$t" "$tot" "$([ "$pass" = "1" ] && echo PASS || echo FAIL)"
done

emit_stats() {
    local name="$1"; shift
    local stats
    stats=$(printf '%s\n' "$@" | stat_line)
    read -r min med max <<< "$stats"
    local spread="?"
    if [ "$med" != "0" ] && [ "$med" -gt 0 ] 2>/dev/null; then
        spread=$(awk "BEGIN { printf \"%.1f%%\", ($max - $min) / $med * 100 }")
    fi
    printf "  %-10s min=%-6s med=%-6s max=%-6s spread=%s\n" "$name" "$min" "$med" "$max" "$spread"
}

[ -n "$DRAFTER" ] && emit_stats "compress" "${c_samples[@]}"
emit_stats "prefill"  "${p_samples[@]}"
emit_stats "decode"   "${d_samples[@]}"
emit_stats "ttft"     "${t_samples[@]}"
emit_stats "total"    "${tot_samples[@]}"
printf "  passes=%d/%d\n" "$pass_count" "$RUNS"

if [ "$pass_count" -lt "$RUNS" ]; then
    exit 1
fi
