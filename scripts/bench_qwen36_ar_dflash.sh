#!/usr/bin/env bash
# Qwen3.6 prefill + TG bench (2026-05-30).
#  Group 1: raw AR throughput via bench_qwen35_mq4 (synthetic 232-tok prefill,
#           JIT-stripped, DPM-warmed) for 27B and 35B-A3B.
#  Group 2: 27B real-prompt AR vs DFlash via dflash_spec_demo (PEP-8 prompt).
# Fresh process per measure; RUNS timed runs/cell; median reported.
set -u
cd "$(dirname "$0")/.."

MODELS="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
T27="$MODELS/qwen3.6-27b.mq4"
T35="$MODELS/qwen3.6-35b-a3b.mq4"
DRAFT="$MODELS/qwen36-27b-dflash-mq4.hfq"
PROMPT="benchmarks/prompts/lru_cache_pep8_strict.txt"
BENCH="./target/release/examples/bench_qwen35_mq4"
SPEC="./target/release/examples/dflash_spec_demo"
RUNS="${HIPFIRE_BENCH_RUNS:-3}"
export HIPFIRE_DPM_WARMUP_SECS="${HIPFIRE_DPM_WARMUP_SECS:-10}"

. ./scripts/gpu-lock.sh
gpu_acquire "qwen36-ar-dflash-bench" || { echo "no GPU lock" >&2; exit 2; }
trap 'gpu_release 2>/dev/null || true' EXIT

med() { printf '%s\n' "$@" | grep -v '^NA$' | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]; else print "NA"}'; }
g()   { printf '%s' "$1" | grep -oP "(?<=$2=)[0-9.]+" | tail -1; }   # KEY=val
m()   { printf '%s' "$1" | grep -oP "(?<=^$2: )[0-9.]+" | tail -1; } # KEY: val (BENCH METRICS)

echo "===================================================================="
echo "Qwen3.6 prefill + TG bench   $(date '+%Y-%m-%d %H:%M')"
echo "prompt(group2): $PROMPT  md5=$(md5sum "$PROMPT" | cut -d' ' -f1)"
echo "RUNS=$RUNS  DPM_WARMUP=${HIPFIRE_DPM_WARMUP_SECS}s  kv-mode=q8"
echo "===================================================================="

echo; echo "### Group 1 — raw AR throughput (bench_qwen35_mq4, synthetic 232-tok prefill)"
printf '%-12s %-7s %-12s %-12s\n' model run prefill_tps decode_tps
for cell in "27B|$T27" "35B-A3B|$T35"; do
    label=${cell%|*}; model=${cell#*|}; pre=(); dec=()
    for i in $(seq 1 "$RUNS"); do
        out=$($BENCH "$model" --prefill 232 --prefill-runs 3 --gen 96 --warmup 8 2>&1)
        p=$(g "$out" prefill_tok_s); d=$(g "$out" gen_tok_s)
        pre+=("${p:-NA}"); dec+=("${d:-NA}")
        printf '%-12s %-7s %-12s %-12s\n' "$label" "$i" "${p:-NA}" "${d:-NA}"
    done
    printf '%-12s %-7s %-12s %-12s\n' "$label" MED "$(med "${pre[@]}")" "$(med "${dec[@]}")"
    echo "--------------------------------------------------------------------"
done

echo; echo "### Group 2 — 27B real-prompt AR vs DFlash (dflash_spec_demo, max=256)"
printf '%-14s %-7s %-12s %-12s %-6s\n' config run prefill_tps decode_tps tau
for cell in "27b-AR|--ar-baseline --draft $DRAFT" "27b-DFlash|--draft $DRAFT"; do
    label=${cell%|*}; extra=${cell#*|}; pre=(); dec=(); tau=()
    # untimed warmup
    $SPEC --target "$T27" $extra --prompt-file "$PROMPT" --max 16 --ctx 2048 --kv-mode q8 --no-chatml >/dev/null 2>&1 || true
    for i in $(seq 1 "$RUNS"); do
        out=$($SPEC --target "$T27" $extra --prompt-file "$PROMPT" --max 256 --ctx 2048 --kv-mode q8 --no-chatml 2>&1)
        p=$(m "$out" prefill_tok_s); d=$(m "$out" decode_tok_s); t=$(m "$out" decode_tau)
        pre+=("${p:-NA}"); dec+=("${d:-NA}"); tau+=("${t:-NA}")
        printf '%-14s %-7s %-12s %-12s %-6s\n' "$label" "$i" "${p:-NA}" "${d:-NA}" "${t:-NA}"
    done
    printf '%-14s %-7s %-12s %-12s %-6s\n' "$label" MED "$(med "${pre[@]}")" "$(med "${dec[@]}")" "$(med "${tau[@]}")"
    echo "--------------------------------------------------------------------"
done
echo "done."
