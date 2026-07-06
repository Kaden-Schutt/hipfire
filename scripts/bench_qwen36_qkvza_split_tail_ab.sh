#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Qwen3.6-27B RDNA3 QKVZA split-tail A/B prefill benchmark.
#
# This is intentionally narrow: it exercises the production
# forward_prefill_batch path through bench_qwen35_mq4 while toggling only
# HIPFIRE_QKVZA_SPLIT_TAIL. It is meant to accompany the gfx1100/1101/1102
# opt-in patch and provide reviewer-reproducible evidence.

set -euo pipefail

cd "$(dirname "$0")/.."

EXE="${EXE:-./target/release/examples/bench_qwen35_mq4}"
MODEL="${MODEL:-$HOME/.hipfire/models/qwen3.6-27b.mq4}"
GPU_ID="${GPU_ID:-0}"
PREFILL="${PREFILL:-4096}"
PREFILL_RUNS="${PREFILL_RUNS:-3}"
GEN="${GEN:-1}"
WARMUP="${WARMUP:-0}"
KV_MODE="${KV_MODE:-q8}"
DPM_WARMUP_SECS="${DPM_WARMUP_SECS:-5}"
TIMEOUT_SECS="${TIMEOUT_SECS:-360}"
MODE_SEQUENCE="${MODE_SEQUENCE:-off on}"
RESULT_DIR="${RESULT_DIR:-benchmarks/results/qkvza_split_tail_rdna3_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$RESULT_DIR"

export HIP_VISIBLE_DEVICES="$GPU_ID"
export HIPFIRE_KV_MODE="$KV_MODE"
export HIPFIRE_DPM_WARMUP_SECS="$DPM_WARMUP_SECS"

summary_tsv="$RESULT_DIR/summary.tsv"
meta_txt="$RESULT_DIR/meta.txt"

{
    echo "git_head=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "git_branch=$(git branch --show-current 2>/dev/null || true)"
    echo "date=$(date -Is)"
    echo "exe=$EXE"
    echo "model=$MODEL"
    echo "gpu_id=$GPU_ID"
    echo "prefill=$PREFILL"
    echo "prefill_runs=$PREFILL_RUNS"
    echo "gen=$GEN"
    echo "warmup=$WARMUP"
    echo "kv_mode=$KV_MODE"
    echo "dpm_warmup_secs=$DPM_WARMUP_SECS"
    echo "timeout_secs=$TIMEOUT_SECS"
    echo "mode_sequence=$MODE_SEQUENCE"
    echo
    echo "git_status:"
    git status --short 2>/dev/null || true
    echo
    echo "rocm_smi:"
    rocm-smi 2>/dev/null || true
} >"$meta_txt"

printf "seq\tmode\trun\tprefill_ms\tprefill_tok_s\n" >"$summary_tsv"

run_mode() {
    local seq="$1"
    local mode="$2"
    local log="$RESULT_DIR/$(printf "%02d" "$seq")_${mode}.log"

    if [[ "$mode" == "on" ]]; then
        export HIPFIRE_QKVZA_SPLIT_TAIL=1
    else
        unset HIPFIRE_QKVZA_SPLIT_TAIL
    fi

    echo "===== mode=$mode ====="
    timeout "$TIMEOUT_SECS" "$EXE" "$MODEL" \
        --prefill "$PREFILL" \
        --prefill-runs "$PREFILL_RUNS" \
        --gen "$GEN" \
        --warmup "$WARMUP" \
        2>&1 | tee "$log"

    awk -v seq="$seq" -v mode="$mode" '
        /run[[:space:]]+[0-9]+:/ {
            run=$2
            sub(":", "", run)
            ms=$3
            sub("ms", "", ms)
            tps=$4
            printf "%s\t%s\t%s\t%s\t%s\n", seq, mode, run, ms, tps
            printed++
        }
        /^PREFILL_SUMMARY/ && printed == 0 {
            tps = ""
            ms = ""
            for (i = 1; i <= NF; i++) {
                split($i, kv, "=")
                if (kv[1] == "prefill_tok_s") {
                    tps = kv[2]
                } else if (kv[1] == "prefill_wall_ms") {
                    ms = kv[2]
                }
            }
            if (tps != "" && ms != "") {
                printf "%s\t%s\t%s\t%s\t%s\n", seq, mode, 1, ms, tps
                printed++
            }
        }
    ' "$log" >>"$summary_tsv"
}

seq=1
for mode in $MODE_SEQUENCE; do
    case "$mode" in
        off|on) ;;
        *)
            echo "unknown mode in MODE_SEQUENCE: $mode (expected off/on)" >&2
            exit 2
            ;;
    esac
    if [[ "$seq" -gt 1 ]]; then
        sleep "${SLEEP_BETWEEN_MODES:-10}"
    fi
    run_mode "$seq" "$mode"
    seq=$((seq + 1))
done

python3 - "$summary_tsv" <<'PY'
import csv
import statistics
import sys

path = sys.argv[1]
rows = list(csv.DictReader(open(path, newline=""), delimiter="\t"))
by_mode = {}
for row in rows:
    by_mode.setdefault(row["mode"], []).append(float(row["prefill_tok_s"]))

print("\n===== median summary =====")
for mode in ("off", "on"):
    vals = by_mode.get(mode, [])
    if not vals:
        print(f"{mode}\tNA")
        continue
    print(f"{mode}\tmedian_prefill_tok_s={statistics.median(vals):.3f}\truns={len(vals)}")

if by_mode.get("off") and by_mode.get("on"):
    off = statistics.median(by_mode["off"])
    on = statistics.median(by_mode["on"])
    delta = (on / off - 1.0) * 100.0
    print(f"delta_on_vs_off={delta:.2f}%")
PY

echo
echo "results: $RESULT_DIR"
