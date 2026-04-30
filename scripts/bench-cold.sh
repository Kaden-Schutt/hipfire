#!/usr/bin/env bash
# Cold-process bench harness — fresh-process distribution for any single
# kernel-tuning experiment.
#
# Use it whenever you need a comparison across two code states that has to
# be NOISE-RESISTANT — i.e. anywhere the kernel-tuning playbook's "trust
# the speed-gate, not your gut" rule kicks in
# (`.skills/hipfire-kernel-tuning/playbook.md` §6). Within-session A/B on
# RDNA is ±10–15 % wide on its own; this wrapper drives the noise band
# down by replacing in-shell repetition with N fresh processes per
# (model, pp) point.
#
# What it does that a single bench invocation doesn't:
#   1. Runs N fresh processes per (model, prefill) point — kernel JIT
#      cost is reset, but the weights file ends up in pagecache so the
#      processes share that.
#   2. Inside each process, runs `--prefill-runs 3` so the reported
#      prefill_tok_s is the third (warm) iteration. The bench's first
#      prefill run pays HIP module-load latency that distorts the timer
#      otherwise.
#   3. Forces HIPFIRE_DPM_WARMUP_SECS=10 so the GPU is at DPM peak before
#      the timed prefill window. Without this you measure clock-ramp on
#      cold launches, particularly on RDNA4 where the gap between idle
#      and peak DPM step is ~30 % of clock.
#   4. Reports min / max / median / spread% per metric so you can eyeball
#      whether the run was clean — anything above ~5 % spread should be
#      treated with suspicion (case-study §2 in the kernel-tuning skill).
#   5. Runs one untimed warm-up process per (model, pp) before the timed
#      runs so pagecache + DPM are both warm at the start of measurement.
#
# This script was the measurement floor for issue #65 (gfx12 K4 unroll on
# the WMMA family). The pre-existing speed-gate.sh remains the source of
# truth for the regression floor; this is for one-off experiments where
# you want a tight median + spread report quickly.
#
# Usage:
#   ./scripts/bench-cold.sh <model.mq4> [--pp 32,128] [--runs 5]
#                                       [--gen 50] [--label tag]
#                                       [--no-warmup] [--sleep N]
#
# Example:
#   ./scripts/bench-cold.sh ~/.hipfire/models/qwen3.5-9b.mq4 \
#       --pp 32,128 --runs 5 --label "before"
#
# Env overrides honored:
#   HIP_VISIBLE_DEVICES   pin a specific GPU on multi-card hosts. Required
#                         on hosts where another card is doing other work
#                         (display compositor, second daemon, etc.) — without
#                         it the contention shows up as 5-10 % spread.
#   HIPFIRE_KERNEL_CACHE  per-checkout HSA cache dir.

set -u
cd "$(dirname "$0")/.."

EXE="./target/release/examples/bench_qwen35_mq4"
MODEL=""
PREFILLS="32,128"
RUNS=5
GEN=50
LABEL=""
SLEEP_BETWEEN=2
WARMUP_RUN=1   # do one untimed warm-up run to stabilize before measuring

while [ $# -gt 0 ]; do
    case "$1" in
        --pp) PREFILLS="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --gen) GEN="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        --no-warmup) WARMUP_RUN=0; shift ;;
        --sleep) SLEEP_BETWEEN="$2"; shift 2 ;;
        -h|--help) sed -n '2,55p' "$0"; exit 0 ;;
        *)
            if [ -z "$MODEL" ]; then MODEL="$1"; shift; continue; fi
            echo "unknown arg: $1" >&2; exit 2
            ;;
    esac
done

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "ERR: missing or invalid model path" >&2; exit 2
fi
if [ ! -x "$EXE" ]; then
    echo "ERR: bench binary missing — build with:" >&2
    echo "  cargo build --release --features deltanet -p engine --example bench_qwen35_mq4" >&2
    exit 2
fi

# ── GPU lock + telemetry ────────────────────────────────────────────────────
LOCK="./scripts/gpu-lock.sh"
if [ -r "$LOCK" ]; then
    # shellcheck disable=SC1090
    . "$LOCK"
    gpu_acquire "bench-cold" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

read_temp() {
    local t
    t=$(cat /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input 2>/dev/null | head -1)
    [ -n "$t" ] && echo $((t / 1000)) || echo "?"
}

# Print "min max median q1 q3" from numbers on stdin (one per line).
stat_line() {
    sort -n | awk '
    { a[NR] = $1 }
    END {
        n = NR
        if (n == 0) { print "?,?,?,?,?"; exit }
        if (n % 2) med = a[(n+1)/2]
        else med = (a[n/2] + a[n/2+1]) / 2
        q1 = a[int((n+1)/4)]; q3 = a[int((3*(n+1))/4)]
        printf "%.1f %.1f %.1f %.1f %.1f", a[1], a[n], med, q1, q3
    }'
}

run_once() {
    local model="$1" pp="$2"
    local out
    # `--prefill-runs 3`: bench reports the LAST prefill's tok/s, so 3 runs
    # gives within-process warm-up for free (skip first-run module-load
    # overhead). Combined with N fresh processes outside, this isolates
    # the per-arch perf from caching artifacts.
    # HIPFIRE_DPM_WARMUP_SECS=10 forces DPM stabilization BEFORE prefill —
    # the bench supports this since the issue #65 work; without it the
    # first ~50 ms of prefill measures clock-ramp not steady-state.
    out=$(HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1 HIPFIRE_DPM_WARMUP_SECS=10 \
        "$EXE" "$model" --prefill "$pp" --prefill-runs 3 --warmup 5 --gen "$GEN" 2>&1 \
        | grep '^SUMMARY' | tail -1)
    local p d
    p=$(echo "$out" | sed -nE 's/.*prefill_tok_s=([0-9.]+).*/\1/p')
    d=$(echo "$out" | sed -nE 's/.*gen_tok_s=([0-9.]+).*/\1/p')
    if [ -n "$p" ] && [ -n "$d" ]; then
        echo "$p $d"
    fi
}

printf "model=%s  label=%s  runs=%s  pp=[%s]  gen=%d  start_temp=%s°C\n" \
    "$(basename "$MODEL")" "$LABEL" "$RUNS" "$PREFILLS" "$GEN" "$(read_temp)"

IFS=',' read -ra PP_ARR <<< "$PREFILLS"
for pp in "${PP_ARR[@]}"; do
    p_samples=()
    d_samples=()
    if [ "$WARMUP_RUN" = "1" ]; then
        # Discard one untimed warm-up run per pp size (loads weights into pagecache,
        # JITs all kernels in this process). Fresh processes still pay JIT, but the
        # weights file ends up in pagecache and DPM has ramped.
        run_once "$MODEL" "$pp" >/dev/null
        sleep "$SLEEP_BETWEEN"
    fi
    for run in $(seq 1 "$RUNS"); do
        r=$(run_once "$MODEL" "$pp")
        if [ -z "$r" ]; then
            printf "  pp%-4s run %d: CRASH\n" "$pp" "$run"
            continue
        fi
        read -r p d <<< "$r"
        p_samples+=("$p")
        d_samples+=("$d")
        printf "  pp%-4s run %d: prefill=%-7s tok/s  decode=%-6s tok/s  temp=%s°C\n" \
            "$pp" "$run" "$p" "$d" "$(read_temp)"
        sleep "$SLEEP_BETWEEN"
    done
    if [ ${#p_samples[@]} -gt 0 ]; then
        p_stats=$(printf '%s\n' "${p_samples[@]}" | stat_line)
        d_stats=$(printf '%s\n' "${d_samples[@]}" | stat_line)
        read -r p_min p_max p_med p_q1 p_q3 <<< "$p_stats"
        read -r d_min d_max d_med d_q1 d_q3 <<< "$d_stats"
        # Spread = (max-min)/median × 100. Anything > ~5 % means the run
        # was contaminated (other GPU users, thermal step, etc.) and the
        # numbers should not be relied on for an A/B comparison.
        spread_p=$(awk "BEGIN { printf \"%.1f\", ($p_max - $p_min) / $p_med * 100 }")
        spread_d=$(awk "BEGIN { printf \"%.1f\", ($d_max - $d_min) / $d_med * 100 }")
        printf "  pp%-4s STATS prefill: median=%s min=%s max=%s spread=%s%%\n" \
            "$pp" "$p_med" "$p_min" "$p_max" "$spread_p"
        printf "  pp%-4s STATS decode:  median=%s min=%s max=%s spread=%s%%\n" \
            "$pp" "$d_med" "$d_min" "$d_max" "$spread_d"
    fi
done
printf "end_temp=%s°C\n" "$(read_temp)"
