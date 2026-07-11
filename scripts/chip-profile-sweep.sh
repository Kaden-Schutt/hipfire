#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# chip-profile-sweep.sh — folded warm-DPM multi-device chip-profile sweep.
#
# Orchestrates the two chip-profile microbenches from
# docs/superpowers/plans/2026-07-01-gfx1201-phaseA-perf-instrument.md
# Task 3/4 (`pointer_chase_latency` -> mem_latency_ns, `peak_bw_probe` ->
# peak_bw_gbps, both under crates/rdna-compute/examples/, both timed via the
# hipEvent pattern in crates/rdna-compute/src/profile.rs) across every
# HIP-visible device on this box, applying the cold-vs-warm DPM discipline
# this session had to do BY HAND to land tests/chip-profiles/gfx1201.json
# (commits 542af081/db5e1efd/c1863874): a cold, DPM-asleep card can read
# 2-3x worse than a warm one (gfx1201 R9700: cold mem_latency_ns 548.8 vs
# warm 205.9 — 2.65x), so a naive single-shot measurement silently ships a
# pessimistic number.
#
# Protocol per device:
#   1. COLD pass  — run each microbench as-found (no forced DPM state).
#      Informational only; box-dependent idle-clock noise, per the
#      cross-arch cold sweep already recorded in gfx1201.json's provenance
#      (gfx1010=769.7 gfx1030=152.1 gfx1100=310.1 gfx1151=220.0(no-sleep
#      iGPU) gfx1201=548.8 ns cold).
#   2. Best-effort `rocm-smi --setperflevel high` (box-wide — see the
#      HIP/ROCR-vs-rocm-smi index-mismatch note below for why this is
#      NEVER done with a per-device `-d N`).
#   3. WARM pass  — run each microbench again immediately after.
#   4. FOLD — reduce {cold, warm} down to the single number that gets
#      copied into tests/chip-profiles/<arch>.json: the warm value, always
#      (matching the gfx1201.json precedent — cold is diagnostic-only, never
#      the number of record, even when it happens to look "better"). A warm
#      pass that does NOT improve on cold by at least $ANOMALY_PCT gets
#      flagged loud in the summary: that pattern means force-high likely did
#      not take effect (no rocm-smi permission, thermal throttle mid-sweep,
#      etc.), not that the number is trustworthy.
#
# Multi-device + eGPU-vs-native corroboration: HIP_VISIBLE_DEVICES indices
# are the ONLY trustworthy device selector here. rocm-smi's `GPU[N]` order
# does NOT match ROCR/HIP's enumeration order on multi-GPU boxes (hipx once
# ran hours of "5700 XT" benches that were actually the R9700 via the
# mismatch — see project-memory feedback_rocr_hip_visible_enumeration).
# Every summary row is tagged with the `arch: gfxNNNN` line the microbench
# itself prints at runtime (the HIP-runtime ground truth), never with an
# assumed index->arch mapping — that's what makes rows comparable across
# boxes (e.g. hiptrx native-PCIe R9700 vs hipx TBT5-eGPU R9700, which this
# session found agree within 0.5% on mem_latency_ns once both are warm).
#
# Usage:
#   bash scripts/chip-profile-sweep.sh [--devices 0,1,2] [--skip-latency]
#       [--skip-bw] [--no-force-dpm] [--out DIR]
#
# Flags:
#   --devices LIST     comma-separated HIP_VISIBLE_DEVICES indices to sweep.
#                       Default: auto-detect via rocminfo agent count (GPU
#                       agents = total Agent lines minus 1 CPU agent).
#   --skip-latency      don't run pointer_chase_latency.
#   --skip-bw           don't run peak_bw_probe.
#   --no-force-dpm      don't attempt rocm-smi --setperflevel high; the warm
#                       pass then relies solely on the microbenches' own
#                       internal untimed warmup (weaker signal — expect a
#                       smaller cold/warm delta than the gfx1201.json
#                       precedent, not none).
#   --out DIR           override the output directory (default: durable
#                       benchmarks/results/<timestamp>-chip-profile-sweep/,
#                       never /tmp — see project-memory
#                       feedback_perf_bench_tmp_fragile).
#
# Output:
#   <out>/summary.md    folded table (human, includes anomaly flags)
#   <out>/summary.tsv   folded table (machine-parseable)
#   <out>/device<N>.{cold,warm}.{latency,bw}.log   raw per-pass stdout
#
# GPU-required. Coordinates via scripts/gpu-lock.sh when present (best-effort
# — the sweep still runs if the lock script is unreadable, matching the
# fallback in scripts/mq3-mq2-sweep.sh).
#
# Exit codes:
#   0  every device/pass produced a plausible measurement
#   1  at least one device/pass failed (binary crash, implausible-value
#      assert, missing arch/value line) — summary is still written for the
#      rest of the sweep
#   2  setup error (build failed, binaries missing, bad flags)

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$ROOT" || exit 2

LATENCY_BIN="target/release/examples/pointer_chase_latency"
BW_BIN="target/release/examples/peak_bw_probe"
LOCK_SCRIPT="scripts/gpu-lock.sh"

DEVICES=""
RUN_LATENCY=1
RUN_BW=1
FORCE_DPM=1
OUT_DIR=""
ANOMALY_PCT=5   # warm not beating cold by more than this % gets flagged

log() { printf '[chip-profile-sweep] %s\n' "$*"; }

usage() {
    cat <<'EOF'
Usage: bash scripts/chip-profile-sweep.sh [--devices 0,1,2] [--skip-latency]
    [--skip-bw] [--no-force-dpm] [--out DIR]

Folded warm-DPM multi-device sweep over pointer_chase_latency (mem_latency_ns)
and peak_bw_probe (peak_bw_gbps). See the header comment in this file for the
full cold/warm/fold protocol and the rocm-smi-vs-HIP index-mismatch rationale.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --devices) DEVICES="$2"; shift 2 ;;
        --skip-latency) RUN_LATENCY=0; shift ;;
        --skip-bw) RUN_BW=0; shift ;;
        --no-force-dpm) FORCE_DPM=0; shift ;;
        --out) OUT_DIR="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown flag: $1" >&2; usage; exit 2 ;;
    esac
done

if [ "$RUN_LATENCY" = 0 ] && [ "$RUN_BW" = 0 ]; then
    echo "ERROR: --skip-latency and --skip-bw together leave nothing to sweep" >&2
    exit 2
fi

# ── Build ────────────────────────────────────────────────────────────────
build_args=()
[ "$RUN_LATENCY" = 1 ] && build_args+=(--example pointer_chase_latency)
[ "$RUN_BW" = 1 ] && build_args+=(--example peak_bw_probe)
log "building: cargo build --release -p rdna-compute ${build_args[*]}"
if ! cargo build --release -p rdna-compute "${build_args[@]}"; then
    echo "ERROR: build failed" >&2
    exit 2
fi
if [ "$RUN_LATENCY" = 1 ] && [ ! -x "$LATENCY_BIN" ]; then
    echo "ERROR: $LATENCY_BIN missing after build" >&2
    exit 2
fi
if [ "$RUN_BW" = 1 ] && [ ! -x "$BW_BIN" ]; then
    echo "ERROR: $BW_BIN missing after build" >&2
    exit 2
fi

# ── Device enumeration ──────────────────────────────────────────────────
# HIP_VISIBLE_DEVICES / ROCR indices are the ONLY trustworthy device
# selector (see module header) — never derived from rocm-smi indices.
declare -a DEV_LIST=()
if [ -n "$DEVICES" ]; then
    IFS=',' read -ra DEV_LIST <<< "$DEVICES"
else
    n_agents=0
    if command -v rocminfo >/dev/null 2>&1; then
        n_agents=$(rocminfo 2>/dev/null | grep -c "Agent ")
    fi
    n_gpus=$(( n_agents > 0 ? n_agents - 1 : 0 ))
    if [ "$n_gpus" -lt 1 ]; then
        log "WARN: could not detect GPU count via rocminfo; defaulting to device 0 only"
        n_gpus=1
    fi
    for ((i = 0; i < n_gpus; i++)); do DEV_LIST+=("$i"); done
fi
log "devices to sweep: ${DEV_LIST[*]}"

# ── Cleanup trap — installed BEFORE any state-mutating action below (GPU
# lock acquire, DPM force) so a crash/signal between "state mutated" and
# "trap installed" can never leak the GPU lock or leave DPM forced-high.
# All vars cleanup() reads are initialized here, ahead of the trap, so it's
# always safe to fire even if the very next line is what dies.
USE_GPU_LOCK=0
PRIOR_PERFLEVEL="auto"
DPM_FORCED=0

try_rocm_smi_set() {
    rocm-smi --setperflevel "$1" >/dev/null 2>&1 \
        || sudo -n rocm-smi --setperflevel "$1" >/dev/null 2>&1
}

cleanup() {
    if [ "$DPM_FORCED" = 1 ]; then
        if try_rocm_smi_set "$PRIOR_PERFLEVEL"; then
            log "DPM: restored power_dpm_force_performance_level=$PRIOR_PERFLEVEL"
        else
            log "WARN: failed to restore prior DPM perf level ($PRIOR_PERFLEVEL)"
        fi
    fi
    if [ "$USE_GPU_LOCK" = 1 ]; then
        gpu_release 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── GPU lock (best-effort, matches scripts/mq3-mq2-sweep.sh) ────────────
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "chip-profile-sweep" || { echo "could not acquire GPU lock" >&2; exit 2; }
    USE_GPU_LOCK=1
fi

# ── DPM force-high (box-wide, best-effort) ──────────────────────────────
# NEVER `rocm-smi -d N` here — rocm-smi's GPU[N] order and HIP/ROCR's
# HIP_VISIBLE_DEVICES order are DIFFERENT enumerations on multi-GPU boxes
# (observed reversed on hipx: rocm-smi GPU[0]=6950XT was HIP index 1).
# Forcing the perf level box-wide sidesteps the mismatch entirely; it is
# harmless to also force-high a device this sweep isn't currently measuring.
# (For the same reason this never reads a per-card `pp_dpm_mclk`/sysfs star
# line either — that requires a /sys/class/drm/cardN -> HIP-index mapping,
# i.e. exactly the enumeration this comment says to never rely on. DPM
# provenance below is recorded box-wide only, not per-device.)
if [ "$FORCE_DPM" = 1 ] && command -v rocm-smi >/dev/null 2>&1; then
    parsed=$(rocm-smi --showperflevel 2>/dev/null \
        | grep -oE 'Performance Level:.*' | head -1 | awk -F': ' '{print $2}')
    [ -n "$parsed" ] && PRIOR_PERFLEVEL="$parsed"
    # NEVER force `high`: on RDNA3/4 perf_level=high PINS a MID sclk state that
    # UNDER-clocks the GPU (~14% below auto's boost — measured 2026-07-03: R9700
    # high=2350MHz/116tok-s vs auto=2838-3260MHz/132tok-s). `auto` is the
    # representative regime the user actually gets. See feedback_rdna4_perf_level.
    if try_rocm_smi_set auto; then
        DPM_FORCED=1
        log "DPM: set power_dpm_force_performance_level=auto box-wide (was: $PRIOR_PERFLEVEL) — auto boosts higher than high on RDNA3/4"
    else
        log "WARN: rocm-smi --setperflevel auto failed (no permission?) — continuing on the prior DPM state"
    fi
else
    log "DPM force-high skipped (--no-force-dpm or rocm-smi unavailable)"
fi

OUT_DIR="${OUT_DIR:-benchmarks/results/$(date +%Y%m%d-%H%M%S)-chip-profile-sweep}"
mkdir -p "$OUT_DIR"
log "output dir: $OUT_DIR"

# Box-wide DPM provenance (matches the `provenance.dpm_state` field consumed
# by ChipProfile / tests/chip-profiles/<arch>.json — see the "INSTRUMENT
# TODO: ... record dpm_state/mclk in provenance" note left in gfx1201.json).
# Deliberately box-wide, not per-device: see the rocm-smi-index-mismatch
# comment above the DPM force-high block for why a per-card mclk/dpm_state
# reading is NOT attempted here.
if [ "$DPM_FORCED" = 1 ]; then
    DPM_STATE_NOTE="auto (power_dpm_force_performance_level=auto box-wide; was: $PRIOR_PERFLEVEL) — auto is the representative regime; high under-clocks RDNA3/4"
elif [ "$FORCE_DPM" = 1 ]; then
    DPM_STATE_NOTE="set-auto FAILED (no permission?) — running on the prior DPM state"
else
    DPM_STATE_NOTE="force-high skipped (--no-force-dpm)"
fi

SUMMARY_TSV="$OUT_DIR/summary.tsv"
SUMMARY_MD="$OUT_DIR/summary.md"
{
    printf '# dpm_state_boxwide\t%s\n' "$DPM_STATE_NOTE"
    printf 'device\tarch\tmetric\tcold\twarm\tfolded\tanomaly\n'
} > "$SUMMARY_TSV"
{
    echo "# chip-profile-sweep — $(date -Is 2>/dev/null || date)"
    echo
    echo "Folded (warm, sanity-checked against cold) chip-profile measurements."
    echo "Copy the warm figures below into \`tests/chip-profiles/<arch>.json\`"
    echo "per \`crates/rdna-compute/src/chip_profile.rs\` (Task 4) — never the cold ones."
    echo
    echo "**DPM state (box-wide):** $DPM_STATE_NOTE"
    echo
    echo "| device | arch | metric | cold | warm | folded | anomaly |"
    echo "|---|---|---|---|---|---|---|"
} > "$SUMMARY_MD"

FAILED_ROWS=0

# fold_row BETTER_DIR COLD WARM
# BETTER_DIR: "lower" (latency, smaller ns is better) or "higher"
# (bandwidth, bigger GB/s is better). Folds to WARM always — this function
# only classifies whether that fold is trustworthy: a warm pass that fails
# to beat cold by more than $ANOMALY_PCT is flagged (force-high likely
# didn't take, or a thermal/contention artifact hit the warm pass).
fold_row() {
    local better_dir="$1" cold="$2" warm="$3"
    if [ -z "$warm" ]; then
        printf 'MISSING_WARM_VALUE'
        return
    fi
    if [ -z "$cold" ]; then
        printf 'ok(no-cold-baseline)'
        return
    fi
    # pct = % improvement of warm over cold, positive == warm is better.
    # "lower" (latency): improvement is warm < cold -> d = (cold - warm) / cold.
    # "higher" (bandwidth): improvement is warm > cold -> d = (warm - cold) / cold.
    local pct
    pct=$(awk -v c="$cold" -v w="$warm" -v dir="$better_dir" 'BEGIN{
        if (c == 0) { print "0.0"; exit }
        if (dir == "lower") { d = (c - w) / c * 100.0 } else { d = (w - c) / c * 100.0 }
        printf "%.1f", d
    }')
    if awk -v p="$pct" -v t="$ANOMALY_PCT" 'BEGIN{exit !(p > t)}'; then
        printf 'ok(warm %s%% better than cold)' "$pct"
    else
        printf 'WARM_DID_NOT_IMPROVE(%s%% vs cold, threshold %s%%)' "$pct" "$ANOMALY_PCT"
    fi
}

run_pass() {
    # run_pass BIN DEV LOGFILE — runs BIN with HIP_VISIBLE_DEVICES=DEV,
    # capturing combined stdout+stderr to LOGFILE. Returns BIN's exit status.
    local bin="$1" dev="$2" logf="$3"
    HIP_VISIBLE_DEVICES="$dev" "$bin" > "$logf" 2>&1
}

extract() {
    # extract KEY LOGFILE — value after "KEY: " on the first matching line
    # (matches the `println!("key: {value}")` convention shared by
    # pointer_chase_latency.rs and peak_bw_probe.rs).
    grep -m1 "^$1:" "$2" 2>/dev/null | sed -E "s/^$1:[[:space:]]*//"
}

sweep_metric() {
    # sweep_metric BIN DEV METRIC_KEY BETTER_DIR COLD_LOG WARM_LOG LABEL
    local bin="$1" dev="$2" key="$3" dir="$4" cold_log="$5" warm_log="$6" label="$7"

    log "device $dev: $label cold pass"
    if ! run_pass "$bin" "$dev" "$cold_log"; then
        log "WARN: device $dev cold $label pass FAILED (see $cold_log)"
    fi

    log "device $dev: $label warm pass"
    if run_pass "$bin" "$dev" "$warm_log"; then
        local arch cold_val warm_val anomaly
        arch=$(extract arch "$warm_log")
        cold_val=$(extract "$key" "$cold_log")
        warm_val=$(extract "$key" "$warm_log")
        anomaly=$(fold_row "$dir" "$cold_val" "$warm_val")
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$dev" "$arch" "$key" "$cold_val" "$warm_val" "$warm_val" "$anomaly" >> "$SUMMARY_TSV"
        printf '| %s | %s | %s | %s | %s | **%s** | %s |\n' \
            "$dev" "$arch" "$key" "$cold_val" "$warm_val" "$warm_val" "$anomaly" >> "$SUMMARY_MD"
    else
        log "FAIL: device $dev warm $label pass FAILED (see $warm_log)"
        printf '%s\t?\t%s\tFAIL\tFAIL\tFAIL\tFAIL\n' "$dev" "$key" >> "$SUMMARY_TSV"
        printf '| %s | ? | %s | FAIL | FAIL | FAIL | FAIL |\n' "$dev" "$key" >> "$SUMMARY_MD"
        FAILED_ROWS=$((FAILED_ROWS + 1))
    fi
}

for dev in "${DEV_LIST[@]}"; do
    log "──────── device $dev ────────"

    if [ "$RUN_LATENCY" = 1 ]; then
        sweep_metric "$LATENCY_BIN" "$dev" "mem_latency_ns" "lower" \
            "$OUT_DIR/device${dev}.cold.latency.log" \
            "$OUT_DIR/device${dev}.warm.latency.log" \
            "latency"
    fi

    if [ "$RUN_BW" = 1 ]; then
        sweep_metric "$BW_BIN" "$dev" "peak_bw_gbps" "higher" \
            "$OUT_DIR/device${dev}.cold.bw.log" \
            "$OUT_DIR/device${dev}.warm.bw.log" \
            "peak-bw"
    fi
done

{
    echo
    echo "Raw per-pass logs: ${OUT_DIR}/device<N>.{cold,warm}.{latency,bw}.log"
} >> "$SUMMARY_MD"

log "summary: $SUMMARY_MD"
cat "$SUMMARY_MD"

if [ "$FAILED_ROWS" -gt 0 ]; then
    log "FAILED: $FAILED_ROWS device/pass rows did not produce a measurement"
    exit 1
fi
exit 0
