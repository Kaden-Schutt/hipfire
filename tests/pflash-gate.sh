#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# PFlash regression gate.
#
# Runs the 6 Phase 5 NIAH fixtures (8K, 16K, multi-16K, longcode,
# longprose, 32K) under both baseline and PFlash modes against
# qwen3.5-27b-mq3.hfq + qwen3.5-0.8b-mq4.hfq drafter, then asserts:
#   1. Verdict matches the recorded baseline (PASS stays PASS,
#      FAIL stays FAIL; flips are quality regressions).
#   2. Total wall-clock within ±10% of the recorded median.
#
# Baseline JSON: benchmarks/perf-baselines/<gfx>-<hardware-profile-hash>.json
# Default: auto-select by detected GFX arch.
#
# Exit codes:
#   0  every fixture matches verdict and stays inside ±10%
#   1  one or more fixtures regressed (verdict flip OR drift > 10%)
#   2  build / environment error
#
# Usage:
#   ./tests/pflash-gate.sh                # auto-select perf baseline
#   ./tests/pflash-gate.sh --baseline benchmarks/perf-baselines/<file>.json
#   ./tests/pflash-gate.sh --tolerance 15 # widen to ±15%
#
# Hooks into tests/coherence-gate.sh as a follow-up stage.

set -u
cd "$(dirname "$0")/.."

BASELINE="${HIPFIRE_PERF_BASELINE:-}"
BASELINE_ROOT="${HIPFIRE_PERF_BASELINE_DIR:-benchmarks/perf-baselines}"
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

gfx_target_version_to_arch() {
    case "$1" in
        900000) echo "gfx900" ;;
        906000) echo "gfx906" ;;
        908000) echo "gfx908" ;;
        1010000) echo "gfx1010" ;;
        1030000) echo "gfx1030" ;;
        1100000|1100001) echo "gfx1100" ;;
        1105001) echo "gfx1151" ;;
        1200000) echo "gfx1200" ;;
        1201000) echo "gfx1201" ;;
        *) return 1 ;;
    esac
}

detect_gfx_arch() {
    if [ -n "${HIPFIRE_BASELINE_ARCH:-}" ]; then
        printf '%s\n' "$HIPFIRE_BASELINE_ARCH"
        return 0
    fi
    if command -v hipfire-eval >/dev/null 2>&1; then
        arch="$(hipfire-eval --version 2>/dev/null | awk '/^arch / {print $2; exit}')"
        [ -n "$arch" ] && { printf '%s\n' "$arch"; return 0; }
    fi
    for exe in ./target/release/hipfire-eval ./target/debug/hipfire-eval; do
        if [ -x "$exe" ]; then
            arch="$("$exe" --version 2>/dev/null | awk '/^arch / {print $2; exit}')"
            [ -n "$arch" ] && { printf '%s\n' "$arch"; return 0; }
        fi
    done
    for props in /sys/class/kfd/kfd/topology/nodes/*/properties; do
        [ -r "$props" ] || continue
        raw="$(awk '$1 == "gfx_target_version" {print $2; exit}' "$props")"
        [ -n "$raw" ] || continue
        gfx_target_version_to_arch "$raw" && return 0
    done
    return 1
}

select_perf_baseline() {
    local arch matches count
    if [ -n "$BASELINE" ]; then
        printf '%s\n' "$BASELINE"
        return 0
    fi
    arch="$(detect_gfx_arch || true)"
    if [ -z "$arch" ]; then
        echo "pflash-gate: could not detect GFX arch; set HIPFIRE_BASELINE_ARCH or --baseline" >&2
        return 1
    fi
    # shellcheck disable=SC2206
    matches=( "$BASELINE_ROOT/$arch"-*.json )
    count=0
    for path in "${matches[@]}"; do
        [ -f "$path" ] || continue
        count=$((count + 1))
        BASELINE="$path"
    done
    if [ "$count" -eq 0 ]; then
        echo "pflash-gate: no perf baseline for $arch under $BASELINE_ROOT" >&2
        echo "             capture one with hipfire eval, then curate it as $BASELINE_ROOT/$arch-<host_profile_hash>.json" >&2
        return 1
    fi
    if [ "$count" -gt 1 ]; then
        echo "pflash-gate: multiple perf baselines for $arch; choose one with --baseline or HIPFIRE_PERF_BASELINE" >&2
        for path in "$BASELINE_ROOT/$arch"-*.json; do
            [ -f "$path" ] && echo "  $path" >&2
        done
        return 1
    fi
    printf '%s\n' "$BASELINE"
}

if ! BASELINE="$(select_perf_baseline)"; then
    if [ "${HIPFIRE_SKIP_MISSING_PERF_BASELINE:-0}" = "1" ]; then
        echo "pflash-gate: SKIPPED (no matching perf baseline)"
        exit 0
    fi
    exit 2
fi
if [ ! -f "$BASELINE" ]; then
    echo "pflash-gate: baseline not found: $BASELINE" >&2
    exit 2
fi

# Read baseline rows. We use python for robust JSON without adding deps.
if ! BASELINE_INFO_AND_ROWS=$(python3 -c '
import json, sys
b = json.load(open(sys.argv[1]))
schema = b.get("schema", "legacy.pflash_baseline.v0")
arch = b.get("arch") or b.get("_meta", {}).get("gpu", "unknown").split()[0]
profile = b.get("hardware_profile_hash") or b.get("_meta", {}).get("gpu", "legacy")
suite = b.get("baselines", {}).get("pflash_niah")
if suite is None:
    suite = b.get("fixtures")
if not isinstance(suite, list):
    print("__missing_pflash_suite__|{}|{}|{}".format(schema, arch, profile))
    raise SystemExit(0)
print(f"__baseline__|{schema}|{arch}|{profile}")
for r in suite:
    print("{}|{}|{}|{}|{}".format(r["label"], r["fixture"], r["mode"], r["total_ms"], r["verdict"]))
' "$BASELINE"); then
    echo "pflash-gate: failed to parse perf baseline: $BASELINE" >&2
    exit 2
fi

BASELINE_HEADER=$(printf '%s\n' "$BASELINE_INFO_AND_ROWS" | head -1)
ROWS_JSON=$(printf '%s\n' "$BASELINE_INFO_AND_ROWS" | sed '1d')
IFS='|' read -r _baseline_tag baseline_schema baseline_arch baseline_profile <<< "$BASELINE_HEADER"
if [ "$_baseline_tag" = "__missing_pflash_suite__" ]; then
    if [ "${HIPFIRE_SKIP_MISSING_PERF_BASELINE:-0}" = "1" ]; then
        echo "pflash-gate: SKIPPED (baseline has no baselines.pflash_niah rows)"
        exit 0
    fi
    echo "pflash-gate: baseline has no baselines.pflash_niah rows: $BASELINE" >&2
    exit 2
fi

current_arch="$(detect_gfx_arch || echo unknown)"
if [ "$current_arch" != "unknown" ] && [ "$baseline_arch" != "unknown" ] && [ "$current_arch" != "$baseline_arch" ]; then
    echo "pflash-gate: refusing cross-arch perf comparison: current=$current_arch baseline=$baseline_arch" >&2
    echo "             choose an explicit same-arch baseline with --baseline only if this is intentional" >&2
    exit 2
fi

EXE="./target/release/examples/pflash_niah_bench"

# Rebuild if the binary is missing OR any relevant source is newer than it.
# Without this freshness check, the gate would run against a stale binary
# whenever someone forgot to rebuild after editing pflash.rs / qwen35.rs /
# llama.rs / the bench itself / the score kernel and silently report a
# pass on outdated code. Mirror the pattern from tests/coherence-gate.sh.
#
# Coverage extends through the full PFlash kernel wiring chain so a change
# anywhere in the dispatch path forces a rebuild:
#   - engine logic (pflash.rs, qwen35.rs, llama.rs, hfq.rs, tokenizer.rs)
#   - bench harness (pflash_niah_bench.rs)
#   - hipfire-rdna layer:
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
        crates/hipfire-arch-qwen35/src/pflash.rs \
        crates/hipfire-arch-qwen35/src/qwen35.rs \
        crates/hipfire-runtime/src/llama.rs \
        crates/hipfire-runtime/src/hfq.rs \
        crates/hipfire-runtime/src/tokenizer.rs \
        crates/hipfire-runtime/examples/pflash_niah_bench.rs \
        crates/hipfire-rdna/src/lib.rs \
        crates/hipfire-rdna/src/kernels.rs \
        crates/hipfire-rdna/src/dispatch.rs \
        crates/hipfire-rdna/src/compiler.rs \
        crates/hipfire-rdna/src/pool.rs \
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
    if ! cargo build --release --features deltanet --example pflash_niah_bench -p hipfire-runtime >&2; then
        echo "pflash-gate: build failed" >&2
        exit 2
    fi
fi

TARGET="${HIPFIRE_PFLASH_TARGET:-$HOME/.hipfire/models/qwen3.5-27b-mq3.hfq}"
DRAFTER="${HIPFIRE_PFLASH_DRAFTER:-$HOME/.hipfire/models/qwen3.5-0.8b-mq4.hfq}"
if [ ! -f "$TARGET" ] || [ ! -f "$DRAFTER" ]; then
    echo "pflash-gate: target or drafter not present" >&2
    echo "  target  = $TARGET" >&2
    echo "  drafter = $DRAFTER" >&2
    exit 2
fi

HIPFIRE_GPULOCK_BIN="${HIPFIRE_BIN:-$(command -v hipfire 2>/dev/null || echo ./target/release/hipfire)}"
if { [ -x "$HIPFIRE_GPULOCK_BIN" ] || command -v "$HIPFIRE_GPULOCK_BIN" >/dev/null 2>&1; }; then
    # shellcheck disable=SC1090
    "$HIPFIRE_GPULOCK_BIN" gpu-lock acquire "pflash-gate" --watch-pid "$$" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap '"$HIPFIRE_GPULOCK_BIN" gpu-lock release 2>/dev/null || true' EXIT
fi

echo "pflash-gate: baseline=$BASELINE tolerance=±${TOLERANCE_PCT}%"
echo "pflash-gate: baseline_schema=$baseline_schema arch=$baseline_arch hardware_profile=$baseline_profile"
echo "pflash-gate: target=$(basename "$TARGET") drafter=$(basename "$DRAFTER")"
echo

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
