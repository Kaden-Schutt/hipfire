#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# speed-gate.sh — MQ4 speed regression wrapper for hipfire-eval.
#
# The benchmark and baseline comparison live in `hipfire-eval --battery speed`.
# Baselines are selected from benchmarks/perf-baselines/<gfx>-*.json and the
# wrapper fails if any speed row falls below its floor.

set -u
cd "$(dirname "$0")/.."

MODELS_DIR="${HIPFIRE_MODELS_DIR:-${HIPFIRE_DIR:-$HOME/.hipfire}/models}"
BASELINE_ROOT="${HIPFIRE_PERF_BASELINE_DIR:-benchmarks/perf-baselines}"
HIPFIRE_GPULOCK_BIN="${HIPFIRE_BIN:-$(command -v hipfire 2>/dev/null || echo ./target/release/hipfire)}"
GPULOCK_HELD=0
FAST=0
VERBOSE=0
MODEL_TMP_DIR=""
SPEED_GATE_ATTEMPTS="${HIPFIRE_SPEED_GATE_ATTEMPTS:-2}"

cleanup() {
    if [ "$GPULOCK_HELD" = 1 ]; then
        "$HIPFIRE_GPULOCK_BIN" gpu-lock release 2>/dev/null || true
    fi
    [ -n "$MODEL_TMP_DIR" ] && rm -rf "$MODEL_TMP_DIR"
}

trap cleanup EXIT

while [ $# -gt 0 ]; do
    case "$1" in
        --fast) FAST=1 ;;
        --verbose | -v) VERBOSE=1 ;;
        --update | --update-baselines)
            echo "speed-gate: baseline capture now belongs in hipfire-eval reports; curate benchmarks/perf-baselines/*.json from the report artifact" >&2
            exit 2
            ;;
        --tolerance)
            echo "speed-gate: tolerance is stored in benchmarks/perf-baselines/*.json" >&2
            shift
            ;;
        -h | --help)
            sed -n '7,18p' "$0"
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
    shift
done

color() {
    if [ -t 1 ]; then
        case "$1" in
            green) printf '\033[32m%s\033[0m' "$2" ;;
            red) printf '\033[31m%s\033[0m' "$2" ;;
            yellow) printf '\033[33m%s\033[0m' "$2" ;;
            bold) printf '\033[1m%s\033[0m' "$2" ;;
            *) printf '%s' "$2" ;;
        esac
    else
        printf '%s' "$2"
    fi
}

resolve_eval_bin() {
    if [ -n "${HIPFIRE_EVAL_BIN:-}" ] && [ -x "$HIPFIRE_EVAL_BIN" ]; then
        printf '%s\n' "$HIPFIRE_EVAL_BIN"
        return 0
    fi
    for cand in ./target/debug/hipfire-eval ./target/release/hipfire-eval; do
        [ -x "$cand" ] && {
            printf '%s\n' "$cand"
            return 0
        }
    done
    return 1
}

find_model_file() {
    local name="$1"
    local dir
    IFS=: read -r -a model_dirs <<<"$MODELS_DIR"
    for dir in "${model_dirs[@]}"; do
        [ -f "$dir/$name" ] && {
            printf '%s\n' "$dir/$name"
            return 0
        }
    done
    # Older staged models used a dotted quant token. Keep the speed gate
    # usable for local legacy artifacts while preserving canonical output names.
    case "$name" in
        *-mq[1-8].hfq)
            local quant prefix legacy
            quant="${name##*-}"
            prefix="${name%-"$quant"}"
            prefix="${prefix%-}"
            legacy="${prefix}.${quant}"
            for dir in "${model_dirs[@]}"; do
                if [ -f "$dir/$legacy" ]; then
                    if [ -z "$MODEL_TMP_DIR" ]; then
                        MODEL_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hipfire-speed-gate-models.XXXXXX")" || return 1
                    fi
                    ln -sf "$dir/$legacy" "$MODEL_TMP_DIR/$name"
                    printf '%s\n' "$MODEL_TMP_DIR/$name"
                    return 0
                fi
            done
            ;;
    esac
    return 1
}

detect_arch() {
    if [ -n "${HIPFIRE_BASELINE_ARCH:-}" ]; then
        printf '%s\n' "$HIPFIRE_BASELINE_ARCH"
        return 0
    fi
    local probe arch node_props ver
    for probe in amdgpu-arch offload-arch /opt/rocm/bin/amdgpu-arch /opt/rocm/bin/offload-arch /opt/rocm/llvm/bin/amdgpu-arch; do
        if command -v "$probe" >/dev/null 2>&1 || [ -x "$probe" ]; then
            arch="$("$probe" 2>/dev/null | head -1)"
            [ -n "$arch" ] && {
                printf '%s\n' "$arch"
                return 0
            }
        fi
    done
    for node_props in /sys/class/kfd/kfd/topology/nodes/*/properties; do
        [ -f "$node_props" ] || continue
        ver=$(awk '/gfx_target_version/ {print $2; exit}' "$node_props" 2>/dev/null || true)
        case "$ver" in
            90006)
                printf '%s\n' gfx906
                return 0
                ;;
            90008)
                printf '%s\n' gfx908
                return 0
                ;;
            100100)
                printf '%s\n' gfx1010
                return 0
                ;;
            100300 | 100302)
                printf '%s\n' gfx1030
                return 0
                ;;
            110000 | 110001)
                printf '%s\n' gfx1100
                return 0
                ;;
            110501)
                printf '%s\n' gfx1151
                return 0
                ;;
            120000)
                printf '%s\n' gfx1200
                return 0
                ;;
            120001)
                printf '%s\n' gfx1201
                return 0
                ;;
        esac
    done
    return 1
}

select_baseline() {
    [ -n "${HIPFIRE_PERF_BASELINE:-}" ] && {
        printf '%s\n' "$HIPFIRE_PERF_BASELINE"
        return 0
    }
    local arch="$1"
    local matches=()
    shopt -s nullglob
    matches=("$BASELINE_ROOT/${arch}-"*.json)
    shopt -u nullglob
    if [ "${#matches[@]}" -eq 1 ]; then
        printf '%s\n' "${matches[0]}"
        return 0
    fi
    if [ "${#matches[@]}" -eq 0 ]; then
        echo "speed-gate: no perf baseline in $BASELINE_ROOT for $arch" >&2
    else
        echo "speed-gate: multiple perf baselines for $arch; set HIPFIRE_PERF_BASELINE" >&2
    fi
    return 1
}

if ! EVAL_BIN="$(resolve_eval_bin)"; then
    echo "speed-gate: building hipfire-eval..."
    cargo build -p hipfire-eval >/dev/null || exit 2
    EVAL_BIN="$(resolve_eval_bin)" || {
        echo "speed-gate: hipfire-eval build did not produce a binary" >&2
        exit 2
    }
fi

if [ ! -x ./target/release/examples/bench_qwen35_speed ]; then
    echo "speed-gate: building bench_qwen35_speed..."
    cargo build --release -p hipfire-runtime --example bench_qwen35_speed --features deltanet >/dev/null || exit 2
fi

ARCH="$(detect_arch)" || {
    echo "speed-gate: cannot detect GPU arch; set HIPFIRE_BASELINE_ARCH" >&2
    exit 2
}
BASELINE="$(select_baseline "$ARCH")" || exit 2

if [ "$FAST" -eq 1 ]; then
    SIZES=("4b")
else
    SIZES=("0.8b" "4b" "9b" "27b")
fi

if [ -x "$HIPFIRE_GPULOCK_BIN" ] || command -v "$HIPFIRE_GPULOCK_BIN" >/dev/null 2>&1; then
    "$HIPFIRE_GPULOCK_BIN" gpu-lock acquire "speed-gate" --watch-pid "$$" || {
        color red "could not acquire GPU lock"
        echo
        exit 2
    }
    GPULOCK_HELD=1
fi

color bold "=== MQ4 Speed Gate via hipfire-eval ==="
echo
echo "baseline: $BASELINE"
echo

pass=0
fail=0
skip=0
for size in "${SIZES[@]}"; do
    model="$(find_model_file "qwen3.5-${size}-mq4.hfq" || true)"
    if [ -z "$model" ]; then
        printf "  %-5s " "$size"
        color yellow "SKIP"
        echo " (model not present: qwen3.5-${size}-mq4.hfq)"
        skip=$((skip + 1))
        continue
    fi
    attempt=1
    model_ok=0
    out=""
    while [ "$attempt" -le "$SPEED_GATE_ATTEMPTS" ]; do
        out="${HIPFIRE_SPEED_GATE_OUT:-/tmp/hipfire-speed-gate-${size}-$(date +%Y%m%d-%H%M%S)-try${attempt}}"
        if ! HIPFIRE_REQUIRE_PERF_BASELINE=1 HIPFIRE_PERF_BASELINE="$BASELINE" \
            "$EVAL_BIN" --model "$model" --battery speed --executor examples \
            --max-tokens 50 --no-cache --out "$out" >/tmp/hipfire-speed-gate.stdout 2>/tmp/hipfire-speed-gate.stderr; then
            [ "$VERBOSE" -eq 1 ] && cat /tmp/hipfire-speed-gate.stderr
        elif python3 - "$out/results.jsonl" "$VERBOSE" <<'PY'; then
import json, sys
path = sys.argv[1]
verbose = sys.argv[2] == "1"
rows = [json.loads(line) for line in open(path) if line.strip()]
failed = [r for r in rows if r.get("status") == "fail"]
for row in rows:
    m = row.get("metrics", {})
    case = row.get("case_id")
    status = row.get("status")
    pre = m.get("prefill_tok_s")
    gen = m.get("gen_tok_s")
    bpre = m.get("baseline_prefill_tok_s")
    bgen = m.get("baseline_gen_tok_s")
    print(f"    {case:28s} {status:4s} prefill={pre} baseline={bpre} gen={gen} baseline_gen={bgen}")
    if verbose and row.get("reason"):
        print(f"      reason: {row['reason']}")
sys.exit(1 if failed else 0)
PY
            model_ok=1
            break
        fi
        if [ "$attempt" -lt "$SPEED_GATE_ATTEMPTS" ]; then
            echo "    retrying ${size} speed sample (attempt ${attempt}/${SPEED_GATE_ATTEMPTS})"
        fi
        attempt=$((attempt + 1))
    done
    if [ "$model_ok" -eq 1 ]; then
        printf "  %-5s " "$size"
        color green "OK"
        echo " (report: $out)"
        pass=$((pass + 1))
    else
        printf "  %-5s " "$size"
        color red "FAIL"
        echo " (report: $out)"
        fail=$((fail + 1))
    fi
done

echo
echo "summary: pass=$pass fail=$fail skip=$skip"
[ "$fail" -eq 0 ] || exit 1
exit 0
