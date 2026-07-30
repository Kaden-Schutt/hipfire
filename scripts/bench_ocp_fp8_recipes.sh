#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
#
# Reproducible gfx11 software-decode vs gfx12 native OCP FP8 WMMA probe.
# This measures recipe-level kernels, not model serving performance.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/crates/radiowave/bench/ocp_fp8_recipe_bench.hip"
ROCM_PATH="${ROCM_PATH:-/opt/rocm/core}"
if [[ ! -x "$ROCM_PATH/bin/hipcc" ]]; then
    ROCM_PATH=/opt/rocm
fi
HIPCC="${HIPCC:-$ROCM_PATH/bin/hipcc}"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -z "${ARCH:-}" ]]; then
    mapfile -t DETECTED_ARCHES < <(
        "$ROCM_PATH/bin/rocminfo" 2>/dev/null |
            awk '/^[[:space:]]*Name:[[:space:]]+gfx[0-9]+/{print $2}' |
            sort -u
    )
    if (( ${#DETECTED_ARCHES[@]} != 1 )); then
        echo "detected ${#DETECTED_ARCHES[@]} GPU architectures; set ARCH to match HIP_VISIBLE_DEVICES" >&2
        printf '  %s\n' "${DETECTED_ARCHES[@]}" >&2
        exit 2
    fi
    ARCH="${DETECTED_ARCHES[0]}"
fi
BUILD_DIR="${BUILD_DIR:-/tmp/hipfire-ocp-fp8-bench-${USER:-user}}"
BIN="$BUILD_DIR/ocp_fp8_recipe_bench-$ARCH"
TILES="${TILES:-262144}"
WARMUP="${WARMUP:-20}"
TRIALS="${TRIALS:-31}"
INNER="${INNER:-4}"
COOLDOWN_SECS="${COOLDOWN_SECS:-5}"
OUT="${OUT:-}"

if [[ ! "$ARCH" =~ ^gfx(11|12)[0-9][0-9]$ ]]; then
    echo "unsupported or undetected architecture: '$ARCH'" >&2
    exit 2
fi
if [[ ! -x "$HIPCC" ]]; then
    echo "hipcc not found: $HIPCC" >&2
    exit 2
fi

mkdir -p "$BUILD_DIR"
if [[ "$ARCH" =~ ^gfx11 ]]; then
    ARCH_DEFINE=-DRADIOWAVE_BENCH_GFX11=1
else
    ARCH_DEFINE=-DRADIOWAVE_BENCH_GFX12=1
fi
"$HIPCC" -O3 -std=c++17 "$ARCH_DEFINE" "--offload-arch=$ARCH" "$SOURCE" -o "$BIN"

run_all() {
    local first=1
    local mode
    local modes=(ocp-e4m3 ocp-e5m2 f16-e4m3 f16-e5m2)
    if [[ "$ARCH" =~ ^gfx11 ]]; then
        modes=(ocp-e4m3 ocp-e4m3-hip-cvt ocp-e5m2 f16-e4m3 f16-e5m2)
    fi
    if [[ -n "${MODES:-}" ]]; then
        read -r -a modes <<<"$MODES"
    fi
    for mode in "${modes[@]}"; do
        if (( first )); then
            "$BIN" "--mode=$mode" "--tiles=$TILES" "--warmup=$WARMUP" \
                "--trials=$TRIALS" "--inner=$INNER" "--expected-arch=$ARCH"
            first=0
        else
            sleep "$COOLDOWN_SECS"
            "$BIN" "--mode=$mode" "--tiles=$TILES" "--warmup=$WARMUP" \
                "--trials=$TRIALS" "--inner=$INNER" \
                "--expected-arch=$ARCH" --no-header
        fi
    done
}

if [[ -n "$OUT" ]]; then
    mkdir -p "$(dirname "$OUT")"
    run_all | tee "$OUT"
else
    run_all
fi
