#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kevin Read
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Pre-compile all HIP kernels for target GPU architectures.
# Usage: ./scripts/compile-kernels.sh [arch1 arch2 ...]
#        ./scripts/compile-kernels.sh --print-rocm-resolution
# Default: gfx906 gfx1010 gfx1030 gfx1100 gfx1200 gfx1201
#
# Parallelism: jobs run in parallel via `xargs -P`. Default is $(nproc);
# override with `JOBS=4 ./scripts/compile-kernels.sh ...`.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$SCRIPT_DIR/kernels/src"
OUT_BASE="$SCRIPT_DIR/kernels/compiled"

# Default target architectures
if [ $# -gt 0 ]; then
    ARCHS=("$@")
else
    ARCHS=(gfx906 gfx1010 gfx1030 gfx1100 gfx1200 gfx1201)
fi

JOBS="${JOBS:-$(nproc)}"

ROCM_RESOLUTION="$(
    cargo run --quiet --locked --manifest-path "$SCRIPT_DIR/Cargo.toml" \
        -p hipfire-config --bin hipfire-rocm-resolve
)" || exit 1
SELECTED_ROCM_ROOT="$(printf '%s\n' "$ROCM_RESOLUTION" | sed -n 's/^ROCM_ROOT=//p')"
HIPCC_BIN="$(printf '%s\n' "$ROCM_RESOLUTION" | sed -n 's/^HIPCC=//p')"
if [ -z "$SELECTED_ROCM_ROOT" ] || [ -z "$HIPCC_BIN" ]; then
    echo "ERROR: internal ROCm resolver returned incomplete output." >&2
    exit 1
fi
export HIPCC_BIN SELECTED_ROCM_ROOT

echo "=== hipfire kernel compiler ==="
echo "hipcc: $HIPCC_BIN"
echo "ROCm root: $SELECTED_ROCM_ROOT"
if [ "${1:-}" = "--print-rocm-resolution" ]; then
    exit 0
fi
echo "Source: $SRC_DIR"
echo "Architectures: ${ARCHS[*]}"
echo "Parallel jobs: $JOBS"

# Pre-build the packaging-hash helper (reuses compiler.rs hash_parts with toolchain_id="").
# This is the SAME hashing path the runtime uses; KERNEL_CACHE_ABI or field-order
# changes cannot diverge between JIT and packaging because both call hash_parts.
if ! cargo build --quiet --locked --manifest-path "$SCRIPT_DIR/Cargo.toml" \
    -p rdna-compute --bin hipfire-kernel-hash 2>/dev/null; then
    echo "ERROR: failed to build hipfire-kernel-hash" >&2
    exit 1
fi
# Cargo places the binary under target/debug or target/release depending on profile
HASH_BIN_CANDIDATE="$SCRIPT_DIR/target/debug/hipfire-kernel-hash"
if [ ! -x "$HASH_BIN_CANDIDATE" ]; then
    HASH_BIN_CANDIDATE="$SCRIPT_DIR/target/release/hipfire-kernel-hash"
fi
if [ ! -x "$HASH_BIN_CANDIDATE" ]; then
    # Fallback: locate via cargo metadata
    HASH_BIN_CANDIDATE="$(cargo metadata --format-version 1 --no-deps 2>/dev/null | python3 -c 'import json,sys,os; d=json.load(sys.stdin); print(os.path.join(d["target_directory"],"debug","hipfire-kernel-hash"))' 2>/dev/null || echo "")"
fi
HASH_BIN="$HASH_BIN_CANDIDATE"
if [ ! -x "$HASH_BIN" ]; then
    echo "ERROR: hipfire-kernel-hash binary not found after build" >&2
    exit 1
fi
export HASH_BIN
echo "hash helper: $HASH_BIN"

# Variant-tag regex: matches .gfxNNNN.hip (chip, e.g. .gfx1201.hip) and
# .gfxNN.hip (family, e.g. .gfx12.hip). Files matching this are treated as
# overrides for their parent name, not as independent kernels.
VARIANT_TAG_RE='\.gfx[0-9]+\.hip$'

# ── Phase 1: enumerate jobs ──────────────────────────────────────────────
# Emit one job per line: <arch>|<name>|<src>|<out>
# (Skips and variant resolution applied here so the worker stays simple.)

JOB_FILE="$(mktemp)"
trap 'rm -f "$JOB_FILE"' EXIT

for arch in "${ARCHS[@]}"; do
    out_dir="$OUT_BASE/$arch"
    mkdir -p "$out_dir"
    arch_family="${arch:0:5}"

    for src in "$SRC_DIR"/*.hip; do
        base=$(basename "$src")

        # Skip variant-tagged files during the parent iteration; they get
        # picked up below via the override lookup.
        if [[ "$base" =~ $VARIANT_TAG_RE ]]; then
            continue
        fi

        name=$(basename "$src" .hip)

        # gfx906 (Vega 20 / GCN5) is wave64-native but predates the RDNA3/4
        # WMMA builtins and the dot8 instruction used by MQ8.
        if [ "$arch" = "gfx906" ]; then
            case "$name" in
                *_wmma*|gemv_mq8g256)
                    echo "  - $name SKIP (unsupported ISA on gfx906)"
                    continue
                    ;;
            esac
        fi

        # gfx906-specific kernels (sdot4 dp4a, etc.) only build on gfx906.
        if [ "$arch" != "gfx906" ]; then
            case "$name" in
                *_gfx906|*_gfx906_*|*_dp4a)
                    echo "  - $name SKIP (gfx906-only)"
                    continue
                    ;;
            esac
        fi

        # Broken kernels: missing TURBO_C3_512 / fwht_shfl_forward_512 (WIP hd512)
        case "$name" in
            attention_flash_fwht3_tile_hd512|attention_flash_fwht3_tile_hd512_batched)
                echo "  - $name SKIP (broken hd512, missing TURBO_C3_512)"
                continue
                ;;
        esac

        # Variant precedence:
        #   1. ${name}.${arch}.hip          (chip-specific, e.g. .gfx1100.)
        #   2. ${name}.${arch_family}.hip   (family, e.g. .gfx12.)
        #   3. ${name}.hip                  (default)
        chip_variant="$SRC_DIR/${name}.${arch}.hip"
        family_variant="$SRC_DIR/${name}.${arch_family}.hip"
        if [ -f "$chip_variant" ]; then
            src="$chip_variant"
        elif [ -f "$family_variant" ]; then
            src="$family_variant"
        fi

        out="$out_dir/${name}.hsaco"
        printf '%s|%s|%s|%s\n' "$arch" "$name" "$src" "$out" >> "$JOB_FILE"
    done
done

TOTAL=$(wc -l < "$JOB_FILE")
echo "=== Compiling $TOTAL jobs across $JOBS workers... ==="

# ── Phase 2: parallel dispatch ───────────────────────────────────────────
# Each worker compiles one (arch, kernel) and prints exactly one status
# line. xargs runs $JOBS workers concurrently. Failures are captured by
# emitting "FAIL <name>" so the post-pass can count them without relying
# on xargs' exit propagation (which only signals "≥1 failed").

worker() {
    local job="$1"
    local arch name src out
    IFS='|' read -r arch name src out <<< "$job"

    if ROCM_PATH="$SELECTED_ROCM_ROOT" "$HIPCC_BIN" \
        --genco --offload-arch="$arch" -O3 \
        --rocm-path="$SELECTED_ROCM_ROOT" --hip-path="$SELECTED_ROCM_ROOT" \
        -I "$SELECTED_ROCM_ROOT/include" -I "$SCRIPT_DIR/kernels/src" \
        -o "$out" "$src" 2>/dev/null; then
        local hash_out="${out%.hsaco}.hash"
        # Packaging hash: same source+arch+flags+ABI as runtime, but toolchain_id=""
        # (what a hipcc-free runtime computes). Reuses compiler.rs hash_parts.
        local hash_ok=0
        if [ -n "${HIPFIRE_HIPCC_EXTRA_FLAGS:-}" ]; then
            if "$HASH_BIN" --arch "$arch" --extra-flags "$HIPFIRE_HIPCC_EXTRA_FLAGS" --name "$name" "$src" > "$hash_out" 2>/dev/null; then
                hash_ok=1
            fi
        else
            if "$HASH_BIN" --arch "$arch" --name "$name" "$src" > "$hash_out" 2>/dev/null; then
                hash_ok=1
            fi
        fi
        if [ "$hash_ok" -eq 1 ]; then
            local size
            size=$(stat -c%s "$out" 2>/dev/null || stat -f%z "$out" 2>/dev/null)
            printf 'OK  %-8s %s (%d KB)\n' "$arch" "$name" "$(( size / 1024 ))"
        else
            rm -f "$out" "$hash_out"
            printf 'FAIL %-8s %s (hash)\n' "$arch" "$name"
        fi
    else
        rm -f "$out"
        printf 'FAIL %-8s %s\n' "$arch" "$name"
    fi
}
export -f worker
export SCRIPT_DIR

# `xargs -P $JOBS -I {}` spawns up to $JOBS workers, one job per line.
# The status output is captured to a temp so we can count failures.
RESULT_FILE="$(mktemp)"
trap 'rm -f "$JOB_FILE" "$RESULT_FILE"' EXIT

xargs -a "$JOB_FILE" -P "$JOBS" -I {} bash -c 'worker "$@"' _ {} \
    | tee "$RESULT_FILE"

FAILED=$(grep -c '^FAIL ' "$RESULT_FILE" || true)
COMPILED=$(grep -c '^OK ' "$RESULT_FILE" || true)

echo ""
echo "=== Done: $COMPILED/$TOTAL compiled, $FAILED failed ==="
[ "$FAILED" -eq 0 ] || exit 1
