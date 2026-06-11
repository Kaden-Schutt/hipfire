#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kevin Read
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Pre-compile all HIP kernels for target GPU architectures.
# Usage: ./scripts/compile-kernels.sh [arch1 arch2 ...]
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

echo "=== hipfire kernel compiler ==="
echo "Source: $SRC_DIR"
echo "Architectures: ${ARCHS[*]}"
echo "Parallel jobs: $JOBS"

# Variant-tag regex: matches .gfxNNNN.hip, .gfxNN.hip, and .gfxNNNN_xxx.hip
# (e.g. .gfx11_dgpu.hip) at the end of a filename. Files matching this in the
# root are treated as arch overrides, not independent kernels.
VARIANT_TAG_RE='\.gfx[0-9]+(_[a-z_]+)?\.hip$'

# ── Phase 1: enumerate jobs ──────────────────────────────────────────────
# Emit one job per line: <arch>|<name>|<src>|<out>
# (Skips and variant resolution applied here so the worker stays simple.)
#
# Arch-specific kernels live in kernels/src/$arch/ subdirs. Variant precedence
# for a root base kernel (highest to lowest):
#   1. $arch_dir/${name}.${arch}.hip          chip-tagged in chip dir
#   2. $arch_dir/${name}.hip                  clean name in chip dir
#   3. $family_dir/${name}.${arch_family}.hip family-tagged in family dir
#   4. $family_dir/${name}.hip                clean name in family dir
#   5. root/${name}.${arch}.hip               chip tag in root (backward compat)
#   6. root/${name}.${arch_family}.hip        family tag in root (backward compat)
#   7. root/${name}.hip                       generic fallback
#
# Phase 1b picks up kernels that live only in an arch subdir (no root .hip).

JOB_FILE="$(mktemp)"
trap 'rm -f "$JOB_FILE"' EXIT

for arch in "${ARCHS[@]}"; do
    out_dir="$OUT_BASE/$arch"
    mkdir -p "$out_dir"
    arch_family="${arch:0:5}"
    arch_dir="$SRC_DIR/$arch"
    family_dir="$SRC_DIR/$arch_family"

    # ── Phase 1a: root base kernels ──────────────────────────────────────
    for src in "$SRC_DIR"/*.hip; do
        base=$(basename "$src")

        # Skip variant-tagged files in root; they get picked up via the
        # override lookup below or have moved to an arch subdir.
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

        # Override lookup: arch subdir first, then root tags (backward compat).
        if   [ -f "$arch_dir/${name}.${arch}.hip" ];          then src="$arch_dir/${name}.${arch}.hip"
        elif [ -f "$arch_dir/${name}.hip" ];                  then src="$arch_dir/${name}.hip"
        elif [ -f "$family_dir/${name}.${arch_family}.hip" ]; then src="$family_dir/${name}.${arch_family}.hip"
        elif [ -f "$family_dir/${name}.hip" ];                then src="$family_dir/${name}.hip"
        elif [ -f "$SRC_DIR/${name}.${arch}.hip" ];           then src="$SRC_DIR/${name}.${arch}.hip"
        elif [ -f "$SRC_DIR/${name}.${arch_family}.hip" ];    then src="$SRC_DIR/${name}.${arch_family}.hip"
        fi

        out="$out_dir/${name}.hsaco"
        printf '%s|%s|%s|%s\n' "$arch" "$name" "$src" "$out" >> "$JOB_FILE"
    done

    # ── Phase 1b: arch-subdir-only kernels ───────────────────────────────
    # Kernels that live exclusively in the arch or family subdir (no root .hip
    # counterpart). Chip dir is scanned first; family dir skips names already
    # added from chip dir to avoid duplicate jobs.
    phase1b_seen=""
    processed_subdirs=""
    for subdir in "$arch_dir" "$family_dir"; do
        [ -d "$subdir" ] || continue
        # Skip if we already scanned this exact dir (arch == arch_family edge case)
        case "$processed_subdirs" in *"|${subdir}|"*) continue ;; esac
        processed_subdirs="${processed_subdirs}|${subdir}|"

        for arch_src in "$subdir"/*.hip; do
            [ -f "$arch_src" ] || continue
            arch_base=$(basename "$arch_src")

            # Derive canonical kernel name: strip arch/family tag, else .hip
            arch_name="${arch_base%.${arch}.hip}"
            [ "$arch_name" != "$arch_base" ] || arch_name="${arch_base%.${arch_family}.hip}"
            [ "$arch_name" != "$arch_base" ] || arch_name="${arch_base%.hip}"

            # Skip if a root base exists (already handled by Phase 1a)
            [ -f "$SRC_DIR/${arch_name}.hip" ] && continue
            # Skip if already emitted from chip dir in this Phase 1b pass
            case "$phase1b_seen" in *"|${arch_name}|"*) continue ;; esac

            if [ "$arch" = "gfx906" ]; then
                case "$arch_name" in
                    *_wmma*|gemv_mq8g256)
                        echo "  - $arch_name SKIP (unsupported ISA on gfx906)"
                        continue ;;
                esac
            fi
            if [ "$arch" != "gfx906" ]; then
                case "$arch_name" in
                    *_gfx906|*_gfx906_*|*_dp4a)
                        echo "  - $arch_name SKIP (gfx906-only)"
                        continue ;;
                esac
            fi

            phase1b_seen="${phase1b_seen}|${arch_name}|"
            out="$out_dir/${arch_name}.hsaco"
            printf '%s|%s|%s|%s\n' "$arch" "$arch_name" "$arch_src" "$out" >> "$JOB_FILE"
        done
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

    if hipcc --genco --offload-arch="$arch" -O3 -I "$SCRIPT_DIR/kernels/src" \
        -o "$out" "$src" 2>/dev/null; then
        local size
        size=$(stat -c%s "$out" 2>/dev/null || stat -f%z "$out" 2>/dev/null)
        printf 'OK  %-8s %s (%d KB)\n' "$arch" "$name" "$(( size / 1024 ))"
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
