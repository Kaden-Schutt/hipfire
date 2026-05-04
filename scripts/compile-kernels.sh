#!/bin/bash
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

# Variant-tag regex: matches .gfxNNNN. (chip, e.g. .gfx1201.) and .gfxNN.
# (family, e.g. .gfx12.). Files matching this are treated as overrides for
# their parent name, not as independent kernels.
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
                *_gfx906)
                    echo "  - $name SKIP (gfx906-only)"
                    continue
                    ;;
            esac
        fi

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
