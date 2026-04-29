#!/bin/bash
# Pre-compile all HIP kernels for target GPU architectures.
# Usage: ./scripts/compile-kernels.sh [arch1 arch2 ...]
# Default: gfx906 gfx1010 gfx1030 gfx1100 gfx1200
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$SCRIPT_DIR/kernels/src"
OUT_BASE="$SCRIPT_DIR/kernels/compiled"

# Default target architectures
if [ $# -gt 0 ]; then
    ARCHS=("$@")
else
    ARCHS=(gfx906 gfx1010 gfx1030 gfx1100 gfx1200 gfx1201)
fi

echo "=== hipfire kernel compiler ==="
echo "Source: $SRC_DIR"
echo "Architectures: ${ARCHS[*]}"

TOTAL=0
FAILED=0

# Variant-tag regex: matches .gfxNNNN. (chip, e.g. .gfx1201.) and .gfxNN.
# (family, e.g. .gfx12.). Files matching this are treated as overrides for
# their parent name, not as independent kernels.
VARIANT_TAG_RE='\.gfx[0-9]+\.hip$'

for arch in "${ARCHS[@]}"; do
    out_dir="$OUT_BASE/$arch"
    mkdir -p "$out_dir"
    echo ""
    echo "--- $arch ---"

    # Family tag: first 5 chars of arch ("gfx12" for gfx1201/gfx1200,
    # "gfx94" for gfx940/gfx942, etc.). Falls back gracefully for any
    # non-standard chip name (just won't match family variants).
    arch_family="${arch:0:5}"

    for src in "$SRC_DIR"/*.hip; do
        base=$(basename "$src")

        # Skip variant-tagged files during the parent iteration; they get
        # picked up below via the override lookup. This prevents the script
        # from trying to compile e.g. a gfx12-only kernel for gfx1100.
        if [[ "$base" =~ $VARIANT_TAG_RE ]]; then
            continue
        fi

        name=$(basename "$src" .hip)

        # gfx906 (Vega 20 / GCN5) is wave64-native but predates the RDNA3/4
        # WMMA builtins and the dot8 instruction used by MQ8. These kernels
        # are not dispatched on gfx906, so don't make packaging fail on blobs
        # this arch cannot legally compile.
        if [ "$arch" = "gfx906" ]; then
            case "$name" in
                *_wmma*|gemv_mq8g256)
                    echo "  - $name SKIP (unsupported ISA on gfx906)"
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
            echo "  [variant] $name ($arch chip-specific)"
        elif [ -f "$family_variant" ]; then
            src="$family_variant"
            echo "  [variant] $name ($arch_family family)"
        fi

        out="$out_dir/${name}.hsaco"
        TOTAL=$((TOTAL + 1))

        if hipcc --genco --offload-arch="$arch" -O3 -I "$SRC_DIR" \
            -o "$out" "$src" 2>/dev/null; then
            size=$(stat -c%s "$out" 2>/dev/null || stat -f%z "$out" 2>/dev/null)
            echo "  ✓ $name ($(( size / 1024 )) KB)"
        else
            echo "  ✗ $name FAILED"
            FAILED=$((FAILED + 1))
            rm -f "$out"
        fi
    done
done

echo ""
echo "=== Done: $((TOTAL - FAILED))/$TOTAL compiled, $FAILED failed ==="
[ $FAILED -eq 0 ] || exit 1
