#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# End-to-end: does swapping the hand-rolled E4M3 decode for the gfx12 native
# OCP conversion move a REAL MFP4-G32-E8 GEMV, or does the 46% decode win
# vanish once amortized over the weights?
#
# Takes the shipped kernels/src/gemv_mfp4g32_e8.hip verbatim and produces two
# variants that differ ONLY in the body of cvt_e4m3_scale_to_f32:
#   A = shipped hand-rolled idiom
#   B = __hip_fp8_e4m3 native conversion (radiowave Fp8Format::E4M3Ocp)
# Both keep the identical kernel name, launch shape and wire layout, so the
# comparison isolates the decode.
set -euo pipefail

ARCH="${1:-gfx1201}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../../../kernels/src/gemv_mfp4g32_e8.hip"
OUT="$HERE/build"
mkdir -p "$OUT"

case "$ARCH" in
  gfx1200|gfx1201) ;;
  *) echo "native OCP FP8 requires gfx1200/gfx1201 (got $ARCH); variant B would" \
          "silently fall back to the hand-rolled path and measure nothing." >&2
     exit 2 ;;
esac

cp "$SRC" "$OUT/variant_a.hip"

# Variant B: replace ONLY the function body. The signature, name and all call
# sites are untouched, so nothing else in the kernel shifts.
python3 - "$SRC" "$OUT/variant_b.hip" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
native = '''__device__ __forceinline__ float cvt_e4m3_scale_to_f32(unsigned char b) {
    // radiowave Fp8Format::E4M3Ocp — native gfx12 OCP conversion.
    // Bit-identical to the hand-rolled idiom over codes 1..126, the only
    // range e4m3_encode_roundup can emit (verified by probe.hip: 0 mismatches
    // on the reachable domain, 129 on the full 256 — all at the NaN slot 0x7F
    // and the negative half, neither of which a SCALE can occupy).
    __hip_fp8_e4m3 v;
    v.__x = b;
    return (float)v;
}'''
pat = re.compile(
    r'__device__ __forceinline__ float cvt_e4m3_scale_to_f32\(unsigned char b\) \{.*?\n\}',
    re.S)
out, n = pat.subn(native, src, count=1)
if n != 1:
    sys.exit(f"expected exactly 1 decoder definition, replaced {n}")
out = out.replace('#include <hip/hip_runtime.h>',
                  '#include <hip/hip_runtime.h>\n#include <hip/hip_fp8.h>', 1)
open(sys.argv[2], 'w').write(out)
PY

echo "=== ISA: instruction counts per variant ($ARCH) ==="
for v in a b; do
  hipcc -O3 --offload-arch="$ARCH" -std=c++17 --cuda-device-only -S \
        "$OUT/variant_$v.hip" -o "$OUT/variant_$v.s" 2>/dev/null
  printf "  variant %s: VALU=%-5s vgpr=%-5s cvt_f32_fp8=%s\n" \
    "$v" \
    "$(grep -cE '^\s+v_' "$OUT/variant_$v.s")" \
    "$(grep -oE 'vgpr_count:\s+[0-9]+' "$OUT/variant_$v.s" | head -1 | grep -oE '[0-9]+')" \
    "$(grep -cE 'v_cvt_f32_fp8|cvt_pk_f32_fp8' "$OUT/variant_$v.s")"
done
echo
echo "Kernel sources: $OUT/variant_{a,b}.hip"
echo "Diff is confined to cvt_e4m3_scale_to_f32:"
diff <(sed -n '/cvt_e4m3_scale_to_f32/,/^}/p' "$OUT/variant_a.hip") \
     <(sed -n '/cvt_e4m3_scale_to_f32/,/^}/p' "$OUT/variant_b.hip") || true
