#!/usr/bin/env bash
# Offline: build + cache an R6 W4A8 GEMM xclbin for a tile config (MT NT KCHUNK), for
# the runtime to load via NpuGemm. The xclbin depends only on the TILE config, not the
# GEMM (K,N) — NpuGemm tiles arbitrary shapes over the fixed block — so a small set of
# configs covers a whole model. Python/aiecc stays OFFLINE here (AGENTS.md: no Python
# in the hot path; the inference binary only loads the cached bytes).
#
# Produces ~/.hipfire/npu/r6_<MT>x<NT>x<KCHUNK>/{final.xclbin,insts.bin}, the layout
# NpuGemm::load expects (single-core, one (MT*4)x(NT*16)x(KCHUNK*16) block/dispatch =
# r6_gen.py COLS=1 NB=1).
#
# Usage: r6_cache.sh [MT] [NT] [KCHUNK]   (defaults 16 4 16)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MT="${1:-16}"; NT="${2:-4}"; KCHUNK="${3:-16}"
[ "$NT" = 4 ] || { echo "NT must be 4 (r6_mac accumulator count)"; exit 1; }

: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="$PEANO/bin:$MA_ROOT/bin:$PATH"

OUT="$HOME/.hipfire/npu/r6_${MT}x${NT}x${KCHUNK}"
rm -rf "$OUT"; mkdir -p "$OUT"

# Tile buffer sizes: A = MT*KCHUNK tiles (MR*MK=64 int8), W = NT*KCHUNK tiles (128 B),
# C = MT*NT tiles (MR*MN=64 int32).
AW=$((MT * KCHUNK * 64)); WW=$((NT * KCHUNK * 128)); CW=$((MT * NT * 64))
"$PEANO/bin/clang++" "$HERE/r6_gemm.cc" -c -o "$OUT/r6_mac.o" -I"$MA_ROOT/include" \
  -std=c++20 -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -O2 -DNDEBUG --target=aie2p-none-unknown-elf -DMT="$MT" -DNT="$NT" -DKCHUNK="$KCHUNK"
python3 "$HERE/r6_gen.py" 1 1 "$AW" "$WW" "$CW" > "$OUT/aie.mlir"

aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
echo "cached: $OUT/final.xclbin  $OUT/insts.bin  (block ${MT}x4 x ${NT}x16 x ${KCHUNK}x16 => M=$((MT*4)) N=$((NT*16)) K=$((KCHUNK*16)))"
