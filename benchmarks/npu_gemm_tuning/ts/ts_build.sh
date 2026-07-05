#!/usr/bin/env bash
# Build the isolated tensor-buffer-stream reshuffle test xclbin.
# Usage: ts_build.sh [MT] [KCHUNK]   (defaults 2 2 => N=256)
# Produces ~/.hipfire/npu/ts_<MT>x<KCHUNK>/{final.xclbin,insts.bin}. Offline (aiecc);
# the ts_a_verify example only loads the bytes and dispatches.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MT="${1:-2}"; KCHUNK="${2:-2}"
N=$((MT * 4 * KCHUNK * 16))

: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="$PEANO/bin:$MA_ROOT/bin:$PATH"

OUT="$HOME/.hipfire/npu/ts_${MT}x${KCHUNK}"
rm -rf "$OUT"; mkdir -p "$OUT"

"$PEANO/bin/clang++" "$HERE/ts_a.cc" -c -o "$OUT/ts_a.o" -I"$MA_ROOT/include" \
  -std=c++20 -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -O2 -DNDEBUG --target=aie2p-none-unknown-elf -DMT="$MT" -DKCHUNK="$KCHUNK"
python3 "$HERE/ts_gen.py" "$N" > "$OUT/aie.mlir"

aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$OUT/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >/dev/null
echo "cached: $OUT/final.xclbin  $OUT/insts.bin  (N=$N MT=$MT KCHUNK=$KCHUNK)"
