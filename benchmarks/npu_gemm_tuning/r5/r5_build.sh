#!/usr/bin/env bash
# Build an R5 cascade design from a hand-written .mlir + the r5_cascade.cc roles,
# entirely through the low-level flow (no IRON). Compiles the head/middle/tail
# kernel objects, drops them next to the .mlir in a work dir, and runs aiecc to
# produce final.xclbin + insts.bin — ready to dispatch via hipfire's NpuKernel
# (e.g. `cargo run -p hipfire-xdna --example run_smoke -- <workdir> <asz> <wsz> <csz>`).
#
# Usage: r5_build.sh <mlir-file> <workdir> [KSLICE]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLIR="${1:?usage: r5_build.sh <mlir> <workdir> [KSLICE]}"
W="${2:?workdir}"
KSLICE="${3:-16}"

: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="$PEANO/bin:$MA_ROOT/bin:$PATH"
INC="$MA_ROOT/include"
CF=(-std=c++20 -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
    -O2 -DNDEBUG --target=aie2p-none-unknown-elf "-DKSLICE=$KSLICE")

rm -rf "$W"; mkdir -p "$W"
# One object per cascade role (the .mlir's link_with picks which each core needs).
"$PEANO/bin/clang++" "$HERE/r5_cascade.cc" -c -o "$W/r5_head.o" -I"$INC" "${CF[@]}" -DROLE=0
"$PEANO/bin/clang++" "$HERE/r5_cascade.cc" -c -o "$W/r5_mid.o"  -I"$INC" "${CF[@]}" -DROLE=1
"$PEANO/bin/clang++" "$HERE/r5_cascade.cc" -c -o "$W/r5_tail.o" -I"$INC" "${CF[@]}" -DROLE=2
cp "$MLIR" "$W/aie.mlir"

aiecc "$W/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
  --aie-generate-npu-insts --npu-insts-name="$W/insts.bin" \
  --aie-generate-xclbin --xclbin-name="$W/final.xclbin" --tmpdir="$W"
echo "built: $W/final.xclbin  $W/insts.bin"
