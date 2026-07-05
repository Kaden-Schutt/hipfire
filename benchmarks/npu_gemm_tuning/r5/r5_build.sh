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
MLIR="${1:?usage: r5_build.sh <mlir> <workdir> [KSLICE] [INNER]}"
W="${2:?workdir}"
KSLICE="${3:-16}"
INNER="${4:-0}"

# R5_ARCH selects the NPU: aie2p = Strix Halo/npu2 (default, the validated line);
# aie2 = XDNA1/Phoenix/npu1 (gfx1103 boxes). The generator that produced <mlir> must
# be run with the SAME R5_ARCH (device + tile sizes must match the kernel shape).
: "${R5_ARCH:=aie2p}"
case "$R5_ARCH" in
  aie2)  TGT="aie2-none-unknown-elf";  ARCHDEF=(-DNPU_AIE2 -DMMUL_N=8) ;;   # int8xint4 = <4,16,8>
  aie2p) TGT="aie2p-none-unknown-elf"; ARCHDEF=(-DMMUL_N=16) ;;             # int8xint4 = <4,16,16>
  *) echo "unknown R5_ARCH=$R5_ARCH (want aie2 or aie2p)" >&2; exit 2 ;;
esac

: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="$PEANO/bin:$MA_ROOT/bin:$PATH"
INC="$MA_ROOT/include"

# xclbin packaging shells out to xclbinutil (XRT). On XDNA1 dev boxes it may not be
# on PATH, and it links libboost 1.83 which the distro doesn't ship — a user-space
# copy lives under the hipfire-npu-deps cache. Both guards are no-ops where absent
# (e.g. a box that already has xclbinutil + boost on the default paths).
[ -x /opt/xilinx/xrt/bin/xclbinutil ] && command -v xclbinutil >/dev/null 2>&1 || export PATH="/opt/xilinx/xrt/bin:$PATH"
for BOOSTDIR in "$HOME/.cache/hipfire-npu-deps/lib" "$HOME/.cache/hipfire-npu-deps/extract/usr/lib/x86_64-linux-gnu"; do
  [ -e "$BOOSTDIR/libboost_program_options.so.1.83.0" ] && export LD_LIBRARY_PATH="$BOOSTDIR:${LD_LIBRARY_PATH:-}" && break
done
CF=(-std=c++20 -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
    -O2 -DNDEBUG --target="$TGT" "${ARCHDEF[@]}" "-DKSLICE=$KSLICE")

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
