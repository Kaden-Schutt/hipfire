#!/usr/bin/env bash
# R8 — matrix-unit throughput microbench (aie2/XDNA1). Builds r8_1core.mlir + r8_ubench.cc
# per SHAPE and reports ns/mmul, ns/MAC, and GMAC/s/core from npu_gemm_bench. Because the
# hot loop runs on one RESIDENT tile (no feed, no per-iter sync), this is the pure matrix
# rate — the number R5-R7 never isolated (they timed the virtual int4 op through streaming).
#
# Usage: ./r8_run.sh [REPEAT] [NACC] [ITERS] [SHAPE...]   (default 20000 4 100, all shapes)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
REPEAT="${1:-20000}"; NACC="${2:-4}"; ITERS="${3:-100}"
shift $(( $# < 3 ? $# : 3 )) || true
SHAPES=("$@"); [ ${#SHAPES[@]} -eq 0 ] && SHAPES=(0 1 2 3 4)
declare -A NAME=([0]="mmul<4,16,8> int4 (virtual)" [1]="mmul<2,8,8> int8 (native)" \
  [2]="mmul<4,8,4> int8 (native)" [3]="mmul<4,16,8> int8 (virtual)" [4]="mmul<8,8,8> int8")
declare -A MKN=([0]=512 [1]=128 [2]=128 [3]=512 [4]=512)

: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"; source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="$PEANO/bin:$MA_ROOT/bin:$PATH"
command -v xclbinutil >/dev/null 2>&1 || export PATH="/opt/xilinx/xrt/bin:$PATH"
for B in "$HOME/.cache/hipfire-npu-deps/lib" "$HOME/.cache/hipfire-npu-deps/extract/usr/lib/x86_64-linux-gnu"; do
  [ -e "$B/libboost_program_options.so.1.83.0" ] && export LD_LIBRARY_PATH="$B:${LD_LIBRARY_PATH:-}" && break
done
INC="$MA_ROOT/include"
WD="${R8_WD:-/tmp/r8_ubench}"; mkdir -p "$WD"

echo "R8 matrix-unit microbench: REPEAT=$REPEAT NACC=$NACC iters=$ITERS  (one resident tile, no feed)"
printf '%-30s %-11s %-11s %-11s %s\n' SHAPE ns/mmul ns/MAC GMAC/s/core note
for S in "${SHAPES[@]}"; do
  dir="$WD/s$S"; rm -rf "$dir"; mkdir -p "$dir"
  if ! "$PEANO/bin/clang++" "$HERE/r8_ubench.cc" -c -o "$dir/r8.o" -I"$INC" \
      -std=c++20 -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
      -O2 -DNDEBUG --target=aie2-none-unknown-elf -DSHAPE=$S -DREPEAT=$REPEAT -DNACC=$NACC \
      >"$dir/cc.log" 2>&1; then
    printf '%-30s COMPILE FAILED (%s)\n' "${NAME[$S]}" "$dir/cc.log"; continue
  fi
  cp "$HERE/r8_1core.mlir" "$dir/aie.mlir"
  if ! aiecc "$dir/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
      --aie-generate-npu-insts --npu-insts-name="$dir/insts.bin" \
      --aie-generate-xclbin --xclbin-name="$dir/final.xclbin" --tmpdir="$dir" \
      >"$dir/aiecc.log" 2>&1; then
    printf '%-30s AIECC FAILED (%s)\n' "${NAME[$S]}" "$dir/aiecc.log"; continue
  fi
  mmuls=$(( NACC * REPEAT )); mkn=${MKN[$S]}; macs=$(( mmuls * mkn ))
  out=$(cd "$REPO" && cargo run -q -p hipfire-xdna --example npu_gemm_bench -- \
        "$dir" 64 128 256 "$macs" "$ITERS" 2>&1) || {
    printf '%-30s DISPATCH FAILED: %s\n' "${NAME[$S]}" "$(echo "$out"|tail -1)"; continue; }
  per=$(echo "$out" | sed -n 's/.*per_dispatch=\([0-9.]*\)us.*/\1/p')
  # ns/mmul = per_us*1000/mmuls ; ns/MAC = per_us*1000/macs ; GMAC/s/core = macs/per_us/1000
  read nsmmul nsmac gmac <<<"$(awk -v p="$per" -v mm="$mmuls" -v ma="$macs" \
      'BEGIN{printf "%.3f %.4f %.1f", p*1000/mm, p*1000/ma, ma/p/1000}')"
  printf '%-30s %-11s %-11s %-11s %s\n' "${NAME[$S]}" "$nsmmul" "$nsmac" "$gmac" "MKN=$mkn"
done
