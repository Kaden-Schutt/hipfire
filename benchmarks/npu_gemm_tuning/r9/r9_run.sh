#!/usr/bin/env bash
# R9 — load-reuse register-tiling sweep (aie2/XDNA1). Builds r9_ubench.cc + r9_1core.mlir
# per MTxNT output tile and reports ns/mmul + GMAC/s/core from npu_gemm_bench. Reuse =
# MT*NT/(MT+NT) macs-per-load; higher reuse should push the L1-load-bound rate toward
# R8's ~1 ns/mmul register-resident ceiling. Base op is int4 <4,16,8> (512 MAC/op).
#
# Usage: ./r9_run.sh [KT] [REPEAT] [ITERS] ["MTxNT" ...]
#   default: KT=16 REPEAT=20000 ITERS=80, tiles 1x1 2x2 2x4 4x4
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
KT="${1:-16}"; REPEAT="${2:-20000}"; ITERS="${3:-80}"
shift $(( $# < 3 ? $# : 3 )) || true
TILES=("$@"); [ ${#TILES[@]} -eq 0 ] && TILES=(1x1 2x2 2x4 4x4)

: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"; source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="$PEANO/bin:$MA_ROOT/bin:$PATH"
command -v xclbinutil >/dev/null 2>&1 || export PATH="/opt/xilinx/xrt/bin:$PATH"
for B in "$HOME/.cache/hipfire-npu-deps/lib" "$HOME/.cache/hipfire-npu-deps/extract/usr/lib/x86_64-linux-gnu"; do
  [ -e "$B/libboost_program_options.so.1.83.0" ] && export LD_LIBRARY_PATH="$B:${LD_LIBRARY_PATH:-}" && break
done
INC="$MA_ROOT/include"
WD="${R9_WD:-/tmp/r9_tiling}"; mkdir -p "$WD"

echo "R9 load-reuse sweep: KT=$KT REPEAT=$REPEAT iters=$ITERS  (int4 <4,16,8>, 512 MAC/op)"
printf '%-7s %-7s %-11s %-13s %s\n' tile reuse ns/mmul GMAC/s/core note
for T in "${TILES[@]}"; do
  MT="${T%x*}"; NT="${T#*x}"; dir="$WD/$T"; rm -rf "$dir"; mkdir -p "$dir"
  if ! "$PEANO/bin/clang++" "$HERE/r9_ubench.cc" -c -o "$dir/r9.o" -I"$INC" \
      -std=c++20 -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
      -O2 -DNDEBUG --target=aie2-none-unknown-elf -DMT=$MT -DNT=$NT -DKT=$KT -DREPEAT=$REPEAT \
      >"$dir/cc.log" 2>&1; then
    printf '%-7s COMPILE FAILED (%s)\n' "$T" "$dir/cc.log"; continue
  fi
  cp "$HERE/r9_1core.mlir" "$dir/aie.mlir"
  if ! aiecc "$dir/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
      --aie-generate-npu-insts --npu-insts-name="$dir/insts.bin" \
      --aie-generate-xclbin --xclbin-name="$dir/final.xclbin" --tmpdir="$dir" \
      >"$dir/aiecc.log" 2>&1; then
    printf '%-7s AIECC FAILED (%s)\n' "$T" "$dir/aiecc.log"; continue
  fi
  mmuls=$(( REPEAT * KT * MT * NT )); macs=$(( mmuls * 512 ))
  reuse=$(awk -v m=$MT -v n=$NT 'BEGIN{printf "%.2f", (m*n)/(m+n)}')
  out=$(cd "$REPO" && cargo run -q -p hipfire-xdna --example npu_gemm_bench -- \
        "$dir" 16384 16384 2048 "$macs" "$ITERS" 2>&1) || {
    printf '%-7s DISPATCH FAILED: %s\n' "$T" "$(echo "$out"|tail -1)"; continue; }
  per=$(echo "$out" | sed -n 's/.*per_dispatch=\([0-9.]*\)us.*/\1/p')
  read nsmmul gmac <<<"$(awk -v p="$per" -v mm="$mmuls" 'BEGIN{printf "%.3f %.1f", p*1000/mm, mm*512/p/1000}')"
  printf '%-7s %-7s %-11s %-13s %s\n' "$T" "$reuse" "$nsmmul" "$gmac" "mmuls/disp=$mmuls"
done
