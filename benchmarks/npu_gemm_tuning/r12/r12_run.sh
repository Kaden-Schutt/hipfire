#!/usr/bin/env bash
# R12 — array scaling of the L2-tiled streaming GEMM (aie2/XDNA1). Builds r11_gemm.cc
# (reused) + r12_gen.py for a COLSxROWS core grid, each core streaming its OWN block feed,
# and reports AGGREGATE + per-core GMAC/s from npu_gemm_bench. Sweeping the grid shows the
# shared-shim saturation: cols should scale ~linearly (independent shims, R6); rows-per-col
# contend. This is the pessimistic distinct-data feed (no cross-core reuse) — a lower bound
# on a real broadcast-tiled GEMM.
#
# Usage: ./r12_run.sh [LMxLNxKT] [N_BLK] [ITERS] ["COLSxROWS" ...]  default 6x12x16 64 60
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R11="$HERE/../r11"; REPO="$(cd "$HERE/../../.." && pwd)"
BLK="${1:-6x12x16}"; NBLK="${2:-64}"; ITERS="${3:-60}"
LM="${BLK%%x*}"; rest="${BLK#*x}"; LN="${rest%%x*}"; KT="${rest#*x}"
shift $(( $# < 3 ? $# : 3 )) || true
GRIDS=("$@"); [ ${#GRIDS[@]} -eq 0 ] && GRIDS=(1x1 4x1 4x2 4x4)

: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"; source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="$PEANO/bin:$MA_ROOT/bin:$PATH"
command -v xclbinutil >/dev/null 2>&1 || export PATH="/opt/xilinx/xrt/bin:$PATH"
for B in "$HOME/.cache/hipfire-npu-deps/lib" "$HOME/.cache/hipfire-npu-deps/extract/usr/lib/x86_64-linux-gnu"; do
  [ -e "$B/libboost_program_options.so.1.83.0" ] && export LD_LIBRARY_PATH="$B:${LD_LIBRARY_PATH:-}" && break
done
INC="$MA_ROOT/include"
WD="${R12_WD:-/tmp/r12_array}"; mkdir -p "$WD"
AB=$(( LM*KT*64 )); WB=$(( LN*KT*64 )); CB=$(( LM*LN*32 )); EXPECT=$(( KT*16 ))
# one shared r11.o (block = $BLK)
"$PEANO/bin/clang++" "$R11/r11_gemm.cc" -c -o "$WD/r11.o" -I"$INC" \
  -std=c++20 -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -O2 -DNDEBUG --target=aie2-none-unknown-elf -DLM=$LM -DLN=$LN -DKT=$KT

echo "R12 array scaling: block=${BLK} N_BLK=$NBLK iters=$ITERS  (distinct feed per core, lower bound)"
printf '%-8s %-6s %-11s %-15s %-15s %-7s %s\n' grid cores per_disp_us GMAC/s_aggregate GMAC/s_per_core c0 note
for G in "${GRIDS[@]}"; do
  COLS="${G%x*}"; ROWS="${G#*x}"; NC=$(( COLS*ROWS )); dir="$WD/$G"; rm -rf "$dir"; mkdir -p "$dir"
  cp "$WD/r11.o" "$dir/r11.o"
  python3 "$HERE/r12_gen.py" "$COLS" "$ROWS" "$LM" "$LN" "$KT" "$NBLK" > "$dir/aie.mlir"
  if ! aiecc "$dir/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
      --aie-generate-npu-insts --npu-insts-name="$dir/insts.bin" \
      --aie-generate-xclbin --xclbin-name="$dir/final.xclbin" --tmpdir="$dir" \
      >"$dir/aiecc.log" 2>&1; then
    printf '%-8s %-6s AIECC FAILED: %s\n' "$G" "$NC" "$(grep -oiE 'exceeded available memory|no space|channel|error:[^\n]*' "$dir/aiecc.log" | head -1)"; continue
  fi
  asz=$(( NC*NBLK*AB )); wsz=$(( NC*NBLK*WB )); csz=$(( NC*NBLK*CB*4 ))
  macs=$(( NC*NBLK*LM*LN*KT*512 ))
  out=$(cd "$REPO" && cargo run -q -p hipfire-xdna --example npu_gemm_bench -- \
        "$dir" "$asz" "$wsz" "$csz" "$macs" "$ITERS" "$EXPECT" 2>&1) || {
    printf '%-8s %-6s DISPATCH FAILED: %s\n' "$G" "$NC" "$(echo "$out"|tail -1)"; continue; }
  c0=$(echo "$out" | sed -n 's/.*C\[0\] = \([0-9-]*\).*/\1/p')
  per=$(echo "$out" | sed -n 's/.*per_dispatch=\([0-9.]*\)us.*/\1/p')
  read agg pc <<<"$(awk -v p="$per" -v m="$macs" -v nc="$NC" 'BEGIN{printf "%.1f %.1f", m/p/1000, m/p/1000/nc}')"
  note=$([ "$c0" = "$EXPECT" ] && echo "ok" || echo "MISMATCH exp=$EXPECT")
  printf '%-8s %-6s %-11s %-15s %-15s %-7s %s\n' "$G" "$NC" "$per" "$agg" "$pc" "$c0" "$note"
done
