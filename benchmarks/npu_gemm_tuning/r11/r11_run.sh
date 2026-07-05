#!/usr/bin/env bash
# R11 — L2-tiled streaming-GEMM sweep (aie2/XDNA1, single core). Builds r11_gemm.cc +
# r11_stream_gen.py per (LM,LN,KT) and streams N_BLK blocks through npu_gemm_bench. As the
# L1 block LMxLN grows, DMA intensity (8*LM*LN/(LM+LN)) rises and the DMA-bound rate should
# climb from R10's ~70 back toward R9's ~150 GMAC/s/core register ceiling.
#
# Usage: ./r11_run.sh [N_BLK] [ITERS] ["LMxLNxKT" ...]   default 512 80, 3x3/6x6/9x9 @KT16
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
NBLK="${1:-512}"; ITERS="${2:-80}"
shift $(( $# < 2 ? $# : 2 )) || true
CFGS=("$@"); [ ${#CFGS[@]} -eq 0 ] && CFGS=(3x3x16 6x6x16 9x9x16 6x6x32)

: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"; source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="$PEANO/bin:$MA_ROOT/bin:$PATH"
command -v xclbinutil >/dev/null 2>&1 || export PATH="/opt/xilinx/xrt/bin:$PATH"
for B in "$HOME/.cache/hipfire-npu-deps/lib" "$HOME/.cache/hipfire-npu-deps/extract/usr/lib/x86_64-linux-gnu"; do
  [ -e "$B/libboost_program_options.so.1.83.0" ] && export LD_LIBRARY_PATH="$B:${LD_LIBRARY_PATH:-}" && break
done
INC="$MA_ROOT/include"
WD="${R11_WD:-/tmp/r11_gemm}"; mkdir -p "$WD"

echo "R11 L2-tiled streaming-GEMM (1 core, real feed): N_BLK=$NBLK iters=$ITERS  (int4 <4,16,8>, 3x3 reg tile)"
printf '%-10s %-9s %-8s %-11s %-13s %-8s %s\n' LMxLNxKT L1C_KB intens ns/mmul GMAC/s/core c0 note
for C in "${CFGS[@]}"; do
  LM="${C%%x*}"; rest="${C#*x}"; LN="${rest%%x*}"; KT="${rest#*x}"; dir="$WD/$C"; rm -rf "$dir"; mkdir -p "$dir"
  if ! "$PEANO/bin/clang++" "$HERE/r11_gemm.cc" -c -o "$dir/r11.o" -I"$INC" \
      -std=c++20 -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
      -O2 -DNDEBUG --target=aie2-none-unknown-elf -DLM=$LM -DLN=$LN -DKT=$KT >"$dir/cc.log" 2>&1; then
    printf '%-10s COMPILE FAILED (%s)\n' "$C" "$dir/cc.log"; continue
  fi
  python3 "$HERE/r11_stream_gen.py" "$LM" "$LN" "$KT" "$NBLK" > "$dir/aie.mlir"
  if ! aiecc "$dir/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
      --aie-generate-npu-insts --npu-insts-name="$dir/insts.bin" \
      --aie-generate-xclbin --xclbin-name="$dir/final.xclbin" --tmpdir="$dir" \
      >"$dir/aiecc.log" 2>&1; then
    printf '%-10s AIECC FAILED (%s)\n' "$C" "$(grep -oiE 'exceeded available memory|error:[^\n]*' "$dir/aiecc.log" | head -1)"; continue
  fi
  AB=$(( LM*KT*64 )); WB=$(( LN*KT*64 )); CB=$(( LM*LN*32 ))
  asz=$(( NBLK*AB )); wsz=$(( NBLK*WB )); csz=$(( NBLK*CB*4 ))
  mmuls=$(( NBLK*LM*LN*KT )); macs=$(( mmuls*512 )); expect=$(( KT*16 ))
  intens=$(awk -v m=$LM -v n=$LN 'BEGIN{printf "%.0f", 8*m*n/(m+n)}')
  l1c=$(awk -v c=$CB 'BEGIN{printf "%.1f", c*4*2/1024}')   # double-buffered C, KB
  out=$(cd "$REPO" && cargo run -q -p hipfire-xdna --example npu_gemm_bench -- \
        "$dir" "$asz" "$wsz" "$csz" "$macs" "$ITERS" "$expect" 2>&1) || {
    printf '%-10s DISPATCH FAILED: %s\n' "$C" "$(echo "$out"|tail -1)"; continue; }
  c0=$(echo "$out" | sed -n 's/.*C\[0\] = \([0-9-]*\).*/\1/p')
  per=$(echo "$out" | sed -n 's/.*per_dispatch=\([0-9.]*\)us.*/\1/p')
  read nsmmul gmac <<<"$(awk -v p="$per" -v mm="$mmuls" 'BEGIN{printf "%.3f %.1f", p*1000/mm, mm*512/p/1000}')"
  note=$([ "$c0" = "$expect" ] && echo "ok" || echo "C MISMATCH exp=$expect")
  printf '%-10s %-9s %-8s %-11s %-13s %-8s %s\n' "$C" "$l1c" "${intens}mac/B" "$nsmmul" "$gmac" "$c0" "$note"
done
