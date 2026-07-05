#!/usr/bin/env bash
# R10 — streaming-GEMM end-to-end rate (aie2/XDNA1, single core). Builds r10_gemm.cc +
# r10_stream_gen.py per (MT,NT,KT) and streams N_BLK blocks through npu_gemm_bench, so the
# reported GMAC/s INCLUDES the shim->L1 DMA feed. Compare to R9's L1-resident 150 GMAC/s:
# if R10 holds ~150 the feed keeps up; if it drops, the GEMM is DMA-bound at ~intensity
# (=NT*4 mac/byte for the MTxNT tile) x shim BW.
#
# Usage: ./r10_run.sh [N_BLK] [ITERS] ["MTxNTxKT" ...]   default 1024 80, 3x3x16/32/64
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
NBLK="${1:-1024}"; ITERS="${2:-80}"
shift $(( $# < 2 ? $# : 2 )) || true
CFGS=("$@"); [ ${#CFGS[@]} -eq 0 ] && CFGS=(3x3x16 3x3x32 3x3x64 2x2x64)

: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"; source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')"
export PATH="$PEANO/bin:$MA_ROOT/bin:$PATH"
command -v xclbinutil >/dev/null 2>&1 || export PATH="/opt/xilinx/xrt/bin:$PATH"
for B in "$HOME/.cache/hipfire-npu-deps/lib" "$HOME/.cache/hipfire-npu-deps/extract/usr/lib/x86_64-linux-gnu"; do
  [ -e "$B/libboost_program_options.so.1.83.0" ] && export LD_LIBRARY_PATH="$B:${LD_LIBRARY_PATH:-}" && break
done
INC="$MA_ROOT/include"
WD="${R10_WD:-/tmp/r10_gemm}"; mkdir -p "$WD"

echo "R10 streaming-GEMM (1 core, real DDR->L1 feed): N_BLK=$NBLK iters=$ITERS  (int4 <4,16,8>)"
printf '%-10s %-9s %-11s %-13s %-8s %s\n' MTxNTxKT intens ns/mmul GMAC/s/core c0 note
for C in "${CFGS[@]}"; do
  MT="${C%%x*}"; rest="${C#*x}"; NT="${rest%%x*}"; KT="${rest#*x}"; dir="$WD/$C"; rm -rf "$dir"; mkdir -p "$dir"
  if ! "$PEANO/bin/clang++" "$HERE/r10_gemm.cc" -c -o "$dir/r10.o" -I"$INC" \
      -std=c++20 -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
      -O2 -DNDEBUG --target=aie2-none-unknown-elf -DMT=$MT -DNT=$NT -DKT=$KT >"$dir/cc.log" 2>&1; then
    printf '%-10s COMPILE FAILED (%s)\n' "$C" "$dir/cc.log"; continue
  fi
  python3 "$HERE/r10_stream_gen.py" "$MT" "$NT" "$KT" "$NBLK" > "$dir/aie.mlir"
  if ! aiecc "$dir/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge --peano="$PEANO" \
      --aie-generate-npu-insts --npu-insts-name="$dir/insts.bin" \
      --aie-generate-xclbin --xclbin-name="$dir/final.xclbin" --tmpdir="$dir" \
      >"$dir/aiecc.log" 2>&1; then
    printf '%-10s AIECC FAILED (%s)\n' "$C" "$dir/aiecc.log"; continue
  fi
  AB=$(( MT*KT*64 )); WB=$(( NT*KT*64 )); CB=$(( MT*NT*32 ))
  asz=$(( NBLK*AB )); wsz=$(( NBLK*WB )); csz=$(( NBLK*CB*4 ))
  mmuls=$(( NBLK*MT*NT*KT )); macs=$(( mmuls*512 )); expect=$(( KT*16 ))
  intens=$(( NT*4 ))
  out=$(cd "$REPO" && cargo run -q -p hipfire-xdna --example npu_gemm_bench -- \
        "$dir" "$asz" "$wsz" "$csz" "$macs" "$ITERS" "$expect" 2>&1) || {
    printf '%-10s DISPATCH FAILED: %s\n' "$C" "$(echo "$out"|tail -1)"; continue; }
  c0=$(echo "$out" | sed -n 's/.*C\[0\] = \([0-9-]*\).*/\1/p')
  per=$(echo "$out" | sed -n 's/.*per_dispatch=\([0-9.]*\)us.*/\1/p')
  read nsmmul gmac <<<"$(awk -v p="$per" -v mm="$mmuls" 'BEGIN{printf "%.3f %.1f", p*1000/mm, mm*512/p/1000}')"
  note=$([ "$c0" = "$expect" ] && echo "ok" || echo "C MISMATCH exp=$expect")
  printf '%-10s %-9s %-11s %-13s %-8s %s\n' "$C" "${intens}mac/B" "$nsmmul" "$gmac" "$c0" "$note"
done
