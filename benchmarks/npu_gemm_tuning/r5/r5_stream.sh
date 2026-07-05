#!/usr/bin/env bash
# R5 streaming-cascade PAYOFF measurement. For each N_BTILES in the sweep: generate
# the streaming array (r5_stream_gen.py), build it (r5_build.sh), and time the
# dispatch loop through the npu_gemm_bench example. TOPS vs N_BTILES shows the fixed
# per-dispatch overhead (ERT submit + hwctx sync + DMA setup/await) amortizing across
# more output tiles, so the plateau is the steady-state cascade-dataflow throughput —
# the number to compare against SOTA's ~5 TOPS and the 15.7 reference.
#
# The design pins KSLICE=16 (the per-tile A/W memref sizes the generator emits assume
# it), so KSLICE is not a sweep knob here.
#
# Usage: R5_ARCH=aie2 ./r5_stream.sh [COLS] [ROWS] [ITERS] [NBT...]
#   defaults: aie2 -> 4 4 500, sweep 1 2 4 8 16 32 64  (XDNA1 max array is 4x4)
#   aie2p    -> 8 4 500                                (Strix Halo max is 8x4)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
: "${R5_ARCH:=aie2}"
KSLICE=16
case "$R5_ARCH" in
  aie2)  DEF_COLS=4; AW=1024; WW=1024; CWi=32; MK_N=512  ;;  # <4,16,8>:  4*16*8 MACs/mmul
  aie2p) DEF_COLS=8; AW=1024; WW=2048; CWi=64; MK_N=1024 ;;  # <4,16,16>: 4*16*16
  *) echo "bad R5_ARCH=$R5_ARCH (want aie2 or aie2p)" >&2; exit 2 ;;
esac
COLS="${1:-$DEF_COLS}"; ROWS="${2:-4}"; ITERS="${3:-500}"
shift $(( $# < 3 ? $# : 3 )) || true
NBTS=("$@"); [ ${#NBTS[@]} -eq 0 ] && NBTS=(1 2 4 8 16 32 64)
EXPECT=$(( ROWS * KSLICE * 16 ))
WD="${R5_WD:-/tmp/r5_stream_$R5_ARCH}"; mkdir -p "$WD"

echo "R5 streaming-cascade sweep: arch=$R5_ARCH array=${COLS}x${ROWS} KSLICE=$KSLICE iters=$ITERS expect_c0=$EXPECT"
printf '%-6s %-11s %-13s %-9s %s\n' NBT per_disp_us MACs/dispatch TOPS c0
for NBT in "${NBTS[@]}"; do
  mlir="$WD/nbt$NBT.mlir"; dir="$WD/nbt$NBT"
  R5_ARCH="$R5_ARCH" python3 "$HERE/r5_stream_gen.py" "$COLS" "$ROWS" "$NBT" > "$mlir"
  if ! R5_ARCH="$R5_ARCH" "$HERE/r5_build.sh" "$mlir" "$dir" "$KSLICE" >"$dir.build.log" 2>&1; then
    printf '%-6s BUILD FAILED (see %s)\n' "$NBT" "$dir.build.log"; continue
  fi
  asz=$(( NBT * AW )); wsz=$(( NBT * WW )); csz=$(( COLS * NBT * CWi * 4 ))
  macs=$(( COLS * ROWS * KSLICE * MK_N * NBT ))
  out=$(cd "$REPO" && cargo run -q -p hipfire-xdna --example npu_gemm_bench -- \
        "$dir" "$asz" "$wsz" "$csz" "$macs" "$ITERS" "$EXPECT" 2>&1) || {
    printf '%-6s DISPATCH FAILED: %s\n' "$NBT" "$(echo "$out" | tail -1)"; continue; }
  c0=$(echo "$out" | sed -n 's/.*all-ones C\[0\] = \([0-9-]*\).*/\1/p')
  per=$(echo "$out" | sed -n 's/.*per_dispatch=\([0-9.]*\)us.*/\1/p')
  tops=$(echo "$out" | sed -n 's/.*=> \([0-9.]*\) TOPS.*/\1/p')
  printf '%-6s %-11s %-13s %-9s %s\n' "$NBT" "$per" "$macs" "$tops" "$c0"
done
