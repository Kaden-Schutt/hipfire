#!/usr/bin/env bash
# R6 — compute-per-sync sweep. R5 proved the streaming K-cascade on XDNA1 is bound by
# per-objectfifo-iteration overhead (sync + cascade beat), NOT compute or global feed:
# the tiny <4,16,8> tile does ~8k MACs (~26ns) wrapped in ~470ns of sync. The direct
# test of that diagnosis: raise the work done per objectfifo iteration by deepening
# KSLICE (mmuls each core contracts per output tile) and watch TOPS.
#
# KSLICE deepens K per tile, so per iteration BOTH compute (KSLICE mmuls) and feed
# (KSLICE*(A+W) bytes) scale, while the fixed per-iteration sync stays constant. So:
#   - sync-bound  -> TOPS climbs steeply with KSLICE (more work per fixed sync)
#   - feed-bound  -> TOPS flat (compute and feed scale together)
# This reuses the r5 rig unchanged (kernel is already -DKSLICE-parametrized; the
# generator derives buffer sizes from KSLICE). Array + N_BTILES are held fixed.
#
# L1 cap: per-core double-buffered A+W = 2*KSLICE*(64 + 8*MMUL_N) bytes. On aie2
# (MMUL_N=8) that is 256*KSLICE, so KSLICE=256 fills the 64KB tile L1 — sweep <=128.
#
# Usage: R5_ARCH=aie2 ./r6_intensity.sh [COLS] [ROWS] [NBT] [ITERS] [KSLICE...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R5="$HERE/../r5"
REPO="$(cd "$HERE/../../.." && pwd)"
: "${R5_ARCH:=aie2}"
case "$R5_ARCH" in
  aie2)  DEF_COLS=4; MMUL_N=8  ;;
  aie2p) DEF_COLS=8; MMUL_N=16 ;;
  *) echo "bad R5_ARCH=$R5_ARCH" >&2; exit 2 ;;
esac
COLS="${1:-$DEF_COLS}"; ROWS="${2:-4}"; NBT="${3:-1024}"; ITERS="${4:-300}"
shift $(( $# < 4 ? $# : 4 )) || true
KSS=("$@"); [ ${#KSS[@]} -eq 0 ] && KSS=(16 32 64 128)
WD="${R6_WD:-/tmp/r6_intensity_$R5_ARCH}"; mkdir -p "$WD"

echo "R6 compute-per-sync sweep: arch=$R5_ARCH array=${COLS}x${ROWS} N_BTILES=$NBT iters=$ITERS"
printf '%-8s %-11s %-14s %-9s %s\n' KSLICE per_disp_us MACs/dispatch TOPS c0
for KS in "${KSS[@]}"; do
  mlir="$WD/ks$KS.mlir"; dir="$WD/ks$KS"
  R5_ARCH="$R5_ARCH" python3 "$R5/r5_stream_gen.py" "$COLS" "$ROWS" "$NBT" "$KS" > "$mlir"
  if ! R5_ARCH="$R5_ARCH" "$R5/r5_build.sh" "$mlir" "$dir" "$KS" >"$dir.build.log" 2>&1; then
    printf '%-8s BUILD FAILED (see %s)\n' "$KS" "$dir.build.log"; continue
  fi
  AW=$(( KS * 64 )); WW=$(( KS * 8 * MMUL_N )); CW=$(( 4 * MMUL_N ))
  asz=$(( NBT * AW )); wsz=$(( NBT * WW )); csz=$(( COLS * NBT * CW * 4 ))
  macs=$(( COLS * ROWS * KS * 4 * 16 * MMUL_N * NBT ))
  expect=$(( ROWS * KS * 16 ))
  out=$(cd "$REPO" && cargo run -q -p hipfire-xdna --example npu_gemm_bench -- \
        "$dir" "$asz" "$wsz" "$csz" "$macs" "$ITERS" "$expect" 2>&1) || {
    printf '%-8s DISPATCH FAILED: %s\n' "$KS" "$(echo "$out" | tail -1)"; continue; }
  c0=$(echo "$out" | sed -n 's/.*all-ones C\[0\] = \([0-9-]*\).*/\1/p')
  per=$(echo "$out" | sed -n 's/.*per_dispatch=\([0-9.]*\)us.*/\1/p')
  tops=$(echo "$out" | sed -n 's/.*=> \([0-9.]*\) TOPS.*/\1/p')
  printf '%-8s %-11s %-14s %-9s %s\n' "$KS" "$per" "$macs" "$tops" "$c0"
done
