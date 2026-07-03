#!/usr/bin/env bash
# NPU int8 GEMM tuning loop (C++-host path).
#
# Drives the mlir-aie `whole_array` matmul design: builds an xclbin per config
# via Peano/aiecc, runs it on the NPU through the *compiled C++ host* (the only
# reliable run path on this box — the python test.py harness segfaults on the
# mlir_aie native bindings under Python 3.14), parses sustained throughput, and
# appends one comparable CSV row per config.
#
# The C++-host path is used on purpose: it sidesteps python / mlir_aie-native /
# pyxrt entirely. Compilation and hardware run both go through it.
#
# Toolchain deps are LOCAL to this machine (halo) and configurable via env so
# nothing here assumes these paths exist elsewhere:
#   MLIR_AIE_DIR      mlir-aie source tree            (default: ~/build/mlir-aie)
#   HIPFIRE_NPU_VENV  venv with mlir_aie + llvm-aie   (default: ~/.venv)
#   XRT_SETUP         XRT setup.sh                    (default: /opt/xilinx/xrt/setup.sh)
#   DEV               device name                     (default: npu2 = Strix Halo aie2p)
#   DTYPE_IN/DTYPE_OUT  i8/i8 (int8 MAC, int32 acc in-register; i32 out overflows 64KB L1)
#   WARMUP/ITERS      host timing iterations          (default: 5 / 20)
#   PEAK_TOPS         ceiling for %-of-peak column    (default: 55, measured NPU peak)
#   OUT               results CSV path                (default: results/tune-<utc>.csv)
#
# Usage:
#   ./tune.sh                      # runs the default config matrix below
#   ./tune.sh "M K N m k n cols" ["M K N m k n cols" ...]   # explicit configs
set -uo pipefail

MLIR_AIE_DIR="${MLIR_AIE_DIR:-$HOME/build/mlir-aie}"
NPU_VENV="${HIPFIRE_NPU_VENV:-$HOME/.venv}"
XRT_SETUP="${XRT_SETUP:-/opt/xilinx/xrt/setup.sh}"
DEV="${DEV:-npu2}"
DTYPE_IN="${DTYPE_IN:-i8}"
DTYPE_OUT="${DTYPE_OUT:-i8}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
PEAK_TOPS="${PEAK_TOPS:-55}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/results/tune-$(date -u +%Y%m%dT%H%M%SZ).csv}"

WA_DIR="$MLIR_AIE_DIR/programming_examples/basic/matrix_multiplication/whole_array"
[ -d "$WA_DIR" ] || { echo "ERROR: whole_array design not found at $WA_DIR"; exit 1; }

# --- toolchain env ---
# shellcheck disable=SC1091
source "$NPU_VENV/bin/activate" || { echo "ERROR: venv $NPU_VENV missing"; exit 1; }
PEANO_INSTALL_DIR="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
export PEANO_INSTALL_DIR
export PATH="$PEANO_INSTALL_DIR/bin:$PATH"
# shellcheck disable=SC1090
source "$XRT_SETUP" >/dev/null 2>&1
[ -x "$PEANO_INSTALL_DIR/bin/clang" ] || { echo "ERROR: Peano not found at $PEANO_INSTALL_DIR"; exit 1; }

# Default config matrix: "M K N m k n cols".
# Baselines first (already characterized), then the MAC-tile levers to sweep.
DEFAULT_CFGS=(
  "4096 4096 4096 64 128 64 8"   # best baseline so far (~6.2 TOPS)
  "4096 4096 4096 64 128 64 4"   # half the columns (~3.5 TOPS)
  "4096 4096 4096 64 64  64 8"   # smaller K tile
  "4096 4096 4096 64 128 128 8"  # wider N tile (more C reuse per load)
)

if [ "$#" -gt 0 ]; then CFGS=("$@"); else CFGS=("${DEFAULT_CFGS[@]}"); fi

mkdir -p "$(dirname "$OUT")"
[ -f "$OUT" ] || echo "utc,dtype_in,dtype_out,M,K,N,m,k,n,cols,warmup,iters,avg_us,avg_gflops,tops,pct_peak,status" > "$OUT"

printf '%-22s %-7s %-9s %8s %7s %7s  %s\n' "config(MxKxN mxkxn)" "cols" "avg_us" "gflops" "TOPS" "%peak" "status"
echo "-------------------------------------------------------------------------------------"

cd "$WA_DIR"
for cfg in "${CFGS[@]}"; do
  read -r M K N m k n cols <<<"$cfg"
  suffix="${M}x${K}x${N}_${m}x${k}x${n}_${cols}c"
  log="$(mktemp)"
  # Build (per-config xclbin, cached by suffix) + run on NPU with stable timing.
  if ! env dtype_in="$DTYPE_IN" dtype_out="$DTYPE_OUT" m="$m" k="$k" n="$n" \
        M="$M" K="$K" N="$N" n_aie_cols="$cols" \
        runargs="-v 2 --warmup $WARMUP --iters $ITERS" \
        make -f ./Makefile run devicename="$DEV" >"$log" 2>&1; then
    status="BUILD_OR_RUN_FAIL"
    us=""; gf=""; tops=""; pct=""
  else
    us="$(grep -m1 'Avg NPU matmul time' "$log" | grep -oE '[0-9.]+us' | tr -d 'us')"
    gf="$(awk -F': ' '/Avg NPU gflops/{print $2; exit}' "$log")"
    if grep -q 'PASS!' "$log"; then status="PASS"; else status="VERIFY_FAIL"; fi
    if [ -n "$gf" ]; then
      tops="$(awk -v g="$gf" 'BEGIN{printf "%.2f", g/1000}')"
      pct="$(awk -v g="$gf" -v p="$PEAK_TOPS" 'BEGIN{printf "%.1f", (g/1000)/p*100}')"
    fi
  fi
  rm -f "$log"
  printf '%-22s %-7s %-9s %8s %7s %6s%%  %s\n' \
    "${M}x${K}x${N} ${m}x${k}x${n}" "${cols}c" "${us:-–}" "${gf:-–}" "${tops:-–}" "${pct:-–}" "$status"
  echo "$(date -u +%Y%m%dT%H%M%SZ),$DTYPE_IN,$DTYPE_OUT,$M,$K,$N,$m,$k,$n,$cols,$WARMUP,$ITERS,${us},${gf},${tops},${pct},$status" >> "$OUT"
done

echo
echo "results -> $OUT"
