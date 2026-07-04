#!/usr/bin/env bash
# R1b feed measurement driver: sets up the halo NPU toolchain env, coordinates
# the NPU, and runs the differential-slope sweep (sweep_r1b.py).
#
# The slope method is deliberate: a warm single-shot host-wall run is dominated by
# ~16 ms of FIXED per-call overhead (device load + BO alloc + dispatch), NOT the
# feed -- that is what made R1a read ~0.9 GB/s. Fitting call_ms vs bytes over an
# N_TILES sweep cancels the fixed cost and recovers the true byte-proportional
# rate. Totals must be large (tens of MB) for the feed to clear the 16 ms floor.
#
# Env (LOCAL to halo; override to run elsewhere):
#   MLIR_AIE_DIR / HIPFIRE_NPU_VENV / XRT_SETUP  as in tune.sh
#   TILE_N / DEPTHS / NT_SWEEP / MINIMALS / REPEAT  forwarded to sweep_r1b.py
set -uo pipefail

NPU_VENV="${HIPFIRE_NPU_VENV:-$HOME/.venv}"
XRT_SETUP="${XRT_SETUP:-/opt/xilinx/xrt/setup.sh}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- toolchain env ---
# shellcheck disable=SC1091
source "$NPU_VENV/bin/activate" || { echo "ERROR: venv $NPU_VENV missing"; exit 1; }
PEANO_INSTALL_DIR="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
export PEANO_INSTALL_DIR
export PATH="$PEANO_INSTALL_DIR/bin:$PATH"
# shellcheck disable=SC1090
source "$XRT_SETUP" >/dev/null 2>&1 || true
# mlir_aie is a namespace package (__file__ is None); resolve via __path__.
MA_ROOT="$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])' 2>/dev/null)"
: "${MLIR_AIE_INC:=$MA_ROOT/include}"
export MLIR_AIE_INC
[ -d "$MLIR_AIE_INC" ] || { echo "ERROR: MLIR_AIE_INC not found ($MLIR_AIE_INC)"; exit 1; }
# The `aie` python package lives under mlir_aie/python; pyxrt ships with XRT.
PYXRT_DIR="$(dirname "$(find /opt/xilinx/xrt -name 'pyxrt*.so' 2>/dev/null | head -1)")"
export PYTHONPATH="$MA_ROOT/python${PYXRT_DIR:+:$PYXRT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

# --- NPU coordination (best-effort; benches self-coordinate per AGENTS.md) ---
if command -v hipfire >/dev/null 2>&1 && hipfire lock acquire >/dev/null 2>&1; then
  trap 'hipfire lock release >/dev/null 2>&1' EXIT
else
  echo "note: proceeding without hipfire lock (not on PATH or busy)"
fi

exec python "$HERE/sweep_r1b.py"
