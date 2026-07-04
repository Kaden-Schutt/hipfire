#!/usr/bin/env bash
# R0: compile the native aie2p MAC-conf microkernels and disassemble them to
# confirm they lower to the hardware VMAC. Set PEANO (llvm-aie install) and
# MLIR_AIE_INC (mlir_aie include dir) to your toolchain.
set -euo pipefail
: "${PEANO:?set PEANO=/path/to/llvm-aie}"
: "${MLIR_AIE_INC:?set MLIR_AIE_INC=/path/to/mlir_aie/include}"
cd "$(dirname "$0")"

"$PEANO/bin/clang" --target=aie2p-none-unknown-elf -std=c++20 \
    -I"$MLIR_AIE_INC" -O2 -c r0_conf.cc -o r0_conf.o
"$PEANO/bin/llvm-objdump" -d --no-show-raw-insn r0_conf.o | tee r0_conf.aie2p.dis \
  | grep -iE 'vmac|vmul' && echo "OK: native VMAC emitted"
