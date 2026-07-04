#!/usr/bin/env bash
# Interestingness oracle for the GENUINE mlir-aie #3281 bug (delta-debugging).
#
# The real aiecc pipeline assigns lock IDs BEFORE the objectFifo transform, so
# the true bug is: aie-objectFifo-stateful-transform's unrollForLoops aborts on
# WELL-FORMED (ID-assigned) input, because unrolling an aie.objectfifo.acquire
# inside an scf.for nest fabricates a lock whose ID is never assigned
# (message-less std::optional::value abort under -fno-exceptions; crash frame is
# AIEObjectFifoStatefulTransformPass::unrollForLoops).
#
# This oracle guards against ddmin drifting to an unrelated trigger:
#   (1) --aie-assign-lock-ids ALONE must SUCCEED (input stays well-formed), AND
#   (2) --aie-assign-lock-ids + --aie-objectFifo-stateful-transform must ABORT.
# LLVM_DISABLE_SYMBOLIZATION skips the ~90s backtrace symbolization over the
# huge aie-opt binary, making the oracle ~0.2s; we only need the return code.
#
# Set AIE_OPT to your mlir_aie aie-opt (else "aie-opt" from PATH).
set -u
AIEOPT="${AIE_OPT:-aie-opt}"
E=(env LLVM_DISABLE_SYMBOLIZATION=1 LLVM_SYMBOLIZER_PATH=/bin/false)

"${E[@]}" "$AIEOPT" --aie-assign-lock-ids "$1" >/dev/null 2>&1 || exit 1
"${E[@]}" "$AIEOPT" --aie-assign-lock-ids --aie-objectFifo-stateful-transform "$1" >/dev/null 2>&1
[ $? -ge 128 ]
