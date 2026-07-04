# mlir-aie #3281 — minimal repro (objectFifo unroll fabricates an ID-less lock)

Upstream: https://github.com/Xilinx/mlir-aie/issues/3281

## The bug (corrected characterization)

`aie-opt --aie-objectFifo-stateful-transform` aborts in
`AIEObjectFifoStatefulTransformPass::unrollForLoops` when an
`aie.objectfifo.acquire` sits inside an `scf.for` nest that the pass unrolls.
Unrolling clones the acquire and fabricates one lock per copy, but those newly
created `aie.lock`s never get an ID assigned — so a later
`getLockID().value()` aborts.

**Real-pipeline framing (this is the genuine bug).** aiecc runs
`--aie-assign-lock-ids` *before* the objectFifo transform, so all *input* locks
already have IDs. The abort therefore comes from a lock the *pass itself*
creates during unroll, on well-formed input. Under `-fno-exceptions` this is a
**message-less abort** (no "Lock has no ID value" text), crash frame in
`unrollForLoops`.

**Single-pass framing (misleading — avoid).** Running
`--aie-objectFifo-stateful-transform` alone on IR whose *input* locks are still
ID-less prints the explicit assert `LockOp::getLockIDValue(): "Lock has no ID
value"`. That is easy to dismiss as "run assign-lock-ids first," so the repro
below uses the faithful two-pass order to prove the pass creates the bad lock.

## Reproduce

```bash
export AIE_OPT=/path/to/mlir_aie/bin/aie-opt
# genuine bug — aborts in unrollForLoops on well-formed input:
"$AIE_OPT" --aie-assign-lock-ids --aie-objectFifo-stateful-transform \
    objectfifo_unroll_lock_3281.mlir
```

`objectfifo_unroll_lock_3281.mlir` is 35 lines: one `aie.device`, one
`aie.objectfifo`, one `aie.core` whose triple-nested `scf.for` contains a single
`aie.objectfifo.acquire`.

## How it was reduced

- **Oracle:** `interesting.sh` (faithful two-pass; guards that assign-lock-ids
  alone succeeds so ddmin can't cheat with a pre-existing ID-less lock).
- **Symbolization trick:** the abort backtrace symbolization over the ~GB
  `aie-opt` binary costs ~90 s; `LLVM_DISABLE_SYMBOLIZATION=1` +
  `LLVM_SYMBOLIZER_PATH=/bin/false` drops the oracle to ~0.2 s (the assert/rc
  is emitted before symbolization).
- **Reducer:** `ddmin_lines.py` (line-level Zeller ddmin). Path:
  IRON fused design at min config (M=K=N=256, 1 col) → 282-line
  `resource_alloc_crash.mlir` → line-ddmin + region pruning → 35 lines.
- No dialect-aware `mlir-reduce` exists in the toolchain (mlir_aie ships only
  `aie-opt`; the ROCm `mlir-reduce` can't parse the `aie.*` dialect), so this
  homemade ddmin is the reduction path.
