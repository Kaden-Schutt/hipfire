# R8 — the matrix unit is FAST; R5–R7 were load-bound, not op-bound

R7 concluded the ~0.6-TOPS XDNA1 wall was the `mac_4x16_16x8_conf` op throughput
(~32 cyc/mmul), a hard floor. **That was wrong.** R7 timed the op *through* a
streaming K-loop that does two L1 loads (`ldA`+`ldW`) per mac with zero data reuse —
so it measured the load path, not the matrix unit. On the tip that aie2 int4 `<4,16,8>`
is a virtual op (int4 has no native MAC; it unpacks to int8 `2x8x8`), R8 isolates the
real matrix rate with a microbench: one core hammers a chosen mmul shape `REPEAT` times
over a **single resident tile** (loaded once into registers — no loads, no sync in the
hot loop). `r8_run.sh` builds `r8_ubench.cc` + `r8_1core.mlir` per shape and reports
ns/mmul from `npu_gemm_bench`.

## Matrix-unit rates (resident data, aie2/XDNA1)

Verified the hot loop is real (not folded): per-dispatch time is linear in REPEAT with
a ~130 µs fixed intercept (the dispatch floor). The **slope** is the true rate:

| shape (dtype) | MAC/op | true ns/mmul | ns/MAC | GMAC/s/core |
|---|---|---|---|---|
| `<4,16,8>` int8×**int4** (virtual) | 512 | **~1.05** | 0.0021 | **~489** |
| `<4,16,8>` int8×int8 (virtual) | 512 | ~2.46 | 0.0048 | ~208 |
| `<8,8,8>`  int8×int8 | 512 | ~2.57 | 0.0050 | ~199 |
| `<2,8,8>`  int8×int8 (native) | 128 | ~1.42 | 0.0111 | ~90 |

(ns/mmul from REPEAT=80k except the int4 row, whose 40k→80k slope gives 1.05; the
others are near-asymptotic high-REPEAT reads. `<4,8,4>` aiecc-failed, dropped.)

**~489 GMAC/s/core × 16 cores ≈ 7.8 TMAC/s = ~15.6 TOPS** compute ceiling for the int4
op — vs the R5 streaming result of **0.56 TOPS**. The streaming kernel runs at **~4% of
the matrix unit's rate**; it is ~25× **load-bound**, not op-bound.

## Three corrections to the R5–R7 record

1. **The op is not the wall.** ~1 ns/mmul (~1.9 cyc), not ~32. R7's "op-throughput
   floor" measured `ldA`+`ldW` per mac (~2 loads/mmul), which NACC can't reduce — hence
   R7's weak ~1.35×. The bottleneck is loads-per-mac (no reuse), and it is fixable.
2. **int4 is the *best* op here, not a tax.** int4 `<4,16,8>` (1.05 ns) beats int8
   `<4,16,8>` (2.46 ns): the unpack is ~free in the pipeline and int4 halves W
   bandwidth. On XDNA1, W4A8 gives no compute *penalty* and a real feed advantage.
3. **Big tiles beat native small ops.** The 512-MAC `<4,16,8>` yields ~4× the GMAC/s of
   the "native" 128-MAC `2x8x8`, because per-op issue overhead amortizes over more MACs.
   Emitting native micro-shapes directly is a *pessimization* for throughput.

## R9 (the real lever, at last)

Register-tile the microkernel for **load reuse** — the actual point of the reference
`mm.cc` 2×2 idiom (one loaded A row serves several output-N tiles; one loaded W tile
serves several output-M rows), so N loads feed N² macs instead of 1. That amortizes the
load path toward the ~15-TOPS compute rate. This is where XDNA1 GEMM performance
actually lives; R5's cascade and R6/R7's knobs were all downstream of the wrong wall.
Only after load-reuse closes the gap does the DMA feed (shim→L1) become the next limit.

Repro: `./r8_run.sh 80000 4 80 0 1 3 4`; linearity: `for R in 20000 40000 80000; do
./r8_run.sh $R 4 80 0; done` (slope over REPEAT = true ns/mmul).
