# R2 — W4A8 compute fused with weight streaming (int4×int8 GEMM)

R0 established the compute *microkernel* (disasm-only), R1 the *feed* (~14 GB/s/
stream, ~55 GB/s aggregate NPU fabric link). R2 fuses them: stream int4 weights
into a core doing `mac_4x16_16x16` and measure the **sustained** rate on hardware.

## R2a — single-core streaming W4A8 (`r2a_gemm.cc` + `r2a_run.py`)

Streams N_BTILES int4 weight tiles (128 B) into one core; per tile the core does
(INNER+1)·NACC macs against NACC resident int8 activation tiles. int4 has no numpy
dtype, so weights are passed as packed int8 and reinterpreted in-kernel. Sustained
rate via host-wall differential across N_BTILES (cancels the ~16 ms fixed cost).

### Findings

1. **Named accumulators + `chess_prepare_for_pipelining` are mandatory.** An
   `acc[]/a[]` array loop compiles to ~24 cyc/vmac (0.15 TOPS): it carries
   per-iteration address arithmetic and collapses the accumulators. Four *named*
   accumulators + the pipelining hint (the R0b recipe) reach the tight loop.

2. **int4 W4A8 is NOT 2× int8 — it's the same compute rate.** Measured in the
   same harness:

   | weights | MAC/cyc | TOPS/core |
   |---|---|---|
   | int4 (W4A8, `mmul<4,16,16>`) | ~460 | ~1.7 |
   | int8 (W8A8, `mmul<8,8,8>`)   | ~418 | ~1.5 |

   Both sit at ~85–90% of the 512 MAC/cyc int8 peak. The disasm shows why: each
   `mac_4x16_16x16` emits **~2 vmacs** (the 16-deep contraction splits into two
   8-deep `y0`/`y1` passes), so 1024 nominal MACs cost 2 cycles = ~512 MAC/cyc —
   the same as int8's 8×8×8 in one cycle. **R0's 117-TOPS W4A8 ceiling assumed
   1024 MACs/vmac at II=1, which the codegen does not realize.** On this toolchain
   int4's advantage is **halved weight bandwidth, not 2× compute.**

3. **Consequences for the roadmap.** W4A8 aggregate compute is ~58 TOPS (≈ int8),
   not 117. That *lowers* the feed/compute crossover: feed needed = `29e12/M` B/s
   vs the ~55 GB/s link ⇒ crossover **M ≈ 264** (was ~530 under the 117 premise).
   So W4A8 prefill is compute-bound for M ≳ 256 — the feed is even less of a wall.
   The int4 win is now correctly located: it feeds from **half the weight bytes**,
   which matters for the *bandwidth*-bound decode case, not compute-bound prefill.

### Open / next

- Codegen: can a different int4 tiling (or intrinsic) hit true 1024 MAC/cyc, or is
  ~512 MAC/cyc the aie2p int4 hardware rate? (The 2-vmac split looks structural.)
- Per-tile dataflow overhead still caps single-core at ~1.7 of the ~1.8 ceiling at
  the largest INNER; a real tiled GEMM (memtile-staged C, no per-tile store) is the
  next efficiency step.
- Scale to the 8×4 array and confirm aggregate ~58 TOPS with the R1 feed at M≥256.

Harness: NACC=4 (named), INNER = reuse knob, INT8W=1 for the int8 control,
N_BTILES = stream length. Fresh process per run (pyxrt segfaults on repeat, py3.14).
