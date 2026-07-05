# R14 — the whole_array broadcast GEMM: 1.44 TOPS (2.6× the R5 line)

The compute-bound array number. Output C is a 4×4 grid of LM×LN blocks; block (i,j) runs on
core (col=j, row=2+i). Two orthogonal reuses, both memtile-routed (mechanism from R13),
using all 4 column shims:
- **W in-column broadcast** — column j's shim feeds W-stripe_j → memtile_j → the 4 cores of
  column j (all block-rows share the same W cols).
- **A cross-column broadcast** — block-row i's A-stripe enters shim_i → memtile_i → broadcast
  to core (c, 2+i) across **all** columns (all block-cols share the same A rows). This is the
  horizontal broadcast one column can't do (R13's ceiling); aiecc routes it via the mesh.
- **C in-column join** — each column's 4 cores' C gathered through its memtile → its shim.

So 4 A-stripes + 4 W-stripes feed 16 cores' macs → DMA intensity `32·LM·LN/(LM+LN)` = 128
mac/byte (4× the R12 distinct-feed 32). Built + correct on hardware (c0 = 256, `r14_gen.py`).

| N_BLK | per-dispatch | TOPS |
|---|---|---|
| 256 | 4.25 ms | 1.14 |
| 512 | 7.51 ms | 1.29 |
| 1024 | 13.4 ms | **1.44** |

**1.44 TOPS and still climbing** (per-dispatch overhead amortizes with N_BLK). That is
**2.6× R5's cascade (0.56)** and 1.9× R12's distinct-feed array (0.75) — the broadcast reuse
is the real array lever. At intensity 128 the 1.44 TOPS is only ~5.6 GB/s of DMA, well under
the ~13–16 GB/s DDR ceiling (R6), so it is now **memtile-fabric / per-dispatch-overhead bound,
not DDR-bound** — there is still headroom (deeper N_BLK, or fewer/larger DMA descriptors).
L1 caps the block at ~6×12×16 (the W-broadcast buffer dominates), which pins KT=16.

## Full line (R5→R14), all measured on gfx1103 XDNA1

| stage | what | result |
|---|---|---|
| R5 | cascade streaming | 0.56 TOPS (guessed sync-bound) |
| R6/R7 | KSLICE + multi-accum | disproved sync; wrongly "op floor" |
| R8 | resident microbench (int4 is virtual) | matrix unit ~15 TOPS/core; it was **load**-bound |
| R9 | 3×3 register tiling | 150 GMAC/s/core resident (8.6× R5) |
| R10/R11 | streaming + L2 tiling | 143 GMAC/s/core, 95% of ceiling on real feed |
| R12 | array, distinct feed | 0.75 TOPS, DDR-bound; shim = 1 core/col |
| R13 | memtile routing (1 column) | mechanism works; 0.39 (1 DDR path) |
| **R14** | **whole_array broadcast (4×4)** | **1.44 TOPS**, fabric-bound, headroom left |

The bottleneck walked sync → op → loads → DDR-bandwidth → memtile-fabric, and the NPU line
went from a 0.56-TOPS "dead end" to a working, correct, broadcast-reuse W4A8 GEMM at 1.44
TOPS on a 16-core Phoenix NPU. Further gains are fabric/overhead tuning (larger descriptors,
deeper pipelining), not a new dataflow.

Repro: `python3 r14_gen.py 6 12 16 1024 > r14.mlir`; build `r11.o` (LM=6 LN=12 KT=16); aiecc;
dispatch (A=4·N·6144, W=4·N·12288, C=4·N·9216·4 bytes, macs=16·N·6·12·16·512, expect 256).
