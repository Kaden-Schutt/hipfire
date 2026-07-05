# R12 — the real array number: DDR-bandwidth-bound at ~1 TOPS (distinct-data GEMM)

R11 hit 143 GMAC/s/core single-core (95% of the compute ceiling). R12 scales the L2-tiled
streaming GEMM across the npu1 array: `r12_gen.py` places a COLS×ROWS core grid, each core
running `r11_gemm` on its OWN distinct A/W stream from DDR (no cross-core reuse — the
feed-heavy lower bound), and reports aggregate GMAC/s. Block 6×12×16 (intensity 32), correct
throughout (c0 = KT·16).

## Column scaling (1 core/column, direct shim feed)

| grid | cores | GMAC/s aggregate | GMAC/s/core |
|---|---|---|---|
| 1×1 | 1 | 118 | 118 |
| 2×1 | 2 | 228 | 114 |
| 3×1 | 3 | 247 | 82 |
| 4×1 | 4 | **375** | 94 |

Near-linear to 2 cores, then bends: **4 cores = 375 GMAC/s = 0.75 TOPS**. At intensity 32
that is 375/32 ≈ **11.7 GB/s of DDR** — right at R6's ~13–16 GB/s shim/DDR ceiling. So the
array is **DDR-bandwidth-bound**, not compute-bound: the shared DDR→shim path saturates by
~4 cores.

## Why 16 direct-feed cores don't help (two independent limits)

1. **Shim DMA channels.** Each column's shim has ~2 MM2S channels — enough to feed A+W to
   **one** core. `4×2` fails aiecc with *"number of output DMA channel exceeded"*. Feeding
   4 cores/column requires **memtile routing** (shim→memtile→cores, the whole_array pattern).
2. **DDR bandwidth.** Even with memtile routing, distinct-data feed saturates DDR at ~4–8
   cores (already ~12 GB/s at 4), so more cores can't raise aggregate past
   `~15 GB/s × intensity 32 ≈ ~1 TOPS`.

So the honest **real array number for a distinct-data W4A8 GEMM on XDNA1 is ~0.75 TOPS
measured (4 cores), ~1 TOPS DDR-saturated ceiling** — ~1.3–1.8× the R5 cascade line, and it
explains R5–R7: every one of those dataflow variants was already near the DDR wall for its
intensity, which is why none of the cascade/pipelining knobs moved it.

## The one lever left to exceed ~1 TOPS

Raise the *effective array intensity* with **cross-core broadcast reuse**: tile C across the
16 cores so each A-stripe is broadcast to a row of cores and each W-stripe to a column, cutting
DDR traffic ~4× (each stripe fed once, reused by 4 cores) — the mlir-aie `whole_array`
dataflow that reaches ~15.7 TOPS on aie2p (`../findings.md`). That is the natural continuation:
a memtile-routed 4×4 GEMM with 2D broadcast, where DDR intensity becomes ~4× (128+ mac/byte)
and the array can approach the compute roofline instead of the DDR floor.

## Full line (R5→R12), all on gfx1103 XDNA1

| stage | result |
|---|---|
| R5 cascade streaming | 0.56 TOPS (guessed sync-bound) |
| R6/R7 KSLICE + multi-accum | disproved sync, then wrongly "op floor" |
| R8 resident microbench | matrix unit ~489 GMAC/s/core (~15 TOPS); it was **load**-bound |
| R9 3×3 register tiling | 150 GMAC/s/core resident (8.6× R5) |
| R10 naive streaming | ~70 GMAC/s/core (DMA-intensity-bound) |
| R11 L2 tiling (6×12) | **143 GMAC/s/core**, 95% of ceiling on real feed |
| R12 array (4 cores) | **375 GMAC/s = 0.75 TOPS**, DDR-bound; ~1 TOPS ceiling |

Repro: `./r12_run.sh 6x12x16 128 60 1x1 2x1 4x1`.
