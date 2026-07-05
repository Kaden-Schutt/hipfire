# R10 — streaming GEMM with real DMA feed: the rate is intensity-bound

R9 measured 150 GMAC/s/core with the working set L1-**resident** (DMA'd once, reused).
R10 is the first *real* streaming GEMM: the R9 3×3 load-reuse block kernel (`r10_gemm.cc`),
but fed a **fresh A/W block from DDR every iteration** through double-buffered objectfifos
(`r10_stream_gen.py`), so the measured rate includes the shim→L1 DMA feed.

Single core, N_BLK=1024 blocks/dispatch, int4 `<4,16,8>`, correctness verified (c0 = KT·16):

| tile (MT×NT×KT) | intensity | GMAC/s/core |
|---|---|---|
| 2×2×64 | 8 mac/B | 48.3 |
| 3×3×16 | 12 mac/B | 69.7 |
| 3×3×32 | 12 mac/B | 66.9 |
| 3×3×64 | 12 mac/B | 71.0 |

Real feed **halves** the R9 rate (150 → ~70) and it is cleanly **DMA-intensity-bound**:
rate scales ~linearly with arithmetic intensity (8→12 mac/B gives 48→70 GMAC/s), i.e.
the single core pulls ~6 GB/s from the shim and delivers `~6 GB/s × intensity`. A bare
3×3 register tile only reaches intensity `NT·4 = 12`, so it caps at ~70.

**The fix (R11):** raise intensity with a second tiling level — hold a larger L1 output
block and sweep the 3×3 register tile over it, reusing A/W stripes DMA'd once per block
(intensity `8·LM·LN/(LM+LN)`). See `../r11/`, which recovers the rate to ~143 GMAC/s.

Repro: `./r10_run.sh 1024 80 3x3x16 3x3x64 2x2x64`.
