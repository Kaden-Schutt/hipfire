# R11 — L2 tiling recovers the streaming rate: 143 GMAC/s/core (95% of the ceiling)

R10 showed the real streaming GEMM is DMA-intensity-bound (~70 GMAC/s at a bare 3×3
register tile, intensity 12). R11 adds the second tiling level: `r11_gemm.cc` holds an
**LM×LN base-tile output block's C in L1** and sweeps the 3×3 register tile over it, so
the A/W stripes DMA'd once per block are reused across all sub-tiles. DMA intensity rises
to `8·LM·LN/(LM+LN)` mac/byte, climbing back toward the register-tiling compute rate.

Single core, real DDR feed, int4 `<4,16,8>`, correctness verified (c0 = KT·16):

| L1 block (LM×LN×KT) | intensity | L1 C (KB) | GMAC/s/core |
|---|---|---|---|
| 3×3×16 (= R10) | 12 | 2.2 | 60.9 |
| 6×6×16 | 24 | 9.0 | 79.4 |
| 9×9×16 | 36 | 20.2 | 107.8 |
| 6×6×32 | 24 | 9.0 | 118.6 |
| 8×8×20 | 32 | 16.0 | 114.8 |
| 12×6×16 | 32 | 18.0 | 136.1 |
| **6×12×16** | **32** | 18.0 | **143.0** |

**6×12×16 → 143 GMAC/s/core = 95% of R9's 150 L1-resident ceiling.** L2 tiling fully
amortizes the DMA feed: at intensity 32 the ~4.5 GB/s of shim the block needs is small
enough that the core runs at its 3×3 register-tiling compute limit. Two second-order
effects: deeper K (KT) amortizes per-sub-tile accumulator fill/drain (6×6×32 > 9×9×16
at *lower* intensity), and wide-N blocks win (6×12 > 8×8 > 12×6 at equal intensity 32) —
more N-columns reused per A load. L1 (64 KB) caps the block near here.

## Integration verdict (single core → array)

The load-reuse chain is proven end-to-end on real feed: **R5 cascade 0.56 TOPS → R10
naive streaming ~70 GMAC/s → R11 L2-tiled 143 GMAC/s/core** (≈2.6× the effective R5 rate,
95% of the matrix-unit-fed ceiling). The kernel is a standard two-level-tiled register
GEMM; the R5–R7 cascade/pipelining machinery was all downstream of the load wall and is
unused here.

**The remaining wall is the shared shim DMA (array-level).** One core at 143 GMAC/s pulls
~4.5 GB/s. The whole 16-core array shares only ~13–16 GB/s aggregate (R6), so it cannot
feed 16 cores at 4.5 GB/s each — to keep all 16 at the compute rate would need intensity
~150 mac/byte, far beyond what L1 (64 KB) can hold. So the realistic **array** ceiling for
a distinct-data W4A8 GEMM is shim-bound at roughly `~15 GB/s × intensity ≈ 1–1.5 TOPS`
(intensity 32–48), still ~2–3× the R5 cascade line. Confirming the exact array number
needs a 16-core build that distributes the output tiles across cores and shares the shim —
the honest next step; the single-core kernel and the roofline are now established.

Repro: `./r11_run.sh 512 80 3x3x16 6x6x16 9x9x16 6x12x16`.
