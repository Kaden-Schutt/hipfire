# R9 — load-reuse register tiling: 8.6× over R5, the real XDNA1 GEMM lever

R8 showed the matrix unit runs int4 `<4,16,8>` at ~489 GMAC/s/core (register-resident)
while R5's streaming hit only 0.56 TOPS — the wall is L1 **loads** (the K-loop did 2
loads per mac, zero reuse). R9 tiles the OUTPUT MT×NT (in base-op tiles): each K-step
loads MT A-tiles + NT W-tiles and issues MT·NT macs, so each A row is reused across NT
columns and each W tile across MT rows. Reuse = MT·NT/(MT+NT) macs-per-load. Single-core
microbench (`r9_ubench.cc` + `r9_1core.mlir`, hot loop streams the tiled working set from
L1 — real load traffic); `r9_run.sh` sweeps the tile.

## Sweep (aie2/XDNA1, int4 `<4,16,8>`, 512 MAC/op)

| tile | accs | reuse | ns/mmul | GMAC/s/core |
|---|---|---|---|---|
| 1×1 | 1 | 0.50 | 16.95 | 30.2 |
| 1×4 | 4 | 0.80 | 6.48 | 79.0 |
| 2×2 | 4 | 1.00 | 5.94 | 86.2 |
| 2×3 / 3×2 | 6 | 1.20 | 4.6 | ~112 |
| 2×4 | 8 | 1.33 | 3.95 | 129.5 |
| **3×3** | **9** | **1.50** | **3.41** | **150.0** |
| 2×6 | 12 | 1.50 | 6.62 | 77.3 |
| 3×4 | 12 | 1.71 | 6.39 | 80.2 |

**3×3 is the optimum: 150 GMAC/s/core**, an **8.6× jump** over R5's 0.56-TOPS streaming
(≈35 GMAC/s effective) and ~31% of the R8 register-resident ceiling. Two clear regimes:

- **1×1 (no reuse) = 17 ns/mmul, 30 GMAC/s** — reproduces the R5–R7 load-bound rate,
  confirming that was always the loads, not the op.
- **Reuse pays down the load cost** monotonically until the **accumulator register file
  caps the tile at 9 base-tiles**: 3×3 (9 accs × size_C=32 = 288 acc32) fits, but 2×6 and
  3×4 (12 accs = 384) spill and collapse back to ~78 GMAC/s. Square tiles win — best
  reuse per accumulator. (Same spill wall as R7's NACC=8.)

## Bottom line for the XDNA1 NPU line

Array projection: 150 GMAC/s/core × 16 cores ≈ **~4.8 TOPS** for W4A8 GEMM — vs the
0.56 TOPS the cascade line plateaued at. The five rounds relocated the wall three times
(sync → op → **loads**) and the fix is the classic register-tiled matmul, not any of the
cascade/streaming/pipelining machinery. Remaining gap to the ~15-TOPS matrix ceiling is
the accumulator file (reuse capped at 1.5); past that, shim→L1 DMA feed (~13–16 GB/s,
R6) is the next limit.

**Next (integration, not microbench):** port the 3×3-tiled inner loop into a real
streaming GEMM (tile A/W blocks through the objectfifos with enough K-depth that the DMA
feed amortizes) and measure end-to-end TOPS with real feed — that converts this ~4.8-TOPS
microbench rate into a usable W4A8 NPU GEMM. The cascade K-split (R5) is orthogonal and
almost certainly unnecessary now that the per-core rate is load-tiling-bound.

Repro: `./r9_run.sh 16 20000 80 1x1 2x2 2x4 3x3 3x4`; default build is the 3×3 optimum.
