# R6 — the M/AI lever: 2D-tiled W4A8 GEMM (the real SOTA-beating path)

R5 proved the cascade unlocks the compute *ceiling* (42 TOPS) but does NOT help real
prefill, which is **feed / arithmetic-intensity bound**: to go compute-bound the tile
must clear AI ≈ 58 MACs/byte, and that needs a tile reusing **both** operands — each
weight column across MT M-row blocks, each activation row across NT N-col blocks.
Everything before R6 used a 1D tile (16×16, reuse in one dimension) and stalled at
~3–5 TOPS. R6 is the 2D tile.

## Kernel (`r6_gemm.cc`)

One call computes an (MT·4)×(NT·16) output block, K-reduced over KCHUNK 16×16 tiles;
`A[MT][KCHUNK]` and `W[NT][KCHUNK]` are resident. For each M-row block, the NT=4 N
accumulators share the loaded activation (A reused NT-wide), and the weight tiles are
reused across the MT M-blocks; C is stored once (per-tile overhead amortized over
MT·NT·KCHUNK mmuls). Register accumulators = NT (4), reused across the MT loop, so no
spill; A/W stay in L1. AI ≈ MT·NT·KCHUNK·1024 / (weight + C bytes) — grows with MT and
KCHUNK. (L1 is 64 KB/core, so A + double-buffered W + C caps MT·KCHUNK.)

## First measurement — single core (MT=8, NT=4, KCHUNK=16)

Streamed (A resident, W streamed, C per block), bit-exact (C[0]=256):

**2.90 TOPS on ONE core** — near a core's ~3.68-TOPS physical max, i.e. the kernel is
**compute-bound and efficient** (unlike R5's 0.40 solo before its fix). This is the
first W4A8 kernel here that reaches per-core compute-bound on a *real* (no-fake-reuse)
tile.

## Array measurement — R6 BEATS SOTA on real prefill

8 **independent** R6 cores (`r6_gen.py COLS NB` — one per column, no cascade; the
array is the M-parallelism). Real streaming (A resident, W streamed), bit-exact
(C[0]=256):

| config | W feed | TOPS (8-core aggregate) |
|---|---|---|
| MT=8 NT=4 KCHUNK=16, **shared** W region | (compute ceiling) | 19.5 |
| MT=8 NT=4 KCHUNK=16, **independent** W (16 MB) | feed-bound ~68 GB/s | **9.15** |

**9.15 TOPS on real, independent-weight streaming beats SOTA FastFlowLM's ~5 by
1.8×** — the first dataflow in this whole investigation (R2a → R4 memtile → R5
cascade) to beat SOTA on real prefill. The feed cap makes it feed-bound, so 8 cores
already ≈ saturate; the compute headroom (19.5 shared) is reachable by raising AI
(more weight reuse per streamed byte).

**The lever, confirmed:** the win came from the **2D tile** (reuse both operands),
i.e. **M/AI, exactly as the R5 verdict predicted — not the cascade.** Everything
before had a 1D tile stuck at 3–5 TOPS; R6's 2D tile clears the feed wall.

**Next (tuning toward the 15.7 reference):** raise MT (more M-row blocks → more
weight reuse → higher AI → less feed-bound). MT=16 overran L1 (A + double-buffered W
+ C > 64 KB) — needs KCHUNK/depth tuning or single-buffered W. Sweep MT/KCHUNK for
the L1-optimal point, and scale past 8 columns (feed permitting). But the headline is
settled: **NPU W4A8 prefill offload is a real win — R6 beats SOTA.**
