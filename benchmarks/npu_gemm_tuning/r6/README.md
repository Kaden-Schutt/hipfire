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

## Next — the array measurement (the payoff)

32 **independent** R6 cores (no cascade — each does its own M×N tile; the array is the
M-parallelism). Single-core wants ~22 GB/s of weight feed; 32 cores would need ~700
GB/s but the fabric caps at ~55 GB/s, so the array goes **feed-bound at 55 GB/s** — at
this tile's effective AI (~66, A resident) that projects to **~7 TOPS aggregate, above
SOTA's ~5**. Build a 32-core independent-R6 array (low-level MLIR generator, like the
R5 array but no `cascade_flow`), sweep MT/KCHUNK to push AI, and measure the real
aggregate vs SOTA ~5 and the 15.7 reference. That is the number that decides whether
NPU prefill offload is a real win.
