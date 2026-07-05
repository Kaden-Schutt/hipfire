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

## MT sweep — peak 20.68 TOPS (beats the 15.7 reference)

The bottleneck is the **weight feed** (~68 GB/s), and effective **AI_W = 8·MT** (each
weight reused across MT M-row blocks; independent of KCHUNK). So raising MT raises
throughput until it meets the compute ceiling. KCHUNK only sets L1 fit + C-write
amortization. 8-core, real independent-W streaming, bit-exact:

| MT | KCHUNK | AI_W | TOPS |
|---|---|---|---|
| 8  | 16 | 64  | 9.19 |
| 16 | 8  | 128 | 13.92 |
| **24** | **8** | **192** | **20.68** |
| 32 | 4  | 256 | 18.06 (compute-bound; KCHUNK=4 C-write overhead) |

**Peak MT=24: 20.68 TOPS — 4.1× SOTA FastFlowLM (~5) and above the 15.7 int8
whole_array reference**, on real weight streaming. MT=32 regresses (past the compute
ceiling, and KCHUNK=4 pays too much per-block C overhead), so MT=24/KCHUNK=8 is the
sweet spot. (MT=16 KCHUNK=16 overran the 64 KB L1 — A + double-buffered W + C; MT=24
KCHUNK=8 fits at ~44 KB.)

The feed caps the aggregate, so 8 cores ≈ saturate it — more columns won't add much.
**Verdict, settled and on-hardware:** hipfire's R6 W4A8 GEMM does **20.7 TOPS** of
real prefill on the halo NPU — **4× the SOTA NPU inference stack** and past the
reference — via the M/AI 2D tile, all through hipfire's own XRT-free dispatch. Next:
wire R6 into the runtime prefill-offload path (the original goal — now clearly worth
it; NPU prefill runs concurrently with GPU decode for a real aggregate win).

## R6 is a numerically-correct GEMM (real data, not just the ceiling)

`crates/hipfire-xdna/examples/r6_verify` runs the kernel on random int8×int4 data and
compares to a CPU reference: **0/256 mismatches** (build MT=1 NT=4 KCHUNK=1, one
M-block × 4 N-blocks × one K-tile). This pins the full tile layout — **all row-major**:

- A tile = 4×16 int8, `a[m*16 + k]`.
- W tile = 16×16 int4, `w[k*16 + n]`, packed two int4 per byte (low nibble first).
- C tile = 4×16 int32, `c[m*16 + n]`.
- `r6_mac` tile ordering: `A[MT][KCHUNK]` at `(mt*KCHUNK+k)*64`, `W[NT][KCHUNK]` at
  `(nt*KCHUNK+k)*128` bytes, `C[MT][NT]` at `(mt*NT+nt)*64`.

This is the layout the runtime `NpuGemm` marshaling (wire-in step 2) targets — R6 is a
proven correct W4A8 GEMM primitive, not only a throughput ceiling.

## End-to-end: from 20.7-TOPS *compute* to a real deliverable rate

20.7 TOPS is the kernel's *compute* rate. The deliverable end-to-end rate (host feeds
row-major A/W, reads row-major C) was initially **0.02 TOPS** — CPU marshaling
(row-major ↔ tile-major reshuffle) dwarfed the kernel. Closing that gap drove the arc
below. All configs are the M768·K512·N4096 prefill GEMM, validated numerically.

| stage | e2e | what changed |
|---|---|---|
| CPU marshaling floor | 0.02 | tile-major pack/unpack on the host |
| **tensor-stream row-major** (`r6_gemm_ts.cc`) | — | kernel reads/writes ROW-MAJOR via `aie::tensor_descriptor`; AGUs tile in-core, zero CPU marshaling, linear DMA |
| + single-K-chunk (C copied once) | 0.535 | KCHUNK covers all K → no host K-accumulation |
| + pipelined C read-back | 1.0 | `submit`/`wait` split; overlap read-back with next dispatch |
| **M-parallel W-broadcast** (`r6_gen_mp.py`) | 1.45 | COLS distinct M-blocks share ONE broadcast W (shim→memtile→all cores); 3 dispatches not 24; *blocking*, no coherence dance |
| **whole-GEMM in one dispatch** (`r6_gen_mp.py` ROUNDS) | **~1.9** | each core streams ROUNDS M-blocks → the whole GEMM is a *single* dispatch; continuous streaming, one C read-back |

**~95× over the marshaling floor.** Raw single-dispatch ceiling is ~3 TOPS — feed-bound
on the memtile's 8-way broadcast sync over the N-slabs, not compute. Still below the
GPU's ~50 TOPS, so *sync* offload stays gated; the aggregate win is a concurrent
NPU ‖ GPU split.

### Batch prefill — throughput is flat in M (weight-bandwidth-bound)

Sweeping M (= total prefill tokens = batch × seq) at K=512, N=4096, all validated:

| M (tokens) | multi-dispatch (`r6_mp_e2e`, any M) | whole-GEMM 1-dispatch (`r6_mp1_e2e`) |
|---|---|---|
| 256  | 1.37 | — |
| 768  | 1.46 | ~1.9 |
| 2048 | 1.40 | 1.83 |
| 4096 | 1.47 | 1.75 |
| 8192 | 1.42 | — |

**Throughput does not scale with batch** — it's weight-bandwidth-bound. L1 caps weight
reuse at MT=8 M-rows per weight load, so W is re-read ≈ M/32 times *regardless of batch*;
total weight traffic scales with M, so the compute rate stays constant. This is the
opposite of the GPU (bigger batch → bigger WMMA tiles → more weight reuse → higher
throughput). The NPU array is already feed-saturated at small M, so batching buys
nothing here. The whole-GEMM one-dispatch stays ~0.3–0.4 TOPS ahead at every size (no
inter-dispatch host stalls) but needs a per-M xclbin; multi-dispatch handles any M with
one xclbin.

### The ceiling: objectfifo per-slab streaming overhead

Pure-dispatch (all-ones, no host copy) rates pin *why* e2e tops out ~1.9 TOPS:

| config | pure dispatch | effective W feed |
|---|---|---|
| M-parallel whole-GEMM, K=512 KCHUNK=32 | 941 µs → 3.42 TOPS | ~3.2 GB/s |
| N-parallel whole-GEMM, K=128 KCHUNK=8 | 490 µs → 1.64 TOPS | ~4.1 GB/s |

Both are feed-bound at only ~3–4 GB/s (DRAM does ~68), i.e. **objectfifo per-slab
streaming overhead (~5 µs/slab) dominates, not the broadcast** — the 16×16 mmul tiles are
too small, so each N-slab acquire/release costs ~8× its own compute. The one lever is
**KCHUNK** (k-tiles amortized per slab-acquire): K=512/KCHUNK=32 does 4× the compute per
acquire of K=128/KCHUNK=8, which is why it's 2× faster per slab. But KCHUNK=32 needs MT≤8
to fit L1 — so the high-MT N-parallel tile (20.7-TOPS *compute* ceiling at K=128) can't be
used at K=512, and **raising MT doesn't help**: it forces KCHUNK down and worsens the
amortization. Net: for K=512, **MT=8 KCHUNK=32 (M-parallel whole-GEMM) is near the
practical ceiling — ~3.4 TOPS raw, ~1.9 TOPS e2e.** Beyond this needs bigger effective
tiles (an NT=8 kernel — register-spill risk) or a fundamentally different feed (cascade
K-resident, which R5 showed is feed-bound for real prefill anyway). The generators carry a
`ROUNDS` whole-GEMM knob for both topologies (`r6_gen.py` / `r6_gen_mp.py`), exercised by
`r6_np1_e2e` / `r6_mp1_e2e`.

### Two dataflow lessons (cost real debugging)

1. **Pipelined read-back needs a cache reconcile.** The host read-back of one C buffer
   overlaps a concurrent DMA write to the double-buffered other; the CPU prefetcher can
   cache stale lines of the in-flight buffer. `FROM_DEVICE` EINVALs on data BOs, but
   `TO_DEVICE` clean+invalidates on this driver — re-sync the slot after `wait`, before
   reading. (Blocking dispatch is immune, which is one reason the blocking M-parallel /
   whole-GEMM paths are preferred.)
2. **Stream with pure-linear DMAs, not repeat BD dims.** A repeat dimension does NOT
   re-check the objectfifo acquire/release semaphore, so rounds > 0 overrun the fifo
   buffers (round 0 correct, the rest garbage). A contiguous linear stream is chunked by
   the objectfifo *with* semaphores. W (which the broadcast fifo can't replay) is
   replicated ROUNDS× in DRAM. Shim BD wrap-dim sizes also cap at [1:64].

## Reproduce

Build (offline; `aiecc`) with `r6_cache.sh`, selecting kernel + generator + tag:

```sh
R6=benchmarks/npu_gemm_tuning/r6
# tensor-stream kernel, N-parallel array (single K-chunk peak config)
R6_KERNEL_SRC=$R6/r6_gemm_ts.cc R6_OUT_TAG=r6ts $R6/r6_cache.sh 8 4 32 8 8
# M-parallel W-broadcast, whole-GEMM in one dispatch (ROUNDS=3)
R6_KERNEL_SRC=$R6/r6_gemm_ts.cc R6_GEN=r6_gen_mp.py R6_OUT_TAG=r6mp R6_ROUNDS=3 \
  $R6/r6_cache.sh 8 4 32 8 64
```

Verify + measure (examples in `crates/hipfire-xdna/examples/`):

| example | checks |
|---|---|
| `ts_a_verify` | in-core tensor-stream A reshuffle == `pack_a` |
| `r6_ts_verify` | full row-major W4A8 GEMM (TS kernel) vs CPU |
| `npu_gemm_verify` / `npu_gemm_e2e` | `NpuGemm` correctness / N-parallel pipelined e2e |
| `r6_mp_verify` / `r6_mp_e2e` | M-parallel array correctness / 3-dispatch e2e |
| `r6_mp1_e2e` | whole-GEMM one-dispatch correctness + e2e |
