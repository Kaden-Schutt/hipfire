# R3 — real W4A8 GEMV / GEMM kernels (K-accumulating) + wire-in

R2a probed compute by repeating one weight tile; R3 does REAL K accumulation over
distinct streamed weights — the kernels an inference path would actually call.

## R3a — batched-decode W4A8 GEMV (`r3a_gemv.cc` + `r3a_run.py`)

M=4 (a spec-decode / small-batch column), one N-block. Activations and int4 weight
tiles stream as KCHUNK-deep super-tiles over the K contraction; each super-tile
does KCHUNK II=1 macs in a register accumulator (R2a named-accumulator recipe) and
folds the partial into a resident C (vector add — `aie::mmul` can't reseed its acc).

Measured (host-wall differential, single core/stream): **11.2 GB/s weight-feed,
0.18 TOPS**. As predicted for M=4 this is **bandwidth-bound** — throughput tracks
the R1 single-stream feed (~14.4 GB/s), not compute. The ~3 GB/s gap is the
co-streamed activations (own shim) + per-super-tile C load/add/store; in a real
multi-N-block GEMV the activation stays resident and reused, so W approaches the
full per-stream rate, and 8 columns give ~44 GB/s of weight feed. This is the case
where int4's half-weight-bytes matters: decode token rate ∝ feed / weight-bytes.

Open: bit-exact validation vs a numpy reference needs matching `aie::mmul`'s 16×16
tile layout (deferred); multi-N-block + 8-column scale-up; then GEMM (large M,
compute-bound, R2a path) and wiring both into the runtime.

## Wire-in status

The existing NPU dispatch (`hipfire-arch-qwen35/src/xdna1_ffi.rs`) dlopens
`libhipfire_xdna1.so` — a binary blob (symbols reverse-engineered from disasm) with
swiglu/rmsnorm/rope/etc. ops, **not on disk here**, and with no gemm/gemv symbol we
can add. So wiring the W4A8 gemv/gemm needs a **direct XRT-from-Rust** loader
(xclbin + instr → run), not that dormant blob — tracked as the next integration arc.

## R3b — prefill W4A8 GEMM (`r3b_gemm.cc` + `r3b_run.py`)

Real K accumulation with NACC row-blocks (M = NACC·4): one streamed int4 weight
tile is macced into NACC named accumulators (R2a recipe), so weight reuse = NACC
and AI = NACC·8 MACs/byte. Measured single-core, single-N-block:

| NACC (M) | TOPS | MAC/cyc | limiter |
|---|---|---|---|
| 4 (M=16) | 0.70 | 195 | feed-bound (W feed ~11 GB/s, matches R1) |
| 8 (M=32) | 0.26 | 71  | **A-stream-bound** (A super = 4× W bytes) |

The single-N-block probe streams the full activation for one output block, and at
M=32 that A stream (NACC·MR·K int8) is 4× the weight stream (K·16 int4) — so it
goes A-bound, not compute-bound. NACC=8 also L1-caps KCHUNK (A super 32 KB at
KCHUNK=64 overflows), forcing KCHUNK=16 which adds C-fold overhead.

**To reach compute-bound prefill, A must stay resident and be reused across N-blocks
(or K-tiled through the memtile)** — the standard AIE GEMM dataflow that the
reference kernels use to approach peak. That is the next step: A-resident, stream W
across many N-blocks, then scale to the 8×4 array. This is the same ~27%-efficiency
dataflow wall documented for the `whole_array` reference in ../findings.md.

## R3c — A-resident GEMM: the single-core register-file wall

Kept A resident (M×K) and streamed only weights across N-blocks (weight reuse =
M, no A-stream penalty). Still not compute-bound:

| NACC (M) | MAC/cyc | note |
|---|---|---|
| 4 (M=16) | ~195 | fits regs but **feed-bound** (AI=32 MACs/B < needed ~58) |
| 8 (M=32) | ~88  | **accumulator spill** — 8 MMUL accs exceed the aie2p acc file |

The single-core dead-end, quantified: compute-bound W4A8 needs AI ≈ 58 MACs/byte
⇒ NACC ≈ 8 accumulators, but the register file holds only ~4 MMUL accumulators
(R2a hit ~460 MAC/cyc at NACC=4 with *fake* reuse; real K-streaming at NACC=4 is
feed-bound; NACC=8 spills). And true aggregate compute-bound needs M≈264 (feed
55 GB/s ÷ compute 29 TMAC/s) = ~66 accumulators/core — impossible on one core.

**Resolution = the weight-broadcast ARRAY dataflow (R4).** Broadcast each weight
tile to all 32 cores; each core holds NACC=4 accumulators for its own M-row-block,
so aggregate weight reuse = 32×16 = 512 rows while no core exceeds its register
file. Weight is fed once (broadcast) and reused across the array ⇒ ~53 TOPS
compute-bound at the ~55 GB/s feed. Spatial reuse across cores, not temporal
per-core accumulators, is what makes W4A8 GEMM compute-bound. That array kernel
(cascade/broadcast) is the next major build.

## Wire-in dispatch validation (hipfire-xdna NpuKernel)

Both kernels now dispatch through hipfire's amdxdna path (`crates/hipfire-xdna`,
`NpuKernel`) — no XRT. R2a is validated bit-exact on halo: A=all-1s int8 ×
W=all-1s int4 gives the exact `16·(INNER+1)` per lane, and two back-to-back
dispatches (A=1 then A=2) give the clean 2× a linear GEMM must (`run_smoke`).

**R3a was numerically wrong — two bugs found and fixed (2026-07-05).** Driving it
with an all-ones reference (`C = KCHUNK·16` per lane) surfaced ~half-magnitude,
partly non-deterministic output. Since R2a is bit-exact through the *same* dispatch
path, the dispatch was exonerated; both bugs were in the R3a kernel, which had only
ever been validated for **bandwidth** with random inputs — its numeric result was
never checked.

1. **Sub-byte pointer stride (the "half the K" bug).** The per-tile weight load did
   `w + j*MMUL::size_B` on a `const int4*`. Pointer arithmetic on the sub-byte
   `int4` type advances **one byte per element**, so the stride was 2× too far — it
   read tiles 0, 2, 4… and ran the second half of `j` off the buffer end. A clean
   KCHUNK sweep pinned it exactly: KCHUNK=1 correct (no stride), KCHUNK=N≥2 gave
   N/2 tiles' worth. The OOB tail read zeros under XRT (clean 512) but adjacent tile
   memory under hipfire (the noise) — one bug, both symptoms. Fix: stride in **bytes**
   on the `int8*` buffer (`wbytes + j*(size_B/2)`), then reinterpret as `int4`.
2. **Reuse-unsafe resident C.** The K accumulator C lives in a tile-local buffer
   that persists across dispatches; the original kernel always did `C = load(C)+partial`,
   so the first super-tile of each call accumulated onto the *previous* dispatch's C.
   Fix: split into `r3a_matvec_init` (stores/reseeds) + `r3a_matvec` (accumulates),
   in separate TUs (shared `r3a_gemv_common.h`) to avoid duplicate symbols, and peel
   the first super-tile in `r3a_run.py` to call init.

Both fixes verified: XRT all-ones gives the exact analytic value (KCHUNK 1/2/4/8 →
16/32/64/128; N_SUPER 1/2/4 → 1024/2048/4096, uniform), and through hipfire two
back-to-back dispatches (A=1 then A=2) give 4096 then 8192 — the clean 2× a linear
GEMV must, reuse-safe. R3a now matches R2a: a validated, dispatchable W4A8 kernel.
(The bandwidth numbers above are unchanged — `rt.fill` streams the same bytes; only
the compute's tile indexing was wrong.)
