# Wiring R6 W4A8 into the runtime prefill-offload path

## Why now

R6 (2D-tiled W4A8 GEMM, `benchmarks/npu_gemm_tuning/r6/`) does **20.7 TOPS of real
prefill on the halo NPU** — 4× the SOTA NPU inference stack (FastFlowLM ~5) and past
the 15.7 int8 reference — through hipfire's own XRT-free amdxdna dispatch
(`crates/hipfire-xdna`). Against the gfx1151 GPU's ~50 TOPS real W4A8, running NPU
prefill **concurrently** with GPU work is a genuine ~+40% aggregate win, not a
rounding error. The kernel is proven; this doc scopes making the runtime use it.

## The seam

The runtime issues quantized linears through `crates/hipfire-rdna/src/dispatch/quant.rs`
(+ `gemm_misc.rs`, `gemv.rs`). A W4A8 prefill linear is a GEMM `C[M,N] = A[M,K]·W[K,N]`
with M = prompt length (large), W = the oq4/mq4 weight. That is exactly R6's shape.
The offload decision belongs **at this dispatch seam**: for an eligible prefill GEMM,
route to the NPU instead of (or concurrently with) the GPU iu4 kernel.

## Build steps (each independently testable, none touches the hot path until step 4)

1. **Offline xclbin build (tooling, not hot path).** R6 is shape-specialized
   (MT/NT/KCHUNK compiled). Extend the `r6_gen.py` + `r6_build.sh` flow into a small
   offline tool: given a model's distinct `(K, N)` linear shapes, emit + cache R6
   xclbins (`~/.hipfire/npu/<shape>.xclbin`). Python/aiecc stays offline — the
   inference binary only *loads* cached bytes (AGENTS.md: no Python in the hot path;
   compat/build tooling lives outside the daemon).

2. **`NpuGemm` primitive (`hipfire-xdna`, isolated + tested).** A general
   `npu_gemm(M, K, N, a_i8, w_oq4, c_i32)` that: tiles M/N/K into R6's
   (MT·4)×(NT·16)×KCHUNK blocks, marshals A/W into the tile-major SHMEM layout the
   kernel expects, K-accumulates across KCHUNK chunks (add into C), and drives the
   dispatches via `NpuKernel`. Validate **numerically vs a CPU int8×int4 reference**
   on real (non-all-ones) data — the current benches only prove the all-ones ceiling.
   This is the unit the runtime calls; it carries no hot-path risk.

3. **Marshaling splits by static vs dynamic — and the layouts do NOT match the GPU.**
   Measured (`prepack_weights`/`run_packed`): CPU marshaling dominates end-to-end
   (0.02 TOPS). Verified against the GPU iu4 kernel (`fused_qkvza_oq4_wmma.hip`): it
   stores W as `[N_out, K/2]` nibbles + per-group f32 scales, loaded through the WMMA
   lane-distributed fragment; R6 wants `[K, N]` 16×16 aie2p tile-major raw int4. So
   **the buffer cannot be shared** — different orientation (transposed), different
   tiling (WMMA fragment vs aie2p mmul tile), and R6 applies **no scales**. Transposing
   either kernel to match orientation doesn't help: it only moves the transpose onto
   the *dynamic* activation, and the tiling still differs. Split:

   - **3a. Weights → the loader (static, once).** Produce an NPU-specific weight buffer
     at load: read GPU `W[n][k]` → write NPU `[k][n]` 16×16 tile-major + int4 pack. The
     `[N][K]→[K][N]` transpose is *absorbed for free* into the re-tile index mapping (no
     separate pass). Store it alongside the GPU copy for offloaded layers (4-bit, small).
     The hot path then DMAs it linearly — zero per-inference weight work. `prepack_weights`
     is the reference impl; move it into the loader / NPU-quantize path.
   - **3b. R6 scale handling (a real correctness item).** R6 is a raw int4×int8 GEMM;
     the `0/256` validation used *unscaled* int4. oq4 carries per-group f32 scales, so
     the NPU path must accumulate int32 then apply the group scale (at the tail, per
     group) to match the GPU. Design this with 3a (scales travel with the arranged W).
   - **3c. Activations + output → the DMA (dynamic, per inference).** A and C are
     computed at runtime, so the loader can't pre-arrange them. Feed A row-major and
     let the shim DMA tile it (`dims_to_stream` on the `dma_bd`); write C tiled and let
     the DMA de-tile — the tile reshuffle happens in hardware, not the CPU. This is how
     IRON's whole_array gemm avoids CPU marshaling; my R6 MLIR uses plain linear
     `dma_bd`, which is why marshaling landed on the CPU. Plus stream the whole GEMM in
     one dispatch to amortize the ~78 µs latency.

4. **Runtime offload hook (the hot-path change — smallest possible).** In
   `dispatch/quant.rs`, add an opt-in path: if a prefill W4A8 GEMM is large enough to
   amortize the ~78 µs dispatch latency AND an R6 xclbin is cached for its shape,
   dispatch it on the NPU. Gate behind a flag first (`HIPFIRE_NPU_PREFILL`), measure
   end-to-end, then consider default-on.

## CRITICAL measured finding — marshaling, not the kernel, is the bottleneck

Array `NpuGemm` is validated correct, but end-to-end it is **catastrophically slow**:
`NpuGemm::run` on M=768 K=512 N=4096 (peak config, 32 dispatches) = **351 ms/run =
0.01 TOPS** (result numerically correct). The kernel computes at 20.7 TOPS in ~µs; the
**CPU marshaling** (re-shuffling row-major A/W into the tile-major int4 SHMEM layout,
per-element bit-packing) takes ~348 ms and dwarfs everything. Dispatch latency is
~78 µs × 32 = 2.5 ms — also non-trivial but small next to marshaling.

**So the wire-in's hard problem is marshaling + dispatch overhead, not the kernel.**
The fixes, in impact order:
1. **Pre-marshal weights once at load** (weights are static): re-marshaling W every
   dispatch is most of the 348 ms. Marshal each layer's W into its tile-major SHMEM
   form at model load; per inference, only activations move. Projected: 0.01 → ~0.6
   TOPS (then latency-bound).
2. **Fewer dispatches**: the array can stream the whole GEMM in one dispatch (large
   NB) instead of one M-block/K-chunk per call — removes the ×32 latency and the
   per-dispatch re-marshal. The 20.7-TOPS bench used one huge-NB dispatch; realistic
   shapes must stream similarly.
3. **Fast A marshal / C un-marshal** (SIMD/memcpy-shaped), and keep A resident.

Until (1)+(2) land, the offload is a net loss vs the GPU (which needs no marshaling).
The 20.7 TOPS is a real *compute* rate; the deliverable end-to-end rate depends
entirely on beating this overhead — that, not the kernel, is now the open question.

### Measured: pre-packing W helps 2×, but CPU marshaling is still a dead end

`prepack_weights` (once, 23 ms) + `run_packed` (per inference, weight cost = memcpy):
**351 → 177 ms/run (0.01 → 0.02 TOPS)** on the same shape, still correct. So the W
re-pack was ~half — but the residual 177 ms is the **C tile→row-major reshuffle** (M·N
= 3.1 M scalar index ops per inference) plus per-dispatch A-pack. Even a perfect CPU
version floors around ~20 ms (0.16 TOPS) — still below the GPU.

**The real fix is DMA-side reshuffle, not CPU.** The shim DMA supports strided access
(`dims_to_stream` on the objectfifo `dma_bd`), so A/W/C can be fed **row-major** and
the DMA does the tile-major reshuffle *for free* during transfer — which is exactly
how IRON's whole_array gemm avoids CPU marshaling. My hand-written R6 MLIR uses plain
linear `dma_bd` (hence the CPU marshaling). Porting the tile-major stride pattern into
the `dma_bd` eliminates CPU marshaling entirely; that is the substantive next step and
the true gate on offload viability (alongside the ~78 µs/dispatch latency, which still
argues for streaming the whole GEMM in one dispatch). `prepack_weights`/`run_packed`
stay useful (weights still pre-arranged once), but the reshuffle must move to the DMA.

## The one architectural decision (needs a call)

**Concurrency model.** The win is *concurrent* NPU-prefill ‖ GPU-work. Options:
- **Sync offload** (simplest): block on the NPU GEMM. Only wins if the NPU GEMM is
  faster than the GPU for that shape — at 20.7 vs ~50 TOPS it usually is *not* alone,
  so sync offload mostly doesn't help. Good for a first correctness wiring only.
- **Concurrent split** (the real win): split each prefill GEMM (or alternate
  layers/experts) between NPU and GPU so both run in parallel, ~+40% aggregate.
  Needs a work-splitting policy + a join, and careful interaction with the
  HIP-direct scheduler.
- **Async pipeline**: NPU prefills layer L+1's projections while the GPU finishes
  layer L. Most throughput, most complexity.

Recommend: land steps 1–3 + a **sync**, flag-gated step 4 first (proves the whole
path end-to-end on a real model), then design the concurrent split as its own effort
— that is where the aggregate win actually lives and where the HIP/NPU scheduler
interaction needs deliberate design.
