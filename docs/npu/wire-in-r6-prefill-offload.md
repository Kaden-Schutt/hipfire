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

3. **Weight marshaling.** oq4 weights are already 4-bit packed; confirm the packing
   matches R6's `int4` tile layout (the R3a byte-stride lesson) or add a one-time
   repack at load into the NPU-resident tile-major form. Belongs with the loader, not
   per-dispatch.

4. **Runtime offload hook (the hot-path change — smallest possible).** In
   `dispatch/quant.rs`, add an opt-in path: if a prefill W4A8 GEMM is large enough to
   amortize the ~78 µs dispatch latency AND an R6 xclbin is cached for its shape,
   dispatch it on the NPU. Gate behind a flag first (`HIPFIRE_NPU_PREFILL`), measure
   end-to-end, then consider default-on.

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
