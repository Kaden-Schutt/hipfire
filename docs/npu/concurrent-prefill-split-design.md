# Concurrent NPU ‖ GPU prefill split — design & honest ROI

The R6 investigation delivered a validated, runtime-callable NPU W4A8 prefill GEMM
(`NpuGemmMp`, ~1.9 TOPS e2e) and **proved the concurrency premise** (`npu_concurrency_demo`:
an async `submit` hides the whole NPU GEMM behind concurrent host work). This doc is the
go/no-go for actually building the split — grounded in the measured numbers, not the peak
spec — so we don't build a complex hot-path feature for a marginal win.

## The aggregate-win arithmetic

Running the NPU and GPU concurrently on the same prefill adds the NPU's throughput on top
of the GPU's: `speedup = 1 + NPU / GPU`, i.e. the NPU contributes a `NPU / (GPU + NPU)`
share of the combined work.

The **real, measured** NPU rate is ~1.9 TOPS (feed-bound, flat across batch — see
r6/README). The GPU rate is the lever, and it depends heavily on batch:

| GPU W4A8 rate (regime) | NPU share = win |
|---|---|
| ~50 TOPS (large-batch, tuned — `reference_gfx1151_iu4_gemm_tuning`) | **+3.7%** |
| ~25 TOPS | +7.1% |
| ~10 TOPS | **+16%** |
| ~5 TOPS | +28% |

The original "+40%" hope assumed the NPU could hit ~its ~56-TOPS *peak* ≈ the GPU. It
can't: the real NPU GEMM is objectfifo-per-slab-overhead-bound at ~1.9. So **at the batch
sizes where prefill throughput actually matters (large batch, GPU ~50), the split is only
~+4%.**

## The one place it could matter: low batch

The NPU is flat ~1.9 TOPS at *every* batch (weight-bandwidth-bound). The GPU, by contrast,
is *underutilized* at low batch — small M can't fill the WMMA pipeline, so its effective
W4A8 rate drops well below 50. That's the one regime where the NPU's fixed 1.9 is a
non-trivial share (rows 3–4 above). And low batch = interactive / single-request prefill,
which is latency-sensitive.

So the split's real value proposition is narrow and specific: **shave interactive
single-request prefill latency by running the NPU alongside an under-utilized GPU.** Not a
bulk-throughput play.

## Go/no-go — the missing measurement

The decision hinges on one number we don't have: **the GPU's W4A8 GEMM rate at low batch
(M≈256–768, K/N of a real model)**. There is no standalone GPU W4A8 microbench today; it
would need a HIP harness on the (locked) GPU.

- If GPU low-batch ≳ 30 TOPS → NPU adds <6% → **not worth the hot-path complexity.**
- If GPU low-batch ≈ 10–15 TOPS → NPU adds ~12–16% of interactive prefill → **worth it**,
  as a latency feature gated to low batch.

Recommended first step before any wiring: add that GPU microbench (or read it off an
existing prefill trace) and put a real number in the table above.

## Implementation sketch (only if green)

Lives at the dispatch seam (`crates/hipfire-rdna/src/dispatch/quant.rs`), where the runtime
issues a quantized prefill linear `C[M,N] = A[M,K]·W[K,N]`:

1. **Split by N-columns.** Give the NPU a slab `N_npu` sized so `N_npu/N ≈ 1.9/(GPU+1.9)`
   — just enough that the NPU finishes about when the GPU finishes its `N-N_npu` columns.
   N-split (not M) keeps each side a contiguous, independent GEMM with no cross-dependency.
2. **Async coordinate.** `NpuKernel::submit` the NPU slab (returns immediately) → issue the
   GPU GEMM on its stream (also async) → GPU stream sync + `NpuKernel::wait` → both C slabs
   are ready. The `submit`/`wait` split (already built + proven) is exactly this.
3. **Join.** The two C column-ranges are disjoint; no reduction, just adjacent writes. The
   NPU weights for its `N_npu` slab are prepacked once at load (`NpuGemmMp::prepack_weights`).
4. **Gate.** Flag + a batch/shape predicate (only fire at low batch where it wins). Default
   off; the NPU path must never regress the GPU-only critical path.

## Honest recommendation

The concurrency mechanism is proven and the primitive is production-ready — the NPU offload
*works* and is a net positive. But the **aggregate win is modest (~4% at throughput batch,
maybe ~12–16% for interactive prefill if the GPU is as underutilized at low batch as
expected).** Build the split only if (a) the low-batch GPU measurement confirms a
double-digit win and (b) interactive prefill latency is a real target. Otherwise the R6
work stands as a validated, documented capability (`NpuGemmMp` + benches) to revisit when a
concrete workload justifies the hot-path integration — nothing is lost by waiting.
