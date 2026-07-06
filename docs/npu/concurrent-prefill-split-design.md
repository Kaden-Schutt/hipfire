# Concurrent NPU ‖ GPU prefill split — design & honest ROI

The R6 investigation delivered a validated, runtime-callable NPU W4A8 prefill GEMM
(`NpuGemmMp`, ~1.9 TOPS e2e) and **proved the concurrency premise** (`npu_concurrency_demo`:
an async `submit` hides the whole NPU GEMM behind concurrent host work). This doc is the
go/no-go for actually building the split — grounded in the measured numbers, not the peak
spec — so we don't build a complex hot-path feature for a marginal win.

## The aggregate-win arithmetic — MEASURED

Running the NPU and GPU concurrently adds the NPU's throughput on top of the GPU's:
`speedup = 1 + NPU / GPU`. The **real** NPU rate is ~1.9 TOPS (feed-bound, flat across
batch — r6/README). The GPU rate is now measured too (`gpu_w4a4_lowbatch_bench`,
`gemm_iu4_i32_wmma_lds`, the tuned LDS kernel; K=512, N=4096; W4A4 ≈ W4A8 within ~1.05× on
gfx1151):

| tokens (B) | GPU W4A4 TOPS | NPU share = concurrent-split win |
|---|---|---|
| 64   | 6.76  | **+21.9%** |
| 128  | 18.35 | +9.4% |
| 256  | 24.81 | +7.1% |
| 512  | 28.09 | +6.3% |
| 768  | 32.65 | +5.5% |
| 2048 | 38.15 | +4.7% |
| 4096 | 32.95 | +5.5% |

The original "+40%" hope assumed the NPU could hit ~its ~56-TOPS *peak* ≈ the GPU. It
can't (objectfifo per-slab ceiling → 1.9). And the GPU ramps to 24–38 TOPS by B≥256 — so
across the realistic interactive range (256–768 tokens) **the split adds only ~5–7%**, and
it only reaches double digits (+22%) at trivially short prompts (≤64 tokens, where the GPU
sits at 6.8 TOPS, badly under-utilized).

## Go/no-go — RESOLVED: don't build (except a ≤128-token niche)

The measurement settles it. The concurrent split's win is **marginal (~5–7%) across the
range that matters** and only meaningful (+9–22%) below ~128 tokens. That does **not**
justify a hot-path feature (an N-split at the dispatch seam + async join + a GPU that must
never regress). The NPU offload *works* and is proven net-positive — but the ceiling on the
NPU GEMM (1.9 TOPS) makes it too small a slice of a 25–38-TOPS GPU to matter for real
prefill.

**Recommendation: do not build the split now.** The R6 work stands as a validated,
documented capability (`NpuGemmMp` + benches + this analysis). Revisit only if a concrete
product targets sub-128-token interactive prefill latency specifically, or if a future NPU
kernel breaks the per-slab ceiling (bigger effective tile / different feed) and lifts the
1.9-TOPS floor materially.

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

## Honest recommendation — measured, resolved

The concurrency mechanism is proven and the primitive is production-ready — the NPU offload
*works* and is net positive. But the go/no-go measurement (`gpu_w4a4_lowbatch_bench`) settled
the ROI: **the split adds only ~5–7% across the realistic interactive range (256–768
tokens), reaching double digits only below ~128 tokens.** That does not justify the hot-path
complexity. **Recommendation: do not build the *throughput* split now.** The R6 work stands as a
validated, documented capability (`NpuGemmMp` + benches + this analysis) — revisit only for
a specific sub-128-token latency product, or if a future NPU kernel lifts the 1.9-TOPS
per-slab ceiling materially. Nothing is lost by waiting.

## Still worth investigating (this verdict is throughput-framed only)

The "don't build" above rejects one specific thing — splitting a single prefill GEMM for
*throughput*. It does **not** close the NPU. Three angles the throughput lens misses:

1. **Power / battery — the NPU's real edge on a laptop.** The NPU is far more perf/**watt**
   than the iGPU. On battery, running inference (or part of it) on the low-power NPU — even
   at 1.9 TOPS — instead of the power-hungry GPU could materially extend battery life. This
   whole analysis measured *throughput*, never *power*. Next step: measure NPU vs GPU power
   at equal work (`amd_pmf` `power_mw` sensor / `xrt-smi` for the NPU vs `rocm-smi` for the
   GPU) and reframe the value as Wh/token, not TOPS. A modest-throughput NPU that sips power
   is a genuine win for a plugged-out laptop.
2. **Draft model on the NPU (speculative decode).** Instead of splitting one GEMM, run the
   *whole* small draft-model forward on the NPU while the GPU does the target-model verify —
   a clean concurrent split at the **model** level. It frees the GPU, and the draft's
   small-batch decode is exactly the NPU's regime. A much better NPU fit than the prefill
   GEMM split, and the concurrency mechanism (`submit`/`wait`) is already proven.
3. **NPU prefill more broadly** — offloading whole layers/experts, or prefill for a
   secondary/background request, rather than co-splitting the foreground GEMM.

These reframe NPU value away from raw prefill throughput and are **not** ruled out by the
split ROI above.
