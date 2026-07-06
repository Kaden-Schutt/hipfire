# wmmqa — int8-WMMA HFQ4 dense residual GEMM (gfx1100)

- **Date:** 2026-06-26
- **Status:** design → implementation (ultracode workflow)
- **Arch scope:** gfx1100 (RDNA3 dGPU) only — *prove-first*. gfx1151/gfx12 deferred (antibleed).

## Motivation

The HFQ4 dense residual prefill GEMM dominates prefill on gfx1100 — rocprofv3 (verified,
27B, ~3.2k-tok prefill): `gemm_hfq4g256_residual_mmq_*` = **72% of prefill kernel time**.
It runs the **scalar sdot4 MMQ** path (llama.cpp lineage, `#158`) — light int8 traffic but
**no matrix cores**.

A force-WMMA A/B this session (k9lin, WMMA confirmed via rocprof) showed fp16-WMMA is
**~9–15% slower** than MMQ at every ctx (1k/4k/16k) — but only ~15% *despite moving 2× the
bytes* (fp16 vs int8 activations). If it were purely memory-bound, 2× traffic → ~2× slower.
It's only 15% → the GEMM is **partially compute-bound, and the matrix cores DO help**; fp16's
only liability is its traffic.

So the missing kernel is **wmmqa = int8-WMMA**: MMQ's light int8 traffic **+** WMMA's matrix
cores. The int8-WMMA (`wmma_*_mmqload`) pattern exists only for the MQ2-Lloyd MoE
(`gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload`) — never built for the dense HFQ4 residual,
which is stuck choosing between scalar-MMQ and fp16-WMMA.

**Scope honesty:** this lifts the prefill **floor** (the ~750 t/s short/mid plateau). The
long-ctx 1/x decay is the O(n²) compute of the 16 full-attention layers — structural to
qwen3.6 (full attention, *already* flash-tiled for memory above the 8192 crossover). wmmqa
does **not** touch that; flash-tiling can't either (it's a memory opt, not a FLOP reduction).

## Approach

Base on the existing **fp16-WMMA residual** (its 16×16×16 WMMA tiling carries to int8 unchanged
on RDNA3 — only input dtype half→int8 and accumulator float→int32 change), graft the **Q8_1
int8 load + 4-bit unpack** from the MMQ body. Self-contained `.hip` — **NOT** a `.cuh`
body-include.

## Kernel — `kernels/src/gemm_hfq4g256_residual_wmma_mmqload.hip`

- **Load:** Q8_1 activation-quant + HFQ4 4-bit nibble unpack → int8 LDS tiles laid out for
  `wmma_load` (lift from `gemm_hfq4g256_residual_mmq_body.cuh` lines ~84–128).
- **Matmul:** `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` (RDNA3 wave32, int8 in / **int32**
  accumulate).
- **Epilogue:** dequant (Q8_1 `ds` × HFQ4 `sc`/`zp_eff`, body lines ~164–172) → float, `Y += sum`.
- **Critical correctness seam:** Q8_1 scale is **per-32-element block** → flush+scale the int32
  accumulator every 32 K (= 2 WMMA 16-tiles) to match MMQ granularity exactly. This is the
  bug-prone part; NRMSE catches it.
- **Determinism:** int32 accumulate is order-independent → K-split is deterministic for free
  (no `_ksplit_det` machinery).

## Dispatch — `crates/rdna-compute/src/gemm.rs`, `gemm_hfq4g256_residual` ladder

- New method `gemm_hfq4g256_residual_wmma_mmqload` + kernel-source registration.
- **Prove-first:** env-gated early-return at the top — `HIPFIRE_HFQ4_RES_WMMQA=1`, gfx1100-only
  (`has_wmma_w32() && !has_wmma_w32_gfx12() && !is_wave64_native()`).
- **After it wins:** replace with an `is_gfx1100`-specific predicate — deliberately **NOT**
  `has_wmma_w32()` (catches gfx1151) and **NOT** folded into `has_hfq4_mmq()`'s shared allowlist.
  gfx1151 stays on MMQ until separately validated. The win cannot bleed onto the iGPU.

## Validation

- **Correctness:** coherence (`HIPFIRE_HFQ4_RES_WMMQA=1`, fluent output) + NRMSE vs MMQ on real
  weights (target ≈ MMQ's 0.024% — identical Q8_1 int8 math, only the reduction engine differs).
- **Gate:** `./scripts/coherence-gate.sh` (mandatory — kernel change).
- **Perf:** A/B vs MMQ at 1k/4k/16k prefill on gfx1100. **Win bar: >5% over MMQ** (above the
  ±1–3% noise band).
- **Fleet:** gfx1100 (k9lin) only.

## Off-ramp

If wmmqa ties/loses (NRMSE or perf): the GEMM was more memory-bound than the fp16 A/B implied
(matrix cores can't pay off). There is **no attention fallback** — flash-tiling is done and the
O(n²) is structural. Accept the floor and close.

## References

- int8 load: `kernels/src/gemm_hfq4g256_residual_mmq_body.cuh`
- fp16-WMMA tiling: `kernels/src/gemm_hfq4g256_residual_wmma_k2.hip` (+ `_ksplit`, `_k2x32`)
- int8-WMMA / iu8 intrinsic reference: `kernels/src/gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload.hip`
- dispatch: `crates/rdna-compute/src/gemm.rs` `gemm_hfq4g256_residual` (MMQ gate: `batch_size > 1 && has_hfq4_mmq()`)
