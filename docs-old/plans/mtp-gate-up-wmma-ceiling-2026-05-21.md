# gate_up_wmma 32% peak BW ceiling — 3 variants falsified

**Date:** 2026-05-21
**Branch:** `mtp-hiptrx-rocprof`
**Target:** Lift MTP K=2 p_min=0.65 throughput above the 60.44 tok/s
Goal A ceiling on k9lin / 7900 XTX (gfx1100) by optimizing the small-M
batched WMMA fused gate+up projection kernel.
**Outcome:** All 3 variants failed to materially shift overall tok/s.
Kept all 3 as opt-in artifacts via `HIPFIRE_GATE_UP_VARIANT`.

## Baseline (default kernel)

`gemm_gate_up_hfq4g256_wmma` at M=27648, N=3, K=5120 on gfx1100:
- Per-call wall: ~239 µs
- Effective BW: 305 GB/s = **32%** of 960 GB/s peak (7900 XTX)
- Register pressure: 73 VGPR, 23 SGPR, 0 LDS, 0 spills
- Theoretical 2× headroom remains, but not capturable via the levers below.

## Variants tested

| Variant   | Per-block tile | Threads | Strategy                              | Per-call wall | Overall tok/s (median, n=5) | Outcome                  |
| --------- | -------------- | ------- | ------------------------------------- | ------------- | --------------------------- | ------------------------ |
| default   | 16 × 16        | 32      | k2 SW pipeline (2 K-tiles in-flight)  | 239 µs        | 58.09 (mean 59.04)          | baseline                 |
| k4        | 16 × 16        | 32      | k4 SW pipeline (4 K-tiles in-flight)  | 232 µs (-3%)  | 59.55 (~0% overall)         | FALSIFIED — load latency not bottleneck |
| ldscoop   | 16 × 16        | 32      | Cooperative LDS weight staging (128B coalesced) | (worse) | 51.0 (-14%) | FALSIFIED — LDS round-trip > coalescing gain |
| 2tile     | 32 × 16        | 64 (2 waves) | Halve grid, share X tile via L0/L1 cache | (unknown) | 60.97 (mean 59.97, +0.76% τ-controlled) | FALSIFIED — kernel opt not rate-limiting |

## Why each variant failed

**k4 (more in-flight B loads):** The 3% per-call improvement is real
but didn't translate to overall tok/s. Memory subsystem already had
enough concurrency; the bottleneck wasn't latency hiding.

**ldscoop (coalesced LDS weight loads):** My theory ("per-thread per-row
weight loads scatter across 16 cache lines, kills coalescing") was wrong.
The base kernel's per-thread sequential access gets good L1 caching, so
the "scatter" framing was wrong. LDS round-trip adds latency exceeding
coalescing savings — net regression.

**2tile (wider M tile, X-tile cache reuse):** Per-block X amortization
is a real lever in theory (halves dispatch count, both waves share the
same FP16 batch tile), but at the Goal A K=2 recipe the gate_up
projection fires too few times per cycle for kernel-level optimization
to move overall tok/s. The +4.96% median delta is sample-selection
noise (3 high-τ runs in 2tile vs 2 in default); τ-controlled delta is
+0.76%, well within the ±1-3% session noise band.

## What the bottleneck IS (hypotheses)

At Goal A K=2 p_min=0.65, the MTP draft phase is heavily early-exited
(60% of cycles via p_min trim, 60% replay-skipped). The remaining
non-kernel time per cycle is:

1. **Sampling on GPU + transfer to CPU for accept/reject decision** —
   per-cycle overhead amortized over only 2-3 committed tokens.
2. **KV cache traffic (q8 mode)** at N=3 — dequant overhead per layer
   per draft step.
3. **Per-cycle dispatch overhead** — `hipLaunchKernel` calls add up
   when cycle = ~20-30 launches and only 2-3 tokens commit.

None of these are addressable by gate_up_wmma optimization. The natural
next levers (if pursued) would be:

- **CUDA Graph / hipGraph capture of the K=2 draft cycle** — replay
  the entire draft+verify chain as one dispatch. This is the prefill
  hipGraph pattern (A3B prefill `feat/a3b-prefill-hipgraph` falsified
  but for different reasons — see `feedback_a3b_prefill_levers_falsified_2026_05_19`).
- **Fused sampling+commit kernel** — skip the round-trip to CPU for
  the τ=2 typical case.
- **MTP head training at higher Q precision** (Q8 vs default Q4) —
  per `project_mtp_unsloth_target_2026_05_15`, stronger head → higher
  τ → fewer cycles for same output → less dispatch amortization burden.

## Methodology notes

- Fresh process per measure (cargo binary invocation).
- 1 warmup + 5 measured per cell.
- gpu-tcas-coordinated (legacy `scripts/gpu-lock.sh`).
- Byte-identical prompt (`benchmarks/prompts/lru_cache_pep8_strict.txt`,
  md5 `1e74f17934fe759468dbe1471b732067`).
- Per-cell σ is wide (~3 tok/s) because τ has 2 modes at this recipe;
  τ-controlled comparison is the discriminating signal, not raw median.

## Artifacts shipped (all opt-in, default off)

- `kernels/src/gemm_gate_up_hfq4g256_wmma_k4.hip` (commit `3c3cac2e`)
- `kernels/src/gemm_gate_up_hfq4g256_wmma_ldscoop.hip` (commit `bc87d791`)
- `kernels/src/gemm_gate_up_hfq4g256_wmma_2tile.hip` (commit `e6a8930a`)

Wire: `HIPFIRE_GATE_UP_VARIANT={k4,ldscoop,2tile}` in
`crates/rdna-compute/src/dispatch.rs:9988`.

## Decision

**Close the gate_up_wmma optimization line for the MTP K=2 recipe.**
Goal A ceiling (60.44 mean) on canonical 27B-3.5 + cvs16384 head is
structural at the per-cycle dispatch/sampling level, not at the
kernel level. Future MTP perf work should target dispatch fusion
(hipGraph K=2 capture, fused sampling) or head quality (Q8 head
retraining for τ uplift), not gate_up_wmma kernel tuning.
