# Adversarial Review: PR #115 Lloyd-Max MQ3 Perf Optimization

**Reviewer:** Gemini CLI (Adversarial Mode)
**Date:** 2026-05-06
**Subject:** `docs/plans/PR-115-lload-max-codebooks-mq3.md`

## 1. Executive Summary

The plan correctly identifies the core bottleneck (instruction overhead from ternary-based dispatch) and proposes a sound architectural fix (K4 unrolling + LDS staging). However, the plan has critical gaps in **tail-case safety**, **architectural completeness (residual variants)**, and **LDS synchronization overhead**. While it likely clears the 120 tok/s gate, it risks shipping a kernel with OOB memory access in the tail and leaves the codebase inconsistent by omitting residual support.

## 2. Critical Risks & Potential Bugs

### 2.1. Tail-Case Out-of-Bounds (OOB) Load
The plan states: *"Tail iterations: same LDS-staged path, one group at a time"*.
*   **The Flaw:** The "LDS-staged path" described for the main loop uses a 32-thread cooperative load to fetch 32 codebook entries (4 groups). If the tail case (1, 2, or 3 groups) blindly reuses this logic, it will attempt to load codebooks for non-existent groups `quads*4 + 1..3`, potentially reading past the end of the `A` tensor or into the next row.
*   **Correction:** The tail load must be guarded with `if (quads*4 + i < groups_per_row)` or use a separate scalar-fallback load for codebook entries.

### 2.2. FP16 to FP32 Conversion Latency
The plan proposes converting fp16 to fp32 *at load time* into LDS.
*   **The Flaw:** While this avoids bank conflicts, performing 32 `__half2float` conversions in the load phase of every quad introduces a small stall.
*   **Adversarial Perspective:** On RDNA3, `v_cvt_f32_f16` is fast, but if the load is not well-pipelined with the `ds_write`, the `s_waitcnt` will be longer.
*   **Mitigation:** Ensure the conversion happens in the same instruction slot as the load where possible (e.g., using `tbuffer_load_f16` or similar if applicable, though standard `global_load` + `v_cvt` is more likely).

### 2.3. LDS Bank Conflict vs. Broadcast
The plan assumes fp32 eliminates conflicts.
*   **The Reality:** If all 32 threads in a wave read the *same* `q` (e.g., a constant padding value), the LDS hardware performs a broadcast (1 cycle). If they read 32 different values, it's a conflict unless they hit different banks. Since there are only 8 unique entries per group, and they are stored contiguously as fp32 (4 bytes), they occupy 8 banks. Even with 32 threads, they only hit 8 banks. This is a "limited conflict" and will likely be 1-2 cycles, far better than the 112 cycles of ternaries.

## 3. Architectural Omissions

### 3.1. Residual Variant Gap
The plan explicitly skips the residual variant (`gemv_mq3g256_lloyd_residual`).
*   **The Flaw:** `hipfire` provides residual variants for all other primary kernels (HFQ3, MQ4). By omitting this, any model utilizing residual connections with Lloyd-MQ3 will fall back to the slow runtime path (alloc + gemv + add + free). This makes Lloyd-MQ3 a "second-class citizen" in the runtime and creates a performance trap for users.
*   **Recommendation:** Implement the residual variant simultaneously; it is a trivial addition to the existing plan (mirroring `gemv_hfq3g256_residual_for_arch`).

### 3.2. Architecture Coverage (gfx1101/1102)
The plan targets `gfx1100` but doesn't explicitly mention verification on `gfx1101` (7900 GRE) or `gfx1102` (7600).
*   **The Flaw:** While they share the RDNA3 ISA, the memory bandwidth and CU counts differ. The "120 tok/s" target might be cleared on 7900 XTX but failed on 7600.
*   **Recommendation:** Clarify if the ship gate is "gfx1100 only" or "RDNA3 family".

## 4. Optimization Opportunities

### 4.1. `__syncthreads()` Overkill
*   In a single-wave kernel (`launch_bounds(32, 1)`), `__syncthreads()` is technically unnecessary for execution safety if `s_waitcnt lgkmcnt(0)` is used. While the plan keeps it for "readability", it can introduce a barrier that inhibits some compiler optimizations.
*   **Recommendation:** Use a scoped comment and `__builtin_amdgcn_s_waitcnt(0)` for the tightest possible quad loop.

### 4.2. Codebook Prefetching
*   The current plan loads the codebook at the top of the quad loop. For maximal performance, the codebook for the *next* quad could be prefetched into registers during the current quad's FMAs, then written to LDS.

## 5. Validation Gaps

### 5.1. Divergence Testing
The plan focuses on perplexity (ppl).
*   **The Gap:** It does not specify checking for **logit divergence** against the original switch-based kernel.
*   **Requirement:** Run `scripts/gfx906_logit_divergence.sh` (or equivalent for gfx1100) to ensure the LDS-based results are bit-identical to the switch-based results. Even a small drift (±1e-5) indicates an indexing error or precision loss in the fp16 conversion.

### 5.2. Tail Case Exhaustion
*   **The Gap:** Perplexity tests on 9B/4B models might not hit all tail cases (groups % 4 == 1, 2, 3).
*   **Requirement:** Add a unit test in `cli/chat_pure.test.ts` or a standalone script that sweeps `K` from 256 to 2048 in increments of 256 to verify every possible tail configuration.
