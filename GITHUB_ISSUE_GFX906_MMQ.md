# GitHub Issue Template: gfx906 (MI50) prefill performance investigation

**Title:** gfx906 (MI50) prefill performance: 3.29× gap vs llama.cpp - MMQ adaptation plan

**Labels:** performance, gfx906, investigation

---

## Performance Gap

**Hardware:** AMD Instinct MI50 (gfx906, 60 CUs, 1024 GB/s HBM2)
**Model:** Qwen 3.5 9B Q4_K_M (llama.cpp) / HFQ4-G256 (hipfire)
**Workload:** 512-token prefill + 10-token generation

### Measured Performance (verified without profiler overhead)

| System | Prefill (tk/s) | Decode (tk/s) | Wall-clock (512 tok) |
|--------|----------------|---------------|----------------------|
| **llama.cpp-gfx906** | 244.75 ± 0.15 | 60.84 ± 0.54 | 2.09s |
| **hipfire** | 74.3 ± 0.0 | 49.0 ± 0.2 | 6.89s |
| **Gap** | **3.29×** | 1.24× | 3.30× |

The gap is primarily in **prefill**, not decode.

---

## Investigation Summary

### What We Ruled Out

#### ❌ Hypothesis 1: Profiler overhead
- **Claim:** rocprof skews measurements
- **Result:** Verified both systems WITHOUT rocprof - gap is real (3.29×)
- **Evidence:** `plans/llama_cpp_baseline_verification.md`

#### ❌ Hypothesis 2: Kernel granularity (BATCH_TILE)
- **Claim:** Smaller batch tiles would improve latency hiding
- **Test:** BATCH_TILE=8, 4, 2, 1 all achieve **identical 74 tk/s**
- **Conclusion:** NOT memory-latency bound in the expected way
- **Evidence:** `BATCH_TILE_RESULTS.md`

#### ❌ Hypothesis 3: Scalar vs vectorized loads
- **Claim:** Manual vectorization of memory loads would close gap
- **Analysis:** Compiler likely already vectorizes; HFQ4 format (136-byte stride) blocks cross-group vectorization
- **Estimated gain:** 2-7% (not 30-50% as initially hoped)
- **Evidence:** `plans/gfx906_vectorized_loads.md`, `gfx906_vec_plan_rev_claude.md`, `gfx906_vec_plan_rev_glm5.md`, `gfx906_vec_plan_rev_gemini.md`

---

## Root Cause: Architectural Difference

### hipfire's approach (row-parallel)
- **Kernel:** `gemm_gate_up_hfq4g256_fp16_wave64.hip`
- **Strategy:** One wave64 per output row, no LDS, no data reuse
- **Dispatch:** 64 large kernels (113ms avg), sequential execution
- **GPU utilization:** 1.04× GPU/wall-clock ratio (near-sequential)

### llama.cpp's approach (tiled MMQ)
- **Kernel:** MMQ (Multi-Matrix Quantization) with 128×128 tiles
- **Strategy:** LDS staging, pre-quantized activations (Q8_1), int8 dot-products, 8× weight reuse per tile
- **Dispatch:** 1,182 small kernels (1.77ms avg), heavy overlap
- **GPU utilization:** 6.28× GPU/wall-clock ratio (async overlap)

### Key findings from profiling

**hipfire:**
- GPU kernel time: 7.25s (with rocprof)
- Wall-clock time: 6.96s (with rocprof) → **1.04× ratio** (sequential)
- HBM bandwidth: 300 GB/s (29% of peak)
- L2 hit rate: 65%

**llama.cpp:**
- GPU kernel time: 13.69s (with rocprof)
- Wall-clock time: 2.18s (with rocprof) → **6.28× ratio** (heavy overlap)
- Kernel count: 50× more launches than hipfire (1,182 vs 64)

**Conclusion:** The dispatch strategy difference (tiled MMQ with kernel overlap vs row-parallel sequential) IS explained by the MMQ tiled architecture. llama.cpp's many small tiles naturally enable async overlap; hipfire's large monolithic kernels cannot overlap effectively.

**Evidence:** `plans/llama_cpp_baseline_verification.md` Section "Implications for Optimization Plan"

---

## Proposed Solution: MMQ Kernel Adaptation for gfx906

hipfire already has an MMQ kernel (`kernels/src/gemm_hfq4g256_residual_mmq.hip`) that implements llama.cpp's tiled approach, but it's **gated to RDNA3 only** because it uses WMMA instructions unavailable on gfx906.

### Technical approach

**Challenge:** Replace WMMA with gfx906-native instructions
- Current: `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` (2 instructions for 16×16×16 tile)
- gfx906 replacement: `__builtin_amdgcn_sdot4` (dp4a) loops (1024 instructions per tile)

**Supporting evidence from skyne98/wiki-gfx906:**
- gfx906 has `v_dot4_i32_i8` (dp4a): 43-44 TOPS with ILP
- gfx906 has `v_dot8_i32_i4` (dot8): 85-86 TOPS with ILP
- QDQ amortization: 21.7 TOPS (dp4a) when dequant overhead is amortized across tile reuse
- LDS bandwidth: 9.5-11.2 TB/s with 128-bit loads

### Implementation paths

**Path A: dp4a MMQ (pragmatic first step)**
- Keep Q8_1 quantization, replace WMMA with dp4a loops
- Expected: 1.5-2× improvement on affected kernels
- Effort: 5-7 days (original estimate)
- **Phase 1 target: `gemm_gate_up` (44.76% of prefill time), NOT residual (27%)**

**Path B: dot8 MMQ (llama.cpp parity)**
- Use `v_dot8_u32_u4` with raw HFQ4 nibbles, new Q4_1 quantization
- Expected: 2-3× improvement, approaching llama.cpp performance
- Effort: 8-12 days (after Path A validation)

### Detailed plan

Full implementation plan: `plans/gfx906_mmq_plan.md`

Adversarial reviews identifying risks:
- `gfx906_mmq_plan_claude.md` (critical issues: dp4a loop complexity, wave64 indexing, phase ordering)

---

## Critical Issues from Adversarial Review

1. **dp4a loop complexity underestimated** - Replacing 2 WMMA instructions with 1024 dp4a requires complex cross-thread shuffling, risks register spill → occupancy loss
2. **Wave64 indexing not verified** - MMQ kernel assumes warp32 semantics; gfx906 is wave64 native → requires verification (80% bug probability without testing)
3. **Wrong phase ordering** - Original plan targets residual first (27% of time), but `gemm_gate_up` is 44.76% → **must be Phase 1**
4. **Tile size reduction doubles overhead** - Reducing to 64×64 for 2 workgroups/CU cuts arithmetic intensity in half → may negate dp4a advantage

**Revised timeline:** 13-21 days to dp4a production (vs original 5-8 days estimate)

**Revised success probabilities:**
- Reaching 1.5× overall (111 tk/s): 70% (plan implied 90%)
- Reaching 2× overall (148 tk/s): 40% (plan implied 70%)
- Reaching 3× overall (222 tk/s): 15% (plan implied 50%)

---

## Request for Review

@Kaden-Schutt - Could you take a quick look at `plans/gfx906_mmq_plan.md` and the adversarial review at `gfx906_mmq_plan_claude.md`?

**Specific questions:**
1. Is the MMQ adaptation approach sound for gfx906?
2. Are there any gfx906-specific gotchas we're missing (wave64 indexing, LDS bank conflicts, etc.)?
3. Does the phase ordering (gate_up → residual → qkvza) make sense given profiling data?
4. Should we validate wave64 indexing BEFORE implementing dp4a loops, or are they independent enough to do in parallel?

**Key decision point:** Phase 0 verification (wave64 indexing + HBM bottleneck analysis, 2-3 days) before committing to full dp4a implementation.

---

## References

- Baseline verification: `plans/llama_cpp_baseline_verification.md`
- Vectorized loads investigation (ruled out): `plans/gfx906_vectorized_loads.md`
- MMQ adaptation plan: `plans/gfx906_mmq_plan.md`
- Adversarial reviews: `gfx906_vec_plan_rev_*.md`, `gfx906_mmq_plan_claude.md`
- Architecture research: https://skyne98.github.io/wiki-gfx906/
- llama.cpp-gfx906 fork: https://github.com/skyne98/llama.cpp-gfx906

---

## To Submit This Issue

1. Go to: https://github.com/Kaden-Schutt/hipfire/issues/new
2. Copy the content above (everything after "---" in the first section)
3. Paste into the issue body
4. Set title: "gfx906 (MI50) prefill performance: 3.29× gap vs llama.cpp - MMQ adaptation plan"
5. Add labels: `performance`, `gfx906`, `investigation`
6. Submit
