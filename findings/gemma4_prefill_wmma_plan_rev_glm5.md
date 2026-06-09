# Adversarial Review: Gemma 4 WMMA Prefill Plan (Phase 6 Milestone 1)

**Reviewer:** Self-review (adversarial mode)
**Date:** 2026-06-09
**Plan under review:** `docs/plans/gemma4_prefill_wmma.md`
**Verdict:** 🔴 **NOT READY** — 3 critical findings block implementation as written. Plan needs structural revision.

---

## Finding 1 — CRITICAL: `gemm_hfq4g256_wmma` does NOT handle F32→F16 conversion

**Plan claims (§3.3):** "The GPU method itself calls `ensure_fp16_x` internally. So the F32→F16 staging is already handled."

**Reality:** The GPU method `gemm_hfq4g256_wmma` (gemm.rs:19038) takes `x_f16: &GpuTensor` and passes `x_f16.buf.as_ptr()` directly to the HIP kernel. **No conversion happens.** The kernel interprets the raw bytes as `_Float16*`. If the input is actually F32 data, the kernel reads garbage (every pair of F32 values interpreted as one F16).

**Evidence:** The dispatch arm at gemm.rs:150 passes `GemmParams.x` (F32) directly:
```rust
K::GemmHfq4G256Wmma => hip!(gpu.gemm_hfq4g256_wmma(w.buf, x, y, m, k, batch_size)),
```

**Contrast with Q8_0 WMMA:** `gemm_q8_0_wmma` (gemm.rs:16948) correctly checks `x.dtype` and calls `self.ensure_fp16_x(x, batch_size * k)?` for F32 inputs. The HFQ4G256 WMMA method has no such guard.

**Contrast with DeepSeek V4:** The DeepSeek V4 caller (forward.rs:303-312) does explicit `convert_f32_to_f16` before calling `gemm_hfq4g256_wmma`. The staging is the *caller's* responsibility, not built into the GPU method.

**Impact:** `GemmFamily::run()` → `run_key(GemmHfq4G256Wmma)` → `gpu.gemm_hfq4g256_wmma()` with F32 input → **silent garbage output**. The plan's Step 1 (`run_prefill_gemm_wmma` via `GemmFamily::run()`) would produce wrong results for HFQ4G256 weights.

**Fix required before Step 1:** Either:
- (a) Fix `gemm_hfq4g256_wmma` to call `ensure_fp16_x` like `gemm_q8_0_wmma` does (preferred — matches the Q8 pattern), or
- (b) Add explicit F32→F16 staging in `run_prefill_gemm_wmma` before calling `GemmFamily::run()`.

Option (a) is a one-line fix to the GPU method and benefits all callers.

---

## Finding 2 — CRITICAL: MQ4G256 weights have no GEMM family entry

**Plan claims (§3.2, Step 1):** "`GemmFamily::run()` auto-selects WMMA when available."

**Reality:** `GemmFamily::resolve()` handles `F32`, `F16`, `Q8_0`, `HFQ4G256`, `HFQ4G128`. It returns `DispatchError::UnsupportedVariant` for all other dtypes. The 26B-A4B production model (`-mq4-q8down.hfq`) uses **MQ4G256** for gate/up projections. Calling `GemmFamily::run()` on those weights will error.

**Evidence:** `grep MQ4G256 crates/hipfire-dispatch/src/families/gemm.rs` returns nothing.

**Impact:** Step 1 (`run_prefill_gemm_wmma`) would crash on the 26B-A4B production model. The plan's Step 4 coherence test ("26B-A4B model — coherent output") cannot pass.

**Fix required:** The `run_prefill_gemm_wmma` helper must fall back to the existing per-token GEMV loop for dtypes not supported by `GemmFamily`. This is exactly what the current `run_prefill_gemm` does for its `_` catch-all arm. The new helper should try `GemmFamily::run()` first, and on `UnsupportedVariant` error, fall back to repeated GEMV.

---

## Finding 3 — CRITICAL: WMMA path is NOT byte-identical to scalar

**Plan claims (§3.2):** "GemmFamily::run() auto-selects WMMA... On older archs, it falls back to scalar."

**Plan claims (§3.3):** "WMMA GEMM is byte-identical to scalar"

**Plan claims (§Step 3):** "Default ON (no env opt-in needed — WMMA GEMM is byte-identical to scalar)."

**Reality:** The WMMA kernel (`gemm_hfq4g256_wmma`) operates on **F16 inputs** while the scalar kernel (`gemm_hfq4g256`) operates on **F32 inputs**. The F32→F16 conversion loses ~3 decimal digits per element (F16 has 10 mantissa bits vs F32's 23). The WMMA accumulator is F32, but the input quantization is lossy.

**Impact:** The WMMA and scalar paths produce **different numerical results**. This means:
1. Cannot be default-ON without validating that the F16 precision loss doesn't cause coherence regressions
2. The plan's Step 4 ("argmax must match per-token path") will fail — argmax may differ due to F16 rounding
3. The oracle comparison (argmax at position 1024+1200) may fail

**Fix required:** WMMA prefill should be opt-in (`HIPFIRE_PREFILL_WMMA=1`) until coherence is validated. If the argmax differs, the plan needs a relaxed correctness criterion (e.g., top-5 overlap instead of exact argmax match).

---

## Finding 4 — MAJOR: Per-token attention loop replicates v1's structure (and possibly its bug)

**Plan claims (§3.2):** "Per-token attention preserved... This avoids the v1/v2 q8 KV bug entirely."

**Reality:** The plan's Step 2 describes: "Per-token loop: for each token, copy q/k/v to per-token scratch, run Step::Attend, copy output back." This is structurally identical to v1's per-token attention loop in `forward_prefill_batch_v1` (gemma4.rs:2523-2544). The v1 path was verified to produce garbage with q8 ring-buffer KV.

**Analysis:** The v1 bug was NOT in batched attention (v1 uses per-token attention). The bug is in the v1 prefill orchestration — the interaction between batch scratch buffers (`pb_residual`, `pb_attn_out`, `pb_ffn_out`) and per-token decode scratch (`scratch.x`, `scratch.residual`, `scratch.ffn_out`). The plan reproduces this exact same structure.

**Evidence:** Verified that `forward_prefill_batch_v1` with `HIPFIRE_FORWARD_LOWERED=0` (old decode path, not the lowered pipeline) still produces garbage. The bug is in the v1 batch orchestration, not in the decode path.

**Impact:** The plan's Step 2 will likely reproduce the v1 garbage-output bug. The plan's claim that it "avoids the v1/v2 q8 KV bug entirely" is incorrect.

**Fix required:** Before implementing Step 2, root-cause the v1 prefill bug. Two possible approaches:
- (a) Debug and fix v1's batch orchestration, then build WMMA on top
- (b) Skip per-token attention entirely — use batched attention with a ring-buffer-aware kernel

---

## Finding 5 — MAJOR: Per-token attention for B=128 produces 128× more kernel launches than projections

**Plan claims (§1):** "The projections dominate: 6 GEMMs per layer × 48 layers = 288 weight matrix traversals per token."

**Reality for B=128 tokens:**

| Operation | Launches per layer | Launches total (48 layers) |
|---|---|---|
| WMMA GEMM projections | 6 | 288 |
| Per-token attention (B=128) | 128 × ~4 (kv_write K + V + flash_tile + reduce) | ~24,576 |
| Per-token Q/K/V copy in/out | 128 × 6 memcpy | N/A (not launches, but bandwidth) |
| rmsnorm_batched (Q/K/V norms) | 3 | 144 |
| rope_batched | 1 | 48 |

**Attention launches outnumber projections ~85:1.** The per-token attention loop dominates kernel launch overhead. Each launch has ~5-20µs overhead on RDNA. At 24,576 attention launches × 10µs = 246ms of pure launch overhead, vs 288 projection launches × 10µs = 3ms.

**Impact:** The WMMA speedup on projections (even if 30×) saves maybe 50ms per prefill. The per-token attention adds 246ms of launch overhead. The net effect could be **slower** than the current per-token decode path, which amortizes attention across 48 layers without the batch memcpy overhead.

**Fix required:** The plan needs a quantitative model of where time is actually spent. Before implementing WMMA prefill, profile the current per-token prefill to determine:
1. What fraction of time is in GEMV projections vs attention vs other?
2. What is the kernel launch overhead?
3. Is the bottleneck compute, bandwidth, or launch overhead?

If launch overhead dominates, the fix is batched attention (not WMMA projections). If bandwidth dominates, WMMA won't help (bandwidth is the same — weight matrices are loaded once either way). WMMA only helps if the bottleneck is **compute throughput**.

---

## Finding 6 — MAJOR: F32→F16 staging cost is understated

**Plan claims (§5 Risks):** "`ensure_fp16_x` has pointer-keyed caching; same x is reused for q/k/v projections"

**Reality:** The pointer-keyed cache in `ensure_fp16_x` is keyed on the **GPU buffer pointer**. In the prefill path, `x` is the batch activation tensor (`pb_tmp`). This tensor is **overwritten** each layer by `rmsnorm_batched` — same pointer, different data. The cache will serve the **stale** F16 conversion from the previous layer (or the first use of the same buffer).

The `convert_fp16_x_uncached` method exists for exactly this scenario (dispatch.rs:855). But `gemm_hfq4g256_wmma` doesn't use it — and neither does `GemmFamily::run()`.

Even with `convert_fp16_x_uncached`, the F32→F16 staging is an extra pass over the input data: B × K × 4 bytes read + B × K × 2 bytes written. For q_proj (B=128, K=3840): 128 × 3840 × 4 = 1.97 MB read + 0.98 MB written. For 3 projections (q/k/v) per layer: ~9 MB per layer × 48 layers = ~432 MB extra traffic. At 115 GB/s (gfx1151 bandwidth), that's ~3.8ms — not free.

**Impact:** The F32→F16 staging adds measurable bandwidth cost. The cached variant is incorrect for this use case (same pointer, different data each layer). Must use the uncached variant.

---

## Finding 7 — MINOR: The scalar `gemm_hfq4g256` is already batched (BATCH_TILE=8)

**Plan implies (§2):** The scalar path is "per-token GEMV" — suggesting it processes one token at a time.

**Reality:** `gemm_hfq4g256` has `BATCH_TILE=8` — it processes 8 batch elements per workgroup. For B=128, it's not 128 single-token GEMVs; it's 16 batched-tile launches per row. The scalar path IS already a batched GEMM, just without WMMA acceleration.

The actual per-token GEMV happens when `run_prefill_gemm` falls through to the `_` catch-all (for Q8_0 and other dtypes). The HFQ4G256 scalar path benefits from batching; Q8_0 does not.

---

## Finding 8 — MINOR: Q8_0 weights in -q8 model get no WMMA benefit from `run_prefill_gemm`

**Current `run_prefill_gemm`:** Only maps `HFQ4G256` and `HFQ4G128` to batched GEMM keys. For `Q8_0`, falls through to per-token GEMV loop.

**Impact on 12B -q8 model:** The 12B -q8 model uses Q8_0 weights for projections. Even with the plan's `run_prefill_gemm_wmma` routing through `GemmFamily::run()`, the Q8_0 path would go to `GemmQ8_0Wmma` which correctly handles F32→F16. This IS an improvement. But the plan doesn't call out that the primary beneficiary on the 12B model is the Q8_0 WMMA path, not HFQ4G256 WMMA.

---

## Finding 9 — MINOR: "11–30× over scalar" claim needs qualification

**Plan claims (§1):** "achieve 11–30× over the scalar batched path per microbenchmarks"

**Source:** The claim comes from the kernel comment in `gemm_q8_0_wmma.hip`:
> Microbench (per bench_q8_wmma_variants results): 11–30× over the scalar gemm_q8_0_batched at typical DeepSeek V4 shapes

**Issues:**
1. "DeepSeek V4 shapes" — not gemma4 shapes. DeepSeek V4 has hidden_dim=7168 vs gemma4 12B's 3840.
2. "11–30×" range is enormous — the low end (11×) may apply to small M/K where launch overhead dominates.
3. The comparison baseline is `gemm_q8_0_batched` (the chunked scalar path), not `gemv_q8_0` (single-token). For single-token decode, the WMMA GEMM has high overhead (16×16 tiles for M=single-digit is wasteful).

**Impact:** The expected speedup for gemma4 shapes may be at the low end (closer to 5× than 30×). The plan should measure, not assume.

---

## Finding 10 — MINOR: Plan doesn't address embed_tokens batched lookup

**Plan's Step 2:** "Embeds all tokens into pb_residual [B, dim]"

**Current code:** Embedding lookup is per-token (`for (i, &tok) in tokens.iter().enumerate() { gpu.embedding_lookup_*(..., tok, ...) }`). No batched embedding kernel exists.

**Impact:** For B=128, that's 128 individual kernel launches just for embedding. This is 128 out of ~25,000 total launches — not a bottleneck, but worth noting for completeness.

---

## Finding 11 — MINOR: Plan omits the full-layer (hd=512) attention path

**Plan's Step 2:** Describes sliding-layer attention flow but doesn't mention the 5 full-attention layers (L5, L11, L17, L23, L29 on 26B-A4B; L5, L11, L17, L23, L29, L35, L41, L47 on 12B with 48 layers).

Full layers have hd=512, no v_proj (V←K copy), partial RoPE, and use asym3 KV (not q8 ring-buffer). The per-token attention loop needs to branch on layer type, calling the full-layer attention path for these layers.

The existing v2 code handles this (gemma4.rs:2820-2900). The plan should acknowledge this branching.

---

## Summary table

| # | Severity | Finding | Blocks plan? |
|---|---|---|---|
| 1 | 🔴 CRITICAL | `gemm_hfq4g256_wmma` missing F32→F16 conversion — will produce garbage | YES |
| 2 | 🔴 CRITICAL | MQ4G256 not in GemmFamily — crashes on 26B-A4B production model | YES |
| 3 | 🔴 CRITICAL | WMMA not byte-identical to scalar — cannot be default-ON | YES |
| 4 | 🔴 MAJOR | Per-token attention loop replicates v1 structure (same bug risk) | YES |
| 5 | 🔴 MAJOR | Per-token attention dominates launch count (85:1 over projections) | Likely |
| 6 | 🟡 MAJOR | F32→F16 caching is stale for reused activation pointers | Needs fix |
| 7 | 🟢 MINOR | Scalar gemm_hfq4g256 is already batched (BATCH_TILE=8) | Clarification |
| 8 | 🟢 MINOR | 12B -q8 model benefits from Q8_0 WMMA, not HFQ4G256 WMMA | Clarification |
| 9 | 🟢 MINOR | 11–30× claim is for DeepSeek V4 shapes, not gemma4 | Needs measurement |
| 10 | 🟢 MINOR | No batched embedding lookup | Not blocking |
| 11 | 🟢 MINOR | Full-layer (hd=512) attention path not mentioned | Needs acknowledgment |

## Recommended revision path

1. **Fix Finding 1** first: add `ensure_fp16_x` to `gemm_hfq4g256_wmma` (mirroring the Q8_0 pattern). This is a one-line fix that benefits all callers, not just gemma4.

2. **Fix Finding 2**: the `run_prefill_gemm_wmma` helper must fall back to repeated GEMV for unsupported dtypes (MQ4G256, etc.).

3. **Address Finding 5 before implementing**: profile the current per-token prefill to determine where time is actually spent. If launch overhead dominates, the right fix is batched attention (not WMMA projections). If compute throughput dominates, WMMA projections are the right call. This changes the entire plan direction.

4. **Address Finding 4**: root-cause the v1 prefill bug before replicating its structure. Alternatively, implement batched q8 attention with ring-buffer support instead of per-token attention.

5. **Address Finding 3**: make WMMA opt-in until coherence is validated with relaxed criteria (top-5 overlap, not exact argmax).

---

*Review completed 2026-06-09. Plan at `docs/plans/gemma4_prefill_wmma.md`.*
