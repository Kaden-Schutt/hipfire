# Gemma 4 WMMA Prefill — Phase 6 Milestone 1

**Date:** 2026-06-09 (revised after profiling)
**Branch:** `feat/dispatch-unification-gemma4` (tip `d1b1a488`)
**Goal:** Route gemma4 prefill projections through batched WMMA GEMM, closing the prefill performance gap.

## 0 · Profiled baseline

rocprofv3 kernel-trace on 12B-Q8, 20-token prompt, per-token decode:

| Category | Calls | Time (ms) | % |
|---|---|---|---|
| **GEMV/GEMM (projections)** | 9,212 | 1,629.5 | **93.6%** |
| Normalization (rmsnorm) | 9,436 | 52.6 | 3.0% |
| Attention (tile + reduce) | 2,688 | 29.1 | 1.7% |
| Memory (copy/fill) | 6,214 | 9.2 | 0.5% |
| Elementwise (scale/add/mul/gelu) | 8,092 | 9.0 | 0.5% |
| RoPE | 1,344 | 5.9 | 0.3% |
| KV cache write | 2,688 | 4.7 | 0.3% |
| Embedding | 28 | 0.3 | 0.0% |
| Logits (softcap) | 28 | 0.2 | 0.0% |

**Projections are 93.6% of GPU time.** Per-token GEMV is the bottleneck. Attention at 1.7% is negligible for short prefill. Batching projections through WMMA is the correct target.

Each `gemv_q8_0` call takes 177µs average (including launch overhead). 9.6 GEMVs per token-layer × 48 layers ≈ 460 GEMV launches per token. With B=20 tokens that's 9,212 total. WMMA batched GEMM reduces this to ~7 per layer × 48 layers = 336 total launches.

Full profile data: `findings/gemma4_prefill_profile_12b_q8.md`

## 1 · Bug fixes already landed

Three critical bugs were fixed before this plan's implementation phase:

### Bug 1 (CRITICAL): `gemm_hfq4g256_wmma` missing F32→F16 conversion

The GPU method took `x_f16` by name but never verified or performed F32→F16 conversion. Callers passing F32 data (e.g. via `GemmFamily` dispatch) would silently produce garbage — F32 bytes reinterpreted as F16.

**Fix (commit `d1b1a488`):** Added `ensure_fp16_x` conversion matching the `gemm_q8_0_wmma` pattern. If input is already F16, skips conversion. Also added `launch_maybe_blob` + `KernargBlob` for graph-capture compatibility and profiling timer.

### Bug 2 (CRITICAL): `GemmFamily::resolve` had no arm for `DType::MQ4G256`

The dispatch arm returned `UnsupportedVariant` for MQ4G256 weights, crashing on 26B-A4B production model.

**Fix (commit `d1b1a488`):** Added `DType::MQ4G256 → GemmHfq4G256Wmma / GemmHfq4G256` mapping. MQ4G256 shares the same 136-byte/group layout as HFQ4G256 — same kernel binary.

### Bug 3 (CRITICAL): WMMA results not byte-identical to scalar

F16 input quantization loses ~3 mantissa bits vs F32 scalar GEMV. Cannot be default-ON without relaxing coherence criteria.

**Fix (commit `d1b1a488`):** Added `HIPFIRE_WMMA_PREFILL` env var gate, default OFF. Set `HIPFIRE_WMMA_PREFILL=1` to opt in.

## 2 · Adversarial review findings

Three reviews were produced: self-review (`findings/gemma4_prefill_wmma_plan_rev_glm5.md`), Gemini (`findings/gemma4_prefill_wmma_plan_rev_gemini.md`), and Claude (`findings/gemma4_prefill_wmma_plan_rev_claude.md`). Cross-review consolidation in Appendix A of the self-review.

Key findings incorporated:

| # | Finding | Source | Status |
|---|---|---|---|
| 1 | F32→F16 bug in `gemm_hfq4g256_wmma` | Self + Claude | **Fixed** (Bug 1) |
| 2 | MQ4G256 not in `GemmFamily` | Self | **Fixed** (Bug 2) |
| 3 | WMMA not byte-identical — cannot default-ON | Self + Claude | **Fixed** (Bug 3) |
| 4 | Adapt v2, don't write greenfield | Claude C4 | **Accepted** — restructure Step 2 |
| 5 | Stale F16 cache across layers | Self + Gemini G3 | **Open** — use `convert_fp16_x_uncached` or invalidate per-layer |
| 6 | Add lm_head to prefill, eliminate redundant re-run | Gemini G6 | **Accepted** — add to Step 2 |
| 7 | Drop 26B-A4B from Milestone 1 success criteria | Claude C5 | **Accepted** — MoE dominates, small gain expected |
| 8 | Add gfx1100 correctness gate | Claude C8 | **Accepted** — add to validation |

## 3 · Architecture

### 3.1 Current flow (per-token decode reused for prefill)

```
for each prompt token:
  forward_scratch_inner_lowered():
    for each layer:
      Step::Gemv (q_proj)     ← single-token GEMV, 177µs each
      Step::Gemv (k_proj)
      Step::Gemv (v_proj)
      Step::Attend (kv_write + flash_attn)  ← per-token, works with q8 ring-buffer
      Step::Gemv (o_proj)
      Step::Gemv (gate_proj)
      Step::Gemv (up_proj)
      Step::Gemv (down_proj)
    final_norm + lm_head (Step::Gemv) + softcap
```

9.6 GEMVs per token-layer × 48 layers × B tokens = 460B total GEMV launches.
93.6% of GPU time in GEMV, 177µs average each (mostly launch overhead + latency).

### 3.2 Target flow (batched WMMA prefill)

```
forward_prefill_batch_wmma():
  embed all tokens → pb_residual [B, dim]

  for each layer:
    // Batched projections via WMMA (336 total, vs 460B GEMV)
    rmsnorm_batched(pb_residual, ...) → pb_tmp
    GemmFamily::run(q_proj, pb_tmp → pb_q)     ← WMMA [B×q_dim, K] GEMM
    GemmFamily::run(k_proj, pb_tmp → pb_k)     ← (F32→F16 conversion inside)
    GemmFamily::run(v_proj, pb_tmp → pb_v)
    rmsnorm_batched + rope_batched_f32 (batched proportional RoPE already exists)

    // Per-token attention (unchanged — works with q8 ring-buffer)
    for each token:
      Step::Attend(kv_write + flash_attn)

    GemmFamily::run(o_proj, pb_q → pb_attn_out)
    rmsnorm_batched + residual_add

    // Dense FFN (12B) or per-token MoE (26B-A4B)
    rmsnorm_batched
    GemmFamily::run(gate_proj, ...) → pb_gate
    GemmFamily::run(up_proj, ...) → pb_up
    gelu_tanh + mul
    GemmFamily::run(down_proj, ...) → pb_ffn_out
    rmsnorm + residual_add + layer_scalar

  final_norm + lm_head + softcap (on last token only)
```

**Key design choices:**
- **Batched GEMM for projections, per-token attention.** Profile data confirms 93.6% in projections, 1.7% in attention. Optimizing attention (Finding 5/13 in the original review) is premature — it's not the bottleneck.
- **Per-token attention preserved.** The q8 ring-buffer KV write and flash attention work correctly per-token. No need for batched attention (which would require ring-buffer-aware batched kernels). This avoids the v1/v2 q8 KV bug entirely.
- **`GemmFamily::run()` auto-selects WMMA.** On HasWmma archs (gfx1100+), resolves to WMMA variant. On older archs, falls back to scalar.
- **`HIPFIRE_WMMA_PREFILL=1` gate.** WMMA F16 quantization is not byte-identical to scalar F32. Must be explicitly opted into until coherence criteria are relaxed.
- **C4 recommendation: adapt v2, don't rewrite from scratch.** The existing `forward_prefill_batch_v2` (gemma4.rs:2612) already has all the structure. Fix its KvTierInputs bug, wire `run_prefill_gemm` → WMMA path, done.

### 3.3 What changes from the existing code

The existing `run_prefill_gemm` (gemma4.rs:39) already routes through `GemmFamily` dispatch. With `HIPFIRE_WMMA_PREFILL=1`, it calls `GemmFamily::run()` which resolves to WMMA. With the gate OFF, it uses the explicit scalar key mapping.

The existing `forward_prefill_batch_v2` (gemma4.rs:2612) already has batched `rmsnorm_batched`, batched projection routing, per-token attention, per-token expert loop for MoE. It needs:
1. KvTierInputs bug fix (hardcoded `quant_asym3: true` → read from cache)
2. Replace `run_prefill_gemm` calls with the WMMA-enabled version
3. Add final norm + lm_head (eliminate redundant last-token re-run)
4. Fix F32→F16 caching across layers (use `convert_fp16_x_uncached` or invalidate)

## 4 · Implementation plan

### Step 1 — Fix v2 KvTierInputs + wire WMMA (2 hours)

Fix the existing `forward_prefill_batch_v2`:
- Replace hardcoded `quant_asym3: true` / `quant_q8: false` with dynamic cache reads
- Replace `run_prefill_gemm` calls with the WMMA-enabled path (already works via `HIPFIRE_WMMA_PREFILL=1`)
- Validate: short prompt produces coherent output with v2 path

### Step 2 — Fix F16 cache + add lm_head (1 hour)

- Use `convert_fp16_x_uncached` for prefill F32→F16 (the pointer-keyed cache in `ensure_fp16_x` is wrong for reused activation buffers across layers — same pointer, different data)
- Add final norm + lm_head (G6 recommendation: eliminate redundant last-token re-run from daemon)

### Step 3 — Daemon wiring (1 hour)

- Wire `forward_prefill_batch_v2` into the daemon behind `HIPFIRE_WMMA_PREFILL=1`
- Set `PREFILL_BATCH_THRESHOLD = 16` (use per-token decode for prompts ≤15 tokens)
- Last-token logits from v2 itself (no daemon re-run)

### Step 4 — Coherence validation (1 hour)

1. 12B Q8, short prompt ("Capital of France?") — argmax must match scalar path
2. 12B Q8, long prompt (1266 tokens) — summary must be coherent
3. 12B Q8, WMMA vs scalar — first ~26 tokens identical, then small divergence (expected)
4. 26B-A4B MQ4+Q8, short prompt — coherent
5. gfx1100 correctness (C8 recommendation) — if hardware available

### Step 5 — Perf measurement (30 min)

Measure tok/s on gfx1151 for:
- 12B Q8 × 20-token prompt
- 12B Q8 × 1266-token prompt

Expected: 5–10× improvement on projection-dominated path. Actual speedup depends on launch overhead reduction and WMMA compute throughput.

## 5 · Risks

| Risk | Mitigation |
|---|---|
| Stale F16 cache serves wrong data across layers | Use `convert_fp16_x_uncached`; or invalidate `fp16_x_source_ptr` between layers |
| v2 has an unknown bug beyond KvTierInputs (v1 garbage root cause still unknown) | v1 calls `sliding_layer_decode_impl` dynamically; v2 hardcodes. Test thoroughly. If v2 still produces garbage after KvTierInputs fix, root-cause before shipping |
| Per-token attention loop is still slow for long prefill (B>512) | Accept for Milestone 1. B=128 per chunk is the initial target. Batched attention is Milestone 2+ |
| WMMA F16 quantization changes numerical results — small divergence after ~26 tokens | Expected. Documented as opt-in. `HIPFIRE_WMMA_PREFILL=1` required |
| 26B-A4B MoE per-expert loop dominates prefill | Expected (C5). MoE batching is separate work. 26B-A4B gain will be small |
| `GemmFamily::run()` for Q8_0 on gfx1151 resolves to `GemmQ8_0Wmma` — new for this arch | `gemm_q8_0_wmma` has `ensure_fp16_x` and `has_wmma()` assertion. Works on gfx1151 but untested for prefill specifically |

## 6 · Out of scope

- **Batched attention for prefill** — per-token attention is correct with q8 ring-buffer and fast enough (1.7%) for B≤128
- **MoE prefill batching** — per-token expert loop is adequate; indexed kernels handle decode
- **v1/v2 batched prefill debug** — preserved on `feat/gemma4-batched-prefill-jukefr` for reference
- **Default-ON WMMA** — requires relaxed coherence criteria (byte-identical → within epsilon)
- **gfx1100 validation** — if hardware is unavailable, skip for Milestone 1

---

*Revised 2026-06-09 after profiling and bug fixes. Profile data in `findings/gemma4_prefill_profile_12b_q8.md`. Adversarial reviews in `findings/gemma4_prefill_wmma_plan_rev_glm5.md`, `findings/gemma4_prefill_wmma_plan_rev_gemini.md`, `findings/gemma4_prefill_wmma_plan_rev_claude.md`. Bug fixes in commit `d1b1a488`.*
## 8 · Measured perf results (2026-06-09)

### Short prompt (17 tokens, "What is France?")

| Path | Prefill time | Prefill tok/s | Decode tok/s | TTFT |
|---|---|---|---|---|
| Per-token decode | 1041ms | 16.3 | 13.9 | 1041ms |
| Batched scalar | 876ms | 19.4 | 15.7 | 876ms |
| **WMMA batched** | **160ms** | **106.2** | **16.9** | **160ms** |

**6.5× prefill speedup for short prompts.** TTFT drops from 1.04s to 0.16s.

### Long prompt (1279 tokens)

| Path | Prefill time | Prefill tok/s | Decode tok/s | TTFT |
|---|---|---|---|---|
| Per-token decode | 93,610ms | 13.7 | 10.6 | 93.6s |
| Batched scalar | 93,659ms | 13.7 | 10.6 | 93.7s |
| WMMA batched | 93,668ms | 13.7 | 10.5 | 93.7s |

**0× improvement.** Per-token attention dominates wall-clock time at long contexts.

### Root cause: GPU utilization

rocprof on 1279-token prompt:
- GPU compute: 26,836ms (GEMV 23,870 + attn 2,966)
- Wall time: 93,610ms
- **GPU utilization: 28.7%** — the CPU is the bottleneck

The per-token attention loop issues ~700K HIP operations (1279 tokens × 48 layers × ~12 calls each). The GPU is idle 71% of the time waiting for the CPU to stage the next dispatch. Batched GEMM helps projections but doesn't reduce the attention dispatch count.

### Revised milestone plan

**Milestone 1 (SHIPPED):** WMMA batched projections for short/medium prefill. 6.5× for ≤32 tokens, diminishing returns for longer contexts. `HIPFIRE_WMMA_PREFILL=1` and `HIPFIRE_BATCHED_PREFILL=1` gates.

**Milestone 2 (NEXT):** Batched attention for long-context prefill. This is the critical missing piece for 1279+ token contexts. Options:
- **a)** Batched q8 KV write + batched flash attention (new ring-buffer-aware kernels)
- **b)** CPU-side pipelining — overlap attention dispatch with GEMV computation
- **c)** CuDNN-style flash attention with batched inputs (leverage ROCm library)

Each approach needs ring-buffer cache_capacity support for q8 sliding KV.

**Milestone 3:** 26B-A4B MoE batched prefill (currently gated out due to `apply_moe_branch_batched` token attractor).
