# Gemma 4 WMMA Prefill — Phase 6 Milestone 1

**Date:** 2026-06-09
**Branch:** `feat/dispatch-unification-gemma4` (off `6c7d3128`)
**Goal:** Close the prefill performance gap by routing gemma4 prefill GEMMs through WMMA tensor-core kernels instead of scalar GEMV.

## 1 · Problem statement

The daemon's prefill is **per-token**: each token runs `forward_scratch` → `forward_scratch_inner_lowered` which uses `Step::Gemv` (single-token GEMV) for all projections. For a 20-token prompt on 12B dense, this produces ~4 tok/s on gfx1151 — dominated by kernel launch overhead and scalar FMA throughput.

The existing WMMA GEMM kernels (`gemm_hfq4g256_wmma`, `gemm_q8_0_wmma`) achieve **11–30×** over the scalar batched path per microbenchmarks. They exist and are registered in the dispatch framework with `ArchPredicate::HasWmma` gating. But gemma4's prefill never invokes them — it uses per-token GEMV decode exclusively.

**The fix is not to batch more tokens through the same scalar path** (jukefr's v1/v2 approach — ~2× gain, still 5× from parity). The fix is to route the existing prefill GEMM projections through the WMMA substrate.

## 2 · Why jukefr's approach was insufficient

Jukefr's `forward_prefill_batch_{v1,v2}` batched the **MoE branch and dense projections** across N tokens, then ran per-token (v1) or batched (v2) attention. Measured 130→342 tok/s on 26B-A4B / gfx1201 — a 2.6× gain.

But 342 tok/s vs llama.cpp's 3925 tok/s is still **11.5× behind**. The batched scalar GEMV (`gemm_hfq4g256` with `BATCH_TILE=8`) is bandwidth-bound: each weight row is loaded once but only 8 dot products computed per load. It doesn't use tensor cores. The WMMA kernel does [16×16] tiles with data reuse — fundamentally higher throughput.

Additionally, v1/v2 assumed **asym3** sliding KV. Our branch uses **q8 ring-buffer** KV. The v1 path produces garbage with q8 KV (verified, root cause TBD). V2's batched q8 attention also needs ring-buffer-aware batched kernels that don't exist yet.

**Preserved for reference:** `feat/gemma4-batched-prefill-jukefr` branch (commit `e833f552`).

## 3 · Architecture

### 3.1 Current flow (per-token decode reused for prefill)

```
for each prompt token:
  forward_scratch_inner_lowered():
    for each layer:
      Step::Gemv  (q_proj)     ← single-token GEMV
      Step::Gemv  (k_proj)
      Step::Gemv  (v_proj)
      Step::Attend (kv_write + flash_attn)  ← per-token, works with q8 ring-buffer
      Step::Gemv  (o_proj)
      Step::Gemv  (gate_proj)
      Step::Gemv  (up_proj)
      Step::Gemv  (down_proj)
    final_norm + lm_head (Step::Gemv) + softcap
```

Every projection is a single-row GEMV — launched 48 × 6 = 288 kernel launches per token.

### 3.2 Target flow (batched WMMA prefill)

```
forward_prefill_batch_wmma():
  embed all tokens → pb_residual [B, dim]

  for each layer:
    // Batched projections via WMMA
    rmsnorm_batched(pb_residual, ...) → pb_tmp
    GemmFamily::run(q_proj, pb_tmp → pb_q)    ← WMMA [B×q_dim, K] GEMM
    GemmFamily::run(k_proj, pb_tmp → pb_k)
    GemmFamily::run(v_proj, pb_tmp → pb_v)
    rmsnorm_batched + rope_batched

    // Per-token attention (unchanged — works with q8 ring-buffer)
    for each token:
      Step::Attend(kv_write + flash_attn)

    GemmFamily::run(o_proj, pb_q → pb_attn_out)
    rmsnorm_batched + residual_add

    // Dense FFN
    rmsnorm_batched
    GemmFamily::run(gate_proj, ...) → pb_gate
    GemmFamily::run(up_proj, ...) → pb_up
    gelu_tanh + mul
    GemmFamily::run(down_proj, ...) → pb_ffn_out
    rmsnorm + residual_add + layer_scalar

  final_norm + lm_head + softcap (on last token only)
```

**Key design choices:**
- **Batched GEMM for projections, per-token attention.** Attention is O(N²) in seq_len but each token's Q·K is O(head_dim × seq_len) — not the bottleneck for short-to-medium prefill. The projections dominate: 6 GEMMs per layer × 48 layers = 288 weight matrix traversals per token.
- **Per-token attention preserved.** The q8 ring-buffer KV write and flash attention work correctly per-token. No need for batched attention (which would require ring-buffer-aware batched kernels). This avoids the v1/v2 q8 KV bug entirely.
- **GemmFamily::run() auto-selects WMMA.** On gfx1100+ (HasWmma), the registry resolves `GemmHfq4G256Wmma` or `GemmQ8_0Wmma`. On older archs, it falls back to scalar `GemmHfq4G256` / `GemmQ8_0BatchedChunked`.

### 3.3 F32→F16 staging

The WMMA kernels accept F16 input, not F32. The dispatch framework already has `ensure_fp16_x()` which does pointer-keyed F32→F16 conversion with caching. The existing WMMA GEMM methods (`gemm_hfq4g256_wmma`, `gemm_q8_0_wmma`) call this internally.

**But:** `GemmFamily::run_key` dispatches to `gpu.gemm_hfq4g256_wmma(w.buf, x, y, m, k, batch_size)` which takes the raw `x` tensor (F32). The GPU method itself calls `ensure_fp16_x` internally. So the F32→F16 staging is already handled — no changes needed in the calling code.

Wait — let me verify this. Looking at `gemm_hfq4g256_wmma` at line 19038 of gemm.rs, it takes `x_f16: &GpuTensor` and `y_f32: &GpuTensor`. But the dispatch arm at line 150 passes `x` directly:

```rust
K::GemmHfq4G256Wmma => hip!(gpu.gemm_hfq4g256_wmma(w.buf, x, y, m, k, batch_size)),
```

And `GemmParams.x` is a `&GpuTensor` which is F32. So the GPU method must handle the conversion. Let me check... Actually, looking at the GPU method signature, it takes `x_f16` — but the dispatch passes the raw F32 tensor. This means either (a) the tensor is already F16 by the time it reaches here, or (b) there's a mismatch.

Looking at how the existing qwen2/DeepSeek V4 prefill paths work — they use `GemmFamily::run()` for prefill projections and it works. The F32→F16 conversion must be handled somewhere in the chain. I need to verify this during implementation (Step 0 below).

## 4 · Implementation plan

### Step 0 — Verify WMMA prefill plumbing (30 min)

Before writing any gemma4 code, verify the WMMA GEMM path works end-to-end for an existing arch (qwen2 or DeepSeek V4). Run a prefill that uses `GemmFamily::run()` and confirm it selects WMMA on gfx1151.

**Verify:**
1. `GemmFamily::run()` resolves to `GemmHfq4G256Wmma` on gfx1151
2. The F32→F16 staging happens correctly
3. Output is byte-identical to scalar path

### Step 1 — Add `run_prefill_gemm_wmma` helper (1 hour)

Replace `run_prefill_gemm()` with a version that routes through `GemmFamily::run()` instead of hardcoding the scalar key:

```rust
fn run_prefill_gemm_wmma(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    batch_size: usize,
) -> HipResult<()> {
    let ctx = DispatchCtx::new(gpu);
    let w_ref = WeightRef { ... };
    let params = GemmParams { w: &w_ref, x, y, batch_size };
    llama::gemm_family()
        .run(&ctx, gpu, &params)
        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))
}
```

This auto-selects WMMA when available. Falls back to scalar on older archs.

**Gate:** existing per-token decode still works (Step::Gemv path unchanged).

### Step 2 — Write `forward_prefill_batch_wmma` (3 hours)

A new function that batches projections but runs attention per-token. Roughly modeled on the existing `forward_prefill_batch_v2` structure but:
- Uses `run_prefill_gemm_wmma` for all projections
- Runs per-token attention (not batched) — reuses existing `Step::Attend` path
- Does NOT need batched MoE for 12B dense (no MoE)
- For 26B-A4B MoE: per-token expert loop (same as current decode path)

The function:
1. Embeds all tokens into `pb_residual [B, dim]`
2. For each layer:
   a. `rmsnorm_batched` → `pb_tmp`
   b. Batched GEMM: q_proj, k_proj, v_proj → `pb_q`, `pb_k`, `pb_v`
   c. `rmsnorm_batched` for Q/K/V norms + `rope_batched_f32`
   d. **Per-token loop**: for each token, copy q/k/v to per-token scratch, run `Step::Attend` (kv_write + flash_attn), copy output back
   e. Batched GEMM: o_proj → `pb_attn_out`
   f. Residual add, pre-FFN norm
   g. Batched GEMM: gate_proj, up_proj, down_proj
   h. gelu_tanh + mul, residual add, layer_scalar
3. Final norm + lm_head + softcap (only on last token for sampling)

**Key difference from v2:** attention is per-token (not batched), so q8 ring-buffer KV works correctly. No batched q8 write/attention kernels needed.

### Step 3 — Wire into daemon (1 hour)

Same structure as the jukefr wiring on the preserved branch, but calling `forward_prefill_batch_wmma`:

```rust
const PREFILL_BATCH_THRESHOLD: usize = 16;
const PREFILL_BATCH_SIZE: usize = 128;
if prompt_ids.len() >= PREFILL_BATCH_THRESHOLD {
    // Chunk and call forward_prefill_batch_wmma
    // Last token re-run for logits
}
```

Default ON (no env opt-in needed — WMMA GEMM is byte-identical to scalar).

### Step 4 — Coherence validation (1 hour)

1. Short prompt ("Capital of France?") — argmax must match per-token path
2. Long prompt (1266 tokens) — summary must be coherent
3. 26B-A4B model — coherent output
4. Oracle comparison: argmax at position 1024+1200 must match HF

### Step 5 — Perf measurement (30 min)

Measure tok/s on gfx1151 for:
- 12B dense, 20-token prompt
- 12B dense, 1266-token prompt
- 26B-A4B, 20-token prompt
- 26B-A4B, 1266-token prompt

Compare against per-token baseline. Expected: **5–10×** improvement on the projection-dominated path (actual speedup depends on how much of the total time is projections vs attention vs other).

## 5 · Risks

| Risk | Mitigation |
|---|---|
| F32→F16 staging doubles memory traffic for inputs | `ensure_fp16_x` has pointer-keyed caching; same x is reused for q/k/v projections |
| Per-token attention loop is still slow for long prefill | Attention is O(N²) but N≤128 per chunk; projections dominate. Can add batched attention later |
| WMMA kernel not registered on gfx1151 | HasWmma predicate should pass on gfx1151 (RDNA3.5). Verify in Step 0 |
| 12B model uses Q8_0 weights — `GemmQ8_0Wmma` must handle them | Q8_0 WMMA kernel exists and is registered. Microbench shows 11–30× over scalar |
| 26B-A4B MoE per-token expert loop dominates | MoE optimization is separate work (indexed kernels already exist for decode). Prefill MoE batching deferred |
| `forward_prefill_batch_wmma` needs batch scratch tensors | Already allocated in `Gemma4Scratch::new()` — `pb_*` buffers sized for `MAX_PREFILL_BATCH=128` |

## 6 · Out of scope

- **Batched attention for prefill** — per-token attention is correct with q8 ring-buffer and fast enough for B≤128
- **MoE prefill batching** — per-token expert loop is adequate; indexed kernels handle decode
- **v1/v2 batched prefill debug** — preserved on `feat/gemma4-batched-prefill-jukefr` for reference
- **MFMA (CDNA) prefill** — WMMA (RDNA) only for now. CDNA path uses different kernel variants
- **Prefill benchmarking against llama.cpp** — Milestone 2 work. Milestone 1 targets WMMA activation

## 7 · Estimated timeline

| Step | Time | Dependency |
|---|---|---|
| Step 0: Verify WMMA plumbing | 30 min | — |
| Step 1: `run_prefill_gemm_wmma` helper | 1 hr | Step 0 |
| Step 2: `forward_prefill_batch_wmma` | 3 hr | Step 1 |
| Step 3: Daemon wiring | 1 hr | Step 2 |
| Step 4: Coherence validation | 1 hr | Step 3 |
| Step 5: Perf measurement | 30 min | Step 4 |
| **Total** | **~7 hr** | |

---

*Plan authored 2026-06-09. Jukefr's batched-prefill approach preserved on `feat/gemma4-batched-prefill-jukefr` (commit `e833f552`) for reference.*
