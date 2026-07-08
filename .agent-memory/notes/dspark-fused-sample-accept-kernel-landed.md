---
title: DSpark temp>0 verify — fused on-GPU sample+accept kernel LANDED (byte-identical, perf-neutral; head <1% of ds4 window confirmed again)
date: 2026-07-02
tags: [dspark,spec-decode,deepseek4,sampler,gpu-residency,kernel,byte-identical,lazy-verify]
---

Branch feature/dspark-qwen3. Replaced the per-position `sample_top_p_pf` host loop
in DSpark temp>0 sampled verify (`final_norm_and_sample_all_batched_lazy`,
deepseek4 forward.rs) with a single fused on-GPU kernel. Requested by user:
"for dspark we use sample_top_p_pf … each a launch + tiny D2H … replace with fused
kernel entirely on gpu."

## What shipped
- **New kernel** `kernels/src/dspark_sample_accept_lazy.hip` (`dspark_sample_accept_lazy_f32`):
  over the resident batched target logits `[n × vocab]`, replays the single-block
  `sample_top_p` draw per verify position (top-K gather → softmax(temp) → top-p →
  xorshift32 categorical), threads the RNG across positions, and LAZILY early-exits
  on the first mismatch vs `draft[pos+1]`, padding the tail `u32::MAX`. Output
  `out[n+1]` = sampled ids + advanced rng. Penalty phase dropped (DSpark passes all
  penalties off). One launch + one `(n+1)×4`-byte D2H replaces τ×(launch + 8-byte
  D2H + host sync).
- Wrapper `Gpu::sample_accept_lazy_f32` (rdna-compute/src/sampling.rs); SRC reg in
  kernels.rs.
- forward.rs `final_norm_and_sample_all_batched_lazy` rewritten: **batched head**
  (`compute_batched_head_logits`, one 565 MB lm-head read for all n) + fused kernel.
  FWHT-head / `HIPFIRE_DEEPSEEK4_BATCH_HEAD=0` keep the old per-position fallback.
  Mirrors the greedy `final_norm_and_argmax_all_batched` architecture.

## Validation (all PASS)
- **Kernel parity gate** `rdna-compute/examples/sample_accept_parity.rs`: 60/60 cases
  **byte-identical** (ids + new_rng) vs per-position `sample_top_p_pf` — n=2..8,
  temp 0.3–1.0, top_p 0.9–1.0, top_k 0/20/40, 4 seeds × 3 draft regimes
  (full-accept / immediate-mismatch / partial). No GPU model needed; run under lock.
- **End-to-end** dspark_bench temp0.7 top_p0.95 (LRU code, prompt-md5 939c222a,
  token-md5 bb914484, warmup128/max128): NEW (batched+fused, default) vs OLD
  (`BATCH_HEAD=0`, old per-position path) = **byte-identical output** (same 128 ids,
  τ=2.462, accept=0.292, windows=52) and **coherent** (clean LRU class, no attractor).
- Fused kernel dispatch **confirmed live** on the real `deepseek-v4-flash.mq2lloyd`
  (AMD_LOG_LEVEL ShaderName `dspark_sample_accept_lazy_f32`, once/window) — ds4 head
  is NON-FWHT so the batched path is taken, not the fallback.

## Perf: NEUTRAL (within ±1–3% ds4 noise). WHY it doesn't move the needle
15.06 vs 15.31 tok/s (−1.6%, noise). The change eliminates τ per-position head reads
+ τ host round-trips, but that is **<1% of the ds4 DSpark verify window — the MoE
trunk forward dominates** (re-confirms [[mtp-lmhead-not-the-lever]] and the greedy
finding in [[dspark-ds4-greedy-lazy-verify-falsified]]). So this is a correctness-
preserving GPU-residency cleanup (fully on-GPU sampler, no per-position sync), NOT a
tok/s win on ds4. Would matter more on an arch where the head/D2H is a larger window
fraction (e.g. smaller-trunk qwen3 DSpark — untested here).

## gfx1151 LDS gotcha (cost ~2 debug cycles)
The single-block `sample_top_p` uses 128 threads → **64 KiB dynamic LDS**, which
**aborts on gfx1151** (`HSA_STATUS_ERROR_INVALID_ALLOCATION` 0x1003) — gfx1151's
usable dynamic LDS is < 64 KiB (its parallel sampler tops out at 40 KiB, which is why
production never uses the single-block path here). The fused kernel therefore launches
**64 threads → 32 KiB LDS**; byte-identical because the top-K gather + tree reduction
select the same global top-64 for any thread count (RNG draw is thread-0 only).

## Follow-ups
- Re-run the paused DSpark block-modulation sweep now that this landed — per
  [[dspark-tau-adaptive-block-modulation-resume]] the per-window cost shifted (though
  measured ≈flat, re-confirm the block=2/5 code peaks before the switch-axis decision).
- Not committed yet; fmt-clean (0 rustfmt changes on all touched files). Run clippy +
  commit when ready.
