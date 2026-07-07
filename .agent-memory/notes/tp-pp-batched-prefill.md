---
title: TP/PP batched prefill (single-batch <=256) — feature complete on feature/device-mesh
date: 2026-07-07
tags: [device-mesh, tp, pp, prefill, batched-prefill, step-gemm, execute_steps_tp, prefill_forward_band, dense-serve, mq4g256, rotation]
---

**Branch:** `feature/device-mesh` (worktree `.claude/worktrees/feature+device-mesh`). Commits `10184aa0..6426d068` (5). Spec/plan (local, gitignored `docs/superpowers/`): `specs/2026-07-07-tp-pp-batched-prefill-design.md`, `plans/2026-07-07-tp-pp-batched-prefill.md`. Built via subagent-driven-development (5 tasks + per-task review + whole-branch review); SDD ledger `.superpowers/sdd/progress.md`. Continues the device-mesh pivot — see [[device-mesh-pivot-execute-steps-spine]].

## What it does
The dense multi-GPU serve paths (`TpModel` tensor-parallel, `PpModel` pipeline-parallel) prefilled the prompt ONE TOKEN AT A TIME (`generate_dense` loop). This adds **batched prefill**: one forward for prompts `n <= PREFILL_MAX_BATCH (256)`; `n > 256` falls back to the per-token path (no cross-chunk in this cut — that needs the flash `forward_prefill_batch` cache-attention, deferred). Llama-family qk-norm arches (0/1), MQ4G256 + Q8 KV, stateless per request.

## Architecture (mirrors the existing TP=Steps / PP=imperative asymmetry, on the prefill axis)
- **Seam:** `DenseServed::prefill(&[u32])` trait method, default = the old per-token loop (byte-identical fallback). `generate_dense` calls it once. `daemon.rs`.
- **T1 shared core:** extracted `prefill_forward`'s layer loop into `llama::prefill_forward_band(gpu, w, cfg, x_batch[n×d], layers: Range, kv, positions[n]i32, scratch: &PrefillScratch, batch)` + `PrefillScratch::{alloc,free}`. Byte-identical extraction. `prefill_forward` still returns LAST-position logits.
- **PP (imperative banded):** `PpModel::prefill` runs `prefill_forward_band` per stage over `self.bands[s]` / `self.kv[s]`, `[n×dim]` `boundary_copy` (n*dim*4) between stages, last-position row → `scratch[last].x` for the unchanged `logits()`. Per-stage KV is `new_gpu_q8(g, n_layers, ...)` = ABSOLUTE-indexed, so band's absolute `layer_idx` just works. **Parity BYTE-EXACT** (max|Δ|=0 vs single-GPU `prefill_forward`).
- **TP (batched Steps):** new `Step::Gemm { w, x, y, batch }` (hipfire-dispatch steps.rs) — B>1 GEMM. `TpModel::prefill` batch-embeds on rank0 + broadcasts `[n×d]`, builds per-rank batched step lists run through `execute_steps_tp`: col `Step::Gemm` qkv/gate-up, batched `Step::Attend` on owned heads, row `Step::Gemm` wo/down → `TpCollective::AllReduceOut{dim: n*d}` → `Step::ResidualAdd` (once, after the collective). Last-position row → `ranks[0].x`. **Parity argmax-identical** (max|Δ|=0.85 = TP reads Q8 KV cache vs the F32 in-batch reference — not a bug; the argmax gate IS discriminating). `tp_decode_parity` unchanged (decode uses `Step::Gemv`).

## Traps / hard-won facts (the load-bearing ones)
- **`execute_steps_tp` is already batch-agnostic** — it all-reduces the out buffer with a caller-supplied `count`. TP batching = `Step::Gemm` carries `[n×*]` bufs + `AllReduceOut{dim: n*d}` (element count). No executor-loop change.
- **`Step::Gemm` MQ4G256 rotation** uses the DISPATCH-layer `GemvFamily::rotate(batch_size=n)` (NOT the runtime `rotate_x_mq_batched_for` — unreachable from hipfire-dispatch) → `gpu.gemm_hfq4g256_batched_lmhead`. HFQ4G256/HFQ4G128 dispatch direct.
- **LATENT BUG fixed (rotation.rs):** `RotationFamily`'s batched `Plain`/`AWQ` arms resolved the batched kernel key but called the SINGLE-ROW `rotate_x_mq`/`_awq` → only row 0 of the `n×k` activation rotated. Nothing hit those arms with batch>1 before (all callers batch_size=1) so it was dead/latent; TP's `Step::Gemm` is the first batched caller. Fix = call `rotate_x_mq_batched`/`_awq_batched`. **Regression-safe** (batch=1 arms untouched). This is why the TP parity CAUGHT it: argmax 13 (Δ=99.8) pre-fix → 7281 (==ref) post-fix.
- **`gpu.scratch.mq_x_rot` must be grown before batched `Step::Gemm`.** `prepare_rotation_scratch` (gemv.rs:416) ALIASES the persistent buffer (sized for B=1 decode) — it does NOT grow it. `TpModel::prefill` grows each rank's to `n × max_k` where `max_k = d.max(q_dim_r).max(inter_r)` (largest rotation INPUT dim). Too-small = silent OOB/garbage.
- **Batched `Step::Attend` writes its B keys to KV internally** (`run_attention`→`dispatch_kv_write` has `*_batched` arms on `io.batch_size`) → NO separate KV-write step needed.
- **Step IR has NO batched rmsnorm/rope/qknorm forms** (`Step::RmsnormAutomatic`/`Rope`/`QkNorm` call the B=1 kernels). TP prefill is a HYBRID: batchable ops (`Gemm`/`Attend`/`SiluMul`/`ResidualAdd`) through `execute_steps_tp`; the 3 non-batchable ops as direct per-rank `rmsnorm_batched`/`rope_batched_f32`/qknorm calls between segments (same kernels as `prefill_forward_band` → parity holds).
- **Both prefill paths write the SAME Q8 KV that decode reads** → no prefill→decode gap (the only load-bearing handoff; whole-branch-review-verified).

## Deferred follow-ups (reviewer-recommended, NON-blocking)
1. **[highest] Tighten `tp_prefill_parity` to a numeric bound** — it's argmax + eyeball-Δ only (PP sibling asserts `<1e-3`). Use an all-Q8-KV reference (both sides read the cache) OR compare all `n` positions → ~1e-3 bound. The one place a TP precision regression could slip past the greedy-argmax guard.
2. **>256 cross-chunk batched prefill** — needs the flash `forward_prefill_batch` cache-attention (deferred from this cut).
3. Revert/shrink or document the one-time `mq_x_rot` prefill VRAM growth (persists for the daemon session).
4. AWQ-batched rotate (`rotation.rs (true,true)` arm) coverage — untested by this feature (MQ4G256 is non-AWQ).
5. Live `serve --tp 2` coherence + perf capture (deferred as environmentally fiddly; the daemon forwarder is a 3-line delegation to the exact path the parity validates).

## Pre-existing (surfaced, OUT OF SCOPE — file separately)
`hipfire-loader/src/lib.rs:1702` panics `"pp>1 must carry pp_gpus"` on unload after a qwen35 pp>1 session — the loader's `Model` teardown, orthogonal to the dense `PpModel`/`TpModel` (which own their `Gpus`, Drop-teardown). This feature does not touch or worsen it.
