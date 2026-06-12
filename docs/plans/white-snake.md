# White Snake — Speculative Drafter Design

**Status:** design + partial implementation · **Date:** 2026-06-12 · **Target:** Qwen3.5/3.6 DeltaNet-hybrid (gfx942 / MI300X)
**Branches:** inference infra on `feat/vector-plate-q8ef`; trainer on `feat/dflash-lfm2-drafter`.

## Thesis

White Snake is a **KL-distilled block-diffusion speculative drafter with DeltaNet recurrent-state coupling**, targeting **Orthrus-level acceptance (τ≈11.7 vs DFlash's 7.9)** on DeltaNet-hybrid targets that Orthrus itself cannot serve (Orthrus assumes per-layer softmax KV; DeltaNet layers have none — only a recurrent state S).

It is "**Orthrus for DeltaNet hybrids**":
- **block-diffusion parallel drafting** (DFlash architecture — one forward predicts a masked block; cheap),
- **KL-distillation** (Orthrus's τ-lifting objective — match the teacher's *full* distribution, not hard tokens),
- **recurrent-state coupling via the GDN tape** (the DeltaNet analog of Orthrus's shared KV; the byte-exact q8_ef rewind is the substrate),
- with **tree branching** as an optional multi-path acceptance lever.

The τ lift comes from **training** (KL + state-coupling), not from autoregression. We verified this: an autoregressive (MTP-AR/EAGLE-style) drafter is path-conditioned but pays N sequential forwards and was capacity-starved at 1 layer (τ=1.91). Block-diffusion is one forward and Orthrus proves the linear τ ceiling is high (11.7) for a *strong* block drafter.

## Why this shape (the convergence)

1. Orthrus is infeasible directly on Qwen3.5/3.6: its diffusion view shares per-layer KV, which DeltaNet layers don't have.
2. DFlash already *is* a block-diffusion drafter for the hybrid (separate softmax drafter consuming the trunk's per-position hidden states — DeltaNet-agnostic). It hits τ≈7.9 trained on hard targets.
3. Orthrus's edge over DFlash (11.7 vs 7.9) is the **objective** (forward-KL distillation) + **state sharing**, not the architecture. So: keep DFlash's block-diffusion body, swap CE→KL, add DeltaNet state-coupling.
4. Tree branching (Vector Plate) is a secondary lever — it catches the positions where the block's top-1 misses. Banked as `--ddtree-path-c phase2` (break-even + free acceptance); the single-pass full-tree win additionally needs a stronger drafter (this one).

## Architecture

| Component | What | Lifts |
|---|---|---|
| **Block-diffusion body** | DFlash-shape, ~5-layer, masked-block parallel (z-lab drafter capacity, per_pos 0.273) | capacity + one-forward draft |
| **KL-distillation training** | forward-KL to the trunk teacher's top-k distribution | **τ (7.9 → ~11.7)** — the main lever |
| **GDN-tape state-coupling** | drafter conditions on the trunk's DeltaNet recurrent state; verify rewinds S byte-exactly per branch | DeltaNet-native fidelity + lossless tree |
| **Tree branching** | multi-path tree from per-position top-k, verified single-pass | extra acceptance on top-1 misses |

## Built this session (infrastructure — drafter is the only missing piece)

- **Verify**: single-pass tree verify RoPE-fixed + coherent (`6db0d861`, attractor dead); `--ddtree-path-c phase2` banked.
- **Rewind**: byte-exact q8_ef DeltaNet rewind — `DeltaNetSnapshot` now saves `s_ef_residual` (`a66c51d0`; probe: 0-diff with fix, 255-diff without). This is the state-coupling substrate.
- **Tree-GDN kernel**: deterministic q8_ef (parent-indexed EF residual tape, `7341b5ec`).
- **KL objective**: `kl_topk_loss_bwd_f32` (gradcheck-validated 3e-7) + top-k teacher datagen (`HIPFIRE_DUMP_TOPK`, `.topk` HFTK format).
- **zlab-init**: finetune the 5-layer z-lab DFlash drafter (`HIPFIRE_ZLAB_INIT`); confirmed **KL > CE on robustness**.
- **MTP-AR scaffold**: `build_mtp_tree_ar` (tree expansion + per-branch state forking) — reusable for tree construction; the autoregressive *drafter* is not the chosen path, the scaffold is.

## Training plan

- **Drafter architecture**: z-lab DFlash 5-layer block-diffusion (capacity-proven). Do NOT reinvent — `dt::Cfg::zlab_27b` + `load_zlab` exist.
- **Objective**: forward-KL distillation from the trunk teacher (top-k logits), the Orthrus recipe. Built.
- **State-coupling (the new training feature)**: condition the drafter on the trunk's recurrent DeltaNet state, not just post-layer hiddens. v1: dump a projection of the trunk S (or the GDN-tape recurrence summary) per position alongside the hidden taps, feed it as an extra `fc` conditioning input. v2 (research): a DeltaNet-native bidirectional chunk drafter that re-runs the recurrence.
- **Data**: regenerate a **larger, cleaner** top-k teacher set (the 40-seq loopy greedy set caused forgetting; use more seqs + filter low-uniqueness + diverse prompts).
- **Regime**: from-scratch KL on the z-lab arch, or a *gentle* finetune that doesn't catastrophically forget. Gate on the held-out per_pos trajectory climbing, not just train.

## Trainer extension (the delta to build — feat/dflash-lfm2-drafter)

Current trainer already has: block-diffusion CE, `HIPFIRE_KL_DISTILL` (KL), `HIPFIRE_ZLAB_INIT` (z-lab finetune), top-k datagen. The White Snake delta:

1. **State-coupling conditioning input** — datagen dumps the trunk's per-position DeltaNet state summary (projection of S, or the GDN-tape α/β/qkv summary); trainer's `fc` consumes it as an extra channel alongside the 5 hidden taps. Gated `HIPFIRE_WS_STATE_COUPLE=1`.
2. **Clean/large data pipeline** — bigger top-k regen, low-uniqueness filtering, diverse seeds (avoid the loopy-greedy degradation).
3. **From-scratch z-lab-arch KL training path** (vs finetune-only) so the τ lift can be measured cleanly without converged-model forgetting.

## Validation

- **τ target**: beat DFlash (7.9) toward Orthrus (11.7) on canonical code/instruct prompts, **online** τ (not offline argmax — the MTP-head dead-end was offline gains that died online).
- Gates: `coherence-gate-dflash.sh` (attractor checks), byte-identical prompt + md5, warm + fresh-process medians.
- **Prove/disprove**: does KL + state-coupling lift online τ over the CE-trained z-lab DFlash baseline? (KL>CE robustness already shown; the open question is the magnitude of the τ lift on a properly-trained, capacity-adequate drafter.)

## Risks

- **Training capacity/data** — the recurring wall this session. The z-lab arch is capacity-adequate, but the in-repo trainer must produce a *converged* drafter (the LFM2-350M body underfit; finetuning a converged drafter on thin data degrades). Mitigation: from-scratch on z-lab arch + clean/large data.
- **State-coupling value uncertain** — the post-layer hiddens already encode S; explicit S-conditioning may add little over hidden taps. Measure v1 (hidden-only KL) first; add state-coupling only if it moves τ.
- **Draft cost** — block-diffusion is one forward (cheap), so unlike the MTP-AR route, draft cost is not the bottleneck; the verify (body GEMMs) is, which is why tree is secondary and a high-τ linear drafter is the primary win.
