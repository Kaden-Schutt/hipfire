---
title: DSpark→qwen35 (ORNITH-35B) — PORT PROVEN CORRECT (fwd+heads parity byte-match DeepSpec); τ=0 ROOT-CAUSED = BROKEN drafter ckpt (all-ones untrained norms), NOT hipfire
date: 2026-07-03
tags: [dspark,spec-decode,qwen35,arch6,moe,quantize,gate_up_proj,eagle3,ornith,blocker,feature-dspark-qwen35]
---

## STATUS 2026-07-03 — quantization COMPLETE, engine port NOT started
- **A0 (expert fusion) DONE + committed `54e99d9d`**: arch-6 quantizer now fuses pre-split
  `experts.{N}.gate_proj`+`up_proj`→`gate_up_proj` (gate||up, [1024,2048]). `ornith-35b-aeon.mq6`
  (27.7GB, text-only, 10240 experts fused) LOADS + coherence probe 0-hard/0-soft 8/8.
- **A (drafter sidecar) DONE + committed `eb66c4f1`**: `qwen3-dspark-q8` generalized to
  DSparkDraftModel/speculators-v0.6.0 (nested `transformer_layer_config`, `aux_hidden_state_layer_ids`,
  d2t/t2d→F32, full `dspark_*` metadata). `ornith-35b-aeon-dspark.mq6` (1.5GB, 44 tensors).
  Alias `--format qwen35-dspark-q8`. Both models on disk under `~/.hipfire/models/`.
- **ENGINE INTEGRATED (commit 3d42af18) + debug aid (b451ee8b):** B (dspark_core reduced-vocab d2t remap),
  C (llama drafter: rope_theta 1e7 + reduced 32k lm_head + d2t load + partial-interleaved RoPE n_rot=64),
  D (qwen35 SpecTarget hooks verify_block_capture_gpu/_sampled/capture_seed_main_hidden over DeltaNet;
  capture_seed side-effect-free snapshot→forward→restore; ModelSlot.dspark_extract_layers field),
  E (qwen35 carrier arm in lib.rs finish_qwen35_load, DSpark>DFlash>MTP>ngram), F (coherence-gate-qwen35-dspark.sh).
  Daemon compiles + runs; DSpark engages (block=8, layers=[9,19,29], draft_vocab=32000); output COHERENT.
- **BLOCKER τ=0.00** (exact, even on trivial predictable seq w/ closed_think — NOT thinking-OOD): drafter accepts
  ZERO drafts. HIPFIRE_DSPARK_DEBUG=1 shows drafts DEGENERATE (low/repetitive/markov-dominated, non-contextual:
  100,0,108) vs correct target picks; capture_seed main_hidden non-zero but rms≈0.05 (residual usually ~1-10 →
  suspect). Bug in drafter path: capture point/scale/layer-order (D) OR partial-rotary halfsplit/n_rot (C) OR a
  #492-class numeric convention. d2t verified lossless (NOT it). **NEXT: x_head parity harness vs DeepSpec ref
  (~/dspark-work/DeepSpec modeling.py) — the tool #492 used for double-norm/RoPE-offset/confidence.**
  Sidecar discovery = `<target-stem>-dspark.<ext>` (⇒ ornith-35b-aeon-dspark.mq6).
- **SYSTEMATIC DEBUG (2026-07-04) — RULED OUT (all τ still 0.00):** (1) layers off-by-one — swept
  HIPFIRE_DSPARK_LAYER_OFFSET -1/0/+1, all τ=0; reference `extract_context_feature` (DeepSpec
  common.py:52) = raw concat of `hidden_states[layer_id+1]` (OUTPUT of layers [9,19,29]) matches
  hipfire post-layer capture. (2) hidden scale (rms 0.05) — RED HERRING: fc→hidden_norm is RMSNorm,
  scale-invariant. (3) thinking-OOD — τ=0 exact even w/ closed_think + predictable seq. (4) d2t —
  lossless F32 round-trip. (5) dims — all derive correctly from tensor shapes (hd256/16h/2kv/6144/3L).
  (6) rope style — halfsplit, SAME kernel the coherent TARGET uses (rope_partial_interleaved_f32_batched
  n_rot=64). Drafts DEGENERATE/markov-dominated ⇒ x_head not encoding context ⇒ subtle drafter-FORWARD
  numeric bug (block-attn ctx handling / fc ingest / a #492-class convention). **BLOCKED on x_head parity:
  `speculators` lib NOT installed; DeepSpec dspark/qwen3/modeling.py is FULL-rotary (needs Qwen3.5
  partial-rotary adaptation). Parity ex template = crates/hipfire-arch-llama/examples/qwen3_dspark_parity.rs.**
  Debug: HIPFIRE_DSPARK_DEBUG=1 (drafts vs picks + capture hidden stats).
- **ZERO-CTX PROBE (HIPFIRE_DSPARK_ZERO_CTX=1, commit 4d874c1b):** drafts DIFFER real-vs-zeroed ctx ⇒
  drafter IS wired to + attends the context, but output stays DEGENERATE/low-id (frequent-token) either
  way. Not a disconnected context — a subtle FORWARD under-conditioning (weak x_head → lm_head defaults
  to frequent tokens). Suspects: attention phase (partial-rotary halfsplit/positions), a norm, or fc.
  Commits this session: 54e99d9d eb66c4f1 3d7848f5 3d42af18 b451ee8b fad6ae04 4d874c1b.
- **ROOT CAUSE FOUND (2026-07-04, x_head parity harness, commit 07e7556a) — NOT a hipfire bug.**
  Built torch parity: `crates/hipfire-arch-llama/examples/qwen35_dspark_parity.rs` (GPU side) +
  `~/dspark-work/ornith_dspark_cpu_ref.py` (DeepSpec Qwen3DSparkModel + FAITHFUL partial-rotary,
  byte-exact vs transformers qwen3_5 rope; strict=False forward-weights-only load). Torch env:
  `.venv` + `nix develop` + LD_LIBRARY_PATH `/nix/store/si4q3...gcc-15.2.0-lib/lib` (+ zlib for numpy)
  + PYTHONPATH=~/dspark-work/DeepSpec. RESULTS: (a) main_x cosine **1.000000**, (b) x_head cosine
  **0.999873**, and run_heads (lm_head+markov+d2t) **byte-identical** to ref on the same x_head
  (`[0,74,0,...]`). ⇒ entire hipfire drafter MATH (fc/partial-rotary attn/heads/d2t) is CORRECT.
  **The drafter CHECKPOINT is broken:** every learnable RMSNorm weight (norm/hidden_norm/all layer
  norms/QK-norms) is EXACTLY 1.0 std=0.0 (untrained/unexported) while matmuls (q_proj/fc/lm_head
  std~0.02) are trained. Contrast the WORKING qwen3-8B DSpark drafter (~/dspark-work/qwen3/ckpt):
  norm mean 1.52/std 0.14, q_norm mean 1.39/std 0.17 (TRAINED). Trained matmuls w/o their learned
  norm scaling ⇒ drafter can't predict ⇒ τ=0. **FIX = re-export the drafter with trained norms
  (pablogrant/author-side), NOT engine code.** (all-ones norms could theoretically be folded-into-
  matmuls, but the same DeepSpec pipeline exported qwen3-8B with UNFOLDED trained norms ⇒ broken.)

## Scope (branch feature/dspark-qwen35, off feature/dspark-qwen3/PR#492)
Port DSpark spec-decode to **qwen35 MoE arch_id 6** (the DeltaNet-hybrid crate), target =
`pablogrant/ORNITH-1.0_35B_AEON_*` (a Qwen3_5MoeForConditionalGeneration **VL** finetune:
`model.language_model.*` text + `model.visual.*`; 40L linear×3→full every 4th, 256 exp/top-8,
hd256, partial-rope 0.25). Draft = `*_DSPARK-DRAFT_BF16` = EAGLE-3 DSpark head: **3-layer dense
qwen3**, fuses target hidden [9,19,29] (`fc`[2048,6144]), block_size 8, **reduced 32k draft vocab
(d2t/t2d)**, vanilla markov rank256 + confidence head. val: accept≈0.275, accept_len≈1.67 (MODEST →
small adaptive block, modest τ). Full sketch: `docs/design/2026-07-03-qwen35-dspark-port.md`.

## KEY FINDING: the hard parts are already done by PR#492 + qwen35
PR#492 generalized DSpark into arch-agnostic `dspark_core.rs` (DsparkBody trait, τ-block controller,
kernels, MtpDrafter/generate_spec) + ported to qwen3-**dense**/llama. qwen35 already has DeltaNet
snapshot/rewind + verify_block/commit (the hard recurrent-state part). Port = target-side: 3 capture
hooks on `impl SpecTarget for ModelSlot` (qwen35/spec_impl.rs, missing→Err defaults), carrier arm,
quantizer arm. One genuinely-NEW core piece: **reduced-vocab d2t remap (NOT plumbed anywhere today** —
#492's drafter used full vocab). Drafter forward = reuse `Qwen3DsparkBody`/`dspark_qwen3_block_forward`
but make dims config-driven (this head hd256/2048/3L/3-target vs #492 hd128/4096/5L/5-target).

## BLOCKER (Task A0, found by validating the mq6 — validation earned its keep)
`--format mq6` on ORNITH produced a clean 27.7GB mq6 that **panics on load**:
`tensor not found: layers.0.mlp.experts.0.gate_up_proj.weight`. ORNITH stores experts **un-stacked**
(separate `experts.{N}.{gate,up,down}_proj`, DeepSeek-V4-style). The arch-6 quantizer
(main.rs:7333/7360 "split 3D expert tensors per-expert") assumes Qwen3.5 **canonical stacked-3D**
`experts.gate_up_proj` and has NO branch to fuse separate gate+up → the loader's per-expert
`experts.{X}.gate_up_proj.weight` (qwen35.rs:508-510). Experts fell to generic 2D path → unloadable.
**Fix:** arch-6 ingest, detect pre-split, **vstack gate-then-up** ([2·inter,hidden], order load-bearing:
`silu(gu[:inter])*gu[inter:]`; swap = silent lobotomy, coherence gate catches), emit `gate_up_proj`.
Blocks even plain-AR ORNITH. Broken mq6 still at `~/.hipfire/models/ornith-35b-aeon.mq6` (delete+re-quant
after A0). Vision skipped (text-only; `--include-vision` if VL wanted).

## TRAPS
- `tail -N` on a quant/build cmd DROPS the early arch-detection banner from the captured log.
- Drafter (Q8/F16, user decision) sidecar deferred to task A — nothing consumes it until loader exists.
- Disk tight (1.8T @94%); freed 52G by pruning re-downloadable Qwen3.5-27B hf-cache. Related [[dspark-tau-adaptive-block-modulation-resume]].
