# DSpark → qwen35 (arch_id 6, MoE DeltaNet-hybrid) — implementation sketch

**Branch:** `feature/dspark-qwen35` (stacked on `feature/dspark-qwen3` / PR #492, which
generalized DSpark into an arch-agnostic core and ported it to qwen3-*dense* via the
llama crate).

**Goal:** run DSpark speculative decode with the ORNITH-1.0-35B target + its shipped
DSpark draft, where the **target** is `hipfire-arch-qwen35` (arch_id 6), *not* the
dense-qwen3/llama path #492 already covers.

## The two models (verified from HF configs + safetensors headers)

**Target** — `pablogrant/ORNITH-1.0_35B_AEON_..._BF16`, 70 GB BF16.
`Qwen3_5MoeForConditionalGeneration` / `model_type: qwen3_5_moe` ⇒ **arch_id 6**.
- 40 layers, DeltaNet **hybrid**: `linear_attention ×3 → full_attention` (`full_attention_interval:4`);
  full-attn layers = `[3,7,11,15,19,23,27,31,35,39]`.
- hidden 2048, attn 16h / 2kv / **head_dim 256**, **partial_rotary 0.25** (rotary dim 64),
  `attn_output_gate:true`, θ=1e7, mrope `[11,11,10]`.
- MoE: 256 experts / top-8, `moe_intermediate_size 512`.
- **MTP head present** (`mtp_num_hidden_layers:1`, tied embeddings) — an alternative spec path,
  not what we're wiring.
- vocab 248320, eos 248046, `mask_token_id 248077`.

**Draft** — `..._DSPARK-DRAFT_BF16`, 1.7 GB BF16.
`DSparkDraftModel` / `speculators_model_type: dspark` v0.6.0. EAGLE-3-flavored DSpark:
- **3 dense Qwen3 layers** (`layers.0..2`): q[4096,2048] k/v[512,2048] o[2048,4096],
  QK-norm[256], MLP inter 6144. head_dim 256, partial-rotary 0.25 (matches target).
- `fc`[2048,6144] = `[hidden, 3·hidden]` fuses aux hidden from target layers
  **`aux_hidden_state_layer_ids:[9,19,29]`** (n_targets=3) → `hidden_norm` → layer stack → `norm`.
- `embed_tokens`[248320,2048] full vocab; **`lm_head`[32000,2048] reduced draft vocab**;
  `d2t`[32000] i64 + `t2d`[248320] bool vocab maps.
- `markov_head` (vanilla, rank 256): `markov_w1`[248320,256] indexed by prev **target** token
  → `markov_w2`[32000,256] draft-vocab logit bias.
- `confidence_head.proj`[1,2304] (=2048 hidden ⊕ 256 markov) **+ bias**[1];
  `confidence_head_with_markov:true`.
- `block_size:8`, `speculative_tokens:7`, greedy proposal.
- val_metrics: **accept_rate ≈ 0.275, mean accept_len ≈ 1.67** (pos-1 0.43 → pos-7 0.21).
  ⇒ a *modest* drafter — expect small adaptive blocks and a modest τ, not a large tok/s jump.

## ⚠ Prerequisite blocker (found by validating the mq6, 2026-07-03)

The target mq6 quantized cleanly (27.7 GB, 100% of text params) **but does not load** — the daemon
panics: `tensor not found: layers.0.mlp.experts.0.gate_up_proj.weight`.

**Root cause:** ORNITH-35B stores its qwen35-MoE experts **un-stacked** — separate
`model.language_model.layers.{L}.mlp.experts.{N}.{gate,up,down}_proj.weight` 2D tensors (the
DeepSeek-V4-style layout). The quantizer's arch-6 path (`main.rs:7333/7360`, "will split 3D expert
tensors per-expert") assumes Qwen3.5's **canonical stacked-3D** `experts.gate_up_proj`
(`[num_experts, 2·inter, hidden]`); it has **no branch to fuse separate per-expert gate+up** into the
`experts.{X}.gate_up_proj.weight` the arch-6 loader requires (`qwen35.rs:508-510`). So the experts
fell to the generic 2D path with their original separate names → the loader can't find the fused
tensor. This blocks **any** use of ORNITH (even plain AR), independent of dspark — it's the true
first task.

### Task A0 (prerequisite): fuse pre-split experts for arch 6 in the quantizer
- In the arch-6 (`is_moe`) ingest, detect the pre-split layout (`experts.{N}.gate_proj`/`up_proj`
  present, no stacked `experts.gate_up_proj`).
- Per expert N: read `gate_proj` [inter, hidden] + `up_proj` [inter, hidden], **vstack gate-then-up**
  → [2·inter, hidden] — Qwen convention: forward does `silu(gu[:inter]) * gu[inter:]`, so order is
  load-bearing (a swap silently lobotomizes; the coherence gate catches it). Quantize via the normal
  k-map/MQ6 path, emit as `experts.{N}.gate_up_proj.weight`. `down_proj` already matches the loader
  name — just route it through the expert path.
- Validate: re-quant → daemon load → factual coherence smoke (the check that caught this).
- Alternatively a narrow pre-pass (stack experts in a copy of the safetensors) — rejected: ~70 GB
  re-save, disk-bound, hacky. The quantizer branch is the clean fix and helps any future un-stacked
  qwen35-MoE checkpoint.

## What already exists and is reused unchanged

The engine is arch-generic; qwen35's hard recurrent-state machinery is battle-tested.

| Piece | Location | Status |
|---|---|---|
| DSpark engine loop (bootstrap→draft_block→run_heads→conf-truncate→verify→accept→commit→multi-slot ctx) | `hipfire-runtime/src/dspark_core.rs` (`DsparkDrafter::mtp_step:1046`) | reuse |
| `DsparkBody` trait (drafter seam) + `build_dspark_speculator:1346` | `dspark_core.rs:210` | reuse |
| τ-adaptive block controller (cost-model argmax) | `hipfire-runtime/src/dspark_block_controller.rs` | reuse |
| Kernels: fused sample+accept, KV stage | `kernels/src/dspark_sample_accept_lazy.hip`, `dspark_stage_kv.hip` | reuse |
| MtpDrafter/MtpSpeculator adapter + `generate_spec` unified loop + daemon | `spec.rs:736/799`, `daemon.rs:generate_spec:4273` | reuse, **zero daemon changes** |
| Dense-qwen3 drafter forward (bidirectional block, target-ctx KV fusion, confidence-with-markov) | `hipfire-arch-llama/src/dspark_body.rs` (`dspark_qwen3_block_forward:634`, `Qwen3DsparkBody:1115`) | reuse, **make dims config-driven** |
| qwen35 DeltaNet snapshot/rewind + verify + partial-accept replay | `hipfire-arch-qwen35/src/spec_impl.rs` (`verify_block:209`, `commit_prefix:326`), `speculative.rs` (`DeltaNetSnapshot`, `verify_dflash_block` w/ `num_extract`) | reuse |
| qwen35 SpecTargetGuard (#462 borrow guard) + emitter | `hipfire-loader/src/spec_build.rs:41`, `hipfire-arch-qwen35/src/spec_emit.rs` | reuse |

**Precedent for a *separate draft model* on qwen35 already exists** = DFlash
(`dflash_spec.rs`, loads a `-dflash` sidecar). DSpark rides the same "separate model +
sidecar" shape, but as an `MtpSpeculator<DsparkDrafter>`.

## Delta vs the qwen3-8B drafter #492 built for (drives the new work)

| | qwen3-8B (#492) | ORNITH qwen35 (this) |
|---|---|---|
| target arch | dense qwen3 = llama (1) | **qwen35-MoE DeltaNet hybrid (6)** |
| drafter layers / hidden / head_dim | 5 / 4096 / 128 | **3 / 2048 / 256** (partial-rotary 0.25) |
| n_targets (aux layers) | 5 `[1,9,17,25,33]` | **3 `[9,19,29]`** |
| draft vocab | full 151936 | **reduced 32000 (+d2t/t2d)** |
| config schema / arch name | top-level keys, `Qwen3DSparkModel` | **speculators v0.6.0** (`aux_hidden_state_layer_ids`, `transformer_layer_config`), **`DSparkDraftModel`** |
| block_size | 7 | 8 |

## The port — task breakdown

**A. Quantizer** (`hipfire-quantize/src/main.rs`, generalize the `qwen3-dspark-q8` block @6304).
- Accept arch names `DSparkDraftModel`/`DSparkSpeculator` (currently only `Qwen3DSparkModel`).
- Read the **speculators v0.6.0** schema: `aux_hidden_state_layer_ids` (not `target_layer_ids`),
  `block_size`, `draft_vocab_size`, `mask_token_id`, `transformer_layer_config.{num_hidden_layers,
  head_dim, hidden_size, partial_rotary_factor, rope_theta, intermediate_size}`.
- **Carry `d2t`(i64) + `t2d`(bool) + reduced `lm_head`[32000,·]** into the sidecar.
- Emit metadata (mirrors `DsparkConfig::from_metadata_json`): add `dspark_draft_vocab_size`,
  `dspark_head_dim`, `dspark_partial_rotary`, `dspark_confidence_with_markov`, keep
  `dspark_block_size/target_layer_ids/markov_rank/noise_token_id/enable_confidence`.
- Drafter quant recipe unchanged: attn/MLP 2D → Q8F16, everything else (norms, embed,
  main_proj/norm, markov, confidence, lm_head, d2t/t2d, bias) → F16/raw. **(See "drafter
  precision" under Open decisions — mq6 on this modest head is risky.)**

**B. Reduced-vocab remap in dspark-core** (`dspark_core.rs` — *genuinely new; not plumbed today*).
- Add `d2t`/`t2d` to `DsparkWeights`; carry `draft_vocab_size` in `DsparkConfig`.
- `run_heads:649`: lm-head + markov_w2 produce **draft-space** logits [32000]; argmax in draft
  space → **`d2t` → target token id** before verify/commit. markov_w1 is indexed by prev
  *target* token (no remap). Confidence unaffected (operates on hidden ⊕ markov embed).

**C. Config-driven drafter dims** (`hipfire-arch-llama/src/dspark_body.rs`).
- Ensure `dspark_qwen3_block_forward` / `Qwen3DrafterAssets` read head_dim, partial_rotary,
  hidden, n_layers, n_targets from metadata (this drafter is hd256/pr0.25/2048/3L/3-targets vs
  #492's hd128/4096/5L/5-targets). Generalize anything hardcoded to the qwen3-8B shape.

**D. qwen35 target capture hooks** (`hipfire-arch-qwen35/src/spec_impl.rs` — *the bulk*).
Currently missing ⇒ they hit the `Err` defaults and DSpark fails at runtime. Each layers a
hidden-extraction sink onto an existing, working forward:
- `verify_block_capture_gpu` — reuse `verify_block:209` (DeltaNet snapshot + verify + replay) +
  `verify_dflash_block` with `num_extract = 3` into a `HiddenStateRingBuffer`.
- `verify_block_sampled_capture_gpu` — temp>0 twin of the above (`verify_block_sampled:241`).
- `capture_seed_main_hidden` — 1-slot bootstrap: advance recurrent+KV by the seed, capture
  extract-layer hidden (template: `hipfire-arch-llama/src/spec_impl.rs:384`).
- Parameterize `new_spec_scratch:117` (`num_extract` is hardcoded 0) and expose
  `dflash_extract_layers = [9,19,29]` on `ModelSlot` (default `None` today).

**E. qwen35 carrier DSpark arm** (`hipfire-loader/src/carriers.rs`, mirror llama `584-705`).
- Sidecar discovery `<stem>-dspark.<ext>`; load (reuse `load_qwen3_dspark`/`build_qwen3_dspark_body`
  since the drafter *is* dense qwen3); `build_dspark_speculator`.
- Precedence **DSpark > DFlash > MTP > n-gram**, inserted before the `build_speculator` registry
  (`spec_build.rs:158`) in the qwen35 `load`.

**F. Gates / parity** (new: `scripts/coherence-gate-qwen35-dspark.sh`).
- Parity harness vs a golden forward (DeepSpec `modeling.py`, cited in the #492 topology doc):
  `x_head` cosine, markov token-identical, confidence cosine — catches aux-layer/convention bugs
  *before* perf.
- `serve-multiturn-gate.sh` (**mandatory** — DeltaNet recurrent-state bleed across requests is the
  #462 class; the hardest correctness axis here).
- τ/tok-s bench with byte-identical committed prompts (prompt-md5 rule).

## Correctness-critical risks (ranked)

1. **DeltaNet state rewind ↔ EAGLE-3 hidden capture consistency across partial accept.** The
   captured accepted-prefix hidden feeds the next block's bootstrap; it must align with qwen35's
   committed KV+recurrent positions. #462 hazard. → `serve-multiturn-gate` + multi-slot ctx audit.
2. **Aux-hidden extraction convention.** `fc` expects the *exact* residual point the drafter was
   trained on, at layers `[9,19,29]` (mixed linear/full — extraction is residual-stream so
   arch-agnostic, but the *position* convention must match). → parity `x_head` cosine.
3. **Reduced-vocab d2t remap** (task B) — off-by-one/space-mismatch silently tanks acceptance.
4. **Modest drafter (0.275 accept).** Set expectations: adaptive block will settle low; the win
   is bounded by accept_len ≈ 1.67. Validate τ on the daemon, not a demo (memory: demos under-report).

## Effort estimate

Mostly mechanical given the existing machinery. Rough order: **D** (target hooks) > **B**
(reduced vocab) > **A** (quantizer) > **E** (carrier) > **C** (dims) > **F** (gates). No new
kernels, no daemon changes.

## Status / decisions (2026-07-03)

- Branch `feature/dspark-qwen35` created; both models downloaded (target 70 GB BF16, draft 1.7 GB);
  disk freed (pruned re-downloadable `Qwen3.5-27B` base cache, 52 GB).
- **Drafter precision:** Q8/F16 (decided). Target = mq6. The drafter *sidecar* itself is deferred to
  task A (it needs the quantizer + loader arms; nothing consumes it until then). Raw BF16 on disk.
- **Target mq6 — DONE** (`~/.hipfire/models/ornith-35b-aeon.mq6`, 27.7 GB, text-only; re-quant with
  `--include-vision` for multimodal). Task **A0 landed** (commit `54e99d9d`): loads + coherence 0/0.
- **Drafter sidecar — DONE** (`~/.hipfire/models/ornith-35b-aeon-dspark.mq6`, 1.5 GB, Q8/F16). Task
  **A landed** (commit `eb66c4f1`): `--format qwen35-dspark-q8`, metadata complete.
- **Remaining = the engine** (none started, coupled — not testable until all land): **D** (qwen35
  capture hooks, the crux) · **B** (reduced-vocab remap) · **C** (drafter dims) · **E** (carrier) ·
  **F** (gates).

## Task order (revised)

**A0** (quantizer: fuse pre-split experts — unblocks even plain AR) → then the dspark port:
**D** (target capture hooks) > **B** (reduced vocab) > **A** (drafter-sidecar quantizer) > **E**
(carrier) > **C** (dims) > **F** (gates).
