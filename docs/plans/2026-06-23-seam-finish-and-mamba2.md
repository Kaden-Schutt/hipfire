# Finish the serving seam, migrate every arch onto it, then land Mamba-2

Status: **active** — scoped 2026-06-23. Owner: chaingun.
Supersedes the open phases of [daemon-family-seam](2026-06-19-daemon-family-seam.md)
and absorbs the Mamba-2 work from
[arch-roster-feature-matrix](2026-06-19-arch-roster-feature-matrix.md) (E4/E5) and
[multi-family-master-plan](2026-06-19-multi-family-master-plan.md).

## Goal

hipfire grew organically: the daemon/serving stack is welded to qwen35 and every
new family is bolted on as a bespoke `generate_*` path behind an `arch_id`
ladder. This plan **finishes the serving seam** (the strangler-fig started in the
daemon-family-seam plan), **routes every existing arch through it**, deletes the
per-arch dispatch, and **then adds full Mamba-2 support** as a clean consumer of
the finished seam. Training is explicitly **out of scope** for this plan.

## Current reality (measured 2026-06-23, corrects the stale plan numbers)

The daemon-family-seam plan was written against an 18k-line `main.rs`; the daemon
has since been refactored to **5,452 lines**, with serving extracted into
`hipfire-serving-core`. Re-measured state:

| Layer | State today |
|---|---|
| Seam traits (`Architecture`, `ArchCaps`, `SimpleAr`, `ServingBackend`, `GenerateCtx`, `run_simple_ar`/`decode_loop`) | **Defined** in `crates/hipfire-runtime/src/arch.rs` (P0 ✓) |
| Crate-side `ServingBackend` impls | **qwen2, gemma3, gemma3-vl only** (P2/P3 ✓) |
| Archs actually *served through* the seam | **Zero** — every arch routes through a bespoke `generate_*` behind an `arch_id` ladder (`hipfire-serving-core/src/generate.rs:1865+`, `load.rs:526+`) |
| `LoadedModel` Option-soup | **Intact** (`hipfire-serving-core/src/model.rs`) |
| qwen35 coupling | 219 `qwen35`/`DeltaNet`/`q35_` refs in daemon `main.rs`; `is_qwen35_family_arch_id` branches across `session.rs`, `qwen35_prefill.rs`, `qwen35_decode.rs` |

So P1, P4, P5, P6 and the unresolved **daemon-wiring decision point** are all
open. This plan closes them, then adds P7 (Mamba-2).

## Arch roster, by serving complexity (drives the phase order)

- **Dense AR → `SimpleAr` only:** llama (0/1), qwen2 (7), gemma3 (12).
- **Vision-splice prefill → `serve` override + shared `decode_loop`:** gemma3-vl (13), dots-ocr (10).
- **Bespoke recurrent/MoE loops → `serve` override behind `ArchCaps`:** minimax (8, lightning-attn), lfm2moe (11, short-conv + MoE), deepseek4 (9, MTP).
- **Risk-concentrated → `serve` override, perf-critical:** qwen35 (5/6) — DFlash, MTP, pipeline-parallel, grouped-MoE, hybrid DeltaNet+KV state.
- **New → clean `SimpleAr`/`ServingBackend`:** mamba2 (pure SSM, **no KV**).

## The load-bearing design problem

The seam's `State` associated type implicitly assumes "a KV cache." The real
roster has **heterogeneous per-layer state**: KV (attn) + conv-state (short-conv)
+ recurrent S-state (DeltaNet) + SSM state (Mamba-2), mixed per layer. The honest
abstraction is a **per-layer `Mixer` state list**, not a monolithic KV cache.

**Mamba-2 is the forcing function** that makes this real: a pure no-KV arch
cannot hide behind the KV-cache assumption. Getting the per-layer `Mixer` state
model right (P2) is what lets both "migrate everything" and "add Mamba-2" land as
infrastructure rather than two more special cases. The existing qwen35 hybrid
(FA + DeltaNet) already proves heterogeneity and is the validation anchor for P2.

## Decisions locked (2026-06-23)

- **Mamba-2 lands last (P7)**, after the seam is built and every transformer arch
  is migrated — lowest integration risk; Mamba-2 is a pure consumer of finished infra.
- **qwen35 fast paths migrate fully (P5)**, gated by `ArchCaps`, so the
  Option-soup can be fully deleted in P6. This is the riskiest phase and is
  therefore ordered last among migrations, on a proven seam.

## Phases (strangler-fig; commit each; coherence gate on every forward-touching step)

1. **P1 — neutralize the base. ✅ Already satisfied on `chaingun` (verified
   2026-06-23).** The original premise ("5 arch crates depend on qwen35;
   relocate `KvCache`/scratch/session-state out of it") no longer holds — prior
   branch work already broke it:
   - `KvCache` already lives in a neutral home, `hipfire-runtime/src/kv.rs`
     (the seam plan's "lives in `hipfire-runtime::llama`" is stale).
   - The four sibling arch crates (qwen2, minimax, qwen35-vl, dots-ocr) carry
     **no real `hipfire-arch-qwen35` dependency** — the Cargo references that
     remain are comments, and there are zero `hipfire_arch_qwen35::` symbol uses
     in their source.
   - `hipfire-runtime`'s only `hipfire-arch-qwen35` edge is a **dev-dependency**
     (examples/tests); its library source references qwen35 only in doc
     comments. So the arch→runtime direction is clean (no cycle).
   - Proof: `cargo tree -i hipfire-arch-qwen35 -e normal` lists **only**
     `hipfire-daemon` and `hipfire-serving-core` as non-dev dependents.

   The residual qwen35 coupling is therefore confined to the **serving /
   fast-path layer** (serving-core + daemon: `DeltaNetState`, `Qwen35ScratchSet`,
   `LayerType`, `StateQuant`, `pflash`, `mtp`, `speculative::ModelSlot`). Those
   types are genuinely DeltaNet/qwen35-specific and belong with the qwen35
   backend — they are dissolved by **P3 (daemon wiring)** and **P5 (qwen35 fast
   paths behind `ArchCaps`)**, not by a P1 type-relocation. **Net: P1 needs no
   code move; proceed directly to P2.**
2. **P2 — generalize `State` into a per-layer `Mixer` state model.** Define
   `enum Mixer { FullAttn, Swa, ShortConv, DeltaNet, Mamba2 }` + per-layer
   heterogeneous state with a **no-KV path**. The design keystone. Validate
   against the existing qwen35 FA+DeltaNet hybrid.
   - **P2a ✅ (taxonomy keystone landed).** New `hipfire-mixer` crate — pure,
     GPU-free `MixerKind` + `MixerProfile` with `needs_kv_cache()` (the no-KV
     detector that replaces `is_qwen35_family_arch_id`), `is_hybrid()`, per-kind
     counts. 4 unit tests (pure-SSM/pure-attn/qwen35-hybrid) pass in the no-GPU
     subset. Buffer layouts deliberately unpinned — migration reuses existing
     `KvCache`/`DeltaNetState` allocations.
   - **P2b ▸ in progress.** `MixerProfile` is now the authoritative source for
     qwen35 KV-topology across serving-core, via a shared
     `qwen35_mixer_profile(layer_types)` helper (`FullAttention→FullAttn` KV,
     `LinearAttention→DeltaNet` no-KV). Consolidated three hand-rolled
     `layer_types == FullAttention` derivations, all **behavior-preserving**:
     - `session.rs` `qwen35_allocate_session_state` (fp32) KV mask →
       `MixerProfile::kv_layer_mask()`.
     - `load.rs` load-time KV mask (fp32) → `kv_layer_mask()`.
     - `load.rs` CASK/TriAttention eviction `fa_layer_ids` →
       `MixerProfile::kv_layer_indices()`.

     **Deliberately deferred (NOT behavior-preserving):** flipping the q8/asym
     KV-mode branches from dense (`config.n_layers`) to the existing
     `new_gpu_q8_capped_filtered` is a real **VRAM optimization** (placeholders
     for DeltaNet layers), and it interacts with the spec-decode snapshot/rollback
     path — so it needs `coherence-gate-dflash` + the q8/max256 perf gate, not a
     blind edit. Tracked as a P2b/P5 optimization, not done here. The
     spec-decode `n_fa_layers` count (`load.rs` DDTree setup) is left for P5.
     **Next:** build the `MixerLayerState` buffer model on the taxonomy. NB the
     `is_qwen35_family_arch_id` branches are qwen35 *fast-path* gates, **not**
     KV-topology — dissolved in P5, not rewired here.
3. **P3 — wire the daemon to the seam + migrate the dense archs.** Resolve the
   daemon-wiring decision point via the **full-collapse** route (goal is
   organization, not a quick ship): thread the daemon sampler/sessions/streaming
   through `decode_loop` so the seam is not greedy-only, and route
   llama/qwen2/gemma3 through `ServingBackend` end-to-end.
4. **P4 — migrate the bespoke loops.** minimax, lfm2moe, deepseek4, and the VL
   archs (gemma3-vl, dots-ocr) as `serve` overrides behind `ArchCaps`.
5. **P5 — migrate qwen35 (risk-concentrated).** DFlash/MTP/PP/grouped-MoE as
   `serve` overrides gated by `ArchCaps`. **Perf gates mandatory** — canonical
   q8/max256 bench, byte-identical prompts, the ±5% investigation rule. Must NOT
   be forced through `SimpleAr::decode_step`.
6. **P6 — collapse.** `LoadedModel` Option-soup → single
   `backend: Box<dyn ServingBackend>`; delete the `generate_*` free functions and
   the `arch_id` / `is_qwen35_family_arch_id` ladders. The 219 daemon qwen35 refs
   → ~0.
7. **P7 — Mamba-2 on the finished seam (the payoff).**
   - New **SSD / selective-scan kernel** (the one genuinely new kernel):
     `h_t = exp(dt·A) ⊙ h_{t-1} + dt·B·x_t; y_t = C·h_t + D·x_t`. Sequential
     single-token decode step + chunked-SSD prefill. f32 first, q8 later
     (mirror the GDN q8 work). Cannot reuse `gated_delta_net.hip` (different
     recurrence: scalar-per-head decay vs delta-rule outer-product).
   - **conv1d xBC variant** over `[x, B, C]` (dim `d_inner + 2·n_groups·d_state`)
     — new split layout over the existing `conv1d_silu_split` kernels.
   - **Reuse:** `gated_norm.hip` is Mamba-2's `RMSNormGated`; conv-state rolling
     cache and the no-KV `Mixer` state from P2.
   - `hipfire-arch-mamba2` crate implementing `SimpleAr` over `Mixer::Mamba2`;
     loader for HF `Mamba2ForCausalLM` (safetensors) + the state-spaces format
     (`pytorch_model.bin` pickle, minimal `ssm_cfg`, implicit layer defaults).
   - Validate on local `/srv/huggingface/models--state-spaces--mamba2-130m`
     (clean, no attn/MoE confounds) → coherence + numeric bisect vs HF reference.
   - **nemotron_h** (Mamba2 + attn + MoE hybrid, local Nemotron-3 checkpoints)
     composes for free because P2 made per-layer mixers real.

## Effort & risk

| Phase | Effort | Risk |
|---|---|---|
| P1 neutralize base | M | Low (refactor) |
| P2 Mixer state model | M–L | Low runtime, high leverage |
| P3 daemon wiring + dense | M | Med — touches production streaming/sampler |
| P4 bespoke loops | M | Med, mostly mechanical |
| P5 qwen35 fast paths | L | **Highest** — perf-critical; do last on proven seam |
| P6 collapse | S–M | Low — deletion |
| P7 Mamba-2 + nemotron_h | L (SSD kernel) + M (crate/loader) | Med — integration cheap post-seam |

## Validation vehicles (local, under /srv/huggingface)

- `state-spaces/mamba2-130m`, `mamba2-2.7b` — pure SSM kernel validation.
- `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (+ 30B-A3B, Super-120B) — hybrid compose.
- Existing qwen35 / gemma3 / lfm2moe checkpoints — seam-migration regression anchors.

## Testing

- `./tests/no-gpu-ci.sh` after each structural (trait/type) step.
- `./tests/coherence-gate-dflash.sh` on any forward/kernel/dispatch change.
- Perf gates (q8/max256 canonical, byte-identical prompts, ±5% rule) on P5.
- Quiet box only for perf gates (UMA APU bandwidth contention → false regressions).

## Out of scope

- Training (SSD backward, Mamba-2 fine-tune). `hipfire-train`'s `ssm_block` /
  `gated_scan` is a GLA-lite drafter, **not** Mamba-2; deferred.
- Vulkan / cross-vendor backend (project rule 4/7).
- Block-diffusion, audio/omni, image-gen (later epochs of the master plan).
