# Cross-arch `WeightBackend` adoption (qwen2 + llama) — Design

**Date:** 2026-06-11
**Branch:** `feature/paro-transparent-loading`
**Status:** design (approved)
**Context:** Gap #2 from the unified-loading review
(`unified-loading-review-todos` memory). qwen35 already runs all three of its
loaders through `layer_driver::load_layer` over `WeightBackend`; qwen2 and llama
do not. This closes that gap with **trait-only adoption** (not a generic
arch-agnostic walk — that was tried as `WeightSpec`/`LayerSpec` in commit
`611e10ce` and reverted in `f2a5895b` as "superseded by weight_backend +
load_layer").

## Goal

Migrate qwen2 and llama **single-tensor** loading onto the existing
`WeightBackend` trait, keeping each arch's own layer-walk and `LayerWeights`
struct. Removes the per-arch duplicated dequant helpers and collapses llama's
two loaders into one.

## Non-goals

- A generic, arch-agnostic `load_layer` (rejected; already reverted once). Each
  arch keeps its own walk because the `LayerWeights` structs genuinely differ
  (qwen2 has biases; llama is dense-only; qwen35 has MoE/DeltaNet variants).
- qwen35 (already migrated).
- Multi-GPU for qwen2/llama (neither has a multi-GPU path; out of scope).
- Embedding / tied-lm_head assembly unification — stays arch-local (the same
  "residual fork 2" decision qwen35 made).

## Why this is low-risk

The `WeightBackend` trait was already built with the two exact knobs cross-arch
adoption needs:

- `candidates: fn(&str) -> Vec<String>` — `flat_name_candidates` resolves the
  `model.layers.{i}.…` flat names qwen2/llama use, vs `hf_name_candidates` /
  `qwen35_tensor_name_candidates` for qwen35's nested `model.language_model.…`.
- `norm_bias: f32` — `0.0` for qwen2/llama (standard RMSNorm), `1.0` for
  qwen3.5/gemma (`(1 + weight)`).

The only **new** trait surface is `bias()`.

## Design

### §1 Trait change (one addition)

In `crates/hipfire-runtime/src/weight_backend.rs`, add to `trait WeightBackend`:

```rust
/// Load a bias vector (f32). Only qwen2 attention biases use this today.
fn bias(&mut self, rel: &str, n: usize) -> HipResult<GpuTensor>;
```

- `HfqBackend::bias` wraps today's qwen2 `load_bias_f32` logic (resolve via
  `self.candidates`, dequant/upload as f32, length `n`).
- `ParoBackend::bias` returns `Err("ParoBackend: biases unsupported")` — no PARO
  arch carries attention biases. Honest dead surface (accepted trade-off).

qwen35 and llama walks never call `bias()`.

### §2 qwen2 migration (HFQ-only)

`crates/hipfire-arch-qwen2/src/qwen2.rs`:

- `load_layer` (currently `qwen2.rs:351`) keeps its `Qwen2LayerWeights` return
  type. Its body is rewritten to:
  ```rust
  let mut b = HfqBackend { hfq, gpu, norm_bias: 0.0,
      candidates: flat_name_candidates, read_proj: <candidates-aware reader>, layer: i };
  // attn_norm  = b.norm("input_layernorm.weight", &[hidden])?
  // wq/wk/wv/wo = b.proj(...); wq_bias/wk_bias/wv_bias = b.bias(...)
  // ffn_norm   = b.norm("post_attention_layernorm.weight", &[hidden])?
  // w_gate/w_up/w_down = b.proj(...)
  ```
  `read_proj` is a fn-ptr of signature `fn(&HfqFile, &Gpu, &str, usize, usize,
  fn(&str)->Vec<String>) -> HipResult<WeightTensor>` (the field qwen35 sets to
  its `load_weight_tensor`); qwen2 supplies a candidates-aware reader of that
  shape — its current 5-arg `load_weight_tensor` is widened to take the
  `candidates` fn, or a shared runtime reader is used.

  **Name construction is already correct — no new builder needed.**
  `HfqBackend::proj` calls `hfq_proj_name(layer, rel)` → `layers.{i}.{rel}.weight`
  (prefix-less); `flat_name_candidates` then prepends `model.` →
  `model.layers.{i}.{rel}.weight`, byte-for-byte today's qwen2 names. Norms use
  `hfq_plain_name` (`layers.{i}.{rel}`) + `flat_name_candidates`. Relative leaves
  follow the bare-leaf convention `load_layer` uses (`self_attn.q_proj`; bias as
  `self_attn.q_proj.bias`).
- Delete qwen2's local `load_weight_tensor`, `load_norm_weight_raw`,
  `load_bias_f32` once unused (the `TODO(transformer-extraction)` helpers).
- No Dir/PARO path is added (none exists; `Qwen2Carrier` is HFQ-only).

qwen2 has **no** `q_norm`/`k_norm` and **no** MoE — its walk stays a flat dense
walk, not routed through qwen35's `load_layer` (different struct).

### §3 llama migration (collapse two loaders into one)

`crates/hipfire-runtime/src/hfq.rs` (+ `crates/hipfire-runtime/src/llama.rs`):

- Add `llama::load_layer<B: WeightBackend>(b, config, i) -> HipResult<LayerWeights>`
  producing the dense-only `llama::LayerWeights` (`llama.rs:658`):
  wq/wk/wv/wo, optional `q_norm`/`k_norm` gated on `config.has_qk_norm`,
  ffn_norm, w_gate/w_up/w_down. **No MoE closure** (llama `LayerWeights` has no
  MoE variant — unlike qwen35's `load_layer`).
  The qk-norm optionality stays a conditional in the walk: call `b.norm(...)`
  only when `has_qk_norm`, wrap in `Some`.
- `load_weights_hfq` (`hfq.rs:748`): build
  `HfqBackend { norm_bias: 0.0, candidates: flat_name_candidates, .. }`, funnel
  each layer through `load_layer`. **Keep** its qwen2-bias rejection guard
  (`hfq.rs:~769`) and its embed / tied-lm_head handling (arch-local).
- `load_weights_paroquant_llama` (`hfq.rs:1117`): **deleted**. `LlamaCarrier`'s
  Dir arm (`carriers.rs:239`) instead builds `ParoBackend` and funnels through
  the same `load_layer`; PARO transparency comes from the existing
  `ParoAugmentor`. The Dir arm keeps its own embed/output + tokenizer/template
  resolution (already there).
- Delete llama's local `load_weight_tensor` / `load_norm_weight_raw` /
  `load_f16_tensor` (norm path) once unused — **but only after §Risk-1 proves
  byte-identical**.

### §4 Verification (byte-identical gate)

Prerequisite — source models into `~/.hipfire/models/`:
- one HFQ qwen2 (arch_id 7),
- one HFQ llama / Qwen3-dense (arch_id < 5),
- one PARO/safetensors llama directory.

Procedure (mirrors the qwen35 migration's net):
1. **Pre-refactor capture:** first-token logits (or a hash of all loaded
   `WeightTensor`/`GpuTensor` device buffers) for each of the three models.
2. **Post-refactor assert:** identical capture from the unified path.
3. `./scripts/coherence-gate.sh` green for qwen2 + llama (HFQ) and the PARO
   llama — fluent, on-topic, no verbatim loop.
4. `cargo build` clean; deleted fns gone; no new `dead_code` warnings;
   `load_weights_paroquant_llama` removed and `LlamaCarrier` Dir arm green.

**Blocked on #1:** the byte-identical capture and coherence gate require a
compiling tree. Gap #1 (`qwen35.rs:1387 shallow_clone` E0599) must be confirmed
green before this verification can run.

## Risks

1. **Norm dequant equivalence (highest).** llama currently loads norms via
   `load_f16_tensor`; `HfqBackend::norm` routes through
   `dequant_norm(info.quant_type, …, norm_bias=0.0)`. These must be
   byte-identical for llama's stored norm dtype. The byte-identical gate (§4) is
   the proof; do not delete `load_f16_tensor` until it passes. Also confirm
   `flat_name_candidates` resolves the same tensor name `load_f16_tensor` used.
2. **Name-prefix construction (low — already handled).** `hfq_proj_name` /
   `hfq_plain_name` are prefix-less; `flat_name_candidates` owns the `model.`
   prefix, exactly reproducing qwen2/llama's `model.layers.{i}.…` names. No new
   builder is written — the residual check is only that `read_proj` is wired to
   a candidates-aware reader (else the `candidates` fn is ignored). Caught by the
   byte-identical gate regardless.
3. **`bias()` on `ParoBackend`.** Returns `Err`; ensure no walk except qwen2's
   (HFQ-only) ever calls it. Compile-checked by construction (only qwen2 calls
   `bias`, and qwen2 has no `ParoBackend`).

## Out of scope / follow-ups

- Generic arch-agnostic `load_layer` (rejected).
- qwen2 safetensors/PARO + multi-GPU (no path exists).
- Embedding/output unification across arches (stays arch-local).
