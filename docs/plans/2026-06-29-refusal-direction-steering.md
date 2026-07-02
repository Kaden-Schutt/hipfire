# Refusal-direction steering / abliteration — MVP plan

Status: READY — design finalized, residual-boundary + LoRA verification done.
First build step is the `hipfire-steer` crate + a gemma3 block-boundary hook.
Date: 2026-06-29
Target model (MVP): MedGemma (Gemma-3 family, incl. the `-vl` variant).

## Motivation

Some models over-refuse their *intended* audience. The concrete driver is
**MedGemma declining legitimate clinical/research queries** with disclaimer
boilerplate. Example — a brain-MRI image plus "I have a headache" yields:

> It is not appropriate for me to provide medical advice. … If you are
> experiencing a headache, especially if it's severe … it is crucial to seek
> medical attention immediately. A doctor can properly evaluate your symptoms …

This is not "decensoring" in the edgy sense — it is suppressing a
**medical-disclaimer refusal mode** in a model whose whole purpose is medical
reasoning for professionals. It is the cleanest possible refusal-direction
target: the contrast set is "medical questions it answers" (good) vs. "medical
questions it deflects" (bad), and the refusal markers are disclaimer phrases
("not appropriate", "seek medical attention", "consult a doctor", "I cannot
provide medical advice").

The same machinery generalizes to the whole **contrastive-direction family** —
sycophancy, forced cheerfulness, persona traits — by swapping only the +/−
prompt sets and the scoring objective. Build it once, get the family.

## Background — the technique and our reference

Heretic (`third_party/heretic`, AGPL, vendored) is the reference implementation
of *automatic abliteration*: collect residual-stream activations on a +set and
−set of prompts, derive a per-layer **refusal direction**
`dir = normalize(mean_bad − mean_good)` (optionally orthogonalized against the
good direction), then either **ablate** it from the weights or **steer** the
residual, and search the apply-strength against two co-objectives: refusal count
↓ and KL-divergence-from-base ↓ (capability preservation). Heretic uses an
Optuna TPE multi-objective optimizer over ~5 params/component and exports the
result as a merged model or a LoRA adapter.

### The algebraic insight that drives our design

Directional **ablation of a weight equals projecting the direction out of the
activation**:

```
W·a − λ·v·(vᵀW·a)  =  o − λ·v·(vᵀo)        where o = W·a
```

So we never need to edit (and re-quantize) weights to ablate. Ablation and
steering collapse into **one runtime op on the residual buffer**:

- **Steer** (additive):   `x += α·v`
- **Ablate** (projective): `x −= λ·(vᵀx)·v`

Both run on the **quantized model, with no LoRA and no quant round-trip**, in
prefill and decode alike.

## Verification findings (2026-06-29)

Two parallel Explore passes confirmed the build surface.

### Residual boundary — uniform *as a buffer*, not as a discrete op

The llama-style discrete `add_inplace_f32(x, o)` does **not** hold uniformly:
decode paths and **all MoE outputs fuse** the residual add into the
projection/combine kernel (`weight_gemv_residual`, `moe_down_combine_k8_batched`,
`…moe_down_residual_scaled…`). qwen35 is the most fused; gemma3 is the cleanest
(discrete `add_f32` in *both* prefill and decode, by design — post-norms sit
inside the residual so fusion is impossible, see `gemma3/src/forward.rs:9-10`).

But the residual stream is **uniformly an addressable f32 buffer**
(`state.x` / `state.h` / `x_residual`) in every arch and path. Therefore:

- **Tap (capture): uniform.** Read the residual buffer after any block.
  (`gpu.maybe_capture_activation` already exists inside `weight_gemv_residual`.)
- **Inject (`x += α·v` / projective): uniform.** We issue *our own*
  `add_inplace_f32` at the **block boundary**; the fusion folds the
  *projection's* add into the residual, not ours, so **no fused kernel is
  touched**. Hooking at the block boundary makes **MoE fusion irrelevant** — the
  MoE-down writes its output *into* `x`, so after the block `x` is already the
  post-residual stream we read/inject on.
- `o_proj` / `down_proj` / expert-down are discrete addressable weight tensors
  everywhere (editable for the deferred permanent-bake path).

Per-arch evidence (file:line):

| Arch | Post-attn | Post-FFN/MoE | Discrete add? |
|---|---|---|---|
| gemma3 decode `forward_after_x` | `add_f32` `forward.rs:349` | `add_f32` `:359` | **Yes (both)** |
| gemma3 prefill `forward_prefill_batch` | `add_f32` `:526` | `add_f32` `:535` | **Yes (both)** |
| qwen2 prefill | `add_inplace_f32` `qwen2.rs:1451` | `add_inplace_f32` `:1466` | Yes |
| qwen2 decode | fused `Step::GemvResidual` `:1178` | fused `:1229` | No |
| qwen35 decode | fused `weight_gemv_residual` `qwen35.rs:786` | fused MoE `:8256/:8474/:8541` | Mostly no |
| lfm2moe prefill | `add_inplace_f32` `forward.rs:864` | dense `:891` / MoE fused `:1003` | Partial |
| lfm2moe decode | fused `:1262` | fused `:1288/:1411` | No |
| minimax prefill | `add_inplace_f32` `forward.rs:1050` | MoE fused `:1170` | Partial |
| minimax decode | fused `:220` | MoE fused `:381/:395` | No |

Conclusion: **block-boundary tap + inject works on every arch** because the
residual is always an addressable buffer. gemma3 is additionally the cleanest
and *is the MVP target family* — so it is the Phase-1 arch.

### LoRA — train-only, out of scope

LoRA exists **only** as a forward+backward training op in `hipfire-train`
(`ops/lora.rs`, used by `block.rs` and a gradcheck example). There is **no**
inference apply (prefill or decode), **no** load-time merge, **no** adapter
loader/format/serialization bridge. (The `lora` hits in `deepseek4` are the
model's *native* Q/O low-rank attention projections — intrinsic weights, not
loadable adapters.) Heretic's LoRA-adapter export path is therefore a
non-starter without building a whole inference-LoRA subsystem. The MVP avoids
LoRA entirely; our runtime-hook design needs none.

## Design — runtime block-boundary steering hook

A new crate **`hipfire-steer`** (depends on `hipfire-runtime`, `hipfire-kld`,
`hipfire-eval`) plus a minimal hook into one arch forward.

The forward consults an optional spec at each **block boundary** (after the
full transformer block's residual is settled):

```
struct SteerSpec {
    per_layer_dir: Vec<DeviceTensor>,   // normalized direction v_L, one per block
    mode: SteerMode,                    // Steer (x += α·v) | Ablate (x −= λ (vᵀx) v)
    strength: f32,                      // α or λ
    layer_range: Range<usize>,          // which blocks are active
}
```

Two gated call sites per arch forward:
1. **capture** — when deriving directions, copy `x` (last-token position) to a
   host/accumulation buffer after block L.
2. **inject** — when applying, run the mode op on `x` after block L if `L` is in
   `layer_range`.

Granularity for the MVP is **block-boundary only** (uniform, MoE-agnostic).
Per-component (separate attn-out vs mlp-out strength, Heretic-parity) is
deferred — it needs hooks at the projection outputs, which in fused-decode arches
means the discrete fallback path.

## Scope / phases

MVP arch = **gemma3** (then gemma3-vl, which reuses the same hook — the vision
encoder is upstream of the language residual stream and is untouched, exactly as
Heretic taps `language_model.layers`).

- **Phase 1 — Residual hook + capture (gemma3).** Add `SteerSpec`, wire the two
  gated call sites into the gemma3 prefill+decode forward. ~1–2 days.
- **Phase 2 — Direction derivation.** Run +set/−set as 1-token forwards,
  accumulate per-layer residual means (reuse the calibration accumulation
  idiom), `dir_L = normalize(mean_bad − mean_good)`, optional orthogonalize
  against the good direction. ~1 day.
- **Phase 3 — Scoring + driver.** `hipfire-kld` for capability-damage; a
  disclaimer/refusal-marker counter (later: a classifier objective) via
  `hipfire-eval`. Driver: load → derive → apply → score. ~1–2 days.
- **Phase 4 — Search.** Single-shot (mid-late layers, sweep one strength) →
  small grid/random over (layer_range, strength); Pareto reporting à la Heretic.
  No TPE yet. ~1–2 days.

**MVP total ≈ 1 week**: gemma3, runtime-only, both steer + ablate modes, KLD +
refusal-marker scoring, single-shot/grid search.

### Success criterion (the numbers that matter)

On MedGemma, a steer/ablate config that drives the **disclaimer-refusal rate on
a held-out medical −set from high to near-zero** while keeping
**KL-divergence-from-base on a medical +set low** (Heretic treats KLD > 0.5 as
significant capability damage; aim well under that). Validated on a fleet box
with `hipfire lock`.

## Non-goals (MVP)

- No per-component (attn vs mlp) granularity — block-boundary only.
- No TPE/Bayesian optimizer — grid/random search.
- No permanent fp32 weight-bake / standalone artifact.
- No inference-LoRA subsystem.
- No multi-arch — gemma3 (+ gemma3-vl) only.

## Deferred (each independent, additive)

- **Multi-arch**: replicate the two hook call sites per arch (~hours each; MoE
  does not complicate block-boundary hooks).
- **Per-component granularity**: projection-output hooks for Heretic-style
  per-component weights.
- **TPE optimizer**: port/wrap a multi-objective Bayesian sampler — the largest
  remaining piece for true Heretic parity.
- **Permanent weight-bake export**: pre-quant transform in `hipfire-quantize`
  (where LDLQ already operates on fp32) for a zero-overhead standalone artifact
  — a `.heretic`-style sidecar in the canonical naming scheme.
- **Inference-LoRA subsystem**: only if loadable adapters are ever wanted;
  large, and not required by anything above.

## Open questions

- **Direction at last-token only vs. mean over response positions** — Heretic
  uses the last prompt-token residual; confirm that transfers for the
  image+text MedGemma case (the image tokens shift positions).
- **+/− set construction for MedGemma** — source medical prompts it answers vs.
  deflects; size (~400 each, per Heretic defaults) and whether to include the
  image modality in the contrast or text-only.
- **Steer vs. ablate as the MedGemma default** — ablation is more surgical
  (removes the mode), steering is tunable but may need per-prompt strength.

## References

- Reference impl: `third_party/heretic/src/heretic/{model,evaluator,main}.py`
- Capture machinery: `crates/hipfire-runtime/src/calibration.rs`
  (`CalibCollector`, `collect`, `maybe_capture_activation`)
- Residual idioms: `crates/hipfire-runtime/src/weights.rs:1102`
  (`weight_gemv_residual`); gemma3 `crates/hipfire-arch-gemma3/src/forward.rs`
- Train-only LoRA: `crates/hipfire-train/src/ops/lora.rs`,
  `crates/hipfire-train/src/block.rs`
- Scoring: `crates/hipfire-kld/`, `crates/hipfire-eval/`
