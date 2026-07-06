# Unified qwen35 weight loading via `WeightBackend`

**Date:** 2026-06-11
**Branch:** `feature/paro-transparent-loading`
**Status:** SUPERSEDED (2026-06-11). This design was written against a stale tree.
The feature branch already implements a more general version (stateful
`WeightBackend` with `candidates` resolver + `norm_bias` + Paro augmentor chain,
covering qwen2/llama) and layers a **Carrier registry** on top — see
`2026-06-11-carrier-registry-unified-design.md` (authoritative) and the reconciled
`2026-06-11-qwen35-unified-weight-loading-impl.md`. Kept as a design-rationale record.

## Problem

`crates/hipfire-arch-qwen35/src/qwen35.rs` has **three** weight-load entry
points, each with its own near-duplicate body:

| Entry point | Source type | MoE loader | Consumer |
|-------------|-------------|------------|----------|
| `load_weights` (single-GPU) | `HfqFile` | inline | HFQ daemon path + ~30 examples |
| `load_weights_multi` → `load_layer_into` | `HfqFile` | `load_moe_ffn` | multi-GPU HFQ |
| `load_weights_paroquant` | `dyn ModelSource` | `paro_load_moe_ffn` | daemon safetensors/PARO |

`paro_load_moe_ffn` and `load_moe_ffn` are explicit near-duplicates (the paro
one comments *"mirrors load_moe_ffn"*). The `match (LayerType, is_moe)` layer
walk is written three times. `ModelSource` already unifies **data access**
(tensor bytes) but not **interpretation** (quant decode, name mapping, the
ParoQuant AWQ→HFQ4G128 repack + Givens-rotation sidecar), so the loaders still
fork.

This is not dead code — every path is live (the daemon uses
`load_weights_paroquant` at `daemon.rs:1969` and `load_weights` for HFQ). The
loose end is the triplicated structure.

## Goal

One backend abstraction so all three qwen35 loaders share a single layer-walk
and MoE assembly. Format-specific logic (HFQ quant decode vs ParoQuant repack)
moves behind a `WeightBackend` trait with two impls: `HfqBackend`,
`ParoBackend`.

## Non-goals

- **Llama family** (`hfq.rs:load_weights_paroquant_llama`) — deferred to a later
  cut. The trait is designed so llama can adopt it later, but this cut does not
  touch llama.
- No change to forward-pass, kernels, quant formats, or the `MoeFfnWeights` /
  `LayerWeights` struct shapes.
- No new quant format support.

## Architecture

### `WeightBackend` trait (new, in `hipfire-runtime`)

Lives next to `ModelSource` in `crates/hipfire-runtime/src/`. Returns only
runtime-level types (`WeightTensor`, `GpuTensor` from `rdna-compute`), so it sits
upstream of qwen35 (which owns `MoeFfnWeights`).

```rust
/// Routed-expert on-disk packing (the MoE inner-loop fork).
pub enum MoeExpertLayout { Fused, ParoRepack }

/// Coarse backend identity for the embed/output forks that can't be expressed
/// through single-tensor primitives (they produce qwen35 types).
pub enum BackendKind { Hfq, Paro }

pub trait WeightBackend {
    /// Load a single matmul weight. `leaf` is a BARE logical name with NO
    /// `.weight` suffix (e.g. "self_attn.q_proj", "mlp.experts.3.gate_up_proj").
    /// BOTH backends prepend `model.language_model.{p}.` internally — the seam
    /// difference is ONLY the suffix:
    ///   HFQ  -> "model.language_model.{p}.{leaf}.weight"   (quant_type byte)
    ///   Paro -> "model.language_model.{p}.{leaf}.qweight"  (+ .qzeros/.scales)
    ///        -> "model.language_model.{p}.{leaf}.weight"   (FP16 fallback)
    /// Backend owns quant interpretation, FP16 fallback, rotation-sidecar attach.
    fn load_wt(&self, gpu: &mut Gpu, p: &str, leaf: &str, m: usize, k: usize)
        -> HipResult<WeightTensor>;

    /// RMSNorm weight — applies the GemmaRMSNorm `+= 1.0` bake. `p` MAY be ""
    /// for the top-level `norm.weight` (load_output): the impl must join `p` and
    /// `leaf` skipping an empty `p` (→ `model.language_model.norm.weight`, not a
    /// double-dot).
    fn load_norm(&self, gpu: &mut Gpu, p: &str, leaf: &str, shape: &[usize])
        -> HipResult<GpuTensor>;

    /// Raw f32 tensor — NO `+= 1.0` bake. HfqBackend decodes arbitrary quant
    /// types (Q8_0/MQ8/FWHT via `load_any_as_f32`); ParoBackend only F16/F32.
    fn load_f32(&self, gpu: &mut Gpu, p: &str, leaf: &str, n: usize)
        -> HipResult<GpuTensor>;

    fn moe_expert_layout(&self) -> MoeExpertLayout;
    fn kind(&self) -> BackendKind;

    /// Raw tensor bytes for Paro CPU repack. Paro-only: HfqBackend's pread path
    /// returns owned data and `Fused` layout never calls this — HfqBackend may
    /// `unimplemented!()`. `Cow` allows the owned-Vec case. (Three concurrent
    /// SHARED `&self` borrows in `paro_repack_moe_projection` are fine.)
    fn raw_bytes(&self, name: &str) -> Option<std::borrow::Cow<'_, [u8]>>;

    /// Post-layer page-management hook. HfqBackend calls
    /// `drop_pages_range(layer_data_range(p))` (UMA APU page-cache control,
    /// `qwen35.rs:2013`); ParoBackend is a no-op. Called by BOTH the single- and
    /// multi-GPU generic drivers after each layer. NOT observable in the
    /// byte-identical gate — it's a memory/perf behavior, so it must be on the
    /// trait, not skipped.
    fn after_layer(&self, p: &str) {}

    fn quant_config(&self) -> Option<&QuantConfig>;  // None for HfqBackend
}
```

`drop_mmap()` (the one `&mut HfqFile` use, `qwen35.rs:1814`) happens in the HFQ
thin wrapper *before* `HfqBackend::new(&hfq)` is constructed (the backend borrows
`&HfqFile`), so it stays out of the generic driver.

**Name mapping (corrected after review).** Contrary to an earlier draft, HFQ is
NOT prefix-free: `load_weight_tensor` does `format!("model.language_model.{name}")`
(`qwen35.rs:989`) and `paro_load_wt` does the same (`qwen35.rs:2030`). Both
prepend the prefix; the only divergence is the `.weight` suffix (HFQ call sites
bake it into the name; Paro appends `.qweight`/`.weight` internally). The unified
`load_layer<B>` therefore passes **bare leaves** and each backend appends its own
suffix.

**`load_norm` vs `load_f32` is a correctness fork, not cosmetic.** `load_norm`
bakes `+= 1.0`; `load_f32` must not. Tensor→method mapping the walk MUST use:
- `load_norm` (bake): `input_layernorm`, `post_attention_layernorm`,
  `self_attn.q_norm`, `self_attn.k_norm`, top-level `norm.weight`.
- `load_f32` (no bake): `linear_attn.A_log`, `linear_attn.dt_bias`,
  `linear_attn.conv1d.weight`, **and `linear_attn.norm.weight`** — a "norm" by
  name that is loaded RAW today (`load_any_as_f32`/`paro_load_f32`,
  `qwen35.rs:1945/2121`). Routing it through `load_norm` would wrongly +1.0 a
  DeltaNet head-norm.

**Router / shared-gate are "identical-by-fallback."** The unified MoE loader
uses `load_wt` for the router and `shared_expert_gate`. Paro's `load_wt` FP16
fallback reproduces today's `load_fp16_weight_from_source` exactly (same fn, same
`[m,k]` shape); HFQ's `load_wt` keeps its quant-aware + AWQ-sidecar path. The
loader must NEVER call `load_fp16_weight_from_source` directly.

### Backend impls (in `hipfire-runtime`)

- **`HfqBackend<'a>`** wraps `&'a HfqFile`. Absorbs today's `load_weight_tensor`,
  `load_norm_weight`, `load_raw_f32`, `load_any_as_f32`. `moe_expert_layout() =
  Fused`.
- **`ParoBackend<'a>`** wraps `&'a dyn ModelSource` (+ cached `gs`/`kr` from
  `quant_config`). Absorbs today's `paro_load_wt`, `paro_load_norm`,
  `paro_load_f32`, `load_fp16_weight_from_source`. `moe_expert_layout() =
  ParoRepack`.

These free functions currently live in qwen35.rs; they move into the backend
impls. They reference only `Gpu`/`GpuTensor`/`WeightTensor`/`HfqFile`/
`ModelSource` — all available in `hipfire-runtime`.

### Unified qwen35 code (generic over `&dyn WeightBackend`)

Written **once**:

- `load_layer<B>(backend, gpu, config, layer_idx, p) -> HipResult<LayerWeights>`
  — the single `match (LayerType, is_moe)` walk. Replaces both the inline body
  in `load_weights` and `load_layer_into`.
- `load_moe_ffn<B>(backend, gpu, p, config, layer_idx) -> HipResult<MoeFfnWeights>`
  — router, shared-expert (gate/up/down), shared-expert-gate, device pointer
  tables, and `MoeFfnWeights` assembly. Replaces `load_moe_ffn` +
  `paro_load_moe_ffn`.
- `load_weights_generic(backend: &dyn WeightBackend, config, gpu)` — single-GPU
  driver over `load_layer`.
- `load_weights_multi(backend, config, gpus)` — **HFQ-only** (see fork 3),
  drives the same `load_layer`, selecting the device per layer (current
  band-routing + `drop_pages_range` preserved).

### Residual format forks (3 — all kept in qwen35, none on the runtime trait)

The trait unifies single-tensor loading and the whole layer walk. Three pieces
genuinely diverge because they produce qwen35 types or use HFQ-only mechanics;
each stays in qwen35 behind a `match backend.kind()` / `moe_expert_layout()`:

1. **Routed-expert inner loop** (`load_moe_ffn<B>`, `match moe_expert_layout()`):
   - `Fused`: per expert one `load_wt(p,"mlp.experts.{x}.gate_up_proj",2*mi,dim)`
     + `load_wt(…"down_proj"…)`. `paro_shared = None`.
   - `ParoRepack`: upload per-projection-group shared sidecars once
     (`paro_load_moe_shared_sidecars`), per-expert CPU-repack
     (`paro_repack_moe_projection`), byte-concat, upload raw, alias rotation
     (`alias_paro_rotation`). `paro_shared = Some(owning MoeParoSidecars)`.
   - Everything around the loop (router, shared expert, gate, pointer tables,
     `MoeFfnWeights` assembly) is shared.

2. **Embedding + output** (`load_token_embd`/`load_output`, `match kind()`):
   HFQ runs `load_weight_tensor_raw` over ~12 quant types + AWQ-sidecar attach
   across three candidate names + tied-vs-separate probe incl.
   `model.language_model.lm_head.weight` (`qwen35.rs:1908-1913, 2271-2320`); Paro
   is F32-only, no AWQ, single-name tied detection (`qwen35.rs:2086-2099`). They
   diverge in assembly logic (tied-detection + AWQ probing), not just single
   tensors. (`EmbeddingFormat` is a runtime type — `llama.rs:540` — so it is NOT
   a placement constraint; keeping these qwen35-local is a complexity choice, not
   forced.) They stay as two `kind()`-selected free fns in qwen35.

3. **Multi-GPU is HFQ-only.** No safetensors multi-GPU path exists today, and
   `drop_pages_range` is an HFQ-mmap concept with no `ModelSource` equivalent.
   `load_weights_multi` keeps requiring `HfqBackend` (or asserts
   `kind() == Hfq`); generic multi-GPU Paro is explicitly out of scope.

A second sub-trait for these would be over-abstraction (decided against) — they
are MoE-layout / model-assembly logic, not pure single-tensor format logic.

## Call-site changes

The `Architecture::load_weights(hfq: &mut HfqFile, …)` trait method
(`runtime/src/arch.rs:134`, impl `qwen35/arch.rs:69`) is the contract for the
daemon's generic HFQ path (`daemon.rs:1705`, shared with Llama/Qwen2) — its
signature **cannot** change. So (correcting the earlier "take the churn" plan):

- **Keep `qwen35::load_weights(hfq: &mut HfqFile, config, gpu)`** as a thin
  wrapper: `hfq.drop_mmap(); load_weights_generic(&HfqBackend::new(hfq), …)`. The
  `Architecture` impl and **all ~30 single-GPU examples are UNCHANGED**.
- **Keep `qwen35::load_weights_multi(hfq: &HfqFile, config, gpus)`** likewise as a
  thin wrapper building `HfqBackend` internally — its 6 callers (`daemon.rs:2084`,
  `pp_parity.rs:78`, `pp_parity_chatml.rs:120`, `pp2_vram_probe.rs:66`,
  `test_qwen35_load_multi.rs:38`, `tests/pp_parity.rs:108`) stay UNCHANGED. The
  generic multi driver lives behind it; multi stays HFQ-only (fork 3).
- **Daemon safetensors path** (`daemon.rs:1969`): replace
  `load_weights_paroquant(source, …)` with
  `load_weights_generic(&ParoBackend::new(&*source), …)`. `load_weights_paroquant`
  is deleted (its body becomes `ParoBackend` + the shared generic).
- `config_from_safetensors` / `config_from_hfq` selection is unchanged.
- No other callers of the deleted fns exist (`load_weights_paroquant`,
  `load_layer_into`, `paro_load_moe_ffn` — verified; VL crate has its own loader,
  llama's `load_weights_paroquant_llama` is out of scope).

## Verification

The forward path is untouched, but loader output must be **byte-identical** to
today (this is a pure refactor, not a numerical fix).

1. **Pre-refactor capture (baseline):**
   - A3B PARO model through `load_weights_paroquant`: capture first-token logits
     (or a hash of all loaded `WeightTensor` device buffers) via an example.
   - An HFQ model through `load_weights`: same capture.
2. **Post-refactor assert:** identical capture from the unified path for both.
3. **`./scripts/coherence-gate.sh`** — mandatory (touches the loader → forward
   input). Must pass for both an HFQ and the A3B PARO model.
4. `cargo build` clean for the workspace; `load_weights_paroquant`,
   `load_layer_into`, `paro_load_moe_ffn` gone, no new `dead_code` warnings.
   `Architecture::load_weights` signature unchanged (compile-checks Blocker 2).

The byte-identical capture (steps 1–2) is the primary net: it catches any
name-mapping miss and any `load_norm`/`load_f32` bake mis-routing immediately.
It does NOT catch: the trait-signature pin (compile failure — step 4);
multi-GPU-Paro (no test — prevented by scoping, fork 3); or the **HFQ page-drop
hook** (`after_layer`/`drop_mmap` — a memory/perf behavior, not an output diff).
For the last, add a manual check: run the HFQ daemon path on a UMA/APU-style
model and confirm RSS does not balloon (per-layer `drop_pages_range` still
fires) — or assert `after_layer` is invoked per layer in a unit test.

## Risks

- **`load_norm` vs `load_f32` mis-routing** (highest): `linear_attn.norm.weight`
  is a norm by name but must use `load_f32` (no `+1.0`). The walk's tensor→method
  table must follow the list in the trait section, not infer from the name.
- **Name-mapping drift**: both backends prepend `model.language_model.{p}.`; the
  seam is the `.weight`/`.qweight` suffix. The unified walk must pass BARE leaves.
  MoE leaf names (`mlp.experts.{x}.gate_up_proj`, `mlp.gate`,
  `mlp.shared_expert.gate_proj`, `mlp.shared_expert_gate`) must reproduce exactly.
  Mitigated by the byte-identical gate.
- **`load_f32` dequant breadth**: HfqBackend's `load_f32` must keep
  `load_any_as_f32`'s full quant-type decode (Q8_0/MQ8/FWHT), not be simplified to
  F16/F32 like ParoBackend's. Document in the impl.
- **ParoBackend sidecar ownership**: shared `MoeParoSidecars` stays owned by
  `MoeFfnWeights.paro_shared` (freed once in `free_moe_ffn`), experts aliasing.
  The `ParoRepack` arm preserves current ownership — nothing moves into the
  backend.
- **HFQ name-probe fallback**: `load_norm_weight`/`load_any_as_f32` try
  `model.language_model.{name}` then bare `{name}` (`qwen35.rs:819, 1397`); Paro
  variants do not. Move this verbatim into `HfqBackend` — do not "unify" the
  lookup.

## Out of scope / follow-ups

- Llama-family adoption of `WeightBackend` (separate cut).
- Multi-GPU for safetensors/Paro (`load_weights_multi` stays HFQ-only; fork 3).
- Moving the MoE-expert / embed / output forks into the backend (would require
  the trait to return qwen35 types or a qwen35-local sub-trait — rejected).
