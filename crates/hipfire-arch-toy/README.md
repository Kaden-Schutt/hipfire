# hipfire-arch-toy

The reference template for adding a model architecture to hipfire. **It is
deliberately unshippable**: `arch_id = 0xFF`, no carrier claims it, and
`load_toy_bundle` always returns `Err`. Nothing can ever dispatch it — that
is the point. Copy the directory, claim a real id, fill in the bodies.

## What the crate shows

A faithful skeleton of the three things every shippable arch crate has:

| File | Mirrors | Contract |
|---|---|---|
| `src/arch_model.rs` | `hipfire-arch-minimax/src/arch_model.rs` | `ToyBundle` + `impl hipfire_runtime::arch_model::ArchModel` (6 required methods: `dim`, `n_layers`, `vocab_size`, `arch_key`, `kv_cache_mut`, `free_gpu`; `reset_session_state` has a default) |
| `src/carrier.rs` | `hipfire-arch-minimax/src/carrier.rs` | `pub fn load_toy_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<ToyBundle, String>` — honestly stubbed |
| `src/arch.rs` | `hipfire-arch-cohere2moe/src/arch.rs` | `impl hipfire_runtime::arch::Architecture` — intra-crate typed bring-up helper (no consumers outside arch crates today) |

The layer rule that shapes this: **arch crates must not depend on
`hipfire-loader`** (cycle). So the `Carrier` impl — which returns the
loader's `LoadedModel` — lives in `crates/hipfire-loader/src/carriers.rs`,
and it calls *into* your crate's `load_<arch>_bundle` for all model work.

## Checklist: adding a new architecture

Verified against the tree on branch `arch/saddle` (2026-08-15). Every path
below was opened and confirmed. Sites marked ⚙️ are **compile-enforced** —
skip one and the workspace stops building (exhaustive `match` on
`ModelState`, a missing `GenerationRoute` arm, etc.).

### Tier A — the arch crate itself

1. `crates/hipfire-arch-<name>/` — copy this crate. Replace the stub types in
   `src/toy_model.rs` (config parsed from HFQ `metadata_json` /
   safetensors `config.json`; GPU-resident weights via
   `hipfire_runtime::llama::WeightTensor`; per-decode state sized by config),
   fill in `load_<name>_bundle`, and add your forward as free functions
   (`src/forward.rs`) — forward is statically dispatched, never `dyn`.

### Tier B — loader (all ⚙️)

2. ⚙️ `crates/hipfire-loader/Cargo.toml` — add the arch crate dependency
   (plain, non-optional, matching the five newest arches).
3. ⚙️ `crates/hipfire-loader/src/lib.rs` —
   `ModelState` variant; arms in `as_arch_model`, `as_arch_model_mut`,
   the `free_gpu` match, and the pp>1 `unload_model` match (no wildcards —
   that is the leak guard); `&NewCarrier` in `REGISTRY`; optional
   `LoadedModel` accessors (`minimax()`/`cohere2moe()` pattern).
4. ⚙️ `crates/hipfire-loader/src/carriers.rs` — `pub struct <Name>Carrier`
   implementing `Carrier`: `name`, `claims_arch_id` (exact id match —
   never an open range), `load` (calls your `load_<name>_bundle`, then
   `resolve_source_meta` + `build_speculator` + `LoadedModel::skeleton`),
   `caps`, `sampling_defaults`; `spec_target_guard` / `make_spec_emitter`
   when the arch joins the n-gram spec path; `arch_default_template` arm
   if you ship a built-in chat template. Registry tests at the bottom of
   `lib.rs` (`carriers_are_disjoint`, `known_ids_route_as_expected`) pin
   your claim.

### Tier C — generation (all ⚙️)

5. ⚙️ `crates/hipfire-generate/Cargo.toml` — add the arch crate dependency.
6. ⚙️ `crates/hipfire-generate/src/ar.rs` — `GenerationRoute` variants +
   `ALL` + `name()`; the `match i.arch_id` arm in
   `select_generation_route`; the dispatch arm in `generate` that calls
   your body.
7. ⚙️ `crates/hipfire-generate/src/dense.rs` — `generate_<name>`
   (prefill loop + decode loop + JSONL events; `generate_lfm2moe` /
   `generate_cohere2moe` are the current small references).
8. `crates/hipfire-generate/src/common.rs` — session-reset arm for your
   `ModelState` variant (drop recurrent state, reset `compact_offset`).
   Not compile-enforced: the `if let Some(ModelState::…)` chains just never
   fire for your variant, so stale state survives resets until you add one.

### Tier D — routing into the crate

9. `crates/hipfire-runtime/src/arch_mapping.rs` — `MODEL_TYPE_TO_ARCH_ID`
   row(s) for your HF `model_type` string(s). **Sole** source of truth for
   all `model_type`/`general.architecture` → `arch_id` routing — the
   safetensors-dir path (`derive_arch_id`) and both quantize pipelines now
   derive from this table, so a new arch needs no edit elsewhere for
   `arch_id` routing.
10. `crates/hipfire-runtime/Cargo.toml` — `arch-<name>` feature in the
    default list + the crate as a `[dev-dependencies]` entry (the dev-dep
    cycle-exclusion trick; see the comment block there).
11. `crates/hipfire-quantize/src/pipeline.rs` — per-arch ingest flags
    (`is_<name> = arch_id == <N>`) when your tensors need special quant
    routing (MoE experts, tied-embed guards). Dense plain-vanilla arches
    may need nothing.
    (`pipeline_gguf.rs` needs no edit — it calls the same
    `lookup_model_type`.)
12. `crates/hipfire-daemon/src/main.rs` — session-reset arm(s) beside the
    `ModelState::Gemma4`/`MuseGlimmer` blocks (only if your state isn't
    fully covered by `ArchModel::reset_session_state`).

### Tier E — optional capability surfaces

13. `crates/hipfire-generate/src/batch.rs` — continuous-batch admission, if
    your arch supports it.
14. `crates/hipfire-generate/src/redline.rs` — bench-fixture routes, if you
    want Redline capture.
15. `crates/hipfire-runtime/src/reset_core.rs` — retry-eligibility
    inventory row (your `arch_key()` string must match).
16. `crates/hipfire-cli/Cargo.toml` — only if the CLI itself needs your
    types (precedent: qwen35 multi-slot).

### Tier F — registry & docs

17. `docs/architecture-ids.md` — claim your id. Never reuse 2–4 (deliberately
    unassigned) or 0xFF (this template).
18. `docs/ARCHITECTURE.md` — carrier table row + crate table row.
19. `registry/models.json` — model entries for your artifacts, then
    `scripts/registry_gen.py` regenerates `registry/v1.json` (which is where
    the `arch_id` column lands; never hand-edit `v1.json`).
20. `docs/MODELS.md` — catalog rows.
21. `docs/env-vars.md` — any new `HIPFIRE_*` knobs and feature-flag notes.
22. `CLAUDE.md` — the arch-id list in the crate summary.

Also run `scripts/check-crate-maps.py <name>` to seed your crate's own
`map.md` (in-crate, so not counted above).

## The count

**21 out-of-crate files** a new architecture must touch today (checklist
items 2–22 above), down from 23 before this change (and ~28 pre-programme)
— despite *more* arches, because the loader's `LoadedModel` per-arch
`Option<…>` fields, the daemon's per-arch `arch_id` match ladders, and the
bespoke spec-decode wiring were folded into `ModelState`/`ArchModel`/`Carrier`,
and `arch_id` routing is now a single site (`arch_mapping.rs`): `safetensors_source.rs`
is table-driven (longest-key substring match) and `tests/arch_id_unification.rs`
iterates `MODEL_TYPE_TO_ARCH_ID` directly, so neither needs a per-arch edit.
Six are compile-enforced (all of Tier B plus Tier C items 5–7: the
`ModelState` and `GenerationRoute` matches are exhaustive, so a new variant
without its arms does not build); the rest fail closed or silently skip.

## What this crate is not

- Not a real model. `load_toy_bundle` never returns `Ok`. There is no
  forward pass.
- Not registered. Do not add a `ToyCarrier` to the loader; do not add 0xFF
  to `arch_mapping.rs`. `docs/architecture-ids.md` lists 0xFF as reserved
  precisely so nobody ships it.
- Not consumed by the daemon, generate, or any binary. The workspace builds
  it only to keep the template from rotting.

## Production references

- Smallest current load body: `crates/hipfire-arch-minimax/src/carrier.rs`.
- Smallest `ArchModel` impl: `crates/hipfire-arch-qwen2/src/arch_model.rs`.
- Full bar (hybrid attention, MoE, paging, spec decode):
  `crates/hipfire-arch-qwen35/`.
