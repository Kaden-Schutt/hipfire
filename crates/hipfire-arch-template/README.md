# hipfire-arch-template

Copy-paste **template** for adding a new model family. Not a model — a starting
point that already compiles against every seam a real family plugs into.

A family has **two crates**:

| crate | deps | holds | linked by |
|---|---|---|---|
| `hipfire-arch-template` (this) | runtime + kernels | serving: `Architecture` impl (config/weights/state/forward) + serving capabilities (`ToyModel`, `BatchedPrefill`, …) | the daemon |
| `hipfire-arch-template-spec` | **only** `hipfire-arch-api` | offline: the `Ingest` quant-policy | the **quantizer** |

Both `register_arch!` on the **same `ArchId`**; the registry **merges** them, so the
daemon sees the full capability set and the (CPU-only) quantizer links just the lean
spec. See the `bundle_merges_template_spec_and_serving` test in `hipfire-archs`.

## What this crate is / is not

- The smallest impl that compiles and demonstrates every required method + the
  capability wiring. Each method is a one-liner with a doc-comment.
- **Not** a real model (`config_from_hfq` returns constants, `load_weights` returns
  zeros, no forward pass) and **not** consumed at runtime — the workspace builds it
  only to keep the template from rotting.
- **Not** a home for shared scaffolding — if code helps more than one arch, it
  belongs in `hipfire-runtime` (serving) or `hipfire-arch-api` (policy), not here.

## Add-a-family checklist

**1. Serving crate** — copy `crates/hipfire-arch-template/` → `crates/hipfire-arch-<family>/`:
   - `Cargo.toml`: set `name`/`description`; add the crate to the workspace `members`.
   - rename `Template`/`TemplateConfig`/`TemplateWeights`/`TemplateState` → your
     family; rename `src/template_model.rs`.
   - `src/arch.rs`: set `arch_id()` + `name()`; fill `config_from_hfq` / `load_weights`
     / `new_state` with real logic; implement the forward pass as **free functions**
     in your model module (the trait deliberately doesn't route forward through dyn
     dispatch — see `hipfire-arch-qwen35/src/arch.rs`).
   - `src/caps.rs`: `impl Arch` (id + family); `impl` the **serving** capabilities you
     genuinely support (`ToyModel` for a CI fixture; add `BatchedPrefill` etc. as you
     wire them); `register_arch!(INSTANCE, …)` listing exactly those.

**2. Offline spec crate** — copy `crates/hipfire-arch-template-spec/` → `crates/hipfire-arch-<family>-spec/`:
   - `Cargo.toml`: set `name`; add to workspace `members`.
   - rename `TemplateSpec` → `<Family>Spec`; set the **same** `ArchId` and family name.
   - `impl Ingest`: usually just delegate to `transformer_role` / `default_importance`
     / `default_requires`. Override `importance`/`requires` only for genuinely special
     tensors (e.g. an MLA compressor — see `hipfire-arch-deepseek4-spec`). Never name a
     format or a codec here.
   - `register_arch!(<FAMILY>_SPEC, Ingest);`
   - add the crate to **`hipfire-arch-specs`** (one `dependencies` line + one
     `use … as _;`) so the quantizer links it. **No quantizer edit is needed.**

**3. Wire it up:**
   - if daemon-served, add the serving crate to **`hipfire-archs`** (`dependencies` +
     a `use … as _;` force-link line).
   - add the `arch_str → arch_id` arm in the quantizer's detection match
     (`crates/hipfire-quantize/src/main.rs`) — the id must match your `-spec`'s `ArchId`.
   - bump the migration **ledger** in `hipfire-archs` (`registry_integrity_and_migration_ledger`).
   - claim the `arch_id` in `docs/architecture-ids.md` (when it exists) or via PR review.

**4. Validate:** `./tests/no-gpu-ci.sh` (add your `-spec`/serving crates to the test
list), then `tests/coherence-gate.sh` + `tests/speed-gate.sh` for the real forward path.

Rough effort for a real port: bring-up triple (config/weights/state) a few hundred
lines; forward pass a couple thousand for a dense LLaMA-style model, more for hybrid
attention or MoE; new ops go in `kernels/src/*.hip` + `crates/hipfire-rdna`, **not**
your arch crate (see `.github/CONTRIBUTING.md` "Crate topology").

## Production reference

Read `crates/hipfire-arch-qwen35/` for a complete implementation with hybrid
DeltaNet attention, MoE routing, weight paging, speculative decoding, and long-context
prefill compression. That's the bar.

[`Architecture`]: ../../crates/hipfire-runtime/src/arch.rs
