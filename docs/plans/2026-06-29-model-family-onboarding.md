# Model-family onboarding — make adding a family a fixed, mechanical checklist

Status: design / not started. Owner: arch-seam effort. Validation box: gfx1151
(re-quantize + serve one existing family end-to-end as the regression anchor; no
new family required to land this refactor).

## Goal

Reduce "add a new model family" from *"edit ~83 dispatch sites across 11 files +
a 42-field struct + three parallel if-ladders"* to a **fixed, small checklist**:

1. New `crates/hipfire-arch-<family>/` implementing the serving trait(s).
2. **One** registry entry (detected family → constructor + ingest rules).
3. **One** `docs/model-support.toml` row.

No edits to `LoadedModel`, no new scattered `arch_id` literals, no new
`generate_*` function, no new `if arch_id == N` branch in the quantizer.

This plan is the **cross-cutting onboarding spine**. It deliberately does NOT
re-plan the serving seam — that is owned by two in-flight docs (see
*Relationship to existing plans*). It adds the pillars those docs do not cover:
**arch identity + a registry**, the **quantizer ingest side**, and the
**onboarding contract** that ties every touch-point together.

## Current reality (measured 2026-06-29)

| Friction | Evidence |
|---|---|
| `LoadedModel` Option-soup | ~42 `Option`/registry/backend/config fields in `crates/hipfire-serving-core/src/model.rs`; every `load.rs` arch branch must initialize **all** of them (the ~50-line `None, None, …` block per family). |
| `arch_id` dispatch sprawl | **83** `arch_id ==` sites across 11 files: `load.rs` (19), `session.rs` (17), `daemon/main.rs` (17), `quantize/main.rs` (11), `generate.rs` (9), `qwen35_decode.rs` (4), examples, etc. |
| `arch_id` is a hand-assigned magic int | Next free is **17** (0–16 taken); allocated by hand in `quantize/main.rs:6264`; literals were centralized to named constants (step 1, `9fca940ae`) but the **value is still hand-maintained and redundant** — the HFQ metadata already names the family. Known collision class: `arch_id 0` (llama) vs the HFQM non-weight sentinel `0`. |
| Quantizer ingest is a parallel if-ladder | `architectures` string → `arch_id` map, `should_quantize` rules, 3D-MoE expert-split detection, and norm-baking all live as central branches in `hipfire-quantize/src/main.rs` (11 `arch_id` sites). No arch-local ownership. |
| No single onboarding contract | A family touches the quantizer, an arch crate, serving registration, `model-support.toml`, and calibration — with nothing tying them together or failing CI when one is missed. |

The serving-side abstractions (`SimpleAr`, `ServingBackend`,
`SessionServingBackend`, `ArchCaps`, `GenerateCtx`) are **already defined** and
designed to "replace the per-arch `generate_*` dispatch and the `LoadedModel`
Option-soup" (their own docstrings). The friction is **incomplete adoption +
no arch-identity/ingest layer**, not missing design.

## Relationship to existing plans (build on, do not duplicate)

- **`2026-06-23-seam-finish-and-mamba2.md`** — migrates *every* arch onto
  `ServingBackend`/`SimpleAr`, generalizes `State` into the per-layer `Mixer`
  model (`hipfire-mixer`), deletes the *simple-tier* Option-soup, deletes the
  `generate_*` ladder. **This is the serving half of "smooth onboarding."** This
  plan treats its completion as the foundation for Pillar A's serving registry
  and does not re-specify it.
- **`2026-06-29-session-serving-backend.md`** — hoists the *rich-tier* session
  protocol (qwen35 + lfm2) onto `SessionServingBackend`, introduces
  `SessionRegistry<S>`, deletes the rich-tier duplication + Option-soup.
- **`2026-06-29-concurrent-session-execution.md`** — per-session-slot restructure
  (C1) that moves session state into per-arch backends.

**What none of them own — and this plan adds:**
- A) Arch **identity** (metadata-derived, not a hand-numbered int) + a **registry**
  that the loader, quantizer, and matrix all read.
- B) An arch-local **ingest descriptor** so the quantizer stops branching on
  `arch_id`.
- C) The **onboarding contract**: a CI-checkable "you added a family iff you
  touched exactly these things" gate, plus crate scaffolding.

## Pillar A — arch identity + a registry

**Problem.** `arch_id` is simultaneously (1) a serialized HFQ header byte,
(2) a hand-assigned allocation, and (3) the dispatch key smeared across 83 sites.

**Target.** Two layers:

1. **Identity from metadata.** The HFQ metadata already carries the family
   (`architectures` / `model_type`). Resolve the family **by name** at load and
   quantize time; keep the numeric `arch_id` only as a **stable serialized tag**
   derived from the family, never hand-typed at a call site. This removes the
   magic-int allocation step and the `0`-vs-sentinel collision class
   (closes the open half of the `arch_id-centralization` work).

2. **One registry.** A single table:

   ```rust
   pub struct ArchRegistration {
       pub family: ArchFamily,            // canonical id + name; serialized tag
       pub detect: fn(&Metadata) -> bool, // family match from HFQ/HF metadata
       pub load:   fn(LoadArgs) -> Result<Box<dyn ServingBackend>, String>,
       pub ingest: &'static dyn ArchIngest, // Pillar B
       pub features: ArchFeatures,        // from model-support.toml (Pillar D)
   }
   ```

   The loader does **one** lookup → `Box<dyn ServingBackend>` (+ optional
   `Box<dyn SessionServingBackend>` downcast for the rich tier). The ~83
   `if arch_id == N` sites collapse into trait-method calls (`caps()`,
   `eos_token()`, `serve()`) plus registry lookups. The quantizer reads the same
   table for its ingest path (Pillar B) and the matrix reads it for features
   (Pillar D) — **one source of truth, three consumers.**

**Dependency.** The `load: fn(...) -> Box<dyn ServingBackend>` form requires the
seam-finish migration (every arch served through `ServingBackend`) to be far
enough along that the loader can return a boxed backend instead of populating
typed `LoadedModel` Options. Pillar A's registry **lands incrementally behind
that**: register families as they cross onto the seam; the registry is empty-safe
and falls back to the existing ladder for not-yet-migrated archs.

## Pillar B — quantizer ingest descriptor

**Problem.** Adding a family means editing central if-ladders in
`hipfire-quantize/src/main.rs`: the `architectures → arch_id` map, `should_quantize`
(which tensors stay f16/bf16), 3D-MoE expert-split detection, and norm-baking
(e.g. Gemma's `(1+w)` offset).

**Target.** An arch-local trait the registry carries:

```rust
pub trait ArchIngest: Sync {
    /// Which tensors stay full-precision vs. get quantized, by name + shape.
    fn quant_policy(&self, name: &str, shape: &[usize]) -> QuantPolicy;
    /// Expert/3D-split + tensor-name remapping (MoE gate_up/down, tied lm_head…).
    fn tensor_plan(&self, meta: &Metadata) -> TensorPlan;
    /// Ingest-time transforms (norm-offset bake, q-prescale bake, …).
    fn transforms(&self, cfg: &Value) -> Vec<IngestTransform>;
}
```

The quantizer's `main.rs` ingest body becomes generic over `&dyn ArchIngest`
fetched from the registry; the family-specific knowledge moves **next to the arch
crate** (or into `hipfire-arch-<family>/src/ingest.rs`). New family = implement
`ArchIngest`, register it — zero edits to `quantize/main.rs`.

**Note.** Keep the `--arch-id` override as an escape hatch during bring-up, but it
becomes `--family <name>` once identity is name-based.

## Pillar C — onboarding contract + scaffolding

**Problem.** Nothing enforces that a new family touched every required surface; an
incomplete family compiles and silently misbehaves (cf. the gemma3 grounding bug,
the nemotron quantizer bugs).

**Target.**

1. **A `cargo xtask new-arch <family>`** (or a `.agents/` skill) that scaffolds
   the 5-file crate from the gemma3 template (config/weights/forward/arch/ingest)
   with TODO markers, adds the workspace member, and stubs the registry entry +
   matrix row.
2. **A CI completeness gate** (extend `no-gpu-ci.sh`): for every registered
   family assert it has (a) a registry entry, (b) a `model-support.toml` row,
   (c) an `ArchIngest`, (d) a `ServingBackend` impl. A family present in one but
   missing from another **fails CI** — the same drift-gate pattern the capability
   matrix already uses successfully.
3. **A `docs/MODEL-FAMILY-CHECKLIST.md`** generated from the registry, so the
   "how to add a family" contract is documentation that cannot go stale.

## Pillar D — declarative overrides (extend the matrix pattern)

The `model-support.toml` → generated table → `--check` drift gate is the *good*
model and already smooth. Extend the same declarative-source pattern to the
remaining hand-coded per-arch knobs that are currently trait-override functions:
`eos_filter_overrides`, `sampler_overrides`, `prompt_frame_overrides`,
`loop_guard_overrides`. Where an override is pure data (blocked tokens, raw-vs-
ChatML, eos id, repeat penalty), move it into the registry/TOML so a new family
declares it instead of writing four override fns. Keep the trait override for
genuinely code-shaped behavior.

## Phases (additive; each compiles + passes `no-gpu-ci.sh`; strangler-fig)

Ordered so each phase stands alone and the registry can coexist with the legacy
ladder until the last delete.

- **O0 — identity by name.** Add `ArchFamily` (name ↔ stable tag) and resolve
  family from metadata at load + quantize. Keep the numeric ladder working;
  derive the int from the family instead of hand-typing it. Gate: build +
  `no-gpu-ci.sh`; re-quantize one existing family and byte-diff the HFQ header.
- **O1 — empty-safe registry.** Introduce `ArchRegistration` + a global registry;
  the loader consults it first and **falls back to the existing ladder** on miss.
  No behavior change yet. Register the already-on-seam archs (qwen2, gemma3,
  gemma3-vl) as the first entries.
- **O2 — ingest descriptor.** Define `ArchIngest`; port the gemma3 + qwen2 ingest
  rules into arch-local impls; make `quantize/main.rs` generic over the registry
  for those families (others stay on the ladder). Gate: re-quantize gemma3 + qwen2
  and diff tensors/coherence vs pre-refactor artifacts.
- **O3 — migrate-with-the-seam.** As `seam-finish` moves each arch onto
  `ServingBackend`, register it (load + ingest + features) and **delete its
  `arch_id` branches** from `load.rs`/`generate.rs`/`session.rs`/`quantize`. This
  phase is paced by seam-finish; it is bookkeeping on top of that work, not new
  serving code.
- **O4 — completeness gate + scaffolding.** Land the `new-arch` scaffolder, the
  CI completeness gate, and the generated checklist (Pillar C). Land the
  declarative overrides (Pillar D).
- **O5 — delete the ladder.** Once every family is registered, remove the residual
  `if arch_id == N` dispatch and the `--arch-id` int override. The Option-soup is
  already gone via seam-finish P6 + session-serving S5; this removes the *dispatch*
  remnant. Final gate: full re-quantize + serve sweep of all families on gfx1151.

## Definition of done

Adding a dense text family is exactly:
1. `cargo xtask new-arch <family>` → fill `config.rs`/`weights.rs`/`forward.rs`.
2. Implement `SimpleAr` + `ServingBackend` (+ `ArchIngest`).
3. One registry line + one `model-support.toml` row.

The CI completeness gate refuses a half-wired family; no other file changes.

## Risks / open questions

- **Sequencing vs seam-finish.** Pillar A's `load → Box<dyn ServingBackend>` is
  gated on seam adoption. Mitigated by the empty-safe registry (O1) + per-arch
  fallback (O3), so this plan never blocks on a big-bang serving cutover.
- **Serialized `arch_id` stability.** Existing `.hfq` artifacts carry numeric
  `arch_id`; O0 must keep the family→int map back-compatible for already-shipped
  files (read int → family for old artifacts; write family-derived int for new).
- **Rich-tier double-dispatch.** Registry returns `ServingBackend`; the rich tier
  also needs `SessionServingBackend`. Resolve via a `caps()`-gated downcast, not a
  second registry — keep one table.
- **Quant policy expressiveness.** `should_quantize` has accreted shape-edge cases
  (e.g. the hidden=3136 non-divisibility fallback). `ArchIngest::quant_policy`
  must expose shape, not just name, so those cases stay arch-local rather than
  leaking back into a central branch.

## Verification

Per phase: workspace build + `tests/no-gpu-ci.sh` (incl. the new completeness
`--check`). On gfx1151 under `hipfire lock`: re-quantize one dense family (gemma3)
+ one MoE family (qwen35) through the registry path and run
`tests/coherence-gate-dflash.sh`, diffing argmax/KLD against pre-refactor
artifacts. No new family is needed to validate the refactor — an existing family
re-onboarded through the new path is the regression anchor.

## Out of scope

- The serving-seam migration itself (`seam-finish`), the `Mixer` state model, and
  the rich-tier session hoist (`session-serving-backend`) — consumed as
  foundations, not re-planned here.
- Any specific new family (e.g. Gemma4) — this plan is the *infrastructure* that
  makes that family cheap; bring-up of a concrete family is its own doc.
