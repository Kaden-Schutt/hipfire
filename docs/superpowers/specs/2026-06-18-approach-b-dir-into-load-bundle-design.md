# Approach B — Push `Dir` into each arch's `load_bundle`

**Status:** design (brainstorming output — no implementation)
**Date:** 2026-06-18
**Branch:** `feature/transparent-loading-all-models`
**Task:** `t_cd92a643` — "Approach B: push Dir into load_bundle"
**Follows:** `2026-06-18-carrier-arm-unification-design.md` (Approach A, committed
58d76ddd…2acb3acb). Approach A added the `resolve_source_meta` seam + shared
`LoadedModel` tail in `carriers.rs`; the per-carrier `match src` now shrinks to
only the `(config, weights)` (or bundle) block-per-arm.

## Problem

After Approach A, the carrier is *half* source-agnostic: the metadata front
(`resolve_source_meta`) and the `LoadedModel` tail are shared, but the **middle
still branches on source** inside `carriers.rs`. Concretely, the Dir arm of each
core carrier is inlined in the loader:

- `LlamaCarrier` (`carriers.rs:430-477`) inlines the entire Dir bundle:
  `config_from_safetensors_llama` + `load_weights_paroquant_llama` + KV policy
  (`DIR_SAFETENSORS_POLICY`) + `ForwardScratch`.
- `Qwen35Carrier` (`carriers.rs:319-394`) inlines the Dir bundle:
  `config_from_safetensors` + `ParoSource` + `load_weights` + KV policy
  (`QWEN35_PARO_POLICY`) + `DeltaNetState::new` + `Qwen35Scratch::new`.
- `MinimaxCarrier` (`carriers.rs:562-582`) inlines both arms' `(config, weights)`
  match (it has no `load_bundle` at all — the whole carrier is in the loader).

Meanwhile each arch crate *already* has a `load_bundle(src, ctx)` that takes a
`ModelSource` but **rejects `Dir`**:

```rust
// crates/hipfire-arch-llama/src/carrier.rs:14
pub fn load_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<LlamaBundle, String> {
    let ModelSource::Hfq(mut hfq) = src else {
        return Err("llama: directory source unsupported".into());
    };
    ...
}
```

So the Dir knowledge is split: the *signature* says "I take any `ModelSource`",
but the *body* punts Dir back to the carrier. This is information leakage — the
loader has to know each arch's Dir-loading recipe (which config fn, which weight
loader, which KV policy, which scratch) even though that recipe is 100%
arch-private. Adding a source kind (GGUF) today means touching **both**
`carriers.rs` *and* the arch crate.

## Goal

Make `load_bundle` honor its own signature: **the bundle handles `Hfq` and `Dir`
internally**, so the carrier becomes source-agnostic for everything that
produces an arch `Bundle`:

```rust
fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
    // refusals + per-source diagnostics (source-varying, stay in carrier)
    let meta = resolve_source_meta(&src, ctx.path)?;
    let bundle = arch::load_bundle(src, ctx)?;   // ← no `match src` here anymore
    Ok(LoadedModel { state: Some(ModelState::X(bundle)), ..skeleton(meta..) })
}
```

Success criteria:

- The `match src` that produces `(config, weights)`/the bundle leaves
  `carriers.rs` for every carrier in scope and moves into the arch crate.
- Dir-specific loader imports disappear from `carriers.rs`:
  `hipfire_arch_minimax::{config_from_safetensors, load_weights_from_safetensors}`,
  `hipfire_runtime::hfq::{config_from_safetensors_llama, load_weights_paroquant_llama}`,
  and the qwen35 `ParoSource`/`config_from_safetensors`/policy wiring.
- Adding GGUF becomes a **single-crate** change: one new arm inside the arch's
  `load_bundle` (plus the source-only arm in `resolve_source_meta`). No
  carrier edit.
- **Zero behavior change.** Every KV policy site, every `from_mode` call, every
  scratch size, every `DeltaNetState` constructor, and — critically — the
  **Dir-skips-`finish_qwen35_load`** asymmetry is byte-identical to today.
- `cargo build` + `coherence-gate.sh` (+ `coherence-gate-dflash.sh` for qwen35
  spec) green. Load-time-only refactor → fresh-process perf Δ≈0%.

## What `load_bundle` can and cannot absorb

The carrier's `load()` body has four layers. Approach B moves layer 2 into the
arch crate; layers 1, 3, 4 are structurally pinned to the loader.

| Layer | Content | Can move into `load_bundle`? |
|---|---|---|
| 1. Refusals + diagnostics | `pp>1` guard, per-source `eprintln!`, error strings | **No** — source/ctx-varying, and the pp>1 error must fire *before* any work. Stays in carrier. |
| 2. **Bundle build** | config + weights + KV + scratch + arch state | **Yes** — this is the whole point. Both `Hfq` and `Dir` arms fold inside. |
| 3. `LoadedModel`-level data | `pp_gpus`, `pp_scratch_set`, `pp_dn_la_to_device` (pp>1); `eviction`, `dflash`, `vision_*` (qwen35 finish) | **No** — these are `LoadedModel` fields, not bundle fields. `load_bundle` returns a `Bundle`, not a `LoadedModel`. |
| 4. Skeleton tail | `LoadedModel { state, ..skeleton(meta) }` | **No** — already shared by Approach A; needs `meta` (tokenizer). Stays in carrier. |

The Tier-2 `WeightSource` trait (`model_load.rs:57`) is what makes layer 2
foldable cleanly: the **only** irreducible Hfq-vs-Dir difference at the tensor
level is *which `WeightSource` impl is constructed* (`HfqSource` vs `ParoSource`
for qwen35), driving one shared `load_weights` orchestrator. The config fn and
KV policy site also branch, but those are small and already live in the arch
crate's namespace.

## Per-carrier analysis

### Minimax — cleanest, but needs a `load_bundle` created

Minimax has no `load_bundle` today; the whole carrier (both `(config, weights)`
arms + shared tail) is in the loader. Its tail is already fully shared
(Approach A). Pushing Dir in means **creating** `hipfire_arch_minimax::load_bundle`
that internally matches `Hfq`/`Dir` and returns the bundle.

One wrinkle: `MiniMaxBundle` is defined in the **loader** (`lib.rs:254`), not the
arch crate, and it carries `eos_tok`, which is **tokenizer-derived** (source-
invariant, computed in the shared tail from `meta.tokenizer`). So `load_bundle`
cannot build the full `MiniMaxBundle` — `eos_tok` isn't available until `meta`
exists. Two clean options:

- **(M-a)** `load_bundle` returns `(MiniMaxConfig, MiniMaxWeights, MiniMaxState)`;
  the carrier tail builds `MiniMaxBundle { config, weights, state, eos_tok }`.
  Smallest change, keeps `MiniMaxBundle` in the loader.
- **(M-b)** Move `MiniMaxBundle` into the arch crate (matching `LlamaBundle`/
  `Qwen35Bundle`), have `load_bundle` return it without `eos_tok`, and either
  pass the tokenizer in or set `eos_tok` in the tail. More consistent with the
  other arches but touches `ModelState::Minimax(..)` wiring.

Recommendation: **M-a** for the no-behavior-change pass; M-b is a follow-on
consistency cleanup (own task). `eos_tok` stays in the shared tail either way —
it is not a bundle-build concern.

### Llama — clean fold, no asymmetry

`LlamaBundle` lives in the arch crate and `load_bundle` already returns it. The
Dir arm currently inlined in `carriers.rs:430-477` moves verbatim into a `Dir`
arm of `load_bundle`. The Dir-side deps (`config_from_safetensors_llama`,
`load_weights_paroquant_llama` — both in `hipfire-runtime::hfq`) are already
reachable from the arch crate (it depends on `hipfire-runtime`). No
`LoadedModel`-level data, no VL, no finish. **Both arms return the same
`LlamaBundle` and the carrier's shared tail is untouched.** This is the model
case for Approach B.

### Qwen35 — the entanglement (drives the approach choice)

Three structural facts block a naive uniform fold:

1. **pp>1 returns `LoadedModel`-level data.** `load_qwen35_pp` (`carriers.rs:114`)
   populates `pp_gpus`/`pp_scratch_set`/`pp_dn_la_to_device` via `skeleton_pp`.
   `load_bundle` returns a `Qwen35Bundle` — it *cannot* carry pp data. pp>1 stays
   a loader-level helper. (It's HFQ-only and Dir+pp>1 already early-errors, so
   this never collides with the Dir fold.)
2. **VL detection needs `hipfire_arch_qwen35_vl`, which `arch-qwen35` does not
   depend on** (verified: only the loader has the dep; the vl crate is
   standalone). Moving VL into `load_bundle` would force a new arch-crate
   dependency. VL stays in the loader/carrier, HFQ-only.
3. **`finish_qwen35_load` produces `LoadedModel` fields** (`eviction`, `dflash`,
   `vision_*`) and lives in the loader because it needs top-of-DAG types
   (`TriAttnCenters`, `load_dflash_state`, `ModelState`). It cannot move into the
   arch crate. **Today HFQ-pp1 goes through `finish`; Dir-pp1 returns a plain
   skeleton (no eviction/dflash/vl).** This asymmetry is load-bearing behavior.

What *can* fold: the Dir-pp1 **bundle build** (`config_from_safetensors` +
`ParoSource` + `load_weights` + `QWEN35_PARO_POLICY` KV + `DeltaNetState::new` +
`Qwen35Scratch::new`) moves into a `Dir` arm of `load_bundle`. After that,
`load_bundle(src, ctx)` returns a `Qwen35Bundle` for both Hfq-pp1 and Dir-pp1.
But the carrier must **still branch after the bundle** to preserve fact 3:

```rust
// pp>1 → load_qwen35_pp (unchanged, HFQ-only)
let bundle = load_qwen35_bundle(src, ctx)?;   // now handles Hfq + Dir
match was_hfq {
    true  => finish_qwen35_load(bundle, meta.., vision..),  // eviction/dflash/vl
    false => Ok(LoadedModel { state: Qwen35(bundle), ..skeleton(meta..) }), // plain
}
```

So even with the Dir fold, the carrier keeps a **post-bundle source branch** —
just a different one. The `(config, weights)` match leaves `carriers.rs`, but a
`was_hfq` / VL-applicability branch arrives. The carrier shrinks but does **not**
become fully source-agnostic for qwen35. This is the cost the prior design
flagged. (Note: `was_hfq` must be captured *before* `src` is moved into
`load_bundle` — `let was_hfq = matches!(src, ModelSource::Hfq(_));`.)

## Approaches

Three ways to scope/shape the push. They differ in how far qwen35 goes and how
the Hfq/Dir selection is expressed inside the arch crate.

### B1 — Tiered push (minimax + llama now; qwen35 deferred) — **RECOMMENDED**

Fold Dir into `load_bundle` for **minimax** (create it, option M-a) and
**llama** (verbatim move). Leave **qwen35** in its Approach-A shape.

- **Pro:** Captures the clean ~80% with near-zero risk. Llama and minimax become
  fully source-agnostic in the carrier; their Dir loader-imports vanish. No
  qwen35 behavior-asymmetry tripwire touched. Each stage independently gated and
  committed.
- **Pro:** Honest scoping — matches the prior design's explicit deferral of
  qwen35 and the project's "land the no-regression refactor" disposition.
- **Con:** `carriers.rs` still imports the qwen35 Dir recipe; GGUF-for-qwen35
  would still touch the carrier. Partial win.
- **Validation gap:** llama (arch <5) and Dir paths are not locally testable on
  k9lin (same gaps Approach A documents); build + clippy + minimax HFQ smoke
  are the local gates, with the Dir/llama paths covered by code review +
  coherence-gate where a model is reachable.

### B2 — Uniform full push (all three, qwen35 included)

Everything in B1 **plus** fold qwen35's Dir-pp1 bundle build into `load_bundle`,
keeping VL/finish/pp in the loader and HFQ-conditional via the post-bundle
`was_hfq` branch shown above.

- **Pro:** The `(config, weights)`/`ParoSource`/`config_from_safetensors` block
  leaves `carriers.rs` for *every* carrier; qwen35's Dir recipe moves to its
  arch crate. Most complete realization of "carrier is source-agnostic."
- **Con:** qwen35's carrier keeps a *different* source branch (`was_hfq` for
  finish/VL), so it is not actually source-agnostic — the win is "moved the
  recipe" not "removed the branch." Marginal structural gain over B1 for qwen35.
- **Con / risk:** The Dir-skips-`finish` asymmetry must be preserved exactly. The
  refactor's correctness hinges on the `was_hfq` capture and on Dir continuing to
  *not* attempt eviction/dflash. A subtle slip (e.g. routing Dir through
  `finish`) silently changes behavior for ParoQuant + `--cask-sidecar`/`--draft`
  users. Higher review burden for a thinner payoff.
- **Stronger only if** combined with B3 (below) to also kill the internal
  branch — otherwise the qwen35 fold mostly relocates code.

### B3 — `WeightSource`-factory push (deepest; generalizes the Tier-2 seam)

Recognize that the irreducible Hfq/Dir difference is *which `WeightSource` is
constructed*. Add an arch-crate constructor that selects the source impl, and a
single bundle path parameterized by it:

```rust
// inside arch-qwen35
fn weight_source<'a>(src: &'a mut ModelSource, c: &'a Qwen35Config)
    -> Result<Box<dyn WeightSource<Layer = LayerWeights> + 'a>, String>
```

The bundle build then has *one* config branch + *one* KV-policy branch and a
shared weights/scratch tail driven by the boxed source.

- **Pro:** Deepest module — the Hfq/Dir axis collapses to the existing Tier-2
  trait it was designed for. Future source kinds slot in as new `WeightSource`
  impls, not new match arms.
- **Con:** Introduces `dyn WeightSource` (boxing + vtable) on the load path, or
  an enum wrapper, for **two** source kinds. This is exactly the
  object-safety/associated-type tension the June-13 ArchSpec/dyn-boundary design
  and the rejected Approach C from the carrier-arm design warned against. The
  config fn and KV policy still branch (they're not part of `WeightSource`), so
  it does **not** fully unify the path — it trades two small matches for one box.
- **Con:** llama's Dir path doesn't even use `WeightSource` (it calls
  `load_weights_paroquant_llama` directly), so B3 only helps qwen35; llama would
  need a `WeightSource` retrofit first. Scope creep.
- **Verdict:** YAGNI now. Revisit only if a 3rd+ source kind (GGUF *and* another)
  lands, making the factory pay for itself. Tracked as a future note, not this
  task.

## Recommendation

**Ship B1 (tiered push).** It is the surgical, zero-behavior-change step that
delivers the clean wins (llama + minimax fully source-agnostic, Dir loader-deps
removed from `carriers.rs` for those two) without touching qwen35's
finish/VL/pp asymmetry. It directly extends Approach A's seam in the same spirit.

Defer the **qwen35 Dir fold (B2)** to its own follow-on task once B1 is landed
and gated, because for qwen35 the fold *relocates* the recipe but leaves a
`was_hfq` branch in place — a smaller payoff that deserves its own focused review
of the Dir-skips-`finish` invariant. Defer **B3** indefinitely (YAGNI; revisit at
the 3rd source kind).

This staging keeps every commit independently gated and preserves the project's
"land the no-regression refactor with an honestly-narrowed claim" posture.

## Migration order (each stage: compiles + gated + commit)

1. **Llama Dir fold.** Move `carriers.rs:430-477` (config/weights/KV/scratch)
   into a `Dir` arm of `hipfire_arch_llama::load_bundle`. Carrier becomes
   `let bundle = load_llama_bundle(src, ctx)?;` + shared tail. Remove now-unused
   `config_from_safetensors_llama`/`load_weights_paroquant_llama` imports from
   `carriers.rs`. `cargo build -p hipfire-loader` + clippy.
2. **Minimax `load_bundle` (option M-a).** Create
   `hipfire_arch_minimax::load_bundle` returning
   `(MiniMaxConfig, MiniMaxWeights, MiniMaxState)` with both arms inside; carrier
   calls it, then builds `MiniMaxBundle { .., eos_tok }` in the shared tail.
   Remove `config_from_safetensors`/`load_weights_from_safetensors` imports from
   `carriers.rs`. Build + `coherence-gate.sh` (minimax HFQ smoke; Dir smoke if a
   safetensors MiniMax dir is reachable).
3. **Sweep.** Delete any orphaned imports/helpers the folds made unused. Verify
   `cargo clippy` clean and `cargo build --example daemon -p hipfire-runtime`.
4. **(Deferred, separate task `t_<new>`) Qwen35 Dir fold (B2).** Only after 1–3
   land: move the Dir-pp1 bundle build into a `Dir` arm of `load_qwen35_bundle`;
   capture `was_hfq` before the move; preserve `finish_qwen35_load` for HFQ-pp1
   and the plain-skeleton return for Dir-pp1. Build + `coherence-gate.sh` +
   `coherence-gate-dflash.sh`.

## Out of scope

- **KV policy unification** — untouched (same as Approach A). Every policy site
  (`LLAMA_HFQ_POLICY`, `DIR_SAFETENSORS_POLICY`, `QWEN35_HFQ_POLICY`,
  `QWEN35_PARO_POLICY`, `QWEN35_PP_POLICY`) moves *with* its arm, byte-identical.
- **qwen35 pp>1** (`load_qwen35_pp`) — stays a loader helper (returns
  `LoadedModel`-level pp data). Not a bundle.
- **VL detection** — stays in the loader (arch-qwen35 has no vl dep).
- **`finish_qwen35_load`** — stays in the loader (top-of-DAG types).
- **`MiniMaxBundle` relocation into the arch crate (M-b)** — optional consistency
  follow-on, its own task.
- **B3 `WeightSource` factory** — YAGNI; revisit at 3rd source kind.
- No change to `ModelState`, `generate*`, the registry/`probe` dispatch layer,
  or kernel/format coverage.

## Verification / required-before-merge

- `cargo build --example daemon -p hipfire-runtime` clean; `cargo clippy` clean
  (post-sweep).
- `coherence-gate.sh` green on affected models + human eyeball.
- Minimax HFQ smoke (Dir smoke if a safetensors MiniMax dir is available).
- `scripts/probe_commits.sh` perf A/B: Δ≈0% (load-time structure refactor).
- Registry disjointness tests (`registry_tests`) unaffected (dispatch untouched).
- **Cannot validate locally (k9lin):** llama (arch <5) and Dir/ParoQuant paths —
  same gaps Approach A flags; not regressed. Covered by code review of the
  verbatim moves + gates where a model is reachable.
