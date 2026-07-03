# Arch capability layer — plan

Status: planned (Phase 0 not started)
Scope: build the capability-discovery + registry layer FIRST, prove it on the toy
arch, then migrate real families onto it in later phases (separate effort).

## Why

The model math is per-family (`crates/hipfire-arch-*`), but every *integration*
concern is scattered as `arch_id == N` / `is_<family>` branches across the shared
crates — 72 `arch_id == <lit>` sites, plus per-arch quant policy in
`quantize/main.rs`, session/state logic in `serving-core` + `daemon`, and toy-model
generation in a monolithic `quantize/fixture.rs` `match arch` (`emit_fixture`).
Optional serving features (DFlash, DDTree, batched prefill, …) are *not* expressed
in the type system, so the daemon can reach `arch_batch_prefill` on an arch that
doesn't support it and only fail at runtime.

Goal: each family declares what it can do by *implementing traits in its own crate*;
support is **discovered from the impls at compile time**; crates **auto-register**
by being linked; and unsupported paths are **unrepresentable**, not runtime-guarded.

## What already exists (build on, don't replace)

`crates/hipfire-runtime/src/arch.rs` already has `Architecture` (base), `SimpleAr`,
`ServingBackend`, `SessionServingBackend`, `run_simple_ar`, `decode_loop`. The
capability layer does NOT rewrite these — it (a) adds an arch-identity registry so
they're reached without `match arch_id`, (b) expresses the currently-branched
optional features as *optional* capability traits, and (c) auto-derives which caps
each family has. `SessionServingBackend` becomes the `Sessions` capability;
`Architecture` (or a thin new `Arch`) is the mandatory base.

## Phase 0 — the capability layer (this plan)

### 0.1 New leaf crate `hipfire-arch-api`

Minimal deps. Defines the whole surface so arch crates and shared crates depend on
*it*, not on each other:

- `ArchId` (newtype over the existing arch_id ints; the literals move here as named
  consts — the ONLY place they live).
- **Mandatory** base trait — reuse/relocate `Architecture` (identity + config +
  forward entry). Every arch impls this.
- **Optional capability traits**, one per feature — named for WHAT THEY DO on WHICH
  AXIS, not the legacy codenames (see the mapping table below). By axis:
  - decode: `Autoregressive` (plain AR), `SpecDecodeChain` (chain-draft speculative
    decode), `SpecDecodeTree` (tree-of-candidates speculative decode).
  - prefill: `BatchedPrefill`, `PrefillCompress` (drafter-scored prompt compression).
  - drafter: `MtpDrafter` (multi-token-prediction heads; orthogonal to the above).
  - stateful serving: `Sessions` (= `SessionServingBackend`).
  - offline/tooling: `Ingest` (quantizer tensor→precision policy, kills the
    `is_lfm2`/`is_minimax` blocks in `quantize/main.rs`), `Calibration`, `KldEval`,
    `ToyModel` (see 0.4).
  - policy: `KvPolicy` (KV mode per arch).
  Start Phase 0 with just `SpecDecodeChain` + `BatchedPrefill` + `ToyModel` to prove
  it; the rest are added trait-by-trait as arches migrate.

#### Naming: describe the behaviour + axis, not internal codenames

Today's names are opaque and overload "Flash" across unrelated axes (DFlash =
batched-verify *decode*; PFlash = prompt *compression at prefill* — a different
thing entirely), which hides that DFlash/DDTree are the same strategy differing only
in draft topology. The capability traits use descriptive, literature-aligned names;
the codenames become **legacy identifiers renamed per-arch during migration** (a
mechanical, contained rename):

| legacy | capability trait | what it is |
|---|---|---|
| SimpleAr / AR | `Autoregressive` | plain AR — one target token per forward |
| DFlash | `SpecDecodeChain` | speculative decode, chain draft, batched ("flash") verify |
| DDTree | `SpecDecodeTree` | speculative decode, tree of draft candidates, batched verify |
| (normal prefill) | `BatchedPrefill` | batched prompt prefill |
| PFlash | `PrefillCompress` | drafter-scores prompt blocks, keeps important spans + anchors |
| MTP | `MtpDrafter` | multi-token-prediction draft heads (draft source) |

Rule going forward: a capability trait's name says its behaviour and axis; no new
`*Flash`/codename traits. The `no-gpu-ci` gate (0.5) can also lint the codenames out
of new public trait/`Caps` surface.
- `Caps` struct: `Option<&'static dyn Cap>` per capability.

### 0.2 Compile-time capability discovery (the core trick)

Support must be *derived from the impls*, never hand-listed (a manifest drifts).
Use autoref specialization ("spez" / the `impls!` pattern): for each cap, a helper
`maybe_<cap><T>(&T) -> Option<&'static dyn Cap>` that resolves to `Some` iff
`T: Cap`, else `None`, at compile time.

```rust
// resolves at COMPILE time via autoref; Some(..) only if T: SpecDecodeChain
fn maybe_spec_decode_chain<T>(x: &'static T) -> Option<&'static dyn SpecDecodeChain> { … }
```

`register_arch!($T)` then builds `Caps` by calling every `maybe_*` on the arch
singleton — so `impl SpecDecodeChain for Lfm2Moe {}` is *all* it takes for
`Caps.spec_decode_chain` to become `Some`. Adding a feature = impl a trait; discovery
is automatic.

RISK: autoref specialization is arcane and can be toolchain-fragile. Mitigation /
fallback: a declarative form `register_arch!(Lfm2Moe { caps: [SpecDecodeChain, BatchedPrefill] })`
where each listed cap expands to `Some(&INSTANCE as &dyn SpecDecodeChain)` — which only
COMPILES if the impl exists (so you can't over-claim), paired with the 0.5 gate to
catch under-claiming (impl'd but not listed). Decide autoref-vs-declarative by a
spike in 0.2 before committing.

SPIKE RESULT (2026-07-03): autoref specialization is too fragile — two call forms
gave two wrong behaviours (`Probe(x)` errors on the unsatisfied bound instead of
falling through; `(&Probe(x))` compiles but resolves inverted, `T: Cap → None`).
**Decision: DECLARATIVE.** `register_arch!(Ty { caps: [SpecDecodeChain, …] })` where a
fixed `set_cap!` dispatch maps each cap ident to its `Caps` field with a
`Some($inst as &'static dyn Cap)` cast — compile-fails on over-claim; the 0.5 gate
(caps ⟺ impls) catches under-claim. Adding a capability = one `set_cap!` arm + one
`Caps` field (central, small). No unstable features, no toolchain fragility.

### 0.3 Auto-registration by linking (`inventory`)

Each arch crate does `register_arch!(Foo)`, which `inventory::submit!`s an
`ArchEntry { id, make_caps, base }`. `hipfire-arch-api` exposes `ArchRegistry` that
reads the inventory once and maps `ArchId → (&dyn Architecture, Caps)`. Central code
gains NO per-arch list.

Cargo caveat (state it plainly): Rust must LINK the crate for its `inventory`
submission to exist. So a thin bundle crate `hipfire-archs` `pub use`s each arch
crate (also forces linking against dead-code elimination). "Add a family" =
`crates/hipfire-arch-foo/` + one `pub use` line in the bundle. Optionally a
`build.rs` scans `crates/hipfire-arch-*` and generates that bundle so even the line
is automatic — evaluate in 0.3, don't block on it.

### 0.4 The `ToyModel` capability (move fixtures into the crates)

`quantize/fixture.rs::emit_fixture(arch, out, seed)` currently `match`es on the arch
name and holds every family's fixture spec. Define:

```rust
pub trait ToyModel { fn emit_fixture(&self, out: &Path, seed: u64) -> Result<(),String>; }
```

Each arch's fixture spec moves into its crate's `ToyModel` impl (e.g. the toy crate's
`toy_model.rs` already has the shape). `fixture.rs` becomes a 3-line dispatcher:
`registry.get(arch)?.caps.toy_model.ok_or("no toy for arch")?.emit_fixture(...)`.
This is a clean first migration target because it's offline, self-contained, and has
an obvious completeness meaning.

### 0.4b Offline vs serving capabilities — respect the dependency direction

CRITICAL constraint: `hipfire-quantize` deps NO arch/runtime crate today (standalone
offline), but arch crates dep `hipfire-runtime` + `rdna-compute` (GPU). So the OFFLINE
capabilities (`Ingest`, `ToyModel`, `Calibration`) must be implementable WITHOUT
dragging the serving/GPU stack into the quantizer.

The quantizer's arch-specific policy that becomes `Ingest` (today in
`quantize/main.rs`): tensor classifiers `is_conv1d_tensor`,
`is_routed_expert_tensor_name`, `is_nemotron_h_mq4_q8_protected`,
`is_nemotron_h_residual_writer`, `is_deepseek4_keep_f16`, `is_q8_tensor`,
`is_positional_promote`; per-arch AWQ-alpha defaults; MoE / mixed-precision-promotion
policy; `match arch_str` config detection. All pure offline policy (names → precision),
no GPU.

**DON'T relocate these name-matchers verbatim — they're the WRONG SHAPE.** Each
conflates a *universal tensor role* with an *arch-specific naming convention*:
`is_q8_tensor` is one shared function that knows qwen3.5 DeltaNet + qwen3.5-MoE router
+ nemotron-H router names at once (edited for every new arch); `is_nemotron_h_residual_writer`
/ `is_deepseek4_keep_f16` are arch-name-prefixed one-offs for the universal roles
"residual writer" / "precision-sensitive". Relocating them into per-arch `Ingest`
impls just decentralizes the smell. A per-role precision FLOOR is still wrong: it lets the arch name a format (Q8/f16),
which the arch has no business knowing. Absolute rule: **the arch declares NEEDS,
never SOLUTIONS — no format name and no arch name appears in the arch's data** (the
crate is arch-named; the fields inside are format-agnostic). Three decoupled layers:

- **ARCH declares needs** (in its `-spec` crate; `role(name) -> TensorRole` is the
  mechanical name→role map these derive from):
  - `importance(name) -> u8` — 0..255 saliency prior (soft). "keep f16" is not a
    concept; a numerically-critical tensor is just importance ≈255.
  - `requires(name) -> CapReq` — format-AGNOSTIC capability needs, e.g. `RandomAccess`
    (a gather-indexed tensor like embed/lm_head needs a random-accessible codec).
    Shape divisibility is NOT declared — it's derived generically from tensor shape vs
    each codec's group size.
- **CODEC REGISTRY**: each codec declares a `CodecCaps` — `random_access`, `group_size`,
  `bits_per_weight`, `act_bits`, …. No arch names here either.
- **DEPLOYMENT allocator** (owns the budget): per tensor, (1) filter codecs to those
  satisfying `requires(name)` AND whose group size divides k; (2) over that valid set,
  pick by `importance ⊕ measured-saliency` vs the target avg bits/weight — e.g. a curve
  <50 → qtip2, 50..250 → qtip4, >250 → oq8. (Measured saliency = GuidedQuant Fisher/
  Hessian when calibration exists.)

So NOTHING arch-side names a format. `is_q8_tensor`'s embed/lm_head become
importance-high + `RandomAccess`, and the allocator DERIVES a random-access codec at
high bits (which happens to be Q8F16) — "Q8" is never written down. `is_deepseek4_keep_f16`'s
compressor/indexer become importance ≈255, and the allocator lands them at the top of
the budget — neither "f16" nor "deepseek4" appears (it's just that crate's data).
`is_nemotron_h_residual_writer` → elevated importance on the `ResidualWriter` role.
Adding a codec, or re-budgeting, touches neither the arch nor the tensor policy.

Two ways to keep deps clean (decide before migrating `Ingest`):
- **(a) split each family** into a lean offline core `hipfire-arch-<x>-spec`
  (identity, config, `Ingest`, `ToyModel`, `Calibration` — deps only
  `hipfire-arch-api` + `hipfire-quant-format`/`-primitives`) and the serving crate
  `hipfire-arch-<x>` (forward, `Sessions`, `SpecDecode*` — deps the core + runtime +
  rdna-compute). Quantizer deps the `-spec` bundle; daemon deps the serving crates;
  the registry collects both tiers. Cleanest boundary; mirrors the existing leaf-crate
  / `hipfire-coexistence` split. More crates.
- **(b) feature-gate one crate**: offline caps always compiled, serving caps + runtime/
  GPU deps behind a default `serving` feature; quantizer deps `default-features = false`.
  Fewer crates, but offline modules must never touch a runtime/GPU type (fragile; needs
  a no-default CI build to enforce). Matches the pattern `hipfire-quantize`'s own `gpu`
  feature already uses.

Recommend **(a)** for a strict, non-fragile boundary. What STAYS arch-GENERIC in the
quantizer (NOT a capability): the codec math (shared lib), LDLQ, byte format, CLI, the
GPU trellis encoder, FWHT, on-disk packers.

### 0.5 Completeness gate (no-gpu-ci) — freeze the scatter

A pure-CPU test that:
- every `ArchId` in `model-support.toml` has a registered `ArchEntry` (and vice-versa);
- **`model-support.toml` per-arch feature flags are GENERATED-from / validated-against
  the registered `Caps`** — so the doc/admission matrix can't lie about which arch has spec-decode / prefill-compress / etc.;
- a grep-lint failing on any NEW raw `arch_id == <lit>` outside `hipfire-arch-api`
  (freezes the count at today's 72 while migrations drain it).

### 0.6 Reference implementation on `hipfire-arch-toy`

Wire the WHOLE mechanism for the toy arch only (`hipfire-arch-toy`, 240 lines):
impl the base trait + `SpecDecodeChain` (stub) + `BatchedPrefill` (stub) + `ToyModel`
(real — it already has toy_model.rs), `register_arch!`, add to the bundle, and have
the daemon/scheduler consult `caps.batched_prefill` with the `Some/None` match. Prove
end to end that: (a) a supported cap dispatches, (b) an un-impl'd cap is `None` and
the compiler forces the fallback arm, (c) the gate passes. NO real family migrates in
Phase 0.

## Deliverable of Phase 0

`hipfire-arch-api` (traits + `Caps` + discovery + `register_arch!` + `ArchRegistry`),
the `hipfire-archs` bundle, the completeness gate, and the toy arch fully on it — with
the daemon calling one capability (`BatchedPrefill`) through the registry as the
proof. Real-family migration (`Ingest` quant policy first, `Sessions`/serving last —
per the existing model-family-onboarding + session-serving-backend plans) is the NEXT
phase and is out of scope here.

## Open questions

- Autoref-spec vs declarative `register_arch!` (0.2 spike decides).
- `inventory` dead-code / bundle vs `build.rs` folder scan (0.3).
- dyn dispatch cost: negligible — caps are selected once per load/forward, not per
  op (per-op kernel dispatch stays the separate `KernelKey` system). Confirm no cap
  method sits in the per-token inner loop; if one does, hold a resolved fn-ptr.
- Reconcile the existing `Architecture`/`SimpleAr`/`ServingBackend` trio: does base
  `Arch` == `Architecture`, and are `SimpleAr`/`ServingBackend` two more capabilities
  or the base? Settle before 0.1 to avoid a fourth abstraction.
