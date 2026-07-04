# Spec-decode extraction: `hipfire-arch-qwen35` → `hipfire-specdecode*` crates

## Goal

Lift the ~19.4k-LOC generic draft-verify speculative-decode engine out of the
qwen arch crate into a `hipfire-specdecode` family, inverting the dependency so
the strategy code is generic over a **target-model trait** that qwen (and future
arches) implement — instead of the strategy code naming `Qwen35*` directly.

## Target crate layout

- `hipfire-specdecode` — **core**. The `SpecDecodeTarget` trait (model boundary)
  + arch-agnostic seam types: `SpecPair`, `SpecStepResult`, `KvMode`,
  `ModelSlotConfig`, `SpecVerifyGraphMode`, `SpecRollbackReplayKind`,
  `VerifyGraphPolicy`, `SpecRollbackParityDecision`, `HiddenStateRingBuffer`,
  `NgramCache`, `PldMatcher`/`PldMatch`, `SpecStats`, the seed-oracle /
  ddtree-meta stats. `ModelSlot<T: SpecDecodeTarget>` lives here.
- `hipfire-specdecode-dflash` — the dflash strategy (bulk of `speculative.rs`,
  446 mentions): draft-flash verify graph, `DflashVerifyOutput`, `VerifyScratch`.
- `hipfire-specdecode-ddtree` — the ddtree strategy (127 mentions):
  `DdtreeScratch`, tree-path verify, ddtree-meta.
- `hipfire-specdecode-mtp` — the MTP strategy: `mtp_spec.rs` + `mtp_head.rs`
  (~6.7k LOC) — `MtpSpecState`, `MtpProposalGraphPolicy`, mtp head/kv/scratch.
- `hipfire-specdecode-dspark` — **empty scaffold** for a future strategy
  (manifest + a `SpecDecodeStrategy` trait-impl stub + `//! TODO`), so the slot
  exists in the workspace.

Dependency direction (no cycle):
`hipfire-arch-qwen35` → `hipfire-specdecode-{dflash,ddtree,mtp}` → `hipfire-specdecode`.
Qwen implements `SpecDecodeTarget` for its `{Config,Weights,Scratch}` in the arch
crate; the strategy crates never name `Qwen35*`.

## The coupling to break (measured)

- `ModelSlot` owns `Qwen35Config`, `Qwen35Weights`, `Qwen35Scratch`,
  `DeltaNetState` (`speculative.rs:574-682`).
- ~15 strategy fns take `&qwen35::Qwen35Weights` + `&qwen35::Qwen35Config` and
  drive the qwen forward + DeltaNet state snapshot/compare.
- The DeltaNet snapshot/parity cluster (~10 `compare_*`/`*_snapshot`/`diff_stats`
  fns, `DeltaNetSnapshot`/`DeltaNetTape`/`GdnTape`) is **genuinely qwen-specific**
  (hybrid linear-attn state) → stays in the arch crate as trait-method impls,
  behind an associated `StateSnapshot` type.
- External consumers = **only 2 demo examples** (`dflash_mtp_demo.rs`,
  `dflash_spec_demo.rs`). No serving-core/daemon reach-in → small public surface.

## `SpecDecodeTarget` trait (the boundary)

```rust
pub trait SpecDecodeTarget {
    type Config;
    type Weights;
    type Scratch;
    type StateSnapshot;                 // DeltaNet/KV rows snapshot (arch-specific)

    fn new_scratch(gpu: &mut Gpu, cfg: &Self::Config, repeat_window: usize) -> HipResult<Self::Scratch>;
    fn n_kv_heads(cfg: &Self::Config) -> usize;
    fn head_dim(cfg: &Self::Config) -> usize;
    // forward/verify entry the strategy calls (draft + verify passes):
    fn verify_forward(...) -> HipResult<...>;
    // state snapshot/compare for rollback-replay parity:
    fn snapshot_state(...) -> Self::StateSnapshot;
    fn compare_state(a: &Self::StateSnapshot, b: &Self::StateSnapshot) -> StateDiff;
    // ... (final method set discovered by making the ~15 fns generic, one at a time)
}
```

The exact method set is *derived*, not guessed: in Phase 2 each `&Weights,&Config`
fn is made generic over `T`, and every `qwen35::` call it makes becomes a trait
method. The trait is "done" when `speculative.rs` compiles with `Qwen35` referenced
only in the `impl SpecDecodeTarget for Qwen35` block.

## Phases (each its own commit on `chaingun`; gate between)

- **P0 — Scaffold (safe, reversible).** Create the 5 crate dirs + manifests +
  empty `lib.rs`, add to `[workspace] members` and `[workspace.dependencies]`.
  `cargo build --workspace` green. No code moved yet.
- **P1 — Move arch-agnostic seam types to core.** Relocate the seam types listed
  above (the ones with **no** `Qwen35*`/`DeltaNet*` reference) into
  `hipfire-specdecode`. `speculative.rs`/`mtp_*` re-export from core so nothing
  breaks. Gate: `cargo build`, `no-gpu-ci.sh`, the 2 examples compile.
- **P2 — Introduce `SpecDecodeTarget`; generify `ModelSlot` + strategy fns.**
  Define the trait in core, `impl` it for `Qwen35` in the arch crate, make
  `ModelSlot<T>` + the ~15 fns generic. Still all in the arch crate — pure
  in-place refactor. Gate: build + `no-gpu-ci.sh`. **GPU gate deferred to P5.**
- **P3 — Move strategies to their crates.** dflash → `-dflash`, ddtree →
  `-ddtree`, `mtp_spec.rs`+`mtp_head.rs` → `-mtp`. Arch crate keeps the
  `impl SpecDecodeTarget for Qwen35` + the DeltaNet snapshot impls + thin
  re-exports for the 2 examples. Gate per crate: build + `no-gpu-ci.sh`.
- **P4 — dspark scaffold.** Manifest + trait-stub + TODO. Build green.
- **P5 — GPU validation (blocking, on capable hardware).**
  `./tests/coherence-gate-dflash.sh` must stay CONFIRMED (this gate is the
  canonical spec-decode correctness check). **Run on a non-LDS-hazard box**
  (halo/gfx1151 or medusa), not nix1, and coordinate with
  `hipfire lock acquire`. Byte-identical dflash/mtp output vs pre-refactor.

## Risk / sequencing notes

- P0–P1 are pure structure + re-export, zero behavior change, immediately
  shippable. P2 is the hard design work but stays in-crate (compiler-verified).
  P3 is mechanical file moves once the trait boundary holds. P5 is the only
  GPU-behavioral gate and must not run on the nix1 LDS-hazard box.
- Portability: pure Rust crate plumbing + a trait indirection; no kernel /
  dispatch / quant-format change. Arch-agnostic.
- This is multi-session. Land P0–P1 (foundation) first; P2 is the commit that
  earns the payoff and needs the most care.
