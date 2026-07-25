> **Current handover — 2026-07-23.** This is a self-contained handover for a
> new session. The authoritative task status remains
> [device-mesh-refactor-tracker.md](device-mesh-refactor-tracker.md); this file
> records the current working-tree stopping point and the next implementation
> boundary.

# Device-mesh / STEP-002 handover

## Current goal

Complete **STEP-002 — Adopt Step/Manifest for MoE** for the all-family MoE
surface: DeepSeek4, MiniMax, and Qwen35 contracts must use the shared
manifest/mesh/dispatch vocabulary for routed-expert ownership, routing,
zero/dummy handling, and collectives.

STEP-002 remains `ready`, not complete. The direct-builder fulfillment
prerequisite (Tasks 2–4) and its evidence documentation (Task 5) have
passed applicable Oracle gates. DeepSeek4 and MiniMax retain their accepted
Single behavior and named structural EP regressions. Full physical EP closure
remains HW-001/HW-002. Vertical architecture cutovers and GPU end-to-end
upload/freeze/rollback evidence remain outstanding. Qwen35 production EP
remains a planned, refused-before-allocation capability owned by AXIS-002
and HW-011; it is not made production-ready by STEP-002.

## Working-tree snapshot

The session started from a dirty worktree. `git status --short` currently shows
changes in:

- `.agent-progress/device-mesh-refactor-tracker.md`;
- `crates/hipfire-runtime/src/{weight_manifest.rs,weight_store.rs,arch.rs}`;
- `crates/hipfire-loader/src/model_parallel.rs`;
- `crates/hipfire-dispatch/src/families/moe.rs`;
- `crates/hipfire-arch-deepseek4/src/arch.rs`;
- `crates/hipfire-arch-minimax/src/minimax.rs`;
- `crates/hipfire-arch-qwen35/src/{store.rs,qwen35.rs,paro_moe.rs}`; and
- untracked `scripts/check_moe_residency_boundary.sh` and
  `scripts/check-weight-store-hybrid-boundary.sh`.

Do not reset, clean, or reformat this worktree. The changes above predate this
documentation update and are not to be committed by the next session without
the appropriate implementation review.

## Completed / approved work

### Tracker contract

The STEP-002 tracker row now makes acceptance explicit: permanent WeightStore
ownership for routed-expert placements and derived resources; private
read-only typed projections; origin-enforcing rank-branded allocation tokens;
and rejection of raw-pointer `WeightStoreView` values. It also records the
Qwen refusal invariant and preserves the canonical Single-vs-emulated-EP2 gate.
The tracker remains the status authority; its STEP-002 `Evidence` is still
`Pending`.

### Manifest contracts

The generic manifest layer is present in
`crates/hipfire-runtime/src/weight_manifest.rs`:

- `ShardPolicy`, including `ExpertSharded` and `ExpertTensorSharded`;
- `WeightEntry` / `StateEntry` and placement validation;
- `collective_for_policy`, `layer_collectives`, and `placement_devices`;
- deterministic `plan_manifest` / `ManifestPlan`; and
- validation of expert shape, shard policy, placement, and collective
  contracts.

`crates/hipfire-runtime/src/weight_store.rs` fulfills whole tensors and
expert-compact placement through the existing generic path, with projection
metadata for static, compact-expert, column, and row placements. This is
manifest/placement foundation work, not the selected frozen-store ownership
implementation.

### Dispatch contracts

`crates/hipfire-dispatch/src/families/moe.rs` now centralizes the MoE dispatch
vocabulary and resolution boundary: `MoeDtypes`, `MoeResolution`, typed MoE
parameter records, prefill resolution, and `MoeFamily` routing. The loader's
`MoEExecutionPolicy` in
`crates/hipfire-loader/src/model_parallel.rs` validates that the effective
named mesh axis matches Single, TP, or EP and rejects competing TP×EP axes.

These are dispatch contracts and policy seams. They do not prove permanent
resident ownership or Qwen production admission.

### Failed ExpertShard ownership removal

The unfinished `ExpertShardResourceKind`, `ExpertShardResource`,
`ExpertShardResident`, `ExpertShardAssembly`, `ExpertShardTarget`, and
`ExpertShardSlot` layer has been removed from tracked Rust sources. The
pointer-table/dummy lifetime fixes in the current DS4, MiniMax, and Qwen diffs
avoid the former `mem::forget` leak path by threading the dummy allocation into
the per-layer owner.

This removal is an approved **reset boundary**, not proof that the new hybrid
store exists. The old ownership model must not be reconstructed.

## Rejected ownership approaches

1. **Architecture-owned `ExpertShardResident` / resource assembly.** Rejected
   because it split ownership between generic placement and architecture
   structs, made partial-rank rollback ambiguous, and invited leaks/double
   frees when pointer tables and dummy buffers outlived their source records.
2. **`WeightStoreAuxiliary` plus `WeightStoreView`.** Rejected because it added
   a second ownership vocabulary, allowed raw-pointer descriptors to outlive
   their actual owner, and made typed extraction/freeing look valid without a
   lifetime or origin proof. `WeightStoreView` is explicitly non-accepting,
   even when it is described as “non-owning.”
3. **Raw `take`/replacement from a mutable store.** Rejected for the final
   architecture because it turns cell identity into temporal mutation and
   lets architecture assembly silently become an owner.
4. **Launch leases in this slice.** Deferred. Do not add a lease abstraction
   while resetting residency ownership; kernel launch lifetime/argument leases
   require a separate contract and are not needed to establish the store
   ownership invariant.

## Selected hybrid architecture

The authoritative design is
`docs/superpowers/plans/2026-07-22-weight-store-moe-residency-recovery.md`:

- `WeightStoreAllocation` is a non-forgeable, non-cloneable, rank-branded
  free authority containing origin mesh epoch, logical rank, physical device,
  and pool epoch. A fallible free consumes the token on success and returns
  the original token with the error on failure.
- `WeightStoreBuilder` owns all staged allocations until freeze.
  `FrozenWeightStore` owns one immutable cell arena keyed by opaque
  `WeightCellId` values. Original routed placements, pointer tables, dummy
  buffers, dtype/layout metadata, and shared sidecars are store cells; there
  is no auxiliary owner.
- Alias resolution happens at freeze. After freeze there is no cell `take`,
  replacement, mutable lookup, or transfer of ownership.
- Qwen35, DeepSeek4, and MiniMax keep private typed read-only projections of
  IDs and aliases. A forward borrows bindings from the frozen owner; the
  projection cannot extract tensors, clone raw views, or free typed weights.
- The loader owns the builder during construction and publishes exactly one
  frozen store into `LoadedModel.weight_store` or `EpState`. Unload consumes
  that same owner. Architecture teardown frees only architecture-owned scratch
  and state.
- **Launch leases are deferred.** The selected architecture uses borrowed
  bindings for this migration and does not claim a launch-lease solution.

## Mandatory Qwen35 acceptance invariant

The canonical gate is preserved exactly in intent and must remain in the next
session's acceptance evidence:

- use the pinned canonical Qwen35-MoE 35B fixture;
- record model SHA-256, prompt MD5, binary digest, exact command, and topology;
- the emulated EP harness uses **Single as the sole baseline**; EP=1 is its
  alias and is not a second required run;
- prefill parity is exact final-prefill logits plus the first token emitted
  after prefill;
- decode parity is exact generated token IDs, with reset and multi-turn
  behavior; and
- report the first logit divergence if tokens differ.

The explicit negative invariant is equally mandatory: throughout STEP-002,
Qwen35Moe EP remains `Planned`/refused before allocation. No emulated test may
construct `EpArch::Qwen35`; no daemon Qwen EP admission may be added; and
**AXIS-002 is the sole Qwen admission owner**. HW-011 owns physical closure
only after AXIS-002 admits the cell.

## Updated stopping point — direct-builder prerequisite complete

This handover was originally written after the unsafe-foundation reset, when
`WeightStoreAllocation`, `WeightCellId`, `WeightStoreBuilder`, and
`FrozenWeightStore` did not yet exist. Tasks 2–4 of the direct-builder
fulfillment plan have since been implemented and passed applicable Oracle gates.
The current tree now contains:

- `WeightStoreAllocation` with live-origin-gated free and retry ownership
  (Task 2).
- `WeightStoreBuilder` with `for_target` full-binding capture,
  `stage_bytes`/`stage_alias` keyed placement, and private adoption surface
  (Task 3, after full-binding remediation).
- `fulfill_manifest_builder` with transactional rollback, retry-owning freeze,
  global-device/local-shard rank separation, whole-arena structural validation,
  and panic-free shard helpers (Task 4, after rank/slicing/arena/no-panic
  remediations).

Legacy `WeightStore`, `WeightHandle`, `WeightStoreAssembly`, and
`fulfill_manifest*` remain untouched for unmigrated callers. Pre-existing
architecture-file worktree changes (Qwen/DS4/MiniMax/depth) are unrelated and
were not modified by Tasks 2–4.

Both boundary scripts pass:

```text
MOE residency boundary check passed: no forbidden ownership symbols in tracked Rust sources under crates/.
weight_store hybrid foundation boundary: passed
```

CPU evidence: `cargo test -p hipfire-runtime weight_store --lib` 102 passed /
9 GPU ignored; `cargo test -p hipfire-runtime weight_manifest --lib` 40 passed;
`cargo test -p hipfire-hardware --lib` 20 CPU passed / 10 GPU ignored (GPU
tests executed separately as allocation-domain identity evidence, NOT
direct-builder GPU proof). GPU upload/freeze/rollback fixtures are explicitly
ignored — unavailable/unexecuted direct-path evidence.

## Next work — architecture vertical cutovers and GPU evidence

The direct builder fulfillment prerequisite is complete. Outstanding work on the
roadmap:

1. **Architecture vertical cutovers** (Qwen, DeepSeek4, MiniMax) — each
   architecture's private typed projection, frozen-store publication, and unload.
   Qwen EP admission remains **refused** (AXIS-002 owner). No architecture code
   has been changed by Tasks 2–4.
2. **GPU end-to-end evidence** — single and emulated mesh upload/freeze/rollback
   tests remain `#[ignore]`d because compatible AMD hardware/fixtures are not
   verified for these scenarios. They are unavailable evidence, not passes.

Do not claim STEP-002 completed, Qwen EP admission, or production GPU
validation. Do not begin adding another view, auxiliary ledger, or launch
lease.

## Verification already passed

- `bash scripts/check_moe_residency_boundary.sh` passed.
- `bash scripts/check-weight-store-hybrid-boundary.sh` passed.
- The tracker records STEP-001 manifest/Step parity, Qwen35 coherence, and
  serve-multiturn evidence as complete; those are prior evidence, not STEP-002
  completion.
- The tracker records the existing manifest, dispatch, model-parallel, and
  failed-ownership-reset work as the current foundation; STEP-002 evidence
  remains pending.

Before handing off implementation, run `git diff --check` and inspect only the
intended source changes. Do not commit from this handover session.

## Active risks

- The current mutable store can still be mistaken for the selected immutable
  store; keep the Phase 0 boundary script active until every old API is gone.
- `GpuTensor` has no ordinary freeing `Drop`; any temporary or derived
  allocation not registered under the eventual token owner can leak on a
  later-rank failure.
- Pointer tables bake physical addresses. Borrowed binding construction must
  prove the cell/store lifetime and must not recreate raw cloneable views.
- DS4/MiniMax EP has physical RCCL gates HW-001/HW-002; emulation is not
  production hardware evidence.
- Qwen's canonical 35B fixture and all required digests are acceptance
  blockers. Missing evidence is a failed/incomplete gate, not an invitation to
  substitute a smaller model.
- Existing dirty changes include dispatch/manifest and dummy-buffer work. Keep
  the hybrid ownership migration separate and review the full diff before any
  integration.

## Files to read first

1. `.agent-progress/device-mesh-refactor-tracker.md` — authoritative status,
   especially STEP-002 at lines 412–420 and AXIS-002 at lines 482–490.
2. `docs/superpowers/plans/2026-07-22-weight-store-moe-residency-recovery.md` —
   selected hybrid phases, exact paths, TDD tasks, and old-task supersession.
3. `crates/hipfire-runtime/src/weight_store.rs` — current mutable store and
   the exact Phase-0-to-Phase-1 seam.
4. `crates/hipfire-runtime/src/weight_manifest.rs` — placement and collective
   contracts.
5. `crates/hipfire-dispatch/src/families/moe.rs` — centralized MoE dispatch
   contracts.
6. `crates/hipfire-loader/src/model_parallel.rs` — named-axis MoE execution
   policy and refusal checks.
7. `crates/hipfire-arch-qwen35/src/{store.rs,qwen35.rs,paro_moe.rs}` — raw
   projection/ownership surfaces to migrate only after the frozen store.
8. `crates/hipfire-arch-deepseek4/src/arch.rs` and
   `crates/hipfire-arch-minimax/src/minimax.rs` — routed placement consumers.
9. `scripts/check_moe_residency_boundary.sh` and
   `scripts/check-weight-store-hybrid-boundary.sh` — reset and hybrid boundary
   checks.
