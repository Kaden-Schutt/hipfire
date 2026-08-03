> **Current handover — 2026-08-03 (Phase C boundary/evidence).** This is a
> self-contained handover for a new session. The authoritative task status
> remains
> [device-mesh-refactor-tracker.md](device-mesh-refactor-tracker.md); this file
> records the current working-tree stopping point and the next implementation
> boundary.

# Device-mesh / STEP-002 handover

## Current goal

Complete **STEP-002 — Adopt Step/Manifest for MoE** for the all-family MoE
surface: DeepSeek4, MiniMax, and Qwen35 contracts must use the shared
manifest/mesh/dispatch vocabulary for routed-expert ownership, routing,
zero/dummy handling, and collectives.

STEP-002 remains `ready`, not complete. The Qwen35 single-device HFQ Frozen
MoE cutover (single-target facade, ID-only projection/bindings, direct Frozen
staging, exact C2 admission, source-bound preflight/Legacy fallback, checked
published/unpublished unload/backlog) passed Oracle Gate B on 2026-08-03.
DeepSeek4 and MiniMax retain their accepted Single behavior and named
structural EP regressions. Full physical EP closure remains HW-001/HW-002.
The canonical Qwen35-MoE GPU fixture/parity evidence and VRAM recovery
evidence remain absent, and STEP-002R (best-effort pre-publication common/
auxiliary rollback) is accepted debt — neither is closed. Qwen35 production EP
remains a planned, refused-before-allocation capability owned by AXIS-002 and
HW-011; it is not made production-ready by STEP-002.

## Working-tree snapshot

The session started from a dirty worktree. `git status --short` currently shows
changes in the Phase A/B/C implementation set:

- `.agent-progress/device-mesh-refactor-tracker.md`,
  `.agent-progress/device-mesh-HANDOVER.md`;
- `crates/hipfire-arch-qwen35/src/{carrier.rs,dflash_spec.rs,layer_driver.rs,
  lib.rs,mtp_head.rs,mtp_speculator.rs,qwen35.rs,store.rs}`;
- `crates/hipfire-arch-cohere2moe/src/forward.rs`,
  `crates/hipfire-arch-deepseek4/src/{dspark_speculator.rs,mtp_speculator.rs}`,
  `crates/hipfire-arch-llama/src/dspark_body.rs`;
- `crates/hipfire-dispatch/src/{context.rs,coverage_tests.rs,families/moe.rs,
  resource/mod.rs,tests.rs}`, `crates/hipfire-dispatch-tests/src/qwen35.rs`;
- `crates/hipfire-hardware/src/lib.rs`,
  `crates/hipfire-loader/src/{carriers.rs,lib.rs}`,
  `crates/hipfire-runtime/src/{dflash_generic.rs,dspark_core.rs,llama.rs,
  spec.rs,spec_ngram.rs,weight_store.rs}`;
- `crates/rdna-compute/src/{dispatch.rs,pool.rs}`;
- `scripts/check_moe_residency_boundary.sh` (tracked; Phase C expanded it with
  executable boundary assertions) plus untracked docs/deepwork files.
- Unrelated untracked files are omitted from the task diff and were not
  touched: `crates/graphify-out/`, `graphify-out/`, `docs/pr-dspark-qwen3.md`,
  and `docs/pr-dspark-qwen35.md`.

Do not reset, clean, or reformat this worktree. The changes above predate this
documentation update and are not to be committed by the next session without
the appropriate implementation review. Two destructive recovery incidents in
this history are recorded below; do not use git checkout/restore/reset/stash
here.

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

Historical CPU evidence (2026-07-23 direct-builder prerequisite, superseded by
the Phase C numbers below): `cargo test -p hipfire-runtime weight_store --lib`
102 passed / 9 GPU ignored; `cargo test -p hipfire-runtime weight_manifest
--lib` 40 passed; `cargo test -p hipfire-hardware --lib` 20 CPU passed / 10 GPU
ignored (GPU tests executed separately as allocation-domain identity evidence,
NOT direct-builder GPU proof). GPU upload/freeze/rollback fixtures are
explicitly ignored — unavailable/unexecuted direct-path evidence.

## Phase B → C checkpoint (2026-08-03)

### Oracle Gate B approval

Oracle Gate B **APPROVED** (2026-08-03) with zero Critical or Important
findings after iterative remediation. The final state: the exact dispatch
snapshot is bound into a sealed source-owning plan; the target is verified
before vision or Frozen allocation; Legacy fallback is preserved only for
preflight Ineligible outcomes; complete cleanup aggregates are retained;
exact-domain retry and pool drain are serialized; and an abort-on-unwind guard
protects the post-selection vision owner. Focused evidence at the gate: Qwen
374 passed / 15 ignored, four unwind-guard tests passed.

### STEP-002R debt (accepted, not closed)

By explicit user decision on 2026-07-27, best-effort rollback of
pre-publication common and auxiliary allocations is accepted for this slice
and tracked as STEP-002R. It does **not** count as exact failed-free retention
evidence. The relaxation applies only to failed common/auxiliary construction
rollback; it does not relax sole Frozen MoE ownership after publication,
complete Frozen failure propagation where already available, checked
unload/backlog ordering, dispatch admission, refusal boundaries, or honest
GPU-evidence reporting. STEP-002R remains open.

### Phase B TDD ordering violation

Phase B did not follow test-first ordering. This is an explicit process
failure; no strict-TDD claim is permitted for Phase B. The violation is
preserved in the tracker and deepwork records.

### Two destructive recovery incidents

1. **First recovery incident (2026-07-26):** the lifecycle remediation writer
   ran `git checkout HEAD --` on dirty `qwen35.rs`, `store.rs`, `carrier.rs`,
   loader `lib.rs`, and loader `carriers.rs`, erasing all unstaged Phase 2
   work in those files. No blob, stash, index entry, reflog entry, or
   temporary copy was recoverable. The user chose fresh reimplementation;
   surviving dirty changes are checkpointed at
   `/tmp/opencode/qwen35-recovery-survivors-20260726.patch`.
2. **Second recovery incident (2026-07-27):** the delegated common-transaction
   writer violated the no-checkout rule again with `git checkout --
   crates/hipfire-arch-qwen35/src/store.rs`, erasing the full unstaged file.
   Recovery used the pre-incident snapshot `/tmp/opencode/oracle-gate-a.diff`;
   its `store.rs` patch base exactly matched HEAD (`f95095dc`) and restored
   blob `f59537b5`. The full worktree was then checkpointed at
   `/tmp/opencode/qwen35-post-second-recovery-20260727.patch`.

Rule for every delegated writer: **no git checkout/restore/reset/stash** in
this worktree, and future lanes must be smaller than the failed
all-common-transaction assignment.

### Implementation state (post-Gate B, Phase C)

- Single-target frozen-store facade with retained target identity
  (`SingleFrozenWeightStore` + `SingleWeightStoreBuilder` in
  `crates/hipfire-runtime/src/weight_store.rs`).
- Private ID-only Qwen35 MoE projection (`Qwen35MoeResident` +
  `Qwen35MoeLayerProjection<WeightCellId>` + borrowed `MoeFfnBindings`) in
  `crates/hipfire-arch-qwen35/src/store.rs`; no `GpuTensor` fields, no raw
  views, no typed-free-authority exposure.
- Direct Frozen MoE staging (`build_frozen_moe_resident`) with exact C2
  indexed dispatch admission and routed-down AWQ (routed gate-up AWQ rejected
  before upload).
- Source-bound preflight with Legacy fallback; Frozen refused at every
  multi-device entry (`reject_frozen_multi` in all three multi forward
  entries).
- Qwen35Moe EP remains `Planned`/refused before allocation: `EpArchKind`
  (loader) has no Qwen35 variant, `validate_ep_layout` refuses non-DS4/MiniMax
  architectures, the capability matrix keeps `(Qwen35Moe, Ep) => Planned`
  owned by AXIS-002, and there is no `EpArch::Qwen35` or daemon admission.
- Residency boundary script now asserts all of the above
  (`scripts/check_moe_residency_boundary.sh`, Phase C).

### Exact verification commands and results (2026-08-03, Phase C)

```text
$ bash scripts/check_moe_residency_boundary.sh
MOE residency boundary check passed: no forbidden ownership symbols in tracked Rust sources under crates/.
Also passed: ID-only projection fields, Frozen staging-path purity (no from_raw/alias),
from_raw legacy whitelist, no public ownership-surface exposure, multi-device Frozen refusal,
and Qwen35 EP Planned/refused admission (no EpArch::Qwen35, no daemon admission).
exit 0

$ bash scripts/check_moe_residency_boundary.sh --self-test
24 assertion failure(s) total; all 16 expected category/categories caught.
exit 0  — every expected violation category (ExpertShard family, WeightCellId::for_test,
Check 4 public WeightStoreAllocation/raw/adoption surface, ID-only projection, Frozen
from_raw/alias, from_raw whitelist, multi-entry refusal, EP arch/layout/matrix) caught
independently; the script fails closed at startup if a required tool is missing

$ cargo test -p hipfire-runtime weight_store --lib
110 passed; 0 failed; 13 ignored

$ cargo test -p hipfire-runtime --doc
5 passed; 0 failed  (all five weight_store compile-fail doctests)

$ cargo test -p hipfire-arch-qwen35 --lib
374 passed; 0 failed; 15 ignored

$ cargo test -p hipfire-loader --lib
132 passed; 0 failed; 10 ignored

$ cargo test -p hipfire-dispatch --lib
171 passed; 0 failed; 1 ignored

$ cargo test -p hipfire-dispatch-tests
70 passed; 0 failed

$ cargo test -p rdna-compute --lib dispatch   (narrow)
34 passed; 0 failed; 27 filtered

$ cargo test -p rdna-compute --lib            (full, covers pool-affected)
61 passed; 0 failed; 0 ignored

$ cargo check -p hipfire-loader --all-targets
Finished; warnings only (3 pre-existing warnings in lib test)

$ cargo check --workspace --all-targets
Finished; warnings only (pre-existing; no errors)

$ rustfmt --edition 2021 --check --config skip_children=true <29 changed .rs files>
exit 1 — pre-existing formatting drift in mtp_head.rs, qwen35.rs, weight_store.rs
(Phase A/B worktree drift; NOT formatted/churned per scoping rule)

$ git diff --check
exit 0
```

Rust files changed by Phase C itself: none (boundary script is bash; docs are
Markdown). The rustfmt failure above is entirely pre-existing worktree drift.

### Pending GPU evidence and final gate

Required evidence that remains **unavailable/skipped** (no compatible
canonical fixture + time):

- graph-enabled generate/unload/reload on the canonical Qwen35-MoE fixture;
- post-unload VRAM recovery with no monotonic growth across cycles;
- DFlash coherence on the Frozen cutover;
- final-prefill logits / first token / decode token IDs / reset / multi-turn
  parity against a pinned Single baseline;
- model SHA-256, prompt MD5, and binary digest for the canonical fixture.

Hardware/fixture facts (2026-08-03): one AMD GPU present (gfx1151, RYZEN AI
MAX+ 395 / Radeon 8060S). `~/.hipfire/models/qwen3.6-35b-a3b.mq4` (22 GB) is
present and its MD5 matches the AGENTS.md-pinned A3B digest
`edde51ec1dac0f2bd42cff5ef1cb8944`, but that pin is documented for a DFlash
perf thread on a differently-named artifact and is not the STEP-002 canonical
Qwen35-MoE acceptance fixture (no model SHA-256 / prompt MD5 / binary digest
pinned for this gate, no paired A3B draft). GPU evidence therefore remains
absent, not passed. The final merged Oracle gate APPROVED the complete
ownership/lifecycle cutover on 2026-08-03 with zero Critical or Important
findings outside accepted STEP-002R and named this evidence gap explicitly.

## Next work — STEP-002R and GPU evidence

Outstanding work on the roadmap:

1. **GPU end-to-end evidence** — graph-enabled generate/unload/reload, VRAM
   recovery, coherence, and canonical Qwen35-MoE parity with pinned hashes.
   These are unavailable evidence, not passes.
2. **STEP-002R** — origin-preserving, retryable common/auxiliary construction
   rollback (accepted debt; exact failed-free retention is not claimed).

Do not claim STEP-002 completed, Qwen EP admission, or production GPU
validation. Do not begin adding another view, auxiliary ledger, or launch
lease. Do not use git checkout/restore/reset/stash in this worktree.

## Final merged Oracle remediation (2026-08-03, fresh narrow lane)

All four final merged Oracle findings are fixed with strict TDD (focused RED
tests first), plus the final Legacy assembly Important finding:

1. **FROZEN MODEL-WIDE MQ6 FENCE (final Oracle Important gap, widened)** —
   the fence is defined as ANY MoE FFN projection in any layer: router /
   shared_expert_gate / shared gate/up/down plus every routed expert
   gate_up/down (uniform or graded), for BOTH Legacy and Frozen, through
   ONE shared metadata predicate `MoeFfnMetaView::has_mq6` (generic over
   the projection key; metadata-only, no tensor lookup). `layers_have_mq6_moe`
   (Legacy) and the Frozen resident publication both consume it, so the
   two storage kinds cannot diverge. This closes two gaps: the old
   snapshot predicate required uniform routed experts (graded MQ6 was
   missed) and the Frozen path covered only routed experts (structural
   MQ6 was missed). Derived BEFORE publication/attachment at the Phase 6
   seam of `load_qwen35_hfq_weights_frozen_prepared`. The gfx1151 prefill
   fence (`force_mq4_grouped_fp16 = model_has_mq6_moe && is_gfx1151 &&
   moe_grouped_i8.is_none()` in `prefill_moe_ffn_body_batched`) reads the
   published field. RED: `moe_view_has_mq6_detects_graded_routed_mq6`
   failed behaviorally (snapshot missed graded MQ6); the store.rs shared-
   field table tests could not compile (non-generic meta view). GREEN:
   `projection_layers_mq6_fence_mixed_true_pure_mq4_false` (routed, preserved),
   `moe_ffn_meta_view_mq6_fence_covers_every_shared_field` (cross-layer:
   layer A pure MQ4 + layer B each shared field MQ6 → true; pure all-MQ4 →
   false), `moe_ffn_meta_view_mq6_fence_covers_graded_routed_experts`,
   `moe_view_has_mq6_detects_graded_routed_mq6` /
   `moe_view_has_mq6_detects_every_shared_projection` (Legacy), and the
   GPU-ignored `frozen_publication_derives_model_wide_mq6_fence` extended
   with a structural-MQ6 fixture (layer-1 shared projections MQ6, routed
   MQ4 → true) — all pass on gfx1151 with `--ignored`. Also repaired the
   stale GPU-ignored `frozen_moe_resident_build_and_bind` (used a k=1
   config that Frozen admission refuses, and its panic poisoned
   `GPU_TEST_LOCK` for the other GPU tests); it now uses the valid
   k=8/MQ4 fixture.
   **Legacy assembly gap (final Important finding):** the Legacy
   assembly (`assemble_qwen35_weights_inner_with_mode`) still derived
   `moe_has_mq6` from its own inline routed-only scan — structural MQ6
   (router/shared) in a Legacy checkpoint was missed. It now publishes the
   fence via the shared CPU-testable seam
   `assembled_legacy_layers_have_mq6` (per-layer
   `MoeFfnMetaView::Legacy(ffn).has_mq6()`; Frozen markers → false, the
   resident publication derives later); `layers_have_mq6_moe` delegates to
   the same seam so there is exactly one layer-scan implementation. RED:
   `legacy_assembly_derives_model_wide_mq6_fence` failed behaviorally
   (shared-only MQ6 fixture published `moe_has_mq6=false`); CPU seam test
   was compile-RED (seam missing). GREEN: the same GPU test passes on
   gfx1151 (shared-only MQ6 → true, pure MQ4 → false through the real
   `load_qwen35_hfq_weights` legacy loader), and CPU
   `assembled_layers_mq6_seam_shared_only_layer_and_pure_layer` covers the
   seam (shared-MQ6 + pure-MQ4 layers → true; all-MQ4 → false; Frozen
   markers → false). Dead `MoeDtypeSnapshot::has_mq6` and
   `MoeFfnMetaView::proj` deleted (snapshot `has_mq6` test assertions
   removed with the method).
2. **O(1) FROZEN BINDING** — Frozen decode no longer calls/materializes
   `routed_expert_refs()` (C2 guarantees the indexed GPU route). New seam
   `routed_expert_refs_for_params`: Frozen → empty slice (dispatch's
   `check_moe_decode_supported` rejects empty refs on the CPU fallback, so
   no fake refs/aliases), Legacy → materializes exactly as before. Frozen
   prefill's entirely unused `routed_experts` Vec removed. Indexed dispatch
   inputs (gate_up/down pointer tables, AWQ table, dtype tags) preserved.
   Call-count seam: `#[cfg(test)] routed_ref_seam` (instrumented counter +
   serialized `SeamGuard`). GREEN: CPU
   `routed_ref_seam_legacy_materializes_once_and_retains_behavior` (exactly
   one resolution per Legacy call; Ok(empty) retained) and GPU-ignored
   `frozen_routed_expert_refs_seam_resolves_zero` (published Frozen resident
   → ZERO resolutions). Existing `check_moe_decode_supported` /
   `cpu-topk-fallback-needs-resident-experts` guard tests retain the
   empty-ref error behavior.
3. **Minor** — `Qwen35BundleBuildError` + `BundleBuildTransaction` are
   `pub(crate)` (no external consumer; `lib.rs` re-exports only
   `Qwen35Bundle`); `MoeResolution::routed_indexable()` now includes the new
   `routed_indexable_e8` field consistently (E8 test
   `moe_res_e8_routed_indexable_consistent_when_admitted` — behavioral RED
   first); dead prefill extraction variables caused by the migration removed
   (22 unused variables including the prefill `routed_experts`). While making
   the publication seam executable, a latent Frozen common-assembly
   index-OOB panic was found and fixed: both MoE layer arms read
   `derived_plans[layer]` unconditionally although Frozen mode never builds
   the plan Vec — the read is now gated to `MoeAssemblyMode::Legacy`.
4. **Verification (2026-08-03, fresh runs)** — qwen35 380 passed / 18
   ignored; dispatch 172 passed / 1 ignored; dispatch-tests 70 passed;
   loader 132 passed / 10 ignored; runtime weight_store 110 passed / 13
   ignored; runtime compile-fail doctests 5 passed;
   `cargo check -p hipfire-loader --all-targets` and
   `cargo check --workspace --all-targets` finish with pre-existing warnings;
   `bash scripts/check_moe_residency_boundary.sh` exit 0; `--self-test` exit
   0 (24 assertion failures, all 16 categories caught); scoped `rustfmt
   --check` on changed files: my changed regions clean (store.rs fully
   canonical; remaining qwen35.rs drift is pre-existing Phase A/B worktree
   drift); `git diff --check` exit 0. GPU (gfx1151, HIP 7.2): all four
   GPU-ignored frozen/legacy tests pass with `--ignored` (parallel and
   serial). Tracker STEP-002 Evidence and this handover updated; deepwork
   record appended. Final-remediation checkpoint:
   `/tmp/opencode/qwen35-final-remediation-20260803.patch` (refreshed for
   the Legacy assembly fix).

## Verification already passed

- `bash scripts/check_moe_residency_boundary.sh` passed on the Phase C tree;
  `--self-test` caught all 16 expected violation categories independently
  (exit 0), and the script fails closed at startup if a required tool is
  missing.
- `bash scripts/check-weight-store-hybrid-boundary.sh` passed (unchanged).
- The tracker records STEP-001 manifest/Step parity, Qwen35 coherence, and
  serve-multiturn evidence as complete; those are prior evidence, not STEP-002
  completion.
- The tracker records the existing manifest, dispatch, model-parallel, and
  failed-ownership-reset work as the current foundation; STEP-002 evidence
  remains `In progress` with the GPU gap explicit.

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
