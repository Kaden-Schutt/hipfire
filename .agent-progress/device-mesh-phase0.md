> **Historical document.** This file preserves dated implementation and validation evidence. Current status and remaining work are tracked only in [device-mesh-refactor-tracker.md](device-mesh-refactor-tracker.md).

# Device-mesh implementation — progress

Plan: docs/superpowers/plans/2026-07-05-device-mesh-transparent-parallelism.md
Branch: feature/device-mesh (off feature/parallel-expansion, has HIPFIRE_EMULATE_GPUS)

## Phase 0 — extraction + collective seam
- [x] 0.1 extract hipfire-hardware leaf crate (multi_gpu → hardware; config→DeviceResolveOpts::from_env; runtime re-exports). ff709bdc. Byte-identical: 31 crates + daemon build, hardware tests 3/3, config tests green.
- [x] 0.2 group:&[usize] param on all_reduce_sum_f32[_peer] (peer sub-group-capable; RCCL full-group-only until 5b ncclCommSplit). f1be1fac. Byte-identical (callers pass 0..n). Daemon builds (10 crates).
- [ ] 0.3 DeviceMesh Dimension tree + group_along + rect() + resolve_mesh — DEFER to land WITH its consumer (Phase 0b executor), per the plan's anti-speculative-scaffolding discipline.
- [x] Phase-0 exit gate: coherence-gate.sh — COHERENCE CLEAN (fluent, no hard errors); exit 1 is the known gfx1100-baseline pflash artifact, not this change. PHASE 0 VALIDATED.

## Next: Phase 0b — unify single-GPU run_layer_program + decode ep.rs into ONE
mesh-driven run_layer_program(mesh,…) in hipfire-dispatch (dispatch→hardware dep).
This is where DeviceMesh gets its first real consumer. Needs the byte-exact
HIPFIRE_FORWARD_LOWERED oracle (single-GPU) + EP-decode byte-identity → GPU.

## Phase 0b — executor unification (in progress)
- [x] crate edge dispatch->hardware + relocate ep.rs into hipfire-dispatch (runtime re-exports). 33731539. Byte-identical, daemon builds 10 crates.
- [x] DeviceMesh (rectangular named-axis core: DimKind{Pp,Tp,Ep}, Axis, rect()/single()/coord_of/device_of/group_along). In hipfire-hardware::mesh. 8 unit tests pass (1x1, Nx1, 1xN, 2x2 coords+groups), no GPU. Tree/raggedness = Phase 5b extension (documented, not built).
- [x] EP executor mesh-driven: run_layer_program_ep takes &DeviceMesh, all-reduce group from group_along(Ep,..). a3cd931c. Byte-identical (1xN group == 0..n, unit-tested).
- [x] resolve_mesh(pp,tp,emulate) -> DeviceMesh producer + tests (config.rs). Replaces flat resolve_parallelism as daemon adopts mesh routing.
- [ ] Unify single-GPU run_layer_program + EP entry into ONE run_layer_program(mesh,..) — param reconciliation (single needs ctx, EP needs partials); single-GPU MoE != EP-1-rank MoE so it is a router. Hot-path → GPU oracle validation. NEXT.

## Validation summary (Phase 0 + 0b foundation)
- Phase 0 crate extraction: coherence-gate CLEAN on GPU (fluent qwen35 matrix, no hard errors).
- Mesh-driven EP executor: `ep_decode_parity` tp=1 ANCHOR PASS — EP argmax stream == production forward_scratch on qwen3.6-35b-a3b (GPU-confirmed byte-identical, HEAD 5f4b581c).
- DeviceMesh + resolve_mesh: 13 unit tests (8 mesh + 5 config), no GPU.
- Every commit builds; daemon chain clean.
=> Foundation is a clean, GPU-validated, self-contained PR unit (Phase 0 + 0b-foundation).

## Remaining (multi-session, each a GPU-gated PR per the plan):
- 0b finish: single-GPU + EP entry-point merge (hot path; needs HIPFIRE_FORWARD_LOWERED oracle).
- Phase 1a: CollectiveHint derived-from-ShardPolicy (needs manifest) + n_rows IR.
- Phase 1b/1c: PP band→BandXfer + PP oracle; llama PP walking skeleton.
- Phase 2: weight/state manifests + mesh-driven fulfill_manifest.
- Phase 3: WeightStore/StateStore + ModelParallel/ArchDispatch (hoist god-struct out of daemon example).
- Phase 4: qwen2 ForwardBindings reach.
- Phase 5/5a/5b: live head-axis TP + DeltaNet head-shard + emulation heterogeneity + ragged/mixed-arch.
- Phase 6: Tier-2 slot-binding (optional). Phase 7: initiate follow-ups (spec-decode/VL/TP tracks).

## DECISION (RETRACTED 2026-07-06): the single-GPU+EP merge IS the target
**Superseded by user directive** — the "keep two mesh-aware executors" compromise
below was a mistake; it drifted from plan §1 ("exactly ONE executor"). All paths
must be merged into one `run_layer_program(mesh, gpus, bindings: &mut [B],
program)`. The original N1 rejection was right about a *bad shape* (a union
params-struct threaded through every hot-path call site) but wrong to conclude
"therefore don't merge." The union struct is avoidable — each objection dissolves:
- **DispatchCtx**: built INSIDE the executor per rank (EP already does this;
  single-GPU builds it for `devices[0]`) → no longer a param.
- **partials / residual_dim**: moved BEHIND `ForwardBindings` (the arch already
  owns its routed-partial scratch for EP) → executor asks the binding, not a param.
- **run_moe vs run_moe_ep**: an internal branch on `mesh.has_axis(Ep)` — no Ep
  axis → `dispatch_super_op → run_moe` (byte-identical single-GPU hot path); Ep
  axis → the zero/run_moe_ep/all-reduce/add path. Exactly plan §1's "per-op branch
  not taken on 1×1".

So the merged signature carries NO EP-specific params; single-GPU passes a
1-element `bindings` slice + `DeviceMesh::single()` + a `Gpus` wrapping its one
`Gpu` (`Gpus::single`, the anticipated "zero runtime cost" Phase-0b refactor).
Byte-identity guarded by the existing `HIPFIRE_FORWARD_LOWERED=0/1` oracle (single)
+ EP decode byte-identity. Sequencing: see the "Executor merge" plan below.

--- ORIGINAL (retained for history; DO NOT act on it) ---
### DECISION: literal single-GPU+EP one-signature merge NOT pursued (0b)
The plan's "ONE run_layer_program(mesh,...)" is served at the module level (both
executors in dispatch, mesh-driven, sharing dispatch_super_op/ForwardBindings/
LayerProgram). A literal single-signature merge would need a union params struct
(single needs DispatchCtx; EP needs partials/residual_dim; single-GPU MoE uses
run_moe while EP uses run_moe_ep+partial+all-reduce) threaded through every
arch's per-token hot-path call site — high ripple, hot-path risk, cosmetic gain.
The plan's OWN history (memory: N1 "unified step contract") REJECTED exactly this
shape as overcomplicated (3 review rounds). Engineering call: keep the two
mesh-aware executors; the mesh is the unification. Revisit only if a concrete
consumer needs the single entry.
