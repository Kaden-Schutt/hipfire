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
