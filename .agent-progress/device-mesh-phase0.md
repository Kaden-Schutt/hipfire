# Device-mesh implementation — progress

Plan: docs/superpowers/plans/2026-07-05-device-mesh-transparent-parallelism.md
Branch: feature/device-mesh (off feature/parallel-expansion, has HIPFIRE_EMULATE_GPUS)

## Phase 0 — extraction + collective seam
- [x] 0.1 extract hipfire-hardware leaf crate (multi_gpu → hardware; config→DeviceResolveOpts::from_env; runtime re-exports). ff709bdc. Byte-identical: 31 crates + daemon build, hardware tests 3/3, config tests green.
- [x] 0.2 group:&[usize] param on all_reduce_sum_f32[_peer] (peer sub-group-capable; RCCL full-group-only until 5b ncclCommSplit). f1be1fac. Byte-identical (callers pass 0..n). Daemon builds (10 crates).
- [ ] 0.3 DeviceMesh Dimension tree + group_along + rect() + resolve_mesh — DEFER to land WITH its consumer (Phase 0b executor), per the plan's anti-speculative-scaffolding discipline.
- [ ] Phase-0 exit gate: coherence-gate.sh (proves forward-pass byte-unaffected by the extraction). RUNNING.

## Next: Phase 0b — unify single-GPU run_layer_program + decode ep.rs into ONE
mesh-driven run_layer_program(mesh,…) in hipfire-dispatch (dispatch→hardware dep).
This is where DeviceMesh gets its first real consumer. Needs the byte-exact
HIPFIRE_FORWARD_LOWERED oracle (single-GPU) + EP-decode byte-identity → GPU.
