# Device-mesh implementation — consolidated status

Branch: feature/device-mesh (off feature/parallel-expansion; has HIPFIRE_EMULATE_GPUS)
Plan: docs/superpowers/plans/2026-07-05-device-mesh-transparent-parallelism.md

## DONE + tested (17 commits, all build green; 2 GPU-validated)

### Phase 0 — COMPLETE, coherence-gate GPU-validated
- ff709bdc hipfire-hardware leaf crate (multi_gpu → hardware; config → DeviceResolveOpts::from_env; runtime re-exports). Breaks the dispatch→runtime cycle.
- f1be1fac group:&[usize] param on all_reduce_sum_f32[_peer] (peer sub-group-capable; RCCL full-group until 5b).

### Phase 0b — foundation done; EP mesh-driven GPU-validated (ep_decode_parity anchor PASS, 35B-A3B)
- 33731539 relocate EP executor ep.rs → hipfire-dispatch + dispatch→hardware edge.
- 0b95b89c DeviceMesh named-axis type (coord_of/device_of/group_along/n_devices/…). 
- a3cd931c EP all-reduce group from mesh.group_along(Ep,..). byte-identical (1×N == 0..n).
- 5f4b581c resolve_mesh(pp,tp,emulate) → DeviceMesh producer.
- e66d6f94 stage_for_layer + band_xfer_after (PP side of the mesh).
- DECISION (610f6f0e): literal single+EP one-signature merge NOT pursued — N1-rejected shape; both executors mesh-aware in dispatch is the unification.

### Phase 1a — collective-hint-from-policy (mini-partitioner)
- e565d110 CollectiveHint + collective_for_policy(&ShardPolicy).
- 69c61c05 layer_collectives(manifest) — per-layer all-reduce schedule.

### Phase 2 — manifest system + placement core + first arches
- a6a0acb9 WeightEntry/ShardPolicy/StateEntry/FusedQkvLayout/PinTarget + Architecture::{weight_manifest,state_manifest} seam.
- 41b63cdb placement_devices (placement = manifest × mesh, pure).
- 10f726c7 toy weight_manifest reference impl.
- 59eebbec llama weight_manifest (first PRODUCTION arch; real GQA FusedQkv).

Total unit tests added: ~20 (10 mesh + 7 manifest + config + toy). No GPU needed for any.

## REMAINING (GPU-integration / hot-path; multi-session, one PR per phase)
- Phase 2 cont: fulfill_manifest GPU slice+upload (the "how"); production-arch manifests for qwen2/qwen35/ds4/minimax; state_manifest impls; transactional-OOM guard.
- Phase 1a/1b: wire collective hints + band_xfer into the executor; PP executor loop; PP byte-exact oracle (needs building); 1c llama-PP walking skeleton.
- Phase 3: WeightStore/StateStore + ModelParallel + ArchDispatch (hoist EpArch/LoadedModel out of daemon example binary → runtime lib). Highest-risk.
- Phase 4: qwen2 ForwardBindings reach.
- Phase 5/5a/5b: live head-axis TP + DeltaNet head-shard + s_ef_residual; emulation heterogeneity (HIPFIRE_EMULATE_ARCHS/_VRAM); ragged/mixed-arch per-arch LoweredForward.
- Phase 6: Tier-2 slot-binding (optional). Phase 7: initiate spec-decode/VL/TP follow-up tracks.

## Validation done
- coherence-gate.sh CLEAN on qwen35 matrix (Phase 0). 
- ep_decode_parity tp=1 ANCHOR PASS (mesh-driven EP == production, 35B-A3B).
- All commits build; per-file rustfmt only (never qwen35/ds4/minimax/daemon fmt-debt files).

## Capstone regression validation (HEAD 49aef4df, 23 commits)
- `cargo build --workspace --features hipfire-runtime/deltanet`: 0 errors (whole engine).
- No-GPU lib tests across hipfire-hardware/runtime/toy/llama/qwen2/minimax: 0 failures
  (217 in hipfire-runtime alone + arch crates). No regression from the foundation.
- GPU-validated: coherence-gate (Phase 0) + ep_decode_parity anchor (mesh EP).
=> Foundation is production-safe + landable as one "Phase 0 + 0b-foundation + Phase-1a/2
   pure-logic + §6 validate" PR. GPU-integration phases (fulfill upload, executor wiring,
   ModelParallel hoist, TP kernels, ragged) are the mapped multi-session remainder.
