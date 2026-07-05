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
- Phase 2 cont: fulfill_manifest DENSE-TP slice+upload (whole-tensor + ExpertSharded DONE, see below); production-arch manifests for qwen2/qwen35/ds4/minimax; state_manifest impls; transactional-OOM guard.
- Phase 1a/1b: wire collective hints + band_xfer into the executor; PP executor loop; PP byte-exact oracle (needs building); 1c llama-PP walking skeleton.
- Phase 3: WeightStore/StateStore + ModelParallel + ArchDispatch (hoist EpArch/LoadedModel out of daemon example binary → runtime lib). Highest-risk.
- Phase 4: qwen2 ForwardBindings reach.
- Phase 5/5a/5b: live head-axis TP + DeltaNet head-shard + s_ef_residual; emulation heterogeneity (HIPFIRE_EMULATE_ARCHS/_VRAM); ragged/mixed-arch per-arch LoweredForward.
- Phase 6: Tier-2 slot-binding (optional). Phase 7: initiate spec-decode/VL/TP follow-up tracks.

### Phase 2 — fulfill_manifest (GPU execution, whole-tensor path) — DONE + GPU-validated
- `crates/hipfire-runtime/src/weight_store.rs`: `WeightStore` (keyed `(name, layer, device)`),
  `WeightHandle{Resident(GpuTensor)|Alias(String)}`, `FulfillError`, and
  `fulfill_manifest(weights, mesh, n_layers, gpus, source) -> Result<WeightStore, FulfillError>`.
  Additive — does NOT touch the forward/hot path (Tier-1; forward-read-from-store is Phase 3).
- Scope: whole-tensor upload (single + all PP + Replicate/Pin + group-size-1 degenerate) via
  `Gpu::upload_raw`, Tied→Alias; **ExpertSharded on Ep>1** = each rank gets a compact blob of its
  owned experts (generic expert-outermost host gather; `expert_compact_blob` + `ShardConfig`;
  the arch's forward owns the per-expert ptr-table + zeroed-dummy — that's forward-indexing, not
  placement); **dense TP slice (Column/Row/FusedQkv/Head/Vocab @ Tp>1) returns `Err`** (Phase 5).
- DECISION: takes a `source(entry) -> raw bytes` closure, NOT `&HfqFile` — manifest names are
  *logical* ("wq"), on-disk HFQ names are arch-specific; the closure keeps the engine free of
  on-disk naming (pulls complexity to the arch; Tier-1). Same shape as the plan's
  `fulfill_manifest(manifest, hfq, mesh)` with the name-resolution seam made explicit.
- GPU-validated on gfx1151 via `examples/fulfill_manifest_probe.rs` (synthetic byte source,
  no model file): single-1×1 + emulated PP-2 + emulated EP-2 all PASS — placement matches
  `placement_devices`, byte-oracle readback (`memcpy_dtoh`) == uploaded bytes on every device,
  Tied→Alias, dense-TP refusal, EP compact-blob per rank (rank0 experts [0,2,4,6], rank1 [1,3,5,7]).
  **The byte-oracle caught a real bug**: `(name, device)` key aliased all layers' `wq` onto one
  cell → fixed by adding `layer` to the key.
- 4 no-GPU unit tests (classifier + expert_compact_blob + store keying + refusal decision).

## Validation done
- coherence-gate.sh CLEAN on qwen35 matrix (Phase 0). 
- fulfill_manifest_probe: single-1×1 + PP-2 + EP-2 emulated PASS on gfx1151 (placement + byte-oracle).
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
