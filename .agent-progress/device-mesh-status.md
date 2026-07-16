> **Historical document.** This file preserves dated implementation and validation evidence. Current status and remaining work are tracked only in [device-mesh-refactor-tracker.md](device-mesh-refactor-tracker.md).

> **Current synchronization (2026-07-16):** COR-003 is complete. The
> implementation covers terminal `StopQuarantine`/`EosFilter`
> behavior, `StreamParser` finalization, Cohere recovery, generic AR/spec
> normal-versus-discard handling, sealed Qwen speculative-turn authority and
> cache/reset behavior, Qwen PP sealed-boundary/reset behavior, DeepSeek AR
> discard reset/cache zeroing, and native Qwen/DeepSeek/DSpark MTP in-flight
> cancellation with production-owned lifecycle tests. The full CPU
> `nix develop --command cargo test --workspace --locked` suite passed (GPU
> tests ignored as applicable); `nix develop --command
> ./scripts/coherence-gate-dflash.sh` passed with no hard or soft warnings
> (report `/tmp/coherence-dflash-20260716-110721.md`);
> `nix develop --command ./scripts/serve-multiturn-gate.sh` passed (report
> `/tmp/serve-multiturn-20260716-110919.md`); and `git diff --check` passed.
> Native Qwen MTP cancellation is done; transactional target loading remains
> deferred to `SPEC-003`. Remaining architecture migrations and separate
> physical PP/TP/EP hardware tasks remain open. See the canonical tracker for
> current task status and the mandatory terminal lifecycle migration matrix.

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
- DECISION RETRACTED (2026-07-06, user directive): the single+EP merge IS the target — the "keep two executors" compromise (610f6f0e) was a mistake. ONE `run_layer_program(mesh, gpus, bindings: &mut [B], program)`; the N1 union-struct is dissolved (ctx built inside; partials/residual_dim behind ForwardBindings; run_moe vs run_moe_ep = internal `mesh.has_axis(Ep)` branch). See phase0.md "Executor merge" + plan §1.

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
- Phase 2 cont: fulfill_manifest DENSE-TP slice+upload (whole-tensor + ExpertSharded + transactional-OOM guard DONE, see below); qwen35 state_manifest DONE (Kv+Recurrent+Conv by layer_type); deepseek4 weight+state manifest DONE (MLA + EP MoE, ExpertSharded validated on its origin arch); remaining Phase-2 arch = qwen35 WEIGHT_manifest (DeltaNet fused projections + MoE variants — loader study).
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
- **Transactional (§6):** outer/inner split — on any mid-load failure (source-read/shard-math/
  upload), `WeightStore::free_all` frees every already-uploaded buffer on its own device (best-
  effort via `Gpu::hip.free`) and returns `Err`. Never a half-loaded VRAM-leaking mesh (unlike the
  bespoke loaders). `Alias` cells hold no buffer.
- DECISION: takes a `source(entry) -> (raw bytes, DType)` closure, NOT `&HfqFile` — manifest names
  are *logical* ("wq"), on-disk HFQ names are arch-specific; the closure keeps the engine free of
  on-disk naming (pulls complexity to the arch; Tier-1). The dtype is the tensor's REAL on-disk quant
  type, stamped onto the store tensor so it is forward-ready (correct kernel dispatch), not `Raw`.

### Phase 3 START — store-backed REAL llama load, byte+dtype-identical (GPU-validated)
- `examples/llama_store_load.rs`: loads `qwen3-0.6b-llama.mq4`'s quantized projections
  (wq/wk/wv/wo/ffn_gate/ffn_up/ffn_down) through generic `fulfill_manifest` + a llama HFQ-backed
  `source` (HF names `model.layers.{i}.self_attn.q_proj.weight`; quant_type→DType) and asserts each
  store tensor is byte+dtype IDENTICAL to bespoke `Llama::load_weights`.
- GPU-validated gfx1151: **196 projection tensors / 28 layers, byte+dtype identical (MQ4G256)**.
  These are uploaded raw/verbatim by the loader (no transform) → raw-byte fulfill matches exactly.
- Scoped out (byte-validation): norms/embed/tied-lm_head (F16→F32 host dequant) — source-side follow-up.
- **WHOLE-MODEL store→forward done + GPU-validated (bit-exact):** llama `weight_manifest` gained
  q_norm/k_norm (qwen3 per-head RMSNorm, gated on `has_qk_norm` — manifest was incomplete). A
  universal `source` mirrors the HFQ loader rule (quant_type 1→F16→F32, 2→F32, else raw+real dtype),
  covering norms/embed/lm_head(+tied)/projections. `llama_store_load` fulfills the FULL manifest,
  assembles a complete `LlamaWeights` from the store (embd_format derived from dtype), and runs a
  forward: **311 tensors, logits IDENTICAL to bespoke (max |Δ|=0, same argmax) on gfx1151**. The
  generic manifest path is a drop-in, bit-exact replacement for the bespoke llama loader (single-GPU).
- **PP-2 store PLACEMENT done + GPU-validated:** llama `output_norm` Replicate→Pin(Output) fix
  (final norm must co-locate with lm_head on the last stage). `llama_store_pp`: `HIPFIRE_EMULATE_GPUS=2`
  → PP-2 mesh; fulfill the WHOLE manifest across 2 stages; assert every tensor on its mesh-correct
  stage (`placement_devices` == `store.devices_for`); gather + forward. **311 tensors banded 155/156
  (embed→0, output_norm+lm_head→1, 28 layers by `stage_for_layer`); gathered forward logit-IDENTICAL
  to bespoke (max |Δ|=0), gfx1151.** Mesh-driven PP placement correct + forward-usable.
- **PP-2 banded EXECUTION done + GPU-validated (Phase 1c):** refactored llama's monolithic forward —
  extracted `forward_scratch_band(gpu, w, cfg, layer_range, pos, kv, scratch)` (range-parameterized
  layer loop) + `forward_scratch_head` (final norm + lm_head); `forward_scratch_compute` = `band(0..n)`
  + `head` (bit-exact — stable argmax across the refactor). `llama_store_pp` now runs a REAL banded
  PP forward: **stage 0 embed+band(0..14) on dev0 → `boundary_copy` residual → stage 1 band(14..28)+
  head on dev1, logit-IDENTICAL to bespoke (max |Δ|=0)**. Full pipeline-parallel LOAD + EXECUTE on a
  real model, mesh-driven. (Emulated 2-GPU; each band touches only its stage's layers on its device.)
  Next: `ModelParallel`/`ArchDispatch` daemon hoist (serve-reach); real 2-GPU (hiptrx/hipx).
- GPU-validated on gfx1151 via `examples/fulfill_manifest_probe.rs` (synthetic byte source,
  no model file): single-1×1 + emulated PP-2 + emulated EP-2 all PASS — placement matches
  `placement_devices`, byte-oracle readback (`memcpy_dtoh`) == uploaded bytes on every device,
  Tied→Alias, dense-TP refusal, EP compact-blob per rank (rank0 experts [0,2,4,6], rank1 [1,3,5,7]),
  transactional rollback (source-fail mid-load → Err + partial uploads freed).
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
