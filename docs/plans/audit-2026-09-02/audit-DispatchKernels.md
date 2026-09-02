<!-- SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Kaden Schutt; hipfire — see LICENSE and NOTICE in the project root. -->

# Audit: DispatchKernels

## Broken

### 1. MQ8 family path: rotate does not produce INT8 scratch; prerotated unwraps (VERIFIED)

**Citations:** `crates/hipfire-dispatch/src/types.rs:130` (MQ8G256 => RotationPlan::Mq8Internal); `crates/hipfire-dispatch/src/families/gemv.rs:143-148` (Mq8Internal 	o WithRmsnorm or Plain); `crates/hipfire-dispatch/src/families/rotation.rs:62-199` (no MQ8 arm; Plain 	o rotate_x_mq F32 FWHT); `crates/hipfire-dispatch/src/families/gemv.rs:246-260` (run_input Raw 	o rotate then post-rotation Prerotated); `crates/hipfire-dispatch/src/families/gemv.rs:512` (gemv_mq8g256_prerotated(w.buf, y, m, k) — no x); `crates/rdna-compute/src/gemv.rs:6151-6164` (docs require prior rotate_quantize_x_mq8; mq_x_q8.as_ref().unwrap()).

**How known:** Cross-read rotation plan 	o select_rotation_variant 	o RotationFamily::run 	o launch. Pipeline path is fixed: `crates/hipfire-dispatch/src/pipeline/steps.rs:940-948` special-cases Mq8Internal to rmsnorm_f32 + rotate_quantize_x_mq8. Correct low-level API is gemv_mq8g256_with_rotate (gemv.rs:6178-6204). Family run_auto/rotate never calls it.

**Effect:** GemvFamily::run_auto / RotInput::Raw on MQ8 panics on unwrap. Tests leave RotationPlan::Mq8Internal empty (tests.rs:770).

### 2. Plain KernelKey::GemvMq8G256 (and other plain MQ keys) missing from launch (VERIFIED)

**Citations:** types.rs:703 maps (MQ8G256, Plain) => GemvMq8G256; gemv_table.rs registers MQ8 plain+prerotated; gemv.rs:537 other => MissingImpl.

**How known:** Exhaustive match inspection — only GemvMq8G256Prerotated handled. run_auto uses post-rotation Prerotated so hits bug #1; direct Plain 	o MissingImpl.

### 3. gemm_table V2 comment contradicts registrations and sources (VERIFIED contradiction)

**Citations:** gemm_table.rs:364-370 claims MQ6/5/3/2V2 and MQ4CG256 remain gfx12-only while registering GemmMq{6,5,4,3,2}G256V2* as ArchPredicate::HasWmma (368-482). MQ4C correctly uses HasWmmaGfx12 (489-506). kernels.rs:3124-3141 includes gfx11 residual WMMA sources for MQ4/5/6/3/2 V2. Runtime gemm_mq6g256v2_residual_wmma branches gfx12 then gfx11 (gemm.rs:32190-32210).

**How known:** Comment vs register vs include_str! vs residual dispatcher. Not a silent wrong route on WMMA GPUs; documentation/registry intent drift.

### 4. MQ6 residual module name suffix _mq5v2 (VERIFIED smell)

**Citations:** gemm.rs:32097 format!("{}_mq5v2", kname) inside gemm_mq6g256v2_residual_wmma_gfx12; same pattern at 31884, 33308, 34558. Contrast QKV MQ6: module_v2 = "…_mq6v2" (31639).

**How known:** Grep _mq5v2. Functional collision avoided because kname embeds mq6 vs mq5. Copy-paste defect.

### 5. verify-bind-thread.sh does not cover most impl Gpu surfaces (VERIFIED)

**Citations:** Script default FILE=crates/rdna-compute/src/dispatch.rs (line 21); finds single impl Gpu { … impl Drop for Gpu (40-45). Actual impl Gpu also in gemv.rs:171, gemm.rs:168, attention.rs:164, moe.rs:13, norm.rs:90, embedding.rs:13, sampling.rs:95, gemma4_ext.rs, etc.

**How known:** Grep ^impl Gpu + script read. Multi-GPU half only flags first call after `let <name> = &mut gpus.devices[X]` if it is `<name>.hip.` — misses other patterns.

---

## Missing

### 1. RotationVariant / RotationFamily support for MQ8 INT8 quantize (VERIFIED gap)
Root of broken #1. Only pipeline special-case implements the contract documented on rotate_quantize_x_mq8.

### 2. Automated bind_thread proof for gemv/gemm/attention/moe (VERIFIED gap)
Spot checks show many self.bind_thread()? first lines in gemv.rs; script cannot fail CI on regressions outside dispatch.rs.

### 3. DeviceBuffer RAII Drop (VERIFIED design gap)
hip-bridge malloc returns owning buffer; free is explicit (ffi.rs:911-924); no Drop for DeviceBuffer. SAFETY docs on from_raw/alias/from_vmm_owner present (lib.rs:114-147). Leak-on-forget, not double-free for borrowed.

### 4. hsa-bridge per-site SAFETY comments (VERIFIED gap)
hsa-bridge/src/lib.rs dense unsafe for init, agents, queues, signals without // SAFETY: blocks. Experimental Phase-2 surface.

### 5. MoE shape_gate: BatchGt(1) not enforced by resolve (documented intentional)
moe_table.rs:12-20,40-45. Prefill executor dtype/env dispatch bypasses registry.

### 6. CK feature optional soft-fail at load; try_* fail-closed (mostly OK)
**Citations:** feature flash-attn-ck (rdna-compute lib.rs:14-15); load soft-fail WARN (dispatch.rs:1157-1172); reject matrix Decode/SmallQuery/NonCausal/Graph/Replay/Tree/Window/Block/CapabilityMiss (flash_attn_ck.rs:101-116,131+); attention family cfg + HIPFIRE_FLASH_PREFILL force-off (attention.rs:743-755). try_* returns Ok(false) 	o native path. **Verified fail-closed at launch selection**; load failure soft by design.

### 7. Fused MQ V2 table Always vs GEMM WMMA gates (coverage clarity missing)
fused_qkv_table.rs:17-22 Always for FusedQkvMq*V2 and MQ4C. Fused impls load scalar HIP (fused_qkv_mq6g256v2 at gemm.rs:32323+). GEMM MQ4C Err without gfx12 (35461-35468).

---

## Would change (ranked)

1. **Wire MQ8 through family rotation + launch plain arm** — hours  
   Add RotationVariant::Mq8Quantize 	o gpu.rotate_quantize_x_mq8; fix select_rotation_variant; map plain key to gemv_mq8g256_with_rotate or delete plain registration; GPU-free test that rotate fills scratch; regression against steps.rs Mq8 path.

2. **Extend verify-bind-thread.sh to all impl Gpu modules** — hours  
   Plus broader multi-GPU patterns. Highest leverage against multi-device silent wrong-device allocs.

3. **Fix gemm_table V2 comment + _mq5v2 suffixes** — hours  
   Cheap correctness-of-docs and module-cache clarity.

4. **Fused V2 Always contract test / or HasWmma gate** — hours  
   Prevent admit-then-JIT-fail if fused sources are arch-limited.

5. **DeviceBuffer Drop or must_use discipline** — days  
   Coordinate VMM owners; reduce leak class.

6. **CK first-miss diagnostics** — hours  
   Keep soft-fail load; make capability/workspace misses obvious once.

7. **hsa-bridge SAFETY comment pass** — hours  
   Low urgency until HSA path is production-hot.

---

## Confidence

**Did:** Core dispatch tables (gemm/gemv/moe/fused_qkv), GemvFamily+RotationFamily+pipeline MQ8 path, gemm MQ V2 residual/plain/gfx11/gfx12, kernels.rs V2 includes, CK load+select+attention cfg, bind_thread script, hip-bridge DeviceBuffer ownership, hsa-bridge unsafe survey, fused MQ4C Always vs gemm gfx12.

**Did not fully:** Exhaustive (op	imes qt	imes arch) matrix for every KernelKey; dead kernels/src/*.hip vs include_str! full set-diff; every impl Gpu bind_thread first-stmt proof; WMMA gfx11 vs gfx12 numerical drift inside HIP sources; open GitHub issue cross-check via gh (unavailable this turn — novelty not claimed against issue tracker); live GPU/JIT validation.

**Suspicion (not verified broken):** gate_up HFQ6/MQ6 fusion requiring dp4a_eligible may leave non-dp4a arches on slower unfused path only — appears intentional. MQ6 GEMV on gfx906 uses HFQ6 generic path (gemv_mq6g256_prerotated 	o gemv_hfq6g256) — not proven wrong without arch kernel internals.

**Architecture note:** hipfire-dispatch KernelRegistry+families resolve (op,dtype,arch) 	o rdna-compute Gpu methods (impl Gpu split across gemv/gemm/attention/moe/…) JIT via kernels.rs include_str! 	o hip-bridge launch; optional flash-attn-ck sidecar under feature; bind_thread intended on every device-touching Gpu entry; rotation plans (FwhtG256/Mq8Internal/Givens) sit above GEMV prerotated kernels.
