// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
use crate::context::DispatchCtx;
use crate::families::gemv::{GemvFamily, WeightRef};
use crate::families::moe::{checked_deepseek_grouped_bounds, MOE_GROUPED_BLOCK_M};
use crate::tables::KernelRegistry;
use crate::types::*;
#[allow(unused_imports)]
use hip_bridge;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::OnceLock;

pub(crate) mod steps;
#[cfg(feature = "deltanet")]
pub use steps::{
    build_delta_net_batch_steps, build_delta_net_decode_steps, build_delta_net_tree_steps,
    DeltaNetOperandDescriptor, DeltaRecurrenceParams,
};
pub use steps::{
    execute_steps, execute_steps_mesh, execute_steps_parallel, execute_steps_tp, FusedPattern,
    GemvInput, MoeActivationVariant, MoeProj, QwenDownMode, ScoreActKind, Step, StepCollective,
    TpCollective,
};

pub struct Pipeline {
    pub ops: &'static [PipelineOp],
}

impl Pipeline {
    pub fn new(ops: &'static [PipelineOp]) -> Self {
        Self { ops }
    }

    pub fn can_satisfy(&self, requested: &[PipelineOp]) -> bool {
        if self.ops.len() > requested.len() {
            return false;
        }
        self.ops.iter().zip(requested.iter()).all(|(a, b)| a == b)
    }
}

pub struct LinearParams<'a> {
    pub x: &'a GpuTensor,
    pub y: &'a GpuTensor,
    pub buf: &'a GpuTensor,
    pub m: usize,
    pub k: usize,
}

// Keep the public enum's direct `MoeParams` variant; boxing it would break
// existing callers constructing `PipelineParams::Moe`.
#[allow(clippy::large_enum_variant)]
pub enum PipelineParams<'a> {
    Linear(LinearParams<'a>),
    Moe(crate::families::moe::MoeParams<'a>),
}

pub fn execute_pipeline(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    steps: &[PipelineOp],
    params: &PipelineParams,
    dtype: rdna_compute::DType,
    registry: &KernelRegistry,
) -> Result<(), DispatchError> {
    if let PipelineParams::Moe(p) = params {
        return run_moe_decode(ctx, gpu, p);
    }
    if let Some(key) = find_fused(registry, ctx, dtype, steps) {
        return dispatch_fused(ctx, gpu, key, params);
    }
    let params = match params {
        PipelineParams::Linear(p) => p,
        PipelineParams::Moe(_) => unreachable!(),
    };
    for &step in steps {
        match step {
            PipelineOp::RotateFwht => {
                use crate::families::rotation::{RotationFamily, RotationParams};
                let rot = RotationFamily::new();
                gpu.ensure_mq_signs()
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                let x_rot = unsafe {
                    GpuTensor {
                        buf: gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias(),
                        shape: vec![params.k],
                        dtype: rdna_compute::DType::F32,
                    }
                };
                rot.run(
                    ctx,
                    gpu,
                    RotationParams {
                        x: params.x,
                        x_up: None,
                        w_norm: None,
                        x_plain: &x_rot,
                        x_rot: &x_rot,
                        awq_scale: None,
                        k: params.k,
                        eps: 1e-6,
                        batch_size: 1,
                        variant: RotationVariant::Plain,
                        givens_pairs: None,
                        givens_theta: None,
                        givens_scales: None,
                        givens_krot: None,
                    },
                )
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            }
            PipelineOp::Gemv => {
                static GEMV_PIPELINE: OnceLock<GemvFamily> = OnceLock::new();
                let gemv = GEMV_PIPELINE.get_or_init(GemvFamily::new);
                let w = WeightRef {
                    buf: params.buf,
                    dtype,
                    m: params.m,
                    k: params.k,
                    row_stride: params.k,
                    rotation: None,
                    awq_scale: None,
                };
                gemv.run_auto(ctx, gpu, &w, params.x, params.y)?;
            }
            _ => {
                return Err(DispatchError::UnsupportedVariant {
                    family: "pipeline",
                    variant: "step",
                    arch: "",
                    quant: "",
                });
            }
        }
    }
    Ok(())
}

fn find_fused(
    registry: &KernelRegistry,
    ctx: &DispatchCtx,
    dtype: rdna_compute::DType,
    requested: &[PipelineOp],
) -> Option<KernelKey> {
    use rdna_compute::DType;
    if dtype == DType::MFP4G32
        && requested.len() == 2
        && requested[0] == PipelineOp::RotateFwht
        && requested[1] == PipelineOp::Gemv
    {
        let key = KernelKey::GemvMfp4G32Fused;
        if registry.resolve(key, ctx, None).is_ok() {
            return Some(key);
        }
    }
    None
}

/// Slice a subrange of a flat F32 GpuTensor by element offset + length.
/// Mirrors qwen35::slice_f32_view — unsafe because it aliases device memory.
unsafe fn slice_moe_f32_view(src: &GpuTensor, offset_elems: usize, len_elems: usize) -> GpuTensor {
    let base = src.buf.as_ptr() as *mut u8;
    let ptr = base.add(offset_elems * 4);
    GpuTensor {
        buf: hip_bridge::DeviceBuffer::from_raw(ptr as *mut _, len_elems * 4),
        shape: vec![len_elems],
        dtype: DType::F32,
    }
}

/// GPU-free unit for the runtime decode batch-size guard (CB5).
/// Extracted so the guard is testable without a GPU or `MoeParams`.
pub fn check_moe_decode_batch_size(batch_size: usize) -> Result<(), DispatchError> {
    if batch_size != 1 {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "decode-requires-batch-1",
            arch: "",
            quant: "",
        });
    }
    Ok(())
}

/// GPU-free pre-guard for MoE decode (#397 Ship 4c). Rejects the two
/// truly-unsupported cases up front — *before* any GPU work — so the caller
/// gets a clean [`DispatchError`] instead of a deep panic in the CPU-top-K
/// fallback (`select_nth_unstable_by(k-1)` panics when `k == 0 || k > n_exp`)
/// or in a kernel launch with no expert to run.
///
/// IMPORTANT: `k != 8` is NOT itself an error. The CPU-top-K fallback
/// (`run_moe_decode_cpu_fallback`) legitimately handles any `k ∈ [1, n_exp]`
/// (k=4 for MQ4, k=2 for an F32 router, etc.). This guard must only reject:
///
/// - **(a)** `k` outside `[1, n_exp]` — invalid for top-K selection on either
///   the GPU-top-K fast path or the CPU fallback.
/// - **(b)** a routed dtype that neither path supports: the dtype is not on the
///   GPU-top-K fast path (`!use_gpu_topk`) *and* there are no resident per-expert
///   weights for the CPU fallback to iterate. (When the routed dtype is the only
///   issue but experts are resident, the fallback runs it and its inner
///   `gemv.run_auto` surfaces any genuinely-unsupported dtype as its own clean
///   `DispatchError` — so we must NOT reject that case here.)
///
/// `routed_experts_resident` mirrors `!MoeParams::routed_experts.is_empty()`
/// (false under paged residency, where only the GPU-top-K path is available).
pub fn check_moe_decode_supported(
    use_gpu_topk: bool,
    k: usize,
    n_exp: usize,
    routed_experts_resident: bool,
) -> Result<(), DispatchError> {
    // (a) k-range — required by BOTH the GPU-top-K path and the CPU fallback's
    // `select_nth_unstable_by(k-1)`. Universal precondition, not a k==8 check.
    if k == 0 || k > n_exp {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "decode-k-out-of-range",
            arch: "",
            quant: "",
        });
    }
    // (b) routed dtype on neither path: not GPU-top-K-indexable AND no resident
    // experts to drive the CPU fallback. A non-fast-path dtype WITH resident
    // experts is a valid fallback case (do not reject it here).
    if !use_gpu_topk && !routed_experts_resident {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "decode-routed-dtype-unsupported-no-fallback",
            arch: "",
            quant: "",
        });
    }
    Ok(())
}

/// MoE decode executor. Ports the body of `moe_ffn_decode_impl` verbatim,
/// substituting `ffn.*`/`config.*`/`s.*` references with `MoeParams` fields.
/// Resolution is owned here (computed from `MoeDtypes` + k), and `ctx` is
/// threaded to every inner GEMV so the call site builds one `DispatchCtx`.
pub fn run_moe_decode(
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeParams,
) -> Result<(), DispatchError> {
    use crate::families::moe::{build_moe_decode_steps, decode_expert_refs, MoeResolution};
    macro_rules! hip {
        ($e:expr) => {
            $e.map_err(|e| DispatchError::Hip(e.to_string()))
        };
    }

    // Runtime guard matching the bias-aware decode guard (not debug_assert —
    // that would be stripped in release). batch_size=1 is the only valid
    // decode width; >1 must route to grouped prefill (Step 8).
    check_moe_decode_batch_size(p.batch_size)?;

    // gfx11 E8 port: widen E8 GPU-topK admission to the whole RDNA3 wave32-WMMA family
    // (has_wmma_w32 == is_rdna3, excludes CDNA). gfx1100 (dGPU) shares the scalar-E8
    // indexed-GEMV ISA with gfx1151; routing it onto use_gpu_topk removes the
    // host-side router-logits D2H that crashes hipGraph capture on the dGPU.
    // gfx12 (RDNA4) port: widen further to has_wmma() (is_rdna3 || is_rdna4) so that
    // gfx1200/gfx1201 also get use_gpu_topk + needs_x_rot_local for E8 experts.
    // arch_has_e8_wmma ONLY gates routed_indexable_e8 in resolve_arch — no other dtype
    // path is affected, so widening here is safe for all non-E8 models.
    let res = MoeResolution::resolve_arch(&p.dtypes, p.k, ctx.arch.has_wmma());

    // Pre-guard (#397 Ship 4c): reject out-of-range k and routed dtypes that
    // neither the GPU-top-K fast path nor the CPU fallback can run, BEFORE any
    // GPU work. `resolve` is a pure, side-effect-free function of dtypes + k, so
    // running it first then guarding is equivalent to guarding pre-resolve while
    // letting us key the dtype check off `res.use_gpu_topk`. This turns the
    // deep `select_nth_unstable_by` panic in the fallback into a clean error.
    // NOTE: k != 8 is intentionally NOT rejected — the fallback handles k ∈
    // [1, n_exp] (MQ4 k=4, F32 k=2, …).
    check_moe_decode_supported(res.use_gpu_topk, p.k, p.n_exp, !p.routed_experts.is_empty())?;

    // ── Build selection (binding contract) ─────────────────────────────────
    // The builder returns the discriminated build: `CpuFallback` exactly when
    // `!use_gpu_topk` (k != 8 OR routed dtype not indexable), `Gpu` exactly
    // when the GPU top-K path is admissible. The gate-side projection runs
    // BEFORE the branch on both paths (the fallback consumes the shared
    // gate/up outputs); on the fallback path it runs directly through the
    // extracted helpers — never as a Step program.
    let (gu_experts, dn_experts) = decode_expert_refs(p);
    match build_moe_decode_steps(p, &res, &gu_experts, &dn_experts)? {
        crate::families::moe::MoeStepBuild::CpuFallback => {
            let x_rot_local = launch_decode_rotate(gpu, p, &res)?;
            launch_gate_side(ctx, gpu, p, &res, x_rot_local)?;
            return run_moe_decode_cpu_fallback(ctx, gpu, p);
        }
        crate::families::moe::MoeStepBuild::Gpu(phases) => {
            // Step-native GPU top-K path: one Step program per layer, in the
            // exact legacy launch order (rotate → gate side → softmax+top-K →
            // shared down → routed block). `routed_out`/`skip_shared`
            // semantics are preserved: the program accumulates the routed
            // (and, unless skipped, shared) contributions into the EP
            // partial, and `ep_partial` marks the reduce target for the mesh
            // executor. The HIPFIRE_DUMP_HIDDEN diagnostic reads router
            // logits after the gate-side and before softmax, so the program
            // runs in two segments.
            let split = usize::from(phases.rotate.is_some()) + phases.gate_side.len();
            let build = phases.into_build();

            // Signs back the FWHT used by every MQ4/MQ6 gate_up rotation +
            // silu-rotate (idempotent/cached). Only the paro path is
            // sign-free. Position mirrors the legacy executor's first explicit
            // ensure_mq_signs (before the rotate); the MQ4 shared-down step
            // re-ensures at its legacy position.
            if !res.routed_indexable_paro {
                hip!(gpu.ensure_mq_signs())?;
            }
            let mesh = &hipfire_hardware::DeviceMesh::single();
            execute_steps_mesh(mesh, gpu, ctx, &build.steps[..split])?;

            // DIAG: dump router logits before softmax (mirrors qwen35
            // HIPFIRE_DUMP_HIDDEN)
            if let Ok(dump_path) = std::env::var("HIPFIRE_DUMP_HIDDEN") {
                if gpu.hip.device_synchronize().is_ok() {
                    if let Ok(all) = gpu.download_f32(p.router_logits) {
                        use std::io::Write;
                        let path = format!("{dump_path}.router_raw_p");
                        if let Ok(mut f) = std::fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(&path)
                        {
                            let _ = f.write_all(&(0u32).to_le_bytes());
                            for v in &all[..all.len().min(p.n_exp * 4 / 4)] {
                                let _ = f.write_all(&v.to_le_bytes());
                            }
                        }
                    }
                }
            }
            execute_steps_mesh(mesh, gpu, ctx, &build.steps[split..])?;
        }
    }
    Ok(())
}

/// Run the decode activation rotation directly (non-Step leaf path — the
/// CPU-top-K fallback is not a Step program). Mirrors the legacy
/// `x_rot_local` block; returns the rotated activation (or `None` when the
/// routed block does not need one).
fn launch_decode_rotate<'a>(
    gpu: &mut Gpu,
    p: &'a crate::families::moe::MoeParams<'a>,
    res: &crate::families::moe::MoeResolution,
) -> Result<Option<&'a GpuTensor>, DispatchError> {
    macro_rules! hip {
        ($e:expr) => {
            $e.map_err(|e| DispatchError::Hip(e.to_string()))
        };
    }
    if !res.needs_x_rot_local {
        return Ok(None);
    }
    if !p.x_rot_prerotated {
        if res.routed_indexable_paro {
            let paro = p
                .routed_gate_up_paro
                .as_ref()
                .expect("routed_indexable_paro implies gate_up paro sidecar");
            hip!(gpu.givens_rotate_to(
                p.x_norm,
                p.x_rot_local,
                paro.pairs,
                paro.theta,
                paro.scales,
                1,
                p.hidden,
                paro.krot,
            ))?;
        } else if res.gate_side_mq4 {
            if let Some(awq) = p.router.awq_scale {
                hip!(gpu.rotate_x_mq_awq(p.x_norm, awq, p.x_rot_local, p.hidden))?;
            } else {
                hip!(gpu.rotate_x_mq(p.x_norm, p.x_rot_local, p.hidden))?;
            }
        } else {
            // !gate_side_mq4 but routed MQ4/MQ6: no AWQ on MoE expert weights
            // in Phase 1 targets (A3B). Byte-identical for models without AWQ.
            hip!(gpu.rotate_x_mq(p.x_norm, p.x_rot_local, p.hidden))?;
        }
    }
    Ok(Some(p.x_rot_local))
}

/// Run the gate-side projection directly (non-Step leaf path — the CPU-top-K
/// fallback is not a Step program). Uses the SAME extracted helpers as the
/// Step program's gate-side steps, so the kernels can never drift. The
/// shared gate/up slice views are created inside the helpers.
fn launch_gate_side(
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeParams,
    res: &crate::families::moe::MoeResolution,
    x_rot_local: Option<&GpuTensor>,
) -> Result<(), DispatchError> {
    use crate::families::moe::{launch_fused_shared_gate, launch_shared_gate_side};
    if res.gate_fusable {
        launch_fused_shared_gate(
            gpu,
            &p.router,
            &p.shared_expert_gate,
            &p.shared_gate_w,
            &p.shared_up_w,
            x_rot_local.expect("gate_fusable implies x_rot_local (needs_x_rot_local)"),
            p.router_logits,
            p.scalar_buf,
            p.gate_buf,
            p.up_buf,
            p.smi,
        )
    } else {
        launch_shared_gate_side(
            ctx,
            gpu,
            &p.router,
            &p.shared_expert_gate,
            &p.shared_gate_w,
            &p.shared_up_w,
            p.x_norm,
            x_rot_local,
            p.router_logits,
            p.scalar_buf,
            p.gate_buf,
            p.up_buf,
            p.smi,
        )
    }
}

/// Intra-layer MIXED-TIER routed-expert dispatch (REAP SP2 Task 3).
///
/// [GPU-GATE DEFERRED] Implemented under embargo; the numeric bucketing-
/// equivalence gate is owed to a GPU session (see
/// `mixed_dispatch_bucketing_equivalence` stub + the rationale block at the
/// `res.mixed` branch in `run_moe_decode`).
///
/// Runs the gate_up + fused-activation + down for each quant tier present in
/// the top-k, reusing the existing single-tier indexed kernels. The crux —
/// rank alignment — is solved by **permute-to-contiguous (Approach A)**:
/// the top-k is reordered so each tier owns a contiguous rank range, then each
/// tier's kernels are driven over that range through `GpuTensor::sub_offset`
/// byte-offset views (the kernels have no output-rank offset arg, but a base
/// offset on the per-rank buffers gives the same effect since their internal
/// `krank` starts at 0).
///
/// On return, the per-rank `down_expanded` buffer is fully populated and the
/// device-side `topk_weights` are PERMUTED to match the new rank order, so the
/// caller's shared `moe_down_combine_k8_batched` (which sums all k slots with
/// `topk_weights`) yields the correct, permutation-invariant per-token result.
///
/// Does NOT call the combine itself (the caller's shared line does), and does
/// NOT touch `out_target` directly.
#[allow(dead_code, clippy::too_many_arguments)]
fn run_moe_decode_mixed(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeParams,
    _res: &crate::families::moe::MoeResolution,
    xr: &GpuTensor,
    gate_up_k: usize,
    down_m: usize,
    down_k: usize,
    _out_target: &GpuTensor,
) -> Result<(), DispatchError> {
    use crate::families::moe_buckets::bucket_topk_by_tier;
    macro_rules! hip {
        ($e:expr) => {
            $e.map_err(|e| DispatchError::Hip(e.to_string()))
        };
    }

    let k = p.k;
    let mi = p.mi;

    // The mixed path is only reachable when per_expert_gate_up is Some AND spans
    // >1 tier (resolve() guarantees this). per_expert_down is the parallel
    // table; we bucket by gate_up tier and assert the same expert's down tier
    // matches (REAP re-quantizes a whole expert to one tier, so gate_up and down
    // share it — a future split-tier-per-expert model would need separate
    // gate_up/down bucketing).
    let tier_of_gate_up =
        p.dtypes
            .per_expert_gate_up
            .as_ref()
            .ok_or(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "mixed-without-per-expert-gate-up-table",
                arch: "",
                quant: "",
            })?;
    let tier_of_down = p.dtypes.per_expert_down.as_ref();

    // ── Validate the per-expert tier tables up front (review #6/#7) ──────────
    // The tables are indexed by routed-expert id (0..n_exp). A mis-sized table
    // would panic on `tier_of[e]` while bucketing; a table holding a tier the
    // mixed path has no kernel for would otherwise fail deep in the per-bucket
    // dispatch with a cryptic `quant: "other"`. Catch both here, clearly.
    if tier_of_gate_up.len() != p.n_exp {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "mixed-gate_up-tier-table-len-ne-n_exp",
            arch: "",
            quant: "",
        });
    }
    if let Some(td) = tier_of_down {
        if td.len() != p.n_exp {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "mixed-down-tier-table-len-ne-n_exp",
                arch: "",
                quant: "",
            });
        }
    }
    // Every gate_up tier must have per-tier kernels (down tiers are checked to
    // equal the gate_up tier below, so they are transitively covered).
    if let Some(&bad) = tier_of_gate_up
        .iter()
        .find(|t| !crate::families::moe::MIXED_SUPPORTED_TIERS.contains(t))
    {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "mixed-unsupported-routed-tier",
            arch: "",
            quant: dtype_name(bad),
        });
    }

    // ── Host top-k sync (single small D2H; see rationale block) ──────────────
    // topk_indices are i32 stored in an F32-typed scratch tensor (4-byte slots).
    let mut idx_bytes = vec![0u8; k * 4];
    hip!(gpu.hip.memcpy_dtoh(&mut idx_bytes, &p.topk_indices.buf))?;
    let topk_idx: Vec<u32> = idx_bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
        .collect();
    let mut w_bytes = vec![0u8; k * 4];
    hip!(gpu.hip.memcpy_dtoh(&mut w_bytes, &p.topk_weights.buf))?;
    let topk_w: Vec<f32> = w_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // ── Partition by tier, then build the permutation to contiguous ranges ───
    // tier_of_gate_up.len() == n_exp is enforced above and the router emits ids
    // < n_exp, so this only fails on a corrupt routing readback — surface it as
    // a clean error instead of an out-of-bounds panic in the bucketer.
    let buckets = bucket_topk_by_tier(&topk_idx, tier_of_gate_up).map_err(|bad| {
        eprintln!(
            "moe mixed: routed expert id {bad} >= tier table len {}",
            tier_of_gate_up.len()
        );
        DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "mixed-routed-expert-id-out-of-range",
            arch: "",
            quant: "",
        }
    })?;
    let (perm, ranges) = build_contiguous_permutation(&buckets, k);

    // Assert per-expert down tier agrees with the gate_up tier we bucketed on.
    if let Some(td) = tier_of_down {
        for b in &buckets {
            for &e in &b.experts {
                if td[e as usize] != b.tier {
                    return Err(DispatchError::UnsupportedVariant {
                        family: "moe",
                        variant: "mixed-expert-gate_up-down-tier-mismatch",
                        arch: "",
                        quant: "",
                    });
                }
            }
        }
    }

    // ── Re-upload permuted topk_indices + topk_weights to device ─────────────
    // gate_up/down read topk_indices[rank]; the shared combine reads
    // topk_weights[rank]. Writing the permuted order into the SAME scratch
    // buffers keeps every downstream kernel using the contiguous-by-tier rank
    // order. (Identity permutation ⇒ bytes unchanged ⇒ uniform-equivalent.)
    let perm_idx: Vec<i32> = perm.iter().map(|&old| topk_idx[old] as i32).collect();
    let perm_w: Vec<f32> = perm.iter().map(|&old| topk_w[old]).collect();
    let idx_out: Vec<u8> = perm_idx.iter().flat_map(|v| v.to_le_bytes()).collect();
    let w_out: Vec<u8> = perm_w.iter().flat_map(|v| v.to_le_bytes()).collect();
    hip!(gpu.hip.memcpy_htod(&p.topk_indices.buf, &idx_out))?;
    hip!(gpu.hip.memcpy_htod(&p.topk_weights.buf, &w_out))?;

    // MQ tiers need the FWHT sign tables resident; ensure once if any bucket is
    // an MQ tier (cheap idempotent no-op when already loaded).
    if buckets
        .iter()
        .any(|b| matches!(b.tier, DType::MQ4G256 | DType::MQ6G256))
    {
        hip!(gpu.ensure_mq_signs())?;
    }

    // ── Per-tier dispatch over each bucket's contiguous rank range ───────────
    // Every per-rank buffer is sub-offset to the bucket's [lo, lo+n) range so
    // the kernel's internal `krank ∈ 0..n` lands in the right global slots:
    //   topk_indices : i32/F32 slots, stride 1   → offset lo,    len n
    //   gate_batch   : [k × mi] f32              → offset lo*mi, len n*mi
    //   up_batch     : [k × mi] f32              → offset lo*mi, len n*mi
    //   rot_batch    : [k × K]  f32 (K=down_k)   → offset lo*K,  len n*K
    //   down_expanded: [k × M]  f32 (M=down_m)   → offset lo*M,  len n*M
    // x (xr) is the shared per-token rotated activation — NOT per-rank — so it
    // is passed whole to every bucket.
    for (bi, b) in buckets.iter().enumerate() {
        let (lo, n) = ranges[bi];
        if n == 0 {
            continue;
        }
        let idx_view = p.topk_indices.sub_offset(lo, n);
        let gate_view = p.gate_batch.sub_offset(lo * mi, n * mi);
        let up_view = p.up_batch.sub_offset(lo * mi, n * mi);
        let rot_view = p.rot_batch.sub_offset(lo * down_k, n * down_k);
        let down_view = p.down_expanded.sub_offset(lo * down_m, n * down_m);

        // gate_up GEMV (m = 2*mi, gate and up halves), per tier.
        match b.tier {
            DType::MQ4G256 => hip!(gpu.gemv_hfq4g256_moe_gate_up_k8_indexed(
                p.expert_gate_up_ptrs,
                &idx_view,
                xr,
                &gate_view,
                &up_view,
                2 * mi,
                gate_up_k,
                n,
            ))?,
            DType::MQ6G256 => hip!(gpu.gemv_hfq6g256_moe_gate_up_k8_indexed(
                p.expert_gate_up_ptrs,
                &idx_view,
                xr,
                &gate_view,
                &up_view,
                2 * mi,
                gate_up_k,
                n,
            ))?,
            DType::ParoQ4G128 => hip!(gpu.gemv_paro_q4g128_moe_gate_up_k8_indexed(
                p.expert_gate_up_ptrs,
                &idx_view,
                xr,
                &gate_view,
                &up_view,
                2 * mi,
                gate_up_k,
                n,
            ))?,
            other => {
                return Err(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "mixed-gate_up-unsupported-tier",
                    arch: "",
                    quant: dtype_name(other),
                })
            }
        }

        // Fused silu·mul·rotate over this bucket's n ranks.
        match b.tier {
            DType::ParoQ4G128 => {
                let paro_down =
                    p.routed_down_paro
                        .as_ref()
                        .ok_or(DispatchError::UnsupportedVariant {
                            family: "moe",
                            variant: "mixed-paro-without-down-sidecar",
                            arch: "",
                            quant: "",
                        })?;
                hip!(gpu.fused_silu_mul_givens_rotate_f32(
                    &gate_view,
                    &up_view,
                    &rot_view,
                    paro_down.pairs,
                    paro_down.theta,
                    paro_down.scales,
                    n,
                    mi,
                    paro_down.krot,
                ))?;
            }
            // MQ4 / MQ6 share the FWHT activation (no AWQ on expert down for A3B).
            _ => {
                hip!(gpu.fused_silu_mul_rotate_mq_batched(&gate_view, &up_view, &rot_view, mi, n,))?
            }
        }

        // down GEMV → expanded write into this bucket's rank slots. k_top = n,
        // batch_size = 1 ⇒ routing_base = 0 ⇒ reads idx_view[krank], writes
        // down_view[krank*M] for krank ∈ 0..n. FIXME(Step 8): batch_size.
        match b.tier {
            DType::MQ4G256 => hip!(gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                p.expert_down_ptrs,
                &idx_view,
                &rot_view,
                &down_view,
                down_m,
                down_k,
                n,
                1,
            ))?,
            DType::MQ6G256 => hip!(gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
                p.expert_down_ptrs,
                &idx_view,
                &rot_view,
                &down_view,
                down_m,
                down_k,
                n,
                1,
            ))?,
            DType::ParoQ4G128 => hip!(gpu.gemv_paro_q4g128_moe_down_k8_indexed_batched(
                p.expert_down_ptrs,
                &idx_view,
                &rot_view,
                &down_view,
                down_m,
                down_k,
                n,
                1,
            ))?,
            other => {
                return Err(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "mixed-down-unsupported-tier",
                    arch: "",
                    quant: dtype_name(other),
                })
            }
        }
    }

    // Combine is the caller's shared `moe_down_combine_k8_batched` (it now reads
    // the permuted topk_weights → permutation-invariant per-token sum).
    Ok(())
}

/// Build the permute-to-contiguous mapping for the mixed-tier path (pure; CPU-
/// unit-tested). Given the per-tier `buckets` (first-seen order) and the top-k
/// width `k`, returns `(perm, ranges)` where:
///   - `perm[new_rank] = old_rank` — concatenating each bucket's `ranks` makes
///     every tier a contiguous block in the new order.
///   - `ranges[b] = (lo, n)` — bucket `b`'s contiguous `[lo, lo+n)` slice.
///
/// EQUIVALENCE INVARIANT (CPU-checkable here): for an all-ONE-tier table there
/// is exactly one bucket whose `ranks` are already `0..k` in order, so `perm`
/// is the IDENTITY and `ranges == [(0, k)]`. That is what makes the mixed path
/// emit the same kernel calls as the uniform path for a uniform table.
#[allow(dead_code)]
fn build_contiguous_permutation(
    buckets: &[crate::families::moe_buckets::TierBucket],
    k: usize,
) -> (Vec<usize>, Vec<(usize, usize)>) {
    let mut perm: Vec<usize> = Vec::with_capacity(k);
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(buckets.len());
    for b in buckets {
        let lo = perm.len();
        perm.extend_from_slice(&b.ranks);
        ranges.push((lo, b.ranks.len()));
    }
    debug_assert_eq!(perm.len(), k, "permutation must cover all k ranks");
    (perm, ranges)
}

/// Static name for a DType (for UnsupportedVariant.quant in the mixed path).
/// Covers the tiers a routed expert can realistically carry so an
/// unsupported-tier error names the actual offending tier (e.g. "Q8_0")
/// instead of a useless "other".
#[allow(dead_code)]
fn dtype_name(d: DType) -> &'static str {
    match d {
        DType::MQ4G256 => "MQ4G256",
        DType::MQ6G256 => "MQ6G256",
        DType::ParoQ4G128 => "ParoQ4G128",
        DType::Q8_0 => "Q8_0",
        DType::MQ3G256 => "MQ3G256",
        DType::MQ2G256 => "MQ2G256",
        _ => "other",
    }
}

/// Generic CPU-top-K MoE decode fallback. Restores the per-expert loop #393
/// deleted from `moe_ffn_decode_impl` (origin/master qwen35.rs). Fires for any
/// MoE layer the GPU-top-K fast path can't serve: `k != 8`, or a routed expert
/// dtype outside `{MQ4G256, MQ5G256, MQ6G256, ParoQ4G128}` (e.g. a Q8-routed MoE).
///
/// Sequence mirrors master exactly:
///   1. softmax(router_logits)
///   2. download probs → CPU top-K select + sort + renorm
///   3. shared-expert down (identical to the GPU-top-K path's shared-down block)
///   4. per-expert routed loop: gate_up GEMV → silu·mul → down GEMV → scaled add
///
/// Step 4 uses `GemvFamily::run_auto`, which is the dispatch-crate equivalent of
/// master's `weight_gemv`: it auto-rotates (FWHT for MQ family / Givens for Paro)
/// when the routed dtype requires it, and runs plain otherwise — so this single
/// loop covers every routed dtype, matching master's generic `weight_gemv` arm.
///
/// `ctx` is threaded through every inner GEMV (no internal `DispatchCtx::new`).
/// The gate-side projection (which feeds this fallback's shared-down math)
/// ran in the caller before the branch — the shared gate/up slice views are
/// re-derived inside [`launch_shared_expert_down`].
fn run_moe_decode_cpu_fallback(
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeParams,
) -> Result<(), DispatchError> {
    macro_rules! hip {
        ($e:expr) => {
            $e.map_err(|e| DispatchError::Hip(e.to_string()))
        };
    }

    // EP (Ship 6 substrate-EP) is not wired through the generic CPU-top-K
    // fallback yet — it still accumulates into x_residual directly. The
    // fast-path (use_gpu_topk) covers all current EP-target MoE models
    // (qwen3.6-A3B k=8 MQ4). Reject EP here so it can't silently emit
    // wrong (un-redirected) output rather than the all-reduced partial.
    if p.routed_out.is_some() {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "ep-routed-out-unsupported-in-cpu-topk-fallback",
            arch: "",
            quant: "",
        });
    }

    // Per-expert weights are required to iterate (master indexed
    // `ffn.experts[expert_idx]`). They are empty under paged residency, where
    // only the indexed GPU-top-K path is supported — same invariant as master.
    if p.routed_experts.is_empty() {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "cpu-topk-fallback-needs-resident-experts",
            arch: "",
            quant: "",
        });
    }

    // hipGraph capture safety. This fallback's per-expert dispatch loop is
    // indexed by host-side `topk_indices` (downloaded from the device), so it is
    // fundamentally non-capturable: even with the [k] D2H made capture-safe, a
    // captured graph would bake in THIS token's expert selection and mis-route on
    // every replay. Refuse loudly instead of corrupting output (or crashing with
    // a cryptic hipError 906). Only reachable when `!use_gpu_topk` (k != 8 or
    // non-indexable routed dtype); every shipping model (A3B k=8, indexable) takes
    // the GPU-top-K fast path and never lands here. A future k!=8 / non-indexable
    // MoE model must run with HIPFIRE_GRAPH_MOE=0 (or HIPFIRE_AR_GRAPH=0). This
    // replaces "graph safety depends on model-config luck" with a hard guard.
    if gpu.graphs.replay.capturing.is_some() {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "cpu-topk-fallback-not-capture-safe(set HIPFIRE_GRAPH_MOE=0)",
            arch: "",
            quant: "",
        });
    }

    let k = p.k;
    let mi = p.mi;
    let n_exp = p.n_exp;

    // Defensive: select_nth_unstable_by(k-1) panics if k > n_exp or k == 0.
    // No known model violates k ∈ [1, n_exp], but Step 8 brings new families.
    if k == 0 || k > n_exp {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "cpu-topk-k-out-of-range",
            arch: "",
            quant: "",
        });
    }

    // ── 1+2. softmax → top-K + renorm ─────────────────────────────────────────
    // For k==8 we use the same two GPU kernels as the fast path
    // (`softmax_f32` + `moe_topk_renorm_k8`) so this code is capture-safe
    // under hipGraph.  Only a tiny [k] D2H follows (32 bytes for A3B k=8)
    // to get the selected indices/weights for the CPU expert loop below.
    // For k != 8 (no current production model) we fall back to the original
    // [n_exp] D2H path — that case cannot reach a graph capture site anyway
    // because `use_gpu_topk` requires `k == 8`.
    hip!(gpu.softmax_f32(p.router_logits))?;
    let (topk_indices, topk_weights): (Vec<usize>, Vec<f32>) = if k == 8 {
        hip!(gpu.moe_topk_renorm_k8(
            p.router_logits,
            p.topk_indices,
            p.topk_weights,
            n_exp,
            p.norm_topk_prob
        ))?;
        // topk_indices is i32 values stored in an F32 GpuTensor (same 4 B/elem);
        // download as f32 bits and reinterpret.
        let idx_f32 = hip!(gpu.download_f32(p.topk_indices))?;
        let wts = hip!(gpu.download_f32(p.topk_weights))?;
        let idx_usize: Vec<usize> = idx_f32
            .iter()
            .map(|&f| i32::from_ne_bytes(f.to_ne_bytes()) as usize)
            .collect();
        (idx_usize, wts)
    } else {
        // Original [n_exp] D2H path for non-k8 models (not capture-eligible).
        let probs = hip!(gpu.download_f32(p.router_logits))?;
        let mut indices: Vec<usize> = (0..n_exp).collect();
        indices.select_nth_unstable_by(k - 1, |&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut sel: Vec<usize> = indices.into_iter().take(k).collect();
        sel.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut wts: Vec<f32> = sel.iter().map(|&i| probs[i]).collect();
        if p.norm_topk_prob {
            let sum: f32 = wts.iter().sum();
            if sum > 0.0 {
                for w in wts.iter_mut() {
                    *w /= sum;
                }
            }
        }
        (sel, wts)
    };

    // ── 3. Shared-expert down (identical to the GPU-top-K shared-down block) ──
    // Extracted helper: the fallback shares ONE implementation with the Step
    // program's MoeSharedDown step. The fallback never consults skip_shared
    // and rejects routed_out up front, so `out_target` is always x_residual —
    // exactly the legacy fallback body.
    crate::families::moe::launch_shared_expert_down(
        ctx,
        gpu,
        &p.shared_down_w,
        p.gate_buf,
        p.up_buf,
        p.scalar_buf,
        p.ffn_hidden,
        p.ffn_out,
        p.x_residual,
        p.smi,
    )?;

    // ── 4. Per-expert routed loop (master's generic `weight_gemv` arm) ────────
    static GEMV_FB: OnceLock<GemvFamily> = OnceLock::new();
    let gemv = GEMV_FB.get_or_init(GemvFamily::new);

    // GPTQ-on-E8 native Hessian capture (gpu.hessian_capture is Some only
    // under the collect_e8_hessian_native calibration driver; None == zero
    // overhead in production). x_norm is the RAW pre-rotation gate_up input
    // (post-rmsnorm hidden, pre-FWHT) and is identical for every top-k expert
    // of this token, so download it ONCE here. Keyed by the FULL safetensors
    // name == hipfire-quantize::main::hessian_key.
    let hess_x_norm: Option<Vec<f32>> = if gpu.hessian_capture.is_some() {
        Some(hip!(gpu.download_f32(p.x_norm))?)
    } else {
        None
    };
    // GPTQ-on-E8 capture staging: gather this token's per-expert down activations
    // (silu(g)*u) so the per-(tensor,expert) XX^T accumulate — the capture
    // bottleneck (single-threaded f64 rank-1 over a ~30 GB cold working set while
    // the GPU sits idle) — runs ONCE, in PARALLEL across the token's disjoint
    // accumulators, after the expert loop. `hid_host` Vecs must outlive the
    // batched call, hence the owning stash. Zero overhead when capture is off.
    let mut hess_down_keys: Vec<(String, Vec<f32>)> = if hess_x_norm.is_some() {
        Vec::with_capacity(topk_indices.len())
    } else {
        Vec::new()
    };
    let mut hess_gate_keys: Vec<String> = if hess_x_norm.is_some() {
        Vec::with_capacity(topk_indices.len())
    } else {
        Vec::new()
    };

    for (&expert_idx, &weight) in topk_indices.iter().zip(topk_weights.iter()) {
        let (gate_up_w, down_w) = &p.routed_experts[expert_idx];

        // gate_up: y = W·x  (run_auto auto-rotates for MQ/Paro dtypes).
        {
            gemv.run_auto(ctx, gpu, gate_up_w, p.x_norm, p.gate_up_buf)?;
        }
        let gate_view = unsafe { slice_moe_f32_view(p.gate_up_buf, 0, mi) };
        let up_view = unsafe { slice_moe_f32_view(p.gate_up_buf, mi, mi) };

        // silu(gate)·up → ffn_hidden, then down GEMV, then weighted residual add.
        let hid_view = unsafe { slice_moe_f32_view(p.ffn_hidden, 0, mi) };
        hip!(gpu.silu_mul_f32(&gate_view, &up_view, &hid_view))?;
        // GPTQ-on-E8 Hessian capture: hid_view = silu(g)*u is the RAW
        // PRE-rotation down input (run_auto below applies the FWHT internally),
        // so download it NOW, before the down GEMV, and STAGE it (gate_up shares
        // the single pre-downloaded x_norm). The actual XX^T accumulate is
        // deferred to one parallel `accumulate_token` after the loop.
        if hess_x_norm.is_some() {
            let hid_host = hip!(gpu.download_f32(&hid_view))?;
            let l = p.layer_idx;
            let e = expert_idx;
            hess_gate_keys.push(format!(
                "model.language_model.layers.{l}.mlp.experts.{e}.gate_up_proj.weight"
            ));
            hess_down_keys.push((
                format!("model.language_model.layers.{l}.mlp.experts.{e}.down_proj.weight"),
                hid_host,
            ));
        }
        {
            gemv.run_auto(ctx, gpu, down_w, &hid_view, p.ffn_out)?;
        }
        hip!(gpu.scaled_add_inplace_cpu_scalar_f32(p.x_residual, p.ffn_out, weight))?;
    }

    // GPTQ-on-E8: one batched, rayon-parallel accumulate over the token's
    // disjoint (tensor,expert) accumulators (distinct expert ids + distinct
    // gate_up/down tensors ⇒ disjoint targets ⇒ bit-identical to the per-expert
    // serial accumulate; see `accumulate_token`).
    if let Some(ref xn) = hess_x_norm {
        let mut items: Vec<(String, &[f32], usize)> =
            Vec::with_capacity(hess_gate_keys.len() + hess_down_keys.len());
        for gk in &hess_gate_keys {
            items.push((gk.clone(), xn.as_slice(), p.hidden));
        }
        for (dk, hid) in &hess_down_keys {
            items.push((dk.clone(), hid.as_slice(), mi));
        }
        if let Some(cap) = gpu.hessian_capture.as_mut() {
            cap.accumulate_token(&items);
        }
    }

    Ok(())
}

/// DeepSeek-V4 bias-aware MoE decode executor. Transcribes the routed sub-graph
/// of `hipfire-arch-deepseek4::forward::ffn_routed` (the fused
/// `expert_gate_up_blob` branch): bias-aware top-k select → indexed MQ2-Lloyd
/// gate_up → batched silu·mul·clamp → batched FWHT rotate → indexed MQ2-Lloyd
/// down with route-scaled residual accumulation into `ffn_out`.
///
/// The router GEMV + `sqrt_softplus` (producing `p.scores`) and the shared
/// expert stay model-owned — the shared expert seeds `p.ffn_out` and this arm
/// accumulates into it, so the model must run it first. Decode only
/// (`batch_size == 1`); batched prefill is the grouped executor (Step 8).
pub fn run_moe_decode_bias_aware(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeBiasAwareParams,
) -> Result<(), DispatchError> {
    macro_rules! hip {
        ($e:expr) => {
            $e.map_err(|e| DispatchError::Hip(e.to_string()))
        };
    }
    if p.batch_size != 1 {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "bias-aware-decode-requires-batch-1",
            arch: "",
            quant: "",
        });
    }

    // 1. Bias-aware top-K: select on (scores + bias), weight on the unbiased
    //    scores, normalize, then fold in route_scale — all in one launch.
    hip!(gpu.deepseek4_moe_topk_bias_aware_f32(
        p.scores,
        p.gate_bias,
        p.topk_indices,
        p.topk_weights,
        p.n_exp as i32,
        p.k_top as i32,
        p.route_scale,
    ))?;

    // 2. Indexed MQ2-Lloyd gate_up: all k_top experts in one launch
    //    (M = 2*mi; the kernel splits rows r<mi → gate, r>=mi → up).
    hip!(gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
        p.expert_gate_up_ptrs,
        p.topk_indices,
        p.x_rot,
        p.gate_batch,
        p.up_batch,
        2 * p.mi,
        p.hidden,
        p.k_top,
    ))?;

    // 3. Batched silu·mul·clamp (in-place into gate_batch) then batched FWHT rotate.
    hip!(gpu.deepseek4_silu_mul_clamp_f32_batched(
        p.gate_batch,
        p.up_batch,
        p.gate_batch,
        p.mi,
        p.k_top,
        p.swiglu_limit,
    ))?;
    hip!(gpu.rotate_x_mq_batched(p.gate_batch, p.rot_batch, p.mi, p.k_top))?;

    // 4. Indexed MQ2-Lloyd down. Deterministic (default): expanded per-expert
    //    write + fixed-order non-atomic combine into ffn_out — bit-reproducible
    //    for greedy/spec-decode. MOE_DETERMINISTIC=0 uses the faster
    //    atomicAdd-fused path (nondeterministic; bench only).
    let deterministic = std::env::var("HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC").as_deref() != Ok("0");
    if deterministic {
        hip!(gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4(
            p.expert_down_ptrs,
            p.topk_indices,
            p.rot_batch,
            p.down_expanded,
            p.hidden,
            p.mi,
            p.k_top,
            1,
        ))?;
        hip!(gpu.moe_down_combine_k8_batched(
            p.down_expanded,
            p.topk_weights,
            p.ffn_out,
            p.hidden,
            p.k_top,
            1,
        ))?;
    } else {
        hip!(
            gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
                p.expert_down_ptrs,
                p.topk_indices,
                p.topk_weights,
                p.rot_batch,
                p.ffn_out,
                p.hidden,
                p.mi,
                p.k_top,
            )
        )?;
    }

    Ok(())
}

/// MQ2-Lloyd grouped-GEMM kernel variant (deepseek4 research levers; default
/// `Lloyd4w` on gfx11+, `Base` otherwise). Selected once per gate_up/down call.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum GroupedLloydVariant {
    N32,
    Cnd,
    EightW,
    Nosync,
    Mmqload,
    Lloyd4w,
    Base,
}

/// Mirror of `ffn_batched`'s grouped-GEMM if/else-if ladder (priority order:
/// n32 > cnd > 8w > nosync > mmqload > 4w > base). `n32`/`cnd`/`eightw` apply
/// only on the 4w path; `use_nosync` ⊂ `use_mmqload` ⊂ `use_lloyd_4w`.
pub(crate) fn select_grouped_lloyd_variant(
    use_lloyd_4w: bool,
    n32: bool,
    cnd: bool,
    eightw: bool,
    use_mmqload: bool,
    use_nosync: bool,
) -> GroupedLloydVariant {
    if use_lloyd_4w && n32 {
        GroupedLloydVariant::N32
    } else if use_lloyd_4w && cnd {
        GroupedLloydVariant::Cnd
    } else if use_lloyd_4w && eightw {
        GroupedLloydVariant::EightW
    } else if use_nosync {
        GroupedLloydVariant::Nosync
    } else if use_mmqload {
        GroupedLloydVariant::Mmqload
    } else if use_lloyd_4w {
        GroupedLloydVariant::Lloyd4w
    } else {
        GroupedLloydVariant::Base
    }
}

/// Dispatch one MQ2-Lloyd grouped GEMM. All seven variants share the signature
/// `(ptrs, tile_ids, slot_index, x, y, m, k, x_row_div, m_total_max, rows)`, so
/// this is called identically for gate_up (m=2*im, k=hidden, x_row_div=k_top,
/// rows=B) and down (m=hidden, k=im, x_row_div=1, rows=B*k_top).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_grouped_lloyd(
    gpu: &mut Gpu,
    variant: GroupedLloydVariant,
    ptrs: &GpuTensor,
    tile_ids: &GpuTensor,
    slot_index: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    x_row_div: usize,
    m_total_max: usize,
    rows: usize,
) -> Result<(), DispatchError> {
    use GroupedLloydVariant as V;
    let r = match variant {
        V::N32 => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32(
            ptrs,
            tile_ids,
            slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total_max,
            rows,
        ),
        V::Cnd => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_cnd(
            ptrs,
            tile_ids,
            slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total_max,
            rows,
        ),
        V::EightW => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_8w_k2(
            ptrs,
            tile_ids,
            slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total_max,
            rows,
        ),
        V::Nosync => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload_nosync(
            ptrs,
            tile_ids,
            slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total_max,
            rows,
        ),
        V::Mmqload => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload(
            ptrs,
            tile_ids,
            slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total_max,
            rows,
        ),
        V::Lloyd4w => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2(
            ptrs,
            tile_ids,
            slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total_max,
            rows,
        ),
        V::Base => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_k2(
            ptrs,
            tile_ids,
            slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total_max,
            rows,
        ),
    };
    r.map_err(|e| DispatchError::Hip(e.to_string()))
}

/// DeepSeek-V4 batched/prefill MoE executor. Transcribes the routed block of
/// `hipfire-arch-deepseek4::forward::ffn_batched`: routing (hash or bias-aware)
/// → routed experts (grouped GEMM when `batch_size >= gate`, else scalar K4
/// indexed) → combine into `p.ffn_out` (the shared expert already seeded it).
/// Router GEMV + `sqrt_softplus` and the shared expert stay model-owned.
pub fn run_moe_prefill_bias_aware(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeBiasAwarePrefillParams,
) -> Result<(), DispatchError> {
    use crate::families::moe::MoePrefillRouting;
    macro_rules! hip {
        ($e:expr) => {
            $e.map_err(|e| DispatchError::Hip(e.to_string()))
        };
    }
    let (hidden, im, n_exp, k_top, batch_size) = (p.hidden, p.mi, p.n_exp, p.k_top, p.batch_size);

    // ── Routing → topk_indices / topk_weights ────────────────────────────────
    match &p.routing {
        MoePrefillRouting::Hash { tid2eid, tokens } => {
            hip!(gpu.hash_router_normalize_f32_batched(
                tid2eid,
                p.scores,
                tokens,
                p.topk_indices,
                p.topk_weights,
                n_exp as i32,
                k_top as i32,
                p.route_scale,
                batch_size as i32,
            ))?;
        }
        MoePrefillRouting::BiasAware { gate_bias } => {
            hip!(gpu.deepseek4_moe_topk_bias_aware_batched_f32(
                p.scores,
                gate_bias,
                p.topk_indices,
                p.topk_weights,
                n_exp as i32,
                k_top as i32,
                p.route_scale,
                batch_size as i32,
            ))?;
        }
    }

    // DIAG: dump per-layer topk indices ([B, k_top] i32) — off by default.
    if let Ok(path) = std::env::var("HIPFIRE_DEEPSEEK4_DUMP_TOPK") {
        use std::io::Write;
        let raw = hip!(gpu.download_f32(p.topk_indices))?;
        let n = batch_size * k_top;
        let indices: Vec<i32> = raw
            .iter()
            .take(n)
            .map(|value| value.to_bits() as i32)
            .collect();
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| DispatchError::Hip(format!("dump_topk open {path}: {e:?}")))?;
        let header = [p.layer_idx as i32, batch_size as i32, k_top as i32];
        let header_bytes = unsafe { std::slice::from_raw_parts(header.as_ptr() as *const u8, 12) };
        f.write_all(header_bytes)
            .map_err(|e| DispatchError::Hip(format!("dump_topk header: {e:?}")))?;
        let data_bytes =
            unsafe { std::slice::from_raw_parts(indices.as_ptr() as *const u8, indices.len() * 4) };
        f.write_all(data_bytes)
            .map_err(|e| DispatchError::Hip(format!("dump_topk data: {e:?}")))?;
    }

    // ── Grouped vs scalar gate ────────────────────────────────────────────────
    let gate_threshold: usize = std::env::var("HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let use_grouped = batch_size >= gate_threshold
        && std::env::var("HIPFIRE_DEEPSEEK4_MOE_GROUPED").as_deref() != Ok("0");

    // Shared research levers (read once; default 4w on gfx11+).
    let lloyd_4w_base = match std::env::var("HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W").as_deref() {
        Ok("0") => Some(false),
        Ok("1") => Some(true),
        _ => None,
    };
    let arch_4w = gpu.arch.starts_with("gfx11") || gpu.arch.starts_with("gfx12");
    let n32 = std::env::var("HIPFIRE_DEEPSEEK4_MOE_N32").as_deref() == Ok("1");
    let cnd = std::env::var("HIPFIRE_DEEPSEEK4_MOE_CND").as_deref() == Ok("1");
    let eightw = std::env::var("HIPFIRE_DEEPSEEK4_MOE_8W").as_deref() == Ok("1");
    let mmqload_env = std::env::var("HIPFIRE_DEEPSEEK4_MOE_MMQLOAD").as_deref() == Ok("1");
    let nosync_env = std::env::var("HIPFIRE_DEEPSEEK4_MOE_NOSYNC").as_deref() == Ok("1");

    if use_grouped {
        // Checked aligned bound: total_slots = batch*k, m_total_max aligned
        // up to the 16-wide block (raw + expert pad). The old unaligned
        // local formula under-counted tiles for nonaligned batches.
        let bounds = checked_deepseek_grouped_bounds(batch_size, k_top, n_exp)?;
        let m_total_max = bounds.m_total_max;

        // Scatter: histogram + offsets + permute (single launch).
        hip!(gpu.moe_scatter_fused_k8(
            p.topk_indices,
            p.expert_token_counts,
            p.expert_offsets,
            p.sorted_slot_index,
            p.expert_tile_ids,
            p.inverse_perm,
            bounds.total_slots,
            n_exp,
            m_total_max,
            MOE_GROUPED_BLOCK_M,
        ))?;

        // Grouped gate_up GEMM (M=2*im, K=hidden, x_row_div=k_top, rows=B).
        let use_lloyd_4w_gu =
            lloyd_4w_base.unwrap_or(arch_4w) && (2 * im) % 64 == 0 && hidden % 256 == 0;
        let use_mmqload_gu = use_lloyd_4w_gu && mmqload_env;
        let use_nosync_gu = use_mmqload_gu && nosync_env;
        let v_gu = select_grouped_lloyd_variant(
            use_lloyd_4w_gu,
            n32,
            cnd,
            eightw,
            use_mmqload_gu,
            use_nosync_gu,
        );
        dispatch_grouped_lloyd(
            gpu,
            v_gu,
            p.expert_gate_up_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            p.x_rot,
            p.y_gate_up_grouped,
            2 * im,
            hidden,
            k_top,
            m_total_max,
            batch_size,
        )?;

        // Unscatter + SwiGLU·clamp.
        let use_fused_unscatter_silu = std::env::var("HIPFIRE_DEEPSEEK4_FUSED_UNSCATTER_SILU")
            .map(|s| s != "0")
            .unwrap_or(false);
        if use_fused_unscatter_silu {
            hip!(gpu.moe_unscatter_silu_clamp_k8(
                p.y_gate_up_grouped,
                p.sorted_slot_index,
                p.gate_batch,
                im,
                k_top,
                m_total_max,
                p.swiglu_limit,
            ))?;
        } else {
            hip!(gpu.moe_gate_up_unscatter_k8(
                p.y_gate_up_grouped,
                p.sorted_slot_index,
                p.gate_batch,
                p.up_batch,
                im,
                k_top,
                m_total_max,
            ))?;
            hip!(gpu.deepseek4_silu_mul_clamp_f32_batched(
                p.gate_batch,
                p.up_batch,
                p.gate_batch,
                im,
                bounds.total_slots,
                p.swiglu_limit,
            ))?;
        }

        // FWHT rotate.
        hip!(gpu.rotate_x_mq_batched(p.gate_batch, p.rot_batch, im, bounds.total_slots))?;

        // Grouped down GEMM (M=hidden, K=im, x_row_div=1, rows=B*k_top).
        let use_lloyd_4w_dn = lloyd_4w_base.unwrap_or(arch_4w) && hidden % 64 == 0 && im % 256 == 0;
        let use_mmqload_dn = use_lloyd_4w_dn && mmqload_env;
        let use_nosync_dn = use_mmqload_dn && nosync_env;
        let v_dn = select_grouped_lloyd_variant(
            use_lloyd_4w_dn,
            n32,
            cnd,
            eightw,
            use_mmqload_dn,
            use_nosync_dn,
        );
        dispatch_grouped_lloyd(
            gpu,
            v_dn,
            p.expert_down_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            p.rot_batch,
            p.y_down_grouped,
            hidden,
            im,
            1,
            m_total_max,
            bounds.total_slots,
        )?;

        // Down-combine: weighted Σ over k_top slots, per (token, m), into ffn_out.
        hip!(gpu.moe_down_combine_grouped_k8(
            p.y_down_grouped,
            p.inverse_perm,
            p.topk_weights,
            p.ffn_out,
            hidden,
            k_top,
            batch_size,
        ))?;
    } else {
        // ── Scalar K4 path (batch_size < gate, or grouped opt-out) ──
        hip!(
            gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4(
                p.expert_gate_up_ptrs,
                p.topk_indices,
                p.x_rot,
                p.gate_batch,
                p.up_batch,
                2 * im,
                hidden,
                k_top,
                batch_size,
            )
        )?;
        hip!(gpu.deepseek4_silu_mul_clamp_f32_batched(
            p.gate_batch,
            p.up_batch,
            p.gate_batch,
            im,
            batch_size * k_top,
            p.swiglu_limit,
        ))?;
        hip!(gpu.rotate_x_mq_batched(p.gate_batch, p.rot_batch, im, batch_size * k_top))?;

        // Down: deterministic expanded+combine (default; bit-reproducible for
        // spec-decode) vs non-deterministic atomic-accumulate.
        let deterministic =
            std::env::var("HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC").as_deref() != Ok("0");
        if deterministic {
            hip!(gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4(
                p.expert_down_ptrs,
                p.topk_indices,
                p.rot_batch,
                p.down_expert_outputs,
                hidden,
                im,
                k_top,
                batch_size,
            ))?;
            hip!(gpu.moe_down_combine_k8_batched(
                p.down_expert_outputs,
                p.topk_weights,
                p.ffn_out,
                hidden,
                k_top,
                batch_size,
            ))?;
        } else {
            hip!(
                gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed_batched_k4(
                    p.expert_down_ptrs,
                    p.topk_indices,
                    p.topk_weights,
                    p.rot_batch,
                    p.ffn_out,
                    hidden,
                    im,
                    k_top,
                    batch_size,
                )
            )?;
        }
    }

    Ok(())
}

// ── Qwen3.5 batched MoE prefill (Ship 4.2) ──────────────────────────

/// Dispatch one grouped-GEMM for the given routed expert dtype.
///
/// Deduplicates the per-dtype×i8×k8 grouped-kernel match for gate_up
/// and down — the only difference is `x` (gate_up reads `x_rot_batch`
/// `[N×dim]`, down reads `rot_batch` `[N*k_top×mi]`), `m`, `k`, and
/// `x_row_div`.
///
/// The Paro gate_up `givens_rotate_to` preamble is NOT in this helper —
/// it stays in the gate_up block above the call site. Down has no
/// preamble because `rot_batch` is already Givens-rotated by the
/// silu+rotate step.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_grouped_gemm(
    gpu: &mut Gpu,
    dtype: DType,
    expert_dtype_tags: Option<&GpuTensor>,
    ptrs: &GpuTensor,
    tile_ids: &GpuTensor,
    sorted_slot_index: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    x_row_div: usize,
    m_total: usize,
    rows: usize,
    force_mq4_fp16: bool,
    paro_i8: bool,
    paro_i8_k8: bool,
) -> Result<(), DispatchError> {
    macro_rules! hip {
        ($e:expr) => {
            $e.map_err(|e| DispatchError::Hip(e.to_string()))
        };
    }
    // Mixed per-expert: the merged grouped kernel carries the per-expert stride
    // via the dtype_tags table; takes priority over the uniform dtype dispatch.
    if let Some(tags) = expert_dtype_tags {
        return hip!(gpu.gemm_mixed_moe_grouped_wmma(
            ptrs,
            tags,
            tile_ids,
            sorted_slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total,
            rows,
        ));
    }
    match dtype {
        DType::MQ4G256 => {
            if force_mq4_fp16 {
                hip!(gpu.gemm_hfq4g256_moe_grouped_wmma_k2_fp16(
                    ptrs,
                    tile_ids,
                    sorted_slot_index,
                    x,
                    y,
                    m,
                    k,
                    x_row_div,
                    m_total,
                    rows,
                ))
            } else {
                hip!(gpu.gemm_hfq4g256_moe_grouped_wmma_k2(
                    ptrs,
                    tile_ids,
                    sorted_slot_index,
                    x,
                    y,
                    m,
                    k,
                    x_row_div,
                    m_total,
                    rows,
                ))
            }
        }
        // DType::MQ5G256: grouped-WMMA path is gfx12-only and the kernel
        // (`gemm_hfq5g256_moe_grouped_wmma`) is not yet wired in rdna-compute.
        // MQ5 falls through to `_other => UnsupportedVariant`; on gfx942 the
        // `mq5_on_non_gfx12` guard forces Path 1 so this is never reached.
        DType::MQ6G256 => hip!(gpu.gemm_hfq6g256_moe_grouped_wmma(
            ptrs,
            tile_ids,
            sorted_slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total,
            rows,
        )),
        // mfp4-E8 grouped-WMMA (gfx1151 + gfx12/RDNA4; MoePrefillResolution admits
        // Path 2 for E8 on gfx1151 and gfx1200/gfx1201). The launcher selects the
        // correct WMMA intrinsic variant (gfx1151 vs _gfx12) internally.
        // Amortizes expert-weight reads vs the indexed GEMV — the memory-bound
        // prefill / batched-verify lever.
        DType::MFP4G32E8 => hip!(gpu.gemm_mfp4g32_e8_moe_grouped_wmma(
            ptrs,
            tile_ids,
            sorted_slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total,
            rows,
        )),
        DType::ParoQ4G128 => {
            if paro_i8_k8 {
                hip!(gpu.gemm_paro_q4g128_moe_grouped_mmq_k8_gfx1151(
                    ptrs,
                    tile_ids,
                    sorted_slot_index,
                    x,
                    y,
                    m,
                    k,
                    x_row_div,
                    m_total,
                    rows,
                ))
            } else if paro_i8 {
                hip!(gpu.gemm_paro_q4g128_moe_grouped_mmq_gfx1151(
                    ptrs,
                    tile_ids,
                    sorted_slot_index,
                    x,
                    y,
                    m,
                    k,
                    x_row_div,
                    m_total,
                    rows,
                ))
            } else {
                hip!(gpu.gemm_paro_q4g128_moe_grouped_wmma_k2(
                    ptrs,
                    tile_ids,
                    sorted_slot_index,
                    x,
                    y,
                    m,
                    k,
                    x_row_div,
                    m_total,
                    rows,
                ))
            }
        }
        DType::MQ3G256Lloyd => hip!(gpu.gemm_mq3g256_lloyd_moe_grouped_wmma(
            ptrs,
            tile_ids,
            sorted_slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total,
            rows,
        )),
        _other => Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "prefill-grouped-gemm-dtype",
            arch: "",
            quant: "other",
        }),
    }
}

/// Qwen3.5 batched MoE prefill routed-expert executor. Verbatim transcription
/// of the routed block from `prefill_moe_ffn_body_batched` (qwen35.rs:7281).
///
/// Sequence: scatter → gate_up (Path 2 grouped / Path 1 indexed) → unscatter →
/// SwiGLU+rotate → down (Path 2 / Path 1 / Path 0) → combine into `x_batch`.
///
/// `ctx` is decision-only (arch/env) — resolution is computed from
/// `MoeDtypes` + `ArchCaps` + `FeatureFlags` once at entry. The raw
/// `gpu.gemm_*`/`gpu.gemv_*` kernel calls do not take `ctx`.
pub fn run_moe_prefill(
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    p: &crate::families::moe::MoePrefillParams,
) -> Result<(), DispatchError> {
    use crate::families::moe::{
        build_moe_prefill_steps, prefill_expert_refs, MoePrefillResolution,
    };

    let res = MoePrefillResolution::resolve(&p.dtypes, &ctx.arch, &ctx.flags);
    let force_mq4_grouped_fp16 = res.force_mq4_grouped_fp16 || p.force_mq4_grouped_fp16;
    if std::env::var("HIPFIRE_MOE_PREFILL_TRACE").ok().as_deref() == Some("1") {
        eprintln!(
            "[moe-prefill] arch={} shared=({:?},{:?},{:?},{:?}) routed=({:?},{:?}) \
             path2={} force_mq4_fp16={} grouped_i8={:?}",
            ctx.arch.arch(),
            p.dtypes.shared_gate,
            p.dtypes.shared_expert_gate,
            p.dtypes.shared_expert_up,
            p.dtypes.shared_expert_down,
            p.dtypes.routed_gate_up,
            p.dtypes.routed_down,
            res.use_path2,
            force_mq4_grouped_fp16,
            ctx.flags.moe_grouped_i8,
        );
    }

    // Step-native program: the routed block (scatter → gate_up → unscatter →
    // activation → down → combine) in the exact legacy launch order. The
    // model owns RMSNorm, routing, and the shared expert; `routed_out`
    // redirects the combine into the EP partial (`ep_partial` marks it).
    let (gu_experts, dn_experts) = prefill_expert_refs(p);
    let phases = build_moe_prefill_steps(p, &res, &gu_experts, &dn_experts)?;
    let build = phases.into_build();
    // Mesh spine: single-device mesh (P-A) — byte-identical to execute_steps.
    execute_steps_mesh(
        &hipfire_hardware::DeviceMesh::single(),
        gpu,
        ctx,
        &build.steps,
    )
}

pub fn dispatch_fused(
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    key: KernelKey,
    params: &PipelineParams,
) -> Result<(), DispatchError> {
    let params = match params {
        PipelineParams::Linear(p) => p,
        PipelineParams::Moe(p) => return run_moe_decode(ctx, gpu, p),
    };
    macro_rules! hip {
        ($e:expr) => {
            $e.map_err(|e| DispatchError::Hip(e.to_string()))
        };
    }
    match key {
        KernelKey::GemvMfp4G32Fused => {
            gpu.ensure_mq_signs()
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            let x_rot = unsafe {
                GpuTensor {
                    buf: gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias(),
                    shape: vec![params.k],
                    dtype: rdna_compute::DType::F32,
                }
            };
            hip!(gpu.gemv_mfp4g32_with_rotate(
                params.buf, params.x, params.y, &x_rot, params.m, params.k,
            ))
        }
        _ => Err(DispatchError::UnsupportedVariant {
            family: "pipeline_fused",
            variant: "unknown",
            arch: "",
            quant: "",
        }),
    }
}

#[cfg(test)]
mod mixed_dispatch_tests {
    use super::build_contiguous_permutation;
    use crate::families::moe_buckets::bucket_topk_by_tier;
    use rdna_compute::DType::*;

    /// EQUIVALENCE INVARIANT (host half): an all-ONE-tier table yields the
    /// IDENTITY permutation and a single full-width range. This is the
    /// host-side proof that the mixed path emits the same per-rank kernel
    /// addressing as the uniform path for a uniform table — the device half
    /// (bit-identical `down_expanded`) is the GPU-deferred gate below.
    #[test]
    fn all_one_tier_is_identity_permutation() {
        let topk = [3u32, 7, 1, 5, 0, 2, 6, 4];
        let tier_of = vec![MQ4G256; 8];
        let buckets = bucket_topk_by_tier(&topk, &tier_of).unwrap();
        assert_eq!(buckets.len(), 1, "uniform table ⇒ one bucket");
        let (perm, ranges) = build_contiguous_permutation(&buckets, 8);
        assert_eq!(perm, (0..8).collect::<Vec<_>>(), "identity perm");
        assert_eq!(ranges, vec![(0, 8)], "single full-width range");
    }

    /// A mixed table groups each tier into a contiguous range; `perm` is a
    /// bijection over 0..k and `ranges` tile [0, k) with no gaps/overlap.
    #[test]
    fn mixed_table_is_contiguous_partition() {
        // experts: even→MQ4, odd→MQ6. top-k interleaves tiers.
        let tier_of = vec![MQ4G256, MQ6G256, MQ4G256, MQ6G256, MQ4G256, MQ6G256];
        let topk = [1u32, 0, 3, 2, 5, 4]; // ranks 0..5; tiers MQ6,MQ4,MQ6,MQ4,MQ6,MQ4
        let buckets = bucket_topk_by_tier(&topk, &tier_of).unwrap();
        assert_eq!(buckets.len(), 2);
        let (perm, ranges) = build_contiguous_permutation(&buckets, 6);

        // perm is a bijection over 0..6.
        let mut seen = perm.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..6).collect::<Vec<_>>());

        // ranges tile [0,6) contiguously, summing to k.
        let total: usize = ranges.iter().map(|&(_, n)| n).sum();
        assert_eq!(total, 6);
        let mut cursor = 0;
        for &(lo, n) in &ranges {
            assert_eq!(lo, cursor, "ranges must be gap-free & contiguous");
            cursor += n;
        }
        assert_eq!(cursor, 6);

        // Within each range, perm lists exactly that bucket's original ranks.
        for (bi, b) in buckets.iter().enumerate() {
            let (lo, n) = ranges[bi];
            assert_eq!(&perm[lo..lo + n], b.ranks.as_slice());
        }
    }

    // ── [GPU — DEFERRED] bucketing-equivalence numeric gate ─────────────────
    //
    // NOTE: deferred — GPU under embargo. Cannot run; left as an executable
    // stub so a future GPU session has the exact contract.
    //
    // WHY MULTI-BUCKET (not uniform/identity): the uniform table produces the
    // IDENTITY permutation — a single bucket with lo=0, n=k — so every per-rank
    // sub-view is the full buffer and grid.y = k. That case CANNOT expose the
    // class of bug this gate guards against (gate_up grid.y must equal the
    // bucket's rank count `n`, not a hardwired 8; OOB only manifests for a
    // bucket with lo>0 and/or n<8). So the gate MUST use a real ≥2-tier table.
    //
    // WHAT IT MUST VERIFY: build a real qwen35/lfm2moe MoE decode layer with a
    // GENUINELY MIXED ≥2-tier per-expert table — e.g. n_exp=8 experts split as
    //   5 × MQ4G256  +  3 × MQ6G256
    // with top-k routing (k=8) chosen so that BOTH tiers are selected. After
    // `bucket_topk_by_tier` + `build_contiguous_permutation` this yields at
    // least two contiguous buckets, e.g. ranges [(0, n0), (n0, n1)] with
    //   - a non-first bucket whose base offset lo = n0 > 0, and
    //   - at least one bucket with n < 8,
    // which is EXACTLY the OOB-trigger geometry (gate_up over a `lo`-based
    // sub-view, grid.y = n). Run it TWICE on identical inputs:
    //   (1) MIXED path: per_expert_gate_up/down = Some(<the mixed table>)
    //                   (mixed = true; real bucketing exercised).
    //   (2) REFERENCE  : a per-rank reference that runs each selected expert's
    //                    gate_up→silu·mul·rotate→down→combine for its OWN tier
    //                    in natural (unpermuted) rank order — i.e. the
    //                    mathematically-correct mixed result with no bucketing.
    // ASSERT the `down_expanded` slots (compared rank-for-rank under the
    // permutation) and the final residual `out_target` match within tight fp
    // tolerance (bit-identical if the reference replays the same kernels). This
    // proves the permute-to-contiguous + per-tier-sub-view decomposition is
    // exact for a TRUE multi-bucket layout. Run on BOTH dispatch sites once the
    // ds4 bias-aware/hash sites gain the same bucket loop (two-dispatch-site
    // gotcha).
    #[test]
    #[ignore = "GPU-deferred: requires device; see NOTE — multi-bucket (5×MQ4 + 3×MQ6) bucketing-equivalence numeric gate"]
    fn mixed_dispatch_bucketing_equivalence() {
        // Intentionally empty under the GPU embargo. A GPU session must build
        // the MIXED ≥2-tier MoeParams described in the NOTE above (5×MQ4G256 +
        // 3×MQ6G256, routing that selects both tiers so ≥1 bucket has lo>0 and
        // ≥1 has n<8), call run_moe_decode, download down_expanded + out_target,
        // and assert equality against the per-rank mixed reference.
    }
}
