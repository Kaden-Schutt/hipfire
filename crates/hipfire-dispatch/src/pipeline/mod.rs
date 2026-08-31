// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
use crate::context::DispatchCtx;
use crate::families::gemv::{GemvFamily, WeightRef};
use crate::tables::KernelRegistry;
use crate::types::*;
#[allow(unused_imports)]
use hip_bridge;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::{LazyLock, OnceLock};

pub(crate) mod steps;
pub use steps::{execute_steps, FusedPattern, GemvInput, Step};

// #397 Ship 6 — forward-as-pipeline C-design lowered super-op substrate (types
// only at this step; not on any live path until wired behind HIPFIRE_FORWARD_LOWERED).
pub mod superop;

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

/// True when the gate→down step must NOT fuse the FWHT rotation, because the
/// routed `down` weights were packed in the NATURAL basis.
///
/// Every rotated dtype folds the rotation into the silu+mul step so the down
/// GEMV receives an FWHT-basis activation. `MQ2G256LloydU` is the unrotated
/// sibling of MQ2-Lloyd: rotating here would feed a rotated activation to
/// unrotated weights, which produces plausible-looking garbage rather than an
/// error. Extracted as a predicate so the invariant "unrotated dtype ⇔ no
/// rotation anywhere in its path" can be pinned by test — see
/// `unrotated_dtype_skips_both_rotations`.
pub(crate) fn gate_down_skips_rotation(routed_down: DType) -> bool {
    matches!(routed_down, DType::MQ2G256LloydU)
}

/// Frozen mixed-expert in-memory dtype tags (process-local loader metadata,
/// not HFQ wire ABI). Tags 0..6 retain V1 meaning; 7..18 are the MQV2 pairs.
/// Unknown pairs return `None` — callers must Err loudly, never collapse V2→V1.
pub(crate) fn mixed_expert_dtype_tag(gate: DType, down: DType) -> Option<u8> {
    // GL in either position is always rejected — the tag-branched decoder has
    // no GL branch and would silently mis-decode as MQ4.
    if matches!(gate, DType::MQ2G256GL | DType::MQ3G256GL)
        || matches!(down, DType::MQ2G256GL | DType::MQ3G256GL)
    {
        return None;
    }
    match (gate, down) {
        // Tags 0..6 — V1 pair identities (unchanged).
        (DType::MQ4G256, DType::MQ6G256) => Some(0),
        (DType::MQ4G256, DType::MQ2G256Lloyd) => Some(1),
        (DType::MQ4G256, DType::MQ4G256) => Some(2),
        (DType::MQ4G256, DType::MQ3G256Lloyd) => Some(3),
        (DType::MQ4G256, DType::MFP4G32E8) => Some(4),
        (DType::MQ4G256, DType::MFP3G32E8) => Some(5),
        (DType::MQ4G256, DType::MFP2G32E8) => Some(6),
        // Matching non-MQ4 V1 pairs reuse the same tag numbers.
        (DType::MQ6G256, DType::MQ6G256) => Some(0),
        (DType::MQ2G256Lloyd, DType::MQ2G256Lloyd) => Some(1),
        (DType::MQ3G256Lloyd, DType::MQ3G256Lloyd) => Some(3),
        (DType::MFP4G32E8, DType::MFP4G32E8) => Some(4),
        (DType::MFP3G32E8, DType::MFP3G32E8) => Some(5),
        (DType::MFP2G32E8, DType::MFP2G32E8) => Some(6),
        // Tags 7..18 — frozen MQV2 mixed identities (never collapse to 0..6).
        (DType::MQ4G256V2, DType::MQ4G256V2) => Some(7),
        (DType::MQ6G256V2, DType::MQ6G256V2) => Some(8),
        (DType::MQ4G256V2, DType::MQ6G256) => Some(9),
        (DType::MQ4G256V2, DType::MQ2G256Lloyd) => Some(10),
        (DType::MQ4G256V2, DType::MQ4G256) => Some(11),
        (DType::MQ4G256, DType::MQ4G256V2) => Some(12),
        (DType::MQ4G256V2, DType::MQ3G256Lloyd) => Some(13),
        (DType::MQ4G256V2, DType::MFP4G32E8) => Some(14),
        (DType::MQ4G256V2, DType::MFP3G32E8) => Some(15),
        (DType::MQ4G256V2, DType::MFP2G32E8) => Some(16),
        (DType::MQ4G256V2, DType::MQ6G256V2) => Some(17),
        (DType::MQ4G256, DType::MQ6G256V2) => Some(18),
        _ => None,
    }
}

/// Which ninepath D3 family a uniform (gate, down) pair selects, if any.
/// Only HFQ4/MQ4V1 (MQ4G256) may call `gemv_hfq4g256_moe_ninepath_d3`;
/// V2 pairs (MQ4G256V2, MQ6G256V2) must use exact native indexed gate + V2 D4.
/// Split V1/V2 pairings never share a family — wrong header is silent garbage.
pub(crate) fn ninepath_d3_family(gate: DType, down: DType) -> Option<&'static str> {
    match (gate, down) {
        (DType::MQ4G256, DType::MQ4G256) => Some("hfq4"),
        _ => None,
    }
}

/// True when per-expert gate_up dtypes contain >1 distinct DType (exact equality,
/// not family). Used to decide mixed gate precedence: whenever gate_up exact
/// dtype varies, mixed gate kernel must run before representative MQ4V2/MQ6V2/V1 arms.
/// `None` (uniform table) or all-equal `Some` ⇒ false (uniform shortcut allowed).
pub(crate) fn gate_up_varies(per_expert_gate_up: Option<&[DType]>) -> bool {
    match per_expert_gate_up {
        Some(slice) => slice
            .split_first()
            .map_or(false, |(first, rest)| rest.iter().any(|dt| dt != first)),
        None => false,
    }
}

/// Whether decode should use the mixed gate_up kernel. Precedence gate:
/// whenever `has_tags && gate_up_varies`, mixed gate must run before
/// representative V1/V2 arms; uniform shortcut only when !gate_up_varies.
pub(crate) fn decode_gate_uses_mixed(has_tags: bool, gate_up_varies: bool) -> bool {
    has_tags && gate_up_varies
}

/// Prefill Path-1 batched gate_up kind with tag awareness.
/// When `has_tags && gate_up_varies`, the tagged layer must use the mixed
/// batched gate launcher (`gemv_mixed_moe_gate_up_k8_indexed_batched`);
/// uniform shortcut only when `!gate_up_varies` (exact DType equality).
/// Returns `Some("mixed")` for the mixed case, else delegates to
/// `prefill_path1_gate_up_kind`.
pub(crate) fn prefill_path1_gate_up_kind_tag_aware(
    gate_up: DType,
    has_tags: bool,
    gate_up_varies: bool,
) -> Option<&'static str> {
    if decode_gate_uses_mixed(has_tags, gate_up_varies) {
        Some("mixed")
    } else {
        prefill_path1_gate_up_kind(gate_up)
    }
}

/// Prefill Path-1 expanded-down kind with tag awareness.
/// Mixed Path1 must use the mixed batched down launcher
/// (`gemv_mixed_moe_down_k8_indexed_batched_expanded`) when tags exist;
/// do not representative-dispatch tagged layers. Returns `Some("mixed")`
/// when `has_tags`, else delegates to `prefill_path1_down_kind`.
pub(crate) fn prefill_path1_down_kind_tag_aware(
    down: DType,
    has_tags: bool,
) -> Option<&'static str> {
    if has_tags {
        Some("mixed")
    } else {
        prefill_path1_down_kind(down)
    }
}

/// Which ninepath D4 family a uniform (gate, down) pair selects, if any.
/// Split V1/V2 pairings never share a family — wrong header is silent garbage.
pub(crate) fn ninepath_d4_family(gate: DType, down: DType) -> Option<&'static str> {
    match (gate, down) {
        (DType::MQ4G256, DType::MQ4G256) => Some("hfq4"),
        (DType::MQ2G256Lloyd, DType::MQ3G256Lloyd) => Some("mq3l"),
        (DType::MQ4G256V2, DType::MQ4G256V2) => Some("mq4v2"),
        (DType::MQ6G256V2, DType::MQ6G256V2) => Some("mq6v2"),
        _ => None,
    }
}

/// Decode non-ninepath expanded-down kernel kind for a uniform routed_down.
/// V2 never aliases V1 HFQ4/HFQ6 — the final HFQ4 fallthrough is V1-only.
pub(crate) fn decode_expanded_down_kind(down: DType) -> Option<&'static str> {
    match down {
        DType::MQ4G256 => Some("hfq4"),
        DType::MQ4G256V2 => Some("mq4v2"),
        DType::MQ5G256 => Some("hfq5"),
        DType::MQ6G256 => Some("hfq6"),
        DType::MQ6G256V2 => Some("mq6v2"),
        DType::MFP4G32E8 => Some("mfp4e8"),
        DType::ParoQ4G128 => Some("paro"),
        DType::MQ2G256Lloyd | DType::MQ2G256LloydU => Some("mq2lloyd_atomic"),
        DType::MQ3G256Lloyd => Some("mq3lloyd_atomic"),
        DType::MQ2G256GL => Some("mq2gl_atomic"),
        DType::MQ3G256GL => Some("mq3gl_atomic"),
        _ => None,
    }
}

/// Prefill Path-1 batched gate_up kernel kind. V2 has native batched launchers.
pub(crate) fn prefill_path1_gate_up_kind(gate_up: DType) -> Option<&'static str> {
    match gate_up {
        DType::MQ4G256 => Some("hfq4"),
        DType::MQ4G256V2 => Some("mq4v2"),
        DType::MQ5G256 => Some("hfq5"),
        DType::MQ6G256 => Some("hfq6"),
        DType::MQ6G256V2 => Some("mq6v2"),
        DType::MFP4G32E8 => Some("mfp4e8"),
        DType::ParoQ4G128 => Some("paro"),
        _ => None,
    }
}

/// Prefill Path-1 expanded-down kernel kind.
pub(crate) fn prefill_path1_down_kind(down: DType) -> Option<&'static str> {
    match down {
        DType::MQ4G256 => Some("hfq4"),
        DType::MQ4G256V2 => Some("mq4v2"),
        DType::MQ5G256 => Some("hfq5"),
        DType::MQ6G256 => Some("hfq6"),
        DType::MQ6G256V2 => Some("mq6v2"),
        DType::MFP4G32E8 => Some("mfp4e8"),
        DType::ParoQ4G128 => Some("paro"),
        _ => None,
    }
}

/// Uniform grouped-GEMM kernel kind admitted on gfx11/gfx12 Path 2.
pub(crate) fn grouped_gemm_kind(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::MQ4G256 => Some("hfq4"),
        DType::MQ4G256V2 => Some("mq4v2"),
        DType::MQ6G256 => Some("hfq6"),
        DType::MQ6G256V2 => Some("mq6v2"),
        DType::MFP4G32E8 => Some("mfp4e8"),
        DType::ParoQ4G128 => Some("paro"),
        DType::MQ2G256Lloyd | DType::MQ2G256LloydU => Some("mq2lloyd"),
        DType::MQ3G256Lloyd => Some("mq3lloyd"),
        _ => None,
    }
}

/// Shared-expert dense down route. V2 must use exact dense V2 GEMV — never the
/// V1 HFQ4 residual_sigmoid_scaled launcher.
pub(crate) fn shared_dense_down_kind(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::MQ4G256 => Some("hfq4_sigmoid_scaled"),
        DType::MQ4G256V2 => Some("mq4v2_prerotated"),
        DType::MQ6G256V2 => Some("mq6v2_prerotated"),
        _ => Some("run_auto"),
    }
}

/// Dispatch a uniform single-token routed-expert gate/up projection by its exact
/// wire dtype. Architecture code supplies tensors and dimensions; format
/// selection remains in the dispatch layer so V1/V2 headers cannot be confused.
#[allow(clippy::too_many_arguments)]
pub fn run_uniform_moe_gate_up(
    gpu: &mut Gpu,
    dtype: DType,
    expert_ptrs: &GpuTensor,
    topk_indices: &GpuTensor,
    x_rot: &GpuTensor,
    gate: &GpuTensor,
    up: &GpuTensor,
    m: usize,
    k: usize,
    k_top: usize,
) -> Result<(), DispatchError> {
    let hip = |result: hip_bridge::HipResult<()>| {
        result.map_err(|error| DispatchError::Hip(error.to_string()))
    };
    match dtype {
        DType::MQ4G256 => hip(gpu.gemv_hfq4g256_moe_gate_up_k8_indexed(
            expert_ptrs,
            topk_indices,
            x_rot,
            gate,
            up,
            m,
            k,
            k_top,
        )),
        DType::MQ6G256 => hip(gpu.gemv_hfq6g256_moe_gate_up_k8_indexed(
            expert_ptrs,
            topk_indices,
            x_rot,
            gate,
            up,
            m,
            k,
            k_top,
        )),
        DType::MQ4G256V2 => {
            if k_top != 8 {
                return Err(DispatchError::Hip(format!(
                    "MQ4G256V2 indexed gate/up requires top_k=8, got {k_top}"
                )));
            }
            hip(gpu.gemv_mq4g256v2_moe_gate_up_k8_indexed(
                expert_ptrs,
                topk_indices,
                x_rot,
                gate,
                up,
                m,
                k,
            ))
        }
        DType::MQ6G256V2 => {
            if k_top != 8 {
                return Err(DispatchError::Hip(format!(
                    "MQ6G256V2 indexed gate/up requires top_k=8, got {k_top}"
                )));
            }
            hip(gpu.gemv_mq6g256v2_moe_gate_up_k8_indexed(
                expert_ptrs,
                topk_indices,
                x_rot,
                gate,
                up,
                m,
                k,
            ))
        }
        other => Err(DispatchError::Hip(format!(
            "uniform indexed gate/up unsupported dtype {other:?}"
        ))),
    }
}

/// Dispatch a uniform routed-expert expanded-down projection by its exact wire
/// dtype. The expanded result remains uncombined for the caller's weighted sum.
#[allow(clippy::too_many_arguments)]
pub fn run_uniform_moe_down_expanded(
    gpu: &mut Gpu,
    dtype: DType,
    expert_ptrs: &GpuTensor,
    topk_indices: &GpuTensor,
    x_rot: &GpuTensor,
    out: &GpuTensor,
    m: usize,
    k: usize,
    k_top: usize,
    batch_size: usize,
) -> Result<(), DispatchError> {
    let hip = |result: hip_bridge::HipResult<()>| {
        result.map_err(|error| DispatchError::Hip(error.to_string()))
    };
    match dtype {
        DType::MQ4G256 => hip(gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
            expert_ptrs,
            topk_indices,
            x_rot,
            out,
            m,
            k,
            k_top,
            batch_size,
        )),
        DType::MQ6G256 => hip(gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
            expert_ptrs,
            topk_indices,
            x_rot,
            out,
            m,
            k,
            k_top,
            batch_size,
        )),
        DType::MQ4G256V2 => hip(gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
            expert_ptrs,
            topk_indices,
            x_rot,
            out,
            m,
            k,
            k_top,
            batch_size,
        )),
        DType::MQ6G256V2 => hip(gpu.gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded(
            expert_ptrs,
            topk_indices,
            x_rot,
            out,
            m,
            k,
            k_top,
            batch_size,
        )),
        other => Err(DispatchError::Hip(format!(
            "uniform expanded down unsupported dtype {other:?}"
        ))),
    }
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
    use crate::families::moe::MoeResolution;
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

    // EP (Ship 6 substrate-EP): when `routed_out` is set, the shared-down and
    // routed-combine accumulate into that zeroed partial (all-reduced by the EP
    // executor and added into x_residual once). `None` → x_residual directly
    // (single-GPU, byte-identical).
    let out_target: &GpuTensor = p.routed_out.unwrap_or(p.x_residual);
    // gfx1100 experiment: retain one independently schedulable workgroup per
    // expert rank, but let the last rank for each four-row tile perform the
    // deterministic expanded-output fold. This is deliberately narrower than
    // the dtype resolver: mixed/Paro/E8/Lloyd paths keep their existing
    // kernels and combine semantics.
    static DOWN_LAST_COMBINE: OnceLock<bool> = OnceLock::new();
    let down_last_combine = ctx.arch.is_gfx1100()
        && p.batch_size == 1
        && p.k == 8
        && p.expert_dtype_tags.is_none()
        && p.dtypes.routed_down == DType::MQ4G256
        && *DOWN_LAST_COMBINE.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_MOE_DOWN_LAST_COMBINE").as_deref() == Ok("1")
        });

    // ── Activation rotation (mirrors qwen35.rs x_rot_local block) ──────────
    let x_rot_local: Option<&GpuTensor> = if res.needs_x_rot_local {
        if !res.routed_indexable_paro {
            hip!(gpu.ensure_mq_signs())?;
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
                    &paro.pairs,
                    &paro.theta,
                    &paro.scales,
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
        Some(p.x_rot_local)
    } else {
        None
    };

    // ── Gate-side GEMV ───────────────────────────────────────────────────────
    // SAFETY: all slice views alias device memory owned by MoEParams' scratch tensors.
    let shared_gate = unsafe { slice_moe_f32_view(p.gate_buf, 0, p.smi) };
    let shared_up = unsafe { slice_moe_f32_view(p.up_buf, 0, p.smi) };
    if res.gate_fusable {
        let xr = x_rot_local.expect("gate_fusable implies x_rot_local (needs_x_rot_local)");
        // Exact-uniform V1 MQ4G256 gate quartet → one fused launch.
        hip!(gpu.fused_qkvza_hfq4g256(
            &p.router.buf,
            &p.shared_expert_gate.buf,
            &p.shared_gate_w.buf,
            &p.shared_up_w.buf,
            xr,
            p.router_logits,
            p.scalar_buf,
            &shared_gate,
            &shared_up,
            p.router.m,
            p.shared_expert_gate.m,
            p.shared_gate_w.m,
            p.shared_up_w.m,
            p.router.k,
        ))?;
    } else if res.gate_fusable_mq4v2
        && (ctx.arch.is_gfx1100() || ctx.arch.is_gfx1201())
        && p.batch_size == 1
        && p.router.awq_scale.is_none()
        && p.shared_expert_gate.awq_scale.is_none()
        && p.shared_gate_w.awq_scale.is_none()
        && p.shared_up_w.awq_scale.is_none()
        && p.router.k == 2048
        && p.shared_expert_gate.k == 2048
        && p.shared_gate_w.k == 2048
        && p.shared_up_w.k == 2048
        && p.router.m == 256
        && p.shared_expert_gate.m == 1
        && p.shared_gate_w.m == 512
        && p.shared_up_w.m == 512
    {
        let xr = x_rot_local.expect("gate_fusable_mq4v2 implies x_rot_local (needs_x_rot_local)");
        // Exact official Ornith qt44 gate quartet on gfx1100/gfx1201 only →
        // one V2 fused launch. Any miss (arch/AWQ/shape/batch) falls through
        // to the generic four-GEMV branch below — never errors.
        hip!(gpu.fused_qkvza_hfq4g256_mq4v2(
            &p.router.buf,
            &p.shared_expert_gate.buf,
            &p.shared_gate_w.buf,
            &p.shared_up_w.buf,
            xr,
            p.router_logits,
            p.scalar_buf,
            &shared_gate,
            &shared_up,
            p.router.m,
            p.shared_expert_gate.m,
            p.shared_gate_w.m,
            p.shared_up_w.m,
            p.router.k,
        ))?;
    } else {
        static GEMV_GATE: OnceLock<GemvFamily> = OnceLock::new();
        let gemv = GEMV_GATE.get_or_init(GemvFamily::new);
        // Reuse the single normalized FWHT activation for structural MQ
        // router/scalar weights when the fused quartet is not admitted
        // (mixed dtype, non-MQ4 gate side, AWQ, etc.).
        let router_prerot = x_rot_local.is_some()
            && p.router.awq_scale.is_none()
            && p.shared_expert_gate.awq_scale.is_none()
            && matches!(
                crate::types::dtype_post_rotation_variant(p.router.dtype),
                crate::types::GemvVariant::Prerotated
            )
            && matches!(
                crate::types::dtype_post_rotation_variant(p.shared_expert_gate.dtype),
                crate::types::GemvVariant::Prerotated
            )
            && crate::types::KernelKey::dtype_arch_predicate(p.router.dtype).eval_arch(ctx)
            && crate::types::KernelKey::dtype_arch_predicate(p.shared_expert_gate.dtype)
                .eval_arch(ctx);
        if router_prerot {
            let xr = x_rot_local.expect("router_prerot implies x_rot_local");
            gemv.run(
                ctx,
                gpu,
                &crate::families::gemv::GemvParams {
                    w: &p.router,
                    x: xr,
                    y: p.router_logits,
                    variant: crate::types::GemvVariant::Prerotated,
                    residual: None,
                    gate: None,
                    up: None,
                },
            )
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
            gemv.run(
                ctx,
                gpu,
                &crate::families::gemv::GemvParams {
                    w: &p.shared_expert_gate,
                    x: xr,
                    y: p.scalar_buf,
                    variant: crate::types::GemvVariant::Prerotated,
                    residual: None,
                    gate: None,
                    up: None,
                },
            )
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
        } else {
            gemv.run_auto(ctx, gpu, &p.router, p.x_norm, p.router_logits)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            gemv.run_auto(ctx, gpu, &p.shared_expert_gate, p.x_norm, p.scalar_buf)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
        }
        // Shared-expert gate/up: on a graded file the all-MQ4 fused gate path
        // (fused_qkvza_hfq4g256 on the single rotated `xr`) doesn't apply because
        // the router is Q8. But the dense shared gate/up are still MQ-family, and
        // `x_rot_local` has ALREADY been FWHT-rotated once for the routed experts.
        // `run_auto` here would re-rotate x_norm per call (+2 mq_rotate_x/layer);
        // instead reuse the existing rotation via the Prerotated path. Numerically
        // identical (same rotated activation). Q8/HFQ shared weights (no rotation)
        // or AWQ-scaled shared weights fall through to run_auto unchanged.
        let shared_prerot = x_rot_local.is_some()
            && p.shared_gate_w.awq_scale.is_none()
            && p.shared_up_w.awq_scale.is_none()
            && matches!(crate::types::dtype_post_rotation_variant(p.shared_gate_w.dtype), crate::types::GemvVariant::Prerotated)
            && matches!(crate::types::dtype_post_rotation_variant(p.shared_up_w.dtype), crate::types::GemvVariant::Prerotated)
            // The prerotated MQ GEMV must actually exist for this arch. MQ6/HFQ6
            // prerotated is HasMmq (gfx906/RDNA3/RDNA4) → ABSENT on gfx942/CDNA, so
            // taking this shortcut there hits MissingImpl. When unavailable, fall
            // through to run_auto (the pre-2f38a16e gfx942 path that worked).
            && crate::types::KernelKey::dtype_arch_predicate(p.shared_gate_w.dtype).eval_arch(ctx)
            && crate::types::KernelKey::dtype_arch_predicate(p.shared_up_w.dtype).eval_arch(ctx);
        if shared_prerot {
            let xr = x_rot_local.expect("shared_prerot implies x_rot_local");
            gemv.run(
                ctx,
                gpu,
                &crate::families::gemv::GemvParams {
                    w: &p.shared_gate_w,
                    x: xr,
                    y: &shared_gate,
                    variant: crate::types::GemvVariant::Prerotated,
                    residual: None,
                    gate: None,
                    up: None,
                },
            )
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
            gemv.run(
                ctx,
                gpu,
                &crate::families::gemv::GemvParams {
                    w: &p.shared_up_w,
                    x: xr,
                    y: &shared_up,
                    variant: crate::types::GemvVariant::Prerotated,
                    residual: None,
                    gate: None,
                    up: None,
                },
            )
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
        } else {
            gemv.run_auto(ctx, gpu, &p.shared_gate_w, p.x_norm, &shared_gate)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            gemv.run_auto(ctx, gpu, &p.shared_up_w, p.x_norm, &shared_up)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
        }
    }

    // ── Top-K + routed experts: CPU-top-K generic fallback ───────────────────
    // Fires when `!use_gpu_topk` (k != 8 OR routed dtype not indexable). This
    // ports master's `moe_ffn_decode_impl` CPU-fallback per-expert loop
    // (origin/master qwen35.rs, the `else` arm of `if use_gpu_topk`) so MoE
    // layers outside the {k=8, MQ4G256|MQ5G256|MQ6G256|ParoQ4G128-routed} fast path
    // run instead of hard-panicking. #393 deleted this; restoring it keeps the
    // dispatch migration behavior-preserving.
    //
    // The fallback is self-contained: it does softmax → CPU top-K + renorm →
    // shared-expert down → generic per-expert routed loop, then returns. It
    // does NOT fall through to the indexed GPU-top-K path below (which assumes
    // k=8 + an indexable routed dtype).
    if !res.use_gpu_topk {
        return run_moe_decode_cpu_fallback(ctx, gpu, p, &shared_gate, &shared_up);
    }
    // DIAG: dump router logits before softmax (mirrors qwen35 HIPFIRE_DUMP_HIDDEN)
    if let Ok(dump_path) = hipfire_config::developer_var("HIPFIRE_DUMP_HIDDEN") {
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
    let gfx1100_router_mode = hipfire_config::developer_var("HIPFIRE_GFX1100_ROUTER_W64").ok();
    let gfx1151_radiowave_fusions = ctx.arch.is_gfx1151();
    let exact_wave64_router = p.n_exp == 256
        && ((ctx.arch.is_gfx1100()
            // The exact fused router is the production gfx1100 path. `0` retains
            // the two-launch reference path for A/B diagnosis; `approx` retains
            // the old non-bit-exact research kernel without exposing it by
            // accident through the former `1` opt-in.
            && !matches!(gfx1100_router_mode.as_deref(), Some("0" | "approx")))
            || gfx1151_radiowave_fusions);
    static ROUTER_SHARED_FUSE: OnceLock<bool> = OnceLock::new();
    let router_shared_fuse = exact_wave64_router
        && p.batch_size == 1
        && !p.skip_shared
        && p.smi == 512
        && p.shared_down_w.dtype == DType::MQ4G256
        && p.shared_down_w.awq_scale.is_none()
        && *ROUTER_SHARED_FUSE.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_MOE_ROUTER_SHARED_FUSE").as_deref() == Ok("1")
        });
    let wave64_router = (ctx.arch.is_gfx1201()
        && hipfire_config::developer_var("HIPFIRE_GFX1201_ROUTER_W64").as_deref() != Ok("0"))
        || (ctx.arch.is_gfx1100()
            && p.n_exp == 256
            // Research-only: faster on gfx1100, but its routing drift can
            // change greedy trajectories and trigger an attractor.
            && gfx1100_router_mode.as_deref() == Some("approx"));
    if router_shared_fuse {
        let shared_x_rot = unsafe {
            GpuTensor {
                buf: gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias(),
                shape: vec![gpu.scratch.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            }
        };
        hip!(
            gpu.moe_router_softmax_topk_k8_wave64_exact_shared_silu_mq_rotate(
                p.router_logits,
                p.topk_indices,
                p.topk_weights,
                p.n_exp,
                p.norm_topk_prob,
                &shared_gate,
                &shared_up,
                &shared_x_rot,
                p.smi,
            )
        )?;
    } else if exact_wave64_router {
        hip!(gpu.moe_router_softmax_topk_k8_wave64_exact(
            p.router_logits,
            p.topk_indices,
            p.topk_weights,
            p.n_exp,
            p.norm_topk_prob
        ))?;
    } else if wave64_router {
        hip!(gpu.moe_router_softmax_topk_k8_wave64(
            p.router_logits,
            p.topk_indices,
            p.topk_weights,
            p.n_exp,
            p.norm_topk_prob
        ))?;
    } else {
        hip!(gpu.softmax_f32(p.router_logits))?;
        hip!(gpu.moe_topk_renorm_k8(
            p.router_logits,
            p.topk_indices,
            p.topk_weights,
            p.n_exp,
            p.norm_topk_prob
        ))?;
    }

    // ── Shared expert down ───────────────────────────────────────────────────
    // EP: on rank>0 `skip_shared` is set so the replicated shared expert is
    // summed exactly once (computed on rank 0 only). Router + shared gate/up
    // still ran above (fused with the router GEMV) — only the down/accumulate
    // is skipped here. Accumulates into `out_target` (= the EP partial when
    // `routed_out` is set, else `x_residual`).
    if !p.skip_shared {
        if p.shared_down_w.dtype == DType::MQ4G256 {
            hip!(gpu.ensure_mq_signs())?;
            let x_rot_alias = unsafe {
                GpuTensor {
                    buf: gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias(),
                    shape: vec![gpu.scratch.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                    dtype: DType::F32,
                }
            };
            if let Some(awq) = p.shared_down_w.awq_scale {
                hip!(gpu.fused_silu_mul_rotate_mq_awq(
                    &shared_gate,
                    &shared_up,
                    awq,
                    &x_rot_alias,
                    p.smi
                ))?;
            } else if !router_shared_fuse {
                hip!(gpu.fused_silu_mul_rotate_mq(&shared_gate, &shared_up, &x_rot_alias, p.smi))?;
            }
            hip!(gpu.gemv_hfq4g256_residual_sigmoid_scaled_gpu(
                &p.shared_down_w.buf,
                &x_rot_alias,
                out_target,
                p.scalar_buf,
                p.shared_down_w.m,
                p.shared_down_w.k,
            ))?;
        } else if matches!(p.shared_down_w.dtype, DType::MQ4G256V2 | DType::MQ6G256V2) {
            // Exact dense V2 shared-down. qt44/qt47 dual-half headers MUST NOT
            // ride the V1 HFQ4 residual_sigmoid kernel (silent fluent corruption).
            // Sequence mirrors MQ4: silu+FWHT → prerotated dense V2 GEMV →
            // sigmoid(c)·add. Dense routing goes through GemvFamily so the
            // container dtype selects gemv_mq{4,6}g256v2 / residual sisters.
            hip!(gpu.ensure_mq_signs())?;
            let x_rot_alias = unsafe {
                GpuTensor {
                    buf: gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias(),
                    shape: vec![gpu.scratch.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                    dtype: DType::F32,
                }
            };
            if let Some(awq) = p.shared_down_w.awq_scale {
                hip!(gpu.fused_silu_mul_rotate_mq_awq(
                    &shared_gate,
                    &shared_up,
                    awq,
                    &x_rot_alias,
                    p.smi
                ))?;
            } else if !router_shared_fuse {
                hip!(gpu.fused_silu_mul_rotate_mq(&shared_gate, &shared_up, &x_rot_alias, p.smi))?;
            }
            // Default-on one-launch fuse for the exact Ornith qt44 shared-down
            // shape. The scalar has no later read in this executor. `ffn_out`
            // is dead here on the indexed path; on the generic fallback its
            // next access is an overwrite by routed-expert down. `0` or an
            // invalid override keeps sigmoid + plain V2 GEMV + scaled_add.
            static MQ4V2_SHARED_DOWN_FUSED: LazyLock<bool> = LazyLock::new(|| {
                hipfire_config::developer_var("HIPFIRE_MQ4V2_SHARED_DOWN_FUSED")
                    .map(|value| value == "1")
                    .unwrap_or(true)
            });
            let use_mq4v2_shared_down_fused = *MQ4V2_SHARED_DOWN_FUSED
                && p.shared_down_w.dtype == DType::MQ4G256V2
                && p.shared_down_w.awq_scale.is_none()
                && (ctx.arch.is_gfx1100() || ctx.arch.is_gfx1201())
                && p.shared_down_w.m == 2_048
                && p.shared_down_w.k == 512;
            if use_mq4v2_shared_down_fused {
                // Default-off R2 four-row A/B. Exact `1` only; unset/0/invalid stays R1.
                static MQ4V2_SHARED_DOWN_R2: LazyLock<bool> = LazyLock::new(|| {
                    hipfire_config::developer_var("HIPFIRE_MQ4V2_SHARED_DOWN_R2")
                        .map(|value| value == "1")
                        .unwrap_or(false)
                });
                if *MQ4V2_SHARED_DOWN_R2 {
                    hip!(gpu.gemv_mq4g256v2_residual_sigmoid_scaled_k512_r2(
                        &p.shared_down_w.buf,
                        &x_rot_alias,
                        out_target,
                        p.scalar_buf,
                        p.shared_down_w.m,
                        p.shared_down_w.k,
                    ))?;
                } else {
                    hip!(gpu.gemv_mq4g256v2_residual_sigmoid_scaled_k512(
                        &p.shared_down_w.buf,
                        &x_rot_alias,
                        out_target,
                        p.scalar_buf,
                        p.shared_down_w.m,
                        p.shared_down_w.k,
                    ))?;
                }
            } else {
                #[cfg(feature = "deltanet")]
                {
                    hip!(gpu.sigmoid_f32(p.scalar_buf))?;
                    static GEMV_SHARED_V2: OnceLock<GemvFamily> = OnceLock::new();
                    let gemv = GEMV_SHARED_V2.get_or_init(GemvFamily::new);
                    gemv.run(
                        ctx,
                        gpu,
                        &crate::families::gemv::GemvParams {
                            w: &p.shared_down_w,
                            x: &x_rot_alias,
                            y: p.ffn_out,
                            variant: crate::types::GemvVariant::Prerotated,
                            residual: None,
                            gate: None,
                            up: None,
                        },
                    )?;
                    hip!(gpu.scaled_add_inplace_gpu_scalar_f32(
                        out_target,
                        p.ffn_out,
                        p.scalar_buf
                    ))?;
                }
                #[cfg(not(feature = "deltanet"))]
                return Err(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "shared-down-v2-requires-deltanet",
                    arch: "",
                    quant: dtype_name(p.shared_down_w.dtype),
                });
            }
        } else {
            // Non-MQ4 / non-V2 shared expert down. Requires deltanet feature for
            // sigmoid_f32. Returns UnsupportedVariant for builds without the
            // feature to keep hipfire-dispatch compilable without deltanet.
            #[cfg(feature = "deltanet")]
            {
                hip!(gpu.sigmoid_f32(p.scalar_buf))?;
                let shared_hid = unsafe { slice_moe_f32_view(p.ffn_hidden, 0, p.smi) };
                hip!(gpu.silu_mul_f32(&shared_gate, &shared_up, &shared_hid))?;
                static GEMV_DOWN: OnceLock<GemvFamily> = OnceLock::new();
                let gemv = GEMV_DOWN.get_or_init(GemvFamily::new);
                gemv.run_auto(ctx, gpu, &p.shared_down_w, &shared_hid, p.ffn_out)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                hip!(gpu.scaled_add_inplace_gpu_scalar_f32(out_target, p.ffn_out, p.scalar_buf))?;
            }
            #[cfg(not(feature = "deltanet"))]
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "shared-down-non-mq4-requires-deltanet",
                arch: "",
                quant: "",
            });
        }
    }

    // ── Indexed routed experts ────────────────────────────────────────────────
    // Signs back the FWHT used by every MQ4/MQ6 gate_up rotation + silu-rotate
    // (idempotent/cached). Only the paro path is sign-free.
    if !res.routed_indexable_paro {
        hip!(gpu.ensure_mq_signs())?;
    }
    // The activation the routed-expert GEMVs consume.
    //
    // Every rotated dtype gets the FWHT-rotated `x_rot_local`. MQ2G256LloydU is
    // the unrotated sibling — the resolver leaves `needs_x_rot_local` false for
    // it, so `x_rot_local` is None and the correct input is the natural-basis
    // activation. The old unconditional `.expect()` encoded the now-false
    // invariant "use_gpu_topk implies x_rot_local is Some"; keep it as a
    // debug_assert on the rotated branch so a dtype that genuinely needs
    // rotation still trips loudly instead of silently reading unrotated x.
    let xr = match x_rot_local {
        Some(xr) => xr,
        None => {
            debug_assert!(
                !res.needs_x_rot_local,
                "needs_x_rot_local is set but no rotated activation was produced"
            );
            p.x_norm
        }
    };
    let gate_up_k = p.routed_gate_up_k;
    let down_m = p.routed_down_m;
    let down_k = p.routed_down_k;

    // Nine-path fused MoE (NInfer D3+D4 graft, microbenched 1.40× at A3B dims):
    // one x-staged CTA per row tile with 8 routed-expert warps replaces the
    // per-(row,krank) indexed gate_up GEMV, and the down+weighted-combine fold
    // replaces the expanded down GEMV + combine kernel — 3 launches total
    // (D3, silu_rotate, D4) vs 4 with 64× less x restaging. Byte-exact with
    // the replaced kernels by construction (same per-row accumulate order,
    // same fold order). Gated to the measured shape: k=8, uniform MQ4G256,
    // no graded tags, no AWQ, mi=512 (down_k), hidden ≤ 2048 (x LDS stage).
    // Shape rule from .research/microbench/FINDINGS-moe.md: fused wins only
    // for k≥8 with small per-expert intermediate; LFM-class (k=4, I=1792)
    // stays on the chain. HIPFIRE_MOE_NINEPATH=0 opts out.
    static MOE_NINEPATH: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_var("HIPFIRE_MOE_NINEPATH").unwrap_or_default()
    });
    let ninepath_mode = MOE_NINEPATH.as_str();
    // The ninepath down kernel is dtype-specific but shape-generic: it requires
    // k==8, down_k==512 (2 groups), and down_m % RPB == 0. HFQ4 and MQ3-Lloyd
    // each have their own port; the gate_up dtype is irrelevant to it (only the
    // DOWN weights are read here), but it is pinned per-family below so an
    // untested pairing cannot silently select a kernel.
    let ninepath_shape_ok = p.k == 8
        && p.batch_size == 1
        && p.hidden <= 2048
        && p.mi == 512
        && p.expert_dtype_tags.is_none()
        && p.expert_down_awq_ptrs.is_none()
        && !p.defer_routed_combine;
    let ninepath_hfq4 = ninepath_shape_ok
        && p.dtypes.routed_gate_up == DType::MQ4G256
        && p.dtypes.routed_down == DType::MQ4G256;
    // mq2r: routed gate_up MQ2G256Lloyd / down MQ3G256Lloyd. The incumbent down
    // kernel is 16,384 single-wave workgroups at 166 GB/s against gate_up's 543
    // (measured PM4 attribution), because every block re-reads the same rotated
    // activation. This routes it through the same stage-once structure HFQ4 has.
    let ninepath_mq3l = ninepath_shape_ok
        && p.dtypes.routed_gate_up == DType::MQ2G256Lloyd
        && p.dtypes.routed_down == DType::MQ3G256Lloyd;
    // qt44. The published Ornith 1.5 artifact clears `ninepath_shape_ok`
    // exactly (hidden_size 2048, moe_intermediate_size 512,
    // num_experts_per_tok 8) but matched neither family above, so it paid the
    // expanded+combine cost on the shape this path was tuned for.
    let ninepath_mq4v2 = ninepath_shape_ok
        && p.dtypes.routed_gate_up == DType::MQ4G256V2
        && p.dtypes.routed_down == DType::MQ4G256V2;
    // qt47. Same shape gate as qt44; dual-half f16 header is wire-incompatible
    // with V1 MQ6's f32 scale/zero — a fallthrough to HFQ4 ninepath would be
    // silent fluent garbage (200 B stride still "fits").
    let ninepath_mq6v2 = ninepath_shape_ok
        && p.dtypes.routed_gate_up == DType::MQ6G256V2
        && p.dtypes.routed_down == DType::MQ6G256V2;
    let ninepath_eligible = ninepath_hfq4 || ninepath_mq3l || ninepath_mq4v2 || ninepath_mq6v2;
    // Modes: "0"/off = chain; "d3" = D3 only (RESEARCH: 1-ULP codegen
    // divergence from the baseline gate_up — not byte-exact, and slower);
    // "1"/"on" = D3+D4 (research); anything else incl. unset = D4 only
    // (production default: byte-exact with the chain, +0.8% on the A3B
    // serve battery — .research/microbench/FINDINGS-moe.md).
    // Ninepath D3 is HFQ4/MQ4V1 only — V2 (qt44/qt47) must use exact native
    // indexed gate + V2 D4. Same shape but V2 dual-half header is wire-incompatible
    // with the V1 f32 header; a V2 D3 would be silent fluent garbage.
    let ninepath_d3 = ninepath_hfq4 && matches!(ninepath_mode, "1" | "d3" | "on");
    let ninepath_d4 = ninepath_eligible && !matches!(ninepath_mode, "0" | "off" | "d3");

    {
        // ── Routed-expert dispatch via device-indexed merged kernels ──────────
        //
        // Mixed-tier graded quants (mq4p/mq3p/mq4r/mq4rug) carry a per-expert
        // `expert_dtype_tags` table.  The merged kernels
        // (`gemv_mixed_moe_gate_up_k8_indexed_batched` and
        // `gemv_mixed_moe_down_k8_indexed_batched_expanded`) read that table
        // on-device — no D2H, fully hipGraph-capturable — and branch the dequant
        // per-block.  Uniform quants simply have `expert_dtype_tags = None` and
        // fall through to the single-dtype arms below, which is byte-identical to
        // the old pre-SP2 behaviour.
        // Select gate_up + down GEMVs by their INDIVIDUAL dtypes, not a coupled
        // Precedence: mixed gate when gate_up exact dtype varies must run
        // before representative MQ4V2/MQ6V2/V1 arms. Uniform shortcut only
        // when gate_up exact DType equality (per_expert_gate_up uniform).
        let gate_up_varies = gate_up_varies(p.dtypes.per_expert_gate_up.as_deref());
        if ninepath_d3 {
            // Only HFQ4/MQ4V1 may call the V1 ninepath D3 kernel. V2 uses
            // exact native indexed gate (below) + V2 D4.
            debug_assert_eq!(
                ninepath_d3_family(p.dtypes.routed_gate_up, p.dtypes.routed_down),
                Some("hfq4"),
                "ninepath D3 is HFQ4-only"
            );
            hip!(gpu.gemv_hfq4g256_moe_ninepath_d3(
                p.expert_gate_up_ptrs,
                p.topk_indices,
                xr,
                p.gate_batch,
                p.up_batch,
                p.mi,
                gate_up_k,
            ))?;
        } else if decode_gate_uses_mixed(p.expert_dtype_tags.is_some(), gate_up_varies) {
            // Per-expert mixed gate_up (V1/V2 graded, N-tier etc). Must
            // precede the representative MQ4V2/MQ6V2/V1 arms — otherwise a
            // layer with mixed V1+V2 gate dtypes would silently mis-decode
            // the minority tier via the representative's header.
            let tags = p.expert_dtype_tags.expect("mixed gate requires tags");
            hip!(gpu.gemv_mixed_moe_gate_up_k8_indexed_batched(
                p.expert_gate_up_ptrs,
                tags,
                p.topk_indices,
                xr,
                p.gate_batch,
                p.up_batch,
                2 * p.mi,
                gate_up_k,
                p.k,
                1,
            ))?;
        } else if res.routed_indexable_mq4v2 || p.dtypes.routed_gate_up == DType::MQ4G256V2 {
            // qt44. MUST precede the trailing `else`, which dispatches the qt13
            // kernel: qt13 reads bytes [0..8) as one f32 scale + one f32 zero,
            // where qt44 stores two f16 scale/zero pairs. Same 136 B stride, so
            // the misread is silent — fluent text, wrong numbers. Match on the
            // gate_up dtype (not only the coupled flag) so a split pair still
            // selects the V2 gate decoder.
            hip!(gpu.gemv_mq4g256v2_moe_gate_up_k8_indexed(
                p.expert_gate_up_ptrs,
                p.topk_indices,
                xr,
                p.gate_batch,
                p.up_batch,
                2 * p.mi,
                gate_up_k,
            ))?;
        } else if res.routed_indexable_mq6v2 || p.dtypes.routed_gate_up == DType::MQ6G256V2 {
            // qt47. MUST precede HFQ4/HFQ6 arms: dual-half f16 header is
            // incompatible with V1 MQ6 f32 scale/zero; same 200 B stride so a
            // misread is silent fluent corruption.
            hip!(gpu.gemv_mq6g256v2_moe_gate_up_k8_indexed(
                p.expert_gate_up_ptrs,
                p.topk_indices,
                xr,
                p.gate_batch,
                p.up_batch,
                2 * p.mi,
                gate_up_k,
            ))?;
        } else if res.routed_indexable_paro {
            hip!(gpu.gemv_paro_q4g128_moe_gate_up_k8_indexed(
                p.expert_gate_up_ptrs,
                p.topk_indices,
                xr,
                p.gate_batch,
                p.up_batch,
                2 * p.mi,
                gate_up_k,
                p.k,
            ))?;
        } else if matches!(
            p.dtypes.routed_gate_up,
            DType::MQ2G256Lloyd | DType::MQ2G256LloydU
        ) {
            // Uniform MQ2-Lloyd routed experts: ds4/minimax indexed Lloyd gate_up
            // GEMV. y_gate/y_up are separate buffers; m = 2*p.mi (kernel splits at
            // M/2 internally); trailing k_top = p.k. X is the FWHT-rotated xr.
            hip!(gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
                p.expert_gate_up_ptrs,
                p.topk_indices,
                xr,
                p.gate_batch,
                p.up_batch,
                2 * p.mi,
                gate_up_k,
                p.k,
            ))?;
        } else if p.dtypes.routed_gate_up == DType::MQ3G256Lloyd {
            // Uniform MQ3-Lloyd routed experts: same indexed-Lloyd gate_up path,
            // MQ3 launcher.
            hip!(gpu.deepseek4_gemv_mq3g256_lloyd_moe_gate_up_indexed(
                p.expert_gate_up_ptrs,
                p.topk_indices,
                xr,
                p.gate_batch,
                p.up_batch,
                2 * p.mi,
                gate_up_k,
                p.k,
            ))?;
        } else if p.dtypes.routed_gate_up == DType::MQ2G256GL {
            // Uniform MQ2-GL routed gate_up: 2-bit indices against the
            // TENSOR-GLOBAL codebook (GL_CB2, passed as scalar kernel args) plus
            // a per-block fp16 scale, SoA. Same call shape as the MQ2-Lloyd arm
            // above — y_gate/y_up separate, m = 2*p.mi (kernel splits at M/2),
            // X is the FWHT-rotated xr.
            hip!(gpu.gemv_mq2g256gl_moe_gate_up_indexed(
                p.expert_gate_up_ptrs,
                p.topk_indices,
                xr,
                p.gate_batch,
                p.up_batch,
                2 * p.mi,
                gate_up_k,
                p.k,
            ))?;
        } else if p.dtypes.routed_gate_up == DType::MQ3G256GL {
            // Uniform MQ3-GL routed gate_up: same path, 8-entry global codebook
            // (GL_CB3) and 96 B of indices per group.
            hip!(gpu.gemv_mq3g256gl_moe_gate_up_indexed(
                p.expert_gate_up_ptrs,
                p.topk_indices,
                xr,
                p.gate_batch,
                p.up_batch,
                2 * p.mi,
                gate_up_k,
                p.k,
            ))?;
        } else if p.dtypes.routed_gate_up == DType::MQ5G256 {
            hip!(gpu.gemv_hfq5g256_moe_gate_up_k8_indexed(
                p.expert_gate_up_ptrs,
                p.topk_indices,
                xr,
                p.gate_batch,
                p.up_batch,
                2 * p.mi,
                gate_up_k,
            ))?;
        } else if p.dtypes.routed_gate_up == DType::MQ6G256 {
            hip!(gpu.gemv_hfq6g256_moe_gate_up_k8_indexed(
                p.expert_gate_up_ptrs,
                p.topk_indices,
                xr,
                p.gate_batch,
                p.up_batch,
                2 * p.mi,
                gate_up_k,
                p.k,
            ))?;
        } else if p.dtypes.routed_gate_up == DType::MFP4G32E8 {
            // mfp4-E8 grouped experts (gfx1151-only; gated in MoeResolution).
            hip!(gpu.gemv_mfp4g32_e8_moe_gate_up_k8_indexed(
                p.expert_gate_up_ptrs,
                p.topk_indices,
                xr,
                p.gate_batch,
                p.up_batch,
                2 * p.mi,
                gate_up_k,
            ))?;
        } else {
            hip!(gpu.gemv_hfq4g256_moe_gate_up_k8_indexed(
                p.expert_gate_up_ptrs,
                p.topk_indices,
                xr,
                p.gate_batch,
                p.up_batch,
                2 * p.mi,
                gate_up_k,
                p.k,
            ))?;
        }

        // Gate→down: fused silu+mul+rotate
        if res.routed_indexable_paro {
            let paro_down = p
                .routed_down_paro
                .as_ref()
                .expect("routed_indexable_paro implies down paro sidecar");
            hip!(gpu.fused_silu_mul_givens_rotate_f32(
                p.gate_batch,
                p.up_batch,
                p.rot_batch,
                &paro_down.pairs,
                &paro_down.theta,
                &paro_down.scales,
                p.k,
                p.mi,
                paro_down.krot,
            ))?;
        } else if let Some(awq_ptrs) = p.expert_down_awq_ptrs {
            // Route A MoE-AWQ: per-routed-expert down.awq_scale selected by
            // topk_indices[krank]. Divides silu(g)*u by the expert's scale before
            // the FWHT (AWQ math (W·s)·(x/s)=W·x). Only reached on .hfq files
            // carrying per-expert down sidecars — byte-identical otherwise.
            hip!(gpu.fused_silu_mul_rotate_mq_awq_indexed_batched(
                p.gate_batch,
                p.up_batch,
                awq_ptrs,
                p.topk_indices,
                p.rot_batch,
                p.mi,
                p.k,
            ))?;
        } else if gate_down_skips_rotation(p.dtypes.routed_down) {
            // UNROTATED down weights: silu+mul with NO FWHT.
            //
            // Every other arm here fuses the rotation in because its down
            // weights were packed in the FWHT basis. MQ2G256LloydU is the
            // unrotated sibling — rotating the intermediate before an
            // unrotated down GEMV is exactly the silent-garbage failure this
            // dtype exists to avoid, and it would not crash or even look
            // wrong until the model generated text.
            //
            // silu_mul is elementwise, so one launch over the whole
            // [k_top * mi] buffer covers every selected expert.
            hip!(gpu.silu_mul_f32(p.gate_batch, p.up_batch, p.rot_batch))?;
        } else {
            // MQ4/MQ6, no AWQ on expert down weights (the common case for A3B).
            hip!(gpu.fused_silu_mul_rotate_mq_batched(
                p.gate_batch,
                p.up_batch,
                p.rot_batch,
                p.mi,
                p.k
            ))?;
        }

        // Expanded write — down GEMV by the DOWN dtype (mixed mq6-down lands here).
        // FIXME(Step 8): replace hardcoded 1 with p.batch_size when grouped prefill lands
        if ninepath_d4 && ninepath_mq3l {
            // MQ3-Lloyd codebook port. Folds the 8 partials in LDS in ascending
            // krank order (single owner per row, no atomics), so the shared
            // combine below is skipped exactly as it is for the HFQ4 arm.
            hip!(gpu.gemv_mq3g256_lloyd_moe_ninepath_d4(
                p.expert_down_ptrs,
                p.topk_indices,
                p.topk_weights,
                p.rot_batch,
                out_target,
                down_m,
                down_k,
            ))?;
        } else if ninepath_d4 && ninepath_mq4v2 {
            // MUST precede the bare `ninepath_d4` arm below, which calls the
            // qt13 kernel. qt13 and qt44 share a 136 B stride and nibble
            // packing and differ only in the header, so that fallthrough would
            // misread qt44 silently.
            hip!(gpu.gemv_mq4g256v2_moe_ninepath_d4(
                p.expert_down_ptrs,
                p.topk_indices,
                p.topk_weights,
                p.rot_batch,
                out_target,
                down_m,
                down_k,
            ))?;
        } else if ninepath_d4 && ninepath_mq6v2 {
            // qt47. Same silent-header hazard vs HFQ4 ninepath: 200 B stride
            // still "fits" a wrong decoder. Pin the V2 dual-half path first.
            hip!(gpu.gemv_mq6g256v2_moe_ninepath_d4(
                p.expert_down_ptrs,
                p.topk_indices,
                p.topk_weights,
                p.rot_batch,
                out_target,
                down_m,
                down_k,
            ))?;
        } else if ninepath_d4 {
            hip!(gpu.gemv_hfq4g256_moe_ninepath_d4(
                p.expert_down_ptrs,
                p.topk_indices,
                p.topk_weights,
                p.rot_batch,
                out_target,
                down_m,
                down_k,
            ))?;
        } else if let Some(tags) = p.expert_dtype_tags {
            // Per-expert mixed down (graded MQ6 hot / MQ2-Lloyd cold). One
            // merged kernel; block-per-(row,krank,token) reads tags[expert_id]
            // (block-uniform → no warp divergence) and branches the dequant.
            // Writes the EXPANDED buffer for BOTH dtypes → the single shared
            // moe_down_combine_k8_batched runs below (self-combine forced off).
            hip!(gpu.gemv_mixed_moe_down_k8_indexed_batched_expanded(
                p.expert_down_ptrs,
                tags,
                p.topk_indices,
                p.rot_batch,
                p.down_expanded,
                down_m,
                down_k,
                p.k,
                1,
            ))?;
        } else if res.routed_indexable_paro {
            hip!(gpu.gemv_paro_q4g128_moe_down_k8_indexed_batched(
                p.expert_down_ptrs,
                p.topk_indices,
                p.rot_batch,
                p.down_expanded,
                down_m,
                down_k,
                p.k,
                1,
            ))?;
        } else if matches!(
            p.dtypes.routed_down,
            DType::MQ2G256Lloyd | DType::MQ2G256LloydU
        ) {
            // MQ2-Lloyd down: atomic, weighted, SELF-COMBINING residual GEMV.
            // silu-output rotate (rot_batch) -> down -> * topk_weight[krank] ->
            // atomicAdd into out_target, all in one launch. NO separate combine
            // (skipped below). out_target = routed_out (EP zeroed partial) or
            // x_residual; the atomic accumulate is EP-correct unchanged.
            hip!(
                gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
                    p.expert_down_ptrs,
                    p.topk_indices,
                    p.topk_weights,
                    p.rot_batch,
                    out_target,
                    down_m,
                    down_k,
                    p.k,
                    false,
                )
            )?;
        } else if p.dtypes.routed_down == DType::MQ3G256Lloyd {
            // MQ3-Lloyd down: same atomic self-combining residual GEMV, MQ3 launcher.
            hip!(
                gpu.deepseek4_gemv_mq3g256_lloyd_moe_down_residual_scaled_indexed(
                    p.expert_down_ptrs,
                    p.topk_indices,
                    p.topk_weights,
                    p.rot_batch,
                    out_target,
                    down_m,
                    down_k,
                    p.k,
                )
            )?;
        } else if p.dtypes.routed_down == DType::MQ2G256GL {
            // MQ2-GL down: atomic, weighted, SELF-COMBINING residual GEMV —
            // same epilogue contract as the MQ2/MQ3-Lloyd down kernels (one
            // launch does down -> * topk_weight[krank] -> atomicAdd into
            // out_target). NO separate combine; `routed_down_self_combines`
            // below MUST include this dtype or every MoE layer double-counts.
            hip!(gpu.gemv_mq2g256gl_moe_down_residual_scaled_indexed(
                p.expert_down_ptrs,
                p.topk_indices,
                p.topk_weights,
                p.rot_batch,
                out_target,
                down_m,
                down_k,
                p.k,
            ))?;
        } else if p.dtypes.routed_down == DType::MQ3G256GL {
            // MQ3-GL down: same atomic self-combining residual GEMV, 8-entry
            // global codebook.
            hip!(gpu.gemv_mq3g256gl_moe_down_residual_scaled_indexed(
                p.expert_down_ptrs,
                p.topk_indices,
                p.topk_weights,
                p.rot_batch,
                out_target,
                down_m,
                down_k,
                p.k,
            ))?;
        } else if p.dtypes.routed_down == DType::MQ5G256 {
            hip!(gpu.gemv_hfq5g256_moe_down_k8_indexed_batched_expanded(
                p.expert_down_ptrs,
                p.topk_indices,
                p.rot_batch,
                p.down_expanded,
                down_m,
                down_k,
                p.k,
                1,
            ))?;
        } else if p.dtypes.routed_down == DType::MQ4G256V2 {
            // qt44 non-ninepath expanded down. MUST precede the final HFQ4
            // else: same 136 B stride, dual-half header — silent misread.
            hip!(gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
                p.expert_down_ptrs,
                p.topk_indices,
                p.rot_batch,
                p.down_expanded,
                down_m,
                down_k,
                p.k,
                1,
            ))?;
        } else if p.dtypes.routed_down == DType::MQ6G256 {
            hip!(gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
                p.expert_down_ptrs,
                p.topk_indices,
                p.rot_batch,
                p.down_expanded,
                down_m,
                down_k,
                p.k,
                1,
            ))?;
        } else if p.dtypes.routed_down == DType::MQ6G256V2 {
            // qt47 non-ninepath expanded down. Same silent-header hazard vs
            // V1 MQ6 / HFQ4 fallthrough.
            hip!(gpu.gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded(
                p.expert_down_ptrs,
                p.topk_indices,
                p.rot_batch,
                p.down_expanded,
                down_m,
                down_k,
                p.k,
                1,
            ))?;
        } else if p.dtypes.routed_down == DType::MFP4G32E8 {
            // mfp4-E8 grouped expert down (atomic-free expanded; combine below).
            hip!(gpu.gemv_mfp4g32_e8_moe_down_k8_indexed_batched_expanded(
                p.expert_down_ptrs,
                p.topk_indices,
                p.rot_batch,
                p.down_expanded,
                down_m,
                down_k,
                p.k,
                1,
            ))?;
        } else if down_last_combine {
            hip!(gpu.gemv_hfq4g256_moe_down_k8_indexed_last_combine(
                p.expert_down_ptrs,
                p.topk_indices,
                p.rot_batch,
                p.down_expanded,
                p.topk_weights,
                out_target,
                down_m,
                down_k,
                p.k,
                1,
            ))?;
        } else {
            hip!(gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                p.expert_down_ptrs,
                p.topk_indices,
                p.rot_batch,
                p.down_expanded,
                down_m,
                down_k,
                p.k,
                1,
            ))?;
        }
    } // end routed-expert dispatch block

    // FIXME(Step 8): replace hardcoded 1 with p.batch_size when grouped prefill lands
    // EP: routed combine accumulates into `out_target` (the zeroed partial when
    // `routed_out` is set, else `x_residual`). Under EP each rank's non-owned
    // experts read zeroed weights (load-time dummy-fill) → contribute 0, so the
    // all-reduced sum of partials equals the full single-GPU combine.
    // The four CODEBOOK down kernels — MQ2/MQ3-Lloyd and MQ2/MQ3-G256-GL —
    // self-combine via their atomic `_residual_scaled_indexed` GEMV above
    // (weighted accumulate straight into out_target; nothing is written to
    // down_expanded). Running the expanded combine here would double-count the
    // routed contribution (atomic residual + combine of stale down_expanded),
    // so skip it for those down dtypes.
    //
    // This set MUST stay in lockstep with the atomic-down arms in the dispatch
    // chain above. Miss a dtype here and it double-counts; add one whose kernel
    // writes down_expanded instead and it zeroes out. Both are silent numerical
    // corruption with no error, so treat this list as load-bearing.
    //
    // Per-expert mixed mode writes the EXPANDED down buffer for BOTH dtypes
    // (incl. the MQ2-Lloyd experts), so the single shared combine MUST run.
    // Never take the atomic self-combine path here, or the codebook-tier
    // experts double-count (atomic + combine) or zero out (expanded written,
    // combine skipped) — silent numerical corruption. The merged kernel's
    // expanded write replaces the standalone Lloyd atomic GEMV. (There is no GL
    // branch in the merged dtype-tag kernel at all — graded files carrying a GL
    // tier are rejected at load in `hipfire-arch-qwen35::load_moe_ffn`.)
    let routed_down_self_combines = down_last_combine
        || (p.expert_dtype_tags.is_none()
            && matches!(
                p.dtypes.routed_down,
                DType::MQ2G256Lloyd
                    | DType::MQ2G256LloydU
                    | DType::MQ3G256Lloyd
                    | DType::MQ2G256GL
                    | DType::MQ3G256GL
            ));
    if !ninepath_d4 && !routed_down_self_combines && !p.defer_routed_combine {
        hip!(gpu.moe_down_combine_k8_batched(
            p.down_expanded,
            p.topk_weights,
            out_target,
            down_m,
            p.k,
            1
        ))?;
    }

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
fn dtype_name(d: DType) -> &'static str {
    match d {
        DType::MQ4G256 => "MQ4G256",
        DType::MQ4G256V2 => "MQ4G256V2",
        DType::MQ6G256 => "MQ6G256",
        DType::MQ6G256V2 => "MQ6G256V2",
        DType::ParoQ4G128 => "ParoQ4G128",
        DType::Q8_0 => "Q8_0",
        DType::MQ3G256 => "MQ3G256",
        DType::MQ2G256 => "MQ2G256",
        DType::MQ2G256Lloyd => "MQ2G256Lloyd",
        DType::MQ2G256LloydU => "MQ2G256LloydU",
        DType::MQ3G256Lloyd => "MQ3G256Lloyd",
        DType::MQ2G256GL => "MQ2G256GL",
        DType::MQ3G256GL => "MQ3G256GL",
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
/// `shared_gate` / `shared_up` are the gate-side GEMV outputs computed by the
/// caller (`run_moe_decode`), passed through so the shared-expert math is shared.
/// `ctx` is threaded through every inner GEMV (no internal `DispatchCtx::new`).
fn run_moe_decode_cpu_fallback(
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeParams,
    shared_gate: &GpuTensor,
    shared_up: &GpuTensor,
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
    if p.shared_down_w.dtype == DType::MQ4G256 {
        hip!(gpu.ensure_mq_signs())?;
        let x_rot_alias = unsafe {
            GpuTensor {
                buf: gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias(),
                shape: vec![gpu.scratch.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            }
        };
        if let Some(awq) = p.shared_down_w.awq_scale {
            hip!(gpu.fused_silu_mul_rotate_mq_awq(
                shared_gate,
                shared_up,
                awq,
                &x_rot_alias,
                p.smi
            ))?;
        } else {
            hip!(gpu.fused_silu_mul_rotate_mq(shared_gate, shared_up, &x_rot_alias, p.smi))?;
        }
        hip!(gpu.gemv_hfq4g256_residual_sigmoid_scaled_gpu(
            &p.shared_down_w.buf,
            &x_rot_alias,
            p.x_residual,
            p.scalar_buf,
            p.shared_down_w.m,
            p.shared_down_w.k,
        ))?;
    } else if matches!(p.shared_down_w.dtype, DType::MQ4G256V2 | DType::MQ6G256V2) {
        // Exact dense V2 — never residual_sigmoid_scaled HFQ4 (V1 header).
        hip!(gpu.ensure_mq_signs())?;
        let x_rot_alias = unsafe {
            GpuTensor {
                buf: gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias(),
                shape: vec![gpu.scratch.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            }
        };
        if let Some(awq) = p.shared_down_w.awq_scale {
            hip!(gpu.fused_silu_mul_rotate_mq_awq(
                shared_gate,
                shared_up,
                awq,
                &x_rot_alias,
                p.smi
            ))?;
        } else {
            hip!(gpu.fused_silu_mul_rotate_mq(shared_gate, shared_up, &x_rot_alias, p.smi))?;
        }
        // `scalar_buf` has no later read; `ffn_out` is overwritten by the first
        // routed-expert down before being consumed. The exact product shape
        // therefore defaults to the fused arm; `0` or an invalid override
        // retains the generic three-launch route.
        static MQ4V2_SHARED_DOWN_FUSED_FB: LazyLock<bool> = LazyLock::new(|| {
            hipfire_config::developer_var("HIPFIRE_MQ4V2_SHARED_DOWN_FUSED")
                .map(|value| value == "1")
                .unwrap_or(true)
        });
        let use_mq4v2_shared_down_fused = *MQ4V2_SHARED_DOWN_FUSED_FB
            && p.shared_down_w.dtype == DType::MQ4G256V2
            && p.shared_down_w.awq_scale.is_none()
            && (ctx.arch.is_gfx1100() || ctx.arch.is_gfx1201())
            && p.shared_down_w.m == 2_048
            && p.shared_down_w.k == 512;
        if use_mq4v2_shared_down_fused {
            // Default-off R2 four-row A/B. Exact `1` only; unset/0/invalid stays R1.
            static MQ4V2_SHARED_DOWN_R2_FB: LazyLock<bool> = LazyLock::new(|| {
                hipfire_config::developer_var("HIPFIRE_MQ4V2_SHARED_DOWN_R2")
                    .map(|value| value == "1")
                    .unwrap_or(false)
            });
            if *MQ4V2_SHARED_DOWN_R2_FB {
                hip!(gpu.gemv_mq4g256v2_residual_sigmoid_scaled_k512_r2(
                    &p.shared_down_w.buf,
                    &x_rot_alias,
                    p.x_residual,
                    p.scalar_buf,
                    p.shared_down_w.m,
                    p.shared_down_w.k,
                ))?;
            } else {
                hip!(gpu.gemv_mq4g256v2_residual_sigmoid_scaled_k512(
                    &p.shared_down_w.buf,
                    &x_rot_alias,
                    p.x_residual,
                    p.scalar_buf,
                    p.shared_down_w.m,
                    p.shared_down_w.k,
                ))?;
            }
        } else {
            #[cfg(feature = "deltanet")]
            {
                hip!(gpu.sigmoid_f32(p.scalar_buf))?;
                static GEMV_SHARED_V2_FB: OnceLock<GemvFamily> = OnceLock::new();
                let gemv = GEMV_SHARED_V2_FB.get_or_init(GemvFamily::new);
                gemv.run(
                    ctx,
                    gpu,
                    &crate::families::gemv::GemvParams {
                        w: &p.shared_down_w,
                        x: &x_rot_alias,
                        y: p.ffn_out,
                        variant: crate::types::GemvVariant::Prerotated,
                        residual: None,
                        gate: None,
                        up: None,
                    },
                )?;
                hip!(gpu.scaled_add_inplace_gpu_scalar_f32(p.x_residual, p.ffn_out, p.scalar_buf))?;
            }
            #[cfg(not(feature = "deltanet"))]
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "shared-down-v2-requires-deltanet",
                arch: "",
                quant: dtype_name(p.shared_down_w.dtype),
            });
        }
    } else {
        #[cfg(feature = "deltanet")]
        {
            hip!(gpu.sigmoid_f32(p.scalar_buf))?;
            let shared_hid = unsafe { slice_moe_f32_view(p.ffn_hidden, 0, p.smi) };
            hip!(gpu.silu_mul_f32(shared_gate, shared_up, &shared_hid))?;
            static GEMV_DOWN_FB: OnceLock<GemvFamily> = OnceLock::new();
            let gemv = GEMV_DOWN_FB.get_or_init(GemvFamily::new);
            gemv.run_auto(ctx, gpu, &p.shared_down_w, &shared_hid, p.ffn_out)?;
            hip!(gpu.scaled_add_inplace_gpu_scalar_f32(p.x_residual, p.ffn_out, p.scalar_buf))?;
        }
        #[cfg(not(feature = "deltanet"))]
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "shared-down-non-mq4-requires-deltanet",
            arch: "",
            quant: "",
        });
    }

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

    run_moe_decode_selected(gpu, &p.selected())
}

/// Execute the routed-expert decode subgraph after model-owned route
/// selection. Keeping this boundary in the shared family guarantees that the
/// ordinary and heterogeneous DS4 paths use the same gate/up, activation,
/// rotation, down, and combine sequence.
pub fn run_moe_decode_selected(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeSelectedParams,
) -> Result<(), DispatchError> {
    macro_rules! hip {
        ($e:expr) => {
            $e.map_err(|e| DispatchError::Hip(e.to_string()))
        };
    }
    if p.batch_size != 1 {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "selected-decode-requires-batch-1",
            arch: "",
            quant: "",
        });
    }

    // 2. Indexed MQ2-Lloyd gate_up: all k_top experts in one launch
    //    (M = 2*mi; the kernel splits rows r<mi → gate, r>=mi → up).
    if let Some(native) = p.native_mq2_backend {
        hip!(native.gate_up(
            gpu,
            p.expert_gate_up_ptrs,
            p.nonowned_gate_up_dummy,
            p.topk_indices,
            p.x_rot,
            p.gate_batch,
            p.up_batch,
            2 * p.mi,
            p.hidden,
            p.k_top,
        ))?;
    } else {
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
    }

    // 3. Batched silu·mul·clamp (in-place into gate_batch) then batched FWHT rotate.
    hip!(gpu.deepseek4_silu_mul_clamp_f32_batched(
        p.gate_batch,
        p.up_batch,
        p.gate_batch,
        p.mi,
        p.k_top,
        p.swiglu_limit,
    ))?;
    if let Some(native) = p.native_mq2_backend {
        hip!(native.rotate_x_batched(gpu, p.gate_batch, p.rot_batch, p.mi, p.k_top,))?;
    } else {
        hip!(gpu.rotate_x_mq_batched(p.gate_batch, p.rot_batch, p.mi, p.k_top,))?;
    }

    // 4. Indexed MQ2-Lloyd down. Deterministic (default): expanded per-expert
    //    write + fixed-order non-atomic combine into ffn_out — bit-reproducible
    //    for greedy/spec-decode. MOE_DETERMINISTIC=0 uses the faster
    //    atomicAdd-fused path (nondeterministic; bench only).
    let deterministic = !p.uses_atomic_moe_down
        && hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC").as_deref()
            != Ok("0");
    if deterministic {
        if let Some(native) = p.native_mq2_backend {
            hip!(native.down_expanded(
                gpu,
                p.expert_down_ptrs,
                p.expert_gate_up_ptrs,
                p.nonowned_gate_up_dummy,
                p.topk_indices,
                p.rot_batch,
                p.down_expanded,
                p.hidden,
                p.mi,
                p.k_top,
                1,
            ))?;
        } else {
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
        }
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
                p.uses_atomic_moe_down,
            )
        )?;
    }

    Ok(())
}

/// MQ2-Lloyd grouped-GEMM kernel variant (deepseek4 research levers; default
/// `Lloyd4w` on gfx11+, `Base` otherwise). Selected once per gate_up/down call.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GroupedLloydVariant {
    /// Native CDNA3 wave64 MFMA path. Selected only for gfx942 so the
    /// wave32 RDNA WMMA variants below retain their existing routes.
    MfmaGfx942,
    /// i8 WMMA MMQ path (gfx1151): decodes the 2-bit Lloyd index via an int8
    /// codebook LUT and runs i8 WMMA at ~2x the FP16 rate. Top priority when
    /// enabled — ~1.7x the FP16 grouped GEMM on the DeepSeek-V4 prefill shape.
    I8,
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
fn select_grouped_lloyd_variant(
    mfma_gfx942: bool,
    use_lloyd_4w: bool,
    i8: bool,
    n32: bool,
    cnd: bool,
    eightw: bool,
    use_mmqload: bool,
    use_nosync: bool,
) -> GroupedLloydVariant {
    if mfma_gfx942 {
        GroupedLloydVariant::MfmaGfx942
    } else if i8 {
        GroupedLloydVariant::I8
    } else if use_lloyd_4w && n32 {
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

fn use_gfx1151_i8_moe(arch: &str) -> bool {
    arch == "gfx1151"
}

fn use_gfx1151_i8_moe_perm() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MQ2_PERM")
            .ok()
            .as_deref()
            != Some("0")
    })
}

/// Dispatch one MQ2-Lloyd grouped GEMM. All seven variants share the signature
/// `(ptrs, tile_ids, slot_index, x, y, m, k, x_row_div, m_total_max, rows)`, so
/// this is called identically for gate_up (m=2*im, k=hidden, x_row_div=k_top,
/// rows=B) and down (m=hidden, k=im, x_row_div=1, rows=B*k_top).
#[allow(clippy::too_many_arguments)]
fn dispatch_grouped_lloyd(
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
        V::MfmaGfx942 => gpu.gemm_mq2g256_lloyd_moe_grouped_mfma_gfx942(
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
        V::I8 if use_gfx1151_i8_moe_perm() => gpu.gemm_mq2g256_lloyd_moe_grouped_mmq_perm_gfx1151(
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
        V::I8 => gpu.gemm_mq2g256_lloyd_moe_grouped_mmq_gfx1151(
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
    if let Ok(path) = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_DUMP_TOPK") {
        use std::io::Write;
        let raw = hip!(gpu.download_f32(p.topk_indices))?;
        let n = batch_size * k_top;
        let mut indices: Vec<i32> = Vec::with_capacity(n);
        for i in 0..n {
            indices.push(raw[i].to_bits() as i32);
        }
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
    let gate_threshold: usize = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let use_grouped = batch_size >= gate_threshold
        && hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_GROUPED").as_deref() != Ok("0");

    // Shared research levers (read once; default 4w on gfx11+).
    let lloyd_4w_base =
        match hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W").as_deref() {
            Ok("0") => Some(false),
            Ok("1") => Some(true),
            _ => None,
        };
    let arch_4w = gpu.arch.starts_with("gfx11") || gpu.arch.starts_with("gfx12");
    let n32 = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_N32").as_deref() == Ok("1");
    let cnd = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_CND").as_deref() == Ok("1");
    let eightw = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_8W").as_deref() == Ok("1");
    let mmqload_env =
        hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_MMQLOAD").as_deref() == Ok("1");
    let nosync_env =
        hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_NOSYNC").as_deref() == Ok("1");
    // i8 MMQ path (gfx1151 only): 2-bit Lloyd → int8 codebook LUT + i8 WMMA.
    let i8_moe = use_gfx1151_i8_moe(&gpu.arch);

    if use_grouped {
        const BLOCK_M: usize = 16;
        let m_total_max = batch_size * k_top + n_exp * BLOCK_M;

        // Scatter: histogram + offsets + permute (single launch).
        hip!(gpu.moe_scatter_fused_k8(
            p.topk_indices,
            p.expert_token_counts,
            p.expert_offsets,
            p.sorted_slot_index,
            p.expert_tile_ids,
            p.inverse_perm,
            batch_size * k_top,
            n_exp,
            m_total_max,
            BLOCK_M,
        ))?;

        // Grouped gate_up GEMM (M=2*im, K=hidden, x_row_div=k_top, rows=B).
        let use_lloyd_4w_gu =
            lloyd_4w_base.unwrap_or(arch_4w) && (2 * im) % 64 == 0 && hidden % 256 == 0;
        let use_mmqload_gu = use_lloyd_4w_gu && mmqload_env;
        let use_nosync_gu = use_mmqload_gu && nosync_env;
        // i8 path requires (2*im)%16==0 && hidden%256==0 (looser than 4w's %64).
        let use_i8_gu = i8_moe && (2 * im) % 16 == 0 && hidden % 256 == 0;
        let v_gu = select_grouped_lloyd_variant(
            false,
            use_lloyd_4w_gu,
            use_i8_gu,
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
        // Exact gfx1151 defaults to the raw-bit-certified fused consumer: it
        // removes the full up_batch write/read and one launch per routed MoE
        // layer. Other architectures retain the established two-kernel path.
        // The developer override remains an explicit rollback/screening aid;
        // shipping gfx1151 behavior does not depend on an environment flag.
        let use_fused_unscatter_silu = match hipfire_config::developer_var(
            "HIPFIRE_DEEPSEEK4_FUSED_UNSCATTER_SILU",
        )
        .as_deref()
        {
            Ok("0") => false,
            Ok(_) => true,
            Err(_) => gpu.arch.eq_ignore_ascii_case("gfx1151"),
        };
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
                batch_size * k_top,
                p.swiglu_limit,
            ))?;
        }

        // FWHT rotate.
        hip!(gpu.rotate_x_mq_batched(p.gate_batch, p.rot_batch, im, batch_size * k_top))?;

        // Grouped down GEMM (M=hidden, K=im, x_row_div=1, rows=B*k_top).
        let use_lloyd_4w_dn = lloyd_4w_base.unwrap_or(arch_4w) && hidden % 64 == 0 && im % 256 == 0;
        let use_mmqload_dn = use_lloyd_4w_dn && mmqload_env;
        let use_nosync_dn = use_mmqload_dn && nosync_env;
        let use_i8_dn = i8_moe && hidden % 16 == 0 && im % 256 == 0;
        let v_dn = select_grouped_lloyd_variant(
            false,
            use_lloyd_4w_dn,
            use_i8_dn,
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
            batch_size * k_top,
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
        let use_gate_up_k4096_lds = gpu.arch.eq_ignore_ascii_case("gfx1151")
            && 2 * im == 4096
            && hidden == 4096
            && hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_GATE_UP_BATCHED_K4096_LDS")
                .ok()
                .as_deref()
                != Some("0");
        if use_gate_up_k4096_lds {
            hip!(
                gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4096_lds(
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
        } else {
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
        }
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
        let deterministic = !p.uses_atomic_moe_down
            && hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC").as_deref()
                != Ok("0");
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
            // DeepSeek4's routed down projection has fixed K=2048. The ordinary
            // AR route already uses the K8-all body; use its position-batched
            // twin for gfx1151 verify as well. Keep an explicit rollback switch
            // while this route is certified for retained replay.
            let use_batched_k8all = gpu.arch.eq_ignore_ascii_case("gfx1151")
                && im == 2048
                && hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_DOWN_BATCHED_K8ALL")
                    .as_deref()
                    != Ok("0");
            if use_batched_k8all {
                hip!(gpu
                    .deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed_batched_k8all(
                        p.expert_down_ptrs,
                        p.topk_indices,
                        p.topk_weights,
                        p.rot_batch,
                        p.ffn_out,
                        hidden,
                        im,
                        k_top,
                        batch_size,
                    ))?;
            } else {
                hip!(gpu
                    .deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed_batched_k4(
                        p.expert_down_ptrs,
                        p.topk_indices,
                        p.topk_weights,
                        p.rot_batch,
                        p.ffn_out,
                        hidden,
                        im,
                        k_top,
                        batch_size,
                    ))?;
            }
        }
    }

    Ok(())
}

// ── Qwen3.5 batched MoE prefill (Ship 4.2) ──────────────────────────

/// MoE grouped-GEMM block size (WMMA tile row count). Must match the
/// constant in qwen35.rs and the scatter kernel.
const MOE_GROUPED_BLOCK_M: usize = 16;

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
fn dispatch_grouped_gemm(
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
        // Uniform-per-projection codebook routed experts (the antirez asymmetric
        // recipe: gate_up = MQ2-Lloyd 72 B/group, down = MQ3-Lloyd 112 B/group).
        // Both entries are arch-selecting (gfx11 `_k2` / gfx12 `_gfx12`) — do NOT
        // swap either for the bare `_k2` launcher, which fails the JIT on RDNA4.
        DType::MQ2G256Lloyd | DType::MQ2G256LloydU => hip!(gpu
            .gemm_mq2g256_lloyd_moe_grouped_wmma(
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
        // qt44. Routed experts are ~99% of an A3B MoE's tensors, so without
        // this arm a qt44 MoE model cannot prefill at all — it falls into the
        // `_other` error below. Arch-selecting (gfx11 `_k2` / gfx12 `_gfx12`)
        // like the MQ2/MQ3-Lloyd sisters — do NOT swap it for the bare `_k2`
        // launcher, which fails the JIT on RDNA4.
        DType::MQ4G256V2 => hip!(gpu.gemm_mq4g256v2_moe_grouped_wmma_k2(
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
        // qt47. Same as qt44 — without this arm a pure-MQ6V2 MoE cannot Path-2
        // prefill. Arch-selecting launcher (gfx11 `_k2` / gfx12 `_gfx12`).
        DType::MQ6G256V2 => hip!(gpu.gemm_mq6g256v2_moe_grouped_wmma_k2(
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
    use crate::families::moe::MoePrefillResolution;
    macro_rules! hip {
        ($e:expr) => {
            $e.map_err(|e| DispatchError::Hip(e.to_string()))
        };
    }

    let res = MoePrefillResolution::resolve(&p.dtypes, &ctx.arch, &ctx.flags);
    let force_mq4_grouped_fp16 = res.force_mq4_grouped_fp16 || p.force_mq4_grouped_fp16;
    if hipfire_config::developer_var("HIPFIRE_MOE_PREFILL_TRACE")
        .ok()
        .as_deref()
        == Some("1")
    {
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
    let (n, mi, k_top, n_exp) = (p.batch_size, p.mi, p.k_top, p.n_exp);
    let (down_m, down_k, gate_up_k) = (p.down_m, p.down_k, p.gate_up_k);
    let total_slots = n * k_top;

    // EP (Ship 6 substrate-EP prefill): the routed combine accumulates into
    // `out_target` — the zeroed `[batch × dim]` partial when `routed_out` is set
    // (each rank holds only its owned experts; the EP driver all-reduce-sums the
    // partials and adds into `x_batch`), else `x_batch` directly (byte-identical
    // default). The shared expert already accumulated into `x_batch` upstream and
    // is NOT redirected (replicated per rank). Under EP the non-owned experts
    // read load-time zero-dummy weights → contribute 0, so the all-reduced sum of
    // partials equals the full single-GPU routed combine.
    let out_target: &GpuTensor = p.routed_out.unwrap_or(p.x_batch);

    // ── Path 2 scatter pipeline ───────────────────────────────────────
    let mut path2_m_total: usize = 0;
    if res.use_path2 {
        let m_total_max = p.m_total_max;
        hip!(gpu.moe_scatter_fused_k8(
            p.topk_indices,
            p.expert_token_counts,
            p.expert_offsets,
            p.sorted_slot_index,
            p.expert_tile_ids,
            p.inverse_perm,
            total_slots,
            n_exp,
            m_total_max,
            MOE_GROUPED_BLOCK_M,
        ))?;
        path2_m_total = m_total_max;
    }

    // ── Gate_up ────────────────────────────────────────────────────────
    if res.use_path2 {
        // Path 2: grouped-WMMA-GEMM. Paro gate_up Givens preamble in-line
        // (above the helper — D3).
        if res.paro_mode {
            let paro = p
                .paro_gate_up
                .as_ref()
                .expect("paro_mode implies paro_gate_up sidecar");
            hip!(gpu.givens_rotate_to(
                p.x_norm_batch,
                p.x_rot_batch,
                paro.pairs,
                paro.theta,
                paro.scales,
                n,
                gate_up_k, /* hidden dim */
                paro.krot,
            ))?;
        }
        // Down-only-graded redline: the tag table describes the DOWN dtypes, so
        // for a UNIFORM gate_up (exact DType equality) it must NOT be passed
        // here (the mixed grouped kernel would read MQ4 gate_up bytes with
        // the down's MQ6/MQ3L tags → garbage). Pass None → the uniform
        // grouped kernel. When gate_up exact dtype VARIES (V1/V2 graded,
        // N-tier), preserve tags — mixed gate before representative.
        // Mirrors the decode gate_up fix and satisfies issue 1's
        // "Grouped Path2 preserves tags on V1/V2 variation".
        let gate_up_varies = gate_up_varies(p.dtypes.per_expert_gate_up.as_deref());
        let gate_up_tags = if gate_up_varies {
            p.expert_dtype_tags
        } else {
            None
        };
        dispatch_grouped_gemm(
            gpu,
            p.dtypes.routed_gate_up,
            gate_up_tags,
            p.expert_gate_up_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            p.x_rot_batch,
            p.y_gate_up_grouped,
            2 * mi,
            gate_up_k,
            k_top,
            path2_m_total,
            n,
            force_mq4_grouped_fp16,
            res.use_paro_i8,
            res.use_paro_i8_k8,
        )?;
        hip!(gpu.moe_gate_up_unscatter_k8(
            p.y_gate_up_grouped,
            p.sorted_slot_index,
            p.gate_batch,
            p.up_batch,
            mi,
            k_top,
            path2_m_total,
        ))?;
    } else {
        // Path 1 fallback: per-token indexed GEMV, batched over N tokens.
        // Mixed Path1 (issue 2): when tags exist, use mixed batched gate
        // launcher; do not representative-dispatch tagged layers. Gate uses
        // mixed only when gate_up exact dtype varies (V1/V2 graded); uniform
        // gate shortcut requires exact DType equality. Down always uses mixed
        // when tags exist (handled in the down block below).
        if res.paro_mode {
            let paro = p
                .paro_gate_up
                .as_ref()
                .expect("paro_mode implies paro_gate_up sidecar");
            hip!(gpu.givens_rotate_to(
                p.x_norm_batch,
                p.x_rot_batch,
                paro.pairs,
                paro.theta,
                paro.scales,
                n,
                gate_up_k,
                paro.krot,
            ))?;
            hip!(gpu.gemv_paro_q4g128_moe_gate_up_k8_indexed_batched(
                p.expert_gate_up_ptrs,
                p.topk_indices,
                p.x_rot_batch,
                p.gate_batch,
                p.up_batch,
                2 * mi,
                gate_up_k,
                k_top,
                n,
            ))?;
        } else if let Some(tags) = p.expert_dtype_tags {
            let gate_up_varies = gate_up_varies(p.dtypes.per_expert_gate_up.as_deref());
            if decode_gate_uses_mixed(true, gate_up_varies) {
                // Gate varies → mixed gate kernel before representative arms.
                hip!(gpu.gemv_mixed_moe_gate_up_k8_indexed_batched(
                    p.expert_gate_up_ptrs,
                    tags,
                    p.topk_indices,
                    p.x_rot_batch,
                    p.gate_batch,
                    p.up_batch,
                    2 * p.mi,
                    gate_up_k,
                    k_top,
                    n,
                ))?;
            } else {
                // Gate uniform (exact equality) → representative uniform gate.
                let gate_up_result = match p.dtypes.routed_gate_up {
                    DType::MQ4G256 => hip!(gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
                        p.expert_gate_up_ptrs,
                        p.topk_indices,
                        p.x_rot_batch,
                        p.gate_batch,
                        p.up_batch,
                        2 * mi,
                        gate_up_k,
                        k_top,
                        n,
                    )),
                    DType::MQ4G256V2 => hip!(gpu.gemv_mq4g256v2_moe_gate_up_k8_indexed_batched(
                        p.expert_gate_up_ptrs,
                        p.topk_indices,
                        p.x_rot_batch,
                        p.gate_batch,
                        p.up_batch,
                        2 * mi,
                        gate_up_k,
                        k_top,
                        n,
                    )),
                    DType::MQ5G256 => hip!(gpu.gemv_hfq5g256_moe_gate_up_k8_indexed_batched(
                        p.expert_gate_up_ptrs,
                        p.topk_indices,
                        p.x_rot_batch,
                        p.gate_batch,
                        p.up_batch,
                        2 * mi,
                        gate_up_k,
                        k_top,
                        n,
                    )),
                    DType::MQ6G256 => hip!(gpu.gemv_hfq6g256_moe_gate_up_k8_indexed_batched(
                        p.expert_gate_up_ptrs,
                        p.topk_indices,
                        p.x_rot_batch,
                        p.gate_batch,
                        p.up_batch,
                        2 * mi,
                        gate_up_k,
                        k_top,
                        n,
                    )),
                    DType::MQ6G256V2 => hip!(gpu.gemv_mq6g256v2_moe_gate_up_k8_indexed_batched(
                        p.expert_gate_up_ptrs,
                        p.topk_indices,
                        p.x_rot_batch,
                        p.gate_batch,
                        p.up_batch,
                        2 * mi,
                        gate_up_k,
                        k_top,
                        n,
                    )),
                    DType::MFP4G32E8 => hip!(gpu.gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched(
                        p.expert_gate_up_ptrs,
                        p.topk_indices,
                        p.x_rot_batch,
                        p.gate_batch,
                        p.up_batch,
                        2 * mi,
                        gate_up_k,
                        k_top,
                        n,
                    )),
                    _other => {
                        return Err(DispatchError::UnsupportedVariant {
                            family: "moe",
                            variant: "prefill-gate-up-path1-dtype",
                            arch: "",
                            quant: "other",
                        });
                    }
                };
                gate_up_result?;
            }
        } else {
            // MQ4/MQ6 indexed batched GEMV (x_rot_batch is already FWHT-rotated
            // by the model). Uniform path, no tags.
            let gate_up_result = match p.dtypes.routed_gate_up {
                DType::MQ4G256 => hip!(gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
                    p.expert_gate_up_ptrs,
                    p.topk_indices,
                    p.x_rot_batch,
                    p.gate_batch,
                    p.up_batch,
                    2 * mi,
                    gate_up_k,
                    k_top,
                    n,
                )),
                DType::MQ4G256V2 => hip!(gpu.gemv_mq4g256v2_moe_gate_up_k8_indexed_batched(
                    p.expert_gate_up_ptrs,
                    p.topk_indices,
                    p.x_rot_batch,
                    p.gate_batch,
                    p.up_batch,
                    2 * mi,
                    gate_up_k,
                    k_top,
                    n,
                )),
                DType::MQ5G256 => hip!(gpu.gemv_hfq5g256_moe_gate_up_k8_indexed_batched(
                    p.expert_gate_up_ptrs,
                    p.topk_indices,
                    p.x_rot_batch,
                    p.gate_batch,
                    p.up_batch,
                    2 * mi,
                    gate_up_k,
                    k_top,
                    n,
                )),
                DType::MQ6G256 => hip!(gpu.gemv_hfq6g256_moe_gate_up_k8_indexed_batched(
                    p.expert_gate_up_ptrs,
                    p.topk_indices,
                    p.x_rot_batch,
                    p.gate_batch,
                    p.up_batch,
                    2 * mi,
                    gate_up_k,
                    k_top,
                    n,
                )),
                DType::MQ6G256V2 => hip!(gpu.gemv_mq6g256v2_moe_gate_up_k8_indexed_batched(
                    p.expert_gate_up_ptrs,
                    p.topk_indices,
                    p.x_rot_batch,
                    p.gate_batch,
                    p.up_batch,
                    2 * mi,
                    gate_up_k,
                    k_top,
                    n,
                )),
                DType::MFP4G32E8 => hip!(gpu.gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched(
                    p.expert_gate_up_ptrs,
                    p.topk_indices,
                    p.x_rot_batch,
                    p.gate_batch,
                    p.up_batch,
                    2 * mi,
                    gate_up_k,
                    k_top,
                    n,
                )),
                _other => {
                    return Err(DispatchError::UnsupportedVariant {
                        family: "moe",
                        variant: "prefill-gate-up-path1-dtype",
                        arch: "",
                        quant: "other",
                    });
                }
            };
            gate_up_result?;
        }
    }

    // ── SwiGLU + rotate over [N*K_TOP × mi] ────────────────────────────
    if res.paro_mode {
        let paro = p
            .paro_down
            .as_ref()
            .expect("paro_mode implies paro_down sidecar");
        hip!(gpu.fused_silu_mul_givens_rotate_f32(
            p.gate_batch,
            p.up_batch,
            p.rot_batch,
            paro.pairs,
            paro.theta,
            paro.scales,
            total_slots,
            mi,
            paro.krot,
        ))?;
    } else if p.expert_dtype_tags.is_some() {
        // Graded/mixed routed experts: the silu+rotate is weight-agnostic (the
        // per-expert down dtype only affects the down GEMM that READS rot_batch;
        // graded files carry no expert AWQ). This mirrors run_moe_decode, which
        // calls this unconditionally. Without this, the routed_down dtype match
        // below rejects the cold-tier Lloyd dtype (experts[0].down) as `_other`
        // and the prefill forward panics.
        hip!(gpu.fused_silu_mul_rotate_mq_batched(
            p.gate_batch,
            p.up_batch,
            p.rot_batch,
            mi,
            total_slots,
        ))?;
    } else {
        // MQ4/MQ6: the silu+rotate kernel is weight-agnostic (reads only
        // activations, not weight data). AWQ-aware variant when down has AWQ.
        match p.dtypes.routed_down {
            // MFP4G32E8 reuses the weight-agnostic silu+FWHT-rotate (E8 down expects
            // FWHT(silu(g)*u), same as MQ4 — see the decode E8 path).
            //
            // MQ2/MQ3-Lloyd likewise: the decode path feeds those down GEMVs
            // `rot_batch` = FWHT(silu(g)·u) built by the same weight-agnostic
            // kernel family, so the batched twin is the correct analogue. This
            // was a pure MATCH gap — the kernel reads activations only, never
            // weight bytes. The GL dtypes are deliberately absent: they are not
            // batched-prefill admissible (no grouped GEMM, no batched GEMV), so
            // reaching here with a GL down is a bug and must stay a loud error.
            // qt44 belongs here for the same reason as every other MQ dtype:
            // the silu/rotate kernels below consume f32 gate_batch/up_batch
            // activations and never touch quantized weights, so this match is
            // selecting a branch, not a decode path.
            DType::MQ4G256
            | DType::MQ4G256V2
            | DType::MQ5G256
            | DType::MQ6G256
            | DType::MQ6G256V2
            | DType::MQ2G256Lloyd
            | DType::MQ3G256Lloyd
            | DType::MFP4G32E8
            | DType::MFP3G32E8
            | DType::MFP2G32E8 => {
                if let Some(awq_ptrs) = p.expert_down_awq_ptrs {
                    // Route A MoE-AWQ (per-routed-expert, indexed by topk slot).
                    // total_slots rows = N·k_top; each slot's expert is
                    // topk_indices[slot] — the same slot→expert mapping the
                    // indexed down GEMV below uses. Supersedes the single-scale
                    // `down_awq_scale` (Ship 4.2 stub) which incorrectly applied
                    // experts[0]'s scale to every routed slot.
                    //
                    // NOTE: correct for the indexed batched gate_up (Path 0/1,
                    // gfx9*/non-grouped) where rot_batch[slot] aligns with
                    // topk_indices[slot]. Path 2 grouped-WMMA (gfx11/gfx12)
                    // reorders via sorted_slot_index — AWQ+Path2 ordering is
                    // unverified; the only current MoE-AWQ target is A3B on
                    // gfx942 (Path 0). See docs/moe-awq/MOE_AWQ_EXPERTS.md.
                    hip!(gpu.fused_silu_mul_rotate_mq_awq_indexed_batched(
                        p.gate_batch,
                        p.up_batch,
                        awq_ptrs,
                        p.topk_indices,
                        p.rot_batch,
                        mi,
                        total_slots,
                    ))?;
                } else if let Some(awq) = p.down_awq_scale {
                    hip!(gpu.fused_silu_mul_rotate_mq_awq_batched(
                        p.gate_batch,
                        p.up_batch,
                        awq,
                        p.rot_batch,
                        mi,
                        total_slots,
                    ))?;
                } else {
                    hip!(gpu.fused_silu_mul_rotate_mq_batched(
                        p.gate_batch,
                        p.up_batch,
                        p.rot_batch,
                        mi,
                        total_slots,
                    ))?;
                }
            }
            _other => {
                return Err(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "prefill-silu-rotate-dtype",
                    arch: "",
                    quant: "other",
                });
            }
        }
    }

    // ── Down projection ───────────────────────────────────────────────
    if res.use_path2 {
        // Path 2: grouped-WMMA-GEMM + non-atomic combine via inverse_perm.
        dispatch_grouped_gemm(
            gpu,
            p.dtypes.routed_down,
            p.expert_dtype_tags,
            p.expert_down_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            p.rot_batch,
            p.y_down_grouped,
            down_m,
            down_k,
            1, /* x_row_div */
            path2_m_total,
            total_slots,
            force_mq4_grouped_fp16,
            res.use_paro_i8,
            res.use_paro_i8_k8,
        )?;
        hip!(gpu.moe_down_combine_grouped_k8(
            p.y_down_grouped,
            p.inverse_perm,
            p.topk_weights,
            out_target,
            down_m,
            k_top,
            n,
        ))?;
    } else if res.down_path0 {
        // Path 0: gfx9* wave64 — residual-scaled atomic GEMV (MQ4 only;
        // MQ6/Paro never reach here — their admit predicates require WMMA).
        let down_result = match p.dtypes.routed_down {
            DType::MQ4G256 => hip!(
                gpu.gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched(
                    p.expert_down_ptrs,
                    p.topk_indices,
                    p.topk_weights,
                    p.rot_batch,
                    out_target,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )
            ),
            _other => {
                return Err(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "prefill-down-path0-dtype",
                    arch: "",
                    quant: "other",
                });
            }
        };
        down_result?;
    } else {
        // Path 1: atomic-free expanded GEMV write + combine.
        // Mixed Path1 (issue 2): when `expert_dtype_tags` exists, use the
        // mixed batched down launcher (`gemv_mixed_moe_down_k8_indexed_batched_expanded`);
        // do not representative-dispatch tagged layers. Otherwise dispatch
        // the uniform dtype's native batched expanded kernel or reject loudly.
        if let Some(tags) = p.expert_dtype_tags {
            hip!(gpu.gemv_mixed_moe_down_k8_indexed_batched_expanded(
                p.expert_down_ptrs,
                tags,
                p.topk_indices,
                p.rot_batch,
                p.down_expanded,
                down_m,
                down_k,
                k_top,
                n,
            ))?;
        } else {
            let down_result = match p.dtypes.routed_down {
                DType::MQ4G256 => hip!(gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                    p.expert_down_ptrs,
                    p.topk_indices,
                    p.rot_batch,
                    p.down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )),
                DType::MQ4G256V2 => hip!(gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
                    p.expert_down_ptrs,
                    p.topk_indices,
                    p.rot_batch,
                    p.down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )),
                DType::MQ5G256 => hip!(gpu.gemv_hfq5g256_moe_down_k8_indexed_batched_expanded(
                    p.expert_down_ptrs,
                    p.topk_indices,
                    p.rot_batch,
                    p.down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )),
                DType::MQ6G256 => hip!(gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
                    p.expert_down_ptrs,
                    p.topk_indices,
                    p.rot_batch,
                    p.down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )),
                DType::MQ6G256V2 => hip!(gpu.gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded(
                    p.expert_down_ptrs,
                    p.topk_indices,
                    p.rot_batch,
                    p.down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )),
                DType::MFP4G32E8 => hip!(gpu.gemv_mfp4g32_e8_moe_down_k8_indexed_batched_expanded(
                    p.expert_down_ptrs,
                    p.topk_indices,
                    p.rot_batch,
                    p.down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )),
                DType::ParoQ4G128 => hip!(gpu.gemv_paro_q4g128_moe_down_k8_indexed_batched(
                    p.expert_down_ptrs,
                    p.topk_indices,
                    p.rot_batch,
                    p.down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )),
                _other => {
                    return Err(DispatchError::UnsupportedVariant {
                        family: "moe",
                        variant: "prefill-down-path1-dtype",
                        arch: "",
                        quant: "other",
                    });
                }
            };
            down_result?;
        }
        hip!(gpu.moe_down_combine_k8_batched(
            p.down_expanded,
            p.topk_weights,
            out_target,
            down_m,
            k_top,
            n,
        ))?;
    }

    Ok(())
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
    use super::{
        build_contiguous_permutation, decode_expanded_down_kind, decode_gate_uses_mixed,
        gate_down_skips_rotation, gate_up_varies, grouped_gemm_kind, mixed_expert_dtype_tag,
        ninepath_d3_family, ninepath_d4_family, prefill_path1_down_kind,
        prefill_path1_down_kind_tag_aware, prefill_path1_gate_up_kind,
        prefill_path1_gate_up_kind_tag_aware, shared_dense_down_kind, use_gfx1151_i8_moe,
    };
    use crate::families::moe_buckets::bucket_topk_by_tier;
    use rdna_compute::DType::*;

    #[test]
    fn deepseek4_i8_moe_is_exact_gfx1151_only() {
        assert!(use_gfx1151_i8_moe("gfx1151"));
        for arch in ["gfx1100", "gfx1150", "gfx1152", "gfx1200", "gfx1201"] {
            assert!(!use_gfx1151_i8_moe(arch), "unexpected i8 route on {arch}");
        }
    }

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

    // ── MQV2 MoE wiring predicates (decode / prefill / shared / mixed tags) ──

    #[test]
    fn ninepath_d4_separates_v1_and_v2_families() {
        assert_eq!(ninepath_d4_family(MQ4G256, MQ4G256), Some("hfq4"));
        assert_eq!(ninepath_d4_family(MQ4G256V2, MQ4G256V2), Some("mq4v2"));
        assert_eq!(ninepath_d4_family(MQ6G256V2, MQ6G256V2), Some("mq6v2"));
        // Split V1/V2 pairs never ride a ninepath family.
        assert_eq!(ninepath_d4_family(MQ4G256V2, MQ4G256), None);
        assert_eq!(ninepath_d4_family(MQ4G256, MQ4G256V2), None);
        assert_eq!(ninepath_d4_family(MQ6G256V2, MQ6G256), None);
        assert_eq!(ninepath_d4_family(MQ4G256V2, MQ6G256V2), None);
    }

    #[test]
    fn decode_expanded_down_never_aliases_v2_to_v1() {
        assert_eq!(decode_expanded_down_kind(MQ4G256), Some("hfq4"));
        assert_eq!(decode_expanded_down_kind(MQ4G256V2), Some("mq4v2"));
        assert_eq!(decode_expanded_down_kind(MQ6G256), Some("hfq6"));
        assert_eq!(decode_expanded_down_kind(MQ6G256V2), Some("mq6v2"));
        // V2 kinds are distinct string identities from V1.
        assert_ne!(
            decode_expanded_down_kind(MQ4G256V2),
            decode_expanded_down_kind(MQ4G256)
        );
        assert_ne!(
            decode_expanded_down_kind(MQ6G256V2),
            decode_expanded_down_kind(MQ6G256)
        );
    }

    #[test]
    fn path1_batched_admits_mq4v2_and_mq6v2() {
        assert_eq!(prefill_path1_gate_up_kind(MQ4G256V2), Some("mq4v2"));
        assert_eq!(prefill_path1_gate_up_kind(MQ6G256V2), Some("mq6v2"));
        assert_eq!(prefill_path1_down_kind(MQ4G256V2), Some("mq4v2"));
        assert_eq!(prefill_path1_down_kind(MQ6G256V2), Some("mq6v2"));
        // V1 remains on its own kinds.
        assert_eq!(prefill_path1_gate_up_kind(MQ4G256), Some("hfq4"));
        assert_eq!(prefill_path1_gate_up_kind(MQ6G256), Some("hfq6"));
    }

    #[test]
    fn grouped_gemm_admits_mq6v2() {
        assert_eq!(grouped_gemm_kind(MQ4G256V2), Some("mq4v2"));
        assert_eq!(grouped_gemm_kind(MQ6G256V2), Some("mq6v2"));
        assert_eq!(grouped_gemm_kind(MQ4G256), Some("hfq4"));
        assert_eq!(grouped_gemm_kind(MQ6G256), Some("hfq6"));
        assert_ne!(grouped_gemm_kind(MQ6G256V2), grouped_gemm_kind(MQ6G256));
    }

    #[test]
    fn shared_dense_v2_never_uses_v1_sigmoid_scaled() {
        assert_eq!(shared_dense_down_kind(MQ4G256), Some("hfq4_sigmoid_scaled"));
        assert_eq!(shared_dense_down_kind(MQ4G256V2), Some("mq4v2_prerotated"));
        assert_eq!(shared_dense_down_kind(MQ6G256V2), Some("mq6v2_prerotated"));
        assert_ne!(
            shared_dense_down_kind(MQ4G256V2),
            shared_dense_down_kind(MQ4G256)
        );
    }

    #[test]
    fn frozen_mixed_tags_7_to_18_are_v2_identities() {
        // Uniform V2 pairs.
        assert_eq!(mixed_expert_dtype_tag(MQ4G256V2, MQ4G256V2), Some(7));
        assert_eq!(mixed_expert_dtype_tag(MQ6G256V2, MQ6G256V2), Some(8));
        // Split V2/V1 and V1/V2 pairs occupy 9..18 and NEVER collapse to 0..6.
        assert_eq!(mixed_expert_dtype_tag(MQ4G256V2, MQ6G256), Some(9));
        assert_eq!(mixed_expert_dtype_tag(MQ4G256V2, MQ2G256Lloyd), Some(10));
        assert_eq!(mixed_expert_dtype_tag(MQ4G256V2, MQ4G256), Some(11));
        assert_eq!(mixed_expert_dtype_tag(MQ4G256, MQ4G256V2), Some(12));
        assert_eq!(mixed_expert_dtype_tag(MQ4G256V2, MQ3G256Lloyd), Some(13));
        assert_eq!(mixed_expert_dtype_tag(MQ4G256V2, MFP4G32E8), Some(14));
        assert_eq!(mixed_expert_dtype_tag(MQ4G256V2, MFP3G32E8), Some(15));
        assert_eq!(mixed_expert_dtype_tag(MQ4G256V2, MFP2G32E8), Some(16));
        assert_eq!(mixed_expert_dtype_tag(MQ4G256V2, MQ6G256V2), Some(17));
        assert_eq!(mixed_expert_dtype_tag(MQ4G256, MQ6G256V2), Some(18));
        // V1 pairs keep 0..6.
        assert_eq!(mixed_expert_dtype_tag(MQ4G256, MQ4G256), Some(2));
        assert_eq!(mixed_expert_dtype_tag(MQ4G256, MQ6G256), Some(0));
        // Unknown / GL pairs refuse loudly (None).
        assert_eq!(mixed_expert_dtype_tag(MQ6G256V2, MQ4G256V2), None);
        assert_eq!(mixed_expert_dtype_tag(MQ2G256GL, MQ4G256), None);
        assert_eq!(mixed_expert_dtype_tag(MQ4G256, MQ2G256GL), None);
        // No V2 pair may share a tag with a V1 pair.
        for tag in 7u8..=18 {
            let v1_tags: [u8; 7] = [0, 1, 2, 3, 4, 5, 6];
            assert!(!v1_tags.contains(&tag));
        }
    }

    #[test]
    fn unrotated_dtype_still_skips_rotation() {
        assert!(gate_down_skips_rotation(MQ2G256LloydU));
        assert!(!gate_down_skips_rotation(MQ4G256V2));
        assert!(!gate_down_skips_rotation(MQ6G256V2));
    }

    #[test]
    fn ninepath_d3_is_hfq4_only_and_v2_uses_native_d4() {
        // Only HFQ4/MQ4V1 (MQ4G256) may call the V1 ninepath D3 kernel.
        assert_eq!(ninepath_d3_family(MQ4G256, MQ4G256), Some("hfq4"));
        // V2 pairs must NOT use D3 — they use native indexed gate + V2 D4.
        assert_eq!(ninepath_d3_family(MQ4G256V2, MQ4G256V2), None);
        assert_eq!(ninepath_d3_family(MQ6G256V2, MQ6G256V2), None);
        // Split V1/V2 pairs never share D3.
        assert_eq!(ninepath_d3_family(MQ4G256V2, MQ4G256), None);
        assert_eq!(ninepath_d3_family(MQ4G256, MQ4G256V2), None);
        assert_eq!(ninepath_d3_family(MQ4G256V2, MQ6G256V2), None);
        // D4 still separates V2 families correctly.
        assert_eq!(ninepath_d4_family(MQ4G256V2, MQ4G256V2), Some("mq4v2"));
        assert_eq!(ninepath_d4_family(MQ6G256V2, MQ6G256V2), Some("mq6v2"));
    }

    #[test]
    fn gate_up_varies_is_exact_dtype_equality() {
        // None => uniform (no tags).
        assert!(!gate_up_varies(None));
        // All-equal Some => uniform.
        assert!(!gate_up_varies(Some(&[MQ4G256, MQ4G256, MQ4G256])));
        assert!(!gate_up_varies(Some(&[MQ4G256V2, MQ4G256V2])));
        // Exact DType variation => true (V1 vs V2 is a variation).
        assert!(gate_up_varies(Some(&[MQ4G256, MQ4G256V2])));
        assert!(gate_up_varies(Some(&[MQ4G256V2, MQ6G256V2])));
        assert!(gate_up_varies(Some(&[MQ4G256, MQ6G256])));
        // Single element => uniform.
        assert!(!gate_up_varies(Some(&[MQ6G256V2])));
    }

    #[test]
    fn mixed_gate_precedence_before_representative_v1_v2_arms() {
        // Whenever gate_up exact dtype varies, mixed gate must be chosen
        // before representative MQ4V2/MQ6V2/V1 arms. Uniform shortcut only
        // when exact equality.
        // Gate varies + tags => mixed.
        assert!(decode_gate_uses_mixed(true, true));
        // Gate uniform + tags => uniform (down-only-graded optimization).
        assert!(!decode_gate_uses_mixed(true, false));
        // No tags => never mixed, even if varies flag somehow true.
        assert!(!decode_gate_uses_mixed(false, true));
        assert!(!decode_gate_uses_mixed(false, false));
        // Prefill Path1 tag-aware gate mirrors the same precedence.
        assert_eq!(
            prefill_path1_gate_up_kind_tag_aware(MQ4G256V2, true, true),
            Some("mixed")
        );
        assert_eq!(
            prefill_path1_gate_up_kind_tag_aware(MQ4G256, true, true),
            Some("mixed")
        );
        // Uniform gate (exact equality) uses representative kind, not mixed.
        assert_eq!(
            prefill_path1_gate_up_kind_tag_aware(MQ4G256, true, false),
            Some("hfq4")
        );
        assert_eq!(
            prefill_path1_gate_up_kind_tag_aware(MQ4G256V2, true, false),
            Some("mq4v2")
        );
        assert_eq!(
            prefill_path1_gate_up_kind_tag_aware(MQ6G256V2, true, false),
            Some("mq6v2")
        );
        // No tags => uniform regardless of varies flag.
        assert_eq!(
            prefill_path1_gate_up_kind_tag_aware(MQ4G256, false, false),
            Some("hfq4")
        );
        assert_eq!(
            prefill_path1_gate_up_kind_tag_aware(MQ4G256V2, false, false),
            Some("mq4v2")
        );
    }

    #[test]
    fn mixed_down_path1_tag_aware_uses_mixed_when_tagged() {
        // Mixed Path1 down: when tags exist, use mixed batched down launcher;
        // do not representative-dispatch tagged layers.
        assert_eq!(
            prefill_path1_down_kind_tag_aware(MQ4G256, true),
            Some("mixed")
        );
        assert_eq!(
            prefill_path1_down_kind_tag_aware(MQ4G256V2, true),
            Some("mixed")
        );
        assert_eq!(
            prefill_path1_down_kind_tag_aware(MQ6G256V2, true),
            Some("mixed")
        );
        // No tags => uniform native kind or loud error (None) is delegated.
        assert_eq!(
            prefill_path1_down_kind_tag_aware(MQ4G256, false),
            Some("hfq4")
        );
        assert_eq!(
            prefill_path1_down_kind_tag_aware(MQ4G256V2, false),
            Some("mq4v2")
        );
        assert_eq!(
            prefill_path1_down_kind_tag_aware(MQ6G256V2, false),
            Some("mq6v2")
        );
        // Unknown dtype without tags should be None (loud reject).
        assert_eq!(prefill_path1_down_kind_tag_aware(MQ2G256Lloyd, false), None);
        // Tagged layer with unsupported pair still returns mixed (the mixed
        // kernel itself will reject loudly per tag table) — never collapses
        // to representative V1.
        assert_eq!(
            prefill_path1_down_kind_tag_aware(MQ2G256Lloyd, true),
            Some("mixed")
        );
    }

    #[test]
    fn grouped_path2_preserves_tags_on_v1_v2_variation() {
        // Grouped Path2 must preserve tags when gate_up exact dtype varies
        // (V1/V2 graded). Uniform shortcut only with exact DType equality.
        // Simulate the dispatch decision: tags preserved iff gate_up_varies.
        let uniform_mq4 = Some(vec![MQ4G256, MQ4G256, MQ4G256]);
        let uniform_v2 = Some(vec![MQ4G256V2, MQ4G256V2]);
        let mixed_v1_v2 = Some(vec![MQ4G256, MQ4G256V2]);
        let mixed_mq4_mq6 = Some(vec![MQ4G256V2, MQ6G256V2]);
        assert!(!gate_up_varies(uniform_mq4.as_deref()));
        assert!(!gate_up_varies(uniform_v2.as_deref()));
        assert!(gate_up_varies(mixed_v1_v2.as_deref()));
        assert!(gate_up_varies(mixed_mq4_mq6.as_deref()));
        // Tags preserved only when varies.
        let has_tags = true;
        assert!(!decode_gate_uses_mixed(
            has_tags,
            gate_up_varies(uniform_mq4.as_deref())
        ));
        assert!(decode_gate_uses_mixed(
            has_tags,
            gate_up_varies(mixed_v1_v2.as_deref())
        ));
        assert!(decode_gate_uses_mixed(
            has_tags,
            gate_up_varies(mixed_mq4_mq6.as_deref())
        ));
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
