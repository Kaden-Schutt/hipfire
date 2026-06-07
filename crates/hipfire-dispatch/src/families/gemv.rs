// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
//! GEMV kernel family: dispatching matrix-vector multiply across quant formats.
//!
//! Supports 4 variants:
//! - **Plain**: y = W·x  (for HFQ / F32 / F16 / Q8_0 quant; MQ-family requires Prerotated)
//! - **Prerotated**: y = W·x_rot (for MQ-family + MFP4 — rotation handled by caller)
//! - **WithResidual**: y += W·x  (HFQ only — MQ-family needs rotation + residual via caller)
//! - **WithSwiGLUResidual**: y += W·silu(gate·up)  (HFQ only)

use rdna_compute::{DType, Gpu, GpuTensor};

use crate::context::DispatchCtx;
use crate::families::rotation::{RotationFamily, RotationParams};
use crate::pipeline::{LinearParams, PipelineParams, dispatch_fused};
use crate::tables::gemv_table;
use crate::tables::KernelRegistry;
use crate::traits::KernelFamily;
use crate::types::*;

// ── Lightweight weight descriptor ──────────────────────

/// Givens rotation metadata for ParoQuant weights (mirrors ParoRotation
/// fields, which are all rdna_compute::GpuTensor — no circular dep).
pub struct GivensRef<'a> {
    pub pairs: &'a GpuTensor,
    pub theta: &'a GpuTensor,
    pub scales: &'a GpuTensor,
    pub krot: usize,
}

/// Minimal weight reference for dispatch. Carries buffer, dtype, shape,
/// the padded row stride (Q8HFQ), and rotation metadata.
pub struct WeightRef<'a> {
    pub buf: &'a GpuTensor,
    pub dtype: DType,
    pub m: usize,
    pub k: usize,
    pub row_stride: usize,
    pub rotation: Option<GivensRef<'a>>,
    pub awq_scale: Option<&'a GpuTensor>,
}

// ── Dispatch parameters ────────────────────────────────

pub struct GemvParams<'a> {
    pub w: &'a WeightRef<'a>,
    pub x: &'a GpuTensor,
    pub y: &'a GpuTensor,
    pub variant: GemvVariant,
    pub residual: Option<&'a GpuTensor>,
    pub gate: Option<&'a GpuTensor>,
    pub up: Option<&'a GpuTensor>,
}

/// Full rotation signature carried by a RotatedActivation. Records the
/// sign-domain plan AND the awq/batched sub-variant so `run()` can reject
/// a buffer rotated under a different kernel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RotationTag {
    pub plan: crate::types::RotationPlan,
    pub awq: bool,
    pub batched: bool,
}

/// A rotated activation buffer plus the tag of the rotation that produced it.
pub struct RotatedActivation {
    pub(crate) x_rot: GpuTensor,
    pub(crate) tag: RotationTag,
}

impl Clone for RotatedActivation {
    fn clone(&self) -> Self {
        RotatedActivation {
            x_rot: GpuTensor {
                buf: unsafe { self.x_rot.buf.alias() },
                shape: self.x_rot.shape.clone(),
                dtype: self.x_rot.dtype,
            },
            tag: self.tag,
        }
    }
}

impl RotatedActivation {
    /// Fused QKV / gate-up / MoE kernels take a raw &GpuTensor; expose the
    /// rotated buffer for them. The tag still guards the `run()` path.
    pub fn buf(&self) -> &GpuTensor { &self.x_rot }
    pub fn tag(&self) -> RotationTag { self.tag }
    pub fn into_buf(self) -> GpuTensor { self.x_rot }
}

/// GEMV input: raw (family rotates if the plan needs it) or pre-rotated.
pub enum RotInput<'a> {
    Raw(&'a GpuTensor),
    Rotated(RotatedActivation),
}

/// Caller's fusion intent + inputs for `rotate()`. The combination of which
/// fields are Some, plus the weight's plan + awq_scale + batch_size, selects
/// the concrete RotationVariant inside RotationFamily.
pub struct RotateInputs<'a> {
    pub norm_weight: Option<&'a GpuTensor>,
    pub eps: f32,
    pub swiglu_up: Option<&'a GpuTensor>,
    pub batch_size: usize,
}

impl Default for RotateInputs<'_> {
    fn default() -> Self {
        Self { norm_weight: None, eps: 1e-6, swiglu_up: None, batch_size: 1 }
    }
}

// ── Family ─────────────────────────────────────────────

/// Pure selection of the RotationFamily variant from the sign-domain plan
/// and the caller's fusion intent. AWQ/batched are NOT part of this — they
/// are derived inside RotationFamily from awq_scale/batch_size.
pub fn select_rotation_variant(plan: RotationPlan, has_norm: bool, has_swiglu: bool)
    -> RotationVariant
{
    match plan {
        RotationPlan::FwhtG128 => RotationVariant::PlainG128,
        RotationPlan::Givens   => RotationVariant::Givens,
        RotationPlan::Mq8Internal => {
            if has_norm { RotationVariant::WithRmsnorm }
            else { RotationVariant::Plain }
        }
        // FwhtG256 shares the fusion axis.
        _ => {
            if has_swiglu { RotationVariant::WithSwiGLU }
            else if has_norm { RotationVariant::WithRmsnorm }
            else { RotationVariant::Plain }
        }
    }
}

/// Verify a pre-rotated input's tag matches what this weight expects.
pub fn check_rotation_tag(expected: RotationTag, got: RotationTag)
    -> Result<(), DispatchError>
{
    if expected == got { Ok(()) } else {
        Err(DispatchError::UnsupportedVariant {
            family: "gemv", variant: "rotation-tag-mismatch", arch: "", quant: "",
        })
    }
}

pub struct GemvFamily {
    registry: KernelRegistry,
    rotation: RotationFamily,
}

impl GemvFamily {
    pub fn new() -> Self {
        let mut registry = KernelRegistry::new();
        gemv_table::populate(&mut registry);
        registry.validate().expect("gemv kernel table has empty entries");
        let rotation = RotationFamily::new();
        Self { registry, rotation }
    }

    pub fn registry(&self) -> &KernelRegistry {
        &self.registry
    }

    /// Resolve the best kernel key for the given dtype and variant.
    ///
    /// Applies arch gating through `KernelRegistry::resolve`.
    pub fn resolve(
        &self,
        dtype: DType,
        variant: GemvVariant,
        has_awq: bool,
        ctx: &DispatchCtx,
        shape: Option<&ShapeInfo>,
    ) -> Result<&KernelVariant, DispatchError> {
        let key = match variant {
            GemvVariant::Plain => KernelKey::for_gemv(dtype, variant, has_awq)?,
            GemvVariant::Prerotated => KernelKey::for_gemv_prerotated(dtype)?,
            GemvVariant::WithResidual => KernelKey::for_gemv_residual(dtype)?,
            GemvVariant::WithSwiGLUResidual => KernelKey::for_gemv_swiglu_residual(dtype)?,
        };
        self.registry.resolve(key, ctx, shape)
    }

    /// Run a GEMV with automatic variant selection.
    ///
    /// Uses `dtype_post_rotation_variant` so ParoQ4G128 resolves to Plain
    /// (rotate-then-HFQ4G128) and MQ-family to Prerotated.
    pub fn run_auto(
        &self,
        ctx: &DispatchCtx,
        gpu: &mut Gpu,
        w: &WeightRef,
        x: &GpuTensor,
        y: &GpuTensor,
    ) -> Result<(), DispatchError> {
        self.run_input(ctx, gpu, w, RotInput::Raw(x), y)
    }

    /// Run a GEMV with a raw or pre-rotated input.
    ///
    /// `RotInput::Raw(x)` — family auto-rotates if the dtype requires it,
    /// then dispatches the post-rotation variant.
    ///
    /// `RotInput::Rotated(h)` — uses the pre-rotated buffer directly,
    /// validating the rotation tag matches the weight's expected plan.
    pub fn run_input(
        &self,
        ctx: &DispatchCtx,
        gpu: &mut Gpu,
        w: &WeightRef,
        input: RotInput,
        y: &GpuTensor,
    ) -> Result<(), DispatchError> {
        let x_buf = match input {
            RotInput::Raw(x) => {
                let plan = crate::types::dtype_rotation_plan(w.dtype);
                if plan == RotationPlan::None {
                    GpuTensor { buf: unsafe { x.buf.alias() }, shape: x.shape.clone(), dtype: x.dtype }
                } else {
                    let h = self.rotate(ctx, gpu, w, x, &RotateInputs::default())?;
                    h.into_buf()
                }
            }
            RotInput::Rotated(h) => {
                let plan = crate::types::dtype_rotation_plan(w.dtype);
                let expected = RotationTag {
                    plan, awq: w.awq_scale.is_some(), batched: false,
                };
                check_rotation_tag(expected, h.tag())?;
                h.into_buf()
            }
        };
        let variant = crate::types::dtype_post_rotation_variant(w.dtype);
        self.run(ctx, gpu, &GemvParams {
            w, x: &x_buf, y, variant,
            residual: None, gate: None, up: None,
        })
    }

    /// Run a GEMV operation.
    ///
    /// Validates arch compatibility via `resolve()`, then dispatches to the
    /// correct `Gpu` method.
    ///
    /// ## Rotation contract
    ///
    /// - `Plain` → `x` is raw input (no FWHT). The dispatch calls a kernel that
    ///   does NOT apply FWHT rotation. Use this for F32, F16, HFQ, Q8_0, etc.
    /// - `Prerotated` → `x` must be the FWHT-rotated activation. The dispatch
    ///   calls `gemv_*_prerotated()` which expects rotated input. The caller
    ///   is responsible for `ensure_mq_signs()` + `rotate_x_mq()` first.
    /// - `WithResidual` → `x` is raw input; the kernel fuses `y += W·x`.
    /// - `WithSwiGLUResidual` → `gate` and `up` are the SiLU-multiply inputs;
    ///   `residual` receives the accumulated result.
    pub fn run(
        &self,
        ctx: &DispatchCtx,
        gpu: &mut Gpu,
        params: &GemvParams,
    ) -> Result<(), DispatchError> {
        let shape = ShapeInfo { batch_size: 1, head_dim: 0, m: params.w.m, is_tree: false };
        match params.variant {
            GemvVariant::Plain => {
                let key = self.resolve(params.w.dtype, params.variant, false, ctx, Some(&shape))?.key;
                launch(gpu, key, params)
            }
            GemvVariant::Prerotated => {
                if params.w.dtype == DType::MFP4G32
                    && self.registry.resolve(KernelKey::GemvMfp4G32Fused, ctx, None).is_ok()
                {
                    let pipe_params = PipelineParams::Linear(LinearParams {
                        x: params.x, y: params.y, buf: params.w.buf,
                        m: params.w.m, k: params.w.k,
                    });
                    return dispatch_fused(ctx, gpu, KernelKey::GemvMfp4G32Fused, &pipe_params);
                }
                let key = self.resolve(params.w.dtype, params.variant, false, ctx, Some(&shape))?.key;
                launch(gpu, key, params)
            }
            GemvVariant::WithResidual => dispatch_residual(gpu, params),
            GemvVariant::WithSwiGLUResidual => dispatch_swiglu_residual(gpu, params),
        }
    }

    /// Rotate an activation vector through RotationFamily. Returns a typed
    /// RotatedActivation handle whose tag encodes the plan, awq, and batching.
    pub fn rotate(
        &self,
        ctx: &DispatchCtx,
        gpu: &mut Gpu,
        w: &WeightRef,
        x: &GpuTensor,
        inputs: &RotateInputs,
    ) -> Result<RotatedActivation, DispatchError> {
        let plan = crate::types::dtype_rotation_plan(w.dtype);
        if plan == RotationPlan::None {
            return Err(DispatchError::UnsupportedVariant {
                family: "gemv", variant: "rotate-none", arch: "", quant: "",
            });
        }
        let variant = select_rotation_variant(
            plan, inputs.norm_weight.is_some(), inputs.swiglu_up.is_some(),
        );
        let (x_rot, awq, batched) = prepare_rotation_scratch(gpu, w, inputs)?;
        let g = w.rotation.as_ref();
        self.rotation.run(ctx, gpu, RotationParams {
            x,
            x_up: inputs.swiglu_up,
            w_norm: inputs.norm_weight,
            x_plain: &x_rot,
            x_rot: &x_rot,
            awq_scale: w.awq_scale,
            k: w.k,
            eps: inputs.eps,
            batch_size: inputs.batch_size,
            variant,
            givens_pairs: g.map(|g| g.pairs),
            givens_theta: g.map(|g| g.theta),
            givens_scales: g.map(|g| g.scales),
            givens_krot: g.map(|g| g.krot),
        }).map_err(|e| DispatchError::Hip(e.to_string()))?;
        Ok(RotatedActivation {
            x_rot,
            tag: RotationTag { plan, awq, batched },
        })
    }
}

impl KernelFamily for GemvFamily {
    fn name(&self) -> &'static str {
        "gemv"
    }
}

// ── Rotation scratch prep (shared by rotate() + legacy callers) ─

fn prepare_rotation_scratch(
    gpu: &mut Gpu, w: &WeightRef, inputs: &RotateInputs,
) -> Result<(GpuTensor, bool, bool), DispatchError> {
    let plan = crate::types::dtype_rotation_plan(w.dtype);
    let awq = w.awq_scale.is_some();
    let batched = inputs.batch_size > 1;
    let mq = |gpu: &mut Gpu| -> Result<GpuTensor, DispatchError> {
        let buf = unsafe { gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias() };
        let size = gpu.scratch.mq_x_rot.as_ref().unwrap().buf.size();
        Ok(GpuTensor { buf, shape: vec![size / 4], dtype: DType::F32 })
    };
    match plan {
        RotationPlan::FwhtG256 | RotationPlan::Mq8Internal => {
            gpu.ensure_mq_signs().map_err(|e| DispatchError::Hip(e.to_string()))?;
            Ok((mq(gpu)?, awq, batched))
        }
        RotationPlan::FwhtG128 => {
            gpu.ensure_mq_signs_128().map_err(|e| DispatchError::Hip(e.to_string()))?;
            Ok((mq(gpu)?, awq, batched))
        }
        RotationPlan::Givens => {
            gpu.ensure_paro_scratch(w.k).map_err(|e| DispatchError::Hip(e.to_string()))?;
            Ok((GpuTensor {
                buf: unsafe { gpu.scratch.paro_x_scratch.as_ref().unwrap().buf.alias() },
                shape: vec![w.k], dtype: DType::F32,
            }, awq, batched))
        }
        RotationPlan::None => Err(DispatchError::UnsupportedVariant {
            family: "gemv", variant: "rotate-none", arch: "", quant: "",
        }),
    }
}

// ── Central KernelKey-keyed launch ─────────────────────

/// Launch the concrete GEMV kernel for a resolved key. 1:1 with KernelKey.
fn launch(gpu: &mut Gpu, key: KernelKey, p: &GemvParams) -> Result<(), DispatchError> {
    use KernelKey as K;
    let (w, x, y, m, k) = (p.w, p.x, p.y, p.w.m, p.w.k);
    macro_rules! hip { ($e:expr) => { $e.map_err(|e| DispatchError::Hip(e.to_string())) }; }
    match key {
        K::GemvF32 => hip!(gpu.gemv_f32(w.buf, x, y)),
        K::GemvF16 => hip!(gpu.gemm_f16_batched_lmhead(w.buf, x, y, m, k, 1)),
        K::GemvQ8_0 => hip!(gpu.gemv_q8_0(w.buf, x, y, m, k)),
        K::GemvQ4K => hip!(gpu.gemv_q4k(w.buf, x, y, m, k)),
        K::GemvQ6K => hip!(gpu.gemv_q6k(w.buf, x, y, m, k)),
        K::GemvHfq4G256 => hip!(gpu.gemv_hfq4g256(w.buf, x, y, m, k)),
        K::GemvHfq4G128 | K::GemvParoQ4G128 => hip!(gpu.gemv_hfq4g128(w.buf, x, y, m, k)),
        K::GemvHfq3G256 => hip!(gpu.gemv_hfq3g256(w.buf, x, y, m, k)),
        K::GemvHfq3G128 => hip!(gpu.gemv_hfq3g128(w.buf, x, y, m, k)),
        K::GemvHfq2G256 => hip!(gpu.gemv_hfq2g256(w.buf, x, y, m, k)),
        K::GemvHfq2G128 => hip!(gpu.gemv_hfq2g128(w.buf, x, y, m, k)),
        K::GemvHfq6G256 => hip!(gpu.gemv_hfq6g256(w.buf, x, y, m, k)),
        K::GemvHfp4G32 => hip!(gpu.gemv_hfp4g32(w.buf, x, y, m, k)),
        K::GemvQ4F16G64 => hip!(gpu.gemv_q4f16_g64(w.buf, x, y, m, k)),
        K::GemvQ4F16G32 => hip!(gpu.gemv_q4f16_g32(w.buf, x, y, m, k)),
        K::GemvQ8HFQ => hip!(gpu.gemv_q8hfq(w.buf, x, y, m, k, w.row_stride)),
        // prerotated
        K::GemvMq4G256Prerotated => hip!(gpu.gemv_mq4g256_prerotated(w.buf, x, y, m, k)),
        K::GemvMq3G256Prerotated => hip!(gpu.gemv_mq3g256_prerotated(w.buf, x, y, m, k)),
        K::GemvMq2G256Prerotated => hip!(gpu.gemv_mq2g256_prerotated(w.buf, x, y, m, k)),
        K::GemvMq6G256Prerotated => hip!(gpu.gemv_mq6g256_prerotated(w.buf, x, y, m, k)),
        K::GemvMq4G128 => hip!(gpu.gemv_mq4g128_prerotated(w.buf, x, y, m, k)),
        K::GemvMq8G256Prerotated => hip!(gpu.gemv_mq8g256_prerotated(w.buf, y, m, k)),
        K::GemvMq2G256Lloyd | K::GemvMq2G256LloydPrerotated => hip!(gpu.gemv_mq2g256_lloyd(w.buf, x, y, m, k)),
        K::GemvMq3G256Lloyd | K::GemvMq3G256LloydPrerotated => hip!(gpu.gemv_mq3g256_lloyd(w.buf, x, y, m, k)),
        K::GemvMq4G256Lloyd | K::GemvMq4G256LloydPrerotated => hip!(gpu.gemv_mq4g256_lloyd(w.buf, x, y, m, k)),
        K::GemvMfp4G32Prerotated => hip!(gpu.gemv_mfp4g32_prerotated(w.buf, x, y, m, k)),
        other => return Err(DispatchError::MissingImpl { key: other }),
    }
}

// ── Residual GEMV dispatch ─────────────────────────────

fn dispatch_residual(gpu: &mut Gpu, params: &GemvParams) -> Result<(), DispatchError> {
    let w = params.w;
    let x = params.x;
    let y = params.y;
    let m = w.m;
    let k = w.k;
    use DType::*;

    macro_rules! hip {
        ($e:expr) => {
            $e.map_err(|e| DispatchError::Hip(e.to_string()))
        };
    }

    match w.dtype {
        HFQ4G256 => hip!(gpu.gemv_hfq4g256_residual(w.buf, x, y, m, k)),
        HFQ3G256 => hip!(gpu.gemv_hfq3g256_residual(w.buf, x, y, m, k)),
        HFQ6G256 => hip!(gpu.gemv_hfq6g256_residual(w.buf, x, y, m, k)),
        // MQ-family WithResidual requires caller-supplied pre-rotated x
        // (same contract as Prerotated) — dispatch through HFQ residual kernel.
        MQ4G256 => hip!(gpu.gemv_hfq4g256_residual(w.buf, x, y, m, k)),
        MQ3G256 => hip!(gpu.gemv_hfq3g256_residual(w.buf, x, y, m, k)),
        MQ6G256 => hip!(gpu.gemv_hfq6g256_residual(w.buf, x, y, m, k)),
        MQ3G256Lloyd => hip!(gpu.gemv_mq3g256_lloyd_residual(w.buf, x, y, m, k)),
        MQ4G256Lloyd => hip!(gpu.gemv_mq4g256_lloyd_residual(w.buf, x, y, m, k)),
        _ => Err(DispatchError::UnsupportedVariant {
            family: "gemv", variant: "residual",
            arch: "", quant: "",
        }),
    }
}

// ── SwiGLU + Residual GEMV dispatch ────────────────────

fn dispatch_swiglu_residual(gpu: &mut Gpu, params: &GemvParams) -> Result<(), DispatchError> {
    let w = params.w;
    let x_in = params.gate.ok_or_else(|| DispatchError::MissingImpl {
        key: KernelKey::GemvF32,
    })?;
    let residual = params.residual.unwrap_or(params.y);
    let m = w.m;
    let k = w.k;
    use DType::*;

    macro_rules! hip {
        ($e:expr) => {
            $e.map_err(|e| DispatchError::Hip(e.to_string()))
        };
    }

    // SwiGLU+Residual dispatch.
    //
    // HFQ dtypes: caller must pre-compute silu(gate)*up and pass as `gate`
    // (the GEMV then does y += W · silu(gate*up) with the pre-fused input).
    // MQ-family: caller must also pre-rotate via fused_silu_mul_rotate_mq.
    // True fused SwiGLU+Residual kernels (PARO) are not yet wired here.
    match w.dtype {
        HFQ4G256 => hip!(gpu.gemv_hfq4g256_residual(w.buf, x_in, residual, m, k)),
        HFQ3G256 => hip!(gpu.gemv_hfq3g256_residual(w.buf, x_in, residual, m, k)),
        HFQ6G256 => hip!(gpu.gemv_hfq6g256_residual(w.buf, x_in, residual, m, k)),
        MQ4G256 => hip!(gpu.gemv_hfq4g256_residual(w.buf, x_in, residual, m, k)),
        MQ3G256 => hip!(gpu.gemv_hfq3g256_residual(w.buf, x_in, residual, m, k)),
        MQ6G256 => hip!(gpu.gemv_hfq6g256_residual(w.buf, x_in, residual, m, k)),
        MQ3G256Lloyd => hip!(gpu.gemv_mq3g256_lloyd_residual(w.buf, x_in, residual, m, k)),
        MQ4G256Lloyd => hip!(gpu.gemv_mq4g256_lloyd_residual(w.buf, x_in, residual, m, k)),
        _ => Err(DispatchError::UnsupportedVariant {
            family: "gemv", variant: "swiglu_residual",
            arch: "", quant: "",
        }),
    }
}
