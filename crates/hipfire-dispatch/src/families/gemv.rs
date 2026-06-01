// SPDX-License-Identifier: MIT OR Apache-2.0
//! GEMV kernel family: dispatching matrix-vector multiply across quant formats.
//!
//! Supports 4 variants:
//! - **Plain**: y = W·x  (for HFQ / F32 / F16 / Q8_0 quant; MQ-family requires Prerotated)
//! - **Prerotated**: y = W·x_rot (for MQ-family + MFP4 — rotation handled by caller)
//! - **WithResidual**: y += W·x  (HFQ only — MQ-family needs rotation + residual via caller)
//! - **WithSwiGLUResidual**: y += W·silu(gate·up)  (HFQ only)

use rdna_compute::{DType, Gpu, GpuTensor};

use crate::context::DispatchCtx;
use crate::pipeline::{PipelineParams, dispatch_fused};
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

// ── Family ─────────────────────────────────────────────

pub struct GemvFamily {
    registry: KernelRegistry,
}

impl GemvFamily {
    pub fn new() -> Self {
        let registry = KernelRegistry::new();
        gemv_table::populate(&registry);
        Self { registry }
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
    ) -> Result<KernelKey, DispatchError> {
        let key = match variant {
            GemvVariant::Plain => KernelKey::for_gemv(dtype, variant, has_awq)?,
            GemvVariant::Prerotated => KernelKey::for_gemv_prerotated(dtype)?,
            GemvVariant::WithResidual => KernelKey::for_gemv_residual(dtype)?,
            GemvVariant::WithSwiGLUResidual => KernelKey::for_gemv_swiglu_residual(dtype)?,
        };
        self.registry.resolve(key, ctx, None)
    }

    /// Run a GEMV with automatic variant selection.
    ///
    /// Picks `Prerotated` when `dtype_needs_fwht(w.dtype)`, `Plain` otherwise.
    /// Replaces per-model `gemv_prerotated_or_plain` / `dispatch_gemv` helpers.
    pub fn run_auto(
        &self,
        ctx: &DispatchCtx,
        gpu: &mut Gpu,
        w: &WeightRef,
        x: &GpuTensor,
        y: &GpuTensor,
    ) -> Result<(), DispatchError> {
        let variant = if crate::types::dtype_needs_fwht(w.dtype) {
            GemvVariant::Prerotated
        } else {
            GemvVariant::Plain
        };
        self.run(ctx, gpu, &GemvParams {
            w, x, y, variant,
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
        let _resolved = self.resolve(params.w.dtype, params.variant, false, ctx)?;

        match params.variant {
            GemvVariant::Plain => dispatch_plain(gpu, params),
            GemvVariant::Prerotated => {
                let dtype = params.w.dtype;
                if dtype == DType::MFP4G32 {
                    let key = KernelKey::GemvMfp4G32Fused;
                    if self.registry.resolve(key, ctx, None).is_ok() {
                        let pipe_params = PipelineParams {
                            x: params.x, y: params.y, buf: params.w.buf,
                            m: params.w.m, k: params.w.k,
                        };
                        return dispatch_fused(gpu, KernelKey::GemvMfp4G32Fused, &pipe_params);
                    }
                }
                dispatch_prerotated(gpu, params)
            }
            GemvVariant::WithResidual => dispatch_residual(gpu, params),
            GemvVariant::WithSwiGLUResidual => dispatch_swiglu_residual(gpu, params),
        }
    }
}

impl KernelFamily for GemvFamily {
    fn name(&self) -> &'static str {
        "gemv"
    }
}

// ── Plain GEMV dispatch ────────────────────────────────

fn dispatch_plain(gpu: &mut Gpu, params: &GemvParams) -> Result<(), DispatchError> {
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
        F32 => hip!(gpu.gemv_f32(w.buf, x, y)),
        F16 => {
            // F16 GEMV uses batched GEMM with batch=1
            hip!(gpu.gemm_f16_batched_lmhead(w.buf, x, y, m, k, 1))
        }
        Q8_0 => hip!(gpu.gemv_q8_0(w.buf, x, y, m, k)),
        Q4K => hip!(gpu.gemv_q4k(w.buf, x, y, m, k)),
        Q6K => hip!(gpu.gemv_q6k(w.buf, x, y, m, k)),
        HFQ4G256 => hip!(gpu.gemv_hfq4g256(w.buf, x, y, m, k)),
        HFQ4G128 => hip!(gpu.gemv_hfq4g128(w.buf, x, y, m, k)),
        HFQ3G256 => hip!(gpu.gemv_hfq3g256(w.buf, x, y, m, k)),
        HFQ3G128 => hip!(gpu.gemv_hfq3g128(w.buf, x, y, m, k)),
        HFQ2G256 => hip!(gpu.gemv_hfq2g256(w.buf, x, y, m, k)),
        HFQ2G128 => hip!(gpu.gemv_hfq2g128(w.buf, x, y, m, k)),
        HFQ6G256 => hip!(gpu.gemv_hfq6g256(w.buf, x, y, m, k)),
        HFP4G32 => hip!(gpu.gemv_hfp4g32(w.buf, x, y, m, k)),
        Q4F16G64 => hip!(gpu.gemv_q4f16_g64(w.buf, x, y, m, k)),
        Q4F16G32 => hip!(gpu.gemv_q4f16_g32(w.buf, x, y, m, k)),
        Q8HFQ => hip!(gpu.gemv_q8hfq(w.buf, x, y, m, k, w.row_stride)),
        // ParoQ4G128: weights are HFQ4G128 data; caller has already applied Givens
        // rotation to x, so dispatch to the plain HFQ4G128 GEMV kernel.
        ParoQ4G128 => hip!(gpu.gemv_hfq4g128(w.buf, x, y, m, k)),
        // MQ-family Plain requires the caller to use Prerotated variant:
        // rotation + prerotated GEMV is a two-step process managed externally.
        _ => Err(DispatchError::UnsupportedVariant {
            family: "gemv", variant: "plain",
            arch: "", quant: "",
        }),
    }
}

// ── Prerotated GEMV dispatch ───────────────────────────

fn dispatch_prerotated(gpu: &mut Gpu, params: &GemvParams) -> Result<(), DispatchError> {
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
        MQ4G256 => hip!(gpu.gemv_mq4g256_prerotated(w.buf, x, y, m, k)),
        MQ3G256 => hip!(gpu.gemv_mq3g256_prerotated(w.buf, x, y, m, k)),
        MQ2G256 => hip!(gpu.gemv_mq2g256_prerotated(w.buf, x, y, m, k)),
        MQ6G256 => hip!(gpu.gemv_mq6g256_prerotated(w.buf, x, y, m, k)),
        MQ4G128 => hip!(gpu.gemv_mq4g128_prerotated(w.buf, x, y, m, k)),
        // MQ8 prerotated reads x from internal scratch — no x parameter.
        MQ8G256 => hip!(gpu.gemv_mq8g256_prerotated(w.buf, y, m, k)),
        MQ2G256Lloyd => hip!(gpu.gemv_mq2g256_lloyd(w.buf, x, y, m, k)),
        MQ3G256Lloyd => hip!(gpu.gemv_mq3g256_lloyd(w.buf, x, y, m, k)),
        MQ4G256Lloyd => hip!(gpu.gemv_mq4g256_lloyd(w.buf, x, y, m, k)),
        MFP4G32 => hip!(gpu.gemv_mfp4g32_prerotated(w.buf, x, y, m, k)),
        _ => Err(DispatchError::UnsupportedVariant {
            family: "gemv", variant: "prerotated",
            arch: "", quant: "",
        }),
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
