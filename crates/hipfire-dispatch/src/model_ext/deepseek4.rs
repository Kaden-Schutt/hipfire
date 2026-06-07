// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
// DeepSeek V4 model-specific kernel extensions.
//
// These operations are unique to DeepSeek V4 — the compressor, joint K=V,
// and Q-LoRA paths. Each calls through standard dispatch families (GemvFamily)
// for the GEMM/GEMV portions; the custom kernels (compressor pool, ring write,
// overlap concat, APE add) are called directly on Gpu.

#![cfg(any())]

use rdna_compute::{DType, Gpu, GpuTensor};
use crate::context::DispatchCtx;
use crate::families::gemv::{GemvFamily, GemvParams, GemvVariant, WeightRef};
use crate::types::DispatchError;

// ── Weight dispatch helpers ─────────────────────────────

/// True if the weight's dtype expects FWHT-rotated input.
/// Mirrors `weight_needs_fwht` in the model code.
#[inline]
pub fn weight_needs_fwht(weight: &GpuTensor) -> bool {
    !matches!(weight.dtype, DType::F32 | DType::F16 | DType::Q8_0)
}

/// Select the GEMV variant based on weight dtype:
///   - F32 / F16 / Q8_0 → Plain (reads raw RMSNorm output)
///   - Quantized (MQ4G256, MQ3G256, etc.) → Prerotated (reads FWHT-rotated input)
fn gemv_variant_for_weight(weight: &GpuTensor) -> GemvVariant {
    if weight_needs_fwht(weight) {
        GemvVariant::Prerotated
    } else {
        GemvVariant::Plain
    }
}

/// Build a `WeightRef` from a GpuTensor whose M/K are the output/input dims.
fn weight_ref<'a>(w: &'a GpuTensor, m: usize, k: usize) -> WeightRef<'a> {
    WeightRef {
        buf: w,
        dtype: w.dtype,
        m,
        k,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    }
}

/// Run a single GEMV through the dispatch layer, auto-selecting Plain vs
/// Prerotated. Consumes x_rotated when the weight requires FWHT input,
/// x_plain otherwise — the caller must provide both (the model code's
/// standard pattern). Pre-computed rotation is expected when needed; this
/// helper does NOT rotate.
fn gemv_auto_dispatch(
    gemv: &GemvFamily,
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x_rotated: &GpuTensor,
    x_plain: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
) -> Result<(), DispatchError> {
    let variant = gemv_variant_for_weight(weight);
    let x = match variant {
        GemvVariant::Prerotated => x_rotated,
        _ => x_plain,
    };
    let w = weight_ref(weight, m, k);
    let params = GemvParams {
        w: &w,
        x,
        y,
        variant,
        residual: None,
        gate: None,
        up: None,
    };
    gemv.run(ctx, gpu, &params)
}

// ── Parameter structs ──────────────────────────────────

/// Parameters for the DeepSeek V4 compressor forward step.
///
/// The compressor runs two GEMV projections (wkv @ x → kv, wgate @ x → score),
/// then applies APE, ring-buffer write, and a softmax-weighted pool to produce
/// a single compressed KV entry.
pub struct CompressorParams<'a> {
    pub wkv: &'a GpuTensor,
    pub wgate: &'a GpuTensor,
    pub ape: &'a GpuTensor,
    pub x_rotated: &'a GpuTensor,
    pub x_plain: &'a GpuTensor,
    pub kv_buf: &'a GpuTensor,
    pub score_buf: &'a GpuTensor,
    pub proj_dim: usize,
    pub hidden_size: usize,
    pub position: u32,
    pub ratio: usize,
}

/// Parameters for the joint K=V projection.
///
/// A single weight (wkv) projects the attention input into the shared K=V space,
/// followed by RMSNorm.
pub struct JointKvParams<'a> {
    pub wkv: &'a GpuTensor,
    pub kv_norm: &'a GpuTensor,
    pub tmp: &'a GpuTensor,
    pub tmp_plain: &'a GpuTensor,
    pub kv: &'a GpuTensor,
    pub kv_dim: usize,
    pub hidden_size: usize,
    pub rms_norm_eps: f32,
}

/// Parameters for the Q-LoRA path.
///
/// Two-stage low-rank projection (wq_a → bottleneck, wq_b → full Q head space)
/// with dual-domain RMSNorm and an intermediate FWHT rotation step.
pub struct QLoraParams<'a> {
    pub wq_a: &'a GpuTensor,
    pub wq_b: &'a GpuTensor,
    pub attn_norm: &'a GpuTensor,
    pub q_norm: &'a GpuTensor,
    pub hc_x_in: &'a GpuTensor,
    pub tmp: &'a GpuTensor,
    pub tmp_plain: &'a GpuTensor,
    pub q_lat: &'a GpuTensor,
    pub q_lat_rot: &'a GpuTensor,
    pub q: &'a GpuTensor,
    pub q_head_ones: &'a GpuTensor,
    pub q_lora_rank: usize,
    pub hidden_size: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
}

// ── Trait ──────────────────────────────────────────────

pub trait Deepseek4ModelExt {
    /// Run the compressor forward: wkv/wgate GEMVs then APE add.
    ///
    /// The caller is responsible for ring-buffer write, overlap concat,
    /// and softmax-pool after this returns — those are model-owned custom
    /// kernels called directly on `gpu`.
    fn run_compressor(
        &self,
        gemv: &GemvFamily,
        ctx: &DispatchCtx,
        gpu: &mut Gpu,
        params: &CompressorParams,
    ) -> Result<(), DispatchError>;

    /// Run joint K=V: wkv GEMV followed by RMSNorm.
    fn run_joint_kv(
        &self,
        gemv: &GemvFamily,
        ctx: &DispatchCtx,
        gpu: &mut Gpu,
        params: &JointKvParams,
    ) -> Result<(), DispatchError>;

    /// Run Q-LoRA: dual GEMV (wq_a + wq_b) with RMSNorm + FWHT rotation.
    ///
    /// Calls the fused RMSNorm+rotate kernel when wq_a needs FWHT,
    /// then dispatches wq_a and wq_b GEMVs through the GemvFamily.
    fn run_q_lora(
        &self,
        gemv: &GemvFamily,
        ctx: &DispatchCtx,
        gpu: &mut Gpu,
        params: &QLoraParams,
    ) -> Result<(), DispatchError>;
}

// ── Default implementations ────────────────────────────

impl Deepseek4ModelExt for () {
    fn run_compressor(
        &self,
        gemv: &GemvFamily,
        ctx: &DispatchCtx,
        gpu: &mut Gpu,
        params: &CompressorParams,
    ) -> Result<(), DispatchError> {
        gemv_auto_dispatch(
            gemv, ctx, gpu,
            params.wkv, params.x_rotated, params.x_plain,
            params.kv_buf, params.proj_dim, params.hidden_size,
        )?;
        gemv_auto_dispatch(
            gemv, ctx, gpu,
            params.wgate, params.x_rotated, params.x_plain,
            params.score_buf, params.proj_dim, params.hidden_size,
        )?;
        Ok(())
    }

    fn run_joint_kv(
        &self,
        gemv: &GemvFamily,
        ctx: &DispatchCtx,
        gpu: &mut Gpu,
        params: &JointKvParams,
    ) -> Result<(), DispatchError> {
        gemv_auto_dispatch(
            gemv, ctx, gpu,
            params.wkv, params.tmp, params.tmp_plain,
            params.kv, params.kv_dim, params.hidden_size,
        )?;
        gpu.rmsnorm_f32(params.kv, params.kv_norm, params.kv, params.rms_norm_eps)
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
        Ok(())
    }

    fn run_q_lora(
        &self,
        gemv: &GemvFamily,
        ctx: &DispatchCtx,
        gpu: &mut Gpu,
        params: &QLoraParams,
    ) -> Result<(), DispatchError> {
        let wq_a_needs_fwht = weight_needs_fwht(params.wq_a);
        let wq_b_needs_fwht = weight_needs_fwht(params.wq_b);

        // Step 1: RMSNorm (+ optional FWHT rotation).
        if wq_a_needs_fwht {
            gpu.fused_rmsnorm_rotate_mq_plain(
                params.hc_x_in, params.attn_norm,
                params.tmp, params.tmp_plain,
                params.hidden_size, params.rms_norm_eps,
            )
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
        } else {
            gpu.rmsnorm_f32(params.hc_x_in, params.attn_norm, params.tmp_plain, params.rms_norm_eps)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
        }

        // Step 2: wq_a GEMV → q_lat bottleneck.
        gemv_auto_dispatch(
            gemv, ctx, gpu,
            params.wq_a, params.tmp, params.tmp_plain,
            params.q_lat, params.q_lora_rank, params.hidden_size,
        )?;

        // Step 3: q_norm on the bottleneck.
        gpu.rmsnorm_f32(params.q_lat, params.q_norm, params.q_lat, params.rms_norm_eps)
            .map_err(|e| DispatchError::Hip(e.to_string()))?;

        // Step 4: FWHT rotate q_lat for wq_b (if needed).
        if wq_b_needs_fwht {
            gpu.rotate_x_mq(params.q_lat, params.q_lat_rot, params.q_lora_rank)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
        }

        // Step 5: wq_b GEMV → full Q head space.
        let q_total = params.n_heads * params.head_dim;
        gemv_auto_dispatch(
            gemv, ctx, gpu,
            params.wq_b, params.q_lat_rot, params.q_lat,
            params.q, q_total, params.q_lora_rank,
        )?;

        // Step 6: per-head RMSNorm.
        gpu.rmsnorm_f32(params.q, params.q_head_ones, params.q, params.rms_norm_eps)
            .map_err(|e| DispatchError::Hip(e.to_string()))?;

        Ok(())
    }
}
