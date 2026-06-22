// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Generic GPU weight/embedding types and the weight-GEMV / rotation /
//! rmsnorm-fusion operations on them.
//!
//! `WeightTensor` (a GPU-resident quantized/F32 weight + its metadata),
//! `EmbeddingFormat`, `LayerWeights`, `ParoRotation`, and the dense
//! `weight_gemv*` / `rotate_x_*` / `fused_*` kernels are arch-agnostic — every
//! dense/MoE arch builds its layers from these. They historically lived in
//! `llama.rs`; relocated here as part of the de-llama-ify cleanup. The
//! llama-specific `LlamaWeights` aggregate stays in `llama.rs`.

use crate::dispatch::gemv_family;
use hip_bridge::HipResult;
use rdna_compute::generic_warn::{warn_generic_once, KernelMode, Quality};
use rdna_compute::{DType, Gpu, GpuTensor};

pub struct ParoRotation {
    pub pairs: GpuTensor, // I16 [krot, in_dim] — pair indices per rotation layer
    pub theta: GpuTensor, // F16 [krot, in_dim/2] — learned angles
    pub channel_scales: GpuTensor, // F16 [in_dim] — per-channel scaling factor alpha
    pub krot: u32,        // number of rotation layers (typically 8)
    pub group_size: u32,  // quantization group size (typically 128)
    /// True if `pairs`/`theta`/`channel_scales` are non-owning aliases into
    /// shared per-layer sidecars (e.g. MoE routed experts that share one
    /// rotation tuple across all experts in a layer). Owning ParoRotations
    /// set this to false; `WeightTensor::free_all` skips tensor frees when
    /// is_alias is true so the shared sidecars aren't double-freed.
    pub is_alias: bool,
}

pub struct WeightTensor {
    pub buf: GpuTensor,
    pub gpu_dtype: DType,  // dispatch type for kernel selection
    pub m: usize,          // output dim (rows)
    pub k: usize,          // input dim (cols)
    pub row_stride: usize, // padded row bytes (Q8HFQ only, 0 for others)
    /// ParoQuant Givens rotation metadata. None for all non-ParoQuant formats.
    pub paro: Option<ParoRotation>,
    /// Phase A Stage A — AWQ per-channel scale vector, length K, dtype F16.
    ///
    /// Populated by the loader when the .hfq carries a sibling sidecar tensor
    /// named `<weight>.awq_scale.weight`. The forward path (specifically the
    /// fused-rmsnorm-rotate call upstream of this linear) must apply
    /// `x[i] /= awq_scale[i]` before the rotation, completing the AWQ
    /// math `(W·s) · (x/s) = W·x`. The pre-scaling of W·s was done at
    /// quantize time (see `compute_awq_scales` in hipfire-quantize/main.rs).
    ///
    /// `None` for tensors that weren't AWQ-pre-scaled — backward-compatible
    /// with all existing .hfq files.
    pub awq_scale: Option<GpuTensor>,
}

impl WeightTensor {
    /// Free the weight buffer and any associated metadata (ParoQuant rotation,
    /// AWQ sidecar) from GPU.
    pub fn free_all(self, gpu: &mut Gpu) {
        if let Some(paro) = self.paro {
            // Aliased rotations point into shared per-layer sidecars; the owner
            // (e.g. MoeFfnWeights.paro_shared) frees them. Skip here.
            if !paro.is_alias {
                let _ = gpu.free_tensor(paro.pairs);
                let _ = gpu.free_tensor(paro.theta);
                let _ = gpu.free_tensor(paro.channel_scales);
            }
        }
        if let Some(awq) = self.awq_scale {
            let _ = gpu.free_tensor(awq);
        }
        let _ = gpu.free_tensor(self.buf);
    }
}

impl WeightTensor {
    /// Logic-free adapter to the dispatch-layer WeightRef. Wires Givens +
    /// AWQ + row_stride so GemvFamily sees everything a weight needs.
    pub fn dispatch_ref(&self) -> hipfire_dispatch::families::gemv::WeightRef<'_> {
        use hipfire_dispatch::families::gemv::{GivensRef, WeightRef};
        WeightRef {
            buf: &self.buf,
            dtype: self.gpu_dtype,
            m: self.m,
            k: self.k,
            row_stride: self.row_stride,
            rotation: self.paro.as_ref().map(|p| GivensRef {
                pairs: &p.pairs,
                theta: &p.theta,
                scales: &p.channel_scales,
                krot: p.krot as usize,
            }),
            awq_scale: self.awq_scale.as_ref(),
        }
    }
}

/// How the embedding table is stored on GPU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmbeddingFormat {
    F32,      // dequantized to F32, use D2D copy
    Q4K,      // raw Q4K blocks, use GPU dequant kernel
    HFQ4G256, // raw HFQ4-G256 blocks, use GPU dequant kernel
    HFQ4G128, // raw HFQ4-G128 blocks, use GPU dequant kernel
    Q8_0,     // raw Q8_0 blocks, use GPU dequant kernel
}

pub struct LayerWeights {
    pub attn_norm: GpuTensor,
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,
    pub wo: WeightTensor,
    pub q_norm: Option<GpuTensor>, // Qwen3: per-head Q normalization
    pub k_norm: Option<GpuTensor>, // Qwen3: per-head K normalization
    pub ffn_norm: GpuTensor,
    pub w_gate: WeightTensor,
    pub w_up: WeightTensor,
    pub w_down: WeightTensor,
}

/// Dispatch GEMV for a weight tensor (quantized or F32).
/// y = W * x where W is the weight tensor, x is F32 input, y is F32 output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
enum DenseGemvRoute {
    Mq6RotateThenMq6Prerotated,
    Hfq6Direct,
    Unclassified,
}

#[cfg(test)]
fn dense_gemv_route(dtype: DType) -> DenseGemvRoute {
    match dtype {
        DType::MQ6G256 => DenseGemvRoute::Mq6RotateThenMq6Prerotated,
        DType::HFQ6G256 => DenseGemvRoute::Hfq6Direct,
        _ => DenseGemvRoute::Unclassified,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DensePrerotatedGemvRoute {
    Mq6Prerotated,
    Unclassified,
}

fn dense_prerotated_gemv_route(dtype: DType) -> DensePrerotatedGemvRoute {
    match dtype {
        DType::MQ6G256 => DensePrerotatedGemvRoute::Mq6Prerotated,
        _ => DensePrerotatedGemvRoute::Unclassified,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DenseResidualRoute {
    Mq6RotateThenHfq6Residual,
    Hfq6ResidualDirect,
    Unclassified,
}

fn dense_residual_route(dtype: DType) -> DenseResidualRoute {
    match dtype {
        DType::MQ6G256 => DenseResidualRoute::Mq6RotateThenHfq6Residual,
        DType::HFQ6G256 => DenseResidualRoute::Hfq6ResidualDirect,
        _ => DenseResidualRoute::Unclassified,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DenseSwigluResidualRoute {
    Mq6RotateThenHfq6Residual,
    Unclassified,
}

fn dense_swiglu_residual_route(dtype: DType) -> DenseSwigluResidualRoute {
    match dtype {
        DType::MQ6G256 => DenseSwigluResidualRoute::Mq6RotateThenHfq6Residual,
        _ => DenseSwigluResidualRoute::Unclassified,
    }
}

pub fn weight_gemv(gpu: &mut Gpu, w: &WeightTensor, x: &GpuTensor, y: &GpuTensor) -> HipResult<()> {
    use hipfire_dispatch::context::DispatchCtx;
    use hipfire_dispatch::families::gemv::{GemvParams, WeightRef};
    use hipfire_dispatch::types::{dtype_needs_rotation, GemvVariant};

    let gemv = gemv_family();
    let ctx = DispatchCtx::new(gpu);
    let wr = WeightRef {
        buf: &w.buf,
        dtype: w.gpu_dtype,
        m: w.m,
        k: w.k,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    };

    if !dtype_needs_rotation(w.gpu_dtype) {
        // BF16 weights use WMMA GEMM directly (dispatch family has no BF16 GEMV entry).
        if w.gpu_dtype == DType::BF16 {
            return gpu.gemm_bf16_x_bf16_wmma(&w.buf, x, y, w.m, w.k, 1);
        }
        return gemv
            .run_auto(&ctx, gpu, &wr, x, y)
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()));
    }

    macro_rules! xr {
        () => {{
            GpuTensor {
                buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            }
        }};
    }

    match w.gpu_dtype {
        // MQ8 reads from internal scratch — no rotation or x alias needed
        DType::MQ8G256 => {
            gpu.ensure_mq_signs()?;
            gemv.run_auto(&ctx, gpu, &wr, x, y)
                .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))
        }
        DType::HFQ3G256 => gpu.gemv_hfq3g256(&w.buf, x, y, w.m, w.k),
        DType::HFQ3G128 => gpu.gemv_hfq3g128(&w.buf, x, y, w.m, w.k),
        DType::HFQ2G256 => gpu.gemv_hfq2g256(&w.buf, x, y, w.m, w.k),
        DType::HFQ2G128 => gpu.gemv_hfq2g128(&w.buf, x, y, w.m, w.k),
        DType::Q4F16G64 => gpu.gemv_q4f16_g64(&w.buf, x, y, w.m, w.k),
        DType::Q4F16G32 => gpu.gemv_q4f16_g32(&w.buf, x, y, w.m, w.k),
        // MQ4G128 uses G128 rotation (rotate_x_mq_128, sign seeds 43/1043)
        DType::MQ4G128 => {
            use hipfire_dispatch::families::rotation::{RotationFamily, RotationParams};
            use std::sync::OnceLock;
            static ROTATION: OnceLock<RotationFamily> = OnceLock::new();
            let rotation = ROTATION.get_or_init(|| RotationFamily::new());
            let xr = xr!();
            rotation
                .run(
                    &ctx,
                    gpu,
                    RotationParams {
                        x,
                        x_up: None,
                        w_norm: None,
                        x_plain: &xr,
                        x_rot: &xr,
                        awq_scale: None,
                        k: w.k,
                        eps: 1e-6,
                        batch_size: 1,
                        variant: hipfire_dispatch::types::RotationVariant::PlainG128,
                        givens_pairs: None,
                        givens_theta: None,
                        givens_scales: None,
                        givens_krot: None,
                    },
                )
                .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
            // xr is ALREADY FWHT-rotated by rotate_x_mq_for above. Use the
            // Prerotated GEMV directly — calling run_auto here would re-rotate
            // (dtype_rotation_plan(MQ*) != None), double-applying the involutory
            // FWHT and feeding effectively-unrotated activations to the
            // prerotated kernel (garbage logits). Mirrors master's
            // rotate_x_mq_for + gemv_*_prerotated.
            gemv.run(
                &ctx,
                gpu,
                &GemvParams {
                    w: &wr,
                    x: &xr,
                    y,
                    variant: GemvVariant::Prerotated,
                    residual: None,
                    gate: None,
                    up: None,
                },
            )
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))
        }
        // ParoQ4G128: Givens rotation (model-layer ParoRotation metadata) +
        // HFQ4-G128 GEMV. Uses RotationFamily::run(Givens) which calls
        // givens_rotate_to (copy_d2d + rotate in one kernel).
        DType::ParoQ4G128 => {
            use hipfire_dispatch::families::rotation::{RotationFamily, RotationParams};
            use std::sync::OnceLock;
            static ROTATION: OnceLock<RotationFamily> = OnceLock::new();
            let rotation = ROTATION.get_or_init(|| RotationFamily::new());
            let paro = w
                .paro
                .as_ref()
                .expect("ParoQ4G128 weight missing ParoRotation metadata");
            gpu.ensure_paro_scratch(w.k)?;
            let xr = GpuTensor {
                buf: unsafe { gpu.paro_x_scratch.as_ref().unwrap().buf.alias() },
                shape: vec![w.k],
                dtype: DType::F32,
            };
            rotation
                .run(
                    &ctx,
                    gpu,
                    RotationParams {
                        x,
                        x_up: None,
                        w_norm: None,
                        x_plain: &xr,
                        x_rot: &xr,
                        awq_scale: None,
                        k: w.k,
                        eps: 1e-6,
                        batch_size: 1,
                        variant: hipfire_dispatch::types::RotationVariant::Givens,
                        givens_pairs: Some(&paro.pairs),
                        givens_theta: Some(&paro.theta),
                        givens_scales: Some(&paro.channel_scales),
                        givens_krot: Some(paro.krot as usize),
                    },
                )
                .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
            // After Givens rotation xr is ready; use Plain (HFQ4G128 kernel), not Prerotated.
            gemv.run(
                &ctx,
                gpu,
                &GemvParams {
                    w: &wr,
                    x: &xr,
                    y,
                    variant: GemvVariant::Plain,
                    residual: None,
                    gate: None,
                    up: None,
                },
            )
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))
        }
        // Opus Quant W4A4 (Oq4G256) — int4 activations + int4 weights. Distinct
        // from every other dtype here: those keep f16 activations and dequant the
        // weight in-kernel; Opus quantizes the (AWQ-divided, FWHT-rotated) x to
        // int4 at runtime then runs an integer iu4·iu4 GEMM. Decode is B=1.
        //   x → rotate_x_mq_for (x/=awq_scale if sidecar + FWHT-256) → x_rot (f32)
        //     → quantize_act_oq4 (int4 + per-group f32 scale) → gemm_oq4_grouped_wmma
        // Weight buffer holds [packed nibbles | per-group f32 scales]; the scale
        // pointer is a sub_offset view (see the qt=32 loader arm).
        DType::Oq4G256 => {
            const GROUP: usize = 256;
            assert_eq!(w.k % GROUP, 0, "Oq4G256 weight_gemv: K must be % 256");
            let ng = w.k / GROUP;
            gpu.ensure_mq_signs()?;
            gpu.ensure_oq4_scratch()?;
            let xr = xr!();
            rotate_x_mq_for(gpu, w, x, &xr, w.k)?;
            // Persistent int4 activation scratch (B=1) — aliased, NOT per-call
            // alloc, so the forward stays hipGraph-capture-clean (no hipMalloc/Free
            // inside the captured region). quantize_act_oq4 fully overwrites xq/xs;
            // stream-ordered reuse across sequential projections is safe.
            let xq = GpuTensor {
                buf: unsafe { gpu.oq4_xq.as_ref().unwrap().buf.alias() },
                shape: vec![w.k / 2],
                dtype: DType::Raw,
            };
            let xs = GpuTensor {
                buf: unsafe { gpu.oq4_xs.as_ref().unwrap().buf.alias() },
                shape: vec![ng],
                dtype: DType::F32,
            };
            gpu.quantize_act_oq4(&xr, &xq, &xs, 1, w.k, GROUP)?;
            // Weight scales view: byte offset M*(K/2) into the combined buffer.
            let ws = w.buf.sub_offset(w.m * (w.k / 2), w.m * ng * 4);
            gpu.gemm_oq4_grouped_wmma(&w.buf, &ws, &xq, &xs, y, w.m, w.k, 1, GROUP)
        }
        // W8A8 reference (decode = B=1): reuse the weight_gemm path. buf =
        // [M*K int8 | M f32]. Per-vector int8 act-quant, iu8 WMMA (B=1), rowcol dequant.
        DType::W8A8Ref => {
            warn_generic_once("weight_gemv", "W8A8", KernelMode::Decode, &gpu.arch, Quality::Reference);
            let xq = gpu.alloc_tensor(&[w.k], DType::Raw)?; // int8
            let xs = gpu.alloc_tensor(&[1], DType::F32)?;
            let yi = gpu.alloc_tensor(&[w.m * 4], DType::Raw)?; // int32
            let w_scale = w.buf.sub_offset(w.m * w.k, w.m * 4);
            gpu.quantize_act_int8_per_token(x, &xq, &xs, 1, w.k)?;
            gpu.gemm_iu8_i32_wmma(&w.buf, &xq, &yi, w.m, w.k, 1)?;
            gpu.dequant_i32_rowcol(&yi, &xs, &w_scale, y, 1, w.m)?;
            gpu.free_tensor(xq)?;
            gpu.free_tensor(xs)?;
            gpu.free_tensor(yi)?;
            Ok(())
        }
        // Opus Quant W8A8 (Oq8G256) — int8 activations + int8 weights, near-lossless.
        // The int8 generalization of the Oq4 arm (no nibble packing; iu8 WMMA).
        // x → rotate_x_mq_for (FWHT-256) → quantize_act_oq8 (int8 + per-group scale)
        // → gemm_oq8_grouped_wmma. Weight buffer = [int8 M*K | f32 scales M*ng].
        // The decode path (attn-output / lm_head) reaches Oq8 through
        // dispatch_ref → run_auto → GemvOq8G256Prerotated in the gemv registry;
        // generate and other direct weight_gemv callers hit this arm.
        DType::Oq8G256 => {
            const GROUP: usize = 256;
            assert_eq!(w.k % GROUP, 0, "Oq8G256 weight_gemv: K must be % 256");
            let ng = w.k / GROUP;
            gpu.ensure_mq_signs()?;
            gpu.ensure_oq4_scratch()?;
            let xr = xr!();
            rotate_x_mq_for(gpu, w, x, &xr, w.k)?;
            let xq = gpu.alloc_tensor(&[w.k], DType::Raw)?;
            let xs = gpu.alloc_tensor(&[ng], DType::F32)?;
            gpu.quantize_act_oq8(&xr, &xq, &xs, 1, w.k, GROUP)?;
            let ws = w.buf.sub_offset(w.m * w.k, w.m * ng * 4);
            gpu.gemm_oq8_grouped_wmma(&w.buf, &ws, &xq, &xs, y, w.m, w.k, 1, GROUP)
        }
        // All other FWHT-requiring dtypes (MQ4G256, MQ6G256, MQ3G256, MQ2G256,
        // MQ2G256Lloyd, MQ3G256Lloyd, MQ4G256Lloyd, MFP4G32):
        // ensure_mq_signs + rotate_x_mq_for + run_auto
        // The fused MFP4G32 optimization is handled inside
        // GemvFamily::run() -> Prerotated arm.
        _ => {
            debug_assert!(
                w.gpu_dtype != DType::Oq4G256,
                "Oq4G256 reached weight_gemv generic arm — should hit the dedicated arm"
            );
            if std::env::var_os("HIPFIRE_OQ4_TRACE").is_some() {
                eprintln!(
                    "[oq4-trace] weight_gemv generic _ => arm dtype={:?}",
                    w.gpu_dtype
                );
            }
            gpu.ensure_mq_signs()?;
            let xr = xr!();
            rotate_x_mq_for(gpu, w, x, &xr, w.k)?;
            // xr is ALREADY FWHT-rotated by rotate_x_mq_for above. Use the
            // Prerotated GEMV directly — calling run_auto here would re-rotate
            // (dtype_rotation_plan(MQ*) != None), double-applying the involutory
            // FWHT and feeding effectively-unrotated activations to the
            // prerotated kernel (garbage logits). Mirrors master's
            // rotate_x_mq_for + gemv_*_prerotated.
            gemv.run(
                &ctx,
                gpu,
                &GemvParams {
                    w: &wr,
                    x: &xr,
                    y,
                    variant: GemvVariant::Prerotated,
                    residual: None,
                    gate: None,
                    up: None,
                },
            )
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))
        }
    }
}

/// Fused RMSNorm + FWHT rotation for a batch of MagnumQuant GEMVs sharing x.
///
/// Replaces the split `rmsnorm_f32` + `rotate_x_for_mq` pair with a single kernel
/// launch (Phase 3.6 kernel fusion). The caller should subsequently use
/// `weight_gemv_prerotated` with the returned `Option<&GpuTensor>`:
///
/// - MQ4 `sample_weight`: launches `fused_rmsnorm_rotate_mq`, writes FWHT(rmsnorm(x))
///   into `x_rot_scratch`, returns `Some(x_rot_scratch)`.
/// - MQ8 `sample_weight`: not yet supported by the fused kernel, falls back to
///   plain `rmsnorm_f32` + `rotate_quantize_x_mq8` (the INT8 quantize step can't
///   share LDS with rmsnorm the same way). Returns `None` — MQ8 consumes the
///   internal `mq_x_q8` buffer inside `weight_gemv_prerotated`.
/// - Any other dtype: plain `rmsnorm_f32` into `tmp`, returns `None`.
/// Phase A Stage A — batched AWQ-aware dispatch helper.
///
/// Wraps `Gpu::fused_rmsnorm_rotate_mq_batched` with AWQ-aware kernel
/// selection: if the upcoming linear (`next_linear`) carries an AWQ
/// scale sidecar, dispatch the `_awq_batched` kernel which divides
/// activations by `awq_scale[i]` before the FWHT. Otherwise use the
/// standard non-AWQ kernel.
///
/// Callers in qwen35.rs forward path pass the WeightTensor of the
/// FIRST linear after the rotation (e.g. `layer.wqkv` for LinearAttention,
/// `layer.wq` for FullAttention with separate Q/K/V, `ffn.router` for
/// MoE preamble). For fused QKV: Q/K/V share the same input tensor and
/// hence the same imatrix → byte-identical AWQ scales; picking any
/// of them is mathematically correct.
pub fn fused_rmsnorm_rotate_mq_batched_for(
    gpu: &mut Gpu,
    x: &GpuTensor,
    norm_weight: &GpuTensor,
    next_linear: &WeightTensor,
    x_rot: &GpuTensor,
    k: usize,
    eps: f32,
    batch_size: usize,
) -> HipResult<()> {
    if let Some(awq) = next_linear.awq_scale.as_ref() {
        gpu.fused_rmsnorm_rotate_mq_awq_batched(x, norm_weight, awq, x_rot, k, eps, batch_size)
    } else {
        gpu.fused_rmsnorm_rotate_mq_batched(x, norm_weight, x_rot, k, eps, batch_size)
    }
}

/// Lever 1 — Fused RMSNorm + PARO4G128T per-group Givens rotation.
///
/// When `next_linear` is PARO4G128T and `HIPFIRE_PARO_FUSE_RMSNORM` is enabled
/// (default: on, opt-out with `=0`), runs a single fused kernel that produces
/// BOTH x_rot (for the immediate prerotated GEMV on next_linear) AND
/// post-rmsnorm x_norm (written into `tmp` for subsequent linears in the
/// same residual block). Saves 1 launch vs the separated rmsnorm_f32 +
/// paro4g128t_rotate path.
///
/// Returns `Some(x_rot_scratch)` if fused — caller should run
/// `gemv_paro4g128t_prerotated` for `next_linear`, then standard
/// `weight_gemv` for subsequent paro linears (they consume `tmp`).
///
/// Returns `None` if fusion was skipped (non-PARO dtype or env opt-out) —
/// in that case `tmp` contains plain rmsnorm output and caller should use
/// `weight_gemv` as usual.
pub fn fused_rmsnorm_rotate_for_paro<'a>(
    gpu: &mut Gpu,
    next_linear: &WeightTensor,
    x: &GpuTensor,
    norm_weight: &GpuTensor,
    tmp: &GpuTensor,
    x_rot_scratch: &'a GpuTensor,
    eps: f32,
) -> HipResult<Option<&'a GpuTensor>> {
    // IMPORTANT: callers chain this AFTER `fused_rmsnorm_rotate_for_mq`,
    // which already runs rmsnorm_f32 in its non-MQ fallthrough. So when we
    // return None (opt-out or wrong dtype), `tmp` already contains the
    // rmsnorm output from the prior call — DO NOT run rmsnorm_f32 again.
    //
    // STATUS: Lever 1 falsified at -2.4% on 0.8B PARO4G128T (gfx1201, 2026-05-22).
    // The single-workgroup fused kernel runs ~K-rotation serially within one block,
    // losing the M/8 cross-CU parallelism that the split rotate kernel (grid=[K/128])
    // gets for free. Saves ~10µs launch overhead but adds ~30-70µs serial rotate
    // time per call. Net loss on every site. Default OFF; explicit opt-in for
    // research / future-redesign comparison.
    let opt_in = std::env::var("HIPFIRE_PARO_FUSE_RMSNORM")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if !opt_in {
        return Ok(None);
    }
    match next_linear.gpu_dtype {
        DType::ParoQ4G128 => {
            // Fast path: fused kernel emits x_rot (for next_linear's
            // prerotated GEMV) + tmp (post-rmsnorm x for subsequent linears).
            // Math identity: the kernel computes the same rmsnorm into tmp
            // that fused_rmsnorm_rotate_for_mq's fallthrough would have, so
            // overwriting tmp here is fine.
            gpu.fused_rmsnorm_paro4g128t_rotate(
                &next_linear.buf,
                x,
                norm_weight,
                x_rot_scratch,
                Some(tmp),
                next_linear.m,
                next_linear.k,
                eps,
            )?;
            Ok(Some(x_rot_scratch))
        }
        _ => {
            // Non-PARO dtype: tmp already has rmsnorm output from prior call.
            Ok(None)
        }
    }
}

pub fn fused_rmsnorm_rotate_for_mq<'a>(
    gpu: &mut Gpu,
    sample_weight: &WeightTensor,
    x: &GpuTensor,
    norm_weight: &GpuTensor,
    tmp: &GpuTensor,
    x_rot_scratch: &'a GpuTensor,
    eps: f32,
) -> HipResult<Option<&'a GpuTensor>> {
    match sample_weight.gpu_dtype {
        DType::MQ4G256
        | DType::MQ6G256
        | DType::MQ3G256
        | DType::MQ2G256
        | DType::MQ2G256Lloyd
        | DType::MQ3G256Lloyd
        | DType::MQ4G256Lloyd
        | DType::MFP4G32 => {
            // Phase A Stage A — AWQ-aware dispatch. When the upcoming linear
            // carries an AWQ scale sidecar, use the AWQ variant of the fused
            // kernel which divides activations by `awq_scale[i]` before the
            // FWHT rotation, completing the math `(W·s) · (x/s) = W·x`.
            // For Stage A specifically, only MQ4G256 actually emits AWQ
            // sidecars from the quantizer (`--awq` flag), but routing all
            // MQ-family dtypes through the AWQ check is correct + cheap.
            if let Some(awq) = sample_weight.awq_scale.as_ref() {
                gpu.fused_rmsnorm_rotate_mq_awq(
                    x,
                    norm_weight,
                    awq,
                    x_rot_scratch,
                    sample_weight.k,
                    eps,
                )?;
            } else {
                gpu.fused_rmsnorm_rotate_mq(x, norm_weight, x_rot_scratch, sample_weight.k, eps)?;
            }
            Ok(Some(x_rot_scratch))
        }
        DType::MQ8G256 => {
            // MQ8 rotate+quantize produces INT8 scratch; can't fuse with rmsnorm the
            // same way. Keep the split path for now.
            gpu.rmsnorm_f32(x, norm_weight, tmp, eps)?;
            gpu.rotate_quantize_x_mq8(tmp, sample_weight.k)?;
            Ok(None)
        }
        _ => {
            gpu.rmsnorm_f32(x, norm_weight, tmp, eps)?;
            Ok(None)
        }
    }
}

/// Pre-rotate x once for a batch of MagnumQuant weight GEMVs that share the same input.
///
/// - MQ4: writes FWHT(x) into `x_rot_scratch`, returns `Some(x_rot_scratch)`.
///   Pass the returned buffer to `weight_gemv_prerotated` for each MQ4 call.
/// - MQ8: rotates+quantizes x into the GPU's internal INT8 scratch, returns `None`.
///   Subsequent `weight_gemv_prerotated` calls pick up the internal buffers automatically.
/// - Any other dtype: no-op, returns `None` (caller should use plain `x`).
///
/// `sample_weight` is any weight from the batch — only its `gpu_dtype` and `k` are read.
pub fn rotate_x_for_mq<'a>(
    gpu: &mut Gpu,
    sample_weight: &WeightTensor,
    x: &GpuTensor,
    x_rot_scratch: &'a GpuTensor,
) -> HipResult<Option<&'a GpuTensor>> {
    match sample_weight.gpu_dtype {
        DType::MQ4G256
        | DType::MQ6G256
        | DType::MQ3G256
        | DType::MQ2G256
        | DType::MQ2G256Lloyd
        | DType::MQ3G256Lloyd
        | DType::MQ4G256Lloyd
        | DType::MFP4G32 => {
            // Phase A Stage A — F2: route to AWQ variant when the
            // downstream linear (the GEMV consuming x_rot) carries an
            // `awq_scale` sidecar. `sample_weight` IS that downstream
            // linear (callers pass o_proj / out_proj here). For Stage A
            // pre-F2 quants, awq_scale is None on these tensors so the
            // non-AWQ kernel runs — byte-identical to pre-F2 behavior.
            if let Some(awq) = sample_weight.awq_scale.as_ref() {
                gpu.rotate_x_mq_awq(x, awq, x_rot_scratch, sample_weight.k)?;
            } else {
                gpu.rotate_x_mq(x, x_rot_scratch, sample_weight.k)?;
            }
            Ok(Some(x_rot_scratch))
        }
        DType::MQ8G256 => {
            gpu.rotate_quantize_x_mq8(x, sample_weight.k)?;
            Ok(None)
        }
        _ => Ok(None),
    }
}

/// Phase A Stage A — F2: standalone AWQ-aware variant of `rotate_x_mq`.
///
/// Wraps `Gpu::rotate_x_mq` / `Gpu::rotate_x_mq_awq` with AWQ-aware
/// dispatch. The `next_linear` is the downstream weight that consumes
/// `x_rot` (typically `o_proj` / `out_proj` / `down_proj`). When its
/// `awq_scale` is `Some`, dispatches the AWQ kernel which divides
/// activations by `awq_scale[i]` before the FWHT — completes the math
/// `(W·s) · (x/s) = W·x`.
///
/// Byte-identical to `Gpu::rotate_x_mq` on .hfq files without AWQ
/// sidecars (which is the only state that exists pre-F2).
pub fn rotate_x_mq_for(
    gpu: &mut Gpu,
    next_linear: &WeightTensor,
    x: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
) -> HipResult<()> {
    if let Some(awq) = next_linear.awq_scale.as_ref() {
        gpu.rotate_x_mq_awq(x, awq, x_rot, k)
    } else {
        gpu.rotate_x_mq(x, x_rot, k)
    }
}

/// MQ4G128 activation rotate helper. Always takes the non-AWQ path —
/// no AWQ sidecar is supported for G128 weights in this PR.
/// The `_next_linear` argument is reserved for a future AWQ branch;
/// pass the upcoming weight tensor but its `awq_scale` is ignored here.
pub fn rotate_x_mq_128_for(
    gpu: &mut Gpu,
    _next_linear: &WeightTensor,
    x: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
) -> HipResult<()> {
    // NOTE: no AWQ branch for G128. If AWQ support for MQ4G128 is added
    // in a follow-up, mirror the AWQ branch from `rotate_x_mq_for` here.
    gpu.rotate_x_mq_128(x, x_rot, k)
}

/// ParoQuant single-token rotation: read x, write Givens-rotated
/// activation to x_rot via a single out-of-place kernel. Earlier
/// versions did `copy_d2d(x → x_rot)` then `givens_rotate(x_rot)`;
/// fusing into one launch eliminates an inter-node dependency that
/// the hipGraph dependency analyzer can fail to enforce (observed
/// numerical delta direct-vs-graph on gfx1151 / HIP 7.13).
pub fn rotate_x_paro_for(
    gpu: &mut Gpu,
    paro: &ParoRotation,
    x: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
) -> HipResult<()> {
    gpu.givens_rotate_to(
        x,
        x_rot,
        &paro.pairs,
        &paro.theta,
        &paro.channel_scales,
        1,
        k,
        paro.krot as usize,
    )
}

/// Phase A Stage A — F2: batched AWQ-aware variant of `rotate_x_mq`.
/// Grid.y is the batch dim. See `rotate_x_mq_for` for routing logic.
pub fn rotate_x_mq_batched_for(
    gpu: &mut Gpu,
    next_linear: &WeightTensor,
    x: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
    batch_size: usize,
) -> HipResult<()> {
    if let Some(awq) = next_linear.awq_scale.as_ref() {
        gpu.rotate_x_mq_awq_batched(x, awq, x_rot, k, batch_size)
    } else {
        gpu.rotate_x_mq_batched(x, x_rot, k, batch_size)
    }
}

/// Phase A Stage A — F2: standalone AWQ-aware variant of
/// `fused_silu_mul_rotate_mq`. The `down_proj_weight` is the downstream
/// linear consuming x_rot (e.g. `w_down` / `down_proj`). When its
/// `awq_scale` is `Some`, dispatches the AWQ kernel which divides
/// silu(gate)*up by `awq_scale[i]` before the FWHT.
///
/// Byte-identical to `Gpu::fused_silu_mul_rotate_mq` on .hfq files
/// without AWQ sidecars (pre-F2 state).
pub fn fused_silu_mul_rotate_mq_for(
    gpu: &mut Gpu,
    down_proj_weight: &WeightTensor,
    gate: &GpuTensor,
    up: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
) -> HipResult<()> {
    if let Some(awq) = down_proj_weight.awq_scale.as_ref() {
        gpu.fused_silu_mul_rotate_mq_awq(gate, up, awq, x_rot, k)
    } else {
        gpu.fused_silu_mul_rotate_mq(gate, up, x_rot, k)
    }
}

/// Phase A Stage A — F2: batched AWQ-aware variant of
/// `fused_silu_mul_rotate_mq`. Grid.y is the batch dim.
pub fn fused_silu_mul_rotate_mq_batched_for(
    gpu: &mut Gpu,
    down_proj_weight: &WeightTensor,
    gate: &GpuTensor,
    up: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
    batch_size: usize,
) -> HipResult<()> {
    if let Some(awq) = down_proj_weight.awq_scale.as_ref() {
        gpu.fused_silu_mul_rotate_mq_awq_batched(gate, up, awq, x_rot, k, batch_size)
    } else {
        gpu.fused_silu_mul_rotate_mq_batched(gate, up, x_rot, k, batch_size)
    }
}

/// GEMV with optional pre-rotated x for MagnumQuant weights.
///
/// - MQ4 + `x_rot = Some(..)`: calls the arch-tuned HFQ4 GEMV on the pre-rotated buffer,
///   skipping the per-call FWHT pass. Use with `rotate_x_for_mq` to batch rotations across
///   multiple projections that share the same input (Q/K/V, gate/up).
/// - MQ4 + `x_rot = None`: falls back to the auto-rotate path in `weight_gemv`.
/// - MQ8: uses the internal x_q8/x_scales set by `rotate_quantize_x_mq8`; caller must have
///   called `rotate_x_for_mq` (which invokes that helper) before this.
/// - Any other dtype: `x_rot` is ignored; equivalent to `weight_gemv`.
pub fn weight_gemv_prerotated(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    x_rot: Option<&GpuTensor>,
    y: &GpuTensor,
) -> HipResult<()> {
    if dense_prerotated_gemv_route(w.gpu_dtype) == DensePrerotatedGemvRoute::Mq6Prerotated {
        if let Some(xr) = x_rot {
            return gpu.gemv_mq6g256_prerotated(&w.buf, xr, y, w.m, w.k);
        }
        return weight_gemv(gpu, w, x, y);
    }

    match w.gpu_dtype {
        DType::MQ4G256 => {
            if let Some(xr) = x_rot {
                gpu.gemv_mq4g256_prerotated(&w.buf, xr, y, w.m, w.k)
            } else {
                weight_gemv(gpu, w, x, y)
            }
        }
        DType::MFP4G32 => {
            if let Some(xr) = x_rot {
                gpu.gemv_mfp4g32_prerotated(&w.buf, xr, y, w.m, w.k)
            } else {
                weight_gemv(gpu, w, x, y)
            }
        }
        DType::MQ3G256 => {
            if let Some(xr) = x_rot {
                gpu.gemv_mq3g256_prerotated(&w.buf, xr, y, w.m, w.k)
            } else {
                weight_gemv(gpu, w, x, y)
            }
        }
        DType::MQ2G256 => {
            if let Some(xr) = x_rot {
                gpu.gemv_mq2g256_prerotated(&w.buf, xr, y, w.m, w.k)
            } else {
                weight_gemv(gpu, w, x, y)
            }
        }
        DType::MQ2G256Lloyd => {
            if let Some(xr) = x_rot {
                gpu.gemv_mq2g256_lloyd(&w.buf, xr, y, w.m, w.k)
            } else {
                weight_gemv(gpu, w, x, y)
            }
        }
        DType::MQ3G256Lloyd => {
            if let Some(xr) = x_rot {
                gpu.gemv_mq3g256_lloyd(&w.buf, xr, y, w.m, w.k)
            } else {
                weight_gemv(gpu, w, x, y)
            }
        }
        DType::MQ4G256Lloyd => {
            if let Some(xr) = x_rot {
                gpu.gemv_mq4g256_lloyd(&w.buf, xr, y, w.m, w.k)
            } else {
                weight_gemv(gpu, w, x, y)
            }
        }
        DType::MQ8G256 => gpu.gemv_mq8g256_prerotated(&w.buf, y, w.m, w.k),
        _ => weight_gemv(gpu, w, x, y),
    }
}

/// Weight GEMV with fused residual add: `y += W * x`.
///
/// For HFQ4-G256 weights, routes through `gemv_hfq4g256_residual`, which
/// saves one `add_inplace_f32` launch per residual stream update.
///
/// For MQ4 weights, performs `rotate_x_mq` into the internal scratch and
/// then calls the residual GEMV against the rotated x. Equivalent to the
/// standard prerotated path plus a fused residual epilogue.
///
/// For any other dtype, falls back to plain `weight_gemv` followed by an
/// explicit `add_inplace_f32` — same observable behavior as before.
pub fn weight_gemv_residual(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
) -> HipResult<()> {
    match dense_residual_route(w.gpu_dtype) {
        DenseResidualRoute::Hfq6ResidualDirect => {
            return gpu.gemv_hfq6g256_residual(&w.buf, x, y, w.m, w.k);
        }
        DenseResidualRoute::Mq6RotateThenHfq6Residual => {
            // FWHT-rotate x into the shared mq_x_rot scratch, then dispatch
            // hfq6g256_residual against the rotated activations. Saves one
            // add_inplace_f32 launch per layer per token vs the generic path.
            gpu.ensure_mq_signs()?;
            let x_rot_alias = GpuTensor {
                buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            };
            // F2: AWQ-aware routing. `w` is the downstream linear that
            // consumes x_rot; route via _for helper so its awq_scale (if
            // any) is applied via the AWQ kernel variant.
            rotate_x_mq_for(gpu, w, x, &x_rot_alias, w.k)?;
            return gpu.gemv_hfq6g256_residual(&w.buf, &x_rot_alias, y, w.m, w.k);
        }
        DenseResidualRoute::Unclassified => {}
    }

    match w.gpu_dtype {
        DType::F16 => gpu.gemv_f16_xf32_residual(&w.buf, x, y, w.m, w.k),
        DType::HFQ4G256 => gpu.gemv_hfq4g256_residual(&w.buf, x, y, w.m, w.k),
        DType::ParoQ4G128 if std::env::var_os("HIPFIRE_PARO_PREROTATE").is_some() => {
            gpu.ensure_mq_signs()?;
            let x_rot_alias = GpuTensor {
                buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            };
            gpu.gemv_paro4g128_residual_with_prerotate(&w.buf, x, y, &x_rot_alias, w.m, w.k)
        }
        DType::ParoQ4G128 => gpu.gemv_paro4g128_residual(&w.buf, x, y, w.m, w.k),
        DType::HFQ3G256 => gpu.gemv_hfq3g256_residual(&w.buf, x, y, w.m, w.k),
        DType::MQ4G256 => {
            gpu.ensure_mq_signs()?;
            let x_rot_alias = GpuTensor {
                buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            };
            // F2: AWQ-aware routing. `w` is the downstream linear that
            // consumes x_rot; route via _for helper so its awq_scale (if
            // any) is applied via the AWQ kernel variant.
            rotate_x_mq_for(gpu, w, x, &x_rot_alias, w.k)?;
            gpu.gemv_hfq4g256_residual(&w.buf, &x_rot_alias, y, w.m, w.k)
        }
        DType::MQ3G256 => {
            // FWHT-rotate x into the shared mq_x_rot scratch, then dispatch
            // hfq3g256_residual against the rotated activations. Saves one
            // add_inplace_f32 launch per layer per token vs the generic
            // path. gfx1100 picks the K4-unrolled chip variant (commit
            // 0003103, 9B MQ3 decode 114 to 141 tok/s).
            gpu.ensure_mq_signs()?;
            let x_rot_alias = GpuTensor {
                buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            };
            // F2: AWQ-aware routing. `w` is the downstream linear that
            // consumes x_rot; route via _for helper so its awq_scale (if
            // any) is applied via the AWQ kernel variant.
            rotate_x_mq_for(gpu, w, x, &x_rot_alias, w.k)?;
            gpu.gemv_hfq3g256_residual(&w.buf, &x_rot_alias, y, w.m, w.k)
        }
        DType::MQ3G256Lloyd => {
            // FWHT-rotate x into the shared mq_x_rot scratch, then dispatch
            // the Lloyd residual GEMV. Eliminates the alloc + gemv +
            // add_inplace_f32 + free fallback chain (~4.4% of decode time on
            // 9B Lloyd-MQ3 per the 2026-05-06 decode profile). gfx1100
            // picks the K4 + LDS-codebook chip variant.
            gpu.ensure_mq_signs()?;
            let x_rot_alias = GpuTensor {
                buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            };
            // F2: AWQ-aware routing. `w` is the downstream linear that
            // consumes x_rot; route via _for helper so its awq_scale (if
            // any) is applied via the AWQ kernel variant.
            rotate_x_mq_for(gpu, w, x, &x_rot_alias, w.k)?;
            gpu.gemv_mq3g256_lloyd_residual(&w.buf, &x_rot_alias, y, w.m, w.k)
        }
        DType::MQ4G256Lloyd => {
            // Same fusion shape as MQ3-Lloyd; gfx1100 picks the K4 + LDS +
            // single-acc fast variant (see kernel header for why single-acc).
            gpu.ensure_mq_signs()?;
            let x_rot_alias = GpuTensor {
                buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            };
            rotate_x_mq_for(gpu, w, x, &x_rot_alias, w.k)?;
            gpu.gemv_mq4g256_lloyd_residual(&w.buf, &x_rot_alias, y, w.m, w.k)
        }
        _ => {
            let tmp = gpu.alloc_tensor(&[w.m], DType::F32)?;
            weight_gemv(gpu, w, x, &tmp)?;
            gpu.add_inplace_f32(y, &tmp)?;
            gpu.free_tensor(tmp)?;
            Ok(())
        }
    }
}

/// SwiGLU FFN epilogue fused into the w_down input stage for MQ4 weights.
///
/// Replaces:
///   silu_mul_f32(gate, up, ffn_hidden)  // eliminated for MQ4
///   weight_gemv_residual(w_down, ffn_hidden, x)
/// with (for MQ4):
///   fused_silu_mul_rotate_mq(gate, up, mq_x_rot)   // one kernel
///   gemv_hfq4g256_residual(w_down, mq_x_rot, x)    // fused residual add
/// so the entire w_down epilogue is two launches instead of four
/// (silu_mul + rotate + gemv + add_inplace → fused_silu_rotate + gemv_residual).
///
/// Non-MQ path falls back to the pre-Phase-3.8 sequence (silu_mul_f32 +
/// weight_gemv_residual). Byte-equivalent modulo FP reordering on the
/// FWHT butterfly, which is the same butterfly as the standalone path.
pub fn weight_gemv_swiglu_residual(
    gpu: &mut Gpu,
    w_down: &WeightTensor,
    gate: &GpuTensor,
    up: &GpuTensor,
    ffn_hidden_scratch: &GpuTensor,
    x: &GpuTensor,
) -> HipResult<()> {
    if dense_swiglu_residual_route(w_down.gpu_dtype)
        == DenseSwigluResidualRoute::Mq6RotateThenHfq6Residual
    {
        // MQ6 down + residual fusion: same FWHT rotate + fused-residual
        // pattern as MQ3 / MQ4, dispatched against the HFQ6 kernel.
        gpu.ensure_mq_signs()?;
        let x_rot_alias = GpuTensor {
            buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
            shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
            dtype: DType::F32,
        };
        // F2: AWQ-aware routing for the down_proj input stage.
        // `w_down` IS the downstream weight; route through _for helper.
        fused_silu_mul_rotate_mq_for(gpu, w_down, gate, up, &x_rot_alias, w_down.k)?;
        return gpu.gemv_hfq6g256_residual(&w_down.buf, &x_rot_alias, x, w_down.m, w_down.k);
    }
    match w_down.gpu_dtype {
        DType::MQ4G256 => {
            gpu.ensure_mq_signs()?;
            let x_rot_alias = GpuTensor {
                buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            };
            // F2: AWQ-aware routing for the down_proj input stage.
            // `w_down` IS the downstream weight; route through _for helper.
            fused_silu_mul_rotate_mq_for(gpu, w_down, gate, up, &x_rot_alias, w_down.k)?;
            gpu.gemv_hfq4g256_residual(&w_down.buf, &x_rot_alias, x, w_down.m, w_down.k)
        }
        DType::MQ3G256 => {
            // Same shape as MQ4: silu(gate)*up rotated through the FWHT into
            // the shared mq_x_rot scratch, then the fused HFQ3 residual GEMV
            // does the down projection plus residual add in one launch. Saves
            // one silu_mul_f32 launch and one add_inplace_f32 launch versus
            // the four-step generic path.
            gpu.ensure_mq_signs()?;
            let x_rot_alias = GpuTensor {
                buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            };
            // F2: AWQ-aware routing for the down_proj input stage.
            // `w_down` IS the downstream weight; route through _for helper.
            fused_silu_mul_rotate_mq_for(gpu, w_down, gate, up, &x_rot_alias, w_down.k)?;
            gpu.gemv_hfq3g256_residual(&w_down.buf, &x_rot_alias, x, w_down.m, w_down.k)
        }
        DType::MQ3G256Lloyd => {
            // Same fusion as MQ3 / MQ4 / MQ6: silu(gate)*up rotated into
            // mq_x_rot, then the Lloyd-MQ3 residual GEMV does down +
            // residual in one launch. Saves one silu_mul_f32 launch versus
            // the generic three-step path (silu_mul + rotate + gemv_residual).
            gpu.ensure_mq_signs()?;
            let x_rot_alias = GpuTensor {
                buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            };
            // F2: AWQ-aware routing for the down_proj input stage.
            // `w_down` IS the downstream weight; route through _for helper.
            fused_silu_mul_rotate_mq_for(gpu, w_down, gate, up, &x_rot_alias, w_down.k)?;
            gpu.gemv_mq3g256_lloyd_residual(&w_down.buf, &x_rot_alias, x, w_down.m, w_down.k)
        }
        DType::MQ4G256Lloyd => {
            // Same fusion shape as MQ3-Lloyd; gfx1100 picks the K4 + LDS +
            // single-acc fast variant of the residual GEMV.
            gpu.ensure_mq_signs()?;
            let x_rot_alias = GpuTensor {
                buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            };
            fused_silu_mul_rotate_mq_for(gpu, w_down, gate, up, &x_rot_alias, w_down.k)?;
            gpu.gemv_mq4g256_lloyd_residual(&w_down.buf, &x_rot_alias, x, w_down.m, w_down.k)
        }
        DType::ParoQ4G128 if std::env::var_os("HIPFIRE_PARO_PREROTATE").is_some() => {
            gpu.ensure_mq_signs()?;
            let x_rot_alias = GpuTensor {
                buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            };
            gpu.gemv_paro4g128_swiglu_residual_with_prerotate(
                &w_down.buf,
                gate,
                up,
                x,
                &x_rot_alias,
                w_down.m,
                w_down.k,
            )
        }
        DType::ParoQ4G128 if std::env::var_os("HIPFIRE_PARO_SWIGLU_FUSED").is_some() => {
            gpu.gemv_paro4g128_swiglu_residual(&w_down.buf, gate, up, x, w_down.m, w_down.k)
        }
        DType::ParoQ4G128 => {
            gpu.ensure_mq_signs()?;
            let x_rot_alias = GpuTensor {
                buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            };
            gpu.gemv_paro4g128t_swiglu_residual_with_prerotate(
                &w_down.buf,
                gate,
                up,
                x,
                &x_rot_alias,
                w_down.m,
                w_down.k,
            )
        }
        _ => {
            // Non-MQ fallback: plain two-step.
            gpu.silu_mul_f32(gate, up, ffn_hidden_scratch)?;
            weight_gemv_residual(gpu, w_down, ffn_hidden_scratch, x)
        }
    }
}

/// Batched weight GEMM: y[b] = W * x[b] for all batch elements.
/// x: [batch_size × K], y: [batch_size × M]. Falls back to repeated GEMV for unsupported formats.
pub fn weight_gemm(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    batch_size: usize,
) -> HipResult<()> {
    match w.gpu_dtype {
        DType::F16 => gpu.gemm_f16_x_f32_wmma(&w.buf, x, y, w.m, w.k, batch_size),
        DType::BF16 => gpu.gemm_bf16_x_bf16_wmma(&w.buf, x, y, w.m, w.k, batch_size),
        DType::HFQ4G256 => gpu.gemm_hfq4g256(&w.buf, x, y, w.m, w.k, batch_size),
        DType::HFQ4G128 => gpu.gemm_hfq4g128(&w.buf, x, y, w.m, w.k, batch_size),
        DType::W8A8Ref => {
            // W8A8 reference path. buf = [M*K int8 weights | M f32 per-channel scales].
            // Quantize activations per-token to int8, iu8 WMMA, dequant by w·x scale.
            // Per-call scratch (alloc/free, mirrors the fallback) — correctness over
            // perf for the reference floor; warn so the slow path is visible.
            warn_generic_once("weight_gemm", "W8A8", KernelMode::Prefill, &gpu.arch, Quality::Reference);
            let xq = gpu.alloc_tensor(&[batch_size * w.k], DType::Raw)?; // int8
            let xs = gpu.alloc_tensor(&[batch_size], DType::F32)?;
            let yi = gpu.alloc_tensor(&[batch_size * w.m * 4], DType::Raw)?; // int32
            // Per-channel scale lives in the byte tail of buf (W8A8Ref is byte-level).
            let w_scale = w.buf.sub_offset(w.m * w.k, w.m * 4);
            gpu.quantize_act_int8_per_token(x, &xq, &xs, batch_size, w.k)?;
            gpu.gemm_iu8_i32_wmma(&w.buf, &xq, &yi, w.m, w.k, batch_size)?;
            gpu.dequant_i32_rowcol(&yi, &xs, &w_scale, y, batch_size, w.m)?;
            gpu.free_tensor(xq)?;
            gpu.free_tensor(xs)?;
            gpu.free_tensor(yi)?;
            Ok(())
        }
        _ => {
            // Generic fallback: no batched kernel for this weight dtype, so loop a
            // per-token GEMV. Correct but slow — warn once per (dtype, mode, arch) so
            // the missing batched-kernel coverage shows up in logs (reference layer).
            let mode = if batch_size > 1 {
                KernelMode::Prefill
            } else {
                KernelMode::Decode
            };
            warn_generic_once(
                "weight_gemm",
                &format!("{:?}", w.gpu_dtype),
                mode,
                &gpu.arch,
                Quality::Reference,
            );
            // Fallback: repeated GEMV (no batched kernel for this format)
            let x_tok = gpu.alloc_tensor(&[w.k], DType::F32)?;
            let y_tok = gpu.alloc_tensor(&[w.m], DType::F32)?;
            for b in 0..batch_size {
                gpu.hip
                    .memcpy_dtod_at(&x_tok.buf, 0, &x.buf, b * w.k * 4, w.k * 4)?;
                weight_gemv(gpu, w, &x_tok, &y_tok)?;
                gpu.hip
                    .memcpy_dtod_at(&y.buf, b * w.m * 4, &y_tok.buf, 0, w.m * 4)?;
            }
            gpu.free_tensor(x_tok)?;
            gpu.free_tensor(y_tok)?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rdna_compute::DType;

    #[test]
    fn mq6_dense_decode_routes_through_hfq6_family() {
        assert_eq!(
            dense_gemv_route(DType::MQ6G256),
            DenseGemvRoute::Mq6RotateThenMq6Prerotated
        );
        assert_eq!(
            dense_gemv_route(DType::HFQ6G256),
            DenseGemvRoute::Hfq6Direct
        );
        assert_eq!(
            dense_prerotated_gemv_route(DType::MQ6G256),
            DensePrerotatedGemvRoute::Mq6Prerotated
        );
        assert_eq!(
            dense_residual_route(DType::MQ6G256),
            DenseResidualRoute::Mq6RotateThenHfq6Residual
        );
        assert_eq!(
            dense_residual_route(DType::HFQ6G256),
            DenseResidualRoute::Hfq6ResidualDirect
        );
        assert_eq!(
            dense_swiglu_residual_route(DType::MQ6G256),
            DenseSwigluResidualRoute::Mq6RotateThenHfq6Residual
        );

        assert_eq!(
            dense_gemv_route(DType::MQ4G256),
            DenseGemvRoute::Unclassified
        );
        assert_eq!(
            dense_residual_route(DType::MQ4G256),
            DenseResidualRoute::Unclassified
        );
        assert_eq!(
            dense_swiglu_residual_route(DType::MQ4G256),
            DenseSwigluResidualRoute::Unclassified
        );
    }
}
