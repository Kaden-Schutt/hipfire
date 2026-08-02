// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Parent-checkpoint Hyper-Connections + RMSNorm on GPU.
//!
//! Operator authority (transcribed, not guessed):
//! - `.codeinsight+research/ds4-parent-ref/inference/model.py`
//!   `Block.hc_pre` / `hc_post` / `hc_head` (680-716)
//! - `.codeinsight+research/ds4-parent-ref/inference/kernel.py`
//!   `hc_split_sinkhorn` (372-439)
//!
//! # Why this is not the MQ2R `mhc_pre` path
//!
//! Production `forward.rs::mhc_pre` funnels through F16-weight kernels
//! (`hc_compute_control`, `hc_apply_alpha`, `hc_finalize_*`) that:
//! 1. hardcode `norm_eps = 1e-6` inside the HIP source,
//! 2. bake `base` into the control projection before α-rescale, and
//! 3. take `post_scale` from `HIPFIRE_DEEPSEEK4_POST_SCALE` (default 1.5)
//!    instead of the checkpoint's per-segment `hc_*_scale[1]` with the
//!    reference's fixed `2 * sigmoid(...)`.
//!
//! The parent checkpoint stores HC tensors as **F32** (`hc_*_fn`,
//! `hc_*_base`, `hc_*_scale`) and the reference applies
//! `sigmoid(mixes * scale + base)` with scale taken from config tensors —
//! never from an env default. This module therefore calls format-agnostic
//! F32 Gpu kernels (`gemm_f32_register_tiled`, `hc_sinkhorn_4x4_batched`,
//! `hc_input_map_4stream_batched`, `hc_mix_4stream_batched`,
//! `rmsnorm_batched`) and performs the scale/base/sigmoid split in lock-
//! step with `kernel.py`, not the MQ2R α path.
//!
//! # `hc_sinkhorn_4x4.hip` vs `hc_split_sinkhorn_kernel`
//!
//! The HIP kernel documents itself as mirroring the TileLang reference.
//! Verified against `kernel.py:401-423`:
//! 1. row-softmax + eps          (`v/rs + eps`)
//! 2. column normalize           (`v / (cs + eps)`)
//! 3. `iters - 1` further row/col normalizations
//!
//! That order matches. The HIP path does **not** apply pre/post sigmoid
//! (those live in the split step before sinkhorn). Parent code therefore
//! builds `comb` logits with scale/base first, then hands only the 4×4
//! comb region to `hc_sinkhorn_4x4_batched`.
//!
//! # `gfx1151` flags on HC helpers
//!
//! `hc_compute_control` gates a vec4 ILP variant when
//! `arch == "gfx1151"` and the MQ2R route / env opt-in is set. That is an
//! F16-weight decode optimisation and is **not** used here (parent weights
//! are F32; gfx942 has no such gate on the F32 GEMM / sinkhorn / mix
//! kernels this module calls). `rmsnorm_f32` similarly gates a warp-reduce
//! symbol only on gfx1151 under an env flag; we call `rmsnorm_batched`
//! which always launches the portable `rmsnorm_f32` symbol.

use crate::parent::Ds4ParentBackend;
use rdna_compute::{DType, Gpu, GpuTensor};

#[inline]
fn err(msg: impl Into<String>) -> String {
    format!("deepseek4 parent: {}", msg.into())
}

/// Per-layer HC parameters already resident as F32 GpuTensors.
///
/// For `hc_pre` / layer HC:
/// - `fn_mat`: `[mix_hc, hc_mult * dim]` with `mix_hc = (2 + hc_mult) * hc_mult`
/// - `base`:   `[mix_hc]`
/// - `scale`:  `[3]`
///
/// For `hc_head` the same struct is reused with the head shapes:
/// - `fn_mat`: `[hc_mult, hc_mult * dim]`
/// - `base`:   `[hc_mult]`
/// - `scale`:  `[1]` (only `scale[0]` is consumed)
pub struct ParentHcParams<'a> {
    pub fn_mat: &'a GpuTensor, // F32
    pub base: &'a GpuTensor,   // F32
    pub scale: &'a GpuTensor,  // F32
}

#[inline]
fn require_f32(t: &GpuTensor, name: &str) -> Result<(), String> {
    if t.dtype != DType::F32 {
        return Err(err(format!(
            "{name} must be F32 (got {:?})",
            t.dtype
        )));
    }
    Ok(())
}

#[inline]
fn require_elems(t: &GpuTensor, n: usize, name: &str) -> Result<(), String> {
    if t.numel() < n {
        return Err(err(format!(
            "{name} too short (have {} need {n})",
            t.numel()
        )));
    }
    Ok(())
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// HC `post` multiplier. Reference (`kernel.py:394`) hardcodes `2 * sigmoid(...)`.
///
/// Default is **2.0** (reference-faithful). Override only for diagnostics via
/// `HIPFIRE_DEEPSEEK4_PARENT_POST_SCALE` — never adopt production's serving
/// default of 1.5 as the parent path's permanent value.
fn hc_post_scale() -> f32 {
    use std::sync::LazyLock;
    static SCALE: LazyLock<f32> = LazyLock::new(|| {
        let v = std::env::var("HIPFIRE_DEEPSEEK4_PARENT_POST_SCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2.0);
        if (v - 2.0f32).abs() > f32::EPSILON {
            eprintln!(
                "deepseek4 parent: DIAGNOSTIC HIPFIRE_DEEPSEEK4_PARENT_POST_SCALE={v} \
                 (reference default is 2.0; do not promote serving-tuned values)"
            );
        }
        v
    });
    *SCALE
}


/// Host-side `hc_split_sinkhorn` control split (pre/post + comb logits),
/// matching `kernel.py:391-396` **before** the sinkhorn iterations.
///
/// Sinkhorn itself is left to the GPU kernel so that path is exercised.
fn split_pre_post_comb_logits(
    mixes: &[f32],
    scale: &[f32],
    base: &[f32],
    rows: usize,
    hc_mult: usize,
    hc_eps: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let mix_hc = (2 + hc_mult) * hc_mult;
    if scale.len() < 3 {
        return Err(err(format!("hc_scale len {} < 3", scale.len())));
    }
    if base.len() < mix_hc {
        return Err(err(format!(
            "hc_base len {} < mix_hc {mix_hc}",
            base.len()
        )));
    }
    if mixes.len() < rows * mix_hc {
        return Err(err(format!(
            "mixes len {} < rows*mix_hc {}",
            mixes.len(),
            rows * mix_hc
        )));
    }
    let s0 = scale[0];
    let s1 = scale[1];
    let s2 = scale[2];
    let mut pre = vec![0.0f32; rows * hc_mult];
    let mut post = vec![0.0f32; rows * hc_mult];
    let mut comb = vec![0.0f32; rows * hc_mult * hc_mult];
    for r in 0..rows {
        let mbase = r * mix_hc;
        for j in 0..hc_mult {
            // pre = sigmoid(mixes * scale[0] + base) + eps
            pre[r * hc_mult + j] =
                sigmoid_f32(mixes[mbase + j] * s0 + base[j]) + hc_eps;
            // post = post_scale * sigmoid(mixes * scale[1] + base)
            // Reference hardcodes post_scale=2.0 (kernel.py:394).
            post[r * hc_mult + j] = hc_post_scale()
                * sigmoid_f32(mixes[mbase + j + hc_mult] * s1 + base[j + hc_mult]);
        }
        let cbase = r * hc_mult * hc_mult;
        for j in 0..hc_mult {
            for k in 0..hc_mult {
                let idx = j * hc_mult + k + hc_mult * 2;
                comb[cbase + j * hc_mult + k] =
                    mixes[mbase + idx] * s2 + base[idx];
            }
        }
    }
    Ok((pre, post, comb))
}

/// `mixes[r, :] *= rsqrt(mean(x[r]^2) + eps)` with rsqrt over the full
/// flattened `hc_mult * dim` vector (not per stream) — `model.py:683-685`.
fn apply_flat_rsqrt_to_mixes(
    x: &[f32],
    mixes: &mut [f32],
    rows: usize,
    hc_dim: usize,
    mix_hc: usize,
    norm_eps: f32,
) {
    for r in 0..rows {
        let xbase = r * hc_dim;
        let mut acc = 0.0f64;
        for d in 0..hc_dim {
            let v = x[xbase + d] as f64;
            acc += v * v;
        }
        let rsqrt = (acc / hc_dim as f64 + norm_eps as f64).sqrt().recip() as f32;
        let mbase = r * mix_hc;
        for o in 0..mix_hc {
            mixes[mbase + o] *= rsqrt;
        }
    }
}

/// Project `x_flat [rows, k]` through F32 weight `w [m, k]` → `y [rows, m]`
/// via the register-tiled F32 GEMM (`y = x @ W^T`).
fn parent_f32_linear(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    m: usize,
    k: usize,
) -> Result<(), String> {
    require_f32(w, "linear weight")?;
    require_f32(x, "linear x")?;
    require_f32(y, "linear y")?;
    require_elems(w, m * k, "linear weight")?;
    require_elems(x, rows * k, "linear x")?;
    require_elems(y, rows * m, "linear y")?;
    gpu.gemm_f32_register_tiled(w, x, y, m, k, rows)
        .map_err(|e| err(format!("gemm_f32_register_tiled: {e:?}")))
}

/// Free a tensor, ignoring errors (scratch cleanup).
fn free_scratch(gpu: &mut Gpu, t: GpuTensor) {
    let _ = gpu.free_tensor(t);
}

/// `Block.hc_pre`. `x` is `[rows, hc_mult, dim]` f32. Writes `y` `[rows, dim]`
/// and returns the post/comb tensors `hc_post` needs via the provided buffers.
///
/// `post` is `[rows, hc_mult]`, `comb` is `[rows, hc_mult, hc_mult]`.
pub fn parent_hc_pre(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    x: &GpuTensor,
    p: ParentHcParams<'_>,
    rows: usize,
    hc_mult: usize,
    dim: usize,
    norm_eps: f32,
    sinkhorn_iters: i32,
    hc_eps: f32,
    y: &GpuTensor,
    post: &GpuTensor,
    comb: &GpuTensor,
) -> Result<(), String> {
    backend.ensure_device(gpu)?;
    if rows == 0 || hc_mult == 0 || dim == 0 {
        return Err(err("hc_pre: rows, hc_mult, dim must be > 0"));
    }
    if hc_mult != 4 {
        // Sinkhorn + input_map + mix kernels are hard-wired to HC_MULT=4.
        return Err(err(format!(
            "hc_pre: only hc_mult=4 is supported by the GPU kernels (got {hc_mult})"
        )));
    }
    if sinkhorn_iters < 1 {
        return Err(err(
            "hc_pre: sinkhorn_iters must be >= 1 (first pass is softmax+col)",
        ));
    }
    let hc_dim = hc_mult
        .checked_mul(dim)
        .ok_or_else(|| err("hc_pre: hc_dim overflow"))?;
    let mix_hc = (2 + hc_mult)
        .checked_mul(hc_mult)
        .ok_or_else(|| err("hc_pre: mix_hc overflow"))?;

    require_f32(x, "hc_pre x")?;
    require_f32(p.fn_mat, "hc_pre fn_mat")?;
    require_f32(p.base, "hc_pre base")?;
    require_f32(p.scale, "hc_pre scale")?;
    require_f32(y, "hc_pre y")?;
    require_f32(post, "hc_pre post")?;
    require_f32(comb, "hc_pre comb")?;
    require_elems(x, rows * hc_dim, "hc_pre x")?;
    require_elems(p.fn_mat, mix_hc * hc_dim, "hc_pre fn_mat")?;
    require_elems(p.base, mix_hc, "hc_pre base")?;
    require_elems(p.scale, 3, "hc_pre scale")?;
    require_elems(y, rows * dim, "hc_pre y")?;
    require_elems(post, rows * hc_mult, "hc_pre post")?;
    require_elems(comb, rows * hc_mult * hc_mult, "hc_pre comb")?;

    // 1. mixes = F.linear(x_flat, hc_fn)   — GPU F32 GEMM
    let mixes_t = gpu
        .alloc_tensor(&[rows, mix_hc], DType::F32)
        .map_err(|e| err(format!("hc_pre mixes alloc: {e:?}")))?;
    if let Err(e) = parent_f32_linear(gpu, p.fn_mat, x, &mixes_t, rows, mix_hc, hc_dim) {
        free_scratch(gpu, mixes_t);
        return Err(e);
    }

    // 2. Download x + mixes; apply flat rsqrt; split pre/post/comb logits.
    //    The rsqrt is over hc_mult*dim (model.py:683-685), which no existing
    //    F32 HC control kernel accepts as a parameter (hc_compute_control
    //    hardcodes 1e-6 and F16 weights). Host f32 keeps the contract exact.
    let x_host = match gpu.download_f32(x) {
        Ok(v) => v,
        Err(e) => {
            free_scratch(gpu, mixes_t);
            return Err(err(format!("hc_pre download x: {e:?}")));
        }
    };
    let mut mixes = match gpu.download_f32(&mixes_t) {
        Ok(v) => v,
        Err(e) => {
            free_scratch(gpu, mixes_t);
            return Err(err(format!("hc_pre download mixes: {e:?}")));
        }
    };
    free_scratch(gpu, mixes_t);
    apply_flat_rsqrt_to_mixes(&x_host, &mut mixes, rows, hc_dim, mix_hc, norm_eps);

    let scale = gpu
        .download_f32(p.scale)
        .map_err(|e| err(format!("hc_pre download scale: {e:?}")))?;
    let base = gpu
        .download_f32(p.base)
        .map_err(|e| err(format!("hc_pre download base: {e:?}")))?;
    let (pre_h, post_h, comb_logits) =
        split_pre_post_comb_logits(&mixes, &scale, &base, rows, hc_mult, hc_eps)?;

    // 3. Upload post (finished) and comb logits; run GPU sinkhorn in-place.
    gpu.hip
        .memcpy_htod(
            &post.buf,
            unsafe {
                std::slice::from_raw_parts(post_h.as_ptr() as *const u8, post_h.len() * 4)
            },
        )
        .map_err(|e| err(format!("hc_pre upload post: {e:?}")))?;
    gpu.hip
        .memcpy_htod(
            &comb.buf,
            unsafe {
                std::slice::from_raw_parts(
                    comb_logits.as_ptr() as *const u8,
                    comb_logits.len() * 4,
                )
            },
        )
        .map_err(|e| err(format!("hc_pre upload comb: {e:?}")))?;
    gpu.hc_sinkhorn_4x4_batched(comb, hc_eps, sinkhorn_iters, rows as i32)
        .map_err(|e| err(format!("hc_sinkhorn_4x4_batched: {e:?}")))?;

    // 4. y = sum_h pre[h] * x[h, :] via batched input map.
    let pre_t = gpu
        .upload_f32(&pre_h, &[rows, hc_mult])
        .map_err(|e| err(format!("hc_pre upload pre: {e:?}")))?;
    let map = gpu.hc_input_map_4stream_batched(&pre_t, x, y, dim as i32, rows as i32);
    free_scratch(gpu, pre_t);
    map.map_err(|e| err(format!("hc_input_map_4stream_batched: {e:?}")))?;
    Ok(())
}

/// `Block.hc_post` (`model.py:690-693`).
///
/// `x` `[rows, dim]`, residual `[rows, hc, dim]`, post `[rows, hc]`,
/// comb `[rows, hc, hc]` → `out` `[rows, hc, dim]`.
///
/// Uses `hc_mix_4stream_batched`:
/// `out[b,s,d] = sum_t comb[b,s,t] * residual[b,t,d] + post[b,s] * x[b,d]`.
pub fn parent_hc_post(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    x: &GpuTensor,
    residual: &GpuTensor,
    post: &GpuTensor,
    comb: &GpuTensor,
    rows: usize,
    hc_mult: usize,
    dim: usize,
    out: &GpuTensor,
) -> Result<(), String> {
    backend.ensure_device(gpu)?;
    if rows == 0 || hc_mult == 0 || dim == 0 {
        return Err(err("hc_post: rows, hc_mult, dim must be > 0"));
    }
    if hc_mult != 4 {
        return Err(err(format!(
            "hc_post: only hc_mult=4 is supported by the GPU kernels (got {hc_mult})"
        )));
    }
    require_f32(x, "hc_post x")?;
    require_f32(residual, "hc_post residual")?;
    require_f32(post, "hc_post post")?;
    require_f32(comb, "hc_post comb")?;
    require_f32(out, "hc_post out")?;
    require_elems(x, rows * dim, "hc_post x")?;
    require_elems(residual, rows * hc_mult * dim, "hc_post residual")?;
    require_elems(post, rows * hc_mult, "hc_post post")?;
    require_elems(comb, rows * hc_mult * hc_mult, "hc_post comb")?;
    require_elems(out, rows * hc_mult * dim, "hc_post out")?;

    gpu.hc_mix_4stream_batched(
        residual,
        comb,
        post,
        x,
        out,
        dim as i32,
        rows as i32,
    )
    .map_err(|e| err(format!("hc_mix_4stream_batched: {e:?}")))
}

/// `Block.hc_head` — plain sigmoid path, **no** sinkhorn. Output head only.
///
/// `p.fn_mat` is `[hc_mult, hc_mult*dim]`, `p.base` `[hc_mult]`,
/// `p.scale` length ≥ 1 (only element 0 is used).
pub fn parent_hc_head(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    x: &GpuTensor,
    p: ParentHcParams<'_>,
    rows: usize,
    hc_mult: usize,
    dim: usize,
    norm_eps: f32,
    hc_eps: f32,
    y: &GpuTensor,
) -> Result<(), String> {
    backend.ensure_device(gpu)?;
    if rows == 0 || hc_mult == 0 || dim == 0 {
        return Err(err("hc_head: rows, hc_mult, dim must be > 0"));
    }
    if hc_mult != 4 {
        return Err(err(format!(
            "hc_head: only hc_mult=4 is supported by the GPU kernels (got {hc_mult})"
        )));
    }
    let hc_dim = hc_mult
        .checked_mul(dim)
        .ok_or_else(|| err("hc_head: hc_dim overflow"))?;

    require_f32(x, "hc_head x")?;
    require_f32(p.fn_mat, "hc_head fn_mat")?;
    require_f32(p.base, "hc_head base")?;
    require_f32(p.scale, "hc_head scale")?;
    require_f32(y, "hc_head y")?;
    require_elems(x, rows * hc_dim, "hc_head x")?;
    require_elems(p.fn_mat, hc_mult * hc_dim, "hc_head fn_mat")?;
    require_elems(p.base, hc_mult, "hc_head base")?;
    require_elems(p.scale, 1, "hc_head scale")?;
    require_elems(y, rows * dim, "hc_head y")?;

    // mixes = F.linear(x_flat, hc_head_fn)
    let mixes_t = gpu
        .alloc_tensor(&[rows, hc_mult], DType::F32)
        .map_err(|e| err(format!("hc_head mixes alloc: {e:?}")))?;
    if let Err(e) = parent_f32_linear(gpu, p.fn_mat, x, &mixes_t, rows, hc_mult, hc_dim) {
        free_scratch(gpu, mixes_t);
        return Err(e);
    }

    let x_host = match gpu.download_f32(x) {
        Ok(v) => v,
        Err(e) => {
            free_scratch(gpu, mixes_t);
            return Err(err(format!("hc_head download x: {e:?}")));
        }
    };
    let mut mixes = match gpu.download_f32(&mixes_t) {
        Ok(v) => v,
        Err(e) => {
            free_scratch(gpu, mixes_t);
            return Err(err(format!("hc_head download mixes: {e:?}")));
        }
    };
    free_scratch(gpu, mixes_t);
    apply_flat_rsqrt_to_mixes(&x_host, &mut mixes, rows, hc_dim, hc_mult, norm_eps);

    let scale_v = gpu
        .download_f32(p.scale)
        .map_err(|e| err(format!("hc_head download scale: {e:?}")))?;
    let base = gpu
        .download_f32(p.base)
        .map_err(|e| err(format!("hc_head download base: {e:?}")))?;
    let scale = scale_v[0];

    // pre = sigmoid(mixes * scale + base) + hc_eps   — NO sinkhorn
    let mut pre = vec![0.0f32; rows * hc_mult];
    for r in 0..rows {
        for h in 0..hc_mult {
            let m = mixes[r * hc_mult + h];
            pre[r * hc_mult + h] = sigmoid_f32(m * scale + base[h]) + hc_eps;
        }
    }

    let pre_t = gpu
        .upload_f32(&pre, &[rows, hc_mult])
        .map_err(|e| err(format!("hc_head upload pre: {e:?}")))?;
    let map = gpu.hc_input_map_4stream_batched(&pre_t, x, y, dim as i32, rows as i32);
    free_scratch(gpu, pre_t);
    map.map_err(|e| err(format!("hc_head input_map: {e:?}")))?;
    Ok(())
}

/// RMSNorm with a BF16 weight tensor (parent norms are BF16 on disk).
///
/// The reference keeps weights BF16 and widens at use (`model.py` RMSNorm).
/// We widen explicitly here into a transient F32 buffer for `rmsnorm_batched`
/// (which is F32×F32); the resident BF16 weight is not mutated or replaced.
pub fn parent_rms_norm(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    x: &GpuTensor,
    weight: &GpuTensor,
    out: &GpuTensor,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<(), String> {
    backend.ensure_device(gpu)?;
    if rows == 0 || dim == 0 {
        return Err(err("rms_norm: rows and dim must be > 0"));
    }
    require_f32(x, "rms_norm x")?;
    require_f32(out, "rms_norm out")?;
    require_elems(x, rows * dim, "rms_norm x")?;
    require_elems(out, rows * dim, "rms_norm out")?;
    if weight.numel() < dim {
        return Err(err(format!(
            "rms_norm weight too short (have {} need {dim})",
            weight.numel()
        )));
    }

    // Widen BF16 → F32 at the call (high-16-bit shift; lossless).
    let w_f32 = match weight.dtype {
        DType::F32 => {
            // Already F32 (tests / synthetic). Use a shallow view via download
            // path that just re-uploads — keep a private copy so the caller's
            // tensor lifetime is undisturbed.
            let host = gpu
                .download_f32(weight)
                .map_err(|e| err(format!("rms_norm download f32 weight: {e:?}")))?;
            gpu.upload_f32(&host[..dim], &[dim])
                .map_err(|e| err(format!("rms_norm upload f32 weight: {e:?}")))?
        }
        DType::BF16 | DType::Raw => {
            let nbytes = dim * 2;
            let mut bytes = vec![0u8; nbytes];
            gpu.hip
                .memcpy_dtoh(&mut bytes, &weight.buf)
                .map_err(|e| err(format!("rms_norm download bf16 weight: {e:?}")))?;
            let mut host = vec![0.0f32; dim];
            for i in 0..dim {
                let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
                host[i] = f32::from_bits((bits as u32) << 16);
            }
            gpu.upload_f32(&host, &[dim])
                .map_err(|e| err(format!("rms_norm upload widened weight: {e:?}")))?
        }
        other => {
            return Err(err(format!(
                "rms_norm weight must be BF16 or F32 (got {other:?})"
            )));
        }
    };

    let result = gpu
        .rmsnorm_batched(x, &w_f32, out, rows, dim, eps)
        .map_err(|e| err(format!("rmsnorm_batched: {e:?}")));
    free_scratch(gpu, w_f32);
    result
}
