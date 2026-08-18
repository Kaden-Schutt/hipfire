// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU execution path for parent-checkpoint dense and expert linears.
//!
//! # Why BF16 decode + BF16 GEMM equals the reference FP8×FP8 / FP4 path
//!
//! The bundled reference (`inference/model.py::linear` + `kernel.py::{fp8,fp4}_gemm`)
//! does dynamic FP8 act-quant (block 128, UE8M0 scales) and then an FP8×FP8
//! (dense) or FP8×FP4 (expert) MFMA with per-block scale products applied after
//! the unscaled block product.
//!
//! We do **not** reimplement those FP8/FP4 GEMMs. The parent path is:
//!
//! 1. Decode weights to BF16 (dense at load → resident; experts on demand into
//!    caller scratch) via the Gate-2-verified dequant kernels.
//! 2. Simulate the activation quant **in place** on the BF16 activation tile
//!    (`act_quant_fp8_ue8m0_inplace_gfx942`, block 128) — same Gate-2 kernel.
//! 3. BF16 × BF16 → FP32 MFMA GEMM (`gemm_bf16_mfma_gfx942`).
//!
//! This is the same mathematical function because **every scale in play is a
//! power of two** (`scale_fmt = "ue8m0"` for both activations and weights). An
//! E4M3 code times a power of two has a 3-bit mantissa; an E2M1 code times a
//! power of two has a 1-bit mantissa. Both are **exactly** representable in
//! BF16 (7-bit mantissa, wide exponent). Each product term is therefore the
//! identical real number in both formulations, and exact in FP32 (≤ 8
//! significand bits). The *only* residual difference is FP32 summation order
//! and intermediate rounding — measured by Gate 3 against the reference-order
//! CPU oracle (`err_gpu <= 4 * err_ref`).
//!
//! If a case appears where this fails to hold (overflow, underflow to
//! subnormal, a NaN scale), that is a FINDING — do not paper over it.

use crate::Ds4ParentBackend;
use rdna_compute::{DType, Gpu, GpuTensor};

/// A dense parent projection, decoded to resident BF16 at construction.
///
/// Holds only the expanded BF16 tensor. Staging E4M3 codes and UE8M0 scales
/// are released inside [`ParentDenseWeight::decode_resident`] so VRAM matches
/// the Gate 1 projection (2× stored F8_E4M3 weight bytes).
pub struct ParentDenseWeight {
    tensor: GpuTensor,
    n: usize,
    k: usize,
}

impl ParentDenseWeight {
    /// Uploads E4M3 codes + UE8M0 scales, decodes to BF16 on device, and
    /// releases the code/scale staging buffers.
    ///
    /// `codes` is row-major `F8_E4M3 [n, k]`; `scales` is row-major
    /// `F8_E8M0 [ceil(n/128), ceil(k/128)]`.
    pub fn decode_resident(
        gpu: &mut Gpu,
        backend: Ds4ParentBackend,
        codes: &[u8],
        scales: &[u8],
        n: usize,
        k: usize,
    ) -> Result<Self, String> {
        backend.ensure_device(gpu)?;
        if n == 0 || k == 0 {
            return Err(format!(
                "deepseek4 parent: dense decode requires positive n,k (got n={n} k={k})"
            ));
        }
        let need_codes = n
            .checked_mul(k)
            .ok_or_else(|| "deepseek4 parent: dense codes size overflow".to_owned())?;
        let s_rows = n.div_ceil(128);
        let s_cols = k.div_ceil(128);
        let need_scales = s_rows
            .checked_mul(s_cols)
            .ok_or_else(|| "deepseek4 parent: dense scale size overflow".to_owned())?;
        if codes.len() < need_codes {
            return Err(format!(
                "deepseek4 parent: dense codes too short (have {} need {need_codes} for [{n},{k}])",
                codes.len()
            ));
        }
        if scales.len() < need_scales {
            return Err(format!(
                "deepseek4 parent: dense scales too short (have {} need {need_scales} for scale [{s_rows},{s_cols}])",
                scales.len()
            ));
        }

        let codes_t = gpu
            .upload_raw(&codes[..need_codes], &[n, k])
            .map_err(|e| format!("deepseek4 parent: dense codes upload: {e:?}"))?;
        let scales_t = match gpu.upload_raw(&scales[..need_scales], &[s_rows, s_cols]) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(codes_t);
                return Err(format!("deepseek4 parent: dense scales upload: {e:?}"));
            }
        };
        let out = match gpu.alloc_tensor(&[n, k], DType::BF16) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(codes_t);
                let _ = gpu.free_tensor(scales_t);
                return Err(format!("deepseek4 parent: dense BF16 alloc: {e:?}"));
            }
        };

        let dequant = gpu.dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942(
            &codes_t.buf,
            &scales_t.buf,
            &out.buf,
            n,
            k,
        );
        // Staging must not remain resident regardless of dequant outcome.
        let _ = gpu.free_tensor(codes_t);
        let _ = gpu.free_tensor(scales_t);
        dequant.map_err(|e| format!("deepseek4 parent: dense FP8→BF16 dequant: {e:?}"))?;

        Ok(Self {
            tensor: out,
            n,
            k,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn tensor(&self) -> &GpuTensor {
        &self.tensor
    }

    /// Bytes occupied by the resident BF16 weight tensor (Gate 1: 2× stored).
    pub fn resident_bytes(&self) -> usize {
        self.tensor.buf.size()
    }
}

/// A routed expert, left COMPRESSED in HBM. Decoded per use into scratch.
///
/// Do **not** expand all 256 experts at load — the residency projection depends
/// on codes + scales staying at their stored footprint.
pub struct ParentExpertWeight {
    codes: GpuTensor,
    scales: GpuTensor,
    n: usize,
    k: usize,
}

impl ParentExpertWeight {
    /// Upload packed E2M1 codes + per-row UE8M0 scales; leave compressed.
    ///
    /// `codes` is `I8 [n, k/2]` (two E2M1 nibbles per byte along K);
    /// `scales` is `F8_E8M0 [n, k/32]`. Logical weight shape is `[n, k]`.
    pub fn upload_compressed(
        gpu: &mut Gpu,
        backend: Ds4ParentBackend,
        codes: &[u8],
        scales: &[u8],
        n: usize,
        k: usize,
    ) -> Result<Self, String> {
        backend.ensure_device(gpu)?;
        if n == 0 || k == 0 {
            return Err(format!(
                "deepseek4 parent: expert upload requires positive n,k (got n={n} k={k})"
            ));
        }
        if k % 32 != 0 {
            return Err(format!(
                "deepseek4 parent: expert K must be a multiple of 32 (got k={k})"
            ));
        }
        let need_codes = n
            .checked_mul(k / 2)
            .ok_or_else(|| "deepseek4 parent: expert codes size overflow".to_owned())?;
        let n_groups = k / 32;
        let need_scales = n
            .checked_mul(n_groups)
            .ok_or_else(|| "deepseek4 parent: expert scale size overflow".to_owned())?;
        if codes.len() < need_codes {
            return Err(format!(
                "deepseek4 parent: expert codes too short (have {} need {need_codes} for packed [{n},{}])",
                codes.len(),
                k / 2
            ));
        }
        if scales.len() < need_scales {
            return Err(format!(
                "deepseek4 parent: expert scales too short (have {} need {need_scales} for [{n},{n_groups}])",
                scales.len()
            ));
        }

        let codes_t = gpu
            .upload_raw(&codes[..need_codes], &[n, k / 2])
            .map_err(|e| format!("deepseek4 parent: expert codes upload: {e:?}"))?;
        let scales_t = match gpu.upload_raw(&scales[..need_scales], &[n, n_groups]) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(codes_t);
                return Err(format!("deepseek4 parent: expert scales upload: {e:?}"));
            }
        };

        Ok(Self {
            codes: codes_t,
            scales: scales_t,
            n,
            k,
        })
    }

    /// Decode into a caller-owned BF16 scratch tensor of ≥ `n*k` elements.
    ///
    /// Scratch must be `DType::BF16` with `buf.size() >= n*k*2`. The compressed
    /// codes/scales stay resident; only the scratch is written.
    pub fn decode_into(&self, gpu: &mut Gpu, scratch: &GpuTensor) -> Result<(), String> {
        if gpu.try_gfx942().is_none() {
            return Err(
                "deepseek4 parent: expert decode requires gfx942 (no portable fallback)".to_owned(),
            );
        }
        let need = self
            .n
            .checked_mul(self.k)
            .and_then(|e| e.checked_mul(2))
            .ok_or_else(|| "deepseek4 parent: expert scratch size overflow".to_owned())?;
        if scratch.dtype != DType::BF16 {
            return Err(format!(
                "deepseek4 parent: expert scratch must be BF16 (got {:?})",
                scratch.dtype
            ));
        }
        if scratch.buf.size() < need {
            return Err(format!(
                "deepseek4 parent: expert scratch too small (have {} need {need} for BF16 [{},{}])",
                scratch.buf.size(),
                self.n,
                self.k
            ));
        }
        gpu.dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942(
            &self.codes.buf,
            &self.scales.buf,
            &scratch.buf,
            self.n,
            self.k,
        )
        .map_err(|e| format!("deepseek4 parent: expert FP4→BF16 dequant: {e:?}"))
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// Bytes occupied by compressed codes + scales (Gate 1: 1× stored).
    pub fn compressed_bytes(&self) -> usize {
        self.codes.buf.size() + self.scales.buf.size()
    }
}

/// Dense parent linear: act-quant simulation then BF16 GEMM.
///
/// # `x_bf16` is destroyed
///
/// `x_bf16` is `[m, k]` BF16 and is **mutated in place** by
/// `act_quant_fp8_ue8m0_inplace_gfx942(..., block = 128)`, matching the
/// reference's `inplace=True` at the linear boundary
/// (`inference/model.py::linear` + `act_quant(..., inplace=True)`). After this
/// call the buffer holds the simulated post-quant BF16 values
/// `bf16(e4m3(x/s) * s)`, **not** the original activations.
///
/// A caller that reuses the same `x_bf16` for a second projection (e.g. gate
/// then up, or `w1` then `w3` on a shared residual tile) would **double-
/// quantize**. The DS4 forward must either:
/// - keep a pristine BF16 copy of the residual and re-quant from it at each
///   linear, or
/// - quantize once into a dedicated act buffer and feed that to every linear
///   that shares the pre-quant activation (the reference quantizes inside
///   each `linear()` call independently — so the pristine-copy model is the
///   faithful one).
///
/// **Conclusion:** a non-destructive variant is **not** required for
/// correctness against the reference, because the reference also quantizes
/// inside each `linear()` from the caller-supplied `x` (and the Python tensor
/// is only mutated when the caller passed the same storage). What the forward
/// *does* need is discipline at the call site: never pass a residual stream
/// buffer that must survive the linear. Prefer an explicit act-scratch that is
/// filled from the residual (copy or cast) immediately before
/// `parent_linear_*`. Adding a non-destructive API that allocates an internal
/// quant buffer would hide that contract and encourage residual aliasing; we
/// keep the destructive signature and document it loudly instead.
///
/// `out` is `[m, n]` FP32 and is overwritten with `x_q @ W^T`.
pub fn parent_linear_dense(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    w: &ParentDenseWeight,
    x_bf16: &GpuTensor,
    m: usize,
    out: &GpuTensor,
) -> Result<(), String> {
    backend.ensure_device(gpu)?;
    validate_linear_args("dense", w.n, w.k, m, x_bf16, out, &w.tensor)?;

    // Destructive act-quant simulation — see doc comment above.
    gpu.act_quant_fp8_ue8m0_inplace_gfx942(&x_bf16.buf, m, w.k, 128)
        .map_err(|e| format!("deepseek4 parent: dense act-quant: {e:?}"))?;

    gpu.gemm_bf16_mfma_gfx942(&w.tensor.buf, &x_bf16.buf, &out.buf, w.n, w.k, m)
        .map_err(|e| format!("deepseek4 parent: dense BF16 GEMM: {e:?}"))
}

/// Expert parent linear against an already-decoded BF16 weight scratch.
///
/// Same destructive `x_bf16` contract as [`parent_linear_dense`]. `w_bf16` is
/// the BF16 `[n, k]` tensor produced by [`ParentExpertWeight::decode_into`].
pub fn parent_linear_expert(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    w_bf16: &GpuTensor,
    n: usize,
    k: usize,
    x_bf16: &GpuTensor,
    m: usize,
    out: &GpuTensor,
) -> Result<(), String> {
    backend.ensure_device(gpu)?;
    validate_linear_args("expert", n, k, m, x_bf16, out, w_bf16)?;

    gpu.act_quant_fp8_ue8m0_inplace_gfx942(&x_bf16.buf, m, k, 128)
        .map_err(|e| format!("deepseek4 parent: expert act-quant: {e:?}"))?;

    gpu.gemm_bf16_mfma_gfx942(&w_bf16.buf, &x_bf16.buf, &out.buf, n, k, m)
        .map_err(|e| format!("deepseek4 parent: expert BF16 GEMM: {e:?}"))
}

fn validate_linear_args(
    which: &str,
    n: usize,
    k: usize,
    m: usize,
    x_bf16: &GpuTensor,
    out: &GpuTensor,
    w_bf16: &GpuTensor,
) -> Result<(), String> {
    if m == 0 || n == 0 || k == 0 {
        return Err(format!(
            "deepseek4 parent: {which} linear requires positive m,n,k (got m={m} n={n} k={k})"
        ));
    }
    if k % 128 != 0 {
        return Err(format!(
            "deepseek4 parent: {which} linear K must be a multiple of act-quant block 128 (got k={k})"
        ));
    }
    if x_bf16.dtype != DType::BF16 {
        return Err(format!(
            "deepseek4 parent: {which} x_bf16 must be BF16 (got {:?})",
            x_bf16.dtype
        ));
    }
    if w_bf16.dtype != DType::BF16 {
        return Err(format!(
            "deepseek4 parent: {which} weight must be BF16 (got {:?})",
            w_bf16.dtype
        ));
    }
    if out.dtype != DType::F32 {
        return Err(format!(
            "deepseek4 parent: {which} out must be F32 (got {:?})",
            out.dtype
        ));
    }
    let x_need = m
        .checked_mul(k)
        .and_then(|e| e.checked_mul(2))
        .ok_or_else(|| format!("deepseek4 parent: {which} x size overflow"))?;
    let w_need = n
        .checked_mul(k)
        .and_then(|e| e.checked_mul(2))
        .ok_or_else(|| format!("deepseek4 parent: {which} w size overflow"))?;
    let out_need = m
        .checked_mul(n)
        .and_then(|e| e.checked_mul(4))
        .ok_or_else(|| format!("deepseek4 parent: {which} out size overflow"))?;
    if x_bf16.buf.size() < x_need {
        return Err(format!(
            "deepseek4 parent: {which} x_bf16 too small (have {} need {x_need} for [{m},{k}] BF16)",
            x_bf16.buf.size()
        ));
    }
    if w_bf16.buf.size() < w_need {
        return Err(format!(
            "deepseek4 parent: {which} weight too small (have {} need {w_need} for [{n},{k}] BF16)",
            w_bf16.buf.size()
        ));
    }
    if out.buf.size() < out_need {
        return Err(format!(
            "deepseek4 parent: {which} out too small (have {} need {out_need} for [{m},{n}] F32)",
            out.buf.size()
        ));
    }
    Ok(())
}
