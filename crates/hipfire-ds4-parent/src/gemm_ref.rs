// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gate 3 CPU oracle for the parent checkpoint's quantized GEMMs.
//!
//! The bundled tilelang kernels (`fp8_gemm` / `fp4_gemm`) cannot run on
//! `mi300x` (no torch/tilelang), so this module is the only executable
//! statement of their operator semantics. Two accumulation modes are
//! provided so Gate 3 can build a relative acceptance criterion from the
//! reference's own FP32 rounding error rather than an invented tolerance:
//!
//! - [`AccumMode::Exact`] — f64 throughout, no intermediate rounding (ground truth).
//! - [`AccumMode::ReferenceOrder`] — mirrors the tilelang block structure in f32
//!   (per-block unscaled product, then scale, then accumulate).
//!
//! Authority:
//! - `local://ds4-parent-gate3-contract.md` §2 / §5
//! - `.codeinsight+research/ds4-parent-ref/inference/kernel.py:204-254` (`fp8_gemm`)
//! - `.codeinsight+research/ds4-parent-ref/inference/kernel.py:442-515` (`fp4_gemm`)
//! - `.codeinsight+research/ds4-parent-ref/inference/model.py:114-126` (`linear`)
//!
//! ## Why the GPU BF16 path is the same mathematical function
//!
//! Every scale in play is a power of two (UE8M0). An E4M3 code times a power
//! of two has a 3-bit mantissa; an E2M1 code times a power of two has a 1-bit
//! mantissa — both exactly representable in BF16. Each product term is
//! therefore the identical real number in the reference (scaled FP8×FP8) and
//! GPU (BF16×BF16) formulations, and exact in FP32 (≤ 8 significand bits).
//! The only difference is FP32 summation order and intermediate rounding,
//! which is exactly what the two [`AccumMode`]s quantify.

use super::codec::{
    e2m1_to_f32, e4m3_to_f32, f32_to_e4m3, fast_round_scale, round_to_bf16, ue8m0_to_f32,
    FP8_E4M3_MAX,
};

/// Accumulation mode for the GEMM oracles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccumMode {
    /// f64 accumulation, no intermediate rounding. Ground truth for Gate 3.
    Exact,
    /// f32, reproducing the tilelang block structure of `fp8_gemm` / `fp4_gemm`.
    ReferenceOrder,
}

#[inline]
fn err_msg(msg: &str) -> String {
    format!("deepseek4 parent: {msg}")
}

/// Encode a positive power-of-two f32 as a UE8M0 byte.
///
/// `float8_e8m0fnu` stores exponent `b` meaning `2^(b - 127)`. Byte `0xFF` is
/// NaN and is never produced here. If `s` is not exactly a representable
/// UE8M0 value, that is a defect in the scale pipeline — we fail closed
/// rather than silently rounding.
#[inline]
fn f32_to_ue8m0_exact(s: f32) -> Result<u8, String> {
    if !s.is_finite() || s <= 0.0 {
        return Err(err_msg(&format!(
            "act_quant_fp8_codes: scale {s} is not a positive finite power of two"
        )));
    }
    // Subnormal path: only 2^-127 maps to byte 0.
    if s == 2.0f32.powi(-127) {
        return Ok(0);
    }
    let bits = s.to_bits();
    let exp = (bits >> 23) & 0xff;
    let mant = bits & 0x7f_ffff;
    if mant != 0 || exp == 0 || exp == 0xff {
        return Err(err_msg(&format!(
            "act_quant_fp8_codes: scale {s} (bits={bits:#010x}) is not exactly representable as UE8M0"
        )));
    }
    // Normal power of two: value = 2^(exp - 127), so the UE8M0 byte is `exp`.
    // exp ∈ 1..=254 (255 would be the NaN encoding).
    Ok(exp as u8)
}

/// Quantize activations exactly as `act_quant(x, block, "ue8m0", e8m0)` does,
/// returning the E4M3 codes and the UE8M0 scale **bytes** — the operands the
/// reference GEMMs actually consume.
///
/// `x` is interpreted as a row-major `[rows, last_dim]` matrix with
/// `rows = x.len() / last_dim`. `last_dim` must be divisible by `block`.
///
/// # BF16 input domain
///
/// Matches [`crate::codec::act_quant_fp8_inplace_ref`]: every element is
/// rounded to BF16 before amax / scale / quant (`kernel.py` `in_dtype=BF16`).
/// Callers may pass full-precision f32; the rounding is applied here.
pub fn act_quant_fp8_codes(
    x: &[f32],
    last_dim: usize,
    block: usize,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    if last_dim == 0 {
        return Err(err_msg("act_quant_fp8_codes: last_dim must be > 0"));
    }
    if block == 0 {
        return Err(err_msg("act_quant_fp8_codes: block must be > 0"));
    }
    if last_dim % block != 0 {
        return Err(err_msg(&format!(
            "act_quant_fp8_codes: last_dim {last_dim} not divisible by block {block}"
        )));
    }
    if x.len() % last_dim != 0 {
        return Err(err_msg(&format!(
            "act_quant_fp8_codes: buffer len {} not divisible by last_dim {last_dim}",
            x.len()
        )));
    }

    let rows = x.len() / last_dim;
    let groups = last_dim / block;
    let mut codes = vec![0u8; x.len()];
    let mut scales = vec![0u8; rows * groups];
    let max_inv = 1.0 / FP8_E4M3_MAX;

    // Materialise the BF16-domain view once (kernel.py in_dtype=BF16).
    let x_bf16: Vec<f32> = x.iter().copied().map(round_to_bf16).collect();

    for r in 0..rows {
        for g in 0..groups {
            let base = r * last_dim + g * block;
            let group = &x_bf16[base..base + block];
            let mut amax = 0.0f32;
            for &v in group {
                amax = amax.max(v.abs());
            }
            // FP8 path clamps amax to max(amax, 1e-4) BEFORE rounding the scale.
            amax = amax.max(1e-4);
            let s = fast_round_scale(amax, max_inv);
            let s_byte = f32_to_ue8m0_exact(s)?;
            scales[r * groups + g] = s_byte;
            for i in 0..block {
                let clamped = (group[i] / s).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX);
                codes[base + i] = f32_to_e4m3(clamped);
            }
        }
    }
    Ok((codes, scales))
}

/// Dense tier. `a` FP8 codes `[m,k]`; `a_s` UE8M0 bytes `[m, k/128]`;
/// `b` FP8 codes `[n,k]`; `b_s` UE8M0 bytes `[ceil(n/128), ceil(k/128)]`.
///
/// Computes `C[m,n] = A[m,k] @ B[n,k]^T` with the per-128 block scaling of
/// `kernel.py:204-254`.
pub fn fp8_gemm_ref(
    a: &[u8],
    a_s: &[u8],
    b: &[u8],
    b_s: &[u8],
    m: usize,
    n: usize,
    k: usize,
    mode: AccumMode,
) -> Result<Vec<f32>, String> {
    const GROUP: usize = 128;

    if k % GROUP != 0 {
        return Err(err_msg(&format!(
            "fp8_gemm_ref: k {k} not divisible by {GROUP}"
        )));
    }
    if m == 0 || n == 0 || k == 0 {
        return Err(err_msg("fp8_gemm_ref: m, n, k must be > 0"));
    }

    let k_blocks = k / GROUP;
    let n_scale_rows = n.div_ceil(GROUP);

    if a.len() != m * k {
        return Err(err_msg(&format!(
            "fp8_gemm_ref: a len {} != m*k {}",
            a.len(),
            m * k
        )));
    }
    if b.len() != n * k {
        return Err(err_msg(&format!(
            "fp8_gemm_ref: b len {} != n*k {}",
            b.len(),
            n * k
        )));
    }
    if a_s.len() != m * k_blocks {
        return Err(err_msg(&format!(
            "fp8_gemm_ref: a_s len {} != m*(k/128) {}",
            a_s.len(),
            m * k_blocks
        )));
    }
    if b_s.len() != n_scale_rows * k_blocks {
        return Err(err_msg(&format!(
            "fp8_gemm_ref: b_s len {} != ceil(n/128)*(k/128) {}",
            b_s.len(),
            n_scale_rows * k_blocks
        )));
    }

    let mut out = vec![0.0f32; m * n];

    match mode {
        AccumMode::Exact => {
            // f64 throughout: each term is sa * sb * a * b with no f32 rounding.
            for mi in 0..m {
                for ni in 0..n {
                    let mut acc = 0.0f64;
                    let n_sb = ni / GROUP;
                    for kb in 0..k_blocks {
                        let sa = ue8m0_to_f32(a_s[mi * k_blocks + kb]) as f64;
                        let sb = ue8m0_to_f32(b_s[n_sb * k_blocks + kb]) as f64;
                        let k0 = kb * GROUP;
                        for kk in 0..GROUP {
                            let av = e4m3_to_f32(a[mi * k + k0 + kk]) as f64;
                            let bv = e4m3_to_f32(b[ni * k + k0 + kk]) as f64;
                            acc += sa * sb * av * bv;
                        }
                    }
                    out[mi * n + ni] = acc as f32;
                }
            }
        }
        AccumMode::ReferenceOrder => {
            // Mirror kernel.py:243-250:
            //   C_local  = sum_{k in block} a*b          (f32, unscaled)
            //   Scale_C  = scale_a[m,kb] * scale_b[n/128,kb]
            //   C_accum += C_local * Scale_C             (separate f32 accumulator)
            for mi in 0..m {
                for ni in 0..n {
                    let mut c_accum = 0.0f32;
                    let n_sb = ni / GROUP;
                    for kb in 0..k_blocks {
                        let mut c_local = 0.0f32;
                        let k0 = kb * GROUP;
                        for kk in 0..GROUP {
                            let av = e4m3_to_f32(a[mi * k + k0 + kk]);
                            let bv = e4m3_to_f32(b[ni * k + k0 + kk]);
                            c_local += av * bv;
                        }
                        let sa = ue8m0_to_f32(a_s[mi * k_blocks + kb]);
                        let sb = ue8m0_to_f32(b_s[n_sb * k_blocks + kb]);
                        // Single product formed once, matching Scale_C_shared.
                        let scale_c = sa * sb;
                        c_accum += c_local * scale_c;
                    }
                    out[mi * n + ni] = c_accum;
                }
            }
        }
    }

    Ok(out)
}

/// Expert tier. `a` FP8 codes `[m,k]`; `a_s` UE8M0 `[m, k/128]`;
/// `b` packed E2M1 `[n, k/2]`; `b_s` UE8M0 `[n, k/32]`.
///
/// Computes `C[m,n] = A_fp8[m,k] @ B_fp4[n,k]^T` with per-32 weight scales and
/// per-128 act scales (`kernel.py:442-515`).
pub fn fp4_gemm_ref(
    a: &[u8],
    a_s: &[u8],
    b: &[u8],
    b_s: &[u8],
    m: usize,
    n: usize,
    k: usize,
    mode: AccumMode,
) -> Result<Vec<f32>, String> {
    const ACT_GROUP: usize = 128;
    const WGT_GROUP: usize = 32;
    // n_sub = act_group / block_K = 4 sub-blocks per act scale group.
    const N_SUB: usize = ACT_GROUP / WGT_GROUP;

    if k % 2 != 0 {
        return Err(err_msg(&format!(
            "fp4_gemm_ref: k {k} must be even (packed E2M1)"
        )));
    }
    if k % ACT_GROUP != 0 {
        return Err(err_msg(&format!(
            "fp4_gemm_ref: k {k} not divisible by {ACT_GROUP}"
        )));
    }
    // WGT_GROUP divides ACT_GROUP, so k % 32 == 0 follows from k % 128 == 0,
    // but keep the check explicit for clarity if constants ever diverge.
    if k % WGT_GROUP != 0 {
        return Err(err_msg(&format!(
            "fp4_gemm_ref: k {k} not divisible by {WGT_GROUP}"
        )));
    }
    if m == 0 || n == 0 || k == 0 {
        return Err(err_msg("fp4_gemm_ref: m, n, k must be > 0"));
    }

    let k_act_blocks = k / ACT_GROUP;
    let k_wgt_blocks = k / WGT_GROUP;
    let b_packed_row = k / 2;

    if a.len() != m * k {
        return Err(err_msg(&format!(
            "fp4_gemm_ref: a len {} != m*k {}",
            a.len(),
            m * k
        )));
    }
    if b.len() != n * b_packed_row {
        return Err(err_msg(&format!(
            "fp4_gemm_ref: b len {} != n*(k/2) {}",
            b.len(),
            n * b_packed_row
        )));
    }
    if a_s.len() != m * k_act_blocks {
        return Err(err_msg(&format!(
            "fp4_gemm_ref: a_s len {} != m*(k/128) {}",
            a_s.len(),
            m * k_act_blocks
        )));
    }
    if b_s.len() != n * k_wgt_blocks {
        return Err(err_msg(&format!(
            "fp4_gemm_ref: b_s len {} != n*(k/32) {}",
            b_s.len(),
            n * k_wgt_blocks
        )));
    }

    let mut out = vec![0.0f32; m * n];

    match mode {
        AccumMode::Exact => {
            for mi in 0..m {
                for ni in 0..n {
                    let mut acc = 0.0f64;
                    for kb in 0..k_wgt_blocks {
                        let sa = ue8m0_to_f32(a_s[mi * k_act_blocks + kb / N_SUB]) as f64;
                        let sb = ue8m0_to_f32(b_s[ni * k_wgt_blocks + kb]) as f64;
                        let k0 = kb * WGT_GROUP;
                        for kk in 0..WGT_GROUP {
                            let ki = k0 + kk;
                            let av = e4m3_to_f32(a[mi * k + ki]) as f64;
                            // Packed E2M1: low nibble = even k, high = odd k.
                            let byte = b[ni * b_packed_row + ki / 2];
                            let nibble = if ki & 1 == 0 {
                                byte & 0x0f
                            } else {
                                byte >> 4
                            };
                            let bv = e2m1_to_f32(nibble) as f64;
                            acc += sa * sb * av * bv;
                        }
                    }
                    out[mi * n + ni] = acc as f32;
                }
            }
        }
        AccumMode::ReferenceOrder => {
            // Mirror kernel.py:498-510:
            //   C_local  = sum_{k in 32} a * cast_fp8(b_fp4)   (f32, unscaled)
            //   C_accum += C_local * scale_a * scale_b         (left to right)
            // FP4→FP8→float is lossless for all 16 E2M1 codes (asserted in tests),
            // so decoding E2M1 straight to f32 is equivalent.
            for mi in 0..m {
                for ni in 0..n {
                    let mut c_accum = 0.0f32;
                    for kb in 0..k_wgt_blocks {
                        let mut c_local = 0.0f32;
                        let k0 = kb * WGT_GROUP;
                        for kk in 0..WGT_GROUP {
                            let ki = k0 + kk;
                            let av = e4m3_to_f32(a[mi * k + ki]);
                            let byte = b[ni * b_packed_row + ki / 2];
                            let nibble = if ki & 1 == 0 {
                                byte & 0x0f
                            } else {
                                byte >> 4
                            };
                            let bv = e2m1_to_f32(nibble);
                            c_local += av * bv;
                        }
                        let sa = ue8m0_to_f32(a_s[mi * k_act_blocks + kb / N_SUB]);
                        let sb = ue8m0_to_f32(b_s[ni * k_wgt_blocks + kb]);
                        // Left-to-right: (C_local * scale_a) * scale_b.
                        c_accum += c_local * sa * sb;
                    }
                    out[mi * n + ni] = c_accum;
                }
            }
        }
    }

    Ok(out)
}

/// `model.py::linear()` end-to-end for the dense FP8 weight tier:
/// dynamic FP8 act-quant of `x` (block 128, ue8m0) followed by `fp8_gemm`.
///
/// `x` is f32 holding BF16-representable values, shape `[m, k]`.
/// `b` / `b_s` are the weight codes and scales. Returns `[m, n]`.
pub fn linear_fp8_ref(
    x: &[f32],
    b: &[u8],
    b_s: &[u8],
    m: usize,
    n: usize,
    k: usize,
    mode: AccumMode,
) -> Result<Vec<f32>, String> {
    if x.len() != m * k {
        return Err(err_msg(&format!(
            "linear_fp8_ref: x len {} != m*k {}",
            x.len(),
            m * k
        )));
    }
    let (a, a_s) = act_quant_fp8_codes(x, k, 128)?;
    fp8_gemm_ref(&a, &a_s, b, b_s, m, n, k, mode)
}

/// `model.py::linear()` end-to-end for the routed-expert FP4 weight tier:
/// dynamic FP8 act-quant of `x` (block 128, ue8m0) followed by `fp4_gemm`.
pub fn linear_fp4_ref(
    x: &[f32],
    b: &[u8],
    b_s: &[u8],
    m: usize,
    n: usize,
    k: usize,
    mode: AccumMode,
) -> Result<Vec<f32>, String> {
    if x.len() != m * k {
        return Err(err_msg(&format!(
            "linear_fp4_ref: x len {} != m*k {}",
            x.len(),
            m * k
        )));
    }
    let (a, a_s) = act_quant_fp8_codes(x, k, 128)?;
    fp4_gemm_ref(&a, &a_s, b, b_s, m, n, k, mode)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{f32_to_e4m3, E2M1_LUT};

    /// UE8M0 byte for `2^e` (e.g. `ue8(0) == 0x7F` → 1.0).
    fn ue8(exp_offset: i32) -> u8 {
        // value = 2^exp_offset → byte = exp_offset + 127
        let b = exp_offset + 127;
        assert!((0..=254).contains(&b), "ue8 offset {exp_offset} out of range");
        b as u8
    }

    fn rel_err(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let d = (x as f64) - (y as f64);
            num += d * d;
            den += (y as f64) * (y as f64);
        }
        if den == 0.0 {
            return num.sqrt();
        }
        num.sqrt() / den.sqrt()
    }

    /// Hand-computed tiny case: m=1, n=1, k=128.
    ///
    /// Codes / scales chosen so the expected value is trivial arithmetic:
    ///
    /// ```text
    /// a[k] = 1.0  (E4M3 0x38) for all k
    /// b[k] = 0.5  (E4M3 0x30) for all k
    /// a_s  = 2.0  (UE8M0 0x80)
    /// b_s  = 4.0  (UE8M0 0x81)
    ///
    /// C_local = sum_k 1.0 * 0.5 = 64.0
    /// Scale_C = 2.0 * 4.0 = 8.0
    /// C       = 64.0 * 8.0 = 512.0
    /// ```
    #[test]
    fn fp8_hand_computed_1x1x128() {
        let m = 1usize;
        let n = 1usize;
        let k = 128usize;
        let a = vec![0x38u8; m * k]; // 1.0
        let b = vec![0x30u8; n * k]; // 0.5
        let a_s = vec![ue8(1)]; // 2.0
        let b_s = vec![ue8(2)]; // 4.0

        // Sanity on the codecs used in the derivation above.
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0x30), 0.5);
        assert_eq!(ue8m0_to_f32(ue8(1)), 2.0);
        assert_eq!(ue8m0_to_f32(ue8(2)), 4.0);

        for mode in [AccumMode::Exact, AccumMode::ReferenceOrder] {
            let c = fp8_gemm_ref(&a, &a_s, &b, &b_s, m, n, k, mode).unwrap();
            assert_eq!(c.len(), 1);
            assert_eq!(
                c[0], 512.0,
                "mode={mode:?}: expected hand-computed 512.0, got {}",
                c[0]
            );
        }
    }

    /// Hand-computed FP4 tiny case: m=1, n=1, k=128.
    ///
    /// ```text
    /// a[k]     = 1.0          (E4M3 0x38)
    /// b nibble = 2.0          (E2M1 code 4)  for every logical k
    /// a_s[0]   = 2.0          (one act group covering all 128)
    /// b_s[kb]  = 4.0          for each of the 4 weight groups
    ///
    /// Per weight group (32 terms): C_local = 32 * 1.0 * 2.0 = 64.0
    /// scaled += 64.0 * 2.0 * 4.0 = 512.0
    /// four groups → C = 2048.0
    /// ```
    #[test]
    fn fp4_hand_computed_1x1x128() {
        let m = 1usize;
        let n = 1usize;
        let k = 128usize;
        let a = vec![0x38u8; m * k]; // 1.0
        // Pack E2M1 code 4 (value 2.0) into both nibbles of every byte.
        let b = vec![0x44u8; n * (k / 2)];
        let a_s = vec![ue8(1)]; // 2.0 — one act group
        let b_s = vec![ue8(2); k / 32]; // 4.0 — four weight groups

        assert_eq!(e2m1_to_f32(4), 2.0);

        for mode in [AccumMode::Exact, AccumMode::ReferenceOrder] {
            let c = fp4_gemm_ref(&a, &a_s, &b, &b_s, m, n, k, mode).unwrap();
            assert_eq!(c.len(), 1);
            assert_eq!(
                c[0], 2048.0,
                "mode={mode:?}: expected hand-computed 2048.0, got {}",
                c[0]
            );
        }
    }

    /// `Exact` and `ReferenceOrder` agree to ~sqrt(K)*2^-24 relative, and are
    /// NOT bit-identical (otherwise ReferenceOrder would be vacuous).
    #[test]
    fn fp8_exact_vs_reference_order_differs_but_close() {
        let m = 2usize;
        let n = 128usize;
        let k = 4096usize;
        // Large-magnitude E4M3 codes with odd mantissas so the 128-term f32
        // block sum rounds (small codes stay exact in f32 and make the mode
        // comparison vacuous).
        let mut a = vec![0u8; m * k];
        let mut b = vec![0u8; n * k];
        for i in 0..a.len() {
            // ~{1.125, 1.25, ..., 448} range with sticky mantissas.
            let codes = [0x39u8, 0x4b, 0x5d, 0x6e, 0x78, 0x72, 0x6a, 0x55];
            a[i] = codes[i % codes.len()];
        }
        for i in 0..b.len() {
            let codes = [0x3au8, 0x4c, 0x5e, 0x70, 0x77, 0x61, 0x53, 0x48];
            b[i] = codes[i % codes.len()];
        }
        let k_blocks = k / 128;
        let mut a_s = vec![0u8; m * k_blocks];
        let mut b_s = vec![0u8; (n / 128) * k_blocks];
        for i in 0..a_s.len() {
            a_s[i] = ue8((i % 3) as i32 - 1); // 0.5, 1, 2
        }
        for i in 0..b_s.len() {
            b_s[i] = ue8(((i % 4) as i32) - 2); // 0.25, 0.5, 1, 2
        }

        let exact = fp8_gemm_ref(&a, &a_s, &b, &b_s, m, n, k, AccumMode::Exact).unwrap();
        let pref =
            fp8_gemm_ref(&a, &a_s, &b, &b_s, m, n, k, AccumMode::ReferenceOrder).unwrap();

        // Not bit-identical.
        let any_diff = exact.iter().zip(pref.iter()).any(|(e, r)| e.to_bits() != r.to_bits());
        assert!(
            any_diff,
            "ReferenceOrder is bit-identical to Exact — f32 block accum is not engaged"
        );

        let err = rel_err(&pref, &exact);
        // Expectation: roughly sqrt(K) * 2^-24 ≈ 64 * 5.96e-8 ≈ 3.8e-6.
        // Allow a generous band; the point is "same order", not a tight bound.
        let expect = (k as f64).sqrt() * (2.0f64).powi(-24);
        assert!(
            err > 0.0,
            "err_ref must be > 0 (got {err})"
        );
        assert!(
            err < 50.0 * expect,
            "err_ref={err} far exceeds ~sqrt(K)*2^-24={expect}"
        );
        // Also keep it below a hard relative ceiling that would indicate a real bug.
        assert!(err < 1e-4, "err_ref={err} suspiciously large");

        eprintln!(
            "fp8 k={k}: err_ref={err:.6e}, sqrt(K)*2^-24={expect:.6e}, ratio={:.3}",
            err / expect
        );
    }

    #[test]
    fn fp4_exact_vs_reference_order_differs_but_close() {
        let m = 2usize;
        let n = 4usize;
        let k = 4096usize;
        let mut a = vec![0u8; m * k];
        let mut b = vec![0u8; n * (k / 2)];
        for i in 0..a.len() {
            // Large E4M3 acts so 32-term f32 products with E2M1 round.
            let codes = [0x58u8, 0x5a, 0x6c, 0x70, 0x78, 0x74, 0x62, 0x4e];
            a[i] = codes[i % codes.len()];
        }
        for i in 0..b.len() {
            // Prefer large E2M1 magnitudes {3,4,6} in both nibbles.
            let mags = [5u8, 6, 7, 4, 7, 6, 5];
            let lo = mags[i % mags.len()];
            let hi = mags[(i / 3) % mags.len()];
            b[i] = lo | (hi << 4);
        }
        let k_act = k / 128;
        let k_wgt = k / 32;
        let mut a_s = vec![0u8; m * k_act];
        let mut b_s = vec![0u8; n * k_wgt];
        for i in 0..a_s.len() {
            a_s[i] = ue8((i % 3) as i32 - 1);
        }
        for i in 0..b_s.len() {
            b_s[i] = ue8(((i % 5) as i32) - 2);
        }

        let exact = fp4_gemm_ref(&a, &a_s, &b, &b_s, m, n, k, AccumMode::Exact).unwrap();
        let pref =
            fp4_gemm_ref(&a, &a_s, &b, &b_s, m, n, k, AccumMode::ReferenceOrder).unwrap();

        let any_diff = exact.iter().zip(pref.iter()).any(|(e, r)| e.to_bits() != r.to_bits());
        assert!(
            any_diff,
            "ReferenceOrder is bit-identical to Exact — f32 block accum is not engaged"
        );

        let err = rel_err(&pref, &exact);
        let expect = (k as f64).sqrt() * (2.0f64).powi(-24);
        assert!(err > 0.0, "err_ref must be > 0 (got {err})");
        assert!(
            err < 50.0 * expect,
            "err_ref={err} far exceeds ~sqrt(K)*2^-24={expect}"
        );
        assert!(err < 1e-4, "err_ref={err} suspiciously large");

        eprintln!(
            "fp4 k={k}: err_ref={err:.6e}, sqrt(K)*2^-24={expect:.6e}, ratio={:.3}",
            err / expect
        );
    }

    /// Perturb a single `scales_b` entry and assert only the matching N-block moves.
    #[test]
    fn fp8_scale_b_axis() {
        let m = 2usize;
        let n = 256usize; // two N-scale rows
        let k = 128usize; // one K-block
        let a = vec![0x38u8; m * k]; // 1.0
        let b = vec![0x38u8; n * k]; // 1.0
        let a_s = vec![ue8(0); m]; // 1.0
        let mut b_s = vec![ue8(0); 2]; // [n/128=2, k/128=1] all 1.0

        let base = fp8_gemm_ref(&a, &a_s, &b, &b_s, m, n, k, AccumMode::Exact).unwrap();
        // Every output = 128 * 1 * 1 * 1 = 128.
        for &v in &base {
            assert_eq!(v, 128.0);
        }

        // Bump scales_b[0, 0] from 1.0 → 2.0. Only n in 0..128 should double.
        b_s[0] = ue8(1);
        let pert = fp8_gemm_ref(&a, &a_s, &b, &b_s, m, n, k, AccumMode::Exact).unwrap();
        for mi in 0..m {
            for ni in 0..n {
                let v = pert[mi * n + ni];
                if ni < 128 {
                    assert_eq!(v, 256.0, "m={mi} n={ni}: expected 256 from scales_b[0]");
                } else {
                    assert_eq!(v, 128.0, "m={mi} n={ni}: expected unchanged 128");
                }
            }
        }
    }

    /// Perturb a single `scales_a` entry and assert only that row moves.
    #[test]
    fn fp8_scale_a_axis() {
        let m = 2usize;
        let n = 128usize;
        let k = 256usize; // two K-blocks
        let a = vec![0x38u8; m * k];
        let b = vec![0x38u8; n * k];
        let mut a_s = vec![ue8(0); m * 2]; // [m, 2]
        let b_s = vec![ue8(0); 1 * 2]; // [1, 2]

        let base = fp8_gemm_ref(&a, &a_s, &b, &b_s, m, n, k, AccumMode::Exact).unwrap();
        // Two K-blocks × 128 = 256 per output.
        for &v in &base {
            assert_eq!(v, 256.0);
        }

        // Bump a_s[0, 1] (row 0, second K-block) from 1 → 2.
        // Contribution of kb=1 doubles for m=0 only: 128 + 2*128 = 384.
        a_s[0 * 2 + 1] = ue8(1);
        let pert = fp8_gemm_ref(&a, &a_s, &b, &b_s, m, n, k, AccumMode::Exact).unwrap();
        for mi in 0..m {
            for ni in 0..n {
                let v = pert[mi * n + ni];
                if mi == 0 {
                    assert_eq!(v, 384.0, "m=0 n={ni}: expected 384");
                } else {
                    assert_eq!(v, 256.0, "m=1 n={ni}: expected unchanged 256");
                }
            }
        }
    }

    /// k=256 → 8 weight-scale groups and 2 act-scale groups, all distinct.
    #[test]
    fn fp4_mixed_scale_groups_k256() {
        let m = 1usize;
        let n = 1usize;
        let k = 256usize;
        let a = vec![0x38u8; m * k]; // 1.0
        // E2M1 = 1.0 (code 2) in every nibble.
        let b = vec![0x22u8; n * (k / 2)];
        assert_eq!(e2m1_to_f32(2), 1.0);

        // Act scales: two groups. sa0=1, sa1=2.
        let a_s = vec![ue8(0), ue8(1)];
        // Weight scales: 8 groups with 2^0 .. and a couple of distinct values.
        // b_s[kb] = 2^(kb % 3)  → 1, 2, 4, 1, 2, 4, 1, 2
        let b_s: Vec<u8> = (0..8).map(|kb| ue8((kb % 3) as i32)).collect();

        // Expected:
        // kb0: 32 * 1 * 1 * 1 = 32
        // kb1: 32 * 1 * 2     = 64
        // kb2: 32 * 1 * 4     = 128
        // kb3: 32 * 1 * 1     = 32     (still act group 0: kb//4 == 0)
        // kb4: 32 * 2 * 2     = 128    (act group 1: kb//4 == 1, sa=2)
        // kb5: 32 * 2 * 4     = 256
        // kb6: 32 * 2 * 1     = 64
        // kb7: 32 * 2 * 2     = 128
        // sum = 32+64+128+32+128+256+64+128 = 832
        let expected = 832.0f32;

        for mode in [AccumMode::Exact, AccumMode::ReferenceOrder] {
            let c = fp4_gemm_ref(&a, &a_s, &b, &b_s, m, n, k, mode).unwrap();
            assert_eq!(
                c[0], expected,
                "mode={mode:?}: expected {expected}, got {}",
                c[0]
            );
        }

        // Also verify that flipping only act scale group 1 moves only kb4..7.
        let mut a_s2 = a_s.clone();
        a_s2[1] = ue8(2); // 4.0 instead of 2.0
        // kb0..3 unchanged: 32+64+128+32 = 256
        // kb4: 32*4*2=256, kb5:32*4*4=512, kb6:32*4*1=128, kb7:32*4*2=256
        // sum = 256+256+512+128+256 = 1408
        let c2 = fp4_gemm_ref(&a, &a_s2, &b, &b_s, m, n, k, AccumMode::Exact).unwrap();
        assert_eq!(c2[0], 1408.0);
    }

    /// FP4→FP8→float equals direct E2M1 decode for all 16 codes.
    #[test]
    fn fp4_via_fp8_equals_direct_e2m1() {
        for code in 0u8..16 {
            let direct = e2m1_to_f32(code);
            // kernel.py:496 — Cast(FP8, Cast(FP32, fp4))
            let via_fp8 = e4m3_to_f32(f32_to_e4m3(direct));
            assert_eq!(
                direct.to_bits(),
                via_fp8.to_bits(),
                "code {code}: direct={direct} via_fp8={via_fp8} (LUT={})",
                E2M1_LUT[code as usize]
            );
        }
    }

    /// `linear_fp8_ref` is exactly act_quant_fp8_codes + fp8_gemm_ref.
    #[test]
    fn linear_fp8_composes_quant_then_gemm() {
        let m = 2usize;
        let n = 128usize;
        let k = 128usize;
        // BF16-representable activations.
        let mut x = vec![0.0f32; m * k];
        for i in 0..x.len() {
            x[i] = (i as f32) * 0.25 - 8.0;
        }
        let b = vec![0x38u8; n * k]; // 1.0 codes
        let b_s = vec![ue8(0)]; // 1.0

        let via_linear =
            linear_fp8_ref(&x, &b, &b_s, m, n, k, AccumMode::ReferenceOrder).unwrap();
        let (a, a_s) = act_quant_fp8_codes(&x, k, 128).unwrap();
        let via_parts =
            fp8_gemm_ref(&a, &a_s, &b, &b_s, m, n, k, AccumMode::ReferenceOrder).unwrap();
        assert_eq!(via_linear.len(), via_parts.len());
        for (i, (l, p)) in via_linear.iter().zip(via_parts.iter()).enumerate() {
            assert_eq!(
                l.to_bits(),
                p.to_bits(),
                "mismatch at {i}: linear={l} parts={p}"
            );
        }
    }

    /// `linear_fp4_ref` is exactly act_quant_fp8_codes + fp4_gemm_ref.
    #[test]
    fn linear_fp4_composes_quant_then_gemm() {
        let m = 2usize;
        let n = 3usize;
        let k = 128usize;
        let mut x = vec![0.0f32; m * k];
        for i in 0..x.len() {
            x[i] = (i as f32) * 0.125 - 4.0;
        }
        let b = vec![0x22u8; n * (k / 2)]; // E2M1 = 1.0
        let b_s = vec![ue8(0); n * (k / 32)];

        let via_linear =
            linear_fp4_ref(&x, &b, &b_s, m, n, k, AccumMode::ReferenceOrder).unwrap();
        let (a, a_s) = act_quant_fp8_codes(&x, k, 128).unwrap();
        let via_parts =
            fp4_gemm_ref(&a, &a_s, &b, &b_s, m, n, k, AccumMode::ReferenceOrder).unwrap();
        assert_eq!(via_linear.len(), via_parts.len());
        for (i, (l, p)) in via_linear.iter().zip(via_parts.iter()).enumerate() {
            assert_eq!(
                l.to_bits(),
                p.to_bits(),
                "mismatch at {i}: linear={l} parts={p}"
            );
        }
    }

    #[test]
    fn act_quant_fp8_codes_scale_is_ue8m0_byte() {
        // A single block whose amax is 1.0 → scale = fast_round_scale(1, 1/448)
        // = 2^ceil(log2(1/448)). Spot-check that the returned byte round-trips.
        let x = vec![1.0f32; 128];
        let (codes, scales) = act_quant_fp8_codes(&x, 128, 128).unwrap();
        assert_eq!(codes.len(), 128);
        assert_eq!(scales.len(), 1);
        let s = ue8m0_to_f32(scales[0]);
        // Scale must be a power of two and match fast_round_scale.
        let expect = fast_round_scale(1.0f32.max(1e-4), 1.0 / FP8_E4M3_MAX);
        assert_eq!(s, expect);
        // Codes: 1.0 / s quantized to e4m3.
        let q = f32_to_e4m3((1.0 / s).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX));
        for &c in &codes {
            assert_eq!(c, q);
        }
    }

    #[test]
    fn rejects_bad_shapes() {
        // k not multiple of 128
        let err = fp8_gemm_ref(&[0; 100], &[0], &[0; 100], &[0], 1, 1, 100, AccumMode::Exact)
            .unwrap_err();
        assert!(err.contains("not divisible by 128"), "{err}");

        // a length mismatch
        let err = fp8_gemm_ref(&[0; 10], &[0], &[0; 128], &[0], 1, 1, 128, AccumMode::Exact)
            .unwrap_err();
        assert!(err.contains("a len"), "{err}");

        // fp4: k=100 is even but not % 128
        let err = fp4_gemm_ref(&[0; 100], &[0], &[0; 50], &[0], 1, 1, 100, AccumMode::Exact)
            .unwrap_err();
        assert!(
            err.contains("not divisible by 128") || err.contains("even"),
            "{err}"
        );

        let err = fp4_gemm_ref(&[0; 127], &[0], &[0; 64], &[0], 1, 1, 127, AccumMode::Exact)
            .unwrap_err();
        assert!(err.contains("even"), "{err}");

        // act_quant bad block
        let err = act_quant_fp8_codes(&[0.0; 100], 100, 128).unwrap_err();
        assert!(err.contains("not divisible"), "{err}");

        // linear length mismatch
        let err = linear_fp8_ref(&[0.0; 10], &[0; 128], &[0], 1, 1, 128, AccumMode::Exact)
            .unwrap_err();
        assert!(err.contains("x len"), "{err}");
    }

    /// Report measured err_ref at k=4096 for both GEMMs (acceptance criterion
    /// denominator). Runs as a test so `cargo test` prints the numbers.
    #[test]
    fn report_err_ref_at_k4096() {
        let k = 4096usize;

        // --- FP8 ---
        {
            let m = 4usize;
            let n = 128usize;
            let mut a = vec![0u8; m * k];
            let mut b = vec![0u8; n * k];
            for i in 0..a.len() {
                a[i] = [0x39u8, 0x4b, 0x5d, 0x6e, 0x78, 0x72, 0x6a, 0x55][i % 8];
            }
            for i in 0..b.len() {
                b[i] = [0x3au8, 0x4c, 0x5e, 0x70, 0x77, 0x61, 0x53, 0x48][i % 8];
            }
            let kb = k / 128;
            let a_s: Vec<u8> = (0..m * kb).map(|i| ue8((i % 3) as i32 - 1)).collect();
            let b_s: Vec<u8> = (0..(n / 128) * kb)
                .map(|i| ue8(((i % 4) as i32) - 2))
                .collect();
            let exact = fp8_gemm_ref(&a, &a_s, &b, &b_s, m, n, k, AccumMode::Exact).unwrap();
            let pref =
                fp8_gemm_ref(&a, &a_s, &b, &b_s, m, n, k, AccumMode::ReferenceOrder).unwrap();
            let err = rel_err(&pref, &exact);
            let expect = (k as f64).sqrt() * (2.0f64).powi(-24);
            println!(
                "REPORT fp8_gemm_ref k={k}: err_ref={err:.8e}  sqrt(K)*2^-24={expect:.8e}  ratio={:.4}",
                err / expect
            );
            assert!(err > 0.0 && err < 1e-4);
        }

        // --- FP4 ---
        {
            let m = 4usize;
            let n = 8usize;
            let mut a = vec![0u8; m * k];
            let mut b = vec![0u8; n * (k / 2)];
            for i in 0..a.len() {
                a[i] = [0x58u8, 0x5a, 0x6c, 0x70, 0x78, 0x74, 0x62, 0x4e][i % 8];
            }
            for i in 0..b.len() {
                let mags = [5u8, 6, 7, 4, 7, 6, 5];
                let lo = mags[i % mags.len()];
                let hi = mags[(i / 3) % mags.len()];
                b[i] = lo | (hi << 4);
            }
            let k_act = k / 128;
            let k_wgt = k / 32;
            let a_s: Vec<u8> = (0..m * k_act).map(|i| ue8((i % 3) as i32 - 1)).collect();
            let b_s: Vec<u8> = (0..n * k_wgt)
                .map(|i| ue8(((i % 5) as i32) - 2))
                .collect();
            let exact = fp4_gemm_ref(&a, &a_s, &b, &b_s, m, n, k, AccumMode::Exact).unwrap();
            let pref =
                fp4_gemm_ref(&a, &a_s, &b, &b_s, m, n, k, AccumMode::ReferenceOrder).unwrap();
            let err = rel_err(&pref, &exact);
            let expect = (k as f64).sqrt() * (2.0f64).powi(-24);
            println!(
                "REPORT fp4_gemm_ref k={k}: err_ref={err:.8e}  sqrt(K)*2^-24={expect:.8e}  ratio={:.4}",
                err / expect
            );
            assert!(err > 0.0 && err < 1e-4);
        }
    }
}
