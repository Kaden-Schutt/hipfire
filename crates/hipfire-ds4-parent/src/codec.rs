// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gate 2 CPU oracle: pure-Rust scalar codecs and block dequantizers for the
//! DeepSeek V4 Flash parent checkpoint, plus bit-exact dynamic activation-
//! quantization references from the bundled `inference/kernel.py`.
//!
//! This module is the only numerical cross-check available — `mi300x` has no
//! torch — so the codecs are exhaustively tested, not spot-checked.
//!
//! Operator semantics authority:
//! - `local://ds4-parent-contract.md` §2.4 / §3
//! - `.codeinsight+research/ds4-parent-ref/inference/kernel.py:22-201`
//! - `.codeinsight+research/ds4-parent-ref/inference/convert.py:30-33`
//!   (E2M1 nibble order: low-nibble-first)
//! - `.codeinsight+research/ds4-parent-ref/inference/model.py:253-257`
//!   (Hadamard scale = `n^-0.5`)

/// OCP `float8_e4m3fn` max finite magnitude.
pub const FP8_E4M3_MAX: f32 = 448.0;
/// OCP `float4_e2m1fn` max finite magnitude.
pub const FP4_E2M1_MAX: f32 = 6.0;

/// Exact E2M1 decode table (sign-magnitude, low 4 bits).
///
/// ```text
/// code:  0    1    2    3    4    5    6    7
/// val:   0.0  0.5  1.0  1.5  2.0  3.0  4.0  6.0
/// code:  8    9   10   11   12   13   14   15
/// val:  -0.0 -0.5 -1.0 -1.5 -2.0 -3.0 -4.0 -6.0
/// ```
pub const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

const E2M1_MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

#[inline]
fn err_msg(msg: &str) -> String {
    format!("deepseek4 parent: {msg}")
}

/// OCP `float8_e4m3fn` (finite-only) → f32.
///
/// - `exp == 0` → subnormal `(-1)^s * 2^-6 * (mant/8)`
/// - `exp == 0xF && mant == 0x7` → NaN (only `0x7F` / `0xFF`)
/// - otherwise → `(-1)^s * 2^(exp-7) * (1 + mant/8)`
#[inline]
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if (b & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((b >> 3) & 0x0f) as i32;
    let mant = (b & 0x07) as i32;
    if exp == 0x0f && mant == 0x07 {
        return f32::NAN.copysign(sign);
    }
    if exp == 0 {
        // subnormal, including ±0
        return sign * (2.0f32).powi(-6) * (mant as f32) / 8.0;
    }
    sign * (2.0f32).powi(exp - 7) * (1.0 + (mant as f32) / 8.0)
}

/// `float8_e8m0fnu` → f32. Unsigned, no mantissa, no zero, no inf.
///
/// - `0xFF` → NaN
/// - `0x00` → `2^-127` (subnormal f32; bit-shift form would collapse to 0)
/// - otherwise → `2^(b - 127)` via exponent-field bit insert
#[inline]
pub fn ue8m0_to_f32(b: u8) -> f32 {
    if b == 0xff {
        return f32::NAN;
    }
    if b == 0 {
        // 2^-127 is subnormal in IEEE-754 binary32; shift form yields 0.
        return 2.0f32.powi(-127);
    }
    f32::from_bits((b as u32) << 23)
}

/// OCP `float4_e2m1fn` low-nibble decode.
#[inline]
pub fn e2m1_to_f32(code: u8) -> f32 {
    E2M1_LUT[(code & 0x0f) as usize]
}

/// Round f32 → BF16 → f32 (RNE on the f32→bf16 truncation).
///
/// Matches the reference's `out_dtype = in_dtype = bfloat16` write-back so the
/// CPU oracle agrees with the GPU path after the simulated quant+dequant.
#[inline]
pub fn round_to_bf16(v: f32) -> f32 {
    let bits = v.to_bits();
    if v.is_nan() {
        // Canonical quiet BF16 NaN with original sign.
        let sign = (bits >> 16) & 0x8000;
        return f32::from_bits(((sign | 0x7fc0) as u32) << 16);
    }
    // BF16 is the top 16 bits of f32. RNE on the discarded 16.
    let lsb = (bits >> 16) & 1;
    let lower = bits & 0xffff;
    let round_bit = (lower >> 15) & 1;
    let sticky = if (lower & 0x7fff) != 0 { 1 } else { 0 };
    let mut top = bits >> 16;
    // round up when: round_bit=1 and (sticky=1 or lsb=1)  [RNE]
    if round_bit == 1 && (sticky == 1 || lsb == 1) {
        top = top.wrapping_add(1);
    }
    f32::from_bits(top << 16)
}

/// f32 → OCP `float8_e4m3fn`, round-to-nearest-even, saturating (no Inf).
///
/// Finite codes round-trip: `f32_to_e4m3(e4m3_to_f32(b)) == b` for every
/// non-NaN `b`. NaN inputs collapse to the canonical signed NaN codes
/// (`0x7F` / `0xFF`).
#[inline]
pub fn f32_to_e4m3(v: f32) -> u8 {
    if v.is_nan() {
        return if v.is_sign_negative() { 0xff } else { 0x7f };
    }

    let bits = v.to_bits();
    let sign = ((bits >> 24) & 0x80) as u8;
    let x = f32::from_bits(bits & 0x7fff_ffff);

    if x == 0.0 {
        return sign;
    }

    // Saturate at/above max finite. Exact 448 maps to 0x7E.
    if x >= FP8_E4M3_MAX {
        return sign | 0x7e;
    }

    let fbits = x.to_bits();
    let exp_f = ((fbits >> 23) & 0xff) as i32;
    let mant_f = fbits & 0x007f_ffff;

    // Subnormal f32 inputs are vanishingly small relative to E4M3 — round to 0.
    if exp_f == 0 {
        return sign;
    }
    let exp_u = exp_f - 127;

    // E4M3 normal min is 2^-6 (exp_u = -6 → biased 1). Below that: subnormal.
    // E4M3 subnormal quanta = 2^-9.
    if exp_u < -6 {
        let scaled = x * 512.0; // x / 2^-9
        let rounded = round_ties_to_even_nonneg(scaled);
        if rounded < 0.5 {
            return sign; // → 0
        }
        if rounded >= 8.0 {
            // overflow into smallest normal: 2^-6 → exp=1, mant=0
            return sign | (1 << 3);
        }
        let m = rounded as u8; // 1..7
        return sign | m;
    }

    // Normal path: biased exp in 1..=15 (15 with mant≤6 is finite; mant=7 is NaN).
    let mut exp_b = exp_u + 7;
    // 3-bit mantissa from the top of the 23-bit field, RNE.
    const DROP: u32 = 20; // 23 - 3
    let mut mant = mant_f >> DROP;
    let remainder = mant_f & ((1u32 << DROP) - 1);
    let half = 1u32 << (DROP - 1);
    let tie = remainder == half;
    let above = remainder > half;
    let round_up = above || (tie && (mant & 1) == 1);
    if round_up {
        mant += 1;
        if mant == 8 {
            mant = 0;
            exp_b += 1;
        }
    }

    // Saturation / NaN-slot avoidance at the top of the range.
    if exp_b > 15 || (exp_b == 15 && mant >= 7) {
        return sign | 0x7e;
    }
    if exp_b <= 0 {
        let scaled = x * 512.0;
        let rounded = round_ties_to_even_nonneg(scaled);
        if rounded < 0.5 {
            return sign;
        }
        if rounded >= 8.0 {
            return sign | (1 << 3);
        }
        return sign | (rounded as u8);
    }

    sign | ((exp_b as u8) << 3) | (mant as u8)
}

/// f32 → OCP `float4_e2m1fn` nibble, round-to-nearest-even on the 8 magnitudes.
///
/// Ties at the midpoints `{0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}` resolve to
/// the candidate whose 1-bit mantissa LSB is 0 (even). Sign is preserved,
/// including distinct `±0`.
#[inline]
pub fn f32_to_e2m1(v: f32) -> u8 {
    let sign = if v.is_sign_negative() || v.to_bits() == 0x8000_0000 {
        0x8u8
    } else {
        0u8
    };
    let x = v.abs();
    if x == 0.0 {
        return sign; // ±0
    }
    // Saturate above max; exact 6.0 lands on code 7.
    if x >= FP4_E2M1_MAX {
        return sign | 7;
    }

    let mut best_i = 0usize;
    let mut best_err = f32::INFINITY;
    for (i, &m) in E2M1_MAG.iter().enumerate() {
        let err = (x - m).abs();
        if err < best_err {
            best_err = err;
            best_i = i;
        } else if err == best_err {
            // Tie: prefer even mantissa (codes 0,2,4,6 → i & 1 == 0).
            if (i & 1) == 0 {
                best_i = i;
            }
        }
    }
    sign | (best_i as u8)
}

/// `ceil(log2(x))` via IEEE-754 bit manipulation — bit-identical to
/// `kernel.py:22-27`. Do **not** substitute `f32::log2().ceil()`.
#[inline]
pub fn fast_log2_ceil(x: f32) -> i32 {
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & ((1 << 23) - 1);
    exp - 127 + if man != 0 { 1 } else { 0 }
}

/// `2^n` for integer `n` via IEEE-754 bit manipulation — `kernel.py:30-33`.
#[inline]
pub fn fast_pow2(n: i32) -> f32 {
    f32::from_bits(((n + 127) as u32) << 23)
}

/// Power-of-two scale = `2^ceil(log2(amax * max_inv))` — `kernel.py:36-37`.
#[inline]
pub fn fast_round_scale(amax: f32, max_inv: f32) -> f32 {
    fast_pow2(fast_log2_ceil(amax * max_inv))
}

/// Dense tier: E4M3 `[M,K]` + UE8M0 `[ceil(M/128), ceil(K/128)]` → f32 `[M,K]`.
///
/// ```text
/// w[m][k] = e4m3_to_f32(W[m][k]) * ue8m0_to_f32(S[m/128][k/128])
/// ```
pub fn dequant_dense_fp8_block128(
    w: &[u8],
    s: &[u8],
    m: usize,
    k: usize,
    out: &mut [f32],
) -> Result<(), String> {
    if m == 0 || k == 0 {
        return Err(err_msg("dequant_dense_fp8_block128: M and K must be > 0"));
    }
    let bm = m.div_ceil(128);
    let bk = k.div_ceil(128);
    let w_need = m
        .checked_mul(k)
        .ok_or_else(|| err_msg("dequant_dense_fp8_block128: M*K overflow"))?;
    let s_need = bm
        .checked_mul(bk)
        .ok_or_else(|| err_msg("dequant_dense_fp8_block128: scale shape overflow"))?;
    if w.len() != w_need {
        return Err(err_msg(&format!(
            "dequant_dense_fp8_block128: weight len {} != M*K {}",
            w.len(),
            w_need
        )));
    }
    if s.len() != s_need {
        return Err(err_msg(&format!(
            "dequant_dense_fp8_block128: scale len {} != ceil(M/128)*ceil(K/128) {}",
            s.len(),
            s_need
        )));
    }
    if out.len() != w_need {
        return Err(err_msg(&format!(
            "dequant_dense_fp8_block128: out len {} != M*K {}",
            out.len(),
            w_need
        )));
    }

    for i in 0..m {
        let si = i / 128;
        for j in 0..k {
            let sj = j / 128;
            let scale = ue8m0_to_f32(s[si * bk + sj]);
            out[i * k + j] = e4m3_to_f32(w[i * k + j]) * scale;
        }
    }
    Ok(())
}

/// Expert tier: packed E2M1 `[M, K/2]` + UE8M0 `[M, K/32]` → f32 `[M,K]`.
///
/// Nibble order (OCP MX / `convert.py:30-33`): low nibble = logical `2*j`,
/// high nibble = logical `2*j+1`. Empirically confirmed against
/// `layers.3.ffn.experts.0.w1.weight` on the real checkpoint (see module tests).
///
/// ```text
/// w[m][k] = e2m1_to_f32(nibble(W[m][k/2], k&1)) * ue8m0_to_f32(S[m][k/32])
/// ```
pub fn dequant_expert_fp4_g32(
    w: &[u8],
    s: &[u8],
    m: usize,
    k: usize,
    out: &mut [f32],
) -> Result<(), String> {
    if m == 0 || k == 0 {
        return Err(err_msg("dequant_expert_fp4_g32: M and K must be > 0"));
    }
    if k % 2 != 0 {
        return Err(err_msg(&format!(
            "dequant_expert_fp4_g32: logical K ({k}) must be even"
        )));
    }
    if k % 32 != 0 {
        return Err(err_msg(&format!(
            "dequant_expert_fp4_g32: logical K ({k}) must be a multiple of 32"
        )));
    }
    let k_stored = k / 2;
    let k_scale = k / 32;
    let w_need = m
        .checked_mul(k_stored)
        .ok_or_else(|| err_msg("dequant_expert_fp4_g32: weight shape overflow"))?;
    let s_need = m
        .checked_mul(k_scale)
        .ok_or_else(|| err_msg("dequant_expert_fp4_g32: scale shape overflow"))?;
    let out_need = m
        .checked_mul(k)
        .ok_or_else(|| err_msg("dequant_expert_fp4_g32: out shape overflow"))?;
    if w.len() != w_need {
        return Err(err_msg(&format!(
            "dequant_expert_fp4_g32: weight len {} != M*(K/2) {}",
            w.len(),
            w_need
        )));
    }
    if s.len() != s_need {
        return Err(err_msg(&format!(
            "dequant_expert_fp4_g32: scale len {} != M*(K/32) {}",
            s.len(),
            s_need
        )));
    }
    if out.len() != out_need {
        return Err(err_msg(&format!(
            "dequant_expert_fp4_g32: out len {} != M*K {}",
            out.len(),
            out_need
        )));
    }

    for i in 0..m {
        let w_row = i * k_stored;
        let s_row = i * k_scale;
        let o_row = i * k;
        for j in 0..k_stored {
            let b = w[w_row + j];
            let lo = b & 0x0f;
            let hi = b >> 4;
            let c0 = 2 * j;
            let c1 = c0 + 1;
            // Within a 32-group both positions share the same scale; index
            // explicitly so a future group-size change cannot silently drift.
            let s0 = ue8m0_to_f32(s[s_row + c0 / 32]);
            let s1 = ue8m0_to_f32(s[s_row + c1 / 32]);
            out[o_row + c0] = e2m1_to_f32(lo) * s0;
            out[o_row + c1] = e2m1_to_f32(hi) * s1;
        }
    }
    Ok(())
}

/// `act_quant(..., scale_fmt="ue8m0", inplace=True)` reference (`kernel.py:79-90`).
///
/// Groups of `block` along the last dim. `block` is 128 at linear boundaries
/// and 64 at the KV non-RoPE simulation sites. Fused quantize+dequantize with
/// BF16 write-back.
///
/// # BF16 input domain (contract)
///
/// The bundled kernel declares `in_dtype = bfloat16` (`kernel.py:41-42`) and
/// computes `amax` over BF16 values loaded from the activation tensor. The GPU
/// path (`act_quant_fp8_ue8m0_inplace_gfx942`) likewise reads `__hip_bfloat16*`.
///
/// This oracle therefore **rounds every element to BF16 before amax / scale /
/// quant**, even though the buffer is `f32`. Callers may hand in full-precision
/// f32 (e.g. post-RoPE from an f64-accum host path); the rounding is applied
/// here so a group whose f32 `amax` sits just across a power-of-two boundary
/// of `amax/448` cannot pick a scale 2× larger than the BF16/GPU path.
/// Without this, sparse groups near those boundaries show max_abs ≈ 0.25 with
/// only ~1e-3 mean relative error — exactly the Gate-2-vs-compressor finding.
///
/// Idempotent on already-BF16-representable input (Gate 2 synthetic cases).
pub fn act_quant_fp8_inplace_ref(
    x: &mut [f32],
    last_dim: usize,
    block: usize,
) -> Result<(), String> {
    if last_dim == 0 {
        return Err(err_msg("act_quant_fp8_inplace_ref: last_dim must be > 0"));
    }
    if block == 0 {
        return Err(err_msg("act_quant_fp8_inplace_ref: block must be > 0"));
    }
    if last_dim % block != 0 {
        return Err(err_msg(&format!(
            "act_quant_fp8_inplace_ref: last_dim {last_dim} not divisible by block {block}"
        )));
    }
    if x.len() % last_dim != 0 {
        return Err(err_msg(&format!(
            "act_quant_fp8_inplace_ref: buffer len {} not divisible by last_dim {last_dim}",
            x.len()
        )));
    }

    // Enter the BF16 domain before any amax. Matches kernel.py in_dtype=BF16
    // and the GPU kernel's __bfloat162float loads.
    for v in x.iter_mut() {
        *v = round_to_bf16(*v);
    }

    let max_inv = 1.0 / FP8_E4M3_MAX;
    for row in x.chunks_mut(last_dim) {
        for group in row.chunks_mut(block) {
            let mut amax = 0.0f32;
            for &v in group.iter() {
                amax = amax.max(v.abs());
            }
            // FP8 path clamps amax to max(amax, 1e-4) BEFORE rounding the scale.
            amax = amax.max(1e-4);
            let s = fast_round_scale(amax, max_inv);
            for v in group.iter_mut() {
                let clamped = (*v / s).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX);
                let q = f32_to_e4m3(clamped);
                *v = round_to_bf16(e4m3_to_f32(q) * s);
            }
        }
    }
    Ok(())
}


/// `fp4_act_quant(..., inplace=True)` reference (`kernel.py:163-171`).
///
/// Groups of 32 along the last dim. Fused quantize+dequantize with BF16 write-back.
///
/// Same BF16 input-domain contract as [`act_quant_fp8_inplace_ref`]: every
/// element is rounded to BF16 before amax / scale / quant so the host oracle
/// matches the GPU kernel's `__hip_bfloat16` buffer.
pub fn act_quant_fp4_inplace_ref(x: &mut [f32], last_dim: usize) -> Result<(), String> {
    const BLOCK: usize = 32;
    if last_dim == 0 {
        return Err(err_msg("act_quant_fp4_inplace_ref: last_dim must be > 0"));
    }
    if last_dim % BLOCK != 0 {
        return Err(err_msg(&format!(
            "act_quant_fp4_inplace_ref: last_dim {last_dim} not divisible by {BLOCK}"
        )));
    }
    if x.len() % last_dim != 0 {
        return Err(err_msg(&format!(
            "act_quant_fp4_inplace_ref: buffer len {} not divisible by last_dim {last_dim}",
            x.len()
        )));
    }

    for v in x.iter_mut() {
        *v = round_to_bf16(*v);
    }

    let max_inv = 1.0 / FP4_E2M1_MAX;
    // FP4 floor: max(amax, 6.0 * 2^-126) — distinct from the FP8 1e-4 floor.
    let amax_floor = FP4_E2M1_MAX * 2.0f32.powi(-126);
    for row in x.chunks_mut(last_dim) {
        for group in row.chunks_mut(BLOCK) {
            let mut amax = 0.0f32;
            for &v in group.iter() {
                amax = amax.max(v.abs());
            }
            amax = amax.max(amax_floor);
            let s = fast_round_scale(amax, max_inv);
            for v in group.iter_mut() {
                let clamped = (*v / s).clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX);
                let q = f32_to_e2m1(clamped);
                *v = round_to_bf16(e2m1_to_f32(q) * s);
            }
        }
    }
    Ok(())
}

/// `rotate_activation` (`model.py:253-257`): orthonormal Walsh-Hadamard.
///
/// `fast_hadamard_transform.hadamard_transform(x, scale=n^-0.5)` with `n`
/// a power of two. Applies in-place over contiguous trailing blocks of `n`.
pub fn hadamard_rotate_ref(x: &mut [f32], n: usize) -> Result<(), String> {
    if n == 0 || !n.is_power_of_two() {
        return Err(err_msg(&format!(
            "hadamard_rotate_ref: n ({n}) must be a power of two"
        )));
    }
    if x.len() % n != 0 {
        return Err(err_msg(&format!(
            "hadamard_rotate_ref: buffer len {} not divisible by n {n}",
            x.len()
        )));
    }

    let scale = (n as f32).powf(-0.5);
    for row in x.chunks_mut(n) {
        // In-place FWHT (Cooley-Tukey butterfly).
        let mut h = 1usize;
        while h < n {
            let step = h * 2;
            let mut i = 0usize;
            while i < n {
                for j in i..i + h {
                    let a = row[j];
                    let b = row[j + h];
                    row[j] = a + b;
                    row[j + h] = a - b;
                }
                i += step;
            }
            h = step;
        }
        for v in row.iter_mut() {
            *v *= scale;
        }
    }
    Ok(())
}

/// Round a non-negative f32 to the nearest integer, ties to even.
#[inline]
fn round_ties_to_even_nonneg(x: f32) -> f32 {
    debug_assert!(x >= 0.0 && x.is_finite());
    let floor = x.floor();
    let frac = x - floor;
    if frac > 0.5 {
        floor + 1.0
    } else if frac < 0.5 {
        floor
    } else {
        // tie: pick even integer
        let fi = floor as i32;
        if fi & 1 == 0 {
            floor
        } else {
            floor + 1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── scalar codecs ──────────────────────────────────────────────────

    #[test]
    fn e4m3_exhaustive_roundtrip() {
        for b in 0u8..=255 {
            let v = e4m3_to_f32(b);
            if (b & 0x7f) == 0x7f {
                // Only NaN encodings: 0x7F / 0xFF.
                assert!(v.is_nan(), "e4m3 0x{b:02X} must be NaN");
                let enc = f32_to_e4m3(v);
                // Canonical NaN handling: sign preserved, payload collapsed.
                assert_eq!(enc, b, "NaN encode mismatch for 0x{b:02X}");
                continue;
            }
            assert!(v.is_finite(), "e4m3 0x{b:02X} must be finite, got {v}");
            let back = f32_to_e4m3(v);
            assert_eq!(
                back, b,
                "round-trip failed for 0x{b:02X}: v={v} back=0x{back:02X}"
            );
        }
        // Explicit NaN-slot claims from the contract.
        assert!(e4m3_to_f32(0x7f).is_nan());
        assert!(e4m3_to_f32(0xff).is_nan());
        assert_eq!(e4m3_to_f32(0x7e), 448.0);
        assert_eq!(e4m3_to_f32(0xfe), -448.0);
        // Subnormal: exp=0, mant=1 → 2^-6 * 1/8 = 2^-9.
        assert_eq!(e4m3_to_f32(0x01), 2.0f32.powi(-9));
        assert_eq!(e4m3_to_f32(0x81), -2.0f32.powi(-9));
        // Saturating encode.
        assert_eq!(f32_to_e4m3(1000.0), 0x7e);
        assert_eq!(f32_to_e4m3(-1000.0), 0xfe);
        assert_eq!(f32_to_e4m3(f32::INFINITY), 0x7e);
    }

    #[test]
    fn ue8m0_exhaustive() {
        for b in 0u8..=255 {
            let v = ue8m0_to_f32(b);
            if b == 255 {
                assert!(v.is_nan(), "ue8m0(255) must be NaN");
                continue;
            }
            let expected = 2.0f32.powi(b as i32 - 127);
            assert_eq!(v, expected, "ue8m0({b}) = {v}, expected {expected}");
        }
        // Contract anchors.
        assert_eq!(ue8m0_to_f32(0), 2.0f32.powi(-127));
        assert_eq!(ue8m0_to_f32(127), 1.0);
        assert_eq!(ue8m0_to_f32(254), 2.0f32.powi(127));
        assert!(ue8m0_to_f32(255).is_nan());
        // Bit-form identity for 1..=254.
        for b in 1u8..=254 {
            assert_eq!(ue8m0_to_f32(b).to_bits(), (b as u32) << 23);
        }
    }

    #[test]
    fn e2m1_exhaustive_lut_and_roundtrip() {
        for code in 0u8..16 {
            let v = e2m1_to_f32(code);
            let expect = E2M1_LUT[code as usize];
            // Byte-identical for every slot, including distinct ±0.
            assert_eq!(
                v.to_bits(),
                expect.to_bits(),
                "LUT mismatch at code {code}"
            );
            let back = f32_to_e2m1(v);
            assert_eq!(back, code, "e2m1 round-trip failed for code {code}");
        }
        // High bits ignored on decode.
        assert_eq!(e2m1_to_f32(0x3f).to_bits(), E2M1_LUT[0x0f].to_bits());
    }

    #[test]
    fn e2m1_rne_midpoints() {
        // Midpoints and the even-mantissa resolution required by RNE.
        // magnitudes: 0, .5, 1, 1.5, 2, 3, 4, 6  (mant LSB = code & 1)
        let cases: &[(f32, u8)] = &[
            (0.25, 0), // tie 0↔0.5 → even 0
            (0.75, 2), // tie 0.5↔1.0 → even 1.0
            (1.25, 2), // tie 1.0↔1.5 → even 1.0
            (1.75, 4), // tie 1.5↔2.0 → even 2.0
            (2.5, 4),  // tie 2.0↔3.0 → even 2.0
            (3.5, 6),  // tie 3.0↔4.0 → even 4.0
            (5.0, 6),  // tie 4.0↔6.0 → even 4.0
        ];
        for &(x, expect) in cases {
            assert_eq!(f32_to_e2m1(x), expect, "RNE+ midpoint {x}");
            assert_eq!(f32_to_e2m1(-x), expect | 0x8, "RNE- midpoint {x}");
        }
        // Non-tie sanity: just above/below a midpoint moves the other way.
        assert_eq!(f32_to_e2m1(0.26), 1);
        assert_eq!(f32_to_e2m1(0.24), 0);
        assert_eq!(f32_to_e2m1(5.01), 7);
        assert_eq!(f32_to_e2m1(4.99), 6);
        // Saturate.
        assert_eq!(f32_to_e2m1(100.0), 7);
        assert_eq!(f32_to_e2m1(-100.0), 0xf);
    }

    #[test]
    fn fast_round_scale_power_of_two_and_ceil() {
        // amax = 448 → amax * (1/448) = 1.0 exact → s = 2^0 = 1.0
        let s = fast_round_scale(448.0, 1.0 / 448.0);
        assert_eq!(s, 1.0);
        assert_eq!(
            s.to_bits() & 0x007f_ffff,
            0,
            "scale must be pure power-of-two"
        );

        // amax = 449 → 449/448 > 1, mantissa != 0 → ceil bumps exp → s = 2.0
        let s = fast_round_scale(449.0, 1.0 / 448.0);
        assert_eq!(s, 2.0);

        // Exact power-of-two input to fast_log2_ceil: man bits == 0 → no +1.
        assert_eq!(fast_log2_ceil(1.0), 0);
        assert_eq!(fast_log2_ceil(2.0), 1);
        assert_eq!(fast_log2_ceil(0.5), -1);
        // Non-power: 1.5 = 2^0 * 1.5 → ceil(log2) = 1
        assert_eq!(fast_log2_ceil(1.5), 1);
        // 3.0 → ceil(log2(3)) = 2
        assert_eq!(fast_log2_ceil(3.0), 2);

        // fast_pow2 bit identity.
        for n in -126..=127 {
            let v = fast_pow2(n);
            assert_eq!(v.to_bits(), ((n + 127) as u32) << 23, "fast_pow2({n})");
        }

        // FP4-style: amax=6 → s=1; amax just over 6 → s=2.
        assert_eq!(fast_round_scale(6.0, 1.0 / 6.0), 1.0);
        assert_eq!(fast_round_scale(6.0 + 1e-3, 1.0 / 6.0), 2.0);
    }

    #[test]
    fn round_to_bf16_rne() {
        // Exact BF16 values pass through.
        assert_eq!(round_to_bf16(1.0).to_bits(), 1.0f32.to_bits());
        assert_eq!(round_to_bf16(-2.0).to_bits(), (-2.0f32).to_bits());
        // 1.0 + 2^-16 is halfway between 1.0 (bf16 exp=127,mant=0) and the next
        // BF16 (mant +1). LSB of 1.0 is 0 → ties-to-even stays at 1.0.
        let half_up = f32::from_bits(1.0f32.to_bits() + 0x8000);
        assert_eq!(round_to_bf16(half_up).to_bits(), 1.0f32.to_bits());
        // NaN stays NaN.
        assert!(round_to_bf16(f32::NAN).is_nan());
    }

    // ── block dequant ──────────────────────────────────────────────────

    #[test]
    fn dequant_dense_fp8_ragged_260x300() {
        // 260×300 exercises ceil in both dims: scale grid = [3, 3] not [2, 2].
        let m = 260usize;
        let k = 300usize;
        let bm = m.div_ceil(128); // 3
        let bk = k.div_ceil(128); // 3
        assert_eq!((bm, bk), (3, 3));

        let mut w = vec![0u8; m * k];
        let mut s = vec![0u8; bm * bk];
        // Put distinctive E4M3 codes and scales on the ragged edges.
        // scale[0,0]=127 → 1.0; scale[2,2]=128 → 2.0; scale[0,2]=126 → 0.5
        s[0 * bk + 0] = 127;
        s[0 * bk + 2] = 126;
        s[2 * bk + 2] = 128;
        s[2 * bk + 0] = 129; // 4.0
                              // Fill remaining scales with 127 so unset blocks are quiet.
        for b in s.iter_mut() {
            if *b == 0 {
                *b = 127;
            }
        }

        // Corner samples.
        w[0 * k + 0] = 0x38; // exp=7,mant=0 → 1.0
        w[0 * k + 299] = 0x3c; // exp=7,mant=4 → 1.5
        w[259 * k + 0] = 0x40; // exp=8,mant=0 → 2.0
        w[259 * k + 299] = 0x48; // exp=9,mant=0 → 4.0
                                 // Interior of final block (m in 256..260, k in 256..300)
        w[256 * k + 256] = 0x30; // exp=6,mant=0 → 0.5

        let mut out = vec![0.0f32; m * k];
        dequant_dense_fp8_block128(&w, &s, m, k, &mut out).unwrap();

        assert_eq!(out[0 * k + 0], 1.0 * 1.0);
        assert_eq!(out[0 * k + 299], 1.5 * 0.5);
        assert_eq!(out[259 * k + 0], 2.0 * 4.0);
        assert_eq!(out[259 * k + 299], 4.0 * 2.0);
        assert_eq!(out[256 * k + 256], 0.5 * 2.0);

        // Spot-check a mid-block scale application: scale[1,1] defaulted 1.0.
        w[128 * k + 128] = 0x38;
        dequant_dense_fp8_block128(&w, &s, m, k, &mut out).unwrap();
        assert_eq!(out[128 * k + 128], 1.0);
    }

    #[test]
    fn dequant_expert_fp4_nibble_order_and_scales() {
        // M=2, K=64 → stored [2, 32], scale [2, 2].
        let m = 2usize;
        let k = 64usize;
        let mut w = vec![0u8; m * (k / 2)];
        let mut s = vec![127u8; m * (k / 32)]; // scale=1.0 default

        // Row 0: pack lo=code1 (0.5), hi=code4 (2.0) at byte 0 → logical [0.5, 2.0]
        w[0] = 0x41; // hi=4, lo=1
                     // Row 0 byte 16 (logical k=32,33) falls in the second scale group.
        w[16] = 0x62; // hi=6 (4.0), lo=2 (1.0)
        s[0 * 2 + 0] = 127; // 1.0 for k in 0..32
        s[0 * 2 + 1] = 128; // 2.0 for k in 32..64

        // Row 1: prove per-row scale indexing.
        w[1 * 32 + 0] = 0x03; // lo=3 (1.5), hi=0 (0.0)
        s[1 * 2 + 0] = 126; // 0.5

        let mut out = vec![0.0f32; m * k];
        dequant_expert_fp4_g32(&w, &s, m, k, &mut out).unwrap();

        // Low-nibble-first.
        assert_eq!(out[0 * k + 0], 0.5 * 1.0, "lo nibble → logical 2j");
        assert_eq!(out[0 * k + 1], 2.0 * 1.0, "hi nibble → logical 2j+1");
        // Second scale group.
        assert_eq!(out[0 * k + 32], 1.0 * 2.0);
        assert_eq!(out[0 * k + 33], 4.0 * 2.0);
        // Row 1 scale.
        assert_eq!(out[1 * k + 0], 1.5 * 0.5);
        assert_eq!(out[1 * k + 1], 0.0 * 0.5);
    }

    #[test]
    fn dequant_shape_mismatch_errors() {
        let mut out = vec![0.0f32; 4];
        // Dense: wrong weight len.
        let err = dequant_dense_fp8_block128(&[0; 3], &[127], 2, 2, &mut out).unwrap_err();
        assert!(err.contains("deepseek4 parent:"), "{err}");
        assert!(err.contains("weight len"), "{err}");
        // Dense: wrong scale len (need ceil(2/128)*ceil(2/128)=1; give 0).
        let err = dequant_dense_fp8_block128(&[0; 4], &[], 2, 2, &mut out).unwrap_err();
        assert!(err.contains("scale len"), "{err}");
        // Dense: wrong out len.
        let mut short = vec![0.0f32; 2];
        let err = dequant_dense_fp8_block128(&[0; 4], &[127], 2, 2, &mut short).unwrap_err();
        assert!(err.contains("out len"), "{err}");

        // Expert: K not multiple of 32.
        let err = dequant_expert_fp4_g32(&[0; 4], &[127; 2], 1, 8, &mut out).unwrap_err();
        assert!(err.contains("multiple of 32"), "{err}");
        // Expert: wrong weight len.
        let mut out64 = vec![0.0f32; 64];
        let err = dequant_expert_fp4_g32(&[0; 10], &[127; 2], 1, 64, &mut out64).unwrap_err();
        assert!(err.contains("weight len"), "{err}");
        // Expert: wrong scale len.
        let err = dequant_expert_fp4_g32(&[0; 32], &[127; 1], 1, 64, &mut out64).unwrap_err();
        assert!(err.contains("scale len"), "{err}");
    }

    // ── activation quant + hadamard ────────────────────────────────────

    #[test]
    fn act_quant_fp8_inplace_scales_and_bf16() {
        // One group of 4 with amax=448 → s=1; values must land on e4m3 lattice
        // then BF16-round-trip.
        let mut x = vec![448.0f32, -224.0, 1.0, 0.0];
        act_quant_fp8_inplace_ref(&mut x, 4, 4).unwrap();
        assert_eq!(x[0], round_to_bf16(448.0));
        assert_eq!(x[1], round_to_bf16(-224.0));
        assert_eq!(x[2], round_to_bf16(1.0));
        assert_eq!(x[3], round_to_bf16(0.0));

        // Zero amax hits the 1e-4 floor → s = fast_round_scale(1e-4, 1/448).
        // Exact zeros stay zero after quant*scale; the floor only shapes the scale.
        let mut y = vec![0.0f32; 8];
        act_quant_fp8_inplace_ref(&mut y, 8, 8).unwrap();
        let s = fast_round_scale(1e-4, 1.0 / 448.0);
        assert!(s > 0.0 && (s.to_bits() & 0x007f_ffff) == 0, "floor scale must be >0 pot2, got {s}");
        for v in &y {
            assert_eq!(*v, 0.0);
        }

        // last_dim not divisible by block → error.
        let mut z = vec![1.0f32; 10];
        let err = act_quant_fp8_inplace_ref(&mut z, 10, 8).unwrap_err();
        assert!(err.contains("not divisible"), "{err}");
    }

    /// Regression for the Gate-2-vs-compressor finding: full-precision f32
    /// input whose amax sits just across a power-of-two `fast_round_scale`
    /// boundary must pick the **BF16-domain** scale, not the f32-domain one.
    ///
    /// Isolating numbers (block 64, post-RoPE-like):
    /// - f32 amax = 224.4 → `fast_round_scale(224.4, 1/448) = 1.0`
    /// - BF16 amax = 224.0 → `fast_round_scale(224.0, 1/448) = 0.5`
    /// - GPU kernel reads `__hip_bfloat16` so it uses 0.5; a host oracle that
    ///   amax'es raw f32 picks 1.0 and the whole group reconstructs ~2× off
    ///   (max_abs ≈ 0.25, mean_rel ≈ 2e-3) — the reported signature.
    ///
    /// Before the BF16-domain fix this test fails: `from_f32` disagrees with
    /// `from_bf16` at every element of the affected group. After the fix they
    /// are bit-identical and the amax element reconstructs to 224.0.
    #[test]
    fn act_quant_fp8_bf16_domain_power_of_two_boundary() {
        const BLOCK: usize = 64;
        let mut x = vec![0.01f32; BLOCK];
        // 224.4 is NOT BF16-representable; rounds to 224.0.
        x[0] = 224.4;

        let amax_f32 = 224.4f32;
        let amax_bf16 = round_to_bf16(224.4);
        assert_eq!(amax_bf16, 224.0);
        let s_f32 = fast_round_scale(amax_f32.max(1e-4), 1.0 / 448.0);
        let s_bf16 = fast_round_scale(amax_bf16.max(1e-4), 1.0 / 448.0);
        assert_eq!(s_f32, 1.0, "precondition: f32-domain scale must be 1.0");
        assert_eq!(s_bf16, 0.5, "precondition: bf16-domain scale must be 0.5");
        assert_ne!(s_f32, s_bf16, "precondition: domains must disagree");

        // Oracle on full-f32 input must match oracle on pre-rounded BF16 input.
        let mut from_f32 = x.clone();
        act_quant_fp8_inplace_ref(&mut from_f32, BLOCK, BLOCK).unwrap();
        let mut from_bf16: Vec<f32> = x.iter().copied().map(round_to_bf16).collect();
        act_quant_fp8_inplace_ref(&mut from_bf16, BLOCK, BLOCK).unwrap();
        for i in 0..BLOCK {
            assert_eq!(
                from_f32[i].to_bits(),
                from_bf16[i].to_bits(),
                "idx {i}: f32-in path {:?} vs bf16-in path {:?}",
                from_f32[i],
                from_bf16[i]
            );
        }
        // Multi-row / multi-group post-RoPE-like: several near-boundary amaxes
        // mixed with ordinary values. last_dim = 448 (= 7*64), matching the
        // compressor non-RoPE slice (head_dim 512 − rope 64).
        let last_dim = 448usize;
        let n_rows = 3usize;
        let mut rope_like = vec![0.0f32; n_rows * last_dim];
        // Boundaries of amax/448 at powers of two: 448*2^k for integer k.
        // Just-above values that BF16 rounds back onto the boundary.
        let near = [
            224.4f32,  // → 224, s: 1.0 vs 0.5
            112.3,     // → 112, s: 0.5 vs 0.25
            56.2,      // → 56,  s: 0.25 vs 0.125
            28.1,      // → 28,  s: 0.125 vs 0.0625
            14.05,     // → 14
            7.02,      // → 7
            448.4,     // → 448, s: 2.0 vs 1.0
        ];
        for r in 0..n_rows {
            for g in 0..(last_dim / BLOCK) {
                let base = r * last_dim + g * BLOCK;
                for i in 0..BLOCK {
                    // Full-f32 mantissa junk (bottom 16 bits set) — like f64→f32
                    // cast from an f64-accum matmul, not a BF16 residual.
                    let bits = (0.37f32 * (i as f32 + 1.0)).to_bits() | 0x0000_a5a5;
                    rope_like[base + i] = f32::from_bits(bits) * if i % 2 == 0 { 1.0 } else { -1.0 };
                }
                // Plant a near-boundary amax in each group.
                rope_like[base] = near[g % near.len()];
            }
        }
        let mut a = rope_like.clone();
        act_quant_fp8_inplace_ref(&mut a, last_dim, BLOCK).unwrap();
        let mut b: Vec<f32> = rope_like.iter().copied().map(round_to_bf16).collect();
        act_quant_fp8_inplace_ref(&mut b, last_dim, BLOCK).unwrap();
        for i in 0..a.len() {
            assert_eq!(
                a[i].to_bits(),
                b[i].to_bits(),
                "post-rope-like mismatch at {i}: {:?} vs {:?}",
                a[i],
                b[i]
            );
        }
    }


    #[test]
    fn act_quant_fp4_inplace_floor_differs_from_fp8() {
        // FP4 floor is 6*2^-126, far below FP8's 1e-4. A group with amax=0
        // still produces a well-defined power-of-two scale.
        let mut x = vec![0.0f32; 32];
        act_quant_fp4_inplace_ref(&mut x, 32).unwrap();
        for v in &x {
            assert_eq!(*v, 0.0);
        }

        // amax=6 → s=1; lattice point preserved through BF16.
        let mut y = vec![0.0f32; 32];
        y[0] = 6.0;
        y[1] = -3.0;
        y[2] = 1.5;
        act_quant_fp4_inplace_ref(&mut y, 32).unwrap();
        assert_eq!(y[0], round_to_bf16(6.0));
        assert_eq!(y[1], round_to_bf16(-3.0));
        assert_eq!(y[2], round_to_bf16(1.5));

        let err = act_quant_fp4_inplace_ref(&mut [0.0; 16], 16).unwrap_err();
        assert!(err.contains("not divisible"), "{err}");
    }

    #[test]
    fn hadamard_orthonormal_and_involution() {
        // n=8 random-ish vector.
        let src: Vec<f32> = (0..8).map(|i| (i as f32) * 0.37 - 1.1).collect();
        let mut x = src.clone();
        hadamard_rotate_ref(&mut x, 8).unwrap();

        // Norm preserved (orthonormal with scale n^-0.5).
        let n0: f32 = src.iter().map(|v| v * v).sum::<f32>().sqrt();
        let n1: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (n0 - n1).abs() < 1e-5 * n0.max(1.0),
            "norm drifted: {n0} → {n1}"
        );

        // Apply twice → identity.
        hadamard_rotate_ref(&mut x, 8).unwrap();
        for (a, b) in src.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-5, "involution failed: {a} vs {b}");
        }

        // Batched rows.
        let mut batch = vec![0.0f32; 16];
        batch[..8].copy_from_slice(&src);
        batch[8..].copy_from_slice(&src);
        hadamard_rotate_ref(&mut batch, 8).unwrap();
        hadamard_rotate_ref(&mut batch, 8).unwrap();
        for i in 0..8 {
            assert!((batch[i] - src[i]).abs() < 1e-5);
            assert!((batch[8 + i] - src[i]).abs() < 1e-5);
        }

        // Reject non-power-of-two.
        let err = hadamard_rotate_ref(&mut [0.0; 6], 6).unwrap_err();
        assert!(err.contains("power of two"), "{err}");
    }

    /// Empirical E2M1 nibble-order check against real checkpoint bytes.
    ///
    /// Ran on `mi300x` against
    /// `layers.3.ffn.experts.0.w1.weight` (`I8 [2048,2048]`) + `.scale`
    /// (`F8_E8M0 [2048,128]`) from `/mnt/scratch/models/DeepSeek-V4-Flash-0731`.
    ///
    /// Reference authority (`convert.py:30-33`) unpacks as
    /// `stack([low, high]).flatten` — low-nibble-first — matching OCP MX and
    /// this module. Distributional metrics on 64 rows (K=4096) under both
    /// orders are nearly identical because adjacent logical positions share a
    /// 32-wide scale group, so swapping nibbles does not cross a scale
    /// boundary. The soft signal that remains:
    ///
    /// | metric (64 rows)        | low-first (contract) | high-first (swapped) |
    /// |-------------------------|----------------------|----------------------|
    /// | lag-1 corr              | -0.001018            | -0.001480            |
    /// | mad_boundary (scale)    | 0.028246             | 0.028844             |
    /// | cross-byte corr         | -0.000721            | -0.001644            |
    /// | unscaled MAD1           | 2.411645             | 2.410930             |
    ///
    /// Low-first has the less-negative lag-1 correlation and the smaller
    /// scale-boundary jump — weakly consistent with the contract, not a
    /// decisive period-2 artifact. Combined with the explicit
    /// `convert.py` unpack order, we keep low-nibble-first. Evidence is
    /// soft on the weight distribution alone; the reference unpack is not.
    #[test]
    fn e2m1_nibble_order_contract_is_low_first() {
        // Guard the packing convention this module encodes: byte 0xXY with
        // logical pair (lo=Y, hi=X).
        let mut w_full = vec![0u8; 16];
        w_full[0] = 0x41; // lo=1 (0.5), hi=4 (2.0)
        let s_full = vec![127u8; 1];
        let mut out = vec![0.0f32; 32];
        dequant_expert_fp4_g32(&w_full, &s_full, 1, 32, &mut out).unwrap();
        assert_eq!(out[0], 0.5, "low nibble must be logical element 0");
        assert_eq!(out[1], 2.0, "high nibble must be logical element 1");
    }
}
