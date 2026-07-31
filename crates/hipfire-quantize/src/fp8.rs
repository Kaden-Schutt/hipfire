// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Unsigned E4M3 (FP8) scale codec.
//!
//! Used for per-block scale bytes in the MFP4-E8 family and in MQ4N. The
//! decode here MUST stay bit-identical to the GPU decode
//! (`cvt_e4m3_scale_to_f32_dq` in `kernels/src/dequantize_mfp4g32_{p,e8}_to_f16.hip`),
//! so this is deliberately NOT delegated to a third-party FP8 crate: a
//! different rounding rule would silently change emitted model bytes.
//!
//! Representable unsigned E4M3 (exp 0..15, bias 7, 3 mantissa bits):
//!   * `exp == 0`            → subnormal `2^-6 * mant/8` (includes +0)
//!   * `exp 1..14`, and `exp == 15 && mant < 7` → `2^(exp-7) * (1 + mant/8)`
//!   * `exp == 15 && mant == 7` → NaN, never emitted; max finite is 448.

/// Decode an unsigned E4M3 byte to f32.
///
/// The single NaN code decodes defensively to the max finite value (448) so a
/// stray byte cannot poison an entire block.
#[inline]
pub fn e4m3_decode(byte: u8) -> f32 {
    let exp = ((byte >> 3) & 0xf) as i32;
    let mant = (byte & 0x7) as u32;
    if exp == 0 {
        return (2.0f32).powi(-6) * (mant as f32) / 8.0;
    }
    if exp == 0xf && mant == 7 {
        return 448.0;
    }
    (2.0f32).powi(exp - 7) * (1.0 + (mant as f32) / 8.0)
}

/// Encode a non-negative f32 to the smallest E4M3 code whose decoded value is
/// `>= s` (ceil).
///
/// Round-up, not round-nearest: a block scale must COVER the block maximum, or
/// the subsequent nearest-code search clips the largest magnitude in the block.
/// Defined as a search over [`e4m3_decode`] rather than a closed form, so the
/// pair is round-trip-exact by construction. `s <= 0` maps to `0x00` (+0.0);
/// `s >= 448` saturates to `0x7E`, the largest finite code.
#[inline]
pub fn e4m3_encode_roundup(s: f32) -> u8 {
    if !(s > 0.0) {
        return 0x00;
    }
    if s >= 448.0 {
        return 0x7E;
    }
    // Codes are monotonically non-decreasing in `byte` for sign=0, so the first
    // hit is the ceil. 127 entries, called once per block at quantize time.
    for code in 0u8..=0x7E {
        if e4m3_decode(code) >= s {
            return code;
        }
    }
    0x7E
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_matches_the_ieee_field_construction() {
        // The GPU decode builds the float straight from the exponent/mantissa
        // fields. For every normal code the two must agree bit-for-bit.
        for byte in 0u8..=0xFF {
            let exp = ((byte >> 3) & 0xf) as u32;
            let mant = (byte & 0x7) as u32;
            if exp == 0 || (exp == 0xf && mant == 7) {
                continue;
            }
            let field = f32::from_bits(((exp + 120) << 23) | (mant << 20));
            assert_eq!(
                e4m3_decode(byte).to_bits(),
                field.to_bits(),
                "byte 0x{byte:02x}"
            );
        }
    }

    #[test]
    fn codes_are_monotonic_so_the_ceil_scan_is_correct() {
        let mut prev = -1.0f32;
        for code in 0u8..=0x7E {
            let v = e4m3_decode(code);
            assert!(v >= prev, "code 0x{code:02x} broke monotonicity");
            prev = v;
        }
    }

    #[test]
    fn encode_is_a_true_ceil() {
        for code in 1u8..=0x7E {
            let v = e4m3_decode(code);
            // Exactly representable -> same code back.
            assert_eq!(e4m3_encode_roundup(v), code, "exact value, code 0x{code:02x}");
            // Just above the previous representable -> rounds UP to this code.
            let below = e4m3_decode(code - 1);
            if below < v {
                let mid = below + (v - below) * 0.5;
                assert_eq!(e4m3_encode_roundup(mid), code, "midpoint below 0x{code:02x}");
            }
        }
    }

    #[test]
    fn roundup_never_undershoots_below_saturation() {
        let mut s = 1e-4f32;
        while s < 448.0 {
            assert!(
                e4m3_decode(e4m3_encode_roundup(s)) >= s,
                "undershot at s={s}"
            );
            s *= 1.037;
        }
    }

    #[test]
    fn above_max_finite_saturates_rather_than_overflowing() {
        // 448 is the largest finite unsigned E4M3. Anything above it clamps to
        // 0x7E and therefore DOES undershoot -- deliberately, since the
        // alternative code is NaN.
        for &s in &[448.5f32, 455.64313, 1e6] {
            assert_eq!(e4m3_encode_roundup(s), 0x7E);
            assert_eq!(e4m3_decode(0x7E), 448.0);
        }
    }

    #[test]
    fn edge_cases() {
        assert_eq!(e4m3_encode_roundup(0.0), 0x00);
        assert_eq!(e4m3_encode_roundup(-1.0), 0x00);
        assert_eq!(e4m3_encode_roundup(f32::NAN), 0x00);
        assert_eq!(e4m3_encode_roundup(1e9), 0x7E);
        assert_eq!(e4m3_decode(0x7F), 448.0, "NaN code decodes to max finite");
        assert_eq!(e4m3_decode(0x00), 0.0);
    }
}
