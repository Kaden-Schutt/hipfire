// SPDX-License-Identifier: see LICENSE
// hipfire — see LICENSE
// Copyright (c) 2026 Kaden Schutt
//
//! Canonical CPU-side half-precision <-> f32 conversions.
//!
//! These pure bit-twiddling helpers were copied verbatim into a dozen call
//! sites (quantizer binaries, runtime, GPU-kernel example harnesses). They are
//! GPU-free, so this is the single shared home reachable by every crate that
//! already depends on `rdna-compute`. IEEE-754 binary16 (half) and bfloat16.

/// Convert IEEE-754 binary16 (half, 5-bit exponent / 10-bit fraction) bits to
/// f32. Handles subnormals, inf, and NaN.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        // Denormalized half -> normalized f32.
        let mut e = 0i32;
        let mut f = frac;
        while f & 0x400 == 0 {
            f <<= 1;
            e -= 1;
        }
        f &= 0x3FF;
        let exp32 = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13));
    }
    if exp == 31 {
        // inf / NaN
        let frac32 = if frac == 0 { 0 } else { frac << 13 | 1 };
        return f32::from_bits((sign << 31) | (0xFF << 23) | frac32);
    }
    let exp32 = exp + 127 - 15;
    f32::from_bits((sign << 31) | (exp32 << 23) | (frac << 13))
}

/// Convert f32 to IEEE-754 binary16 (half) bits. Round-toward-zero on the
/// fraction; overflow -> inf, underflow -> zero (with subnormal range handled).
#[inline]
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;

    if exp == 0xFF {
        // inf / NaN
        let f16_frac = if frac == 0 { 0 } else { (frac >> 13) | 1 };
        return ((sign << 15) | (0x1F << 10) | f16_frac) as u16;
    }

    let new_exp = exp - 127 + 15;

    if new_exp >= 31 {
        return ((sign << 15) | (0x1F << 10)) as u16; // overflow -> inf
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return (sign << 15) as u16; // underflow -> zero
        }
        let f = frac | 0x800000;
        let shift = (1 - new_exp + 13) as u32;
        return ((sign << 15) | (f >> shift)) as u16;
    }

    ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

/// Convert bfloat16 (8-bit exponent / 7-bit fraction, same exponent range as
/// f32) bits to f32. bf16 is simply the top 16 bits of an f32.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Convert f32 to bfloat16 bits by truncating the low 16 bits (round-toward-zero).
#[inline]
pub fn f32_to_bf16(val: f32) -> u16 {
    (val.to_bits() >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip_exact_powers() {
        // Values exactly representable in f16 round-trip losslessly.
        for &v in &[0.0f32, 1.0, -1.0, 0.5, 2.0, -2.0, 0.25, 65504.0, -65504.0] {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            assert_eq!(back, v, "f16 round-trip {v} -> {h:#06x} -> {back}");
        }
    }

    #[test]
    fn f16_special_values() {
        assert!(f16_to_f32(0x7C00).is_infinite()); // +inf
        assert!(f16_to_f32(0xFC00).is_infinite()); // -inf
        assert!(f16_to_f32(0x7E00).is_nan()); // NaN
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), 0.0); // -0.0 == 0.0
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xFC00);
    }

    #[test]
    fn f16_subnormal() {
        // Smallest positive f16 subnormal = 2^-24.
        let h = 0x0001u16;
        let v = f16_to_f32(h);
        assert!((v - 2f32.powi(-24)).abs() < 1e-30, "subnormal {v}");
        assert_eq!(f32_to_f16(v), h, "subnormal round-trip");
    }

    #[test]
    fn bf16_round_trips_high_bits() {
        for &v in &[0.0f32, 1.0, -1.0, 1234.5, -0.001, 3.5e30] {
            let b = f32_to_bf16(v);
            let back = bf16_to_f32(b);
            // bf16 keeps the top 16 bits, so the masked f32 must match.
            let masked = f32::from_bits(v.to_bits() & 0xFFFF_0000);
            assert_eq!(back, masked, "bf16 round-trip {v}");
        }
    }
}
