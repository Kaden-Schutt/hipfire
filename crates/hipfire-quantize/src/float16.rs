//! FP16/BF16 conversion helpers shared by the quantizer binaries.
//!
//! Decoding delegates to the well-tested `half` crate. Encoding deliberately
//! preserves Hipfire's historical truncating conversion: changing it to
//! `half::f16::from_f32` would change emitted HFQ bytes because that API rounds
//! to nearest instead.

use half::{bf16, f16};

#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    bf16::from_bits(bits).to_f32()
}

/// Convert F32 to F16 bits using Hipfire's established truncating semantics.
#[inline]
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;
    if exp == 0xFF {
        let f16_frac = if frac == 0 { 0 } else { (frac >> 13) | 1 };
        return ((sign << 15) | (0x1F << 10) | f16_frac) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | (0x1F << 10)) as u16;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return (sign << 15) as u16;
        }
        let f = frac | 0x800000;
        let shift = (1 - new_exp + 13) as u32;
        return ((sign << 15) | (f >> shift)) as u16;
    }
    ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_decode_roundtrips_every_non_nan_bit_pattern() {
        for bits in 0u16..=u16::MAX {
            let value = f16_to_f32(bits);
            if !value.is_nan() {
                assert_eq!(f16::from_f32(value).to_bits(), bits, "bits=0x{bits:04x}");
            }
        }
    }

    #[test]
    fn encoder_keeps_legacy_truncation() {
        let halfway_plus = f32::from_bits(1.0f32.to_bits() + 0x1fff);
        assert_eq!(f32_to_f16(halfway_plus), 0x3c00);
        assert_eq!(f16::from_f32(halfway_plus).to_bits(), 0x3c01);
    }
}
