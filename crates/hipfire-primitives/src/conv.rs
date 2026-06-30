// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — half-precision <-> f32 bit conversions (no external deps).

/// IEEE binary16 (half) bit pattern → `f32`. Handles subnormals, inf, and NaN.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
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
        let frac32 = if frac == 0 { 0 } else { frac << 13 | 1 };
        return f32::from_bits((sign << 31) | (0xFF << 23) | frac32);
    }
    f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
}

/// `f32` → IEEE binary16 (half) bit pattern. Handles overflow→inf,
/// subnormals, and NaN.
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

/// `f32` → IEEE binary16 (half) bit pattern.
pub fn f32_to_f16_bits(val: f32) -> u16 {
    f32_to_f16(val)
}

/// IEEE binary16 (half) bit pattern → `f32`.
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    f16_to_f32(bits)
}

/// IEEE bfloat16 bit pattern → `f32`.
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// `f32` → IEEE bfloat16 bit pattern, round-to-nearest-even.
pub fn f32_to_bf16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    if (bits >> 23) & 0xFF == 0xFF {
        return (bits >> 16) as u16;
    }
    let bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(bias) >> 16) as u16
}

/// Round an `f32` to bfloat16 precision and return it as `f32`.
pub fn round_f32_to_bf16(val: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(val))
}

/// Decode little-endian IEEE binary16 bytes to f32 values.
pub fn decode_f16_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Decode little-endian IEEE bfloat16 bytes to f32 values.
pub fn decode_bf16_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Decode little-endian IEEE f32 bytes to f32 values.
pub fn decode_f32_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Size in bytes for the plain safetensors dtype names hipfire loaders accept.
pub fn plain_dtype_size(dtype: &str) -> Option<usize> {
    match dtype {
        "F16" | "BF16" => Some(2),
        "F32" => Some(4),
        _ => None,
    }
}

/// Decode plain safetensors-style F16/BF16/F32 bytes to f32 values.
pub fn decode_plain_dtype_to_f32(dtype: &str, bytes: &[u8]) -> Result<Vec<f32>, String> {
    match dtype {
        "F16" => Ok(decode_f16_slice(bytes)),
        "BF16" => Ok(decode_bf16_slice(bytes)),
        "F32" => Ok(decode_f32_slice(bytes)),
        other => Err(format!("unsupported dtype {other:?}")),
    }
}

/// Decode plain safetensors-style F16/BF16/F32 bytes to f32 values, panicking
/// on unsupported dtype. This matches the common bin-tool call-site shape.
pub fn plain_dtype_to_f32(bytes: &[u8], dtype: &str) -> Vec<f32> {
    decode_plain_dtype_to_f32(dtype, bytes).unwrap_or_else(|e| panic!("{e}"))
}

/// Encode f32 values as little-endian IEEE binary16 bytes.
pub fn f32_slice_to_f16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&f32_to_f16(v).to_le_bytes());
    }
    out
}

/// Encode f32 values as little-endian IEEE bfloat16 bytes.
pub fn f32_slice_to_bf16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_exact_halfs() {
        // Values exactly representable in f16 must round-trip f32→f16→f32.
        for &v in &[0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0, 65504.0, -65504.0] {
            assert_eq!(f16_to_f32(f32_to_f16(v)), v, "roundtrip {v}");
        }
    }

    #[test]
    fn specials() {
        assert!(f16_to_f32(f32_to_f16(f32::INFINITY)).is_infinite());
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
        assert_eq!(f16_to_f32(0), 0.0);
    }

    #[test]
    fn bf16_round_ties_to_even() {
        assert_eq!(f32_to_bf16_bits(1.0), 0x3f80);
        assert_eq!(bf16_bits_to_f32(0x3f80), 1.0);

        let halfway_to_odd = f32::from_bits(0x3f80_8000);
        assert_eq!(f32_to_bf16_bits(halfway_to_odd), 0x3f80);

        let halfway_to_even_up = f32::from_bits(0x3f81_8000);
        assert_eq!(f32_to_bf16_bits(halfway_to_even_up), 0x3f82);
    }

    #[test]
    fn plain_dtype_decoders_match_scalar_conversions() {
        let f16 = f32_slice_to_f16_bytes(&[1.0, -2.0]);
        assert_eq!(
            decode_plain_dtype_to_f32("F16", &f16).unwrap(),
            vec![1.0, -2.0]
        );

        let bf16 = f32_slice_to_bf16_bytes(&[1.0, -2.0]);
        assert_eq!(
            decode_plain_dtype_to_f32("BF16", &bf16).unwrap(),
            vec![1.0, -2.0]
        );

        let f32_bytes = [1.25f32.to_le_bytes(), (-3.5f32).to_le_bytes()].concat();
        assert_eq!(
            decode_plain_dtype_to_f32("F32", &f32_bytes).unwrap(),
            vec![1.25, -3.5]
        );
        assert!(decode_plain_dtype_to_f32("I8", &[]).is_err());
    }
}
