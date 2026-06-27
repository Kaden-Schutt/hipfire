// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Tensor payload decoders: dequantize HFQ diffusion tensor formats (f16, bf16,
//! f32, Q4F16, Q8F16, Q4_K, HFQ4, HFQ6) into f32, plus f16<->f32 bit helpers.

use super::*;

pub(crate) fn decode_f16_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| f16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect()
}

pub(crate) fn decode_bf16_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| f32::from_bits((u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16))
        .collect()
}

pub(crate) fn decode_q4f16_g64_slice(
    name: &str,
    bytes: &[u8],
    elem_count: usize,
) -> DiffusionResult<Vec<f32>> {
    let expected_blocks = elem_count.div_ceil(64);
    let expected_bytes = expected_blocks * 36;
    if bytes.len() < expected_bytes {
        return Err(DiffusionError::InvalidMetadata(format!(
            "Q4F16_G64 tensor {name:?} has {} bytes but shape requires at least {expected_bytes}",
            bytes.len()
        )));
    }
    let mut out = vec![0.0f32; elem_count];
    for block in 0..expected_blocks {
        let offset = block * 36;
        let scale = f16_bits_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]));
        let min = f16_bits_to_f32(u16::from_le_bytes([bytes[offset + 2], bytes[offset + 3]]));
        for idx in 0..32 {
            let packed = bytes[offset + 4 + idx];
            let lo = (packed & 0x0f) as f32;
            let hi = (packed >> 4) as f32;
            let lo_idx = block * 64 + idx;
            let hi_idx = lo_idx + 32;
            if lo_idx < elem_count {
                out[lo_idx] = min + lo * scale;
            }
            if hi_idx < elem_count {
                out[hi_idx] = min + hi * scale;
            }
        }
    }
    Ok(out)
}

pub(crate) fn decode_q8f16_slice(name: &str, bytes: &[u8], elem_count: usize) -> DiffusionResult<Vec<f32>> {
    let expected_blocks = elem_count.div_ceil(32);
    let expected_bytes = expected_blocks * 34;
    if bytes.len() < expected_bytes {
        return Err(DiffusionError::InvalidMetadata(format!(
            "Q8F16 tensor {name:?} has {} bytes but shape requires at least {expected_bytes}",
            bytes.len()
        )));
    }
    Ok(hipfire_runtime::quant::dequantize_q8_0(bytes, elem_count))
}

pub(crate) fn decode_q4_k_slice(name: &str, bytes: &[u8], elem_count: usize) -> DiffusionResult<Vec<f32>> {
    let expected_blocks = elem_count.div_ceil(256);
    let expected_bytes = expected_blocks * 144;
    if bytes.len() < expected_bytes {
        return Err(DiffusionError::InvalidMetadata(format!(
            "Q4_K tensor {name:?} has {} bytes but shape requires at least {expected_bytes}",
            bytes.len()
        )));
    }
    Ok(hipfire_runtime::quant::dequantize_q4_k(bytes, elem_count))
}

pub(crate) fn decode_hfq4_slice(
    name: &str,
    bytes: &[u8],
    elem_count: usize,
    group_size: usize,
    block_bytes: usize,
    label: &str,
) -> DiffusionResult<Vec<f32>> {
    let expected_blocks = elem_count.div_ceil(group_size);
    let expected_bytes = expected_blocks * block_bytes;
    if bytes.len() < expected_bytes {
        return Err(DiffusionError::InvalidMetadata(format!(
            "{label} tensor {name:?} has {} bytes but shape requires at least {expected_bytes}",
            bytes.len()
        )));
    }
    let mut out = vec![0.0f32; elem_count];
    for block in 0..expected_blocks {
        let offset = block * block_bytes;
        let scale = f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        let min = f32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        for idx in 0..(group_size / 2) {
            let packed = bytes[offset + 8 + idx];
            let lo_idx = block * group_size + idx * 2;
            let hi_idx = lo_idx + 1;
            if lo_idx < elem_count {
                out[lo_idx] = min + (packed & 0x0f) as f32 * scale;
            }
            if hi_idx < elem_count {
                out[hi_idx] = min + (packed >> 4) as f32 * scale;
            }
        }
    }
    Ok(out)
}

pub(crate) fn decode_hfq6_g256_slice(
    name: &str,
    bytes: &[u8],
    elem_count: usize,
) -> DiffusionResult<Vec<f32>> {
    let expected_blocks = elem_count.div_ceil(256);
    let expected_bytes = expected_blocks * 200;
    if bytes.len() < expected_bytes {
        return Err(DiffusionError::InvalidMetadata(format!(
            "HFQ6G256 tensor {name:?} has {} bytes but shape requires at least {expected_bytes}",
            bytes.len()
        )));
    }
    let mut out = vec![0.0f32; elem_count];
    for block in 0..expected_blocks {
        let offset = block * 200;
        let scale = f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        let min = f32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        for i in (0..256).step_by(4) {
            let byte_offset = offset + 8 + (i / 4) * 3;
            let b0 = bytes[byte_offset];
            let b1 = bytes[byte_offset + 1];
            let b2 = bytes[byte_offset + 2];
            let values = [
                b0 & 0x3f,
                ((b0 >> 6) | ((b1 & 0x0f) << 2)) & 0x3f,
                ((b1 >> 4) | ((b2 & 0x03) << 4)) & 0x3f,
                (b2 >> 2) & 0x3f,
            ];
            for (lane, value) in values.into_iter().enumerate() {
                let idx = block * 256 + i + lane;
                if idx < elem_count {
                    out[idx] = min + value as f32 * scale;
                }
            }
        }
    }
    Ok(out)
}

pub(crate) fn decode_f32_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x03ff) as u32;
    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut frac_norm = frac;
            let mut exp_norm = -14i32;
            while (frac_norm & 0x0400) == 0 {
                frac_norm <<= 1;
                exp_norm -= 1;
            }
            frac_norm &= 0x03ff;
            sign | (((exp_norm + 127) as u32) << 23) | (frac_norm << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        sign | (((exp - 15 + 127) as u32) << 23) | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

#[cfg(test)]
pub(crate) fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 255 {
        return sign | if mant == 0 { 0x7c00 } else { 0x7e00 };
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mant = mant | 0x80_0000;
        let shift = (14 - half_exp) as u32;
        let rounded = (mant + (1 << (shift - 1))) >> shift;
        return sign | rounded as u16;
    }
    let rounded = mant + 0x1000;
    sign | ((half_exp as u16) << 10) | ((rounded >> 13) as u16 & 0x03ff)
}
