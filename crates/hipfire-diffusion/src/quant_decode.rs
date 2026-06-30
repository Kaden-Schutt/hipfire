// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Tensor payload decoders: dequantize HFQ diffusion tensor formats (f16, bf16,
//! f32, Q4F16, Q8F16, Q4_K, HFQ4, HFQ6) into f32, plus f16<->f32 bit helpers.

use super::*;
#[cfg(test)]
pub(crate) use hipfire_primitives::conv::f32_to_f16_bits;
pub(crate) use hipfire_primitives::conv::{
    decode_bf16_slice, decode_f16_slice, decode_f32_slice, f16_bits_to_f32,
};

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

pub(crate) fn decode_q8f16_slice(
    name: &str,
    bytes: &[u8],
    elem_count: usize,
) -> DiffusionResult<Vec<f32>> {
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

pub(crate) fn decode_q4_k_slice(
    name: &str,
    bytes: &[u8],
    elem_count: usize,
) -> DiffusionResult<Vec<f32>> {
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
