// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Generic dequantization codecs and half/bf16 ↔ f32 conversions.
//!
//! These are arch-agnostic numeric primitives (GGML-format dequant, IEEE
//! half-float conversion, Q4_K→Q4_F16 transcode). They historically lived in
//! `llama.rs` but are not llama-specific — every arch and the HFQ/GGUF readers
//! use them. Relocated here as part of the de-llama-ify cleanup.

/// Dequantize Q4_0 data to f32.
/// Q4_0 block: 2 bytes (f16 scale) + 16 bytes (32 x 4-bit values)
pub fn dequantize_q4_0(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 32;
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];

    for b in 0..nblocks {
        let block_offset = b * 18; // 2 + 16 bytes per block
        if block_offset + 18 > data.len() {
            break;
        }
        let scale_bytes = [data[block_offset], data[block_offset + 1]];
        let scale = f16_to_f32(u16::from_le_bytes(scale_bytes));

        for j in 0..16 {
            let byte = data[block_offset + 2 + j];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;

            let idx = b * block_size + j * 2;
            if idx < n {
                out[idx] = lo as f32 * scale;
            }
            if idx + 1 < n {
                out[idx + 1] = hi as f32 * scale;
            }
        }
    }
    out
}

/// Dequantize Q8_0 data to f32.
/// Q8_0 block: 2 bytes (f16 scale) + 32 bytes (32 x int8)
pub fn dequantize_q8_0(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 32;
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];

    for b in 0..nblocks {
        let block_offset = b * 34; // 2 + 32 bytes per block
        if block_offset + 34 > data.len() {
            break;
        }
        let scale_bytes = [data[block_offset], data[block_offset + 1]];
        let scale = f16_to_f32(u16::from_le_bytes(scale_bytes));

        for j in 0..32 {
            let idx = b * block_size + j;
            if idx < n {
                let val = data[block_offset + 2 + j] as i8;
                out[idx] = val as f32 * scale;
            }
        }
    }
    out
}

pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        // Denormalized
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
    let exp32 = exp + 127 - 15;
    f32::from_bits((sign << 31) | (exp32 << 23) | (frac << 13))
}

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
        return ((sign << 15) | (0x1F << 10)) as u16; // overflow → inf
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return (sign << 15) as u16; // underflow → zero
        }
        let f = frac | 0x800000;
        let shift = (1 - new_exp + 13) as u32;
        return ((sign << 15) | (f >> shift)) as u16;
    }

    ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

/// Dequantize Q4_K data to f32.
/// Q4_K super-block: 256 elements
///   2 bytes: f16 d (super-block scale)
///   2 bytes: f16 dmin (super-block min)
///   12 bytes: scales/mins for 8 sub-blocks (6 bits each, packed)
///   128 bytes: 256 x 4-bit quantized values
pub fn dequantize_q4_k(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 256;
    let block_bytes = 144; // 2+2+12+128
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];

    for b in 0..nblocks {
        let off = b * block_bytes;
        if off + block_bytes > data.len() {
            break;
        }

        let d = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[off + 2], data[off + 3]]));

        // Unpack scales and mins from 12 bytes (at off+4)
        let sc_data = &data[off + 4..off + 16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        // First 4 sub-blocks: lower 6 bits from bytes 0-3 (scales) and 4-7 (mins)
        for i in 0..4 {
            scales[i] = sc_data[i] & 63;
            mins[i] = sc_data[4 + i] & 63;
        }
        // Next 4 sub-blocks: lower 4 bits from bytes 8-11, upper 2 bits from bytes 0-7
        for i in 0..4 {
            scales[4 + i] = (sc_data[8 + i] & 0xF) | ((sc_data[i] >> 6) << 4);
            mins[4 + i] = (sc_data[8 + i] >> 4) | ((sc_data[4 + i] >> 6) << 4);
        }

        // Dequantize 256 values from 128 bytes of 4-bit data.
        // GGML layout: 4 groups of 64 elements. Each group has 2 sub-blocks
        // sharing 32 bytes: lower nibble → even sub-block, upper nibble → odd.
        let qdata = &data[off + 16..off + 16 + 128];
        for group in 0..4 {
            let sb_even = group * 2;
            let sb_odd = group * 2 + 1;
            let sc_even = d * scales[sb_even] as f32;
            let m_even = dmin * mins[sb_even] as f32;
            let sc_odd = d * scales[sb_odd] as f32;
            let m_odd = dmin * mins[sb_odd] as f32;

            for l in 0..32 {
                let byte = qdata[group * 32 + l];
                let idx_even = b * block_size + group * 64 + l;
                let idx_odd = idx_even + 32;
                if idx_even < n {
                    out[idx_even] = (byte & 0x0F) as f32 * sc_even - m_even;
                }
                if idx_odd < n {
                    out[idx_odd] = ((byte >> 4) & 0x0F) as f32 * sc_odd - m_odd;
                }
            }
        }
    }
    out
}

/// Convert Q4_K raw data to Q4_F16_G64 format.
/// Dequantizes Q4_K to F32 intermediates, then re-quantizes to Q4_F16_G64.
/// Q4_K: 144 bytes per 256 elements → Q4_F16_G64: 4×36=144 bytes per 256 elements (same size).
pub fn convert_q4k_to_q4f16_g64(q4k_data: &[u8], n_elements: usize) -> Vec<u8> {
    let f32_values = dequantize_q4_k(q4k_data, n_elements);

    let group_size = 64;
    let block_bytes = 36;
    let n_blocks = (n_elements + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n_elements);
        let group = &f32_values[start..end];

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
        output[out_off + 2..out_off + 4].copy_from_slice(&f32_to_f16(min_val).to_le_bytes());

        // Pack nibbles: byte[i] = low_nibble(element i) | high_nibble(element i+32)
        let actual_len = end - start;
        for i in 0..32 {
            let lo_val = if i < actual_len { group[i] } else { min_val };
            let hi_val = if 32 + i < actual_len {
                group[32 + i]
            } else {
                min_val
            };

            let lo_q = ((lo_val - min_val) * inv_scale + 0.5) as u8;
            let hi_q = ((hi_val - min_val) * inv_scale + 0.5) as u8;

            output[out_off + 4 + i] = lo_q.min(15) | (hi_q.min(15) << 4);
        }
    }

    output
}

/// Convert Q4_K raw data to Q4_F16_G32 format — nearly lossless.
/// Each Q4_K sub-block (32 elements) maps directly to one Q4_F16_G32 block.
/// Nibbles are preserved exactly; only scale/min are converted to FP16.
/// Q4_K: 144 bytes per 256 elements → Q4_F16_G32: 8×20=160 bytes per 256 elements (11% larger).
pub fn convert_q4k_to_q4f16_g32(q4k_data: &[u8], n_elements: usize) -> Vec<u8> {
    let q4k_block_bytes = 144;
    let q4k_block_elems = 256;
    let g32_block_bytes = 20;
    let nblocks = (n_elements + q4k_block_elems - 1) / q4k_block_elems;
    // 8 sub-blocks per Q4_K super-block → 8 G32 blocks
    let mut output = vec![0u8; nblocks * 8 * g32_block_bytes];

    for b in 0..nblocks {
        let off = b * q4k_block_bytes;
        if off + q4k_block_bytes > q4k_data.len() {
            break;
        }

        let d = f16_to_f32(u16::from_le_bytes([q4k_data[off], q4k_data[off + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([q4k_data[off + 2], q4k_data[off + 3]]));

        let sc_data = &q4k_data[off + 4..off + 16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        for i in 0..4 {
            scales[i] = sc_data[i] & 63;
            mins[i] = sc_data[4 + i] & 63;
        }
        for i in 0..4 {
            scales[4 + i] = (sc_data[8 + i] & 0xF) | ((sc_data[i] >> 6) << 4);
            mins[4 + i] = (sc_data[8 + i] >> 4) | ((sc_data[4 + i] >> 6) << 4);
        }

        let qdata = &q4k_data[off + 16..off + 16 + 128];

        // Each of 4 groups has 2 sub-blocks (even=lower nibble, odd=upper nibble)
        for group in 0..4 {
            let sb_even = group * 2;
            let sb_odd = group * 2 + 1;

            // Sub-block even (elements group*64+0..group*64+31) → G32 block
            let eff_scale_even = d * scales[sb_even] as f32;
            let eff_min_even = -(dmin * mins[sb_even] as f32);
            let out_off_even = (b * 8 + sb_even) * g32_block_bytes;
            output[out_off_even..out_off_even + 2]
                .copy_from_slice(&f32_to_f16(eff_scale_even).to_le_bytes());
            output[out_off_even + 2..out_off_even + 4]
                .copy_from_slice(&f32_to_f16(eff_min_even).to_le_bytes());

            // Sub-block odd (elements group*64+32..group*64+63) → G32 block
            let eff_scale_odd = d * scales[sb_odd] as f32;
            let eff_min_odd = -(dmin * mins[sb_odd] as f32);
            let out_off_odd = (b * 8 + sb_odd) * g32_block_bytes;
            output[out_off_odd..out_off_odd + 2]
                .copy_from_slice(&f32_to_f16(eff_scale_odd).to_le_bytes());
            output[out_off_odd + 2..out_off_odd + 4]
                .copy_from_slice(&f32_to_f16(eff_min_odd).to_le_bytes());

            // Copy nibbles: Q4_K stores them as byte[l] where low=even, high=odd.
            // G32 packing: byte[i] = lo_nibble(elem i) | hi_nibble(elem i+16)
            // Q4_K byte[l] has: elem l in low nibble, elem l+32 in high nibble.
            // For sub-block even: we want the 32 lower nibbles from group*32 bytes.
            // For sub-block odd: we want the 32 upper nibbles.
            // G32 block maps: thread t reads byte[t&15], lo nibble = elem t (t<16), hi nibble = elem t-16+16=t (t>=16)
            // So G32 byte[i] = nibble(elem i) | nibble(elem i+16) << 4
            for i in 0..16 {
                let src_byte_0 = qdata[group * 32 + i];
                let src_byte_1 = qdata[group * 32 + 16 + i];
                // Even sub-block: lower nibbles
                let nib_0 = src_byte_0 & 0xF;
                let nib_1 = src_byte_1 & 0xF;
                output[out_off_even + 4 + i] = nib_0 | (nib_1 << 4);
                // Odd sub-block: upper nibbles
                let nib_2 = src_byte_0 >> 4;
                let nib_3 = src_byte_1 >> 4;
                output[out_off_odd + 4 + i] = nib_2 | (nib_3 << 4);
            }
        }
    }

    output
}

/// Dequantize Q6_K data to f32 (matches GGML reference exactly).
/// Q6_K super-block: 256 elements = 2 groups of 128
///   ql[128]: lower 4 bits (shared between lo/hi nibble pairs)
///   qh[64]: upper 2 bits (packed 4 per byte)
///   scales[16]: int8 scales for sub-groups of 16 elements
///   d: f16 super-block scale
pub fn dequantize_q6_k(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 256;
    let block_bytes = 210; // 128 + 64 + 16 + 2
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];

    for b in 0..nblocks {
        let off = b * block_bytes;
        if off + block_bytes > data.len() {
            break;
        }

        let mut ql = &data[off..off + 128];
        let mut qh = &data[off + 128..off + 192];
        let mut sc = &data[off + 192..off + 208];
        let d = f16_to_f32(u16::from_le_bytes([data[off + 208], data[off + 209]]));

        let base = b * block_size;

        // Process 2 groups of 128 elements each
        for group in 0..2 {
            let y_off = base + group * 128;
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;

                let idx0 = y_off + l;
                let idx1 = y_off + l + 32;
                let idx2 = y_off + l + 64;
                let idx3 = y_off + l + 96;

                if idx0 < n {
                    out[idx0] = d * sc[is] as i8 as f32 * q1 as f32;
                }
                if idx1 < n {
                    out[idx1] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                }
                if idx2 < n {
                    out[idx2] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                }
                if idx3 < n {
                    out[idx3] = d * sc[is + 6] as i8 as f32 * q4 as f32;
                }
            }
            // Advance pointers for next group
            ql = &ql[64..];
            qh = &qh[32..];
            sc = &sc[8..];
        }
    }
    out
}

/// Re-export the canonical on-disk byte-contract so arch loaders can reach it
/// as `hipfire_runtime::quant::QuantType` without each depending on the leaf
/// `hipfire-quant-format` crate directly.
pub use hipfire_quant_format::QuantType;

/// Canonical map from an on-disk HFQ `quant_type` byte to the GPU dispatch
/// [`DType`], for the **pure** formats: ones the loader handles as a plain
/// `upload_raw` + dtype tag with no host-side repack.
///
/// This is the single source of truth that replaces the per-arch
/// `slab_dtype_for_quant` / `dtype_for_quant` copies that drifted across
/// qwen35, minimax, lfm2, gemma3, and qwen2 (each carried a divergent subset —
/// e.g. only qwen35 mapped `31 => Qtip3G256`). Routing every loader through
/// here means a new pure format lands in all arches with one edit.
///
/// Returns `None` for:
/// - unknown codes, and
/// - formats that require a host-side transform before upload (bf16 buffer
///   retag `16`; Opus-Quant arch-repack `33/34/35/37`). Callers keep those
///   transform branches and fall through to this map for the pure cases.
///
/// `k` (the input/column dim) gates the FP4 group-32 formats, which require
/// `k % 256 == 0`.
///
/// Matches on the canonical [`QuantType`] (the shared byte-contract) rather
/// than raw integers, so the on-disk ids stay authoritative in one crate.
pub fn dtype_for_quant_type(qt: u8, k: usize) -> Option<rdna_compute::DType> {
    use hipfire_quant_format::QuantType as Q;
    use rdna_compute::DType;
    Some(match Q::from_code(qt)? {
        Q::F16 => DType::F16,
        Q::Q8F16 => DType::Q8_0,
        Q::HFQ4G256 => DType::HFQ4G256,
        Q::HFQ4G128 => DType::HFQ4G128,
        Q::HFQ6G256 => DType::HFQ6G256,
        Q::HFQ3G256 => DType::HFQ3G256,
        Q::HFQ3G128 => DType::HFQ3G128,
        Q::MQ4G256 => DType::MQ4G256,
        Q::MQ8G256 => DType::MQ8G256,
        Q::MQ6G256 => DType::MQ6G256,
        Q::MQ3G256 => DType::MQ3G256,
        Q::MQ2G256 => DType::MQ2G256,
        Q::MQ2G256Lloyd => DType::MQ2G256Lloyd,
        Q::MQ3G256Lloyd => DType::MQ3G256Lloyd,
        Q::HFP4G32 if k % 256 == 0 => DType::HFP4G32,
        Q::MFP4G32 if k % 256 == 0 => DType::MFP4G32,
        Q::MQ4G256Lloyd => DType::MQ4G256Lloyd,
        Q::Qtip3G256 => DType::Qtip3G256,
        _ => return None,
    })
}
