// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
//! Pure quantization codecs (decomposed from main.rs). Each fn maps f32 weights
//! to packed bytes (or back) with no I/O, globals, or arch awareness. Behavior
//! is locked by the `codec_golden` battery in main.rs — moving a codec here must
//! not change its byte output.

// Helpers still defined in main.rs (crate root); codecs is a descendant module
// so it can reference these private items. They will move here in a later batch.
use crate::{cpu_fwht_256, gen_fwht_signs, f16_to_f32, f32_to_f16};

/// Quantize F32 weights to HFQ3-G256: 3-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][96B packed 3-bit] = 104 bytes per 256 weights (0.406 B/w).
/// Packing: 8 weights × 3 bits = 24 bits = 3 bytes per thread-group.
/// Little-endian bitstream within each 3-byte chunk.
pub(crate) fn quantize_hfq3g256(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 104; // 8 metadata + 96 packed 3-bit
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 7.0 } else { 1.0 }; // 3-bit: 8 levels (0-7)
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        let actual_len = end - start;
        // Pack 256 weights as 32 chunks of 8 weights × 3 bits = 3 bytes each = 96 bytes
        // Matches the GEMV kernel's unpack: tid * 3 byte offset, 8 weights per thread.
        for chunk in 0..32 {
            let ci = chunk * 8; // index into group
            let mut q = [0u8; 8];
            for j in 0..8 {
                let idx = ci + j;
                let val = if idx < actual_len {
                    group[idx]
                } else {
                    min_val
                };
                q[j] = ((val - min_val) * inv_scale + 0.5).clamp(0.0, 7.0) as u8;
            }
            // Pack 8 × 3-bit into 3 bytes (little-endian bitstream)
            // Matches kernel unpack:
            //   q0 = b0 & 7
            //   q1 = (b0 >> 3) & 7
            //   q2 = ((b0 >> 6) | (b1 << 2)) & 7
            //   q3 = (b1 >> 1) & 7
            //   q4 = (b1 >> 4) & 7
            //   q5 = ((b1 >> 7) | (b2 << 1)) & 7
            //   q6 = (b2 >> 2) & 7
            //   q7 = (b2 >> 5) & 7
            let b0 = (q[0] & 7) | ((q[1] & 7) << 3) | ((q[2] & 3) << 6);
            let b1 = ((q[2] >> 2) & 1) | ((q[3] & 7) << 1) | ((q[4] & 7) << 4) | ((q[5] & 1) << 7);
            let b2 = ((q[5] >> 1) & 3) | ((q[6] & 7) << 2) | ((q[7] & 7) << 5);

            let bo = out_off + 8 + chunk * 3;
            output[bo] = b0;
            output[bo + 1] = b1;
            output[bo + 2] = b2;
        }
    }

    output
}

/// Quantize F32 weights to HFQ3-G128: 3-bit with 128-weight groups (finer granularity).
/// Block: [f32 scale][f32 zero][48B packed 3-bit] = 56 bytes per 128 weights (0.4375 B/w).
pub(crate) fn quantize_hfq3g128(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 128;
    let block_bytes = 56; // 8 metadata + 48 packed 3-bit
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 7.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        let actual_len = end - start;
        // 16 chunks of 8 weights × 3 bits = 3 bytes each = 48 bytes
        for chunk in 0..16 {
            let ci = chunk * 8;
            let mut q = [0u8; 8];
            for j in 0..8 {
                let idx = ci + j;
                let val = if idx < actual_len {
                    group[idx]
                } else {
                    min_val
                };
                q[j] = ((val - min_val) * inv_scale + 0.5).clamp(0.0, 7.0) as u8;
            }
            let b0 = (q[0] & 7) | ((q[1] & 7) << 3) | ((q[2] & 3) << 6);
            let b1 = ((q[2] >> 2) & 1) | ((q[3] & 7) << 1) | ((q[4] & 7) << 4) | ((q[5] & 1) << 7);
            let b2 = ((q[5] >> 1) & 3) | ((q[6] & 7) << 2) | ((q[7] & 7) << 5);

            let bo = out_off + 8 + chunk * 3;
            output[bo] = b0;
            output[bo + 1] = b1;
            output[bo + 2] = b2;
        }
    }

    output
}

/// Quantize F32 weights to HFQ2-G256: 2-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][64B packed 2-bit] = 72 bytes per 256 weights (0.281 B/w).
pub(crate) fn quantize_hfq2g256(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 72; // 8 metadata + 64 packed
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 3.0 } else { 1.0 }; // 2-bit: 4 levels (0-3)
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        let actual_len = end - start;
        // Pack 256 weights into 64 bytes (4 per byte at 2-bit)
        for i in 0..64 {
            let mut byte_val = 0u8;
            for j in 0..4 {
                let idx = 4 * i + j;
                let val = if idx < actual_len {
                    group[idx]
                } else {
                    min_val
                };
                let q = ((val - min_val) * inv_scale + 0.5) as u8;
                byte_val |= q.min(3) << (j * 2);
            }
            output[out_off + 8 + i] = byte_val;
        }
    }

    output
}

/// Quantize F32 weights to HFQ2-G128: 2-bit with 128-weight groups (finer granularity).
/// Block: [f32 scale][f32 zero][32B packed 2-bit] = 40 bytes per 128 weights (0.3125 B/w).
pub(crate) fn quantize_hfq2g128(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 128;
    let block_bytes = 40;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 3.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        let actual_len = end - start;
        for i in 0..32 {
            let mut byte_val = 0u8;
            for j in 0..4 {
                let idx = 4 * i + j;
                let val = if idx < actual_len {
                    group[idx]
                } else {
                    min_val
                };
                let q = ((val - min_val) * inv_scale + 0.5) as u8;
                byte_val |= q.min(3) << (j * 2);
            }
            output[out_off + 8 + i] = byte_val;
        }
    }

    output
}

/// Quantize F32 weights to HFQ6-G256: 6-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][192B packed 6-bit] = 200 bytes per 256 weights (0.78125 B/w).
pub(crate) fn quantize_hfq6g256(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 200; // 8 (scale+zero) + 192 (packed 6-bit)
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 63.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        let actual_len = end - start;
        // Pack 4 values per 3 bytes: v0[5:0]|v1[1:0], v1[5:2]|v2[3:0], v2[5:4]|v3[5:0]
        for i in (0..256).step_by(4) {
            let q0 = if i < actual_len {
                ((group[i] - min_val) * inv_scale + 0.5) as u8
            } else {
                0
            };
            let q1 = if i + 1 < actual_len {
                ((group[i + 1] - min_val) * inv_scale + 0.5) as u8
            } else {
                0
            };
            let q2 = if i + 2 < actual_len {
                ((group[i + 2] - min_val) * inv_scale + 0.5) as u8
            } else {
                0
            };
            let q3 = if i + 3 < actual_len {
                ((group[i + 3] - min_val) * inv_scale + 0.5) as u8
            } else {
                0
            };
            let q0 = q0.min(63);
            let q1 = q1.min(63);
            let q2 = q2.min(63);
            let q3 = q3.min(63);

            let byte_off = 8 + (i / 4) * 3;
            output[out_off + byte_off] = q0 | (q1 << 6);
            output[out_off + byte_off + 1] = (q1 >> 2) | (q2 << 4);
            output[out_off + byte_off + 2] = (q2 >> 4) | (q3 << 2);
        }
    }
    output
}

/// Quantize F32 weights to HFQ4-G128: flat 4-bit with 128-weight groups.
/// Block: [f32 scale][f32 zero][64B nibbles] = 72 bytes per 128 weights (0.5625 B/w).
/// 14 VGPRs, 100% occupancy. Better quality for small K dimensions.
pub(crate) fn quantize_hfq4g128(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 128;
    let block_bytes = 72;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        let actual_len = end - start;
        for i in 0..64 {
            let idx_lo = 2 * i;
            let idx_hi = 2 * i + 1;
            let lo_val = if idx_lo < actual_len {
                group[idx_lo]
            } else {
                min_val
            };
            let hi_val = if idx_hi < actual_len {
                group[idx_hi]
            } else {
                min_val
            };

            let lo_q = ((lo_val - min_val) * inv_scale + 0.5) as u8;
            let hi_q = ((hi_val - min_val) * inv_scale + 0.5) as u8;

            output[out_off + 8 + i] = lo_q.min(15) | (hi_q.min(15) << 4);
        }
    }

    output
}

// ─── MQ-family (FWHT-rotated) codecs ───
/// Same binary format as HFQ4-G256 (136 bytes/group) — the rotation is baked
/// into the weights. The GEMV kernel rotates x instead of inverse-rotating w.
pub(crate) fn quantize_mq4g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 136;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);

        // Copy group and pad to 256
        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);

        // Apply FWHT rotation — this equalizes outliers across the group
        cpu_fwht_256(&mut group, signs1, signs2);

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        for i in 0..128 {
            let lo_q = ((group[2 * i] - min_val) * inv_scale + 0.5) as u8;
            let hi_q = ((group[2 * i + 1] - min_val) * inv_scale + 0.5) as u8;
            output[out_off + 8 + i] = lo_q.min(15) | (hi_q.min(15) << 4);
        }
    }

    output
}

/// Dequantize MQ4G256 packed bytes back to f32, EXACTLY mirroring the GEMV
/// kernel (and `quant_quality_mse`'s reference): per 136-byte group read scale+min
/// (f32), expand 128 nibble bytes to 256 values `min + scale*q` (lo=2i, hi=2i+1),
/// then inverse FWHT (signs swapped). Used by the roughquant real format to form
/// the protected-channel correction residual `R = W − dequant(mq4(W))`, so adding
/// `R_S·x_S` to the kernel's mq4 output yields the EXACT bf16 contribution for the
/// protected channels (the kernel and this dequant agree bit-for-bit).
pub(crate) fn dequant_mq4g256(data: &[u8], n: usize, signs1: &[f32], signs2: &[f32]) -> Vec<f32> {
    let group = 256usize;
    let block = 136usize;
    let n_blocks = n.div_ceil(group);
    let mut out = Vec::with_capacity(n_blocks * group);
    for b in 0..n_blocks {
        let off = b * block;
        if off + block > data.len() {
            break;
        }
        let scale = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        let min_val =
            f32::from_le_bytes([data[off + 4], data[off + 5], data[off + 6], data[off + 7]]);
        let mut group_buf = [0.0f32; 256];
        for i in 0..128 {
            let byte = data[off + 8 + i];
            let lo = (byte & 0xF) as f32;
            let hi = (byte >> 4) as f32;
            group_buf[2 * i] = min_val + scale * lo;
            group_buf[2 * i + 1] = min_val + scale * hi;
        }
        // Inverse FWHT: forward op with signs1/signs2 swapped (matches encode).
        cpu_fwht_256(&mut group_buf, signs2, signs1);
        out.extend_from_slice(&group_buf);
    }
    out.truncate(n);
    out
}

/// MagnumQuant MQ6-G256: FWHT-rotated 6-bit quantization.
/// Same binary format as HFQ6-G256 (200 bytes/group) — the rotation is baked
/// into the weights. The GEMV kernel rotates x instead of inverse-rotating w.
pub(crate) fn quantize_mq6g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 200; // 8 (scale+zero) + 192 (packed 6-bit)
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);

        // Copy group and pad to 256
        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);

        // Apply FWHT rotation — this equalizes outliers across the group
        cpu_fwht_256(&mut group, signs1, signs2);

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 63.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        // Pack 4 values per 3 bytes: v0[5:0]|v1[1:0], v1[5:2]|v2[3:0], v2[5:4]|v3[5:0]
        for i in (0..256).step_by(4) {
            let q0 = ((group[i] - min_val) * inv_scale + 0.5) as u8;
            let q1 = ((group[i + 1] - min_val) * inv_scale + 0.5) as u8;
            let q2 = ((group[i + 2] - min_val) * inv_scale + 0.5) as u8;
            let q3 = ((group[i + 3] - min_val) * inv_scale + 0.5) as u8;
            let q0 = q0.min(63);
            let q1 = q1.min(63);
            let q2 = q2.min(63);
            let q3 = q3.min(63);

            let byte_off = 8 + (i / 4) * 3;
            output[out_off + byte_off] = q0 | (q1 << 6);
            output[out_off + byte_off + 1] = (q1 >> 2) | (q2 << 4);
            output[out_off + byte_off + 2] = (q2 >> 4) | (q3 << 2);
        }
    }

    output
}

/// MagnumQuant MQ8-G256: FWHT-rotated symmetric INT8 quantization.
/// Format: [f16 scale][int8 × 256] = 258 bytes per 256 weights (1.008 B/w).
/// Symmetric: scale = max(abs(group)) / 127, q = round(val / scale), no zero-point.
/// Target: dp4a (v_dot4_i32_iu8) on gfx1100 for 4x VALU throughput.
pub(crate) fn quantize_mq8g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 258; // 2 (f16 scale) + 256 (int8 values)
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);

        // Copy and pad to 256
        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);

        // FWHT rotation
        cpu_fwht_256(&mut group, signs1, signs2);

        // Symmetric quantization: scale = max(|val|) / 127
        let amax = group.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        let inv_scale = if amax > 0.0 { 127.0 / amax } else { 0.0 };

        let out_off = b * block_bytes;
        // Store scale as f16 (2 bytes)
        let scale_f16 = f32_to_f16(scale);
        output[out_off] = (scale_f16 & 0xFF) as u8;
        output[out_off + 1] = (scale_f16 >> 8) as u8;

        // Quantize to signed INT8
        for i in 0..256 {
            let q = (group[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
            output[out_off + 2 + i] = q as u8;
        }
    }

    output
}

// ─── HFQ4-G256 + HFP4/MFP4 (FP4) codecs ───
/// MagnumQuant HFQ4-G256: FWHT-rotated 4-bit quantization.

pub(crate) fn quantize_hfq4g256(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 136;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        let actual_len = end - start;
        // Pack 256 weights into 128 bytes of nibbles
        // byte[i] = weight[2*i] (lo nibble) | weight[2*i+1] (hi nibble)
        for i in 0..128 {
            let idx_lo = 2 * i;
            let idx_hi = 2 * i + 1;
            let lo_val = if idx_lo < actual_len {
                group[idx_lo]
            } else {
                min_val
            };
            let hi_val = if idx_hi < actual_len {
                group[idx_hi]
            } else {
                min_val
            };

            let lo_q = ((lo_val - min_val) * inv_scale + 0.5) as u8;
            let hi_q = ((hi_val - min_val) * inv_scale + 0.5) as u8;

            output[out_off + 8 + i] = lo_q.min(15) | (hi_q.min(15) << 4);
        }
    }

    output
}

// ─── HFP4G32 — RDNA-optimal FP4 (E2M1 + UE8M0 g32 + FP16 row scale) ────────────────
//
// Spec: docs/quant-formats/hfp4.md
//
// Per-row layout: 16-B header (row_scale_a:f16, row_scale_b:f16, block_count:u16, flags:u8, ...)
//                 followed by (K/32) blocks × 17 B (UE8M0:u8 + 16 B nibbles).
// Per element:    value = row_scale_a * 2^(block_e - 127) * E2M1_LUT[nibble]

/// OCP E2M1 magnitude lattice (signed 4-bit FP). 16 codes: {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}.
/// Order: positive 0..7, then negative 0..7 (mirrors hardware-canonical sign-magnitude packing).
pub(crate) const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// E2M1 round-to-nearest in the 16-code lattice. Returns the nibble (0..15).
/// Ties broken away from zero (consistent with FP rounding).
pub(crate) fn e2m1_round(x: f32) -> u8 {
    let mut best_idx = 0u8;
    let mut best_err = f32::INFINITY;
    for (i, &code) in E2M1_LUT.iter().enumerate() {
        let err = (code - x).abs();
        // Strict < ensures consistent tie-breaking by code-table order.
        // The lattice has +0 at index 0 and -0 at index 8; +0 wins ties at zero.
        if err < best_err {
            best_err = err;
            best_idx = i as u8;
        }
    }
    best_idx
}

/// Quantize one row of K FP32 weights to HFP4G32 byte format.
///
/// K must be a multiple of 32 (hipfire model dims always satisfy this).
/// Returns 16-B header + (K/32) × 17-B blocks = 16 + 17 * (K/32) bytes.
pub(crate) fn quantize_hfp4g32_row(row: &[f32]) -> Vec<u8> {
    assert!(
        row.len() % 32 == 0,
        "HFP4G32 requires K%32 == 0, got K={}",
        row.len()
    );
    let k = row.len();
    let n_blocks = k / 32;
    let row_bytes = 16 + n_blocks * 17;
    let mut out = vec![0u8; row_bytes];

    // Per-row FP16 second-level scale: row_scale_a = max_abs(row) / 6.0  (E2M1 max = 6.0).
    let row_max_abs = row.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
    let row_scale_a = if row_max_abs > 0.0 {
        row_max_abs / 6.0
    } else {
        1.0
    };
    let inv_row_scale = if row_max_abs > 0.0 {
        1.0 / row_scale_a
    } else {
        0.0
    };

    // Header.
    out[0..2].copy_from_slice(&f32_to_f16(row_scale_a).to_le_bytes());
    out[2..4].copy_from_slice(&0u16.to_le_bytes()); // row_scale_b unused in v1
    out[4..6].copy_from_slice(&(n_blocks as u16).to_le_bytes()); // block_count
    out[6] = 0u8; // format_flags = 0 (no rotation)
    out[7] = 0u8; // reserved
                  // out[8..16] reserved zeros (already zeroed by vec![0u8; ...])

    // Per-block payload.
    for b in 0..n_blocks {
        let block_start = b * 32;
        let block = &row[block_start..block_start + 32];

        // Normalize block by row scale.
        // block_max_normalized in units of [-6.0, +6.0] (because row_scale_a = max_abs/6.0).
        // Pick UE8M0 block exponent so block fits cleanly into E2M1 lattice [-6, +6].
        let block_max_abs = block.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
        let block_max_normalized = block_max_abs * inv_row_scale;

        // Choose smallest UE8M0 exponent that covers block_max_normalized without clipping:
        //   6 * 2^(e - 127) ≥ block_max_normalized   →   e ≥ ceil(log2(block_max_normalized / 6)) + 127
        // ceil (not round) prevents clipping; the precision cost is bounded by 1 bit at the top
        // of the block. Clamp to UE8M0 range [0, 254] (255 = NaN, reserved per OCP spec).
        let block_e: u8 = if block_max_normalized > 0.0 {
            let log_ratio = (block_max_normalized / 6.0).log2();
            let e_signed = log_ratio.ceil() as i32 + 127;
            e_signed.clamp(0, 254) as u8
        } else {
            0u8 // empty block — smallest scale, all nibbles round to 0
        };

        let block_scale = (block_e as i32 - 127) as f32;
        let block_scale_factor = block_scale.exp2(); // 2^(block_e - 127)
        let inv_block_scale = if block_scale_factor > 0.0 {
            1.0 / block_scale_factor
        } else {
            0.0
        };

        // Block payload offset in the row buffer.
        let payload_off = 16 + b * 17;
        out[payload_off] = block_e;

        // Pack 32 elements as 16 bytes, low nibble = even index, high nibble = odd index.
        for i in 0..16 {
            let lo = block[2 * i] * inv_row_scale * inv_block_scale;
            let hi = block[2 * i + 1] * inv_row_scale * inv_block_scale;
            let lo_nibble = e2m1_round(lo);
            let hi_nibble = e2m1_round(hi);
            out[payload_off + 1 + i] = (lo_nibble & 0x0F) | ((hi_nibble & 0x0F) << 4);
        }
    }

    out
}

/// Quantize a row-major 2D weight tensor of shape `[m, k]` to HFP4G32.
/// Returns `m * (16 + 17 * (k/32))` bytes — 16-B row header + per-block payloads, repeated per row.
///
/// K%256 — not K%32 — because the v1 GEMV kernel
/// (`crates/rdna-compute/src/dispatch.rs::gemv_hfp4g32`) iterates 256 elements
/// per work-item and panics on K%256!=0. The byte format itself is K%32-aligned;
/// the K%256 limit is a kernel-side constraint that v2 will lift. Refusing here
/// makes the failure mode "quantize rejects bad input" rather than "runtime
/// panics on first dispatch with a tensor a previous step already accepted."
pub(crate) fn quantize_hfp4g32_2d(f32_data: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(
        f32_data.len(),
        m * k,
        "2D shape mismatch: {} vs {}*{}",
        f32_data.len(),
        m,
        k
    );
    assert!(k % 256 == 0, "HFP4G32 v1 requires K%256==0 (gemv_hfp4g32 kernel constraint; v2 will lift to K%32==0), got K={}", k);
    let row_bytes = 16 + 17 * (k / 32);
    let mut out = Vec::with_capacity(m * row_bytes);
    for r in 0..m {
        let row = &f32_data[r * k..(r + 1) * k];
        out.extend_from_slice(&quantize_hfp4g32_row(row));
    }
    out
}

/// MFP4G32 = HFP4G32 + offline FWHT rotation. Drop-in MQ4 replacement.
///
/// Applies the same per-256-element FWHT as `cpu_fwht_256` (used by MQ4) to the
/// weight matrix before HFP4G32 quantization. Runtime path applies the same
/// FWHT to activations via `mq_rotate_x`, so `dot(rot(W), rot(x)) == dot(W, x)`
/// (the FWHT is orthogonal). K must be a multiple of LCM(32, 256) = 256.
///
/// Sets per-row `format_flags` to `0x05` (bit 0 = rotation present, bits 2-3 = 01
/// = offline FWHT). This is metadata only — the kernel can still consume the
/// row as plain HFP4G32 because the rotation is baked into the codes.
pub(crate) fn quantize_mfp4g32_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    assert_eq!(
        f32_data.len(),
        m * k,
        "2D shape mismatch: {} vs {}*{}",
        f32_data.len(),
        m,
        k
    );
    assert!(
        k % 256 == 0,
        "MFP4G32 requires k % 256 == 0 for 256-element FWHT, got k={}",
        k
    );
    let row_bytes = 16 + 17 * (k / 32);
    let mut out = Vec::with_capacity(m * row_bytes);

    // Rotate one row's worth of weights in-place per 256-element segment, then
    // quantize as HFP4G32 and stamp the rotation flag. Reuses signs1/signs2
    // from the same `gen_fwht_signs(42, 256)` / `gen_fwht_signs(1042, 256)`
    // pair MQ4 ships with so the runtime's mq_rotate_x undoes this rotation.
    let mut row_buf = vec![0.0f32; k];
    for r in 0..m {
        row_buf.copy_from_slice(&f32_data[r * k..(r + 1) * k]);
        // Apply 256-element FWHT to each segment of the row.
        for seg in 0..(k / 256) {
            cpu_fwht_256(&mut row_buf[seg * 256..(seg + 1) * 256], signs1, signs2);
        }
        let mut row_packed = quantize_hfp4g32_row(&row_buf);
        // Stamp format_flags = 0x05 (bit 0 set + bits 2-3 = 01 = offline FWHT).
        row_packed[6] = 0x05;
        out.extend_from_slice(&row_packed);
    }
    out
}

/// CPU reference dequantization for HFP4G32 — bit-exact mirror of `gemv_hfp4g32.hip`'s dequant.
/// Returns the K reconstructed FP32 weights for one row.
#[allow(dead_code)] // used by tests + future round-trip diagnostics
pub(crate) fn dequant_hfp4g32_row(packed: &[u8], k: usize) -> Vec<f32> {
    assert!(k % 32 == 0, "HFP4G32 requires K%32 == 0");
    let n_blocks = k / 32;
    assert_eq!(
        packed.len(),
        16 + n_blocks * 17,
        "HFP4G32 row size mismatch"
    );

    let row_scale_a_bits = u16::from_le_bytes([packed[0], packed[1]]);
    let row_scale_a = f16_to_f32(row_scale_a_bits);

    let mut out = vec![0.0f32; k];
    for b in 0..n_blocks {
        let payload_off = 16 + b * 17;
        let block_e = packed[payload_off] as i32;
        let block_scale = (block_e - 127) as f32;
        let block_scale_factor = block_scale.exp2();
        let scale = row_scale_a * block_scale_factor;

        for i in 0..16 {
            let byte = packed[payload_off + 1 + i];
            let lo_nibble = (byte & 0x0F) as usize;
            let hi_nibble = ((byte >> 4) & 0x0F) as usize;
            out[b * 32 + 2 * i] = scale * E2M1_LUT[lo_nibble];
            out[b * 32 + 2 * i + 1] = scale * E2M1_LUT[hi_nibble];
        }
    }
    out
}

// ─── Q-family codecs (Q4_F16, Q4_K, Q8) ───
/// Quantize F32 weights to Q4_F16_G64 format.
/// Group size 64: 36 bytes per 64 elements (0.5625 bytes/weight).
/// Block: f16 scale (2B) + f16 min (2B) + u8[32] packed nibbles (32B).
pub(crate) fn quantize_q4f16_g64(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 64;
    let block_bytes = 36;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
        output[out_off + 2..out_off + 4].copy_from_slice(&f32_to_f16(min_val).to_le_bytes());

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

// ─── Q4_K Quantization (GGML-compatible) ─────────────────────────────────────

/// Quantize F32 weights to Q4_K format (144 bytes per 256 elements, 0.5625 B/w).
/// GGML-compatible block layout: f16 d + f16 dmin + 12B packed scales + 128B nibbles.
/// This produces blocks that work with the existing gemv_q4k kernel.
pub(crate) fn quantize_q4k(f32_data: &[f32]) -> Vec<u8> {
    let super_block_size = 256;
    let block_bytes = 144;
    let n = f32_data.len();
    let n_blocks = (n + super_block_size - 1) / super_block_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let sb_start = b * super_block_size;
        let sb_end = (sb_start + super_block_size).min(n);
        let out_off = b * block_bytes;

        // Compute per-sub-block scales and mins (8 sub-blocks of 32 elements)
        let mut sub_scales = [0.0f32; 8];
        let mut sub_mins = [0.0f32; 8];

        for sb in 0..8 {
            let start = sb_start + sb * 32;
            let end = (start + 32).min(sb_end);
            if start >= sb_end {
                break;
            }
            let group = &f32_data[start..end];

            let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = max_val - min_val;
            sub_scales[sb] = if range > 0.0 { range / 15.0 } else { 0.0 };
            sub_mins[sb] = min_val;
        }

        // Find super-block d and dmin that best represent the sub-block scales/mins
        // d * scale_int ≈ sub_scale, dmin * min_int ≈ -sub_min (where sub_min is negative offset)
        let max_scale = sub_scales.iter().cloned().fold(0.0f32, f32::max);
        let max_min = sub_mins.iter().map(|m| -m).fold(0.0f32, f32::max); // mins are typically negative

        let d = if max_scale > 0.0 {
            max_scale / 63.0
        } else {
            0.0
        }; // 6-bit scale range
        let dmin = if max_min > 0.0 { max_min / 63.0 } else { 0.0 };

        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
        let inv_dmin = if dmin > 0.0 { 1.0 / dmin } else { 0.0 };

        // Quantize sub-block scales/mins to 6-bit integers
        let mut scale_ints = [0u8; 8];
        let mut min_ints = [0u8; 8];
        for sb in 0..8 {
            scale_ints[sb] = (sub_scales[sb] * inv_d + 0.5).min(63.0) as u8;
            min_ints[sb] = ((-sub_mins[sb]) * inv_dmin + 0.5).min(63.0) as u8;
        }

        // Write super-block header
        output[out_off..out_off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        output[out_off + 2..out_off + 4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());

        // Pack 6-bit scales/mins into 12 bytes (GGML encoding)
        let sc = &mut output[out_off + 4..out_off + 16];
        // First 4 sub-blocks: lower 6 bits in bytes 0-3 (scales) and 4-7 (mins)
        for i in 0..4 {
            sc[i] = (scale_ints[i] & 63) | ((scale_ints[4 + i] >> 4) << 6);
            sc[4 + i] = (min_ints[i] & 63) | ((min_ints[4 + i] >> 4) << 6);
        }
        // Remaining bits in bytes 8-11
        for i in 0..4 {
            sc[8 + i] = (scale_ints[4 + i] & 0xF) | ((min_ints[4 + i] & 0xF) << 4);
        }

        // Quantize and pack nibbles (128 bytes for 256 elements)
        // Layout: 4 groups of 32 bytes. Group g covers elements g*64..g*64+63.
        // Byte l in group g: low nibble = elem g*64+l, high nibble = elem g*64+32+l.
        let qs = &mut output[out_off + 16..out_off + 144];
        for group in 0..4 {
            let sb_even = group * 2;
            let sb_odd = group * 2 + 1;

            let eff_scale_e = d * scale_ints[sb_even] as f32;
            let eff_min_e = dmin * min_ints[sb_even] as f32;
            let inv_se = if eff_scale_e > 0.0 {
                1.0 / eff_scale_e
            } else {
                0.0
            };

            let eff_scale_o = d * scale_ints[sb_odd] as f32;
            let eff_min_o = dmin * min_ints[sb_odd] as f32;
            let inv_so = if eff_scale_o > 0.0 {
                1.0 / eff_scale_o
            } else {
                0.0
            };

            for l in 0..32 {
                let idx_e = sb_start + group * 64 + l;
                let idx_o = sb_start + group * 64 + 32 + l;

                let val_e = if idx_e < sb_end { f32_data[idx_e] } else { 0.0 };
                let val_o = if idx_o < sb_end { f32_data[idx_o] } else { 0.0 };

                let q_e = ((val_e + eff_min_e) * inv_se + 0.5).max(0.0).min(15.0) as u8;
                let q_o = ((val_o + eff_min_o) * inv_so + 0.5).max(0.0).min(15.0) as u8;

                qs[group * 32 + l] = q_e | (q_o << 4);
            }
        }
    }

    output
}

// ─── Q8_FP16 Quantization ────────────────────────────────────────────────────

/// Quantize to Q4-as-Q8: 4-bit precision (range [-8,7]) stored in Q8_0 format.
/// Same storage as Q8 (34 bytes per 32 elements, 1.0625 B/w) but values use only 4 bits.
/// Gets Q8 kernel speed (82% peak BW) with 4-bit quality. Best for VRAM-fitting models.
pub(crate) fn quantize_q4_as_q8(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 32;
    let block_bytes = 34;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let max_abs = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 7.0; // 4-bit symmetric: -8 to 7
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());

        for i in 0..32 {
            let val = if start + i < end { group[i] } else { 0.0 };
            let q = (val * inv_scale).round().max(-8.0).min(7.0) as i8;
            output[out_off + 2 + i] = q as u8;
        }
    }

    output
}

/// Quantize F32 weights to Q8_0 format (compatible with GGML Q8_0).
/// Block: f16 scale (2B) + 32 × int8 = 34 bytes per 32 elements (1.0625 bytes/weight).
/// Symmetric quantization: scale = max(|w|) / 127, q = round(w / scale).
pub(crate) fn quantize_q8f16(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 32;
    let block_bytes = 34;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let max_abs = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 127.0;
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());

        for i in 0..32 {
            let val = if start + i < end { group[i] } else { 0.0 };
            let q = (val * inv_scale).round().max(-128.0).min(127.0) as i8;
            output[out_off + 2 + i] = q as u8;
        }
    }

    output
}

// ─── Q8_HFQ Quantization (Split-Metadata Row Layout) ─────────────────────────

/// Quantize F32 weights to Q8_HFQ format (split-metadata, 128B-aligned rows).
/// Row layout: [f16 scales × n_groups | int8 values × K | padding to 128B].
/// Returns (data, row_stride). Same 1.0625 B/w as Q8_0 for K=2048/4096 (zero padding waste).
pub(crate) fn quantize_q8hfq(f32_data: &[f32], m: usize, k: usize) -> (Vec<u8>, usize) {
    let group_size = 32;
    let n_groups = k / group_size;
    let scales_bytes = n_groups * 2;
    let raw_row = scales_bytes + k;
    let row_stride = (raw_row + 127) & !127; // pad to 128-byte boundary

    let mut output = vec![0u8; m * row_stride];

    for row in 0..m {
        let row_data = &f32_data[row * k..(row + 1) * k];
        let row_out = &mut output[row * row_stride..(row + 1) * row_stride];

        for g in 0..n_groups {
            let start = g * group_size;
            let group = &row_data[start..start + group_size];

            let max_abs = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = max_abs / 127.0;
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

            // Write f16 scale into scale array
            row_out[g * 2..g * 2 + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());

            // Write int8 values into value array (after all scales)
            for i in 0..group_size {
                let q = (group[i] * inv_scale).round().max(-128.0).min(127.0) as i8;
                row_out[scales_bytes + start + i] = q as u8;
            }
        }
    }

    (output, row_stride)
}

// ─── HFQ4-G256 Quantization ─────────────────────────────────────────────────
