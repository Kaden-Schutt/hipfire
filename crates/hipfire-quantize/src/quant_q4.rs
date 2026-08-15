// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.


#![allow(dead_code, unused_imports, unused_variables, non_snake_case, clippy::all)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::Write;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use clap::Parser;
use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use hipfire_quantize::safetensors_file::{SafetensorsFile, TensorMeta};
use hipfire_quantize::hessian_io;
use crate::e8;
use crate::e8_gptq;
use crate::gguf_input;
use crate::reap_overlay;

// ─── Q4_F16_G64 Quantization ────────────────────────────────────────────────

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
