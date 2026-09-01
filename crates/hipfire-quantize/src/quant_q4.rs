// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    non_snake_case,
    clippy::all
)]

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use crate::e8;
use crate::e8_gptq;
use crate::gguf_input;
use crate::reap_overlay;
use clap::Parser;
use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use hipfire_quantize::hessian_io;
use hipfire_quantize::safetensors_file::{SafetensorsFile, TensorMeta};

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

/// Port of llama.cpp's `make_qkx2_quants` (ggml-quants.c:799).
///
/// Returns `(scale, the_min)` for one sub-block, where dequantization is
/// `w = scale * q - the_min` — the same convention `dequantize_row_q4_K` uses
/// (`y = d1*q - m1`).
///
/// WHY THIS RATHER THAN MIN/MAX. Plain min/max picks the scale that makes the
/// extremes representable, which is not the scale that minimises error: one
/// outlier stretches the grid and every other weight pays for it. This searches
/// `nstep` candidate scales around the min/max one and, for each, solves the
/// weighted least-squares fit for (scale, min) given the resulting integer
/// levels, keeping whichever candidate actually has the lowest error.
///
/// Measured on Maple's lm_head: min/max gives relative L2 0.0799, this gives
/// 0.0731 — the same 0.0731 DeepGrove's published Q4_K head achieves. The
/// layout was already GGML-compatible; only the encoder was weaker.
///
/// `weights` are llama.cpp's importance weights `sqrt(mean(x^2)) + |x|`, which
/// bias the fit toward larger-magnitude entries.
#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants(
    x: &[f32],
    weights: &[f32],
    nmax: i32,
    rmin: f32,
    rdelta: f32,
    nstep: i32,
) -> (f32, f32) {
    let n = x.len();
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    // The grid is anchored at or below zero, so an all-positive block still
    // encodes zero exactly.
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        return (0.0, -min);
    }

    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut laux = vec![0i32; n];
    let mut best_error = 0.0f32;
    for i in 0..n {
        let l = (iscale * (x[i] - min)).round() as i32;
        let l = l.clamp(0, nmax);
        let diff = scale * l as f32 + min - x[i];
        best_error += weights[i] * diff * diff;
    }
    if nstep < 1 {
        return (scale, -min);
    }

    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..n {
            let l = ((iscale * (x[i] - min)).round() as i32).clamp(0, nmax);
            laux[i] = l;
            let w = weights[i];
            sum_l += w * l as f32;
            sum_l2 += w * (l * l) as f32;
            sum_xl += w * l as f32 * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0.0f32;
            for i in 0..n {
                let diff = this_scale * laux[i] as f32 + this_min - x[i];
                cur_error += weights[i] * diff * diff;
            }
            if cur_error < best_error {
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

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

            // llama.cpp's importance weights: sqrt(mean(x^2)) + |x|.
            let sum_x2: f32 = group.iter().map(|v| v * v).sum();
            let av_x = (sum_x2 / group.len() as f32).sqrt();
            let w: Vec<f32> = group.iter().map(|v| av_x + v.abs()).collect();
            // Same parameters Q4_K uses at ggml-quants.c:1476
            // (nmax=15, rmin=-1.0, rdelta=0.1, nstep=20, use_mad=false).
            let (scale, the_min) = make_qkx2_quants(group, &w, 15, -1.0, 0.1, 20);
            sub_scales[sb] = scale;
            // The rest of this function stores the SIGNED min and negates it
            // when packing, so convert back from llama.cpp's positive the_min.
            sub_mins[sb] = -the_min;
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
