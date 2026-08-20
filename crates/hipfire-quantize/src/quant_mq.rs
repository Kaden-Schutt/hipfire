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

use crate::dequant::{e2m1_to_f32, e4m3_to_f32, ue8m0_to_scale};
use crate::quant_fwht::{cpu_fwht_256, gen_fwht_signs};
use crate::quant_hfp4::{e2m1_round, e4m3_scale_decode, e4m3_scale_encode_roundup, E2M1_LUT};

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use crate::e8;
use crate::e8_gptq;
use crate::gguf_input;
use crate::hfq::QuantType;
use crate::pipeline_gguf::GgufFormat;
use crate::reap_overlay;
use clap::Parser;
use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use hipfire_quantize::hessian_io;
use hipfire_quantize::safetensors_file::{SafetensorsFile, TensorMeta};

/// MagnumQuant MQ3-G256: FWHT-rotated 3-bit quantization.
/// Same binary format as HFQ3-G256 (104 bytes/group). Rotation is baked into
/// the weights via cpu_fwht_256; the GEMV kernel rotates x instead.
pub(crate) fn quantize_mq3g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 104;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);

        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);

        // FWHT rotation — equalizes outliers across the group (QuIP#-style RHT)
        cpu_fwht_256(&mut group, signs1, signs2);

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 7.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        // Pack 256 weights as 32 chunks of 8 weights × 3 bits = 3 bytes each.
        // Bit layout matches the HFQ3-G256 GEMV kernel unpack (cross-byte).
        for chunk in 0..32 {
            let ci = chunk * 8;
            let mut q = [0u8; 8];
            for j in 0..8 {
                q[j] = ((group[ci + j] - min_val) * inv_scale + 0.5).clamp(0.0, 7.0) as u8;
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

/// MagnumQuant MQ2-G256: FWHT-rotated 2-bit quantization.
/// Same binary format as HFQ2-G256 (72 bytes/group). Rotation is baked into
/// the weights via cpu_fwht_256; the GEMV kernel rotates x instead.
pub(crate) fn quantize_mq2g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 72;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);

        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);

        cpu_fwht_256(&mut group, signs1, signs2);

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 3.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        // Pack 256 weights into 64 bytes (4 per byte at 2-bit).
        for i in 0..64 {
            let mut byte_val = 0u8;
            for j in 0..4 {
                let q = ((group[4 * i + j] - min_val) * inv_scale + 0.5) as u8;
                byte_val |= q.min(3) << (j * 2);
            }
            output[out_off + 8 + i] = byte_val;
        }
    }

    output
}
pub(crate) const MQ3V2_GROUP_BYTES: usize = 104;
pub(crate) const MQ2V2_GROUP_BYTES: usize = 72;
/// MQ3G256V2 encoder — per-128 asymmetric fp16 header, neutral-size.
///
/// Layout per 256-weight group: `[0..2) fp16 s0,[2..4) fp16 z0,[4..6) fp16 s1,[6..8) fp16 z1,[8..104) 96B packed 3-bit`.
/// Payload unchanged from MQ3G256: 8 values per 3 bytes, little-endian bitstream.
pub(crate) fn quantize_mq3g256v2(
    w: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    assert!(k % 256 == 0, "MQ3G256V2 requires K % 256 == 0, got K={k}");
    let n = w.len();
    assert_eq!(n, m * k, "w.len() {} != m*k {}*{}={}", n, m, k, m * k);
    let gpr = k / 256;
    let total_groups = m * gpr;
    let block_bytes = MQ3V2_GROUP_BYTES;
    let mut output = vec![0u8; total_groups * block_bytes];
    for b in 0..total_groups {
        let start = b * 256;
        let mut group = [0.0f32; 256];
        group.copy_from_slice(&w[start..start + 256]);
        cpu_fwht_256(&mut group, signs1, signs2);
        let mut scales = [0u16; 2];
        let mut zeros = [0u16; 2];
        let mut sts = [0.0f32; 2];
        let mut zs = [0.0f32; 2];
        let mut degenerate = [false; 2];
        for h in 0..2 {
            let off = h * 128;
            let slice = &group[off..off + 128];
            let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let step_f32 = if hi > lo { (hi - lo) / 7.0 } else { 0.0 };
            let mut sc_bits = f32_to_f16(step_f32);
            if hi == lo {
                sc_bits = 0u16;
            }
            let z_bits = f32_to_f16(lo);
            let st = f16_to_f32(sc_bits);
            let z = f16_to_f32(z_bits);
            scales[h] = sc_bits;
            zeros[h] = z_bits;
            sts[h] = st;
            zs[h] = z;
            degenerate[h] = hi == lo || step_f32 == 0.0 || st == 0.0;
        }
        let out_off = b * block_bytes;
        output[out_off..out_off + 2].copy_from_slice(&scales[0].to_le_bytes());
        output[out_off + 2..out_off + 4].copy_from_slice(&zeros[0].to_le_bytes());
        output[out_off + 4..out_off + 6].copy_from_slice(&scales[1].to_le_bytes());
        output[out_off + 6..out_off + 8].copy_from_slice(&zeros[1].to_le_bytes());
        let mut q = [0u8; 256];
        for h in 0..2 {
            let off = h * 128;
            if degenerate[h] {
                for i in 0..128 {
                    q[off + i] = 0;
                }
            } else {
                let st = sts[h];
                let z = zs[h];
                let inv = 1.0 / st;
                for i in 0..128 {
                    let v = group[off + i];
                    let qq = ((v - z) * inv + 0.5).floor().clamp(0.0, 7.0) as u8;
                    q[off + i] = qq;
                }
            }
        }
        for chunk in 0..32 {
            let ci = chunk * 8;
            let mut qq = [0u8; 8];
            for j in 0..8 {
                qq[j] = q[ci + j] & 7;
            }
            let b0 = (qq[0] & 7) | ((qq[1] & 7) << 3) | ((qq[2] & 3) << 6);
            let b1 =
                ((qq[2] >> 2) & 1) | ((qq[3] & 7) << 1) | ((qq[4] & 7) << 4) | ((qq[5] & 1) << 7);
            let b2 = ((qq[5] >> 1) & 3) | ((qq[6] & 7) << 2) | ((qq[7] & 7) << 5);
            let bo = out_off + 8 + chunk * 3;
            output[bo] = b0;
            output[bo + 1] = b1;
            output[bo + 2] = b2;
        }
    }
    output
}
/// MQ2G256V2 encoder — per-128 asymmetric fp16 header, neutral-size.
///
/// Layout per 256-weight group: `[0..2) fp16 s0,[2..4) fp16 z0,[4..6) fp16 s1,[6..8) fp16 z1,[8..72) 64B packed 2-bit`.
/// Payload unchanged from MQ2G256: 4 values per byte.
pub(crate) fn quantize_mq2g256v2(
    w: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    assert!(k % 256 == 0, "MQ2G256V2 requires K % 256 == 0, got K={k}");
    let n = w.len();
    assert_eq!(n, m * k, "w.len() {} != m*k {}*{}={}", n, m, k, m * k);
    let gpr = k / 256;
    let total_groups = m * gpr;
    let block_bytes = MQ2V2_GROUP_BYTES;
    let mut output = vec![0u8; total_groups * block_bytes];
    for b in 0..total_groups {
        let start = b * 256;
        let mut group = [0.0f32; 256];
        group.copy_from_slice(&w[start..start + 256]);
        cpu_fwht_256(&mut group, signs1, signs2);
        let mut scales = [0u16; 2];
        let mut zeros = [0u16; 2];
        let mut sts = [0.0f32; 2];
        let mut zs = [0.0f32; 2];
        let mut degenerate = [false; 2];
        for h in 0..2 {
            let off = h * 128;
            let slice = &group[off..off + 128];
            let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let step_f32 = if hi > lo { (hi - lo) / 3.0 } else { 0.0 };
            let mut sc_bits = f32_to_f16(step_f32);
            if hi == lo {
                sc_bits = 0u16;
            }
            let z_bits = f32_to_f16(lo);
            let st = f16_to_f32(sc_bits);
            let z = f16_to_f32(z_bits);
            scales[h] = sc_bits;
            zeros[h] = z_bits;
            sts[h] = st;
            zs[h] = z;
            degenerate[h] = hi == lo || step_f32 == 0.0 || st == 0.0;
        }
        let out_off = b * block_bytes;
        output[out_off..out_off + 2].copy_from_slice(&scales[0].to_le_bytes());
        output[out_off + 2..out_off + 4].copy_from_slice(&zeros[0].to_le_bytes());
        output[out_off + 4..out_off + 6].copy_from_slice(&scales[1].to_le_bytes());
        output[out_off + 6..out_off + 8].copy_from_slice(&zeros[1].to_le_bytes());
        let mut q = [0u8; 256];
        for h in 0..2 {
            let off = h * 128;
            if degenerate[h] {
                for i in 0..128 {
                    q[off + i] = 0;
                }
            } else {
                let st = sts[h];
                let z = zs[h];
                let inv = 1.0 / st;
                for i in 0..128 {
                    let v = group[off + i];
                    let qq = ((v - z) * inv + 0.5).floor().clamp(0.0, 3.0) as u8;
                    q[off + i] = qq;
                }
            }
        }
        for i in 0..64 {
            let mut byte_val = 0u8;
            for j in 0..4 {
                let qq = q[4 * i + j] & 3;
                byte_val |= qq << (j * 2);
            }
            output[out_off + 8 + i] = byte_val;
        }
    }
    output
}
/// Encode an f32 to IEEE-754 fp16 bits (round-to-nearest-even, no NaN/Inf preservation
/// beyond the trivial case — block centroids are bounded means of fp32 weights so
/// the simple path is safe).
pub(crate) fn f32_to_fp16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xFF) as i32;
    let mant = (bits & 0x7FFFFF) as u32;
    if exp == 0xFF {
        // Inf or NaN
        let m16 = if mant != 0 { 0x200 } else { 0 };
        return sign | 0x7C00 | m16;
    }
    exp -= 127 - 15;
    if exp >= 0x1F {
        return sign | 0x7C00; // overflow → ±Inf
    }
    if exp <= 0 {
        if exp < -10 {
            return sign; // underflow → ±0
        }
        // Subnormal: shift mantissa
        let m = mant | 0x800000;
        let shift = (1 - exp) as u32 + 13;
        let mut m16 = (m >> shift) as u16;
        // Round-half-to-even via remainder
        let lost = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if lost > half || (lost == half && (m16 & 1) == 1) {
            m16 = m16.wrapping_add(1);
        }
        return sign | m16;
    }
    let mut m16 = (mant >> 13) as u16;
    let lost = mant & 0x1FFF;
    if lost > 0x1000 || (lost == 0x1000 && (m16 & 1) == 1) {
        m16 = m16.wrapping_add(1);
        if m16 == 0x400 {
            // Mantissa overflow → carry into exponent
            m16 = 0;
            exp += 1;
            if exp >= 0x1F {
                return sign | 0x7C00;
            }
        }
    }
    sign | ((exp as u16) << 10) | m16
}

/// Lloyd's-algorithm iteration cap, shared by EVERY per-block Lloyd codebook fit
/// (MQ2/MQ3/MQ4, plain / weighted / GPTQ).
///
/// **8, not 16.** History: `f8cd234` (2026-05-19) raised 8 → 16 on the strength of
/// the `lloyd_iteration_headroom` synthetic probe (+0.4–0.9% MSE on heavy-tailed +
/// sparse distributions). On 2026-05-20 a DeepSeek V4 re-quant at 16 iterations
/// measured **60× worse wikitext2 PPL (758 vs 12)** against the byte-identical
/// 8-iter build, and the plain path was reverted. The synthetic probe never
/// captured FWHT-rotated MoE statistics — classic synth-win → prod-falsify.
///
/// The revert only landed on the plain arm. `quantize_mq2g256_lloyd_weighted` and
/// `quantize_mq2g256_lloyd_gptq` were left at 16, each carrying a comment claiming
/// it "matches the plain Lloyd path" — which was false. Any `--imatrix` build
/// therefore silently took the falsified iteration count, confounding every
/// calibration A/B with a known-bad knob. Hoisted to one constant so the three
/// arms cannot drift again.
///
/// Do NOT raise this without first running wikitext2 PPL on a DeepSeek V4 build.
/// Note the 8-vs-16 difference does NOT show up in block MSE (11.19% vs 11.15%) —
/// it is a pathological-local-minimum effect, so MSE is not a valid gate for it.
pub(crate) const LLOYD_MAX_ITER: usize = 8;

/// MagnumQuant HFQ3-G256-Lloyd: per-block 8-entry fp16 codebook fitted via
/// Lloyd's algorithm. 16 B header (8 fp16) + 96 B packed 3-bit indices = 112 B/group
/// (vs uniform MQ3's 104 B — only +7.7% bandwidth). Direct extension of MQ2-Lloyd
/// with K=8; targets sub-9B MQ3 collapse rescue (#114) and 9B MQ3 → MQ4 ppl gap.
pub(crate) fn quantize_mq3g256_lloyd(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    use rayon::prelude::*;
    let group_size = 256;
    let block_bytes = 112;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    output
        .par_chunks_mut(block_bytes)
        .enumerate()
        .for_each(|(b, out_chunk)| {
            let start = b * group_size;
            let end = (start + group_size).min(n);
            let actual_len = end - start;

            let mut group = [0.0f32; 256];
            group[..actual_len].copy_from_slice(&f32_data[start..end]);
            cpu_fwht_256(&mut group, signs1, signs2);

            // Initial centroid placement: 8 evenly-spaced percentiles
            // (1/16, 3/16, ..., 15/16) of the rotated block.
            let mut sorted: [f32; 256] = group;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mut cb: [f32; 8] = [0.0; 8];
            for k in 0..8 {
                let frac = (2 * k + 1) as f32 / 16.0;
                let idx = ((frac * 255.0).round() as usize).min(255);
                cb[k] = sorted[idx];
            }

            let range = sorted[255] - sorted[0];
            let mut indices = [0u8; 256];
            if range > 0.0 {
                let max_iter = 8;
                let mut prev_assignments = [0u8; 256];
                for it in 0..max_iter {
                    let mut sums = [0.0f64; 8];
                    let mut counts = [0u32; 8];
                    let mut changed = 0u32;
                    for i in 0..256 {
                        let w = group[i];
                        let mut best = 0usize;
                        let mut best_d = (w - cb[0]).abs();
                        for k in 1..8 {
                            let d = (w - cb[k]).abs();
                            if d < best_d {
                                best_d = d;
                                best = k;
                            }
                        }
                        if it == 0 || prev_assignments[i] != best as u8 {
                            changed += 1;
                        }
                        prev_assignments[i] = best as u8;
                        indices[i] = best as u8;
                        sums[best] += w as f64;
                        counts[best] += 1;
                    }
                    if it > 0 && changed == 0 {
                        break;
                    }
                    for k in 0..8 {
                        if counts[k] > 0 {
                            cb[k] = (sums[k] / counts[k] as f64) as f32;
                        }
                    }
                }
            }

            // Sort centroids ascending; remap indices.
            let mut order: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
            order.sort_by(|&a, &b| {
                cb[a]
                    .partial_cmp(&cb[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut sorted_cb = [0.0f32; 8];
            let mut inv: [u8; 8] = [0; 8];
            for new_idx in 0..8 {
                sorted_cb[new_idx] = cb[order[new_idx]];
                inv[order[new_idx]] = new_idx as u8;
            }
            for i in 0..256 {
                indices[i] = inv[indices[i] as usize];
            }

            // Header: 8 fp16 centroids = 16 bytes.
            for k in 0..8 {
                let bits = f32_to_fp16_bits(sorted_cb[k]);
                out_chunk[2 * k] = (bits & 0xFF) as u8;
                out_chunk[2 * k + 1] = (bits >> 8) as u8;
            }

            // Data: 96 bytes — same cross-byte 3-bit packing as uniform MQ3, so
            // the kernel unpack code is identical (only the recon changes from
            // `scale*q + zero` to `cb[q]`).
            for chunk in 0..32 {
                let ci = chunk * 8;
                let q = [
                    indices[ci] & 7,
                    indices[ci + 1] & 7,
                    indices[ci + 2] & 7,
                    indices[ci + 3] & 7,
                    indices[ci + 4] & 7,
                    indices[ci + 5] & 7,
                    indices[ci + 6] & 7,
                    indices[ci + 7] & 7,
                ];
                let b0 = q[0] | (q[1] << 3) | ((q[2] & 3) << 6);
                let b1 = (q[2] >> 2) | (q[3] << 1) | (q[4] << 4) | ((q[5] & 1) << 7);
                let b2 = (q[5] >> 1) | (q[6] << 2) | (q[7] << 5);
                let bo = 16 + chunk * 3;
                out_chunk[bo] = b0;
                out_chunk[bo + 1] = b1;
                out_chunk[bo + 2] = b2;
            }
        });

    output
}

/// MagnumQuant HFQ4-G256-Lloyd: per-block 16-entry fp16 codebook fitted via
/// Lloyd's algorithm. 32 B header (16 fp16) + 128 B packed 4-bit indices =
/// 160 B/group (vs uniform MQ4's 136 B — +17.6% bandwidth). Direct extension
/// of MQ3-Lloyd with K=16; the conjecture (from
/// `benchmarks/results/devlog_20260506_lloyd_mq4_extension.md`) is that the
/// 16-centroid placement narrows the MQ4 → MQ6 ppl gap at lower bandwidth
/// than uniform MQ6 (200 B/group).
pub(crate) fn quantize_mq4g256_lloyd(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    use rayon::prelude::*;
    let group_size = 256;
    let block_bytes = 160;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    output
        .par_chunks_mut(block_bytes)
        .enumerate()
        .for_each(|(b, out_chunk)| {
            let start = b * group_size;
            let end = (start + group_size).min(n);
            let actual_len = end - start;

            let mut group = [0.0f32; 256];
            group[..actual_len].copy_from_slice(&f32_data[start..end]);
            cpu_fwht_256(&mut group, signs1, signs2);

            // Initial centroid placement: 16 evenly-spaced percentiles
            // (1/32, 3/32, ..., 31/32) of the rotated block.
            let mut sorted: [f32; 256] = group;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mut cb: [f32; 16] = [0.0; 16];
            for k in 0..16 {
                let frac = (2 * k + 1) as f32 / 32.0;
                let idx = ((frac * 255.0).round() as usize).min(255);
                cb[k] = sorted[idx];
            }

            let range = sorted[255] - sorted[0];
            let mut indices = [0u8; 256];
            if range > 0.0 {
                let max_iter = 8;
                let mut prev_assignments = [0u8; 256];
                for it in 0..max_iter {
                    let mut sums = [0.0f64; 16];
                    let mut counts = [0u32; 16];
                    let mut changed = 0u32;
                    for i in 0..256 {
                        let w = group[i];
                        let mut best = 0usize;
                        let mut best_d = (w - cb[0]).abs();
                        for k in 1..16 {
                            let d = (w - cb[k]).abs();
                            if d < best_d {
                                best_d = d;
                                best = k;
                            }
                        }
                        if it == 0 || prev_assignments[i] != best as u8 {
                            changed += 1;
                        }
                        prev_assignments[i] = best as u8;
                        indices[i] = best as u8;
                        sums[best] += w as f64;
                        counts[best] += 1;
                    }
                    if it > 0 && changed == 0 {
                        break;
                    }
                    for k in 0..16 {
                        if counts[k] > 0 {
                            cb[k] = (sums[k] / counts[k] as f64) as f32;
                        }
                    }
                }
            }

            // Sort centroids ascending; remap indices.
            let mut order: [usize; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
            order.sort_by(|&a, &b| {
                cb[a]
                    .partial_cmp(&cb[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut sorted_cb = [0.0f32; 16];
            let mut inv: [u8; 16] = [0; 16];
            for new_idx in 0..16 {
                sorted_cb[new_idx] = cb[order[new_idx]];
                inv[order[new_idx]] = new_idx as u8;
            }
            for i in 0..256 {
                indices[i] = inv[indices[i] as usize];
            }

            // Header: 16 fp16 centroids = 32 bytes.
            for k in 0..16 {
                let bits = f32_to_fp16_bits(sorted_cb[k]);
                out_chunk[2 * k] = (bits & 0xFF) as u8;
                out_chunk[2 * k + 1] = (bits >> 8) as u8;
            }

            // Data: 128 bytes — same nibble packing as uniform MQ4
            // (low nibble = idx[2i], high nibble = idx[2i+1]) so kernel
            // unpack code is identical; only the recon changes from
            // `min + scale*q` to `cb[q]`.
            for i in 0..128 {
                let lo = indices[2 * i] & 0x0F;
                let hi = indices[2 * i + 1] & 0x0F;
                out_chunk[32 + i] = lo | (hi << 4);
            }
        });

    output
}

/// MagnumQuant HFQ2-G256-Lloyd: per-block 4-entry fp16 codebook fitted via
/// Lloyd's algorithm to minimize squared reconstruction error on FWHT-rotated
/// weights. 8 B header (4 fp16) + 64 B packed 2-bit indices = 72 B/group —
/// bandwidth-identical to uniform MQ2. The "true non-uniform 4-entry codebook"
/// described in `docs/plans/mq-sub4bit-research-queue.md` Q1.
/// Map a safetensors parent tensor name to the corresponding llama.cpp
/// imatrix tensor base name. Returns None if the safetensors tensor isn't
/// one of the routed-expert MoE tensors we have imatrix data for.
///
/// Examples:
///   `model.language_model.layers.0.mlp.experts.gate_up_proj`
///     → Some(("blk.0.ffn_gate_exps.weight", 0))
///   `model.language_model.layers.7.mlp.experts.down_proj`
///     → Some(("blk.7.ffn_down_exps.weight", 7))
pub(crate) fn safetensors_to_imatrix_key(parent: &str) -> Option<(String, usize)> {
    // Expected pattern: model.language_model.layers.{N}.mlp.experts.{gate_up_proj|down_proj}
    let suffix_gate = ".mlp.experts.gate_up_proj";
    let suffix_down = ".mlp.experts.down_proj";
    let (prefix, kind) = if let Some(p) = parent.strip_suffix(suffix_gate) {
        (p, "ffn_gate_exps")
    } else if let Some(p) = parent.strip_suffix(suffix_down) {
        (p, "ffn_down_exps")
    } else {
        return None;
    };
    // Extract layer N from "...layers.{N}".
    let layer_marker = ".layers.";
    let layer_idx_start = prefix.rfind(layer_marker)? + layer_marker.len();
    let layer_str = &prefix[layer_idx_start..];
    let n: usize = layer_str.parse().ok()?;
    Some((format!("blk.{}.{}.weight", n, kind), n))
}

/// Pull per-expert column-weights from an imatrix GGUF for a given
/// MoE-expert parent tensor (e.g. `...experts.gate_up_proj`). Returns
/// `Some(per_expert_col_weights)` where the outer Vec has `n_experts`
/// entries, each an inner Vec of length K with `sqrt(in_sum2[j] / counts)`
/// (the per-column importance scale).
///
/// Returns None when the parent doesn't map to a known imatrix key, or
/// the tensor isn't present in the imatrix.
pub(crate) fn imatrix_col_weights_for_parent(
    gguf: &gguf_input::GgufFile,
    parent: &str,
    n_experts: usize,
) -> Option<Vec<Vec<f32>>> {
    let (base_key, _layer) = safetensors_to_imatrix_key(parent)?;
    let in_sum2_name = format!("{}.in_sum2", base_key);
    let counts_name = format!("{}.counts", base_key);
    let in_sum2 = gguf.tensors.iter().find(|t| t.name == in_sum2_name)?;
    let counts = gguf.tensors.iter().find(|t| t.name == counts_name)?;
    // Shape: in_sum2 is [K, n_experts] (GGUF column-major-ish: shape[0]=K is innermost).
    if in_sum2.shape.len() != 2 || counts.shape.len() != 2 {
        return None;
    }
    let k = in_sum2.shape[0];
    let n_exp = in_sum2.shape[1];
    if n_exp != n_experts {
        eprintln!(
            "  imatrix: {} n_experts mismatch ({} vs {})",
            in_sum2_name, n_exp, n_experts
        );
        return None;
    }
    let in_sum2_bytes = gguf.tensor_data(in_sum2);
    let counts_bytes = gguf.tensor_data(counts);
    let in_sum2_flat: Vec<f32> = in_sum2_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let counts_flat: Vec<f32> = counts_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if in_sum2_flat.len() != k * n_exp || counts_flat.len() != n_exp {
        eprintln!("  imatrix: {} length mismatch", in_sum2_name);
        return None;
    }
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(n_exp);
    for e in 0..n_exp {
        let count = counts_flat[e].max(1.0);
        let offset = e * k;
        let mut col_w: Vec<f32> = Vec::with_capacity(k);
        for j in 0..k {
            // in_sum2 stores SUM of x_j² over `count` activations; mean is
            // in_sum2/count. Take sqrt for the per-column importance scale
            // (matches the C-norm used by GPTQ / Hessian-diagonal methods).
            col_w.push((in_sum2_flat[offset + j] / count).sqrt());
        }
        out.push(col_w);
    }
    Some(out)
}

/// Returns the per-expert routing COUNT vector for a 3D MoE expert parent
/// tensor (e.g. `...mlp.experts.gate_up_proj`). The imatrix GGUF stores a
/// `{base_key}.counts` tensor of shape `[1, n_experts]` whose element `e` is
/// the number of tokens routed to expert `e` during calibration. Used by the
/// graded per-expert mixed-precision path (HIPFIRE_MOE_GRADED) to rank
/// experts hot→cold within each layer. Returns `None` when the tensor is
/// missing or shaped unexpectedly.
pub(crate) fn imatrix_expert_counts_for_parent(
    gguf: &gguf_input::GgufFile,
    parent: &str,
    n_experts: usize,
) -> Option<Vec<f32>> {
    let (base_key, _layer) = safetensors_to_imatrix_key(parent)?;
    let counts_name = format!("{}.counts", base_key);
    let counts = gguf.tensors.iter().find(|t| t.name == counts_name)?;
    // Shape is [1, n_experts] (2D); element e = routing count for expert e.
    if counts.shape.len() != 2 {
        return None;
    }
    let n_exp = counts.shape[1];
    if n_exp != n_experts {
        eprintln!(
            "  imatrix(counts): {} n_experts mismatch ({} vs {})",
            counts_name, n_exp, n_experts
        );
        return None;
    }
    let counts_bytes = gguf.tensor_data(counts);
    let counts_flat: Vec<f32> = counts_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if counts_flat.len() != n_exp {
        eprintln!("  imatrix(counts): {} length mismatch", counts_name);
        return None;
    }
    Some(counts_flat)
}

/// Like `imatrix_col_weights_for_parent` but returns the RAW per-expert
/// `in_sum2[K]` (not `sqrt(in_sum2/count)`). AWQ's `compute_awq_scales` takes
/// raw in_sum2 — it applies `^(alpha/2)` internally (≡ `rms_act^alpha` after
/// geo-mean normalization), so feeding it rms_act would halve the effective
/// alpha vs the dense AWQ path. Used by the per-expert AWQ branch (Route A).
pub(crate) fn imatrix_in_sum2_for_parent(
    gguf: &gguf_input::GgufFile,
    parent: &str,
    n_experts: usize,
) -> Option<Vec<Vec<f32>>> {
    let (base_key, _layer) = safetensors_to_imatrix_key(parent)?;
    let in_sum2_name = format!("{}.in_sum2", base_key);
    let in_sum2 = gguf.tensors.iter().find(|t| t.name == in_sum2_name)?;
    if in_sum2.shape.len() != 2 {
        return None;
    }
    let k = in_sum2.shape[0];
    let n_exp = in_sum2.shape[1];
    if n_exp != n_experts {
        eprintln!(
            "  imatrix(awq): {} n_experts mismatch ({} vs {})",
            in_sum2_name, n_exp, n_experts
        );
        return None;
    }
    let in_sum2_flat: Vec<f32> = gguf
        .tensor_data(in_sum2)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if in_sum2_flat.len() != k * n_exp {
        return None;
    }
    Some(
        (0..n_exp)
            .map(|e| in_sum2_flat[e * k..(e + 1) * k].to_vec())
            .collect(),
    )
}

/// Per-layer "importance score" from an imatrix GGUF, used by Phase 5
/// tiered MQ-Lloyd to rank routed-expert layers.
///
/// Importance proxy: **mean activation magnitude per expert** =
/// `sum(in_sum2) / sum(counts)`. The mean (not sum) is the right
/// per-layer comparator because `counts` is approximately constant
/// across layers in a typical imatrix calibration (every layer sees
/// the same total tokens). Per-expert mean activation magnitude varies
/// substantially because different layers operate at different
/// activation scales.
///
/// Returns `None` if the imatrix doesn't have ffn_gate_exps tensors
/// (non-MoE imatrix). Returns a Vec<f64> of length n_layers; layers
/// not present get f64::NAN.
pub(crate) fn imatrix_layer_activation_counts(
    gguf: &gguf_input::GgufFile,
    n_layers: usize,
) -> Option<Vec<f64>> {
    let mut out = vec![f64::NAN; n_layers];
    let mut found_any = false;
    for n in 0..n_layers {
        let in_sum2_name = format!("blk.{}.ffn_gate_exps.weight.in_sum2", n);
        let counts_name = format!("blk.{}.ffn_gate_exps.weight.counts", n);
        let sum2 = gguf.tensors.iter().find(|t| t.name == in_sum2_name);
        let cts = gguf.tensors.iter().find(|t| t.name == counts_name);
        if let (Some(s2), Some(c)) = (sum2, cts) {
            let s2_bytes = gguf.tensor_data(s2);
            let c_bytes = gguf.tensor_data(c);
            let sum2_total: f64 = s2_bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
                .sum();
            let counts_total: f64 = c_bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
                .sum();
            if counts_total > 0.0 {
                // mean activation magnitude per K-column per expert in this layer
                out[n] = sum2_total / counts_total;
                found_any = true;
            }
        }
    }
    if found_any {
        Some(out)
    } else {
        None
    }
}

/// Imatrix-weighted MQ2-Lloyd quantization. Per-column importance weights
/// from a calibration imatrix shift the Lloyd codebook centroids toward
/// values that minimize the IMPORTANCE-WEIGHTED MSE rather than uniform
/// MSE. Helps preserve precision on high-activation columns.
///
/// Mathematical caveat: the FWHT rotation mixes columns within a block, so
/// per-position weighting in the rotated domain is not exactly equivalent
/// to per-column weighting in the original domain (off-diagonal terms in
/// the rotated Hessian are non-zero). This is a first-order approximation:
/// it tilts centroid choice toward high-importance positions but misses
/// the cross-column coupling that a proper GPTQ-LDLQ solve would capture.
///
/// `col_weights` is shape [K] (per-original-column importance values, e.g.
/// sqrt(E[x²]) from an imatrix). For each 256-weight block at offset b in
/// `f32_data` row-major, the relevant slice is
/// `col_weights[(b % blocks_per_row) * 256 .. + 256]`.
pub(crate) fn quantize_mq2g256_lloyd_weighted(
    f32_data: &[f32],
    col_weights: &[f32],
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    use rayon::prelude::*;
    let group_size = 256;
    let block_bytes = 72;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let blocks_per_row = col_weights.len() / group_size;
    assert!(blocks_per_row > 0, "col_weights too short");
    let mut output = vec![0u8; n_blocks * block_bytes];

    output
        .par_chunks_mut(block_bytes)
        .enumerate()
        .for_each(|(b, out_chunk)| {
            let start = b * group_size;
            let end = (start + group_size).min(n);
            let actual_len = end - start;

            let mut group = [0.0f32; 256];
            group[..actual_len].copy_from_slice(&f32_data[start..end]);
            cpu_fwht_256(&mut group, signs1, signs2);

            // Per-position weights for this block — from the matching column
            // slice of the importance vector. (See caveat above re: FWHT.)
            let col_off = (b % blocks_per_row) * group_size;
            let block_w: &[f32] = &col_weights[col_off..col_off + group_size];

            let mut sorted: [f32; 256] = group;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let percentile = |frac: f32| -> f32 {
                let idx = ((frac * 255.0).round() as usize).min(255);
                sorted[idx]
            };
            let mut cb: [f32; 4] = [
                percentile(0.125),
                percentile(0.375),
                percentile(0.625),
                percentile(0.875),
            ];

            let range = sorted[255] - sorted[0];
            let mut indices = [0u8; 256];
            if range > 0.0 {
                // Shared with the plain + GPTQ arms — see LLOYD_MAX_ITER. This
                // arm ran 16 until 2026-08-04 while claiming to match the plain
                // path, which had been reverted to 8 on 2026-05-20.
                let max_iter = LLOYD_MAX_ITER;
                let mut prev_assignments = [0u8; 256];
                for it in 0..max_iter {
                    // Weighted centroid update: cb[k] = sum_{i in k} w_i * v_i / sum_{i in k} w_i.
                    // (The assignment step is UNWEIGHTED — w_i is a per-point
                    // scalar that cancels from argmin_k |v_i - cb[k]|²; only
                    // the centroid update changes from uniform Lloyd.)
                    let mut weighted_sums = [0.0f64; 4];
                    let mut weight_totals = [0.0f64; 4];
                    let mut changed = 0u32;
                    for i in 0..256 {
                        let w = group[i];
                        let mut best = 0usize;
                        let mut best_d = (w - cb[0]).abs();
                        for k in 1..4 {
                            let d = (w - cb[k]).abs();
                            if d < best_d {
                                best_d = d;
                                best = k;
                            }
                        }
                        if it == 0 || prev_assignments[i] != best as u8 {
                            changed += 1;
                        }
                        prev_assignments[i] = best as u8;
                        indices[i] = best as u8;
                        let pw = block_w[i] as f64;
                        weighted_sums[best] += pw * w as f64;
                        weight_totals[best] += pw;
                    }
                    if it > 0 && changed == 0 {
                        break;
                    }
                    for k in 0..4 {
                        if weight_totals[k] > 0.0 {
                            cb[k] = (weighted_sums[k] / weight_totals[k]) as f32;
                        }
                    }
                }
            }

            // Sort centroids ascending (canonical header).
            let mut order: [usize; 4] = [0, 1, 2, 3];
            order.sort_by(|&a, &b| {
                cb[a]
                    .partial_cmp(&cb[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut sorted_cb = [0.0f32; 4];
            let mut inv: [u8; 4] = [0; 4];
            for new_idx in 0..4 {
                sorted_cb[new_idx] = cb[order[new_idx]];
                inv[order[new_idx]] = new_idx as u8;
            }
            for i in 0..256 {
                indices[i] = inv[indices[i] as usize];
            }

            for k in 0..4 {
                let bits = f32_to_fp16_bits(sorted_cb[k]);
                out_chunk[2 * k] = (bits & 0xFF) as u8;
                out_chunk[2 * k + 1] = (bits >> 8) as u8;
            }
            for i in 0..64 {
                let mut byte_val = 0u8;
                for j in 0..4 {
                    byte_val |= (indices[4 * i + j] & 0x3) << (j * 2);
                }
                out_chunk[8 + i] = byte_val;
            }
        });

    output
}

/// Sequential-error-feedback MQ2-Lloyd. Simplified GPTQ-style quant: for
/// each 256-block, fit the Lloyd codebook normally, then quantize columns
/// LEFT-TO-RIGHT with the residual quantization error propagated into
/// the next column's target. Captures the "compensate for past errors"
/// insight of GPTQ-LDLQ without the full Cholesky-of-Hessian solve.
///
/// Mathematical caveat: true LDLQ would use the rotated Hessian
/// `R·diag(c)·R^T` to compute the precise per-column propagation weights.
/// This implementation uses pure forward-propagation (no decay, no off-
/// diagonal Hessian) — a first-order approximation that empirically
/// recovers most of LDLQ's benefit at a fraction of the cost. Per-
/// position imatrix weighting still drives the underlying Lloyd
/// codebook fit.
///
/// Empirical sweep (Qwen3.6-35B-A3B, mq2lloyd_coherence_harness.py,
/// all-MQ2-GPTQ recipe, greedy decode): damping=0.8 lands at 9 ok /
/// 1 warn / 0 fail on the 10-prompt coherence battery — best in the
/// [0.3, 1.0] sweep. See commit history for full bench numbers.
pub(crate) fn quantize_mq2g256_lloyd_gptq(
    f32_data: &[f32],
    col_weights: &[f32],
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    let damping: f32 = hipfire_config::developer_var("HIPFIRE_GPTQ_DAMPING")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.0);
    quantize_mq2g256_lloyd_gptq_with_damping(f32_data, col_weights, signs1, signs2, damping)
}

pub(crate) fn quantize_mq2g256_lloyd_gptq_with_damping(
    f32_data: &[f32],
    col_weights: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    damping: f32,
) -> Vec<u8> {
    use rayon::prelude::*;
    let group_size = 256;
    let block_bytes = 72;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let blocks_per_row = col_weights.len() / group_size;
    assert!(blocks_per_row > 0, "col_weights too short");
    let mut output = vec![0u8; n_blocks * block_bytes];

    // Tunable: forward-propagation damping.
    //
    // 2026-05-19 update — damping default changed to 0.0 (was 0.8) after
    // the gptq_damping_probe synthetic-data sweep showed monotonic MSE
    // regression at every d>0, on every tested distribution including
    // strongly-correlated AR(1) inputs (decay=0.9). The Qwen3.6-35B-A3B
    // sweep below historically picked d=0.8 because the model was
    // quantized with a REAL imatrix file → the imatrix-weighted codebook
    // fit step paid for the noise the sequential pass injects. On models
    // built with unit imatrix (DeepSeek V4 all-MQ2-GPTQ), the codebook fit
    // degenerates to plain Lloyd and the sequential pass contributes ONLY
    // noise — DeepSeek V4 mq2-gptq-all.hfq measured 1.9-3.3x worse PPL than
    // mq2lloyd on wikitext2-test as a direct consequence. See
    // project_gptq_lloyd_pretendgptq_finding memory + the probe results.
    //
    //   d=0.3 → PPL 12.24 | 7 ok / 3 warn — fails fibonacci_c (Qwen3.6)
    //   d=0.5 → PPL 12.84 | 6 ok / 4 warn (Qwen3.6)
    //   d=0.8 → PPL 14.66 | 9 ok / 1 warn — passes fibonacci_c (Qwen3.6)
    //   d=1.0 → PPL 18.28 | 9 ok / 1 warn (Qwen3.6)
    //
    // At d=0 the sequential pass is a no-op and the function is byte-
    // identical to quantize_mq2g256_lloyd_weighted (which is the right
    // thing to use directly if you don't need the GPTQ name in the
    // pipeline log). Override with `[developer] gptq_damping = 0.8`.
    if damping > 0.0 {
        let has_real_imatrix = col_weights.iter().any(|&w| (w - 1.0).abs() > 1e-6);
        if !has_real_imatrix {
            eprintln!(
                "warning: developer.gptq_damping={damping} with unit imatrix → \
                 strictly worse than plain Lloyd (see gptq_damping_probe). \
                 Either provide --imatrix or use --format mq4-mq2lloyd-native."
            );
        }
    }

    output
        .par_chunks_mut(block_bytes)
        .enumerate()
        .for_each(|(b, out_chunk)| {
            let start = b * group_size;
            let end = (start + group_size).min(n);
            let actual_len = end - start;

            let mut group = [0.0f32; 256];
            group[..actual_len].copy_from_slice(&f32_data[start..end]);
            cpu_fwht_256(&mut group, signs1, signs2);

            let col_off = (b % blocks_per_row) * group_size;
            let block_w: &[f32] = &col_weights[col_off..col_off + group_size];

            // Step 1: Lloyd codebook fit (imatrix-weighted, same as
            // `quantize_mq2g256_lloyd_weighted`). Used to seed the 4
            // centroids before sequential assignment.
            let mut sorted: [f32; 256] = group;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let percentile = |frac: f32| -> f32 {
                let idx = ((frac * 255.0).round() as usize).min(255);
                sorted[idx]
            };
            let mut cb: [f32; 4] = [
                percentile(0.125),
                percentile(0.375),
                percentile(0.625),
                percentile(0.875),
            ];
            let range = sorted[255] - sorted[0];
            if range > 0.0 {
                // Shared with the plain + weighted arms — see LLOYD_MAX_ITER.
                let max_iter = LLOYD_MAX_ITER;
                let mut prev_assignments = [0u8; 256];
                for it in 0..max_iter {
                    let mut weighted_sums = [0.0f64; 4];
                    let mut weight_totals = [0.0f64; 4];
                    let mut changed = 0u32;
                    for i in 0..256 {
                        let w = group[i];
                        let mut best = 0usize;
                        let mut best_d = (w - cb[0]).abs();
                        for k in 1..4 {
                            let d = (w - cb[k]).abs();
                            if d < best_d {
                                best_d = d;
                                best = k;
                            }
                        }
                        if it == 0 || prev_assignments[i] != best as u8 {
                            changed += 1;
                        }
                        prev_assignments[i] = best as u8;
                        let pw = block_w[i] as f64;
                        weighted_sums[best] += pw * w as f64;
                        weight_totals[best] += pw;
                    }
                    if it > 0 && changed == 0 {
                        break;
                    }
                    for k in 0..4 {
                        if weight_totals[k] > 0.0 {
                            cb[k] = (weighted_sums[k] / weight_totals[k]) as f32;
                        }
                    }
                }
            }

            // Sort centroids ascending (canonical header).
            let mut order: [usize; 4] = [0, 1, 2, 3];
            order.sort_by(|&a, &b| {
                cb[a]
                    .partial_cmp(&cb[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut sorted_cb = [0.0f32; 4];
            for new_idx in 0..4 {
                sorted_cb[new_idx] = cb[order[new_idx]];
            }
            let cb_final = sorted_cb;

            // Step 2: Sequential GPTQ-style quantize.
            // Forward-propagate the residual error into each next column's
            // target. The "damping" factor controls how aggressively past
            // errors influence future assignments. Empirically:
            //   factor=1.0 — pure forward propagation (full residual)
            //   factor=0.5 — half-damping; safer against runaway accumulation
            //   factor=0.0 — no propagation (degenerates to standard Lloyd)
            // 0.5 is a conservative starting point.
            let mut indices = [0u8; 256];
            let mut residual = 0.0f32;
            for i in 0..256 {
                let target = group[i] + residual;
                let mut best = 0usize;
                let mut best_d = (target - cb_final[0]).abs();
                for k in 1..4 {
                    let d = (target - cb_final[k]).abs();
                    if d < best_d {
                        best_d = d;
                        best = k;
                    }
                }
                indices[i] = best as u8;
                let err = target - cb_final[best];
                residual = err * damping;
            }

            // Pack header + indices.
            for k in 0..4 {
                let bits = f32_to_fp16_bits(cb_final[k]);
                out_chunk[2 * k] = (bits & 0xFF) as u8;
                out_chunk[2 * k + 1] = (bits >> 8) as u8;
            }
            for i in 0..64 {
                let mut byte_val = 0u8;
                for j in 0..4 {
                    byte_val |= (indices[4 * i + j] & 0x3) << (j * 2);
                }
                out_chunk[8 + i] = byte_val;
            }
        });

    output
}

pub(crate) fn quantize_mq2g256_lloyd(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    use rayon::prelude::*;
    let group_size = 256;
    let block_bytes = 72;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    // Parallelize across blocks: each block is independent (own FWHT, own
    // Lloyd's iterations, own centroids). On 24-core boxes this is ~10-15× over
    // the serial path on 9B (single tensor can have >20M blocks).
    output
        .par_chunks_mut(block_bytes)
        .enumerate()
        .for_each(|(b, out_chunk)| {
            let start = b * group_size;
            let end = (start + group_size).min(n);
            let actual_len = end - start;

            let mut group = [0.0f32; 256];
            group[..actual_len].copy_from_slice(&f32_data[start..end]);
            cpu_fwht_256(&mut group, signs1, signs2);

            // Initial centroid placement: percentiles of the rotated block.
            // 12.5/37.5/62.5/87.5 gives a good starting partition — heavy-tail
            // blocks adapt across iterations.
            let mut sorted: [f32; 256] = group;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let percentile = |frac: f32| -> f32 {
                let idx = ((frac * 255.0).round() as usize).min(255);
                sorted[idx]
            };
            let mut cb: [f32; 4] = [
                percentile(0.125),
                percentile(0.375),
                percentile(0.625),
                percentile(0.875),
            ];

            let range = sorted[255] - sorted[0];
            let mut indices = [0u8; 256];
            if range > 0.0 {
                // Lloyd's iterations — cap at 8 (REVERTED from 16 on 2026-05-20).
                //
                // History: f8cd234 (2026-05-19) bumped 8 → 16 based on the
                // `lloyd_iteration_headroom` synthetic-distribution probe,
                // which showed +0.4-0.9% MSE improvement on heavy-tailed +
                // sparse distributions. Free-on-paper, but never gated on a
                // real-model coherence run.
                //
                // 2026-05-20 DeepSeek V4 re-quant under 16-iter measured 60x worse
                // PPL on wikitext2 (758 vs 12 baseline) vs the known-good 8-iter
                // build (byte-identical routed experts → identical bytes hash →
                // "8-iter is the prod-good config").
                //
                // Hypothesis: 16-iter pushes centroids into pathological local
                // minima on FWHT-rotated MoE expert weight distributions. The
                // synthetic probe's "heavy-tailed + sparse" categories didn't
                // capture FWHT-rotated MoE statistics. Classic synth-win →
                // prod-falsify per CLAUDE.md's "Δ ≥ 5% investigation rule".
                //
                // Reverting to 8-iter to match the known-good build until
                // a real-model coherence-gated sweep validates a different
                // value. Do NOT raise this back to 16 (or higher) without
                // running wikitext2 PPL on a DeepSeek V4 build first.
                //
                // 2026-08-04: hoisted to LLOYD_MAX_ITER (see its doc comment) so
                // the weighted + GPTQ arms cannot silently diverge again.
                let max_iter = LLOYD_MAX_ITER;
                let mut prev_assignments = [0u8; 256];
                for it in 0..max_iter {
                    let mut sums = [0.0f64; 4];
                    let mut counts = [0u32; 4];
                    let mut changed = 0u32;
                    for i in 0..256 {
                        let w = group[i];
                        let mut best = 0usize;
                        let mut best_d = (w - cb[0]).abs();
                        for k in 1..4 {
                            let d = (w - cb[k]).abs();
                            if d < best_d {
                                best_d = d;
                                best = k;
                            }
                        }
                        if it == 0 || prev_assignments[i] != best as u8 {
                            changed += 1;
                        }
                        prev_assignments[i] = best as u8;
                        indices[i] = best as u8;
                        sums[best] += w as f64;
                        counts[best] += 1;
                    }
                    if it > 0 && changed == 0 {
                        break;
                    }
                    for k in 0..4 {
                        if counts[k] > 0 {
                            cb[k] = (sums[k] / counts[k] as f64) as f32;
                        }
                    }
                }
            }

            // Sort centroids ascending; remap indices to keep header canonical
            // and the permutation deterministic across re-runs.
            let mut order: [usize; 4] = [0, 1, 2, 3];
            order.sort_by(|&a, &b| {
                cb[a]
                    .partial_cmp(&cb[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut sorted_cb = [0.0f32; 4];
            let mut inv: [u8; 4] = [0; 4];
            for new_idx in 0..4 {
                sorted_cb[new_idx] = cb[order[new_idx]];
                inv[order[new_idx]] = new_idx as u8;
            }
            for i in 0..256 {
                indices[i] = inv[indices[i] as usize];
            }

            for k in 0..4 {
                let bits = f32_to_fp16_bits(sorted_cb[k]);
                out_chunk[2 * k] = (bits & 0xFF) as u8;
                out_chunk[2 * k + 1] = (bits >> 8) as u8;
            }
            // 256 indices × 2 bits = 64 bytes. Same packing as uniform MQ2.
            for i in 0..64 {
                let mut byte_val = 0u8;
                for j in 0..4 {
                    byte_val |= (indices[4 * i + j] & 0x3) << (j * 2);
                }
                out_chunk[8 + i] = byte_val;
            }
        });

    output
}

/// Ternary "MQ1.58" probe: K=3 Lloyd-placed codebook packed into the MQ2-Lloyd
/// container (slot 3 = duplicate of slot 2, never indexed) so it runs on the
/// existing MQ2G256Lloyd kernel with NO new kernel. Measures sub-2-bit
/// *information* (3 levels = log2(3) ≈ 1.58 bit) coherence; storage stays
/// 72 B/group (true 1.58-bpw packing — 5 ternary/byte — is a mechanical
/// follow-up once coherence is established). Gated by HIPFIRE_LLOYD_K3=1 on the
/// `--format mq2lloyd` path. Output DType = MQ2G256Lloyd (kernel-agnostic to K).
pub(crate) fn quantize_mq2g256_lloyd_k3(
    f32_data: &[f32],
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    use rayon::prelude::*;
    let group_size = 256;
    let block_bytes = 72;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];
    output
        .par_chunks_mut(block_bytes)
        .enumerate()
        .for_each(|(b, out_chunk)| {
            let start = b * group_size;
            let end = (start + group_size).min(n);
            let actual_len = end - start;
            let mut group = [0.0f32; 256];
            group[..actual_len].copy_from_slice(&f32_data[start..end]);
            cpu_fwht_256(&mut group, signs1, signs2);

            let mut sorted: [f32; 256] = group;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let percentile = |frac: f32| -> f32 {
                let idx = ((frac * 255.0).round() as usize).min(255);
                sorted[idx]
            };
            // 3 centroids: ~1/6, 1/2, 5/6 percentiles.
            let mut cb: [f32; 3] = [percentile(0.167), percentile(0.5), percentile(0.833)];
            let range = sorted[255] - sorted[0];
            let mut indices = [0u8; 256];
            if range > 0.0 {
                let max_iter = 8;
                let mut prev = [0u8; 256];
                for it in 0..max_iter {
                    let mut sums = [0.0f64; 3];
                    let mut counts = [0u32; 3];
                    let mut changed = 0u32;
                    for i in 0..256 {
                        let w = group[i];
                        let mut best = 0usize;
                        let mut best_d = (w - cb[0]).abs();
                        for k in 1..3 {
                            let d = (w - cb[k]).abs();
                            if d < best_d {
                                best_d = d;
                                best = k;
                            }
                        }
                        if it == 0 || prev[i] != best as u8 {
                            changed += 1;
                        }
                        prev[i] = best as u8;
                        indices[i] = best as u8;
                        sums[best] += w as f64;
                        counts[best] += 1;
                    }
                    if it > 0 && changed == 0 {
                        break;
                    }
                    for k in 0..3 {
                        if counts[k] > 0 {
                            cb[k] = (sums[k] / counts[k] as f64) as f32;
                        }
                    }
                }
            }
            // Sort the 3 centroids ascending; remap indices.
            let mut order: [usize; 3] = [0, 1, 2];
            order.sort_by(|&a, &b| {
                cb[a]
                    .partial_cmp(&cb[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut sorted_cb = [0.0f32; 3];
            let mut inv: [u8; 3] = [0; 3];
            for new_idx in 0..3 {
                sorted_cb[new_idx] = cb[order[new_idx]];
                inv[order[new_idx]] = new_idx as u8;
            }
            for i in 0..256 {
                indices[i] = inv[indices[i] as usize];
            }
            // Header: slots 0..2 = the 3 centroids; slot 3 = dup of slot 2 (never indexed).
            let header = [sorted_cb[0], sorted_cb[1], sorted_cb[2], sorted_cb[2]];
            for k in 0..4 {
                let bits = f32_to_fp16_bits(header[k]);
                out_chunk[2 * k] = (bits & 0xFF) as u8;
                out_chunk[2 * k + 1] = (bits >> 8) as u8;
            }
            for i in 0..64 {
                let mut byte_val = 0u8;
                for j in 0..4 {
                    byte_val |= (indices[4 * i + j] & 0x3) << (j * 2);
                }
                out_chunk[8 + i] = byte_val;
            }
        });
    output
}

/// Inverse FWHT for MQ-family dequantization (sibling of cpu_fwht_256).
pub(crate) fn cpu_inv_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert!(x.len() == 256);
    for i in 0..256 {
        x[i] *= signs2[i];
    }
    let mut stride = 1;
    while stride < 256 {
        let mut i = 0;
        while i < 256 {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let scale = 0.0625; // 1/sqrt(256) = 1/16
    for i in 0..256 {
        x[i] *= scale * signs1[i];
    }
}

/// MQ2-Lloyd dequantize for round-trip / re-quant pipelines. Mirrors
/// the kernel's decode: 4-entry fp16 codebook + 2-bit indices per 256-
/// weight group, then inverse FWHT.
pub(crate) fn dequantize_mq2g256_lloyd_to_f32(
    data: &[u8],
    n_weights: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<f32> {
    let group_size = 256;
    let block_bytes = 72;
    let n_blocks = (n_weights + group_size - 1) / group_size;
    assert!(data.len() == n_blocks * block_bytes);
    let mut out = vec![0.0f32; n_weights];
    use rayon::prelude::*;
    out.par_chunks_mut(group_size)
        .enumerate()
        .for_each(|(b, out_chunk)| {
            let blk = &data[b * block_bytes..(b + 1) * block_bytes];
            let cb: [f32; 4] = [
                f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])),
                f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])),
                f16_to_f32(u16::from_le_bytes([blk[4], blk[5]])),
                f16_to_f32(u16::from_le_bytes([blk[6], blk[7]])),
            ];
            let mut group = [0.0f32; 256];
            for i in 0..64 {
                let byte_val = blk[8 + i];
                for j in 0..4 {
                    let idx = (byte_val >> (j * 2)) & 0x3;
                    group[4 * i + j] = cb[idx as usize];
                }
            }
            cpu_inv_fwht_256(&mut group, signs1, signs2);
            let actual = out_chunk.len();
            out_chunk.copy_from_slice(&group[..actual]);
        });
    out
}

/// MQ2-GL ("global Lloyd") round-trip: quantize → dequantize, returning weights
/// in the ORIGINAL (unrotated) basis. Same pipeline as
/// `quantize_mq2g256_lloyd` + `dequantize_mq2g256_lloyd_to_f32`, except the
/// per-block 4-entry fitted codebook is replaced by ONE tensor-global codebook
/// plus a per-block fp16 scale.
///
/// The codebook is the textbook Lloyd–Max optimum for a unit Gaussian. That is
/// not an approximation of convenience: post-FWHT blocks are Gaussian by CLT,
/// and fitting a global codebook on 28.3M real a3b expert weights reproduces
/// these levels to three decimals (measured 2026-08-04, see
/// docs/investigations/2026-08-04-a3b-lowbit-quality.md §5c).
///
/// Cost/benefit on those same real weights: +2.35% NRMSE for −0.1875 bpw
/// (72 B/group → 64 B payload + 2 B scale).
///
/// Used by `--format mq4-mq2glexp`, the GL twin of `mq4-mq2lloydexp`: it injects
/// the GL codec's noise and re-packs as HFQ4G256 so the file loads on today's
/// runtime with no engine, loader, or kernel changes. Both probes land in the
/// same HFQ4 container, so a KLD delta between them isolates the codec.
pub(crate) fn mq2g256gl_roundtrip_f32(
    f32_data: &[f32],
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<f32> {
    /// Lloyd–Max levels for a unit Gaussian at 2 bit.
    pub(crate) const CB: [f32; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];
    let group_size = 256;
    let n = f32_data.len();
    let mut out = vec![0.0f32; n];
    use rayon::prelude::*;
    out.par_chunks_mut(group_size)
        .enumerate()
        .for_each(|(b, out_chunk)| {
            let start = b * group_size;
            let end = (start + group_size).min(n);
            let actual = end - start;

            let mut group = [0.0f32; 256];
            group[..actual].copy_from_slice(&f32_data[start..end]);
            cpu_fwht_256(&mut group, signs1, signs2);

            // Per-block scale, rounded through fp16 exactly as the on-disk
            // format would store it.
            let ss: f64 = group.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            let rms = (ss / 256.0).sqrt() as f32;
            let scale = if rms > 0.0 {
                f16_to_f32(f32_to_fp16_bits(rms))
            } else {
                0.0
            };
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };

            for v in group.iter_mut() {
                let z = *v * inv;
                let mut best = 0usize;
                let mut best_d = (z - CB[0]).abs();
                for (k, &c) in CB.iter().enumerate().skip(1) {
                    let d = (z - c).abs();
                    if d < best_d {
                        best_d = d;
                        best = k;
                    }
                }
                *v = scale * CB[best];
            }

            cpu_inv_fwht_256(&mut group, signs1, signs2);
            let take = out_chunk.len();
            out_chunk.copy_from_slice(&group[..take]);
        });
    out
}

/// Lloyd–Max optimal reconstruction levels for a unit Gaussian.
/// 2-bit MSE = 0.1175, 3-bit MSE = 0.03454 — both reproduced to 3 decimals by
/// fitting on 28.3M real a3b post-FWHT expert weights (2026-08-04).
pub(crate) const GL_CB2: [f32; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];
pub(crate) const GL_CB3: [f32; 8] = [
    -2.1520, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1520,
];

/// Encode one FWHT-rotated 256-block against a global codebook.
/// Returns the fp16-rounded per-block scale and writes indices into `idx`.
#[inline]
pub(crate) fn gl_encode_block(group: &[f32; 256], cb: &[f32], idx: &mut [u8; 256]) -> u16 {
    let ss: f64 = group.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let rms = (ss / 256.0).sqrt() as f32;
    let sbits = f32_to_fp16_bits(rms);
    let scale = f16_to_f32(sbits);
    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    for (i, v) in group.iter().enumerate() {
        let z = *v * inv;
        let mut best = 0usize;
        let mut best_d = (z - cb[0]).abs();
        for (k, &c) in cb.iter().enumerate().skip(1) {
            let d = (z - c).abs();
            if d < best_d {
                best_d = d;
                best = k;
            }
        }
        idx[i] = best as u8;
    }
    sbits
}

/// MQ2-G256-GL: 2-bit codes vs one tensor-global codebook + per-block fp16
/// scale, structure-of-arrays. 2.0625 bpw.
///
/// Layout: `[m*gpr*64 B packed indices][m*gpr*2 B fp16 scales]`, both regions
/// row-major in (row, group). Index packing matches MQ2-Lloyd (4 codes/byte,
/// little-endian) so the GEMV decode path is unchanged apart from where the
/// codebook comes from. `k` must be a multiple of 256.
pub(crate) fn quantize_mq2g256gl(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    assert_eq!(k % 256, 0, "MQ2GL: K must be a multiple of 256 (got {k})");
    let gpr = k / 256;
    let idx_bytes = m * gpr * 64;
    let mut out = vec![0u8; idx_bytes + m * gpr * 2];
    let (idx_region, scale_region) = out.split_at_mut(idx_bytes);
    use rayon::prelude::*;
    idx_region
        .par_chunks_mut(gpr * 64)
        .zip(scale_region.par_chunks_mut(gpr * 2))
        .enumerate()
        .for_each(|(row, (row_idx, row_scale))| {
            for g in 0..gpr {
                let start = row * k + g * 256;
                let mut group = [0.0f32; 256];
                group.copy_from_slice(&f32_data[start..start + 256]);
                cpu_fwht_256(&mut group, signs1, signs2);
                let mut codes = [0u8; 256];
                let sbits = gl_encode_block(&group, &GL_CB2, &mut codes);
                let base = g * 64;
                for b in 0..64 {
                    row_idx[base + b] = codes[4 * b]
                        | (codes[4 * b + 1] << 2)
                        | (codes[4 * b + 2] << 4)
                        | (codes[4 * b + 3] << 6);
                }
                row_scale[g * 2] = (sbits & 0xFF) as u8;
                row_scale[g * 2 + 1] = (sbits >> 8) as u8;
            }
        });
    out
}

/// MQ3-G256-GL: 3-bit sibling of `quantize_mq2g256gl`. 3.0625 bpw.
/// 96 B of indices per group — 8 codes packed into every 3 bytes,
/// little-endian bitstream (same convention as HFQ3-G256).
pub(crate) fn quantize_mq3g256gl(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    assert_eq!(k % 256, 0, "MQ3GL: K must be a multiple of 256 (got {k})");
    let gpr = k / 256;
    let idx_bytes = m * gpr * 96;
    let mut out = vec![0u8; idx_bytes + m * gpr * 2];
    let (idx_region, scale_region) = out.split_at_mut(idx_bytes);
    use rayon::prelude::*;
    idx_region
        .par_chunks_mut(gpr * 96)
        .zip(scale_region.par_chunks_mut(gpr * 2))
        .enumerate()
        .for_each(|(row, (row_idx, row_scale))| {
            for g in 0..gpr {
                let start = row * k + g * 256;
                let mut group = [0.0f32; 256];
                group.copy_from_slice(&f32_data[start..start + 256]);
                cpu_fwht_256(&mut group, signs1, signs2);
                let mut codes = [0u8; 256];
                let sbits = gl_encode_block(&group, &GL_CB3, &mut codes);
                let base = g * 96;
                // 8 codes × 3 bits = 24 bits = 3 bytes.
                for c in 0..32 {
                    let mut acc: u32 = 0;
                    for j in 0..8 {
                        acc |= ((codes[8 * c + j] & 0x7) as u32) << (3 * j);
                    }
                    row_idx[base + 3 * c] = (acc & 0xFF) as u8;
                    row_idx[base + 3 * c + 1] = ((acc >> 8) & 0xFF) as u8;
                    row_idx[base + 3 * c + 2] = ((acc >> 16) & 0xFF) as u8;
                }
                row_scale[g * 2] = (sbits & 0xFF) as u8;
                row_scale[g * 2 + 1] = (sbits >> 8) as u8;
            }
        });
    out
}

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
// ---- TQ2G128 / BQ1G128 low-bit packers (ported from pr-597 main.rs) ----

pub(crate) fn quantize_tq2g128(f32_data: &[f32]) -> Vec<u8> {
    const GROUP: usize = 128;
    const BLOCK_BYTES: usize = 34;
    let n = f32_data.len();
    let n_blocks = (n + GROUP - 1) / GROUP;
    let mut output = vec![0u8; n_blocks * BLOCK_BYTES];

    for b in 0..n_blocks {
        let start = b * GROUP;
        let end = (start + GROUP).min(n);
        let group = &f32_data[start..end];

        let mut amax = 0.0f32;
        for &w in group {
            amax = amax.max(w.abs());
        }
        let d = amax;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };

        let out_off = b * BLOCK_BYTES;
        output[out_off..out_off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());

        let actual_len = end - start;
        for i in 0..32 {
            let mut byte_val = 0u8;
            for j in 0..4 {
                let idx = 4 * i + j;
                let w = if idx < actual_len { group[idx] } else { 0.0 };
                let mut q = (w * id).round() as i32 + 1;
                // Clamp to the TERNARY set {0,1,2} == {-d, 0, +d}. NOT 3: both
                // decoders compute `(code-1)*d`, so code 3 decodes to +2d and
                // makes this an asymmetric 4-level quantizer (negatives
                // saturating at -d while positives reach +2d). Only reachable
                // once d < max|w|, i.e. under the scale sweep.
                if q < 0 {
                    q = 0;
                }
                if q > 2 {
                    q = 2;
                }
                byte_val |= (q as u8) << (j * 2);
            }
            output[out_off + 2 + i] = byte_val;
        }
    }

    output
}

#[cfg(test)]
mod tq2g128_tests {
    use super::*;

    /// Byte-exactness of `quantize_tq2g128` against a hand-derived expected
    /// block. Group of 128: idx0=-2.0, idx1=0.0, idx2=2.0, idx3..127=0.0.
    /// d = max(|w|) = 2.0, id = 0.5.
    ///
    /// Per-element codes (q = clamp(round(w*id)+1, 0, 3)):
    ///   idx0: round(-2.0*0.5)+1 = round(-1.0)+1 = 0
    ///   idx1: round( 0.0*0.5)+1 = round( 0.0)+1 = 1
    ///   idx2: round( 2.0*0.5)+1 = round( 1.0)+1 = 2
    ///   idx3..127 (all 0.0): round(0.0)+1 = 1
    ///
    /// NOTE: the zero-valued tail encodes to code 1 (not code 0) because
    /// id=0.5 != 0 here (d != 0) — only an all-zero *group* (d==0, id==0)
    /// would encode zeros as code 0. This matters for the packed bytes:
    /// byte0 packs elements 0..3 LSB-first (`code << ((j%4)*2)`):
    ///   byte0 = code0 | code1<<2 | code2<<4 | code3<<6
    ///         =    0  |   1<<2   |   2<<4   |   1<<6
    ///         = 0b01_10_01_00 = 0x64
    /// bytes[1..32] cover elements 4..127, all code=1:
    ///   byte = 1 | 1<<2 | 1<<4 | 1<<6 = 0b01_01_01_01 = 0x55
    ///
    /// FP16(2.0) = 0x4000 (sign=0, exp=16, frac=0), little-endian [0x00,0x40] —
    /// cross-checked against the existing `dequant_q2_0` CPU-oracle test in
    /// gguf_input.rs, which uses the identical byte pair for scale 2.0.
    #[test]
    fn quantize_tq2g128_byte_exact() {
        let mut group = vec![0.0f32; 128];
        group[0] = -2.0;
        group[1] = 0.0;
        group[2] = 2.0;
        // group[3..128] stay 0.0

        let out = quantize_tq2g128(&group);
        assert_eq!(out.len(), 34, "one 128-group -> one 34-byte block");

        // FP16 scale d=2.0, little-endian.
        assert_eq!(&out[0..2], &[0x00, 0x40]);

        // qs[0] (elements 0..3): codes 0,1,2,1 -> 0x64.
        assert_eq!(out[2], 0x64);

        // qs[1..32] (elements 4..127): all code=1 -> 0x55, repeated 31 times.
        assert_eq!(&out[3..34], &[0x55u8; 31][..]);
    }

    /// Round-trip losslessness for an already-ternary input: values drawn
    /// from {-d, 0, +d} with d a power of two (4.0) so the FP16 scale is
    /// exact and every code round-trips without rounding error. Decodes via
    /// the crate's public `tensor_to_f32` dispatcher (which routes
    /// `GgmlType::Q2_0` to the private `dequant_q2_0` CPU oracle added in
    /// Task 5) — this exercises the *real* decode path rather than a
    /// second hand-rolled decoder, so it also proves TQ2G128's block
    /// layout is byte-compatible with GGUF `Q2_0` end-to-end.
    #[test]
    fn quantize_tq2g128_round_trips_ternary_input() {
        let d = 4.0f32;
        let f32_data: Vec<f32> = (0..128)
            .map(|i| match i % 3 {
                0 => -d,
                1 => 0.0,
                _ => d,
            })
            .collect();

        let packed = quantize_tq2g128(&f32_data);
        assert_eq!(packed.len(), 34);

        let info = gguf_input::TensorInfo {
            name: "round_trip".to_string(),
            shape: vec![128],
            dtype: gguf_input::GgmlType::Q2_0,
            offset: 0,
        };
        let decoded = gguf_input::tensor_to_f32(&info, &packed);

        assert_eq!(decoded, f32_data);
    }

    /// Multi-block input (2 groups of 128) — checks block offsets/lengths
    /// are computed correctly, not just the single-block case above.
    #[test]
    fn quantize_tq2g128_multi_block_length() {
        let f32_data = vec![1.0f32; 256];
        let out = quantize_tq2g128(&f32_data);
        assert_eq!(out.len(), 2 * 34);
    }
}

pub(crate) fn quantize_tq2g128_gptq(
    f32_data: &[f32],
    col_weights: &[f32],
    damping: f32,
) -> Vec<u8> {
    const GROUP: usize = 128;
    const BLOCK_BYTES: usize = 34;
    let n = f32_data.len();
    let n_blocks = (n + GROUP - 1) / GROUP;
    let blocks_per_row = (col_weights.len() / GROUP).max(1);
    let mut output = vec![0u8; n_blocks * BLOCK_BYTES];

    for b in 0..n_blocks {
        let start = b * GROUP;
        let end = (start + GROUP).min(n);
        let group = &f32_data[start..end];
        let actual = end - start;

        let col_off = (b % blocks_per_row) * GROUP;
        let block_w = &col_weights[col_off..col_off + GROUP];

        let mut amax = 0.0f32;
        for &w in group {
            amax = amax.max(w.abs());
        }

        // Sweep candidate scales, pick the least imatrix-weighted *decoded*
        // error. factor=1.0 reproduces `quantize_tq2g128` exactly (same codes,
        // same f16 header, same decode), so the winner is always ≤ plain.
        //
        // The sweep MUST reach well below 0.6·amax. Levels are {-d, 0, +d} with
        // nearest-neighbour coding, so a weight survives only when |w| ≥ d/2;
        // at d = amax over a 128-sample Gaussian block (amax ≈ 2.9σ) that
        // threshold is ~1.45σ and ~85% of the block is zeroed. The MSE-optimal
        // 3-level scale for a Gaussian is d ≈ 1.22σ ≈ 0.42·amax (~54% non-zero;
        // PrismML's own Bonsai ternary runs ~69% non-zero). A sweep floored at
        // 0.6 bottoms out at ~38% non-zero and never reaches the optimum.
        let mut best_d = amax;
        let mut best_err = f64::INFINITY;
        for &factor in &[
            0.20f32, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.0,
        ] {
            let d = amax * factor;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            let dd = f16_to_f32(f32_to_f16(d)) as f64; // decode uses the f16 scale
            let mut err = 0.0f64;
            for i in 0..GROUP {
                let w = if i < actual { group[i] } else { 0.0 };
                let mut q = (w * id).round() as i32 + 1;
                // Clamp to the TERNARY set {0,1,2} == {-d, 0, +d}. NOT 3: both
                // decoders compute `(code-1)*d`, so code 3 decodes to +2d and
                // makes this an asymmetric 4-level quantizer (negatives
                // saturating at -d while positives reach +2d). Only reachable
                // once d < max|w|, i.e. under the scale sweep.
                if q < 0 {
                    q = 0;
                }
                if q > 2 {
                    q = 2;
                }
                let recon = (q - 1) as f64 * dd;
                let diff = w as f64 - recon;
                err += block_w[i] as f64 * diff * diff;
            }
            if err < best_err {
                best_err = err;
                best_d = d;
            }
        }

        let d = best_d;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let dd = f16_to_f32(f32_to_f16(d));

        // Damped residual feed-forward across columns → codes.
        let mut codes = [0u8; GROUP];
        let mut residual = 0.0f32;
        for i in 0..GROUP {
            let w = if i < actual { group[i] } else { 0.0 };
            let target = w + residual;
            let mut q = (target * id).round() as i32 + 1;
            // Ternary set {0,1,2}; see the note above — 3 would decode to +2d.
            if q < 0 {
                q = 0;
            }
            if q > 2 {
                q = 2;
            }
            let recon = (q - 1) as f32 * dd;
            residual = (target - recon) * damping;
            codes[i] = q as u8;
        }

        let out_off = b * BLOCK_BYTES;
        output[out_off..out_off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for i in 0..32 {
            let mut byte_val = 0u8;
            for j in 0..4 {
                byte_val |= (codes[4 * i + j] & 0x3) << (j * 2);
            }
            output[out_off + 2 + i] = byte_val;
        }
    }
    output
}

#[cfg(test)]
mod tq2g128_gptq_tests {
    use super::*;

    fn weighted_mse(orig: &[f32], packed: &[u8], w: &[f32]) -> f64 {
        let info = gguf_input::TensorInfo {
            name: "wmse".to_string(),
            shape: vec![orig.len()],
            dtype: gguf_input::GgmlType::Q2_0,
            offset: 0,
        };
        let deq = gguf_input::tensor_to_f32(&info, packed);
        orig.iter()
            .zip(&deq)
            .zip(w)
            .map(|((o, q), wi)| (*wi as f64) * ((o - q) as f64).powi(2))
            .sum()
    }

    #[test]
    fn gptq_ternary_not_worse_than_plain() {
        // skewed importance: first 32 columns matter 10×.
        let g: Vec<f32> = (0..128).map(|i| ((i as f32) - 64.0) * 0.03).collect();
        let mut w = vec![1.0f32; 128];
        for k in 0..32 {
            w[k] = 10.0;
        }
        let plain = quantize_tq2g128(&g);
        let gptq = quantize_tq2g128_gptq(&g, &w, 0.0);
        assert!(
            weighted_mse(&g, &gptq, &w) <= weighted_mse(&g, &plain, &w) + 1e-9,
            "imatrix-weighted GPTQ error must be ≤ plain packer error"
        );
    }

    /// Deterministic standard-normal block (Box–Muller over an LCG) — real
    /// weights are Gaussian-ish, and the scale sweep must work on THOSE, not
    /// just on the uniform ramp above (a ramp's `max|w|` is ~1 sigma-equivalent
    /// so it hides a mis-ranged sweep entirely).
    fn gaussian_block(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 + 0.5) / ((1u32 << 24) as f32)
        };
        (0..n)
            .map(|_| {
                let u1: f32 = next();
                let u2: f32 = next();
                (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
            })
            .collect()
    }

    /// `tq2_pack_stats` is the shipping guard's eye: it must count the two
    /// things that silently broke real requants — how much of the model got
    /// zeroed, and whether any out-of-set code was emitted.
    #[test]
    fn tq2_pack_stats_counts_zeros_and_out_of_set_codes() {
        // One block, 128 codes. Byte 0 = codes [0,1,2,3] (LSB-first, 2 b each)
        // = 0b11_10_01_00 = 0xE4. Remaining 31 bytes = 0x55 = codes [1,1,1,1],
        // i.e. all zero-level.
        let mut blk = vec![0u8; 34];
        blk[0..2].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
        blk[2] = 0xE4;
        for b in blk.iter_mut().skip(3) {
            *b = 0x55;
        }

        let st = tq2_pack_stats(&blk);
        assert_eq!(st.n_codes, 128);
        // Non-zero = code != 1 → the 0, 2 and 3 in byte 0.
        assert_eq!(st.nonzero, 3);
        assert_eq!(st.out_of_set, 1, "exactly one code 3");
    }

    /// Raw 2-bit codes out of a TQ2G128 buffer (34 B/block: [f16 d][32 B qs]).
    fn tq2_codes(packed: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        for blk in packed.chunks_exact(34) {
            for &byte in &blk[2..] {
                for j in 0..4 {
                    out.push((byte >> (j * 2)) & 0x3);
                }
            }
        }
        out
    }

    /// The ternary level set is {-d, 0, +d} == codes {0, 1, 2}. Code 3 decodes
    /// to `(3-1)*d = +2d` in BOTH decoders (gguf_input::dequant_q2_0 and
    /// kernels/src/gemv_tq2g128.hip), so emitting it silently turns the
    /// quantizer into an ASYMMETRIC 4-level one: negatives saturate at -d while
    /// positives reach +2d. PrismML's own Bonsai ternary never emits code 3
    /// (measured 0.0000 across 4 tensors).
    ///
    /// The `q > 3 => 3` clamp was unreachable while d = max|w| (|w/d| <= 1, so
    /// round() is at most 1). Widening the scale sweep to 0.20*amax makes
    /// |w/d| >> 1 reachable — which is how this surfaced: 5.9% of codes on a
    /// real 27B requant.
    #[test]
    fn ternary_packers_never_emit_the_out_of_set_code() {
        let g = gaussian_block(128, 4242);
        let w = vec![1.0f32; 128];
        for (label, packed) in [
            ("plain", quantize_tq2g128(&g)),
            ("gptq", quantize_tq2g128_gptq(&g, &w, 0.0)),
        ] {
            let codes = tq2_codes(&packed);
            let bad = codes.iter().filter(|&&c| c == 3).count();
            assert_eq!(
                bad,
                0,
                "{label}: emitted out-of-set code 3 ({bad} of {} codes) — decodes to +2d",
                codes.len()
            );
        }
    }

    /// Saturation must be symmetric: with a scale below max|w|, mirrored
    /// out-of-range weights must land on mirrored levels.
    #[test]
    fn ternary_saturation_is_symmetric() {
        let mut g = gaussian_block(128, 99);
        g[0] = 6.0; // far above any swept d
        g[1] = -6.0;
        let w = vec![1.0f32; 128];
        let codes = tq2_codes(&quantize_tq2g128_gptq(&g, &w, 0.0));
        assert_eq!(codes[0], 2, "large positive must saturate to +d (code 2)");
        assert_eq!(codes[1], 0, "large negative must saturate to -d (code 0)");
    }

    /// The ternary level set is `{-d, 0, +d}` with nearest-neighbour coding, so
    /// a weight survives only when `|w| >= d/2`. With `d = max|w|` over a
    /// 128-sample Gaussian block (`amax ~ 2.9 sigma`) the threshold lands at
    /// ~1.45 sigma and ~85% of weights are zeroed — which is exactly what the
    /// plain packer shipped for qwen3.6-27b (16.3% non-zero measured, vs 69.4%
    /// for PrismML's own Bonsai ternary).
    ///
    /// The MSE-optimal 3-level scale for a Gaussian is `d ~ 1.22 sigma`
    /// (thresholds at +/-0.61 sigma, ~54% non-zero) = ~0.42 * amax. The sweep
    /// must therefore reach well below 0.6 * amax.
    #[test]
    fn gptq_ternary_scale_sweep_reaches_mse_optimum_on_gaussian() {
        let g = gaussian_block(128, 12345);
        let w = vec![1.0f32; 128];

        let plain = quantize_tq2g128(&g);
        let gptq = quantize_tq2g128_gptq(&g, &w, 0.0);

        let nz = tq2_pack_stats(&gptq).nonzero_fraction();
        assert!(
            (0.45..=0.80).contains(&nz),
            "swept-scale ternary should keep roughly half the weights alive on \
             Gaussian input (MSE-optimal ~0.54); got non-zero fraction {nz:.3}"
        );

        let (e_gptq, e_plain) = (weighted_mse(&g, &gptq, &w), weighted_mse(&g, &plain, &w));
        assert!(
            e_gptq <= 0.6 * e_plain,
            "swept scale must substantially beat d=max|w| on Gaussian input; \
             got gptq={e_gptq:.5} vs plain={e_plain:.5}"
        );
    }
}

pub(crate) fn quantize_bq1g128(f32_data: &[f32]) -> Vec<u8> {
    const GROUP: usize = 128;
    const BLOCK_BYTES: usize = 18;
    let n = f32_data.len();
    let n_blocks = (n + GROUP - 1) / GROUP;
    let mut output = vec![0u8; n_blocks * BLOCK_BYTES];
    for b in 0..n_blocks {
        let start = b * GROUP;
        let end = (start + GROUP).min(n);
        let group = &f32_data[start..end];
        let actual = end - start;

        let mut sum = 0.0f32;
        for &w in group {
            sum += w.abs();
        }
        let d = if actual > 0 { sum / actual as f32 } else { 0.0 };

        let out_off = b * BLOCK_BYTES;
        output[out_off..out_off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for j in 0..GROUP {
            let w = if j < actual { group[j] } else { 0.0 };
            if w >= 0.0 {
                output[out_off + 2 + (j >> 3)] |= 1u8 << (j & 7);
            }
        }
    }
    output
}

#[cfg(test)]
mod bq1g128_tests {
    use super::*;

    #[test]
    fn quantize_bq1g128_byte_exact() {
        // group of 128: [0]=+2, [1]=-2, rest 0 (0 → sign bit set → +d)
        let mut g = vec![0.0f32; 128];
        g[0] = 2.0;
        g[1] = -2.0;
        // d = mean|w| = (2+2)/128 = 0.03125
        let out = quantize_bq1g128(&g);
        assert_eq!(out.len(), 18);
        assert_eq!(&out[0..2], &f32_to_f16(0.03125).to_le_bytes());
        // byte 2 = elements 0..7: bit0 (e0,+)=1, bit1 (e1,-)=0,
        // bits2..7 (0.0≥0)=1 → 0b1111_1101 = 0xFD
        assert_eq!(out[2], 0xFD);
        // bytes 3..18 = elements 8..127 all 0.0 ≥ 0 → all bits set → 0xFF
        assert_eq!(&out[3..18], &[0xFFu8; 15]);
    }

    #[test]
    fn quantize_bq1g128_round_trips() {
        let g: Vec<f32> = (0..256).map(|i| ((i as f32) - 128.0) * 0.01).collect();
        let packed = quantize_bq1g128(&g);
        let info = gguf_input::TensorInfo {
            name: "rt".to_string(),
            shape: vec![256],
            dtype: gguf_input::GgmlType::Q1_0,
            offset: 0,
        };
        let deq = gguf_input::tensor_to_f32(&info, &packed);
        // every element recovers sign; magnitude == block mean|w|
        for i in 0..256 {
            assert_eq!(deq[i] >= 0.0, g[i] >= 0.0);
        }
    }
}

pub(crate) fn quantize_bq1g128_gptq(
    f32_data: &[f32],
    col_weights: &[f32],
    damping: f32,
) -> Vec<u8> {
    const GROUP: usize = 128;
    const BLOCK_BYTES: usize = 18;
    let n = f32_data.len();
    let n_blocks = (n + GROUP - 1) / GROUP;
    let blocks_per_row = (col_weights.len() / GROUP).max(1);
    let mut output = vec![0u8; n_blocks * BLOCK_BYTES];

    for b in 0..n_blocks {
        let start = b * GROUP;
        let end = (start + GROUP).min(n);
        let group = &f32_data[start..end];
        let actual = end - start;

        let col_off = (b % blocks_per_row) * GROUP;
        let block_w = &col_weights[col_off..col_off + GROUP];

        // Importance-weighted optimal magnitude d = Σ w|x| / Σ w.
        let mut wsum = 0.0f64;
        let mut wabs = 0.0f64;
        for i in 0..actual {
            let wi = block_w[i] as f64;
            wsum += wi;
            wabs += wi * (group[i].abs() as f64);
        }
        let d = if wsum > 0.0 {
            (wabs / wsum) as f32
        } else {
            // fallback: plain unweighted mean-abs (matches quantize_bq1g128)
            let mut s = 0.0f32;
            for &w in group {
                s += w.abs();
            }
            if actual > 0 {
                s / actual as f32
            } else {
                0.0
            }
        };
        let dd = f16_to_f32(f32_to_f16(d));

        let out_off = b * BLOCK_BYTES;
        output[out_off..out_off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());

        let mut residual = 0.0f32;
        for j in 0..GROUP {
            let w = if j < actual { group[j] } else { 0.0 };
            let target = w + residual;
            let bit = target >= 0.0;
            let recon = if bit { dd } else { -dd };
            residual = (target - recon) * damping;
            if bit {
                output[out_off + 2 + (j >> 3)] |= 1u8 << (j & 7);
            }
        }
    }
    output
}

#[cfg(test)]
mod bq1g128_gptq_tests {
    use super::*;

    fn wmse_bq1(orig: &[f32], packed: &[u8], w: &[f32]) -> f64 {
        let info = gguf_input::TensorInfo {
            name: "wmse".to_string(),
            shape: vec![orig.len()],
            dtype: gguf_input::GgmlType::Q1_0,
            offset: 0,
        };
        let deq = gguf_input::tensor_to_f32(&info, packed);
        orig.iter()
            .zip(&deq)
            .zip(w)
            .map(|((o, q), wi)| (*wi as f64) * ((o - q) as f64).powi(2))
            .sum()
    }

    #[test]
    fn gptq_binary_not_worse_than_plain() {
        let g: Vec<f32> = (0..128).map(|i| ((i as f32) - 64.0) * 0.05).collect();
        let mut w = vec![1.0f32; 128];
        for k in 96..128 {
            w[k] = 8.0;
        } // tail columns important
        let plain = quantize_bq1g128(&g);
        let gptq = quantize_bq1g128_gptq(&g, &w, 0.0);
        assert!(wmse_bq1(&g, &gptq, &w) <= wmse_bq1(&g, &plain, &w) + 1e-9);
    }
}

#[cfg(test)]
mod mqv2_lowbit_tests {
    use super::*;

    #[test]
    fn mq3v2_mq2v2_wire_layout() {
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        // byte count = m*ceil(k/256)*B
        for (m, k, b) in [
            (1usize, 256usize, MQ3V2_GROUP_BYTES),
            (2, 512, MQ3V2_GROUP_BYTES),
            (4, 1024, MQ3V2_GROUP_BYTES),
        ] {
            let w = vec![0.3f32; m * k];
            let blob = quantize_mq3g256v2(&w, m, k, &s1, &s2);
            assert_eq!(blob.len(), m * (k / 256) * b, "mq3v2 bytes m={m} k={k}");
            assert_eq!(b, 104);
            // header 8B, payload 96B
            assert_eq!(b - 8, 96);
        }
        for (m, k, b) in [
            (1usize, 256usize, MQ2V2_GROUP_BYTES),
            (2, 512, MQ2V2_GROUP_BYTES),
        ] {
            let w = vec![-0.5f32; m * k];
            let blob = quantize_mq2g256v2(&w, m, k, &s1, &s2);
            assert_eq!(blob.len(), m * (k / 256) * b, "mq2v2 bytes m={m} k={k}");
            assert_eq!(b, 72);
            assert_eq!(b - 8, 64);
        }
        // fp16 header positions and payload offset
        let m = 2;
        let k = 512;
        let w: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013).sin()).collect();
        let blob = quantize_mq3g256v2(&w, m, k, &s1, &s2);
        let gpr = k / 256;
        for g in 0..m * gpr {
            let base = g * MQ3V2_GROUP_BYTES;
            // header is 4 fp16 LE at 0,2,4,6
            let s0 = u16::from_le_bytes([blob[base], blob[base + 1]]);
            let z0 = u16::from_le_bytes([blob[base + 2], blob[base + 3]]);
            let s1b = u16::from_le_bytes([blob[base + 4], blob[base + 5]]);
            let z1 = u16::from_le_bytes([blob[base + 6], blob[base + 7]]);
            // reconstruction would be q*f32(s)+f32(z)
            let st0 = f16_to_f32(s0);
            let zt0 = f16_to_f32(z0);
            // at least not both scales zero for non-degenerate synthetic
            assert!(
                st0 != 0.0 || f16_to_f32(s1b) != 0.0,
                "expected non-degenerate header"
            );
            assert!(blob[base + 8..base + MQ3V2_GROUP_BYTES].len() == 96);
            let _ = (zt0, z1, s1b); // silence unused
        }
        // degenerate halves: zero weights => both halves scale=0, payload zero
        let w_zero = vec![0.0f32; m * k];
        let blob = quantize_mq2g256v2(&w_zero, m, k, &s1, &s2);
        for g in 0..m * gpr {
            let base = g * MQ2V2_GROUP_BYTES;
            let s0 = u16::from_le_bytes([blob[base], blob[base + 1]]);
            let s1b = u16::from_le_bytes([blob[base + 4], blob[base + 5]]);
            assert_eq!(s0, 0);
            assert_eq!(s1b, 0);
            // payload must be all zero for degenerate
            assert!(blob[base + 8..base + MQ2V2_GROUP_BYTES]
                .iter()
                .all(|&b| b == 0));
        }
        // payload pack invariants: 2b 4/byte => payload len 64, 3b 8/3B => 96
        assert_eq!(MQ2V2_GROUP_BYTES - 8, 64);
        assert_eq!(MQ3V2_GROUP_BYTES - 8, 96);
    }

    #[test]
    fn mqv2_roundtrip_fp16_before_select() {
        // Verify encoder round-trips s/z through fp16 before selecting q:
        // construct a weight that would be near a quantization boundary with f32 scale
        // but flips when scale is truncated to fp16.
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let m = 1;
        let k = 256;
        // Use a value that exercises fp16 quantization of scale
        let w: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let blob2 = quantize_mq2g256v2(&w, m, k, &s1, &s2);
        let blob3 = quantize_mq3g256v2(&w, m, k, &s1, &s2);
        // header scales must be fp16 exactly (re-encode matches)
        for (blob, bbytes) in [(blob2, MQ2V2_GROUP_BYTES), (blob3, MQ3V2_GROUP_BYTES)] {
            let s0 = u16::from_le_bytes([blob[0], blob[1]]);
            let z0 = u16::from_le_bytes([blob[2], blob[3]]);
            // f32_to_f16 truncates, f16_to_f32 round-trips, ensure consistency
            let st = f16_to_f32(s0);
            let zt = f16_to_f32(z0);
            assert!(st >= 0.0);
            let _ = (st, zt, bbytes);
        }
    }
}

/// Code-level statistics of a packed TQ2G128 buffer.
#[derive(Default, Debug, Clone, Copy)]
struct Tq2PackStats {
    /// Codes decoding to a non-zero level (i.e. code != 1).
    nonzero: u64,
    /// Codes equal to 3 — outside the ternary set, decoding to +2d.
    out_of_set: u64,
    n_codes: u64,
}

impl Tq2PackStats {
    fn add(&mut self, o: Tq2PackStats) {
        self.nonzero += o.nonzero;
        self.out_of_set += o.out_of_set;
        self.n_codes += o.n_codes;
    }
    fn nonzero_fraction(&self) -> f64 {
        if self.n_codes == 0 {
            return 0.0;
        }
        self.nonzero as f64 / self.n_codes as f64
    }
}

/// Count zero / non-zero / out-of-set codes in a TQ2G128 buffer
/// (34 B per 128-weight block: `[f16 d][32 B codes]`, 4 codes per byte).
fn tq2_pack_stats(data: &[u8]) -> Tq2PackStats {
    let mut st = Tq2PackStats::default();
    for blk in data.chunks_exact(34) {
        for &byte in &blk[2..] {
            for j in 0..4 {
                let code = (byte >> (j * 2)) & 0x3;
                st.n_codes += 1;
                if code != 1 {
                    st.nonzero += 1;
                }
                if code == 3 {
                    st.out_of_set += 1;
                }
            }
        }
    }
    st
}

/// Refuse to write a ternary model that the code histogram says is broken.
///
/// This is the cheap observable that would have caught both shipped defects
/// immediately, without a GPU or an eval:
///
///   * `d = max|w|` as an ENCODER zeroed 83.7% of a real 27B requant (16.3%
///     non-zero). Healthy is ~54% (Gaussian MSE optimum) to ~69% (PrismML's
///     own Bonsai ternary). A model below `MIN_NONZERO` is not "lossy", it is
///     mostly deleted.
///   * code 3 decodes to `+2d` in both decoders, turning the quantizer into an
///     asymmetric 4-level one. PrismML never emits it; neither should we.
///
/// `--allow-degenerate-ternary` (env HIPFIRE_ALLOW_DEGENERATE_TERNARY) downgrades
/// this to a warning. Read at the CLI boundary, never inside the pipeline:
/// a getenv racing another test's setenv in a threaded test binary is unsound.
fn check_ternary_pack_health(st: Tq2PackStats, allow_degenerate: bool) {
    const MIN_NONZERO: f64 = 0.25;
    if st.n_codes == 0 {
        return;
    }
    let nz = st.nonzero_fraction();
    eprintln!(
        "ternary pack health: {:.1}% non-zero codes ({} of {}), {} out-of-set",
        nz * 100.0,
        st.nonzero,
        st.n_codes,
        st.out_of_set
    );
    let degenerate = nz < MIN_NONZERO;
    if !degenerate && st.out_of_set == 0 {
        return;
    }
    let mut why = Vec::new();
    if degenerate {
        why.push(format!(
            "only {:.1}% of codes are non-zero (expected >={:.0}%; healthy 54-69%) \
             — {:.1}% of the model is zeroed",
            nz * 100.0,
            MIN_NONZERO * 100.0,
            (1.0 - nz) * 100.0
        ));
    }
    if st.out_of_set > 0 {
        why.push(format!(
            "{} codes are 3, which decodes to +2d (outside the ternary set)",
            st.out_of_set
        ));
    }
    let msg = why.join("; ");
    if allow_degenerate {
        eprintln!(
            "WARNING: degenerate ternary pack ({msg}) — allowed by --allow-degenerate-ternary"
        );
        return;
    }
    eprintln!("error: refusing to write a degenerate ternary model: {msg}.");
    eprintln!(
        "       Set HIPFIRE_ALLOW_DEGENERATE_TERNARY=1 to write it anyway (it will \
         not serve coherently)."
    );
    std::process::exit(3);
}

/// Per-input-column importance weights for a requantized tensor.
///
/// Returns the `--imatrix` row for `name` when one was supplied and its length
/// is a usable multiple of the 128 group, else all-ones (a pure unweighted-MSE
/// scale search). The GPTQ packers slice this as
/// `col_weights[(b % blocks_per_row) * 128 ..][..128]`, so the length must be a
/// multiple of 128 — an all-ones length-128 vector makes every block reuse the
/// same (uniform) weights, which is exactly the no-imatrix behaviour.

fn lowbit_ptq_gate(format: GgufFormat, allowed: bool) -> Result<(), String> {
    if allowed || !matches!(format, GgufFormat::Ternary | GgufFormat::Binary) {
        return Ok(());
    }
    let bpw = if matches!(format, GgufFormat::Binary) {
        "1.14"
    } else {
        "2.125"
    };
    Err(format!(
        "error: --input <.hfq> --format {} re-quantizes an ordinary checkpoint into a \
         UNIFORM {bpw}-bpw level set, which is a measured collapse — not a supported \
         build.\n\
         \n\
         Measured on qwen3.6-27b (KLD vs the mq4 teacher, 8 chunks):\n\
         \x20 ternary uniform            2.125 bpw  KLD 5.10  PPL 1436   (token soup)\n\
         \x20 ternary + AWQ imatrix      2.125 bpw  KLD 2.24  PPL   86.6 (immediate EOS)\n\
         \x20 mq2lloyd (non-uniform)     2.25  bpw  KLD 0.61  PPL   17.0 (usable)\n\
         \x20 PrismML Bonsai ternary     2.125 bpw  KLD 0.54  PPL   16.7\n\
         \n\
         The bit budget is NOT the problem — the fixed uniform level set is. Q2_0/Q1_0 \
         leave the encoder only the block scale to choose, and no scale search or \
         importance weighting recovers it.\n\
         \n\
         For ~2 bpw from an ordinary checkpoint use --format mq2lloyd instead. \
         Ternary/binary ship coherently as byte-verbatim passthrough of an \
         already-transformed source (PrismML Bonsai Q2_0/Q1_0) — convert that GGUF \
         directly.\n\
         \n\
         To do it anyway for research, pass --allow-lowbit-ptq or set \
         HIPFIRE_ALLOW_LOWBIT_PTQ=1.",
        format.label(),
    ))
}
