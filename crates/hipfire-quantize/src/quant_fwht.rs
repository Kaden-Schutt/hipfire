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

// ─── HFQ4-G256 Quantization ─────────────────────────────────────────────────

/// Quantize F32 weights to HFQ4-G256: flat 4-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][128B nibbles] = 136 bytes per 256 weights (0.531 B/w).
/// 18 VGPRs, 100% occupancy on RDNA1. Beats Q4_K at all matrix sizes.
/// CPU-side FWHT (Walsh-Hadamard Transform) on a 256-element group.
/// Matches the GPU-side fwht_forward_256 in turbo_common: signs1 → butterfly → scale → signs2.
pub(crate) fn cpu_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert!(x.len() == 256);
    for i in 0..256 {
        x[i] *= signs1[i];
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
        x[i] *= scale * signs2[i];
    }
}

/// Generate FWHT sign table (matches engine's gen_fwht_signs).
pub(crate) fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (state >> 16) & 1 == 1 {
                1.0f32
            } else {
                -1.0f32
            }
        })
        .collect()
}

/// MagnumQuant HFQ4-G256: FWHT-rotated 4-bit quantization.
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

/// MagnumQuant MQ5-G256: FWHT-rotated 5-bit quantization.
/// 168 bytes/group = 8 B affine header (f32 scale + f32 min) + 160 B payload
/// (5 bits x 256 weights = 1280 bits). 5.25 bpw. Sits between MQ4 (136 B/group)
/// and MQ6 (200 B/group). The 5-bit codes cross byte boundaries: 8 values pack
/// into 5 bytes (8*5 = 40 bits). The rotation is baked into the weights; the
/// GEMV kernel rotates x instead of inverse-rotating w.
///
/// Pack layout (q0..q7 each 5-bit, clamped to 31), per 5-byte chunk:
///   b0 = q0        | (q1 << 5)              // q0[4:0], q1[2:0]
///   b1 = (q1 >> 3) | (q2 << 2) | (q3 << 7)  // q1[4:3], q2[4:0], q3[0]
///   b2 = (q3 >> 1) | (q4 << 4)              // q3[4:1], q4[3:0]
///   b3 = (q4 >> 4) | (q5 << 1) | (q6 << 6)  // q4[4], q5[4:0], q6[1:0]
///   b4 = (q6 >> 2) | (q7 << 3)              // q6[4:2], q7[4:0]
/// The HIP unpacker reverses this (read 5 bytes -> 8 codes), then
///   val = q * scale + min.
pub(crate) fn quantize_mq5g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 168; // 8 (scale+min) + 160 (packed 5-bit)
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
        let scale = if range > 0.0 { range / 31.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        // Pack 8 values per 5 bytes (8*5 = 40 bits). 256/8 = 32 chunks ->
        // 32*5 = 160 payload bytes. byte_off = 8 + (i/8)*5.
        for i in (0..256).step_by(8) {
            let q0 = (((group[i] - min_val) * inv_scale + 0.5) as u8).min(31);
            let q1 = (((group[i + 1] - min_val) * inv_scale + 0.5) as u8).min(31);
            let q2 = (((group[i + 2] - min_val) * inv_scale + 0.5) as u8).min(31);
            let q3 = (((group[i + 3] - min_val) * inv_scale + 0.5) as u8).min(31);
            let q4 = (((group[i + 4] - min_val) * inv_scale + 0.5) as u8).min(31);
            let q5 = (((group[i + 5] - min_val) * inv_scale + 0.5) as u8).min(31);
            let q6 = (((group[i + 6] - min_val) * inv_scale + 0.5) as u8).min(31);
            let q7 = (((group[i + 7] - min_val) * inv_scale + 0.5) as u8).min(31);

            let byte_off = 8 + (i / 8) * 5;
            output[out_off + byte_off] = q0 | (q1 << 5);
            output[out_off + byte_off + 1] = (q1 >> 3) | (q2 << 2) | (q3 << 7);
            output[out_off + byte_off + 2] = (q3 >> 1) | (q4 << 4);
            output[out_off + byte_off + 3] = (q4 >> 4) | (q5 << 1) | (q6 << 6);
            output[out_off + byte_off + 4] = (q6 >> 2) | (q7 << 3);
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
