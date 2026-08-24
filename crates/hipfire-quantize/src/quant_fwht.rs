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

pub(crate) const MQ4V2_GROUP_BYTES: usize = 136;
pub(crate) const MQ4C_GROUP_BYTES: usize = 136;
pub(crate) const MQ6V2_GROUP_BYTES: usize = 200;
pub(crate) const MQ5V2_GROUP_BYTES: usize = 168;

/// MQ4CG256 encoder — single-scale per-256 asymmetric, fp16 header, 136 B/group (pad layout).
///
/// For each 256-weight group after FWHT:
///   lo = min(group); hi = max(group)
///   step_f32 = (hi - lo) / 15; scale=f16(step); zero=f16(lo)
///   st=f32(scale); z=f32(zero); // round-tripped
///   for i in 0..256: q[i]=clamp(rint((w[i]-z)/st),0,15)
/// Degenerate group (hi==lo): scale=0, zero=f16(lo), all q=0.
///
/// Pad layout (per weight tensor with `m` rows, `gpr = K/256` groups/row, `n = m*gpr`):
///   per group, 136 B stride: `[0..4)` fp16 header, `[4..8)` zero padding, `[8..136)` 128 B nibbles.
pub(crate) fn quantize_mq4cg256(
    w: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    let group_size = 256;
    let n = w.len();
    assert!(k % 256 == 0, "MQ4CG256 requires K % 256 == 0, got K={k}");
    let gpr = k / 256;
    let total_groups = m * gpr;
    assert_eq!(n, m * k, "w.len() {} != m*k {}*{}={}", n, m, k, m * k);
    let expected_len = total_groups * MQ4C_GROUP_BYTES;
    assert_eq!(expected_len, m * gpr * 136, "total bytes must be m*gpr*136");
    assert_eq!(expected_len, total_groups * 136);
    let mut output = vec![0u8; expected_len];
    for b in 0..total_groups {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&w[start..end]);
        cpu_fwht_256(&mut group, signs1, signs2);
        let lo = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let step_f32 = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
        let sc_bits = if hi == lo { 0u16 } else { f32_to_f16(step_f32) };
        let z_bits = f32_to_f16(lo);
        let st = f16_to_f32(sc_bits);
        let z = f16_to_f32(z_bits);
        let degenerate = hi == lo || step_f32 == 0.0 || st == 0.0;
        let base = b * 136;
        output[base..base + 2].copy_from_slice(&sc_bits.to_le_bytes());
        output[base + 2..base + 4].copy_from_slice(&z_bits.to_le_bytes());
        if degenerate {
        } else {
            let inv = 1.0 / st;
            let mut q = [0u8; 256];
            for i in 0..256 {
                let qq = ((group[i] - z) * inv + 0.5).floor().clamp(0.0, 15.0) as u8;
                q[i] = qq;
            }
            for i in 0..128 {
                let lo_q = q[2 * i];
                let hi_q = q[2 * i + 1];
                output[base + 8 + i] = (lo_q & 0xF) | ((hi_q & 0xF) << 4);
            }
        }
    }
    assert_eq!(output.len(), m * gpr * 136);
    output
}

/// MQ4G256 v2 encoder — per-128 asymmetric, fp16 header, byte-identical nibble payload to v1.
///
/// See `docs/quant-formats/mq4-v2.md` §3. For each 256-weight group after FWHT:
/// for each half `h` in {0,1} over weights `128h..128h+127`:
///   lo = min(w in half); hi = max(w in half)
///   step_f32 = (hi-lo)/15; scale[h]=f16(step); zero[h]=f16(lo)
///   st=f32(scale[h]); z=f32(zero[h]); // round-tripped
///   for i in half: q[i]=clamp(rint((w[i]-z)/st),0,15)
/// Degenerate half (hi==lo): scale=0, zero=f16(lo), all q=0.
/// Header: [0..2) fp16 scale h0, [2..4) fp16 zero h0, [4..6) fp16 scale h1, [6..8) fp16 zero h1, [8..136) nibbles.
pub(crate) fn quantize_mq4g256v2(
    w: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    let _ = (m, k);
    let group_size = 256;
    let block_bytes = MQ4V2_GROUP_BYTES;
    let n = w.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];
    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&w[start..end]);
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
            let step_f32 = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
            let sc_bits = f32_to_f16(step_f32);
            let z_bits = f32_to_f16(lo);
            let sc_bits = if hi == lo { 0u16 } else { sc_bits };
            scales[h] = sc_bits;
            zeros[h] = z_bits;
            let st = f16_to_f32(sc_bits);
            let z = f16_to_f32(z_bits);
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
                    let qq = ((v - z) * inv + 0.5).floor().clamp(0.0, 15.0) as u8;
                    q[off + i] = qq;
                }
            }
        }
        for i in 0..128 {
            let lo_q = q[2 * i];
            let hi_q = q[2 * i + 1];
            output[out_off + 8 + i] = (lo_q & 0xF) | ((hi_q & 0xF) << 4);
        }
    }
    output
}
/// MQ6G256V2 encoder — per-128 asymmetric fp16 header, neutral-size GEMM/GEMV.
///
/// Layout per 256-weight group: `[0..2) fp16 s0,[2..4) fp16 z0,[4..6) fp16 s1,[6..8) fp16 z1,[8..200) 192B packed 6-bit`.
/// Payload unchanged from MQ6G256: 4 values per 3 bytes (q0 | q1<<6, etc).
/// Half 0 covers q[0..128), half 1 q[128..256); reconstruction q*f32(s[h])+f32(z[h]).
/// Encoder round-trips s/z through fp16 before selecting q; degenerate half: scale=0, zero=f16(lo), q=0.
/// K%256 required.
pub(crate) fn quantize_mq6g256v2(
    w: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    assert!(k % 256 == 0, "MQ6G256V2 requires K % 256 == 0, got K={k}");
    let n = w.len();
    assert_eq!(n, m * k, "w.len() {} != m*k {}*{}={}", n, m, k, m * k);
    let gpr = k / 256;
    let total_groups = m * gpr;
    let block_bytes = MQ6V2_GROUP_BYTES;
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
            let step_f32 = if hi > lo { (hi - lo) / 63.0 } else { 0.0 };
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
                    let qq = ((v - z) * inv + 0.5).floor().clamp(0.0, 63.0) as u8;
                    q[off + i] = qq;
                }
            }
        }
        for i in (0..256).step_by(4) {
            let bo = out_off + 8 + (i / 4) * 3;
            let q0 = q[i] & 63;
            let q1 = q[i + 1] & 63;
            let q2 = q[i + 2] & 63;
            let q3 = q[i + 3] & 63;
            output[bo] = q0 | (q1 << 6);
            output[bo + 1] = (q1 >> 2) | (q2 << 4);
            output[bo + 2] = (q2 >> 4) | (q3 << 2);
        }
    }
    output
}
/// MQ5G256V2 encoder — per-128 asymmetric fp16 header, neutral-size.
///
/// Layout per 256-weight group: `[0..2) fp16 s0,[2..4) fp16 z0,[4..6) fp16 s1,[6..8) fp16 z1,[8..168) 160B packed 5-bit`.
/// Payload unchanged from MQ5G256: 8 values per 5 bytes.
pub(crate) fn quantize_mq5g256v2(
    w: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    assert!(k % 256 == 0, "MQ5G256V2 requires K % 256 == 0, got K={k}");
    let n = w.len();
    assert_eq!(n, m * k, "w.len() {} != m*k {}*{}={}", n, m, k, m * k);
    let gpr = k / 256;
    let total_groups = m * gpr;
    let block_bytes = MQ5V2_GROUP_BYTES;
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
            let step_f32 = if hi > lo { (hi - lo) / 31.0 } else { 0.0 };
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
                    let qq = ((v - z) * inv + 0.5).floor().clamp(0.0, 31.0) as u8;
                    q[off + i] = qq;
                }
            }
        }
        for i in (0..256).step_by(8) {
            let bo = out_off + 8 + (i / 8) * 5;
            let q0 = q[i] & 31;
            let q1 = q[i + 1] & 31;
            let q2 = q[i + 2] & 31;
            let q3 = q[i + 3] & 31;
            let q4 = q[i + 4] & 31;
            let q5 = q[i + 5] & 31;
            let q6 = q[i + 6] & 31;
            let q7 = q[i + 7] & 31;
            output[bo] = q0 | (q1 << 5);
            output[bo + 1] = (q1 >> 3) | (q2 << 2) | (q3 << 7);
            output[bo + 2] = (q3 >> 1) | (q4 << 4);
            output[bo + 3] = (q4 >> 4) | (q5 << 1) | (q6 << 6);
            output[bo + 4] = (q6 >> 2) | (q7 << 3);
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
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mq4g256v2_roundtrip_shape() {
        let m = 2usize;
        let k = 256usize;
        let w: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin()).collect();
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let blob = quantize_mq4g256v2(&w, m, k, &s1, &s2);
        let n_blocks = (w.len() + 255) / 256;
        assert_eq!(blob.len(), n_blocks * MQ4V2_GROUP_BYTES);
        assert_eq!(MQ4V2_GROUP_BYTES, 136);
        // header 8B + 128B payload per block, nibbles present
        assert!(blob.len() >= 136);
    }

    #[test]
    fn mq4cg256_pad_layout_136() {
        let m = 2usize;
        let k = 512usize;
        let w: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.5).collect();
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let blob = quantize_mq4cg256(&w, m, k, &s1, &s2);
        let gpr = k / 256;
        assert_eq!(blob.len(), m * gpr * MQ4C_GROUP_BYTES);
        assert_eq!(MQ4C_GROUP_BYTES, 136);
        // pad bytes [4..8) must be zero
        for g in 0..m * gpr {
            let base = g * 136;
            assert_eq!(
                &blob[base + 4..base + 8],
                &[0u8; 4],
                "pad bytes must be zero at group {g}"
            );
            // header scale/zero are fp16 values — at least not all zero for non-degenerate
            // (first group should have non-zero scale for this synthetic data)
        }
    }

    #[test]
    fn mq4_flags_map_to_quant_types() {
        // codec mapping reachability: labels map via QuantType and GgufFormat
        use crate::hfq::QuantType;
        use crate::pipeline_gguf::GgufFormat;
        assert_eq!(GgufFormat::from_flag("mq4"), Some(GgufFormat::Mq4V2));
        assert_eq!(GgufFormat::from_flag("mq4v2"), Some(GgufFormat::Mq4V2));
        assert_eq!(GgufFormat::from_flag("mq4c"), Some(GgufFormat::Mq4C));
        assert_eq!(GgufFormat::from_flag("mq4v1"), Some(GgufFormat::Mq4));
        assert_eq!(GgufFormat::from_flag("mq4g256v2"), Some(GgufFormat::Mq4V2));
        assert_eq!(GgufFormat::from_flag("mq4cg256"), Some(GgufFormat::Mq4C));
        assert_eq!(QuantType::from_u8(44), Some(QuantType::MQ4G256V2));
        assert_eq!(QuantType::from_u8(45), Some(QuantType::MQ4CG256));
        assert_eq!(QuantType::MQ4G256V2 as u8, 44);
        assert_eq!(QuantType::MQ4CG256 as u8, 45);
        // new V2 family
        assert_eq!(GgufFormat::from_flag("mq6v2"), Some(GgufFormat::Mq6V2));
        assert_eq!(GgufFormat::from_flag("mq5v2"), Some(GgufFormat::Mq5V2));
        assert_eq!(GgufFormat::from_flag("mq3v2"), Some(GgufFormat::Mq3V2));
        assert_eq!(GgufFormat::from_flag("mq2v2"), Some(GgufFormat::Mq2V2));
        assert_eq!(GgufFormat::from_flag("mq6g256v2"), Some(GgufFormat::Mq6V2));
        assert_eq!(GgufFormat::from_flag("mq2g256v2"), Some(GgufFormat::Mq2V2));
        assert_eq!(QuantType::from_u8(47), Some(QuantType::MQ6G256V2));
        assert_eq!(QuantType::from_u8(48), Some(QuantType::MQ5G256V2));
        assert_eq!(QuantType::from_u8(49), Some(QuantType::MQ3G256V2));
        assert_eq!(QuantType::from_u8(50), Some(QuantType::MQ2G256V2));
        assert_eq!(QuantType::MQ6G256V2 as u8, 47);
        assert_eq!(QuantType::MQ5G256V2 as u8, 48);
        assert_eq!(QuantType::MQ3G256V2 as u8, 49);
        assert_eq!(QuantType::MQ2G256V2 as u8, 50);
        // unknown remains rejected
        assert_eq!(QuantType::from_u8(46), None);
        assert_eq!(QuantType::from_u8(51), None);
        assert_eq!(QuantType::from_u8(255), None);
    }

    #[test]
    fn mq6v2_mq5v2_wire_layout() {
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        // byte count: m*ceil(k/256)*B
        for (m, k, b) in [
            (1usize, 256usize, MQ6V2_GROUP_BYTES),
            (2, 512, MQ6V2_GROUP_BYTES),
            (4, 1024, MQ6V2_GROUP_BYTES),
        ] {
            let w = vec![0.1f32; m * k];
            let blob = quantize_mq6g256v2(&w, m, k, &s1, &s2);
            assert_eq!(blob.len(), m * (k / 256) * b, "mq6v2 bytes m={m} k={k}");
            assert_eq!(b, 200);
        }
        for (m, k, b) in [
            (1usize, 256usize, MQ5V2_GROUP_BYTES),
            (3, 768, MQ5V2_GROUP_BYTES),
        ] {
            let w = vec![-0.2f32; m * k];
            let blob = quantize_mq5g256v2(&w, m, k, &s1, &s2);
            assert_eq!(blob.len(), m * (k / 256) * b, "mq5v2 bytes m={m} k={k}");
            assert_eq!(b, 168);
        }
        // fp16 header positions, payload offset, degenerate halves
        let m = 2;
        let k = 256;
        // degenerate: all zeros => both halves scale=0, zero=f16(lo), payload q=0
        let w_zero = vec![0.0f32; m * k];
        let blob = quantize_mq6g256v2(&w_zero, m, k, &s1, &s2);
        for g in 0..m * (k / 256) {
            let base = g * MQ6V2_GROUP_BYTES;
            let s0 = u16::from_le_bytes([blob[base], blob[base + 1]]);
            let z0 = u16::from_le_bytes([blob[base + 2], blob[base + 3]]);
            let s1b = u16::from_le_bytes([blob[base + 4], blob[base + 5]]);
            let z1 = u16::from_le_bytes([blob[base + 6], blob[base + 7]]);
            assert_eq!(s0, 0, "degenerate scale half0 must be 0");
            assert_eq!(s1b, 0, "degenerate scale half1 must be 0");
            // zero is f16(lo) where lo==0 after FWHT of zeros => 0
            assert_eq!(z0, f32_to_f16(0.0));
            assert_eq!(z1, f32_to_f16(0.0));
            // payload offset 8 must be zeros
            assert!(
                blob[base + 8..base + MQ6V2_GROUP_BYTES]
                    .iter()
                    .all(|&b| b == 0),
                "degenerate payload must be 0"
            );
        }
        // non-degenerate: ensure header bytes are valid fp16 and payload non-zero
        let w_rand: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.007).sin() * 2.0).collect();
        let blob = quantize_mq5g256v2(&w_rand, m, k, &s1, &s2);
        for g in 0..m * (k / 256) {
            let base = g * MQ5V2_GROUP_BYTES;
            let s0 = u16::from_le_bytes([blob[base], blob[base + 1]]);
            let s1b = u16::from_le_bytes([blob[base + 4], blob[base + 5]]);
            // At least one half must be non-degenerate for this synthetic data
            assert!(s0 != 0 || s1b != 0, "expected non-degenerate header");
            // payload after 8 must contain packed codes (not all zero)
            // 5-bit payload is 160B, check not all zero
            let payload = &blob[base + 8..base + MQ5V2_GROUP_BYTES];
            assert_eq!(payload.len(), MQ5V2_GROUP_BYTES - 8);
            assert!(payload.iter().any(|&b| b != 0));
        }
    }
}
