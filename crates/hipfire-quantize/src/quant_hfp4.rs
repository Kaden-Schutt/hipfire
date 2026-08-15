// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.


#![allow(dead_code, unused_imports, unused_variables, non_snake_case, clippy::all)]
use crate::quant_fwht::{cpu_fwht_256, gen_fwht_signs};
use crate::dequant::e2m1_to_f32;

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
    assert!(
        k % 256 == 0,
        "HFP4G32 v1 requires K%256==0 (gemv_hfp4g32 kernel constraint; v2 will lift to K%32==0), got K={}",
        k
    );
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

// ─── mfp4+P — mfp4 with E4M3 (non-power-of-2) per-block scale ────────────────
//
// mfp4+P = mfp4 (E2M1 4-bit element + FP16 per-row scale + offline FWHT) but the
// per-32-block scale byte is an **E4M3 FP8** encoding of the EXACT scale ratio
// s = block_max_normalized / 6.0, instead of mfp4's UE8M0 ceil-log2 (power-of-2-
// only). E4M3 carries ~3 mantissa bits, so each block's E2M1 [-6,6] grid is more
// fully used (UE8M0 wastes up to ~1 bit by rounding the scale up to the next
// power of 2). Byte layout is BYTE-IDENTICAL to mfp4 (16-B header + n_blocks×17 B,
// NO codebook prefix); the only change is the meaning of the per-block scale byte.
//
// Reconstruction:  value = row_scale_a · e4m3_decode(scale_byte) · e2m1_to_f32(nibble)

/// Decode an UNSIGNED E4M3 (FP8, 4 exp bias 7 + 3 mantissa) byte to f32.
/// Bit-identical to `e4m3_to_f32` for sign=0, restated here standalone so the
/// mfp4+P scale path is self-contained and matches the gfx942 kernel decode
/// exactly. exp=0 → subnormal 2^-6·(mant/8); exp=15,mant=7 → NaN (never emitted
/// by our round-up encoder, which clamps to 448); otherwise 2^(exp-7)·(1+mant/8).
#[inline]
pub(crate) fn e4m3_scale_decode(byte: u8) -> f32 {
    let exp = ((byte >> 3) & 0xf) as i32;
    let mant = (byte & 0x7) as u32;
    if exp == 0 {
        // subnormal (incl. zero): 2^-6 * (mant/8)
        return (2.0f32).powi(-6) * (mant as f32) / 8.0;
    }
    if exp == 0xf && mant == 7 {
        // E4M3's single NaN code — our encoder never emits it; decode defensively
        // to the max finite (448) so a stray byte cannot poison a block.
        return 448.0;
    }
    (2.0f32).powi(exp - 7) * (1.0 + (mant as f32) / 8.0)
}

/// Encode a NON-NEGATIVE f32 scale `s` to an UNSIGNED E4M3 byte, ROUNDED UP
/// (ceil) to the nearest representable E4M3 value ≥ s. Round-up mirrors mfp4's
/// UE8M0 ceil intent: the block scale must COVER block_max so e2m1_round never
/// clips the block max above the [-6,6] E2M1 grid. Sign bit is always 0.
///
/// The decode side (`e4m3_scale_decode` / the gfx942 kernel) MUST be bit-identical;
/// this function is defined as "smallest E4M3 code whose DECODED value ≥ s", which
/// is round-trip-exact by construction (we search the decode, not a formula).
///
/// Representable unsigned E4M3 (exp 0..15, bias 7, 3 mantissa):
///   exp=0  : subnormals 0, 2^-9, 2·2^-9 … 7·2^-9   (2^-6·mant/8)
///   exp1..14, exp15&mant<7 : 2^(exp-7)·(1+mant/8)
///   exp=15,mant=7 : NaN (excluded)  → max finite = 2^8·1.875 = 448
#[inline]
pub(crate) fn e4m3_scale_encode_roundup(s: f32) -> u8 {
    // Non-finite / non-positive guard. s<=0 → smallest code (0x00 == +0.0).
    if !(s > 0.0) {
        return 0x00;
    }
    if s >= 448.0 {
        // Saturate to the largest finite E4M3 (exp=15, mant=6 → 0x7E).
        return 0x7E;
    }
    // Find the smallest code in [0x00, 0x7E] (sign=0, NaN 0x7F excluded) whose
    // decoded value is ≥ s. Codes are monotonically non-decreasing in `byte`
    // for sign=0 across the exp/mantissa range (standard FP8 ordering), so a
    // forward scan returns the ceil code. Exhaustive 127-entry scan is trivially
    // cheap (called once per 32-element block at quant time, offline).
    for code in 0u8..=0x7E {
        if e4m3_scale_decode(code) >= s {
            return code;
        }
    }
    0x7E
}

/// Quantize one row of K FP32 weights to mfp4+P byte format. Byte-identical to
/// `quantize_hfp4g32_row` EXCEPT the per-block scale byte is an E4M3 round-up
/// encoding of the exact ratio s = block_max_normalized / 6.0 (NOT UE8M0
/// ceil-log2). Returns 16-B header + (K/32) × 17-B blocks.
pub(crate) fn quantize_mfp4g32_p_row(row: &[f32]) -> Vec<u8> {
    assert!(
        row.len() % 32 == 0,
        "mfp4+P requires K%32 == 0, got K={}",
        row.len()
    );
    let k = row.len();
    let n_blocks = k / 32;
    let row_bytes = 16 + n_blocks * 17;
    let mut out = vec![0u8; row_bytes];

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

    out[0..2].copy_from_slice(&f32_to_f16(row_scale_a).to_le_bytes());
    out[2..4].copy_from_slice(&0u16.to_le_bytes());
    out[4..6].copy_from_slice(&(n_blocks as u16).to_le_bytes());
    out[6] = 0x05; // format_flags: bit0 + bits2-3=01 = offline FWHT (same as mfp4)
    out[7] = 0u8;

    for b in 0..n_blocks {
        let block = &row[b * 32..b * 32 + 32];
        let block_max_abs = block.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
        let block_max_normalized = block_max_abs * inv_row_scale;

        // Exact scale ratio s = block_max_normalized / 6.0. E4M3 round-UP so the
        // decoded scale covers block_max (no clip above the [-6,6] E2M1 grid).
        // Empty block → s=0 → code 0x00 (decodes to +0.0; inv → 0 → all nibbles 0).
        let s = if block_max_normalized > 0.0 {
            block_max_normalized / 6.0
        } else {
            0.0
        };
        let scale_byte = e4m3_scale_encode_roundup(s);

        // Reconstruct the ACTUAL decoded scale (round-up may exceed s) and invert.
        let block_scale_factor = e4m3_scale_decode(scale_byte);
        let inv_block_scale = if block_scale_factor > 0.0 {
            1.0 / block_scale_factor
        } else {
            0.0
        };

        let payload_off = 16 + b * 17;
        out[payload_off] = scale_byte;
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

/// mfp4+P 2D: FWHT-rotate the tensor (same signs as mfp4), then per-row
/// `quantize_mfp4g32_p_row`. Byte layout identical to mfp4 (NO prefix). Stamps
/// format_flags=0x05 per row (handled inside the row fn).
pub(crate) fn quantize_mfp4g32_p_2d(
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
        "mfp4+P requires k % 256 == 0 for 256-element FWHT, got k={}",
        k
    );
    let row_bytes = 16 + 17 * (k / 32);
    let mut out = Vec::with_capacity(m * row_bytes);
    let mut row_buf = vec![0.0f32; k];
    for r in 0..m {
        row_buf.copy_from_slice(&f32_data[r * k..(r + 1) * k]);
        for seg in 0..(k / 256) {
            cpu_fwht_256(&mut row_buf[seg * 256..(seg + 1) * 256], signs1, signs2);
        }
        out.extend_from_slice(&quantize_mfp4g32_p_row(&row_buf));
    }
    debug_assert_eq!(out.len(), m * row_bytes);
    out
}

/// CPU reference dequant for mfp4+P. Returns row-major f32 [m*k] in the ROTATED
/// domain. Bit-exact mirror of the gfx942 gemv/dequant kernels' E4M3 decode.
#[allow(dead_code)]
pub(crate) fn dequant_mfp4g32_p(packed: &[u8], m: usize, k: usize) -> Vec<f32> {
    let row_bytes = 16 + 17 * (k / 32);
    assert_eq!(packed.len(), m * row_bytes, "mfp4+P size mismatch");
    let mut out = vec![0.0f32; m * k];
    for r in 0..m {
        let base = r * row_bytes;
        let row_scale_a = f16_to_f32(u16::from_le_bytes([packed[base], packed[base + 1]]));
        for b in 0..(k / 32) {
            let po = base + 16 + b * 17;
            let scale = row_scale_a * e4m3_scale_decode(packed[po]);
            for i in 0..16 {
                let byte = packed[po + 1 + i];
                out[r * k + b * 32 + 2 * i] = scale * e2m1_to_f32(byte & 0x0F);
                out[r * k + b * 32 + 2 * i + 1] = scale * e2m1_to_f32((byte >> 4) & 0x0F);
            }
        }
    }
    out
}
