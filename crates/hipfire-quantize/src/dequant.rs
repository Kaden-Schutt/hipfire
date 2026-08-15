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

pub(crate) fn to_f32(data: &[u8], dtype: &str) -> Vec<f32> {
    match dtype {
        "F16" => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "BF16" => data
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "F32" => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => panic!("unsupported dtype: {other}"),
    }
}

// ─── FP8 E4M3 + UE8M0-scale dequant (DeepSeek V4 Flash) ─────────────────────
//
// DeepSeek V4 ships its quantized weights as paired safetensors entries:
//   <name>.weight  : I8 raw bytes, each byte one FP8 E4M3 value
//   <name>.scale   : F8_E8M0 raw bytes, each byte one UE8M0 exponent
//
// The block shape on DeepSeek V4-shipped checkpoints is [1, 16] (per-row, 16-col
// groups) — i.e. scale shape `[R, C/16]` for weight shape `[R, C]` — even
// though the `quantization_config.weight_block_size` in `config.json`
// reads `[128, 128]`. We verify the implied block from the actual scale
// shape rather than the config to avoid being misled.
//
// E4M3 format (1 sign + 4 exp + 3 mant, bias=7):
//   - exp=0, mant=0      → ±0
//   - exp=0, mant!=0     → denormal: (-1)^s · 2^-6 · (mant/8)
//   - exp=15, mant=7     → NaN (only one NaN code in E4M3)
//   - otherwise normal:  (-1)^s · 2^(exp-7) · (1 + mant/8)
//
// UE8M0 format (8-bit unsigned exponent only, no sign, no mantissa):
//   scale = 2^(byte - 127)
//
// Returns f32 in row-major order matching `weight_shape`.

pub(crate) fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if (byte & 0x80) != 0 { -1.0 } else { 1.0 };
    let exp = ((byte >> 3) & 0xf) as i32;
    let mant = (byte & 0x7) as f32;
    if exp == 0xf && mant == 7.0 {
        // E4M3's single NaN code — treat as 0 for quant purposes (clean
        // bytes flagged elsewhere; downstream MQ-family quant has no
        // NaN handling and would emit garbage).
        return 0.0;
    }
    if exp == 0 {
        if mant == 0.0 {
            return 0.0;
        }
        return sign * (2.0f32.powi(-6)) * (mant / 8.0);
    }
    sign * (2.0f32.powi(exp - 7)) * (1.0 + mant / 8.0)
}

#[inline]
pub(crate) fn ue8m0_to_scale(byte: u8) -> f32 {
    // 2^(exp - 127). Cheap: shift into f32's exponent field directly.
    // byte=127 → 1.0, byte=0 → 2^-127 (subnormal range — fine, we return 0
    // implicitly through f32 rounding), byte=255 → +inf (won't appear on
    // well-formed checkpoints; if it does we propagate inf and the
    // downstream MQ quant will produce extreme outputs detectable in QA).
    2.0f32.powi(byte as i32 - 127)
}

/// Helper for the main quantize loop: convert one tensor's raw bytes to
/// f32, transparently handling DeepSeek V4's FP8 E4M3 + UE8M0-scale pairs.
///
/// If `meta.dtype == "I8"` and a scale sibling is registered in
/// `fp8_scale_for[weight_name]`, dequant the pair. Otherwise fall back
/// to `to_f32(data, dtype)`.
pub(crate) fn tensor_to_f32_with_optional_fp8_scale(
    name: &str,
    raw_data: &[u8],
    meta: &TensorMeta,
    fp8_scale_for: &HashMap<String, (usize, String)>,
    st_files: &[SafetensorsFile],
) -> Vec<f32> {
    // FP8 E4M3 + UE8M0 paired storage (DeepSeek V4). The dtype tag is either
    // `I8` (older safetensors writer) or `F8_E4M3` (newer); both
    // store identical E4M3 bytes, so the dequant math is the same.
    if (meta.dtype == "I8" || meta.dtype == "F8_E4M3") && fp8_scale_for.contains_key(name) {
        let (sfi, sname) = &fp8_scale_for[name];
        let (smeta, sbytes) = st_files[*sfi]
            .tensor_data(sname)
            .unwrap_or_else(|| panic!("FP8 scale tensor missing: {sname}"));
        if smeta.dtype == "F8_E8M0" {
            return dequantize_e4m3_ue8m0_to_f32(raw_data, &meta.shape, sbytes, &smeta.shape);
        } else if smeta.dtype == "F32" {
            // MiniMax-M2: e4m3 + F32 block-[128,128] weight_scale_inv (multiply).
            return dequantize_e4m3_f32scale_to_f32(raw_data, &meta.shape, sbytes, &smeta.shape);
        } else {
            panic!(
                "expected F8_E8M0 or F32 scale for {name}, got {}",
                smeta.dtype
            );
        }
    }
    if meta.dtype == "I8" {
        panic!(
            "tensor {name} has dtype I8 but no .scale sibling registered \
                — unexpected on a non-DeepSeek V4 checkpoint."
        );
    }
    to_f32(raw_data, &meta.dtype)
}

/// Convert one E2M1 nibble (4-bit FP: 1 sign + 2 exp + 1 mantissa, bias=1) to f32.
///
/// E2M1 codes (signed magnitude on the 3 low bits, high bit is sign):
///   nibble & 0x7 → magnitude  → value
///   0  → 0          → 0.0
///   1  → denorm 0.5 → 0.5
///   2  → normal 1.0 → 1.0
///   3  → normal 1.5 → 1.5
///   4  → normal 2.0 → 2.0
///   5  → normal 3.0 → 3.0
///   6  → normal 4.0 → 4.0
///   7  → normal 6.0 → 6.0
/// Sign bit: bit 3 (0x8).
///
/// Total range: ±6.0. Per OCP MX spec (FP4 E2M1).
#[inline]
pub(crate) fn e2m1_to_f32(nibble: u8) -> f32 {
    // Lookup table for the 8 magnitude codes; sign is applied after.
    pub(crate) const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let n = (nibble & 0x0f) as usize;
    let mag = MAG[n & 0x7];
    if (n & 0x8) != 0 {
        -mag
    } else {
        mag
    }
}

/// Dequantize a paired E2M1 weight + UE8M0 scale tensor to f32.
///
/// `storage_shape` is the byte-shape from safetensors: [rows, cols_stored]
/// where `cols_stored = logical_cols / 2` (two E2M1 nibbles per byte; low
/// nibble is the even logical column, high nibble is the odd column).
/// `scale_shape` is [scale_rows, scale_cols]; the implied block size in
/// logical-element units is [rows / scale_rows, logical_cols / scale_cols].
/// Per DeepSeek V4 spec (model.py:132-137): block 32 along logical K → scale_cols
/// = logical_cols / 32.
///
/// Returns row-major f32 of LOGICAL shape, length = rows * cols_stored * 2.
pub(crate) fn dequantize_e2m1_ue8m0_to_f32(
    weight_bytes: &[u8],
    storage_shape: &[usize],
    scale_bytes: &[u8],
    scale_shape: &[usize],
) -> (Vec<f32>, Vec<usize>) {
    assert_eq!(
        storage_shape.len(),
        2,
        "expected 2D storage shape, got {:?}",
        storage_shape
    );
    assert_eq!(
        scale_shape.len(),
        2,
        "expected 2D scale shape, got {:?}",
        scale_shape
    );
    let (rows, cols_stored) = (storage_shape[0], storage_shape[1]);
    let logical_cols = cols_stored * 2;
    let (sr, sc) = (scale_shape[0], scale_shape[1]);
    assert_eq!(
        weight_bytes.len(),
        rows * cols_stored,
        "FP4 weight byte count mismatch"
    );
    assert_eq!(scale_bytes.len(), sr * sc, "FP4 scale byte count mismatch");
    assert!(
        rows % sr == 0 && logical_cols % sc == 0,
        "FP4 scale shape {:?} doesn't tile logical weight shape [{}, {}]",
        scale_shape,
        rows,
        logical_cols
    );
    let block_rows = rows / sr;
    let block_cols_logical = logical_cols / sc;

    let mut out = vec![0.0f32; rows * logical_cols];
    for sr_i in 0..sr {
        for sc_j in 0..sc {
            let scale = ue8m0_to_scale(scale_bytes[sr_i * sc + sc_j]);
            for di in 0..block_rows {
                let r = sr_i * block_rows + di;
                for dj in 0..block_cols_logical {
                    let c = sc_j * block_cols_logical + dj;
                    // c is the LOGICAL column. Byte storing it sits at
                    // (c / 2); low nibble for even c, high nibble for odd.
                    let byte = weight_bytes[r * cols_stored + (c / 2)];
                    let nibble = if (c & 1) == 0 { byte & 0x0f } else { byte >> 4 };
                    out[r * logical_cols + c] = e2m1_to_f32(nibble) * scale;
                }
            }
        }
    }
    (out, vec![rows, logical_cols])
}

/// Dequantize a paired E4M3 weight + UE8M0 scale tensor to f32.
///
/// `weight_shape` is the LOGICAL [rows, cols] of the weight matrix.
/// `scale_shape` is [scale_rows, scale_cols]; the implied block size is
/// [weight_rows / scale_rows, weight_cols / scale_cols].
///
/// Returns row-major f32, length = rows * cols.
pub(crate) fn dequantize_e4m3_ue8m0_to_f32(
    weight_bytes: &[u8],
    weight_shape: &[usize],
    scale_bytes: &[u8],
    scale_shape: &[usize],
) -> Vec<f32> {
    assert_eq!(
        weight_shape.len(),
        2,
        "expected 2D weight, got {:?}",
        weight_shape
    );
    assert_eq!(
        scale_shape.len(),
        2,
        "expected 2D scale,  got {:?}",
        scale_shape
    );
    let (rows, cols) = (weight_shape[0], weight_shape[1]);
    let (sr, sc) = (scale_shape[0], scale_shape[1]);
    assert_eq!(
        weight_bytes.len(),
        rows * cols,
        "weight byte count mismatch"
    );
    assert_eq!(scale_bytes.len(), sr * sc, "scale  byte count mismatch");
    assert!(
        rows % sr == 0 && cols % sc == 0,
        "scale shape {:?} doesn't tile weight shape {:?}",
        scale_shape,
        weight_shape
    );
    let block_rows = rows / sr;
    let block_cols = cols / sc;

    let mut out = vec![0.0f32; rows * cols];
    // Each (sr_i, sc_j) scale governs the block weight[sr_i*block_rows .. (sr_i+1)*block_rows,
    //                                                  sc_j*block_cols .. (sc_j+1)*block_cols].
    for sr_i in 0..sr {
        for sc_j in 0..sc {
            let scale = ue8m0_to_scale(scale_bytes[sr_i * sc + sc_j]);
            for di in 0..block_rows {
                let r = sr_i * block_rows + di;
                for dj in 0..block_cols {
                    let c = sc_j * block_cols + dj;
                    let b = weight_bytes[r * cols + c];
                    out[r * cols + c] = e4m3_to_f32(b) * scale;
                }
            }
        }
    }
    out
}

/// Dequantize FP8 E4M3 weights paired with an F32 block-[128,128]
/// `weight_scale_inv` (MiniMax-M2 / DeepSeek-V3 fp8 block quant). Dequant is
/// MULTIPLY: `out = e4m3_to_f32(b) * scale` (the stored scale ≈ amax/448 per
/// block, verified ~5e-4 on the real checkpoint). Scale tile is [rows/sr, cols/sc]
/// = [128, 128] on MiniMax.
pub(crate) fn dequantize_e4m3_f32scale_to_f32(
    weight_bytes: &[u8],
    weight_shape: &[usize],
    scale_bytes: &[u8],
    scale_shape: &[usize],
) -> Vec<f32> {
    assert_eq!(
        weight_shape.len(),
        2,
        "expected 2D weight, got {:?}",
        weight_shape
    );
    assert_eq!(
        scale_shape.len(),
        2,
        "expected 2D scale, got {:?}",
        scale_shape
    );
    let (rows, cols) = (weight_shape[0], weight_shape[1]);
    let (sr, sc) = (scale_shape[0], scale_shape[1]);
    assert_eq!(
        weight_bytes.len(),
        rows * cols,
        "weight byte count mismatch"
    );
    assert_eq!(
        scale_bytes.len(),
        sr * sc * 4,
        "f32 scale byte count mismatch"
    );
    assert!(
        rows % sr == 0 && cols % sc == 0,
        "scale shape {:?} doesn't tile weight shape {:?}",
        scale_shape,
        weight_shape
    );
    let block_rows = rows / sr;
    let block_cols = cols / sc;
    let mut out = vec![0.0f32; rows * cols];
    for sr_i in 0..sr {
        for sc_j in 0..sc {
            let so = (sr_i * sc + sc_j) * 4;
            let scale = f32::from_le_bytes([
                scale_bytes[so],
                scale_bytes[so + 1],
                scale_bytes[so + 2],
                scale_bytes[so + 3],
            ]);
            for di in 0..block_rows {
                let r = sr_i * block_rows + di;
                for dj in 0..block_cols {
                    let c = sc_j * block_cols + dj;
                    out[r * cols + c] = e4m3_to_f32(weight_bytes[r * cols + c]) * scale;
                }
            }
        }
    }
    out
}
