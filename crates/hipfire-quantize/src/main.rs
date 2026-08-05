// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-quantize: Quantize raw FP16/BF16/FP32 model weights to Q4_F16 format.
//!
//! Usage: hipfire-quantize --input <model_dir-or-gguf> --output <output.hfq> [--format mq4]
//!
//! Reads safetensors files from a HuggingFace model directory OR a single
//! `.gguf` file and produces a `.hfq` (HipFire Quantized) file with
//! RDNA-native quantized weights.

mod e8;
mod e8_gptq;
mod gguf_input;
mod reap_overlay;

use clap::Parser;
use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use hipfire_quantize::safetensors_file::{SafetensorsFile, TensorMeta};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

// imatrix lookup populated once in main() when --imatrix is supplied; keyed by
// ggml-style tensor name (see safetensors_to_ggml_name), value is the
// per-input-channel `Σ_token act²` vector. Consumed by AWQ pre-scaling to
// derive per-channel `RMS_act` for the smoothing-quant scale.
static IMATRIX: OnceLock<HashMap<String, Vec<f32>>> = OnceLock::new();

// Phase A Stage A — AWQ (Activation-aware Weight Quantization, Lin et al
// 2023). When AWQ_ALPHA is set (via --awq [<alpha>=0.55]), each linear-layer
// weight gets per-input-channel pre-scaling applied BEFORE the standard
// quantize+rotation path:
//
//   s[j] = (rms_act[j])^α   where rms_act[j] = sqrt(imatrix.in_sum2[j] / n_tok)
//
// Then W'[i,j] = W[i,j] * s[j] is what gets quantized + (for MQ4/MFP4) FWHT-
// rotated + packed into the wire format.
//
// At inference, the runtime must apply x / s element-wise BEFORE the rotation
// kernel — the math `(W·s) · (x/s) = W·x` cancels exactly at infinite
// precision. The quantizer writes the `s` vector as a sidecar 1D F16 tensor
// alongside each weight (name = `<weight_name>.awq_scale`); the runtime
// loader reads it and passes to fused_rmsnorm_rotate_mq (or equivalent for
// HFP4/MFP4).
//
// Why per-channel pre-scaling helps where per-block weighted-LS (L5c)
// failed on rotated formats:
//   - L5c weights individual block-level errors by per-channel importance.
//     For FWHT-rotated weights, rotation flattens per-channel importance
//     within blocks (Var[x_rot[i]] = Σ_j Var[x[j]] = const). The lever
//     has nothing to weight.
//   - AWQ applies its scaling in the UNROTATED basis before the FWHT bake-
//     in. The math composes: rot(W·s) is stored, rot(x/s) is computed at
//     inference. Per-channel importance attribution survives the rotation
//     because s is folded into the activation flow.
//   - Egiazarian et al (2509.23202 §3.2) also caution: at small group sizes
//     (g=16 NVFP4, g=32 MXFP4), "outlier mitigation is provably neutralized".
//     This applies to MFP4G32 but NOT to MQ4G256 — AWQ should work on MQ4.
//
// Default alpha = 0.55 (hipfire F2 sweep winner). --awq alone enables
// AWQ at alpha=0.55; --awq <value> sets explicit alpha. Alpha=0 disables;
// alpha=1 is pure activation-magnitude scaling (no smoothing).
static AWQ_ALPHA: OnceLock<f32> = OnceLock::new();

#[derive(Debug, Parser)]
#[command(
    name = "hipfire-quantize",
    version,
    about = "Quantize Hugging Face safetensors or GGUF weights into Hipfire HFQ"
)]
struct QuantizeArgs {
    /// Hugging Face model directory, model ID, or GGUF file.
    #[arg(long, value_name = "PATH_OR_MODEL_ID")]
    input: String,

    /// Destination HFQ file.
    #[arg(long, value_name = "PATH")]
    output: String,

    /// Quantization recipe or wire format.
    #[arg(long, default_value = "q8f16")]
    format: String,

    /// Rayon worker threads (defaults to 80% of available cores).
    #[arg(long, env = "HIPFIRE_QUANT_THREADS", value_name = "N")]
    threads: Option<usize>,

    /// Override the architecture ID stamped into the HFQ header.
    #[arg(long, value_name = "ID")]
    arch_id: Option<u32>,

    /// Allow an architecture override to move Qwen3 off its pillar IDs.
    #[arg(long)]
    force_arch_id: bool,

    /// Emit only tensors selected by a REAP plan.
    #[arg(long, value_name = "PLAN_DIR", conflicts_with = "reap_bake")]
    reap_overlay: Option<String>,

    /// Apply a REAP plan while baking a complete model.
    #[arg(long, value_name = "PLAN_DIR", conflicts_with = "reap_overlay")]
    reap_bake: Option<String>,

    /// Output path for a REAP overlay or baked model.
    #[arg(long, value_name = "PATH")]
    reap_out: Option<String>,

    /// Architecture family used to interpret a REAP plan.
    #[arg(long, value_name = "ARCH")]
    reap_arch: Option<String>,

    /// llama.cpp imatrix GGUF used for activation-aware quantization.
    #[arg(long, value_name = "PATH")]
    imatrix: Option<PathBuf>,

    /// Per-tensor Hessian directory used by GPTQ-E8 recipes.
    #[arg(long, value_name = "DIR")]
    hessian_dir: Option<PathBuf>,

    /// Fraction of hot layers assigned the higher-precision Lloyd tier.
    #[arg(long, env = "HIPFIRE_TIER_RATIO", default_value_t = 0.30)]
    tier_ratio: f64,

    /// Force router tensors to Q8.
    #[arg(long)]
    q8_router: bool,

    /// Disable the default Q8 protection for conv1d tensors.
    #[arg(long)]
    no_q8_conv1d: bool,

    /// Disable K-map precision promotion.
    #[arg(long)]
    no_kmap: bool,

    /// Alias for --no-kmap for uniform quantization.
    #[arg(long)]
    uniform: bool,

    /// Enable AWQ pre-scaling with the default alpha.
    #[arg(long)]
    awq: bool,

    /// Enable AWQ pre-scaling with an explicit alpha.
    #[arg(long, value_name = "ALPHA")]
    awq_alpha: Option<f32>,

    /// Enable K-map promotion for dense models.
    #[arg(long)]
    kmap_dense: bool,

    /// K-map policy: full, alternating/alt, or typed.
    #[arg(long, default_value = "alternating", value_name = "MODE")]
    kmap_mode: String,

    /// Permit research-only uniform MQ2 output.
    #[arg(long)]
    allow_mq2: bool,

    /// Permit research-only MQ2-Lloyd output.
    #[arg(long)]
    allow_mq2_lloyd: bool,

    /// Permit research-only MQ3-Lloyd output.
    #[arg(long)]
    allow_mq3_lloyd: bool,

    /// Permit research-only MQ4-Lloyd output.
    #[arg(long)]
    allow_mq4_lloyd: bool,

    /// Include vision tensors that are skipped by default.
    #[arg(long)]
    include_vision: bool,

    /// Quantization recipe for included vision tensors.
    #[arg(long, default_value = "", value_name = "FORMAT")]
    vision_quant: String,

    /// Ingest only tensors whose names start with this prefix.
    #[arg(long, value_name = "PREFIX")]
    include_prefix: Option<String>,
}

/// Refuse an `--arch-id` override that strips a qwen3* model (auto-detected
/// arch 5 or 6) off the pillar arches, unless `--force-arch-id` is given.
/// Keeps the froggeric chat-template pillar + qwen35-crate dispatch intact by
/// construction. No-op for non-qwen models or overrides that stay in {5,6}.
fn guard_qwen3_arch_override(auto_arch_id: u32, arch_id: u32, force_arch_id: bool) {
    let auto_is_qwen3 = matches!(auto_arch_id, 5 | 6);
    let override_off_pillar = !matches!(arch_id, 5 | 6);
    if auto_is_qwen3 && arch_id != auto_arch_id && override_off_pillar && !force_arch_id {
        eprintln!(
            "error: --arch-id {arch_id} moves an auto-detected qwen3* model (arch {auto_arch_id}) \
             OFF the pillar arches {{5,6}}; this would break the froggeric chat-template pillar \
             and the qwen35-crate dispatch. Pass --force-arch-id to override anyway."
        );
        std::process::exit(1);
    }
}

/// Convert raw tensor bytes to F32 based on dtype string
fn to_f32(data: &[u8], dtype: &str) -> Vec<f32> {
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

fn e4m3_to_f32(byte: u8) -> f32 {
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
fn ue8m0_to_scale(byte: u8) -> f32 {
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
fn tensor_to_f32_with_optional_fp8_scale(
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
fn e2m1_to_f32(nibble: u8) -> f32 {
    // Lookup table for the 8 magnitude codes; sign is applied after.
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
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
fn dequantize_e2m1_ue8m0_to_f32(
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
fn dequantize_e4m3_ue8m0_to_f32(
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
fn dequantize_e4m3_f32scale_to_f32(
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

// ─── Q4_F16_G64 Quantization ────────────────────────────────────────────────

/// Quantize F32 weights to Q4_F16_G64 format.
/// Group size 64: 36 bytes per 64 elements (0.5625 bytes/weight).
/// Block: f16 scale (2B) + f16 min (2B) + u8[32] packed nibbles (32B).
fn quantize_q4f16_g64(f32_data: &[f32]) -> Vec<u8> {
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
fn quantize_q4k(f32_data: &[f32]) -> Vec<u8> {
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
fn quantize_q4_as_q8(f32_data: &[f32]) -> Vec<u8> {
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
fn quantize_q8hfq(f32_data: &[f32], m: usize, k: usize) -> (Vec<u8>, usize) {
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

/// Quantize F32 weights to HFQ4-G256: flat 4-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][128B nibbles] = 136 bytes per 256 weights (0.531 B/w).
/// 18 VGPRs, 100% occupancy on RDNA1. Beats Q4_K at all matrix sizes.
/// CPU-side FWHT (Walsh-Hadamard Transform) on a 256-element group.
/// Matches the GPU-side fwht_forward_256 in turbo_common: signs1 → butterfly → scale → signs2.
fn cpu_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
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
fn quantize_mq5g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
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
fn quantize_mq8g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
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

// ─── HFP4G32 — RDNA-optimal FP4 (E2M1 + UE8M0 g32 + FP16 row scale) ────────────────
//
// Spec: docs/quant-formats/hfp4.md
//
// Per-row layout: 16-B header (row_scale_a:f16, row_scale_b:f16, block_count:u16, flags:u8, ...)
//                 followed by (K/32) blocks × 17 B (UE8M0:u8 + 16 B nibbles).
// Per element:    value = row_scale_a * 2^(block_e - 127) * E2M1_LUT[nibble]

/// OCP E2M1 magnitude lattice (signed 4-bit FP). 16 codes: {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}.
/// Order: positive 0..7, then negative 0..7 (mirrors hardware-canonical sign-magnitude packing).
const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// E2M1 round-to-nearest in the 16-code lattice. Returns the nibble (0..15).
/// Ties broken away from zero (consistent with FP rounding).
fn e2m1_round(x: f32) -> u8 {
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
fn quantize_hfp4g32_row(row: &[f32]) -> Vec<u8> {
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
fn quantize_hfp4g32_2d(f32_data: &[f32], m: usize, k: usize) -> Vec<u8> {
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
fn quantize_mfp4g32_2d(
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
fn e4m3_scale_decode(byte: u8) -> f32 {
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
fn e4m3_scale_encode_roundup(s: f32) -> u8 {
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
fn quantize_mfp4g32_p_row(row: &[f32]) -> Vec<u8> {
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
fn quantize_mfp4g32_p_2d(
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
fn dequant_mfp4g32_p(packed: &[u8], m: usize, k: usize) -> Vec<f32> {
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

/// Quantize one row of K FP32 weights to mfp4-E8 byte format.
/// Same E4M3 scale as mfp4+P; per-32-weight-block data = 4 E8 codewords (u32 each).
/// Returns 16-B header + (K/32) x 17-B blocks. Byte-identical footprint to mfp4+P.
fn quantize_mfp4g32_e8_row(row: &[f32]) -> Vec<u8> {
    assert!(
        row.len() % 32 == 0,
        "mfp4-E8 requires K%32==0, got K={}",
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
    out[6] = 0x05; // FWHT flag, identical to mfp4+P
    out[7] = 0u8;
    for b in 0..n_blocks {
        let block = &row[b * 32..b * 32 + 32];
        let block_max_abs = block.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
        let block_max_normalized = block_max_abs * inv_row_scale;
        let s = if block_max_normalized > 0.0 {
            block_max_normalized / 6.0
        } else {
            0.0
        };
        let scale_byte = e4m3_scale_encode_roundup(s);
        let block_scale_factor = e4m3_scale_decode(scale_byte);
        let inv_block_scale = if block_scale_factor > 0.0 {
            1.0 / block_scale_factor
        } else {
            0.0
        };
        let payload_off = 16 + b * 17;
        out[payload_off] = scale_byte;
        // 4 E8 codewords, 8 weights each, 32 bits little-endian.
        for g in 0..4 {
            let mut v = [0.0f32; 8];
            for i in 0..8 {
                v[i] = block[g * 8 + i] * inv_row_scale * inv_block_scale; // => [-6,6]
            }
            let idx = e8::quantize8(&v, e8::QUANT_STEP);
            out[payload_off + 1 + g * 4..payload_off + 1 + g * 4 + 4]
                .copy_from_slice(&idx.to_le_bytes());
        }
    }
    out
}

/// Generic mfpN-E8 row encoder (n=2 or n=3). Same outer container as mfp4-E8
/// (16 B row header + (K/32) blocks) but each block is 1 + 4*n bytes:
///   - 1 B: E4M3 block scale
///   - 4 × n B: codewords (low 8n bits of u32, LE)
/// Row bytes = 16 + (1 + 4*n) * (K/32). Callers use the thin wrappers below.
fn quantize_mfpn_e8_row(row: &[f32], n: u32, quant_step: f32) -> Vec<u8> {
    assert!(
        row.len() % 32 == 0,
        "mfpN-E8 requires K%32==0, got K={}",
        row.len()
    );
    assert!(n == 2 || n == 3, "only n=2 or n=3 supported by this helper");
    let k = row.len();
    let n_blocks = k / 32;
    let block_bytes = 1 + 4 * n as usize;
    let row_bytes = 16 + n_blocks * block_bytes;
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
    out[6] = 0x05; // same FWHT flag as mfp4-E8
    out[7] = 0u8;
    for b in 0..n_blocks {
        let block = &row[b * 32..b * 32 + 32];
        let block_max_abs = block.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
        let block_max_normalized = block_max_abs * inv_row_scale;
        let s = if block_max_normalized > 0.0 {
            block_max_normalized / 6.0
        } else {
            0.0
        };
        let scale_byte = e4m3_scale_encode_roundup(s);
        let block_scale_factor = e4m3_scale_decode(scale_byte);
        let inv_block_scale = if block_scale_factor > 0.0 {
            1.0 / block_scale_factor
        } else {
            0.0
        };
        let payload_off = 16 + b * block_bytes;
        out[payload_off] = scale_byte;
        // 4 codewords per block, 8 weights each, packed into low 8n bits (LE).
        for g in 0..4 {
            let mut v = [0.0f32; 8];
            for i in 0..8 {
                v[i] = block[g * 8 + i] * inv_row_scale * inv_block_scale;
            }
            let idx = e8::quantize8_n(&v, quant_step, n);
            // Write only the low n bytes (upper bits of u32 are guaranteed zero
            // by encode_index_n — see e8.rs high-bit-zero invariant).
            let bytes = idx.to_le_bytes();
            let cw_off = payload_off + 1 + g * n as usize;
            out[cw_off..cw_off + n as usize].copy_from_slice(&bytes[..n as usize]);
        }
    }
    out
}

fn quantize_mfp3g32_e8_row(row: &[f32]) -> Vec<u8> {
    quantize_mfpn_e8_row(row, 3, e8::QUANT_STEP_MFP3)
}

fn quantize_mfp2g32_e8_row(row: &[f32]) -> Vec<u8> {
    quantize_mfpn_e8_row(row, 2, e8::QUANT_STEP_MFP2)
}

/// mfpN-E8 2D: FWHT-rotate (same signs as mfp4-E8), then per-row encode.
fn quantize_mfpn_e8_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
    n: u32,
    quant_step: f32,
) -> Vec<u8> {
    assert_eq!(f32_data.len(), m * k);
    assert!(k % 256 == 0, "mfpN-E8 requires k%256==0, got k={}", k);
    let block_bytes = 1 + 4 * n as usize;
    let row_bytes = 16 + block_bytes * (k / 32);
    let mut out = Vec::with_capacity(m * row_bytes);
    let mut row_buf = vec![0.0f32; k];
    for r in 0..m {
        row_buf.copy_from_slice(&f32_data[r * k..(r + 1) * k]);
        for seg in 0..(k / 256) {
            cpu_fwht_256(&mut row_buf[seg * 256..(seg + 1) * 256], signs1, signs2);
        }
        out.extend_from_slice(&quantize_mfpn_e8_row(&row_buf, n, quant_step));
    }
    out
}

pub fn quantize_mfp3g32_e8_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    quantize_mfpn_e8_2d(f32_data, m, k, signs1, signs2, 3, e8::QUANT_STEP_MFP3)
}

pub fn quantize_mfp2g32_e8_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    quantize_mfpn_e8_2d(f32_data, m, k, signs1, signs2, 2, e8::QUANT_STEP_MFP2)
}

/// CPU reference dequant for mfp3-E8. Returns row-major f32 in the ROTATED domain.
/// Mirrors the kernel mfp3_decode_index decode exactly (3-bit nibbles, center 3,
/// coset bit 23, e7_high 2b at bit 21).
#[allow(dead_code)]
fn dequant_mfp3g32_e8(packed: &[u8], m: usize, k: usize) -> Vec<f32> {
    let block_bytes = 13usize; // 1 + 4*3
    let row_bytes = 16 + block_bytes * (k / 32);
    assert_eq!(packed.len(), m * row_bytes, "mfp3-E8 size mismatch");
    let mut out = vec![0.0f32; m * k];
    for r in 0..m {
        let base = r * row_bytes;
        let row_scale_a = f16_to_f32(u16::from_le_bytes([packed[base], packed[base + 1]]));
        for b in 0..(k / 32) {
            let po = base + 16 + b * block_bytes;
            let scale_byte = packed[po];
            let scale = row_scale_a * e4m3_scale_decode(scale_byte) * e8::QUANT_STEP_MFP3;
            for g in 0..4usize {
                let cw_off = po + 1 + g * 3;
                // 3-byte LE narrow read (safe — block is 13 B, max cw_off = po+1+3*3=po+10, reads to po+12)
                let idx: u32 = (packed[cw_off] as u32)
                    | ((packed[cw_off + 1] as u32) << 8)
                    | ((packed[cw_off + 2] as u32) << 16);
                // mfp3_decode_index: 3-bit nibbles, center 3, coset bit 23, e7_high @21 (2b)
                let coset = (idx >> 23) & 1;
                let mut e = [0u32; 8];
                let mut sl: u32 = 0;
                for i in 0..7 {
                    e[i] = (idx >> (3 * i as u32)) & 0x7;
                    sl += e[i];
                }
                let e7_high = (idx >> 21) & 0x3;
                let p7 = e7_high << 1;
                e[7] = p7 | ((sl + p7) & 1);
                for i in 0..8usize {
                    let c = (e[i] as i32 - 3) as f32;
                    let coord = if coset == 1 { c + 0.5 } else { c };
                    out[r * k + b * 32 + g * 8 + i] = scale * coord;
                }
            }
        }
    }
    out
}

/// CPU reference dequant for mfp2-E8. Returns row-major f32 in the ROTATED domain.
/// Mirrors the kernel mfp2_decode_index decode exactly (2-bit nibbles, center 1,
/// coset bit 15, e7_high 1b at bit 14).
#[allow(dead_code)]
fn dequant_mfp2g32_e8(packed: &[u8], m: usize, k: usize) -> Vec<f32> {
    let block_bytes = 9usize; // 1 + 4*2
    let row_bytes = 16 + block_bytes * (k / 32);
    assert_eq!(packed.len(), m * row_bytes, "mfp2-E8 size mismatch");
    let mut out = vec![0.0f32; m * k];
    for r in 0..m {
        let base = r * row_bytes;
        let row_scale_a = f16_to_f32(u16::from_le_bytes([packed[base], packed[base + 1]]));
        for b in 0..(k / 32) {
            let po = base + 16 + b * block_bytes;
            let scale_byte = packed[po];
            let scale = row_scale_a * e4m3_scale_decode(scale_byte) * e8::QUANT_STEP_MFP2;
            for g in 0..4usize {
                let cw_off = po + 1 + g * 2;
                // 2-byte LE narrow read (safe — block is 9 B, max cw_off = po+1+3*2=po+7, reads to po+8)
                let idx: u32 = (packed[cw_off] as u32) | ((packed[cw_off + 1] as u32) << 8);
                // mfp2_decode_index: 2-bit nibbles, center 1, coset bit 15, e7_high @14 (1b)
                let coset = (idx >> 15) & 1;
                let mut e = [0u32; 8];
                let mut sl: u32 = 0;
                for i in 0..7 {
                    e[i] = (idx >> (2 * i as u32)) & 0x3;
                    sl += e[i];
                }
                let e7_high = (idx >> 14) & 0x1;
                let p7 = e7_high << 1;
                e[7] = p7 | ((sl + p7) & 1);
                for i in 0..8usize {
                    let c = (e[i] as i32 - 1) as f32;
                    let coord = if coset == 1 { c + 0.5 } else { c };
                    out[r * k + b * 32 + g * 8 + i] = scale * coord;
                }
            }
        }
    }
    out
}

/// mfp4-E8 2D: FWHT-rotate the tensor (same signs as mfp4+P), then per-row
/// quantize_mfp4g32_e8_row. Byte layout identical to mfp4+P (NO prefix).
/// On-disk per-256-block Hessian magic ("E8H1").
const E8_HESSIAN_MAGIC: u32 = 0x45_38_48_31;

/// Sanitize a full safetensors tensor name into a filesystem-safe key.
fn hessian_key(tensor_name: &str) -> String {
    tensor_name.replace(['/', '\\'], "_").replace("..", "_")
}

/// Read per-256-block Hessians for one tensor from `<dir>/<key>.hblk`.
/// File layout: [u32 magic][u32 n_blocks][u32 k][f32 ... n_blocks*256*256].
/// Returns empty Vec if the file is missing (-> caller falls back to RTN).
fn load_hessian_blocks(dir: &Path, tensor_name: &str) -> Vec<e8_gptq::HBlock> {
    let path = dir.join(format!("{}.hblk", hessian_key(tensor_name)));
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) => return Vec::new(),
    };
    if bytes.len() < 12 {
        return Vec::new();
    }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != E8_HESSIAN_MAGIC {
        eprintln!("warning: {} bad Hessian magic; ignoring", path.display());
        return Vec::new();
    }
    let n_blocks = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let want = 12 + n_blocks * 256 * 256 * 4;
    if bytes.len() < want {
        eprintln!(
            "warning: {} truncated Hessian ({} < {}); ignoring",
            path.display(),
            bytes.len(),
            want
        );
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n_blocks);
    let mut off = 12usize;
    for _ in 0..n_blocks {
        let mut blk = vec![0.0f32; 256 * 256];
        for v in blk.iter_mut() {
            *v = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
            off += 4;
        }
        out.push(blk);
    }
    out
}

/// GPTQ-on-E8 fired/fallback telemetry. `fired` = a non-empty `.hblk` was
/// loaded for this (tensor,expert) (Hessian-aware LDLQ ran); `fallback` =
/// empty/missing Hessian -> silent RTN E8. ~0 fired with --hessian-dir set
/// means a KEY-MISMATCH BUG (filenames != hessian_key), not a flat result.
static GPTQ_E8_FIRED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GPTQ_E8_FALLBACK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// GPTQ-E8 wrapper that wires the production helpers into the e8_gptq module.
/// `h_blocks` empty -> RTN fallback (byte-identical to quantize_mfp4g32_e8_2d).
fn quantize_mfp4g32_e8_gptq_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
    h_blocks: &[e8_gptq::HBlock],
) -> Vec<u8> {
    e8_gptq::quantize_mfp4g32_e8_gptq_2d(
        f32_data,
        m,
        k,
        signs1,
        signs2,
        h_blocks,
        &cpu_fwht_256,
        &e4m3_scale_encode_roundup,
        &e4m3_scale_decode,
        &f32_to_f16,
    )
}

/// GPTQ-mfp3-E8 wrapper. `h_blocks` empty -> RTN fallback (byte-identical to
/// quantize_mfp3g32_e8_2d). Output layout: 16-byte header + 13 B per 32-block.
fn quantize_mfp3g32_e8_gptq_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
    h_blocks: &[e8_gptq::HBlock],
) -> Vec<u8> {
    e8_gptq::quantize_mfp3g32_e8_gptq_2d(
        f32_data,
        m,
        k,
        signs1,
        signs2,
        h_blocks,
        &cpu_fwht_256,
        &e4m3_scale_encode_roundup,
        &e4m3_scale_decode,
        &f32_to_f16,
    )
}

/// GPTQ-mfp2-E8 wrapper. `h_blocks` empty -> RTN fallback (byte-identical to
/// quantize_mfp2g32_e8_2d). Output layout: 16-byte header + 9 B per 32-block.
fn quantize_mfp2g32_e8_gptq_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
    h_blocks: &[e8_gptq::HBlock],
) -> Vec<u8> {
    e8_gptq::quantize_mfp2g32_e8_gptq_2d(
        f32_data,
        m,
        k,
        signs1,
        signs2,
        h_blocks,
        &cpu_fwht_256,
        &e4m3_scale_encode_roundup,
        &e4m3_scale_decode,
        &f32_to_f16,
    )
}

fn quantize_mfp4g32_e8_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    assert_eq!(f32_data.len(), m * k);
    assert!(k % 256 == 0, "mfp4-E8 requires k%256==0, got k={}", k);
    let row_bytes = 16 + 17 * (k / 32);
    let mut out = Vec::with_capacity(m * row_bytes);
    let mut row_buf = vec![0.0f32; k];
    for r in 0..m {
        row_buf.copy_from_slice(&f32_data[r * k..(r + 1) * k]);
        for seg in 0..(k / 256) {
            cpu_fwht_256(&mut row_buf[seg * 256..(seg + 1) * 256], signs1, signs2);
        }
        out.extend_from_slice(&quantize_mfp4g32_e8_row(&row_buf));
    }
    out
}

/// CPU reference dequant for mfp4-E8. Returns row-major f32 [m*k] in the ROTATED domain.
/// Bit-exact mirror of the gfx942 dequantize_mfp4g32_e8_to_f16 kernel decode.
#[allow(dead_code)]
fn dequant_mfp4g32_e8(packed: &[u8], m: usize, k: usize) -> Vec<f32> {
    let row_bytes = 16 + 17 * (k / 32);
    assert_eq!(packed.len(), m * row_bytes, "mfp4-E8 size mismatch");
    let mut out = vec![0.0f32; m * k];
    for r in 0..m {
        let base = r * row_bytes;
        let row_scale_a = f16_to_f32(u16::from_le_bytes([packed[base], packed[base + 1]]));
        for b in 0..(k / 32) {
            let po = base + 16 + b * 17;
            let scale = row_scale_a * e4m3_scale_decode(packed[po]);
            for g in 0..4 {
                let idx = u32::from_le_bytes([
                    packed[po + 1 + g * 4],
                    packed[po + 2 + g * 4],
                    packed[po + 3 + g * 4],
                    packed[po + 4 + g * 4],
                ]);
                let vd = e8::dequantize8(idx, e8::QUANT_STEP);
                for i in 0..8 {
                    out[r * k + b * 32 + g * 8 + i] = scale * vd[i];
                }
            }
        }
    }
    out
}

/// Convert an AoS-packed mfp4-E8 row to SoA layout.
/// AoS layout: [16B hdr] + n_blocks * [1B E4M3 scale + 16B codewords]
/// SoA layout: [16B hdr] + [n_blocks B scales, pad to 16B] + [n_blocks * 16B codewords]
/// The header is the same except byte[6] (flag) is set to 0x06 (was 0x05).
fn aos_to_soa_row(aos: &[u8], n_blocks: usize) -> Vec<u8> {
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let soa_len = 16 + scale_padded + n_blocks * 16;
    let mut out = vec![0u8; soa_len];
    // Copy header, change flag byte
    out[..16].copy_from_slice(&aos[..16]);
    out[6] = 0x06; // SoA flag
                   // Gather scales
    for b in 0..n_blocks {
        out[16 + b] = aos[16 + b * 17]; // scale byte at start of each 17B AoS block
    }
    // Gather codewords (4 x u32 = 16B per block)
    let cw_start = 16 + scale_padded;
    for b in 0..n_blocks {
        let src = 16 + b * 17 + 1; // skip the 1B scale
        let dst = cw_start + b * 16;
        out[dst..dst + 16].copy_from_slice(&aos[src..src + 16]);
    }
    out
}

/// mfp4-E8 SoA quantizer: same E8 encoding as quantize_mfp4g32_e8_2d, then permuted to SoA.
fn quantize_mfp4g32_e8_soa_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    assert_eq!(f32_data.len(), m * k);
    assert!(k % 256 == 0, "mfp4-E8-SoA requires k%256==0, got k={}", k);
    let n_blocks = k / 32;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let soa_row_bytes = 16 + scale_padded + n_blocks * 16;
    let mut out = Vec::with_capacity(m * soa_row_bytes);
    let mut row_buf = vec![0.0f32; k];
    for r in 0..m {
        row_buf.copy_from_slice(&f32_data[r * k..(r + 1) * k]);
        for seg in 0..(k / 256) {
            cpu_fwht_256(&mut row_buf[seg * 256..(seg + 1) * 256], signs1, signs2);
        }
        let aos_row = quantize_mfp4g32_e8_row(&row_buf);
        out.extend_from_slice(&aos_to_soa_row(&aos_row, n_blocks));
    }
    out
}

/// CPU reference dequant for mfp4-E8 SoA. Returns row-major f32 [m*k] in ROTATED domain.
/// Bit-exact with dequant_mfp4g32_e8 for the same weight data.
#[allow(dead_code)]
fn dequant_mfp4g32_e8_soa(packed: &[u8], m: usize, k: usize) -> Vec<f32> {
    let n_blocks = k / 32;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let soa_row_bytes = 16 + scale_padded + n_blocks * 16;
    assert_eq!(packed.len(), m * soa_row_bytes, "mfp4-E8-SoA size mismatch");
    let mut out = vec![0.0f32; m * k];
    for r in 0..m {
        let base = r * soa_row_bytes;
        let row_scale_a = f16_to_f32(u16::from_le_bytes([packed[base], packed[base + 1]]));
        let scale_arr = &packed[base + 16..base + 16 + n_blocks];
        let cw_arr = &packed[base + 16 + scale_padded..base + 16 + scale_padded + n_blocks * 16];
        for b in 0..n_blocks {
            let scale = row_scale_a * e4m3_scale_decode(scale_arr[b]);
            for g in 0..4 {
                let co = b * 16 + g * 4;
                let idx = u32::from_le_bytes([
                    cw_arr[co],
                    cw_arr[co + 1],
                    cw_arr[co + 2],
                    cw_arr[co + 3],
                ]);
                let vd = e8::dequantize8(idx, e8::QUANT_STEP);
                for i in 0..8 {
                    out[r * k + b * 32 + g * 8 + i] = scale * vd[i];
                }
            }
        }
    }
    out
}

/// Fit ONE 16-entry fp16 Lloyd-Max codebook over ALL normalized values of a
/// FWHT-rotated tensor, in the same ~[-6,6] domain that feeds e2m1_round in
/// mfp4. `vals` are pre-collected: for every element, value*inv_row_scale*
/// inv_block_scale (i.e. exactly what mfp4 hands to e2m1_round). Mirrors the
/// percentile-init + 8 k-means iters + sort-ascending recipe of
/// quantize_mq4g256_lloyd, but tensor-global instead of per-block.
fn fit_mfp4_lloyd_codebook(vals: &[f32]) -> [f32; 16] {
    let mut sorted: Vec<f32> = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let mut cb = [0.0f32; 16];
    if n == 0 {
        for k in 0..16 {
            cb[k] = E2M1_LUT[k];
        }
        return cb;
    }
    for k in 0..16 {
        let frac = (2 * k + 1) as f32 / 32.0;
        let idx = ((frac * (n as f32 - 1.0)).round() as usize).min(n - 1);
        cb[k] = sorted[idx];
    }
    let range = sorted[n - 1] - sorted[0];
    if range > 0.0 {
        let mut _it = 0usize;
        loop {
            let mut sums = [0.0f64; 16];
            let mut counts = [0u64; 16];
            for &w in vals {
                let mut best = 0usize;
                let mut best_d = (w - cb[0]).abs();
                for k in 1..16 {
                    let d = (w - cb[k]).abs();
                    if d < best_d {
                        best_d = d;
                        best = k;
                    }
                }
                sums[best] += w as f64;
                counts[best] += 1;
            }
            let mut moved = false;
            for k in 0..16 {
                if counts[k] > 0 {
                    let c = (sums[k] / counts[k] as f64) as f32;
                    if c != cb[k] {
                        moved = true;
                    }
                    cb[k] = c;
                }
            }
            _it += 1;
            if !moved || _it >= 8 {
                break;
            }
        }
    }
    cb.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    cb
}

/// Nearest-codebook index for mfp4L. The codebook is fp16-rounded BEFORE search
/// so quantize matches kernel recon (kernel reads fp16 cb).
fn nearest_cb_idx(x: f32, cb: &[f32; 16]) -> u8 {
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for (i, &c) in cb.iter().enumerate() {
        let cf = f16_to_f32(f32_to_f16(c));
        let e = (cf - x).abs();
        if e < best_err {
            best_err = e;
            best = i as u8;
        }
    }
    best
}

/// Quantize one mfp4-normalized row to 17-B-block bytes using a fixed
/// per-tensor codebook (nibble = nearest-codebook-index).
fn quantize_mfp4g32_lloyd_row(row: &[f32], cb: &[f32; 16]) -> Vec<u8> {
    assert!(
        row.len() % 32 == 0,
        "mfp4L requires K%32==0, got K={}",
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
    out[6] = 0x05;
    out[7] = 0u8;
    for b in 0..n_blocks {
        let block = &row[b * 32..b * 32 + 32];
        let block_max_abs = block.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
        let block_max_normalized = block_max_abs * inv_row_scale;
        let block_e: u8 = if block_max_normalized > 0.0 {
            let e_signed = (block_max_normalized / 6.0).log2().ceil() as i32 + 127;
            e_signed.clamp(0, 254) as u8
        } else {
            0u8
        };
        let block_scale_factor = ((block_e as i32 - 127) as f32).exp2();
        let inv_block_scale = if block_scale_factor > 0.0 {
            1.0 / block_scale_factor
        } else {
            0.0
        };
        let payload_off = 16 + b * 17;
        out[payload_off] = block_e;
        for i in 0..16 {
            let lo = block[2 * i] * inv_row_scale * inv_block_scale;
            let hi = block[2 * i + 1] * inv_row_scale * inv_block_scale;
            out[payload_off + 1 + i] =
                (nearest_cb_idx(lo, cb) & 0x0F) | ((nearest_cb_idx(hi, cb) & 0x0F) << 4);
        }
    }
    out
}

/// MFP4G32-Lloyd 2D: FWHT-rotate the tensor, fit ONE per-tensor 16-entry codebook,
/// then emit [32-B fp16 codebook][M rows].
fn quantize_mfp4g32_lloyd_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    assert_eq!(f32_data.len(), m * k);
    assert!(k % 256 == 0, "MFP4G32Lloyd requires k%256==0, got k={}", k);
    let mut rotated = vec![0.0f32; m * k];
    let mut norm_vals: Vec<f32> = Vec::with_capacity(m * k);
    let mut row_buf = vec![0.0f32; k];
    for r in 0..m {
        row_buf.copy_from_slice(&f32_data[r * k..(r + 1) * k]);
        for seg in 0..(k / 256) {
            cpu_fwht_256(&mut row_buf[seg * 256..(seg + 1) * 256], signs1, signs2);
        }
        rotated[r * k..(r + 1) * k].copy_from_slice(&row_buf);
        let row_max_abs = row_buf.iter().cloned().fold(0.0f32, |a, v| a.max(v.abs()));
        let inv_row_scale = if row_max_abs > 0.0 {
            6.0 / row_max_abs
        } else {
            0.0
        };
        for b in 0..(k / 32) {
            let block = &row_buf[b * 32..b * 32 + 32];
            let bmax = block.iter().cloned().fold(0.0f32, |a, v| a.max(v.abs()));
            let bmn = bmax * inv_row_scale;
            let block_e: i32 = if bmn > 0.0 {
                (bmn / 6.0).log2().ceil() as i32 + 127
            } else {
                0
            };
            let inv_block_scale = (-(block_e.clamp(0, 254) - 127) as f32).exp2();
            for &v in block {
                norm_vals.push(v * inv_row_scale * inv_block_scale);
            }
        }
    }
    let cb = fit_mfp4_lloyd_codebook(&norm_vals);
    let row_bytes = 16 + 17 * (k / 32);
    let mut out = Vec::with_capacity(32 + m * row_bytes);
    for k16 in 0..16usize {
        out.extend_from_slice(&f32_to_f16(cb[k16]).to_le_bytes());
    }
    for r in 0..m {
        out.extend_from_slice(&quantize_mfp4g32_lloyd_row(
            &rotated[r * k..(r + 1) * k],
            &cb,
        ));
    }
    debug_assert_eq!(out.len(), 32 + m * row_bytes);
    out
}

/// CPU reference dequant for mfp4L.
/// Returns row-major f32 [m*k] in the ROTATED domain.
#[allow(dead_code)]
fn dequant_mfp4g32_lloyd(packed: &[u8], m: usize, k: usize) -> Vec<f32> {
    let row_bytes = 16 + 17 * (k / 32);
    assert_eq!(packed.len(), 32 + m * row_bytes, "mfp4L size mismatch");
    let mut cb = [0.0f32; 16];
    for i in 0..16 {
        cb[i] = f16_to_f32(u16::from_le_bytes([packed[2 * i], packed[2 * i + 1]]));
    }
    let mut out = vec![0.0f32; m * k];
    for r in 0..m {
        let base = 32 + r * row_bytes;
        let row_scale_a = f16_to_f32(u16::from_le_bytes([packed[base], packed[base + 1]]));
        for b in 0..(k / 32) {
            let po = base + 16 + b * 17;
            let block_e = packed[po] as i32;
            let scale = row_scale_a * ((block_e - 127) as f32).exp2();
            for i in 0..16 {
                let byte = packed[po + 1 + i];
                out[r * k + b * 32 + 2 * i] = scale * cb[(byte & 0x0F) as usize];
                out[r * k + b * 32 + 2 * i + 1] = scale * cb[((byte >> 4) & 0x0F) as usize];
            }
        }
    }
    out
}

/// CPU reference dequantization for HFP4G32 — bit-exact mirror of `gemv_hfp4g32.hip`'s dequant.
/// Returns the K reconstructed FP32 weights for one row.
#[allow(dead_code)] // used by tests + future round-trip diagnostics
fn dequant_hfp4g32_row(packed: &[u8], k: usize) -> Vec<f32> {
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

#[cfg(test)]
mod awq_tests {
    use super::*;

    /// Verify geometric mean of computed AWQ scales is ~1.0 — the
    /// normalization in compute_awq_scales should center the scale
    /// vector so downstream min-max quantization isn't perturbed.
    #[test]
    fn awq_scales_geomean_is_one() {
        // Realistic-ish imatrix: log-normal-ish per-channel statistics
        let in_sum2: Vec<f32> = (0..256)
            .map(|j| (1.0 + 10.0 * (j as f32 / 256.0)).exp()) // 1.0 → e^11
            .collect();
        for &alpha in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let s = compute_awq_scales(&in_sum2, alpha);
            assert_eq!(s.len(), in_sum2.len());
            // Geometric mean = exp(mean(log(s)))
            let log_mean = s.iter().map(|&v| (v as f64).ln()).sum::<f64>() / s.len() as f64;
            let geo_mean = log_mean.exp();
            assert!(
                (geo_mean - 1.0).abs() < 1e-4,
                "alpha={alpha}: geo_mean={geo_mean} (want 1.0)"
            );
        }
    }

    /// Alpha = 0 should produce all-ones scales (AWQ disabled at layer level).
    #[test]
    fn awq_scales_alpha_zero_is_identity() {
        let in_sum2: Vec<f32> = (1..=128).map(|j| j as f32).collect();
        let s = compute_awq_scales(&in_sum2, 0.0);
        for &v in &s {
            assert!((v - 1.0).abs() < 1e-5, "alpha=0 scale {v} should be 1.0");
        }
    }

    /// Larger imatrix values should produce larger scales for alpha > 0.
    /// Monotonicity check.
    #[test]
    fn awq_scales_monotonic_in_imatrix() {
        let in_sum2 = vec![1.0_f32, 4.0, 16.0, 64.0, 256.0];
        let s = compute_awq_scales(&in_sum2, 0.5);
        for w in s.windows(2) {
            assert!(w[1] > w[0], "scales not monotonic: {} -> {}", w[0], w[1]);
        }
    }

    /// AWQ math identity: `(W · diag(s)) · (x / s) == W · x` at infinite
    /// precision. With fp32 weights + fp32 activations, error should be
    /// at floating-point rounding precision (~1e-5 relative).
    #[test]
    fn awq_math_identity_holds() {
        // Tiny test: 4 output × 8 input matmul
        let m = 4;
        let k = 8;
        // Random-ish weights and activations
        let w: Vec<f32> = (0..m * k).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let x: Vec<f32> = (0..k).map(|j| (j as f32 + 1.0) * 0.5).collect();

        // Reference: y = W * x
        let mut y_ref = vec![0.0_f32; m];
        for i in 0..m {
            for j in 0..k {
                y_ref[i] += w[i * k + j] * x[j];
            }
        }

        // AWQ-scaled: pre-scale W, pre-divide x
        let in_sum2: Vec<f32> = (1..=k).map(|j| j as f32 * 10.0).collect();
        let s = compute_awq_scales(&in_sum2, 0.5);
        let mut w_scaled = w.clone();
        awq_pre_scale_weights(&mut w_scaled, m, k, &s);
        let x_div: Vec<f32> = x.iter().zip(&s).map(|(&xv, &sv)| xv / sv).collect();

        // y' = (W * diag(s)) * (x / s)
        let mut y_awq = vec![0.0_f32; m];
        for i in 0..m {
            for j in 0..k {
                y_awq[i] += w_scaled[i * k + j] * x_div[j];
            }
        }

        // Compare
        for i in 0..m {
            let rel = (y_awq[i] - y_ref[i]).abs() / y_ref[i].abs().max(1e-6);
            assert!(
                rel < 1e-5,
                "row {i}: AWQ y={} ref y={} rel_err={}",
                y_awq[i],
                y_ref[i],
                rel
            );
        }
    }

    /// Edge case: zero imatrix entries should produce finite scales
    /// (clamped via 1e-12 floor in compute_awq_scales).
    #[test]
    fn awq_handles_zero_imatrix() {
        let in_sum2 = vec![0.0_f32, 1.0, 4.0, 0.0];
        let s = compute_awq_scales(&in_sum2, 0.5);
        for &v in &s {
            assert!(
                v.is_finite() && v > 0.0,
                "scale {v} should be finite + positive"
            );
        }
    }
}

#[cfg(test)]
mod hfp4_tests {
    use super::*;

    #[test]
    fn e2m1_round_matches_lattice() {
        // Each lattice value should round to its own code.
        for (i, &val) in E2M1_LUT.iter().enumerate() {
            let nibble = e2m1_round(val);
            // +0 and -0 are both at value 0.0; either nibble is acceptable.
            if val.abs() < 1e-6 {
                assert!(
                    nibble == 0 || nibble == 8,
                    "zero rounds to nibble {}",
                    nibble
                );
            } else {
                assert_eq!(
                    nibble, i as u8,
                    "code {} rounded to nibble {} not {}",
                    i, nibble, i
                );
            }
        }
    }

    #[test]
    fn e2m1_round_midpoint() {
        // Halfway between +1.0 and +1.5 → either is acceptable (tie).
        let n = e2m1_round(1.25);
        assert!(n == 2 || n == 3, "midpoint rounded to {}", n);
        // Halfway between +4.0 and +6.0 (= 5.0) → either is acceptable.
        let n = e2m1_round(5.0);
        assert!(n == 6 || n == 7, "5.0 rounded to {}", n);
    }

    #[test]
    fn round_trip_constant_row() {
        // All-1.0 row: row_scale_a = 1/6, every block_e ≈ 127 + log2(1) = 127, every nibble = 2 (=1.0).
        let row = vec![1.0f32; 64];
        let packed = quantize_hfp4g32_row(&row);
        let recovered = dequant_hfp4g32_row(&packed, 64);
        for (i, &v) in recovered.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-2, "elem {} recovered to {}", i, v);
        }
    }

    #[test]
    fn round_trip_mixed_magnitudes() {
        // Row with mixed positive/negative E2M1 magnitudes — should round-trip exactly.
        let row: Vec<f32> = (0..64)
            .map(|i| {
                let v = E2M1_LUT[i % 16];
                v * 6.0 // scale up so row_scale_a sees max abs at 6 * 6 = 36, brings code lattice back to [-6, 6]
            })
            .collect();
        let packed = quantize_hfp4g32_row(&row);
        let recovered = dequant_hfp4g32_row(&packed, 64);
        // Bound: |recovered - input| ≤ row_scale * 2^(block_e - 127) * 0.5 (half min E2M1 step).
        // With row_scale_a = 36/6 = 6, and block_max_normalized = 6, block_e = 127 → step ≈ 0.5 → tol = 3.0.
        // Actual tolerance should be much tighter for exact lattice values; allow some headroom.
        for (i, (&got, &want)) in recovered.iter().zip(row.iter()).enumerate() {
            let rel_err = (got - want).abs() / want.abs().max(1.0);
            assert!(
                rel_err < 0.1,
                "elem {}: got {} want {} rel_err {}",
                i,
                got,
                want,
                rel_err
            );
        }
    }

    #[test]
    fn round_trip_per_block_error_bound() {
        // Mathematical guarantee: for every element, |recovered - original| must be ≤
        //   row_scale_a * 2^(block_e - 127) * (max_E2M1_step / 2)
        // = effective_block_scale * 1.0  (max E2M1 step is 2.0, half = 1.0)
        //
        // This is the format's correctness contract; if this fails we have a real bug.
        // NRMSE quality on raw weights is a downstream concern (MXFP4 family is documented
        // as needing rotation+smoothing for production accuracy — that's MFP4G32 in v1.5).
        let mut rng_state: u64 = 0xdead_beef_dead_beef;
        let mut next_uniform = || -> f32 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            ((rng_state & 0x00FF_FFFF) as f32 / 0x0100_0000 as f32).max(1e-7)
        };
        // Box-Muller Gaussian std=0.5.
        let row: Vec<f32> = (0..512)
            .flat_map(|_| {
                let u1 = next_uniform();
                let u2 = next_uniform();
                let r = (-2.0 * u1.ln()).sqrt();
                let t = 2.0 * std::f32::consts::PI * u2;
                [r * t.cos() * 0.5, r * t.sin() * 0.5]
            })
            .collect();

        let k = row.len();
        let packed = quantize_hfp4g32_row(&row);
        let recovered = dequant_hfp4g32_row(&packed, k);

        let row_scale_a = f16_to_f32(u16::from_le_bytes([packed[0], packed[1]]));

        // Per-block half-max-step bound. Allow 1% slack for FP16 row-scale rounding.
        for b in 0..(k / 32) {
            let payload_off = 16 + b * 17;
            let block_e = packed[payload_off] as i32;
            let block_scale = ((block_e - 127) as f32).exp2();
            // Max E2M1 step is 2.0 (between 4 and 6); half = 1.0. Round-trip element error must
            // be ≤ effective block scale × 1.0 × (1 + slack). Slack absorbs FP16 row-scale rounding.
            let bound = row_scale_a * block_scale * 1.0 * 1.01 + 1e-5;
            for i in 0..32 {
                let idx = b * 32 + i;
                let err = (recovered[idx] - row[idx]).abs();
                assert!(err <= bound,
                        "block {} elem {} err {} exceeds bound {} (block_e={}, row_scale_a={}, block_scale={})",
                        b, i, err, bound, block_e, row_scale_a, block_scale);
            }
        }
    }

    #[test]
    fn header_layout_matches_spec() {
        // 64 elements = 2 blocks. Row size: 16 + 2*17 = 50 bytes.
        let row = vec![3.0f32; 64];
        let packed = quantize_hfp4g32_row(&row);
        assert_eq!(packed.len(), 50);
        // Block count == 2.
        let bc = u16::from_le_bytes([packed[4], packed[5]]);
        assert_eq!(bc, 2);
        // Format flags: rotation off, no row_scale_b.
        assert_eq!(packed[6] & 0x0F, 0);
        // First block UE8M0 byte at offset 16.
        // Last block payload ends at 16 + 2*17 = 50 (= total).
        // Sanity: row_scale_a > 0 (FP16 bits non-zero).
        let rs_bits = u16::from_le_bytes([packed[0], packed[1]]);
        assert_ne!(rs_bits, 0);
    }

    #[test]
    fn mfp4_stamps_rotation_flag() {
        // MFP4G32 must stamp format_flags = 0x05 (bit 0 + bits 2-3 = 01) in every row
        // header so loaders/tooling can detect the offline-FWHT variant. Byte length must
        // match HFP4G32 (only the flag byte and the rotated weight content differ).
        let m = 3;
        let k = 256;
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let f32_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
        let packed = quantize_mfp4g32_2d(&f32_data, m, k, &signs1, &signs2);
        let row_bytes = 16 + 17 * (k / 32);
        assert_eq!(packed.len(), m * row_bytes, "MFP4G32 byte length mismatch");
        for r in 0..m {
            let off = r * row_bytes;
            assert_eq!(
                packed[off + 6],
                0x05,
                "row {} format_flags expected 0x05, got {:#x}",
                r,
                packed[off + 6]
            );
            // block_count must equal k/32.
            let bc = u16::from_le_bytes([packed[off + 4], packed[off + 5]]);
            assert_eq!(bc as usize, k / 32);
        }
    }

    // Orthogonality of the FWHT (`dot(R(W), R(x)) ≈ dot(W, x)`) is the load-bearing
    // correctness property and is empirically validated by `examples/test_gemv_mfp4g32.rs`
    // across K = {512, 1024, 1280, 1536, 1792, 2048} on real GPU hardware (max-abs error
    // ≤ 1.14e-5 vs 5e-3 tolerance — three orders of magnitude under). A CPU-only unit test
    // can't tighten that further without duplicating the GPU's CPU-reference path.

    #[test]
    fn mfp4_lloyd_round_trip_cpu() {
        let (m, k) = (64usize, 256usize);
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        // Deterministic pseudo-random weights (xorshift64).
        let mut s: u64 = 0x1234_5678_9abc_def0;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s & 0xFF_FFFF) as f32 / 0xFF_FFFF_u32 as f32) * 2.0 - 1.0
        };
        let data: Vec<f32> = (0..m * k).map(|_| 0.5 * rnd()).collect();
        let packed = quantize_mfp4g32_lloyd_2d(&data, m, k, &signs1, &signs2);
        let row_bytes = 16 + 17 * (k / 32);
        assert_eq!(
            packed.len(),
            32 + m * row_bytes,
            "byte size incl 32-B prefix"
        );
        // Build reference: FWHT-rotated original (same as what dequant returns).
        let mut rotated_ref = vec![0.0f32; m * k];
        let mut row_buf = vec![0.0f32; k];
        for r in 0..m {
            row_buf.copy_from_slice(&data[r * k..(r + 1) * k]);
            for seg in 0..(k / 256) {
                cpu_fwht_256(&mut row_buf[seg * 256..(seg + 1) * 256], &signs1, &signs2);
            }
            rotated_ref[r * k..(r + 1) * k].copy_from_slice(&row_buf);
        }
        // Compare dequant output (rotated domain) to the ROTATED reference.
        let recon_rot = dequant_mfp4g32_lloyd(&packed, m, k);
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..m * k {
            let e = (recon_rot[i] - rotated_ref[i]) as f64;
            num += e * e;
            den += (rotated_ref[i] as f64).powi(2);
        }
        let nrmse = (num / den).sqrt();
        // FWHT homogenizes magnitudes; 4-bit Lloyd error in rotated domain should be small.
        assert!(
            nrmse < 0.15,
            "mfp4L NRMSE {} too high (layout/codebook bug)",
            nrmse
        );
    }
}

/// MagnumQuant MQ3-G256: FWHT-rotated 3-bit quantization.
/// Same binary format as HFQ3-G256 (104 bytes/group). Rotation is baked into
/// the weights via cpu_fwht_256; the GEMV kernel rotates x instead.
fn quantize_mq3g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
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
fn quantize_mq2g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
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

/// Encode an f32 to IEEE-754 fp16 bits (round-to-nearest-even, no NaN/Inf preservation
/// beyond the trivial case — block centroids are bounded means of fp32 weights so
/// the simple path is safe).
fn f32_to_fp16_bits(v: f32) -> u16 {
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
fn safetensors_to_imatrix_key(parent: &str) -> Option<(String, usize)> {
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
fn imatrix_col_weights_for_parent(
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
fn imatrix_expert_counts_for_parent(
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
fn imatrix_in_sum2_for_parent(
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
fn imatrix_layer_activation_counts(
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
fn quantize_mq2g256_lloyd_weighted(
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
fn quantize_mq2g256_lloyd_gptq(
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

fn quantize_mq2g256_lloyd_gptq_with_damping(
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
fn quantize_mq2g256_lloyd_k3(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
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
fn cpu_inv_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
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
fn dequantize_mq2g256_lloyd_to_f32(
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
fn mq2g256gl_roundtrip_f32(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<f32> {
    /// Lloyd–Max levels for a unit Gaussian at 2 bit.
    const CB: [f32; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];
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
fn gl_encode_block(group: &[f32; 256], cb: &[f32], idx: &mut [u8; 256]) -> u16 {
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
fn quantize_hfq3g256(f32_data: &[f32]) -> Vec<u8> {
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
fn quantize_hfq3g128(f32_data: &[f32]) -> Vec<u8> {
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
fn quantize_hfq2g256(f32_data: &[f32]) -> Vec<u8> {
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
fn quantize_hfq2g128(f32_data: &[f32]) -> Vec<u8> {
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
fn quantize_hfq4g128(f32_data: &[f32]) -> Vec<u8> {
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

// ─── HFQ File Format ────────────────────────────────────────────────────────

const HFQ_MAGIC: &[u8; 4] = b"HFQM";
const HFQ_VERSION: u32 = 1;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QuantType {
    Q4F16G64 = 0,
    F16 = 1,
    F32 = 2,
    Q8F16 = 3,
    Q4K = 4,
    Q8HFQ = 5,
    HFQ4G256 = 6,
    HFQ4G128 = 7,
    HFQ6G256 = 8,
    HFQ2G256 = 9,
    HFQ2G128 = 10,
    HFQ3G256 = 11,
    HFQ3G128 = 12,
    MQ4G256 = 13,      // MagnumQuant: FWHT-rotated HFQ4-G256
    MQ8G256 = 14,      // MagnumQuant: FWHT-rotated symmetric INT8, dp4a target
    MQ6G256 = 15,      // MagnumQuant: FWHT-rotated HFQ6-G256 (6-bit, 200 B/group)
    BF16 = 16,         // Original BF16 weights (zero precision loss for vision)
    MQ3G256 = 17,      // MagnumQuant: FWHT-rotated HFQ3-G256 (3-bit, 104 B/group)
    MQ2G256 = 18,      // MagnumQuant: FWHT-rotated HFQ2-G256 (2-bit, 72 B/group)
    MQ2G256Lloyd = 19, // MagnumQuant 2-bit + per-block Lloyd-Max 4-entry fp16 codebook (72 B/group)
    MQ3G256Lloyd = 20, // MagnumQuant 3-bit + per-block Lloyd-Max 8-entry fp16 codebook (112 B/group)
    // HFP4 family — RDNA-optimal FP4 (E2M1 elements + UE8M0 block scale + FP16 row scale).
    // See docs/quant-formats/hfp4.md for byte layout, dequant, rotation modes.
    // Per-row header is 16 B; per-block payload is (1 + g/2) bytes (UE8M0 + nibbles).
    HFP4G32 = 21, // E2M1 + UE8M0 g32 + FP16 row scale — canonical (FP8-WMMA-K aligned)
    // MFP4G32 = HFP4G32 + offline FWHT rotation (256-element FWHT applied to weights at quant time;
    // runtime applies the same FWHT to x via mq_rotate_x). format_flags bit 0 + bits 2-3 = 0b0101
    // signals "rotation present, offline FWHT" for future interop/detection.
    MFP4G32 = 24, // v1.5 — HFP4G32 + offline FWHT (drop-in MQ4 replacement)
    /// I64→U32 downcast of DeepSeek V4 hash-routing `tid2eid` lookup tables.
    /// Shape `[vocab, num_experts_per_tok]`. Stored as raw u32 LE; the
    /// loader reads `bytes.chunks_exact(4)`. ID 22 was reserved for the
    /// HFP4G16 NV-aligned ablation (never built) — we re-use the slot
    /// for tid2eid storage to stay byte-compatible with antirezQ8.hfq.
    TidI32 = 22,
    // Reserved IDs — DO NOT REUSE for unrelated formats. Documented in docs/quant-formats/hfp4.md.
    // HFP4G16     = 22, // v1.5 — NV-aligned FP16-WMMA-K alignment ablation (re-used by TidI32)
    // HFP4G64     = 23, // v1.5 — RDNA1/2 sweet-spot ablation
    // HFP4G32MX   = 25, // v2  — strict OCP MXFP4 interop alias (no row scale, UE8M0 only)
    // HFP4G16NV   = 26, // v2  — strict NVFP4 interop alias (E4M3 scale + FP32 tensor)
    // HFP8E4M3G32 = 27, // v2  — HFP8 E4M3 family
    PARO4G128 = 28,  // ParoQuant native AWQ W4 + pairwise activation rotation metadata
    PARO4G128T = 29, // ParoQuant engine-tiled qweight [M/8, K] for coalesced GEMV reads
    // MFP4G32R    = 29, // v3  — HFP4G32 + online block-diag-128 rotation (AMD recipe)
    // HFP8E5M2G32 = 30, // v2  — HFP8 E5M2 family
    MQ4G256Lloyd = 30, // MagnumQuant 4-bit + per-block Lloyd-Max 16-entry fp16 codebook (160 B/group)
    // Renumbered from 21 → 30 in mq4-lloyd merge to avoid HFP4G32=21 collision.
    // Models quantized pre-renumber MUST be re-quantized.
    MQ5G256 = 31,      // MagnumQuant: FWHT-rotated 5-bit (168 B/group, 5.25 bpw).
    MFP4G32Lloyd = 32, // mfp4 (E2M1+UE8M0 g32+FP16 row scale+offline FWHT) with the fixed
    // E2M1 grid replaced by ONE per-tensor 16-entry fp16 Lloyd codebook
    // prepended (32 B) before row 0. Rows byte-identical to MFP4G32 (qt 24).
    // 8B affine header (f32 scale + f32 min) + 160B payload
    // (5 bits × 256, cross-byte: 8 codes per 5 bytes). NOTE: 16=BF16.
    MFP4G32P = 33, // mfp4+P: mfp4 (E2M1+FP16 row scale+offline FWHT) with the per-32-block
    // UE8M0 scale promoted to E4M3 (FP8, non-power-of-2). Byte layout
    // BYTE-IDENTICAL to MFP4G32 (qt 24): 16-B hdr + n_blocks×17 B, NO prefix.
    // Only the per-block scale byte's meaning differs (E4M3 vs UE8M0).
    MFP4G32E8 = 34, // mfp4-E8: mfp4+P container (E4M3 block scale, NO prefix, same row_bytes)
    // with the 32 E2M1 nibbles replaced by 4x32-bit E8-lattice codewords
    // (8 weights/codeword, QUANT_STEP=0.88). 4.25 bpw, FWHT rotation.
    // MQ*-GL ("global Lloyd"): N-bit codes against ONE tensor-global codebook
    // plus a per-block fp16 scale, in structure-of-arrays layout:
    //     [0 .. M*gpr*P)                  packed N-bit indices, P B/group
    //     [M*gpr*P .. +M*gpr*2)           fp16 per-block scales
    // vs the per-block Lloyd formats (qt 19/20), which interleave a fitted
    // 2^N-entry fp16 codebook into every group.
    //
    // Rationale (measured 2026-08-04, docs/investigations/2026-08-04-a3b-lowbit-quality.md):
    // post-FWHT blocks are Gaussian by CLT, so the optimal LEVEL SHAPE is the
    // same in every block — a per-block fit re-derives it ~4000x per tensor and
    // differs only by scale. Fitting a global codebook on 28.3M real a3b expert
    // weights reproduces the textbook Lloyd-Max Gaussian levels to 3 decimals.
    // Cost on real weights: +2.35% NRMSE / +1.16% end-to-end KLD, for -0.1875 bpw
    // (MQ2) — and the group base becomes naturally aligned (64 B vs 72 B stride).
    MQ2G256GL = 38, // 2-bit + global codebook: 64 B idx/group + 2 B scale = 2.0625 bpw
    MQ3G256GL = 39, // 3-bit + global codebook: 96 B idx/group + 2 B scale = 3.0625 bpw
    MFP4G32E8SOA = 35, // mfp4-E8 SoA: same E8 data as qt=34 but in structure-of-arrays layout.
    // [16B hdr] + [n_blocks B E4M3 scales, pad 16B] + [n_blocks*16B codewords].
    MFP3G32E8 = 36, // mfp3-E8: MFP4G32E8 frame, 3-bit lattice (center 3), 13 B/blk, 3.25 bpw.
    // Drop-in cold tier for MQ3G256Lloyd (tag 3 → tag 5).
    MFP2G32E8 = 37, // mfp2-E8: MFP4G32E8 frame, 2-bit lattice (center 1), 9 B/blk, 2.25 bpw.
                    // Drop-in cold tier for MQ2G256Lloyd (tag 1 → tag 6).
}

/// Per-tensor precision level assigned by the K-map pre-pass.
/// Determines whether a tensor gets the base format, a 6-bit promotion,
/// Q8, or F16. See docs/superpowers/specs/2026-05-08-mixed-quant-kmap-design.md.
#[derive(Clone, Copy, Debug, PartialEq)]
enum QuantLevel {
    /// Store as F16 (norms, biases, 1D tensors).
    F16,
    /// Store as Q8_F16 (embeddings, lm_head, MoE routers).
    Q8,
    /// Promote to 6-bit variant of the base format (edge layers, MoE expert FFN).
    Promote6,
    /// Override the default for a specific tensor class (today: lm_head)
    /// to a CLI-specified format. Currently unused on this branch (no emission
    /// site); kept so origin/master's lm_head-format override match arms
    /// compile after the merge. Re-wire to `--lm-head-format` when the
    /// configurable-kmap-pair refactor lands here.
    #[allow(dead_code)]
    Override(GgufFormat),
    /// Use the base format as-is.
    Base,
}

/// Default kmap promote target for a given base format. Preserves the
/// pre-`--kmap-promote` behavior byte-for-byte: MQ-family bases promote to
/// MQ6, HFQ-family to HFQ6, FP4-family is a no-op (no FP6 sibling).
fn default_promote_target(base: GgufFormat) -> GgufFormat {
    match base {
        GgufFormat::Mq2
        | GgufFormat::Mq3
        | GgufFormat::Mq4
        | GgufFormat::Mq5
        | GgufFormat::Mq6
        | GgufFormat::Mq2Lloyd
        | GgufFormat::Mq3Lloyd
        | GgufFormat::Mq4Lloyd => GgufFormat::Mq6,
        GgufFormat::Hfq4 | GgufFormat::Hfq6 => GgufFormat::Hfq6,
        GgufFormat::Hfp4 => GgufFormat::Hfp4,
        GgufFormat::Mfp4 => GgufFormat::Mfp4,
        GgufFormat::Mfp4Lloyd => GgufFormat::Mfp4Lloyd,
        GgufFormat::Mfp4P => GgufFormat::Mfp4P,
        GgufFormat::Mfp4E8 => GgufFormat::Mfp4E8,
        GgufFormat::Mfp4E8Soa => GgufFormat::Mfp4E8Soa,
        GgufFormat::Mfp3E8 => GgufFormat::Mfp3E8,
        GgufFormat::Mfp2E8 => GgufFormat::Mfp2E8,
    }
}

/// Allowlist for explicit `--kmap-promote` overrides. Runtime mixed-format
/// dispatch (post-#257) is validated only within same-rotation-family,
/// upward-in-bit-width pairings. Cross-family (MQ↔HFQ, MQ↔HFP) and
/// downward-in-bits promotions are rejected at parse time.
fn is_promote_pair_supported(base: GgufFormat, promote: GgufFormat) -> bool {
    use GgufFormat::*;
    if base == promote {
        return true; // no-op promotion is always safe
    }
    match (base, promote) {
        // Lloyd-to-Lloyd only — Lloyd variants use different codebooks +
        // different runtime kernel families from standard MQ. Lloyd→non-Lloyd
        // mixed-format dispatch has no runtime support today; the plan's
        // "Future expansion" section targets the MQ2-Lloyd + MQ3-Lloyd pair
        // specifically. Tightened per combined-review finding G2.
        (Mq2Lloyd, Mq3Lloyd) => true,
        (Mq2Lloyd | Mq3Lloyd, _) => false,
        (_, Mq2Lloyd | Mq3Lloyd) => false,

        // MQ-family upward bit-width (non-Lloyd)
        (Mq2, Mq3 | Mq4 | Mq5 | Mq6) => true,
        (Mq3, Mq4 | Mq5 | Mq6) => true,
        (Mq4, Mq5 | Mq6) => true,
        (Mq5, Mq6) => true,

        // HFQ-family upward bit-width
        (Hfq4, Hfq6) => true,

        // Everything else: explicitly not in the supported matrix.
        // Cross-family (MQ↔HFQ↔FP4) rejected — runtime mixed-format dispatch
        // (post-#257) is only same-rotation-family-safe.
        _ => false,
    }
}

/// Extract layer index from a tensor name.
/// Handles both safetensors (`layers.{N}.`) and GGUF (`blk.{N}.`) patterns.
/// Uses unanchored search to handle any prefix (model.layers, model.language_model.layers, etc.).
fn parse_layer_idx(name: &str) -> Option<usize> {
    // Try safetensors pattern: "layers.{N}."
    if let Some(pos) = name.find("layers.") {
        let after = &name[pos + 7..]; // skip "layers."
        if let Some(dot) = after.find('.') {
            if let Ok(idx) = after[..dot].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    // Try GGUF pattern: "blk.{N}."
    if let Some(pos) = name.find("blk.") {
        let after = &name[pos + 4..]; // skip "blk."
        if let Some(dot) = after.find('.') {
            if let Ok(idx) = after[..dot].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    None
}

/// Stride for alternating-mode promotion: edge layers always promoted,
/// plus every Nth middle layer. 3 was chosen empirically — promotes ~40%
/// of middle layers, matching llama.cpp Q4_K_M's budget-allocation pattern.
/// On MoE 3.6-35B-A3B: stride=3 gives PPL 8K=19.96 at 21.8 GB vs full
/// K-map PPL 8K=20.07 at 27.7 GB.
const ALTERNATING_STRIDE: usize = 3;

/// llama.cpp-style alternating promotion: edge layers always promoted,
/// middle layers promoted every `stride` layers.
fn is_positional_promote(idx: usize, n_layers: usize, stride: usize) -> bool {
    if n_layers == 0 || stride == 0 {
        return false;
    }
    if idx < 2 || idx >= n_layers.saturating_sub(2) {
        return true;
    }
    (idx - 2) % stride == 0
}

/// Resolve the quantization level for a tensor based on its name, the model's
/// layer count, whether the model is MoE, and the K-map mode.
///
/// `kmap_mode`: 0 = full (all candidates promoted), 1 = alternating
/// (experts + ffn_down every 3rd middle layer, edge layers always),
/// 2 = typed (ffn_down + attn_v everywhere).
///
/// Note: In the safetensors path, norms/biases are filtered by `should_quantize()`
/// before this function is called. Rules 1-2 exist for the GGUF path and completeness.
fn kmap_resolve(name: &str, n_layers: usize, is_moe: bool) -> QuantLevel {
    kmap_resolve_mode(name, n_layers, is_moe, 0)
}

fn kmap_resolve_mode(name: &str, n_layers: usize, is_moe: bool, kmap_mode: u8) -> QuantLevel {
    // Rule 1: norms, biases, 1D (GGUF path mainly)
    if name.contains("norm") || name.contains("bias") {
        return QuantLevel::F16;
    }

    // Rule 2: embeddings, lm_head, output projection
    if name.contains("embed_tokens")
        || name.contains("token_embd")
        || name.contains("lm_head")
        || name.ends_with("output.weight")
    {
        return QuantLevel::Q8;
    }

    // Rule 3: MoE routers
    if is_moe && (name.ends_with("mlp.gate.weight") || name.contains("shared_expert_gate")) {
        return QuantLevel::Q8;
    }

    // Rule 4: MoE expert FFN weights
    if is_moe && name.contains("mlp.experts.") {
        if kmap_mode == 1 {
            // Alternating: promote expert groups only in positional layers
            if let Some(idx) = parse_layer_idx(name) {
                if is_positional_promote(idx, n_layers, ALTERNATING_STRIDE) {
                    return QuantLevel::Promote6;
                }
                return QuantLevel::Base;
            }
        }
        return QuantLevel::Promote6;
    }

    // Mode 2 (typed): promote ffn_down and attn_v in all layers.
    if kmap_mode == 2 {
        let is_down = name.contains("down_proj") || name.contains("ffn_down");
        let is_v = name.contains("v_proj") || name.contains("attn_v");
        if is_down || is_v {
            return QuantLevel::Promote6;
        }
        if n_layers > 0 {
            if let Some(idx) = parse_layer_idx(name) {
                if idx < 2 || idx >= n_layers.saturating_sub(2) {
                    return QuantLevel::Promote6;
                }
            }
        }
        return QuantLevel::Base;
    }

    // Mode 1 (alternating): ffn_down in edge + every 3rd middle layer.
    // Edge-layer rule mirrors mode 0 below: attn+FFN for MoE (full promotion
    // gives -19.8% PPL on 3.6-35B-A3B), FFN only for dense (attn promotion
    // regresses PPL +3.1% on 27B). Bench: asym4 KV, ctx=8192, wikitext-2-test.
    // See ppl_kmap_20260508.md.
    if kmap_mode == 1 {
        let is_down = name.contains("down_proj") || name.contains("ffn_down");
        if n_layers > 0 {
            if let Some(idx) = parse_layer_idx(name) {
                if is_down && is_positional_promote(idx, n_layers, ALTERNATING_STRIDE) {
                    return QuantLevel::Promote6;
                }
                // Edge layers: attn+FFN for MoE, FFN only for dense.
                if idx < 2 || idx >= n_layers.saturating_sub(2) {
                    if is_moe {
                        return QuantLevel::Promote6;
                    }
                    let is_ffn = name.contains("mlp.") || name.contains("ffn");
                    if is_ffn {
                        return QuantLevel::Promote6;
                    }
                }
            }
        }
        return QuantLevel::Base;
    }

    // Rule 5 (full mode 0): edge layers (first 2 + last 2).
    // Dense models: FFN only — attn promotion regresses PPL (+3.1% on 27B).
    // MoE models: attn+FFN — full promotion gives -19.8% PPL on 3.6-35B-A3B.
    // Bench: asym4 KV, ctx=8192, wikitext-2-test. See ppl_kmap_20260508.md.
    if n_layers > 0 {
        if let Some(idx) = parse_layer_idx(name) {
            if idx < 2 || idx >= n_layers.saturating_sub(2) {
                if is_moe {
                    // MoE: promote all tensors in edge layers (attn + FFN)
                    return QuantLevel::Promote6;
                }
                // Dense: promote FFN only — attn stays at Base
                let is_ffn = name.contains("mlp.") || name.contains("ffn");
                if is_ffn {
                    return QuantLevel::Promote6;
                }
            }
        }
    }

    // Rule 6: everything else
    QuantLevel::Base
}

#[derive(Debug)]
pub(crate) struct HfqTensor {
    pub(crate) name: String,
    pub(crate) quant_type: QuantType,
    pub(crate) shape: Vec<u32>,
    pub(crate) group_size: u32,
    pub(crate) data: Vec<u8>,
    /// When data is spilled to disk, this holds the byte count.
    /// `data` is empty and the bytes live in the spill file.
    pub(crate) spilled_len: u64,
}

/// Streaming tensor spill file. When the quantizer accumulates more than
/// `SPILL_THRESHOLD` bytes of tensor data in memory, it flushes completed
/// tensors to this file. At write_hfq time, spilled data is copied from
/// the spill file instead of from memory, keeping peak RSS bounded.
struct TensorSpill {
    file: std::io::BufWriter<File>,
    path: PathBuf,
    offset: u64,
}

impl TensorSpill {
    fn new(dir: &Path) -> std::io::Result<Self> {
        // PID-unique so concurrent quantize runs in the same output dir don't
        // share a spill path (a sibling run's Drop would otherwise delete this
        // run's spill file → write_hfq NotFound panic).
        let path = dir.join(format!(".hipfire_quant_spill.{}.tmp", std::process::id()));
        let file = std::io::BufWriter::with_capacity(4 * 1024 * 1024, File::create(&path)?);
        Ok(Self {
            file,
            path,
            offset: 0,
        })
    }

    /// Write tensor data to the spill file. Returns the byte count written.
    fn spill(&mut self, data: &[u8]) -> std::io::Result<u64> {
        use std::io::Write;
        self.file.write_all(data)?;
        self.offset += data.len() as u64;
        Ok(data.len() as u64)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        use std::io::Write;
        self.file.flush()
    }

    fn cleanup(self) {
        // Explicit cleanup — Drop impl handles the actual removal.
        drop(self);
    }
}

impl Drop for TensorSpill {
    fn drop(&mut self) {
        // Ensure the temp file is removed even on panic.
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Spill tensors whose data is in memory to the spill file, freeing RAM.
/// Called after each layer's expert batch to keep peak RSS bounded.
fn maybe_spill(tensors: &mut [HfqTensor], spill: &mut TensorSpill, threshold: usize) {
    let in_mem: usize = tensors
        .iter()
        .filter(|t| t.spilled_len == 0)
        .map(|t| t.data.len())
        .sum();
    if in_mem < threshold {
        return;
    }
    for t in tensors.iter_mut() {
        if t.spilled_len == 0 && !t.data.is_empty() {
            let len = spill.spill(&t.data).unwrap_or(0);
            t.spilled_len = len;
            t.data = Vec::new(); // free the memory
        }
    }
    let _ = spill.flush();
}

/// The arch-correct config field naming the routed-expert count, as the arch's
/// loader config parser reads it from the HFQ metadata `config` object:
///   * deepseek4 → `n_routed_experts`
///   * qwen3.5-moe / lfm2moe → `num_experts`
///   * minimax → `num_local_experts`
fn reap_expert_count_field(arch: reap_overlay::ReapArch) -> &'static str {
    match arch {
        reap_overlay::ReapArch::Deepseek4 => "n_routed_experts",
        reap_overlay::ReapArch::Qwen35 => "num_experts",
        reap_overlay::ReapArch::Lfm2Moe => "num_experts",
        reap_overlay::ReapArch::Minimax => "num_local_experts",
    }
}

/// Patch the HFQ metadata envelope's `config` so the routed-expert count reads
/// `kept` (the pruned/compact count) for `arch`. The arch loaders parse the
/// inner `config` object (qwen3.5 additionally descends into `config.text_config`
/// when present), so patch the field WHEREVER it currently exists under `config`:
/// at `config[field]` and, if present, at `config.text_config[field]`. Erroring
/// (rather than silently no-op'ing) when the field is absent prevents shipping a
/// baked model whose metadata still claims the original expert count.
fn patch_expert_count_metadata(
    metadata_json: &str,
    arch: reap_overlay::ReapArch,
    kept: usize,
) -> Result<String, String> {
    let field = reap_expert_count_field(arch);
    let mut v: serde_json::Value =
        serde_json::from_str(metadata_json).map_err(|e| format!("metadata not valid JSON: {e}"))?;
    let config = v
        .get_mut("config")
        .ok_or_else(|| "metadata missing `config` object".to_string())?;
    let mut patched = false;
    if config.get(field).is_some() {
        config[field] = serde_json::json!(kept);
        patched = true;
    }
    if let Some(tc) = config.get_mut("text_config") {
        if tc.get(field).is_some() {
            tc[field] = serde_json::json!(kept);
            patched = true;
        }
    }
    if !patched {
        return Err(format!(
            "expert-count field '{field}' not found under config (or config.text_config) for {arch:?}"
        ));
    }
    serde_json::to_string(&v).map_err(|e| format!("re-serialize metadata: {e}"))
}

fn write_hfq(
    path: &Path,
    arch: u32,
    metadata_json: &str,
    tensors: &[HfqTensor],
    spill: Option<&mut TensorSpill>,
) -> std::io::Result<()> {
    let mut f = File::create(path)?;

    let metadata_bytes = metadata_json.as_bytes();

    // Calculate offsets
    let header_size = 32u64;
    let metadata_offset = header_size;
    let metadata_size = metadata_bytes.len() as u64;

    // Tensor index follows metadata
    let index_offset = metadata_offset + metadata_size;
    let mut index_bytes = Vec::new();
    // Write tensor count
    index_bytes.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for t in tensors {
        // name length + name
        let name_bytes = t.name.as_bytes();
        index_bytes.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        index_bytes.extend_from_slice(name_bytes);
        // quant type
        index_bytes.push(t.quant_type as u8);
        // n_dims + shape
        index_bytes.push(t.shape.len() as u8);
        for &d in &t.shape {
            index_bytes.extend_from_slice(&d.to_le_bytes());
        }
        // group size
        index_bytes.extend_from_slice(&t.group_size.to_le_bytes());
        // data size (offset computed at read time from cumulative sizes)
        let data_len = if t.spilled_len > 0 {
            t.spilled_len
        } else {
            t.data.len() as u64
        };
        index_bytes.extend_from_slice(&data_len.to_le_bytes());
    }

    // Data starts after index, aligned to 4096
    let data_start_unaligned = index_offset + index_bytes.len() as u64;
    let data_offset = (data_start_unaligned + 4095) & !4095;

    // Write header (32 bytes)
    f.write_all(HFQ_MAGIC)?;
    f.write_all(&HFQ_VERSION.to_le_bytes())?;
    f.write_all(&arch.to_le_bytes())?;
    f.write_all(&(tensors.len() as u32).to_le_bytes())?;
    f.write_all(&metadata_offset.to_le_bytes())?;
    f.write_all(&data_offset.to_le_bytes())?;

    // Write metadata
    f.write_all(metadata_bytes)?;

    // Write tensor index
    f.write_all(&index_bytes)?;

    // Pad to data alignment
    let pad_size = (data_offset - data_start_unaligned) as usize;
    f.write_all(&vec![0u8; pad_size])?;

    // Write tensor data — from spill file or from memory
    if let Some(spill) = spill {
        let _ = spill.flush();
        let mut spill_reader = std::io::BufReader::new(File::open(&spill.path)?);
        let mut buf = vec![0u8; 4 * 1024 * 1024]; // 4 MB copy buffer
        for t in tensors {
            if t.spilled_len > 0 {
                // Copy from spill file
                let mut remaining = t.spilled_len as usize;
                while remaining > 0 {
                    let chunk = remaining.min(buf.len());
                    use std::io::Read;
                    spill_reader.read_exact(&mut buf[..chunk])?;
                    f.write_all(&buf[..chunk])?;
                    remaining -= chunk;
                }
            } else {
                f.write_all(&t.data)?;
            }
        }
    } else {
        for t in tensors {
            f.write_all(&t.data)?;
        }
    }

    Ok(())
}

// ─── Model Discovery ────────────────────────────────────────────────────────

fn find_safetensors(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    files.sort();
    files
}

/// Determine which tensors to quantize (weight matrices) vs keep as F16 (norms, embeddings)
fn should_quantize(name: &str) -> bool {
    // Vision encoder weights stay FP16 (only ~500M params, run once per image).
    // Qwen3.5-VL uses `model.visual.*` / `visual.*`; dots.ocr uses
    // `vision_tower.*`. Both arches keep vision F16 during bring-up so the
    // per-stage diff against the HF reference activations
    // (`benchmarks/references/<image>_activations/`) doesn't have to absorb
    // both forward-pass implementation noise AND quant noise — clean
    // attribution. See memory `feedback_dots_ocr_vision_f16_during_bringup`.
    if name.starts_with("model.visual.")
        || name.starts_with("visual.")
        || name.starts_with("vision_tower.")
    {
        return false;
    }
    if name.contains("norm") || name.contains("bias") {
        return false;
    }
    // Quantize everything including embeddings (Q8 embedding saves ~2.3GB for 8B models)
    name.contains("weight")
}

/// antirez ds4 reference keeps three classes at F16 because Q8 measurably
/// regresses PPL on DeepSeek V4: (1) attn compressor wkv + wgate, (2) indexer wq_b +
/// weights_proj, (3) indexer compressor wkv + wgate. All small (≤32 MiB
/// combined across 43 layers).
///
/// Router gate.weight (.ffn.gate.weight) is NOT kept at F16: antirez
/// actually ships it as MQ4G256, and the known-good DeepSeek V4 quant
/// matches. Falling back to the format's default (Q8F16 in deepseek4-q8-mtp)
/// is fine — the router is dispatched via `gemv_auto`.
///
/// `attn.indexer.compressor.*` is a substring of `attn.compressor.*` only
/// in the literal-prefix sense, so order doesn't matter — the substring
/// `.compressor.wkv.weight` matches both `.attn.compressor.wkv.weight` and
/// `.attn.indexer.compressor.wkv.weight` deliberately.
fn is_deepseek4_keep_f16(name: &str) -> bool {
    name.ends_with(".compressor.wkv.weight")
        || name.ends_with(".compressor.wgate.weight")
        || name.ends_with(".indexer.wq_b.weight")
        || name.ends_with(".indexer.weights_proj.weight")
}

/// For mixed quant: should this tensor be Q8 (fast) or Q4 (compressed)?
/// Q8: attention weights, embeddings, lm_head (need occupancy)
/// Q4: FFN weights (bulk of model, benefits from compression)
/// Which fixed-tier classes a tensor belongs to, for `HIPFIRE_Q8_CLASSES`.
/// Ordered cheapest-to-keep first by measured per-token bytes on a3b:
/// router+gate 11.1 MB, lm_head 270 MB, attention 682 MB (of a 1031 MB fixed
/// tier at MQ4). Attention is 66% of the fixed tier, lm_head 26%.
fn q8_class_of(name: &str) -> Option<&'static str> {
    if name.contains("lm_head") {
        Some("lm_head")
    } else if name.contains("embed") {
        Some("embed")
    } else if name.ends_with("mlp.gate.weight") || name.ends_with("mlp.shared_expert_gate.weight") {
        Some("router")
    } else if name.contains("self_attn")
        || name.contains("attn_q")
        || name.contains("attn_k")
        || name.contains("attn_v")
        || name.contains("attn_output")
        || name.contains("q_proj")
        || name.contains("k_proj")
        || name.contains("v_proj")
        || name.contains("o_proj")
        // Qwen3.5 DeltaNet attention
        || name.contains("linear_attn")
    {
        Some("attn")
    } else {
        None
    }
}

/// Fixed-tier tensors held at Q8F16 regardless of `--format`.
///
/// `HIPFIRE_Q8_CLASSES=<comma list>` narrows this to a subset of
/// {`lm_head`, `embed`, `router`, `attn`} — the lever for attributing which
/// fixed class actually carries the quality. Measured 2026-08-04: dropping the
/// WHOLE fixed tier Q8 -> MQ4 costs **+35.2% KLD** (0.1742 -> 0.2356) while
/// buying 1.75x decode speed, so the tier is emphatically not free — but the
/// +35% is unattributed across classes whose byte costs differ by 25x.
/// `--no-q8-router` (all classes off) still wins if both are set.
///
/// Note the router (`mlp.gate.weight`) is small but precision-sensitive —
/// flat-routing on a quantized router shifts which experts a token sees — so
/// prefer keeping `router` in the set unless you are explicitly testing it.
fn is_q8_tensor(name: &str) -> bool {
    let Some(class) = q8_class_of(name) else {
        return false;
    };
    // A class named in HIPFIRE_FIXED_TIER is held above --format even if it is
    // not in HIPFIRE_Q8_CLASSES — it just lands on the named dtype instead of Q8.
    if fixed_tier_dtype_for(name).is_some() {
        return true;
    }
    match std::env::var("HIPFIRE_Q8_CLASSES") {
        Ok(list) => list.split(',').any(|c| c.trim() == class),
        Err(_) => true,
    }
}

/// `HIPFIRE_FIXED_TIER=lm_head:mfp4e8soa,attn:mq4` — per-class dtype for the
/// fixed tier. Returns the dtype token for `name`'s class, or `None` to fall
/// back to Q8F16 (the historic behaviour).
///
/// Accepted dtypes: `q8`, `mq4`, `mq3l`, `mfp4e8`, `mfp4e8soa`. Accepted
/// classes: `lm_head`, `embed`, `router`, `attn` (see `q8_class_of`).
fn fixed_tier_dtype_for(name: &str) -> Option<&'static str> {
    let class = q8_class_of(name)?;
    let spec = std::env::var("HIPFIRE_FIXED_TIER").ok()?;
    for entry in spec.split(',') {
        let (c, d) = entry.split_once(':')?;
        if c.trim() == class {
            return match d.trim() {
                "mfp4e8soa" => Some("mfp4e8soa"),
                "mfp4e8" => Some("mfp4e8"),
                "mq4" => Some("mq4"),
                "mq3l" => Some("mq3l"),
                "q8" => None, // explicit q8 == default
                other => {
                    eprintln!(
                        "error: HIPFIRE_FIXED_TIER: unknown dtype '{other}' \
                         (expected q8|mq4|mq3l|mfp4e8|mfp4e8soa)"
                    );
                    std::process::exit(2);
                }
            };
        }
    }
    None
}

/// Qwen3.5 DeltaNet conv1d weight: `{prefix}.linear_attn.conv1d.weight`,
/// shape [conv_channels, 1, 4]. Small (~32K elem) and runs every token —
/// Q8 is the safe default; lossy 4-bit FWHT formats (mq4/mq3) measurably
/// hurt the gated-delta path.
fn is_conv1d_tensor(name: &str) -> bool {
    name.ends_with("conv1d.weight")
}

// ─── Main ────────────────────────────────────────────────────────────────────

/// Resolve a model input to a local directory path.
/// Accepts: local path, HuggingFace model ID (org/name), or HF cache path.
/// If the input looks like a HF model ID and isn't a local path, tries to find it
/// in the HF cache or downloads it via huggingface-cli.
fn resolve_model_path(input: &str) -> String {
    let path = Path::new(input);

    // If it's already a valid local directory with config.json, use it directly
    if path.join("config.json").exists() {
        return input.to_string();
    }

    // Check if it looks like a HuggingFace model ID (contains exactly one /)
    if input.contains('/') && !input.contains(std::path::MAIN_SEPARATOR)
        || (cfg!(unix) && input.matches('/').count() == 1)
    {
        let parts: Vec<&str> = input.splitn(2, '/').collect();
        if parts.len() == 2 {
            let org = parts[0];
            let name = parts[1];

            // Check HF cache: ~/.cache/huggingface/hub/models--{org}--{name}/snapshots/*/
            let home = std::env::var("HOME").unwrap_or_default();
            let cache_dir = format!("{home}/.cache/huggingface/hub/models--{org}--{name}");
            let snapshots_dir = Path::new(&cache_dir).join("snapshots");

            if snapshots_dir.exists() {
                // Find the first snapshot directory
                if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                    for entry in entries.flatten() {
                        let snap_path = entry.path();
                        if snap_path.is_dir() && snap_path.join("config.json").exists() {
                            eprintln!("Resolved {input} -> {}", snap_path.display());
                            return snap_path.to_string_lossy().to_string();
                        }
                    }
                }
            }

            // Not in cache — try to download
            eprintln!("Model {input} not found locally. Downloading via huggingface-cli...");
            let status = std::process::Command::new("huggingface-cli")
                .args(["download", input])
                .status();

            match status {
                Ok(s) if s.success() => {
                    // Retry cache lookup after download
                    if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                        for entry in entries.flatten() {
                            let snap_path = entry.path();
                            if snap_path.is_dir() && snap_path.join("config.json").exists() {
                                eprintln!("Downloaded {input} -> {}", snap_path.display());
                                return snap_path.to_string_lossy().to_string();
                            }
                        }
                    }
                }
                Ok(s) => eprintln!("huggingface-cli download failed with status {s}"),
                Err(e) => eprintln!(
                    "Failed to run huggingface-cli: {e}. Install with: pip install huggingface_hub"
                ),
            }
        }
    }

    // Fall through: return as-is, will fail at config.json read with a helpful error
    input.to_string()
}

// ─── GGUF input pipeline ────────────────────────────────────────────────────

/// True if the path points to a `.gguf` file on disk.
fn is_gguf_input(p: &Path) -> bool {
    p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("gguf")
}

/// Translate llama.cpp GGUF tensor names to the HuggingFace safetensors
/// names that `hipfire_runtime::hfq::load_weights_hfq` expects. The mapping is
/// the canonical llama.cpp ↔ HF convention.
///
/// Returns None for tensors that don't have a known safetensors equivalent
/// (we then keep them under their GGUF name; the future loader can decide
/// what to do, or they're skipped).
fn gguf_to_safetensors_name(gguf_name: &str) -> Option<String> {
    // Top-level tensors.
    match gguf_name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".to_string()),
        "output.weight" => return Some("lm_head.weight".to_string()),
        "output_norm.weight" => return Some("model.norm.weight".to_string()),
        _ => {}
    }
    // Per-layer: blk.{N}.<slot>.weight  →  model.layers.{N}.<slot>.weight
    if let Some(rest) = gguf_name.strip_prefix("blk.") {
        // rest = "{N}.<slot>.weight"
        let dot = rest.find('.')?;
        let layer_idx = &rest[..dot];
        let slot_full = &rest[dot + 1..]; // "<slot>.weight"
                                          // Drop the trailing ".weight" so we can rewrite slots like "attn_q"→"self_attn.q_proj".
        let slot = slot_full.strip_suffix(".weight")?;
        let translated = match slot {
            "attn_norm" => "input_layernorm".to_string(),
            "ffn_norm" => "post_attention_layernorm".to_string(),
            "attn_q" => "self_attn.q_proj".to_string(),
            "attn_k" => "self_attn.k_proj".to_string(),
            "attn_v" => "self_attn.v_proj".to_string(),
            "attn_output" => "self_attn.o_proj".to_string(),
            "attn_q_norm" => "self_attn.q_norm".to_string(),
            "attn_k_norm" => "self_attn.k_norm".to_string(),
            "ffn_gate" => "mlp.gate_proj".to_string(),
            "ffn_up" => "mlp.up_proj".to_string(),
            "ffn_down" => "mlp.down_proj".to_string(),
            other => return Some(format!("model.layers.{layer_idx}.{other}.weight")),
        };
        return Some(format!("model.layers.{layer_idx}.{translated}.weight"));
    }
    None
}

/// True if the GGUF tensor's name is a 1D norm / RMSNorm scaling vector.
/// These stay F16 in the .hfq (no benefit from quantization, precision-sensitive).
fn gguf_is_norm_tensor(name: &str) -> bool {
    name.contains("_norm") || name.contains("norm.weight")
}

/// Translate a hipfire safetensors-style tensor name to the ggml-style name
/// used by llama.cpp's imatrix output (and the rest of llama.cpp's tooling).
///
/// Verified by shape-alignment on Qwen3.5-0.8B imatrix vs safetensors load log
/// (2026-05-11):
///   - K dims match for every covered tensor class (mlp.* , self_attn.* ,
///     linear_attn.in_proj_qkv/z/a/b, linear_attn.out_proj).
///   - Layer-pattern: FullAttention layers (3, 7, 11, ...) carry standard
///     `attn_q/k/v/output`; LinearAttention layers carry `attn_qkv`/
///     `attn_gate`/`ssm_alpha`/`ssm_beta`/`ssm_out` — the SSM-naming
///     convention llama.cpp uses for Mamba-style sub-blocks.
///
/// Returns `None` for tensors that don't have an imatrix counterpart
/// (norms / biases / 1D scalars / lookup-only tables). Those fall back to
/// non-imatrix-weighted quantization in the call site.
fn safetensors_to_ggml_name(name: &str) -> Option<String> {
    // Drop the architecture-specific "language_model." prefix (Qwen3.5
    // structure has model.language_model.layers.{N}.* — the linear-attn
    // crate uses this nested layout, llama.cpp flattens to blk.{N}.*).
    let normalized = name
        .strip_prefix("model.language_model.")
        .or_else(|| name.strip_prefix("model."))
        .unwrap_or(name);

    // Top-level (currently no imatrix coverage; default is --process-output OFF).
    match normalized {
        "embed_tokens.weight" => return Some("token_embd.weight".to_string()),
        "lm_head.weight" => return Some("output.weight".to_string()),
        "norm.weight" => return Some("output_norm.weight".to_string()),
        _ => {}
    }

    // Per-layer: "layers.{N}.<slot>.weight"
    let rest = normalized.strip_prefix("layers.")?;
    let dot = rest.find('.')?;
    let layer_idx = &rest[..dot];
    let slot_full = &rest[dot + 1..];
    let slot = slot_full.strip_suffix(".weight")?;

    let translated = match slot {
        // MLP — present on every layer.
        "mlp.gate_proj" => "ffn_gate",
        "mlp.up_proj" => "ffn_up",
        "mlp.down_proj" => "ffn_down",
        // FullAttention layer tensors (standard names).
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" => "attn_output",
        // LinearAttention layer tensors (Mamba-2 / hybrid-arch SSM naming).
        "linear_attn.in_proj_qkv" => "attn_qkv",
        "linear_attn.in_proj_z" => "attn_gate",
        "linear_attn.in_proj_a" => "ssm_alpha",
        "linear_attn.in_proj_b" => "ssm_beta",
        "linear_attn.out_proj" => "ssm_out",
        // Unmapped: conv1d.weight (special-cased to HFQ4G128 at quantize
        // time; small, not multiplied by activation in the standard sense),
        // norm.weight, A_log, dt_bias (1D or scalars, no imatrix entry).
        _ => return None,
    };

    Some(format!("blk.{layer_idx}.{translated}.weight"))
}

/// Load an llama.cpp-compatible imatrix GGUF file and build a lookup
/// keyed by ggml-style tensor name. The GGUF stores per-linear-layer
/// pairs:
///   {name}.in_sum2     F32[k, n_mat]   sum of squared activations per channel
///   {name}.counts      F32[1, n_mat]   token count contributing per matrix
///
/// For non-MoE models n_mat=1; the [k] vector goes into the map directly.
/// For MoE we'd need per-expert handling — out of scope for Step 5a
/// (Qwen3.5 dense + Qwen3.6 dense are the first cohort targets; A3B MoE
/// is deferred to a future iteration that handles n_mat > 1).
///
/// Returns `HashMap<ggml_name, Vec<f32>>` with the .in_sum2 values keyed by
/// the BASE tensor name (the ".in_sum2" suffix stripped).
fn load_imatrix(path: &Path) -> HashMap<String, Vec<f32>> {
    use gguf_input::GgmlType;
    let gguf = gguf_input::GgufFile::open(path).unwrap_or_else(|e| {
        eprintln!("error: failed to open imatrix file {}: {e}", path.display());
        std::process::exit(1);
    });

    let mut map: HashMap<String, Vec<f32>> = HashMap::new();
    let mut total_entries = 0usize;
    let mut skipped_moe = 0usize;
    for t in &gguf.tensors {
        let name = match t.name.strip_suffix(".in_sum2") {
            Some(n) => n.to_string(),
            None => continue, // ignore .counts and any other entries
        };
        if t.dtype != GgmlType::F32 {
            eprintln!(
                "warning: imatrix entry {} has non-F32 dtype {:?}; skipping",
                t.name, t.dtype
            );
            continue;
        }
        // Shape is [k] (1D) for non-MoE; [k, n_mat] for MoE. Skip multi-mat
        // tensors with a warning — Step 5a doesn't handle them yet.
        let n_mat = if t.shape.len() >= 2 { t.shape[1] } else { 1 };
        if n_mat != 1 {
            skipped_moe += 1;
            continue;
        }
        let k = t.shape[0];

        // Read the F32 values from the tensor data segment.
        let data = gguf.tensor_data(t);
        let mut values = Vec::with_capacity(k);
        for i in 0..k {
            let off = i * 4;
            let v = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
            values.push(v);
        }
        map.insert(name, values);
        total_entries += 1;
    }

    eprintln!(
        "imatrix: loaded {} entries from {} ({} MoE multi-matrix entries skipped — Step 5a is dense-only)",
        total_entries,
        path.display(),
        skipped_moe,
    );
    if total_entries == 0 {
        if skipped_moe > 0 {
            // MoE-only imatrix (e.g. MiniMax routed experts): no 1D dense
            // entries for the legacy dense-AWQ table, but the file IS valid.
            // The MiniMax AWQ-on-experts path reads the raw imatrix GGUF
            // (imatrix_gguf) directly, so an empty dense table is harmless —
            // dense tensors just fall back to non-imatrix quantization.
            eprintln!(
                "imatrix: 0 dense entries, {skipped_moe} MoE multi-matrix entries — \
                 dense table empty (MoE-only imatrix; expert AWQ uses the raw GGUF)"
            );
        } else {
            eprintln!("error: imatrix file contains no usable .in_sum2 entries");
            std::process::exit(1);
        }
    }
    map
}

/// Look up imatrix per-channel weights for a given safetensors tensor name.
/// Returns `None` (caller falls back to non-imatrix-weighted quantization) if:
///   - --imatrix wasn't passed (IMATRIX not initialized), OR
///   - the tensor name doesn't have a ggml-mapping (norms, small 1D, etc.), OR
///   - the imatrix file doesn't carry this tensor (rare; usually means the
///     tensor wasn't exercised by the calibration corpus).
fn imatrix_weights_for(safetensors_name: &str) -> Option<&'static [f32]> {
    let im = IMATRIX.get()?;
    // `load_imatrix` keys the map by the imatrix FILE's tensor names (`.in_sum2`
    // stripped). hipfire's `collect_imatrix` emits *safetensors* names
    // (`model.language_model.layers.N.linear_attn.in_proj_qkv.weight`), so try the
    // direct safetensors name FIRST — this was the AWQ no-op: the map is
    // safetensors-keyed but we only tried the GGML-converted name, which always
    // missed (and 27B-3.6 hybrid linear_attn names don't round-trip anyway).
    // Fall back to the GGML name for llama.cpp-style (blk.*) imatrices.
    if let Some(v) = im.get(safetensors_name) {
        return Some(v.as_slice());
    }
    let ggml_name = safetensors_to_ggml_name(safetensors_name)?;
    im.get(&ggml_name).map(|v| v.as_slice())
}

/// Compute AWQ per-channel scales `s[j]` for one linear-layer weight tensor.
///
/// Inputs:
///   - `in_sum2`: imatrix data — Σ_token act²[j] per input channel, length K.
///     Source: hipfire's `imatrix_collect` (llama.cpp `--imatrix` output).
///   - `alpha`: AWQ tuning parameter ∈ [0, 1]. Paper-original default = 0.5.
///
/// Output:
///   - `Vec<f32>` of length K, with geometric mean normalized to ≈ 1.0.
///
/// Formula (AWQ-paper-original simplified for hipfire's data shape):
///   1. RMS_act[j] = sqrt(in_sum2[j] / N_tok). The N_tok term is a global
///      constant for the tensor and gets absorbed by the geo-mean normalization
///      below, so we can omit it from the per-channel computation.
///      Equivalent: use sqrt(in_sum2[j]) directly.
///   2. s_raw[j] = (RMS_act[j])^alpha
///   3. Normalize: s[j] = s_raw[j] / exp(mean_j log(s_raw[j]))
///      This keeps the post-AWQ-scaled weight tensor's overall magnitude
///      in the same range as the input — important for the downstream MQ4
///      min-max scale fitter not to suddenly compress/expand its dynamic
///      range based on alpha.
///
/// Edge cases:
///   - Zero in_sum2[j] (channel never exercised by calibration): clamp to
///     a tiny floor (1e-12) before sqrt to avoid log(0). Practically rare;
///     would mean a channel is unused in the calibration corpus.
///   - alpha == 0 → all s[j] = 1.0 (AWQ disabled at this layer). Caller
///     can short-circuit before invoking this function.
///
/// Cost: O(K). For 9B Qwen3.5 ~32 calls × ~4096 elements = ~131K ops total
/// across the whole quantize. Negligible.
/// Parse the layer index N from a MiniMax expert tensor name
/// `…layers.N.block_sparse_moe.experts.E.wX.weight`.
fn minimax_layer_index(name: &str) -> Option<usize> {
    let after = name.split(".layers.").nth(1)?;
    after.split('.').next()?.parse::<usize>().ok()
}

/// True if layer `l` falls in the comma-separated range list held in process config `var`
/// (e.g. "12-45,50,55-60"; inclusive ranges or bare singles). Unset/empty →
/// false. Drives per-layer mixed-precision expert promotion for MiniMax.
fn minimax_layer_in_config_set(var: &str, l: usize) -> bool {
    let spec = match hipfire_config::developer_var(var) {
        Ok(v) => v,
        Err(_) => return false,
    };
    for tok in spec.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        if let Some((a, b)) = tok.split_once('-') {
            if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                if l >= a.min(b) && l <= a.max(b) {
                    return true;
                }
            }
        } else if let Ok(n) = tok.parse::<usize>() {
            if l == n {
                return true;
            }
        }
    }
    false
}

/// Shared-per-layer AWQ scales for MiniMax routed experts from an imatrix GGUF.
/// Aggregates per-expert activation energy (in_sum2) across ALL experts of
/// layer `n` into one shared per-input-channel scale: gate(w1)/up(w3) share the
/// MoE-input channels (s_gate_up, len hidden); down(w2) uses the intermediate
/// channels (s_down, len inter). The forward applies these via experts[0], so
/// one scale per layer is exactly what the runtime consumes. None if absent.
fn minimax_layer_awq_scales(
    gguf: &gguf_input::GgufFile,
    n: usize,
    alpha: f32,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let agg = |kind: &str| -> Option<Vec<f32>> {
        let nm = format!("blk.{n}.ffn_{kind}_exps.weight.in_sum2");
        let t = gguf.tensors.iter().find(|t| t.name == nm)?;
        if t.shape.len() != 2 {
            return None;
        }
        let k = t.shape[0];
        let n_exp = t.shape[1];
        let flat: Vec<f32> = gguf
            .tensor_data(t)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if flat.len() != k * n_exp {
            return None;
        }
        let mut a = vec![0.0f32; k];
        for e in 0..n_exp {
            let off = e * k;
            for j in 0..k {
                a[j] += flat[off + j];
            }
        }
        Some(a)
    };
    let g = agg("gate")?;
    let gu: Vec<f32> = match agg("up") {
        Some(u) if u.len() == g.len() => g.iter().zip(&u).map(|(a, b)| a + b).collect(),
        _ => g.clone(),
    };
    let d = agg("down")?;
    Some((
        compute_awq_scales(&gu, alpha),
        compute_awq_scales(&d, alpha),
    ))
}

fn compute_awq_scales(in_sum2: &[f32], alpha: f32) -> Vec<f32> {
    let k = in_sum2.len();
    debug_assert!(k > 0, "empty imatrix vector");

    // Step 1+2: RMS_act^alpha, with the constant N_tok factor absorbed into
    // the geo-mean normalization. The sqrt and (·)^alpha combine into
    // (·)^(alpha/2) on the raw in_sum2 values.
    //
    // Implementation choice: compute log(s_raw) directly so we can do the
    // geo-mean normalization in log space (numerically more stable for
    // wide dynamic-range imatrix values).
    let half_alpha = (alpha as f64) * 0.5;
    let mut log_s_raw = Vec::with_capacity(k);
    let mut sum_log: f64 = 0.0;
    for &v in in_sum2 {
        // Floor dead channels to 1e-12 (NaN also maps here: f64::max returns the
        // non-NaN arg) AND cap non-finite / pathologically-large values to a
        // finite ceiling. An inf in_sum2 — f32 overflow during imatrix
        // collection, which the 27B tier1 imatrix actually contains — would
        // otherwise make this tensor's `mean_log = inf`, and then `l - mean_log`
        // = inf - inf = NaN for the inf channel. That NaN survives the output
        // clamp below (f32::clamp propagates NaN), poisoning the F16 sidecar and
        // NaN'ing the whole forward (37747 such values measured pre-fix).
        // Capping the input keeps mean_log finite; the output clamp then bounds
        // the final scale. 1e30 is well inside f64 range (ln ≈ 69).
        let v_clamped = (v as f64).max(1e-12).min(1e30);
        let log_s = half_alpha * v_clamped.ln(); // log(v^(alpha/2)) = (alpha/2) * log(v)
        log_s_raw.push(log_s);
        sum_log += log_s;
    }
    let mean_log = sum_log / (k as f64);

    // Step 3: subtract mean in log space, then exp back. After this,
    // geo_mean(s) = exp(0) = 1.0 exactly (within floating-point precision).
    //
    // Step 4 (CRITICAL — f16 safety): clamp to an f16-representable,
    // non-exploding range. The geo-mean is 1.0 by construction, so the bulk
    // of channels sit near 1; only pathological outliers reach the rails —
    // dead channels floored to 1e-12, or hot channels with huge activation
    // sums. Without this, exp() overflows to f32 inf and/or the F16 sidecar
    // under/overflows, and the inference-time `x / awq_scale` divide produces
    // inf → NaN. (Verified via dump_awq_scales on the 27B tier1 imatrix:
    // 49293 scales underflowed to 0.0 and 37747 stored as inf/NaN pre-clamp,
    // which NaN'd the whole forward — KLD 0.0 / PPL NaN on gfx11.)
    //
    // The SAME clamped vector is used for both the weight pre-scale (W*s) and
    // the emitted sidecar (x/s at inference), so the cancellation stays exact;
    // clamping only limits how aggressively pathological channels redistribute
    // quant difficulty. Real AWQ scales live in ~[0.2, 5]; [1e-2, 1e2] keeps
    // all genuine signal while removing the representability blow-ups.
    const AWQ_SCALE_MIN: f32 = 1e-2;
    const AWQ_SCALE_MAX: f32 = 1e2;
    log_s_raw
        .into_iter()
        .map(|l| ((l - mean_log).exp() as f32).clamp(AWQ_SCALE_MIN, AWQ_SCALE_MAX))
        .collect()
}

/// Apply AWQ pre-scaling to a row-major [m, k] weight tensor in place:
/// `W'[i,j] = W[i,j] * s[j]` for every (i, j).
///
/// AWQ scales are per-INPUT-channel (length K). The same s[j] vector
/// broadcasts across every output row i.
///
/// Done in-place to avoid allocating a second [m, k] buffer. The caller
/// owns the W slice and is responsible for ensuring this pre-scaling
/// happens BEFORE any subsequent transformation (e.g. FWHT rotation).
fn awq_pre_scale_weights(weights: &mut [f32], m: usize, k: usize, scales: &[f32]) {
    debug_assert_eq!(weights.len(), m * k, "weight buffer size mismatch");
    debug_assert_eq!(scales.len(), k, "AWQ scale vector must have length K");
    for r in 0..m {
        let row = &mut weights[r * k..(r + 1) * k];
        for j in 0..k {
            row[j] *= scales[j];
        }
    }
}

/// Helper: convert a `Vec<f32>` AWQ-scale vector into the F16 byte
/// payload that `HfqTensor` consumes for sidecar emission.
fn awq_scales_to_f16_bytes(scales: &[f32]) -> Vec<u8> {
    scales
        .iter()
        .flat_map(|&s| f32_to_f16(s).to_le_bytes())
        .collect()
}

/// AWQ pre-scaling is mathematically valid only for weights whose runtime
/// path applies the inverse divide-by-scale. As of F2 (2026-05-14), this
/// covers both the input-side projections (fed via the AWQ-aware variants
/// of `fused_rmsnorm_rotate_mq` from F1) AND the output-side projections
/// (`o_proj` / `out_proj` / `down_proj` / `w_down`, fed via the AWQ-aware
/// variants `rotate_x_mq_awq` and `fused_silu_mul_mq_rotate_awq` from F2).
///
/// Runtime path mapping for AWQ inverse divide-by-scale:
/// - `fused_rmsnorm_mq_rotate_awq`: post-RMSNorm input projections
///   (q/k/v/qkv, gate/up, in_proj_*, router, gate_up_proj)
/// - `rotate_x_mq_awq`: post-attention input to o_proj / out_proj
/// - `fused_silu_mul_mq_rotate_awq`: post-SwiGLU input to down_proj
///
/// Pre-F2 history: until 2026-05-14, output-side projections (o_proj /
/// out_proj / down_proj / w_down) were NOT on this whitelist because
/// their runtime path lacked AWQ-aware kernels. Pre-scaling them without
/// a runtime compensating divide produces `(W·s) · x ≠ W · x` — measured
/// 0.8B Qwen3.5 KLD blowup 0.6721 → 13.4893; see `awq_fix_claude.md`.
/// F2 added those kernels (`rotate_x_mq_awq` / `fused_silu_mul_mq_rotate_awq`)
/// plus `_for` helper routing in hipfire-runtime/llama.rs, so the whitelist
/// is now safe to expand.
///
/// Whitelist (vs blacklist) is still the safe default: a new tensor name
/// in a future arch fails closed (no AWQ) until someone confirms its
/// runtime path is AWQ-aware.
fn awq_eligible(name: &str) -> bool {
    // F1-vs-F2 A/B gate. When `HIPFIRE_AWQ_F1_ONLY=1` is set, the F2
    // additions below (o_proj / wo / out_proj / down_proj / w_down)
    // are excluded — produces an F1-equivalent quant for comparison
    // bench against the same binary's F2 quant. Default (env unset):
    // the full F2 whitelist applies.
    let f1_only = hipfire_config::developer_var("HIPFIRE_AWQ_F1_ONLY")
        .ok()
        .as_deref()
        == Some("1");
    let f1_match =
    // Full-attention input projections (HF naming + fused variants).
    name.ends_with("q_proj.weight")
        || name.ends_with("k_proj.weight")
        || name.ends_with("v_proj.weight")
        || name.ends_with("qkv_proj.weight")
        || name.ends_with("wqkv.weight")
        // MLP input projections (HF + hipfire-internal naming).
        || name.ends_with("gate_proj.weight")
        || name.ends_with("up_proj.weight")
        || name.ends_with("w_gate.weight")
        || name.ends_with("w_up.weight")
        // MoE fused expert gate+up projection (Qwen3-MoE convention —
        // experts.gate_up_proj is [num_experts, 2*intermediate, hidden]
        // with rows split between gate and up halves). Same input-side
        // semantics as gate_proj/up_proj: post-RMSNorm hidden state
        // routed via the MoE dispatch.
        || name.ends_with("gate_up_proj.weight")
        // Linear-attention input projections (Qwen3.5 Gated-DeltaNet).
        // Suffix varies (in_proj_qkv / _z / _a / _b); the substring is
        // anchored enough that no non-linear-attn tensor name should match.
        || name.contains(".in_proj_")
        // MoE router (HF naming for Qwen3-MoE / DeepSeek family — single
        // linear projecting post-RMSNorm hidden state to num_experts
        // logits). The quantizer's q8_router rule (set when is_moe)
        // promotes this to Q8 before reaching the MQ4G256 branch, so
        // this match is effectively dead code today. Kept for intent:
        // if Q8 auto-promotion is ever disabled, this preserves
        // correctness. `router.weight` would be a non-HF naming an
        // arch might choose; kept for safety.
        || name.ends_with("mlp.gate.weight")
        // MiniMax-M2 MoE router (block_sparse_moe.gate.weight). Same intent
        // as mlp.gate.weight: q8_router (set for is_minimax via is_moe_like)
        // keeps the router at Q8 so HFQ4 noise can't flip top-k selection.
        || name.ends_with("block_sparse_moe.gate.weight")
        || name.ends_with("router.weight");
    if f1_only {
        return f1_match;
    }
    let f2_match =
        // ── F2 (2026-05-14): output-side projections ────────────────────
        // These now have AWQ-aware runtime kernels (rotate_x_mq_awq for
        // o_proj/out_proj/wo; fused_silu_mul_mq_rotate_awq for down_proj/w_down).
        // Runtime dispatch routes through _for helpers in llama.rs based on
        // WeightTensor.awq_scale.
        //
        // FullAttention output projection (HF + hipfire-internal naming).
        name.ends_with("o_proj.weight")
        || name.ends_with("wo.weight")
        // LinearAttention output projection (Qwen3.5 Gated-DeltaNet).
        || name.ends_with("out_proj.weight")
        // MLP down projection (HF + hipfire-internal naming).
        || name.ends_with("down_proj.weight")
        || name.ends_with("w_down.weight");
    f1_match || f2_match
}

/// True if the tensor is the token embedding. We Q8 these (matches the
/// safetensors path's `is_embed` rule — Q4 is too lossy for embedding tables).
fn gguf_is_embed_tensor(name: &str) -> bool {
    name == "token_embd.weight"
}

/// Build the `config` JSON object that `hipfire_runtime::hfq::config_from_hfq`
/// reads. Mirrors the field names HuggingFace uses in `config.json` for
/// LlamaForCausalLM / Qwen3ForCausalLM, populated from the GGUF
/// `<arch>.*` metadata keys.
fn config_json_from_gguf(gguf: &gguf_input::GgufFile, arch_str: &str) -> serde_json::Value {
    // GGUF prefixes its model hyperparameters with the architecture name —
    // e.g. for `general.architecture=llama` the keys live under `llama.*`.
    let prefix = arch_str;

    let read_u = |k: &str| -> Option<u64> {
        gguf.metadata.get(k).and_then(|v| match v {
            gguf_input::MetaValue::U8(x) => Some(*x as u64),
            gguf_input::MetaValue::I8(x) => Some(*x as u64),
            gguf_input::MetaValue::U16(x) => Some(*x as u64),
            gguf_input::MetaValue::I16(x) => Some(*x as u64),
            gguf_input::MetaValue::U32(x) => Some(*x as u64),
            gguf_input::MetaValue::I32(x) => Some(*x as u64),
            gguf_input::MetaValue::U64(x) => Some(*x),
            gguf_input::MetaValue::I64(x) => Some(*x as u64),
            _ => None,
        })
    };
    let read_f = |k: &str| -> Option<f64> {
        gguf.metadata.get(k).and_then(|v| match v {
            gguf_input::MetaValue::F32(x) => Some(*x as f64),
            gguf_input::MetaValue::F64(x) => Some(*x),
            _ => None,
        })
    };

    let dim = read_u(&format!("{prefix}.embedding_length"));
    let n_layers = read_u(&format!("{prefix}.block_count"));
    let n_heads = read_u(&format!("{prefix}.attention.head_count"));
    let n_kv_heads = read_u(&format!("{prefix}.attention.head_count_kv")).or(n_heads);
    let hidden_dim = read_u(&format!("{prefix}.feed_forward_length"));
    // vocab_size: prefer metadata, fall back to token_embd shape[1].
    let vocab_size = read_u(&format!("{prefix}.vocab_size")).or_else(|| {
        gguf.tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .and_then(|t| t.shape.get(1).map(|&s| s as u64))
    });
    let max_seq_len = read_u(&format!("{prefix}.context_length"));
    let rope_theta = read_f(&format!("{prefix}.rope.freq_base"));
    let rms_eps = read_f(&format!("{prefix}.attention.layer_norm_rms_epsilon"));
    let head_dim = read_u(&format!("{prefix}.attention.key_length")).or_else(|| {
        // Fall back: head_dim = dim / n_heads.
        dim.zip(n_heads).map(|(d, h)| if h > 0 { d / h } else { d })
    });
    let bos = read_u("tokenizer.ggml.bos_token_id").unwrap_or(1);
    let eos = read_u("tokenizer.ggml.eos_token_id").unwrap_or(2);

    let mut cfg = serde_json::Map::new();
    cfg.insert(
        "model_type".to_string(),
        serde_json::Value::from(arch_str.to_string()),
    );
    if let Some(v) = dim {
        cfg.insert("hidden_size".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = n_layers {
        cfg.insert("num_hidden_layers".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = n_heads {
        cfg.insert(
            "num_attention_heads".to_string(),
            serde_json::Value::from(v),
        );
    }
    if let Some(v) = n_kv_heads {
        cfg.insert(
            "num_key_value_heads".to_string(),
            serde_json::Value::from(v),
        );
    }
    if let Some(v) = hidden_dim {
        cfg.insert("intermediate_size".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = vocab_size {
        cfg.insert("vocab_size".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = max_seq_len {
        cfg.insert(
            "max_position_embeddings".to_string(),
            serde_json::Value::from(v),
        );
    }
    if let Some(v) = rope_theta {
        cfg.insert("rope_theta".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = rms_eps {
        cfg.insert("rms_norm_eps".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = head_dim {
        cfg.insert("head_dim".to_string(), serde_json::Value::from(v));
    }
    cfg.insert("bos_token_id".to_string(), serde_json::Value::from(bos));
    cfg.insert("eos_token_id".to_string(), serde_json::Value::from(eos));
    serde_json::Value::Object(cfg)
}

/// Translate the GGUF metadata HashMap into a JSON object that ends up in
/// the `.hfq` header's metadata blob. A future engine-side `from_hfq` for
/// Llama-style models can read these fields the same way the existing
/// `from_gguf` reads them today.
fn gguf_meta_to_json(meta: &HashMap<String, gguf_input::MetaValue>) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (k, v) in meta {
        let json_v = mv_to_json(v);
        map.insert(k.clone(), json_v);
    }
    serde_json::Value::Object(map)
}

fn mv_to_json(v: &gguf_input::MetaValue) -> serde_json::Value {
    use gguf_input::MetaValue as MV;
    match v {
        MV::U8(x) => serde_json::Value::from(*x),
        MV::I8(x) => serde_json::Value::from(*x),
        MV::U16(x) => serde_json::Value::from(*x),
        MV::I16(x) => serde_json::Value::from(*x),
        MV::U32(x) => serde_json::Value::from(*x),
        MV::I32(x) => serde_json::Value::from(*x),
        MV::F32(x) => serde_json::Value::from(*x),
        MV::Bool(x) => serde_json::Value::from(*x),
        MV::String(s) => serde_json::Value::from(s.clone()),
        MV::U64(x) => serde_json::Value::from(*x),
        MV::I64(x) => serde_json::Value::from(*x),
        MV::F64(x) => serde_json::Value::from(*x),
        // Tokenizer arrays (tokens, scores, merges, ...) can be huge —
        // serialize them as JSON arrays so the engine side can re-parse.
        MV::Array(arr) => serde_json::Value::Array(arr.iter().map(mv_to_json).collect()),
    }
}

/// 2D-weight quantization target chosen at the per-tensor level. The choice
/// per format flag:
///
/// | --format | 2D weights      | embedding | comment                          |
/// |----------|-----------------|-----------|----------------------------------|
/// | hfq4     | HFQ4G256        | Q8F16     | dense default — no FWHT, plain   |
/// | hfq6     | HFQ6G256        | Q8F16     | dense + higher quality           |
/// | mq4      | MQ4G256         | Q8F16     | Qwen3.5+ (DeltaNet) — FWHT-rot   |
/// | mq5      | MQ5G256         | Q8F16     | 5-bit FWHT (5.25 bpw, 168 B/grp) |
/// | mq6      | MQ6G256         | Q8F16     | Qwen3.5+ (DeltaNet) + higher q   |
/// | mq3      | MQ3G256         | Q8F16     | Sub-4-bit FWHT (3.25 bpw)        |
/// | mq2      | MQ2G256         | Q8F16     | Sub-4-bit FWHT (2.25 bpw)        |
///
/// **MQ4/MQ6 for non-Qwen3.5 dense produces correct output on the Llama path
/// (the rotation cancels via `gemv_mq4g256_with_rotate`) but adds per-layer
/// `rotate_x_mq` overhead with no quality benefit — those rotations were
/// calibrated for Qwen3.5+ training.** Default is HFQ4 for dense GGUFs;
/// pass `--format mq4` only when the source is a Qwen3.5+ family model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GgufFormat {
    Hfq4,
    Hfq6,
    Mq4,
    Mq5,
    Mq6,
    Mq3,
    Mq2,
    Mq2Lloyd,
    Mq3Lloyd,
    Mq4Lloyd,
    Hfp4,      // HFP4G32 — RDNA-optimal FP4 (E2M1 + UE8M0 g32 + FP16 row scale)
    Mfp4,      // MFP4G32 — HFP4G32 + offline FWHT rotation (drop-in MQ4 replacement)
    Mfp4Lloyd, // mfp4 + per-tensor 16-entry Lloyd codebook
    Mfp4P,     // mfp4+P — mfp4 with E4M3 (non-power-of-2) per-block scale
    Mfp4E8, // mfp4-E8 — mfp4+P container with E8-lattice vector quantization (4 codewords/32 weights)
    Mfp4E8Soa, // mfp4-E8 SoA — same E8 data in structure-of-arrays layout for coalesced GEMV
    Mfp3E8, // mfp3-E8 — mfp4-E8 frame with 3-bit lattice (13 B/blk, 3.25 bpw; drop-in for MQ3-Lloyd cold)
    Mfp2E8, // mfp2-E8 — mfp4-E8 frame with 2-bit lattice (9 B/blk, 2.25 bpw; drop-in for MQ2-Lloyd cold)
}

impl GgufFormat {
    fn from_flag(flag: &str) -> Option<Self> {
        match flag {
            "hfq4" | "hfq4g256" | "hf4" => Some(Self::Hfq4),
            "hfq6" | "hfq6g256" | "hf6" => Some(Self::Hfq6),
            "mq4" | "mq4g256" | "magnum" => Some(Self::Mq4),
            "mq5" | "mq5g256" => Some(Self::Mq5),
            "mq6" | "mq6g256" => Some(Self::Mq6),
            "mq3" | "mq3g256" => Some(Self::Mq3),
            "mq2" | "mq2g256" => Some(Self::Mq2),
            "mq2-lloyd" | "mq2g256-lloyd" | "mq2lloyd" => Some(Self::Mq2Lloyd),
            "mq3-lloyd" | "mq3g256-lloyd" | "mq3lloyd" => Some(Self::Mq3Lloyd),
            "mq4-lloyd" | "mq4g256-lloyd" | "mq4lloyd" => Some(Self::Mq4Lloyd),
            "hfp4" | "hfp4g32" | "hf4p" | "fp4" => Some(Self::Hfp4),
            "mfp4" | "mfp4g32" | "mf4p" => Some(Self::Mfp4),
            "mfp4l" | "mfp4-lloyd" | "mfp4g32-lloyd" | "mfp4lloyd" => Some(Self::Mfp4Lloyd),
            "mfp4p" | "mfp4+p" | "mfp4-p" => Some(Self::Mfp4P),
            "mfp4e8" | "mfp4-e8" | "mfp4l8" => Some(Self::Mfp4E8),
            "mfp4e8soa" | "mfp4-e8-soa" | "mfp4e8-soa" => Some(Self::Mfp4E8Soa),
            "mfp3e8" | "mfp3-e8" => Some(Self::Mfp3E8),
            "mfp2e8" | "mfp2-e8" => Some(Self::Mfp2E8),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Hfq4 => "HFQ4G256",
            Self::Hfq6 => "HFQ6G256",
            Self::Mq4 => "MQ4G256",
            Self::Mq5 => "MQ5G256",
            Self::Mq6 => "MQ6G256",
            Self::Mq3 => "MQ3G256",
            Self::Mq2 => "MQ2G256",
            Self::Mq2Lloyd => "MQ2G256Lloyd",
            Self::Mq3Lloyd => "MQ3G256Lloyd",
            Self::Mq4Lloyd => "MQ4G256Lloyd",
            Self::Hfp4 => "HFP4G32",
            Self::Mfp4 => "MFP4G32",
            Self::Mfp4Lloyd => "MFP4G32Lloyd",
            Self::Mfp4P => "MFP4G32P",
            Self::Mfp4E8 => "MFP4G32E8",
            Self::Mfp4E8Soa => "MFP4G32E8SOA",
            Self::Mfp3E8 => "MFP3G32E8",
            Self::Mfp2E8 => "MFP2G32E8",
        }
    }
}

/// Convert a GGUF file to a hipfire `.hfq`. Per-format quantization target
/// applies to 2D weight matrices; the embedding table is always Q8F16
/// (Q4-grade is too lossy for embeddings) and 1D norms stay F16. Tensor
/// names are translated GGUF → safetensors style so the engine's existing
/// `load_weights_hfq` can consume the output.
fn run_gguf_pipeline(
    input: &Path,
    output: &Path,
    format: GgufFormat,
    no_kmap: bool,
    kmap_dense: bool,
    kmap_mode: u8,
    arch_id_override: Option<u32>,
    force_arch_id: bool,
) -> std::io::Result<()> {
    eprintln!("=== GGUF → {} conversion ===", format.label());
    eprintln!("Input:  {}", input.display());
    eprintln!("Output: {}", output.display());

    let gguf = gguf_input::GgufFile::open(input)?;
    eprintln!("GGUF version: {}", gguf.version);
    eprintln!("Tensors: {}", gguf.tensors.len());

    let arch_str = gguf
        .meta_str("general.architecture")
        .unwrap_or("llama")
        .to_string();
    let auto_arch_id: u32 = match arch_str.as_str() {
        "llama" => 0,
        "qwen3" | "qwen2" => 1,
        "qwen3_5" | "qwen3_5_text" | "qwen35" => 5,
        "qwen3moe" => 6,
        other => {
            // Structural-pillar guard: a qwen3* GGUF that doesn't match an
            // explicit arm above must NOT be silently stamped arch_id=0
            // (llama). That would route it off the qwen35 crate AND off the
            // froggeric chat-template pillar at serve time. Loud-fail so the
            // operator stamps it explicitly with --arch-id 5 or 6.
            if other.to_lowercase().contains("qwen3") {
                eprintln!(
                    "error: GGUF architecture '{other}' looks like a qwen3* family model but maps to no known arch_id; \
                     refusing to silently stamp arch_id=0 (would break the froggeric pillar). \
                     Re-run with an explicit --arch-id 5 (dense) or 6 (MoE)."
                );
                std::process::exit(1);
            }
            eprintln!("warning: unknown GGUF architecture '{other}', tagging as llama-compatible");
            0
        }
    };
    // --arch-id <u32> overrides the auto-detected id. Use when the
    // model's family maps to a different crate than the default
    // (e.g. plain Qwen2 → arch_id=7 for the hipfire-arch-qwen2 crate
    // instead of the LLaMA-family default 1, which silently drops
    // Q/K/V bias on the LLaMA loader path). See docs/plans/
    // dots-ocr-devlog.md §7 (R1) for the bring-up context.
    let arch_id: u32 = arch_id_override.unwrap_or(auto_arch_id);
    guard_qwen3_arch_override(auto_arch_id, arch_id, force_arch_id);
    if arch_id != auto_arch_id {
        eprintln!("Architecture: {arch_str} (auto id={auto_arch_id}, overridden via --arch-id to {arch_id})");
    } else {
        eprintln!("Architecture: {arch_str} (id={arch_id})");
    }

    // Metadata JSON: must populate `config.*` so engine's `config_from_hfq`
    // can reconstruct LlamaConfig at load time. Also keep the raw GGUF
    // metadata tree under `gguf_meta` for any consumer that wants original
    // values (chat template, vocab, scores, merges, etc.).
    let config_json = config_json_from_gguf(&gguf, &arch_str);
    let metadata = serde_json::json!({
        "architecture": arch_str,
        "source": "gguf",
        "config": config_json,
        "gguf_meta": gguf_meta_to_json(&gguf.metadata),
    });
    let metadata_json = serde_json::to_string(&metadata)?;

    // FWHT signs — only used when --format is mq4/mq6. Same seed pair as the
    // safetensors path so the engine's runtime FWHT inverse stays identical.
    let needs_signs = matches!(
        format,
        GgufFormat::Mq4
            | GgufFormat::Mq6
            | GgufFormat::Mq3
            | GgufFormat::Mq2
            | GgufFormat::Mq2Lloyd
            | GgufFormat::Mq3Lloyd
            | GgufFormat::Mq4Lloyd
            | GgufFormat::Mfp4
            | GgufFormat::Mfp4P
            | GgufFormat::Mfp4E8
            | GgufFormat::Mfp3E8
            | GgufFormat::Mfp2E8
    );
    let signs1 = if needs_signs {
        gen_fwht_signs(42, 256)
    } else {
        Vec::new()
    };
    let signs2 = if needs_signs {
        gen_fwht_signs(1042, 256)
    } else {
        Vec::new()
    };

    // K-map setup for GGUF path
    let is_moe = arch_id == 6;
    let n_layers: usize = config_json
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    // Build K-map using translated (safetensors-style) names where available,
    // falling back to raw GGUF names for untranslated tensors.
    //
    // K-map is gated to MoE models only. On dense models the author's own
    // bench shows a mixed picture (PPL +1.5% to +2.5% at 2K context on 4B
    // and 27B; PPL -4.8% on 27B at 8K context — crossover at ~3K). The
    // ship-default is the conservative shape per maintainer directive
    // (2026-05-08): never silently change dense quantization. Users who
    // want K-map on dense pass `--kmap-dense` (see flag parsing below).
    let kmap: HashMap<String, QuantLevel> = if no_kmap || (!is_moe && !kmap_dense) {
        HashMap::new()
    } else {
        let mut map = HashMap::new();
        let mut counts = [0u32; 4];
        for info in &gguf.tensors {
            let out_name =
                gguf_to_safetensors_name(&info.name).unwrap_or_else(|| info.name.clone());
            let level = kmap_resolve_mode(&out_name, n_layers, is_moe, kmap_mode);
            match level {
                QuantLevel::F16 => counts[0] += 1,
                QuantLevel::Q8 => counts[1] += 1,
                QuantLevel::Promote6 => counts[2] += 1,
                QuantLevel::Override(_) => counts[3] += 1,
                QuantLevel::Base => counts[3] += 1,
            }
            map.insert(out_name, level);
        }
        if !map.is_empty() {
            let mode_label = match kmap_mode {
                0 => "full",
                1 => "alternating",
                2 => "typed",
                _ => "?",
            };
            eprintln!(
                "K-map plan ({} base, {n_layers} layers{}, mode={mode_label}):",
                format.label(),
                if is_moe { ", MoE" } else { "" }
            );
            eprintln!("  F16:       {:>4} tensors", counts[0]);
            eprintln!("  Q8:        {:>4} tensors", counts[1]);
            eprintln!("  Promote6:  {:>4} tensors", counts[2]);
            eprintln!("  Base:      {:>4} tensors", counts[3]);
        }
        map
    };

    let mut hfq_tensors: Vec<HfqTensor> = Vec::with_capacity(gguf.tensors.len());
    let mut total_params: u64 = 0;
    let mut quant_params: u64 = 0;
    let mut total_bytes_in: u64 = 0;
    let mut total_bytes_out: u64 = 0;

    for info in &gguf.tensors {
        let raw = gguf.tensor_data(info);
        let n_elements = info.numel();
        total_params += n_elements as u64;
        total_bytes_in += raw.len() as u64;

        let shape: Vec<u32> = info.shape.iter().map(|&s| s as u32).collect();

        // Tensor classification (uses the original GGUF name).
        let is_norm = gguf_is_norm_tensor(&info.name);
        let is_embed = gguf_is_embed_tensor(&info.name);
        let is_2d = info.shape.len() == 2;
        let k_dim = if is_2d { info.shape[0] } else { n_elements };

        // Translate to the safetensors-style name `hipfire_runtime::hfq::load_weights_hfq`
        // expects. If we don't have a translation, keep the original name —
        // the future loader can ignore unknown tensors.
        let out_name = gguf_to_safetensors_name(&info.name).unwrap_or_else(|| info.name.clone());

        let kmap_level = kmap.get(&out_name).copied().unwrap_or(QuantLevel::Base);

        let (data, quant_type, group_size, label) = if is_norm || !is_2d {
            // Norms and 1D tensors always F16 (primary gate)
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            let f16_bytes: Vec<u8> = f32_data
                .iter()
                .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                .collect();
            (f16_bytes, QuantType::F16, 0u32, "F16")
        } else if kmap_level == QuantLevel::Q8 || is_embed {
            // K-map Q8 or embedding
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            let q = quantize_q8f16(&f32_data);
            quant_params += n_elements as u64;
            (q, QuantType::Q8F16, 32u32, "Q8_F16")
        } else if kmap_level == QuantLevel::Promote6 && k_dim % 256 == 0 {
            // K-map promote to 6-bit
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            quant_params += n_elements as u64;
            match format {
                GgufFormat::Mq4
                | GgufFormat::Mq3
                | GgufFormat::Mq2
                | GgufFormat::Mq2Lloyd
                | GgufFormat::Mq3Lloyd
                | GgufFormat::Mq6 => {
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::Hfq4 | GgufFormat::Hfq6 => {
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
                GgufFormat::Hfp4 => {
                    // No HFP6 variant in v1. Promote6 for HFP4 stays at HFP4G32 (4.25 bpw).
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_hfp4g32_2d(&f32_data, m, k);
                    (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                }
                GgufFormat::Mfp4 => {
                    // No MFP6 variant. Promote6 for MFP4 stays at MFP4G32 (4.25 bpw).
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                }
                GgufFormat::Mfp4Lloyd => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_lloyd_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32Lloyd, 32u32, "MFP4G32Lloyd")
                }
                GgufFormat::Mfp4P => {
                    // No MFP6 variant. Promote6 for mfp4+P stays at MFP4G32P (4.25 bpw).
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_p_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32P, 32u32, "MFP4G32P")
                }
                GgufFormat::Mfp4E8 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_e8_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32E8, 32u32, "MFP4G32E8")
                }
                GgufFormat::Mfp4E8Soa => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_e8_soa_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32E8SOA, 32u32, "MFP4G32E8SOA")
                }
                GgufFormat::Mfp3E8 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp3g32_e8_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP3G32E8, 32u32, "MFP3G32E8")
                }
                GgufFormat::Mfp2E8 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp2g32_e8_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP2G32E8, 32u32, "MFP2G32E8")
                }
                // Sub-6-bit promote targets: available for `--kmap-promote mq{2,3,4}`
                // pairings (e.g. MQ2 base + MQ3 promote alternating). Same kernels
                // as the Base arm below; just dispatched via the promote target.
                GgufFormat::Mq4 => {
                    let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                }
                GgufFormat::Mq5 => {
                    let q = quantize_mq5g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ5G256, 256u32, "MQ5G256")
                }
                GgufFormat::Mq3 => {
                    let q = quantize_mq3g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256, 256u32, "MQ3G256")
                }
                GgufFormat::Mq2 => {
                    let q = quantize_mq2g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256, 256u32, "MQ2G256")
                }
                GgufFormat::Mq2Lloyd => {
                    let q = quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                }
                GgufFormat::Mq3Lloyd => {
                    let q = quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                }
                GgufFormat::Mq4Lloyd => {
                    // Promote6 → MQ6, consistent with default_promote_target
                    // (Mq4Lloyd→Mq6) and its Lloyd siblings Mq2Lloyd/Mq3Lloyd
                    // (in the first arm). Previously this stayed at MQ4G256Lloyd
                    // (4-bit) — no actual promotion under --kmap-promote 6.
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::Hfq4 => {
                    let q = quantize_hfq4g256(&f32_data);
                    (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                }
            }
        } else if let (QuantLevel::Override(override_fmt), true) = (kmap_level, k_dim % 256 == 0) {
            // K-map says override (lm_head when --lm-head-format set).
            // GGUF pipeline has no AWQ wiring (AWQ is safetensors-only today),
            // so this is a plain quantize on the carried target format.
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            quant_params += n_elements as u64;
            match override_fmt {
                GgufFormat::Mq6 => {
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::Hfq6 => {
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
                GgufFormat::Mq4 => {
                    let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                }
                GgufFormat::Mq5 => {
                    let q = quantize_mq5g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ5G256, 256u32, "MQ5G256")
                }
                GgufFormat::Hfq4 => {
                    let q = quantize_hfq4g256(&f32_data);
                    (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                }
                GgufFormat::Mq3 => {
                    let q = quantize_mq3g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256, 256u32, "MQ3G256")
                }
                GgufFormat::Mq2 => {
                    let q = quantize_mq2g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256, 256u32, "MQ2G256")
                }
                GgufFormat::Mq2Lloyd => {
                    let q = quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                }
                GgufFormat::Mq3Lloyd => {
                    let q = quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                }
                GgufFormat::Mq4Lloyd => {
                    let q = quantize_mq4g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ4G256Lloyd, 256u32, "MQ4G256Lloyd")
                }
                GgufFormat::Hfp4 => {
                    let m = info.shape[0] as usize;
                    let q = quantize_hfp4g32_2d(&f32_data, m, k_dim);
                    (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                }
                GgufFormat::Mfp4 => {
                    let m = info.shape[0] as usize;
                    let q = quantize_mfp4g32_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                }
                GgufFormat::Mfp4Lloyd => {
                    let m = info.shape[0] as usize;
                    let q = quantize_mfp4g32_lloyd_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP4G32Lloyd, 32u32, "MFP4G32Lloyd")
                }
                GgufFormat::Mfp4P => {
                    let m = info.shape[0] as usize;
                    let q = quantize_mfp4g32_p_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP4G32P, 32u32, "MFP4G32P")
                }
                GgufFormat::Mfp4E8 => {
                    let m = info.shape[0] as usize;
                    let q = quantize_mfp4g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP4G32E8, 32u32, "MFP4G32E8")
                }
                GgufFormat::Mfp4E8Soa => {
                    let m = info.shape[0] as usize;
                    let q = quantize_mfp4g32_e8_soa_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP4G32E8SOA, 32u32, "MFP4G32E8SOA")
                }
                GgufFormat::Mfp3E8 => {
                    let m = info.shape[0] as usize;
                    let q = quantize_mfp3g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP3G32E8, 32u32, "MFP3G32E8")
                }
                GgufFormat::Mfp2E8 => {
                    let m = info.shape[0] as usize;
                    let q = quantize_mfp2g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP2G32E8, 32u32, "MFP2G32E8")
                }
            }
        } else if k_dim % 256 == 0 {
            // 256-aligned 2D weight — quantize per the chosen format (Base level).
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            quant_params += n_elements as u64;
            match format {
                GgufFormat::Hfq4 => {
                    let q = quantize_hfq4g256(&f32_data);
                    (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                }
                GgufFormat::Hfq6 => {
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
                GgufFormat::Mq4 => {
                    let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                }
                GgufFormat::Mq5 => {
                    let q = quantize_mq5g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ5G256, 256u32, "MQ5G256")
                }
                GgufFormat::Mq6 => {
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::Mq3 => {
                    let q = quantize_mq3g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256, 256u32, "MQ3G256")
                }
                GgufFormat::Mq2 => {
                    let q = quantize_mq2g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256, 256u32, "MQ2G256")
                }
                GgufFormat::Mq2Lloyd => {
                    let q = quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                }
                GgufFormat::Mq3Lloyd => {
                    let q = quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                }
                GgufFormat::Mq4Lloyd => {
                    let q = quantize_mq4g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ4G256Lloyd, 256u32, "MQ4G256Lloyd")
                }
                GgufFormat::Hfp4 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_hfp4g32_2d(&f32_data, m, k);
                    (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                }
                GgufFormat::Mfp4 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                }
                GgufFormat::Mfp4Lloyd => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_lloyd_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32Lloyd, 32u32, "MFP4G32Lloyd")
                }
                GgufFormat::Mfp4P => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_p_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32P, 32u32, "MFP4G32P")
                }
                GgufFormat::Mfp4E8 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_e8_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32E8, 32u32, "MFP4G32E8")
                }
                GgufFormat::Mfp4E8Soa => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_e8_soa_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32E8SOA, 32u32, "MFP4G32E8SOA")
                }
                GgufFormat::Mfp3E8 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp3g32_e8_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP3G32E8, 32u32, "MFP3G32E8")
                }
                GgufFormat::Mfp2E8 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp2g32_e8_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP2G32E8, 32u32, "MFP2G32E8")
                }
            }
        } else {
            // K not divisible by 256 — fall back to HFQ4-G128 (no rotation).
            // This branch fires for the rare ragged dim; ignores --format
            // (no G128 variant of mq4/mq6 exists).
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            let q = quantize_hfq4g128(&f32_data);
            quant_params += n_elements as u64;
            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
        };

        total_bytes_out += data.len() as u64;
        eprintln!(
            "  {label:>9}: {} → {} {:?} ({} src={:?}, {:.1} KB → {:.1} KB)",
            info.name,
            out_name,
            info.shape,
            n_elements,
            info.dtype,
            raw.len() as f64 / 1024.0,
            data.len() as f64 / 1024.0,
        );

        hfq_tensors.push(HfqTensor {
            name: out_name,
            quant_type,
            shape,
            group_size,
            data,
            spilled_len: 0,
        });
    }

    eprintln!("\n=== GGUF → MQ4 Summary ===");
    eprintln!("  Tensors:        {}", hfq_tensors.len());
    eprintln!("  Total params:   {total_params}");
    eprintln!(
        "  Quant'd params: {quant_params} ({:.1}%)",
        100.0 * quant_params as f64 / total_params as f64
    );
    eprintln!("  Input size:     {:.1} MB", total_bytes_in as f64 / 1e6);
    eprintln!(
        "  Output size:    {:.1} MB ({:.1}% of input)",
        total_bytes_out as f64 / 1e6,
        100.0 * total_bytes_out as f64 / total_bytes_in as f64,
    );

    write_hfq(output, arch_id, &metadata_json, &hfq_tensors, None)?;
    eprintln!("\nWrote: {}", output.display());
    Ok(())
}

fn main() {
    let args = QuantizeArgs::parse();

    // Bound rayon's pool to 80% of cores (default cap; override with --threads N
    // or HIPFIRE_QUANT_THREADS env). Quantization is CPU-bound and saturates
    // memory bandwidth, so leaving headroom for the rest of the system avoids
    // making the whole box unresponsive during a multi-hour quantize run.
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    let default_threads = ((cores * 8) / 10).max(1);
    let threads = args.threads.unwrap_or(default_threads);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global();
    eprintln!("Rayon: {threads} worker threads ({cores} cores available, default 80% = {default_threads})");

    let input_dir = args.input.as_str();
    let output_path = args.output.as_str();
    let format = args.format.as_str();

    // ── qwen3-dspark-q8: Qwen3DSparkModel drafter sidecar emission ──────────
    // Produces a `<stem>-dspark.<ext>` HFQ carrying the 5-layer dense drafter
    // body + DSpark globals (main_proj, main_norm, markov_w1/w2, confidence_proj
    // + confidence_bias) + lm_head, with DSpark metadata keys so the arch-side
    // loader can detect and configure the speculator.
    //
    // Quant recipe (small trained drafter — preserve precision):
    //   2D matmul weights (attn q/k/v/o, mlp gate/up/down) → Q8F16
    //   Everything else (norms, embed, main_proj/main_norm, markov, confidence,
    //   lm_head, bias) → F16 (or F32 for scalar bias)
    //
    // Tensor name mapping (source → sidecar):
    //   fc.weight           → main_proj.weight   (the `[hidden, 5*hidden]` concat)
    //   hidden_norm.weight  → main_norm.weight    (RMSNorm after fc)
    //   all others          → kept as-is
    if format == "qwen3-dspark-q8" || format == "qwen35-dspark-q8" {
        let input_dir = Path::new(input_dir);
        let output_path = Path::new(output_path);

        // Read config
        let config_path = input_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path).unwrap_or_else(|e| {
            eprintln!(
                "qwen3-dspark-q8: cannot read {}: {e}",
                config_path.display()
            );
            std::process::exit(1);
        });
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap_or_else(|e| {
            eprintln!("qwen3-dspark-q8: config.json parse error: {e}");
            std::process::exit(1);
        });

        // Verify architecture
        let archs = config.get("architectures").and_then(|v| v.as_array());
        let is_dspark = archs
            .map(|a| {
                a.iter().any(|v| {
                    matches!(
                        v.as_str(),
                        Some("Qwen3DSparkModel" | "DSparkDraftModel" | "DSparkSpeculator")
                    )
                })
            })
            .unwrap_or(false);
        if !is_dspark {
            eprintln!(
                "dspark-q8: architectures is not a DSpark drafter \
                 (Qwen3DSparkModel / DSparkDraftModel / DSparkSpeculator); got {:?}",
                archs
            );
            std::process::exit(1);
        }

        // Read DSpark config fields. speculators v0.6.0 (DSparkDraftModel) nests
        // the body dims under `transformer_layer_config` and names the target
        // taps `aux_hidden_state_layer_ids`; the legacy Qwen3DSparkModel puts
        // dims / `target_layer_ids` at the top level. Handle both.
        let tlc = config.get("transformer_layer_config");
        let cfg_u64 = |k: &str, d: u64| -> u64 {
            config
                .get(k)
                .or_else(|| tlc.and_then(|t| t.get(k)))
                .and_then(|v| v.as_u64())
                .unwrap_or(d)
        };
        let block_size = config
            .get("block_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(7) as usize;
        let target_layer_ids: Vec<u64> = config
            .get("target_layer_ids")
            .or_else(|| config.get("aux_hidden_state_layer_ids"))
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
            .unwrap_or_else(|| vec![1, 9, 17, 25, 33]);
        let markov_rank = config
            .get("markov_rank")
            .and_then(|v| v.as_u64())
            .unwrap_or(256) as usize;
        let noise_token_id = config
            .get("mask_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(151669) as u32;
        let draft_vocab_size = cfg_u64("draft_vocab_size", 0);
        let confidence_with_markov = config
            .get("confidence_head_with_markov")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let hidden_size = cfg_u64("hidden_size", 2048);
        let head_dim = cfg_u64("head_dim", 128);
        let num_hidden_layers = cfg_u64("num_hidden_layers", 0);
        let num_attention_heads = cfg_u64("num_attention_heads", 0);
        let num_key_value_heads = cfg_u64("num_key_value_heads", 0);
        let intermediate_size = cfg_u64("intermediate_size", 0);
        let vocab_size = cfg_u64("vocab_size", 0);
        // rope params nest under transformer_layer_config.rope_parameters in v0.6.0.
        let rope = tlc
            .and_then(|t| t.get("rope_parameters"))
            .or_else(|| config.get("rope_parameters"));
        let partial_rotary_factor = rope
            .and_then(|r| r.get("partial_rotary_factor"))
            .or_else(|| config.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let rope_theta = rope
            .and_then(|r| r.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(10000000.0);

        eprintln!(
            "qwen3-dspark-q8: block_size={block_size} target_layer_ids={target_layer_ids:?} \
             markov_rank={markov_rank} noise_token_id={noise_token_id}"
        );

        // Build metadata JSON — mirrors the keys DsparkConfig::from_metadata_json reads.
        let metadata = serde_json::json!({
            "architecture": "qwen3",
            "config": {
                "dspark_block_size": block_size,
                "dspark_target_layer_ids": target_layer_ids,
                "dspark_num_targets": target_layer_ids.len(),
                "dspark_markov_rank": markov_rank,
                "dspark_noise_token_id": noise_token_id,
                "dspark_enable_confidence": true,
                "dspark_confidence_with_markov": confidence_with_markov,
                "dspark_draft_vocab_size": draft_vocab_size,
                "dspark_hidden_size": hidden_size,
                "dspark_head_dim": head_dim,
                "dspark_num_hidden_layers": num_hidden_layers,
                "dspark_num_attention_heads": num_attention_heads,
                "dspark_num_key_value_heads": num_key_value_heads,
                "dspark_intermediate_size": intermediate_size,
                "dspark_vocab_size": vocab_size,
                "dspark_partial_rotary_factor": partial_rotary_factor,
                "dspark_rope_theta": rope_theta,
            },
        });
        let metadata_json = serde_json::to_string(&metadata).unwrap();

        // Load safetensors
        let st_paths = find_safetensors(input_dir);
        if st_paths.is_empty() {
            eprintln!(
                "qwen3-dspark-q8: no safetensors found in {}",
                input_dir.display()
            );
            std::process::exit(1);
        }
        let st_files: Vec<SafetensorsFile> = st_paths
            .iter()
            .map(|p| {
                eprintln!("Loading: {}", p.display());
                SafetensorsFile::open(p).unwrap()
            })
            .collect();

        let mut all_tensors: Vec<(&str, usize)> = Vec::new();
        for (fi, st) in st_files.iter().enumerate() {
            for name in st.tensor_names() {
                all_tensors.push((name, fi));
            }
        }
        all_tensors.sort_by_key(|(name, _)| name.to_string());
        eprintln!("qwen3-dspark-q8: {} tensors found", all_tensors.len());

        // Determine which 2D weights get Q8F16 (attn projections + MLP projections)
        let is_dspark_matmul_weight = |name: &str| -> bool {
            // Attn projections: q/k/v/o_proj
            let is_attn = name.contains("self_attn.")
                && (name.ends_with("q_proj.weight")
                    || name.ends_with("k_proj.weight")
                    || name.ends_with("v_proj.weight")
                    || name.ends_with("o_proj.weight"));
            // MLP projections: gate/up/down_proj
            let is_mlp = name.contains("mlp.")
                && (name.ends_with("gate_proj.weight")
                    || name.ends_with("up_proj.weight")
                    || name.ends_with("down_proj.weight"));
            is_attn || is_mlp
        };

        let mut hfq_tensors: Vec<HfqTensor> = Vec::new();
        let mut total_params = 0u64;
        let mut q8_params = 0u64;
        let mut f16_params = 0u64;

        for (name, file_idx) in &all_tensors {
            let (meta, raw_data) = st_files[*file_idx].tensor_data(name).unwrap();
            let n_elements: usize = meta.shape.iter().product();
            total_params += n_elements as u64;

            // Map source tensor name → sidecar name
            let sidecar_name = if *name == "fc.weight" {
                "main_proj.weight".to_string()
            } else if *name == "hidden_norm.weight" {
                "main_norm.weight".to_string()
            } else {
                name.to_string()
            };

            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();

            // Reduced-vocab maps: `d2t` (draft→target token id, I64) and `t2d`
            // (target→draft membership, BOOL). Store as F32 — token indices are
            // < 2^24 so exact; the DSpark loader casts d2t→u32, t2d→bool. The
            // float `to_f32` path can't read I64/BOOL.
            if *name == "d2t" || *name == "t2d" {
                let f32_data: Vec<f32> = if meta.dtype == "I64" {
                    raw_data
                        .chunks_exact(8)
                        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
                        .collect()
                } else if meta.dtype == "BOOL" || meta.dtype == "U8" {
                    raw_data
                        .iter()
                        .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                        .collect()
                } else {
                    to_f32(raw_data, &meta.dtype)
                };
                let bytes: Vec<u8> = f32_data.iter().flat_map(|&v| v.to_le_bytes()).collect();
                eprintln!(
                    "  {:>8}: {} {:?} ({} elems) [reduced-vocab map]",
                    "F32", sidecar_name, meta.shape, n_elements
                );
                f16_params += n_elements as u64;
                hfq_tensors.push(HfqTensor {
                    name: sidecar_name,
                    quant_type: QuantType::F32,
                    shape,
                    group_size: 0,
                    data: bytes,
                    spilled_len: 0,
                });
                continue;
            }

            if is_dspark_matmul_weight(name) && n_elements >= 32 {
                // 2D matmul weight → Q8F16 (body layers, trained precision preserved)
                let f32_data = to_f32(raw_data, &meta.dtype);
                let q = quantize_q8f16(&f32_data);
                eprintln!(
                    "  {:>8}: {} {:?} ({} elems, {:.1} KB → {:.1} KB)",
                    "Q8_F16",
                    sidecar_name,
                    meta.shape,
                    n_elements,
                    raw_data.len() as f64 / 1024.0,
                    q.len() as f64 / 1024.0
                );
                q8_params += n_elements as u64;
                hfq_tensors.push(HfqTensor {
                    name: sidecar_name,
                    quant_type: QuantType::Q8F16,
                    shape,
                    group_size: 32,
                    data: q,
                    spilled_len: 0,
                });
            } else {
                // Everything else → F16 (norms, embeds, main_proj, markov, confidence, lm_head)
                let f32_data = to_f32(raw_data, &meta.dtype);
                let f16_bytes: Vec<u8> = f32_data
                    .iter()
                    .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                    .collect();
                eprintln!(
                    "  {:>8}: {} {:?} ({} elems, {:.1} KB → {:.1} KB)",
                    "F16",
                    sidecar_name,
                    meta.shape,
                    n_elements,
                    raw_data.len() as f64 / 1024.0,
                    f16_bytes.len() as f64 / 1024.0
                );
                f16_params += n_elements as u64;
                hfq_tensors.push(HfqTensor {
                    name: sidecar_name,
                    quant_type: QuantType::F16,
                    shape,
                    group_size: 0,
                    data: f16_bytes,
                    spilled_len: 0,
                });
            }
        }

        eprintln!(
            "\n=== qwen3-dspark-q8 Summary ===\n\
             Total params:  {total_params}\n\
             Q8F16 params:  {q8_params} ({:.1}%)\n\
             F16 params:    {f16_params} ({:.1}%)\n\
             Tensors:       {}",
            100.0 * q8_params as f64 / total_params as f64,
            100.0 * f16_params as f64 / total_params as f64,
            hfq_tensors.len()
        );

        eprintln!("\nWriting: {}", output_path.display());
        write_hfq(output_path, 1u32, &metadata_json, &hfq_tensors, None).unwrap_or_else(|e| {
            eprintln!("qwen3-dspark-q8: write_hfq failed: {e}");
            std::process::exit(2);
        });

        let file_size = std::fs::metadata(output_path).unwrap().len();
        eprintln!("Done: {:.1} MB written", file_size as f64 / 1e6);
        return;
    }

    // SP4 selective re-quant overlay mode. `--reap-overlay <plan-dir>` activates
    // it: instead of quantizing the whole model, only the tensors named by the
    // plan's `quant_overrides` are decoded from the original safetensors and
    // re-quantized into a small `overlay.hfq` (written to `--reap-out`).
    // `--reap-arch` overrides the auto-detected arch family used for
    // tensor-name matching. See reap_overlay.rs / SP4 plan Task 4.
    let reap_overlay_dir = args.reap_overlay.clone();
    let reap_out = args.reap_out.clone();
    let reap_arch_flag = args.reap_arch.clone();
    // SP4b bake mode. `--reap-bake <plan-dir>` runs the NORMAL whole-model
    // quantize to completion BUT with a per-tensor override hook active: any
    // tensor the plan's `quant_overrides` name is re-quantized to its override
    // tier; every other tensor keeps its arch-specific default quant. The whole
    // model is written via the usual `write_hfq` to `--reap-out` (or the normal
    // `--format` output path). Mutually exclusive with `--reap-overlay`.
    let reap_bake_dir = args.reap_bake.clone();
    if reap_bake_dir.is_some() && reap_overlay_dir.is_some() {
        eprintln!("reap: --reap-bake and --reap-overlay are mutually exclusive");
        std::process::exit(1);
    }

    // Optional imatrix (llama.cpp GGUF format with .in_sum2 / .counts per-tensor).
    // When provided, MQ2-Lloyd quantization uses per-column importance weights
    // to bias centroid placement. See `quantize_mq2g256_lloyd_weighted`.
    let imatrix_path: Option<&Path> = args.imatrix.as_deref();
    let imatrix_gguf: Option<gguf_input::GgufFile> = imatrix_path.map(|p| {
        eprintln!("Loading imatrix: {}", p.display());
        gguf_input::GgufFile::open(p).unwrap_or_else(|e| {
            eprintln!("imatrix open failed: {e}");
            std::process::exit(2);
        })
    });
    if let Some(ref gg) = imatrix_gguf {
        let n_in_sum2 = gg
            .tensors
            .iter()
            .filter(|t| t.name.ends_with(".in_sum2"))
            .count();
        let n_counts = gg
            .tensors
            .iter()
            .filter(|t| t.name.ends_with(".counts"))
            .count();
        eprintln!(
            "  imatrix: {} in_sum2 + {} counts tensors",
            n_in_sum2, n_counts
        );
    }
    // q8f16 = all weights Q8 (interleaved blocks)
    // q4f16 = all weights Q4_F16_G64
    // q8-mixed = Q8 attn + Q4_K FFN (best tok/s for VRAM-constrained)
    // q8-fast = Q8 attn + Q4-as-Q8 FFN (all Q8 occupancy, most VRAM)
    // q8hfq = all weights Q8_HFQ (split-metadata, 128B-aligned rows)
    let use_q8 = format == "q8f16" || format == "q8";
    // F1 native-bf16 oracle: full-precision passthrough. Every tensor stored
    // as QuantType::F32 (qt=2) -- weights, norms, embeddings. The bf16 source
    // is widened bf16->f32 (lossless), giving the engine a superset-precision
    // reference forward for self-sufficient KLD eval.
    let use_f32_passthrough =
        format == "f32" || format == "f32-passthrough" || format == "bf16" || format == "oracle";
    let use_mixed = format == "q8-mixed" || format == "mixed";
    let use_fast = format == "q8-fast" || format == "fast";
    let use_q8hfq = format == "q8hfq";
    let use_q4k_all = format == "q4k";
    let use_q4k_q8embed = format == "q4k-q8embed";
    let use_mq8g256 = format == "mq8" || format == "mq8g256";
    // DeepSeek V4 recipe (2026-05-20): routed experts → MQ2-Lloyd, every other
    // 2D weight → Q8F16, with norms/biases/HC matrices falling through
    // to the F16 fallback path via `should_quantize() == false`.
    // No K-map, no imatrix promotions, no source-dtype distinctions in
    // the quant branch — uniform Q8F16 for everything that's a real
    // matmul weight. Designed to re-quant DeepSeek-V4-Flash including
    // the MTP head at maximum precision for the dense path.
    let use_deepseek4_source_precision = format == "deepseek4-q8-mtp"
        || format == "deepseek4-q8"
        || format == "deepseek4-source-precision"
        || format == "deepseek4-source"
        || format == "deepseek4-mtp-precise"
        || format == "deepseek4-mq4lloyd"
        || format == "deepseek4-mq3lloyd";
    // deepseek4-mq4lloyd / deepseek4-mq3lloyd: identical recipe to deepseek4-q8
    // (non-expert 2D → Q8F16, norms/HC → F16) EXCEPT routed experts ship as
    // MQ4G256Lloyd (qt=30, 160 B/group) resp. MQ3G256Lloyd (qt=20, 112 B/group)
    // instead of MQ2G256Lloyd. Both require the matching MoE GEMV kernels in the
    // ds4 forward (MQ3-Lloyd kernels pre-existed; MQ4-Lloyd added alongside).
    let use_deepseek4_mq4_experts = format == "deepseek4-mq4lloyd";
    let use_deepseek4_mq3_experts = format == "deepseek4-mq3lloyd";
    // deepseek4-mtp-precise: addon-only build (use with --include-prefix mtp.) that
    // keeps every mtp.0.* DENSE weight at F16 instead of Q8F16. Doubles the
    // addon size (~2 GB → ~3 GB) but eliminates Q8 quant noise on the MTP
    // attn projections, e_proj, h_proj, and shared experts. MTP is small
    // enough that the precision matters disproportionately — V3 paper's
    // 60-80% acceptance benchmark assumes weights at training precision,
    // not 8-bit. Routed experts stay MQ2-Lloyd (no precision-upgrade option
    // available without a new MoE GEMV kernel).
    let use_mtp_precise = format == "deepseek4-mtp-precise";
    let use_mq4g256 = format == "mq4" || format == "mq4g256" || format == "magnum";
    let use_hfq4g256 = format == "hfq4g256" || format == "hfq4" || format == "hf4";
    let use_hfq3g256 = format == "hfq3g256";
    let use_hfq3g128 = format == "hfq3g128" || format == "hfq3" || format == "hf3"; // default HF3 = G128
    let use_hfq2g256 = format == "hfq2g256";
    let use_hfq2g128 = format == "hfq2g128" || format == "hfq2" || format == "hf2";
    let use_hfq_mixed = format == "hfq-mixed"; // Q8 attn + HFQ4 FFN
    let use_mq6g256 = format == "mq6" || format == "mq6g256";
    let use_mq5g256 = format == "mq5" || format == "mq5g256";
    // Native-bf16 reference (the KLD/PPL oracle): cohere2moe stores matmul
    // weights as the EXACT downloaded bf16 bytes (~61 GB, fits gfx1151's GTT;
    // the all-F32 `oracle` passthrough is 122 GB and does NOT fit). `f16` is a
    // lossy-reconvert alternative tier, kept available but not the reference.
    let use_bf16 = format == "bf16" || format == "bf16-passthrough" || format == "oracle";
    let use_f16 = format == "f16" || format == "f16-passthrough";
    // ── Graded per-expert mixed precision (HIPFIRE_MOE_GRADED) ────────────
    // When set, each routed 3D-MoE expert in a layer is assigned its OWN
    // dtype: the top `HIPFIRE_MOE_HOT_FRAC` (default 0.2) experts BY IMATRIX
    // ROUTING COUNT → MQ6 (hot), the rest → MQ2-Lloyd (cold). A single
    // parent (one layer's gate_up_proj or down_proj) therefore emits MIXED
    // per-expert dtypes; the runtime builds a per-expert dtype-tag table
    // from each expert's gpu_dtype and dispatches the merged MQ6/MQ2-Lloyd
    // decode kernel. Requires --imatrix (the .counts tensor). Mutually
    // exclusive with the AWQ / Lloyd-tier expert paths (graded is the first
    // arm in the rayon dispatch). Compose with --format mq4 --no-kmap so the
    // DENSE attn/shared weights stay MQ4 and only the 3D experts are graded.
    let use_moe_graded = hipfire_config::developer_var("HIPFIRE_MOE_GRADED")
        .ok()
        .as_deref()
        == Some("1");
    let moe_hot_frac: f64 = hipfire_config::developer_var("HIPFIRE_MOE_HOT_FRAC")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map(|f| f.clamp(0.0, 1.0))
        .unwrap_or(0.2);
    if use_moe_graded && imatrix_path.is_none() {
        eprintln!(
            "error: HIPFIRE_MOE_GRADED=1 requires --imatrix <PATH> (uses per-expert .counts)"
        );
        std::process::exit(2);
    }
    if use_moe_graded {
        eprintln!(
            "note: HIPFIRE_MOE_GRADED=1 — top {:.0}% routed experts per layer (by\n\
             imatrix .counts) -> MQ6, rest -> MQ2-Lloyd. Emits MIXED per-expert\n\
             dtypes; requires the merged MQ6/MQ2-Lloyd decode kernel at runtime.",
            moe_hot_frac * 100.0
        );
    }
    // ── N-tier graded MoE (HIPFIRE_MOE_TIER_MAP) ─────────────────────────────
    // When set to a path, reads a file of "LAYER EXPERT DTYPE" lines and
    // builds a per-(layer,expert) tier assignment for BOTH gate_up and down
    // projections. Supported DTYPE values: MQ6, MQ4, MQ3L, MQ2L.
    // Placed BEFORE the graded_hot arm in the rayon dispatch (takes priority).
    // Does NOT require --imatrix; does NOT restrict to down_proj.
    // Compose with --format mq4 --no-kmap --awq (dense attn/shared keep AWQ-MQ4).
    let moe_tier_map: Option<std::collections::HashMap<(usize, usize), QuantType>> = if let Ok(
        path,
    ) =
        hipfire_config::developer_var("HIPFIRE_MOE_TIER_MAP")
    {
        let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!("error: HIPFIRE_MOE_TIER_MAP={path}: {e}");
            std::process::exit(2);
        });
        let mut map = std::collections::HashMap::new();
        for (lineno, line) in content.lines().enumerate() {
            let cols: Vec<&str> = line.split_whitespace().collect();
            if cols.len() < 3 {
                continue;
            }
            let lay: usize = cols[0].parse().unwrap_or_else(|_| {
                eprintln!("error: {path}:{}: bad layer '{}'", lineno + 1, cols[0]);
                std::process::exit(2);
            });
            let exp: usize = cols[1].parse().unwrap_or_else(|_| {
                eprintln!("error: {path}:{}: bad expert '{}'", lineno + 1, cols[1]);
                std::process::exit(2);
            });
            let qt = match cols[2] {
                "MQ6" => QuantType::MQ6G256,
                "MQ4" => QuantType::MQ4G256,
                "MQ3L" => QuantType::MQ3G256Lloyd,
                "MQ2L" => QuantType::MQ2G256Lloyd,
                // GL = global codebook (one per tensor, shipped as kernel scalar
                // args) + per-block fp16 scale, SoA. 2.0625 / 3.0625 bpw vs the
                // per-block Lloyd family's 2.25 / 3.5 -- 0.1875 bpw cheaper for a
                // measured +1.16% KLD and -0.08% decode. DECODE-ONLY: no grouped
                // or batched GL kernels exist, so a GL model prefills per-token.
                "MQ2GL" => QuantType::MQ2G256GL,
                "MQ3GL" => QuantType::MQ3G256GL,
                "E8" | "MFP4E8" | "MFP4G32E8" => QuantType::MFP4G32E8,
                "MFP3E8" | "MFP3G32E8" => QuantType::MFP3G32E8,
                "MFP2E8" | "MFP2G32E8" => QuantType::MFP2G32E8,
                other => {
                    eprintln!(
                            "error: {path}:{}: unknown dtype '{}' (expected MQ6/MQ4/MQ3L/MQ2L/MQ2GL/MQ3GL/E8/MFP3E8/MFP2E8)",
                            lineno + 1, other
                        );
                    std::process::exit(2);
                }
            };
            map.insert((lay, exp), qt);
        }
        eprintln!(
            "note: HIPFIRE_MOE_TIER_MAP={path} — {} (layer,expert) tier assignments loaded.",
            map.len()
        );
        Some(map)
    } else {
        None
    };
    // Mixed: MQ4 for attention/shared-expert + MQ6 for routed experts only.
    // Saves ~15 GB vs full MQ6 on 122B-A10B (75 GB vs 90 GB), fits in 125 GB UMA.
    let use_mq4_mq6exp = format == "mq4-mq6exp" || format == "mq4-mq6experts";
    // Round-trip quality probe: route routed-MoE experts through MQ2-Lloyd
    // quantize → dequantize → re-quantize as HFQ4. The .hfq ships as plain
    // MQ4 (HFQ4G256), no runtime changes. Measures whether 2-bit noise on
    // routed experts survives the MoE sparse-usage rescue, before sinking
    // a week into new MoE-2bit GEMV kernels.
    let use_mq4_mq2lloydexp = format == "mq4-mq2lloydexp"
        || format == "mq4-mq2lloydexperts"
        || format == "mq4-mq2lloyd-exp";
    // GL twin of the probe above: identical pipeline, but routed experts go
    // through the GLOBAL-codebook codec (one tensor-wide Lloyd–Max Gaussian
    // codebook + per-block fp16 scale) instead of a per-block fitted codebook.
    // Ships as HFQ4G256 exactly like `mq4-mq2lloydexp`, so both probes land in
    // the same container and a KLD delta between them isolates the codec —
    // no engine, loader, or kernel changes on either arm.
    let use_mq4_mq2glexp =
        format == "mq4-mq2glexp" || format == "mq4-mq2glexperts" || format == "mq4-mq2gl-exp";
    if use_mq4_mq2glexp {
        eprintln!(
            "note: --format mq4-mq2glexp is a quality probe — routed MoE experts\n\
             go through the GLOBAL-codebook 2-bit codec (one tensor-wide\n\
             Lloyd–Max Gaussian codebook + per-block fp16 scale) round-trip\n\
             (quantize → dequantize) and ship as HFQ4G256. Identical container\n\
             to --format mq4-mq2lloydexp, so a KLD delta between the two\n\
             isolates the codec. No engine/loader/kernel changes on either arm."
        );
    }
    if use_mq4_mq2lloydexp {
        eprintln!(
            "note: --format mq4-mq2lloydexp is a quality probe — routed MoE\n\
             experts go through MQ2-Lloyd round-trip (quantize → dequantize)\n\
             before being re-quantized as MQ4. Output is shipped as plain\n\
             MQ4 (no runtime changes needed). Measures whether MoE sparse\n\
             usage rescues MQ2-Lloyd at the experts before investing in new\n\
             MoE-2bit GEMV kernels."
        );
    }
    // Native Phase-2 form: routed MoE experts ship as native MQ2G256Lloyd
    // (qt=19). Requires runtime support — the qwen35 MoE forward path must
    // dispatch the new gemv_mq2g256_lloyd_moe_*_indexed* kernels (or fall
    // through to weight_gemv's MQ2G256Lloyd arm for the slow per-expert
    // path).
    let use_mq4_mq2lloyd_native = format == "mq4-mq2lloyd-native"
        || format == "mq4-mq2lloydexp-native"
        || format == "mq4-mq2lloyd-routed";
    // kmap-respecting variant: like mq4-mq2lloyd-native, but routed-expert
    // tensors that the kmap flags as Promote6 stay at MQ6 (instead of being
    // demoted to MQ2-Lloyd). Reduces precision-loss on the ~30% of layers
    // that the alternating K-map identifies as important. Larger file
    // (extra MQ6 layers) but expected to recover quality on attractor-prone
    // prompts that mq4-mq2lloyd-native truncated early.
    let use_mq4_mq2lloyd_kmap = format == "mq4-mq2lloyd-kmap"
        || format == "mq4-mq2lloyd-respectkmap"
        || format == "mq4-mq2lloyd-kmap-promote";
    // Imatrix-weighted variant: like mq4-mq2lloyd-kmap, but the Lloyd
    // codebook for each non-promoted expert is fit with per-column
    // importance weights from a llama.cpp imatrix file (--imatrix flag).
    // The kmap-promoted ~30 % of expert layers still stay at MQ6.
    let use_mq4_mq2lloyd_imatrix = format == "mq4-mq2lloyd-imatrix"
        || format == "mq4-mq2lloyd-kmap-imatrix"
        || format == "mq4-mq2lloyd-imatrix-kmap";
    // MQ3-Lloyd-on-routed-experts: 3 bpw alternative when 2 bpw isn't enough.
    // Kmap-respecting: promoted experts → MQ6, rest → MQ3-Lloyd (qt=20).
    // No imatrix variant for MQ3 in this commit — MQ3-Lloyd is empirically
    // production-grade on Qwen3.5-MoE A3B, so uniform Lloyd is the baseline.
    let use_mq4_mq3lloyd_kmap = format == "mq4-mq3lloyd-kmap"
        || format == "mq4-mq3lloyd-routed"
        || format == "mq4-mq3lloyd-exp";
    let allow_mq3_lloyd_for_mixed = args.allow_mq3_lloyd
        || hipfire_config::developer_var("HIPFIRE_ALLOW_MQ3_LLOYD")
            .ok()
            .as_deref()
            == Some("1");
    if use_mq4_mq3lloyd_kmap && !allow_mq3_lloyd_for_mixed {
        eprintln!(
            "note: --format mq4-mq3lloyd-kmap requires --allow-mq3-lloyd or\n\
             HIPFIRE_ALLOW_MQ3_LLOYD=1 (same gate as bare --format mq3-lloyd)."
        );
        std::process::exit(2);
    }
    if use_mq4_mq3lloyd_kmap {
        eprintln!(
            "note: --format mq4-mq3lloyd-kmap ships routed experts as MQ3G256Lloyd\n\
             (qt=20, 112 B / 256 weights, 3.5 bpw). Promoted experts stay at MQ6.\n\
             3 bpw fallback when 2 bpw can't avoid attractors on code-gen."
        );
    }
    // Phase 5: importance-aware MQ2/MQ3 layer tiering. Requires --imatrix.
    // Per-layer aggregate counts rank layers by routing activity; the top
    // `tier_ratio` fraction of NON-PROMOTED layers gets MQ3-Lloyd (3.5 bpw)
    // for higher precision on hot layers, the bottom fraction gets
    // MQ2-Lloyd (2.25 bpw) for size. K-map-promoted layers stay at MQ6.
    //
    // Granularity is PER LAYER (not per expert within a layer) because the
    // MoE-indexed kernels require uniform dtype across experts within a
    // tensor — the kernel reads expert_ptrs and assumes a fixed byte
    // stride per group (72 B for MQ2 vs 112 B for MQ3).
    let use_mq4_mqlloyd_tiered = format == "mq4-mqlloyd-tiered"
        || format == "mq4-mqlloyd-tiered-imatrix"
        || format == "mqlloyd-tiered";
    // Phase 6: antirez-style asymmetric-tensor recipe. Routed-expert
    // gate_up_proj → MQ2-Lloyd (imatrix-weighted), routed-expert
    // down_proj → MQ3-Lloyd (no imatrix, fixed-precision protection of
    // the residual-write direction). K-map promoted layers still get
    // MQ6 on both tensors.
    //
    // Rationale: antirez (V4 Flash) uses IQ2_XXS on up/gate and Q2_K
    // on down. The empirical claim is that `down` is the more sensitive
    // direction because it writes back into the residual stream — gate/up
    // errors get partially absorbed by silu. Mirror that asymmetry in
    // MQ-family: 2-bit on gate_up, 3-bit on down.
    let use_mq4_mqlloyd_antirez =
        format == "mq4-mqlloyd-antirez" || format == "mq4-mqlloyd-asym" || format == "antirez-mq";
    // HIPFIRE_ROUTED_GL=1: keep the per-projection bit allocation but swap the
    // per-block fp16 codebook for a GLOBAL one (qt=38/39). Post-FWHT blocks are
    // Gaussian by CLT, so the optimal LEVEL SHAPE is identical in every block and
    // a per-block fit re-derives it ~4000x per tensor, differing only by scale —
    // which the fp16 per-block scale already carries. Costs 0.1875 bpw less
    // (2.0625/3.0625 vs 2.25/3.5) for a measured +1.16% KLD and -0.08% decode.
    //
    // DECODE-ONLY: no grouped-WMMA or batched GL kernels exist, so a GL model
    // takes the per-token prefill path. Fine for eval; not yet a serving SKU.
    let routed_gl = std::env::var("HIPFIRE_ROUTED_GL").ok().as_deref() == Some("1");
    if routed_gl {
        eprintln!(
            "note: HIPFIRE_ROUTED_GL=1 — routed experts ship the GLOBAL-codebook\n\
             variants (MQ2G256GL qt=38 / MQ3G256GL qt=39) instead of the per-block\n\
             Lloyd ones. Decode-only: batched prefill rejects GL."
        );
    }
    // Lever 2: same recipe as antirez but with sequential-GPTQ Lloyd
    // on the gate_up_proj path instead of plain imatrix-weighted Lloyd.
    // Aims to reduce attractor risk at 2 bpw — if successful, opens path
    // to ALL-MQ2 routed experts (no down=MQ3 compensation needed) and
    // a further size reduction.
    let use_mq4_mqlloyd_antirez_gptq = format == "mq4-mqlloyd-antirez-gptq"
        || format == "mq4-mqlloyd-asym-gptq"
        || format == "antirez-mq-gptq";
    if use_mq4_mqlloyd_antirez_gptq && imatrix_path.is_none() {
        eprintln!("error: --format mq4-mqlloyd-antirez-gptq requires --imatrix <PATH>");
        std::process::exit(2);
    }
    if use_mq4_mqlloyd_antirez_gptq && !allow_mq3_lloyd_for_mixed {
        eprintln!(
            "note: --format mq4-mqlloyd-antirez-gptq requires --allow-mq3-lloyd or\n\
             HIPFIRE_ALLOW_MQ3_LLOYD=1 (down_proj uses MQ3-Lloyd)."
        );
        std::process::exit(2);
    }
    if use_mq4_mqlloyd_antirez_gptq {
        eprintln!(
            "note: --format mq4-mqlloyd-antirez-gptq — same routed-expert split\n\
             as antirez (gate_up=MQ2-Lloyd, down=MQ3-Lloyd), but gate_up uses\n\
             SEQUENTIAL-error-feedback Lloyd (simplified GPTQ-LDLQ) for\n\
             reduced attractor risk at 2 bpw."
        );
    }
    // All-MQ2-GPTQ: route BOTH gate_up AND down through MQ2-Lloyd-GPTQ.
    // Tests whether sequential error feedback closes the attractor gap
    // enough to drop the down=MQ3 compensation antirez uses, saving
    // ~30 % more on routed-expert size.
    let use_mq4_mq2lloyd_gptq_all = format == "mq4-mq2lloyd-gptq-all"
        || format == "mq4-mq2lloyd-gptq"
        || format == "all-mq2-gptq";
    if use_mq4_mq2lloyd_gptq_all
        && imatrix_path.is_none()
        && hipfire_config::developer_var("HIPFIRE_ALLOW_UNIT_IMATRIX")
            .ok()
            .as_deref()
            != Some("1")
    {
        eprintln!("error: --format mq4-mq2lloyd-gptq-all requires --imatrix <PATH>");
        eprintln!(
            "       (DeepSeek V4: set HIPFIRE_ALLOW_UNIT_IMATRIX=1 to use unit column weights —"
        );
        eprintln!(
            "        captures GPTQ sequential error-feedback win without imatrix calibration.)"
        );
        std::process::exit(2);
    }
    if use_mq4_mq2lloyd_gptq_all {
        eprintln!(
            "note: --format mq4-mq2lloyd-gptq-all — ALL routed experts (both\n\
             gate_up AND down) at MQ2-Lloyd with sequential-GPTQ codebook\n\
             assignment. Tests the size-reduction hypothesis from Lever 2."
        );
    }
    if use_mq4_mqlloyd_antirez {
        if imatrix_path.is_none() {
            eprintln!("error: --format mq4-mqlloyd-antirez requires --imatrix <PATH>");
            std::process::exit(2);
        }
        if !allow_mq3_lloyd_for_mixed {
            eprintln!(
                "note: --format mq4-mqlloyd-antirez requires --allow-mq3-lloyd or\n\
                 HIPFIRE_ALLOW_MQ3_LLOYD=1 (down_proj uses MQ3-Lloyd)."
            );
            std::process::exit(2);
        }
        eprintln!(
            "note: --format mq4-mqlloyd-antirez ships routed experts as\n\
             gate_up_proj → MQ2-Lloyd (imatrix-weighted, qt=19), down_proj\n\
             → MQ3-Lloyd (qt=20). K-map-promoted layers stay at MQ6 on both.\n\
             Mirrors antirez/ds4 V4 Flash recipe (IQ2_XXS gate/up, Q2_K down).\n\
             Estimated DeepSeek V4 size: 70% × MQ2 + 20% × MQ3 + 10% × MQ4 ≈ 96 GB."
        );
    }
    let tier_ratio = args.tier_ratio;
    if use_mq4_mqlloyd_tiered {
        if imatrix_path.is_none() {
            eprintln!("error: --format mq4-mqlloyd-tiered requires --imatrix <PATH>");
            std::process::exit(2);
        }
        if !allow_mq3_lloyd_for_mixed {
            eprintln!(
                "note: --format mq4-mqlloyd-tiered requires --allow-mq3-lloyd or\n\
                 HIPFIRE_ALLOW_MQ3_LLOYD=1 (uses MQ3-Lloyd on the hot layers)."
            );
            std::process::exit(2);
        }
        eprintln!(
            "note: --format mq4-mqlloyd-tiered uses imatrix .counts to rank\n\
             routed-expert layers by aggregate activation. Top {:.0}% of\n\
             non-promoted layers go to MQ3-Lloyd (3.5 bpw); the rest go to\n\
             MQ2-Lloyd (2.25 bpw). K-map-promoted layers stay at MQ6.",
            tier_ratio * 100.0
        );
    }
    if use_mq4_mq2lloyd_imatrix {
        if imatrix_path.is_none() {
            eprintln!("error: --format mq4-mq2lloyd-imatrix requires --imatrix <PATH>");
            std::process::exit(2);
        }
        eprintln!(
            "note: --format mq4-mq2lloyd-imatrix uses per-column importance\n\
             weights from the supplied calibration imatrix. Promoted experts\n\
             still stay at MQ6 (kmap-respect). Falls back to uniform Lloyd\n\
             for any expert whose imatrix tensor is missing."
        );
    }
    if use_mq4_mq2lloyd_kmap {
        eprintln!(
            "note: --format mq4-mq2lloyd-kmap respects K-map promotion —\n\
             experts flagged Promote6 (~30 % of layers) stay at MQ6G256;\n\
             remaining ~70 % get MQ2G256Lloyd (qt=19). File size is larger\n\
             than mq4-mq2lloyd-native but quality on attractor-prone prompts\n\
             should be markedly better."
        );
    }
    if use_mq4_mq2lloyd_native {
        eprintln!(
            "note: --format mq4-mq2lloyd-native ships routed MoE experts as\n\
             native MQ2G256Lloyd (qt=19, 72 B/group). Runtime must support\n\
             the MQ2-Lloyd MoE dispatch (weight_gemv arm exists; indexed\n\
             fast path requires forward-path arms in hipfire-arch-qwen35)."
        );
    }
    if use_mq4_mq6exp {
        eprintln!(
            "warning: --format mq4-mq6exp is deprecated. Use --format mq4 instead — \
             K-map promotes expert FFNs (and edge layers) to MQ6 automatically. \
             Proceeding as --format mq4."
        );
    }
    let use_mq3g256 = format == "mq3" || format == "mq3g256";
    let use_mq2g256 = format == "mq2" || format == "mq2g256";
    let use_mq2g256_lloyd =
        format == "mq2-lloyd" || format == "mq2g256-lloyd" || format == "mq2lloyd";
    let use_mq3g256_lloyd =
        format == "mq3-lloyd" || format == "mq3g256-lloyd" || format == "mq3lloyd";
    let use_mq4g256_lloyd =
        format == "mq4-lloyd" || format == "mq4g256-lloyd" || format == "mq4lloyd";
    let use_hfq6 = format == "hfq6" || format == "hfq6g256" || format == "hf6";
    // HFP4G32 — RDNA-optimal FP4 (E2M1 + UE8M0 g32 + FP16 row scale). Spec at docs/quant-formats/hfp4.md.
    let use_hfp4 = format == "hfp4" || format == "hfp4g32" || format == "hf4p" || format == "fp4";
    // MFP4G32 — HFP4G32 + offline FWHT (drop-in MQ4 replacement). Same per-row layout
    // as HFP4G32 with format_flags bit 0 + bits 2-3 = 01 stamping the rotation kind.
    let use_mfp4 = format == "mfp4" || format == "mfp4g32" || format == "mf4p";
    let use_mfp4l = format == "mfp4l"
        || format == "mfp4-lloyd"
        || format == "mfp4g32-lloyd"
        || format == "mfp4lloyd";
    // mfp4+P — mfp4 with E4M3 (non-power-of-2) per-block scale. Byte layout
    // identical to mfp4 (no prefix); only the per-block scale byte's meaning differs.
    let use_mfp4p = format == "mfp4p" || format == "mfp4+p" || format == "mfp4-p";
    // mfp4-E8 — mfp4+P container with E8-lattice vector quantization.
    // The `-gptq` suffix activates Hessian-aware sequential rounding (LDLQ on
    // the E8 lattice) — output bytes are IDENTICAL format (same E4M3 scale + 4
    // E8 codewords); GPTQ only changes the lattice-point assignment.
    let use_gptq_e8 = format == "mfp4e8-gptq" || format == "mfp4-e8-gptq";
    let use_mfp4e8 = format == "mfp4e8" || format == "mfp4-e8" || format == "mfp4l8" || use_gptq_e8;
    let use_mfp4e8soa = format == "mfp4e8soa" || format == "mfp4-e8-soa" || format == "mfp4e8-soa";
    // mfp3-E8 and mfp2-E8: 3-bit and 2-bit narrowed E8 lattice variants.
    // The `-gptq` suffix activates LDLQ — output bytes are IDENTICAL format to
    // the corresponding RTN paths; GPTQ only changes the lattice-point assignment.
    let use_gptq_mfp3e8 = format == "mfp3e8-gptq" || format == "mfp3-e8-gptq";
    let use_mfp3e8_gptq_fmt = format == "mfp3e8" || format == "mfp3-e8" || use_gptq_mfp3e8;
    let use_gptq_mfp2e8 = format == "mfp2e8-gptq" || format == "mfp2-e8-gptq";
    let use_mfp2e8_gptq_fmt = format == "mfp2e8" || format == "mfp2-e8" || use_gptq_mfp2e8;
    // GPTQ-E8 Hessian directory: per-(tensor,expert) 256-block XX^T captured by
    // the collect_e8_hessian binary. Missing/degenerate Hessians silently fall
    // back to RTN per-block (never worse than baseline). REQUIRED when --format
    // mfp{2,3,4}e8-gptq is set.
    let hessian_dir = args.hessian_dir.clone();
    if use_gptq_e8 && hessian_dir.is_none() {
        eprintln!("warning: --format mfp4e8-gptq without --hessian-dir; every tensor falls back to RTN E8 (== plain mfp4e8). Pass --hessian-dir <dir> to enable GPTQ.");
    }
    if use_gptq_mfp3e8 && hessian_dir.is_none() {
        eprintln!("warning: --format mfp3e8-gptq without --hessian-dir; every tensor falls back to RTN mfp3-E8. Pass --hessian-dir <dir> to enable GPTQ.");
    }
    if use_gptq_mfp2e8 && hessian_dir.is_none() {
        eprintln!("warning: --format mfp2e8-gptq without --hessian-dir; every tensor falls back to RTN mfp2-E8. Pass --hessian-dir <dir> to enable GPTQ.");
    }
    if let Some(hd) = &hessian_dir {
        if !hd.exists() {
            eprintln!("error: --hessian-dir not found: {}", hd.display());
            std::process::exit(1);
        }
    }
    let q8_router_flag = args.q8_router;
    // Conv1d (DeltaNet) defaults to Q8 regardless of --format — the tensor is
    // small (~32K elem) but runs every token and lossy 4-bit FWHT formats
    // measurably hurt the gated-delta path. Override with --no-q8-conv1d to
    // keep conv1d at the same quant as the rest of the model.
    let q8_conv1d_default = !args.no_q8_conv1d;
    let no_kmap = args.no_kmap || args.uniform;

    // ── imatrix loader (consumed by AWQ pre-scaling) ──
    // --imatrix <path>: load an llama-imatrix-produced GGUF (per `examples/
    // imatrix_collect.rs`). Populates the IMATRIX OnceLock with per-channel
    // `Σ_token act²` values keyed by ggml-style tensor name. Quantizer behavior
    // with no `--imatrix` is byte-equivalent to baseline.
    //
    // For Qwen3.5 hybrid layers, the mapper covers: ffn_{gate,up,down},
    // self_attn.{q,k,v,o}_proj (full-attention layers), and
    // linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}
    // (linear-attention layers via SSM-naming). Norms / biases / 1D scalars /
    // conv1d / lookup tables have no imatrix entry.
    let imatrix_path = args.imatrix.clone();
    if let Some(path) = &imatrix_path {
        if !path.exists() {
            eprintln!("error: --imatrix path not found: {}", path.display());
            std::process::exit(1);
        }
        let table = load_imatrix(path);
        IMATRIX
            .set(table)
            .expect("IMATRIX set twice — should not happen");
        eprintln!("imatrix loaded from {}", path.display());
    }

    // ── Phase A Stage A: AWQ (Activation-aware Weight Quantization) ──
    // --awq           → enable AWQ at default alpha=0.55
    // --awq-alpha <f> → enable AWQ at explicit alpha (overrides default)
    // Requires --imatrix (we derive RMS_act from imatrix's in_sum2 values).
    // Per-channel scaling: W' = W · diag(s) at quantize time, sidecar
    // 1D F16 tensor <weight>.awq_scale stored alongside the parent weight.
    // Runtime path divides activations by s before the rotation kernel —
    // separate change, not in this patch. Implementation reference:
    // docs/plans/awq_hipfire.md.
    //
    // Stage A targets MQ4G256 specifically (large g=256 → AWQ's outlier-
    // mitigation works; per Egiazarian et al 2509.23202 §3.2, small-group
    // formats (g=16/32 NVFP4/MXFP4) "provably neutralize traditional
    // outlier mitigation techniques" — MR-GPTQ is the right lever there,
    // tracked as Stage C). HFP4/MFP4 are explicitly NOT awq-pre-scaled
    // in this patch.
    let awq_enabled = args.awq || args.awq_alpha.is_some();
    let awq_alpha = args.awq_alpha.unwrap_or(0.55);
    if awq_enabled {
        if IMATRIX.get().is_none() {
            eprintln!("error: --awq requires --imatrix (we derive RMS_act per channel from imatrix in_sum2 values)");
            std::process::exit(1);
        }
        if !(0.0..=1.0).contains(&awq_alpha) {
            eprintln!(
                "warning: --awq-alpha {awq_alpha} outside typical [0, 1] range; using anyway"
            );
        }
        AWQ_ALPHA
            .set(awq_alpha)
            .expect("AWQ_ALPHA set twice — should not happen");
        eprintln!("AWQ pre-scaling: ENABLED (alpha={awq_alpha}, formula: s[j]=(RMS_act[j])^alpha, geo-mean normalized to 1)");
    }
    // K-map gate: applies to MoE models by default. Dense models opt in
    // via --kmap-dense (the K-map dense PPL effect is mixed: regression at
    // short context, win at long context — see benchmarks/results/
    // ppl_kmap_20260508.md). Maintainer directive 2026-05-08: "intends to
    // help ONLY (never on dense)" by default.
    let kmap_dense = args.kmap_dense;
    // K-map mode: 0=full (all candidates promoted), 1=alternating (edge + every 3rd),
    // 2=typed (ffn_down+attn_v everywhere). Default: alternating — same PPL as full
    // at 17% less model size on MoE (22.9 vs 27.7 GB, PPL 8K: 19.96 vs 20.07).
    let kmap_mode: u8 = match args.kmap_mode.as_str() {
        "full" | "0" => 0,
        "alternating" | "alt" | "1" => 1,
        "typed" | "2" => 2,
        _ => {
            eprintln!(
                "warning: unknown --kmap-mode '{}', using alternating",
                args.kmap_mode
            );
            1
        }
    };

    // ── Sub-4-bit guards (2026-04-30 sweep) ─────────────────────────────
    // MQ2 with the current uniform 4-level codebook collapses at every
    // model size validated locally (0.8B / 4B / 9B Qwen 3.5 → multilingual
    // mojibake on all 4 coherence-gate prompts). Refuse by default until
    // Path D Lloyd-Max non-uniform codebooks land (PRD §5.2).
    let allow_mq2 = args.allow_mq2
        || hipfire_config::developer_var("HIPFIRE_ALLOW_MQ2")
            .ok()
            .as_deref()
            == Some("1");
    if use_mq2g256 && !allow_mq2 {
        eprintln!(
            "error: --format mq2 is reserved — empirical quality verdict is collapse on every model\n\
             size validated locally (0.8B / 4B / 9B Qwen 3.5 → mojibake / symbol soup on all 4\n\
             coherence-gate prompts). The current uniform 4-level codebook is fundamentally too\n\
             lossy; Path D Lloyd-Max non-uniform codebooks (per-block squared-error-minimising)\n\
             are the planned remediation per PRD §5.2.\n\
             \n\
             To opt in for research / ablation purposes anyway, pass --allow-mq2 or set\n\
             HIPFIRE_ALLOW_MQ2=1. Don't ship MQ2 artifacts to users until the codebook\n\
             improvement lands."
        );
        std::process::exit(1);
    }
    // MQ2-Lloyd: rescues uniform MQ2 by 41–55× (per benchmarks/results/
    // lloyd_max_findings_20260501.md) but still text-collapse — 9B ppl=2,163
    // vs 9B MQ4 ppl=10. Research-only: same opt-in gate so users don't
    // accidentally ship a 2-bpw model that won't produce coherent output.
    let allow_mq3_lloyd = args.allow_mq3_lloyd
        || hipfire_config::developer_var("HIPFIRE_ALLOW_MQ3_LLOYD")
            .ok()
            .as_deref()
            == Some("1");
    if use_mq3g256_lloyd && !allow_mq3_lloyd {
        eprintln!(
            "note: --format mq3-lloyd is research — Lloyd-Max 8-entry codebook +\n\
             3-bit indices (112 B/group, +7.7% over uniform MQ3). Hypothesis is\n\
             non-uniform codebook lifts sub-9B MQ3 out of collapse (#114) and\n\
             tightens 9B MQ3's 4× ppl gap vs MQ4. Ppl evidence pending — DO NOT\n\
             ship MQ3-Lloyd artifacts to users until quality is validated against\n\
             baseline MQ3/MQ4 ppl.\n\
             \n\
             To proceed, pass --allow-mq3-lloyd or set HIPFIRE_ALLOW_MQ3_LLOYD=1."
        );
        std::process::exit(1);
    }
    let allow_mq2_lloyd = args.allow_mq2_lloyd
        || hipfire_config::developer_var("HIPFIRE_ALLOW_MQ2_LLOYD")
            .ok()
            .as_deref()
            == Some("1");
    if (use_mq2g256_lloyd
        || use_mq4_mq2lloydexp
        || use_mq4_mq2glexp
        || use_mq4_mq2lloyd_native
        || use_mq4_mq2lloyd_kmap
        || use_mq4_mq2lloyd_imatrix
        || use_mq4_mq3lloyd_kmap
        || use_mq4_mq2lloyd_kmap
        || use_mq4_mqlloyd_tiered
        || use_mq4_mqlloyd_antirez
        || use_mq4_mqlloyd_antirez_gptq
        || use_mq4_mq2lloyd_gptq_all
        || use_deepseek4_source_precision)
        && !allow_mq2_lloyd
    {
        eprintln!(
            "error: --format mq2-lloyd is research-only — Lloyd-Max codebook lifts\n\
             uniform MQ2 by 41–55× ppl but absolute quality is still collapse\n\
             (9B Qwen 3.5 wikitext2-test ppl=2,163 vs MQ4=10, MQ3=42; 0.8B ppl=19,651).\n\
             2 bpw is fundamentally too aggressive for usable text; the format\n\
             is plumbed for follow-on Lloyd-Max MQ3 (qt=20) experiments only.\n\
             \n\
             To opt in for research anyway, pass --allow-mq2-lloyd or set\n\
             HIPFIRE_ALLOW_MQ2_LLOYD=1. Don't ship MQ2-Lloyd artifacts to users."
        );
        std::process::exit(1);
    }
    // MQ4-Lloyd: extension of MQ3-Lloyd to K=16 centroids. Conjectured to
    // narrow the MQ4 → MQ6 ppl gap at +17.6% bandwidth over uniform MQ4
    // (160 vs 136 B/group). Per
    // benchmarks/results/devlog_20260506_lloyd_mq4_extension.md the
    // 9B projection is ppl 8.0–9.3 (vs uniform MQ4 ppl 10.34, MQ6 ppl 9.36).
    // Quality not yet validated — same opt-in gate as MQ3-Lloyd until ppl
    // numbers land.
    let allow_mq4_lloyd = args.allow_mq4_lloyd
        || hipfire_config::developer_var("HIPFIRE_ALLOW_MQ4_LLOYD")
            .ok()
            .as_deref()
            == Some("1");
    if use_mq4g256_lloyd && !allow_mq4_lloyd {
        eprintln!(
            "note: --format mq4-lloyd is research — Lloyd-Max 16-entry codebook +\n\
             4-bit indices (160 B/group, +17.6% over uniform MQ4). Hypothesis is\n\
             non-uniform codebook narrows the MQ4 → MQ6 ppl gap at lower bandwidth\n\
             than uniform MQ6. Ppl evidence pending — DO NOT ship MQ4-Lloyd\n\
             artifacts to users until quality is validated against baseline\n\
             MQ4/MQ6 ppl on the target model.\n\
             \n\
             To proceed, pass --allow-mq4-lloyd or set HIPFIRE_ALLOW_MQ4_LLOYD=1."
        );
        std::process::exit(1);
    }
    // MQ3 quality threshold ≈ 9B from the same sweep — 27B + 9B fluent,
    // 4B partial-collapse (intent recognised, language drifts), 0.8B
    // gibberish. Print a soft advisory so users running --format mq3
    // against small models don't think the engine is broken.
    if use_mq3g256 {
        eprintln!(
            "note: MQ3 empirical quality threshold ≈ 9B params. 27B / 9B Qwen 3.5 produce\n\
             fluent output across the coherence-gate battery; 4B partially collapses\n\
             (intent recognised, language mixes / loops); 0.8B is incoherent. For models\n\
             below ~9B, prefer --format mq4 (same kernel family, ~30% larger but\n\
             reliably coherent).\n"
        );
    }

    // GGUF input branch: if --input is a `.gguf` file, run the GGUF
    // pipeline and exit. Tensor names are translated GGUF → safetensors
    // style. The 2D quantization target follows --format:
    //   hfq4 (default for GGUF) | hfq6 | mq4 | mq6
    // Per CLAUDE.md guidance: dense (non-DeltaNet) models should use
    // hfq4/hfq6. mq4/mq6 are calibrated for Qwen3.5+ — using them on a
    // Llama-style model produces correct output (the FWHT cancels in
    // `gemv_mq4g256_with_rotate`) but adds runtime rotation overhead
    // with no quality benefit.
    {
        let raw_input = Path::new(input_dir);
        if is_gguf_input(raw_input) {
            let gguf_format = GgufFormat::from_flag(format).unwrap_or_else(|| {
                eprintln!(
                    "GGUF input: --format '{format}' not recognized. \
                     Supported: hfq4 (default), hfq6, mq4, mq6. \
                     Falling back to hfq4."
                );
                GgufFormat::Hfq4
            });
            let out = Path::new(output_path);
            if let Err(e) = run_gguf_pipeline(
                raw_input,
                out,
                gguf_format,
                no_kmap,
                kmap_dense,
                kmap_mode,
                args.arch_id,
                args.force_arch_id,
            ) {
                eprintln!("GGUF pipeline failed: {e}");
                std::process::exit(2);
            }
            return;
        }
    }

    // Resolve input: local path or HuggingFace model ID (e.g. "Qwen/Qwen3-8B")
    let input_dir = resolve_model_path(input_dir);
    let input_dir = Path::new(&input_dir);
    let output_path = Path::new(output_path);

    // Read model config
    let config_path = input_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|_| panic!("Cannot read {}. If using a HuggingFace model ID, ensure it's downloaded: huggingface-cli download {}", config_path.display(), input_dir.display()));
    let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

    let arch_str = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("llama");
    let auto_arch_id = match arch_str {
        "llama" => 0u32,
        "qwen3" | "qwen2" => 1,
        "qwen3_5" | "qwen3_5_text" | "qwen35" => 5,
        // Qwen3.5 MoE (Qwen3.5-35B-A3B and friends): hybrid LA+FA attention identical
        // to qwen3_5 dense, but every layer's FFN is MoE with stacked-3D expert
        // tensors (mlp.experts.gate_up_proj/down_proj are [num_experts, ...]).
        "qwen3_5_moe" | "qwen3_5_moe_text" => 6,
        // dots.ocr (Qwen2-VL family layout-extraction VLM): plain Qwen2-1.5B
        // text decoder + 42-block DotsVisionTransformer with 2-D RoPE,
        // SwiGLU, RMSNorm. Crate: hipfire-arch-dots-ocr. See docs/plans/
        // dots-ocr-prd.md.
        "dots_ocr" => 8,
        // DeepSeek V4 Flash: 256 routed + 1 shared experts, Hyper-Connections,
        // compressed-KV indexer, FP8 E4M3 + UE8M0 block-scale storage. See
        // crates/hipfire-arch-deepseek4. Phase 1 ingest only — no forward
        // path yet; tensor names ship in DeepSeek V4's native shape (split w1/w2/w3,
        // per-expert) and are translated when the forward bring-up lands.
        "deepseek_v4" => 9,
        // MiniMax-M2 (Mixtral-style MoE): GQA + per-layer QK-norm + partial
        // rotate_half RoPE; 256 routed experts top-8 sigmoid+e_score_bias, no
        // shared expert; FP8 E4M3 + F32 weight_scale_inv block-128 storage;
        // split per-expert w1/w3/w2 (like deepseek_v4). Crate hipfire-arch-minimax.
        "minimax_m2" => 10,
        // LFM2.5 (LiquidAI): hybrid short-conv + GQA-attn layers, SwiGLU FFN.
        //   "lfm2_moe" = A1B (dense MLP head layers + top-4 MoE); per-expert
        //               pre-split w1/w2/w3 → MQ4G256, everything else → Q8.
        //   "lfm2"     = dense (Lfm2ForCausalLM, e.g. 350M/1.2B) — no experts,
        //               every layer dense SwiGLU; the ingest Q8s all tensors.
        // Crate hipfire-arch-lfm2moe (arch_id 11); loader handles both via
        // num_dense_layers == num_hidden_layers for the dense variant.
        "lfm2_moe" | "lfm2" => 11,
        // Cohere2-MoE (CohereLabs/North-Mini-Code-1.0): parallel-block
        // transformer, interleaved sliding(RoPE)/global(NoPE) GQA, mean-centered
        // Cohere2LayerNorm, sigmoid 128-expert MoE (norm_topk_prob=false, no bias,
        // no shared), dense layer-0 (first_k_dense_replace=1), tied embeddings.
        // Per-expert pre-split tensors (mlp.experts.{j}.{gate,up,down}_proj) like
        // lfm2/deepseek. Crate hipfire-arch-cohere2moe (arch_id 12).
        "cohere2_moe" => 12,
        other => {
            eprintln!("Warning: unknown architecture '{other}', treating as llama");
            0
        }
    };
    // --arch-id <u32> overrides the auto-detected id. Use when the
    // model's family maps to a different crate than the default
    // (e.g. plain Qwen2 → arch_id=7 for the hipfire-arch-qwen2 crate
    // instead of the LLaMA-family default 1, which silently drops
    // Q/K/V bias on the LLaMA loader path). See docs/plans/
    // dots-ocr-devlog.md §7 (R1) for the bring-up context.
    let arch_id = args.arch_id.unwrap_or(auto_arch_id);
    guard_qwen3_arch_override(auto_arch_id, arch_id, args.force_arch_id);
    if arch_id != auto_arch_id {
        eprintln!("Architecture: {arch_str} (auto id={auto_arch_id}, overridden via --arch-id to {arch_id})");
    } else {
        eprintln!("Architecture: {arch_str} (id={arch_id})");
    }
    let is_moe = arch_id == 6;
    // DeepSeek V4 (arch_id=9 post-2026-05-26 upstream merge that promoted
    // Qwen2-dense to 7 and dots.ocr to 8) is also MoE but ships per-expert
    // separate 2D tensors (`layers.L.ffn.experts.E.{w1,w2,w3}.weight`)
    // instead of Qwen3.5's stacked 3D `mlp.experts.gate_up_proj`. Phase 1
    // ingest handles DeepSeek V4's per-expert tensors individually through
    // the standard 2D quant path; the routing fan-out into top-k experts
    // happens at forward time, not quant time.
    let is_deepseek4 = arch_id == 9;
    // LFM2.5 (arch_id 11): A1B routes per-expert w1/w2/w3 → MQ4G256, expert_bias
    // → F32, everything else → Q8; dense lfm2 (Lfm2ForCausalLM, e.g. 350M/1.2B)
    // has no experts so the ingest just Q8s every tensor (the loader's load_f32
    // dequantizes norms / conv-filter back to F32).
    let is_lfm2moe = arch_id == 11;
    // Cohere2-MoE (arch_id 12): per-expert pre-split tensors; experts carry the
    // bit-width knob (--format f16|q8|mq6|mq4), attention/dense/router stay Q8
    // (F16 in the oracle), tied embed stays Q8, norms -> F16.
    let is_cohere2moe = arch_id == 12;
    // MiniMax-M2 (arch_id=10): MoE like DeepSeek V4, ships per-expert pre-split
    // 2D tensors (`...block_sparse_moe.experts.E.{w1,w2,w3}.weight`). Quantized
    // as HFQ4G256 (the only 4-bit format with a complete indexed-MoE GEMV
    // kernel family). Raw HF tensor names are written verbatim (no rename);
    // the hipfire loader looks them up.
    let is_minimax = arch_id == 10;
    let is_moe_like = is_moe || is_deepseek4 || is_lfm2moe || is_minimax || is_cohere2moe;
    // Q8 "router" — a misnomer: `is_q8_tensor` covers the whole FIXED tier
    // (attention q/k/v/o, linear_attn projections, conv1d, lm_head, embed, and
    // the MoE router), not just `mlp.gate.weight`. On for MoE-class models by
    // default, since the fixed tier is quality-critical and cheap relative to
    // the routed experts *by parameter count*.
    //
    // `--no-q8-router` restores the historic opt-out. It matters far more than
    // the name suggests: the fixed tier is **66% of per-token decode bytes** on
    // a3b (mq4r: 1030.8 MB fixed vs 534.8 MB routed), so forcing it to Q8
    // (1.0625 B/w) instead of MQ4 (0.53125 B/w) doubles the dominant term. That
    // is why `.mq2` reads 45% MORE bytes/token than `.mq4r` despite being 7 GB
    // smaller on disk, and why `.mq4r` — which needs this flag off — is not
    // byte-reproducible from HEAD without it.
    let no_q8_router_flag = args.iter().any(|a| a == "--no-q8-router")
        || std::env::var("HIPFIRE_NO_Q8_ROUTER").ok().as_deref() == Some("1");
    let q8_router = (is_moe_like || q8_router_flag) && !no_q8_router_flag;
    if no_q8_router_flag {
        eprintln!(
            "note: --no-q8-router — the fixed tier (attention / lm_head / router /\n\
             embed / conv1d) follows --format instead of being forced to Q8F16.\n\
             This is the mq4r recipe and the lever for a sub-MQ4 fixed tier;\n\
             embed_tokens still stays Q8 via its own arm."
        );
    }
    if is_moe {
        eprintln!("  MoE detected — will split 3D expert tensors per-expert before quantization.");
    }
    if is_deepseek4 {
        eprintln!("  DeepSeek V4 detected — per-expert tensors ship pre-split; quantizing each as 2D weight.");
    }
    if is_lfm2moe {
        eprintln!("  LFM2.5 detected — experts → MQ4G256, expert_bias → F32, all else (conv/attn/dense/router/embed/norms) → Q8.");
    }
    if is_minimax {
        eprintln!("  MiniMax-M2 detected — per-expert tensors ship pre-split; quantizing each as HFQ4G256 2D weight.");
    }
    if is_cohere2moe {
        eprintln!("  Cohere2-MoE detected — experts → --format ({{f16|q8|mq6|mq4}}); attn/dense → Q8 (F16 in oracle); router/embed → Q8; norms → F16.");
    }

    // Extract layer count for K-map edge-layer promotion.
    // Qwen3.5+ nests config under "text_config"; try both paths.
    let n_layers: usize = config
        .get("num_hidden_layers")
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|tc| tc.get("num_hidden_layers"))
        })
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    if n_layers == 0 {
        eprintln!(
            "  warning: num_hidden_layers not found in config.json — edge-layer promotion disabled"
        );
    }

    // Read tokenizer if present
    let tokenizer_json = input_dir.join("tokenizer.json");
    let tokenizer_str = if tokenizer_json.exists() {
        std::fs::read_to_string(&tokenizer_json).ok()
    } else {
        None
    };

    // Read tokenizer_config.json (has chat_template)
    let tokenizer_config_path = input_dir.join("tokenizer_config.json");
    let tokenizer_config: Option<serde_json::Value> = if tokenizer_config_path.exists() {
        std::fs::read_to_string(&tokenizer_config_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
    } else {
        None
    };

    // Some checkpoints (e.g. LFM2.5) ship the Jinja chat template in a separate
    // `chat_template.jinja` file rather than inside tokenizer_config.json. The
    // daemon extracts its template from `tokenizer_config.chat_template` (then
    // renders via minijinja); fold the sidecar in when tokenizer_config lacks
    // one, else the daemon falls back to Plain framing and a chat-tuned model
    // produces garbage (LFM2.5-350M bring-up, 2026-06-07).
    let tokenizer_config = {
        let mut tc = tokenizer_config;
        let jinja_path = input_dir.join("chat_template.jinja");
        if jinja_path.exists() {
            let has_template = tc
                .as_ref()
                .and_then(|v| v.get("chat_template"))
                .map(|v| !v.is_null())
                .unwrap_or(false);
            if !has_template {
                if let Ok(jinja) = std::fs::read_to_string(&jinja_path) {
                    let n = jinja.len();
                    let obj = tc.get_or_insert_with(|| serde_json::json!({}));
                    if let Some(map) = obj.as_object_mut() {
                        map.insert(
                            "chat_template".to_string(),
                            serde_json::Value::String(jinja),
                        );
                        eprintln!(
                            "  embedded chat_template.jinja into tokenizer_config ({n} bytes)"
                        );
                    }
                }
            }
        }
        tc
    };

    // Read generation_config.json. HF stores some sampler-side defaults
    // here (eos_token_id, pad_token_id, bos_token_id, do_sample, etc.)
    // separately from config.json. For most checkpoints these duplicate
    // config.json fields, but dots.ocr's config.json carries no
    // eos_token_id at all — the [151643, 151673] array lives only in
    // generation_config.json. Packing it here lets the arch-side parser
    // (e.g. `hipfire-arch-qwen2::Qwen2Config::from_hfq`) fall back to
    // generation_config when config.eos_token_id is absent. Resolves
    // R5 in docs/plans/dots-ocr-devlog.md §7.
    let generation_config_path = input_dir.join("generation_config.json");
    let generation_config: Option<serde_json::Value> = if generation_config_path.exists() {
        std::fs::read_to_string(&generation_config_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
    } else {
        None
    };

    // Build metadata JSON for .hfq
    let metadata = serde_json::json!({
        "architecture": arch_str,
        "config": config,
        "tokenizer": tokenizer_str.as_deref().unwrap_or("{}"),
        "tokenizer_config": tokenizer_config,
        "generation_config": generation_config,
    });
    // `mut` so the SP4b bake-prune path can patch the routed-expert count down to
    // the kept count before write_hfq (so the baked model loads with the compact
    // count and NO env var). Untouched in every non-prune path.
    let mut metadata_json = serde_json::to_string(&metadata).unwrap();

    // Load all safetensors files
    let st_files: Vec<SafetensorsFile> = find_safetensors(input_dir)
        .iter()
        .map(|p| {
            eprintln!("Loading: {}", p.display());
            SafetensorsFile::open(p).unwrap()
        })
        .collect();

    // Collect all tensor names.
    //
    // DeepSeek V4 note: tensors come in `<name>.weight` (I8 = E4M3) + `<name>.scale`
    // (F8_E8M0) pairs. We index the `.scale` siblings into a side map
    // keyed by the weight tensor's full name and skip them in the main
    // iteration. When we encounter the `.weight` half we look up the
    // sibling and call `dequantize_e4m3_ue8m0_to_f32` to recover f32
    // before the existing MQ-family pipeline runs.
    let mut all_tensors: Vec<(&str, usize)> = Vec::new();
    let mut fp8_scale_for: HashMap<String, (usize, String)> = HashMap::new();
    for (fi, st) in st_files.iter().enumerate() {
        for name in st.tensor_names() {
            // MiniMax-M2 FP8: `<w>.weight` (e4m3) + `<w>.weight_scale_inv` (F32
            // block-[128,128] scale). Strip the longer suffix FIRST.
            if let Some(stem) = name.strip_suffix(".weight_scale_inv") {
                let w_name = format!("{stem}.weight");
                fp8_scale_for.insert(w_name, (fi, name.to_string()));
                continue;
            }
            if let Some(stem) = name.strip_suffix(".scale") {
                // Sibling weight name (drop `.scale`, add `.weight`).
                let w_name = format!("{stem}.weight");
                fp8_scale_for.insert(w_name, (fi, name.to_string()));
                continue;
            }
            all_tensors.push((name, fi));
        }
    }
    all_tensors.sort_by_key(|(name, _)| name.to_string());
    eprintln!(
        "Found {} tensors ({} FP8 scale siblings indexed)",
        all_tensors.len(),
        fp8_scale_for.len()
    );

    // ── SP4: selective re-quant overlay mode ────────────────────────────────
    // When `--reap-overlay <plan-dir>` is set, this branch fully replaces the
    // normal whole-model quantize: it loads the reap plan, resolves the arch
    // family (auto from arch_id, or `--reap-arch` override), then iterates the
    // model tensors and — for ONLY the tensors the plan overrides — decodes
    // f32 and re-quantizes via `quantize_to_format`. Non-matched tensors skip
    // the (expensive) f32 decode entirely. The subset is written to
    // `--reap-out` via the existing `write_hfq`, keyed by original tensor name
    // so a load-time splice (SP3) can overlay them onto the base model.
    if let Some(plan_dir) = reap_overlay_dir.as_deref() {
        let reap_out_path = reap_out.as_deref().unwrap_or_else(|| {
            eprintln!("--reap-overlay requires --reap-out <overlay.hfq path>");
            std::process::exit(1);
        });
        // Resolve arch: explicit --reap-arch overrides the auto-detection.
        let arch: reap_overlay::ReapArch = match reap_arch_flag.as_deref() {
            Some(s) => reap_overlay::ReapArch::from_flag(s).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1);
            }),
            None => reap_overlay::ReapArch::from_arch_id(arch_id).unwrap_or_else(|| {
                eprintln!(
                    "reap overlay: could not auto-detect arch family from arch_id={arch_id}; \
                     pass --reap-arch <deepseek4|qwen35|lfm2moe|minimax>"
                );
                std::process::exit(1);
            }),
        };
        let plan = hipfire_reap::plan::ReapPlan::load_unchecked(plan_dir).unwrap_or_else(|e| {
            eprintln!("reap overlay: failed to load plan from {plan_dir}: {e}");
            std::process::exit(1);
        });
        eprintln!(
            "REAP overlay mode: arch={arch:?}, {} quant_overrides, out={reap_out_path}",
            plan.quant_overrides.len()
        );

        let mut hfq_tensors: Vec<HfqTensor> = Vec::new();
        for (name, file_idx) in &all_tensors {
            // Check the plan BEFORE decoding f32 — skipping the decode of
            // non-matched tensors is the whole point of an overlay build.
            if reap_overlay::reap_override_for(name, arch, &plan).is_none() {
                continue;
            }
            let (meta, raw_data) = st_files[*file_idx].tensor_data(name).unwrap();
            let f32 = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                &meta,
                &fp8_scale_for,
                &st_files,
            );
            let shape: Vec<usize> = meta.shape.clone();
            let tier = reap_overlay::reap_override_for(name, arch, &plan).unwrap();
            match reap_overlay::quantize_to_format(name, tier, &f32, &shape) {
                Ok(t) => {
                    eprintln!("  overlay: {name} → {tier} ({} bytes)", t.data.len());
                    hfq_tensors.push(t);
                }
                Err(e) => {
                    eprintln!("reap overlay: {e}");
                    std::process::exit(2);
                }
            }
        }

        if hfq_tensors.is_empty() {
            eprintln!(
                "reap overlay: no tensors matched the plan's quant_overrides \
                 (check arch/layer/expert names)"
            );
            std::process::exit(1);
        }

        eprintln!(
            "REAP overlay: {} tensors quantized; writing {reap_out_path}",
            hfq_tensors.len()
        );
        write_hfq(
            Path::new(reap_out_path),
            arch_id,
            &metadata_json,
            &hfq_tensors,
            None,
        )
        .unwrap_or_else(|e| {
            eprintln!("reap overlay: failed to write {reap_out_path}: {e}");
            std::process::exit(2);
        });
        eprintln!("REAP overlay written: {reap_out_path}");
        return;
    }

    // ── SP4b: bake-mode setup ────────────────────────────────────────────────
    // `--reap-bake <plan-dir>` keeps the normal whole-model quantize loop but
    // activates the per-tensor override hook (at the top of the loop below).
    // Resolve the plan + arch family up front; the loop reads `reap_bake_plan`
    // and `reap_arch`. When bake is inactive these are unused / None and the
    // loop is byte-identical to today. If `--reap-out` is given, the whole
    // baked model is written there instead of the normal `--output` path.
    let reap_bake_plan: Option<hipfire_reap::plan::ReapPlan> = match reap_bake_dir.as_deref() {
        Some(plan_dir) => Some(
            hipfire_reap::plan::ReapPlan::load_unchecked(plan_dir).unwrap_or_else(|e| {
                eprintln!("reap bake: failed to load plan from {plan_dir}: {e}");
                std::process::exit(1);
            }),
        ),
        None => None,
    };
    // Arch family for tensor-name matching: explicit --reap-arch overrides the
    // auto-detection from arch_id (only consulted when bake is active).
    let reap_arch: reap_overlay::ReapArch = if reap_bake_plan.is_some() {
        match reap_arch_flag.as_deref() {
            Some(s) => reap_overlay::ReapArch::from_flag(s).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1);
            }),
            None => reap_overlay::ReapArch::from_arch_id(arch_id).unwrap_or_else(|| {
                eprintln!(
                    "reap bake: could not auto-detect arch family from arch_id={arch_id}; \
                     pass --reap-arch <deepseek4|qwen35|lfm2moe|minimax>"
                );
                std::process::exit(1);
            }),
        }
    } else {
        // Placeholder (never read when reap_bake_plan is None).
        reap_overlay::ReapArch::Qwen35
    };
    // Redirect the whole-model output to --reap-out when baking with that flag.
    let bake_out_path = reap_bake_plan
        .as_ref()
        .and(reap_out.as_deref())
        .map(Path::new);
    let output_path: &Path = bake_out_path.unwrap_or(output_path);
    if let Some(plan) = &reap_bake_plan {
        eprintln!(
            "REAP bake mode: arch={reap_arch:?}, {} quant_overrides, out={}",
            plan.quant_overrides.len(),
            output_path.display()
        );
    }
    // Is expert pruning active? (A bake plan with a per-layer keep-map.) When
    // active, the loop's prune hook drops pruned per-expert tensors, the kept
    // per-expert tensors are recorded in `bake_rename` for a post-loop renumber
    // to compact slots, routers/biases are row-gathered to the kept set, and the
    // output metadata's expert count is patched to `kept_per_layer`.
    let bake_keep_active = reap_bake_plan
        .as_ref()
        .map(|p| p.keep.is_some())
        .unwrap_or(false);
    // original-name → compact-renamed-name for kept per-expert tensors
    // (ds4 score layers / lfm2 / minimax). Applied as a post-loop rename pass so
    // the per-expert quant branches keep using the ORIGINAL name to read source
    // bytes, then we rewrite `HfqTensor.name` to the compact slot before write.
    let mut bake_rename: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    // Task A0: original gate_proj name → fused `experts.{N}.gate_up_proj.weight`,
    // for ORNITH-class Qwen3.5-MoE checkpoints that ship experts un-stacked.
    // Applied as an UNCONDITIONAL post-loop rename pass (see below).
    let mut expert_fuse_rename: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    // ── K-map pre-pass ──────────────────────────────────────────────────────
    // Build per-tensor quant level map. Gated to MoE models by default
    // (maintainer directive 2026-05-08): K-map's dense PPL effect is mixed
    // (+1.5% to +2.5% at 2K, -4.8% at 8K — crossover at ~3K context). To
    // avoid silently changing dense quantization output, dense models opt
    // out by default and require `--kmap-dense` to enable. MoE models keep
    // the K-map default-on path because the routed-expert promotion is
    // the headline win and the empirical regression there is tighter
    // (+1.7% PPL at 2K, gated below the dense regression threshold).
    let kmap: HashMap<String, QuantLevel> = if no_kmap || (!is_moe && !kmap_dense) {
        HashMap::new()
    } else {
        let mut map = HashMap::new();
        let mut counts = [0u32; 4]; // F16, Q8, Promote6, Base
        for (name, _fi) in &all_tensors {
            let level = kmap_resolve_mode(name, n_layers, is_moe, kmap_mode);
            match level {
                QuantLevel::F16 => counts[0] += 1,
                QuantLevel::Q8 => counts[1] += 1,
                QuantLevel::Promote6 => counts[2] += 1,
                QuantLevel::Override(_) => counts[3] += 1,
                QuantLevel::Base => counts[3] += 1,
            }
            map.insert(name.to_string(), level);
        }
        if !map.is_empty() {
            let mode_label = match kmap_mode {
                0 => "full",
                1 => "alternating",
                2 => "typed",
                _ => "?",
            };
            eprintln!(
                "K-map plan ({format} base, {n_layers} layers{}, mode={mode_label}):",
                if is_moe { ", MoE" } else { "" }
            );
            eprintln!("  F16:       {:>4} tensors (norms, biases)", counts[0]);
            eprintln!(
                "  Q8:        {:>4} tensors (embed, lm_head, routers)",
                counts[1]
            );
            eprintln!("  Promote6:  {:>4} tensors", counts[2]);
            eprintln!("  Base:      {:>4} tensors (remaining)", counts[3]);
        }
        map
    };

    // Phase 5: per-layer tier set — which routed-expert layers go MQ3-Lloyd
    // vs MQ2-Lloyd. Only populated for `--format mq4-mqlloyd-tiered`.
    // Computed once from imatrix .counts; kmap-promoted layers are excluded
    // (they always go MQ6).
    let mq3_tier_layers: std::collections::HashSet<usize> = if use_mq4_mqlloyd_tiered {
        if let Some(ref gguf) = imatrix_gguf {
            if let Some(layer_counts) = imatrix_layer_activation_counts(gguf, n_layers) {
                // Indexes of layers NOT promoted by K-map. We need a name
                // representative of each layer's expert tensor to query
                // kmap; use the canonical safetensors name format.
                let candidates: Vec<usize> = (0..n_layers)
                    .filter(|&l| {
                        let probe_name =
                            format!("model.language_model.layers.{}.mlp.experts.gate_up_proj", l);
                        kmap.get(&probe_name) != Some(&QuantLevel::Promote6)
                    })
                    .collect();
                let mut ranked: Vec<(usize, f64)> = candidates
                    .iter()
                    .filter(|&&l| layer_counts[l].is_finite())
                    .map(|&l| (l, layer_counts[l]))
                    .collect();
                // Sort by count DESC (hot layers first).
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let n_mq3 = ((ranked.len() as f64) * tier_ratio).round() as usize;
                let n_mq3 = n_mq3.min(ranked.len());
                let set: std::collections::HashSet<usize> =
                    ranked.iter().take(n_mq3).map(|&(l, _)| l).collect();
                eprintln!(
                    "Tiered MQ-Lloyd: {} candidate non-promoted layers; \
                     {} (top {:.0}%) → MQ3-Lloyd, {} → MQ2-Lloyd",
                    ranked.len(),
                    set.len(),
                    tier_ratio * 100.0,
                    ranked.len().saturating_sub(set.len())
                );
                if set.len() <= 16 {
                    eprintln!(
                        "  MQ3-Lloyd layers (by count): {:?}",
                        ranked
                            .iter()
                            .take(n_mq3)
                            .map(|&(l, c)| (l, c as u64))
                            .collect::<Vec<_>>()
                    );
                }
                set
            } else {
                eprintln!("warning: imatrix has no ffn_gate_exps counts — tiering disabled");
                std::collections::HashSet::new()
            }
        } else {
            std::collections::HashSet::new()
        }
    } else {
        std::collections::HashSet::new()
    };

    // Quantize
    let mut hfq_tensors = Vec::new();
    let mut total_params = 0u64;
    let mut quantized_params = 0u64;
    // Spill file for large models — keeps peak RSS bounded by flushing
    // completed tensor data to disk when accumulated memory exceeds 32 GB.
    // HIPFIRE_SPILL_DIR overrides the spill location (default = output dir).
    // Point it at a RAM-backed tmpfs (e.g. /dev/shm) to keep peak DISK usage
    // = output size only, when disk is tight but RAM is ample.
    let spill_dir_override = std::env::var("HIPFIRE_SPILL_DIR").ok();
    let spill_dir = match spill_dir_override.as_deref() {
        Some(d) => Path::new(d),
        None => output_path.parent().unwrap_or(Path::new(".")),
    };
    // HIPFIRE_NO_SPILL=1 disables the disk spill entirely (hold all tensors in
    // RAM, write output directly). Needed for huge f32 oracles where spill+output
    // would be ~2x the output size on disk — but RAM is ample.
    let mut spill = if hipfire_config::developer_var("HIPFIRE_NO_SPILL")
        .ok()
        .as_deref()
        == Some("1")
    {
        None
    } else {
        TensorSpill::new(spill_dir).ok()
    };
    let mut total_quant_error = 0.0f64;
    let mut max_quant_error = 0.0f32;
    let mut _n_quant_groups = 0u64;

    let include_vision = args.include_vision;
    let vision_quant = args.vision_quant.as_str();
    // --include-prefix <prefix>: when set, ONLY tensors whose name starts
    // with this prefix are ingested; everything else is silently skipped.
    // Used to produce side-car HFQs (e.g. `--include-prefix mtp.` builds an
    // MTP-only addon that pairs with an existing base HFQ via the loader's
    // `.mtp-addon.hfq` discovery). When unset (default), all tensors pass
    // this gate and the usual mtp/vision skip rules below apply.
    let include_prefix = args.include_prefix.as_deref();
    if let Some(p) = include_prefix {
        eprintln!(
            "  [filter] --include-prefix {p:?} — only tensors with this prefix will be ingested"
        );
    }
    let mut skipped_params = 0u64;
    // MiniMax AWQ: shared-per-layer expert scales, cached + sidecars emitted once.
    let mut mm_awq_cache: std::collections::HashMap<usize, Option<(Vec<f32>, Vec<f32>)>> =
        std::collections::HashMap::new();
    let mut mm_awq_emitted: std::collections::HashSet<usize> = std::collections::HashSet::new();
    // Task A0: name → shard index, so the pre-split expert fusion can fetch a
    // gate_proj's up_proj sibling (which may live in a different shard).
    let name_to_file: std::collections::HashMap<&str, usize> =
        all_tensors.iter().map(|(n, fi)| (*n, *fi)).collect();
    for (name, file_idx) in &all_tensors {
        // --include-prefix filter (highest priority — runs before mtp/vision skips).
        if let Some(p) = include_prefix {
            if !name.starts_with(p) {
                let (meta, _) = st_files[*file_idx].tensor_data(name).unwrap();
                let n: usize = meta.shape.iter().product();
                skipped_params += n as u64;
                continue;
            }
        }
        // Skip MTP head; optionally include vision encoder for VL inference.
        // Qwen3.5-VL names vision tensors `model.visual.*` / `visual.*`;
        // dots.ocr names them `vision_tower.*`. Both fall through to the
        // F16 fallback path (see should_quantize: vision_tower.* is
        // skipped from quantization) when --include-vision is set.
        let is_vision = name.starts_with("model.visual.")
            || name.starts_with("visual.")
            || name.starts_with("vision_tower.");
        if is_vision && !include_vision {
            let (meta, _) = st_files[*file_idx].tensor_data(name).unwrap();
            let n: usize = meta.shape.iter().product();
            skipped_params += n as u64;
            continue;
        }
        // MTP (Multi-Token Prediction) head: pre-Phase-5 quants skipped these
        // because no forward path consumed them. deepseek4-q8-mtp is the first format
        // that ingests the MTP layer; v3 spec-decode requires it. For other
        // formats we still skip to avoid bloating the HFQ with unused tensors.
        if name.starts_with("mtp.") && !use_deepseek4_source_precision {
            let (meta, _) = st_files[*file_idx].tensor_data(name).unwrap();
            let n: usize = meta.shape.iter().product();
            skipped_params += n as u64;
            continue;
        }

        let (meta, raw_data) = st_files[*file_idx].tensor_data(name).unwrap();
        let mut n_elements: usize = meta.shape.iter().product();
        total_params += n_elements as u64;

        // ── SP4b: bake prune hook ──────────────────────────────────────────────
        // BEFORE the override hook. When `--reap-bake`'s plan carries a keep-map,
        // prune routed experts not in `keep[L]`, renumber kept experts to compact
        // slots, and row-gather routers / per-expert biases to the kept set so the
        // baked model loads with the compact expert count and NO load-time
        // keep-map. `meta`/`raw_data` may be shadowed below with gathered owned
        // copies so the override hook + arch branches transparently quantize the
        // gathered tensor with the arch's normal encoder.
        //
        // Two owned holders are pre-declared so a gather can rebind the borrowed
        // (`meta`, `raw_data`) to point at gathered data for the rest of the body.
        let _gathered_meta: TensorMeta;
        let _gathered_bytes: Vec<u8>;
        let mut meta: &TensorMeta = meta;
        let mut raw_data: &[u8] = raw_data;
        if bake_keep_active {
            // SAFETY of indexing: bake_keep_active ⇒ reap_bake_plan.keep is Some.
            let plan = reap_bake_plan.as_ref().unwrap();
            let keep = plan.keep.as_ref().unwrap();
            let layer = reap_overlay::bake_layer_of(name);

            // Per-arch routed-expert (per-expert-named) tensors: drop pruned,
            // record kept→compact rename for the post-loop pass.
            if reap_overlay::expert_index_of(name, reap_arch).is_some() {
                let l = layer.unwrap_or_else(|| {
                    eprintln!("reap bake: routed-expert tensor '{name}' has no parseable layer");
                    std::process::exit(2);
                });
                // ds4 hash-layer guard: pruning layers 0..=2 requires a tid2eid
                // remap to the compact expert space — not supported in bake.
                if reap_arch == reap_overlay::ReapArch::Deepseek4 && l <= 2 {
                    eprintln!(
                        "reap bake: ds4 hash-layer (0-2) tid2eid remap not supported in bake; \
                         use the load-time keep-map for pruned ds4 hash layers"
                    );
                    std::process::exit(2);
                }
                if l >= keep.len() {
                    eprintln!("reap bake: layer {l} for '{name}' out of keep-map range");
                    std::process::exit(2);
                }
                match reap_overlay::bake_expert_rename(name, reap_arch, l, &keep[l]) {
                    None => {
                        // Pruned expert: drop entirely.
                        st_files[*file_idx].drop_tensor_pages(name);
                        continue;
                    }
                    Some(new_name) => {
                        if &new_name != name {
                            bake_rename.insert(name.to_string(), new_name);
                        }
                        // Fall through: the arch branch quantizes the ORIGINAL
                        // bytes; the post-loop pass renames the output tensor.
                    }
                }
            } else if let Some(l) = layer {
                // Router weight (`*.gate.weight`, shape `[orig_experts, hidden]`)
                // and per-expert bias (`*.gate.bias` / `e_score_correction_bias` /
                // `expert_bias`, shape `[orig_experts]`): row-gather to the kept
                // set BEFORE quant so the baked router/bias emit only kept rows in
                // compact-slot order (mirrors the loader's load-time gather).
                let is_router_w = reap_overlay::is_reap_router_weight(name, reap_arch)
                    && meta.shape.len() == 2
                    && meta.shape[0] == plan.original_experts;
                let is_expert_bias = reap_overlay::is_reap_expert_bias(name, reap_arch)
                    && meta.shape.len() == 1
                    && meta.shape[0] == plan.original_experts;
                if is_router_w || is_expert_bias {
                    // ds4 hash-layer guard also covers the router of layers 0..=2.
                    if reap_arch == reap_overlay::ReapArch::Deepseek4 && l <= 2 {
                        eprintln!(
                            "reap bake: ds4 hash-layer (0-2) router/tid2eid remap not supported \
                             in bake; use the load-time keep-map for pruned ds4 hash layers"
                        );
                        std::process::exit(2);
                    }
                    if l >= keep.len() {
                        eprintln!(
                            "reap bake: layer {l} for router/bias '{name}' out of keep-map range"
                        );
                        std::process::exit(2);
                    }
                    let keep_l = &keep[l];
                    match hipfire_reap::gather::gather_rows(&meta.shape, raw_data, keep_l) {
                        Ok((new_shape, gathered)) => {
                            _gathered_meta = TensorMeta {
                                dtype: meta.dtype.clone(),
                                shape: new_shape,
                                data_offsets: meta.data_offsets,
                            };
                            _gathered_bytes = gathered;
                            eprintln!(
                                "  {:>8}: {} {:?} → rows[{}] (kept {} of {})",
                                "GATHER",
                                name,
                                meta.shape,
                                keep_l.len(),
                                keep_l.len(),
                                plan.original_experts
                            );
                            meta = &_gathered_meta;
                            raw_data = &_gathered_bytes;
                        }
                        Err(e) => {
                            eprintln!("reap bake: router/bias gather '{name}': {e}");
                            std::process::exit(2);
                        }
                    }
                }
            }
        }

        // ── SP4b: bake override hook ───────────────────────────────────────────
        // When `--reap-bake` is active and the plan overrides this tensor,
        // re-quantize it to the override tier and skip the arch-specific default
        // branch below. Non-overridden tensors fall through UNCHANGED. The hook
        // is entirely behind `if let Some(plan) = &reap_bake_plan`, so default
        // mode (no `--reap-bake`) is byte-identical to before. Bookkeeping
        // mirrors the arch branches: f32-decode → push → drop_tensor_pages →
        // quantized_params → maybe_spill → continue.
        if let Some(plan) = &reap_bake_plan {
            if let Some(fmt) = reap_overlay::reap_override_for(name, reap_arch, plan) {
                let f32 = tensor_to_f32_with_optional_fp8_scale(
                    name,
                    raw_data,
                    meta,
                    &fp8_scale_for,
                    &st_files,
                );
                let shape: Vec<usize> = meta.shape.clone();
                match reap_overlay::quantize_to_format(name, fmt, &f32, &shape) {
                    Ok(t) => {
                        eprintln!("  {:>8}: {} {:?} → {fmt}", "BAKE", name, meta.shape);
                        hfq_tensors.push(t);
                    }
                    Err(e) => {
                        eprintln!("reap bake: {e}");
                        std::process::exit(2);
                    }
                }
                quantized_params += n_elements as u64;
                st_files[*file_idx].drop_tensor_pages(name);
                if let Some(ref mut s) = spill {
                    maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
                }
                continue;
            }
        }

        // ── Task A0: Qwen3.5-MoE pre-split routed-expert fusion (arch_id 6) ──
        // Canonical Qwen3.5-MoE ships routed experts stacked-3D as
        // `mlp.experts.gate_up_proj` (the paths below split it per-expert).
        // ORNITH-class finetunes instead ship them UN-stacked as separate 2D
        // `mlp.experts.{N}.{gate,up,down}_proj.weight` (DeepSeek-V4 layout). The
        // qwen35 loader only knows the fused per-expert
        // `mlp.experts.{N}.gate_up_proj.weight` ([2*inter, hidden], gate||up), so
        // fuse gate+up here and rename the output post-loop; the normal quant
        // path below encodes the [2*inter, hidden] tensor (k-map still selects
        // the level by the gate_proj name). `down_proj` already matches the
        // loader name and takes the normal path unchanged; `shared_expert` (kept
        // un-fused by the loader) is excluded by the `.mlp.experts.` guard.
        let _fused_meta: TensorMeta;
        let _fused_bytes: Vec<u8>;
        if is_moe && meta.shape.len() == 2 && name.contains(".mlp.experts.") {
            if name.ends_with(".up_proj.weight") {
                // Consumed by its gate_proj sibling (fused below).
                st_files[*file_idx].drop_tensor_pages(name);
                continue;
            }
            if let Some(stem) = name.strip_suffix(".gate_proj.weight") {
                let up_name = format!("{stem}.up_proj.weight");
                let up_fi = match name_to_file.get(up_name.as_str()) {
                    Some(fi) => *fi,
                    None => {
                        eprintln!("qwen35 expert fusion: missing sibling {up_name} for {name}");
                        std::process::exit(2);
                    }
                };
                let (up_meta, up_raw) = st_files[up_fi].tensor_data(&up_name).unwrap();
                if up_meta.shape != meta.shape || up_meta.dtype != meta.dtype {
                    eprintln!(
                        "qwen35 expert fusion: gate {:?}/{} vs up {:?}/{} mismatch at {name}",
                        meta.shape, meta.dtype, up_meta.shape, up_meta.dtype
                    );
                    std::process::exit(2);
                }
                // gate rows first, then up rows → [2*inter, hidden]. Same source
                // dtype ⇒ a raw byte concat is lossless. Order is load-bearing:
                // loader stores gate_up = gate||up; forward is silu(gate)*up.
                let mut fused = Vec::with_capacity(raw_data.len() + up_raw.len());
                fused.extend_from_slice(raw_data);
                fused.extend_from_slice(up_raw);
                _fused_bytes = fused;
                _fused_meta = TensorMeta {
                    dtype: meta.dtype.clone(),
                    shape: vec![meta.shape[0] * 2, meta.shape[1]],
                    data_offsets: meta.data_offsets,
                };
                n_elements *= 2; // fused tensor carries gate + up params
                meta = &_fused_meta;
                raw_data = &_fused_bytes;
                expert_fuse_rename.insert(name.to_string(), format!("{stem}.gate_up_proj.weight"));
                st_files[up_fi].drop_tensor_pages(&up_name);
                eprintln!(
                    "  {:>8}: {name} + up_proj → {stem}.gate_up_proj.weight {:?}",
                    "FUSE", _fused_meta.shape
                );
            }
        }

        // ── F1 native-bf16 oracle passthrough ──────────────────────────────
        // Store EVERY tensor as F32 (qt=2): no quantization, bf16/f16->f32
        // widened losslessly. This bypasses every per-format branch below so
        // the produced .hfq is a full-precision reference the qwen35 loader
        // reads via its qt=2 arm and the engine forwards through the existing
        // F32 GEMV / attention_f32 path.
        if use_f32_passthrough && !is_cohere2moe {
            // 3D MoE experts MUST be split per-expert: the qwen35 loader reads
            // `...experts.{X}.{base}.weight`, never the stacked 3D tensor. Without
            // this, the oracle stores `experts.gate_up_proj [256,...]` and load
            // panics "tensor not found: ...experts.0.gate_up_proj.weight".
            if is_moe
                && name.contains("mlp.experts.")
                && (name.ends_with("gate_up_proj") || name.ends_with("down_proj"))
                && meta.shape.len() == 3
            {
                let n_exp = meta.shape[0];
                let inner_n: usize = meta.shape[1..].iter().product();
                let base_name = if name.ends_with("gate_up_proj") {
                    "gate_up_proj"
                } else {
                    "down_proj"
                };
                let parent = &name[..name.len() - base_name.len()]; // ends with "experts."
                let inner_shape: Vec<u32> = meta.shape[1..].iter().map(|&d| d as u32).collect();
                let f32_all = tensor_to_f32_with_optional_fp8_scale(
                    name,
                    raw_data,
                    meta,
                    &fp8_scale_for,
                    &st_files,
                );
                for x in 0..n_exp {
                    let slice = &f32_all[x * inner_n..(x + 1) * inner_n];
                    let bytes: Vec<u8> = slice.iter().flat_map(|&v| v.to_le_bytes()).collect();
                    hfq_tensors.push(HfqTensor {
                        name: format!("{parent}{x}.{base_name}.weight"),
                        quant_type: QuantType::F32,
                        shape: inner_shape.clone(),
                        group_size: 0,
                        data: bytes,
                        spilled_len: 0,
                    });
                }
                quantized_params += n_elements as u64;
                eprintln!(
                    "  {:>8}: {} {:?} -> {} per-expert F32 [oracle split]",
                    "F32", name, meta.shape, n_exp
                );
                st_files[*file_idx].drop_tensor_pages(name);
                if let Some(ref mut sp) = spill {
                    maybe_spill(&mut hfq_tensors, sp, 2 * 1024 * 1024 * 1024);
                }
                continue;
            }
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            let bytes: Vec<u8> = f32_data.iter().flat_map(|&v| v.to_le_bytes()).collect();
            quantized_params += n_elements as u64;
            eprintln!(
                "  {:>8}: {} {:?} ({} elements, {:.1} KB -> {:.1} KB) [F32 oracle passthrough]",
                "F32",
                name,
                meta.shape,
                n_elements,
                raw_data.len() as f64 / 1024.0,
                bytes.len() as f64 / 1024.0
            );
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::F32,
                shape,
                group_size: 0,
                data: bytes,
                spilled_len: 0,
            });
            st_files[*file_idx].drop_tensor_pages(name);
            if let Some(ref mut sp) = spill {
                maybe_spill(&mut hfq_tensors, sp, 2 * 1024 * 1024 * 1024);
            }
            continue;
        }

        // ── LFM2.5 ingest (arch_id 11) ─────────────────────────────────────────
        // Routed experts (A1B only) → MQ4G256; expert_bias → F32; everything else
        // (conv in/out_proj, conv depthwise filter, attn q/k/v/out_proj + qk-norm,
        // dense w1/w2/w3, router gate, operator/ffn/embedding norms, tied embed/
        // lm_head) → Q8 (qt=3 Q8F16). Dense lfm2 (350M/1.2B) has no experts, so
        // every tensor takes the final Q8 path. The loader's load_f32 dequantizes
        // Q8 norms / conv-filter back to F32 on load.
        if is_lfm2moe {
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            if name.contains(".feed_forward.experts.")
                && (name.ends_with(".w1.weight")
                    || name.ends_with(".w2.weight")
                    || name.ends_with(".w3.weight"))
                && meta.shape.len() == 2
                && meta.shape[1] % 256 == 0
            {
                let f32_data = tensor_to_f32_with_optional_fp8_scale(
                    name,
                    raw_data,
                    meta,
                    &fp8_scale_for,
                    &st_files,
                );
                let signs1 = gen_fwht_signs(42, 256);
                let signs2 = gen_fwht_signs(1042, 256);
                let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
                eprintln!(
                    "  {:>8}: {} {:?} ({:.1} KB → {:.1} KB)",
                    "MQ4-LFM",
                    name,
                    meta.shape,
                    raw_data.len() as f64 / 1024.0,
                    q.len() as f64 / 1024.0
                );
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: QuantType::MQ4G256,
                    shape,
                    group_size: 256,
                    data: q,
                    spilled_len: 0,
                });
                quantized_params += (meta.shape[0] * meta.shape[1]) as u64;
                st_files[*file_idx].drop_tensor_pages(name);
                if let Some(ref mut s) = spill {
                    maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
                }
                continue;
            }
            if name.ends_with(".feed_forward.expert_bias") {
                let f32_data = tensor_to_f32_with_optional_fp8_scale(
                    name,
                    raw_data,
                    meta,
                    &fp8_scale_for,
                    &st_files,
                );
                let mut bytes = Vec::with_capacity(f32_data.len() * 4);
                for v in &f32_data {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                eprintln!(
                    "  {:>8}: {} {:?} (expert_bias F32)",
                    "F32-LFM", name, meta.shape
                );
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: QuantType::F32,
                    shape,
                    group_size: 1,
                    data: bytes,
                    spilled_len: 0,
                });
                st_files[*file_idx].drop_tensor_pages(name);
                continue;
            }
            // Dense mq4 (--format mq4): route the big 2D proj/FFN weight matrices
            // (conv in/out_proj, attn q/k/v/out_proj, dense w1/w2/w3) → MQ4G256.
            // The loader's weight_gemv / weight_gemv_residual auto-FWHT-rotate
            // MQ4G256, so no forward change is needed. Keep the tied embed/lm_head
            // (model.embed_tokens.weight), the router gate, norms, and the depthwise
            // conv filter at Q8/F32 (small + precision-sensitive). Default (no mq4
            // format) keeps the full-precision Q8 bring-up recipe.
            if use_mq4g256
                && meta.shape.len() == 2
                && meta.shape[1] % 256 == 0
                && !name.ends_with("embed_tokens.weight")
                && (name.ends_with("_proj.weight")
                    || name.ends_with(".w1.weight")
                    || name.ends_with(".w2.weight")
                    || name.ends_with(".w3.weight"))
            {
                let f32_data = tensor_to_f32_with_optional_fp8_scale(
                    name,
                    raw_data,
                    meta,
                    &fp8_scale_for,
                    &st_files,
                );
                let signs1 = gen_fwht_signs(42, 256);
                let signs2 = gen_fwht_signs(1042, 256);
                let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
                eprintln!(
                    "  {:>8}: {} {:?} ({:.1} KB → {:.1} KB)",
                    "MQ4-LFM",
                    name,
                    meta.shape,
                    raw_data.len() as f64 / 1024.0,
                    q.len() as f64 / 1024.0
                );
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: QuantType::MQ4G256,
                    shape,
                    group_size: 256,
                    data: q,
                    spilled_len: 0,
                });
                quantized_params += (meta.shape[0] * meta.shape[1]) as u64;
                st_files[*file_idx].drop_tensor_pages(name);
                if let Some(ref mut s) = spill {
                    maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
                }
                continue;
            }

            // All remaining LFM2 tensors → Q8 (qt=3). quantize_q8f16 handles any
            // 1D/2D/3D shape elementwise (conv.conv.weight is [hidden,1,K]).
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            let q = quantize_q8f16(&f32_data);
            eprintln!("  {:>8}: {} {:?} (Q8)", "Q8-LFM", name, meta.shape);
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::Q8F16,
                shape,
                group_size: 32,
                data: q,
                spilled_len: 0,
            });
            quantized_params += n_elements as u64;
            st_files[*file_idx].drop_tensor_pages(name);
            continue;
        }

        // ── Cohere2-MoE ingest (arch_id 12) ─────────────────────────────────
        // North-Mini-Code-1.0. Sweep tiers via --format: f16 (BF16-class oracle)
        // | q8 | mq6 | mq4. The EXPERTS carry the bit-width knob; attention/dense
        // stay Q8 (F16 in the oracle); the router (mlp.gate.weight) and the tied
        // embed_tokens stay Q8 (selection- / lookup-sensitive, held constant
        // across the sweep so KLD isolates expert/attention precision); all
        // *norm* tensors -> F16. Experts ship per-expert pre-split (gate_proj/
        // up_proj/down_proj); the loader byte-fuses gate_proj||up_proj.
        if is_cohere2moe {
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            if name.contains("norm") {
                let f32_data = tensor_to_f32_with_optional_fp8_scale(
                    name,
                    raw_data,
                    meta,
                    &fp8_scale_for,
                    &st_files,
                );
                let f16_bytes: Vec<u8> = f32_data
                    .iter()
                    .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                    .collect();
                eprintln!("  {:>8}: {} {:?} (norm F16)", "F16-COH", name, meta.shape);
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: QuantType::F16,
                    shape,
                    group_size: 0,
                    data: f16_bytes,
                    spilled_len: 0,
                });
                quantized_params += n_elements as u64;
                st_files[*file_idx].drop_tensor_pages(name);
                continue;
            }

            if name.ends_with("embed_tokens.weight") {
                let f32_data = tensor_to_f32_with_optional_fp8_scale(
                    name,
                    raw_data,
                    meta,
                    &fp8_scale_for,
                    &st_files,
                );
                let q = quantize_q8f16(&f32_data);
                eprintln!(
                    "  {:>8}: {} {:?} (tied embed Q8)",
                    "Q8-COH", name, meta.shape
                );
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: QuantType::Q8F16,
                    shape,
                    group_size: 32,
                    data: q,
                    spilled_len: 0,
                });
                quantized_params += n_elements as u64;
                st_files[*file_idx].drop_tensor_pages(name);
                continue;
            }

            if meta.shape.len() == 2 && meta.shape[1] % 256 == 0 {
                let is_expert = name.contains(".mlp.experts.");
                let is_router = name.ends_with(".mlp.gate.weight");
                if use_bf16 && !is_router && meta.dtype == "BF16" {
                    eprintln!(
                        "  {:>8}: {} {:?} (native bf16)",
                        "BF16-COH", name, meta.shape
                    );
                    hfq_tensors.push(HfqTensor {
                        name: name.to_string(),
                        quant_type: QuantType::BF16,
                        shape,
                        group_size: 0,
                        data: raw_data.to_vec(),
                        spilled_len: 0,
                    });
                    quantized_params += (meta.shape[0] * meta.shape[1]) as u64;
                    st_files[*file_idx].drop_tensor_pages(name);
                    if let Some(ref mut s) = spill {
                        maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
                    }
                    continue;
                }

                let dt = if is_router {
                    QuantType::Q8F16
                } else if use_f16 {
                    QuantType::F16
                } else if is_expert {
                    if use_mq6g256 {
                        QuantType::MQ6G256
                    } else if use_mq4g256 {
                        QuantType::MQ4G256
                    } else {
                        QuantType::Q8F16
                    }
                } else {
                    QuantType::Q8F16
                };
                let f32_data = tensor_to_f32_with_optional_fp8_scale(
                    name,
                    raw_data,
                    meta,
                    &fp8_scale_for,
                    &st_files,
                );
                let (data, gs, tag): (Vec<u8>, u32, &str) = match dt {
                    QuantType::F16 => (
                        f32_data
                            .iter()
                            .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                            .collect(),
                        0,
                        "F16-COH",
                    ),
                    QuantType::MQ6G256 => {
                        let s1 = gen_fwht_signs(42, 256);
                        let s2 = gen_fwht_signs(1042, 256);
                        (quantize_mq6g256(&f32_data, &s1, &s2), 256, "MQ6-COH")
                    }
                    QuantType::MQ4G256 => {
                        let s1 = gen_fwht_signs(42, 256);
                        let s2 = gen_fwht_signs(1042, 256);
                        (quantize_mq4g256(&f32_data, &s1, &s2), 256, "MQ4-COH")
                    }
                    _ => (quantize_q8f16(&f32_data), 32, "Q8-COH"),
                };
                eprintln!(
                    "  {:>8}: {} {:?} ({:.1} KB -> {:.1} KB)",
                    tag,
                    name,
                    meta.shape,
                    raw_data.len() as f64 / 1024.0,
                    data.len() as f64 / 1024.0
                );
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: dt,
                    shape,
                    group_size: gs,
                    data,
                    spilled_len: 0,
                });
                quantized_params += (meta.shape[0] * meta.shape[1]) as u64;
                st_files[*file_idx].drop_tensor_pages(name);
                if let Some(ref mut s) = spill {
                    maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
                }
                continue;
            }

            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            let q = quantize_q8f16(&f32_data);
            eprintln!("  {:>8}: {} {:?} (Q8)", "Q8-COH", name, meta.shape);
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::Q8F16,
                shape,
                group_size: 32,
                data: q,
                spilled_len: 0,
            });
            quantized_params += n_elements as u64;
            st_files[*file_idx].drop_tensor_pages(name);
            continue;
        }

        // DeepSeek V4's `tid2eid` hash-routing tables: source I64 in safetensors,
        // shape [vocab=129280, k=6]. The values are token-id × expert-id
        // pairs that all fit in i32 (vocab < 2^31, n_experts < 2^31), so
        // we downcast I64 → U32 (4 bytes/element) before write — antirez
        // does the same and the DeepSeek V4 loader at arch.rs reads them as U32
        // (`bytes.chunks_exact(4)`). Without these in the HFQ, the loader
        // sees an empty `tid2eid_host` and `ffn_hash_routed` falls back
        // to shared-only on the first `num_hash_layers` (3) layers —
        // measured 2× wikitext2 PPL regression on deepseek4-q8-mtp (21.85
        // vs 11.42 antirez) before this fix landed.
        //
        // QuantType=22 is "reserved-but-unused" in our enum (HFP4G16
        // ablation slot, never built); we use it for tid2eid storage to
        // stay byte-compatible with antirezQ8.hfq which also writes 22.
        // The loader is name-gated (looks for "tid2eid" substring), so
        // qt value doesn't actually steer dispatch — only matters for
        // cross-tooling identification.
        if meta.dtype == "I64" {
            if name.ends_with("tid2eid") {
                if n_elements * 8 != raw_data.len() {
                    panic!(
                        "tid2eid '{name}': expected {} bytes (8 × {}), got {}",
                        n_elements * 8,
                        n_elements,
                        raw_data.len()
                    );
                }
                let mut u32_bytes: Vec<u8> = Vec::with_capacity(n_elements * 4);
                for i in 0..n_elements {
                    let off = i * 8;
                    let v = i64::from_le_bytes(raw_data[off..off + 8].try_into().unwrap());
                    let v_u32 = v as u32; // downcast — values fit
                    u32_bytes.extend_from_slice(&v_u32.to_le_bytes());
                }
                let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
                eprintln!(
                    "  {:>8}: {} {:?} (I64 → U32, {} elements, {:.1} KB)",
                    "TID2EID",
                    name,
                    meta.shape,
                    n_elements,
                    u32_bytes.len() as f64 / 1024.0
                );
                quantized_params += n_elements as u64;
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: QuantType::TidI32,
                    shape,
                    group_size: 0,
                    data: u32_bytes,
                    spilled_len: 0,
                });
                st_files[*file_idx].drop_tensor_pages(name);
                continue;
            }
            // Other I64 (none expected in DeepSeek V4): skip with explicit warning.
            eprintln!(
                "  [skip-I64] {} {:?} ({} elements) — unexpected I64 tensor, not ingested",
                name, meta.shape, n_elements
            );
            skipped_params += n_elements as u64;
            continue;
        }

        // ── MiniMax-M2 router: keep Q8 ─────────────────────────────────────────
        // The MoE router (`block_sparse_moe.gate.weight`) is precision-sensitive
        // (4-bit noise flips top-k on borderline tokens) but must NOT be F16:
        // weight_gemv's F16 arm dispatches gemm_f16_batched_lmhead, which is a
        // WMMA lm-head kernel that produces garbage for the router's tiny m
        // (=n_exp). Q8 (gemv_q8_0) is well-behaved at any m and ~0.4% noise.
        if is_minimax && name.ends_with("block_sparse_moe.gate.weight") {
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            let q = quantize_q8f16(&f32_data);
            eprintln!("  {:>8}: {} {:?} (router Q8)", "Q8-MM", name, meta.shape);
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::Q8F16,
                shape,
                group_size: 32,
                data: q,
                spilled_len: 0,
            });
            st_files[*file_idx].drop_tensor_pages(name);
            continue;
        }

        // ── MiniMax-M2 per-expert pre-split path ───────────────────────────────
        // Experts ship as 2D `...block_sparse_moe.experts.E.{w1,w2,w3}.weight`
        // (F32 in the tiny oracle; FP8 e4m3 + F32 weight_scale_inv in the 229B
        // ckpt — handled transparently by tensor_to_f32_with_optional_fp8_scale).
        // Quantize each as MQ4G256 (FWHT-pre-rotated 4-bit): byte-compatible with
        // the gemv_hfq4g256_moe_* indexed kernels — passing FWHT-rotated input to
        // those kernels is mathematically equivalent to gemv_mq4g256 (the exact
        // path qwen35's MoE uses). This IS the user-facing "mq4" format. Names
        // are written verbatim; the loader fuses w1||w3 into the gate_up blob.
        if is_minimax
            && name.contains(".block_sparse_moe.experts.")
            && name.ends_with(".weight")
            && meta.shape.len() == 2
        {
            let mut f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            let k = meta.shape[1];
            let m = meta.shape[0];
            if k % 256 == 0 {
                // AWQ shared-per-layer pre-scaling of the routed experts (--awq +
                // --imatrix). w1/w3 use s_gate_up (MoE-input channels), w2 uses
                // s_down (intermediate channels). Math W·s @ x/s = W·x is exact;
                // the forward divides the activation by experts[0]'s scale.
                if awq_enabled {
                    if let (Some(layer_n), Some(gg)) =
                        (minimax_layer_index(name), imatrix_gguf.as_ref())
                    {
                        let alpha = AWQ_ALPHA.get().copied().unwrap_or(0.55);
                        let entry = mm_awq_cache
                            .entry(layer_n)
                            .or_insert_with(|| minimax_layer_awq_scales(gg, layer_n, alpha));
                        if let Some((s_gu, s_dn)) = entry.as_ref() {
                            let scale = if name.ends_with(".w2.weight") {
                                s_dn
                            } else {
                                s_gu
                            };
                            if scale.len() == k {
                                awq_pre_scale_weights(&mut f32_data, m, k, scale);
                            } else {
                                eprintln!("  minimax AWQ L{layer_n}: scale len {} != k {} ({name}); skipped", scale.len(), k);
                            }
                            if mm_awq_emitted.insert(layer_n) {
                                let p = name.split(".block_sparse_moe.").next().unwrap();
                                hfq_tensors.push(HfqTensor {
                                    name: format!("{p}.block_sparse_moe.awq_scale_gate_up.weight"),
                                    quant_type: QuantType::F16,
                                    shape: vec![s_gu.len() as u32],
                                    group_size: 0,
                                    data: awq_scales_to_f16_bytes(s_gu),
                                    spilled_len: 0,
                                });
                                hfq_tensors.push(HfqTensor {
                                    name: format!("{p}.block_sparse_moe.awq_scale_down.weight"),
                                    quant_type: QuantType::F16,
                                    shape: vec![s_dn.len() as u32],
                                    group_size: 0,
                                    data: awq_scales_to_f16_bytes(s_dn),
                                    spilled_len: 0,
                                });
                                eprintln!("  AWQ-MM: emitted shared expert scales for L{layer_n}");
                            }
                        }
                    }
                }
                let signs1 = gen_fwht_signs(42, 256);
                let signs2 = gen_fwht_signs(1042, 256);
                // Expert format by --format: mq2-lloyd (MQ2G256Lloyd, hipx sub-4-bit
                // target — has deepseek4 indexed-MoE kernels), mq3-lloyd / mq6 (oracle
                // check / HIPFIRE_MINIMAX_EXPERT_*), else mq4 (MQ4G256, default + validated).
                let mm_mq6 = use_mq6g256
                    || hipfire_config::developer_var_os("HIPFIRE_MINIMAX_EXPERT_MQ6").is_some();
                let mm_mq2l = use_mq2g256_lloyd
                    || hipfire_config::developer_var_os("HIPFIRE_MINIMAX_EXPERT_MQ2L").is_some();
                let mm_mq3l = use_mq3g256_lloyd
                    || hipfire_config::developer_var_os("HIPFIRE_MINIMAX_EXPERT_MQ3L").is_some();
                // Per-layer mixed-precision promotion. HIPFIRE_MINIMAX_PROMOTE_MQ4 /
                // _MQ6 hold comma-separated layer ranges ("12-45,50") whose experts are
                // forced UP to MQ4 / MQ6 regardless of the base --format. The forward
                // dispatches expert dtype per-layer (experts[0].gpu_dtype), so the model
                // carries an MQ2-Lloyd base with MQ4 on the quant-sensitive middle layers.
                let mm_layer = minimax_layer_index(name);
                let promote_mq6 = mm_layer.map_or(false, |l| {
                    minimax_layer_in_config_set("HIPFIRE_MINIMAX_PROMOTE_MQ6", l)
                });
                let promote_mq4 = mm_layer.map_or(false, |l| {
                    minimax_layer_in_config_set("HIPFIRE_MINIMAX_PROMOTE_MQ4", l)
                });
                let (q, qt, label) = if promote_mq6 {
                    (
                        quantize_mq6g256(&f32_data, &signs1, &signs2),
                        QuantType::MQ6G256,
                        "MQ6-PROMO",
                    )
                } else if promote_mq4 {
                    (
                        quantize_mq4g256(&f32_data, &signs1, &signs2),
                        QuantType::MQ4G256,
                        "MQ4-PROMO",
                    )
                } else if mm_mq3l {
                    (
                        quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2),
                        QuantType::MQ3G256Lloyd,
                        "MQ3L-MM",
                    )
                } else if mm_mq2l {
                    (
                        quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2),
                        QuantType::MQ2G256Lloyd,
                        "MQ2L-MM",
                    )
                } else if mm_mq6 {
                    (
                        quantize_mq6g256(&f32_data, &signs1, &signs2),
                        QuantType::MQ6G256,
                        "MQ6-MM",
                    )
                } else {
                    (
                        quantize_mq4g256(&f32_data, &signs1, &signs2),
                        QuantType::MQ4G256,
                        "MQ4-MM",
                    )
                };
                let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
                eprintln!(
                    "  {label:>8}: {} {:?} ({:.1} KB → {:.1} KB)",
                    name,
                    meta.shape,
                    raw_data.len() as f64 / 1024.0,
                    q.len() as f64 / 1024.0
                );
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: qt,
                    shape,
                    group_size: 256,
                    data: q,
                    spilled_len: 0,
                });
                quantized_params += (meta.shape[0] * meta.shape[1]) as u64;
                st_files[*file_idx].drop_tensor_pages(name);
                if let Some(ref mut s) = spill {
                    maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
                }
                continue;
            }
            // k not %256 → fall through to standard path (real MiniMax inter=1536,
            // hidden=3072 are both %256, so this only guards degenerate tinies).
        }

        // ── MoE 3D-stacked expert tensor split ─────────────────────────────────
        // Qwen3.5-MoE stores routed experts as 3D tensors:
        //   model.language_model.layers.{N}.mlp.experts.gate_up_proj
        //     shape: [num_experts, 2 * moe_intermediate, hidden_size]
        //   model.language_model.layers.{N}.mlp.experts.down_proj
        //     shape: [num_experts, hidden_size, moe_intermediate]
        // Note: no `.weight` suffix on these, so should_quantize() returns false
        // and the standard path would store them as F16 — defeating the purpose.
        // We split into per-expert 2D MQ4G256 quantized tensors named
        //   model.language_model.layers.{N}.mlp.experts.{X}.{base}.weight
        // so the engine loader can fish them out by expert index.
        // ── DeepSeek V4 per-expert tensor path ─────────────────────────────────────
        // DeepSeek V4 ships per-expert 2D tensors at `layers.L.ffn.experts.E.{w1,w2,w3}.weight`.
        // (Not 3D-stacked like Qwen3.5 MoE.) Route them through the MQ-family
        // quant path directly. No imatrix yet for DeepSeek V4 — pass unit column
        // weights so the underlying Lloyd codebook fit is uniform; the
        // GPTQ sequential error-feedback assignment still applies and is
        // worth +1-2 % coherence (project_gptq_lloyd_mq2_win.md).
        if is_deepseek4
            && name.contains(".ffn.experts.")
            && name.ends_with(".weight")
            && meta.shape.len() == 2
        {
            // DeepSeek V4 routed experts are FP4 (E2M1) per upstream `inference/
            // model.py:132-137` and config `expert_dtype:"fp4"`. Safetensors
            // shape is [out, in/2] with each byte packing two nibbles; the
            // paired scale tensor is `<name>.scale` UE8M0 with block size 32
            // along logical K.
            //
            // The outer condition `name.contains(".ffn.experts.")` already
            // excludes shared_experts (which use the non-routed `.shared_
            // experts.` infix). So everything reaching here is a routed
            // expert → unconditionally FP4 unpack. Logical K dim doubles.
            let name_owned = name.to_string();
            let (f32_data, logical_shape) = if (meta.dtype == "I8" || meta.dtype == "F8_E4M3")
                && fp8_scale_for.contains_key(&name_owned)
            {
                let (sfi, sname) = &fp8_scale_for[&name_owned];
                let (smeta, sbytes) = st_files[*sfi]
                    .tensor_data(sname)
                    .unwrap_or_else(|| panic!("FP scale tensor missing: {sname}"));
                dequantize_e2m1_ue8m0_to_f32(raw_data, &meta.shape, sbytes, &smeta.shape)
            } else {
                let vals = tensor_to_f32_with_optional_fp8_scale(
                    name,
                    raw_data,
                    meta,
                    &fp8_scale_for,
                    &st_files,
                );
                (vals, meta.shape.clone())
            };
            let k = logical_shape[1];
            if k % 256 == 0
                && (use_mq4_mq2lloyd_gptq_all
                    || use_mq4_mqlloyd_antirez_gptq
                    || use_mq4_mq2lloyd_native
                    || use_mq4_mq2lloyd_imatrix
                    || use_mq4_mqlloyd_antirez
                    || use_deepseek4_source_precision)
            {
                let signs1 = gen_fwht_signs(42, 256);
                let signs2 = gen_fwht_signs(1042, 256);
                let unit_col_weights: Vec<f32> = vec![1.0; k];
                let (q, expert_qt, expert_label): (Vec<u8>, QuantType, &str) =
                    if use_deepseek4_mq4_experts {
                        (
                            quantize_mq4g256_lloyd(&f32_data, &signs1, &signs2),
                            QuantType::MQ4G256Lloyd,
                            "MQ4L-DeepSeek V4",
                        )
                    } else if use_deepseek4_mq3_experts {
                        (
                            quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2),
                            QuantType::MQ3G256Lloyd,
                            "MQ3L-DeepSeek V4",
                        )
                    } else if use_mq4_mq2lloyd_gptq_all || use_mq4_mqlloyd_antirez_gptq {
                        (
                            quantize_mq2g256_lloyd_gptq(
                                &f32_data,
                                &unit_col_weights,
                                &signs1,
                                &signs2,
                            ),
                            QuantType::MQ2G256Lloyd,
                            "MQ2L-DeepSeek V4",
                        )
                    } else {
                        (
                            quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2),
                            QuantType::MQ2G256Lloyd,
                            "MQ2L-DeepSeek V4",
                        )
                    };
                let shape: Vec<u32> = logical_shape.iter().map(|&s| s as u32).collect();
                eprintln!(
                    "  {:>8}: {} storage{:?} → logical{:?} ({:.1} KB → {:.1} KB)",
                    expert_label,
                    name,
                    meta.shape,
                    logical_shape,
                    raw_data.len() as f64 / 1024.0,
                    q.len() as f64 / 1024.0
                );
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: expert_qt,
                    shape,
                    group_size: 256,
                    data: q,
                    spilled_len: 0,
                });
                quantized_params += (logical_shape[0] * logical_shape[1]) as u64;
                st_files[*file_idx].drop_tensor_pages(name);
                if let Some(ref mut s) = spill {
                    maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
                }
                continue;
            }
            // Fall through to standard path for non-MQ2 formats.
        }

        if is_moe
            && name.contains("mlp.experts.")
            && (name.ends_with("gate_up_proj") || name.ends_with("down_proj"))
            && meta.shape.len() == 3
        {
            let n_experts = meta.shape[0];
            let inner_n: usize = meta.shape[1..].iter().product();
            let elem_size = match meta.dtype.as_str() {
                "F32" => 4,
                "F16" | "BF16" => 2,
                other => panic!("unsupported expert tensor dtype: {other}"),
            };
            let inner_bytes = inner_n * elem_size;
            let inner_shape: Vec<u32> = meta.shape[1..].iter().map(|&s| s as u32).collect();
            let base_name = if name.ends_with("gate_up_proj") {
                "gate_up_proj"
            } else {
                "down_proj"
            };
            // Strip the trailing base; what remains is the parent path with `experts.` already on the end
            let parent = &name[..name.len() - base_name.len()];

            // Inner quantization for experts — respects --format flag.
            // MQ6 reduces quantization error that compounds across 48 MoE
            // layers × 9 expert contributions per layer at the cost of ~50%
            // more VRAM per expert. MQ4 is the default for VRAM efficiency.
            let signs1 = gen_fwht_signs(42, 256);
            let signs2 = gen_fwht_signs(1042, 256);
            let inner_k = inner_shape[1] as usize;
            let supports_g256 = inner_k % 256 == 0;
            // K-map: check the parent tensor name directly. The parent
            // (e.g. "...mlp.experts.gate_up_proj") contains "mlp.experts."
            // so kmap_resolve rule 4 matches it. The kmap HashMap was built
            // from all_tensors which has these parent names as keys.
            let kmap_promote = kmap.get(*name) == Some(&QuantLevel::Promote6);
            // Phase 5 tiering decision needs the layer index for this parent.
            // Computed once here and reused by both expert_mq2lloyd_native
            // and expert_mq3lloyd_native below.
            let parent_layer: Option<usize> = {
                let marker = ".layers.";
                parent.rfind(marker).and_then(|i| {
                    let rest = &parent[i + marker.len()..];
                    rest.split('.').next().and_then(|s| s.parse().ok())
                })
            };
            let tiered_layer_is_mq3 = use_mq4_mqlloyd_tiered
                && !kmap_promote
                && parent_layer
                    .map(|l| mq3_tier_layers.contains(&l))
                    .unwrap_or(false);
            let tiered_layer_is_mq2 = use_mq4_mqlloyd_tiered
                && !kmap_promote
                && parent_layer
                    .map(|l| !mq3_tier_layers.contains(&l))
                    .unwrap_or(false);
            // Antirez-style: gate_up → MQ2, down → MQ3 (kmap-respecting).
            // Selects based on `base_name` ("gate_up_proj" vs "down_proj").
            let is_gate_up = base_name == "gate_up_proj";
            let antirez_mq3 = (use_mq4_mqlloyd_antirez || use_mq4_mqlloyd_antirez_gptq)
                && !kmap_promote
                && !is_gate_up;
            let antirez_mq2 = (use_mq4_mqlloyd_antirez || use_mq4_mqlloyd_antirez_gptq)
                && !kmap_promote
                && is_gate_up;
            // Lever 2: GPTQ-style sequential Lloyd specifically for the
            // gate_up MQ2 path. Sets a flag the inner quant dispatch will
            // honor (separate from the imatrix-only path).
            let use_gptq_for_gate_up = use_mq4_mqlloyd_antirez_gptq && antirez_mq2;
            // For the kmap-respecting MQ2-Lloyd variants, kmap_promote experts
            // get MQ6 instead of MQ2-Lloyd. Falls through to expert_mq6 below.
            let expert_mq6 = (use_mq6g256
                || use_mq4_mq6exp
                || (kmap_promote && use_mq4g256)
                || (kmap_promote && use_mq4_mq2lloyd_kmap)
                || (kmap_promote && use_mq4_mq2lloyd_imatrix)
                || (kmap_promote && use_mq4_mq2lloyd_gptq_all)
                || (kmap_promote && use_mq4_mq3lloyd_kmap))
                && supports_g256;
            // MQ5 routed experts: `--format mq5` ships ALL experts at MQ5
            // (mirrors expert_mq6's use_mq6g256 base-format case). The env-var
            // levers (HIPFIRE_MOE_EXPERTS_MQ5 / _DOWN_MQ5) below add the
            // gate_up-stays-MQ4 + down-only-MQ5 recipe via `down_mq5`.
            let expert_mq5 = use_mq5g256 && supports_g256;
            let expert_hfq6 = (use_hfq6 || (kmap_promote && use_hfq4g256)) && supports_g256;
            let expert_hfq4 = use_hfq4g256 && !kmap_promote && supports_g256;
            // HIPFIRE_MOE_DOWN_MQ6=1: promote ONLY the expert down_proj to MQ6
            // (gate_up stays MQ4) — the "mq6-down" precision lever, composable
            // with down-AWQ. Kept OUT of `expert_mq6` so `expert_awq_active` still
            // fires; the AWQ branch below switches its output format to MQ6.
            // HIPFIRE_MOE_EXPERTS_MQ6=1 promotes BOTH gate_up + down to MQ6 (the
            // experts-level "+P" / kmap-experts recipe, minus the gfx12-only dense
            // attn promotion). HIPFIRE_MOE_DOWN_MQ6=1 promotes only down. `down_mq6`
            // means "promote THIS expert tensor to MQ6" (gate_up or down).
            let experts_mq6_all = hipfire_config::developer_var("HIPFIRE_MOE_EXPERTS_MQ6")
                .ok()
                .as_deref()
                == Some("1");
            let down_mq6 = supports_g256
                && (experts_mq6_all
                    || (hipfire_config::developer_var("HIPFIRE_MOE_DOWN_MQ6")
                        .ok()
                        .as_deref()
                        == Some("1")
                        && base_name == "down_proj"));
            // HIPFIRE_MOE_EXPERTS_MQ5=1 promotes BOTH gate_up + down to MQ5; the
            // experts-level 5-bit recipe (5.25 bpw, between MQ4 and MQ6).
            // HIPFIRE_MOE_DOWN_MQ5=1 promotes ONLY the expert down_proj to MQ5
            // (gate_up stays MQ4). Kept OUT of `expert_mq5` so `expert_awq_active`
            // still fires; the AWQ branch switches its output format to MQ5.
            let experts_mq5_all = hipfire_config::developer_var("HIPFIRE_MOE_EXPERTS_MQ5")
                .ok()
                .as_deref()
                == Some("1");
            let down_mq5 = supports_g256
                && (experts_mq5_all
                    || (hipfire_config::developer_var("HIPFIRE_MOE_DOWN_MQ5")
                        .ok()
                        .as_deref()
                        == Some("1")
                        && base_name == "down_proj"));
            // mq4-mq2lloydexp round-trip probe: ALWAYS hits routed experts
            // (overrides any kmap promotion). The intent is to inject MQ2
            // noise specifically on the routed-expert tensors, so even
            // K-map "Promote6" experts get the MQ2-Lloyd round-trip here.
            let expert_mq2lloyd_roundtrip = use_mq4_mq2lloydexp && supports_g256;
            // GL twin — same "always hits routed experts" intent as above.
            let expert_mq2gl_roundtrip = use_mq4_mq2glexp && supports_g256;
            // Native MQ2-Lloyd: ship qt=19 bytes directly, no round-trip.
            // Requires runtime support for DType::MQ2G256Lloyd on experts.
            // For -native (no kmap respect): always MQ2-Lloyd on every expert.
            // For -kmap / -imatrix (kmap respect): only non-promoted experts
            // go MQ2-Lloyd; promoted ones hit `expert_mq6` above.
            // All-MQ2-GPTQ test: ALL routed experts at MQ2-Lloyd, both
            // gate_up and down. Respects kmap_promote (promoted layers
            // still get MQ6). Uses sequential-GPTQ Lloyd everywhere via
            // the `use_gptq_for_all_mq2` flag below.
            let all_mq2_gptq = use_mq4_mq2lloyd_gptq_all && !kmap_promote;
            let expert_mq2lloyd_native = (use_mq4_mq2lloyd_native
                || (use_mq4_mq2lloyd_kmap && !kmap_promote)
                || (use_mq4_mq2lloyd_imatrix && !kmap_promote)
                || tiered_layer_is_mq2
                || antirez_mq2
                || all_mq2_gptq)
                && supports_g256;
            // GPTQ assignment fires for both gate_up and down when in
            // all-MQ2-GPTQ mode (not just gate_up like the antirez split).
            let use_gptq_for_gate_up =
                use_gptq_for_gate_up || (all_mq2_gptq && imatrix_path.is_some());
            // MQ3-Lloyd asymmetric: non-promoted experts → qt=20 (3.5 bpw).
            // Promoted ones hit `expert_mq6` above (note: kmap_promote already
            // includes use_mq4_mq3lloyd_kmap via the expert_mq6 expression).
            //
            // Phase 5 tiered variant: also MQ3-Lloyd on hot non-promoted
            // layers (the ones in `mq3_tier_layers`, decided above by imatrix
            // .counts ranking).
            let expert_mq3lloyd_native =
                ((use_mq4_mq3lloyd_kmap && !kmap_promote) || tiered_layer_is_mq3 || antirez_mq3)
                    && supports_g256;
            // Per-expert column-weights from the imatrix file, used only by
            // the imatrix variant. Built once per parent (cheap), then sliced
            // per expert inside the rayon loop. Falls back to None when the
            // imatrix tensor for this parent isn't found (e.g. a non-expert
            // tensor we accidentally route here, or a layer that wasn't in
            // the calibration set).
            let imatrix_lookup_name = format!("{}{}", parent, base_name);
            let imatrix_per_expert: Option<Vec<Vec<f32>>> = if (use_mq4_mq2lloyd_imatrix
                || use_mq4_mqlloyd_antirez
                || use_mq4_mqlloyd_antirez_gptq
                || use_mq4_mq2lloyd_gptq_all)
                && imatrix_gguf.is_some()
                && expert_mq2lloyd_native
            {
                imatrix_col_weights_for_parent(
                    imatrix_gguf.as_ref().unwrap(),
                    &imatrix_lookup_name,
                    n_experts,
                )
            } else {
                None
            };
            if use_mq4_mq2lloyd_imatrix && expert_mq2lloyd_native && imatrix_per_expert.is_none() {
                eprintln!(
                    "  imatrix: no entry for {} → falling back to uniform Lloyd",
                    imatrix_lookup_name
                );
            }

            // ── SP4b bake prune (3D-stacked experts) ───────────────────────────
            // Qwen3.5-MoE (and any 3D-stacked MoE) ships routed experts as one
            // `[n_experts, ...]` tensor that this branch splits per-expert. Under
            // an active bake keep, emit ONLY kept slices, renumbered to compact
            // slots: `slots[slot] = orig_expert`. The slice offset + imatrix
            // lookup key off `orig`; the output name uses `slot`. No keep ⇒
            // identity (`slots[i] = (i, i)`), byte-identical to baseline.
            let bake_slots: Vec<(usize, usize)> = if bake_keep_active {
                let plan = reap_bake_plan.as_ref().unwrap();
                let keep = plan.keep.as_ref().unwrap();
                let l = parent_layer.unwrap_or_else(|| {
                    eprintln!(
                        "reap bake: 3D-stacked expert tensor '{name}' has no parseable layer"
                    );
                    std::process::exit(2);
                });
                if l >= keep.len() {
                    eprintln!(
                        "reap bake: layer {l} for stacked experts '{name}' out of keep-map range"
                    );
                    std::process::exit(2);
                }
                keep[l]
                    .iter()
                    .enumerate()
                    .map(|(slot, &orig)| (slot, orig as usize))
                    .collect()
            } else {
                (0..n_experts).map(|i| (i, i)).collect()
            };
            let n_out_experts = bake_slots.len();

            // ── Per-expert AWQ (Route A) ──────────────────────────────────────
            // When `--awq` is active with a GGUF imatrix, MQ4 experts get
            // activation-aware per-expert pre-scaling + a per-expert
            // `.awq_scale.weight` sidecar (length K). The runtime divides x by
            // the per-expert scale inside the indexed/grouped expert GEMM. Takes
            // priority over plain MQ4G256; the Lloyd branches above are mutually
            // exclusive (selected by their own flags), so AWQ only fires when
            // none of them claimed this expert.
            // HIPFIRE_AWQ_EXPERTS=down restricts expert AWQ to down_proj (the
            // sensitive residual-write projection + the free runtime kernel);
            // unset/=all does both gate_up and down (default).
            let awq_down_only = hipfire_config::developer_var("HIPFIRE_AWQ_EXPERTS")
                .ok()
                .as_deref()
                == Some("down");
            // HIPFIRE_AWQ_EXPERTS=none keeps DENSE AWQ (attn/lm_head) but emits
            // NO per-expert AWQ — the clean baseline for isolating the expert
            // contribution against an HIPFIRE_AWQ_EXPERTS=down treatment.
            let awq_experts_none = hipfire_config::developer_var("HIPFIRE_AWQ_EXPERTS")
                .ok()
                .as_deref()
                == Some("none");
            let expert_awq_active = AWQ_ALPHA.get().is_some()
                && !awq_experts_none
                && imatrix_gguf.is_some()
                && supports_g256
                && !(awq_down_only && base_name == "gate_up_proj")
                && !expert_mq3lloyd_native
                && !expert_mq2lloyd_native
                && !expert_mq2lloyd_roundtrip
                && !expert_mq6
                && !expert_hfq6
                && !expert_hfq4;
            let awq_in_sum2_per_expert: Option<Vec<Vec<f32>>> = if expert_awq_active {
                imatrix_in_sum2_for_parent(
                    imatrix_gguf.as_ref().unwrap(),
                    &imatrix_lookup_name,
                    n_experts,
                )
            } else {
                None
            };
            let awq_alpha_e = AWQ_ALPHA.get().copied().unwrap_or(0.5);
            let inner_m = inner_shape[0] as usize; // out features
            let inner_k_e = inner_shape[1] as usize; // in features (K, awq scale length)
            if expert_awq_active && awq_in_sum2_per_expert.is_none() {
                eprintln!(
                    "  imatrix(awq): no entry for {} → plain MQ4G256 experts (no AWQ)",
                    imatrix_lookup_name
                );
            }

            // ── Graded mixed-precision hot-set (HIPFIRE_MOE_GRADED) ───────────
            // Rank this parent's experts by imatrix routing count; the top
            // `moe_hot_frac` (DESC) get MQ6, the rest MQ2-Lloyd. Mirrors the
            // per-layer tier formula (n_hot = round(frac*n), sort DESC, take
            // top-n) but applied PER-PARENT over experts. Read-only; captured
            // by reference into the rayon closure below.
            // De-risk (Verify): the runtime wires the merged dtype-tag kernel
            // for the DOWN projection only — gate_up stays uniform MQ4. Grade
            // ONLY down_proj so the emitted file matches the wired decode path
            // (mixed MQ6/MQ2-Lloyd down, uniform MQ4 gate_up). Grading gate_up
            // would emit mixed bytes the single-dtype gate_up GEMV cannot read,
            // producing NaN logits.
            let graded_hot: Option<std::collections::HashSet<usize>> =
                if use_moe_graded && base_name == "down_proj" {
                    let counts = imatrix_gguf.as_ref().and_then(|g| {
                        imatrix_expert_counts_for_parent(g, &imatrix_lookup_name, n_experts)
                    });
                    match counts {
                        Some(c) => {
                            let mut ranked: Vec<(usize, f32)> = c
                                .iter()
                                .enumerate()
                                .filter(|(_, v)| v.is_finite())
                                .map(|(e, &v)| (e, v))
                                .collect();
                            ranked.sort_by(|a, b| {
                                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let n_hot = ((n_experts as f64) * moe_hot_frac).round() as usize;
                            let n_hot = n_hot.min(ranked.len());
                            let set: std::collections::HashSet<usize> =
                                ranked.iter().take(n_hot).map(|&(e, _)| e).collect();
                            eprintln!(
                                "  Graded {}{}: {} hot (MQ6) / {} cold (MQ2-Lloyd) experts",
                                parent,
                                base_name,
                                set.len(),
                                n_experts - set.len()
                            );
                            Some(set)
                        }
                        None => {
                            eprintln!(
                                "  Graded: no imatrix .counts for {} → ALL experts MQ2-Lloyd",
                                imatrix_lookup_name
                            );
                            Some(std::collections::HashSet::new())
                        }
                    }
                } else {
                    None
                };

            // Parallelize across the expert slices via rayon. Each slice
            // dequant→FWHT→quant→pack is a CPU-bound, self-contained job.
            // The outer Rayon pool size is set in main() before this runs.
            use rayon::prelude::*;
            let dtype = meta.dtype.clone();
            let parent_owned = parent.to_string();
            let inner_shape_clone = inner_shape.clone();
            let base_owned = base_name.to_string();
            // GPTQ-E8: borrow the Hessian dir into the rayon closure. Each
            // expert reads its own per-(tensor,expert) 256-block file; missing
            // -> RTN fallback. None unless --format mfp{2,3,4}e8-gptq + --hessian-dir.
            let hessian_dir_ref: Option<&Path> =
                if use_gptq_e8 || use_gptq_mfp3e8 || use_gptq_mfp2e8 {
                    hessian_dir.as_deref()
                } else {
                    None
                };
            let new_pairs: Vec<(HfqTensor, Option<HfqTensor>)> = bake_slots
                .into_par_iter()
                .map(|(slot, x)| {
                    let slice_off = x * inner_bytes;
                    let slice = &raw_data[slice_off..slice_off + inner_bytes];
                    let f32_slice = to_f32(slice, &dtype);
                    // Per-expert AWQ override (Route A): when this expert has a
                    // raw in_sum2 row, pre-scale W·s and remember s for the
                    // sidecar. Falls through to the format branches otherwise.
                    let awq_scales: Option<Vec<f32>> = awq_in_sum2_per_expert
                        .as_ref()
                        .and_then(|t| t.get(x))
                        .filter(|v| v.len() == inner_k_e)
                        .map(|v| compute_awq_scales(v, awq_alpha_e));
                    let (quantized, qt, gs) = if let (Some(tm), Some(lay)) =
                        (moe_tier_map.as_ref(), parent_layer)
                    {
                        // N-tier TIER_MAP dispatch: look up (layer, expert) -> QuantType.
                        // Fires for BOTH gate_up and down (unlike graded_hot which is down-only).
                        // Falls back to uniform MQ4 for unmapped (layer,expert) pairs.
                        // Uses the outer-scope signs1/signs2 captured by the rayon closure.
                        match tm.get(&(lay, x)).copied().unwrap_or(QuantType::MQ4G256) {
                            QuantType::MQ6G256 => (
                                quantize_mq6g256(&f32_slice, &signs1, &signs2),
                                QuantType::MQ6G256,
                                256u32,
                            ),
                            QuantType::MQ4G256 => (
                                quantize_mq4g256(&f32_slice, &signs1, &signs2),
                                QuantType::MQ4G256,
                                256u32,
                            ),
                            QuantType::MQ3G256Lloyd => (
                                quantize_mq3g256_lloyd(&f32_slice, &signs1, &signs2),
                                QuantType::MQ3G256Lloyd,
                                256u32,
                            ),
                            QuantType::MQ2G256Lloyd => (
                                quantize_mq2g256_lloyd(&f32_slice, &signs1, &signs2),
                                QuantType::MQ2G256Lloyd,
                                256u32,
                            ),
                            // GL = GLOBAL codebook: one codebook for the whole tensor
                            // (GL_CB2/GL_CB3, passed to the kernel as scalar args) plus a
                            // per-block fp16 scale, in a two-region SoA blob. Saves the
                            // 0.1875 bpw the per-block fp16 codebook costs — measured at
                            // +1.16% KLD and -0.08% decode, i.e. size for free.
                            //
                            // NOTE these take the 2D (m, k) form like the E8 encoders, NOT
                            // the flat form the Lloyd ones use — the SoA layout needs the
                            // row count to place the scale region.
                            QuantType::MQ2G256GL => (
                                quantize_mq2g256gl(&f32_slice, inner_m, inner_k_e, &signs1, &signs2),
                                QuantType::MQ2G256GL,
                                256u32,
                            ),
                            QuantType::MQ3G256GL => (
                                quantize_mq3g256gl(&f32_slice, inner_m, inner_k_e, &signs1, &signs2),
                                QuantType::MQ3G256GL,
                                256u32,
                            ),
                            // T3-3L-E8 experiment: mfp4-E8 mid tier (4.25 bpw,
                            // MQ6-class quality) in place of MQ4. group_size 32.
                            QuantType::MFP4G32E8 => (
                                quantize_mfp4g32_e8_2d(
                                    &f32_slice, inner_m, inner_k_e, &signs1, &signs2,
                                ),
                                QuantType::MFP4G32E8,
                                32u32,
                            ),
                            // [NaN-CRITICAL] mfp3-E8 cold tier: 3-bit lattice, 13 B/blk, 3.25 bpw.
                            // Drop-in for MQ3G256Lloyd (tag 3 → tag 5 in the kernel tag table).
                            QuantType::MFP3G32E8 => {
                                // GPTQ/LDLQ when a Hessian is available (graded cold
                                // tier), else RTN. Same per-tensor key + fallback
                                // accounting as the uniform mfp3e8-gptq path.
                                // Use the RAW --hessian-dir (not the format-gated
                                // hessian_dir_ref): graded base --format is mq4, so
                                // the gptq-format flags are off, but a passed Hessian
                                // still means "GPTQ the E8 cold tier".
                                let q = if let Some(hdir) = hessian_dir.as_deref() {
                                    let tname = format!("{parent_owned}{x}.{base_owned}.weight");
                                    let hblk = load_hessian_blocks(hdir, &tname);
                                    if hblk.is_empty() {
                                        GPTQ_E8_FALLBACK
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    } else {
                                        GPTQ_E8_FIRED
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    }
                                    quantize_mfp3g32_e8_gptq_2d(
                                        &f32_slice, inner_m, inner_k_e, &signs1, &signs2, &hblk,
                                    )
                                } else {
                                    quantize_mfp3g32_e8_2d(
                                        &f32_slice, inner_m, inner_k_e, &signs1, &signs2,
                                    )
                                };
                                (q, QuantType::MFP3G32E8, 32u32)
                            }
                            // [NaN-CRITICAL] mfp2-E8 cold tier: 2-bit lattice, 9 B/blk, 2.25 bpw.
                            // Drop-in for MQ2G256Lloyd (tag 1 → tag 6 in the kernel tag table).
                            QuantType::MFP2G32E8 => {
                                // GPTQ/LDLQ when a Hessian is available (graded cold
                                // tier), else RTN. Raw --hessian-dir (see MFP3 arm).
                                let q = if let Some(hdir) = hessian_dir.as_deref() {
                                    let tname = format!("{parent_owned}{x}.{base_owned}.weight");
                                    let hblk = load_hessian_blocks(hdir, &tname);
                                    if hblk.is_empty() {
                                        GPTQ_E8_FALLBACK
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    } else {
                                        GPTQ_E8_FIRED
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    }
                                    quantize_mfp2g32_e8_gptq_2d(
                                        &f32_slice, inner_m, inner_k_e, &signs1, &signs2, &hblk,
                                    )
                                } else {
                                    quantize_mfp2g32_e8_2d(
                                        &f32_slice, inner_m, inner_k_e, &signs1, &signs2,
                                    )
                                };
                                (q, QuantType::MFP2G32E8, 32u32)
                            }
                            // Any other QuantType in the map → MQ4 safe fallback
                            _ => (
                                quantize_mq4g256(&f32_slice, &signs1, &signs2),
                                QuantType::MQ4G256,
                                256u32,
                            ),
                        }
                    } else if let Some(hot) = graded_hot.as_ref() {
                        // Graded mixed precision: hot expert -> MQ6, cold ->
                        // MQ2-Lloyd. Each expert's HfqTensor carries its own qt
                        // so this single parent emits MIXED dtypes; the runtime
                        // builds the per-expert dtype-tag table from gpu_dtype.
                        if hot.contains(&x) {
                            (
                                quantize_mq6g256(&f32_slice, &signs1, &signs2),
                                QuantType::MQ6G256,
                                256u32,
                            )
                        } else {
                            (
                                quantize_mq2g256_lloyd(&f32_slice, &signs1, &signs2),
                                QuantType::MQ2G256Lloyd,
                                256u32,
                            )
                        }
                    } else if let Some(scales) = awq_scales.as_ref() {
                        let mut scaled = f32_slice.clone();
                        awq_pre_scale_weights(&mut scaled, inner_m, inner_k_e, scales);
                        if down_mq6 {
                            (
                                quantize_mq6g256(&scaled, &signs1, &signs2),
                                QuantType::MQ6G256,
                                256u32,
                            )
                        } else if down_mq5 || expert_mq5 {
                            (
                                quantize_mq5g256(&scaled, &signs1, &signs2),
                                QuantType::MQ5G256,
                                256u32,
                            )
                        } else {
                            (
                                quantize_mq4g256(&scaled, &signs1, &signs2),
                                QuantType::MQ4G256,
                                256u32,
                            )
                        }
                    } else if expert_mq3lloyd_native && routed_gl {
                        // GL swap: same 3-bit allocation, global codebook instead of
                        // a per-block fp16 one. 3.0625 vs 3.5 bpw.
                        let q = quantize_mq3g256gl(&f32_slice, inner_m, inner_k_e, &signs1, &signs2);
                        (q, QuantType::MQ3G256GL, 256u32)
                    } else if expert_mq3lloyd_native {
                        let q = quantize_mq3g256_lloyd(&f32_slice, &signs1, &signs2);
                        (q, QuantType::MQ3G256Lloyd, 256u32)
                    } else if expert_mq2lloyd_native && routed_gl {
                        // GL swap: 2.0625 vs 2.25 bpw. NOTE the imatrix-weighted and
                        // GPTQ arms below are DELIBERATELY not mirrored here — the
                        // weighted fit is provably inert after the FWHT (every
                        // R[i][j]^2 = 1/256, so a rotated diagonal importance vector
                        // is constant), so plain Lloyd is the honest baseline and
                        // there is nothing to lose by taking it.
                        let q = quantize_mq2g256gl(&f32_slice, inner_m, inner_k_e, &signs1, &signs2);
                        (q, QuantType::MQ2G256GL, 256u32)
                    } else if expert_mq2lloyd_native {
                        // Native MQ2-Lloyd: ship qt=19 bytes (72 B / 256 weights).
                        // Selection order:
                        //   1. GPTQ-Lloyd (sequential error feedback) — Lever 2
                        //      path, requires imatrix.
                        //   2. Imatrix-weighted Lloyd — standard Phase 3b path.
                        //   3. Uniform Lloyd — fallback when no imatrix available.
                        let q = match imatrix_per_expert.as_ref() {
                            Some(table)
                                if x < table.len()
                                    && !table[x].is_empty()
                                    && use_gptq_for_gate_up =>
                            {
                                quantize_mq2g256_lloyd_gptq(&f32_slice, &table[x], &signs1, &signs2)
                            }
                            Some(table) if x < table.len() && !table[x].is_empty() => {
                                quantize_mq2g256_lloyd_weighted(
                                    &f32_slice, &table[x], &signs1, &signs2,
                                )
                            }
                            _ => quantize_mq2g256_lloyd(&f32_slice, &signs1, &signs2),
                        };
                        (q, QuantType::MQ2G256Lloyd, 256u32)
                    } else if expert_mq2lloyd_roundtrip {
                        // MQ2-Lloyd → F32 → HFQ4 round-trip. The MQ2 step injects
                        // the 2-bit Lloyd-codebook noise; the HFQ4 step re-packs
                        // for runtime. Final on-disk format is HFQ4G256, no
                        // engine changes required.
                        let mq2_bytes = quantize_mq2g256_lloyd(&f32_slice, &signs1, &signs2);
                        let dequant = dequantize_mq2g256_lloyd_to_f32(
                            &mq2_bytes,
                            f32_slice.len(),
                            &signs1,
                            &signs2,
                        );
                        let q = quantize_hfq4g256(&dequant);
                        (q, QuantType::HFQ4G256, 256u32)
                    } else if expert_mq2gl_roundtrip {
                        // MQ2-GL → F32 → HFQ4 round-trip. Identical to the arm
                        // above except the 2-bit step uses ONE tensor-global
                        // codebook + per-block fp16 scale rather than a
                        // per-block fitted codebook. Same HFQ4G256 output, so
                        // the two arms differ ONLY in the injected codec noise.
                        let dequant = mq2g256gl_roundtrip_f32(&f32_slice, &signs1, &signs2);
                        let q = quantize_hfq4g256(&dequant);
                        (q, QuantType::HFQ4G256, 256u32)
                    } else if expert_mq5 || down_mq5 {
                        let q = quantize_mq5g256(&f32_slice, &signs1, &signs2);
                        (q, QuantType::MQ5G256, 256u32)
                    } else if expert_mq6 || down_mq6 {
                        let q = quantize_mq6g256(&f32_slice, &signs1, &signs2);
                        (q, QuantType::MQ6G256, 256u32)
                    } else if expert_hfq6 {
                        let q = quantize_hfq6g256(&f32_slice);
                        (q, QuantType::HFQ6G256, 256u32)
                    } else if expert_hfq4 {
                        let q = quantize_hfq4g256(&f32_slice);
                        (q, QuantType::HFQ4G256, 256u32)
                    } else if use_mfp4 && supports_g256 {
                        let q =
                            quantize_mfp4g32_2d(&f32_slice, inner_m, inner_k_e, &signs1, &signs2);
                        (q, QuantType::MFP4G32, 32u32)
                    } else if use_mfp4p && supports_g256 {
                        let q =
                            quantize_mfp4g32_p_2d(&f32_slice, inner_m, inner_k_e, &signs1, &signs2);
                        (q, QuantType::MFP4G32P, 32u32)
                    } else if use_mfp4e8 && supports_g256 {
                        let q = if let Some(hdir) = hessian_dir_ref {
                            let tname = format!("{parent_owned}{x}.{base_owned}.weight");
                            let hblk = load_hessian_blocks(hdir, &tname);
                            if hblk.is_empty() {
                                GPTQ_E8_FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            } else {
                                GPTQ_E8_FIRED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            quantize_mfp4g32_e8_gptq_2d(
                                &f32_slice, inner_m, inner_k_e, &signs1, &signs2, &hblk,
                            )
                        } else {
                            quantize_mfp4g32_e8_2d(&f32_slice, inner_m, inner_k_e, &signs1, &signs2)
                        };
                        (q, QuantType::MFP4G32E8, 32u32)
                    } else if use_mfp3e8_gptq_fmt && supports_g256 {
                        // mfp3e8-gptq: 3-bit E8 with LDLQ. Falls back to RTN if no Hessian.
                        let q = if let Some(hdir) = hessian_dir_ref {
                            let tname = format!("{parent_owned}{x}.{base_owned}.weight");
                            let hblk = load_hessian_blocks(hdir, &tname);
                            if hblk.is_empty() {
                                GPTQ_E8_FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            } else {
                                GPTQ_E8_FIRED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            quantize_mfp3g32_e8_gptq_2d(
                                &f32_slice, inner_m, inner_k_e, &signs1, &signs2, &hblk,
                            )
                        } else {
                            quantize_mfp3g32_e8_2d(&f32_slice, inner_m, inner_k_e, &signs1, &signs2)
                        };
                        (q, QuantType::MFP3G32E8, 32u32)
                    } else if use_mfp2e8_gptq_fmt && supports_g256 {
                        // mfp2e8-gptq: 2-bit E8 with LDLQ. Falls back to RTN if no Hessian.
                        let q = if let Some(hdir) = hessian_dir_ref {
                            let tname = format!("{parent_owned}{x}.{base_owned}.weight");
                            let hblk = load_hessian_blocks(hdir, &tname);
                            if hblk.is_empty() {
                                GPTQ_E8_FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            } else {
                                GPTQ_E8_FIRED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            quantize_mfp2g32_e8_gptq_2d(
                                &f32_slice, inner_m, inner_k_e, &signs1, &signs2, &hblk,
                            )
                        } else {
                            quantize_mfp2g32_e8_2d(&f32_slice, inner_m, inner_k_e, &signs1, &signs2)
                        };
                        (q, QuantType::MFP2G32E8, 32u32)
                    } else if use_mfp4e8soa && supports_g256 {
                        let q = quantize_mfp4g32_e8_soa_2d(
                            &f32_slice, inner_m, inner_k_e, &signs1, &signs2,
                        );
                        (q, QuantType::MFP4G32E8SOA, 32u32)
                    } else if supports_g256 {
                        let q = quantize_mq4g256(&f32_slice, &signs1, &signs2);
                        (q, QuantType::MQ4G256, 256u32)
                    } else {
                        let q = quantize_hfq4g128(&f32_slice);
                        (q, QuantType::HFQ4G128, 128u32)
                    };
                    let weight = HfqTensor {
                        name: format!("{parent_owned}{slot}.{base_owned}.weight"),
                        quant_type: qt,
                        shape: inner_shape_clone.clone(),
                        group_size: gs,
                        data: quantized,
                        spilled_len: 0,
                    };
                    let sidecar = awq_scales.map(|s| HfqTensor {
                        name: format!("{parent_owned}{slot}.{base_owned}.awq_scale.weight"),
                        quant_type: QuantType::F16,
                        shape: vec![inner_k_e as u32],
                        group_size: 0,
                        data: awq_scales_to_f16_bytes(&s),
                        spilled_len: 0,
                    });
                    (weight, sidecar)
                })
                .collect();
            // Flatten weight+sidecar pairs; each AWQ expert emits two tensors.
            let n_awq = new_pairs.iter().filter(|(_, s)| s.is_some()).count();
            let mut new_tensors: Vec<HfqTensor> = Vec::with_capacity(new_pairs.len() + n_awq);
            for (w, s) in new_pairs {
                new_tensors.push(w);
                if let Some(sc) = s {
                    new_tensors.push(sc);
                }
            }
            quantized_params += inner_n as u64 * n_out_experts as u64;
            // Single eprintln to summarize the whole expert sweep.
            let label = if moe_tier_map.is_some() && parent_layer.is_some() {
                "TierMap"
            } else if graded_hot.is_some() {
                "Graded(MQ6/MQ2L)"
            } else if expert_awq_active && awq_in_sum2_per_expert.is_some() {
                if down_mq6 {
                    "MQ6G256+AWQ"
                } else if down_mq5 || expert_mq5 {
                    "MQ5G256+AWQ"
                } else {
                    "MQ4G256+AWQ"
                }
            } else if expert_mq3lloyd_native {
                "MQ3G256L"
            } else if expert_mq2lloyd_native {
                if imatrix_per_expert.is_some() {
                    "MQ2L+imatrix"
                } else {
                    "MQ2G256L"
                }
            } else if expert_mq2lloyd_roundtrip {
                "MQ2L→HFQ4"
            } else if expert_mq5 || down_mq5 {
                "MQ5G256"
            } else if expert_mq6 || down_mq6 {
                "MQ6G256"
            } else if expert_hfq6 {
                "HFQ6G256"
            } else if expert_hfq4 {
                "HFQ4G256"
            } else if use_mfp4 && supports_g256 {
                "MFP4G32"
            } else if use_mfp4p && supports_g256 {
                "MFP4G32P"
            } else if use_mfp4e8 && supports_g256 {
                if use_gptq_e8 {
                    "MFP4E8-GPTQ"
                } else {
                    "MFP4G32E8"
                }
            } else if use_mfp3e8_gptq_fmt && supports_g256 {
                if use_gptq_mfp3e8 {
                    "MFP3E8-GPTQ"
                } else {
                    "MFP3G32E8"
                }
            } else if use_mfp2e8_gptq_fmt && supports_g256 {
                if use_gptq_mfp2e8 {
                    "MFP2E8-GPTQ"
                } else {
                    "MFP2G32E8"
                }
            } else if use_mfp4e8soa && supports_g256 {
                "MFP4G32E8SOA"
            } else if supports_g256 {
                "MQ4G256"
            } else {
                "HFQ4G128"
            };
            let bytes_per = new_tensors.first().map(|t| t.data.len()).unwrap_or(0);
            eprintln!("  {label:>8}: {parent_owned}{{0..{n_out_experts}}}.{base_owned}.weight {:?} (×{n_out_experts} experts of {n_experts} || {:.1} KB/expert, parallel)",
                inner_shape, bytes_per as f64 / 1024.0);
            hfq_tensors.append(&mut new_tensors);
            // Drop source pages and spill quantized data after each expert batch.
            st_files[*file_idx].drop_tensor_pages(name);
            if let Some(ref mut s) = spill {
                maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024); // 2 GB threshold
            }
            continue;
        }

        // ── deepseek4-q8-mtp short-circuit ───────────────────────────────────────
        // Routed experts (.ffn.experts.*) were claimed by the MQ2-Lloyd
        // branch above. Here we handle everything else:
        //
        //   - antirez-precision-sensitive (compressor / indexer /
        //     router gate.weight): keep as F16 on disk. The compressor
        //     class alone regresses PPL +40-81% if dropped to MQ4
        //     (memory: project_deepseek4_compressor_must_stay_f16); F16 → Q8
        //     on these classes is a smaller hit but still unnecessary.
        //   - All other weights: uniform Q8F16.
        //   - Norms / biases / HC matrices: should_quantize() returns
        //     false → fall through to F16 fallback at the bottom.
        // deepseek4-mtp-precise: all mtp.0.* dense weights (anything that goes
        // through gemv_auto in mtp_forward — wq_a/b, wkv, wo_a/b, e_proj,
        // h_proj, shared experts, gate.weight) stay F16 to eliminate Q8
        // quant noise on the MTP block. Routed experts (".ffn.experts.")
        // are excluded — they MUST stay MQ2-Lloyd because the MoE GEMV
        // kernel (`deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed`) only
        // handles that format.
        let keep_f16_mtp = use_mtp_precise
            && name.starts_with("mtp.")
            && !name.contains(".ffn.experts.")
            && should_quantize(name);
        if (use_deepseek4_source_precision && is_deepseek4_keep_f16(name) || keep_f16_mtp)
            && n_elements >= 32
        {
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            let src_dtype = meta.dtype.as_str();
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            quantized_params += n_elements as u64;
            let f16_bytes: Vec<u8> = f32_data
                .iter()
                .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                .collect();
            eprintln!(
                "  {:>8}: {} {:?} ({} elements, {:.1} KB → {:.1} KB) [src={src_dtype}, keep-F16]",
                "F16",
                name,
                meta.shape,
                n_elements,
                raw_data.len() as f64 / 1024.0,
                f16_bytes.len() as f64 / 1024.0
            );
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::F16,
                shape,
                group_size: 0,
                data: f16_bytes,
                spilled_len: 0,
            });
            st_files[*file_idx].drop_tensor_pages(name);
            if let Some(ref mut s) = spill {
                maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
            }
            continue;
        }
        if use_deepseek4_source_precision && should_quantize(name) && n_elements >= 32 {
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            let src_dtype = meta.dtype.as_str();
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            quantized_params += n_elements as u64;
            let q = quantize_q8f16(&f32_data);
            eprintln!(
                "  {:>8}: {} {:?} ({} elements, {:.1} KB → {:.1} KB) [src={src_dtype}]",
                "Q8_F16",
                name,
                meta.shape,
                n_elements,
                raw_data.len() as f64 / 1024.0,
                q.len() as f64 / 1024.0
            );
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::Q8F16,
                shape,
                group_size: 32,
                data: q,
                spilled_len: 0,
            });
            st_files[*file_idx].drop_tensor_pages(name);
            if let Some(ref mut s) = spill {
                maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
            }
            continue;
        }

        if should_quantize(name) && n_elements >= 32 {
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            quantized_params += n_elements as u64;

            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();

            // Q8HFQ path: split-metadata per-row layout (needs M and K)
            // Exclude embeddings — they use a lookup kernel, not GEMV
            if use_q8hfq && meta.shape.len() == 2 && !name.contains("embed_tokens") {
                let m = meta.shape[0];
                let k = meta.shape[1];
                let (quantized, row_stride) = quantize_q8hfq(&f32_data, m, k);

                // Compute quantization error for Q8HFQ
                let n_groups = k / 32;
                let scales_bytes = n_groups * 2;
                for row in 0..m {
                    let row_off = row * row_stride;
                    for g in 0..n_groups {
                        let scale = f16_to_f32(u16::from_le_bytes([
                            quantized[row_off + g * 2],
                            quantized[row_off + g * 2 + 1],
                        ]));
                        for i in 0..32 {
                            let qval = quantized[row_off + scales_bytes + g * 32 + i] as i8;
                            let dequant = scale * qval as f32;
                            let orig_idx = row * k + g * 32 + i;
                            let err = (dequant - f32_data[orig_idx]).abs();
                            total_quant_error += err as f64;
                            max_quant_error = max_quant_error.max(err);
                        }
                        _n_quant_groups += 1;
                    }
                }

                eprintln!(
                    "  {:>8}: {} {:?} ({} elements, {:.1} KB → {:.1} KB, stride={})",
                    "Q8_HFQ",
                    name,
                    meta.shape,
                    n_elements,
                    raw_data.len() as f64 / 1024.0,
                    quantized.len() as f64 / 1024.0,
                    row_stride
                );

                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: QuantType::Q8HFQ,
                    shape,
                    group_size: 32,
                    data: quantized,
                    spilled_len: 0,
                });
            } else {
                // ── K-map override ──────────────────────────────────────────────
                let kmap_level = kmap.get(&**name).copied().unwrap_or(QuantLevel::Base);

                // AWQ sidecar scales for this tensor — populated only inside the
                // MQ4G256 arm when --awq is enabled and an imatrix entry exists
                // for this tensor's ggml-translated name. After the main tensor
                // push, we emit an `<name>.awq_scale` 1D F16 sidecar tensor so
                // the runtime can apply `x / s` before the rotation kernel at
                // inference time.
                let mut awq_sidecar_scales: Option<Vec<f32>> = None;

                let (quantized, qt, gs, label) = if q8_conv1d_default && is_conv1d_tensor(name) {
                    // DeltaNet conv1d defaults to Q8 (see --no-q8-conv1d to disable).
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else if kmap_level == QuantLevel::Q8 {
                    // K-map says Q8 (embed, lm_head, router)
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else if kmap_level == QuantLevel::F16 {
                    // K-map says F16 (should not normally reach here — should_quantize filters first)
                    let f16_bytes: Vec<u8> = f32_data
                        .iter()
                        .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                        .collect();
                    (f16_bytes, QuantType::F16, 0u32, "F16")
                } else if kmap_level == QuantLevel::Promote6 {
                    // K-map says promote to 6-bit
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if (use_mq4g256
                        || use_mq4_mq6exp
                        || use_mq4_mq2lloydexp
                        || use_mq4_mq2glexp
                        || use_mq4_mq2lloyd_native
                        || use_mq4_mq2lloyd_kmap
                        || use_mq4_mq2lloyd_imatrix
                        || use_mq4_mq3lloyd_kmap
                        || use_mq4_mqlloyd_tiered
                        || use_mq4_mqlloyd_antirez
                        || use_mq4_mqlloyd_antirez_gptq
                        || use_mq4_mq2lloyd_gptq_all
                        || use_mq3g256
                        || use_mq2g256
                        || use_mq2g256_lloyd
                        || use_mq3g256_lloyd)
                        && k_dim % 256 == 0
                    {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                        (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                    } else if (use_hfq4g256
                        || use_hfq3g256
                        || use_hfq3g128
                        || use_hfq2g256
                        || use_hfq2g128)
                        && k_dim % 256 == 0
                    {
                        let q = quantize_hfq6g256(&f32_data);
                        (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                    } else if use_mq6g256 && k_dim % 256 == 0 {
                        // Already 6-bit MQ — no-op promotion
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                        (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                    } else if use_hfq6 && k_dim % 256 == 0 {
                        // Already 6-bit HFQ — no-op promotion
                        let q = quantize_hfq6g256(&f32_data);
                        (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                    } else {
                        // Non-256-aligned fallback: Q8
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    }
                } else if let QuantLevel::Override(override_fmt) = kmap_level {
                    // K-map says override (today: lm_head when --lm-head-format set).
                    // Dispatch on the carried format. For MQ4 with AWQ enabled,
                    // apply AWQ pre-scaling + emit a sidecar so the runtime
                    // (once the CUDA-branch AWQ-aware lm_head dispatch lands)
                    // sees scaled bytes and inverse-divides correctly. For any
                    // other format, plain quantize (the AWQ wiring outside MQ4
                    // is a follow-up).
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        match override_fmt {
                            GgufFormat::Mq4 => {
                                // Inline AWQ + MQ4 dance (mirrors the Base MQ4 arm).
                                let q = if let (Some(alpha), Some(im_weights)) =
                                    (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                                {
                                    if awq_eligible(name) {
                                        let scales = compute_awq_scales(im_weights, alpha);
                                        awq_sidecar_scales = Some(scales.clone());
                                        let m_dim = meta.shape[0];
                                        let mut scaled = f32_data.clone();
                                        awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                        quantize_mq4g256(&scaled, &signs1, &signs2)
                                    } else {
                                        quantize_mq4g256(&f32_data, &signs1, &signs2)
                                    }
                                } else {
                                    quantize_mq4g256(&f32_data, &signs1, &signs2)
                                };
                                (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                            }
                            GgufFormat::Mq5 => {
                                // MQ5 + AWQ on lm_head: MQ5G256 is in
                                // DType::supports_awq_sidecar, so the runtime applies the
                                // inverse divide via rotate_x_mq. Same AWQ inline dance as MQ4.
                                let q = if let (Some(alpha), Some(im_weights)) =
                                    (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                                {
                                    if awq_eligible(name) {
                                        let scales = compute_awq_scales(im_weights, alpha);
                                        awq_sidecar_scales = Some(scales.clone());
                                        let m_dim = meta.shape[0];
                                        let mut scaled = f32_data.clone();
                                        awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                        quantize_mq5g256(&scaled, &signs1, &signs2)
                                    } else {
                                        quantize_mq5g256(&f32_data, &signs1, &signs2)
                                    }
                                } else {
                                    quantize_mq5g256(&f32_data, &signs1, &signs2)
                                };
                                (q, QuantType::MQ5G256, 256u32, "MQ5G256")
                            }
                            GgufFormat::Mq6 => {
                                let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                                (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                            }
                            GgufFormat::Mq3 => {
                                // MQ3 + AWQ on lm_head: runtime supports the sidecar via
                                // DType::supports_awq_sidecar(MQ3G256)=true (per the
                                // fix/lm-head-awq-runtime branch). Wire the same AWQ
                                // inline-quantize dance as the MQ4 arm.
                                let q = if let (Some(alpha), Some(im_weights)) =
                                    (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                                {
                                    if awq_eligible(name) {
                                        let scales = compute_awq_scales(im_weights, alpha);
                                        awq_sidecar_scales = Some(scales.clone());
                                        let m_dim = meta.shape[0];
                                        let mut scaled = f32_data.clone();
                                        awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                        quantize_mq3g256(&scaled, &signs1, &signs2)
                                    } else {
                                        quantize_mq3g256(&f32_data, &signs1, &signs2)
                                    }
                                } else {
                                    quantize_mq3g256(&f32_data, &signs1, &signs2)
                                };
                                (q, QuantType::MQ3G256, 256u32, "MQ3G256")
                            }
                            GgufFormat::Hfq4 => {
                                let q = quantize_hfq4g256(&f32_data);
                                (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                            }
                            GgufFormat::Hfq6 => {
                                let q = quantize_hfq6g256(&f32_data);
                                (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                            }
                            // Other Override targets: not yet wired with AWQ;
                            // emit plain quantization. Used in Phase 0 sweeps
                            // for non-AWQ lm_head experiments.
                            GgufFormat::Mq2 => {
                                let q = quantize_mq2g256(&f32_data, &signs1, &signs2);
                                (q, QuantType::MQ2G256, 256u32, "MQ2G256")
                            }
                            GgufFormat::Mq2Lloyd => {
                                let q = quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2);
                                (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                            }
                            GgufFormat::Mq3Lloyd => {
                                let q = quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2);
                                (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                            }
                            GgufFormat::Mq4Lloyd => {
                                let q = quantize_mq4g256_lloyd(&f32_data, &signs1, &signs2);
                                (q, QuantType::MQ4G256Lloyd, 256u32, "MQ4G256Lloyd")
                            }
                            GgufFormat::Mfp4 => {
                                let m = if meta.shape.len() == 2 {
                                    meta.shape[0]
                                } else {
                                    1
                                };
                                let q = quantize_mfp4g32_2d(&f32_data, m, k_dim, &signs1, &signs2);
                                (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                            }
                            GgufFormat::Mfp4Lloyd => {
                                let m = if meta.shape.len() == 2 {
                                    meta.shape[0]
                                } else {
                                    1
                                };
                                let q = quantize_mfp4g32_lloyd_2d(
                                    &f32_data, m, k_dim, &signs1, &signs2,
                                );
                                (q, QuantType::MFP4G32Lloyd, 32u32, "MFP4G32Lloyd")
                            }
                            GgufFormat::Mfp4P => {
                                let m = if meta.shape.len() == 2 {
                                    meta.shape[0]
                                } else {
                                    1
                                };
                                let q =
                                    quantize_mfp4g32_p_2d(&f32_data, m, k_dim, &signs1, &signs2);
                                (q, QuantType::MFP4G32P, 32u32, "MFP4G32P")
                            }
                            GgufFormat::Mfp4E8 => {
                                let m = if meta.shape.len() == 2 {
                                    meta.shape[0]
                                } else {
                                    1
                                };
                                let q =
                                    quantize_mfp4g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2);
                                (q, QuantType::MFP4G32E8, 32u32, "MFP4G32E8")
                            }
                            GgufFormat::Mfp4E8Soa => {
                                let m = if meta.shape.len() == 2 {
                                    meta.shape[0]
                                } else {
                                    1
                                };
                                let q = quantize_mfp4g32_e8_soa_2d(
                                    &f32_data, m, k_dim, &signs1, &signs2,
                                );
                                (q, QuantType::MFP4G32E8SOA, 32u32, "MFP4G32E8SOA")
                            }
                            GgufFormat::Mfp3E8 => {
                                let m = if meta.shape.len() == 2 {
                                    meta.shape[0]
                                } else {
                                    1
                                };
                                let q =
                                    quantize_mfp3g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2);
                                (q, QuantType::MFP3G32E8, 32u32, "MFP3G32E8")
                            }
                            GgufFormat::Mfp2E8 => {
                                let m = if meta.shape.len() == 2 {
                                    meta.shape[0]
                                } else {
                                    1
                                };
                                let q =
                                    quantize_mfp2g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2);
                                (q, QuantType::MFP2G32E8, 32u32, "MFP2G32E8")
                            }
                            GgufFormat::Hfp4 => {
                                let m = if meta.shape.len() == 2 {
                                    meta.shape[0]
                                } else {
                                    1
                                };
                                let q = quantize_hfp4g32_2d(&f32_data, m, k_dim);
                                (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                            }
                        }
                    } else {
                        // Non-256-aligned override target: Q8 fallback.
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    }
                } else {
                    // QuantLevel::Base — existing format-specific logic below

                    // Choose quant format per tensor
                    let this_q8 = if use_q4k_all {
                        false // everything Q4_K
                    } else if use_q4k_q8embed {
                        name.contains("embed") || name.contains("lm_head") // only embed/output Q8
                    } else if use_mixed || use_fast {
                        is_q8_tensor(name)
                    } else {
                        use_q8 || use_q8hfq // 1D Q8HFQ tensors fall back to Q8F16
                    };
                    let this_q4as8 = use_fast && !this_q8; // FFN tensors in q8-fast mode
                    let this_q4k = use_q4k_all || use_q4k_q8embed || use_mixed;

                    // Embeddings stored as Q8 in HFQ4 mode — Q4 is too lossy for
                    // large-dim models (9B: dim=4096, values ~0.016, Q4 step ~0.007)
                    let is_embed = name.contains("embed_tokens");

                    if use_hfq_mixed {
                        // hfq-mixed: Q8 for attention, HFQ4 for FFN (fits 9B in 8GB VRAM)
                        let is_ffn = name.contains("mlp.") || name.contains("ffn");
                        if !is_ffn {
                            let q = quantize_q8f16(&f32_data);
                            (q, QuantType::Q8F16, 32u32, "Q8_F16")
                        } else {
                            let k_dim = if meta.shape.len() == 2 {
                                meta.shape[1]
                            } else {
                                n_elements
                            };
                            if k_dim % 256 == 0 {
                                let q = quantize_hfq4g256(&f32_data);
                                (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                            } else {
                                let q = quantize_hfq4g128(&f32_data);
                                (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                            }
                        }
                    } else if use_hfq6 {
                        // HFQ6-G256: all weights 6-bit, embeddings Q8
                        if is_embed {
                            let q = quantize_q8f16(&f32_data);
                            (q, QuantType::Q8F16, 32u32, "Q8_F16")
                        } else {
                            let q = quantize_hfq6g256(&f32_data);
                            (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                        }
                    } else if (use_hfq2g256 || use_hfq2g128) && is_embed {
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_hfq2g128 {
                        let q = quantize_hfq2g128(&f32_data);
                        (q, QuantType::HFQ2G128, 128u32, "HFQ2G128")
                    } else if use_hfq2g256 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let q = quantize_hfq2g256(&f32_data);
                            (q, QuantType::HFQ2G256, 256u32, "HFQ2G256")
                        } else {
                            // Fallback to HFQ4 for non-256-aligned
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        }
                    } else if use_mq8g256 && is_embed {
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_mq8g256 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            let q = quantize_mq8g256(&f32_data, &signs1, &signs2);
                            (q, QuantType::MQ8G256, 256u32, "MQ8G256")
                        } else {
                            // Fallback to Q8 for non-256-aligned
                            let q = quantize_q8f16(&f32_data);
                            (q, QuantType::Q8F16, 32u32, "Q8_F16")
                        }
                    } else if q8_router && is_q8_tensor(name) {
                        // Fixed tier (attention / lm_head / embed / router) held above
                        // --format. Default Q8F16; `HIPFIRE_FIXED_TIER=<class>:<dtype>`
                        // overrides per class — the bit-allocation lever.
                        //
                        // Why this matters: the fixed tier is 66% of per-token decode
                        // bytes on a3b, and dropping the WHOLE tier Q8 -> MQ4 measured
                        // +35.2% KLD (0.1742 -> 0.2356) for 1.75x speed. MFP4G32E8SOA
                        // is the interesting middle: ~same bytes as MQ4 (4.3 vs 4.25
                        // bpw) but E8 lattice VQ instead of scalar affine, and it is
                        // already dispatchable for lm_head (plain GEMV, gemv_table.rs
                        // registers it Plain + Prerotated). NOTE it is NOT a whole-tier
                        // replacement: FusedQkvza's E8 arm is gfx1151-decode-only and
                        // there is no E8 residual GEMV for o_proj.
                        match fixed_tier_dtype_for(name) {
                            Some(dt) => {
                                // Canonical FWHT sign seeds — identical to every other
                                // rotated encoder, so bytes match `--format <tier>`.
                                let s1 = gen_fwht_signs(42, 256);
                                let s2 = gen_fwht_signs(1042, 256);
                                let m = meta.shape[0];
                                let k = meta.shape[1];
                                match dt {
                                    "mfp4e8soa" => {
                                        let q =
                                            quantize_mfp4g32_e8_soa_2d(&f32_data, m, k, &s1, &s2);
                                        (q, QuantType::MFP4G32E8SOA, 32u32, "MFP4G32E8SOA")
                                    }
                                    "mfp4e8" => {
                                        let q = quantize_mfp4g32_e8_2d(&f32_data, m, k, &s1, &s2);
                                        (q, QuantType::MFP4G32E8, 32u32, "MFP4G32E8")
                                    }
                                    "mq3l" => {
                                        let q = quantize_mq3g256_lloyd(&f32_data, &s1, &s2);
                                        (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256L")
                                    }
                                    _ => {
                                        let q = quantize_mq4g256(&f32_data, &s1, &s2);
                                        (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                                    }
                                }
                            }
                            None => {
                                let q = quantize_q8f16(&f32_data);
                                (q, QuantType::Q8F16, 32u32, "Q8_F16")
                            }
                        }
                    } else if (use_mq4g256
                        || use_mq4_mq6exp
                        || use_mq4_mq2lloydexp
                        || use_mq4_mq2glexp
                        || use_mq4_mq2lloyd_native
                        || use_mq4_mq2lloyd_kmap
                        || use_mq4_mq2lloyd_imatrix
                        || use_mq4_mq3lloyd_kmap
                        || use_mq4_mqlloyd_tiered
                        || use_mq4_mqlloyd_antirez
                        || use_mq4_mqlloyd_antirez_gptq
                        || use_mq4_mq2lloyd_gptq_all)
                        && is_embed
                    {
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_mq4g256
                        || use_mq4_mq6exp
                        || use_mq4_mq2lloydexp
                        || use_mq4_mq2glexp
                        || use_mq4_mq2lloyd_native
                        || use_mq4_mq2lloyd_kmap
                        || use_mq4_mq2lloyd_imatrix
                        || use_mq4_mq3lloyd_kmap
                        || use_mq4_mqlloyd_tiered
                        || use_mq4_mqlloyd_antirez
                        || use_mq4_mqlloyd_antirez_gptq
                        || use_mq4_mq2lloyd_gptq_all
                    {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            // Phase A Stage A — AWQ pre-scaling, when --awq is enabled
                            // AND we have imatrix data for this tensor AND the tensor
                            // is on the AWQ whitelist (see `awq_eligible`). Mutates a
                            // local copy of the weights so the original f32_data
                            // returned by to_f32() is left intact for downstream
                            // consumers (we don't currently have any here, but this
                            // is hygienic).
                            //
                            // The `awq_eligible(name)` guard is critical: pre-scaling
                            // weights whose runtime path lacks the inverse divide
                            // produces `(W·s)·x ≠ W·x` and catastrophically corrupts
                            // logits (KLD 0.67 → 13.5 measured on 0.8B Qwen3.5 before
                            // this guard landed). See `docs/plans/awq_fix_claude.md`.
                            let q = if let (Some(alpha), Some(im_weights)) =
                                (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                            {
                                if awq_eligible(name) {
                                    debug_assert_eq!(
                                        im_weights.len(),
                                        k_dim,
                                        "imatrix length ({}) != K dim ({}) for {}",
                                        im_weights.len(),
                                        k_dim,
                                        name
                                    );
                                    let scales = compute_awq_scales(im_weights, alpha);
                                    // Stash for sidecar emission after the main tensor push.
                                    awq_sidecar_scales = Some(scales.clone());
                                    let m_dim = meta.shape[0];
                                    // Copy weights so we don't mutate to_f32's buffer
                                    // (might be shared/borrowed depending on dtype path).
                                    let mut scaled = f32_data.clone();
                                    awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                    quantize_mq4g256(&scaled, &signs1, &signs2)
                                } else {
                                    // Runtime path for this weight has no AWQ inverse
                                    // (rotate_x_mq for o_proj/out_proj/wo, or
                                    // fused_silu_mul_rotate_mq for down_proj/w_down).
                                    // Skip AWQ for this tensor — emit plain MQ4 and
                                    // no sidecar.
                                    quantize_mq4g256(&f32_data, &signs1, &signs2)
                                }
                            } else {
                                quantize_mq4g256(&f32_data, &signs1, &signs2)
                            };
                            (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                        } else {
                            // Fallback to standard HFQ4-G128 for non-256-aligned
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        }
                    } else if use_hfp4 && is_embed {
                        // HFP4 embeddings stay Q8F16 (matches MQ4 / HFQ4 pattern — embedding lookup is
                        // accuracy-sensitive, FP4 codes too lossy for vocab-sized tables).
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_hfp4 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 32 == 0 && meta.shape.len() == 2 {
                            let m = meta.shape[0];
                            let q = quantize_hfp4g32_2d(&f32_data, m, k_dim);
                            (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                        } else {
                            // Fallback to HFQ4-G128 for non-32-aligned ragged dims (rare).
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        }
                    } else if use_mfp4 && is_embed {
                        // MFP4 embeddings stay Q8F16 (same rationale as HFP4 / MQ4).
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_mfp4 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 && meta.shape.len() == 2 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            let m = meta.shape[0];
                            let q = quantize_mfp4g32_2d(&f32_data, m, k_dim, &signs1, &signs2);
                            (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                        } else {
                            // Fallback to HFQ4-G128 for non-256-aligned ragged dims (rotation
                            // requires 256-element segments). Matches MQ4's ragged fallback.
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        }
                    } else if use_mfp4l && is_embed {
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_mfp4l {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 && meta.shape.len() == 2 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            let m = meta.shape[0];
                            let q =
                                quantize_mfp4g32_lloyd_2d(&f32_data, m, k_dim, &signs1, &signs2);
                            (q, QuantType::MFP4G32Lloyd, 32u32, "MFP4G32Lloyd")
                        } else {
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        }
                    } else if use_mfp4p && is_embed {
                        // mfp4+P embeddings stay Q8F16 (same rationale as mfp4 / mfp4L).
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_mfp4p {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 && meta.shape.len() == 2 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            let m = meta.shape[0];
                            let q = quantize_mfp4g32_p_2d(&f32_data, m, k_dim, &signs1, &signs2);
                            (q, QuantType::MFP4G32P, 32u32, "MFP4G32P")
                        } else {
                            // Ragged dim fallback — matches mfp4 / mfp4L (HFQ4-G128, no rotation).
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        }
                    } else if (use_mfp4e8
                        || use_mfp4e8soa
                        || use_mfp3e8_gptq_fmt
                        || use_mfp2e8_gptq_fmt)
                        && is_embed
                    {
                        // mfp{2,3,4}-E8 embeddings stay Q8F16 (embedding lookup is accuracy-
                        // sensitive; matches the mfp4 / mfp4L pattern).
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_mfp4e8 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 && meta.shape.len() == 2 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            let m = meta.shape[0];
                            // GPTQ-E8 for dense tensors: keyed by the full
                            // safetensors name (no expert idx). Missing Hessian
                            // -> RTN fallback (byte-identical to plain mfp4e8).
                            let q = if use_gptq_e8 {
                                if let Some(hdir) = hessian_dir.as_deref() {
                                    let hblk = load_hessian_blocks(hdir, name);
                                    if hblk.is_empty() {
                                        GPTQ_E8_FALLBACK
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    } else {
                                        GPTQ_E8_FIRED
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    }
                                    quantize_mfp4g32_e8_gptq_2d(
                                        &f32_data, m, k_dim, &signs1, &signs2, &hblk,
                                    )
                                } else {
                                    quantize_mfp4g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2)
                                }
                            } else {
                                quantize_mfp4g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2)
                            };
                            (q, QuantType::MFP4G32E8, 32u32, "MFP4G32E8")
                        } else {
                            // Ragged dim fallback — matches mfp4+P (HFQ4-G128, no rotation).
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        }
                    } else if use_mfp3e8_gptq_fmt {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 && meta.shape.len() == 2 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            let m = meta.shape[0];
                            // GPTQ-mfp3-E8 for dense tensors. Missing Hessian -> RTN fallback.
                            let q = if use_gptq_mfp3e8 {
                                if let Some(hdir) = hessian_dir.as_deref() {
                                    let hblk = load_hessian_blocks(hdir, name);
                                    if hblk.is_empty() {
                                        GPTQ_E8_FALLBACK
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    } else {
                                        GPTQ_E8_FIRED
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    }
                                    quantize_mfp3g32_e8_gptq_2d(
                                        &f32_data, m, k_dim, &signs1, &signs2, &hblk,
                                    )
                                } else {
                                    quantize_mfp3g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2)
                                }
                            } else {
                                quantize_mfp3g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2)
                            };
                            (q, QuantType::MFP3G32E8, 32u32, "MFP3G32E8")
                        } else {
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        }
                    } else if use_mfp2e8_gptq_fmt {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 && meta.shape.len() == 2 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            let m = meta.shape[0];
                            // GPTQ-mfp2-E8 for dense tensors. Missing Hessian -> RTN fallback.
                            let q = if use_gptq_mfp2e8 {
                                if let Some(hdir) = hessian_dir.as_deref() {
                                    let hblk = load_hessian_blocks(hdir, name);
                                    if hblk.is_empty() {
                                        GPTQ_E8_FALLBACK
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    } else {
                                        GPTQ_E8_FIRED
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    }
                                    quantize_mfp2g32_e8_gptq_2d(
                                        &f32_data, m, k_dim, &signs1, &signs2, &hblk,
                                    )
                                } else {
                                    quantize_mfp2g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2)
                                }
                            } else {
                                quantize_mfp2g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2)
                            };
                            (q, QuantType::MFP2G32E8, 32u32, "MFP2G32E8")
                        } else {
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        }
                    } else if use_mfp4e8soa {
                        // mfp4-E8-SoA: same E8 encoding permuted to SoA layout for coalesced GEMV.
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 && meta.shape.len() == 2 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            let m = meta.shape[0];
                            let q =
                                quantize_mfp4g32_e8_soa_2d(&f32_data, m, k_dim, &signs1, &signs2);
                            (q, QuantType::MFP4G32E8SOA, 32u32, "MFP4G32E8SOA")
                        } else {
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        }
                    } else if use_mq5g256 && is_embed {
                        // MQ5 embeddings stay Q8F16 (embedding lookup is accuracy-
                        // sensitive; matches MQ4 / MQ6 / HFQ4 pattern).
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_mq5g256 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            // AWQ pre-scaling for the MQ5 base body (mirrors the MQ4
                            // base arm). MQ5G256 is on DType::supports_awq_sidecar, so
                            // the runtime applies the inverse divide via rotate_x_mq.
                            // awq_eligible gates to tensors whose runtime path has the
                            // inverse (skips o_proj / down_proj which lack it).
                            let q = if let (Some(alpha), Some(im_weights)) =
                                (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                            {
                                if awq_eligible(name) {
                                    let scales = compute_awq_scales(im_weights, alpha);
                                    awq_sidecar_scales = Some(scales.clone());
                                    let m_dim = meta.shape[0];
                                    let mut scaled = f32_data.clone();
                                    awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                    quantize_mq5g256(&scaled, &signs1, &signs2)
                                } else {
                                    quantize_mq5g256(&f32_data, &signs1, &signs2)
                                }
                            } else {
                                quantize_mq5g256(&f32_data, &signs1, &signs2)
                            };
                            (q, QuantType::MQ5G256, 256u32, "MQ5G256")
                        } else {
                            // Fallback to HFQ4-G128 for non-256-aligned (no MQ5G128).
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        }
                    } else if use_mq6g256 && is_embed {
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_mq6g256 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                            (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                        } else {
                            // Fallback to HFQ6-G256 for non-256-aligned (no rotation)
                            let q = quantize_hfq6g256(&f32_data);
                            (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                        }
                    } else if (use_mq3g256
                        || use_mq2g256
                        || use_mq2g256_lloyd
                        || use_mq3g256_lloyd
                        || use_mq4g256_lloyd)
                        && is_embed
                    {
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_mq4g256_lloyd {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            let q = quantize_mq4g256_lloyd(&f32_data, &signs1, &signs2);
                            (q, QuantType::MQ4G256Lloyd, 256u32, "MQ4G256Lloyd")
                        } else {
                            // Fallback to HFQ4-G128 for non-256-aligned (no rotation).
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        }
                    } else if use_mq3g256_lloyd {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            // AWQ × MQ3-Lloyd composition (MQ3G256Lloyd is forward-path-ready +
                            // now in supports_awq_sidecar). Pre-scale by imatrix, then Lloyd-fit.
                            let q = if let (Some(alpha), Some(im_weights)) =
                                (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                            {
                                if awq_eligible(name) {
                                    let scales = compute_awq_scales(im_weights, alpha);
                                    awq_sidecar_scales = Some(scales.clone());
                                    let m_dim = meta.shape[0];
                                    let mut scaled = f32_data.clone();
                                    awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                    quantize_mq3g256_lloyd(&scaled, &signs1, &signs2)
                                } else {
                                    quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2)
                                }
                            } else {
                                quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2)
                            };
                            (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                        } else {
                            let q = quantize_hfq3g128(&f32_data);
                            (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                        }
                    } else if use_mq2g256_lloyd {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            // AWQ × MQ2-Lloyd (MQ2G256Lloyd is in supports_awq_sidecar): pre-scale
                            // by imatrix first, then Lloyd-fit (K=4, or K=3-ternary under the flag).
                            let awq_scaled: Option<Vec<f32>> =
                                if let (Some(alpha), Some(im_weights)) =
                                    (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                                {
                                    if awq_eligible(name) {
                                        let scales = compute_awq_scales(im_weights, alpha);
                                        awq_sidecar_scales = Some(scales.clone());
                                        let m_dim = meta.shape[0];
                                        let mut scaled = f32_data.clone();
                                        awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                        Some(scaled)
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                };
                            let data: &[f32] = awq_scaled.as_deref().unwrap_or(&f32_data);
                            // HIPFIRE_LLOYD_K3=1 → ternary "MQ1.58" (3-level codebook, reuses kernel).
                            let q = if hipfire_config::developer_var("HIPFIRE_LLOYD_K3")
                                .ok()
                                .as_deref()
                                == Some("1")
                            {
                                quantize_mq2g256_lloyd_k3(data, &signs1, &signs2)
                            } else {
                                quantize_mq2g256_lloyd(data, &signs1, &signs2)
                            };
                            (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                        } else {
                            // Fallback to HFQ2-G128 for non-256-aligned (no rotation)
                            let q = quantize_hfq2g128(&f32_data);
                            (q, QuantType::HFQ2G128, 128u32, "HFQ2G128")
                        }
                    } else if use_mq3g256 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            // AWQ pre-scaling for MQ3 base body (mirrors the MQ4 base arm).
                            // MQ3G256 is on DType::supports_awq_sidecar, so the runtime applies
                            // the inverse divide via rotate_x_mq. Without this, `--format mq3
                            // --awq` was a silent no-op on body tensors (md5(mq3-awq)==md5(mq3)).
                            // awq_eligible gates to tensors whose runtime path has the inverse.
                            let q = if let (Some(alpha), Some(im_weights)) =
                                (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                            {
                                if awq_eligible(name) {
                                    let scales = compute_awq_scales(im_weights, alpha);
                                    awq_sidecar_scales = Some(scales.clone());
                                    let m_dim = meta.shape[0];
                                    let mut scaled = f32_data.clone();
                                    awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                    quantize_mq3g256(&scaled, &signs1, &signs2)
                                } else {
                                    quantize_mq3g256(&f32_data, &signs1, &signs2)
                                }
                            } else {
                                quantize_mq3g256(&f32_data, &signs1, &signs2)
                            };
                            (q, QuantType::MQ3G256, 256u32, "MQ3G256")
                        } else {
                            // Fallback to HFQ3-G128 for non-256-aligned (no rotation)
                            let q = quantize_hfq3g128(&f32_data);
                            (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                        }
                    } else if use_mq2g256 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            // AWQ × plain MQ2 (MQ2G256 now in supports_awq_sidecar). Pre-scale by
                            // imatrix, then quantize. (Plain MQ2 collapses uncalibrated; AWQ is the
                            // test of whether activation-aware scaling rescues uniform 2-bit.)
                            let q = if let (Some(alpha), Some(im_weights)) =
                                (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                            {
                                if awq_eligible(name) {
                                    let scales = compute_awq_scales(im_weights, alpha);
                                    awq_sidecar_scales = Some(scales.clone());
                                    let m_dim = meta.shape[0];
                                    let mut scaled = f32_data.clone();
                                    awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                    quantize_mq2g256(&scaled, &signs1, &signs2)
                                } else {
                                    quantize_mq2g256(&f32_data, &signs1, &signs2)
                                }
                            } else {
                                quantize_mq2g256(&f32_data, &signs1, &signs2)
                            };
                            (q, QuantType::MQ2G256, 256u32, "MQ2G256")
                        } else {
                            // Fallback to HFQ2-G128 for non-256-aligned (no rotation)
                            let q = quantize_hfq2g128(&f32_data);
                            (q, QuantType::HFQ2G128, 128u32, "HFQ2G128")
                        }
                    } else if (use_hfq3g256 || use_hfq3g128) && is_embed {
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_hfq3g128 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 128 == 0 {
                            let q = quantize_hfq3g128(&f32_data);
                            (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                        } else {
                            let q = quantize_hfq3g128(&f32_data);
                            (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                        }
                    } else if use_hfq3g256 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let q = quantize_hfq3g256(&f32_data);
                            (q, QuantType::HFQ3G256, 256u32, "HFQ3G256")
                        } else {
                            let q = quantize_hfq3g128(&f32_data);
                            (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                        }
                    } else if use_hfq4g256 && is_embed {
                        // HFQ4 embeddings: half the size of Q8, same 18-VGPR lookup kernel
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let q = quantize_hfq4g256(&f32_data);
                            (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                        } else {
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        }
                    } else if use_hfq4g256 {
                        // Auto-select G128 vs G256 based on K dimension
                        // G256 preferred: better coalescing, fewer scale/zero overheads
                        // G128 only as fallback when K isn't divisible by 256
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let q = quantize_hfq4g256(&f32_data);
                            (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                        } else if k_dim % 128 == 0 {
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        } else {
                            // Pad to 128-element boundary
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        }
                    } else if this_q8 {
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_FP16")
                    } else if this_q4as8 {
                        let q = quantize_q4_as_q8(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q4asQ8")
                    } else if this_q4k {
                        let q = quantize_q4k(&f32_data);
                        (q, QuantType::Q4K, 256u32, "Q4_K")
                    } else {
                        let q = quantize_q4f16_g64(&f32_data);
                        (q, QuantType::Q4F16G64, 64u32, "Q4_F16")
                    }
                }; // end K-map outer if-else

                // Compute quantization error (skip for Q8 embeddings — always negligible)
                let block_size = gs as usize;
                let is_hfq4 = label == "HFQ4G256" || label == "HFQ4G128";
                // Only compute detailed error for HFQ4 tensors — Q8/HFQ6 error is negligible
                let skip_error = !is_hfq4;
                let n_blocks = if !skip_error {
                    (n_elements + block_size - 1) / block_size
                } else {
                    0
                };
                for b in 0..n_blocks {
                    let start = b * block_size;
                    let end = (start + block_size).min(n_elements);
                    if is_hfq4 {
                        // Both G128 (72B) and G256 (136B): [f32 scale][f32 zero][nibbles]
                        let block_bytes = if block_size == 256 { 136 } else { 72 };
                        let off = b * block_bytes;
                        let scale = f32::from_le_bytes([
                            quantized[off],
                            quantized[off + 1],
                            quantized[off + 2],
                            quantized[off + 3],
                        ]);
                        let zero = f32::from_le_bytes([
                            quantized[off + 4],
                            quantized[off + 5],
                            quantized[off + 6],
                            quantized[off + 7],
                        ]);
                        for i in 0..(end - start) {
                            let byte_idx = i / 2;
                            let nibble = if i % 2 == 0 {
                                quantized[off + 8 + byte_idx] & 0xF
                            } else {
                                quantized[off + 8 + byte_idx] >> 4
                            };
                            let dequant = scale * nibble as f32 + zero;
                            let err = (dequant - f32_data[start + i]).abs();
                            total_quant_error += err as f64;
                            max_quant_error = max_quant_error.max(err);
                        }
                    } else if label == "Q8_FP16" || label == "Q4asQ8" || label == "Q8_F16" {
                        // NB: string match because this_q8/this_q4as8 are scoped inside Base block.
                        let off = b * 34;
                        let scale =
                            f16_to_f32(u16::from_le_bytes([quantized[off], quantized[off + 1]]));
                        for i in 0..(end - start) {
                            let qval = quantized[off + 2 + i] as i8;
                            let dequant = scale * qval as f32;
                            let err = (dequant - f32_data[start + i]).abs();
                            total_quant_error += err as f64;
                            max_quant_error = max_quant_error.max(err);
                        }
                    } else {
                        let off = b * 36;
                        let scale =
                            f16_to_f32(u16::from_le_bytes([quantized[off], quantized[off + 1]]));
                        let min_val = f16_to_f32(u16::from_le_bytes([
                            quantized[off + 2],
                            quantized[off + 3],
                        ]));
                        for i in 0..(end - start) {
                            let byte_idx = if i < 32 { i } else { i - 32 };
                            let nibble = if i < 32 {
                                quantized[off + 4 + byte_idx] & 0xF
                            } else {
                                quantized[off + 4 + byte_idx] >> 4
                            };
                            let dequant = nibble as f32 * scale + min_val;
                            let err = (dequant - f32_data[start + i]).abs();
                            total_quant_error += err as f64;
                            max_quant_error = max_quant_error.max(err);
                        }
                    }
                    _n_quant_groups += 1;
                }

                eprintln!(
                    "  {label:>8}: {} {:?} ({} elements, {:.1} KB → {:.1} KB)",
                    name,
                    meta.shape,
                    n_elements,
                    raw_data.len() as f64 / 1024.0,
                    quantized.len() as f64 / 1024.0
                );

                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: qt,
                    shape: shape.clone(),
                    group_size: gs,
                    data: quantized,
                    spilled_len: 0,
                });
                // Phase A Stage A — emit AWQ scale sidecar tensor immediately
                // after the parent weight. Naming convention:
                // `<weight_name>.awq_scale` (strip the trailing `.weight` and
                // append `.awq_scale.weight` so the runtime loader recognizes
                // it as a 1D F16 tensor of length K). 1D shape [K]; runtime
                // pairs it with the parent weight at model open.
                if let Some(scales) = awq_sidecar_scales.take() {
                    let sidecar_name = match name.strip_suffix(".weight") {
                        Some(stem) => format!("{stem}.awq_scale.weight"),
                        None => format!("{name}.awq_scale.weight"),
                    };
                    let bytes = awq_scales_to_f16_bytes(&scales);
                    eprintln!(
                        "    AWQ:    {} [{}] (1D F16, {} B)",
                        sidecar_name,
                        scales.len(),
                        bytes.len()
                    );
                    hfq_tensors.push(HfqTensor {
                        name: sidecar_name,
                        quant_type: QuantType::F16,
                        shape: vec![scales.len() as u32],
                        group_size: 0,
                        data: bytes,
                        spilled_len: 0,
                    });
                }
            } // end else (non-Q8HFQ path)
        } else if is_vision && vision_quant == "hfq4" && n_elements >= 32 {
            // Quantize vision weights to HFQ4G256 (for speed-critical VL workloads)
            let f32_data = to_f32(raw_data, &meta.dtype);
            quantized_params += n_elements as u64;
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            let k_dim = if shape.len() == 2 {
                shape[1] as usize
            } else {
                n_elements
            };
            let (quantized, gs) = if k_dim % 256 == 0 {
                (quantize_hfq4g256(&f32_data), 256u32)
            } else {
                (quantize_hfq4g128(&f32_data), 128u32)
            };
            let qt = if gs == 256 {
                QuantType::HFQ4G256
            } else {
                QuantType::HFQ4G128
            };
            let label = if gs == 256 { "HFQ4G256" } else { "HFQ4G128" };
            eprintln!(
                "  {label:>8}: {} {:?} ({} elements, {:.1} KB -> {:.1} KB) [vision]",
                name,
                meta.shape,
                n_elements,
                raw_data.len() as f64 / 1024.0,
                quantized.len() as f64 / 1024.0
            );
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: qt,
                shape,
                group_size: gs,
                data: quantized,
                spilled_len: 0,
            });
        } else if is_vision && vision_quant == "bf16" && meta.dtype == "BF16" {
            // Store vision weights as original BF16 (zero precision loss)
            quantized_params += n_elements as u64;
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            eprintln!(
                "  BF16:       {} {:?} ({} elements, {:.1} KB) [vision, lossless]",
                name,
                meta.shape,
                n_elements,
                raw_data.len() as f64 / 1024.0
            );
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::BF16,
                shape,
                group_size: 0,
                data: raw_data.to_vec(),
                spilled_len: 0,
            });
        } else if is_vision && vision_quant == "bf16" {
            // Non-BF16 source (F16/F32) — store as F16
            let data = if meta.dtype == "F16" {
                raw_data.to_vec()
            } else {
                let f32_vals = to_f32(raw_data, &meta.dtype);
                f32_vals
                    .iter()
                    .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                    .collect()
            };
            quantized_params += n_elements as u64;
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            eprintln!(
                "  F16:        {} {:?} ({:.1} KB) [vision, bf16 fallback]",
                name,
                meta.shape,
                data.len() as f64 / 1024.0
            );
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::F16,
                shape,
                group_size: 0,
                data,
                spilled_len: 0,
            });
        } else {
            // Keep as F16 (convert BF16 -> F16 if needed)
            let f16_data = match meta.dtype.as_str() {
                "F16" => raw_data.to_vec(),
                "BF16" => {
                    // BF16 → F32 → F16
                    let f32_vals = to_f32(raw_data, "BF16");
                    f32_vals
                        .iter()
                        .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                        .collect()
                }
                "F32" => {
                    let f32_vals = to_f32(raw_data, "F32");
                    f32_vals
                        .iter()
                        .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                        .collect()
                }
                other => panic!("unsupported dtype for norm/embd: {other}"),
            };

            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            eprintln!(
                "  F16:        {} {:?} ({} elements, {:.1} KB)",
                name,
                meta.shape,
                n_elements,
                f16_data.len() as f64 / 1024.0
            );

            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::F16,
                shape,
                group_size: 0,
                data: f16_data,
                spilled_len: 0,
            });
        }
        // Release source file page cache after each tensor to prevent
        // mmap'd pages from starving GPU allocations on UMA systems.
        st_files[*file_idx].drop_tensor_pages(name);
    }

    // Summary
    let total_bytes: usize = hfq_tensors
        .iter()
        .map(|t| {
            if t.spilled_len > 0 {
                t.spilled_len as usize
            } else {
                t.data.len()
            }
        })
        .sum();
    {
        let fired = GPTQ_E8_FIRED.load(std::sync::atomic::Ordering::Relaxed);
        let fb = GPTQ_E8_FALLBACK.load(std::sync::atomic::Ordering::Relaxed);
        if fired + fb > 0 {
            eprintln!(
                "  GPTQ-on-E8: {fired} tensors FIRED (Hessian-aware LDLQ), {fb} RTN-fallback (missing/singular H). {:.1}% fired.",
                100.0 * fired as f64 / (fired + fb) as f64
            );
            if fired == 0 {
                eprintln!("  WARNING: 0 GPTQ tensors fired with --hessian-dir set — likely a KEY-MISMATCH (.hblk filenames != hessian_key), NOT a flat result.");
            }
        }
    }
    let mean_quant_error = if quantized_params > 0 {
        total_quant_error / quantized_params as f64
    } else {
        0.0
    };

    eprintln!("\n=== Quantization Summary ===");
    if skipped_params > 0 {
        eprintln!(
            "  Skipped params:   {skipped_params} (mtp/visual — use --include-vision for VL)"
        );
    }
    eprintln!("  Total params:     {total_params}");
    eprintln!(
        "  Quantized params: {quantized_params} ({:.1}%)",
        100.0 * quantized_params as f64 / total_params as f64
    );
    eprintln!("  Mean quant error: {mean_quant_error:.8}");
    eprintln!("  Max quant error:  {max_quant_error:.8}");
    eprintln!("  Output size:      {:.1} MB", total_bytes as f64 / 1e6);

    // ── SP4b: bake prune finalize (rename kept per-expert tensors + patch count) ──
    // Applied only when a bake keep-map is active. Renames the per-expert-named
    // kept tensors (ds4 score layers / lfm2 / minimax) recorded during the loop to
    // their compact slots, then patches the output metadata's routed-expert count
    // to `kept_per_layer` so the baked model loads standalone (no env var, no
    // load-time keep-map). Spill preserves `.name`, so rename order vs. spill is
    // irrelevant. (Qwen3.5 stacked experts + all routers/biases were already
    // pruned/gathered in-loop.)
    // Task A0: apply Qwen3.5-MoE pre-split expert fusion renames. Unconditional
    // (unlike the bake rename below, which is gated on `bake_keep_active`):
    // rewrite each fused gate_proj output tensor to the loader's
    // `experts.{N}.gate_up_proj.weight`. The in-loop quant path kept the original
    // gate_proj name so k-map/encoding used it. Spill preserves `.name`, so this
    // works regardless of spill order. No-op (byte-identical) for every model
    // that isn't a pre-split Qwen3.5-MoE.
    if !expert_fuse_rename.is_empty() {
        for t in hfq_tensors.iter_mut() {
            if let Some(new_name) = expert_fuse_rename.get(&t.name) {
                t.name = new_name.clone();
            }
        }
        eprintln!(
            "qwen35 expert fusion: renamed {} fused gate_up_proj tensors",
            expert_fuse_rename.len()
        );
    }

    if bake_keep_active {
        let plan = reap_bake_plan.as_ref().unwrap();
        if !bake_rename.is_empty() {
            for t in hfq_tensors.iter_mut() {
                if let Some(new_name) = bake_rename.get(&t.name) {
                    t.name = new_name.clone();
                }
            }
            eprintln!(
                "REAP bake: renamed {} kept per-expert tensors to compact slots",
                bake_rename.len()
            );
        }
        let kept = plan.kept_per_layer();
        match patch_expert_count_metadata(&metadata_json, reap_arch, kept) {
            Ok(patched) => {
                metadata_json = patched;
                eprintln!("REAP bake: patched output metadata expert count → {kept}");
            }
            Err(e) => {
                eprintln!("reap bake: failed to patch expert-count metadata: {e}");
                std::process::exit(2);
            }
        }
    }

    // Write .hfq file
    eprintln!("\nWriting: {}", output_path.display());
    // Final spill before writing
    if let Some(ref mut s) = spill {
        maybe_spill(&mut hfq_tensors, s, 0); // spill everything remaining
    }
    write_hfq(
        output_path,
        arch_id,
        &metadata_json,
        &hfq_tensors,
        spill.as_mut(),
    )
    .unwrap();
    if let Some(s) = spill {
        s.cleanup();
    }

    let file_size = std::fs::metadata(output_path).unwrap().len();
    eprintln!("Done: {:.1} MB written", file_size as f64 / 1e6);
}

#[cfg(test)]
mod gptq_damping_probe {
    //! Offline GPTQ-Lloyd damping sweep. Runs the GPTQ-Lloyd quant pipeline
    //! against synthetic DeepSeek V4-realistic weight distributions across a damping
    //! range, compares per-block reconstruction MSE to plain Lloyd. Catches
    //! a known failure mode where forward-error-propagation on FWHT-rotated
    //! (largely-decorrelated) weights INJECTS noise rather than removing it
    //! at moderate-to-high damping values — what the DeepSeek V4 MQ2-GPTQ-all run
    //! is suspected to be hitting.
    //!
    //! Run with:
    //!   cargo test -p hipfire-quantize gptq_damping_probe -- --nocapture
    use super::*;

    /// Deterministic Box-Muller-from-LCG Gaussian sampler — no external dep.
    /// Returns N samples with zero mean and unit variance.
    fn gaussian_samples(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut step = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as u64 & ((1u64 << 53) - 1)) as f64 / (1u64 << 53) as f64
        };
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let mut u1 = step() as f64;
            if u1 < 1e-12 {
                u1 = 1e-12;
            }
            let u2 = step() as f64;
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            out.push((r * theta.cos()) as f32);
            if out.len() < n {
                out.push((r * theta.sin()) as f32);
            }
        }
        out
    }

    fn mse(a: &[f32], b: &[f32]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        let mut acc = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = *x as f64 - *y as f64;
            acc += d * d;
        }
        acc / a.len() as f64
    }

    fn run_one_distribution(label: &str, weights: &[f32]) {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let n = weights.len();
        // Unit column weights — what DeepSeek V4's mq2-gptq-all build passes.
        let unit: Vec<f32> = vec![1.0; n];

        eprintln!("\n=== {label} (n={n}) ===");

        let lloyd_bytes = quantize_mq2g256_lloyd(weights, &signs1, &signs2);
        let lloyd_recon = dequantize_mq2g256_lloyd_to_f32(&lloyd_bytes, n, &signs1, &signs2);
        let lloyd_mse = mse(weights, &lloyd_recon);
        eprintln!("  Lloyd                  MSE = {:.6e}", lloyd_mse);

        for damping in [0.0_f32, 0.1, 0.3, 0.5, 0.8, 1.0] {
            let gptq_bytes =
                quantize_mq2g256_lloyd_gptq_with_damping(weights, &unit, &signs1, &signs2, damping);
            let gptq_recon = dequantize_mq2g256_lloyd_to_f32(&gptq_bytes, n, &signs1, &signs2);
            let gptq_mse = mse(weights, &gptq_recon);
            let delta = ((gptq_mse - lloyd_mse) / lloyd_mse) * 100.0;
            eprintln!(
                "  GPTQ d={damping:>4.1}             MSE = {:.6e}  ({:+.2}% vs Lloyd)",
                gptq_mse, delta
            );
        }
    }

    /// Variant of plain Lloyd with tunable iteration count. Used to test
    /// whether the production 8-iter cap is leaving headroom.
    fn quantize_mq2g256_lloyd_niter(
        f32_data: &[f32],
        signs1: &[f32],
        signs2: &[f32],
        max_iter: usize,
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
                let mut cb: [f32; 4] = [
                    percentile(0.125),
                    percentile(0.375),
                    percentile(0.625),
                    percentile(0.875),
                ];
                let range = sorted[255] - sorted[0];
                let mut indices = [0u8; 256];
                if range > 0.0 {
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

    fn run_lloyd_iter_sweep(label: &str, weights: &[f32]) {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let n = weights.len();
        eprintln!("\n=== {label} (n={n}) — Lloyd iteration sweep ===");
        let mut prev = f64::NAN;
        for niter in [1usize, 2, 4, 8, 16, 32, 64] {
            let bytes = quantize_mq2g256_lloyd_niter(weights, &signs1, &signs2, niter);
            let recon = dequantize_mq2g256_lloyd_to_f32(&bytes, n, &signs1, &signs2);
            let m = mse(weights, &recon);
            let delta = if prev.is_finite() {
                format!("  ({:+.3}% vs niter=prev)", ((m - prev) / prev) * 100.0)
            } else {
                String::new()
            };
            eprintln!("  Lloyd niter={niter:>3}        MSE = {m:.6e}{delta}");
            prev = m;
        }
    }

    /// Huber-Lloyd: same Lloyd loop but the centroid update is the
    /// weighted-mean of points with |w - cb| ≤ k_huber * sigma, where
    /// sigma is the within-cluster standard deviation. Points with
    /// larger residuals get clipped (treated as `cb ± k_huber * sigma`)
    /// so they don't drag centroids toward outlier values. With FWHT-
    /// rotated weights the long tails are dampened but not eliminated;
    /// this tests whether residual heavy-tailedness is hurting MSE.
    fn quantize_mq2g256_huber_lloyd(
        f32_data: &[f32],
        signs1: &[f32],
        signs2: &[f32],
        k_huber: f32,
        max_iter: usize,
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
                let mut cb: [f32; 4] = [
                    percentile(0.125),
                    percentile(0.375),
                    percentile(0.625),
                    percentile(0.875),
                ];
                let range = sorted[255] - sorted[0];
                let mut indices = [0u8; 256];
                if range > 0.0 {
                    let mut prev_assignments = [0u8; 256];
                    for it in 0..max_iter {
                        // Assignment pass — same as plain Lloyd.
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
                            prev_assignments[i] = best as u8;
                            indices[i] = best as u8;
                        }
                        // Within-cluster sigma estimate (one pass).
                        let mut sums = [0.0f64; 4];
                        let mut sqs = [0.0f64; 4];
                        let mut cnts = [0u32; 4];
                        for i in 0..256 {
                            let k = indices[i] as usize;
                            let d = (group[i] - cb[k]) as f64;
                            sums[k] += group[i] as f64;
                            sqs[k] += d * d;
                            cnts[k] += 1;
                        }
                        let mut sigma = [0.0f64; 4];
                        for k in 0..4 {
                            if cnts[k] > 0 {
                                sigma[k] = (sqs[k] / cnts[k] as f64).sqrt();
                            }
                        }
                        // Huber-clipped update.
                        let mut wsums = [0.0f64; 4];
                        let mut wcnts = [0.0f64; 4];
                        for i in 0..256 {
                            let k = indices[i] as usize;
                            let lim = (k_huber as f64) * sigma[k].max(1e-9);
                            let resid = (group[i] - cb[k]) as f64;
                            let clipped = resid.max(-lim).min(lim);
                            let effective_w = cb[k] as f64 + clipped;
                            wsums[k] += effective_w;
                            wcnts[k] += 1.0;
                        }
                        let mut changed = 0u32;
                        for k in 0..4 {
                            if wcnts[k] > 0.0 {
                                let new_cb = (wsums[k] / wcnts[k]) as f32;
                                if new_cb != cb[k] {
                                    changed += 1;
                                }
                                cb[k] = new_cb;
                            }
                        }
                        // Suppress unused warnings on sums.
                        let _ = sums;
                        if it > 0 && changed == 0 {
                            break;
                        }
                    }
                    // Final argmin pass to lock indices to the final centroids.
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
                        indices[i] = best as u8;
                    }
                }
                // Sort centroids, remap, pack.
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

    fn run_huber_sweep(label: &str, weights: &[f32]) {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let n = weights.len();
        eprintln!("\n=== {label} (n={n}) — Huber-Lloyd sweep (16 iter) ===");
        // Reference: plain Lloyd at 16 iter.
        let ref_bytes = quantize_mq2g256_lloyd_niter(weights, &signs1, &signs2, 16);
        let ref_recon = dequantize_mq2g256_lloyd_to_f32(&ref_bytes, n, &signs1, &signs2);
        let ref_mse = mse(weights, &ref_recon);
        eprintln!("  Lloyd (niter=16)          MSE = {ref_mse:.6e}");
        for k_huber in [1.0_f32, 1.5, 2.0, 2.5, 3.0, 10.0] {
            let bytes = quantize_mq2g256_huber_lloyd(weights, &signs1, &signs2, k_huber, 16);
            let recon = dequantize_mq2g256_lloyd_to_f32(&bytes, n, &signs1, &signs2);
            let m = mse(weights, &recon);
            let delta = ((m - ref_mse) / ref_mse) * 100.0;
            eprintln!(
                "  Huber k={k_huber:>4.1} (niter=16)   MSE = {m:.6e}  ({delta:+.2}% vs Lloyd16)"
            );
        }
    }

    /// GPTQ sequential pass on already-FWHT'd weights, no inner FWHT.
    /// Used to A/B test the FWHT-position hypothesis: production GPTQ
    /// FWHTs then propagates → noise injection. Pre-FWHT GPTQ
    /// (correlated input) should help when input weights have
    /// channel correlation.
    fn quantize_mq2g256_lloyd_gptq_no_fwht(
        f32_data: &[f32],
        damping: f32,
        max_iter: usize,
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
                // NO FWHT here — operate on raw correlated weights.
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
                    let mut prev = [0u8; 256];
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
                            if it == 0 || prev[i] != best as u8 {
                                changed += 1;
                            }
                            prev[i] = best as u8;
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
                // Sequential GPTQ with no inner FWHT.
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

    /// Dequant the no-FWHT variant: indices + codebook, no inv-FWHT step.
    fn dequant_no_fwht(data: &[u8], n_weights: usize) -> Vec<f32> {
        let group_size = 256;
        let block_bytes = 72;
        let n_blocks = (n_weights + group_size - 1) / group_size;
        let mut out = vec![0.0f32; n_weights];
        for b in 0..n_blocks {
            let blk = &data[b * block_bytes..(b + 1) * block_bytes];
            let cb: [f32; 4] = [
                f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])),
                f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])),
                f16_to_f32(u16::from_le_bytes([blk[4], blk[5]])),
                f16_to_f32(u16::from_le_bytes([blk[6], blk[7]])),
            ];
            for i in 0..64 {
                let bv = blk[8 + i];
                for j in 0..4 {
                    let global_i = b * 256 + 4 * i + j;
                    if global_i < n_weights {
                        let idx = (bv >> (j * 2)) & 0x3;
                        out[global_i] = cb[idx as usize];
                    }
                }
            }
        }
        out
    }

    fn correlated_weights(n: usize, seed: u64, decay: f32) -> Vec<f32> {
        // AR(1) process: x_t = decay * x_{t-1} + sqrt(1 - decay^2) * z_t.
        // Produces channel-correlated weights (decay > 0).
        let gauss = gaussian_samples(n, seed);
        let mut out = Vec::with_capacity(n);
        let mut prev = 0.0f32;
        let noise_scale = (1.0f32 - decay * decay).sqrt();
        for &g in &gauss {
            let v = decay * prev + noise_scale * g;
            out.push(v);
            prev = v;
        }
        out
    }

    /// Dequant for MQ3-Lloyd (qt=20): 16 B fp16 codebook (8 entries) +
    /// 96 B 3-bit packed indices = 112 B / 256 weights.
    fn dequantize_mq3g256_lloyd_to_f32(
        data: &[u8],
        n_weights: usize,
        signs1: &[f32],
        signs2: &[f32],
    ) -> Vec<f32> {
        let group_size = 256;
        let block_bytes = 112;
        let n_blocks = (n_weights + group_size - 1) / group_size;
        assert!(data.len() >= n_blocks * block_bytes);
        let mut out = vec![0.0f32; n_weights];
        for b in 0..n_blocks {
            let blk = &data[b * block_bytes..(b + 1) * block_bytes];
            let mut cb = [0.0f32; 8];
            for k in 0..8 {
                cb[k] = f16_to_f32(u16::from_le_bytes([blk[2 * k], blk[2 * k + 1]]));
            }
            let mut group = [0.0f32; 256];
            for chunk in 0..32 {
                let bo = 16 + chunk * 3;
                let b0 = blk[bo];
                let b1 = blk[bo + 1];
                let b2 = blk[bo + 2];
                let mut q = [0u8; 8];
                q[0] = b0 & 7;
                q[1] = (b0 >> 3) & 7;
                q[2] = ((b0 >> 6) & 3) | ((b1 & 1) << 2);
                q[3] = (b1 >> 1) & 7;
                q[4] = (b1 >> 4) & 7;
                q[5] = ((b1 >> 7) & 1) | ((b2 & 3) << 1);
                q[6] = (b2 >> 2) & 7;
                q[7] = (b2 >> 5) & 7;
                for j in 0..8 {
                    group[chunk * 8 + j] = cb[q[j] as usize];
                }
            }
            cpu_inv_fwht_256(&mut group, signs1, signs2);
            let actual = (n_weights - b * 256).min(256);
            for j in 0..actual {
                out[b * 256 + j] = group[j];
            }
        }
        out
    }

    /// Quantifies the MSE cost of antirez's MQ3 → MQ2 down-projection
    /// downgrade. Procedure: take a synthetic DeepSeek V4-realistic weight
    /// distribution, quantize via MQ3-Lloyd (treat its dequant as the
    /// best-fit-available reference), then RE-quantize that dequant via
    /// MQ2-Lloyd. MSE delta = "what antirez loses by dropping MQ3 down".
    ///
    /// Result feeds the question: is the antirez precision tax (2/3 × MQ2
    /// + 1/3 × MQ3 ≈ 2.7 bpw vs 2.25 bpw all-MQ2, ~13 GB on a 256-expert
    /// 43-layer DeepSeek V4) buying meaningful per-tensor MSE reduction, or is
    /// the antirez win at high ctx mostly from Q8 attention?
    fn antirez_downgrade_cost(label: &str, weights: &[f32]) {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let n = weights.len();
        let mq3_bytes = quantize_mq3g256_lloyd(weights, &signs1, &signs2);
        let mq3_recon = dequantize_mq3g256_lloyd_to_f32(&mq3_bytes, n, &signs1, &signs2);
        let mq2_bytes = quantize_mq2g256_lloyd(weights, &signs1, &signs2);
        let mq2_recon = dequantize_mq2g256_lloyd_to_f32(&mq2_bytes, n, &signs1, &signs2);
        // Direct MSE against the synthetic input (ground truth):
        let mq3_mse = mse(weights, &mq3_recon);
        let mq2_mse = mse(weights, &mq2_recon);
        let downgrade_pct = ((mq2_mse - mq3_mse) / mq3_mse) * 100.0;
        eprintln!("  {label} (n={n})");
        eprintln!("    MQ3-Lloyd (3.5 bpw) MSE = {mq3_mse:.6e}");
        eprintln!("    MQ2-Lloyd (2.25 bpw) MSE = {mq2_mse:.6e}");
        eprintln!("    MQ3→MQ2 downgrade cost: {downgrade_pct:+.1}% MSE");
    }

    #[test]
    fn antirez_mq3_to_mq2_downgrade_cost() {
        // Tests on the same DeepSeek V4-realistic distributions as the GPTQ probe.
        eprintln!("\n=== Antirez MQ3-down → MQ2-down downgrade cost ===");
        antirez_downgrade_cost("Gaussian 16x256", &gaussian_samples(16 * 256, 0xc001cafe));
        let mut htw = gaussian_samples(16 * 256, 0xfeed);
        let tail = gaussian_samples((16 * 256) / 20, 0xbeef);
        for (i, t) in tail.iter().enumerate() {
            htw[i * 20] = t * 3.0;
        }
        antirez_downgrade_cost("Heavy-tailed 16x256", &htw);
        let mut sw = gaussian_samples(16 * 256, 0x5_a55e);
        for v in sw.iter_mut() {
            *v *= 0.1;
        }
        for i in 0..(16 * 256 / 20) {
            sw[i * 20] *= 30.0;
        }
        antirez_downgrade_cost("Sparse + outliers 16x256", &sw);
    }

    #[test]
    fn gptq_on_correlated_pre_fwht() {
        // The whole point of GPTQ is to exploit channel correlation.
        // Test it on correlated (decay=0.7), modestly-correlated (0.4),
        // and uncorrelated (0.0) inputs WITHOUT the inner FWHT step.
        //
        // If d>0 wins on correlated inputs but loses on uncorrelated,
        // that confirms: the production code's mistake is FWHT-then-GPTQ.
        // Fix path: drop the FWHT before the sequential pass (move it
        // into dequant or change the runtime kernel to apply it on
        // dequant'd values).
        eprintln!("\n=== GPTQ on correlated weights (no inner FWHT) ===");
        for (label, decay) in [
            ("decay=0.0 (uncorrelated)", 0.0f32),
            ("decay=0.4 (moderately correlated)", 0.4),
            ("decay=0.7 (strongly correlated)", 0.7),
            ("decay=0.9 (very correlated)", 0.9),
        ] {
            let n = 16 * 256;
            let w = correlated_weights(n, 0xc011a7ed, decay);
            // Reference: plain Lloyd via no-FWHT path with d=0.
            let ref_bytes = quantize_mq2g256_lloyd_gptq_no_fwht(&w, 0.0, 16);
            let ref_recon = dequant_no_fwht(&ref_bytes, n);
            let ref_mse = mse(&w, &ref_recon);
            eprintln!("\n  {label} (n={n})");
            eprintln!("    Lloyd                  MSE = {ref_mse:.6e}");
            for damping in [0.05f32, 0.1, 0.2, 0.3, 0.5, 0.8] {
                let b = quantize_mq2g256_lloyd_gptq_no_fwht(&w, damping, 16);
                let r = dequant_no_fwht(&b, n);
                let m = mse(&w, &r);
                let delta = ((m - ref_mse) / ref_mse) * 100.0;
                eprintln!(
                    "    GPTQ d={damping:>4.2} (no-fwht)   MSE = {m:.6e}  ({delta:+.2}% vs Lloyd)"
                );
            }
        }
    }

    #[test]
    fn huber_lloyd_headroom() {
        let mut htw = gaussian_samples(16 * 256, 0xfeed);
        let tail = gaussian_samples((16 * 256) / 20, 0xbeef);
        for (i, t) in tail.iter().enumerate() {
            htw[i * 20] = t * 3.0;
        }
        run_huber_sweep("Heavy-tailed 16x256", &htw);
        let mut sw = gaussian_samples(16 * 256, 0x5_a55e);
        for v in sw.iter_mut() {
            *v *= 0.1;
        }
        for i in 0..(16 * 256 / 20) {
            sw[i * 20] *= 30.0;
        }
        run_huber_sweep("Sparse + outliers 16x256", &sw);
        run_huber_sweep("Gaussian 16x256", &gaussian_samples(16 * 256, 0xc001cafe));
    }

    /// Test "weight-norm proxy imatrix": a calibration-free approximation
    /// using column 2-norm of the weight matrix itself as the per-channel
    /// importance signal. Real AWQ uses sum_t |a_tj|^2; we substitute
    /// sum_i |w_ij|^2. Both produce a [K]-shaped vector that's used to
    /// weight the Lloyd codebook fit.
    ///
    /// If this gives meaningful MSE improvement over uniform Lloyd on
    /// heavy-tailed distributions, it's a viable calibration-free path
    /// to better DeepSeek V4 quants. Bench-falsified if it doesn't beat uniform
    /// by a clear margin.
    fn weight_norm_proxy_imatrix(weights: &[f32], m: usize, k: usize) -> Vec<f32> {
        let mut col_norms = vec![0.0f32; k];
        for r in 0..m {
            for j in 0..k {
                let w = weights[r * k + j];
                col_norms[j] += w * w;
            }
        }
        for v in col_norms.iter_mut() {
            *v = v.sqrt();
        }
        // Normalize so geometric mean is 1.0 (matches AWQ convention).
        let mut sum_log = 0.0f64;
        for &v in &col_norms {
            sum_log += (v.max(1e-12) as f64).ln();
        }
        let mean_log = sum_log / k as f64;
        for v in col_norms.iter_mut() {
            *v = ((*v as f64).ln() - mean_log).exp() as f32;
        }
        col_norms
    }

    fn run_weight_norm_proxy_sweep(label: &str, weights: &[f32], m: usize, k: usize) {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let n = weights.len();
        eprintln!("\n=== {label} (m={m}, k={k}, n={n}) ===");
        // Uniform Lloyd baseline.
        let ref_bytes = quantize_mq2g256_lloyd(weights, &signs1, &signs2);
        let ref_recon = dequantize_mq2g256_lloyd_to_f32(&ref_bytes, n, &signs1, &signs2);
        let ref_mse = mse(weights, &ref_recon);
        eprintln!("  Uniform Lloyd                MSE = {ref_mse:.6e}");
        // Weight-norm proxy imatrix.
        let col_imatrix = weight_norm_proxy_imatrix(weights, m, k);
        let proxy_bytes = quantize_mq2g256_lloyd_weighted(weights, &col_imatrix, &signs1, &signs2);
        let proxy_recon = dequantize_mq2g256_lloyd_to_f32(&proxy_bytes, n, &signs1, &signs2);
        let proxy_mse = mse(weights, &proxy_recon);
        let delta = ((proxy_mse - ref_mse) / ref_mse) * 100.0;
        eprintln!(
            "  Weight-norm-proxy Lloyd      MSE = {proxy_mse:.6e}  ({delta:+.2}% vs uniform)"
        );
    }

    /// Quantize via Lloyd WITHOUT the FWHT step — Lloyd applied directly
    /// to the natural (pre-rotation) weight distribution. Same 4-codepoint
    /// codebook + 2-bit indices.
    fn quantize_mq2g256_lloyd_no_fwht(f32_data: &[f32]) -> Vec<u8> {
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
                // NO FWHT — Lloyd directly on natural distribution.
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
                    let max_iter = 16;
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

    fn dequant_mq2_no_fwht(data: &[u8], n_weights: usize) -> Vec<f32> {
        let group_size = 256;
        let block_bytes = 72;
        let n_blocks = (n_weights + group_size - 1) / group_size;
        let mut out = vec![0.0f32; n_weights];
        for b in 0..n_blocks {
            let blk = &data[b * block_bytes..(b + 1) * block_bytes];
            let cb: [f32; 4] = [
                f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])),
                f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])),
                f16_to_f32(u16::from_le_bytes([blk[4], blk[5]])),
                f16_to_f32(u16::from_le_bytes([blk[6], blk[7]])),
            ];
            for i in 0..64 {
                let bv = blk[8 + i];
                for j in 0..4 {
                    let global_i = b * 256 + 4 * i + j;
                    if global_i < n_weights {
                        let idx = (bv >> (j * 2)) & 0x3;
                        out[global_i] = cb[idx as usize];
                    }
                }
            }
        }
        out
    }

    /// Quantize W (natural basis) with imatrix-weighted Lloyd, no FWHT.
    /// Returns (codebook, indices) — both in natural basis.
    fn lloyd_imatrix_no_fwht(weights: &[f32], col_weights: &[f32]) -> Vec<u8> {
        use rayon::prelude::*;
        let group_size = 256;
        let block_bytes = 72;
        let n = weights.len();
        let n_blocks = (n + group_size - 1) / group_size;
        let mut output = vec![0u8; n_blocks * block_bytes];
        let blocks_per_row = col_weights.len() / group_size;
        output
            .par_chunks_mut(block_bytes)
            .enumerate()
            .for_each(|(b, out_chunk)| {
                let start = b * group_size;
                let end = (start + group_size).min(n);
                let actual_len = end - start;
                let mut group = [0.0f32; 256];
                group[..actual_len].copy_from_slice(&weights[start..end]);
                // Use natural distribution; NO FWHT.
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
                    let max_iter = 16;
                    let mut prev_assignments = [0u8; 256];
                    for it in 0..max_iter {
                        let mut wsums = [0.0f64; 4];
                        let mut wtotals = [0.0f64; 4];
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
                            wsums[best] += pw * w as f64;
                            wtotals[best] += pw;
                        }
                        if it > 0 && changed == 0 {
                            break;
                        }
                        for k in 0..4 {
                            if wtotals[k] > 0.0 {
                                cb[k] = (wsums[k] / wtotals[k]) as f32;
                            }
                        }
                    }
                }
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

    fn dequant_no_fwht_natural(data: &[u8], n_weights: usize) -> Vec<f32> {
        let group_size = 256;
        let block_bytes = 72;
        let n_blocks = (n_weights + group_size - 1) / group_size;
        let mut out = vec![0.0f32; n_weights];
        for b in 0..n_blocks {
            let blk = &data[b * block_bytes..(b + 1) * block_bytes];
            let cb: [f32; 4] = [
                f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])),
                f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])),
                f16_to_f32(u16::from_le_bytes([blk[4], blk[5]])),
                f16_to_f32(u16::from_le_bytes([blk[6], blk[7]])),
            ];
            for i in 0..64 {
                let bv = blk[8 + i];
                for j in 0..4 {
                    let gi = b * 256 + 4 * i + j;
                    if gi < n_weights {
                        let idx = (bv >> (j * 2)) & 0x3;
                        out[gi] = cb[idx as usize];
                    }
                }
            }
        }
        out
    }

    fn gemv_f32(w: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; m];
        for r in 0..m {
            let mut acc = 0.0f64;
            for j in 0..k {
                acc += w[r * k + j] as f64 * x[j] as f64;
            }
            y[r] = acc as f32;
        }
        y
    }

    #[test]
    fn prefwht_imatrix_lloyd_value() {
        // Activation-weighted A/B test of post-FWHT vs pre-FWHT imatrix-Lloyd.
        // Generate W [m=256, k=4096] with HETEROGENEOUS column variances —
        // some columns have stddev=3, others stddev=0.1. Imatrix captures the
        // ground-truth importance. Run a gemv with this W against a random
        // unit-Gaussian X, then compare gemv-error for the two quant methods.
        //
        // If pre-FWHT-imatrix-Lloyd reduces gemv error meaningfully on
        // activations vs post-FWHT, that's the green light for the
        // pre-FWHT-Lloyd refactor (Action 5 in playbook).
        let m = 256;
        let k = 4096;
        let n = m * k;

        // Build heterogeneous-column W: column j has scale = log-uniform in
        // [0.1, 3.0] — gives 30x spread, mimics real LLM channel importance.
        let mut w = gaussian_samples(n, 0xc011c011);
        let mut col_scales = vec![0.0f32; k];
        let mut state: u64 = 0xc0ffeeed;
        for j in 0..k {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 11) & ((1u64 << 53) - 1)) as f64 / (1u64 << 53) as f64;
            // log-uniform in [0.1, 3.0]
            col_scales[j] = (0.1_f64.ln() + u * (3.0_f64.ln() - 0.1_f64.ln())).exp() as f32;
        }
        for r in 0..m {
            for j in 0..k {
                w[r * k + j] *= col_scales[j];
            }
        }
        // Imatrix: per-column 2-norm of W (mimics what a real activation
        // imatrix produces — bigger for important channels). Geomean-normalize.
        let mut imatrix = vec![0.0f32; k];
        for j in 0..k {
            let mut sum2 = 0.0f64;
            for r in 0..m {
                sum2 += (w[r * k + j] as f64).powi(2);
            }
            imatrix[j] = sum2.sqrt() as f32;
        }
        let mut sum_log = 0.0f64;
        for &v in &imatrix {
            sum_log += (v.max(1e-12) as f64).ln();
        }
        let mean_log = sum_log / k as f64;
        for v in imatrix.iter_mut() {
            *v = ((*v as f64).ln() - mean_log).exp() as f32;
        }

        // Random unit-Gaussian X for activations.
        let x = gaussian_samples(k, 0xacd1ac);
        let y_ref = gemv_f32(&w, &x, m, k);

        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);

        // METHOD A: post-FWHT imatrix-Lloyd (production).
        let bytes_a = quantize_mq2g256_lloyd_weighted(&w, &imatrix, &signs1, &signs2);
        let recon_a = dequantize_mq2g256_lloyd_to_f32(&bytes_a, n, &signs1, &signs2);
        let y_a = gemv_f32(&recon_a, &x, m, k);
        let err_a: f64 = y_ref
            .iter()
            .zip(y_a.iter())
            .map(|(r, q)| (*r as f64 - *q as f64).powi(2))
            .sum::<f64>()
            / m as f64;

        // METHOD B: pre-FWHT imatrix-Lloyd (proposed refactor).
        let bytes_b = lloyd_imatrix_no_fwht(&w, &imatrix);
        let recon_b = dequant_no_fwht_natural(&bytes_b, n);
        let y_b = gemv_f32(&recon_b, &x, m, k);
        let err_b: f64 = y_ref
            .iter()
            .zip(y_b.iter())
            .map(|(r, q)| (*r as f64 - *q as f64).powi(2))
            .sum::<f64>()
            / m as f64;

        // METHOD C: post-FWHT uniform Lloyd (current production w/o imatrix).
        let bytes_c = quantize_mq2g256_lloyd(&w, &signs1, &signs2);
        let recon_c = dequantize_mq2g256_lloyd_to_f32(&bytes_c, n, &signs1, &signs2);
        let y_c = gemv_f32(&recon_c, &x, m, k);
        let err_c: f64 = y_ref
            .iter()
            .zip(y_c.iter())
            .map(|(r, q)| (*r as f64 - *q as f64).powi(2))
            .sum::<f64>()
            / m as f64;

        eprintln!("\n=== Pre-FWHT vs post-FWHT imatrix-Lloyd (activation-weighted) ===");
        eprintln!("  W shape [{m}, {k}], heterogeneous column variances (0.1-3.0x)");
        eprintln!("  Method A: post-FWHT imatrix-Lloyd (current prod)   gemv MSE = {err_a:.6e}");
        eprintln!("  Method B: pre-FWHT  imatrix-Lloyd (proposed)       gemv MSE = {err_b:.6e}");
        eprintln!("  Method C: post-FWHT uniform Lloyd (no imatrix)     gemv MSE = {err_c:.6e}");
        eprintln!();
        let ab = ((err_b - err_a) / err_a) * 100.0;
        let ac = ((err_a - err_c) / err_c) * 100.0;
        let bc = ((err_b - err_c) / err_c) * 100.0;
        eprintln!("  Δ A→B (pre-FWHT win):              {ab:+.2}%");
        eprintln!("  Δ C→A (current imatrix vs uniform):{ac:+.2}%");
        eprintln!("  Δ C→B (pre-FWHT vs uniform):       {bc:+.2}%");
    }

    #[test]
    fn fwht_value_audit() {
        // Hypothesis: FWHT-rotation makes Lloyd more accurate because the
        // rotation decorrelates weights toward a Gaussian distribution, and
        // Lloyd's 4 codepoints are MSE-optimal for Gaussian.
        //
        // Test: quantize the SAME synthetic distribution two ways:
        //   A) Lloyd with FWHT (production path)
        //   B) Lloyd without FWHT (natural distribution)
        // Compute MSE for each. If FWHT wins consistently, the rotation is
        // earning its complexity. If they're close, dropping FWHT unblocks
        // proper imatrix integration (per
        // project_lloyd_imatrix_fwht_channel_mixing).
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);

        let cases: &[(&str, Box<dyn Fn() -> Vec<f32>>)] = &[
            (
                "Gaussian 16x256",
                Box::new(|| gaussian_samples(16 * 256, 0xc001cafe)),
            ),
            (
                "Heavy-tailed 16x256",
                Box::new(|| {
                    let mut htw = gaussian_samples(16 * 256, 0xfeed);
                    let tail = gaussian_samples((16 * 256) / 20, 0xbeef);
                    for (i, t) in tail.iter().enumerate() {
                        htw[i * 20] = t * 3.0;
                    }
                    htw
                }),
            ),
            (
                "Sparse + outliers 16x256",
                Box::new(|| {
                    let mut sw = gaussian_samples(16 * 256, 0x5_a55e);
                    for v in sw.iter_mut() {
                        *v *= 0.1;
                    }
                    for i in 0..(16 * 256 / 20) {
                        sw[i * 20] *= 30.0;
                    }
                    sw
                }),
            ),
            (
                "Bimodal (50% near -1, 50% near +1)",
                Box::new(|| {
                    let mut bw = gaussian_samples(16 * 256, 0xb1ba1);
                    for (i, v) in bw.iter_mut().enumerate() {
                        *v = 0.3 * *v + if i % 2 == 0 { -1.0 } else { 1.0 };
                    }
                    bw
                }),
            ),
        ];

        eprintln!("\n=== FWHT value audit ===");
        eprintln!(
            "{:35} {:>14} {:>14} {:>10}",
            "distribution", "fwht MSE", "no-fwht MSE", "fwht win %"
        );
        for (label, gen) in cases {
            let w = gen();
            let n = w.len();
            let fwht_bytes = quantize_mq2g256_lloyd(&w, &signs1, &signs2);
            let fwht_recon = dequantize_mq2g256_lloyd_to_f32(&fwht_bytes, n, &signs1, &signs2);
            let fwht_mse = mse(&w, &fwht_recon);
            let nofwht_bytes = quantize_mq2g256_lloyd_no_fwht(&w);
            let nofwht_recon = dequant_mq2_no_fwht(&nofwht_bytes, n);
            let nofwht_mse = mse(&w, &nofwht_recon);
            let win_pct = ((nofwht_mse - fwht_mse) / nofwht_mse) * 100.0;
            eprintln!(
                "{:35} {:14.6e} {:14.6e} {:+9.2}%",
                label, fwht_mse, nofwht_mse, win_pct
            );
        }
    }

    #[test]
    fn weight_norm_proxy_imatrix_sweep() {
        // Generate synthetic [m, k] matrices that mimic DeepSeek V4's expert
        // shapes (m=2048, k=4096 for gate; m=4096, k=2048 for down).
        // Use heavy-tailed and sparse-outlier variants to stress the
        // proxy.
        let m = 2048;
        let k = 4096;
        let n = m * k;
        eprintln!("\n=== Weight-norm proxy imatrix sweep ===");
        run_weight_norm_proxy_sweep(
            "Gaussian [2048, 4096]",
            &gaussian_samples(n, 0xc001cafe),
            m,
            k,
        );
        // Heavy-tailed: 5% of weights drawn from N(0, 3).
        let mut htw = gaussian_samples(n, 0xfeed);
        let tail_count = n / 20;
        let tail = gaussian_samples(tail_count, 0xbeef);
        for (i, t) in tail.iter().enumerate() {
            htw[i * 20] = t * 3.0;
        }
        run_weight_norm_proxy_sweep("Heavy-tailed [2048, 4096]", &htw, m, k);
        // Per-column variance heterogeneity: make column j scale with j/k.
        let mut col_het = gaussian_samples(n, 0xc011c011);
        for r in 0..m {
            for j in 0..k {
                let scale = 0.1 + 1.9 * (j as f32 / k as f32);
                col_het[r * k + j] *= scale;
            }
        }
        run_weight_norm_proxy_sweep("Per-column var heterogeneity", &col_het, m, k);
    }

    #[test]
    fn lloyd_iteration_headroom() {
        // The production 8-iter cap may or may not converge on heavy-tailed
        // distributions. Sweep niter ∈ {1, 2, 4, 8, 16, 32, 64} to find the
        // convergence floor — if 32 or 64 iter gives meaningfully lower
        // MSE than 8, that's free headroom (offline quant cost only).
        run_lloyd_iter_sweep("Gaussian 16x256", &gaussian_samples(16 * 256, 0xc001cafe));
        let mut htw = gaussian_samples(16 * 256, 0xfeed);
        let tail = gaussian_samples((16 * 256) / 20, 0xbeef);
        for (i, t) in tail.iter().enumerate() {
            htw[i * 20] = t * 3.0;
        }
        run_lloyd_iter_sweep("Heavy-tailed 16x256", &htw);
        let mut sw = gaussian_samples(16 * 256, 0x5_a55e);
        for v in sw.iter_mut() {
            *v *= 0.1;
        }
        for i in 0..(16 * 256 / 20) {
            sw[i * 20] *= 30.0;
        }
        run_lloyd_iter_sweep("Sparse + outliers 16x256", &sw);
    }

    #[test]
    fn sweep_deepseek4_like_distributions() {
        // 1) Pure Gaussian — baseline.
        run_one_distribution("N(0,1), 256 weights", &gaussian_samples(256, 0xc001cafe));

        // 2) Pure Gaussian, larger sample — averages across multiple blocks.
        run_one_distribution(
            "N(0,1), 16x256 weights",
            &gaussian_samples(16 * 256, 0xc001cafe),
        );

        // 3) Heavy-tailed mixture — 5% from N(0, 3), rest N(0, 1).
        //    Mimics DeepSeek V4's expert distributions with occasional outliers.
        let mut htw = gaussian_samples(16 * 256, 0xfeed);
        let tail = gaussian_samples((16 * 256) / 20, 0xbeef);
        for (i, t) in tail.iter().enumerate() {
            // Sprinkle the tail in every 20th slot.
            htw[i * 20] = t * 3.0;
        }
        run_one_distribution("Heavy-tailed, 16x256 weights", &htw);

        // 4) Sparse weights — most near zero, a few large. Sometimes
        //    happens in attention-related projections.
        let mut sw = gaussian_samples(16 * 256, 0x5_a55e);
        for v in sw.iter_mut() {
            *v *= 0.1;
        }
        // Inject 5% large values.
        for i in 0..(16 * 256 / 20) {
            sw[i * 20] *= 30.0;
        }
        run_one_distribution("Sparse (10% scale, 5% × 30 outliers)", &sw);
    }
}

/// Real-DeepSeek V4 per-block diagnostic. Reads an HFQ file directly via memmap2
/// (bypasses the hipfire-runtime hfq reader which currently has a broken
/// arch dep — keeps this probe self-contained inside hipfire-quantize).
/// For each MQ2-Lloyd (qt=19) and MQ3-Lloyd (qt=20) tensor, samples up to
/// MAX_SAMPLE_BLOCKS blocks and computes per-block stats:
///   - codebook range (max_cb - min_cb)
///   - codepoint spacing variance (how uneven the codebook is)
///   - index entropy (uniform = 2 bits for MQ2, log2(8)=3 for MQ3)
/// Then ranks tensors by mean per-block range to identify which tensors
/// have the highest dynamic range (= hardest to compress at given bpw).
///
/// Run with: cargo test --release -p hipfire-quantize --
///           --ignored hfq_block_range_diag -- --nocapture
///
/// Reads path from HIPFIRE_QUANT_DIAG_PATH env var (default points at
/// a local DeepSeek V4 HFQ snapshot).
#[cfg(test)]
mod hfq_block_diag {
    use super::*;
    use memmap2::Mmap;
    use std::fs::File;
    use std::path::Path;

    struct TensorInfo {
        name: String,
        quant_type: u8,
        shape: Vec<u32>,
        data_offset: usize,
        data_size: usize,
    }

    fn parse_hfq_metadata(path: &Path) -> std::io::Result<String> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        assert_eq!(&mmap[0..4], b"HFQM");
        let metadata_offset = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let data_offset = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;
        let mut depth: i32 = 0;
        let mut in_str = false;
        let mut esc = false;
        let mut json_end = 0usize;
        for (i, &b) in mmap[metadata_offset..data_offset].iter().enumerate() {
            if esc {
                esc = false;
                continue;
            }
            if in_str {
                if b == b'\\' {
                    esc = true;
                    continue;
                }
                if b == b'"' {
                    in_str = false;
                }
                continue;
            }
            if b == b'"' {
                in_str = true;
                continue;
            }
            if b == b'{' {
                depth += 1;
            }
            if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    json_end = i + 1;
                    break;
                }
            }
        }
        Ok(String::from_utf8_lossy(&mmap[metadata_offset..metadata_offset + json_end]).to_string())
    }

    #[test]
    #[ignore]
    fn hfq_dump_metadata() {
        let path_str = std::env::var("HIPFIRE_QUANT_DIAG_PATH")
            .unwrap_or_else(|_| "/data/hipfire-models/deepseek-v4-flash.mq2lloyd".to_string());
        let path = Path::new(&path_str);
        let json = parse_hfq_metadata(path).expect("parse");
        // Print just keys at top level + any "source" / "path" / "input" hints.
        eprintln!("=== Metadata from {path:?} (top 2000 chars) ===");
        let truncated: String = json.chars().take(2000).collect();
        eprintln!("{}", truncated);
        if json.len() > 2000 {
            eprintln!("... ({} chars total)", json.len());
        }
    }

    fn parse_hfq(path: &Path) -> std::io::Result<(Mmap, Vec<TensorInfo>)> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        assert_eq!(&mmap[0..4], b"HFQM", "Not HFQ");
        let n_tensors = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
        let metadata_offset = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let data_offset = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;
        // Find JSON end by brace-matching.
        let mut depth: i32 = 0;
        let mut in_str = false;
        let mut esc = false;
        let mut json_end = 0usize;
        for (i, &b) in mmap[metadata_offset..data_offset].iter().enumerate() {
            if esc {
                esc = false;
                continue;
            }
            if in_str {
                if b == b'\\' {
                    esc = true;
                    continue;
                }
                if b == b'"' {
                    in_str = false;
                }
                continue;
            }
            if b == b'"' {
                in_str = true;
                continue;
            }
            if b == b'{' {
                depth += 1;
            }
            if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    json_end = i + 1;
                    break;
                }
            }
        }
        let mut pos = metadata_offset + json_end;
        let idx_n = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()) as usize;
        assert_eq!(idx_n, n_tensors);
        pos += 4;
        let mut tensors = Vec::with_capacity(n_tensors);
        let mut cum = data_offset;
        for _ in 0..n_tensors {
            let name_len = u16::from_le_bytes(mmap[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            let name = String::from_utf8_lossy(&mmap[pos..pos + name_len]).into_owned();
            pos += name_len;
            let quant_type = mmap[pos];
            pos += 1;
            let n_dims = mmap[pos] as usize;
            pos += 1;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()));
                pos += 4;
            }
            // Skip group_size u32.
            pos += 4;
            let data_size = u64::from_le_bytes(mmap[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            tensors.push(TensorInfo {
                name,
                quant_type,
                shape,
                data_offset: cum,
                data_size,
            });
            cum += data_size;
        }
        Ok((mmap, tensors))
    }

    fn classify(name: &str) -> &'static str {
        if name.contains("ffn.experts.") && name.ends_with("w1.weight") {
            return "routed.w1 (gate)";
        }
        if name.contains("ffn.experts.") && name.ends_with("w2.weight") {
            return "routed.w2 (down)";
        }
        if name.contains("ffn.experts.") && name.ends_with("w3.weight") {
            return "routed.w3 (up)";
        }
        if name.contains("shared_experts.w1") {
            return "shared.w1";
        }
        if name.contains("shared_experts.w2") {
            return "shared.w2";
        }
        if name.contains("shared_experts.w3") {
            return "shared.w3";
        }
        if name.ends_with("attn.wq_a.weight") || name.ends_with("attn.wq_b.weight") {
            return "attn.q";
        }
        if name.ends_with("attn.wkv.weight") {
            return "attn.kv";
        }
        if name.ends_with("attn.wo_a.weight") || name.ends_with("attn.wo_b.weight") {
            return "attn.wo";
        }
        if name.contains("compressor.wkv") || name.contains("compressor.wgate") {
            return "compressor";
        }
        if name.contains("indexer.") {
            return "indexer";
        }
        "other"
    }

    /// Stats per block at MQ2 (4 codepoints, 8 B codebook + 64 B indices = 72 B/group).
    fn block_stats_mq2(data: &[u8]) -> Option<(f32, f32, f32)> {
        if data.len() < 8 {
            return None;
        }
        let mut cb = [0.0f32; 4];
        for k in 0..4 {
            cb[k] = f16_to_f32(u16::from_le_bytes([data[2 * k], data[2 * k + 1]]));
        }
        let lo = cb.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = cb.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = hi - lo;
        let mean = cb.iter().sum::<f32>() / 4.0;
        let spacing_var = cb.iter().map(|c| (c - mean).powi(2)).sum::<f32>() / 4.0;
        // Index histogram.
        let mut hist = [0u32; 4];
        for i in 0..64 {
            let b = data[8 + i];
            for j in 0..4 {
                hist[((b >> (j * 2)) & 0x3) as usize] += 1;
            }
        }
        let total: u32 = hist.iter().sum();
        let mut h_bits = 0.0f32;
        for &c in &hist {
            if c > 0 {
                let p = c as f32 / total as f32;
                h_bits -= p * p.log2();
            }
        }
        Some((range, spacing_var, h_bits))
    }

    /// Stats per block at MQ3 (8 codepoints, 16 B codebook + 96 B indices = 112 B/group).
    fn block_stats_mq3(data: &[u8]) -> Option<(f32, f32, f32)> {
        if data.len() < 16 {
            return None;
        }
        let mut cb = [0.0f32; 8];
        for k in 0..8 {
            cb[k] = f16_to_f32(u16::from_le_bytes([data[2 * k], data[2 * k + 1]]));
        }
        let lo = cb.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = cb.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = hi - lo;
        let mean = cb.iter().sum::<f32>() / 8.0;
        let spacing_var = cb.iter().map(|c| (c - mean).powi(2)).sum::<f32>() / 8.0;
        // Reconstruct indices.
        let mut hist = [0u32; 8];
        for chunk in 0..32 {
            let bo = 16 + chunk * 3;
            let b0 = data[bo];
            let b1 = data[bo + 1];
            let b2 = data[bo + 2];
            let q = [
                b0 & 7,
                (b0 >> 3) & 7,
                ((b0 >> 6) & 3) | ((b1 & 1) << 2),
                (b1 >> 1) & 7,
                (b1 >> 4) & 7,
                ((b1 >> 7) & 1) | ((b2 & 3) << 1),
                (b2 >> 2) & 7,
                (b2 >> 5) & 7,
            ];
            for v in q {
                hist[v as usize] += 1;
            }
        }
        let total: u32 = hist.iter().sum();
        let mut h_bits = 0.0f32;
        for &c in &hist {
            if c > 0 {
                let p = c as f32 / total as f32;
                h_bits -= p * p.log2();
            }
        }
        Some((range, spacing_var, h_bits))
    }

    fn cpu_inv_fwht_local(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
        super::cpu_inv_fwht_256(x, signs1, signs2);
    }

    fn dequant_mq3_lloyd(
        data: &[u8],
        n_weights: usize,
        signs1: &[f32],
        signs2: &[f32],
    ) -> Vec<f32> {
        let group_size = 256;
        let block_bytes = 112;
        let n_blocks = (n_weights + group_size - 1) / group_size;
        let mut out = vec![0.0f32; n_weights];
        for b in 0..n_blocks {
            let blk = &data[b * block_bytes..(b + 1) * block_bytes];
            let mut cb = [0.0f32; 8];
            for k in 0..8 {
                cb[k] = f16_to_f32(u16::from_le_bytes([blk[2 * k], blk[2 * k + 1]]));
            }
            let mut group = [0.0f32; 256];
            for chunk in 0..32 {
                let bo = 16 + chunk * 3;
                let b0 = blk[bo];
                let b1 = blk[bo + 1];
                let b2 = blk[bo + 2];
                let q = [
                    b0 & 7,
                    (b0 >> 3) & 7,
                    ((b0 >> 6) & 3) | ((b1 & 1) << 2),
                    (b1 >> 1) & 7,
                    (b1 >> 4) & 7,
                    ((b1 >> 7) & 1) | ((b2 & 3) << 1),
                    (b2 >> 2) & 7,
                    (b2 >> 5) & 7,
                ];
                for j in 0..8 {
                    group[chunk * 8 + j] = cb[q[j] as usize];
                }
            }
            cpu_inv_fwht_local(&mut group, signs1, signs2);
            let actual = (n_weights - b * 256).min(256);
            for j in 0..actual {
                out[b * 256 + j] = group[j];
            }
        }
        out
    }

    fn qt_name(qt: u8) -> &'static str {
        match qt {
            1 => "F16",
            2 => "F32",
            3 => "Q8F16",
            5 => "Q8HFQ",
            6 => "HFQ4G256",
            7 => "HFQ4G128",
            13 => "MQ4G256",
            14 => "MQ8G256",
            15 => "MQ6G256",
            17 => "MQ3G256",
            18 => "MQ2G256",
            19 => "MQ2G256Lloyd",
            20 => "MQ3G256Lloyd",
            21 => "HFP4G32",
            24 => "MFP4G32",
            34 => "MFP4G32E8",
            35 => "MFP4G32E8SOA",
            36 => "MFP3G32E8",
            37 => "MFP2G32E8",
            _ => "?",
        }
    }

    /// Sample a real DeepSeek V4 MQ2-Lloyd tensor, dequant a few blocks, and
    /// report the distribution shape. Compares against the synthetic
    /// distributions used in fwht_value_audit + GPTQ probes to see which
    /// our DeepSeek V4 weights actually resemble.
    #[test]
    #[ignore]
    fn hfq_dist_sample() {
        let path_str = std::env::var("HIPFIRE_QUANT_DIAG_PATH")
            .unwrap_or_else(|_| "/data/hipfire-models/deepseek-v4-flash.mq2lloyd".to_string());
        let path = Path::new(&path_str);
        let (mmap, tensors) = parse_hfq(path).expect("parse hfq");

        // Take 8 different routed-expert tensors (w1, w2, w3 from a few
        // layers/experts) and one attention tensor + one shared tensor.
        let sample_names = [
            "layers.5.ffn.experts.0.w1.weight",   // gate (mid layer)
            "layers.5.ffn.experts.0.w2.weight",   // down
            "layers.5.ffn.experts.0.w3.weight",   // up
            "layers.20.ffn.experts.50.w1.weight", // gate (later layer)
            "layers.20.ffn.experts.50.w2.weight",
            "layers.40.ffn.experts.100.w2.weight", // down (deep layer)
            "layers.5.ffn.shared_experts.w2.weight", // shared down
            "layers.5.attn.wo_b.weight",           // attention output
        ];
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);

        eprintln!("\n=== Real DeepSeek V4 weight distribution stats (4096 weights per tensor) ===");
        eprintln!(
            "{:55} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "tensor", "qt", "mean", "stddev", "p99/sd", "kurtosis"
        );
        for sname in sample_names {
            let t_idx = tensors.iter().position(|t| t.name == sname);
            let t = match t_idx {
                Some(i) => &tensors[i],
                None => continue,
            };
            // Sample first 16 blocks = 4096 weights. Skip unsupported qts.
            let block_bytes = match t.quant_type {
                19 => 72,
                20 => 112,
                3 => 34,
                _ => {
                    eprintln!("  {:55} {:>2} (skip qt)", sname, t.quant_type);
                    continue;
                }
            };
            let n_blocks = (t.data_size / block_bytes).min(16);
            if n_blocks == 0 {
                continue;
            }
            let n_w = n_blocks * 256;
            let recon: Vec<f32> = if t.quant_type == 19 {
                let raw = &mmap[t.data_offset..t.data_offset + n_blocks * 72];
                super::dequantize_mq2g256_lloyd_to_f32(raw, n_w, &signs1, &signs2)
            } else if t.quant_type == 20 {
                let raw = &mmap[t.data_offset..t.data_offset + n_blocks * 112];
                dequant_mq3_lloyd(raw, n_w, &signs1, &signs2)
            } else {
                eprintln!(
                    "  {:55} {:>2} (unsupported qt for dequant, skipping)",
                    sname, t.quant_type
                );
                continue;
            };
            // Compute stats.
            let n = recon.len() as f64;
            let mean = recon.iter().map(|&x| x as f64).sum::<f64>() / n;
            let var = recon
                .iter()
                .map(|&x| (x as f64 - mean).powi(2))
                .sum::<f64>()
                / n;
            let stddev = var.sqrt();
            // Kurtosis (Pearson) — measures heavy-tailedness; Gaussian = 3.
            let mu4 = recon
                .iter()
                .map(|&x| (x as f64 - mean).powi(4))
                .sum::<f64>()
                / n;
            let kurt = mu4 / var.powi(2);
            // p99/sd — ratio of 99th percentile abs value to sd.
            let mut absvals: Vec<f64> = recon.iter().map(|&x| (x as f64 - mean).abs()).collect();
            absvals
                .sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p99 = absvals[(absvals.len() * 99 / 100).min(absvals.len() - 1)];
            let p99_over_sd = p99 / stddev;
            eprintln!(
                "{:55} {:>2} {:>10.4e} {:>10.4e} {:>10.3} {:>10.3}",
                sname, t.quant_type, mean, stddev, p99_over_sd, kurt
            );
        }
        // Reference values from synthetic distributions:
        eprintln!("\nReference (synthetic):");
        eprintln!("  Gaussian:            p99/sd ≈ 2.33    kurtosis ≈ 3.0");
        eprintln!("  Heavy-tailed (5% × 3): p99/sd ≈ 2.5-3   kurtosis ≈ 3-6");
        eprintln!("  Sparse outliers:     p99/sd ≈ 10+     kurtosis ≈ 30+");
        eprintln!("  Bimodal:             p99/sd ≈ 1.5-2   kurtosis < 3 (platykurtic)");
    }

    #[test]
    #[ignore]
    fn hfq_inventory() {
        let path_str = std::env::var("HIPFIRE_QUANT_DIAG_PATH")
            .unwrap_or_else(|_| "/data/hipfire-models/deepseek-v4-flash.mq2lloyd".to_string());
        let path = Path::new(&path_str);
        eprintln!("opening {path:?}");
        let (_mmap, tensors) = parse_hfq(path).expect("parse hfq");
        eprintln!("{} tensors", tensors.len());
        // Bucket by (family, qt).
        use std::collections::BTreeMap;
        let mut counts: BTreeMap<(&'static str, u8), (u64, u64)> = BTreeMap::new();
        let mut total_bytes: u64 = 0;
        for t in &tensors {
            let fam = classify(&t.name);
            let e = counts.entry((fam, t.quant_type)).or_default();
            e.0 += 1;
            e.1 += t.data_size as u64;
            total_bytes += t.data_size as u64;
        }
        eprintln!(
            "{:30} {:>14} {:>8} {:>14}",
            "family", "qt", "count", "bytes"
        );
        for ((fam, qt), (cnt, bytes)) in &counts {
            eprintln!(
                "{:30} {:>2} {:12} {:>8} {:>14}",
                fam,
                qt,
                qt_name(*qt),
                cnt,
                bytes
            );
        }
        eprintln!(
            "\ntotal data bytes: {} ({:.2} GiB)",
            total_bytes,
            total_bytes as f64 / (1024.0_f64.powi(3))
        );
    }

    #[test]
    #[ignore]
    fn hfq_block_range_diag() {
        let path_str = std::env::var("HIPFIRE_QUANT_DIAG_PATH")
            .unwrap_or_else(|_| "/data/hipfire-models/deepseek-v4-flash.mq2lloyd".to_string());
        let path = Path::new(&path_str);
        eprintln!("opening {path:?}");
        let (mmap, tensors) = parse_hfq(path).expect("parse hfq");
        eprintln!("{} tensors, file mapped", tensors.len());

        // Bucket by (family, qt) → list of (mean_range, mean_var, mean_entropy, n_blocks).
        use std::collections::BTreeMap;
        let mut buckets: BTreeMap<(&'static str, u8), Vec<(f32, f32, f32, usize)>> =
            BTreeMap::new();

        // Sample at most this many blocks per tensor; routed-expert tensors are
        // huge (~1 MB each in the layer's batched blob form, 256 experts × 43
        // layers = ~30k tensors). Cap CPU time.
        const MAX_BLOCKS_PER_TENSOR: usize = 64;

        for t in &tensors {
            if t.quant_type != 19 && t.quant_type != 20 {
                continue;
            }
            let block_bytes = if t.quant_type == 19 { 72 } else { 112 };
            let raw = &mmap[t.data_offset..t.data_offset + t.data_size];
            let n_blocks = t.data_size / block_bytes;
            if n_blocks == 0 {
                continue;
            }
            let stride = (n_blocks / MAX_BLOCKS_PER_TENSOR.min(n_blocks)).max(1);
            let mut sum_range = 0.0f64;
            let mut sum_var = 0.0f64;
            let mut sum_h = 0.0f64;
            let mut n_sampled = 0usize;
            let mut bi = 0;
            while bi < n_blocks {
                let blk = &raw[bi * block_bytes..(bi + 1) * block_bytes];
                let stats = if t.quant_type == 19 {
                    block_stats_mq2(blk)
                } else {
                    block_stats_mq3(blk)
                };
                if let Some((r, v, h)) = stats {
                    sum_range += r as f64;
                    sum_var += v as f64;
                    sum_h += h as f64;
                    n_sampled += 1;
                }
                bi += stride;
            }
            if n_sampled == 0 {
                continue;
            }
            let fam = classify(&t.name);
            buckets.entry((fam, t.quant_type)).or_default().push((
                (sum_range / n_sampled as f64) as f32,
                (sum_var / n_sampled as f64) as f32,
                (sum_h / n_sampled as f64) as f32,
                n_sampled,
            ));
        }

        eprintln!("\n=== Per-family block stats (sampled {MAX_BLOCKS_PER_TENSOR}/tensor) ===");
        eprintln!(
            "{:30} {:3} {:>6} {:>10} {:>10} {:>10}",
            "family", "qt", "tensors", "mean_range", "mean_var", "mean_entropy"
        );
        for ((fam, qt), entries) in &buckets {
            let n_tensors = entries.len();
            let mean_range =
                entries.iter().map(|(r, _, _, _)| *r as f64).sum::<f64>() / n_tensors as f64;
            let mean_var =
                entries.iter().map(|(_, v, _, _)| *v as f64).sum::<f64>() / n_tensors as f64;
            let mean_h =
                entries.iter().map(|(_, _, h, _)| *h as f64).sum::<f64>() / n_tensors as f64;
            eprintln!(
                "{:30} {:3} {:>6} {:>10.4} {:>10.4} {:>10.4}",
                fam, qt, n_tensors, mean_range, mean_var, mean_h
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The MQ*-G256-GL codebooks are NOT stored in the `.hfq` file: the encoder
    /// bakes them in via `gl_encode_block(&GL_CB2 | &GL_CB3, ..)` and the runtime
    /// re-supplies them as scalar kernel args from
    /// `rdna_compute::{GL_CB2, GL_CB3}`. Drift between the two arrays is a
    /// SILENT accuracy failure — every weight decodes to a plausible-but-wrong
    /// level and nothing errors. This test is the only thing standing between
    /// the two copies, so keep it.
    ///
    /// If it fails: fix whichever side you did NOT intend to change. Bumping the
    /// codebook is a FORMAT CHANGE — existing qt-38/39 `.hfq` files decode wrong
    /// against a new codebook, so it needs a new quant_type, not an edit.
    #[test]
    fn gl_codebooks_match_runtime() {
        assert_eq!(
            GL_CB2,
            rdna_compute::GL_CB2,
            "hipfire-quantize GL_CB2 has drifted from rdna_compute::GL_CB2 — \
             qt=38 weights would decode against the wrong 2-bit levels"
        );
        assert_eq!(
            GL_CB3,
            rdna_compute::GL_CB3,
            "hipfire-quantize GL_CB3 has drifted from rdna_compute::GL_CB3 — \
             qt=39 weights would decode against the wrong 3-bit levels"
        );
    }

    /// Pins the GL on-disk geometry the runtime loader + kernels assume:
    /// SoA `[m*gpr*IDX B indices][m*gpr*2 B fp16 scales]`, IDX = 64 (MQ2-GL) /
    /// 96 (MQ3-GL). The kernels derive the scale-region base as `M*gpr*IDX`, so
    /// a size change here is a silent read past/short of the scales.
    #[test]
    fn gl_blob_layout_matches_runtime_constants() {
        let (m, k) = (4usize, 512usize);
        let gpr = k / 256;
        let signs1 = vec![1.0f32; 256];
        let signs2 = vec![1.0f32; 256];
        let w: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.01 - 0.08).collect();

        let b2 = quantize_mq2g256gl(&w, m, k, &signs1, &signs2);
        assert_eq!(
            b2.len(),
            m * gpr * (rdna_compute::GL_MQ2_GROUP_IDX_BYTES + rdna_compute::GL_GROUP_SCALE_BYTES),
            "MQ2G256GL blob size disagrees with GL_MQ2_GROUP_IDX_BYTES/GL_GROUP_SCALE_BYTES"
        );

        let b3 = quantize_mq3g256gl(&w, m, k, &signs1, &signs2);
        assert_eq!(
            b3.len(),
            m * gpr * (rdna_compute::GL_MQ3_GROUP_IDX_BYTES + rdna_compute::GL_GROUP_SCALE_BYTES),
            "MQ3G256GL blob size disagrees with GL_MQ3_GROUP_IDX_BYTES/GL_GROUP_SCALE_BYTES"
        );
    }

    #[test]
    fn e2m1_lookup_matches_ocp_spec() {
        // OCP MX FP4 (E2M1) spec values for the 8 magnitude codes.
        // Sign bit (0x8) flips sign of the magnitude.
        let expected: &[(u8, f32)] = &[
            (0x0, 0.0),
            (0x1, 0.5),
            (0x2, 1.0),
            (0x3, 1.5),
            (0x4, 2.0),
            (0x5, 3.0),
            (0x6, 4.0),
            (0x7, 6.0),
            (0x8, -0.0),
            (0x9, -0.5),
            (0xA, -1.0),
            (0xB, -1.5),
            (0xC, -2.0),
            (0xD, -3.0),
            (0xE, -4.0),
            (0xF, -6.0),
        ];
        for &(nib, want) in expected {
            assert_eq!(
                e2m1_to_f32(nib),
                want,
                "e2m1_to_f32(0x{:x}) = {} want {}",
                nib,
                e2m1_to_f32(nib),
                want
            );
        }
    }

    #[test]
    fn e2m1_dequant_unpacks_nibbles_and_doubles_logical_cols() {
        // Storage: 1 row × 1 col-byte. Byte = 0x42 → low nibble 0x2 (=1.0),
        // high nibble 0x4 (=2.0). Scale: 1 row × 1 col, UE8M0=127 (=2^0=1.0).
        // → logical row should be [1.0, 2.0] (length 2).
        let (vals, shape) = dequantize_e2m1_ue8m0_to_f32(&[0x42], &[1, 1], &[127], &[1, 1]);
        assert_eq!(shape, vec![1, 2]);
        assert_eq!(vals, vec![1.0, 2.0]);
    }

    #[test]
    fn e2m1_dequant_applies_ue8m0_scale() {
        // Byte = 0x12 → low=2 (=1.0), high=1 (=0.5). Scale byte 128 → 2^1=2.0.
        // → logical [2.0, 1.0].
        let (vals, _) = dequantize_e2m1_ue8m0_to_f32(&[0x12], &[1, 1], &[128], &[1, 1]);
        assert_eq!(vals, vec![2.0, 1.0]);
    }

    #[test]
    fn parse_layer_idx_safetensors_dense() {
        assert_eq!(
            parse_layer_idx("model.layers.0.self_attn.q_proj.weight"),
            Some(0)
        );
        assert_eq!(
            parse_layer_idx("model.layers.63.mlp.gate_proj.weight"),
            Some(63)
        );
    }

    #[test]
    fn parse_layer_idx_safetensors_moe() {
        assert_eq!(
            parse_layer_idx("model.language_model.layers.5.mlp.experts.0.gate_up_proj.weight"),
            Some(5)
        );
    }

    #[test]
    fn parse_layer_idx_gguf() {
        assert_eq!(parse_layer_idx("blk.0.attn_q.weight"), Some(0));
        assert_eq!(parse_layer_idx("blk.31.ffn_gate.weight"), Some(31));
    }

    #[test]
    fn parse_layer_idx_no_match() {
        assert_eq!(parse_layer_idx("token_embd.weight"), None);
        assert_eq!(parse_layer_idx("output.weight"), None);
    }

    #[test]
    fn kmap_norms_are_f16() {
        assert_eq!(
            kmap_resolve("model.layers.0.input_layernorm.weight", 64, false),
            QuantLevel::F16
        );
        assert_eq!(
            kmap_resolve("model.layers.30.post_attention_layernorm.weight", 64, false),
            QuantLevel::F16
        );
    }

    #[test]
    fn kmap_embeds_are_q8() {
        assert_eq!(
            kmap_resolve("model.embed_tokens.weight", 64, false),
            QuantLevel::Q8
        );
        assert_eq!(kmap_resolve("lm_head.weight", 64, false), QuantLevel::Q8);
        assert_eq!(kmap_resolve("output.weight", 64, false), QuantLevel::Q8);
    }

    #[test]
    fn kmap_moe_router_q8() {
        assert_eq!(
            kmap_resolve("model.language_model.layers.5.mlp.gate.weight", 64, true),
            QuantLevel::Q8
        );
        assert_eq!(
            kmap_resolve(
                "model.language_model.layers.5.mlp.shared_expert_gate.weight",
                64,
                true
            ),
            QuantLevel::Q8
        );
    }

    #[test]
    fn kmap_moe_router_not_promoted_on_dense() {
        // On a dense model, mlp.gate.weight is not a router — falls to edge/base
        assert_ne!(
            kmap_resolve("model.layers.30.mlp.gate.weight", 64, false),
            QuantLevel::Q8
        );
    }

    #[test]
    fn kmap_moe_expert_ffn_promote6() {
        assert_eq!(
            kmap_resolve(
                "model.language_model.layers.30.mlp.experts.5.gate_up_proj.weight",
                64,
                true
            ),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve(
                "model.language_model.layers.30.mlp.experts.5.down_proj.weight",
                64,
                true
            ),
            QuantLevel::Promote6
        );
    }

    #[test]
    fn kmap_edge_layers_dense_ffn_only() {
        // Dense: FFN in edge layers — promoted
        assert_eq!(
            kmap_resolve("model.layers.0.mlp.gate_proj.weight", 64, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.1.mlp.down_proj.weight", 64, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.62.mlp.up_proj.weight", 64, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.63.mlp.down_proj.weight", 64, false),
            QuantLevel::Promote6
        );
        // Dense: attn in edge layers — NOT promoted
        assert_eq!(
            kmap_resolve("model.layers.0.self_attn.q_proj.weight", 64, false),
            QuantLevel::Base
        );
        assert_eq!(
            kmap_resolve("model.layers.63.self_attn.v_proj.weight", 64, false),
            QuantLevel::Base
        );
        assert_eq!(
            kmap_resolve("model.layers.0.linear_attn.in_proj_qkv.weight", 64, false),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_edge_layers_moe_attn_and_ffn() {
        // MoE: both attn and FFN in edge layers — promoted
        assert_eq!(
            kmap_resolve(
                "model.language_model.layers.0.self_attn.q_proj.weight",
                64,
                true
            ),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve(
                "model.language_model.layers.0.mlp.gate_proj.weight",
                64,
                true
            ),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve(
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                64,
                true
            ),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve(
                "model.language_model.layers.63.self_attn.v_proj.weight",
                64,
                true
            ),
            QuantLevel::Promote6
        );
    }

    #[test]
    fn kmap_middle_layers_base() {
        assert_eq!(
            kmap_resolve("model.layers.2.self_attn.q_proj.weight", 64, false),
            QuantLevel::Base
        );
        assert_eq!(
            kmap_resolve("model.layers.30.mlp.gate_proj.weight", 64, false),
            QuantLevel::Base
        );
        assert_eq!(
            kmap_resolve("model.layers.61.mlp.down_proj.weight", 64, false),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_edge_layers_small_model_24_layers() {
        // 24 layers: edge = 0,1 and 22,23
        assert_eq!(
            kmap_resolve("model.layers.0.mlp.gate_proj.weight", 24, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.1.mlp.gate_proj.weight", 24, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.2.mlp.gate_proj.weight", 24, false),
            QuantLevel::Base
        );
        assert_eq!(
            kmap_resolve("model.layers.22.mlp.gate_proj.weight", 24, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.23.mlp.gate_proj.weight", 24, false),
            QuantLevel::Promote6
        );
    }

    #[test]
    fn kmap_n_layers_zero_disables_edge() {
        assert_eq!(
            kmap_resolve("model.layers.0.mlp.gate_proj.weight", 0, false),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_edge_layers_tiny_model_3_layers() {
        // 3 layers: first-2 = {0,1}, last-2 = {1,2}. All layers promoted.
        assert_eq!(
            kmap_resolve("model.layers.0.mlp.gate_proj.weight", 3, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.1.mlp.gate_proj.weight", 3, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.2.mlp.gate_proj.weight", 3, false),
            QuantLevel::Promote6
        );
    }

    #[test]
    fn kmap_expert_not_promoted_on_dense() {
        // "mlp.experts." in name but is_moe=false — should NOT trigger rule 4
        assert_eq!(
            kmap_resolve(
                "model.layers.30.mlp.experts.5.gate_up_proj.weight",
                64,
                false
            ),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_gguf_names() {
        // GGUF edge-layer FFN (dense) — promoted
        assert_eq!(
            kmap_resolve("blk.0.ffn_gate.weight", 64, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("blk.63.ffn_gate.weight", 64, false),
            QuantLevel::Promote6
        );
        // GGUF edge-layer attn (dense) — NOT promoted
        assert_eq!(
            kmap_resolve("blk.0.attn_q.weight", 64, false),
            QuantLevel::Base
        );
        // GGUF edge-layer attn (MoE) — promoted
        assert_eq!(
            kmap_resolve("blk.0.attn_q.weight", 64, true),
            QuantLevel::Promote6
        );
        // GGUF middle-layer — base
        assert_eq!(
            kmap_resolve("blk.30.ffn_gate.weight", 64, false),
            QuantLevel::Base
        );
    }

    // ── Alternating mode tests ───────────────────────────────────────────

    #[test]
    fn positional_promote_edges() {
        assert!(is_positional_promote(0, 40, 3));
        assert!(is_positional_promote(1, 40, 3));
        assert!(is_positional_promote(38, 40, 3));
        assert!(is_positional_promote(39, 40, 3));
    }

    #[test]
    fn positional_promote_stride3() {
        // Middle layers: every 3rd starting from idx 2
        assert!(is_positional_promote(2, 40, 3)); // edge
        assert!(!is_positional_promote(3, 40, 3));
        assert!(!is_positional_promote(4, 40, 3));
        assert!(is_positional_promote(5, 40, 3));
        assert!(!is_positional_promote(6, 40, 3));
        assert!(!is_positional_promote(7, 40, 3));
        assert!(is_positional_promote(8, 40, 3));
    }

    #[test]
    fn kmap_alternating_moe_experts() {
        // MoE experts: promoted in positional layers, base in others
        assert_eq!(
            kmap_resolve_mode(
                "model.language_model.layers.0.mlp.experts.5.gate_up_proj.weight",
                40,
                true,
                1
            ),
            QuantLevel::Promote6 // edge layer
        );
        assert_eq!(
            kmap_resolve_mode(
                "model.language_model.layers.5.mlp.experts.5.gate_up_proj.weight",
                40,
                true,
                1
            ),
            QuantLevel::Promote6 // stride hit (5-2=3, 3%3==0)
        );
        assert_eq!(
            kmap_resolve_mode(
                "model.language_model.layers.3.mlp.experts.5.gate_up_proj.weight",
                40,
                true,
                1
            ),
            QuantLevel::Base // not on stride
        );
    }

    #[test]
    fn kmap_alternating_ffn_down() {
        // ffn_down promoted in positional layers, base in others
        assert_eq!(
            kmap_resolve_mode("model.layers.0.mlp.down_proj.weight", 40, false, 1),
            QuantLevel::Promote6 // edge
        );
        assert_eq!(
            kmap_resolve_mode("model.layers.5.mlp.down_proj.weight", 40, false, 1),
            QuantLevel::Promote6 // stride
        );
        assert_eq!(
            kmap_resolve_mode("model.layers.3.mlp.down_proj.weight", 40, false, 1),
            QuantLevel::Base // not on stride
        );
        // gate_proj NOT promoted in middle layers
        assert_eq!(
            kmap_resolve_mode("model.layers.5.mlp.gate_proj.weight", 40, false, 1),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_alternating_n_layers_zero() {
        // With n_layers=0, alternating mode should return Base for everything
        assert_eq!(
            kmap_resolve_mode("model.layers.0.mlp.down_proj.weight", 0, false, 1),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_alternating_gguf_names() {
        // GGUF ffn_down in edge layer
        assert_eq!(
            kmap_resolve_mode("blk.0.ffn_down.weight", 40, false, 1),
            QuantLevel::Promote6
        );
        // GGUF ffn_down in middle non-stride layer
        assert_eq!(
            kmap_resolve_mode("blk.3.ffn_down.weight", 40, false, 1),
            QuantLevel::Base
        );
        // GGUF ffn_gate stays base in middle
        assert_eq!(
            kmap_resolve_mode("blk.5.ffn_gate.weight", 40, false, 1),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_typed_promotes_down_and_v() {
        assert_eq!(
            kmap_resolve_mode("model.layers.15.mlp.down_proj.weight", 40, false, 2),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve_mode("model.layers.15.self_attn.v_proj.weight", 40, false, 2),
            QuantLevel::Promote6
        );
        // gate_proj stays base
        assert_eq!(
            kmap_resolve_mode("model.layers.15.mlp.gate_proj.weight", 40, false, 2),
            QuantLevel::Base
        );
    }
}
