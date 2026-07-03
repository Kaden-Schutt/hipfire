// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Tensor payload encoders + a post-process quantizer for diffusion `.hfq`
//! artifacts. Reads a source artifact (whose weights decode to f32 via
//! [`CpuTensor::from_hfq`]), re-encodes the large 2D+ `.weight` tensors into a
//! packed format, and copies every other entry (biases, norms, configs,
//! tokenizers) through verbatim. The decode path keys purely off each tensor's
//! `quant_type`, so the resulting artifact loads with no metadata changes beyond
//! the informational `weight_format` string.

use super::*;
use crate::quant_decode::{OQ_FWHT_SEED1, OQ_FWHT_SEED2};
use hipfire_quantize::codecs::{quantize_oq4g256, quantize_oq8g256};
use hipfire_quantize::gen_fwht_signs;
pub use hipfire_quantize::hessian_io::HessianSidecar;
use hipfire_quantize::ldlq::oq4_ldlq_pack;
use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqFile, HfqMemTensor};
use std::path::Path;

/// Quantization formats this tool can emit. Both round-trip bit-exactly with the
/// matching decoder in `quant_decode.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionQuantFormat {
    /// q8_0: per-32 group, symmetric int8 with an f16 scale (34 bytes/block).
    Q8F16,
    /// Q4F16_G64: per-64 group, affine 4-bit with f16 scale+min (36 bytes/block).
    Q4F16G64,
    /// Q4F16_G64 storage, but encoded with per-group MSE clip-search (the `+` of
    /// `oq4+`): instead of using the raw group min/max, search the quantization
    /// range that minimizes reconstruction error, trading clipped outliers for
    /// finer resolution on the bulk of the distribution. Data-free.
    Q4F16G64Clip,
    /// Q4_K (llama.cpp k-quant): 256-superblock, 8x 32-element sub-blocks each
    /// with its own 6-bit scale+min under a per-superblock f16 d/dmin. Reuses the
    /// hipfire LLM-path codec; finer/hierarchical vs the flat group-64 affine.
    Q4K,
    /// Opus Quant 4-bit (FWHT-rotated, 256-group), RTN. Linears -> oq4, convs ->
    /// oq8 (convs are sensitive and have no Hessian). Data-free.
    Oq4,
    /// Opus Quant 4-bit, activation-calibrated: linears use LDLQ Hessian error
    /// feedback (oq4_ldlq_pack) when a `.calib.hfq` Hessian is available, else
    /// fall back to oq4 RTN; convs -> oq8. Requires `--calib`.
    Oq4PlusPlus,
    /// Opus Quant 8-bit (FWHT-rotated, 256-group), RTN. Near-lossless.
    Oq8,
}

impl DiffusionQuantFormat {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "q8" | "q8f16" | "q80" => Some(Self::Q8F16),
            "q4" | "q4f16" | "q4f16g64" => Some(Self::Q4F16G64),
            "q4+" | "q4c" | "q4clip" | "q4f16g64clip" => Some(Self::Q4F16G64Clip),
            "q4k" | "q4_k" => Some(Self::Q4K),
            "oq4" => Some(Self::Oq4),
            "oq4+" | "oq4++" => Some(Self::Oq4PlusPlus),
            "oq8" => Some(Self::Oq8),
            _ => None,
        }
    }

    /// Opus formats are FWHT-rotated, 256-group, and quantize per-tensor (linears
    /// vs convs differ), so they bypass the single-format `encode`/`quant_type`.
    fn is_opus(self) -> bool {
        matches!(self, Self::Oq4 | Self::Oq4PlusPlus | Self::Oq8)
    }

    fn quant_type(self) -> u8 {
        match self {
            Self::Q8F16 => QT_DIFFUSION_TENSOR_Q8F16,
            Self::Q4F16G64 | Self::Q4F16G64Clip => QT_DIFFUSION_TENSOR_Q4F16_G64,
            Self::Q4K => QT_DIFFUSION_TENSOR_Q4_K,
            Self::Oq4 | Self::Oq4PlusPlus | Self::Oq8 => QT_DIFFUSION_TENSOR_OQ4_G256,
        }
    }

    fn group_size(self) -> u32 {
        match self {
            Self::Q8F16 => 32,
            Self::Q4F16G64 | Self::Q4F16G64Clip => 64,
            Self::Q4K | Self::Oq4 | Self::Oq4PlusPlus | Self::Oq8 => 256,
        }
    }

    fn weight_format_label(self) -> &'static str {
        match self {
            Self::Q8F16 => "q8",
            Self::Q4F16G64 => "q4",
            Self::Q4F16G64Clip => "q4+",
            Self::Q4K => "q4k",
            Self::Oq4 => "oq4",
            Self::Oq4PlusPlus => "oq4++",
            Self::Oq8 => "oq8",
        }
    }

    fn encode(self, data: &[f32]) -> Vec<u8> {
        match self {
            Self::Q8F16 => encode_q8f16(data),
            Self::Q4F16G64 => encode_q4f16_g64(data),
            Self::Q4F16G64Clip => encode_q4f16_g64_clipsearch(data),
            Self::Q4K => encode_q4k(data),
            // Opus formats are handled per-tensor in quantize_diffusion_hfq.
            Self::Oq4 | Self::Oq4PlusPlus | Self::Oq8 => unreachable!("opus uses encode_opus_tensor"),
        }
    }
}

/// Per-tensor Opus (oq4/oq8) encoding with optional LDLQ calibration.
/// Returns `(quant_type, group_size, bytes, ldlq_used)`. Convs (rank-4) and full
/// `oq8` runs go to oq8; linears go to oq4 (LDLQ when a Hessian is available,
/// else RTN).
#[allow(clippy::too_many_arguments)]
fn encode_opus_tensor(
    format: DiffusionQuantFormat,
    name: &str,
    shape: &[u32],
    data: &[f32],
    calib: Option<&HessianSidecar>,
    signs1: &[f32],
    signs2: &[f32],
) -> (u8, u32, Vec<u8>, bool) {
    let is_conv = shape.len() == 4;
    if is_conv || matches!(format, DiffusionQuantFormat::Oq8) {
        let bytes = quantize_oq8g256(data, signs1, signs2);
        return (QT_DIFFUSION_TENSOR_OQ8_G256, 256, bytes, false);
    }
    // Linear (rank-2) at oq4 / oq4++.
    let m = shape[0] as usize;
    let k = shape[1] as usize;
    if matches!(format, DiffusionQuantFormat::Oq4PlusPlus) && k % 256 == 0 && m * k == data.len() {
        if let Some(sc) = calib {
            let base = name.strip_suffix(".weight").unwrap_or(name);
            if let Some(href) = sc.get(base, 0) {
                if href.k == k {
                    let h: Vec<f32> = (0..k * k).map(|i| href.at(i / k, i % k) as f32).collect();
                    let diag: f64 = (0..k).map(|i| h[i * k + i] as f64).sum();
                    let damp = 0.01 * (diag / k as f64).max(1e-12);
                    if let Some(packed) = oq4_ldlq_pack(data, m, k, &h, signs1, signs2, damp) {
                        return (QT_DIFFUSION_TENSOR_OQ4_G256, 256, packed, true);
                    }
                }
            }
        }
    }
    let bytes = quantize_oq4g256(data, signs1, signs2);
    (QT_DIFFUSION_TENSOR_OQ4_G256, 256, bytes, false)
}

/// q8_0 encoder: groups of 32, symmetric int8, `scale = max_abs / 127`, stored
/// as `[f16 scale][32 x i8]` (34 bytes/block). Mirrors `dequant_q8f16`.
pub(crate) fn encode_q8f16(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len().div_ceil(32) * 34);
    for group in data.chunks(32) {
        let max_abs = group.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        bytes.extend_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
        for idx in 0..32 {
            let value = group.get(idx).copied().unwrap_or(0.0);
            let quantized = (value / scale).round().clamp(-128.0, 127.0) as i8;
            bytes.push(quantized as u8);
        }
    }
    bytes
}

/// Q4F16_G64 encoder: groups of 64, affine 4-bit, `scale = (max-min)/15`, stored
/// as `[f16 scale][f16 min][32 packed bytes]` (36 bytes/block) with the low 32
/// values in the low nibbles and the high 32 in the high nibbles. Mirrors
/// `decode_q4f16_g64_slice`.
pub(crate) fn encode_q4f16_g64(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len().div_ceil(64) * 36);
    for group in data.chunks(64) {
        let min = group.iter().copied().fold(f32::INFINITY, f32::min);
        let max = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = if max > min { (max - min) / 15.0 } else { 1.0 };
        bytes.extend_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
        bytes.extend_from_slice(&f32_to_f16_bits(min).to_le_bytes());
        for idx in 0..32 {
            let lo = group.get(idx).copied().unwrap_or(min);
            let hi = group.get(idx + 32).copied().unwrap_or(min);
            let lo_q = ((lo - min) / scale).round().clamp(0.0, 15.0) as u8;
            let hi_q = ((hi - min) / scale).round().clamp(0.0, 15.0) as u8;
            bytes.push(lo_q | (hi_q << 4));
        }
    }
    bytes
}

/// Pack one 64-element group into a Q4F16_G64 block (`[f16 scale][f16 min][32
/// packed bytes]`) given an explicit affine range [`lo`, `lo + 15*scale`]. Mirrors
/// `decode_q4f16_g64_slice` so it round-trips bit-for-bit.
fn pack_q4_block(group: &[f32], lo: f32, scale: f32) -> [u8; 36] {
    let mut block = [0u8; 36];
    block[0..2].copy_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
    block[2..4].copy_from_slice(&f32_to_f16_bits(lo).to_le_bytes());
    let q = |v: f32| ((v - lo) / scale).round().clamp(0.0, 15.0) as u8;
    for idx in 0..32 {
        let lo_q = q(group.get(idx).copied().unwrap_or(lo));
        let hi_q = q(group.get(idx + 32).copied().unwrap_or(lo));
        block[4 + idx] = lo_q | (hi_q << 4);
    }
    block
}

/// Reconstruction MSE of a group quantized to the affine range [`lo`, `lo +
/// 15*scale`], using the same f16-rounded scale/min the decoder will see so the
/// search optimizes the value that is actually stored.
fn q4_group_mse(group: &[f32], lo: f32, scale: f32) -> f32 {
    let lo = f16_bits_to_f32(f32_to_f16_bits(lo));
    let scale = f16_bits_to_f32(f32_to_f16_bits(scale)).max(1e-12);
    group
        .iter()
        .map(|&v| {
            let q = ((v - lo) / scale).round().clamp(0.0, 15.0);
            let recon = lo + q * scale;
            (v - recon) * (v - recon)
        })
        .sum()
}

/// Calibrated Q4F16_G64 encoder (the `+` in `oq4+`): per 64-group, search the
/// quantization range that minimizes reconstruction MSE rather than using the
/// raw min/max. The range is shrunk symmetrically around the group midpoint over
/// a grid of clip ratios; tighter ranges give finer resolution on the bulk of
/// the values at the cost of clipping outliers, which is a net win whenever the
/// group has heavy tails. Data-free (weight-only). rayon-parallel over groups.
pub(crate) fn encode_q4f16_g64_clipsearch(data: &[f32]) -> Vec<u8> {
    use rayon::prelude::*;
    // 17 clip ratios from 1.0 (raw min/max) down to 0.2 of the half-range.
    const RATIOS: usize = 17;
    let blocks: Vec<[u8; 36]> = data
        .par_chunks(64)
        .map(|group| {
            let min = group.iter().copied().fold(f32::INFINITY, f32::min);
            let max = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if !(max > min) {
                return pack_q4_block(group, min, 1.0);
            }
            let mid = 0.5 * (min + max);
            let half = 0.5 * (max - min);
            let mut best_lo = min;
            let mut best_scale = (max - min) / 15.0;
            let mut best_mse = q4_group_mse(group, best_lo, best_scale);
            for step in 1..RATIOS {
                let ratio = 1.0 - 0.8 * (step as f32) / ((RATIOS - 1) as f32);
                let lo = mid - ratio * half;
                let hi = mid + ratio * half;
                let scale = (hi - lo) / 15.0;
                if !(scale > 0.0) {
                    continue;
                }
                let mse = q4_group_mse(group, lo, scale);
                if mse < best_mse {
                    best_mse = mse;
                    best_lo = lo;
                    best_scale = scale;
                }
            }
            pack_q4_block(group, best_lo, best_scale)
        })
        .collect();
    let mut bytes = Vec::with_capacity(blocks.len() * 36);
    for block in blocks {
        bytes.extend_from_slice(&block);
    }
    bytes
}

/// Q4_K encoder ported from `hipfire_quantize::codecs::quantize_q4k` (the proven
/// LLM-path codec). 256-element super-blocks with 8 sub-blocks of 32, each with
/// its own 6-bit scale+min under a per-super-block f16 `d`/`dmin` — finer and
/// hierarchical vs the flat group-64 affine of `encode_q4f16_g64`. The byte
/// layout must match `hipfire_runtime::quant::dequant_q4k` (the diffusion
/// decoder); `q4k_encoder_round_trips_through_diffusion_decoder` guards that.
pub(crate) fn encode_q4k(f32_data: &[f32]) -> Vec<u8> {
    let super_block_size = 256;
    let block_bytes = 144;
    let n = f32_data.len();
    let n_blocks = n.div_ceil(super_block_size);
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let sb_start = b * super_block_size;
        let sb_end = (sb_start + super_block_size).min(n);
        let out_off = b * block_bytes;

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

        let max_scale = sub_scales.iter().cloned().fold(0.0f32, f32::max);
        let max_min = sub_mins.iter().map(|m| -m).fold(0.0f32, f32::max);
        let d = if max_scale > 0.0 { max_scale / 63.0 } else { 0.0 };
        let dmin = if max_min > 0.0 { max_min / 63.0 } else { 0.0 };
        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
        let inv_dmin = if dmin > 0.0 { 1.0 / dmin } else { 0.0 };

        let mut scale_ints = [0u8; 8];
        let mut min_ints = [0u8; 8];
        for sb in 0..8 {
            scale_ints[sb] = (sub_scales[sb] * inv_d + 0.5).min(63.0) as u8;
            min_ints[sb] = ((-sub_mins[sb]) * inv_dmin + 0.5).min(63.0) as u8;
        }

        output[out_off..out_off + 2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        output[out_off + 2..out_off + 4].copy_from_slice(&f32_to_f16_bits(dmin).to_le_bytes());

        let sc = &mut output[out_off + 4..out_off + 16];
        for i in 0..4 {
            sc[i] = (scale_ints[i] & 63) | ((scale_ints[4 + i] >> 4) << 6);
            sc[4 + i] = (min_ints[i] & 63) | ((min_ints[4 + i] >> 4) << 6);
        }
        for i in 0..4 {
            sc[8 + i] = (scale_ints[4 + i] & 0xF) | ((min_ints[4 + i] & 0xF) << 4);
        }

        let qs = &mut output[out_off + 16..out_off + 144];
        for group in 0..4 {
            let sb_even = group * 2;
            let sb_odd = group * 2 + 1;
            let eff_scale_e = d * scale_ints[sb_even] as f32;
            let eff_min_e = dmin * min_ints[sb_even] as f32;
            let inv_se = if eff_scale_e > 0.0 { 1.0 / eff_scale_e } else { 0.0 };
            let eff_scale_o = d * scale_ints[sb_odd] as f32;
            let eff_min_o = dmin * min_ints[sb_odd] as f32;
            let inv_so = if eff_scale_o > 0.0 { 1.0 / eff_scale_o } else { 0.0 };
            for l in 0..32 {
                let idx_e = sb_start + group * 64 + l;
                let idx_o = sb_start + group * 64 + 32 + l;
                let val_e = if idx_e < sb_end { f32_data[idx_e] } else { 0.0 };
                let val_o = if idx_o < sb_end { f32_data[idx_o] } else { 0.0 };
                let q_e = ((val_e + eff_min_e) * inv_se + 0.5).clamp(0.0, 15.0) as u8;
                let q_o = ((val_o + eff_min_o) * inv_so + 0.5).clamp(0.0, 15.0) as u8;
                qs[group * 32 + l] = q_e | (q_o << 4);
            }
        }
    }
    output
}

/// True when a tensor entry is a large weight matrix worth quantizing: a
/// `.weight` with rank >= 2 (conv 4D / linear 2D), excluding 1D norm/bias
/// vectors which are cheap and precision-sensitive. Configs/tokenizers (rank-1
/// byte blobs) are excluded by the rank check.
fn is_quantizable_weight(name: &str, shape: &[u32]) -> bool {
    name.ends_with(".weight") && shape.len() >= 2 && shape.iter().all(|&d| d > 0)
}

#[derive(Debug, Default)]
pub struct DiffusionQuantizeSummary {
    pub quantized_tensors: usize,
    pub copied_tensors: usize,
    /// Linears packed with LDLQ Hessian error feedback (Opus calibrated path).
    pub ldlq_tensors: usize,
    pub source_bytes: u64,
    pub output_bytes: u64,
}

/// Re-encode the weight tensors of `source` into `format`, copying all other
/// entries verbatim, and write the result to `output`. For the Opus formats
/// (`oq4`/`oq4++`/`oq8`), `calib` (a `.calib.hfq` opened via `HessianSidecar`)
/// supplies per-linear Hessians for LDLQ error feedback.
pub fn quantize_diffusion_hfq(
    source: &Path,
    output: &Path,
    format: DiffusionQuantFormat,
    calib: Option<&HessianSidecar>,
) -> anyhow::Result<DiffusionQuantizeSummary> {
    let hfq = HfqFile::open(source)?;
    let mut summary = DiffusionQuantizeSummary {
        source_bytes: std::fs::metadata(source)?.len(),
        ..Default::default()
    };
    let (signs1, signs2) = if format.is_opus() {
        (
            gen_fwht_signs(OQ_FWHT_SEED1, 256),
            gen_fwht_signs(OQ_FWHT_SEED2, 256),
        )
    } else {
        (Vec::new(), Vec::new())
    };

    let names: Vec<String> = hfq.tensors().iter().map(|t| t.name.clone()).collect();
    let mut out_tensors: Vec<HfqMemTensor> = Vec::with_capacity(names.len());
    for name in &names {
        let (info, bytes) = hfq
            .tensor_data_vec(name)
            .ok_or_else(|| anyhow::anyhow!("tensor {name:?} vanished from source index"))?;
        if is_quantizable_weight(name, &info.shape) {
            let decoded = CpuTensor::from_hfq(&hfq, name)
                .map_err(|e| anyhow::anyhow!("decode {name:?}: {e}"))?;
            let (quant_type, group_size, data) = if format.is_opus() {
                let (qt, gs, bytes, ldlq) = encode_opus_tensor(
                    format, name, &info.shape, &decoded.data, calib, &signs1, &signs2,
                );
                if ldlq {
                    summary.ldlq_tensors += 1;
                }
                (qt, gs, bytes)
            } else {
                (
                    format.quant_type(),
                    format.group_size(),
                    format.encode(&decoded.data),
                )
            };
            out_tensors.push(HfqMemTensor {
                name: name.clone(),
                quant_type,
                shape: info.shape.clone(),
                group_size,
                data,
            });
            summary.quantized_tensors += 1;
        } else {
            out_tensors.push(HfqMemTensor {
                name: name.clone(),
                quant_type: info.quant_type,
                shape: info.shape.clone(),
                group_size: info.group_size,
                data: bytes,
            });
            summary.copied_tensors += 1;
        }
    }

    // Update the informational weight_format string; per-tensor decoding keys off
    // quant_type, so this does not affect loading.
    let metadata_json = rewrite_weight_format(&hfq.metadata_json, format.weight_format_label());
    write_hfqm_package_mem(output, hfq.arch_id, &metadata_json, &out_tensors)?;
    summary.output_bytes = std::fs::metadata(output)?.len();
    Ok(summary)
}

/// Repack canonical `oq4g256` (`[f16 scale][128 nibbles]` per 256-group,
/// row-contiguous — the [`quantize_oq4g256`] / `decode_oq4g256_slice` format)
/// into the arch "combined" device layout consumed by
/// `rdna_compute::Gpu::gemm_oq4_grouped_f16_wmma` (W4A16). Mirrors
/// `hipfire_arch_qwen35::qwen35::oq4_pack_arch_combined` byte-for-byte (replicated
/// here to avoid a diffusion→arch dependency; both read the same qt-34 bytes).
///
/// Output (`[m,k]`, `ng=k/256`): `[nibbles m*(k/2)] [f32 scales m*ng]
/// [interleaved m*ng*132]`. The W4A16 GEMM reads the first two regions; the
/// interleaved tail is for decode GEMVs (kept for layout parity).
/// Byte length of [`pack_oq4_arch_combined`]'s output for an `[m, k]` matrix.
pub fn oq4_arch_combined_len(m: usize, k: usize) -> usize {
    let ng = k / 256;
    m * (k / 2) + m * ng * 4 + m * ng * (4 + 128)
}

pub fn pack_oq4_arch_combined(data: &[u8], m: usize, k: usize) -> Vec<u8> {
    use crate::quant_decode::f16_bits_to_f32;
    const GROUP: usize = 256;
    const BLOCK: usize = 130;
    const ILB: usize = 4 + 128;
    assert_eq!(k % GROUP, 0, "oq4 arch pack requires K % 256 == 0 (got K={k})");
    let ng = k / GROUP;
    let packed_bytes = m * (k / 2);
    let scales_bytes = m * ng * 4;
    let il_bytes = m * ng * ILB;
    assert_eq!(data.len(), m * ng * BLOCK, "oq4 weight byte length mismatch");
    let mut combined = vec![0u8; packed_bytes + scales_bytes + il_bytes];
    let il_base = packed_bytes + scales_bytes;
    for r in 0..m {
        for g in 0..ng {
            let src = (r * ng + g) * BLOCK;
            let scale = f16_bits_to_f32(u16::from_le_bytes([data[src], data[src + 1]]));
            let dst = r * (k / 2) + g * (GROUP / 2);
            combined[dst..dst + 128].copy_from_slice(&data[src + 2..src + BLOCK]);
            let so = packed_bytes + (r * ng + g) * 4;
            combined[so..so + 4].copy_from_slice(&scale.to_le_bytes());
            let io = il_base + (r * ng + g) * ILB;
            combined[io..io + 4].copy_from_slice(&scale.to_le_bytes());
            combined[io + 4..io + ILB].copy_from_slice(&data[src + 2..src + BLOCK]);
        }
    }
    combined
}

/// Open a `.calib.hfq` sidecar for use as the `calib` argument to
/// [`quantize_diffusion_hfq`].
pub fn open_calib_sidecar(path: &Path) -> anyhow::Result<HessianSidecar> {
    HessianSidecar::open(path).map_err(|e| anyhow::anyhow!("{e}"))
}

fn rewrite_weight_format(metadata_json: &str, label: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(metadata_json) {
        Ok(mut value) => {
            if let Some(quant) = value.get_mut("quantization").and_then(|q| q.as_object_mut()) {
                quant.insert(
                    "weight_format".to_string(),
                    serde_json::Value::String(label.to_string()),
                );
            }
            serde_json::to_string(&value).unwrap_or_else(|_| metadata_json.to_string())
        }
        Err(_) => metadata_json.to_string(),
    }
}
