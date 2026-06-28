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
}

impl DiffusionQuantFormat {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "q8" | "q8f16" | "q80" => Some(Self::Q8F16),
            "q4" | "q4f16" | "q4f16g64" => Some(Self::Q4F16G64),
            "q4+" | "q4c" | "q4clip" | "oq4+" | "q4f16g64clip" => Some(Self::Q4F16G64Clip),
            "q4k" | "q4_k" => Some(Self::Q4K),
            _ => None,
        }
    }

    fn quant_type(self) -> u8 {
        match self {
            Self::Q8F16 => QT_DIFFUSION_TENSOR_Q8F16,
            Self::Q4F16G64 | Self::Q4F16G64Clip => QT_DIFFUSION_TENSOR_Q4F16_G64,
            Self::Q4K => QT_DIFFUSION_TENSOR_Q4_K,
        }
    }

    fn group_size(self) -> u32 {
        match self {
            Self::Q8F16 => 32,
            Self::Q4F16G64 | Self::Q4F16G64Clip => 64,
            Self::Q4K => 256,
        }
    }

    fn weight_format_label(self) -> &'static str {
        match self {
            Self::Q8F16 => "q8",
            Self::Q4F16G64 => "q4",
            Self::Q4F16G64Clip => "q4+",
            Self::Q4K => "q4k",
        }
    }

    fn encode(self, data: &[f32]) -> Vec<u8> {
        match self {
            Self::Q8F16 => encode_q8f16(data),
            Self::Q4F16G64 => encode_q4f16_g64(data),
            Self::Q4F16G64Clip => encode_q4f16_g64_clipsearch(data),
            Self::Q4K => encode_q4k(data),
        }
    }
}

/// q8_0 encoder: groups of 32, symmetric int8, `scale = max_abs / 127`, stored
/// as `[f16 scale][32 x i8]` (34 bytes/block). Mirrors `dequantize_q8_0`.
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
/// layout must match `hipfire_runtime::quant::dequantize_q4_k` (the diffusion
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
    pub source_bytes: u64,
    pub output_bytes: u64,
}

/// Re-encode the weight tensors of `source` into `format`, copying all other
/// entries verbatim, and write the result to `output`.
pub fn quantize_diffusion_hfq(
    source: &Path,
    output: &Path,
    format: DiffusionQuantFormat,
) -> anyhow::Result<DiffusionQuantizeSummary> {
    let hfq = HfqFile::open(source)?;
    let mut summary = DiffusionQuantizeSummary {
        source_bytes: std::fs::metadata(source)?.len(),
        ..Default::default()
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
            out_tensors.push(HfqMemTensor {
                name: name.clone(),
                quant_type: format.quant_type(),
                shape: info.shape.clone(),
                group_size: format.group_size(),
                data: format.encode(&decoded.data),
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
