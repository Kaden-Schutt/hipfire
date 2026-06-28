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
}

impl DiffusionQuantFormat {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "q8" | "q8f16" | "q80" => Some(Self::Q8F16),
            "q4" | "q4f16" | "q4f16g64" => Some(Self::Q4F16G64),
            _ => None,
        }
    }

    fn quant_type(self) -> u8 {
        match self {
            Self::Q8F16 => QT_DIFFUSION_TENSOR_Q8F16,
            Self::Q4F16G64 => QT_DIFFUSION_TENSOR_Q4F16_G64,
        }
    }

    fn group_size(self) -> u32 {
        match self {
            Self::Q8F16 => 32,
            Self::Q4F16G64 => 64,
        }
    }

    fn weight_format_label(self) -> &'static str {
        match self {
            Self::Q8F16 => "q8",
            Self::Q4F16G64 => "q4",
        }
    }

    fn encode(self, data: &[f32]) -> Vec<u8> {
        match self {
            Self::Q8F16 => encode_q8f16(data),
            Self::Q4F16G64 => encode_q4f16_g64(data),
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
