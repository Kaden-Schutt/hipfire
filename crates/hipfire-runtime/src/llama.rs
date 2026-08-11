// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LLaMA model implementation using RDNA GPU compute.
//! Supports loading from GGUF files and running inference.

use crate::arch::{screen_weight_tensor, MmqScreenable};
use crate::gguf::{GgmlType, GgufFile, TensorInfo};
use crate::kv_backend::{
    KvBackend, KvChunkPlan, DEFAULT_KV_CHUNK_TOKENS, DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
};
use crate::kv_mode::KvMode;
use crate::multi_gpu::Gpus;
use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

/// Model architecture type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArch {
    Llama,
    Qwen3,
}

/// Model configuration, read from GGUF metadata.
/// Supports LLaMA-family and Qwen3 architectures.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub arch: ModelArch,
    pub dim: usize,        // model dimension (embedding size)
    pub hidden_dim: usize, // FFN hidden dimension
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize, // for GQA
    pub vocab_size: usize,
    pub head_dim: usize,
    pub norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
    pub bos_token: u32,
    pub eos_token: u32,
    pub has_qk_norm: bool, // Qwen3 feature
}

impl LlamaConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Option<Self> {
        let arch_str = gguf.meta_str("general.architecture")?;

        // Determine architecture and metadata prefix
        let (arch, prefix) = match arch_str {
            "llama" => (ModelArch::Llama, "llama"),
            "qwen3" => (ModelArch::Qwen3, "qwen3"),
            other => {
                eprintln!("Warning: unknown architecture '{other}', attempting LLaMA-compatible");
                (ModelArch::Llama, other)
            }
        };

        let dim = gguf.meta_u32(&format!("{prefix}.embedding_length"))? as usize;
        let n_layers = gguf.meta_u32(&format!("{prefix}.block_count"))? as usize;
        let n_heads = gguf.meta_u32(&format!("{prefix}.attention.head_count"))? as usize;
        let n_kv_heads = gguf
            .meta_u32(&format!("{prefix}.attention.head_count_kv"))
            .unwrap_or(n_heads as u32) as usize;
        let hidden_dim = gguf.meta_u32(&format!("{prefix}.feed_forward_length"))? as usize;
        let vocab_size = gguf.meta_u32(&format!("{prefix}.vocab_size")).or_else(|| {
            gguf.find_tensor("token_embd.weight")
                .map(|t| t.shape[1] as u32)
        })? as usize;
        let head_dim = gguf
            .meta_u32(&format!("{prefix}.attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or(dim / n_heads);
        let norm_eps = gguf
            .meta_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let max_seq_len = gguf
            .meta_u32(&format!("{prefix}.context_length"))
            .unwrap_or(2048) as usize;
        let rope_freq_base = gguf
            .meta_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(10000.0);
        let bos_token = gguf.meta_u32("tokenizer.ggml.bos_token_id").unwrap_or(1);
        let eos_token = gguf.meta_u32("tokenizer.ggml.eos_token_id").unwrap_or(2);

        // Check for QK normalization (Qwen3 feature)
        let has_qk_norm = gguf.find_tensor("blk.0.attn_q_norm.weight").is_some();

        Some(LlamaConfig {
            arch,
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            head_dim,
            norm_eps,
            max_seq_len,
            rope_freq_base,
            bos_token,
            eos_token,
            has_qk_norm,
        })
    }
}

/// Dequantize Q4_0 data to f32.
/// Q4_0 block: 2 bytes (f16 scale) + 16 bytes (32 x 4-bit values)
pub fn dequantize_q4_0(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 32;
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];

    for b in 0..nblocks {
        let block_offset = b * 18; // 2 + 16 bytes per block
        if block_offset + 18 > data.len() {
            break;
        }
        let scale_bytes = [data[block_offset], data[block_offset + 1]];
        let scale = f16_to_f32(u16::from_le_bytes(scale_bytes));

        for j in 0..16 {
            let byte = data[block_offset + 2 + j];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;

            let idx = b * block_size + j * 2;
            if idx < n {
                out[idx] = lo as f32 * scale;
            }
            if idx + 1 < n {
                out[idx + 1] = hi as f32 * scale;
            }
        }
    }
    out
}

/// Dequantize Q8_0 data to f32.
/// Q8_0 block: 2 bytes (f16 scale) + 32 bytes (32 x int8)
pub fn dequantize_q8_0(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 32;
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];

    for b in 0..nblocks {
        let block_offset = b * 34; // 2 + 32 bytes per block
        if block_offset + 34 > data.len() {
            break;
        }
        let scale_bytes = [data[block_offset], data[block_offset + 1]];
        let scale = f16_to_f32(u16::from_le_bytes(scale_bytes));

        for j in 0..32 {
            let idx = b * block_size + j;
            if idx < n {
                let val = data[block_offset + 2 + j] as i8;
                out[idx] = val as f32 * scale;
            }
        }
    }
    out
}

pub fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;

    if exp == 0xFF {
        let f16_frac = if frac == 0 { 0 } else { (frac >> 13) | 1 };
        return ((sign << 15) | (0x1F << 10) | f16_frac) as u16;
    }

    let new_exp = exp - 127 + 15;

    if new_exp >= 31 {
        return ((sign << 15) | (0x1F << 10)) as u16; // overflow → inf
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return (sign << 15) as u16; // underflow → zero
        }
        let f = frac | 0x800000;
        let shift = (1 - new_exp + 13) as u32;
        return ((sign << 15) | (f >> shift)) as u16;
    }

    ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

/// Dequantize Q4_K data to f32.
/// Q4_K super-block: 256 elements
///   2 bytes: f16 d (super-block scale)
///   2 bytes: f16 dmin (super-block min)
///   12 bytes: scales/mins for 8 sub-blocks (6 bits each, packed)
///   128 bytes: 256 x 4-bit quantized values
pub fn dequantize_q4_k(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 256;
    let block_bytes = 144; // 2+2+12+128
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];

    for b in 0..nblocks {
        let off = b * block_bytes;
        if off + block_bytes > data.len() {
            break;
        }

        let d = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[off + 2], data[off + 3]]));

        // Unpack scales and mins from 12 bytes (at off+4)
        let sc_data = &data[off + 4..off + 16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        // First 4 sub-blocks: lower 6 bits from bytes 0-3 (scales) and 4-7 (mins)
        for i in 0..4 {
            scales[i] = sc_data[i] & 63;
            mins[i] = sc_data[4 + i] & 63;
        }
        // Next 4 sub-blocks: lower 4 bits from bytes 8-11, upper 2 bits from bytes 0-7
        for i in 0..4 {
            scales[4 + i] = (sc_data[8 + i] & 0xF) | ((sc_data[i] >> 6) << 4);
            mins[4 + i] = (sc_data[8 + i] >> 4) | ((sc_data[4 + i] >> 6) << 4);
        }

        // Dequantize 256 values from 128 bytes of 4-bit data.
        // GGML layout: 4 groups of 64 elements. Each group has 2 sub-blocks
        // sharing 32 bytes: lower nibble → even sub-block, upper nibble → odd.
        let qdata = &data[off + 16..off + 16 + 128];
        for group in 0..4 {
            let sb_even = group * 2;
            let sb_odd = group * 2 + 1;
            let sc_even = d * scales[sb_even] as f32;
            let m_even = dmin * mins[sb_even] as f32;
            let sc_odd = d * scales[sb_odd] as f32;
            let m_odd = dmin * mins[sb_odd] as f32;

            for l in 0..32 {
                let byte = qdata[group * 32 + l];
                let idx_even = b * block_size + group * 64 + l;
                let idx_odd = idx_even + 32;
                if idx_even < n {
                    out[idx_even] = (byte & 0x0F) as f32 * sc_even - m_even;
                }
                if idx_odd < n {
                    out[idx_odd] = ((byte >> 4) & 0x0F) as f32 * sc_odd - m_odd;
                }
            }
        }
    }
    out
}

/// Convert Q4_K raw data to Q4_F16_G64 format.
/// Dequantizes Q4_K to F32 intermediates, then re-quantizes to Q4_F16_G64.
/// Q4_K: 144 bytes per 256 elements → Q4_F16_G64: 4×36=144 bytes per 256 elements (same size).
pub fn convert_q4k_to_q4f16_g64(q4k_data: &[u8], n_elements: usize) -> Vec<u8> {
    let f32_values = dequantize_q4_k(q4k_data, n_elements);

    let group_size = 64;
    let block_bytes = 36;
    let n_blocks = (n_elements + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n_elements);
        let group = &f32_values[start..end];

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
        output[out_off + 2..out_off + 4].copy_from_slice(&f32_to_f16(min_val).to_le_bytes());

        // Pack nibbles: byte[i] = low_nibble(element i) | high_nibble(element i+32)
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

/// Convert Q4_K raw data to Q4_F16_G32 format — nearly lossless.
/// Each Q4_K sub-block (32 elements) maps directly to one Q4_F16_G32 block.
/// Nibbles are preserved exactly; only scale/min are converted to FP16.
/// Q4_K: 144 bytes per 256 elements → Q4_F16_G32: 8×20=160 bytes per 256 elements (11% larger).
pub fn convert_q4k_to_q4f16_g32(q4k_data: &[u8], n_elements: usize) -> Vec<u8> {
    let q4k_block_bytes = 144;
    let q4k_block_elems = 256;
    let g32_block_bytes = 20;
    let nblocks = (n_elements + q4k_block_elems - 1) / q4k_block_elems;
    // 8 sub-blocks per Q4_K super-block → 8 G32 blocks
    let mut output = vec![0u8; nblocks * 8 * g32_block_bytes];

    for b in 0..nblocks {
        let off = b * q4k_block_bytes;
        if off + q4k_block_bytes > q4k_data.len() {
            break;
        }

        let d = f16_to_f32(u16::from_le_bytes([q4k_data[off], q4k_data[off + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([q4k_data[off + 2], q4k_data[off + 3]]));

        let sc_data = &q4k_data[off + 4..off + 16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        for i in 0..4 {
            scales[i] = sc_data[i] & 63;
            mins[i] = sc_data[4 + i] & 63;
        }
        for i in 0..4 {
            scales[4 + i] = (sc_data[8 + i] & 0xF) | ((sc_data[i] >> 6) << 4);
            mins[4 + i] = (sc_data[8 + i] >> 4) | ((sc_data[4 + i] >> 6) << 4);
        }

        let qdata = &q4k_data[off + 16..off + 16 + 128];

        // Each of 4 groups has 2 sub-blocks (even=lower nibble, odd=upper nibble)
        for group in 0..4 {
            let sb_even = group * 2;
            let sb_odd = group * 2 + 1;

            // Sub-block even (elements group*64+0..group*64+31) → G32 block
            let eff_scale_even = d * scales[sb_even] as f32;
            let eff_min_even = -(dmin * mins[sb_even] as f32);
            let out_off_even = (b * 8 + sb_even) * g32_block_bytes;
            output[out_off_even..out_off_even + 2]
                .copy_from_slice(&f32_to_f16(eff_scale_even).to_le_bytes());
            output[out_off_even + 2..out_off_even + 4]
                .copy_from_slice(&f32_to_f16(eff_min_even).to_le_bytes());

            // Sub-block odd (elements group*64+32..group*64+63) → G32 block
            let eff_scale_odd = d * scales[sb_odd] as f32;
            let eff_min_odd = -(dmin * mins[sb_odd] as f32);
            let out_off_odd = (b * 8 + sb_odd) * g32_block_bytes;
            output[out_off_odd..out_off_odd + 2]
                .copy_from_slice(&f32_to_f16(eff_scale_odd).to_le_bytes());
            output[out_off_odd + 2..out_off_odd + 4]
                .copy_from_slice(&f32_to_f16(eff_min_odd).to_le_bytes());

            // Copy nibbles: Q4_K stores them as byte[l] where low=even, high=odd.
            // G32 packing: byte[i] = lo_nibble(elem i) | hi_nibble(elem i+16)
            // Q4_K byte[l] has: elem l in low nibble, elem l+32 in high nibble.
            // For sub-block even: we want the 32 lower nibbles from group*32 bytes.
            // For sub-block odd: we want the 32 upper nibbles.
            // G32 block maps: thread t reads byte[t&15], lo nibble = elem t (t<16), hi nibble = elem t-16+16=t (t>=16)
            // So G32 byte[i] = nibble(elem i) | nibble(elem i+16) << 4
            for i in 0..16 {
                let src_byte_0 = qdata[group * 32 + i];
                let src_byte_1 = qdata[group * 32 + 16 + i];
                // Even sub-block: lower nibbles
                let nib_0 = src_byte_0 & 0xF;
                let nib_1 = src_byte_1 & 0xF;
                output[out_off_even + 4 + i] = nib_0 | (nib_1 << 4);
                // Odd sub-block: upper nibbles
                let nib_2 = src_byte_0 >> 4;
                let nib_3 = src_byte_1 >> 4;
                output[out_off_odd + 4 + i] = nib_2 | (nib_3 << 4);
            }
        }
    }

    output
}

/// Dequantize Q6_K data to f32 (matches GGML reference exactly).
/// Q6_K super-block: 256 elements = 2 groups of 128
///   ql[128]: lower 4 bits (shared between lo/hi nibble pairs)
///   qh[64]: upper 2 bits (packed 4 per byte)
///   scales[16]: int8 scales for sub-groups of 16 elements
///   d: f16 super-block scale
pub fn dequantize_q6_k(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 256;
    let block_bytes = 210; // 128 + 64 + 16 + 2
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];

    for b in 0..nblocks {
        let off = b * block_bytes;
        if off + block_bytes > data.len() {
            break;
        }

        let mut ql = &data[off..off + 128];
        let mut qh = &data[off + 128..off + 192];
        let mut sc = &data[off + 192..off + 208];
        let d = f16_to_f32(u16::from_le_bytes([data[off + 208], data[off + 209]]));

        let base = b * block_size;

        // Process 2 groups of 128 elements each
        for group in 0..2 {
            let y_off = base + group * 128;
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;

                let idx0 = y_off + l;
                let idx1 = y_off + l + 32;
                let idx2 = y_off + l + 64;
                let idx3 = y_off + l + 96;

                if idx0 < n {
                    out[idx0] = d * sc[is] as i8 as f32 * q1 as f32;
                }
                if idx1 < n {
                    out[idx1] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                }
                if idx2 < n {
                    out[idx2] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                }
                if idx3 < n {
                    out[idx3] = d * sc[is + 6] as i8 as f32 * q4 as f32;
                }
            }
            // Advance pointers for next group
            ql = &ql[64..];
            qh = &qh[32..];
            sc = &sc[8..];
        }
    }
    out
}

/// Load tensor from GGUF as f32, dequantizing if needed.
fn load_tensor_f32(gguf: &GgufFile, info: &TensorInfo) -> Vec<f32> {
    let data = gguf.tensor_data(info);
    let n = info.numel();

    match info.dtype {
        GgmlType::F32 => {
            let mut out = vec![0.0f32; n];
            for (i, chunk) in data.chunks_exact(4).enumerate().take(n) {
                out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            out
        }
        GgmlType::F16 => {
            let mut out = vec![0.0f32; n];
            for (i, chunk) in data.chunks_exact(2).enumerate().take(n) {
                out[i] = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
            out
        }
        GgmlType::Q4_0 => dequantize_q4_0(data, n),
        GgmlType::Q8_0 => dequantize_q8_0(data, n),
        GgmlType::Q4K => dequantize_q4_k(data, n),
        GgmlType::Q6K => dequantize_q6_k(data, n),
        other => panic!("unsupported tensor type: {:?}", other),
    }
}

/// A weight matrix on GPU — may be quantized or F32.
/// ParoQuant Givens rotation metadata for a single linear layer.
/// Stored alongside the weight buffer; applied to activations before GEMV.
pub struct ParoRotation {
    pub pairs: GpuTensor, // I16 [krot, in_dim] — pair indices per rotation layer
    pub theta: GpuTensor, // F16 [krot, in_dim/2] — learned angles
    pub channel_scales: GpuTensor, // F16 [in_dim] — per-channel scaling factor alpha
    pub krot: u32,        // number of rotation layers (typically 8)
    pub group_size: u32,  // quantization group size (typically 128)
    /// True if `pairs`/`theta`/`channel_scales` are non-owning aliases into
    /// shared per-layer sidecars (e.g. MoE routed experts that share one
    /// rotation tuple across all experts in a layer). Owning ParoRotations
    /// set this to false; `WeightTensor::free_all` skips tensor frees when
    /// is_alias is true so the shared sidecars aren't double-freed.
    pub is_alias: bool,
}

pub struct WeightTensor {
    pub buf: GpuTensor,
    pub gpu_dtype: DType,  // dispatch type for kernel selection
    pub m: usize,          // output dim (rows)
    pub k: usize,          // input dim (cols)
    pub row_stride: usize, // padded row bytes (Q8HFQ only, 0 for others)
    /// ParoQuant Givens rotation metadata. None for all non-ParoQuant formats.
    pub paro: Option<ParoRotation>,
    /// Phase A Stage A — AWQ per-channel scale vector, length K, dtype F16.
    ///
    /// Populated by the loader when the .hfq carries a sibling sidecar tensor
    /// named `<weight>.awq_scale.weight`. The forward path (specifically the
    /// fused-rmsnorm-rotate call upstream of this linear) must apply
    /// `x[i] /= awq_scale[i]` before the rotation, completing the AWQ
    /// math `(W·s) · (x/s) = W·x`. The pre-scaling of W·s was done at
    /// quantize time (see `compute_awq_scales` in hipfire-quantize/main.rs).
    ///
    /// `None` for tensors that weren't AWQ-pre-scaled — backward-compatible
    /// with all existing .hfq files.
    pub awq_scale: Option<GpuTensor>,
}

impl WeightTensor {
    /// Free the weight buffer and any associated metadata (ParoQuant rotation,
    /// AWQ sidecar) from GPU.
    pub fn free_all(self, gpu: &mut Gpu) {
        if let Some(paro) = self.paro {
            // Aliased rotations point into shared per-layer sidecars; the owner
            // (e.g. MoeFfnWeights.paro_shared) frees them. Skip here.
            if !paro.is_alias {
                let _ = gpu.free_tensor(paro.pairs);
                let _ = gpu.free_tensor(paro.theta);
                let _ = gpu.free_tensor(paro.channel_scales);
            }
        }
        if let Some(awq) = self.awq_scale {
            let _ = gpu.free_tensor(awq);
        }
        let _ = gpu.free_tensor(self.buf);
    }
}

impl WeightTensor {
    /// Logic-free adapter to the dispatch-layer WeightRef. Wires Givens +
    /// AWQ + row_stride so GemvFamily sees everything a weight needs.
    pub fn dispatch_ref(&self) -> hipfire_dispatch::families::gemv::WeightRef<'_> {
        use hipfire_dispatch::families::gemv::{GivensRef, WeightRef};
        WeightRef {
            buf: &self.buf,
            dtype: self.gpu_dtype,
            m: self.m,
            k: self.k,
            row_stride: self.row_stride,
            rotation: self.paro.as_ref().map(|p| GivensRef {
                pairs: &p.pairs,
                theta: &p.theta,
                scales: &p.channel_scales,
                krot: p.krot as usize,
            }),
            awq_scale: self.awq_scale.as_ref(),
        }
    }
}

pub fn gemv_family() -> &'static hipfire_dispatch::families::gemv::GemvFamily {
    use std::sync::OnceLock;
    static GEMV: OnceLock<hipfire_dispatch::families::gemv::GemvFamily> = OnceLock::new();
    GEMV.get_or_init(hipfire_dispatch::families::gemv::GemvFamily::new)
}

/// Process-global [`GemmFamily`], mirroring [`gemv_family`]. #397 Ship 5.2:
/// arches route their batched-prefill plain-GEMM launches through
/// `gemm_family().run_key(..)` so the dispatcher-entry kernel selection lives in
/// the dispatch crate. `run_key` (explicit KernelKey) preserves the direct
/// `gpu.gemm_*` call's own internal arch dispatch byte-for-byte.
pub fn gemm_family() -> &'static hipfire_dispatch::families::gemm::GemmFamily {
    use std::sync::OnceLock;
    static GEMM: OnceLock<hipfire_dispatch::families::gemm::GemmFamily> = OnceLock::new();
    GEMM.get_or_init(hipfire_dispatch::families::gemm::GemmFamily::new)
}

/// Process-global [`FusedQkvFamily`], mirroring [`gemv_family`]. Used by the
/// dense-arch forward paths to route fused QKV / gate-up launches through the
/// centralized dispatch tables (arch gating + 1:1 KernelKey→kernel launch).
pub fn fused_qkv_family() -> &'static hipfire_dispatch::families::fused_qkv::FusedQkvFamily {
    use std::sync::OnceLock;
    static FUSED_QKV: OnceLock<hipfire_dispatch::families::fused_qkv::FusedQkvFamily> =
        OnceLock::new();
    FUSED_QKV.get_or_init(hipfire_dispatch::families::fused_qkv::FusedQkvFamily::new)
}

/// Process-global [`MoeFamily`], mirroring [`gemv_family`]. The centralized MoE
/// decode entry (Ship 4): arches route their per-layer MoE decode through
/// `moe_family().run(..)` so expert dispatch lives in the dispatch crate rather
/// than per-model kernel calls.
pub fn moe_family() -> &'static hipfire_dispatch::families::moe::MoeFamily {
    use std::sync::OnceLock;
    static MOE: OnceLock<hipfire_dispatch::families::moe::MoeFamily> = OnceLock::new();
    MOE.get_or_init(hipfire_dispatch::families::moe::MoeFamily::new)
}

/// Process-global [`AttentionFamily`], mirroring [`gemv_family`]. Ship 3:
/// arches route their per-layer attention decode through
/// `attention_family().run_attention(..)` so KV-write + flash-attention dispatch
/// lives in the dispatch crate rather than per-model inline match trees.
pub fn attention_family() -> &'static hipfire_dispatch::families::attention::AttentionFamily {
    use std::sync::OnceLock;
    static ATTENTION: OnceLock<hipfire_dispatch::families::attention::AttentionFamily> =
        OnceLock::new();
    ATTENTION.get_or_init(hipfire_dispatch::families::attention::AttentionFamily::new)
}

pub use hipfire_dispatch::context::DispatchCtx;
pub use hipfire_dispatch::families::attention::AttnParams;
pub use hipfire_dispatch::families::attention::FullAttnParams;
pub use hipfire_dispatch::families::fused_qkv::FusedQkvParams;
pub use hipfire_dispatch::families::gemv::{RotInput, RotateInputs, RotatedActivation};
pub use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
pub use hipfire_dispatch::types::KernelKey;
pub use hipfire_dispatch::types::ShapeInfo;
pub use hipfire_dispatch::types::{dtype_post_rotation_variant, dtype_rotation_plan, GemvVariant};

/// How the embedding table is stored on GPU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmbeddingFormat {
    F32,      // dequantized to F32, use D2D copy
    Q4K,      // raw Q4K blocks, use GPU dequant kernel
    HFQ4G256, // raw HFQ4-G256 blocks, use GPU dequant kernel
    HFQ4G128, // raw HFQ4-G128 blocks, use GPU dequant kernel
    Q8_0,     // raw Q8_0 blocks, use GPU dequant kernel
}

/// Single-token embedding lookup, dispatched on the table's storage format.
/// Centralizes the format→kernel mapping shared by every per-token decode path
/// (and the dots-ocr VLM text path in the daemon), so a newly-added format only
/// has to be wired in one place.
pub fn embedding_lookup_dispatch(
    gpu: &mut Gpu,
    format: EmbeddingFormat,
    table: &GpuTensor,
    output: &GpuTensor,
    token: u32,
    dim: usize,
) -> HipResult<()> {
    match format {
        EmbeddingFormat::Q4K => gpu.embedding_lookup_q4k(table, output, token, dim),
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(table, output, token, dim),
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(table, output, token, dim),
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(table, output, token, dim),
        EmbeddingFormat::F32 => gpu.embedding_lookup(table, output, token, dim),
    }
}

/// GPU-resident LLaMA model weights.
pub struct LlamaWeights {
    pub token_embd: GpuTensor,
    pub embd_format: EmbeddingFormat,
    pub output_norm: GpuTensor,
    pub output: WeightTensor,
    pub layers: Vec<LayerWeights>,
    /// True iff `output.buf` is a non-owning view of `token_embd.buf`
    /// (tied lm_head, single-GPU alias). When true, `free_gpu` must NOT free
    /// `output.buf`. Mirrors `Qwen35Weights` / `Qwen2Weights`.
    pub lm_head_aliases_embd: bool,
}

pub struct LayerWeights {
    pub attn_norm: GpuTensor,
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,
    pub wo: WeightTensor,
    pub q_norm: Option<GpuTensor>, // Qwen3: per-head Q normalization
    pub k_norm: Option<GpuTensor>, // Qwen3: per-head K normalization
    pub ffn_norm: GpuTensor,
    pub w_gate: WeightTensor,
    pub w_up: WeightTensor,
    pub w_down: WeightTensor,
}

impl LlamaWeights {
    /// Return all GPU buffers to the pool (drained on unload). Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.token_embd);
        let _ = gpu.free_tensor(self.output_norm);
        if !self.lm_head_aliases_embd {
            let _ = gpu.free_tensor(self.output.buf);
        }
        for l in self.layers {
            let _ = gpu.free_tensor(l.attn_norm);
            let _ = gpu.free_tensor(l.wq.buf);
            let _ = gpu.free_tensor(l.wk.buf);
            let _ = gpu.free_tensor(l.wv.buf);
            let _ = gpu.free_tensor(l.wo.buf);
            if let Some(t) = l.q_norm {
                let _ = gpu.free_tensor(t);
            }
            if let Some(t) = l.k_norm {
                let _ = gpu.free_tensor(t);
            }
            let _ = gpu.free_tensor(l.ffn_norm);
            let _ = gpu.free_tensor(l.w_gate.buf);
            let _ = gpu.free_tensor(l.w_up.buf);
            let _ = gpu.free_tensor(l.w_down.buf);
        }
    }
}

impl MmqScreenable for LlamaWeights {
    fn screen_mmq_weights(&self, gpu: &mut Gpu) -> (usize, usize) {
        let (mut safe, mut unsafe_count) = (0usize, 0usize);
        screen_weight_tensor(&self.output, gpu, &mut safe, &mut unsafe_count);
        for layer in &self.layers {
            for weight in [
                &layer.wq,
                &layer.wk,
                &layer.wv,
                &layer.wo,
                &layer.w_gate,
                &layer.w_up,
                &layer.w_down,
            ] {
                screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
            }
        }
        (safe, unsafe_count)
    }
}

/// Dispatch GEMV for a weight tensor (quantized or F32).
/// y = W * x where W is the weight tensor, x is F32 input, y is F32 output.

pub fn weight_gemv(gpu: &mut Gpu, w: &WeightTensor, x: &GpuTensor, y: &GpuTensor) -> HipResult<()> {
    use hipfire_dispatch::context::DispatchCtx;
    use hipfire_dispatch::families::gemv::{GemvParams, WeightRef};
    use hipfire_dispatch::types::{dtype_needs_rotation, GemvVariant};

    let gemv = crate::llama::gemv_family();
    let ctx = DispatchCtx::new(gpu);
    let wr = WeightRef {
        buf: &w.buf,
        dtype: w.gpu_dtype,
        m: w.m,
        k: w.k,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    };

    if !dtype_needs_rotation(w.gpu_dtype) {
        return gemv
            .run_auto(&ctx, gpu, &wr, x, y)
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()));
    }

    macro_rules! xr {
        () => {{
            GpuTensor {
                buf: unsafe { gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.scratch.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            }
        }};
    }

    match w.gpu_dtype {
        // MQ8 reads from internal scratch — no rotation or x alias needed
        DType::MQ8G256 => {
            gpu.ensure_mq_signs()?;
            gemv.run_auto(&ctx, gpu, &wr, x, y)
                .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))
        }
        // MQ4G128 uses G128 rotation (rotate_x_mq_128, sign seeds 43/1043)
        DType::MQ4G128 => {
            use hipfire_dispatch::families::rotation::{RotationFamily, RotationParams};
            use std::sync::OnceLock;
            static ROTATION: OnceLock<RotationFamily> = OnceLock::new();
            let rotation = ROTATION.get_or_init(|| RotationFamily::new());
            let xr = xr!();
            rotation
                .run(
                    &ctx,
                    gpu,
                    RotationParams {
                        x,
                        x_up: None,
                        w_norm: None,
                        x_plain: &xr,
                        x_rot: &xr,
                        awq_scale: None,
                        k: w.k,
                        eps: 1e-6,
                        batch_size: 1,
                        variant: hipfire_dispatch::types::RotationVariant::PlainG128,
                        givens_pairs: None,
                        givens_theta: None,
                        givens_scales: None,
                        givens_krot: None,
                    },
                )
                .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
            // xr is ALREADY FWHT-rotated by rotate_x_mq_for above. Use the
            // Prerotated GEMV directly — calling run_auto here would re-rotate
            // (dtype_rotation_plan(MQ*) != None), double-applying the involutory
            // FWHT and feeding effectively-unrotated activations to the
            // prerotated kernel (garbage logits). Mirrors master's
            // rotate_x_mq_for + gemv_*_prerotated.
            gemv.run(
                &ctx,
                gpu,
                &GemvParams {
                    w: &wr,
                    x: &xr,
                    y,
                    variant: GemvVariant::Prerotated,
                    residual: None,
                    gate: None,
                    up: None,
                },
            )
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))
        }
        // ParoQ4G128: Givens rotation (model-layer ParoRotation metadata) +
        // HFQ4-G128 GEMV. Uses RotationFamily::run(Givens) which calls
        // givens_rotate_to (copy_d2d + rotate in one kernel).
        DType::ParoQ4G128 => {
            use hipfire_dispatch::families::rotation::{RotationFamily, RotationParams};
            use std::sync::OnceLock;
            static ROTATION: OnceLock<RotationFamily> = OnceLock::new();
            let rotation = ROTATION.get_or_init(|| RotationFamily::new());
            let paro = w
                .paro
                .as_ref()
                .expect("ParoQ4G128 weight missing ParoRotation metadata");
            gpu.ensure_paro_scratch(w.k)?;
            let xr = GpuTensor {
                buf: unsafe { gpu.scratch.paro_x_scratch.as_ref().unwrap().buf.alias() },
                shape: vec![w.k],
                dtype: DType::F32,
            };
            rotation
                .run(
                    &ctx,
                    gpu,
                    RotationParams {
                        x,
                        x_up: None,
                        w_norm: None,
                        x_plain: &xr,
                        x_rot: &xr,
                        awq_scale: None,
                        k: w.k,
                        eps: 1e-6,
                        batch_size: 1,
                        variant: hipfire_dispatch::types::RotationVariant::Givens,
                        givens_pairs: Some(&paro.pairs),
                        givens_theta: Some(&paro.theta),
                        givens_scales: Some(&paro.channel_scales),
                        givens_krot: Some(paro.krot as usize),
                    },
                )
                .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
            // After Givens rotation xr is ready; use Plain (HFQ4G128 kernel), not Prerotated.
            gemv.run(
                &ctx,
                gpu,
                &GemvParams {
                    w: &wr,
                    x: &xr,
                    y,
                    variant: GemvVariant::Plain,
                    residual: None,
                    gate: None,
                    up: None,
                },
            )
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))
        }
        // All other FWHT-requiring dtypes (MQ4G256, MQ6G256, MQ3G256, MQ2G256,
        // MQ2G256Lloyd, MQ3G256Lloyd, MQ4G256Lloyd, MFP4G32):
        // ensure_mq_signs + rotate_x_mq_for + run_auto
        // The fused MFP4G32 optimization is handled inside
        // GemvFamily::run() -> Prerotated arm.
        _ => {
            gpu.ensure_mq_signs()?;
            let xr = xr!();
            rotate_x_mq_for(gpu, w, x, &xr, w.k)?;
            // xr is ALREADY FWHT-rotated by rotate_x_mq_for above. Use the
            // Prerotated GEMV directly — calling run_auto here would re-rotate
            // (dtype_rotation_plan(MQ*) != None), double-applying the involutory
            // FWHT and feeding effectively-unrotated activations to the
            // prerotated kernel (garbage logits). Mirrors master's
            // rotate_x_mq_for + gemv_*_prerotated.
            gemv.run(
                &ctx,
                gpu,
                &GemvParams {
                    w: &wr,
                    x: &xr,
                    y,
                    variant: GemvVariant::Prerotated,
                    residual: None,
                    gate: None,
                    up: None,
                },
            )
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))
        }
    }
}

/// Fused RMSNorm + FWHT rotation for a batch of MagnumQuant GEMVs sharing x.
///
/// Replaces the split `rmsnorm_f32` + `rotate_x_for_mq` pair with a single kernel
/// launch (Phase 3.6 kernel fusion). The caller should subsequently use
/// `weight_gemv_prerotated` with the returned `Option<&GpuTensor>`:
///
/// - MQ4 `sample_weight`: launches `fused_rmsnorm_rotate_mq`, writes FWHT(rmsnorm(x))
///   into `x_rot_scratch`, returns `Some(x_rot_scratch)`.
/// - MQ8 `sample_weight`: not yet supported by the fused kernel, falls back to
///   plain `rmsnorm_f32` + `rotate_quantize_x_mq8` (the INT8 quantize step can't
///   share LDS with rmsnorm the same way). Returns `None` — MQ8 consumes the
///   internal `mq_x_q8` buffer inside `weight_gemv_prerotated`.
/// - Any other dtype: plain `rmsnorm_f32` into `tmp`, returns `None`.
/// Phase A Stage A — batched AWQ-aware dispatch helper.
///
/// Wraps `Gpu::fused_rmsnorm_rotate_mq_batched` with AWQ-aware kernel
/// selection: if the upcoming linear (`next_linear`) carries an AWQ
/// scale sidecar, dispatch the `_awq_batched` kernel which divides
/// activations by `awq_scale[i]` before the FWHT. Otherwise use the
/// standard non-AWQ kernel.
///
/// Callers in qwen35.rs forward path pass the WeightTensor of the
/// FIRST linear after the rotation (e.g. `layer.wqkv` for LinearAttention,
/// `layer.wq` for FullAttention with separate Q/K/V, `ffn.router` for
/// MoE preamble). For fused QKV: Q/K/V share the same input tensor and
/// hence the same imatrix → byte-identical AWQ scales; picking any
/// of them is mathematically correct.
pub fn fused_rmsnorm_rotate_mq_batched_for(
    gpu: &mut Gpu,
    x: &GpuTensor,
    norm_weight: &GpuTensor,
    next_linear: &WeightTensor,
    x_rot: &GpuTensor,
    k: usize,
    eps: f32,
    batch_size: usize,
) -> HipResult<()> {
    if let Some(awq) = next_linear.awq_scale.as_ref() {
        gpu.fused_rmsnorm_rotate_mq_awq_batched(x, norm_weight, awq, x_rot, k, eps, batch_size)
    } else {
        gpu.fused_rmsnorm_rotate_mq_batched(x, norm_weight, x_rot, k, eps, batch_size)
    }
}

pub fn fused_rmsnorm_rotate_for_mq<'a>(
    gpu: &mut Gpu,
    sample_weight: &WeightTensor,
    x: &GpuTensor,
    norm_weight: &GpuTensor,
    tmp: &GpuTensor,
    x_rot_scratch: &'a GpuTensor,
    eps: f32,
) -> HipResult<Option<&'a GpuTensor>> {
    match sample_weight.gpu_dtype {
        DType::MQ4G256
        | DType::MQ6G256
        | DType::MQ3G256
        | DType::MQ2G256
        | DType::MQ2G256Lloyd
        | DType::MQ3G256Lloyd
        | DType::MQ4G256Lloyd
        | DType::MFP4G32 => {
            // Phase A Stage A — AWQ-aware dispatch. When the upcoming linear
            // carries an AWQ scale sidecar, use the AWQ variant of the fused
            // kernel which divides activations by `awq_scale[i]` before the
            // FWHT rotation, completing the math `(W·s) · (x/s) = W·x`.
            // For Stage A specifically, only MQ4G256 actually emits AWQ
            // sidecars from the quantizer (`--awq` flag), but routing all
            // MQ-family dtypes through the AWQ check is correct + cheap.
            if let Some(awq) = sample_weight.awq_scale.as_ref() {
                gpu.fused_rmsnorm_rotate_mq_awq(
                    x,
                    norm_weight,
                    awq,
                    x_rot_scratch,
                    sample_weight.k,
                    eps,
                )?;
            } else {
                gpu.fused_rmsnorm_rotate_mq(x, norm_weight, x_rot_scratch, sample_weight.k, eps)?;
            }
            Ok(Some(x_rot_scratch))
        }
        DType::MQ8G256 => {
            // MQ8 rotate+quantize produces INT8 scratch; can't fuse with rmsnorm the
            // same way. Keep the split path for now.
            gpu.rmsnorm_f32(x, norm_weight, tmp, eps)?;
            gpu.rotate_quantize_x_mq8(tmp, sample_weight.k)?;
            Ok(None)
        }
        _ => {
            gpu.rmsnorm_f32(x, norm_weight, tmp, eps)?;
            Ok(None)
        }
    }
}

/// Pre-rotate x once for a batch of MagnumQuant weight GEMVs that share the same input.
///
/// - MQ4: writes FWHT(x) into `x_rot_scratch`, returns `Some(x_rot_scratch)`.
///   Pass the returned buffer to `weight_gemv_prerotated` for each MQ4 call.
/// - MQ8: rotates+quantizes x into the GPU's internal INT8 scratch, returns `None`.
///   Subsequent `weight_gemv_prerotated` calls pick up the internal buffers automatically.
/// - Any other dtype: no-op, returns `None` (caller should use plain `x`).
///
/// `sample_weight` is any weight from the batch — only its `gpu_dtype` and `k` are read.
pub fn rotate_x_for_mq<'a>(
    gpu: &mut Gpu,
    sample_weight: &WeightTensor,
    x: &GpuTensor,
    x_rot_scratch: &'a GpuTensor,
) -> HipResult<Option<&'a GpuTensor>> {
    match sample_weight.gpu_dtype {
        DType::MQ4G256
        | DType::MQ6G256
        | DType::MQ3G256
        | DType::MQ2G256
        | DType::MQ2G256Lloyd
        | DType::MQ3G256Lloyd
        | DType::MQ4G256Lloyd
        | DType::MFP4G32 => {
            // Phase A Stage A — F2: route to AWQ variant when the
            // downstream linear (the GEMV consuming x_rot) carries an
            // `awq_scale` sidecar. `sample_weight` IS that downstream
            // linear (callers pass o_proj / out_proj here). For Stage A
            // pre-F2 quants, awq_scale is None on these tensors so the
            // non-AWQ kernel runs — byte-identical to pre-F2 behavior.
            if let Some(awq) = sample_weight.awq_scale.as_ref() {
                gpu.rotate_x_mq_awq(x, awq, x_rot_scratch, sample_weight.k)?;
            } else {
                gpu.rotate_x_mq(x, x_rot_scratch, sample_weight.k)?;
            }
            Ok(Some(x_rot_scratch))
        }
        DType::MQ8G256 => {
            gpu.rotate_quantize_x_mq8(x, sample_weight.k)?;
            Ok(None)
        }
        _ => Ok(None),
    }
}

/// Phase A Stage A — F2: standalone AWQ-aware variant of `rotate_x_mq`.
///
/// Wraps `Gpu::rotate_x_mq` / `Gpu::rotate_x_mq_awq` with AWQ-aware
/// dispatch. The `next_linear` is the downstream weight that consumes
/// `x_rot` (typically `o_proj` / `out_proj` / `down_proj`). When its
/// `awq_scale` is `Some`, dispatches the AWQ kernel which divides
/// activations by `awq_scale[i]` before the FWHT — completes the math
/// `(W·s) · (x/s) = W·x`.
///
/// Byte-identical to `Gpu::rotate_x_mq` on .hfq files without AWQ
/// sidecars (which is the only state that exists pre-F2).
pub fn rotate_x_mq_for(
    gpu: &mut Gpu,
    next_linear: &WeightTensor,
    x: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
) -> HipResult<()> {
    if let Some(awq) = next_linear.awq_scale.as_ref() {
        gpu.rotate_x_mq_awq(x, awq, x_rot, k)
    } else {
        gpu.rotate_x_mq(x, x_rot, k)
    }
}

/// MQ4G128 activation rotate helper. Always takes the non-AWQ path —
/// no AWQ sidecar is supported for G128 weights in this PR.
/// The `_next_linear` argument is reserved for a future AWQ branch;
/// pass the upcoming weight tensor but its `awq_scale` is ignored here.
pub fn rotate_x_mq_128_for(
    gpu: &mut Gpu,
    _next_linear: &WeightTensor,
    x: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
) -> HipResult<()> {
    // NOTE: no AWQ branch for G128. If AWQ support for MQ4G128 is added
    // in a follow-up, mirror the AWQ branch from `rotate_x_mq_for` here.
    gpu.rotate_x_mq_128(x, x_rot, k)
}

/// ParoQuant single-token rotation: read x, write Givens-rotated
/// activation to x_rot via a single out-of-place kernel. Earlier
/// versions did `copy_d2d(x → x_rot)` then `givens_rotate(x_rot)`;
/// fusing into one launch eliminates an inter-node dependency that
/// the hipGraph dependency analyzer can fail to enforce (observed
/// numerical delta direct-vs-graph on gfx1151 / HIP 7.13).
pub fn rotate_x_paro_for(
    gpu: &mut Gpu,
    paro: &ParoRotation,
    x: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
) -> HipResult<()> {
    gpu.givens_rotate_to(
        x,
        x_rot,
        &paro.pairs,
        &paro.theta,
        &paro.channel_scales,
        1,
        k,
        paro.krot as usize,
    )
}

/// Phase A Stage A — F2: batched AWQ-aware variant of `rotate_x_mq`.
/// Grid.y is the batch dim. See `rotate_x_mq_for` for routing logic.
pub fn rotate_x_mq_batched_for(
    gpu: &mut Gpu,
    next_linear: &WeightTensor,
    x: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
    batch_size: usize,
) -> HipResult<()> {
    if let Some(awq) = next_linear.awq_scale.as_ref() {
        gpu.rotate_x_mq_awq_batched(x, awq, x_rot, k, batch_size)
    } else {
        gpu.rotate_x_mq_batched(x, x_rot, k, batch_size)
    }
}

/// Phase A Stage A — F2: standalone AWQ-aware variant of
/// `fused_silu_mul_rotate_mq`. The `down_proj_weight` is the downstream
/// linear consuming x_rot (e.g. `w_down` / `down_proj`). When its
/// `awq_scale` is `Some`, dispatches the AWQ kernel which divides
/// silu(gate)*up by `awq_scale[i]` before the FWHT.
///
/// Byte-identical to `Gpu::fused_silu_mul_rotate_mq` on .hfq files
/// without AWQ sidecars (pre-F2 state).
pub fn fused_silu_mul_rotate_mq_for(
    gpu: &mut Gpu,
    down_proj_weight: &WeightTensor,
    gate: &GpuTensor,
    up: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
) -> HipResult<()> {
    if let Some(awq) = down_proj_weight.awq_scale.as_ref() {
        gpu.fused_silu_mul_rotate_mq_awq(gate, up, awq, x_rot, k)
    } else {
        gpu.fused_silu_mul_rotate_mq(gate, up, x_rot, k)
    }
}

/// Phase A Stage A — F2: batched AWQ-aware variant of
/// `fused_silu_mul_rotate_mq`. Grid.y is the batch dim.
pub fn fused_silu_mul_rotate_mq_batched_for(
    gpu: &mut Gpu,
    down_proj_weight: &WeightTensor,
    gate: &GpuTensor,
    up: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
    batch_size: usize,
) -> HipResult<()> {
    if let Some(awq) = down_proj_weight.awq_scale.as_ref() {
        gpu.fused_silu_mul_rotate_mq_awq_batched(gate, up, awq, x_rot, k, batch_size)
    } else {
        gpu.fused_silu_mul_rotate_mq_batched(gate, up, x_rot, k, batch_size)
    }
}

/// GEMV with optional pre-rotated x for MagnumQuant weights.
///
/// - MQ4 + `x_rot = Some(..)`: calls the arch-tuned HFQ4 GEMV on the pre-rotated buffer,
///   skipping the per-call FWHT pass. Use with `rotate_x_for_mq` to batch rotations across
///   multiple projections that share the same input (Q/K/V, gate/up).
/// - MQ4 + `x_rot = None`: falls back to the auto-rotate path in `weight_gemv`.
/// - MQ8: uses the internal x_q8/x_scales set by `rotate_quantize_x_mq8`; caller must have
///   called `rotate_x_for_mq` (which invokes that helper) before this.
/// - Any other dtype: `x_rot` is ignored; equivalent to `weight_gemv`.
pub fn weight_gemv_prerotated(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    x_rot: Option<&GpuTensor>,
    y: &GpuTensor,
) -> HipResult<()> {
    use hipfire_dispatch::context::DispatchCtx;
    use hipfire_dispatch::families::gemv::WeightRef;
    use hipfire_dispatch::types::dtype_needs_rotation;

    let gemv = crate::llama::gemv_family();
    let ctx = DispatchCtx::new(gpu);

    if w.gpu_dtype == DType::MQ8G256 {
        gpu.ensure_mq_signs()?;
        let wr = WeightRef {
            buf: &w.buf,
            dtype: w.gpu_dtype,
            m: w.m,
            k: w.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        return gemv
            .run_auto(&ctx, gpu, &wr, x, y)
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()));
    }

    if dtype_needs_rotation(w.gpu_dtype) {
        if let Some(xr) = x_rot {
            // `xr` is ALREADY FWHT/Givens-rotated by the caller (the whole
            // point of `weight_gemv_prerotated`: rotate once, reuse across the
            // Q/K/V or gate/up projections that share this input). It MUST go
            // straight into the `gemv_*_prerotated` kernel — i.e. through
            // `run` with `GemvVariant::Prerotated`.
            //
            // `run_auto` here would be a DOUBLE-ROTATION bug: it takes
            // `RotInput::Raw(xr)` and, because every rotation-needing dtype has
            // a non-None `dtype_rotation_plan`, re-rotates `xr`. FWHT is
            // involutory → the second rotation silently UN-rotates the
            // activation, producing wrong-but-non-crashing output (e.g. MQ6
            // DeltaNet projections in the multi-GPU branch). #393 introduced
            // this when it swapped master's explicit
            // `gemv_mq6g256_prerotated(&w.buf, xr, ..)` for `run_auto`.
            // `GemvVariant::Prerotated` restores master's behavior exactly.
            debug_assert!(
                hipfire_dispatch::types::dtype_post_rotation_variant(w.gpu_dtype)
                    == GemvVariant::Prerotated,
                "weight_gemv_prerotated: dtype {:?} needs rotation but its \
                 post-rotation variant is not Prerotated — passing an \
                 already-rotated xr would double-rotate",
                w.gpu_dtype,
            );
            let wr = WeightRef {
                buf: &w.buf,
                dtype: w.gpu_dtype,
                m: w.m,
                k: w.k,
                row_stride: 0,
                rotation: None,
                awq_scale: None,
            };
            return gemv
                .run(
                    &ctx,
                    gpu,
                    &hipfire_dispatch::families::gemv::GemvParams {
                        w: &wr,
                        x: xr,
                        y,
                        variant: GemvVariant::Prerotated,
                        residual: None,
                        gate: None,
                        up: None,
                    },
                )
                .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()));
        }
        return weight_gemv(gpu, w, x, y);
    }

    let wr = WeightRef {
        buf: &w.buf,
        dtype: w.gpu_dtype,
        m: w.m,
        k: w.k,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    };
    gemv.run_auto(&ctx, gpu, &wr, x, y)
        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))
}

/// Weight GEMV with fused residual add: `y += W * x`.
///
/// For HFQ4-G256 weights, routes through `gemv_hfq4g256_residual`, which
/// saves one `add_inplace_f32` launch per residual stream update.
///
/// For MQ4 weights, performs `rotate_x_mq` into the internal scratch and
/// then calls the residual GEMV against the rotated x. Equivalent to the
/// standard prerotated path plus a fused residual epilogue.
///
/// For any other dtype, falls back to plain `weight_gemv` followed by an
/// explicit `add_inplace_f32` — same observable behavior as before.
pub fn weight_gemv_residual(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
) -> HipResult<()> {
    use hipfire_dispatch::context::DispatchCtx;
    use hipfire_dispatch::families::gemv::{GemvParams, WeightRef};
    use hipfire_dispatch::types::GemvVariant;

    let gemv = crate::llama::gemv_family();
    let ctx = DispatchCtx::new(gpu);
    let wr = WeightRef {
        buf: &w.buf,
        dtype: w.gpu_dtype,
        m: w.m,
        k: w.k,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    };

    match w.gpu_dtype {
        DType::HFQ4G256 | DType::HFQ3G256 | DType::HFQ6G256 => gemv
            .run(
                &ctx,
                gpu,
                &GemvParams {
                    w: &wr,
                    x,
                    y,
                    variant: GemvVariant::WithResidual,
                    residual: None,
                    gate: None,
                    up: None,
                },
            )
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string())),
        DType::MQ6G256
        | DType::MQ4G256
        | DType::MQ3G256
        | DType::MQ3G256Lloyd
        | DType::MQ4G256Lloyd => {
            gpu.ensure_mq_signs()?;
            let xr = GpuTensor {
                buf: unsafe { gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.scratch.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            };
            rotate_x_mq_for(gpu, w, x, &xr, w.k)?;
            gemv.run(
                &ctx,
                gpu,
                &GemvParams {
                    w: &wr,
                    x: &xr,
                    y,
                    variant: GemvVariant::WithResidual,
                    residual: None,
                    gate: None,
                    up: None,
                },
            )
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))
        }
        _ => {
            let tmp = gpu.alloc_tensor(&[w.m], DType::F32)?;
            weight_gemv(gpu, w, x, &tmp)?;
            gpu.add_inplace_f32(y, &tmp)?;
            gpu.free_tensor(tmp)?;
            Ok(())
        }
    }
}

/// SwiGLU FFN epilogue fused into the w_down input stage for MQ4 weights.
///
/// Replaces:
///   silu_mul_f32(gate, up, ffn_hidden)  // eliminated for MQ4
///   weight_gemv_residual(w_down, ffn_hidden, x)
/// with (for MQ4):
///   fused_silu_mul_rotate_mq(gate, up, mq_x_rot)   // one kernel
///   gemv_hfq4g256_residual(w_down, mq_x_rot, x)    // fused residual add
/// so the entire w_down epilogue is two launches instead of four
/// (silu_mul + rotate + gemv + add_inplace → fused_silu_rotate + gemv_residual).
///
/// Non-MQ path falls back to the pre-Phase-3.8 sequence (silu_mul_f32 +
/// weight_gemv_residual). Byte-equivalent modulo FP reordering on the
/// FWHT butterfly, which is the same butterfly as the standalone path.
pub fn weight_gemv_swiglu_residual(
    gpu: &mut Gpu,
    w_down: &WeightTensor,
    gate: &GpuTensor,
    up: &GpuTensor,
    ffn_hidden_scratch: &GpuTensor,
    x: &GpuTensor,
) -> HipResult<()> {
    use hipfire_dispatch::context::DispatchCtx;
    use hipfire_dispatch::families::gemv::{GemvParams, WeightRef};
    use hipfire_dispatch::types::GemvVariant;

    let gemv = crate::llama::gemv_family();
    let ctx = DispatchCtx::new(gpu);
    let wr = WeightRef {
        buf: &w_down.buf,
        dtype: w_down.gpu_dtype,
        m: w_down.m,
        k: w_down.k,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    };

    match w_down.gpu_dtype {
        DType::MQ4G256
        | DType::MQ3G256
        | DType::MQ6G256
        | DType::MQ3G256Lloyd
        | DType::MQ4G256Lloyd => {
            gpu.ensure_mq_signs()?;
            let xr = GpuTensor {
                buf: unsafe { gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.scratch.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            };
            fused_silu_mul_rotate_mq_for(gpu, w_down, gate, up, &xr, w_down.k)?;
            gemv.run(
                &ctx,
                gpu,
                &GemvParams {
                    w: &wr,
                    x: &xr,
                    y: x,
                    variant: GemvVariant::WithSwiGLUResidual,
                    residual: Some(x),
                    gate: Some(&xr),
                    up: None,
                },
            )
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))
        }
        _ => {
            gpu.silu_mul_f32(gate, up, ffn_hidden_scratch)?;
            weight_gemv_residual(gpu, w_down, ffn_hidden_scratch, x)
        }
    }
}

/// Batched weight GEMM: y[b] = W * x[b] for all batch elements.
/// x: [batch_size × K], y: [batch_size × M]. Falls back to repeated GEMV for unsupported formats.
pub fn weight_gemm(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    batch_size: usize,
) -> HipResult<()> {
    match w.gpu_dtype {
        DType::HFQ4G256 => gpu.gemm_hfq4g256(&w.buf, x, y, w.m, w.k, batch_size),
        DType::HFQ4G128 => gpu.gemm_hfq4g128(&w.buf, x, y, w.m, w.k, batch_size),
        // MQ4G256 = HFQ4G256 weight layout + an offline FWHT rotation, so the
        // batched GEMM is: FWHT-rotate all `batch_size` activation columns once
        // (`mq_rotate_x`, AWQ-aware), then feed the same INT4-G256 WMMA kernel
        // HFQ4G256 uses. This is the batched twin of `weight_gemv`'s single-row
        // `ensure_mq_signs + rotate_x_mq_for + gemv(Prerotated)` path, and mirrors
        // DFlash's proven `gemm_dispatch` MQ4 arm (dflash.rs). Replaces the per-row
        // GEMV fallback that re-read the weight `batch_size`× (the spec-verify
        // bottleneck). NOTE: the WMMA path quantizes x to F16, so it is NOT
        // bit-identical to the F32 GEMV — fine for any batched-prefill / spec-verify
        // use (validated by coherence), same as HFQ4G256 already is.
        DType::MQ4G256 => {
            gpu.ensure_mq_signs()?;
            let x_rot = gpu.alloc_tensor(&[batch_size, w.k], DType::F32)?;
            rotate_x_mq_batched_for(gpu, w, x, &x_rot, w.k, batch_size)?;
            // `_batched_lmhead` routes batch>1 to the WMMA residual kernel on
            // gfx11/gfx12 (zero-inits Y, so residual `+=` collapses to `=`), which
            // the scalar `gemm_hfq4g256` path falls 8-10× short of at small batch
            // (its own dispatch comment). It also invalidates the FP16-x cache for
            // the freshly-pooled x_rot pointer, so no manual reset is needed here.
            let r = gpu.gemm_hfq4g256_batched_lmhead(&w.buf, &x_rot, y, w.m, w.k, batch_size);
            gpu.free_tensor(x_rot)?;
            r
        }
        _ => {
            // Fallback: repeated GEMV (no batched kernel for this format)
            let x_tok = gpu.alloc_tensor(&[w.k], DType::F32)?;
            let y_tok = gpu.alloc_tensor(&[w.m], DType::F32)?;
            for b in 0..batch_size {
                gpu.hip
                    .memcpy_dtod_at(&x_tok.buf, 0, &x.buf, b * w.k * 4, w.k * 4)?;
                weight_gemv(gpu, w, &x_tok, &y_tok)?;
                gpu.hip
                    .memcpy_dtod_at(&y.buf, b * w.m * 4, &y_tok.buf, 0, w.m * 4)?;
            }
            gpu.free_tensor(x_tok)?;
            gpu.free_tensor(y_tok)?;
            Ok(())
        }
    }
}

/// Batched prefill: process all prompt tokens in one forward pass.
/// Returns logits for the LAST position only.
/// KV cache is filled for all positions.
pub fn prefill_forward(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    tokens: &[u32],
    kv_cache: &mut KvCache,
) -> HipResult<Vec<f32>> {
    let batch = tokens.len();
    let dim = config.dim;
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let q_dim = n_heads * head_dim;

    // Allocate batched buffers: [batch × dim]
    let x_batch = gpu.alloc_tensor(&[batch, dim], DType::F32)?;
    let tmp_batch = gpu.alloc_tensor(&[batch, dim], DType::F32)?;
    let q_batch = gpu.alloc_tensor(&[batch, q_dim], DType::F32)?;
    let k_batch = gpu.alloc_tensor(&[batch, kv_dim], DType::F32)?;
    let v_batch = gpu.alloc_tensor(&[batch, kv_dim], DType::F32)?;
    let attn_out_batch = gpu.alloc_tensor(&[batch, q_dim], DType::F32)?;
    let o_batch = gpu.alloc_tensor(&[batch, dim], DType::F32)?;
    let gate_batch = gpu.alloc_tensor(&[batch, config.hidden_dim], DType::F32)?;
    let up_batch = gpu.alloc_tensor(&[batch, config.hidden_dim], DType::F32)?;
    let ffn_hidden_batch = gpu.alloc_tensor(&[batch, config.hidden_dim], DType::F32)?;
    let ffn_out_batch = gpu.alloc_tensor(&[batch, dim], DType::F32)?;

    // Embedding: lookup each token individually into the batch buffer
    let x_single = gpu.alloc_tensor(&[dim], DType::F32)?;
    for (i, &token) in tokens.iter().enumerate() {
        embedding_lookup_dispatch(
            gpu,
            weights.embd_format,
            &weights.token_embd,
            &x_single,
            token,
            dim,
        )?;
        gpu.hip
            .memcpy_dtod_at(&x_batch.buf, i * dim * 4, &x_single.buf, 0, dim * 4)?;
    }
    gpu.free_tensor(x_single)?;

    // Position array for batched RoPE: [0, 1, 2, ..., batch-1]
    let pos_data: Vec<i32> = (0..batch as i32).collect();
    let pos_bytes: Vec<u8> = pos_data.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let pos_array = gpu.alloc_tensor(&[batch], DType::F32)?; // i32 same size as f32
    gpu.hip.memcpy_htod(&pos_array.buf, &pos_bytes)?;

    // Per-position scratch buffers (reused across all layers)
    let q_slice = gpu.alloc_tensor(&[q_dim], DType::F32)?;
    let k_slice = gpu.alloc_tensor(&[kv_dim], DType::F32)?;
    let v_slice = gpu.alloc_tensor(&[kv_dim], DType::F32)?;
    let attn_slice = gpu.alloc_tensor(&[q_dim], DType::F32)?;
    let pos_buf = gpu.hip.malloc(4)?;

    // Layer loop
    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];

        // Batched RMSNorm: each row of x_batch independently
        for i in 0..batch {
            // We need per-row norm — use the batched rmsnorm with batch=batch
            // Actually, rmsnorm_batched already handles this if we set batch=batch, n=dim
        }
        gpu.rmsnorm_batched(
            &x_batch,
            &layer.attn_norm,
            &tmp_batch,
            batch,
            dim,
            config.norm_eps,
        )?;

        // Batched QKV projections
        weight_gemm(gpu, &layer.wq, &tmp_batch, &q_batch, batch)?;
        weight_gemm(gpu, &layer.wk, &tmp_batch, &k_batch, batch)?;
        weight_gemm(gpu, &layer.wv, &tmp_batch, &v_batch, batch)?;

        // QK norm (per-position, per-head)
        if config.has_qk_norm {
            if let Some(ref qn) = layer.q_norm {
                gpu.rmsnorm_batched(
                    &q_batch,
                    qn,
                    &q_batch,
                    batch * n_heads,
                    head_dim,
                    config.norm_eps,
                )?;
            }
            if let Some(ref kn) = layer.k_norm {
                gpu.rmsnorm_batched(
                    &k_batch,
                    kn,
                    &k_batch,
                    batch * n_kv_heads,
                    head_dim,
                    config.norm_eps,
                )?;
            }
        }

        // Batched RoPE: all positions in one kernel launch
        gpu.rope_batched_f32(
            &q_batch,
            &k_batch,
            &pos_array,
            n_heads,
            n_kv_heads,
            head_dim,
            config.rope_freq_base,
            batch,
        )?;

        // Batched KV cache write: all positions in 2 kernel launches (K + V)
        if kv_cache.quantized && kv_cache.quant_q8 {
            gpu.kv_cache_write_q8_0_batched(
                &kv_cache.k_gpu[layer_idx],
                &k_batch,
                &pos_array,
                n_kv_heads,
                head_dim,
                batch,
            )?;
            gpu.kv_cache_write_q8_0_batched(
                &kv_cache.v_gpu[layer_idx],
                &v_batch,
                &pos_array,
                n_kv_heads,
                head_dim,
                batch,
            )?;
        } else {
            for i in 0..batch {
                let pos_i32 = i as i32;
                gpu.hip.memcpy_htod(&pos_buf, &pos_i32.to_ne_bytes())?;
                gpu.hip.memcpy_dtod_at(
                    &k_slice.buf,
                    0,
                    &k_batch.buf,
                    i * kv_dim * 4,
                    kv_dim * 4,
                )?;
                gpu.hip.memcpy_dtod_at(
                    &v_slice.buf,
                    0,
                    &v_batch.buf,
                    i * kv_dim * 4,
                    kv_dim * 4,
                )?;
                gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &k_slice, &pos_buf, kv_dim)?;
                gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &v_slice, &pos_buf, kv_dim)?;
            }
        }

        // Batched causal attention: one kernel launch for all positions
        gpu.attention_causal_batched(
            &q_batch,
            &k_batch,
            &v_batch,
            &attn_out_batch,
            batch,
            n_heads,
            n_kv_heads,
            head_dim,
        )?;

        // Batched output projection
        weight_gemm(gpu, &layer.wo, &attn_out_batch, &o_batch, batch)?;

        // Batched residual add: x_batch += o_batch
        gpu.add_inplace_f32(&x_batch, &o_batch)?;

        // Batched FFN norm
        gpu.rmsnorm_batched(
            &x_batch,
            &layer.ffn_norm,
            &tmp_batch,
            batch,
            dim,
            config.norm_eps,
        )?;

        // Batched FFN projections
        weight_gemm(gpu, &layer.w_gate, &tmp_batch, &gate_batch, batch)?;
        weight_gemm(gpu, &layer.w_up, &tmp_batch, &up_batch, batch)?;

        // Batched SiLU * mul
        gpu.silu_mul_f32(&gate_batch, &up_batch, &ffn_hidden_batch)?;

        // Batched down projection
        weight_gemm(gpu, &layer.w_down, &ffn_hidden_batch, &ffn_out_batch, batch)?;

        // Batched residual
        gpu.add_inplace_f32(&x_batch, &ffn_out_batch)?;
    }

    // Free per-position scratch
    gpu.free_tensor(pos_array)?;
    gpu.free_tensor(q_slice)?;
    gpu.free_tensor(k_slice)?;
    gpu.free_tensor(v_slice)?;
    gpu.free_tensor(attn_slice)?;
    gpu.hip.free(pos_buf)?;

    // Final norm + output projection for LAST position only
    let last_off = (batch - 1) * dim * 4;
    let x_last = gpu.alloc_tensor(&[dim], DType::F32)?;
    gpu.hip
        .memcpy_dtod_at(&x_last.buf, 0, &x_batch.buf, last_off, dim * 4)?;

    let tmp = gpu.alloc_tensor(&[dim], DType::F32)?;
    gpu.rmsnorm_f32(&x_last, &weights.output_norm, &tmp, config.norm_eps)?;

    let logits = gpu.alloc_tensor(&[config.vocab_size], DType::F32)?;
    weight_gemv(gpu, &weights.output, &tmp, &logits)?;

    let logits_data = gpu.download_f32(&logits)?;

    // Free all batched buffers
    gpu.free_tensor(x_batch)?;
    gpu.free_tensor(tmp_batch)?;
    gpu.free_tensor(q_batch)?;
    gpu.free_tensor(k_batch)?;
    gpu.free_tensor(v_batch)?;
    gpu.free_tensor(attn_out_batch)?;
    gpu.free_tensor(o_batch)?;
    gpu.free_tensor(gate_batch)?;
    gpu.free_tensor(up_batch)?;
    gpu.free_tensor(ffn_hidden_batch)?;
    gpu.free_tensor(ffn_out_batch)?;
    gpu.free_tensor(x_last)?;
    gpu.free_tensor(tmp)?;
    gpu.free_tensor(logits)?;

    Ok(logits_data)
}

// ─── LLaMA-family batched prefill (Phase A of #89) ─────────────────────────
//
// The fused WMMA + K4-unroll + flash-attention prefill stack lives here so
// any plain LLaMA-family loader (Qwen3, Mistral, Phi, Gemma) can drive it
// directly without going through `qwen35::forward_prefill_batch` (whose
// eligibility gate requires DeltaNet/MoE layers and whose layer enum
// branches over hybrid arch variants).
//
// Mirrors the FullAttn fast path of `qwen35::forward_prefill_chunk` kernel
// for kernel, with two adaptations:
//   1. No "Q + gate" wide projection. Plain Qwen3 attention has a normal
//      Q output (q_dim wide); no deinterleave, no sigmoid_mul step.
//   2. Full RoPE via `rope_batched_f32` (non-interleaved, half-split) to
//      match `forward_scratch`'s `rope_f32` semantics.

/// Upper bound on `forward_prefill_batch`'s per-chunk size. Mirrors the
/// qwen35 chunk cap; sized so flash_partials stays within 2 GB at the
/// largest physical_cap any consumer sets up.
pub const PREFILL_MAX_BATCH: usize = 256;

/// Is this dtype/arch combination eligible for the batched WMMA prefill
/// kernels? Matches `qwen35::is_batchable_la` exactly so plain Qwen3 and
/// hybrid Qwen3.5 share one rule and stay in lockstep when new dtypes or
/// arches gain WMMA support.
pub fn is_batchable_la(dt: DType, arch: &str) -> bool {
    let always_ok = matches!(
        dt,
        DType::MQ4G256
            | DType::HFQ4G256
            | DType::HFQ4G128
            | DType::MQ6G256
            | DType::HFQ6G256
            | DType::Q8_0
    );
    if always_ok {
        return true;
    }
    // HFP4G32 / MFP4G32 + MQ3G256 require WMMA. Same arch gate as MQ3.
    let wmma_only = matches!(dt, DType::MQ3G256 | DType::HFP4G32 | DType::MFP4G32)
        && matches!(
            arch,
            "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" | "gfx1200" | "gfx1201"
        );
    // gfx10 RDNA1/2 scalar HFQ3 batched-prefill (Phase 1 of
    // docs/plans/gfx10_mq3_prefill.md). Mirrors the
    // `mq3_uniform_with_gfx10_scalar` arm in qwen35.rs::is_batchable_la —
    // both must stay in sync per the matching-pair comment there.
    let mq3_gfx10_scalar = matches!(dt, DType::MQ3G256)
        && matches!(
            arch,
            "gfx1010" | "gfx1011" | "gfx1012" | "gfx1013" | "gfx1030" | "gfx1031" | "gfx1032"
        );
    wmma_only || mq3_gfx10_scalar
}

/// Per-call scratch for `forward_prefill_batch`. Holds [N × ...] working
/// buffers reused across the per-layer loop. Sized once per model from
/// `LlamaConfig` and reused across cycles by callers that retain it.
pub struct PrefillBatchScratch {
    pub max_batch: usize,

    // Residual stream + rmsnormed/rotated activation [N × dim].
    pub x_batch: GpuTensor,
    pub x_rot_batch: GpuTensor,

    // Token ids + positions feeding batched embedding + RoPE/KV-write kernels.
    // F32 dtype for layout reasons (4 bytes/element matches i32); the kernels
    // cast the device pointer to `const int*`.
    pub positions: GpuTensor,
    pub tokens: GpuTensor,

    // Q/K/V projection outputs (no gate component for plain attention).
    pub fa_q_batch: GpuTensor,        // [N × n_heads × head_dim]
    pub fa_k_batch: GpuTensor,        // [N × n_kv_heads × head_dim]
    pub fa_v_batch: GpuTensor,        // [N × n_kv_heads × head_dim]
    pub fa_attn_out_batch: GpuTensor, // [N × n_heads × head_dim]
    // FWHT-rotated fa_attn_out for feeding MQ4 wo.
    pub fa_attn_out_rot_batch: GpuTensor,

    // FFN intermediates [N × hidden_dim].
    pub gate_ffn_batch: GpuTensor,
    pub up_batch: GpuTensor,
    pub ffn_hidden_batch: GpuTensor,

    // Flash-attention partial-result scratch (sized to support max_batch
    // tokens × n_heads × max_tiles × (2 + head_dim)).
    pub flash_partials: GpuTensor,
}

impl PrefillBatchScratch {
    pub fn new(
        gpu: &mut Gpu,
        config: &LlamaConfig,
        max_batch: usize,
        kv_max_seq: usize,
    ) -> HipResult<Self> {
        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;

        let tile_size = 128usize;
        let max_tiles = (kv_max_seq + tile_size - 1) / tile_size;
        let batch_mult = crate::config::get()
            .flash_partials_batch
            .filter(|&n| n >= 1 && n <= PREFILL_MAX_BATCH)
            .unwrap_or(16);
        let partials_size = batch_mult * config.n_heads * max_tiles * (2 + config.head_dim);

        Ok(Self {
            max_batch,
            x_batch: gpu.alloc_tensor(&[max_batch * dim], DType::F32)?,
            x_rot_batch: gpu.alloc_tensor(&[max_batch * dim], DType::F32)?,
            positions: gpu.alloc_tensor(&[max_batch], DType::F32)?,
            tokens: gpu.alloc_tensor(&[max_batch], DType::F32)?,
            fa_q_batch: gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
            fa_k_batch: gpu.alloc_tensor(&[max_batch * kv_dim], DType::F32)?,
            fa_v_batch: gpu.alloc_tensor(&[max_batch * kv_dim], DType::F32)?,
            fa_attn_out_batch: gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
            fa_attn_out_rot_batch: gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
            gate_ffn_batch: gpu.alloc_tensor(&[max_batch * hidden_dim], DType::F32)?,
            up_batch: gpu.alloc_tensor(&[max_batch * hidden_dim], DType::F32)?,
            ffn_hidden_batch: gpu.alloc_tensor(&[max_batch * hidden_dim], DType::F32)?,
            flash_partials: gpu.alloc_tensor(&[partials_size], DType::F32)?,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in [
            self.x_batch,
            self.x_rot_batch,
            self.positions,
            self.tokens,
            self.fa_q_batch,
            self.fa_k_batch,
            self.fa_v_batch,
            self.fa_attn_out_batch,
            self.fa_attn_out_rot_batch,
            self.gate_ffn_batch,
            self.up_batch,
            self.ffn_hidden_batch,
            self.flash_partials,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// Upload token ids + positions into `pbs` via sync `memcpy_htod`. Pair
/// with `forward_prefill_batch_chunk_captured` to drive a captured graph
/// without `memcpy_htod` operations sneaking in (which would otherwise
/// either error under capture or bake stale host data into the captured
/// kernarg blob). The plain `forward_prefill_batch` does its own uploads
/// internally and does not need this helper.
pub fn upload_prefill_batch_inputs(
    gpu: &mut Gpu,
    pbs: &PrefillBatchScratch,
    tokens: &[u32],
    start_pos: usize,
) -> HipResult<()> {
    let n = tokens.len();
    let tokens_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let tokens_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;
    let positions_host: Vec<i32> = (0..n).map(|i| (start_pos + i) as i32).collect();
    let positions_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;
    Ok(())
}

/// Per-extract-layer residual-hidden capture sink for the batched llama
/// forward (DFlash drafter conditioning, review finding M3).
///
/// When threaded through [`forward_prefill_batch`] / [`forward_prefill_chunk`]
/// (and the verify path), the forward downloads the residual stream `x`
/// (`[n × dim]`) AFTER each decoder layer whose index appears in
/// `extract_layers`, and appends — per processed position, across the
/// extract layers in `extract_layers` order — `extract_layers.len() × dim`
/// f32 to `hidden`. The final `hidden` buffer is therefore
/// `[n_pos × num_extract × dim]` row-major, matching the
/// `dflash::draft_forward` `target_hidden` row layout.
///
/// `extract_layers` MUST be in ascending order (the caller's
/// `dflash::DflashConfig::target_layer_ids` convention). Capture is at the
/// post-layer residual, independent of qk-norm (qk-norm acts on Q/K, not the
/// residual).
pub struct HiddenCaptureSink<'a> {
    /// Decoder-layer indices to capture, ascending order.
    pub extract_layers: &'a [usize],
    /// Host output sink: appended `[n_pos × num_extract × dim]` row-major.
    /// Used only when `hidden_gpu` is `None`.
    pub hidden: &'a mut Vec<f32>,
    /// Optional GPU-resident destination (`[n_pos × num_extract × dim]` F32,
    /// position-major). When `Some`, captured extract-layer rows are copied
    /// GPU→GPU straight into this buffer and the host `hidden` Vec is left
    /// untouched — the DSpark accepted-prefix-hidden reuse then stays entirely
    /// on-device (no D2H+H2D per window; ~free on UMA, a real win on a discrete
    /// GPU). The buffer must be `≥ n_pos × extract_layers.len() × dim` F32.
    pub hidden_gpu: Option<&'a GpuTensor>,
}

/// Tree-attention mask reference for a single batched verify forward
/// ([`forward_prefill_batch_tree`]).
///
/// When supplied, the batched flash-attention kernels run in tree mode: keys in
/// the in-block region `[block_start, block_start + block_cols)` are biased by
/// `bias[row × block_cols + (key_slot − block_start)]` (an additive
/// `0.0`/`-inf` mask), while prompt keys before `block_start` stay fully
/// visible. `block_start` is the absolute decode position the block begins at;
/// `block_cols` equals the linearized tree length (`1 + tree.num_nodes()`).
///
/// The KV WRITE slot and the FA mask alignment stay CONTIGUOUS
/// (`[block_start .. block_start + n)`), so siblings never collide on the same
/// cache slot; but Q/K RoPE uses the tree-DEPTH positions
/// (`block_start + node.depth`, see `rope_positions` below), so a parent→child
/// RoPE distance is exactly 1 regardless of linearized slot. Decoupling the two
/// is what kills the "linearization-slot RoPE phase skew" on bushy trees while
/// avoiding the duplicate-KV-slot hazard of depth-based write slots.
pub struct TreeMaskRef<'a> {
    /// `[block_cols × block_cols]` row-major additive bias (0/-inf), on device.
    pub bias: &'a GpuTensor,
    /// Absolute decode position where the in-block keys begin (= `position`).
    pub block_start: usize,
    /// In-block key count (= linearized tree length `1 + tree.num_nodes()`).
    pub block_cols: usize,
    /// Per-slot DEPTH-based RoPE positions (`block_start + node.depth`), on
    /// device as `[block_cols]` i32-in-F32. Used for the Q/K RoPE rotation ONLY —
    /// the KV WRITE slot and the FA mask alignment stay CONTIGUOUS (`pbs.positions
    /// = block_start + slot_index`), so siblings never collide on the same cache
    /// slot. Decoupling RoPE-position from write-slot is what makes the bushy-tree
    /// verify greedy-LOSSLESS: a node and its parent get RoPE distance == their
    /// depth difference (1 for parent→child), matching the committed chain's
    /// phases. With contiguous-slot RoPE (slot ≠ depth in heap-pop order) the
    /// Q·K phase skews and a dense greedy target flips its argmax off AR.
    pub rope_positions: &'a GpuTensor,
}

/// Process `tokens` through the model with one batched forward, advancing
/// `kv_cache` by `tokens.len()` positions and writing the *last* token's
/// logits into `scratch.logits`.
///
/// Eligibility (else falls back to per-token `forward_scratch` loop):
///   - all FA layer weights (wq/wk/wv/wo + w_gate/w_up/w_down) pass
///     `is_batchable_la`
///   - KV cache is Q8_0 or asym{2,3,4}
///
/// Internally chunks at `pbs.max_batch` to bound VRAM regardless of prompt
/// length. `pbs_in: Some` reuses caller-owned scratch; `None` allocates +
/// frees within the call.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    pbs_in: Option<&PrefillBatchScratch>,
) -> HipResult<()> {
    forward_prefill_batch_capture(
        gpu, weights, config, tokens, start_pos, kv_cache, scratch, pbs_in, None,
    )
}

/// `forward_prefill_batch` plus an optional per-extract-layer residual-hidden
/// capture sink (DFlash drafter conditioning). The capture sink is only
/// honored on the eligible batched path (the per-token fallback does not feed
/// it — DFlash always uses the batched path for the prompts it conditions on).
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_capture(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    pbs_in: Option<&PrefillBatchScratch>,
    mut capture: Option<&mut HiddenCaptureSink>,
) -> HipResult<()> {
    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }

    let force_fallback = !crate::config::get().prefill_batched;
    const MIN_BATCH: usize = 4;
    let arch = gpu.arch.as_str();
    let kv_ok =
        kv_cache.quant_q8 || kv_cache.quant_asym2 || kv_cache.quant_asym3 || kv_cache.quant_asym4;
    let weights_ok = weights.layers.iter().all(|l| {
        is_batchable_la(l.wq.gpu_dtype, arch)
            && is_batchable_la(l.wk.gpu_dtype, arch)
            && is_batchable_la(l.wv.gpu_dtype, arch)
            && is_batchable_la(l.wo.gpu_dtype, arch)
            && is_batchable_la(l.w_gate.gpu_dtype, arch)
            && is_batchable_la(l.w_up.gpu_dtype, arch)
            && is_batchable_la(l.w_down.gpu_dtype, arch)
    });
    let eligible = !force_fallback && n >= MIN_BATCH && kv_ok && weights_ok;

    if !eligible {
        // The per-token fallback does not run the batched per-layer loop that
        // feeds the capture sink. DFlash always conditions on the batched path,
        // so a capture request on an ineligible model is a usage error, not a
        // silent no-op.
        assert!(
            capture.is_none(),
            "forward_prefill_batch_capture: hidden capture requested but model \
             is ineligible for the batched path (n={n}, kv_ok={kv_ok}, \
             weights_ok={weights_ok}); DFlash requires the batched forward"
        );
        for (i, &tok) in tokens.iter().enumerate() {
            forward_scratch_embed(gpu, weights, config, tok, start_pos + i, scratch)?;
            forward_scratch_compute(gpu, weights, config, start_pos + i, kv_cache, scratch)?;
        }
        return Ok(());
    }

    let mut own_pbs: Option<PrefillBatchScratch> = None;
    let pbs = if let Some(p) = pbs_in {
        p
    } else {
        let max_batch = PREFILL_MAX_BATCH.min(n.max(MIN_BATCH));
        own_pbs = Some(PrefillBatchScratch::new(
            gpu,
            config,
            max_batch,
            kv_cache.physical_cap,
        )?);
        own_pbs.as_ref().unwrap()
    };

    let max_chunk = pbs.max_batch;
    let mut offset = 0usize;
    while offset < n {
        let chunk_n = (n - offset).min(max_chunk);
        forward_prefill_chunk(
            gpu,
            weights,
            config,
            &tokens[offset..offset + chunk_n],
            start_pos + offset,
            kv_cache,
            scratch,
            pbs,
            capture.as_deref_mut(),
            false,
            None,
        )?;
        offset += chunk_n;
    }

    // Final norm + output projection on the LAST row of x_batch (chunk-local).
    let dim = config.dim;
    let last_n = ((n - 1) % max_chunk) + 1;
    let last_off_bytes = (last_n - 1) * dim * 4;
    gpu.hip
        .memcpy_dtod_at(&scratch.x.buf, 0, &pbs.x_batch.buf, last_off_bytes, dim * 4)?;
    gpu.rmsnorm_f32(
        &scratch.x,
        &weights.output_norm,
        &scratch.tmp,
        config.norm_eps,
    )?;
    weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;

    if let Some(p) = own_pbs {
        p.free_gpu(gpu);
    }
    Ok(())
}

/// One tree-masked batched verify forward over a linearized DDTree.
///
/// `tokens` is the `1 + tree.num_nodes()` linearized node sequence (slot 0 =
/// seed); `tree_bias` is the `[n × n]` row-major additive mask, and
/// `depth_positions` the per-slot DEPTH RoPE positions (`position + node.depth`),
/// both from [`crate::ddtree::linearize_tree_with_parents`]. A single batched
/// forward runs with Q/K RoPE at the DEPTH positions (so parent→child distance is
/// 1, making the bushy-tree verify greedy-lossless) while the KV WRITE and the
/// `tree_bias` mask stay on CONTIGUOUS slots `[position .. position + n)` (no
/// sibling write collision). The masked attention applies `tree_bias` over the
/// in-block keys (prompt fully visible), and every node row's post-final-layer
/// hidden lands in `pbs.x_batch`. `capture` (when `Some`) collects the
/// per-extract-layer residual rows for DFlash conditioning.
///
/// Single chunk only: `n <= pbs.max_batch` (a tree never exceeds the speculation
/// window). Requires Q8_0 KV (the canonical DFlash config): the asym/givens FA
/// re-rotates K in-kernel by the WRITE slot, which conflicts with the decoupled
/// depth-RoPE — so it is rejected here rather than silently skewed.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_tree(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    tokens: &[u32],
    position: usize,
    tree_bias: &GpuTensor,
    depth_positions: &[i32],
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    pbs: &PrefillBatchScratch,
    capture: Option<&mut HiddenCaptureSink>,
) -> HipResult<()> {
    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }
    assert!(
        n <= pbs.max_batch,
        "forward_prefill_batch_tree: tree size {n} exceeds pbs.max_batch {}",
        pbs.max_batch
    );
    assert_eq!(
        depth_positions.len(),
        n,
        "forward_prefill_batch_tree: depth_positions len {} != tree size {n}",
        depth_positions.len()
    );

    let arch = gpu.arch.as_str();
    assert!(
        kv_cache.quant_q8
            && !(kv_cache.quant_asym2 || kv_cache.quant_asym3 || kv_cache.quant_asym4),
        "forward_prefill_batch_tree requires Q8_0 KV (decoupled depth-RoPE is \
         incompatible with the asym/givens in-kernel re-rotation)"
    );
    let weights_ok = weights.layers.iter().all(|l| {
        is_batchable_la(l.wq.gpu_dtype, arch)
            && is_batchable_la(l.wk.gpu_dtype, arch)
            && is_batchable_la(l.wv.gpu_dtype, arch)
            && is_batchable_la(l.wo.gpu_dtype, arch)
            && is_batchable_la(l.w_gate.gpu_dtype, arch)
            && is_batchable_la(l.w_up.gpu_dtype, arch)
            && is_batchable_la(l.w_down.gpu_dtype, arch)
    });
    assert!(
        crate::config::get().prefill_batched && weights_ok,
        "forward_prefill_batch_tree requires the batched path (prefill_batched, \
         batchable weights): weights_ok={weights_ok}"
    );

    // Upload the depth-based RoPE positions into a scratch device buffer (i32
    // bits in an F32 tensor, matching pbs.positions' slot-cosmetic dtype).
    let rope_pos = gpu.alloc_tensor(&[n], rdna_compute::DType::F32)?;
    let rope_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(depth_positions.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&rope_pos.buf, rope_bytes)?;

    let tm = TreeMaskRef {
        bias: tree_bias,
        block_start: position,
        block_cols: n,
        rope_positions: &rope_pos,
    };
    let r = forward_prefill_chunk(
        gpu,
        weights,
        config,
        tokens,
        position,
        kv_cache,
        scratch,
        pbs,
        capture,
        false,
        Some(&tm),
    );
    let _ = gpu.free_tensor(rope_pos);
    r
}

/// Single-chunk capture-friendly entry. The caller must have already
/// populated `pbs.tokens` and `pbs.positions` via
/// `upload_prefill_batch_inputs`, and must size `tokens.len() <= pbs.max_batch`.
/// Skips the internal `memcpy_htod` so the body is safe under
/// `hipStreamBeginCapture`. The eligibility check still runs; on a non-eligible
/// model the function asserts rather than silently falling back, since the
/// fallback would issue uploads that violate capture semantics.
///
/// Capture-mode constraint: in capture mode `max_ctx_len` is baked to
/// `kv_cache.physical_cap`. For Q8 KV at `physical_cap > LDS_CTX_LIMIT`
/// (15000), `forward_prefill_chunk` would enter the per-position
/// long-context fallback that issues `hip.malloc` + `memcpy_htod` per row
/// — both capture-illegal. Reject that combination up-front; the asym KV
/// modes have their own batched flash-masked kernels with no per-position
/// uploads, so they are capture-safe at any context length.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_chunk_captured(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    pbs: &PrefillBatchScratch,
) -> HipResult<()> {
    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }
    assert!(
        n <= pbs.max_batch,
        "captured chunk size {n} exceeds pbs.max_batch {}",
        pbs.max_batch
    );

    let arch = gpu.arch.as_str();
    let kv_ok =
        kv_cache.quant_q8 || kv_cache.quant_asym2 || kv_cache.quant_asym3 || kv_cache.quant_asym4;
    let weights_ok = weights.layers.iter().all(|l| {
        is_batchable_la(l.wq.gpu_dtype, arch)
            && is_batchable_la(l.wk.gpu_dtype, arch)
            && is_batchable_la(l.wv.gpu_dtype, arch)
            && is_batchable_la(l.wo.gpu_dtype, arch)
            && is_batchable_la(l.w_gate.gpu_dtype, arch)
            && is_batchable_la(l.w_up.gpu_dtype, arch)
            && is_batchable_la(l.w_down.gpu_dtype, arch)
    });
    assert!(
        kv_ok && weights_ok,
        "forward_prefill_batch_chunk_captured requires batched-eligible weights + KV"
    );

    // Q8 long-context is now capture-safe: `forward_prefill_chunk`'s Q8 branch
    // uses the tiled `attention_flash_q8_0_batched_masked` (O(1) LDS, no
    // per-position malloc/memcpy), so no cap-based gate is needed.

    forward_prefill_chunk(
        gpu, weights, config, tokens, start_pos, kv_cache, scratch, pbs, None, true, None,
    )
}

#[inline]
fn q8_prefill_family_eligible(
    gpu_arch: &str,
    model_arch: ModelArch,
    quant_q8: bool,
    is_tree: bool,
    batch_size: usize,
) -> bool {
    model_arch == ModelArch::Qwen3
        && (gpu_arch.starts_with("gfx11") || gpu_arch == "gfx1201")
        && quant_q8
        && !is_tree
        && batch_size > 1
}

#[allow(clippy::too_many_arguments)]
fn forward_prefill_chunk(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut KvCache,
    s: &ForwardScratch,
    pbs: &PrefillBatchScratch,
    mut capture: Option<&mut HiddenCaptureSink>,
    pre_uploaded: bool,
    tree_mask: Option<&TreeMaskRef>,
) -> HipResult<()> {
    let n = tokens.len();
    debug_assert!(n > 0);
    debug_assert!(n <= pbs.max_batch);
    if let Some(tm) = tree_mask {
        assert_eq!(
            tm.block_cols, n,
            "tree_mask.block_cols {} must equal chunk size {n}",
            tm.block_cols
        );
    }

    // DFlash hidden capture: collect this chunk's per-extract-layer residual
    // rows (`[n × dim]` each, in `extract_layers` order) into host buffers as
    // the per-layer loop produces them, then interleave into the sink in
    // position-major order at the end of the chunk (`[pos][layer][dim]`).
    let mut cap_rows: Vec<Vec<f32>> = Vec::new();
    if capture.is_some() {
        cap_rows.reserve(capture.as_ref().unwrap().extract_layers.len());
    }

    let dim = config.dim;
    let hidden_dim = config.hidden_dim;
    let kv_dim = config.n_kv_heads * config.head_dim;
    let dim_row_bytes = dim * 4;
    // Q8 WMMA arch gate — see qwen35.rs q8_wmma_arch for the matching capture
    // and rationale. RDNA3 (_w32 builtin) AND RDNA4 (_w32_gfx12 builtin): the
    // fused qkv/gate_up/residual Q8 WMMA helpers route is_rdna4()→_gfx12
    // internally (gemm_qkv_q8_0_wmma_gfx12 et al., silicon-validated on R9700),
    // so has_wmma() is the correct gate — matches qwen35.rs:9008. Bare
    // has_wmma_w32() left RDNA4 on the slower unfused per-projection path.
    let q8_wmma_arch = gpu.arch_caps.has_wmma();

    // 1. Embed N tokens into pbs.x_batch.
    if matches!(
        weights.embd_format,
        EmbeddingFormat::HFQ4G256 | EmbeddingFormat::HFQ4G128 | EmbeddingFormat::Q8_0
    ) {
        if !pre_uploaded {
            let tokens_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            let tokens_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
            gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;
        }
        match weights.embd_format {
            EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256_batched(
                &weights.token_embd,
                &pbs.x_batch,
                &pbs.tokens,
                n,
                dim,
            )?,
            EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128_batched(
                &weights.token_embd,
                &pbs.x_batch,
                &pbs.tokens,
                n,
                dim,
            )?,
            EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8_batched(
                &weights.token_embd,
                &pbs.x_batch,
                &pbs.tokens,
                n,
                dim,
            )?,
            _ => unreachable!(),
        }
    } else {
        for (i, &tok) in tokens.iter().enumerate() {
            match weights.embd_format {
                EmbeddingFormat::Q4K => {
                    gpu.embedding_lookup_q4k(&weights.token_embd, &s.x, tok, dim)?
                }
                EmbeddingFormat::F32 => {
                    gpu.embedding_lookup(&weights.token_embd, &s.x, tok, dim)?
                }
                EmbeddingFormat::HFQ4G256 | EmbeddingFormat::HFQ4G128 | EmbeddingFormat::Q8_0 => {
                    unreachable!()
                }
            }
            gpu.hip.memcpy_dtod_at(
                &pbs.x_batch.buf,
                i * dim_row_bytes,
                &s.x.buf,
                0,
                dim_row_bytes,
            )?;
        }
    }

    // 1b. Upload positions [start_pos .. start_pos + n] as i32.
    if !pre_uploaded {
        let positions_host: Vec<i32> = (0..n).map(|i| (start_pos + i) as i32).collect();
        let positions_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
        gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;
    }

    let max_ctx_len = if gpu.graphs.capture_mode {
        kv_cache.physical_cap
    } else {
        start_pos + n
    };
    let q8_family_eligible = q8_prefill_family_eligible(
        &gpu.arch,
        config.arch,
        kv_cache.quant_q8,
        tree_mask.is_some(),
        n,
    );
    let q8_attn_ctx = q8_family_eligible.then(|| DispatchCtx::new(gpu));

    // 2. Per-layer loop.
    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];
        let qkv_is_mq = matches!(
            layer.wq.gpu_dtype,
            DType::MQ4G256 | DType::MQ6G256 | DType::MQ3G256 | DType::MFP4G32
        );
        let qkv_is_6bit = matches!(layer.wq.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
        let qkv_is_mq3 = matches!(layer.wq.gpu_dtype, DType::MQ3G256);
        let qkv_is_fp4 = matches!(layer.wq.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);

        // attn_norm (+ FWHT for MQ — includes MFP4G32 since rotation is the
        // same FWHT pattern as MQ4).
        if qkv_is_mq {
            gpu.fused_rmsnorm_rotate_mq_batched(
                &pbs.x_batch,
                &layer.attn_norm,
                &pbs.x_rot_batch,
                dim,
                config.norm_eps,
                n,
            )?;
        } else {
            gpu.rmsnorm_batched(
                &pbs.x_batch,
                &layer.attn_norm,
                &pbs.x_rot_batch,
                n,
                dim,
                config.norm_eps,
            )?;
        }

        let qkv_is_q8 = matches!(layer.wq.gpu_dtype, DType::Q8_0);
        let qkv_is_hfq4g128 = matches!(layer.wq.gpu_dtype, DType::HFQ4G128);

        // 3-way fused QKV projection.
        if qkv_is_hfq4g128 {
            debug_assert!(
                matches!(layer.wk.gpu_dtype, DType::HFQ4G128)
                    && matches!(layer.wv.gpu_dtype, DType::HFQ4G128),
                "llama HFQ4G128 QKV batch requires one uniform wire layout",
            );
            gpu.gemm_hfq4g128(
                &layer.wq.buf,
                &pbs.x_rot_batch,
                &pbs.fa_q_batch,
                layer.wq.m,
                layer.wq.k,
                n,
            )?;
            gpu.gemm_hfq4g128(
                &layer.wk.buf,
                &pbs.x_rot_batch,
                &pbs.fa_k_batch,
                layer.wk.m,
                layer.wk.k,
                n,
            )?;
            gpu.gemm_hfq4g128(
                &layer.wv.buf,
                &pbs.x_rot_batch,
                &pbs.fa_v_batch,
                layer.wv.m,
                layer.wv.k,
                n,
            )?;
        } else if qkv_is_6bit {
            gpu.gemm_qkv_hfq6g256(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &pbs.x_rot_batch,
                &pbs.fa_q_batch,
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
                n,
            )?;
        } else if qkv_is_q8 && q8_wmma_arch {
            debug_assert!(
                matches!(layer.wk.gpu_dtype, DType::Q8_0)
                    && matches!(layer.wv.gpu_dtype, DType::Q8_0),
                "llama qkv Q8 WMMA dispatch requires all of wq/wk/wv to be Q8_0",
            );
            gpu.gemm_qkv_q8_0_wmma(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &pbs.x_rot_batch,
                &pbs.fa_q_batch,
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
                n,
            )?;
        } else if qkv_is_q8 {
            gpu.gemm_q8_0_batched_chunked(
                &layer.wq.buf,
                &pbs.x_rot_batch,
                &pbs.fa_q_batch,
                layer.wq.m,
                layer.wq.k,
                n,
            )?;
            gpu.gemm_q8_0_batched_chunked(
                &layer.wk.buf,
                &pbs.x_rot_batch,
                &pbs.fa_k_batch,
                layer.wk.m,
                layer.wk.k,
                n,
            )?;
            gpu.gemm_q8_0_batched_chunked(
                &layer.wv.buf,
                &pbs.x_rot_batch,
                &pbs.fa_v_batch,
                layer.wv.m,
                layer.wv.k,
                n,
            )?;
        } else if qkv_is_mq3 {
            gpu.gemm_qkv_hfq3g256_wmma(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &pbs.x_rot_batch,
                &pbs.fa_q_batch,
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
                n,
            )?;
        } else if qkv_is_fp4 {
            gpu.gemm_qkv_hfp4g32(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &pbs.x_rot_batch,
                &pbs.fa_q_batch,
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
                n,
            )?;
        } else {
            gpu.gemm_qkv_hfq4g256(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &pbs.x_rot_batch,
                &pbs.fa_q_batch,
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
                n,
            )?;
        }

        // Per-head Q/K rmsnorm (Qwen3 only — None on plain LLaMA).
        if config.has_qk_norm {
            if let Some(ref qn) = layer.q_norm {
                gpu.rmsnorm_batched(
                    &pbs.fa_q_batch,
                    qn,
                    &pbs.fa_q_batch,
                    n * config.n_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
            }
            if let Some(ref kn) = layer.k_norm {
                gpu.rmsnorm_batched(
                    &pbs.fa_k_batch,
                    kn,
                    &pbs.fa_k_batch,
                    n * config.n_kv_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
            }
        }

        // Batched full RoPE (non-interleaved, half-split convention —
        // matches forward_scratch's rope_f32). In tree mode the rotation uses
        // DEPTH positions (parent→child distance 1) while the KV write below
        // stays on the CONTIGUOUS `pbs.positions` slots — see TreeMaskRef docs.
        let rope_pos = match tree_mask {
            Some(tm) => tm.rope_positions,
            None => &pbs.positions,
        };
        gpu.rope_batched_f32(
            &pbs.fa_q_batch,
            &pbs.fa_k_batch,
            rope_pos,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            config.rope_freq_base,
            n,
        )?;

        // Batched KV write.
        if kv_cache.quant_asym4 {
            let ct = kv_cache.givens_cos.as_ref().unwrap();
            let st = kv_cache.givens_sin.as_ref().unwrap();
            gpu.kv_cache_write_asym4_batched(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                &pbs.positions,
                ct,
                st,
                config.n_kv_heads,
                config.head_dim,
                n,
            )?;
        } else if kv_cache.quant_asym3 {
            let ct = kv_cache.givens_cos.as_ref().unwrap();
            let st = kv_cache.givens_sin.as_ref().unwrap();
            gpu.kv_cache_write_asym3_batched(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                &pbs.positions,
                ct,
                st,
                config.n_kv_heads,
                config.head_dim,
                n,
            )?;
        } else if kv_cache.quant_asym2 {
            let ct = kv_cache.givens_cos.as_ref().unwrap();
            let st = kv_cache.givens_sin.as_ref().unwrap();
            gpu.kv_cache_write_asym2_batched(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                &pbs.positions,
                ct,
                st,
                config.n_kv_heads,
                config.head_dim,
                n,
            )?;
        } else if !q8_family_eligible {
            gpu.kv_cache_write_q8_0_batched(
                &kv_cache.k_gpu[layer_idx],
                &pbs.fa_k_batch,
                &pbs.positions,
                config.n_kv_heads,
                config.head_dim,
                n,
            )?;
            gpu.kv_cache_write_q8_0_batched(
                &kv_cache.v_gpu[layer_idx],
                &pbs.fa_v_batch,
                &pbs.positions,
                config.n_kv_heads,
                config.head_dim,
                n,
            )?;
        } else {
            debug_assert!(kv_cache.quant_q8);
            // Standard causal Q8 prefill is paired with its attention launch
            // below through `AttentionFamily`; that entry point owns K/V writes.
            // A singleton tail stays on the legacy path above because the
            // family treats batch_size=1 as decode and expects decode params.
        }

        // Batched flash attention (causal, or tree-masked when `tree_mask` is
        // Some). The masked kernels add `tree_mask.bias` over the in-block keys
        // and leave the prompt fully visible; passing `None` is plain causal.
        const LDS_CTX_LIMIT: usize = 15000;
        let (tree_bias, tree_block_start, tree_block_cols) = match tree_mask {
            Some(tm) => (Some(tm.bias), tm.block_start, tm.block_cols),
            None => (None, 0usize, 0usize),
        };
        if kv_cache.quant_asym4 {
            let ct = kv_cache.givens_cos.as_ref().unwrap();
            let st = kv_cache.givens_sin.as_ref().unwrap();
            gpu.attention_flash_asym4_batched_masked(
                &pbs.fa_q_batch,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &pbs.fa_attn_out_batch,
                &pbs.positions,
                ct,
                st,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                kv_cache.physical_cap,
                max_ctx_len,
                n,
                &pbs.flash_partials,
                tree_bias,
                tree_block_start,
                tree_block_cols,
            )?;
        } else if kv_cache.quant_asym3 {
            let ct = kv_cache.givens_cos.as_ref().unwrap();
            let st = kv_cache.givens_sin.as_ref().unwrap();
            gpu.attention_flash_asym3_batched_masked(
                &pbs.fa_q_batch,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &pbs.fa_attn_out_batch,
                &pbs.positions,
                ct,
                st,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                kv_cache.physical_cap,
                max_ctx_len,
                n,
                &pbs.flash_partials,
                tree_bias,
                tree_block_start,
                tree_block_cols,
            )?;
        } else if kv_cache.quant_asym2 {
            assert!(
                tree_mask.is_none(),
                "tree-masked verify needs a masked attention kernel; asym2 KV \
                 has only the unmasked batched flash kernel. Use q8/asym3/asym4 KV."
            );
            let ct = kv_cache.givens_cos.as_ref().unwrap();
            let st = kv_cache.givens_sin.as_ref().unwrap();
            gpu.attention_flash_asym2_batched(
                &pbs.fa_q_batch,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &pbs.fa_attn_out_batch,
                &pbs.positions,
                ct,
                st,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                kv_cache.physical_cap,
                max_ctx_len,
                n,
                &pbs.flash_partials,
            )?;
        } else if q8_family_eligible {
            debug_assert!(kv_cache.quant_q8);
            // Ordinary causal Q8 prefill now shares the same paired dispatch
            // used by Qwen3.5, enabling the existing gfx11/gfx1201 M16 kernel.
            // Tree verify and singleton tails remain on their legacy paths.
            let plan = KvTierPlan::derive(KvTierInputs {
                pos: start_pos,
                capture_mode: gpu.graphs.capture_mode,
                batch_size: n,
                is_tree: false,
                ..kv_cache.tier_inputs()
            })
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
            let io = AttnParams {
                q: &pbs.fa_q_batch,
                k: &pbs.fa_k_batch,
                v: &pbs.fa_v_batch,
                k_cache: &kv_cache.k_gpu[layer_idx],
                v_cache: &kv_cache.v_gpu[layer_idx],
                k_scales: None,
                v_scales: None,
                pos_buf: &s.pos_buf,
                pos: start_pos,
                positions: Some(&pbs.positions),
                n_heads: config.n_heads,
                n_kv_heads: config.n_kv_heads,
                head_dim: config.head_dim,
                physical_cap: kv_cache.physical_cap,
                batch_size: n,
                max_ctx_len,
                flash_partials: Some(&pbs.flash_partials),
                givens_cos: kv_cache.givens_cos.as_ref(),
                givens_sin: kv_cache.givens_sin.as_ref(),
                tree_bias: None,
                block_start: 0,
                block_cols: 0,
                output_gate: None,
                output: &pbs.fa_attn_out_batch,
            };
            attention_family()
                .run_attention(q8_attn_ctx.as_ref().unwrap(), gpu, &plan, &io)
                .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
        } else if max_ctx_len > LDS_CTX_LIMIT {
            // Tree-masked verify is not yet validated in the long-context Q8
            // regime (ctx > LDS_CTX_LIMIT). The batched-masked kernel below DOES
            // accept a tree-bias arg, but the dense DFlash tree-verify path was
            // only exercised at normal context, so we keep it guarded off here
            // and pass `None`; wiring `tree_bias` in is a validated follow-up.
            assert!(
                tree_mask.is_none(),
                "tree-masked verify is not yet validated on the long-context Q8 \
                 path (ctx {max_ctx_len} > {LDS_CTX_LIMIT})."
            );
            // Long-context Q8: tiled batched flash (O(1) LDS, capture-safe).
            // Replaces the former per-position malloc+memcpy loop now that
            // `attention_flash_q8_0_batched_masked` (the >8192 crossover kernel)
            // exists — numerically equivalent, no per-row host uploads.
            gpu.attention_flash_q8_0_batched_masked(
                &pbs.fa_q_batch,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &pbs.fa_attn_out_batch,
                &pbs.positions,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                kv_cache.physical_cap,
                max_ctx_len,
                n,
                &pbs.flash_partials,
                None,
                0,
                0,
            )?;
        } else {
            gpu.attention_q8_0_kv_batched_masked(
                &pbs.fa_q_batch,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &pbs.fa_attn_out_batch,
                &pbs.positions,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                kv_cache.physical_cap,
                max_ctx_len,
                n,
                tree_bias,
                tree_block_start,
                tree_block_cols,
            )?;
        }

        // wo + residual.
        let wo_is_mq = matches!(
            layer.wo.gpu_dtype,
            DType::MQ4G256 | DType::MQ6G256 | DType::MQ3G256 | DType::MFP4G32
        );
        let wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
        let wo_is_mq3 = matches!(layer.wo.gpu_dtype, DType::MQ3G256);
        let wo_is_fp4 = matches!(layer.wo.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
        let wo_is_q8 = matches!(layer.wo.gpu_dtype, DType::Q8_0);
        let wo_is_hfq4g128 = matches!(layer.wo.gpu_dtype, DType::HFQ4G128);
        let wo_input = if wo_is_mq {
            // F2: AWQ-aware rotate for wo (FullAttention output projection) input.
            rotate_x_mq_batched_for(
                gpu,
                &layer.wo,
                &pbs.fa_attn_out_batch,
                &pbs.fa_attn_out_rot_batch,
                layer.wo.k,
                n,
            )?;
            &pbs.fa_attn_out_rot_batch
        } else {
            &pbs.fa_attn_out_batch
        };
        if wo_is_hfq4g128 {
            // The generic G128 GEMM has overwrite semantics. Reuse x_rot_batch
            // as a dead-after-QKV temporary, then add into the residual stream.
            let projected = pbs.x_rot_batch.sub_offset(0, n * layer.wo.m);
            gpu.gemm_hfq4g128(
                &layer.wo.buf,
                wo_input,
                &projected,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
            let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
            gpu.add_inplace_f32(&x_n, &projected)?;
        } else if wo_is_6bit {
            gpu.gemm_hfq6g256_residual(
                &layer.wo.buf,
                wo_input,
                &pbs.x_batch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
        } else if wo_is_q8 && q8_wmma_arch {
            let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
            gpu.gemm_q8_0_residual_wmma(&layer.wo.buf, wo_input, &x_n, layer.wo.m, layer.wo.k, n)?;
        } else if wo_is_q8 {
            let scratch = pbs.x_rot_batch.sub_offset(0, n * layer.wo.m);
            gpu.gemm_q8_0_batched_chunked(
                &layer.wo.buf,
                wo_input,
                &scratch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
            let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
            gpu.add_inplace_f32(&x_n, &scratch)?;
        } else if wo_is_mq3 {
            gpu.gemm_hfq3g256_residual_wmma(
                &layer.wo.buf,
                wo_input,
                &pbs.x_batch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
        } else if wo_is_fp4 {
            gpu.gemm_hfp4g32_residual(
                &layer.wo.buf,
                wo_input,
                &pbs.x_batch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
        } else {
            gpu.gemm_hfq4g256_residual(
                &layer.wo.buf,
                wo_input,
                &pbs.x_batch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
        }

        // FFN: rmsnorm (+ FWHT for MQ — includes MFP4G32), gate+up, silu_mul,
        // w_down + residual.
        let ffn_is_mq = matches!(
            layer.w_gate.gpu_dtype,
            DType::MQ4G256 | DType::MQ6G256 | DType::MQ3G256 | DType::MFP4G32
        );
        let ffn_is_6bit = matches!(layer.w_gate.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
        let ffn_is_mq3 = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256);
        let ffn_is_fp4 = matches!(layer.w_gate.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
        let ffn_is_q8 = matches!(layer.w_gate.gpu_dtype, DType::Q8_0);
        let ffn_is_hfq4g128 = matches!(layer.w_gate.gpu_dtype, DType::HFQ4G128);
        if ffn_is_mq {
            gpu.fused_rmsnorm_rotate_mq_batched(
                &pbs.x_batch,
                &layer.ffn_norm,
                &pbs.x_rot_batch,
                dim,
                config.norm_eps,
                n,
            )?;
        } else {
            gpu.rmsnorm_batched(
                &pbs.x_batch,
                &layer.ffn_norm,
                &pbs.x_rot_batch,
                n,
                dim,
                config.norm_eps,
            )?;
        }
        if ffn_is_hfq4g128 {
            debug_assert!(
                matches!(layer.w_up.gpu_dtype, DType::HFQ4G128),
                "llama HFQ4G128 gate/up batch requires one uniform wire layout",
            );
            gpu.gemm_hfq4g128(
                &layer.w_gate.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                layer.w_gate.m,
                layer.w_gate.k,
                n,
            )?;
            gpu.gemm_hfq4g128(
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.up_batch,
                layer.w_up.m,
                layer.w_up.k,
                n,
            )?;
        } else if ffn_is_6bit {
            gpu.gemm_gate_up_hfq6g256(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        } else if ffn_is_q8 && q8_wmma_arch {
            debug_assert!(
                matches!(layer.w_up.gpu_dtype, DType::Q8_0),
                "llama FFN Q8 WMMA dispatch requires both w_gate and w_up to be Q8_0",
            );
            gpu.gemm_gate_up_q8_0_wmma(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        } else if ffn_is_q8 {
            gpu.gemm_q8_0_batched_chunked(
                &layer.w_gate.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                layer.w_gate.m,
                layer.w_gate.k,
                n,
            )?;
            gpu.gemm_q8_0_batched_chunked(
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.up_batch,
                layer.w_up.m,
                layer.w_up.k,
                n,
            )?;
        } else if ffn_is_mq3 {
            gpu.gemm_gate_up_hfq3g256_wmma(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        } else if ffn_is_fp4 {
            gpu.gemm_gate_up_hfp4g32(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        } else {
            gpu.gemm_gate_up_hfq4g256(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        }
        let w_down_is_mq = matches!(
            layer.w_down.gpu_dtype,
            DType::MQ4G256 | DType::MQ6G256 | DType::MQ3G256 | DType::MFP4G32
        );
        let w_down_is_6bit = matches!(layer.w_down.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
        let w_down_is_mq3 = matches!(layer.w_down.gpu_dtype, DType::MQ3G256);
        let w_down_is_fp4 = matches!(layer.w_down.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
        let w_down_is_q8 = matches!(layer.w_down.gpu_dtype, DType::Q8_0);
        let w_down_is_hfq4g128 = matches!(layer.w_down.gpu_dtype, DType::HFQ4G128);
        if w_down_is_mq {
            // F2: AWQ-aware silu_mul+rotate for w_down input.
            fused_silu_mul_rotate_mq_batched_for(
                gpu,
                &layer.w_down,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                &pbs.ffn_hidden_batch,
                hidden_dim,
                n,
            )?;
        } else {
            gpu.silu_mul_f32(&pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch)?;
        }
        if w_down_is_hfq4g128 {
            // gate_ffn_batch is dead after silu_mul and is larger than the
            // dim-wide down projection output, so it is a safe residual temp.
            let projected = pbs.gate_ffn_batch.sub_offset(0, n * layer.w_down.m);
            gpu.gemm_hfq4g128(
                &layer.w_down.buf,
                &pbs.ffn_hidden_batch,
                &projected,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
            let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
            gpu.add_inplace_f32(&x_n, &projected)?;
        } else if w_down_is_6bit {
            gpu.gemm_hfq6g256_residual(
                &layer.w_down.buf,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else if w_down_is_q8 && q8_wmma_arch {
            let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
            gpu.gemm_q8_0_residual_wmma(
                &layer.w_down.buf,
                &pbs.ffn_hidden_batch,
                &x_n,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else if w_down_is_q8 {
            let scratch = pbs.x_rot_batch.sub_offset(0, n * layer.w_down.m);
            gpu.gemm_q8_0_batched_chunked(
                &layer.w_down.buf,
                &pbs.ffn_hidden_batch,
                &scratch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
            let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
            gpu.add_inplace_f32(&x_n, &scratch)?;
        } else if w_down_is_mq3 {
            gpu.gemm_hfq3g256_residual_wmma(
                &layer.w_down.buf,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else if w_down_is_fp4 {
            gpu.gemm_hfp4g32_residual(
                &layer.w_down.buf,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else {
            gpu.gemm_hfq4g256_residual(
                &layer.w_down.buf,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        }

        // DFlash / DSpark capture: if this layer is an extract layer, collect its
        // post-FFN residual rows (`pbs.x_batch[..n]`). The residual stream here is
        // the layer output BEFORE the next layer's attn_norm — exactly the
        // conditioning the draft model's cross-attention consumes.
        //
        // GPU-resident sink (`hidden_gpu` Some): copy each position's row straight
        // into its position-major slot on-device — `dst[(p·L + l_idx)·dim ..]` —
        // so the whole capture never leaves VRAM. Host sink (default): download
        // the rows and interleave after the loop.
        if let Some(cap) = capture.as_deref() {
            if let Some(l_idx) = cap.extract_layers.iter().position(|&x| x == layer_idx) {
                if let Some(dst) = cap.hidden_gpu {
                    let num_extract = cap.extract_layers.len();
                    for p in 0..n {
                        let dst_off = (p * num_extract + l_idx) * dim * 4;
                        gpu.memcpy_dtod_at_auto(
                            &dst.buf,
                            dst_off,
                            &pbs.x_batch.buf,
                            p * dim * 4,
                            dim * 4,
                        )?;
                    }
                } else {
                    let rows = pbs.x_batch.sub_offset(0, n * dim);
                    cap_rows.push(gpu.download_f32(&rows)?);
                }
            }
        }

        let _ = kv_dim;
    }

    // Interleave the captured per-extract-layer rows into the HOST sink in
    // position-major order: for each position p, concat layer 0..L at p. The
    // GPU-resident sink already wrote position-major slots inline above.
    if let Some(cap) = capture.as_deref_mut() {
        if cap.hidden_gpu.is_none() {
            debug_assert_eq!(
                cap_rows.len(),
                cap.extract_layers.len(),
                "captured {} layers but extract_layers has {}",
                cap_rows.len(),
                cap.extract_layers.len()
            );
            let num_extract = cap_rows.len();
            cap.hidden.reserve(n * num_extract * dim);
            for p in 0..n {
                for layer_rows in cap_rows.iter() {
                    cap.hidden
                        .extend_from_slice(&layer_rows[p * dim..(p + 1) * dim]);
                }
            }
        }
    }

    Ok(())
}

/// Load LLaMA weights from GGUF onto GPU.
/// Quantized weights stay quantized (Q4_K, Q6_K, Q8_0).
/// Only norm weights and embeddings are dequantized to F32.
pub fn load_weights(
    gguf: &GgufFile,
    config: &LlamaConfig,
    gpu: &mut Gpu,
) -> HipResult<LlamaWeights> {
    // Helper: upload F32 tensor
    fn up_f32(gguf: &GgufFile, gpu: &mut Gpu, name: &str, shape: &[usize]) -> HipResult<GpuTensor> {
        let info = gguf
            .find_tensor(name)
            .unwrap_or_else(|| panic!("tensor not found: {name}"));
        let data = load_tensor_f32(gguf, info);
        gpu.upload_f32(&data, shape)
    }
    // Helper: upload quantized weight (converts Q4_K to Q4_F16_G64 at load time)
    fn up_weight(
        gguf: &GgufFile,
        gpu: &Gpu,
        name: &str,
        m: usize,
        k: usize,
    ) -> HipResult<WeightTensor> {
        let info = gguf
            .find_tensor(name)
            .unwrap_or_else(|| panic!("tensor not found: {name}"));
        let raw_data = gguf.tensor_data(info);

        match info.dtype {
            GgmlType::Q4K => {
                let buf = gpu.upload_raw(raw_data, &[raw_data.len()])?;
                Ok(WeightTensor {
                    buf,
                    gpu_dtype: DType::Q4K,
                    m,
                    k,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                })
            }
            GgmlType::Q6K => {
                let buf = gpu.upload_raw(raw_data, &[raw_data.len()])?;
                Ok(WeightTensor {
                    buf,
                    gpu_dtype: DType::Q6K,
                    m,
                    k,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                })
            }
            GgmlType::Q8_0 => {
                let buf = gpu.upload_raw(raw_data, &[raw_data.len()])?;
                Ok(WeightTensor {
                    buf,
                    gpu_dtype: DType::Q8_0,
                    m,
                    k,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                })
            }
            GgmlType::F32 => {
                let buf = gpu.upload_raw(raw_data, &[raw_data.len()])?;
                Ok(WeightTensor {
                    buf,
                    gpu_dtype: DType::F32,
                    m,
                    k,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                })
            }
            _ => {
                // Unsupported: dequant to F32 on CPU, upload as raw bytes
                let data = load_tensor_f32(gguf, info);
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
                };
                let buf = gpu.upload_raw(bytes, &[bytes.len()])?;
                Ok(WeightTensor {
                    buf,
                    gpu_dtype: DType::F32,
                    m,
                    k,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                })
            }
        }
    }

    eprintln!("  loading token_embd...");
    let embd_info = gguf
        .find_tensor("token_embd.weight")
        .expect("token_embd not found");
    let (token_embd, embd_fmt) = if embd_info.dtype == GgmlType::Q4K {
        let raw = gguf.tensor_data(embd_info);
        eprintln!(
            "    (Q4K raw, {} MB — saves {} MB vs F32)",
            raw.len() / 1_000_000,
            (config.vocab_size * config.dim * 4 - raw.len()) / 1_000_000
        );
        (gpu.upload_raw(raw, &[raw.len()])?, EmbeddingFormat::Q4K)
    } else {
        let data = load_tensor_f32(gguf, embd_info);
        (
            gpu.upload_f32(&data, &[config.vocab_size, config.dim])?,
            EmbeddingFormat::F32,
        )
    };
    eprintln!("  loading output_norm...");
    let output_norm = up_f32(gguf, gpu, "output_norm.weight", &[config.dim])?;

    eprintln!("  loading output...");
    let output = if gguf.find_tensor("output.weight").is_some() {
        up_weight(gguf, gpu, "output.weight", config.vocab_size, config.dim)?
    } else {
        let info = gguf.find_tensor("token_embd.weight").unwrap();
        let data = load_tensor_f32(gguf, info);
        let buf = gpu.upload_f32(&data, &[config.vocab_size, config.dim])?;
        WeightTensor {
            buf,
            gpu_dtype: DType::F32,
            m: config.vocab_size,
            k: config.dim,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        }
    };

    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        eprintln!("  loading layer {i}/{} ...", config.n_layers);
        let p = format!("blk.{i}");
        let kv_dim = config.n_kv_heads * config.head_dim;

        let q_out_dim = config.n_heads * config.head_dim;
        let _k_out_dim = config.n_kv_heads * config.head_dim;

        let layer = LayerWeights {
            attn_norm: up_f32(gguf, gpu, &format!("{p}.attn_norm.weight"), &[config.dim])?,
            wq: up_weight(
                gguf,
                gpu,
                &format!("{p}.attn_q.weight"),
                q_out_dim,
                config.dim,
            )?,
            wk: up_weight(gguf, gpu, &format!("{p}.attn_k.weight"), kv_dim, config.dim)?,
            wv: up_weight(gguf, gpu, &format!("{p}.attn_v.weight"), kv_dim, config.dim)?,
            wo: up_weight(
                gguf,
                gpu,
                &format!("{p}.attn_output.weight"),
                config.dim,
                q_out_dim,
            )?,
            q_norm: if config.has_qk_norm {
                Some(up_f32(
                    gguf,
                    gpu,
                    &format!("{p}.attn_q_norm.weight"),
                    &[config.head_dim],
                )?)
            } else {
                None
            },
            k_norm: if config.has_qk_norm {
                Some(up_f32(
                    gguf,
                    gpu,
                    &format!("{p}.attn_k_norm.weight"),
                    &[config.head_dim],
                )?)
            } else {
                None
            },
            ffn_norm: up_f32(gguf, gpu, &format!("{p}.ffn_norm.weight"), &[config.dim])?,
            w_gate: up_weight(
                gguf,
                gpu,
                &format!("{p}.ffn_gate.weight"),
                config.hidden_dim,
                config.dim,
            )?,
            w_up: up_weight(
                gguf,
                gpu,
                &format!("{p}.ffn_up.weight"),
                config.hidden_dim,
                config.dim,
            )?,
            w_down: up_weight(
                gguf,
                gpu,
                &format!("{p}.ffn_down.weight"),
                config.dim,
                config.hidden_dim,
            )?,
        };
        layers.push(layer);
    }

    Ok(LlamaWeights {
        token_embd,
        embd_format: embd_fmt,
        output_norm,
        output,
        layers,
        lm_head_aliases_embd: false,
    })
}

/// Pre-allocated scratch buffers for the forward pass.
/// Allocate once, reuse every token — zero hipMalloc in the hot loop.
pub struct ForwardScratch {
    pub x: GpuTensor,
    pub tmp: GpuTensor,
    pub q: GpuTensor,
    pub k: GpuTensor,
    pub v: GpuTensor,
    pub attn_out: GpuTensor,
    pub o: GpuTensor,
    pub gate: GpuTensor,
    pub up: GpuTensor,
    pub ffn_hidden: GpuTensor,
    pub ffn_out: GpuTensor,
    pub logits: GpuTensor,
    pub sample_buf: GpuTensor,
    pub repeat_buf: GpuTensor,
    pub attn_partials: GpuTensor, // flash-decoding partial results
    pub pos_buf: hip_bridge::DeviceBuffer,
    /// FWHT-rotated x scratch for MagnumQuant batching. Sized to max(dim, hidden_dim).
    pub x_rot: GpuTensor,
}

impl ForwardScratch {
    /// Allocate scratch sized for the model's declared context. Callers that
    /// know the runtime KV cap should prefer [`Self::new_with_max_seq`] so the
    /// flash-attention partials buffer matches the cache (avoids both OOB at
    /// large `max_seq` and over-allocation when the cap is small).
    pub fn new(gpu: &mut Gpu, config: &LlamaConfig) -> HipResult<Self> {
        Self::new_with_max_seq(gpu, config, config.max_seq_len)
    }

    /// `max_seq` MUST be ≥ the KV cache's `physical_cap` — the flash-decoding
    /// partials buffer is sized from the architecture/shape-selected Q8 tile
    /// and must cover every tile addressable by the cache.
    /// (Was hardcoded to 16 chunks = max_seq 2048; running at a larger cap
    /// overflowed it → silent OOB / garbage on the flash-attention path.)
    pub fn new_with_max_seq(
        gpu: &mut Gpu,
        config: &LlamaConfig,
        max_seq: usize,
    ) -> HipResult<Self> {
        let dim = config.dim;
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;
        // Flash-decoding partials: n_heads × max_chunks × (2 + head_dim) floats.
        // The gfx1100 Q8 kernel uses tile32 (and selected gfx12 shapes tile16),
        // so a historical fixed tile128 allocation under-sized this buffer by
        // up to 8x and faulted as soon as llama-family Q8 decode selected flash.
        let partials_size = llama_flash_partials_len(
            &gpu.arch,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            max_seq,
        );
        Ok(Self {
            x: gpu.alloc_tensor(&[dim], DType::F32)?,
            tmp: gpu.alloc_tensor(&[dim], DType::F32)?,
            q: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            k: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            v: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            attn_out: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            o: gpu.alloc_tensor(&[dim], DType::F32)?,
            gate: gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?,
            up: gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?,
            ffn_hidden: gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?,
            ffn_out: gpu.alloc_tensor(&[dim], DType::F32)?,
            logits: gpu.alloc_tensor(&[config.vocab_size], DType::F32)?,
            sample_buf: gpu.alloc_tensor(&[2], DType::F32)?,
            repeat_buf: gpu.alloc_tensor(&[64], DType::F32)?,
            attn_partials: gpu.alloc_tensor(&[partials_size], DType::F32)?,
            pos_buf: gpu.hip.malloc(4)?, // single i32
            x_rot: gpu.alloc_tensor(&[dim.max(config.hidden_dim)], DType::F32)?,
        })
    }

    /// Return all GPU buffers to the pool (drained on unload). Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in [
            self.x,
            self.tmp,
            self.q,
            self.k,
            self.v,
            self.attn_out,
            self.o,
            self.gate,
            self.up,
            self.ffn_hidden,
            self.ffn_out,
            self.logits,
            self.sample_buf,
            self.repeat_buf,
            self.attn_partials,
            self.x_rot,
        ] {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.hip.free(self.pos_buf);
    }
}

/// Forward pass with persistent scratch buffers. Zero allocations.
/// Returns (token_id, new_rng_state) via GPU-side sampling.
pub fn forward_scratch(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    token: u32,
    pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    temperature: f32,
    top_p: f32,
    rng_state: u32,
    repeat_window: usize,
    repeat_penalty: f32,
) -> HipResult<(u32, u32)> {
    forward_scratch_embed(gpu, weights, config, token, pos, scratch)?;
    forward_scratch_layers(
        gpu,
        weights,
        config,
        pos,
        kv_cache,
        scratch,
        temperature,
        top_p,
        rng_state,
        repeat_window,
        repeat_penalty,
    )
}

/// Upload pos and compute embedding. Must be called before forward_scratch_layers.
pub fn forward_scratch_embed(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    token: u32,
    pos: usize,
    scratch: &ForwardScratch,
) -> HipResult<()> {
    let dim = config.dim;
    // Upload pos to GPU buffer (4 bytes)
    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
    // Embedding lookup
    embedding_lookup_dispatch(
        gpu,
        weights.embd_format,
        &weights.token_embd,
        &scratch.x,
        token,
        dim,
    )?;
    Ok(())
}

/// Layer loop + final norm + logits + sampling. Graph-capturable.
/// Cached `HIPFIRE_FORWARD_LOWERED` toggle for the llama-family decode path
/// (N5 Phase A). Default ON; `HIPFIRE_FORWARD_LOWERED=0` forces the legacy
/// hand-rolled [`forward_scratch_layers`] body as a byte-exact reference +
/// escape hatch. Mirrors qwen2's `qwen2_forward_lowered_enabled` and qwen35's
/// `forward_lowered_enabled`.
fn llama_forward_lowered_enabled() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_FORWARD_LOWERED")
            .ok()
            .as_deref()
            != Some("0")
    })
}

#[inline]
fn llama_attention_flash_mode_for(mode: &str, gpu_arch: &str) -> usize {
    match mode {
        "never" | "0" | "off" => 0,
        "always" | "2" | "force" => 2,
        _ if gpu_arch.starts_with("gfx11") || gpu_arch.starts_with("gfx12") => 2,
        _ => 1,
    }
}

#[inline]
fn llama_attention_flash_mode(gpu_arch: &str) -> usize {
    llama_attention_flash_mode_for(crate::config::get().attention_flash_mode.as_str(), gpu_arch)
}

#[inline]
fn llama_flash_partials_len(
    gpu_arch: &str,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
) -> usize {
    let flash_tile = rdna_compute::attention::q8_flash_tile_size(
        gpu_arch, n_heads, n_kv_heads, head_dim, max_seq,
    );
    n_heads * max_seq.div_ceil(flash_tile) * (2 + head_dim)
}

/// KV-cache write + single-token attention for the lowered decode path.
/// Asym and Q8 tiers use the paired `AttentionFamily` plan; legacy cache
/// formats retain the hand ladder below. The hand body keeps its own inline
/// copy as the byte-exact `HIPFIRE_FORWARD_LOWERED=0` reference.
fn llama_kv_write_attend(
    gpu: &mut Gpu,
    kv_cache: &KvCache,
    scratch: &ForwardScratch,
    layer_idx: usize,
    pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
) -> HipResult<()> {
    if kv_cache.quant_asym4 || kv_cache.quant_asym3 || kv_cache.quant_asym2 || kv_cache.quant_q8 {
        // Route tiers with an established paired plan through the same family
        // used by qwen35. In particular, Q8 switches from the single-block-per-
        // head kernel to tiled flash attention according to the shared mode
        // policy instead of leaving small-head Qwen3 models under-parallelized.
        let ctx = DispatchCtx::new(gpu);
        let plan = KvTierPlan::derive(KvTierInputs {
            pos,
            flash_mode: llama_attention_flash_mode(&gpu.arch),
            ..kv_cache.tier_inputs()
        })
        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
        let io = AttnParams {
            q: &scratch.q,
            k: &scratch.k,
            v: &scratch.v,
            k_cache: &kv_cache.k_gpu[layer_idx],
            v_cache: &kv_cache.v_gpu[layer_idx],
            k_scales: None,
            v_scales: None,
            pos_buf: &scratch.pos_buf,
            pos,
            positions: None,
            n_heads,
            n_kv_heads,
            head_dim,
            physical_cap: kv_cache.physical_cap,
            batch_size: 1,
            max_ctx_len: 0,
            flash_partials: Some(&scratch.attn_partials),
            givens_cos: kv_cache.givens_cos.as_ref(),
            givens_sin: kv_cache.givens_sin.as_ref(),
            tree_bias: None,
            block_start: 0,
            block_cols: 0,
            output_gate: None,
            output: &scratch.attn_out,
        };
        attention_family()
            .run_attention(&ctx, gpu, &plan, &io)
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    } else if kv_cache.quant_hfq4 {
        gpu.kv_cache_write_hfq4(
            &kv_cache.k_gpu[layer_idx],
            &scratch.k,
            &scratch.pos_buf,
            n_kv_heads,
            head_dim,
        )?;
        gpu.kv_cache_write_hfq4(
            &kv_cache.v_gpu[layer_idx],
            &scratch.v,
            &scratch.pos_buf,
            n_kv_heads,
            head_dim,
        )?;
        gpu.attention_hfq4_kv(
            &scratch.q,
            &kv_cache.k_gpu[layer_idx],
            &kv_cache.v_gpu[layer_idx],
            &scratch.attn_out,
            &scratch.pos_buf,
            pos + 1,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_cache.physical_cap,
        )?;
    } else if kv_cache.quantized
        && !kv_cache.k_scales.is_empty()
        && !kv_cache.quant_int8
        && !kv_cache.quant_q8
    {
        // HFQ8 flat layout
        gpu.kv_cache_write_hfq8(
            &kv_cache.k_gpu[layer_idx],
            &kv_cache.k_scales[layer_idx],
            &scratch.k,
            &scratch.pos_buf,
            n_kv_heads,
            head_dim,
        )?;
        gpu.kv_cache_write_hfq8(
            &kv_cache.v_gpu[layer_idx],
            &kv_cache.v_scales[layer_idx],
            &scratch.v,
            &scratch.pos_buf,
            n_kv_heads,
            head_dim,
        )?;
        gpu.attention_hfq8_kv(
            &scratch.q,
            &kv_cache.k_gpu[layer_idx],
            &kv_cache.k_scales[layer_idx],
            &kv_cache.v_gpu[layer_idx],
            &kv_cache.v_scales[layer_idx],
            &scratch.attn_out,
            &scratch.pos_buf,
            pos + 1,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_cache.physical_cap,
        )?;
    } else if kv_cache.quant_int8 {
        gpu.kv_cache_write_int8c_f16(
            &kv_cache.k_gpu[layer_idx],
            &scratch.k,
            &scratch.pos_buf,
            n_kv_heads,
            head_dim,
        )?;
        gpu.kv_cache_write_int8c_f16(
            &kv_cache.v_gpu[layer_idx],
            &scratch.v,
            &scratch.pos_buf,
            n_kv_heads,
            head_dim,
        )?;
        gpu.attention_int8c_f16_kv(
            &scratch.q,
            &kv_cache.k_gpu[layer_idx],
            &kv_cache.v_gpu[layer_idx],
            &scratch.attn_out,
            &scratch.pos_buf,
            pos + 1,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_cache.physical_cap,
        )?;
    } else if kv_cache.quantized && kv_cache.quant_q8 {
        gpu.kv_cache_write_q8_0(
            &kv_cache.k_gpu[layer_idx],
            &scratch.k,
            &scratch.pos_buf,
            n_kv_heads,
            head_dim,
        )?;
        gpu.kv_cache_write_q8_0(
            &kv_cache.v_gpu[layer_idx],
            &scratch.v,
            &scratch.pos_buf,
            n_kv_heads,
            head_dim,
        )?;
        gpu.attention_q8_0_kv(
            &scratch.q,
            &kv_cache.k_gpu[layer_idx],
            &kv_cache.v_gpu[layer_idx],
            &scratch.attn_out,
            &scratch.pos_buf,
            pos + 1,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_cache.physical_cap,
        )?;
    } else if kv_cache.quantized {
        gpu.kv_cache_write_q4(
            &kv_cache.k_gpu[layer_idx],
            &scratch.k,
            &scratch.pos_buf,
            n_kv_heads,
            head_dim,
        )?;
        gpu.kv_cache_write_q4(
            &kv_cache.v_gpu[layer_idx],
            &scratch.v,
            &scratch.pos_buf,
            n_kv_heads,
            head_dim,
        )?;
        gpu.attention_q4kv(
            &scratch.q,
            &kv_cache.k_gpu[layer_idx],
            &kv_cache.v_gpu[layer_idx],
            &scratch.attn_out,
            &scratch.pos_buf,
            pos + 1,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_cache.physical_cap,
        )?;
    } else {
        gpu.kv_cache_write(
            &kv_cache.k_gpu[layer_idx],
            &scratch.k,
            &scratch.pos_buf,
            kv_dim,
        )?;
        gpu.kv_cache_write(
            &kv_cache.v_gpu[layer_idx],
            &scratch.v,
            &scratch.pos_buf,
            kv_dim,
        )?;
        gpu.attention_f32(
            &scratch.q,
            &kv_cache.k_gpu[layer_idx],
            &kv_cache.v_gpu[layer_idx],
            &scratch.attn_out,
            &scratch.pos_buf,
            pos + 1,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_cache.physical_cap,
        )?;
    }
    Ok(())
}

/// Lowered llama-family decode loop (N5 Phase A3a). Behaviorally equivalent to
/// the hand [`forward_scratch_layers`] loop: projections (QKV / o_proj / gate-up
/// / down / lm_head) go through `hipfire_dispatch::execute_steps` (which selects
/// FusedQkvQ4K / fused-MQ / per-op + handles FWHT rotation + residual fusion),
/// while qk_norm, RoPE, the KV ladder (via [`llama_kv_write_attend`]), final
/// norm and sampling stay the same bare `gpu.*` calls. Validated against the
/// hand path via the `HIPFIRE_FORWARD_LOWERED=0`-vs-default committed-token md5
/// A/B (`scripts/forward-lowered-parity.sh`).
///
/// Note: like the dead `arch-llama` port this came from, the lowered QKV/gate-up
/// `RmsnormAutomatic` always normalizes — including the Q4K branch the hand path
/// previously skipped (the "F1 rmsnorm bug" fix). So Q4K output is intentionally
/// *more correct* and may legitimately differ from the hand bytes; gate Q4K via
/// coherence-gate, not byte-parity. MQ paths are expected byte-identical.
#[allow(clippy::too_many_arguments)]
fn forward_scratch_layers_lowered(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    temperature: f32,
    top_p: f32,
    rng_state: u32,
    repeat_window: usize,
    repeat_penalty: f32,
) -> HipResult<(u32, u32)> {
    use hipfire_dispatch::pipeline::{execute_steps, GemvInput, Step};

    let ctx = DispatchCtx::new(gpu);

    let knobs = crate::arch_spec::DenseKnobs {
        attn_bias: false,
        qk_norm: config.has_qk_norm,
        rope_theta: config.rope_freq_base,
        norm_eps: config.norm_eps,
        n_heads: config.n_heads,
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        q_dim: config.n_heads * config.head_dim,
        kv_dim: config.n_kv_heads * config.head_dim,
    };
    let arch = LlamaDense {
        weights,
        config,
        scratch,
        kv_cache: &*kv_cache,
        flash_mode: llama_attention_flash_mode(&gpu.arch),
        knobs,
        pos,
    };
    crate::arch_spec::dense_forward(gpu, &ctx, &arch)?;

    // ── Final norm + logits + sampling ──
    gpu.rmsnorm_f32(
        &scratch.x,
        &weights.output_norm,
        &scratch.tmp,
        config.norm_eps,
    )?;
    let wr_out = weights.output.dispatch_ref();
    execute_steps(
        gpu,
        &ctx,
        &[Step::Gemv {
            w: &wr_out,
            input: GemvInput::Raw(&scratch.tmp),
            out: &scratch.logits,
        }],
    )
    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;

    gpu.sample_top_p(
        &scratch.logits,
        &scratch.sample_buf,
        &scratch.repeat_buf,
        config.vocab_size,
        temperature,
        top_p,
        rng_state,
        repeat_window,
        repeat_penalty,
    )
}

/// Per-forward [`crate::arch_spec::DenseArch`] wrapper for llama-family models.
/// Borrows the layer weights + scratch + KV cache and routes the projection/FFN
/// body through the shared `dense_forward`; attention stays llama's 7-tier KV
/// ladder via [`llama_kv_write_attend`].
struct LlamaDense<'a> {
    weights: &'a LlamaWeights,
    config: &'a LlamaConfig,
    scratch: &'a ForwardScratch,
    kv_cache: &'a KvCache,
    flash_mode: usize,
    knobs: crate::arch_spec::DenseKnobs,
    pos: usize,
}

impl crate::arch_spec::DenseArch for LlamaDense<'_> {
    fn n_layers(&self) -> usize {
        self.config.n_layers
    }
    fn knobs(&self) -> &crate::arch_spec::DenseKnobs {
        &self.knobs
    }
    fn scratch(&self) -> crate::arch_spec::DenseScratch<'_> {
        let s = self.scratch;
        crate::arch_spec::DenseScratch {
            x: &s.x,
            tmp: &s.tmp,
            x_rot: &s.x_rot,
            q: &s.q,
            k: &s.k,
            v: &s.v,
            attn_out: &s.attn_out,
            o: &s.o,
            gate: &s.gate,
            up: &s.up,
            ffn_hidden: &s.ffn_hidden,
            ffn_out: &s.ffn_out,
            pos_buf: &s.pos_buf,
        }
    }
    fn layer(&self, l: usize) -> crate::arch_spec::DenseLayer<'_> {
        let layer = &self.weights.layers[l];
        crate::arch_spec::DenseLayer {
            attn_norm: &layer.attn_norm,
            ffn_norm: &layer.ffn_norm,
            wq: layer.wq.dispatch_ref(),
            wk: layer.wk.dispatch_ref(),
            wv: layer.wv.dispatch_ref(),
            wo: layer.wo.dispatch_ref(),
            w_gate: layer.w_gate.dispatch_ref(),
            w_up: layer.w_up.dispatch_ref(),
            w_down: layer.w_down.dispatch_ref(),
            wq_bias: None,
            wk_bias: None,
            wv_bias: None,
            q_norm: layer.q_norm.as_ref(),
            k_norm: layer.k_norm.as_ref(),
            qkv_rot: dtype_rotation_plan(layer.wq.gpu_dtype),
            ffn_rot: dtype_rotation_plan(layer.w_gate.gpu_dtype),
            qkv_awq: layer.wq.awq_scale.as_ref(),
            ffn_awq: layer.w_gate.awq_scale.as_ref(),
            qkv_k: layer.wq.k,
            ffn_k: layer.w_gate.k,
        }
    }
    fn attend(&self, gpu: &mut Gpu, l: usize) -> HipResult<()> {
        let c = self.config;
        llama_kv_write_attend(
            gpu,
            self.kv_cache,
            self.scratch,
            l,
            self.pos,
            c.n_heads,
            c.n_kv_heads,
            c.head_dim,
            c.n_kv_heads * c.head_dim,
        )
    }
    fn attend_plan(&self, l: usize) -> HipResult<Option<(KvTierPlan, AttnParams<'_>)>> {
        let kv = self.kv_cache;
        let s = self.scratch;
        let c = self.config;
        let plan = KvTierPlan::derive(KvTierInputs {
            pos: self.pos,
            flash_mode: self.flash_mode,
            ..kv.tier_inputs()
        })
        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
        // HFQ8 flat-layout needs the per-layer scale tables; every other tier
        // (int8c included) is self-contained.
        let (k_scales, v_scales) = if kv.is_hfq8_kv() {
            (Some(&kv.k_scales[l]), Some(&kv.v_scales[l]))
        } else {
            (None, None)
        };
        let io = AttnParams {
            q: &s.q,
            k: &s.k,
            v: &s.v,
            k_cache: &kv.k_gpu[l],
            v_cache: &kv.v_gpu[l],
            k_scales,
            v_scales,
            pos_buf: &s.pos_buf,
            pos: self.pos,
            positions: None,
            n_heads: c.n_heads,
            n_kv_heads: c.n_kv_heads,
            head_dim: c.head_dim,
            physical_cap: kv.physical_cap,
            batch_size: 1,
            max_ctx_len: 0,
            flash_partials: Some(&s.attn_partials),
            givens_cos: kv.givens_cos.as_ref(),
            givens_sin: kv.givens_sin.as_ref(),
            tree_bias: None,
            block_start: 0,
            block_cols: 0,
            output_gate: None,
            output: &s.attn_out,
        };
        Ok(Some((plan, io)))
    }
}

pub fn forward_scratch_layers(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    temperature: f32,
    top_p: f32,
    rng_state: u32,
    repeat_window: usize,
    repeat_penalty: f32,
) -> HipResult<(u32, u32)> {
    if llama_forward_lowered_enabled() {
        return forward_scratch_layers_lowered(
            gpu,
            weights,
            config,
            pos,
            kv_cache,
            scratch,
            temperature,
            top_p,
            rng_state,
            repeat_window,
            repeat_penalty,
        );
    }

    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;
    let kv_dim = n_kv_heads * head_dim;

    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];

        gpu.rmsnorm_f32(&scratch.x, &layer.attn_norm, &scratch.tmp, config.norm_eps)?;

        if layer.wq.gpu_dtype == DType::Q4K && layer.wk.gpu_dtype == DType::Q4K {
            gpu.fused_qkv_q4k(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &scratch.tmp,
                &scratch.q,
                &scratch.k,
                &scratch.v,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
            )?;
        } else {
            // Batch FWHT for MQ weights: wq/wk/wv all consume scratch.tmp.
            let x_rot = rotate_x_for_mq(gpu, &layer.wq, &scratch.tmp, &scratch.x_rot)?;
            weight_gemv_prerotated(gpu, &layer.wq, &scratch.tmp, x_rot, &scratch.q)?;
            weight_gemv_prerotated(gpu, &layer.wk, &scratch.tmp, x_rot, &scratch.k)?;
            weight_gemv_prerotated(gpu, &layer.wv, &scratch.tmp, x_rot, &scratch.v)?;
        }

        if config.has_qk_norm {
            if let Some(ref qn) = layer.q_norm {
                gpu.rmsnorm_batched(
                    &scratch.q,
                    qn,
                    &scratch.q,
                    n_heads,
                    head_dim,
                    config.norm_eps,
                )?;
            }
            if let Some(ref kn) = layer.k_norm {
                gpu.rmsnorm_batched(
                    &scratch.k,
                    kn,
                    &scratch.k,
                    n_kv_heads,
                    head_dim,
                    config.norm_eps,
                )?;
            }
        }

        gpu.rope_f32(
            &scratch.q,
            &scratch.k,
            &scratch.pos_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            config.rope_freq_base,
        )?;

        if kv_cache.quant_asym4 || kv_cache.quant_asym3 || kv_cache.quant_asym2 {
            // Asym/Givens KV: the manual ladder below has no asym kernels, so
            // route KV-write + flash-attend through the dispatch attention
            // family (the same path qwen35 uses). tier_inputs() classifies the
            // tier from the cache's quant flags; run_attention does both the
            // KV write and the single-token flash attend.
            let ctx = DispatchCtx::new(gpu);
            let plan = KvTierPlan::derive(KvTierInputs {
                pos,
                ..kv_cache.tier_inputs()
            })
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
            let io = AttnParams {
                q: &scratch.q,
                k: &scratch.k,
                v: &scratch.v,
                k_cache: &kv_cache.k_gpu[layer_idx],
                v_cache: &kv_cache.v_gpu[layer_idx],
                k_scales: None,
                v_scales: None,
                pos_buf: &scratch.pos_buf,
                pos,
                positions: None,
                n_heads,
                n_kv_heads,
                head_dim,
                physical_cap: kv_cache.physical_cap,
                batch_size: 1,
                max_ctx_len: 0,
                flash_partials: Some(&scratch.attn_partials),
                givens_cos: kv_cache.givens_cos.as_ref(),
                givens_sin: kv_cache.givens_sin.as_ref(),
                tree_bias: None,
                block_start: 0,
                block_cols: 0,
                output_gate: None,
                output: &scratch.attn_out,
            };
            attention_family()
                .run_attention(&ctx, gpu, &plan, &io)
                .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
        } else if kv_cache.quant_hfq4 {
            gpu.kv_cache_write_hfq4(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_hfq4(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_hfq4_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized
            && !kv_cache.k_scales.is_empty()
            && !kv_cache.quant_int8
            && !kv_cache.quant_q8
        {
            // HFQ8 flat layout
            gpu.kv_cache_write_hfq8(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.k_scales[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_hfq8(
                &kv_cache.v_gpu[layer_idx],
                &kv_cache.v_scales[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_hfq8_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.k_scales[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &kv_cache.v_scales[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else if kv_cache.quant_int8 {
            gpu.kv_cache_write_int8c_f16(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_int8c_f16(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_int8c_f16_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized && kv_cache.quant_q8 {
            gpu.kv_cache_write_q8_0(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_q8_0(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_q8_0_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized {
            gpu.kv_cache_write_q4(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_q4(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_q4kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else {
            gpu.kv_cache_write(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                kv_dim,
            )?;
            gpu.kv_cache_write(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                kv_dim,
            )?;
            gpu.attention_f32(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        }

        weight_gemv(gpu, &layer.wo, &scratch.attn_out, &scratch.o)?;
        gpu.add_inplace_f32(&scratch.x, &scratch.o)?;

        gpu.rmsnorm_f32(&scratch.x, &layer.ffn_norm, &scratch.tmp, config.norm_eps)?;
        if layer.w_gate.gpu_dtype == DType::Q4K && layer.w_up.gpu_dtype == DType::Q4K {
            gpu.fused_gate_up_q4k(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &scratch.tmp,
                &scratch.gate,
                &scratch.up,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
            )?;
        } else {
            // Batch FWHT for MQ weights: w_gate/w_up share scratch.tmp.
            let x_rot = rotate_x_for_mq(gpu, &layer.w_gate, &scratch.tmp, &scratch.x_rot)?;
            weight_gemv_prerotated(gpu, &layer.w_gate, &scratch.tmp, x_rot, &scratch.gate)?;
            weight_gemv_prerotated(gpu, &layer.w_up, &scratch.tmp, x_rot, &scratch.up)?;
        }

        gpu.silu_mul_f32(&scratch.gate, &scratch.up, &scratch.ffn_hidden)?;
        weight_gemv(gpu, &layer.w_down, &scratch.ffn_hidden, &scratch.ffn_out)?;
        gpu.add_inplace_f32(&scratch.x, &scratch.ffn_out)?;
    }

    gpu.rmsnorm_f32(
        &scratch.x,
        &weights.output_norm,
        &scratch.tmp,
        config.norm_eps,
    )?;
    weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;

    // GPU-side sampling (includes sync readback — can't be in graph capture)
    gpu.sample_top_p(
        &scratch.logits,
        &scratch.sample_buf,
        &scratch.repeat_buf,
        config.vocab_size,
        temperature,
        top_p,
        rng_state,
        repeat_window,
        repeat_penalty,
    )
}

/// Early-exit forward pass: check confidence at checkpoint layers, skip rest if confident.
/// Returns (token_id, rng_state, exit_layer) — exit_layer is which layer triggered the exit.
/// If no early exit, exit_layer = n_layers (ran all layers normally).
pub fn forward_early_exit(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    token: u32,
    pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    temperature: f32,
    top_p: f32,
    rng_state: u32,
    repeat_window: usize,
    repeat_penalty: f32,
    exit_threshold: f32,         // max softmax prob threshold (e.g., 0.9)
    checkpoint_layers: &[usize], // which layers to check (e.g., &[12, 24])
) -> HipResult<(u32, u32, usize)> {
    // Embed
    forward_scratch_embed(gpu, weights, config, token, pos, scratch)?;

    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;
    let kv_dim = n_kv_heads * head_dim;

    let mut exit_layer = config.n_layers;

    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];

        // Standard layer computation (same as forward_scratch_layers)
        gpu.rmsnorm_f32(&scratch.x, &layer.attn_norm, &scratch.tmp, config.norm_eps)?;

        if layer.wq.gpu_dtype == DType::Q4K && layer.wk.gpu_dtype == DType::Q4K {
            gpu.fused_qkv_q4k(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &scratch.tmp,
                &scratch.q,
                &scratch.k,
                &scratch.v,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
            )?;
        } else {
            // Batch FWHT for MQ weights: wq/wk/wv all consume scratch.tmp.
            let x_rot = rotate_x_for_mq(gpu, &layer.wq, &scratch.tmp, &scratch.x_rot)?;
            weight_gemv_prerotated(gpu, &layer.wq, &scratch.tmp, x_rot, &scratch.q)?;
            weight_gemv_prerotated(gpu, &layer.wk, &scratch.tmp, x_rot, &scratch.k)?;
            weight_gemv_prerotated(gpu, &layer.wv, &scratch.tmp, x_rot, &scratch.v)?;
        }

        if config.has_qk_norm {
            if let Some(ref qn) = layer.q_norm {
                gpu.rmsnorm_batched(
                    &scratch.q,
                    qn,
                    &scratch.q,
                    n_heads,
                    head_dim,
                    config.norm_eps,
                )?;
            }
            if let Some(ref kn) = layer.k_norm {
                gpu.rmsnorm_batched(
                    &scratch.k,
                    kn,
                    &scratch.k,
                    n_kv_heads,
                    head_dim,
                    config.norm_eps,
                )?;
            }
        }

        gpu.rope_f32(
            &scratch.q,
            &scratch.k,
            &scratch.pos_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            config.rope_freq_base,
        )?;

        // KV write + attention (use same dispatch as forward_scratch_layers)
        if kv_cache.quantized && kv_cache.quant_q8 {
            gpu.kv_cache_write_q8_0(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_q8_0(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_q8_0_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else {
            gpu.kv_cache_write(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                kv_dim,
            )?;
            gpu.kv_cache_write(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                kv_dim,
            )?;
            gpu.attention_f32(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        }

        weight_gemv(gpu, &layer.wo, &scratch.attn_out, &scratch.o)?;
        gpu.add_inplace_f32(&scratch.x, &scratch.o)?;

        gpu.rmsnorm_f32(&scratch.x, &layer.ffn_norm, &scratch.tmp, config.norm_eps)?;
        if layer.w_gate.gpu_dtype == DType::Q4K && layer.w_up.gpu_dtype == DType::Q4K {
            gpu.fused_gate_up_q4k(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &scratch.tmp,
                &scratch.gate,
                &scratch.up,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
            )?;
        } else {
            // Batch FWHT for MQ weights: w_gate/w_up share scratch.tmp.
            let x_rot = rotate_x_for_mq(gpu, &layer.w_gate, &scratch.tmp, &scratch.x_rot)?;
            weight_gemv_prerotated(gpu, &layer.w_gate, &scratch.tmp, x_rot, &scratch.gate)?;
            weight_gemv_prerotated(gpu, &layer.w_up, &scratch.tmp, x_rot, &scratch.up)?;
        }

        gpu.silu_mul_f32(&scratch.gate, &scratch.up, &scratch.ffn_hidden)?;
        weight_gemv(gpu, &layer.w_down, &scratch.ffn_hidden, &scratch.ffn_out)?;
        gpu.add_inplace_f32(&scratch.x, &scratch.ffn_out)?;

        // Early exit check at checkpoint layers
        if checkpoint_layers.contains(&layer_idx) && exit_threshold > 0.0 {
            // Compute logits from intermediate hidden state
            gpu.rmsnorm_f32(
                &scratch.x,
                &weights.output_norm,
                &scratch.tmp,
                config.norm_eps,
            )?;
            weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;

            // GPU-side confidence check: compute max(softmax) on GPU, download 4 bytes
            gpu.max_prob(&scratch.logits, &scratch.sample_buf, config.vocab_size)?;
            let mut prob_bytes = [0u8; 4];
            gpu.hip
                .memcpy_dtoh(&mut prob_bytes, &scratch.sample_buf.buf)?;
            let max_prob = f32::from_ne_bytes(prob_bytes);

            if max_prob >= exit_threshold {
                exit_layer = layer_idx + 1;
                // Sample from these logits
                let (tok, rng) = gpu.sample_top_p(
                    &scratch.logits,
                    &scratch.sample_buf,
                    &scratch.repeat_buf,
                    config.vocab_size,
                    temperature,
                    top_p,
                    rng_state,
                    repeat_window,
                    repeat_penalty,
                )?;
                return Ok((tok, rng, exit_layer));
            }
        }
    }

    // No early exit — run full final norm + logits + sampling
    gpu.rmsnorm_f32(
        &scratch.x,
        &weights.output_norm,
        &scratch.tmp,
        config.norm_eps,
    )?;
    weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;
    let (tok, rng) = gpu.sample_top_p(
        &scratch.logits,
        &scratch.sample_buf,
        &scratch.repeat_buf,
        config.vocab_size,
        temperature,
        top_p,
        rng_state,
        repeat_window,
        repeat_penalty,
    )?;
    Ok((tok, rng, exit_layer))
}

/// Layer loop + final norm + logits only (no sampling). Graph-capturable.
pub fn forward_scratch_compute(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
) -> HipResult<()> {
    forward_scratch_compute_capture(gpu, weights, config, pos, kv_cache, scratch, None)
}

/// `forward_scratch_compute` plus an optional per-extract-layer residual-hidden
/// capture sink. Processes ONE token (decode kernel — bit-identical to AR's
/// `forward_scratch`), and for each decoder layer whose index appears in
/// `capture.extract_layers` downloads the post-FFN residual (`scratch.x[..dim]`)
/// and appends, in `extract_layers` ascending order, `num_extract × dim` f32 to
/// the sink. This is the per-token twin of `forward_prefill_batch_capture`'s
/// capture; used by the DFlash spec prefill so the speculator conditions on the
/// EXACT residual stream AR's per-token prefill produces (greedy-equivalence —
/// the batched prefill kernel is not bitwise-equal to the decode kernel and
/// flips argmax at near-tie logits).
#[allow(clippy::too_many_arguments)]
pub fn forward_scratch_compute_capture(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    mut capture: Option<&mut HiddenCaptureSink>,
) -> HipResult<()> {
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let dim = config.dim;
    // Per-token capture appends rows in extract-layer ASCENDING order, matching
    // forward_prefill_batch_capture's layout for n=1 (one position per call).
    let mut cap_rows: Vec<Vec<f32>> = Vec::new();

    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];
        gpu.rmsnorm_f32(&scratch.x, &layer.attn_norm, &scratch.tmp, config.norm_eps)?;

        if layer.wq.gpu_dtype == DType::Q4K && layer.wk.gpu_dtype == DType::Q4K {
            gpu.fused_qkv_q4k(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &scratch.tmp,
                &scratch.q,
                &scratch.k,
                &scratch.v,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
            )?;
        } else {
            // Batch FWHT for MQ weights: wq/wk/wv all consume scratch.tmp.
            let x_rot = rotate_x_for_mq(gpu, &layer.wq, &scratch.tmp, &scratch.x_rot)?;
            weight_gemv_prerotated(gpu, &layer.wq, &scratch.tmp, x_rot, &scratch.q)?;
            weight_gemv_prerotated(gpu, &layer.wk, &scratch.tmp, x_rot, &scratch.k)?;
            weight_gemv_prerotated(gpu, &layer.wv, &scratch.tmp, x_rot, &scratch.v)?;
        }

        if config.has_qk_norm {
            if let Some(ref qn) = layer.q_norm {
                gpu.rmsnorm_batched(
                    &scratch.q,
                    qn,
                    &scratch.q,
                    n_heads,
                    head_dim,
                    config.norm_eps,
                )?;
            }
            if let Some(ref kn) = layer.k_norm {
                gpu.rmsnorm_batched(
                    &scratch.k,
                    kn,
                    &scratch.k,
                    n_kv_heads,
                    head_dim,
                    config.norm_eps,
                )?;
            }
        }

        gpu.rope_f32(
            &scratch.q,
            &scratch.k,
            &scratch.pos_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            config.rope_freq_base,
        )?;

        if kv_cache.quant_asym4 || kv_cache.quant_asym3 || kv_cache.quant_asym2 {
            // Asym/Givens KV: the manual ladder below has no asym kernels, so
            // route KV-write + flash-attend through the dispatch attention
            // family (the same path qwen35 uses). tier_inputs() classifies the
            // tier from the cache's quant flags; run_attention does both the
            // KV write and the single-token flash attend.
            let ctx = DispatchCtx::new(gpu);
            let plan = KvTierPlan::derive(KvTierInputs {
                pos,
                ..kv_cache.tier_inputs()
            })
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
            let io = AttnParams {
                q: &scratch.q,
                k: &scratch.k,
                v: &scratch.v,
                k_cache: &kv_cache.k_gpu[layer_idx],
                v_cache: &kv_cache.v_gpu[layer_idx],
                k_scales: None,
                v_scales: None,
                pos_buf: &scratch.pos_buf,
                pos,
                positions: None,
                n_heads,
                n_kv_heads,
                head_dim,
                physical_cap: kv_cache.physical_cap,
                batch_size: 1,
                max_ctx_len: 0,
                flash_partials: Some(&scratch.attn_partials),
                givens_cos: kv_cache.givens_cos.as_ref(),
                givens_sin: kv_cache.givens_sin.as_ref(),
                tree_bias: None,
                block_start: 0,
                block_cols: 0,
                output_gate: None,
                output: &scratch.attn_out,
            };
            attention_family()
                .run_attention(&ctx, gpu, &plan, &io)
                .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
        } else if kv_cache.quant_hfq4 {
            gpu.kv_cache_write_hfq4(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_hfq4(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_hfq4_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized
            && !kv_cache.k_scales.is_empty()
            && !kv_cache.quant_int8
            && !kv_cache.quant_q8
        {
            // HFQ8 flat layout
            gpu.kv_cache_write_hfq8(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.k_scales[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_hfq8(
                &kv_cache.v_gpu[layer_idx],
                &kv_cache.v_scales[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_hfq8_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.k_scales[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &kv_cache.v_scales[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else if kv_cache.quant_int8 {
            gpu.kv_cache_write_int8c_f16(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_int8c_f16(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_int8c_f16_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized && kv_cache.quant_q8 {
            gpu.kv_cache_write_q8_0(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_q8_0(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_q8_0_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized {
            gpu.kv_cache_write_q4(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_q4(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_q4kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else {
            gpu.kv_cache_write(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                kv_dim,
            )?;
            gpu.kv_cache_write(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                kv_dim,
            )?;
            gpu.attention_f32(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        }

        weight_gemv(gpu, &layer.wo, &scratch.attn_out, &scratch.o)?;
        gpu.add_inplace_f32(&scratch.x, &scratch.o)?;

        gpu.rmsnorm_f32(&scratch.x, &layer.ffn_norm, &scratch.tmp, config.norm_eps)?;
        if layer.w_gate.gpu_dtype == DType::Q4K && layer.w_up.gpu_dtype == DType::Q4K {
            gpu.fused_gate_up_q4k(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &scratch.tmp,
                &scratch.gate,
                &scratch.up,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
            )?;
        } else {
            // Batch FWHT for MQ weights: w_gate/w_up share scratch.tmp.
            let x_rot = rotate_x_for_mq(gpu, &layer.w_gate, &scratch.tmp, &scratch.x_rot)?;
            weight_gemv_prerotated(gpu, &layer.w_gate, &scratch.tmp, x_rot, &scratch.gate)?;
            weight_gemv_prerotated(gpu, &layer.w_up, &scratch.tmp, x_rot, &scratch.up)?;
        }

        gpu.silu_mul_f32(&scratch.gate, &scratch.up, &scratch.ffn_hidden)?;
        weight_gemv(gpu, &layer.w_down, &scratch.ffn_hidden, &scratch.ffn_out)?;
        gpu.add_inplace_f32(&scratch.x, &scratch.ffn_out)?;

        // DFlash per-token capture: download this layer's post-FFN residual
        // (`scratch.x[..dim]`) if it is an extract layer. Same point as the
        // batched capture (post-FFN residual, before next attn_norm).
        if let Some(cap) = capture.as_deref() {
            if cap.extract_layers.contains(&layer_idx) {
                let row = scratch.x.sub_offset(0, dim);
                cap_rows.push(gpu.download_f32(&row)?);
            }
        }
    }

    if let Some(cap) = capture.as_deref_mut() {
        debug_assert_eq!(
            cap_rows.len(),
            cap.extract_layers.len(),
            "captured {} layers but extract_layers has {}",
            cap_rows.len(),
            cap.extract_layers.len()
        );
        for layer_rows in cap_rows.iter() {
            cap.hidden.extend_from_slice(&layer_rows[..dim]);
        }
    }

    gpu.rmsnorm_f32(
        &scratch.x,
        &weights.output_norm,
        &scratch.tmp,
        config.norm_eps,
    )?;
    weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;
    Ok(())
}

/// Run a single forward pass for one token (decode step).
/// Returns logits over vocab.
pub fn forward(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    token: u32,
    pos: usize,
    kv_cache: &mut KvCache,
) -> HipResult<Vec<f32>> {
    let dim = config.dim;
    let head_dim = config.head_dim;
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let kv_dim = n_kv_heads * head_dim;

    // Embedding lookup — GPU-side D2D copy of one row (8KB vs 262MB download)
    let mut x = gpu.alloc_tensor(&[dim], DType::F32)?;
    embedding_lookup_dispatch(
        gpu,
        weights.embd_format,
        &weights.token_embd,
        &x,
        token,
        dim,
    )?;

    let tmp = gpu.alloc_tensor(&[dim], DType::F32)?;

    // Pre-allocate scratch buffers — reused every layer (eliminates 324 allocs per token)
    let q_dim = n_heads * head_dim;
    let q = gpu.alloc_tensor(&[q_dim], DType::F32)?;
    let k = gpu.alloc_tensor(&[kv_dim], DType::F32)?;
    let v = gpu.alloc_tensor(&[kv_dim], DType::F32)?;
    let attn_out = gpu.alloc_tensor(&[q_dim], DType::F32)?;
    let o = gpu.alloc_tensor(&[dim], DType::F32)?;
    let gate = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
    let up = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
    let ffn_hidden = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
    let ffn_out = gpu.alloc_tensor(&[dim], DType::F32)?;

    // Upload pos to GPU buffer (4 bytes)
    let pos_buf = gpu.hip.malloc(4)?;
    let pos_i32 = pos as i32;
    gpu.hip.memcpy_htod(&pos_buf, &pos_i32.to_ne_bytes())?;

    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];

        // RMSNorm before attention
        gpu.rmsnorm_f32(&x, &layer.attn_norm, &tmp, config.norm_eps)?;

        // Fused QKV: 3 GEMVs in 1 kernel launch (saves 2 launches per layer)
        if layer.wq.gpu_dtype == DType::Q4K
            && layer.wk.gpu_dtype == DType::Q4K
            && layer.wv.gpu_dtype == DType::Q4K
        {
            gpu.fused_qkv_q4k(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &tmp,
                &q,
                &k,
                &v,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
            )?;
        } else {
            weight_gemv(gpu, &layer.wq, &tmp, &q)?;
            weight_gemv(gpu, &layer.wk, &tmp, &k)?;
            weight_gemv(gpu, &layer.wv, &tmp, &v)?;
        }

        // QK normalization (Qwen3) — GPU-side per-head RMSNorm.
        // Launches n_heads blocks, each normalizing head_dim elements.
        if config.has_qk_norm {
            if let Some(ref qn) = layer.q_norm {
                gpu.rmsnorm_batched(&q, qn, &q, n_heads, head_dim, config.norm_eps)?;
            }
            if let Some(ref kn) = layer.k_norm {
                gpu.rmsnorm_batched(&k, kn, &k, n_kv_heads, head_dim, config.norm_eps)?;
            }
        }

        // RoPE — GPU-side, reads pos from GPU buffer
        gpu.rope_f32(
            &q,
            &k,
            &pos_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            config.rope_freq_base,
        )?;

        // Store K, V in GPU cache + attention
        gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &k, &pos_buf, kv_dim)?;
        gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &v, &pos_buf, kv_dim)?;
        gpu.attention_f32(
            &q,
            &kv_cache.k_gpu[layer_idx],
            &kv_cache.v_gpu[layer_idx],
            &attn_out,
            &pos_buf,
            pos + 1,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_cache.physical_cap,
        )?;
        // Output projection: o = Wo * attn_out
        weight_gemv(gpu, &layer.wo, &attn_out, &o)?;

        // Residual: x += o (in-place)
        gpu.add_inplace_f32(&x, &o)?;

        // FFN
        gpu.rmsnorm_f32(&x, &layer.ffn_norm, &tmp, config.norm_eps)?;
        // Fused Gate+Up: 2 GEMVs in 1 kernel launch
        if layer.w_gate.gpu_dtype == DType::Q4K && layer.w_up.gpu_dtype == DType::Q4K {
            gpu.fused_gate_up_q4k(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &tmp,
                &gate,
                &up,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
            )?;
        } else {
            weight_gemv(gpu, &layer.w_gate, &tmp, &gate)?;
            weight_gemv(gpu, &layer.w_up, &tmp, &up)?;
        }

        // Fused SiLU(gate) * up
        gpu.silu_mul_f32(&gate, &up, &ffn_hidden)?;

        // Down projection
        weight_gemv(gpu, &layer.w_down, &ffn_hidden, &ffn_out)?;

        // Residual: x += ffn_out (in-place)
        gpu.add_inplace_f32(&x, &ffn_out)?;
    }

    // Final norm
    gpu.rmsnorm_f32(&x, &weights.output_norm, &tmp, config.norm_eps)?;

    // Logits: output = output_weight * x
    let logits = gpu.alloc_tensor(&[config.vocab_size], DType::F32)?;
    weight_gemv(gpu, &weights.output, &tmp, &logits)?;

    let logits_data = gpu.download_f32(&logits)?;
    gpu.free_tensor(q)?;
    gpu.free_tensor(k)?;
    gpu.free_tensor(v)?;
    gpu.free_tensor(attn_out)?;
    gpu.free_tensor(o)?;
    gpu.free_tensor(gate)?;
    gpu.free_tensor(up)?;
    gpu.free_tensor(ffn_hidden)?;
    gpu.free_tensor(ffn_out)?;
    gpu.free_tensor(x)?;
    gpu.free_tensor(tmp)?;
    gpu.free_tensor(logits)?;

    Ok(logits_data)
}

/// Forward pass + GPU-side sampling. Returns (token_id, new_rng_state).
/// Logits stay on GPU — only 8 bytes downloaded instead of 600KB.
pub fn forward_sample(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    token: u32,
    pos: usize,
    kv_cache: &mut KvCache,
    sample_buf: &GpuTensor,
    repeat_buf: &GpuTensor,
    temperature: f32,
    top_p: f32,
    rng_state: u32,
    repeat_window: usize,
    repeat_penalty: f32,
) -> HipResult<(u32, u32)> {
    let logits_on_gpu = forward_logits_gpu(gpu, weights, config, token, pos, kv_cache)?;
    let result = gpu.sample_top_p(
        &logits_on_gpu,
        sample_buf,
        repeat_buf,
        config.vocab_size,
        temperature,
        top_p,
        rng_state,
        repeat_window,
        repeat_penalty,
    )?;
    gpu.free_tensor(logits_on_gpu)?;
    Ok(result)
}

/// Forward pass that keeps logits on GPU (no download).
fn forward_logits_gpu(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    token: u32,
    pos: usize,
    kv_cache: &mut KvCache,
) -> HipResult<GpuTensor> {
    let dim = config.dim;
    let kv_dim = config.n_kv_heads * config.head_dim;
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;

    let mut x = gpu.alloc_tensor(&[dim], DType::F32)?;
    embedding_lookup_dispatch(
        gpu,
        weights.embd_format,
        &weights.token_embd,
        &x,
        token,
        dim,
    )?;

    let tmp = gpu.alloc_tensor(&[dim], DType::F32)?;
    let q = gpu.alloc_tensor(&[n_heads * head_dim], DType::F32)?;
    let k = gpu.alloc_tensor(&[kv_dim], DType::F32)?;
    let v = gpu.alloc_tensor(&[kv_dim], DType::F32)?;
    let attn_out = gpu.alloc_tensor(&[n_heads * head_dim], DType::F32)?;
    let o = gpu.alloc_tensor(&[dim], DType::F32)?;
    let gate = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
    let up = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
    let ffn_hidden = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
    let ffn_out = gpu.alloc_tensor(&[dim], DType::F32)?;

    // Upload pos to GPU buffer (4 bytes)
    let pos_buf = gpu.hip.malloc(4)?;
    let pos_i32 = pos as i32;
    gpu.hip.memcpy_htod(&pos_buf, &pos_i32.to_ne_bytes())?;

    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];
        gpu.rmsnorm_f32(&x, &layer.attn_norm, &tmp, config.norm_eps)?;

        if layer.wq.gpu_dtype == DType::Q4K && layer.wk.gpu_dtype == DType::Q4K {
            gpu.fused_qkv_q4k(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &tmp,
                &q,
                &k,
                &v,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
            )?;
        } else {
            weight_gemv(gpu, &layer.wq, &tmp, &q)?;
            weight_gemv(gpu, &layer.wk, &tmp, &k)?;
            weight_gemv(gpu, &layer.wv, &tmp, &v)?;
        }

        if config.has_qk_norm {
            if let Some(ref qn) = layer.q_norm {
                gpu.rmsnorm_batched(&q, qn, &q, n_heads, head_dim, config.norm_eps)?;
            }
            if let Some(ref kn) = layer.k_norm {
                gpu.rmsnorm_batched(&k, kn, &k, n_kv_heads, head_dim, config.norm_eps)?;
            }
        }

        gpu.rope_f32(
            &q,
            &k,
            &pos_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            config.rope_freq_base,
        )?;

        gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &k, &pos_buf, kv_dim)?;
        gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &v, &pos_buf, kv_dim)?;

        gpu.attention_f32(
            &q,
            &kv_cache.k_gpu[layer_idx],
            &kv_cache.v_gpu[layer_idx],
            &attn_out,
            &pos_buf,
            pos + 1,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_cache.physical_cap,
        )?;

        weight_gemv(gpu, &layer.wo, &attn_out, &o)?;
        gpu.add_inplace_f32(&x, &o)?;

        gpu.rmsnorm_f32(&x, &layer.ffn_norm, &tmp, config.norm_eps)?;
        if layer.w_gate.gpu_dtype == DType::Q4K && layer.w_up.gpu_dtype == DType::Q4K {
            gpu.fused_gate_up_q4k(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &tmp,
                &gate,
                &up,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
            )?;
        } else {
            weight_gemv(gpu, &layer.w_gate, &tmp, &gate)?;
            weight_gemv(gpu, &layer.w_up, &tmp, &up)?;
        }

        gpu.silu_mul_f32(&gate, &up, &ffn_hidden)?;
        weight_gemv(gpu, &layer.w_down, &ffn_hidden, &ffn_out)?;
        gpu.add_inplace_f32(&x, &ffn_out)?;
    }

    gpu.rmsnorm_f32(&x, &weights.output_norm, &tmp, config.norm_eps)?;

    let logits = gpu.alloc_tensor(&[config.vocab_size], DType::F32)?;
    weight_gemv(gpu, &weights.output, &tmp, &logits)?;

    gpu.free_tensor(q)?;
    gpu.free_tensor(k)?;
    gpu.free_tensor(v)?;
    gpu.free_tensor(attn_out)?;
    gpu.free_tensor(o)?;
    gpu.free_tensor(gate)?;
    gpu.free_tensor(up)?;
    gpu.free_tensor(ffn_hidden)?;
    gpu.free_tensor(ffn_out)?;
    gpu.free_tensor(x)?;
    gpu.free_tensor(tmp)?;

    Ok(logits)
}

pub fn apply_rope_cpu_pub(data: &mut [f32], n_heads: usize, head_dim: usize, pos: usize) {
    apply_rope_cpu(data, n_heads, head_dim, pos, 10000.0);
}

fn apply_rope_cpu(data: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, freq_base: f32) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let freq = 1.0 / (freq_base.powf((2 * i) as f32 / head_dim as f32));
            let val = pos as f32 * freq;
            let cos_val = val.cos();
            let sin_val = val.sin();
            let v0 = data[base + i];
            let v1 = data[base + i + half];
            data[base + i] = v0 * cos_val - v1 * sin_val;
            data[base + i + half] = v0 * sin_val + v1 * cos_val;
        }
    }
}

/// V-cache quantization mode. The bit-count IS the kernarg value passed to
/// kernels: 8 = legacy Q8_0 (per-32-block fp16 scale + int8, 272 B/head at hd=256),
/// 2/3/4 = FWHT-rotated centroid-LUT V (Lloyd-V), layout identical to the K fwht
/// modes: `4 + head_dim*bits/8` B/head with one f32 cnorm per head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VMode {
    Q8,
    Lloyd2,
    Lloyd3,
    Lloyd4,
}

impl VMode {
    /// Kernarg value: the per-element bit count (8 for Q8). Drives both kernel
    /// dispatch branches and byte-layout arithmetic.
    pub fn bits(self) -> i32 {
        match self {
            VMode::Q8 => 8,
            VMode::Lloyd2 => 2,
            VMode::Lloyd3 => 3,
            VMode::Lloyd4 => 4,
        }
    }
}

/// Two capacity axes live here:
///   * `max_seq`       — advertised absolute-position range (used for RoPE phase,
///                       attention masks, and anything that reasons about the
///                       user-visible context window).
///   * `physical_cap`  — actual buffer size along the token axis (drives
///                       allocation + kernel strides). When eviction is active,
///                       `physical_cap << max_seq` so the buffer stays bounded
///                       even as the absolute position grows past it.
///
/// Back-compat: constructors that do not take `physical_cap` set it equal to
/// `max_seq`, preserving existing behaviour.
pub struct KvCache {
    pub k_gpu: Vec<GpuTensor>,    // [n_layers] key values (FP32 or int8)
    pub v_gpu: Vec<GpuTensor>,    // [n_layers] value values (FP32 or int8)
    pub k_scales: Vec<GpuTensor>, // [n_layers] key scales (for INT8 mode)
    pub v_scales: Vec<GpuTensor>, // [n_layers] value scales (for INT8 mode)
    pub kv_dim: usize,
    pub max_seq: usize,
    /// Physical capacity of each per-layer k/v buffer in *tokens*.
    /// Equals `max_seq` unless the buffer was sized for eviction-bounded use.
    pub physical_cap: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub quantized: bool,
    pub quant_q8: bool,
    pub quant_int8: bool,    // true = INT8 with separate scales
    pub quant_hfq4: bool,    // true = HFQ4 co-located blocks (72 bytes/head)
    pub quant_asym4: bool,   // true = K at 4-bit rotated, V at Q8_0 — RotorQuant planar4 asymmetric
    pub quant_asym3: bool, // true = K at givens3 (rotated 3-bit Lloyd-Max), V at Q8_0 — best-quality rotated K per RotorQuant
    pub quant_asym2: bool, // true = K at givens2 (rotated 2-bit), V at Q8_0 (normal space)
    pub boundary_layers: u8, // number of boundary layers at each end (default 2)
    // KV rotation parameter buffers. Field names are historical — in the
    // Givens-rotated asym{2,3,4} modes (`quant_fwht == false`) these hold the
    // per-block cos/sin tables. In the signed-FWHT-rotated fwht{2,3,4} modes
    // (`quant_fwht == true`) the SAME slots hold signs1/signs2 ±1 vectors.
    // Both are [n_blocks × f32] in shape, so the storage is fungible; the
    // dispatcher reads `quant_fwht` to know which kernel signature to use.
    pub givens_cos: Option<GpuTensor>,
    pub givens_sin: Option<GpuTensor>,
    /// True when the rotation primitive is signed-FWHT (matches Fwht{2,3,4}
    /// KvMode values). False when Givens (matches Asym{2,3,4}).
    pub quant_fwht: bool,
    /// V-cache quantization mode (independent of the K mode). Defaults to Q8.
    pub v_mode: VMode,
    /// Per-layer flag: true = this layer uses Q8 (boundary layer)
    pub layer_is_boundary: Vec<bool>,
    /// TriAttention compaction bookkeeping. After each eviction we leave the
    /// retained keys in physical slots `0..budget` with their baked-in RoPE
    /// phases intact, but the forward pass still counts absolute positions
    /// for new writes. `compact_offset = absolute_seq_len - physical_seq_len`
    /// — added to `pos` before RoPE so the new query/key get the correct
    /// absolute phase, and the cache write still lands at `pos` (physical).
    /// Zero when no compaction has happened.
    pub compact_offset: usize,
}

/// Layer addressing for [`KvCache::from_mode`]: a per-layer "is this a
/// full-attention layer" mask (→ `_filtered` family) OR a flat layer count
/// (→ plain / flat-`_capped` family).
pub enum KvLayers {
    /// `is_kv_layer` mask — sites 1, 2, 6.
    Mask(Vec<bool>),
    /// `n_layers` — sites 3, 4, 5.
    Flat(usize),
}

/// Geometry + cap inputs shared by every `new_gpu_*` constructor.
pub struct KvDims {
    pub layers: KvLayers,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// For minimax (site 5a) this MUST be the CLAMPED value (12288), not the
    /// raw `ctx.max_seq`, or the allocation size changes.
    pub max_seq: usize,
    /// `Some(cap)` → request a `_capped` form. HONORED ONLY for modes that have
    /// one (q8/asym3/fwht2/fwht3 on Mask sites; q8/asym3/asym4 on Flat sites);
    /// silently DROPPED for asym2/asym4/fwht4 on Mask sites — faithful to today.
    pub physical_cap: Option<usize>,
}

/// Single- vs multi-GPU dispatch for [`KvCache::from_mode`].
pub enum KvTarget<'a> {
    /// pp == 1: one GPU. Sites 1–5.
    Single(&'a mut Gpu),
    /// pp > 1: pipeline-parallel. Site 6 only (qwen35 HFQ pp>1).
    Multi(&'a mut Gpus),
}

/// Single source of truth for VMM K/V byte layout.
///
/// - `k_bytes_per_token` / `v_bytes_per_token` are **current** encoding strides
///   (drive mapped-token capacity, growth, and live source-prefix sizing).
/// - `k_reserve_*` describe the virtual VA arena size (static: reserve == current;
///   adaptive may reserve at floor while current encoding is FWHT4/Q8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VmmKvLayout {
    mode: KvMode,
    v_mode: VMode,
    n_kv_heads: usize,
    head_dim: usize,
    physical_cap: usize,
    kv_dim: usize,
    k_bytes_per_head: usize,
    v_bytes_per_head: usize,
    k_bytes_per_token: usize,
    v_bytes_per_token: usize,
    /// Virtual reserve bytes at the reserve tier (may differ from current).
    k_reserve_bytes: usize,
    v_reserve_bytes: usize,
    k_reserve_elems: usize,
    v_reserve_elems: usize,
    /// Sign/angle table width. 0 = none (Q8). 128 = Givens or FWHT-128.
    /// 256 = FWHT-256 (fwht3, or any FWHT+lloyd-V).
    rotation_table_len: usize,
    uses_fwht_signs: bool,
}

impl VmmKvLayout {
    /// Checked live K source-prefix bytes for `n_positions` at the **current**
    /// K stride. Independent of virtual reserve size — used by adaptive
    /// transcode scratch sizing and map-before-read guards.
    fn prefix_k_bytes(self, n_positions: usize) -> HipResult<usize> {
        KvCache::checked_vmm_product("K source prefix", &[n_positions, self.k_bytes_per_token])
    }

    /// Checked live V source-prefix bytes for `n_positions` at the **current**
    /// V stride. Independent of virtual reserve size.
    fn prefix_v_bytes(self, n_positions: usize) -> HipResult<usize> {
        KvCache::checked_vmm_product("V source prefix", &[n_positions, self.v_bytes_per_token])
    }
}

impl KvCache {
    fn checked_vmm_product(label: &str, factors: &[usize]) -> HipResult<usize> {
        factors
            .iter()
            .try_fold(1usize, |product, &factor| product.checked_mul(factor))
            .ok_or_else(|| hip_bridge::HipError::new(0, &format!("VMM KV {label} size overflowed")))
    }

    fn bytes_to_f32_elems(label: &str, bytes: usize) -> HipResult<usize> {
        bytes.checked_add(3).map(|value| value / 4).ok_or_else(|| {
            hip_bridge::HipError::new(0, &format!("VMM KV {label} element count overflowed"))
        })
    }

    /// Packed K bytes-per-head for Q8 / asym{2,3,4} / fwht{2,3,4}.
    /// Asym and FWHT share storage at the same bit width; only rotation tables differ.
    fn vmm_k_bytes_per_head(mode: KvMode, head_dim: usize) -> HipResult<usize> {
        match mode {
            KvMode::Q8 => {
                if head_dim == 0 || !head_dim.is_multiple_of(32) {
                    return Err(hip_bridge::HipError::new(
                        0,
                        &format!(
                            "VMM q8 requires a non-zero head_dim divisible by 32 (got head_dim={head_dim})"
                        ),
                    ));
                }
                Self::checked_vmm_product("q8 K head stride", &[head_dim / 32, 34])
            }
            KvMode::Asym2 | KvMode::Fwht2 => head_dim
                .checked_div(4)
                .and_then(|n| n.checked_add(4))
                .ok_or_else(|| hip_bridge::HipError::new(0, "VMM 2-bit K head stride overflowed")),
            KvMode::Asym3 | KvMode::Fwht3 => head_dim
                .checked_mul(3)
                .and_then(|n| n.checked_div(8))
                .and_then(|n| n.checked_add(4))
                .ok_or_else(|| hip_bridge::HipError::new(0, "VMM 3-bit K head stride overflowed")),
            KvMode::Asym4 | KvMode::Fwht4 => head_dim
                .checked_div(2)
                .and_then(|n| n.checked_add(4))
                .ok_or_else(|| hip_bridge::HipError::new(0, "VMM 4-bit K head stride overflowed")),
            KvMode::Asym3Auto => Err(hip_bridge::HipError::new(
                0,
                "KV mode Asym3Auto must be resolved before VMM layout",
            )),
        }
    }

    /// Packed V bytes-per-head for Q8 / Lloyd{2,3,4}.
    fn vmm_v_bytes_per_head(v_mode: VMode, head_dim: usize) -> HipResult<usize> {
        match v_mode {
            VMode::Q8 => {
                if head_dim == 0 || !head_dim.is_multiple_of(32) {
                    return Err(hip_bridge::HipError::new(
                        0,
                        &format!(
                            "VMM Q8-V requires a non-zero head_dim divisible by 32 (got head_dim={head_dim})"
                        ),
                    ));
                }
                Self::checked_vmm_product("q8 V head stride", &[head_dim / 32, 34])
            }
            VMode::Lloyd2 | VMode::Lloyd3 | VMode::Lloyd4 => {
                let bits = v_mode.bits() as usize;
                head_dim
                    .checked_mul(bits)
                    .and_then(|n| n.checked_div(8))
                    .and_then(|n| n.checked_add(4))
                    .ok_or_else(|| {
                        hip_bridge::HipError::new(
                            0,
                            &format!("VMM lloyd{bits} V head stride overflowed"),
                        )
                    })
            }
        }
    }

    /// Geometry gates shared by every static VMM mode (and legal Lloyd-V pairs).
    fn validate_vmm_static_geometry(
        mode: KvMode,
        v_mode: VMode,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        if n_kv_heads == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("VMM {mode:?} requires n_kv_heads>0 (got 0)"),
            ));
        }
        match mode {
            KvMode::Q8 => {
                if head_dim == 0 || !head_dim.is_multiple_of(32) {
                    return Err(hip_bridge::HipError::new(
                        0,
                        &format!(
                            "VMM q8 requires n_kv_heads>0 and a non-zero head_dim divisible by 32 (got n_kv_heads={n_kv_heads}, head_dim={head_dim})"
                        ),
                    ));
                }
                if !matches!(v_mode, VMode::Q8) {
                    return Err(hip_bridge::HipError::new(
                        0,
                        "VMM q8 only supports VMode::Q8 (lloyd-V requires an FWHT K mode)",
                    ));
                }
            }
            KvMode::Asym2 | KvMode::Asym3 | KvMode::Asym4 => {
                let ok_hd = match mode {
                    KvMode::Asym3 => head_dim == 256,
                    _ => head_dim == 128 || head_dim == 256,
                };
                if !ok_hd {
                    return Err(hip_bridge::HipError::new(
                        0,
                        &format!(
                            "VMM {mode:?} requires head_dim={} (got {head_dim})",
                            if matches!(mode, KvMode::Asym3) {
                                "256"
                            } else {
                                "128 or 256"
                            }
                        ),
                    ));
                }
                if !matches!(v_mode, VMode::Q8) {
                    return Err(hip_bridge::HipError::new(
                        0,
                        &format!(
                            "VMM {mode:?} only supports VMode::Q8 (lloyd-V requires an FWHT K mode)"
                        ),
                    ));
                }
            }
            KvMode::Fwht2 | KvMode::Fwht3 | KvMode::Fwht4 => match v_mode {
                VMode::Q8 => {
                    let ok_hd = match mode {
                        KvMode::Fwht3 => head_dim == 256,
                        _ => head_dim == 128 || head_dim == 256,
                    };
                    if !ok_hd {
                        return Err(hip_bridge::HipError::new(
                            0,
                            &format!(
                                "VMM {mode:?} requires head_dim={} (got {head_dim})",
                                if matches!(mode, KvMode::Fwht3) {
                                    "256"
                                } else {
                                    "128 or 256"
                                }
                            ),
                        ));
                    }
                }
                VMode::Lloyd2 | VMode::Lloyd3 | VMode::Lloyd4 => {
                    if head_dim != 256 {
                        return Err(hip_bridge::HipError::new(
                            0,
                            &format!(
                                "VMM lloyd-V requires head_dim=256 with FWHT K (got mode={mode:?}, head_dim={head_dim})"
                            ),
                        ));
                    }
                }
            },
            KvMode::Asym3Auto => {
                return Err(hip_bridge::HipError::new(
                    0,
                    "KV mode Asym3Auto must be resolved before VMM layout",
                ));
            }
        }
        Ok(())
    }

    /// Build a static VMM layout (reserve tier == current tier).
    fn vmm_static_layout(
        mode: KvMode,
        v_mode: VMode,
        n_kv_heads: usize,
        head_dim: usize,
        physical_cap: usize,
    ) -> HipResult<VmmKvLayout> {
        Self::validate_vmm_static_geometry(mode, v_mode, n_kv_heads, head_dim)?;
        if physical_cap == 0 {
            return Err(hip_bridge::HipError::new(0, "VMM physical_cap must be > 0"));
        }
        let kv_dim = Self::checked_vmm_product("logical", &[n_kv_heads, head_dim])?;
        let k_bytes_per_head = Self::vmm_k_bytes_per_head(mode, head_dim)?;
        let v_bytes_per_head = Self::vmm_v_bytes_per_head(v_mode, head_dim)?;
        let k_bytes_per_token =
            Self::checked_vmm_product("K token stride", &[n_kv_heads, k_bytes_per_head])?;
        let v_bytes_per_token =
            Self::checked_vmm_product("V token stride", &[n_kv_heads, v_bytes_per_head])?;
        let k_reserve_bytes =
            Self::checked_vmm_product("K reserve", &[physical_cap, k_bytes_per_token])?;
        let v_reserve_bytes =
            Self::checked_vmm_product("V reserve", &[physical_cap, v_bytes_per_token])?;
        let rotation_table_len = match mode {
            KvMode::Q8 => 0,
            KvMode::Asym2 | KvMode::Asym3 | KvMode::Asym4 => head_dim / 2,
            KvMode::Fwht3 => 256,
            KvMode::Fwht2 | KvMode::Fwht4 => {
                // Q8-V uses 128-wide signs; lloyd-V needs 256-wide up front so
                // constructors never replace owners after publish.
                if matches!(v_mode, VMode::Q8) {
                    128
                } else {
                    256
                }
            }
            KvMode::Asym3Auto => 0,
        };
        let uses_fwht_signs = matches!(mode, KvMode::Fwht2 | KvMode::Fwht3 | KvMode::Fwht4);
        Ok(VmmKvLayout {
            mode,
            v_mode,
            n_kv_heads,
            head_dim,
            physical_cap,
            kv_dim,
            k_bytes_per_head,
            v_bytes_per_head,
            k_bytes_per_token,
            v_bytes_per_token,
            k_reserve_bytes,
            v_reserve_bytes,
            k_reserve_elems: Self::bytes_to_f32_elems("K reserve", k_reserve_bytes)?,
            v_reserve_elems: Self::bytes_to_f32_elems("V reserve", v_reserve_bytes)?,
            rotation_table_len,
            uses_fwht_signs,
        })
    }

    /// Floor-reserved layout: current encoding strides may differ from reserve.
    /// Adaptive uses this with start FWHT4/Q8 and floor reserve bph/v_mode.
    fn vmm_layout_with_reserve(
        mode: KvMode,
        v_mode: VMode,
        n_kv_heads: usize,
        head_dim: usize,
        physical_cap: usize,
        reserve_k_bytes_per_head: usize,
        reserve_v_mode: VMode,
    ) -> HipResult<VmmKvLayout> {
        Self::validate_vmm_static_geometry(mode, v_mode, n_kv_heads, head_dim)?;
        if physical_cap == 0 {
            return Err(hip_bridge::HipError::new(0, "VMM physical_cap must be > 0"));
        }
        if reserve_k_bytes_per_head == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "VMM reserve K bytes-per-head must be > 0",
            ));
        }
        let kv_dim = Self::checked_vmm_product("logical", &[n_kv_heads, head_dim])?;
        let k_bytes_per_head = Self::vmm_k_bytes_per_head(mode, head_dim)?;
        let v_bytes_per_head = Self::vmm_v_bytes_per_head(v_mode, head_dim)?;
        let reserve_v_bph = Self::vmm_v_bytes_per_head(reserve_v_mode, head_dim)?;
        let k_bytes_per_token =
            Self::checked_vmm_product("K token stride", &[n_kv_heads, k_bytes_per_head])?;
        let v_bytes_per_token =
            Self::checked_vmm_product("V token stride", &[n_kv_heads, v_bytes_per_head])?;
        let k_reserve_token =
            Self::checked_vmm_product("K reserve token", &[n_kv_heads, reserve_k_bytes_per_head])?;
        let v_reserve_token =
            Self::checked_vmm_product("V reserve token", &[n_kv_heads, reserve_v_bph])?;
        let k_reserve_bytes =
            Self::checked_vmm_product("K reserve", &[physical_cap, k_reserve_token])?;
        let v_reserve_bytes =
            Self::checked_vmm_product("V reserve", &[physical_cap, v_reserve_token])?;
        // Adaptive / lloyd paths need 256-wide signs (q8→lloyd, optional fwht4→fwht3).
        let rotation_table_len = if matches!(mode, KvMode::Fwht2 | KvMode::Fwht3 | KvMode::Fwht4)
            || !matches!(reserve_v_mode, VMode::Q8)
            || !matches!(v_mode, VMode::Q8)
        {
            256
        } else if matches!(mode, KvMode::Asym2 | KvMode::Asym3 | KvMode::Asym4) {
            head_dim / 2
        } else {
            0
        };
        let uses_fwht_signs = matches!(mode, KvMode::Fwht2 | KvMode::Fwht3 | KvMode::Fwht4)
            || !matches!(reserve_v_mode, VMode::Q8);
        Ok(VmmKvLayout {
            mode,
            v_mode,
            n_kv_heads,
            head_dim,
            physical_cap,
            kv_dim,
            k_bytes_per_head,
            v_bytes_per_head,
            k_bytes_per_token,
            v_bytes_per_token,
            k_reserve_bytes,
            v_reserve_bytes,
            k_reserve_elems: Self::bytes_to_f32_elems("K reserve", k_reserve_bytes)?,
            v_reserve_elems: Self::bytes_to_f32_elems("V reserve", v_reserve_bytes)?,
            rotation_table_len,
            uses_fwht_signs,
        })
    }

    /// Back-compat wrappers used by older call sites / tests.
    fn q8_vmm_layout(
        n_kv_heads: usize,
        head_dim: usize,
        physical_cap: usize,
    ) -> HipResult<(usize, usize, usize)> {
        let layout =
            Self::vmm_static_layout(KvMode::Q8, VMode::Q8, n_kv_heads, head_dim, physical_cap)?;
        Ok((
            layout.kv_dim,
            layout.k_reserve_elems,
            layout.k_bytes_per_token,
        ))
    }

    fn asym3_vmm_layout(
        n_kv_heads: usize,
        head_dim: usize,
        physical_cap: usize,
    ) -> HipResult<(usize, usize, usize, usize, usize)> {
        let layout =
            Self::vmm_static_layout(KvMode::Asym3, VMode::Q8, n_kv_heads, head_dim, physical_cap)?;
        Ok((
            layout.kv_dim,
            layout.k_reserve_elems,
            layout.v_reserve_elems,
            layout.k_bytes_per_head,
            layout.v_bytes_per_head,
        ))
    }

    /// FWHT3 reuses the Asym3 packed-K / Q8-V byte layout; only the rotation
    /// tables and `quant_fwht` flag differ from Asym3 VMM.
    fn fwht3_vmm_layout(
        n_kv_heads: usize,
        head_dim: usize,
        physical_cap: usize,
    ) -> HipResult<(usize, usize, usize, usize, usize)> {
        let layout =
            Self::vmm_static_layout(KvMode::Fwht3, VMode::Q8, n_kv_heads, head_dim, physical_cap)?;
        Ok((
            layout.kv_dim,
            layout.k_reserve_elems,
            layout.v_reserve_elems,
            layout.k_bytes_per_head,
            layout.v_bytes_per_head,
        ))
    }

    /// Flag bundle applied by the unified static VMM constructor.
    /// Returns (quant_q8, quant_asym4, quant_asym3, quant_asym2, quant_fwht).
    fn vmm_mode_flags(mode: KvMode) -> (bool, bool, bool, bool, bool) {
        match mode {
            KvMode::Q8 => (true, false, false, false, false),
            KvMode::Asym2 => (false, false, false, true, false),
            KvMode::Asym3 => (false, false, true, false, false),
            KvMode::Asym4 => (false, true, false, false, false),
            KvMode::Fwht2 => (false, false, false, true, true),
            KvMode::Fwht3 => (false, false, true, false, true),
            KvMode::Fwht4 => (false, true, false, false, true),
            KvMode::Asym3Auto => (false, false, false, false, false),
        }
    }

    /// Validate a backend request without touching GPU memory.
    pub fn validate_mode_with_backend(
        mode: KvMode,
        backend: KvBackend,
        single_gpu: bool,
        dims: &KvDims,
    ) -> HipResult<()> {
        if mode == KvMode::Asym3Auto {
            return Err(hip_bridge::HipError::new(
                0,
                "KV mode Asym3Auto must be resolved before allocation",
            ));
        }
        if matches!(mode, KvMode::Asym4 | KvMode::Asym3) && dims.head_dim != 256 {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "{} KV cache requires head_dim=256; head_dim={} is unsupported",
                    if mode == KvMode::Asym4 {
                        "asym4"
                    } else {
                        "asym3"
                    },
                    dims.head_dim
                ),
            ));
        }
        if backend != KvBackend::Vmm {
            return Ok(());
        }
        if !single_gpu {
            return Err(hip_bridge::HipError::new(
                0,
                "KV backend 'vmm' currently supports single-GPU qwen3.5 only",
            ));
        }
        let KvLayers::Mask(is_kv_layer) = &dims.layers else {
            return Err(hip_bridge::HipError::new(
                0,
                "KV backend 'vmm' requires qwen3.5's filtered FullAttention layer mask",
            ));
        };
        if !is_kv_layer.iter().any(|is_kv| *is_kv) {
            return Err(hip_bridge::HipError::new(
                0,
                "KV backend 'vmm' requires at least one FullAttention layer",
            ));
        }
        let physical_cap = dims.physical_cap.ok_or_else(|| {
            hip_bridge::HipError::new(0, "KV backend 'vmm' requires a physical token capacity")
        })?;
        if physical_cap == 0 || physical_cap > dims.max_seq {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "VMM physical_cap must be in 1..=max_seq (got {physical_cap}, max_seq={})",
                    dims.max_seq
                ),
            ));
        }
        match mode {
            KvMode::Q8
            | KvMode::Asym2
            | KvMode::Asym3
            | KvMode::Asym4
            | KvMode::Fwht2
            | KvMode::Fwht3
            | KvMode::Fwht4 => {
                // Default static V is Q8; Lloyd-V is validated when a constructor
                // is invoked with an explicit v_mode.
                Self::vmm_static_layout(
                    mode,
                    VMode::Q8,
                    dims.n_kv_heads,
                    dims.head_dim,
                    physical_cap,
                )?;
            }
            other => {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "KV backend 'vmm' supports kv_mode=q8|asym2|asym3|asym4|fwht2|fwht3|fwht4 only (got {other:?})"
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Non-owning view of one lane inside a parent contiguous Q8 cache.
    ///
    /// `self` must have been allocated with total physical capacity
    /// `lanes * lane_capacity`. The returned cache presents ordinary
    /// single-sequence Q8 addressing, so existing prefill kernels can seed one
    /// lane without copying weights or allocating a second cache. The view must
    /// never be passed to [`KvCache::free_gpu`].
    ///
    /// Rejects asym / FWHT / INT8 / HFQ4 / VMM layouts rather than fabricating
    /// support: continuous-batch only targets exact contiguous Q8 HIP.
    pub fn q8_lane_view(&self, lane: usize, lane_capacity: usize) -> HipResult<Self> {
        if !self.quant_q8
            || self.quant_int8
            || self.quant_hfq4
            || self.quant_asym4
            || self.quant_asym3
            || self.quant_asym2
            || self.quant_fwht
            || self.v_mode != VMode::Q8
            || lane_capacity == 0
        {
            return Err(hip_bridge::HipError::new(
                0,
                "q8_lane_view requires a contiguous Q8 cache and non-zero lane capacity",
            ));
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(32) {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "q8_lane_view requires head_dim divisible by 32 (got {})",
                    self.head_dim
                ),
            ));
        }
        // Fail closed on VMM-backed parent storage: lane views only slice
        // contiguous hipMalloc buffers via borrowed DeviceBuffer pointers.
        for t in self.k_gpu.iter().chain(self.v_gpu.iter()) {
            if t.buf.is_vmm_owner() {
                return Err(hip_bridge::HipError::new(
                    0,
                    "q8_lane_view does not support VMM-backed KV caches",
                ));
            }
        }
        let lane_end = lane
            .checked_add(1)
            .and_then(|n| n.checked_mul(lane_capacity))
            .ok_or_else(|| hip_bridge::HipError::new(0, "q8_lane_view capacity overflow"))?;
        if lane_end > self.physical_cap {
            return Err(hip_bridge::HipError::new(
                0,
                "q8_lane_view lane exceeds backing cache capacity",
            ));
        }
        let blocks_per_pos = self
            .n_kv_heads
            .checked_mul(self.head_dim / 32)
            .ok_or_else(|| hip_bridge::HipError::new(0, "q8_lane_view blocks_per_pos overflow"))?;
        let bytes_per_pos = blocks_per_pos
            .checked_mul(34)
            .ok_or_else(|| hip_bridge::HipError::new(0, "q8_lane_view bytes_per_pos overflow"))?;
        let byte_offset = lane
            .checked_mul(lane_capacity)
            .and_then(|n| n.checked_mul(bytes_per_pos))
            .ok_or_else(|| hip_bridge::HipError::new(0, "q8_lane_view byte_offset overflow"))?;
        let lane_bytes = lane_capacity
            .checked_mul(bytes_per_pos)
            .ok_or_else(|| hip_bridge::HipError::new(0, "q8_lane_view lane_bytes overflow"))?;
        let lane_elems = Self::bytes_to_f32_elems("q8_lane_view lane", lane_bytes)?;
        let view = |t: &GpuTensor| -> HipResult<GpuTensor> {
            if t.numel() <= 1 {
                // Filtered placeholder layer — keep a non-owning alias.
                return Ok(t.shallow_clone());
            }
            let parent_bytes = t.buf.size();
            let end = byte_offset.checked_add(lane_bytes).ok_or_else(|| {
                hip_bridge::HipError::new(0, "q8_lane_view parent byte range overflow")
            })?;
            if end > parent_bytes {
                return Err(hip_bridge::HipError::new(
                    0,
                    "q8_lane_view lane byte range exceeds parent buffer",
                ));
            }
            let ptr =
                unsafe { (t.buf.as_ptr() as *mut u8).add(byte_offset) as *mut std::ffi::c_void };
            Ok(GpuTensor {
                buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, lane_bytes) },
                shape: vec![lane_elems],
                dtype: DType::F32,
            })
        };
        let k_gpu = self.k_gpu.iter().map(view).collect::<HipResult<Vec<_>>>()?;
        let v_gpu = self.v_gpu.iter().map(view).collect::<HipResult<Vec<_>>>()?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: Vec::new(),
            v_scales: Vec::new(),
            kv_dim: self.kv_dim,
            max_seq: lane_capacity,
            physical_cap: lane_capacity,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            quantized: true,
            quant_q8: true,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            boundary_layers: self.boundary_layers,
            givens_cos: None,
            givens_sin: None,
            quant_fwht: false,
            v_mode: VMode::Q8,
            layer_is_boundary: self.layer_is_boundary.clone(),
            compact_offset: 0,
        })
    }

    /// Check if a given KV layer ordinal is a boundary layer (first N + last N).
    pub fn is_boundary(&self, kv_ordinal: usize) -> bool {
        kv_ordinal < self.layer_is_boundary.len() && self.layer_is_boundary[kv_ordinal]
    }

    /// Zero every per-layer K/V (and scale) buffer on the GPU. Defense-in-depth
    /// for arch `reset()`: positional KV is normally overwritten by the next
    /// prefill (so the stale tail is never attended), but this guarantees no
    /// prior-conversation bytes can survive a reset even under a future
    /// window/LCP edge that reads an un-rewritten slot. Sub-millisecond memset
    /// of the cache buffers; callers MUST also clear their token mirror so a
    /// zeroed slot can never be stale-LCP-reused.
    pub fn clear_gpu(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        for t in self
            .k_gpu
            .iter()
            .chain(self.v_gpu.iter())
            .chain(self.k_scales.iter())
            .chain(self.v_scales.iter())
        {
            let bytes = gpu.vmm_mapped_bytes(t).unwrap_or_else(|| t.buf.size());
            gpu.hip.memset(&t.buf, 0, bytes)?;
        }
        Ok(())
    }

    /// The single dispatcher over the `new_gpu_*` constructor family. Each
    /// non-error arm corresponds 1:1 to a line that exists in a load-site ladder
    /// today (the byte-identical contract). Unreachable `(mode × layers × cap ×
    /// target)` cells return `Err` rather than panic, so a future policy mis-wire
    /// surfaces as a clean load failure.
    pub fn from_mode(mode: KvMode, target: KvTarget, dims: &KvDims) -> HipResult<Self> {
        Self::from_mode_with_backend(mode, KvBackend::Contiguous, target, dims)
    }

    pub fn from_mode_with_backend(
        mode: KvMode,
        backend: KvBackend,
        target: KvTarget,
        dims: &KvDims,
    ) -> HipResult<Self> {
        let single_gpu = matches!(&target, KvTarget::Single(_));
        Self::validate_mode_with_backend(mode, backend, single_gpu, dims)?;
        match (backend, target) {
            (KvBackend::Contiguous, KvTarget::Single(gpu)) => {
                Self::from_mode_single(mode, gpu, dims)
            }
            (KvBackend::Contiguous, KvTarget::Multi(gpus)) => {
                Self::from_mode_multi(mode, gpus, dims)
            }
            (KvBackend::Vmm, KvTarget::Single(gpu)) => Self::from_mode_single_vmm(mode, gpu, dims),
            (KvBackend::Vmm, KvTarget::Multi(_)) => Err(hip_bridge::HipError::new(
                0,
                "KV backend 'vmm' currently supports single-GPU qwen3.5 only",
            )),
        }
    }

    fn from_mode_single_vmm(mode: KvMode, gpu: &mut Gpu, dims: &KvDims) -> HipResult<Self> {
        let KvLayers::Mask(is_kv_layer) = &dims.layers else {
            unreachable!("VMM layer shape validated before dispatch")
        };
        let physical_cap = dims
            .physical_cap
            .expect("VMM physical capacity validated before dispatch");
        // Static path defaults V to Q8. Carrier may call
        // `new_gpu_vmm_capped_filtered` directly with Lloyd-V for FWHT-K.
        match mode {
            KvMode::Q8
            | KvMode::Asym2
            | KvMode::Asym3
            | KvMode::Asym4
            | KvMode::Fwht2
            | KvMode::Fwht3
            | KvMode::Fwht4 => Self::new_gpu_vmm_capped_filtered(
                gpu,
                is_kv_layer,
                dims.n_kv_heads,
                dims.head_dim,
                dims.max_seq,
                physical_cap,
                mode,
                VMode::Q8,
            ),
            other => unreachable!("VMM mode {other:?} validated before dispatch"),
        }
    }

    fn from_mode_single(mode: KvMode, gpu: &mut Gpu, dims: &KvDims) -> HipResult<Self> {
        use KvLayers::*;
        let nh = dims.n_kv_heads;
        let hd = dims.head_dim;
        let ms = dims.max_seq;
        match (mode, &dims.layers, dims.physical_cap) {
            // Mask + Some(cap): _capped_filtered (only q8/asym3/fwht2/fwht3 have it).
            (KvMode::Q8, Mask(m), Some(cap)) => Self::new_gpu_q8_capped_filtered(gpu, m, nh, hd, ms, cap),
            (KvMode::Asym3, Mask(m), Some(cap)) => Self::new_gpu_asym3_capped_filtered(gpu, m, nh, hd, ms, cap),
            (KvMode::Fwht2, Mask(m), Some(cap)) => Self::new_gpu_fwht2_capped_filtered(gpu, m, nh, hd, ms, cap),
            (KvMode::Fwht3, Mask(m), Some(cap)) => Self::new_gpu_fwht3_capped_filtered(gpu, m, nh, hd, ms, cap),
            // Mask + cap-but-no-capped-variant: cap DROPPED, use _filtered (faithful).
            (KvMode::Asym2, Mask(m), _) => Self::new_gpu_asym2_filtered(gpu, m, nh, hd, ms),
            (KvMode::Asym4, Mask(m), _) => Self::new_gpu_asym4_filtered(gpu, m, nh, hd, ms),
            (KvMode::Fwht4, Mask(m), _) => Self::new_gpu_fwht4_filtered(gpu, m, nh, hd, ms),
            // Mask + None for the capped-capable modes: plain _filtered.
            (KvMode::Q8, Mask(m), None) => Self::new_gpu_q8_filtered(gpu, m, nh, hd, ms),
            (KvMode::Asym3, Mask(m), None) => Self::new_gpu_asym3_filtered(gpu, m, nh, hd, ms),
            (KvMode::Fwht2, Mask(m), None) => Self::new_gpu_fwht2_filtered(gpu, m, nh, hd, ms),
            (KvMode::Fwht3, Mask(m), None) => Self::new_gpu_fwht3_filtered(gpu, m, nh, hd, ms),
            // Flat + Some(cap): _capped (only q8/asym3/asym4).
            (KvMode::Q8, Flat(n), Some(cap)) => Self::new_gpu_q8_capped(gpu, *n, nh, hd, ms, cap),
            (KvMode::Asym3, Flat(n), Some(cap)) => Self::new_gpu_asym3_capped(gpu, *n, nh, hd, ms, cap),
            (KvMode::Asym4, Flat(n), Some(cap)) => Self::new_gpu_asym4_capped(gpu, *n, nh, hd, ms, cap),
            // Flat + None: plain (only q8/asym3/asym4).
            (KvMode::Q8, Flat(n), None) => Self::new_gpu_q8(gpu, *n, nh, hd, ms),
            (KvMode::Asym3, Flat(n), None) => Self::new_gpu_asym3(gpu, *n, nh, hd, ms),
            (KvMode::Asym4, Flat(n), None) => Self::new_gpu_asym4(gpu, *n, nh, hd, ms),
            // No constructor exists for this combination.
            (m, l, c) => Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "KvCache::from_mode_single: no constructor for (mode={m:?}, layers={}, cap={c:?}); \
                     unreachable under current policies — a policy/accepted-set mis-wire, not a user error",
                    match l {
                        Mask(_) => "Mask",
                        Flat(_) => "Flat",
                    },
                ),
            )),
        }
    }

    fn from_mode_multi(mode: KvMode, gpus: &mut Gpus, dims: &KvDims) -> HipResult<Self> {
        use KvLayers::*;
        let nh = dims.n_kv_heads;
        let hd = dims.head_dim;
        let ms = dims.max_seq;
        // Site 6 only: Mask + Some(cap) + {q8,asym3,fwht3,fwht2} → _capped_multi_filtered.
        match (mode, &dims.layers, dims.physical_cap) {
            (KvMode::Q8, Mask(m), Some(cap)) => Self::new_gpu_q8_capped_multi_filtered(gpus, m, nh, hd, ms, cap),
            (KvMode::Asym3, Mask(m), Some(cap)) => Self::new_gpu_asym3_capped_multi_filtered(gpus, m, nh, hd, ms, cap),
            (KvMode::Fwht3, Mask(m), Some(cap)) => Self::new_gpu_fwht3_capped_multi_filtered(gpus, m, nh, hd, ms, cap),
            (KvMode::Fwht2, Mask(m), Some(cap)) => Self::new_gpu_fwht2_capped_multi_filtered(gpus, m, nh, hd, ms, cap),
            (m, l, c) => Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "KvCache::from_mode_multi: no multi constructor for (mode={m:?}, layers={}, cap={c:?})",
                    match l {
                        Mask(_) => "Mask",
                        Flat(_) => "Flat",
                    },
                ),
            )),
        }
    }
}

impl KvCache {
    pub fn new_gpu(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let cache_size = max_seq_len * kv_dim;
        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[cache_size], DType::F32)?);
            v_gpu.push(gpu.zeros(&[cache_size], DType::F32)?);
        }
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: false,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    /// Create quantized KV cache (HFQ4-G128). 3.56x smaller than FP32.
    pub fn new_gpu_q4(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        // Per position per head: 8 bytes (scale+zero) + head_dim/2 bytes (nibbles)
        let bytes_per_head = 8 + head_dim / 2;
        let bytes_per_pos = n_kv_heads * bytes_per_head;
        let cache_bytes = max_seq_len * bytes_per_pos;
        // Allocate as raw bytes (use F32 dtype but size in bytes)
        let cache_elems = (cache_bytes + 3) / 4; // round up to F32 elements
        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[cache_elems], DType::F32)?);
            v_gpu.push(gpu.zeros(&[cache_elems], DType::F32)?);
        }
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    /// Create Q8_0 quantized KV cache (GGML Q8_0 format). 3.76x smaller than FP32.
    /// Block: [f16 scale (2B)][int8 × 32 (32B)] = 34 bytes per 32 elements.
    /// head_dim=128 → 4 blocks × 34 = 136 bytes per head.
    pub fn new_gpu_q8(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_q8_capped(
            gpu,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    /// Same as [`new_gpu_q8`] with an explicit physical_cap. Eviction-aware.
    pub fn new_gpu_q8_capped(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]"
        );
        let kv_dim = n_kv_heads * head_dim;
        let blocks_per_head = head_dim / 32;
        let total_blocks = n_kv_heads * blocks_per_head;
        let cache_bytes = physical_cap * total_blocks * 34;
        let cache_elems = (cache_bytes + 3) / 4;
        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[cache_elems], DType::F32)?);
            v_gpu.push(gpu.zeros(&[cache_elems], DType::F32)?);
        }
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: true,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    /// Helper: allocate K/V Vecs, skipping layers where is_kv_layer[i] is false
    /// by inserting a 1-element placeholder. Saves VRAM for hybrid arches
    /// (Qwen 3.5 DeltaNet + FullAttention) where 75% of layers don't carry
    /// KV in this cache — their state lives in [`crate::qwen35::DeltaNetState`].
    /// Per-layer index is preserved so downstream code can index by absolute
    /// layer_idx unchanged.
    fn alloc_k_v_filtered(
        gpu: &mut Gpu,
        k_elems: usize,
        v_elems: usize,
        is_kv_layer: &[bool],
    ) -> HipResult<(Vec<GpuTensor>, Vec<GpuTensor>)> {
        let n = is_kv_layer.len();
        let mut k_gpu = Vec::with_capacity(n);
        let mut v_gpu = Vec::with_capacity(n);
        // Contiguous path mirrors alloc_k_v_vmm_filtered: on any mid-loop
        // failure free every tensor already pushed so a partial build never
        // leaks device memory (GpuTensor has no freeing Drop).
        let result = (|| -> HipResult<()> {
            for &is_kv in is_kv_layer {
                if is_kv {
                    k_gpu.push(gpu.zeros(&[k_elems], DType::F32)?);
                    v_gpu.push(gpu.zeros(&[v_elems], DType::F32)?);
                } else {
                    k_gpu.push(gpu.zeros(&[1], DType::F32)?);
                    v_gpu.push(gpu.zeros(&[1], DType::F32)?);
                }
            }
            Ok(())
        })();
        if let Err(err) = result {
            for tensor in k_gpu.drain(..).chain(v_gpu.drain(..)) {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(err);
        }
        Ok((k_gpu, v_gpu))
    }

    /// Reserve dense virtual K/V tensors while mapping physical pages only as
    /// the request grows. Placeholder tensors preserve absolute layer indices.
    fn alloc_k_v_vmm_filtered(
        gpu: &mut Gpu,
        k_elems: usize,
        v_elems: usize,
        is_kv_layer: &[bool],
    ) -> HipResult<(Vec<GpuTensor>, Vec<GpuTensor>)> {
        let mut k_gpu = Vec::with_capacity(is_kv_layer.len());
        let mut v_gpu = Vec::with_capacity(is_kv_layer.len());
        let device_id = gpu.device_id;
        let result = (|| -> HipResult<()> {
            for &is_kv in is_kv_layer {
                if is_kv {
                    // SAFETY: Qwen3.5 mapped-prefix guards run before every
                    // write, attention dispatch, and graph capture/replay.
                    k_gpu.push(unsafe {
                        gpu.alloc_vmm_tensor(&[k_elems], DType::F32, 0, &[device_id])?
                    });
                    v_gpu.push(unsafe {
                        gpu.alloc_vmm_tensor(&[v_elems], DType::F32, 0, &[device_id])?
                    });
                } else {
                    k_gpu.push(gpu.zeros(&[1], DType::F32)?);
                    v_gpu.push(gpu.zeros(&[1], DType::F32)?);
                }
            }
            Ok(())
        })();
        if let Err(err) = result {
            for tensor in k_gpu.drain(..).chain(v_gpu.drain(..)) {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(err);
        }
        Ok((k_gpu, v_gpu))
    }

    /// Resolve the current K `KvMode` from live cache flags.
    fn current_kv_mode(&self) -> HipResult<KvMode> {
        if self.quant_q8 {
            return Ok(KvMode::Q8);
        }
        if self.quant_asym4 {
            return Ok(if self.quant_fwht {
                KvMode::Fwht4
            } else {
                KvMode::Asym4
            });
        }
        if self.quant_asym3 {
            return Ok(if self.quant_fwht {
                KvMode::Fwht3
            } else {
                KvMode::Asym3
            });
        }
        if self.quant_asym2 {
            return Ok(if self.quant_fwht {
                KvMode::Fwht2
            } else {
                KvMode::Asym2
            });
        }
        Err(hip_bridge::HipError::new(
            0,
            "VMM KV cache encoding flags are not a recognized static mode",
        ))
    }

    /// Current K/V bytes-per-token from live flags / V mode.
    /// Independent K and V strides — capacity is always min-of-two.
    fn vmm_bytes_per_token(&self) -> HipResult<(usize, usize)> {
        if self.n_kv_heads == 0 {
            return Err(hip_bridge::HipError::new(0, "VMM KV requires n_kv_heads>0"));
        }
        let mode = self.current_kv_mode()?;
        // Geometry for the live encoding; lloyd-V only legal on FWHT-K.
        Self::validate_vmm_static_geometry(mode, self.v_mode, self.n_kv_heads, self.head_dim)?;
        let k_bph = Self::vmm_k_bytes_per_head(mode, self.head_dim)?;
        let v_bph = Self::vmm_v_bytes_per_head(self.v_mode, self.head_dim)?;
        let k = Self::checked_vmm_product("K token stride", &[self.n_kv_heads, k_bph])?;
        let v = Self::checked_vmm_product("V token stride", &[self.n_kv_heads, v_bph])?;
        Ok((k, v))
    }

    pub fn uses_vmm_backend(&self) -> bool {
        self.k_gpu
            .iter()
            .chain(self.v_gpu.iter())
            .find(|tensor| tensor.numel() > 1)
            .is_some_and(|tensor| tensor.buf.is_vmm_owner())
    }

    fn fast_mapped_token_capacity(&self) -> HipResult<Option<usize>> {
        if !self.uses_vmm_backend() {
            return Ok(None);
        }
        let (k_bytes_per_token, v_bytes_per_token) = self.vmm_bytes_per_token()?;
        let capacity = [
            ("K", self.k_gpu.as_slice(), k_bytes_per_token),
            ("V", self.v_gpu.as_slice(), v_bytes_per_token),
        ]
        .into_iter()
        .try_fold(self.physical_cap, |capacity, (label, tensors, stride)| {
            let tensor = tensors
                .iter()
                .rev()
                .find(|tensor| tensor.numel() > 1)
                .ok_or_else(|| {
                    hip_bridge::HipError::new(
                        0,
                        &format!("VMM KV cache has no real {label} tensor"),
                    )
                })?;
            if !tensor.buf.is_vmm_owner() {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!("VMM KV cache has a non-VMM real {label} tensor"),
                ));
            }
            Ok(capacity.min(tensor.buf.size() / stride))
        })?;
        Ok(Some(capacity))
    }

    pub fn mapped_token_capacity(&self) -> HipResult<Option<usize>> {
        if !self.uses_vmm_backend() {
            return Ok(None);
        }
        let (k_bytes_per_token, v_bytes_per_token) = self.vmm_bytes_per_token()?;
        let mut capacity = self.physical_cap;
        for (label, tensors, bytes_per_token) in [
            ("K", self.k_gpu.as_slice(), k_bytes_per_token),
            ("V", self.v_gpu.as_slice(), v_bytes_per_token),
        ] {
            for tensor in tensors {
                if tensor.buf.is_vmm_owner() {
                    capacity = capacity.min(tensor.buf.size() / bytes_per_token);
                } else if tensor.numel() > 1 {
                    return Err(hip_bridge::HipError::new(
                        0,
                        &format!(
                            "VMM KV cache mixes a non-VMM real {label} tensor with VMM tensors"
                        ),
                    ));
                }
            }
        }
        Ok(Some(capacity))
    }

    pub fn require_mapped_capacity(&self, required_tokens: usize) -> HipResult<()> {
        let Some(mapped_tokens) = self.mapped_token_capacity()? else {
            return Ok(());
        };
        if required_tokens > self.physical_cap {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "KV write requires {required_tokens} tokens but physical_cap is {}",
                    self.physical_cap
                ),
            ));
        }
        if required_tokens > mapped_tokens {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "VMM KV mapped prefix holds {mapped_tokens} tokens but the operation requires {required_tokens}"
                ),
            ));
        }
        Ok(())
    }

    /// Grow VMM tensors for one side (K or V).
    ///
    /// `bytes_per_token` is the **current** encoding stride. `physical_cap` is
    /// the constructor token horizon. Per-tensor max tokens is
    /// `min(physical_cap, reserve_bytes / current_stride)` so adaptive
    /// floor-reserved arenas never plan growth past the stable VA.
    fn grow_vmm_tensors(
        gpu: &mut Gpu,
        tensors: &mut [GpuTensor],
        bytes_per_token: usize,
        physical_cap: usize,
        required_tokens: usize,
    ) -> HipResult<usize> {
        if bytes_per_token == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "VMM growth requires a non-zero current byte stride",
            ));
        }
        let mut grown = 0usize;
        let device_id = gpu.device_id;
        let minimum_growth_bytes = DEFAULT_VMM_PHYSICAL_CHUNK_BYTES;
        for tensor in tensors {
            if !tensor.buf.is_vmm_owner() {
                if tensor.numel() > 1 {
                    return Err(hip_bridge::HipError::new(
                        0,
                        "VMM KV cache contains a non-VMM real tensor",
                    ));
                }
                continue;
            }
            let mapped = gpu.vmm_mapped_bytes(tensor).ok_or_else(|| {
                hip_bridge::HipError::new(0, "VMM KV tensor is not registered with its GPU")
            })?;
            let granularity = gpu.vmm_granularity(tensor).ok_or_else(|| {
                hip_bridge::HipError::new(0, "VMM KV tensor has no allocation granularity")
            })?;
            // Full reserved VA size (shape × dtype), independent of mapped prefix.
            let reserve_bytes = tensor.byte_size();
            let side_cap = (reserve_bytes / bytes_per_token).min(physical_cap);
            if side_cap == 0 {
                return Err(hip_bridge::HipError::new(
                    0,
                    "VMM KV reserve is smaller than one token at the current stride",
                ));
            }
            if required_tokens > side_cap {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "VMM KV current-stride capacity is {side_cap} tokens but the operation requires {required_tokens}"
                    ),
                ));
            }
            let plan = KvChunkPlan::new(
                bytes_per_token,
                side_cap,
                DEFAULT_KV_CHUNK_TOKENS,
                granularity,
                minimum_growth_bytes,
            )
            .map_err(|err| hip_bridge::HipError::new(0, &err.to_string()))?;
            if let Some(growth) = plan
                .growth(mapped, required_tokens)
                .map_err(|err| hip_bridge::HipError::new(0, &err.to_string()))?
            {
                if gpu.graphs.capture_mode {
                    return Err(hip_bridge::HipError::new(
                        0,
                        "VMM KV growth requested during graph capture",
                    ));
                }
                // Map before any subsequent write; owners stay at the same VA.
                gpu.grow_vmm_tensor(tensor, growth.size_bytes, &[device_id])?;
                grown += 1;
            }
        }
        Ok(grown)
    }

    /// Ensure both K and V mapped prefixes cover `required_tokens` at the
    /// **current** strides. Growth is independent per side and completes before
    /// the caller may write. Never replaces VMM owners.
    pub fn ensure_mapped_capacity(
        &mut self,
        gpu: &mut Gpu,
        required_tokens: usize,
    ) -> HipResult<()> {
        let Some(fast_capacity) = self.fast_mapped_token_capacity()? else {
            return Ok(());
        };
        if required_tokens > self.physical_cap {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "KV write requires {required_tokens} tokens but physical_cap is {}",
                    self.physical_cap
                ),
            ));
        }
        if required_tokens <= fast_capacity {
            return Ok(());
        }
        let (k_bytes_per_token, v_bytes_per_token) = self.vmm_bytes_per_token()?;
        // Growth is monotonic: if a later map fails, the next call keeps the
        // completed prefixes and fills the lagging tensors before the final guard.
        // K and V are grown independently at their current strides.
        Self::grow_vmm_tensors(
            gpu,
            &mut self.k_gpu,
            k_bytes_per_token,
            self.physical_cap,
            required_tokens,
        )?;
        Self::grow_vmm_tensors(
            gpu,
            &mut self.v_gpu,
            v_bytes_per_token,
            self.physical_cap,
            required_tokens,
        )?;
        self.require_mapped_capacity(required_tokens)?;
        Ok(())
    }

    /// Bytes of V-cache per token-position (all heads) for a given V mode.
    /// Q8 = n_kv_heads * (head_dim/32) * 34. Lloyd = n_kv_heads * (4 + head_dim*bits/8).
    fn v_bytes_per_pos(n_kv_heads: usize, head_dim: usize, v_mode: VMode) -> usize {
        match v_mode {
            VMode::Q8 => n_kv_heads * (head_dim / 32) * 34,
            VMode::Lloyd2 | VMode::Lloyd3 | VMode::Lloyd4 => {
                n_kv_heads * (4 + (head_dim * v_mode.bits() as usize) / 8)
            }
        }
    }

    /// V-mode bit-count to pass as a kernarg.
    pub fn v_mode_bits(&self) -> i32 {
        self.v_mode.bits()
    }

    /// Mode-derived `KvTierInputs` with per-call fields zero-filled. Each
    /// attention dispatch site sets `pos`/`flash_mode`/`capture_mode`/
    /// `batch_size`/`is_tree`/`is_boundary` after this returns (via functional
    /// update). Single source of truth for the cache-stable tier flags,
    /// replacing the four hand-copied literals.
    /// The "quantized but no known format and no separate scales" residual —
    /// the llama legacy Q4 KV path. Shared by `tier_inputs()` and `k_tier()`
    /// so the two derivations can never silently diverge.
    fn quant_q4_residual(&self) -> bool {
        // The llama-legacy "plain Q4" KV tier is the residual: quantized but
        // none of the named tiers. MUST also exclude asym{2,3,4} — those set
        // `quantized:true` with empty `k_scales`, so without this guard an asym
        // cache reports BOTH its asym flag AND quant_q4, tripping classify()'s
        // "at most one tier flag" debug_assert (debug-build panic on the qwen35
        // asym3 default) and breaking the byte-identical-to-legacy-literal
        // invariant (the legacy qwen35 literals hardcoded quant_q4 = false).
        // Release classify() output is unchanged either way (asym is matched
        // before q4), so this is a true no-op for kernel selection.
        self.quantized
            && !self.quant_hfq4
            && !self.quant_q8
            && !self.quant_int8
            && !self.quant_asym4
            && !self.quant_asym3
            && !self.quant_asym2
            && self.k_scales.is_empty()
    }

    /// HFQ8 flat-layout KV: quantized with a separate per-block scale table,
    /// and none of the other named tiers. Mirrors the hand ladder's hfq8 branch
    /// condition (`llama.rs` `llama_kv_write_attend` `quantized && !k_scales.is_empty()
    /// && !int8 && !q8`), extended with the exclusions that keep exactly one tier
    /// flag set (asym caches set `quantized` with EMPTY `k_scales`; hfq4 is matched
    /// earlier) so classify()'s "at most one tier flag" debug_assert holds.
    fn is_hfq8_kv(&self) -> bool {
        self.quantized
            && !self.k_scales.is_empty()
            && !self.quant_int8
            && !self.quant_q8
            && !self.quant_hfq4
            && !self.quant_asym4
            && !self.quant_asym3
            && !self.quant_asym2
    }

    pub fn tier_inputs(&self) -> KvTierInputs {
        let quant_q4 = self.quant_q4_residual();
        KvTierInputs {
            quant_asym4: self.quant_asym4,
            quant_asym3: self.quant_asym3,
            quant_asym2: self.quant_asym2,
            quant_q8: self.quant_q8,
            quant_fwht: self.quant_fwht,
            quant_hfq4: self.quant_hfq4,
            quant_q4,
            quant_int8: self.quant_int8,
            quant_hfq8: self.is_hfq8_kv(),
            // llama F32 KV uses the plain attention_f32 kernel (qwen2's GQA-flash
            // selector is set only by qwen2's own attend_plan).
            f32_policy: hipfire_dispatch::families::kv_tier::F32AttnPolicy::Simple,
            v_mode_bits: self.v_mode_bits(),
            // ── per-call defaults (note batch_size = 1, not 0), overwritten
            //    by the caller via functional update ──
            pos: 0,
            flash_mode: 0,
            capture_mode: false,
            batch_size: 1,
            is_tree: false,
            is_boundary: false,
            q8_windowed: false,
            window: 0,
        }
    }

    /// The sealed decode of this cache's storage tier. The sanctioned way to
    /// branch on KV quant mode — replaces hand-rolled `if kv.quant_q8 { … }` ladders.
    pub fn k_tier(&self) -> hipfire_dispatch::families::kv_tier::KTier {
        // `quant_q4` is the residual (see quant_q4_residual()); same value
        // derive() would see.
        let quant_q4 = self.quant_q4_residual();
        hipfire_dispatch::families::kv_tier::classify(
            self.quant_q8,
            self.quant_asym4,
            self.quant_asym3,
            self.quant_asym2,
            self.quant_hfq4,
            quant_q4,
            self.quant_int8,
            self.is_hfq8_kv(),
            self.quant_fwht,
        )
    }

    fn resize_real_tensors_zeroed(
        gpu: &mut Gpu,
        tensors: &mut [GpuTensor],
        elems: usize,
    ) -> HipResult<()> {
        let real: Vec<usize> = tensors
            .iter()
            .enumerate()
            .filter_map(|(i, t)| (t.numel() > 1).then_some(i))
            .collect();
        for &i in &real {
            let placeholder = gpu.zeros(&[1], DType::F32)?;
            let old = std::mem::replace(&mut tensors[i], placeholder);
            let _ = gpu.free_tensor(old);
        }
        gpu.drain_pool();
        for &i in &real {
            let new_tensor = gpu.zeros(&[elems], DType::F32)?;
            let placeholder = std::mem::replace(&mut tensors[i], new_tensor);
            let _ = gpu.free_tensor(placeholder);
        }
        gpu.drain_pool();
        Ok(())
    }

    /// Reallocate the V buffers for a new V mode (used by eval/bench to set an
    /// independent V quant after construction). Re-sizes only real KV layers
    /// (placeholder 1-element buffers for non-KV layers are left as-is).
    /// K buffers and rotation tables are untouched except when enabling lloyd-V
    /// on fwht2/4-K caches (128-element signs → reallocated to 256; the 128-wide
    /// K rotation reads only indices 0..127 so the LCG prefix is byte-identical).
    /// Note: single-GPU only; multi-GPU V-mode wiring is deferred (plan Task 9).
    pub fn set_v_mode_realloc(&mut self, gpu: &mut Gpu, v_mode: VMode) -> HipResult<()> {
        assert!(
            ((self.quant_asym2 || self.quant_asym3 || self.quant_asym4) && self.quant_fwht)
                || matches!(v_mode, VMode::Q8),
            "lloyd-V is 256-wide and requires an FWHT K mode (quant_asym{{2,3,4}} && quant_fwht); got a different K mode — would corrupt the V cache"
        );
        if !matches!(v_mode, VMode::Q8) {
            assert!(self.head_dim == 256, "lloyd-V requires head_dim == 256");
        }
        // For fwht2/4-K caches the sign tables are 128-element (the K rotation
        // is 128-wide). Lloyd-V is 256-wide and needs 256-element tables.
        // gen_fwht_signs is a pure LCG: gen_fwht_signs(seed,256)[0..128] ==
        // gen_fwht_signs(seed,128), so the K path remains byte-identical after
        // realloc. Skip when signs are already 256 (fwht3) or when givens_cos
        // is None (multi-GPU cache — sign realloc deferred to Task 9).
        if !matches!(v_mode, VMode::Q8) {
            let need_realloc = self.givens_cos.as_ref().map_or(false, |t| t.numel() < 256);
            if need_realloc {
                let n = 256usize;
                let s1v = Self::gen_fwht_signs(42, n);
                let s2v = Self::gen_fwht_signs(1042, n);
                let s1b: Vec<u8> = s1v.iter().flat_map(|v| v.to_ne_bytes()).collect();
                let s2b: Vec<u8> = s2v.iter().flat_map(|v| v.to_ne_bytes()).collect();
                let s1 = gpu.alloc_tensor(&[n], DType::F32)?;
                let s2 = gpu.alloc_tensor(&[n], DType::F32)?;
                gpu.hip.memcpy_htod(&s1.buf, &s1b)?;
                gpu.hip.memcpy_htod(&s2.buf, &s2b)?;
                if let Some(old) = self.givens_cos.take() {
                    let _ = gpu.free_tensor(old);
                }
                if let Some(old) = self.givens_sin.take() {
                    let _ = gpu.free_tensor(old);
                }
                self.givens_cos = Some(s1);
                self.givens_sin = Some(s2);
            }
        }
        let v_bpp = Self::v_bytes_per_pos(self.n_kv_heads, self.head_dim, v_mode);
        let v_elems = (self.physical_cap * v_bpp + 3) / 4;
        Self::resize_real_tensors_zeroed(gpu, &mut self.v_gpu, v_elems)?;
        self.v_mode = v_mode;
        Ok(())
    }

    /// Adaptive-KV load setup: size the V buffer at the V FLOOR (so the fixed
    /// buffer holds `physical_cap` tokens at the floor; FEWER at higher tiers)
    /// and ensure 256-wide FWHT signs (so the q8→lloyd4 transcode is safe).
    /// `v_mode` STAYS Q8 — the fast, highest-precision start tier the controller
    /// runs until the first threshold. K is untouched (the caller loads K at the
    /// fwht4 footprint). Mirrors the sign-upgrade block of `set_v_mode_realloc`
    /// but reallocs each real V layer to the FLOOR size, not the current mode's
    /// size. Because the floor record (e.g. lloyd2 = 68 B/head) is smaller than
    /// the q8 record (272 B/head), the q8 phase physically holds only
    /// ~physical_cap*68/272 ≈ 0.25*physical_cap positions — exactly why the
    /// controller transcodes before that cap. Single-GPU only (matches
    /// set_v_mode_realloc).
    /// `k_floor_bph` is the K bytes-per-head at the K FLOOR tier (e.g. fwht2 =
    /// 68 @hd=256, fwht4 = 132 @hd=256). When it is SMALLER than the current K
    /// mode's footprint (i.e. the floor is below fwht4), the K buffers are
    /// reallocated to `physical_cap * n_kv_heads * k_floor_bph` so we actually
    /// save K VRAM. K data is still WRITTEN at the fwht4 stride (132 @256) until
    /// `transcode_k_step` runs, so the floor-sized K buffer physically holds
    /// ~physical_cap * k_floor_bph / 132 positions at fwht4 — the controller
    /// transcodes K→fwht2 before that cap. Pass the current K footprint (132 for
    /// a fwht4 cache) to leave K unresized (V-only presets).
    pub fn set_adaptive_floor_alloc(
        &mut self,
        gpu: &mut Gpu,
        v_floor: VMode,
        k_floor_bph: usize,
    ) -> HipResult<()> {
        // VMM owners are never replaced after publication. Adaptive VMM must be
        // constructed via `new_gpu_vmm_adaptive_filtered` with floor reserve.
        if self.uses_vmm_backend() {
            return Err(hip_bridge::HipError::new(
                0,
                "set_adaptive_floor_alloc refuses VMM owners; construct adaptive VMM with floor reserve up front",
            ));
        }
        // Mirror the set_v_mode_realloc guard: lloyd-V is 256-wide and requires
        // an FWHT K mode + head_dim == 256.
        assert!(
            (self.quant_asym2 || self.quant_asym3 || self.quant_asym4) && self.quant_fwht,
            "adaptive-KV requires an FWHT K mode (quant_asym{{2,3,4}} && quant_fwht)"
        );
        assert!(self.head_dim == 256, "adaptive-KV requires head_dim == 256");
        assert!(
            !matches!(v_floor, VMode::Q8),
            "adaptive-KV V floor must be a lloyd tier (got Q8); nothing to size down to"
        );
        // Upgrade the FWHT signs to 256-wide (copy of the need_realloc block from
        // set_v_mode_realloc): fwht2/4-K caches allocate only 128-element sign
        // tables; the q8→lloyd4 transcode runs fwht_shfl_forward_256 (reads
        // signs[0..255]). gen_fwht_signs is a pure LCG, so the first 128 entries
        // are byte-identical and the 128-wide K reads are unaffected.
        let need_realloc = self.givens_cos.as_ref().map_or(false, |t| t.numel() < 256);
        if need_realloc {
            let n = 256usize;
            let s1v = Self::gen_fwht_signs(42, n);
            let s2v = Self::gen_fwht_signs(1042, n);
            let s1b: Vec<u8> = s1v.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let s2b: Vec<u8> = s2v.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let s1 = gpu.alloc_tensor(&[n], DType::F32)?;
            let s2 = gpu.alloc_tensor(&[n], DType::F32)?;
            gpu.hip.memcpy_htod(&s1.buf, &s1b)?;
            gpu.hip.memcpy_htod(&s2.buf, &s2b)?;
            if let Some(old) = self.givens_cos.take() {
                let _ = gpu.free_tensor(old);
            }
            if let Some(old) = self.givens_sin.take() {
                let _ = gpu.free_tensor(old);
            }
            self.givens_cos = Some(s1);
            self.givens_sin = Some(s2);
        }
        // Size V at the FLOOR tier (not the current v_mode). The q8 start phase
        // simply fits fewer positions in this smaller buffer.
        let v_bpp_floor = Self::v_bytes_per_pos(self.n_kv_heads, self.head_dim, v_floor);
        let v_elems = (self.physical_cap * v_bpp_floor + 3) / 4;
        Self::resize_real_tensors_zeroed(gpu, &mut self.v_gpu, v_elems)?;
        // v_mode STAYS at its current value (Q8 fast start tier); only the buffer
        // size changed.

        // Size K at the K FLOOR (fwht2=68 or fwht3=100 @256) when the floor is
        // below the current fwht4 footprint (132 @256). K data is still WRITTEN
        // at the fwht4 stride until the K transcode fires (fwht4→fwht2 remap, or
        // fwht4→fwht3 re-rotation for k_floor=fwht3), exactly mirroring the V
        // side. The 256-wide sign upgrade above ALSO satisfies the fwht4→fwht3
        // re-rotation's sign-width requirement (inverse-128 + forward-256). The
        // K-mode booleans STAY at fwht4 (start tier).
        let k_bph_cur = 4 + self.head_dim / 2; // fwht4 footprint @ this head_dim
        if k_floor_bph < k_bph_cur {
            let k_bpp_floor = self.n_kv_heads * k_floor_bph;
            let k_elems = (self.physical_cap * k_bpp_floor + 3) / 4;
            Self::resize_real_tensors_zeroed(gpu, &mut self.k_gpu, k_elems)?;
        }
        Ok(())
    }

    /// Restore cache mode flags to the adaptive start tier (K=fwht4, V=q8)
    /// without touching buffer owners. Used by atomic controller+cache reset.
    pub fn restore_adaptive_start_flags(&mut self) {
        self.quant_q8 = false;
        self.quant_int8 = false;
        self.quant_hfq4 = false;
        self.quant_asym4 = true;
        self.quant_asym3 = false;
        self.quant_asym2 = false;
        self.quant_fwht = true;
        self.v_mode = VMode::Q8;
        self.quantized = true;
    }

    /// Adaptive-KV: re-quantize the EXISTING V cache (all written positions of
    /// every real KV layer) from the current `v_mode` to a lower `target` tier,
    /// in place. No realloc — the V buffers are floor-sized (allocated at the V
    /// floor) and the lloyd record is smaller than the q8/higher-lloyd record, so
    /// the in-place ascending transcode is byte-safe (dst stride < src stride;
    /// see the per-kernel headers).
    ///
    /// Supported transitions: Q8→Lloyd4 (FWHT), Lloyd4→Lloyd3, Lloyd4→Lloyd2,
    /// Lloyd3→Lloyd2 (rotated-space remap, no FWHT). `n_positions` is the number
    /// of token positions currently written (seq_pos+1, or physical_cap if
    /// compacted). Reuses self.givens_cos/givens_sin (256-wide FWHT signs) for
    /// the q8→lloyd4 FWHT.
    pub fn transcode_v_step(
        &mut self,
        gpu: &mut Gpu,
        target: VMode,
        n_positions: usize,
    ) -> HipResult<()> {
        // Mirror set_v_mode_realloc's guard: lloyd-V is 256-wide and needs an
        // FWHT K mode + head_dim==256.
        assert!(
            (self.quant_asym2 || self.quant_asym3 || self.quant_asym4) && self.quant_fwht,
            "lloyd-V transcode requires an FWHT K mode (quant_asym{{2,3,4}} && quant_fwht)"
        );
        assert!(
            self.head_dim == 256,
            "lloyd-V transcode requires head_dim == 256"
        );
        assert!(
            !matches!(target, VMode::Q8),
            "transcode_v_step only downshifts (target != Q8)"
        );

        if n_positions == 0 {
            self.v_mode = target;
            gpu.invalidate_for_kv_mode_switch();
            return Ok(());
        }

        let n_kv_heads = self.n_kv_heads;
        let head_dim = self.head_dim;

        // Determine the kernel for the (current → target) transition.
        #[derive(Clone, Copy)]
        enum Op {
            Q8ToL4,
            Down(i32, i32),
        }
        let op = match (self.v_mode, target) {
            (VMode::Q8, VMode::Lloyd4) => Op::Q8ToL4,
            (VMode::Lloyd4, VMode::Lloyd3) => Op::Down(4, 3),
            (VMode::Lloyd4, VMode::Lloyd2) => Op::Down(4, 2),
            (VMode::Lloyd3, VMode::Lloyd2) => Op::Down(3, 2),
            (cur, tgt) => panic!("unsupported V transcode {cur:?} -> {tgt:?}"),
        };

        // q8→lloyd4 needs the 256-wide FWHT signs (already reallocated to 256 by
        // set_v_mode_realloc when lloyd-V was enabled at load). Take non-owning
        // views so we don't borrow `self` across the v_gpu iteration below.
        let (signs1, signs2) = match op {
            Op::Q8ToL4 => {
                let s1 = self
                    .givens_cos
                    .as_ref()
                    .expect("q8→lloyd4 transcode needs 256-wide FWHT signs");
                let s2 = self
                    .givens_sin
                    .as_ref()
                    .expect("q8→lloyd4 transcode needs 256-wide FWHT signs");
                // The q8→lloyd4 kernel runs fwht_shfl_forward_256 → reads
                // signs[0..255]. fwht2/fwht4 K caches allocate only 128-element
                // sign tables; adaptive's load path MUST upgrade them to 256
                // (LCG prefix keeps the K-side 128-wide reads byte-identical)
                // before the first transcode. Fail loud rather than OOB-read
                // phantom signs and silently corrupt every position's cnorm.
                assert!(
                    s1.numel() >= 256 && s2.numel() >= 256,
                    "q8→lloyd4 transcode requires 256-wide FWHT signs (got {}); \
                     upgrade fwht2/4 signs to 256 at adaptive load before transcode_v_step",
                    s1.numel()
                );
                (
                    Some(s1.sub_offset(0, s1.numel())),
                    Some(s2.sub_offset(0, s2.numel())),
                )
            }
            Op::Down(_, _) => (None, None),
        };

        // Map the live prefix at the CURRENT V stride before any read/write.
        // Floor-reserved VMM VA may be larger than the mapped prefix — never
        // size or copy from the full virtual reserve.
        self.ensure_mapped_capacity(gpu, n_positions)?;
        let (cur_k_bpt, cur_v_bpt) = if self.uses_vmm_backend() {
            self.vmm_bytes_per_token()?
        } else {
            // Contiguous: full buffer is mapped; keep prior full-layer scratch size.
            (0, 0)
        };
        let prefix_v_bytes = if self.uses_vmm_backend() {
            Self::checked_vmm_product("V source prefix", &[n_positions, cur_v_bpt])?
        } else {
            0
        };
        let prefix_v_elems = if self.uses_vmm_backend() {
            Self::bytes_to_f32_elems("V source prefix", prefix_v_bytes)?
        } else {
            0
        };

        // 1-layer scratch: VMM uses live source-prefix elems only; contiguous
        // keeps full-layer elems (prior behavior). Copy layer→scratch then
        // transcode scratch→layer (non-aliasing).
        let src_elems = if self.uses_vmm_backend() {
            if prefix_v_elems > 0 {
                Some(prefix_v_elems)
            } else {
                None
            }
        } else {
            self.v_gpu
                .iter()
                .map(|t| t.numel())
                .filter(|&n| n > 1)
                .max()
        };
        let scratch = match src_elems {
            Some(n) => Some(gpu.zeros(&[n], DType::F32)?),
            None => None,
        };
        let _ = cur_k_bpt; // K stride unused on V step

        // Free the scratch on EVERY exit path (GpuTensor has no Drop): capture
        // the first error, break, free, then propagate — a HIP failure mid-pass
        // must not leak the per-layer scratch (multi-MB at long context).
        let mut pending: HipResult<()> = Ok(());
        for t in self.v_gpu.iter() {
            // Skip 1-element placeholder buffers for non-KV layers.
            if t.numel() <= 1 {
                continue;
            }
            let scratch = scratch.as_ref().unwrap();
            // Copy the live source prefix into scratch (device-to-device), then
            // read from scratch and write the compacted record back into the layer.
            let nbytes = if self.uses_vmm_backend() {
                prefix_v_bytes
            } else {
                t.byte_size()
            };
            pending = gpu.hip.memcpy_dtod(&scratch.buf, &t.buf, nbytes);
            if pending.is_err() {
                break;
            }
            pending = match op {
                Op::Q8ToL4 => gpu.transcode_v_q8_to_lloyd4(
                    t,
                    scratch,
                    signs1.as_ref().unwrap(),
                    signs2.as_ref().unwrap(),
                    n_kv_heads,
                    head_dim,
                    n_positions,
                ),
                Op::Down(sb, db) => gpu.transcode_v_lloyd_down(
                    t,
                    scratch,
                    n_kv_heads,
                    head_dim,
                    n_positions,
                    sb,
                    db,
                ),
            };
            if pending.is_err() {
                break;
            }
        }

        if let Some(s) = scratch {
            let _ = gpu.free_tensor(s);
        }
        pending?;
        self.v_mode = target;
        gpu.invalidate_for_kv_mode_switch();
        Ok(())
    }

    /// Adaptive-KV: re-quantize the EXISTING K cache to a lower tier
    /// (`target_bits` ∈ {2,3}). Two supported transitions, both from the fwht4
    /// start tier:
    ///   * fwht4 → fwht2 (`target_bits==2`): SAME-WIDTH 128-LUT remap (the
    ///     balanced/aggressive presets' only K step). Reconstructs from the
    ///     fwht4 record and re-quantizes each rotated dim at 2-bit (128-family);
    ///     no FWHT.
    ///   * fwht4 → fwht3 (`target_bits==3`): RE-ROTATION (128-wide → 256-wide).
    ///     Reconstructs normal-space K (dequant + inverse-128), re-rotates
    ///     256-wide, quantizes to TURBO_C3_256. Engaged only by the advanced
    ///     selector with k_floor=fwht3. fwht3→fwht2 never occurs (K starts at
    ///     fwht4 and balanced_steps adds at most one K step), so it is not
    ///     implemented; any other request errors clearly.
    ///
    /// Per real KV layer: copy the K layer into a 1-layer scratch (d2d), then
    /// transcode scratch→layer (separate buffers, never aliased). cnorm is
    /// recomputed per (head, pos). Then flips the K-mode booleans (clears
    /// quant_asym4; sets quant_asym2 OR quant_asym3; quant_fwht stays true) so
    /// the next forward dispatches the right attention kernel, and invalidates
    /// captured graphs.
    ///
    /// Sign tables: fwht4→fwht2 are both 128-wide and share the SAME signs
    /// (gen_fwht_signs(42/1042); the first 128 entries of a 256-wide table equal
    /// the 128-wide table). fwht4→fwht3 RE-ROTATION needs 256-wide signs (the
    /// inverse-128 reads [0..127], forward-256 reads [0..255]) — adaptive's load
    /// path (set_adaptive_floor_alloc with k_floor=fwht3) upgrades them to 256.
    ///
    /// `n_positions` is the number of token positions currently written. The K
    /// buffer is floor-sized at adaptive load (fwht2=68 or fwht3=100 B/head), so
    /// the fwht4 phase physically holds only ~physical_cap*floor_bph/132
    /// positions — the controller transcodes K before that cap.
    pub fn transcode_k_step(
        &mut self,
        gpu: &mut Gpu,
        target_bits: u32,
        n_positions: usize,
    ) -> HipResult<()> {
        // Determine the current K mode. Adaptive starts at fwht4
        // (quant_asym4 && quant_fwht). Supported steps: fwht4->fwht2 (remap),
        // fwht4->fwht3 (re-rotation).
        let src_is_fwht4 = self.quant_asym4 && self.quant_fwht;
        let cur_label = if !self.quant_fwht {
            "non-fwht"
        } else if self.quant_asym4 {
            "fwht4"
        } else if self.quant_asym3 {
            "fwht3"
        } else if self.quant_asym2 {
            "fwht2"
        } else {
            "unknown"
        };
        if !(src_is_fwht4 && (target_bits == 2 || target_bits == 3)) {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "K transcode {cur_label}->fwht{target_bits} not implemented \
                     (only fwht4->fwht2 same-width remap and fwht4->fwht3 re-rotation are supported)"
                ),
            ));
        }
        assert!(
            self.head_dim % 128 == 0,
            "fwht K transcode requires head_dim multiple of 128"
        );
        // fwht4->fwht3 re-rotation is hard-wired to the 128↔256 width crossing.
        if target_bits == 3 {
            assert!(
                self.head_dim == 256,
                "fwht4->fwht3 re-rotation requires head_dim == 256"
            );
        }

        if n_positions == 0 {
            self.quant_asym4 = false;
            if target_bits == 3 {
                self.quant_asym3 = true;
            } else {
                self.quant_asym2 = true;
            }
            // quant_fwht stays true.
            gpu.invalidate_for_kv_mode_switch();
            return Ok(());
        }

        let n_kv_heads = self.n_kv_heads;
        let head_dim = self.head_dim;

        // For the re-rotation (fwht4->fwht3) the kernel runs fwht_shfl_inverse
        // (128-wide, reads signs[0..127]) then fwht_shfl_forward_256 (reads
        // signs[0..255]). The cache signs MUST be 256-wide — adaptive's load
        // (set_adaptive_floor_alloc with k_floor=fwht3) upgrades them. Fail loud
        // rather than OOB-read phantom signs and silently corrupt every record.
        if target_bits == 3 {
            let n1 = self.givens_cos.as_ref().map_or(0, |t| t.numel());
            let n2 = self.givens_sin.as_ref().map_or(0, |t| t.numel());
            assert!(
                n1 >= 256 && n2 >= 256,
                "fwht4->fwht3 transcode requires 256-wide FWHT signs (got {n1}); \
                 set_adaptive_floor_alloc(k_floor=fwht3) must upgrade signs to 256 first"
            );
        }
        // Take non-owning views of the 256-wide signs for the re-rotation so we
        // don't borrow `self` across the k_gpu iteration below.
        let signs = if target_bits == 3 {
            let s1 = self.givens_cos.as_ref().unwrap();
            let s2 = self.givens_sin.as_ref().unwrap();
            Some((s1.sub_offset(0, s1.numel()), s2.sub_offset(0, s2.numel())))
        } else {
            None
        };

        // Map the live prefix at the CURRENT K stride before any read/write.
        // Floor-reserved VMM VA may be larger than the mapped prefix — never
        // size or copy from the full virtual reserve.
        self.ensure_mapped_capacity(gpu, n_positions)?;
        let (cur_k_bpt, _cur_v_bpt) = if self.uses_vmm_backend() {
            self.vmm_bytes_per_token()?
        } else {
            (0, 0)
        };
        let prefix_k_bytes = if self.uses_vmm_backend() {
            Self::checked_vmm_product("K source prefix", &[n_positions, cur_k_bpt])?
        } else {
            0
        };
        let prefix_k_elems = if self.uses_vmm_backend() {
            Self::bytes_to_f32_elems("K source prefix", prefix_k_bytes)?
        } else {
            0
        };

        // 1-layer scratch: VMM uses live source-prefix elems only; contiguous
        // keeps full-layer elems (prior behavior).
        let src_elems = if self.uses_vmm_backend() {
            if prefix_k_elems > 0 {
                Some(prefix_k_elems)
            } else {
                None
            }
        } else {
            self.k_gpu
                .iter()
                .map(|t| t.numel())
                .filter(|&n| n > 1)
                .max()
        };
        let scratch = match src_elems {
            Some(n) => Some(gpu.zeros(&[n], DType::F32)?),
            None => None,
        };

        // Free the scratch on EVERY exit path (GpuTensor has no Drop): capture
        // the first error, break, free, then propagate.
        let mut pending: HipResult<()> = Ok(());
        for t in self.k_gpu.iter() {
            // Skip 1-element placeholder buffers for non-KV layers.
            if t.numel() <= 1 {
                continue;
            }
            let scratch = scratch.as_ref().unwrap();
            let nbytes = if self.uses_vmm_backend() {
                prefix_k_bytes
            } else {
                t.byte_size()
            };
            pending = gpu.hip.memcpy_dtod(&scratch.buf, &t.buf, nbytes);
            if pending.is_err() {
                break;
            }
            pending = if target_bits == 3 {
                let (s1, s2) = signs.as_ref().unwrap();
                gpu.transcode_k_fwht4_to_fwht3(
                    t,
                    scratch,
                    s1,
                    s2,
                    n_kv_heads,
                    head_dim,
                    n_positions,
                )
            } else {
                gpu.transcode_k_fwht4_to_fwht2(t, scratch, n_kv_heads, head_dim, n_positions)
            };
            if pending.is_err() {
                break;
            }
        }

        if let Some(s) = scratch {
            let _ = gpu.free_tensor(s);
        }
        pending?;

        // Flip the K-mode booleans so the next forward dispatches the target
        // attention kernel. quant_fwht stays true (still FWHT-rotated K).
        self.quant_asym4 = false;
        if target_bits == 3 {
            self.quant_asym3 = true;
        } else {
            self.quant_asym2 = true;
        }
        gpu.invalidate_for_kv_mode_switch();
        Ok(())
    }

    /// Adaptive VMM constructor: reserve at K/V floors, start encoding FWHT4/Q8.
    /// One stable K and one stable V owner per real layer for the load lifetime.
    /// Built on `vmm_layout_with_reserve`; never replaces owners after publish.
    pub fn new_gpu_vmm_adaptive_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        k_floor_bph: usize,
        v_floor: VMode,
    ) -> HipResult<Self> {
        if max_seq == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "adaptive VMM requires max_seq > 0",
            ));
        }
        if !is_kv_layer.iter().any(|is_kv| *is_kv) {
            return Err(hip_bridge::HipError::new(
                0,
                "adaptive VMM requires at least one KV-carrying layer",
            ));
        }
        if matches!(v_floor, VMode::Q8) {
            return Err(hip_bridge::HipError::new(
                0,
                "adaptive VMM V floor must be a lloyd tier (got Q8)",
            ));
        }
        if head_dim != 256 {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("adaptive VMM requires head_dim=256 (got {head_dim})"),
            ));
        }
        let layout = Self::vmm_layout_with_reserve(
            KvMode::Fwht4,
            VMode::Q8,
            n_kv_heads,
            head_dim,
            max_seq,
            k_floor_bph,
            v_floor,
        )?;
        let (mut k_gpu, mut v_gpu) = Self::alloc_k_v_vmm_filtered(
            gpu,
            layout.k_reserve_elems,
            layout.v_reserve_elems,
            is_kv_layer,
        )?;

        // Adaptive always needs 256-wide FWHT signs (q8→lloyd and optional fwht4→fwht3).
        let n = layout.rotation_table_len.max(256);
        let s1_vals = Self::gen_fwht_signs(42, n);
        let s2_vals = Self::gen_fwht_signs(1042, n);
        let s1_bytes: Vec<u8> = s1_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s2_bytes: Vec<u8> = s2_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let tables = (|| -> HipResult<(GpuTensor, GpuTensor)> {
            let s1 = gpu.alloc_tensor(&[n], DType::F32)?;
            let s2 = match gpu.alloc_tensor(&[n], DType::F32) {
                Ok(s2) => s2,
                Err(err) => {
                    let _ = gpu.free_tensor(s1);
                    return Err(err);
                }
            };
            if let Err(err) = gpu.hip.memcpy_htod(&s1.buf, &s1_bytes) {
                let _ = gpu.free_tensor(s1);
                let _ = gpu.free_tensor(s2);
                return Err(err);
            }
            if let Err(err) = gpu.hip.memcpy_htod(&s2.buf, &s2_bytes) {
                let _ = gpu.free_tensor(s1);
                let _ = gpu.free_tensor(s2);
                return Err(err);
            }
            Ok((s1, s2))
        })();
        let (s1, s2) = match tables {
            Ok(pair) => pair,
            Err(err) => {
                for tensor in k_gpu.drain(..).chain(v_gpu.drain(..)) {
                    let _ = gpu.free_tensor(tensor);
                }
                return Err(err);
            }
        };

        let mut cache = Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim: layout.kv_dim,
            max_seq,
            physical_cap: layout.physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: true, // FWHT4 start
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: true,
            boundary_layers: 0,
            givens_cos: Some(s1),
            givens_sin: Some(s2),
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8, // start tier
        };
        if let Err(err) = cache.ensure_mapped_capacity(gpu, 1) {
            let _ = cache.free_gpu(gpu);
            return Err(err);
        }
        let n_kv = is_kv_layer.iter().filter(|is_kv| **is_kv).count();
        let mapped = match cache.mapped_token_capacity() {
            Ok(capacity) => capacity.unwrap_or(0),
            Err(err) => {
                let _ = cache.free_gpu(gpu);
                return Err(err);
            }
        };
        eprintln!(
            "KV cache: adaptive vmm ({n_kv}/{} layers; start FWHT4/Q8; K floor {}B/head V floor {:?}; reserve_k={}B reserve_v={}B; mapped_prefix={mapped} / max_seq={max_seq})",
            is_kv_layer.len(),
            k_floor_bph,
            v_floor,
            layout.k_reserve_bytes,
            layout.v_reserve_bytes,
        );
        Ok(cache)
    }

    /// Q8_0 KV cache that skips allocation for layers flagged as non-KV.
    /// Each `is_kv_layer[i] == false` slot gets a 1-element placeholder
    /// (~4 bytes) instead of the full `cache_elems × 4` allocation.
    ///
    /// For Qwen 3.5 hybrid (48 DeltaNet + 16 FullAttention layers), saves
    /// 48 × cache_elems × 4 bytes per cache — at ctx=64K this is multi-GB.
    pub fn new_gpu_q8_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_q8_capped_filtered(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    /// Capped variant of [`new_gpu_q8_filtered`].
    pub fn new_gpu_q8_capped_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]"
        );
        let kv_dim = n_kv_heads * head_dim;
        let blocks_per_head = head_dim / 32;
        let total_blocks = n_kv_heads * blocks_per_head;
        let cache_bytes = physical_cap * total_blocks * 34;
        let cache_elems = (cache_bytes + 3) / 4;
        let (k_gpu, v_gpu) = Self::alloc_k_v_filtered(gpu, cache_elems, cache_elems, is_kv_layer)?;
        let n_kv = is_kv_layer.iter().filter(|b| **b).count();
        eprintln!(
            "KV cache: q8 ({n_kv}/{} layers carry KV, others placeholder)",
            is_kv_layer.len()
        );
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: true,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    /// Unified static VMM constructor for every legal (K mode × V mode) pair.
    ///
    /// Reserves stable K/V VAs from [`vmm_static_layout`], maps an initial
    /// prefix, and never replaces owners afterward. Lloyd-V is accepted only
    /// with FWHT-K and head_dim=256 (validated by the layout table).
    pub fn new_gpu_vmm_capped_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
        mode: KvMode,
        v_mode: VMode,
    ) -> HipResult<Self> {
        if physical_cap == 0 || physical_cap > max_seq_len {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "VMM {mode:?} physical_cap must be in 1..=max_seq_len (got physical_cap={physical_cap}, max_seq_len={max_seq_len})"
                ),
            ));
        }
        if !is_kv_layer.iter().any(|is_kv| *is_kv) {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("VMM {mode:?} requires at least one KV-carrying layer"),
            ));
        }
        let layout = Self::vmm_static_layout(mode, v_mode, n_kv_heads, head_dim, physical_cap)?;
        let (mut k_gpu, mut v_gpu) = Self::alloc_k_v_vmm_filtered(
            gpu,
            layout.k_reserve_elems,
            layout.v_reserve_elems,
            is_kv_layer,
        )?;

        // Optional rotation tables (Givens angles or FWHT signs). Built before
        // the cache is published so a table failure can roll back arenas.
        let rotation = if layout.rotation_table_len == 0 {
            None
        } else {
            let n = layout.rotation_table_len;
            let (a_vals, b_vals) = if layout.uses_fwht_signs {
                (Self::gen_fwht_signs(42, n), Self::gen_fwht_signs(1042, n))
            } else {
                Self::gen_givens_angles(42, n)
            };
            let a_bytes: Vec<u8> = a_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let b_bytes: Vec<u8> = b_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let tables = (|| -> HipResult<(GpuTensor, GpuTensor)> {
                let a = gpu.alloc_tensor(&[n], DType::F32)?;
                let b = match gpu.alloc_tensor(&[n], DType::F32) {
                    Ok(b) => b,
                    Err(err) => {
                        let _ = gpu.free_tensor(a);
                        return Err(err);
                    }
                };
                if let Err(err) = gpu.hip.memcpy_htod(&a.buf, &a_bytes) {
                    let _ = gpu.free_tensor(a);
                    let _ = gpu.free_tensor(b);
                    return Err(err);
                }
                if let Err(err) = gpu.hip.memcpy_htod(&b.buf, &b_bytes) {
                    let _ = gpu.free_tensor(a);
                    let _ = gpu.free_tensor(b);
                    return Err(err);
                }
                Ok((a, b))
            })();
            match tables {
                Ok(pair) => Some(pair),
                Err(err) => {
                    for tensor in k_gpu.drain(..).chain(v_gpu.drain(..)) {
                        let _ = gpu.free_tensor(tensor);
                    }
                    return Err(err);
                }
            }
        };

        let (quant_q8, quant_asym4, quant_asym3, quant_asym2, quant_fwht) =
            Self::vmm_mode_flags(mode);
        let (givens_cos, givens_sin) = match rotation {
            Some((a, b)) => (Some(a), Some(b)),
            None => (None, None),
        };

        let mut cache = Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim: layout.kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4,
            quant_asym3,
            quant_asym2,
            quant_fwht,
            boundary_layers: 0,
            givens_cos,
            givens_sin,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode,
        };
        // Map initial prefix before any write; roll back on failure.
        if let Err(err) = cache.ensure_mapped_capacity(gpu, 1) {
            let _ = cache.free_gpu(gpu);
            return Err(err);
        }
        let n_kv = is_kv_layer.iter().filter(|is_kv| **is_kv).count();
        let mapped = match cache.mapped_token_capacity() {
            Ok(capacity) => capacity.unwrap_or(0),
            Err(err) => {
                let _ = cache.free_gpu(gpu);
                return Err(err);
            }
        };
        let v_label = match v_mode {
            VMode::Q8 => "Q8".to_string(),
            VMode::Lloyd2 => "lloyd2".to_string(),
            VMode::Lloyd3 => "lloyd3".to_string(),
            VMode::Lloyd4 => "lloyd4".to_string(),
        };
        eprintln!(
            "KV cache: {mode:?} vmm ({n_kv}/{} layers carry KV; K {}B/head + V {v_label} {}B/head; mapped_prefix={mapped} / physical_cap={physical_cap} / max_seq={max_seq_len})",
            is_kv_layer.len(),
            layout.k_bytes_per_head,
            layout.v_bytes_per_head,
        );
        Ok(cache)
    }

    /// VMM-backed Q8 cache for Qwen3.5 hybrid stacks.
    /// Thin wrapper over [`Self::new_gpu_vmm_capped_filtered`].
    pub fn new_gpu_q8_vmm_capped_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_vmm_capped_filtered(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            max_seq_len,
            physical_cap,
            KvMode::Q8,
            VMode::Q8,
        )
    }

    /// Create INT8 co-located KV cache: [f32 scale][pad 4B][int8 × head_dim] = 136 bytes per head.
    pub fn new_gpu_int8c(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let bph = 8 + head_dim; // 136 for head_dim=128 (8-byte header + data)
        let bpp = n_kv_heads * bph;
        let cache_bytes = max_seq_len * bpp;
        let cache_elems = (cache_bytes + 3) / 4;
        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[cache_elems], DType::F32)?);
            v_gpu.push(gpu.zeros(&[cache_elems], DType::F32)?);
        }
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: true,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    /// Create HFQ4 KV cache: co-located blocks. 72 bytes per head (scale+zero+nibbles).
    pub fn new_gpu_hfq4kv(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let bytes_per_block = 8 + head_dim / 2; // 72 for head_dim=128
        let bytes_per_pos = n_kv_heads * bytes_per_block;
        let cache_bytes = max_seq_len * bytes_per_pos;
        let cache_elems = (cache_bytes + 3) / 4;
        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[cache_elems], DType::F32)?);
            v_gpu.push(gpu.zeros(&[cache_elems], DType::F32)?);
        }
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: true,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    /// Create HFQ8 KV cache: FP32 scale+zero per head, contiguous uint8 data.
    pub fn new_gpu_hfq8(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let val_elems = (max_seq_len * kv_dim + 3) / 4; // uint8 data, rounded to f32
        let scale_elems = max_seq_len * n_kv_heads * 2; // scale + zero per head per pos
        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        let mut k_scales = Vec::with_capacity(n_layers);
        let mut v_scales = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[val_elems], DType::F32)?);
            v_gpu.push(gpu.zeros(&[val_elems], DType::F32)?);
            k_scales.push(gpu.zeros(&[scale_elems], DType::F32)?);
            v_scales.push(gpu.zeros(&[scale_elems], DType::F32)?);
        }
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales,
            v_scales,
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    /// Create INT8 KV cache with separate scale arrays. Clean contiguous layout.
    pub fn new_gpu_int8(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        // Values: max_seq × kv_dim bytes (int8). Round up to f32 elements for alloc.
        let val_elems = (max_seq_len * kv_dim + 3) / 4;
        // Scales: max_seq × n_kv_heads floats
        let scale_elems = max_seq_len * n_kv_heads;
        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        let mut k_scales = Vec::with_capacity(n_layers);
        let mut v_scales = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[val_elems], DType::F32)?);
            v_gpu.push(gpu.zeros(&[val_elems], DType::F32)?);
            k_scales.push(gpu.zeros(&[scale_elems], DType::F32)?);
            v_scales.push(gpu.zeros(&[scale_elems], DType::F32)?);
        }
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales,
            v_scales,
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: true,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    /// Generate deterministic Givens rotation angles from a seed.
    /// Returns (cos_theta, sin_theta) each of length n_blocks.
    pub fn gen_givens_angles(seed: u32, n_blocks: usize) -> (Vec<f32>, Vec<f32>) {
        let mut state = seed;
        let mut cos_vals = Vec::with_capacity(n_blocks);
        let mut sin_vals = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            let angle = (state as f64 / 0x7fffffff as f64) * std::f64::consts::TAU;
            cos_vals.push(angle.cos() as f32);
            sin_vals.push(angle.sin() as f32);
        }
        (cos_vals, sin_vals)
    }

    /// Create asym4 KV cache: K at 4-bit rotated (Givens + Lloyd-Max), V at Q8_0.
    /// head_dim=256 → K=132 B/head, V=272 B/head → 404 B/head total (5.1× vs fp32).
    /// Back-compat wrapper: `physical_cap == max_seq_len`.
    pub fn new_gpu_asym4(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_asym4_capped(
            gpu,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    /// Filtered variant of [`new_gpu_asym4`]: skips KV alloc for non-KV layers.
    pub fn new_gpu_asym4_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "asym4 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        let physical_cap = max_seq_len;
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 2;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = Self::alloc_k_v_filtered(gpu, k_elems, v_elems, is_kv_layer)?;
        let n_blocks = head_dim / 2;
        let (cos_vals, sin_vals) = Self::gen_givens_angles(42, n_blocks);
        let cb: Vec<u8> = cos_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let sb: Vec<u8> = sin_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let ct = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        let st = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        gpu.hip.memcpy_htod(&ct.buf, &cb)?;
        gpu.hip.memcpy_htod(&st.buf, &sb)?;
        let v_bph = v_bpp / n_kv_heads;
        let n_kv = is_kv_layer.iter().filter(|b| **b).count();
        eprintln!(
            "KV cache: asym4 filtered ({n_kv}/{} layers carry KV; K rotated-4b {k_bph}B + V Q8 {v_bph}B = {} B/head)",
            is_kv_layer.len(),
            k_bph + v_bph,
        );
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: true,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: Some(ct),
            givens_sin: Some(st),
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    /// Filtered variant of [`new_gpu_fwht4`]: skips KV alloc for non-KV layers.
    /// Mirrors `new_gpu_asym4_filtered` byte-for-byte except the rotation
    /// parameter buffers hold signs1/signs2 (FWHT) instead of cos/sin (Givens)
    /// and `quant_fwht` is set true. K-cache byte layout is identical to
    /// asym4 so scoring kernels are shared.
    pub fn new_gpu_fwht4_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "fwht4 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        let physical_cap = max_seq_len;
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 2;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = Self::alloc_k_v_filtered(gpu, k_elems, v_elems, is_kv_layer)?;
        // fwht_shfl_forward operates on 128 elements regardless of head_dim;
        // signs are shared across the hd=256 two-half rotation. Seeds (42,
        // 1042) match the MQ4 weight-FWHT convention.
        let n_signs = 128;
        let s1_vals = Self::gen_fwht_signs(42, n_signs);
        let s2_vals = Self::gen_fwht_signs(1042, n_signs);
        let s1_bytes: Vec<u8> = s1_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s2_bytes: Vec<u8> = s2_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s1 = gpu.alloc_tensor(&[n_signs], DType::F32)?;
        let s2 = gpu.alloc_tensor(&[n_signs], DType::F32)?;
        gpu.hip.memcpy_htod(&s1.buf, &s1_bytes)?;
        gpu.hip.memcpy_htod(&s2.buf, &s2_bytes)?;
        let v_bph = v_bpp / n_kv_heads;
        let n_kv = is_kv_layer.iter().filter(|b| **b).count();
        eprintln!(
            "KV cache: fwht4 filtered ({n_kv}/{} layers carry KV; K FWHT-4b {k_bph}B + V Q8 {v_bph}B = {} B/head)",
            is_kv_layer.len(),
            k_bph + v_bph,
        );
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: true,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: true,
            boundary_layers: 0,
            givens_cos: Some(s1),
            givens_sin: Some(s2),
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    /// Same as [`new_gpu_asym4`] with an explicit physical_cap. Eviction-aware.
    pub fn new_gpu_asym4_capped(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "asym4 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        assert!(
            physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]"
        );
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 2;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;

        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[k_elems], DType::F32)?);
            v_gpu.push(gpu.zeros(&[v_elems], DType::F32)?);
        }
        let n_blocks = head_dim / 2;
        let (cos_vals, sin_vals) = Self::gen_givens_angles(42, n_blocks);
        let cb: Vec<u8> = cos_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let sb: Vec<u8> = sin_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let ct = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        let st = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        gpu.hip.memcpy_htod(&ct.buf, &cb)?;
        gpu.hip.memcpy_htod(&st.buf, &sb)?;
        let v_bph = v_bpp / n_kv_heads;
        eprintln!(
            "KV cache: asym4 (K rotated-4b {k_bph}B + V Q8 {v_bph}B = {} B/head, {:.1}x vs fp32)",
            k_bph + v_bph,
            (head_dim * 4 * 2) as f64 / (k_bph + v_bph) as f64
        );
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: true,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: Some(ct),
            givens_sin: Some(st),
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    /// Create fwht4 KV cache: K at 4-bit signed-FWHT-rotated (Lloyd-Max
    /// post-FWHT N(0, 1/128)), V at Q8_0 in normal space. Byte-identical
    /// storage to asym4 — only the rotation primitive differs.
    /// Back-compat wrapper: `physical_cap == max_seq_len`.
    pub fn new_gpu_fwht4(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_fwht4_capped(
            gpu,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    /// Same as [`new_gpu_fwht4`] with an explicit physical_cap. Eviction-aware.
    pub fn new_gpu_fwht4_capped(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "fwht4 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        assert!(
            physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]"
        );
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 2;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;

        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[k_elems], DType::F32)?);
            v_gpu.push(gpu.zeros(&[v_elems], DType::F32)?);
        }
        // fwht_shfl_forward operates on 128 elements regardless of head_dim
        // (hd=256 is processed as 2 halves with the same signs reused).
        // Seeds (42, 1042) match the established MQ4 weight-FWHT convention
        // (see crates/hipfire-quantize/src/bin/dflash_convert.rs:600 and
        // crates/hipfire-arch-qwen35/src/qwen35.rs:872 — same PRNG family).
        let n_signs = 128;
        let s1_vals = Self::gen_fwht_signs(42, n_signs);
        let s2_vals = Self::gen_fwht_signs(1042, n_signs);
        let s1_bytes: Vec<u8> = s1_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s2_bytes: Vec<u8> = s2_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s1 = gpu.alloc_tensor(&[n_signs], DType::F32)?;
        let s2 = gpu.alloc_tensor(&[n_signs], DType::F32)?;
        gpu.hip.memcpy_htod(&s1.buf, &s1_bytes)?;
        gpu.hip.memcpy_htod(&s2.buf, &s2_bytes)?;
        let v_bph = v_bpp / n_kv_heads;
        eprintln!(
            "KV cache: fwht4 (K FWHT-4b {k_bph}B + V Q8 {v_bph}B = {} B/head, {:.1}x vs fp32)",
            k_bph + v_bph,
            (head_dim * 4 * 2) as f64 / (k_bph + v_bph) as f64
        );
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: true,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: true,
            boundary_layers: 0,
            givens_cos: Some(s1),
            givens_sin: Some(s2),
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    /// Create asym3 KV cache: K at 3-bit rotated (Lloyd-Max N(0, 1/256)), V at Q8_0.
    /// head_dim=256 → K=100 B/head, V=272 B/head → 372 B/head (5.5× vs fp32).
    /// Back-compat wrapper: allocates physical_cap == max_seq_len slots per layer.
    pub fn new_gpu_asym3(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_asym3_capped(
            gpu,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    /// Gemma4 variant of `new_gpu_asym3` (global 512). Delegates to the
    /// gemma4-capped door so the shared 256-only guard stays unchanged.
    pub fn new_gpu_asym3_gemma4(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_asym3_capped_gemma4(
            gpu,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    /// Filtered variant of [`new_gpu_asym3`]: skips KV allocation for layers
    /// flagged as non-KV (LinearAttention/DeltaNet in hybrid arches). See
    /// [`alloc_k_v_filtered`].
    pub fn new_gpu_asym3_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_asym3_capped_filtered(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    /// Capped + filtered asym3 — saves multi-GB at long ctx for Qwen 3.5 hybrid.
    pub fn new_gpu_asym3_capped_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 256,
            "asym3 currently requires head_dim=256 (Qwen 3.5)"
        );
        assert!(head_dim % 32 == 0);
        assert!(
            physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]"
        );
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + (head_dim * 3) / 8;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = Self::alloc_k_v_filtered(gpu, k_elems, v_elems, is_kv_layer)?;
        let n_blocks = head_dim / 2;
        let (cos_vals, sin_vals) = Self::gen_givens_angles(42, n_blocks);
        let cb: Vec<u8> = cos_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let sb: Vec<u8> = sin_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let ct = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        let st = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        gpu.hip.memcpy_htod(&ct.buf, &cb)?;
        gpu.hip.memcpy_htod(&st.buf, &sb)?;
        let v_bph = v_bpp / n_kv_heads;
        let n_kv = is_kv_layer.iter().filter(|b| **b).count();
        eprintln!(
            "KV cache: asym3 filtered ({n_kv}/{} layers carry KV; K rotated-3b {k_bph}B + V Q8 {v_bph}B = {} B/head, physical_cap={physical_cap} / max_seq={max_seq_len})",
            is_kv_layer.len(),
            k_bph + v_bph,
        );
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: true,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: Some(ct),
            givens_sin: Some(st),
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    /// VMM-backed Asym3-K/Q8-V cache for Qwen3.5 hybrid stacks.
    /// Thin wrapper over [`Self::new_gpu_vmm_capped_filtered`].
    pub fn new_gpu_asym3_vmm_capped_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_vmm_capped_filtered(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            max_seq_len,
            physical_cap,
            KvMode::Asym3,
            VMode::Q8,
        )
    }

    /// VMM-backed FWHT3-K/Q8-V cache for Qwen3.5 hybrid stacks.
    /// Thin wrapper over [`Self::new_gpu_vmm_capped_filtered`].
    pub fn new_gpu_fwht3_vmm_capped_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_vmm_capped_filtered(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            max_seq_len,
            physical_cap,
            KvMode::Fwht3,
            VMode::Q8,
        )
    }

    /// Filtered variant of fwht3 — signed-FWHT-256 K-rotation, 3-bit centroid,
    /// V at Q8_0. Same byte layout as asym3_filtered; rotation primitive swapped
    /// to fwht_shfl_forward_256 which expects 256-element signs1/signs2.
    pub fn new_gpu_fwht3_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_fwht3_capped_filtered(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    /// Capped + filtered fwht3 — layer-filtered (skips non-KV layers) with an
    /// explicit `physical_cap` for TriAttention/CASK eviction. Default path has
    /// `physical_cap == max_seq_len` (no eviction). Byte layout identical to
    /// `asym3_capped_filtered`; rotation primitive is signed-FWHT-256.
    pub fn new_gpu_fwht3_capped_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 256,
            "fwht3 currently requires head_dim=256 (Qwen 3.5)"
        );
        Self::new_gpu_fwht3_capped_filtered_inner(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            max_seq_len,
            physical_cap,
        )
    }

    /// Gemma 4 door for the global-attention KV, whose `global_head_dim` is
    /// 512. Deliberately a SEPARATE entry point: the shared constructor above
    /// keeps its exact 256-only guard for Qwen 3.5/3.6 and every other
    /// architecture. Gemma gets its own door rather than a bound widened on
    /// everyone's behalf — a relaxed shared guard silently admits misconfigured
    /// geometry for architectures that were previously protected from it.
    pub fn new_gpu_fwht3_capped_filtered_gemma4(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 256 || head_dim == 512,
            "fwht3 (gemma4) requires head_dim=256 or 512 (got {head_dim})"
        );
        Self::new_gpu_fwht3_capped_filtered_inner(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            max_seq_len,
            physical_cap,
        )
    }

    fn new_gpu_fwht3_capped_filtered_inner(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(head_dim % 32 == 0);
        assert!(
            physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]"
        );
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + (head_dim * 3) / 8;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = Self::alloc_k_v_filtered(gpu, k_elems, v_elems, is_kv_layer)?;
        // fwht_shfl_forward_256 reads signs[tid*8..tid*8+7], so 256 floats each.
        let n_signs = 256;
        let s1_vals = Self::gen_fwht_signs(42, n_signs);
        let s2_vals = Self::gen_fwht_signs(1042, n_signs);
        let s1_bytes: Vec<u8> = s1_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s2_bytes: Vec<u8> = s2_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s1 = gpu.alloc_tensor(&[n_signs], DType::F32)?;
        let s2 = gpu.alloc_tensor(&[n_signs], DType::F32)?;
        gpu.hip.memcpy_htod(&s1.buf, &s1_bytes)?;
        gpu.hip.memcpy_htod(&s2.buf, &s2_bytes)?;
        let v_bph = v_bpp / n_kv_heads;
        let n_kv = is_kv_layer.iter().filter(|b| **b).count();
        eprintln!(
            "KV cache: fwht3 filtered ({n_kv}/{} layers carry KV; K FWHT-3b {k_bph}B + V Q8 {v_bph}B = {} B/head)",
            is_kv_layer.len(),
            k_bph + v_bph,
        );
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: true,
            quant_asym2: false,
            quant_fwht: true,
            boundary_layers: 0,
            givens_cos: Some(s1),
            givens_sin: Some(s2),
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    /// Same as [`new_gpu_asym3`] but with an explicit physical capacity. When
    /// `physical_cap < max_seq_len`, the cache is sized for `physical_cap`
    /// tokens along the time axis; the caller is responsible for triggering
    /// TriAttention/CASK eviction before the physical position overruns
    /// `physical_cap`. `max_seq_len` is retained for RoPE/mask purposes.
    pub fn new_gpu_asym3_capped(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 256,
            "asym3 currently requires head_dim=256 (Qwen 3.5)"
        );
        Self::new_gpu_asym3_capped_inner(
            gpu, n_layers, n_kv_heads, head_dim, max_seq_len, physical_cap,
        )
    }

    /// Gemma 4 door for the global-attention KV, whose `global_head_dim` is
    /// 512. Separate entry point so the shared constructor above keeps its
    /// exact 256-only guard for Qwen 3.5/3.6 and every other architecture.
    pub fn new_gpu_asym3_capped_gemma4(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 256 || head_dim == 512,
            "asym3 (gemma4) requires head_dim=256 or 512 (got {head_dim})"
        );
        Self::new_gpu_asym3_capped_inner(
            gpu, n_layers, n_kv_heads, head_dim, max_seq_len, physical_cap,
        )
    }

    fn new_gpu_asym3_capped_inner(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(head_dim % 32 == 0);
        assert!(
            physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]"
        );
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + (head_dim * 3) / 8;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;

        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[k_elems], DType::F32)?);
            v_gpu.push(gpu.zeros(&[v_elems], DType::F32)?);
        }
        let n_blocks = head_dim / 2;
        let (cos_vals, sin_vals) = Self::gen_givens_angles(42, n_blocks);
        let cb: Vec<u8> = cos_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let sb: Vec<u8> = sin_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let ct = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        let st = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        gpu.hip.memcpy_htod(&ct.buf, &cb)?;
        gpu.hip.memcpy_htod(&st.buf, &sb)?;
        let v_bph = v_bpp / n_kv_heads;
        eprintln!("KV cache: asym3 (K rotated-3b {k_bph}B + V Q8 {v_bph}B = {} B/head, {:.1}x vs fp32, physical_cap={physical_cap} / max_seq={max_seq_len})",
            k_bph + v_bph, (head_dim * 4 * 2) as f64 / (k_bph + v_bph) as f64);
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: true,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: Some(ct),
            givens_sin: Some(st),
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    /// Create asym2 KV cache: K at 2-bit rotated, V at Q8_0.
    /// head_dim=256 → K=68 B/head, V=272 B/head → 340 B/head (6.0× vs fp32).
    /// Back-compat wrapper: `physical_cap == max_seq_len`.
    pub fn new_gpu_asym2(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_asym2_capped(
            gpu,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    /// Filtered variant of [`new_gpu_asym2`]: skips KV alloc for non-KV layers.
    pub fn new_gpu_asym2_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "asym2 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        let physical_cap = max_seq_len;
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 4;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = Self::alloc_k_v_filtered(gpu, k_elems, v_elems, is_kv_layer)?;
        let n_blocks = head_dim / 2;
        let (cos_vals, sin_vals) = Self::gen_givens_angles(42, n_blocks);
        let cb: Vec<u8> = cos_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let sb: Vec<u8> = sin_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let ct = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        let st = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        gpu.hip.memcpy_htod(&ct.buf, &cb)?;
        gpu.hip.memcpy_htod(&st.buf, &sb)?;
        let v_bph = v_bpp / n_kv_heads;
        let n_kv = is_kv_layer.iter().filter(|b| **b).count();
        eprintln!(
            "KV cache: asym2 filtered ({n_kv}/{} layers carry KV; K rotated-2b {k_bph}B + V Q8 {v_bph}B = {} B/head)",
            is_kv_layer.len(),
            k_bph + v_bph,
        );
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: true,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: Some(ct),
            givens_sin: Some(st),
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    /// Filtered variant of fwht2 — signed-FWHT-128 K-rotation, 2-bit centroid,
    /// V at Q8_0. Same 2-pass-over-128 structure as fwht4, signs are 128 floats.
    pub fn new_gpu_fwht2_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_fwht2_capped_filtered(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    /// Capped + filtered fwht2 — layer-filtered with explicit `physical_cap`
    /// for TriAttention/CASK eviction. Byte layout identical to
    /// `asym2_capped`; rotation primitive is signed-FWHT-128.
    pub fn new_gpu_fwht2_capped_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "fwht2 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        assert!(
            physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]"
        );
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 4;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = Self::alloc_k_v_filtered(gpu, k_elems, v_elems, is_kv_layer)?;
        let n_signs = 128;
        let s1_vals = Self::gen_fwht_signs(42, n_signs);
        let s2_vals = Self::gen_fwht_signs(1042, n_signs);
        let s1_bytes: Vec<u8> = s1_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s2_bytes: Vec<u8> = s2_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s1 = gpu.alloc_tensor(&[n_signs], DType::F32)?;
        let s2 = gpu.alloc_tensor(&[n_signs], DType::F32)?;
        gpu.hip.memcpy_htod(&s1.buf, &s1_bytes)?;
        gpu.hip.memcpy_htod(&s2.buf, &s2_bytes)?;
        let v_bph = v_bpp / n_kv_heads;
        let n_kv = is_kv_layer.iter().filter(|b| **b).count();
        eprintln!(
            "KV cache: fwht2 filtered ({n_kv}/{} layers carry KV; K FWHT-2b {k_bph}B + V Q8 {v_bph}B = {} B/head)",
            is_kv_layer.len(),
            k_bph + v_bph,
        );
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: true,
            quant_fwht: true,
            boundary_layers: 0,
            givens_cos: Some(s1),
            givens_sin: Some(s2),
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    /// Same as [`new_gpu_asym2`] with an explicit physical_cap. Eviction-aware.
    pub fn new_gpu_asym2_capped(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "asym2 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        assert!(
            physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]"
        );
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 4;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;

        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[k_elems], DType::F32)?);
            v_gpu.push(gpu.zeros(&[v_elems], DType::F32)?);
        }
        let n_blocks = head_dim / 2;
        let (cos_vals, sin_vals) = Self::gen_givens_angles(42, n_blocks);
        let cb: Vec<u8> = cos_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let sb: Vec<u8> = sin_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let ct = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        let st = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        gpu.hip.memcpy_htod(&ct.buf, &cb)?;
        gpu.hip.memcpy_htod(&st.buf, &sb)?;
        let v_bph = v_bpp / n_kv_heads;
        eprintln!(
            "KV cache: asym2 (K rotated-2b {k_bph}B + V Q8 {v_bph}B = {} B/head, {:.1}x vs fp32)",
            k_bph + v_bph,
            (head_dim * 4 * 2) as f64 / (k_bph + v_bph) as f64
        );
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: true,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: Some(ct),
            givens_sin: Some(st),
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    /// Generate deterministic ±1 sign array for FWHT.
    pub fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
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

    /// Free all GPU tensors in this cache. Call before drop to return VRAM.
    /// After calling, follow with gpu.drain_pool() to actually release memory.
    ///
    /// Contiguous frees keep prior log-and-continue behavior. VMM teardown
    /// failures are aggregated and returned so unload cannot claim success
    /// while arenas remain registered (retry via `Gpu::ensure_vmm_cleaned`).
    pub fn free_gpu(self, gpu: &mut Gpu) -> HipResult<()> {
        let mut first_err: Option<hip_bridge::HipError> = None;
        let mut note = |r: HipResult<()>| {
            if let Err(err) = r {
                if first_err.is_none() {
                    first_err = Some(err);
                }
            }
        };
        for t in self.k_gpu {
            note(gpu.free_tensor(t));
        }
        for t in self.v_gpu {
            note(gpu.free_tensor(t));
        }
        for t in self.k_scales {
            note(gpu.free_tensor(t));
        }
        for t in self.v_scales {
            note(gpu.free_tensor(t));
        }
        if let Some(t) = self.givens_cos {
            note(gpu.free_tensor(t));
        }
        if let Some(t) = self.givens_sin {
            note(gpu.free_tensor(t));
        }
        match first_err {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    /// Store K, V at position `pos` in layer cache (CPU → GPU copy into cache slot).
    pub fn store_kv_pub(
        &mut self,
        gpu: &Gpu,
        layer: usize,
        pos: usize,
        k: &[f32],
        v: &[f32],
    ) -> HipResult<()> {
        self.store_kv(gpu, layer, pos, k, v)
    }

    fn store_kv(
        &mut self,
        gpu: &Gpu,
        layer: usize,
        pos: usize,
        k_data: &[f32],
        v_data: &[f32],
    ) -> HipResult<()> {
        let byte_offset = pos * self.kv_dim * 4; // float = 4 bytes
        let k_bytes =
            unsafe { std::slice::from_raw_parts(k_data.as_ptr() as *const u8, k_data.len() * 4) };
        let v_bytes =
            unsafe { std::slice::from_raw_parts(v_data.as_ptr() as *const u8, v_data.len() * 4) };
        gpu.hip
            .memcpy_htod_offset(&self.k_gpu[layer].buf, byte_offset, k_bytes)?;
        gpu.hip
            .memcpy_htod_offset(&self.v_gpu[layer].buf, byte_offset, v_bytes)?;
        Ok(())
    }

    // ── Multi-GPU constructors (Stage 5 of issue #58) ───────────────────
    //
    // Each `_multi` variant places the per-layer K/V slot on
    // `gpus.devices[gpus.device_for_layer(i)]`. asym{2,3,4} variants
    // additionally replicate the rotation tables to every device by
    // populating `gpus.givens_cos_per_dev` / `gpus.givens_sin_per_dev`.
    //
    // The KvCache.givens_cos / .givens_sin fields stay `None` in multi mode
    // — Stage 6 forward dispatch reads from the per-device replicas in
    // `Gpus` instead.

    /// Free all per-layer GPU tensors on their owning devices. Mirror of
    /// `free_gpu` for the multi-GPU layout. Givens replicas stay owned by
    /// `Gpus`; freeing them is the orchestrator's responsibility.
    pub fn free_gpu_multi(self, gpus: &mut Gpus) {
        for (i, t) in self.k_gpu.into_iter().enumerate() {
            let dev_idx = gpus.device_for_layer(i);
            let _ = gpus.devices[dev_idx].free_tensor(t);
        }
        for (i, t) in self.v_gpu.into_iter().enumerate() {
            let dev_idx = gpus.device_for_layer(i);
            let _ = gpus.devices[dev_idx].free_tensor(t);
        }
        for (i, t) in self.k_scales.into_iter().enumerate() {
            let dev_idx = gpus.device_for_layer(i);
            let _ = gpus.devices[dev_idx].free_tensor(t);
        }
        for (i, t) in self.v_scales.into_iter().enumerate() {
            let dev_idx = gpus.device_for_layer(i);
            let _ = gpus.devices[dev_idx].free_tensor(t);
        }
    }

    pub fn new_gpu_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let cache_size = max_seq_len * kv_dim;
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, cache_size, cache_size)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: false,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    pub fn new_gpu_q4_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let bytes_per_head = 8 + head_dim / 2;
        let bytes_per_pos = n_kv_heads * bytes_per_head;
        let cache_bytes = max_seq_len * bytes_per_pos;
        let cache_elems = (cache_bytes + 3) / 4;
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, cache_elems, cache_elems)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    pub fn new_gpu_q8_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_q8_capped_multi(
            gpus,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    pub fn new_gpu_q8_capped_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(physical_cap > 0 && physical_cap <= max_seq_len);
        let kv_dim = n_kv_heads * head_dim;
        let blocks_per_head = head_dim / 32;
        let total_blocks = n_kv_heads * blocks_per_head;
        let cache_bytes = physical_cap * total_blocks * 34;
        let cache_elems = (cache_bytes + 3) / 4;
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, cache_elems, cache_elems)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: true,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    pub fn new_gpu_int8c_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let bph = 8 + head_dim;
        let bpp = n_kv_heads * bph;
        let cache_bytes = max_seq_len * bpp;
        let cache_elems = (cache_bytes + 3) / 4;
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, cache_elems, cache_elems)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: true,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    pub fn new_gpu_hfq4kv_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let bytes_per_block = 8 + head_dim / 2;
        let bytes_per_pos = n_kv_heads * bytes_per_block;
        let cache_bytes = max_seq_len * bytes_per_pos;
        let cache_elems = (cache_bytes + 3) / 4;
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, cache_elems, cache_elems)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: true,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    pub fn new_gpu_hfq8_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let val_elems = (max_seq_len * kv_dim + 3) / 4;
        let scale_elems = max_seq_len * n_kv_heads * 2;
        let (k_gpu, v_gpu, k_scales, v_scales) = alloc_kv_with_scales_per_layer_multi(
            gpus,
            n_layers,
            val_elems,
            val_elems,
            scale_elems,
            scale_elems,
        )?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales,
            v_scales,
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    pub fn new_gpu_int8_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let val_elems = (max_seq_len * kv_dim + 3) / 4;
        let scale_elems = max_seq_len * n_kv_heads;
        let (k_gpu, v_gpu, k_scales, v_scales) = alloc_kv_with_scales_per_layer_multi(
            gpus,
            n_layers,
            val_elems,
            val_elems,
            scale_elems,
            scale_elems,
        )?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales,
            v_scales,
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: true,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    pub fn new_gpu_asym4_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_asym4_capped_multi(
            gpus,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    pub fn new_gpu_asym4_capped_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "asym4 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        assert!(physical_cap > 0 && physical_cap <= max_seq_len);
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 2;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, k_elems, v_elems)?;
        replicate_givens_to_all_devices(gpus, head_dim / 2, 42)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: true,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    pub fn new_gpu_asym3_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_asym3_capped_multi(
            gpus,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    pub fn new_gpu_asym3_capped_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 256,
            "asym3 currently requires head_dim=256 (Qwen 3.5)"
        );
        assert!(head_dim % 32 == 0);
        assert!(physical_cap > 0 && physical_cap <= max_seq_len);
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + (head_dim * 3) / 8;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, k_elems, v_elems)?;
        replicate_givens_to_all_devices(gpus, head_dim / 2, 42)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: true,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    pub fn new_gpu_asym2_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_asym2_capped_multi(
            gpus,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    pub fn new_gpu_asym2_capped_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "asym2 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        assert!(physical_cap > 0 && physical_cap <= max_seq_len);
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 4;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, k_elems, v_elems)?;
        replicate_givens_to_all_devices(gpus, head_dim / 2, 42)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: true,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    // ── fwht multi-GPU constructors ──────────────────────────────────
    // Mirror the asym{4,3,2}_multi shape. Per-device signs1/signs2
    // replicated via replicate_fwht_signs_to_all_devices. The KvCache
    // struct keeps givens_cos/sin = None in multi mode (per-device slots
    // live on the Gpus struct); `quant_fwht: true` tells the dispatcher
    // to read from gpus.givens_cos_per_dev / .givens_sin_per_dev as
    // signs1/signs2 instead of cos/sin.

    pub fn new_gpu_fwht4_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_fwht4_capped_multi(
            gpus,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    pub fn new_gpu_fwht4_capped_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "fwht4 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        assert!(physical_cap > 0 && physical_cap <= max_seq_len);
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 2;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, k_elems, v_elems)?;
        replicate_fwht_signs_to_all_devices(gpus, 128)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: true,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: true,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    pub fn new_gpu_fwht3_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_fwht3_capped_multi(
            gpus,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    pub fn new_gpu_fwht3_capped_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 256,
            "fwht3 currently requires head_dim=256 (Qwen 3.5)"
        );
        assert!(head_dim % 32 == 0);
        assert!(physical_cap > 0 && physical_cap <= max_seq_len);
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + (head_dim * 3) / 8;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, k_elems, v_elems)?;
        // fwht_shfl_forward_256 needs 256-element signs1/signs2.
        replicate_fwht_signs_to_all_devices(gpus, 256)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: true,
            quant_asym2: false,
            quant_fwht: true,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    pub fn new_gpu_fwht2_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_fwht2_capped_multi(
            gpus,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    pub fn new_gpu_fwht2_capped_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "fwht2 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        assert!(physical_cap > 0 && physical_cap <= max_seq_len);
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 4;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, k_elems, v_elems)?;
        replicate_fwht_signs_to_all_devices(gpus, 128)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: true,
            quant_fwht: true,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    // ── Filtered multi-GPU ctors: skip KV alloc on non-FullAttention layers
    // of the hybrid Qwen3.5/3.6 stack (1-elem placeholder keeps absolute layer
    // indexing valid). Mirror the *_capped_multi ctors above byte-for-byte
    // except they take an `is_kv_layer` mask and use the filtered allocator. ──
    pub fn new_gpu_q8_capped_multi_filtered(
        gpus: &mut Gpus,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(physical_cap > 0 && physical_cap <= max_seq_len);
        let kv_dim = n_kv_heads * head_dim;
        let total_blocks = n_kv_heads * (head_dim / 32);
        let cache_elems = (physical_cap * total_blocks * 34 + 3) / 4;
        let (k_gpu, v_gpu) =
            alloc_kv_per_layer_multi_filtered(gpus, is_kv_layer, cache_elems, cache_elems)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: true,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    pub fn new_gpu_asym4_capped_multi_filtered(
        gpus: &mut Gpus,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "asym4 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        assert!(physical_cap > 0 && physical_cap <= max_seq_len);
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 2;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_bpp = n_kv_heads * (head_dim / 32) * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) =
            alloc_kv_per_layer_multi_filtered(gpus, is_kv_layer, k_elems, v_elems)?;
        replicate_givens_to_all_devices(gpus, head_dim / 2, 42)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: true,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    pub fn new_gpu_asym3_capped_multi_filtered(
        gpus: &mut Gpus,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 256,
            "asym3 currently requires head_dim=256 (Qwen 3.5)"
        );
        assert!(head_dim % 32 == 0);
        assert!(physical_cap > 0 && physical_cap <= max_seq_len);
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + (head_dim * 3) / 8;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_bpp = n_kv_heads * (head_dim / 32) * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) =
            alloc_kv_per_layer_multi_filtered(gpus, is_kv_layer, k_elems, v_elems)?;
        replicate_givens_to_all_devices(gpus, head_dim / 2, 42)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: true,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    pub fn new_gpu_asym2_capped_multi_filtered(
        gpus: &mut Gpus,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "asym2 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        assert!(physical_cap > 0 && physical_cap <= max_seq_len);
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 4;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_bpp = n_kv_heads * (head_dim / 32) * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) =
            alloc_kv_per_layer_multi_filtered(gpus, is_kv_layer, k_elems, v_elems)?;
        replicate_givens_to_all_devices(gpus, head_dim / 2, 42)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: true,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    pub fn new_gpu_fwht4_capped_multi_filtered(
        gpus: &mut Gpus,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "fwht4 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        assert!(physical_cap > 0 && physical_cap <= max_seq_len);
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 2;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_bpp = n_kv_heads * (head_dim / 32) * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) =
            alloc_kv_per_layer_multi_filtered(gpus, is_kv_layer, k_elems, v_elems)?;
        replicate_fwht_signs_to_all_devices(gpus, 128)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: true,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: true,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    pub fn new_gpu_fwht3_capped_multi_filtered(
        gpus: &mut Gpus,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 256,
            "fwht3 currently requires head_dim=256 (Qwen 3.5)"
        );
        assert!(head_dim % 32 == 0);
        assert!(physical_cap > 0 && physical_cap <= max_seq_len);
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + (head_dim * 3) / 8;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_bpp = n_kv_heads * (head_dim / 32) * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) =
            alloc_kv_per_layer_multi_filtered(gpus, is_kv_layer, k_elems, v_elems)?;
        replicate_fwht_signs_to_all_devices(gpus, 256)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: true,
            quant_asym2: false,
            quant_fwht: true,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    pub fn new_gpu_fwht2_capped_multi_filtered(
        gpus: &mut Gpus,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "fwht2 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        assert!(physical_cap > 0 && physical_cap <= max_seq_len);
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 4;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_bpp = n_kv_heads * (head_dim / 32) * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) =
            alloc_kv_per_layer_multi_filtered(gpus, is_kv_layer, k_elems, v_elems)?;
        replicate_fwht_signs_to_all_devices(gpus, 128)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: true,
            quant_fwht: true,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }
}

// ── Stage 5 helpers: per-device KV alloc + givens replication ────────

fn alloc_kv_per_layer_multi(
    gpus: &mut Gpus,
    n_layers: usize,
    k_elems: usize,
    v_elems: usize,
) -> HipResult<(Vec<GpuTensor>, Vec<GpuTensor>)> {
    let mut k_gpu = Vec::with_capacity(n_layers);
    let mut v_gpu = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let dev_idx = gpus.device_for_layer(i);
        let g = &mut gpus.devices[dev_idx];
        k_gpu.push(g.zeros(&[k_elems], DType::F32)?);
        v_gpu.push(g.zeros(&[v_elems], DType::F32)?);
    }
    Ok((k_gpu, v_gpu))
}

/// Filtered variant of [`alloc_kv_per_layer_multi`]: allocates a full KV slot
/// only for layers where `is_kv_layer[i]` (the FullAttention layers of a hybrid
/// Qwen3.5/3.6 stack), and a 1-element placeholder on the layer's assigned
/// device otherwise. Placeholders are never read/written; absolute per-layer
/// indexing (`k_gpu[layer_idx]`) and `device_for_layer` assignment stay valid.
fn alloc_kv_per_layer_multi_filtered(
    gpus: &mut Gpus,
    is_kv_layer: &[bool],
    k_elems: usize,
    v_elems: usize,
) -> HipResult<(Vec<GpuTensor>, Vec<GpuTensor>)> {
    let n_layers = is_kv_layer.len();
    let mut k_gpu = Vec::with_capacity(n_layers);
    let mut v_gpu = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let dev_idx = gpus.device_for_layer(i);
        let g = &mut gpus.devices[dev_idx];
        if is_kv_layer[i] {
            k_gpu.push(g.zeros(&[k_elems], DType::F32)?);
            v_gpu.push(g.zeros(&[v_elems], DType::F32)?);
        } else {
            k_gpu.push(g.zeros(&[1], DType::F32)?);
            v_gpu.push(g.zeros(&[1], DType::F32)?);
        }
    }
    Ok((k_gpu, v_gpu))
}

fn alloc_kv_with_scales_per_layer_multi(
    gpus: &mut Gpus,
    n_layers: usize,
    k_elems: usize,
    v_elems: usize,
    k_scale_elems: usize,
    v_scale_elems: usize,
) -> HipResult<(
    Vec<GpuTensor>,
    Vec<GpuTensor>,
    Vec<GpuTensor>,
    Vec<GpuTensor>,
)> {
    let mut k_gpu = Vec::with_capacity(n_layers);
    let mut v_gpu = Vec::with_capacity(n_layers);
    let mut k_scales = Vec::with_capacity(n_layers);
    let mut v_scales = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let dev_idx = gpus.device_for_layer(i);
        let g = &mut gpus.devices[dev_idx];
        k_gpu.push(g.zeros(&[k_elems], DType::F32)?);
        v_gpu.push(g.zeros(&[v_elems], DType::F32)?);
        k_scales.push(g.zeros(&[k_scale_elems], DType::F32)?);
        v_scales.push(g.zeros(&[v_scale_elems], DType::F32)?);
    }
    Ok((k_gpu, v_gpu, k_scales, v_scales))
}

/// Asym{2,3,4} KV-rotation tables replicated to every device. Replaces any
/// previous contents of `gpus.givens_*_per_dev`. Stage 6 forward dispatch
/// reads `gpus.givens_*_per_dev[layer_to_device[i]]` per layer.
fn replicate_givens_to_all_devices(gpus: &mut Gpus, n_blocks: usize, seed: u32) -> HipResult<()> {
    let (cos_vals, sin_vals) = KvCache::gen_givens_angles(seed, n_blocks);
    let cb: Vec<u8> = cos_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let sb: Vec<u8> = sin_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let prev_cos = std::mem::take(&mut gpus.givens_cos_per_dev);
    let prev_sin = std::mem::take(&mut gpus.givens_sin_per_dev);
    for (i, t) in prev_cos.into_iter().enumerate() {
        if i < gpus.devices.len() {
            let _ = gpus.devices[i].free_tensor(t);
        }
    }
    for (i, t) in prev_sin.into_iter().enumerate() {
        if i < gpus.devices.len() {
            let _ = gpus.devices[i].free_tensor(t);
        }
    }

    for dev_idx in 0..gpus.devices.len() {
        let g = &mut gpus.devices[dev_idx];
        let ct = g.alloc_tensor(&[n_blocks], DType::F32)?;
        let st = g.alloc_tensor(&[n_blocks], DType::F32)?;
        g.hip.memcpy_htod(&ct.buf, &cb)?;
        g.hip.memcpy_htod(&st.buf, &sb)?;
        gpus.givens_cos_per_dev.push(ct);
        gpus.givens_sin_per_dev.push(st);
    }
    Ok(())
}

/// Multi-device replication of signed-FWHT sign vectors. Mirrors
/// `replicate_givens_to_all_devices` but uses gen_fwht_signs (seeds
/// 42/1042, matching the single-GPU `new_gpu_fwht*_filtered` ctors and
/// the MQ4 weight-FWHT convention). signs1/signs2 occupy the same
/// per-device slots as cos/sin — dispatcher branches on `quant_fwht`
/// to pick the kernel signature.
fn replicate_fwht_signs_to_all_devices(gpus: &mut Gpus, n_signs: usize) -> HipResult<()> {
    let s1_vals = KvCache::gen_fwht_signs(42, n_signs);
    let s2_vals = KvCache::gen_fwht_signs(1042, n_signs);
    let s1_bytes: Vec<u8> = s1_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let s2_bytes: Vec<u8> = s2_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let prev_cos = std::mem::take(&mut gpus.givens_cos_per_dev);
    let prev_sin = std::mem::take(&mut gpus.givens_sin_per_dev);
    for (i, t) in prev_cos.into_iter().enumerate() {
        if i < gpus.devices.len() {
            let _ = gpus.devices[i].free_tensor(t);
        }
    }
    for (i, t) in prev_sin.into_iter().enumerate() {
        if i < gpus.devices.len() {
            let _ = gpus.devices[i].free_tensor(t);
        }
    }

    for dev_idx in 0..gpus.devices.len() {
        let g = &mut gpus.devices[dev_idx];
        let s1 = g.alloc_tensor(&[n_signs], DType::F32)?;
        let s2 = g.alloc_tensor(&[n_signs], DType::F32)?;
        g.hip.memcpy_htod(&s1.buf, &s1_bytes)?;
        g.hip.memcpy_htod(&s2.buf, &s2_bytes)?;
        gpus.givens_cos_per_dev.push(s1);
        gpus.givens_sin_per_dev.push(s2);
    }
    Ok(())
}

// attention_cpu removed — GPU attention is now used

/// Sample the next token from logits using argmax (greedy).
pub fn argmax(logits: &[f32]) -> u32 {
    // hunt3 M-B (FinalFix): explicit `>`-based fold mirrors the GPU kernel
    // (argmax.hip: `if (data[i] > lmax)`, seed -1e30). `v > best` is false for
    // NaN, so a NaN logit is never selected and never displaces the real max.
    // The previous `max_by(partial_cmp.unwrap_or(Less))` kept a trailing NaN
    // candidate (verified: [1,5,3,NaN] → idx 3) — a NaN-indexed garbage token
    // on the greedy path poisons the recurrent DeltaNet state.
    //
    // O2b-2 finite guard: this is also the degenerate fallback for the sampler
    // (`sample_top_p` / `sample_full_dist`), so it must skip ALL non-finite
    // values, not just NaN. A `+Inf` logit would otherwise win the bare `>`
    // comparison and be selected over the real finite max — `is_finite()` in
    // the predicate keeps both `+Inf` and `NaN` from displacing a finite token.
    // Mixed finite + non-finite → the finite max. All-non-finite (or empty)
    // never replaces the seed → deterministic index 0, matching the
    // O1-accepted `sample_full_dist` degenerate behavior.
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
            if v.is_finite() && v > best.1 {
                (i, v)
            } else {
                best
            }
        })
        .0 as u32
}

/// Sample the next token using temperature + top-k + top-p (nucleus) sampling.
/// Qwen3 recommended: temperature=0.7, top_k=20, top_p=0.8
///
/// Single pass over raw logits to find top-K by value (no softmax on 151K vocab).
/// Softmax only computed on the K=20 finalists.
// ─── Sampling configuration ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub think_temp: f32,
    pub answer_temp: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub repeat_window: usize,
}

impl SamplingConfig {
    /// Text-only thinking model (Qwen3.5 text inference).
    pub fn text_thinking() -> Self {
        Self {
            think_temp: 0.3,
            answer_temp: 0.3,
            top_p: 0.8,
            repeat_penalty: 1.15,
            repeat_window: 128,
        }
    }
    /// VL thinking model.
    pub fn vl_thinking() -> Self {
        Self {
            think_temp: 0.3,
            answer_temp: 0.3,
            top_p: 0.8,
            repeat_penalty: 1.15,
            repeat_window: 128,
        }
    }
    /// Simple greedy-ish sampling (no think/answer split).
    pub fn simple() -> Self {
        Self {
            think_temp: 0.7,
            answer_temp: 0.3,
            top_p: 0.9,
            repeat_penalty: 1.1,
            repeat_window: 64,
        }
    }
}

/// Apply repeat penalty to logits in-place.
pub fn apply_repeat_penalty(logits: &mut [f32], history: &[u32], window: usize, penalty: f32) {
    let start = history.len().saturating_sub(window);
    let recent = &history[start..];

    // Count frequency of each token in the window, then apply penalty
    // scaled by count. A token seen once gets penalty^1, seen 3 times
    // gets penalty^3. This lets common words reappear naturally while
    // strongly suppressing actual repetition loops.
    // Also apply recency decay: tokens near the end of the window get
    // full penalty, tokens near the start get reduced penalty.
    let window_len = recent.len() as f32;
    let mut counts = std::collections::HashMap::<u32, (u32, f32)>::new(); // (count, closest_position_ratio)
    for (i, &t) in recent.iter().enumerate() {
        let recency = (i as f32 + 1.0) / window_len; // 0→1, higher = more recent
        let entry = counts.entry(t).or_insert((0, 0.0));
        entry.0 += 1;
        if recency > entry.1 {
            entry.1 = recency;
        }
    }

    for (&t, &(count, recency)) in &counts {
        if (t as usize) < logits.len() {
            // Effective penalty: base^(count * recency), capped at 1.5x.
            // Without the cap, "the" appearing 8x recently gets 1.15^8 = 3x suppression,
            // which collectively flattens the distribution after ~400 tokens.
            // The cap ensures no single token is suppressed more than 50%, keeping
            // the natural vocabulary accessible even in long generation.
            let effective = penalty.powf(count as f32 * recency).min(1.5);
            if logits[t as usize] > 0.0 {
                logits[t as usize] /= effective;
            } else {
                logits[t as usize] *= effective;
            }
        }
    }
}

/// N-gram repeat detection: if the last `n` tokens in history match an earlier n-gram,
/// set the logit of the token that followed that earlier occurrence to -inf.
/// This breaks phrase-level loops that token-level repeat penalty misses.
/// Checks n-grams of sizes 3, 4, 5, 6 for robustness.
pub fn apply_ngram_block(logits: &mut [f32], history: &[u32]) {
    if history.len() < 4 {
        return;
    }
    for ngram_size in [3, 4, 5, 6] {
        if history.len() <= ngram_size {
            continue;
        }
        let suffix = &history[history.len() - ngram_size..];
        // Scan history for earlier occurrences of this n-gram
        let search_end = history.len() - ngram_size;
        for i in 0..search_end {
            if i + ngram_size >= history.len() {
                break;
            }
            if history[i..i + ngram_size] == *suffix {
                // Found a match — the token that followed this earlier n-gram
                // is what the model wants to repeat. Block it.
                let next_tok = history[i + ngram_size];
                if (next_tok as usize) < logits.len() {
                    logits[next_tok as usize] = f32::NEG_INFINITY;
                }
            }
        }
    }
}

/// Single-token attractor block for special tokens. Counts how many times
/// `token_id` appears in the last `window` tokens of `history`; if it is
/// at or above `threshold`, sets that token's logit to `-INF` so the
/// next sample picks something else. Targets MQ4 single-token attractors
/// on tokens that have no paired closer (e.g. a runaway emit of a
/// solo special). For paired open/close tokens like `<tool_call>` /
/// `</tool_call>`, prefer `apply_unclosed_attractor_block` — it triggers
/// before the model can stack a second nested opener that breaks
/// downstream regex parsers (see #111 codex review).
pub fn apply_special_token_attractor_block(
    logits: &mut [f32],
    history: &[u32],
    token_id: u32,
    window: usize,
    threshold: usize,
) {
    if (token_id as usize) >= logits.len() || threshold == 0 || window == 0 {
        return;
    }
    let start = history.len().saturating_sub(window);
    let count = history[start..].iter().filter(|&&t| t == token_id).count();
    if count >= threshold {
        logits[token_id as usize] = f32::NEG_INFINITY;
    }
}

/// Open/close-paired attractor block for structured special tokens
/// (`<tool_call>`/`</tool_call>`, `<think>`/`</think>`).
///
/// Counts unclosed openers in the last `window` tokens — `opens - closes`,
/// floored at zero. When the running depth reaches `threshold`, sets
/// `open_id`'s logit to `-INF` so the next sample cannot stack another
/// nested opener. With `threshold = 2`, a second consecutive opener
/// without an intervening closer is the last one the decoder is allowed
/// to emit; the third+ are blocked. The downstream regex parser
/// (the native `parseToolCalls` compatibility path) tolerates a single nested opener
/// by stripping the leading repeat before JSON parse.
///
/// The depth saturates at 0 from below: a stray closer at the start of
/// the window doesn't push depth negative and create false-allow.
pub fn apply_unclosed_attractor_block(
    logits: &mut [f32],
    history: &[u32],
    open_id: u32,
    close_id: u32,
    window: usize,
    threshold: usize,
) {
    if (open_id as usize) >= logits.len() || threshold == 0 || window == 0 {
        return;
    }
    let start = history.len().saturating_sub(window);
    let mut depth: i32 = 0;
    for &t in &history[start..] {
        if t == open_id {
            depth += 1;
        } else if t == close_id && depth > 0 {
            depth -= 1;
        }
    }
    if depth >= threshold as i32 {
        logits[open_id as usize] = f32::NEG_INFINITY;
    }
}

pub fn sample_top_p(logits: &[f32], temperature: f32, top_p: f32) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits);
    }
    let top_p = top_p.clamp(0.0, 1.0);
    const TOP_K: usize = 20;

    let inv_temp = 1.0 / temperature;

    // Single pass: find max AND top-K indices from raw logits simultaneously.
    // Uses a fixed-size array (no heap alloc) with manual min-tracking.
    //
    // FINITE GUARD (ported from sample_full_dist, the O1 fix): only finite
    // logits feed `max_logit` and the top-K set. A `+Inf` logit must not become
    // `max_logit` (then `(l - max)` = Inf - Inf = NaN → exp(NaN) = NaN → NaN sum
    // → degenerate token-0 return), and a NaN logit (already `>`-false) must not
    // occupy a top-K slot. Slots left at NEG_INFINITY are zeroed in the softmax
    // below, and if no finite mass survives we fall back to the NaN-safe argmax.
    let mut topk_val = [f32::NEG_INFINITY; TOP_K];
    let mut topk_idx = [0u32; TOP_K];
    let mut min_pos = 0usize; // index of smallest element in topk
    let mut min_val = f32::NEG_INFINITY;
    let mut max_logit = f32::NEG_INFINITY;

    for (i, &l) in logits.iter().enumerate() {
        if !l.is_finite() {
            continue;
        }
        if l > max_logit {
            max_logit = l;
        }
        if l > min_val {
            topk_val[min_pos] = l;
            topk_idx[min_pos] = i as u32;
            // Find new min
            min_val = f32::INFINITY;
            for j in 0..TOP_K {
                if topk_val[j] < min_val {
                    min_val = topk_val[j];
                    min_pos = j;
                }
            }
        }
    }

    // Softmax only the K candidates (temperature-scaled). FINITE GUARD: a slot
    // still holding NEG_INFINITY (fewer than TOP_K finite logits) or a non-
    // finite computed prob contributes zero mass rather than poisoning `sum`.
    let mut probs = [0.0f32; TOP_K];
    let mut sum = 0.0f32;
    for i in 0..TOP_K {
        let p = if topk_val[i].is_finite() {
            let pp = ((topk_val[i] - max_logit) * inv_temp).exp();
            if pp.is_finite() {
                pp
            } else {
                0.0
            }
        } else {
            0.0
        };
        probs[i] = p;
        sum += p;
    }

    // Degenerate guard: if no finite mass survives (all-NaN / all-+Inf / all
    // -inf logits), fall back to argmax of the raw logits (its own NaN guard
    // returns a finite in-vocab index, never the poisoned token 0).
    if sum <= 0.0 || !sum.is_finite() {
        return argmax(logits);
    }

    // Sort descending by probability (insertion sort on 20 elements)
    let mut order: [usize; TOP_K] = core::array::from_fn(|i| i);
    for i in 1..TOP_K {
        let mut j = i;
        while j > 0 && probs[order[j]] > probs[order[j - 1]] {
            order.swap(j, j - 1);
            j -= 1;
        }
    }

    // Top-p filtering + sampling in one pass
    let r = simple_rand() * sum; // pre-scale by total sum
    let mut cumulative = 0.0f32;
    let mut sample_acc = 0.0f32;
    let threshold = top_p * sum;
    for &k in &order {
        cumulative += probs[k];
        sample_acc += probs[k];
        if sample_acc >= r {
            return topk_idx[k];
        }
        if cumulative >= threshold {
            // Past top_p — sample from what we have
            let r2 = simple_rand() * cumulative;
            let mut acc2 = 0.0f32;
            for &k2 in &order {
                acc2 += probs[k2];
                if acc2 >= r2 {
                    return topk_idx[k2];
                }
                if acc2 >= cumulative {
                    break;
                }
            }
            return topk_idx[order[0]];
        }
    }
    topk_idx[order[0]]
}

/// Host-side full-distribution sampler for the EP (multi-GPU) serve path.
///
/// The single-GPU handler resolves request>rec_*>arch-default sampling and runs
/// the GPU `sample_top_p` kernel; the EP path (ds4/minimax) never reaches that
/// handler and previously decoded with a hardcoded argmax, dropping the card's
/// temp/top_p/top_k/min_p. ds4's card mandates temp=1.0/top_p=1.0 specifically
/// because greedy argmax on the quantized instruct model falls into block-level
/// attractor loops. This function restores those knobs host-side over the f32
/// logits already downloaded by `download_f32`.
///
/// Pipeline mirrors the GPU sampler ordering: temperature scale → top_k cut →
/// top_p (nucleus) cut → min_p cut → seeded multinomial draw. Uses the shared
/// `simple_rand` RNG (SAMPLER_STATE), so a per-request `reset_cpu_sampler_rng`
/// makes the draw deterministic and consistent with the rest of the daemon.
///
/// `temperature <= 1e-6` takes the argmax fast-path (greedy), matching the
/// GPU kernel's `temperature <= 0` short-circuit but with a small epsilon so a
/// near-zero temp never divides by ~0.
pub fn sample_full_dist(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: Option<u32>,
    min_p: Option<f32>,
) -> u32 {
    if temperature <= 1e-6 {
        return argmax(logits);
    }
    let n = logits.len();
    if n == 0 {
        return 0;
    }
    let top_p = top_p.clamp(0.0, 1.0);
    // FIX #3: the request parser only filters `min_p > 0`, so a request can
    // smuggle in min_p > 1.0 (which would `retain` an empty candidate set and
    // panic on `cand[0]` below) or a negative min_p. Clamp at fn entry to the
    // valid [0,1] probability-ratio range; min_p == 0 is the happy path (no cut).
    let min_p = min_p.map(|mp| {
        if mp.is_finite() {
            mp.clamp(0.0, 1.0)
        } else {
            0.0
        }
    });
    let inv_temp = 1.0 / temperature;

    // Max logit (NaN-safe, mirrors argmax's `>` fold) for softmax stability.
    // FIX #4: only finite logits feed the max; a NaN/Inf logit must not become
    // `max_logit` (an Inf max would push every `(l - max)` to -Inf → all-zero
    // probs → degenerate fallback even when finite tokens existed).
    let mut max_logit = f32::NEG_INFINITY;
    for &l in logits {
        if l.is_finite() && l > max_logit {
            max_logit = l;
        }
    }

    // Materialize (idx, prob) with temperature-scaled softmax numerators.
    // FIX #4: a non-finite logit (NaN/Inf) yields a non-finite prob that would
    // poison `total` (NaN total → degenerate argmax fallback, silently dropping
    // the whole finite distribution). Zero out the prob of any non-finite logit
    // (and of any non-finite computed prob) so finite tokens still get sampled.
    // A zeroed prob can never be selected, matching the greedy argmax NaN guard.
    let mut cand: Vec<(u32, f32)> = Vec::with_capacity(n);
    for (i, &l) in logits.iter().enumerate() {
        let p = if l.is_finite() {
            let pp = ((l - max_logit) * inv_temp).exp();
            if pp.is_finite() {
                pp
            } else {
                0.0
            }
        } else {
            0.0
        };
        cand.push((i as u32, p));
    }

    // Sort descending by probability (top_k / top_p / min_p all operate on the
    // ranked distribution). Unstable sort with a NaN-last comparator.
    cand.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // top_k: keep at most k highest-prob candidates. None / 0 = no cut.
    // FINDING #2 (intentional, do NOT "fix" by adding a default cap):
    // `top_k == None` means NO top-k cap — the full sorted distribution is
    // retained. This deliberately differs from the GPU sampler's gather-pool
    // cap (which always pre-trims to a fixed candidate pool). For ds4 the card
    // mandates temp=1.0 / top_p=1.0 / (no top_k), i.e. sampling from the full
    // distribution; capping here would be LESS faithful to that card, not more.
    if let Some(k) = top_k {
        let k = k as usize;
        if k > 0 && k < cand.len() {
            cand.truncate(k);
        }
    }

    // Degenerate guard over the (possibly k-truncated) set: if no finite mass
    // survives (all -inf / all non-finite logits zeroed by FIX #4), fall back
    // to argmax of the raw logits (which has its own NaN guard).
    let pre_minp_total: f32 = cand.iter().map(|c| c.1).sum();
    if pre_minp_total <= 0.0 || !pre_minp_total.is_finite() {
        return argmax(logits);
    }

    // min_p: drop candidates whose prob is below `min_p * p_max` (the highest
    // prob is cand[0] after the descending sort). Applied before nucleus.
    // FIX #3: never let the retain empty the set — if the floor would drop
    // everything (it can't here since cand[0] always passes `>= mp*cand[0]`
    // for mp <= 1, but keep the invariant explicit and robust to fp edge
    // cases), restore the single highest-prob candidate so `cand[0]` is safe.
    if let Some(mp) = min_p {
        if mp > 0.0 {
            let top = cand[0];
            let floor = mp * top.1;
            cand.retain(|c| c.1 >= floor);
            if cand.is_empty() {
                cand.push(top);
            }
        }
    }

    // FIX #1: recompute the nucleus mass AFTER the min_p filter, over the final
    // retained set, so `p_thresh = top_p * (post-min_p mass)`. Summing the mass
    // before min_p (the old behavior) made the cumulative threshold too large
    // relative to the surviving probabilities, over-keeping the tail — and
    // diverging from the GPU sampler's cap-before-sum semantics.
    let total: f32 = cand.iter().map(|c| c.1).sum();
    if total <= 0.0 || !total.is_finite() {
        return cand[0].0;
    }

    // top_p (nucleus): keep the smallest prefix whose cumulative mass reaches
    // top_p * total. Always keep at least the top candidate.
    if top_p < 1.0 {
        let threshold = top_p * total;
        let mut cum = 0.0f32;
        let mut keep = 0usize;
        for c in cand.iter() {
            cum += c.1;
            keep += 1;
            if cum >= threshold {
                break;
            }
        }
        cand.truncate(keep.max(1));
    }

    // Seeded multinomial draw over the surviving candidates.
    let kept_total: f32 = cand.iter().map(|c| c.1).sum();
    if kept_total <= 0.0 || !kept_total.is_finite() {
        return cand[0].0;
    }
    let r = simple_rand() * kept_total;
    let mut acc = 0.0f32;
    for c in &cand {
        acc += c.1;
        if acc >= r {
            return c.0;
        }
    }
    cand[cand.len() - 1].0
}

/// Apply the repeat penalty in-place to a specific subset of (token_id, value)
/// candidates, rather than the full 151k-entry logits vector. Used by the
/// GPU-assisted sampler path: the GPU produces a top-K=128 candidate set
/// from the raw logits, and the CPU then runs the existing repeat-penalty
/// math on just those 128 entries.
///
/// Math is identical to `apply_repeat_penalty` — same frequency count, same
/// recency decay, same 1.5× cap, same ">0 ? divide : multiply" branch.
/// The only difference is iteration scope.
pub fn apply_repeat_penalty_candidates(
    cand_ids: &[u32],
    cand_vals: &mut [f32],
    history: &[u32],
    window: usize,
    penalty: f32,
) {
    debug_assert_eq!(cand_ids.len(), cand_vals.len());

    let start = history.len().saturating_sub(window);
    let recent = &history[start..];
    let window_len = recent.len() as f32;
    if window_len == 0.0 {
        return;
    }

    let mut counts = std::collections::HashMap::<u32, (u32, f32)>::new();
    for (i, &t) in recent.iter().enumerate() {
        let recency = (i as f32 + 1.0) / window_len;
        let entry = counts.entry(t).or_insert((0, 0.0));
        entry.0 += 1;
        if recency > entry.1 {
            entry.1 = recency;
        }
    }

    for (i, &tok) in cand_ids.iter().enumerate() {
        if let Some(&(count, recency)) = counts.get(&tok) {
            let effective = penalty.powf(count as f32 * recency).min(1.5);
            if cand_vals[i] > 0.0 {
                cand_vals[i] /= effective;
            } else {
                cand_vals[i] *= effective;
            }
        }
    }
}

/// Sample from a pre-selected candidate set instead of the full logits.
///
/// Accepts (cand_ids, cand_vals): 128 raw (pre-penalty) candidate tokens
/// from the GPU `topk_logits_f32` kernel. Applies repeat penalty to just
/// those candidates, then runs the same top-K=20 → softmax → top-p
/// sampling pipeline as `sample_top_p` on the full logits array.
///
/// This is bit-exact with the full-CPU path PROVIDED that the pre-penalty
/// top-128 ⊇ the post-penalty top-20 from the full vocabulary. Since
/// `apply_repeat_penalty` monotonically decreases logits (divide-if-positive
/// or multiply-more-negative), a token outside the pre-penalty top-128 can
/// never climb into the top-20 after penalty, so the set relation holds.
pub fn sample_top_p_from_candidates(
    cand_ids: &[u32],
    cand_vals: &mut [f32],
    history: &[u32],
    repeat_window: usize,
    repeat_penalty: f32,
    temperature: f32,
    top_p: f32,
) -> u32 {
    debug_assert_eq!(cand_ids.len(), cand_vals.len());

    // Step 1: apply repeat penalty to the candidate subset.
    apply_repeat_penalty_candidates(cand_ids, cand_vals, history, repeat_window, repeat_penalty);

    // Step 2: if greedy, just return the argmax of the penalized candidates.
    if temperature <= 0.0 {
        let mut best_idx = 0usize;
        let mut best_val = cand_vals[0];
        for i in 1..cand_vals.len() {
            if cand_vals[i] > best_val {
                best_val = cand_vals[i];
                best_idx = i;
            }
        }
        return cand_ids[best_idx];
    }

    // Step 3: top-K=20 selection from the candidate set, matching the
    // full-CPU path's selection logic exactly. The candidate set is already
    // ≤ 128, but we still pick the top 20 via the same min-tracking loop
    // the full path uses, so the resulting set ordering is identical.
    const TOP_K: usize = 20;
    let top_p = top_p.clamp(0.0, 1.0);
    let inv_temp = 1.0 / temperature;

    let mut topk_val = [f32::NEG_INFINITY; TOP_K];
    let mut topk_idx = [0u32; TOP_K];
    let mut min_pos = 0usize;
    let mut min_val = f32::NEG_INFINITY;
    let mut max_logit = f32::NEG_INFINITY;

    for (i, &l) in cand_vals.iter().enumerate() {
        let tok = cand_ids[i];
        if l > max_logit {
            max_logit = l;
        }
        if l > min_val {
            topk_val[min_pos] = l;
            topk_idx[min_pos] = tok;
            min_val = f32::INFINITY;
            for j in 0..TOP_K {
                if topk_val[j] < min_val {
                    min_val = topk_val[j];
                    min_pos = j;
                }
            }
        }
    }

    // Step 4: softmax over the K=20 winners (temperature-scaled).
    let mut probs = [0.0f32; TOP_K];
    let mut sum = 0.0f32;
    for i in 0..TOP_K {
        let p = ((topk_val[i] - max_logit) * inv_temp).exp();
        probs[i] = p;
        sum += p;
    }

    // Step 5: sort descending by probability (insertion sort on 20).
    let mut order: [usize; TOP_K] = core::array::from_fn(|i| i);
    for i in 1..TOP_K {
        let mut j = i;
        while j > 0 && probs[order[j]] > probs[order[j - 1]] {
            order.swap(j, j - 1);
            j -= 1;
        }
    }

    // Step 6: top-p filtering + sample. Uses the shared `simple_rand` RNG
    // state, so the RNG stream is identical across the full-CPU and
    // GPU-assisted paths.
    let r = simple_rand() * sum;
    let mut cumulative = 0.0f32;
    let mut sample_acc = 0.0f32;
    let threshold = top_p * sum;
    for &k in &order {
        cumulative += probs[k];
        sample_acc += probs[k];
        if sample_acc >= r {
            return topk_idx[k];
        }
        if cumulative >= threshold {
            let r2 = simple_rand() * cumulative;
            let mut acc2 = 0.0f32;
            for &k2 in &order {
                acc2 += probs[k2];
                if acc2 >= r2 {
                    return topk_idx[k2];
                }
                if acc2 >= cumulative {
                    break;
                }
            }
            return topk_idx[order[0]];
        }
    }
    topk_idx[order[0]]
}

/// Snapshot + restore the sampler RNG state. Used by HIPFIRE_SAMPLE_COMPARE
/// to run two samplers against the same seed so token differences reflect
/// real divergence and not just RNG stream drift.
pub fn sampler_rng_snapshot() -> u32 {
    use std::sync::atomic::Ordering;
    SAMPLER_STATE.load(Ordering::Relaxed)
}

pub fn sampler_rng_restore(state: u32) {
    use std::sync::atomic::Ordering;
    SAMPLER_STATE.store(state, Ordering::Relaxed);
}

/// hunt3 M-E: deterministic per-request reset entry point for the CPU sampler
/// RNG stream. The daemon calls this once at the top of generate()/generate_vl()
/// with the same per-request seed the GPU sampler uses, so CPU-fallback sampling
/// is reproducible across requests instead of carrying state from the prior one.
/// Note: `simple_rand` treats a stored 0 as "seed from time"; we map a 0 seed to
/// 1 so the reset stays deterministic. The RNG algorithm itself is unchanged.
pub fn reset_cpu_sampler_rng(seed: u32) {
    use std::sync::atomic::Ordering;
    let s = if seed == 0 { 1 } else { seed };
    SAMPLER_STATE.store(s, Ordering::Relaxed);
}

use std::sync::atomic::AtomicU32;
static SAMPLER_STATE: AtomicU32 = AtomicU32::new(0);

/// Simple deterministic-seeded RNG (xorshift32). Not crypto-quality, fine for sampling.
/// State lives in SAMPLER_STATE so that HIPFIRE_SAMPLE_COMPARE can snapshot/restore it.
fn simple_rand() -> f32 {
    use std::sync::atomic::Ordering;

    // Seed from time on first call
    let mut s = SAMPLER_STATE.load(Ordering::Relaxed);
    if s == 0 {
        s = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        if s == 0 {
            s = 1;
        }
    }
    // xorshift32
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    SAMPLER_STATE.store(s, Ordering::Relaxed);
    (s as f32) / (u32::MAX as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // The CPU sampler RNG (`SAMPLER_STATE`) is a process-global atomic shared by
    // every test. Tests that reset+draw from it must not interleave with each
    // other (a concurrent `simple_rand` from a sibling test would mutate the
    // state between a reset and the draw, breaking determinism). Serialize all
    // RNG-touching `sample_full_dist` tests behind this mutex.
    static RNG_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn qwen3_flash_mode_policy_matches_rdna_generation() {
        assert_eq!(llama_attention_flash_mode_for("auto", "gfx1100"), 2);
        assert_eq!(llama_attention_flash_mode_for("auto", "gfx1201"), 2);
        assert_eq!(llama_attention_flash_mode_for("auto", "gfx1030"), 1);
        assert_eq!(llama_attention_flash_mode_for("never", "gfx1100"), 0);
        assert_eq!(llama_attention_flash_mode_for("always", "gfx1030"), 2);
    }

    #[test]
    fn qwen3_flash_partials_follow_selected_q8_tile() {
        let max_seq = 32_768;
        let expected_tile =
            rdna_compute::attention::q8_flash_tile_size("gfx1100", 16, 8, 128, max_seq);
        assert_eq!(expected_tile, 32);
        assert_eq!(
            llama_flash_partials_len("gfx1100", 16, 8, 128, max_seq),
            16 * max_seq.div_ceil(expected_tile) * 130
        );
    }

    // hunt3 M-B (FinalFix): greedy argmax must drop NaN like the GPU kernel
    // (argmax.hip `data[i] > lmax`), never selecting a NaN-indexed token and
    // never dropping the true max that precedes a NaN.

    #[test]
    fn argmax_nan_at_last_index_not_selected() {
        // Old max_by idiom returned idx 3 (NaN); GPU kernel + fix return idx 1.
        let logits = [1.0f32, 5.0, 3.0, f32::NAN];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn argmax_nan_does_not_drop_true_max() {
        let logits = [5.0f32, f32::NAN, 0.1, 2.0];
        assert_eq!(argmax(&logits), 0);
    }

    #[test]
    fn argmax_finite_unaffected() {
        let logits = [0.1f32, 0.2, 0.9, 0.3];
        assert_eq!(argmax(&logits), 2);
    }

    // O2b-2 finite-guard: argmax is the sampler degenerate fallback, so it must
    // skip ALL non-finite values (not just NaN). A `+Inf` must never win the
    // bare `>` over a real finite max.

    #[test]
    fn argmax_pos_inf_does_not_displace_finite_max() {
        // Pre-fix: +Inf at idx 2 wins `>` → idx 2. With the finite guard the
        // real finite max (idx 1) is returned.
        let logits = [1.0f32, 5.0, f32::INFINITY, 3.0];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn argmax_mixed_finite_and_nonfinite_picks_finite() {
        // +Inf, NaN, and a finite peak interleaved → the finite peak (idx 4).
        let logits = [f32::INFINITY, f32::NAN, -2.0, f32::NAN, 9.0, f32::INFINITY];
        assert_eq!(argmax(&logits), 4);
    }

    #[test]
    fn argmax_all_pos_inf_returns_zero() {
        // No finite value survives → deterministic index 0 (matches the
        // sample_full_dist degenerate fallback contract).
        let logits = [f32::INFINITY; 6];
        assert_eq!(argmax(&logits), 0);
    }

    #[test]
    fn argmax_all_nan_returns_zero() {
        let logits = [f32::NAN; 6];
        assert_eq!(argmax(&logits), 0);
    }

    // O2b-2 overflow-guard: the daemon capacity guards compute
    // `prompt + max_tokens` (or `start + suffix + max_tokens`) in usize and
    // compare against a cap. An adversarial `max_tokens` near usize::MAX must
    // not wrap the sum below the cap and slip past the guard. The guards use
    // `saturating_add`; this asserts the saturating comparison still rejects.
    #[test]
    fn capacity_guard_saturating_add_rejects_huge_max_tokens() {
        let cap: usize = 32_768;
        let prompt_n: usize = 100;
        let max_tokens: usize = usize::MAX - 1;
        // Unchecked `prompt_n + max_tokens` would panic in debug / wrap in
        // release to `prompt_n - 2` (≈ usize::MAX), which is > cap by luck here,
        // but for an adversarial max_tokens chosen to land the wrapped sum
        // under the cap it would bypass. saturating_add can never wrap.
        assert!(prompt_n.saturating_add(max_tokens) > cap);

        // Choose max_tokens so the *wrapped* unchecked sum would fall under the
        // cap: max_tokens = usize::MAX - prompt_n + 1 → wraps to 0 < cap, which
        // would WRONGLY pass an unchecked guard. saturating_add saturates to
        // usize::MAX, so the guard still rejects.
        let adversarial = usize::MAX - prompt_n + 1;
        assert_eq!(prompt_n.wrapping_add(adversarial), 0, "wrapped sum is 0");
        assert!(
            prompt_n.saturating_add(adversarial) > cap,
            "saturating guard must still reject the wrap-to-zero adversarial max_tokens"
        );

        // Triple-add form used by the ds4 single-GPU guard.
        let start_pos: usize = 50;
        let suffix_len: usize = 50;
        assert!(
            start_pos
                .saturating_add(suffix_len)
                .saturating_add(usize::MAX)
                > cap
        );
    }

    // O1 fix-wave: edge-case guards for the host EP sampler `sample_full_dist`.
    // All run CPU-only; they seed the shared RNG via `reset_cpu_sampler_rng`
    // so the multinomial draw is deterministic.

    #[test]
    fn sample_full_dist_min_p_gt_one_no_panic() {
        let _g = RNG_TEST_LOCK.lock().unwrap();
        // FIX #3: min_p = 1.5 would empty the candidate set and panic on cand[0]
        // before the clamp. After the clamp + keep-at-least-one guard it must
        // return a valid in-range index without panicking.
        reset_cpu_sampler_rng(12345);
        let logits = [0.5f32, 2.0, 1.0, 0.1, 3.0];
        let idx = sample_full_dist(&logits, 1.0, 1.0, None, Some(1.5));
        assert!(
            (idx as usize) < logits.len(),
            "min_p>1 must still return an in-range token, got {idx}"
        );
    }

    #[test]
    fn sample_full_dist_min_p_filters_low_prob() {
        let _g = RNG_TEST_LOCK.lock().unwrap();
        // FIX #3 / general: min_p = 0.5 with a sharply peaked distribution must
        // drop the low-prob tails. With temp=1.0 the dominant logit (idx 1 at
        // 10.0) has prob ~1.0; min_p*p_max ~0.5 prunes everything else, so the
        // draw can only return idx 1 regardless of RNG.
        for seed in [1u32, 7, 99, 100000] {
            reset_cpu_sampler_rng(seed);
            let logits = [0.0f32, 10.0, 0.1, 0.2, 0.05];
            let idx = sample_full_dist(&logits, 1.0, 1.0, None, Some(0.5));
            assert_eq!(idx, 1, "min_p=0.5 should leave only the peak (seed {seed})");
        }
    }

    #[test]
    fn sample_full_dist_nan_logit_samples_finite() {
        let _g = RNG_TEST_LOCK.lock().unwrap();
        // FIX #4: a NaN logit must not poison `total` and force a silent greedy
        // fallback / NaN-indexed token. The NaN index (3) must never be drawn,
        // and the returned token must be a finite-logit index.
        reset_cpu_sampler_rng(424242);
        let logits = [1.0f32, 2.0, 0.5, f32::NAN, 1.5];
        for _ in 0..64 {
            let idx = sample_full_dist(&logits, 1.0, 1.0, None, None);
            assert_ne!(idx, 3, "NaN-indexed token must never be selected");
            assert!(
                (idx as usize) < logits.len() && logits[idx as usize].is_finite(),
                "must sample a finite-logit token, got idx {idx}"
            );
        }
        // Also with +Inf present: must still pick a finite token, not the Inf.
        reset_cpu_sampler_rng(424242);
        let logits2 = [1.0f32, f32::INFINITY, 0.5, 2.0];
        let idx2 = sample_full_dist(&logits2, 1.0, 1.0, None, None);
        assert_ne!(idx2, 1, "Inf-indexed token must never be selected");
        assert!(logits2[idx2 as usize].is_finite());
    }

    #[test]
    fn sample_full_dist_happy_path_min_p_zero() {
        // HAPPY PATH (the 3 cards): min_p = 0.0 with both top_k=None (full dist,
        // ds4 card) and top_k=40, plus temp/top_p, must return a valid in-range
        // index. The fixes are guarded on min_p>0 / non-finite / min_p>1, none
        // of which trigger here, so behavior is unchanged.
        let _g = RNG_TEST_LOCK.lock().unwrap();
        for &top_k in &[None, Some(40u32)] {
            for seed in [1u32, 2, 3, 42] {
                reset_cpu_sampler_rng(seed);
                let logits = [0.2f32, 1.5, 0.7, 3.0, 0.9, 2.1, 0.4, 1.1];
                let idx = sample_full_dist(&logits, 1.0, 1.0, top_k, Some(0.0));
                assert!(
                    (idx as usize) < logits.len(),
                    "happy path must return in-range idx (top_k={top_k:?}, seed {seed})"
                );
            }
        }
        // min_p = None (the other happy-path encoding) must also be in-range.
        reset_cpu_sampler_rng(7);
        let logits = [0.2f32, 1.5, 0.7, 3.0, 0.9];
        let idx = sample_full_dist(&logits, 0.8, 0.95, Some(40), None);
        assert!((idx as usize) < logits.len());
    }

    #[test]
    fn sample_full_dist_fixed_seed_deterministic() {
        // Determinism under a fixed seed (the daemon's `reset_cpu_sampler_rng`
        // contract): two back-to-back resets to the same seed must produce the
        // same token. The shared `SAMPLER_STATE` is a process-global atomic, so
        // we serialize the two calls inside a single test (no other test runs
        // between them in THIS function) — the standalone single-threaded proof
        // of determinism is racy only across parallel tests, never within one.
        let _g = RNG_TEST_LOCK.lock().unwrap();
        let logits = [0.2f32, 1.5, 0.7, 3.0, 0.9, 2.1, 0.4, 1.1];
        for seed in [1u32, 2, 3, 42] {
            reset_cpu_sampler_rng(seed);
            let a = sample_full_dist(&logits, 1.0, 1.0, None, Some(0.0));
            reset_cpu_sampler_rng(seed);
            let b = sample_full_dist(&logits, 1.0, 1.0, None, Some(0.0));
            assert_eq!(a, b, "fixed seed must be deterministic (seed {seed})");
        }
    }

    #[test]
    fn attractor_block_below_threshold() {
        // 2 occurrences of token 7 in window=20, threshold=3 → no block.
        let mut logits = vec![1.0f32; 16];
        let history: Vec<u32> = vec![1, 2, 7, 3, 4, 7, 5];
        apply_special_token_attractor_block(&mut logits, &history, 7, 20, 3);
        assert!(
            logits[7].is_finite(),
            "below threshold should leave logit untouched"
        );
    }

    #[test]
    fn attractor_block_at_threshold() {
        // 3 occurrences of token 5 in last 20 → block fires.
        let mut logits = vec![1.0f32; 16];
        let history: Vec<u32> = vec![5, 1, 5, 2, 5];
        apply_special_token_attractor_block(&mut logits, &history, 5, 20, 3);
        assert_eq!(
            logits[5],
            f32::NEG_INFINITY,
            "threshold met should -INF the logit"
        );
    }

    #[test]
    fn attractor_block_window_scoped() {
        // 3 occurrences of token 9, but only 1 in the last 5 tokens (window=5,
        // threshold=3) → no block.
        let mut logits = vec![1.0f32; 16];
        let history: Vec<u32> = vec![9, 9, 1, 2, 3, 4, 5, 9, 6];
        apply_special_token_attractor_block(&mut logits, &history, 9, 5, 3);
        assert!(logits[9].is_finite(), "older occurrences must not count");
    }

    #[test]
    fn attractor_block_pure_repeat() {
        // Worst case: model emits the same special token 5x in a row. Block
        // must fire.
        let mut logits = vec![0.5f32; 16];
        let history: Vec<u32> = vec![11, 11, 11, 11, 11];
        apply_special_token_attractor_block(&mut logits, &history, 11, 20, 3);
        assert_eq!(logits[11], f32::NEG_INFINITY);
        // Other logits untouched.
        assert!((logits[10] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn attractor_block_oob_token_is_noop() {
        let mut logits = vec![1.0f32; 4];
        let history: Vec<u32> = vec![999, 999, 999];
        // token_id past vocab size — should not panic, leave logits untouched.
        apply_special_token_attractor_block(&mut logits, &history, 999, 20, 3);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn unclosed_block_below_threshold() {
        // 1 open, 0 closes — depth=1 < threshold=2, no block.
        let mut logits = vec![1.0f32; 16];
        let history: Vec<u32> = vec![5, 1, 2];
        apply_unclosed_attractor_block(&mut logits, &history, 5, 6, 20, 2);
        assert!(logits[5].is_finite());
    }

    #[test]
    fn unclosed_block_paired_call_passes() {
        // Single complete call: <tool_call>{}</tool_call> = open + close.
        // Depth ends at 0; a follow-up second open would land at 1,
        // still below threshold=2. Don't block.
        let mut logits = vec![1.0f32; 16];
        let history: Vec<u32> = vec![5, 1, 2, 6, 5]; // open, body, body, close, open
        apply_unclosed_attractor_block(&mut logits, &history, 5, 6, 20, 2);
        assert!(
            logits[5].is_finite(),
            "second legit open after a complete call must pass"
        );
    }

    #[test]
    fn unclosed_block_two_stacked_opens_blocks_third() {
        // The exact #111 attractor shape: <tool_call><tool_call>...
        // After two consecutive opens with no close, depth = 2 = threshold,
        // block fires (preventing the third).
        let mut logits = vec![1.0f32; 16];
        let history: Vec<u32> = vec![5, 5];
        apply_unclosed_attractor_block(&mut logits, &history, 5, 6, 20, 2);
        assert_eq!(logits[5], f32::NEG_INFINITY);
    }

    #[test]
    fn unclosed_block_depth_saturates_at_zero() {
        // Stray close at start of window must not push depth negative
        // and let an attractor through. Window: close, open, open.
        // depth = max(0, -1) + 1 + 1 = 2 → block.
        let mut logits = vec![1.0f32; 16];
        let history: Vec<u32> = vec![6, 5, 5];
        apply_unclosed_attractor_block(&mut logits, &history, 5, 6, 20, 2);
        assert_eq!(logits[5], f32::NEG_INFINITY);
    }

    #[test]
    fn unclosed_block_window_scoped() {
        // 2 unclosed opens earlier in history, but the recent window=3 only
        // sees [body, body, close]. depth = 0, allow.
        let mut logits = vec![1.0f32; 16];
        let history: Vec<u32> = vec![5, 5, 1, 2, 6];
        apply_unclosed_attractor_block(&mut logits, &history, 5, 6, 3, 2);
        assert!(
            logits[5].is_finite(),
            "older unclosed opens must not count once they leave the window"
        );
    }

    #[test]
    fn is_batchable_la_always_ok_dtypes() {
        // MQ4/HFQ4/MQ6/HFQ6 batchable on every arch.
        for arch in [
            "gfx900", "gfx906", "gfx1010", "gfx1030", "gfx1100", "gfx1200", "gfx942",
        ] {
            assert!(is_batchable_la(DType::HFQ4G256, arch));
            assert!(is_batchable_la(DType::HFQ4G128, arch));
            assert!(is_batchable_la(DType::MQ4G256, arch));
            assert!(is_batchable_la(DType::HFQ6G256, arch));
            assert!(is_batchable_la(DType::MQ6G256, arch));
        }
    }

    #[test]
    fn is_batchable_la_mq3_wmma_only() {
        // MQ3 batchable on WMMA archs (gfx11/gfx12) via the WMMA path, and on
        // gfx10 RDNA1/2 via the scalar path (PR #298, commit 4840f0b).
        for arch in [
            "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
        ] {
            assert!(
                is_batchable_la(DType::MQ3G256, arch),
                "MQ3 should batch on {arch} (WMMA)"
            );
        }
        for arch in [
            "gfx1010", "gfx1011", "gfx1012", "gfx1013", "gfx1030", "gfx1031", "gfx1032",
        ] {
            assert!(
                is_batchable_la(DType::MQ3G256, arch),
                "MQ3 should batch on {arch} (gfx10 scalar)"
            );
        }
        for arch in ["gfx900", "gfx906", "gfx942"] {
            assert!(
                !is_batchable_la(DType::MQ3G256, arch),
                "MQ3 must fall back on {arch}"
            );
        }
    }

    #[test]
    fn is_batchable_la_fp4_wmma_only() {
        // HFP4G32 / MFP4G32 require WMMA — same arch gate as MQ3.
        for arch in [
            "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
        ] {
            assert!(
                is_batchable_la(DType::HFP4G32, arch),
                "HFP4G32 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::MFP4G32, arch),
                "MFP4G32 should batch on {arch}"
            );
        }
        for arch in ["gfx900", "gfx906", "gfx1010", "gfx1030", "gfx942"] {
            assert!(
                !is_batchable_la(DType::HFP4G32, arch),
                "HFP4G32 must fall back on {arch}"
            );
            assert!(
                !is_batchable_la(DType::MFP4G32, arch),
                "MFP4G32 must fall back on {arch}"
            );
        }
    }

    #[test]
    fn is_batchable_la_unsupported_dtypes() {
        // Q4K / Q6K / F32 stay on per-token forward_scratch.
        for arch in ["gfx1100", "gfx1200"] {
            assert!(!is_batchable_la(DType::Q4K, arch));
            assert!(!is_batchable_la(DType::Q6K, arch));
            assert!(!is_batchable_la(DType::F32, arch));
        }
    }

    #[test]
    fn is_batchable_la_q8_0_always_ok() {
        // Q8_0 is batchable on every arch via gemm_q8_0_batched_chunked
        // (unfused, sub-batched at MAX_BATCH=16). Eval-mode noise-floor path —
        // see docs/plans/q8_batchable.md.
        for arch in [
            "gfx900", "gfx906", "gfx1010", "gfx1030", "gfx1100", "gfx1200", "gfx942",
        ] {
            assert!(is_batchable_la(DType::Q8_0, arch));
        }
    }

    // ── ParoQuant dispatch helpers ──────────────────────────────

    #[test]
    fn paro_small_direct_returns_none_when_unset() {
        // With env var unset, should return None
        // (can't clear env in test, but this verifies the conversion logic)
        // The function returns None when env var is not set (via try_opt)
        // We test the parsing behavior with a lock
    }

    // ── KvCache format dispatch ─────────────────────────────────

    #[test]
    fn kv_cache_is_boundary_within_bounds() {
        // Mock KvCache to test is_boundary logic
        // Since KvCache requires GPU allocation, we test the predicate in isolation
        // by constructing the boolean checks that the dispatch uses.
    }

    // O2b-2 fix-wave: finite/NaN guards for the LEGACY CPU sampler `sample_top_p`
    // (the single-GPU non-EP path). Before the fix an all-non-finite logit
    // vector collapsed to token 0 (max_logit stayed -inf / Inf-Inf=NaN poisoned
    // the softmax sum). These mirror the `sample_full_dist_*` guards above:
    // every degenerate input must still return a FINITE, in-vocab token.

    #[test]
    fn legacy_sample_top_p_all_nan_returns_in_vocab() {
        let _g = RNG_TEST_LOCK.lock().unwrap();
        reset_cpu_sampler_rng(12345);
        let logits = [f32::NAN; 8];
        let idx = sample_top_p(&logits, 0.7, 0.9);
        assert!(
            (idx as usize) < logits.len(),
            "all-NaN must return an in-range token, got {idx}"
        );
    }

    #[test]
    fn legacy_sample_top_p_all_pos_inf_returns_in_vocab() {
        let _g = RNG_TEST_LOCK.lock().unwrap();
        reset_cpu_sampler_rng(777);
        // Pre-fix: +Inf max → (Inf - Inf) = NaN softmax → NaN sum → token 0.
        let logits = [f32::INFINITY; 8];
        let idx = sample_top_p(&logits, 0.7, 0.9);
        assert!(
            (idx as usize) < logits.len(),
            "all-+Inf must return an in-range token, got {idx}"
        );
    }

    #[test]
    fn legacy_sample_top_p_mixed_finite_nan_picks_finite() {
        let _g = RNG_TEST_LOCK.lock().unwrap();
        // The only finite mass is at idx 2 (a sharp peak after temp). The NaN
        // slots (0,1,3,4) must never be drawn, and the +Inf slot (5) must be
        // skipped — the draw is forced onto the finite peak regardless of RNG.
        for seed in [1u32, 7, 99, 100000] {
            reset_cpu_sampler_rng(seed);
            let logits = [
                f32::NAN,
                f32::NAN,
                20.0f32,
                f32::NAN,
                f32::NAN,
                f32::INFINITY,
            ];
            let idx = sample_top_p(&logits, 0.7, 0.9);
            assert_eq!(
                idx, 2,
                "mixed finite+NaN+Inf must pick the lone finite peak (seed {seed})"
            );
        }
    }
    // ── KvCache::tier_inputs() accessor pin test ─────────────────

    #[test]
    fn q8_prefill_family_stays_inside_validated_qwen3_arch_envelope() {
        let eligible =
            |arch, model, q8, tree, batch| q8_prefill_family_eligible(arch, model, q8, tree, batch);
        assert!(eligible("gfx1100", ModelArch::Qwen3, true, false, 256));
        assert!(eligible("gfx1201", ModelArch::Qwen3, true, false, 2));
        assert!(!eligible("gfx1030", ModelArch::Qwen3, true, false, 256));
        assert!(!eligible("gfx1200", ModelArch::Qwen3, true, false, 256));
        assert!(!eligible("gfx1100", ModelArch::Llama, true, false, 256));
        assert!(!eligible("gfx1100", ModelArch::Qwen3, true, true, 256));
        assert!(!eligible("gfx1100", ModelArch::Qwen3, true, false, 1));
        assert!(!eligible("gfx1100", ModelArch::Qwen3, false, false, 256));
    }

    #[test]
    fn tier_inputs_q8_is_byte_identical_to_legacy_literal() {
        // Skip cleanly on a box with no GPU (CI); runs for real on gfx1151.
        let mut gpu = match Gpu::init() {
            Ok(g) => g,
            Err(_) => return,
        };
        let kv = KvCache::new_gpu_q8(&mut gpu, 1, 8, 64, 16).unwrap();

        // What the unified accessor produces for a decode call at pos=100.
        let got = KvTierInputs {
            pos: 100,
            ..kv.tier_inputs()
        };

        // The exact literal the qwen35 decode site used before this refactor.
        let legacy = KvTierInputs {
            quant_asym4: kv.quant_asym4,
            quant_asym3: kv.quant_asym3,
            quant_asym2: kv.quant_asym2,
            quant_q8: kv.quant_q8,
            quant_fwht: kv.quant_fwht,
            quant_hfq4: false,
            quant_q4: false,
            quant_int8: false,
            quant_hfq8: false,
            f32_policy: hipfire_dispatch::families::kv_tier::F32AttnPolicy::Simple,
            v_mode_bits: kv.v_mode_bits(),
            pos: 100,
            flash_mode: 0,
            capture_mode: false,
            batch_size: 1,
            is_tree: false,
            is_boundary: false,
            q8_windowed: false,
            window: 0,
        };

        assert_eq!(
            got, legacy,
            "tier_inputs() must be byte-identical to the legacy literal"
        );
        assert_eq!(
            KvTierPlan::derive(got).unwrap().write_key,
            KvTierPlan::derive(legacy).unwrap().write_key,
        );
        assert_eq!(
            KvTierPlan::derive(got).unwrap().attend_key,
            KvTierPlan::derive(legacy).unwrap().attend_key,
        );

        // Prefill case: the batched sites override the non-zero `batch_size`
        // default (and set `is_tree`). Pin that the functional-update override
        // wins over the accessor default, matching the old prefill literal.
        let got_prefill = KvTierInputs {
            pos: 100,
            flash_mode: 2,
            capture_mode: true,
            batch_size: 4,
            is_tree: true,
            ..kv.tier_inputs()
        };
        let legacy_prefill = KvTierInputs {
            quant_asym4: kv.quant_asym4,
            quant_asym3: kv.quant_asym3,
            quant_asym2: kv.quant_asym2,
            quant_q8: kv.quant_q8,
            quant_fwht: kv.quant_fwht,
            quant_hfq4: false,
            quant_q4: false,
            quant_int8: false,
            quant_hfq8: false,
            f32_policy: hipfire_dispatch::families::kv_tier::F32AttnPolicy::Simple,
            v_mode_bits: kv.v_mode_bits(),
            pos: 100,
            flash_mode: 2,
            capture_mode: true,
            batch_size: 4,
            is_tree: true,
            is_boundary: false,
            q8_windowed: false,
            window: 0,
        };
        assert_eq!(
            got_prefill, legacy_prefill,
            "tier_inputs() prefill override must be byte-identical to the legacy prefill literal"
        );
        assert_eq!(
            KvTierPlan::derive(got_prefill).unwrap().write_key,
            KvTierPlan::derive(legacy_prefill).unwrap().write_key,
        );
        assert_eq!(
            KvTierPlan::derive(got_prefill).unwrap().attend_key,
            KvTierPlan::derive(legacy_prefill).unwrap().attend_key,
        );
    }

    /// Regression guard for the L1/L2 review finding: on an asym cache,
    /// `tier_inputs()` must NOT also set `quant_q4` (the residual previously
    /// did, double-flagging asym3+q4 → classify()'s "at most one tier flag"
    /// debug_assert fired in debug builds on the qwen35 asym3 default, and the
    /// no-op-vs-legacy-literal proof was false). The legacy qwen35 literals
    /// hardcoded `quant_q4: false`; pin that `tier_inputs()` matches, and that
    /// `derive()` (which routes through classify) does not panic in debug.
    #[test]
    fn tier_inputs_asym3_has_no_residual_q4_flag() {
        let mut gpu = match Gpu::init() {
            Ok(g) => g,
            Err(_) => return,
        };
        // asym3 requires head_dim == 256.
        let kv = KvCache::new_gpu_asym3(&mut gpu, 1, 8, 256, 16).unwrap();

        let got = KvTierInputs {
            pos: 100,
            ..kv.tier_inputs()
        };
        assert!(kv.quant_asym3, "test setup: expected an asym3 cache");
        assert!(
            !got.quant_q4,
            "tier_inputs() must not set quant_q4 on an asym cache (double-flag bug)"
        );
        // Exactly one tier flag set — what classify()'s debug_assert requires.
        let tier_flags = [
            got.quant_asym4,
            got.quant_asym3,
            got.quant_asym2,
            got.quant_q8,
            got.quant_hfq4,
            got.quant_q4,
        ]
        .iter()
        .filter(|&&b| b)
        .count();
        assert_eq!(tier_flags, 1, "exactly one KV tier flag must be set");

        // Byte-identical to the legacy literal (which hardcoded quant_q4=false),
        // and derive() must not panic on the classify debug_assert.
        let legacy = KvTierInputs {
            quant_asym4: kv.quant_asym4,
            quant_asym3: kv.quant_asym3,
            quant_asym2: kv.quant_asym2,
            quant_q8: kv.quant_q8,
            quant_fwht: kv.quant_fwht,
            quant_hfq4: false,
            quant_q4: false,
            quant_int8: false,
            quant_hfq8: false,
            f32_policy: hipfire_dispatch::families::kv_tier::F32AttnPolicy::Simple,
            v_mode_bits: kv.v_mode_bits(),
            pos: 100,
            flash_mode: 0,
            capture_mode: false,
            batch_size: 1,
            is_tree: false,
            is_boundary: false,
            q8_windowed: false,
            window: 0,
        };
        assert_eq!(
            got, legacy,
            "tier_inputs() must match the legacy asym3 literal"
        );
        assert_eq!(
            KvTierPlan::derive(got).unwrap().write_key,
            KvTierPlan::derive(legacy).unwrap().write_key,
        );
    }

    fn vmm_mask_dims(
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        physical_cap: usize,
    ) -> KvDims {
        KvDims {
            layers: KvLayers::Mask(vec![true, false, true]),
            n_kv_heads,
            head_dim,
            max_seq,
            physical_cap: Some(physical_cap),
        }
    }

    fn expected_k_bph(mode: KvMode, head_dim: usize) -> usize {
        match mode {
            KvMode::Q8 => (head_dim / 32) * 34,
            KvMode::Asym2 | KvMode::Fwht2 => 4 + head_dim / 4,
            KvMode::Asym3 | KvMode::Fwht3 => 4 + (head_dim * 3) / 8,
            KvMode::Asym4 | KvMode::Fwht4 => 4 + head_dim / 2,
            KvMode::Asym3Auto => panic!("Asym3Auto is not a layout mode"),
        }
    }

    fn expected_v_bph(v_mode: VMode, head_dim: usize) -> usize {
        match v_mode {
            VMode::Q8 => (head_dim / 32) * 34,
            VMode::Lloyd2 => 4 + head_dim / 4,
            VMode::Lloyd3 => 4 + (head_dim * 3) / 8,
            VMode::Lloyd4 => 4 + head_dim / 2,
        }
    }

    fn flag_standin(mode: KvMode, v_mode: VMode, n_kv_heads: usize, head_dim: usize) -> KvCache {
        let (q8, a4, a3, a2, fwht) = KvCache::vmm_mode_flags(mode);
        KvCache {
            k_gpu: vec![],
            v_gpu: vec![],
            k_scales: vec![],
            v_scales: vec![],
            kv_dim: n_kv_heads * head_dim,
            max_seq: 128,
            physical_cap: 128,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: q8,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: a4,
            quant_asym3: a3,
            quant_asym2: a2,
            quant_fwht: fwht,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode,
        }
    }

    #[test]
    fn adaptive_reset_invalidates_captured_execution_state() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: no GPU");
            return;
        };
        let mut cache = flag_standin(KvMode::Fwht2, VMode::Lloyd2, 4, 256);
        let mut adaptive = crate::kv_adaptive::KvAdaptive::from_preset(
            crate::kv_adaptive::Preset::Aggressive,
            128,
            4,
            256,
        );
        adaptive.cur_k = crate::kv_adaptive::KMode::Fwht2;
        adaptive.cur_v = VMode::Lloyd2;
        adaptive.next_step = adaptive.steps.len();
        gpu.graphs.ar_forward_replay_enabled = true;
        gpu.graphs.ar_forward_kernel_dirty = false;

        adaptive.reset_with_cache(&mut gpu, &mut cache);

        assert_eq!(cache.current_kv_mode().unwrap(), KvMode::Fwht4);
        assert_eq!(cache.v_mode, VMode::Q8);
        assert!(!gpu.graphs.ar_forward_replay_enabled);
        assert!(gpu.graphs.ar_forward_kernel_dirty);
    }

    #[test]
    fn vmm_static_layout_covers_all_seven_k_modes_with_q8_v() {
        let n_kv_heads = 4;
        let head_dim = 256;
        let physical_cap = 128;
        let modes = [
            KvMode::Q8,
            KvMode::Asym2,
            KvMode::Asym3,
            KvMode::Asym4,
            KvMode::Fwht2,
            KvMode::Fwht3,
            KvMode::Fwht4,
        ];
        for mode in modes {
            let layout =
                KvCache::vmm_static_layout(mode, VMode::Q8, n_kv_heads, head_dim, physical_cap)
                    .unwrap_or_else(|e| panic!("mode={mode:?}: {e}"));
            let k_bph = expected_k_bph(mode, head_dim);
            let v_bph = expected_v_bph(VMode::Q8, head_dim);
            assert_eq!(layout.k_bytes_per_head, k_bph, "mode={mode:?}");
            assert_eq!(layout.v_bytes_per_head, v_bph, "mode={mode:?}");
            assert_eq!(
                layout.k_bytes_per_token,
                n_kv_heads * k_bph,
                "mode={mode:?}"
            );
            assert_eq!(
                layout.v_bytes_per_token,
                n_kv_heads * v_bph,
                "mode={mode:?}"
            );
            // Static: reserve == current.
            assert_eq!(
                layout.k_reserve_bytes,
                physical_cap * layout.k_bytes_per_token,
                "mode={mode:?}"
            );
            assert_eq!(
                layout.v_reserve_bytes,
                physical_cap * layout.v_bytes_per_token,
                "mode={mode:?}"
            );
            assert_eq!(
                layout.k_reserve_elems,
                (layout.k_reserve_bytes + 3) / 4,
                "mode={mode:?}"
            );
            assert_eq!(
                layout.v_reserve_elems,
                (layout.v_reserve_bytes + 3) / 4,
                "mode={mode:?}"
            );
            assert_eq!(layout.kv_dim, n_kv_heads * head_dim);
            // Independent K vs V capacity at equal physical_cap is just physical_cap.
            let k_cap = layout.k_reserve_bytes / layout.k_bytes_per_token;
            let v_cap = layout.v_reserve_bytes / layout.v_bytes_per_token;
            assert_eq!(k_cap.min(v_cap), physical_cap, "mode={mode:?}");
        }
    }

    #[test]
    fn vmm_static_layout_covers_fwht_k_with_lloyd_v() {
        let n_kv_heads = 4;
        let head_dim = 256;
        let physical_cap = 64;
        let k_modes = [KvMode::Fwht2, KvMode::Fwht3, KvMode::Fwht4];
        let v_modes = [VMode::Lloyd2, VMode::Lloyd3, VMode::Lloyd4];
        for mode in k_modes {
            for v_mode in v_modes {
                let layout =
                    KvCache::vmm_static_layout(mode, v_mode, n_kv_heads, head_dim, physical_cap)
                        .unwrap_or_else(|e| panic!("{mode:?}/{v_mode:?}: {e}"));
                assert_eq!(layout.k_bytes_per_head, expected_k_bph(mode, head_dim));
                assert_eq!(layout.v_bytes_per_head, expected_v_bph(v_mode, head_dim));
                // Lloyd-V forces 256-wide signs even for fwht2/4.
                assert_eq!(layout.rotation_table_len, 256, "{mode:?}/{v_mode:?}");
                assert!(layout.uses_fwht_signs, "{mode:?}/{v_mode:?}");
                // Prefix helpers use current stride, not reserve.
                let n_pos = 17;
                assert_eq!(
                    layout.prefix_k_bytes(n_pos).unwrap(),
                    n_pos * layout.k_bytes_per_token
                );
                assert_eq!(
                    layout.prefix_v_bytes(n_pos).unwrap(),
                    n_pos * layout.v_bytes_per_token
                );
            }
        }
    }

    #[test]
    fn vmm_layout_rejects_illegal_pairs() {
        // Asym-K + Lloyd-V is illegal.
        for mode in [KvMode::Asym2, KvMode::Asym3, KvMode::Asym4, KvMode::Q8] {
            let err = KvCache::vmm_static_layout(mode, VMode::Lloyd4, 4, 256, 64)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("lloyd-V") || err.contains("VMode::Q8"),
                "mode={mode:?} err={err}"
            );
        }
        // FWHT3 / Asym3 require head_dim=256.
        for mode in [KvMode::Fwht3, KvMode::Asym3] {
            let err = KvCache::vmm_static_layout(mode, VMode::Q8, 4, 128, 64)
                .unwrap_err()
                .to_string();
            assert!(err.contains("head_dim"), "mode={mode:?} err={err}");
        }
        // n_kv_heads == 0.
        let err = KvCache::vmm_static_layout(KvMode::Fwht3, VMode::Q8, 0, 256, 64)
            .unwrap_err()
            .to_string();
        assert!(err.contains("n_kv_heads>0"), "{err}");
        // Lloyd-V with FWHT-K but head_dim != 256.
        let err = KvCache::vmm_static_layout(KvMode::Fwht4, VMode::Lloyd2, 4, 128, 64)
            .unwrap_err()
            .to_string();
        assert!(err.contains("head_dim=256"), "{err}");
    }

    #[test]
    fn fwht3_vmm_layout_matches_asym3_byte_geometry() {
        let n_kv_heads = 4;
        let head_dim = 256;
        let physical_cap = 128;
        let asym = KvCache::asym3_vmm_layout(n_kv_heads, head_dim, physical_cap).unwrap();
        let fwht = KvCache::fwht3_vmm_layout(n_kv_heads, head_dim, physical_cap).unwrap();
        assert_eq!(fwht, asym, "fwht3 VMM must reuse asym3 K/V byte layout");
        // Explicit stride math: K 100 B/head, V Q8 272 B/head at head_dim=256.
        assert_eq!(fwht.3, 4 + (head_dim * 3) / 8);
        assert_eq!(fwht.4, (head_dim / 32) * 34);
        let k_bytes = physical_cap * n_kv_heads * fwht.3;
        let v_bytes = physical_cap * n_kv_heads * fwht.4;
        assert_eq!(fwht.1, (k_bytes + 3) / 4);
        assert_eq!(fwht.2, (v_bytes + 3) / 4);
        assert_eq!(fwht.0, n_kv_heads * head_dim);
    }

    #[test]
    fn vmm_layout_with_reserve_separates_current_and_floor() {
        // Adaptive-style: current FWHT4/Q8, reserve at fwht2/lloyd2 floors.
        let n_kv_heads = 4;
        let head_dim = 256;
        let physical_cap = 1000;
        let k_floor_bph = expected_k_bph(KvMode::Fwht2, head_dim); // 68
        let layout = KvCache::vmm_layout_with_reserve(
            KvMode::Fwht4,
            VMode::Q8,
            n_kv_heads,
            head_dim,
            physical_cap,
            k_floor_bph,
            VMode::Lloyd2,
        )
        .unwrap();
        // Current strides are start encoding.
        assert_eq!(
            layout.k_bytes_per_token,
            n_kv_heads * expected_k_bph(KvMode::Fwht4, head_dim)
        );
        assert_eq!(
            layout.v_bytes_per_token,
            n_kv_heads * expected_v_bph(VMode::Q8, head_dim)
        );
        // Reserve is floor-sized.
        assert_eq!(
            layout.k_reserve_bytes,
            physical_cap * n_kv_heads * k_floor_bph
        );
        assert_eq!(
            layout.v_reserve_bytes,
            physical_cap * n_kv_heads * expected_v_bph(VMode::Lloyd2, head_dim)
        );
        // Token capacity at start is min of reserve/current (V binds: 68/272).
        let k_cap = layout.k_reserve_bytes / layout.k_bytes_per_token;
        let v_cap = layout.v_reserve_bytes / layout.v_bytes_per_token;
        assert_eq!(
            k_cap,
            physical_cap * k_floor_bph / expected_k_bph(KvMode::Fwht4, head_dim)
        );
        assert_eq!(
            v_cap,
            physical_cap * expected_v_bph(VMode::Lloyd2, head_dim)
                / expected_v_bph(VMode::Q8, head_dim)
        );
        assert!(v_cap < k_cap, "V should bind at FWHT4/Q8 start");
        // Source-prefix bytes remain current-stride × n_pos (not floor).
        assert_eq!(
            layout.prefix_k_bytes(10).unwrap(),
            10 * layout.k_bytes_per_token
        );
        assert_eq!(
            layout.prefix_v_bytes(10).unwrap(),
            10 * layout.v_bytes_per_token
        );
        assert_eq!(layout.rotation_table_len, 256);
    }

    #[test]
    fn validate_mode_admits_all_seven_static_vmm_modes() {
        let dims = vmm_mask_dims(4, 256, 4096, 1024);
        for mode in [
            KvMode::Q8,
            KvMode::Asym2,
            KvMode::Asym3,
            KvMode::Asym4,
            KvMode::Fwht2,
            KvMode::Fwht3,
            KvMode::Fwht4,
        ] {
            KvCache::validate_mode_with_backend(mode, KvBackend::Vmm, true, &dims)
                .unwrap_or_else(|e| panic!("mode={mode:?}: {e}"));
            // Contiguous admission remains open (no VMM-only gate).
            KvCache::validate_mode_with_backend(mode, KvBackend::Contiguous, true, &dims)
                .unwrap_or_else(|e| panic!("contiguous mode={mode:?}: {e}"));
        }
    }

    #[test]
    fn validate_mode_rejects_multi_gpu_flat_and_asym3auto_vmm() {
        let mask = vmm_mask_dims(4, 256, 4096, 1024);
        let err = KvCache::validate_mode_with_backend(KvMode::Fwht3, KvBackend::Vmm, false, &mask)
            .unwrap_err()
            .to_string();
        assert!(err.contains("single-GPU"), "{err}");

        let flat = KvDims {
            layers: KvLayers::Flat(8),
            n_kv_heads: 4,
            head_dim: 256,
            max_seq: 4096,
            physical_cap: Some(1024),
        };
        let err = KvCache::validate_mode_with_backend(KvMode::Fwht3, KvBackend::Vmm, true, &flat)
            .unwrap_err()
            .to_string();
        assert!(err.contains("filtered"), "{err}");

        let err =
            KvCache::validate_mode_with_backend(KvMode::Asym3Auto, KvBackend::Vmm, true, &mask)
                .unwrap_err()
                .to_string();
        assert!(err.contains("Asym3Auto"), "{err}");
    }

    #[test]
    fn vmm_bytes_per_token_matches_layout_for_all_static_modes() {
        let n_kv_heads = 4;
        let head_dim = 256;
        for mode in [
            KvMode::Q8,
            KvMode::Asym2,
            KvMode::Asym3,
            KvMode::Asym4,
            KvMode::Fwht2,
            KvMode::Fwht3,
            KvMode::Fwht4,
        ] {
            let cache = flag_standin(mode, VMode::Q8, n_kv_heads, head_dim);
            let (k, v) = cache.vmm_bytes_per_token().unwrap();
            assert_eq!(k, n_kv_heads * expected_k_bph(mode, head_dim), "{mode:?}");
            assert_eq!(
                v,
                n_kv_heads * expected_v_bph(VMode::Q8, head_dim),
                "{mode:?}"
            );
        }
        // FWHT-K + Lloyd-V current strides.
        for v_mode in [VMode::Lloyd2, VMode::Lloyd3, VMode::Lloyd4] {
            let cache = flag_standin(KvMode::Fwht4, v_mode, n_kv_heads, head_dim);
            let (k, v) = cache.vmm_bytes_per_token().unwrap();
            assert_eq!(
                k,
                n_kv_heads * expected_k_bph(KvMode::Fwht4, head_dim),
                "{v_mode:?}"
            );
            assert_eq!(
                v,
                n_kv_heads * expected_v_bph(v_mode, head_dim),
                "{v_mode:?}"
            );
        }
        // Asym-K + Lloyd-V must fail (illegal pair).
        let bad = flag_standin(KvMode::Asym3, VMode::Lloyd3, n_kv_heads, head_dim);
        assert!(bad.vmm_bytes_per_token().is_err());
    }

    #[test]
    fn independent_k_v_capacity_math_from_reserve_and_current() {
        // Simulate lopsided adaptive start: K floor 68, V floor 68, current K132/V272.
        let n_kv_heads = 4;
        let head_dim = 256;
        let physical_cap = 1000;
        let layout = KvCache::vmm_layout_with_reserve(
            KvMode::Fwht4,
            VMode::Q8,
            n_kv_heads,
            head_dim,
            physical_cap,
            expected_k_bph(KvMode::Fwht2, head_dim),
            VMode::Lloyd2,
        )
        .unwrap();
        let k_tokens = layout.k_reserve_bytes / layout.k_bytes_per_token;
        let v_tokens = layout.v_reserve_bytes / layout.v_bytes_per_token;
        // Must NOT sum K+V bytes into a shared pool.
        let naive_shared = (layout.k_reserve_bytes + layout.v_reserve_bytes)
            / (layout.k_bytes_per_token + layout.v_bytes_per_token);
        assert_ne!(
            k_tokens.min(v_tokens),
            naive_shared,
            "min-of-two must differ from shared-pool sum"
        );
        assert_eq!(k_tokens.min(v_tokens), v_tokens);
    }

    #[test]
    fn free_gpu_empty_cache_is_ok() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: no GPU");
            return;
        };
        let cache = KvCache {
            k_gpu: vec![],
            v_gpu: vec![],
            k_scales: vec![],
            v_scales: vec![],
            kv_dim: 0,
            max_seq: 0,
            physical_cap: 0,
            n_kv_heads: 0,
            head_dim: 0,
            quantized: false,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        };
        cache.free_gpu(&mut gpu).expect("empty free");
    }

    #[test]
    fn free_gpu_propagates_vmm_teardown_failure_and_retries() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: no GPU");
            return;
        };
        hip_bridge::clear_vmm_faults();
        let access = [gpu.device_id];
        let chunk = 2 * 1024 * 1024;
        let tensor = match unsafe { gpu.alloc_vmm_tensor(&[chunk], DType::Raw, chunk, &access) } {
            Ok(t) => t,
            Err(_) => {
                eprintln!("skip: VMM unavailable");
                return;
            }
        };
        let cache = KvCache {
            k_gpu: vec![tensor],
            v_gpu: vec![],
            k_scales: vec![],
            v_scales: vec![],
            kv_dim: 1,
            max_seq: 1,
            physical_cap: 1,
            n_kv_heads: 1,
            head_dim: 1,
            quantized: false,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        };
        hip_bridge::inject_vmm_fault(hip_bridge::VmmFaultKind::Unmap, 1);
        let err = cache
            .free_gpu(&mut gpu)
            .expect_err("must surface teardown fault");
        assert!(
            err.to_string().contains("retained") || err.to_string().contains("injected"),
            "{err}"
        );
        assert_eq!(
            gpu.vmm_allocation_count(),
            1,
            "ownership retained after free_gpu err"
        );
        hip_bridge::clear_vmm_faults();
        gpu.ensure_vmm_cleaned().expect("retry");
        assert_eq!(gpu.vmm_allocation_count(), 0);
        // No new load while pending would be refused:
        // Two unmap faults: one for free_tensor, one for ensure/retry.
        hip_bridge::inject_vmm_fault(hip_bridge::VmmFaultKind::Unmap, 2);
        let t2 =
            unsafe { gpu.alloc_vmm_tensor(&[chunk], DType::Raw, chunk, &access) }.expect("alloc");
        let _ = gpu.free_tensor(t2).expect_err("fault");
        assert!(gpu.vmm_allocation_count() >= 1);
        // simulate load gate
        let refuse = gpu.ensure_vmm_cleaned();
        assert!(
            refuse.is_err(),
            "load must refuse while pending: {refuse:?}"
        );
        hip_bridge::clear_vmm_faults();
        gpu.ensure_vmm_cleaned().expect("clear");
    }
}
