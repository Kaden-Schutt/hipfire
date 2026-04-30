//! LLaMA model implementation using RDNA GPU compute.
//! Supports loading from GGUF files and running inference.

use crate::gguf::{GgmlType, GgufFile, TensorInfo};
use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;

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
        let vocab_size = gguf
            .meta_u32(&format!("{prefix}.vocab_size"))
            .or_else(|| {
                gguf.find_tensor("token_embd.weight")
                    .map(|t| t.shape[1] as u32)
            })?
            as usize;
        let head_dim = gguf
            .meta_u32(&format!("{prefix}.attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or(dim / n_heads);
        let norm_eps = gguf.meta_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon")).unwrap_or(1e-5);
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
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        // Denormalized
        let mut e = 0i32;
        let mut f = frac;
        while f & 0x400 == 0 {
            f <<= 1;
            e -= 1;
        }
        f &= 0x3FF;
        let exp32 = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13));
    }
    if exp == 31 {
        let frac32 = if frac == 0 { 0 } else { frac << 13 | 1 };
        return f32::from_bits((sign << 31) | (0xFF << 23) | frac32);
    }
    let exp32 = exp + 127 - 15;
    f32::from_bits((sign << 31) | (exp32 << 23) | (frac << 13))
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
            let hi_val = if 32 + i < actual_len { group[32 + i] } else { min_val };

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
            output[out_off_even..out_off_even + 2].copy_from_slice(&f32_to_f16(eff_scale_even).to_le_bytes());
            output[out_off_even + 2..out_off_even + 4].copy_from_slice(&f32_to_f16(eff_min_even).to_le_bytes());

            // Sub-block odd (elements group*64+32..group*64+63) → G32 block
            let eff_scale_odd = d * scales[sb_odd] as f32;
            let eff_min_odd = -(dmin * mins[sb_odd] as f32);
            let out_off_odd = (b * 8 + sb_odd) * g32_block_bytes;
            output[out_off_odd..out_off_odd + 2].copy_from_slice(&f32_to_f16(eff_scale_odd).to_le_bytes());
            output[out_off_odd + 2..out_off_odd + 4].copy_from_slice(&f32_to_f16(eff_min_odd).to_le_bytes());

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

                if idx0 < n { out[idx0] = d * sc[is] as i8 as f32 * q1 as f32; }
                if idx1 < n { out[idx1] = d * sc[is + 2] as i8 as f32 * q2 as f32; }
                if idx2 < n { out[idx2] = d * sc[is + 4] as i8 as f32 * q3 as f32; }
                if idx3 < n { out[idx3] = d * sc[is + 6] as i8 as f32 * q4 as f32; }
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
pub struct WeightTensor {
    pub buf: GpuTensor,
    pub gpu_dtype: DType, // dispatch type for kernel selection
    pub m: usize,         // output dim (rows)
    pub k: usize,         // input dim (cols)
    pub row_stride: usize, // padded row bytes (Q8HFQ only, 0 for others)
}

/// How the embedding table is stored on GPU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmbeddingFormat {
    F32,       // dequantized to F32, use D2D copy
    Q4K,       // raw Q4K blocks, use GPU dequant kernel
    HFQ4G256,  // raw HFQ4-G256 blocks, use GPU dequant kernel
    HFQ4G128,  // raw HFQ4-G128 blocks, use GPU dequant kernel
    Q8_0,  // raw Q8_0 blocks, use GPU dequant kernel
}

/// GPU-resident LLaMA model weights.
pub struct LlamaWeights {
    pub token_embd: GpuTensor,
    pub embd_format: EmbeddingFormat,
    pub output_norm: GpuTensor,
    pub output: WeightTensor,
    pub layers: Vec<LayerWeights>,
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
        let _ = gpu.free_tensor(self.output.buf);
        for l in self.layers {
            let _ = gpu.free_tensor(l.attn_norm);
            let _ = gpu.free_tensor(l.wq.buf);
            let _ = gpu.free_tensor(l.wk.buf);
            let _ = gpu.free_tensor(l.wv.buf);
            let _ = gpu.free_tensor(l.wo.buf);
            if let Some(t) = l.q_norm { let _ = gpu.free_tensor(t); }
            if let Some(t) = l.k_norm { let _ = gpu.free_tensor(t); }
            let _ = gpu.free_tensor(l.ffn_norm);
            let _ = gpu.free_tensor(l.w_gate.buf);
            let _ = gpu.free_tensor(l.w_up.buf);
            let _ = gpu.free_tensor(l.w_down.buf);
        }
    }
}

/// Dispatch GEMV for a weight tensor (quantized or F32).
/// y = W * x where W is the weight tensor, x is F32 input, y is F32 output.
pub fn weight_gemv(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
) -> HipResult<()> {
    match w.gpu_dtype {
        DType::F32 => gpu.gemv_f32(&w.buf, x, y),
        DType::Q4K => gpu.gemv_q4k(&w.buf, x, y, w.m, w.k),
        DType::Q6K => gpu.gemv_q6k(&w.buf, x, y, w.m, w.k),
        DType::Q8_0 => gpu.gemv_q8_0(&w.buf, x, y, w.m, w.k),
        DType::Q8HFQ => gpu.gemv_q8hfq(&w.buf, x, y, w.m, w.k, w.row_stride),
        DType::HFQ4G256 => gpu.gemv_hfq4g256(&w.buf, x, y, w.m, w.k),
        DType::HFQ4G128 => gpu.gemv_hfq4g128(&w.buf, x, y, w.m, w.k),
        DType::MQ4G256 => {
            gpu.ensure_mq_signs()?;
            let x_rot_alias = GpuTensor {
                buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            };
            gpu.gemv_mq4g256_with_rotate(&w.buf, x, y, &x_rot_alias, w.m, w.k)
        }
        DType::MQ6G256 => {
            gpu.ensure_mq_signs()?;
            let x_rot_alias = GpuTensor {
                buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            };
            gpu.gemv_mq6g256_with_rotate(&w.buf, x, y, &x_rot_alias, w.m, w.k)
        }
        DType::MQ3G256 => {
            gpu.ensure_mq_signs()?;
            let x_rot_alias = GpuTensor {
                buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            };
            gpu.gemv_mq3g256_with_rotate(&w.buf, x, y, &x_rot_alias, w.m, w.k)
        }
        DType::MQ2G256 => {
            gpu.ensure_mq_signs()?;
            let x_rot_alias = GpuTensor {
                buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            };
            gpu.gemv_mq2g256_with_rotate(&w.buf, x, y, &x_rot_alias, w.m, w.k)
        }
        DType::MQ8G256 => {
            gpu.ensure_mq_signs()?;
            gpu.gemv_mq8g256_with_rotate(&w.buf, x, y, w.m, w.k)
        }
        DType::HFQ3G256 => gpu.gemv_hfq3g256(&w.buf, x, y, w.m, w.k),
        DType::HFQ3G128 => gpu.gemv_hfq3g128(&w.buf, x, y, w.m, w.k),
        DType::HFQ2G256 => gpu.gemv_hfq2g256(&w.buf, x, y, w.m, w.k),
        DType::HFQ2G128 => gpu.gemv_hfq2g128(&w.buf, x, y, w.m, w.k),
        DType::HFQ6G256 => gpu.gemv_hfq6g256(&w.buf, x, y, w.m, w.k),
        DType::Q4F16G64 => gpu.gemv_q4f16_g64(&w.buf, x, y, w.m, w.k),
        DType::Q4F16G32 => gpu.gemv_q4f16_g32(&w.buf, x, y, w.m, w.k),
        other => {
            eprintln!("WARNING: no GPU kernel for {:?}", other);
            Err(hip_bridge::HipError::new(0, &format!("unsupported dtype {:?}", other)))
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
        DType::MQ4G256 | DType::MQ6G256 | DType::MQ3G256 | DType::MQ2G256 => {
            gpu.fused_rmsnorm_rotate_mq(x, norm_weight, x_rot_scratch, sample_weight.k, eps)?;
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
        DType::MQ4G256 | DType::MQ6G256 | DType::MQ3G256 | DType::MQ2G256 => {
            gpu.rotate_x_mq(x, x_rot_scratch, sample_weight.k)?;
            Ok(Some(x_rot_scratch))
        }
        DType::MQ8G256 => {
            gpu.rotate_quantize_x_mq8(x, sample_weight.k)?;
            Ok(None)
        }
        _ => Ok(None),
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
    match w.gpu_dtype {
        DType::MQ4G256 => {
            if let Some(xr) = x_rot {
                gpu.gemv_mq4g256_prerotated(&w.buf, xr, y, w.m, w.k)
            } else {
                weight_gemv(gpu, w, x, y)
            }
        }
        DType::MQ6G256 => {
            if let Some(xr) = x_rot {
                gpu.gemv_mq6g256_prerotated(&w.buf, xr, y, w.m, w.k)
            } else {
                weight_gemv(gpu, w, x, y)
            }
        }
        DType::MQ3G256 => {
            if let Some(xr) = x_rot {
                gpu.gemv_mq3g256_prerotated(&w.buf, xr, y, w.m, w.k)
            } else {
                weight_gemv(gpu, w, x, y)
            }
        }
        DType::MQ2G256 => {
            if let Some(xr) = x_rot {
                gpu.gemv_mq2g256_prerotated(&w.buf, xr, y, w.m, w.k)
            } else {
                weight_gemv(gpu, w, x, y)
            }
        }
        DType::MQ8G256 => gpu.gemv_mq8g256_prerotated(&w.buf, y, w.m, w.k),
        _ => weight_gemv(gpu, w, x, y),
    }
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
    match w.gpu_dtype {
        DType::HFQ4G256 => gpu.gemv_hfq4g256_residual(&w.buf, x, y, w.m, w.k),
        DType::HFQ6G256 | DType::MQ6G256 => {
            // No dedicated residual GEMV for HFQ6 yet — fall through to generic path.
            let tmp = gpu.alloc_tensor(&[w.m], DType::F32)?;
            weight_gemv(gpu, w, x, &tmp)?;
            gpu.add_inplace_f32(y, &tmp)?;
            gpu.free_tensor(tmp)?;
            Ok(())
        }
        DType::MQ4G256 => {
            gpu.ensure_mq_signs()?;
            let x_rot_alias = GpuTensor {
                buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            };
            gpu.rotate_x_mq(x, &x_rot_alias, w.k)?;
            gpu.gemv_hfq4g256_residual(&w.buf, &x_rot_alias, y, w.m, w.k)
        }
        _ => {
            // Fallback: plain weight_gemv into a scratch, then add_inplace.
            // Allocates a scratch each call — only used for niche dtypes.
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
    if w_down.gpu_dtype == DType::MQ4G256 {
        gpu.ensure_mq_signs()?;
        let x_rot_alias = GpuTensor {
            buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
            shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
            dtype: DType::F32,
        };
        gpu.fused_silu_mul_rotate_mq(gate, up, &x_rot_alias, w_down.k)?;
        return gpu.gemv_hfq4g256_residual(&w_down.buf, &x_rot_alias, x, w_down.m, w_down.k);
    }
    // Non-MQ fallback: plain two-step.
    gpu.silu_mul_f32(gate, up, ffn_hidden_scratch)?;
    weight_gemv_residual(gpu, w_down, ffn_hidden_scratch, x)
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
        _ => {
            // Fallback: repeated GEMV (no batched kernel for this format)
            let x_tok = gpu.alloc_tensor(&[w.k], DType::F32)?;
            let y_tok = gpu.alloc_tensor(&[w.m], DType::F32)?;
            for b in 0..batch_size {
                gpu.hip.memcpy_dtod_at(&x_tok.buf, 0, &x.buf, b * w.k * 4, w.k * 4)?;
                weight_gemv(gpu, w, &x_tok, &y_tok)?;
                gpu.hip.memcpy_dtod_at(&y.buf, b * w.m * 4, &y_tok.buf, 0, w.m * 4)?;
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
        match weights.embd_format {
            EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.token_embd, &x_single, token, dim)?,
            EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.token_embd, &x_single, token, dim)?,
            EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &x_single, token, dim)?,
            EmbeddingFormat::Q4K => gpu.embedding_lookup_q4k(&weights.token_embd, &x_single, token, dim)?,
            EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &x_single, token, dim)?,
        }
        gpu.hip.memcpy_dtod_at(&x_batch.buf, i * dim * 4, &x_single.buf, 0, dim * 4)?;
    }
    gpu.free_tensor(x_single)?;

    // Position array for batched RoPE: [0, 1, 2, ..., batch-1]
    let pos_data: Vec<i32> = (0..batch as i32).collect();
    let pos_bytes: Vec<u8> = pos_data.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let pos_array = gpu.alloc_tensor(&[batch], DType::F32)?;  // i32 same size as f32
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
        gpu.rmsnorm_batched(&x_batch, &layer.attn_norm, &tmp_batch, batch, dim, config.norm_eps)?;

        // Batched QKV projections
        weight_gemm(gpu, &layer.wq, &tmp_batch, &q_batch, batch)?;
        weight_gemm(gpu, &layer.wk, &tmp_batch, &k_batch, batch)?;
        weight_gemm(gpu, &layer.wv, &tmp_batch, &v_batch, batch)?;

        // QK norm (per-position, per-head)
        if config.has_qk_norm {
            if let Some(ref qn) = layer.q_norm {
                gpu.rmsnorm_batched(&q_batch, qn, &q_batch, batch * n_heads, head_dim, config.norm_eps)?;
            }
            if let Some(ref kn) = layer.k_norm {
                gpu.rmsnorm_batched(&k_batch, kn, &k_batch, batch * n_kv_heads, head_dim, config.norm_eps)?;
            }
        }

        // Batched RoPE: all positions in one kernel launch
        gpu.rope_batched_f32(&q_batch, &k_batch, &pos_array,
            n_heads, n_kv_heads, head_dim, config.rope_freq_base, batch)?;

        // Batched KV cache write: all positions in 2 kernel launches (K + V)
        if kv_cache.quantized && kv_cache.quant_q8 {
            gpu.kv_cache_write_q8_0_batched(&kv_cache.k_gpu[layer_idx], &k_batch, &pos_array, n_kv_heads, head_dim, batch)?;
            gpu.kv_cache_write_q8_0_batched(&kv_cache.v_gpu[layer_idx], &v_batch, &pos_array, n_kv_heads, head_dim, batch)?;
        } else {
            for i in 0..batch {
                let pos_i32 = i as i32;
                gpu.hip.memcpy_htod(&pos_buf, &pos_i32.to_ne_bytes())?;
                gpu.hip.memcpy_dtod_at(&k_slice.buf, 0, &k_batch.buf, i * kv_dim * 4, kv_dim * 4)?;
                gpu.hip.memcpy_dtod_at(&v_slice.buf, 0, &v_batch.buf, i * kv_dim * 4, kv_dim * 4)?;
                gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &k_slice, &pos_buf, kv_dim)?;
                gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &v_slice, &pos_buf, kv_dim)?;
            }
        }

        // Batched causal attention: one kernel launch for all positions
        gpu.attention_causal_batched(
            &q_batch, &k_batch, &v_batch, &attn_out_batch,
            batch, n_heads, n_kv_heads, head_dim,
        )?;

        // Batched output projection
        weight_gemm(gpu, &layer.wo, &attn_out_batch, &o_batch, batch)?;

        // Batched residual add: x_batch += o_batch
        gpu.add_inplace_f32(&x_batch, &o_batch)?;

        // Batched FFN norm
        gpu.rmsnorm_batched(&x_batch, &layer.ffn_norm, &tmp_batch, batch, dim, config.norm_eps)?;

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
    gpu.hip.memcpy_dtod_at(&x_last.buf, 0, &x_batch.buf, last_off, dim * 4)?;

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
        let info = gguf.find_tensor(name).unwrap_or_else(|| panic!("tensor not found: {name}"));
        let data = load_tensor_f32(gguf, info);
        gpu.upload_f32(&data, shape)
    }
    // Helper: upload quantized weight (converts Q4_K to Q4_F16_G64 at load time)
    fn up_weight(gguf: &GgufFile, gpu: &Gpu, name: &str, m: usize, k: usize) -> HipResult<WeightTensor> {
        let info = gguf.find_tensor(name).unwrap_or_else(|| panic!("tensor not found: {name}"));
        let raw_data = gguf.tensor_data(info);

        match info.dtype {
            GgmlType::Q4K => {
                let buf = gpu.upload_raw(raw_data, &[raw_data.len()])?;
                Ok(WeightTensor { buf, gpu_dtype: DType::Q4K, m, k, row_stride: 0 })
            }
            GgmlType::Q6K => {
                let buf = gpu.upload_raw(raw_data, &[raw_data.len()])?;
                Ok(WeightTensor { buf, gpu_dtype: DType::Q6K, m, k, row_stride: 0 })
            }
            GgmlType::Q8_0 => {
                let buf = gpu.upload_raw(raw_data, &[raw_data.len()])?;
                Ok(WeightTensor { buf, gpu_dtype: DType::Q8_0, m, k, row_stride: 0 })
            }
            GgmlType::F32 => {
                let buf = gpu.upload_raw(raw_data, &[raw_data.len()])?;
                Ok(WeightTensor { buf, gpu_dtype: DType::F32, m, k, row_stride: 0 })
            }
            _ => {
                // Unsupported: dequant to F32 on CPU, upload as raw bytes
                let data = load_tensor_f32(gguf, info);
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
                };
                let buf = gpu.upload_raw(bytes, &[bytes.len()])?;
                Ok(WeightTensor { buf, gpu_dtype: DType::F32, m, k, row_stride: 0 })
            }
        }
    }

    eprintln!("  loading token_embd...");
    let embd_info = gguf.find_tensor("token_embd.weight").expect("token_embd not found");
    let (token_embd, embd_fmt) = if embd_info.dtype == GgmlType::Q4K {
        let raw = gguf.tensor_data(embd_info);
        eprintln!("    (Q4K raw, {} MB — saves {} MB vs F32)",
            raw.len() / 1_000_000,
            (config.vocab_size * config.dim * 4 - raw.len()) / 1_000_000);
        (gpu.upload_raw(raw, &[raw.len()])?, EmbeddingFormat::Q4K)
    } else {
        let data = load_tensor_f32(gguf, embd_info);
        (gpu.upload_f32(&data, &[config.vocab_size, config.dim])?, EmbeddingFormat::F32)
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
        WeightTensor { buf, gpu_dtype: DType::F32, m: config.vocab_size, k: config.dim, row_stride: 0 }
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
            wq: up_weight(gguf, gpu, &format!("{p}.attn_q.weight"), q_out_dim, config.dim)?,
            wk: up_weight(gguf, gpu, &format!("{p}.attn_k.weight"), kv_dim, config.dim)?,
            wv: up_weight(gguf, gpu, &format!("{p}.attn_v.weight"), kv_dim, config.dim)?,
            wo: up_weight(gguf, gpu, &format!("{p}.attn_output.weight"), config.dim, q_out_dim)?,
            q_norm: if config.has_qk_norm {
                Some(up_f32(gguf, gpu, &format!("{p}.attn_q_norm.weight"), &[config.head_dim])?)
            } else {
                None
            },
            k_norm: if config.has_qk_norm {
                Some(up_f32(gguf, gpu, &format!("{p}.attn_k_norm.weight"), &[config.head_dim])?)
            } else {
                None
            },
            ffn_norm: up_f32(gguf, gpu, &format!("{p}.ffn_norm.weight"), &[config.dim])?,
            w_gate: up_weight(gguf, gpu, &format!("{p}.ffn_gate.weight"), config.hidden_dim, config.dim)?,
            w_up: up_weight(gguf, gpu, &format!("{p}.ffn_up.weight"), config.hidden_dim, config.dim)?,
            w_down: up_weight(gguf, gpu, &format!("{p}.ffn_down.weight"), config.dim, config.hidden_dim)?,
        };
        layers.push(layer);
    }

    Ok(LlamaWeights {
        token_embd,
        embd_format: embd_fmt,
        output_norm,
        output,
        layers,
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
    pub attn_partials: GpuTensor,  // flash-decoding partial results
    pub pos_buf: hip_bridge::DeviceBuffer,
    /// FWHT-rotated x scratch for MagnumQuant batching. Sized to max(dim, hidden_dim).
    pub x_rot: GpuTensor,
}

impl ForwardScratch {
    pub fn new(gpu: &mut Gpu, config: &LlamaConfig) -> HipResult<Self> {
        let dim = config.dim;
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;
        // Flash-decoding partials: n_heads × max_chunks × (2 + head_dim) floats
        // max_chunks = ceil(2048 / 128) = 16
        let max_chunks = 16;
        let partial_stride = 2 + config.head_dim;
        let partials_size = config.n_heads * max_chunks * partial_stride;
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
            pos_buf: gpu.hip.malloc(4)?,  // single i32
            x_rot: gpu.alloc_tensor(&[dim.max(config.hidden_dim)], DType::F32)?,
        })
    }

    /// Return all GPU buffers to the pool (drained on unload). Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in [self.x, self.tmp, self.q, self.k, self.v, self.attn_out,
                  self.o, self.gate, self.up, self.ffn_hidden, self.ffn_out,
                  self.logits, self.sample_buf, self.repeat_buf,
                  self.attn_partials, self.x_rot] {
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
    forward_scratch_layers(gpu, weights, config, pos, kv_cache, scratch, temperature, top_p, rng_state, repeat_window, repeat_penalty)
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
    gpu.hip.memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
    // Embedding lookup
    match weights.embd_format {
        EmbeddingFormat::Q4K => gpu.embedding_lookup_q4k(&weights.token_embd, &scratch.x, token, dim)?,
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &scratch.x, token, dim)?,
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.token_embd, &scratch.x, token, dim)?,
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.token_embd, &scratch.x, token, dim)?,
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &scratch.x, token, dim)?,
    }
    Ok(())
}

/// Layer loop + final norm + logits + sampling. Graph-capturable.
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
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;
    let kv_dim = n_kv_heads * head_dim;

    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];

        gpu.rmsnorm_f32(&scratch.x, &layer.attn_norm, &scratch.tmp, config.norm_eps)?;

        if layer.wq.gpu_dtype == DType::Q4K && layer.wk.gpu_dtype == DType::Q4K {
            gpu.fused_qkv_q4k(
                &layer.wq.buf, &layer.wk.buf, &layer.wv.buf,
                &scratch.tmp, &scratch.q, &scratch.k, &scratch.v,
                layer.wq.m, layer.wk.m, layer.wv.m, layer.wq.k,
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
                gpu.rmsnorm_batched(&scratch.q, qn, &scratch.q, n_heads, head_dim, config.norm_eps)?;
            }
            if let Some(ref kn) = layer.k_norm {
                gpu.rmsnorm_batched(&scratch.k, kn, &scratch.k, n_kv_heads, head_dim, config.norm_eps)?;
            }
        }

        gpu.rope_f32(&scratch.q, &scratch.k, &scratch.pos_buf, n_heads, n_kv_heads, head_dim, config.rope_freq_base)?;

        if kv_cache.quant_hfq4 {
            gpu.kv_cache_write_hfq4(&kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.kv_cache_write_hfq4(&kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.attention_hfq4_kv(
                &scratch.q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out, &scratch.pos_buf, pos + 1, n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized && !kv_cache.k_scales.is_empty() && !kv_cache.quant_int8 && !kv_cache.quant_q8 {
            // HFQ8 flat layout
            gpu.kv_cache_write_hfq8(&kv_cache.k_gpu[layer_idx], &kv_cache.k_scales[layer_idx], &scratch.k, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.kv_cache_write_hfq8(&kv_cache.v_gpu[layer_idx], &kv_cache.v_scales[layer_idx], &scratch.v, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.attention_hfq8_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx], &kv_cache.k_scales[layer_idx],
                &kv_cache.v_gpu[layer_idx], &kv_cache.v_scales[layer_idx],
                &scratch.attn_out, &scratch.pos_buf, pos + 1, n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
            )?;
        } else if kv_cache.quant_int8 {
            gpu.kv_cache_write_int8c_f16(&kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.kv_cache_write_int8c_f16(&kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.attention_int8c_f16_kv(
                &scratch.q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out, &scratch.pos_buf, pos + 1, n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized && kv_cache.quant_q8 {
            gpu.kv_cache_write_q8_0(&kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.kv_cache_write_q8_0(&kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.attention_q8_0_kv(
                &scratch.q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out, &scratch.pos_buf, pos + 1, n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized {
            gpu.kv_cache_write_q4(&kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.kv_cache_write_q4(&kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.attention_q4kv(
                &scratch.q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out, &scratch.pos_buf, pos + 1, n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
            )?;
        } else {
            gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf, kv_dim)?;
            gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf, kv_dim)?;
            gpu.attention_f32(
                &scratch.q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out, &scratch.pos_buf, pos + 1, n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
            )?;
        }

        weight_gemv(gpu, &layer.wo, &scratch.attn_out, &scratch.o)?;
        gpu.add_inplace_f32(&scratch.x, &scratch.o)?;

        gpu.rmsnorm_f32(&scratch.x, &layer.ffn_norm, &scratch.tmp, config.norm_eps)?;
        if layer.w_gate.gpu_dtype == DType::Q4K && layer.w_up.gpu_dtype == DType::Q4K {
            gpu.fused_gate_up_q4k(
                &layer.w_gate.buf, &layer.w_up.buf,
                &scratch.tmp, &scratch.gate, &scratch.up,
                layer.w_gate.m, layer.w_up.m, layer.w_gate.k,
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

    gpu.rmsnorm_f32(&scratch.x, &weights.output_norm, &scratch.tmp, config.norm_eps)?;
    weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;

    // GPU-side sampling (includes sync readback — can't be in graph capture)
    gpu.sample_top_p(
        &scratch.logits, &scratch.sample_buf, &scratch.repeat_buf,
        config.vocab_size, temperature, top_p, rng_state,
        repeat_window, repeat_penalty,
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
    exit_threshold: f32,       // max softmax prob threshold (e.g., 0.9)
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
                &layer.wq.buf, &layer.wk.buf, &layer.wv.buf,
                &scratch.tmp, &scratch.q, &scratch.k, &scratch.v,
                layer.wq.m, layer.wk.m, layer.wv.m, layer.wq.k,
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
                gpu.rmsnorm_batched(&scratch.q, qn, &scratch.q, n_heads, head_dim, config.norm_eps)?;
            }
            if let Some(ref kn) = layer.k_norm {
                gpu.rmsnorm_batched(&scratch.k, kn, &scratch.k, n_kv_heads, head_dim, config.norm_eps)?;
            }
        }

        gpu.rope_f32(&scratch.q, &scratch.k, &scratch.pos_buf, n_heads, n_kv_heads, head_dim, config.rope_freq_base)?;

        // KV write + attention (use same dispatch as forward_scratch_layers)
        if kv_cache.quantized && kv_cache.quant_q8 {
            gpu.kv_cache_write_q8_0(&kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.kv_cache_write_q8_0(&kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.attention_q8_0_kv(
                &scratch.q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out, &scratch.pos_buf, pos + 1, n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
            )?;
        } else {
            gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf, kv_dim)?;
            gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf, kv_dim)?;
            gpu.attention_f32(
                &scratch.q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out, &scratch.pos_buf, pos + 1, n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
            )?;
        }

        weight_gemv(gpu, &layer.wo, &scratch.attn_out, &scratch.o)?;
        gpu.add_inplace_f32(&scratch.x, &scratch.o)?;

        gpu.rmsnorm_f32(&scratch.x, &layer.ffn_norm, &scratch.tmp, config.norm_eps)?;
        if layer.w_gate.gpu_dtype == DType::Q4K && layer.w_up.gpu_dtype == DType::Q4K {
            gpu.fused_gate_up_q4k(
                &layer.w_gate.buf, &layer.w_up.buf,
                &scratch.tmp, &scratch.gate, &scratch.up,
                layer.w_gate.m, layer.w_up.m, layer.w_gate.k,
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
            gpu.rmsnorm_f32(&scratch.x, &weights.output_norm, &scratch.tmp, config.norm_eps)?;
            weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;

            // GPU-side confidence check: compute max(softmax) on GPU, download 4 bytes
            gpu.max_prob(&scratch.logits, &scratch.sample_buf, config.vocab_size)?;
            let mut prob_bytes = [0u8; 4];
            gpu.hip.memcpy_dtoh(&mut prob_bytes, &scratch.sample_buf.buf)?;
            let max_prob = f32::from_ne_bytes(prob_bytes);

            if max_prob >= exit_threshold {
                exit_layer = layer_idx + 1;
                // Sample from these logits
                let (tok, rng) = gpu.sample_top_p(
                    &scratch.logits, &scratch.sample_buf, &scratch.repeat_buf,
                    config.vocab_size, temperature, top_p, rng_state,
                    repeat_window, repeat_penalty,
                )?;
                return Ok((tok, rng, exit_layer));
            }
        }
    }

    // No early exit — run full final norm + logits + sampling
    gpu.rmsnorm_f32(&scratch.x, &weights.output_norm, &scratch.tmp, config.norm_eps)?;
    weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;
    let (tok, rng) = gpu.sample_top_p(
        &scratch.logits, &scratch.sample_buf, &scratch.repeat_buf,
        config.vocab_size, temperature, top_p, rng_state,
        repeat_window, repeat_penalty,
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
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;
    let kv_dim = n_kv_heads * head_dim;

    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];
        gpu.rmsnorm_f32(&scratch.x, &layer.attn_norm, &scratch.tmp, config.norm_eps)?;

        if layer.wq.gpu_dtype == DType::Q4K && layer.wk.gpu_dtype == DType::Q4K {
            gpu.fused_qkv_q4k(
                &layer.wq.buf, &layer.wk.buf, &layer.wv.buf,
                &scratch.tmp, &scratch.q, &scratch.k, &scratch.v,
                layer.wq.m, layer.wk.m, layer.wv.m, layer.wq.k,
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
                gpu.rmsnorm_batched(&scratch.q, qn, &scratch.q, n_heads, head_dim, config.norm_eps)?;
            }
            if let Some(ref kn) = layer.k_norm {
                gpu.rmsnorm_batched(&scratch.k, kn, &scratch.k, n_kv_heads, head_dim, config.norm_eps)?;
            }
        }

        gpu.rope_f32(&scratch.q, &scratch.k, &scratch.pos_buf, n_heads, n_kv_heads, head_dim, config.rope_freq_base)?;

        if kv_cache.quant_hfq4 {
            gpu.kv_cache_write_hfq4(&kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.kv_cache_write_hfq4(&kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.attention_hfq4_kv(
                &scratch.q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out, &scratch.pos_buf, pos + 1, n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized && !kv_cache.k_scales.is_empty() && !kv_cache.quant_int8 && !kv_cache.quant_q8 {
            // HFQ8 flat layout
            gpu.kv_cache_write_hfq8(&kv_cache.k_gpu[layer_idx], &kv_cache.k_scales[layer_idx], &scratch.k, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.kv_cache_write_hfq8(&kv_cache.v_gpu[layer_idx], &kv_cache.v_scales[layer_idx], &scratch.v, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.attention_hfq8_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx], &kv_cache.k_scales[layer_idx],
                &kv_cache.v_gpu[layer_idx], &kv_cache.v_scales[layer_idx],
                &scratch.attn_out, &scratch.pos_buf, pos + 1, n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
            )?;
        } else if kv_cache.quant_int8 {
            gpu.kv_cache_write_int8c_f16(&kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.kv_cache_write_int8c_f16(&kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.attention_int8c_f16_kv(
                &scratch.q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out, &scratch.pos_buf, pos + 1, n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized && kv_cache.quant_q8 {
            gpu.kv_cache_write_q8_0(&kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.kv_cache_write_q8_0(&kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.attention_q8_0_kv(
                &scratch.q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out, &scratch.pos_buf, pos + 1, n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized {
            gpu.kv_cache_write_q4(&kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.kv_cache_write_q4(&kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf, n_kv_heads, head_dim)?;
            gpu.attention_q4kv(
                &scratch.q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out, &scratch.pos_buf, pos + 1, n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
            )?;
        } else {
            gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf, kv_dim)?;
            gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf, kv_dim)?;
            gpu.attention_f32(
                &scratch.q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out, &scratch.pos_buf, pos + 1, n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
            )?;
        }

        weight_gemv(gpu, &layer.wo, &scratch.attn_out, &scratch.o)?;
        gpu.add_inplace_f32(&scratch.x, &scratch.o)?;

        gpu.rmsnorm_f32(&scratch.x, &layer.ffn_norm, &scratch.tmp, config.norm_eps)?;
        if layer.w_gate.gpu_dtype == DType::Q4K && layer.w_up.gpu_dtype == DType::Q4K {
            gpu.fused_gate_up_q4k(
                &layer.w_gate.buf, &layer.w_up.buf,
                &scratch.tmp, &scratch.gate, &scratch.up,
                layer.w_gate.m, layer.w_up.m, layer.w_gate.k,
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

    gpu.rmsnorm_f32(&scratch.x, &weights.output_norm, &scratch.tmp, config.norm_eps)?;
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
    match weights.embd_format {
        EmbeddingFormat::Q4K => gpu.embedding_lookup_q4k(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &x, token, dim)?,
    }

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
                &layer.wq.buf, &layer.wk.buf, &layer.wv.buf,
                &tmp,
                &q, &k, &v,
                layer.wq.m, layer.wk.m, layer.wv.m, layer.wq.k,
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
        gpu.rope_f32(&q, &k, &pos_buf, n_heads, n_kv_heads, head_dim, config.rope_freq_base)?;

        // Store K, V in GPU cache + attention
        gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &k, &pos_buf, kv_dim)?;
        gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &v, &pos_buf, kv_dim)?;
        gpu.attention_f32(
            &q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
            &attn_out, &pos_buf, pos + 1, n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
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
                &layer.w_gate.buf, &layer.w_up.buf,
                &tmp,
                &gate, &up,
                layer.w_gate.m, layer.w_up.m, layer.w_gate.k,
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
        &logits_on_gpu, sample_buf, repeat_buf,
        config.vocab_size, temperature, top_p, rng_state,
        repeat_window, repeat_penalty,
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
    match weights.embd_format {
        EmbeddingFormat::Q4K => gpu.embedding_lookup_q4k(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &x, token, dim)?,
    }

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
                &layer.wq.buf, &layer.wk.buf, &layer.wv.buf,
                &tmp, &q, &k, &v,
                layer.wq.m, layer.wk.m, layer.wv.m, layer.wq.k,
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

        gpu.rope_f32(&q, &k, &pos_buf, n_heads, n_kv_heads, head_dim, config.rope_freq_base)?;

        gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &k, &pos_buf, kv_dim)?;
        gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &v, &pos_buf, kv_dim)?;

        gpu.attention_f32(
            &q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
            &attn_out, &pos_buf, pos + 1, n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
        )?;

        weight_gemv(gpu, &layer.wo, &attn_out, &o)?;
        gpu.add_inplace_f32(&x, &o)?;

        gpu.rmsnorm_f32(&x, &layer.ffn_norm, &tmp, config.norm_eps)?;
        if layer.w_gate.gpu_dtype == DType::Q4K && layer.w_up.gpu_dtype == DType::Q4K {
            gpu.fused_gate_up_q4k(
                &layer.w_gate.buf, &layer.w_up.buf,
                &tmp, &gate, &up,
                layer.w_gate.m, layer.w_up.m, layer.w_gate.k,
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

/// GPU-resident KV cache for autoregressive generation.
///
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
    pub k_gpu: Vec<GpuTensor>,   // [n_layers] key values (FP32 or int8)
    pub v_gpu: Vec<GpuTensor>,   // [n_layers] value values (FP32 or int8)
    pub k_scales: Vec<GpuTensor>,// [n_layers] key scales (for INT8 mode)
    pub v_scales: Vec<GpuTensor>,// [n_layers] value scales (for INT8 mode)
    pub kv_dim: usize,
    pub max_seq: usize,
    /// Physical capacity of each per-layer k/v buffer in *tokens*.
    /// Equals `max_seq` unless the buffer was sized for eviction-bounded use.
    pub physical_cap: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub quantized: bool,
    pub quant_q8: bool,
    pub quant_int8: bool,        // true = INT8 with separate scales
    pub quant_hfq4: bool,        // true = HFQ4 co-located blocks (72 bytes/head)
    pub quant_asym4: bool,       // true = K at 4-bit rotated, V at Q8_0 — RotorQuant planar4 asymmetric
    pub quant_asym3: bool,       // true = K at givens3 (rotated 3-bit Lloyd-Max), V at Q8_0 — best-quality rotated K per RotorQuant
    pub quant_asym2: bool,       // true = K at givens2 (rotated 2-bit), V at Q8_0 (normal space)
    pub boundary_layers: u8,     // number of boundary layers at each end (default 2)
    pub givens_cos: Option<GpuTensor>,  // Givens rotation cos table (n_blocks × f32)
    pub givens_sin: Option<GpuTensor>,  // Givens rotation sin table (n_blocks × f32)
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

impl KvCache {
    /// Check if a given KV layer ordinal is a boundary layer (first N + last N).
    pub fn is_boundary(&self, kv_ordinal: usize) -> bool {
        kv_ordinal < self.layer_is_boundary.len() && self.layer_is_boundary[kv_ordinal]
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
        Ok(Self { k_gpu, v_gpu, k_scales: vec![], v_scales: vec![], kv_dim, max_seq: max_seq_len, physical_cap: max_seq_len, n_kv_heads, head_dim, quantized: false, quant_q8: false, quant_int8: false, quant_hfq4: false, quant_asym4: false, quant_asym3: false, quant_asym2: false, boundary_layers: 0, givens_cos: None, givens_sin: None, layer_is_boundary: vec![], compact_offset: 0 })
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
        Ok(Self { k_gpu, v_gpu, k_scales: vec![], v_scales: vec![], kv_dim, max_seq: max_seq_len, physical_cap: max_seq_len, n_kv_heads, head_dim, quantized: true, quant_q8: false, quant_int8: false, quant_hfq4: false, quant_asym4: false, quant_asym3: false, quant_asym2: false, boundary_layers: 0, givens_cos: None, givens_sin: None, layer_is_boundary: vec![], compact_offset: 0 })
    }

    /// Create Q8_0 quantized KV cache (GGML Q8_0 format). 3.76x smaller than FP32.
    /// Block: [f16 scale (2B)][int8 × 32 (32B)] = 34 bytes per 32 elements.
    /// head_dim=128 → 4 blocks × 34 = 136 bytes per head.
    pub fn new_gpu_q8(
        gpu: &mut Gpu, n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_q8_capped(gpu, n_layers, n_kv_heads, head_dim, max_seq_len, max_seq_len)
    }

    /// Same as [`new_gpu_q8`] with an explicit physical_cap. Eviction-aware.
    pub fn new_gpu_q8_capped(
        gpu: &mut Gpu, n_layers: usize, n_kv_heads: usize, head_dim: usize,
        max_seq_len: usize, physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]");
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
        Ok(Self { k_gpu, v_gpu, k_scales: vec![], v_scales: vec![], kv_dim, max_seq: max_seq_len, physical_cap, n_kv_heads, head_dim, quantized: true, quant_q8: true, quant_int8: false, quant_hfq4: false, quant_asym4: false, quant_asym3: false, quant_asym2: false, boundary_layers: 0, givens_cos: None, givens_sin: None, layer_is_boundary: vec![], compact_offset: 0 })
    }

    /// Create INT8 co-located KV cache: [f32 scale][pad 4B][int8 × head_dim] = 136 bytes per head.
    pub fn new_gpu_int8c(
        gpu: &mut Gpu, n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize,
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
        Ok(Self { k_gpu, v_gpu, k_scales: vec![], v_scales: vec![], kv_dim, max_seq: max_seq_len, physical_cap: max_seq_len, n_kv_heads, head_dim, quantized: true, quant_q8: false, quant_int8: true, quant_hfq4: false, quant_asym4: false, quant_asym3: false, quant_asym2: false, boundary_layers: 0, givens_cos: None, givens_sin: None, layer_is_boundary: vec![], compact_offset: 0 })
    }

    /// Create HFQ4 KV cache: co-located blocks. 72 bytes per head (scale+zero+nibbles).
    pub fn new_gpu_hfq4kv(
        gpu: &mut Gpu, n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize,
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
        Ok(Self { k_gpu, v_gpu, k_scales: vec![], v_scales: vec![], kv_dim, max_seq: max_seq_len, physical_cap: max_seq_len, n_kv_heads, head_dim, quantized: true, quant_q8: false, quant_int8: false, quant_hfq4: true, quant_asym4: false, quant_asym3: false, quant_asym2: false, boundary_layers: 0, givens_cos: None, givens_sin: None, layer_is_boundary: vec![], compact_offset: 0 })
    }

    /// Create HFQ8 KV cache: FP32 scale+zero per head, contiguous uint8 data.
    pub fn new_gpu_hfq8(
        gpu: &mut Gpu, n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize,
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
        Ok(Self { k_gpu, v_gpu, k_scales, v_scales, kv_dim, max_seq: max_seq_len, physical_cap: max_seq_len, n_kv_heads, head_dim, quantized: true, quant_q8: false, quant_int8: false, quant_hfq4: false, quant_asym4: false, quant_asym3: false, quant_asym2: false, boundary_layers: 0, givens_cos: None, givens_sin: None, layer_is_boundary: vec![], compact_offset: 0 })
    }

    /// Create INT8 KV cache with separate scale arrays. Clean contiguous layout.
    pub fn new_gpu_int8(
        gpu: &mut Gpu, n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize,
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
        Ok(Self { k_gpu, v_gpu, k_scales, v_scales, kv_dim, max_seq: max_seq_len, physical_cap: max_seq_len, n_kv_heads, head_dim, quantized: true, quant_q8: false, quant_int8: true, quant_hfq4: false, quant_asym4: false, quant_asym3: false, quant_asym2: false, boundary_layers: 0, givens_cos: None, givens_sin: None, layer_is_boundary: vec![], compact_offset: 0 })
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
        gpu: &mut Gpu, n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_asym4_capped(gpu, n_layers, n_kv_heads, head_dim, max_seq_len, max_seq_len)
    }

    /// Same as [`new_gpu_asym4`] with an explicit physical_cap. Eviction-aware.
    pub fn new_gpu_asym4_capped(
        gpu: &mut Gpu, n_layers: usize, n_kv_heads: usize, head_dim: usize,
        max_seq_len: usize, physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(head_dim == 128 || head_dim == 256, "asym4 requires head_dim=128 or 256");
        assert!(head_dim % 32 == 0);
        assert!(physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]");
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
        eprintln!("KV cache: asym4 (K rotated-4b {k_bph}B + V Q8 {v_bph}B = {} B/head, {:.1}x vs fp32)",
            k_bph + v_bph, (head_dim * 4 * 2) as f64 / (k_bph + v_bph) as f64);
        Ok(Self {
            k_gpu, v_gpu, k_scales: vec![], v_scales: vec![], kv_dim,
            max_seq: max_seq_len, physical_cap, n_kv_heads, head_dim,
            quantized: true, quant_q8: false, quant_int8: false, quant_hfq4: false,
            quant_asym4: true, quant_asym3: false, quant_asym2: false,
            boundary_layers: 0, givens_cos: Some(ct), givens_sin: Some(st),
            layer_is_boundary: vec![],
            compact_offset: 0,
        })
    }

    /// Create asym3 KV cache: K at 3-bit rotated (Lloyd-Max N(0, 1/256)), V at Q8_0.
    /// head_dim=256 → K=100 B/head, V=272 B/head → 372 B/head (5.5× vs fp32).
    /// Back-compat wrapper: allocates physical_cap == max_seq_len slots per layer.
    pub fn new_gpu_asym3(
        gpu: &mut Gpu, n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_asym3_capped(gpu, n_layers, n_kv_heads, head_dim, max_seq_len, max_seq_len)
    }

    /// Same as [`new_gpu_asym3`] but with an explicit physical capacity. When
    /// `physical_cap < max_seq_len`, the cache is sized for `physical_cap`
    /// tokens along the time axis; the caller is responsible for triggering
    /// TriAttention/CASK eviction before the physical position overruns
    /// `physical_cap`. `max_seq_len` is retained for RoPE/mask purposes.
    pub fn new_gpu_asym3_capped(
        gpu: &mut Gpu, n_layers: usize, n_kv_heads: usize, head_dim: usize,
        max_seq_len: usize, physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(head_dim == 256, "asym3 currently requires head_dim=256 (Qwen 3.5)");
        assert!(head_dim % 32 == 0);
        assert!(physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]");
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
            k_gpu, v_gpu, k_scales: vec![], v_scales: vec![], kv_dim,
            max_seq: max_seq_len, physical_cap, n_kv_heads, head_dim,
            quantized: true, quant_q8: false, quant_int8: false, quant_hfq4: false,
            quant_asym4: false, quant_asym3: true, quant_asym2: false,
            boundary_layers: 0, givens_cos: Some(ct), givens_sin: Some(st),
            layer_is_boundary: vec![],
            compact_offset: 0,
        })
    }

    /// Create asym2 KV cache: K at 2-bit rotated, V at Q8_0.
    /// head_dim=256 → K=68 B/head, V=272 B/head → 340 B/head (6.0× vs fp32).
    /// Back-compat wrapper: `physical_cap == max_seq_len`.
    pub fn new_gpu_asym2(
        gpu: &mut Gpu, n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_asym2_capped(gpu, n_layers, n_kv_heads, head_dim, max_seq_len, max_seq_len)
    }

    /// Same as [`new_gpu_asym2`] with an explicit physical_cap. Eviction-aware.
    pub fn new_gpu_asym2_capped(
        gpu: &mut Gpu, n_layers: usize, n_kv_heads: usize, head_dim: usize,
        max_seq_len: usize, physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(head_dim == 128 || head_dim == 256, "asym2 requires head_dim=128 or 256");
        assert!(head_dim % 32 == 0);
        assert!(physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]");
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
        eprintln!("KV cache: asym2 (K rotated-2b {k_bph}B + V Q8 {v_bph}B = {} B/head, {:.1}x vs fp32)",
            k_bph + v_bph, (head_dim * 4 * 2) as f64 / (k_bph + v_bph) as f64);
        Ok(Self {
            k_gpu, v_gpu, k_scales: vec![], v_scales: vec![], kv_dim,
            max_seq: max_seq_len, physical_cap, n_kv_heads, head_dim,
            quantized: true, quant_q8: false, quant_int8: false, quant_hfq4: false,
            quant_asym4: false, quant_asym3: false, quant_asym2: true,
            boundary_layers: 0, givens_cos: Some(ct), givens_sin: Some(st),
            layer_is_boundary: vec![],
            compact_offset: 0,
        })
    }

    /// Generate deterministic ±1 sign array for FWHT.
    pub fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
        let mut state = seed;
        (0..n).map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (state >> 16) & 1 == 1 { 1.0f32 } else { -1.0f32 }
        }).collect()
    }

    /// Free all GPU tensors in this cache. Call before drop to return VRAM.
    /// After calling, follow with gpu.drain_pool() to actually release memory.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in self.k_gpu { let _ = gpu.free_tensor(t); }
        for t in self.v_gpu { let _ = gpu.free_tensor(t); }
        for t in self.k_scales { let _ = gpu.free_tensor(t); }
        for t in self.v_scales { let _ = gpu.free_tensor(t); }
        if let Some(t) = self.givens_cos { let _ = gpu.free_tensor(t); }
        if let Some(t) = self.givens_sin { let _ = gpu.free_tensor(t); }
    }

    /// Store K, V at position `pos` in layer cache (CPU → GPU copy into cache slot).
    pub fn store_kv_pub(&mut self, gpu: &Gpu, layer: usize, pos: usize, k: &[f32], v: &[f32]) -> HipResult<()> {
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
        let k_bytes = unsafe {
            std::slice::from_raw_parts(k_data.as_ptr() as *const u8, k_data.len() * 4)
        };
        let v_bytes = unsafe {
            std::slice::from_raw_parts(v_data.as_ptr() as *const u8, v_data.len() * 4)
        };
        gpu.hip.memcpy_htod_offset(&self.k_gpu[layer].buf, byte_offset, k_bytes)?;
        gpu.hip.memcpy_htod_offset(&self.v_gpu[layer].buf, byte_offset, v_bytes)?;
        Ok(())
    }
}

// attention_cpu removed — GPU attention is now used

/// Sample the next token from logits using argmax (greedy).
pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap()
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
        Self { think_temp: 0.3, answer_temp: 0.3, top_p: 0.8, repeat_penalty: 1.15, repeat_window: 128 }
    }
    /// VL thinking model.
    pub fn vl_thinking() -> Self {
        Self { think_temp: 0.3, answer_temp: 0.3, top_p: 0.8, repeat_penalty: 1.15, repeat_window: 128 }
    }
    /// Simple greedy-ish sampling (no think/answer split).
    pub fn simple() -> Self {
        Self { think_temp: 0.7, answer_temp: 0.3, top_p: 0.9, repeat_penalty: 1.1, repeat_window: 64 }
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
        if recency > entry.1 { entry.1 = recency; }
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

pub fn sample_top_p(logits: &[f32], temperature: f32, top_p: f32) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits);
    }
    let top_p = top_p.clamp(0.0, 1.0);
    const TOP_K: usize = 20;

    let inv_temp = 1.0 / temperature;

    // Single pass: find max AND top-K indices from raw logits simultaneously.
    // Uses a fixed-size array (no heap alloc) with manual min-tracking.
    let mut topk_val = [f32::NEG_INFINITY; TOP_K];
    let mut topk_idx = [0u32; TOP_K];
    let mut min_pos = 0usize; // index of smallest element in topk
    let mut min_val = f32::NEG_INFINITY;
    let mut max_logit = f32::NEG_INFINITY;

    for (i, &l) in logits.iter().enumerate() {
        if l > max_logit { max_logit = l; }
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

    // Softmax only the K candidates (temperature-scaled)
    let mut probs = [0.0f32; TOP_K];
    let mut sum = 0.0f32;
    for i in 0..TOP_K {
        let p = ((topk_val[i] - max_logit) * inv_temp).exp();
        probs[i] = p;
        sum += p;
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
                if acc2 >= cumulative { break; }
            }
            return topk_idx[order[0]];
        }
    }
    topk_idx[order[0]]
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
    if window_len == 0.0 { return; }

    let mut counts = std::collections::HashMap::<u32, (u32, f32)>::new();
    for (i, &t) in recent.iter().enumerate() {
        let recency = (i as f32 + 1.0) / window_len;
        let entry = counts.entry(t).or_insert((0, 0.0));
        entry.0 += 1;
        if recency > entry.1 { entry.1 = recency; }
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
        if l > max_logit { max_logit = l; }
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
                if acc2 >= cumulative { break; }
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
        if s == 0 { s = 1; }
    }
    // xorshift32
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    SAMPLER_STATE.store(s, Ordering::Relaxed);
    (s as f32) / (u32::MAX as f32)
}
