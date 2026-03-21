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
    F32,   // dequantized to F32, use D2D copy
    Q4K,   // raw Q4K blocks, use GPU dequant kernel
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
        DType::Q4F16G64 => gpu.gemv_q4f16_g64(&w.buf, x, y, w.m, w.k),
        DType::Q4F16G32 => gpu.gemv_q4f16_g32(&w.buf, x, y, w.m, w.k),
        other => {
            eprintln!("WARNING: no GPU kernel for {:?}", other);
            Err(hip_bridge::HipError::new(0, &format!("unsupported dtype {:?}", other)))
        }
    }
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
}

impl ForwardScratch {
    pub fn new(gpu: &mut Gpu, config: &LlamaConfig) -> HipResult<Self> {
        let dim = config.dim;
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;
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
        })
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
) -> HipResult<(u32, u32)> {
    let dim = config.dim;
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;
    let kv_dim = n_kv_heads * head_dim;

    // Embedding lookup
    match weights.embd_format {
        EmbeddingFormat::Q4K => gpu.embedding_lookup_q4k(&weights.token_embd, &scratch.x, token, dim)?,
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &scratch.x, token, dim)?,
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &scratch.x, token, dim)?,
    }

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
            weight_gemv(gpu, &layer.wq, &scratch.tmp, &scratch.q)?;
            weight_gemv(gpu, &layer.wk, &scratch.tmp, &scratch.k)?;
            weight_gemv(gpu, &layer.wv, &scratch.tmp, &scratch.v)?;
        }

        if config.has_qk_norm {
            if let Some(ref qn) = layer.q_norm {
                gpu.rmsnorm_batched(&scratch.q, qn, &scratch.q, n_heads, head_dim, config.norm_eps)?;
            }
            if let Some(ref kn) = layer.k_norm {
                gpu.rmsnorm_batched(&scratch.k, kn, &scratch.k, n_kv_heads, head_dim, config.norm_eps)?;
            }
        }

        gpu.rope_f32(&scratch.q, &scratch.k, pos, n_heads, n_kv_heads, head_dim, config.rope_freq_base)?;

        let cache_byte_offset = pos * kv_dim * 4;
        gpu.hip.memcpy_dtod_at(&kv_cache.k_gpu[layer_idx].buf, cache_byte_offset, &scratch.k.buf, 0, kv_dim * 4)?;
        gpu.hip.memcpy_dtod_at(&kv_cache.v_gpu[layer_idx].buf, cache_byte_offset, &scratch.v.buf, 0, kv_dim * 4)?;

        gpu.attention_f32(
            &scratch.q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
            &scratch.attn_out, pos + 1, n_heads, n_kv_heads, head_dim, kv_cache.max_seq,
        )?;

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
            weight_gemv(gpu, &layer.w_gate, &scratch.tmp, &scratch.gate)?;
            weight_gemv(gpu, &layer.w_up, &scratch.tmp, &scratch.up)?;
        }

        gpu.silu_mul_f32(&scratch.gate, &scratch.up, &scratch.ffn_hidden)?;
        weight_gemv(gpu, &layer.w_down, &scratch.ffn_hidden, &scratch.ffn_out)?;
        gpu.add_inplace_f32(&scratch.x, &scratch.ffn_out)?;
    }

    gpu.rmsnorm_f32(&scratch.x, &weights.output_norm, &scratch.tmp, config.norm_eps)?;
    weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;

    // GPU-side sampling
    gpu.sample_top_p(
        &scratch.logits, &scratch.sample_buf,
        config.vocab_size, temperature, top_p, rng_state,
    )
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

        // RoPE — GPU-side, no CPU round-trip
        gpu.rope_f32(&q, &k, pos, n_heads, n_kv_heads, head_dim, config.rope_freq_base)?;

        // Store K, V in GPU cache — pure D2D copy, no CPU involvement
        let cache_byte_offset = pos * kv_dim * 4;
        gpu.hip.memcpy_dtod_at(
            &kv_cache.k_gpu[layer_idx].buf, cache_byte_offset,
            &k.buf, 0,
            kv_dim * 4,
        )?;
        gpu.hip.memcpy_dtod_at(
            &kv_cache.v_gpu[layer_idx].buf, cache_byte_offset,
            &v.buf, 0,
            kv_dim * 4,
        )?;
        // GPU-side attention
        gpu.attention_f32(
            &q,
            &kv_cache.k_gpu[layer_idx],
            &kv_cache.v_gpu[layer_idx],
            &attn_out,
            pos + 1,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_cache.max_seq,
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
    temperature: f32,
    top_p: f32,
    rng_state: u32,
) -> HipResult<(u32, u32)> {
    // Run forward pass to get logits on GPU
    let logits_on_gpu = forward_logits_gpu(gpu, weights, config, token, pos, kv_cache)?;

    // Sample on GPU — returns token_id + new rng state
    let result = gpu.sample_top_p(
        &logits_on_gpu, sample_buf,
        config.vocab_size, temperature, top_p, rng_state,
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

        gpu.rope_f32(&q, &k, pos, n_heads, n_kv_heads, head_dim, config.rope_freq_base)?;

        let cache_byte_offset = pos * kv_dim * 4;
        gpu.hip.memcpy_dtod_at(
            &kv_cache.k_gpu[layer_idx].buf, cache_byte_offset,
            &k.buf, 0, kv_dim * 4,
        )?;
        gpu.hip.memcpy_dtod_at(
            &kv_cache.v_gpu[layer_idx].buf, cache_byte_offset,
            &v.buf, 0, kv_dim * 4,
        )?;

        gpu.attention_f32(
            &q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
            &attn_out, pos + 1, n_heads, n_kv_heads, head_dim, kv_cache.max_seq,
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

/// Apply per-head RMSNorm (for Qwen3 QK normalization).
fn per_head_rmsnorm_cpu(data: &mut [f32], weight: &[f32], n_heads: usize, head_dim: usize, eps: f32) {
    for h in 0..n_heads {
        let base = h * head_dim;
        let head = &data[base..base + head_dim];
        let ss: f32 = head.iter().map(|x| x * x).sum::<f32>() / head_dim as f32;
        let rms = 1.0 / (ss + eps).sqrt();
        for d in 0..head_dim {
            data[base + d] = data[base + d] * rms * weight[d];
        }
    }
}

/// GPU-resident KV cache for autoregressive generation.
/// Pre-allocates [max_seq * n_kv_heads * head_dim] per layer on GPU.
pub struct KvCache {
    pub k_gpu: Vec<GpuTensor>,   // [n_layers] each [max_seq * kv_dim]
    pub v_gpu: Vec<GpuTensor>,   // [n_layers] each [max_seq * kv_dim]
    pub kv_dim: usize,
    pub max_seq: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
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
        Ok(Self { k_gpu, v_gpu, kv_dim, max_seq: max_seq_len, n_kv_heads, head_dim })
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
pub fn sample_top_p(logits: &[f32], temperature: f32, top_p: f32) -> u32 {
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

/// Simple deterministic-seeded RNG (xorshift32). Not crypto-quality, fine for sampling.
fn simple_rand() -> f32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static STATE: AtomicU32 = AtomicU32::new(0);

    // Seed from time on first call
    let mut s = STATE.load(Ordering::Relaxed);
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
    STATE.store(s, Ordering::Relaxed);
    (s as f32) / (u32::MAX as f32)
}
