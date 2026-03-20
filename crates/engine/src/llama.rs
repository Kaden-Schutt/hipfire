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

fn f16_to_f32(bits: u16) -> f32 {
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

/// GPU-resident LLaMA model weights.
pub struct LlamaWeights {
    pub token_embd: GpuTensor,
    pub output_norm: GpuTensor,
    pub output: GpuTensor,
    pub layers: Vec<LayerWeights>,
}

pub struct LayerWeights {
    pub attn_norm: GpuTensor,
    pub wq: GpuTensor,
    pub wk: GpuTensor,
    pub wv: GpuTensor,
    pub wo: GpuTensor,
    pub q_norm: Option<GpuTensor>, // Qwen3: per-head Q normalization
    pub k_norm: Option<GpuTensor>, // Qwen3: per-head K normalization
    pub ffn_norm: GpuTensor,
    pub w_gate: GpuTensor,
    pub w_up: GpuTensor,
    pub w_down: GpuTensor,
}

/// Load LLaMA weights from GGUF onto GPU, dequantizing to F32.
pub fn load_weights(
    gguf: &GgufFile,
    config: &LlamaConfig,
    gpu: &Gpu,
) -> HipResult<LlamaWeights> {
    let upload = |name: &str, expected_shape: &[usize]| -> HipResult<GpuTensor> {
        let info = gguf
            .find_tensor(name)
            .unwrap_or_else(|| panic!("tensor not found: {name}"));
        let data = load_tensor_f32(gguf, info);
        gpu.upload_f32(&data, expected_shape)
    };

    eprintln!("  loading token_embd...");
    let token_embd = upload("token_embd.weight", &[config.vocab_size, config.dim])?;
    eprintln!("  loading output_norm...");
    let output_norm = upload("output_norm.weight", &[config.dim])?;

    // output.weight may or may not exist (tied embeddings)
    eprintln!("  loading output...");
    let output = if gguf.find_tensor("output.weight").is_some() {
        upload("output.weight", &[config.vocab_size, config.dim])?
    } else {
        // Tied embeddings — reuse token_embd data
        let info = gguf.find_tensor("token_embd.weight").unwrap();
        let data = load_tensor_f32(gguf, info);
        gpu.upload_f32(&data, &[config.vocab_size, config.dim])?
    };

    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        eprintln!("  loading layer {i}/{} ...", config.n_layers);
        let p = format!("blk.{i}");
        let kv_dim = config.n_kv_heads * config.head_dim;

        // Qwen3: Q/K output dim may differ from dim if head_dim != dim/n_heads
        let q_out_dim = config.n_heads * config.head_dim;
        let k_out_dim = config.n_kv_heads * config.head_dim;

        let layer = LayerWeights {
            attn_norm: upload(&format!("{p}.attn_norm.weight"), &[config.dim])?,
            wq: upload(
                &format!("{p}.attn_q.weight"),
                &[q_out_dim, config.dim],
            )?,
            wk: upload(&format!("{p}.attn_k.weight"), &[kv_dim, config.dim])?,
            wv: upload(&format!("{p}.attn_v.weight"), &[kv_dim, config.dim])?,
            wo: upload(
                &format!("{p}.attn_output.weight"),
                &[config.dim, q_out_dim],
            )?,
            q_norm: if config.has_qk_norm {
                Some(upload(&format!("{p}.attn_q_norm.weight"), &[config.head_dim])?)
            } else {
                None
            },
            k_norm: if config.has_qk_norm {
                Some(upload(&format!("{p}.attn_k_norm.weight"), &[config.head_dim])?)
            } else {
                None
            },
            ffn_norm: upload(&format!("{p}.ffn_norm.weight"), &[config.dim])?,
            w_gate: upload(
                &format!("{p}.ffn_gate.weight"),
                &[config.hidden_dim, config.dim],
            )?,
            w_up: upload(
                &format!("{p}.ffn_up.weight"),
                &[config.hidden_dim, config.dim],
            )?,
            w_down: upload(
                &format!("{p}.ffn_down.weight"),
                &[config.dim, config.hidden_dim],
            )?,
        };
        layers.push(layer);
    }

    Ok(LlamaWeights {
        token_embd,
        output_norm,
        output,
        layers,
    })
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

    // Embedding lookup (CPU-side, then upload)
    let embd_data = gpu.download_f32(&weights.token_embd)?;
    let start = (token as usize) * dim;
    let x_data = embd_data[start..start + dim].to_vec();
    let mut x = gpu.upload_f32(&x_data, &[dim])?;

    let tmp = gpu.zeros(&[dim], DType::F32)?;

    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];

        // RMSNorm before attention
        gpu.rmsnorm_f32(&x, &layer.attn_norm, &tmp, config.norm_eps)?;

        // Q, K, V projections
        let q_dim = n_heads * head_dim;
        let q = gpu.zeros(&[q_dim], DType::F32)?;
        let k = gpu.zeros(&[kv_dim], DType::F32)?;
        let v = gpu.zeros(&[kv_dim], DType::F32)?;

        gpu.gemv_f32(&layer.wq, &tmp, &q)?;
        gpu.gemv_f32(&layer.wk, &tmp, &k)?;
        gpu.gemv_f32(&layer.wv, &tmp, &v)?;

        // QK normalization (Qwen3: RMSNorm per-head before RoPE)
        let mut q_data = gpu.download_f32(&q)?;
        let mut k_data = gpu.download_f32(&k)?;
        if config.has_qk_norm {
            if let Some(ref qn) = layer.q_norm {
                let qn_w = gpu.download_f32(qn)?;
                per_head_rmsnorm_cpu(&mut q_data, &qn_w, n_heads, head_dim, config.norm_eps);
            }
            if let Some(ref kn) = layer.k_norm {
                let kn_w = gpu.download_f32(kn)?;
                per_head_rmsnorm_cpu(&mut k_data, &kn_w, n_kv_heads, head_dim, config.norm_eps);
            }
        }

        // RoPE (CPU for now — simple and correct)
        apply_rope_cpu(&mut q_data, n_heads, head_dim, pos, config.rope_freq_base);
        apply_rope_cpu(&mut k_data, n_kv_heads, head_dim, pos, config.rope_freq_base);

        // Upload RoPE'd Q, re-upload K
        let q_rope = gpu.upload_f32(&q_data, &[q_dim])?;
        gpu.free_tensor(q)?;

        // Store K, V in cache
        kv_cache.store_kv(gpu, layer_idx, pos, &k_data, &gpu.download_f32(&v)?)?;
        gpu.free_tensor(k)?;
        gpu.free_tensor(v)?;

        // Attention: score = softmax(Q * K^T / sqrt(head_dim)) * V
        // CPU-side for correctness (GPU attention kernel is complex)
        let attn_out = attention_cpu(
            &q_data,
            kv_cache,
            layer_idx,
            pos,
            config,
        );

        let attn_gpu = gpu.upload_f32(&attn_out, &[q_dim])?;
        gpu.free_tensor(q_rope)?;

        // Output projection: o = Wo * attn_out
        let o = gpu.zeros(&[dim], DType::F32)?;
        gpu.gemv_f32(&layer.wo, &attn_gpu, &o)?;
        gpu.free_tensor(attn_gpu)?;

        // Residual: x = x + o
        let x_new = gpu.zeros(&[dim], DType::F32)?;
        gpu.add_f32(&x, &o, &x_new)?;
        gpu.free_tensor(x)?;
        gpu.free_tensor(o)?;
        x = x_new;

        // FFN
        gpu.rmsnorm_f32(&x, &layer.ffn_norm, &tmp, config.norm_eps)?;

        let gate = gpu.zeros(&[config.hidden_dim], DType::F32)?;
        let up = gpu.zeros(&[config.hidden_dim], DType::F32)?;
        gpu.gemv_f32(&layer.w_gate, &tmp, &gate)?;
        gpu.gemv_f32(&layer.w_up, &tmp, &up)?;

        // SiLU(gate) * up
        let gate_act = gpu.zeros(&[config.hidden_dim], DType::F32)?;
        gpu.silu_f32(&gate, &gate_act)?;
        gpu.free_tensor(gate)?;

        let ffn_hidden = gpu.zeros(&[config.hidden_dim], DType::F32)?;
        gpu.mul_f32(&gate_act, &up, &ffn_hidden)?;
        gpu.free_tensor(gate_act)?;
        gpu.free_tensor(up)?;

        // Down projection
        let ffn_out = gpu.zeros(&[dim], DType::F32)?;
        gpu.gemv_f32(&layer.w_down, &ffn_hidden, &ffn_out)?;
        gpu.free_tensor(ffn_hidden)?;

        // Residual
        let x_new = gpu.zeros(&[dim], DType::F32)?;
        gpu.add_f32(&x, &ffn_out, &x_new)?;
        gpu.free_tensor(x)?;
        gpu.free_tensor(ffn_out)?;
        x = x_new;
    }

    // Final norm
    gpu.rmsnorm_f32(&x, &weights.output_norm, &tmp, config.norm_eps)?;

    // Logits: output = output_weight * x
    let logits = gpu.zeros(&[config.vocab_size], DType::F32)?;
    gpu.gemv_f32(&weights.output, &tmp, &logits)?;

    let logits_data = gpu.download_f32(&logits)?;
    gpu.free_tensor(x)?;
    gpu.free_tensor(tmp)?;
    gpu.free_tensor(logits)?;

    Ok(logits_data)
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

/// KV cache for autoregressive generation.
pub struct KvCache {
    // [layer][position] -> (k_data, v_data)
    k: Vec<Vec<Vec<f32>>>,
    v: Vec<Vec<Vec<f32>>>,
    kv_dim: usize,
}

impl KvCache {
    pub fn new(n_layers: usize, kv_dim: usize, max_seq_len: usize) -> Self {
        Self {
            k: vec![Vec::with_capacity(max_seq_len); n_layers],
            v: vec![Vec::with_capacity(max_seq_len); n_layers],
            kv_dim,
        }
    }

    fn store_kv(
        &mut self,
        _gpu: &Gpu,
        layer: usize,
        pos: usize,
        k_data: &[f32],
        v_data: &[f32],
    ) -> HipResult<()> {
        // Extend cache to include this position
        while self.k[layer].len() <= pos {
            self.k[layer].push(vec![0.0; self.kv_dim]);
            self.v[layer].push(vec![0.0; self.kv_dim]);
        }
        self.k[layer][pos] = k_data.to_vec();
        self.v[layer][pos] = v_data.to_vec();
        Ok(())
    }
}

fn attention_cpu(
    q: &[f32],
    kv_cache: &KvCache,
    layer: usize,
    pos: usize,
    config: &LlamaConfig,
) -> Vec<f32> {
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;
    let kv_group = n_heads / n_kv_heads;
    let seq_len = pos + 1;

    let q_dim = n_heads * head_dim;
    let mut out = vec![0.0f32; q_dim];

    for h in 0..n_heads {
        let kv_h = h / kv_group;
        let q_offset = h * head_dim;

        // Compute attention scores
        let mut scores = vec![0.0f32; seq_len];
        for t in 0..seq_len {
            let k = &kv_cache.k[layer][t];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[q_offset + d] * k[kv_h * head_dim + d];
            }
            scores[t] = dot / (head_dim as f32).sqrt();
        }

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            sum += *s;
        }
        for s in &mut scores {
            *s /= sum;
        }

        // Weighted sum of values
        for t in 0..seq_len {
            let v = &kv_cache.v[layer][t];
            for d in 0..head_dim {
                out[q_offset + d] += scores[t] * v[kv_h * head_dim + d];
            }
        }
    }

    out
}

/// Sample the next token from logits using argmax (greedy).
pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap()
}
