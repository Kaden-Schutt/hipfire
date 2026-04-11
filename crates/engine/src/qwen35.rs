//! Qwen3.5 model: hybrid DeltaNet (linear attention) + standard attention.
//! Feature-gated behind `deltanet`.

use crate::hfq::HfqFile;
use crate::llama::{self, f16_to_f32, EmbeddingFormat, WeightTensor, weight_gemv,
                    weight_gemv_prerotated, rotate_x_for_mq, fused_rmsnorm_rotate_for_mq,
                    weight_gemv_residual, weight_gemv_swiglu_residual};
use crate::speculative::HiddenStateRingBuffer;
use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

// ─── Config ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    LinearAttention,  // DeltaNet
    FullAttention,    // Standard MHA with gated output
}

#[derive(Debug)]
pub struct Qwen35Config {
    pub dim: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub norm_eps: f32,
    pub eos_token: u32,

    // Full attention params
    pub n_heads: usize,        // 8
    pub n_kv_heads: usize,     // 2
    pub head_dim: usize,       // 256
    pub rope_theta: f32,
    pub partial_rotary_factor: f32, // 0.25 — only 64/256 dims get RoPE

    // DeltaNet params
    pub linear_num_key_heads: usize,   // 16
    pub linear_num_value_heads: usize, // 16
    pub linear_key_head_dim: usize,    // 128
    pub linear_value_head_dim: usize,  // 128
    pub conv_kernel_dim: usize,        // 4

    // FFN
    pub hidden_dim: usize,     // 3584

    // Per-layer type dispatch
    pub layer_types: Vec<LayerType>,
}

pub fn config_from_hfq(hfq: &HfqFile) -> Option<Qwen35Config> {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).ok()?;
    let config = meta.get("config")?;
    let tc = config.get("text_config").unwrap_or(config);

    let dim = tc.get("hidden_size")?.as_u64()? as usize;
    let n_layers = tc.get("num_hidden_layers")?.as_u64()? as usize;
    let n_heads = tc.get("num_attention_heads")?.as_u64()? as usize;
    let n_kv_heads = tc.get("num_key_value_heads").and_then(|v| v.as_u64()).unwrap_or(n_heads as u64) as usize;
    let head_dim = tc.get("head_dim").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(dim / n_heads);
    let vocab_size = tc.get("vocab_size")?.as_u64()? as usize;
    let hidden_dim = tc.get("intermediate_size")?.as_u64()? as usize;
    let norm_eps = tc.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32;

    let rope_params = tc.get("rope_parameters");
    let rope_theta = rope_params.and_then(|r| r.get("rope_theta")).and_then(|v| v.as_f64()).unwrap_or(10_000_000.0) as f32;
    let partial_rotary_factor = tc.get("partial_rotary_factor")
        .or_else(|| rope_params.and_then(|r| r.get("partial_rotary_factor")))
        .and_then(|v| v.as_f64()).unwrap_or(0.25) as f32;

    let eos_token = tc.get("eos_token_id").and_then(|v| v.as_u64()).unwrap_or(248044) as u32;

    let linear_num_key_heads = tc.get("linear_num_key_heads").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
    let linear_num_value_heads = tc.get("linear_num_value_heads").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
    let linear_key_head_dim = tc.get("linear_key_head_dim").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
    let linear_value_head_dim = tc.get("linear_value_head_dim").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
    let conv_kernel_dim = tc.get("linear_conv_kernel_dim").and_then(|v| v.as_u64()).unwrap_or(4) as usize;

    let layer_types: Vec<LayerType> = tc.get("layer_types")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().map(|v| {
            match v.as_str().unwrap_or("full_attention") {
                "linear_attention" => LayerType::LinearAttention,
                _ => LayerType::FullAttention,
            }
        }).collect())
        .unwrap_or_else(|| vec![LayerType::FullAttention; n_layers]);

    Some(Qwen35Config {
        dim, n_layers, vocab_size, norm_eps, eos_token,
        n_heads, n_kv_heads, head_dim, rope_theta, partial_rotary_factor,
        linear_num_key_heads, linear_num_value_heads, linear_key_head_dim, linear_value_head_dim, conv_kernel_dim,
        hidden_dim, layer_types,
    })
}

// ─── Weight structs ─────────────────────────────────────────────────────

/// Weights for a DeltaNet (linear attention) layer.
pub struct DeltaNetLayerWeights {
    pub attn_norm: GpuTensor,       // input_layernorm [dim]
    pub wqkv: WeightTensor,         // in_proj_qkv [6144, dim] → Q+K+V concat
    pub wz: WeightTensor,           // in_proj_z [2048, dim] → gate Z
    pub w_alpha: WeightTensor,      // in_proj_a [n_heads, dim] → decay
    pub w_beta: WeightTensor,       // in_proj_b [n_heads, dim] → update
    pub a_log: GpuTensor,           // A_log [n_heads] — learnable log-decay
    pub dt_bias: GpuTensor,         // dt_bias [n_heads]
    pub conv_weight: GpuTensor,     // conv1d.weight [conv_channels, 1, 4] → F32
    pub norm_weight: GpuTensor,     // norm.weight [head_dim] — gated output norm
    pub wo: WeightTensor,           // out_proj [dim, d_inner]
    pub ffn_norm: GpuTensor,        // post_attention_layernorm [dim]
    pub w_gate: WeightTensor,       // mlp.gate_proj
    pub w_up: WeightTensor,         // mlp.up_proj
    pub w_down: WeightTensor,       // mlp.down_proj
}

/// Weights for a full attention (gated) layer — similar to Qwen3 but with q+gate split.
pub struct FullAttnLayerWeights {
    pub attn_norm: GpuTensor,
    pub wq: WeightTensor,           // q_proj [4096, dim] — 2x wide (query + gate)
    pub wk: WeightTensor,           // k_proj
    pub wv: WeightTensor,           // v_proj
    pub wo: WeightTensor,           // o_proj
    pub q_norm: GpuTensor,          // q_norm [head_dim]
    pub k_norm: GpuTensor,          // k_norm [head_dim]
    pub ffn_norm: GpuTensor,
    pub w_gate: WeightTensor,
    pub w_up: WeightTensor,
    pub w_down: WeightTensor,
}

pub enum LayerWeights {
    DeltaNet(DeltaNetLayerWeights),
    FullAttn(FullAttnLayerWeights),
}

pub struct Qwen35Weights {
    pub token_embd: GpuTensor,
    pub embd_format: EmbeddingFormat,
    pub output_norm: GpuTensor,
    pub output: WeightTensor,
    pub layers: Vec<LayerWeights>,
}

// ─── State ──────────────────────────────────────────────────────────────

/// Persistent state for DeltaNet layers across tokens.
/// State quantization mode for DeltaNet S matrix.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum StateQuant {
    FP32,
    Q8,
    Q4,
}

pub struct DeltaNetState {
    /// S matrix storage — FP32 or Q8 depending on quant mode
    pub s_matrices: Vec<GpuTensor>,
    /// Per-head scale factors (only used for Q8 mode)
    pub s_scales: Vec<GpuTensor>,
    /// Conv ring buffer: [n_deltanet_layers × conv_channels × (kernel_size-1)] FP32
    pub conv_states: Vec<GpuTensor>,
    /// Current quantization mode
    pub quant: StateQuant,
}

impl DeltaNetState {
    pub fn new(gpu: &mut Gpu, config: &Qwen35Config) -> HipResult<Self> {
        Self::new_with_quant(gpu, config, StateQuant::Q8)
    }

    pub fn new_with_quant(gpu: &mut Gpu, config: &Qwen35Config, quant: StateQuant) -> HipResult<Self> {
        let n_delta_layers = config.layer_types.iter().filter(|t| **t == LayerType::LinearAttention).count();
        let s_dim = config.linear_key_head_dim; // 128
        let n_heads = config.linear_num_value_heads; // 16
        let s_size = n_heads * s_dim * s_dim; // 16 * 128 * 128 = 262144

        let conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
                          + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_state_size = conv_channels * (config.conv_kernel_dim - 1);

        let mut s_matrices = Vec::with_capacity(n_delta_layers);
        let mut s_scales = Vec::with_capacity(n_delta_layers);
        let mut conv_states = Vec::with_capacity(n_delta_layers);
        for _ in 0..n_delta_layers {
            match quant {
                StateQuant::FP32 => {
                    s_matrices.push(gpu.zeros(&[s_size], DType::F32)?);
                    s_scales.push(gpu.zeros(&[n_heads], DType::F32)?);
                }
                StateQuant::Q8 => {
                    // int8 state: s_size bytes (1 byte each), per-row scales
                    let buf = gpu.hip.malloc(s_size)?;
                    gpu.hip.memset(&buf, 0, s_size)?;
                    s_matrices.push(GpuTensor { buf, shape: vec![s_size], dtype: DType::F32 });
                    s_scales.push(gpu.zeros(&[n_heads * s_dim], DType::F32)?);
                }
                StateQuant::Q4 => {
                    // 4-bit nibble-packed: s_size/2 bytes, per-row scales
                    let buf = gpu.hip.malloc(s_size / 2)?;
                    gpu.hip.memset(&buf, 0, s_size / 2)?;
                    s_matrices.push(GpuTensor { buf, shape: vec![s_size / 2], dtype: DType::F32 });
                    s_scales.push(gpu.zeros(&[n_heads * s_dim], DType::F32)?);
                }
            }
            conv_states.push(gpu.zeros(&[conv_state_size], DType::F32)?);
        }
        Ok(Self { s_matrices, s_scales, conv_states, quant })
    }

    /// Free all GPU tensors. Call before drop to return VRAM.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in self.s_matrices { let _ = gpu.free_tensor(t); }
        for t in self.s_scales { let _ = gpu.free_tensor(t); }
        for t in self.conv_states { let _ = gpu.free_tensor(t); }
    }
}

// ─── Weight loading ─────────────────────────────────────────────────────

/// Load norm weight for Qwen3.5: stored as offset from 1.0 (output = x * (1 + weight))
fn load_norm_weight(hfq: &HfqFile, gpu: &mut Gpu, name: &str, shape: &[usize]) -> HipResult<GpuTensor> {
    let full_name = format!("model.language_model.{name}");
    let (info, data) = hfq.tensor_data(&full_name)
        .or_else(|| hfq.tensor_data(name))
        .unwrap_or_else(|| panic!("tensor not found: {name} or {full_name}"));

    let mut f32_data: Vec<f32> = match info.quant_type {
        1 => data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        2 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        _ => panic!("expected F16/F32 for {name}, got qt={}", info.quant_type),
    };
    // Qwen3.5 RMSNorm: output = x * rsqrt(var+eps) * (1 + weight)
    for v in &mut f32_data { *v += 1.0; }
    gpu.upload_f32(&f32_data, shape)
}

/// Load weight tensor from raw bytes + quant_type (no name lookup needed).
fn load_weight_tensor_raw(gpu: &Gpu, quant_type: u8, data: &[u8], m: usize, k: usize) -> HipResult<WeightTensor> {
    match quant_type {
        6 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ4G256, m, k, row_stride: 0 })
        }
        7 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ4G128, m, k, row_stride: 0 })
        }
        8 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ6G256, m, k, row_stride: 0 })
        }
        11 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ3G256, m, k, row_stride: 0 })
        }
        12 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ3G128, m, k, row_stride: 0 })
        }
        13 => { // MQ4-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::MQ4G256, m, k, row_stride: 0 })
        }
        14 => { // MQ8-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::MQ8G256, m, k, row_stride: 0 })
        }
        3 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::Q8_0, m, k, row_stride: 0 })
        }
        1 => {
            let f32_data: Vec<f32> = data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
            let buf = gpu.upload_raw(bytes, &[m, k])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::F32, m, k, row_stride: 0 })
        }
        _ => panic!("unsupported quant_type {} for lm_head", quant_type),
    }
}

fn load_weight_tensor(hfq: &HfqFile, gpu: &Gpu, name: &str, m: usize, k: usize) -> HipResult<WeightTensor> {
    let full_name = format!("model.language_model.{name}");
    let (info, data) = hfq.tensor_data(&full_name)
        .or_else(|| hfq.tensor_data(name))
        .unwrap_or_else(|| panic!("tensor not found: {name} or {full_name}"));

    match info.quant_type {
        6 => { // HFQ4-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ4G256, m, k, row_stride: 0 })
        }
        7 => { // HFQ4-G128
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ4G128, m, k, row_stride: 0 })
        }
        8 => { // HFQ6-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ6G256, m, k, row_stride: 0 })
        }
        11 => { // HFQ3-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ3G256, m, k, row_stride: 0 })
        }
        12 => { // HFQ3-G128
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ3G128, m, k, row_stride: 0 })
        }
        13 => { // MQ4-G256 — MagnumQuant
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::MQ4G256, m, k, row_stride: 0 })
        }
        14 => { // MQ8-G256 — MagnumQuant dp4a
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::MQ8G256, m, k, row_stride: 0 })
        }
        3 => { // Q8_0
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::Q8_0, m, k, row_stride: 0 })
        }
        1 => { // F16 → F32
            let f32_data: Vec<f32> = data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
            let buf = gpu.upload_raw(bytes, &[m, k])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::F32, m, k, row_stride: 0 })
        }
        _ => panic!("unsupported quant_type {} for {name}", info.quant_type),
    }
}

/// Load a tensor as F32 on GPU, handling any quant type by dequanting on CPU.
fn load_any_as_f32(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    let full_name = format!("model.language_model.{name}");
    let (info, data) = hfq.tensor_data(&full_name)
        .or_else(|| hfq.tensor_data(name))
        .unwrap_or_else(|| panic!("tensor not found: {name} or {full_name}"));

    let f32_data: Vec<f32> = match info.quant_type {
        1 => data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        2 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        3 => crate::llama::dequantize_q8_0(data, n),
        14 => {
            // MQ8-G256: [f16 scale][int8 × 256] = 258 bytes per 256 weights
            let group_size: usize = 256;
            let bytes_per_group: usize = 258;
            let n_groups = data.len() / bytes_per_group;
            let signs1 = crate::llama::KvCache::gen_fwht_signs(42, 256);
            let signs2 = crate::llama::KvCache::gen_fwht_signs(1042, 256);
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale_bits = data[off] as u16 | ((data[off + 1] as u16) << 8);
                let scale = crate::llama::f16_to_f32(scale_bits);
                let start = out.len();
                for i in 0..256 {
                    let q = data[off + 2 + i] as i8;
                    out.push(scale * q as f32);
                }
                // Inverse FWHT to recover original values
                let group = &mut out[start..start + 256];
                for i in 0..256 { group[i] *= signs2[i]; }
                let mut stride = 1;
                while stride < 256 {
                    let mut j = 0;
                    while j < 256 {
                        for k in 0..stride {
                            let a = group[j + k];
                            let b = group[j + k + stride];
                            group[j + k] = a + b;
                            group[j + k + stride] = a - b;
                        }
                        j += stride * 2;
                    }
                    stride <<= 1;
                }
                let inv_s = 0.0625;
                for i in 0..256 { group[i] *= inv_s * signs1[i]; }
            }
            out
        }
        6 | 7 | 13 => {
            // HFQ4-G256 or G128 or MQ4-G256 — CPU dequant
            // MQ4 stores FWHT-rotated weights. For small tensors loaded here,
            // we dequant then inverse-FWHT to recover the original values.
            let group_size: usize = if info.quant_type == 6 || info.quant_type == 13 { 256 } else { 128 };
            let bytes_per_group = 8 + group_size / 2;
            let n_groups = data.len() / bytes_per_group;
            let is_mq4 = info.quant_type == 13;
            let mut out = Vec::with_capacity(n_groups * group_size);
            // Generate inverse FWHT signs (same seeds as quantizer)
            let (signs1, signs2) = if is_mq4 {
                (Some(crate::llama::KvCache::gen_fwht_signs(42, 256)),
                 Some(crate::llama::KvCache::gen_fwht_signs(1042, 256)))
            } else { (None, None) };
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale = f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
                let zero = f32::from_le_bytes([data[off+4], data[off+5], data[off+6], data[off+7]]);
                let start = out.len();
                for i in 0..group_size {
                    let byte_idx = i / 2;
                    let byte_val = data[off + 8 + byte_idx];
                    let nibble = if i % 2 == 0 { byte_val & 0xF } else { byte_val >> 4 };
                    out.push(scale * nibble as f32 + zero);
                }
                // Inverse FWHT for MQ4: recover original weight values
                if is_mq4 && group_size == 256 {
                    let s1 = signs1.as_ref().unwrap();
                    let s2 = signs2.as_ref().unwrap();
                    let group = &mut out[start..start + 256];
                    // Inverse FWHT: signs2 → butterfly → scale → signs1
                    for i in 0..256 { group[i] *= s2[i]; }
                    let mut stride = 1;
                    while stride < 256 {
                        let mut j = 0;
                        while j < 256 {
                            for k in 0..stride {
                                let a = group[j + k];
                                let b = group[j + k + stride];
                                group[j + k] = a + b;
                                group[j + k + stride] = a - b;
                            }
                            j += stride * 2;
                        }
                        stride <<= 1;
                    }
                    let scale_inv = 0.0625; // 1/sqrt(256)
                    for i in 0..256 { group[i] *= scale_inv * s1[i]; }
                }
            }
            out
        }
        8 => {
            // HFQ6-G256 — CPU dequant: [f32 scale][f32 zero][192B packed 6-bit] = 200 bytes per 256 weights
            let group_size: usize = 256;
            let bytes_per_group: usize = 200; // 8 + 192
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale = f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
                let zero = f32::from_le_bytes([data[off+4], data[off+5], data[off+6], data[off+7]]);
                // 4 values per 3 bytes: v0[5:0]|v1[1:0], v1[5:2]|v2[3:0], v2[5:4]|v3[5:0]
                for i in (0..group_size).step_by(4) {
                    let byte_off = 8 + (i / 4) * 3;
                    let b0 = data[off + byte_off] as u32;
                    let b1 = data[off + byte_off + 1] as u32;
                    let b2 = data[off + byte_off + 2] as u32;
                    let q0 = (b0 & 0x3F) as f32;
                    let q1 = (((b0 >> 6) | (b1 << 2)) & 0x3F) as f32;
                    let q2 = (((b1 >> 4) | (b2 << 4)) & 0x3F) as f32;
                    let q3 = ((b2 >> 2) & 0x3F) as f32;
                    out.push(scale * q0 + zero);
                    out.push(scale * q1 + zero);
                    out.push(scale * q2 + zero);
                    out.push(scale * q3 + zero);
                }
            }
            out
        }
        11 => {
            // HFQ3-G256: [f32 scale][f32 zero][96B packed 3-bit] = 104 bytes per 256 weights
            let group_size: usize = 256;
            let bytes_per_group: usize = 104;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale = f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
                let zero = f32::from_le_bytes([data[off+4], data[off+5], data[off+6], data[off+7]]);
                // 8 values per 3 bytes (matching kernel unpack)
                for chunk in 0..32 {
                    let bo = off + 8 + chunk * 3;
                    let b0 = data[bo] as u32;
                    let b1 = data[bo + 1] as u32;
                    let b2 = data[bo + 2] as u32;
                    let q0 = (b0 & 7) as f32;
                    let q1 = ((b0 >> 3) & 7) as f32;
                    let q2 = (((b0 >> 6) | (b1 << 2)) & 7) as f32;
                    let q3 = ((b1 >> 1) & 7) as f32;
                    let q4 = ((b1 >> 4) & 7) as f32;
                    let q5 = (((b1 >> 7) | (b2 << 1)) & 7) as f32;
                    let q6 = ((b2 >> 2) & 7) as f32;
                    let q7 = ((b2 >> 5) & 7) as f32;
                    out.push(scale * q0 + zero);
                    out.push(scale * q1 + zero);
                    out.push(scale * q2 + zero);
                    out.push(scale * q3 + zero);
                    out.push(scale * q4 + zero);
                    out.push(scale * q5 + zero);
                    out.push(scale * q6 + zero);
                    out.push(scale * q7 + zero);
                }
            }
            out
        }
        12 => {
            // HFQ3-G128: [f32 scale][f32 zero][48B packed 3-bit] = 56 bytes per 128 weights
            let group_size: usize = 128;
            let bytes_per_group: usize = 56;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale = f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
                let zero = f32::from_le_bytes([data[off+4], data[off+5], data[off+6], data[off+7]]);
                for chunk in 0..16 {
                    let bo = off + 8 + chunk * 3;
                    let b0 = data[bo] as u32;
                    let b1 = data[bo + 1] as u32;
                    let b2 = data[bo + 2] as u32;
                    let q0 = (b0 & 7) as f32;
                    let q1 = ((b0 >> 3) & 7) as f32;
                    let q2 = (((b0 >> 6) | (b1 << 2)) & 7) as f32;
                    let q3 = ((b1 >> 1) & 7) as f32;
                    let q4 = ((b1 >> 4) & 7) as f32;
                    let q5 = (((b1 >> 7) | (b2 << 1)) & 7) as f32;
                    let q6 = ((b2 >> 2) & 7) as f32;
                    let q7 = ((b2 >> 5) & 7) as f32;
                    out.push(scale * q0 + zero);
                    out.push(scale * q1 + zero);
                    out.push(scale * q2 + zero);
                    out.push(scale * q3 + zero);
                    out.push(scale * q4 + zero);
                    out.push(scale * q5 + zero);
                    out.push(scale * q6 + zero);
                    out.push(scale * q7 + zero);
                }
            }
            out
        }
        _ => panic!("unsupported quant_type {} for {name}", info.quant_type),
    };
    gpu.upload_f32(&f32_data[..n], &[n])
}

/// Alias for load_any_as_f32.
fn load_raw_f32(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    load_any_as_f32(hfq, gpu, name, n)
}

pub fn load_weights(hfq: &HfqFile, config: &Qwen35Config, gpu: &mut Gpu) -> HipResult<Qwen35Weights> {
    eprintln!("  loading token_embd...");
    let embd_info = hfq.tensor_data("model.language_model.embed_tokens.weight")
        .expect("embed_tokens not found");
    let (token_embd, embd_fmt) = if embd_info.0.quant_type == 6 {
        eprintln!("    (HFQ4-G256 raw, {} MB)", embd_info.1.len() / 1_000_000);
        (gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?, EmbeddingFormat::HFQ4G256)
    } else if embd_info.0.quant_type == 7 {
        eprintln!("    (HFQ4-G128 raw, {} MB)", embd_info.1.len() / 1_000_000);
        (gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?, EmbeddingFormat::HFQ4G128)
    } else if embd_info.0.quant_type == 3 {
        // Q8_0: [f16 scale][32 × int8] per block — upload raw, use Q8 embedding lookup
        eprintln!("    (Q8_0 raw, {} MB)", embd_info.1.len() / 1_000_000);
        (gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?, EmbeddingFormat::Q8_0)
    } else {
        let f32_data: Vec<f32> = embd_info.1.chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        (gpu.upload_f32(&f32_data, &[config.vocab_size, config.dim])?, EmbeddingFormat::F32)
    };

    eprintln!("  loading output_norm...");
    let output_norm = load_norm_weight(hfq, gpu, "norm.weight", &[config.dim])?;

    // Try separate lm_head first (untied embeddings, e.g. 9B), fall back to tied embed_tokens
    let lm_head_info = hfq.tensor_data("lm_head.weight")
        .or_else(|| hfq.tensor_data("model.language_model.lm_head.weight"));
    let output = if let Some((lm_info, lm_data)) = lm_head_info {
        eprintln!("  loading output (separate lm_head, qt={})...", lm_info.quant_type);
        load_weight_tensor_raw(gpu, lm_info.quant_type, lm_data, config.vocab_size, config.dim)?
    } else {
        eprintln!("  loading output (tied embeddings)...");
        let embd_data = hfq.tensor_data("model.language_model.embed_tokens.weight").unwrap().1;
        if embd_info.0.quant_type == 6 || embd_info.0.quant_type == 7 || embd_info.0.quant_type == 8 {
            let buf = gpu.upload_raw(embd_data, &[embd_data.len()])?;
            let dtype = match embd_info.0.quant_type {
                6 => DType::HFQ4G256, 7 => DType::HFQ4G128, 8 => DType::HFQ6G256, _ => unreachable!()
            };
            WeightTensor { buf, gpu_dtype: dtype, m: config.vocab_size, k: config.dim, row_stride: 0 }
        } else if embd_info.0.quant_type == 3 {
            let buf = gpu.upload_raw(embd_data, &[embd_data.len()])?;
            WeightTensor { buf, gpu_dtype: DType::Q8_0, m: config.vocab_size, k: config.dim, row_stride: 0 }
        } else {
            let f32_data: Vec<f32> = embd_data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
            let buf = gpu.upload_raw(bytes, &[config.vocab_size, config.dim])?;
            WeightTensor { buf, gpu_dtype: DType::F32, m: config.vocab_size, k: config.dim, row_stride: 0 }
        }
    };

    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        eprintln!("  loading layer {i}/{} ({:?})...", config.n_layers, config.layer_types[i]);
        let p = format!("layers.{i}");

        match config.layer_types[i] {
            LayerType::LinearAttention => {
                let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
                            + config.linear_num_value_heads * config.linear_value_head_dim;
                let d_inner = config.linear_num_value_heads * config.linear_value_head_dim;

                layers.push(LayerWeights::DeltaNet(DeltaNetLayerWeights {
                    attn_norm: load_norm_weight(hfq, gpu, &format!("{p}.input_layernorm.weight"), &[config.dim])?,
                    wqkv: load_weight_tensor(hfq, gpu, &format!("{p}.linear_attn.in_proj_qkv.weight"), qkv_dim, config.dim)?,
                    wz: load_weight_tensor(hfq, gpu, &format!("{p}.linear_attn.in_proj_z.weight"), d_inner, config.dim)?,
                    w_alpha: load_weight_tensor(hfq, gpu, &format!("{p}.linear_attn.in_proj_a.weight"),
                        config.linear_num_value_heads, config.dim)?,
                    w_beta: load_weight_tensor(hfq, gpu, &format!("{p}.linear_attn.in_proj_b.weight"),
                        config.linear_num_value_heads, config.dim)?,
                    a_log: load_raw_f32(hfq, gpu, &format!("{p}.linear_attn.A_log"), config.linear_num_value_heads)?,
                    dt_bias: load_raw_f32(hfq, gpu, &format!("{p}.linear_attn.dt_bias"), config.linear_num_value_heads)?,
                    conv_weight: load_any_as_f32(hfq, gpu, &format!("{p}.linear_attn.conv1d.weight"),
                        qkv_dim * config.conv_kernel_dim)?,  // flatten [channels, 1, kernel] → [channels * kernel]
                    norm_weight: load_any_as_f32(hfq, gpu, &format!("{p}.linear_attn.norm.weight"), config.linear_value_head_dim)?,
                    wo: load_weight_tensor(hfq, gpu, &format!("{p}.linear_attn.out_proj.weight"), config.dim, d_inner)?,
                    ffn_norm: load_norm_weight(hfq, gpu, &format!("{p}.post_attention_layernorm.weight"), &[config.dim])?,
                    w_gate: load_weight_tensor(hfq, gpu, &format!("{p}.mlp.gate_proj.weight"), config.hidden_dim, config.dim)?,
                    w_up: load_weight_tensor(hfq, gpu, &format!("{p}.mlp.up_proj.weight"), config.hidden_dim, config.dim)?,
                    w_down: load_weight_tensor(hfq, gpu, &format!("{p}.mlp.down_proj.weight"), config.dim, config.hidden_dim)?,
                }));
            }
            LayerType::FullAttention => {
                let q_out_dim = config.n_heads * config.head_dim * 2; // 2x for query + gate
                let kv_dim = config.n_kv_heads * config.head_dim;

                layers.push(LayerWeights::FullAttn(FullAttnLayerWeights {
                    attn_norm: load_norm_weight(hfq, gpu, &format!("{p}.input_layernorm.weight"), &[config.dim])?,
                    wq: load_weight_tensor(hfq, gpu, &format!("{p}.self_attn.q_proj.weight"), q_out_dim, config.dim)?,
                    wk: load_weight_tensor(hfq, gpu, &format!("{p}.self_attn.k_proj.weight"), kv_dim, config.dim)?,
                    wv: load_weight_tensor(hfq, gpu, &format!("{p}.self_attn.v_proj.weight"), kv_dim, config.dim)?,
                    wo: load_weight_tensor(hfq, gpu, &format!("{p}.self_attn.o_proj.weight"), config.dim, config.n_heads * config.head_dim)?,
                    q_norm: load_norm_weight(hfq, gpu, &format!("{p}.self_attn.q_norm.weight"), &[config.head_dim])?,
                    k_norm: load_norm_weight(hfq, gpu, &format!("{p}.self_attn.k_norm.weight"), &[config.head_dim])?,
                    ffn_norm: load_norm_weight(hfq, gpu, &format!("{p}.post_attention_layernorm.weight"), &[config.dim])?,
                    w_gate: load_weight_tensor(hfq, gpu, &format!("{p}.mlp.gate_proj.weight"), config.hidden_dim, config.dim)?,
                    w_up: load_weight_tensor(hfq, gpu, &format!("{p}.mlp.up_proj.weight"), config.hidden_dim, config.dim)?,
                    w_down: load_weight_tensor(hfq, gpu, &format!("{p}.mlp.down_proj.weight"), config.dim, config.hidden_dim)?,
                }));
            }
        }
    }

    Ok(Qwen35Weights { token_embd, embd_format: embd_fmt, output_norm, output, layers })
}

// ─── Forward pass (decode, one token at a time) ─────────────────────────

/// Run one token through the Qwen3.5 model. Returns logits.
/// For DeltaNet layers, updates state in-place (S matrix + conv ring buffer).
/// For full attention layers, uses KV cache like standard transformer.
pub fn forward(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<Vec<f32>> {
    let dim = config.dim;

    // Embedding lookup
    let x = gpu.alloc_tensor(&[dim], DType::F32)?;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &x, token, dim)?,
        _ => panic!("unsupported embedding format"),
    }

    forward_from_x(gpu, weights, config, x, pos, kv_cache, dn_state)
}

/// Shared forward pass — returns logits as CPU Vec<f32>.
fn forward_from_x(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    x: GpuTensor,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<Vec<f32>> {
    let logits_gpu = forward_from_x_gpu(gpu, weights, config, x, pos, kv_cache, dn_state)?;
    let logits_data = gpu.download_f32(&logits_gpu)?;
    gpu.free_tensor(logits_gpu)?;
    Ok(logits_data)
}

/// Shared forward pass — returns logits as GPU tensor (no download).
/// Caller must free the returned tensor.
fn forward_from_x_gpu(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    x: GpuTensor,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<GpuTensor> {
    let dim = config.dim;

    let tmp = gpu.alloc_tensor(&[dim], DType::F32)?;
    let pos_buf = gpu.hip.malloc(4)?;
    let pos_i32 = pos as i32;
    gpu.hip.memcpy_htod(&pos_buf, &pos_i32.to_ne_bytes())?;

    let mut delta_layer_idx = 0usize;
    let debug_layers = std::env::var("DEBUG_LAYERS").is_ok();

    if debug_layers && pos == 0 {
        let hid = gpu.download_f32(&x)?;
        let norm: f32 = hid.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!("EMB: first4=[{:.6},{:.6},{:.6},{:.6}] norm={norm:.4}", hid[0], hid[1], hid[2], hid[3]);
    }

    for layer_idx in 0..config.n_layers {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                // ── DeltaNet layer ──
                gpu.rmsnorm_f32(&x, &layer.attn_norm, &tmp, config.norm_eps)?;

                // QKV projection
                let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
                             + config.linear_num_value_heads * config.linear_value_head_dim;
                let qkv = gpu.alloc_tensor(&[qkv_dim], DType::F32)?;
                weight_gemv(gpu, &layer.wqkv, &tmp, &qkv)?;

                // Z (gate) projection
                let d_inner = config.linear_num_value_heads * config.linear_value_head_dim;
                let z = gpu.alloc_tensor(&[d_inner], DType::F32)?;
                weight_gemv(gpu, &layer.wz, &tmp, &z)?;

                // Beta + alpha projections, then fused sigmoid/alpha_gate.
                let n_v_heads = config.linear_num_value_heads;
                let beta_out = gpu.alloc_tensor(&[n_v_heads], DType::F32)?;
                weight_gemv(gpu, &layer.w_beta, &tmp, &beta_out)?;
                let alpha_out = gpu.alloc_tensor(&[n_v_heads], DType::F32)?;
                weight_gemv(gpu, &layer.w_alpha, &tmp, &alpha_out)?;
                gpu.fused_sigmoid_alpha_gate_f32(
                    &beta_out, &alpha_out, &layer.dt_bias, &layer.a_log, n_v_heads,
                )?;

                // Fused conv1d + SiLU (one kernel instead of two)
                let conv_out = gpu.alloc_tensor(&[qkv_dim], DType::F32)?;
                gpu.conv1d_silu_f32(
                    &conv_out, &qkv, &layer.conv_weight,
                    &dn_state.conv_states[delta_layer_idx], qkv_dim,
                )?;

                // Split conv output into Q, K, V
                let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
                let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
                let q_part = gpu.alloc_tensor(&[k_dim], DType::F32)?;
                let k_part = gpu.alloc_tensor(&[k_dim], DType::F32)?;
                let v_part = gpu.alloc_tensor(&[v_dim], DType::F32)?;
                gpu.hip.memcpy_dtod_at(&q_part.buf, 0, &conv_out.buf, 0, k_dim * 4)?;
                gpu.hip.memcpy_dtod_at(&k_part.buf, 0, &conv_out.buf, k_dim * 4, k_dim * 4)?;
                gpu.hip.memcpy_dtod_at(&v_part.buf, 0, &conv_out.buf, k_dim * 2 * 4, v_dim * 4)?;

                // Fused L2-norm(Q) + L2-norm(K) + scale(Q) — 3 launches → 1.
                gpu.fused_qk_l2_norm_scale_f32(
                    &q_part,
                    &k_part,
                    config.linear_num_key_heads,
                    config.linear_key_head_dim,
                    1.0 / (config.linear_key_head_dim as f32).sqrt(),
                    config.norm_eps,
                )?;

                // Repeat Q/K heads if num_k_heads < num_v_heads (GQA-style)
                // Phase 3a-A fix: same fused kernel as forward_scratch_layers.
                let (q_gdn, k_gdn) = if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    let expanded_dim = n_v_heads * config.linear_key_head_dim;
                    let q_exp = gpu.alloc_tensor(&[expanded_dim], DType::F32)?;
                    let k_exp = gpu.alloc_tensor(&[expanded_dim], DType::F32)?;
                    let hd = config.linear_key_head_dim;
                    gpu.repeat_interleave_qk_f32(
                        &q_part, &k_part, &q_exp, &k_exp,
                        config.linear_num_key_heads, ratio, hd,
                    )?;
                    (q_exp, k_exp)
                } else {
                    // Same number of heads — no repeat needed, reuse buffers directly
                    // (we'll skip freeing these in the cleanup below)
                    let q_ref = gpu.alloc_tensor(&[k_dim], DType::F32)?;
                    let k_ref = gpu.alloc_tensor(&[k_dim], DType::F32)?;
                    gpu.hip.memcpy_dtod_at(&q_ref.buf, 0, &q_part.buf, 0, k_dim * 4)?;
                    gpu.hip.memcpy_dtod_at(&k_ref.buf, 0, &k_part.buf, 0, k_dim * 4)?;
                    (q_ref, k_ref)
                };

                // Gated Delta Net recurrence
                let attn_out = gpu.alloc_tensor(&[v_dim], DType::F32)?;
                match dn_state.quant {
                    StateQuant::FP32 => gpu.gated_delta_net_f32(
                        &q_gdn, &k_gdn, &v_part, &alpha_out, &beta_out,
                        &dn_state.s_matrices[delta_layer_idx], &attn_out,
                        1, n_v_heads, config.linear_value_head_dim,
                    )?,
                    StateQuant::Q8 => gpu.gated_delta_net_q8(
                        &q_gdn, &k_gdn, &v_part, &alpha_out, &beta_out,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx], &attn_out,
                        1, n_v_heads, config.linear_value_head_dim,
                    )?,
                    StateQuant::Q4 => gpu.gated_delta_net_q4(
                        &q_gdn, &k_gdn, &v_part, &alpha_out, &beta_out,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx], &attn_out,
                        1, n_v_heads, config.linear_value_head_dim,
                    )?,
                }

                // Q-only scaling. llama.cpp also scales output by 1/sqrt(S_v)
                // in the kernel, but that makes L00 too small (0.175 vs ref 0.501).
                // Q-only gives L00 = 0.489 vs ref 0.501. Keeping Q-only for now.

                // Gated norm: rmsnorm(attn_out) * silu(z)
                let normed_out = gpu.alloc_tensor(&[v_dim], DType::F32)?;
                gpu.gated_norm_f32(&attn_out, &z, &layer.norm_weight, &normed_out,
                    n_v_heads, config.linear_value_head_dim, config.norm_eps)?;

                // Output projection
                let o = gpu.alloc_tensor(&[dim], DType::F32)?;
                weight_gemv(gpu, &layer.wo, &normed_out, &o)?;

                // Residual
                gpu.add_inplace_f32(&x, &o)?;

                // FFN
                gpu.rmsnorm_f32(&x, &layer.ffn_norm, &tmp, config.norm_eps)?;
                let gate = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
                let up = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
                weight_gemv(gpu, &layer.w_gate, &tmp, &gate)?;
                weight_gemv(gpu, &layer.w_up, &tmp, &up)?;
                let ffn_hidden = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
                gpu.silu_mul_f32(&gate, &up, &ffn_hidden)?;
                let ffn_out = gpu.alloc_tensor(&[dim], DType::F32)?;
                weight_gemv(gpu, &layer.w_down, &ffn_hidden, &ffn_out)?;
                gpu.add_inplace_f32(&x, &ffn_out)?;

                // Free temporaries
                for t in [qkv, z, beta_out, alpha_out, conv_out, q_part, k_part, v_part, q_gdn, k_gdn, attn_out, normed_out, o, gate, up, ffn_hidden, ffn_out] {
                    gpu.free_tensor(t)?;
                }
                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttn(layer), LayerType::FullAttention) => {
                // ── Full attention layer (gated) ──
                gpu.rmsnorm_f32(&x, &layer.attn_norm, &tmp, config.norm_eps)?;

                // Q projection (2x wide → split into query + gate)
                let q_full_dim = config.n_heads * config.head_dim * 2;
                let q_full = gpu.alloc_tensor(&[q_full_dim], DType::F32)?;
                weight_gemv(gpu, &layer.wq, &tmp, &q_full)?;

                // Split Q into query and gate — interleaved per head:
                // [Q_h0(256), Gate_h0(256), Q_h1(256), Gate_h1(256), ...]
                let q_dim = config.n_heads * config.head_dim;
                let q = gpu.alloc_tensor(&[q_dim], DType::F32)?;
                let gate_vec = gpu.alloc_tensor(&[q_dim], DType::F32)?;
                // Deinterleave Q and gate with a single kernel dispatch
                // (replaces per-head memcpy loop: n_heads × 2 ioctls → 1 dispatch)
                gpu.deinterleave_f32(&q_full, &q, &gate_vec, config.n_heads, config.head_dim)?;

                // Q norm
                gpu.rmsnorm_batched(&q, &layer.q_norm, &q, config.n_heads, config.head_dim, config.norm_eps)?;

                // K, V projections
                let kv_dim = config.n_kv_heads * config.head_dim;
                let k = gpu.alloc_tensor(&[kv_dim], DType::F32)?;
                let v = gpu.alloc_tensor(&[kv_dim], DType::F32)?;
                weight_gemv(gpu, &layer.wk, &tmp, &k)?;
                weight_gemv(gpu, &layer.wv, &tmp, &v)?;

                // K norm
                gpu.rmsnorm_batched(&k, &layer.k_norm, &k, config.n_kv_heads, config.head_dim, config.norm_eps)?;

                // Partial interleaved RoPE: rotate first n_rot dims, pairs (d0,d1),(d2,d3),...
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize; // 64
                gpu.rope_partial_interleaved_f32(&q, &k, pos as i32,
                    config.n_heads, config.n_kv_heads, config.head_dim, n_rot, config.rope_theta)?;

                // KV cache write + attention (Q8 if available, FP32 fallback)
                let attn_out = gpu.alloc_tensor(&[q_dim], DType::F32)?;
                if kv_cache.quant_q8 {
                    gpu.kv_cache_write_q8_0(&kv_cache.k_gpu[layer_idx], &k, &pos_buf, config.n_kv_heads, config.head_dim)?;
                    gpu.kv_cache_write_q8_0(&kv_cache.v_gpu[layer_idx], &v, &pos_buf, config.n_kv_heads, config.head_dim)?;
                    gpu.attention_q8_0_kv(
                        &q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &attn_out, &pos_buf, pos + 1, config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
                    )?;
                } else {
                    gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &k, &pos_buf, kv_dim)?;
                    gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &v, &pos_buf, kv_dim)?;
                    gpu.attention_f32(
                        &q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &attn_out, &pos_buf, pos + 1, config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
                    )?;
                }

                // Sigmoid gate
                gpu.sigmoid_f32(&gate_vec)?;
                // attn_out *= gate
                gpu.mul_f32(&attn_out, &gate_vec, &attn_out)?;

                // Output projection
                let o = gpu.alloc_tensor(&[dim], DType::F32)?;
                weight_gemv(gpu, &layer.wo, &attn_out, &o)?;

                // Residual
                gpu.add_inplace_f32(&x, &o)?;

                // FFN
                gpu.rmsnorm_f32(&x, &layer.ffn_norm, &tmp, config.norm_eps)?;
                let gate_ffn = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
                let up = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
                weight_gemv(gpu, &layer.w_gate, &tmp, &gate_ffn)?;
                weight_gemv(gpu, &layer.w_up, &tmp, &up)?;
                let ffn_hidden = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
                gpu.silu_mul_f32(&gate_ffn, &up, &ffn_hidden)?;
                let ffn_out = gpu.alloc_tensor(&[dim], DType::F32)?;
                weight_gemv(gpu, &layer.w_down, &ffn_hidden, &ffn_out)?;
                gpu.add_inplace_f32(&x, &ffn_out)?;

                for t in [q_full, q, gate_vec, k, v, attn_out, o, gate_ffn, up, ffn_hidden, ffn_out] {
                    gpu.free_tensor(t)?;
                }
            }

            _ => panic!("layer type mismatch at layer {layer_idx}"),
        }

        if debug_layers && pos == 0 {
            let hid = gpu.download_f32(&x)?;
            let norm: f32 = hid.iter().map(|v| v * v).sum::<f32>().sqrt();
            let lt = match config.layer_types[layer_idx] { LayerType::LinearAttention => "D", LayerType::FullAttention => "F" };
            eprintln!("L{layer_idx:02}({lt}): first4=[{:.4},{:.4},{:.4},{:.4}] norm={norm:.2}", hid[0], hid[1], hid[2], hid[3]);
        }
    }

    // Final norm + output projection
    gpu.rmsnorm_f32(&x, &weights.output_norm, &tmp, config.norm_eps)?;
    let logits = gpu.alloc_tensor(&[config.vocab_size], DType::F32)?;
    weight_gemv(gpu, &weights.output, &tmp, &logits)?;

    gpu.free_tensor(x)?;
    gpu.free_tensor(tmp)?;
    gpu.hip.free(pos_buf)?;

    Ok(logits)
}

/// Pre-allocated scratch buffers for zero-alloc qwen35 forward + GPU sampling.
pub struct Qwen35Scratch {
    // Persistent state
    pub x: GpuTensor,           // [dim]
    pub tmp: GpuTensor,         // [dim]
    pub pos_buf: hip_bridge::DeviceBuffer, // 4 bytes

    // DeltaNet temporaries (reused across layers)
    pub dn_qkv: GpuTensor,     // [qkv_dim]
    pub dn_z: GpuTensor,        // [v_dim]
    pub dn_alpha: GpuTensor,    // [n_v_heads]
    pub dn_beta: GpuTensor,     // [n_v_heads]
    pub dn_conv_out: GpuTensor, // [qkv_dim]
    pub dn_q: GpuTensor,        // [v_dim] (after repeat-interleave)
    pub dn_k: GpuTensor,        // [v_dim]
    pub dn_v: GpuTensor,        // [v_dim]
    pub dn_q_raw: GpuTensor,    // [k_dim] (before repeat)
    pub dn_k_raw: GpuTensor,    // [k_dim]
    pub dn_attn_out: GpuTensor, // [v_dim]
    pub dn_normed: GpuTensor,   // [v_dim]

    // FullAttn temporaries (reused across layers)
    pub fa_q_full: GpuTensor,   // [n_heads * head_dim * 2]
    pub fa_q: GpuTensor,        // [n_heads * head_dim]
    pub fa_gate: GpuTensor,     // [n_heads * head_dim]
    pub fa_k: GpuTensor,        // [n_kv_heads * head_dim]
    pub fa_v: GpuTensor,        // [n_kv_heads * head_dim]
    pub fa_attn_out: GpuTensor, // [n_heads * head_dim]

    // Shared (used by both layer types)
    pub o: GpuTensor,           // [dim]
    pub gate_ffn: GpuTensor,    // [hidden_dim]
    pub up: GpuTensor,          // [hidden_dim]
    pub ffn_hidden: GpuTensor,  // [hidden_dim]
    pub ffn_out: GpuTensor,     // [dim]

    // Sampling
    pub logits: GpuTensor,      // [vocab_size]
    pub sample_buf: GpuTensor,  // [2] — token_id + rng
    pub repeat_buf: GpuTensor,  // [repeat_window]

    // MagnumQuant rotation scratch: FWHT(x) shared across Q/K/V (or gate/up, etc).
    // Sized to max(dim, hidden_dim) — one rotation per batch replaces one per GEMV.
    pub x_rot: GpuTensor,       // [max(dim, hidden_dim)]
}

impl Qwen35Scratch {
    pub fn new(gpu: &mut Gpu, config: &Qwen35Config, repeat_window: usize) -> HipResult<Self> {
        let dim = config.dim;
        let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let qkv_dim = k_dim * 2 + v_dim;
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;

        Ok(Self {
            x: gpu.alloc_tensor(&[dim], DType::F32)?,
            tmp: gpu.alloc_tensor(&[dim], DType::F32)?,
            pos_buf: gpu.hip.malloc(4)?,

            dn_qkv: gpu.alloc_tensor(&[qkv_dim], DType::F32)?,
            dn_z: gpu.alloc_tensor(&[v_dim], DType::F32)?,
            dn_alpha: gpu.alloc_tensor(&[config.linear_num_value_heads], DType::F32)?,
            dn_beta: gpu.alloc_tensor(&[config.linear_num_value_heads], DType::F32)?,
            dn_conv_out: gpu.alloc_tensor(&[qkv_dim], DType::F32)?,
            dn_q: gpu.alloc_tensor(&[v_dim], DType::F32)?,
            dn_k: gpu.alloc_tensor(&[v_dim], DType::F32)?,
            dn_v: gpu.alloc_tensor(&[v_dim], DType::F32)?,
            dn_q_raw: gpu.alloc_tensor(&[k_dim], DType::F32)?,
            dn_k_raw: gpu.alloc_tensor(&[k_dim], DType::F32)?,
            dn_attn_out: gpu.alloc_tensor(&[v_dim], DType::F32)?,
            dn_normed: gpu.alloc_tensor(&[v_dim], DType::F32)?,

            fa_q_full: gpu.alloc_tensor(&[q_dim * 2], DType::F32)?,
            fa_q: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            fa_gate: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            fa_k: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            fa_v: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            fa_attn_out: gpu.alloc_tensor(&[q_dim], DType::F32)?,

            o: gpu.alloc_tensor(&[dim], DType::F32)?,
            gate_ffn: gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?,
            up: gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?,
            ffn_hidden: gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?,
            ffn_out: gpu.alloc_tensor(&[dim], DType::F32)?,

            logits: gpu.alloc_tensor(&[config.vocab_size], DType::F32)?,
            sample_buf: gpu.alloc_tensor(&[2], DType::F32)?,
            repeat_buf: gpu.alloc_tensor(&[repeat_window], DType::F32)?,
            x_rot: gpu.alloc_tensor(&[dim.max(config.hidden_dim)], DType::F32)?,
        })
    }

    /// Free all GPU tensors. Call before drop to return VRAM.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.x);
        let _ = gpu.free_tensor(self.tmp);
        let _ = gpu.hip.free(self.pos_buf);
        for t in [self.dn_qkv, self.dn_z, self.dn_alpha, self.dn_beta, self.dn_conv_out,
                   self.dn_q, self.dn_k, self.dn_v, self.dn_q_raw, self.dn_k_raw,
                   self.dn_attn_out, self.dn_normed,
                   self.fa_q_full, self.fa_q, self.fa_gate, self.fa_k, self.fa_v, self.fa_attn_out,
                   self.o, self.gate_ffn, self.up, self.ffn_hidden, self.ffn_out,
                   self.logits, self.sample_buf, self.repeat_buf, self.x_rot] {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// Zero-alloc forward pass using pre-allocated scratch buffers.
/// Logits stay on GPU in scratch.logits. Returns nothing — caller uses scratch.logits.
pub fn forward_scratch(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    let dim = config.dim;
    let pos_i32 = pos as i32;
    gpu.hip.memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;

    // Embedding lookup into scratch.x
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.token_embd, &scratch.x, token, dim)?,
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.token_embd, &scratch.x, token, dim)?,
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &scratch.x, token, dim)?,
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &scratch.x, token, dim)?,
        _ => panic!("unsupported embedding format"),
    }

    forward_scratch_layers(gpu, weights, config, pos, kv_cache, dn_state, scratch, None)
}

/// Per-layer batched intermediates used by `forward_prefill_batch`. Each
/// row is one token in the batch; rows are contiguous [N × K] blocks so
/// all kernels can treat them as row-major matrices.
///
/// Allocated lazily on the first batched prefill call that takes the MQ4
/// fast path — models that never hit that path (HF4 weights, FA-only
/// models, short prompts) never pay the VRAM cost. Sized to `max_batch`;
/// longer prompts are processed in chunks of `max_batch`.
pub struct PrefillBatchScratch {
    pub max_batch: usize,

    // Residual stream and rotation scratch — both [N × dim]
    pub x_batch: GpuTensor,
    pub x_rot_batch: GpuTensor,

    // LA-layer projection outputs
    pub dn_qkv_batch: GpuTensor,         // [N × qkv_dim]
    pub dn_z_batch: GpuTensor,           // [N × v_dim]
    pub dn_alpha_batch: GpuTensor,       // [N × n_v_heads]
    pub dn_beta_batch: GpuTensor,        // [N × n_v_heads]
    pub dn_q_raw_batch: GpuTensor,       // [N × k_dim] (pre repeat-interleave)
    pub dn_k_raw_batch: GpuTensor,       // [N × k_dim]
    pub dn_v_batch: GpuTensor,           // [N × v_dim]
    pub dn_q_batch: GpuTensor,           // [N × v_dim] (post repeat-interleave)
    pub dn_k_batch: GpuTensor,           // [N × v_dim]
    pub dn_attn_out_batch: GpuTensor,    // [N × v_dim]
    pub dn_normed_batch: GpuTensor,      // [N × v_dim]

    // FFN intermediates [N × hidden_dim]
    pub gate_ffn_batch: GpuTensor,
    pub up_batch: GpuTensor,
    // SwiGLU output (FWHT-rotated for MQ4) feeding w_down.
    pub ffn_hidden_batch: GpuTensor,

    // FWHT-rotated dn_normed [N × v_dim] feeding wo for MQ4 weights.
    // Decode path handles this via an internal mq_x_rot scratch inside
    // weight_gemv_residual; we need an explicit batched equivalent.
    pub dn_normed_rot_batch: GpuTensor,
}

impl PrefillBatchScratch {
    pub fn new(gpu: &mut Gpu, config: &Qwen35Config, max_batch: usize) -> HipResult<Self> {
        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let qkv_dim = k_dim * 2 + v_dim;
        let n_v_heads = config.linear_num_value_heads;

        Ok(Self {
            max_batch,
            x_batch:           gpu.alloc_tensor(&[max_batch * dim], DType::F32)?,
            x_rot_batch:       gpu.alloc_tensor(&[max_batch * dim], DType::F32)?,
            dn_qkv_batch:      gpu.alloc_tensor(&[max_batch * qkv_dim], DType::F32)?,
            dn_z_batch:        gpu.alloc_tensor(&[max_batch * v_dim],   DType::F32)?,
            dn_alpha_batch:    gpu.alloc_tensor(&[max_batch * n_v_heads], DType::F32)?,
            dn_beta_batch:     gpu.alloc_tensor(&[max_batch * n_v_heads], DType::F32)?,
            dn_q_raw_batch:    gpu.alloc_tensor(&[max_batch * k_dim],   DType::F32)?,
            dn_k_raw_batch:    gpu.alloc_tensor(&[max_batch * k_dim],   DType::F32)?,
            dn_v_batch:        gpu.alloc_tensor(&[max_batch * v_dim],   DType::F32)?,
            dn_q_batch:        gpu.alloc_tensor(&[max_batch * v_dim],   DType::F32)?,
            dn_k_batch:        gpu.alloc_tensor(&[max_batch * v_dim],   DType::F32)?,
            dn_attn_out_batch: gpu.alloc_tensor(&[max_batch * v_dim],   DType::F32)?,
            dn_normed_batch:   gpu.alloc_tensor(&[max_batch * v_dim],   DType::F32)?,
            gate_ffn_batch:    gpu.alloc_tensor(&[max_batch * hidden_dim], DType::F32)?,
            up_batch:          gpu.alloc_tensor(&[max_batch * hidden_dim], DType::F32)?,
            ffn_hidden_batch:  gpu.alloc_tensor(&[max_batch * hidden_dim], DType::F32)?,
            dn_normed_rot_batch: gpu.alloc_tensor(&[max_batch * v_dim],   DType::F32)?,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in [
            self.x_batch, self.x_rot_batch,
            self.dn_qkv_batch, self.dn_z_batch,
            self.dn_alpha_batch, self.dn_beta_batch,
            self.dn_q_raw_batch, self.dn_k_raw_batch, self.dn_v_batch,
            self.dn_q_batch, self.dn_k_batch,
            self.dn_attn_out_batch, self.dn_normed_batch,
            self.gate_ffn_batch, self.up_batch, self.ffn_hidden_batch,
            self.dn_normed_rot_batch,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// Batched prefill entry point: processes N prompt tokens in one call,
/// writing the last token's logits into `scratch.logits` and leaving
/// the KV cache + DeltaNet state advanced by N positions.
///
/// Takes the batched kernel path when ALL linear-attention layer weights
/// are MQ4G256 (the batched element-wise kernels are MQ-specific).
/// Otherwise falls back to a per-token loop over `forward_scratch` that's
/// byte-identical to decode. FA layers always use a per-token gather/scatter
/// fallback — the FA causal attention kernel can't yet be batched (task #71).
///
/// `gated_delta_net_q8` is called N times per LA layer (once per token)
/// using `gated_delta_net_q8_batch_seq`, preserving the byte-exact
/// stochastic-rounding trajectory vs decode.
///
/// `tokens`: slice of prompt tokens to prefill in order.
/// `start_pos`: first KV cache / DeltaNet position to write. Positions
/// `start_pos .. start_pos + tokens.len()` get populated.
/// On return, `scratch.logits` holds the logits for the *last* token
/// (position `start_pos + tokens.len() - 1`).
pub fn forward_prefill_batch(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    // Threshold below which the batching overhead isn't worth the alloc +
    // per-layer dispatch. Single-token prefill obviously should not take
    // the batched path.
    const MIN_BATCH: usize = 2;
    // Upper bound on the PrefillBatchScratch — large prompts get split
    // into chunks of this size and processed in a loop.
    const MAX_BATCH: usize = 64;

    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }

    // Fast path requires (a) all LA weights are MQ4G256 and (b) Q8 S-state.
    // The batched element-wise kernels hardcode the MQ4 FWHT rotation, and
    // the batched GDN path uses `gated_delta_net_q8_batch_seq`.
    // `HIPFIRE_PREFILL_BATCHED=0` forces the per-token fallback (escape
    // hatch for regression bisecting or diagnosing hardware-specific issues).
    let force_fallback = std::env::var("HIPFIRE_PREFILL_BATCHED").ok().as_deref() == Some("0");
    let eligible = !force_fallback
        && n >= MIN_BATCH
        && dn_state.quant == StateQuant::Q8
        && weights.layers.iter().any(|lw| matches!(lw, LayerWeights::DeltaNet(_)))
        && weights.layers.iter().all(|lw| match lw {
            LayerWeights::DeltaNet(l) =>
                l.wqkv.gpu_dtype  == DType::MQ4G256 &&
                l.wz.gpu_dtype    == DType::MQ4G256 &&
                l.w_beta.gpu_dtype == DType::MQ4G256 &&
                l.w_alpha.gpu_dtype == DType::MQ4G256 &&
                l.wo.gpu_dtype    == DType::MQ4G256 &&
                l.w_gate.gpu_dtype == DType::MQ4G256 &&
                l.w_up.gpu_dtype   == DType::MQ4G256 &&
                l.w_down.gpu_dtype == DType::MQ4G256,
            LayerWeights::FullAttn(_) => true, // FA layer will take the gather/scatter path
        });

    if !eligible {
        // Fallback: per-token loop, byte-identical to decode.
        for (i, &tok) in tokens.iter().enumerate() {
            forward_scratch(gpu, weights, config, tok, start_pos + i, kv_cache, dn_state, scratch)?;
        }
        return Ok(());
    }

    // Allocate the batch scratch once, reuse across chunks. Scope with an
    // inner closure so the explicit free runs even on error.
    let pbs = PrefillBatchScratch::new(gpu, config, MAX_BATCH)?;
    let result = (|| -> HipResult<()> {
        let mut chunk_start = 0usize;
        while chunk_start < n {
            let chunk_end = (chunk_start + MAX_BATCH).min(n);
            let chunk = &tokens[chunk_start..chunk_end];
            forward_prefill_chunk(
                gpu, weights, config, chunk, start_pos + chunk_start,
                kv_cache, dn_state, scratch, &pbs,
            )?;
            chunk_start = chunk_end;
        }
        Ok(())
    })();
    pbs.free_gpu(gpu);
    result
}

/// Process one chunk of up to `pbs.max_batch` tokens through the batched
/// prefill path. All LA layers go through batched kernels; all FA layers
/// go through a per-token gather/scatter loop with the inline FA body.
#[allow(clippy::too_many_arguments)]
fn forward_prefill_chunk(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    s: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
) -> HipResult<()> {
    let n = tokens.len();
    debug_assert!(n > 0);
    debug_assert!(n <= pbs.max_batch);

    let dim = config.dim;
    let hidden_dim = config.hidden_dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;
    let dim_row_bytes = dim * 4;

    // ── 1. Embed each token into s.x, then copy into pbs.x_batch row ──────
    for (i, &tok) in tokens.iter().enumerate() {
        match weights.embd_format {
            EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.token_embd, &s.x, tok, dim)?,
            EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.token_embd, &s.x, tok, dim)?,
            EmbeddingFormat::Q8_0     => gpu.embedding_lookup_q8(&weights.token_embd, &s.x, tok, dim)?,
            EmbeddingFormat::F32      => gpu.embedding_lookup(&weights.token_embd, &s.x, tok, dim)?,
            _ => panic!("unsupported embedding format"),
        }
        gpu.hip.memcpy_dtod_at(&pbs.x_batch.buf, i * dim_row_bytes, &s.x.buf, 0, dim_row_bytes)?;
    }

    // ── 2. Per-layer loop ────────────────────────────────────────────────
    let mut delta_layer_idx = 0usize;
    let mut kv_layer_idx = 0usize;

    for layer_idx in 0..config.n_layers {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                // Batched rmsnorm + FWHT rotation for the LA preamble.
                // x_batch / x_rot_batch are [N × dim] contiguous.
                gpu.fused_rmsnorm_rotate_mq_batched(
                    &pbs.x_batch, &layer.attn_norm, &pbs.x_rot_batch, dim, config.norm_eps, n,
                )?;

                // Batched 4-way LA projection (wqkv + wz + w_beta + w_alpha).
                gpu.gemm_qkvza_hfq4g256(
                    &layer.wqkv.buf, &layer.wz.buf, &layer.w_beta.buf, &layer.w_alpha.buf,
                    &pbs.x_rot_batch,
                    &pbs.dn_qkv_batch, &pbs.dn_z_batch, &pbs.dn_beta_batch, &pbs.dn_alpha_batch,
                    layer.wqkv.m, layer.wz.m, layer.w_beta.m, layer.w_alpha.m,
                    layer.wqkv.k, n,
                )?;

                // Fused sigmoid(beta) + alpha_gate(alpha) — [N × n_v_heads] each.
                gpu.fused_sigmoid_alpha_gate_f32_batched(
                    &pbs.dn_beta_batch, &pbs.dn_alpha_batch,
                    &layer.dt_bias, &layer.a_log,
                    n_v_heads, n,
                )?;

                // Conv1d + SiLU + Q/K/V split, advancing state N steps.
                // State advance is byte-identical to N single-token calls.
                gpu.conv1d_silu_split_f32_n(
                    &pbs.dn_q_raw_batch, &pbs.dn_k_raw_batch, &pbs.dn_v_batch,
                    &pbs.dn_qkv_batch, &layer.conv_weight,
                    &dn_state.conv_states[delta_layer_idx],
                    k_dim, v_dim, n,
                )?;

                // Batched L2-norm(Q) + L2-norm(K) + scale(Q).
                gpu.fused_qk_l2_norm_scale_f32_batched(
                    &pbs.dn_q_raw_batch, &pbs.dn_k_raw_batch,
                    config.linear_num_key_heads, hd,
                    1.0 / (hd as f32).sqrt(), config.norm_eps, n,
                )?;

                // Repeat-interleave Q/K if n_key_heads < n_v_heads.
                // 0.8B has n_key=n_value=16 so the memcpy path runs.
                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    // repeat_interleave_qk_f32 takes single-token tensors, so
                    // loop per row using scratch.dn_q_raw/dn_k_raw as staging.
                    // Cheap vs. the GEMM + GDN work.
                    for i in 0..n {
                        gpu.hip.memcpy_dtod_at(&s.dn_q_raw.buf, 0, &pbs.dn_q_raw_batch.buf, i * k_dim * 4, k_dim * 4)?;
                        gpu.hip.memcpy_dtod_at(&s.dn_k_raw.buf, 0, &pbs.dn_k_raw_batch.buf, i * k_dim * 4, k_dim * 4)?;
                        gpu.repeat_interleave_qk_f32(
                            &s.dn_q_raw, &s.dn_k_raw, &s.dn_q, &s.dn_k,
                            config.linear_num_key_heads, ratio, hd,
                        )?;
                        gpu.hip.memcpy_dtod_at(&pbs.dn_q_batch.buf, i * v_dim * 4, &s.dn_q.buf, 0, v_dim * 4)?;
                        gpu.hip.memcpy_dtod_at(&pbs.dn_k_batch.buf, i * v_dim * 4, &s.dn_k.buf, 0, v_dim * 4)?;
                    }
                } else {
                    // n_key_heads == n_v_heads → k_dim == v_dim, memcpy the whole block.
                    gpu.hip.memcpy_dtod(&pbs.dn_q_batch.buf, &pbs.dn_q_raw_batch.buf, n * k_dim * 4)?;
                    gpu.hip.memcpy_dtod(&pbs.dn_k_batch.buf, &pbs.dn_k_raw_batch.buf, n * k_dim * 4)?;
                }

                // Gated Delta Net — N sequential calls with offset pointers.
                // Byte-exact with decode because each call rounds S_q8 after
                // its single-token update.
                gpu.gated_delta_net_q8_batch_seq(
                    &pbs.dn_q_batch, &pbs.dn_k_batch, &pbs.dn_v_batch,
                    &pbs.dn_alpha_batch, &pbs.dn_beta_batch,
                    &dn_state.s_matrices[delta_layer_idx],
                    &dn_state.s_scales[delta_layer_idx],
                    &pbs.dn_attn_out_batch,
                    n, n_v_heads, config.linear_value_head_dim,
                )?;

                // Batched gated output norm.
                gpu.gated_norm_f32_batched(
                    &pbs.dn_attn_out_batch, &pbs.dn_z_batch, &layer.norm_weight,
                    &pbs.dn_normed_batch,
                    n_v_heads, config.linear_value_head_dim, config.norm_eps, n,
                )?;

                // Batched wo + residual: x_batch += wo · FWHT(dn_normed_batch).
                // CRITICAL: For MQ4 weights, the decode path's weight_gemv_residual
                // internally FWHT-rotates dn_normed into mq_x_rot before calling
                // gemv_hfq4g256_residual. The batched gemm_hfq4g256_residual kernel
                // does plain HFQ4 math, so we must apply the rotation ourselves
                // (MQ4 weights are pre-rotated at quant time; math requires
                // dot(rot(W), rot(x)) = dot(W, x)).
                gpu.rotate_x_mq_batched(
                    &pbs.dn_normed_batch, &pbs.dn_normed_rot_batch, layer.wo.k, n,
                )?;
                gpu.gemm_hfq4g256_residual(
                    &layer.wo.buf, &pbs.dn_normed_rot_batch, &pbs.x_batch,
                    layer.wo.m, layer.wo.k, n,
                )?;

                // FFN: fused rmsnorm + rotate.
                gpu.fused_rmsnorm_rotate_mq_batched(
                    &pbs.x_batch, &layer.ffn_norm, &pbs.x_rot_batch, dim, config.norm_eps, n,
                )?;

                // Batched gate+up projection.
                gpu.gemm_gate_up_hfq4g256(
                    &layer.w_gate.buf, &layer.w_up.buf,
                    &pbs.x_rot_batch,
                    &pbs.gate_ffn_batch, &pbs.up_batch,
                    layer.w_gate.m, layer.w_up.m,
                    layer.w_gate.k, n,
                )?;

                // Batched fused SwiGLU + FWHT rotation for w_down input.
                gpu.fused_silu_mul_rotate_mq_batched(
                    &pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch,
                    hidden_dim, n,
                )?;

                // Batched w_down + residual.
                gpu.gemm_hfq4g256_residual(
                    &layer.w_down.buf, &pbs.ffn_hidden_batch, &pbs.x_batch,
                    layer.w_down.m, layer.w_down.k, n,
                )?;

                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttn(_layer), LayerType::FullAttention) => {
                // FA path: gather each row of pbs.x_batch into s.x, run the
                // inline FA layer body, scatter back. Byte-identical to the
                // decode path for each token.
                for i in 0..n {
                    let pos = start_pos + i;
                    // Gather: x_batch row i → s.x.
                    gpu.hip.memcpy_dtod_at(&s.x.buf, 0, &pbs.x_batch.buf, i * dim_row_bytes, dim_row_bytes)?;

                    // Write pos_buf for this token.
                    let pos_i32 = pos as i32;
                    gpu.hip.memcpy_htod(&s.pos_buf, &pos_i32.to_ne_bytes())?;

                    // Run ONE FA layer body using existing per-token helper.
                    run_fa_layer_body(gpu, weights, config, layer_idx, kv_layer_idx, pos, kv_cache, s)?;

                    // Scatter: s.x → x_batch row i.
                    gpu.hip.memcpy_dtod_at(&pbs.x_batch.buf, i * dim_row_bytes, &s.x.buf, 0, dim_row_bytes)?;
                }
                kv_layer_idx += 1;
            }

            _ => panic!("layer type mismatch at layer {layer_idx}"),
        }
    }

    // ── 3. Final logits from the last token only ─────────────────────────
    // Copy last row of x_batch into s.x, run output rmsnorm + lm_head.
    let last = n - 1;
    gpu.hip.memcpy_dtod_at(&s.x.buf, 0, &pbs.x_batch.buf, last * dim_row_bytes, dim_row_bytes)?;
    gpu.rmsnorm_f32(&s.x, &weights.output_norm, &s.tmp, config.norm_eps)?;
    weight_gemv(gpu, &weights.output, &s.tmp, &s.logits)?;

    Ok(())
}

/// Run a single FullAttn layer body on s.x at position `pos`. Extracted
/// for use from the batched prefill path's FA-layer fallback. Byte-exact
/// with the FA branch of forward_scratch_layers.
#[allow(clippy::too_many_arguments)]
fn run_fa_layer_body(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    layer_idx: usize,
    _kv_layer_idx: usize,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    s: &Qwen35Scratch,
) -> HipResult<()> {
    let layer = match &weights.layers[layer_idx] {
        LayerWeights::FullAttn(l) => l,
        _ => unreachable!(),
    };

    // Fused rmsnorm + FWHT rotation for wq/wk/wv.
    let x_rot = fused_rmsnorm_rotate_for_mq(
        gpu, &layer.wq, &s.x, &layer.attn_norm, &s.tmp, &s.x_rot, config.norm_eps,
    )?;

    // Cross-arch fast path: fused 3-way projection for wq+wk+wv.
    let dt = layer.wq.gpu_dtype;
    let fused_fa3_ok = (dt == DType::MQ4G256 || dt == DType::HFQ4G256)
        && layer.wk.gpu_dtype == dt
        && layer.wv.gpu_dtype == dt;
    if fused_fa3_ok {
        let eff_x = match x_rot { Some(xr) => xr, None => &s.tmp };
        gpu.fused_qkv_hfq4g256(
            &layer.wq.buf, &layer.wk.buf, &layer.wv.buf,
            eff_x,
            &s.fa_q_full, &s.fa_k, &s.fa_v,
            layer.wq.m, layer.wk.m, layer.wv.m,
            layer.wq.k,
        )?;
    } else {
        weight_gemv_prerotated(gpu, &layer.wq, &s.tmp, x_rot, &s.fa_q_full)?;
        weight_gemv_prerotated(gpu, &layer.wk, &s.tmp, x_rot, &s.fa_k)?;
        weight_gemv_prerotated(gpu, &layer.wv, &s.tmp, x_rot, &s.fa_v)?;
    }

    gpu.deinterleave_f32(&s.fa_q_full, &s.fa_q, &s.fa_gate, config.n_heads, config.head_dim)?;
    gpu.rmsnorm_batched(&s.fa_q, &layer.q_norm, &s.fa_q, config.n_heads, config.head_dim, config.norm_eps)?;
    let kv_dim = config.n_kv_heads * config.head_dim;
    gpu.rmsnorm_batched(&s.fa_k, &layer.k_norm, &s.fa_k, config.n_kv_heads, config.head_dim, config.norm_eps)?;

    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
    gpu.rope_partial_interleaved_f32(&s.fa_q, &s.fa_k, pos as i32,
        config.n_heads, config.n_kv_heads, config.head_dim, n_rot, config.rope_theta)?;

    if kv_cache.quant_hf4v {
        gpu.kv_cache_write_q8_0(&kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf, config.n_kv_heads, config.head_dim)?;
        gpu.kv_cache_write_hf4v_256(&kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf, config.n_kv_heads, config.head_dim)?;
        gpu.attention_q8k_hf4v_256(
            &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
            &s.fa_attn_out, &s.pos_buf, pos + 1,
            config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
        )?;
    } else if kv_cache.quant_asym && kv_cache.is_boundary(_kv_layer_idx) {
        gpu.kv_cache_write_q8_0(&kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf, config.n_kv_heads, config.head_dim)?;
        gpu.kv_cache_write_q8_0(&kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf, config.n_kv_heads, config.head_dim)?;
        gpu.attention_q8_0_kv(
            &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
            &s.fa_attn_out, &s.pos_buf, pos + 1, config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
        )?;
    } else if kv_cache.quant_asym {
        let s1 = kv_cache.turbo_signs1.as_ref().unwrap();
        let s2 = kv_cache.turbo_signs2.as_ref().unwrap();
        gpu.kv_cache_write_q8_0(&kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf, config.n_kv_heads, config.head_dim)?;
        gpu.kv_cache_write_turbo4_v256(&kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf, s1, s2, config.n_kv_heads, config.head_dim)?;
        gpu.attention_q8k_turbo4v_256(
            &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
            &s.fa_attn_out, &s.pos_buf, s1, s2, pos + 1,
            config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
        )?;
    } else if kv_cache.quant_turbo > 0 {
        let s1 = kv_cache.turbo_signs1.as_ref().unwrap();
        let s2 = kv_cache.turbo_signs2.as_ref().unwrap();
        match kv_cache.quant_turbo {
            4 => {
                gpu.kv_cache_write_turbo4_fused(
                    &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                    &s.fa_k, &s.fa_v, &s.pos_buf, s1, s2, config.n_kv_heads, config.head_dim)?;
                gpu.attention_turbo4_kv(&s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                    &s.fa_attn_out, &s.pos_buf, s1, s2, pos + 1, config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq)?;
            }
            3 => {
                gpu.kv_cache_write_turbo3_fused(
                    &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                    &s.fa_k, &s.fa_v, &s.pos_buf, s1, s2, config.n_kv_heads, config.head_dim)?;
                gpu.attention_turbo3_kv(&s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                    &s.fa_attn_out, &s.pos_buf, s1, s2, pos + 1, config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq)?;
            }
            2 => {
                gpu.kv_cache_write_turbo2_fused(
                    &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                    &s.fa_k, &s.fa_v, &s.pos_buf, s1, s2, config.n_kv_heads, config.head_dim)?;
                gpu.attention_turbo2_kv(&s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                    &s.fa_attn_out, &s.pos_buf, s1, s2, pos + 1, config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq)?;
            }
            _ => {}
        }
    } else if kv_cache.quant_q8 {
        gpu.kv_cache_write_q8_0(&kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf, config.n_kv_heads, config.head_dim)?;
        gpu.kv_cache_write_q8_0(&kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf, config.n_kv_heads, config.head_dim)?;
        gpu.attention_q8_0_kv(
            &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
            &s.fa_attn_out, &s.pos_buf, pos + 1,
            config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
        )?;
    } else {
        gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf, kv_dim)?;
        gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf, kv_dim)?;
        gpu.attention_f32(
            &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
            &s.fa_attn_out, &s.pos_buf, pos + 1,
            config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
        )?;
    }

    gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
    weight_gemv_residual(gpu, &layer.wo, &s.fa_attn_out, &s.x)?;

    // FFN: fused rmsnorm + rotate for w_gate/w_up.
    let x_rot = fused_rmsnorm_rotate_for_mq(
        gpu, &layer.w_gate, &s.x, &layer.ffn_norm, &s.tmp, &s.x_rot, config.norm_eps,
    )?;
    let dt_g = layer.w_gate.gpu_dtype;
    let fused_gu_ok = (dt_g == DType::MQ4G256 || dt_g == DType::HFQ4G256)
        && layer.w_up.gpu_dtype == dt_g;
    if fused_gu_ok {
        let eff_x = match x_rot { Some(xr) => xr, None => &s.tmp };
        gpu.fused_gate_up_hfq4g256(
            &layer.w_gate.buf, &layer.w_up.buf,
            eff_x,
            &s.gate_ffn, &s.up,
            layer.w_gate.m, layer.w_up.m,
            layer.w_gate.k,
        )?;
    } else {
        weight_gemv_prerotated(gpu, &layer.w_gate, &s.tmp, x_rot, &s.gate_ffn)?;
        weight_gemv_prerotated(gpu, &layer.w_up, &s.tmp, x_rot, &s.up)?;
    }
    weight_gemv_swiglu_residual(
        gpu, &layer.w_down, &s.gate_ffn, &s.up, &s.ffn_hidden, &s.x,
    )?;

    Ok(())
}

/// Same as `forward_scratch` but also extracts hidden states from the
/// configured target layers into `hidden_rb`. Used by the DFlash draft path
/// during target verification. `hidden_rb.advance_head()` is called once
/// automatically at the end of the forward pass.
pub fn forward_scratch_with_hidden(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    hidden_rb: &mut HiddenStateRingBuffer,
) -> HipResult<()> {
    let dim = config.dim;
    let pos_i32 = pos as i32;
    gpu.hip.memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;

    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.token_embd, &scratch.x, token, dim)?,
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.token_embd, &scratch.x, token, dim)?,
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &scratch.x, token, dim)?,
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &scratch.x, token, dim)?,
        _ => panic!("unsupported embedding format"),
    }

    forward_scratch_layers(gpu, weights, config, pos, kv_cache, dn_state, scratch, Some(hidden_rb))?;
    hidden_rb.advance_head();
    Ok(())
}

/// Zero-alloc forward from pre-computed embedding in scratch.x.
pub fn forward_scratch_embed(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    embedding_data: &[f32],
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    let pos_i32 = pos as i32;
    gpu.hip.memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
    // Upload embedding directly into scratch.x
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(embedding_data.as_ptr() as *const u8, embedding_data.len() * 4)
    };
    gpu.hip.memcpy_htod(&scratch.x.buf, bytes)?;
    forward_scratch_layers(gpu, weights, config, pos, kv_cache, dn_state, scratch, None)
}

/// Layer loop using scratch buffers. Zero alloc/free per token.
///
/// `hidden_rb`: if Some, the layer loop extracts post-residual hidden states
/// from the configured target layers into the ring buffer. When None (default
/// for normal inference) this is branch-free and has zero overhead.
fn forward_scratch_layers(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    s: &Qwen35Scratch,
    mut hidden_rb: Option<&mut HiddenStateRingBuffer>,
) -> HipResult<()> {
    let dim = config.dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let qkv_dim = k_dim * 2 + v_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;

    let mut delta_layer_idx = 0usize;
    let mut kv_layer_idx = 0usize;

    for layer_idx in 0..config.n_layers {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                // Fused RMSNorm + FWHT rotation (Phase 3.6). For MQ4 weights this
                // writes rmsnorm(x) followed by FWHT into s.x_rot in a single
                // kernel launch. For non-MQ weights it falls back to plain rmsnorm
                // into s.tmp. Either way, wqkv/wz/w_beta/w_alpha share this input.
                let x_rot = fused_rmsnorm_rotate_for_mq(
                    gpu, &layer.wqkv, &s.x, &layer.attn_norm, &s.tmp, &s.x_rot, config.norm_eps,
                )?;
                // Cross-arch fast path: one fused 4-way projection kernel
                // (wqkv + wz + w_beta + w_alpha) in a single launch. Works
                // for BOTH MQ4 (weights FWHT-rotated, input x_rot FWHT-rotated)
                // and HF4 (weights not rotated, input is plain rmsnormed x).
                // The kernel math is the same — it's a gemv_hfq4g256 inner
                // loop; MQ4 and HF4 just live in different "rotated spaces"
                // and the caller hands the matching x. Inner loop is unified
                // across all RDNA generations after the 5302926 4-accumulator
                // port to gemv_hfq4g256.hip.
                let dt = layer.wqkv.gpu_dtype;
                let fused_la4_ok = (dt == DType::MQ4G256 || dt == DType::HFQ4G256)
                    && layer.wz.gpu_dtype == dt
                    && layer.w_beta.gpu_dtype == dt
                    && layer.w_alpha.gpu_dtype == dt;
                if fused_la4_ok {
                    // MQ4: x_rot is Some(rotated x); HF4: x_rot is None and
                    // s.tmp holds the plain rmsnormed x from the fallback path.
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_qkvza_hfq4g256(
                        &layer.wqkv.buf, &layer.wz.buf, &layer.w_beta.buf, &layer.w_alpha.buf,
                        eff_x,
                        &s.dn_qkv, &s.dn_z, &s.dn_beta, &s.dn_alpha,
                        layer.wqkv.m, layer.wz.m, layer.w_beta.m, layer.w_alpha.m,
                        layer.wqkv.k,
                    )?;
                } else {
                    weight_gemv_prerotated(gpu, &layer.wqkv, &s.tmp, x_rot, &s.dn_qkv)?;
                    weight_gemv_prerotated(gpu, &layer.wz, &s.tmp, x_rot, &s.dn_z)?;
                    weight_gemv_prerotated(gpu, &layer.w_beta, &s.tmp, x_rot, &s.dn_beta)?;
                    weight_gemv_prerotated(gpu, &layer.w_alpha, &s.tmp, x_rot, &s.dn_alpha)?;
                }
                // Fused sigmoid(dn_beta) + alpha_gate(dn_alpha). Both ops are
                // elementwise scalar transforms on independent buffers of size
                // n_v_heads — merging into one launch shaves one dispatch per LA.
                gpu.fused_sigmoid_alpha_gate_f32(
                    &s.dn_beta, &s.dn_alpha, &layer.dt_bias, &layer.a_log, n_v_heads,
                )?;

                // Fused conv1d+SiLU+split: writes directly to q_raw/k_raw/v,
                // eliminating the 3 DtoD copies that used to follow a
                // contiguous conv1d_silu into dn_conv_out.
                gpu.conv1d_silu_split_f32(
                    &s.dn_q_raw, &s.dn_k_raw, &s.dn_v,
                    &s.dn_qkv, &layer.conv_weight,
                    &dn_state.conv_states[delta_layer_idx],
                    k_dim, v_dim,
                )?;

                // Fused: l2_norm(q_raw) + l2_norm(k_raw) + scale(q_raw).
                // Three launches collapsed to one — saves ~2 dispatches per
                // linear-attention layer (~300 µs/forward on 0.8B MQ4).
                gpu.fused_qk_l2_norm_scale_f32(
                    &s.dn_q_raw,
                    &s.dn_k_raw,
                    config.linear_num_key_heads,
                    hd,
                    1.0 / (hd as f32).sqrt(),
                    config.norm_eps,
                )?;

                // Repeat-interleave Q/K if needed.
                // Phase 3a-A fix: replace per-head memcpy loop with one fused kernel.
                // For 9B (n_key=16, n_val=32, ratio=2): saves 64 hipMemcpy calls
                // per layer × 24 layers = 1536 calls per forward, ~1.7 ms savings.
                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    gpu.repeat_interleave_qk_f32(
                        &s.dn_q_raw, &s.dn_k_raw, &s.dn_q, &s.dn_k,
                        config.linear_num_key_heads, ratio, hd,
                    )?;
                } else {
                    gpu.hip.memcpy_dtod(&s.dn_q.buf, &s.dn_q_raw.buf, k_dim * 4)?;
                    gpu.hip.memcpy_dtod(&s.dn_k.buf, &s.dn_k_raw.buf, k_dim * 4)?;
                }

                match dn_state.quant {
                    StateQuant::FP32 => gpu.gated_delta_net_f32(
                        &s.dn_q, &s.dn_k, &s.dn_v, &s.dn_alpha, &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx], &s.dn_attn_out,
                        1, n_v_heads, config.linear_value_head_dim,
                    )?,
                    StateQuant::Q8 => gpu.gated_delta_net_q8(
                        &s.dn_q, &s.dn_k, &s.dn_v, &s.dn_alpha, &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx], &s.dn_attn_out,
                        1, n_v_heads, config.linear_value_head_dim,
                    )?,
                    StateQuant::Q4 => gpu.gated_delta_net_q4(
                        &s.dn_q, &s.dn_k, &s.dn_v, &s.dn_alpha, &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx], &s.dn_attn_out,
                        1, n_v_heads, config.linear_value_head_dim,
                    )?,
                }

                gpu.gated_norm_f32(&s.dn_attn_out, &s.dn_z, &layer.norm_weight, &s.dn_normed,
                    n_v_heads, config.linear_value_head_dim, config.norm_eps)?;
                // Fused wo GEMV + residual add: s.x += layer.wo * s.dn_normed
                weight_gemv_residual(gpu, &layer.wo, &s.dn_normed, &s.x)?;

                // FFN: fused rmsnorm + rotate for w_gate/w_up.
                let x_rot = fused_rmsnorm_rotate_for_mq(
                    gpu, &layer.w_gate, &s.x, &layer.ffn_norm, &s.tmp, &s.x_rot, config.norm_eps,
                )?;
                // Cross-arch fast path: fused gate+up in one launch. Works
                // for both MQ4 (x_rot Some) and HF4 (x_rot None → s.tmp).
                let dt_g = layer.w_gate.gpu_dtype;
                let fused_gu_ok = (dt_g == DType::MQ4G256 || dt_g == DType::HFQ4G256)
                    && layer.w_up.gpu_dtype == dt_g;
                if fused_gu_ok {
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_gate_up_hfq4g256(
                        &layer.w_gate.buf, &layer.w_up.buf,
                        eff_x,
                        &s.gate_ffn, &s.up,
                        layer.w_gate.m, layer.w_up.m,
                        layer.w_gate.k,
                    )?;
                } else {
                    weight_gemv_prerotated(gpu, &layer.w_gate, &s.tmp, x_rot, &s.gate_ffn)?;
                    weight_gemv_prerotated(gpu, &layer.w_up, &s.tmp, x_rot, &s.up)?;
                }
                // Fused SwiGLU + w_down residual GEMV:
                //   MQ4: fused_silu_rotate(gate,up) + gemv_residual(w_down, rotated, x)
                //   HF4: silu_mul + weight_gemv_residual (unchanged)
                weight_gemv_swiglu_residual(
                    gpu, &layer.w_down, &s.gate_ffn, &s.up, &s.ffn_hidden, &s.x,
                )?;

                if let Some(ref rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_at_head(gpu, slot, &s.x)?;
                    }
                }

                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttn(layer), LayerType::FullAttention) => {
                // Fused rmsnorm + FWHT rotation for wq/wk/wv (all share input).
                let x_rot = fused_rmsnorm_rotate_for_mq(
                    gpu, &layer.wq, &s.x, &layer.attn_norm, &s.tmp, &s.x_rot, config.norm_eps,
                )?;
                // Cross-arch fast path: fused 3-way projection for wq+wk+wv.
                // Works for MQ4 and HF4 — same kernel math as the LA 4-way.
                let dt = layer.wq.gpu_dtype;
                let fused_fa3_ok = (dt == DType::MQ4G256 || dt == DType::HFQ4G256)
                    && layer.wk.gpu_dtype == dt
                    && layer.wv.gpu_dtype == dt;
                if fused_fa3_ok {
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_qkv_hfq4g256(
                        &layer.wq.buf, &layer.wk.buf, &layer.wv.buf,
                        eff_x,
                        &s.fa_q_full, &s.fa_k, &s.fa_v,
                        layer.wq.m, layer.wk.m, layer.wv.m,
                        layer.wq.k,
                    )?;
                } else {
                    weight_gemv_prerotated(gpu, &layer.wq, &s.tmp, x_rot, &s.fa_q_full)?;
                    weight_gemv_prerotated(gpu, &layer.wk, &s.tmp, x_rot, &s.fa_k)?;
                    weight_gemv_prerotated(gpu, &layer.wv, &s.tmp, x_rot, &s.fa_v)?;
                }

                // Split interleaved Q+gate (single kernel instead of per-head memcpy loop)
                gpu.deinterleave_f32(&s.fa_q_full, &s.fa_q, &s.fa_gate, config.n_heads, config.head_dim)?;

                gpu.rmsnorm_batched(&s.fa_q, &layer.q_norm, &s.fa_q, config.n_heads, config.head_dim, config.norm_eps)?;

                let kv_dim = config.n_kv_heads * config.head_dim;
                gpu.rmsnorm_batched(&s.fa_k, &layer.k_norm, &s.fa_k, config.n_kv_heads, config.head_dim, config.norm_eps)?;

                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                gpu.rope_partial_interleaved_f32(&s.fa_q, &s.fa_k, pos as i32,
                    config.n_heads, config.n_kv_heads, config.head_dim, n_rot, config.rope_theta)?;

                if kv_cache.quant_hf4v {
                    // HF4-V: Q8 K + hipfire-native 4-bit V (no FWHT, 1 FMA dequant)
                    gpu.kv_cache_write_q8_0(&kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf, config.n_kv_heads, config.head_dim)?;
                    gpu.kv_cache_write_hf4v_256(&kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf, config.n_kv_heads, config.head_dim)?;
                    gpu.attention_q8k_hf4v_256(
                        &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_attn_out, &s.pos_buf, pos + 1,
                        config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
                    )?;
                } else if kv_cache.quant_asym && kv_cache.is_boundary(kv_layer_idx) {
                    // Boundary layer (LA-V7): Q8 for both K and V (full quality)
                    gpu.kv_cache_write_q8_0(&kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf, config.n_kv_heads, config.head_dim)?;
                    gpu.kv_cache_write_q8_0(&kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf, config.n_kv_heads, config.head_dim)?;
                    gpu.attention_q8_0_kv(
                        &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_attn_out, &s.pos_buf, pos + 1, config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
                    )?;
                } else if kv_cache.quant_asym {
                    // Middle layer: Q8 K + turbo4 V (V compression is free)
                    let s1 = kv_cache.turbo_signs1.as_ref().unwrap();
                    let s2 = kv_cache.turbo_signs2.as_ref().unwrap();
                    gpu.kv_cache_write_q8_0(&kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf, config.n_kv_heads, config.head_dim)?;
                    gpu.kv_cache_write_turbo4_v256(&kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf, s1, s2, config.n_kv_heads, config.head_dim)?;
                    gpu.attention_q8k_turbo4v_256(
                        &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_attn_out, &s.pos_buf, s1, s2, pos + 1,
                        config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
                    )?;
                } else if kv_cache.quant_turbo > 0 {
                    let s1 = kv_cache.turbo_signs1.as_ref().unwrap();
                    let s2 = kv_cache.turbo_signs2.as_ref().unwrap();
                    match kv_cache.quant_turbo {
                        4 => {
                            gpu.kv_cache_write_turbo4_fused(
                                &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                                &s.fa_k, &s.fa_v, &s.pos_buf, s1, s2, config.n_kv_heads, config.head_dim)?;
                            gpu.attention_turbo4_kv(&s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out, &s.pos_buf, s1, s2, pos + 1, config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq)?;
                        }
                        3 => {
                            gpu.kv_cache_write_turbo3_fused(
                                &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                                &s.fa_k, &s.fa_v, &s.pos_buf, s1, s2, config.n_kv_heads, config.head_dim)?;
                            gpu.attention_turbo3_kv(&s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out, &s.pos_buf, s1, s2, pos + 1, config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq)?;
                        }
                        2 => {
                            gpu.kv_cache_write_turbo2_fused(
                                &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                                &s.fa_k, &s.fa_v, &s.pos_buf, s1, s2, config.n_kv_heads, config.head_dim)?;
                            gpu.attention_turbo2_kv(&s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out, &s.pos_buf, s1, s2, pos + 1, config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq)?;
                        }
                        _ => {}
                    }
                } else if kv_cache.quant_q8 {
                    gpu.kv_cache_write_q8_0(&kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf, config.n_kv_heads, config.head_dim)?;
                    gpu.kv_cache_write_q8_0(&kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf, config.n_kv_heads, config.head_dim)?;
                    gpu.attention_q8_0_kv(
                        &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_attn_out, &s.pos_buf, pos + 1,
                        config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
                    )?;
                } else {
                    gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf, kv_dim)?;
                    gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf, kv_dim)?;
                    gpu.attention_f32(
                        &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_attn_out, &s.pos_buf, pos + 1,
                        config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
                    )?;
                }

                // Fused: fa_attn_out *= sigmoid(fa_gate). Two launches → one.
                gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
                // Fused wo GEMV + residual add: s.x += layer.wo * s.fa_attn_out
                weight_gemv_residual(gpu, &layer.wo, &s.fa_attn_out, &s.x)?;

                // FFN: fused rmsnorm + rotate for w_gate/w_up.
                let x_rot = fused_rmsnorm_rotate_for_mq(
                    gpu, &layer.w_gate, &s.x, &layer.ffn_norm, &s.tmp, &s.x_rot, config.norm_eps,
                )?;
                // Cross-arch fast path: fused gate+up in one launch. Works
                // for both MQ4 (x_rot Some) and HF4 (x_rot None → s.tmp).
                let dt_g = layer.w_gate.gpu_dtype;
                let fused_gu_ok = (dt_g == DType::MQ4G256 || dt_g == DType::HFQ4G256)
                    && layer.w_up.gpu_dtype == dt_g;
                if fused_gu_ok {
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_gate_up_hfq4g256(
                        &layer.w_gate.buf, &layer.w_up.buf,
                        eff_x,
                        &s.gate_ffn, &s.up,
                        layer.w_gate.m, layer.w_up.m,
                        layer.w_gate.k,
                    )?;
                } else {
                    weight_gemv_prerotated(gpu, &layer.w_gate, &s.tmp, x_rot, &s.gate_ffn)?;
                    weight_gemv_prerotated(gpu, &layer.w_up, &s.tmp, x_rot, &s.up)?;
                }
                // Fused SwiGLU + w_down residual GEMV:
                //   MQ4: fused_silu_rotate(gate,up) + gemv_residual(w_down, rotated, x)
                //   HF4: silu_mul + weight_gemv_residual (unchanged)
                weight_gemv_swiglu_residual(
                    gpu, &layer.w_down, &s.gate_ffn, &s.up, &s.ffn_hidden, &s.x,
                )?;

                if let Some(ref rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_at_head(gpu, slot, &s.x)?;
                    }
                }

                kv_layer_idx += 1;
            }

            _ => panic!("layer type mismatch at layer {layer_idx}"),
        }
    }
    let _ = &mut hidden_rb; // silence unused mut warning on paths where the branch never writes

    // Final norm + logits into scratch.logits
    gpu.rmsnorm_f32(&s.x, &weights.output_norm, &s.tmp, config.norm_eps)?;
    weight_gemv(gpu, &weights.output, &s.tmp, &s.logits)?;

    Ok(())
}

/// Forward pass returning logits ON GPU (no download). Caller must free the tensor.
/// Use with gpu.sample_top_p() after applying CPU-side n-gram blocking via download/modify/upload.
pub fn forward_gpu(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<GpuTensor> {
    let dim = config.dim;
    let x = gpu.alloc_tensor(&[dim], DType::F32)?;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &x, token, dim)?,
        _ => panic!("unsupported embedding format"),
    }
    forward_from_x_gpu(gpu, weights, config, x, pos, kv_cache, dn_state)
}

/// Run one step with a pre-computed embedding vector (for VL visual token injection).
/// embedding_data: [dim] F32 values on CPU — uploaded to GPU as the initial hidden state.
pub fn forward_with_embedding(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    embedding_data: &[f32],
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<Vec<f32>> {
    let x = gpu.upload_f32(embedding_data, &[config.dim])?;
    forward_from_x(gpu, weights, config, x, pos, kv_cache, dn_state)
}
