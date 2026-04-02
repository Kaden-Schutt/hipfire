//! Qwen3.5 model: hybrid DeltaNet (linear attention) + standard attention.
//! Feature-gated behind `deltanet`.

use crate::hfq::HfqFile;
use crate::llama::{self, f16_to_f32, EmbeddingFormat, WeightTensor, weight_gemv};
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
        6 | 7 => {
            // HFQ4-G256 or G128 — CPU dequant
            let group_size: usize = if info.quant_type == 6 { 256 } else { 128 };
            let bytes_per_group = 8 + group_size / 2;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale = f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
                let zero = f32::from_le_bytes([data[off+4], data[off+5], data[off+6], data[off+7]]);
                for i in 0..group_size {
                    let byte_idx = i / 2;
                    let byte_val = data[off + 8 + byte_idx];
                    let nibble = if i % 2 == 0 { byte_val & 0xF } else { byte_val >> 4 };
                    out.push(scale * nibble as f32 + zero);
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
        if embd_info.0.quant_type == 6 || embd_info.0.quant_type == 7 {
            let buf = gpu.upload_raw(embd_data, &[embd_data.len()])?;
            let dtype = if embd_info.0.quant_type == 6 { DType::HFQ4G256 } else { DType::HFQ4G128 };
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

                // Beta projection + sigmoid
                let n_v_heads = config.linear_num_value_heads;
                let beta_out = gpu.alloc_tensor(&[n_v_heads], DType::F32)?;
                weight_gemv(gpu, &layer.w_beta, &tmp, &beta_out)?;
                gpu.sigmoid_f32(&beta_out)?;

                // Alpha projection + gate compute (all on GPU, no CPU roundtrip)
                let alpha_out = gpu.alloc_tensor(&[n_v_heads], DType::F32)?;
                weight_gemv(gpu, &layer.w_alpha, &tmp, &alpha_out)?;
                gpu.alpha_gate_f32(&alpha_out, &layer.dt_bias, &layer.a_log, n_v_heads)?;

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

                // L2 normalize Q and K
                gpu.l2_norm_f32(&q_part, config.linear_num_key_heads, config.linear_key_head_dim, config.norm_eps)?;
                gpu.l2_norm_f32(&k_part, config.linear_num_key_heads, config.linear_key_head_dim, config.norm_eps)?;

                // Scale Q by 1/sqrt(S_k) before recurrence (all on GPU)
                gpu.scale_f32(&q_part, 1.0 / (config.linear_key_head_dim as f32).sqrt())?;

                // Repeat Q/K heads if num_k_heads < num_v_heads (GQA-style)
                let (q_gdn, k_gdn) = if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    let expanded_dim = n_v_heads * config.linear_key_head_dim;
                    let q_exp = gpu.alloc_tensor(&[expanded_dim], DType::F32)?;
                    let k_exp = gpu.alloc_tensor(&[expanded_dim], DType::F32)?;
                    let hd = config.linear_key_head_dim;
                    for kh in 0..config.linear_num_key_heads {
                        for r in 0..ratio {
                            let dst = (kh * ratio + r) * hd * 4;
                            let src = kh * hd * 4;
                            gpu.hip.memcpy_dtod_at(&q_exp.buf, dst, &q_part.buf, src, hd * 4)?;
                            gpu.hip.memcpy_dtod_at(&k_exp.buf, dst, &k_part.buf, src, hd * 4)?;
                        }
                    }
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
                // Extract interleaved: for each head, copy head_dim floats with stride 2*head_dim
                for h in 0..config.n_heads {
                    let src_q_off = h * config.head_dim * 2 * 4;
                    let src_g_off = (h * config.head_dim * 2 + config.head_dim) * 4;
                    let dst_off = h * config.head_dim * 4;
                    gpu.hip.memcpy_dtod_at(&q.buf, dst_off, &q_full.buf, src_q_off, config.head_dim * 4)?;
                    gpu.hip.memcpy_dtod_at(&gate_vec.buf, dst_off, &q_full.buf, src_g_off, config.head_dim * 4)?;
                }

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
        })
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

    forward_scratch_layers(gpu, weights, config, pos, kv_cache, dn_state, scratch)
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
    forward_scratch_layers(gpu, weights, config, pos, kv_cache, dn_state, scratch)
}

/// Layer loop using scratch buffers. Zero alloc/free per token.
fn forward_scratch_layers(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    s: &Qwen35Scratch,
) -> HipResult<()> {
    let dim = config.dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let qkv_dim = k_dim * 2 + v_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;

    let mut delta_layer_idx = 0usize;

    for layer_idx in 0..config.n_layers {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                gpu.rmsnorm_f32(&s.x, &layer.attn_norm, &s.tmp, config.norm_eps)?;
                weight_gemv(gpu, &layer.wqkv, &s.tmp, &s.dn_qkv)?;
                weight_gemv(gpu, &layer.wz, &s.tmp, &s.dn_z)?;
                weight_gemv(gpu, &layer.w_beta, &s.tmp, &s.dn_beta)?;
                gpu.sigmoid_f32(&s.dn_beta)?;
                weight_gemv(gpu, &layer.w_alpha, &s.tmp, &s.dn_alpha)?;
                gpu.alpha_gate_f32(&s.dn_alpha, &layer.dt_bias, &layer.a_log, n_v_heads)?;

                gpu.conv1d_silu_f32(&s.dn_conv_out, &s.dn_qkv, &layer.conv_weight,
                    &dn_state.conv_states[delta_layer_idx], qkv_dim)?;

                // Split conv output into Q, K, V
                gpu.hip.memcpy_dtod_at(&s.dn_q_raw.buf, 0, &s.dn_conv_out.buf, 0, k_dim * 4)?;
                gpu.hip.memcpy_dtod_at(&s.dn_k_raw.buf, 0, &s.dn_conv_out.buf, k_dim * 4, k_dim * 4)?;
                gpu.hip.memcpy_dtod_at(&s.dn_v.buf, 0, &s.dn_conv_out.buf, k_dim * 2 * 4, v_dim * 4)?;

                gpu.l2_norm_f32(&s.dn_q_raw, config.linear_num_key_heads, hd, config.norm_eps)?;
                gpu.l2_norm_f32(&s.dn_k_raw, config.linear_num_key_heads, hd, config.norm_eps)?;
                gpu.scale_f32(&s.dn_q_raw, 1.0 / (hd as f32).sqrt())?;

                // Repeat-interleave Q/K if needed
                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    for kh in 0..config.linear_num_key_heads {
                        for r in 0..ratio {
                            let dst = (kh * ratio + r) * hd * 4;
                            let src = kh * hd * 4;
                            gpu.hip.memcpy_dtod_at(&s.dn_q.buf, dst, &s.dn_q_raw.buf, src, hd * 4)?;
                            gpu.hip.memcpy_dtod_at(&s.dn_k.buf, dst, &s.dn_k_raw.buf, src, hd * 4)?;
                        }
                    }
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
                weight_gemv(gpu, &layer.wo, &s.dn_normed, &s.o)?;
                gpu.add_inplace_f32(&s.x, &s.o)?;

                // FFN
                gpu.rmsnorm_f32(&s.x, &layer.ffn_norm, &s.tmp, config.norm_eps)?;
                weight_gemv(gpu, &layer.w_gate, &s.tmp, &s.gate_ffn)?;
                weight_gemv(gpu, &layer.w_up, &s.tmp, &s.up)?;
                gpu.silu_mul_f32(&s.gate_ffn, &s.up, &s.ffn_hidden)?;
                weight_gemv(gpu, &layer.w_down, &s.ffn_hidden, &s.ffn_out)?;
                gpu.add_inplace_f32(&s.x, &s.ffn_out)?;

                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttn(layer), LayerType::FullAttention) => {
                gpu.rmsnorm_f32(&s.x, &layer.attn_norm, &s.tmp, config.norm_eps)?;
                weight_gemv(gpu, &layer.wq, &s.tmp, &s.fa_q_full)?;

                // Split interleaved Q+gate
                for h in 0..config.n_heads {
                    let src_q = h * config.head_dim * 2 * 4;
                    let src_g = (h * config.head_dim * 2 + config.head_dim) * 4;
                    let dst = h * config.head_dim * 4;
                    gpu.hip.memcpy_dtod_at(&s.fa_q.buf, dst, &s.fa_q_full.buf, src_q, config.head_dim * 4)?;
                    gpu.hip.memcpy_dtod_at(&s.fa_gate.buf, dst, &s.fa_q_full.buf, src_g, config.head_dim * 4)?;
                }

                gpu.rmsnorm_batched(&s.fa_q, &layer.q_norm, &s.fa_q, config.n_heads, config.head_dim, config.norm_eps)?;

                let kv_dim = config.n_kv_heads * config.head_dim;
                weight_gemv(gpu, &layer.wk, &s.tmp, &s.fa_k)?;
                weight_gemv(gpu, &layer.wv, &s.tmp, &s.fa_v)?;
                gpu.rmsnorm_batched(&s.fa_k, &layer.k_norm, &s.fa_k, config.n_kv_heads, config.head_dim, config.norm_eps)?;

                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                gpu.rope_partial_interleaved_f32(&s.fa_q, &s.fa_k, pos as i32,
                    config.n_heads, config.n_kv_heads, config.head_dim, n_rot, config.rope_theta)?;

                if kv_cache.quant_asym {
                    // Asymmetric: Q8 K + turbo4 V (V compression is free)
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

                gpu.sigmoid_f32(&s.fa_gate)?;
                gpu.mul_f32(&s.fa_attn_out, &s.fa_gate, &s.fa_attn_out)?;
                weight_gemv(gpu, &layer.wo, &s.fa_attn_out, &s.o)?;
                gpu.add_inplace_f32(&s.x, &s.o)?;

                // FFN
                gpu.rmsnorm_f32(&s.x, &layer.ffn_norm, &s.tmp, config.norm_eps)?;
                weight_gemv(gpu, &layer.w_gate, &s.tmp, &s.gate_ffn)?;
                weight_gemv(gpu, &layer.w_up, &s.tmp, &s.up)?;
                gpu.silu_mul_f32(&s.gate_ffn, &s.up, &s.ffn_hidden)?;
                weight_gemv(gpu, &layer.w_down, &s.ffn_hidden, &s.ffn_out)?;
                gpu.add_inplace_f32(&s.x, &s.ffn_out)?;
            }

            _ => panic!("layer type mismatch at layer {layer_idx}"),
        }
    }

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
