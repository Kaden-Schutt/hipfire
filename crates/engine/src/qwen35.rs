//! Qwen3.5 model: hybrid DeltaNet (linear attention) + standard attention.
//! Feature-gated behind `deltanet`.

use crate::hfq::HfqFile;
use crate::llama::{self, f16_to_f32, EmbeddingFormat, WeightTensor, weight_gemv,
                    weight_gemv_prerotated, fused_rmsnorm_rotate_for_mq,
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

    // FFN — dense; for MoE see num_experts below
    pub hidden_dim: usize,     // 3584 (dense) or unused when num_experts > 0

    // MoE (qwen3_5_moe / A3B). num_experts == 0 means plain dense (qwen3_5).
    pub num_experts: usize,                      // 256 for A3B
    pub num_experts_per_tok: usize,              // 8 for A3B
    pub moe_intermediate_size: usize,            // 512 for A3B (per-routed-expert FFN)
    pub shared_expert_intermediate_size: usize,  // 512 for A3B
    pub has_shared_expert: bool,                 // true for A3B (always-on shared expert)
    /// If true, top-K routing weights are re-normalized to sum to 1 after
    /// softmax + top-K selection. Qwen convention (matches HF
    /// `modeling_qwen3_5_moe.py`). DeepSeek-v1 uses false.
    pub norm_topk_prob: bool,

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
    // Dense FFN intermediate dim. MoE configs (qwen3_5_moe / A3B) replace this
    // with `moe_intermediate_size` and don't ship `intermediate_size`, so don't
    // hard-fail here — we still need to load the rest of the config to detect
    // is_moe and route accordingly.
    let hidden_dim = tc.get("intermediate_size").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
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

    // MoE config (zeros = dense fallback). Qwen3.5-MoE / A3B sets these.
    let num_experts = tc.get("num_experts").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let num_experts_per_tok = tc.get("num_experts_per_tok").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let moe_intermediate_size = tc.get("moe_intermediate_size").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let shared_expert_intermediate_size = tc.get("shared_expert_intermediate_size").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let has_shared_expert = shared_expert_intermediate_size > 0;
    // Qwen convention: re-normalize top-K routing weights to sum to 1.
    // Absent from some configs (including the shipped A3B HFQ); default on
    // for Qwen3.5-MoE / A3B to match the HF reference.
    let norm_topk_prob = tc.get("norm_topk_prob").and_then(|v| v.as_bool()).unwrap_or(true);

    Some(Qwen35Config {
        dim, n_layers, vocab_size, norm_eps, eos_token,
        n_heads, n_kv_heads, head_dim, rope_theta, partial_rotary_factor,
        linear_num_key_heads, linear_num_value_heads, linear_key_head_dim, linear_value_head_dim, conv_kernel_dim,
        hidden_dim, layer_types,
        num_experts, num_experts_per_tok, moe_intermediate_size, shared_expert_intermediate_size, has_shared_expert,
        norm_topk_prob,
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

// ─── MoE FFN weights (Qwen3.5-MoE / A3B) ────────────────────────────────
//
// Replaces the dense (w_gate, w_up, w_down) triple with N+1 expert FFNs
// gated by a router, plus a shared always-on expert.
//
// A3B specifics:
//   num_experts = 256, top_k = 8, moe_intermediate = 512, hidden = 2048
//   shared_expert_intermediate = 512 (same as routed)
//
// Per-layer storage:
//   router:               [num_experts, hidden]  MQ4G256 / Q8
//   shared_expert_gate:   [1, hidden]            MQ4G256 / Q8 — projects to scalar
//   experts[X].gate_up:   [2*moe_intermediate, hidden]  MQ4G256
//   experts[X].down:      [hidden, moe_intermediate]    MQ4G256
//   shared_expert.gate:   [shared_expert_intermediate, hidden]   MQ4G256
//   shared_expert.up:     [shared_expert_intermediate, hidden]   MQ4G256
//   shared_expert.down:   [hidden, shared_expert_intermediate]   MQ4G256
//
// The quantizer (hipfire-quantize) splits the safetensors 3D
// `mlp.experts.gate_up_proj` / `down_proj` tensors per-expert into
// `mlp.experts.{X}.gate_up_proj.weight` / `down_proj.weight` so the loader
// can fish them out by index. The shared expert is stored with separate
// gate_proj + up_proj + down_proj (it is not fused in safetensors either).

pub struct ExpertWeights {
    pub gate_up: WeightTensor,  // [2 * moe_intermediate, hidden] — fused (gate || up)
    pub down: WeightTensor,     // [hidden, moe_intermediate]
}

/// Shared expert storage — unlike routed experts, gate_proj and up_proj are
/// NOT fused in the safetensors, so we keep them separate here too. The
/// forward path does two GEMVs + silu_mul + down GEMV.
pub struct SharedExpertWeights {
    pub gate: WeightTensor,  // [shared_expert_intermediate, hidden]
    pub up: WeightTensor,    // [shared_expert_intermediate, hidden]
    pub down: WeightTensor,  // [hidden, shared_expert_intermediate]
}

pub struct MoeFfnWeights {
    pub router: WeightTensor,                 // [num_experts, hidden]
    pub experts: Vec<ExpertWeights>,          // num_experts (= 256 for A3B)
    pub shared_expert: SharedExpertWeights,
    pub shared_expert_gate: WeightTensor,     // [1, hidden] — row-vector projecting to scalar
}

pub struct DeltaNetMoeLayerWeights {
    pub attn_norm: GpuTensor,
    pub wqkv: WeightTensor,
    pub wz: WeightTensor,
    pub w_alpha: WeightTensor,
    pub w_beta: WeightTensor,
    pub a_log: GpuTensor,
    pub dt_bias: GpuTensor,
    pub conv_weight: GpuTensor,
    pub norm_weight: GpuTensor,
    pub wo: WeightTensor,
    pub ffn_norm: GpuTensor,
    pub ffn: MoeFfnWeights,
}

pub struct FullAttnMoeLayerWeights {
    pub attn_norm: GpuTensor,
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,
    pub wo: WeightTensor,
    pub q_norm: GpuTensor,
    pub k_norm: GpuTensor,
    pub ffn_norm: GpuTensor,
    pub ffn: MoeFfnWeights,
}

pub enum LayerWeights {
    DeltaNet(DeltaNetLayerWeights),
    FullAttn(FullAttnLayerWeights),
    // A3B / qwen3_5_moe: same attention as above, MoE FFN instead of dense.
    // Loader + forward path TODO — adding the variants now so the enum is
    // forward-compatible and downstream code that pattern-matches gets a
    // compile-time hint to handle the new case.
    DeltaNetMoe(DeltaNetMoeLayerWeights),
    FullAttnMoe(FullAttnMoeLayerWeights),
}

pub struct Qwen35Weights {
    pub token_embd: GpuTensor,
    pub embd_format: EmbeddingFormat,
    pub output_norm: GpuTensor,
    pub output: WeightTensor,
    pub layers: Vec<LayerWeights>,
}

impl Qwen35Weights {
    /// Return all GPU buffers to the pool (drained on unload). Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.token_embd);
        let _ = gpu.free_tensor(self.output_norm);
        let _ = gpu.free_tensor(self.output.buf);
        for layer in self.layers {
            match layer {
                LayerWeights::DeltaNet(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    let _ = gpu.free_tensor(l.wqkv.buf);
                    let _ = gpu.free_tensor(l.wz.buf);
                    let _ = gpu.free_tensor(l.w_alpha.buf);
                    let _ = gpu.free_tensor(l.w_beta.buf);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    let _ = gpu.free_tensor(l.wo.buf);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    let _ = gpu.free_tensor(l.w_gate.buf);
                    let _ = gpu.free_tensor(l.w_up.buf);
                    let _ = gpu.free_tensor(l.w_down.buf);
                }
                LayerWeights::FullAttn(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    let _ = gpu.free_tensor(l.wq.buf);
                    let _ = gpu.free_tensor(l.wk.buf);
                    let _ = gpu.free_tensor(l.wv.buf);
                    let _ = gpu.free_tensor(l.wo.buf);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    let _ = gpu.free_tensor(l.w_gate.buf);
                    let _ = gpu.free_tensor(l.w_up.buf);
                    let _ = gpu.free_tensor(l.w_down.buf);
                }
                LayerWeights::DeltaNetMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    let _ = gpu.free_tensor(l.wqkv.buf);
                    let _ = gpu.free_tensor(l.wz.buf);
                    let _ = gpu.free_tensor(l.w_alpha.buf);
                    let _ = gpu.free_tensor(l.w_beta.buf);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    let _ = gpu.free_tensor(l.wo.buf);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_ffn(gpu, l.ffn);
                }
                LayerWeights::FullAttnMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    let _ = gpu.free_tensor(l.wq.buf);
                    let _ = gpu.free_tensor(l.wk.buf);
                    let _ = gpu.free_tensor(l.wv.buf);
                    let _ = gpu.free_tensor(l.wo.buf);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_ffn(gpu, l.ffn);
                }
            }
        }
    }
}

fn free_moe_ffn(gpu: &mut Gpu, ffn: MoeFfnWeights) {
    let _ = gpu.free_tensor(ffn.router.buf);
    let _ = gpu.free_tensor(ffn.shared_expert_gate.buf);
    let _ = gpu.free_tensor(ffn.shared_expert.gate.buf);
    let _ = gpu.free_tensor(ffn.shared_expert.up.buf);
    let _ = gpu.free_tensor(ffn.shared_expert.down.buf);
    for e in ffn.experts {
        let _ = gpu.free_tensor(e.gate_up.buf);
        let _ = gpu.free_tensor(e.down.buf);
    }
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
        15 => { // MQ6-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::MQ6G256, m, k, row_stride: 0 })
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
        15 => { // MQ6-G256 — MagnumQuant FWHT-rotated 6-bit
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::MQ6G256, m, k, row_stride: 0 })
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
        6 | 7 | 13 | 15 => {
            // HFQ4-G256 or G128 or MQ4-G256 or MQ6-G256 — CPU dequant
            // MQ4/MQ6 store rotated weights. For small tensors loaded here,
            // we dequant then inverse-rotate to recover the original values.
            let is_6bit = info.quant_type == 15;
            let group_size: usize = if info.quant_type == 6 || info.quant_type == 13 || info.quant_type == 15 { 256 } else { 128 };
            let bytes_per_group = if is_6bit { 200 } else { 8 + group_size / 2 };
            let n_groups = data.len() / bytes_per_group;
            let is_mq = info.quant_type == 13 || info.quant_type == 15;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let (signs1, signs2) = if is_mq {
                (Some(crate::llama::KvCache::gen_fwht_signs(42, 256)),
                 Some(crate::llama::KvCache::gen_fwht_signs(1042, 256)))
            } else { (None, None) };
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale = f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
                let zero = f32::from_le_bytes([data[off+4], data[off+5], data[off+6], data[off+7]]);
                let start = out.len();
                if is_6bit {
                    for i in (0..group_size).step_by(4) {
                        let bo = off + 8 + (i / 4) * 3;
                        let b0 = data[bo] as u32;
                        let b1 = data[bo + 1] as u32;
                        let b2 = data[bo + 2] as u32;
                        out.push(scale * ((b0 & 0x3F) as f32) + zero);
                        out.push(scale * ((((b0 >> 6) | (b1 << 2)) & 0x3F) as f32) + zero);
                        out.push(scale * ((((b1 >> 4) | (b2 << 4)) & 0x3F) as f32) + zero);
                        out.push(scale * (((b2 >> 2) & 0x3F) as f32) + zero);
                    }
                } else {
                    for i in 0..group_size {
                        let byte_idx = i / 2;
                        let byte_val = data[off + 8 + byte_idx];
                        let nibble = if i % 2 == 0 { byte_val & 0xF } else { byte_val >> 4 };
                        out.push(scale * nibble as f32 + zero);
                    }
                }
                // Inverse FWHT for MQ4/MQ6: recover original weight values
                if is_mq && group_size == 256 {
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

    let is_moe = config.num_experts > 0;
    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        eprintln!("  loading layer {i}/{} ({:?}{})...",
            config.n_layers, config.layer_types[i],
            if is_moe { " + MoE" } else { "" });
        let p = format!("layers.{i}");

        match (config.layer_types[i], is_moe) {
            (LayerType::LinearAttention, false) => {
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
            (LayerType::FullAttention, false) => {
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
            (LayerType::LinearAttention, true) => {
                let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
                            + config.linear_num_value_heads * config.linear_value_head_dim;
                let d_inner = config.linear_num_value_heads * config.linear_value_head_dim;

                layers.push(LayerWeights::DeltaNetMoe(DeltaNetMoeLayerWeights {
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
                        qkv_dim * config.conv_kernel_dim)?,
                    norm_weight: load_any_as_f32(hfq, gpu, &format!("{p}.linear_attn.norm.weight"), config.linear_value_head_dim)?,
                    wo: load_weight_tensor(hfq, gpu, &format!("{p}.linear_attn.out_proj.weight"), config.dim, d_inner)?,
                    ffn_norm: load_norm_weight(hfq, gpu, &format!("{p}.post_attention_layernorm.weight"), &[config.dim])?,
                    ffn: load_moe_ffn(hfq, gpu, &p, config)?,
                }));
            }
            (LayerType::FullAttention, true) => {
                let q_out_dim = config.n_heads * config.head_dim * 2;
                let kv_dim = config.n_kv_heads * config.head_dim;

                layers.push(LayerWeights::FullAttnMoe(FullAttnMoeLayerWeights {
                    attn_norm: load_norm_weight(hfq, gpu, &format!("{p}.input_layernorm.weight"), &[config.dim])?,
                    wq: load_weight_tensor(hfq, gpu, &format!("{p}.self_attn.q_proj.weight"), q_out_dim, config.dim)?,
                    wk: load_weight_tensor(hfq, gpu, &format!("{p}.self_attn.k_proj.weight"), kv_dim, config.dim)?,
                    wv: load_weight_tensor(hfq, gpu, &format!("{p}.self_attn.v_proj.weight"), kv_dim, config.dim)?,
                    wo: load_weight_tensor(hfq, gpu, &format!("{p}.self_attn.o_proj.weight"), config.dim, config.n_heads * config.head_dim)?,
                    q_norm: load_norm_weight(hfq, gpu, &format!("{p}.self_attn.q_norm.weight"), &[config.head_dim])?,
                    k_norm: load_norm_weight(hfq, gpu, &format!("{p}.self_attn.k_norm.weight"), &[config.head_dim])?,
                    ffn_norm: load_norm_weight(hfq, gpu, &format!("{p}.post_attention_layernorm.weight"), &[config.dim])?,
                    ffn: load_moe_ffn(hfq, gpu, &p, config)?,
                }));
            }
        }
    }

    Ok(Qwen35Weights { token_embd, embd_format: embd_fmt, output_norm, output, layers })
}

/// Load one layer's full MoE FFN block: router, all routed experts, shared expert,
/// and the per-layer scalar shared-expert gate. Tensor naming follows what the
/// quantizer emits for qwen3_5_moe (commit 4860575): the 3D stacked-expert source
/// tensors get split per-expert into `mlp.experts.{X}.{base}.weight`.
fn load_moe_ffn(hfq: &HfqFile, gpu: &Gpu, p: &str, config: &Qwen35Config) -> HipResult<MoeFfnWeights> {
    let n_exp = config.num_experts;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;

    // Router: hidden_size → num_experts. Precision-sensitive but small.
    let router = load_weight_tensor(hfq, gpu, &format!("{p}.mlp.gate.weight"), n_exp, config.dim)?;

    // Shared expert (always-on, contributes to every token). Unlike routed
    // experts, gate_proj + up_proj are stored separately in the safetensors
    // (routed experts store them fused as `gate_up_proj`).
    let shared_expert = SharedExpertWeights {
        gate: load_weight_tensor(hfq, gpu, &format!("{p}.mlp.shared_expert.gate_proj.weight"), smi, config.dim)?,
        up:   load_weight_tensor(hfq, gpu, &format!("{p}.mlp.shared_expert.up_proj.weight"),   smi, config.dim)?,
        down: load_weight_tensor(hfq, gpu, &format!("{p}.mlp.shared_expert.down_proj.weight"), config.dim, smi)?,
    };
    // Scalar gate on the shared-expert add: sigmoid(shared_expert_gate · x).
    // Stored as a 1×hidden row-vector.
    let shared_expert_gate = load_weight_tensor(hfq, gpu, &format!("{p}.mlp.shared_expert_gate.weight"), 1, config.dim)?;

    // Routed experts — quantizer wrote per-expert tensors named
    // `{p}.mlp.experts.{X}.gate_up_proj.weight` (shape [2*moe_intermediate, hidden_size])
    // and `{p}.mlp.experts.{X}.down_proj.weight` (shape [hidden_size, moe_intermediate]).
    let mut experts = Vec::with_capacity(n_exp);
    for x in 0..n_exp {
        let gate_up = load_weight_tensor(hfq, gpu,
            &format!("{p}.mlp.experts.{x}.gate_up_proj.weight"),
            2 * mi, config.dim)?;
        let down = load_weight_tensor(hfq, gpu,
            &format!("{p}.mlp.experts.{x}.down_proj.weight"),
            config.dim, mi)?;
        experts.push(ExpertWeights { gate_up, down });
    }

    Ok(MoeFfnWeights { router, experts, shared_expert, shared_expert_gate })
}

// ─── MoE FFN (decode, batch=1) ──────────────────────────────────────────

/// Construct a non-owning `GpuTensor` view over `[offset_elems,
/// offset_elems + len_elems)` of `src`. Valid only for F32 (4 bytes/elem).
/// The view MUST NOT outlive `src` — it shares the same GPU pointer.
#[inline]
fn slice_f32_view(src: &GpuTensor, offset_elems: usize, len_elems: usize) -> GpuTensor {
    unsafe {
        let base = src.buf.as_ptr() as *mut u8;
        let ptr = base.add(offset_elems * 4);
        GpuTensor {
            buf: hip_bridge::DeviceBuffer::from_raw(ptr as *mut _, len_elems * 4),
            shape: vec![len_elems],
            dtype: DType::F32,
        }
    }
}

/// One-token MoE FFN: router → top-K → shared expert + top-K routed, added
/// into `x_residual` in place. `x_norm` is the already-RMSNormed FFN input.
///
/// Dense-compute decode reference implementation (Phase 1). Top-K selection
/// runs on CPU via a single D2H sync per layer on the router logits; the
/// shared-expert scalar gate is another D2H sync. Sparse-routing + batched
/// grouped-GEMM variants come in later phases — this version prioritizes
/// correctness and minimal surface area.
///
/// Matches HF `modeling_qwen3_5_moe.py`:
///   router_probs  = softmax(W_router · x_norm)            // [n_exp]
///   (idx, w)      = topk(router_probs, k)                  // [k]
///   if norm_topk:  w /= w.sum()
///   scalar        = sigmoid(W_shared_gate · x_norm)        // [1]
///   y_shared      = scalar * shared_expert(x_norm)         // [hidden]
///   y_moe         = sum_{k} w[k] * expert[idx[k]](x_norm)  // [hidden]
///   x_residual   += y_shared + y_moe
fn moe_ffn_decode(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x_norm: &GpuTensor,
    x_residual: &GpuTensor,
    config: &Qwen35Config,
) -> HipResult<()> {
    let hidden = config.dim;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let k = config.num_experts_per_tok;
    let n_exp = config.num_experts;
    let max_inter = mi.max(smi);

    // Scratch buffers — allocated once per layer, reused across experts.
    let router_logits = gpu.alloc_tensor(&[n_exp], DType::F32)?;
    let gate_up_buf   = gpu.alloc_tensor(&[2 * max_inter], DType::F32)?;
    let gate_buf      = gpu.alloc_tensor(&[max_inter], DType::F32)?;
    let up_buf        = gpu.alloc_tensor(&[max_inter], DType::F32)?;
    let ffn_hidden    = gpu.alloc_tensor(&[max_inter], DType::F32)?;
    let ffn_out       = gpu.alloc_tensor(&[hidden], DType::F32)?;
    let scalar_buf    = gpu.alloc_tensor(&[1], DType::F32)?;

    // Phase 2a-iii: rotate x_norm once per layer and share the rotated
    // buffer across every gate-side GEMV (router, shared_expert_gate,
    // shared.gate, shared.up, and all top-K experts' gate_up). Each
    // weight_gemv on MQ4G256 otherwise does its own rotate_x_mq launch
    // — 12+ redundant rotations per layer. Only MQ4 GEMVs benefit, so
    // we detect the all-MQ4 fast path and fall back to weight_gemv
    // (which rotates internally) for any mixed-dtype layer.
    let gate_side_mq4 = ffn.router.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert_gate.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert.gate.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert.up.gpu_dtype == DType::MQ4G256
        && ffn.experts.iter().all(|e| e.gate_up.gpu_dtype == DType::MQ4G256);
    let x_rot_local = if gate_side_mq4 {
        let buf = gpu.alloc_tensor(&[hidden], DType::F32)?;
        gpu.ensure_mq_signs()?;
        gpu.rotate_x_mq(x_norm, &buf, hidden)?;
        Some(buf)
    } else {
        None
    };

    // ── 1. Router logits → softmax → CPU top-K + (optional) renormalize ──
    if let Some(ref xr) = x_rot_local {
        gpu.gemv_mq4g256_prerotated(&ffn.router.buf, xr, &router_logits, ffn.router.m, ffn.router.k)?;
    } else {
        weight_gemv(gpu, &ffn.router, x_norm, &router_logits)?;
    }
    gpu.softmax_f32(&router_logits)?;
    let probs = gpu.download_f32(&router_logits)?;

    // Partial sort: select top-K indices by probability. `select_nth_unstable_by`
    // places the k-1-th element in its sorted position with O(n) average work.
    let mut indices: Vec<usize> = (0..n_exp).collect();
    indices.select_nth_unstable_by(k - 1, |&a, &b| {
        probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut topk_indices: Vec<usize> = indices.into_iter().take(k).collect();
    // Sort top-K by probability (descending) for deterministic accumulation order.
    topk_indices.sort_by(|&a, &b| {
        probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut topk_weights: Vec<f32> = topk_indices.iter().map(|&i| probs[i]).collect();
    if config.norm_topk_prob {
        let sum: f32 = topk_weights.iter().sum();
        if sum > 0.0 {
            for w in topk_weights.iter_mut() {
                *w /= sum;
            }
        }
    }

    // ── 2. Shared-expert scalar gate: sigmoid(W · x_norm) — stays on GPU ──
    // Phase 2a: compute the sigmoid on-device and pass the result by
    // device pointer into the fused scaled-add below. Eliminates one
    // D2H sync per layer (was 80 syncs per token across 40 layers).
    if let Some(ref xr) = x_rot_local {
        gpu.gemv_mq4g256_prerotated(&ffn.shared_expert_gate.buf, xr, &scalar_buf,
            ffn.shared_expert_gate.m, ffn.shared_expert_gate.k)?;
    } else {
        weight_gemv(gpu, &ffn.shared_expert_gate, x_norm, &scalar_buf)?;
    }
    gpu.sigmoid_f32(&scalar_buf)?;

    // ── 3. Shared expert FFN (gate/up/down) scaled by the sigmoid gate ──
    // Phase 2a-ii: fuse silu_mul + rotate + down-GEMV + scaled-residual-add
    // into two launches (fused_silu_mul_rotate_mq + gemv_residual_scaled_gpu)
    // when down weights are MQ4G256. Eliminates the separate silu_mul,
    // explicit scale_f32, and add_inplace_f32 launches — one kernel does
    // the full `y += sigmoid(gate_scalar) * W_down · silu_mul(gate, up)`.
    let shared_gate = slice_f32_view(&gate_buf, 0, smi);
    let shared_up   = slice_f32_view(&up_buf,   0, smi);
    if let Some(ref xr) = x_rot_local {
        gpu.gemv_mq4g256_prerotated(&ffn.shared_expert.gate.buf, xr, &shared_gate,
            ffn.shared_expert.gate.m, ffn.shared_expert.gate.k)?;
        gpu.gemv_mq4g256_prerotated(&ffn.shared_expert.up.buf, xr, &shared_up,
            ffn.shared_expert.up.m, ffn.shared_expert.up.k)?;
    } else {
        weight_gemv(gpu, &ffn.shared_expert.gate, x_norm, &shared_gate)?;
        weight_gemv(gpu, &ffn.shared_expert.up,   x_norm, &shared_up)?;
    }
    if ffn.shared_expert.down.gpu_dtype == DType::MQ4G256 {
        gpu.ensure_mq_signs()?;
        let x_rot_alias = GpuTensor {
            buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
            shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
            dtype: DType::F32,
        };
        gpu.fused_silu_mul_rotate_mq(&shared_gate, &shared_up, &x_rot_alias, smi)?;
        gpu.gemv_hfq4g256_residual_scaled_gpu(
            &ffn.shared_expert.down.buf, &x_rot_alias, x_residual, &scalar_buf,
            ffn.shared_expert.down.m, ffn.shared_expert.down.k,
        )?;
    } else {
        // Non-MQ fallback: pre-2a-ii path.
        let shared_hid = slice_f32_view(&ffn_hidden, 0, smi);
        gpu.silu_mul_f32(&shared_gate, &shared_up, &shared_hid)?;
        weight_gemv(gpu, &ffn.shared_expert.down, &shared_hid, &ffn_out)?;
        gpu.scaled_add_inplace_gpu_scalar_f32(x_residual, &ffn_out, &scalar_buf)?;
    }

    // ── 4. Top-K routed experts ──
    // Phase 2c step 1: when top_k==8 and all selected experts' gate_up are
    // MQ4G256 (true for A3B), fuse the 8 gate_up GEMVs into a single
    // kernel launch via gemv_hfq4g256_moe_gate_up_k8 — grid.y selects the
    // expert via a kernarg pointer array. Output is a [k × 2*mi] batch.
    //
    // The silu_mul+rotate and down+scaled_residual stages are still
    // per-expert launches for now (Phase 2c step 2 will batch them too).
    let routed_mq4 = ffn.experts.first()
        .map(|e| e.down.gpu_dtype == DType::MQ4G256)
        .unwrap_or(false);
    let routed_gate_up_mq4 = ffn.experts.first()
        .map(|e| e.gate_up.gpu_dtype == DType::MQ4G256)
        .unwrap_or(false);
    if routed_mq4 {
        gpu.ensure_mq_signs()?;
    }

    // Scratch batch output for all k experts' gate_up results: [k × 2*mi].
    let use_fused_gate_up = k == 8 && routed_gate_up_mq4 && x_rot_local.is_some();
    let gate_up_batch = if use_fused_gate_up {
        Some(gpu.alloc_tensor(&[k * 2 * mi], DType::F32)?)
    } else {
        None
    };
    if let (Some(batch_out), Some(xr)) = (gate_up_batch.as_ref(), x_rot_local.as_ref()) {
        // Single fused launch: 8 experts' gate_up GEMV in parallel.
        let e0 = &ffn.experts[topk_indices[0]].gate_up;
        let e1 = &ffn.experts[topk_indices[1]].gate_up;
        let e2 = &ffn.experts[topk_indices[2]].gate_up;
        let e3 = &ffn.experts[topk_indices[3]].gate_up;
        let e4 = &ffn.experts[topk_indices[4]].gate_up;
        let e5 = &ffn.experts[topk_indices[5]].gate_up;
        let e6 = &ffn.experts[topk_indices[6]].gate_up;
        let e7 = &ffn.experts[topk_indices[7]].gate_up;
        gpu.gemv_hfq4g256_moe_gate_up_k8(
            &e0.buf, &e1.buf, &e2.buf, &e3.buf,
            &e4.buf, &e5.buf, &e6.buf, &e7.buf,
            xr, batch_out,
            2 * mi, e0.k,
        )?;
    }

    for (krank, (&expert_idx, &weight)) in topk_indices.iter().zip(topk_weights.iter()).enumerate() {
        let expert = &ffn.experts[expert_idx];
        // Source buffer for silu_mul: either the fused batch output slice
        // or a fresh per-expert scratch from an unfused gate_up GEMV.
        let (gate_view, up_view) = if let Some(ref batch) = gate_up_batch {
            let base = krank * 2 * mi;
            (slice_f32_view(batch, base, mi), slice_f32_view(batch, base + mi, mi))
        } else {
            if let Some(ref xr) = x_rot_local {
                gpu.gemv_mq4g256_prerotated(&expert.gate_up.buf, xr, &gate_up_buf,
                    expert.gate_up.m, expert.gate_up.k)?;
            } else {
                weight_gemv(gpu, &expert.gate_up, x_norm, &gate_up_buf)?;
            }
            (slice_f32_view(&gate_up_buf, 0, mi), slice_f32_view(&gate_up_buf, mi, mi))
        };
        if routed_mq4 {
            let x_rot_alias = GpuTensor {
                buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            };
            gpu.fused_silu_mul_rotate_mq(&gate_view, &up_view, &x_rot_alias, mi)?;
            gpu.gemv_hfq4g256_residual_scaled_cpu(
                &expert.down.buf, &x_rot_alias, x_residual, weight,
                expert.down.m, expert.down.k,
            )?;
        } else {
            // Non-MQ fallback: pre-2a-ii path.
            let hid_view = slice_f32_view(&ffn_hidden, 0, mi);
            gpu.silu_mul_f32(&gate_view, &up_view, &hid_view)?;
            weight_gemv(gpu, &expert.down, &hid_view, &ffn_out)?;
            gpu.scaled_add_inplace_cpu_scalar_f32(x_residual, &ffn_out, weight)?;
        }
    }

    // Free scratch
    for t in [router_logits, gate_up_buf, gate_buf, up_buf, ffn_hidden, ffn_out, scalar_buf] {
        gpu.free_tensor(t)?;
    }
    if let Some(buf) = x_rot_local {
        gpu.free_tensor(buf)?;
    }
    if let Some(buf) = gate_up_batch {
        gpu.free_tensor(buf)?;
    }
    Ok(())
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
                gpu.rope_partial_interleaved_f32(&q, &k, &pos_buf,
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

            // ── MoE variants (Qwen3.5-MoE / A3B) ──
            // Attention is byte-identical to the dense variant above; only
            // the FFN differs (router + top-K + shared + routed experts).
            (LayerWeights::DeltaNetMoe(layer), LayerType::LinearAttention) => {
                // ── DeltaNet attention (same as dense) ──
                gpu.rmsnorm_f32(&x, &layer.attn_norm, &tmp, config.norm_eps)?;

                let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
                             + config.linear_num_value_heads * config.linear_value_head_dim;
                let qkv = gpu.alloc_tensor(&[qkv_dim], DType::F32)?;
                weight_gemv(gpu, &layer.wqkv, &tmp, &qkv)?;

                let d_inner = config.linear_num_value_heads * config.linear_value_head_dim;
                let z = gpu.alloc_tensor(&[d_inner], DType::F32)?;
                weight_gemv(gpu, &layer.wz, &tmp, &z)?;

                let n_v_heads = config.linear_num_value_heads;
                let beta_out = gpu.alloc_tensor(&[n_v_heads], DType::F32)?;
                weight_gemv(gpu, &layer.w_beta, &tmp, &beta_out)?;
                let alpha_out = gpu.alloc_tensor(&[n_v_heads], DType::F32)?;
                weight_gemv(gpu, &layer.w_alpha, &tmp, &alpha_out)?;
                gpu.fused_sigmoid_alpha_gate_f32(
                    &beta_out, &alpha_out, &layer.dt_bias, &layer.a_log, n_v_heads,
                )?;

                let conv_out = gpu.alloc_tensor(&[qkv_dim], DType::F32)?;
                gpu.conv1d_silu_f32(
                    &conv_out, &qkv, &layer.conv_weight,
                    &dn_state.conv_states[delta_layer_idx], qkv_dim,
                )?;

                let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
                let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
                let q_part = gpu.alloc_tensor(&[k_dim], DType::F32)?;
                let k_part = gpu.alloc_tensor(&[k_dim], DType::F32)?;
                let v_part = gpu.alloc_tensor(&[v_dim], DType::F32)?;
                gpu.hip.memcpy_dtod_at(&q_part.buf, 0, &conv_out.buf, 0, k_dim * 4)?;
                gpu.hip.memcpy_dtod_at(&k_part.buf, 0, &conv_out.buf, k_dim * 4, k_dim * 4)?;
                gpu.hip.memcpy_dtod_at(&v_part.buf, 0, &conv_out.buf, k_dim * 2 * 4, v_dim * 4)?;

                gpu.fused_qk_l2_norm_scale_f32(
                    &q_part, &k_part,
                    config.linear_num_key_heads, config.linear_key_head_dim,
                    1.0 / (config.linear_key_head_dim as f32).sqrt(),
                    config.norm_eps,
                )?;

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
                    let q_ref = gpu.alloc_tensor(&[k_dim], DType::F32)?;
                    let k_ref = gpu.alloc_tensor(&[k_dim], DType::F32)?;
                    gpu.hip.memcpy_dtod_at(&q_ref.buf, 0, &q_part.buf, 0, k_dim * 4)?;
                    gpu.hip.memcpy_dtod_at(&k_ref.buf, 0, &k_part.buf, 0, k_dim * 4)?;
                    (q_ref, k_ref)
                };

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

                let normed_out = gpu.alloc_tensor(&[v_dim], DType::F32)?;
                gpu.gated_norm_f32(&attn_out, &z, &layer.norm_weight, &normed_out,
                    n_v_heads, config.linear_value_head_dim, config.norm_eps)?;

                let o = gpu.alloc_tensor(&[dim], DType::F32)?;
                weight_gemv(gpu, &layer.wo, &normed_out, &o)?;

                gpu.add_inplace_f32(&x, &o)?;

                // ── MoE FFN (only difference from dense) ──
                gpu.rmsnorm_f32(&x, &layer.ffn_norm, &tmp, config.norm_eps)?;
                moe_ffn_decode(gpu, &layer.ffn, &tmp, &x, config)?;

                for t in [qkv, z, beta_out, alpha_out, conv_out, q_part, k_part, v_part, q_gdn, k_gdn, attn_out, normed_out, o] {
                    gpu.free_tensor(t)?;
                }
                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttnMoe(layer), LayerType::FullAttention) => {
                // ── Full attention (same as dense FullAttn) ──
                gpu.rmsnorm_f32(&x, &layer.attn_norm, &tmp, config.norm_eps)?;

                let q_full_dim = config.n_heads * config.head_dim * 2;
                let q_full = gpu.alloc_tensor(&[q_full_dim], DType::F32)?;
                weight_gemv(gpu, &layer.wq, &tmp, &q_full)?;

                let q_dim = config.n_heads * config.head_dim;
                let q = gpu.alloc_tensor(&[q_dim], DType::F32)?;
                let gate_vec = gpu.alloc_tensor(&[q_dim], DType::F32)?;
                gpu.deinterleave_f32(&q_full, &q, &gate_vec, config.n_heads, config.head_dim)?;

                gpu.rmsnorm_batched(&q, &layer.q_norm, &q, config.n_heads, config.head_dim, config.norm_eps)?;

                let kv_dim = config.n_kv_heads * config.head_dim;
                let k = gpu.alloc_tensor(&[kv_dim], DType::F32)?;
                let v = gpu.alloc_tensor(&[kv_dim], DType::F32)?;
                weight_gemv(gpu, &layer.wk, &tmp, &k)?;
                weight_gemv(gpu, &layer.wv, &tmp, &v)?;

                gpu.rmsnorm_batched(&k, &layer.k_norm, &k, config.n_kv_heads, config.head_dim, config.norm_eps)?;

                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                gpu.rope_partial_interleaved_f32(&q, &k, &pos_buf,
                    config.n_heads, config.n_kv_heads, config.head_dim, n_rot, config.rope_theta)?;

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

                gpu.sigmoid_f32(&gate_vec)?;
                gpu.mul_f32(&attn_out, &gate_vec, &attn_out)?;

                let o = gpu.alloc_tensor(&[dim], DType::F32)?;
                weight_gemv(gpu, &layer.wo, &attn_out, &o)?;

                gpu.add_inplace_f32(&x, &o)?;

                // ── MoE FFN (only difference from dense) ──
                gpu.rmsnorm_f32(&x, &layer.ffn_norm, &tmp, config.norm_eps)?;
                moe_ffn_decode(gpu, &layer.ffn, &tmp, &x, config)?;

                for t in [q_full, q, gate_vec, k, v, attn_out, o] {
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

    // Flash attention partials buffer for tile+reduce 2-kernel path.
    // Size: n_heads * max_tiles * (2 + head_dim) floats.
    pub flash_partials: GpuTensor,
    // Flash attention tri-state (applies to Q8 path; asym modes are flash-only):
    //   0 = never      force non-flash at all contexts (except >15K sanity)
    //   1 = auto       (default) flash kicks in at ctx >= 2048
    //   2 = always     force flash at all contexts
    pub flash_mode: u8,
}

impl Qwen35Scratch {
    pub fn new(gpu: &mut Gpu, config: &Qwen35Config, repeat_window: usize) -> HipResult<Self> {
        // Flash partials are sized for up to 8192 ctx. Override via new_with_kv_max.
        Self::new_with_kv_max(gpu, config, repeat_window, 8192)
    }

    pub fn new_with_kv_max(gpu: &mut Gpu, config: &Qwen35Config, repeat_window: usize, kv_max_seq: usize) -> HipResult<Self> {
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

            // Flash attention partials: enough for max_seq with tile_size=128.
            // n_heads * max_tiles * (2 + head_dim) floats.
            flash_partials: {
                let tile_size = 128usize;
                let max_tiles = (kv_max_seq + tile_size - 1) / tile_size;
                // Sized for batched flash prefill: 64 positions × per-position partials.
                // Reused across layers (not per-call allocated). ~33MB for 9B at 32K.
                let batch_mult = 64usize;
                gpu.alloc_tensor(&[batch_mult * config.n_heads * max_tiles * (2 + config.head_dim)], DType::F32)?
            },
            // Flash attention tri-state for the Q8 path. Asym modes always
            // flash regardless.
            //   HIPFIRE_ATTN_FLASH=never|0|off    → non-flash at all contexts
            //   HIPFIRE_ATTN_FLASH=auto|1|on      → (default) flash at ctx >= 2048
            //   HIPFIRE_ATTN_FLASH=always|2|force → flash at all contexts
            flash_mode: match std::env::var("HIPFIRE_ATTN_FLASH").as_deref() {
                Ok("never") | Ok("0") | Ok("off") => 0,
                Ok("always") | Ok("2") | Ok("force") => 2,
                _ => 1, // auto / unset / any other value
            },
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
                   self.logits, self.sample_buf, self.repeat_buf, self.x_rot,
                   self.flash_partials] {
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
    let use_graph = std::env::var("HIPFIRE_GRAPH").ok().as_deref() == Some("1");

    // Embedding lookup into scratch.x (always direct, changes per token)
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.token_embd, &scratch.x, token, dim)?,
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.token_embd, &scratch.x, token, dim)?,
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &scratch.x, token, dim)?,
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &scratch.x, token, dim)?,
        _ => panic!("unsupported embedding format"),
    }

    if use_graph && gpu.graph_exec.is_some() {
        // ── Graph replay path ──
        // Update pos_buf on the device via stream write (no host→device copy).
        let stream = gpu.active_stream.as_ref().unwrap();
        gpu.hip.stream_write_value32(stream, &scratch.pos_buf, pos as u32, 0)?;
        gpu.graph_launch()?;
    } else if use_graph && gpu.graph_exec.is_none() {
        // ── First decode: capture the forward pass as a graph ──
        // Ensure we have an explicit stream for capture.
        if gpu.active_stream.is_none() {
            gpu.active_stream = Some(gpu.hip.stream_create()?);
        }
        // Write pos_buf before capture (this write is NOT in the graph)
        let pos_i32 = pos as i32;
        gpu.hip.memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        gpu.begin_graph_capture()?;
        forward_scratch_layers(gpu, weights, config, pos, kv_cache, dn_state, scratch, None)?;
        gpu.end_graph_capture()?;
        eprintln!("[hipGraph] captured {} blobs, instantiated", gpu.capture_blobs.len());
    } else {
        // ── Direct path (no graph) ──
        let pos_i32 = pos as i32;
        gpu.hip.memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        forward_scratch_layers(gpu, weights, config, pos, kv_cache, dn_state, scratch, None)?;
    }
    Ok(())
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

    // ── FullAttention batched intermediates (when FA weights are MQ4G256) ──
    // Positions array: [max_batch] i32, absolute KV positions for this chunk.
    // Uploaded once at the start of each chunk and reused by rope + kv_write
    // + attention kernels.
    pub positions: GpuTensor,
    // QKV projection outputs
    pub fa_q_full_batch: GpuTensor,  // [N × n_heads × head_dim × 2] (Q + gate interleaved)
    pub fa_q_batch: GpuTensor,       // [N × n_heads × head_dim]
    pub fa_gate_batch: GpuTensor,    // [N × n_heads × head_dim]
    pub fa_k_batch: GpuTensor,       // [N × n_kv_heads × head_dim]
    pub fa_v_batch: GpuTensor,       // [N × n_kv_heads × head_dim]
    pub fa_attn_out_batch: GpuTensor, // [N × n_heads × head_dim]
    // FWHT-rotated fa_attn_out for feeding MQ4 wo.
    pub fa_attn_out_rot_batch: GpuTensor, // [N × n_heads × head_dim]
}

impl PrefillBatchScratch {
    pub fn new(gpu: &mut Gpu, config: &Qwen35Config, max_batch: usize) -> HipResult<Self> {
        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let qkv_dim = k_dim * 2 + v_dim;
        let n_v_heads = config.linear_num_value_heads;
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;

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
            // F32 dtype = 4 bytes/element, same layout as i32. The rope /
            // attention / kv_write kernels cast the pointer to `const int*`,
            // so dtype is cosmetic. Upload i32 bits via memcpy_htod.
            positions:         gpu.alloc_tensor(&[max_batch], DType::F32)?,
            fa_q_full_batch:   gpu.alloc_tensor(&[max_batch * q_dim * 2], DType::F32)?,
            fa_q_batch:        gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
            fa_gate_batch:     gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
            fa_k_batch:        gpu.alloc_tensor(&[max_batch * kv_dim], DType::F32)?,
            fa_v_batch:        gpu.alloc_tensor(&[max_batch * kv_dim], DType::F32)?,
            fa_attn_out_batch: gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
            fa_attn_out_rot_batch: gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
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
            self.positions,
            self.fa_q_full_batch, self.fa_q_batch, self.fa_gate_batch,
            self.fa_k_batch, self.fa_v_batch, self.fa_attn_out_batch,
            self.fa_attn_out_rot_batch,
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
    //
    // Tuning note: each extra chunk pays full dispatch-overhead for the LA
    // preamble (rmsnorm, rotate, 4-way fused GEMM) and FFN (gate_up + down).
    // 256 costs ~80 MB of scratch on 9B vs 20 MB at 64 — trivial on modern
    // cards — and drops chunk count for pp2048 from 32 → 8. The inner
    // gated_delta_net_q8_batch_seq loop is still sequential per token, so
    // the per-chunk DeltaNet cost is linear in N either way; raising the
    // batch just amortizes the NON-DeltaNet kernels more.
    const MAX_BATCH: usize = 256;

    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }

    // Fast path requires (a) every LA layer's weights to be either MQ4G256
    // or HFQ4G256 (the batched GEMM kernels are dtype-agnostic but the LA
    // preamble's rmsnorm+rotate and SwiGLU+rotate kernels differ per dtype),
    // and (b) Q8 S-state for the GDN recurrence. Mixed-dtype layers are
    // allowed; each layer is routed to its own path. HFQ6/others fall back.
    // `HIPFIRE_PREFILL_BATCHED=0` forces the per-token fallback (escape
    // hatch for regression bisecting or diagnosing hardware-specific issues).
    let force_fallback = std::env::var("HIPFIRE_PREFILL_BATCHED").ok().as_deref() == Some("0");
    let eligible = !force_fallback
        && n >= MIN_BATCH
        && dn_state.quant == StateQuant::Q8
        && weights.layers.iter().any(|lw| matches!(lw, LayerWeights::DeltaNet(_)))
        && weights.layers.iter().all(|lw| match lw {
            LayerWeights::DeltaNet(l) =>
                is_batchable_la(l.wqkv.gpu_dtype)
                    && is_batchable_la(l.wz.gpu_dtype)
                    && is_batchable_la(l.w_beta.gpu_dtype)
                    && is_batchable_la(l.w_alpha.gpu_dtype)
                    && is_batchable_la(l.wo.gpu_dtype)
                    && is_batchable_la(l.w_gate.gpu_dtype)
                    && is_batchable_la(l.w_up.gpu_dtype)
                    && is_batchable_la(l.w_down.gpu_dtype),
            LayerWeights::FullAttn(_) => true, // FA layer will take the gather/scatter path
            // MoE variants — batched prefill not yet implemented; force the
            // per-token fallback path until the MoE plumbing lands.
            LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_) => false,
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

/// Accepts the dtypes the batched prefill path can handle (shared by the
/// eligibility check in `forward_prefill_batch` and the per-layer dtype
/// branches in `forward_prefill_chunk`).
#[inline]
fn is_batchable_la(dt: DType) -> bool {
    matches!(dt, DType::MQ4G256 | DType::HFQ4G256 | DType::MQ6G256 | DType::HFQ6G256)
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

    // ── 1b. Upload positions array (start_pos .. start_pos + n) ───────────
    // Used by batched rope / kv_write / attention kernels in FA layers.
    let positions_host: Vec<i32> = (0..n).map(|i| (start_pos + i) as i32).collect();
    let positions_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4)
    };
    gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;

    // Decide whether the FA layers can take the batched path. Requires
    // (a) all FA weights to be MQ4G256 or HFQ4G256 (the batched gemm_qkv
    // + wo GEMMs are dtype-agnostic; the rmsnorm+rotate / silu_mul kernels
    // differ by dtype and we branch on that at each layer) and (b) a Q8_0
    // or givens KV cache. If the check fails, FA layers fall back to
    // per-token gather/scatter via run_fa_layer_body.
    let fa_batched_ok = (kv_cache.quant_q8 || kv_cache.quant_asym4 || kv_cache.quant_asym3 || kv_cache.quant_asym2)
        && weights.layers.iter().all(|lw| match lw {
            LayerWeights::FullAttn(l) =>
                is_batchable_la(l.wq.gpu_dtype) &&
                is_batchable_la(l.wk.gpu_dtype) &&
                is_batchable_la(l.wv.gpu_dtype) &&
                is_batchable_la(l.wo.gpu_dtype) &&
                is_batchable_la(l.w_gate.gpu_dtype) &&
                is_batchable_la(l.w_up.gpu_dtype) &&
                is_batchable_la(l.w_down.gpu_dtype),
            _ => true, // LA layers don't gate this check
        });
    let max_ctx_len = start_pos + n;

    // ── 2. Per-layer loop ────────────────────────────────────────────────
    let mut delta_layer_idx = 0usize;
    let mut kv_layer_idx = 0usize;

    for layer_idx in 0..config.n_layers {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                // Per-layer dtype branch: MQ4 needs FWHT-rotation on the
                // activation to match its pre-rotated weights; HFQ4 uses
                // plain rmsnormed activations. The GEMM kernels themselves
                // are dtype-agnostic — they just consume whatever [N × K]
                // activation buffer we point them at.
                let is_mq = matches!(layer.wqkv.gpu_dtype, DType::MQ4G256 | DType::MQ6G256);
                let is_6bit = matches!(layer.wqkv.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);

                // Batched rmsnorm (+ FWHT for MQ) for the LA preamble.
                // x_batch / x_rot_batch are [N × dim] contiguous. For HFQ
                // we reuse x_rot_batch as the "normed, unrotated" output
                // so the subsequent GEMM can read it the same way.
                if is_mq {
                    gpu.fused_rmsnorm_rotate_mq_batched(
                        &pbs.x_batch, &layer.attn_norm, &pbs.x_rot_batch, dim, config.norm_eps, n,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch, &layer.attn_norm, &pbs.x_rot_batch,
                        n, dim, config.norm_eps,
                    )?;
                }

                // Batched 4-way LA projection (wqkv + wz + w_beta + w_alpha).
                if is_6bit {
                    gpu.gemm_qkvza_hfq6g256(
                        &layer.wqkv.buf, &layer.wz.buf, &layer.w_beta.buf, &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch, &pbs.dn_z_batch, &pbs.dn_beta_batch, &pbs.dn_alpha_batch,
                        layer.wqkv.m, layer.wz.m, layer.w_beta.m, layer.w_alpha.m,
                        layer.wqkv.k, n,
                    )?;
                } else {
                    gpu.gemm_qkvza_hfq4g256(
                        &layer.wqkv.buf, &layer.wz.buf, &layer.w_beta.buf, &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch, &pbs.dn_z_batch, &pbs.dn_beta_batch, &pbs.dn_alpha_batch,
                        layer.wqkv.m, layer.wz.m, layer.w_beta.m, layer.w_alpha.m,
                        layer.wqkv.k, n,
                    )?;
                }

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
                    // Batched repeat-interleave: one kernel launch for all N tokens.
                    gpu.repeat_interleave_qk_f32_batched(
                        &pbs.dn_q_raw_batch, &pbs.dn_k_raw_batch,
                        &pbs.dn_q_batch, &pbs.dn_k_batch,
                        config.linear_num_key_heads, ratio, hd, n,
                    )?;
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

                // Batched wo + residual.
                //
                // For MQ weights, the decode path's weight_gemv_residual
                // internally FWHT-rotates dn_normed into mq_x_rot before
                // calling gemv_hfq{4,6}g256_residual (MQ weights are pre-rotated
                // at quant time; math requires dot(rot(W), rot(x)) = dot(W,x)).
                // For HFQ weights no rotation is needed — the activation
                // feeds gemm_hfq{4,6}g256_residual directly.
                let wo_is_mq = matches!(layer.wo.gpu_dtype, DType::MQ4G256 | DType::MQ6G256);
                let wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let wo_input = if wo_is_mq {
                    gpu.rotate_x_mq_batched(
                        &pbs.dn_normed_batch, &pbs.dn_normed_rot_batch, layer.wo.k, n,
                    )?;
                    &pbs.dn_normed_rot_batch
                } else {
                    &pbs.dn_normed_batch
                };
                if wo_is_6bit {
                    gpu.gemm_hfq6g256_residual(
                        &layer.wo.buf, wo_input, &pbs.x_batch,
                        layer.wo.m, layer.wo.k, n,
                    )?;
                } else {
                    gpu.gemm_hfq4g256_residual(
                        &layer.wo.buf, wo_input, &pbs.x_batch,
                        layer.wo.m, layer.wo.k, n,
                    )?;
                }

                // FFN: rmsnorm (+ rotate for MQ).
                let ffn_is_mq = matches!(layer.w_gate.gpu_dtype, DType::MQ4G256 | DType::MQ6G256);
                let ffn_is_6bit = matches!(layer.w_gate.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                if ffn_is_mq {
                    gpu.fused_rmsnorm_rotate_mq_batched(
                        &pbs.x_batch, &layer.ffn_norm, &pbs.x_rot_batch, dim, config.norm_eps, n,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch, &layer.ffn_norm, &pbs.x_rot_batch,
                        n, dim, config.norm_eps,
                    )?;
                }

                // Batched gate+up projection.
                if ffn_is_6bit {
                    gpu.gemm_gate_up_hfq6g256(
                        &layer.w_gate.buf, &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch, &pbs.up_batch,
                        layer.w_gate.m, layer.w_up.m,
                        layer.w_gate.k, n,
                    )?;
                } else {
                    gpu.gemm_gate_up_hfq4g256(
                        &layer.w_gate.buf, &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch, &pbs.up_batch,
                        layer.w_gate.m, layer.w_up.m,
                        layer.w_gate.k, n,
                    )?;
                }

                // SwiGLU activation feeding w_down. For MQ, we need the
                // output FWHT-rotated so it matches the pre-rotated w_down
                // weights. For HFQ, plain silu_mul is enough. silu_mul_f32
                // is purely element-wise and uses numel() as its length,
                // so a [N × hidden_dim] tensor processes all rows in one
                // launch with no batch offset needed.
                let w_down_is_mq = matches!(layer.w_down.gpu_dtype, DType::MQ4G256 | DType::MQ6G256);
                let w_down_is_6bit = matches!(layer.w_down.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                if w_down_is_mq {
                    gpu.fused_silu_mul_rotate_mq_batched(
                        &pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch,
                        hidden_dim, n,
                    )?;
                } else {
                    gpu.silu_mul_f32(
                        &pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch,
                    )?;
                }

                // Batched w_down + residual.
                if w_down_is_6bit {
                    gpu.gemm_hfq6g256_residual(
                        &layer.w_down.buf, &pbs.ffn_hidden_batch, &pbs.x_batch,
                        layer.w_down.m, layer.w_down.k, n,
                    )?;
                } else {
                    gpu.gemm_hfq4g256_residual(
                        &layer.w_down.buf, &pbs.ffn_hidden_batch, &pbs.x_batch,
                        layer.w_down.m, layer.w_down.k, n,
                    )?;
                }

                let _ = is_mq; // retained above for potential future use
                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttn(layer), LayerType::FullAttention) if fa_batched_ok => {
                // Fully batched FA layer. Mirrors the FA branch of
                // forward_scratch_layers kernel-for-kernel, but every
                // launch covers all N tokens at once.
                let kv_dim = config.n_kv_heads * config.head_dim;
                let q_dim = config.n_heads * config.head_dim;
                let qkv_is_mq = matches!(layer.wq.gpu_dtype, DType::MQ4G256 | DType::MQ6G256);
                let qkv_is_6bit = matches!(layer.wq.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);

                // 1. rmsnorm (+ rotate for MQ) for the attn preamble.
                if qkv_is_mq {
                    gpu.fused_rmsnorm_rotate_mq_batched(
                        &pbs.x_batch, &layer.attn_norm, &pbs.x_rot_batch, dim, config.norm_eps, n,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch, &layer.attn_norm, &pbs.x_rot_batch,
                        n, dim, config.norm_eps,
                    )?;
                }

                // 2. Batched 3-way QKV projection (wq+wk+wv).
                if qkv_is_6bit {
                    gpu.gemm_qkv_hfq6g256(
                        &layer.wq.buf, &layer.wk.buf, &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch, &pbs.fa_k_batch, &pbs.fa_v_batch,
                        layer.wq.m, layer.wk.m, layer.wv.m,
                        layer.wq.k, n,
                    )?;
                } else {
                    gpu.gemm_qkv_hfq4g256(
                        &layer.wq.buf, &layer.wk.buf, &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch, &pbs.fa_k_batch, &pbs.fa_v_batch,
                        layer.wq.m, layer.wk.m, layer.wv.m,
                        layer.wq.k, n,
                    )?;
                }

                // 3. Batched deinterleave Q + gate: one kernel launch for all N tokens.
                gpu.deinterleave_f32_batched(
                    &pbs.fa_q_full_batch, &pbs.fa_q_batch, &pbs.fa_gate_batch,
                    config.n_heads, config.head_dim, n,
                )?;

                // 4. Per-head Q/K rmsnorm. rmsnorm_batched uses batch =
                // number of "rows" of head_dim. For [N × n_heads × head_dim]
                // that's batch = N * n_heads.
                gpu.rmsnorm_batched(
                    &pbs.fa_q_batch, &layer.q_norm, &pbs.fa_q_batch,
                    n * config.n_heads, config.head_dim, config.norm_eps,
                )?;
                gpu.rmsnorm_batched(
                    &pbs.fa_k_batch, &layer.k_norm, &pbs.fa_k_batch,
                    n * config.n_kv_heads, config.head_dim, config.norm_eps,
                )?;

                // 5. Batched partial-interleaved RoPE (per-row positions).
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                gpu.rope_partial_interleaved_f32_batched(
                    &pbs.fa_q_batch, &pbs.fa_k_batch, &pbs.positions,
                    config.n_heads, config.n_kv_heads, config.head_dim, n_rot,
                    config.rope_theta, n,
                )?;

                // 6. Batched KV cache writes (per-row positions).
                if kv_cache.quant_asym4 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    gpu.kv_cache_write_asym4_batched(
                        &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_k_batch, &pbs.fa_v_batch, &pbs.positions,
                        ct, st, config.n_kv_heads, config.head_dim, n,
                    )?;
                } else if kv_cache.quant_asym3 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    gpu.kv_cache_write_asym3_batched(
                        &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_k_batch, &pbs.fa_v_batch, &pbs.positions,
                        ct, st, config.n_kv_heads, config.head_dim, n,
                    )?;
                } else if kv_cache.quant_asym2 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    gpu.kv_cache_write_asym2_batched(
                        &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_k_batch, &pbs.fa_v_batch, &pbs.positions,
                        ct, st, config.n_kv_heads, config.head_dim, n,
                    )?;
                } else {
                    gpu.kv_cache_write_q8_0_batched(
                        &kv_cache.k_gpu[layer_idx], &pbs.fa_k_batch, &pbs.positions,
                        config.n_kv_heads, config.head_dim, n,
                    )?;
                    gpu.kv_cache_write_q8_0_batched(
                        &kv_cache.v_gpu[layer_idx], &pbs.fa_v_batch, &pbs.positions,
                        config.n_kv_heads, config.head_dim, n,
                    )?;
                }

                // 7. Batched causal attention.
                // asym{4,3,2}: batched flash (K rotated-quantized + V Q8 in normal space).
                // Q8: batched kernel unless ctx > 15K (LDS overflow), then per-position flash.
                const LDS_CTX_LIMIT: usize = 15000;
                if kv_cache.quant_asym4 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    gpu.attention_flash_asym4_batched(
                        &pbs.fa_q_batch, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch, &pbs.positions, ct, st,
                        config.n_heads, config.n_kv_heads, config.head_dim,
                        kv_cache.max_seq, max_ctx_len, n, &s.flash_partials,
                    )?;
                } else if kv_cache.quant_asym3 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    gpu.attention_flash_asym3_batched(
                        &pbs.fa_q_batch, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch, &pbs.positions, ct, st,
                        config.n_heads, config.n_kv_heads, config.head_dim,
                        kv_cache.max_seq, max_ctx_len, n, &s.flash_partials,
                    )?;
                } else if kv_cache.quant_asym2 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    gpu.attention_flash_asym2_batched(
                        &pbs.fa_q_batch, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch, &pbs.positions, ct, st,
                        config.n_heads, config.n_kv_heads, config.head_dim,
                        kv_cache.max_seq, max_ctx_len, n, &s.flash_partials,
                    )?;
                } else if max_ctx_len > LDS_CTX_LIMIT {
                    // Per-position flash Q8 attention for long-context prefill.
                    let q_dim = config.n_heads * config.head_dim;
                    let pos_host = gpu.download_f32(&pbs.positions)?;
                    let pos_buf_tmp = gpu.hip.malloc(4)?;
                    for b in 0..n {
                        let seq_len_b = pos_host[b] as usize + 1;
                        let pos_i32 = pos_host[b] as i32;
                        gpu.hip.memcpy_htod(&pos_buf_tmp, &pos_i32.to_ne_bytes())?;
                        let q_b = pbs.fa_q_batch.sub_offset(b * q_dim, q_dim);
                        let out_b = pbs.fa_attn_out_batch.sub_offset(b * q_dim, q_dim);
                        gpu.attention_flash_q8_0(
                            &q_b, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                            &out_b, &pos_buf_tmp, seq_len_b,
                            config.n_heads, config.n_kv_heads, config.head_dim,
                            kv_cache.max_seq, &s.flash_partials,
                        )?;
                    }
                    let _ = gpu.hip.free(pos_buf_tmp);
                } else {
                    gpu.attention_q8_0_kv_batched(
                        &pbs.fa_q_batch,
                        &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch, &pbs.positions,
                        config.n_heads, config.n_kv_heads, config.head_dim,
                        kv_cache.max_seq, max_ctx_len, n,
                    )?;
                }

                // 8. Fused sigmoid(gate) * attn_out, element-wise over the
                // full [N × q_dim] tensor.
                gpu.sigmoid_mul_f32(&pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;

                // 9. wo residual: x_batch += wo · (optional rotate)(fa_attn_out_batch).
                // Same MQ rotation requirement as the LA wo path.
                let fa_wo_is_mq = matches!(layer.wo.gpu_dtype, DType::MQ4G256 | DType::MQ6G256);
                let fa_wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let fa_wo_input = if fa_wo_is_mq {
                    gpu.rotate_x_mq_batched(
                        &pbs.fa_attn_out_batch, &pbs.fa_attn_out_rot_batch, layer.wo.k, n,
                    )?;
                    &pbs.fa_attn_out_rot_batch
                } else {
                    &pbs.fa_attn_out_batch
                };
                if fa_wo_is_6bit {
                    gpu.gemm_hfq6g256_residual(
                        &layer.wo.buf, fa_wo_input, &pbs.x_batch,
                        layer.wo.m, layer.wo.k, n,
                    )?;
                } else {
                    gpu.gemm_hfq4g256_residual(
                        &layer.wo.buf, fa_wo_input, &pbs.x_batch,
                        layer.wo.m, layer.wo.k, n,
                    )?;
                }

                // 10. FFN: rmsnorm (+ rotate for MQ), gate+up, silu_mul
                // (+ rotate for MQ), w_down residual.
                let fa_ffn_is_mq = matches!(layer.w_gate.gpu_dtype, DType::MQ4G256 | DType::MQ6G256);
                let fa_ffn_is_6bit = matches!(layer.w_gate.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                if fa_ffn_is_mq {
                    gpu.fused_rmsnorm_rotate_mq_batched(
                        &pbs.x_batch, &layer.ffn_norm, &pbs.x_rot_batch, dim, config.norm_eps, n,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch, &layer.ffn_norm, &pbs.x_rot_batch,
                        n, dim, config.norm_eps,
                    )?;
                }
                if fa_ffn_is_6bit {
                    gpu.gemm_gate_up_hfq6g256(
                        &layer.w_gate.buf, &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch, &pbs.up_batch,
                        layer.w_gate.m, layer.w_up.m,
                        layer.w_gate.k, n,
                    )?;
                } else {
                    gpu.gemm_gate_up_hfq4g256(
                        &layer.w_gate.buf, &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch, &pbs.up_batch,
                        layer.w_gate.m, layer.w_up.m,
                        layer.w_gate.k, n,
                    )?;
                }
                let fa_w_down_is_mq = matches!(layer.w_down.gpu_dtype, DType::MQ4G256 | DType::MQ6G256);
                let fa_w_down_is_6bit = matches!(layer.w_down.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                if fa_w_down_is_mq {
                    gpu.fused_silu_mul_rotate_mq_batched(
                        &pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch,
                        hidden_dim, n,
                    )?;
                } else {
                    gpu.silu_mul_f32(
                        &pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch,
                    )?;
                }
                if fa_w_down_is_6bit {
                    gpu.gemm_hfq6g256_residual(
                        &layer.w_down.buf, &pbs.ffn_hidden_batch, &pbs.x_batch,
                        layer.w_down.m, layer.w_down.k, n,
                    )?;
                } else {
                    gpu.gemm_hfq4g256_residual(
                        &layer.w_down.buf, &pbs.ffn_hidden_batch, &pbs.x_batch,
                        layer.w_down.m, layer.w_down.k, n,
                    )?;
                }

                // Silence unused warning if kv_dim ends up shadowed.
                let _ = kv_dim;
                kv_layer_idx += 1;
            }

            (LayerWeights::FullAttn(_layer), LayerType::FullAttention) => {
                // Per-token gather/scatter fallback for FA layers that don't
                // qualify for batched FA (non-MQ4 weights, non-Q8_0 KV, etc).
                for i in 0..n {
                    let pos = start_pos + i;
                    gpu.hip.memcpy_dtod_at(&s.x.buf, 0, &pbs.x_batch.buf, i * dim_row_bytes, dim_row_bytes)?;
                    let pos_i32 = pos as i32;
                    gpu.hip.memcpy_htod(&s.pos_buf, &pos_i32.to_ne_bytes())?;
                    run_fa_layer_body(gpu, weights, config, layer_idx, kv_layer_idx, pos, kv_cache, s)?;
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
    gpu.rope_partial_interleaved_f32(&s.fa_q, &s.fa_k, &s.pos_buf,
        config.n_heads, config.n_kv_heads, config.head_dim, n_rot, config.rope_theta)?;

    if kv_cache.quant_asym4 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        gpu.kv_cache_write_asym4_fused(
            &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
            &s.fa_k, &s.fa_v, &s.pos_buf, ct, st, config.n_kv_heads, config.head_dim)?;
        gpu.attention_flash_asym4(
            &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
            &s.fa_attn_out, &s.pos_buf, ct, st, pos + 1,
            config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
            &s.flash_partials,
        )?;
    } else if kv_cache.quant_asym3 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        gpu.kv_cache_write_asym3_fused(
            &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
            &s.fa_k, &s.fa_v, &s.pos_buf, ct, st, config.n_kv_heads, config.head_dim)?;
        gpu.attention_flash_asym3(
            &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
            &s.fa_attn_out, &s.pos_buf, ct, st, pos + 1,
            config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
            &s.flash_partials,
        )?;
    } else if kv_cache.quant_asym2 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        gpu.kv_cache_write_asym2_fused(
            &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
            &s.fa_k, &s.fa_v, &s.pos_buf, ct, st, config.n_kv_heads, config.head_dim)?;
        gpu.attention_flash_asym2(
            &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
            &s.fa_attn_out, &s.pos_buf, ct, st, pos + 1,
            config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
            &s.flash_partials,
        )?;
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
                gpu.rope_partial_interleaved_f32(&s.fa_q, &s.fa_k, &s.pos_buf,
                    config.n_heads, config.n_kv_heads, config.head_dim, n_rot, config.rope_theta)?;

                if kv_cache.quant_asym4 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    gpu.kv_cache_write_asym4_fused(
                        &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_k, &s.fa_v, &s.pos_buf, ct, st, config.n_kv_heads, config.head_dim)?;
                    gpu.attention_flash_asym4(
                        &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_attn_out, &s.pos_buf, ct, st, pos + 1,
                        config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
                        &s.flash_partials,
                    )?;
                } else if kv_cache.quant_asym3 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    gpu.kv_cache_write_asym3_fused(
                        &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_k, &s.fa_v, &s.pos_buf, ct, st, config.n_kv_heads, config.head_dim)?;
                    gpu.attention_flash_asym3(
                        &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_attn_out, &s.pos_buf, ct, st, pos + 1,
                        config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
                        &s.flash_partials,
                    )?;
                } else if kv_cache.quant_asym2 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    gpu.kv_cache_write_asym2_fused(
                        &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_k, &s.fa_v, &s.pos_buf, ct, st, config.n_kv_heads, config.head_dim)?;
                    gpu.attention_flash_asym2(
                        &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_attn_out, &s.pos_buf, ct, st, pos + 1,
                        config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
                        &s.flash_partials,
                    )?;
                } else if kv_cache.quant_q8 {
                    gpu.kv_cache_write_q8_0(&kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf, config.n_kv_heads, config.head_dim)?;
                    gpu.kv_cache_write_q8_0(&kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf, config.n_kv_heads, config.head_dim)?;
                    // Flash dispatch (Q8 path):
                    //   - capture_mode (hipGraph): always flash — position-independent grid.
                    //   - flash_mode=2 (always): force flash at any ctx.
                    //   - flash_mode=1 (auto, default): flash at ctx >= 2048.
                    //   - flash_mode=0 (never): non-flash until sanity cap (>15K ctx).
                    //   - >15K: always flash (non-flash VRAM blowup).
                    let use_flash = gpu.capture_mode
                        || s.flash_mode == 2
                        || (s.flash_mode == 1 && pos + 1 >= 2048)
                        || pos + 1 > 15000;
                    if use_flash {
                        gpu.attention_flash_q8_0(
                            &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out, &s.pos_buf, pos + 1,
                            config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
                            &s.flash_partials,
                        )?;
                    } else {
                        gpu.attention_q8_0_kv(
                            &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out, &s.pos_buf, pos + 1,
                            config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
                        )?;
                    }
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

            // ── MoE variants (Qwen3.5-MoE / A3B) ──
            // Attention path mirrors the dense counterpart above; FFN is
            // replaced by moe_ffn_decode (router + top-K + shared + routed).
            // The MQ-rotate pre-FFN fusion used by the dense FFN doesn't
            // apply here — moe_ffn_decode uses plain weight_gemv, which
            // does its own internal MQ rotation once per call. Re-rotation
            // overhead is one of the items targeted by Phase 2/3 speedups.
            (LayerWeights::DeltaNetMoe(layer), LayerType::LinearAttention) => {
                let x_rot = fused_rmsnorm_rotate_for_mq(
                    gpu, &layer.wqkv, &s.x, &layer.attn_norm, &s.tmp, &s.x_rot, config.norm_eps,
                )?;
                let dt = layer.wqkv.gpu_dtype;
                let fused_la4_ok = (dt == DType::MQ4G256 || dt == DType::HFQ4G256)
                    && layer.wz.gpu_dtype == dt
                    && layer.w_beta.gpu_dtype == dt
                    && layer.w_alpha.gpu_dtype == dt;
                if fused_la4_ok {
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
                gpu.fused_sigmoid_alpha_gate_f32(
                    &s.dn_beta, &s.dn_alpha, &layer.dt_bias, &layer.a_log, n_v_heads,
                )?;
                gpu.conv1d_silu_split_f32(
                    &s.dn_q_raw, &s.dn_k_raw, &s.dn_v,
                    &s.dn_qkv, &layer.conv_weight,
                    &dn_state.conv_states[delta_layer_idx],
                    k_dim, v_dim,
                )?;
                gpu.fused_qk_l2_norm_scale_f32(
                    &s.dn_q_raw, &s.dn_k_raw,
                    config.linear_num_key_heads, hd,
                    1.0 / (hd as f32).sqrt(),
                    config.norm_eps,
                )?;
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
                weight_gemv_residual(gpu, &layer.wo, &s.dn_normed, &s.x)?;

                // ── MoE FFN ──
                gpu.rmsnorm_f32(&s.x, &layer.ffn_norm, &s.tmp, config.norm_eps)?;
                moe_ffn_decode(gpu, &layer.ffn, &s.tmp, &s.x, config)?;

                if let Some(ref rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_at_head(gpu, slot, &s.x)?;
                    }
                }
                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttnMoe(layer), LayerType::FullAttention) => {
                let x_rot = fused_rmsnorm_rotate_for_mq(
                    gpu, &layer.wq, &s.x, &layer.attn_norm, &s.tmp, &s.x_rot, config.norm_eps,
                )?;
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

                gpu.deinterleave_f32(&s.fa_q_full, &s.fa_q, &s.fa_gate, config.n_heads, config.head_dim)?;
                gpu.rmsnorm_batched(&s.fa_q, &layer.q_norm, &s.fa_q, config.n_heads, config.head_dim, config.norm_eps)?;

                let kv_dim = config.n_kv_heads * config.head_dim;
                gpu.rmsnorm_batched(&s.fa_k, &layer.k_norm, &s.fa_k, config.n_kv_heads, config.head_dim, config.norm_eps)?;

                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                gpu.rope_partial_interleaved_f32(&s.fa_q, &s.fa_k, &s.pos_buf,
                    config.n_heads, config.n_kv_heads, config.head_dim, n_rot, config.rope_theta)?;

                if kv_cache.quant_asym4 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    gpu.kv_cache_write_asym4_fused(
                        &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_k, &s.fa_v, &s.pos_buf, ct, st, config.n_kv_heads, config.head_dim)?;
                    gpu.attention_flash_asym4(
                        &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_attn_out, &s.pos_buf, ct, st, pos + 1,
                        config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
                        &s.flash_partials,
                    )?;
                } else if kv_cache.quant_asym3 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    gpu.kv_cache_write_asym3_fused(
                        &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_k, &s.fa_v, &s.pos_buf, ct, st, config.n_kv_heads, config.head_dim)?;
                    gpu.attention_flash_asym3(
                        &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_attn_out, &s.pos_buf, ct, st, pos + 1,
                        config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
                        &s.flash_partials,
                    )?;
                } else if kv_cache.quant_asym2 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    gpu.kv_cache_write_asym2_fused(
                        &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_k, &s.fa_v, &s.pos_buf, ct, st, config.n_kv_heads, config.head_dim)?;
                    gpu.attention_flash_asym2(
                        &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_attn_out, &s.pos_buf, ct, st, pos + 1,
                        config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
                        &s.flash_partials,
                    )?;
                } else if kv_cache.quant_q8 {
                    gpu.kv_cache_write_q8_0(&kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf, config.n_kv_heads, config.head_dim)?;
                    gpu.kv_cache_write_q8_0(&kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf, config.n_kv_heads, config.head_dim)?;
                    let use_flash = gpu.capture_mode
                        || s.flash_mode == 2
                        || (s.flash_mode == 1 && pos + 1 >= 2048)
                        || pos + 1 > 15000;
                    if use_flash {
                        gpu.attention_flash_q8_0(
                            &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out, &s.pos_buf, pos + 1,
                            config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
                            &s.flash_partials,
                        )?;
                    } else {
                        gpu.attention_q8_0_kv(
                            &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out, &s.pos_buf, pos + 1,
                            config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
                        )?;
                    }
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

                // ── MoE FFN ──
                gpu.rmsnorm_f32(&s.x, &layer.ffn_norm, &s.tmp, config.norm_eps)?;
                moe_ffn_decode(gpu, &layer.ffn, &s.tmp, &s.x, config)?;

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
