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

/// Optional tree-attention context for `forward_prefill_batch` — activates
/// DDTree batched verify when `Some`.
///
/// Fields:
/// - `positions`: length matches `tokens.len()`. Each slot's logical RoPE
///   position (seed at `start_pos`, node i at `start_pos + depth_i`).
///   Two nodes at the same tree depth share a logical position — they're
///   alternative futures at the same time step, not successive tokens.
/// - `attn_bias`: `[N × N]` f32 additive bias on qk scores (with N = tokens.len()),
///   produced by `crate::ddtree::linearize_tree`. `0.0` on ancestor-or-self
///   entries, `-inf` on non-ancestors. Applied to in-block keys only;
///   prompt keys (positions `[0, start_pos)`) remain unmasked.
///
/// Tree mode requires the batched FA path (`fa_batched_ok`); the per-token
/// FA fallback always uses causal attention and cannot honor a tree mask.
/// `forward_prefill_batch` returns an error if tree mode is requested but
/// any FA layer would take the fallback path.
///
/// GDN (LinearAttention) layers: if `parent_indices` is `Some`, the
/// DeltaNet branch dispatches the tree-aware kernels
/// (`conv1d_silu_split_tree_f32_n` + `gated_delta_net_q8_tree_batch_seq`)
/// which walk per-token ancestor chains via `parent_indices` instead of
/// the linear-sequence predecessor. This eliminates sibling-subtree
/// cross-contamination of recurrent state at topk>1. If `parent_indices`
/// is `None`, LA layers fall back to the linear path (byte-exact with
/// DFlash at topk=1; approximation at topk>1 — used by pre-Phase-3
/// callers that haven't been rewritten).
#[derive(Clone, Copy)]
pub struct TreeVerifyCtx<'a> {
    pub positions: &'a [i32],
    pub attn_bias: &'a GpuTensor,
    /// `[N]` i32 — for each linearized slot, the slot index of its parent
    /// in the same linearization (or -1 for the root / seed). Produced by
    /// `crate::ddtree::linearize_tree_with_parents`. When `Some`, LA layers
    /// use tree-aware kernels that read parent state from the per-layer
    /// s_tape scratch in `PrefillBatchScratch`.
    pub parent_indices: Option<&'a GpuTensor>,
    /// Per-FA-layer F32 scratch buffers for capturing K BEFORE RoPE is
    /// applied. Used by Path B slow-path-kill: on the slow path, the
    /// speculative caller gathers accepted K rows out of these scratches,
    /// re-runs RoPE with COMMITTED slot phases (instead of the
    /// linearization phases the in-cache K carries), and re-quants to
    /// the committed kv_cache slots — avoiding a full re-verify forward
    /// while preserving RoPE phase correctness.
    ///
    /// Slice length must equal the number of FullAttention layers in
    /// `config.layer_types`; each entry is a `[max_n × n_kv_heads × head_dim]`
    /// F32 tensor (max_n = 1 + tree budget). When `None`, capture is
    /// skipped (zero overhead). When `Some`, every tree-verify FA layer
    /// memcpy_dtod's its `pbs.fa_k_batch` (post-norm, pre-RoPE) into the
    /// scratch BEFORE the rope kernel mutates it.
    pub pre_rope_k_capture: Option<&'a [GpuTensor]>,
}

#[derive(Debug, Clone)]
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
    /// Device-side array of `unsigned long long` pointers, one per
    /// expert's `gate_up.buf`. Indexed at runtime by the GPU top-K
    /// kernel's output so the indexed MoE GEMV can stay capture-safe.
    pub expert_gate_up_ptrs: GpuTensor,       // [num_experts * 2] f32 slots = num_experts × u64
    pub expert_down_ptrs:    GpuTensor,       // [num_experts * 2] f32 slots = num_experts × u64
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
    let _ = gpu.free_tensor(ffn.expert_gate_up_ptrs);
    let _ = gpu.free_tensor(ffn.expert_down_ptrs);
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

/// Load norm weight without the +1.0 offset — for standard RMSNorm tensors
/// (e.g., the final `model.language_model.norm.weight` stored as raw scale,
/// mean ~1.6 on Qwen3.5-MoE A3B). Applying +1.0 would over-amplify by ~60%.
fn load_norm_weight_raw(hfq: &HfqFile, gpu: &mut Gpu, name: &str, shape: &[usize]) -> HipResult<GpuTensor> {
    let full_name = format!("model.language_model.{name}");
    let (info, data) = hfq.tensor_data(&full_name)
        .or_else(|| hfq.tensor_data(name))
        .unwrap_or_else(|| panic!("tensor not found: {name} or {full_name}"));
    let f32_data: Vec<f32> = match info.quant_type {
        1 => data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        2 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        _ => panic!("expected F16/F32 for {name}, got qt={}", info.quant_type),
    };
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
    // Final output norm: on Qwen3.5/3.6-MoE (A3B, arch_id=6) this tensor is
    // stored as a raw RMSNorm scale (mean ~+1.6), NOT as deviation-from-0 like
    // the per-block norms. Applying `w += 1.0` (via `load_norm_weight`) would
    // over-amplify the pre-lm_head hidden state by ~60%, which on 3.6 MQ4 tips
    // the model into infinite `<think>` spirals on reasoning prompts (3.5 MQ4
    // tolerates it but is still subtly wrong). Dense Qwen3.5 0.8B/4B/9B use
    // the deviation-from-0 convention and require `+=1.0` — they keep their
    // byte-exact quality-gate baselines unchanged. Gate on num_experts > 0.
    let output_norm = if config.num_experts > 0 {
        load_norm_weight_raw(hfq, gpu, "norm.weight", &[config.dim])?
    } else {
        load_norm_weight(hfq, gpu, "norm.weight", &[config.dim])?
    };

    // Try separate lm_head first (untied embeddings, e.g. 9B), fall back to tied embed_tokens
    let lm_head_info = hfq.tensor_data("lm_head.weight")
        .or_else(|| hfq.tensor_data("model.language_model.lm_head.weight"));
    let output = if let Some((lm_info, lm_data)) = lm_head_info {
        eprintln!("  loading output (separate lm_head, qt={})...", lm_info.quant_type);
        load_weight_tensor_raw(gpu, lm_info.quant_type, lm_data, config.vocab_size, config.dim)?
    } else {
        eprintln!("  loading output (tied embeddings, qt={})...", embd_info.0.quant_type);
        let embd_data = hfq.tensor_data("model.language_model.embed_tokens.weight").unwrap().1;
        if embd_info.0.quant_type == 6 || embd_info.0.quant_type == 7 || embd_info.0.quant_type == 8 {
            let buf = gpu.upload_raw(embd_data, &[embd_data.len()])?;
            let dtype = match embd_info.0.quant_type {
                6 => DType::HFQ4G256, 7 => DType::HFQ4G128, 8 => DType::HFQ6G256, _ => unreachable!()
            };
            WeightTensor { buf, gpu_dtype: dtype, m: config.vocab_size, k: config.dim, row_stride: 0 }
        } else if embd_info.0.quant_type == 13 {
            // MQ4-G256 tied embedding — produced by hipfire-quantize
            // `--format mq4-all`. DFlash uses this to make the target's
            // lm_head (tied to embed_tokens) hit the batched MQ4 GEMM path.
            let buf = gpu.upload_raw(embd_data, &[embd_data.len()])?;
            WeightTensor { buf, gpu_dtype: DType::MQ4G256, m: config.vocab_size, k: config.dim, row_stride: 0 }
        } else if embd_info.0.quant_type == 14 {
            let buf = gpu.upload_raw(embd_data, &[embd_data.len()])?;
            WeightTensor { buf, gpu_dtype: DType::MQ8G256, m: config.vocab_size, k: config.dim, row_stride: 0 }
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
fn load_moe_ffn(hfq: &HfqFile, gpu: &mut Gpu, p: &str, config: &Qwen35Config) -> HipResult<MoeFfnWeights> {
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

    // Build the device-side pointer tables consumed by the indexed MoE
    // GEMV kernels. Each slot is an `unsigned long long` (the device
    // address of an expert's `gate_up.buf` / `down.buf`). Stored as an
    // F32 tensor of length 2 * num_experts because each pointer occupies
    // 8 bytes = 2 F32 slots; the kernel reads them via a u64 cast.
    let mut gu_ptrs: Vec<u64> = Vec::with_capacity(n_exp);
    let mut dn_ptrs: Vec<u64> = Vec::with_capacity(n_exp);
    for e in &experts {
        gu_ptrs.push(e.gate_up.buf.buf.as_ptr() as u64);
        dn_ptrs.push(e.down.buf.buf.as_ptr()    as u64);
    }
    let gu_bytes: Vec<u8> = gu_ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let dn_bytes: Vec<u8> = dn_ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let expert_gate_up_ptrs = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    let expert_down_ptrs    = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    gpu.hip.memcpy_htod(&expert_gate_up_ptrs.buf, &gu_bytes)?;
    gpu.hip.memcpy_htod(&expert_down_ptrs.buf,    &dn_bytes)?;

    Ok(MoeFfnWeights {
        router, experts, shared_expert, shared_expert_gate,
        expert_gate_up_ptrs, expert_down_ptrs,
    })
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
/// Non-owning borrow of the scratch buffers `moe_ffn_decode_impl` needs.
/// Callers construct one of these from either a `Qwen35Scratch` (preallocated,
/// hipGraph-capturable) or from tensors they own locally (heap path).
struct MoeScratchRef<'a> {
    router_logits: &'a GpuTensor,
    scalar_buf:    &'a GpuTensor,
    x_rot_local:   &'a GpuTensor,
    gate_up_buf:   &'a GpuTensor,
    gate_buf:      &'a GpuTensor,
    up_buf:        &'a GpuTensor,
    ffn_hidden:    &'a GpuTensor,
    ffn_out:       &'a GpuTensor,
    gate_batch:    &'a GpuTensor,
    up_batch:      &'a GpuTensor,
    rot_batch:     &'a GpuTensor,
    topk_indices:  &'a GpuTensor,
    topk_weights:  &'a GpuTensor,
}

impl<'a> MoeScratchRef<'a> {
    /// View into a Qwen35Scratch's MoE fields. Panics if the caller didn't
    /// allocate MoE scratch (config.num_experts == 0).
    fn from_scratch(s: &'a Qwen35Scratch) -> Self {
        Self {
            router_logits: s.moe_router_logits.as_ref().expect("MoE scratch not allocated"),
            scalar_buf:    s.moe_scalar_buf.as_ref().expect("MoE scratch"),
            x_rot_local:   s.moe_x_rot.as_ref().expect("MoE scratch"),
            gate_up_buf:   s.moe_gate_up_buf.as_ref().expect("MoE scratch"),
            gate_buf:      s.moe_gate_buf.as_ref().expect("MoE scratch"),
            up_buf:        s.moe_up_buf.as_ref().expect("MoE scratch"),
            ffn_hidden:    s.moe_ffn_hidden.as_ref().expect("MoE scratch"),
            ffn_out:       s.moe_ffn_out.as_ref().expect("MoE scratch"),
            gate_batch:    s.moe_gate_batch.as_ref().expect("MoE scratch"),
            up_batch:      s.moe_up_batch.as_ref().expect("MoE scratch"),
            rot_batch:     s.moe_rot_batch.as_ref().expect("MoE scratch"),
            topk_indices:  s.moe_topk_indices.as_ref().expect("MoE scratch"),
            topk_weights:  s.moe_topk_weights.as_ref().expect("MoE scratch"),
        }
    }
}

/// Heap-allocating wrapper for callers without pre-allocated scratch (the
/// debug `forward()` path). Allocates 11 tensors, runs moe_ffn_decode_impl,
/// frees. NOT hipGraph-compatible. For hot-path decode, callers should go
/// through moe_ffn_decode_with_scratch which reuses pre-allocated buffers.
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

    let router_logits = gpu.alloc_tensor(&[n_exp], DType::F32)?;
    let scalar_buf    = gpu.alloc_tensor(&[1], DType::F32)?;
    let x_rot_local   = gpu.alloc_tensor(&[hidden], DType::F32)?;
    let gate_up_buf   = gpu.alloc_tensor(&[2 * max_inter], DType::F32)?;
    let gate_buf      = gpu.alloc_tensor(&[max_inter], DType::F32)?;
    let up_buf        = gpu.alloc_tensor(&[max_inter], DType::F32)?;
    let ffn_hidden    = gpu.alloc_tensor(&[max_inter], DType::F32)?;
    let ffn_out       = gpu.alloc_tensor(&[hidden], DType::F32)?;
    let gate_batch    = gpu.alloc_tensor(&[k * mi], DType::F32)?;
    let up_batch      = gpu.alloc_tensor(&[k * mi], DType::F32)?;
    let rot_batch     = gpu.alloc_tensor(&[k * mi], DType::F32)?;
    let topk_indices  = gpu.alloc_tensor(&[k], DType::F32)?;
    let topk_weights  = gpu.alloc_tensor(&[k], DType::F32)?;

    let refs = MoeScratchRef {
        router_logits: &router_logits,
        scalar_buf:    &scalar_buf,
        x_rot_local:   &x_rot_local,
        gate_up_buf:   &gate_up_buf,
        gate_buf:      &gate_buf,
        up_buf:        &up_buf,
        ffn_hidden:    &ffn_hidden,
        ffn_out:       &ffn_out,
        gate_batch:    &gate_batch,
        up_batch:      &up_batch,
        rot_batch:     &rot_batch,
        topk_indices:  &topk_indices,
        topk_weights:  &topk_weights,
    };
    let result = moe_ffn_decode_impl(gpu, ffn, x_norm, x_residual, config, &refs, false);

    for t in [router_logits, scalar_buf, x_rot_local, gate_up_buf, gate_buf,
              up_buf, ffn_hidden, ffn_out, gate_batch, up_batch, rot_batch,
              topk_indices, topk_weights] {
        gpu.free_tensor(t)?;
    }
    result
}

/// All gate-side + routed MoE weights are MQ4G256 — the precondition for
/// the prerotated fast path where the caller can fuse rmsnorm+FWHT via
/// `fused_rmsnorm_rotate_mq` and call `moe_ffn_decode_with_scratch_prerotated`.
fn ffn_all_mq4_for_moe(ffn: &MoeFfnWeights) -> bool {
    ffn.router.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert_gate.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert.gate.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert.up.gpu_dtype == DType::MQ4G256
        && ffn.experts.iter().all(|e| e.gate_up.gpu_dtype == DType::MQ4G256)
}

/// Zero-alloc MoE decode for the scratch path. `scratch.moe_*` fields must
/// be populated (done automatically by `Qwen35Scratch::new` when config
/// indicates a MoE model). Safe to call under hipGraph stream capture.
fn moe_ffn_decode_with_scratch(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x_norm: &GpuTensor,
    x_residual: &GpuTensor,
    config: &Qwen35Config,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    let refs = MoeScratchRef::from_scratch(scratch);
    moe_ffn_decode_impl(gpu, ffn, x_norm, x_residual, config, &refs, false)
}

/// Same as `moe_ffn_decode_with_scratch` but expects the caller to have
/// already populated `scratch.moe_x_rot` with FWHT-rotated post-rmsnorm x
/// (e.g. via a fused `fused_rmsnorm_rotate_mq` launch at the call site).
/// For all-MQ4 MoE layers this saves one launch per layer by eliding the
/// internal `rotate_x_mq`. On non-MQ4 layers this flag is ignored.
fn moe_ffn_decode_with_scratch_prerotated(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x_norm: &GpuTensor,
    x_residual: &GpuTensor,
    config: &Qwen35Config,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    let refs = MoeScratchRef::from_scratch(scratch);
    moe_ffn_decode_impl(gpu, ffn, x_norm, x_residual, config, &refs, true)
}

/// The actual MoE FFN implementation. Uses the caller-provided scratch
/// buffers, never allocates.
fn moe_ffn_decode_impl(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x_norm: &GpuTensor,
    x_residual: &GpuTensor,
    config: &Qwen35Config,
    s: &MoeScratchRef<'_>,
    x_rot_prerotated: bool,
) -> HipResult<()> {
    let hidden = config.dim;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let k = config.num_experts_per_tok;
    let n_exp = config.num_experts;
    let _ = hidden;

    let router_logits = s.router_logits;
    let scalar_buf    = s.scalar_buf;
    let gate_up_buf   = s.gate_up_buf;
    let gate_buf      = s.gate_buf;
    let up_buf        = s.up_buf;
    let ffn_hidden    = s.ffn_hidden;
    let ffn_out       = s.ffn_out;

    // Phase 2a-iii: rotate x_norm once per layer and share the rotated
    // buffer across every gate-side GEMV. Only MQ4 GEMVs benefit; mixed
    // configs fall back to weight_gemv which rotates internally.
    let gate_side_mq4 = ffn.router.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert_gate.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert.gate.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert.up.gpu_dtype == DType::MQ4G256
        && ffn.experts.iter().all(|e| e.gate_up.gpu_dtype == DType::MQ4G256);
    let x_rot_local = if gate_side_mq4 {
        gpu.ensure_mq_signs()?;
        if !x_rot_prerotated {
            gpu.rotate_x_mq(x_norm, s.x_rot_local, config.dim)?;
        }
        // else caller guarantees s.x_rot_local already holds FWHT(rmsnorm(x)).
        Some(s.x_rot_local)
    } else {
        None
    };

    // Detect Phase 2b+2c GPU-only fast path. When true, top-K runs on
    // device and the indexed MoE kernels consume topk_indices /
    // topk_weights directly — no D2H sync, hipGraph-capture-safe.
    let routed_mq4 = ffn.experts.first()
        .map(|e| e.down.gpu_dtype == DType::MQ4G256)
        .unwrap_or(false);
    let routed_gate_up_mq4 = ffn.experts.first()
        .map(|e| e.gate_up.gpu_dtype == DType::MQ4G256)
        .unwrap_or(false);
    let use_gpu_topk = k == 8 && gate_side_mq4 && routed_mq4 && routed_gate_up_mq4;

    // ── 1+2b+3a. Fused 4-way GEMV (router + shared_expert_gate + shared.gate + shared.up) ──
    // All four read the SAME rotated x_rot_local with the SAME K. Fusing them
    // into `fused_qkvza_hfq4g256` saves 3 launch submits per MoE layer and
    // lets underused tails (shared_expert_gate_m=1, router_m=256) co-schedule
    // with the larger 512-row gate/up bodies. 40 layers × 3 saved launches
    // = 120 launches/fwd, ~8-12% cycle-time savings on 7900 XTX.
    let shared_gate = slice_f32_view(gate_buf, 0, smi);
    let shared_up   = slice_f32_view(up_buf,   0, smi);
    if let Some(xr) = x_rot_local {
        // All MQ4: use the 4-way fused prerotated GEMV. Router weight, shared
        // sigmoid-gate weight, shared gate weight, shared up weight — all
        // M×K matrices in HFQ4G256 storage (MQ4 weights are HFQ4 bytes pre-
        // rotated at quant time, so `gemv_hfq4g256` inner loop with the
        // FWHT-rotated input is mathematically equivalent to `gemv_mq4g256`).
        gpu.fused_qkvza_hfq4g256(
            &ffn.router.buf, &ffn.shared_expert_gate.buf,
            &ffn.shared_expert.gate.buf, &ffn.shared_expert.up.buf,
            xr,
            router_logits, scalar_buf, &shared_gate, &shared_up,
            ffn.router.m, ffn.shared_expert_gate.m,
            ffn.shared_expert.gate.m, ffn.shared_expert.up.m,
            ffn.router.k,
        )?;
    } else {
        // Mixed-dtype fallback: four separate `weight_gemv` calls. Each
        // weight_gemv handles its own rotation for MQ4 weights internally.
        weight_gemv(gpu, &ffn.router, x_norm, router_logits)?;
        weight_gemv(gpu, &ffn.shared_expert_gate, x_norm, scalar_buf)?;
        weight_gemv(gpu, &ffn.shared_expert.gate, x_norm, &shared_gate)?;
        weight_gemv(gpu, &ffn.shared_expert.up,   x_norm, &shared_up)?;
    }

    // ── 2a. Top-K selection — GPU fast path or CPU fallback ──
    let (topk_indices_cpu, topk_weights_cpu): (Option<Vec<usize>>, Option<Vec<f32>>) = if use_gpu_topk {
        // GPU path: softmax + top-K + renorm in one launch, results land
        // in s.topk_indices [k] (i32) and s.topk_weights [k] (f32) on
        // device. Indexed MoE kernels consume them directly — no D2H.
        gpu.moe_softmax_topk_renorm_k8(
            router_logits, s.topk_indices, s.topk_weights,
            n_exp, config.norm_topk_prob,
        )?;
        (None, None)
    } else {
        // Fallback: GPU softmax → CPU download → CPU top-K + renorm.
        gpu.softmax_f32(router_logits)?;
        let probs = gpu.download_f32(router_logits)?;
        let mut indices: Vec<usize> = (0..n_exp).collect();
        indices.select_nth_unstable_by(k - 1, |&a, &b| {
            probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut topk_indices: Vec<usize> = indices.into_iter().take(k).collect();
        topk_indices.sort_by(|&a, &b| {
            probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut topk_weights: Vec<f32> = topk_indices.iter().map(|&i| probs[i]).collect();
        if config.norm_topk_prob {
            let sum: f32 = topk_weights.iter().sum();
            if sum > 0.0 {
                for w in topk_weights.iter_mut() { *w /= sum; }
            }
        }
        (Some(topk_indices), Some(topk_weights))
    };

    // The shared-expert gate scalar (in `scalar_buf`) is the RAW logit from
    // the 4-way fused GEMV — sigmoid is applied internally by
    // `gemv_hfq4g256_residual_sigmoid_scaled_gpu`, eliminating the separate
    // 1-elem `sigmoid_f32` launch (~40 saved per forward on A3B).
    if ffn.shared_expert.down.gpu_dtype == DType::MQ4G256 {
        gpu.ensure_mq_signs()?;
        let x_rot_alias = GpuTensor {
            buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
            shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
            dtype: DType::F32,
        };
        gpu.fused_silu_mul_rotate_mq(&shared_gate, &shared_up, &x_rot_alias, smi)?;
        gpu.gemv_hfq4g256_residual_sigmoid_scaled_gpu(
            &ffn.shared_expert.down.buf, &x_rot_alias, x_residual, scalar_buf,
            ffn.shared_expert.down.m, ffn.shared_expert.down.k,
        )?;
    } else {
        // Non-MQ fallback path still needs the separate sigmoid + scaled-add.
        gpu.sigmoid_f32(scalar_buf)?;
        // Non-MQ fallback: pre-2a-ii path.
        let shared_hid = slice_f32_view(ffn_hidden, 0, smi);
        gpu.silu_mul_f32(&shared_gate, &shared_up, &shared_hid)?;
        weight_gemv(gpu, &ffn.shared_expert.down, &shared_hid, ffn_out)?;
        gpu.scaled_add_inplace_gpu_scalar_f32(x_residual, ffn_out, scalar_buf)?;
    }

    // ── 4. Top-K routed experts ──
    if routed_mq4 {
        gpu.ensure_mq_signs()?;
    }

    if use_gpu_topk {
        // Phase 2b+2c GPU-only fast path: indexed MoE kernels read expert
        // IDs and weights from device buffers. 3 launches for routed
        // compute, zero D2H sync — hipGraph-capturable.
        let xr = x_rot_local.expect("gate_side_mq4 implies x_rot_local");
        let down_m = ffn.experts[0].down.m;
        let down_k = ffn.experts[0].down.k;
        let gate_up_k = ffn.experts[0].gate_up.k;
        gpu.gemv_hfq4g256_moe_gate_up_k8_indexed(
            &ffn.expert_gate_up_ptrs, s.topk_indices,
            xr, s.gate_batch, s.up_batch,
            2 * mi, gate_up_k,
        )?;
        gpu.fused_silu_mul_rotate_mq_batched(s.gate_batch, s.up_batch, s.rot_batch, mi, k)?;
        gpu.gemv_hfq4g256_moe_down_residual_scaled_k8_indexed(
            &ffn.expert_down_ptrs, s.topk_indices, s.topk_weights,
            s.rot_batch, x_residual,
            down_m, down_k,
        )?;
    } else {
        // CPU-top-K fallback path. Two sub-paths from here:
        //   (a) k==8 && all-MQ4 but gate_side wasn't all-MQ4 (e.g. router
        //       not MQ4): use the kernarg-pointer fused kernels with the
        //       CPU-selected indices.
        //   (b) Mixed-dtype or k != 8: per-expert loop.
        let topk_indices = topk_indices_cpu.expect("CPU-fallback path implies CPU top-K");
        let topk_weights = topk_weights_cpu.expect("CPU-fallback path implies CPU top-K");
        let use_kernarg_fused = k == 8 && routed_gate_up_mq4 && x_rot_local.is_some();
        if use_kernarg_fused {
            let xr = x_rot_local.unwrap();
            let e0 = &ffn.experts[topk_indices[0]];
            let e1 = &ffn.experts[topk_indices[1]];
            let e2 = &ffn.experts[topk_indices[2]];
            let e3 = &ffn.experts[topk_indices[3]];
            let e4 = &ffn.experts[topk_indices[4]];
            let e5 = &ffn.experts[topk_indices[5]];
            let e6 = &ffn.experts[topk_indices[6]];
            let e7 = &ffn.experts[topk_indices[7]];
            gpu.gemv_hfq4g256_moe_gate_up_k8(
                &e0.gate_up.buf, &e1.gate_up.buf, &e2.gate_up.buf, &e3.gate_up.buf,
                &e4.gate_up.buf, &e5.gate_up.buf, &e6.gate_up.buf, &e7.gate_up.buf,
                xr, s.gate_batch, s.up_batch,
                2 * mi, e0.gate_up.k,
            )?;
            gpu.fused_silu_mul_rotate_mq_batched(s.gate_batch, s.up_batch, s.rot_batch, mi, k)?;
            let scales = [
                topk_weights[0], topk_weights[1], topk_weights[2], topk_weights[3],
                topk_weights[4], topk_weights[5], topk_weights[6], topk_weights[7],
            ];
            gpu.gemv_hfq4g256_moe_down_residual_scaled_k8(
                &e0.down.buf, &e1.down.buf, &e2.down.buf, &e3.down.buf,
                &e4.down.buf, &e5.down.buf, &e6.down.buf, &e7.down.buf,
                s.rot_batch, x_residual, scales,
                e0.down.m, e0.down.k,
            )?;
        } else {
            // Per-expert fallback for layers that aren't all-MQ4 or have k != 8.
            for (&expert_idx, &weight) in topk_indices.iter().zip(topk_weights.iter()) {
                let expert = &ffn.experts[expert_idx];
                if let Some(xr) = x_rot_local {
                    gpu.gemv_mq4g256_prerotated(&expert.gate_up.buf, xr, gate_up_buf,
                        expert.gate_up.m, expert.gate_up.k)?;
                } else {
                    weight_gemv(gpu, &expert.gate_up, x_norm, gate_up_buf)?;
                }
                let gate_view = slice_f32_view(gate_up_buf, 0,  mi);
                let up_view   = slice_f32_view(gate_up_buf, mi, mi);
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
                    let hid_view = slice_f32_view(ffn_hidden, 0, mi);
                    gpu.silu_mul_f32(&gate_view, &up_view, &hid_view)?;
                    weight_gemv(gpu, &expert.down, &hid_view, ffn_out)?;
                    gpu.scaled_add_inplace_cpu_scalar_f32(x_residual, ffn_out, weight)?;
                }
            }
        }
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
                        &attn_out, &pos_buf, pos + 1, config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.physical_cap,
                    )?;
                } else {
                    gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &k, &pos_buf, kv_dim)?;
                    gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &v, &pos_buf, kv_dim)?;
                    gpu.attention_f32(
                        &q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &attn_out, &pos_buf, pos + 1, config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.physical_cap,
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
                        &attn_out, &pos_buf, pos + 1, config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.physical_cap,
                    )?;
                } else {
                    gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &k, &pos_buf, kv_dim)?;
                    gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &v, &pos_buf, kv_dim)?;
                    gpu.attention_f32(
                        &q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &attn_out, &pos_buf, pos + 1, config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.physical_cap,
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

    // MoE scratch (allocated only when config.num_experts > 0). Pre-allocated
    // so moe_ffn_decode can be captured by hipGraph — the per-layer allocs
    // it used to do violated the "no allocator ops while capturing" rule.
    pub moe_router_logits: Option<GpuTensor>,   // [num_experts]
    pub moe_scalar_buf:    Option<GpuTensor>,   // [1] shared-expert gate scalar
    pub moe_x_rot:         Option<GpuTensor>,   // [dim]
    pub moe_gate_up_buf:   Option<GpuTensor>,   // [2*max_inter]   fallback path
    pub moe_gate_buf:      Option<GpuTensor>,   // [max_inter]     fallback path
    pub moe_up_buf:        Option<GpuTensor>,   // [max_inter]     fallback path
    pub moe_ffn_hidden:    Option<GpuTensor>,   // [max_inter]     fallback path
    pub moe_ffn_out:       Option<GpuTensor>,   // [dim]           fallback path
    pub moe_gate_batch:    Option<GpuTensor>,   // [k × mi]
    pub moe_up_batch:      Option<GpuTensor>,   // [k × mi]
    pub moe_rot_batch:     Option<GpuTensor>,   // [k × mi]
    /// Phase 2b: GPU-side top-K outputs (kept on-device so moe_ffn_decode
    /// can stay in a graph-capturable stream).
    pub moe_topk_indices:  Option<GpuTensor>,   // [k] i32 stored as f32 alias
    pub moe_topk_weights:  Option<GpuTensor>,   // [k] f32

    // Optional long-prefill scratch. Default is None to preserve VRAM
    // footprint; set HIPFIRE_PREFILL_REUSE_PBS=1 to allocate and reuse it.
    pub prefill_batch: Option<PrefillBatchScratch>,
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
            // n_heads * max_tiles * (2 + head_dim) floats per batched query
            // position; total buffer = batch_mult × per-position-bytes.
            //
            // batch_mult is the maximum query positions a single FA dispatch
            // can fit; the dispatcher (`launch_asym_flash_batched`) reads the
            // buffer's actual capacity at call time and auto-chunks larger
            // prefill batches into multiple sub-launches. So a lower
            // batch_mult here trades ~linear extra dispatch overhead on
            // prefill (PREFILL_MAX_BATCH=256 → ceil(256/batch_mult) calls per
            // FA layer) for ~linearly less VRAM at long context.
            //
            // The per-position size scales with kv_max_seq (= physical_cap
            // post-eviction), and that scaling is what made #85 visible: at
            // max_seq=170k, no CASK, 27B (n_heads=24, head_dim=256) the old
            // batch_mult=64 → 2.1 GB just for these partials, exceeding VRAM
            // headroom on 24 GB cards. Cutting batch_mult by 4× (16) keeps
            // the prefill chunking moderate while saving 1.6 GB at that
            // worst-case shape; CASK-on workloads (small physical_cap) are
            // unaffected because the buffer is already tiny there.
            //
            // Override with HIPFIRE_FLASH_PARTIALS_BATCH for tuning. Power of
            // two preferred (matches FA dispatcher chunking).
            flash_partials: {
                let tile_size = 128usize;
                let max_tiles = (kv_max_seq + tile_size - 1) / tile_size;
                let batch_mult = std::env::var("HIPFIRE_FLASH_PARTIALS_BATCH")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .filter(|&n| n >= 1 && n <= PREFILL_MAX_BATCH)
                    .unwrap_or(16);
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

            moe_router_logits: None,
            moe_scalar_buf:    None,
            moe_x_rot:         None,
            moe_gate_up_buf:   None,
            moe_gate_buf:      None,
            moe_up_buf:        None,
            moe_ffn_hidden:    None,
            moe_ffn_out:       None,
            moe_gate_batch:    None,
            moe_up_batch:      None,
            moe_rot_batch:     None,
            moe_topk_indices:  None,
            moe_topk_weights:  None,
            prefill_batch:     None,
        })
        .and_then(|mut s| {
            // Allocate MoE scratch only for MoE configs. Done after the
            // main struct init so these Options start as None for dense
            // models and never cost VRAM there.
            if config.num_experts > 0 {
                let hidden = config.dim;
                let n_exp = config.num_experts;
                let mi = config.moe_intermediate_size;
                let smi = config.shared_expert_intermediate_size;
                let max_inter = mi.max(smi);
                let k = config.num_experts_per_tok;
                s.moe_router_logits = Some(gpu.alloc_tensor(&[n_exp], DType::F32)?);
                s.moe_scalar_buf    = Some(gpu.alloc_tensor(&[1], DType::F32)?);
                s.moe_x_rot         = Some(gpu.alloc_tensor(&[hidden], DType::F32)?);
                s.moe_gate_up_buf   = Some(gpu.alloc_tensor(&[2 * max_inter], DType::F32)?);
                s.moe_gate_buf      = Some(gpu.alloc_tensor(&[max_inter], DType::F32)?);
                s.moe_up_buf        = Some(gpu.alloc_tensor(&[max_inter], DType::F32)?);
                s.moe_ffn_hidden    = Some(gpu.alloc_tensor(&[max_inter], DType::F32)?);
                s.moe_ffn_out       = Some(gpu.alloc_tensor(&[hidden], DType::F32)?);
                s.moe_gate_batch    = Some(gpu.alloc_tensor(&[k * mi], DType::F32)?);
                s.moe_up_batch      = Some(gpu.alloc_tensor(&[k * mi], DType::F32)?);
                s.moe_rot_batch     = Some(gpu.alloc_tensor(&[k * mi], DType::F32)?);
                // i32 topk_indices stored in an F32 tensor (same byte width).
                // The kernel that writes it casts the buffer to int*, and the
                // indexed MoE GEMV kernels read it as int*.
                s.moe_topk_indices  = Some(gpu.alloc_tensor(&[k], DType::F32)?);
                s.moe_topk_weights  = Some(gpu.alloc_tensor(&[k], DType::F32)?);
                // Pre-warm MQ FWHT sign tables (otherwise the lazy init in
                // ensure_mq_signs fires during the first moe_ffn_decode and
                // blows up hipGraph capture with a hipMalloc-in-capture
                // error). Idempotent if already computed.
                gpu.ensure_mq_signs()?;
            }
            if std::env::var("HIPFIRE_PREFILL_REUSE_PBS").ok().as_deref() == Some("1") {
                let max_batch = std::env::var("HIPFIRE_PREFILL_MAX_BATCH")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .filter(|&v| v >= 2)
                    .unwrap_or(PREFILL_MAX_BATCH);
                s.prefill_batch = Some(PrefillBatchScratch::new(gpu, config, max_batch)?);
            }
            Ok(s)
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
        // MoE scratch — only present for MoE configs.
        for t in [self.moe_router_logits, self.moe_scalar_buf, self.moe_x_rot,
                   self.moe_gate_up_buf, self.moe_gate_buf, self.moe_up_buf,
                   self.moe_ffn_hidden, self.moe_ffn_out,
                   self.moe_gate_batch, self.moe_up_batch, self.moe_rot_batch,
                   self.moe_topk_indices, self.moe_topk_weights] {
            if let Some(buf) = t { let _ = gpu.free_tensor(buf); }
        }
        if let Some(pbs) = self.prefill_batch {
            pbs.free_gpu(gpu);
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
    // hipGraph capture is currently DISABLED for MoE configs. Single-shot
    // replay looks fine for short sequences, but state diverges from the
    // direct-dispatch path after ~30–50 decoded tokens — the model drops
    // a number in a count (skips "8" → "9"), loops on a single token, etc.
    // The divergence is consistent under HIPFIRE_GRAPH=1 with the same
    // prompt that succeeds with HIPFIRE_GRAPH=0. Investigated without
    // finding the root cause: all kernels used by the MoE forward path
    // appear individually graph-safe (pos-dependent ones read pos_buf
    // dynamically; size-dependent ones use max_tiles/max_seq; the indexed
    // MoE kernels have only static pointer kernargs). Suspect a numerical
    // reordering between capture and replay in one of the flash-attn or
    // GDN state-update kernels that compounds over many replays.
    // Until that's isolated, MoE always takes the direct path.
    // HIPFIRE_GRAPH_MOE=1 (diagnostic-only): bypass the MoE guard. Required
    // to reproduce task #100. Under-graph A3B does NOT corrupt at step 1 —
    // it accumulates numerical drift and diverges from direct at step ~6
    // with q8 KV or step ~114 with asym3 KV on the Count-from-1-to-20
    // prompt. Migrating `kv_cache_write_q8_0` to the blob launch path (it
    // was the only remaining non-blob kernel in the MoE hot path) did not
    // resolve the drift — the root cause is elsewhere, likely DeltaNet
    // state accumulating tiny bit-level differences across replays via a
    // numerical-reordering path (atomics or wavefront-scheduling dependent
    // reductions inside gated_delta_net_*). Reproducer for next dig:
    //   HIPFIRE_GRAPH=1 HIPFIRE_GRAPH_MOE=1 HIPFIRE_SMOKE_KV=q8 \
    //   HIPFIRE_SMOKE_MODE=chat HIPFIRE_SMOKE_STEPS=200 \
    //   HIPFIRE_SMOKE_PROMPT="Count from one to twenty in English." \
    //   ./target/release/examples/a3b_smoke_forward <a3b.mq4>
    let allow_moe = std::env::var("HIPFIRE_GRAPH_MOE").ok().as_deref() == Some("1");
    let use_graph = std::env::var("HIPFIRE_GRAPH").ok().as_deref() == Some("1")
        && (config.num_experts == 0 || allow_moe);

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
        let pos_i32 = pos as i32;
        if !gpu.ar_forward_warmed_up {
            // ── Warmup: run direct so kernel JIT and lazy scratch
            // allocations (MQ signs/x_rot/x_q8, FP16 shadow, kernel module
            // load) happen outside any captured region. Capturing the first
            // call hits "hipMalloc not permitted under stream capture" — the
            // same trap `verify_warmed_up` solves for the verify path. The
            // next call will capture. (See bench_qwen35_mq4 / forward_prefill_batch
            // for the production warmup path.)
            gpu.ar_forward_warmed_up = true;
            gpu.hip.memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
            forward_scratch_layers(gpu, weights, config, pos, kv_cache, dn_state, scratch, None)?;
        } else {
            // ── First post-warmup call: capture the forward pass as a graph ──
            // Ensure we have an explicit stream for capture.
            if gpu.active_stream.is_none() {
                gpu.active_stream = Some(gpu.hip.stream_create()?);
            }
            // Write pos_buf before capture (this write is NOT in the graph)
            gpu.hip.memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
            gpu.begin_graph_capture()?;
            forward_scratch_layers(gpu, weights, config, pos, kv_cache, dn_state, scratch, None)?;
            gpu.end_graph_capture()?;
            // hipStreamCaptureModeGlobal RECORDS kernels — they do not execute
            // during capture. Launch the freshly-instantiated graph once so
            // this pos's forward actually runs (KV write, state advance,
            // logits update). Same pattern as the verify path's
            // begin_verify_graph_capture / end_verify_graph_capture /
            // verify_graph_launch sequence.
            gpu.graph_launch()?;
            eprintln!("[hipGraph] captured {} blobs, instantiated", gpu.capture_blobs.len());
        }
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
    // Token-ids buffer feeding the batched embedding kernel. [max_batch] i32
    // stored as F32 (same dtype-cosmetic pattern as `positions`). Uploaded
    // once per batched forward and read by `embedding_lookup_hfq4g256_batched`.
    pub tokens: GpuTensor,
    // QKV projection outputs
    pub fa_q_full_batch: GpuTensor,  // [N × n_heads × head_dim × 2] (Q + gate interleaved)
    pub fa_q_batch: GpuTensor,       // [N × n_heads × head_dim]
    pub fa_gate_batch: GpuTensor,    // [N × n_heads × head_dim]
    pub fa_k_batch: GpuTensor,       // [N × n_kv_heads × head_dim]
    pub fa_v_batch: GpuTensor,       // [N × n_kv_heads × head_dim]
    pub fa_attn_out_batch: GpuTensor, // [N × n_heads × head_dim]
    // FWHT-rotated fa_attn_out for feeding MQ4 wo.
    pub fa_attn_out_rot_batch: GpuTensor, // [N × n_heads × head_dim]

    // ── MoE batched intermediates (allocated only when num_experts > 0) ──
    // All outputs of the fused 4-way router + shared-gate GEMM, plus the
    // per-token routed-expert gate/up/rot buffers consumed by the N-batched
    // indexed MoE kernels. Sized as [max_batch × {n_exp, smi, k_top×mi}].
    pub moe_router_logits_batch: Option<GpuTensor>,   // [N × num_experts]
    pub moe_shared_scalar_batch: Option<GpuTensor>,   // [N × 1] — raw shared_expert_gate logit
    pub moe_shared_gate_batch:   Option<GpuTensor>,   // [N × smi]
    pub moe_shared_up_batch:     Option<GpuTensor>,   // [N × smi]
    pub moe_shared_rot_batch:    Option<GpuTensor>,   // [N × smi] — FWHT(silu(gate) * up)
    pub moe_topk_indices_batch:  Option<GpuTensor>,   // [N × k_top] i32 in F32 slots
    pub moe_topk_weights_batch:  Option<GpuTensor>,   // [N × k_top]
    pub moe_gate_batch:          Option<GpuTensor>,   // [N × k_top × mi]
    pub moe_up_batch:            Option<GpuTensor>,   // [N × k_top × mi]
    pub moe_rot_batch:           Option<GpuTensor>,   // [N × k_top × mi]

    // ── Tree-aware LA scratch (Phase 3b of Task #101) ──
    // Per-token S-state tape consumed by gated_delta_net_q8_tree kernel
    // when TreeVerifyCtx.parent_indices is Some. Reused across LA layers
    // since LA dispatch is serial per-cycle. Only allocated when the model
    // has LA layers (linear_num_value_heads > 0). Call sites that pass
    // parent_indices must ensure these tensors exist.
    //
    // s_tape_q8:     [max_batch × n_v_heads × head_dim × head_dim] Raw/i8
    // s_tape_scales: [max_batch × n_v_heads × head_dim] f32
    //
    // At max_batch=22, n_v_heads=16, head_dim=128 → 5.77 MB + 180 KB total.
    pub dn_s_tape_q8:     Option<GpuTensor>,
    pub dn_s_tape_scales: Option<GpuTensor>,
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
            tokens:            gpu.alloc_tensor(&[max_batch], DType::F32)?,
            fa_q_full_batch:   gpu.alloc_tensor(&[max_batch * q_dim * 2], DType::F32)?,
            fa_q_batch:        gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
            fa_gate_batch:     gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
            fa_k_batch:        gpu.alloc_tensor(&[max_batch * kv_dim], DType::F32)?,
            fa_v_batch:        gpu.alloc_tensor(&[max_batch * kv_dim], DType::F32)?,
            fa_attn_out_batch: gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
            fa_attn_out_rot_batch: gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
            moe_router_logits_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[max_batch * config.num_experts], DType::F32)?)
            } else { None },
            moe_shared_scalar_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[max_batch], DType::F32)?)
            } else { None },
            moe_shared_gate_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[max_batch * config.shared_expert_intermediate_size], DType::F32)?)
            } else { None },
            moe_shared_up_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[max_batch * config.shared_expert_intermediate_size], DType::F32)?)
            } else { None },
            moe_shared_rot_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[max_batch * config.shared_expert_intermediate_size], DType::F32)?)
            } else { None },
            moe_topk_indices_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[max_batch * config.num_experts_per_tok], DType::F32)?)
            } else { None },
            moe_topk_weights_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[max_batch * config.num_experts_per_tok], DType::F32)?)
            } else { None },
            moe_gate_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[max_batch * config.num_experts_per_tok * config.moe_intermediate_size], DType::F32)?)
            } else { None },
            moe_up_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[max_batch * config.num_experts_per_tok * config.moe_intermediate_size], DType::F32)?)
            } else { None },
            moe_rot_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[max_batch * config.num_experts_per_tok * config.moe_intermediate_size], DType::F32)?)
            } else { None },
            dn_s_tape_q8: if config.linear_num_value_heads > 0 {
                let bytes = max_batch * config.linear_num_value_heads * config.linear_value_head_dim * config.linear_value_head_dim;
                Some(gpu.alloc_tensor(&[bytes], DType::Raw)?)
            } else { None },
            dn_s_tape_scales: if config.linear_num_value_heads > 0 {
                Some(gpu.alloc_tensor(&[max_batch * config.linear_num_value_heads * config.linear_value_head_dim], DType::F32)?)
            } else { None },
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
            self.positions, self.tokens,
            self.fa_q_full_batch, self.fa_q_batch, self.fa_gate_batch,
            self.fa_k_batch, self.fa_v_batch, self.fa_attn_out_batch,
            self.fa_attn_out_rot_batch,
        ] {
            let _ = gpu.free_tensor(t);
        }
        for t in [
            self.moe_router_logits_batch, self.moe_shared_scalar_batch,
            self.moe_shared_gate_batch, self.moe_shared_up_batch, self.moe_shared_rot_batch,
            self.moe_topk_indices_batch, self.moe_topk_weights_batch,
            self.moe_gate_batch, self.moe_up_batch, self.moe_rot_batch,
            self.dn_s_tape_q8, self.dn_s_tape_scales,
        ] {
            if let Some(t) = t { let _ = gpu.free_tensor(t); }
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
///
/// `hidden_rb`: if `Some`, post-layer residual hidden states are captured
/// into the ring buffer for the configured extract layers. Used by the
/// DFlash target-side verify path to batch `verify_dflash_block` into a
/// single forward launch (MVP does B per-token forwards — 88 ms on 4B;
/// this path drops it to ~40 ms with batched forward, further improvement
/// possible with batched lm_head). The per-token fallback also honors it,
/// so the fast-path eligibility doesn't change behavior.
///
/// `per_token_hidden_out`: if `Some`, writes post-output-norm hidden state
/// for each of the N tokens into the provided [N × dim] buffer. The caller
/// then loops `weight_gemv(weights.output, hidden_row, logits)` to recover
/// per-token logits. Required for DFlash verify (needs all B positions'
/// logits, not just the last). `None` preserves the existing "last token
/// only" semantics where logits land in `scratch.logits`.
///
/// `gdn_tape`: if `Some`, captures the post-processed `(q, k, v, α, β)` for
/// every DN (LinearAttention) layer and block position BEFORE the batched
/// `gated_delta_net_q8_batch_seq` call. Enables the DFlash rollback path
/// to replay GDN recurrence from a pre-verify S-state snapshot for
/// `accept_len + 1` steps — no full-target re-run needed.
#[allow(clippy::too_many_arguments)]
/// Upper bound on `forward_prefill_batch`'s per-chunk size. Exposed so
/// callers sizing `HiddenStateRingBuffer` staging can match the chunk
/// upper bound (staging that's smaller than a chunk will assert-fail
/// on prompt seeding of long prompts).
pub const PREFILL_MAX_BATCH: usize = 256;

/// Host-side helper: upload token ids and positions to a `PrefillBatchScratch`
/// via sync `memcpy_htod`. Call this BEFORE entering a hipGraph capture to
/// pre-populate `pbs.tokens` and `pbs.positions`, then pass `pre_uploaded:
/// true` (or use `forward_prefill_chunk_captured_safe`) so the forward
/// does not issue any additional uploads inside the captured region.
pub fn upload_prefill_batch_inputs(
    gpu: &mut Gpu,
    pbs: &PrefillBatchScratch,
    tokens: &[u32],
    start_pos: usize,
) -> HipResult<()> {
    let n = tokens.len();
    let tokens_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let tokens_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4)
    };
    gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;
    let positions_host: Vec<i32> = (0..n).map(|i| (start_pos + i) as i32).collect();
    let positions_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4)
    };
    gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;
    Ok(())
}

/// Capture-friendly entry point that runs the batched forward against a
/// SINGLE chunk (`tokens.len() <= pbs.max_batch`), skipping the internal
/// token/position upload and assuming the caller has already populated
/// `pbs.tokens` / `pbs.positions` via `upload_prefill_batch_inputs`.
///
/// This exists so `hipStreamBeginCapture` can wrap the forward without
/// the per-call `memcpy_htod` sync operations (which would either error
/// under capture or bake stale host data into the captured graph nodes).
///
/// Callers still must handle `hidden_rb.commit_staging_to_ring(gpu, n)`
/// AFTER the forward returns (outside any captured region) to scatter
/// staging writes to the ring at the current head.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_single_chunk_captured(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
    hidden_rb: Option<&HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
) -> HipResult<()> {
    let n = tokens.len();
    debug_assert!(n > 0 && n <= pbs.max_batch,
        "single_chunk_captured: n={} but pbs.max_batch={}", n, pbs.max_batch);
    forward_prefill_chunk(
        gpu, weights, config, tokens, start_pos,
        kv_cache, dn_state, scratch, pbs, hidden_rb,
        per_token_hidden_out.map(|t| (t, 0)),
        gdn_tape, 0, tree_verify,
        true, // pre_uploaded: caller must have run upload_prefill_batch_inputs
    )
}

pub fn forward_prefill_batch(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    hidden_rb: Option<&mut HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
) -> HipResult<()> {
    forward_prefill_batch_with_pbs(
        gpu, weights, config, tokens, start_pos, kv_cache, dn_state, scratch,
        hidden_rb, per_token_hidden_out, gdn_tape, tree_verify, scratch.prefill_batch.as_ref(),
    )
}

/// Like `forward_prefill_batch`, but accepts a caller-owned `PrefillBatchScratch`
/// so the ~25 per-cycle tensor allocations can be amortized across many calls.
///
/// `pbs = None` preserves the original behavior (per-call allocate + free);
/// `pbs = Some(&pbs)` reuses the provided scratch. The provided scratch's
/// `max_batch` determines the chunk size — `tokens` is processed in chunks of
/// up to `pbs.max_batch`. Callers driving DFlash verify should size `pbs`
/// to the maximum block size they'll ever request (e.g. `block_size` or
/// `1 + tree_budget`) so everything fits in one chunk.
pub fn forward_prefill_batch_with_pbs(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    mut hidden_rb: Option<&mut HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    mut gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    pbs_in: Option<&PrefillBatchScratch>,
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
    //
    // Exposed via PREFILL_MAX_BATCH so callers sizing `HiddenStateRingBuffer`
    // staging can match the chunk upper bound.
    let max_batch: usize = std::env::var("HIPFIRE_PREFILL_MAX_BATCH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v >= MIN_BATCH)
        .unwrap_or(PREFILL_MAX_BATCH);

    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }

    // Tree-verify mode sanity checks — the downstream path can't silently
    // fall back to per-token FA (that's always causal and would ignore the
    // tree mask), and the positions/bias shapes must match the token count.
    if let Some(ctx) = tree_verify.as_ref() {
        assert_eq!(
            ctx.positions.len(), n,
            "TreeVerifyCtx.positions length {} must equal tokens.len() {}",
            ctx.positions.len(), n,
        );
        assert_eq!(
            ctx.attn_bias.numel(), n * n,
            "TreeVerifyCtx.attn_bias must be [{} × {}] f32 ({}), got numel {}",
            n, n, n * n, ctx.attn_bias.numel(),
        );
    }

    // Fast path requires (a) every LA layer's weights to be either MQ4G256
    // or HFQ4G256 (the batched GEMM kernels are dtype-agnostic but the LA
    // preamble's rmsnorm+rotate and SwiGLU+rotate kernels differ per dtype),
    // and (b) Q8 S-state for the GDN recurrence. Mixed-dtype layers are
    // allowed; each layer is routed to its own path. HFQ6/others fall back.
    // `HIPFIRE_PREFILL_BATCHED=0` forces the per-token fallback (escape
    // hatch for regression bisecting or diagnosing hardware-specific issues).
    let force_fallback = std::env::var("HIPFIRE_PREFILL_BATCHED").ok().as_deref() == Some("0");
    // MoE batched path requires K_TOP=8 (hard-coded in the indexed kernels)
    // and num_experts ≤ 1024 (bound of the batched top-K shared mem). When
    // either constraint is violated, reject all MoE layers so the whole
    // chunk falls through to per-token.
    let moe_topk_ok = config.num_experts_per_tok == 8 && config.num_experts <= 1024;
    let eligible = !force_fallback
        && n >= MIN_BATCH
        && dn_state.quant == StateQuant::Q8
        && weights.layers.iter().any(|lw| matches!(
            lw,
            LayerWeights::DeltaNet(_) | LayerWeights::DeltaNetMoe(_),
        ))
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
            // MoE batched path: LA/FA projections must be MQ4 + every
            // routed/shared MoE weight must be MQ4. Top-K=8 and the
            // scratch tensors must exist on `pbs`.
            LayerWeights::DeltaNetMoe(l) =>
                moe_topk_ok
                    && pbs_in.map(|p| p.moe_router_logits_batch.is_some()).unwrap_or(true)
                    && is_batchable_la(l.wqkv.gpu_dtype)
                    && is_batchable_la(l.wz.gpu_dtype)
                    && is_batchable_la(l.w_beta.gpu_dtype)
                    && is_batchable_la(l.w_alpha.gpu_dtype)
                    && is_batchable_la(l.wo.gpu_dtype)
                    && moe_ffn_all_mq4(&l.ffn),
            LayerWeights::FullAttnMoe(l) =>
                moe_topk_ok
                    && pbs_in.map(|p| p.moe_router_logits_batch.is_some()).unwrap_or(true)
                    && is_batchable_la(l.wq.gpu_dtype)
                    && is_batchable_la(l.wk.gpu_dtype)
                    && is_batchable_la(l.wv.gpu_dtype)
                    && is_batchable_la(l.wo.gpu_dtype)
                    && moe_ffn_all_mq4(&l.ffn),
        });

    if !eligible {
        assert!(
            tree_verify.is_none(),
            "tree-verify mode requires the batched-FA-eligible prefill path; \
             kv quant + FA weight dtypes do not match on this model",
        );
        // Fallback: per-token loop, byte-identical to decode. If hidden
        // extraction is requested, use the with_hidden variant so the ring
        // buffer still gets populated correctly (each call advances head by 1).
        // When per-token hidden output is also requested, extract post-norm
        // hidden row-by-row into the caller's buffer.
        let dim = config.dim;
        for (i, &tok) in tokens.iter().enumerate() {
            if let Some(rb) = hidden_rb.as_mut() {
                forward_scratch_with_hidden(
                    gpu, weights, config, tok, start_pos + i,
                    kv_cache, dn_state, scratch, rb,
                )?;
            } else {
                forward_scratch(gpu, weights, config, tok, start_pos + i, kv_cache, dn_state, scratch)?;
            }
            if let Some(dst) = per_token_hidden_out {
                // scratch.tmp holds post-output-norm hidden after
                // forward_scratch_{with_hidden,layers} — it's the same buffer
                // lm_head reads from. Copy into the caller's output.
                gpu.hip.memcpy_dtod_at(
                    &dst.buf, i * dim * 4,
                    &scratch.tmp.buf, 0,
                    dim * 4,
                )?;
            }
        }
        return Ok(());
    }

    // Tree-verify mode runs as a single chunk (tree is small, O(16) nodes);
    // chunk splitting would require slicing the mask by chunk rows which
    // is extra work for a case we don't need.
    if tree_verify.is_some() {
        assert!(
            n <= max_batch,
            "tree-verify tokens {} exceeds max_batch {}; tree budget must fit",
            n, max_batch,
        );
    }

    // Allocate the batch scratch once per call (or reuse a caller-owned one).
    // When `pbs_in` is Some, we neither allocate nor free — the caller retains
    // ownership across DFlash cycles to avoid ~25 per-cycle tensor alloc/free
    // pairs on the hot verify path. When None we fall back to the original
    // allocate-here / free-on-exit pattern so unmodified callers behave the
    // same. The chunk size is `pbs.max_batch` so a caller-owned scratch sized
    // to e.g. `block_size` or `1 + tree_budget` keeps DFlash verify in one
    // chunk without the full 256-row MAX_BATCH footprint.
    let mut own_pbs: Option<PrefillBatchScratch> = None;
    let result = (|| -> HipResult<()> {
        let pbs: &PrefillBatchScratch = match pbs_in {
            Some(p) => p,
            None => {
                own_pbs = Some(PrefillBatchScratch::new(gpu, config, max_batch)?);
                own_pbs.as_ref().unwrap()
            }
        };
        let chunk_batch = pbs.max_batch;
        let mut chunk_start = 0usize;
        while chunk_start < n {
            let chunk_end = (chunk_start + chunk_batch).min(n);
            let chunk = &tokens[chunk_start..chunk_end];
            let chunk_n = chunk.len();
            // The chunk only reads the ring buffer's head/dims to place its
            // writes. We advance the head AFTER the chunk returns, here in
            // the caller, to keep the mutable borrow scope tight.
            let pth_slot = per_token_hidden_out.map(|t| (t, chunk_start));
            // Reborrow the tape for this chunk so we keep the outer mut
            // after the chunk returns.
            let tape_for_chunk: Option<&mut crate::speculative::GdnTape> =
                gdn_tape.as_mut().map(|t| &mut **t);
            // Tree-verify was asserted to fit in one chunk above, so passing
            // the whole ctx through unconditionally is safe.
            let tv_for_chunk = tree_verify.as_ref().copied();
            forward_prefill_chunk(
                gpu, weights, config, chunk, start_pos + chunk_start,
                kv_cache, dn_state, scratch, pbs, hidden_rb.as_deref(),
                pth_slot, tape_for_chunk, chunk_start, tv_for_chunk,
                false, // pre_uploaded: default path uploads inside
            )?;
            if let Some(rb) = hidden_rb.as_mut() {
                // Scatter fixed-offset staging writes (done inside the chunk)
                // to the ring at the current head, then advance head by n.
                // This is the out-of-capture step: graph-captured writes went
                // to staging[0..n*h], this commit places them at head*h
                // where head is read from CPU state at call time (not baked
                // into a captured graph node).
                rb.commit_staging_to_ring(gpu, chunk_n)?;
            }
            chunk_start = chunk_end;
        }
        Ok(())
    })();
    if let Some(owned) = own_pbs {
        owned.free_gpu(gpu);
    }
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
///
/// `hidden_rb`: if `Some`, post-layer residual hidden states for configured
/// extract layers get written into the ring buffer at its current head. The
/// caller (forward_prefill_batch) advances the head by N after this chunk
/// completes so writes from the next chunk don't overwrite.
///
/// `per_token_hidden_out`: if `Some((dst, offset_rows))`, writes post-output
/// RMSNorm hidden for each of the N tokens into `dst[offset_rows..offset_rows+N]`
/// in row-major order. Required for DFlash verify to compute per-position
/// logits via B sequential `weight_gemv` calls on the caller side.
///
/// `gdn_tape` + `tape_offset`: if `Some`, captures the post-processed
/// `(q, k, v, α, β)` tensors per DN layer at rows
/// `[tape_offset .. tape_offset+N]` right before the batched GDN kernel
/// runs. Used by the DFlash rollback path.
/// Is every weight inside a MoE FFN MQ4G256? Gates the batched fast path —
/// the router + shared-gate + shared.{gate,up,down} + every expert gate_up
/// + every expert down must be MQ4 for the batched kernels to apply
/// (they all assume HFQ4-G256 binary layout and group stride 136).
fn moe_ffn_all_mq4(ffn: &MoeFfnWeights) -> bool {
    ffn.router.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert_gate.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert.gate.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert.up.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert.down.gpu_dtype == DType::MQ4G256
        && ffn.experts.iter().all(|e|
            e.gate_up.gpu_dtype == DType::MQ4G256 && e.down.gpu_dtype == DType::MQ4G256)
}

/// Batched MoE FFN for `forward_prefill_chunk`. Takes the post-attention
/// residual stream in `pbs.x_batch` ([N × dim]) and writes the FFN output
/// residual back into the same buffer in-place.
///
/// Preconditions (caller must guarantee):
/// - all MoE weights are MQ4G256 (see `moe_ffn_all_mq4`)
/// - `pbs.moe_*_batch` tensors are allocated (num_experts > 0 at scratch
///   construction time) and sized to max_batch ≥ N
/// - `config.num_experts_per_tok == 8` and `config.num_experts <= 1024`
///   (hard limits of the batched top-K kernel)
///
/// Sequence mirrors `moe_ffn_decode_impl`'s GPU fast path, with every
/// per-token launch replaced by its N-batched equivalent. Byte-exact
/// except for atomicAdd nondeterminism in the routed-down accumulation
/// (same as the single-token indexed kernel it replaces).
fn prefill_moe_ffn_body_batched(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    ffn_norm: &GpuTensor,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    n: usize,
) -> HipResult<()> {
    let dim = config.dim;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let k_top = config.num_experts_per_tok;
    let n_exp = config.num_experts;

    let router_logits = pbs.moe_router_logits_batch.as_ref().expect("moe scratch");
    let shared_scalar = pbs.moe_shared_scalar_batch.as_ref().expect("moe scratch");
    let shared_gate   = pbs.moe_shared_gate_batch.as_ref().expect("moe scratch");
    let shared_up     = pbs.moe_shared_up_batch.as_ref().expect("moe scratch");
    let shared_rot    = pbs.moe_shared_rot_batch.as_ref().expect("moe scratch");
    let topk_indices  = pbs.moe_topk_indices_batch.as_ref().expect("moe scratch");
    let topk_weights  = pbs.moe_topk_weights_batch.as_ref().expect("moe scratch");
    let gate_batch    = pbs.moe_gate_batch.as_ref().expect("moe scratch");
    let up_batch      = pbs.moe_up_batch.as_ref().expect("moe scratch");
    let rot_batch     = pbs.moe_rot_batch.as_ref().expect("moe scratch");

    // ── 1. rmsnorm + FWHT pre-rotate for MQ4 inputs ──
    gpu.fused_rmsnorm_rotate_mq_batched(
        &pbs.x_batch, ffn_norm, &pbs.x_rot_batch, dim, config.norm_eps, n,
    )?;

    // ── 2. Router + shared-gate + shared.gate + shared.up (4 batched GEMMs) ──
    //
    // The natural fit is `gemm_qkvza_hfq4g256` (4-way fused with one
    // batched launch), but on gfx11+ it routes to a WMMA fast path whose
    // 16×16 tiling breaks at the z_m=1 boundary row (the shared-expert
    // gate is a single row, sandwiched between the 256-row router and
    // the 512-row shared.gate). Symptom was τ≈0 with repeating tokens
    // once the batched MoE path was enabled.
    //
    // Four separate `gemm_hfq4g256` calls hit the portable scalar
    // kernel (no WMMA), which stays byte-exact with the reference.
    // Launch-count cost is +3 per MoE layer; acceptable for correctness.
    // Follow-up: fix the WMMA qkvza to handle z_m=1 and re-fuse.
    gpu.gemm_hfq4g256(
        &ffn.router.buf, &pbs.x_rot_batch, router_logits,
        ffn.router.m, ffn.router.k, n,
    )?;
    gpu.gemm_hfq4g256(
        &ffn.shared_expert_gate.buf, &pbs.x_rot_batch, shared_scalar,
        ffn.shared_expert_gate.m, ffn.shared_expert_gate.k, n,
    )?;
    gpu.gemm_hfq4g256(
        &ffn.shared_expert.gate.buf, &pbs.x_rot_batch, shared_gate,
        ffn.shared_expert.gate.m, ffn.shared_expert.gate.k, n,
    )?;
    gpu.gemm_hfq4g256(
        &ffn.shared_expert.up.buf, &pbs.x_rot_batch, shared_up,
        ffn.shared_expert.up.m, ffn.shared_expert.up.k, n,
    )?;

    // ── 3. GPU softmax + top-K + renorm, batched over N tokens ──
    gpu.moe_softmax_topk_renorm_k8_batched(
        router_logits, topk_indices, topk_weights,
        n_exp, config.norm_topk_prob, n,
    )?;

    // ── 4. Shared-expert SwiGLU + FWHT, batched over N tokens ──
    //
    // fused_silu_mul_rotate_mq_batched expects [batch × k] gate/up with
    // batch on grid.y and writes FWHT(silu(gate) * up) into x_rot. Here
    // batch=N, k=smi; the shared-rot output buffer is [N × smi].
    gpu.fused_silu_mul_rotate_mq_batched(shared_gate, shared_up, shared_rot, smi, n)?;

    // ── 5. Shared-expert down with sigmoid-scaled residual, batched ──
    //
    // Reads shared_scalar[token] as the pre-sigmoid logit, applies sigmoid
    // internally, and atomicAdd's sigmoid(scalar) × (W_down · rot) into
    // pbs.x_batch[token × dim + row].
    gpu.gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched(
        &ffn.shared_expert.down.buf, shared_rot, &pbs.x_batch, shared_scalar,
        ffn.shared_expert.down.m, ffn.shared_expert.down.k, n,
    )?;

    // ── 6. Routed experts: batched gate_up → SwiGLU+FWHT → down ──
    //
    // Gate/up for top-K experts (per token) → [N × K_TOP × mi]. Each
    // output row reads topk_indices[token × K_TOP + krank] to pick its
    // expert weight base from the device-side expert_gate_up_ptrs table.
    let down_m = ffn.experts[0].down.m;
    let down_k = ffn.experts[0].down.k;
    let gate_up_k = ffn.experts[0].gate_up.k;
    gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
        &ffn.expert_gate_up_ptrs, topk_indices,
        &pbs.x_rot_batch, gate_batch, up_batch,
        2 * mi, gate_up_k, k_top, n,
    )?;

    // SwiGLU + FWHT over [N*K_TOP × mi] — batch flatten across tokens and
    // expert ranks, k=mi is per-row width.
    gpu.fused_silu_mul_rotate_mq_batched(gate_batch, up_batch, rot_batch, mi, n * k_top)?;

    // Down projection with per-(token, expert) scaling and atomic
    // residual-add into pbs.x_batch.
    gpu.gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched(
        &ffn.expert_down_ptrs, topk_indices, topk_weights,
        rot_batch, &pbs.x_batch,
        down_m, down_k, k_top, n,
    )?;

    Ok(())
}

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
    hidden_rb: Option<&HiddenStateRingBuffer>,
    per_token_hidden_out: Option<(&GpuTensor, usize)>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tape_offset: usize,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    pre_uploaded: bool,
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

    // ── 1. Embed tokens into pbs.x_batch ─────────────────────────────────
    //
    // Fast path for HFQ4G256 (all MQ4-quantized Qwen3.5 models + friends):
    // upload token ids to a device buffer and dispatch one batched kernel
    // that dequantizes N rows directly into `pbs.x_batch`. This collapses
    // 2N launches (N embed + N memcpy_dtod_at) into 1 upload + 1 launch
    // AND is hipGraph-captureable — the kernel reads token ids from a
    // device pointer instead of taking them as a baked-in scalar arg.
    //
    // Other formats fall back to the per-token loop (kept for correctness
    // breadth; the MQ4-quantized hot path doesn't hit them).
    if matches!(weights.embd_format, EmbeddingFormat::HFQ4G256 | EmbeddingFormat::Q8_0) {
        if !pre_uploaded {
            let tokens_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            let tokens_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4)
            };
            gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;
        }
        match weights.embd_format {
            EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256_batched(&weights.token_embd, &pbs.x_batch, &pbs.tokens, n, dim)?;
            }
            EmbeddingFormat::Q8_0 => {
                gpu.embedding_lookup_q8_batched(&weights.token_embd, &pbs.x_batch, &pbs.tokens, n, dim)?;
            }
            _ => unreachable!(),
        }
    } else {
        for (i, &tok) in tokens.iter().enumerate() {
            match weights.embd_format {
                EmbeddingFormat::HFQ4G256 => unreachable!(),
                EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.token_embd, &s.x, tok, dim)?,
                EmbeddingFormat::Q8_0     => gpu.embedding_lookup_q8(&weights.token_embd, &s.x, tok, dim)?,
                EmbeddingFormat::F32      => gpu.embedding_lookup(&weights.token_embd, &s.x, tok, dim)?,
                _ => panic!("unsupported embedding format"),
            }
            gpu.hip.memcpy_dtod_at(&pbs.x_batch.buf, i * dim_row_bytes, &s.x.buf, 0, dim_row_bytes)?;
        }
    }

    // ── 1b. Upload positions array ────────────────────────────────────────
    //
    // Positions is the per-row RoPE angle AND the physical KV cache slot (the
    // batched kv_write kernels use the same index for both). We always use
    // flat linear `start_pos .. start_pos + n`. Siblings in DDTree mode get
    // DISTINCT slots this way — no write race — and the stored K carries a
    // RoPE angle that matches the physical slot, which keeps subsequent
    // cycles' attention reads consistent.
    //
    // Semantic trade vs. the original depth-based scheme (paper): tree
    // siblings that represent "alternative futures at the same time step"
    // now see a RoPE distance of 1 (or more) instead of 0. Empirically that
    // slight distance shift costs little — the attn_bias mask still gates
    // ancestor visibility exactly, and the Q·K dot products stay consistent
    // across the whole cache (prompt + tree block). In exchange we get
    // DDTree correctness for topk>1 without needing a tree-local KV scratch
    // or a scatter-kernel for commit. `ctx.positions` is accepted for API
    // compatibility but ignored — the DdNode depths it carries are only
    // used by `linearize_tree` to build the attn_bias mask.
    if !pre_uploaded {
        let positions_host: Vec<i32> = (0..n).map(|i| (start_pos + i) as i32).collect();
        let positions_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4)
        };
        gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;
    }

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
            // MoE variant: attention weights must be MQ4-class (FFN is
            // checked separately by moe_ffn_all_mq4 in the eligibility gate).
            LayerWeights::FullAttnMoe(l) =>
                is_batchable_la(l.wq.gpu_dtype) &&
                is_batchable_la(l.wk.gpu_dtype) &&
                is_batchable_la(l.wv.gpu_dtype) &&
                is_batchable_la(l.wo.gpu_dtype),
            _ => true, // LA layers don't gate this check
        });
    // Under hipGraph capture, scalar kernargs get BAKED into the kernarg blob
    // at capture time. `max_ctx_len = start_pos + n` grows per cycle, so the
    // captured value would be stale on replay — the attention kernel would
    // allocate too-small LDS for `scores[]` and over-read. Bake the physical
    // cap instead (LDS sized for the worst case). The kernel still iterates
    // over the actual `positions[b] + 1` per-row seq_len from a device buffer,
    // so correctness is preserved; only the LDS allocation is over-provisioned.
    let max_ctx_len = if gpu.capture_mode {
        kv_cache.physical_cap
    } else {
        start_pos + n
    };

    // ── 2. Per-layer loop ────────────────────────────────────────────────
    let mut delta_layer_idx = 0usize;
    let mut kv_layer_idx = 0usize;
    // Path B: per-FA-layer counter, drives the index into
    // tree_verify.pre_rope_k_capture[]. Increments alongside each
    // FullAttention layer iteration regardless of MoE/non-MoE variant.
    let mut fa_layer_idx = 0usize;

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

                // DFlash tape capture: snap pre-conv1d qkv + post-sigmoid α/β
                // for this layer into the per-layer tape slots. The next LA
                // layer's fused_qkvza / fused_sigmoid_alpha_gate will overwrite
                // dn_qkv_batch / dn_{alpha,beta}_batch, so capture must happen
                // now (after sigmoid_alpha_gate, before conv1d consumes qkv).
                if let Some(tape) = gdn_tape.as_ref() {
                    let qkv_row_bytes = tape.qkv_dim * 4;
                    let alpha_row_bytes = n_v_heads * 4;
                    let off_qkv = tape_offset * qkv_row_bytes;
                    let off_a = tape_offset * alpha_row_bytes;
                    let copy_qkv = n * qkv_row_bytes;
                    let copy_a = n * alpha_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.qkv_bufs[delta_layer_idx].buf, off_qkv,
                        &pbs.dn_qkv_batch.buf, 0, copy_qkv,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.alpha_bufs[delta_layer_idx].buf, off_a,
                        &pbs.dn_alpha_batch.buf, 0, copy_a,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.beta_bufs[delta_layer_idx].buf, off_a,
                        &pbs.dn_beta_batch.buf, 0, copy_a,
                    )?;
                }

                // Tree-aware dispatch gate: when the caller provides
                // parent_indices (Phase 3b+ of Task #101), swap the linear
                // conv1d + GDN for tree-walking variants that eliminate
                // sibling-subtree state cross-contamination. The tree
                // kernels are READ-ONLY on dn_state (don't advance it) —
                // caller runs linear replay on the accepted spine
                // post-acceptance to commit the trajectory.
                let tree_parents = tree_verify.as_ref().and_then(|c| c.parent_indices);
                if let Some(parents) = tree_parents {
                    gpu.conv1d_silu_split_tree_f32_n(
                        &pbs.dn_q_raw_batch, &pbs.dn_k_raw_batch, &pbs.dn_v_batch,
                        &pbs.dn_qkv_batch, &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        parents,
                        k_dim, v_dim, n,
                    )?;
                } else {
                    gpu.conv1d_silu_split_f32_n(
                        &pbs.dn_q_raw_batch, &pbs.dn_k_raw_batch, &pbs.dn_v_batch,
                        &pbs.dn_qkv_batch, &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        k_dim, v_dim, n,
                    )?;
                }

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
                    gpu.memcpy_dtod_auto(&pbs.dn_q_batch.buf, &pbs.dn_q_raw_batch.buf, n * k_dim * 4)?;
                    gpu.memcpy_dtod_auto(&pbs.dn_k_batch.buf, &pbs.dn_k_raw_batch.buf, n * k_dim * 4)?;
                }

                // Gated Delta Net — tree variant reads per-token S from
                // s_tape[parent] (or pre-block s_q8_init at root); linear
                // variant advances dn_state.s_matrices in place.
                if let Some(parents) = tree_parents {
                    let tape_q8 = pbs.dn_s_tape_q8.as_ref()
                        .expect("tree-aware LA requires dn_s_tape_q8 scratch (check PrefillBatchScratch::new)");
                    let tape_sc = pbs.dn_s_tape_scales.as_ref()
                        .expect("tree-aware LA requires dn_s_tape_scales scratch (check PrefillBatchScratch::new)");
                    gpu.gated_delta_net_q8_tree_batch_seq(
                        &pbs.dn_q_batch, &pbs.dn_k_batch, &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch, &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        tape_q8, tape_sc, parents,
                        &pbs.dn_attn_out_batch,
                        n, n_v_heads, config.linear_value_head_dim,
                    )?;
                } else {
                    gpu.gated_delta_net_q8_batch_seq(
                        &pbs.dn_q_batch, &pbs.dn_k_batch, &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch, &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        &pbs.dn_attn_out_batch,
                        n, n_v_heads, config.linear_value_head_dim,
                    )?;
                }

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

                // Post-layer hidden extract for the DFlash draft path.
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
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

                if crate::triattn::tap_enabled() {
                    // Try GPU path first: dispatches a reduce kernel on the
                    // device-resident Q tensor, zero PCIe transfer. Only
                    // succeeds when install_tap_gpu() was used. Falls through
                    // to CPU path otherwise.
                    let gpu_handled = crate::triattn::record_prerope_q_batch_gpu_if_applicable(
                        gpu, layer_idx, &pbs.fa_q_batch.buf,
                        n, config.n_heads, config.head_dim,
                    )?;
                    if !gpu_handled {
                        let n_q = config.n_heads * config.head_dim;
                        let q_cpu = gpu.download_f32(&pbs.fa_q_batch)?;
                        if crate::triattn::tap_needs_k() {
                            let n_k = config.n_kv_heads * config.head_dim;
                            let k_cpu = gpu.download_f32(&pbs.fa_k_batch)?;
                            for b in 0..n {
                                crate::triattn::record_prerope_qk(
                                    layer_idx,
                                    &q_cpu[b * n_q..(b + 1) * n_q],
                                    Some(&k_cpu[b * n_k..(b + 1) * n_k]),
                                );
                            }
                        } else {
                            for b in 0..n {
                                crate::triattn::record_prerope_q(
                                    layer_idx,
                                    &q_cpu[b * n_q..(b + 1) * n_q],
                                );
                            }
                        }
                    }
                }

                // Path B pre-RoPE K capture (slow-path-kill, WIP).
                // The next line mutates pbs.fa_k_batch in place — capture
                // BEFORE so the slow path has the unrotated K available
                // and can apply RoPE for the COMMITTED slot phases instead
                // of these linearization-slot phases. Capture is None
                // unless the env gate + the per-FA-layer scratch are both
                // wired through TreeVerifyCtx.
                if let Some(slots) = tree_verify.as_ref().and_then(|c| c.pre_rope_k_capture) {
                    if let Some(slot) = slots.get(fa_layer_idx) {
                        let kv_dim = config.n_kv_heads * config.head_dim;
                        let n_bytes = n * kv_dim * 4;
                        // Use _auto so the memcpy is recorded onto the
                        // active stream when one exists (matches the
                        // existing GdnTape capture pattern at line ~3193).
                        // Plain gpu.hip.memcpy_dtod_at runs on the null
                        // stream and sync-blocks pending async kernels,
                        // changing kernel-launch order in ways that
                        // perturb DDTree's ksplit-atomic nondeterminism
                        // — output diverges even though no data is
                        // actually changed.
                        gpu.memcpy_dtod_at_auto(
                            &slot.buf, 0,
                            &pbs.fa_k_batch.buf, 0,
                            n_bytes,
                        )?;
                    }
                }

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

                // 7. Batched causal attention (or tree-attention if tree_verify is set).
                // asym{4,3,2}: batched flash (K rotated-quantized + V Q8 in normal space).
                // Q8: batched kernel unless ctx > 15K (LDS overflow), then per-position flash.
                //
                // Tree-verify mode: `block_start = start_pos`, `block_cols = n`.
                // The bias buffer is `[n × n]`; each query row applies its
                // corresponding bias row to in-block keys. Long-context Q8
                // tiled fallback isn't supported in tree mode (we caught
                // that as an assert above — tree blocks are small).
                const LDS_CTX_LIMIT: usize = 15000;
                let tree_bias = tree_verify.as_ref().map(|c| c.attn_bias);
                let (block_start, block_cols) = match tree_verify.as_ref() {
                    Some(_) => (start_pos, n),
                    None => (0, 0),
                };
                if kv_cache.quant_asym4 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    gpu.attention_flash_asym4_batched_masked(
                        &pbs.fa_q_batch, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch, &pbs.positions, ct, st,
                        config.n_heads, config.n_kv_heads, config.head_dim,
                        kv_cache.physical_cap, max_ctx_len, n, &s.flash_partials,
                        tree_bias, block_start, block_cols,
                    )?;
                } else if kv_cache.quant_asym3 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    gpu.attention_flash_asym3_batched_masked(
                        &pbs.fa_q_batch, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch, &pbs.positions, ct, st,
                        config.n_heads, config.n_kv_heads, config.head_dim,
                        kv_cache.physical_cap, max_ctx_len, n, &s.flash_partials,
                        tree_bias, block_start, block_cols,
                    )?;
                } else if kv_cache.quant_asym2 {
                    assert!(
                        tree_verify.is_none(),
                        "tree-verify mode not supported on asym2 KV (use asym3)",
                    );
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    gpu.attention_flash_asym2_batched(
                        &pbs.fa_q_batch, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch, &pbs.positions, ct, st,
                        config.n_heads, config.n_kv_heads, config.head_dim,
                        kv_cache.physical_cap, max_ctx_len, n, &s.flash_partials,
                    )?;
                } else if max_ctx_len > LDS_CTX_LIMIT {
                    assert!(
                        tree_verify.is_none(),
                        "tree-verify mode hits the long-context Q8 fallback \
                         at max_ctx_len={} > {}; tree blocks should stay small",
                        max_ctx_len, LDS_CTX_LIMIT,
                    );
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
                            kv_cache.physical_cap, &s.flash_partials,
                        )?;
                    }
                    let _ = gpu.hip.free(pos_buf_tmp);
                } else {
                    gpu.attention_q8_0_kv_batched_masked(
                        &pbs.fa_q_batch,
                        &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch, &pbs.positions,
                        config.n_heads, config.n_kv_heads, config.head_dim,
                        kv_cache.physical_cap, max_ctx_len, n,
                        tree_bias, block_start, block_cols,
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

                // Post-layer hidden extract for the DFlash draft path.
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }

                // Silence unused warning if kv_dim ends up shadowed.
                let _ = kv_dim;
                kv_layer_idx += 1;
                fa_layer_idx += 1;
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

                // Post-layer hidden extract for the DFlash draft path. After
                // the per-token loop, pbs.x_batch has the full layer output
                // for all N tokens (last copy-back finishes each row).
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }

                kv_layer_idx += 1;
                fa_layer_idx += 1;
            }

            (LayerWeights::DeltaNetMoe(layer), LayerType::LinearAttention) => {
                // Batched MoE LA layer. LA body is the same as DeltaNet
                // (rmsnorm + qkvza + sigmoid_alpha + conv1d + L2norm +
                // repeat_interleave + GDN + gated_norm + wo+residual);
                // only the FFN differs. Duplicated inline for now — can
                // be factored into a `prefill_la_body_batched` helper
                // when dense and MoE LA paths are proven byte-exact.
                let is_mq = matches!(layer.wqkv.gpu_dtype, DType::MQ4G256 | DType::MQ6G256);
                let is_6bit = matches!(layer.wqkv.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);

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
                gpu.fused_sigmoid_alpha_gate_f32_batched(
                    &pbs.dn_beta_batch, &pbs.dn_alpha_batch,
                    &layer.dt_bias, &layer.a_log,
                    n_v_heads, n,
                )?;
                if let Some(tape) = gdn_tape.as_ref() {
                    let qkv_row_bytes = tape.qkv_dim * 4;
                    let alpha_row_bytes = n_v_heads * 4;
                    let off_qkv = tape_offset * qkv_row_bytes;
                    let off_a = tape_offset * alpha_row_bytes;
                    let copy_qkv = n * qkv_row_bytes;
                    let copy_a = n * alpha_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.qkv_bufs[delta_layer_idx].buf, off_qkv,
                        &pbs.dn_qkv_batch.buf, 0, copy_qkv,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.alpha_bufs[delta_layer_idx].buf, off_a,
                        &pbs.dn_alpha_batch.buf, 0, copy_a,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.beta_bufs[delta_layer_idx].buf, off_a,
                        &pbs.dn_beta_batch.buf, 0, copy_a,
                    )?;
                }
                // Same tree-aware dispatch gate as dense LA branch above.
                let tree_parents = tree_verify.as_ref().and_then(|c| c.parent_indices);
                if let Some(parents) = tree_parents {
                    gpu.conv1d_silu_split_tree_f32_n(
                        &pbs.dn_q_raw_batch, &pbs.dn_k_raw_batch, &pbs.dn_v_batch,
                        &pbs.dn_qkv_batch, &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        parents,
                        k_dim, v_dim, n,
                    )?;
                } else {
                    gpu.conv1d_silu_split_f32_n(
                        &pbs.dn_q_raw_batch, &pbs.dn_k_raw_batch, &pbs.dn_v_batch,
                        &pbs.dn_qkv_batch, &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        k_dim, v_dim, n,
                    )?;
                }
                gpu.fused_qk_l2_norm_scale_f32_batched(
                    &pbs.dn_q_raw_batch, &pbs.dn_k_raw_batch,
                    config.linear_num_key_heads, hd,
                    1.0 / (hd as f32).sqrt(), config.norm_eps, n,
                )?;
                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    gpu.repeat_interleave_qk_f32_batched(
                        &pbs.dn_q_raw_batch, &pbs.dn_k_raw_batch,
                        &pbs.dn_q_batch, &pbs.dn_k_batch,
                        config.linear_num_key_heads, ratio, hd, n,
                    )?;
                } else {
                    gpu.memcpy_dtod_auto(&pbs.dn_q_batch.buf, &pbs.dn_q_raw_batch.buf, n * k_dim * 4)?;
                    gpu.memcpy_dtod_auto(&pbs.dn_k_batch.buf, &pbs.dn_k_raw_batch.buf, n * k_dim * 4)?;
                }
                if let Some(parents) = tree_parents {
                    let tape_q8 = pbs.dn_s_tape_q8.as_ref()
                        .expect("tree-aware LA requires dn_s_tape_q8 scratch");
                    let tape_sc = pbs.dn_s_tape_scales.as_ref()
                        .expect("tree-aware LA requires dn_s_tape_scales scratch");
                    gpu.gated_delta_net_q8_tree_batch_seq(
                        &pbs.dn_q_batch, &pbs.dn_k_batch, &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch, &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        tape_q8, tape_sc, parents,
                        &pbs.dn_attn_out_batch,
                        n, n_v_heads, config.linear_value_head_dim,
                    )?;
                } else {
                    gpu.gated_delta_net_q8_batch_seq(
                        &pbs.dn_q_batch, &pbs.dn_k_batch, &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch, &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        &pbs.dn_attn_out_batch,
                        n, n_v_heads, config.linear_value_head_dim,
                    )?;
                }
                gpu.gated_norm_f32_batched(
                    &pbs.dn_attn_out_batch, &pbs.dn_z_batch, &layer.norm_weight,
                    &pbs.dn_normed_batch,
                    n_v_heads, config.linear_value_head_dim, config.norm_eps, n,
                )?;
                // wo + residual. Eligibility gate ensured layer.wo is MQ4.
                gpu.rotate_x_mq_batched(
                    &pbs.dn_normed_batch, &pbs.dn_normed_rot_batch, layer.wo.k, n,
                )?;
                gpu.gemm_hfq4g256_residual(
                    &layer.wo.buf, &pbs.dn_normed_rot_batch, &pbs.x_batch,
                    layer.wo.m, layer.wo.k, n,
                )?;

                // Batched MoE FFN replaces the dense (rmsnorm + gate+up +
                // silu_mul + w_down) block. Takes pbs.x_batch as input AND
                // accumulates the FFN output residual back into it via the
                // batched indexed down kernel's atomicAdd path.
                prefill_moe_ffn_body_batched(gpu, &layer.ffn, &layer.ffn_norm, config, pbs, n)?;

                // Post-layer hidden extract for the DFlash draft path.
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }
                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttnMoe(layer), LayerType::FullAttention) if fa_batched_ok => {
                // Batched MoE FA layer. FA body is the same as FullAttn
                // (rmsnorm + qkv + deinterleave + q/k norm + RoPE +
                // kv_write + attention + sigmoid_mul + wo+residual);
                // only the FFN differs. Duplicated inline — will be
                // consolidated with the dense FA batched body once the
                // MoE path is proven byte-exact.
                let kv_dim = config.n_kv_heads * config.head_dim;
                let q_dim = config.n_heads * config.head_dim;
                let qkv_is_mq = matches!(layer.wq.gpu_dtype, DType::MQ4G256 | DType::MQ6G256);
                let qkv_is_6bit = matches!(layer.wq.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);

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
                gpu.deinterleave_f32_batched(
                    &pbs.fa_q_full_batch, &pbs.fa_q_batch, &pbs.fa_gate_batch,
                    config.n_heads, config.head_dim, n,
                )?;
                gpu.rmsnorm_batched(
                    &pbs.fa_q_batch, &layer.q_norm, &pbs.fa_q_batch,
                    n * config.n_heads, config.head_dim, config.norm_eps,
                )?;
                gpu.rmsnorm_batched(
                    &pbs.fa_k_batch, &layer.k_norm, &pbs.fa_k_batch,
                    n * config.n_kv_heads, config.head_dim, config.norm_eps,
                )?;
                if crate::triattn::tap_enabled() {
                    let gpu_handled = crate::triattn::record_prerope_q_batch_gpu_if_applicable(
                        gpu, layer_idx, &pbs.fa_q_batch.buf,
                        n, config.n_heads, config.head_dim,
                    )?;
                    if !gpu_handled {
                        let n_q = config.n_heads * config.head_dim;
                        let q_cpu = gpu.download_f32(&pbs.fa_q_batch)?;
                        if crate::triattn::tap_needs_k() {
                            let n_k = config.n_kv_heads * config.head_dim;
                            let k_cpu = gpu.download_f32(&pbs.fa_k_batch)?;
                            for b in 0..n {
                                crate::triattn::record_prerope_qk(
                                    layer_idx,
                                    &q_cpu[b * n_q..(b + 1) * n_q],
                                    Some(&k_cpu[b * n_k..(b + 1) * n_k]),
                                );
                            }
                        } else {
                            for b in 0..n {
                                crate::triattn::record_prerope_q(
                                    layer_idx,
                                    &q_cpu[b * n_q..(b + 1) * n_q],
                                );
                            }
                        }
                    }
                }
                // Path B pre-RoPE K capture (MoE FA variant). See same
                // block in the FullAttn branch for rationale.
                if let Some(slots) = tree_verify.as_ref().and_then(|c| c.pre_rope_k_capture) {
                    if let Some(slot) = slots.get(fa_layer_idx) {
                        let kv_dim = config.n_kv_heads * config.head_dim;
                        let n_bytes = n * kv_dim * 4;
                        gpu.memcpy_dtod_at_auto(
                            &slot.buf, 0,
                            &pbs.fa_k_batch.buf, 0,
                            n_bytes,
                        )?;
                    }
                }
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                gpu.rope_partial_interleaved_f32_batched(
                    &pbs.fa_q_batch, &pbs.fa_k_batch, &pbs.positions,
                    config.n_heads, config.n_kv_heads, config.head_dim, n_rot,
                    config.rope_theta, n,
                )?;
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
                const LDS_CTX_LIMIT: usize = 15000;
                let tree_bias = tree_verify.as_ref().map(|c| c.attn_bias);
                let (block_start, block_cols) = match tree_verify.as_ref() {
                    Some(_) => (start_pos, n),
                    None => (0, 0),
                };
                if kv_cache.quant_asym4 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    gpu.attention_flash_asym4_batched_masked(
                        &pbs.fa_q_batch, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch, &pbs.positions, ct, st,
                        config.n_heads, config.n_kv_heads, config.head_dim,
                        kv_cache.physical_cap, max_ctx_len, n, &s.flash_partials,
                        tree_bias, block_start, block_cols,
                    )?;
                } else if kv_cache.quant_asym3 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    gpu.attention_flash_asym3_batched_masked(
                        &pbs.fa_q_batch, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch, &pbs.positions, ct, st,
                        config.n_heads, config.n_kv_heads, config.head_dim,
                        kv_cache.physical_cap, max_ctx_len, n, &s.flash_partials,
                        tree_bias, block_start, block_cols,
                    )?;
                } else if kv_cache.quant_asym2 {
                    assert!(
                        tree_verify.is_none(),
                        "tree-verify mode not supported on asym2 KV (use asym3)",
                    );
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    gpu.attention_flash_asym2_batched(
                        &pbs.fa_q_batch, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch, &pbs.positions, ct, st,
                        config.n_heads, config.n_kv_heads, config.head_dim,
                        kv_cache.physical_cap, max_ctx_len, n, &s.flash_partials,
                    )?;
                } else if max_ctx_len > LDS_CTX_LIMIT {
                    assert!(
                        tree_verify.is_none(),
                        "tree-verify mode hits the long-context Q8 fallback \
                         at max_ctx_len={} > {}; tree blocks should stay small",
                        max_ctx_len, LDS_CTX_LIMIT,
                    );
                    let q_dim_local = config.n_heads * config.head_dim;
                    let pos_host = gpu.download_f32(&pbs.positions)?;
                    let pos_buf_tmp = gpu.hip.malloc(4)?;
                    for b in 0..n {
                        let seq_len_b = pos_host[b] as usize + 1;
                        let pos_i32 = pos_host[b] as i32;
                        gpu.hip.memcpy_htod(&pos_buf_tmp, &pos_i32.to_ne_bytes())?;
                        let q_b = pbs.fa_q_batch.sub_offset(b * q_dim_local, q_dim_local);
                        let out_b = pbs.fa_attn_out_batch.sub_offset(b * q_dim_local, q_dim_local);
                        gpu.attention_flash_q8_0(
                            &q_b, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                            &out_b, &pos_buf_tmp, seq_len_b,
                            config.n_heads, config.n_kv_heads, config.head_dim,
                            kv_cache.physical_cap, &s.flash_partials,
                        )?;
                    }
                    let _ = gpu.hip.free(pos_buf_tmp);
                } else {
                    gpu.attention_q8_0_kv_batched_masked(
                        &pbs.fa_q_batch,
                        &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch, &pbs.positions,
                        config.n_heads, config.n_kv_heads, config.head_dim,
                        kv_cache.physical_cap, max_ctx_len, n,
                        tree_bias, block_start, block_cols,
                    )?;
                }
                gpu.sigmoid_mul_f32(&pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;
                // wo + residual. Eligibility gate ensured layer.wo is MQ4.
                gpu.rotate_x_mq_batched(
                    &pbs.fa_attn_out_batch, &pbs.fa_attn_out_rot_batch, layer.wo.k, n,
                )?;
                gpu.gemm_hfq4g256_residual(
                    &layer.wo.buf, &pbs.fa_attn_out_rot_batch, &pbs.x_batch,
                    layer.wo.m, layer.wo.k, n,
                )?;

                // Batched MoE FFN.
                prefill_moe_ffn_body_batched(gpu, &layer.ffn, &layer.ffn_norm, config, pbs, n)?;

                // Post-layer hidden extract for the DFlash draft path.
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }

                let _ = kv_dim;
                let _ = q_dim;
                kv_layer_idx += 1;
                fa_layer_idx += 1;
            }

            _ => panic!("layer type mismatch at layer {layer_idx}"),
        }
    }

    // ── 3. Final output norm + logits ───────────────────────────────────
    // If the caller requested per-token hidden output (DFlash verify path),
    // run rmsnorm over all N rows into their buffer. Otherwise use the
    // legacy last-token-only path.
    if let Some((dst, offset_rows)) = per_token_hidden_out {
        let dst_view = dst.sub_offset(offset_rows * dim, n * dim);
        gpu.rmsnorm_batched(
            &pbs.x_batch, &weights.output_norm, &dst_view,
            n, dim, config.norm_eps,
        )?;
        // Still populate s.logits with the last-token logits for callers
        // that rely on it (the legacy prefill path's post-condition).
        let last = n - 1;
        let last_view = dst.sub_offset((offset_rows + last) * dim, dim);
        weight_gemv(gpu, &weights.output, &last_view, &s.logits)?;
    } else {
        // Legacy path: only last-token logits.
        let last = n - 1;
        gpu.hip.memcpy_dtod_at(&s.x.buf, 0, &pbs.x_batch.buf, last * dim_row_bytes, dim_row_bytes)?;
        gpu.rmsnorm_f32(&s.x, &weights.output_norm, &s.tmp, config.norm_eps)?;
        weight_gemv(gpu, &weights.output, &s.tmp, &s.logits)?;
    }

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

    if crate::triattn::tap_enabled() {
        // Try GPU path first (matches the batched FA tap at line ~3499 in
        // forward_prefill_batch). When the calibration tap is GPU-resident
        // (CalibrateGpu) we MUST dispatch the kernel here — falling
        // through to record_prerope_qk would either silently drop the
        // sample (pre-Phase-2) or panic (post-Phase-2).
        let gpu_handled = crate::triattn::record_prerope_q_batch_gpu_if_applicable(
            gpu, layer_idx, &s.fa_q.buf,
            1, config.n_heads, config.head_dim,
        )?;
        if !gpu_handled {
            let n_q = config.n_heads * config.head_dim;
            let q_cpu = gpu.download_f32(&s.fa_q)?;
            if crate::triattn::tap_needs_k() {
                let n_k = config.n_kv_heads * config.head_dim;
                let k_cpu = gpu.download_f32(&s.fa_k)?;
                crate::triattn::record_prerope_qk(layer_idx, &q_cpu[..n_q], Some(&k_cpu[..n_k]));
            } else {
                crate::triattn::record_prerope_q(layer_idx, &q_cpu[..n_q]);
            }
        }
    }

    // If TriAttention has compacted the cache, absolute RoPE phase diverges
    // from the physical cache index. Temporarily load the absolute position
    // into pos_buf for the rope call, then restore the physical position
    // for kv_cache_write + flash attention (which both want the write slot).
    if kv_cache.compact_offset > 0 {
        let abs = (pos + kv_cache.compact_offset) as i32;
        gpu.hip.memcpy_htod(&s.pos_buf, &abs.to_ne_bytes())?;
    }
    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
    gpu.rope_partial_interleaved_f32(&s.fa_q, &s.fa_k, &s.pos_buf,
        config.n_heads, config.n_kv_heads, config.head_dim, n_rot, config.rope_theta)?;
    if kv_cache.compact_offset > 0 {
        let phys = pos as i32;
        gpu.hip.memcpy_htod(&s.pos_buf, &phys.to_ne_bytes())?;
    }

    if kv_cache.quant_asym4 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        gpu.kv_cache_write_asym4_fused(
            &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
            &s.fa_k, &s.fa_v, &s.pos_buf, ct, st, config.n_kv_heads, config.head_dim)?;
        gpu.attention_flash_asym4(
            &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
            &s.fa_attn_out, &s.pos_buf, ct, st, pos + 1,
            config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.physical_cap,
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
            config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.physical_cap,
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
            config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.physical_cap,
            &s.flash_partials,
        )?;
    } else if kv_cache.quant_q8 {
        gpu.kv_cache_write_q8_0(&kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf, config.n_kv_heads, config.head_dim)?;
        gpu.kv_cache_write_q8_0(&kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf, config.n_kv_heads, config.head_dim)?;
        gpu.attention_q8_0_kv(
            &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
            &s.fa_attn_out, &s.pos_buf, pos + 1,
            config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.physical_cap,
        )?;
    } else {
        gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf, kv_dim)?;
        gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf, kv_dim)?;
        gpu.attention_f32(
            &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
            &s.fa_attn_out, &s.pos_buf, pos + 1,
            config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.physical_cap,
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
                    // Use the capture-aware auto helper: routes to async on the
                    // active stream when capturing, sync otherwise. The raw
                    // gpu.hip.memcpy_dtod hits "would make the legacy stream
                    // depend on a capturing blocking stream" under hipGraph.
                    gpu.memcpy_dtod_auto(&s.dn_q.buf, &s.dn_q_raw.buf, k_dim * 4)?;
                    gpu.memcpy_dtod_auto(&s.dn_k.buf, &s.dn_k_raw.buf, k_dim * 4)?;
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

                if crate::triattn::tap_enabled() {
                    let gpu_handled = crate::triattn::record_prerope_q_batch_gpu_if_applicable(
                        gpu, layer_idx, &s.fa_q.buf,
                        1, config.n_heads, config.head_dim,
                    )?;
                    if !gpu_handled {
                        let n_q = config.n_heads * config.head_dim;
                        let q_cpu = gpu.download_f32(&s.fa_q)?;
                        if crate::triattn::tap_needs_k() {
                            let n_k = config.n_kv_heads * config.head_dim;
                            let k_cpu = gpu.download_f32(&s.fa_k)?;
                            crate::triattn::record_prerope_qk(layer_idx, &q_cpu[..n_q], Some(&k_cpu[..n_k]));
                        } else {
                            crate::triattn::record_prerope_q(layer_idx, &q_cpu[..n_q]);
                        }
                    }
                }

                if kv_cache.compact_offset > 0 {
                    let abs = (pos + kv_cache.compact_offset) as i32;
                    gpu.hip.memcpy_htod(&s.pos_buf, &abs.to_ne_bytes())?;
                }
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                gpu.rope_partial_interleaved_f32(&s.fa_q, &s.fa_k, &s.pos_buf,
                    config.n_heads, config.n_kv_heads, config.head_dim, n_rot, config.rope_theta)?;
                if kv_cache.compact_offset > 0 {
                    let phys = pos as i32;
                    gpu.hip.memcpy_htod(&s.pos_buf, &phys.to_ne_bytes())?;
                }

                if kv_cache.quant_asym4 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    gpu.kv_cache_write_asym4_fused(
                        &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_k, &s.fa_v, &s.pos_buf, ct, st, config.n_kv_heads, config.head_dim)?;
                    gpu.attention_flash_asym4(
                        &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_attn_out, &s.pos_buf, ct, st, pos + 1,
                        config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.physical_cap,
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
                        config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.physical_cap,
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
                        config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.physical_cap,
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
                            config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.physical_cap,
                            &s.flash_partials,
                        )?;
                    } else {
                        gpu.attention_q8_0_kv(
                            &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out, &s.pos_buf, pos + 1,
                            config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.physical_cap,
                        )?;
                    }
                } else {
                    gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf, kv_dim)?;
                    gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf, kv_dim)?;
                    gpu.attention_f32(
                        &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_attn_out, &s.pos_buf, pos + 1,
                        config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.physical_cap,
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
                    // Capture-aware: see matching path in the GroupQuery layer above.
                    gpu.memcpy_dtod_auto(&s.dn_q.buf, &s.dn_q_raw.buf, k_dim * 4)?;
                    gpu.memcpy_dtod_auto(&s.dn_k.buf, &s.dn_k_raw.buf, k_dim * 4)?;
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
                // Fuse rmsnorm + FWHT-rotate when all MoE weights are MQ4:
                // one `fused_rmsnorm_rotate_mq` kernel writes FWHT(rmsnorm(s.x))
                // directly into `s.moe_x_rot`, replacing the separate
                // `rmsnorm_f32` + internal `rotate_x_mq` pair. When the
                // prerotated flag is set, `moe_ffn_decode_impl` consumes
                // s.x_rot_local only — `x_norm` becomes a dummy on that path.
                if ffn_all_mq4_for_moe(&layer.ffn) {
                    gpu.fused_rmsnorm_rotate_mq(
                        &s.x, &layer.ffn_norm,
                        s.moe_x_rot.as_ref().expect("MoE scratch"),
                        config.dim, config.norm_eps,
                    )?;
                    moe_ffn_decode_with_scratch_prerotated(gpu, &layer.ffn, &s.x, &s.x, config, s)?;
                } else {
                    gpu.rmsnorm_f32(&s.x, &layer.ffn_norm, &s.tmp, config.norm_eps)?;
                    moe_ffn_decode_with_scratch(gpu, &layer.ffn, &s.tmp, &s.x, config, s)?;
                }

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

                if crate::triattn::tap_enabled() {
                    let gpu_handled = crate::triattn::record_prerope_q_batch_gpu_if_applicable(
                        gpu, layer_idx, &s.fa_q.buf,
                        1, config.n_heads, config.head_dim,
                    )?;
                    if !gpu_handled {
                        let n_q = config.n_heads * config.head_dim;
                        let q_cpu = gpu.download_f32(&s.fa_q)?;
                        if crate::triattn::tap_needs_k() {
                            let n_k = config.n_kv_heads * config.head_dim;
                            let k_cpu = gpu.download_f32(&s.fa_k)?;
                            crate::triattn::record_prerope_qk(layer_idx, &q_cpu[..n_q], Some(&k_cpu[..n_k]));
                        } else {
                            crate::triattn::record_prerope_q(layer_idx, &q_cpu[..n_q]);
                        }
                    }
                }

                if kv_cache.compact_offset > 0 {
                    let abs = (pos + kv_cache.compact_offset) as i32;
                    gpu.hip.memcpy_htod(&s.pos_buf, &abs.to_ne_bytes())?;
                }
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                gpu.rope_partial_interleaved_f32(&s.fa_q, &s.fa_k, &s.pos_buf,
                    config.n_heads, config.n_kv_heads, config.head_dim, n_rot, config.rope_theta)?;
                if kv_cache.compact_offset > 0 {
                    let phys = pos as i32;
                    gpu.hip.memcpy_htod(&s.pos_buf, &phys.to_ne_bytes())?;
                }

                if kv_cache.quant_asym4 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    gpu.kv_cache_write_asym4_fused(
                        &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_k, &s.fa_v, &s.pos_buf, ct, st, config.n_kv_heads, config.head_dim)?;
                    gpu.attention_flash_asym4(
                        &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_attn_out, &s.pos_buf, ct, st, pos + 1,
                        config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.physical_cap,
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
                        config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.physical_cap,
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
                        config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.physical_cap,
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
                            config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.physical_cap,
                            &s.flash_partials,
                        )?;
                    } else {
                        gpu.attention_q8_0_kv(
                            &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out, &s.pos_buf, pos + 1,
                            config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.physical_cap,
                        )?;
                    }
                } else {
                    gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf, kv_dim)?;
                    gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf, kv_dim)?;
                    gpu.attention_f32(
                        &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                        &s.fa_attn_out, &s.pos_buf, pos + 1,
                        config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.physical_cap,
                    )?;
                }

                gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
                weight_gemv_residual(gpu, &layer.wo, &s.fa_attn_out, &s.x)?;

                // ── MoE FFN ──
                // Fuse rmsnorm + FWHT-rotate when all MoE weights are MQ4:
                // one `fused_rmsnorm_rotate_mq` kernel writes FWHT(rmsnorm(s.x))
                // directly into `s.moe_x_rot`, replacing the separate
                // `rmsnorm_f32` + internal `rotate_x_mq` pair. When the
                // prerotated flag is set, `moe_ffn_decode_impl` consumes
                // s.x_rot_local only — `x_norm` becomes a dummy on that path.
                if ffn_all_mq4_for_moe(&layer.ffn) {
                    gpu.fused_rmsnorm_rotate_mq(
                        &s.x, &layer.ffn_norm,
                        s.moe_x_rot.as_ref().expect("MoE scratch"),
                        config.dim, config.norm_eps,
                    )?;
                    moe_ffn_decode_with_scratch_prerotated(gpu, &layer.ffn, &s.x, &s.x, config, s)?;
                } else {
                    gpu.rmsnorm_f32(&s.x, &layer.ffn_norm, &s.tmp, config.norm_eps)?;
                    moe_ffn_decode_with_scratch(gpu, &layer.ffn, &s.tmp, &s.x, config, s)?;
                }

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
