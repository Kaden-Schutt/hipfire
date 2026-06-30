// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! ZAYA1 host (f32) weight loader. Reads the **native** converted checkpoint
//! (`convert_zaya_weights_to_hf.py` output: 40 hybrid layers, HF-native tensor
//! names) from a [`hipfire_model::ModelSource`] and dequantizes every tensor to
//! `Vec<f32>` for the CPU reference forward (`forward_cpu`, validated against
//! `golden/zaya_golden.npz`). The GPU/HFQ upload path lands later (Z3+).
//!
//! Tensor-name map (per layer `L`, native layout):
//! - `model.layers.L.input_layernorm.weight`, `post_attention_layernorm.weight`
//! - `self_attn.qkv_proj.{q_proj,k_proj,v_proj_current,v_proj_delayed}.weight`
//! - `self_attn.qkv_proj.conv_qk_{depthwise,grouped}.{weight,bias}`
//! - `self_attn.qk_norm.temp`, `self_attn.o_proj.weight`
//! - `mlp.gate.down_proj.{weight,bias}`, `mlp.gate.router_states_scale` (L>0),
//!   `mlp.gate.router_mlp.{norm.weight,fc1.{weight,bias},fc2.{weight,bias},out_proj.weight}`,
//!   `mlp.gate.balancing_biases`
//! - `mlp.experts.{gate_up_proj,down_proj}` — stacked `[num_experts, out, in]`
//! - `post_{attention,mlp}_residual_scale.{hidden_states,residual}_{scale,bias}`
//! Globals: `model.embed_tokens.weight`, `model.input_hidden_states_{scale,bias}`,
//! `model.norm.weight`; `lm_head.weight` is tied to the embedding when absent.

use crate::ZayaConfig;
use hipfire_model::ModelSource;
use hipfire_primitives::conv::decode_plain_dtype_to_f32;

/// Decode a safetensors tensor's raw bytes to `Vec<f32>` per its dtype string.
fn dequant(dtype: &str, bytes: &[u8]) -> Result<Vec<f32>, String> {
    decode_plain_dtype_to_f32(dtype, bytes).map_err(|e| format!("zaya loader: {e}"))
}

/// Read tensor `name`, dequantized to `Vec<f32>`.
fn get(src: &dyn ModelSource, name: &str) -> Result<Vec<f32>, String> {
    let (info, bytes) = src
        .tensor_data(name)
        .ok_or_else(|| format!("zaya loader: missing tensor {name:?}"))?;
    dequant(&info.dtype, bytes)
}

/// Read tensor `name` and its shape, dequantized to `Vec<f32>`.
fn get_shaped(src: &dyn ModelSource, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
    let (info, bytes) = src
        .tensor_data(name)
        .ok_or_else(|| format!("zaya loader: missing tensor {name:?}"))?;
    Ok((dequant(&info.dtype, bytes)?, info.shape.clone()))
}

/// CCA attention projection weights (host f32, row-major `[out, in]`).
pub struct ZayaCcaWeights {
    /// `[num_heads*head_dim, hidden]`.
    pub q_proj: Vec<f32>,
    /// `[num_kv_heads*head_dim, hidden]`.
    pub k_proj: Vec<f32>,
    /// `[num_kv_heads*head_dim/2, hidden]` — current-token value half.
    pub v_proj_current: Vec<f32>,
    /// `[num_kv_heads*head_dim/2, hidden]` — delayed (prev-token) value half.
    pub v_proj_delayed: Vec<f32>,
    /// Depthwise causal conv over concat(q,k): weight `[conv_ch, 1, k0]`, bias `[conv_ch]`.
    pub conv_qk_depthwise_w: Vec<f32>,
    pub conv_qk_depthwise_b: Vec<f32>,
    /// Grouped causal conv over concat(q,k): weight `[conv_ch, conv_ch/groups, k1]`, bias `[conv_ch]`.
    pub conv_qk_grouped_w: Vec<f32>,
    pub conv_qk_grouped_b: Vec<f32>,
    /// Learned per-kv-head key temperature `[num_kv_heads]`.
    pub qk_norm_temp: Vec<f32>,
    /// `[hidden, num_heads*head_dim]`.
    pub o_proj: Vec<f32>,
}

/// EDA/MoD router weights (host f32).
pub struct ZayaRouterWeights {
    /// `[router_hidden, hidden]` + `[router_hidden]`.
    pub down_proj_w: Vec<f32>,
    pub down_proj_b: Vec<f32>,
    /// `[router_hidden]` cross-layer state scale (None on block 0, where EDA is off).
    pub router_states_scale: Option<Vec<f32>>,
    /// router_mlp: RMSNorm `[router_hidden]`, fc1/fc2 `[router_hidden, router_hidden]`+bias,
    /// out_proj `[num_router_experts, router_hidden]`.
    pub norm_w: Vec<f32>,
    pub fc1_w: Vec<f32>,
    pub fc1_b: Vec<f32>,
    pub fc2_w: Vec<f32>,
    pub fc2_b: Vec<f32>,
    pub out_proj_w: Vec<f32>,
    /// `[num_router_experts]` (last entry is the MoD skip route, init −1.0).
    pub balancing_biases: Vec<f32>,
}

/// One SwiGLU expert (host f32).
pub struct ZayaExpertWeights {
    /// Fused gate+up `[2*moe_intermediate, hidden]`.
    pub gate_up_proj: Vec<f32>,
    /// Down `[hidden, moe_intermediate]`.
    pub down_proj: Vec<f32>,
}

/// Learned affine residual merge `(h+bias)*scale + (res+bias)*scale` (host f32).
pub struct ZayaResidualScale {
    pub hidden_states_scale: Vec<f32>,
    pub hidden_states_bias: Vec<f32>,
    pub residual_scale: Vec<f32>,
    pub residual_bias: Vec<f32>,
}

impl ZayaResidualScale {
    fn load(src: &dyn ModelSource, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            hidden_states_scale: get(src, &format!("{prefix}.hidden_states_scale"))?,
            hidden_states_bias: get(src, &format!("{prefix}.hidden_states_bias"))?,
            residual_scale: get(src, &format!("{prefix}.residual_scale"))?,
            residual_bias: get(src, &format!("{prefix}.residual_bias"))?,
        })
    }
}

/// One hybrid decoder block (CCA attention + EDA/MoD MoE), host f32.
pub struct ZayaLayerWeights {
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub cca: ZayaCcaWeights,
    pub router: ZayaRouterWeights,
    pub experts: Vec<ZayaExpertWeights>,
    pub post_attention_residual_scale: ZayaResidualScale,
    pub post_mlp_residual_scale: ZayaResidualScale,
}

/// Full ZAYA1 model weights (host f32), for the CPU reference forward.
pub struct ZayaWeights {
    /// `[vocab, hidden]`.
    pub embed_tokens: Vec<f32>,
    /// Global input residual affine applied to the embeddings.
    pub input_hidden_states_scale: Vec<f32>,
    pub input_hidden_states_bias: Vec<f32>,
    pub layers: Vec<ZayaLayerWeights>,
    /// Final RMSNorm `[hidden]`.
    pub norm: Vec<f32>,
    /// `[vocab, hidden]` — tied to `embed_tokens` when no separate head is present.
    pub lm_head: Vec<f32>,
}

impl ZayaWeights {
    /// Load all host f32 weights from the native converted checkpoint `src`.
    /// Experts are read from the stacked 3D `mlp.experts.{gate_up,down}_proj`
    /// tensors and sliced per-expert.
    pub fn load_host(src: &dyn ModelSource, cfg: &ZayaConfig) -> Result<Self, String> {
        let embed_tokens = get(src, "model.embed_tokens.weight")?;
        let input_hidden_states_scale = get(src, "model.input_hidden_states_scale")?;
        let input_hidden_states_bias = get(src, "model.input_hidden_states_bias")?;
        let norm = get(src, "model.norm.weight")?;
        let lm_head = match get(src, "lm_head.weight") {
            Ok(w) => w,
            Err(_) if cfg.tie_word_embeddings => embed_tokens.clone(),
            Err(e) => return Err(e),
        };

        let mut layers = Vec::with_capacity(cfg.num_blocks);
        for l in 0..cfg.num_blocks {
            let p = format!("model.layers.{l}");
            let attn = format!("{p}.self_attn");
            let qkv = format!("{attn}.qkv_proj");
            let gate = format!("{p}.mlp.gate");
            let rmlp = format!("{gate}.router_mlp");

            let cca = ZayaCcaWeights {
                q_proj: get(src, &format!("{qkv}.q_proj.weight"))?,
                k_proj: get(src, &format!("{qkv}.k_proj.weight"))?,
                v_proj_current: get(src, &format!("{qkv}.v_proj_current.weight"))?,
                v_proj_delayed: get(src, &format!("{qkv}.v_proj_delayed.weight"))?,
                conv_qk_depthwise_w: get(src, &format!("{qkv}.conv_qk_depthwise.weight"))?,
                conv_qk_depthwise_b: get(src, &format!("{qkv}.conv_qk_depthwise.bias"))?,
                conv_qk_grouped_w: get(src, &format!("{qkv}.conv_qk_grouped.weight"))?,
                conv_qk_grouped_b: get(src, &format!("{qkv}.conv_qk_grouped.bias"))?,
                qk_norm_temp: get(src, &format!("{attn}.qk_norm.temp"))?,
                o_proj: get(src, &format!("{attn}.o_proj.weight"))?,
            };

            // EDA cross-layer state scale is absent on block 0 (use_eda = idx != 0).
            let router_states_scale = if l == 0 {
                None
            } else {
                Some(get(src, &format!("{gate}.router_states_scale"))?)
            };
            let router = ZayaRouterWeights {
                down_proj_w: get(src, &format!("{gate}.down_proj.weight"))?,
                down_proj_b: get(src, &format!("{gate}.down_proj.bias"))?,
                router_states_scale,
                norm_w: get(src, &format!("{rmlp}.norm.weight"))?,
                fc1_w: get(src, &format!("{rmlp}.fc1.weight"))?,
                fc1_b: get(src, &format!("{rmlp}.fc1.bias"))?,
                fc2_w: get(src, &format!("{rmlp}.fc2.weight"))?,
                fc2_b: get(src, &format!("{rmlp}.fc2.bias"))?,
                out_proj_w: get(src, &format!("{rmlp}.out_proj.weight"))?,
                balancing_biases: get(src, &format!("{gate}.balancing_biases"))?,
            };

            let experts = load_experts(src, cfg, &format!("{p}.mlp.experts"))?;

            layers.push(ZayaLayerWeights {
                input_layernorm: get(src, &format!("{p}.input_layernorm.weight"))?,
                post_attention_layernorm: get(
                    src,
                    &format!("{p}.post_attention_layernorm.weight"),
                )?,
                cca,
                router,
                experts,
                post_attention_residual_scale: ZayaResidualScale::load(
                    src,
                    &format!("{p}.post_attention_residual_scale"),
                )?,
                post_mlp_residual_scale: ZayaResidualScale::load(
                    src,
                    &format!("{p}.post_mlp_residual_scale"),
                )?,
            });
        }

        Ok(Self {
            embed_tokens,
            input_hidden_states_scale,
            input_hidden_states_bias,
            layers,
            norm,
            lm_head,
        })
    }
}

/// Slice the stacked 3D `mlp.experts.{gate_up,down}_proj` tensors per-expert. The
/// native checkpoint stores them as `[num_experts, out, in]`; the per-expert HFQ
/// split path produces the same per-expert slices under
/// `mlp.experts.{E}.{base}.weight` (handled by the future HFQ loader).
fn load_experts(
    src: &dyn ModelSource,
    cfg: &ZayaConfig,
    prefix: &str,
) -> Result<Vec<ZayaExpertWeights>, String> {
    let n = cfg.moe.num_experts;
    let (gate_up, gu_shape) = get_shaped(src, &format!("{prefix}.gate_up_proj"))?;
    let (down, dn_shape) = get_shaped(src, &format!("{prefix}.down_proj"))?;
    if gu_shape.first() != Some(&n) || dn_shape.first() != Some(&n) {
        return Err(format!(
            "zaya loader: expert stack first dim {gu_shape:?}/{dn_shape:?} != num_experts {n}"
        ));
    }
    let gu_per = gate_up.len() / n;
    let dn_per = down.len() / n;
    let mut experts = Vec::with_capacity(n);
    for e in 0..n {
        experts.push(ZayaExpertWeights {
            gate_up_proj: gate_up[e * gu_per..(e + 1) * gu_per].to_vec(),
            down_proj: down[e * dn_per..(e + 1) * dn_per].to_vec(),
        });
    }
    Ok(experts)
}
