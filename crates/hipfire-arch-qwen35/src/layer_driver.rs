// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The single qwen35 per-layer weight schema, generic over a runtime
//! `WeightBackend`. `load_weights` (HFQ), `load_weights_paroquant` (PaRo), and
//! `load_layer_into` (multi-GPU HFQ) all funnel through `load_layer`.

use crate::qwen35::weights::{DeltaNetBiases, FullAttnBiases};
use crate::qwen35::{
    DeltaNetLayerWeights, DeltaNetMoeLayerWeights, FullAttnLayerWeights, FullAttnMoeLayerWeights,
    LayerType, LayerWeights, MoeFfnWeights, Qwen35Config,
};
use hip_bridge::HipResult;
use hipfire_runtime::weight_backend::WeightBackend;

/// Load one layer's weights. `load_moe` builds the MoE FFN block for MoE layers
/// (format-specific: HFQ `load_moe_ffn` vs PaRo `paro_load_moe_ffn`), supplied by
/// the caller so MoE layout stays arch-owned.
/// A bias that MUST be there.
///
/// Escha ships all of a layer's biases together or none, so once the first
/// probe finds one the rest are mandatory. A half-biased layer is a corrupt
/// checkpoint, and silently substituting zeros would degrade output without
/// failing — the same trap as the dropped MTP head.
fn need_bias<B: WeightBackend>(
    b: &mut B,
    rel: &str,
    n: usize,
) -> hip_bridge::HipResult<rdna_compute::GpuTensor> {
    b.bias_opt(rel, n)?.ok_or_else(|| {
        hip_bridge::HipError::new(
            0,
            &format!(
                "{rel}: layer has some escha biases but not this one — a partially biased \
                 layer is a corrupt checkpoint, not a model to run with zeros"
            ),
        )
    })
}

pub(crate) fn load_layer<B: WeightBackend>(
    b: &mut B,
    config: &Qwen35Config,
    layer_idx: usize,
    mut load_moe: impl FnMut(&mut B, &Qwen35Config, usize) -> HipResult<MoeFfnWeights>,
) -> HipResult<LayerWeights> {
    b.set_layer(layer_idx);
    let is_moe = config.num_experts > 0;
    let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
        + config.linear_num_value_heads * config.linear_value_head_dim;
    let d_inner = config.linear_num_value_heads * config.linear_value_head_dim;
    let q_out_dim = config.n_heads * config.head_dim * 2;
    let kv_dim = config.n_kv_heads * config.head_dim;
    let o_in = config.n_heads * config.head_dim;

    Ok(match (config.layer_types[layer_idx], is_moe) {
        (LayerType::LinearAttention, false) => LayerWeights::DeltaNet(DeltaNetLayerWeights {
            attn_norm: b.norm("input_layernorm.weight", &[config.dim])?,
            wqkv: b.proj("linear_attn.in_proj_qkv", qkv_dim, config.dim)?,
            wz: b.proj("linear_attn.in_proj_z", d_inner, config.dim)?,
            w_alpha: b.proj(
                "linear_attn.in_proj_a",
                config.linear_num_value_heads,
                config.dim,
            )?,
            w_beta: b.proj(
                "linear_attn.in_proj_b",
                config.linear_num_value_heads,
                config.dim,
            )?,
            a_log: b.raw_f32("linear_attn.A_log", config.linear_num_value_heads)?,
            dt_bias: b.raw_f32("linear_attn.dt_bias", config.linear_num_value_heads)?,
            conv_weight: b.raw_f32(
                "linear_attn.conv1d.weight",
                qkv_dim * config.conv_kernel_dim,
            )?,
            norm_weight: b.raw_f32("linear_attn.norm.weight", config.linear_value_head_dim)?,
            wo: b.proj("linear_attn.out_proj", config.dim, d_inner)?,
            ffn_norm: b.norm("post_attention_layernorm.weight", &[config.dim])?,
            w_gate: b.proj("mlp.gate_proj", config.hidden_dim, config.dim)?,
            w_up: b.proj("mlp.up_proj", config.hidden_dim, config.dim)?,
            w_down: b.proj("mlp.down_proj", config.dim, config.hidden_dim)?,
            // Escha dense exports (Qwen3.8-27B) carry an additive output bias
            // on every coded projection. All six are present together or not
            // at all, so one probe decides. `in_proj_a`/`in_proj_b` have none
            // — escha's `ignore` list keeps them as plain weights.
            biases: match b.bias_opt("linear_attn.in_proj_qkv.bias", qkv_dim)? {
                None => None,
                Some(qkv) => Some(DeltaNetBiases {
                    qkv,
                    z: need_bias(b, "linear_attn.in_proj_z.bias", d_inner)?,
                    o: need_bias(b, "linear_attn.out_proj.bias", config.dim)?,
                    gate: need_bias(b, "mlp.gate_proj.bias", config.hidden_dim)?,
                    up: need_bias(b, "mlp.up_proj.bias", config.hidden_dim)?,
                    down: need_bias(b, "mlp.down_proj.bias", config.dim)?,
                }),
            },
        }),
        (LayerType::FullAttention, false) => LayerWeights::FullAttn(FullAttnLayerWeights {
            attn_norm: b.norm("input_layernorm.weight", &[config.dim])?,
            wq: b.proj("self_attn.q_proj", q_out_dim, config.dim)?,
            wk: b.proj("self_attn.k_proj", kv_dim, config.dim)?,
            wv: b.proj("self_attn.v_proj", kv_dim, config.dim)?,
            wo: b.proj("self_attn.o_proj", config.dim, o_in)?,
            q_norm: b.norm("self_attn.q_norm.weight", &[config.head_dim])?,
            k_norm: b.norm("self_attn.k_norm.weight", &[config.head_dim])?,
            ffn_norm: b.norm("post_attention_layernorm.weight", &[config.dim])?,
            w_gate: b.proj("mlp.gate_proj", config.hidden_dim, config.dim)?,
            w_up: b.proj("mlp.up_proj", config.hidden_dim, config.dim)?,
            w_down: b.proj("mlp.down_proj", config.dim, config.hidden_dim)?,
            biases: match b.bias_opt("self_attn.q_proj.bias", q_out_dim)? {
                None => None,
                Some(q) => Some(FullAttnBiases {
                    q,
                    k: need_bias(b, "self_attn.k_proj.bias", kv_dim)?,
                    v: need_bias(b, "self_attn.v_proj.bias", kv_dim)?,
                    o: need_bias(b, "self_attn.o_proj.bias", config.dim)?,
                    gate: need_bias(b, "mlp.gate_proj.bias", config.hidden_dim)?,
                    up: need_bias(b, "mlp.up_proj.bias", config.hidden_dim)?,
                    down: need_bias(b, "mlp.down_proj.bias", config.dim)?,
                }),
            },
        }),
        (LayerType::LinearAttention, true) => LayerWeights::DeltaNetMoe(DeltaNetMoeLayerWeights {
            attn_norm: b.norm("input_layernorm.weight", &[config.dim])?,
            wqkv: b.proj("linear_attn.in_proj_qkv", qkv_dim, config.dim)?,
            wz: b.proj("linear_attn.in_proj_z", d_inner, config.dim)?,
            w_alpha: b.proj(
                "linear_attn.in_proj_a",
                config.linear_num_value_heads,
                config.dim,
            )?,
            w_beta: b.proj(
                "linear_attn.in_proj_b",
                config.linear_num_value_heads,
                config.dim,
            )?,
            a_log: b.raw_f32("linear_attn.A_log", config.linear_num_value_heads)?,
            dt_bias: b.raw_f32("linear_attn.dt_bias", config.linear_num_value_heads)?,
            conv_weight: b.raw_f32(
                "linear_attn.conv1d.weight",
                qkv_dim * config.conv_kernel_dim,
            )?,
            norm_weight: b.raw_f32("linear_attn.norm.weight", config.linear_value_head_dim)?,
            wo: b.proj("linear_attn.out_proj", config.dim, d_inner)?,
            ffn_norm: b.norm("post_attention_layernorm.weight", &[config.dim])?,
            ffn: load_moe(b, config, layer_idx)?,
        }),
        (LayerType::FullAttention, true) => LayerWeights::FullAttnMoe(FullAttnMoeLayerWeights {
            attn_norm: b.norm("input_layernorm.weight", &[config.dim])?,
            wq: b.proj("self_attn.q_proj", q_out_dim, config.dim)?,
            wk: b.proj("self_attn.k_proj", kv_dim, config.dim)?,
            wv: b.proj("self_attn.v_proj", kv_dim, config.dim)?,
            wo: b.proj("self_attn.o_proj", config.dim, o_in)?,
            q_norm: b.norm("self_attn.q_norm.weight", &[config.head_dim])?,
            k_norm: b.norm("self_attn.k_norm.weight", &[config.head_dim])?,
            ffn_norm: b.norm("post_attention_layernorm.weight", &[config.dim])?,
            ffn: load_moe(b, config, layer_idx)?,
        }),
    })
}
