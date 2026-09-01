// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The single qwen35 per-layer weight schema, generic over a runtime
//! `WeightBackend`. `load_weights` (HFQ), `load_weights_paroquant` (PaRo), and
//! `load_layer_into` (multi-GPU HFQ) all funnel through `load_layer`.

use crate::qwen35::{
    DeltaNetLayerWeights, DeltaNetMoeLayerWeights, FullAttnLayerWeights, FullAttnMoeLayerWeights,
    LayerType, LayerWeights, MoeFfnWeights, Qwen35Config,
};
use hip_bridge::HipResult;
use hipfire_runtime::llama::WeightTensor;
use hipfire_runtime::weight_backend::WeightBackend;
use rdna_compute::{Gpu, GpuTensor};

#[derive(Default)]
struct LayerLoadStaging {
    attn_norm: Option<GpuTensor>,
    wqkv: Option<WeightTensor>,
    wz: Option<WeightTensor>,
    w_alpha: Option<WeightTensor>,
    w_beta: Option<WeightTensor>,
    a_log: Option<GpuTensor>,
    dt_bias: Option<GpuTensor>,
    conv_weight: Option<GpuTensor>,
    norm_weight: Option<GpuTensor>,
    wo: Option<WeightTensor>,
    ffn_norm: Option<GpuTensor>,
    wq: Option<WeightTensor>,
    wk: Option<WeightTensor>,
    wv: Option<WeightTensor>,
    q_norm: Option<GpuTensor>,
    k_norm: Option<GpuTensor>,
    w_gate: Option<WeightTensor>,
    w_up: Option<WeightTensor>,
    w_down: Option<WeightTensor>,
    ffn: Option<MoeFfnWeights>,
}

impl LayerLoadStaging {
    fn free_gpu(&mut self, gpu: &mut Gpu) {
        for tensor in [
            self.attn_norm.take(),
            self.a_log.take(),
            self.dt_bias.take(),
            self.conv_weight.take(),
            self.norm_weight.take(),
            self.ffn_norm.take(),
            self.q_norm.take(),
            self.k_norm.take(),
        ]
        .into_iter()
        .flatten()
        {
            let _ = gpu.free_tensor(tensor);
        }
        for weight in [
            self.wqkv.take(),
            self.wz.take(),
            self.w_alpha.take(),
            self.w_beta.take(),
            self.wo.take(),
            self.wq.take(),
            self.wk.take(),
            self.wv.take(),
            self.w_gate.take(),
            self.w_up.take(),
            self.w_down.take(),
        ]
        .into_iter()
        .flatten()
        {
            weight.free_all(gpu);
        }
        if let Some(ffn) = self.ffn.take() {
            crate::qwen35::weights::free_moe_ffn(gpu, ffn);
        }
    }
}

/// Load one layer's weights. `load_moe` builds the MoE FFN block for MoE layers
/// (format-specific: HFQ `load_moe_ffn` vs PaRo `paro_load_moe_ffn`), supplied by
/// the caller so MoE layout stays arch-owned.
pub(crate) fn load_layer<B: WeightBackend>(
    b: &mut B,
    config: &Qwen35Config,
    layer_idx: usize,
    mut load_moe: impl FnMut(&mut B, &Qwen35Config, usize) -> HipResult<MoeFfnWeights>,
) -> HipResult<LayerWeights> {
    let is_moe = config.num_experts > 0;
    let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
        + config.linear_num_value_heads * config.linear_value_head_dim;
    let d_inner = config.linear_num_value_heads * config.linear_value_head_dim;
    let q_out_dim = config.n_heads * config.head_dim * 2;
    let kv_dim = config.n_kv_heads * config.head_dim;
    let o_in = config.n_heads * config.head_dim;
    let mut staged = LayerLoadStaging::default();

    let result = (|| -> HipResult<LayerWeights> {
        b.set_layer(layer_idx);
        match (config.layer_types[layer_idx], is_moe) {
            (LayerType::LinearAttention, false) => {
                staged.attn_norm = Some(b.norm("input_layernorm.weight", &[config.dim])?);
                staged.wqkv = Some(b.proj("linear_attn.in_proj_qkv", qkv_dim, config.dim)?);
                staged.wz = Some(b.proj("linear_attn.in_proj_z", d_inner, config.dim)?);
                staged.w_alpha = Some(b.proj(
                    "linear_attn.in_proj_a",
                    config.linear_num_value_heads,
                    config.dim,
                )?);
                staged.w_beta = Some(b.proj(
                    "linear_attn.in_proj_b",
                    config.linear_num_value_heads,
                    config.dim,
                )?);
                staged.a_log = Some(b.raw_f32("linear_attn.A_log", config.linear_num_value_heads)?);
                staged.dt_bias =
                    Some(b.raw_f32("linear_attn.dt_bias", config.linear_num_value_heads)?);
                staged.conv_weight = Some(b.raw_f32(
                    "linear_attn.conv1d.weight",
                    qkv_dim * config.conv_kernel_dim,
                )?);
                staged.norm_weight =
                    Some(b.raw_f32("linear_attn.norm.weight", config.linear_value_head_dim)?);
                staged.wo = Some(b.proj("linear_attn.out_proj", config.dim, d_inner)?);
                staged.ffn_norm = Some(b.norm("post_attention_layernorm.weight", &[config.dim])?);
                staged.w_gate = Some(b.proj("mlp.gate_proj", config.hidden_dim, config.dim)?);
                staged.w_up = Some(b.proj("mlp.up_proj", config.hidden_dim, config.dim)?);
                staged.w_down = Some(b.proj("mlp.down_proj", config.dim, config.hidden_dim)?);
                Ok(LayerWeights::DeltaNet(DeltaNetLayerWeights {
                    attn_norm: staged.attn_norm.take().expect("staged attn_norm"),
                    wqkv: staged.wqkv.take().expect("staged wqkv"),
                    wz: staged.wz.take().expect("staged wz"),
                    w_alpha: staged.w_alpha.take().expect("staged w_alpha"),
                    w_beta: staged.w_beta.take().expect("staged w_beta"),
                    a_log: staged.a_log.take().expect("staged a_log"),
                    dt_bias: staged.dt_bias.take().expect("staged dt_bias"),
                    conv_weight: staged.conv_weight.take().expect("staged conv_weight"),
                    norm_weight: staged.norm_weight.take().expect("staged norm_weight"),
                    wo: staged.wo.take().expect("staged wo"),
                    ffn_norm: staged.ffn_norm.take().expect("staged ffn_norm"),
                    w_gate: staged.w_gate.take().expect("staged w_gate"),
                    w_up: staged.w_up.take().expect("staged w_up"),
                    w_down: staged.w_down.take().expect("staged w_down"),
                }))
            }
            (LayerType::FullAttention, false) => {
                staged.attn_norm = Some(b.norm("input_layernorm.weight", &[config.dim])?);
                staged.wq = Some(b.proj("self_attn.q_proj", q_out_dim, config.dim)?);
                staged.wk = Some(b.proj("self_attn.k_proj", kv_dim, config.dim)?);
                staged.wv = Some(b.proj("self_attn.v_proj", kv_dim, config.dim)?);
                staged.wo = Some(b.proj("self_attn.o_proj", config.dim, o_in)?);
                staged.q_norm = Some(b.norm("self_attn.q_norm.weight", &[config.head_dim])?);
                staged.k_norm = Some(b.norm("self_attn.k_norm.weight", &[config.head_dim])?);
                staged.ffn_norm = Some(b.norm("post_attention_layernorm.weight", &[config.dim])?);
                staged.w_gate = Some(b.proj("mlp.gate_proj", config.hidden_dim, config.dim)?);
                staged.w_up = Some(b.proj("mlp.up_proj", config.hidden_dim, config.dim)?);
                staged.w_down = Some(b.proj("mlp.down_proj", config.dim, config.hidden_dim)?);
                Ok(LayerWeights::FullAttn(FullAttnLayerWeights {
                    attn_norm: staged.attn_norm.take().expect("staged attn_norm"),
                    wq: staged.wq.take().expect("staged wq"),
                    wk: staged.wk.take().expect("staged wk"),
                    wv: staged.wv.take().expect("staged wv"),
                    wo: staged.wo.take().expect("staged wo"),
                    q_norm: staged.q_norm.take().expect("staged q_norm"),
                    k_norm: staged.k_norm.take().expect("staged k_norm"),
                    ffn_norm: staged.ffn_norm.take().expect("staged ffn_norm"),
                    w_gate: staged.w_gate.take().expect("staged w_gate"),
                    w_up: staged.w_up.take().expect("staged w_up"),
                    w_down: staged.w_down.take().expect("staged w_down"),
                }))
            }
            (LayerType::LinearAttention, true) => {
                staged.attn_norm = Some(b.norm("input_layernorm.weight", &[config.dim])?);
                staged.wqkv = Some(b.proj("linear_attn.in_proj_qkv", qkv_dim, config.dim)?);
                staged.wz = Some(b.proj("linear_attn.in_proj_z", d_inner, config.dim)?);
                staged.w_alpha = Some(b.proj(
                    "linear_attn.in_proj_a",
                    config.linear_num_value_heads,
                    config.dim,
                )?);
                staged.w_beta = Some(b.proj(
                    "linear_attn.in_proj_b",
                    config.linear_num_value_heads,
                    config.dim,
                )?);
                staged.a_log = Some(b.raw_f32("linear_attn.A_log", config.linear_num_value_heads)?);
                staged.dt_bias =
                    Some(b.raw_f32("linear_attn.dt_bias", config.linear_num_value_heads)?);
                staged.conv_weight = Some(b.raw_f32(
                    "linear_attn.conv1d.weight",
                    qkv_dim * config.conv_kernel_dim,
                )?);
                staged.norm_weight =
                    Some(b.raw_f32("linear_attn.norm.weight", config.linear_value_head_dim)?);
                staged.wo = Some(b.proj("linear_attn.out_proj", config.dim, d_inner)?);
                staged.ffn_norm = Some(b.norm("post_attention_layernorm.weight", &[config.dim])?);
                staged.ffn = Some(load_moe(b, config, layer_idx)?);
                Ok(LayerWeights::DeltaNetMoe(DeltaNetMoeLayerWeights {
                    attn_norm: staged.attn_norm.take().expect("staged attn_norm"),
                    wqkv: staged.wqkv.take().expect("staged wqkv"),
                    wz: staged.wz.take().expect("staged wz"),
                    w_alpha: staged.w_alpha.take().expect("staged w_alpha"),
                    w_beta: staged.w_beta.take().expect("staged w_beta"),
                    a_log: staged.a_log.take().expect("staged a_log"),
                    dt_bias: staged.dt_bias.take().expect("staged dt_bias"),
                    conv_weight: staged.conv_weight.take().expect("staged conv_weight"),
                    norm_weight: staged.norm_weight.take().expect("staged norm_weight"),
                    wo: staged.wo.take().expect("staged wo"),
                    ffn_norm: staged.ffn_norm.take().expect("staged ffn_norm"),
                    ffn: staged.ffn.take().expect("staged ffn"),
                }))
            }
            (LayerType::FullAttention, true) => {
                staged.attn_norm = Some(b.norm("input_layernorm.weight", &[config.dim])?);
                staged.wq = Some(b.proj("self_attn.q_proj", q_out_dim, config.dim)?);
                staged.wk = Some(b.proj("self_attn.k_proj", kv_dim, config.dim)?);
                staged.wv = Some(b.proj("self_attn.v_proj", kv_dim, config.dim)?);
                staged.wo = Some(b.proj("self_attn.o_proj", config.dim, o_in)?);
                staged.q_norm = Some(b.norm("self_attn.q_norm.weight", &[config.head_dim])?);
                staged.k_norm = Some(b.norm("self_attn.k_norm.weight", &[config.head_dim])?);
                staged.ffn_norm = Some(b.norm("post_attention_layernorm.weight", &[config.dim])?);
                staged.ffn = Some(load_moe(b, config, layer_idx)?);
                Ok(LayerWeights::FullAttnMoe(FullAttnMoeLayerWeights {
                    attn_norm: staged.attn_norm.take().expect("staged attn_norm"),
                    wq: staged.wq.take().expect("staged wq"),
                    wk: staged.wk.take().expect("staged wk"),
                    wv: staged.wv.take().expect("staged wv"),
                    wo: staged.wo.take().expect("staged wo"),
                    q_norm: staged.q_norm.take().expect("staged q_norm"),
                    k_norm: staged.k_norm.take().expect("staged k_norm"),
                    ffn_norm: staged.ffn_norm.take().expect("staged ffn_norm"),
                    ffn: staged.ffn.take().expect("staged ffn"),
                }))
            }
        }
    })();
    if result.is_err() {
        staged.free_gpu(b.gpu_mut());
    }
    result
}
