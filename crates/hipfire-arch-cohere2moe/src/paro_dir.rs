// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Transparent ParoQuant safetensors-Dir weight loading for Cohere2-MoE.
//!
//! Mirrors the qwen35 live Dir path (`qwen35::ParoSource` + `paro_moe`): the
//! heavy lifting (paro→fused-HFQ4G128 repack, rotation-sidecar aliasing) is done
//! by the SHARED `hipfire_runtime::paro` primitives; this module only maps
//! cohere2moe's tensor names + layer structure (parallel block, dense prefix
//! layer, sliding/full layers) and assembles cohere2moe's weight structs.
//!
//! Differences from qwen35's `paro_load_moe_ffn`: Cohere2-MoE has NO shared
//! expert (`num_shared_experts=0`), and uses **plain** RMSNorm (no `+1` bake),
//! so norms load via `paro_load_f32`, not qwen35's `paro_load_norm`.

use crate::cohere2moe::{
    Cohere2MoeLayerWeights, Cohere2MoeParoSidecars, Cohere2MoeWeights, DenseFfn, ExpertWeights,
    Ffn, MoeFfn,
};
use crate::config::Cohere2MoeConfig;
use hipfire_runtime::llama::WeightTensor;
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::paro::{
    alias_paro_rotation, load_fp16_weight_from_source, paro_load_f32, paro_load_wt,
    paro_repack_moe_projection, paro_text_prefix,
};
use rdna_compute::{DType, Gpu, GpuTensor};

fn e<T: std::fmt::Debug>(ctx: &str) -> impl Fn(T) -> String + '_ {
    move |err| format!("cohere2moe Dir: {ctx}: {err:?}")
}

/// Load the per-layer shared PARO sidecars for a MoE layer (the qwen35
/// `experts.{gate_up,down}_weight_{pairs,theta,channel_scales}` 6-tensor set).
fn load_moe_sidecars(
    source: &dyn ModelSource,
    gpu: &Gpu,
    mp: &str,
    p: &str,
) -> Result<Cohere2MoeParoSidecars, String> {
    let base = format!("{mp}.{p}.mlp.experts");
    let load = |name: &str| -> Result<GpuTensor, String> {
        let full = format!("{base}.{name}");
        let (_, data) = source
            .tensor_data(&full)
            .ok_or_else(|| format!("cohere2moe Dir: MoE sidecar not found: {full}"))?;
        gpu.upload_raw(data, &[data.len()]).map_err(e("sidecar"))
    };
    Ok(Cohere2MoeParoSidecars {
        gate_up_pairs: load("gate_up_weight_pairs")?,
        gate_up_theta: load("gate_up_weight_theta")?,
        gate_up_channel_scales: load("gate_up_weight_channel_scales")?,
        down_pairs: load("down_weight_pairs")?,
        down_theta: load("down_weight_theta")?,
        down_channel_scales: load("down_weight_channel_scales")?,
    })
}

/// Load the full ParoQuant weights for Cohere2-MoE from a safetensors Dir.
pub fn load_from_source(
    source: &dyn ModelSource,
    cfg: &Cohere2MoeConfig,
    gpu: &mut Gpu,
) -> Result<Cohere2MoeWeights, String> {
    let hidden = cfg.hidden_size;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let dense_inter = cfg.dense_intermediate_size;
    let moe_inter = cfg.moe_intermediate_size;
    let n_exp = cfg.num_experts;

    let qc = source.quant_config().ok_or_else(|| {
        "cohere2moe Dir: ParoQuant model requires quantization_config".to_string()
    })?;
    let gs = qc.group_size;
    let kr = qc.krot;

    let mp = paro_text_prefix(source).map_err(e("text prefix"))?;

    // ── Globals. embed_tokens is the shared (tied) lm_head; F16 in the Dir. ──
    let (_, embed_bytes) = source
        .tensor_data(&format!("{mp}.embed_tokens.weight"))
        .ok_or_else(|| "cohere2moe Dir: embed_tokens.weight not found".to_string())?;
    let embed = gpu
        .upload_raw(embed_bytes, &[embed_bytes.len()])
        .map_err(e("upload embed"))?;
    let lm_head = load_fp16_weight_from_source(
        source,
        gpu,
        &format!("{mp}.embed_tokens.weight"),
        cfg.vocab_size,
        hidden,
    )
    .map_err(e("lm_head"))?;
    let embed_dtype = lm_head.gpu_dtype;
    // NOTE: paro_load_f32 / paro_load_wt prepend the text prefix internally —
    // pass the suffix WITHOUT `mp` (load_fp16_weight_from_source /
    // paro_repack_moe_projection take the FULL name WITH `mp`).
    let final_norm = paro_load_f32(source, gpu, "norm.weight", hidden).map_err(e("final norm"))?;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for l in 0..cfg.num_hidden_layers {
        // `p` is the prefix-RELATIVE layer path. paro_load_{wt,f32} prepend `mp`
        // (→ `model.layers.N...`); the `{mp}.{p}` call sites form the same.
        let p = format!("layers.{l}");
        let input_norm = paro_load_f32(source, gpu, &format!("{p}.input_layernorm.weight"), hidden)
            .map_err(e("input norm"))?;
        let wq = paro_load_wt(
            source,
            gpu,
            &format!("{p}.self_attn.q_proj"),
            q_dim,
            hidden,
            gs,
            kr,
        )
        .map_err(e("wq"))?;
        let wk = paro_load_wt(
            source,
            gpu,
            &format!("{p}.self_attn.k_proj"),
            kv_dim,
            hidden,
            gs,
            kr,
        )
        .map_err(e("wk"))?;
        let wv = paro_load_wt(
            source,
            gpu,
            &format!("{p}.self_attn.v_proj"),
            kv_dim,
            hidden,
            gs,
            kr,
        )
        .map_err(e("wv"))?;
        let wo = paro_load_wt(
            source,
            gpu,
            &format!("{p}.self_attn.o_proj"),
            hidden,
            q_dim,
            gs,
            kr,
        )
        .map_err(e("wo"))?;

        let ffn = if cfg.is_dense_ffn(l) {
            let gate = paro_load_wt(
                source,
                gpu,
                &format!("{p}.mlp.gate_proj"),
                dense_inter,
                hidden,
                gs,
                kr,
            )
            .map_err(e("dense gate"))?;
            let up = paro_load_wt(
                source,
                gpu,
                &format!("{p}.mlp.up_proj"),
                dense_inter,
                hidden,
                gs,
                kr,
            )
            .map_err(e("dense up"))?;
            let down = paro_load_wt(
                source,
                gpu,
                &format!("{p}.mlp.down_proj"),
                hidden,
                dense_inter,
                gs,
                kr,
            )
            .map_err(e("dense down"))?;
            Ffn::Dense(DenseFfn { gate, up, down })
        } else {
            // Router is F16 dense.
            let router = load_fp16_weight_from_source(
                source,
                gpu,
                &format!("{mp}.{p}.mlp.gate.weight"),
                n_exp,
                hidden,
            )
            .map_err(e("router"))?;

            let shared = load_moe_sidecars(source, gpu, mp, &p)?;

            // Routed experts: repack gate/up → fused HFQ4G128 (ParoQ4G128 dtype),
            // repack down, both aliasing the layer-shared rotation sidecars.
            let mut experts = Vec::with_capacity(n_exp);
            for x in 0..n_exp {
                let gate_prefix = format!("{mp}.{p}.mlp.experts.{x}.gate_proj");
                let up_prefix = format!("{mp}.{p}.mlp.experts.{x}.up_proj");
                let down_prefix = format!("{mp}.{p}.mlp.experts.{x}.down_proj");

                let gate_bytes = paro_repack_moe_projection(
                    source,
                    &gate_prefix,
                    moe_inter,
                    hidden,
                    gs as usize,
                )
                .map_err(e("repack gate"))?;
                let up_bytes =
                    paro_repack_moe_projection(source, &up_prefix, moe_inter, hidden, gs as usize)
                        .map_err(e("repack up"))?;
                let mut gate_up_bytes = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
                gate_up_bytes.extend_from_slice(&gate_bytes);
                gate_up_bytes.extend_from_slice(&up_bytes);
                let gate_up_buf = gpu
                    .upload_raw(&gate_up_bytes, &[gate_up_bytes.len()])
                    .map_err(e("upload gate_up"))?;

                let down_bytes = paro_repack_moe_projection(
                    source,
                    &down_prefix,
                    hidden,
                    moe_inter,
                    gs as usize,
                )
                .map_err(e("repack down"))?;
                let down_buf = gpu
                    .upload_raw(&down_bytes, &[down_bytes.len()])
                    .map_err(e("upload down"))?;

                let gate_up = WeightTensor {
                    buf: gate_up_buf,
                    gpu_dtype: DType::ParoQ4G128,
                    m: 2 * moe_inter,
                    k: hidden,
                    row_stride: 0,
                    paro: Some(alias_paro_rotation(
                        &shared.gate_up_pairs,
                        &shared.gate_up_theta,
                        &shared.gate_up_channel_scales,
                        kr as u32,
                        gs,
                    )),
                    awq_scale: None,
                };
                let down = WeightTensor {
                    buf: down_buf,
                    gpu_dtype: DType::ParoQ4G128,
                    m: hidden,
                    k: moe_inter,
                    row_stride: 0,
                    paro: Some(alias_paro_rotation(
                        &shared.down_pairs,
                        &shared.down_theta,
                        &shared.down_channel_scales,
                        kr as u32,
                        gs,
                    )),
                    awq_scale: None,
                };
                experts.push(ExpertWeights { gate_up, down });
            }

            // Device-side expert pointer tables (same layout as the HFQ path).
            let gu_bytes: Vec<u8> = experts
                .iter()
                .flat_map(|e| (e.gate_up.buf.buf.as_ptr() as u64).to_ne_bytes())
                .collect();
            let dn_bytes: Vec<u8> = experts
                .iter()
                .flat_map(|e| (e.down.buf.buf.as_ptr() as u64).to_ne_bytes())
                .collect();
            let expert_gate_up_ptrs = gpu
                .alloc_tensor(&[2 * n_exp], DType::F32)
                .map_err(e("alloc gu_ptrs"))?;
            let expert_down_ptrs = gpu
                .alloc_tensor(&[2 * n_exp], DType::F32)
                .map_err(e("alloc dn_ptrs"))?;
            gpu.hip
                .memcpy_htod(&expert_gate_up_ptrs.buf, &gu_bytes)
                .map_err(e("htod gu_ptrs"))?;
            gpu.hip
                .memcpy_htod(&expert_down_ptrs.buf, &dn_bytes)
                .map_err(e("htod dn_ptrs"))?;

            Ffn::Moe(MoeFfn {
                router,
                experts,
                expert_gate_up_ptrs,
                expert_down_ptrs,
                paro_shared: Some(shared),
            })
        };

        layers.push(Cohere2MoeLayerWeights {
            input_norm,
            wq,
            wk,
            wv,
            wo,
            ffn,
            attn_kind: cfg.attn_kind(l),
        });
    }

    Ok(Cohere2MoeWeights {
        embed,
        embed_dtype,
        final_norm,
        lm_head,
        layers,
        moe_group_plans: std::sync::OnceLock::new(),
    })
}
