// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MoE-specific PaRo weight loaders for Qwen3.5.
//!
//! These cannot live in `hipfire-runtime` because they return/accept
//! Qwen3.5-specific types: `MoeParoSidecars`, `MoeFfnWeights`, `Qwen35Config`,
//! `SharedExpertWeights`, and `ExpertWeights`.

use crate::qwen35::{
    ExpertWeights, MoeFfnWeights, MoeParoSidecars, Qwen35Config, SharedExpertWeights,
};
use hip_bridge::{HipError, HipResult};
use hipfire_runtime::llama::WeightTensor;
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::paro::{
    alias_paro_rotation, load_fp16_weight_from_source, paro_load_wt, paro_repack_moe_projection,
    paro_text_prefix,
};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Upload the per-layer shared PARO rotation sidecars (one tuple for gate||up,
/// one for down). All 256 experts will reference these via non-owning
/// `ParoRotation` aliases.
pub(crate) fn paro_load_moe_shared_sidecars(
    source: &dyn ModelSource,
    gpu: &Gpu,
    p: &str,
) -> HipResult<MoeParoSidecars> {
    let mp = paro_text_prefix(source)?;
    let base = format!("{mp}.{p}.mlp.experts");
    let load = |name: &str| -> HipResult<GpuTensor> {
        let full = format!("{base}.{name}");
        let (_, data) = source.tensor_data(&full).ok_or_else(|| {
            HipError::new(
                0,
                &format!("ParoQuant MoE shared sidecar not found: {full}"),
            )
        })?;
        gpu.upload_raw(data, &[data.len()])
    };
    let qc = source
        .quant_config()
        .ok_or_else(|| HipError::new(0, "ParoQuant: quant_config required"))?;
    Ok(MoeParoSidecars {
        gate_up_pairs: load("gate_up_weight_pairs")?,
        gate_up_theta: load("gate_up_weight_theta")?,
        gate_up_channel_scales: load("gate_up_weight_channel_scales")?,
        down_pairs: load("down_weight_pairs")?,
        down_theta: load("down_weight_theta")?,
        down_channel_scales: load("down_weight_channel_scales")?,
        krot: qc.krot as u32,
        group_size: qc.group_size,
    })
}

/// Load the full ParoQuant MoE FFN block for one layer:
///   - dense FP16 router (`mlp.gate.weight [n_exp, hidden]`)
///   - dense FP16 shared-expert scalar gate (`mlp.shared_expert_gate.weight [1, hidden]`)
///   - shared expert (three per-projection PARO tensors: gate, up, down)
///   - routed experts, each with a fused gate||up HFQ4G128 buffer + a down
///     HFQ4G128 buffer, all referencing layer-shared PARO sidecars
pub(crate) fn paro_load_moe_ffn(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    p: &str,
    config: &Qwen35Config,
    layer_idx: u16,
) -> HipResult<MoeFfnWeights> {
    let n_exp = config.num_experts;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let dim = config.dim;
    let qc = source
        .quant_config()
        .ok_or_else(|| HipError::new(0, "ParoQuant MoE requires quant_config"))?;
    let gs = qc.group_size;
    let kr = qc.krot;

    let mp = paro_text_prefix(source)?;

    // ── Router (FP16 dense in shisa-ai's PARO checkpoint) ──
    let router = load_fp16_weight_from_source(
        source,
        gpu,
        &format!("{mp}.{p}.mlp.gate.weight"),
        n_exp,
        dim,
    )?;

    // Scalar gate on the shared-expert add — also FP16 dense.
    let shared_expert_gate = load_fp16_weight_from_source(
        source,
        gpu,
        &format!("{mp}.{p}.mlp.shared_expert_gate.weight"),
        1,
        dim,
    )?;

    // ── Shared expert ──
    let shared_expert = SharedExpertWeights {
        gate: paro_load_wt(
            source,
            gpu,
            &format!("{p}.mlp.shared_expert.gate_proj"),
            smi,
            dim,
            gs,
            kr,
        )?,
        up: paro_load_wt(
            source,
            gpu,
            &format!("{p}.mlp.shared_expert.up_proj"),
            smi,
            dim,
            gs,
            kr,
        )?,
        down: paro_load_wt(
            source,
            gpu,
            &format!("{p}.mlp.shared_expert.down_proj"),
            dim,
            smi,
            gs,
            kr,
        )?,
    };

    // ── Routed experts ──
    let shared = paro_load_moe_shared_sidecars(source, gpu, p)?;

    let groups_per_row_hidden = dim / (gs as usize);
    let bytes_per_row_hidden = groups_per_row_hidden * 72;
    let groups_per_row_mi = mi / (gs as usize);
    let bytes_per_row_mi = groups_per_row_mi * 72;

    let mut experts = Vec::with_capacity(n_exp);
    for x in 0..n_exp {
        let gate_prefix = format!("{mp}.{p}.mlp.experts.{x}.gate_proj");
        let up_prefix = format!("{mp}.{p}.mlp.experts.{x}.up_proj");
        let down_prefix = format!("{mp}.{p}.mlp.experts.{x}.down_proj");

        let gate_bytes = paro_repack_moe_projection(source, &gate_prefix, mi, dim, gs as usize)?;
        let up_bytes = paro_repack_moe_projection(source, &up_prefix, mi, dim, gs as usize)?;
        debug_assert_eq!(gate_bytes.len(), mi * bytes_per_row_hidden);
        debug_assert_eq!(up_bytes.len(), mi * bytes_per_row_hidden);
        let mut gate_up_bytes = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
        gate_up_bytes.extend_from_slice(&gate_bytes);
        gate_up_bytes.extend_from_slice(&up_bytes);
        let gate_up_buf = gpu.upload_raw(&gate_up_bytes, &[gate_up_bytes.len()])?;

        let down_bytes = paro_repack_moe_projection(source, &down_prefix, dim, mi, gs as usize)?;
        debug_assert_eq!(down_bytes.len(), dim * bytes_per_row_mi);
        let down_buf = gpu.upload_raw(&down_bytes, &[down_bytes.len()])?;

        let gate_up = WeightTensor {
            buf: gate_up_buf,
            gpu_dtype: DType::ParoQ4G128,
            m: 2 * mi,
            k: dim,
            row_stride: 0,
            paro: Some(alias_paro_rotation(
                &shared.gate_up_pairs,
                &shared.gate_up_theta,
                &shared.gate_up_channel_scales,
                shared.krot,
                shared.group_size,
            )),
            awq_scale: None,
        };
        let down = WeightTensor {
            buf: down_buf,
            gpu_dtype: DType::ParoQ4G128,
            m: dim,
            k: mi,
            row_stride: 0,
            paro: Some(alias_paro_rotation(
                &shared.down_pairs,
                &shared.down_theta,
                &shared.down_channel_scales,
                shared.krot,
                shared.group_size,
            )),
            awq_scale: None,
        };
        experts.push(ExpertWeights { gate_up, down });
    }

    // ── Device-side expert pointer tables ──
    let mut gu_ptrs: Vec<u64> = Vec::with_capacity(n_exp);
    let mut dn_ptrs: Vec<u64> = Vec::with_capacity(n_exp);
    for e in &experts {
        gu_ptrs.push(e.gate_up.buf.buf.as_ptr() as u64);
        dn_ptrs.push(e.down.buf.buf.as_ptr() as u64);
    }
    let gu_bytes: Vec<u8> = gu_ptrs.iter().flat_map(|q| q.to_ne_bytes()).collect();
    let dn_bytes: Vec<u8> = dn_ptrs.iter().flat_map(|q| q.to_ne_bytes()).collect();
    let expert_gate_up_ptrs = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    let expert_down_ptrs = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    gpu.hip.memcpy_htod(&expert_gate_up_ptrs.buf, &gu_bytes)?;
    gpu.hip.memcpy_htod(&expert_down_ptrs.buf, &dn_bytes)?;

    Ok(MoeFfnWeights {
        router,
        experts,
        packed_expert_owners: None,
        shared_expert,
        shared_expert_gate,
        expert_gate_up_ptrs,
        expert_down_ptrs,
        expert_down_awq_ptrs: None,
        expert_dtype_tags: None,
        layer_idx,
        expert_shape: None,
        paro_shared: Some(shared),
    })
}
