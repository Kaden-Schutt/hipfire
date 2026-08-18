// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 expert-parallel / multi-GPU: EP batch compatibility attestation,
//! `Qwen35DecodeBatchEpState`, EP decode/prefill, and the pp>1 band forwards.

use super::batch::ep_tick_inputs_prepared;
use super::batch::for_each_active_span;
use super::batch::lane_bit;
use super::batch::lm_head_batched;
use super::batch::valid_lane_mask;
use super::batch::BatchSemantics;
use super::batch::PrefillBatchScratch;
use super::batch::Qwen35DecodeBatchState;
use super::config::LayerType;
use super::config::Qwen35BatchCompatibility;
use super::config::Qwen35BatchLoadConfig;
use super::config::Qwen35Config;
use super::config::Qwen35EpBatchReceipt;
use super::config::Qwen35EpReduce;
use super::config::Qwen35EpTopology;
use super::forward::ffn_all_mq4_for_moe;
use super::forward::lower_variant;
use super::forward::moe_ffn_decode_with_scratch;
use super::forward::moe_ffn_decode_with_scratch_prerotated;
use super::forward::variant_of;
use super::forward::Qwen35Bindings;
use super::forward::Qwen35Scratch;
use super::forward::Qwen35ScratchSet;
use super::prefill::forward_batch_chunk_impl;
use super::prefill::forward_prefill_chunk;
use super::prefill::is_batchable_la;
use super::prefill::qwen35_layer_batch_admissible;
use super::prefill::PrefillBandCtx;
use super::prefill::PREFILL_MAX_BATCH;
use super::weights::mixed_expert_tag;
use super::weights::DeltaNetState;
use super::weights::GpuTensorDescriptor;
use super::weights::LayerWeights;
use super::weights::MoeFfnWeights;
use super::weights::Qwen35EpConfigFingerprint;
use super::weights::Qwen35HfqSourceIdentity;
use super::weights::Qwen35Weights;
use super::weights::StateQuant;
use super::weights::WeightTensorDescriptor;
use hip_bridge::HipError;
use hip_bridge::HipResult;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::execute_steps;
use hipfire_dispatch::pipeline::GemvInput;
use hipfire_dispatch::pipeline::Step;
use hipfire_runtime::llama;
use hipfire_runtime::llama::fused_rmsnorm_rotate_for_mq;
use hipfire_runtime::llama::weight_gemv_prerotated;
use hipfire_runtime::llama::weight_gemv_swiglu_residual;
use hipfire_runtime::llama::EmbeddingFormat;
use hipfire_runtime::llama::WeightTensor;
use hipfire_runtime::multi_gpu::Gpus;
use rdna_compute::DType;
use rdna_compute::GpuTensor;

fn layer_moe_ffn(layer: &LayerWeights) -> Option<&MoeFfnWeights> {
    match layer {
        LayerWeights::DeltaNetMoe(weights) => Some(&weights.ffn),
        LayerWeights::FullAttnMoe(weights) => Some(&weights.ffn),
        _ => None,
    }
}

/// Validate EP batch compatibility for the exact 4×gfx1201 MQ4R route.
/// Fails closed on any mismatch: arch, PP, provenance, geometry, dtype,
/// REAP/paged/AWQ/GL, Q8/EF, capacities, LDS. Returns attested receipt on success.
pub fn validate_ep_batch_compatibility(
    gpus: &Gpus,
    weights_per_rank: &[Qwen35Weights],
    config: &Qwen35Config,
    load_cfg: &Qwen35BatchLoadConfig,
) -> HipResult<Qwen35BatchCompatibility> {
    if load_cfg.max_batch == 0
        || load_cfg.lane_capacity == 0
        || load_cfg.repeat_capacity == 0
        || load_cfg.prefill_chunk == 0
    {
        return Err(HipError::new(0, "EP batch: zero capacity in load_cfg"));
    }
    if load_cfg.max_batch > 64 {
        return Err(HipError::new(
            0,
            &format!(
                "EP batch: max_batch {} >64 not supported",
                load_cfg.max_batch
            ),
        ));
    }
    if config.layer_types.len() != config.n_layers {
        return Err(HipError::new(
            0,
            "EP batch: config.layer_types length mismatch n_layers",
        ));
    }
    if gpus.devices.len() != 4 {
        return Err(HipError::new(
            0,
            &format!("EP batch: requires 4 ranks, got {}", gpus.devices.len()),
        ));
    }
    if weights_per_rank.len() != 4 {
        return Err(HipError::new(
            0,
            &format!(
                "EP batch: requires 4 weight shards, got {}",
                weights_per_rank.len()
            ),
        ));
    }
    for (i, dev) in gpus.devices.iter().enumerate() {
        if !dev.arch_caps.is_gfx1201() {
            return Err(HipError::new(
                0,
                &format!("EP batch: rank {i} arch {} != gfx1201", dev.arch),
            ));
        }
    }
    if gpus.layer_to_device.len() != config.n_layers {
        return Err(HipError::new(
            0,
            "EP batch: Gpus layer_to_device length mismatch config.n_layers",
        ));
    }
    if !gpus.layer_to_device.iter().all(|&d| d == 0) {
        return Err(HipError::new(
            0,
            "EP batch: pure EP requires PP=1 (all layers on rank 0)",
        ));
    }
    if gpus.band_starts.len() != 4 {
        return Err(HipError::new(
            0,
            "EP batch: band_starts must have 4 entries for 4-rank EP",
        ));
    }
    if gpus.band_starts[0] != 0 {
        return Err(HipError::new(0, "EP batch: band_starts[0] must be 0"));
    }
    for b in 1..4 {
        if gpus.band_starts[b] != config.n_layers {
            return Err(HipError::new(
                0,
                &format!("EP batch: band_starts[{b}] must be n_layers for pure EP"),
            ));
        }
    }
    if gpus.output_device != 0 {
        return Err(HipError::new(
            0,
            "EP batch: pure EP requires output_device==0",
        ));
    }
    if config.reap_keep.is_some() {
        return Err(HipError::new(0, "EP batch: REAP + EP not supported"));
    }
    if config.paged_experts {
        return Err(HipError::new(0, "EP batch: paged experts not supported"));
    }
    if config.num_experts == 0 || config.num_experts % 4 != 0 {
        return Err(HipError::new(
            0,
            "EP batch: num_experts must be divisible by 4",
        ));
    }
    if config.num_experts_per_tok == 0 || config.num_experts_per_tok > config.num_experts {
        return Err(HipError::new(0, "EP batch: invalid num_experts_per_tok"));
    }
    let mut seen_ranks = [false; 4];
    let mut ref_assign: Option<Box<[u8]>> = None;
    let mut ref_source: Option<std::sync::Arc<Qwen35HfqSourceIdentity>> = None;
    let mut ref_config_fp: Option<Qwen35EpConfigFingerprint> = None;
    // capture replicated seals for cross-rank equality
    let mut ref_token_embd: Option<GpuTensorDescriptor> = None;
    let mut ref_embd_format: Option<EmbeddingFormat> = None;
    let mut ref_output: Option<WeightTensorDescriptor> = None;
    let mut ref_output_norm: Option<GpuTensorDescriptor> = None;
    for (idx, w) in weights_per_rank.iter().enumerate() {
        let prov = w.ep_shard.as_ref().ok_or_else(|| HipError::new(0, &format!("EP batch: rank {idx} missing EP shard provenance (only load_weights_ep_rank may attach)")))?;
        if prov.rank_count() != 4 {
            return Err(HipError::new(
                0,
                &format!("EP batch: rank {idx} rank_count {} !=4", prov.rank_count()),
            ));
        }
        let r = prov.rank() as usize;
        if r >= 4 {
            return Err(HipError::new(
                0,
                &format!("EP batch: rank {idx} provenance rank {r} out of 0..3"),
            ));
        }
        if r != idx {
            return Err(HipError::new(
                0,
                &format!(
                    "EP batch: rank {idx} provenance rank {r} mismatched — permuted weight vector"
                ),
            ));
        }
        if prov.device_id() != gpus.devices[idx].device_id {
            return Err(HipError::new(
                0,
                &format!(
                    "EP batch: rank {idx} provenance device_id {} != physical device_id {}",
                    prov.device_id(),
                    gpus.devices[idx].device_id
                ),
            ));
        }
        if seen_ranks[r] {
            return Err(HipError::new(
                0,
                &format!("EP batch: duplicate provenance rank {r}"),
            ));
        }
        seen_ranks[r] = true;
        if prov.expert_to_rank().len() != config.num_experts {
            return Err(HipError::new(
                0,
                &format!(
                    "EP batch: rank {idx} expert assignment len {} != num_experts {}",
                    prov.expert_to_rank().len(),
                    config.num_experts
                ),
            ));
        }
        for &owner in prov.expert_to_rank() {
            if owner >= 4 {
                return Err(HipError::new(
                    0,
                    &format!("EP batch: rank {idx} expert owner {owner} out of range"),
                ));
            }
        }
        match &ref_assign {
            None => ref_assign = Some(prov.expert_to_rank().to_vec().into_boxed_slice()),
            Some(a) => {
                if a.as_ref() != prov.expert_to_rank() {
                    return Err(HipError::new(
                        0,
                        &format!("EP batch: rank {idx} expert assignment mismatched (reordered)"),
                    ));
                }
            }
        }
        // source identity and config fingerprint equality across ranks
        match &ref_source {
            None => ref_source = Some(std::sync::Arc::clone(&prov.source_identity)),
            Some(s) => {
                if s.as_ref() != prov.source_identity() {
                    return Err(HipError::new(
                        0,
                        &format!("EP batch: rank {idx} source identity mismatch"),
                    ));
                }
            }
        }
        match &ref_config_fp {
            None => ref_config_fp = Some(prov.config_fingerprint().clone()),
            Some(c) => {
                if c != prov.config_fingerprint() {
                    return Err(HipError::new(
                        0,
                        &format!("EP batch: rank {idx} config fingerprint mismatch"),
                    ));
                }
            }
        }
        // replicated layout seals equality (excluding per-rank local expert shards)
        let seal = prov.rank_seal();
        let cur_token = seal.token_embd.clone();
        let cur_output = seal.output.clone();
        let cur_out_norm = seal.output_norm.clone();
        match &ref_token_embd {
            None => ref_token_embd = Some(cur_token),
            Some(v) => {
                if v != &cur_token {
                    return Err(HipError::new(
                        0,
                        &format!("EP batch: rank {idx} token_embd seal mismatch"),
                    ));
                }
            }
        }
        match &ref_embd_format {
            None => ref_embd_format = Some(seal.embd_format),
            Some(v) => {
                if v != &seal.embd_format {
                    return Err(HipError::new(
                        0,
                        &format!("EP batch: rank {idx} embd_format mismatch"),
                    ));
                }
            }
        }
        match &ref_output {
            None => ref_output = Some(cur_output),
            Some(v) => {
                if v != &cur_output {
                    return Err(HipError::new(
                        0,
                        &format!("EP batch: rank {idx} output seal mismatch"),
                    ));
                }
            }
        }
        match &ref_output_norm {
            None => ref_output_norm = Some(cur_out_norm),
            Some(v) => {
                if v != &cur_out_norm {
                    return Err(HipError::new(
                        0,
                        &format!("EP batch: rank {idx} output_norm seal mismatch"),
                    ));
                }
            }
        }
        if w.layers.len() != config.n_layers {
            return Err(HipError::new(
                0,
                &format!(
                    "EP batch: rank {idx} layer count {} != config {}",
                    w.layers.len(),
                    config.n_layers
                ),
            ));
        }
        if w.output.m != config.vocab_size || w.output.k != config.dim {
            return Err(HipError::new(
                0,
                &format!(
                    "EP batch: rank {idx} output shape [{},{}] != [{},{}]",
                    w.output.m, w.output.k, config.vocab_size, config.dim
                ),
            ));
        }
        if w.output_norm.shape != vec![config.dim] {
            return Err(HipError::new(
                0,
                &format!("EP batch: rank {idx} output_norm shape mismatch"),
            ));
        }
        for (li, layer) in w.layers.iter().enumerate() {
            let expected = config.layer_types[li];
            let is_moe = matches!(
                layer,
                LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_)
            );
            let expect_moe = config.num_experts > 0;
            if expect_moe != is_moe {
                return Err(HipError::new(
                    0,
                    &format!("EP batch: rank {idx} layer {li} variant mismatch"),
                ));
            }
            let want_la = expected == LayerType::LinearAttention;
            let got_la = matches!(
                layer,
                LayerWeights::DeltaNet(_) | LayerWeights::DeltaNetMoe(_)
            );
            if want_la != got_la {
                return Err(HipError::new(
                    0,
                    &format!("EP batch: rank {idx} layer {li} type mismatch"),
                ));
            }
        }
        if !matches!(
            w.embd_format,
            EmbeddingFormat::HFQ4G256 | EmbeddingFormat::Q8_0
        ) {
            return Err(HipError::new(
                0,
                &format!(
                    "EP batch: rank {idx} unsupported embedding format {:?}",
                    w.embd_format
                ),
            ));
        }
        if !matches!(
            w.output.gpu_dtype,
            DType::Q8_0
                | DType::HFQ4G256
                | DType::MQ4G256
                | DType::HFQ6G256
                | DType::MQ6G256
                | DType::MQ3G256
        ) {
            return Err(HipError::new(
                0,
                &format!(
                    "EP batch: rank {idx} unsupported lm_head {:?}",
                    w.output.gpu_dtype
                ),
            ));
        }
        // EP rejects page/REAP already; also reject any paged-owned state
        if w.pager.is_some() {
            return Err(HipError::new(
                0,
                &format!("EP batch: rank {idx} pager present — paged experts not supported"),
            ));
        }
        // Reject any AWQ/GL/PARO presence on any weight
        for layer in &w.layers {
            let mut check_weight = |wt: &WeightTensor, name: &str| -> HipResult<()> {
                if wt.awq_scale.is_some() {
                    return Err(HipError::new(
                        0,
                        &format!("EP batch: AWQ not supported ({name})"),
                    ));
                }
                if wt.paro.is_some() {
                    return Err(HipError::new(
                        0,
                        &format!("EP batch: PARO not supported ({name})"),
                    ));
                }
                if wt.gpu_dtype == DType::ParoQ4G128 {
                    return Err(HipError::new(
                        0,
                        &format!("EP batch: PARO dtype not supported ({name})"),
                    ));
                }
                if matches!(wt.gpu_dtype, DType::MQ2G256GL | DType::MQ3G256GL) {
                    return Err(HipError::new(
                        0,
                        &format!("EP batch: GL not supported ({name})"),
                    ));
                }
                Ok(())
            };
            match layer {
                LayerWeights::DeltaNet(l) => {
                    check_weight(&l.wqkv, "wqkv")?;
                    check_weight(&l.wz, "wz")?;
                    check_weight(&l.w_alpha, "w_alpha")?;
                    check_weight(&l.w_beta, "w_beta")?;
                    check_weight(&l.wo, "wo")?;
                    check_weight(&l.w_gate, "w_gate")?;
                    check_weight(&l.w_up, "w_up")?;
                    check_weight(&l.w_down, "w_down")?;
                }
                LayerWeights::FullAttn(l) => {
                    check_weight(&l.wq, "wq")?;
                    check_weight(&l.wk, "wk")?;
                    check_weight(&l.wv, "wv")?;
                    check_weight(&l.wo, "wo")?;
                    check_weight(&l.w_gate, "w_gate")?;
                    check_weight(&l.w_up, "w_up")?;
                    check_weight(&l.w_down, "w_down")?;
                }
                LayerWeights::DeltaNetMoe(l) => {
                    check_weight(&l.wqkv, "wqkv")?;
                    check_weight(&l.wz, "wz")?;
                    check_weight(&l.w_alpha, "w_alpha")?;
                    check_weight(&l.w_beta, "w_beta")?;
                    check_weight(&l.wo, "wo")?;
                    check_weight(&l.ffn.router, "router")?;
                    check_weight(&l.ffn.shared_expert.gate, "shared_gate")?;
                    check_weight(&l.ffn.shared_expert.up, "shared_up")?;
                    check_weight(&l.ffn.shared_expert.down, "shared_down")?;
                    check_weight(&l.ffn.shared_expert_gate, "shared_expert_gate")?;
                    if l.ffn.paro_shared.is_some() {
                        return Err(HipError::new(
                            0,
                            "EP batch: PARO not supported (paro_shared)",
                        ));
                    }
                    if l.ffn.expert_down_awq_ptrs.is_some() {
                        return Err(HipError::new(0, "EP batch: AWQ not supported"));
                    }
                    for e in &l.ffn.experts {
                        check_weight(&e.gate_up, "expert gate_up")?;
                        check_weight(&e.down, "expert down")?;
                        let _ = mixed_expert_tag(e.gate_up.gpu_dtype, e.down.gpu_dtype).map_err(
                            |err| {
                                HipError::new(
                                    0,
                                    &format!(
                                        "EP batch: expert unsupported tag ({:?}/{:?}): {}",
                                        e.gate_up.gpu_dtype, e.down.gpu_dtype, err.message
                                    ),
                                )
                            },
                        )?;
                    }
                    // Global dtype table must be present and exact
                    let global = l.ffn.global_expert_dtypes.as_ref().ok_or_else(|| {
                        HipError::new(
                            0,
                            &format!("EP batch: rank {idx} MoE layer missing global_expert_dtypes"),
                        )
                    })?;
                    if global.len() != config.num_experts {
                        return Err(HipError::new(0, &format!("EP batch: rank {idx} global_expert_dtypes len {} != num_experts {}", global.len(), config.num_experts)));
                    }
                    for (gid, (g, d)) in global.iter().enumerate() {
                        let _ = mixed_expert_tag(*g, *d).map_err(|err| {
                            HipError::new(
                                0,
                                &format!(
                                    "EP batch: global pair {gid} unsupported ({g:?}/{d:?}): {}",
                                    err.message
                                ),
                            )
                        })?;
                    }
                    // tag table presence: must be Some iff global is mixed
                    let is_mixed = {
                        let first = global[0];
                        global.iter().any(|(g, d)| *g != first.0 || *d != first.1)
                    };
                    match (&l.ffn.expert_dtype_tags, is_mixed) {
                        (Some(tags), true) => {
                            if tags.shape[0] != config.num_experts {
                                return Err(HipError::new(0, "EP batch: tag table size mismatch"));
                            }
                        }
                        (None, false) => {}
                        (Some(_), false) => {
                            return Err(HipError::new(
                                0,
                                "EP batch: unexpected tag table on uniform layer",
                            ))
                        }
                        (None, true) => {
                            return Err(HipError::new(
                                0,
                                "EP batch: missing tag table on mixed layer",
                            ))
                        }
                    }
                    // pointer tables must be exactly [2*num_experts] F32 slots
                    if l.ffn.expert_gate_up_ptrs.shape != vec![2 * config.num_experts] {
                        return Err(HipError::new(
                            0,
                            "EP batch: gate_up pointer table shape mismatch",
                        ));
                    }
                    if l.ffn.expert_down_ptrs.shape != vec![2 * config.num_experts] {
                        return Err(HipError::new(
                            0,
                            "EP batch: down pointer table shape mismatch",
                        ));
                    }
                }
                LayerWeights::FullAttnMoe(l) => {
                    check_weight(&l.wq, "wq")?;
                    check_weight(&l.wk, "wk")?;
                    check_weight(&l.wv, "wv")?;
                    check_weight(&l.wo, "wo")?;
                    check_weight(&l.ffn.router, "router")?;
                    check_weight(&l.ffn.shared_expert.gate, "shared_gate")?;
                    check_weight(&l.ffn.shared_expert.up, "shared_up")?;
                    check_weight(&l.ffn.shared_expert.down, "shared_down")?;
                    check_weight(&l.ffn.shared_expert_gate, "shared_expert_gate")?;
                    if l.ffn.paro_shared.is_some() {
                        return Err(HipError::new(
                            0,
                            "EP batch: PARO not supported (paro_shared)",
                        ));
                    }
                    if l.ffn.expert_down_awq_ptrs.is_some() {
                        return Err(HipError::new(0, "EP batch: AWQ not supported"));
                    }
                    for e in &l.ffn.experts {
                        check_weight(&e.gate_up, "expert gate_up")?;
                        check_weight(&e.down, "expert down")?;
                        let _ = mixed_expert_tag(e.gate_up.gpu_dtype, e.down.gpu_dtype).map_err(
                            |err| {
                                HipError::new(
                                    0,
                                    &format!(
                                        "EP batch: expert unsupported tag ({:?}/{:?}): {}",
                                        e.gate_up.gpu_dtype, e.down.gpu_dtype, err.message
                                    ),
                                )
                            },
                        )?;
                    }
                    let global = l.ffn.global_expert_dtypes.as_ref().ok_or_else(|| {
                        HipError::new(
                            0,
                            &format!("EP batch: rank {idx} MoE layer missing global_expert_dtypes"),
                        )
                    })?;
                    if global.len() != config.num_experts {
                        return Err(HipError::new(0, &format!("EP batch: rank {idx} global_expert_dtypes len {} != num_experts {}", global.len(), config.num_experts)));
                    }
                    for (gid, (g, d)) in global.iter().enumerate() {
                        let _ = mixed_expert_tag(*g, *d).map_err(|err| {
                            HipError::new(
                                0,
                                &format!(
                                    "EP batch: global pair {gid} unsupported ({g:?}/{d:?}): {}",
                                    err.message
                                ),
                            )
                        })?;
                    }
                    let is_mixed = {
                        let first = global[0];
                        global.iter().any(|(g, d)| *g != first.0 || *d != first.1)
                    };
                    match (&l.ffn.expert_dtype_tags, is_mixed) {
                        (Some(tags), true) => {
                            if tags.shape[0] != config.num_experts {
                                return Err(HipError::new(0, "EP batch: tag table size mismatch"));
                            }
                        }
                        (None, false) => {}
                        (Some(_), false) => {
                            return Err(HipError::new(
                                0,
                                "EP batch: unexpected tag table on uniform layer",
                            ))
                        }
                        (None, true) => {
                            return Err(HipError::new(
                                0,
                                "EP batch: missing tag table on mixed layer",
                            ))
                        }
                    }
                    if l.ffn.expert_gate_up_ptrs.shape != vec![2 * config.num_experts] {
                        return Err(HipError::new(
                            0,
                            "EP batch: gate_up pointer table shape mismatch",
                        ));
                    }
                    if l.ffn.expert_down_ptrs.shape != vec![2 * config.num_experts] {
                        return Err(HipError::new(
                            0,
                            "EP batch: down pointer table shape mismatch",
                        ));
                    }
                }
            }
        }
    }
    if !seen_ranks.iter().all(|&b| b) {
        return Err(HipError::new(
            0,
            "EP batch: missing provenance rank (not all 0..3 present)",
        ));
    }
    // Global table equality across ranks, plus per-layer shape/batchability via single predicate
    for li in 0..config.n_layers {
        // collect global tables for this layer across ranks
        let mut ref_global: Option<Vec<(DType, DType)>> = None;
        for (r_idx, w) in weights_per_rank.iter().enumerate() {
            let layer = &w.layers[li];
            if let Some(ffn) = layer_moe_ffn(layer) {
                let global = ffn.global_expert_dtypes.as_ref().ok_or_else(|| {
                    HipError::new(
                        0,
                        &format!(
                            "EP batch: rank {r_idx} layer {li} missing global table for equality"
                        ),
                    )
                })?;
                let cur: Vec<(DType, DType)> = global.to_vec();
                match &ref_global {
                    None => ref_global = Some(cur),
                    Some(v) => {
                        if v != &cur {
                            return Err(HipError::new(0, &format!("EP batch: rank {r_idx} layer {li} global_expert_dtypes mismatch")));
                        }
                    }
                }
            }
        }
        // Validate ownership: each global expert occurs on exactly its mapped owner with sealed layout
        if let (Some(assign), Some(global)) = (&ref_assign, &ref_global) {
            for (gid, &owner) in assign.iter().enumerate() {
                let owner = owner as usize;
                for (rank, weights) in weights_per_rank.iter().enumerate() {
                    let locals = &weights
                        .ep_shard
                        .as_ref()
                        .expect("EP shard checked above")
                        .rank_seal()
                        .local_expert_descriptors[li];
                    let mut matching = locals
                        .iter()
                        .filter(|expert| expert.global_expert_id == gid);
                    let descriptor = matching.next();
                    if matching.next().is_some() {
                        return Err(HipError::new(
                            0,
                            &format!(
                                "EP batch: rank {rank} layer {li} duplicates global expert {gid}"
                            ),
                        ));
                    }
                    if rank == owner {
                        let descriptor = descriptor.ok_or_else(|| {
                            HipError::new(
                                0,
                                &format!(
                                    "EP batch: owner rank {rank} layer {li} missing global expert {gid}"
                                ),
                            )
                        })?;
                        if (descriptor.gate_up.gpu_dtype, descriptor.down.gpu_dtype) != global[gid]
                        {
                            return Err(HipError::new(
                                0,
                                &format!(
                                    "EP batch: owner rank {rank} layer {li} expert {gid} dtype seal mismatch"
                                ),
                            ));
                        }
                    } else if descriptor.is_some() {
                        return Err(HipError::new(
                            0,
                            &format!(
                                "EP batch: non-owner rank {rank} layer {li} contains global expert {gid}"
                            ),
                        ));
                    }
                }
            }
        }
        // Per-layer batchability via single source of truth and replicated seal equality for dense layers
        let arch = gpus.devices[0].arch.as_str();
        for (r_idx, w) in weights_per_rank.iter().enumerate() {
            qwen35_layer_batch_admissible(&w.layers[li], config, arch).map_err(|e| {
                HipError::new(
                    0,
                    &format!(
                        "EP batch: rank {r_idx} layer {li} not batch-admissible: {}",
                        e.message
                    ),
                )
            })?;
        }
        // Replicated seals equality for dense layers: compare descriptors across ranks
        let first_layer = &weights_per_rank[0].layers[li];
        let is_dense = matches!(
            first_layer,
            LayerWeights::DeltaNet(_) | LayerWeights::FullAttn(_)
        );
        if is_dense {
            let first_seal = &weights_per_rank[0]
                .ep_shard
                .as_ref()
                .unwrap()
                .rank_seal()
                .layer_seals[li];
            for (r_idx, w) in weights_per_rank.iter().enumerate().skip(1) {
                let cur = &w.ep_shard.as_ref().unwrap().rank_seal().layer_seals[li];
                if first_seal != cur {
                    return Err(HipError::new(
                        0,
                        &format!("EP batch: rank {r_idx} layer {li} replicated seal mismatch"),
                    ));
                }
            }
        }
    }
    // Cross-layer counts already validated; now ownership counts.
    if let Some(assign) = &ref_assign {
        let mut counts = vec![0usize; 4];
        for &owner in assign.iter() {
            counts[owner as usize] += 1;
        }
        for (r, c) in counts.iter().enumerate() {
            if *c == 0 {
                return Err(HipError::new(
                    0,
                    &format!("EP batch: expert assignment leaves rank {r} empty"),
                ));
            }
        }
        for (idx, w) in weights_per_rank.iter().enumerate() {
            let prov = w.ep_shard.as_ref().ok_or_else(|| {
                HipError::new(0, &format!("EP batch: rank {idx} missing provenance"))
            })?;
            let rank = prov.rank() as usize;
            let expected_owned = counts[rank];
            // Validate every MoE layer's local expert count matches owned set
            for (li, layer) in w.layers.iter().enumerate() {
                if let Some(ffn) = layer_moe_ffn(layer) {
                    if ffn.experts.len() != expected_owned {
                        return Err(HipError::new(0, &format!("EP batch: rank {idx} layer {li} loaded {} experts but assignment expects {expected_owned}", ffn.experts.len())));
                    }
                }
            }
        }
    }
    let ef_enabled = hipfire_config::developer_var("HIPFIRE_DN_STATE_EF")
        .map(|v| v != "0")
        .unwrap_or(true);
    if !ef_enabled {
        return Err(HipError::new(
            0,
            "EP batch: requires default F16 EF (HIPFIRE_DN_STATE_EF !=0)",
        ));
    }
    for (i, dev) in gpus.devices.iter().enumerate() {
        dev.ensure_attention_q8_0_kv_independent_lds(load_cfg.lane_capacity, config.head_dim)
            .map_err(|e| HipError::new(0, &format!("EP batch: rank {i} LDS {e}")))?;
    }
    let per_rank_decode = Qwen35DecodeBatchState::projected_allocation_bytes(
        config,
        load_cfg.max_batch,
        load_cfg.lane_capacity,
        load_cfg.repeat_capacity,
    )?;
    let per_rank_seed_pbs =
        PrefillBatchScratch::projected_allocation_bytes(config, load_cfg.prefill_chunk, false)?;
    let dim = config.dim as u64;
    let decode_partial: u64 = (load_cfg.max_batch as u64)
        .checked_mul(dim)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| HipError::new(0, "decode partial bytes overflow"))?;
    let seed_partial: u64 = (load_cfg.prefill_chunk as u64)
        .checked_mul(dim)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| HipError::new(0, "seed partial bytes overflow"))?;
    let requested_bytes_u64 = (load_cfg.max_batch.max(load_cfg.prefill_chunk) as u64)
        .checked_mul(dim)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| HipError::new(0, "peer requested bytes overflow"))?;
    let requested_bytes = usize::try_from(requested_bytes_u64)
        .map_err(|_| HipError::new(0, "peer requested bytes overflow usize"))?;
    let peer_bytes_per_rank =
        hipfire_runtime::multi_gpu::peer_reduce_scratch_bytes_per_rank(4, requested_bytes)
            .ok_or_else(|| HipError::new(0, "peer per-rank projection overflow"))?;
    let per_rank_total = per_rank_decode
        .checked_add(per_rank_seed_pbs)
        .and_then(|v| v.checked_add(decode_partial))
        .and_then(|v| v.checked_add(seed_partial))
        .and_then(|v| v.checked_add(peer_bytes_per_rank as u64))
        .ok_or_else(|| HipError::new(0, "per-rank total overflow"))?;
    // Per-device VRAM check using shared helper, not aggregate total vs min_free.
    for (idx, dev) in gpus.devices.iter().enumerate() {
        if let Ok((free, _)) = dev.hip.get_vram_info() {
            if per_rank_total > free as u64 {
                return Err(HipError::new(
                    0,
                    &format!(
                        "EP batch: rank {idx} projected {} bytes exceeds available {} bytes",
                        per_rank_total, free
                    ),
                ));
            }
        }
    }
    let moe_cnt = weights_per_rank[0]
        .layers
        .iter()
        .filter(|l| {
            matches!(
                l,
                LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_)
            )
        })
        .count();
    let rank_mask: u64 = 0x0f;
    Ok(Qwen35BatchCompatibility {
        rank_count: 4,
        rank_mask,
        moe_layer_count: moe_cnt,
        topology: Qwen35EpTopology::ExpertParallel,
        reduce: Qwen35EpReduce::PeerRootedF32,
        max_batch: load_cfg.max_batch,
        lane_capacity: load_cfg.lane_capacity,
        repeat_capacity: load_cfg.repeat_capacity,
        prefill_chunk: load_cfg.prefill_chunk,
        per_rank_decode_bytes: per_rank_decode,
        per_rank_seed_pbs_bytes: per_rank_seed_pbs,
        decode_partial_bytes: decode_partial,
        seed_partial_bytes: seed_partial,
        peer_bytes_per_rank,
        per_rank_total_bytes: per_rank_total,
    })
}
impl Qwen35Weights {
    pub fn validate_ep_batch_compatibility(
        gpus: &Gpus,
        weights_per_rank: &[Qwen35Weights],
        config: &Qwen35Config,
        load_cfg: &Qwen35BatchLoadConfig,
    ) -> HipResult<Qwen35BatchCompatibility> {
        validate_ep_batch_compatibility(gpus, weights_per_rank, config, load_cfg)
    }
}
/// Private lane state for the 4-rank EP batch owner.
/// `Vacant` = empty, `Seeding` = prompt seeding in-flight, `Ready{next_position}` = decode ready,
/// `Poisoned` = failed mutation, must be destructively reset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LaneState {
    Vacant,
    Seeding,
    Ready { next_position: usize },
    Poisoned,
}

/// All-rank EP batch owner with replicated batch state and attested receipts.
pub struct Qwen35DecodeBatchEpState {
    ranks: Vec<Qwen35DecodeBatchState>,
    decode_partials: Vec<GpuTensor>,
    seed_pbs: Vec<PrefillBatchScratch>,
    seed_partials: Vec<GpuTensor>,
    scratches: Vec<Qwen35Scratch>,
    lane_states: Vec<LaneState>,
    poison_mask: u64,
    epoch: u64,
    max_batch: usize,
    lane_capacity: usize,
    repeat_capacity: usize,
    prefill_chunk: usize,
    moe_layer_count: usize,
    dim: usize,
    norm_eps: f32,
    expert_to_rank: Box<[u8]>,
    peer_lease: Option<hipfire_runtime::multi_gpu::PeerReduceScratchLease>,
}

/// Transactional ownership guard for `Qwen35DecodeBatchEpState::new`.
/// Holds per-rank allocations plus the peer lease; on error rolls back on
/// owning devices, preserving the first error and the first cleanup failure.
struct EpBatchBuildGuard {
    ranks: Vec<Option<Qwen35DecodeBatchState>>,
    decode_partials: Vec<Option<GpuTensor>>,
    seed_pbs: Vec<Option<PrefillBatchScratch>>,
    seed_partials: Vec<Option<GpuTensor>>,
    scratches: Vec<Option<Qwen35Scratch>>,
    lease: Option<hipfire_runtime::multi_gpu::PeerReduceScratchLease>,
}

impl EpBatchBuildGuard {
    fn new(n: usize) -> Self {
        Self {
            ranks: (0..n).map(|_| None).collect(),
            decode_partials: (0..n).map(|_| None).collect(),
            seed_pbs: (0..n).map(|_| None).collect(),
            seed_partials: (0..n).map(|_| None).collect(),
            scratches: (0..n).map(|_| None).collect(),
            lease: None,
        }
    }
    fn set_rank(&mut self, idx: usize, v: Qwen35DecodeBatchState) {
        self.ranks[idx] = Some(v);
    }
    fn set_decode_partial(&mut self, idx: usize, v: GpuTensor) {
        self.decode_partials[idx] = Some(v);
    }
    fn set_seed_pbs(&mut self, idx: usize, v: PrefillBatchScratch) {
        self.seed_pbs[idx] = Some(v);
    }
    fn set_seed_partial(&mut self, idx: usize, v: GpuTensor) {
        self.seed_partials[idx] = Some(v);
    }
    fn set_scratch(&mut self, idx: usize, v: Qwen35Scratch) {
        self.scratches[idx] = Some(v);
    }
    fn set_lease(&mut self, lease: hipfire_runtime::multi_gpu::PeerReduceScratchLease) {
        self.lease = Some(lease);
    }
    /// Rollback on owning devices, attempting every free, preserving init error plus first cleanup error.
    fn rollback(mut self, gpus: &mut Gpus, init_err: HipError) -> HipError {
        let mut cleanup_first: Option<HipError> = None;
        let n = self.ranks.len();
        for rank in 0..n {
            let _ = gpus.devices[rank].bind_thread().map_err(|e| {
                if cleanup_first.is_none() {
                    cleanup_first = Some(e);
                }
            });
            // Free each slot on its owner; capture first cleanup failure.
            if let Some(state) = self.ranks[rank].take() {
                if let Err(e) = state.free_gpu(&mut gpus.devices[rank]) {
                    if cleanup_first.is_none() {
                        cleanup_first = Some(e);
                    }
                }
            }
            if let Some(t) = self.decode_partials[rank].take() {
                if let Err(e) = gpus.devices[rank].free_tensor(t) {
                    if cleanup_first.is_none() {
                        cleanup_first = Some(e);
                    }
                }
            }
            if let Some(p) = self.seed_pbs[rank].take() {
                if let Err(e) = p.free_gpu(&mut gpus.devices[rank]) {
                    if cleanup_first.is_none() {
                        cleanup_first = Some(e);
                    }
                }
            }
            if let Some(t) = self.seed_partials[rank].take() {
                if let Err(e) = gpus.devices[rank].free_tensor(t) {
                    if cleanup_first.is_none() {
                        cleanup_first = Some(e);
                    }
                }
            }
            if let Some(s) = self.scratches[rank].take() {
                if let Err(e) = s.free_gpu(&mut gpus.devices[rank]) {
                    if cleanup_first.is_none() {
                        cleanup_first = Some(e);
                    }
                }
            }
            let _ = gpus.devices[rank].hip.device_synchronize().map_err(|e| {
                if cleanup_first.is_none() {
                    cleanup_first = Some(e);
                }
            });
        }
        // Release lease last.
        if let Some(lease) = self.lease.take() {
            if let Err(e) = gpus.release_peer_reduce_scratch(&lease) {
                if cleanup_first.is_none() {
                    cleanup_first = Some(e);
                }
            }
        }
        if let Some(ce) = cleanup_first {
            HipError::new(0, &format!("{}; cleanup: {}", init_err.message, ce.message))
        } else {
            init_err
        }
    }
    fn commit(
        self,
    ) -> (
        Vec<Qwen35DecodeBatchState>,
        Vec<GpuTensor>,
        Vec<PrefillBatchScratch>,
        Vec<GpuTensor>,
        Vec<Qwen35Scratch>,
        Option<hipfire_runtime::multi_gpu::PeerReduceScratchLease>,
    ) {
        let ranks = self
            .ranks
            .into_iter()
            .map(|o| o.expect("commit: rank slot empty"))
            .collect();
        let decode_partials = self
            .decode_partials
            .into_iter()
            .map(|o| o.expect("commit: decode_partial empty"))
            .collect();
        let seed_pbs = self
            .seed_pbs
            .into_iter()
            .map(|o| o.expect("commit: seed_pbs empty"))
            .collect();
        let seed_partials = self
            .seed_partials
            .into_iter()
            .map(|o| o.expect("commit: seed_partial empty"))
            .collect();
        let scratches = self
            .scratches
            .into_iter()
            .map(|o| o.expect("commit: scratch empty"))
            .collect();
        (
            ranks,
            decode_partials,
            seed_pbs,
            seed_partials,
            scratches,
            self.lease,
        )
    }
}
impl Qwen35DecodeBatchEpState {
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }
    pub fn lane_capacity(&self) -> usize {
        self.lane_capacity
    }
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
    pub fn poison_mask(&self) -> u64 {
        self.poison_mask
    }
    pub fn lane_state(&self, lane: usize) -> Option<LaneState> {
        self.lane_states.get(lane).copied()
    }
    pub fn new(
        gpus: &mut Gpus,
        weights_per_rank: &[Qwen35Weights],
        config: &Qwen35Config,
        load_cfg: &Qwen35BatchLoadConfig,
    ) -> HipResult<Self> {
        let compat = validate_ep_batch_compatibility(gpus, weights_per_rank, config, load_cfg)?;
        // Central max_batch bounds before shifts.
        valid_lane_mask(load_cfg.max_batch)?;
        hipfire_runtime::ep::ensure_rank_streams(gpus)
            .map_err(|e| HipError::new(0, &e.to_string()))?;
        let n = gpus.devices.len();
        debug_assert_eq!(n, 4);
        // Acquire unique peer lease transactionally before any rank publication, using sibling's exact projection.
        // requested_bytes = max_elems * 4 where max_elems = max_batch.max(prefill_chunk)*dim.
        let max_elems = load_cfg
            .max_batch
            .max(load_cfg.prefill_chunk)
            .checked_mul(config.dim)
            .ok_or_else(|| HipError::new(0, "EP batch: max_elems overflow"))?;
        let bytes = max_elems
            .checked_mul(4)
            .ok_or_else(|| HipError::new(0, "EP batch: peer bytes overflow"))?;
        let mut guard = EpBatchBuildGuard::new(n);
        let lease = match gpus.acquire_peer_reduce_scratch(bytes) {
            Ok(l) => l,
            Err(e) => return Err(guard.rollback(gpus, e)),
        };
        guard.set_lease(lease);
        for rank in 0..n {
            if let Err(e) = gpus.devices[rank].bind_thread() {
                return Err(guard.rollback(gpus, e));
            }
            let state = match Qwen35DecodeBatchState::new(
                &mut gpus.devices[rank],
                config,
                load_cfg.max_batch,
                load_cfg.lane_capacity,
                load_cfg.repeat_capacity,
            ) {
                Ok(s) => s,
                Err(e) => return Err(guard.rollback(gpus, e)),
            };
            guard.set_rank(rank, state);
            let partial =
                match gpus.devices[rank].zeros(&[load_cfg.max_batch * config.dim], DType::F32) {
                    Ok(t) => t,
                    Err(e) => return Err(guard.rollback(gpus, e)),
                };
            guard.set_decode_partial(rank, partial);
            let spbs = match PrefillBatchScratch::new_opt(
                &mut gpus.devices[rank],
                config,
                load_cfg.prefill_chunk,
                false,
            ) {
                Ok(p) => p,
                Err(e) => return Err(guard.rollback(gpus, e)),
            };
            guard.set_seed_pbs(rank, spbs);
            let spartial = match gpus.devices[rank]
                .zeros(&[load_cfg.prefill_chunk * config.dim], DType::F32)
            {
                Ok(t) => t,
                Err(e) => return Err(guard.rollback(gpus, e)),
            };
            guard.set_seed_partial(rank, spartial);
            let scratch =
                match Qwen35Scratch::new(&mut gpus.devices[rank], config, load_cfg.repeat_capacity)
                {
                    Ok(s) => s,
                    Err(e) => return Err(guard.rollback(gpus, e)),
                };
            guard.set_scratch(rank, scratch);
        }
        let (ranks, decode_partials, seed_pbs, seed_partials, scratches, lease_opt) =
            guard.commit();
        let lane_states = vec![LaneState::Vacant; load_cfg.max_batch];
        let expert_to_rank = weights_per_rank[0]
            .ep_shard
            .as_ref()
            .expect("admission validated provenance")
            .expert_to_rank()
            .to_vec()
            .into_boxed_slice();
        Ok(Self {
            ranks,
            decode_partials,
            seed_pbs,
            seed_partials,
            scratches,
            lane_states,
            poison_mask: 0,
            epoch: 0,
            max_batch: load_cfg.max_batch,
            lane_capacity: load_cfg.lane_capacity,
            repeat_capacity: load_cfg.repeat_capacity,
            prefill_chunk: load_cfg.prefill_chunk,
            moe_layer_count: compat.moe_layer_count(),
            dim: config.dim,
            norm_eps: config.norm_eps,
            expert_to_rank,
            peer_lease: lease_opt,
        })
    }
    fn reserve_next_epoch(&self) -> HipResult<u64> {
        self.epoch
            .checked_add(1)
            .ok_or_else(|| HipError::new(0, "EP batch: epoch overflow"))
    }
    fn commit_or_poison<T>(
        &mut self,
        affected_mask: u64,
        reserved_epoch: u64,
        result: HipResult<T>,
    ) -> HipResult<T> {
        match result {
            Ok(v) => {
                self.epoch = reserved_epoch;
                Ok(v)
            }
            Err(e) => {
                self.poison_lanes(affected_mask);
                Err(e)
            }
        }
    }
    fn checked_advance_epoch(&mut self) -> HipResult<u64> {
        let next = self
            .epoch
            .checked_add(1)
            .ok_or_else(|| HipError::new(0, "EP batch: epoch overflow"))?;
        self.epoch = next;
        Ok(next)
    }
    fn poison_lanes(&mut self, mask: u64) {
        for lane in 0..self.lane_states.len() {
            if (mask >> lane) & 1 != 0 {
                self.lane_states[lane] = LaneState::Poisoned;
            }
        }
        self.poison_mask |= mask;
    }
    fn clear_poison_lane(&mut self, lane: usize) {
        self.poison_mask &= !(1u64 << lane);
        if self.lane_states[lane] == LaneState::Poisoned {
            self.lane_states[lane] = LaneState::Vacant;
        }
    }
    fn is_poisoned(&self, lane: usize) -> bool {
        (self.poison_mask >> lane) & 1 != 0
    }
    fn reset_lane_internal(
        &mut self,
        gpus: &mut Gpus,
        config: &Qwen35Config,
        lane: usize,
    ) -> HipResult<()> {
        if lane >= self.max_batch {
            return Err(HipError::new(0, "reset_lane: lane out of range"));
        }
        for rank in 0..gpus.devices.len() {
            gpus.devices[rank].bind_thread()?;
            let state = &mut self.ranks[rank];
            state.reset_lane(&mut gpus.devices[rank], config, lane)?;
            let lane_slice = self.decode_partials[rank].sub_offset(lane * config.dim, config.dim);
            if let Some(stream) = gpus.devices[rank].active_stream.as_ref() {
                gpus.devices[rank].hip.memset_async(
                    &lane_slice.buf,
                    0,
                    lane_slice.buf.size(),
                    stream,
                )?;
            } else {
                gpus.devices[rank]
                    .hip
                    .memset(&lane_slice.buf, 0, lane_slice.buf.size())?;
            }
        }
        for rank in 0..gpus.devices.len() {
            gpus.devices[rank].bind_thread()?;
            gpus.devices[rank].hip.device_synchronize()?;
        }
        Ok(())
    }
    pub fn reset_all(&mut self, gpus: &mut Gpus) -> HipResult<()> {
        let n = gpus.devices.len();
        if n != 4 || self.ranks.len() != 4 {
            return Err(HipError::new(0, "reset_all: rank count mismatch"));
        }
        let affected_mask = valid_lane_mask(self.max_batch)?;
        let reserved_epoch = self.reserve_next_epoch()?;
        // All preflight (validations + epoch reservation) done before any device mutation.
        let inner = (|| -> HipResult<()> {
            let mut first_err: Option<HipError> = None;
            for rank in 0..n {
                if let Err(e) = gpus.devices[rank].bind_thread() {
                    first_err.get_or_insert(e);
                    continue;
                }
                let state = &mut self.ranks[rank];
                if let Err(e) = state.reset(&mut gpus.devices[rank]) {
                    first_err.get_or_insert(e);
                    continue;
                }
                let buf = &self.decode_partials[rank].buf;
                let res = if let Some(stream) = gpus.devices[rank].active_stream.as_ref() {
                    gpus.devices[rank]
                        .hip
                        .memset_async(buf, 0, buf.size(), stream)
                } else {
                    gpus.devices[rank].hip.memset(buf, 0, buf.size())
                };
                if let Err(e) = res {
                    first_err.get_or_insert(e);
                }
            }
            // Attempt every sync even after failure, preserve first error.
            let mut sync_err: Option<HipError> = None;
            for rank in 0..n {
                if let Err(e) = gpus.devices[rank]
                    .bind_thread()
                    .and_then(|_| gpus.devices[rank].hip.device_synchronize())
                {
                    if first_err.is_none() {
                        first_err.get_or_insert(e);
                    } else if sync_err.is_none() {
                        sync_err = Some(e);
                    }
                }
            }
            if let Some(e) = first_err {
                return Err(e);
            }
            if let Some(e) = sync_err {
                return Err(e);
            }
            Ok(())
        })();
        match inner {
            Ok(()) => {
                self.poison_mask = 0;
                for s in self.lane_states.iter_mut() {
                    *s = LaneState::Vacant;
                }
                self.epoch = reserved_epoch;
                Ok(())
            }
            Err(e) => {
                self.poison_lanes(affected_mask);
                Err(e)
            }
        }
    }
    pub fn reset_lane(
        &mut self,
        gpus: &mut Gpus,
        config: &Qwen35Config,
        lane: usize,
    ) -> HipResult<()> {
        // Bounds precede shifts/poison checks per 3.5.
        if lane >= self.max_batch {
            return Err(HipError::new(0, "reset_lane: lane out of range"));
        }
        let lane_mask = lane_bit(lane, self.max_batch)?;
        let reserved_epoch = self.reserve_next_epoch()?;
        let inner = self.reset_lane_internal(gpus, config, lane);
        match inner {
            Ok(()) => {
                self.poison_mask &= !lane_mask;
                if self.lane_states[lane] == LaneState::Poisoned {
                    self.lane_states[lane] = LaneState::Vacant;
                }
                self.lane_states[lane] = LaneState::Vacant;
                self.epoch = reserved_epoch;
                Ok(())
            }
            Err(e) => {
                self.poison_lanes(lane_mask);
                Err(e)
            }
        }
    }
    pub fn prefill_lane(
        &mut self,
        gpus: &mut Gpus,
        weights_per_rank: &[Qwen35Weights],
        config: &Qwen35Config,
        lane: usize,
        tokens: &[u32],
    ) -> HipResult<Qwen35EpBatchReceipt> {
        let n = gpus.devices.len();
        if n != 4 || weights_per_rank.len() != 4 || self.ranks.len() != 4 {
            return Err(HipError::new(0, "prefill_lane: rank count mismatch"));
        }
        if lane >= self.max_batch {
            return Err(HipError::new(0, "prefill_lane: lane out of range"));
        }
        let lane_mask = lane_bit(lane, self.max_batch)?;
        if tokens.is_empty() || tokens.len() >= self.lane_capacity {
            return Err(HipError::new(
                0,
                "prefill_lane: token count invalid or leaves no decode capacity",
            ));
        }
        if self.is_poisoned(lane) {
            return Err(HipError::new(
                0,
                "prefill_lane: lane is poisoned, reset required",
            ));
        }
        let prov = weights_per_rank[0]
            .ep_shard
            .as_ref()
            .ok_or_else(|| HipError::new(0, "prefill_lane: missing provenance"))?;
        if prov.expert_to_rank() != self.expert_to_rank.as_ref() {
            return Err(HipError::new(
                0,
                "prefill_lane: expert assignment mismatch vs admitted",
            ));
        }
        if config.dim != self.dim {
            return Err(HipError::new(0, "prefill_lane: config dim mismatch"));
        }
        if config.norm_eps.to_bits() != self.norm_eps.to_bits() {
            return Err(HipError::new(0, "prefill_lane: config norm_eps mismatch"));
        }
        let chunks: Vec<&[u32]> = tokens.chunks(self.prefill_chunk).collect();
        let num_chunks = chunks.len();
        if num_chunks == 0 {
            return Err(HipError::new(0, "prefill_lane: zero chunks"));
        }
        let expected_collectives_u64 = (num_chunks as u64)
            .checked_mul(self.moe_layer_count as u64)
            .ok_or_else(|| HipError::new(0, "prefill expected collectives overflow"))?;
        let expected_collectives = u32::try_from(expected_collectives_u64)
            .map_err(|_| HipError::new(0, "prefill collectives overflow u32"))?;
        let rows_u32 = Qwen35EpBatchReceipt::rows_from_usize(tokens.len())?;
        let reserved_epoch = self.reserve_next_epoch()?;
        hipfire_runtime::ep::ensure_rank_streams(gpus)
            .map_err(|e| HipError::new(0, &e.to_string()))?;
        let dim = config.dim;
        let inner = (|| -> HipResult<u32> {
            // First device mutation is reset_lane_internal — after this, every error poisons.
            self.reset_lane_internal(gpus, config, lane)?;
            self.lane_states[lane] = LaneState::Seeding;
            let mut kv_lanes: Vec<llama::KvCache> = Vec::with_capacity(n);
            let mut dn_lanes: Vec<DeltaNetState> = Vec::with_capacity(n);
            for rank in 0..n {
                gpus.devices[rank].bind_thread()?;
                let kv = self.ranks[rank]
                    .kv_cache
                    .q8_lane_view(lane, self.lane_capacity)
                    .map_err(|e| {
                        HipError::new(
                            0,
                            &format!("prefill lane kv view rank {rank}: {}", e.message),
                        )
                    })?;
                let dn = self.ranks[rank]
                    .dn_state
                    .q8_lane_view(config, lane, self.max_batch)
                    .map_err(|e| {
                        HipError::new(
                            0,
                            &format!("prefill lane dn view rank {rank}: {}", e.message),
                        )
                    })?;
                kv_lanes.push(kv);
                dn_lanes.push(dn);
            }
            let mut observed: u32 = 0;
            for (chunk_idx, chunk) in chunks.iter().enumerate() {
                let chunk_n = chunk.len();
                let start_pos = chunk_idx * self.prefill_chunk;
                let mut delta_off: usize = 0;
                let mut fa_off: usize = 0;
                for layer_idx in 0..config.n_layers {
                    let is_moe = matches!(
                        &weights_per_rank[0].layers[layer_idx],
                        LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_)
                    );
                    if is_moe {
                        for rank in 0..n {
                            gpus.devices[rank].bind_thread()?;
                            let partial = &self.seed_partials[rank];
                            let bytes = chunk_n * dim * 4;
                            if let Some(stream) = gpus.devices[rank].active_stream.as_ref() {
                                gpus.devices[rank].hip.memset_async(
                                    &partial.buf,
                                    0,
                                    bytes,
                                    stream,
                                )?;
                            } else {
                                gpus.devices[rank].hip.memset(&partial.buf, 0, bytes)?;
                            }
                        }
                    }
                    for rank in 0..n {
                        gpus.devices[rank].bind_thread()?;
                        let band = PrefillBandCtx {
                            layer_start: layer_idx,
                            layer_end: layer_idx + 1,
                            delta_layer_offset: delta_off,
                            kv_layer_offset: fa_off,
                            is_first_band: layer_idx == 0,
                            is_last_band: false,
                            givens_cos: None,
                            givens_sin: None,
                        };
                        let routed_out = if is_moe {
                            Some(self.seed_partials[rank].sub_offset(0, chunk_n * dim))
                        } else {
                            None
                        };
                        let rank_scratch = &self.scratches[rank];
                        let rank_pbs = &self.seed_pbs[rank];
                        let rank_kv = &mut kv_lanes[rank];
                        let rank_dn = &mut dn_lanes[rank];
                        forward_batch_chunk_impl(
                            &mut gpus.devices[rank],
                            &weights_per_rank[rank],
                            config,
                            chunk,
                            start_pos,
                            rank_kv,
                            rank_dn,
                            rank_scratch,
                            rank_pbs,
                            None,
                            None,
                            None,
                            0,
                            None,
                            false,
                            false,
                            Some(&band),
                            None,
                            false,
                            None,
                            routed_out.as_ref(),
                            BatchSemantics::Sequential,
                        )?;
                    }
                    if is_moe {
                        let count = chunk_n * dim;
                        let bufs: Vec<&hip_bridge::DeviceBuffer> =
                            self.seed_partials.iter().map(|t| &t.buf).collect();
                        let lease = self
                            .peer_lease
                            .as_ref()
                            .ok_or_else(|| HipError::new(0, "prefill_lane: missing peer lease"))?;
                        gpus.all_reduce_sum_f32_peer_rooted_leased(lease, &bufs, count)?;
                        observed = observed
                            .checked_add(1)
                            .ok_or_else(|| HipError::new(0, "prefill observed overflow"))?;
                        for rank in 0..n {
                            gpus.devices[rank].bind_thread()?;
                            let dst = self.seed_pbs[rank].x_batch.sub_offset(0, chunk_n * dim);
                            let src = self.seed_partials[rank].sub_offset(0, chunk_n * dim);
                            gpus.devices[rank].add_inplace_f32(&dst, &src)?;
                        }
                    }
                    match config.layer_types[layer_idx] {
                        LayerType::LinearAttention => delta_off += 1,
                        LayerType::FullAttention => fa_off += 1,
                    }
                }
            }
            if observed != expected_collectives {
                return Err(HipError::new(
                    0,
                    &format!(
                        "prefill observed {} != expected {}",
                        observed, expected_collectives
                    ),
                ));
            }
            {
                gpus.devices[0].bind_thread()?;
                let gpu = &mut gpus.devices[0];
                let last_chunk_n = chunks.last().unwrap().len();
                let last_x = self.seed_pbs[0]
                    .x_batch
                    .sub_offset((last_chunk_n - 1) * dim, dim);
                let tmp = &self.scratches[0].tmp;
                gpu.rmsnorm_f32(
                    &last_x,
                    &weights_per_rank[0].output_norm,
                    tmp,
                    config.norm_eps,
                )?;
                let logits_lane = self.ranks[0]
                    .logits
                    .sub_offset(lane * config.vocab_size, config.vocab_size);
                let rot = &self.scratches[0].x_rot;
                lm_head_batched(gpu, &weights_per_rank[0].output, tmp, rot, &logits_lane, 1)?;
            }
            for rank in 0..n {
                gpus.devices[rank].bind_thread()?;
                gpus.devices[rank].hip.device_synchronize()?;
            }
            Ok(observed)
        })();
        match inner {
            Ok(observed) => {
                self.lane_states[lane] = LaneState::Ready {
                    next_position: tokens.len(),
                };
                self.poison_mask &= !lane_mask;
                self.epoch = reserved_epoch;
                Ok(Qwen35EpBatchReceipt::new_attested(
                    reserved_epoch,
                    rows_u32,
                    observed,
                ))
            }
            Err(e) => {
                self.poison_lanes(lane_mask);
                Err(e)
            }
        }
    }
    pub fn forward_tick(
        &mut self,
        gpus: &mut Gpus,
        weights_per_rank: &[Qwen35Weights],
        config: &Qwen35Config,
        active_mask: u64,
        tokens: &[u32],
        positions: &[usize],
    ) -> HipResult<Qwen35EpBatchReceipt> {
        let n = gpus.devices.len();
        if n != 4 || weights_per_rank.len() != 4 || self.ranks.len() != 4 {
            return Err(HipError::new(0, "forward_tick: rank count mismatch"));
        }
        if tokens.len() != self.max_batch || positions.len() != self.max_batch {
            return Err(HipError::new(
                0,
                "forward_tick: tokens/positions must be fixed-slot max_batch",
            ));
        }
        if active_mask == 0 {
            return Err(HipError::new(0, "forward_tick: active_mask empty"));
        }
        let valid_mask = valid_lane_mask(self.max_batch)?;
        if active_mask & !valid_mask != 0 {
            return Err(HipError::new(0, "forward_tick: active_mask out of range"));
        }
        // Preflight: each active lane is Ready and exact next_position.
        for lane in 0..self.max_batch {
            if (active_mask >> lane) & 1 == 0 {
                continue;
            }
            if self.is_poisoned(lane) {
                return Err(HipError::new(
                    0,
                    &format!("forward_tick: lane {lane} is poisoned"),
                ));
            }
            match self.lane_states[lane] {
                LaneState::Ready { next_position } => {
                    if positions[lane] != next_position {
                        return Err(HipError::new(
                            0,
                            &format!(
                                "forward_tick: lane {lane} position {} != expected {}",
                                positions[lane], next_position
                            ),
                        ));
                    }
                    if positions[lane] >= self.lane_capacity {
                        return Err(HipError::new(
                            0,
                            &format!("forward_tick: lane {lane} position exceeds capacity"),
                        ));
                    }
                }
                _ => {
                    return Err(HipError::new(
                        0,
                        &format!("forward_tick: lane {lane} not Ready"),
                    ))
                }
            }
        }
        // Validate provenance/config before mutation (allocation-free matches).
        let prov = weights_per_rank[0]
            .ep_shard
            .as_ref()
            .ok_or_else(|| HipError::new(0, "forward_tick: missing provenance"))?;
        if prov.expert_to_rank() != self.expert_to_rank.as_ref() {
            return Err(HipError::new(0, "forward_tick: provenance mismatch"));
        }
        if config.dim != self.dim {
            return Err(HipError::new(0, "forward_tick: config mismatch"));
        }
        if config.norm_eps.to_bits() != self.norm_eps.to_bits() {
            return Err(HipError::new(0, "forward_tick: config norm_eps mismatch"));
        }
        // Pure preflight of conversions and checked arithmetic before any device operation.
        let reserved_epoch = self.reserve_next_epoch()?;
        let rows_u32 = Qwen35EpBatchReceipt::rows_from_usize(active_mask.count_ones() as usize)?;
        let moe_layer_count_u32 = u32::try_from(self.moe_layer_count)
            .map_err(|_| HipError::new(0, "forward_tick: moe_layer_count overflow u32"))?;
        let b = self.max_batch;
        let dim = config.dim;
        let elem_count = b
            .checked_mul(dim)
            .ok_or_else(|| HipError::new(0, "forward_tick: elem count overflow"))?;
        let _bytes = elem_count
            .checked_mul(4)
            .ok_or_else(|| HipError::new(0, "forward_tick: bytes overflow"))?;
        // Validate token/position capacities before mutation.
        for &p in positions.iter() {
            if p >= self.lane_capacity {
                return Err(HipError::new(
                    0,
                    "forward_tick: position exceeds lane_capacity",
                ));
            }
        }
        // Also ensure each active position was already checked above; inactive positions still range-checked.
        hipfire_runtime::ep::ensure_rank_streams(gpus)
            .map_err(|e| HipError::new(0, &e.to_string()))?;
        let full_mask = valid_mask;
        let inner = (|| -> HipResult<u32> {
            let mut delta_off: usize = 0;
            let mut fa_off: usize = 0;
            let mut observed: u32 = 0;
            for layer_idx in 0..config.n_layers {
                let is_moe = matches!(
                    &weights_per_rank[0].layers[layer_idx],
                    LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_)
                );
                if is_moe {
                    for rank in 0..n {
                        gpus.devices[rank].bind_thread()?;
                        let partial = &self.decode_partials[rank];
                        let bytes = b * dim * 4;
                        if let Some(stream) = gpus.devices[rank].active_stream.as_ref() {
                            gpus.devices[rank]
                                .hip
                                .memset_async(&partial.buf, 0, bytes, stream)?;
                        } else {
                            gpus.devices[rank].hip.memset(&partial.buf, 0, bytes)?;
                        }
                    }
                }
                // Invariant: decode `pbs` per rank and `seed_pbs` are separate
                // allocations; `prefill_lane` correctly uses `seed_pbs` with
                // `false,false`, while `forward_tick` has no external
                // `prepare_decode_batch_inputs` and no peer-copy. Only band 0
                // on every rank owns staging of host token/position/embedding
                // inputs; later bands must reuse the transformed residual in
                // `pbs.x_batch` and must not re-upload or re-embed.
                let inputs_prepared = ep_tick_inputs_prepared(layer_idx);
                for rank in 0..n {
                    gpus.devices[rank].bind_thread()?;
                    let band = PrefillBandCtx {
                        layer_start: layer_idx,
                        layer_end: layer_idx + 1,
                        delta_layer_offset: delta_off,
                        kv_layer_offset: fa_off,
                        is_first_band: layer_idx == 0,
                        is_last_band: false,
                        givens_cos: None,
                        givens_sin: None,
                    };
                    let routed_out = if is_moe {
                        Some(self.decode_partials[rank].sub_offset(0, b * dim))
                    } else {
                        None
                    };
                    let rank_state = &mut self.ranks[rank];
                    let rank_scratch = &self.scratches[rank];
                    let rank_pbs = &rank_state.pbs;
                    forward_batch_chunk_impl(
                        &mut gpus.devices[rank],
                        &weights_per_rank[rank],
                        config,
                        tokens,
                        0,
                        &mut rank_state.kv_cache,
                        &mut rank_state.dn_state,
                        rank_scratch,
                        rank_pbs,
                        None,
                        None,
                        None,
                        0,
                        None,
                        inputs_prepared,
                        inputs_prepared,
                        Some(&band),
                        None,
                        false,
                        None,
                        routed_out.as_ref(),
                        BatchSemantics::Independent {
                            positions,
                            lane_capacity: self.lane_capacity,
                            active_mask,
                        },
                    )?;
                }
                if is_moe {
                    // Zero inactive rows before the leased reduce so they contribute +0.0f32 in rooted order.
                    if active_mask != full_mask {
                        for rank in 0..n {
                            gpus.devices[rank].bind_thread()?;
                            gpus.devices[rank].zero_inactive_rows_f32(
                                &self.decode_partials[rank],
                                b,
                                dim,
                                active_mask,
                            )?;
                        }
                    }
                    let count = b * dim;
                    let bufs: Vec<&hip_bridge::DeviceBuffer> =
                        self.decode_partials.iter().map(|t| &t.buf).collect();
                    let lease = self
                        .peer_lease
                        .as_ref()
                        .ok_or_else(|| HipError::new(0, "forward_tick: missing peer lease"))?;
                    gpus.all_reduce_sum_f32_peer_rooted_leased(lease, &bufs, count)?;
                    observed = observed
                        .checked_add(1)
                        .ok_or_else(|| HipError::new(0, "forward_tick observed overflow"))?;
                    for rank in 0..n {
                        gpus.devices[rank].bind_thread()?;
                        let rank_state = &self.ranks[rank];
                        let dst = rank_state.pbs.x_batch.sub_offset(0, b * dim);
                        let src = self.decode_partials[rank].sub_offset(0, b * dim);
                        gpus.devices[rank].add_inplace_f32(&dst, &src)?;
                    }
                }
                match config.layer_types[layer_idx] {
                    LayerType::LinearAttention => delta_off += 1,
                    LayerType::FullAttention => fa_off += 1,
                }
            }
            // Final rank-0 norm/head only over contiguous active spans (allocation-free).
            {
                gpus.devices[0].bind_thread()?;
                let gpu = &mut gpus.devices[0];
                let rank0 = &self.ranks[0];
                if active_mask == full_mask {
                    let src = &rank0.pbs.x_batch;
                    let dst = &rank0.final_hidden;
                    gpu.rmsnorm_batched(
                        src,
                        &weights_per_rank[0].output_norm,
                        dst,
                        b,
                        dim,
                        config.norm_eps,
                    )?;
                    let logits = rank0.logits.sub_offset(0, b * config.vocab_size);
                    let rot = rank0.lm_rot.sub_offset(0, b * dim);
                    lm_head_batched(gpu, &weights_per_rank[0].output, dst, &rot, &logits, b)?;
                } else {
                    for_each_active_span(active_mask, b, |start, len| {
                        let src = rank0.pbs.x_batch.sub_offset(start * dim, len * dim);
                        let dst = rank0.final_hidden.sub_offset(start * dim, len * dim);
                        gpu.rmsnorm_batched(
                            &src,
                            &weights_per_rank[0].output_norm,
                            &dst,
                            len,
                            dim,
                            config.norm_eps,
                        )?;
                        let logits = rank0
                            .logits
                            .sub_offset(start * config.vocab_size, len * config.vocab_size);
                        let rot = rank0.lm_rot.sub_offset(start * dim, len * dim);
                        lm_head_batched(
                            gpu,
                            &weights_per_rank[0].output,
                            &dst,
                            &rot,
                            &logits,
                            len,
                        )?;
                        Ok(())
                    })?;
                }
            }
            for rank in 0..n {
                gpus.devices[rank].bind_thread()?;
                gpus.devices[rank].hip.device_synchronize()?;
            }
            if observed != moe_layer_count_u32 {
                return Err(HipError::new(
                    0,
                    &format!(
                        "forward_tick observed {} != expected {}",
                        observed, moe_layer_count_u32
                    ),
                ));
            }
            Ok(observed)
        })();
        match inner {
            Ok(observed) => {
                for lane in 0..self.max_batch {
                    if (active_mask >> lane) & 1 == 0 {
                        continue;
                    }
                    if let LaneState::Ready { next_position } = self.lane_states[lane] {
                        self.lane_states[lane] = LaneState::Ready {
                            next_position: next_position + 1,
                        };
                    }
                }
                self.epoch = reserved_epoch;
                Ok(Qwen35EpBatchReceipt::new_attested(
                    reserved_epoch,
                    rows_u32,
                    observed,
                ))
            }
            Err(e) => {
                self.poison_lanes(active_mask);
                Err(e)
            }
        }
    }
    pub fn sample(
        &self,
        gpus: &mut Gpus,
        config: &Qwen35Config,
        batch_size: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<u32>,
        rng_state: u32,
    ) -> HipResult<(Vec<u32>, u32)> {
        if batch_size == 0 || batch_size > self.max_batch {
            return Err(HipError::new(0, "sample: batch size out of range"));
        }
        // Bounds precede poison/Ready inspection. Require every requested prefix lane to be Ready.
        for lane in 0..batch_size {
            match self.lane_states.get(lane) {
                Some(LaneState::Ready { .. }) => {}
                Some(_) => return Err(HipError::new(0, &format!("sample: lane {lane} not Ready"))),
                None => {
                    return Err(HipError::new(
                        0,
                        &format!("sample: lane {lane} out of range"),
                    ))
                }
            }
        }
        if config.dim != self.dim {
            return Err(HipError::new(0, "sample: config mismatch"));
        }
        gpus.devices[0].bind_thread()?;
        self.ranks[0].sample(
            &mut gpus.devices[0],
            config,
            batch_size,
            temperature,
            top_p,
            top_k,
            rng_state,
        )
    }
    pub fn sample_product(
        &self,
        gpus: &mut Gpus,
        config: &Qwen35Config,
        batch_size: usize,
        repeat_tokens: &[u32],
        repeat_lengths: &[u32],
        rng_states: &[u32],
        temperature: f32,
        top_p: f32,
        top_k: Option<u32>,
        min_p: Option<f32>,
        repeat_penalty: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
    ) -> HipResult<Vec<(u32, u32)>> {
        if batch_size == 0 || batch_size > self.max_batch {
            return Err(HipError::new(0, "sample_product: batch size out of range"));
        }
        for lane in 0..batch_size {
            match self.lane_states.get(lane) {
                Some(LaneState::Ready { .. }) => {}
                Some(_) => {
                    return Err(HipError::new(
                        0,
                        &format!("sample_product: lane {lane} not Ready"),
                    ))
                }
                None => {
                    return Err(HipError::new(
                        0,
                        &format!("sample_product: lane {lane} out of range"),
                    ))
                }
            }
        }
        if config.dim != self.dim {
            return Err(HipError::new(0, "sample_product: config mismatch"));
        }
        gpus.devices[0].bind_thread()?;
        let gpu0 = &mut gpus.devices[0];
        self.ranks[0].sample_product(
            gpu0,
            config,
            batch_size,
            repeat_tokens,
            repeat_lengths,
            rng_states,
            temperature,
            top_p,
            top_k,
            min_p,
            repeat_penalty,
            presence_penalty,
            frequency_penalty,
        )
    }
    pub fn sample_lane(
        &self,
        gpus: &mut Gpus,
        config: &Qwen35Config,
        lane: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<u32>,
        rng_state: u32,
    ) -> HipResult<(u32, u32)> {
        if lane >= self.max_batch {
            return Err(HipError::new(0, "sample_lane: lane out of range"));
        }
        // Lane bit bound check before poison inspection per spec (use lane_bit).
        let _ = lane_bit(lane, self.max_batch)?;
        if config.dim != self.dim {
            return Err(HipError::new(0, "sample_lane: config mismatch"));
        }
        match self.lane_states.get(lane) {
            Some(LaneState::Ready { .. }) => {}
            Some(_) => return Err(HipError::new(0, "sample_lane: lane not Ready")),
            None => return Err(HipError::new(0, "sample_lane: lane out of range")),
        }
        gpus.devices[0].bind_thread()?;
        self.ranks[0].sample_lane(
            &mut gpus.devices[0],
            config,
            lane,
            temperature,
            top_p,
            top_k,
            rng_state,
        )
    }
    pub fn free_gpu(self, gpus: &mut Gpus) -> HipResult<()> {
        let n = self.ranks.len();
        if n != gpus.devices.len() {
            return Err(HipError::new(0, "free_gpu: rank count mismatch"));
        }
        // First bind and synchronize all ranks before any free, as required.
        for rank in 0..n {
            gpus.devices[rank].bind_thread()?;
            gpus.devices[rank].hip.device_synchronize()?;
        }
        let mut first_err: Option<HipError> = None;
        let mut note = |r: HipResult<()>| {
            if let Err(e) = r {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        };
        for (i, r) in self.ranks.into_iter().enumerate() {
            note(r.free_gpu(&mut gpus.devices[i]));
        }
        for (i, t) in self.decode_partials.into_iter().enumerate() {
            note(gpus.devices[i].free_tensor(t));
        }
        for (i, p) in self.seed_pbs.into_iter().enumerate() {
            note(p.free_gpu(&mut gpus.devices[i]));
        }
        for (i, t) in self.seed_partials.into_iter().enumerate() {
            note(gpus.devices[i].free_tensor(t));
        }
        for (i, s) in self.scratches.into_iter().enumerate() {
            note(s.free_gpu(&mut gpus.devices[i]));
        }
        // Release lease last, checked and owner-bound.
        if let Some(lease) = self.peer_lease {
            note(gpus.release_peer_reduce_scratch(&lease));
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// EP (Ship 6 substrate-EP) replicated N-rank decode forward for ONE token.
///
/// Every rank holds **full replicated** weights / scratch / KV / DeltaNet
/// state EXCEPT the MoE routed experts, which were sharded per rank at load by
/// [`shard_moe_experts`]. Behaviorally this mirrors the single-GPU
/// [`forward_scratch`] → [`forward_scratch_layers_lowered`] pipeline (embed →
/// per-layer `LayerProgram` → final norm + lm_head), but runs each layer's
/// program through the EP executor ([`hipfire_runtime::ep::run_layer_program_ep`]):
/// the `Moe` super-op is all-reduce-EP'd across ranks (each rank computes only
/// its owned experts into a zeroed routed partial, the partials are
/// all-reduce-summed, then added into each rank's residual); every other
/// super-op runs **replicated** and stays bit-identical across ranks.
///
/// Logits land in `scratch_per_rank[0].logits` (rank 0 = `output_device`); the
/// caller reads them with `gpu.download_f32` after this returns (this fn
/// device-synchronizes every rank before returning, so the read is safe even
/// though work ran on each rank's `active_stream`).
///
/// All parallel slices (`weights_per_rank`, `kv_per_rank`, `dn_per_rank`,
/// `scratch_per_rank`, `partials`) must have length `gpus.devices.len()`, with
/// element `r` allocated on `gpus.devices[r]`. Every device must have an
/// `active_stream` set ([`hipfire_runtime::ep::ensure_rank_streams`]).
///
/// TP=1 is the degenerate reference: one rank owns all experts (no zero-dummy),
/// the all-reduce short-circuits to identity, and the result is the same as the
/// single-GPU lowered decode (validated byte-/argmax-identical on the fleet).
#[allow(clippy::too_many_arguments)]
pub fn forward_ep(
    gpus: &mut Gpus,
    weights_per_rank: &[Qwen35Weights],
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_per_rank: &mut [llama::KvCache],
    dn_per_rank: &[DeltaNetState],
    scratch_per_rank: &[Qwen35Scratch],
    partials: &[GpuTensor],
) -> HipResult<()> {
    let n = gpus.devices.len();
    assert_eq!(
        weights_per_rank.len(),
        n,
        "forward_ep: weights_per_rank.len() != n_ranks"
    );
    assert_eq!(
        kv_per_rank.len(),
        n,
        "forward_ep: kv_per_rank.len() != n_ranks"
    );
    assert_eq!(
        dn_per_rank.len(),
        n,
        "forward_ep: dn_per_rank.len() != n_ranks"
    );
    assert_eq!(
        scratch_per_rank.len(),
        n,
        "forward_ep: scratch_per_rank.len() != n_ranks"
    );
    assert_eq!(partials.len(), n, "forward_ep: partials.len() != n_ranks");

    let dim = config.dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;
    let pos_i32 = pos as i32;

    // 1. Embed token + write pos on each rank (replicated; deterministic, since
    //    weights are byte-identical replicas → s.x is bit-identical per rank).
    for r in 0..n {
        gpus.devices[r].bind_thread()?;
        let w = &weights_per_rank[r];
        let s = &scratch_per_rank[r];
        let gpu = &mut gpus.devices[r];
        match w.embd_format {
            EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256(&w.token_embd, &s.x, token, dim)?
            }
            EmbeddingFormat::HFQ4G128 => {
                gpu.embedding_lookup_hfq4g128(&w.token_embd, &s.x, token, dim)?
            }
            EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&w.token_embd, &s.x, token, dim)?,
            EmbeddingFormat::F32 => gpu.embedding_lookup(&w.token_embd, &s.x, token, dim)?,
            other => {
                return Err(HipError::new(
                    0,
                    &format!("forward_ep: unsupported embedding format {other:?}"),
                ));
            }
        }
        gpu.hip.memcpy_htod(&s.pos_buf, &pos_i32.to_ne_bytes())?;
    }

    // 2. Per-layer EP program. Variant + delta-layer counter are replicated
    //    (sharding frees experts but never changes the layer variant), so rank 0
    //    is authoritative for both.
    let mut delta_layer_idx = 0usize;
    for layer_idx in 0..config.n_layers {
        let program = lower_variant(variant_of(&weights_per_rank[0].layers[layer_idx]));
        // Build the N per-rank bindings. `kv_per_rank.iter_mut()` yields the
        // disjoint `&mut KvCache` each binding needs; weights/scratch/dn are
        // shared `&`. This Vec is dropped at the end of the iteration, releasing
        // the mutable KV borrows before the next layer's `iter_mut`.
        let mut binds: Vec<Qwen35Bindings> = Vec::with_capacity(n);
        for (((w, s), kv), dn) in weights_per_rank
            .iter()
            .zip(scratch_per_rank.iter())
            .zip(kv_per_rank.iter_mut())
            .zip(dn_per_rank.iter())
        {
            binds.push(Qwen35Bindings {
                layer: &w.layers[layer_idx],
                s,
                config,
                kv_cache: kv,
                dn_state: dn,
                pos,
                layer_idx,
                delta_layer_idx,
                k_dim,
                v_dim,
                n_v_heads,
                hd,
                precomputed_attn_x_rot: false,
                fa_output_prerotated: false,
                defer_routed_combine: false,
            });
        }
        hipfire_runtime::ep::run_layer_program_ep(
            gpus,
            binds.as_mut_slice(),
            partials,
            &program,
            dim,
        )
        .map_err(|e| HipError::new(0, &e.to_string()))?;
        if matches!(
            &weights_per_rank[0].layers[layer_idx],
            LayerWeights::DeltaNet(_) | LayerWeights::DeltaNetMoe(_)
        ) {
            delta_layer_idx += 1;
        }
    }

    // 3. Final norm + lm_head on rank 0 (output_device). Logits → rank0 scratch.
    {
        gpus.devices[0].bind_thread()?;
        let w = &weights_per_rank[0];
        let s = &scratch_per_rank[0];
        let gpu = &mut gpus.devices[0];
        gpu.rmsnorm_f32(&s.x, &w.output_norm, &s.tmp, config.norm_eps)?;
        let ctx = DispatchCtx::new(gpu);
        let wr = w.output.dispatch_ref();
        let step = Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(&s.tmp),
            out: &s.logits,
        };
        execute_steps(gpu, &ctx, &[step]).map_err(|e| HipError::new(0, &e.to_string()))?;
    }

    // 4. Sync every rank — work ran on each device's active_stream, so a host
    //    download of rank 0's logits (on the null stream) would otherwise race.
    for r in 0..n {
        gpus.devices[r].bind_thread()?;
        gpus.devices[r].hip.device_synchronize()?;
    }
    Ok(())
}

/// EP (Ship 6 substrate-EP) **WMMA batched prefill** for qwen3.x-A3B (E6b).
///
/// The batched analog of [`forward_ep`]: processes all `tokens` as one batch
/// through the WMMA/grouped-GEMM prefill kernels (NOT token-by-token), replicated
/// across `gpus.devices.len()` EP ranks, with MoE experts sharded per rank.
///
/// Driven **layer-granularly** by calling [`forward_prefill_chunk`] with a
/// single-layer band per rank, because EP needs a per-MoE-layer all-reduce: the
/// next layer's replicated attention must read the FULL (cross-rank-summed)
/// residual. For each layer:
///   1. (MoE only) zero each rank's `[n × dim]` routed partial,
///   2. run the layer's batched chunk on every rank — the **shared** expert
///      accumulates into `pbs.x_batch` (replicated, added once per rank), the
///      **routed** combine into the zeroed partial (owned experts only; non-owned
///      read load-time zero-dummy → 0),
///   3. (MoE only) `all_reduce_sum_f32` the `[n × dim]` partials across ranks and
///      add into each rank's `pbs.x_batch`.
/// Non-MoE (dense DeltaNet / FullAttn) layers run replicated, no partial, no
/// all-reduce. Final norm + lm_head (last token) run on rank 0 → `scratch_per_rank[0].logits`.
///
/// **v1 constraints:** the whole prompt must fit one batch (`tokens.len() <=
/// pbs.max_batch`; no chunk loop yet) and KV must be a non-asym mode (q8/q4/…)
/// so no per-rank Givens replicas are needed (asym EP prefill = future work). The
/// per-layer chunk dispatch trades some launch overhead for the per-layer
/// all-reduce seam; a fused EP prefill layer loop is a later perf refinement.
///
/// Slices (`weights_per_rank`, `kv_per_rank`, `dn_per_rank`, `scratch_per_rank`,
/// `pbs_per_rank`, `partials`) must have length `gpus.devices.len()`; element `r`
/// lives on `gpus.devices[r]`. Each `partials[r]` must hold >= `n × dim` f32.
/// Every device must have an `active_stream` ([`hipfire_runtime::ep::ensure_rank_streams`]).
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_ep(
    gpus: &mut Gpus,
    weights_per_rank: &[Qwen35Weights],
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_per_rank: &mut [llama::KvCache],
    dn_per_rank: &mut [DeltaNetState],
    scratch_per_rank: &[Qwen35Scratch],
    pbs_per_rank: &[PrefillBatchScratch],
    partials: &[GpuTensor],
) -> HipResult<()> {
    let n_rank = gpus.devices.len();
    assert_eq!(
        weights_per_rank.len(),
        n_rank,
        "forward_prefill_batch_ep: weights_per_rank len"
    );
    assert_eq!(
        kv_per_rank.len(),
        n_rank,
        "forward_prefill_batch_ep: kv_per_rank len"
    );
    assert_eq!(
        dn_per_rank.len(),
        n_rank,
        "forward_prefill_batch_ep: dn_per_rank len"
    );
    assert_eq!(
        scratch_per_rank.len(),
        n_rank,
        "forward_prefill_batch_ep: scratch_per_rank len"
    );
    assert_eq!(
        pbs_per_rank.len(),
        n_rank,
        "forward_prefill_batch_ep: pbs_per_rank len"
    );
    assert_eq!(
        partials.len(),
        n_rank,
        "forward_prefill_batch_ep: partials len"
    );

    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }
    let dim = config.dim;
    // Per-call contract: one window must fit max_batch. Long prompts are driven
    // by calling this repeatedly with advancing start_pos + persistent kv/dn
    // (KV + DeltaNet state accumulate in place across calls, identical to the
    // single-GPU chunk loop in forward_prefill_batch) — see ep_decode_parity's
    // chunked-prefill loop. That bounds the prefill scratch to chunk_size, so
    // context length is limited by the KV cache, not the activation buffers.
    assert!(
        n <= pbs_per_rank[0].max_batch,
        "forward_prefill_batch_ep: window ({n} toks) must fit one batch (max_batch={}); \
         caller chunks long prompts into max_batch-sized windows",
        pbs_per_rank[0].max_batch,
    );

    // Per-layer cumulative LA / FA counters (replicated → identical across ranks;
    // they index dn_state.s_matrices / kv_cache.k_gpu exactly like the band
    // offsets the PP driver threads). kv_layer_offset == fa_layer_offset.
    let mut delta_off = 0usize;
    let mut fa_off = 0usize;

    let ep_timing = hipfire_config::developer_var("HIPFIRE_EP_PREFILL_TIMING").is_ok();
    let ep_skip_ar = hipfire_config::developer_var("HIPFIRE_EP_SKIP_ALLREDUCE").is_ok(); // DIAGNOSTIC ONLY (wrong output)
                                                                                         // Peer-direct all-reduce (bypass RCCL): the routed-partial sum goes through
                                                                                         // Gpus::all_reduce_sum_f32_peer (direct P2P copy + local add), which is ~1 ms
                                                                                         // vs RCCL's ~40 ms/call on hiptrx (gfx1201, PCIe). DEFAULT ON; opt back to
                                                                                         // RCCL with HIPFIRE_EP_PEER_ALLREDUCE=0. The peer temps live in Gpus (shared
                                                                                         // with TP), lazily sized to the largest count seen.
    let ep_peer_ar =
        hipfire_config::developer_var("HIPFIRE_EP_PEER_ALLREDUCE").as_deref() != Ok("0");
    let mut t_chunk = 0.0f64;
    let mut t_ar = 0.0f64;
    let mut t_add = 0.0f64;
    for layer_idx in 0..config.n_layers {
        let is_moe = matches!(
            &weights_per_rank[0].layers[layer_idx],
            LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_)
        );

        // 1. Zero each rank's routed partial (on its active_stream, so it's
        //    ordered before the chunk's routed combine that writes into it).
        if is_moe {
            for r in 0..n_rank {
                gpus.devices[r].bind_thread()?;
                let stream = gpus.devices[r].active_stream.as_ref().ok_or_else(|| {
                    HipError::new(
                        0,
                        "forward_prefill_batch_ep: no active_stream (call ensure_rank_streams)",
                    )
                })?;
                gpus.devices[r]
                    .hip
                    .memset_async(&partials[r].buf, 0, n * dim * 4, stream)?;
            }
        }

        // 2. Run the layer's batched chunk on every rank (single-layer band).
        let t_c = std::time::Instant::now();
        for r in 0..n_rank {
            gpus.devices[r].bind_thread()?;
            let band = PrefillBandCtx {
                layer_start: layer_idx,
                layer_end: layer_idx + 1,
                delta_layer_offset: delta_off,
                kv_layer_offset: fa_off,
                is_first_band: layer_idx == 0,
                is_last_band: false, // final norm + lm_head done explicitly below
                // v1 EP prefill is q8/non-asym KV → no per-rank Givens replicas.
                givens_cos: None,
                givens_sin: None,
            };
            let routed_out = if is_moe { Some(&partials[r]) } else { None };
            forward_prefill_chunk(
                &mut gpus.devices[r],
                &weights_per_rank[r],
                config,
                tokens,
                start_pos,
                &mut kv_per_rank[r],
                &mut dn_per_rank[r],
                &scratch_per_rank[r],
                &pbs_per_rank[r],
                None,  // hidden_rb
                None,  // per_token_hidden_out
                None,  // gdn_tape
                0,     // tape_offset
                None,  // tree_verify
                false, // pre_uploaded
                Some(&band),
                None,  // mask_override
                false, // needs_last_token_logits (no lm_head in band)
                None,  // max_layer
                routed_out,
            )?;
        }

        if ep_timing {
            t_chunk += t_c.elapsed().as_secs_f64() * 1000.0;
        }

        // 3. All-reduce the routed partials, add into each rank's residual.
        if is_moe && !ep_skip_ar {
            let t_a = std::time::Instant::now();
            let refs: Vec<&hip_bridge::DeviceBuffer> = partials.iter().map(|p| &p.buf).collect();
            if ep_peer_ar {
                gpus.all_reduce_sum_f32_peer(&refs, n * dim)
                    .map_err(|e| HipError::new(0, &e.to_string()))?;
            } else {
                gpus.all_reduce_sum_f32(&refs, n * dim)
                    .map_err(|e| HipError::new(0, &e.to_string()))?;
            }
            if ep_timing {
                t_ar += t_a.elapsed().as_secs_f64() * 1000.0;
            }
            let t_d = std::time::Instant::now();
            for r in 0..n_rank {
                gpus.devices[r].bind_thread()?;
                let x_n = pbs_per_rank[r].x_batch.sub_offset(0, n * dim);
                let p_n = partials[r].sub_offset(0, n * dim);
                gpus.devices[r].add_inplace_f32(&x_n, &p_n)?;
            }
            if ep_timing {
                t_add += t_d.elapsed().as_secs_f64() * 1000.0;
            }
        }

        match config.layer_types[layer_idx] {
            LayerType::LinearAttention => delta_off += 1,
            LayerType::FullAttention => fa_off += 1,
        }
    }

    // Final norm + lm_head on rank 0 (last token) → scratch_per_rank[0].logits.
    // Done explicitly (not via the chunk) so it runs AFTER the last layer's
    // all-reduce — the last MoE layer's routed output is only in x_batch after
    // step 3, so an in-chunk lm_head would read an incomplete residual.
    {
        gpus.devices[0].bind_thread()?;
        let gpu = &mut gpus.devices[0];
        let w = &weights_per_rank[0];
        let s = &scratch_per_rank[0];
        let pbs = &pbs_per_rank[0];
        let last_x = pbs.x_batch.sub_offset((n - 1) * dim, dim);
        gpu.rmsnorm_f32(&last_x, &w.output_norm, &s.tmp, config.norm_eps)?;
        let ctx = DispatchCtx::new(gpu);
        let wr = w.output.dispatch_ref();
        let step = Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(&s.tmp),
            out: &s.logits,
        };
        execute_steps(gpu, &ctx, &[step]).map_err(|e| HipError::new(0, &e.to_string()))?;
    }

    // Sync every rank — work ran on active_streams; the host logits read on rank
    // 0 (null stream) would otherwise race.
    let t_s = std::time::Instant::now();
    for r in 0..n_rank {
        gpus.devices[r].bind_thread()?;
        gpus.devices[r].hip.device_synchronize()?;
    }
    if ep_timing {
        let t_sync = t_s.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "EP-PREFILL-TIMING (host ms): chunk-loop={t_chunk:.1} all_reduce={t_ar:.1} add={t_add:.1} final-sync={t_sync:.1}",
        );
    }
    Ok(())
}

/// Multi-GPU layer-loop dispatcher (Stage 5 of multi-GPU pp migration #58).
/// Mirrors `forward_scratch_layers` but routes per-layer work to
/// `gpus.devices[gpus.device_for_layer(i)]` and copies the residual
/// stream `s.x` across band boundaries via `Gpus::boundary_copy`.
/// Final `output_norm + lm_head` runs on `gpus.output_device`
/// (Variant 2 — no copy back to dev_0). Spec-decode `hidden_rb` is
/// not threaded — refused at load time when pp > 1.
fn forward_scratch_layers_multi(
    gpus: &mut Gpus,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch_set: &Qwen35ScratchSet,
) -> HipResult<()> {
    let dim = config.dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let qkv_dim = k_dim * 2 + v_dim;
    let _ = qkv_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;

    let mut delta_layer_idx = 0usize;
    let mut prev_dev: Option<usize> = None;

    for layer_idx in 0..config.n_layers {
        let dev_idx = gpus.device_for_layer(layer_idx);

        if let Some(pd) = prev_dev {
            if dev_idx != pd {
                let src_buf = &scratch_set.per_device[pd].x.buf;
                let dst_buf = &scratch_set.per_device[dev_idx].x.buf;
                let evt = gpus.boundary_copy(pd, dev_idx, src_buf, dst_buf, dim * 4)?;
                gpus.wait_boundary(evt)?;
            }
        }

        {
            let s = &scratch_set.per_device[dev_idx];
            let givens_cos_dev = gpus.givens_cos_per_dev.get(dev_idx);
            let givens_sin_dev = gpus.givens_sin_per_dev.get(dev_idx);
            let gpu = &mut gpus.devices[dev_idx];

            // Resolve givens lazily — asym{2,3,4} branches use these,
            // others don't. Multi-GPU prefers the per-device replica
            // populated by the KV ctor; fall back to kv_cache.givens_*
            // for single-GPU shape compatibility (shouldn't fire in
            // pp > 1 since asym ctors always populate per-device).
            macro_rules! ct {
                () => {
                    givens_cos_dev.unwrap_or_else(|| kv_cache.givens_cos.as_ref().unwrap())
                };
            }
            macro_rules! st {
                () => {
                    givens_sin_dev.unwrap_or_else(|| kv_cache.givens_sin.as_ref().unwrap())
                };
            }

            match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
                (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                    let x_rot = fused_rmsnorm_rotate_for_mq(
                        gpu,
                        &layer.wqkv,
                        &s.x,
                        &layer.attn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )?;
                    let dt = layer.wqkv.gpu_dtype;
                    let la4_same_dtype = layer.wz.gpu_dtype == dt
                        && layer.w_beta.gpu_dtype == dt
                        && layer.w_alpha.gpu_dtype == dt;
                    let fused_la4_mq4 =
                        la4_same_dtype && (dt == DType::MQ4G256 || dt == DType::HFQ4G256);
                    let fused_la4_lloyd_mq3 = la4_same_dtype && dt == DType::MQ3G256Lloyd;
                    let fused_la4_lloyd_mq4 = la4_same_dtype && dt == DType::MQ4G256Lloyd;
                    if fused_la4_mq4 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_qkvza_hfq4g256(
                            &layer.wqkv.buf,
                            &layer.wz.buf,
                            &layer.w_beta.buf,
                            &layer.w_alpha.buf,
                            eff_x,
                            &s.dn_qkv,
                            &s.dn_z,
                            &s.dn_beta,
                            &s.dn_alpha,
                            layer.wqkv.m,
                            layer.wz.m,
                            layer.w_beta.m,
                            layer.w_alpha.m,
                            layer.wqkv.k,
                        )?;
                    } else if fused_la4_lloyd_mq3 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_qkvza_mq3g256_lloyd(
                            &layer.wqkv.buf,
                            &layer.wz.buf,
                            &layer.w_beta.buf,
                            &layer.w_alpha.buf,
                            eff_x,
                            &s.dn_qkv,
                            &s.dn_z,
                            &s.dn_beta,
                            &s.dn_alpha,
                            layer.wqkv.m,
                            layer.wz.m,
                            layer.w_beta.m,
                            layer.w_alpha.m,
                            layer.wqkv.k,
                        )?;
                    } else {
                        weight_gemv_prerotated(gpu, &layer.wqkv, &s.tmp, x_rot, &s.dn_qkv)?;
                        weight_gemv_prerotated(gpu, &layer.wz, &s.tmp, x_rot, &s.dn_z)?;
                        weight_gemv_prerotated(gpu, &layer.w_beta, &s.tmp, x_rot, &s.dn_beta)?;
                        weight_gemv_prerotated(gpu, &layer.w_alpha, &s.tmp, x_rot, &s.dn_alpha)?;
                    }
                    gpu.fused_sigmoid_alpha_gate_f32(
                        &s.dn_beta,
                        &s.dn_alpha,
                        &layer.dt_bias,
                        &layer.a_log,
                        n_v_heads,
                    )?;
                    gpu.conv1d_silu_split_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        &s.dn_v,
                        &s.dn_qkv,
                        &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        k_dim,
                        v_dim,
                    )?;
                    gpu.fused_qk_l2_norm_scale_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        config.linear_num_key_heads,
                        hd,
                        1.0 / (hd as f32).sqrt(),
                        config.norm_eps,
                    )?;
                    if config.linear_num_key_heads < n_v_heads {
                        let ratio = n_v_heads / config.linear_num_key_heads;
                        gpu.repeat_interleave_qk_f32(
                            &s.dn_q_raw,
                            &s.dn_k_raw,
                            &s.dn_q,
                            &s.dn_k,
                            config.linear_num_key_heads,
                            ratio,
                            hd,
                        )?;
                    } else {
                        gpu.memcpy_dtod_auto(&s.dn_q.buf, &s.dn_q_raw.buf, k_dim * 4)?;
                        gpu.memcpy_dtod_auto(&s.dn_k.buf, &s.dn_k_raw.buf, k_dim * 4)?;
                    }
                    match dn_state.quant {
                        StateQuant::FP32 => gpu.gated_delta_net_f32(
                            &s.dn_q,
                            &s.dn_k,
                            &s.dn_v,
                            &s.dn_alpha,
                            &s.dn_beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &s.dn_attn_out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                        StateQuant::Q8 => gpu.gated_delta_net_q8(
                            &s.dn_q,
                            &s.dn_k,
                            &s.dn_v,
                            &s.dn_alpha,
                            &s.dn_beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &s.dn_attn_out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                            dn_state.ef_residual(delta_layer_idx),
                        )?,
                        StateQuant::Q4 => gpu.gated_delta_net_q4(
                            &s.dn_q,
                            &s.dn_k,
                            &s.dn_v,
                            &s.dn_alpha,
                            &s.dn_beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &s.dn_attn_out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                    }
                    gpu.gated_norm_f32(
                        &s.dn_attn_out,
                        &s.dn_z,
                        &layer.norm_weight,
                        &s.dn_normed,
                        n_v_heads,
                        config.linear_value_head_dim,
                        config.norm_eps,
                    )?;
                    {
                        let ctx = DispatchCtx::new(gpu);
                        let wr = layer.wo.dispatch_ref();
                        execute_steps(
                            gpu,
                            &ctx,
                            &[Step::GemvResidual {
                                w: &wr,
                                input: GemvInput::Raw(&s.dn_normed),
                                residual: &s.x,
                                out: &s.x,
                            }],
                        )
                        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                    }

                    let x_rot = fused_rmsnorm_rotate_for_mq(
                        gpu,
                        &layer.w_gate,
                        &s.x,
                        &layer.ffn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )?;
                    let dt_g = layer.w_gate.gpu_dtype;
                    let same_dtype = layer.w_up.gpu_dtype == dt_g;
                    let fused_gu_mq4 =
                        same_dtype && (dt_g == DType::MQ4G256 || dt_g == DType::HFQ4G256);
                    let fused_gu_lloyd_mq3 = same_dtype && dt_g == DType::MQ3G256Lloyd;
                    let fused_gu_lloyd_mq4 = same_dtype && dt_g == DType::MQ4G256Lloyd;
                    if fused_gu_mq4 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_gate_up_hfq4g256(
                            &layer.w_gate.buf,
                            &layer.w_up.buf,
                            eff_x,
                            &s.gate_ffn,
                            &s.up,
                            layer.w_gate.m,
                            layer.w_up.m,
                            layer.w_gate.k,
                        )?;
                    } else if fused_gu_lloyd_mq3 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_gate_up_mq3g256_lloyd(
                            &layer.w_gate.buf,
                            &layer.w_up.buf,
                            eff_x,
                            &s.gate_ffn,
                            &s.up,
                            layer.w_gate.m,
                            layer.w_up.m,
                            layer.w_gate.k,
                        )?;
                    } else {
                        weight_gemv_prerotated(gpu, &layer.w_gate, &s.tmp, x_rot, &s.gate_ffn)?;

                        weight_gemv_prerotated(gpu, &layer.w_up, &s.tmp, x_rot, &s.up)?;
                    }
                    weight_gemv_swiglu_residual(
                        gpu,
                        &layer.w_down,
                        &s.gate_ffn,
                        &s.up,
                        &s.ffn_hidden,
                        &s.x,
                    )?;
                    delta_layer_idx += 1;
                }

                (LayerWeights::FullAttn(layer), LayerType::FullAttention) => {
                    let x_rot = fused_rmsnorm_rotate_for_mq(
                        gpu,
                        &layer.wq,
                        &s.x,
                        &layer.attn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )?;
                    let dt = layer.wq.gpu_dtype;
                    let fa3_same_dtype = layer.wk.gpu_dtype == dt && layer.wv.gpu_dtype == dt;
                    let fused_fa3_mq4 =
                        fa3_same_dtype && (dt == DType::MQ4G256 || dt == DType::HFQ4G256);
                    let fused_fa3_lloyd_mq3 = fa3_same_dtype && dt == DType::MQ3G256Lloyd;
                    let fused_fa3_lloyd_mq4 = fa3_same_dtype && dt == DType::MQ4G256Lloyd;
                    if fused_fa3_mq4 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_qkv_hfq4g256(
                            &layer.wq.buf,
                            &layer.wk.buf,
                            &layer.wv.buf,
                            eff_x,
                            &s.fa_q_full,
                            &s.fa_k,
                            &s.fa_v,
                            layer.wq.m,
                            layer.wk.m,
                            layer.wv.m,
                            layer.wq.k,
                        )?;
                    } else if fused_fa3_lloyd_mq3 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_qkv_mq3g256_lloyd(
                            &layer.wq.buf,
                            &layer.wk.buf,
                            &layer.wv.buf,
                            eff_x,
                            &s.fa_q_full,
                            &s.fa_k,
                            &s.fa_v,
                            layer.wq.m,
                            layer.wk.m,
                            layer.wv.m,
                            layer.wq.k,
                        )?;
                    } else {
                        weight_gemv_prerotated(gpu, &layer.wq, &s.tmp, x_rot, &s.fa_q_full)?;

                        weight_gemv_prerotated(gpu, &layer.wk, &s.tmp, x_rot, &s.fa_k)?;
                        weight_gemv_prerotated(gpu, &layer.wv, &s.tmp, x_rot, &s.fa_v)?;
                    }
                    gpu.deinterleave_f32(
                        &s.fa_q_full,
                        &s.fa_q,
                        &s.fa_gate,
                        config.n_heads,
                        config.head_dim,
                    )?;
                    gpu.rmsnorm_batched(
                        &s.fa_q,
                        &layer.q_norm,
                        &s.fa_q,
                        config.n_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;
                    let kv_dim = config.n_kv_heads * config.head_dim;
                    gpu.rmsnorm_batched(
                        &s.fa_k,
                        &layer.k_norm,
                        &s.fa_k,
                        config.n_kv_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;

                    if kv_cache.compact_offset > 0 {
                        let abs = (pos + kv_cache.compact_offset) as i32;
                        gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
                    }
                    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                    gpu.rope_partial_interleaved_f32(
                        &s.fa_q,
                        &s.fa_k,
                        &s.pos_buf,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        n_rot,
                        config.rope_theta,
                    )?;
                    if kv_cache.compact_offset > 0 {
                        let phys = pos as i32;
                        gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
                    }

                    if kv_cache.quant_asym4 {
                        let ct = ct!();
                        let st = st!();
                        if kv_cache.quant_fwht {
                            gpu.kv_cache_write_fwht4_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.v_mode_bits(),
                            )?;
                            gpu.attention_flash_fwht4(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                                kv_cache.v_mode_bits(),
                            )?;
                        } else {
                            gpu.kv_cache_write_asym4_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                            )?;
                            gpu.attention_flash_asym4(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                            )?;
                        }
                    } else if kv_cache.quant_asym3 {
                        let ct = ct!();
                        let st = st!();
                        if kv_cache.quant_fwht {
                            gpu.kv_cache_write_fwht3_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.v_mode_bits(),
                            )?;
                            gpu.attention_flash_fwht3(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                                kv_cache.v_mode_bits(),
                            )?;
                        } else {
                            gpu.kv_cache_write_asym3_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                            )?;
                            gpu.attention_flash_asym3(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                                None,
                            )?;
                        }
                    } else if kv_cache.quant_asym2 {
                        let ct = ct!();
                        let st = st!();
                        if kv_cache.quant_fwht {
                            gpu.kv_cache_write_fwht2_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.v_mode_bits(),
                            )?;
                            gpu.attention_flash_fwht2(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                                kv_cache.v_mode_bits(),
                            )?;
                        } else {
                            gpu.kv_cache_write_asym2_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                            )?;
                            gpu.attention_flash_asym2(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                            )?;
                        }
                    } else if kv_cache.quant_q8 {
                        gpu.kv_cache_write_q8_0(
                            &kv_cache.k_gpu[layer_idx],
                            &s.fa_k,
                            &s.pos_buf,
                            config.n_kv_heads,
                            config.head_dim,
                        )?;
                        gpu.kv_cache_write_q8_0(
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_v,
                            &s.pos_buf,
                            config.n_kv_heads,
                            config.head_dim,
                        )?;
                        let use_flash = gpu.graphs.capture_mode
                            || s.flash_mode == 2
                            || (s.flash_mode == 1 && pos + 1 >= 2048)
                            || pos + 1 > 15000;
                        if use_flash {
                            gpu.attention_flash_q8_0(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                            )?;
                        } else {
                            gpu.attention_q8_0_kv(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                            )?;
                        }
                    } else {
                        gpu.kv_cache_write(
                            &kv_cache.k_gpu[layer_idx],
                            &s.fa_k,
                            &s.pos_buf,
                            kv_dim,
                        )?;
                        gpu.kv_cache_write(
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_v,
                            &s.pos_buf,
                            kv_dim,
                        )?;
                        gpu.attention_f32(
                            &s.fa_q,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out,
                            &s.pos_buf,
                            pos + 1,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                        )?;
                    }

                    gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
                    {
                        let ctx = DispatchCtx::new(gpu);
                        let wr = layer.wo.dispatch_ref();
                        execute_steps(
                            gpu,
                            &ctx,
                            &[Step::GemvResidual {
                                w: &wr,
                                input: GemvInput::Raw(&s.fa_attn_out),
                                residual: &s.x,
                                out: &s.x,
                            }],
                        )
                        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                    }

                    let x_rot = fused_rmsnorm_rotate_for_mq(
                        gpu,
                        &layer.w_gate,
                        &s.x,
                        &layer.ffn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )?;
                    let dt_g = layer.w_gate.gpu_dtype;
                    let same_dtype = layer.w_up.gpu_dtype == dt_g;
                    let fused_gu_mq4 =
                        same_dtype && (dt_g == DType::MQ4G256 || dt_g == DType::HFQ4G256);
                    let fused_gu_lloyd_mq3 = same_dtype && dt_g == DType::MQ3G256Lloyd;
                    let fused_gu_lloyd_mq4 = same_dtype && dt_g == DType::MQ4G256Lloyd;
                    if fused_gu_mq4 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_gate_up_hfq4g256(
                            &layer.w_gate.buf,
                            &layer.w_up.buf,
                            eff_x,
                            &s.gate_ffn,
                            &s.up,
                            layer.w_gate.m,
                            layer.w_up.m,
                            layer.w_gate.k,
                        )?;
                    } else if fused_gu_lloyd_mq3 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_gate_up_mq3g256_lloyd(
                            &layer.w_gate.buf,
                            &layer.w_up.buf,
                            eff_x,
                            &s.gate_ffn,
                            &s.up,
                            layer.w_gate.m,
                            layer.w_up.m,
                            layer.w_gate.k,
                        )?;
                    } else {
                        weight_gemv_prerotated(gpu, &layer.w_gate, &s.tmp, x_rot, &s.gate_ffn)?;

                        weight_gemv_prerotated(gpu, &layer.w_up, &s.tmp, x_rot, &s.up)?;
                    }
                    weight_gemv_swiglu_residual(
                        gpu,
                        &layer.w_down,
                        &s.gate_ffn,
                        &s.up,
                        &s.ffn_hidden,
                        &s.x,
                    )?;
                }

                (LayerWeights::DeltaNetMoe(layer), LayerType::LinearAttention) => {
                    let x_rot = fused_rmsnorm_rotate_for_mq(
                        gpu,
                        &layer.wqkv,
                        &s.x,
                        &layer.attn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )?;
                    let dt = layer.wqkv.gpu_dtype;
                    let la4_same_dtype = layer.wz.gpu_dtype == dt
                        && layer.w_beta.gpu_dtype == dt
                        && layer.w_alpha.gpu_dtype == dt;
                    let fused_la4_mq4 =
                        la4_same_dtype && (dt == DType::MQ4G256 || dt == DType::HFQ4G256);
                    let fused_la4_lloyd_mq3 = la4_same_dtype && dt == DType::MQ3G256Lloyd;
                    let fused_la4_lloyd_mq4 = la4_same_dtype && dt == DType::MQ4G256Lloyd;
                    if fused_la4_mq4 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_qkvza_hfq4g256(
                            &layer.wqkv.buf,
                            &layer.wz.buf,
                            &layer.w_beta.buf,
                            &layer.w_alpha.buf,
                            eff_x,
                            &s.dn_qkv,
                            &s.dn_z,
                            &s.dn_beta,
                            &s.dn_alpha,
                            layer.wqkv.m,
                            layer.wz.m,
                            layer.w_beta.m,
                            layer.w_alpha.m,
                            layer.wqkv.k,
                        )?;
                    } else if fused_la4_lloyd_mq3 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_qkvza_mq3g256_lloyd(
                            &layer.wqkv.buf,
                            &layer.wz.buf,
                            &layer.w_beta.buf,
                            &layer.w_alpha.buf,
                            eff_x,
                            &s.dn_qkv,
                            &s.dn_z,
                            &s.dn_beta,
                            &s.dn_alpha,
                            layer.wqkv.m,
                            layer.wz.m,
                            layer.w_beta.m,
                            layer.w_alpha.m,
                            layer.wqkv.k,
                        )?;
                    } else {
                        weight_gemv_prerotated(gpu, &layer.wqkv, &s.tmp, x_rot, &s.dn_qkv)?;
                        weight_gemv_prerotated(gpu, &layer.wz, &s.tmp, x_rot, &s.dn_z)?;
                        weight_gemv_prerotated(gpu, &layer.w_beta, &s.tmp, x_rot, &s.dn_beta)?;
                        weight_gemv_prerotated(gpu, &layer.w_alpha, &s.tmp, x_rot, &s.dn_alpha)?;
                    }
                    gpu.fused_sigmoid_alpha_gate_f32(
                        &s.dn_beta,
                        &s.dn_alpha,
                        &layer.dt_bias,
                        &layer.a_log,
                        n_v_heads,
                    )?;
                    gpu.conv1d_silu_split_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        &s.dn_v,
                        &s.dn_qkv,
                        &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        k_dim,
                        v_dim,
                    )?;
                    gpu.fused_qk_l2_norm_scale_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        config.linear_num_key_heads,
                        hd,
                        1.0 / (hd as f32).sqrt(),
                        config.norm_eps,
                    )?;
                    if config.linear_num_key_heads < n_v_heads {
                        let ratio = n_v_heads / config.linear_num_key_heads;
                        gpu.repeat_interleave_qk_f32(
                            &s.dn_q_raw,
                            &s.dn_k_raw,
                            &s.dn_q,
                            &s.dn_k,
                            config.linear_num_key_heads,
                            ratio,
                            hd,
                        )?;
                    } else {
                        gpu.memcpy_dtod_auto(&s.dn_q.buf, &s.dn_q_raw.buf, k_dim * 4)?;
                        gpu.memcpy_dtod_auto(&s.dn_k.buf, &s.dn_k_raw.buf, k_dim * 4)?;
                    }
                    match dn_state.quant {
                        StateQuant::FP32 => gpu.gated_delta_net_f32(
                            &s.dn_q,
                            &s.dn_k,
                            &s.dn_v,
                            &s.dn_alpha,
                            &s.dn_beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &s.dn_attn_out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                        StateQuant::Q8 => gpu.gated_delta_net_q8(
                            &s.dn_q,
                            &s.dn_k,
                            &s.dn_v,
                            &s.dn_alpha,
                            &s.dn_beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &s.dn_attn_out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                            dn_state.ef_residual(delta_layer_idx),
                        )?,
                        StateQuant::Q4 => gpu.gated_delta_net_q4(
                            &s.dn_q,
                            &s.dn_k,
                            &s.dn_v,
                            &s.dn_alpha,
                            &s.dn_beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &s.dn_attn_out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                    }
                    gpu.gated_norm_f32(
                        &s.dn_attn_out,
                        &s.dn_z,
                        &layer.norm_weight,
                        &s.dn_normed,
                        n_v_heads,
                        config.linear_value_head_dim,
                        config.norm_eps,
                    )?;
                    {
                        let ctx = DispatchCtx::new(gpu);
                        let wr = layer.wo.dispatch_ref();
                        execute_steps(
                            gpu,
                            &ctx,
                            &[Step::GemvResidual {
                                w: &wr,
                                input: GemvInput::Raw(&s.dn_normed),
                                residual: &s.x,
                                out: &s.x,
                            }],
                        )
                        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                    }

                    if ffn_all_mq4_for_moe(&layer.ffn) {
                        gpu.fused_rmsnorm_rotate_mq(
                            &s.x,
                            &layer.ffn_norm,
                            s.moe_x_rot.as_ref().expect("MoE scratch"),
                            config.dim,
                            config.norm_eps,
                        )?;
                        moe_ffn_decode_with_scratch_prerotated(
                            gpu, &layer.ffn, &s.x, &s.x, config, s,
                        )?;
                    } else {
                        gpu.rmsnorm_f32(&s.x, &layer.ffn_norm, &s.tmp, config.norm_eps)?;
                        moe_ffn_decode_with_scratch(gpu, &layer.ffn, &s.tmp, &s.x, config, s)?;
                    }
                    delta_layer_idx += 1;
                }

                (LayerWeights::FullAttnMoe(layer), LayerType::FullAttention) => {
                    let x_rot = fused_rmsnorm_rotate_for_mq(
                        gpu,
                        &layer.wq,
                        &s.x,
                        &layer.attn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )?;
                    let dt = layer.wq.gpu_dtype;
                    let fa3_same_dtype = layer.wk.gpu_dtype == dt && layer.wv.gpu_dtype == dt;
                    let fused_fa3_mq4 =
                        fa3_same_dtype && (dt == DType::MQ4G256 || dt == DType::HFQ4G256);
                    let fused_fa3_lloyd_mq3 = fa3_same_dtype && dt == DType::MQ3G256Lloyd;
                    let fused_fa3_lloyd_mq4 = fa3_same_dtype && dt == DType::MQ4G256Lloyd;
                    if fused_fa3_mq4 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_qkv_hfq4g256(
                            &layer.wq.buf,
                            &layer.wk.buf,
                            &layer.wv.buf,
                            eff_x,
                            &s.fa_q_full,
                            &s.fa_k,
                            &s.fa_v,
                            layer.wq.m,
                            layer.wk.m,
                            layer.wv.m,
                            layer.wq.k,
                        )?;
                    } else if fused_fa3_lloyd_mq3 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_qkv_mq3g256_lloyd(
                            &layer.wq.buf,
                            &layer.wk.buf,
                            &layer.wv.buf,
                            eff_x,
                            &s.fa_q_full,
                            &s.fa_k,
                            &s.fa_v,
                            layer.wq.m,
                            layer.wk.m,
                            layer.wv.m,
                            layer.wq.k,
                        )?;
                    } else {
                        weight_gemv_prerotated(gpu, &layer.wq, &s.tmp, x_rot, &s.fa_q_full)?;

                        weight_gemv_prerotated(gpu, &layer.wk, &s.tmp, x_rot, &s.fa_k)?;
                        weight_gemv_prerotated(gpu, &layer.wv, &s.tmp, x_rot, &s.fa_v)?;
                    }
                    gpu.deinterleave_f32(
                        &s.fa_q_full,
                        &s.fa_q,
                        &s.fa_gate,
                        config.n_heads,
                        config.head_dim,
                    )?;
                    gpu.rmsnorm_batched(
                        &s.fa_q,
                        &layer.q_norm,
                        &s.fa_q,
                        config.n_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;
                    let kv_dim = config.n_kv_heads * config.head_dim;
                    gpu.rmsnorm_batched(
                        &s.fa_k,
                        &layer.k_norm,
                        &s.fa_k,
                        config.n_kv_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;

                    if kv_cache.compact_offset > 0 {
                        let abs = (pos + kv_cache.compact_offset) as i32;
                        gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
                    }
                    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                    gpu.rope_partial_interleaved_f32(
                        &s.fa_q,
                        &s.fa_k,
                        &s.pos_buf,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        n_rot,
                        config.rope_theta,
                    )?;
                    if kv_cache.compact_offset > 0 {
                        let phys = pos as i32;
                        gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
                    }

                    if kv_cache.quant_asym4 {
                        let ct = ct!();
                        let st = st!();
                        if kv_cache.quant_fwht {
                            gpu.kv_cache_write_fwht4_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.v_mode_bits(),
                            )?;
                            gpu.attention_flash_fwht4(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                                kv_cache.v_mode_bits(),
                            )?;
                        } else {
                            gpu.kv_cache_write_asym4_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                            )?;
                            gpu.attention_flash_asym4(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                            )?;
                        }
                    } else if kv_cache.quant_asym3 {
                        let ct = ct!();
                        let st = st!();
                        if kv_cache.quant_fwht {
                            gpu.kv_cache_write_fwht3_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.v_mode_bits(),
                            )?;
                            gpu.attention_flash_fwht3(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                                kv_cache.v_mode_bits(),
                            )?;
                        } else {
                            gpu.kv_cache_write_asym3_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                            )?;
                            gpu.attention_flash_asym3(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                                None,
                            )?;
                        }
                    } else if kv_cache.quant_asym2 {
                        let ct = ct!();
                        let st = st!();
                        if kv_cache.quant_fwht {
                            gpu.kv_cache_write_fwht2_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.v_mode_bits(),
                            )?;
                            gpu.attention_flash_fwht2(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                                kv_cache.v_mode_bits(),
                            )?;
                        } else {
                            gpu.kv_cache_write_asym2_fused(
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_k,
                                &s.fa_v,
                                &s.pos_buf,
                                ct,
                                st,
                                config.n_kv_heads,
                                config.head_dim,
                            )?;
                            gpu.attention_flash_asym2(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                ct,
                                st,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                            )?;
                        }
                    } else if kv_cache.quant_q8 {
                        gpu.kv_cache_write_q8_0(
                            &kv_cache.k_gpu[layer_idx],
                            &s.fa_k,
                            &s.pos_buf,
                            config.n_kv_heads,
                            config.head_dim,
                        )?;
                        gpu.kv_cache_write_q8_0(
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_v,
                            &s.pos_buf,
                            config.n_kv_heads,
                            config.head_dim,
                        )?;
                        let use_flash = gpu.graphs.capture_mode
                            || s.flash_mode == 2
                            || (s.flash_mode == 1 && pos + 1 >= 2048)
                            || pos + 1 > 15000;
                        if use_flash {
                            gpu.attention_flash_q8_0(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                            )?;
                        } else {
                            gpu.attention_q8_0_kv(
                                &s.fa_q,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &s.fa_attn_out,
                                &s.pos_buf,
                                pos + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                            )?;
                        }
                    } else {
                        gpu.kv_cache_write(
                            &kv_cache.k_gpu[layer_idx],
                            &s.fa_k,
                            &s.pos_buf,
                            kv_dim,
                        )?;
                        gpu.kv_cache_write(
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_v,
                            &s.pos_buf,
                            kv_dim,
                        )?;
                        gpu.attention_f32(
                            &s.fa_q,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out,
                            &s.pos_buf,
                            pos + 1,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                        )?;
                    }

                    gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
                    {
                        let ctx = DispatchCtx::new(gpu);
                        let wr = layer.wo.dispatch_ref();
                        execute_steps(
                            gpu,
                            &ctx,
                            &[Step::GemvResidual {
                                w: &wr,
                                input: GemvInput::Raw(&s.fa_attn_out),
                                residual: &s.x,
                                out: &s.x,
                            }],
                        )
                        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                    }

                    if ffn_all_mq4_for_moe(&layer.ffn) {
                        gpu.fused_rmsnorm_rotate_mq(
                            &s.x,
                            &layer.ffn_norm,
                            s.moe_x_rot.as_ref().expect("MoE scratch"),
                            config.dim,
                            config.norm_eps,
                        )?;
                        moe_ffn_decode_with_scratch_prerotated(
                            gpu, &layer.ffn, &s.x, &s.x, config, s,
                        )?;
                    } else {
                        gpu.rmsnorm_f32(&s.x, &layer.ffn_norm, &s.tmp, config.norm_eps)?;
                        moe_ffn_decode_with_scratch(gpu, &layer.ffn, &s.tmp, &s.x, config, s)?;
                    }
                }

                _ => panic!("layer type mismatch at layer {layer_idx}"),
            }
        }

        prev_dev = Some(dev_idx);
    }

    let dev_last = gpus.output_device;
    let s_last = &scratch_set.per_device[dev_last];
    let gpu_last = &mut gpus.devices[dev_last];
    gpu_last.rmsnorm_f32(
        &s_last.x,
        &weights.output_norm,
        &s_last.tmp,
        config.norm_eps,
    )?;
    {
        let ctx = DispatchCtx::new(gpu_last);
        let wr = weights.output.dispatch_ref();
        let step = Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(&s_last.tmp),
            out: &s_last.logits,
        };
        execute_steps(gpu_last, &ctx, &[step])
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    }

    Ok(())
}

/// Multi-GPU decode forward (Stage 5 of multi-GPU pp migration #58).
/// Embedding lookup on dev 0 (token_embd lives there per Stage 4 placement),
/// then the layer loop via `forward_scratch_layers_multi`. `s.logits` ends
/// up on `gpus.output_device`. hipGraph capture is bypassed for pp > 1.
pub fn forward_scratch_multi(
    gpus: &mut Gpus,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch_set: &Qwen35ScratchSet,
) -> HipResult<()> {
    // F3 (review): asym{2,3,4} KV requires per-device givens replicas. The
    // ct!()/st!() macros in forward_scratch_layers_multi fall back to
    // kv_cache.givens_* if the per-device replica is None — which silently
    // hands a wrong-device tensor to attention kernels. Refuse up-front.
    if (kv_cache.quant_asym2 || kv_cache.quant_asym3 || kv_cache.quant_asym4)
        && (gpus.givens_cos_per_dev.len() != gpus.devices.len()
            || gpus.givens_sin_per_dev.len() != gpus.devices.len())
    {
        return Err(hip_bridge::HipError::new(
            0,
            "forward_scratch_multi: asym KV mode requires gpus.givens_*_per_dev \
             populated for every device. Construct KvCache via the *_multi ctor \
             (e.g. KvCache::new_gpu_asym3_capped_multi) — single-GPU ctors leave \
             gpus.givens_*_per_dev empty.",
        ));
    }

    let dim = config.dim;
    let pos_bytes = (pos as i32).to_ne_bytes();
    {
        let gpu0 = &mut gpus.devices[0];
        let s0 = &scratch_set.per_device[0];
        match weights.embd_format {
            EmbeddingFormat::HFQ4G256 => {
                gpu0.embedding_lookup_hfq4g256(&weights.token_embd, &s0.x, token, dim)?
            }
            EmbeddingFormat::HFQ4G128 => {
                gpu0.embedding_lookup_hfq4g128(&weights.token_embd, &s0.x, token, dim)?
            }
            EmbeddingFormat::Q8_0 => {
                gpu0.embedding_lookup_q8(&weights.token_embd, &s0.x, token, dim)?
            }
            EmbeddingFormat::F32 => {
                gpu0.embedding_lookup(&weights.token_embd, &s0.x, token, dim)?
            }
            _ => panic!("unsupported embedding format"),
        }
    }
    // pos_buf written to every device's scratch — every band reads it inside
    // RoPE / KV write for FullAttention layers. F1 (review): bind_thread
    // before each raw gpu.hip.memcpy_htod — HipRuntime methods bypass the
    // Stage 2b bind audit, so without explicit bind the writes land on
    // whatever device was last bound (dev 0 from the embedding lookup above).
    for dev_idx in 0..gpus.devices.len() {
        let gpu = &mut gpus.devices[dev_idx];
        gpu.bind_thread()?;
        let s = &scratch_set.per_device[dev_idx];
        gpu.hip.memcpy_htod(&s.pos_buf, &pos_bytes)?;
    }
    forward_scratch_layers_multi(gpus, weights, config, pos, kv_cache, dn_state, scratch_set)
}

/// Multi-GPU batched prefill (Stage 6 of #58 — multi-gpu pipeline-parallel).
/// Closes the daemon-time pp=1 vs pp=2 divergence — single-GPU
/// `forward_prefill_batch` runs through the WMMA-batched fast path, while
/// pp=2 was previously stuck on per-token `forward_scratch_multi` (a
/// different kernel sequence with a different reduction order). This
/// routes both paths through the same `forward_prefill_chunk` body, just
/// band-restricted via `PrefillBandCtx`.
///
/// Flow per chunk of up to `max_batch` tokens:
///   1. Allocate per-band `PrefillBatchScratch` on each device's pbs.
///   2. Run `forward_prefill_chunk` on dev 0 with band 0 layers,
///      `is_first_band=true` (does the embedding) and
///      `is_last_band=(n_bands==1)`.
///   3. peer-copy band 0's `pbs.x_batch` into band 1's `pbs.x_batch`.
///   4. Run `forward_prefill_chunk` on dev 1 with band 1 layers,
///      `is_first_band=false` (skips embedding, reads already-populated
///      `x_batch`) and `is_last_band=true` (does final norm + lm_head).
///   5. Repeat for any further bands.
///
/// `tree_verify`, DFlash hidden-rb, GdnTape, and per_token_hidden_out
/// are pp=1 only in v1. They've been refused at the daemon load-time
/// gate, so this function does not accept them as parameters.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_multi(
    gpus: &mut Gpus,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch_set: &Qwen35ScratchSet,
) -> HipResult<()> {
    let n_total = tokens.len();
    if n_total == 0 {
        return Ok(());
    }

    let n_bands = gpus.devices.len();
    if n_bands == 0 {
        return Err(hip_bridge::HipError::new(
            0,
            "forward_prefill_batch_multi: no devices",
        ));
    }

    // F3 (review-pattern from forward_scratch_multi): asym{2,3,4} KV requires
    // per-device givens replicas. Refuse up-front — the band-mode macros in
    // forward_prefill_chunk fall back to kv_cache.givens_* if the band's
    // givens override is None, which silently hands a wrong-device tensor
    // to attention kernels.
    if (kv_cache.quant_asym2 || kv_cache.quant_asym3 || kv_cache.quant_asym4)
        && (gpus.givens_cos_per_dev.len() != n_bands || gpus.givens_sin_per_dev.len() != n_bands)
    {
        return Err(hip_bridge::HipError::new(
            0,
            "forward_prefill_batch_multi: asym KV mode requires gpus.givens_*_per_dev \
             populated for every device. Construct KvCache via the *_multi ctor \
             (e.g. KvCache::new_gpu_asym3_capped_multi) — single-GPU ctors leave \
             gpus.givens_*_per_dev empty.",
        ));
    }

    let max_batch: usize = hipfire_config::developer_var("HIPFIRE_PREFILL_MAX_BATCH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v >= 2)
        .unwrap_or(PREFILL_MAX_BATCH);

    let force_fallback = !hipfire_runtime::config::get().prefill_batched;

    // Eligibility: same checks as `forward_prefill_batch_with_pbs`. If any
    // layer fails the batched gate, fall back to per-token forward —
    // correctness preserved at the cost of per-token kernel sequence.
    let arch0 = gpus.devices[0].arch.as_str();
    let moe_topk_ok = config.num_experts_per_tok == 8 && config.num_experts <= 1024;
    let eligible = !force_fallback
        && n_total >= 2
        && dn_state.quant == StateQuant::Q8
        && weights
            .layers
            .iter()
            .any(|lw| matches!(lw, LayerWeights::DeltaNet(_) | LayerWeights::DeltaNetMoe(_),))
        && weights.layers.iter().all(|lw| match lw {
            LayerWeights::DeltaNet(l) => {
                is_batchable_la(l.wqkv.gpu_dtype, arch0)
                    && is_batchable_la(l.wz.gpu_dtype, arch0)
                    && is_batchable_la(l.w_beta.gpu_dtype, arch0)
                    && is_batchable_la(l.w_alpha.gpu_dtype, arch0)
                    && is_batchable_la(l.wo.gpu_dtype, arch0)
                    && is_batchable_la(l.w_gate.gpu_dtype, arch0)
                    && is_batchable_la(l.w_up.gpu_dtype, arch0)
                    && is_batchable_la(l.w_down.gpu_dtype, arch0)
            }
            LayerWeights::FullAttn(l) => {
                is_batchable_la(l.wq.gpu_dtype, arch0)
                    && is_batchable_la(l.wk.gpu_dtype, arch0)
                    && is_batchable_la(l.wv.gpu_dtype, arch0)
                    && is_batchable_la(l.wo.gpu_dtype, arch0)
                    && is_batchable_la(l.w_gate.gpu_dtype, arch0)
                    && is_batchable_la(l.w_up.gpu_dtype, arch0)
                    && is_batchable_la(l.w_down.gpu_dtype, arch0)
            }
            LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_) => moe_topk_ok,
        });

    if !eligible {
        // Per-token fallback. Correctness over speed when the batched
        // path's preconditions are not met.
        for (i, &tok) in tokens.iter().enumerate() {
            forward_scratch_multi(
                gpus,
                weights,
                config,
                tok,
                start_pos + i,
                kv_cache,
                dn_state,
                scratch_set,
            )?;
        }
        return Ok(());
    }

    // Per-band cumulative offsets into LA / FA layer indices. The band's
    // first layer of a given type (DeltaNet or FullAttn) reads
    // `dn_state.s_matrices[delta_off]` / `kv_cache.k_caches[fa_off]`.
    let mut delta_off_per_band = vec![0usize; n_bands];
    let mut fa_off_per_band = vec![0usize; n_bands];
    {
        let mut delta_run = 0usize;
        let mut fa_run = 0usize;
        for b in 0..n_bands {
            delta_off_per_band[b] = delta_run;
            fa_off_per_band[b] = fa_run;
            let band_start = gpus.band_starts[b];
            let band_end = if b + 1 < n_bands {
                gpus.band_starts[b + 1]
            } else {
                config.n_layers
            };
            for li in band_start..band_end {
                match config.layer_types[li] {
                    LayerType::LinearAttention => delta_run += 1,
                    LayerType::FullAttention => fa_run += 1,
                }
            }
        }
    }

    // Allocate one PrefillBatchScratch per band. Each lives on the band's
    // device. Freed at the end of the call (matches forward_prefill_batch's
    // own_pbs pattern). Future opt: cache on Qwen35ScratchSet.
    let mut pbs_per_band: Vec<PrefillBatchScratch> = Vec::with_capacity(n_bands);
    for b in 0..n_bands {
        // hunt3 H-E: PrefillBatchScratch has no Drop impl, so a mid-loop OOM
        // here would silently leak every already-allocated band's ~40 GpuTensors
        // (incl. tens-of-MB MoE grouped-GEMM scratch). On the first failing
        // PrefillBatchScratch::new, free the bands pushed so far on their own
        // devices before propagating the error. Mirrors the single-GPU own_pbs
        // cleanup pattern (allocation failure must not leak prior allocations).
        // The intra-`new` partial-literal leak (a `?` failing partway through
        // the struct literal) is handled inside PrefillBatchScratch::new itself
        // via its alloc ledger, so the failing band's own allocations are also
        // freed before its error reaches here.
        let alloc = {
            let g = &mut gpus.devices[b];
            g.bind_thread()
                .and_then(|()| PrefillBatchScratch::new(g, config, max_batch))
        };
        match alloc {
            Ok(pbs) => pbs_per_band.push(pbs),
            Err(e) => {
                for (prev_b, prev_pbs) in pbs_per_band.into_iter().enumerate() {
                    let pg = &mut gpus.devices[prev_b];
                    let _ = pg.bind_thread();
                    prev_pbs.free_gpu(pg);
                }
                return Err(e);
            }
        }
    }

    let dim = config.dim;
    let dim_row_bytes = dim * 4;

    let result = (|| -> HipResult<()> {
        let mut chunk_start = 0usize;
        while chunk_start < n_total {
            let chunk_end = (chunk_start + max_batch).min(n_total);
            let chunk = &tokens[chunk_start..chunk_end];
            let chunk_n = chunk.len();

            for b in 0..n_bands {
                let band_layer_start = gpus.band_starts[b];
                let band_layer_end = if b + 1 < n_bands {
                    gpus.band_starts[b + 1]
                } else {
                    config.n_layers
                };
                let givens_cos = gpus.givens_cos_per_dev.get(b);
                let givens_sin = gpus.givens_sin_per_dev.get(b);
                let band_ctx = PrefillBandCtx {
                    layer_start: band_layer_start,
                    layer_end: band_layer_end,
                    delta_layer_offset: delta_off_per_band[b],
                    kv_layer_offset: fa_off_per_band[b],
                    is_first_band: b == 0,
                    is_last_band: b + 1 == n_bands,
                    givens_cos,
                    givens_sin,
                };
                {
                    let pbs_b: &PrefillBatchScratch = &pbs_per_band[b];
                    let s_b = &scratch_set.per_device[b];
                    let g_b = &mut gpus.devices[b];
                    forward_prefill_chunk(
                        g_b,
                        weights,
                        config,
                        chunk,
                        start_pos + chunk_start,
                        kv_cache,
                        dn_state,
                        s_b,
                        pbs_b,
                        None, // hidden_rb: pp=1 only
                        None, // per_token_hidden_out: pp=1 only
                        None, // gdn_tape: pp=1 only
                        0,
                        None,  // tree_verify: pp=1 only
                        false, // pre_uploaded
                        Some(&band_ctx),
                        None, // mask_override: multi-GPU PP path doesn't use the MTP probe hook
                        true, // needs_last_token_logits: preserve multi-GPU post-condition
                        None, // max_layer: multi-GPU PP path runs full stack
                        None, // routed_out: PP bands are multi-layer, not EP
                    )?;
                }

                if b + 1 < n_bands {
                    // Hand off the chunk's residual stream to the next band.
                    // pbs.x_batch holds [N × dim] f32 — copy `chunk_n` rows
                    // from band b to band b+1. wait_boundary makes the dst
                    // device wait on the copy's completion event before the
                    // next forward_prefill_chunk dispatch reads x_batch.
                    let copy_bytes = chunk_n * dim_row_bytes;
                    let (left, right) = pbs_per_band.split_at(b + 1);
                    let pbs_src = &left[b];
                    let pbs_dst = &right[0];
                    let evt = gpus.boundary_copy(
                        b,
                        b + 1,
                        &pbs_src.x_batch.buf,
                        &pbs_dst.x_batch.buf,
                        copy_bytes,
                    )?;
                    gpus.wait_boundary(evt)?;
                }
            }

            chunk_start = chunk_end;
        }
        Ok(())
    })();

    for (b, pbs) in pbs_per_band.into_iter().enumerate() {
        let g = &mut gpus.devices[b];
        let _ = g.bind_thread();
        pbs.free_gpu(g);
    }

    result
}
#[cfg(test)]
mod tests {
    use super::super::config::Qwen35BatchParallelism;
    use super::super::config::Qwen35EpBatchReceipt;
    use super::super::config::Qwen35EpReduce;
    use super::*;

    use super::super::batch::for_each_active_span;
    use super::super::batch::lane_bit;
    use super::super::batch::valid_lane_mask;
    use super::super::weights::mixed_expert_tag;
    use hip_bridge::HipError;
    use hip_bridge::HipResult;
    use rdna_compute::DType;

    // ── Qwen3.5 EP pure helpers (CPU-only; no GPU/model) ─────────────────

    /// Minimal shell for private epoch/poison helpers. Never touches device
    /// buffers; `free_gpu` is not Drop, so empty rank vectors are safe.
    fn ep_shell(max_batch: usize) -> Qwen35DecodeBatchEpState {
        Qwen35DecodeBatchEpState {
            ranks: Vec::new(),
            decode_partials: Vec::new(),
            seed_pbs: Vec::new(),
            seed_partials: Vec::new(),
            scratches: Vec::new(),
            lane_states: vec![LaneState::Vacant; max_batch],
            poison_mask: 0,
            epoch: 0,
            max_batch,
            lane_capacity: 128,
            repeat_capacity: 64,
            prefill_chunk: 32,
            moe_layer_count: 0,
            dim: 64,
            norm_eps: 1e-6,
            expert_to_rank: Box::new([]),
            peer_lease: None,
        }
    }

    fn collect_spans(mask: u64, max_batch: usize) -> HipResult<Vec<(usize, usize)>> {
        let mut spans = Vec::new();
        for_each_active_span(mask, max_batch, |start, len| {
            spans.push((start, len));
            Ok(())
        })?;
        Ok(spans)
    }

    /// Exhaustive accepted/rejected matrix for `mixed_expert_tag`, including
    /// unconditional GL rejection in either position.
    #[test]
    fn qwen35_ep_mixed_expert_tag_accepted_rejected_matrix() {
        use DType::*;
        let accepted: &[(DType, DType, u8)] = &[
            (MQ4G256, MQ6G256, 0),
            (MQ4G256, MQ2G256Lloyd, 1),
            (MQ4G256, MQ4G256, 2),
            (MQ4G256, MQ3G256Lloyd, 3),
            (MQ4G256, MFP4G32E8, 4),
            (MQ4G256, MFP3G32E8, 5),
            (MQ4G256, MFP2G32E8, 6),
            (MQ6G256, MQ6G256, 0),
            (MQ2G256Lloyd, MQ2G256Lloyd, 1),
            (MQ3G256Lloyd, MQ3G256Lloyd, 3),
            (MFP4G32E8, MFP4G32E8, 4),
            (MFP3G32E8, MFP3G32E8, 5),
            (MFP2G32E8, MFP2G32E8, 6),
        ];
        for &(g, d, tag) in accepted {
            assert_eq!(
                mixed_expert_tag(g, d).expect("accepted pair"),
                tag,
                "accepted gate={g:?} down={d:?}"
            );
        }

        // Domain under test: every dtype that participates in the tag table,
        // both GL dtypes, plus representative outsiders that must stay Err.
        let domain = [
            MQ4G256,
            MQ6G256,
            MQ2G256Lloyd,
            MQ3G256Lloyd,
            MFP4G32E8,
            MFP3G32E8,
            MFP2G32E8,
            MQ2G256GL,
            MQ3G256GL,
            MQ4G256Lloyd,
            MQ3G256,
            MQ2G256,
            MQ5G256,
            Q8_0,
            F16,
            F32,
            HFP4G32,
            MFP4G32,
            ParoQ4G128,
        ];
        let is_accepted = |gate: DType, down: DType| -> bool {
            accepted.iter().any(|&(g, d, _)| g == gate && d == down)
        };

        for &gate in &domain {
            for &down in &domain {
                let got = mixed_expert_tag(gate, down);
                let gl =
                    matches!(gate, MQ2G256GL | MQ3G256GL) || matches!(down, MQ2G256GL | MQ3G256GL);
                if is_accepted(gate, down) {
                    assert!(
                        got.is_ok(),
                        "expected Ok for gate={gate:?} down={down:?}, got {got:?}"
                    );
                    assert!(!gl, "accepted pair must not include GL");
                } else {
                    let err =
                        got.expect_err(&format!("expected Err for gate={gate:?} down={down:?}"));
                    if gl {
                        assert!(
                            err.message.contains("GL dtype not supported"),
                            "GL rejection message: {}",
                            err.message
                        );
                    } else {
                        assert!(
                            err.message.contains("unsupported dtype pair"),
                            "non-GL rejection message: {}",
                            err.message
                        );
                    }
                }
            }
        }

        // Explicit GL-in-either-position matrix (including both sides GL).
        for gate in [MQ2G256GL, MQ3G256GL, MQ4G256] {
            for down in [MQ2G256GL, MQ3G256GL, MQ4G256] {
                if matches!((gate, down), (MQ4G256, MQ4G256)) {
                    continue; // accepted pair covered above
                }
                if matches!(gate, MQ2G256GL | MQ3G256GL) || matches!(down, MQ2G256GL | MQ3G256GL) {
                    let err = mixed_expert_tag(gate, down).expect_err("GL must reject");
                    assert!(
                        err.message.contains("GL dtype not supported"),
                        "gate={gate:?} down={down:?}: {}",
                        err.message
                    );
                }
            }
        }
    }

    #[test]
    fn qwen35_ep_valid_lane_mask_boundaries() {
        assert!(valid_lane_mask(0).is_err());
        assert_eq!(valid_lane_mask(1).unwrap(), 0b1);
        assert_eq!(valid_lane_mask(2).unwrap(), 0b11);
        assert_eq!(valid_lane_mask(8).unwrap(), 0xff);
        assert_eq!(valid_lane_mask(63).unwrap(), (1u64 << 63) - 1);
        assert_eq!(valid_lane_mask(64).unwrap(), u64::MAX);
        let e65 = valid_lane_mask(65).expect_err("65 out of range");
        assert!(e65.message.contains("max_batch must be 1..64"));
        let e0 = valid_lane_mask(0).expect_err("0 out of range");
        assert!(e0.message.contains("max_batch must be 1..64"));
    }

    #[test]
    fn qwen35_ep_lane_bit_boundaries() {
        assert_eq!(lane_bit(0, 1).unwrap(), 1u64 << 0);
        assert_eq!(lane_bit(0, 64).unwrap(), 1u64 << 0);
        assert_eq!(lane_bit(1, 64).unwrap(), 1u64 << 1);
        assert_eq!(lane_bit(63, 64).unwrap(), 1u64 << 63);

        // lane == max_batch is always out of range (covers 0/1/64/65 edges).
        assert!(lane_bit(0, 0).is_err());
        assert!(lane_bit(1, 1).is_err());
        assert!(lane_bit(63, 63).is_err());
        assert!(lane_bit(64, 64).is_err());
        assert!(lane_bit(65, 64).is_err());
        assert!(lane_bit(64, 65).is_err()); // max_batch > 64
        assert!(lane_bit(0, 65).is_err());

        let e = lane_bit(8, 8).expect_err("lane==max");
        assert!(e.message.contains("lane out of range"));
        let e_mb = lane_bit(0, 65).expect_err("max>64");
        assert!(e_mb.message.contains("max_batch must be 1..64"));
    }

    #[test]
    fn qwen35_ep_for_each_active_span_coverage() {
        // Empty mask → no callbacks.
        assert_eq!(collect_spans(0, 8).unwrap(), Vec::<(usize, usize)>::new());

        // Full low-bit mask.
        assert_eq!(collect_spans(0b1111, 4).unwrap(), vec![(0, 4)]);
        assert_eq!(
            collect_spans(valid_lane_mask(8).unwrap(), 8).unwrap(),
            vec![(0, 8)]
        );

        // Alternating bits.
        assert_eq!(
            collect_spans(0b0101_0101, 8).unwrap(),
            vec![(0, 1), (2, 1), (4, 1), (6, 1)]
        );
        assert_eq!(
            collect_spans(0b1010_1010, 8).unwrap(),
            vec![(1, 1), (3, 1), (5, 1), (7, 1)]
        );

        // Leading ones, trailing zeros.
        assert_eq!(collect_spans(0b0000_1111, 8).unwrap(), vec![(0, 4)]);

        // Trailing ones (high lanes), leading zeros.
        assert_eq!(collect_spans(0b1111_0000, 8).unwrap(), vec![(4, 4)]);

        // Discontiguous multi-span.
        assert_eq!(collect_spans(0b1100_0111, 8).unwrap(), vec![(0, 3), (6, 2)]);
        assert_eq!(
            collect_spans(0b1001_0110, 8).unwrap(),
            vec![(1, 2), (4, 1), (7, 1)]
        );

        // Single bit at edges of a 64-lane batch.
        assert_eq!(collect_spans(1u64 << 0, 64).unwrap(), vec![(0, 1)]);
        assert_eq!(collect_spans(1u64 << 63, 64).unwrap(), vec![(63, 1)]);
        assert_eq!(
            collect_spans((1u64 << 63) | 1, 64).unwrap(),
            vec![(0, 1), (63, 1)]
        );

        // Bits beyond max_batch are rejected.
        let e = collect_spans(0b1000, 3).expect_err("bit3 beyond max=3");
        assert!(e.message.contains("mask has bits beyond max_batch"));
        assert!(collect_spans(1u64 << 8, 8).is_err());
        assert!(for_each_active_span(0, 0, |_, _| Ok(())).is_err());
        assert!(for_each_active_span(0, 65, |_, _| Ok(())).is_err());

        // Callback error propagates.
        let mut n = 0usize;
        let pe = for_each_active_span(0b111, 4, |_, _| {
            n += 1;
            if n == 1 {
                Err(HipError::new(0, "span fail"))
            } else {
                Ok(())
            }
        })
        .expect_err("callback err");
        assert!(pe.message.starts_with("span fail"));
        assert_eq!(n, 1);
    }

    #[test]
    fn qwen35_ep_epoch_and_poison_mask_helpers() {
        let mut ep = ep_shell(8);
        assert_eq!(ep.epoch(), 0);
        assert_eq!(ep.poison_mask(), 0);

        // reserve_next_epoch is pure: returns next without mutating.
        assert_eq!(ep.reserve_next_epoch().unwrap(), 1);
        assert_eq!(ep.epoch(), 0);
        ep.epoch = 41;
        assert_eq!(ep.reserve_next_epoch().unwrap(), 42);
        assert_eq!(ep.epoch(), 41);

        // Overflow exactness.
        ep.epoch = u64::MAX;
        let overflow = ep.reserve_next_epoch().expect_err("epoch overflow");
        assert!(overflow.message.contains("epoch overflow"));
        assert_eq!(ep.epoch(), u64::MAX);

        let overflow2 = ep.checked_advance_epoch().expect_err("advance overflow");
        assert!(overflow2.message.contains("epoch overflow"));
        assert_eq!(ep.epoch(), u64::MAX);

        // Successful advance mutates.
        ep.epoch = 7;
        assert_eq!(ep.checked_advance_epoch().unwrap(), 8);
        assert_eq!(ep.epoch(), 8);

        // poison_lanes: exact OR into mask + only selected lanes → Poisoned.
        ep.epoch = 0;
        ep.lane_states = vec![
            LaneState::Vacant,
            LaneState::Seeding,
            LaneState::Ready { next_position: 3 },
            LaneState::Vacant,
            LaneState::Ready { next_position: 9 },
            LaneState::Vacant,
            LaneState::Vacant,
            LaneState::Vacant,
        ];
        ep.poison_lanes(0b0_0101); // lanes 0 and 2
        assert_eq!(ep.poison_mask(), 0b00101);
        assert_eq!(ep.lane_state(0), Some(LaneState::Poisoned));
        assert_eq!(ep.lane_state(1), Some(LaneState::Seeding)); // untouched
        assert_eq!(ep.lane_state(2), Some(LaneState::Poisoned));
        assert_eq!(
            ep.lane_state(4),
            Some(LaneState::Ready { next_position: 9 })
        );
        assert!(ep.is_poisoned(0));
        assert!(!ep.is_poisoned(1));
        assert!(ep.is_poisoned(2));

        // Second poison ORs bits; already-poisoned stay poisoned.
        ep.poison_lanes(0b1_0010); // lanes 1 and 4
        assert_eq!(ep.poison_mask(), 0b10111);
        assert_eq!(ep.lane_state(1), Some(LaneState::Poisoned));
        assert_eq!(ep.lane_state(4), Some(LaneState::Poisoned));

        // Bits beyond lane_states length are mask-only (loop bounds by len).
        let before = ep.poison_mask();
        ep.poison_lanes(1u64 << 63);
        assert_eq!(ep.poison_mask(), before | (1u64 << 63));

        // clear_poison_lane clears bit and Vacants a Poisoned lane only.
        ep.lane_states[3] = LaneState::Ready { next_position: 1 };
        ep.poison_mask |= 1u64 << 3; // bit set without state=Poisoned
        ep.clear_poison_lane(3);
        assert_eq!(ep.poison_mask() & (1u64 << 3), 0);
        // Ready is preserved when the bit was set without Poisoned state.
        assert_eq!(
            ep.lane_state(3),
            Some(LaneState::Ready { next_position: 1 })
        );

        ep.clear_poison_lane(0);
        assert_eq!(ep.poison_mask() & 1, 0);
        assert_eq!(ep.lane_state(0), Some(LaneState::Vacant));

        // commit_or_poison: Ok commits reserved epoch, leaves poison alone.
        ep.epoch = 10;
        ep.poison_mask = 0;
        ep.lane_states = vec![LaneState::Vacant; 8];
        let v = ep
            .commit_or_poison(0b11, 11, Ok::<u32, HipError>(99))
            .unwrap();
        assert_eq!(v, 99);
        assert_eq!(ep.epoch(), 11);
        assert_eq!(ep.poison_mask(), 0);

        // commit_or_poison: Err poisons affected_mask and does not advance epoch.
        let err = ep
            .commit_or_poison(0b1010, 12, Err::<u32, _>(HipError::new(0, "boom")))
            .expect_err("poison path");
        assert!(err.message.starts_with("boom"));
        assert_eq!(ep.epoch(), 11); // unchanged
        assert_eq!(ep.poison_mask(), 0b1010);
        assert_eq!(ep.lane_state(1), Some(LaneState::Poisoned));
        assert_eq!(ep.lane_state(3), Some(LaneState::Poisoned));
        assert_eq!(ep.lane_state(0), Some(LaneState::Vacant));
        assert_eq!(ep.lane_state(2), Some(LaneState::Vacant));
    }

    /// Ready-only sampling preconditions are pure lane-state decisions.
    /// Full `sample_lane`/`sample_product` need a live `Gpus`; here we assert
    /// the observable Ready gate inputs the production match arms require.
    #[test]
    fn qwen35_ep_ready_only_sampling_decision_preconditions() {
        let mut ep = ep_shell(4);
        // Default shell: every lane Vacant → not Ready.
        for lane in 0..4 {
            assert_ne!(
                ep.lane_state(lane),
                Some(LaneState::Ready { next_position: 0 })
            );
            match ep.lane_state(lane) {
                Some(LaneState::Ready { .. }) => panic!("Vacant must not match Ready"),
                Some(_) => {}
                None => panic!("in-range lane must be Some"),
            }
        }
        // Out-of-range lane is None (sample_lane's final arm).
        assert_eq!(ep.lane_state(4), None);
        assert_eq!(ep.lane_state(64), None);

        // Mark a single Ready lane; only that lane satisfies the Ready arm.
        ep.lane_states[2] = LaneState::Ready { next_position: 17 };
        ep.lane_states[0] = LaneState::Seeding;
        ep.lane_states[1] = LaneState::Poisoned;
        ep.lane_states[3] = LaneState::Vacant;

        for lane in 0..4 {
            let ready = matches!(ep.lane_state(lane), Some(LaneState::Ready { .. }));
            assert_eq!(ready, lane == 2, "lane {lane}");
        }
        match ep.lane_state(2) {
            Some(LaneState::Ready { next_position }) => assert_eq!(next_position, 17),
            other => panic!("expected Ready, got {other:?}"),
        }

        // lane_bit bound check used by sample_lane before state inspection.
        assert!(lane_bit(2, ep.max_batch()).is_ok());
        assert!(lane_bit(4, ep.max_batch()).is_err());

        // Poisoned lane remains not-Ready even if next_position-shaped data
        // would have been valid under Ready.
        ep.poison_lanes(1u64 << 2);
        assert_eq!(ep.lane_state(2), Some(LaneState::Poisoned));
        assert!(!matches!(ep.lane_state(2), Some(LaneState::Ready { .. })));

        // Receipt pure helpers (no GPU): attested invariants + rows overflow.
        let r = Qwen35EpBatchReceipt::new_attested(3, 7, 11);
        assert_eq!(r.epoch(), 3);
        assert_eq!(r.rank_count(), 4);
        assert_eq!(r.rank_mask(), 0x0f);
        assert_eq!(r.rows(), 7);
        assert_eq!(r.moe_collectives(), 11);
        assert_eq!(r.reduce(), Qwen35EpReduce::PeerRootedF32);
        assert_eq!(r.parallelism(), Qwen35BatchParallelism::ExpertParallel);
        assert_eq!(Qwen35EpBatchReceipt::rows_from_usize(0).unwrap(), 0);
        assert_eq!(Qwen35EpBatchReceipt::rows_from_usize(7).unwrap(), 7);
        assert!(Qwen35EpBatchReceipt::rows_from_usize(usize::MAX).is_err());
    }

    #[test]
    fn qwen35_ep_tick_first_band_owns_input_preparation() {
        // CPU-pure guard for the EP tick residency decision: only layer 0
        // stages host tokens/positions/embeddings into each rank's decode
        // `pbs` (separate allocation from `seed_pbs`). Later bands must not
        // overwrite the transformed residual; they are logically pre-prepared.
        assert!(!ep_tick_inputs_prepared(0));
        assert!(ep_tick_inputs_prepared(1));
        assert!(ep_tick_inputs_prepared(39));
        assert!(ep_tick_inputs_prepared(usize::MAX));
        // Saturating representative: any non-zero later band is prepared.
        assert!(ep_tick_inputs_prepared(usize::MAX.saturating_sub(1)));
    }
}
