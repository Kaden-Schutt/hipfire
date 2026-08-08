// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Focused test seams for the legacy Qwen3.5 DeltaNet path.
//!
//! This module is enabled only by the non-default `test-utils` feature. It
//! contains the frozen raw DeltaNet attention-block seam used by parity tests;
//! the production outer Qwen35 forward still owns embeddings, full attention,
//! FFN, output norm, and LM head.

use crate::qwen35::{
    self, DeltaNetState, DeltaNetStateSlot, PrefillBatchScratch, Qwen35Config, Qwen35Scratch,
    Qwen35Weights, StateQuant,
};
use hip_bridge::HipResult;
use hipfire_runtime::llama::KvCache;
use rdna_compute::{Gpu, GpuTensor};
use std::cell::Cell;

thread_local! {
    static RAW_DELTANET: Cell<bool> = const { Cell::new(false) };
}

struct RawDeltaNetScopeGuard {
    previous: bool,
}

impl Drop for RawDeltaNetScopeGuard {
    fn drop(&mut self) {
        RAW_DELTANET.with(|enabled| enabled.set(self.previous));
    }
}

pub fn raw_delta_net_enabled() -> bool {
    RAW_DELTANET.with(Cell::get)
}

/// Run any production outer-forward call with the test-only raw DeltaNet
/// attention seam enabled. This is the behavioral switch used by Task 6/7
/// parity harnesses for decode, batched prefill/tree, and replay paths.
pub fn with_raw_delta_net_scope<T>(f: impl FnOnce() -> T) -> T {
    RAW_DELTANET.with(|enabled| {
        let _guard = RawDeltaNetScopeGuard {
            previous: enabled.replace(true),
        };
        f()
    })
}

/// Invalidate the plain-AR replay state before the legacy oracle runs on
/// caller-owned buffers. Kept separate from HIP graph destruction so the
/// state transition can be tested without a GPU or captured graph handle.
pub fn invalidate_legacy_replay(graphs: &mut rdna_compute::graph::GraphState) {
    graphs.ar_graph_eligible = false;
    graphs.mark_kernels_dirty();
}

/// Run the existing legacy Qwen3.5 forward sequence with caller-owned state.
///
/// `dn_state` must be independently initialized by the caller, and `scratch`
/// must likewise be caller-owned. This wrapper exists solely so focused
/// DeltaNet tests can exercise the legacy sequence without adding a production
/// call site or silently allocating hidden state.
pub fn legacy_delta_net_sequence(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    // The legacy oracle must always run directly against the caller's buffers.
    // Destroy the captured AR graph first: its retained kernarg blobs may point
    // at older scratch/state allocations, and merely disabling replay would
    // leave those pointers live for a later capture.
    gpu.graphs.graph_destroy_checked(&gpu.hip, gpu.device_id)?;
    invalidate_legacy_replay(&mut gpu.graphs);

    with_raw_delta_net_scope(|| {
        qwen35::forward_scratch(
            gpu, weights, config, token, pos, kv_cache, dn_state, scratch,
        )
    })
}

/// Frozen, pre-Step DeltaNet decode body. The caller owns every tensor and
/// recurrent state object; this function performs only the raw kernel sequence
/// that the production Step builder replaces.
#[allow(clippy::too_many_arguments)]
pub(crate) fn raw_delta_net_decode_body(
    gpu: &mut Gpu,
    dt_bias: &GpuTensor,
    a_log: &GpuTensor,
    conv_weight: &GpuTensor,
    norm_weight: &GpuTensor,
    slot: DeltaNetStateSlot<'_>,
    s: &Qwen35Scratch,
    config: &Qwen35Config,
    quant: StateQuant,
) -> HipResult<()> {
    let n_v_heads = config.linear_num_value_heads;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let hd = config.linear_key_head_dim;

    gpu.fused_sigmoid_alpha_gate_f32(&s.dn_beta, &s.dn_alpha, dt_bias, a_log, n_v_heads)?;
    gpu.conv1d_silu_split_f32(
        &s.dn_q_raw,
        &s.dn_k_raw,
        &s.dn_v,
        &s.dn_qkv,
        conv_weight,
        slot.conv,
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
        gpu.repeat_interleave_qk_f32(
            &s.dn_q_raw,
            &s.dn_k_raw,
            &s.dn_q,
            &s.dn_k,
            config.linear_num_key_heads,
            n_v_heads / config.linear_num_key_heads,
            hd,
        )?;
    } else {
        gpu.memcpy_dtod_auto(&s.dn_q.buf, &s.dn_q_raw.buf, k_dim * 4)?;
        gpu.memcpy_dtod_auto(&s.dn_k.buf, &s.dn_k_raw.buf, k_dim * 4)?;
    }
    match quant {
        StateQuant::FP32 => gpu.gated_delta_net_f32(
            &s.dn_q,
            &s.dn_k,
            &s.dn_v,
            &s.dn_alpha,
            &s.dn_beta,
            slot.s,
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
            slot.s,
            slot.scales,
            &s.dn_attn_out,
            1,
            n_v_heads,
            config.linear_value_head_dim,
            slot.ef,
        )?,
        StateQuant::Q4 => gpu.gated_delta_net_q4(
            &s.dn_q,
            &s.dn_k,
            &s.dn_v,
            &s.dn_alpha,
            &s.dn_beta,
            slot.s,
            slot.scales,
            &s.dn_attn_out,
            1,
            n_v_heads,
            config.linear_value_head_dim,
        )?,
    }
    gpu.gated_norm_f32(
        &s.dn_attn_out,
        &s.dn_z,
        norm_weight,
        &s.dn_normed,
        n_v_heads,
        config.linear_value_head_dim,
        config.norm_eps,
    )
}

/// Raw batch/tree DeltaNet body after the caller has performed the pre-GDN
/// alpha/beta gate and optional DFlash tape copy. This intentionally contains
/// direct kernel calls rather than `Step` construction.
#[allow(clippy::too_many_arguments)]
pub(crate) fn raw_delta_net_batch_body(
    gpu: &mut Gpu,
    conv_weight: &GpuTensor,
    norm_weight: &GpuTensor,
    slot: DeltaNetStateSlot<'_>,
    pbs: &PrefillBatchScratch,
    n: usize,
    parent_indices: Option<&GpuTensor>,
    tape_f32: Option<&GpuTensor>,
    tape_q8: Option<&GpuTensor>,
    tape_scales: Option<&GpuTensor>,
    quant: StateQuant,
    intent: hipfire_dispatch::ops::delta_net::DeltaNetBatchIntent,
    config: &Qwen35Config,
) -> HipResult<()> {
    let n_v_heads = config.linear_num_value_heads;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let hd = config.linear_key_head_dim;
    let parents = parent_indices;

    if let Some(parents) = parents {
        gpu.conv1d_silu_split_tree_f32_n(
            &pbs.dn_q_raw_batch,
            &pbs.dn_k_raw_batch,
            &pbs.dn_v_batch,
            &pbs.dn_qkv_batch,
            conv_weight,
            slot.conv,
            parents,
            k_dim,
            v_dim,
            n,
        )?;
    } else {
        gpu.conv1d_silu_split_f32_n(
            &pbs.dn_q_raw_batch,
            &pbs.dn_k_raw_batch,
            &pbs.dn_v_batch,
            &pbs.dn_qkv_batch,
            conv_weight,
            slot.conv,
            k_dim,
            v_dim,
            n,
        )?;
    }
    gpu.fused_qk_l2_norm_scale_f32_batched(
        &pbs.dn_q_raw_batch,
        &pbs.dn_k_raw_batch,
        config.linear_num_key_heads,
        hd,
        1.0 / (hd as f32).sqrt(),
        config.norm_eps,
        n,
    )?;
    if config.linear_num_key_heads < n_v_heads {
        gpu.repeat_interleave_qk_f32_batched(
            &pbs.dn_q_raw_batch,
            &pbs.dn_k_raw_batch,
            &pbs.dn_q_batch,
            &pbs.dn_k_batch,
            config.linear_num_key_heads,
            n_v_heads / config.linear_num_key_heads,
            hd,
            n,
        )?;
    } else {
        gpu.memcpy_dtod_auto(&pbs.dn_q_batch.buf, &pbs.dn_q_raw_batch.buf, n * k_dim * 4)?;
        gpu.memcpy_dtod_auto(&pbs.dn_k_batch.buf, &pbs.dn_k_raw_batch.buf, n * k_dim * 4)?;
    }

    match (parents, quant) {
        (Some(parents), StateQuant::FP32) => gpu.gated_delta_net_f32_tree_batch_seq(
            &pbs.dn_q_batch,
            &pbs.dn_k_batch,
            &pbs.dn_v_batch,
            &pbs.dn_alpha_batch,
            &pbs.dn_beta_batch,
            slot.s,
            tape_f32.expect("FP32 tree tape"),
            parents,
            &pbs.dn_attn_out_batch,
            n,
            n_v_heads,
            config.linear_value_head_dim,
        )?,
        (Some(parents), StateQuant::Q8) => gpu.gated_delta_net_q8_tree_batch_seq(
            &pbs.dn_q_batch,
            &pbs.dn_k_batch,
            &pbs.dn_v_batch,
            &pbs.dn_alpha_batch,
            &pbs.dn_beta_batch,
            slot.s,
            slot.scales,
            tape_q8.expect("Q8 tree tape"),
            tape_scales.expect("Q8 tree scales"),
            parents,
            &pbs.dn_attn_out_batch,
            n,
            n_v_heads,
            config.linear_value_head_dim,
        )?,
        (Some(_), StateQuant::Q4) => {
            return Err(hip_bridge::HipError::new(
                0,
                "Q4 DeltaNet tree tape is unsupported",
            ))
        }
        (None, StateQuant::FP32)
            if matches!(
                intent,
                hipfire_dispatch::ops::delta_net::DeltaNetBatchIntent::NormalPrefill
            ) && rdna_compute::norm::gdn_chunked()
                && n > 1 =>
        {
            gpu.gated_delta_net_f32_chunked(
                &pbs.dn_q_batch,
                &pbs.dn_k_batch,
                &pbs.dn_v_batch,
                &pbs.dn_alpha_batch,
                &pbs.dn_beta_batch,
                slot.s,
                &pbs.dn_attn_out_batch,
                n,
                n_v_heads,
                config.linear_value_head_dim,
                rdna_compute::norm::gdn_chunk_size(),
            )?
        }
        (None, StateQuant::FP32) => gpu.gated_delta_net_f32_batch_seq(
            &pbs.dn_q_batch,
            &pbs.dn_k_batch,
            &pbs.dn_v_batch,
            &pbs.dn_alpha_batch,
            &pbs.dn_beta_batch,
            slot.s,
            &pbs.dn_attn_out_batch,
            n,
            n_v_heads,
            config.linear_value_head_dim,
        )?,
        (None, StateQuant::Q8) => gpu.gated_delta_net_q8_batch_seq(
            &pbs.dn_q_batch,
            &pbs.dn_k_batch,
            &pbs.dn_v_batch,
            &pbs.dn_alpha_batch,
            &pbs.dn_beta_batch,
            slot.s,
            slot.scales,
            &pbs.dn_attn_out_batch,
            n,
            n_v_heads,
            config.linear_value_head_dim,
            slot.ef,
        )?,
        (None, StateQuant::Q4) => gpu.gated_delta_net_q4(
            &pbs.dn_q_batch,
            &pbs.dn_k_batch,
            &pbs.dn_v_batch,
            &pbs.dn_alpha_batch,
            &pbs.dn_beta_batch,
            slot.s,
            slot.scales,
            &pbs.dn_attn_out_batch,
            n,
            n_v_heads,
            config.linear_value_head_dim,
        )?,
    }
    gpu.gated_norm_f32_batched(
        &pbs.dn_attn_out_batch,
        &pbs.dn_z_batch,
        norm_weight,
        &pbs.dn_normed_batch,
        n_v_heads,
        config.linear_value_head_dim,
        config.norm_eps,
        n,
    )
}

/// Independent raw DeltaNet gate preparation for batched prefill/tree. Kept
/// separate from the body so callers can place the DFlash tape boundary after
/// alpha/beta preparation exactly as the legacy sequence did.
pub(crate) fn raw_delta_net_gate_prep(
    gpu: &mut Gpu,
    beta: &GpuTensor,
    alpha: &GpuTensor,
    dt_bias: &GpuTensor,
    a_log: &GpuTensor,
    n_heads: usize,
    batch: usize,
) -> HipResult<()> {
    gpu.fused_sigmoid_alpha_gate_f32_batched(beta, alpha, dt_bias, a_log, n_heads, batch)
}

/// Raw speculative-replay body. Replay owns no output normalization; it only
/// advances the caller-owned convolution and recurrent state across B tokens.
#[allow(clippy::too_many_arguments)]
pub(crate) fn raw_delta_net_replay_body(
    gpu: &mut Gpu,
    conv_weight: &GpuTensor,
    slot: DeltaNetStateSlot<'_>,
    qkv: &GpuTensor,
    q_raw: &GpuTensor,
    k_raw: &GpuTensor,
    v: &GpuTensor,
    q: &GpuTensor,
    k: &GpuTensor,
    alpha: &GpuTensor,
    beta: &GpuTensor,
    out: &GpuTensor,
    n: usize,
    n_key_heads: usize,
    n_value_heads: usize,
    head_dim: usize,
    key_dim: usize,
    value_dim: usize,
    eps: f32,
    quant: StateQuant,
) -> HipResult<()> {
    gpu.conv1d_silu_split_f32_n(
        q_raw,
        k_raw,
        v,
        qkv,
        conv_weight,
        slot.conv,
        key_dim,
        value_dim,
        n,
    )?;
    gpu.fused_qk_l2_norm_scale_f32_batched(
        q_raw,
        k_raw,
        n_key_heads,
        head_dim,
        1.0 / (head_dim as f32).sqrt(),
        eps,
        n,
    )?;
    if n_key_heads < n_value_heads {
        gpu.repeat_interleave_qk_f32_batched(
            q_raw,
            k_raw,
            q,
            k,
            n_key_heads,
            n_value_heads / n_key_heads,
            head_dim,
            n,
        )?;
    } else {
        gpu.memcpy_dtod_auto(&q.buf, &q_raw.buf, n * key_dim * 4)?;
        gpu.memcpy_dtod_auto(&k.buf, &k_raw.buf, n * key_dim * 4)?;
    }
    match quant {
        StateQuant::FP32 => gpu.gated_delta_net_f32_batch_seq(
            q,
            k,
            v,
            alpha,
            beta,
            slot.s,
            out,
            n,
            n_value_heads,
            value_dim / n_value_heads,
        ),
        StateQuant::Q8 => gpu.gated_delta_net_q8_batch_seq(
            q,
            k,
            v,
            alpha,
            beta,
            slot.s,
            slot.scales,
            out,
            n,
            n_value_heads,
            value_dim / n_value_heads,
            slot.ef,
        ),
        StateQuant::Q4 => gpu.gated_delta_net_q4(
            q,
            k,
            v,
            alpha,
            beta,
            slot.s,
            slot.scales,
            out,
            n,
            n_value_heads,
            value_dim / n_value_heads,
        ),
    }
}
