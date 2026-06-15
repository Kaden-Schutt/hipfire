// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MiniMax-M2 forward pass (free functions — hot-path static dispatch).
//!
//! Per-layer pipeline (validated vs the HF `MiniMaxM2` modeling oracle to
//! cosine 0.9996):
//!   h += o_proj · attn( qk_norm(q/k/v_proj(rmsnorm(h))) + partial-RoPE )   [GQA, Q8 KV]
//!   h += combine( experts( sigmoid+bias top-8 route( rmsnorm(h) ) ) )       [MoE]
//! then logits = lm_head( rmsnorm(h) ).
//!
//! Attention weights are Q8 (plain input). The router is Q8 (plain). Routed
//! experts are FWHT-pre-rotated (MQ4G256 / MQ2G256Lloyd / MQ6G256): the input
//! is rotated (`rotate_x_mq_for`) and the silu output rotated
//! (`fused_silu_mul_rotate_mq_batched_for`) before the indexed-MoE GEMV kernels
//! — exactly qwen35's / deepseek4's MoE path. Routing uses `sigmoid_f32` +
//! `deepseek4_moe_topk_bias_aware_f32` with route_scale = 1.0 (MiniMax-M2
//! applies no routed-scaling factor).
//!
//! Decode has two entry points: `decode_step` (eager, used for prefill +
//! warmup) and `decode_step_with_graph` (hipGraph capture/replay of the
//! 62-layer body + lm_head, recovering the ~9% per-token launch-latency gap on
//! gfx11/gfx12 — see the gfx1151 perfmaxx characterization). Both share
//! `decode_step_body`; the only per-token-varying GPU input is the device
//! position scalar (`pos_buf`), staged from the heap-stable `state.pos_host`
//! so the captured memcpy re-reads it on each replay. The embedding lookup is
//! kept OUTSIDE the captured region (token_id is baked into its kernarg).

use crate::minimax::{MiniMaxConfig, MiniMaxLayerWeights, MiniMaxState, MiniMaxWeights};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::superop::{
    self, ForwardBindings, OpBinding, OpFlavor, SuperOp, SuperOpKind, WeightSlot,
};
use hipfire_dispatch::types::DispatchError;
use hipfire_runtime::llama::{
    fused_silu_mul_rotate_mq_batched_for, rotate_x_mq_batched_for, rotate_x_mq_for, weight_gemv,
    weight_gemv_residual,
};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Decode one token (eager); returns the full logits vector. Used for prefill,
/// the warm pass, and as the `HIPFIRE_MINIMAX_GRAPH=0` fallback.
pub fn decode_step(
    cfg: &MiniMaxConfig,
    weights: &MiniMaxWeights,
    state: &mut MiniMaxState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    gpu.embedding_lookup_q8(&weights.embed, &state.h, token_id, cfg.hidden_size)
        .map_err(|e| format!("minimax: embed lookup: {e:?}"))?;
    decode_step_body(cfg, weights, state, gpu, position, None)?;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("minimax: download logits: {e:?}"))
}

/// Decode one token, appending each layer's post-residual hidden state
/// (pre final-norm) to `capture[layer]` — used by the oracle dumper. Set
/// `HIPFIRE_MINIMAX_CAPTURE_POSTATTN` to capture the post-attention residual
/// (pre-MoE) instead, for attention-vs-MoE divergence localization. Eager
/// only (the per-layer D2H downloads are incompatible with graph capture).
pub fn decode_step_capture(
    cfg: &MiniMaxConfig,
    weights: &MiniMaxWeights,
    state: &mut MiniMaxState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    capture: &mut [Vec<f32>],
) -> Result<(), String> {
    gpu.embedding_lookup_q8(&weights.embed, &state.h, token_id, cfg.hidden_size)
        .map_err(|e| format!("minimax: embed lookup: {e:?}"))?;
    decode_step_body(cfg, weights, state, gpu, position, Some(capture))
}

/// Decode one token via hipGraph capture/replay. **Opt-in, default OFF**
/// (`HIPFIRE_MINIMAX_GRAPH=1` to enable). The 62-layer body + lm_head are
/// captured once and replayed per token.
///
/// Output is byte-for-byte identical to eager `decode_step` (validated over 96
/// greedy tokens). But the perf payoff is marginal: on gfx1151 (Strix Halo —
/// the only arch MiniMax's 86 GB footprint fits) it measured **+1.0%**
/// (27.68 → 27.95 tok/s, tight variance), NOT the ~9% the inter-kernel-gap
/// analysis predicted. Root cause: the 9.7% decode launch/idle gap is GPU
/// command-processor inter-kernel dispatch latency, not host-launch overhead —
/// the host thread already runs ahead of the 90%-busy iGPU, so removing the
/// host launch API cost (all hipGraph does) recovers ~nothing. This matches the
/// DeepSeek-V4 "hipGraph dead on gfx1151 decode" finding. Kept as a validated
/// opt-in (may help on a faster CP, e.g. a gfx12 dGPU, if MiniMax ever fits one).
///
/// Capture-safety invariants (mirrors the proven DeepSeek-V4 path):
///   - token_id is per-token → embedding runs OUTSIDE the capture.
///   - position is per-token → staged via `state.pos_host` (stable `Box`); the
///     captured `memcpy_htod_auto` re-reads it on every replay.
///   - attention launch geometry is sized for `state.max_seq` (constant), not
///     the live `seq_len`, so the baked grid/shared-mem stays valid as the KV
///     length grows (the kernel reads the true length from `pos_buf[0]+1`).
pub fn decode_step_with_graph(
    cfg: &MiniMaxConfig,
    weights: &MiniMaxWeights,
    state: &mut MiniMaxState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    use std::sync::OnceLock;
    static GRAPH_ENV: OnceLock<Option<bool>> = OnceLock::new();
    let env_override =
        *GRAPH_ENV.get_or_init(
            || match std::env::var("HIPFIRE_MINIMAX_GRAPH").ok().as_deref() {
                Some("1") => Some(true),
                Some("0") => Some(false),
                _ => None,
            },
        );
    // Default OFF — measured only +1.0% on gfx1151 (the sole arch MiniMax fits);
    // the decode gap is GPU-CP dispatch latency, not host-launch overhead, so
    // hipGraph recovers ~nothing here. Opt in with HIPFIRE_MINIMAX_GRAPH=1.
    let graph_on = env_override.unwrap_or(false);
    if !graph_on {
        return decode_step(cfg, weights, state, gpu, token_id, position);
    }

    // Warmup: first decode after a fresh load runs eager (JITs kernels + settles
    // DPM) and drops any stale graph from a previously-loaded model so the next
    // call captures fresh for THIS model's weight pointers.
    if !state.ar_warmed_up {
        state.ar_warmed_up = true;
        gpu.graphs.graph_exec = None;
        return decode_step(cfg, weights, state, gpu, token_id, position);
    }

    // Capture + replay both need an explicit (non-null) stream.
    if gpu.active_stream.is_none() {
        let s = gpu
            .hip
            .stream_create()
            .map_err(|e| format!("minimax graph: stream_create: {e:?}"))?;
        gpu.active_stream = Some(s);
    }

    // Embedding lookup OUTSIDE the captured region — token_id is baked into the
    // embedding kernarg. Runs on the active stream, ordered before the captured
    // body that reads `state.h`.
    gpu.embedding_lookup_q8(&weights.embed, &state.h, token_id, cfg.hidden_size)
        .map_err(|e| format!("minimax graph: embed lookup: {e:?}"))?;

    if gpu.graphs.graph_exec.is_none() {
        // ── Capture phase ──────────────────────────────────────────────
        // decode_step_body stages pos_host → pos_buf via memcpy_htod_auto
        // INSIDE the capture, so the recorded memcpy node re-reads pos_host
        // on each replay.
        //
        // API drift (integration/dispatch-migration): the hipGraph capture
        // helpers moved into the `gpu.graphs` substruct and now take
        // (&hip, device_id, &stream) — same shape as the LFM2.5-MoE +
        // DeepSeek-V4 graph paths on this branch.
        gpu.graphs
            .begin_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("minimax begin_graph_capture: {e:?}"))?;
        decode_step_body(cfg, weights, state, gpu, position, None)?;
        gpu.graphs
            .end_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("minimax end_graph_capture: {e:?}"))?;
        // Captured kernels were RECORDED, not run — launch once so this token's
        // logits actually get produced.
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("minimax graph_launch (capture): {e:?}"))?;
        eprintln!(
            "[MiniMax hipGraph] captured decode forward — {} kernarg blobs retained",
            gpu.graphs.capture_blobs.len()
        );
    } else {
        // ── Replay phase ───────────────────────────────────────────────
        // Host-only update of the stable position source; the captured memcpy
        // re-reads it and propagates to pos_buf (read by rope / kv-write /
        // attention).
        state.pos_host[0] = position as i32;
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("minimax graph_launch (replay): {e:?}"))?;
    }
    state.n_tokens = position as usize + 1;

    // Logits download is outside the captured region (sync dtoh completes after
    // the captured kernels, which the device observes on the active stream).
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("minimax graph: download logits: {e:?}"))
}

/// The capturable core: stage the device position scalar, run the 62-layer
/// attention+MoE pipeline, then final-norm + lm_head. Does NOT do the embedding
/// lookup (the caller stages `state.h`). Under graph capture, `capture` is
/// `None` (no D2H); the oracle dumper passes `Some(..)` and runs eager only.
fn decode_step_body(
    cfg: &MiniMaxConfig,
    weights: &MiniMaxWeights,
    state: &mut MiniMaxState,
    gpu: &mut Gpu,
    position: u32,
    mut capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let inter = cfg.intermediate_size;
    let n_exp = cfg.num_local_experts;
    let k_top = cfg.num_experts_per_tok;
    let eps = cfg.rms_norm_eps;
    let seq_len = position as usize + 1;
    let capture_postattn = std::env::var_os("HIPFIRE_MINIMAX_CAPTURE_POSTATTN").is_some();

    // Device position scalar (i32) for rope / kv-write / attention. Staged from
    // the heap-stable `state.pos_host` so the captured memcpy re-reads it on
    // replay (memcpy_htod_auto → async on the capture stream when capturing).
    state.pos_host[0] = position as i32;
    {
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(state.pos_host.as_ptr() as *const u8, 4) };
        gpu.memcpy_htod_auto(&state.pos_buf, pos_bytes)
            .map_err(|e| format!("minimax: htod pos: {e:?}"))?;
    }

    // #397 Ship 6 — forward-as-pipeline. HIPFIRE_FORWARD_LOWERED=1 routes the
    // per-layer decode through the super-op executor (run_layer_program). Skipped
    // when capturing (oracle dumper needs the hand path). Default off (opt-in)
    // until hipx byte-parity validated (minimax only fits on hipx).
    if minimax_forward_lowered_enabled() && capture.is_none() {
        return decode_step_body_lowered(cfg, weights, state, gpu, position);
    }

    for (l, layer) in weights.layers.iter().enumerate() {
        // ── Attention block (Q8 projections → plain input) ──────────────────
        gpu.rmsnorm_f32(&state.h, &layer.attn_norm, &state.tmp, eps)
            .map_err(|e| format!("minimax L{l}: attn rmsnorm: {e:?}"))?;
        weight_gemv(gpu, &layer.wq, &state.tmp, &state.fa_q)
            .map_err(|e| format!("minimax L{l}: q_proj: {e}"))?;
        weight_gemv(gpu, &layer.wk, &state.tmp, &state.fa_k)
            .map_err(|e| format!("minimax L{l}: k_proj: {e}"))?;
        weight_gemv(gpu, &layer.wv, &state.tmp, &state.fa_v)
            .map_err(|e| format!("minimax L{l}: v_proj: {e}"))?;

        // Per-LAYER QK-norm: RMSNorm over the whole flat q[q_dim]/k[kv_dim]
        // vector (batch=1), BEFORE head reshape.
        if cfg.use_qk_norm {
            gpu.rmsnorm_batched(&state.fa_q, &layer.q_norm, &state.fa_q, 1, q_dim, eps)
                .map_err(|e| format!("minimax L{l}: q_norm: {e:?}"))?;
            gpu.rmsnorm_batched(&state.fa_k, &layer.k_norm, &state.fa_k, 1, kv_dim, eps)
                .map_err(|e| format!("minimax L{l}: k_norm: {e:?}"))?;
        }

        // Partial rotate_half RoPE on the first `rotary_dim` of each head.
        gpu.rope_partial_interleaved_f32(
            &state.fa_q,
            &state.fa_k,
            &state.pos_buf,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.rotary_dim,
            cfg.rope_theta,
        )
        .map_err(|e| format!("minimax L{l}: rope: {e:?}"))?;

        // KV cache write (Q8) + GQA attention. The attention kernel reads the
        // live KV length from `pos_buf[0]+1`; we pass `state.max_seq` as the
        // geometry hint (NOT `seq_len`) so the captured launch grid / shared-mem
        // is sized for the max and stays valid as the cache grows on replay.
        gpu.kv_cache_write_q8_0(
            &state.kv.k_gpu[l],
            &state.fa_k,
            &state.pos_buf,
            cfg.num_key_value_heads,
            cfg.head_dim,
        )
        .map_err(|e| format!("minimax L{l}: kv write k: {e:?}"))?;
        gpu.kv_cache_write_q8_0(
            &state.kv.v_gpu[l],
            &state.fa_v,
            &state.pos_buf,
            cfg.num_key_value_heads,
            cfg.head_dim,
        )
        .map_err(|e| format!("minimax L{l}: kv write v: {e:?}"))?;
        gpu.attention_q8_0_kv(
            &state.fa_q,
            &state.kv.k_gpu[l],
            &state.kv.v_gpu[l],
            &state.fa_attn_out,
            &state.pos_buf,
            state.max_seq,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            state.kv.physical_cap,
        )
        .map_err(|e| format!("minimax L{l}: attention: {e:?}"))?;

        // o_proj + residual: h += W_o · attn_out.
        weight_gemv_residual(gpu, &layer.wo, &state.fa_attn_out, &state.h)
            .map_err(|e| format!("minimax L{l}: o_proj: {e}"))?;

        if capture_postattn {
            if let Some(cap) = capture.as_deref_mut() {
                let h = gpu
                    .download_f32(&state.h)
                    .map_err(|e| format!("minimax L{l}: postattn capture: {e:?}"))?;
                cap[l].extend_from_slice(&h);
            }
        }

        // ── MoE block (no shared expert) ────────────────────────────────────
        // ffn_tmp = rmsnorm(h) (plain, feeds the Q8 router); ffn_x_rot =
        // FWHT(ffn_tmp) (feeds the FWHT-pre-rotated experts).
        gpu.rmsnorm_f32(&state.h, &layer.ffn_norm, &state.ffn_tmp, eps)
            .map_err(|e| format!("minimax L{l}: ffn rmsnorm: {e:?}"))?;
        rotate_x_mq_for(
            gpu,
            &layer.experts[0].gate_up,
            &state.ffn_tmp,
            &state.ffn_x_rot,
            hidden,
        )
        .map_err(|e| format!("minimax L{l}: ffn rotate: {e:?}"))?;

        // Router: sigmoid(logits) + bias-aware top-k (gather unbiased + normalize;
        // route_scale = 1.0 — MiniMax-M2 applies no routed-scaling factor).
        weight_gemv(gpu, &layer.router, &state.ffn_tmp, &state.router_logits)
            .map_err(|e| format!("minimax L{l}: router: {e}"))?;
        gpu.sigmoid_f32(&state.router_logits)
            .map_err(|e| format!("minimax L{l}: sigmoid: {e:?}"))?;
        gpu.deepseek4_moe_topk_bias_aware_f32(
            &state.router_logits,
            &layer.routing_bias,
            &state.topk_indices,
            &state.topk_weights,
            n_exp as i32,
            k_top as i32,
            1.0,
        )
        .map_err(|e| format!("minimax L{l}: topk: {e:?}"))?;

        // Routed experts: gate_up (rotated input) → silu·mul·rotate → down → combine.
        // Dispatch the indexed-MoE GEMV by expert dtype. MQ4/MQ6/MQ2-Lloyd are
        // FWHT-pre-rotated (byte-compatible with the matching hfq/lloyd kernels
        // given rotated input). The hfq4/hfq6 family uses a separate down +
        // `moe_down_combine`; the MQ2-Lloyd down is residual-scaled (fuses the
        // weighted combine into the down GEMV, accumulating into h directly).
        let edt = layer.experts[0].gate_up.gpu_dtype;
        match edt {
            DType::MQ4G256 | DType::HFQ4G256 => gpu
                .gemv_hfq4g256_moe_gate_up_k8_indexed(
                    &layer.expert_gate_up_ptrs,
                    &state.topk_indices,
                    &state.ffn_x_rot,
                    &state.gate_batch,
                    &state.up_batch,
                    2 * inter,
                    hidden,
                )
                .map_err(|e| format!("minimax L{l}: gate_up hfq4: {e:?}"))?,
            DType::MQ6G256 | DType::HFQ6G256 => gpu
                .gemv_hfq6g256_moe_gate_up_k8_indexed(
                    &layer.expert_gate_up_ptrs,
                    &state.topk_indices,
                    &state.ffn_x_rot,
                    &state.gate_batch,
                    &state.up_batch,
                    2 * inter,
                    hidden,
                )
                .map_err(|e| format!("minimax L{l}: gate_up hfq6: {e:?}"))?,
            DType::MQ2G256Lloyd => gpu
                .deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
                    &layer.expert_gate_up_ptrs,
                    &state.topk_indices,
                    &state.ffn_x_rot,
                    &state.gate_batch,
                    &state.up_batch,
                    2 * inter,
                    hidden,
                    k_top,
                )
                .map_err(|e| format!("minimax L{l}: gate_up mq2l: {e:?}"))?,
            DType::MQ3G256Lloyd => gpu
                .deepseek4_gemv_mq3g256_lloyd_moe_gate_up_indexed(
                    &layer.expert_gate_up_ptrs,
                    &state.topk_indices,
                    &state.ffn_x_rot,
                    &state.gate_batch,
                    &state.up_batch,
                    2 * inter,
                    hidden,
                    k_top,
                )
                .map_err(|e| format!("minimax L{l}: gate_up mq3l: {e:?}"))?,
            other => return Err(format!("minimax L{l}: unsupported expert dtype {other:?}")),
        }

        fused_silu_mul_rotate_mq_batched_for(
            gpu,
            &layer.experts[0].down,
            &state.gate_batch,
            &state.up_batch,
            &state.rot_batch,
            inter,
            k_top,
        )
        .map_err(|e| format!("minimax L{l}: silu_mul_rotate: {e:?}"))?;

        // Down dispatches on the DOWN proj's own dtype (may differ from gate_up:
        // e.g. gate_up=mq2-lloyd + down=mq4, since down carries ~24x the energy).
        let ddt = layer.experts[0].down.gpu_dtype;
        match ddt {
            DType::MQ4G256 | DType::HFQ4G256 => {
                gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                    &layer.expert_down_ptrs,
                    &state.topk_indices,
                    &state.rot_batch,
                    &state.down_expanded,
                    hidden,
                    inter,
                    k_top,
                    1,
                )
                .map_err(|e| format!("minimax L{l}: down hfq4: {e:?}"))?;
                gpu.moe_down_combine_k8_batched(
                    &state.down_expanded,
                    &state.topk_weights,
                    &state.h,
                    hidden,
                    k_top,
                    1,
                )
                .map_err(|e| format!("minimax L{l}: combine: {e:?}"))?;
            }
            DType::MQ6G256 | DType::HFQ6G256 => {
                gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
                    &layer.expert_down_ptrs,
                    &state.topk_indices,
                    &state.rot_batch,
                    &state.down_expanded,
                    hidden,
                    inter,
                    k_top,
                    1,
                )
                .map_err(|e| format!("minimax L{l}: down hfq6: {e:?}"))?;
                gpu.moe_down_combine_k8_batched(
                    &state.down_expanded,
                    &state.topk_weights,
                    &state.h,
                    hidden,
                    k_top,
                    1,
                )
                .map_err(|e| format!("minimax L{l}: combine: {e:?}"))?;
            }
            DType::MQ2G256Lloyd => {
                // Fused down + weighted residual accumulate (no separate combine).
                gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
                    &layer.expert_down_ptrs,
                    &state.topk_indices,
                    &state.topk_weights,
                    &state.rot_batch,
                    &state.h,
                    hidden,
                    inter,
                    k_top,
                )
                .map_err(|e| format!("minimax L{l}: down mq2l: {e:?}"))?;
            }
            DType::MQ3G256Lloyd => {
                // Fused down + weighted residual accumulate (no separate combine).
                gpu.deepseek4_gemv_mq3g256_lloyd_moe_down_residual_scaled_indexed(
                    &layer.expert_down_ptrs,
                    &state.topk_indices,
                    &state.topk_weights,
                    &state.rot_batch,
                    &state.h,
                    hidden,
                    inter,
                    k_top,
                )
                .map_err(|e| format!("minimax L{l}: down mq3l: {e:?}"))?;
            }
            other => return Err(format!("minimax L{l}: unsupported expert dtype {other:?}")),
        }

        // Capture post-layer residual (pre final-norm) for the oracle compare.
        if !capture_postattn {
            if let Some(cap) = capture.as_deref_mut() {
                let h = gpu
                    .download_f32(&state.h)
                    .map_err(|e| format!("minimax L{l}: capture download: {e:?}"))?;
                cap[l].extend_from_slice(&h);
            }
        }
    }
    state.n_tokens = seq_len;

    // Final RMSNorm + lm_head (Q8 → plain).
    gpu.rmsnorm_f32(&state.h, &weights.final_norm, &state.final_norm_buf, eps)
        .map_err(|e| format!("minimax: final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
        .map_err(|e| format!("minimax: lm_head: {e}"))?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// #397 Ship 6 — forward-as-pipeline: MiniMax-M2 lowered decode (mechanical reuse).
//
// MiniMax is a standard MoE transformer — every layer is [Attend, Moe] (no conv,
// no dense, one variant), so it reuses the Attend + Moe super-ops with no new op
// kind. ADDITIVE + opt-in (HIPFIRE_FORWARD_LOWERED, default off until hipx
// byte-parity validated — minimax only fits on hipx). The hand loop in
// decode_step_body is untouched → default path byte-identical. The block fns
// mirror the hand-loop arms verbatim; the lowered handlers call them.
// ─────────────────────────────────────────────────────────────────────────

/// Attention block (attn-norm folded in). Mirrors the hand-loop attention arm.
fn minimax_attn_block(
    gpu: &mut Gpu, cfg: &MiniMaxConfig, layer: &MiniMaxLayerWeights, state: &MiniMaxState, l: usize,
) -> Result<(), String> {
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let eps = cfg.rms_norm_eps;
    gpu.rmsnorm_f32(&state.h, &layer.attn_norm, &state.tmp, eps)
        .map_err(|e| format!("minimax L{l}: attn rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &layer.wq, &state.tmp, &state.fa_q).map_err(|e| format!("minimax L{l}: q_proj: {e}"))?;
    weight_gemv(gpu, &layer.wk, &state.tmp, &state.fa_k).map_err(|e| format!("minimax L{l}: k_proj: {e}"))?;
    weight_gemv(gpu, &layer.wv, &state.tmp, &state.fa_v).map_err(|e| format!("minimax L{l}: v_proj: {e}"))?;
    if cfg.use_qk_norm {
        gpu.rmsnorm_batched(&state.fa_q, &layer.q_norm, &state.fa_q, 1, q_dim, eps)
            .map_err(|e| format!("minimax L{l}: q_norm: {e:?}"))?;
        gpu.rmsnorm_batched(&state.fa_k, &layer.k_norm, &state.fa_k, 1, kv_dim, eps)
            .map_err(|e| format!("minimax L{l}: k_norm: {e:?}"))?;
    }
    gpu.rope_partial_interleaved_f32(
        &state.fa_q, &state.fa_k, &state.pos_buf, cfg.num_attention_heads,
        cfg.num_key_value_heads, cfg.head_dim, cfg.rotary_dim, cfg.rope_theta,
    )
    .map_err(|e| format!("minimax L{l}: rope: {e:?}"))?;
    gpu.kv_cache_write_q8_0(&state.kv.k_gpu[l], &state.fa_k, &state.pos_buf, cfg.num_key_value_heads, cfg.head_dim)
        .map_err(|e| format!("minimax L{l}: kv write k: {e:?}"))?;
    gpu.kv_cache_write_q8_0(&state.kv.v_gpu[l], &state.fa_v, &state.pos_buf, cfg.num_key_value_heads, cfg.head_dim)
        .map_err(|e| format!("minimax L{l}: kv write v: {e:?}"))?;
    gpu.attention_q8_0_kv(
        &state.fa_q, &state.kv.k_gpu[l], &state.kv.v_gpu[l], &state.fa_attn_out, &state.pos_buf,
        state.max_seq, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, state.kv.physical_cap,
    )
    .map_err(|e| format!("minimax L{l}: attention: {e:?}"))?;
    weight_gemv_residual(gpu, &layer.wo, &state.fa_attn_out, &state.h)
        .map_err(|e| format!("minimax L{l}: o_proj: {e}"))
}

/// MoE block (ffn-norm folded in). Mirrors the hand-loop MoE arm (8-arm dtype dispatch).
#[allow(clippy::too_many_arguments)]
fn minimax_moe_block(
    gpu: &mut Gpu, cfg: &MiniMaxConfig, layer: &MiniMaxLayerWeights, state: &MiniMaxState, l: usize,
    // EP (Ship 6 substrate-EP): when `Some`, the routed combine/down accumulates
    // into this zeroed partial instead of `state.h` (the EP driver all-reduces it
    // and adds into each rank's `state.h`). MiniMax has NO shared expert, so the
    // entire MoE output is routed → the whole block redirects. `None` = normal.
    routed_out: Option<&GpuTensor>,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let n_exp = cfg.num_local_experts;
    let k_top = cfg.num_experts_per_tok;
    let eps = cfg.rms_norm_eps;
    let out_target: &GpuTensor = routed_out.unwrap_or(&state.h);
    gpu.rmsnorm_f32(&state.h, &layer.ffn_norm, &state.ffn_tmp, eps)
        .map_err(|e| format!("minimax L{l}: ffn rmsnorm: {e:?}"))?;
    rotate_x_mq_for(gpu, &layer.experts[0].gate_up, &state.ffn_tmp, &state.ffn_x_rot, hidden)
        .map_err(|e| format!("minimax L{l}: ffn rotate: {e:?}"))?;
    weight_gemv(gpu, &layer.router, &state.ffn_tmp, &state.router_logits)
        .map_err(|e| format!("minimax L{l}: router: {e}"))?;
    gpu.sigmoid_f32(&state.router_logits).map_err(|e| format!("minimax L{l}: sigmoid: {e:?}"))?;
    gpu.deepseek4_moe_topk_bias_aware_f32(
        &state.router_logits, &layer.routing_bias, &state.topk_indices, &state.topk_weights,
        n_exp as i32, k_top as i32, 1.0,
    )
    .map_err(|e| format!("minimax L{l}: topk: {e:?}"))?;
    let edt = layer.experts[0].gate_up.gpu_dtype;
    match edt {
        DType::MQ4G256 | DType::HFQ4G256 => gpu
            .gemv_hfq4g256_moe_gate_up_k8_indexed(
                &layer.expert_gate_up_ptrs, &state.topk_indices, &state.ffn_x_rot,
                &state.gate_batch, &state.up_batch, 2 * inter, hidden,
            )
            .map_err(|e| format!("minimax L{l}: gate_up hfq4: {e:?}"))?,
        DType::MQ6G256 | DType::HFQ6G256 => gpu
            .gemv_hfq6g256_moe_gate_up_k8_indexed(
                &layer.expert_gate_up_ptrs, &state.topk_indices, &state.ffn_x_rot,
                &state.gate_batch, &state.up_batch, 2 * inter, hidden,
            )
            .map_err(|e| format!("minimax L{l}: gate_up hfq6: {e:?}"))?,
        DType::MQ2G256Lloyd => gpu
            .deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
                &layer.expert_gate_up_ptrs, &state.topk_indices, &state.ffn_x_rot,
                &state.gate_batch, &state.up_batch, 2 * inter, hidden, k_top,
            )
            .map_err(|e| format!("minimax L{l}: gate_up mq2l: {e:?}"))?,
        DType::MQ3G256Lloyd => gpu
            .deepseek4_gemv_mq3g256_lloyd_moe_gate_up_indexed(
                &layer.expert_gate_up_ptrs, &state.topk_indices, &state.ffn_x_rot,
                &state.gate_batch, &state.up_batch, 2 * inter, hidden, k_top,
            )
            .map_err(|e| format!("minimax L{l}: gate_up mq3l: {e:?}"))?,
        other => return Err(format!("minimax L{l}: unsupported expert dtype {other:?}")),
    }
    fused_silu_mul_rotate_mq_batched_for(
        gpu, &layer.experts[0].down, &state.gate_batch, &state.up_batch, &state.rot_batch, inter, k_top,
    )
    .map_err(|e| format!("minimax L{l}: silu_mul_rotate: {e:?}"))?;
    let ddt = layer.experts[0].down.gpu_dtype;
    match ddt {
        DType::MQ4G256 | DType::HFQ4G256 => {
            gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                &layer.expert_down_ptrs, &state.topk_indices, &state.rot_batch, &state.down_expanded,
                hidden, inter, k_top, 1,
            )
            .map_err(|e| format!("minimax L{l}: down hfq4: {e:?}"))?;
            gpu.moe_down_combine_k8_batched(&state.down_expanded, &state.topk_weights, out_target, hidden, k_top, 1)
                .map_err(|e| format!("minimax L{l}: combine: {e:?}"))?;
        }
        DType::MQ6G256 | DType::HFQ6G256 => {
            gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
                &layer.expert_down_ptrs, &state.topk_indices, &state.rot_batch, &state.down_expanded,
                hidden, inter, k_top, 1,
            )
            .map_err(|e| format!("minimax L{l}: down hfq6: {e:?}"))?;
            gpu.moe_down_combine_k8_batched(&state.down_expanded, &state.topk_weights, out_target, hidden, k_top, 1)
                .map_err(|e| format!("minimax L{l}: combine: {e:?}"))?;
        }
        DType::MQ2G256Lloyd => {
            gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
                &layer.expert_down_ptrs, &state.topk_indices, &state.topk_weights, &state.rot_batch,
                out_target, hidden, inter, k_top,
            )
            .map_err(|e| format!("minimax L{l}: down mq2l: {e:?}"))?;
        }
        DType::MQ3G256Lloyd => {
            gpu.deepseek4_gemv_mq3g256_lloyd_moe_down_residual_scaled_indexed(
                &layer.expert_down_ptrs, &state.topk_indices, &state.topk_weights, &state.rot_batch,
                out_target, hidden, inter, k_top,
            )
            .map_err(|e| format!("minimax L{l}: down mq3l: {e:?}"))?;
        }
        other => return Err(format!("minimax L{l}: unsupported expert dtype {other:?}")),
    }
    Ok(())
}

/// Per-layer execution context for the lowered decode path (rebuilt each layer).
struct MinimaxBindings<'a> {
    cfg: &'a MiniMaxConfig,
    layer: &'a MiniMaxLayerWeights,
    state: &'a MiniMaxState,
    l: usize,
}

impl<'a> ForwardBindings for MinimaxBindings<'a> {
    fn run_attend(&mut self, gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), DispatchError> {
        minimax_attn_block(gpu, self.cfg, self.layer, self.state, self.l).map_err(DispatchError::Hip)
    }
    fn run_moe(&mut self, gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), DispatchError> {
        minimax_moe_block(gpu, self.cfg, self.layer, self.state, self.l, None).map_err(DispatchError::Hip)
    }
    fn run_moe_ep(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        routed_out: &GpuTensor,
        _skip_shared: bool,
    ) -> Result<(), DispatchError> {
        // MiniMax has no shared expert → the entire MoE output is routed, so the
        // whole block redirects into `routed_out` (zeroed by the EP executor);
        // `state.h` (the replicated attention residual) is added after all-reduce
        // via ep_add_into_residual. `skip_shared` is irrelevant (no shared expert).
        minimax_moe_block(gpu, self.cfg, self.layer, self.state, self.l, Some(routed_out))
            .map_err(DispatchError::Hip)
    }
    fn ep_add_into_residual(&mut self, gpu: &mut Gpu, partial: &GpuTensor) -> Result<(), DispatchError> {
        gpu.add_inplace_f32(&self.state.h, partial)
            .map_err(|e| DispatchError::Hip(e.to_string()))
    }
    fn run_proj(&mut self, _gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("minimax has no Proj super-op".into()))
    }
    fn run_residual_gemv(&mut self, _gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("minimax has no ResidualGemv super-op".into()))
    }
    fn run_norm(&mut self, _gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("minimax has no Norm super-op".into()))
    }
    fn run_conv(&mut self, _gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("minimax has no Conv super-op".into()))
    }
    fn run_recurrent(&mut self, _gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("minimax has no Recurrent super-op".into()))
    }
    fn run_escape(&mut self, _gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding, kind: superop::EscapeKind) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(format!("minimax has no Escape super-op ({kind:?})")))
    }
}

#[inline]
fn mm_superop(kind: SuperOpKind) -> SuperOp {
    SuperOp {
        kind,
        binding: OpBinding { key: None, weights: Vec::new(), scratch: Vec::new(), flavor: OpFlavor::None },
    }
}

/// MiniMax has ONE layer shape (all layers Attn+MoE) → the same 2-op program for
/// every layer. Pure → unit-testable.
fn minimax_lower_program() -> superop::LayerProgram {
    vec![mm_superop(SuperOpKind::Attend), mm_superop(SuperOpKind::Moe)]
}

/// Cached HIPFIRE_FORWARD_LOWERED toggle for minimax. #397 Ship 6: the minimax
/// lowered decode is **DEFAULT ON** as of 2026-06-07 — hipx/gfx1151 byte-parity
/// validated (lowered == hand token-text md5 2a46c35e… on the mq2-lloyd tier,
/// "Paris is the capital of France."). Escape hatch: `HIPFIRE_FORWARD_LOWERED=0`
/// forces the legacy hand loop (still present in decode_step_body).
fn minimax_forward_lowered_enabled() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var("HIPFIRE_FORWARD_LOWERED").ok().as_deref() != Some("0"))
}

/// Lowered (#397 Ship 6) per-layer decode loop + final norm/head. Pos scalar is
/// already staged by the caller (decode_step_body). Behaviorally equivalent to
/// the hand loop (validated via FORWARD_LOWERED=0-vs-=1 token-text md5 on hipx).
fn decode_step_body_lowered(
    cfg: &MiniMaxConfig,
    weights: &MiniMaxWeights,
    state: &mut MiniMaxState,
    gpu: &mut Gpu,
    position: u32,
) -> Result<(), String> {
    let eps = cfg.rms_norm_eps;
    let seq_len = position as usize + 1;
    let ctx = DispatchCtx::new(gpu);
    let program = minimax_lower_program();
    for (l, layer) in weights.layers.iter().enumerate() {
        let mut bind = MinimaxBindings { cfg, layer, state, l };
        superop::run_layer_program(gpu, &ctx, &program, &mut bind)
            .map_err(|e| format!("minimax L{l}: lowered run_layer_program: {e}"))?;
    }
    state.n_tokens = seq_len;
    gpu.rmsnorm_f32(&state.h, &weights.final_norm, &state.final_norm_buf, eps)
        .map_err(|e| format!("minimax: final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
        .map_err(|e| format!("minimax: lm_head: {e}"))
}

/// True iff every layer's expert gate_up + down dtypes have batched kernels, so
/// `forward_batch` won't `Err` partway through a pass. Pre-check this before
/// enabling batched prefill: unsupported tiers (MQ3-Lloyd, HFQ6-gate_up) then
/// cleanly take the sequential path instead of corrupting state on a mid-layer
/// `Err`. Mirrors the dtype match arms in `forward_batch`.
pub fn forward_batch_supported(weights: &MiniMaxWeights) -> bool {
    weights.layers.iter().all(|layer| {
        let gate_up_ok = matches!(
            layer.experts[0].gate_up.gpu_dtype,
            DType::MQ4G256 | DType::HFQ4G256 | DType::MQ2G256Lloyd
        );
        let down_ok = matches!(
            layer.experts[0].down.gpu_dtype,
            DType::MQ4G256
                | DType::HFQ4G256
                | DType::MQ6G256
                | DType::HFQ6G256
                | DType::MQ2G256Lloyd
        );
        gate_up_ok && down_ok
    })
}

/// Batched forward over `B` tokens in ONE pass — the spec-decode VERIFY forward
/// and fast-prefill keystone. Fills the KV cache for all B positions and returns
/// the LAST token's logits. Reads each weight matrix ONCE for all B tokens
/// (bandwidth-amortized — verifying B tokens costs ~1× the 6.2 GB/token weight
/// read, not B×), which is the basis of the 2-5× spec-decode / fast-TTFT win.
///
/// `tokens`: B token ids. `start_pos`: absolute position of `tokens[0]` (the KV
/// cache must already hold positions `[0, start_pos)`). `B` must be 1..=64
/// (`gemm_q8_0_batched` kernel cap); the caller chunks longer prompts.
///
/// Batched twin of `decode_step_body`: every op uses its batched kernel variant
/// (audited present in rdna-compute), dense Q8 projections go through
/// `gemm_q8_0_batched` directly (the `weight_gemm` helper falls back to per-row
/// GEMV for Q8). Per-row causal masking + the growing KV length are handled
/// inside `attention_q8_0_kv_batched` via the `positions[B]` array.
///
/// Supported expert dtypes (batched kernels that exist today): gate_up ∈
/// {HFQ4/MQ4, MQ2-Lloyd}; down ∈ {HFQ4/MQ4, HFQ6, MQ2-Lloyd}. HFQ6-gate_up and
/// MQ3-Lloyd have no batched kernel yet → Err (caller falls back to sequential).
#[allow(clippy::too_many_arguments)]
pub fn forward_batch(
    cfg: &MiniMaxConfig,
    weights: &MiniMaxWeights,
    state: &mut MiniMaxState,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: usize,
) -> Result<Vec<f32>, String> {
    let b = tokens.len();
    if b == 0 {
        return Err("minimax forward_batch: empty token slice".to_string());
    }
    if b > 64 {
        return Err(format!(
            "minimax forward_batch: B={b} exceeds kernel cap 64"
        ));
    }
    let hidden = cfg.hidden_size;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let inter = cfg.intermediate_size;
    let n_exp = cfg.num_local_experts;
    let k_top = cfg.num_experts_per_tok;
    let eps = cfg.rms_norm_eps;
    let max_ctx = start_pos + b; // largest seq_len across the B rows (geometry)
    let max_seq = state.kv.physical_cap; // KV cache stride

    // ── Batched scratch (allocated per call; prefill/verify is not the hot
    //    per-kernel inner loop, and this keeps MiniMaxState unchanged). ──
    let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
        g.alloc_tensor(&[n], DType::F32)
            .map_err(|e| format!("forward_batch alloc {label}: {e:?}"))
    };
    let x = alloc(gpu, b * hidden, "x")?;
    let tmp = alloc(gpu, b * hidden, "tmp")?;
    let fq = alloc(gpu, b * q_dim, "fq")?;
    let fk = alloc(gpu, b * kv_dim, "fk")?;
    let fv = alloc(gpu, b * kv_dim, "fv")?;
    let attn_out = alloc(gpu, b * q_dim, "attn_out")?;
    let o = alloc(gpu, b * hidden, "o")?;
    let ffn_tmp = alloc(gpu, b * hidden, "ffn_tmp")?;
    let ffn_x_rot = alloc(gpu, b * hidden, "ffn_x_rot")?;
    let router_logits = alloc(gpu, b * n_exp, "router_logits")?;
    let topk_idx = alloc(gpu, b * k_top, "topk_idx")?;
    let topk_w = alloc(gpu, b * k_top, "topk_w")?;
    let gate = alloc(gpu, b * k_top * inter, "gate")?;
    let up = alloc(gpu, b * k_top * inter, "up")?;
    let rot = alloc(gpu, b * k_top * inter, "rot")?;
    let down_exp = alloc(gpu, b * k_top * hidden, "down_exp")?;

    // positions [B] i32 (stored in an f32-sized buffer; kernels read it as i32).
    let pos_data: Vec<i32> = (0..b).map(|i| (start_pos + i) as i32).collect();
    let pos_bytes: Vec<u8> = pos_data.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let pos_array = alloc(gpu, b, "pos_array")?;
    gpu.hip
        .memcpy_htod(&pos_array.buf, &pos_bytes)
        .map_err(|e| format!("forward_batch htod pos: {e:?}"))?;

    // Embedding: per-token lookup into x[B, hidden] (token_id is a scalar arg).
    {
        let x_single = alloc(gpu, hidden, "x_single")?;
        for (i, &tok) in tokens.iter().enumerate() {
            gpu.embedding_lookup_q8(&weights.embed, &x_single, tok, hidden)
                .map_err(|e| format!("forward_batch embed lookup: {e:?}"))?;
            gpu.hip
                .memcpy_dtod_at(&x.buf, i * hidden * 4, &x_single.buf, 0, hidden * 4)
                .map_err(|e| format!("forward_batch embed copy: {e:?}"))?;
        }
        gpu.free_tensor(x_single).ok();
    }

    for (l, layer) in weights.layers.iter().enumerate() {
        // ── Attention (batched, per-row causal via positions) ──────────────
        gpu.rmsnorm_batched(&x, &layer.attn_norm, &tmp, b, hidden, eps)
            .map_err(|e| format!("minimax L{l} batch attn rmsnorm: {e:?}"))?;
        gpu.gemm_q8_0_batched(&layer.wq.buf, &tmp, &fq, q_dim, hidden, b)
            .map_err(|e| format!("minimax L{l} batch q_proj: {e:?}"))?;
        gpu.gemm_q8_0_batched(&layer.wk.buf, &tmp, &fk, kv_dim, hidden, b)
            .map_err(|e| format!("minimax L{l} batch k_proj: {e:?}"))?;
        gpu.gemm_q8_0_batched(&layer.wv.buf, &tmp, &fv, kv_dim, hidden, b)
            .map_err(|e| format!("minimax L{l} batch v_proj: {e:?}"))?;
        if cfg.use_qk_norm {
            // Per-row RMSNorm over the full flat q/k vector (MiniMax convention).
            gpu.rmsnorm_batched(&fq, &layer.q_norm, &fq, b, q_dim, eps)
                .map_err(|e| format!("minimax L{l} batch q_norm: {e:?}"))?;
            gpu.rmsnorm_batched(&fk, &layer.k_norm, &fk, b, kv_dim, eps)
                .map_err(|e| format!("minimax L{l} batch k_norm: {e:?}"))?;
        }
        gpu.rope_partial_interleaved_f32_batched(
            &fq,
            &fk,
            &pos_array,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.rotary_dim,
            cfg.rope_theta,
            b,
            // pos_offset (API drift on integration): added to positions[b] for
            // the RoPE angle only. MiniMax prefill does no KV compaction, so 0
            // is the behavior-preserving no-op offset.
            0,
        )
        .map_err(|e| format!("minimax L{l} batch rope: {e:?}"))?;
        gpu.kv_cache_write_q8_0_batched(
            &state.kv.k_gpu[l],
            &fk,
            &pos_array,
            cfg.num_key_value_heads,
            cfg.head_dim,
            b,
        )
        .map_err(|e| format!("minimax L{l} batch kv write k: {e:?}"))?;
        gpu.kv_cache_write_q8_0_batched(
            &state.kv.v_gpu[l],
            &fv,
            &pos_array,
            cfg.num_key_value_heads,
            cfg.head_dim,
            b,
        )
        .map_err(|e| format!("minimax L{l} batch kv write v: {e:?}"))?;
        gpu.attention_q8_0_kv_batched(
            &fq,
            &state.kv.k_gpu[l],
            &state.kv.v_gpu[l],
            &attn_out,
            &pos_array,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            max_seq,
            max_ctx,
            b,
        )
        .map_err(|e| format!("minimax L{l} batch attention: {e:?}"))?;
        gpu.gemm_q8_0_batched(&layer.wo.buf, &attn_out, &o, hidden, q_dim, b)
            .map_err(|e| format!("minimax L{l} batch o_proj: {e:?}"))?;
        gpu.add_inplace_f32(&x, &o)
            .map_err(|e| format!("minimax L{l} batch o residual: {e:?}"))?;

        // ── MoE (batched; no shared expert) ────────────────────────────────
        gpu.rmsnorm_batched(&x, &layer.ffn_norm, &ffn_tmp, b, hidden, eps)
            .map_err(|e| format!("minimax L{l} batch ffn rmsnorm: {e:?}"))?;
        // AWQ-aware FWHT rotate (gate_up may carry an AWQ activation scale —
        // MQ2-Lloyd+AWQ); the raw rotate_x_mq_batched would drop it.
        rotate_x_mq_batched_for(
            gpu,
            &layer.experts[0].gate_up,
            &ffn_tmp,
            &ffn_x_rot,
            hidden,
            b,
        )
        .map_err(|e| format!("minimax L{l} batch ffn rotate: {e}"))?;
        gpu.gemm_q8_0_batched(
            &layer.router.buf,
            &ffn_tmp,
            &router_logits,
            n_exp,
            hidden,
            b,
        )
        .map_err(|e| format!("minimax L{l} batch router: {e:?}"))?;
        gpu.sigmoid_f32(&router_logits)
            .map_err(|e| format!("minimax L{l} batch sigmoid: {e:?}"))?;
        gpu.deepseek4_moe_topk_bias_aware_batched_f32(
            &router_logits,
            &layer.routing_bias,
            &topk_idx,
            &topk_w,
            n_exp as i32,
            k_top as i32,
            1.0,
            b as i32,
        )
        .map_err(|e| format!("minimax L{l} batch topk: {e:?}"))?;

        let edt = layer.experts[0].gate_up.gpu_dtype;
        match edt {
            DType::MQ4G256 | DType::HFQ4G256 => gpu
                .gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
                    &layer.expert_gate_up_ptrs,
                    &topk_idx,
                    &ffn_x_rot,
                    &gate,
                    &up,
                    2 * inter,
                    hidden,
                    k_top,
                    b,
                )
                .map_err(|e| format!("minimax L{l} batch gate_up hfq4: {e:?}"))?,
            DType::MQ2G256Lloyd => gpu
                .deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4(
                    &layer.expert_gate_up_ptrs,
                    &topk_idx,
                    &ffn_x_rot,
                    &gate,
                    &up,
                    2 * inter,
                    hidden,
                    k_top,
                    b,
                )
                .map_err(|e| format!("minimax L{l} batch gate_up mq2l: {e:?}"))?,
            other => {
                return Err(format!(
                    "minimax L{l} forward_batch: gate_up dtype {other:?} has no batched kernel yet"
                ))
            }
        }

        // AWQ-aware silu·mul·rotate (down weight; b*k_top expert streams).
        fused_silu_mul_rotate_mq_batched_for(
            gpu,
            &layer.experts[0].down,
            &gate,
            &up,
            &rot,
            inter,
            b * k_top,
        )
        .map_err(|e| format!("minimax L{l} batch silu_mul_rotate: {e}"))?;

        let ddt = layer.experts[0].down.gpu_dtype;
        match ddt {
            DType::MQ4G256 | DType::HFQ4G256 => {
                gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                    &layer.expert_down_ptrs,
                    &topk_idx,
                    &rot,
                    &down_exp,
                    hidden,
                    inter,
                    k_top,
                    b,
                )
                .map_err(|e| format!("minimax L{l} batch down hfq4: {e:?}"))?;
                gpu.moe_down_combine_k8_batched(&down_exp, &topk_w, &x, hidden, k_top, b)
                    .map_err(|e| format!("minimax L{l} batch combine: {e:?}"))?;
            }
            DType::MQ6G256 | DType::HFQ6G256 => {
                gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
                    &layer.expert_down_ptrs,
                    &topk_idx,
                    &rot,
                    &down_exp,
                    hidden,
                    inter,
                    k_top,
                    b,
                )
                .map_err(|e| format!("minimax L{l} batch down hfq6: {e:?}"))?;
                gpu.moe_down_combine_k8_batched(&down_exp, &topk_w, &x, hidden, k_top, b)
                    .map_err(|e| format!("minimax L{l} batch combine: {e:?}"))?;
            }
            DType::MQ2G256Lloyd => gpu
                .deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed_batched_k4(
                    &layer.expert_down_ptrs,
                    &topk_idx,
                    &topk_w,
                    &rot,
                    &x,
                    hidden,
                    inter,
                    k_top,
                    b,
                )
                .map_err(|e| format!("minimax L{l} batch down mq2l: {e:?}"))?,
            other => {
                return Err(format!(
                    "minimax L{l} forward_batch: down dtype {other:?} has no batched kernel yet"
                ))
            }
        }
    }
    state.n_tokens = start_pos + b;

    // ── Final RMSNorm + lm_head on the LAST row only (verify/prefill need the
    //    last position's logits; per-position logits = a future all-rows head). ──
    let x_last = alloc(gpu, hidden, "x_last")?;
    gpu.hip
        .memcpy_dtod_at(&x_last.buf, 0, &x.buf, (b - 1) * hidden * 4, hidden * 4)
        .map_err(|e| format!("forward_batch last copy: {e:?}"))?;
    gpu.rmsnorm_f32(&x_last, &weights.final_norm, &state.final_norm_buf, eps)
        .map_err(|e| format!("minimax batch final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
        .map_err(|e| format!("minimax batch lm_head: {e}"))?;
    let logits = gpu
        .download_f32(&state.logits)
        .map_err(|e| format!("forward_batch download logits: {e:?}"))?;

    for t in [
        x,
        tmp,
        fq,
        fk,
        fv,
        attn_out,
        o,
        ffn_tmp,
        ffn_x_rot,
        router_logits,
        topk_idx,
        topk_w,
        gate,
        up,
        rot,
        down_exp,
        pos_array,
        x_last,
    ] {
        gpu.free_tensor(t).ok();
    }
    Ok(logits)
}

// ───────────────────────── Ship 6 substrate-EP (MiniMax) ─────────────────────
//
// Mirror of the qwen35 EP wiring. MiniMax packs all experts into ONE blob per
// projection (too big to load-then-free on a 32 GB card), so sharding is done at
// LOAD time: `MiniMaxWeights::load(.., Some((shard, rank)))` uploads only the
// rank-owned experts (non-owned → zeroed gate_up dummy). MiniMax has NO shared
// expert, so the entire MoE output is routed → the whole MoE block redirects
// into the per-rank partial. Attention (Q8 KV) is replicated; only the MoE
// routed sum crosses ranks (peer-direct all-reduce).

/// EP (Ship 6 substrate-EP) replicated N-rank decode forward for ONE token.
/// Mirror of qwen35::forward_ep: every rank holds full replicated weights /
/// state / KV EXCEPT MoE experts (sharded at load). Embeds + stages pos per
/// rank, runs each layer's 2-op program (Attend replicated, Moe all-reduce-EP'd)
/// via [`hipfire_runtime::ep::run_layer_program_ep`], then final norm + lm_head
/// on rank 0 → `state_per_rank[0].logits`. Every device must have an
/// `active_stream` ([`hipfire_runtime::ep::ensure_rank_streams`]); peer access
/// enabled for the fast peer-direct all-reduce.
#[allow(clippy::too_many_arguments)]
pub fn forward_ep(
    gpus: &mut hipfire_runtime::multi_gpu::Gpus,
    weights_per_rank: &[MiniMaxWeights],
    cfg: &MiniMaxConfig,
    state_per_rank: &mut [MiniMaxState],
    partials: &[GpuTensor],
    token: u32,
    position: u32,
) -> Result<(), String> {
    let n = gpus.devices.len();
    assert_eq!(weights_per_rank.len(), n, "forward_ep: weights_per_rank len");
    assert_eq!(state_per_rank.len(), n, "forward_ep: state_per_rank len");
    assert_eq!(partials.len(), n, "forward_ep: partials len");
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;

    // 1. Embed + stage pos per rank (replicated, deterministic).
    for r in 0..n {
        gpus.devices[r].bind_thread().map_err(|e| format!("forward_ep bind {r}: {e:?}"))?;
        gpus.devices[r]
            .embedding_lookup_q8(&weights_per_rank[r].embed, &state_per_rank[r].h, token, hidden)
            .map_err(|e| format!("forward_ep embed {r}: {e:?}"))?;
        state_per_rank[r].pos_host[0] = position as i32;
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(state_per_rank[r].pos_host.as_ptr() as *const u8, 4) };
        gpus.devices[r]
            .memcpy_htod_auto(&state_per_rank[r].pos_buf, pos_bytes)
            .map_err(|e| format!("forward_ep pos {r}: {e:?}"))?;
    }

    // 2. Per-layer EP program (Attend replicated; Moe all-reduce-EP'd).
    let timing = std::env::var("HIPFIRE_EP_DECODE_TIMING").is_ok();
    let t_layers = std::time::Instant::now();
    let program = minimax_lower_program();
    let n_layers = weights_per_rank[0].layers.len();
    for l in 0..n_layers {
        let mut binds: Vec<MinimaxBindings> = Vec::with_capacity(n);
        for r in 0..n {
            binds.push(MinimaxBindings {
                cfg,
                layer: &weights_per_rank[r].layers[l],
                state: &state_per_rank[r],
                l,
            });
        }
        hipfire_runtime::ep::run_layer_program_ep(gpus, binds.as_mut_slice(), partials, &program, hidden)
            .map_err(|e| format!("forward_ep run_layer_program_ep L{l}: {e}"))?;
    }

    // 3. Final norm + lm_head on rank 0 → state_per_rank[0].logits.
    {
        gpus.devices[0].bind_thread().map_err(|e| format!("forward_ep bind0: {e:?}"))?;
        let w = &weights_per_rank[0];
        let s = &state_per_rank[0];
        let gpu = &mut gpus.devices[0];
        gpu.rmsnorm_f32(&s.h, &w.final_norm, &s.final_norm_buf, eps)
            .map_err(|e| format!("forward_ep final norm: {e:?}"))?;
        weight_gemv(gpu, &w.lm_head, &s.final_norm_buf, &s.logits)
            .map_err(|e| format!("forward_ep lm_head: {e}"))?;
    }

    let layers_ms = t_layers.elapsed().as_secs_f64() * 1000.0;
    // 4. Sync every rank (work ran on active_streams; host logits read races otherwise).
    let t_sync = std::time::Instant::now();
    for r in 0..n {
        gpus.devices[r].bind_thread().map_err(|e| format!("forward_ep sync bind {r}: {e:?}"))?;
        gpus.devices[r].hip.device_synchronize().map_err(|e| format!("forward_ep sync {r}: {e:?}"))?;
    }
    if timing {
        // layers_ms = host enqueue + any blocking (RCCL/backpressure); sync_ms =
        // GPU drain remaining at the barrier. host-launch-bound ⇒ layers_ms is
        // the bulk and sync_ms is small; GPU-bound ⇒ sync_ms is the bulk.
        eprintln!(
            "EP-DECODE-TIMING: layers(host)={layers_ms:.2} ms  final-sync(gpu)={:.2} ms",
            t_sync.elapsed().as_secs_f64() * 1000.0,
        );
    }
    for s in state_per_rank.iter_mut() {
        s.n_tokens = position as usize + 1;
    }
    Ok(())
}

#[cfg(test)]
mod ship6_lower_tests {
    use super::*;
    use superop::SuperOpKind::{Attend, Moe};

    // #397 Ship 6 — minimax is one variant (every layer Attn+MoE).
    #[test]
    fn minimax_program_is_attend_then_moe() {
        let kinds: Vec<_> = minimax_lower_program().iter().map(|o| o.kind).collect();
        assert_eq!(kinds, vec![Attend, Moe]);
    }
}
