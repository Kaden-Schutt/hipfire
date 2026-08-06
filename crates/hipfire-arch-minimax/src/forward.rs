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

use crate::minimax::{
    minimax_single_moe_authority, MiniMaxConfig, MiniMaxLayerWeights, MiniMaxState, MiniMaxWeights,
};
use hipfire_runtime::llama::{
    fused_silu_mul_rotate_mq_batched_for, rotate_x_mq_batched_for, rotate_x_mq_for, weight_gemv,
    weight_gemv_residual,
};
use hipfire_runtime::moe_plan::{
    execute_lowered_moe, lower_moe_steps, MoEExecutionKind, MoEExecutionPolicy, MoeExecutionTarget,
    MoeProgramParts, RoutedMoeStepPhases,
};
use hipfire_runtime::weight_manifest::ExpertGroupPlan;
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

    // Single-rank MoE authority: the common accessor shared with
    // `forward_batch` — the stable canonical single policy (ONE object per
    // model — never a per-token reconstruction: a fresh
    // `MoEExecutionPolicy::single()` carries a fresh mesh epoch, which the
    // exact-policy cache key would reject) plus the model-owned cached
    // expert-manifest resolution bound to it. The cache resolves ONCE through
    // the shared policy-aware resolver (authoritative `MiniMaxM2::weight_manifest`
    // + policy-aware `expert_group_manifest` specs); every later token
    // re-checks the exact key and borrows the identical cached plans by layer
    // — no per-token allocation, no local plan fabrication.
    let authority = minimax_single_moe_authority(weights, cfg)?;

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

        // ── MoE block (no shared expert) — lowered via the runtime lowerer ──
        // Admission pin (shared with forward_batch): the layer-`l` plan must
        // exist, be layer-scoped, and be Single-admitted before the MoE
        // kernels of this layer run; the lowered program then uses that
        // validated plan.
        let plan = authority.plan_for_layer(l)?;
        // ffn_tmp = rmsnorm(h) (plain, feeds the Q8 router); ffn_x_rot =
        // FWHT(ffn_tmp) (feeds the FWHT-pre-rotated experts). Input prep and
        // the router GEMV stay direct: they live OUTSIDE the routed program
        // (no Step twin for fused rmsnorm+rotate / the router projection).
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
        weight_gemv(gpu, &layer.router, &state.ffn_tmp, &state.router_logits)
            .map_err(|e| format!("minimax L{l}: router: {e}"))?;

        // Routed program (sigmoid → bias-aware top-k → gate_up →
        // silu·mul·rotate → down [+ combine]): built per layer from the shared
        // Step building blocks (ScoreActivation / MoeRoute / IndexedMoeGemv /
        // MoeActivation / MoeCombine) and executed through the runtime
        // lowerer's sealed Single executor. The launch schedule (zeroing /
        // collective placement) is derived exclusively from the concrete
        // borrowed steps plus the canonical single policy — the dtype
        // dispatch that used to live here is now the parts builder's down
        // classification (expanded+combine vs residual self-combine).
        let gu_ref = hipfire_dispatch::families::moe::MoeExpertRef {
            gate_up_ptrs: &layer.expert_gate_up_ptrs,
            down_ptrs: &layer.expert_down_ptrs,
            dummy_gate_up: layer.dummy_gate_up.as_ref(),
            dtype: layer.experts[0].gate_up.gpu_dtype,
            n_experts: n_exp,
            expert_m: inter,
            expert_k: hidden,
            owned: &[],
        };
        let down_ref = hipfire_dispatch::families::moe::MoeExpertRef {
            gate_up_ptrs: &layer.expert_gate_up_ptrs,
            down_ptrs: &layer.expert_down_ptrs,
            dummy_gate_up: layer.dummy_gate_up.as_ref(),
            dtype: layer.experts[0].down.gpu_dtype,
            n_experts: n_exp,
            expert_m: inter,
            expert_k: hidden,
            owned: &[],
        };
        let inputs = MinimaxMoeInputs {
            scores: &state.router_logits,
            gate_bias: &layer.routing_bias,
            topk_indices: &state.topk_indices,
            topk_weights: &state.topk_weights,
            x_rot: &state.ffn_x_rot,
            gate_batch: &state.gate_batch,
            up_batch: &state.up_batch,
            rot_batch: &state.rot_batch,
            down_expanded: &state.down_expanded,
            partial: &state.h,
            // The single-rank axis never admits the int64 down (the lowerer
            // rejects I64OnNonAdmittedAxis), so the single path is always f32.
            partial_i64: None,
            gu_ref: &gu_ref,
            down_ref: &down_ref,
            awq_scale: layer.experts[0].down.awq_scale.as_ref(),
        };
        let parts = minimax_moe_program(&inputs, k_top, n_exp, inter, hidden, false);
        let program = lower_moe_steps(plan, authority.policy(), parts)
            .map_err(|e| format!("minimax L{l}: lower_moe_steps: {e}"))?;
        let ctx = hipfire_dispatch::context::DispatchCtx::new(gpu);
        execute_lowered_moe(&program, MoeExecutionTarget::Single { gpu, ctx: &ctx })
            .map_err(|e| format!("minimax L{l}: execute_lowered_moe: {e:?}"))?;

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
// Phase 3 · Task 7 — MoE program parts + lowered execution (adapter core).
//
// The bespoke per-layer MoE sequencing (the old decode_step_body MoE arm /
// minimax_ep_moe_step step lists) is replaced by a single shared parts
// builder over the dispatch crate's Step building blocks (ScoreActivation /
// MoeRoute / IndexedMoeGemv / MoeActivation / MoeCombine / ConvertI64ToF32),
// lowered by the runtime lowerer and executed by the sealed executor. The
// launch schedule (zeroing, i64-vs-f32 collective placement) is derived
// exclusively from the concrete borrowed steps plus the caller-owned
// `MoEExecutionPolicy` — never from a locally reconstructed mesh. Family
// scaling (the FWHT input rotation, the router projection, rmsnorm) lives
// OUTSIDE the routed program as direct kernels; route_scale = 1.0 (MiniMax
// applies no routed-scaling factor) is captured inside the MoeRoute step.
// ─────────────────────────────────────────────────────────────────────────

/// Attention block (attn-norm folded in). Mirrors the hand-loop attention arm.
fn minimax_attn_block(
    gpu: &mut Gpu,
    cfg: &MiniMaxConfig,
    layer: &MiniMaxLayerWeights,
    state: &MiniMaxState,
    l: usize,
) -> Result<(), String> {
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let eps = cfg.rms_norm_eps;
    gpu.rmsnorm_f32(&state.h, &layer.attn_norm, &state.tmp, eps)
        .map_err(|e| format!("minimax L{l}: attn rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &layer.wq, &state.tmp, &state.fa_q)
        .map_err(|e| format!("minimax L{l}: q_proj: {e}"))?;
    weight_gemv(gpu, &layer.wk, &state.tmp, &state.fa_k)
        .map_err(|e| format!("minimax L{l}: k_proj: {e}"))?;
    weight_gemv(gpu, &layer.wv, &state.tmp, &state.fa_v)
        .map_err(|e| format!("minimax L{l}: v_proj: {e}"))?;
    if cfg.use_qk_norm {
        gpu.rmsnorm_batched(&state.fa_q, &layer.q_norm, &state.fa_q, 1, q_dim, eps)
            .map_err(|e| format!("minimax L{l}: q_norm: {e:?}"))?;
        gpu.rmsnorm_batched(&state.fa_k, &layer.k_norm, &state.fa_k, 1, kv_dim, eps)
            .map_err(|e| format!("minimax L{l}: k_norm: {e:?}"))?;
    }
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
    // KV write (Q8) + attention via the shared KV-usage abstraction. minimax is
    // Q8 non-flash unconditional → derive returns AttnQ8_0Kv at pos+1<=15000
    // (no partials → flash_partials: None; >15k flips to flash — documented
    // Q8-fidelity edge). capture_mode NOT threaded (non-flash kernel is
    // capture-safe; minimax captures it under HIPFIRE_MINIMAX_GRAPH).
    // NOTE: the hand path passed `state.max_seq` as the kernel's seq_len_hint
    // (LDS-sizing only; the kernel reads the true length from pos_buf), while
    // the dispatch arm passes pos+1. The attended-position count comes from
    // pos_buf either way, so output is unchanged — VALIDATED by hand≡lowered A/B.
    let pos = state.pos_host[0] as usize;
    let ctx = hipfire_dispatch::context::DispatchCtx::new(gpu);
    let plan = hipfire_dispatch::families::kv_tier::KvTierPlan::derive(
        hipfire_dispatch::families::kv_tier::KvTierInputs {
            pos,
            ..state.kv.tier_inputs()
        },
    )
    .map_err(|e| format!("minimax L{l}: kv tier: {e}"))?;
    let io = hipfire_dispatch::families::attention::AttnParams {
        q: &state.fa_q,
        k: &state.fa_k,
        v: &state.fa_v,
        k_cache: &state.kv.k_gpu[l],
        v_cache: &state.kv.v_gpu[l],
        k_scales: None,
        v_scales: None,
        pos_buf: &state.pos_buf,
        pos,
        positions: None,
        n_heads: cfg.num_attention_heads,
        n_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        physical_cap: state.kv.physical_cap,
        batch_size: 1,
        max_ctx_len: 0,
        flash_partials: None,
        givens_cos: None,
        givens_sin: None,
        tree_bias: None,
        block_start: 0,
        block_cols: 0,
        output: &state.fa_attn_out,
    };
    hipfire_dispatch::pipeline::execute_steps(
        gpu,
        &ctx,
        &[hipfire_dispatch::pipeline::Step::Attend { plan, io }],
    )
    .map_err(|e| format!("minimax L{l}: attention: {e:?}"))?;
    weight_gemv_residual(gpu, &layer.wo, &state.fa_attn_out, &state.h)
        .map_err(|e| format!("minimax L{l}: o_proj: {e}"))
}

/// Borrowed per-rank inputs of one layer's routed MoE program. Production
/// callers fill these from `MiniMaxState` + `MiniMaxLayerWeights`; the
/// no-GPU tests fill them from synthetic tensors. `partial` is the EP/TP
/// partial (or `state.h` on the single path — the combine/down accumulate
/// into it); `partial_i64` is the per-rank int64 scratch, required only when
/// `use_i64_down` selects the reproducible i64 down.
pub(crate) struct MinimaxMoeInputs<'a> {
    pub scores: &'a GpuTensor,
    pub gate_bias: &'a GpuTensor,
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub x_rot: &'a GpuTensor,
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
    pub down_expanded: &'a GpuTensor,
    pub partial: &'a GpuTensor,
    pub partial_i64: Option<&'a GpuTensor>,
    pub gu_ref: &'a hipfire_dispatch::families::moe::MoeExpertRef<'a>,
    pub down_ref: &'a hipfire_dispatch::families::moe::MoeExpertRef<'a>,
    pub awq_scale: Option<&'a GpuTensor>,
}

/// The shared typed router identity of the MiniMax routed program:
/// sigmoid scores (MiniMax's scoring_func), renormalized, route_scale = 1.0
/// (MiniMax applies no routed-scaling factor). The lowerer validates this
/// identity against the group's manifest-declared `sigmoid_topk`.
pub(crate) fn minimax_moe_router_plan<'a>(
    scores: &'a GpuTensor,
    topk_indices: &'a GpuTensor,
    topk_weights: &'a GpuTensor,
    k_top: usize,
) -> hipfire_dispatch::families::moe::RouterPlan<'a> {
    hipfire_dispatch::families::moe::RouterPlan::SigmoidTopK {
        scores,
        topk_indices,
        topk_weights,
        k_top,
        normalize: true,
        route_scale: 1.0,
    }
}

/// The canonical execution identity of the MiniMax routed program: the
/// indexed per-expert GEMV family (`IndexedMoeGemv`), matching the manifest's
/// declared `indexed_quantized` identity.
pub(crate) fn minimax_moe_execution_plan() -> hipfire_dispatch::families::moe::ExpertExecutionPlan {
    hipfire_dispatch::families::moe::ExpertExecutionPlan::IndexedQuantized
}

/// Build ONE rank's phased routed-MoE Step program from the shared Step
/// building blocks. Phase partition (router / gate_up / activation / down /
/// combine / finish) is the lowerer's canonical contract:
///
/// - **router**: sigmoid `ScoreActivation` feeding the bias-aware `MoeRoute`
///   (the layer's `e_score_correction_bias` survives as `gate_bias`).
/// - **gate_up**: one `IndexedMoeGemv` `GateUp` over the FWHT-rotated input.
/// - **activation**: one `MoeActivation` `MinimaxFused` (carrying the down
///   weight's AWQ activation scale).
/// - **down**: exactly one protocol-bearing down projection, keyed on the
///   DOWN ref's dtype:
///   - Lloyd (MQ2G256Lloyd / MQ3G256Lloyd): `DownResidual` (f32 self-combine,
///     no combine phase) or, when `use_i64_down` AND MQ3G256Lloyd,
///     `DownResidualI64` (reproducible int64 accumulator).
///   - Non-Lloyd (MQ4/HFQ4/MQ6/HFQ6): `DownExpanded` writing per-expert
///     outputs; the combine phase then REQUIRES exactly one `MoeCombine`
///     (no inverse permutation — decode indexed path).
/// - **finish**: exactly one `ConvertI64ToF32` on the i64 path (after the
///   collective-bearing down step); empty otherwise.
///
/// The EP-shard dummy gate_up buffer survives on BOTH expert refs, so
/// non-owned experts contribute exactly zero on every path.
pub(crate) fn minimax_moe_rank_phases<'a>(
    inputs: &MinimaxMoeInputs<'a>,
    k_top: usize,
    n_experts: usize,
    inter: usize,
    hidden: usize,
    use_i64_down: bool,
) -> RoutedMoeStepPhases<'a> {
    use hipfire_dispatch::pipeline::{
        GemvInput, MoeActivationVariant, MoeProj, ScoreActKind, Step,
    };
    let ddt = inputs.down_ref.dtype;
    let is_lloyd = matches!(ddt, DType::MQ2G256Lloyd | DType::MQ3G256Lloyd);
    let use_i64 = use_i64_down && matches!(ddt, DType::MQ3G256Lloyd);
    let down_i64 = use_i64.then(|| {
        inputs
            .partial_i64
            .expect("minimax i64 down requires the per-rank i64 partial")
    });
    let (down, combine, finish) = if use_i64 {
        (
            vec![Step::IndexedMoeGemv {
                experts: inputs.down_ref,
                which: MoeProj::DownResidualI64 {
                    topk_weights: inputs.topk_weights,
                },
                topk_indices: inputs.topk_indices,
                input: GemvInput::Prerotated(inputs.rot_batch),
                out: down_i64.expect("minimax i64 down requires the per-rank i64 partial"),
                k_top,
                batch_size: 1,
            }],
            Vec::new(),
            vec![Step::ConvertI64ToF32 {
                src: down_i64.expect("minimax i64 down requires the per-rank i64 partial"),
                dst: inputs.partial,
                n: hidden,
            }],
        )
    } else if is_lloyd {
        (
            vec![Step::IndexedMoeGemv {
                experts: inputs.down_ref,
                which: MoeProj::DownResidual {
                    topk_weights: inputs.topk_weights,
                },
                topk_indices: inputs.topk_indices,
                input: GemvInput::Prerotated(inputs.rot_batch),
                out: inputs.partial,
                k_top,
                batch_size: 1,
            }],
            Vec::new(),
            Vec::new(),
        )
    } else {
        (
            vec![Step::IndexedMoeGemv {
                experts: inputs.down_ref,
                which: MoeProj::DownExpanded,
                topk_indices: inputs.topk_indices,
                input: GemvInput::Prerotated(inputs.rot_batch),
                out: inputs.down_expanded,
                k_top,
                batch_size: 1,
            }],
            vec![Step::MoeCombine {
                down_out: inputs.down_expanded,
                topk_weights: inputs.topk_weights,
                out: inputs.partial,
                k: k_top,
                hidden,
                batch_size: 1,
                inverse_perm: None,
            }],
            Vec::new(),
        )
    };
    RoutedMoeStepPhases {
        router: vec![
            Step::ScoreActivation {
                scores: inputs.scores,
                kind: ScoreActKind::Sigmoid,
            },
            Step::MoeRoute {
                scores: inputs.scores,
                gate_bias: inputs.gate_bias,
                topk_indices: inputs.topk_indices,
                topk_weights: inputs.topk_weights,
                k: k_top,
                n_experts,
                route_scale: 1.0,
            },
        ],
        gate_up: vec![Step::IndexedMoeGemv {
            experts: inputs.gu_ref,
            which: MoeProj::GateUp {
                up_out: inputs.up_batch,
            },
            topk_indices: inputs.topk_indices,
            input: GemvInput::Prerotated(inputs.x_rot),
            out: inputs.gate_batch,
            k_top,
            batch_size: 1,
        }],
        activation: vec![Step::MoeActivation {
            variant: MoeActivationVariant::MinimaxFused {
                awq_scale: inputs.awq_scale,
            },
            gate: inputs.gate_batch,
            up: inputs.up_batch,
            rot_out: inputs.rot_batch,
            inter,
            k_top,
        }],
        down,
        combine,
        finish,
    }
}

/// Assemble the single-rank `MoeProgramParts` (shared typed router/execution
/// identity + one rank's phased program). The multi-rank paths build the
/// shared identity once and one rank program per rank.
pub(crate) fn minimax_moe_program<'a>(
    inputs: &MinimaxMoeInputs<'a>,
    k_top: usize,
    n_experts: usize,
    inter: usize,
    hidden: usize,
    use_i64_down: bool,
) -> MoeProgramParts<'a> {
    MoeProgramParts {
        router: minimax_moe_router_plan(
            inputs.scores,
            inputs.topk_indices,
            inputs.topk_weights,
            k_top,
        ),
        execution: minimax_moe_execution_plan(),
        ranks: vec![minimax_moe_rank_phases(
            inputs,
            k_top,
            n_experts,
            inter,
            hidden,
            use_i64_down,
        )],
    }
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

    // ── Single MoE authority admission — BEFORE any GPU allocation/launch ──
    // The common Single-authority accessor (shared with `decode_step_body`):
    // the stable canonical single policy + the model-owned cached
    // expert-manifest resolution. A stale-config / different-policy / failed
    // resolution is refused HERE, before any scratch is allocated or any
    // kernel launched. The batched path previously bypassed the manifest
    // authority entirely; the per-layer admission pin inside the loop gates
    // the direct established batched MoE dispatch.
    let authority = minimax_single_moe_authority(weights, cfg)?;

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
        // Admission pin (shared with `decode_step_body`): the layer-`l` plan
        // must exist, be layer-scoped, and be Single-admitted before any
        // kernel of this layer runs — in particular before the direct
        // established batched MoE dispatch below. Authority/admission only:
        // the batched kernels themselves are unchanged.
        authority.plan_for_layer(l)?;
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

/// Lowered EP/TP MoE for ONE layer — the sole parallel path in `forward_ep`
/// / `forward_tp`.
///
/// - **Phase 1** (pre-down): rmsnorm → input FWHT → router GEMV. These have
///   no bit-identical Step twin (fused rmsnorm+rotate / router GEMV), so they
///   stay direct arch kernels, producing `ffn_x_rot` / `router_logits` per
///   rank. The routed program (sigmoid → bias-aware top-K → gate_up →
///   silu·mul·rotate → down [+ combine / i64 convert]) is deferred to Phase 2.
/// - **Phase 2** (routed program): the per-rank `MoeProgramParts` — built by
///   the shared parts builder over the dispatch `Step` building blocks — are
///   lowered by the runtime lowerer (`lower_moe_steps`) and executed by the
///   sealed executor (`execute_lowered_moe`). The launch schedule (zeroing,
///   i64-vs-f32 collective placement) is derived exclusively from the concrete
///   borrowed steps plus the caller-owned policy and group plan:
///   - **Lloyd down** (MQ2G256Lloyd / MQ3G256Lloyd): a `DownResidual` step
///     writes the weighted combine directly into the per-rank partial
///     (pre-zeroed). With `use_i64_down` + MQ3G256Lloyd, `DownResidualI64`
///     accumulates the reproducible int64 partial; the finish phase converts
///     it to f32 after the i64 collective (Tp) or after local zeroing (Ep,
///     where the FP32 all-reduce lands on the convert). Matches the shipped
///     M2.7.mq2 path (down=MQ3L) — byte-identical.
///   - **Non-Lloyd down** (MQ4G256/HFQ4G256/MQ6G256/HFQ6G256): `DownExpanded`
///     writes per-expert outputs to `down_expanded`, then `MoeCombine`
///     (`inverse_perm: None`) folds them with `topk_weights` into the
///     pre-zeroed partial.
/// - **Phase 3** (residual fold): `state.h += partial` per rank.
///
/// Preconditions: every rank has an `active_stream` (`ensure_rank_streams`)
/// and peer access enabled. The `Gpus` must be bound to `policy.mesh()`
/// (mesh-identity check inside `execute_lowered_moe`).
#[allow(clippy::too_many_arguments)]
fn minimax_ep_moe_step(
    gpus: &mut hipfire_runtime::multi_gpu::Gpus,
    weights_per_rank: &[MiniMaxWeights],
    cfg: &MiniMaxConfig,
    state_per_rank: &[MiniMaxState],
    partials: &[GpuTensor],
    partials_i64: Option<&[GpuTensor]>,
    use_i64_down: bool,
    policy: &MoEExecutionPolicy,
    group: &ExpertGroupPlan,
    l: usize,
) -> Result<(), String> {
    use hipfire_dispatch::families::moe::{ExpertExecutionPlan, MoeExpertRef};
    use hipfire_runtime::multi_gpu::DimKind;
    let n = gpus.devices.len();
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let n_exp = cfg.num_local_experts;
    let k_top = cfg.num_experts_per_tok;
    let eps = cfg.rms_norm_eps;
    // TP-of-experts: inter_local = inter / tp. When tp==1 (no Tp axis),
    // size_of returns 1, inter_local == inter → byte-identical.
    let tp = policy.mesh().size_of(DimKind::Tp).max(1);
    let inter_local = inter / tp;
    // Per-rank expert refs (gate_up vs down dtype + EP dummy gate_up), built
    // before Phase 1 borrows gpus mutably so they outlive the step programs.
    // expert_m uses inter_local (the per-rank intermediate dim under TP).
    let gu_refs: Vec<MoeExpertRef> = (0..n)
        .map(|r| {
            let layer = &weights_per_rank[r].layers[l];
            MoeExpertRef {
                gate_up_ptrs: &layer.expert_gate_up_ptrs,
                down_ptrs: &layer.expert_down_ptrs,
                dummy_gate_up: layer.dummy_gate_up.as_ref(),
                dtype: layer.experts[0].gate_up.gpu_dtype,
                n_experts: n_exp,
                expert_m: inter_local,
                expert_k: hidden,
                owned: &[],
            }
        })
        .collect();
    let down_refs: Vec<MoeExpertRef> = (0..n)
        .map(|r| {
            let layer = &weights_per_rank[r].layers[l];
            MoeExpertRef {
                gate_up_ptrs: &layer.expert_gate_up_ptrs,
                down_ptrs: &layer.expert_down_ptrs,
                dummy_gate_up: layer.dummy_gate_up.as_ref(),
                dtype: layer.experts[0].down.gpu_dtype,
                n_experts: n_exp,
                expert_m: inter_local,
                expert_k: hidden,
                owned: &[],
            }
        })
        .collect();

    // ── Phase 1: per-rank pre-down MoE compute (direct, input-prep boundary) ─
    for r in 0..n {
        let layer = &weights_per_rank[r].layers[l];
        let s = &state_per_rank[r];
        let gpu = &mut gpus.devices[r];
        gpu.bind_thread()
            .map_err(|e| format!("moe-step bind {r} L{l}: {e:?}"))?;
        gpu.rmsnorm_f32(&s.h, &layer.ffn_norm, &s.ffn_tmp, eps)
            .map_err(|e| format!("moe-step L{l} r{r}: ffn rmsnorm: {e:?}"))?;
        rotate_x_mq_for(
            gpu,
            &layer.experts[0].gate_up,
            &s.ffn_tmp,
            &s.ffn_x_rot,
            hidden,
        )
        .map_err(|e| format!("moe-step L{l} r{r}: ffn rotate: {e:?}"))?;
        weight_gemv(gpu, &layer.router, &s.ffn_tmp, &s.router_logits)
            .map_err(|e| format!("moe-step L{l} r{r}: router: {e}"))?;
    }

    // ── Phase 2: lowered routed program (shared parts builder + sealed executor) ─
    // The i64 down path is keyed on the DOWN dtype (same across ranks — same
    // recipe): MQ3G256Lloyd + use_i64_down → DownResidualI64 → ConvertI64ToF32.
    let ddt = weights_per_rank[0].layers[l].experts[0].down.gpu_dtype;
    let use_i64 = use_i64_down && matches!(ddt, DType::MQ3G256Lloyd);
    let parts = MoeProgramParts {
        router: minimax_moe_router_plan(
            &state_per_rank[0].router_logits,
            &state_per_rank[0].topk_indices,
            &state_per_rank[0].topk_weights,
            k_top,
        ),
        execution: ExpertExecutionPlan::IndexedQuantized,
        ranks: (0..n)
            .map(|r| {
                let s = &state_per_rank[r];
                let layer = &weights_per_rank[r].layers[l];
                let inputs = MinimaxMoeInputs {
                    scores: &s.router_logits,
                    gate_bias: &layer.routing_bias,
                    topk_indices: &s.topk_indices,
                    topk_weights: &s.topk_weights,
                    x_rot: &s.ffn_x_rot,
                    gate_batch: &s.gate_batch,
                    up_batch: &s.up_batch,
                    rot_batch: &s.rot_batch,
                    down_expanded: &s.down_expanded,
                    partial: &partials[r],
                    partial_i64: partials_i64.map(|i64s| &i64s[r]),
                    gu_ref: &gu_refs[r],
                    down_ref: &down_refs[r],
                    awq_scale: layer.experts[0].down.awq_scale.as_ref(),
                };
                minimax_moe_rank_phases(&inputs, k_top, n_exp, inter_local, hidden, use_i64)
            })
            .collect(),
    };
    let program = lower_moe_steps(group, policy, parts)
        .map_err(|e| format!("moe-step L{l}: lower_moe_steps: {e}"))?;
    execute_lowered_moe(&program, MoeExecutionTarget::Parallel { gpus })
        .map_err(|e| format!("moe-step L{l}: execute_lowered_moe: {e:?}"))?;

    // ── Phase 3: fold the all-reduced partial into each rank's residual ──
    for r in 0..n {
        let gpu = &mut gpus.devices[r];
        gpu.bind_thread()
            .map_err(|e| format!("moe-step add bind {r} L{l}: {e:?}"))?;
        gpu.add_inplace_f32(&state_per_rank[r].h, &partials[r])
            .map_err(|e| format!("moe-step L{l} r{r}: add residual: {e:?}"))?;
    }
    Ok(())
}

/// Pure mesh-entry policy validation (CPU-testable seam; the mesh entries
/// call it through [`mesh_entry_authority_core`] BEFORE any GPU work or
/// authority acquisition): the policy kind must be the entry's required
/// kind (EP vs TP), the policy mesh must be the EXACT mesh the `Gpus` are
/// bound to (epoch identity), and the policy rank count must equal the device
/// count. Executes even when MoE is disabled/shared-only — wrong policies
/// refuse regardless. Mirror of deepseek4's `validate_mesh_policy_binding`.
pub(crate) fn validate_mesh_policy_binding(
    policy: &MoEExecutionPolicy,
    required_kind: MoEExecutionKind,
    bound_epoch: Option<hipfire_runtime::multi_gpu::MeshEpoch>,
    device_count: usize,
) -> Result<(), String> {
    if policy.kind() != required_kind {
        return Err(format!(
            "expected a {required_kind:?} execution policy, got {:?}",
            policy.kind()
        ));
    }
    let epoch = bound_epoch
        .ok_or_else(|| "the Gpus are not bound to a DeviceMesh (from_mesh required)".to_string())?;
    if epoch != policy.mesh().epoch() {
        return Err(
            "policy mesh epoch differs from the Gpus-bound mesh epoch (stale or different mesh)"
                .to_string(),
        );
    }
    if policy.rank_count() != device_count {
        return Err(format!(
            "policy rank count {} != device count {device_count}",
            policy.rank_count()
        ));
    }
    Ok(())
}

/// CPU-testable validate-then-acquire core of the mesh entries: the pure
/// binding seam (kind → exact epoch → rank count) runs first, and ONLY on
/// success is the authority acquisition invoked — the injected callback is
/// the production acquisition slot ([`MiniMaxWeights::expert_manifest_for_policy`]
/// via [`mesh_entry_authority`]); tests inject a counting closure into this
/// SAME production-consumed core. A refused policy never reaches the
/// acquisition.
fn mesh_entry_authority_core<T>(
    policy: &MoEExecutionPolicy,
    required_kind: MoEExecutionKind,
    bound_epoch: Option<hipfire_runtime::multi_gpu::MeshEpoch>,
    device_count: usize,
    acquire: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    validate_mesh_policy_binding(policy, required_kind, bound_epoch, device_count)?;
    acquire()
}

/// The production validate-then-acquire seam used by BOTH mesh entries at
/// their very start (before any GPU work / state mutation): kind-first (a
/// wrong-kind policy is never masked by a stale-mesh binding error), then the
/// exact mesh/epoch binding via the approved
/// [`hipfire_runtime::multi_gpu::Gpus::weight_origin_in`] API (UnboundMesh /
/// MeshEpochMismatch), then the composition core — and only after every check
/// passes does the acquisition run. The acquisition closure carries the real
/// caller-owned-policy resolution (`expert_manifest_for_policy`), preserving
/// the exact cache semantics and error text of the entries' former
/// mid-forward acquisition.
fn mesh_entry_authority<T>(
    gpus: &hipfire_runtime::multi_gpu::Gpus,
    policy: &MoEExecutionPolicy,
    required_kind: MoEExecutionKind,
    acquire: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    // 1. Kind FIRST: refused deterministically before any mesh/epoch binding
    //    check (a wrong-kind policy must never be masked by a stale-mesh
    //    binding error).
    if policy.kind() != required_kind {
        return Err(format!(
            "minimax {required_kind:?} entry policy: expected a {required_kind:?} execution \
             policy, got {:?}",
            policy.kind()
        ));
    }
    // 2. Exact mesh/epoch binding via the approved API, then rank/device
    //    agreement + acquisition through the composition core.
    let origin = gpus
        .weight_origin_in(policy.mesh(), 0)
        .map_err(|e| format!("minimax {required_kind:?} entry policy mesh binding: {e}"))?;
    mesh_entry_authority_core(
        policy,
        required_kind,
        Some(origin.mesh_epoch()),
        gpus.devices.len(),
        acquire,
    )
    .map_err(|e| format!("minimax {required_kind:?} entry policy: {e}"))
}

/// EP (Ship 6 substrate-EP) replicated N-rank decode forward for ONE token.
/// Mirror of qwen35::forward_ep: every rank holds full replicated weights /
/// state / KV EXCEPT MoE experts (sharded at load). Embeds + stages pos per
/// rank, runs each layer's 2-op program (Attend replicated, Moe all-reduce-EP'd)
/// via [`minimax_ep_moe_step`], then final norm + lm_head on rank 0 →
/// `state_per_rank[0].logits`. Every device must have an `active_stream`
/// ([`hipfire_runtime::multi_gpu::Gpus::ensure_rank_streams`]); peer access enabled for
/// the fast peer-direct all-reduce.
///
/// The MoE execution context is the CALLER-OWNED [`MoEExecutionPolicy`]:
/// `policy.mesh()` is the EP mesh (no local reconstruction), the per-layer
/// expert-group plans resolve from the policy-aware manifest hook, and the
/// `Gpus` must be bound to `policy.mesh()` (see `Gpus::from_mesh`).
#[allow(clippy::too_many_arguments)]
pub fn forward_ep(
    gpus: &mut hipfire_runtime::multi_gpu::Gpus,
    weights_per_rank: &[MiniMaxWeights],
    cfg: &MiniMaxConfig,
    state_per_rank: &mut [MiniMaxState],
    partials: &[GpuTensor],
    partials_i64: &[GpuTensor],
    policy: &MoEExecutionPolicy,
    token: u32,
    position: u32,
) -> Result<(), String> {
    let n = gpus.devices.len();
    // Gate-3 validate-then-acquire BEFORE any GPU work: the caller policy
    // must be the exact Ep kind + mesh/epoch binding of these Gpus, and the
    // authority resolution is acquired only after validation passes (never on
    // refusal) — enforced even when MoE is disabled (no local alternate mesh
    // model).
    let resolution = mesh_entry_authority(gpus, policy, MoEExecutionKind::Ep, || {
        weights_per_rank[0]
            .expert_manifest_for_policy(cfg, policy)
            .map_err(|e| format!("forward_ep: expert manifest: {e}"))
    })?;
    assert_eq!(
        weights_per_rank.len(),
        n,
        "forward_ep: weights_per_rank len"
    );
    assert_eq!(state_per_rank.len(), n, "forward_ep: state_per_rank len");
    assert_eq!(partials.len(), n, "forward_ep: partials len");
    assert_eq!(partials_i64.len(), n, "forward_ep: partials_i64 len");
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;

    // 1. Embed + stage pos per rank (replicated, deterministic).
    for r in 0..n {
        gpus.devices[r]
            .bind_thread()
            .map_err(|e| format!("forward_ep bind {r}: {e:?}"))?;
        gpus.devices[r]
            .embedding_lookup_q8(
                &weights_per_rank[r].embed,
                &state_per_rank[r].h,
                token,
                hidden,
            )
            .map_err(|e| format!("forward_ep embed {r}: {e:?}"))?;
        state_per_rank[r].pos_host[0] = position as i32;
        let pos_bytes = unsafe {
            std::slice::from_raw_parts(state_per_rank[r].pos_host.as_ptr() as *const u8, 4)
        };
        gpus.devices[r]
            .memcpy_htod_auto(&state_per_rank[r].pos_buf, pos_bytes)
            .map_err(|e| format!("forward_ep pos {r}: {e:?}"))?;
    }

    // 2. Per-layer EP program (Attend replicated; Moe all-reduce-EP'd).
    let timing = std::env::var("HIPFIRE_EP_DECODE_TIMING").is_ok();
    let t_layers = std::time::Instant::now();
    // Caller-owned policy: the EP mesh IS policy.mesh() — no local rect
    // reconstruction. Per-layer plans are borrowed by layer from the
    // resolution acquired at entry (identical cached object on every call —
    // no per-token plan allocation).
    let n_layers = weights_per_rank[0].layers.len();
    for l in 0..n_layers {
        // Attend replicated: every rank holds full weights + full KV → the
        // per-rank attention is a deterministic function of replicated inputs
        // and stays bit-identical across ranks (the only EP divergence is Moe).
        for r in 0..n {
            gpus.devices[r]
                .bind_thread()
                .map_err(|e| format!("forward_ep attn bind {r} L{l}: {e:?}"))?;
            minimax_attn_block(
                &mut gpus.devices[r],
                cfg,
                &weights_per_rank[r].layers[l],
                &state_per_rank[r],
                l,
            )
            .map_err(|e| format!("forward_ep attn L{l} r{r}: {e}"))?;
        }
        // Moe all-reduce EP: each rank runs its owned routed experts into a
        // partial (minimax has no shared expert → whole MoE is routed), the
        // partials all-reduce over the Ep group, and each rank folds the reduced
        // sum into its replicated attention residual `state.h`.
        minimax_ep_moe_step(
            gpus,
            weights_per_rank,
            cfg,
            state_per_rank,
            partials,
            Some(partials_i64), // EP i64: DownResidualI64 → ConvertI64ToF32 → AllReduce{Ep}
            true,               // use_i64_down: true → reproducible int64 down per rank
            policy,
            &resolution.plans[l],
            l,
        )
        .map_err(|e| format!("forward_ep moe-step L{l}: {e}"))?;
    }

    // 3. Final norm + lm_head on rank 0 → state_per_rank[0].logits.
    {
        gpus.devices[0]
            .bind_thread()
            .map_err(|e| format!("forward_ep bind0: {e:?}"))?;
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
        gpus.devices[r]
            .bind_thread()
            .map_err(|e| format!("forward_ep sync bind {r}: {e:?}"))?;
        gpus.devices[r]
            .hip
            .device_synchronize()
            .map_err(|e| format!("forward_ep sync {r}: {e:?}"))?;
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

/// TP-of-experts replicated N-rank decode forward for ONE token.
///
/// Every rank holds ALL experts but with column/row-sliced weight matrices
/// (loaded via `MiniMaxWeights::load(.., tp_slice=Some(TpExpertSlice{tp,rank}))`).
/// Each rank computes a partial output from its `inter/tp` intermediate slice;
/// the partials all-reduce over the Tp group, and each rank folds the full result
/// into its residual `state.h`. Attention is fully replicated (identical to `forward_ep`).
///
/// This function is parallel to `forward_ep`; the only structural difference is the
/// mesh axis (`DimKind::Tp` instead of `DimKind::Ep`), which drives `minimax_ep_moe_step`
/// to use `inter_local = inter/tp` and emit `AllReduce{Tp}`. The MoE execution
/// context is the CALLER-OWNED [`MoEExecutionPolicy`]: `policy.mesh()` is the Tp
/// mesh (no local reconstruction) and the `Gpus` must be bound to it.
#[allow(clippy::too_many_arguments)]
pub fn forward_tp(
    gpus: &mut hipfire_runtime::multi_gpu::Gpus,
    weights_per_rank: &[MiniMaxWeights],
    cfg: &MiniMaxConfig,
    state_per_rank: &mut [MiniMaxState],
    partials: &[GpuTensor],
    partials_i64: &[GpuTensor],
    policy: &MoEExecutionPolicy,
    token: u32,
    position: u32,
) -> Result<(), String> {
    let n = gpus.devices.len();
    // Gate-3 validate-then-acquire BEFORE any GPU work: the caller policy
    // must be the exact Tp kind + mesh/epoch binding of these Gpus, and the
    // authority resolution is acquired only after validation passes (never on
    // refusal) — enforced even when MoE is disabled (no local alternate mesh
    // model).
    let resolution = mesh_entry_authority(gpus, policy, MoEExecutionKind::Tp, || {
        weights_per_rank[0]
            .expert_manifest_for_policy(cfg, policy)
            .map_err(|e| format!("forward_tp: expert manifest: {e}"))
    })?;
    assert_eq!(
        weights_per_rank.len(),
        n,
        "forward_tp: weights_per_rank len"
    );
    assert_eq!(state_per_rank.len(), n, "forward_tp: state_per_rank len");
    assert_eq!(partials.len(), n, "forward_tp: partials len");
    assert_eq!(partials_i64.len(), n, "forward_tp: partials_i64 len");
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;

    // 1. Embed + stage pos per rank (replicated, deterministic).
    for r in 0..n {
        gpus.devices[r]
            .bind_thread()
            .map_err(|e| format!("forward_tp bind {r}: {e:?}"))?;
        gpus.devices[r]
            .embedding_lookup_q8(
                &weights_per_rank[r].embed,
                &state_per_rank[r].h,
                token,
                hidden,
            )
            .map_err(|e| format!("forward_tp embed {r}: {e:?}"))?;
        state_per_rank[r].pos_host[0] = position as i32;
        let pos_bytes = unsafe {
            std::slice::from_raw_parts(state_per_rank[r].pos_host.as_ptr() as *const u8, 4)
        };
        gpus.devices[r]
            .memcpy_htod_auto(&state_per_rank[r].pos_buf, pos_bytes)
            .map_err(|e| format!("forward_tp pos {r}: {e:?}"))?;
    }

    // 2. Per-layer TP program (Attend replicated; MoE all-reduce-Tp'd).
    let t_layers = std::time::Instant::now();
    // Caller-owned policy: the Tp mesh IS policy.mesh() — no local rect
    // reconstruction; per-layer plans are borrowed by layer from the
    // resolution acquired at entry (identical cached object on every call).
    let n_layers = weights_per_rank[0].layers.len();
    for l in 0..n_layers {
        // Attend replicated (identical to forward_ep).
        for r in 0..n {
            gpus.devices[r]
                .bind_thread()
                .map_err(|e| format!("forward_tp attn bind {r} L{l}: {e:?}"))?;
            minimax_attn_block(
                &mut gpus.devices[r],
                cfg,
                &weights_per_rank[r].layers[l],
                &state_per_rank[r],
                l,
            )
            .map_err(|e| format!("forward_tp attn L{l} r{r}: {e}"))?;
        }
        // MoE all-reduce Tp: each rank holds all experts with sliced inter/tp weights.
        // int64 down path: DownResidualI64 → AllReduceI64Tp → ConvertI64ToF32.
        // Partition-invariant: tp=1 and tp=2 produce bit-identical f32 outputs.
        minimax_ep_moe_step(
            gpus,
            weights_per_rank,
            cfg,
            state_per_rank,
            partials,
            Some(partials_i64), // i64 scratch buffers for reproducible TP down
            true,               // use_i64_down: always true on the TP path
            policy,
            &resolution.plans[l],
            l,
        )
        .map_err(|e| format!("forward_tp moe-step L{l}: {e}"))?;
    }

    // 3. Final norm + lm_head on rank 0 → state_per_rank[0].logits.
    {
        gpus.devices[0]
            .bind_thread()
            .map_err(|e| format!("forward_tp bind0: {e:?}"))?;
        let w = &weights_per_rank[0];
        let s = &state_per_rank[0];
        let gpu = &mut gpus.devices[0];
        gpu.rmsnorm_f32(&s.h, &w.final_norm, &s.final_norm_buf, eps)
            .map_err(|e| format!("forward_tp final norm: {e:?}"))?;
        weight_gemv(gpu, &w.lm_head, &s.final_norm_buf, &s.logits)
            .map_err(|e| format!("forward_tp lm_head: {e}"))?;
    }

    let _layers_ms = t_layers.elapsed().as_secs_f64() * 1000.0;
    // 4. Sync every rank.
    for r in 0..n {
        gpus.devices[r]
            .bind_thread()
            .map_err(|e| format!("forward_tp sync bind {r}: {e:?}"))?;
        gpus.devices[r]
            .hip
            .device_synchronize()
            .map_err(|e| format!("forward_tp sync {r}: {e:?}"))?;
    }
    for s in state_per_rank.iter_mut() {
        s.n_tokens = position as usize + 1;
    }
    Ok(())
}

// ─────────────────────────── Phase 3 · Task 7 tests ───────────────────────────
// No-GPU program-shape tests: the lowered MiniMax MoE program (phased
// `MoeProgramParts` built from the shared Step building blocks) must preserve
// the old bespoke sequencing's shape, the sigmoid+bias router, dummy experts,
// AWQ-down activation, Lloyd self-combine omission, expanded-down combine
// requirement, the i64-down→convert ordering, and the caller-owned policy
// threading of the EP/TP entries.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::MiniMaxM2;
    use hip_bridge::DeviceBuffer;
    use hipfire_dispatch::families::moe::{MoeExpertRef, RouterSelection};
    use hipfire_dispatch::pipeline::{
        GemvInput, MoeActivationVariant, MoeProj, ScoreActKind, Step,
    };
    use hipfire_runtime::arch::Architecture;
    use hipfire_runtime::moe_plan::{
        lower_moe_steps, select_moe_executor, MoEExecutionKind, MoEExecutionPolicy,
        MoeExecutorKind, MoeLowerError,
    };
    use hipfire_runtime::multi_gpu::{CollectiveHint, DeviceMesh, DimKind};
    use hipfire_runtime::tp_shard::ExpertAssign;
    use hipfire_runtime::weight_manifest::{
        resolve_expert_manifest_for_policy, validate_expert_group_spec, ExpertExecutionIdentity,
        ExpertParallelism, ExpertPostCombineAllReduce,
    };
    use rdna_compute::{DType, GpuTensor};

    // ── synthetic no-GPU tensors (byte capacities feed the lowerer's
    //    buf.size() capacity checks directly) ───────────────────────────────

    fn synth_with_bytes(dtype: DType, numel: usize, bytes: usize) -> &'static GpuTensor {
        let buffer = Box::leak(vec![0u8; bytes].into_boxed_slice());
        let tensor = Box::leak(Box::new(GpuTensor {
            buf: unsafe { DeviceBuffer::from_raw(buffer.as_mut_ptr().cast(), bytes) },
            shape: vec![numel],
            dtype,
        }));
        tensor
    }

    fn synth_i64(numel: usize) -> &'static GpuTensor {
        synth_with_bytes(DType::Raw, numel, numel * 8)
    }

    fn synth_f32(numel: usize) -> &'static GpuTensor {
        synth_with_bytes(DType::F32, numel, numel * 4)
    }

    fn expert_ref(
        dtype: DType,
        dummy: Option<&'static GpuTensor>,
    ) -> &'static MoeExpertRef<'static> {
        Box::leak(Box::new(MoeExpertRef {
            gate_up_ptrs: synth_f32(2 * N_EXP),
            down_ptrs: synth_f32(2 * N_EXP),
            dummy_gate_up: dummy,
            dtype,
            n_experts: N_EXP,
            expert_m: INTER,
            expert_k: HIDDEN,
            owned: &[],
        }))
    }

    // ── fixture shapes (1 layer, hidden 4, inter 8, 4 experts, top-2) ──────

    const N_EXP: usize = 4;
    const K_TOP: usize = 2;
    const INTER: usize = 8;
    const HIDDEN: usize = 4;

    fn test_inputs() -> MinimaxMoeInputs<'static> {
        test_inputs_with(DType::MQ4G256, None, None)
    }

    fn test_inputs_with(
        down_dtype: DType,
        dummy: Option<&'static GpuTensor>,
        awq: Option<&'static GpuTensor>,
    ) -> MinimaxMoeInputs<'static> {
        MinimaxMoeInputs {
            scores: synth_f32(N_EXP),
            gate_bias: synth_f32(N_EXP),
            topk_indices: synth_i64(K_TOP),
            topk_weights: synth_f32(K_TOP),
            x_rot: synth_f32(HIDDEN),
            gate_batch: synth_f32(K_TOP * INTER),
            up_batch: synth_f32(K_TOP * INTER),
            rot_batch: synth_f32(K_TOP * INTER),
            down_expanded: synth_f32(K_TOP * HIDDEN),
            partial: synth_f32(HIDDEN),
            partial_i64: Some(synth_with_bytes(DType::Raw, HIDDEN, HIDDEN * 8)),
            gu_ref: expert_ref(DType::MQ4G256, dummy),
            down_ref: expert_ref(down_dtype, dummy),
            awq_scale: awq,
        }
    }

    /// TP-valid fixture: even attention head counts (Tp=2 divisibility for the
    /// dense attention entries) and inter=512 so every projected TP local slice
    /// is 256-aligned (gate/up axis-1 and down axis-2 both slice to 256 under
    /// Tp=2; the projection gate runs for Tp=1 as well).
    fn resolution_config() -> MiniMaxConfig {
        MiniMaxConfig {
            vocab_size: 16,
            hidden_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 64,
            intermediate_size: 512,
            num_local_experts: N_EXP,
            num_experts_per_tok: K_TOP,
            rotary_dim: 2,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 64,
            use_qk_norm: true,
            use_routing_bias: true,
            scoring_func: "sigmoid".into(),
            num_mtp_modules: 0,
            reap_keep: None,
        }
    }

    /// `resolution_config` with an intermediate dim the projection path must
    /// refuse under Tp (8/2 = 4 is not a multiple of 256) — a genuine
    /// resolution failure, used to pin the cached-failure behavior.
    fn tp_invalid_config() -> MiniMaxConfig {
        let mut cfg = resolution_config();
        cfg.intermediate_size = 8;
        cfg
    }

    fn projection_policy(kind: MoEExecutionKind, ranks: usize) -> MoEExecutionPolicy {
        let mesh = match kind {
            MoEExecutionKind::Single => DeviceMesh::single(),
            MoEExecutionKind::Tp => DeviceMesh::rect(&[(DimKind::Tp, ranks)]),
            MoEExecutionKind::Ep => DeviceMesh::rect(&[(DimKind::Ep, ranks)]),
        };
        MoEExecutionPolicy::new(kind, mesh).unwrap()
    }

    fn minimax_resolved_plans(
        cfg: &MiniMaxConfig,
        policy: &MoEExecutionPolicy,
    ) -> Vec<ExpertGroupPlan> {
        let specs = MiniMaxM2::expert_group_manifest(cfg, policy);
        let manifest = MiniMaxM2::weight_manifest(cfg);
        resolve_expert_manifest_for_policy(&specs, &manifest, policy)
            .expect("policy-aware manifest resolution must succeed")
            .plans
    }

    fn minimax_resolved_plan(cfg: &MiniMaxConfig, policy: &MoEExecutionPolicy) -> ExpertGroupPlan {
        // The shared policy-aware resolver over the authoritative manifests —
        // the same resolution the production authority caches. (The cache
        // itself is sealed in minimax.rs; its admission/identity behavior is
        // pinned by the minimax.rs tests.)
        let specs = MiniMaxM2::expert_group_manifest(cfg, policy);
        let manifest = MiniMaxM2::weight_manifest(cfg);
        resolve_expert_manifest_for_policy(&specs, &manifest, policy)
            .expect("policy-aware manifest resolution must succeed")
            .plans[0]
            .clone()
    }

    /// Typed semantic projection of one routed Step — the comparison
    /// vocabulary for the old-vs-lowered differential. Carries real values
    /// (buffer identities, k, n_experts, route_scale, down dtype, dummy/AWQ
    /// presence, conversion n) so the comparison is semantic, never
    /// label-based.
    #[derive(Debug, Clone, PartialEq)]
    enum MoeSemantic {
        SigmoidScore {
            scores: *const GpuTensor,
        },
        BiasAwareRoute {
            gate_bias: *const GpuTensor,
            k: usize,
            n_experts: usize,
            route_scale: f32,
        },
        GateUp {
            x_rot: *const GpuTensor,
            gate_out: *const GpuTensor,
            up_out: *const GpuTensor,
            dummy: bool,
        },
        FusedSiluMulRotate {
            awq_scale: Option<*const GpuTensor>,
            inter: usize,
            k_top: usize,
        },
        Down {
            kind: DownSemantic,
            dtype: DType,
            dummy: bool,
        },
        Combine {
            down_out: *const GpuTensor,
            out: *const GpuTensor,
            k: usize,
            hidden: usize,
            batch_size: usize,
        },
        ConvertI64 {
            src: *const GpuTensor,
            dst: *const GpuTensor,
            n: usize,
        },
    }

    #[derive(Debug, Clone, PartialEq)]
    enum DownSemantic {
        ResidualF32 { out: *const GpuTensor },
        ResidualI64 { out: *const GpuTensor },
        Expanded { out: *const GpuTensor },
    }

    fn tensor_ptr(t: &GpuTensor) -> *const GpuTensor {
        t as *const GpuTensor
    }

    /// Project a flat Step program into the typed semantic trace.
    fn semantic_trace(steps: &[&Step<'static>]) -> Vec<MoeSemantic> {
        steps
            .iter()
            .map(|s| match s {
                Step::ScoreActivation {
                    scores,
                    kind: ScoreActKind::Sigmoid,
                } => MoeSemantic::SigmoidScore {
                    scores: tensor_ptr(scores),
                },
                Step::MoeRoute {
                    gate_bias,
                    k,
                    n_experts,
                    route_scale,
                    ..
                } => MoeSemantic::BiasAwareRoute {
                    gate_bias: tensor_ptr(gate_bias),
                    k: *k,
                    n_experts: *n_experts,
                    route_scale: *route_scale,
                },
                Step::IndexedMoeGemv {
                    experts,
                    which: MoeProj::GateUp { up_out },
                    input: GemvInput::Prerotated(x_rot),
                    out,
                    ..
                } => MoeSemantic::GateUp {
                    x_rot: tensor_ptr(x_rot),
                    gate_out: tensor_ptr(out),
                    up_out: tensor_ptr(up_out),
                    dummy: experts.dummy_gate_up.is_some(),
                },
                Step::MoeActivation {
                    variant: MoeActivationVariant::MinimaxFused { awq_scale },
                    inter,
                    k_top,
                    ..
                } => MoeSemantic::FusedSiluMulRotate {
                    awq_scale: awq_scale.map(tensor_ptr),
                    inter: *inter,
                    k_top: *k_top,
                },
                Step::IndexedMoeGemv {
                    experts,
                    which: MoeProj::DownResidual { .. },
                    out,
                    ..
                } => MoeSemantic::Down {
                    kind: DownSemantic::ResidualF32 {
                        out: tensor_ptr(out),
                    },
                    dtype: experts.dtype,
                    dummy: experts.dummy_gate_up.is_some(),
                },
                Step::IndexedMoeGemv {
                    experts,
                    which: MoeProj::DownResidualI64 { .. },
                    out,
                    ..
                } => MoeSemantic::Down {
                    kind: DownSemantic::ResidualI64 {
                        out: tensor_ptr(out),
                    },
                    dtype: experts.dtype,
                    dummy: experts.dummy_gate_up.is_some(),
                },
                Step::IndexedMoeGemv {
                    experts,
                    which: MoeProj::DownExpanded,
                    out,
                    ..
                } => MoeSemantic::Down {
                    kind: DownSemantic::Expanded {
                        out: tensor_ptr(out),
                    },
                    dtype: experts.dtype,
                    dummy: experts.dummy_gate_up.is_some(),
                },
                Step::MoeCombine {
                    down_out,
                    out,
                    k,
                    hidden,
                    batch_size,
                    inverse_perm: None,
                    ..
                } => MoeSemantic::Combine {
                    down_out: tensor_ptr(down_out),
                    out: tensor_ptr(out),
                    k: *k,
                    hidden: *hidden,
                    batch_size: *batch_size,
                },
                Step::ConvertI64ToF32 { src, dst, n } => MoeSemantic::ConvertI64 {
                    src: tensor_ptr(src),
                    dst: tensor_ptr(dst),
                    n: *n,
                },
                _ => panic!("unexpected step in MiniMax semantic trace"),
            })
            .collect()
    }

    /// Flatten the six routed phases in the lowerer's canonical phase order.
    fn flatten_steps<'a>(phases: &'a RoutedMoeStepPhases<'static>) -> Vec<&'a Step<'static>> {
        let mut flat = Vec::new();
        for phase in [
            &phases.router,
            &phases.gate_up,
            &phases.activation,
            &phases.down,
            &phases.combine,
            &phases.finish,
        ] {
            flat.extend(phase.iter());
        }
        flat
    }

    /// The old bespoke routed MoE program — the independent test-only oracle.
    /// Transcribed from the pre-migration sequencing (HEAD
    /// `minimax_ep_moe_step` step lists, which encode the same semantics as
    /// `minimax_moe_block`'s raw kernels: sigmoid → bias-aware top-k with
    /// route_scale 1.0 → gate_up → silu·mul·rotate (AWQ from the down
    /// weight) → Lloyd residual-fused down OR expanded down + combine, with
    /// the i64 conversion after the collective-bearing down). Production never
    /// uses this — the sealed lowerer/executor is the only production path.
    fn legacy_minimax_flat_program(
        inputs: &MinimaxMoeInputs<'static>,
        k_top: usize,
        n_experts: usize,
        inter: usize,
        hidden: usize,
        use_i64_down: bool,
    ) -> Vec<Step<'static>> {
        let mut steps = vec![
            Step::ScoreActivation {
                scores: inputs.scores,
                kind: ScoreActKind::Sigmoid,
            },
            Step::MoeRoute {
                scores: inputs.scores,
                gate_bias: inputs.gate_bias,
                topk_indices: inputs.topk_indices,
                topk_weights: inputs.topk_weights,
                k: k_top,
                n_experts,
                route_scale: 1.0,
            },
            Step::IndexedMoeGemv {
                experts: inputs.gu_ref,
                which: MoeProj::GateUp {
                    up_out: inputs.up_batch,
                },
                topk_indices: inputs.topk_indices,
                input: GemvInput::Prerotated(inputs.x_rot),
                out: inputs.gate_batch,
                k_top,
                batch_size: 1,
            },
            Step::MoeActivation {
                variant: MoeActivationVariant::MinimaxFused {
                    awq_scale: inputs.awq_scale,
                },
                gate: inputs.gate_batch,
                up: inputs.up_batch,
                rot_out: inputs.rot_batch,
                inter,
                k_top,
            },
        ];
        let ddt = inputs.down_ref.dtype;
        let use_i64 = use_i64_down && matches!(ddt, DType::MQ3G256Lloyd);
        if use_i64 {
            let i64_out = inputs
                .partial_i64
                .expect("minimax i64 down requires the per-rank i64 partial");
            steps.push(Step::IndexedMoeGemv {
                experts: inputs.down_ref,
                which: MoeProj::DownResidualI64 {
                    topk_weights: inputs.topk_weights,
                },
                topk_indices: inputs.topk_indices,
                input: GemvInput::Prerotated(inputs.rot_batch),
                out: i64_out,
                k_top,
                batch_size: 1,
            });
            steps.push(Step::ConvertI64ToF32 {
                src: i64_out,
                dst: inputs.partial,
                n: hidden,
            });
        } else if matches!(ddt, DType::MQ2G256Lloyd | DType::MQ3G256Lloyd) {
            steps.push(Step::IndexedMoeGemv {
                experts: inputs.down_ref,
                which: MoeProj::DownResidual {
                    topk_weights: inputs.topk_weights,
                },
                topk_indices: inputs.topk_indices,
                input: GemvInput::Prerotated(inputs.rot_batch),
                out: inputs.partial,
                k_top,
                batch_size: 1,
            });
        } else {
            steps.push(Step::IndexedMoeGemv {
                experts: inputs.down_ref,
                which: MoeProj::DownExpanded,
                topk_indices: inputs.topk_indices,
                input: GemvInput::Prerotated(inputs.rot_batch),
                out: inputs.down_expanded,
                k_top,
                batch_size: 1,
            });
            steps.push(Step::MoeCombine {
                down_out: inputs.down_expanded,
                topk_weights: inputs.topk_weights,
                out: inputs.partial,
                k: k_top,
                hidden,
                batch_size: 1,
                inverse_perm: None,
            });
        }
        steps
    }

    #[test]
    fn minimax_manifest_projects_single_tp_ep_with_exact_placements() {
        // Production plan authority (Phase-3 manifest migration): the per-layer
        // expert-group plans originate from the shared policy-aware resolver
        // over the authoritative `MiniMaxM2::weight_manifest` + policy-aware
        // `expert_group_manifest` specs. The static `ExpertSharded` source
        // declaration projects to the effective resident placement for EVERY
        // exact execution policy — Single(1), Tp(1), Tp(2), Ep(1), Ep(2) —
        // with exact placements and the single parallelism-derived
        // post-combine collective. (Replaces the pinned Single/Tp failure test
        // of the pre-projection shared contract.)
        let cfg = resolution_config();
        let manifest = MiniMaxM2::weight_manifest(&cfg);
        let cases = [
            (MoEExecutionKind::Single, 1, ExpertParallelism::Single, None),
            (
                MoEExecutionKind::Tp,
                1,
                ExpertParallelism::TensorParallel,
                Some(ExpertPostCombineAllReduce::TensorParallel),
            ),
            (
                MoEExecutionKind::Tp,
                2,
                ExpertParallelism::TensorParallel,
                Some(ExpertPostCombineAllReduce::TensorParallel),
            ),
            (
                MoEExecutionKind::Ep,
                1,
                ExpertParallelism::ExpertParallel,
                Some(ExpertPostCombineAllReduce::ExpertParallel),
            ),
            (
                MoEExecutionKind::Ep,
                2,
                ExpertParallelism::ExpertParallel,
                Some(ExpertPostCombineAllReduce::ExpertParallel),
            ),
        ];
        for (kind, ranks, parallelism, collective) in cases {
            let policy = projection_policy(kind, ranks);
            let specs = MiniMaxM2::expert_group_manifest(&cfg, &policy);
            let resolution = resolve_expert_manifest_for_policy(&specs, &manifest, &policy)
                .unwrap_or_else(|err| panic!("{kind:?} ranks={ranks} must resolve: {err}"));
            assert_eq!(resolution.plans.len(), specs.len());
            let plan = &resolution.plans[0];
            assert_eq!(plan.group_size, ranks);
            assert_eq!(plan.parallelism, parallelism);
            assert_eq!(plan.assignment, ExpertAssign::Stride);
            assert_eq!(plan.collective, collective);
            // Exact effective placements, in declaration (global-expert) order.
            match parallelism {
                ExpertParallelism::Single => {
                    assert_eq!(plan.experts.len(), N_EXP);
                    assert!(plan.experts.iter().all(|e| e.owner == 0
                        && e.local_slot == e.global_id
                        && e.global_id < N_EXP));
                }
                ExpertParallelism::TensorParallel => {
                    assert_eq!(plan.experts.len(), N_EXP * ranks);
                    for global_id in 0..N_EXP {
                        let slots: Vec<_> = plan
                            .experts
                            .iter()
                            .filter(|e| e.global_id == global_id)
                            .map(|e| (e.owner, e.local_slot))
                            .collect();
                        let expected: Vec<_> = (0..ranks).map(|owner| (owner, global_id)).collect();
                        assert_eq!(slots, expected, "{kind:?} ranks={ranks} global {global_id}");
                    }
                }
                ExpertParallelism::ExpertParallel => {
                    assert_eq!(plan.experts.len(), N_EXP);
                    for global_id in 0..N_EXP {
                        let e = &plan.experts[global_id];
                        assert_eq!(e.owner, global_id % ranks, "{kind:?} ranks={ranks}");
                        assert_eq!(e.local_slot, global_id / ranks, "{kind:?} ranks={ranks}");
                    }
                }
            }
            // Residual per-weight schedule: the claimed expert sources are
            // excluded exactly once; only the dense row-parallel `wo` remains
            // (a Tp all-reduce — placement evidence, never extra expert
            // collective authority).
            assert_eq!(
                resolution.layer_collectives,
                vec![(0, CollectiveHint::AllReduce { kind: DimKind::Tp })],
                "{kind:?} ranks={ranks}"
            );
        }
    }

    #[test]
    fn minimax_single_old_vs_lowered_program_shape() {
        // GENUINE old-vs-lowered differential: the hand-transcribed legacy
        // oracle (`legacy_minimax_flat_program`, test-only) is compared — via
        // the typed `MoeSemantic` trace carrying real values — against the
        // phased program that an ACTUAL `lower_moe_steps` result admits for
        // the same inputs. Every required semantic dimension is exercised:
        // sigmoid+bias routing, route scale, dummy experts, AWQ down, both
        // Lloyd modes, every expanded mode, and the i64 conversion placement.
        let single = MoEExecutionPolicy::single();
        let tp_policy =
            MoEExecutionPolicy::new(MoEExecutionKind::Tp, DeviceMesh::rect(&[(DimKind::Tp, 2)]))
                .unwrap();
        struct Case {
            down_dtype: DType,
            use_i64: bool,
            policy: MoEExecutionPolicy,
            ranks: usize,
        }
        let cases = [
            Case {
                down_dtype: DType::MQ4G256,
                use_i64: false,
                policy: single.clone(),
                ranks: 1,
            },
            Case {
                down_dtype: DType::HFQ4G256,
                use_i64: false,
                policy: single.clone(),
                ranks: 1,
            },
            Case {
                down_dtype: DType::MQ6G256,
                use_i64: false,
                policy: single.clone(),
                ranks: 1,
            },
            Case {
                down_dtype: DType::HFQ6G256,
                use_i64: false,
                policy: single.clone(),
                ranks: 1,
            },
            Case {
                down_dtype: DType::MQ2G256Lloyd,
                use_i64: false,
                policy: single.clone(),
                ranks: 1,
            },
            Case {
                down_dtype: DType::MQ3G256Lloyd,
                use_i64: false,
                policy: single.clone(),
                ranks: 1,
            },
            Case {
                down_dtype: DType::MQ3G256Lloyd,
                use_i64: true,
                policy: tp_policy,
                ranks: 2,
            },
        ];
        for case in cases {
            let label = format!("{:?} i64={}", case.down_dtype, case.use_i64);
            // Dummy expert pack + AWQ down scale present in every case so the
            // trace proves they survive into both programs.
            let dummy = synth_f32(4);
            let awq = synth_f32(4);
            let inputs = test_inputs_with(case.down_dtype, Some(dummy), Some(awq));
            let legacy =
                legacy_minimax_flat_program(&inputs, K_TOP, N_EXP, INTER, HIDDEN, case.use_i64);
            let legacy_trace = semantic_trace(&legacy.iter().collect::<Vec<_>>());
            let phases =
                minimax_moe_rank_phases(&inputs, K_TOP, N_EXP, INTER, HIDDEN, case.use_i64);
            let new_trace = semantic_trace(&flatten_steps(&phases));
            // The typed semantics must be identical to the old sequencing.
            assert_eq!(
                new_trace, legacy_trace,
                "{label}: lowered program semantics differ from the legacy oracle"
            );
            // And an ACTUAL lower_moe_steps result must admit this program
            // under the case's policy (Single for the f32 paths, Tp for the
            // i64 path — the i64 conversion placement is validated as
            // post-collective by the lowerer).
            let group = minimax_resolved_plan(&resolution_config(), &case.policy);
            let parts = MoeProgramParts {
                router: minimax_moe_router_plan(
                    inputs.scores,
                    inputs.topk_indices,
                    inputs.topk_weights,
                    K_TOP,
                ),
                execution: minimax_moe_execution_plan(),
                ranks: (0..case.ranks)
                    .map(|_| {
                        minimax_moe_rank_phases(&inputs, K_TOP, N_EXP, INTER, HIDDEN, case.use_i64)
                    })
                    .collect(),
            };
            let lowered = lower_moe_steps(&group, &case.policy, parts)
                .unwrap_or_else(|e| panic!("{label}: lower_moe_steps must admit: {e}"));
            if case.use_i64 {
                // Sealed-result schedule evidence: the actual lowered program
                // carries the typed i64 Tp collective on the down step (the
                // conversion stays after it in the finish phase).
                let schedule = format!("{lowered:?}");
                assert!(
                    schedule.contains("AllReduceI64Tp"),
                    "{label}: lowered schedule must place the i64 collective: {schedule}"
                );
            }
        }
    }

    #[test]
    fn minimax_sigmoid_bias_router_is_preserved() {
        // Typed router identity: sigmoid_topk, normalized, route_scale = 1.0
        // (MiniMax applies no routed-scaling factor), on the indexed-quantized
        // execution plan.
        let inputs = test_inputs();
        let parts = minimax_moe_program(&inputs, K_TOP, N_EXP, INTER, HIDDEN, false);
        assert_eq!(parts.router.selection(), RouterSelection::SigmoidTopK);
        assert_eq!(parts.router.k_top(), K_TOP);
        assert!(parts.router.normalizes());
        assert_eq!(parts.router.route_scale(), 1.0);
        assert_eq!(
            parts.execution,
            hipfire_dispatch::families::moe::ExpertExecutionPlan::IndexedQuantized
        );
        // The concrete router phase preserves sigmoid score activation feeding
        // the bias-aware top-k route with the layer's routing bias.
        let phases = &parts.ranks[0];
        assert_eq!(phases.router.len(), 2);
        assert!(matches!(
            phases.router[0],
            Step::ScoreActivation {
                kind: ScoreActKind::Sigmoid,
                ..
            }
        ));
        let Step::MoeRoute {
            gate_bias,
            route_scale,
            k,
            n_experts,
            ..
        } = &phases.router[1]
        else {
            panic!("router[1] is not MoeRoute");
        };
        assert!(std::ptr::eq(*gate_bias, inputs.gate_bias));
        assert_eq!(*route_scale, 1.0);
        assert_eq!(*k, K_TOP);
        assert_eq!(*n_experts, N_EXP);
        // The manifest-declared identity must match the typed plan: lowering
        // accepts `sigmoid_topk` and rejects a `bias_aware_topk` declaration.
        let policy = MoEExecutionPolicy::single();
        let group = minimax_resolved_plan(&resolution_config(), &policy);
        assert_eq!(group.router_identity, "sigmoid_topk");
        assert!(lower_moe_steps(&group, &policy, parts).is_ok());
        let mut wrong_identity = group.clone();
        wrong_identity.router_identity = "bias_aware_topk".into();
        let err = lower_moe_steps(
            &wrong_identity,
            &policy,
            minimax_moe_program(&inputs, K_TOP, N_EXP, INTER, HIDDEN, false),
        )
        .unwrap_err();
        assert!(matches!(err, MoeLowerError::RouterIdentityMismatch { .. }));
    }

    #[test]
    fn minimax_dummy_experts_and_awq_down_are_preserved() {
        // EP-shard dummy gate_up and the down weight's AWQ activation scale
        // survive into the lowered program's steps.
        let dummy = synth_f32(4);
        let awq = synth_f32(4);
        let inputs = test_inputs_with(DType::MQ4G256, Some(dummy), Some(awq));
        let parts = minimax_moe_program(&inputs, K_TOP, N_EXP, INTER, HIDDEN, false);
        let phases = &parts.ranks[0];
        let Step::IndexedMoeGemv {
            experts,
            which: MoeProj::GateUp { .. },
            ..
        } = &phases.gate_up[0]
        else {
            panic!("gate_up[0] is not an IndexedMoeGemv GateUp");
        };
        assert_eq!(
            experts.dummy_gate_up.map(|d| d as *const GpuTensor),
            Some(dummy as *const GpuTensor)
        );
        let Step::IndexedMoeGemv { experts, .. } = &phases.down[0] else {
            panic!("down[0] is not an IndexedMoeGemv");
        };
        assert_eq!(
            experts.dummy_gate_up.map(|d| d as *const GpuTensor),
            Some(dummy as *const GpuTensor)
        );
        let Step::MoeActivation {
            variant: MoeActivationVariant::MinimaxFused { awq_scale },
            ..
        } = &phases.activation[0]
        else {
            panic!("activation[0] is not MinimaxFused");
        };
        assert_eq!(
            awq_scale.map(|s| s as *const GpuTensor),
            Some(awq as *const GpuTensor)
        );
    }

    #[test]
    fn minimax_lloyd_down_omits_combine() {
        // Lloyd down (MQ2G256Lloyd / MQ3G256Lloyd, f32 path): the residual-
        // fused down IS the weighted combine, so no combine phase and no
        // conversion phase may exist.
        for down_dtype in [DType::MQ2G256Lloyd, DType::MQ3G256Lloyd] {
            let inputs = test_inputs_with(down_dtype, None, None);
            let parts = minimax_moe_program(&inputs, K_TOP, N_EXP, INTER, HIDDEN, false);
            let phases = &parts.ranks[0];
            assert_eq!(
                phases.combine.len(),
                0,
                "{down_dtype:?}: combine phase must be empty"
            );
            assert_eq!(
                phases.finish.len(),
                0,
                "{down_dtype:?}: finish phase must be empty"
            );
            assert!(matches!(
                &phases.down[0],
                Step::IndexedMoeGemv {
                    which: MoeProj::DownResidual { .. },
                    out,
                    ..
                } if std::ptr::eq(*out, inputs.partial)
            ));
        }
        // The f32 self-combining program lowers under a Single policy.
        let inputs = test_inputs_with(DType::MQ2G256Lloyd, None, None);
        let parts = minimax_moe_program(&inputs, K_TOP, N_EXP, INTER, HIDDEN, false);
        let policy = MoEExecutionPolicy::single();
        let group = minimax_resolved_plan(&resolution_config(), &policy);
        assert!(lower_moe_steps(&group, &policy, parts).is_ok());
    }

    #[test]
    fn minimax_expanded_down_requires_combine() {
        // Non-Lloyd down (MQ4/HFQ4/MQ6/HFQ6): the expanded per-expert write
        // requires exactly one combine folding into the EP partial, with no
        // inverse permutation (decode indexed path) and no conversion phase.
        for down_dtype in [
            DType::MQ4G256,
            DType::HFQ4G256,
            DType::MQ6G256,
            DType::HFQ6G256,
        ] {
            let inputs = test_inputs_with(down_dtype, None, None);
            let parts = minimax_moe_program(&inputs, K_TOP, N_EXP, INTER, HIDDEN, false);
            let phases = &parts.ranks[0];
            assert!(matches!(
                &phases.down[0],
                Step::IndexedMoeGemv {
                    which: MoeProj::DownExpanded,
                    out,
                    ..
                } if std::ptr::eq(*out, inputs.down_expanded)
            ));
            assert_eq!(
                phases.combine.len(),
                1,
                "{down_dtype:?}: expanded down requires combine"
            );
            assert_eq!(phases.finish.len(), 0, "{down_dtype:?}: no finish phase");
            let Step::MoeCombine {
                down_out,
                out,
                k,
                hidden,
                batch_size,
                inverse_perm,
                ..
            } = &phases.combine[0]
            else {
                panic!("combine[0] is not MoeCombine");
            };
            assert!(std::ptr::eq(*down_out, inputs.down_expanded));
            assert!(std::ptr::eq(*out, inputs.partial));
            assert_eq!(*k, K_TOP);
            assert_eq!(*hidden, HIDDEN);
            assert_eq!(*batch_size, 1);
            assert!(inverse_perm.is_none());
        }
    }

    #[test]
    fn minimax_tp_i64_reduction_converts_after_collective() {
        // MQ3-Lloyd + int64 down: the self-combining i64 down writes the i64
        // partial; the finish phase converts it to f32 AFTER the
        // collective-bearing down step (the lowerer lands the Tp i64
        // all-reduce on the down step and the f32 convert after it).
        let inputs = test_inputs_with(DType::MQ3G256Lloyd, None, None);
        let parts = minimax_moe_program(&inputs, K_TOP, N_EXP, INTER, HIDDEN, true);
        let phases = &parts.ranks[0];
        assert_eq!(phases.combine.len(), 0);
        assert!(matches!(
            &phases.down[0],
            Step::IndexedMoeGemv {
                which: MoeProj::DownResidualI64 { .. },
                out,
                ..
            } if std::ptr::eq(*out, inputs.partial_i64.unwrap())
        ));
        assert_eq!(phases.finish.len(), 1);
        let Step::ConvertI64ToF32 { src, dst, n } = &phases.finish[0] else {
            panic!("finish[0] is not ConvertI64ToF32");
        };
        assert!(std::ptr::eq(*src, inputs.partial_i64.unwrap()));
        assert!(std::ptr::eq(*dst, inputs.partial));
        assert_eq!(*n, HIDDEN);

        // The typed Tp group admits the i64 program (TpI64 protocol).
        let tp_policy =
            MoEExecutionPolicy::new(MoEExecutionKind::Tp, DeviceMesh::rect(&[(DimKind::Tp, 2)]))
                .unwrap();
        let group = minimax_resolved_plan(&resolution_config(), &tp_policy);
        assert_eq!(
            select_moe_executor(&group, &tp_policy).unwrap(),
            MoeExecutorKind::Parallel
        );
        // Two identical rank programs (one per Tp rank).
        let two_ranks = || MoeProgramParts {
            router: minimax_moe_router_plan(
                inputs.scores,
                inputs.topk_indices,
                inputs.topk_weights,
                K_TOP,
            ),
            execution: minimax_moe_execution_plan(),
            ranks: vec![
                minimax_moe_rank_phases(&inputs, K_TOP, N_EXP, INTER, HIDDEN, true),
                minimax_moe_rank_phases(&inputs, K_TOP, N_EXP, INTER, HIDDEN, true),
            ],
        };
        assert!(lower_moe_steps(&group, &tp_policy, two_ranks()).is_ok());

        // The Ep policy admits the same i64 program (EpLocalI64: local i64
        // zeroing at the down, FP32 EP all-reduce at the convert).
        let ep_policy =
            MoEExecutionPolicy::new(MoEExecutionKind::Ep, DeviceMesh::rect(&[(DimKind::Ep, 2)]))
                .unwrap();
        let group = minimax_resolved_plan(&resolution_config(), &ep_policy);
        assert!(lower_moe_steps(&group, &ep_policy, two_ranks()).is_ok());

        // A single-rank policy rejects the i64 program: the int64 down is not
        // admitted on a single-rank axis.
        let single = MoEExecutionPolicy::single();
        let group = minimax_resolved_plan(&resolution_config(), &single);
        let parts = minimax_moe_program(&inputs, K_TOP, N_EXP, INTER, HIDDEN, true);
        let err = lower_moe_steps(&group, &single, parts).unwrap_err();
        assert!(matches!(err, MoeLowerError::I64OnNonAdmittedAxis { .. }));
    }

    #[test]
    fn minimax_forward_entry_requires_caller_policy() {
        // The production EP/TP entries take the caller-owned policy by
        // reference; removing the parameter is a compile error at these pins.
        let ep: fn(
            &mut hipfire_runtime::multi_gpu::Gpus,
            &[MiniMaxWeights],
            &MiniMaxConfig,
            &mut [MiniMaxState],
            &[GpuTensor],
            &[GpuTensor],
            &MoEExecutionPolicy,
            u32,
            u32,
        ) -> Result<(), String> = forward_ep;
        let tp: fn(
            &mut hipfire_runtime::multi_gpu::Gpus,
            &[MiniMaxWeights],
            &MiniMaxConfig,
            &mut [MiniMaxState],
            &[GpuTensor],
            &[GpuTensor],
            &MoEExecutionPolicy,
            u32,
            u32,
        ) -> Result<(), String> = forward_tp;
        let _ = (ep, tp);

        // The policy is genuinely consumed: parallelism, assignment, group
        // size, and the post-combine collective derive from the caller's
        // policy only (no locally reconstructed mesh), for every policy kind,
        // through the shared policy-aware resolver (the same resolution the
        // production authority caches; the cache itself is sealed in
        // minimax.rs and pinned by its tests).
        let cfg = resolution_config();
        let single = MoEExecutionPolicy::single();
        let plans = minimax_resolved_plans(&cfg, &single);
        assert_eq!(plans[0].group_size, 1);
        assert_eq!(plans[0].parallelism, ExpertParallelism::Single);
        assert_eq!(plans[0].collective, None);
        assert_eq!(plans[0].assignment, ExpertAssign::Stride);
        assert_eq!(
            plans[0].allowed_executions,
            vec![ExpertExecutionIdentity::IndexedQuantized]
        );
        assert_eq!(
            validate_expert_group_spec(&MiniMaxM2::expert_group_manifest(&cfg, &single)[0], 1,)
                .map_err(|e| e),
            Ok(())
        );

        let tp_policy =
            MoEExecutionPolicy::new(MoEExecutionKind::Tp, DeviceMesh::rect(&[(DimKind::Tp, 2)]))
                .unwrap();
        let plans = minimax_resolved_plans(&cfg, &tp_policy);
        assert_eq!(plans[0].group_size, 2);
        assert_eq!(plans[0].parallelism, ExpertParallelism::TensorParallel);
        assert_eq!(
            plans[0].collective,
            Some(ExpertPostCombineAllReduce::TensorParallel)
        );
        assert_eq!(
            select_moe_executor(&plans[0], &tp_policy).unwrap(),
            MoeExecutorKind::Parallel
        );

        let ep_policy =
            MoEExecutionPolicy::new(MoEExecutionKind::Ep, DeviceMesh::rect(&[(DimKind::Ep, 2)]))
                .unwrap();
        let plans = minimax_resolved_plans(&cfg, &ep_policy);
        assert_eq!(plans[0].group_size, 2);
        assert_eq!(plans[0].parallelism, ExpertParallelism::ExpertParallel);
        assert_eq!(
            plans[0].collective,
            Some(ExpertPostCombineAllReduce::ExpertParallel)
        );
        assert_eq!(
            select_moe_executor(&plans[0], &ep_policy).unwrap(),
            MoeExecutorKind::Parallel
        );
    }

    // ───────────────────── Gate-3 · mesh-entry policy authority ──────────
    // The EP/TP entries compose expected-kind / exact mesh / rank validation
    // with authority acquisition in ONE production validate-then-acquire seam
    // (`mesh_entry_authority` → `mesh_entry_authority_core`) before any GPU
    // work (mirror of deepseek4's `validate_mesh_entry_policy`): kind-first,
    // then the exact mesh/epoch the `Gpus` are bound to, then rank count —
    // and only on success is the acquisition reached. The CPU-testable
    // production core takes the counting acquisition callback.

    #[test]
    fn minimax_mesh_entry_requires_exact_policy_kind() {
        // Each entry's required kind accepts its own kind and refuses the
        // other kind deterministically (kind-first — a wrong-kind policy is
        // never masked by a stale-mesh binding error).
        for (entry_kind, other_kind) in [
            (MoEExecutionKind::Ep, MoEExecutionKind::Tp),
            (MoEExecutionKind::Tp, MoEExecutionKind::Ep),
        ] {
            let axis = match entry_kind {
                MoEExecutionKind::Ep => DimKind::Ep,
                MoEExecutionKind::Tp => DimKind::Tp,
                MoEExecutionKind::Single => unreachable!(),
            };
            let mesh = DeviceMesh::rect(&[(axis, 2)]);
            let policy = MoEExecutionPolicy::new(entry_kind, mesh.clone()).unwrap();
            // The entry's own kind + the exact bound mesh/epoch passes.
            validate_mesh_policy_binding(&policy, entry_kind, Some(mesh.epoch()), 2)
                .expect("correct kind + bound mesh must pass");
            // The other kind refuses (kind is part of the entry contract).
            let err = validate_mesh_policy_binding(&policy, other_kind, Some(mesh.epoch()), 2)
                .unwrap_err();
            assert!(
                err.contains("expected a") && err.contains("execution policy"),
                "{err}"
            );
        }
    }

    #[test]
    fn minimax_mesh_entry_binding_refusals() {
        // Stale/different mesh epoch refuses: a second independently
        // constructed mesh — even with identical topology — has a different
        // epoch identity, so a policy bound to it refuses.
        let mesh = DeviceMesh::rect(&[(DimKind::Ep, 2)]);
        let policy = MoEExecutionPolicy::new(MoEExecutionKind::Ep, mesh.clone()).unwrap();
        let stale = DeviceMesh::rect(&[(DimKind::Ep, 2)]);
        let err =
            validate_mesh_policy_binding(&policy, MoEExecutionKind::Ep, Some(stale.epoch()), 2)
                .unwrap_err();
        assert!(err.contains("epoch differs"), "{err}");
        // Rank-count mismatch refuses.
        let err =
            validate_mesh_policy_binding(&policy, MoEExecutionKind::Ep, Some(mesh.epoch()), 1)
                .unwrap_err();
        assert!(err.contains("rank count"), "{err}");
        // Unbound Gpus (no from_mesh epoch) refuses — the binding is required
        // even when MoE is disabled / shared-only.
        let err = validate_mesh_policy_binding(&policy, MoEExecutionKind::Ep, None, 2).unwrap_err();
        assert!(err.contains("not bound"), "{err}");
        // Rank-one NAMED meshes pass (rank-one Tp AND Ep support preserved).
        for (kind, axis) in [
            (MoEExecutionKind::Tp, DimKind::Tp),
            (MoEExecutionKind::Ep, DimKind::Ep),
        ] {
            let mesh1 = DeviceMesh::rect(&[(axis, 1)]);
            let p1 = MoEExecutionPolicy::new(kind, mesh1.clone()).unwrap();
            validate_mesh_policy_binding(&p1, kind, Some(mesh1.epoch()), 1)
                .unwrap_or_else(|e| panic!("{kind:?}=1 rank-one named mesh must pass: {e}"));
        }
    }

    #[test]
    fn minimax_mesh_entry_authority_composes_validate_then_acquire() {
        // The production validate-then-acquire composition (`mesh_entry_authority`
        // → `mesh_entry_authority_core`, the seam BOTH mesh entries call at
        // their start): expected-kind / exact mesh / rank validation runs
        // FIRST, and the acquisition callback — the slot production fills with
        // the real `expert_manifest_for_policy` resolution — is reached ONLY
        // on a valid binding, exactly once. A refused policy must never reach
        // the acquisition.
        use std::cell::Cell;
        let calls = Cell::new(0usize);
        let acquire = || {
            calls.set(calls.get() + 1);
            Ok::<(), String>(())
        };

        // Wrong kind refuses BEFORE the acquisition.
        let mesh = DeviceMesh::rect(&[(DimKind::Ep, 2)]);
        let ep = MoEExecutionPolicy::new(MoEExecutionKind::Ep, mesh.clone()).unwrap();
        assert!(mesh_entry_authority_core(
            &ep,
            MoEExecutionKind::Tp,
            Some(mesh.epoch()),
            2,
            &acquire
        )
        .is_err());
        assert_eq!(calls.get(), 0, "wrong kind must not reach the acquisition");
        // Stale/different mesh epoch refuses BEFORE the acquisition.
        let stale = DeviceMesh::rect(&[(DimKind::Ep, 2)]);
        assert!(mesh_entry_authority_core(
            &ep,
            MoEExecutionKind::Ep,
            Some(stale.epoch()),
            2,
            &acquire
        )
        .is_err());
        assert_eq!(
            calls.get(),
            0,
            "stale/different mesh epoch must not reach the acquisition"
        );
        // Unbound Gpus refuses BEFORE the acquisition.
        assert!(mesh_entry_authority_core(&ep, MoEExecutionKind::Ep, None, 2, &acquire).is_err());
        assert_eq!(
            calls.get(),
            0,
            "unbound Gpus must not reach the acquisition"
        );
        // Rank-count mismatch refuses BEFORE the acquisition.
        assert!(mesh_entry_authority_core(
            &ep,
            MoEExecutionKind::Ep,
            Some(mesh.epoch()),
            1,
            &acquire
        )
        .is_err());
        assert_eq!(
            calls.get(),
            0,
            "rank mismatch must not reach the acquisition"
        );

        // Valid binding reaches the acquisition EXACTLY once.
        calls.set(0);
        mesh_entry_authority_core(&ep, MoEExecutionKind::Ep, Some(mesh.epoch()), 2, &acquire)
            .expect("correct kind + bound mesh must pass");
        assert_eq!(calls.get(), 1, "valid binding acquires exactly once");

        // Rank-one NAMED meshes pass the composition and acquire exactly once
        // (EP=1 alongside TP=1).
        for (kind, axis) in [
            (MoEExecutionKind::Tp, DimKind::Tp),
            (MoEExecutionKind::Ep, DimKind::Ep),
        ] {
            let mesh1 = DeviceMesh::rect(&[(axis, 1)]);
            let p1 = MoEExecutionPolicy::new(kind, mesh1.clone()).unwrap();
            calls.set(0);
            mesh_entry_authority_core(&p1, kind, Some(mesh1.epoch()), 1, &acquire)
                .unwrap_or_else(|e| panic!("{kind:?}=1 rank-one named mesh must pass: {e}"));
            assert_eq!(calls.get(), 1, "{kind:?}=1 must acquire exactly once");
        }
    }
}
