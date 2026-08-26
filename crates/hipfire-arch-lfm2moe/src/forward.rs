// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! LFM2.5-MoE forward pass (free functions — hot-path static dispatch).
//!
//! Per-layer pipeline (pre-norm; mixer = conv OR attention, FFN = dense OR MoE):
//!   tmp = operator_norm(h)
//!   if conv:   h += out_proj( C_gate ⊙ depthwise_causal_conv( B_gate ⊙ x ) )   [in_proj→conv→out_proj]
//!   if attn:   h += out_proj( attn( qk_norm(q/k) + full-RoPE, v ) )             [GQA, Q8 KV]
//!   ffn_tmp = ffn_norm(h)
//!   if dense:  h += w2( silu(w1·ffn_tmp) ⊙ (w3·ffn_tmp) )                        [SwiGLU, Q8]
//!   if moe:    h += combine( experts( sigmoid+bias top-4 route(ffn_tmp) ) )      [FWHT MQ4 experts]
//! then logits = lm_head( embedding_norm(h) )   (lm_head tied to embed_tokens).
//!
//! Non-expert linears (attention q/k/v/out, conv in/out, dense w1/w2/w3, router)
//! are Q8 (plain input). Routed experts are FWHT-pre-rotated MQ4G256: the input
//! is rotated (`rotate_x_mq_for`) and the silu output rotated
//! (`fused_silu_mul_rotate_mq_batched_for`) before the indexed-MoE GEMVs —
//! exactly qwen35's / minimax's MoE path, but with k_top = num_experts_per_tok
//! = 4 (the batched GEMV variants take k_top as a runtime arg).

use crate::config::Lfm2MoeConfig;
use crate::lfm2moe::{
    AttnWeights, ConvWeights, DenseFfn, Ffn, Lfm2MoeLayerWeights, Lfm2MoeState, Lfm2MoeWeights,
    Mixer, MoeFfn,
};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::moe::{ExpertExecutionPlan, RouterPlan};
use hipfire_dispatch::pipeline::superop::{
    self, ForwardBindings, OpBinding, OpFlavor, SuperOp, SuperOpKind, WeightSlot,
};
use hipfire_dispatch::pipeline::{
    execute_steps, GemvInput, MoeActivationVariant, MoeProj, ScoreActKind, Step,
};
use hipfire_dispatch::types::DispatchError;
use hipfire_runtime::llama::KvCacheExt;
use hipfire_runtime::llama::{rotate_x_mq_for, weight_gemv, weight_gemv_residual};
use hipfire_runtime::moe_plan::{
    execute_lowered_moe, lower_moe_steps, LoweredMoeProgram, MoEExecutionPolicy,
    MoeExecutionTarget, MoeProgramParts, RoutedMoeStepPhases,
};
use rdna_compute::{DType, Gpu};

/// Decode one token; returns the full logits vector.
///
/// Routes to the retained replay path when the Redline controller is enabled
/// (dense-only) — otherwise falls back to the hipGraph path when
/// `HIPFIRE_LFM2_GRAPH=1` (default OFF → exact prior behavior). Retained takes
/// precedence over HipGraph. Dense-only: non-dense models poison/disable the
/// controller and continue via the existing HIP path without falsely claiming
/// replay. Any decode while `retained_state_poisoned` is true rejects until
/// `reset()` clears it.
pub fn decode_step(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    if state.retained_state_poisoned {
        return Err("lfm2moe: retained state poisoned until reset".to_string());
    }
    if gpu.replay.is_enabled() {
        if !cfg.is_dense() {
            gpu.replay.set_forward_eligible(false);
            gpu.replay
                .poison("LFM retained replay requires dense model");
        } else {
            gpu.replay.set_forward_eligible(true);
            return decode_step_with_retained_replay(cfg, weights, state, gpu, token_id, position);
        }
    }
    if graph_enabled() {
        return decode_step_with_graph(cfg, weights, state, gpu, token_id, position);
    }
    decode_step_inner(cfg, weights, state, gpu, token_id, position, None)?;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("lfm2moe: download logits: {e:?}"))
}

fn decode_step_with_retained_replay(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    if state.retained_state_poisoned {
        return Err("lfm2moe: retained state poisoned until reset".to_string());
    }
    if !cfg.is_dense() {
        gpu.replay.set_forward_eligible(false);
        gpu.replay
            .poison("LFM retained replay requires dense model");
        decode_step_inner(cfg, weights, state, gpu, token_id, position, None)?;
        return gpu
            .download_f32(&state.logits)
            .map_err(|e| format!("lfm2moe: download logits (non-dense fallback): {e:?}"));
    }
    gpu.replay.set_forward_eligible(true);
    if !state.retained_warmed_up {
        decode_step_inner(cfg, weights, state, gpu, token_id, position, None)?;
        state.retained_warmed_up = true;
        return gpu
            .download_f32(&state.logits)
            .map_err(|e| format!("lfm2moe: download logits (retained warmup): {e:?}"));
    }
    prepare_retained_decode_inputs(cfg, weights, state, gpu, token_id, position)?;
    if !gpu.replay.should_route_aql() && !gpu.replay.should_route_pm4() {
        let _ = gpu.replay.begin_auto_capture_if_armed();
        if gpu.replay.state() == rdna_compute::replay::ReplayState::Armed {
            let _ = gpu.replay.begin_capture();
        }
    }
    if gpu.replay.should_route_aql() || gpu.replay.should_route_pm4() {
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("lfm2moe: retained adapter sync: {e:?}"))?;
        let replay_result = if gpu.replay.should_route_aql() {
            unsafe { gpu.replay.replay_linear_aql(position as usize) }.map(|_| ())
        } else {
            unsafe { gpu.replay.replay_pm4(position as usize) }.map(|_| ())
        };
        match replay_result {
            Ok(_) => {
                state.n_tokens = position as usize + 1;
                return gpu
                    .download_f32(&state.logits)
                    .map_err(|e| format!("lfm2moe: download logits (retained replay): {e:?}"));
            }
            Err(reason) => {
                let msg = format!("LFM retained replay failed: {reason}");
                gpu.replay.poison(msg);
                state.retained_state_poisoned = true;
                return Err(reason);
            }
        }
    }
    run_retained_decode_body(cfg, weights, state, gpu, position)?;
    if gpu.replay.should_auto_finalize_capture() {
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("lfm2moe: retained capture sync: {e:?}"))?;
        let capture = gpu
            .replay
            .finish_capture()
            .map_err(|e| format!("lfm2moe: finish capture: {e}"))?;
        let launches = gpu.replay.recorded_launches().len();
        let prepare = if gpu.replay.uses_pm4_transport() {
            gpu.replay
                .prepare_pm4_prefix(gpu.device_id as usize, launches)
                .map(|_| ())
        } else {
            gpu.replay
                .prepare_linear_aql_prefix(gpu.device_id as usize, launches)
                .map(|_| ())
        };
        match prepare {
            Ok(()) => {
                eprintln!(
                    "[LFM2.5-MoE redline] retained route ready: capture={capture:?} identity={:?}",
                    gpu.replay.prepared_route_identity()
                );
            }
            Err(reason) => {
                gpu.replay
                    .poison(format!("LFM Redline prepare after warmup failed: {reason}"));
                eprintln!("[LFM2.5-MoE redline] falling back to HIP: {reason}");
            }
        }
    } else if gpu.replay.state() == rdna_compute::replay::ReplayState::RecordingWarmup {
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("lfm2moe: retained capture sync: {e:?}"))?;
        if let Ok(capture) = gpu.replay.finish_capture() {
            let launches = gpu.replay.recorded_launches().len();
            let prepare = if gpu.replay.uses_pm4_transport() {
                gpu.replay
                    .prepare_pm4_prefix(gpu.device_id as usize, launches)
                    .map(|_| ())
            } else {
                // Manual capture needs at least 2 dispatches for linear AQL.
                if launches >= 2 {
                    gpu.replay
                        .prepare_linear_aql_prefix(gpu.device_id as usize, launches)
                        .map(|_| ())
                } else {
                    Err("no captured launch sequence".to_owned())
                }
            };
            match prepare {
                Ok(()) => eprintln!(
                    "[LFM2.5-MoE redline] retained route ready (manual): capture={capture:?} identity={:?}",
                    gpu.replay.prepared_route_identity()
                ),
                Err(reason) => {
                    gpu.replay
                        .poison(format!("LFM Redline manual prepare failed: {reason}"));
                    eprintln!("[LFM2.5-MoE redline] falling back to HIP: {reason}");
                }
            }
        }
    }
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("lfm2moe: download logits (retained capture): {e:?}"))
}

/// `HIPFIRE_LFM2_GRAPH=1` opt-in switch. Default OFF (unset / "0") →
/// byte-identical to the legacy per-launch decode path. Parsed once.
fn graph_enabled() -> bool {
    use std::sync::OnceLock;
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| {
        matches!(
            hipfire_config::developer_var("HIPFIRE_LFM2_GRAPH")
                .ok()
                .as_deref(),
            Some("1")
        )
    })
}

/// Decode one token, appending each layer's post-residual hidden state
/// (after the full layer, before the final norm) to `capture[layer]` — used by
/// the oracle dumper. Set `HIPFIRE_LFM2_CAPTURE_POSTMIXER` to capture the
/// post-mixer residual (pre-FFN) instead, for conv/attn-vs-FFN localization.
pub fn decode_step_capture(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    capture: &mut [Vec<f32>],
) -> Result<(), String> {
    decode_step_inner(cfg, weights, state, gpu, token_id, position, Some(capture))
}

#[doc(hidden)]
pub fn prepare_retained_decode_inputs(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<(), String> {
    if (position as usize) >= state.max_seq {
        return Err(format!(
            "lfm2moe: position {} >= max_seq {}",
            position, state.max_seq
        ));
    }
    // Pre-GPU refusal gate: acquire the manifest authority and lower/validate
    // every reachable MoE layer program (cached per model/config) BEFORE the
    // position H2D / embedding lookup — the first GPU ops of a decode step.
    // Any manifest, protocol, dtype, or capacity refusal surfaces here, so a
    // rejected step mutates no GPU state (KV / conv-state / residual). Dense
    // models short-circuit to Ok.
    weights
        .moe_prevalidated(cfg, state)
        .map_err(|e| format!("lfm2moe: MoE prevalidation refused before GPU work: {e}"))?;
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("lfm2moe: htod pos: {e:?}"))?;
    hipfire_runtime::llama::embedding_lookup_dispatch(
        gpu,
        weights.embd_format,
        &weights.embed,
        &state.h,
        token_id,
        cfg.hidden_size,
    )
    .map_err(|e| format!("lfm2moe: embed lookup: {e:?}"))?;
    Ok(())
}

#[doc(hidden)]
pub fn run_retained_decode_body(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    position: u32,
) -> Result<(), String> {
    decode_step_layers_and_head(cfg, weights, state, gpu, position, None)
}

fn decode_step_inner(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    if capture.is_none() {
        prepare_retained_decode_inputs(cfg, weights, state, gpu, token_id, position)?;
        return run_retained_decode_body(cfg, weights, state, gpu, position);
    }
    // Oracle-capture path: still staged through the same prepare helper for
    // bounds-check / single staging convention, but body must carry capture.
    prepare_retained_decode_inputs(cfg, weights, state, gpu, token_id, position)?;
    decode_step_layers_and_head(cfg, weights, state, gpu, position, capture)
}

/// Per-layer mixer/FFN stack + final norm + lm_head. Reads the residual
/// stream `state.h` (already seeded by the embedding lookup) and the device
/// position scalar `state.pos_buf` (already staged); writes `state.logits`.
///
/// This is the hipGraph-captureable region: it issues only kernel launches
/// that read STABLE device buffers and (on the MoE path) compute their
/// topk/positions on-device, so a single capture replays correctly at every
/// later position once `state.pos_buf` is refreshed. The per-token-varying
/// embedding lookup (token_id is a kernarg) and the `pos_buf` htod are the
/// caller's responsibility OUTSIDE the captured region.
///
/// `capture` (oracle dumper) is incompatible with hipGraph capture — it issues
/// a sync `download_f32` per layer. The graph path always passes `None`.
fn decode_step_layers_and_head(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    position: u32,
    mut capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let eps = cfg.rms_norm_eps;
    let seq_len = position as usize + 1;
    let capture_postmixer =
        hipfire_config::developer_var_os("HIPFIRE_LFM2_CAPTURE_POSTMIXER").is_some();

    // #397 Ship 6 — forward-as-pipeline. HIPFIRE_FORWARD_LOWERED=1 routes the
    // per-layer decode through the super-op executor (run_layer_program). Skipped
    // when capturing (the oracle dumper needs the per-layer hand path) — that path
    // stays byte-identical. Default off (opt-in) until fleet byte-parity validated.
    if lfm2_forward_lowered_enabled() && capture.is_none() {
        return decode_step_layers_and_head_lowered(cfg, weights, state, gpu, position);
    }

    for (l, layer) in weights.layers.iter().enumerate() {
        // ── Mixer block (pre-norm) ──────────────────────────────────────────
        gpu.rmsnorm_f32(&state.h, &layer.operator_norm, &state.tmp, eps)
            .map_err(|e| format!("lfm2moe L{l}: operator rmsnorm: {e:?}"))?;

        match &layer.mixer {
            Mixer::Conv(c) => {
                // in_proj → [3*hidden] (B | C_gate | x), Q8 plain.
                weight_gemv(gpu, &c.in_proj, &state.tmp, &state.conv_bcx)
                    .map_err(|e| format!("lfm2moe L{l}: conv in_proj: {e}"))?;
                // double-gated depthwise causal short-conv (advances conv state).
                gpu.conv1d_gated_decode_f32(
                    &state.conv_bcx,
                    &state.conv_states[c.conv_state_idx],
                    &c.conv_weight,
                    &state.conv_y,
                    1,
                    hidden,
                    cfg.conv_kernel_size,
                )
                .map_err(|e| format!("lfm2moe L{l}: conv gated decode: {e:?}"))?;
                // out_proj + residual: h += W_out · y (Q8).
                weight_gemv_residual(gpu, &c.out_proj, &state.conv_y, &state.h)
                    .map_err(|e| format!("lfm2moe L{l}: conv out_proj: {e}"))?;
            }
            Mixer::Attention(a) => {
                weight_gemv(gpu, &a.wq, &state.tmp, &state.fa_q)
                    .map_err(|e| format!("lfm2moe L{l}: q_proj: {e}"))?;
                weight_gemv(gpu, &a.wk, &state.tmp, &state.fa_k)
                    .map_err(|e| format!("lfm2moe L{l}: k_proj: {e}"))?;
                weight_gemv(gpu, &a.wv, &state.tmp, &state.fa_v)
                    .map_err(|e| format!("lfm2moe L{l}: v_proj: {e}"))?;

                // Per-HEAD QK-norm: RMSNorm over each head's head_dim slice,
                // sharing the [head_dim] weight across heads (batch = n_heads).
                gpu.rmsnorm_batched(&state.fa_q, &a.q_norm, &state.fa_q, n_heads, head_dim, eps)
                    .map_err(|e| format!("lfm2moe L{l}: q_norm: {e:?}"))?;
                gpu.rmsnorm_batched(&state.fa_k, &a.k_norm, &state.fa_k, n_kv, head_dim, eps)
                    .map_err(|e| format!("lfm2moe L{l}: k_norm: {e:?}"))?;

                // Full-dim rotate_half RoPE (no partial rotary).
                gpu.rope_f32(
                    &state.fa_q,
                    &state.fa_k,
                    &state.pos_buf,
                    n_heads,
                    n_kv,
                    head_dim,
                    cfg.rope_theta,
                )
                .map_err(|e| format!("lfm2moe L{l}: rope: {e:?}"))?;

                // KV cache write (Q8) + GQA flash attention.
                let kv_idx = a.kv_idx;
                gpu.kv_cache_write_q8_0(
                    &state.kv.k_gpu[kv_idx],
                    &state.fa_k,
                    &state.pos_buf,
                    n_kv,
                    head_dim,
                )
                .map_err(|e| format!("lfm2moe L{l}: kv write k: {e:?}"))?;
                gpu.kv_cache_write_q8_0(
                    &state.kv.v_gpu[kv_idx],
                    &state.fa_v,
                    &state.pos_buf,
                    n_kv,
                    head_dim,
                )
                .map_err(|e| format!("lfm2moe L{l}: kv write v: {e:?}"))?;
                gpu.attention_q8_0_kv(
                    &state.fa_q,
                    &state.kv.k_gpu[kv_idx],
                    &state.kv.v_gpu[kv_idx],
                    &state.fa_attn_out,
                    &state.pos_buf,
                    seq_len,
                    n_heads,
                    n_kv,
                    head_dim,
                    state.kv.physical_cap,
                )
                .map_err(|e| format!("lfm2moe L{l}: attention: {e:?}"))?;

                // out_proj + residual: h += W_out · attn_out (Q8).
                weight_gemv_residual(gpu, &a.wo, &state.fa_attn_out, &state.h)
                    .map_err(|e| format!("lfm2moe L{l}: out_proj: {e}"))?;
            }
        }

        if capture_postmixer {
            if let Some(cap) = capture.as_deref_mut() {
                let h = gpu
                    .download_f32(&state.h)
                    .map_err(|e| format!("lfm2moe L{l}: postmixer capture: {e:?}"))?;
                cap[l].extend_from_slice(&h);
            }
        }

        // ── FFN block (pre-norm): dense SwiGLU OR top-4 MoE ─────────────────
        gpu.rmsnorm_f32(&state.h, &layer.ffn_norm, &state.ffn_tmp, eps)
            .map_err(|e| format!("lfm2moe L{l}: ffn rmsnorm: {e:?}"))?;

        match &layer.ffn {
            Ffn::Dense(d) => {
                weight_gemv(gpu, &d.w1, &state.ffn_tmp, &state.dense_gate)
                    .map_err(|e| format!("lfm2moe L{l}: dense w1: {e}"))?;
                weight_gemv(gpu, &d.w3, &state.ffn_tmp, &state.dense_up)
                    .map_err(|e| format!("lfm2moe L{l}: dense w3: {e}"))?;
                gpu.silu_mul_f32(&state.dense_gate, &state.dense_up, &state.dense_act)
                    .map_err(|e| format!("lfm2moe L{l}: dense silu_mul: {e:?}"))?;
                weight_gemv_residual(gpu, &d.w2, &state.dense_act, &state.h)
                    .map_err(|e| format!("lfm2moe L{l}: dense w2: {e}"))?;
            }
            Ffn::Moe(m) => {
                // Post-norm MoE FFN through the sealed Single executor — the
                // same shared route the #397 Ship 6 super-op path uses
                // (run_lowered_moe_ffn).
                run_lowered_moe_ffn(gpu, cfg, weights, m, state, l)?;
            }
        }

        // Capture post-layer residual (pre final-norm) for the oracle compare.
        if !capture_postmixer {
            if let Some(cap) = capture.as_deref_mut() {
                let h = gpu
                    .download_f32(&state.h)
                    .map_err(|e| format!("lfm2moe L{l}: capture download: {e:?}"))?;
                cap[l].extend_from_slice(&h);
            }
        }
    }
    state.n_tokens = seq_len;

    // Final RMSNorm + lm_head (tied to embed_tokens, Q8).
    gpu.rmsnorm_f32(
        &state.h,
        &weights.embedding_norm,
        &state.final_norm_buf,
        eps,
    )
    .map_err(|e| format!("lfm2moe: final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
        .map_err(|e| format!("lfm2moe: lm_head: {e}"))?;
    Ok(())
}

/// Per-layer expert refs for the sealed indexed route. The gate_up ref
/// carries the gate_up dtype and the down ref the down dtype — the indexed
/// dispatch selects the HFQ4 vs HFQ6 kernel per ref. The caller keeps both
/// alive across `lower_lfm2_moe_program` + `execute_lowered_moe`.
fn lfm2_moe_expert_refs<'step>(
    cfg: &Lfm2MoeConfig,
    m: &'step MoeFfn,
) -> (
    hipfire_dispatch::families::moe::MoeExpertRef<'step>,
    hipfire_dispatch::families::moe::MoeExpertRef<'step>,
) {
    let hidden = cfg.hidden_size;
    let moe_inter = cfg.moe_intermediate_size;
    let n_exp = cfg.num_experts;
    (
        hipfire_dispatch::families::moe::MoeExpertRef {
            gate_up_ptrs: &m.expert_gate_up_ptrs,
            down_ptrs: &m.expert_down_ptrs,
            dummy_gate_up: None,
            dtype: m.experts[0].gate_up.gpu_dtype,
            n_experts: n_exp,
            expert_m: moe_inter,
            expert_k: hidden,
            owned: &[],
        },
        hipfire_dispatch::families::moe::MoeExpertRef {
            gate_up_ptrs: &m.expert_gate_up_ptrs,
            down_ptrs: &m.expert_down_ptrs,
            dummy_gate_up: None,
            dtype: m.experts[0].down.gpu_dtype,
            n_experts: n_exp,
            expert_m: moe_inter,
            expert_k: hidden,
            owned: &[],
        },
    )
}

/// Build the sealed Single MoE program for one MoE layer: the exact
/// MoeProgramParts/routed phases the decode executes (sigmoid → bias-aware
/// top-k → gate_up → silu·mul·rotate → down → combine), lowered through the
/// manifest-born ExpertGroupPlan via lower_moe_steps. Pure host-side — reads
/// device-buffer sizes only for capacity validation. Refuses per-expert
/// dtypes outside the indexed set (MQ4G256/HFQ4G256/MQ6G256/HFQ6G256 — the
/// loader-guard set) with a clear error BEFORE any GPU work when reached via
/// `prevalidate_lfm2_moe`, never misdispatched to a wrong dequant at
/// execution. The kernels are the same indexed-MoE family the direct
/// sequence launched (MQ4→HFQ4, MQ6→HFQ6 by the expert refs' dtype); the
/// combine accumulates into the residual (the Single executor never zeroes
/// `out`). Single executor only — safe inside graph capture and the retained
/// capture region.
fn lower_lfm2_moe_program<'mesh, 'step>(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    m: &'step MoeFfn,
    gu_ref: &'step hipfire_dispatch::families::moe::MoeExpertRef<'step>,
    dn_ref: &'step hipfire_dispatch::families::moe::MoeExpertRef<'step>,
    state: &'step Lfm2MoeState,
    l: usize,
    policy: &'mesh MoEExecutionPolicy,
) -> Result<LoweredMoeProgram<'mesh, 'step>, String> {
    let hidden = cfg.hidden_size;
    let moe_inter = cfg.moe_intermediate_size;
    let n_exp = cfg.num_experts;
    let k_top = cfg.num_experts_per_tok;

    // Dtype eligibility gate for the sealed indexed route — mirrors the
    // loader guard and the indexed dispatch (MQ4→HFQ4, MQ6→HFQ6 kernels).
    // Anything outside the set is refused here (before any GPU work when
    // reached via prevalidation), never misdispatched at execution.
    for (label, dtype) in [
        ("gate_up", m.experts[0].gate_up.gpu_dtype),
        ("down", m.experts[0].down.gpu_dtype),
    ] {
        if !matches!(
            dtype,
            DType::MQ4G256 | DType::HFQ4G256 | DType::MQ6G256 | DType::HFQ6G256
        ) {
            return Err(format!(
                "lfm2moe L{l}: unsupported MoE expert {label} dtype {dtype:?} — \
                 the sealed indexed route requires MQ4G256/HFQ4G256/MQ6G256/HFQ6G256"
            ));
        }
    }

    let phases = RoutedMoeStepPhases {
        router: vec![
            Step::ScoreActivation {
                scores: &state.router_logits,
                kind: ScoreActKind::Sigmoid,
            },
            Step::MoeRoute {
                scores: &state.router_logits,
                gate_bias: &m.expert_bias,
                topk_indices: &state.topk_indices,
                topk_weights: &state.topk_weights,
                k: k_top,
                n_experts: n_exp,
                route_scale: cfg.routed_scaling_factor,
            },
        ],
        gate_up: vec![Step::IndexedMoeGemv {
            experts: &gu_ref,
            which: MoeProj::GateUp {
                up_out: &state.up_batch,
            },
            topk_indices: &state.topk_indices,
            input: GemvInput::Prerotated(&state.ffn_x_rot),
            out: &state.gate_batch,
            k_top,
            batch_size: 1,
        }],
        activation: vec![Step::MoeActivation {
            variant: MoeActivationVariant::MinimaxFused {
                awq_scale: m.experts[0].down.awq_scale.as_ref(),
            },
            gate: &state.gate_batch,
            up: &state.up_batch,
            rot_out: &state.rot_batch,
            inter: moe_inter,
            k_top,
        }],
        down: vec![Step::IndexedMoeGemv {
            experts: &dn_ref,
            which: MoeProj::DownExpanded,
            topk_indices: &state.topk_indices,
            input: GemvInput::Prerotated(&state.rot_batch),
            out: &state.down_expanded,
            k_top,
            batch_size: 1,
        }],
        combine: vec![Step::MoeCombine {
            down_out: &state.down_expanded,
            topk_weights: &state.topk_weights,
            out: &state.h,
            k: k_top,
            hidden,
            batch_size: 1,
            inverse_perm: None,
        }],
        finish: Vec::new(),
    };
    let parts = MoeProgramParts {
        router: RouterPlan::SigmoidTopK {
            scores: &state.router_logits,
            topk_indices: &state.topk_indices,
            topk_weights: &state.topk_weights,
            k_top,
            normalize: true,
            route_scale: cfg.routed_scaling_factor,
        },
        execution: ExpertExecutionPlan::IndexedQuantized,
        deferred_combine: false,

        ranks: vec![phases],
    };
    let plan = weights
        .moe_group_plans(cfg)
        .map_err(|e| format!("lfm2moe L{l}: expert-group plan: {e}"))?
        .by_layer(l)?;
    lower_moe_steps(plan, policy, parts)
        .map_err(|e| format!("lfm2moe L{l}: lower_moe_steps: {e:?}"))
}

/// Post-norm top-4 MoE FFN through the sealed Single executor — shared by the
/// hand loop (decode_step_layers_and_head) and the #397 Ship 6 super-op path
/// (moe_ffn_block). Requires the caller to have already written the ffn-normed
/// input to `state.ffn_tmp`; this helper runs the router GEMV, FWHT-rotates
/// the input for the MQ4/MQ6 experts, then executes the sealed program built
/// by `lower_lfm2_moe_program` (prevalidated at the decode entry; Single
/// executor only — safe inside graph capture and the retained capture
/// region).
fn run_lowered_moe_ffn(
    gpu: &mut Gpu,
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    m: &MoeFfn,
    state: &Lfm2MoeState,
    l: usize,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let w_r = m.router.dispatch_ref();
    let ctx = DispatchCtx::new(gpu);
    // Router: sigmoid(logits) + bias-aware top-k (gather unbiased,
    // renormalize, scale). expert_bias steers SELECTION only. The
    // Q8 router reads the PLAIN normed input (no rotation); the
    // FWHT-rotated MQ4/MQ6 experts rotate via the direct
    // rotate_x_mq_for below (same split as minimax).
    execute_steps(
        gpu,
        &ctx,
        &[Step::Gemv {
            w: &w_r,
            input: GemvInput::Raw(&state.ffn_tmp),
            out: &state.router_logits,
        }],
    )
    .map_err(|e| format!("lfm2moe L{l}: router: {e:?}"))?;

    // FWHT-rotate the FFN input for the MQ4 experts (router stays plain).
    rotate_x_mq_for(
        gpu,
        &m.experts[0].gate_up,
        &state.ffn_tmp,
        &state.ffn_x_rot,
        hidden,
    )
    .map_err(|e| format!("lfm2moe L{l}: ffn rotate: {e:?}"))?;

    let (gu_ref, dn_ref) = lfm2_moe_expert_refs(cfg, m);
    let policy = MoEExecutionPolicy::single();
    let program = lower_lfm2_moe_program(cfg, weights, m, &gu_ref, &dn_ref, state, l, &policy)?;
    execute_lowered_moe(&program, MoeExecutionTarget::Single { gpu, ctx: &ctx })
        .map_err(|e| format!("lfm2moe L{l}: execute_lowered_moe: {e:?}"))
}

/// Manifest authority + per-layer sealed-lowering prevalidation for every
/// reachable MoE layer. Pure host-side (reads device-buffer sizes only for
/// capacity validation): acquires the manifest-born ExpertGroupPlans and
/// lowers each MoE layer's exact program through `lower_moe_steps`, so any
/// manifest, protocol, dtype, or capacity refusal surfaces BEFORE a decode
/// step performs its first GPU op (position H2D / embedding lookup / KV
/// write / conv-state mutation). Dense-only models short-circuit. The decode
/// then executes the identical program per layer, exactly once.
pub(crate) fn prevalidate_lfm2_moe(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &Lfm2MoeState,
) -> Result<(), String> {
    if cfg.num_experts == 0 {
        return Ok(());
    }
    let policy = MoEExecutionPolicy::single();
    for (l, layer) in weights.layers.iter().enumerate() {
        if let Ffn::Moe(m) = &layer.ffn {
            let (gu_ref, dn_ref) = lfm2_moe_expert_refs(cfg, m);
            let _program =
                lower_lfm2_moe_program(cfg, weights, m, &gu_ref, &dn_ref, state, l, &policy)?;
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// #397 Ship 6 — forward-as-pipeline: LFM2.5 lowered decode (the run_conv slot).
//
// LFM2 is the substrate's Conv super-op proving ground. Each layer lowers to a
// short LayerProgram of coarse super-ops; the per-token executor (run_layer_
// program) calls these arch handlers. ADDITIVE + opt-in (HIPFIRE_FORWARD_LOWERED,
// default off) — the hand loop in decode_step_layers_and_head is untouched, so
// the default path stays byte-identical; the lowered path is validated byte-
// identical via the FORWARD_LOWERED=0-vs-=1 committed-token md5 A/B before flip.
//
// Super-op map (pre-norm folded into each handler):
//   Conv         = operator_norm + in_proj + conv1d_gated + out_proj(+resid)
//   Attend       = operator_norm + q/k/v + qk_norm + rope + kv + attn + o(+resid)
//   Proj(GU)     = ffn_norm + w1 + w3            ResidualGemv(DOWN) = silu·mul + w2(+resid)
//   Moe          = ffn_norm + rotate + router + top-k + experts + combine
// ─────────────────────────────────────────────────────────────────────────

/// Conv mixer block (operator-norm folded in). Mirrors the hand-loop Conv arm.
fn conv_mixer_block(
    gpu: &mut Gpu,
    cfg: &Lfm2MoeConfig,
    op_norm: &rdna_compute::GpuTensor,
    c: &ConvWeights,
    state: &Lfm2MoeState,
    l: usize,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    gpu.rmsnorm_f32(&state.h, op_norm, &state.tmp, cfg.rms_norm_eps)
        .map_err(|e| format!("lfm2moe L{l}: operator rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &c.in_proj, &state.tmp, &state.conv_bcx)
        .map_err(|e| format!("lfm2moe L{l}: conv in_proj: {e}"))?;
    gpu.conv1d_gated_decode_f32(
        &state.conv_bcx,
        &state.conv_states[c.conv_state_idx],
        &c.conv_weight,
        &state.conv_y,
        1,
        hidden,
        cfg.conv_kernel_size,
    )
    .map_err(|e| format!("lfm2moe L{l}: conv gated decode: {e:?}"))?;
    weight_gemv_residual(gpu, &c.out_proj, &state.conv_y, &state.h)
        .map_err(|e| format!("lfm2moe L{l}: conv out_proj: {e}"))
}

/// Attention mixer block (operator-norm folded in). Mirrors the hand-loop Attn arm.
fn attn_mixer_block(
    gpu: &mut Gpu,
    cfg: &Lfm2MoeConfig,
    op_norm: &rdna_compute::GpuTensor,
    a: &AttnWeights,
    state: &Lfm2MoeState,
    l: usize,
    seq_len: usize,
) -> Result<(), String> {
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let eps = cfg.rms_norm_eps;
    gpu.rmsnorm_f32(&state.h, op_norm, &state.tmp, eps)
        .map_err(|e| format!("lfm2moe L{l}: operator rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &a.wq, &state.tmp, &state.fa_q)
        .map_err(|e| format!("lfm2moe L{l}: q_proj: {e}"))?;
    weight_gemv(gpu, &a.wk, &state.tmp, &state.fa_k)
        .map_err(|e| format!("lfm2moe L{l}: k_proj: {e}"))?;
    weight_gemv(gpu, &a.wv, &state.tmp, &state.fa_v)
        .map_err(|e| format!("lfm2moe L{l}: v_proj: {e}"))?;
    gpu.rmsnorm_batched(&state.fa_q, &a.q_norm, &state.fa_q, n_heads, head_dim, eps)
        .map_err(|e| format!("lfm2moe L{l}: q_norm: {e:?}"))?;
    gpu.rmsnorm_batched(&state.fa_k, &a.k_norm, &state.fa_k, n_kv, head_dim, eps)
        .map_err(|e| format!("lfm2moe L{l}: k_norm: {e:?}"))?;
    gpu.rope_f32(
        &state.fa_q,
        &state.fa_k,
        &state.pos_buf,
        n_heads,
        n_kv,
        head_dim,
        cfg.rope_theta,
    )
    .map_err(|e| format!("lfm2moe L{l}: rope: {e:?}"))?;
    let kv_idx = a.kv_idx;
    // KV write (Q8) + attention via the shared KV-usage abstraction. lfm2moe is
    // Q8 non-flash unconditional → derive's q8_attend_key returns AttnQ8_0Kv at
    // pos+1<=15000 (byte-identical; needs no partials, hence flash_partials:
    // None). It flips to AttnFlashQ8_0 at pos+1>15000 (the documented
    // Q8-fidelity edge — rare for this decode model). capture_mode is NOT
    // threaded: the non-flash kernel is capture-safe and lfm2moe captures it.
    let ctx = hipfire_dispatch::context::DispatchCtx::new(gpu);
    let plan = hipfire_dispatch::families::kv_tier::KvTierPlan::derive(
        hipfire_dispatch::families::kv_tier::KvTierInputs {
            pos: seq_len - 1,
            ..state.kv.tier_inputs()
        },
    )
    .map_err(|e| format!("lfm2moe L{l}: kv tier: {e}"))?;
    let io = hipfire_dispatch::families::attention::AttnParams {
        q: &state.fa_q,
        k: &state.fa_k,
        v: &state.fa_v,
        k_cache: &state.kv.k_gpu[kv_idx],
        v_cache: &state.kv.v_gpu[kv_idx],
        k_scales: None,
        v_scales: None,
        pos_buf: &state.pos_buf,
        pos: seq_len - 1,
        positions: None,
        n_heads,
        n_kv_heads: n_kv,
        head_dim,
        physical_cap: state.kv.physical_cap,
        batch_size: 1,
        max_ctx_len: 0,
        flash_partials: None,
        givens_cos: None,
        givens_sin: None,
        tree_bias: None,
        block_start: 0,
        block_cols: 0,
        output_gate: None,
        output: &state.fa_attn_out,
    };
    hipfire_dispatch::pipeline::execute_steps(
        gpu,
        &ctx,
        &[hipfire_dispatch::pipeline::Step::Attend { plan, io }],
    )
    .map_err(|e| format!("lfm2moe L{l}: attention: {e:?}"))?;
    weight_gemv_residual(gpu, &a.wo, &state.fa_attn_out, &state.h)
        .map_err(|e| format!("lfm2moe L{l}: out_proj: {e}"))
}

/// Dense FFN gate/up half (ffn-norm folded in). Mirrors the hand-loop Dense head.
fn dense_gate_up_block(
    gpu: &mut Gpu,
    cfg: &Lfm2MoeConfig,
    ffn_norm: &rdna_compute::GpuTensor,
    d: &DenseFfn,
    state: &Lfm2MoeState,
    l: usize,
) -> Result<(), String> {
    gpu.rmsnorm_f32(&state.h, ffn_norm, &state.ffn_tmp, cfg.rms_norm_eps)
        .map_err(|e| format!("lfm2moe L{l}: ffn rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &d.w1, &state.ffn_tmp, &state.dense_gate)
        .map_err(|e| format!("lfm2moe L{l}: dense w1: {e}"))?;
    weight_gemv(gpu, &d.w3, &state.ffn_tmp, &state.dense_up)
        .map_err(|e| format!("lfm2moe L{l}: dense w3: {e}"))
}

/// Dense FFN down half (silu·mul + w2 residual). Mirrors the hand-loop Dense tail.
fn dense_down_block(
    gpu: &mut Gpu,
    d: &DenseFfn,
    state: &Lfm2MoeState,
    l: usize,
) -> Result<(), String> {
    gpu.silu_mul_f32(&state.dense_gate, &state.dense_up, &state.dense_act)
        .map_err(|e| format!("lfm2moe L{l}: dense silu_mul: {e:?}"))?;
    weight_gemv_residual(gpu, &d.w2, &state.dense_act, &state.h)
        .map_err(|e| format!("lfm2moe L{l}: dense w2: {e}"))
}

/// MoE FFN block (ffn-norm folded in). Mirrors the hand-loop Moe arm: norm,
/// then the sealed lowered MoE program (run_lowered_moe_ffn).
fn moe_ffn_block(
    gpu: &mut Gpu,
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    ffn_norm: &rdna_compute::GpuTensor,
    m: &MoeFfn,
    state: &Lfm2MoeState,
    l: usize,
) -> Result<(), String> {
    gpu.rmsnorm_f32(&state.h, ffn_norm, &state.ffn_tmp, cfg.rms_norm_eps)
        .map_err(|e| format!("lfm2moe L{l}: ffn rmsnorm: {e:?}"))?;
    run_lowered_moe_ffn(gpu, cfg, weights, m, state, l)
}

/// lfm2-local super-op opcodes (encoded in OpBinding.weights[0]).
mod lfm2_op {
    pub const DENSE_GATE_UP: u32 = 0;
    pub const DENSE_DOWN: u32 = 1;
}

/// The four lfm2 decoder-layer shapes (mixer × FFN). Pure → unit-testable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lfm2Variant {
    ConvDense,
    ConvMoe,
    AttnDense,
    AttnMoe,
}

fn lfm2_variant_of(layer: &Lfm2MoeLayerWeights) -> Lfm2Variant {
    match (&layer.mixer, &layer.ffn) {
        (Mixer::Conv(_), Ffn::Dense(_)) => Lfm2Variant::ConvDense,
        (Mixer::Conv(_), Ffn::Moe(_)) => Lfm2Variant::ConvMoe,
        (Mixer::Attention(_), Ffn::Dense(_)) => Lfm2Variant::AttnDense,
        (Mixer::Attention(_), Ffn::Moe(_)) => Lfm2Variant::AttnMoe,
    }
}

#[inline]
fn lfm2_superop(kind: SuperOpKind, code: u32) -> SuperOp {
    SuperOp {
        kind,
        binding: OpBinding {
            key: None,
            weights: vec![WeightSlot(code)],
            scratch: Vec::new(),
            flavor: OpFlavor::None,
        },
    }
}

/// Lower one lfm2 decoder layer to a coarse super-op LayerProgram (mirrors the
/// hand-loop order: mixer block, then FFN). Pure (no GpuTensor) → unit-testable.
fn lfm2_lower_variant(v: Lfm2Variant) -> superop::LayerProgram {
    use lfm2_op::{DENSE_DOWN, DENSE_GATE_UP};
    use SuperOpKind::{Attend, Conv, Moe, Proj, ResidualGemv};
    match v {
        Lfm2Variant::ConvDense => vec![
            lfm2_superop(Conv, 0),
            lfm2_superop(Proj, DENSE_GATE_UP),
            lfm2_superop(ResidualGemv, DENSE_DOWN),
        ],
        Lfm2Variant::AttnDense => vec![
            lfm2_superop(Attend, 0),
            lfm2_superop(Proj, DENSE_GATE_UP),
            lfm2_superop(ResidualGemv, DENSE_DOWN),
        ],
        Lfm2Variant::ConvMoe => vec![lfm2_superop(Conv, 0), lfm2_superop(Moe, 0)],
        Lfm2Variant::AttnMoe => vec![lfm2_superop(Attend, 0), lfm2_superop(Moe, 0)],
    }
}

/// Per-layer execution context for the lowered decode path (rebuilt each layer).
struct Lfm2MoeBindings<'a> {
    cfg: &'a Lfm2MoeConfig,
    weights: &'a Lfm2MoeWeights,
    layer: &'a Lfm2MoeLayerWeights,
    state: &'a Lfm2MoeState,
    l: usize,
    seq_len: usize,
}

impl<'a> ForwardBindings for Lfm2MoeBindings<'a> {
    fn run_conv(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        match &self.layer.mixer {
            Mixer::Conv(c) => conv_mixer_block(
                gpu,
                self.cfg,
                &self.layer.operator_norm,
                c,
                self.state,
                self.l,
            ),
            _ => Err("run_conv on non-Conv layer".to_string()),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_attend(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        match &self.layer.mixer {
            Mixer::Attention(a) => attn_mixer_block(
                gpu,
                self.cfg,
                &self.layer.operator_norm,
                a,
                self.state,
                self.l,
                self.seq_len,
            ),
            _ => Err("run_attend on non-Attention layer".to_string()),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_proj(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let code = op.weights.first().map(|w| w.0).unwrap_or(u32::MAX);
        match (code, &self.layer.ffn) {
            (lfm2_op::DENSE_GATE_UP, Ffn::Dense(d)) => {
                dense_gate_up_block(gpu, self.cfg, &self.layer.ffn_norm, d, self.state, self.l)
            }
            _ => Err(format!("run_proj bad opcode {code} / non-Dense ffn")),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_residual_gemv(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let code = op.weights.first().map(|w| w.0).unwrap_or(u32::MAX);
        match (code, &self.layer.ffn) {
            (lfm2_op::DENSE_DOWN, Ffn::Dense(d)) => dense_down_block(gpu, d, self.state, self.l),
            _ => Err(format!(
                "run_residual_gemv bad opcode {code} / non-Dense ffn"
            )),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_moe(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        match &self.layer.ffn {
            Ffn::Moe(m) => moe_ffn_block(
                gpu,
                self.cfg,
                self.weights,
                &self.layer.ffn_norm,
                m,
                self.state,
                self.l,
            ),
            _ => Err("run_moe on non-Moe ffn".to_string()),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_norm(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(
            "lfm2 has no standalone Norm super-op".into(),
        ))
    }
    fn run_recurrent(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("lfm2 has no Recurrent super-op".into()))
    }
    fn run_escape(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        kind: superop::EscapeKind,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(format!(
            "lfm2 has no Escape super-op ({kind:?})"
        )))
    }
}

/// Cached HIPFIRE_FORWARD_LOWERED toggle for lfm2. #397 Ship 6: the lfm2 lowered
/// decode is **DEFAULT ON** as of 2026-06-07 — fleet byte-parity validated
/// (k9lin gfx1100 / hiptrx gfx1201 / hipx gfx1151, lowered == hand token-text md5
/// 754a38b5…). Escape hatch: `HIPFIRE_FORWARD_LOWERED=0` forces the legacy hand
/// loop (still present in decode_step_layers_and_head); any other value / unset → lowered.
fn lfm2_forward_lowered_enabled() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_FORWARD_LOWERED")
            .ok()
            .as_deref()
            != Some("0")
    })
}

/// Lowered (#397 Ship 6) per-layer decode loop + final norm/head. Behaviorally
/// equivalent to decode_step_layers_and_head's hand loop (validated via the
/// FORWARD_LOWERED=0-vs-=1 committed-token md5 A/B). No oracle-capture support.
fn decode_step_layers_and_head_lowered(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    position: u32,
) -> Result<(), String> {
    let eps = cfg.rms_norm_eps;
    let seq_len = position as usize + 1;
    let ctx = DispatchCtx::new(gpu);
    for (l, layer) in weights.layers.iter().enumerate() {
        let program = lfm2_lower_variant(lfm2_variant_of(layer));
        {
            let mut bind = Lfm2MoeBindings {
                cfg,
                weights,
                layer,
                state,
                l,
                seq_len,
            };
            superop::run_layer_program(gpu, &ctx, &program, &mut bind)
                .map_err(|e| format!("lfm2moe L{l}: lowered run_layer_program: {e}"))?;
        }
    }
    state.n_tokens = seq_len;
    gpu.rmsnorm_f32(
        &state.h,
        &weights.embedding_norm,
        &state.final_norm_buf,
        eps,
    )
    .map_err(|e| format!("lfm2moe: final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
        .map_err(|e| format!("lfm2moe: lm_head: {e}"))?;
    Ok(())
}

/// hipGraph-amortized decode_step. Opt-in via `HIPFIRE_LFM2_GRAPH=1`
/// (default OFF → exact `decode_step_inner` behavior). Mirrors the working
/// DeepSeek-V4 integration (`decode_step_with_graph`).
///
/// Three-state machine driven by `state.graph_warmed_up` and `gpu.graph_exec`:
///   1. !warmed_up                 → direct dispatch once (so kernel JIT and
///                                    any lazy hipMalloc happen OUTSIDE the
///                                    captured region), set the flag.
///   2. warmed_up && no graph      → embedding+pos direct, then capture the
///                                    layer loop + head, instantiate, launch
///                                    once for this position's output.
///   3. graph instantiated         → embedding+pos direct, then `graph_launch`
///                                    re-runs the captured ops which re-read
///                                    `state.pos_buf` (refreshed below) and the
///                                    KV / conv-state / topk device buffers.
///
/// Per-token-varying values handled OUTSIDE the captured region:
///   * `token_id` — baked into the embedding kernarg, so the
///     embedding lookup runs DIRECT each token (writes `state.h`); the
///     captured region begins at layer 0's rmsnorm reading `state.h`.
///   * `position` — staged into the STABLE device buffer `state.pos_buf` via a
///     direct `memcpy_htod` before each `graph_launch`; every captured kernel
///     (rope/kv-write/attention) reads `pos_buf` from the device, so replay at
///     a new position is correct without re-capture. The attention kernel uses
///     a fixed tile/block/shared-mem (ATT_Q8_KV_TILE=2048, block 256) independent
///     of `max_seq`/`seq_len` (tiled online-softmax), so one capture replays
///     correctly at every later position.
///
/// `state.n_tokens` is advanced here to match `decode_step_inner` semantics.
pub fn decode_step_with_graph(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    // ── Warmup phase: direct dispatch, no capture ──────────────────────────
    // Run the legacy path once so inline JIT / lazy scratch alloc happen
    // before any stream capture (capturing a hipMalloc errors).
    if !state.graph_warmed_up {
        state.graph_warmed_up = true;
        decode_step_inner(cfg, weights, state, gpu, token_id, position, None)?;
        return gpu
            .download_f32(&state.logits)
            .map_err(|e| format!("lfm2moe: download logits (graph warmup): {e:?}"));
    }

    // Capture/replay needs an explicit (non-null) stream.
    if gpu.active_stream.is_none() {
        let s = gpu
            .hip
            .stream_create()
            .map_err(|e| format!("lfm2moe: stream_create: {e:?}"))?;
        gpu.active_stream = Some(s);
    }

    // Per-token-varying ops, DIRECT (outside the captured region).
    // Single staging convention: shared helper handles bounds check + pos H2D + embedding.
    prepare_retained_decode_inputs(cfg, weights, state, gpu, token_id, position)?;
    if gpu.graphs.graph_exec.is_none() {
        // ── Capture phase ──────────────────────────────────────────────────
        gpu.graphs
            .begin_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("lfm2moe: begin_graph_capture: {e:?}"))?;
        run_retained_decode_body(cfg, weights, state, gpu, position)?;
        gpu.graphs
            .end_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("lfm2moe: end_graph_capture: {e:?}"))?;
        // Recorded, not executed — launch once so this position's logits are real.
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("lfm2moe: graph_launch (capture-end): {e:?}"))?;
        eprintln!(
            "[LFM2.5-MoE hipGraph] captured forward — {} kernarg blobs retained",
            gpu.graphs.capture_blobs.len()
        );
        // run_retained_decode_body set n_tokens; capture-end launch ran it.
    } else {
        // ── Replay phase ────────────────────────────────────────────────────
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("lfm2moe: graph_launch (replay): {e:?}"))?;
        // Mirror run_retained_decode_body's `state.n_tokens = position + 1`,
        // which the replayed graph does NOT execute (it is host-side state).
        state.n_tokens = position as usize + 1;
    }

    // Logits download outside the captured region (sync D2H on the null stream;
    // completes after the captured kernels finish on the captured stream).
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("lfm2moe: download logits (graph): {e:?}"))
}

#[cfg(test)]
mod ship6_lower_tests {
    use super::*;
    use superop::SuperOpKind::{Attend, Conv, Moe, Proj, ResidualGemv};

    // #397 Ship 6 — lfm2 lowered LayerProgram shapes must mirror the hand-loop
    // order (mixer block, then FFN). CPU-pure (no GPU).
    #[test]
    fn lfm2_variant_shapes() {
        let kinds = |v| {
            lfm2_lower_variant(v)
                .iter()
                .map(|o| o.kind)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            kinds(Lfm2Variant::ConvDense),
            vec![Conv, Proj, ResidualGemv]
        );
        assert_eq!(
            kinds(Lfm2Variant::AttnDense),
            vec![Attend, Proj, ResidualGemv]
        );
        assert_eq!(kinds(Lfm2Variant::ConvMoe), vec![Conv, Moe]);
        assert_eq!(kinds(Lfm2Variant::AttnMoe), vec![Attend, Moe]);
        let p = lfm2_lower_variant(Lfm2Variant::ConvDense);
        assert_eq!(p[1].binding.weights[0].0, lfm2_op::DENSE_GATE_UP);
        assert_eq!(p[2].binding.weights[0].0, lfm2_op::DENSE_DOWN);
    }
}

/// No-GPU tests for the pre-GPU MoE prevalidation gate
/// (`Lfm2MoeWeights::moe_prevalidated` → `prevalidate_lfm2_moe`): the decode
/// entry runs the gate before its first GPU op, so a refusal here provably
/// mutates no GPU state. All tensors are CPU-only synthetic buffers sized
/// exactly like the real `Lfm2MoeState` allocations.
#[cfg(test)]
mod moe_prevalidation_tests {
    use super::*;
    use crate::config::MixerKind;
    use crate::lfm2moe::ExpertWeights;
    use hip_bridge::DeviceBuffer;
    use hipfire_runtime::llama::{EmbeddingFormat, KvCache, VMode, WeightTensor};
    use rdna_compute::GpuTensor;

    fn dummy_f32(numel: usize) -> GpuTensor {
        let bytes = numel * 4;
        let buffer = Box::leak(vec![0u8; bytes].into_boxed_slice());
        GpuTensor {
            buf: unsafe { DeviceBuffer::from_raw(buffer.as_mut_ptr().cast(), bytes) },
            shape: vec![numel],
            dtype: DType::F32,
        }
    }

    fn dummy_bytes(numel: usize, bytes: usize) -> GpuTensor {
        let buffer = Box::leak(vec![0u8; bytes].into_boxed_slice());
        GpuTensor {
            buf: unsafe { DeviceBuffer::from_raw(buffer.as_mut_ptr().cast(), bytes) },
            shape: vec![numel],
            dtype: DType::F32,
        }
    }

    fn dummy_wt(dtype: DType, m: usize, k: usize) -> WeightTensor {
        let bytes = m * k * 4;
        let buffer = Box::leak(vec![0u8; bytes].into_boxed_slice());
        WeightTensor {
            buf: GpuTensor {
                buf: unsafe { DeviceBuffer::from_raw(buffer.as_mut_ptr().cast(), bytes) },
                shape: vec![m, k],
                dtype: DType::F32,
            },
            gpu_dtype: dtype,
            m,
            k,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        }
    }

    fn test_cfg() -> Lfm2MoeConfig {
        Lfm2MoeConfig {
            vocab_size: 512,
            hidden_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            conv_kernel_size: 4,
            intermediate_size: 128,
            moe_intermediate_size: 64,
            num_experts: 8,
            num_experts_per_tok: 4,
            num_dense_layers: 0,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 4096,
            norm_topk_prob: true,
            use_expert_bias: true,
            routed_scaling_factor: 1.0,
            tie_word_embeddings: true,
            layer_types: vec![MixerKind::Conv],
            reap_keep: None,
        }
    }

    fn synthetic_model(expert_dtype: DType) -> Lfm2MoeWeights {
        let hidden = 64;
        let moe_inter = 64;
        let n_exp = 8;
        let m = MoeFfn {
            router: dummy_wt(DType::Q8_0, n_exp, hidden),
            expert_bias: dummy_f32(n_exp),
            experts: vec![ExpertWeights {
                gate_up: dummy_wt(expert_dtype, 2 * moe_inter, hidden),
                down: dummy_wt(expert_dtype, hidden, moe_inter),
            }],
            // [2*n_exp] F32 = n_exp u64 device pointers (loader convention).
            expert_gate_up_ptrs: dummy_bytes(2 * n_exp, n_exp * 8),
            expert_down_ptrs: dummy_bytes(2 * n_exp, n_exp * 8),
        };
        Lfm2MoeWeights {
            embed: dummy_f32(1),
            embd_format: EmbeddingFormat::Q8_0,
            embedding_norm: dummy_f32(hidden),
            lm_head: dummy_wt(DType::Q8_0, 512, hidden),
            layers: vec![Lfm2MoeLayerWeights {
                operator_norm: dummy_f32(hidden),
                ffn_norm: dummy_f32(hidden),
                mixer: Mixer::Conv(ConvWeights {
                    in_proj: dummy_wt(DType::Q8_0, 3 * hidden, hidden),
                    conv_weight: dummy_f32(hidden * 4),
                    out_proj: dummy_wt(DType::Q8_0, hidden, hidden),
                    conv_state_idx: 0,
                }),
                ffn: Ffn::Moe(m),
            }],
            moe_group_plans: std::sync::OnceLock::new(),
            moe_validation: std::sync::OnceLock::new(),
        }
    }

    fn synthetic_state(cfg: &Lfm2MoeConfig) -> Lfm2MoeState {
        let hidden = cfg.hidden_size;
        let k_top = cfg.num_experts_per_tok;
        let pos_buf_bytes = Box::leak(vec![0u8; 4].into_boxed_slice());
        Lfm2MoeState {
            kv: KvCache {
                k_gpu: vec![],
                v_gpu: vec![],
                k_scales: vec![],
                v_scales: vec![],
                kv_dim: cfg.kv_dim(),
                max_seq: 4096,
                physical_cap: 4096,
                n_kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
                quantized: false,
                quant_q8: false,
                quant_int8: false,
                quant_hfq4: false,
                quant_asym4: false,
                quant_asym3: false,
                quant_asym2: false,
                boundary_layers: 0,
                givens_cos: None,
                givens_sin: None,
                quant_fwht: false,
                v_mode: VMode::Q8,
                layer_is_boundary: vec![],
                compact_offset: 0,
            },
            conv_states: vec![],
            pos_buf: unsafe { DeviceBuffer::from_raw(pos_buf_bytes.as_mut_ptr().cast(), 4) },
            graph_warmed_up: false,
            retained_warmed_up: false,
            retained_state_poisoned: false,
            max_seq: 4096,
            n_tokens: 0,
            h: dummy_f32(hidden),
            tmp: dummy_f32(hidden),
            fa_q: dummy_f32(cfg.q_dim()),
            fa_k: dummy_f32(cfg.kv_dim()),
            fa_v: dummy_f32(cfg.kv_dim()),
            fa_attn_out: dummy_f32(cfg.q_dim()),
            conv_bcx: dummy_f32(3 * hidden),
            conv_y: dummy_f32(hidden),
            ffn_tmp: dummy_f32(hidden),
            ffn_x_rot: dummy_f32(hidden),
            dense_gate: dummy_f32(cfg.intermediate_size),
            dense_up: dummy_f32(cfg.intermediate_size),
            dense_act: dummy_f32(cfg.intermediate_size),
            router_logits: dummy_f32(cfg.num_experts),
            topk_indices: dummy_bytes(k_top, k_top * 8),
            topk_weights: dummy_f32(k_top),
            gate_batch: dummy_f32(k_top * cfg.moe_intermediate_size),
            up_batch: dummy_f32(k_top * cfg.moe_intermediate_size),
            rot_batch: dummy_f32(k_top * cfg.moe_intermediate_size),
            down_expanded: dummy_f32(k_top * hidden),
            final_norm_buf: dummy_f32(hidden),
            logits: dummy_f32(cfg.vocab_size),
        }
    }

    /// The pre-GPU refusal gate: a model whose MoE experts carry a dtype
    /// outside the sealed indexed set (F16 here) is refused by
    /// `moe_prevalidated` — the exact function the decode entry runs BEFORE
    /// its first GPU op. No `Gpu` is constructed anywhere in this test: the
    /// gate is pure host-side (manifest resolution + per-layer lowering over
    /// synthetic tensors), so the refusal provably mutates no GPU state.
    #[test]
    fn prevalidation_refuses_unsupported_expert_dtype_before_any_gpu_work() {
        let cfg = test_cfg();
        let weights = synthetic_model(DType::F16);
        let state = synthetic_state(&cfg);
        let err = weights
            .moe_prevalidated(&cfg, &state)
            .expect_err("F16 experts must be refused pre-GPU");
        assert!(
            err.contains("unsupported MoE expert"),
            "unexpected refusal message: {err}"
        );
    }

    /// Every dtype the LFM2 loader guard admits (HFQ4/MQ4/HFQ6/MQ6) lowers
    /// cleanly through the manifest-born sealed program, CPU-only.
    #[test]
    fn prevalidation_accepts_supported_indexed_expert_dtypes() {
        for dtype in [
            DType::MQ4G256,
            DType::HFQ4G256,
            DType::MQ6G256,
            DType::HFQ6G256,
        ] {
            let cfg = test_cfg();
            let weights = synthetic_model(dtype);
            let state = synthetic_state(&cfg);
            weights
                .moe_prevalidated(&cfg, &state)
                .unwrap_or_else(|e| panic!("{dtype:?} must prevalidate: {e}"));
        }
    }

    /// The cached verdict is config-identity-bound: a different config on the
    /// same model refuses rather than silently reusing the stale verdict.
    #[test]
    fn prevalidation_cache_refuses_config_identity_mismatch() {
        let cfg = test_cfg();
        let weights = synthetic_model(DType::MQ4G256);
        let state = synthetic_state(&cfg);
        weights
            .moe_prevalidated(&cfg, &state)
            .expect("valid model prevalidates");
        let mut other = test_cfg();
        other.vocab_size = 1024;
        let err = weights
            .moe_prevalidated(&other, &state)
            .expect_err("config identity mismatch must refuse");
        assert!(
            err.contains("config identity mismatch"),
            "unexpected refusal: {err}"
        );
    }
}
