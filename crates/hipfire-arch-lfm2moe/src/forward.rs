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
use crate::lfm2moe::{Ffn, Lfm2MoeState, Lfm2MoeWeights, Mixer};
use hipfire_dispatch::families::moe::{ExpertExecutionPlan, RouterPlan};
use hipfire_dispatch::pipeline::{
    execute_steps, GemvInput, MoeActivationVariant, MoeProj, ScoreActKind, Step,
};
use hipfire_dispatch::types::{dtype_rotation_plan, RotationPlan};
use hipfire_runtime::llama::rotate_x_mq_for;
use hipfire_runtime::moe_plan::{
    execute_lowered_moe, lower_moe_steps, MoEExecutionPolicy, MoeExecutionTarget, MoeProgramParts,
    RoutedMoeStepPhases,
};
use rdna_compute::Gpu;

/// Decode one token; returns the full logits vector.
///
/// Routes to the hipGraph capture/replay path when `HIPFIRE_LFM2_GRAPH=1`
/// (default OFF → exact prior behavior). The graph path amortizes the ~377
/// per-token kernel launches by replaying a single captured graph; see
/// `decode_step_with_graph`.
pub fn decode_step(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    if graph_enabled() {
        return decode_step_with_graph(cfg, weights, state, gpu, token_id, position);
    }
    decode_step_inner(cfg, weights, state, gpu, token_id, position, None)?;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("lfm2moe: download logits: {e:?}"))
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

fn decode_step_inner(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;

    // Device position scalar (i32) for rope / kv-write / attention.
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("lfm2moe: htod pos: {e:?}"))?;

    // Embedding lookup → residual stream h (Q8 table).
    gpu.embedding_lookup_q8(&weights.embed, &state.h, token_id, hidden)
        .map_err(|e| format!("lfm2moe: embed lookup: {e:?}"))?;

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
    let moe_inter = cfg.moe_intermediate_size;
    let n_exp = cfg.num_experts;
    let k_top = cfg.num_experts_per_tok;
    let eps = cfg.rms_norm_eps;
    let seq_len = position as usize + 1;
    let capture_postmixer =
        hipfire_config::developer_var_os("HIPFIRE_LFM2_CAPTURE_POSTMIXER").is_some();

    for (l, layer) in weights.layers.iter().enumerate() {
        // ── Mixer block (pre-norm) ──────────────────────────────────────────
        // STEP-004 Inc 3: norm + projections + qk-norm + RoPE via Step lists.
        // Rotation is dtype-driven (FWHT for MQ-family weights, none for Q8)
        // so Prerotated inputs stay valid — mirrors the qwen35 helpers.
        match &layer.mixer {
            Mixer::Conv(c) => {
                let rot = dtype_rotation_plan(c.in_proj.gpu_dtype);
                let w_in = c.in_proj.dispatch_ref();
                let ctx = hipfire_dispatch::context::DispatchCtx::new(gpu);
                execute_steps(
                    gpu,
                    &ctx,
                    &[
                        Step::RmsnormAutomatic {
                            x: &state.h,
                            norm_weight: &layer.operator_norm,
                            x_plain: &state.tmp,
                            out: &state.tmp,
                            awq_scale: None,
                            k: hidden,
                            eps,
                            rotation: rot,
                        },
                        Step::Gemv {
                            w: &w_in,
                            input: GemvInput::Prerotated(&state.tmp),
                            out: &state.conv_bcx,
                        },
                    ],
                )
                .map_err(|e| format!("lfm2moe L{l}: conv in_proj: {e:?}"))?;
                // STEP-004 inventory exception #5: the double-gated depthwise
                // causal short-conv mixer is stateful and architecturally
                // unique; the per-call Step path keeps it direct (the lowered
                // substrate handles it via SuperOpKind::Conv).
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
                let w_out = c.out_proj.dispatch_ref();
                execute_steps(
                    gpu,
                    &ctx,
                    &[Step::GemvResidual {
                        w: &w_out,
                        input: GemvInput::Raw(&state.conv_y),
                        residual: &state.h,
                        out: &state.h,
                    }],
                )
                .map_err(|e| format!("lfm2moe L{l}: conv out_proj: {e:?}"))?;
            }
            Mixer::Attention(a) => {
                let rot = dtype_rotation_plan(a.wq.gpu_dtype);
                let (wq, wk, wv) = (
                    a.wq.dispatch_ref(),
                    a.wk.dispatch_ref(),
                    a.wv.dispatch_ref(),
                );
                let ctx = hipfire_dispatch::context::DispatchCtx::new(gpu);
                let steps = [
                    Step::RmsnormAutomatic {
                        x: &state.h,
                        norm_weight: &layer.operator_norm,
                        x_plain: &state.tmp,
                        out: &state.tmp,
                        awq_scale: None,
                        k: hidden,
                        eps,
                        rotation: rot,
                    },
                    Step::Gemv {
                        w: &wq,
                        input: GemvInput::Prerotated(&state.tmp),
                        out: &state.fa_q,
                    },
                    Step::Gemv {
                        w: &wk,
                        input: GemvInput::Prerotated(&state.tmp),
                        out: &state.fa_k,
                    },
                    Step::Gemv {
                        w: &wv,
                        input: GemvInput::Prerotated(&state.tmp),
                        out: &state.fa_v,
                    },
                    Step::QkNorm {
                        x: &state.fa_q,
                        weight: &a.q_norm,
                        n_groups: n_heads,
                        head_dim,
                        eps,
                    },
                    Step::QkNorm {
                        x: &state.fa_k,
                        weight: &a.k_norm,
                        n_groups: n_kv,
                        head_dim,
                        eps,
                    },
                    Step::Rope {
                        q: &state.fa_q,
                        k: &state.fa_k,
                        pos_buf: &state.pos_buf,
                        n_heads,
                        n_kv_heads: n_kv,
                        head_dim,
                        theta: cfg.rope_theta,
                    },
                ];
                execute_steps(gpu, &ctx, &steps)
                    .map_err(|e| format!("lfm2moe L{l}: attention prep: {e:?}"))?;

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
                let wo_ref = a.wo.dispatch_ref();
                let ctx = hipfire_dispatch::context::DispatchCtx::new(gpu);
                execute_steps(
                    gpu,
                    &ctx,
                    &[Step::GemvResidual {
                        w: &wo_ref,
                        input: GemvInput::Raw(&state.fa_attn_out),
                        residual: &state.h,
                        out: &state.h,
                    }],
                )
                .map_err(|e| format!("lfm2moe L{l}: out_proj: {e:?}"))?;
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
        // STEP-004 Inc 3: dense FFN through the Step interpreter (dtype-driven
        // rotation; the gate_up window fuses where eligible).
        match &layer.ffn {
            Ffn::Dense(d) => {
                let rot = dtype_rotation_plan(d.w1.gpu_dtype);
                let (w1, w3, w2) = (
                    d.w1.dispatch_ref(),
                    d.w3.dispatch_ref(),
                    d.w2.dispatch_ref(),
                );
                let ctx = hipfire_dispatch::context::DispatchCtx::new(gpu);
                let steps = [
                    Step::RmsnormAutomatic {
                        x: &state.h,
                        norm_weight: &layer.ffn_norm,
                        x_plain: &state.tmp,
                        out: &state.ffn_tmp,
                        awq_scale: None,
                        k: hidden,
                        eps,
                        rotation: rot,
                    },
                    Step::Gemv {
                        w: &w1,
                        input: GemvInput::Prerotated(&state.ffn_tmp),
                        out: &state.dense_gate,
                    },
                    Step::Gemv {
                        w: &w3,
                        input: GemvInput::Prerotated(&state.ffn_tmp),
                        out: &state.dense_up,
                    },
                    Step::SiluMul {
                        gate: &state.dense_gate,
                        up: &state.dense_up,
                        out: &state.dense_act,
                    },
                ];
                execute_steps(gpu, &ctx, &steps)
                    .map_err(|e| format!("lfm2moe L{l}: dense w1/w3: {e:?}"))?;
                execute_steps(
                    gpu,
                    &ctx,
                    &[Step::GemvResidual {
                        w: &w2,
                        input: GemvInput::Raw(&state.dense_act),
                        residual: &state.h,
                        out: &state.h,
                    }],
                )
                .map_err(|e| format!("lfm2moe L{l}: dense w2: {e:?}"))?;
            }
            Ffn::Moe(m) => {
                let w_r = m.router.dispatch_ref();
                let ctx = hipfire_dispatch::context::DispatchCtx::new(gpu);
                // STEP-004 Inc 3: ffn norm + router + sigmoid + bias-aware
                // top-k via Steps (ScoreActivation + MoeRoute launch the same
                // two kernels the direct path ran, in the same order). The
                // norm stays PLAIN (RotationPlan::None): the Q8 router reads
                // the plain input and the MQ4 experts rotate via the direct
                // rotate_x_mq_for below (same split as minimax).
                let steps = [
                    Step::RmsnormAutomatic {
                        x: &state.h,
                        norm_weight: &layer.ffn_norm,
                        x_plain: &state.tmp,
                        out: &state.ffn_tmp,
                        awq_scale: None,
                        k: hidden,
                        eps,
                        rotation: RotationPlan::None,
                    },
                    Step::Gemv {
                        w: &w_r,
                        input: GemvInput::Raw(&state.ffn_tmp),
                        out: &state.router_logits,
                    },
                ];
                execute_steps(gpu, &ctx, &steps)
                    .map_err(|e| format!("lfm2moe L{l}: router: {e:?}"))?;

                // FWHT-rotate the FFN input for the MQ4 experts (router stays
                // plain). Family scaling lives outside the routed program
                // (same split as minimax).
                rotate_x_mq_for(
                    gpu,
                    &m.experts[0].gate_up,
                    &state.ffn_tmp,
                    &state.ffn_x_rot,
                    hidden,
                )
                .map_err(|e| format!("lfm2moe L{l}: ffn rotate: {e:?}"))?;

                // Routed expert program (sigmoid → bias-aware top-k →
                // gate_up → silu·mul·rotate → down → combine): built from the
                // shared Step building blocks and lowered through the
                // manifest-born ExpertGroupPlan (lower_moe_steps) + the
                // sealed Single executor. The launch schedule is derived from
                // the concrete borrowed steps; the kernels are the same
                // indexed-MoE family the direct sequence launched (MQ4→HFQ4,
                // MQ6→HFQ6 by the expert refs' dtype) — the combine
                // accumulates into the residual (the Single executor never
                // zeroes `out`).
                let gu_ref = hipfire_dispatch::families::moe::MoeExpertRef {
                    gate_up_ptrs: &m.expert_gate_up_ptrs,
                    down_ptrs: &m.expert_down_ptrs,
                    dummy_gate_up: None,
                    dtype: m.experts[0].gate_up.gpu_dtype,
                    n_experts: n_exp,
                    expert_m: moe_inter,
                    expert_k: hidden,
                    owned: &[],
                };
                let dn_ref = hipfire_dispatch::families::moe::MoeExpertRef {
                    gate_up_ptrs: &m.expert_gate_up_ptrs,
                    down_ptrs: &m.expert_down_ptrs,
                    dummy_gate_up: None,
                    dtype: m.experts[0].down.gpu_dtype,
                    n_experts: n_exp,
                    expert_m: moe_inter,
                    expert_k: hidden,
                    owned: &[],
                };
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
                    ranks: vec![phases],
                };
                let plan = weights
                    .moe_group_plans(cfg)
                    .map_err(|e| format!("lfm2moe L{l}: expert-group plan: {e}"))?
                    .by_layer(l)?;
                let policy = MoEExecutionPolicy::single();
                let program = lower_moe_steps(plan, &policy, parts)
                    .map_err(|e| format!("lfm2moe L{l}: lower_moe_steps: {e:?}"))?;
                execute_lowered_moe(&program, MoeExecutionTarget::Single { gpu, ctx: &ctx })
                    .map_err(|e| format!("lfm2moe L{l}: execute_lowered_moe: {e:?}"))?;
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
    let rot = dtype_rotation_plan(weights.lm_head.gpu_dtype);
    let w_head = weights.lm_head.dispatch_ref();
    let ctx = hipfire_dispatch::context::DispatchCtx::new(gpu);
    execute_steps(
        gpu,
        &ctx,
        &[
            Step::RmsnormAutomatic {
                x: &state.h,
                norm_weight: &weights.embedding_norm,
                x_plain: &state.final_norm_buf,
                out: &state.final_norm_buf,
                awq_scale: None,
                k: hidden,
                eps,
                rotation: rot,
            },
            Step::Gemv {
                w: &w_head,
                input: GemvInput::Prerotated(&state.final_norm_buf),
                out: &state.logits,
            },
        ],
    )
    .map_err(|e| format!("lfm2moe: final rmsnorm/lm_head: {e:?}"))?;
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
///   * `token_id` — baked into `embedding_lookup_q8`'s kernarg, so the
///     embedding lookup runs DIRECT each token (writes `state.h`); the
///     captured region begins at layer 0's rmsnorm reading `state.h`.
///   * `position` — staged into the STABLE device buffer `state.pos_buf` via a
///     direct `memcpy_htod` before each `graph_launch`; every captured kernel
///     (rope/kv-write/attention) reads `pos_buf` from the device, so replay at
///     a new position is correct without re-capture. The attention kernel's
///     launch-baked `block_size`/`shared_mem` are sized to `max_seq` under
///     capture (see `attention_q8_0_kv` in dispatch.rs), so one capture
///     replays correctly at every later position.
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
    let hidden = cfg.hidden_size;

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
    // pos_buf: refreshed each token; the captured kernels re-read it on replay.
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("lfm2moe: htod pos (graph): {e:?}"))?;
    // embedding lookup: token_id is a kernarg → must run per-token, not captured.
    gpu.embedding_lookup_q8(&weights.embed, &state.h, token_id, hidden)
        .map_err(|e| format!("lfm2moe: embed lookup (graph): {e:?}"))?;

    if gpu.graphs.graph_exec.is_none() {
        // ── Capture phase ──────────────────────────────────────────────────
        gpu.graphs
            .begin_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("lfm2moe: begin_graph_capture: {e:?}"))?;
        decode_step_layers_and_head(cfg, weights, state, gpu, position, None)?;
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
        // decode_step_layers_and_head set n_tokens; capture-end launch ran it.
    } else {
        // ── Replay phase ────────────────────────────────────────────────────
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("lfm2moe: graph_launch (replay): {e:?}"))?;
        // Mirror decode_step_layers_and_head's `state.n_tokens = position + 1`,
        // which the replayed graph does NOT execute (it is host-side state).
        state.n_tokens = position as usize + 1;
    }

    // Logits download outside the captured region (sync D2H on the null stream;
    // completes after the captured kernels finish on the captured stream).
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("lfm2moe: download logits (graph): {e:?}"))
}
