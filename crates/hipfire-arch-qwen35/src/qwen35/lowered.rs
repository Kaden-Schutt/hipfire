// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 lowered single-GPU decode path (#397 Ship 6): routes the decode
//! layer loop through the dispatch substrate's super-op executor instead of
//! the hand-written arms in `forward_scratch_layers`. Additive and validated
//! byte-identical to the hand path; default-on via `HIPFIRE_FORWARD_LOWERED`.

use super::*;

// ─────────────────────────────────────────────────────────────────────────
// #397 Ship 6 — forward-as-pipeline: qwen35 DECODE lowered path (ADDITIVE).
//
// `HIPFIRE_FORWARD_LOWERED=1` routes the single-GPU decode layer loop through
// the dispatch substrate's `run_layer_program` executor (one pre-resolved
// `LayerProgram` of coarse super-ops per layer) instead of the hand-written
// arms in `forward_scratch_layers`. The hand arms are left UNTOUCHED, so the
// default (flag off) is byte-identical to master by construction; the lowered
// path is validated byte-identical via the external committed-token md5 gate
// (`FORWARD_LOWERED=0` vs `=1`, same prompt) on the fleet before the default is
// flipped per arch. See [[project_ship6_forward_pipeline_design_2026_06_07]].
//
// The super-op handlers call the SAME helper fns the hand path uses
// (`qkv/qkvza/gate_up_via_execute_steps`, `kv_cache_attention_dispatch`,
// `moe_ffn_dispatch`, `weight_gemv_swiglu_residual`) plus the inline attend/
// recurrent/gated-norm fragments. DIAG dumps / trace_finite / hidden_rb are
// output-neutral and omitted here (hidden_rb engages only the hand path).
// ─────────────────────────────────────────────────────────────────────────

/// qwen35-local super-op opcodes, encoded into `OpBinding.weights[0].0`. The
/// `SuperOpKind` routes to the `ForwardBindings` method; the opcode disambiguates
/// *which* op of that kind within the layer (qkv vs gate_up, wo vs down, …).
pub(crate) mod q35_op {
    // Proj
    pub const PROJ_QKV: u32 = 0;
    pub const PROJ_QKVZA: u32 = 1;
    pub const PROJ_GATE_UP: u32 = 2;
    // Attend
    pub const ATTEND_FULL: u32 = 0;
    pub const ATTEND_DN_PREP: u32 = 1;
    // ResidualGemv
    pub const RESID_WO: u32 = 0;
    pub const RESID_DOWN_SWIGLU: u32 = 1;
    // Norm
    pub const NORM_GATED: u32 = 0;
    // Recurrent
    pub const RECUR_GDN: u32 = 0;
    // Moe
    pub const MOE_FFN: u32 = 0;
}

/// The four qwen35 decoder-layer shapes. Derived from the `LayerWeights`
/// discriminant; kept as a plain enum so `lower_variant` is pure (no GpuTensor)
/// and unit-testable without a GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Q35Variant {
    DeltaNet,
    FullAttn,
    DeltaNetMoe,
    FullAttnMoe,
}

pub(crate) fn variant_of(layer: &LayerWeights) -> Q35Variant {
    match layer {
        LayerWeights::DeltaNet(_) => Q35Variant::DeltaNet,
        LayerWeights::FullAttn(_) => Q35Variant::FullAttn,
        LayerWeights::DeltaNetMoe(_) => Q35Variant::DeltaNetMoe,
        LayerWeights::FullAttnMoe(_) => Q35Variant::FullAttnMoe,
    }
}

#[inline]
fn q35_superop(kind: SuperOpKind, code: u32) -> SuperOp {
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

/// Lower one qwen35 decoder layer to a coarse-super-op `LayerProgram`. The op
/// SEQUENCE mirrors the matching hand arm in `forward_scratch_layers` exactly
/// (per the decode-forward variant map). Pure → unit-testable.
pub(crate) fn lower_variant(v: Q35Variant) -> LayerProgram {
    use q35_op::*;
    use SuperOpKind::{Attend, Moe, Norm, Proj, Recurrent, ResidualGemv};
    match v {
        Q35Variant::DeltaNet => vec![
            q35_superop(Proj, PROJ_QKVZA),
            q35_superop(Attend, ATTEND_DN_PREP),
            q35_superop(Recurrent, RECUR_GDN),
            q35_superop(Norm, NORM_GATED),
            q35_superop(ResidualGemv, RESID_WO),
            q35_superop(Proj, PROJ_GATE_UP),
            q35_superop(ResidualGemv, RESID_DOWN_SWIGLU),
        ],
        Q35Variant::FullAttn => vec![
            q35_superop(Proj, PROJ_QKV),
            q35_superop(Attend, ATTEND_FULL),
            q35_superop(ResidualGemv, RESID_WO),
            q35_superop(Proj, PROJ_GATE_UP),
            q35_superop(ResidualGemv, RESID_DOWN_SWIGLU),
        ],
        Q35Variant::DeltaNetMoe => vec![
            q35_superop(Proj, PROJ_QKVZA),
            q35_superop(Attend, ATTEND_DN_PREP),
            q35_superop(Recurrent, RECUR_GDN),
            q35_superop(Norm, NORM_GATED),
            q35_superop(ResidualGemv, RESID_WO),
            q35_superop(Moe, MOE_FFN),
        ],
        Q35Variant::FullAttnMoe => vec![
            q35_superop(Proj, PROJ_QKV),
            q35_superop(Attend, ATTEND_FULL),
            q35_superop(ResidualGemv, RESID_WO),
            q35_superop(Moe, MOE_FFN),
        ],
    }
}

/// Per-layer execution context for the lowered decode path. Holds the current
/// layer's weights + shared scratch/state by reference; rebuilt each layer
/// iteration so the borrows stay scoped. `kv_cache` is the only `&mut` (DeltaNet
/// state is mutated through interior-mutable GpuTensor buffers via shared refs).
pub(crate) struct Qwen35Bindings<'a> {
    pub(crate) layer: &'a LayerWeights,
    pub(crate) s: &'a Qwen35Scratch,
    pub(crate) config: &'a Qwen35Config,
    pub(crate) kv_cache: &'a mut kv::KvCache,
    pub(crate) dn_state: &'a DeltaNetState,
    pub(crate) pos: usize,
    pub(crate) layer_idx: usize,
    pub(crate) delta_layer_idx: usize,
    pub(crate) k_dim: usize,
    pub(crate) v_dim: usize,
    pub(crate) n_v_heads: usize,
    pub(crate) hd: usize,
}

fn op_code(op: &OpBinding) -> u32 {
    op.weights.first().map(|w| w.0).unwrap_or(u32::MAX)
}

impl<'a> ForwardBindings for Qwen35Bindings<'a> {
    fn run_proj(
        &mut self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let res: HipResult<()> = match op_code(op) {
            q35_op::PROJ_QKV => match self.layer {
                LayerWeights::FullAttn(l) => qkv_via_execute_steps(
                    gpu,
                    ctx,
                    &l.wq,
                    &l.wk,
                    &l.wv,
                    &l.attn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.fa_q_full,
                    &s.fa_k,
                    &s.fa_v,
                    config.norm_eps,
                ),
                LayerWeights::FullAttnMoe(l) => qkv_via_execute_steps(
                    gpu,
                    ctx,
                    &l.wq,
                    &l.wk,
                    &l.wv,
                    &l.attn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.fa_q_full,
                    &s.fa_k,
                    &s.fa_v,
                    config.norm_eps,
                ),
                _ => return Err(DispatchError::Hip("PROJ_QKV on non-FullAttn layer".into())),
            },
            q35_op::PROJ_QKVZA => match self.layer {
                LayerWeights::DeltaNet(l) => qkvza_via_execute_steps(
                    gpu,
                    ctx,
                    &l.wqkv,
                    &l.wz,
                    &l.w_beta,
                    &l.w_alpha,
                    &l.attn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.dn_qkv,
                    &s.dn_z,
                    &s.dn_beta,
                    &s.dn_alpha,
                    config.norm_eps,
                ),
                LayerWeights::DeltaNetMoe(l) => qkvza_via_execute_steps(
                    gpu,
                    ctx,
                    &l.wqkv,
                    &l.wz,
                    &l.w_beta,
                    &l.w_alpha,
                    &l.attn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.dn_qkv,
                    &s.dn_z,
                    &s.dn_beta,
                    &s.dn_alpha,
                    config.norm_eps,
                ),
                _ => {
                    return Err(DispatchError::Hip(
                        "PROJ_QKVZA on non-DeltaNet layer".into(),
                    ))
                }
            },
            q35_op::PROJ_GATE_UP => match self.layer {
                LayerWeights::DeltaNet(l) => gate_up_via_execute_steps(
                    gpu,
                    ctx,
                    &l.w_gate,
                    &l.w_up,
                    &l.ffn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.gate_ffn,
                    &s.up,
                    config.norm_eps,
                ),
                LayerWeights::FullAttn(l) => gate_up_via_execute_steps(
                    gpu,
                    ctx,
                    &l.w_gate,
                    &l.w_up,
                    &l.ffn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.gate_ffn,
                    &s.up,
                    config.norm_eps,
                ),
                _ => {
                    return Err(DispatchError::Hip(
                        "PROJ_GATE_UP on MoE/unknown layer".into(),
                    ))
                }
            },
            other => return Err(DispatchError::Hip(format!("unknown PROJ opcode {other}"))),
        };
        res.map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_residual_gemv(
        &mut self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let res: HipResult<()> = (|| match op_code(op) {
            q35_op::RESID_WO => {
                let (wo, input): (&WeightTensor, &GpuTensor) = match self.layer {
                    LayerWeights::FullAttn(l) => (&l.wo, &s.fa_attn_out),
                    LayerWeights::FullAttnMoe(l) => (&l.wo, &s.fa_attn_out),
                    LayerWeights::DeltaNet(l) => (&l.wo, &s.dn_normed),
                    LayerWeights::DeltaNetMoe(l) => (&l.wo, &s.dn_normed),
                };
                let wr = wo.dispatch_ref();
                execute_steps(
                    gpu,
                    ctx,
                    &[Step::GemvResidual {
                        w: &wr,
                        input: GemvInput::Raw(input),
                        residual: &s.x,
                        out: &s.x,
                    }],
                )
                .map_err(|e| HipError::new(0, &e.to_string()))
            }
            q35_op::RESID_DOWN_SWIGLU => {
                let w_down = match self.layer {
                    LayerWeights::DeltaNet(l) => &l.w_down,
                    LayerWeights::FullAttn(l) => &l.w_down,
                    _ => return Err(HipError::new(0, "RESID_DOWN_SWIGLU on MoE layer")),
                };
                hipfire_runtime::weights::weight_gemv_swiglu_residual(
                    gpu,
                    w_down,
                    &s.gate_ffn,
                    &s.up,
                    &s.ffn_hidden,
                    &s.x,
                )
            }
            other => Err(HipError::new(0, &format!("unknown RESID opcode {other}"))),
        })();
        res.map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_norm(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let norm_weight = match self.layer {
            LayerWeights::DeltaNet(l) => &l.norm_weight,
            LayerWeights::DeltaNetMoe(l) => &l.norm_weight,
            _ => {
                return Err(DispatchError::Hip(
                    "NORM_GATED on non-DeltaNet layer".into(),
                ))
            }
        };
        gpu.gated_norm_f32(
            &s.dn_attn_out,
            &s.dn_z,
            norm_weight,
            &s.dn_normed,
            self.n_v_heads,
            config.linear_value_head_dim,
            config.norm_eps,
        )
        .map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_attend(
        &mut self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let res: HipResult<()> = (|| match op_code(op) {
            q35_op::ATTEND_FULL => {
                let (q_norm, k_norm) = match self.layer {
                    LayerWeights::FullAttn(l) => (&l.q_norm, &l.k_norm),
                    LayerWeights::FullAttnMoe(l) => (&l.q_norm, &l.k_norm),
                    _ => return Err(HipError::new(0, "ATTEND_FULL on non-FullAttn layer")),
                };
                gpu.deinterleave_f32(
                    &s.fa_q_full,
                    &s.fa_q,
                    &s.fa_gate,
                    config.n_heads,
                    config.head_dim,
                )?;
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                let npu_hnr_ok = if hipfire_runtime::triattn::tap_enabled() {
                    false
                } else {
                    try_npu_headnorm_rope(
                        gpu,
                        self.layer_idx,
                        &s.fa_q,
                        &s.fa_k,
                        q_norm,
                        k_norm,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        n_rot,
                        config.rope_theta,
                        self.pos,
                    )?
                };
                if !npu_hnr_ok {
                    gpu.rmsnorm_batched(
                        &s.fa_q,
                        q_norm,
                        &s.fa_q,
                        config.n_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;
                    gpu.rmsnorm_batched(
                        &s.fa_k,
                        k_norm,
                        &s.fa_k,
                        config.n_kv_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;
                    if hipfire_runtime::triattn::tap_enabled() {
                        triattn_tap(gpu, self.layer_idx, s, config)?;
                    }
                    if self.kv_cache.compact_offset > 0 {
                        let abs = (self.pos + self.kv_cache.compact_offset) as i32;
                        gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
                    }
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
                }
                if self.kv_cache.compact_offset > 0 {
                    let phys = self.pos as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
                }
                kv_cache_attention_dispatch(
                    ctx,
                    gpu,
                    self.kv_cache,
                    s,
                    config,
                    self.layer_idx,
                    self.pos,
                )?;
                gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
                Ok(())
            }
            q35_op::ATTEND_DN_PREP => {
                let (dt_bias, a_log, conv_weight) = match self.layer {
                    LayerWeights::DeltaNet(l) => (&l.dt_bias, &l.a_log, &l.conv_weight),
                    LayerWeights::DeltaNetMoe(l) => (&l.dt_bias, &l.a_log, &l.conv_weight),
                    _ => return Err(HipError::new(0, "ATTEND_DN_PREP on non-DeltaNet layer")),
                };
                gpu.fused_sigmoid_alpha_gate_f32(
                    &s.dn_beta,
                    &s.dn_alpha,
                    dt_bias,
                    a_log,
                    self.n_v_heads,
                )?;
                gpu.conv1d_silu_split_f32(
                    &s.dn_q_raw,
                    &s.dn_k_raw,
                    &s.dn_v,
                    &s.dn_qkv,
                    conv_weight,
                    &self.dn_state.conv_states[self.delta_layer_idx],
                    self.k_dim,
                    self.v_dim,
                )?;
                gpu.fused_qk_l2_norm_scale_f32(
                    &s.dn_q_raw,
                    &s.dn_k_raw,
                    config.linear_num_key_heads,
                    self.hd,
                    1.0 / (self.hd as f32).sqrt(),
                    config.norm_eps,
                )?;
                if config.linear_num_key_heads < self.n_v_heads {
                    let ratio = self.n_v_heads / config.linear_num_key_heads;
                    gpu.repeat_interleave_qk_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        &s.dn_q,
                        &s.dn_k,
                        config.linear_num_key_heads,
                        ratio,
                        self.hd,
                    )?;
                } else {
                    gpu.memcpy_dtod_auto(&s.dn_q.buf, &s.dn_q_raw.buf, self.k_dim * 4)?;
                    gpu.memcpy_dtod_auto(&s.dn_k.buf, &s.dn_k_raw.buf, self.k_dim * 4)?;
                }
                Ok(())
            }
            other => Err(HipError::new(0, &format!("unknown ATTEND opcode {other}"))),
        })();
        res.map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_moe(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let (ffn, ffn_norm) = match self.layer {
            LayerWeights::DeltaNetMoe(l) => (&l.ffn, &l.ffn_norm),
            LayerWeights::FullAttnMoe(l) => (&l.ffn, &l.ffn_norm),
            _ => return Err(DispatchError::Hip("MOE on dense layer".into())),
        };
        moe_ffn_dispatch(gpu, ffn, &s.x, ffn_norm, config, s, self.layer_idx)
            .map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_moe_ep(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        routed_out: &GpuTensor,
        skip_shared: bool,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let (ffn, ffn_norm) = match self.layer {
            LayerWeights::DeltaNetMoe(l) => (&l.ffn, &l.ffn_norm),
            LayerWeights::FullAttnMoe(l) => (&l.ffn, &l.ffn_norm),
            _ => return Err(DispatchError::Hip("MOE on dense layer".into())),
        };
        // Routed combine + shared-down (rank 0 only) accumulate into `routed_out`
        // (zeroed by the EP executor); s.x (the replicated attention residual) is
        // untouched until ep_add_into_residual after the all-reduce.
        moe_ffn_dispatch_ep(
            gpu,
            ffn,
            &s.x,
            ffn_norm,
            config,
            s,
            self.layer_idx,
            routed_out,
            skip_shared,
        )
        .map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn ep_add_into_residual(
        &mut self,
        gpu: &mut Gpu,
        partial: &GpuTensor,
    ) -> Result<(), DispatchError> {
        // s.x += the all-reduced routed partial (the EP MoE output summed across
        // ranks). Mirrors the prototype's `tp_allreduce_add` residual step.
        let s = self.s;
        gpu.add_inplace_f32(&s.x, partial)
            .map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_recurrent(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let dn = self.dn_state;
        let i = self.delta_layer_idx;
        let res: HipResult<()> = match dn.quant {
            StateQuant::FP32 => gpu.gated_delta_net_f32(
                &s.dn_q,
                &s.dn_k,
                &s.dn_v,
                &s.dn_alpha,
                &s.dn_beta,
                &dn.s_matrices[i],
                &s.dn_attn_out,
                1,
                self.n_v_heads,
                config.linear_value_head_dim,
            ),
            StateQuant::Q8 => gpu.gated_delta_net_q8(
                &s.dn_q,
                &s.dn_k,
                &s.dn_v,
                &s.dn_alpha,
                &s.dn_beta,
                &dn.s_matrices[i],
                &dn.s_scales[i],
                &s.dn_attn_out,
                1,
                self.n_v_heads,
                config.linear_value_head_dim,
            ),
            StateQuant::Q4 => gpu.gated_delta_net_q4(
                &s.dn_q,
                &s.dn_k,
                &s.dn_v,
                &s.dn_alpha,
                &s.dn_beta,
                &dn.s_matrices[i],
                &dn.s_scales[i],
                &s.dn_attn_out,
                1,
                self.n_v_heads,
                config.linear_value_head_dim,
            ),
        };
        res.map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_conv(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("qwen35 has no Conv super-op".into()))
    }

    fn run_escape(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        kind: superop::EscapeKind,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(format!(
            "qwen35 has no Escape super-op ({kind:?})"
        )))
    }
}

/// Cached `HIPFIRE_FORWARD_LOWERED` toggle. #397 Ship 6: the qwen35 single-GPU
/// decode lowered path is **DEFAULT ON** as of 2026-06-07 — validated byte-
/// identical to the hand path via fleet decode byte-parity (RDNA3 k9lin / RDNA4
/// hiptrx / RDNA3.5 hipx, dense + MoE) and the full coherence battery (13 cases,
/// k9lin). Escape hatch: `HIPFIRE_FORWARD_LOWERED=0` forces the legacy hand arms
/// (still present in forward_scratch_layers); any other value (or unset) → lowered.
pub(crate) fn forward_lowered_enabled() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("HIPFIRE_FORWARD_LOWERED").ok().as_deref() != Some("0"))
}

/// Lowered (#397 Ship 6) single-GPU decode layer loop. Behaviorally equivalent
/// to `forward_scratch_layers`'s hand arms (validated byte-identical via the
/// external committed-token md5 gate). Builds a coarse-super-op `LayerProgram`
/// per layer and runs it through the dispatch substrate's executor.
pub(crate) fn forward_scratch_layers_lowered(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &DeltaNetState,
    s: &Qwen35Scratch,
    needs_logits: bool,
) -> HipResult<()> {
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;

    let ctx = DispatchCtx::new(gpu);
    let mut delta_layer_idx = 0usize;

    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];
        let program = lower_variant(variant_of(layer));
        {
            let mut bind = Qwen35Bindings {
                layer,
                s,
                config,
                kv_cache: &mut *kv_cache,
                dn_state,
                pos,
                layer_idx,
                delta_layer_idx,
                k_dim,
                v_dim,
                n_v_heads,
                hd,
            };
            superop::run_layer_program(gpu, &ctx, &program, &mut bind)
                .map_err(|e| HipError::new(0, &e.to_string()))?;
        }
        if matches!(
            layer,
            LayerWeights::DeltaNet(_) | LayerWeights::DeltaNetMoe(_)
        ) {
            delta_layer_idx += 1;
        }
        dump_hidden_localize(gpu, &s.x, 1, pos, config.dim, layer_idx, "pertoken");
    }

    // Final norm always (cheap; populates s.tmp, the hidden some callers read).
    // lm_head (vocab-wide gemv) only when logits are needed — in prefill only the
    // FINAL token needs them, so non-final tokens skip this (~37% of prefill on
    // gfx1103 per rocprof). Without this, the lowered path ignored the caller's
    // no-logits request and computed lm_head every token.
    gpu.rmsnorm_f32(&s.x, &weights.output_norm, &s.tmp, config.norm_eps)?;
    if needs_logits {
        let ctx = DispatchCtx::new(gpu);
        let wr = weights.output.dispatch_ref();
        let step = Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(&s.tmp),
            out: &s.logits,
        };
        execute_steps(gpu, &ctx, &[step]).map_err(|e| HipError::new(0, &e.to_string()))?;
    }
    Ok(())
}
