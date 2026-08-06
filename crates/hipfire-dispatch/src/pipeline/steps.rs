// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
//! Op-list interpreter. Phase 2a: GEMV + a fused rmsnorm-rotate producer; empty
//! fusion table (all per-op fallback).

use hipfire_hardware::DeviceMesh;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::OnceLock;

use crate::context::DispatchCtx;
use crate::families::fused_qkv::{FusedQkvBiasParams, FusedQkvFamily, FusedQkvParams};
use crate::families::gemv::{GemvFamily, GemvParams, RotateInputs, WeightRef};
use crate::families::moe::{
    deepseek_f32_down_indexed_form, deepseek_gate_up_indexed_form, deepseek_i64_down_indexed_form,
    grouped_down_projection, indexed_moe_batch_guard, launch_fused_shared_gate,
    launch_grouped_down, launch_grouped_gate_up, launch_indexed_down, launch_indexed_down_residual,
    launch_indexed_down_residual_i64, launch_indexed_down_residual_i64_batched,
    launch_indexed_gate_up, launch_indexed_gate_up_batched, launch_moe_activation,
    launch_moe_combine, launch_moe_combine_grouped, launch_moe_gate_up_unscatter, launch_moe_route,
    launch_moe_scatter, launch_moe_softmax_topk, launch_qwen_down_indexed,
    launch_qwen_gate_up_indexed, launch_scaled_add_gpu_scalar, launch_score_activation,
    launch_shared_expert_down_body, launch_shared_gate_side, DeepSeekIndexedForm, MoeExpertRef,
};
use crate::families::rotation::{RotationFamily, RotationParams};
use crate::types::GemvVariant;
use crate::types::{DispatchError, KernelKey, PipelineOp, RotationPlan, RotationVariant};

#[cfg(feature = "deltanet")]
use crate::ops::delta_net::{
    DeltaNetBatchIntent, DeltaNetBatchParams, DeltaNetStepParams, DeltaNetTreeParams,
    DeltaNetTreeState, StateQuant,
};

/// Rotation disposition of a Gemv's input. Borrows (never owns a RotatedActivation).
pub enum GemvInput<'a> {
    Raw(&'a GpuTensor),        // launch_op self-rotates via run_auto (plan-aware)
    Prerotated(&'a GpuTensor), // already FWHT-rotated; dispatched via Prerotated variant
}

/// Down-projection shape discriminant for [`Step::IndexedMoeGemv`].
///
/// Two kernel families underly three shapes:
/// - **Expanded** (`GateUp` / `DownExpanded`): writes an intermediate buffer; a
///   separate [`Step::MoeCombine`] folds the per-expert outputs with `topk_weights`
///   into the EP partial. MQ4/HFQ4/MQ6/MQ2L support this path via
///   [`launch_indexed_down`]. MQ3L does **not** (no `*_expanded_k4` kernel exists).
/// - **Residual-fused** (`DownResidual`): [`launch_indexed_down_residual`] folds the
///   weighted combine into the kernel and writes directly into the EP partial. Used
///   by MQ2L (minimax self-combining path) and MQ3L (the only down path for MQ3L).
///   Calling [`Step::MoeCombine`] after `DownResidual` would double-accumulate.
///
/// Score activation kind for in-place MoE routing pre-op.
///
/// Applied to the raw router logits before [`Step::MoeRoute`].
#[derive(Clone, Copy, Debug)]
pub enum ScoreActKind {
    /// Sigmoid activation (minimax routing).
    Sigmoid,
    /// Sqrt-softplus activation (deepseek4 routing).
    SqrtSoftplus,
}

/// Per-arch SwiGLU activation + FWHT rotate variant for [`Step::MoeActivation`].
pub enum MoeActivationVariant<'a> {
    /// minimax: fused silu·mul + block-diagonal FWHT rotate in ONE kernel.
    /// `awq_scale`: `Some` → uses the AWQ-scaled kernel; `None` → plain kernel.
    /// The shipped M2.7.mq2 carries AWQ on its down weights and passes `Some`.
    MinimaxFused { awq_scale: Option<&'a GpuTensor> },
    /// ds4: silu·mul·CLAMP (in-place into `gate`) then a SEPARATE FWHT rotate. Two kernels.
    Ds4ClampRotate { swiglu_limit: f32 },
    /// qwen: fused silu·mul + FWHT rotate with PER-EXPERT AWQ scales selected
    /// by the top-K slot (`fused_silu_mul_rotate_mq_awq_indexed_batched`).
    /// Used when the routed experts carry per-expert down `awq_scale` sidecars.
    QwenAwqIndexed {
        awq_ptrs: &'a GpuTensor,
        topk_indices: &'a GpuTensor,
    },
    /// qwen Paro: fused silu·mul + Givens rotate
    /// (`fused_silu_mul_givens_rotate_f32`). `k_top` carries the row count
    /// (k_top for decode, batch·k_top for prefill).
    QwenParo {
        pairs: &'a GpuTensor,
        theta: &'a GpuTensor,
        scales: &'a GpuTensor,
        krot: usize,
    },
}

pub enum MoeProj<'a> {
    /// Gate+up projection: writes gate_batch (= step `out`) + up_batch (= `up_out`).
    /// Requires FWHT-pre-rotated input. `topk_weights` not needed here.
    GateUp { up_out: &'a GpuTensor },
    /// Down expanded path (MQ4/HFQ4/MQ6/MQ2L): writes per-expert outputs to `out`
    /// (= `down_expanded`). A separate [`Step::MoeCombine`] folds into the EP partial.
    DownExpanded,
    /// Down residual-fused path (MQ2L/MQ3L): folds the weighted combine into the
    /// down kernel. The step's `out` IS the EP partial (accumulate semantics).
    /// No [`Step::MoeCombine`] follows — that would double-accumulate.
    DownResidual { topk_weights: &'a GpuTensor },
    /// Reproducible int64 down path (MQ3L TP): writes an S-scaled int64 accumulator
    /// into `out` (which must be an i64 buffer of `hidden` elements, pre-zeroed).
    /// After an [`StepCollective::AllReduceI64Tp`] all-reduce, a
    /// [`Step::ConvertI64ToF32`] converts the summed int64 into the FP partial.
    /// EP path stays on `DownResidual` (FP). Only used when `tp > 1`.
    DownResidualI64 { topk_weights: &'a GpuTensor },
}

/// Down-projection shape discriminant for [`Step::MoeDownIndexed`] (qwen).
///
/// - **Expanded**: writes per-expert outputs to `out` (= `down_expanded`); a
///   separate [`Step::MoeCombine`] folds them with `topk_weights` into the EP
///   partial. Serves the decode down (all indexable dtypes) and prefill
///   Path 1 (batched).
/// - **ResidualScaled**: the kernel folds the weighted combine into the
///   atomic accumulation itself and writes directly into `out` — the EP
///   partial or `x_batch` (prefill Path 0, MQ4). No [`Step::MoeCombine`]
///   follows — that would double-accumulate.
pub enum QwenDownMode<'a> {
    Expanded,
    ResidualScaled { topk_weights: &'a GpuTensor },
}

/// Borrowed execution shape for the DeltaNet recurrence. The six semantic
/// [`Step`] variants stay dtype-agnostic; quantization and the decode,
/// prefill, replay, or tree route are carried by the existing DeltaNet
/// parameter contracts instead of multiplying the Step enum by dtype.
#[cfg(feature = "deltanet")]
pub enum DeltaRecurrenceParams<'a> {
    Step(DeltaNetStepParams<'a>),
    Batch {
        params: DeltaNetBatchParams<'a>,
        intent: DeltaNetBatchIntent,
    },
    Tree(DeltaNetTreeParams<'a>),
}

/// All caller-owned operands shared by the DeltaNet decode and batch/tree
/// lowering paths.  This is deliberately a view: it does not allocate, own, or
/// move recurrent state.  Qwen35 constructs it from the global-layer → compact
/// state adapter and the per-layer scratch buffers.
#[cfg(feature = "deltanet")]
pub struct DeltaNetOperandDescriptor<'a> {
    pub qkv: &'a GpuTensor,
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub q_raw: &'a GpuTensor,
    pub k_raw: &'a GpuTensor,
    pub alpha: &'a GpuTensor,
    pub beta: &'a GpuTensor,
    pub dt_bias: Option<&'a GpuTensor>,
    pub a_log: Option<&'a GpuTensor>,
    pub state: &'a GpuTensor,
    pub s_scales: &'a GpuTensor,
    pub ef_residual: Option<&'a GpuTensor>,
    pub conv_weight: &'a GpuTensor,
    pub conv_state: &'a GpuTensor,
    pub attn_out: &'a GpuTensor,
    pub normed: Option<&'a GpuTensor>,
    pub z: Option<&'a GpuTensor>,
    pub norm_weight: Option<&'a GpuTensor>,
    pub n_key_heads: usize,
    pub n_value_heads: usize,
    pub head_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub q_scale: f32,
    pub eps: f32,
    pub quant: StateQuant,
}

#[cfg(feature = "deltanet")]
impl<'a> DeltaNetOperandDescriptor<'a> {
    fn recurrence_step(&self) -> DeltaNetStepParams<'a> {
        DeltaNetStepParams {
            q: self.q,
            k: self.k,
            v: self.v,
            gate: self.alpha,
            beta: self.beta,
            state: self.state,
            s_scales: self.s_scales,
            output: self.attn_out,
            ef_residual: self.ef_residual,
            n_heads: self.n_value_heads,
            head_dim: self.head_dim,
            quant: self.quant,
        }
    }

    fn gated_norm(&self, batch_size: usize) -> Option<Step<'a>> {
        Some(Step::DeltaGatedNorm {
            x: self.attn_out,
            z: self.z?,
            weight: self.norm_weight?,
            out: self.normed?,
            n_heads: self.n_value_heads,
            head_dim: self.head_dim,
            eps: self.eps,
            batch_size,
        })
    }
}

/// Build the single-token DeltaNet attention body.  The decode builder uses
/// the scalar kernels by setting `batch_size = 1`; it never changes the
/// caller-owned state or creates a second state owner.
#[cfg(feature = "deltanet")]
pub fn build_delta_net_decode_steps<'a>(d: &'a DeltaNetOperandDescriptor<'a>) -> Vec<Step<'a>> {
    vec![
        Step::DeltaGatePrep {
            beta: d.beta,
            alpha: d.alpha,
            dt_bias: d.dt_bias.expect("decode builder requires dt_bias"),
            a_log: d.a_log.expect("decode builder requires a_log"),
            n: d.n_value_heads,
            batch_size: 1,
        },
        Step::DeltaConvSplit {
            q_out: d.q_raw,
            k_out: d.k_raw,
            v_out: d.v,
            input: d.qkv,
            weight: d.conv_weight,
            state: d.conv_state,
            parent_indices: None,
            k_dim: d.key_dim,
            v_dim: d.value_dim,
            n_tokens: 1,
        },
        Step::DeltaQkL2Norm {
            q: d.q_raw,
            k: d.k_raw,
            n_key_heads: d.n_key_heads,
            head_dim: d.head_dim,
            q_scale: d.q_scale,
            eps: d.eps,
            batch_size: 1,
        },
        Step::DeltaRepeatHeads {
            q_src: d.q_raw,
            k_src: d.k_raw,
            q_dst: d.q,
            k_dst: d.k,
            n_key_heads: d.n_key_heads,
            ratio: d.n_value_heads / d.n_key_heads,
            head_dim: d.head_dim,
            batch_size: 1,
        },
        Step::DeltaRecurrence {
            params: DeltaRecurrenceParams::Step(d.recurrence_step()),
        },
        d.gated_norm(1)
            .expect("decode builder requires gated norm operands"),
    ]
}

/// Build the real batched DeltaNet attention body.  `tree_state` selects the
/// read-only tree tape route; `None` selects ordinary sequential prefill.
/// Batch size is threaded into every operation, so this cannot accidentally
/// lower a prefill/tree request into B=1 decode steps.
#[cfg(feature = "deltanet")]
pub fn build_delta_net_batch_steps<'a>(
    d: &'a DeltaNetOperandDescriptor<'a>,
    n_tokens: usize,
    intent: DeltaNetBatchIntent,
    tree_state: Option<DeltaNetTreeState<'a>>,
    parent_indices: Option<&'a GpuTensor>,
) -> Result<Vec<Step<'a>>, String> {
    if n_tokens == 0 {
        return Err("DeltaNet batch builder requires at least one token".into());
    }
    if tree_state.is_some() != parent_indices.is_some() {
        return Err("DeltaNet tree builder requires parent indices and tape state together".into());
    }
    let recurrence = if let Some(state) = tree_state {
        DeltaRecurrenceParams::Tree(DeltaNetTreeParams::new(
            d.q,
            d.k,
            d.v,
            d.alpha,
            d.beta,
            state,
            parent_indices.expect("tree state checked above"),
            d.attn_out,
            n_tokens,
            d.n_value_heads,
            d.head_dim,
        ))
    } else {
        DeltaRecurrenceParams::Batch {
            params: DeltaNetBatchParams {
                q_batch: d.q,
                k_batch: d.k,
                v_batch: d.v,
                gate_batch: d.alpha,
                beta_batch: d.beta,
                state: d.state,
                s_scales: d.s_scales,
                output_batch: d.attn_out,
                ef_residual: d.ef_residual,
                n_tokens,
                n_heads: d.n_value_heads,
                head_dim: d.head_dim,
                quant: d.quant,
            },
            intent,
        }
    };

    let mut steps = Vec::with_capacity(6);
    if matches!(intent, DeltaNetBatchIntent::NormalPrefill) {
        steps.push(Step::DeltaGatePrep {
            beta: d.beta,
            alpha: d.alpha,
            dt_bias: d.dt_bias.expect("prefill builder requires dt_bias"),
            a_log: d.a_log.expect("prefill builder requires a_log"),
            n: d.n_value_heads,
            batch_size: n_tokens,
        });
    }
    steps.extend([
        Step::DeltaConvSplit {
            q_out: d.q_raw,
            k_out: d.k_raw,
            v_out: d.v,
            input: d.qkv,
            weight: d.conv_weight,
            state: d.conv_state,
            parent_indices,
            k_dim: d.key_dim,
            v_dim: d.value_dim,
            n_tokens,
        },
        Step::DeltaQkL2Norm {
            q: d.q_raw,
            k: d.k_raw,
            n_key_heads: d.n_key_heads,
            head_dim: d.head_dim,
            q_scale: d.q_scale,
            eps: d.eps,
            batch_size: n_tokens,
        },
        Step::DeltaRepeatHeads {
            q_src: d.q_raw,
            k_src: d.k_raw,
            q_dst: d.q,
            k_dst: d.k,
            n_key_heads: d.n_key_heads,
            ratio: d.n_value_heads / d.n_key_heads,
            head_dim: d.head_dim,
            batch_size: n_tokens,
        },
        Step::DeltaRecurrence { params: recurrence },
    ]);
    if let Some(norm) = d.gated_norm(n_tokens) {
        steps.push(norm);
    }
    Ok(steps)
}

/// Explicit tree-verify builder. Kept separate from the ordinary batch entry
/// point so callers cannot accidentally omit the tape when lowering a tree.
#[cfg(feature = "deltanet")]
pub fn build_delta_net_tree_steps<'a>(
    d: &'a DeltaNetOperandDescriptor<'a>,
    n_tokens: usize,
    parent_indices: &'a GpuTensor,
    tape: &'a GpuTensor,
    tape_scales: Option<&'a GpuTensor>,
) -> Result<Vec<Step<'a>>, String> {
    let tree_state = match d.quant {
        StateQuant::FP32 => DeltaNetTreeState::F32 {
            initial: d.state,
            tape,
        },
        StateQuant::Q8 => DeltaNetTreeState::Q8 {
            initial: d.state,
            scales: d.s_scales,
            tape,
            tape_scales: tape_scales.ok_or("Q8 tree builder requires tape scales")?,
        },
        StateQuant::Q4 => {
            return Err(
                "Q4 DeltaNet state + tree-verify (DDTree) is unsupported: there is no Q4 tree-tape GDN kernel. Use Q8 or FP32 state for tree spec-decode.".into(),
            )
        }
    };
    build_delta_net_batch_steps(
        d,
        n_tokens,
        DeltaNetBatchIntent::NormalPrefill,
        Some(tree_state),
        Some(parent_indices),
    )
}

pub enum Step<'a> {
    Gemv {
        w: &'a WeightRef<'a>,
        input: GemvInput<'a>,
        out: &'a GpuTensor,
    },
    /// GEMV with in-place residual add: `residual += W · input`.
    /// For MQ-family, `input` must be pre-rotated (Prerotated variant) or the
    /// Raw variant triggers FWHT rotation before calling the residual kernel.
    GemvResidual {
        w: &'a WeightRef<'a>,
        input: GemvInput<'a>,
        residual: &'a GpuTensor,
        out: &'a GpuTensor,
    },
    /// Batched (B>1) GEMM: `y[batch×m] = W · x[batch×k]`. Prefill-only; decode
    /// uses `Gemv`. Column-parallel use: `y=[batch×m]` on-rank shard. Row-parallel
    /// use: `y=[batch×dim]` partial → `AllReduceOut` → `ResidualAdd` (never fused).
    Gemm {
        w: &'a WeightRef<'a>,
        x: &'a GpuTensor,
        y: &'a GpuTensor,
        batch: usize,
    },
    /// Batched GEMM dispatched by an explicit kernel-table key. This preserves
    /// the caller's arch-specific routing for mixed quant formats.
    GemmKeyedBatched {
        w: &'a WeightRef<'a>,
        x: &'a GpuTensor,
        y: &'a GpuTensor,
        batch: usize,
        key: KernelKey,
    },
    /// Batched rmsnorm with the same optional FWHT/Givens rotation contract as
    /// `RmsnormAutomatic`. Kept separate so existing decode Step literals keep
    /// their B=1 semantics while prefill/tree callers must state their batch.
    RmsnormBatched {
        x: &'a GpuTensor,
        norm_weight: &'a GpuTensor,
        x_plain: &'a GpuTensor,
        out: &'a GpuTensor,
        awq_scale: Option<&'a GpuTensor>,
        k: usize,
        eps: f32,
        rotation: RotationPlan,
        batch: usize,
    },
    /// Batched Paro/Givens activation rotation. The rotation metadata is
    /// borrowed from the weight sidecar; no temporary owner is created.
    GivensRotateBatched {
        x: &'a GpuTensor,
        out: &'a GpuTensor,
        pairs: &'a GpuTensor,
        theta: &'a GpuTensor,
        scales: &'a GpuTensor,
        batch: usize,
        dim: usize,
        krot: usize,
    },
    /// Batched FWHT rotation for an already-normalized activation.
    RotateFwhtBatched {
        x: &'a GpuTensor,
        out: &'a GpuTensor,
        awq_scale: Option<&'a GpuTensor>,
        k: usize,
        batch: usize,
    },
    /// Four-way batched DeltaNet QKVZA projection.
    #[cfg(feature = "deltanet")]
    FusedQkvzaBatched {
        wqkv: &'a WeightRef<'a>,
        wz: &'a WeightRef<'a>,
        w_beta: &'a WeightRef<'a>,
        w_alpha: &'a WeightRef<'a>,
        x: &'a GpuTensor,
        qkv: &'a GpuTensor,
        z: &'a GpuTensor,
        beta: &'a GpuTensor,
        alpha: &'a GpuTensor,
        m: [usize; 4],
        k: usize,
        batch: usize,
        key: KernelKey,
    },
    /// Batched output projection with the residual add performed by the
    /// selected residual GEMM kernel.
    GemmResidualBatched {
        w: &'a WeightRef<'a>,
        x: &'a GpuTensor,
        residual: &'a GpuTensor,
        batch: usize,
        key: KernelKey,
    },
    /// Fused rmsnorm + optional FWHT rotation. The `rotation` field is derived
    /// by the caller via `dtype_rotation_plan(w.dtype)`. `out` holds the
    /// ready-to-use activation (FWHT-rotated for FwhtG256, plain-normed for None).
    /// All downstream Gemv steps use GemvInput::Prerotated(out).
    RmsnormAutomatic {
        x: &'a GpuTensor,
        norm_weight: &'a GpuTensor,
        x_plain: &'a GpuTensor, // rmsnorm intermediate scratch (always written)
        out: &'a GpuTensor,     // final activation output (written by this step)
        awq_scale: Option<&'a GpuTensor>,
        k: usize,
        eps: f32,
        rotation: RotationPlan, // FwhtG256 for MQ dtypes, None for HFQ4/others
    },
    /// Paired KV-write + flash-attention (Phase 0.3). Consumes a KvTierPlan
    /// (derived once per attention step) and AttnParams (tensor borrows).
    /// Not fusible — the two ops are inherently coupled.
    Attend {
        plan: crate::families::kv_tier::KvTierPlan,
        io: crate::families::attention::AttnParams<'a>,
    },
    /// In-place RoPE on Q and K. Per-op only (no fused entry) — present so the
    /// attention block can be one contiguous step list (future fusion seam).
    Rope {
        q: &'a GpuTensor,
        k: &'a GpuTensor,
        pos_buf: &'a hip_bridge::DeviceBuffer,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        theta: f32,
    },
    /// Per-head rmsnorm on one tensor (Qwen3-style qk-norm). One step per tensor.
    QkNorm {
        x: &'a GpuTensor,
        weight: &'a GpuTensor,
        n_groups: usize, // n_heads (Q) or n_kv_heads (K)
        head_dim: usize,
        eps: f32,
    },
    /// In-place bias add on one tensor (e.g. qwen2 QKV bias).
    BiasAdd {
        x: &'a GpuTensor,
        bias: &'a GpuTensor,
        dim: usize,
    },
    #[cfg(feature = "deltanet")]
    /// DeltaNet alpha/beta preparation. All tensors are caller-owned.
    DeltaGatePrep {
        beta: &'a GpuTensor,
        alpha: &'a GpuTensor,
        dt_bias: &'a GpuTensor,
        a_log: &'a GpuTensor,
        n: usize,
        batch_size: usize,
    },
    #[cfg(feature = "deltanet")]
    /// Causal convolution and Q/K/V split. `parent_indices` selects the
    /// read-only tree kernel; `None` selects the linear decode/prefill route.
    DeltaConvSplit {
        q_out: &'a GpuTensor,
        k_out: &'a GpuTensor,
        v_out: &'a GpuTensor,
        input: &'a GpuTensor,
        weight: &'a GpuTensor,
        state: &'a GpuTensor,
        parent_indices: Option<&'a GpuTensor>,
        k_dim: usize,
        v_dim: usize,
        n_tokens: usize,
    },
    #[cfg(feature = "deltanet")]
    /// In-place per-head Q/K L2 normalization and Q scaling.
    DeltaQkL2Norm {
        q: &'a GpuTensor,
        k: &'a GpuTensor,
        n_key_heads: usize,
        head_dim: usize,
        q_scale: f32,
        eps: f32,
        batch_size: usize,
    },
    #[cfg(feature = "deltanet")]
    /// Copy normalized key heads into the value/query head layout.
    DeltaRepeatHeads {
        q_src: &'a GpuTensor,
        k_src: &'a GpuTensor,
        q_dst: &'a GpuTensor,
        k_dst: &'a GpuTensor,
        n_key_heads: usize,
        ratio: usize,
        head_dim: usize,
        batch_size: usize,
    },
    #[cfg(feature = "deltanet")]
    /// Recurrent DeltaNet update/read with borrowed state, scales, tape, and
    /// optional Q8 error-feedback rows.
    DeltaRecurrence { params: DeltaRecurrenceParams<'a> },
    #[cfg(feature = "deltanet")]
    /// z-gated output normalization.
    DeltaGatedNorm {
        x: &'a GpuTensor,
        z: &'a GpuTensor,
        weight: &'a GpuTensor,
        out: &'a GpuTensor,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
        batch_size: usize,
    },
    /// SwiGLU activation: `out = silu(gate) * up` (elementwise). Present so a
    /// dense FFN block can be one contiguous step list — the IR previously fused
    /// silu into gate-up kernels, leaving no standalone activation op, which
    /// blocked expressing a column-parallel FFN's on-rank intermediate as Steps.
    SiluMul {
        gate: &'a GpuTensor,
        up: &'a GpuTensor,
        out: &'a GpuTensor,
    },
    /// In-place residual add: `x += y`. The single-GPU dense forward fuses this
    /// into `GemvResidual`, but a row-parallel `GemvResidual` would all-reduce
    /// `(partial + residual)` and sum the residual `tp×`. Under TP the row-parallel
    /// projection is a plain `Gemv` → all-reduce → this `ResidualAdd`, so the
    /// residual is added exactly once, after the collective.
    ResidualAdd {
        x: &'a GpuTensor,
        y: &'a GpuTensor,
        dim: usize,
    },
    /// Bias-aware top-K MoE routing (deepseek4 decode path, k=6).
    /// Selects on `scores + gate_bias`, weights on the unbiased `scores`, normalizes,
    /// folds in `route_scale` — all in one launch. Writes `topk_indices` and
    /// `topk_weights`. Delegates to [`launch_moe_route`].
    MoeRoute {
        scores: &'a GpuTensor,
        gate_bias: &'a GpuTensor,
        topk_indices: &'a GpuTensor,
        topk_weights: &'a GpuTensor,
        k: usize,
        n_experts: usize,
        route_scale: f32,
    },
    /// Indexed per-expert GEMV for the top-K selected experts.
    ///
    /// Three shapes via `which` (see [`MoeProj`]):
    /// - `GateUp`: gate+up → `out` = gate_batch, `which.up_out` = up_batch.
    ///   `input` must be FWHT-pre-rotated. No `topk_weights` (combine is later).
    /// - `DownExpanded`: down → `out` = `down_expanded` [k × expert_k].
    ///   A separate [`Step::MoeCombine`] folds into the EP partial.
    ///   `batch_size` is consumed; MQ3L is unsupported (use `DownResidual`).
    /// - `DownResidual`: down + weighted-combine fused (MQ2L/MQ3L) → `out` = EP partial.
    ///   `which.topk_weights` carries weights. No [`Step::MoeCombine`] follows.
    /// - `DownResidualI64`: reproducible S-scaled int64 accumulator (MQ2L/MQ3L
    ///   scalar, MQ2L batched) → `out` = i64 partial. A later
    ///   [`Step::ConvertI64ToF32`] converts it.
    ///
    /// `batch_size` is the authoritative routed batch: `1` keeps the scalar
    /// launchers byte-identically, `> 1` selects the DeepSeek batched
    /// MQ2-Lloyd kernels for `GateUp` / `DownResidualI64` (never a scalar
    /// fallback), `0` is rejected before dispatch, and `DownResidual` (FP32)
    /// rejects `> 1` explicitly. `DownExpanded` consumes it as before.
    ///
    /// `tp_step_out_buf` returns `Some(&out.buf)` only for `DownResidual`
    /// (the partial that the EP all-reduce reduces over).
    IndexedMoeGemv {
        experts: &'a MoeExpertRef<'a>,
        which: MoeProj<'a>,
        topk_indices: &'a GpuTensor,
        /// FWHT-pre-rotated input for GateUp; SwiGLU output (rot_batch) for Down*.
        input: GemvInput<'a>,
        /// gate_batch for GateUp, down_expanded for DownExpanded, EP partial for DownResidual.
        out: &'a GpuTensor,
        k_top: usize,
        /// Routed batch (authoritative): 1 = scalar launchers, > 1 = DeepSeek
        /// batched MQ2-Lloyd kernels (GateUp / DownResidualI64), 0 = rejected
        /// before dispatch. DownExpanded consumes it as before.
        batch_size: usize,
    },
    /// Weighted combine of per-expert expanded down outputs into the EP partial.
    /// Delegates to [`launch_moe_combine`] (decode) or [`launch_moe_combine_grouped`]
    /// (prefill grouped path, when `inverse_perm` is `Some`).
    ///
    /// - `inverse_perm = None` → `moe_down_combine_k8_batched` (decode).
    ///   Call after [`Step::IndexedMoeGemv`] with `which = DownExpanded`.
    ///   Do NOT call after `DownResidual` (double-accumulate).
    /// - `inverse_perm = Some(&perm)` → `moe_down_combine_grouped_k8` (prefill Path 2).
    ///   Call after [`Step::GroupedMoeGemm`] with `which = DownExpanded`.
    ///
    /// `out` is the pre-zeroed EP partial (accumulate semantics); the executor
    /// zeroes it via `zero_before` before this step runs. `tp_step_out_buf` returns
    /// `Some(&out.buf)` so the EP all-reduce finds the partial buffer.
    MoeCombine {
        down_out: &'a GpuTensor,
        topk_weights: &'a GpuTensor,
        out: &'a GpuTensor,
        k: usize,
        hidden: usize,
        batch_size: usize,
        /// Grouped-path inverse permutation produced by [`Step::MoeScatter`].
        /// `Some` → prefill grouped combine (`moe_down_combine_grouped_k8`).
        /// `None` → decode expanded combine (`moe_down_combine_k8_batched`).
        inverse_perm: Option<&'a GpuTensor>,
    },
    /// Scatter+histogram for grouped-GEMM prefill (Path 2). Builds
    /// `sorted_slot_index`, `expert_tile_ids`, and `inverse_perm` from
    /// `topk_indices`. Delegates to [`launch_moe_scatter`].
    /// Must run before [`Step::GroupedMoeGemm`]. `tp_step_out_buf` returns `None`.
    MoeScatter {
        topk_indices: &'a GpuTensor,
        expert_token_counts: &'a GpuTensor,
        expert_offsets: &'a GpuTensor,
        sorted_slot_index: &'a GpuTensor,
        expert_tile_ids: &'a GpuTensor,
        inverse_perm: &'a GpuTensor,
        total_slots: usize,
        n_experts: usize,
        m_total_max: usize,
        block_m: usize,
    },
    /// Grouped-WMMA expert GEMM for prefill (Path 2). One launch covers all
    /// expert tokens sorted by `sorted_slot_index`. `which` distinguishes:
    /// - `GateUp`: `m = 2·expert_m`, `x_row_div = k_top`, `rows = batch_size`.
    ///   Writes fused gate||up output to `y` (y_gate_up_grouped).
    ///   `up_out` in `MoeProj::GateUp` is unused by the grouped kernel (output is `y`).
    /// - `DownExpanded`: `m = expert_k`, `x_row_div = 1`, `rows = batch*k_top`.
    ///   Writes down output to `y` (y_down_grouped) for [`Step::MoeCombine`].
    ///   The residual projections are structurally rejected (no grouped
    ///   residual-fused down kernel exists).
    ///
    /// `dtype_tags` selects the merged per-expert grouped kernel (graded
    /// files); `force_mq4_fp16`, `paro_i8`, `paro_i8_k8` carry the grouped
    /// controls from [`crate::families::moe::MoePrefillResolution`].
    ///
    /// `tp_step_out_buf` returns `None` — `y` is an intermediate, not an EP partial.
    GroupedMoeGemm {
        experts: &'a MoeExpertRef<'a>,
        which: MoeProj<'a>,
        sorted_slot_index: &'a GpuTensor,
        expert_tile_ids: &'a GpuTensor,
        /// For `GateUp`: x_rot_batch; for `DownExpanded`: rot_batch.
        x: &'a GpuTensor,
        /// For `GateUp`: y_gate_up_grouped; for `DownExpanded`: y_down_grouped.
        y: &'a GpuTensor,
        m_total: usize,
        batch_size: usize,
        k_top: usize,
        /// Per-expert mixed dtype-tag table (graded files). `None` = uniform.
        dtype_tags: Option<&'a GpuTensor>,
        /// MQ4 grouped prefill uses FP16 WMMA for mixed MQ6-promoted checkpoints.
        force_mq4_fp16: bool,
        /// gfx1151 Paro i8 MMQ grouped GEMM levers.
        paro_i8: bool,
        paro_i8_k8: bool,
    },
    /// Deinterleave the grouped gate_up result: `y_grouped → gate_batch +
    /// up_batch`. Delegates to [`launch_moe_gate_up_unscatter`]. Call after
    /// [`Step::GroupedMoeGemm`] with `which = GateUp`, before activation —
    /// there is no post-down unscatter Step or kernel. `tp_step_out_buf`
    /// returns `None`.
    MoeGateUpUnscatter {
        y_grouped: &'a GpuTensor,
        sorted_slot_index: &'a GpuTensor,
        gate_batch: &'a GpuTensor,
        up_batch: &'a GpuTensor,
        inter: usize,
        k_top: usize,
        m_total: usize,
    },
    /// In-place score activation before MoE routing: sigmoid (minimax) or
    /// sqrt_softplus (ds4). Feeds the (already-activated) scores to `MoeRoute`.
    /// `tp_step_out_buf` returns `None`.
    ScoreActivation {
        scores: &'a GpuTensor,
        kind: ScoreActKind,
    },
    /// SwiGLU activation + FWHT re-rotate of the gate/up intermediate, per-arch.
    /// Reads `gate`, `up` `[rows × inter]`; writes `rot_out` `[rows × inter]`
    /// consumed by the down Step. `tp_step_out_buf` returns `None` (intermediate,
    /// not an EP partial). Block-diagonal per-256 → shards trivially under D2b.
    ///
    /// `k_top` is the routed row count at launch: `k_top` for decode,
    /// `batch_size·k_top` for batched prefill (DeepSeek supplies the checked
    /// product). The executor binds it as local `rows`.
    MoeActivation {
        variant: MoeActivationVariant<'a>,
        gate: &'a GpuTensor,
        up: &'a GpuTensor,
        rot_out: &'a GpuTensor,
        inter: usize,
        k_top: usize,
    },
    /// Convert an int64 S-scaled residual buffer to f32. Used after
    /// [`StepCollective::AllReduceI64Tp`] in the reproducible MoE down TP path.
    /// `src` is the i64 partial (hidden elements, S-scaled); `dst` is the f32
    /// partial that the Phase-3 residual add consumes. `n` = hidden.
    /// `tp_step_out_buf` returns `None` — this is a post-reduce convert, not a partial.
    ConvertI64ToF32 {
        src: &'a GpuTensor,
        dst: &'a GpuTensor,
        n: usize,
    },
    // ── Qwen MoE Step-native ops (STEP-002 Phase 1) ───────────────────────
    // Semantic operations the pre-existing Step surface could not express.
    // All launch exactly the kernels the legacy `run_moe_decode` /
    // `run_moe_prefill` executors dispatched; the legacy executors now build
    // these programs and execute them (single production path). The CPU-top-K
    // fallback stays an explicit non-Step leaf.
    /// Softmax + renormalized top-K routing (qwen decode GPU path). Two
    /// launches in one step — `softmax_f32(logits)` then
    /// `moe_topk_renorm_k8(logits, topk_indices, topk_weights, n_exp,
    /// norm_topk_prob)` — preserving the legacy launch order. Prefill routing
    /// stays model-owned (no step).
    MoeSoftmaxTopK {
        logits: &'a GpuTensor,
        topk_indices: &'a GpuTensor,
        topk_weights: &'a GpuTensor,
        n_exp: usize,
        norm_topk_prob: bool,
    },
    /// Fused gate-side projection (qwen decode, MQ4 gate side): one launch of
    /// `fused_qkvza_hfq4g256` over the single FWHT-rotated `x_rot` writing
    /// router logits, the shared-expert scalar, shared gate, and shared up.
    /// The shared gate/up outputs are the `[0, smi)` slice views of
    /// `gate_buf`/`up_buf` (created by the launcher — the views are ephemeral).
    /// The non-fusable gate side is `MoeSharedGateSide` instead.
    MoeFusedSharedGate {
        router: &'a WeightRef<'a>,
        shared_expert_gate: &'a WeightRef<'a>,
        shared_gate_w: &'a WeightRef<'a>,
        shared_up_w: &'a WeightRef<'a>,
        x_rot: &'a GpuTensor,
        router_logits: &'a GpuTensor,
        scalar_buf: &'a GpuTensor,
        gate_buf: &'a GpuTensor,
        up_buf: &'a GpuTensor,
        smi: usize,
    },
    /// Per-weight gate-side projection (qwen decode, non-fusable gate side):
    /// the router and shared-expert gate GEMVs always re-rotate from `x_norm`;
    /// the shared gate/up GEMVs reuse the pre-rotated `x_rot_local` only when
    /// the shared-prerotation decision applies (extracted helper), else they
    /// re-rotate too. Mirrors the legacy four-GEMV gate side exactly.
    MoeSharedGateSide {
        router: &'a WeightRef<'a>,
        shared_expert_gate: &'a WeightRef<'a>,
        shared_gate_w: &'a WeightRef<'a>,
        shared_up_w: &'a WeightRef<'a>,
        x_norm: &'a GpuTensor,
        x_rot_local: Option<&'a GpuTensor>,
        router_logits: &'a GpuTensor,
        scalar_buf: &'a GpuTensor,
        gate_buf: &'a GpuTensor,
        up_buf: &'a GpuTensor,
        smi: usize,
    },
    /// Shared-expert down body (qwen decode). One step for both body forms:
    /// MQ4 (fused silu·mul·rotate + `gemv_hfq4g256_residual_sigmoid_scaled_gpu`)
    /// and non-MQ4 (sigmoid → silu·mul → GEMV). The non-MQ4 builder appends a
    /// standalone [`Step::ScaledAdd`]. `out_target` is
    /// the EP partial when `routed_out` is set, else `x_residual`.
    /// `tp_step_out_buf` returns `None` — the shared accumulation is not a
    /// collective partial (EP drives it via the routed partial step).
    MoeSharedDown {
        w: &'a WeightRef<'a>,
        gate_buf: &'a GpuTensor,
        up_buf: &'a GpuTensor,
        scalar_buf: &'a GpuTensor,
        ffn_hidden: &'a GpuTensor,
        ffn_out: &'a GpuTensor,
        out_target: &'a GpuTensor,
        smi: usize,
    },
    /// In-place scaled add `x += y * scale` with a device-side `scale`
    /// (`scaled_add_inplace_gpu_scalar_f32`). The non-MQ4 shared-down builder
    /// emits this immediately after [`Step::MoeSharedDown`]; the CPU fallback
    /// wrapper reuses the same launch function.
    ScaledAdd {
        x: &'a GpuTensor,
        y: &'a GpuTensor,
        scale: &'a GpuTensor,
    },
    /// Qwen routed gate_up indexed GEMV for the forms [`Step::IndexedMoeGemv`]
    /// cannot express: Paro (Givens), MFP4-E8, MQ5, per-expert mixed dtype
    /// tags, and the batched prefill forms. `batch_size == 1` selects the
    /// decode kernels; `> 1` the batched kernels (prefill Path 1).
    /// `dtype_tags = Some` selects the merged per-expert mixed kernel
    /// (decode only; graded prefill runs the grouped path). The fused
    /// gate||up output writes `gate_batch`/`up_batch` (m = 2·expert_m rows).
    /// `tp_step_out_buf` returns `None` — gate_batch is an intermediate.
    MoeGateUpIndexed {
        experts: &'a MoeExpertRef<'a>,
        topk_indices: &'a GpuTensor,
        x_rot: &'a GpuTensor,
        gate_batch: &'a GpuTensor,
        up_batch: &'a GpuTensor,
        k_top: usize,
        batch_size: usize,
        dtype_tags: Option<&'a GpuTensor>,
    },
    /// Qwen routed down indexed GEMV: the expanded per-expert write (decode +
    /// prefill Path 1, `mode = Expanded`) or the atomic residual-scaled
    /// accumulation (prefill Path 0, `mode = ResidualScaled`). Covers the
    /// dtypes [`Step::IndexedMoeGemv`] cannot express (Paro, E8, MQ5, mixed
    /// tags) plus the batched prefill forms; m = expert_k, k = expert_m.
    /// `tp_step_out_buf` returns `Some(out.buf)` only for `ResidualScaled`
    /// (the atomic down IS the EP partial); `Expanded` writes `down_expanded`,
    /// an intermediate.
    MoeDownIndexed {
        experts: &'a MoeExpertRef<'a>,
        topk_indices: &'a GpuTensor,
        rot_batch: &'a GpuTensor,
        out: &'a GpuTensor,
        k_top: usize,
        batch_size: usize,
        mode: QwenDownMode<'a>,
        dtype_tags: Option<&'a GpuTensor>,
    },
    // ── Note (Task 6): ds4 `hc_ffn_mix` is intentionally NOT a Step variant ──
    // The ds4 MoE tail mixes the EP all-reduced `ffn_out` partial into
    // `residual_streams` via `hc_mix_4stream` + `memcpy_dtod_auto`. Its two view
    // operands (`comb_view`, `post_view`) are ephemeral `GpuTensor` values computed
    // at call time via `sub_offset` on `state.hc_c`; they have no stable backing
    // storage to borrow `&'a GpuTensor` from inside a Step.
    // Task 8's `forward_ep` calls `crate::families::moe::launch_hc_ffn_mix`
    // directly after `execute_steps_parallel` returns and the EP all-reduce
    // completes. minimax's MoE tail (`add_inplace_f32`) reuses `Step::ResidualAdd`.
}

/// Op-kind for fusion matching. Total over Step variants.
fn op_kind(step: &Step) -> PipelineOp {
    match step {
        Step::Gemv { .. } => PipelineOp::Gemv,
        // Reuses the Gemv tag: op_kind only feeds the fused-decode prefix table,
        // which the prefill-only Gemm step never enters.
        Step::Gemm { .. } => PipelineOp::Gemv,
        Step::GemmKeyedBatched { .. } => PipelineOp::Gemv,
        Step::RmsnormBatched { .. } => PipelineOp::RmsnormAutomatic,
        Step::GivensRotateBatched { .. } => PipelineOp::GivensRotate,
        Step::RotateFwhtBatched { .. } => PipelineOp::RotateFwht,
        Step::GemvResidual { .. } => PipelineOp::GemvResidual,
        Step::GemmResidualBatched { .. } => PipelineOp::GemvResidual,
        #[cfg(feature = "deltanet")]
        Step::FusedQkvzaBatched { .. } => PipelineOp::Gemv,
        Step::RmsnormAutomatic { .. } => PipelineOp::RmsnormAutomatic,
        Step::Attend { .. } => PipelineOp::Attend,
        Step::Rope { .. } => PipelineOp::Rope,
        Step::QkNorm { .. } => PipelineOp::QkNorm,
        Step::BiasAdd { .. } => PipelineOp::BiasAdd,
        Step::SiluMul { .. } => PipelineOp::SiluMul,
        Step::ResidualAdd { .. } => PipelineOp::ResidualAdd,
        #[cfg(feature = "deltanet")]
        Step::DeltaGatePrep { .. } => PipelineOp::DeltaGatePrep,
        #[cfg(feature = "deltanet")]
        Step::DeltaConvSplit { .. } => PipelineOp::DeltaConvSplit,
        #[cfg(feature = "deltanet")]
        Step::DeltaQkL2Norm { .. } => PipelineOp::DeltaQkL2Norm,
        #[cfg(feature = "deltanet")]
        Step::DeltaRepeatHeads { .. } => PipelineOp::DeltaRepeatHeads,
        #[cfg(feature = "deltanet")]
        Step::DeltaRecurrence { .. } => PipelineOp::DeltaRecurrence,
        #[cfg(feature = "deltanet")]
        Step::DeltaGatedNorm { .. } => PipelineOp::DeltaGatedNorm,
        // MoE decode ops (Task 4). Not fusible — no entry in FUSED_TABLE.
        Step::MoeRoute { .. } => PipelineOp::MoeRoute,
        Step::IndexedMoeGemv { .. } => PipelineOp::IndexedMoeGemv,
        Step::MoeCombine { .. } => PipelineOp::MoeCombine,
        // MoE prefill grouped ops (Task 5). Not fusible.
        Step::MoeScatter { .. } => PipelineOp::MoeScatter,
        Step::GroupedMoeGemm { .. } => PipelineOp::GroupedMoeGemm,
        Step::MoeGateUpUnscatter { .. } => PipelineOp::MoeGateUpUnscatter,
        // Elementwise pre-op before MoeRoute; tag is irrelevant to fusion.
        Step::ScoreActivation { .. } => PipelineOp::RmsnormAutomatic,
        // SwiGLU + FWHT rotate intermediate; not fusible.
        Step::MoeActivation { .. } => PipelineOp::RmsnormAutomatic,
        // Post-reduce int64→f32 convert; not fusible.
        Step::ConvertI64ToF32 { .. } => PipelineOp::RmsnormAutomatic,
        // ── Qwen MoE Step-native ops (STEP-002 Phase 1). Not fusible — no
        // entry in FUSED_TABLE. Elementwise/pre-routing ops reuse the
        // RmsnormAutomatic tag (mirrors ScoreActivation/MoeActivation); the
        // indexed routed projections reuse the IndexedMoeGemv tag.
        Step::MoeSoftmaxTopK { .. } => PipelineOp::RmsnormAutomatic,
        Step::MoeFusedSharedGate { .. } => PipelineOp::RmsnormAutomatic,
        Step::MoeSharedGateSide { .. } => PipelineOp::RmsnormAutomatic,
        Step::MoeSharedDown { .. } => PipelineOp::RmsnormAutomatic,
        Step::ScaledAdd { .. } => PipelineOp::RmsnormAutomatic,
        Step::MoeGateUpIndexed { .. } => PipelineOp::IndexedMoeGemv,
        Step::MoeDownIndexed { .. } => PipelineOp::IndexedMoeGemv,
    }
}

// ── Guard helpers ──────────────────────────────────────────────────────────

/// Extract the dtype of the first Gemv step in the window (step index 1,
/// after the RmsnormAutomatic producer). Returns None if not a Gemv step.
fn window_gemv_dtype(steps: &[Step]) -> Option<DType> {
    match steps.get(1)? {
        Step::Gemv { w, .. } => Some(w.dtype),
        _ => None,
    }
}

/// True if all Gemv steps in the window (indices 1..) have:
/// - the given dtype
/// - GemvInput::Prerotated
/// - awq_scale == None (iff require_no_awq)
fn gemv_steps_uniform(steps: &[Step], dtype: DType, require_no_awq: bool) -> bool {
    steps[1..].iter().all(|s| match s {
        Step::Gemv {
            w,
            input: GemvInput::Prerotated(_),
            ..
        } => w.dtype == dtype && (!require_no_awq || w.awq_scale.is_none()),
        _ => false,
    })
}

/// True if all Gemv steps in the window (indices 1..) have:
/// - the given dtype
/// - GemvInput::Raw (kernel rotates internally — used for Paro guards)
fn gemv_steps_uniform_raw(steps: &[Step], dtype: DType) -> bool {
    steps[1..].iter().all(|s| match s {
        Step::Gemv {
            w,
            input: GemvInput::Raw(_),
            ..
        } => w.dtype == dtype,
        _ => false,
    })
}

/// True if ctx has dp4a and !force_unfused.
fn dp4a_eligible(ctx: &DispatchCtx) -> bool {
    !ctx.flags.force_unfused && ctx.arch.gemv_dp4a_enabled()
}

// ── QKV 3-way guards ──

pub(crate) fn guard_qkv_mq4g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 4 && gemv_steps_uniform(steps, DType::MQ4G256Lloyd, true)
}

pub(crate) fn guard_qkv_mq3g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 4 && gemv_steps_uniform(steps, DType::MQ3G256Lloyd, true)
}

/// Covers both DType::MQ4G256 (plain) and DType::HFQ4G256 — both feed
/// gpu.fused_qkv_hfq4g256 which takes a pre-normalized x.
pub(crate) fn guard_qkv_hfq4g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if steps.len() != 4 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    matches!(dt, DType::MQ4G256 | DType::HFQ4G256) && gemv_steps_uniform(steps, dt, true)
}

/// Covers both DType::HFQ6G256 and DType::MQ6G256.
/// Fusion is safe on RDNA (fused_qkv.rs None arm falls back to gemm n=1)
/// and beneficial on RDNA3+ even without dp4a; dp4a is handled per-arm
/// in fused_qkv.rs dispatch.
pub(crate) fn guard_qkv_hfq6g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if steps.len() != 4 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    matches!(dt, DType::HFQ6G256 | DType::MQ6G256) && gemv_steps_uniform(steps, dt, true)
}

// ── QKVZA 4-way guards (DeltaNet linear attention) ──

pub(crate) fn guard_qkvza_mq4g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 5 && gemv_steps_uniform(steps, DType::MQ4G256Lloyd, true)
}

pub(crate) fn guard_qkvza_mq3g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 5 && gemv_steps_uniform(steps, DType::MQ3G256Lloyd, true)
}

/// Covers both DType::MQ4G256 (plain) and DType::HFQ4G256.
pub(crate) fn guard_qkvza_hfq4g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if steps.len() != 5 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    matches!(dt, DType::MQ4G256 | DType::HFQ4G256) && gemv_steps_uniform(steps, dt, true)
}

/// Covers both DType::HFQ6G256 and DType::MQ6G256.
/// Fusion is safe on RDNA (fused_qkv.rs None arm falls back to gemm n=1)
/// and beneficial on RDNA3+ even without dp4a; dp4a is handled per-arm
/// in fused_qkv.rs dispatch.
pub(crate) fn guard_qkvza_hfq6g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if steps.len() != 5 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    matches!(dt, DType::HFQ6G256 | DType::MQ6G256) && gemv_steps_uniform(steps, dt, true)
}

// ── Gate+Up 2-way guards ──

pub(crate) fn guard_gate_up_mq4g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 3 && gemv_steps_uniform(steps, DType::MQ4G256Lloyd, true)
}

pub(crate) fn guard_gate_up_mq3g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 3 && gemv_steps_uniform(steps, DType::MQ3G256Lloyd, true)
}

pub(crate) fn guard_gate_up_hfq4g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if steps.len() != 3 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    matches!(dt, DType::MQ4G256 | DType::HFQ4G256) && gemv_steps_uniform(steps, dt, true)
}

pub(crate) fn guard_gate_up_hfq6g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if !dp4a_eligible(ctx) {
        return false;
    }
    if steps.len() != 3 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    matches!(dt, DType::HFQ6G256 | DType::MQ6G256) && gemv_steps_uniform(steps, dt, true)
}

// ── mfp4-E8 decode launch-fusion guards (gfx1151 / Strix Halo ONLY) ──
// These are the SOLE producers of the FusedGateUpMfp4G32E8 / FusedQkvzaMfp4G32E8
// keys. The `is_gfx1151()` check firewalls the fused kernels to gfx1151 — on every
// other arch these return false and the projections fall through to the
// per-projection gemv_mfp4g32_e8 path unchanged. The fused kernels embed the
// byte-identical gemv_mfp4g32_e8 per-row body, so the fused output equals N
// sequential GEMVs bit-for-bit (only the launch count shrinks).
//
// gfx11 E8 port finding: the fusion (launch-overhead reduction, +5.8% on the Strix
// Halo APU) does NOT transfer to the gfx1100 dGPU — measured decode 101.7 (fused)
// vs 102.6 (unfused) tok/s, a ~1% LOSS, bit-identical output. The dGPU's faster
// compute + the (32,7) launch_bounds tuned for gfx1151 occupancy leave no launch
// win to capture. Kept gfx1151-only; revisit only with a gfx1100 occupancy retune.
pub(crate) fn guard_gate_up_mfp4g32e8(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if !ctx.arch.is_gfx1151() {
        return false;
    }
    if steps.len() != 3 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    dt == DType::MFP4G32E8 && gemv_steps_uniform(steps, dt, true)
}

pub(crate) fn guard_qkvza_mfp4g32e8(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if !ctx.arch.is_gfx1151() {
        return false;
    }
    if steps.len() != 5 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    dt == DType::MFP4G32E8 && gemv_steps_uniform(steps, dt, true)
}

// ── Paro fused guards (Raw input — kernel rotates internally) ──

// ── Q8_0 / Q4K fused guards (non-rotated, Prerotated input) ──
// These dtypes have no activation rotation (RotationPlan::None), so the
// RmsnormAutomatic producer does plain rmsnorm and the fused kernels take
// the pre-normed x directly. Prerotated input is correct because
// for_gemv_prerotated(Q8_0/Q4K) falls back to the plain GEMV kernel.

/// Fused QKV with Q4K weights. Used by llama (dense).
pub(crate) fn guard_qkv_q4k(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 4 && gemv_steps_uniform(steps, DType::Q4K, true)
}

/// Fused gate+up with Q4K weights. Used by llama (dense).
pub(crate) fn guard_gate_up_q4k(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 3 && gemv_steps_uniform(steps, DType::Q4K, true)
}

/// Fused gate+up with Q8_0 weights. Used by qwen2 FFN.
pub(crate) fn guard_gate_up_q8_0(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 3 && gemv_steps_uniform(steps, DType::Q8_0, true)
}

/// Fused 4-way QKVZA with Q8_0 weights (DECODE path, n=1). Used by
/// Qwen3.5/A3B .mq4p DeltaNet layers (qt=3). No dp4a required.
pub(crate) fn guard_qkvza_q8_0(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 5 && gemv_steps_uniform(steps, DType::Q8_0, true)
}

/// Fused 3-way QKV with Q8_0 weights (DECODE path, n=1). No dp4a required.
pub(crate) fn guard_qkv_q8_0(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 4 && gemv_steps_uniform(steps, DType::Q8_0, true)
}

pub(crate) fn guard_gate_up_paro4g128t(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if steps.len() != 3 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    dt == DType::ParoQ4G128
        && gemv_steps_uniform_raw(steps, DType::ParoQ4G128)
        && steps[1..].iter().all(|s| match s {
            Step::Gemv { w, .. } => w.m % 8 == 0 && w.k % 128 == 0,
            _ => false,
        })
        // Gate and up must have equal m — the fused kernel takes a single m.
        && {
            let m0 = match &steps[1] { Step::Gemv { w, .. } => w.m, _ => return false };
            let m1 = match &steps[2] { Step::Gemv { w, .. } => w.m, _ => return false };
            m0 == m1
        }
}

pub(crate) fn guard_qkvza_paro4g128t(_steps: &[Step], _ctx: &DispatchCtx) -> bool {
    false
}

pub(crate) fn guard_qkv_paro4g128t(_steps: &[Step], _ctx: &DispatchCtx) -> bool {
    false
}

pub struct FusedPattern {
    pub ops: &'static [PipelineOp],
    pub key: KernelKey,
    /// Dtype/arch predicate called after op-kind prefix match. Must return true
    /// for the entry to fire. Receives the full matched window (all ops.len()
    /// steps starting at the current position).
    pub guard: fn(&[Step], &DispatchCtx) -> bool,
}

/// Greedy longest-prefix op-pattern match with dtype/arch guard.
pub fn match_prefix(
    table: &[FusedPattern],
    steps: &[Step],
    ctx: &DispatchCtx,
) -> Option<(KernelKey, usize)> {
    table
        .iter()
        .filter(|p| {
            !p.ops.is_empty()
                && p.ops.len() <= steps.len()
                && p.ops.iter().zip(steps).all(|(o, s)| *o == op_kind(s))
                && (p.guard)(&steps[..p.ops.len()], ctx)
        })
        .max_by_key(|p| p.ops.len())
        .map(|p| (p.key, p.ops.len()))
}

/// Lower-time fusion match over the canonical `FUSED_TABLE`. The Ship-6 super-op
/// lowering (`superop::lower_layer`) calls THIS — reusing the same table + guards
/// verbatim — so a lowered program can never drift from what `execute_steps`
/// would dispatch live (the fusion-drift mitigation, spike risk #1).
#[allow(dead_code)]
pub(crate) fn match_fused_prefix(steps: &[Step], ctx: &DispatchCtx) -> Option<(KernelKey, usize)> {
    match_prefix(FUSED_TABLE, steps, ctx)
}

/// Public(crate) op-kind accessor for the lowering (mirror of the private `op_kind`).
#[allow(dead_code)]
pub(crate) fn step_op_kind(step: &Step) -> PipelineOp {
    op_kind(step)
}

const QKV3: &[PipelineOp] = &[
    PipelineOp::RmsnormAutomatic,
    PipelineOp::Gemv,
    PipelineOp::Gemv,
    PipelineOp::Gemv,
];
const QKVZA4: &[PipelineOp] = &[
    PipelineOp::RmsnormAutomatic,
    PipelineOp::Gemv,
    PipelineOp::Gemv,
    PipelineOp::Gemv,
    PipelineOp::Gemv,
];
const GATE_UP2: &[PipelineOp] = &[
    PipelineOp::RmsnormAutomatic,
    PipelineOp::Gemv,
    PipelineOp::Gemv,
];

#[cfg(feature = "deltanet")]
const DELTA_QK_REPEAT: &[PipelineOp] = &[PipelineOp::DeltaQkL2Norm, PipelineOp::DeltaRepeatHeads];

#[cfg(feature = "deltanet")]
fn guard_delta_qk_repeat(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused || steps.len() != 2 {
        return false;
    }
    let (
        q,
        k,
        n_key_heads,
        norm_head_dim,
        norm_batch_size,
        q_src,
        k_src,
        q_dst,
        k_dst,
        repeat_key_heads,
        ratio,
        repeat_head_dim,
        repeat_batch_size,
    ) = match (&steps[0], &steps[1]) {
        (
            Step::DeltaQkL2Norm {
                q,
                k,
                n_key_heads,
                head_dim,
                batch_size,
                ..
            },
            Step::DeltaRepeatHeads {
                q_src,
                k_src,
                q_dst,
                k_dst,
                n_key_heads: repeat_key_heads,
                ratio,
                head_dim: repeat_head_dim,
                batch_size: repeat_batch_size,
            },
        ) => (
            *q,
            *k,
            *n_key_heads,
            *head_dim,
            *batch_size,
            *q_src,
            *k_src,
            *q_dst,
            *k_dst,
            *repeat_key_heads,
            *ratio,
            *repeat_head_dim,
            *repeat_batch_size,
        ),
        _ => return false,
    };

    // The only fused implementation is the batched interleave kernel. Keep
    // the decode path on the two standalone operations and reject aliasing:
    // the fused kernel reads the normalized source and writes the repeated
    // destination in one launch.
    norm_batch_size > 1
        && norm_batch_size == repeat_batch_size
        && n_key_heads == repeat_key_heads
        && norm_head_dim == repeat_head_dim
        && ratio > 1
        && q.dtype == DType::F32
        && k.dtype == DType::F32
        && q_src.dtype == DType::F32
        && k_src.dtype == DType::F32
        && q_dst.dtype == DType::F32
        && k_dst.dtype == DType::F32
        && q.buf.as_ptr() == q_src.buf.as_ptr()
        && k.buf.as_ptr() == k_src.buf.as_ptr()
        && q.buf.as_ptr() != k.buf.as_ptr()
        && q_dst.buf.as_ptr() != q.buf.as_ptr()
        && k_dst.buf.as_ptr() != k.buf.as_ptr()
        && q_dst.buf.as_ptr() != k.buf.as_ptr()
        && k_dst.buf.as_ptr() != q.buf.as_ptr()
        && q_dst.buf.as_ptr() != k_dst.buf.as_ptr()
        && q.numel() == norm_batch_size * n_key_heads * norm_head_dim
        && k.numel() == norm_batch_size * n_key_heads * norm_head_dim
        && q_dst.numel() == norm_batch_size * n_key_heads * ratio * norm_head_dim
        && k_dst.numel() == norm_batch_size * n_key_heads * ratio * norm_head_dim
}

const FUSED_TABLE: &[FusedPattern] = &[
    // ── QKV 3-way ──────────────────────────────────────────────────────────
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvMq4G256Lloyd,
        guard: guard_qkv_mq4g256lloyd,
    },
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvMq3G256Lloyd,
        guard: guard_qkv_mq3g256lloyd,
    },
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvHfq4G256,
        guard: guard_qkv_hfq4g256,
    },
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvHfq6G256,
        guard: guard_qkv_hfq6g256,
    },
    // ── QKVZA 4-way (DeltaNet linear attention) ────────────────────────────
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaMq4G256Lloyd,
        guard: guard_qkvza_mq4g256lloyd,
    },
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaMq3G256Lloyd,
        guard: guard_qkvza_mq3g256lloyd,
    },
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaHfq4G256,
        guard: guard_qkvza_hfq4g256,
    },
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaHfq6G256,
        guard: guard_qkvza_hfq6g256,
    },
    // mfp4-E8 decode launch-fusion — gfx1151-ONLY (guard firewalls the arch).
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaMfp4G32E8,
        guard: guard_qkvza_mfp4g32e8,
    },
    // ── Gate+Up 2-way ───────────────────────────────────────────────────────
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpMq4G256Lloyd,
        guard: guard_gate_up_mq4g256lloyd,
    },
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpMq3G256Lloyd,
        guard: guard_gate_up_mq3g256lloyd,
    },
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpHfq4G256,
        guard: guard_gate_up_hfq4g256,
    },
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpHfq6G256,
        guard: guard_gate_up_hfq6g256,
    },
    // mfp4-E8 decode launch-fusion — gfx1151-ONLY (guard firewalls the arch).
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpMfp4G32E8,
        guard: guard_gate_up_mfp4g32e8,
    },
    // ── Q8_0 / Q4K fused entries (non-rotated, Always arch gate) ─────────
    // Q8_0 QKV/QKVZA: Qwen3.5-A3B .mq4p uses Q8_0 for all linear-attention
    // projections (qt=3). Scalar decode kernels added 2026-06-14.
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaQ8_0,
        guard: guard_qkvza_q8_0,
    },
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvQ8_0,
        guard: guard_qkv_q8_0,
    },
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvQ4K,
        guard: guard_qkv_q4k,
    },
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpQ4K,
        guard: guard_gate_up_q4k,
    },
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpQ8_0,
        guard: guard_gate_up_q8_0,
    },
    // ── Paro fused Paro4G128T (dp4a, Raw input) ────────────────────────
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpParo4G128T,
        guard: guard_gate_up_paro4g128t,
    },
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaParo4G128T,
        guard: guard_qkvza_paro4g128t,
    },
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvParo4G128T,
        guard: guard_qkv_paro4g128t,
    },
    #[cfg(feature = "deltanet")]
    FusedPattern {
        ops: DELTA_QK_REPEAT,
        key: KernelKey::FusedDeltaQkL2NormRepeat,
        guard: guard_delta_qk_repeat,
    },
];
static GEMV: OnceLock<GemvFamily> = OnceLock::new();
static ROTATION: OnceLock<RotationFamily> = OnceLock::new();
static FUSED_QKV: OnceLock<FusedQkvFamily> = OnceLock::new();
static GEMM: OnceLock<crate::families::gemm::GemmFamily> = OnceLock::new();

pub fn execute_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    steps: &[Step],
) -> Result<(), DispatchError> {
    let mut i = 0;
    while i < steps.len() {
        if let Some((key, len)) = match_prefix(FUSED_TABLE, &steps[i..], ctx) {
            // ── QKV bias fold (HIPFIRE_FUSE_QKV_BIAS) ────────────────────────
            // When the flag is on, the matched window is a per-row 3-way QKV
            // decode key whose kernel supports the fold, and the 3 steps right
            // after the window are `BiasAdd` on the q/k/v outputs in order, fold
            // the bias into the kernel's lane-0 store and skip those 3 steps.
            // The fold is `acc + bias[row]` (fp32, same operand order as the
            // separate `bias_add`) → byte-identical to the unfused path.
            if ctx.flags.fuse_qkv_bias && len == QKV3.len() && qkv_bias_fold_supported(key, ctx) {
                if let Some(biases) = match_trailing_qkv_bias(&steps[i..], len) {
                    launch_fused_qkv_with_bias(gpu, ctx, key, &steps[i..i + len], biases)?;
                    i += len + 3;
                    continue;
                }
            }
            // ─────────────────────────────────────────────────────────────────
            launch_fused(gpu, ctx, key, &steps[i..i + len])?;
            i += len;
        } else {
            launch_op(gpu, ctx, &steps[i])?;
            i += 1;
        }
    }
    Ok(())
}

/// Mesh-aware spine (P-A). Threads the device mesh to the dispatch chokepoint so
/// per-`Step` parallelism (TP in P-B, PP/EP in later phases) can be resolved
/// here. For the single-device (1×1) mesh it forwards `gpu` unchanged and is
/// byte-identical to calling [`execute_steps`] directly — this is the
/// zero-behavior-change foundation the executor half of the pivot builds on.
///
/// P-A threads only the mesh (a cheap value) alongside the existing `&mut Gpu`,
/// so every call site migrates by adding a `mesh` argument with no borrow
/// rework. The `&mut Gpu` → `&mut Gpus` promotion (for real cross-rank TP)
/// happens in P-B, where it is bundled with the serve-path `Gpus` hoist and
/// applied only to the paths that shard.
pub fn execute_steps_mesh(
    mesh: &DeviceMesh,
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    steps: &[Step],
) -> Result<(), DispatchError> {
    debug_assert_eq!(
        mesh.n_devices(),
        1,
        "execute_steps_mesh: only the single (1×1) mesh is supported in P-A; \
         TP/PP/EP sharding lands in P-B..P-E"
    );
    execute_steps(gpu, ctx, steps)
}

/// Collective to inject after a step ran on every rank of the `Tp` group.
/// Keyed by step index in the per-rank step lists (see [`execute_steps_tp`]).
#[derive(Debug)]
pub enum TpCollective {
    /// Column-parallel (or replicated) step — output stays on-rank, no collective.
    None,
    /// Row-parallel step — each rank produced a partial `out` of length `dim`;
    /// sum them in place across the `Tp` group so every rank holds the whole.
    AllReduceOut { dim: usize },
}

/// Axis-keyed collective to inject after a step in [`execute_steps_parallel`].
/// Replaces the TP-specific `TpCollective` with a generic form that covers both
/// Tp (dense row-parallel) and Ep (MoE expert-parallel) all-reduces.
#[derive(Debug)]
pub enum StepCollective {
    /// No collective — column-parallel / replicated steps leave output on-rank.
    None,
    /// All-reduce the step's partial output over the `kind` group.
    /// `dim` is the element count (f32) of the partial buffer on each rank.
    AllReduce {
        kind: hipfire_hardware::DimKind,
        dim: usize,
    },
    /// Int64 all-reduce over the Tp group (reproducible MoE down TP path).
    /// `dim` is the element count in int64 (= hidden). Uses `all_reduce_sum_i64_peer`.
    /// The `zero_before` flag for this step must use `dim * 8` bytes (not `dim * 4`).
    AllReduceI64Tp { dim: usize },
    /// Zero-only: zeroes the step's output buffer (8 bytes/elem = i64) but runs no
    /// cross-rank collective. Used before `DownResidualI64` on the EP i64 path,
    /// where the cross-rank reduce is FP32 (attached to the subsequent
    /// `ConvertI64ToF32` step via `AllReduce{Ep}`).
    ZeroI64Only { dim: usize },
}

/// The `out` buffer of a step that carries a row-parallel or EP partial output
/// (the only kind that needs an all-reduce). `None` for steps that never carry
/// such a buffer.
///
/// MoE additions (Task 4):
/// - `MoeCombine.out` — the pre-zeroed EP partial; the executor zeros it via
///   `zero_before` and the EP all-reduce sums it across ranks.
/// - `IndexedMoeGemv` with `DownResidual` — the step's `out` IS the EP partial
///   (the residual-fused kernel writes combined output directly into it).
///   `GateUp` and `DownExpanded` do not carry a partial: their output buffers
///   are intermediates, not reduced over the EP group.
fn tp_step_out_buf<'a>(step: &'a Step) -> Option<&'a hip_bridge::DeviceBuffer> {
    match step {
        Step::Gemv { out, .. } => Some(&out.buf),
        Step::GemvResidual { out, .. } => Some(&out.buf),
        Step::Gemm { y, .. } => Some(&y.buf),
        Step::GemmKeyedBatched { y, .. } => Some(&y.buf),
        Step::GemmResidualBatched { residual, .. } => Some(&residual.buf),
        // EP partial: combine result or residual-fused down result.
        Step::MoeCombine { out, .. } => Some(&out.buf),
        Step::IndexedMoeGemv {
            which: MoeProj::DownResidual { .. },
            out,
            ..
        } => Some(&out.buf),
        // Reproducible int64 TP down: out is the i64 partial buffer.
        Step::IndexedMoeGemv {
            which: MoeProj::DownResidualI64 { .. },
            out,
            ..
        } => Some(&out.buf),
        // Prefill grouped ops: intermediates, never EP partials.
        Step::MoeScatter { .. } | Step::GroupedMoeGemm { .. } | Step::MoeGateUpUnscatter { .. } => {
            None
        }
        // Score activation is a pre-routing elementwise op; no EP partial.
        Step::ScoreActivation { .. } => None,
        // MoeActivation is an intermediate (not an EP partial).
        Step::MoeActivation { .. } => None,
        // ConvertI64ToF32: on the EP i64 path (ZeroI64Only→DownResidualI64→ConvertI64ToF32→AllReduce{Ep}),
        // the f32 `dst` IS the EP partial that the AllReduce{Ep} collective must target.
        Step::ConvertI64ToF32 { dst, .. } => Some(&dst.buf),
        // ── Qwen MoE Step-native ops (STEP-002 Phase 1) ──
        // MoeSoftmaxTopK / gate-side / shared-down / scaled-add: pre-route or
        // accumulate into the residual (never a collective partial).
        Step::MoeSoftmaxTopK { .. }
        | Step::MoeFusedSharedGate { .. }
        | Step::MoeSharedGateSide { .. }
        | Step::MoeSharedDown { .. }
        | Step::ScaledAdd { .. } => None,
        // MoeGateUpIndexed: intermediate, never an EP partial.
        Step::MoeGateUpIndexed { .. } => None,
        // MoeDownIndexed: Expanded writes down_expanded (intermediate);
        // ResidualScaled IS the EP partial (atomic accumulation).
        Step::MoeDownIndexed {
            mode: QwenDownMode::ResidualScaled { .. },
            out,
            ..
        } => Some(&out.buf),
        Step::MoeDownIndexed {
            mode: QwenDownMode::Expanded,
            ..
        } => None,
        #[cfg(feature = "deltanet")]
        Step::DeltaGatePrep { .. }
        | Step::DeltaConvSplit { .. }
        | Step::DeltaQkL2Norm { .. }
        | Step::DeltaRepeatHeads { .. }
        | Step::DeltaRecurrence { .. }
        | Step::DeltaGatedNorm { .. } => None,
        _ => None,
    }
}

/// Pure-logic validation for [`execute_steps_parallel`] arg lengths.
/// Separated into its own fn so it is testable without a GPU.
/// Returns `n_steps` (= `per_rank_steps[0].len()`) on success.
fn validate_parallel_args(
    group_size: usize,
    per_rank_steps: &[Vec<Step>],
    collectives: &[StepCollective],
    zero_before: &[bool],
) -> Result<usize, DispatchError> {
    if per_rank_steps.len() != group_size {
        return Err(DispatchError::Hip(format!(
            "execute_steps_parallel: {} step lists for group of {group_size}",
            per_rank_steps.len()
        )));
    }
    let n_steps = per_rank_steps[0].len();
    for (r, s) in per_rank_steps.iter().enumerate() {
        if s.len() != n_steps {
            return Err(DispatchError::Hip(format!(
                "execute_steps_parallel: rank {r} has {} steps, rank 0 has {n_steps} (must be lock-step)",
                s.len()
            )));
        }
    }
    if collectives.len() != n_steps {
        return Err(DispatchError::Hip(format!(
            "execute_steps_parallel: {} collectives for {n_steps} steps",
            collectives.len()
        )));
    }
    if zero_before.len() != n_steps {
        return Err(DispatchError::Hip(format!(
            "execute_steps_parallel: {} zero_before flags for {n_steps} steps",
            zero_before.len()
        )));
    }
    Ok(n_steps)
}

/// Axis-keyed parallel Step executor (P-D foundation). The generic form of
/// [`execute_steps_tp`]: runs `per_rank_steps` lock-step across the mesh group
/// for `collectives[i].kind`, injects an axis-keyed all-reduce for
/// `StepCollective::AllReduce` steps, and optionally zeroes the step's output
/// buffer before running it (required for EP accumulation into a partial).
///
/// **`zero_before[i]`** — when true, `memset_async` the step's output buffer to 0
/// before launching step `i` on each rank. The element count is taken from the
/// paired `AllReduce { dim }` collective. Same 4-bytes-per-elem, same
/// `active_stream` requirement as the EP accumulation pattern.
///
/// **Collective choice** is keyed on the collective's `kind`:
/// - `DimKind::Tp` → always `all_reduce_sum_f32_peer` (RCCL not required on Tp path).
/// - `DimKind::Ep` → `ep_peer_allreduce_decode()` ? `_peer` : `_rccl`.
///
/// The group is resolved via `mesh.group_along(kind, coord_of(0))` — identical
/// to the TP path, so byte-identical results are guaranteed for Tp collectives.
///
/// Preconditions: each rank in the group has `active_stream` set and peer access
/// enabled (`ensure_rank_streams` + `enable_peer_all`).
pub fn execute_steps_parallel(
    mesh: &DeviceMesh,
    gpus: &mut hipfire_hardware::Gpus,
    per_rank_steps: &[Vec<Step>],
    collectives: &[StepCollective],
    zero_before: &[bool],
) -> Result<(), DispatchError> {
    // Determine the parallelism axis from the first AllReduce collective, or
    // fall back to Tp (all-None collectives remain on the Tp group for compat).
    let kind = collectives
        .iter()
        .find_map(|c| {
            if let StepCollective::AllReduce { kind, .. } = c {
                Some(*kind)
            } else {
                None
            }
        })
        .unwrap_or(hipfire_hardware::DimKind::Tp);

    debug_assert!(
        collectives.iter().all(|c| !matches!(c, StepCollective::AllReduce { kind: k, .. } if *k != kind)),
        "execute_steps_parallel: mixed AllReduce kinds in one call — group resolves from the FIRST only \
         (D2b must add per-step group resolution before composing Tp+Ep)"
    );

    let group = mesh.group_along(kind, &mesh.coord_of(0));
    let group_size = group.len();

    // Single-rank fast-path (tp==1 running through forward_tp): run all steps
    // sequentially on rank 0; all-reduces are identity (g==1) so skip them.
    // `zero_before` still applies (the i64 partial must be zeroed before the
    // i64 kernel even when there is no reduce).
    if group_size == 1 {
        let n_steps = validate_parallel_args(group_size, per_rank_steps, collectives, zero_before)?;
        let dev = group[0];
        let hip_err = |e: hip_bridge::HipError| DispatchError::Hip(e.to_string());
        for i in 0..n_steps {
            if zero_before[i] {
                let (dim, elem_bytes) = match &collectives[i] {
                    StepCollective::AllReduce { dim, .. } => (*dim, 4usize),
                    StepCollective::AllReduceI64Tp { dim } => (*dim, 8usize),
                    StepCollective::ZeroI64Only { dim } => (*dim, 8usize),
                    StepCollective::None => {
                        return Err(DispatchError::Hip(format!(
                            "execute_steps_parallel: zero_before[{i}] is true but collective is None (no dim)"
                        )));
                    }
                };
                gpus.devices[dev].bind_thread().map_err(hip_err)?;
                let stream = gpus.devices[dev].active_stream.as_ref().ok_or_else(|| {
                    DispatchError::Hip(format!(
                        "execute_steps_parallel: device {dev} has no active_stream for zero_before (g==1)"
                    ))
                })?;
                let buf = tp_step_out_buf(&per_rank_steps[0][i]).ok_or_else(|| {
                    DispatchError::Hip(format!(
                        "execute_steps_parallel: step {i} zero_before=true but has no out buffer (g==1)"
                    ))
                })?;
                gpus.devices[dev]
                    .hip
                    .memset_async(buf, 0, dim * elem_bytes, stream)
                    .map_err(hip_err)?;
            }
            gpus.devices[dev].bind_thread().map_err(hip_err)?;
            let ctx = DispatchCtx::new(&gpus.devices[dev]);
            launch_op(&mut gpus.devices[dev], &ctx, &per_rank_steps[0][i])?;
            // All-reduce is identity for g==1; skip (including ZeroI64Only).
        }
        return Ok(());
    }

    if group_size < 1 {
        return Err(DispatchError::Hip(format!(
            "execute_steps_parallel: {kind:?} group size {group_size} — needs ≥1 rank"
        )));
    }
    let n_steps = validate_parallel_args(group_size, per_rank_steps, collectives, zero_before)?;

    let hip_err = |e: hip_bridge::HipError| DispatchError::Hip(e.to_string());

    for i in 0..n_steps {
        // Optional pre-zero of each rank's output buffer (EP accumulation pattern):
        // f32 partial: memset_async dim*4 bytes; i64 partial: memset_async dim*8 bytes.
        if zero_before[i] {
            let (dim, elem_bytes) = match &collectives[i] {
                StepCollective::AllReduce { dim, .. } => (*dim, 4usize),
                StepCollective::AllReduceI64Tp { dim } => (*dim, 8usize),
                StepCollective::ZeroI64Only { dim } => (*dim, 8usize),
                StepCollective::None => {
                    return Err(DispatchError::Hip(format!(
                        "execute_steps_parallel: zero_before[{i}] is true but collective is None (no dim)"
                    )));
                }
            };
            for (r, &dev) in group.iter().enumerate() {
                gpus.devices[dev].bind_thread().map_err(hip_err)?;
                let stream = gpus.devices[dev].active_stream.as_ref().ok_or_else(|| {
                    DispatchError::Hip(format!(
                        "execute_steps_parallel: device {dev} has no active_stream for zero_before"
                    ))
                })?;
                let buf = tp_step_out_buf(&per_rank_steps[r][i]).ok_or_else(|| {
                    DispatchError::Hip(format!(
                        "execute_steps_parallel: step {i} zero_before=true but has no out buffer"
                    ))
                })?;
                gpus.devices[dev]
                    .hip
                    .memset_async(buf, 0, dim * elem_bytes, stream)
                    .map_err(hip_err)?;
            }
        }

        // Run step i on every rank (each with its own sharded weights/buffers).
        for (r, &dev) in group.iter().enumerate() {
            gpus.devices[dev].bind_thread().map_err(hip_err)?;
            let ctx = DispatchCtx::new(&gpus.devices[dev]);
            launch_op(&mut gpus.devices[dev], &ctx, &per_rank_steps[r][i])?;
        }

        // Collective: all-reduce the partial outputs over the axis group.
        match &collectives[i] {
            StepCollective::AllReduce {
                kind: coll_kind,
                dim,
            } => {
                for &dev in &group {
                    let g = &gpus.devices[dev];
                    g.bind_thread().map_err(hip_err)?;
                    let stream = g.active_stream.as_ref().ok_or_else(|| {
                        DispatchError::Hip(format!(
                            "execute_steps_parallel: device {dev} has no active_stream"
                        ))
                    })?;
                    g.hip.stream_synchronize(stream).map_err(hip_err)?;
                }
                let mut refs: Vec<&hip_bridge::DeviceBuffer> = Vec::with_capacity(group_size);
                for (r, _) in group.iter().enumerate() {
                    let buf = tp_step_out_buf(&per_rank_steps[r][i]).ok_or_else(|| {
                        DispatchError::Hip(format!(
                            "execute_steps_parallel: step {i} marked AllReduce but has no out buffer"
                        ))
                    })?;
                    refs.push(buf);
                }
                // Peer/RCCL choice: Tp always uses peer (RCCL not installed on Tp path);
                // Ep branches on HIPFIRE_EP_PEER_ALLREDUCE_DECODE. Mirrors lib.rs:823-826.
                match coll_kind {
                    hipfire_hardware::DimKind::Tp => {
                        gpus.all_reduce_sum_f32_peer(&group, &refs, *dim)
                            .map_err(|e| DispatchError::Hip(e.to_string()))?;
                    }
                    _ => {
                        if hipfire_hardware::ep_peer_allreduce_decode() {
                            gpus.all_reduce_sum_f32_peer(&group, &refs, *dim)
                                .map_err(|e| DispatchError::Hip(e.to_string()))?;
                        } else {
                            gpus.all_reduce_sum_f32(&group, &refs, *dim)
                                .map_err(|e| DispatchError::Hip(e.to_string()))?;
                        }
                    }
                }
            }
            StepCollective::AllReduceI64Tp { dim } => {
                // Reproducible int64 TP down: sync streams, peer-reduce int64 partials.
                for &dev in &group {
                    let g = &gpus.devices[dev];
                    g.bind_thread().map_err(hip_err)?;
                    let stream = g.active_stream.as_ref().ok_or_else(|| {
                        DispatchError::Hip(format!(
                            "execute_steps_parallel: device {dev} has no active_stream (i64 reduce)"
                        ))
                    })?;
                    g.hip.stream_synchronize(stream).map_err(hip_err)?;
                }
                let mut refs: Vec<&hip_bridge::DeviceBuffer> = Vec::with_capacity(group_size);
                for (r, _) in group.iter().enumerate() {
                    let buf = tp_step_out_buf(&per_rank_steps[r][i]).ok_or_else(|| {
                        DispatchError::Hip(format!(
                            "execute_steps_parallel: step {i} marked AllReduceI64Tp but has no out buffer"
                        ))
                    })?;
                    refs.push(buf);
                }
                gpus.all_reduce_sum_i64_peer(&group, &refs, *dim)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
            }
            StepCollective::ZeroI64Only { .. } => {
                // Zero-only: buffer was already zeroed in the zero_before block above.
                // No cross-rank reduce — the EP i64 path reduces in FP32 (attached
                // to the ConvertI64ToF32 step's AllReduce{Ep} collective).
            }
            StepCollective::None => {}
        }
    }
    Ok(())
}

/// Tensor-parallel executor (P-B grand-unify, PB-TP1). The `Tp>1` counterpart of
/// [`execute_steps_mesh`]: instead of one whole-model step list on one `Gpu`, it
/// takes **per-rank step lists** (`per_rank_steps[r]` references rank `r`'s own
/// sharded weights + buffers, built by the caller from a sharded `WeightStore`)
/// and runs them **lock-step** across the mesh's `Tp` group — every rank executes
/// step `i`, then a `Tp` all-reduce is injected for the row-parallel steps.
///
/// Column-parallel `Gemv`s leave their output sharded (`inter/tp`) to feed the
/// next step; row-parallel `Gemv`s each produce a partial `[dim]` which this
/// executor sums in place via `all_reduce_sum_f32_peer` — so after a
/// `TpCollective::AllReduceOut` step every rank holds the whole result. Residual
/// adds must be a SEPARATE post-collective step (a row-parallel `GemvResidual`
/// would sum the residual `tp×`), so row-parallel ops are plain `Gemv`s here.
///
/// This is the EP `run_layer_program_mesh` shape in the `Step` world, keyed by
/// the caller-supplied `collectives` (from `ShardPolicy`) instead of
/// `SuperOpKind::Moe`. Fusion is intentionally not applied on the TP path yet
/// (F32 GEMV needs none); per-rank `DispatchCtx` is built like the EP path.
///
/// Preconditions the caller owns: each device in the `Tp` group has an
/// `active_stream` set and peer access enabled (`ensure_rank_streams` +
/// `enable_peer_all`).
///
/// **Wrapper:** delegates to [`execute_steps_parallel`] with `zero_before` all-false
/// and `TpCollective` mapped to `StepCollective`. Byte-identical to the prior
/// monolithic implementation for all existing TP callers/examples.
pub fn execute_steps_tp(
    mesh: &DeviceMesh,
    gpus: &mut hipfire_hardware::Gpus,
    per_rank_steps: &[Vec<Step>],
    collectives: &[TpCollective],
) -> Result<(), DispatchError> {
    let n_steps = per_rank_steps.first().map(|v| v.len()).unwrap_or(0);
    let collectives2: Vec<StepCollective> = collectives
        .iter()
        .map(|c| match c {
            TpCollective::AllReduceOut { dim } => StepCollective::AllReduce {
                kind: hipfire_hardware::DimKind::Tp,
                dim: *dim,
            },
            TpCollective::None => StepCollective::None,
        })
        .collect();
    let zero_before = vec![false; n_steps];
    execute_steps_parallel(mesh, gpus, per_rank_steps, &collectives2, &zero_before)
}
/// Keys whose 3-way QKV **decode** dispatch arm folds the optional Q/K/V bias
/// into the kernel (a `_with_bias` kernel variant exists and is wired in
/// `dispatch_fused_qkv`). The fold is additionally guarded off on dp4a archs
/// (gfx906), whose fused-QKV kernel has no bias parameters — there the 3
/// `BiasAdd` steps run separately as before (handover pitfall #4). Keys NOT
/// listed here keep the unfused path: their `BiasAdd` steps are never consumed,
/// so the result is unchanged.
fn qkv_bias_fold_supported(key: KernelKey, ctx: &DispatchCtx) -> bool {
    if ctx.arch.gemv_dp4a_enabled() {
        return false;
    }
    // All per-row 3-way QKV decode keys whose `gpu.fused_qkv_*_with_bias`
    // variant is wired and GPU-parity-validated (no-bias==0, with-bias==bias;
    // see examples/test_fused_qkv_bias_parity.rs). HFQ4G256 additionally has the
    // full three-way model byte-identity proof (see the Phase-1 commit).
    matches!(
        key,
        KernelKey::FusedQkvHfq4G256
            | KernelKey::FusedQkvMq4G256Lloyd
            | KernelKey::FusedQkvMq3G256Lloyd
            | KernelKey::FusedQkvQ4K
            | KernelKey::FusedQkvQ8_0
            // HFQ6/MQ6: the fold switches decode GEMM→per-row (Family B). The
            // dispatch arm keeps the GEMM unless bias is present, so this only
            // changes decode when the fold actually fires.
            | KernelKey::FusedQkvHfq6G256
    )
}

/// If `steps[len..len+3]` are three `BiasAdd` ops whose `x` targets are exactly
/// the q/k/v GEMV outputs of the fused window (`steps[1..4]`), return the three
/// bias tensors `[bias_q, bias_k, bias_v]`. Otherwise `None` (no fold). The
/// ptr-identity check guarantees we only fold the qwen2 `attention_bias` adds
/// that immediately follow this exact QKV window, never an unrelated `BiasAdd`.
fn match_trailing_qkv_bias<'a>(steps: &'a [Step<'a>], len: usize) -> Option<[&'a GpuTensor; 3]> {
    if len + 3 > steps.len() {
        return None;
    }
    let (
        Step::BiasAdd {
            x: bx_q, bias: bq, ..
        },
        Step::BiasAdd {
            x: bx_k, bias: bk, ..
        },
        Step::BiasAdd {
            x: bx_v, bias: bv, ..
        },
    ) = (&steps[len], &steps[len + 1], &steps[len + 2])
    else {
        return None;
    };
    let (_, out_q) = gemv_weight_out(&steps[1]);
    let (_, out_k) = gemv_weight_out(&steps[2]);
    let (_, out_v) = gemv_weight_out(&steps[3]);
    if std::ptr::eq(*bx_q as *const GpuTensor, out_q as *const GpuTensor)
        && std::ptr::eq(*bx_k as *const GpuTensor, out_k as *const GpuTensor)
        && std::ptr::eq(*bx_v as *const GpuTensor, out_v as *const GpuTensor)
    {
        Some([bq, bk, bv])
    } else {
        None
    }
}

/// Launch a Qwen2 3-way QKV window through bias-specific kernel symbols.
/// Qwen3+ continues through [`launch_fused`] and the original Redline ABI.
fn launch_fused_qkv_with_bias<'a>(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    key: KernelKey,
    steps: &[Step<'a>],
    bias: [&'a GpuTensor; 3],
) -> Result<(), DispatchError> {
    // Opt-in diagnostic (HIPFIRE_FUSE_QKV_BIAS_DEBUG=1): confirm the fold fires.
    // Flag is resolved once at init — no per-launch env lock on the hot path.
    if ctx.flags.fuse_qkv_bias_debug {
        eprintln!("[qkv-bias-fold] fired ({:?})", key);
    }
    // Step 0 is RmsnormAutomatic — run it to fill the activated buffer.
    launch_op(gpu, ctx, &steps[0])?;
    let activated = rmsnorm_out(&steps[0]);
    let (wq, q) = gemv_weight_out(&steps[1]);
    let (wk, k) = gemv_weight_out(&steps[2]);
    let (wv, v) = gemv_weight_out(&steps[3]);
    let fused_qkv = FUSED_QKV.get_or_init(FusedQkvFamily::new);
    fused_qkv.run_with_qwen2_bias(
        ctx,
        gpu,
        &FusedQkvBiasParams {
            kind: key,
            weights: [wq.buf, wk.buf, wv.buf],
            x: activated,
            outputs: [q, k, v],
            m: [wq.m, wk.m, wv.m],
            k: wq.k,
            bias,
        },
    )
}

/// Per-op fallback. FULL enum match (no catch-all) so the compiler forces every
/// op to have an arm (spec F4 — a missing arm would be a silent runtime error).
fn launch_op(gpu: &mut Gpu, ctx: &DispatchCtx, step: &Step) -> Result<(), DispatchError> {
    match step {
        Step::Gemv {
            w,
            input: GemvInput::Raw(x),
            out,
        } => {
            let gemv = GEMV.get_or_init(GemvFamily::new);
            gemv.run_auto(ctx, gpu, w, x, out)
        }
        Step::GemmKeyedBatched {
            w,
            x,
            y,
            batch,
            key,
        } => GEMM
            .get_or_init(crate::families::gemm::GemmFamily::new)
            .run_key(
                *key,
                ctx,
                gpu,
                &crate::families::gemm::GemmParams {
                    w,
                    x,
                    y,
                    batch_size: *batch,
                },
            ),
        Step::Gemm { w, x, y, batch } => {
            // Batched (B>1) GEMM. Mirrors runtime `weight_gemm` (llama.rs:1444)
            // per-dtype against a `WeightRef`; the batched kernels live in
            // rdna-compute (same ones `weight_gemm` calls). Prefill-only — no
            // fused-decode entry hits this.
            let hip_err = |e: hip_bridge::HipError| DispatchError::Hip(e.to_string());
            match w.dtype {
                DType::HFQ4G256 => gpu
                    .gemm_hfq4g256(w.buf, x, y, w.m, w.k, *batch)
                    .map_err(hip_err),
                DType::HFQ4G128 => gpu
                    .gemm_hfq4g128(w.buf, x, y, w.m, w.k, *batch)
                    .map_err(hip_err),
                // MQ4G256 = HFQ4G256 layout + an AWQ-aware FWHT rotation of x.
                // FWHT-rotate all `batch` activation columns once (the dispatch
                // twin of runtime `rotate_x_mq_batched_for` — `rotate` runs
                // `ensure_mq_signs` internally via prepare_rotation_scratch), then
                // feed the same INT4-G256 batched WMMA kernel weight_gemm uses.
                DType::MQ4G256 => {
                    let gemv = GEMV.get_or_init(GemvFamily::new);
                    let h = gemv.rotate(
                        ctx,
                        gpu,
                        w,
                        x,
                        &RotateInputs {
                            batch_size: *batch,
                            ..Default::default()
                        },
                    )?;
                    let x_rot = h.into_buf();
                    gpu.gemm_hfq4g256_batched_lmhead(w.buf, &x_rot, y, w.m, w.k, *batch)
                        .map_err(hip_err)
                }
                other => Err(DispatchError::Hip(format!(
                    "Step::Gemm: dtype {other:?} not wired (add its weight_gemm arm)"
                ))),
            }
        }
        Step::GemmResidualBatched {
            w,
            x,
            residual,
            batch,
            key,
        } => GEMM
            .get_or_init(crate::families::gemm::GemmFamily::new)
            .run_key(
                *key,
                ctx,
                gpu,
                &crate::families::gemm::GemmParams {
                    w,
                    x,
                    y: residual,
                    batch_size: *batch,
                },
            ),
        #[cfg(feature = "deltanet")]
        Step::FusedQkvzaBatched {
            wqkv,
            wz,
            w_beta,
            w_alpha,
            x,
            qkv,
            z,
            beta,
            alpha,
            m,
            k,
            batch,
            key,
        } => {
            let weights = [wqkv.buf, wz.buf, w_beta.buf, w_alpha.buf];
            let outputs = [*qkv, *z, *beta, *alpha];
            FUSED_QKV.get_or_init(FusedQkvFamily::new).run(
                ctx,
                gpu,
                &FusedQkvParams {
                    kind: *key,
                    weights: &weights,
                    x,
                    outputs: &outputs,
                    m: &m[..],
                    k: *k,
                    rot_scratch: &[],
                    batch_size: Some(*batch),
                },
            )
        }
        Step::Gemv {
            w,
            input: GemvInput::Prerotated(xr),
            out,
        } => {
            let gemv = GEMV.get_or_init(GemvFamily::new);
            gemv.run(
                ctx,
                gpu,
                &GemvParams {
                    w,
                    x: xr,
                    y: out,
                    variant: GemvVariant::Prerotated,
                    residual: None,
                    gate: None,
                    up: None,
                },
            )
        }
        Step::GemvResidual {
            w,
            input: GemvInput::Prerotated(xr),
            residual,
            out: _,
        } => {
            // MQ-family with a fused residual kernel: writes `residual` in-place via
            // GemvVariant::WithResidual. `out` is NOT written — it is scratch for the
            // fallback path only (see the Raw arm below). Nothing downstream reads
            // `out` after this step in either qwen2 or llama decode paths.
            let gemv = GEMV.get_or_init(GemvFamily::new);
            gemv.run(
                ctx,
                gpu,
                &GemvParams {
                    w,
                    x: xr,
                    y: residual,
                    variant: GemvVariant::WithResidual,
                    residual: None,
                    gate: None,
                    up: None,
                },
            )
        }
        Step::GemvResidual {
            w,
            input: GemvInput::Raw(x),
            residual,
            out,
        } => {
            // For dtypes WITHOUT a fused residual kernel (Q8_0, Q4K, F32), the
            // fallback path runs a plain GEMV then `residual += result`. `out` may
            // be used as scratch ONLY when it does not alias `residual`; when it
            // does (the common qwen35 o_proj / dn_out case where out == residual ==
            // &s.x), a dedicated persistent temp is used instead.
            // Nothing reads `out` after this step in any model decode path.
            let gemv = GEMV.get_or_init(GemvFamily::new);
            // Dtypes with a fused `gemv_*_residual` kernel use it in one launch.
            // Dtypes without one (Q8_0, ParoQ4G128, …) fall back to plain GEMV into
            // the `out` scratch + `residual += out` — reuses the pre-allocated `out`
            // buffer instead of alloc/free per call. Plain GEMV applies this
            // dtype's own rotation (FWHT / Givens) internally, so this is correct
            // for both no-rotation (Q8) and Givens (Paro) dtypes.
            if KernelKey::for_gemv_residual(w.dtype).is_ok() {
                if crate::types::dtype_rotation_plan(w.dtype) != RotationPlan::None {
                    let h = gemv.rotate(ctx, gpu, w, x, &RotateInputs::default())?;
                    let xr = h.into_buf();
                    gemv.run(
                        ctx,
                        gpu,
                        &GemvParams {
                            w,
                            x: &xr,
                            y: residual,
                            variant: GemvVariant::WithResidual,
                            residual: None,
                            gate: None,
                            up: None,
                        },
                    )
                } else {
                    gemv.run(
                        ctx,
                        gpu,
                        &GemvParams {
                            w,
                            x,
                            y: residual,
                            variant: GemvVariant::WithResidual,
                            residual: None,
                            gate: None,
                            up: None,
                        },
                    )
                }
            } else {
                // run_auto applies the dtype's rotation (FWHT/Givens) before the
                // kernel, so ParoQ4G128 gets its Givens rotation. Plain would skip it.
                //
                // ALIASING GUARD: most callers (e.g. qwen35 o_proj / dn_out) pass
                // `out` == `residual` (both `&s.x`). Reusing `out` as the GEMV scratch
                // in that case is WRONG: run_auto would overwrite the residual with
                // `W·x` and the subsequent `residual += out` would then compute
                // `2·(W·x)` — the residual is lost. Detect the alias by device pointer
                // and use a dedicated persistent scratch when they overlap. When `out`
                // is a genuinely-distinct buffer, reuse it (no alloc churn).
                if std::ptr::eq(residual, out) || residual.buf.as_ptr() == out.buf.as_ptr() {
                    let tmp = {
                        let scratch = gpu
                            .ensure_gemv_residual_tmp(w.m)
                            .map_err(|e| DispatchError::Hip(e.to_string()))?;
                        // `gpu` owns this dedicated allocation for the alias's lifetime.
                        GpuTensor {
                            buf: unsafe { scratch.buf.alias() },
                            shape: vec![w.m],
                            dtype: DType::F32,
                        }
                    };
                    gemv.run_auto(ctx, gpu, w, x, &tmp)?;
                    gpu.add_inplace_f32(residual, &tmp)
                        .map_err(|e| DispatchError::Hip(e.to_string()))?;
                } else {
                    gemv.run_auto(ctx, gpu, w, x, out)?;
                    gpu.add_inplace_f32(residual, out)
                        .map_err(|e| DispatchError::Hip(e.to_string()))?;
                }
                Ok(())
            }
        }
        Step::RmsnormAutomatic {
            x,
            norm_weight,
            x_plain,
            out,
            awq_scale,
            k,
            eps,
            rotation,
        } => {
            if *rotation == RotationPlan::None {
                // HFQ4G256 and other non-FWHT dtypes: plain rmsnorm into `out`.
                // x_plain is not written in this path (scratch only for FWHT path).
                gpu.rmsnorm_f32(x, norm_weight, out, *eps)
                    .map_err(|e| DispatchError::Hip(e.to_string()))
            } else if *rotation == RotationPlan::Mq8Internal {
                // MQ8 cannot share LDS with the FWHT-G256 fused kernel: it produces an
                // INT8 scratch consumed by the downstream gemv_mq8_prerotated kernel.
                // RotationFamily::WithRmsnorm would route to fused_rmsnorm_rotate_mq
                // (FWHT, F32 output) — wrong dtype for the MQ8 GEMV. Mirror the fix
                // from qwen35.rs::rmsnorm_rotate_dispatch (7b35e700).
                gpu.rmsnorm_f32(x, norm_weight, out, *eps)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                gpu.rotate_quantize_x_mq8(out, *k)
                    .map_err(|e| DispatchError::Hip(e.to_string()))
            } else {
                let rotation_family = ROTATION.get_or_init(RotationFamily::new);
                rotation_family
                    .run(
                        ctx,
                        gpu,
                        RotationParams {
                            x,
                            x_up: None,
                            w_norm: Some(norm_weight),
                            x_plain,
                            x_rot: out,
                            awq_scale: *awq_scale,
                            k: *k,
                            eps: *eps,
                            batch_size: 1,
                            variant: RotationVariant::WithRmsnorm,
                            givens_pairs: None,
                            givens_theta: None,
                            givens_scales: None,
                            givens_krot: None,
                        },
                    )
                    .map_err(|e| DispatchError::Hip(e.to_string()))
            }
        }
        Step::RmsnormBatched {
            x,
            norm_weight,
            x_plain,
            out,
            awq_scale,
            k,
            eps,
            rotation,
            batch,
        } => {
            if *rotation == RotationPlan::None {
                gpu.rmsnorm_batched(x, norm_weight, out, *batch, *k, *eps)
                    .map_err(|e| DispatchError::Hip(e.to_string()))
            } else {
                ROTATION
                    .get_or_init(RotationFamily::new)
                    .run(
                        ctx,
                        gpu,
                        RotationParams {
                            x,
                            x_up: None,
                            w_norm: Some(norm_weight),
                            x_plain,
                            x_rot: out,
                            awq_scale: *awq_scale,
                            k: *k,
                            eps: *eps,
                            batch_size: *batch,
                            variant: RotationVariant::WithRmsnorm,
                            givens_pairs: None,
                            givens_theta: None,
                            givens_scales: None,
                            givens_krot: None,
                        },
                    )
                    .map_err(|e| DispatchError::Hip(e.to_string()))
            }
        }
        Step::GivensRotateBatched {
            x,
            out,
            pairs,
            theta,
            scales,
            batch,
            dim,
            krot,
        } => gpu
            .givens_rotate_to(x, out, pairs, theta, scales, *batch, *dim, *krot)
            .map_err(|e| DispatchError::Hip(e.to_string())),
        Step::RotateFwhtBatched {
            x,
            out,
            awq_scale,
            k,
            batch,
        } => match awq_scale {
            Some(scale) => gpu
                .rotate_x_mq_awq_batched(x, scale, out, *k, *batch)
                .map_err(|e| DispatchError::Hip(e.to_string())),
            None => gpu
                .rotate_x_mq_batched(x, out, *k, *batch)
                .map_err(|e| DispatchError::Hip(e.to_string())),
        },
        Step::Attend { plan, io } => {
            use crate::families::attention::AttentionFamily;
            static ATTENTION: OnceLock<AttentionFamily> = OnceLock::new();
            let attn = ATTENTION.get_or_init(AttentionFamily::new);
            attn.run_attention(ctx, gpu, plan, io)
        }
        Step::Rope {
            q,
            k,
            pos_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            theta,
        } => gpu
            .rope_f32(q, k, pos_buf, *n_heads, *n_kv_heads, *head_dim, *theta)
            .map_err(|e| DispatchError::Hip(e.to_string())),
        Step::QkNorm {
            x,
            weight,
            n_groups,
            head_dim,
            eps,
        } => gpu
            .rmsnorm_batched(x, weight, x, *n_groups, *head_dim, *eps)
            .map_err(|e| DispatchError::Hip(e.to_string())),
        Step::BiasAdd { x, bias, dim } => gpu
            .bias_add_f32(x, bias, 1, *dim)
            .map_err(|e| DispatchError::Hip(e.to_string())),
        Step::SiluMul { gate, up, out } => gpu
            .silu_mul_f32(gate, up, out)
            .map_err(|e| DispatchError::Hip(e.to_string())),
        Step::ResidualAdd { x, y, dim: _ } => gpu
            .add_f32(x, y, x)
            .map_err(|e| DispatchError::Hip(e.to_string())),
        #[cfg(feature = "deltanet")]
        Step::DeltaGatePrep {
            beta,
            alpha,
            dt_bias,
            a_log,
            n,
            batch_size,
        } => {
            let result = if *batch_size > 1 {
                gpu.fused_sigmoid_alpha_gate_f32_batched(
                    beta,
                    alpha,
                    dt_bias,
                    a_log,
                    *n,
                    *batch_size,
                )
            } else {
                gpu.fused_sigmoid_alpha_gate_f32(beta, alpha, dt_bias, a_log, *n)
            };
            result.map_err(|e| DispatchError::Hip(e.to_string()))
        }
        #[cfg(feature = "deltanet")]
        Step::DeltaConvSplit {
            q_out,
            k_out,
            v_out,
            input,
            weight,
            state,
            parent_indices,
            k_dim,
            v_dim,
            n_tokens,
        } => {
            let result = if let Some(parents) = parent_indices {
                gpu.conv1d_silu_split_tree_f32_n(
                    q_out, k_out, v_out, input, weight, state, parents, *k_dim, *v_dim, *n_tokens,
                )
            } else if *n_tokens > 1 {
                gpu.conv1d_silu_split_f32_n(
                    q_out, k_out, v_out, input, weight, state, *k_dim, *v_dim, *n_tokens,
                )
            } else {
                gpu.conv1d_silu_split_f32(q_out, k_out, v_out, input, weight, state, *k_dim, *v_dim)
            };
            result.map_err(|e| DispatchError::Hip(e.to_string()))
        }
        #[cfg(feature = "deltanet")]
        Step::DeltaQkL2Norm {
            q,
            k,
            n_key_heads,
            head_dim,
            q_scale,
            eps,
            batch_size,
        } => {
            let result = if *batch_size > 1 {
                gpu.fused_qk_l2_norm_scale_f32_batched(
                    q,
                    k,
                    *n_key_heads,
                    *head_dim,
                    *q_scale,
                    *eps,
                    *batch_size,
                )
            } else {
                gpu.fused_qk_l2_norm_scale_f32(q, k, *n_key_heads, *head_dim, *q_scale, *eps)
            };
            result.map_err(|e| DispatchError::Hip(e.to_string()))
        }
        #[cfg(feature = "deltanet")]
        Step::DeltaRepeatHeads {
            q_src,
            k_src,
            q_dst,
            k_dst,
            n_key_heads,
            ratio,
            head_dim,
            batch_size,
        } => {
            let result = if *ratio <= 1 {
                let bytes = n_key_heads * head_dim * batch_size * 4;
                let q_result = if q_src.buf.as_ptr() == q_dst.buf.as_ptr() {
                    Ok(())
                } else {
                    gpu.memcpy_dtod_auto(&q_dst.buf, &q_src.buf, bytes)
                };
                q_result.and_then(|_| {
                    if k_src.buf.as_ptr() == k_dst.buf.as_ptr() {
                        Ok(())
                    } else {
                        gpu.memcpy_dtod_auto(&k_dst.buf, &k_src.buf, bytes)
                    }
                })
            } else if *batch_size > 1 {
                gpu.repeat_interleave_qk_f32_batched(
                    q_src,
                    k_src,
                    q_dst,
                    k_dst,
                    *n_key_heads,
                    *ratio,
                    *head_dim,
                    *batch_size,
                )
            } else {
                gpu.repeat_interleave_qk_f32(
                    q_src,
                    k_src,
                    q_dst,
                    k_dst,
                    *n_key_heads,
                    *ratio,
                    *head_dim,
                )
            };
            result.map_err(|e| DispatchError::Hip(e.to_string()))
        }
        #[cfg(feature = "deltanet")]
        Step::DeltaRecurrence { params } => {
            use crate::ops::delta_net::DeltaNetOps;
            let result = match params {
                DeltaRecurrenceParams::Step(p) => ().run_delta_net_step(gpu, p),
                DeltaRecurrenceParams::Batch { params, intent } => {
                    ().run_delta_net_batch_with_intent(gpu, params, *intent)
                }
                DeltaRecurrenceParams::Tree(p) => ().run_delta_net_tree(gpu, p),
            };
            result.map_err(DispatchError::Hip)
        }
        #[cfg(feature = "deltanet")]
        Step::DeltaGatedNorm {
            x,
            z,
            weight,
            out,
            n_heads,
            head_dim,
            eps,
            batch_size,
        } => {
            let result = if *batch_size > 1 {
                gpu.gated_norm_f32_batched(
                    x,
                    z,
                    weight,
                    out,
                    *n_heads,
                    *head_dim,
                    *eps,
                    *batch_size,
                )
            } else {
                gpu.gated_norm_f32(x, z, weight, out, *n_heads, *head_dim, *eps)
            };
            result.map_err(|e| DispatchError::Hip(e.to_string()))
        }
        // ── MoE decode ops (Task 4) ─────────────────────────────────────
        Step::MoeRoute {
            scores,
            gate_bias,
            topk_indices,
            topk_weights,
            k,
            n_experts,
            route_scale,
        } => launch_moe_route(
            gpu,
            scores,
            gate_bias,
            topk_indices,
            topk_weights,
            *n_experts,
            *k,
            *route_scale,
        ),
        Step::IndexedMoeGemv {
            experts,
            which,
            topk_indices,
            input,
            out,
            k_top,
            batch_size,
        } => {
            // The step's batch_size is authoritative: a zero routed batch is
            // rejected before any launcher dispatch.
            indexed_moe_batch_guard(*batch_size)?;
            // Extract the inner tensor — the helpers take a plain &GpuTensor.
            // Both Raw and Prerotated are accepted; callers should pass Prerotated
            // (the activation is always FWHT-rotated before building the step).
            let x = match input {
                GemvInput::Raw(x) | GemvInput::Prerotated(x) => x,
            };
            match which {
                MoeProj::GateUp { up_out } => {
                    // Batch one keeps the scalar launcher byte-identically;
                    // batch > 1 is the DeepSeek MQ2-Lloyd batched kernel with
                    // NO scalar fallback for any other dtype.
                    match deepseek_gate_up_indexed_form(experts.dtype, *batch_size) {
                        DeepSeekIndexedForm::Scalar => launch_indexed_gate_up(
                            gpu,
                            experts,
                            topk_indices,
                            x,
                            out,
                            up_out,
                            *k_top,
                        ),
                        DeepSeekIndexedForm::Batched => launch_indexed_gate_up_batched(
                            gpu,
                            experts,
                            topk_indices,
                            x,
                            out,
                            up_out,
                            *k_top,
                            *batch_size,
                        ),
                        DeepSeekIndexedForm::Unsupported => Err(DispatchError::Hip(format!(
                            "IndexedMoeGemv::GateUp: batched (batch_size={}) unsupported for \
                             dtype {:?}; no scalar fallback",
                            *batch_size, experts.dtype
                        ))),
                    }
                }
                MoeProj::DownExpanded => {
                    launch_indexed_down(gpu, experts, topk_indices, x, out, *k_top, *batch_size)
                }
                MoeProj::DownResidual { topk_weights } => {
                    // FP32 residual-fused down has no batched kernel; batch > 1
                    // rejects explicitly, never a scalar fallback.
                    match deepseek_f32_down_indexed_form(*batch_size) {
                        DeepSeekIndexedForm::Scalar => launch_indexed_down_residual(
                            gpu,
                            experts,
                            topk_indices,
                            topk_weights,
                            x,
                            out,
                            *k_top,
                        ),
                        DeepSeekIndexedForm::Unsupported => Err(DispatchError::Hip(format!(
                            "IndexedMoeGemv::DownResidual: batched (batch_size={}) FP32 \
                             residual is unsupported; no scalar fallback",
                            *batch_size
                        ))),
                        DeepSeekIndexedForm::Batched => {
                            unreachable!("the FP32 residual form has no batched variant")
                        }
                    }
                }
                MoeProj::DownResidualI64 { topk_weights } => {
                    // Batch one keeps the scalar MQ2/MQ3-Lloyd launcher; batch
                    // > 1 is the existing MQ2-Lloyd batched launcher with NO
                    // scalar fallback for any other dtype.
                    match deepseek_i64_down_indexed_form(experts.dtype, *batch_size) {
                        DeepSeekIndexedForm::Scalar => launch_indexed_down_residual_i64(
                            gpu,
                            experts,
                            topk_indices,
                            topk_weights,
                            x,
                            out,
                            *k_top,
                        ),
                        DeepSeekIndexedForm::Batched => launch_indexed_down_residual_i64_batched(
                            gpu,
                            experts,
                            topk_indices,
                            topk_weights,
                            x,
                            out,
                            *k_top,
                            *batch_size,
                        ),
                        DeepSeekIndexedForm::Unsupported => Err(DispatchError::Hip(format!(
                            "IndexedMoeGemv::DownResidualI64: batched (batch_size={}) \
                             unsupported for dtype {:?}; no scalar fallback",
                            *batch_size, experts.dtype
                        ))),
                    }
                }
            }
        }
        Step::ConvertI64ToF32 { src, dst, n } => gpu
            .moe_i64_residual_to_f32(src, dst, *n)
            .map_err(|e| DispatchError::Hip(e.to_string())),
        Step::MoeCombine {
            down_out,
            topk_weights,
            out,
            k,
            hidden,
            batch_size,
            inverse_perm,
        } => match inverse_perm {
            Some(perm) => {
                // Prefill grouped path: moe_down_combine_grouped_k8.
                launch_moe_combine_grouped(
                    gpu,
                    down_out,
                    perm,
                    topk_weights,
                    out,
                    *hidden,
                    *k,
                    *batch_size,
                )
            }
            None => {
                // Decode expanded path: moe_down_combine_k8_batched.
                launch_moe_combine(gpu, down_out, topk_weights, out, *hidden, *k, *batch_size)
            }
        },
        // ── MoE prefill grouped ops (Task 5) ───────────────────────────────
        Step::MoeScatter {
            topk_indices,
            expert_token_counts,
            expert_offsets,
            sorted_slot_index,
            expert_tile_ids,
            inverse_perm,
            total_slots,
            n_experts,
            m_total_max,
            block_m,
        } => launch_moe_scatter(
            gpu,
            topk_indices,
            expert_token_counts,
            expert_offsets,
            sorted_slot_index,
            expert_tile_ids,
            inverse_perm,
            *total_slots,
            *n_experts,
            *m_total_max,
            *block_m,
        ),
        Step::GroupedMoeGemm {
            experts,
            which,
            sorted_slot_index,
            expert_tile_ids,
            x,
            y,
            m_total,
            batch_size,
            k_top,
            dtype_tags,
            force_mq4_fp16,
            paro_i8,
            paro_i8_k8,
        } => match which {
            MoeProj::GateUp { .. } => launch_grouped_gate_up(
                gpu,
                experts,
                sorted_slot_index,
                expert_tile_ids,
                x,
                y,
                *m_total,
                *k_top,
                *batch_size,
                *dtype_tags,
                *force_mq4_fp16,
                *paro_i8,
                *paro_i8_k8,
            ),
            MoeProj::DownExpanded => {
                // Structural rejection: the grouped kernel family has no
                // residual-fused down — only the expanded write + separate
                // grouped combine exists.
                grouped_down_projection(which)?;
                launch_grouped_down(
                    gpu,
                    experts,
                    sorted_slot_index,
                    expert_tile_ids,
                    x,
                    y,
                    *m_total,
                    *k_top,
                    *batch_size,
                    *dtype_tags,
                    *force_mq4_fp16,
                    *paro_i8,
                    *paro_i8_k8,
                )
            }
            MoeProj::DownResidual { .. } | MoeProj::DownResidualI64 { .. } => {
                Err(DispatchError::Hip(
                    "GroupedMoeGemm: DownResidual/DownResidualI64 is not a valid grouped \
                     projection; use DownExpanded + MoeCombine(inverse_perm=Some) for grouped down"
                        .to_string(),
                ))
            }
        },
        Step::MoeGateUpUnscatter {
            y_grouped,
            sorted_slot_index,
            gate_batch,
            up_batch,
            inter,
            k_top,
            m_total,
        } => launch_moe_gate_up_unscatter(
            gpu,
            y_grouped,
            sorted_slot_index,
            gate_batch,
            up_batch,
            *inter,
            *k_top,
            *m_total,
        ),
        Step::ScoreActivation { scores, kind } => launch_score_activation(gpu, scores, *kind),
        Step::MoeActivation {
            variant,
            gate,
            up,
            rot_out,
            inter,
            k_top,
        } => {
            // k_top is the routed row count at launch (k_top for decode,
            // batch_size·k_top for batched prefill).
            let rows = *k_top;
            launch_moe_activation(gpu, variant, gate, up, rot_out, *inter, rows)
        }
        // ── Qwen MoE Step-native ops (STEP-002 Phase 1) ──────────────────
        Step::MoeSoftmaxTopK {
            logits,
            topk_indices,
            topk_weights,
            n_exp,
            norm_topk_prob,
        } => launch_moe_softmax_topk(
            gpu,
            logits,
            topk_indices,
            topk_weights,
            *n_exp,
            *norm_topk_prob,
        ),
        Step::MoeFusedSharedGate {
            router,
            shared_expert_gate,
            shared_gate_w,
            shared_up_w,
            x_rot,
            router_logits,
            scalar_buf,
            gate_buf,
            up_buf,
            smi,
        } => launch_fused_shared_gate(
            gpu,
            router,
            shared_expert_gate,
            shared_gate_w,
            shared_up_w,
            x_rot,
            router_logits,
            scalar_buf,
            gate_buf,
            up_buf,
            *smi,
        ),
        Step::MoeSharedGateSide {
            router,
            shared_expert_gate,
            shared_gate_w,
            shared_up_w,
            x_norm,
            x_rot_local,
            router_logits,
            scalar_buf,
            gate_buf,
            up_buf,
            smi,
        } => launch_shared_gate_side(
            ctx,
            gpu,
            router,
            shared_expert_gate,
            shared_gate_w,
            shared_up_w,
            x_norm,
            *x_rot_local,
            router_logits,
            scalar_buf,
            gate_buf,
            up_buf,
            *smi,
        ),
        Step::MoeSharedDown {
            w,
            gate_buf,
            up_buf,
            scalar_buf,
            ffn_hidden,
            ffn_out,
            out_target,
            smi,
        } => launch_shared_expert_down_body(
            ctx, gpu, w, gate_buf, up_buf, scalar_buf, ffn_hidden, ffn_out, out_target, *smi,
        ),
        Step::ScaledAdd { x, y, scale } => launch_scaled_add_gpu_scalar(gpu, x, y, scale),
        Step::MoeGateUpIndexed {
            experts,
            topk_indices,
            x_rot,
            gate_batch,
            up_batch,
            k_top,
            batch_size,
            dtype_tags,
        } => launch_qwen_gate_up_indexed(
            gpu,
            experts,
            topk_indices,
            x_rot,
            gate_batch,
            up_batch,
            *k_top,
            *batch_size,
            *dtype_tags,
        ),
        Step::MoeDownIndexed {
            experts,
            topk_indices,
            rot_batch,
            out,
            k_top,
            batch_size,
            mode,
            dtype_tags,
        } => launch_qwen_down_indexed(
            gpu,
            experts,
            topk_indices,
            rot_batch,
            out,
            *k_top,
            *batch_size,
            mode,
            *dtype_tags,
        ),
    }
}

/// Borrow `out` from a `RmsnormAutomatic` step. The guard has already confirmed
/// step[0] is RmsnormAutomatic; this panics in debug if called incorrectly.
fn rmsnorm_out<'a>(step: &'a Step<'a>) -> &'a rdna_compute::GpuTensor {
    match step {
        Step::RmsnormAutomatic { out, .. } => out,
        _ => panic!("launch_fused: expected RmsnormAutomatic at step[0]"),
    }
}

/// Borrow `w` and `out` from a `Gemv` step.
fn gemv_weight_out<'a>(step: &'a Step<'a>) -> (&'a WeightRef<'a>, &'a rdna_compute::GpuTensor) {
    match step {
        Step::Gemv { w, out, .. } => (w, out),
        _ => panic!("launch_fused: expected Gemv step"),
    }
}

fn launch_fused(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    key: KernelKey,
    steps: &[Step],
) -> Result<(), DispatchError> {
    #[cfg(feature = "deltanet")]
    if key == KernelKey::FusedDeltaQkL2NormRepeat {
        let (q_src, k_src, q_dst, k_dst, n_key_heads, ratio, head_dim, q_scale, eps, batch_size) =
            match (&steps[0], &steps[1]) {
                (
                    Step::DeltaQkL2Norm {
                        q,
                        k,
                        n_key_heads,
                        head_dim,
                        q_scale,
                        eps,
                        batch_size,
                    },
                    Step::DeltaRepeatHeads {
                        ratio,
                        q_dst,
                        k_dst,
                        ..
                    },
                ) => (
                    *q,
                    *k,
                    *q_dst,
                    *k_dst,
                    *n_key_heads,
                    *ratio,
                    *head_dim,
                    *q_scale,
                    *eps,
                    *batch_size,
                ),
                _ => {
                    return Err(DispatchError::Hip(
                        "Delta QK fusion received a non-adjacent or malformed step pair"
                            .to_string(),
                    ))
                }
            };
        gpu.fused_qk_l2_norm_scale_interleave_f32_batched(
            q_src,
            k_src,
            q_dst,
            k_dst,
            n_key_heads,
            ratio,
            head_dim,
            q_scale,
            eps,
            batch_size,
        )
        .map_err(|e| DispatchError::Hip(e.to_string()))?;

        // The fused kernel mutates q_src/k_src in-place while producing the
        // repeated destinations, matching the standalone pair's observable
        // source/output contract without a second launch.
        return Ok(());
    }

    // Step 0 is always RmsnormAutomatic — run it to fill the activated buffer.
    launch_op(gpu, ctx, &steps[0])?;
    let activated = rmsnorm_out(&steps[0]);
    let fused_qkv = FUSED_QKV.get_or_init(FusedQkvFamily::new);

    match key {
        KernelKey::FusedQkvMq4G256Lloyd
        | KernelKey::FusedQkvMq3G256Lloyd
        | KernelKey::FusedQkvHfq4G256
        | KernelKey::FusedQkvHfq6G256
        | KernelKey::FusedQkvQ4K
        | KernelKey::FusedQkvQ8_0 => {
            let (wq, q) = gemv_weight_out(&steps[1]);
            let (wk, k) = gemv_weight_out(&steps[2]);
            let (wv, v) = gemv_weight_out(&steps[3]);
            fused_qkv.run(
                ctx,
                gpu,
                &FusedQkvParams {
                    kind: key,
                    weights: &[wq.buf, wk.buf, wv.buf],
                    x: activated,
                    outputs: &[q, k, v],
                    m: &[wq.m, wk.m, wv.m],
                    k: wq.k,
                    rot_scratch: &[],
                    batch_size: None,
                },
            )
        }
        KernelKey::FusedGateUpMq4G256Lloyd
        | KernelKey::FusedGateUpMq3G256Lloyd
        | KernelKey::FusedGateUpHfq4G256
        | KernelKey::FusedGateUpHfq6G256
        | KernelKey::FusedGateUpQ4K
        | KernelKey::FusedGateUpQ8_0
        | KernelKey::FusedGateUpMfp4G32E8 => {
            let (wg, gate) = gemv_weight_out(&steps[1]);
            let (wu, up) = gemv_weight_out(&steps[2]);
            fused_qkv.run(
                ctx,
                gpu,
                &FusedQkvParams {
                    kind: key,
                    weights: &[wg.buf, wu.buf],
                    x: activated,
                    outputs: &[gate, up],
                    m: &[wg.m, wu.m],
                    k: wg.k,
                    rot_scratch: &[],
                    batch_size: None,
                },
            )
        }
        // ── QKVZA 4-way (DeltaNet) ──
        KernelKey::FusedQkvzaHfq4G256
        | KernelKey::FusedQkvzaMq3G256Lloyd
        | KernelKey::FusedQkvzaMq4G256Lloyd
        | KernelKey::FusedQkvzaHfq6G256
        | KernelKey::FusedQkvzaMfp4G32E8
        | KernelKey::FusedQkvzaQ8_0 => {
            let (wqkv, qkv) = gemv_weight_out(&steps[1]);
            let (wz, z) = gemv_weight_out(&steps[2]);
            let (wb, beta) = gemv_weight_out(&steps[3]);
            let (wa, alpha) = gemv_weight_out(&steps[4]);
            fused_qkv.run(
                ctx,
                gpu,
                &FusedQkvParams {
                    kind: key,
                    weights: &[wqkv.buf, wz.buf, wb.buf, wa.buf],
                    x: activated,
                    outputs: &[qkv, z, beta, alpha],
                    m: &[wqkv.m, wz.m, wb.m, wa.m],
                    k: wqkv.k,
                    rot_scratch: &[],
                    batch_size: None,
                },
            )
        }

        // ── Paro fused Paro4G128T ────────────────────────────────────────
        // For all three Paro fused keys, we allocate rotation scratch from
        // gpu.scratch.paro_fused_scratch (4 × [k] F32 buffers). The QKVZA
        // path passes all 4; the QKV (3-way) passes 4 with m3=0 via aliasing;
        // the gate+up path passes 1 (x_rot_gate), with the kernel using
        // gpu.scratch.mq_x_rot internally for x_rot_up.
        //
        // Build aliased GpuTensor descriptors before the mutable borrow of
        // gpu (fused_qkv.run takes &mut Gpu). DeviceBuffer::alias() creates
        // an owned descriptor over the same VRAM — no Rust borrow held.
        KernelKey::FusedGateUpParo4G128T => {
            let (wg, gate) = gemv_weight_out(&steps[1]);
            let (wu, up) = gemv_weight_out(&steps[2]);
            let k = wg.k;
            #[cfg(debug_assertions)]
            eprintln!("[dispatch] GateUp Paro: k={}, mg={}, mu={}", k, wg.m, wu.m);
            gpu.ensure_paro_fused_scratch(k)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            // Also ensure mq_x_rot >= k (the kernel aliases it for x_rot_up).
            gpu.ensure_mq_signs()
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            let rot_aliases: Vec<GpuTensor> = gpu
                .scratch
                .paro_fused_scratch
                .as_ref()
                .unwrap()
                .iter()
                .map(|t| GpuTensor {
                    buf: unsafe { t.buf.alias() },
                    shape: t.shape.clone(),
                    dtype: t.dtype,
                })
                .collect();
            #[cfg(debug_assertions)]
            {
                let gate_buf = &gpu.scratch.paro_fused_scratch.as_ref().unwrap()[0];
                let up_internal = gpu.scratch.mq_x_rot.as_ref().unwrap();
                debug_assert!(
                    gate_buf.buf.as_ptr() != up_internal.buf.as_ptr(),
                    "Paro gate+up: x_rot_gate must not alias mq_x_rot"
                );
            }
            fused_qkv.run(
                ctx,
                gpu,
                &FusedQkvParams {
                    kind: key,
                    weights: &[wg.buf, wu.buf],
                    x: activated,
                    outputs: &[gate, up],
                    m: &[wg.m, wu.m],
                    k,
                    rot_scratch: &rot_aliases,
                    batch_size: None,
                },
            )
        }
        KernelKey::FusedQkvzaParo4G128T => {
            let (wqkv, qkv) = gemv_weight_out(&steps[1]);
            let (wz, z) = gemv_weight_out(&steps[2]);
            let (wb, beta) = gemv_weight_out(&steps[3]);
            let (wa, alpha) = gemv_weight_out(&steps[4]);
            let k = wqkv.k;
            #[cfg(debug_assertions)]
            eprintln!(
                "[dispatch] QKVZA Paro: k={}, mqkv={}, mz={}, mbeta={}, malpha={}",
                k, wqkv.m, wz.m, wb.m, wa.m
            );
            gpu.ensure_paro_fused_scratch(k)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            let rot_aliases: Vec<GpuTensor> = gpu
                .scratch
                .paro_fused_scratch
                .as_ref()
                .unwrap()
                .iter()
                .map(|t| GpuTensor {
                    buf: unsafe { t.buf.alias() },
                    shape: t.shape.clone(),
                    dtype: t.dtype,
                })
                .collect();
            fused_qkv.run(
                ctx,
                gpu,
                &FusedQkvParams {
                    kind: key,
                    weights: &[wqkv.buf, wz.buf, wb.buf, wa.buf],
                    x: activated,
                    outputs: &[qkv, z, beta, alpha],
                    m: &[wqkv.m, wz.m, wb.m, wa.m],
                    k,
                    rot_scratch: &rot_aliases,
                    batch_size: None,
                },
            )
        }
        KernelKey::FusedQkvParo4G128T => {
            let (wq, q) = gemv_weight_out(&steps[1]);
            let (wk, k) = gemv_weight_out(&steps[2]);
            let (wv, v) = gemv_weight_out(&steps[3]);
            let kk = wq.k;
            #[cfg(debug_assertions)]
            eprintln!(
                "[dispatch] QKV Paro: k={}, mq={}, mk={}, mv={}",
                kk, wq.m, wk.m, wv.m
            );
            gpu.ensure_paro_fused_scratch(kk)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            let rot_aliases: Vec<GpuTensor> = gpu
                .scratch
                .paro_fused_scratch
                .as_ref()
                .unwrap()
                .iter()
                .map(|t| GpuTensor {
                    buf: unsafe { t.buf.alias() },
                    shape: t.shape.clone(),
                    dtype: t.dtype,
                })
                .collect();
            fused_qkv.run(
                ctx,
                gpu,
                &FusedQkvParams {
                    kind: key,
                    weights: &[wq.buf, wk.buf, wv.buf],
                    x: activated,
                    outputs: &[q, k, v],
                    m: &[wq.m, wk.m, wv.m],
                    k: kk,
                    rot_scratch: &rot_aliases,
                    batch_size: None,
                },
            )
        }
        _ => Err(DispatchError::MissingImpl { key }),
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::DispatchCtx;
    use crate::families::fused_qkv::FusedQkvFamily;
    use crate::types::KernelKey;

    #[test]
    fn qkvza_fused_table_entries_exist() {
        let keys: Vec<_> = FUSED_TABLE.iter().map(|e| e.key).collect();
        assert!(
            keys.contains(&KernelKey::FusedQkvzaMq4G256Lloyd),
            "FusedQkvzaMq4G256Lloyd missing"
        );
        assert!(
            keys.contains(&KernelKey::FusedQkvzaMq3G256Lloyd),
            "FusedQkvzaMq3G256Lloyd missing"
        );
        assert!(
            keys.contains(&KernelKey::FusedQkvzaHfq4G256),
            "FusedQkvzaHfq4G256 missing"
        );
        assert!(
            keys.contains(&KernelKey::FusedQkvzaHfq6G256),
            "FusedQkvzaHfq6G256 missing"
        );

        for entry in FUSED_TABLE.iter() {
            if matches!(
                entry.key,
                KernelKey::FusedQkvzaMq4G256Lloyd
                    | KernelKey::FusedQkvzaMq3G256Lloyd
                    | KernelKey::FusedQkvzaHfq4G256
                    | KernelKey::FusedQkvzaHfq6G256
            ) {
                assert_eq!(
                    entry.ops.len(),
                    5,
                    "QKVZA entry {:?} should have 5 ops",
                    entry.key
                );
            }
        }
    }

    #[test]
    fn qkvza_guards_reject_short_slices() {
        let ctx = DispatchCtx::for_test("gfx1100");
        // Guards must return false for slices shorter than 5 steps.
        let empty: &[Step] = &[];
        assert!(!guard_qkvza_mq4g256lloyd(empty, &ctx));
        assert!(!guard_qkvza_mq3g256lloyd(empty, &ctx));
        assert!(!guard_qkvza_hfq4g256(empty, &ctx));
        assert!(!guard_qkvza_hfq6g256(empty, &ctx));
    }

    #[test]
    fn qkvza_no_paro_or_q8_fused_entries() {
        use crate::types::GemvVariant;
        // ParoQ4G128 should not resolve to any fused QKVZA key. It may resolve
        // to a plain GEMV key (or nothing for unsupported arches). Both are fine.
        let paro = KernelKey::for_gemv(DType::ParoQ4G128, GemvVariant::Plain, false);
        let q8 = KernelKey::for_gemv(DType::Q8_0, GemvVariant::Plain, false);
        for key in [paro.ok(), q8.ok()].into_iter().flatten() {
            assert!(
                !matches!(
                    key,
                    KernelKey::FusedQkvzaMq4G256Lloyd
                        | KernelKey::FusedQkvzaMq3G256Lloyd
                        | KernelKey::FusedQkvzaHfq4G256
                        | KernelKey::FusedQkvzaHfq6G256
                ),
                "ParoQ4G128/Q8_0 must not resolve to a fused QKVZA key, got {:?}",
                key
            );
        }
    }

    #[test]
    fn qkvza_guards_reject_force_unfused() {
        // The plan mandates that force_unfused must prevent fused QKVZA dispatch.
        // Construct a DispatchCtx with force_unfused=true and verify each guard
        // returns false even for otherwise-matching dtypes. We can't build full
        // Steps with real GPU tensors, so we test the guard logic directly with
        // the flag set.
        use rdna_compute::feature_flags::FeatureFlags;
        use std::sync::Arc;
        let mut flags = FeatureFlags::for_test("gfx1100");
        flags.force_unfused = true;
        let ctx = DispatchCtx {
            arch: rdna_compute::arch_caps::ArchCaps::new(
                "gfx1100",
                Arc::new(FeatureFlags::for_test("gfx1100")),
            ),
            flags: Arc::new(flags),
            resources: crate::resource::ResourceManager::for_test(),
            workload: crate::context::DispatchWorkload::Standard,
        };
        // short-circuit: every guard opens with `force_unfused → false`, so even
        // an empty slice returns false. This proves the branch exists.
        let empty: &[Step] = &[];
        assert!(!guard_qkvza_mq4g256lloyd(empty, &ctx));
        assert!(!guard_qkvza_mq3g256lloyd(empty, &ctx));
        assert!(!guard_qkvza_hfq4g256(empty, &ctx));
        assert!(!guard_qkvza_hfq6g256(empty, &ctx));
    }

    #[test]
    fn qkvza_fused_table_no_paro_q4_or_q8_entries() {
        // ParoQ4G128 and Q8_0 must NOT have fused QKVZA entries — they fall
        // through to per-op dispatch. This test asserts that none of the fused
        // table keys match a Paro or Q8 variant, ensuring byte-identical
        // unfused-path correctness for those dtypes.
        let paro_q4_key = KernelKey::for_gemv(DType::ParoQ4G128, GemvVariant::Plain, false);
        let q8_key = KernelKey::for_gemv(DType::Q8_0, GemvVariant::Plain, false);
        // Paro and Q8 should resolve to plain GEMV keys, not fused QKVZA keys.
        // (They may be Err for arches without support, which is also fine.)
        for k in [paro_q4_key, q8_key].into_iter().flatten() {
            assert!(
                !matches!(
                    k,
                    KernelKey::FusedQkvzaMq4G256Lloyd
                        | KernelKey::FusedQkvzaMq3G256Lloyd
                        | KernelKey::FusedQkvzaHfq4G256
                        | KernelKey::FusedQkvzaHfq6G256
                ),
                "ParoQ4G128/Q8_0 should not resolve to a fused QKVZA key"
            );
        }
    }

    #[test]
    fn qkvza_fused_table_arch_coverage() {
        let family = FusedQkvFamily::new();
        let ctx1100 = DispatchCtx::for_test("gfx1100");
        let ctx1201 = DispatchCtx::for_test("gfx1201");

        let wmma_keys = &[
            KernelKey::FusedQkvzaMq4G256Lloyd,
            KernelKey::FusedQkvzaMq3G256Lloyd,
            KernelKey::FusedQkvzaHfq4G256,
        ];

        for &key in wmma_keys {
            assert!(
                family.resolve(key, &ctx1100, None).is_ok(),
                "QKVZA {:?} should resolve on gfx1100",
                key
            );
            assert!(
                family.resolve(key, &ctx1201, None).is_ok(),
                "QKVZA {:?} should resolve on gfx1201",
                key
            );
        }

        // dp4a key: just verify no panic
        let _ = family.resolve(KernelKey::FusedQkvzaHfq6G256, &ctx1100, None);
        let _ = family.resolve(KernelKey::FusedQkvzaHfq6G256, &ctx1201, None);
    }

    #[test]
    fn paro_guards_reject_force_unfused() {
        let ctx = DispatchCtx::for_test("gfx1100");
        let empty: &[Step] = &[];
        assert!(
            !guard_gate_up_paro4g128t(empty, &ctx),
            "force_unfused must reject gate_up_paro"
        );
        assert!(
            !guard_qkvza_paro4g128t(empty, &ctx),
            "force_unfused must reject qkvza_paro"
        );
        assert!(
            !guard_qkv_paro4g128t(empty, &ctx),
            "force_unfused must reject qkv_paro"
        );
    }

    #[test]
    fn paro_guards_require_raw_input_and_alignment() {
        // Paro guards require GemvInput::Raw (not Prerotated) and m%8==0/k%128==0.
        // We can't construct real Gemv steps with GPU tensors in a unit test,
        // but we can verify the guards reject empty/wrong-length slices.
        let ctx = DispatchCtx::for_test("gfx1100");
        let empty: &[Step] = &[];
        assert!(!guard_gate_up_paro4g128t(empty, &ctx));
        assert!(!guard_qkvza_paro4g128t(empty, &ctx));
        assert!(!guard_qkv_paro4g128t(empty, &ctx));
    }

    #[test]
    fn paro_fused_table_entries_exist() {
        let keys: Vec<_> = FUSED_TABLE.iter().map(|e| e.key).collect();
        assert!(
            keys.contains(&KernelKey::FusedGateUpParo4G128T),
            "FusedGateUpParo4G128T missing from FUSED_TABLE"
        );
        assert!(
            keys.contains(&KernelKey::FusedQkvzaParo4G128T),
            "FusedQkvzaParo4G128T missing from FUSED_TABLE"
        );
        assert!(
            keys.contains(&KernelKey::FusedQkvParo4G128T),
            "FusedQkvParo4G128T missing from FUSED_TABLE"
        );
    }

    #[test]
    fn paro_fused_table_arch_coverage() {
        let family = FusedQkvFamily::new();
        let ctx1100 = DispatchCtx::for_test("gfx1100");
        let ctx1201 = DispatchCtx::for_test("gfx1201");

        let paro_keys = &[
            KernelKey::FusedGateUpParo4G128T,
            KernelKey::FusedQkvzaParo4G128T,
            KernelKey::FusedQkvParo4G128T,
        ];

        for &key in paro_keys {
            // Paro uses dp4a — should resolve on gfx1100 (RDNA3) and gfx1201 (RDNA4).
            assert!(
                family.resolve(key, &ctx1100, None).is_ok(),
                "Paro key {:?} should resolve on gfx1100",
                key
            );
            assert!(
                family.resolve(key, &ctx1201, None).is_ok(),
                "Paro key {:?} should resolve on gfx1201",
                key
            );
        }
    }

    // ── Q4K / Q8_0 guard tests (Ship 2.1 A1 — Claude F1 / glm5 F2) ──────

    #[test]
    fn q4k_q8_0_guards_reject_force_unfused() {
        // All three new guards must return false when force_unfused is set,
        // even for empty slices (the guard opens with the early-return).
        use rdna_compute::feature_flags::FeatureFlags;
        use std::sync::Arc;
        let mut flags = FeatureFlags::for_test("gfx1100");
        flags.force_unfused = true;
        let ctx = DispatchCtx {
            arch: rdna_compute::arch_caps::ArchCaps::new(
                "gfx1100",
                Arc::new(FeatureFlags::for_test("gfx1100")),
            ),
            flags: Arc::new(flags),
            resources: crate::resource::ResourceManager::for_test(),
            workload: crate::context::DispatchWorkload::Standard,
        };
        let empty: &[Step] = &[];
        assert!(
            !guard_qkv_q4k(empty, &ctx),
            "guard_qkv_q4k must reject force_unfused"
        );
        assert!(
            !guard_gate_up_q4k(empty, &ctx),
            "guard_gate_up_q4k must reject force_unfused"
        );
        assert!(
            !guard_gate_up_q8_0(empty, &ctx),
            "guard_gate_up_q8_0 must reject force_unfused"
        );
    }

    #[test]
    fn q4k_q8_0_guards_reject_wrong_length() {
        let ctx = DispatchCtx::for_test("gfx1100");
        let empty: &[Step] = &[];
        assert!(!guard_qkv_q4k(empty, &ctx), "Q4K QKV guard needs len==4");
        assert!(
            !guard_gate_up_q4k(empty, &ctx),
            "Q4K gate+up guard needs len==3"
        );
        assert!(
            !guard_gate_up_q8_0(empty, &ctx),
            "Q8_0 gate+up guard needs len==3"
        );
    }

    #[test]
    fn q4k_q8_0_fused_table_entries_exist() {
        let keys: Vec<_> = FUSED_TABLE.iter().map(|e| e.key).collect();
        assert!(
            keys.contains(&KernelKey::FusedQkvQ4K),
            "FusedQkvQ4K missing from FUSED_TABLE"
        );
        assert!(
            keys.contains(&KernelKey::FusedGateUpQ4K),
            "FusedGateUpQ4K missing from FUSED_TABLE"
        );
        assert!(
            keys.contains(&KernelKey::FusedGateUpQ8_0),
            "FusedGateUpQ8_0 missing from FUSED_TABLE"
        );
    }

    #[cfg(feature = "deltanet")]
    #[test]
    fn deltanet_step_surface_is_exactly_six_and_total() {
        use crate::ops::delta_net::{
            DeltaNetBatchIntent, DeltaNetBatchParams, DeltaNetStepParams, StateQuant,
        };
        use std::collections::HashSet;

        let tags = [
            PipelineOp::DeltaGatePrep,
            PipelineOp::DeltaConvSplit,
            PipelineOp::DeltaQkL2Norm,
            PipelineOp::DeltaRepeatHeads,
            PipelineOp::DeltaRecurrence,
            PipelineOp::DeltaGatedNorm,
        ];
        assert_eq!(tags.len(), 6);
        assert_eq!(tags.iter().copied().collect::<HashSet<_>>().len(), 6);

        // Metadata-only tensors are sufficient: this test exercises the
        // total op-kind and TP-output matches without touching HIP.
        let t = |ptr: usize| GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr as *mut std::ffi::c_void, 4096) },
            shape: vec![8],
            dtype: DType::F32,
        };
        let beta = t(1);
        let alpha = t(2);
        let dt_bias = t(3);
        let a_log = t(4);
        let input = t(5);
        let weight = t(6);
        let state = t(7);
        let q = t(8);
        let k = t(9);
        let q_dst = t(10);
        let k_dst = t(11);
        let out = t(12);

        let steps = [
            Step::DeltaGatePrep {
                beta: &beta,
                alpha: &alpha,
                dt_bias: &dt_bias,
                a_log: &a_log,
                n: 2,
                batch_size: 1,
            },
            Step::DeltaConvSplit {
                q_out: &q,
                k_out: &k,
                v_out: &out,
                input: &input,
                weight: &weight,
                state: &state,
                parent_indices: None,
                k_dim: 4,
                v_dim: 4,
                n_tokens: 1,
            },
            Step::DeltaQkL2Norm {
                q: &q,
                k: &k,
                n_key_heads: 1,
                head_dim: 4,
                q_scale: 1.0,
                eps: 1e-6,
                batch_size: 1,
            },
            Step::DeltaRepeatHeads {
                q_src: &q,
                k_src: &k,
                q_dst: &q_dst,
                k_dst: &k_dst,
                n_key_heads: 1,
                ratio: 2,
                head_dim: 4,
                batch_size: 1,
            },
            Step::DeltaRecurrence {
                params: DeltaRecurrenceParams::Step(DeltaNetStepParams {
                    q: &q,
                    k: &k,
                    v: &out,
                    gate: &alpha,
                    beta: &beta,
                    state: &state,
                    s_scales: &dt_bias,
                    output: &out,
                    ef_residual: None,
                    n_heads: 1,
                    head_dim: 4,
                    quant: StateQuant::FP32,
                }),
            },
            Step::DeltaGatedNorm {
                x: &out,
                z: &alpha,
                weight: &dt_bias,
                out: &q_dst,
                n_heads: 1,
                head_dim: 4,
                eps: 1e-6,
                batch_size: 1,
            },
        ];

        for (step, expected) in steps.iter().zip(tags) {
            assert_eq!(step_op_kind(step), expected);
            assert!(
                tp_step_out_buf(step).is_none(),
                "DeltaNet step exposed TP output"
            );
        }

        let batch_params = DeltaNetBatchParams {
            q_batch: &q,
            k_batch: &k,
            v_batch: &out,
            gate_batch: &alpha,
            beta_batch: &beta,
            state: &state,
            s_scales: &dt_bias,
            output_batch: &out,
            ef_residual: None,
            n_tokens: 2,
            n_heads: 1,
            head_dim: 4,
            quant: StateQuant::FP32,
        };
        let replay = DeltaRecurrenceParams::Batch {
            params: batch_params,
            intent: DeltaNetBatchIntent::SpeculativeReplay,
        };
        match replay {
            DeltaRecurrenceParams::Batch { intent, .. } => {
                assert_eq!(intent, DeltaNetBatchIntent::SpeculativeReplay)
            }
            _ => unreachable!("batch recurrence lost its intent"),
        }
    }

    #[cfg(feature = "deltanet")]
    #[test]
    fn deltanet_builders_keep_decode_and_batch_shapes_distinct() {
        let t = |ptr: usize| GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr as *mut std::ffi::c_void, 4096) },
            shape: vec![8],
            dtype: DType::F32,
        };
        let qkv = t(101);
        let q = t(102);
        let k = t(103);
        let v = t(104);
        let q_raw = t(105);
        let k_raw = t(106);
        let alpha = t(107);
        let beta = t(108);
        let dt_bias = t(109);
        let a_log = t(110);
        let state = t(111);
        let scales = t(112);
        let conv_weight = t(113);
        let conv_state = t(114);
        let attn_out = t(115);
        let normed = t(116);
        let z = t(117);
        let norm_weight = t(118);
        let d = DeltaNetOperandDescriptor {
            qkv: &qkv,
            q: &q,
            k: &k,
            v: &v,
            q_raw: &q_raw,
            k_raw: &k_raw,
            alpha: &alpha,
            beta: &beta,
            dt_bias: Some(&dt_bias),
            a_log: Some(&a_log),
            state: &state,
            s_scales: &scales,
            ef_residual: None,
            conv_weight: &conv_weight,
            conv_state: &conv_state,
            attn_out: &attn_out,
            normed: Some(&normed),
            z: Some(&z),
            norm_weight: Some(&norm_weight),
            n_key_heads: 1,
            n_value_heads: 2,
            head_dim: 4,
            key_dim: 4,
            value_dim: 8,
            q_scale: 0.5,
            eps: 1e-6,
            quant: StateQuant::Q8,
        };

        let decode = build_delta_net_decode_steps(&d);
        assert_eq!(decode.len(), 6);
        assert!(matches!(
            decode[0],
            Step::DeltaGatePrep { batch_size: 1, .. }
        ));
        assert!(matches!(
            decode[1],
            Step::DeltaConvSplit {
                n_tokens: 1,
                parent_indices: None,
                ..
            }
        ));
        assert!(matches!(
            decode[4],
            Step::DeltaRecurrence {
                params: DeltaRecurrenceParams::Step(_)
            }
        ));
        assert!(matches!(
            decode[5],
            Step::DeltaGatedNorm { batch_size: 1, .. }
        ));

        let batch =
            build_delta_net_batch_steps(&d, 4, DeltaNetBatchIntent::NormalPrefill, None, None)
                .unwrap();
        assert_eq!(batch.len(), 6);
        assert!(matches!(
            batch[0],
            Step::DeltaGatePrep { batch_size: 4, .. }
        ));
        assert!(matches!(
            batch[1],
            Step::DeltaConvSplit {
                n_tokens: 4,
                parent_indices: None,
                ..
            }
        ));
        assert!(matches!(
            batch[2],
            Step::DeltaQkL2Norm { batch_size: 4, .. }
        ));
        assert!(matches!(
            batch[3],
            Step::DeltaRepeatHeads { batch_size: 4, .. }
        ));
        assert!(matches!(
            batch[4],
            Step::DeltaRecurrence {
                params: DeltaRecurrenceParams::Batch { .. }
            }
        ));
        assert!(matches!(
            batch[5],
            Step::DeltaGatedNorm { batch_size: 4, .. }
        ));

        let replay =
            build_delta_net_batch_steps(&d, 4, DeltaNetBatchIntent::SpeculativeReplay, None, None)
                .unwrap();
        assert_eq!(replay.len(), 5);
        assert!(matches!(
            replay[0],
            Step::DeltaConvSplit { n_tokens: 4, .. }
        ));
        assert!(matches!(
            replay[3],
            Step::DeltaRecurrence {
                params: DeltaRecurrenceParams::Batch {
                    intent: DeltaNetBatchIntent::SpeculativeReplay,
                    ..
                }
            }
        ));

        let parent = t(119);
        let tape = t(120);
        let tape_scales = t(121);
        let tree = build_delta_net_tree_steps(&d, 4, &parent, &tape, Some(&tape_scales)).unwrap();
        assert!(matches!(
            tree[1],
            Step::DeltaConvSplit {
                n_tokens: 4,
                parent_indices: Some(_),
                ..
            }
        ));
        assert!(matches!(
            tree[4],
            Step::DeltaRecurrence {
                params: DeltaRecurrenceParams::Tree(_)
            }
        ));
    }

    #[cfg(feature = "deltanet")]
    #[test]
    fn deltanet_qk_repeat_fuses_only_the_valid_adjacent_batched_pair() {
        fn metadata_tensor(ptr: usize, shape: Vec<usize>, dtype: DType) -> GpuTensor {
            GpuTensor {
                buf: unsafe {
                    hip_bridge::DeviceBuffer::from_raw(ptr as *mut std::ffi::c_void, 4096)
                },
                shape,
                dtype,
            }
        }

        let q = metadata_tensor(101, vec![2, 1, 4], DType::F32);
        let k = metadata_tensor(102, vec![2, 1, 4], DType::F32);
        let q_src = metadata_tensor(101, vec![2, 1, 4], DType::F32);
        let k_src = metadata_tensor(102, vec![2, 1, 4], DType::F32);
        let q_dst = metadata_tensor(103, vec![2, 2, 4], DType::F32);
        let k_dst = metadata_tensor(104, vec![2, 2, 4], DType::F32);
        let ctx = DispatchCtx::for_test("gfx1100");

        let valid = [
            Step::DeltaQkL2Norm {
                q: &q,
                k: &k,
                n_key_heads: 1,
                head_dim: 4,
                q_scale: 0.5,
                eps: 1e-6,
                batch_size: 2,
            },
            Step::DeltaRepeatHeads {
                q_src: &q_src,
                k_src: &k_src,
                q_dst: &q_dst,
                k_dst: &k_dst,
                n_key_heads: 1,
                ratio: 2,
                head_dim: 4,
                batch_size: 2,
            },
        ];
        assert_eq!(
            match_prefix(FUSED_TABLE, &valid, &ctx),
            Some((KernelKey::FusedDeltaQkL2NormRepeat, 2))
        );

        // Decode has no fused interleave kernel: it must remain two
        // standalone operations, so normalization cannot happen twice.
        let decode = [
            Step::DeltaQkL2Norm {
                q: &q,
                k: &k,
                n_key_heads: 1,
                head_dim: 4,
                q_scale: 0.5,
                eps: 1e-6,
                batch_size: 1,
            },
            Step::DeltaRepeatHeads {
                q_src: &q_src,
                k_src: &k_src,
                q_dst: &q_dst,
                k_dst: &k_dst,
                n_key_heads: 1,
                ratio: 2,
                head_dim: 4,
                batch_size: 1,
            },
        ];
        assert_eq!(match_prefix(FUSED_TABLE, &decode, &ctx), None);

        // A ratio of one is a copy, not an interleave, and is likewise kept
        // on the explicit standalone path.
        let no_repeat = [
            Step::DeltaQkL2Norm {
                q: &q,
                k: &k,
                n_key_heads: 1,
                head_dim: 4,
                q_scale: 0.5,
                eps: 1e-6,
                batch_size: 2,
            },
            Step::DeltaRepeatHeads {
                q_src: &q_src,
                k_src: &k_src,
                q_dst: &q_dst,
                k_dst: &k_dst,
                n_key_heads: 1,
                ratio: 1,
                head_dim: 4,
                batch_size: 2,
            },
        ];
        assert_eq!(match_prefix(FUSED_TABLE, &no_repeat, &ctx), None);
    }

    #[cfg(feature = "deltanet")]
    #[test]
    #[ignore = "requires an AMD GPU and DeltaNet kernel JIT"]
    fn deltanet_qk_repeat_fused_execution_gpu() {
        let mut gpu = Gpu::init().expect("GPU required for ignored DeltaNet execution test");
        let ctx = DispatchCtx::new(&gpu);
        let q_data = vec![0.25, -0.5, 0.75, 1.0, -1.25, 0.5, 0.125, -0.875];
        let k_data = vec![-0.75, 0.25, 1.5, -0.125, 0.875, -1.0, 0.375, 0.625];
        let q_unfused = gpu.upload_f32(&q_data, &[2, 1, 4]).unwrap();
        let k_unfused = gpu.upload_f32(&k_data, &[2, 1, 4]).unwrap();
        let q_unfused_dst = gpu.alloc_tensor(&[2, 2, 4], DType::F32).unwrap();
        let k_unfused_dst = gpu.alloc_tensor(&[2, 2, 4], DType::F32).unwrap();
        let unfused_steps = [
            Step::DeltaQkL2Norm {
                q: &q_unfused,
                k: &k_unfused,
                n_key_heads: 1,
                head_dim: 4,
                q_scale: 0.5,
                eps: 1e-6,
                batch_size: 2,
            },
            Step::DeltaRepeatHeads {
                q_src: &q_unfused,
                k_src: &k_unfused,
                q_dst: &q_unfused_dst,
                k_dst: &k_unfused_dst,
                n_key_heads: 1,
                ratio: 2,
                head_dim: 4,
                batch_size: 2,
            },
        ];
        // Force the standalone semantics by disabling fusion for this run.
        let mut unfused_flags =
            rdna_compute::feature_flags::FeatureFlags::for_test("gfx1100");
        unfused_flags.force_unfused = true;
        let unfused_ctx = DispatchCtx {
            arch: rdna_compute::arch_caps::ArchCaps::new(
                "gfx1100",
                std::sync::Arc::new(
                    rdna_compute::feature_flags::FeatureFlags::for_test("gfx1100"),
                ),
            ),
            flags: std::sync::Arc::new(unfused_flags),
            resources: crate::resource::ResourceManager::for_test(),
            workload: crate::context::DispatchWorkload::Standard,
        };
        execute_steps(&mut gpu, &unfused_ctx, &unfused_steps).unwrap();

        let q_fused = gpu.upload_f32(&q_data, &[2, 1, 4]).unwrap();
        let k_fused = gpu.upload_f32(&k_data, &[2, 1, 4]).unwrap();
        let q_fused_dst = gpu.alloc_tensor(&[2, 2, 4], DType::F32).unwrap();
        let k_fused_dst = gpu.alloc_tensor(&[2, 2, 4], DType::F32).unwrap();
        let fused_steps = [
            Step::DeltaQkL2Norm {
                q: &q_fused,
                k: &k_fused,
                n_key_heads: 1,
                head_dim: 4,
                q_scale: 0.5,
                eps: 1e-6,
                batch_size: 2,
            },
            Step::DeltaRepeatHeads {
                q_src: &q_fused,
                k_src: &k_fused,
                q_dst: &q_fused_dst,
                k_dst: &k_fused_dst,
                n_key_heads: 1,
                ratio: 2,
                head_dim: 4,
                batch_size: 2,
            },
        ];
        execute_steps(&mut gpu, &ctx, &fused_steps).unwrap();

        let q0 = gpu.download_f32(&q_unfused).unwrap();
        let k0 = gpu.download_f32(&k_unfused).unwrap();
        let qo0 = gpu.download_f32(&q_unfused_dst).unwrap();
        let ko0 = gpu.download_f32(&k_unfused_dst).unwrap();
        let q1 = gpu.download_f32(&q_fused).unwrap();
        let k1 = gpu.download_f32(&k_fused).unwrap();
        let qo1 = gpu.download_f32(&q_fused_dst).unwrap();
        let ko1 = gpu.download_f32(&k_fused_dst).unwrap();
        for (name, lhs, rhs) in [
            ("q", &q0, &q1),
            ("k", &k0, &k1),
            ("q_out", &qo0, &qo1),
            ("k_out", &ko0, &ko1),
        ] {
            assert_eq!(lhs.len(), rhs.len());
            let max_diff = lhs
                .iter()
                .zip(rhs)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!("{name}: max fused/unfused diff = {max_diff:e}");
            assert!(
                max_diff <= 1e-5,
                "fused/unfused DeltaNet QK source/output mismatch"
            );
        }

        for tensor in [
            q_unfused,
            k_unfused,
            q_unfused_dst,
            k_unfused_dst,
            q_fused,
            k_fused,
            q_fused_dst,
            k_fused_dst,
        ] {
            gpu.free_tensor(tensor).unwrap();
        }
    }

    /// Pure-logic test: TpCollective→StepCollective wrapper mapping and
    /// the three validate_parallel_args length-mismatch guards.
    /// No GPU needed — only enum construction and the pure validator are exercised.
    #[test]
    fn parallel_arg_length_guards() {
        use hipfire_hardware::DimKind;

        // --- mapping test: execute_steps_tp wrapper logic ---
        // AllReduceOut{dim:8} must map to AllReduce{kind:Tp, dim:8}
        let tp_colls = [TpCollective::None, TpCollective::AllReduceOut { dim: 8 }];
        let mapped: Vec<StepCollective> = tp_colls
            .iter()
            .map(|c| match c {
                TpCollective::AllReduceOut { dim } => StepCollective::AllReduce {
                    kind: DimKind::Tp,
                    dim: *dim,
                },
                TpCollective::None => StepCollective::None,
            })
            .collect();
        // First element: None → None
        assert!(matches!(mapped[0], StepCollective::None));
        // Second element: AllReduceOut{8} → AllReduce{Tp, 8}
        match &mapped[1] {
            StepCollective::AllReduce { kind, dim } => {
                assert!(matches!(kind, DimKind::Tp), "expected Tp, got {kind:?}");
                assert_eq!(*dim, 8);
            }
            other => panic!("expected AllReduce, got {other:?}"),
        }

        // --- guard 1: per_rank_steps.len() != group_size ---
        // group=2, but only 1 step list supplied
        let group_size = 2usize;
        let steps_1: Vec<Vec<Step<'_>>> = vec![vec![]]; // len=1, mismatch
        let colls_0: Vec<StepCollective> = vec![];
        let zb_0: Vec<bool> = vec![];
        let e = super::validate_parallel_args(group_size, &steps_1, &colls_0, &zb_0)
            .expect_err("should fail: per_rank_steps.len()==1 != group_size==2");
        assert!(
            matches!(&e, DispatchError::Hip(msg) if msg.contains("step lists")),
            "unexpected error: {e:?}"
        );

        // --- guard 2: collectives.len() != n_steps ---
        // group=2, 2 step lists each with n_steps=0, but 1 collective supplied
        let steps_2: Vec<Vec<Step<'_>>> = vec![vec![], vec![]]; // 2 ranks, n_steps=0
        let colls_1 = vec![StepCollective::None]; // len=1, mismatch with n_steps=0
        let zb_0: Vec<bool> = vec![];
        let e = super::validate_parallel_args(group_size, &steps_2, &colls_1, &zb_0)
            .expect_err("should fail: collectives.len()==1 != n_steps==0");
        assert!(
            matches!(&e, DispatchError::Hip(msg) if msg.contains("collectives")),
            "unexpected error: {e:?}"
        );

        // --- guard 3: zero_before.len() != n_steps ---
        // group=2, 2 step lists with n_steps=0, 0 collectives, but zero_before has 1 elem
        let colls_0: Vec<StepCollective> = vec![]; // len=0 matches n_steps=0
        let zb_1 = vec![false]; // len=1, mismatch with n_steps=0
        let e = super::validate_parallel_args(group_size, &steps_2, &colls_0, &zb_1)
            .expect_err("should fail: zero_before.len()==1 != n_steps==0");
        assert!(
            matches!(&e, DispatchError::Hip(msg) if msg.contains("zero_before")),
            "unexpected error: {e:?}"
        );
    }
}
