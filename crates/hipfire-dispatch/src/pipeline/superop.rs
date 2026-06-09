// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Forward-as-pipeline (#397 Ship 6) — the **C-design lowered super-op
//! substrate**.
//!
//! A model's per-layer forward is lowered ONCE at model load into a
//! [`LoweredForward`] (a `Vec<LayerProgram>`); each [`LayerProgram`] is a short
//! list of COARSE [`SuperOp`]s that map 1:1 onto the existing fused kernels.
//! Per-token execution is then a tight loop over pre-resolved super-ops — no
//! `resolve()`, no `FUSED_TABLE`/`match_prefix` walk, no per-call `WeightRef`
//! construction. The fusion decision (which `FUSED_TABLE` entry fires) and the
//! kernel `KernelKey` are resolved at LOWER time and frozen in [`OpBinding`].
//!
//! ## Why super-ops are POD (no lifetimes / no raw-ptr capture)
//! Design B (a `Box<dyn Fn(&mut Gpu, &Frame)>` per op) was rejected: collapsing
//! the today-disjoint `&mut Gpu / &mut KvCache / &mut DeltaNetState / &Scratch`
//! args into one `Frame` forces `RefCell` (hot-path borrow checks = perf loss)
//! or `UnsafeCell` (the a9e8dfda aliasing-bug class, minus the compiler). So a
//! `SuperOp` carries only **indices** ([`WeightSlot`]/[`ScratchSlot`]) + a
//! resolved [`KernelKey`] + flavor data — pure POD. The per-token executor
//! (built in a later step) re-borrows the live `GpuTensor`s from the model's
//! weight/scratch/state tables BY INDEX and calls the resolved family method,
//! so the compiler still proves disjointness at each call site.
//!
//! ## Coverage (the whole served fleet, one substrate)
//! - `Proj` / `ResidualGemv` / `Moe`     → qwen35, MiniMax(reuse), cohere2moe(reuse)
//! - `Attend` (flavor-carrying)          → all; Gemma SWA/qk-norm/softcap/k_eq_v live here as flavors
//! - `Recurrent`                         → qwen35 DeltaNet linear-attention state
//! - `Conv`                              → LFM2 depthwise causal short-conv mixer (+ conv state)
//! - `Escape(EscapeKind)`                → irregular/stateful: deepseek4 compressor/indexer/SWA, etc.
//!
//! NOTE: nothing here is on a live path yet. The live forward remains
//! `execute_steps`; this substrate is wired behind `HIPFIRE_FORWARD_LOWERED`
//! (default off) in later steps, validated byte-identical via the
//! `HIPFIRE_FORWARD_ORACLE` dual-run.

use crate::context::DispatchCtx;
use crate::types::{KernelKey, PipelineOp};
use super::steps::{match_fused_prefix, step_op_kind, Step};
use crate::types::DispatchError;
use rdna_compute::{Gpu, GpuTensor};

/// Index into the model's per-layer weight table (resolved at lower time, stable
/// for the model's lifetime). The executor maps this to the live `&GpuTensor`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightSlot(pub u32);

/// Typed handle into the live, per-token scratch/state/cache buffers. Kept TYPED
/// (not a bare index) so an activation buffer can never be confused with an
/// advancing-state or KV-cache buffer — the spike's #30/a9e8dfda-class
/// (stateful-rebind silent-wrong-output) mitigation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScratchSlot {
    /// Transient per-token activation (hidden state, rotated x, gate/up bufs…).
    Activation(u32),
    /// Per-token-ADVANCING recurrent state (DeltaNet double-buffer, LFM2 conv
    /// state). Rebind MUST recompute exactly where the hand path does.
    State(u32),
    /// KV-cache buffer + write-offset (advancing). Same rebind-fragility class.
    Cache(u32),
}

/// FFN/gate-up activation flavor. SiLU for qwen-family; GeLU-tanh (GeGLU) for
/// Gemma (`gelu_tanh(gate)·up`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActFlavor {
    SiluMul,
    GeluTanhMul,
}

/// RoPE flavor carried by an `Attend` super-op (resolved from config at load).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RopeFlavor {
    None,
    /// Standard rotate-half (most archs). `theta` = rope base.
    HalfRotate { theta: f32 },
    /// Interleaved full-dim RoPE (e.g. cohere2moe).
    Interleaved { theta: f32 },
}

/// Attention-block flavor — everything that distinguishes one arch's attention
/// from another, resolved at load so the per-token `Attend` is branch-free.
/// Gemma exercises the full surface (SWA window, per-head qk-norm, q·√hd scaling,
/// the k_eq_v weightless-V-RMSNorm prelude, logit softcap).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttnFlavor {
    /// Sliding-window size; 0 = full (global) attention. Gemma alternates
    /// Sliding(1024)/Full per layer.
    pub window: u32,
    /// Per-head q_norm/k_norm over head_dim (Gemma, qwen35).
    pub qk_norm: bool,
    /// q *= sqrt(head_dim) (Gemma query scaling).
    pub q_scale_sqrt_hd: bool,
    /// V = copy of K before k_norm + weightless RMSNorm on V (Gemma full layers).
    pub k_eq_v: bool,
    /// Attention-logit softcap value; 0.0 = none (Gemma-2 style).
    pub logit_softcap: f32,
    pub rope: RopeFlavor,
}

/// Per-super-op flavor payload (None for ops with no flavor axis, e.g. Proj).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OpFlavor {
    None,
    Attn(AttnFlavor),
    Act(ActFlavor),
}

/// Irregular/stateful ops that don't map onto a single fused kernel. Each is a
/// typed tag the executor matches to a bespoke `gpu.*` sequence (NOT dyn-trait).
/// Extensible: a new irregular arch adds a variant + an executor arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EscapeKind {
    /// deepseek4 MLA compressor (called twice/layer: main + indexer sub-compressor).
    Deepseek4Compressor,
    /// deepseek4 indexer top-K selection.
    Deepseek4IndexerTopK,
    /// deepseek4 sparse SWA over the gathered top-K KV.
    Deepseek4SwaTopK,
    /// Gemma final logit softcap (output stage).
    GemmaLogitSoftcap,
}

/// One coarse super-op. For `Proj`/`ResidualGemv`/`Moe` the `key` is the
/// FUSED_TABLE/`resolve()` result frozen at lower time; for `Attend`/`Recurrent`/
/// `Conv`/`Escape` it may be `None` (those route by kind + flavor + escape tag).
#[derive(Clone, Debug)]
pub struct SuperOp {
    pub kind: SuperOpKind,
    pub binding: OpBinding,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SuperOpKind {
    /// Fused projection cluster (QKV / QKVZA / gate+up / plain GEMM).
    Proj,
    /// Output/down projection with fused residual add (o_proj, down_proj).
    ResidualGemv,
    /// Standalone rmsnorm(+optional rotate) producer, emitted when its
    /// projection cluster did NOT fuse — mirrors execute_steps' unfused
    /// per-step launch_op path (rmsnorm, then individual Proj gemvs).
    Norm,
    /// Attention block (flavor in `OpFlavor::Attn`).
    Attend,
    /// MoE FFN block (routes to MoeFamily / run_moe_decode).
    Moe,
    /// Recurrent linear-attention state advance (DeltaNet).
    Recurrent,
    /// Depthwise causal short-conv mixer with advancing conv state (LFM2).
    Conv,
    /// Bespoke irregular/stateful op.
    Escape(EscapeKind),
}

/// Pre-resolved binding for one super-op. Pure POD (indices + key + flavor) — no
/// borrows, no raw pointers; the executor binds against live state by index.
#[derive(Clone, Debug)]
pub struct OpBinding {
    /// Kernel resolved at lower time (FUSED_TABLE/resolve result). `None` for
    /// ops dispatched by kind+flavor+escape rather than a single GEMM key.
    pub key: Option<KernelKey>,
    /// Weight operands, in the order the kernel expects (e.g. [wq,wk,wv] for a
    /// QKV Proj). Indices into the model's weight table.
    pub weights: Vec<WeightSlot>,
    /// Input/output/scratch/state operands the executor binds per token.
    pub scratch: Vec<ScratchSlot>,
    /// Attention/activation/rope flavor (or `None`).
    pub flavor: OpFlavor,
}

/// A lowered per-layer program: the ordered super-ops for one transformer layer.
pub type LayerProgram = Vec<SuperOp>;

/// The whole lowered forward for a model: one `LayerProgram` per layer, plus a
/// generation counter guarding the load-time weight binding against any future
/// on-the-fly requant / adaptive-KV weight-floor realloc (40d98d4d). The
/// executor asserts `weight_gen == live weight-set gen` (debug) and re-lowers on
/// mismatch — the spike's stale-alias (risk #2) mitigation.
#[derive(Clone, Debug)]
pub struct LoweredForward {
    pub layers: Vec<LayerProgram>,
    pub weight_gen: u64,
}

impl LoweredForward {
    pub fn new(weight_gen: u64) -> Self {
        Self { layers: Vec::new(), weight_gen }
    }
}

// ── Step 1b: load-time lowering ────────────────────────────────────────────

/// Pure greedy lowering walk (no `Step`/GpuTensor dependency → unit-testable).
/// Mirrors `execute_steps` exactly: a fused window collapses to one `Proj`
/// super-op carrying the frozen `KernelKey`; an unfused position becomes a
/// single super-op (the `launch_op` equivalent). `try_fuse(pos)` returns the
/// fused `(key, span)` at `pos` (from `match_fused_prefix`); `single_kind(pos)`
/// gives the kind for an unfused step. A `span == 0` defensively falls through
/// to the single-step branch so the walk always advances.
fn lower_walk(
    n: usize,
    single_kind: impl Fn(usize) -> SuperOpKind,
    try_fuse: impl Fn(usize) -> Option<(KernelKey, usize)>,
) -> LayerProgram {
    let mut prog: LayerProgram = Vec::new();
    let mut pos = 0usize;
    while pos < n {
        match try_fuse(pos) {
            Some((key, span)) if span >= 1 => {
                prog.push(SuperOp {
                    kind: SuperOpKind::Proj,
                    binding: OpBinding {
                        key: Some(key),
                        weights: Vec::new(),
                        scratch: Vec::new(),
                        flavor: OpFlavor::None,
                    },
                });
                pos += span;
            }
            _ => {
                prog.push(SuperOp {
                    kind: single_kind(pos),
                    binding: OpBinding {
                        key: None,
                        weights: Vec::new(),
                        scratch: Vec::new(),
                        flavor: OpFlavor::None,
                    },
                });
                pos += 1;
            }
        }
    }
    prog
}

/// Lower one layer's live `Step` sequence into a POD [`LayerProgram`] at MODEL
/// LOAD. Reuses `match_fused_prefix` (the canonical `FUSED_TABLE` + guards) so
/// the frozen `KernelKey`s match `execute_steps` byte-for-byte — no fusion drift.
///
/// TODO(arch-migration, Step 3+): populate `WeightSlot`/`ScratchSlot` bindings
/// and `AttnFlavor`/`ActFlavor` from the arch's weight/scratch tables + config.
/// This step freezes the FUSION STRUCTURE + kernel keys; the per-arch migration
/// supplies the operand slots + flavor data (and the executor binds them).
pub fn lower_layer(steps: &[Step], ctx: &DispatchCtx) -> LayerProgram {
    lower_walk(
        steps.len(),
        |i| match step_op_kind(&steps[i]) {
            PipelineOp::Gemv => SuperOpKind::Proj,
            PipelineOp::GemvResidual => SuperOpKind::ResidualGemv,
            PipelineOp::RmsnormAutomatic => SuperOpKind::Norm,
            PipelineOp::Attend => SuperOpKind::Attend,
            // Current Step has only the 4 variants above; other PipelineOp
            // values are not producible from a Step, so this is unreachable
            // in practice (kept total).
            _ => SuperOpKind::Norm,
        },
        |i| match_fused_prefix(&steps[i..], ctx),
    )
}

// ── Step 1c: the LayerProgram executor ─────────────────────────────────────

/// Per-kind handlers the executor calls, implemented ARCH-SIDE (where the live
/// weight/scratch/state tensors live). This keeps `run_layer_program` itself
/// arch-agnostic: it sequences + matches `SuperOpKind`, the arch's impl does the
/// actual family dispatch (e.g. `GemmFamily::run_key(op.key.unwrap(), …)` with
/// the tensors resolved from `op.weights`/`op.scratch` against its own tables).
///
/// A kind the arch hasn't migrated yet routes to its OWN hand-path fragment
/// inside the impl (the "Fallback") — so there is NO `UnsupportedVariant`
/// catch-all on the executor; the match is total over `SuperOpKind`.
pub trait ForwardBindings {
    fn run_proj(&mut self, gpu: &mut Gpu, ctx: &DispatchCtx, op: &OpBinding) -> Result<(), DispatchError>;
    fn run_residual_gemv(&mut self, gpu: &mut Gpu, ctx: &DispatchCtx, op: &OpBinding) -> Result<(), DispatchError>;
    fn run_norm(&mut self, gpu: &mut Gpu, ctx: &DispatchCtx, op: &OpBinding) -> Result<(), DispatchError>;
    fn run_attend(&mut self, gpu: &mut Gpu, ctx: &DispatchCtx, op: &OpBinding) -> Result<(), DispatchError>;
    fn run_moe(&mut self, gpu: &mut Gpu, ctx: &DispatchCtx, op: &OpBinding) -> Result<(), DispatchError>;
    fn run_recurrent(&mut self, gpu: &mut Gpu, ctx: &DispatchCtx, op: &OpBinding) -> Result<(), DispatchError>;
    fn run_conv(&mut self, gpu: &mut Gpu, ctx: &DispatchCtx, op: &OpBinding) -> Result<(), DispatchError>;
    fn run_escape(&mut self, gpu: &mut Gpu, ctx: &DispatchCtx, op: &OpBinding, kind: EscapeKind) -> Result<(), DispatchError>;

    // ── Expert-parallel (Ship 6 substrate-EP) hooks ─────────────────────────
    // Default to unsupported; only EP-target MoE arches (qwen3.6-A3B, MiniMax,
    // DeepSeek-V4) override these. Non-MoE arches and the single-GPU path never
    // call them, so the defaults keep every existing `ForwardBindings` impl
    // compiling unchanged.

    /// EP variant of [`run_moe`](Self::run_moe): the routed combine + shared
    /// down accumulate into `routed_out` (a **zeroed** per-rank partial the EP
    /// executor all-reduces across ranks), and `skip_shared` gates the
    /// shared-expert down to rank 0 (so the replicated shared expert is summed
    /// once). The router/top-k still run replicated; non-owned experts read the
    /// load-time zero-dummy weights → contribute 0 to the partial.
    fn run_moe_ep(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        _routed_out: &GpuTensor,
        _skip_shared: bool,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::UnsupportedVariant {
            family: "ep",
            variant: "run_moe_ep-not-implemented-for-arch",
            arch: "",
            quant: "",
        })
    }

    /// EP: add the all-reduced routed partial into this rank's residual stream
    /// (the arch-specific buffer that holds the post-attention residual). Called
    /// by the EP executor once, after the MoE all-reduce.
    fn ep_add_into_residual(
        &mut self,
        _gpu: &mut Gpu,
        _partial: &GpuTensor,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::UnsupportedVariant {
            family: "ep",
            variant: "ep_add_into_residual-not-implemented-for-arch",
            arch: "",
            quant: "",
        })
    }
}

/// Dispatch a SINGLE super-op to its [`ForwardBindings`] method. Extracted from
/// [`run_layer_program`] so the EP executor (`hipfire_runtime::ep`) can drive
/// the same per-op dispatch for the replicated (non-MoE) ops while special-
/// casing `Moe` with an all-reduce. Total match over `SuperOpKind`.
pub fn dispatch_super_op<B: ForwardBindings>(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &SuperOp,
    bindings: &mut B,
) -> Result<(), DispatchError> {
    match op.kind {
        SuperOpKind::Proj => bindings.run_proj(gpu, ctx, &op.binding)?,
        SuperOpKind::ResidualGemv => bindings.run_residual_gemv(gpu, ctx, &op.binding)?,
        SuperOpKind::Norm => bindings.run_norm(gpu, ctx, &op.binding)?,
        SuperOpKind::Attend => bindings.run_attend(gpu, ctx, &op.binding)?,
        SuperOpKind::Moe => bindings.run_moe(gpu, ctx, &op.binding)?,
        SuperOpKind::Recurrent => bindings.run_recurrent(gpu, ctx, &op.binding)?,
        SuperOpKind::Conv => bindings.run_conv(gpu, ctx, &op.binding)?,
        SuperOpKind::Escape(k) => bindings.run_escape(gpu, ctx, &op.binding, k)?,
    }
    Ok(())
}

/// Execute one lowered layer program: a tight, branch-predictable loop over the
/// pre-resolved super-ops (no `resolve()`, no `FUSED_TABLE` walk, no per-call
/// `WeightRef` construction — that was all frozen at lower time). Total match
/// over `SuperOpKind`; each arm calls the arch-supplied [`ForwardBindings`].
///
/// This is the forward-as-pipeline hot path. It is NOT on any live path until an
/// arch's forward (qwen35 first) builds a `LoweredForward` at load and calls this
/// behind `HIPFIRE_FORWARD_LOWERED` (default off), oracle-validated byte-identical
/// vs the hand `execute_steps` path before the default is flipped per arch.
pub fn run_layer_program<B: ForwardBindings>(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    program: &LayerProgram,
    bindings: &mut B,
) -> Result<(), DispatchError> {
    for op in program {
        dispatch_super_op(gpu, ctx, op, bindings)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fk() -> KernelKey { KernelKey::FusedQkvHfq4G256 }

    #[test]
    fn lower_walk_collapses_fused_and_passes_through_unfused() {
        // 6 steps: [0..4) fuse (span 4) to one Proj; step 4 unfused Gemv→Proj;
        // step 5 unfused GemvResidual→ResidualGemv.
        let prog = lower_walk(
            6,
            |i| if i == 5 { SuperOpKind::ResidualGemv } else { SuperOpKind::Proj },
            |pos| if pos == 0 { Some((fk(), 4)) } else { None },
        );
        assert_eq!(prog.len(), 3);
        assert_eq!(prog[0].kind, SuperOpKind::Proj);
        assert_eq!(prog[0].binding.key, Some(fk())); // fused → frozen key
        assert_eq!(prog[1].kind, SuperOpKind::Proj);
        assert_eq!(prog[1].binding.key, None); // unfused single
        assert_eq!(prog[2].kind, SuperOpKind::ResidualGemv);
        assert_eq!(prog[2].binding.key, None);
    }

    #[test]
    fn lower_walk_all_unfused_keeps_each_step() {
        let prog = lower_walk(3, |_| SuperOpKind::Norm, |_| None);
        assert_eq!(prog.len(), 3);
        assert!(prog.iter().all(|op| op.binding.key.is_none()));
    }

    #[test]
    fn lower_walk_single_cluster_spans_to_end() {
        let prog = lower_walk(4, |_| SuperOpKind::Proj, |pos| {
            if pos == 0 { Some((fk(), 4)) } else { None }
        });
        assert_eq!(prog.len(), 1);
        assert_eq!(prog[0].binding.key, Some(fk()));
    }

    #[test]
    fn lower_walk_zero_span_does_not_stall() {
        // A defensive (key, 0) must not infinite-loop: falls to single-step.
        let prog = lower_walk(2, |_| SuperOpKind::Proj, |_| Some((fk(), 0)));
        assert_eq!(prog.len(), 2);
        assert!(prog.iter().all(|op| op.binding.key.is_none()));
    }
}
