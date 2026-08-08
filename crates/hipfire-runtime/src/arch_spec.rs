//! Shared dense-transformer decode forward (N5 Phase B).
//!
//! A plain dense transformer layer is the same op sequence across arches —
//! rmsnorm-rotate + QKV, optional attention bias, optional qk-norm, RoPE,
//! attention, o_proj+residual, ffn rmsnorm-rotate + gate/up, SwiGLU,
//! down+residual. llama and qwen2 hand-rolled byte-identical copies of it
//! (qwen2 wrapped in the `SuperOp` interpreter, llama inline). This module
//! factors that body into one [`dense_forward`] driver parameterized by a few
//! config-derived [`DenseKnobs`], with the one genuinely non-shared piece — the
//! KV-cache write + attention kernel family (llama's 7-tier KV ladder vs
//! qwen2's flash/gqa selector) — left to each arch via [`DenseArch::attend`].
//!
//! This is the "ArchSpec" authoring surface (greenfield BET 2), scoped to its
//! load/forward-time-durable core. Per the N4 review the design's `config:`
//! rows are superseded by serde `RawConfig + finalize`, so there is no config
//! schema here — each arch builds its own `Config` and derives `DenseKnobs`.
//!
//! Static dispatch only: [`dense_forward`] is generic over the concrete
//! `A: DenseArch`, so the per-token call graph stays fully inlinable (no
//! per-token `dyn`, per the forward-static rule in `arch.rs`). The driver feeds
//! the already-static `hipfire_dispatch::execute_steps`; it is not a runtime
//! op-interpreter.

use hip_bridge::{DeviceBuffer, HipResult};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::{execute_steps_mesh, GemvInput, Step};
use hipfire_dispatch::types::RotationPlan;
use hipfire_hardware::DeviceMesh;
use rdna_compute::{Gpu, GpuTensor};

/// Config-derived scalars that parameterize the shared dense forward. Built once
/// per forward from each arch's finalized `Config`.
pub struct DenseKnobs {
    /// Add q/k/v projection bias (qwen2: true; llama: false).
    pub attn_bias: bool,
    /// Apply per-head Q/K RMSNorm before RoPE (Qwen3-style; llama/qwen2: false).
    pub qk_norm: bool,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
}

/// Per-forward borrow of the shared decode scratch buffers. `pos_buf` is a raw
/// `DeviceBuffer` (matches both arches' scratch layout).
pub struct DenseScratch<'a> {
    pub x: &'a GpuTensor,
    pub tmp: &'a GpuTensor,
    pub x_rot: &'a GpuTensor,
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub attn_out: &'a GpuTensor,
    pub o: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub ffn_hidden: &'a GpuTensor,
    pub ffn_out: &'a GpuTensor,
    pub pos_buf: &'a DeviceBuffer,
}

/// Per-layer borrow of one decoder layer's weights + its derived rotation plans.
/// Bias/qk-norm tensors are `Option` (present only when the matching knob is on).
pub struct DenseLayer<'a> {
    pub attn_norm: &'a GpuTensor,
    pub ffn_norm: &'a GpuTensor,
    pub wq: WeightRef<'a>,
    pub wk: WeightRef<'a>,
    pub wv: WeightRef<'a>,
    pub wo: WeightRef<'a>,
    pub w_gate: WeightRef<'a>,
    pub w_up: WeightRef<'a>,
    pub w_down: WeightRef<'a>,
    pub wq_bias: Option<&'a GpuTensor>,
    pub wk_bias: Option<&'a GpuTensor>,
    pub wv_bias: Option<&'a GpuTensor>,
    pub q_norm: Option<&'a GpuTensor>,
    pub k_norm: Option<&'a GpuTensor>,
    pub qkv_rot: RotationPlan,
    pub ffn_rot: RotationPlan,
    pub qkv_awq: Option<&'a GpuTensor>,
    pub ffn_awq: Option<&'a GpuTensor>,
    /// Activation dim fed to `RmsnormAutomatic` (the projection's `k`).
    pub qkv_k: usize,
    pub ffn_k: usize,
}

/// A dense transformer arch expressed for the shared [`dense_forward`] driver.
/// Implementors are thin per-forward borrow wrappers over the arch's weights +
/// scratch + KV cache + config.
pub trait DenseArch {
    fn n_layers(&self) -> usize;
    fn knobs(&self) -> &DenseKnobs;
    fn scratch(&self) -> DenseScratch<'_>;
    fn layer(&self, l: usize) -> DenseLayer<'_>;
    /// KV-cache write + single-token attention for layer `l`. By this point q/k/v
    /// are projected, biased, qk-normed and RoPE'd into the shared scratch; write
    /// the attention result into `attn_out`. This is the one op that does NOT
    /// unify across arches (different KV layouts + attention kernel families).
    fn attend(&self, gpu: &mut Gpu, l: usize) -> HipResult<()>;
    /// Optional data-returning companion to [`attend`]: build the
    /// `(KvTierPlan, AttnParams)` for layer `l` so `dense_forward` can emit a
    /// first-class `Step::Attend` in one contiguous step list. Default `None` →
    /// the caller keeps using the side-effecting `attend`. Only arches whose
    /// attention is a `KvTierPlan` family (llama) override this; bespoke-attention
    /// arches (qwen2 GQA-flash) leave it `None`.
    fn attend_plan(
        &self,
        _l: usize,
    ) -> HipResult<
        Option<(
            hipfire_dispatch::families::kv_tier::KvTierPlan,
            hipfire_dispatch::families::attention::AttnParams<'_>,
        )>,
    > {
        Ok(None)
    }
}

#[inline]
fn herr(e: impl std::fmt::Display) -> hip_bridge::HipError {
    hip_bridge::HipError::new(0, &e.to_string())
}

/// Shared dense-transformer decode forward for one token. Runs the per-layer op
/// sequence; the caller does embedding (before) and final norm + lm_head +
/// sampling (after), since those buffers/dtypes differ per arch.
pub fn dense_forward<A: DenseArch>(gpu: &mut Gpu, ctx: &DispatchCtx, arch: &A) -> HipResult<()> {
    let k = arch.knobs();
    let s = arch.scratch();
    // P-A: single (1×1) device mesh threaded to the dispatch chokepoint. Real
    // mesh resolution replaces this construction in P-B; the executor stays the
    // shared dense spine (execute_steps_mesh).
    let mesh = DeviceMesh::single();

    for l in 0..arch.n_layers() {
        let layer = arch.layer(l);

        // The attention block as one contiguous step list: QKV (fuses to
        // FusedQkv*), bias, qk-norm, RoPE — then, on the `Some` path, the
        // first-class `Step::Attend` + o-proj, so the whole block is one
        // `execute_steps` invocation (future cross-boundary fusion seam).
        // match_prefix slices each fused pattern to its own window, so the
        // QKV3/Gemv fusion still fires inside the longer list.
        let mut steps: Vec<Step> = vec![
            Step::RmsnormAutomatic {
                x: s.x,
                norm_weight: layer.attn_norm,
                x_plain: s.tmp,
                out: s.x_rot,
                awq_scale: layer.qkv_awq,
                k: layer.qkv_k,
                eps: k.norm_eps,
                rotation: layer.qkv_rot,
            },
            Step::Gemv {
                w: &layer.wq,
                input: GemvInput::Prerotated(s.x_rot),
                out: s.q,
            },
            Step::Gemv {
                w: &layer.wk,
                input: GemvInput::Prerotated(s.x_rot),
                out: s.k,
            },
            Step::Gemv {
                w: &layer.wv,
                input: GemvInput::Prerotated(s.x_rot),
                out: s.v,
            },
        ];

        // QKV bias (qwen2).
        if k.attn_bias {
            steps.push(Step::BiasAdd {
                x: s.q,
                bias: layer.wq_bias.expect("attn_bias: wq_bias"),
                dim: k.q_dim,
            });
            steps.push(Step::BiasAdd {
                x: s.k,
                bias: layer.wk_bias.expect("attn_bias: wk_bias"),
                dim: k.kv_dim,
            });
            steps.push(Step::BiasAdd {
                x: s.v,
                bias: layer.wv_bias.expect("attn_bias: wv_bias"),
                dim: k.kv_dim,
            });
        }

        // Per-head Q/K norm (Qwen3-style).
        if k.qk_norm {
            if let Some(qn) = layer.q_norm {
                steps.push(Step::QkNorm {
                    x: s.q,
                    weight: qn,
                    n_groups: k.n_heads,
                    head_dim: k.head_dim,
                    eps: k.norm_eps,
                });
            }
            if let Some(kn) = layer.k_norm {
                steps.push(Step::QkNorm {
                    x: s.k,
                    weight: kn,
                    n_groups: k.n_kv_heads,
                    head_dim: k.head_dim,
                    eps: k.norm_eps,
                });
            }
        }

        // RoPE.
        steps.push(Step::Rope {
            q: s.q,
            k: s.k,
            pos_buf: s.pos_buf,
            n_heads: k.n_heads,
            n_kv_heads: k.n_kv_heads,
            head_dim: k.head_dim,
            theta: k.rope_theta,
        });

        let o_proj = Step::GemvResidual {
            w: &layer.wo,
            input: GemvInput::Raw(s.attn_out),
            residual: s.x,
            out: s.o,
        };
        match arch.attend_plan(l)? {
            Some((plan, attn_io)) => {
                // llama: attention is a first-class step → one contiguous list.
                steps.push(Step::Attend { plan, io: attn_io });
                steps.push(o_proj);
                execute_steps_mesh(&mesh, gpu, ctx, &steps).map_err(herr)?;
            }
            None => {
                // Bespoke-attention arch (qwen2 GQA-flash): keep the split —
                // pre-attend steps, then the side-effecting attend, then o-proj.
                // Identical kernels/order to the pre-seam path.
                execute_steps_mesh(&mesh, gpu, ctx, &steps).map_err(herr)?;
                arch.attend(gpu, l)?;
                execute_steps_mesh(&mesh, gpu, ctx, &[o_proj]).map_err(herr)?;
            }
        }

        // FFN: rmsnorm-rotate + gate/up.
        execute_steps_mesh(
            &mesh,
            gpu,
            ctx,
            &[
                Step::RmsnormAutomatic {
                    x: s.x,
                    norm_weight: layer.ffn_norm,
                    x_plain: s.tmp,
                    out: s.x_rot,
                    awq_scale: layer.ffn_awq,
                    k: layer.ffn_k,
                    eps: k.norm_eps,
                    rotation: layer.ffn_rot,
                },
                Step::Gemv {
                    w: &layer.w_gate,
                    input: GemvInput::Prerotated(s.x_rot),
                    out: s.gate,
                },
                Step::Gemv {
                    w: &layer.w_up,
                    input: GemvInput::Prerotated(s.x_rot),
                    out: s.up,
                },
            ],
        )
        .map_err(herr)?;

        // SwiGLU + down projection + residual.
        gpu.silu_mul_f32(s.gate, s.up, s.ffn_hidden)?;
        execute_steps_mesh(
            &mesh,
            gpu,
            ctx,
            &[Step::GemvResidual {
                w: &layer.w_down,
                input: GemvInput::Raw(s.ffn_hidden),
                residual: s.x,
                out: s.ffn_out,
            }],
        )
        .map_err(herr)?;
    }

    Ok(())
}
