// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait implementation for the LLaMA family.
//!
//! Mirrors PR 8's qwen35 pattern. Bring-up triple (`config_from_hfq`,
//! `load_weights`, `new_state`) goes through the trait so daemon and
//! examples can dispatch by `arch_id` without growing a `match` ladder.
//! Forward passes stay direct `llama::*` calls — the hot path doesn't
//! pay dyn dispatch overhead.
//!
//! See `crates/hipfire-arch-qwen35/src/arch.rs` for the canonical
//! design rationale; PR 11 just adds a second implementation of the
//! same trait surface for LLaMA-family bring-up.

use hip_bridge::HipResult;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::{self, HfqFile};
use hipfire_runtime::llama::{ForwardScratch, KvCache, LlamaConfig, LlamaWeights};
use hipfire_runtime::llama::KvCacheExt;
use hipfire_runtime::weight_manifest::{
    DTypeConstraint, FusedQkvLayout, PinTarget, ShardPolicy, StateEntry, StateKind, WeightEntry,
};
use rdna_compute::{DType, Gpu};

use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::{execute_steps, GemvInput, Step};
use hipfire_dispatch::types::dtype_rotation_plan;
use hipfire_runtime::llama::{attention_family, AttnParams, KvTierInputs, KvTierPlan};

/// Type marker for the LLaMA family — covers `arch_id = 0` (LLaMA /
/// Mistral) and `arch_id = 1` (plain Qwen3 / Qwen2). All members of
/// this family share the dense-transformer forward pass owned by
/// [`hipfire_runtime::llama`].
///
/// Qwen3.5 / Qwen3.6 (hybrid DeltaNet, `arch_id = 5`) and Qwen3.5/3.6
/// MoE / Qwen3MoE (`arch_id = 6`) are NOT covered by this marker —
/// see [`hipfire_arch_qwen35::Qwen35`] for those.
pub struct Llama;

fn linear_source_constraint() -> DTypeConstraint {
    DTypeConstraint::source_from_sources(vec![
        DType::F32,
        DType::Q4F16G64,
        DType::Q8_0,
        DType::Q4K,
        DType::Q8HFQ,
        DType::HFQ4G256,
        DType::HFQ4G128,
        DType::HFQ6G256,
        DType::HFQ2G256,
        DType::HFQ2G128,
        DType::HFQ3G256,
        DType::HFQ3G128,
        DType::MQ4G256,
        DType::MQ8G256,
        DType::MQ6G256,
        DType::MQ3G256,
        DType::MQ2G256,
        DType::MQ2G256Lloyd,
        DType::MQ2G256LloydU,
        DType::MQ3G256Lloyd,
        DType::HFP4G32,
        DType::MFP4G32,
        DType::MQ4G256Lloyd,
        DType::MQ2G256GL,
        DType::MQ3G256GL,
        DType::TQ2G128,
        DType::BQ1G128,
        DType::MQ4G256V2,
        DType::MQ4CG256,
        DType::MQ6G256V2,
        DType::MQ5G256V2,
        DType::MQ3G256V2,
        DType::MQ2G256V2,
    ])
}

fn embedding_source_constraint() -> DTypeConstraint {
    DTypeConstraint::source_from_sources(vec![
        DType::F32,
        DType::Q8_0,
        DType::Q4K,
        DType::HFQ4G256,
        DType::HFQ4G128,
    ])
}

fn norm_source_constraint() -> DTypeConstraint {
    DTypeConstraint::source_exact(DType::F32)
}

impl Architecture for Llama {
    type Weights = LlamaWeights;
    type State = ForwardScratch;
    type Config = LlamaConfig;

    fn arch_id() -> u32 {
        // `arch_id = 0` is the canonical LLaMA-family marker. The
        // actual id loaded at runtime is on `HfqFile::arch_id` and may
        // differ for plain Qwen3/Qwen2; config parsing resolves that.
        0
    }

    fn name() -> &'static str {
        "llama"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        hfq::config_from_hfq(hfq)
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        hfq::load_weights_hfq(hfq, cfg, gpu)
            .map_err(|e| format!("llama: load_weights_hfq failed: {e:?}"))
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        ForwardScratch::new(gpu, cfg)
            .map_err(|e| format!("llama: ForwardScratch::new failed: {e:?}"))
    }

    // Optional overrides: defaults from `hipfire_runtime::arch` already
    // assume Qwen3.5 family conventions. LLaMA / Mistral / Qwen3 don't
    // emit `<think>` blocks, but the existing policy choices stay unchanged.
}

impl Llama {
    /// Pure dense LLaMA-family weight declaration. Source names remain
    /// logical; carriers translate them to HFQ/safetensors namespaces.
    pub fn weight_manifest(cfg: &LlamaConfig) -> Vec<WeightEntry> {
        use ShardPolicy::*;
        let (dim, hidden, head_dim) = (cfg.dim, cfg.hidden_dim, cfg.head_dim);
        let (heads, kv_heads) = (cfg.n_heads, cfg.n_kv_heads);
        let linear = linear_source_constraint();
        let embedding = embedding_source_constraint();
        let norm = norm_source_constraint();
        let mut manifest = Vec::with_capacity(cfg.n_layers * 11 + 3);
        manifest.push(WeightEntry::model_with_dtype_constraint(
            "token_embd",
            vec![cfg.vocab_size, dim],
            DType::F16,
            embedding,
            Pin(PinTarget::Embed),
        ));
        for layer in 0..cfg.n_layers {
            manifest.push(WeightEntry::layer_with_dtype_constraint(
                "wq",
                layer,
                vec![heads * head_dim, dim],
                DType::F16,
                linear.clone(),
                FusedQkv {
                    q_heads: heads,
                    kv_heads,
                    head_dim,
                    layout: FusedQkvLayout::Qkv,
                },
            ));
            manifest.push(WeightEntry::layer_with_dtype_constraint(
                "wk",
                layer,
                vec![kv_heads * head_dim, dim],
                DType::F16,
                linear.clone(),
                ColumnShard { axis: 0 },
            ));
            manifest.push(WeightEntry::layer_with_dtype_constraint(
                "wv",
                layer,
                vec![kv_heads * head_dim, dim],
                DType::F16,
                linear.clone(),
                ColumnShard { axis: 0 },
            ));
            manifest.push(WeightEntry::layer_with_dtype_constraint(
                "wo",
                layer,
                vec![dim, heads * head_dim],
                DType::F16,
                linear.clone(),
                RowShard { axis: 1 },
            ));
            manifest.push(WeightEntry::layer_with_dtype_constraint(
                "ffn_gate",
                layer,
                vec![hidden, dim],
                DType::F16,
                linear.clone(),
                ColumnShard { axis: 0 },
            ));
            manifest.push(WeightEntry::layer_with_dtype_constraint(
                "ffn_up",
                layer,
                vec![hidden, dim],
                DType::F16,
                linear.clone(),
                ColumnShard { axis: 0 },
            ));
            manifest.push(WeightEntry::layer_with_dtype_constraint(
                "ffn_down",
                layer,
                vec![dim, hidden],
                DType::F16,
                linear.clone(),
                RowShard { axis: 1 },
            ));
            manifest.push(WeightEntry::layer_with_dtype_constraint(
                "attn_norm",
                layer,
                vec![dim],
                DType::F32,
                norm.clone(),
                Replicate,
            ));
            manifest.push(WeightEntry::layer_with_dtype_constraint(
                "ffn_norm",
                layer,
                vec![dim],
                DType::F32,
                norm.clone(),
                Replicate,
            ));
            if cfg.has_qk_norm {
                manifest.push(WeightEntry::layer_with_dtype_constraint(
                    "q_norm",
                    layer,
                    vec![head_dim],
                    DType::F32,
                    norm.clone(),
                    Replicate,
                ));
                manifest.push(WeightEntry::layer_with_dtype_constraint(
                    "k_norm",
                    layer,
                    vec![head_dim],
                    DType::F32,
                    norm.clone(),
                    Replicate,
                ));
            }
        }
        manifest.push(WeightEntry::model_with_dtype_constraint(
            "output_norm",
            vec![dim],
            DType::F32,
            norm,
            Replicate,
        ));
        manifest.push(WeightEntry::model_with_dtype_constraint(
            "lm_head",
            vec![cfg.vocab_size, dim],
            DType::F16,
            linear,
            Pin(PinTarget::Output),
        ));
        manifest
    }

    /// Build the manifest for an HFQ source after source classification.
    ///
    /// A separate `lm_head.weight` is a resident output projection. When the
    /// source omits it, the declaration is a true tie to `token_embd`; the
    /// output placement remains pinned to the final stage while the source
    /// representation contract is copied from the embedding entry.
    pub fn weight_manifest_for_hfq(
        cfg: &LlamaConfig,
        has_separate_lm_head: bool,
    ) -> Vec<WeightEntry> {
        let mut manifest = Self::weight_manifest(cfg);
        if !has_separate_lm_head {
            let embedding_constraint = manifest
                .first()
                .expect("LLaMA manifest always contains token_embd")
                .dtype_constraint
                .clone();
            let output = manifest
                .last_mut()
                .expect("LLaMA manifest always contains lm_head");
            output.dtype_constraint = embedding_constraint;
            output.policy = ShardPolicy::Tied {
                source: "token_embd".into(),
            };
        }
        manifest
    }

    /// Pure state declaration for the full-attention LLaMA family.
    pub fn state_manifest(cfg: &LlamaConfig) -> Vec<StateEntry> {
        (0..cfg.n_layers)
            .map(|layer| StateEntry::new(StateKind::Kv { quant: String::new() }, layer))
            .collect()
    }
}

// ── Dispatch integration ─────────────────────────────────────────
// When `feature = "new-dispatch"` is active, the crate builds with
// hipfire-dispatch and uses its centralized kernel selection tables
// instead of the inline match-on-DType trees in llama.rs.
//
// Migration pattern for each model forward function:
//
//   #[cfg(feature = "new-dispatch")]
//   fn forward(...) -> HipResult<...> {
//       ModelDispatch::new(gpu).forward_scratch_layers(gpu, weights, config, pos, ...)
//   }
//
//   #[cfg(not(feature = "new-dispatch"))]
//   fn forward(...) -> HipResult<...> {
//       llama::forward_scratch_layers(gpu, weights, config, pos, ...)
//   }
//
// The `ModelDispatch` struct (to be created in a follow-up) wraps all
// 6 families: rotation, gemv, gemm, fused_qkv, attention, moe.
// Each family selects kernel variant via (DType, variant, arch_caps),
// and the pipeline runner handles FWHT rotation, AWQ scaling, residual
// fusion automatically.
//
// Phase 3b: forward_scratch_layers body moved inline, forward_dispatch.rs
// eliminated. See `.opencode/plans/2026-05-30-hipfire-dispatch.md` for the full
// design and migration phases.
//
// ── Phase 1: RotationFamily integration ──────────────────────────

impl Llama {
    /// Forward pass — new-dispatch variant when the feature is active.
    ///
    /// All rotation and GEMV dispatch is handled by [`RotationFamily`] and
    /// [`GemvFamily`] through the centralized dispatch tables. KV cache,
    /// attention, and sampling remain unchanged from the legacy path.
    pub fn forward_scratch_layers(
        gpu: &mut Gpu,
        weights: &LlamaWeights,
        config: &LlamaConfig,
        pos: usize,
        kv_cache: &mut KvCache,
        scratch: &ForwardScratch,
        temperature: f32,
        top_p: f32,
        rng_state: u32,
        repeat_window: usize,
        repeat_penalty: f32,
    ) -> HipResult<(u32, u32)> {
        let ctx = DispatchCtx::new(gpu);

        let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim;
        let _kv_dim = n_kv_heads * head_dim; // legacy: only used by non-dispatch paths

        for layer_idx in 0..config.n_layers {
            let layer = &weights.layers[layer_idx];

            // ── Attention QKV path ──────────────────────────────
            // Single dynamic sequence via execute_steps — no model-side dtype
            // branching. The interpreter selects FusedQkvQ4K / fused-MQ / per-op.
            // This also fixes the F1 rmsnorm bug: RmsnormAutomatic always
            // normalizes, including the Q4K branch (which previously skipped it).
            let qkv_rot = dtype_rotation_plan(layer.wq.gpu_dtype);
            let wrq = layer.wq.dispatch_ref();
            let wrk = layer.wk.dispatch_ref();
            let wrv = layer.wv.dispatch_ref();
            execute_steps(
                gpu,
                &ctx,
                &[
                    Step::RmsnormAutomatic {
                        x: &scratch.x,
                        norm_weight: &layer.attn_norm,
                        x_plain: &scratch.tmp,
                        out: &scratch.x_rot,
                        awq_scale: layer.wq.awq_scale.as_ref(),
                        k: layer.wq.k,
                        eps: config.norm_eps,
                        rotation: qkv_rot,
                    },
                    Step::Gemv {
                        w: &wrq,
                        input: GemvInput::Prerotated(&scratch.x_rot),
                        out: &scratch.q,
                    },
                    Step::Gemv {
                        w: &wrk,
                        input: GemvInput::Prerotated(&scratch.x_rot),
                        out: &scratch.k,
                    },
                    Step::Gemv {
                        w: &wrv,
                        input: GemvInput::Prerotated(&scratch.x_rot),
                        out: &scratch.v,
                    },
                ],
            )?;

            // ── QK norm (optional per config) ───────────────────
            if config.has_qk_norm {
                if let Some(ref qn) = layer.q_norm {
                    gpu.rmsnorm_batched(
                        &scratch.q,
                        qn,
                        &scratch.q,
                        n_heads,
                        head_dim,
                        config.norm_eps,
                    )?;
                }
                if let Some(ref kn) = layer.k_norm {
                    gpu.rmsnorm_batched(
                        &scratch.k,
                        kn,
                        &scratch.k,
                        n_kv_heads,
                        head_dim,
                        config.norm_eps,
                    )?;
                }
            }

            // ── RoPE ────────────────────────────────────────────
            gpu.rope_f32(
                &scratch.q,
                &scratch.k,
                &scratch.pos_buf,
                n_heads,
                n_kv_heads,
                head_dim,
                config.rope_freq_base,
            )?;

            // ── KV cache write + attention (dispatched) ────────
            {
                let ctx = DispatchCtx::new(gpu);
                let family = attention_family();
                let ti = kv_cache.tier_inputs();
                let (k_scales, v_scales) = if ti.quant_hfq8 {
                    (
                        Some(&kv_cache.k_scales[layer_idx]),
                        Some(&kv_cache.v_scales[layer_idx]),
                    )
                } else {
                    (None, None)
                };
                let plan = KvTierPlan::derive(KvTierInputs {
                    pos,
                    flash_mode: 0,
                    capture_mode: gpu.graphs.capture_mode,
                    batch_size: 1,
                    is_tree: false,
                    is_boundary: false,
                    q8_windowed: false,
                    window: 0,
                    ..ti
                })
                .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                let io = AttnParams {
                    q: &scratch.q,
                    k: &scratch.k,
                    v: &scratch.v,
                    k_cache: &kv_cache.k_gpu[layer_idx],
                    v_cache: &kv_cache.v_gpu[layer_idx],
                    k_scales,
                    v_scales,
                    pos_buf: &scratch.pos_buf,
                    pos,
                    positions: None,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    physical_cap: kv_cache.physical_cap,
                    batch_size: 1,
                    max_ctx_len: 0,
                    flash_partials: Some(&scratch.attn_partials),
                    givens_cos: kv_cache.givens_cos.as_ref(),
                    givens_sin: kv_cache.givens_sin.as_ref(),
                    tree_bias: None,
                    block_start: 0,
                    block_cols: 0,
                    output_gate: None,
                    output: &scratch.attn_out,
                };
                family
                    .run_attention(&ctx, gpu, &plan, &io)
                    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
            }

            // ── Attention output projection + residual ─────────
            let wro = layer.wo.dispatch_ref();
            execute_steps(
                gpu,
                &ctx,
                &[Step::GemvResidual {
                    w: &wro,
                    input: GemvInput::Raw(&scratch.attn_out),
                    residual: &scratch.x,
                    out: &scratch.o,
                }],
            )?;

            // ── FFN path ────────────────────────────────────────
            // Single dynamic sequence via execute_steps — no model-side dtype
            // branching. Fixes the F1 rmsnorm bug for Q4K gate+up.
            let ffn_rot = dtype_rotation_plan(layer.w_gate.gpu_dtype);
            let wrg = layer.w_gate.dispatch_ref();
            let wru = layer.w_up.dispatch_ref();
            execute_steps(
                gpu,
                &ctx,
                &[
                    Step::RmsnormAutomatic {
                        x: &scratch.x,
                        norm_weight: &layer.ffn_norm,
                        x_plain: &scratch.tmp,
                        out: &scratch.x_rot,
                        awq_scale: layer.w_gate.awq_scale.as_ref(),
                        k: layer.w_gate.k,
                        eps: config.norm_eps,
                        rotation: ffn_rot,
                    },
                    Step::Gemv {
                        w: &wrg,
                        input: GemvInput::Prerotated(&scratch.x_rot),
                        out: &scratch.gate,
                    },
                    Step::Gemv {
                        w: &wru,
                        input: GemvInput::Prerotated(&scratch.x_rot),
                        out: &scratch.up,
                    },
                ],
            )?;

            // ── SwiGLU + down projection + residual ─────────────
            gpu.silu_mul_f32(&scratch.gate, &scratch.up, &scratch.ffn_hidden)?;
            let wrd = layer.w_down.dispatch_ref();
            execute_steps(
                gpu,
                &ctx,
                &[Step::GemvResidual {
                    w: &wrd,
                    input: GemvInput::Raw(&scratch.ffn_hidden),
                    residual: &scratch.x,
                    out: &scratch.ffn_out,
                }],
            )?;
        }

        // ── Final norm + logits + sampling ──────────────────────
        gpu.rmsnorm_f32(
            &scratch.x,
            &weights.output_norm,
            &scratch.tmp,
            config.norm_eps,
        )?;
        let wr_out = weights.output.dispatch_ref();
        execute_steps(
            gpu,
            &ctx,
            &[Step::Gemv {
                w: &wr_out,
                input: GemvInput::Raw(&scratch.tmp),
                out: &scratch.logits,
            }],
        )?;

        gpu.sample_top_p(
            &scratch.logits,
            &scratch.sample_buf,
            &scratch.repeat_buf,
            config.vocab_size,
            temperature,
            top_p,
            rng_state,
            repeat_window,
            repeat_penalty,
        )
    }
}
