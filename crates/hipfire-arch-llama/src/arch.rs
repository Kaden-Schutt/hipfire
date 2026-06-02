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
use rdna_compute::Gpu;

use rdna_compute::DType;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::{GemvFamily, GemvParams};
use hipfire_dispatch::families::rotation::{RotationFamily, RotationParams};
use hipfire_dispatch::types::{dtype_needs_rotation, GemvVariant, RotationVariant};

/// Type marker for the LLaMA family — covers `arch_id = 0` (LLaMA /
/// Mistral) and `arch_id = 1` (plain Qwen3 / Qwen2). All members of
/// this family share the dense-transformer forward pass owned by
/// [`hipfire_runtime::llama`].
///
/// Qwen3.5 / Qwen3.6 (hybrid DeltaNet, `arch_id = 5`) and Qwen3.5/3.6
/// MoE / Qwen3MoE (`arch_id = 6`) are NOT covered by this marker —
/// see [`hipfire_arch_qwen35::Qwen35`] for those.
pub struct Llama;

impl Architecture for Llama {
    type Weights = LlamaWeights;
    type State = ForwardScratch;
    type Config = LlamaConfig;

    fn arch_id() -> u32 {
        // `arch_id = 0` is the canonical LLaMA-family marker. The
        // actual arch_id loaded at runtime is on `HfqFile::arch_id`
        // and is either 0 (LLaMA / Mistral) or 1 (plain Qwen3 /
        // Qwen2); both share this trait impl. The qwen3-norm flag
        // is read off the HFQ metadata inside `config_from_hfq`,
        // so the bring-up triple does not need a separate marker
        // type per arch_id.
        0
    }

    fn name() -> &'static str {
        "llama"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        // `hfq::config_from_hfq` is the LLaMA-family HFQ metadata
        // parser — emits a `LlamaConfig` with the appropriate
        // `ModelArch` (Llama vs Qwen3) tag. It lives in the runtime
        // crate because the qwen35 hybrid path's pflash drafter also
        // calls it via `hfq::config_from_hfq` for its "Plain"
        // variant. See arch-llama/src/lib.rs for the colocation
        // rationale.
        hfq::config_from_hfq(hfq)
            .ok_or_else(|| "llama: failed to parse config from HFQ metadata".to_string())
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        // `hfq::load_weights_hfq` is the LLaMA-family HFQ tensor
        // loader. Same colocation reasoning as `config_from_hfq`.
        hfq::load_weights_hfq(hfq, cfg, gpu)
            .map_err(|e| format!("llama: load_weights_hfq failed: {e:?}"))
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        // The LLaMA-arch "state" is the `ForwardScratch` — persistent
        // GPU scratch buffers reused across decode steps. There is no
        // separate recurrent state (LLaMA is full-attention only).
        ForwardScratch::new(gpu, cfg)
            .map_err(|e| format!("llama: ForwardScratch::new failed: {e:?}"))
    }

    // Optional overrides: defaults from `hipfire_runtime::arch` already
    // assume Qwen3.5 family conventions. LLaMA / Mistral / Qwen3 don't
    // emit `<think>` blocks, but PR 11 keeps the override surface
    // empty here on purpose — the daemon's existing per-`arch_id`
    // policy choices stay unchanged. Future PRs that consolidate
    // policy through the trait can populate these (LLaMA: no
    // strip_think, no Qwen-specific blocked tokens).
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
        let rotation = RotationFamily::new();
        let gemv = GemvFamily::new();

        let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim;
        let kv_dim = n_kv_heads * head_dim;

        for layer_idx in 0..config.n_layers {
            let layer = &weights.layers[layer_idx];

            // ── Attention QKV path ──────────────────────────────
            if layer.wq.gpu_dtype == DType::Q4K && layer.wk.gpu_dtype == DType::Q4K {
                gpu.fused_qkv_q4k(
                    &layer.wq.buf, &layer.wk.buf, &layer.wv.buf,
                    &scratch.tmp, &scratch.q, &scratch.k, &scratch.v,
                    layer.wq.m, layer.wk.m, layer.wv.m, layer.wq.k,
                )?;
            } else if dtype_needs_rotation(layer.wq.gpu_dtype) {
                rotation.run(&ctx, gpu, RotationParams {
                    x: &scratch.x,
                    x_up: None,
                    w_norm: Some(&layer.attn_norm),
                    x_plain: &scratch.tmp,
                    x_rot: &scratch.x_rot,
                    awq_scale: layer.wq.awq_scale.as_ref(),
                    k: layer.wq.k,
                    eps: config.norm_eps,
                    batch_size: 1,
                    variant: RotationVariant::WithRmsnorm,
                    givens_pairs: None,
                    givens_theta: None,
                    givens_scales: None,
                    givens_krot: None,
                })?;
                gemv.run_auto(&ctx, gpu, &layer.wq.dispatch_ref(), &scratch.x_rot, &scratch.q)?;
                gemv.run_auto(&ctx, gpu, &layer.wk.dispatch_ref(), &scratch.x_rot, &scratch.k)?;
                gemv.run_auto(&ctx, gpu, &layer.wv.dispatch_ref(), &scratch.x_rot, &scratch.v)?;
            } else {
                gpu.rmsnorm_f32(&scratch.x, &layer.attn_norm, &scratch.tmp, config.norm_eps)?;
                gemv.run_auto(&ctx, gpu, &layer.wq.dispatch_ref(), &scratch.tmp, &scratch.q)?;
                gemv.run_auto(&ctx, gpu, &layer.wk.dispatch_ref(), &scratch.tmp, &scratch.k)?;
                gemv.run_auto(&ctx, gpu, &layer.wv.dispatch_ref(), &scratch.tmp, &scratch.v)?;
            }

            // ── QK norm (optional per config) ───────────────────
            if config.has_qk_norm {
                if let Some(ref qn) = layer.q_norm {
                    gpu.rmsnorm_batched(
                        &scratch.q, qn, &scratch.q, n_heads, head_dim, config.norm_eps,
                    )?;
                }
                if let Some(ref kn) = layer.k_norm {
                    gpu.rmsnorm_batched(
                        &scratch.k, kn, &scratch.k, n_kv_heads, head_dim, config.norm_eps,
                    )?;
                }
            }

            // ── RoPE ────────────────────────────────────────────
            gpu.rope_f32(
                &scratch.q, &scratch.k, &scratch.pos_buf,
                n_heads, n_kv_heads, head_dim, config.rope_freq_base,
            )?;

            // ── KV cache write + attention ──────────────────────
            if kv_cache.quant_hfq4 {
                gpu.kv_cache_write_hfq4(
                    &kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf,
                    n_kv_heads, head_dim,
                )?;
                gpu.kv_cache_write_hfq4(
                    &kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf,
                    n_kv_heads, head_dim,
                )?;
                gpu.attention_hfq4_kv(
                    &scratch.q,
                    &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                    &scratch.attn_out, &scratch.pos_buf, pos + 1,
                    n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
                )?;
            } else if kv_cache.quantized
                && !kv_cache.k_scales.is_empty()
                && !kv_cache.quant_int8
                && !kv_cache.quant_q8
            {
                gpu.kv_cache_write_hfq8(
                    &kv_cache.k_gpu[layer_idx], &kv_cache.k_scales[layer_idx],
                    &scratch.k, &scratch.pos_buf, n_kv_heads, head_dim,
                )?;
                gpu.kv_cache_write_hfq8(
                    &kv_cache.v_gpu[layer_idx], &kv_cache.v_scales[layer_idx],
                    &scratch.v, &scratch.pos_buf, n_kv_heads, head_dim,
                )?;
                gpu.attention_hfq8_kv(
                    &scratch.q,
                    &kv_cache.k_gpu[layer_idx], &kv_cache.k_scales[layer_idx],
                    &kv_cache.v_gpu[layer_idx], &kv_cache.v_scales[layer_idx],
                    &scratch.attn_out, &scratch.pos_buf, pos + 1,
                    n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
                )?;
            } else if kv_cache.quant_int8 {
                gpu.kv_cache_write_int8c_f16(
                    &kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf,
                    n_kv_heads, head_dim,
                )?;
                gpu.kv_cache_write_int8c_f16(
                    &kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf,
                    n_kv_heads, head_dim,
                )?;
                gpu.attention_int8c_f16_kv(
                    &scratch.q,
                    &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                    &scratch.attn_out, &scratch.pos_buf, pos + 1,
                    n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
                )?;
            } else if kv_cache.quantized && kv_cache.quant_q8 {
                gpu.kv_cache_write_q8_0(
                    &kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf,
                    n_kv_heads, head_dim,
                )?;
                gpu.kv_cache_write_q8_0(
                    &kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf,
                    n_kv_heads, head_dim,
                )?;
                gpu.attention_q8_0_kv(
                    &scratch.q,
                    &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                    &scratch.attn_out, &scratch.pos_buf, pos + 1,
                    n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
                )?;
            } else if kv_cache.quantized {
                gpu.kv_cache_write_q4(
                    &kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf,
                    n_kv_heads, head_dim,
                )?;
                gpu.kv_cache_write_q4(
                    &kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf,
                    n_kv_heads, head_dim,
                )?;
                gpu.attention_q4kv(
                    &scratch.q,
                    &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                    &scratch.attn_out, &scratch.pos_buf, pos + 1,
                    n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
                )?;
            } else {
                gpu.kv_cache_write(
                    &kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf, kv_dim,
                )?;
                gpu.kv_cache_write(
                    &kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf, kv_dim,
                )?;
                gpu.attention_f32(
                    &scratch.q,
                    &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                    &scratch.attn_out, &scratch.pos_buf, pos + 1,
                    n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
                )?;
            }

            // ── Attention output projection + residual ─────────
            gemv.run_auto(&ctx, gpu, &layer.wo.dispatch_ref(), &scratch.attn_out, &scratch.o)?;
            gpu.add_inplace_f32(&scratch.x, &scratch.o)?;

            // ── FFN path ────────────────────────────────────────
            if layer.w_gate.gpu_dtype == DType::Q4K && layer.w_up.gpu_dtype == DType::Q4K {
                gpu.fused_gate_up_q4k(
                    &layer.w_gate.buf, &layer.w_up.buf,
                    &scratch.tmp, &scratch.gate, &scratch.up,
                    layer.w_gate.m, layer.w_up.m, layer.w_gate.k,
                )?;
            } else if dtype_needs_rotation(layer.w_gate.gpu_dtype) {
                rotation.run(&ctx, gpu, RotationParams {
                    x: &scratch.x,
                    x_up: None,
                    w_norm: Some(&layer.ffn_norm),
                    x_plain: &scratch.tmp,
                    x_rot: &scratch.x_rot,
                    awq_scale: layer.w_gate.awq_scale.as_ref(),
                    k: layer.w_gate.k,
                    eps: config.norm_eps,
                    batch_size: 1,
                    variant: RotationVariant::WithRmsnorm,
                    givens_pairs: None,
                    givens_theta: None,
                    givens_scales: None,
                    givens_krot: None,
                })?;
                gemv.run_auto(&ctx, gpu, &layer.w_gate.dispatch_ref(), &scratch.x_rot, &scratch.gate)?;
                gemv.run_auto(&ctx, gpu, &layer.w_up.dispatch_ref(), &scratch.x_rot, &scratch.up)?;
            } else {
                gpu.rmsnorm_f32(&scratch.x, &layer.ffn_norm, &scratch.tmp, config.norm_eps)?;
                gemv.run_auto(&ctx, gpu, &layer.w_gate.dispatch_ref(), &scratch.tmp, &scratch.gate)?;
                gemv.run_auto(&ctx, gpu, &layer.w_up.dispatch_ref(), &scratch.tmp, &scratch.up)?;
            }

            // ── SwiGLU + down projection + residual ─────────────
            gpu.silu_mul_f32(&scratch.gate, &scratch.up, &scratch.ffn_hidden)?;
            gemv.run(&ctx, gpu, &GemvParams {
                w: &layer.w_down.dispatch_ref(),
                x: &scratch.ffn_hidden,
                y: &scratch.ffn_out,
                variant: GemvVariant::WithResidual,
                residual: Some(&scratch.x),
                gate: None,
                up: None,
            })?;
        }

        // ── Final norm + logits + sampling ──────────────────────
        gpu.rmsnorm_f32(&scratch.x, &weights.output_norm, &scratch.tmp, config.norm_eps)?;
        gemv.run_auto(&ctx, gpu, &weights.output.dispatch_ref(), &scratch.tmp, &scratch.logits)?;

        gpu.sample_top_p(
            &scratch.logits, &scratch.sample_buf, &scratch.repeat_buf,
            config.vocab_size, temperature, top_p, rng_state,
            repeat_window, repeat_penalty,
        )
    }

}
