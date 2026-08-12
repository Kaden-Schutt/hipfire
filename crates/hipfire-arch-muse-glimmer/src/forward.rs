// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Forward pass: gated attention, split-eps sandwich RMSNorm, silu SwiGLU.
//!
//! Per-token pipeline (see `lib.rs`):
//!   x = embed(token)                 // NO sqrt(dim) scale (only Gemma does)
//!   for each layer:
//!     residual = x
//!     n1 = rmsnorm(x, input_layernorm, eps=1e-5) -> tmp
//!     q = q_proj(n1); k = k_proj(n1); v = v_proj(n1); gate = gate_proj(n1)
//!     q = rmsnorm_batched(q, ones, head_dim, 1e-5) // scale-less QK-norm
//!     k = rmsnorm_batched(k, ones, head_dim, 1e-5)
//!     q *= qk_scale_factor (3.87)   // Do NOT pre-scale by sqrt(head_dim)
//!     RoPE only if layer_rope_theta != 0 (copy cohere2moe shape)
//!     kv write + attention_q8_0_kv_swa(window=2048 sliding / 0 full)
//!     attn_out *= sigmoid(gate) via gpu.sigmoid_mul_f32  BEFORE o_proj
//!     tmp = o_proj(attn_out); tmp = rmsnorm(tmp, post_attention_ln, 1e-8)
//!     x = residual + tmp
//!     residual = x
//!     n2 = rmsnorm(x, pre_feedforward_ln, 1e-5) -> tmp
//!     gate_ffn = gate_proj(n2); up = up_proj(n2)
//!     hidden = silu(gate_ffn) * up
//!     ffn_out = down_proj(hidden); ffn_out = rmsnorm(ffn_out, post_ffn_ln, 1e-8)
//!     x = residual + ffn_out
//!   x = rmsnorm(x, final_norm, 1e-5) -> tmp
//!   logits = lm_head(tmp); logits *= output_multiplier; logits = softcap(logits, 20)

use crate::config::{GlimmerConfig, GlimmerLayerType};
use crate::glimmer::{GLIMMER_MAX_SPEC_BLOCK, GlimmerState, GlimmerWeights};
use hipfire_runtime::llama::{
    fused_rmsnorm_rotate_for_mq, fused_rmsnorm_rotate_mq_batched_for,
    rotate_x_mq_batched_for, weight_gemv, weight_gemv_prerotated, EmbeddingFormat, WeightTensor,
};
use rdna_compute::{DType, Gpu, GpuTensor};

// ─────────────────── Prefill dispatch helpers ───────────────────
// Route Glimmer's chunked batched PREFILL projections through hipfire-dispatch
// (caller picks KernelKey, no shared selector touched). This is the migration
// qwen35 and gemma4 already did via run_plain_gemm_key / run_residual_gemm_key.
// We add prefill-only helpers beside proj_gemm_batched; DFlash verify keeps
// the old helpers unchanged. MQ4 GEMMs always read x_rot (FWHT-rotated), Q8
// reads the unrotated buffer — never convert the unrotated buffer for MQ4.
#[inline]
fn run_prefill_plain_gemm_key(
    gpu: &mut Gpu,
    key: hipfire_dispatch::types::KernelKey,
    w_buf: &GpuTensor,
    w_dtype: DType,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), String> {
    use hipfire_dispatch::context::DispatchCtx;
    use hipfire_dispatch::families::gemm::GemmParams;
    use hipfire_dispatch::families::gemv::WeightRef;
    let ctx = DispatchCtx::new(gpu);
    let w = WeightRef {
        buf: w_buf,
        dtype: w_dtype,
        m,
        k,
        row_stride: k,
        rotation: None,
        awq_scale: None,
    };
    let params = GemmParams {
        w: &w,
        x,
        y,
        batch_size: n,
    };
    hipfire_runtime::llama::gemm_family()
        .run_key(key, &ctx, gpu, &params)
        .map_err(|e| format!("{e:?}"))
}

#[inline]
fn run_prefill_residual_gemm_key(
    gpu: &mut Gpu,
    key: hipfire_dispatch::types::KernelKey,
    w_buf: &GpuTensor,
    w_dtype: DType,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), String> {
    use hipfire_dispatch::context::DispatchCtx;
    use hipfire_dispatch::families::gemm::GemmParams;
    use hipfire_dispatch::families::gemv::WeightRef;
    let ctx = DispatchCtx::new(gpu);
    let w = WeightRef {
        buf: w_buf,
        dtype: w_dtype,
        m,
        k,
        row_stride: k,
        rotation: None,
        awq_scale: None,
    };
    let params = GemmParams {
        w: &w,
        x,
        y,
        batch_size: n,
    };
    hipfire_runtime::llama::gemm_family()
        .run_key(key, &ctx, gpu, &params)
        .map_err(|e| format!("{e:?}"))
}

/// Muse-owned residual GEMM with fallback to the shared dispatch key.
///
/// Tries `gemm_hfq4g256_residual_muse` — a Glimmer-only sibling of the shared
/// batch-tiled kernel that drops tail clamping and affine-collapses the X
/// addressing, freeing ~50 VGPRs and lifting occupancy 5 -> 7 waves/SIMD at
/// bt=12. It is BIT-IDENTICAL to the shared kernel (same BV, same K loop, same
/// WMMA order; only address arithmetic differs), verified `bitdiff=0` across
/// every Glimmer prefill shape, so it cannot change decoded output.
///
/// Only applied where it MEASURED faster on gfx1201 at B=384 — the caller
/// decides, because the win is shape-dependent and inverts on wide-M:
///   down_proj (M=6656)  +18.2%      o_proj (M=6656)   +12.0%
///   gate/up   (M=19968) -16.6%      qkvg   (M=8704)   -20.6%
/// Wide-M shapes already saturate the grid, so extra occupancy buys nothing
/// and the affine recompute costs. Those stay on the shared kernel.
///
/// Returns to the shared key when the batch is not a multiple of 16*bt (ragged
/// tail chunk) or the env opt-out is set.
#[inline]
fn run_prefill_residual_muse_or_key(
    gpu: &mut Gpu,
    key: hipfire_dispatch::types::KernelKey,
    w_buf: &GpuTensor,
    w_dtype: DType,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), String> {
    if muse_gemm_enabled() && matches!(w_dtype, DType::MQ4G256 | DType::HFQ4G256) {
        let bt = if n % 192 == 0 { 12 } else { 0 };
        if bt != 0 {
            let used = gpu
                .gemm_hfq4g256_residual_muse(w_buf, x, y, m, k, n, bt)
                .map_err(|e| format!("muse residual: {e:?}"))?;
            if used {
                return Ok(());
            }
        }
    }
    run_prefill_residual_gemm_key(gpu, key, w_buf, w_dtype, x, y, m, k, n)
}

/// Opt-out for the Muse-owned residual GEMM (`HIPFIRE_GLIMMER_MUSE_GEMM=0`).
/// Default ON. Since the kernel is bit-identical to the shared one, this knob
/// is a perf A/B switch, not a correctness escape hatch.
fn muse_gemm_enabled() -> bool {
    std::env::var("HIPFIRE_GLIMMER_MUSE_GEMM")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(true)
}

#[inline]
fn run_prefill_fused_qkvza_key(
    gpu: &mut Gpu,
    key: hipfire_dispatch::types::KernelKey,
    w_q: &GpuTensor,
    w_k: &GpuTensor,
    w_v: &GpuTensor,
    w_gate: &GpuTensor,
    x: &GpuTensor,
    y_q: &GpuTensor,
    y_k: &GpuTensor,
    y_v: &GpuTensor,
    y_gate: &GpuTensor,
    q_m: usize,
    k_m: usize,
    v_m: usize,
    gate_m: usize,
    k: usize,
    n: usize,
) -> Result<(), String> {
    use hipfire_dispatch::context::DispatchCtx;
    use hipfire_dispatch::families::fused_qkv::FusedQkvParams;
    let ctx = DispatchCtx::new(gpu);
    let params = FusedQkvParams {
        kind: key,
        weights: &[w_q, w_k, w_v, w_gate],
        x,
        outputs: &[y_q, y_k, y_v, y_gate],
        m: &[q_m, k_m, v_m, gate_m],
        k,
        rot_scratch: &[],
        batch_size: Some(n),
    };
    hipfire_runtime::llama::fused_qkv_family()
        .run(&ctx, gpu, &params)
        .map_err(|e| format!("{e:?}"))
}

// ───────────────────── Shared rotation gate ─────────────────────
// HIPFIRE_GLIMMER_SHARED_ROT default ON (=1 or unset), =0 selects old path.
// Mirrors Gemma4's attn_input_qkv and Qwen35's run_fa_layer_body precedent.
fn shared_rot_enabled() -> bool {
    std::env::var("HIPFIRE_GLIMMER_SHARED_ROT").as_deref() != Ok("0")
}

fn batched_lm_head_enabled() -> bool {
    !matches!(
        std::env::var("HIPFIRE_GLIMMER_BATCHED_LM_HEAD").ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

/// Chunk size for batched prefill: 192 for prompts >= 512 tokens, else 256.
/// Overridable via `HIPFIRE_GLIMMER_PREFILL_CHUNK` (clamped to 1-512); the
/// rationale for the split is on the function body below.
///
/// VRAM cost (transient per-chunk scratch):
///   dim=6656, q_dim=4096, kv_dim=256, hidden=19968, n_layers=52
///   B=128: ~ 70 MB, B=256: ~140 MB, B=512: ~280 MB
///   Computed as sum of batched tensors:
///   x/residual/nrm/x_rot/o_out/normed ~ B*dim*6 *4 bytes
///   q/attn_gate/attn_out/o_rot ~ B*q_dim*4 *4
///   gate/up/ffn_hidden/down_rot ~ B*hidden*4 *4
///   plus small k/v/pos. Model weights are 15.5 GB on 16 GB card leaving
///   ~500 MB headroom, so B=256 (140 MB) is safe, B=512 (280 MB) is still
///   within budget but closer to limit; B=128 is more conservative.
///   Both defaults (192 and 256) sit well inside that headroom.
pub fn glimmer_prefill_chunk_size(prompt_len: usize) -> usize {
    if let Some(v) = std::env::var("HIPFIRE_GLIMMER_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        return v.clamp(1, 512);
    }
    // 192 for anything long enough to fill at least two Muse tiles, else 256.
    //
    // The Muse residual sibling requires batch % 192 == 0, so only chunks of
    // exactly 192 (or 384) reach it; a ragged tail silently falls back to the
    // shared kernel. 192 also lands the shared batch-tile heuristic on BT12
    // with exact tiling, which 256 cannot reach (256 % 192 != 0 -> BT8).
    //
    // Measured on gfx1201, median of 3 fresh processes, all byte-identical:
    //    270 tok: 752 -> 743 tok/s  (-1.2%)  <- why short prompts keep 256
    //   1080 tok: 725 -> 783 tok/s  (+8.0%)
    //   4241 tok: 539 -> 569 tok/s  (+5.6%)
    // At 270 the split is one 192 tile plus a 78-token tail, and the extra
    // chunk's fixed cost outweighs the single tile's win. From ~512 tokens up
    // there are at least two full tiles and 192 pulls clearly ahead.
    if prompt_len >= 512 {
        192
    } else {
        256
    }
}

// ───────────────────────────── Decode ─────────────────────────────

/// Decode one token; returns the full logits vector.
/// Diagnostic ablation switches. Bring-up only: each disables ONE architectural
/// feature so a divergence can be bisected across GPUs in parallel. All default
/// OFF (i.e. the feature is ON) — setting the var disables that feature.
fn abl(name: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some("1")
}

pub fn decode_step(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    embed_lookup(cfg, weights, state, gpu, token_id)?;
    decode_step_body(cfg, weights, state, gpu, position)?;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("glimmer: download logits: {e:?}"))
}
/// On-device sampled decode: same pipeline as [`decode_step`] but samples on
/// the GPU and returns `(token, new_rng)` with only an 8-byte D2H.
///
/// This exists so AR decode pays an 8-byte D2H instead of `vocab_size * 4`
/// bytes (~808 KB at 202048) per token. Greedy (`temp <= 1e-6`) routes to
/// `argmax_f32` to stay byte-identical with the previous host-side path.
pub fn decode_step_sampled(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    rng_state: u32,
) -> Result<(u32, u32), String> {
    embed_lookup(cfg, weights, state, gpu, token_id)?;
    decode_step_body(cfg, weights, state, gpu, position)?;
    if temp <= 1e-6 {
        let tok = gpu
            .argmax_f32(&state.logits, cfg.vocab_size)
            .map_err(|e| format!("glimmer: argmax: {e:?}"))?;
        return Ok((tok, rng_state));
    }
    let top_p_eff = if top_p <= 0.0 || top_p > 1.0 {
        1.0
    } else {
        top_p
    };
    gpu.sample_top_p_pf(
        &state.logits,
        &state.sample_out,
        &state.sample_out,
        cfg.vocab_size,
        temp,
        top_p_eff,
        rng_state,
        0,
        1.0,
        0.0,
        0.0,
        top_k,
        None,
    )
    .map_err(|e| format!("glimmer: sample: {e:?}"))
}

fn embed_lookup(
    _cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    token_id: u32,
) -> Result<(), String> {
    let dim = _cfg.dim;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu
            .embedding_lookup_hfq4g256(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("glimmer: embed hfq4g256: {e:?}"))?,
        EmbeddingFormat::HFQ4G128 => gpu
            .embedding_lookup_hfq4g128(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("glimmer: embed hfq4g128: {e:?}"))?,
        EmbeddingFormat::Q8_0 => gpu
            .embedding_lookup_q8(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("glimmer: embed q8: {e:?}"))?,
        EmbeddingFormat::F32 => gpu
            .embedding_lookup(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("glimmer: embed f32: {e:?}"))?,
        EmbeddingFormat::Q4K => {
            return Err("glimmer: Q4K embedding format unsupported".to_string())
        }
    }
    // Scale-less RMSNorm over the embedding.
    //
    // HF wraps the table in `MuseGlimmerTextNormedEmbedding`:
    //     forward(ids) = embed_norm(Embedding::forward(ids))
    // with `MuseGlimmerRMSNorm(eps=config.rms_norm_eps, with_scale=False)`.
    // Upstream explicitly does NOT fold this into the embedding matrix because
    // the DFlash path needs to embed without it, so it runs per lookup here.
    //
    // There is no Gemma-style sqrt(dim) embed_scale in Glimmer — this norm is
    // what takes its place, and omitting it leaves the residual stream at the
    // wrong magnitude for every downstream layer.
    if !abl("HIPFIRE_GLIMMER_NO_EMBED_NORM") {
        gpu.rmsnorm_f32(
            &state.x,
            &state.embed_norm_ones,
            &state.x,
            _cfg.rms_norm_eps,
        )
        .map_err(|e| format!("glimmer: embed_norm: {e:?}"))?;
    }
    Ok(())
}

fn decode_step_body(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    position: u32,
) -> Result<(), String> {
    // Device position scalar (i32) — staged from heap-stable Box.
    state.pos_host[0] = position as i32;
    {
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(state.pos_host.as_ptr() as *const u8, 4) };
        gpu.memcpy_htod_auto(&state.pos_buf, pos_bytes)
            .map_err(|e| format!("glimmer: htod pos: {e:?}"))?;
    }

    for layer_idx in 0..cfg.n_layers {
        let slot = state.kv_slot_for_layer[layer_idx];
        let lw = &weights.layers[layer_idx];
        glimmer_layer_decode(gpu, cfg, lw, layer_idx, slot, state)?;
    }
    state.n_tokens = position as usize + 1;

    // Final RMSNorm -> tmp (rms eps)
    gpu.rmsnorm_f32(&state.x, &weights.final_norm, &state.tmp, cfg.rms_norm_eps)
        .map_err(|e| format!("glimmer: final rmsnorm: {e:?}"))?;

    // LM head (untied) -> logits
    weight_gemv(gpu, &weights.lm_head, &state.tmp, &state.logits)
        .map_err(|e| format!("glimmer: lm_head: {e}"))?;

    // output_multiplier BEFORE softcap (brief RESOLVED)
    if cfg.output_multiplier != 1.0 && !abl("HIPFIRE_GLIMMER_NO_OUTMUL") {
        gpu.scale_f32(&state.logits, cfg.output_multiplier)
            .map_err(|e| format!("glimmer: output_multiplier scale: {e:?}"))?;
    }

    // Final logit softcapping: tanh(x/cap)*cap with cap 20.0
    if cfg.final_logit_softcapping > 0.0 && !abl("HIPFIRE_GLIMMER_NO_SOFTCAP") {
        gpu.logit_softcap_f32(&state.logits, cfg.vocab_size, cfg.final_logit_softcapping)
            .map_err(|e| format!("glimmer: logit softcap: {e:?}"))?;
    }
    Ok(())
}

fn glimmer_layer_decode(
    gpu: &mut Gpu,
    cfg: &GlimmerConfig,
    lw: &crate::glimmer::GlimmerLayerWeights,
    layer_idx: usize,
    kv_slot: usize,
    state: &mut GlimmerState,
) -> Result<(), String> {
    let dim = cfg.dim;
    let head_dim = cfg.head_dim;
    let n_heads = cfg.n_heads;
    let n_kv = cfg.n_kv_heads;
    let rms_eps = cfg.rms_norm_eps;
    let post_eps = cfg.post_norm_eps;
    let dim_bytes = dim * 4;

    // residual = x
    gpu.memcpy_dtod_auto(&state.residual.buf, &state.x.buf, dim_bytes)
        .map_err(|e| format!("glimmer L{layer_idx}: save residual: {e:?}"))?;

    // n1 = input_layernorm(x) -> tmp/tmp_rot
    // Shared FWHT rotation: 4 projections (q/k/v/gate) read the SAME normed
    // input. Each weight_gemv on MQ4G256 internally rotates its input (6656-
    // element FWHT), so the old path rotated the identical vector 4 times.
    // Rotate once via fused_rmsnorm_rotate_for_mq (precedent:
    // crates/hipfire-arch-qwen35/src/qwen35.rs:17515 and
    // crates/hipfire-arch-gemma4/src/forward.rs:192-199 — the latter states
    // "Prerotated GEMVs share tmp_rot (NO re-rotation -- byte-identical math)").
    // Byte-identical when the fused kernel's rmsnorm+FWHT sequence matches
    // the separate rmsnorm_f32 + rotate_x_mq sequence (same per-group
    // butterfly, same sign tables from ensure_mq_signs). Accumulation order
    // in the GEMVs is unchanged (each GEMV still accumulates its own M rows).
    let use_shared_attn = shared_rot_enabled()
        && hipfire_dispatch::types::dtype_rotation_plan(lw.q_proj.gpu_dtype)
            != hipfire_dispatch::types::RotationPlan::None;
    if use_shared_attn {
        // Fused rmsnorm+rotate: x --rmsnorm--> x_rot (FWHT). Returns Some(&x_rot)
        // when rotation was applied, None for non-rotating dtypes (fallback,
        // though we already checked rotation is needed, so this is Some).
        // The plain tmp is not populated in the fused path; weight_gemv_prerotated
        // ignores its `x` arg when `x_rot` is Some (see llama.rs:1221).
        let x_rot_opt = fused_rmsnorm_rotate_for_mq(
            gpu, &lw.q_proj, &state.x, &lw.input_layernorm, &state.tmp, &state.x_rot, rms_eps,
        )
        .map_err(|e| format!("glimmer L{layer_idx}: fused input rmsnorm+rotate: {e:?}"))?;
        let x_rot = x_rot_opt.as_ref().copied();
        // x param is ignored when x_rot is Some; pass &state.x as placeholder
        // (matches Gemma4's weight_gemv_prerotated(gpu, q_proj, x, Some(tmp_rot), q_out)).
        weight_gemv_prerotated(gpu, &lw.q_proj, &state.x, x_rot, &state.q)
            .map_err(|e| format!("glimmer L{layer_idx}: q_proj (prerot): {e:?}"))?;
        weight_gemv_prerotated(gpu, &lw.k_proj, &state.x, x_rot, &state.k)
            .map_err(|e| format!("glimmer L{layer_idx}: k_proj (prerot): {e:?}"))?;
        weight_gemv_prerotated(gpu, &lw.v_proj, &state.x, x_rot, &state.v)
            .map_err(|e| format!("glimmer L{layer_idx}: v_proj (prerot): {e:?}"))?;
        weight_gemv_prerotated(gpu, &lw.attn_gate_proj, &state.x, x_rot, &state.attn_gate)
            .map_err(|e| format!("glimmer L{layer_idx}: attn gate_proj (prerot): {e:?}"))?;
    } else {
        gpu.rmsnorm_f32(&state.x, &lw.input_layernorm, &state.tmp, rms_eps)
            .map_err(|e| format!("glimmer L{layer_idx}: input rmsnorm: {e:?}"))?;
        weight_gemv(gpu, &lw.q_proj, &state.tmp, &state.q)
            .map_err(|e| format!("glimmer L{layer_idx}: q_proj: {e}"))?;
        weight_gemv(gpu, &lw.k_proj, &state.tmp, &state.k)
            .map_err(|e| format!("glimmer L{layer_idx}: k_proj: {e}"))?;
        weight_gemv(gpu, &lw.v_proj, &state.tmp, &state.v)
            .map_err(|e| format!("glimmer L{layer_idx}: v_proj: {e}"))?;
        weight_gemv(gpu, &lw.attn_gate_proj, &state.tmp, &state.attn_gate)
            .map_err(|e| format!("glimmer L{layer_idx}: attn gate_proj: {e}"))?;
    }

    // Scale-less QK-norm (no learned weight tensors; ones-filled weight)
    // Still runs RMSNorm per head, then Q *= qk_scale_factor.
    if !abl("HIPFIRE_GLIMMER_NO_QK_NORM") {
        gpu.rmsnorm_batched(&state.q, &state.qk_norm_ones, &state.q, n_heads, head_dim, rms_eps)
            .map_err(|e| format!("glimmer L{layer_idx}: q_norm: {e:?}"))?;
        gpu.rmsnorm_batched(&state.k, &state.qk_norm_ones, &state.k, n_kv, head_dim, rms_eps)
            .map_err(|e| format!("glimmer L{layer_idx}: k_norm: {e:?}"))?;
    }
    // Do NOT pre-scale by sqrt(head_dim); Glimmer wants kernel 1/sqrt AND 3.87
    if !abl("HIPFIRE_GLIMMER_NO_QK_SCALE") {
        gpu.scale_f32(&state.q, cfg.qk_scale_factor)
            .map_err(|e| format!("glimmer L{layer_idx}: q scale qk_factor: {e:?}"))?;
    }

    // RoPE only on layers whose layer_rope_theta != 0 (copy cohere2moe shape)
    if cfg.has_rope(layer_idx) || abl("HIPFIRE_GLIMMER_ROPE_ALL") {
        let theta = cfg.rope_theta_for(layer_idx);
        // RoPE convention. HF reports rope_type "default" (Llama half-split),
        // which is `rope_f32`. HIPFIRE_GLIMMER_ROPE_INTERLEAVED=1 selects the
        // GPT-J interleaved variant for A/B during bring-up — getting this
        // backwards scrambles attention into plausible-looking noise.
        if abl("HIPFIRE_GLIMMER_ROPE_INTERLEAVED") {
            gpu.rope_interleaved_f32(
                &state.q,
                &state.k,
                &state.pos_buf,
                n_heads,
                n_kv,
                head_dim,
                head_dim, // n_rot = full head_dim (no partial rotation)
                theta,
            )
            .map_err(|e| format!("glimmer L{layer_idx}: rope interleaved: {e:?}"))?;
        } else {
            gpu.rope_f32(
                &state.q,
                &state.k,
                &state.pos_buf,
                n_heads,
                n_kv,
                head_dim,
                theta,
            )
            .map_err(|e| format!("glimmer L{layer_idx}: rope: {e:?}"))?;
        }
    }

    // KV write (Q8) + windowed/full attention via attention_q8_0_kv_swa
    // window = sliding_window on sliding layers (rope), 0 on full (NoPE)
    let window = cfg.window_for(layer_idx);
    let kv = match cfg.layer_types[layer_idx] {
        GlimmerLayerType::Sliding => &mut state.kv_sliding,
        GlimmerLayerType::Full => &mut state.kv_full,
    };
    // Map physical pages up to the position about to be written. No-op on the
    // contiguous backend (fast_mapped_token_capacity returns None) and a cheap
    // early-out once the prefix is already mapped, so it is safe unconditional.
    kv.ensure_mapped_capacity(gpu, state.pos_host[0] as usize + 1)
        .map_err(|e| format!("glimmer L{layer_idx}: kv map: {e:?}"))?;
    gpu.kv_cache_write_q8_0(&kv.k_gpu[kv_slot], &state.k, &state.pos_buf, n_kv, head_dim)
        .map_err(|e| format!("glimmer L{layer_idx}: kv write k: {e:?}"))?;
    gpu.kv_cache_write_q8_0(&kv.v_gpu[kv_slot], &state.v, &state.pos_buf, n_kv, head_dim)
        .map_err(|e| format!("glimmer L{layer_idx}: kv write v: {e:?}"))?;

    // Flash decode. `attention_q8_0_kv_swa` launches grid [n_heads] = 32
    // workgroups, one per head, and each walks the whole key range serially
    // while materializing every score in LDS. On a 64-CU part that leaves half
    // the GPU idle at ANY context and makes cost grow with seq_len. Measured
    // via `hipfire bench --matrix`: Glimmer decode falls 30.3% from ctx 128 to
    // 8192 (32.49 -> 22.66 tok/s) where Qwen3.5-27B on the same GPU and harness
    // loses 4.1% (35.85 -> 34.39) because it decodes through the flash family.
    //
    // The batched flash kernel tiles keys at 128 and puts the tile count in the
    // grid, so at ctx 8192 it launches ~64x more workgroups and reduces
    // partials instead of serializing. Glimmer already uses it for prefill;
    // batch_size=1 is the decode case. Same KV layout, same window semantics
    // (window<=0 = full causal), so the 13 NoPE layers pass window=0.
    //
    // Neither kernel wins outright, so this is a THRESHOLD and not a swap. The
    // 32-workgroup SWA is cheaper when there is too little work to spread; the
    // tiled flash wins once there is. Measured on gfx1201 via
    // `hipfire bench --matrix --tg 64`, medians of 3:
    //
    //     ctx      768   1024   1536   2048   4096   8192
    //     SWA    29.99  29.08  27.53  26.22  24.94  22.66
    //     flash  28.65  28.55  28.38  28.20  27.86  27.20
    //
    // Crossover sits between 1024 and 1536, hence the 1280 default. Picking
    // flash above it takes the ctx-128->8192 decay from -30.3% to -5.2%, which
    // is the -4.1% Qwen3.5-27B shows on the same GPU and harness, and is worth
    // +20.0% at ctx 8192. Byte-identical to the SWA path either side.
    //
    // `HIPFIRE_GLIMMER_FLASH_DECODE` overrides the threshold in tokens; 0
    // disables flash decode entirely.
    let flash_min = std::env::var("HIPFIRE_GLIMMER_FLASH_DECODE")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(1280);
    let seq_len_now = state.pos_host[0] as usize + 1;
    let decode_flash = flash_min > 0 && seq_len_now >= flash_min;
    if decode_flash {
        let n_tiles = (state.max_seq + 127) / 128;
        let need = n_heads * n_tiles * (2 + head_dim) * 64;
        if state.prefill_flash_partials.is_none() {
            let buf = gpu
                .alloc_tensor(&[need], DType::F32)
                .map_err(|e| format!("glimmer L{layer_idx}: decode flash partials: {e:?}"))?;
            state.prefill_flash_partials = Some(buf);
        }
        if state.decode_pos.is_none() {
            let t = gpu
                .alloc_tensor(&[1], DType::F32)
                .map_err(|e| format!("glimmer L{layer_idx}: decode pos tensor: {e:?}"))?;
            state.decode_pos = Some(t);
        }
        {
            let pos_bytes =
                unsafe { std::slice::from_raw_parts(state.pos_host.as_ptr() as *const u8, 4) };
            let t = state.decode_pos.as_ref().unwrap();
            gpu.hip
                .memcpy_htod(&t.buf, pos_bytes)
                .map_err(|e| format!("glimmer L{layer_idx}: decode pos htod: {e:?}"))?;
        }
        let kv = match cfg.layer_types[layer_idx] {
            GlimmerLayerType::Sliding => &state.kv_sliding,
            GlimmerLayerType::Full => &state.kv_full,
        };
        let partials = state.prefill_flash_partials.as_ref().unwrap();
        gpu.attention_flash_q8_0_batched_masked_windowed(
            &state.q,
            &kv.k_gpu[kv_slot],
            &kv.v_gpu[kv_slot],
            &state.attn_out,
            state.decode_pos.as_ref().unwrap(),
            n_heads,
            n_kv,
            head_dim,
            state.max_seq,
            seq_len_now,
            1,
            partials,
            None,
            0,
            0,
            window as i32,
        )
        .map_err(|e| format!("glimmer L{layer_idx}: flash decode: {e:?}"))?;
    } else {
    gpu.attention_q8_0_kv_swa(
        &state.q,
        &kv.k_gpu[kv_slot],
        &kv.v_gpu[kv_slot],
        &state.attn_out,
        &state.pos_buf,
        // seq_len_hint: the CURRENT sequence length, not state.max_seq.
        //
        // This argument never reaches the kernel — the host uses it solely to
        // size the launch (attention.rs: block_size, and
        // shared_mem = (hint + block_size + head_dim) * 4). The kernel derives
        // its own `seq_len = pos_buf[0] + 1` and indexes `scores[t]` by
        // absolute t, so the LDS it actually needs is pos+1 floats.
        //
        // Passing state.max_seq over-allocated LDS by the full cache size and,
        // worse, imposed a hard ceiling: shared_mem crosses the 64 KB LDS limit
        // at max_seq ~16000, so LOADING with a larger max_seq made every decode
        // fail with `hipModuleLaunchKernel: invalid argument` at layer 0 —
        // regardless of how few tokens were actually in the cache. Glimmer
        // declares max_position_embeddings = 131072; measured, max_seq=8192
        // decoded at 32.1 tok/s while max_seq=16384 could not decode at all.
        // Every test had used 8192, so it stayed hidden until `hipfire bench
        // --matrix` (whose default --ctx list reaches 20000) tripped it.
        //
        // Now the ceiling tracks REAL context instead of configured capacity.
        // Contexts past ~16k still need the flash path: `scores` is indexed by
        // absolute t, so this kernel is inherently O(seq_len) in LDS.
        state.pos_host[0] as usize + 1,
        n_heads,
        n_kv,
        head_dim,
        kv.physical_cap,
        window,
    )
    .map_err(|e| format!("glimmer L{layer_idx}: attention swa: {e:?}"))?;
    }

    // Gated attention: attn_out *= sigmoid(attn_gate) BEFORE o_proj
    // Uses gpu.sigmoid_mul_f32 (norm.rs:2006) — do not write a new kernel.
    if !abl("HIPFIRE_GLIMMER_NO_ATTN_GATE") {
        gpu.sigmoid_mul_f32(&state.attn_out, &state.attn_gate)
        .map_err(|e| format!("glimmer L{layer_idx}: sigmoid_mul: {e:?}"))?;
    }

    // o_proj(attn_out) -> tmp
    weight_gemv(gpu, &lw.o_proj, &state.attn_out, &state.tmp)
        .map_err(|e| format!("glimmer L{layer_idx}: o_proj: {e}"))?;

    // Sandwich post-attention norm (post_eps 1e-8) + residual add: x = residual + norm(tmp)
    gpu.rmsnorm_f32(&state.tmp, &lw.post_attention_layernorm, &state.tmp, post_eps)
        .map_err(|e| format!("glimmer L{layer_idx}: post_attn rmsnorm: {e:?}"))?;
    gpu.memcpy_dtod_auto(&state.x.buf, &state.residual.buf, dim_bytes)
        .map_err(|e| format!("glimmer L{layer_idx}: reset x (attn): {e:?}"))?;
    gpu.add_inplace_f32(&state.x, &state.tmp)
        .map_err(|e| format!("glimmer L{layer_idx}: attn residual add: {e:?}"))?;

    // residual = x (FFN stream)
    gpu.memcpy_dtod_auto(&state.residual.buf, &state.x.buf, dim_bytes)
        .map_err(|e| format!("glimmer L{layer_idx}: save ffn residual: {e:?}"))?;

    // ── SwiGLU FFN (silu, not gelu_tanh) ──────────────────────────
    // Shared rotation for gate/up: 2 projections sharing the same normed
    // input. Precedent: qwen35.rs:17799-17800 (w_gate/w_up sharing x_rot)
    // and gemma4's fused FFN path. Saves 1 redundant FWHT per layer.
    let use_shared_ffn = shared_rot_enabled()
        && hipfire_dispatch::types::dtype_rotation_plan(lw.gate_proj.gpu_dtype)
            != hipfire_dispatch::types::RotationPlan::None;
    if use_shared_ffn {
        let x_rot_opt = fused_rmsnorm_rotate_for_mq(
            gpu, &lw.gate_proj, &state.x, &lw.pre_feedforward_layernorm, &state.tmp, &state.x_rot, rms_eps,
        )
        .map_err(|e| format!("glimmer L{layer_idx}: fused pre_ffn rmsnorm+rotate: {e:?}"))?;
        let x_rot = x_rot_opt.as_ref().copied();
        weight_gemv_prerotated(gpu, &lw.gate_proj, &state.x, x_rot, &state.gate_ffn)
            .map_err(|e| format!("glimmer L{layer_idx}: gate_proj (prerot): {e:?}"))?;
        weight_gemv_prerotated(gpu, &lw.up_proj, &state.x, x_rot, &state.up_ffn)
            .map_err(|e| format!("glimmer L{layer_idx}: up_proj (prerot): {e:?}"))?;
    } else {
        gpu.rmsnorm_f32(&state.x, &lw.pre_feedforward_layernorm, &state.tmp, rms_eps)
            .map_err(|e| format!("glimmer L{layer_idx}: pre_ffn rmsnorm: {e:?}"))?;
        weight_gemv(gpu, &lw.gate_proj, &state.tmp, &state.gate_ffn)
            .map_err(|e| format!("glimmer L{layer_idx}: gate_proj: {e}"))?;
        weight_gemv(gpu, &lw.up_proj, &state.tmp, &state.up_ffn)
            .map_err(|e| format!("glimmer L{layer_idx}: up_proj: {e}"))?;
    }
    // silu(gate) * up -> ffn_hidden
    gpu.silu_mul_f32(&state.gate_ffn, &state.up_ffn, &state.ffn_hidden)
        .map_err(|e| format!("glimmer L{layer_idx}: silu_mul: {e:?}"))?;
    weight_gemv(gpu, &lw.down_proj, &state.ffn_hidden, &state.ffn_out)
        .map_err(|e| format!("glimmer L{layer_idx}: down_proj: {e}"))?;

    // Sandwich post-FFN norm (post_eps) + residual add
    gpu.rmsnorm_f32(
        &state.ffn_out,
        &lw.post_feedforward_layernorm,
        &state.tmp,
        post_eps,
    )
    .map_err(|e| format!("glimmer L{layer_idx}: post_ffn rmsnorm: {e:?}"))?;
    gpu.memcpy_dtod_auto(&state.x.buf, &state.residual.buf, dim_bytes)
        .map_err(|e| format!("glimmer L{layer_idx}: reset x (ffn): {e:?}"))?;
    gpu.add_inplace_f32(&state.x, &state.tmp)
        .map_err(|e| format!("glimmer L{layer_idx}: ffn residual add: {e:?}"))?;

    Ok(())
}

// ─── Block-parallel verify (DFlash) ──────────────────────────────────

/// Verify a block of `B` tokens at `position` in parallel-AR fashion,
/// returning per-position argmax. Leaves `state` advanced by `B` (KV written
/// at positions [position .. position+B-1], n_tokens bumped). Caller must
/// handle partial-accept rollback via `rollback_to`.
///
/// This is the B-position analogue of `decode_step`: same embed (WITH
/// embed_norm — this is the AR verify path, not the draft noise path),
/// same layer loop (via `glimmer_layer_decode`), same final norm +
/// lm_head * output_multiplier + softcap. No new kernels — it reuses the
/// per-token `decode_step_body` logic in a loop, but batches the lm_head
/// downloads per position so the caller sees per-position logits.
///
/// For `B==1` this is byte-identical to `decode_step` (modulo the extra
/// per-pos alloc). For `B>1` it is the speculative-verify primitive the
/// drafter's accept rule consumes at `spec.rs:130-183`.
pub fn verify_block(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    block: &[u32],
    position: u32,
) -> Result<Vec<u32>, String> {
    let mut picks = Vec::with_capacity(block.len());
    for (i, &tok) in block.iter().enumerate() {
        let pos = position + i as u32;
        // embed (WITH norm — verify is AR) → layers → final_norm → lm_head → scale → softcap
        let logits = decode_step(cfg, weights, state, gpu, tok, pos)?;
        let pick = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();
        picks.push(pick);
    }
    Ok(picks)
}

/// Roll back `state` to `target_pos` after a partial accept. For the
/// pure-attention Glimmer target (no DeltaNet recurrent state), the only
/// state to truncate is `n_tokens`; KV entries beyond `target_pos` will be
/// overwritten by the next verify at that position (see `spec.rs:312-319`
/// stateless commit_prefix contract). No GPU memset needed.
pub fn rollback_to(state: &mut GlimmerState, target_pos: usize) {
    state.n_tokens = target_pos;
}

/// Raw embedding lookup WITHOUT embed_norm — for DFlash draft noise.
/// Mirrors the daemon's `speculative.rs:3087` raw `embedding_lookup_*` and the
/// HF comment at `/tmp/modeling_muse_glimmer.py:439` ("Dflash needs to embed
/// without the norm"). The AR `embed_lookup` DOES apply `rmsnorm_f32` with
/// `embed_norm_ones`; this one deliberately does not.
pub fn embed_raw(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    token_id: u32,
) -> Result<Vec<f32>, String> {
    // Reuse the same scratch `state.x` but skip the norm.
    let dim = cfg.dim;
    match weights.embd_format {
        hipfire_runtime::llama::EmbeddingFormat::HFQ4G256 => gpu
            .embedding_lookup_hfq4g256(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("glimmer embed_raw hfq4g256: {e:?}"))?,
        hipfire_runtime::llama::EmbeddingFormat::HFQ4G128 => gpu
            .embedding_lookup_hfq4g128(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("glimmer embed_raw hfq4g128: {e:?}"))?,
        hipfire_runtime::llama::EmbeddingFormat::Q8_0 => gpu
            .embedding_lookup_q8(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("glimmer embed_raw q8: {e:?}"))?,
        hipfire_runtime::llama::EmbeddingFormat::F32 => gpu
            .embedding_lookup(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("glimmer embed_raw f32: {e:?}"))?,
        hipfire_runtime::llama::EmbeddingFormat::Q4K => {
            return Err("glimmer embed_raw: Q4K unsupported".into())
        }
    }
    let host = gpu
        .download_f32(&state.x)
        .map_err(|e| format!("glimmer embed_raw download: {e:?}"))?;
    Ok(host[..dim].to_vec())
}

/// Decode one token and capture residual hidden at `capture_layers` (0-based).
/// Captured hidden is appended to `hidden_out` as `capture_layers.len()*dim` f32
/// in the order of `capture_layers` (which must be sorted ascending like
/// [1,13,25,37,49] from dflash_extract_layer_ids(52,5)). The capture is the
/// post-layer residual `x` (same tensor the target's `target_hidden` concat
/// expects per dflash.rs:target_hidden contract). Uses `gpu.download_f32`
/// per captured layer — slower but correct for the speculative path; prefill
/// should use a batched variant if hot.
pub fn decode_step_with_capture(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    capture_layers: &[usize],
    hidden_out: &mut Vec<f32>,
) -> Result<Vec<f32>, String> {
    // Embed + per-layer loop, capturing when layer_idx in capture_layers
    embed_lookup(cfg, weights, state, gpu, token_id)?;
    // Need to track which capture idx we are at
    let mut cap_set = std::collections::HashSet::new();
    for &l in capture_layers { cap_set.insert(l); }
    state.pos_host[0] = position as i32;
    {
        let pos_bytes = unsafe { std::slice::from_raw_parts(state.pos_host.as_ptr() as *const u8, 4) };
        gpu.memcpy_htod_auto(&state.pos_buf, pos_bytes)
            .map_err(|e| format!("glimmer capture: htod pos: {e:?}"))?;
    }
    for layer_idx in 0..cfg.n_layers {
        let slot = state.kv_slot_for_layer[layer_idx];
        let lw = &weights.layers[layer_idx];
        glimmer_layer_decode(gpu, cfg, lw, layer_idx, slot, state)?;
        if cap_set.contains(&layer_idx) {
            // Download current residual stream `state.x` (dim f32)
            let host = gpu.download_f32(&state.x).map_err(|e| format!("glimmer capture download L{}: {e:?}", layer_idx))?;
            hidden_out.extend_from_slice(&host[..cfg.dim]);
        }
    }
    state.n_tokens = position as usize + 1;
    gpu.rmsnorm_f32(&state.x, &weights.final_norm, &state.tmp, cfg.rms_norm_eps)
        .map_err(|e| format!("glimmer capture: final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.tmp, &state.logits)
        .map_err(|e| format!("glimmer capture: lm_head: {e}"))?;
    if cfg.output_multiplier != 1.0 {
        gpu.scale_f32(&state.logits, cfg.output_multiplier)
            .map_err(|e| format!("glimmer capture: outmul: {e:?}"))?;
    }
    if cfg.final_logit_softcapping > 0.0 {
        gpu.logit_softcap_f32(&state.logits, cfg.vocab_size, cfg.final_logit_softcapping)
            .map_err(|e| format!("glimmer capture: softcap: {e:?}"))?;
    }
    gpu.download_f32(&state.logits).map_err(|e| format!("glimmer capture: download logits: {e:?}"))
}


// ─── Batched helpers ─────────────────────────────────────────────────

/// Dispatch for a single projection in the batched verify path.
/// Mirrors `crates/hipfire-arch-gemma4/src/forward.rs::proj_gemm_batched`:
///   Q8_0      → `gemm_q8_0_batched_chunked` (WMMA on gfx12)
///   MQ4G256/HFQ4G256 → rotate + `gemm_hfq4g256_batched_lmhead` (prerotated)
///   MQ6G256   → rotate + `gemm_mq6g256_batched_lmhead`
///   others    → per-row `weight_gemv` fallback (explicit, no approximation)
///
/// `x` is [B,k], `y` is [B,m], `x_rot` is [B,k] scratch for the MQ path.
fn proj_gemm_batched(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    x_rot: &GpuTensor,
    b: usize,
    label: &str,
) -> Result<(), String> {
    match w.gpu_dtype {
        DType::Q8_0 => gpu
            .gemm_q8_0_batched_chunked(&w.buf, x, y, w.m, w.k, b)
            .map_err(|e| format!("glimmer batch {label} (q8): {e:?}")),
        DType::MQ4G256 | DType::HFQ4G256 => {
            rotate_x_mq_batched_for(gpu, w, x, x_rot, w.k, b)
                .map_err(|e| format!("glimmer batch {label} rotate: {e:?}"))?;
            gpu.gemm_hfq4g256_batched_lmhead(&w.buf, x_rot, y, w.m, w.k, b)
                .map_err(|e| format!("glimmer batch {label} (mq4): {e:?}"))
        }
        DType::MQ6G256 => {
            rotate_x_mq_batched_for(gpu, w, x, x_rot, w.k, b)
                .map_err(|e| format!("glimmer batch {label} rotate: {e:?}"))?;
            gpu.gemm_mq6g256_batched_lmhead(&w.buf, x_rot, y, w.m, w.k, b)
                .map_err(|e| format!("glimmer batch {label} (mq6): {e:?}"))
        }
        // Fallback: these dtypes have no batched kernel for the given M/K.
        // Explicit per-row scalar GEMV (no approximation).
        //   Fallback dtypes: F32, Q4K, HFQ4G128, HFQ6G256, HFQ3G256, HFQ2G256, MQ3G256, etc.
        _ => {
            for i in 0..b {
                let x_row = x.sub_offset(i * w.k, w.k);
                let y_row = y.sub_offset(i * w.m, w.m);
                weight_gemv(gpu, w, &x_row, &y_row)
                    .map_err(|e| format!("glimmer batch {label} row {i}: {e}"))?;
            }
            Ok(())
        }
    }
}

/// Prerotated variant: `x_rot` is already FWHT-rotated for MQ4. Dispatches
/// without re-rotating. Used for the shared-rotation attn (q/k/v/gate share
/// one rotate) and ffn (gate/up share one). Q8 still reads the unrotated `x`.
fn proj_gemm_batched_prerotated(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x_unrot: &GpuTensor,
    x_rot: &GpuTensor,
    y: &GpuTensor,
    b: usize,
    label: &str,
) -> Result<(), String> {
    match w.gpu_dtype {
        DType::Q8_0 => gpu
            .gemm_q8_0_batched_chunked(&w.buf, x_unrot, y, w.m, w.k, b)
            .map_err(|e| format!("glimmer batch {label} (q8 prerot): {e:?}")),
        DType::MQ4G256 | DType::HFQ4G256 => gpu
            .gemm_hfq4g256_batched_lmhead(&w.buf, x_rot, y, w.m, w.k, b)
            .map_err(|e| format!("glimmer batch {label} (mq4 prerot): {e:?}")),
        DType::MQ6G256 => gpu
            .gemm_mq6g256_batched_lmhead(&w.buf, x_rot, y, w.m, w.k, b)
            .map_err(|e| format!("glimmer batch {label} (mq6 prerot): {e:?}")),
        _ => {
            // No batched kernel — per-row fallback (sedentary dtypes: F32 etc.)
            for i in 0..b {
                let x_row = x_unrot.sub_offset(i * w.k, w.k);
                let y_row = y.sub_offset(i * w.m, w.m);
                weight_gemv(gpu, w, &x_row, &y_row)
                    .map_err(|e| format!("glimmer batch {label} prerot row {i}: {e}"))?;
            }
            Ok(())
        }
    }
}
// ─── Prefill-only projection helpers (dispatch-routed) ─────────────────────
//
// These are PREFILL-ONLY and live beside proj_gemm_batched; DFlash verify
// keeps using proj_gemm_batched / proj_gemm_batched_prerotated unchanged.
// Caller picks KernelKey → no shared selector, no cross-arch clobber.
//   Q8_0            → GemmQ8_0BatchedChunked  (WMMA on gfx12, byte-identical)
//   MQ4G256/HFQ4G256 → GemmHfq4G256BatchedLmhead (prerotated, x_rot, byte-identical)
//   MQ6G256         → direct gemm_mq6g256_batched_lmhead (NO KernelKey, leave direct)
//   o/down plain GEMM + sandwich post-norm + add_inplace → NOT residual-fused,
//     so Gemm*Residual keys do not map 1:1 and are NOT forced.
//   others          → per-row weight_gemv fallback
// Always keep fused_rmsnorm_rotate_mq_batched_for and feed x_rot to every MQ4
// GEMM — never quantize/convert the unrotated nrm buffer.
fn prefill_proj_gemm_batched(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    x_rot: &GpuTensor,
    b: usize,
    label: &str,
) -> Result<(), String> {
    match w.gpu_dtype {
        DType::Q8_0 => run_prefill_plain_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
            &w.buf,
            w.gpu_dtype,
            x,
            y,
            w.m,
            w.k,
            b,
        )
        .map_err(|e| format!("glimmer prefill {label} (q8 dispatch): {e}")),
        DType::MQ4G256 | DType::HFQ4G256 => {
            rotate_x_mq_batched_for(gpu, w, x, x_rot, w.k, b)
                .map_err(|e| format!("glimmer prefill {label} rotate: {e:?}"))?;
            run_prefill_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256BatchedLmhead,
                &w.buf,
                w.gpu_dtype,
                x_rot,
                y,
                w.m,
                w.k,
                b,
            )
            .map_err(|e| format!("glimmer prefill {label} (mq4 dispatch): {e}"))
        }
        DType::MQ6G256 => {
            // No KernelKey — leave on direct call (shipped artifact is MQ4G256).
            rotate_x_mq_batched_for(gpu, w, x, x_rot, w.k, b)
                .map_err(|e| format!("glimmer prefill {label} rotate: {e:?}"))?;
            gpu.gemm_mq6g256_batched_lmhead(&w.buf, x_rot, y, w.m, w.k, b)
                .map_err(|e| format!("glimmer prefill {label} (mq6 direct): {e:?}"))
        }
        _ => {
            for i in 0..b {
                let x_row = x.sub_offset(i * w.k, w.k);
                let y_row = y.sub_offset(i * w.m, w.m);
                weight_gemv(gpu, w, &x_row, &y_row)
                    .map_err(|e| format!("glimmer prefill {label} row {i}: {e}"))?;
            }
            Ok(())
        }
    }
}

/// Prerotated prefill variant: x_rot already FWHT-rotated for MQ4.
fn prefill_proj_gemm_batched_prerotated(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x_unrot: &GpuTensor,
    x_rot: &GpuTensor,
    y: &GpuTensor,
    b: usize,
    label: &str,
) -> Result<(), String> {
    match w.gpu_dtype {
        DType::Q8_0 => run_prefill_plain_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
            &w.buf,
            w.gpu_dtype,
            x_unrot,
            y,
            w.m,
            w.k,
            b,
        )
        .map_err(|e| format!("glimmer prefill {label} (q8 prerot dispatch): {e}")),
        DType::MQ4G256 | DType::HFQ4G256 => run_prefill_plain_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmHfq4G256BatchedLmhead,
            &w.buf,
            w.gpu_dtype,
            x_rot,
            y,
            w.m,
            w.k,
            b,
        )
        .map_err(|e| format!("glimmer prefill {label} (mq4 prerot dispatch): {e}")),
        DType::MQ6G256 => gpu
            .gemm_mq6g256_batched_lmhead(&w.buf, x_rot, y, w.m, w.k, b)
            .map_err(|e| format!("glimmer prefill {label} (mq6 prerot direct): {e:?}")),
        _ => {
            for i in 0..b {
                let x_row = x_unrot.sub_offset(i * w.k, w.k);
                let y_row = y.sub_offset(i * w.m, w.m);
                weight_gemv(gpu, w, &x_row, &y_row)
                    .map_err(|e| format!("glimmer prefill {label} prerot row {i}: {e}"))?;
            }
            Ok(())
        }
    }
}
/// Shared lm_head routine for draft and verify. `hidden_batch` is [B*hidden]
/// already RMSNormed (the lm_head input). Returns argmax picks for rows
/// [0..batch) in order. Uses batched Q8 path when enabled and dtype is
/// Q8_0 (`gemm_q8_0_batched_chunked` + per-row scale/softcap), otherwise
/// per-row `weight_gemv` fallback (explicit, no approximation). This is the
/// single source of truth for draft and verify so they cannot diverge on
/// near-ties due to different accumulation order (the Q8 vs per-row flip that
/// halved tau on the Q8-head artifact).
pub fn glimmer_lm_head_picks(
    gpu: &mut Gpu,
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    hidden_batch: &GpuTensor,
    batch: usize,
    logits_scratch: &GpuTensor,
    logits_batch: &GpuTensor,
) -> Result<Vec<u32>, String> {
    let dim = cfg.dim;
    let vocab = cfg.vocab_size;
    let is_q8_lm = weights.lm_head.gpu_dtype == DType::Q8_0 && batched_lm_head_enabled();
    let mut picks = Vec::with_capacity(batch);
    if is_q8_lm {
        // Batched Q8: one GEMM reads lm_head once for all B rows (WMMA on gfx12).
        // Fallback to per-row if batched disabled via HIPFIRE_GLIMMER_BATCHED_LM_HEAD=0.
        // Persistent buffer, NOT a per-call alloc. A cold hipMalloc of this
        // ~12.9 MB is slow and synchronizing, and it was the entire reason the
        // first batched lm_head of a window cost 69 ms while the second cost
        // 7.6 ms for the same weight through the same kernel.
        if batch > GLIMMER_MAX_SPEC_BLOCK {
            return Err(format!(
                "glimmer lm_head_picks: batch {batch} exceeds GLIMMER_MAX_SPEC_BLOCK {GLIMMER_MAX_SPEC_BLOCK}"
            ));
        }
        let logits_b = logits_batch.sub_offset(0, batch * vocab);
        gpu.gemm_q8_0_batched_chunked(&weights.lm_head.buf, hidden_batch, &logits_b, vocab, dim, batch)
            .map_err(|e| format!("glimmer lm_head_picks q8 batched: {e:?}"))?;
        for i in 0..batch {
            let row = logits_b.sub_offset(i * vocab, vocab);
            if cfg.output_multiplier != 1.0 && !abl("HIPFIRE_GLIMMER_NO_OUTMUL") {
                gpu.scale_f32(&row, cfg.output_multiplier)
                    .map_err(|e| format!("glimmer lm_head_picks scale row {i}: {e:?}"))?;
            }
            if cfg.final_logit_softcapping > 0.0 && !abl("HIPFIRE_GLIMMER_NO_SOFTCAP") {
                gpu.logit_softcap_f32(&row, vocab, cfg.final_logit_softcapping)
                    .map_err(|e| format!("glimmer lm_head_picks softcap row {i}: {e:?}"))?;
            }
            gpu.hip
                .memcpy_dtod_at(&logits_scratch.buf, 0, &row.buf, 0, vocab * 4)
                .map_err(|e| format!("glimmer lm_head_picks copy row {i}: {e:?}"))?;
            let host = gpu
                .download_f32(logits_scratch)
                .map_err(|e| format!("glimmer lm_head_picks download row {i}: {e:?}"))?;
            let pick = host
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap();
            picks.push(pick);
        }
    } else {
        // MQ4 and disabled-Q8: per-row weight_gemv (re-streams lm_head B times).
        // Explicit fallback - same as draft side, so near-tie argmax cannot flip.
        for i in 0..batch {
            let hidden_row = hidden_batch.sub_offset(i * dim, dim);
            weight_gemv(gpu, &weights.lm_head, &hidden_row, logits_scratch)
                .map_err(|e| format!("glimmer lm_head_picks row {i}: {e}"))?;
            if cfg.output_multiplier != 1.0 && !abl("HIPFIRE_GLIMMER_NO_OUTMUL") {
                gpu.scale_f32(logits_scratch, cfg.output_multiplier)
                    .map_err(|e| format!("glimmer lm_head_picks scale row {i}: {e:?}"))?;
            }
            if cfg.final_logit_softcapping > 0.0 && !abl("HIPFIRE_GLIMMER_NO_SOFTCAP") {
                gpu.logit_softcap_f32(logits_scratch, vocab, cfg.final_logit_softcapping)
                    .map_err(|e| format!("glimmer lm_head_picks softcap row {i}: {e:?}"))?;
            }
            let host = gpu
                .download_f32(logits_scratch)
                .map_err(|e| format!("glimmer lm_head_picks download row {i}: {e:?}"))?;
            let pick = host
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap();
            picks.push(pick);
        }
    }
    Ok(picks)
}

// ─── Batched verify core ─────────────────────────────────────────────

/// Block verify that also captures hidden at `capture_layers` for each
/// position in the block, appending to `hidden_out` (same order as
/// decode_step_with_capture). Used by the DFlash spec loop to keep
/// `bundle.target_hidden_host` in sync with the committed prefix.
///
/// Batched: ONE forward over B positions reading each weight ONCE for all
/// rows (memory-bound decode → ~single AR cost per window instead of B×).
/// For each of q/k/v/attn_gate (sharing one input) and gate/up (sharing
/// another) we issue one batched GEMM over B rows. For MQ4G256 we pre-rotate
/// the batched input once (`rotate_x_mq_batched_for`) and call the
/// prerotated batched kernel, mirroring gemma4/qwen35. Where no batched
/// kernel exists we fall back to per-row `weight_gemv` (explicit, no
/// approximation) — f32 etc. The lm_head over B rows is vocab 202048
/// (~700 MB). Gemma4 uses batched Q8 but keeps per-row for MQ4 because the
/// batched MQ4 path faults at that output width on gfx12; Glimmer's lm_head
/// is MQ4G256 so we keep the same per-row scalar fallback explicitly (caps
/// the win: lm_head re-streams B times).
///
/// Attention respects per-layer sliding/full split and NoPE (theta 0) exactly
/// as the single path: sliding window 2048 vs full 0, rope skipped when
/// layer_rope_theta==0. Batched attention uses `attention_q8_0_kv_batched_masked`
/// with a per-row additive mask [B×seq_len] (0 causal / -inf masked) and
/// block_start=0 block_cols=seq_len giving full per-row control for both
/// layer types, matching gemma4's mask construction.
///
/// Capture contract: `hidden_out` is extended with B rows of
/// `capture_layers.len()*dim` f32, position-major in ascending tap order
/// (pos0: [L1,L13,...], pos1: [L1,L13,...], ...), so the caller's
/// `truncate(hidden_before+keep_rows*row_elems)` keeps the committed prefix.
pub fn verify_block_with_capture(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    block: &[u32],
    position: u32,
    capture_layers: &[usize],
    hidden_out: &mut Vec<f32>,
) -> Result<Vec<u32>, String> {
    let b = block.len();
    if b == 0 {
        return Ok(Vec::new());
    }
    if b > 64 {
        return Err(format!("glimmer verify_block: B={b} exceeds kernel cap 64"));
    }
    // Fast path for B==1 uses the single path directly for exact parity with
    // the AR code (no batched kernels). Correctness gate for B>1 is the
    // per-row fallback parity vs sequential.
    // For uniformity we still run the batched path for B==1 as it's valid,
    // but keep the sequential fallback if needed for debugging via env.
    // Use batched for all B to exercise the kernels.
    let dim = cfg.dim;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let hd = cfg.head_dim;
    let n_heads = cfg.n_heads;
    let n_kv = cfg.n_kv_heads;
    let rms_eps = cfg.rms_norm_eps;
    let post_eps = cfg.post_norm_eps;
    let seq_len = position as usize + b;
    let phys_cap = state.max_seq;
    let t_verify_start = std::time::Instant::now();
    let do_timing = std::env::var("HIPFIRE_GLIMMER_TIMING").ok().as_deref() == Some("1");

    // ── Batched scratch (per call; verify is not the hot per-kernel loop) ──
    let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
        g.alloc_tensor(&[n], DType::F32)
            .map_err(|e| format!("glimmer verify_batch alloc {label}: {e:?}"))
    };
    // Same ownership discipline as `prefill_chunk_batched`: take all 17 scratch tensors
    // into one list so a failure part-way through frees what it already holds instead of
    // leaking it for the life of the process. Allocation is where memory pressure lands,
    // and the daemon retries a failed generate, so a leak here compounds per retry.
    let scratch_spec: [(usize, &str); 17] = [
        (b * dim, "x"),
        (b * dim, "residual"),
        (b * dim, "nrm"),
        (b * dim, "x_rot"),
        (b * q_dim, "q"),
        (b * kv_dim, "k"),
        (b * kv_dim, "v"),
        (b * q_dim, "attn_gate"),
        (b * q_dim, "attn_out"),
        (b * q_dim, "o_rot"),
        (b * dim, "o_out"),
        (b * cfg.hidden_dim, "gate_ffn"),
        (b * cfg.hidden_dim, "up_ffn"),
        (b * cfg.hidden_dim, "ffn_hidden"),
        (b * dim, "ffn_out"),
        (b * cfg.hidden_dim, "down_rot"),
        (b, "pos_array"),
    ];
    let mut taken: Vec<GpuTensor> = Vec::with_capacity(scratch_spec.len());
    for (n, label) in scratch_spec {
        match gpu.alloc_tensor(&[n], DType::F32) {
            Ok(t) => taken.push(t),
            Err(e) => {
                for t in taken.drain(..) {
                    gpu.free_tensor(t).ok();
                }
                return Err(format!("glimmer verify_batch alloc {label}: {e:?}"));
            }
        }
    }
    let mut scratch_iter = taken.into_iter();
    let mut next_scratch =
        move || scratch_iter.next().expect("scratch_spec length matches the bindings below");
    let x = next_scratch();
    let residual = next_scratch();
    let nrm = next_scratch();
    let x_rot = next_scratch();
    let q = next_scratch();
    let k = next_scratch();
    let v = next_scratch();
    let attn_gate = next_scratch();
    let attn_out = next_scratch();
    let o_rot = next_scratch();
    let o_out = next_scratch();
    let gate_ffn = next_scratch();
    let up_ffn = next_scratch();
    let ffn_hidden = next_scratch();
    let ffn_out = next_scratch();
    let down_rot = next_scratch();
    let pos_array = next_scratch();

    // positions [B] i32
    let pos_data: Vec<i32> = (0..b).map(|i| (position + i as u32) as i32).collect();
    let pos_bytes: Vec<u8> = pos_data.iter().flat_map(|p| p.to_ne_bytes()).collect();
    gpu.hip
        .memcpy_htod(&pos_array.buf, &pos_bytes)
        .map_err(|e| format!("glimmer verify_batch htod pos: {e:?}"))?;
    let t_after_alloc = t_verify_start.elapsed();

    // Attention mask: for max_seq 2048 sliding_window 2048 == max_seq, causal suffices.
    // For larger max_seq where window < seq_len we would need a windowed mask via
    // attention_q8_0_kv_batched_masked with a per-row additive mask. At 64 tokens
    // window is not restrictive, so we use the faster causal batched kernel.

    // ── Embedding: per-token lookup + embed_norm into x[B,dim] ──
    {
        let x_single = alloc(gpu, dim, "x_single")?;
        for (i, &tok) in block.iter().enumerate() {
            match weights.embd_format {
                EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.embed_tokens, &x_single, tok, dim).map_err(|e| format!("glimmer verify_batch embed hfq4g256: {e:?}"))?,
                EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.embed_tokens, &x_single, tok, dim).map_err(|e| format!("glimmer verify_batch embed hfq4g128: {e:?}"))?,
                EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.embed_tokens, &x_single, tok, dim).map_err(|e| format!("glimmer verify_batch embed q8: {e:?}"))?,
                EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.embed_tokens, &x_single, tok, dim).map_err(|e| format!("glimmer verify_batch embed f32: {e:?}"))?,
                EmbeddingFormat::Q4K => return Err("glimmer verify_batch: Q4K embed unsupported".to_string()),
            }
            if !abl("HIPFIRE_GLIMMER_NO_EMBED_NORM") {
                gpu.rmsnorm_f32(&x_single, &state.embed_norm_ones, &x_single, rms_eps).map_err(|e| format!("glimmer verify_batch embed_norm: {e:?}"))?;
            }
            gpu.hip.memcpy_dtod_at(&x.buf, i * dim * 4, &x_single.buf, 0, dim * 4).map_err(|e| format!("glimmer verify_batch embed copy: {e:?}"))?;
        }
        gpu.free_tensor(x_single).ok();
    }
    let t_after_embed = t_verify_start.elapsed();

    // ── Capture buffer (position-major) ──
    let mut sorted_caps: Vec<usize> = capture_layers.to_vec();
    sorted_caps.sort_unstable();
    let cap_cnt = sorted_caps.len();
    let mut cap_index = vec![None; cfg.n_layers];
    for (ci, &li) in sorted_caps.iter().enumerate() {
        if li < cfg.n_layers { cap_index[li] = Some(ci); }
    }
    let mut cap_buf: Vec<f32> = if cap_cnt > 0 { vec![0.0f32; b * cap_cnt * dim] } else { Vec::new() };

    // ── Per-layer batched forward ──
    for layer_idx in 0..cfg.n_layers {
        let slot = state.kv_slot_for_layer[layer_idx];
        let lw = &weights.layers[layer_idx];
        let has_rope = cfg.has_rope(layer_idx);
        let window = cfg.window_for(layer_idx);

        // residual = x
        gpu.hip.memcpy_dtod_at(&residual.buf, 0, &x.buf, 0, b * dim * 4).map_err(|e| format!("glimmer batch L{layer_idx} save residual: {e:?}"))?;

        // ── Attention input norm + q/k/v/gate projections (shared input) ──
        let need_shared = shared_rot_enabled()
            && hipfire_dispatch::types::dtype_rotation_plan(lw.q_proj.gpu_dtype) != hipfire_dispatch::types::RotationPlan::None;
        if need_shared {
            // Fused rmsnorm+FWHT: one launch writes x_rot directly from x, saving the
            // separate rmsnorm_batched + rotate_x_mq_batched pair (104 launches per
            // window). Byte-identical to the two-step sequence. Uses the batched
            // fused helper which handles AWQ sidecars internally.
            fused_rmsnorm_rotate_mq_batched_for(
                gpu, &x, &lw.input_layernorm, &lw.q_proj, &x_rot, dim, rms_eps, b,
            )
            .map_err(|e| format!("glimmer batch L{layer_idx} fused input rmsnorm+rotate: {e:?}"))?;
            // Prerotated GEMMs share x_rot (no re-rotation). Q8 would use nrm directly
            // but this branch is only taken for rotating dtypes (MQ), so Q8 not present.
            proj_gemm_batched_prerotated(gpu, &lw.q_proj, &nrm, &x_rot, &q, b, "q_proj")?;
            proj_gemm_batched_prerotated(gpu, &lw.k_proj, &nrm, &x_rot, &k, b, "k_proj")?;
            proj_gemm_batched_prerotated(gpu, &lw.v_proj, &nrm, &x_rot, &v, b, "v_proj")?;
            proj_gemm_batched_prerotated(gpu, &lw.attn_gate_proj, &nrm, &x_rot, &attn_gate, b, "attn_gate")?;
        } else {
            gpu.rmsnorm_batched(&x, &lw.input_layernorm, &nrm, b, dim, rms_eps).map_err(|e| format!("glimmer batch L{layer_idx} input rmsnorm: {e:?}"))?;
            proj_gemm_batched(gpu, &lw.q_proj, &nrm, &q, &x_rot, b, "q_proj")?;
            proj_gemm_batched(gpu, &lw.k_proj, &nrm, &k, &x_rot, b, "k_proj")?;
            proj_gemm_batched(gpu, &lw.v_proj, &nrm, &v, &x_rot, b, "v_proj")?;
            proj_gemm_batched(gpu, &lw.attn_gate_proj, &nrm, &attn_gate, &x_rot, b, "attn_gate")?;
        }

        // Scale-less QK-norm + q scale 3.87
        if !abl("HIPFIRE_GLIMMER_NO_QK_NORM") {
            gpu.rmsnorm_batched(&q, &state.qk_norm_ones, &q, b * n_heads, hd, rms_eps).map_err(|e| format!("glimmer batch L{layer_idx} q_norm: {e:?}"))?;
            gpu.rmsnorm_batched(&k, &state.qk_norm_ones, &k, b * n_kv, hd, rms_eps).map_err(|e| format!("glimmer batch L{layer_idx} k_norm: {e:?}"))?;
        }
        if !abl("HIPFIRE_GLIMMER_NO_QK_SCALE") {
            gpu.scale_f32(&q, cfg.qk_scale_factor).map_err(|e| format!("glimmer batch L{layer_idx} q scale: {e:?}"))?;
        }

        // RoPE (skip on NoPE layers where theta==0)
        if has_rope || abl("HIPFIRE_GLIMMER_ROPE_ALL") {
            let theta = cfg.rope_theta_for(layer_idx);
            if abl("HIPFIRE_GLIMMER_ROPE_INTERLEAVED") {
                gpu.rope_interleaved_f32_batched(&q, &k, &pos_array, n_heads, n_kv, hd, hd, theta, b).map_err(|e| format!("glimmer batch L{layer_idx} rope interleaved batched: {e:?}"))?;
            } else {
                gpu.rope_batched_f32(&q, &k, &pos_array, n_heads, n_kv, hd, theta, b).map_err(|e| format!("glimmer batch L{layer_idx} rope batched: {e:?}"))?;
            }
        }

        // KV write batched
        let kv = match cfg.layer_types[layer_idx] {
            GlimmerLayerType::Sliding => &state.kv_sliding,
            GlimmerLayerType::Full => &state.kv_full,
        };
        let k_cache = unsafe { kv.k_gpu[slot].buf.alias() };
        let v_cache = unsafe { kv.v_gpu[slot].buf.alias() };
        let k_cache_t = GpuTensor { buf: k_cache, shape: kv.k_gpu[slot].shape.clone(), dtype: kv.k_gpu[slot].dtype };
        let v_cache_t = GpuTensor { buf: v_cache, shape: kv.v_gpu[slot].shape.clone(), dtype: kv.v_gpu[slot].dtype };
        gpu.kv_cache_write_q8_0_batched(&k_cache_t, &k, &pos_array, n_kv, hd, b).map_err(|e| format!("glimmer batch L{layer_idx} kv write k: {e:?}"))?;
        gpu.kv_cache_write_q8_0_batched(&v_cache_t, &v, &pos_array, n_kv, hd, b).map_err(|e| format!("glimmer batch L{layer_idx} kv write v: {e:?}"))?;

        // Verify previously used a per-query-row kernel whose grid carries the
        // query index ([n_heads, batch, 1]), so each of the B rows re-streamed
        // the entire KV span, making verify attention O(B*ctx) against AR's
        // O(ctx) and inverting spec decode into a loss at depth; and the 39
        // sliding layers were attending over the full span rather than their
        // 2048 window once seq_len exceeded it. Both are fixed by taking the
        // same query-tiled WMMA kernels prefill already uses (one WMMA B
        // fragment reused across 16 query rows: O(ctx)).
        let wmma_full = b > 1
            && std::env::var("HIPFIRE_GLIMMER_WMMA_FULL")
                .map(|v| v != "0" && !v.is_empty())
                .unwrap_or(true);
        let mut wmma_done = false;
        if wmma_full {
            if window == 0 {
                gpu.attention_q8_0_flash_prefill_wmma(
                    &q,
                    &k_cache_t,
                    &v_cache_t,
                    &attn_out,
                    &pos_array,
                    n_heads,
                    n_kv,
                    hd,
                    b,
                )
                .map_err(|e| format!("glimmer batch L{layer_idx} wmma full: {e:?}"))?;
                wmma_done = true;
            } else {
                wmma_done = gpu
                    .attention_q8_0_flash_prefill_wmma_swa(
                        &q,
                        &k_cache_t,
                        &v_cache_t,
                        &attn_out,
                        &pos_array,
                        n_heads,
                        n_kv,
                        hd,
                        b,
                        window,
                    )
                    .map_err(|e| format!("glimmer batch L{layer_idx} wmma swa: {e:?}"))?;
            }
        }
        if !wmma_done {
            gpu.attention_q8_0_kv_batched(&q, &k_cache_t, &v_cache_t, &attn_out, &pos_array, n_heads, n_kv, hd, phys_cap, seq_len, b)
                .map_err(|e| format!("glimmer batch L{layer_idx} attention batched: {e:?}"))?;
        }

        // Gated attention: attn_out *= sigmoid(gate) BEFORE o_proj
        if !abl("HIPFIRE_GLIMMER_NO_ATTN_GATE") {
            gpu.sigmoid_mul_f32(&attn_out, &attn_gate).map_err(|e| format!("glimmer batch L{layer_idx} sigmoid_mul: {e:?}"))?;
        }

        // o_proj: attn_out [B,q_dim] -> o_out [B,dim]
        // Dispatch by dtype; for MQ4 we rotate attn_out, for Q8 direct.
        // Use the per-projection helper that handles rotate internally.
        proj_gemm_batched(gpu, &lw.o_proj, &attn_out, &o_out, &o_rot, b, "o_proj")?;

        // Sandwich post-attention norm + residual add: x = residual + norm(o_out)
        gpu.rmsnorm_batched(&o_out, &lw.post_attention_layernorm, &o_out, b, dim, post_eps).map_err(|e| format!("glimmer batch L{layer_idx} post_attn rmsnorm: {e:?}"))?;
        gpu.hip.memcpy_dtod_at(&x.buf, 0, &residual.buf, 0, b * dim * 4).map_err(|e| format!("glimmer batch L{layer_idx} reset x (attn): {e:?}"))?;
        gpu.add_inplace_f32(&x, &o_out).map_err(|e| format!("glimmer batch L{layer_idx} attn residual add: {e:?}"))?;

        // residual = x for FFN
        gpu.hip.memcpy_dtod_at(&residual.buf, 0, &x.buf, 0, b * dim * 4).map_err(|e| format!("glimmer batch L{layer_idx} save ffn residual: {e:?}"))?;

        // ── FFN: gate/up share input ──
        let ffn_shared = shared_rot_enabled()
            && hipfire_dispatch::types::dtype_rotation_plan(lw.gate_proj.gpu_dtype) != hipfire_dispatch::types::RotationPlan::None;
        if ffn_shared {
            fused_rmsnorm_rotate_mq_batched_for(
                gpu, &x, &lw.pre_feedforward_layernorm, &lw.gate_proj, &x_rot, dim, rms_eps, b,
            )
            .map_err(|e| format!("glimmer batch L{layer_idx} fused pre_ffn rmsnorm+rotate: {e:?}"))?;
            proj_gemm_batched_prerotated(gpu, &lw.gate_proj, &nrm, &x_rot, &gate_ffn, b, "gate_proj")?;
            proj_gemm_batched_prerotated(gpu, &lw.up_proj, &nrm, &x_rot, &up_ffn, b, "up_proj")?;
        } else {
            gpu.rmsnorm_batched(&x, &lw.pre_feedforward_layernorm, &nrm, b, dim, rms_eps).map_err(|e| format!("glimmer batch L{layer_idx} pre_ffn rmsnorm: {e:?}"))?;
            proj_gemm_batched(gpu, &lw.gate_proj, &nrm, &gate_ffn, &x_rot, b, "gate_proj")?;
            proj_gemm_batched(gpu, &lw.up_proj, &nrm, &up_ffn, &x_rot, b, "up_proj")?;
        }

        gpu.silu_mul_f32(&gate_ffn, &up_ffn, &ffn_hidden).map_err(|e| format!("glimmer batch L{layer_idx} silu_mul: {e:?}"))?;
        proj_gemm_batched(gpu, &lw.down_proj, &ffn_hidden, &ffn_out, &down_rot, b, "down_proj")?;

        // post-FFN norm + residual add: x = residual + norm(ffn_out)
        gpu.rmsnorm_batched(&ffn_out, &lw.post_feedforward_layernorm, &ffn_out, b, dim, post_eps).map_err(|e| format!("glimmer batch L{layer_idx} post_ffn rmsnorm: {e:?}"))?;
        gpu.hip.memcpy_dtod_at(&x.buf, 0, &residual.buf, 0, b * dim * 4).map_err(|e| format!("glimmer batch L{layer_idx} reset x (ffn): {e:?}"))?;
        gpu.add_inplace_f32(&x, &ffn_out).map_err(|e| format!("glimmer batch L{layer_idx} ffn residual add: {e:?}"))?;

        // ── Capture after this layer if needed ──
        if let Some(ci) = cap_index[layer_idx] {
            let host = gpu.download_f32(&x).map_err(|e| format!("glimmer batch capture L{layer_idx}: {e:?}"))?;
            // host is [B*dim]; fill cap_buf position-major: cap_buf[(pos*cap_cnt + ci)*dim ..]
            for row in 0..b {
                let src_off = row * dim;
                let dst_off = (row * cap_cnt + ci) * dim;
                cap_buf[dst_off..dst_off + dim].copy_from_slice(&host[src_off..src_off + dim]);
            }
        }
    }
    let t_after_layers = t_verify_start.elapsed();
    // Extend hidden_out in position-major order (pos0 cap0,cap1,... pos1 cap0,...)
    if cap_cnt > 0 {
        hidden_out.extend_from_slice(&cap_buf);
    }
    state.n_tokens = seq_len;

    // ── Final norm + lm_head per position → picks (shared with draft) ──
    let normed = alloc(gpu, b * dim, "normed")?;
    let t_after_norm = t_verify_start.elapsed();
    gpu.rmsnorm_batched(&x, &weights.final_norm, &normed, b, dim, rms_eps).map_err(|e| format!("glimmer batch final rmsnorm: {e:?}"))?;

    let picks = glimmer_lm_head_picks(gpu, cfg, weights, &normed, b, &state.logits, &state.logits_batch)?;
    let t_after_lm = t_verify_start.elapsed();
    if do_timing {
        let total = t_verify_start.elapsed();
        eprintln!(
            "[glimmer-verify-timing] B={} alloc={:.1}ms embed={:.1}ms layers={:.1}ms norm={:.1}ms lm={:.1}ms total={:.1}ms cap={} seq_len={}",
            b,
            t_after_alloc.as_secs_f64() * 1000.0,
            (t_after_embed - t_after_alloc).as_secs_f64() * 1000.0,
            (t_after_layers - t_after_embed).as_secs_f64() * 1000.0,
            (t_after_norm - t_after_layers).as_secs_f64() * 1000.0,
            (t_after_lm - t_after_norm).as_secs_f64() * 1000.0,
            total.as_secs_f64() * 1000.0,
            cap_cnt,
            seq_len
        );
    }

    // Free batched scratch
    for t in [x, residual, nrm, x_rot, q, k, v, attn_gate, attn_out, o_rot, o_out, gate_ffn, up_ffn, ffn_hidden, ffn_out, down_rot, pos_array, normed] {
        gpu.free_tensor(t).ok();
    }

    Ok(picks)
}
// ─── Chunked batched prefill ─────────────────────────────────────────────

/// Inner batched forward for one prefill chunk (B = chunk.len()).
/// Shared layer machinery with `verify_block_with_capture`: same
/// `proj_gemm_batched` / `fused_rmsnorm_rotate_mq_batched_for` helpers,
/// same QK-norm/scale, same RoPE, same KV-write, same sandwich norms.
/// Attention is causal within chunk + attends to all prior KV.
/// For sliding layers (window=2048) where `seq_len > window`, attention goes
/// through `attention_flash_q8_0_batched_masked_windowed` with `tree_bias=None`
/// — a batched windowed path that expresses the sliding mask natively. Full /
/// NoPE layers (`layer_rope_theta == 0`) and any chunk with `seq_len <= window`
/// stay on the fast batched causal `attention_q8_0_kv_batched`.
///
/// Do NOT reach for `attention_q8_0_kv_batched_masked` here. Passing it a
/// non-null bias silently switches it into DDTree mode, where it substitutes
/// `positions[0]` for the scalar `block_start` and sets its key extent to
/// `positions[0] + block_cols`, leaving `[0, positions[0])` unmasked by
/// contract. An earlier revision did exactly that with `block_cols` already
/// equal to `P+B`, so it read `2P+B` keys, shifted the bias by `P`, masked the
/// real current block, and exposed stale prefix plus unwritten KV — at the
/// final chunk addressing 8337 slots of an 8192-entry cache. It returned
/// well-formed tensors and fluent-looking garbage above 2048, and every chunk
/// size agreed with every other chunk size, so only an oracle comparison
/// caught it. That kernel cannot express a sliding window at any bias value.
///
/// If `need_logits` is true, final norm + single-row lm_head is run on
/// the LAST position only (not over the whole chunk). This avoids
/// re-streaming the 202048-wide lm_head weight B times; we stream it
/// once instead of B times. Say what we did: prefill discards
/// intermediate logits, so we compute vocab only for position `pos+B-1`.
fn prefill_chunk_batched(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    chunk: &[u32],
    position: u32,
    capture_layers: &[usize],
    hidden_out: &mut Vec<f32>,
    need_logits: bool,
) -> Result<Option<Vec<f32>>, String> {
    let b = chunk.len();
    if b == 0 {
        return Ok(None);
    }
    if b > 512 {
        return Err(format!("glimmer prefill: B={b} exceeds chunk cap 512"));
    }
    let dim = cfg.dim;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let hd = cfg.head_dim;
    let n_heads = cfg.n_heads;
    let n_kv = cfg.n_kv_heads;
    let rms_eps = cfg.rms_norm_eps;
    let post_eps = cfg.post_norm_eps;
    let seq_len = position as usize + b;
    let phys_cap = state.max_seq;

    let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
        g.alloc_tensor(&[n], DType::F32)
            .map_err(|e| format!("glimmer prefill alloc {label}: {e:?}"))
    };

    // Allocate every batched scratch tensor into ONE owner list before running any
    // kernel. Each `alloc(...)?` used to return early on failure, leaking every tensor
    // it had already taken; and because the daemon RETRIES a failed generate, each retry
    // leaked another full set. Measured on gfx1201 with muse-glimmer-30b.mq4: free VRAM
    // fell 881.9 MB -> 8.5 MB across a handful of oversized requests, after which every
    // request OOM'd until the process was restarted. One oversized request bricked the
    // daemon.
    //
    // `GpuTensor` is not `Clone`, so this Vec *is* the ownership until it is destructured
    // into the named bindings below; on failure it frees exactly what it holds.
    let scratch_spec: [(usize, &str); 17] = [
        (b * dim, "x"),
        (b * dim, "residual"),
        (b * dim, "nrm"),
        (b * dim, "x_rot"),
        (b * q_dim, "q"),
        (b * kv_dim, "k"),
        (b * kv_dim, "v"),
        (b * q_dim, "attn_gate"),
        (b * q_dim, "attn_out"),
        (b * q_dim, "o_rot"),
        (b * dim, "o_out"),
        (b * cfg.hidden_dim, "gate_ffn"),
        (b * cfg.hidden_dim, "up_ffn"),
        (b * cfg.hidden_dim, "ffn_hidden"),
        (b * dim, "ffn_out"),
        (b * cfg.hidden_dim, "down_rot"),
        (b, "pos_array"),
    ];
    let mut taken: Vec<GpuTensor> = Vec::with_capacity(scratch_spec.len());
    for (n, label) in scratch_spec {
        match gpu.alloc_tensor(&[n], DType::F32) {
            Ok(t) => taken.push(t),
            Err(e) => {
                for t in taken.drain(..) {
                    gpu.free_tensor(t).ok();
                }
                return Err(format!("glimmer prefill alloc {label}: {e:?}"));
            }
        }
    }
    let mut scratch_iter = taken.into_iter();
    let mut next_scratch =
        move || scratch_iter.next().expect("scratch_spec length matches the bindings below");
    let x = next_scratch();
    let residual = next_scratch();
    let nrm = next_scratch();
    let x_rot = next_scratch();
    let q = next_scratch();
    let k = next_scratch();
    let v = next_scratch();
    let attn_gate = next_scratch();
    let attn_out = next_scratch();
    let o_rot = next_scratch();
    let o_out = next_scratch();
    let gate_ffn = next_scratch();
    let up_ffn = next_scratch();
    let ffn_hidden = next_scratch();
    let ffn_out = next_scratch();
    let down_rot = next_scratch();
    let pos_array = next_scratch();

    // Everything below runs inside a closure so that EVERY `?` unwinds to the single
    // free site after it, instead of returning straight out of the function and leaking
    // all 17 tensors. The body is otherwise unchanged.
    let body_result: Result<Option<Vec<f32>>, String> = (|| {
    let pos_data: Vec<i32> = (0..b).map(|i| (position + i as u32) as i32).collect();
    let pos_bytes: Vec<u8> = pos_data.iter().flat_map(|p| p.to_ne_bytes()).collect();
    gpu.hip
        .memcpy_htod(&pos_array.buf, &pos_bytes)
        .map_err(|e| format!("glimmer prefill htod pos: {e:?}"))?;

    // Window-bounded tile origin for the sliding layers, computed ONCE per
    // chunk. Every sliding layer shares the same window and the same chunk
    // position, so the rebased positions are identical across all 39 of them;
    // the full (window==0) layers use origin 0 and the unshifted array. This
    // was briefly built inside the layer loop, which allocated 52 tensors per
    // chunk and never freed them.
    let sliding_window = cfg.sliding_window;
    let sliding_origin = if sliding_window > 0 && seq_len > sliding_window {
        ((position as usize + 1).saturating_sub(sliding_window) / 128) * 128
    } else {
        0
    };
    let shifted_pos_array = if sliding_origin > 0 {
        let sd: Vec<i32> = (0..b)
            .map(|i| (position as usize + i - sliding_origin) as i32)
            .collect();
        let sb: Vec<u8> = sd.iter().flat_map(|p| p.to_ne_bytes()).collect();
        let t = alloc(gpu, b, "shifted_pos")?;
        gpu.hip
            .memcpy_htod(&t.buf, &sb)
            .map_err(|e| format!("glimmer prefill htod shifted pos: {e:?}"))?;
        Some(t)
    } else {
        None
    };

    // ── Embedding ──
    {
        let x_single = alloc(gpu, dim, "x_single")?;
        for (i, &tok) in chunk.iter().enumerate() {
            match weights.embd_format {
                EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.embed_tokens, &x_single, tok, dim).map_err(|e| format!("glimmer prefill embed hfq4g256: {e:?}"))?,
                EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.embed_tokens, &x_single, tok, dim).map_err(|e| format!("glimmer prefill embed hfq4g128: {e:?}"))?,
                EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.embed_tokens, &x_single, tok, dim).map_err(|e| format!("glimmer prefill embed q8: {e:?}"))?,
                EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.embed_tokens, &x_single, tok, dim).map_err(|e| format!("glimmer prefill embed f32: {e:?}"))?,
                EmbeddingFormat::Q4K => return Err("glimmer prefill: Q4K embed unsupported".to_string()),
            }
            if !abl("HIPFIRE_GLIMMER_NO_EMBED_NORM") {
                gpu.rmsnorm_f32(&x_single, &state.embed_norm_ones, &x_single, rms_eps).map_err(|e| format!("glimmer prefill embed_norm: {e:?}"))?;
            }
            gpu.hip.memcpy_dtod_at(&x.buf, i * dim * 4, &x_single.buf, 0, dim * 4).map_err(|e| format!("glimmer prefill embed copy: {e:?}"))?;
        }
        gpu.free_tensor(x_single).ok();
    }

    // Capture prep
    let mut sorted_caps: Vec<usize> = capture_layers.to_vec();
    sorted_caps.sort_unstable();
    let cap_cnt = sorted_caps.len();
    let mut cap_index = vec![None; cfg.n_layers];
    for (ci, &li) in sorted_caps.iter().enumerate() {
        if li < cfg.n_layers { cap_index[li] = Some(ci); }
    }
    let mut cap_buf: Vec<f32> = if cap_cnt > 0 { vec![0.0f32; b * cap_cnt * dim] } else { Vec::new() };

    // ── Per-layer batched forward ──
    for layer_idx in 0..cfg.n_layers {
        let slot = state.kv_slot_for_layer[layer_idx];
        let lw = &weights.layers[layer_idx];
        let has_rope = cfg.has_rope(layer_idx);
        let window = cfg.window_for(layer_idx);

        gpu.hip.memcpy_dtod_at(&residual.buf, 0, &x.buf, 0, b * dim * 4).map_err(|e| format!("glimmer prefill L{layer_idx} save residual: {e:?}"))?;

        // ── Q/K/V/attn_gate: fused qkvza on gfx12 ──
        // Microbench: gemm_qkvza_hfq4g256_wmma_gfx12 ONE call = 2.65x at B256
        // vs four sequential GemmHfq4G256BatchedLmhead calls. Arch-gated so
        // non-gfx12 keeps the old four-call path. FWHT contract: feed x_rot.
        let qkvza_fused_eligible = b > 1
            && gpu.arch_caps.has_wmma_w32_gfx12()
            && matches!(lw.q_proj.gpu_dtype, DType::MQ4G256 | DType::HFQ4G256)
            && lw.q_proj.gpu_dtype == lw.k_proj.gpu_dtype
            && lw.q_proj.gpu_dtype == lw.v_proj.gpu_dtype
            && lw.q_proj.gpu_dtype == lw.attn_gate_proj.gpu_dtype;
        if qkvza_fused_eligible {
            let need_shared_qkv = shared_rot_enabled()
                && hipfire_dispatch::types::dtype_rotation_plan(lw.q_proj.gpu_dtype) != hipfire_dispatch::types::RotationPlan::None;
            if need_shared_qkv {
                fused_rmsnorm_rotate_mq_batched_for(
                    gpu, &x, &lw.input_layernorm, &lw.q_proj, &x_rot, dim, rms_eps, b,
                ).map_err(|e| format!("glimmer prefill L{layer_idx} fused input rmsnorm+rotate: {e:?}"))?;
            } else {
                gpu.rmsnorm_batched(&x, &lw.input_layernorm, &nrm, b, dim, rms_eps).map_err(|e| format!("glimmer prefill L{layer_idx} input rmsnorm: {e:?}"))?;
                rotate_x_mq_batched_for(gpu, &lw.q_proj, &nrm, &x_rot, dim, b).map_err(|e| format!("glimmer prefill L{layer_idx} rotate for fused qkvza: {e:?}"))?;
            }
            // Same pointer-keyed F16 hazard as the FFN site below: `x_rot` is
            // reused across all 52 layers, so without this the qkvza call in
            // layer N could be handed layer N-1's converted activations.
            // Inert on RDNA (the ensure_fp16_x is behind `is_cdna3()`); live on
            // CDNA3, and silent if it ever fires.
            gpu.invalidate_fp16_cache();
            // Glimmer's gate multiplies attn_out by sigmoid(gate) BEFORE o_proj.
            // The qkvza kernel was written for DeltaNet z/beta/alpha, but
            // mathematically it is 4 plain GEMMs sharing one x_rot:
            //   Y_i = W_i · x_rot  (i = q, k, v, gate)
            // No DeltaNet recurrence is fused. Fourth output is attn_gate.
            run_prefill_fused_qkvza_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedQkvzaHfq4G256,
                &lw.q_proj.buf,
                &lw.k_proj.buf,
                &lw.v_proj.buf,
                &lw.attn_gate_proj.buf,
                &x_rot,
                &q,
                &k,
                &v,
                &attn_gate,
                lw.q_proj.m,
                lw.k_proj.m,
                lw.v_proj.m,
                lw.attn_gate_proj.m,
                lw.q_proj.k,
                b,
            ).map_err(|e| format!("glimmer prefill L{layer_idx} fused qkvza: {e}"))?;
        } else {
            let need_shared = shared_rot_enabled()
                && hipfire_dispatch::types::dtype_rotation_plan(lw.q_proj.gpu_dtype) != hipfire_dispatch::types::RotationPlan::None;
            if need_shared {
                fused_rmsnorm_rotate_mq_batched_for(
                    gpu, &x, &lw.input_layernorm, &lw.q_proj, &x_rot, dim, rms_eps, b,
                ).map_err(|e| format!("glimmer prefill L{layer_idx} fused input rmsnorm+rotate: {e:?}"))?;
                prefill_proj_gemm_batched_prerotated(gpu, &lw.q_proj, &nrm, &x_rot, &q, b, "q_proj")?;
                prefill_proj_gemm_batched_prerotated(gpu, &lw.k_proj, &nrm, &x_rot, &k, b, "k_proj")?;
                prefill_proj_gemm_batched_prerotated(gpu, &lw.v_proj, &nrm, &x_rot, &v, b, "v_proj")?;
                prefill_proj_gemm_batched_prerotated(gpu, &lw.attn_gate_proj, &nrm, &x_rot, &attn_gate, b, "attn_gate")?;
            } else {
                gpu.rmsnorm_batched(&x, &lw.input_layernorm, &nrm, b, dim, rms_eps).map_err(|e| format!("glimmer prefill L{layer_idx} input rmsnorm: {e:?}"))?;
                prefill_proj_gemm_batched(gpu, &lw.q_proj, &nrm, &q, &x_rot, b, "q_proj")?;
                prefill_proj_gemm_batched(gpu, &lw.k_proj, &nrm, &k, &x_rot, b, "k_proj")?;
                prefill_proj_gemm_batched(gpu, &lw.v_proj, &nrm, &v, &x_rot, b, "v_proj")?;
                prefill_proj_gemm_batched(gpu, &lw.attn_gate_proj, &nrm, &attn_gate, &x_rot, b, "attn_gate")?;
            }
        }
        if !abl("HIPFIRE_GLIMMER_NO_QK_NORM") {
            gpu.rmsnorm_batched(&q, &state.qk_norm_ones, &q, b * n_heads, hd, rms_eps).map_err(|e| format!("glimmer prefill L{layer_idx} q_norm: {e:?}"))?;
            gpu.rmsnorm_batched(&k, &state.qk_norm_ones, &k, b * n_kv, hd, rms_eps).map_err(|e| format!("glimmer prefill L{layer_idx} k_norm: {e:?}"))?;
        }
        if !abl("HIPFIRE_GLIMMER_NO_QK_SCALE") {
            gpu.scale_f32(&q, cfg.qk_scale_factor).map_err(|e| format!("glimmer prefill L{layer_idx} q scale: {e:?}"))?;
        }

        if has_rope || abl("HIPFIRE_GLIMMER_ROPE_ALL") {
            let theta = cfg.rope_theta_for(layer_idx);
            if abl("HIPFIRE_GLIMMER_ROPE_INTERLEAVED") {
                gpu.rope_interleaved_f32_batched(&q, &k, &pos_array, n_heads, n_kv, hd, hd, theta, b).map_err(|e| format!("glimmer prefill L{layer_idx} rope interleaved batched: {e:?}"))?;
            } else {
                gpu.rope_batched_f32(&q, &k, &pos_array, n_heads, n_kv, hd, theta, b).map_err(|e| format!("glimmer prefill L{layer_idx} rope batched: {e:?}"))?;
            }
        }

        // This chunk writes positions [position, position+b). Map that prefix
        // before the batched write. Scoped so the mutable borrow ends before
        // the aliasing immutable borrow below. No-op on the contiguous backend.
        {
            let kv_mut = match cfg.layer_types[layer_idx] {
                GlimmerLayerType::Sliding => &mut state.kv_sliding,
                GlimmerLayerType::Full => &mut state.kv_full,
            };
            kv_mut
                .ensure_mapped_capacity(gpu, position as usize + b)
                .map_err(|e| format!("glimmer prefill L{layer_idx} kv map: {e:?}"))?;
        }
        let kv = match cfg.layer_types[layer_idx] {
            GlimmerLayerType::Sliding => &state.kv_sliding,
            GlimmerLayerType::Full => &state.kv_full,
        };
        let k_cache = unsafe { kv.k_gpu[slot].buf.alias() };
        let v_cache = unsafe { kv.v_gpu[slot].buf.alias() };
        let k_cache_t = GpuTensor { buf: k_cache, shape: kv.k_gpu[slot].shape.clone(), dtype: kv.k_gpu[slot].dtype };
        let v_cache_t = GpuTensor { buf: v_cache, shape: kv.v_gpu[slot].shape.clone(), dtype: kv.v_gpu[slot].dtype };
        gpu.kv_cache_write_q8_0_batched(&k_cache_t, &k, &pos_array, n_kv, hd, b).map_err(|e| format!("glimmer prefill L{layer_idx} kv write k: {e:?}"))?;
        gpu.kv_cache_write_q8_0_batched(&v_cache_t, &v, &pos_array, n_kv, hd, b).map_err(|e| format!("glimmer prefill L{layer_idx} kv write v: {e:?}"))?;

        // Attention: batched windowed flash for over-window sliding.
        // Full layers (window 0) and early chunks (seq_len <= window) stay on
        // the fast batched causal path. Sliding layers where seq_len > window
        // use the existing batched windowed Q8 flash kernel:
        //   attention_flash_q8_0_batched_masked_windowed with window=2048,
        //   tree_bias=None, block_start=0, block_cols=0.
        // Precedent: crates/hipfire-arch-cohere2moe/src/forward.rs:705-757.
        // Toggle HIPFIRE_GLIMMER_NO_FLASH=1 to force per-row SWA for debugging.
        //
        // The 13 NoPE (window==0) layers are the ONLY ones whose cost still
        // grows once past 2048 — the 39 sliding layers pin at the window. The
        // measured decay slopes bear that out: +116 µs/tok per 1k ctx under the
        // window vs +27 above, a 4.30x drop against a 52/13 = 4.00 layer ratio.
        // Those 13 layers were falling through to attention_q8_0_kv_batched for
        // no algorithmic reason: the flash kernel documents `window <= 0 = full
        // causal` (attention_flash_q8_0_tile_batched.hip:62, win_lo = window>0 ?
        // seq_len-window : 0), so passing window=0 gives exactly the causal
        // attention they already want.
        //
        // DEFAULT ON, and that is a REQUIREMENT, not a tuning choice. It was
        // briefly default-off on a perf reading that was true but incomplete:
        // at 4241 tokens it is neutral (7303 ms off vs 7319 on, -0.22%, inside
        // the 0.4% spread, byte-identical). Flash changes the CONSTANT and not
        // the SLOPE — full attention over a growing context is inherently
        // linear per token — so at short context there is nothing to win.
        //
        // But attention_q8_0_kv_batched sizes its launch by the sequence
        // length, and past roughly 16k that exceeds the 64 KB LDS limit. The
        // full (window==0) layers ALWAYS took that path, so prefill died with
        // `glimmer prefill L3 attention batched: hipModuleLaunchKernel:
        // invalid argument`. Measured: pp16384 fails outright with this off
        // and runs at 379.40 tok/s with it on. Same defect class as the decode
        // ceiling fixed in the SWA seq_len_hint.
        //
        // So it costs ~0.2% (inside noise) below the window and is the
        // difference between working and not above it. `=0` restores the
        // batched path and reintroduces the ceiling.
        let flash_full = std::env::var("HIPFIRE_GLIMMER_FLASH_FULL")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(true);
        let use_flash = std::env::var("HIPFIRE_GLIMMER_NO_FLASH").as_deref() != Ok("1")
            && ((window != 0 && seq_len > window)
                || (window == 0 && flash_full && seq_len > 2048));
        if use_flash {
            if state.prefill_flash_partials.is_none() {
                let n_tiles = (state.max_seq + 127) / 128;
                let flash_elems = cfg.n_heads * n_tiles * (2 + cfg.head_dim) * 64;
                let buf = gpu
                    .alloc_tensor(&[flash_elems], DType::F32)
                    .map_err(|e| format!("glimmer prefill flash partials alloc: {e:?}"))?;
                state.prefill_flash_partials = Some(buf);
            }
            let partials = state.prefill_flash_partials.as_ref().unwrap();

            // Window-bounded tile origin.
            //
            // The flash launcher sets grid depth to ceil(max_ctx_len/128) and
            // the kernel maps tile_id -> keys [tile_id*128, +128) from the
            // cache base, so a sliding layer launches a block per 128 keys of
            // the WHOLE context even though only `window` of them can be in
            // range. At ctx 16384 that is 128 blocks of which 112 immediately
            // take the empty-partial early-return: 88% pure launch overhead,
            // and the reduce still walks all 128 partials per head. Measured,
            // this is why Glimmer's over-window growth is 146.4 µs/tok per 1k
            // instead of the 75.8 that 13-of-52 growing layers predicts.
            //
            // Rebasing the KV pointer fixes it with NO kernel change: shift
            // keys, positions and seq_len by the same tile-aligned origin and
            // every comparison the kernel makes (causal t <= query position,
            // win_lo = seq_len - window) is unchanged, because all three are
            // relative. RoPE is already baked into the stored K, so nothing
            // else depends on absolute position, and tree_bias is None here.
            //
            // The origin must be (a) tile-aligned, else tile_start stops
            // landing on a key boundary, and (b) no greater than the SMALLEST
            // win_lo across the chunk, else the earliest row loses in-window
            // keys. Row i has seq_len = position+1+i, so the minimum is at
            // i = 0.
            //
            // FULL (window==0) LAYERS GO TO THE WMMA KERNEL INSTEAD.
            // attention_flash_q8_0_tile_batched is SCALAR for Q8 KV: the
            // WMMA-FA upgrade in launch_asym_flash_batched is gated on
            //     tile_func_name == "attention_flash_asym4_tile_batched"
            // (attention.rs), and there is no Q8 tile-batched WMMA kernel, so
            // Glimmer's Q8 cache silently never qualified. head_dim=128,
            // tree_bias=None and V_MODE_Q8 all pass; only the asym4 name test
            // fails. That is why prefill falls off as if WMMA were inactive —
            // it is inactive.
            //
            // attention_q8_0_flash_prefill_wmma IS a Q8 WMMA flash kernel and
            // is the default elsewhere; its dispatch site records 1.9x the
            // scalar flash kernel, 1.21x @2048 rising to 2.47x @12288. It
            // takes no `window`, so it is full-causal only — which is exactly
            // the 13 NoPE layers, and those are the ones whose cost grows
            // without bound. The 39 sliding layers still need masking and stay
            // on the tile path with the rebased origin below.
            //
            // It computes in f16 (relative L2 ~1e-3 vs the f32 reference), so
            // this is a real precision trade rather than a free win. DEFAULT
            // ON anyway, for two reasons. The engine already makes exactly this
            // trade for every other Q8 arch — the dispatch site defaults
            // `HIPFIRE_FLASH_PREFILL_KERNEL` to "wmma" — so holding Glimmer to
            // a stricter bar would be inconsistent, not careful. And measured
            // on all three prefill fixtures at 64 generated tokens, greedy
            // output is BYTE-IDENTICAL with it on and off (270/1080 ->
            // 1d5af38f9715, 4241 -> 6d71c90cfc07), so on this model the trade
            // costs nothing observable.
            //
            // `HIPFIRE_GLIMMER_WMMA_FULL=0` restores the scalar tile path.
            // Sliding layers get the Muse-owned windowed sibling; full layers
            // get the parent. Both are Q8 WMMA, so all 52 layers are now on
            // matrix hardware instead of 13 of them.
            let wmma_full = b > 1
                && std::env::var("HIPFIRE_GLIMMER_WMMA_FULL")
                    .map(|v| v != "0" && !v.is_empty())
                    .unwrap_or(true);
            let mut wmma_done = false;
            if wmma_full {
                if window == 0 {
                    gpu.attention_q8_0_flash_prefill_wmma(
                        &q,
                        &k_cache_t,
                        &v_cache_t,
                        &attn_out,
                        &pos_array,
                        n_heads,
                        n_kv,
                        hd,
                        b,
                    )
                    .map_err(|e| format!("glimmer prefill L{layer_idx} wmma full: {e:?}"))?;
                    wmma_done = true;
                } else {
                    wmma_done = gpu
                        .attention_q8_0_flash_prefill_wmma_swa(
                            &q,
                            &k_cache_t,
                            &v_cache_t,
                            &attn_out,
                            &pos_array,
                            n_heads,
                            n_kv,
                            hd,
                            b,
                            window,
                        )
                        .map_err(|e| format!("glimmer prefill L{layer_idx} wmma swa: {e:?}"))?;
                }
            }
            if !wmma_done {
            let k_numel = k_cache_t.numel();
            let v_numel = v_cache_t.numel();
            let mut k_view = k_cache_t;
            let mut v_view = v_cache_t;
            let mut pos_view = &pos_array;
            let mut eff_seq_len = seq_len;
            // Sliding layers share one origin, computed once per chunk above;
            // full (window==0) layers keep the unshifted base.
            let origin = if window > 0 { sliding_origin } else { 0 };
            if origin > 0 {
                // 272 B per token per layer (n_kv_heads * head_dim/32 * 34),
                // i.e. 68 F32 elements; tile-aligned origins stay exact.
                let bytes_per_token = n_kv * (hd / 32) * 34;
                let elem_off = origin * bytes_per_token / 4;
                // Size the view to the LIVE extent, not the virtual reserve.
                // The KV cache is VMM-backed, so only the mapped prefix is
                // accessible and sub_offset to numel trips the guard with
                // "tensor sub-view exceeds accessible buffer prefix". The
                // kernel clamps tile_end to seq_len, so it never reads past
                // (seq_len - origin) tokens anyway.
                let live_elems = (seq_len - origin) * bytes_per_token / 4;
                let k_len = live_elems.min(k_numel.saturating_sub(elem_off));
                let v_len = live_elems.min(v_numel.saturating_sub(elem_off));
                k_view = k_view.sub_offset(elem_off, k_len);
                v_view = v_view.sub_offset(elem_off, v_len);
                pos_view = shifted_pos_array
                    .as_ref()
                    .expect("sliding_origin > 0 implies shifted_pos_array");
                eff_seq_len = seq_len - origin;
            }
            gpu.attention_flash_q8_0_batched_masked_windowed(
                &q,
                &k_view,
                &v_view,
                &attn_out,
                pos_view,
                n_heads,
                n_kv,
                hd,
                state.max_seq,
                eff_seq_len,
                b,
                partials,
                None,
                0,
                0,
                window as i32,
            )
            .map_err(|e| format!("glimmer prefill L{layer_idx} flash windowed: {e:?}"))?;
            }
        } else if window != 0 && seq_len > window {
            // Per-row SWA fallback when flash disabled (debug) or if flash not available.
            for b_idx in 0..b {
                let pos = position + b_idx as u32;
                state.pos_host[0] = pos as i32;
                let pos_bytes = (pos as i32).to_ne_bytes();
                gpu.hip.memcpy_htod(&state.pos_buf, &pos_bytes).map_err(|e| format!("glimmer prefill L{layer_idx} htod pos row {b_idx}: {e:?}"))?;
                let q_row = q.sub_offset(b_idx * q_dim, q_dim);
                let out_row = attn_out.sub_offset(b_idx * q_dim, q_dim);
                gpu.attention_q8_0_kv_swa(&q_row, &k_cache_t, &v_cache_t, &out_row, &state.pos_buf, (pos as usize) + 1, n_heads, n_kv, hd, phys_cap, window).map_err(|e| format!("glimmer prefill L{layer_idx} attention swa row {b_idx}: {e:?}"))?;
            }
        } else {
            gpu.attention_q8_0_kv_batched(&q, &k_cache_t, &v_cache_t, &attn_out, &pos_array, n_heads, n_kv, hd, phys_cap, seq_len, b).map_err(|e| format!("glimmer prefill L{layer_idx} attention batched: {e:?}"))?;
        }

        if !abl("HIPFIRE_GLIMMER_NO_ATTN_GATE") {
            gpu.sigmoid_mul_f32(&attn_out, &attn_gate).map_err(|e| format!("glimmer prefill L{layer_idx} sigmoid_mul: {e:?}"))?;
        }

        // o_proj (M=6656, K=4096). The old comment here read "stays on the
        // current baseline (29.8 TF best measured)" — that number is the
        // GemmHfq4G256BatchedLmhead path this used to take, NOT the residual
        // kernel. Re-measured on gfx1201 at B=384 on this exact shape, the
        // residual family runs 46.2 TF shared / 51.7 TF muse, so the plain
        // lmhead route was leaving ~1.7x on the floor. Residual needs Y
        // pre-zeroed (it accumulates), same dance as down_proj below.
        //
        // Default ON. It was briefly parked off on two claims that were both
        // wrong: a "1.84x end-to-end cost" that was really hipRTC compiling the
        // Muse kernel on first run, and an "output changed" that was really a
        // stale golden. Re-measured warm on gfx1201, median of 2 fresh
        // processes per config: 1372 -> 1356 ms at 1080 (+1.2%) and 7366 ->
        // 7310 ms at 4241 (+0.8%), byte-identical both lengths. Env opt-out
        // HIPFIRE_GLIMMER_O_RESIDUAL=0 kept for A/B.
        let o_use_residual = std::env::var("HIPFIRE_GLIMMER_O_RESIDUAL")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(true)
            && b > 1
            && gpu.arch_caps.has_wmma_w32_gfx12()
            && matches!(lw.o_proj.gpu_dtype, DType::MQ4G256 | DType::HFQ4G256);
        // No invalidate_fp16_cache() needed here, unlike the gate/up site
        // below: that cache keys purely on source pointer (scratch.rs:411) and
        // the per-layer buffer order is x_rot (qkvza) -> o_rot (here) -> x_rot
        // (gate) -> x_rot (up, the intended reuse) -> down_rot, so o_rot is
        // always preceded by a DIFFERENT pointer.
        if o_use_residual {
            rotate_x_mq_batched_for(gpu, &lw.o_proj, &attn_out, &o_rot, lw.o_proj.k, b)
                .map_err(|e| format!("glimmer prefill L{layer_idx} rotate for o residual: {e:?}"))?;
            gpu.hip
                .memset(&o_out.buf, 0, b * lw.o_proj.m * 4)
                .map_err(|e| format!("glimmer prefill L{layer_idx} zero o_out: {e:?}"))?;
            run_prefill_residual_muse_or_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                &lw.o_proj.buf,
                lw.o_proj.gpu_dtype,
                &o_rot,
                &o_out,
                lw.o_proj.m,
                lw.o_proj.k,
                b,
            )
            .map_err(|e| format!("glimmer prefill L{layer_idx} o residual: {e}"))?;
        } else {
            prefill_proj_gemm_batched(gpu, &lw.o_proj, &attn_out, &o_out, &o_rot, b, "o_proj")?;
        }
        gpu.rmsnorm_batched(&o_out, &lw.post_attention_layernorm, &o_out, b, dim, post_eps).map_err(|e| format!("glimmer prefill L{layer_idx} post_attn rmsnorm: {e:?}"))?;
        gpu.hip.memcpy_dtod_at(&x.buf, 0, &residual.buf, 0, b * dim * 4).map_err(|e| format!("glimmer prefill L{layer_idx} reset x (attn): {e:?}"))?;
        gpu.add_inplace_f32(&x, &o_out).map_err(|e| format!("glimmer prefill L{layer_idx} attn residual add: {e:?}"))?;
        gpu.hip.memcpy_dtod_at(&residual.buf, 0, &x.buf, 0, b * dim * 4).map_err(|e| format!("glimmer prefill L{layer_idx} save ffn residual: {e:?}"))?;

        // ── FFN gate/up: residual_wmma_gfx12 on gfx12 ──
        // Microbench: gemm_hfq4g256_residual_wmma_gfx12 @ B384 = 65 TF, 1.11x baseline.
        // Plain GEMM (overwrite) via residual kernel: zero y then y += W·x_rot.
        // Arch-gated so non-gfx12 keeps the batched-lmhead path.
        let ffn_shared = shared_rot_enabled()
            && hipfire_dispatch::types::dtype_rotation_plan(lw.gate_proj.gpu_dtype) != hipfire_dispatch::types::RotationPlan::None;
        let ffn_use_residual = b > 1
            && gpu.arch_caps.has_wmma_w32_gfx12()
            && matches!(lw.gate_proj.gpu_dtype, DType::MQ4G256 | DType::HFQ4G256)
            && lw.gate_proj.gpu_dtype == lw.up_proj.gpu_dtype;
        if ffn_use_residual {
            if ffn_shared {
                fused_rmsnorm_rotate_mq_batched_for(
                    gpu, &x, &lw.pre_feedforward_layernorm, &lw.gate_proj, &x_rot, dim, rms_eps, b,
                ).map_err(|e| format!("glimmer prefill L{layer_idx} fused pre_ffn rmsnorm+rotate: {e:?}"))?;
            } else {
                gpu.rmsnorm_batched(&x, &lw.pre_feedforward_layernorm, &nrm, b, dim, rms_eps).map_err(|e| format!("glimmer prefill L{layer_idx} pre_ffn rmsnorm: {e:?}"))?;
                rotate_x_mq_batched_for(gpu, &lw.gate_proj, &nrm, &x_rot, dim, b).map_err(|e| format!("glimmer prefill L{layer_idx} rotate for ffn gate/up residual: {e:?}"))?;
            }
            // `x_rot` was just overwritten with FFN-normed content, but it is the
            // SAME device pointer the attention qkvza call rotated into earlier in
            // this layer. `ensure_fp16_x` keys its F32->F16 conversion purely on
            // that pointer (scratch.rs: `must_convert = capture_mode ||
            // fp16_x_source_ptr != src_ptr`), so a cached hit here would feed
            // gate/up the ATTENTION-normed activations. Invalidate explicitly.
            //
            // This is dead weight on RDNA today: the only `ensure_fp16_x` in the
            // residual path sits behind `rocblas_arch_eligible()`, which is
            // `is_cdna3()`, so gfx11/gfx12 never reach it. It becomes live on
            // CDNA3 or under the `rocblas_all_archs` flag, and the failure mode
            // would be silent wrong activations rather than an error — so the
            // guard is cheap insurance, not redundancy.
            //
            // Deliberately placed BEFORE gate rather than between gate and up:
            // those two share one rotation of identical content, so up's cache
            // hit is correct and is the win we want to keep.
            gpu.invalidate_fp16_cache();
            gpu.hip.memset(&gate_ffn.buf, 0, b * lw.gate_proj.m * 4).map_err(|e| format!("glimmer prefill L{layer_idx} zero gate_ffn: {e:?}"))?;
            run_prefill_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                &lw.gate_proj.buf,
                lw.gate_proj.gpu_dtype,
                &x_rot,
                &gate_ffn,
                lw.gate_proj.m,
                lw.gate_proj.k,
                b,
            ).map_err(|e| format!("glimmer prefill L{layer_idx} gate residual: {e}"))?;
            gpu.hip.memset(&up_ffn.buf, 0, b * lw.up_proj.m * 4).map_err(|e| format!("glimmer prefill L{layer_idx} zero up_ffn: {e:?}"))?;
            run_prefill_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                &lw.up_proj.buf,
                lw.up_proj.gpu_dtype,
                &x_rot,
                &up_ffn,
                lw.up_proj.m,
                lw.up_proj.k,
                b,
            ).map_err(|e| format!("glimmer prefill L{layer_idx} up residual: {e}"))?;
        } else if ffn_shared {
            fused_rmsnorm_rotate_mq_batched_for(
                gpu, &x, &lw.pre_feedforward_layernorm, &lw.gate_proj, &x_rot, dim, rms_eps, b,
            ).map_err(|e| format!("glimmer prefill L{layer_idx} fused pre_ffn rmsnorm+rotate: {e:?}"))?;
            prefill_proj_gemm_batched_prerotated(gpu, &lw.gate_proj, &nrm, &x_rot, &gate_ffn, b, "gate_proj")?;
            prefill_proj_gemm_batched_prerotated(gpu, &lw.up_proj, &nrm, &x_rot, &up_ffn, b, "up_proj")?;
        } else {
            gpu.rmsnorm_batched(&x, &lw.pre_feedforward_layernorm, &nrm, b, dim, rms_eps).map_err(|e| format!("glimmer prefill L{layer_idx} pre_ffn rmsnorm: {e:?}"))?;
            prefill_proj_gemm_batched(gpu, &lw.gate_proj, &nrm, &gate_ffn, &x_rot, b, "gate_proj")?;
            prefill_proj_gemm_batched(gpu, &lw.up_proj, &nrm, &up_ffn, &x_rot, b, "up_proj")?;
        }
        gpu.silu_mul_f32(&gate_ffn, &up_ffn, &ffn_hidden).map_err(|e| format!("glimmer prefill L{layer_idx} silu_mul: {e:?}"))?;
        // ── down_proj: residual_wmma_gfx12 on gfx12 ──
        let down_use_residual = b > 1
            && gpu.arch_caps.has_wmma_w32_gfx12()
            && matches!(lw.down_proj.gpu_dtype, DType::MQ4G256 | DType::HFQ4G256);
        if down_use_residual {
            // ffn_hidden is [B, hidden_dim]; down needs FWHT-rotated input for MQ4.
            rotate_x_mq_batched_for(gpu, &lw.down_proj, &ffn_hidden, &down_rot, lw.down_proj.k, b).map_err(|e| format!("glimmer prefill L{layer_idx} rotate for down residual: {e:?}"))?;
            gpu.hip.memset(&ffn_out.buf, 0, b * lw.down_proj.m * 4).map_err(|e| format!("glimmer prefill L{layer_idx} zero ffn_out: {e:?}"))?;
            run_prefill_residual_muse_or_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                &lw.down_proj.buf,
                lw.down_proj.gpu_dtype,
                &down_rot,
                &ffn_out,
                lw.down_proj.m,
                lw.down_proj.k,
                b,
            ).map_err(|e| format!("glimmer prefill L{layer_idx} down residual: {e}"))?;
        } else {
            prefill_proj_gemm_batched(gpu, &lw.down_proj, &ffn_hidden, &ffn_out, &down_rot, b, "down_proj")?;
        }
        gpu.rmsnorm_batched(&ffn_out, &lw.post_feedforward_layernorm, &ffn_out, b, dim, post_eps).map_err(|e| format!("glimmer prefill L{layer_idx} post_ffn rmsnorm: {e:?}"))?;
        gpu.hip.memcpy_dtod_at(&x.buf, 0, &residual.buf, 0, b * dim * 4).map_err(|e| format!("glimmer prefill L{layer_idx} reset x (ffn): {e:?}"))?;
        gpu.add_inplace_f32(&x, &ffn_out).map_err(|e| format!("glimmer prefill L{layer_idx} ffn residual add: {e:?}"))?;
        if let Some(ci) = cap_index[layer_idx] {
            let host = gpu.download_f32(&x).map_err(|e| format!("glimmer prefill capture L{layer_idx}: {e:?}"))?;
            for row in 0..b {
                let src_off = row * dim;
                let dst_off = (row * cap_cnt + ci) * dim;
                cap_buf[dst_off..dst_off + dim].copy_from_slice(&host[src_off..src_off + dim]);
            }
        }
    }
    if cap_cnt > 0 {
        hidden_out.extend_from_slice(&cap_buf);
    }
    state.n_tokens = seq_len;

    // ── Final norm + lm_head: ONLY last row when needed ──
    let last_logits: Option<Vec<f32>> = if need_logits {
        let normed = alloc(gpu, b * dim, "normed")?;
        // Same reason as the scratch set: `normed` used to leak on any of the five `?`
        // sites below, and this branch runs on the LAST chunk of every prefill.
        let host_result: Result<Vec<f32>, String> = (|| {
            gpu.rmsnorm_batched(&x, &weights.final_norm, &normed, b, dim, rms_eps).map_err(|e| format!("glimmer prefill final rmsnorm: {e:?}"))?;
            let last_norm = normed.sub_offset((b - 1) * dim, dim);
            weight_gemv(gpu, &weights.lm_head, &last_norm, &state.logits)
                .map_err(|e| format!("glimmer prefill lm_head: {e}"))?;
            if cfg.output_multiplier != 1.0 && !abl("HIPFIRE_GLIMMER_NO_OUTMUL") {
                gpu.scale_f32(&state.logits, cfg.output_multiplier).map_err(|e| format!("glimmer prefill outmul: {e:?}"))?;
            }
            if cfg.final_logit_softcapping > 0.0 && !abl("HIPFIRE_GLIMMER_NO_SOFTCAP") {
                gpu.logit_softcap_f32(&state.logits, cfg.vocab_size, cfg.final_logit_softcapping).map_err(|e| format!("glimmer prefill softcap: {e:?}"))?;
            }
            gpu.download_f32(&state.logits).map_err(|e| format!("glimmer prefill download logits: {e:?}"))
        })();
        gpu.free_tensor(normed).ok();
        Some(host_result?)
    } else { None };

    Ok(last_logits)
    })();

    // Free batched scratch on EVERY path — success, `?`, or the early `return Err` in the
    // embedding match. This is the single release site the closure above exists to reach.
    for t in [x, residual, nrm, x_rot, q, k, v, attn_gate, attn_out, o_rot, o_out, gate_ffn, up_ffn, ffn_hidden, ffn_out, down_rot, pos_array] {
        gpu.free_tensor(t).ok();
    }

    body_result
}

/// Chunked batched prefill entry point. Processes `prompt` in chunks of
/// `HIPFIRE_GLIMMER_PREFILL_CHUNK` (default 256, env overridable, 1-512).
/// Reuses the batched layer machinery from `verify_block_with_capture`
/// (same `proj_gemm_batched` helpers, same fused rotate, same attention)
/// but over larger B. Returns the last prompt token's logits, which seed
/// greedy decode. When `capture_layers` is non-empty, `hidden_out` is
/// extended with `B*len(capture_layers)*dim` floats per chunk in ascending
/// tap-layer order, position-major — identical to the per-token
/// `decode_step_with_capture` contract.
///
/// Intermediate chunks skip the lm_head entirely (only the final position's
/// logits are needed to seed generation). This is the "do NOT run it over
/// the whole chunk" optimization: we run one `weight_gemv` on the last row
/// instead of a batched GEMM over B rows, streaming the 700 MB lm_head once
/// instead of B times.
pub fn prefill_with_capture(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    prompt: &[u32],
    start_pos: u32,
    capture_layers: &[usize],
    hidden_out: &mut Vec<f32>,
) -> Result<Vec<f32>, String> {
    if prompt.is_empty() {
        return Err("glimmer prefill: empty prompt".to_string());
    }
    let chunk_size = glimmer_prefill_chunk_size(prompt.len());
    let mut last_logits: Option<Vec<f32>> = None;
    let mut pos = start_pos;
    let mut offset = 0usize;
    while offset < prompt.len() {
        let end = (offset + chunk_size).min(prompt.len());
        let chunk = &prompt[offset..end];
        let is_last = end == prompt.len();
        let logits_opt = prefill_chunk_batched(cfg, weights, state, gpu, chunk, pos, capture_layers, hidden_out, is_last)?;
        if let Some(l) = logits_opt { last_logits = Some(l); }
        pos += chunk.len() as u32;
        offset = end;
    }
    last_logits.ok_or_else(|| "glimmer prefill: no logits produced".to_string())
}

