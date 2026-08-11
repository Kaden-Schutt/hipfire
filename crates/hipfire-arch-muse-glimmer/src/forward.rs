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
use crate::glimmer::{GlimmerState, GlimmerWeights};
use hipfire_runtime::llama::{
    fused_rmsnorm_rotate_for_mq, weight_gemv, weight_gemv_prerotated, EmbeddingFormat,
};
use rdna_compute::Gpu;

// ───────────────────── Shared rotation gate ─────────────────────
// HIPFIRE_GLIMMER_SHARED_ROT default ON (=1 or unset), =0 selects old path.
// Mirrors Gemma4's attn_input_qkv and Qwen35's run_fa_layer_body precedent.
fn shared_rot_enabled() -> bool {
    std::env::var("HIPFIRE_GLIMMER_SHARED_ROT").as_deref() != Ok("0")
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
    gpu.kv_cache_write_q8_0(&kv.k_gpu[kv_slot], &state.k, &state.pos_buf, n_kv, head_dim)
        .map_err(|e| format!("glimmer L{layer_idx}: kv write k: {e:?}"))?;
    gpu.kv_cache_write_q8_0(&kv.v_gpu[kv_slot], &state.v, &state.pos_buf, n_kv, head_dim)
        .map_err(|e| format!("glimmer L{layer_idx}: kv write v: {e:?}"))?;
    gpu.attention_q8_0_kv_swa(
        &state.q,
        &kv.k_gpu[kv_slot],
        &kv.v_gpu[kv_slot],
        &state.attn_out,
        &state.pos_buf,
        state.max_seq,
        n_heads,
        n_kv,
        head_dim,
        kv.physical_cap,
        window,
    )
    .map_err(|e| format!("glimmer L{layer_idx}: attention swa: {e:?}"))?;

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

/// Block verify that also captures hidden at `capture_layers` for each
/// position in the block, appending to `hidden_out` (same order as
/// decode_step_with_capture). Used by the DFlash spec loop to keep
/// `bundle.target_hidden_host` in sync with the committed prefix.
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
    let mut picks = Vec::with_capacity(block.len());
    for (i, &tok) in block.iter().enumerate() {
        let pos = position + i as u32;
        let logits = if !capture_layers.is_empty() {
            decode_step_with_capture(cfg, weights, state, gpu, tok, pos, capture_layers, hidden_out)?
        } else {
            decode_step(cfg, weights, state, gpu, tok, pos)?
        };
        let pick = logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(idx,_)| idx as u32).unwrap();
        picks.push(pick);
    }
    Ok(picks)
}
