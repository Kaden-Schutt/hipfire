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

/// Chunk size for batched prefill. Default 256, overridable via
/// `HIPFIRE_GLIMMER_PREFILL_CHUNK` for tuning (128-512 range).
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
///   We default to 256 as the mid-point and clamp env to [1,512].
pub fn glimmer_prefill_chunk_size() -> usize {
    std::env::var("HIPFIRE_GLIMMER_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(256)
        .clamp(1, 512)
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
    let x = alloc(gpu, b * dim, "x")?;
    let residual = alloc(gpu, b * dim, "residual")?;
    let nrm = alloc(gpu, b * dim, "nrm")?;
    let x_rot = alloc(gpu, b * dim, "x_rot")?;
    let q = alloc(gpu, b * q_dim, "q")?;
    let k = alloc(gpu, b * kv_dim, "k")?;
    let v = alloc(gpu, b * kv_dim, "v")?;
    let attn_gate = alloc(gpu, b * q_dim, "attn_gate")?;
    let attn_out = alloc(gpu, b * q_dim, "attn_out")?;
    let o_rot = alloc(gpu, b * q_dim, "o_rot")?;
    let o_out = alloc(gpu, b * dim, "o_out")?;
    let gate_ffn = alloc(gpu, b * cfg.hidden_dim, "gate_ffn")?;
    let up_ffn = alloc(gpu, b * cfg.hidden_dim, "up_ffn")?;
    let ffn_hidden = alloc(gpu, b * cfg.hidden_dim, "ffn_hidden")?;
    let ffn_out = alloc(gpu, b * dim, "ffn_out")?;
    let down_rot = alloc(gpu, b * cfg.hidden_dim, "down_rot")?;

    // positions [B] i32
    let pos_data: Vec<i32> = (0..b).map(|i| (position + i as u32) as i32).collect();
    let pos_bytes: Vec<u8> = pos_data.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let pos_array = alloc(gpu, b, "pos_array")?;
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

        // Batched attention with mask (causal + sliding window)
        // causal batched attention (window 2048 == max_seq 2048 for this benchmark, so window not restrictive;
        // for seq_len <= window, swa(window) is exactly causal, and NoPE layers already skip rope).
        // If window < seq_len later, switch to masked variant with a per-row window mask (block_start=start_pos).
        gpu.attention_q8_0_kv_batched(&q, &k_cache_t, &v_cache_t, &attn_out, &pos_array, n_heads, n_kv, hd, phys_cap, seq_len, b).map_err(|e| format!("glimmer batch L{layer_idx} attention batched: {e:?}"))?;

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
/// For sliding layers (window=2048) where `seq_len > window`, the
/// windowed mask cannot be expressed with the existing
/// `attention_q8_0_kv_batched` / `attention_q8_0_kv_batched_masked`
/// kernels (masked kernel is tree-mode and leaves prefix unmasked,
/// so it cannot hide old prefix tokens outside the window). No new
/// kernel is added per constraints; instead those layers fall back to
/// per-row `attention_q8_0_kv_swa(window)` — byte-identical to the
/// single-token path and honest about the win. Full layers and early
/// chunks where `seq_len <= window` stay on the fast batched causal
/// path. This is the "if cannot be expressed, fallback" honest path
/// the contract allows, not silent garbage.
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
    let x = alloc(gpu, b * dim, "x")?;
    let residual = alloc(gpu, b * dim, "residual")?;
    let nrm = alloc(gpu, b * dim, "nrm")?;
    let x_rot = alloc(gpu, b * dim, "x_rot")?;
    let q = alloc(gpu, b * q_dim, "q")?;
    let k = alloc(gpu, b * kv_dim, "k")?;
    let v = alloc(gpu, b * kv_dim, "v")?;
    let attn_gate = alloc(gpu, b * q_dim, "attn_gate")?;
    let attn_out = alloc(gpu, b * q_dim, "attn_out")?;
    let o_rot = alloc(gpu, b * q_dim, "o_rot")?;
    let o_out = alloc(gpu, b * dim, "o_out")?;
    let gate_ffn = alloc(gpu, b * cfg.hidden_dim, "gate_ffn")?;
    let up_ffn = alloc(gpu, b * cfg.hidden_dim, "up_ffn")?;
    let ffn_hidden = alloc(gpu, b * cfg.hidden_dim, "ffn_hidden")?;
    let ffn_out = alloc(gpu, b * dim, "ffn_out")?;
    let down_rot = alloc(gpu, b * cfg.hidden_dim, "down_rot")?;

    let pos_data: Vec<i32> = (0..b).map(|i| (position + i as u32) as i32).collect();
    let pos_bytes: Vec<u8> = pos_data.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let pos_array = alloc(gpu, b, "pos_array")?;
    gpu.hip
        .memcpy_htod(&pos_array.buf, &pos_bytes)
        .map_err(|e| format!("glimmer prefill htod pos: {e:?}"))?;

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

        let need_shared = shared_rot_enabled()
            && hipfire_dispatch::types::dtype_rotation_plan(lw.q_proj.gpu_dtype) != hipfire_dispatch::types::RotationPlan::None;
        if need_shared {
            fused_rmsnorm_rotate_mq_batched_for(
                gpu, &x, &lw.input_layernorm, &lw.q_proj, &x_rot, dim, rms_eps, b,
            ).map_err(|e| format!("glimmer prefill L{layer_idx} fused input rmsnorm+rotate: {e:?}"))?;
            proj_gemm_batched_prerotated(gpu, &lw.q_proj, &nrm, &x_rot, &q, b, "q_proj")?;
            proj_gemm_batched_prerotated(gpu, &lw.k_proj, &nrm, &x_rot, &k, b, "k_proj")?;
            proj_gemm_batched_prerotated(gpu, &lw.v_proj, &nrm, &x_rot, &v, b, "v_proj")?;
            proj_gemm_batched_prerotated(gpu, &lw.attn_gate_proj, &nrm, &x_rot, &attn_gate, b, "attn_gate")?;
        } else {
            gpu.rmsnorm_batched(&x, &lw.input_layernorm, &nrm, b, dim, rms_eps).map_err(|e| format!("glimmer prefill L{layer_idx} input rmsnorm: {e:?}"))?;
            proj_gemm_batched(gpu, &lw.q_proj, &nrm, &q, &x_rot, b, "q_proj")?;
            proj_gemm_batched(gpu, &lw.k_proj, &nrm, &k, &x_rot, b, "k_proj")?;
            proj_gemm_batched(gpu, &lw.v_proj, &nrm, &v, &x_rot, b, "v_proj")?;
            proj_gemm_batched(gpu, &lw.attn_gate_proj, &nrm, &attn_gate, &x_rot, b, "attn_gate")?;
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

        // Attention: window-aware dispatch — honest fallback for over-window sliding.
        // Full layers (window 0) and early chunks (seq_len <= window) use fast batched causal.
        // Sliding layers where seq_len > window cannot be expressed with the existing
        // batched kernels (tree mask leaves prefix unmasked), so fall back to per-row
        // attention_q8_0_kv_swa which is byte-identical to the single-token path.
        if window != 0 && seq_len > window {
            // Per-row SWA — B launches, but projections are already batched.
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

        proj_gemm_batched(gpu, &lw.o_proj, &attn_out, &o_out, &o_rot, b, "o_proj")?;
        gpu.rmsnorm_batched(&o_out, &lw.post_attention_layernorm, &o_out, b, dim, post_eps).map_err(|e| format!("glimmer prefill L{layer_idx} post_attn rmsnorm: {e:?}"))?;
        gpu.hip.memcpy_dtod_at(&x.buf, 0, &residual.buf, 0, b * dim * 4).map_err(|e| format!("glimmer prefill L{layer_idx} reset x (attn): {e:?}"))?;
        gpu.add_inplace_f32(&x, &o_out).map_err(|e| format!("glimmer prefill L{layer_idx} attn residual add: {e:?}"))?;
        gpu.hip.memcpy_dtod_at(&residual.buf, 0, &x.buf, 0, b * dim * 4).map_err(|e| format!("glimmer prefill L{layer_idx} save ffn residual: {e:?}"))?;

        let ffn_shared = shared_rot_enabled()
            && hipfire_dispatch::types::dtype_rotation_plan(lw.gate_proj.gpu_dtype) != hipfire_dispatch::types::RotationPlan::None;
        if ffn_shared {
            fused_rmsnorm_rotate_mq_batched_for(
                gpu, &x, &lw.pre_feedforward_layernorm, &lw.gate_proj, &x_rot, dim, rms_eps, b,
            ).map_err(|e| format!("glimmer prefill L{layer_idx} fused pre_ffn rmsnorm+rotate: {e:?}"))?;
            proj_gemm_batched_prerotated(gpu, &lw.gate_proj, &nrm, &x_rot, &gate_ffn, b, "gate_proj")?;
            proj_gemm_batched_prerotated(gpu, &lw.up_proj, &nrm, &x_rot, &up_ffn, b, "up_proj")?;
        } else {
            gpu.rmsnorm_batched(&x, &lw.pre_feedforward_layernorm, &nrm, b, dim, rms_eps).map_err(|e| format!("glimmer prefill L{layer_idx} pre_ffn rmsnorm: {e:?}"))?;
            proj_gemm_batched(gpu, &lw.gate_proj, &nrm, &gate_ffn, &x_rot, b, "gate_proj")?;
            proj_gemm_batched(gpu, &lw.up_proj, &nrm, &up_ffn, &x_rot, b, "up_proj")?;
        }
        gpu.silu_mul_f32(&gate_ffn, &up_ffn, &ffn_hidden).map_err(|e| format!("glimmer prefill L{layer_idx} silu_mul: {e:?}"))?;
        proj_gemm_batched(gpu, &lw.down_proj, &ffn_hidden, &ffn_out, &down_rot, b, "down_proj")?;
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
        let host = gpu.download_f32(&state.logits).map_err(|e| format!("glimmer prefill download logits: {e:?}"))?;
        gpu.free_tensor(normed).ok();
        Some(host)
    } else { None };

    // Free batched scratch
    for t in [x, residual, nrm, x_rot, q, k, v, attn_gate, attn_out, o_rot, o_out, gate_ffn, up_ffn, ffn_hidden, ffn_out, down_rot, pos_array] {
        gpu.free_tensor(t).ok();
    }

    Ok(last_logits)
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
    let chunk_size = glimmer_prefill_chunk_size();
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

