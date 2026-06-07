// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 dense forward pass (free functions — hot-path static dispatch).
//!
//! Ported from the old branch's `forward_scratch` / `sliding_layer_decode` /
//! `full_layer_decode`, dropping MoE / E-series / vision. Per-token pipeline:
//!
//!   x = embed(token) * sqrt(dim)
//!   for each layer (sandwich RMSNorm around BOTH attn and FFN):
//!     residual = x
//!     n1 = input_layernorm(x)
//!     q = q_proj(n1); k = k_proj(n1)
//!       full + attention_k_eq_v: V = copy of k BEFORE k_norm, then weight-less
//!         RMSNorm on V (ones buffer); sliding: V = v_proj(n1)
//!     per-head q_norm / k_norm over head_dim; q *= sqrt(head_dim) (Gemma
//!       scale = 1.0 vs the kernel's 1/sqrt)
//!     RoPE: sliding → rope_f32(theta 10000, full rotate-half);
//!            full   → rope_partial_halved_f32(theta 1e6, n_rot = head_dim*0.25/2)
//!     KV write (Q8); attention_q8_0_kv_swa(window 1024 sliding / 0 full)
//!     attn = o_proj(attn_out); attn = post_attention_layernorm(attn)
//!     x = residual + attn
//!     residual = x
//!     n2 = pre_feedforward_layernorm(x)
//!     ffn = gelu_tanh(gate_proj(n2)) * up_proj(n2); ffn = down_proj(ffn)
//!     ffn = post_feedforward_layernorm(ffn)
//!     x = residual + ffn
//!     x *= layer_scalar
//!   x = norm(x); logits = lm_head(x); logits = logit_softcap(logits, 30)
//!
//! All RMSNorm here is plain `x * w` (baked at load — see `load_norm`).

use crate::config::{Gemma4Config, LayerType, RopeType};
use crate::gemma4::{FullLayerWeights, Gemma4State, Gemma4Weights, LayerWeights, SlidingLayerWeights};
use hipfire_runtime::llama::{weight_gemv, KvCache};
use rdna_compute::{DType, Gpu};

/// Master switch for the qwen35-mirror fused-projection FFN path
/// (`fused_rmsnorm_rotate_mq` + `fused_gate_up_hfq4g256`). Default ON; opt out
/// with `HIPFIRE_GEMMA4_FUSED_FFN=0`. Only fires when the FFN gate/up weights
/// are MQ4G256 (byte-compatible with the HFQ4G256 fused kernel given a
/// pre-FWHT-rotated input).
fn fused_ffn_enabled() -> bool {
    !matches!(
        std::env::var("HIPFIRE_GEMMA4_FUSED_FFN").ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

/// Master switch for the fused Q8 q+k projection path
/// (`fused_gate_up_q8_0`, 2 Q8 GEMVs → 1 launch, shared rmsnorm input).
/// Default ON; opt out with `HIPFIRE_GEMMA4_FUSED_QK=0`. gemma4 attention is
/// Q8 (no Q8 fused-QKV decode kernel exists), so we fuse q+k via the 2-way
/// Q8 gate_up fuser and leave v / o separate. Coherence is preserved (the
/// fused kernel is byte-equivalent to two separate Q8 GEMVs).
fn fused_qk_enabled() -> bool {
    !matches!(
        std::env::var("HIPFIRE_GEMMA4_FUSED_QK").ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

/// q = q_proj(x); k = k_proj(x). Fused into one launch via `fused_gate_up_q8_0`
/// when both are Q8_0 (same input `x`); else two `weight_gemv` calls.
fn qk_proj(
    gpu: &mut Gpu,
    q_proj: &hipfire_runtime::llama::WeightTensor,
    k_proj: &hipfire_runtime::llama::WeightTensor,
    x: &rdna_compute::GpuTensor,
    q_out: &rdna_compute::GpuTensor,
    k_out: &rdna_compute::GpuTensor,
) -> Result<(), String> {
    let both_q8 =
        q_proj.gpu_dtype == DType::Q8_0 && k_proj.gpu_dtype == DType::Q8_0;
    if fused_qk_enabled() && both_q8 {
        gpu.fused_gate_up_q8_0(
            &q_proj.buf,
            &k_proj.buf,
            x,
            q_out,
            k_out,
            q_proj.m,
            k_proj.m,
            q_proj.k,
        )
        .map_err(|e| format!("gemma4: fused q+k: {e:?}"))
    } else {
        weight_gemv(gpu, q_proj, x, q_out).map_err(|e| format!("gemma4: q_proj: {e}"))?;
        weight_gemv(gpu, k_proj, x, k_out).map_err(|e| format!("gemma4: k_proj: {e}"))
    }
}

/// Decode one token (eager); returns the full logits vector. Used for prefill,
/// the warm pass, and as the `HIPFIRE_GEMMA4_GRAPH=0` fallback.
pub fn decode_step(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    embed_lookup(cfg, weights, state, gpu, token_id)?;
    decode_step_body(cfg, weights, state, gpu, position, None)?;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("gemma4: download logits: {e:?}"))
}

/// Decode one token, appending each layer's post-residual hidden state (pre
/// final-norm) to `capture[layer]` — used by the oracle dumper. Eager only
/// (the per-layer D2H downloads are incompatible with graph capture).
pub fn decode_step_capture(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    capture: &mut [Vec<f32>],
) -> Result<(), String> {
    embed_lookup(cfg, weights, state, gpu, token_id)?;
    decode_step_body(cfg, weights, state, gpu, position, Some(capture))
}

/// Decode one token via hipGraph capture/replay. **Opt-in, default OFF**
/// (`HIPFIRE_GEMMA4_GRAPH=1` to enable). The 48-layer body + final-norm +
/// lm_head are captured once and replayed per token, recovering the per-token
/// host launch overhead. This is the biggest launch-bound lever on gemma4
/// decode (~720 kernel launches/token).
///
/// Capture-safety invariants (mirrors the proven MiniMax / DeepSeek-V4 path):
///   - token_id is per-token → embedding lookup + √dim scale run OUTSIDE the
///     capture (token_id is baked into the embedding kernarg).
///   - position is per-token → staged via `state.pos_host` (stable `Box`); the
///     captured `memcpy_htod_auto` re-reads it on every replay.
///   - attention launch geometry is sized for `state.max_seq` (constant), not
///     the live seq_len, so the baked grid/shared-mem stays valid as the KV
///     length grows (the kernel reads the true length from `pos_buf[0]+1`).
pub fn decode_step_with_graph(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    use std::sync::OnceLock;
    static GRAPH_ENV: OnceLock<Option<bool>> = OnceLock::new();
    let env_override =
        *GRAPH_ENV.get_or_init(|| match std::env::var("HIPFIRE_GEMMA4_GRAPH").ok().as_deref() {
            Some("1") => Some(true),
            Some("0") => Some(false),
            _ => None,
        });
    let graph_on = env_override.unwrap_or(false);
    if !graph_on {
        return decode_step(cfg, weights, state, gpu, token_id, position);
    }

    // Warmup: first decode after a fresh load runs eager (JITs kernels + settles
    // DPM) and drops any stale graph so the next call captures fresh for THIS
    // model's weight pointers / device buffers.
    if !state.ar_warmed_up {
        state.ar_warmed_up = true;
        gpu.graphs.graph_exec = None;
        return decode_step(cfg, weights, state, gpu, token_id, position);
    }

    // Capture + replay both need an explicit (non-null) stream.
    if gpu.active_stream.is_none() {
        let s = gpu
            .hip
            .stream_create()
            .map_err(|e| format!("gemma4 graph: stream_create: {e:?}"))?;
        gpu.active_stream = Some(s);
    }

    // Embedding lookup + √dim scale OUTSIDE the captured region — token_id is
    // baked into the embedding kernarg. Runs on the active stream, ordered
    // before the captured body that reads `state.x`.
    embed_lookup(cfg, weights, state, gpu, token_id)?;

    if gpu.graphs.graph_exec.is_none() {
        // ── Capture phase ──────────────────────────────────────────────
        // decode_step_body stages pos_host → pos_buf via memcpy_htod_auto
        // INSIDE the capture, so the recorded memcpy node re-reads pos_host
        // on each replay.
        gpu.graphs
            .begin_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("gemma4 begin_graph_capture: {e:?}"))?;
        decode_step_body(cfg, weights, state, gpu, position, None)?;
        gpu.graphs
            .end_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("gemma4 end_graph_capture: {e:?}"))?;
        // Captured kernels were RECORDED, not run — launch once so this token's
        // logits actually get produced.
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("gemma4 graph_launch (capture): {e:?}"))?;
        eprintln!(
            "[gemma4 hipGraph] captured decode forward — {} kernarg blobs retained",
            gpu.graphs.capture_blobs.len()
        );
    } else {
        // ── Replay phase ───────────────────────────────────────────────
        // Host-only update of the stable position source; the captured memcpy
        // re-reads it and propagates to pos_buf (read by rope / kv-write /
        // attention).
        state.pos_host[0] = position as i32;
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("gemma4 graph_launch (replay): {e:?}"))?;
    }
    state.n_tokens = position as usize + 1;

    // Logits download is outside the captured region (sync dtoh completes after
    // the captured kernels, which the device observes on the active stream).
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("gemma4 graph: download logits: {e:?}"))
}

/// Embedding lookup → x, then scale by sqrt(dim). Kept separate from the body
/// so the hipGraph path can run it OUTSIDE the captured region (token_id is
/// baked into the embedding kernarg).
fn embed_lookup(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    token_id: u32,
) -> Result<(), String> {
    use hipfire_runtime::llama::EmbeddingFormat;
    let dim = cfg.dim;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu
            .embedding_lookup_hfq4g256(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("gemma4: embed hfq4g256: {e:?}"))?,
        EmbeddingFormat::HFQ4G128 => gpu
            .embedding_lookup_hfq4g128(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("gemma4: embed hfq4g128: {e:?}"))?,
        EmbeddingFormat::Q8_0 => gpu
            .embedding_lookup_q8(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("gemma4: embed q8: {e:?}"))?,
        EmbeddingFormat::F32 => gpu
            .embedding_lookup(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("gemma4: embed f32: {e:?}"))?,
        EmbeddingFormat::Q4K => {
            return Err("gemma4: Q4K embedding format unsupported".to_string())
        }
    }
    gpu.scale_f32(&state.x, cfg.embed_scale)
        .map_err(|e| format!("gemma4: embed scale: {e:?}"))?;
    Ok(())
}

fn decode_step_body(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    position: u32,
    mut capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    let eps = cfg.norm_eps;

    // Device position scalar (i32). Staged from the heap-stable `state.pos_host`
    // so the captured memcpy re-reads it on replay (memcpy_htod_auto → async on
    // the capture stream when capturing).
    state.pos_host[0] = position as i32;
    {
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(state.pos_host.as_ptr() as *const u8, 4) };
        gpu.memcpy_htod_auto(&state.pos_buf, pos_bytes)
            .map_err(|e| format!("gemma4: htod pos: {e:?}"))?;
    }

    // Per-layer forward.
    for layer_idx in 0..cfg.n_layers {
        let slot = state.kv_slot_for_layer[layer_idx];
        match (cfg.layer_types[layer_idx], &weights.layers[layer_idx]) {
            (LayerType::Sliding, LayerWeights::Sliding(lw)) => {
                sliding_layer_decode(gpu, cfg, lw, position, slot, state)?;
            }
            (LayerType::Full, LayerWeights::Full(lw)) => {
                full_layer_decode(gpu, cfg, lw, position, slot, state)?;
            }
            _ => {
                return Err(format!(
                    "gemma4 layer {layer_idx} type/weights mismatch"
                ))
            }
        }
        if let Some(cap) = capture.as_deref_mut() {
            let h = gpu
                .download_f32(&state.x)
                .map_err(|e| format!("gemma4 L{layer_idx}: capture download: {e:?}"))?;
            cap[layer_idx].extend_from_slice(&h);
        }
    }
    state.n_tokens = position as usize + 1;

    // 4) Final RMSNorm → tmp.
    gpu.rmsnorm_f32(&state.x, &weights.final_norm, &state.tmp, eps)
        .map_err(|e| format!("gemma4: final rmsnorm: {e:?}"))?;

    // 5) LM head → logits (tied embed bytes via lm_head.buf alias).
    weight_gemv(gpu, &weights.lm_head, &state.tmp, &state.logits)
        .map_err(|e| format!("gemma4: lm_head: {e}"))?;

    // 6) Final logit softcap: logits = tanh(logits / cap) * cap.
    if cfg.final_logit_softcapping > 0.0 {
        gpu.logit_softcap_f32(&state.logits, cfg.vocab_size, cfg.final_logit_softcapping)
            .map_err(|e| format!("gemma4: logit softcap: {e:?}"))?;
    }
    Ok(())
}

/// One sliding-window attention layer (head_dim 256, own v_proj, full RoPE).
fn sliding_layer_decode(
    gpu: &mut Gpu,
    cfg: &Gemma4Config,
    lw: &SlidingLayerWeights,
    _pos: u32,
    kv_slot: usize,
    state: &mut Gemma4State,
) -> Result<(), String> {
    let dim = cfg.dim;
    let head_dim = cfg.sliding_head_dim;
    let n_heads = cfg.n_heads;
    let n_kv = cfg.sliding_n_kv_heads;
    let eps = cfg.norm_eps;
    let dim_bytes = dim * 4;

    // residual = x
    gpu.memcpy_dtod_auto(&state.residual.buf, &state.x.buf, dim_bytes)
        .map_err(|e| format!("gemma4 sliding: save residual: {e:?}"))?;

    // n1 = input_layernorm(x) → tmp.
    gpu.rmsnorm_f32(&state.x, &lw.input_layernorm, &state.tmp, eps)
        .map_err(|e| format!("gemma4 sliding: input rmsnorm: {e:?}"))?;

    // q/k/v projections. q+k fused (shared rmsnorm input) when both Q8.
    qk_proj(gpu, &lw.q_proj, &lw.k_proj, &state.tmp, &state.q, &state.k)?;
    weight_gemv(gpu, &lw.v_proj, &state.tmp, &state.v)
        .map_err(|e| format!("gemma4 sliding: v_proj: {e}"))?;

    // Per-head q_norm / k_norm over head_dim, and weight-less V RMSNorm (ones).
    // (V uses the no-scale RMS pattern — matches full layers and the HF
    // sliding-layer Vcur = rms_norm(Vcur) on the v_norm path.)
    gpu.rmsnorm_batched(&state.q, &lw.q_norm, &state.q, n_heads, head_dim, eps)
        .map_err(|e| format!("gemma4 sliding: q_norm: {e:?}"))?;
    gpu.rmsnorm_batched(&state.k, &lw.k_norm, &state.k, n_kv, head_dim, eps)
        .map_err(|e| format!("gemma4 sliding: k_norm: {e:?}"))?;
    gpu.rmsnorm_batched(&state.v, &state.v_norm_ones, &state.v, n_kv, head_dim, eps)
        .map_err(|e| format!("gemma4 sliding: v_norm: {e:?}"))?;

    // Pre-scale Q by sqrt(head_dim) so the kernel's 1/sqrt(head_dim) cancels →
    // effective Gemma 4 scale of 1.0.
    gpu.scale_f32(&state.q, (head_dim as f32).sqrt())
        .map_err(|e| format!("gemma4 sliding: q scale: {e:?}"))?;

    // RoPE: full rotate-half over the whole head_dim, theta = 10000.
    gpu.rope_f32(
        &state.q,
        &state.k,
        &state.pos_buf,
        n_heads,
        n_kv,
        head_dim,
        cfg.sliding_rope_theta,
    )
    .map_err(|e| format!("gemma4 sliding: rope: {e:?}"))?;

    // KV write (Q8) + windowed attention (window = sliding_window).
    attn_q8_swa(
        gpu,
        &mut state.kv_sliding,
        kv_slot,
        &state.k,
        &state.v,
        &state.q,
        &state.attn_out,
        &state.pos_buf,
        n_heads,
        n_kv,
        head_dim,
        state.max_seq,
        cfg.sliding_window,
    )?;

    // o_proj → tmp, post_attention_layernorm(tmp), x = residual + tmp.
    finish_attn_and_ffn(gpu, cfg, state, &lw_common_sliding(lw))?;
    Ok(())
}

/// One full (global) attention layer (head_dim 512, K=V sharing, partial RoPE).
fn full_layer_decode(
    gpu: &mut Gpu,
    cfg: &Gemma4Config,
    lw: &FullLayerWeights,
    _pos: u32,
    kv_slot: usize,
    state: &mut Gemma4State,
) -> Result<(), String> {
    let dim = cfg.dim;
    let head_dim = cfg.full_head_dim;
    let n_heads = cfg.n_heads;
    let n_kv = cfg.full_n_kv_heads;
    let eps = cfg.norm_eps;
    let dim_bytes = dim * 4;
    let kv_bytes = n_kv * head_dim * 4;

    // residual = x
    gpu.memcpy_dtod_auto(&state.residual.buf, &state.x.buf, dim_bytes)
        .map_err(|e| format!("gemma4 full: save residual: {e:?}"))?;

    // n1 = input_layernorm(x) → tmp.
    gpu.rmsnorm_f32(&state.x, &lw.input_layernorm, &state.tmp, eps)
        .map_err(|e| format!("gemma4 full: input rmsnorm: {e:?}"))?;

    // q/k projections — fused (shared rmsnorm input) when both Q8.
    qk_proj(gpu, &lw.q_proj, &lw.k_proj, &state.tmp, &state.q, &state.k)?;

    // V handling:
    //   attention_k_eq_v (12B): V = K's PRE-k_norm output (memcpy k → v BEFORE
    //     applying k_norm). Then weight-less RMSNorm on V.
    //   else: V = v_proj(n1).
    match &lw.v_proj {
        Some(vw) => {
            weight_gemv(gpu, vw, &state.tmp, &state.v)
                .map_err(|e| format!("gemma4 full: v_proj: {e}"))?;
        }
        None => {
            // CRITICAL ordering: capture V from the PRE-k_norm K output.
            gpu.memcpy_dtod_auto(&state.v.buf, &state.k.buf, kv_bytes)
                .map_err(|e| format!("gemma4 full: k→v copy: {e:?}"))?;
        }
    }

    // q_norm / k_norm over head_dim, weight-less V RMSNorm.
    gpu.rmsnorm_batched(&state.q, &lw.q_norm, &state.q, n_heads, head_dim, eps)
        .map_err(|e| format!("gemma4 full: q_norm: {e:?}"))?;
    gpu.rmsnorm_batched(&state.k, &lw.k_norm, &state.k, n_kv, head_dim, eps)
        .map_err(|e| format!("gemma4 full: k_norm: {e:?}"))?;
    gpu.rmsnorm_batched(&state.v, &state.v_norm_ones, &state.v, n_kv, head_dim, eps)
        .map_err(|e| format!("gemma4 full: v_norm: {e:?}"))?;

    // Pre-scale Q by sqrt(head_dim=512).
    gpu.scale_f32(&state.q, (head_dim as f32).sqrt())
        .map_err(|e| format!("gemma4 full: q scale: {e:?}"))?;

    // Proportional / partial RoPE: rotate the first `partial_rotary_factor ×
    // head_dim` dims; theta = full_rope_theta. n_rot_pairs = factor*head_dim/2.
    let n_rot_pairs = match cfg.full_rope_type {
        RopeType::Proportional => {
            ((head_dim as f32) * cfg.full_partial_rotary_factor * 0.5) as usize
        }
        // Default → all pairs rotate (head_dim/2). rope_partial_halved with
        // n_rot_pairs = head_dim/2 == full rotate-half.
        RopeType::Default => head_dim / 2,
    };
    gpu.rope_partial_halved_f32(
        &state.q,
        &state.k,
        &state.pos_buf,
        n_heads,
        n_kv,
        head_dim,
        n_rot_pairs,
        cfg.full_rope_theta,
    )
    .map_err(|e| format!("gemma4 full: rope: {e:?}"))?;

    // KV write (Q8) + full causal attention (window = 0).
    attn_q8_swa(
        gpu,
        &mut state.kv_full,
        kv_slot,
        &state.k,
        &state.v,
        &state.q,
        &state.attn_out,
        &state.pos_buf,
        n_heads,
        n_kv,
        head_dim,
        state.max_seq,
        0,
    )?;

    finish_attn_and_ffn(gpu, cfg, state, &lw_common_full(lw))?;
    Ok(())
}

/// KV write (Q8) + windowed/full attention via `attention_q8_0_kv_swa`.
/// `window = 0` ⇒ full causal; `window > 0` ⇒ sliding window.
#[allow(clippy::too_many_arguments)]
fn attn_q8_swa(
    gpu: &mut Gpu,
    kv: &mut KvCache,
    kv_slot: usize,
    k: &rdna_compute::GpuTensor,
    v: &rdna_compute::GpuTensor,
    q: &rdna_compute::GpuTensor,
    attn_out: &rdna_compute::GpuTensor,
    pos_buf: &hip_bridge::DeviceBuffer,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    max_seq: usize,
    window: usize,
) -> Result<(), String> {
    gpu.kv_cache_write_q8_0(&kv.k_gpu[kv_slot], k, pos_buf, n_kv, head_dim)
        .map_err(|e| format!("gemma4: kv write k: {e:?}"))?;
    gpu.kv_cache_write_q8_0(&kv.v_gpu[kv_slot], v, pos_buf, n_kv, head_dim)
        .map_err(|e| format!("gemma4: kv write v: {e:?}"))?;
    // DIAG: HIPFIRE_GEMMA4_BASELINE_ATTN routes through the proven baseline
    // attention_q8_0_kv (no window) to isolate the new _swa kernel.
    if std::env::var_os("HIPFIRE_GEMMA4_BASELINE_ATTN").is_some() {
        return gpu
            .attention_q8_0_kv(
                q,
                &kv.k_gpu[kv_slot],
                &kv.v_gpu[kv_slot],
                attn_out,
                pos_buf,
                max_seq,
                n_heads,
                n_kv,
                head_dim,
                kv.physical_cap,
            )
            .map_err(|e| format!("gemma4: attention baseline: {e:?}"));
    }
    gpu.attention_q8_0_kv_swa(
        q,
        &kv.k_gpu[kv_slot],
        &kv.v_gpu[kv_slot],
        attn_out,
        pos_buf,
        max_seq,
        n_heads,
        n_kv,
        head_dim,
        kv.physical_cap,
        window,
    )
    .map_err(|e| format!("gemma4: attention swa: {e:?}"))
}

/// Common per-layer tail shared by sliding + full layers: o_proj, post-attn
/// norm, attn residual add, pre-FFN norm, SwiGLU(gelu_tanh), post-FFN norm,
/// FFN residual add, learned layer_scalar. Operates on `state` scratch.
struct LayerTail<'a> {
    o_proj: &'a hipfire_runtime::llama::WeightTensor,
    post_attention_layernorm: &'a rdna_compute::GpuTensor,
    pre_feedforward_layernorm: &'a rdna_compute::GpuTensor,
    post_feedforward_layernorm: &'a rdna_compute::GpuTensor,
    gate_proj: &'a hipfire_runtime::llama::WeightTensor,
    up_proj: &'a hipfire_runtime::llama::WeightTensor,
    down_proj: &'a hipfire_runtime::llama::WeightTensor,
    layer_scalar_host: f32,
}

fn lw_common_sliding<'a>(lw: &'a SlidingLayerWeights) -> LayerTail<'a> {
    LayerTail {
        o_proj: &lw.o_proj,
        post_attention_layernorm: &lw.post_attention_layernorm,
        pre_feedforward_layernorm: &lw.pre_feedforward_layernorm,
        post_feedforward_layernorm: &lw.post_feedforward_layernorm,
        gate_proj: &lw.gate_proj,
        up_proj: &lw.up_proj,
        down_proj: &lw.down_proj,
        layer_scalar_host: lw.layer_scalar_host,
    }
}

fn lw_common_full<'a>(lw: &'a FullLayerWeights) -> LayerTail<'a> {
    LayerTail {
        o_proj: &lw.o_proj,
        post_attention_layernorm: &lw.post_attention_layernorm,
        pre_feedforward_layernorm: &lw.pre_feedforward_layernorm,
        post_feedforward_layernorm: &lw.post_feedforward_layernorm,
        gate_proj: &lw.gate_proj,
        up_proj: &lw.up_proj,
        down_proj: &lw.down_proj,
        layer_scalar_host: lw.layer_scalar_host,
    }
}

fn finish_attn_and_ffn(
    gpu: &mut Gpu,
    cfg: &Gemma4Config,
    state: &mut Gemma4State,
    tail: &LayerTail,
) -> Result<(), String> {
    let dim = cfg.dim;
    let eps = cfg.norm_eps;
    let dim_bytes = dim * 4;
    let ffn_hd = cfg.hidden_dim;

    // o_proj(attn_out) → tmp.
    weight_gemv(gpu, tail.o_proj, &state.attn_out, &state.tmp)
        .map_err(|e| format!("gemma4: o_proj: {e}"))?;

    // Sandwich post-attn norm (in-place on tmp).
    gpu.rmsnorm_f32(&state.tmp, tail.post_attention_layernorm, &state.tmp, eps)
        .map_err(|e| format!("gemma4: post_attn rmsnorm: {e:?}"))?;

    // x = residual + tmp.
    gpu.memcpy_dtod_auto(&state.x.buf, &state.residual.buf, dim_bytes)
        .map_err(|e| format!("gemma4: reset x: {e:?}"))?;
    gpu.add_inplace_f32(&state.x, &state.tmp)
        .map_err(|e| format!("gemma4: attn residual add: {e:?}"))?;

    // residual = x (FFN residual stream).
    gpu.memcpy_dtod_auto(&state.residual.buf, &state.x.buf, dim_bytes)
        .map_err(|e| format!("gemma4: save ffn residual: {e:?}"))?;

    // SwiGLU FFN with gelu_pytorch_tanh:
    //   gate = gate_proj(pre_ffn_norm(x)); up = up_proj(pre_ffn_norm(x));
    //   hidden = gelu_tanh(gate) * up; ffn_out = down_proj(hidden).
    //
    // Fused path (MQ4G256 gate/up): fuse the pre-FFN rmsnorm + FWHT rotation
    // into one launch (fused_rmsnorm_rotate_mq → tmp_rot), then gate+up in one
    // launch via fused_gate_up_hfq4g256 (MQ4G256 bytes are HFQ4G256-compatible
    // given a pre-rotated input; the kernel does NOT re-rotate). Mirrors the old
    // branch's HIPFIRE_GEMMA4_FUSED_PROJ path + qwen35.
    let fuse_gate_up = fused_ffn_enabled()
        && tail.gate_proj.gpu_dtype == DType::MQ4G256
        && tail.up_proj.gpu_dtype == DType::MQ4G256;
    if fuse_gate_up {
        gpu.fused_rmsnorm_rotate_mq(
            &state.x,
            tail.pre_feedforward_layernorm,
            &state.tmp_rot,
            dim,
            eps,
        )
        .map_err(|e| format!("gemma4: fused pre_ffn rmsnorm+rotate: {e:?}"))?;
        gpu.fused_gate_up_hfq4g256(
            &tail.gate_proj.buf,
            &tail.up_proj.buf,
            &state.tmp_rot,
            &state.gate_ffn,
            &state.up_ffn,
            tail.gate_proj.m,
            tail.up_proj.m,
            tail.gate_proj.k,
        )
        .map_err(|e| format!("gemma4: fused gate_up: {e:?}"))?;
    } else {
        // Eager fallback: plain rmsnorm → two rotation-doing GEMVs.
        gpu.rmsnorm_f32(&state.x, tail.pre_feedforward_layernorm, &state.tmp, eps)
            .map_err(|e| format!("gemma4: pre_ffn rmsnorm: {e:?}"))?;
        weight_gemv(gpu, tail.gate_proj, &state.tmp, &state.gate_ffn)
            .map_err(|e| format!("gemma4: gate_proj: {e}"))?;
        weight_gemv(gpu, tail.up_proj, &state.tmp, &state.up_ffn)
            .map_err(|e| format!("gemma4: up_proj: {e}"))?;
    }
    gpu.gelu_tanh_f32(&state.gate_ffn, &state.ffn_hidden, ffn_hd)
        .map_err(|e| format!("gemma4: gelu_tanh: {e:?}"))?;
    gpu.mul_f32(&state.ffn_hidden, &state.up_ffn, &state.ffn_hidden)
        .map_err(|e| format!("gemma4: silu mul: {e:?}"))?;
    weight_gemv(gpu, tail.down_proj, &state.ffn_hidden, &state.ffn_out)
        .map_err(|e| format!("gemma4: down_proj: {e}"))?;

    // Sandwich post-FFN norm (ffn_out → tmp).
    gpu.rmsnorm_f32(&state.ffn_out, tail.post_feedforward_layernorm, &state.tmp, eps)
        .map_err(|e| format!("gemma4: post_ffn rmsnorm: {e:?}"))?;

    // x = residual + tmp.
    gpu.memcpy_dtod_auto(&state.x.buf, &state.residual.buf, dim_bytes)
        .map_err(|e| format!("gemma4: reset x (ffn): {e:?}"))?;
    gpu.add_inplace_f32(&state.x, &state.tmp)
        .map_err(|e| format!("gemma4: ffn residual add: {e:?}"))?;

    // Learned per-layer scalar multiplier (no-op = 1.0 when tensor absent).
    if tail.layer_scalar_host != 1.0 {
        gpu.scale_f32(&state.x, tail.layer_scalar_host)
            .map_err(|e| format!("gemma4: layer_scalar: {e:?}"))?;
    }
    Ok(())
}
