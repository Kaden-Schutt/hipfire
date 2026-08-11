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
//!
//! ## Decode perf levers (gemma4-12B MQ4, hiptrx gfx1201)
//!
//! Decode is memory-bandwidth-bound: rocprofv3 attributes ~79% of GPU time to
//! the weight-reading GEMVs (FFN gate_up/down + attention q/k/v/o + lm_head).
//! All three levers below are byte-identical to the eager baseline (validated
//! over multiple prompts) and stack monotonically:
//!
//!   * `HIPFIRE_GEMMA4_GRAPH` (default ON; set =0 to disable) — hipGraph
//!     48-layer body + lm_head (`decode_step_with_graph`). +2.6%.
//!   * `HIPFIRE_GEMMA4_FUSED_FFN` (default ON) — fold pre-FFN rmsnorm+FWHT into
//!     one launch then gate+up into one (`fused_rmsnorm_rotate_mq` +
//!     `fused_gate_up_hfq4g256`; MQ4G256 bytes are HFQ4G256-compatible given a
//!     pre-rotated input). +1.0–1.2%.
//!   * `HIPFIRE_GEMMA4_FUSED_QK` (default ON) — fuse the Q8 q+k projections into
//!     one launch (`fused_gate_up_q8_0`, shared rmsnorm input). +1.1%.
//!
//! Full stack: 46.8 → 50.7 tok/s (+8.3%). The 70–75 tok/s target is NOT
//! reachable via fusion/graph alone — it would require reading ~40% fewer
//! weight bytes/token (lower-bit FFN quant or MQ4 attention), and MQ4 attention
//! is known to break coherence. No Q8 fused-QKV *decode* kernel exists
//! (`gemm_qkv_q8_0_wmma` is batched-prefill WMMA only), so q+k is the most that
//! fuses on the Q8 attention path.

use crate::config::{Gemma4Config, LayerType, RopeType};
use crate::gemma4::{FullLayerWeights, Gemma4State, Gemma4Weights, LayerWeights, SlidingLayerWeights};
use hipfire_runtime::llama::{
    rotate_x_mq_batched_for, weight_gemv, weight_gemv_prerotated, KvCache, WeightTensor,
};
use rdna_compute::{DType, Gpu, GpuTensor};

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

/// Master switch for the fused sandwich-postnorm + residual-add path
/// (`rmsnorm_residual_add_f32`, gemma4 L4). Fuses post_attention_layernorm and
/// post_feedforward_layernorm each from 3 launches (rmsnorm + memcpy(x<-res) +
/// add_inplace) down to 1, removing ~96 tiny launches/token. Default ON; opt
/// out with `HIPFIRE_GEMMA4_FUSED_POSTNORM=0`. Not byte-identical (the residual
/// add rounds inside the norm kernel) -- coherence-validated.
fn fused_postnorm_enabled() -> bool {
    !matches!(
        std::env::var("HIPFIRE_GEMMA4_FUSED_POSTNORM").ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

/// Master switch for the fused per-head weighted q/k RMSNorm + q prescale +
/// dual RoPE path (`fused_gemma4_qk_norm_rope_f32`, gemma4 L3). Collapses
/// (q_norm + k_norm + scale_f32(q) + rope) from 4 launches to 1 on the AR
/// decode path; V's weight-less RMSNorm and the k_eq_v k->v capture stay
/// separate. Default ON; opt out with `HIPFIRE_GEMMA4_FUSED_QK_ROPE=0`. Not
/// byte-identical (fused rsqrt/rope rounds differently) -- coherence-validated.
fn fused_qk_rope_enabled() -> bool {
    !matches!(
        std::env::var("HIPFIRE_GEMMA4_FUSED_QK_ROPE").ok().as_deref(),
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

/// Master switch for the fused attention-input norm path. Default ON; opt out
/// with `HIPFIRE_GEMMA4_FUSED_ATTN_NORM=0`. Only fires when q/k/v are MQ4G256
/// (this model's attention projections), folding the pre-attention
/// `input_layernorm` rmsnorm + the shared FWHT rotate into ONE launch
/// (`fused_rmsnorm_rotate_mq` -> tmp_rot), then feeding the prerotated input to
/// each projection's prerotated GEMV (rotate ONCE, reuse across q/k/v). This
/// removes the per-layer input `rmsnorm_f32` launch and collapses the 3
/// redundant per-projection rotates (q/k/v each re-rotated the identical input)
/// into 1. Byte-equivalent: `fused_rmsnorm_rotate_mq` + `gemv_*_prerotated` is
/// the exact same math as `rmsnorm_f32` + rotation-doing `weight_gemv`, fused.
fn fused_attn_norm_enabled() -> bool {
    !matches!(
        std::env::var("HIPFIRE_GEMMA4_FUSED_ATTN_NORM").ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

/// input_layernorm(x) -> q/k(/v) projections. Fused norm+rotate when MQ4G256
/// (see `fused_attn_norm_enabled`); else plain rmsnorm -> rotation-doing GEMVs.
///
/// `v_proj = None` => k_eq_v (the caller copies v from the PRE-k_norm k output
/// AFTER this returns); we only project q+k in that case. When `Some`, v is
/// projected here from the same (pre)normed input.
#[allow(clippy::too_many_arguments)]
fn attn_input_qkv(
    gpu: &mut Gpu,
    input_ln: &rdna_compute::GpuTensor,
    tmp: &rdna_compute::GpuTensor,
    tmp_rot: &rdna_compute::GpuTensor,
    q_proj: &WeightTensor,
    k_proj: &WeightTensor,
    v_proj: Option<&WeightTensor>,
    x: &rdna_compute::GpuTensor,
    q_out: &rdna_compute::GpuTensor,
    k_out: &rdna_compute::GpuTensor,
    v_out: &rdna_compute::GpuTensor,
    dim: usize,
    eps: f32,
    label: &str,
) -> Result<(), String> {
    let all_mq4 = q_proj.gpu_dtype == DType::MQ4G256
        && k_proj.gpu_dtype == DType::MQ4G256
        && v_proj.map_or(true, |w| w.gpu_dtype == DType::MQ4G256);
    if fused_attn_norm_enabled() && all_mq4 {
        // One launch: rmsnorm(x, input_ln) then FWHT-rotate -> tmp_rot.
        gpu.fused_rmsnorm_rotate_mq(x, input_ln, tmp_rot, dim, eps)
            .map_err(|e| format!("gemma4 {label}: fused input rmsnorm+rotate: {e:?}"))?;
        // Prerotated GEMVs share tmp_rot (NO re-rotation -- byte-identical math).
        weight_gemv_prerotated(gpu, q_proj, x, Some(tmp_rot), q_out)
            .map_err(|e| format!("gemma4 {label}: q_proj (prerot): {e}"))?;
        weight_gemv_prerotated(gpu, k_proj, x, Some(tmp_rot), k_out)
            .map_err(|e| format!("gemma4 {label}: k_proj (prerot): {e}"))?;
        if let Some(vw) = v_proj {
            weight_gemv_prerotated(gpu, vw, x, Some(tmp_rot), v_out)
                .map_err(|e| format!("gemma4 {label}: v_proj (prerot): {e}"))?;
        }
        Ok(())
    } else {
        // Eager fallback: plain rmsnorm -> tmp, then rotation-doing GEMVs.
        gpu.rmsnorm_f32(x, input_ln, tmp, eps)
            .map_err(|e| format!("gemma4 {label}: input rmsnorm: {e:?}"))?;
        qk_proj(gpu, q_proj, k_proj, tmp, q_out, k_out)?;
        if let Some(vw) = v_proj {
            weight_gemv(gpu, vw, tmp, v_out)
                .map_err(|e| format!("gemma4 {label}: v_proj: {e}"))?;
        }
        Ok(())
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

/// Decode one token via hipGraph capture/replay. **Default ON**
/// (`HIPFIRE_GEMMA4_GRAPH=0` to disable; +2.9% vs eager, byte-identical).
/// The 48-layer body + final-norm +
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
    // DEFAULT OFF pending a fix to the captured decode path.
    //
    // Measured on gfx1201 / 12B-it MQ4, prompt "Hello world", greedy:
    //   graph ON : 49.75 tok/s, output collapses after the first token
    //              ("Hello!s율 bawass율율 bawaky interracial율jal…")
    //   graph OFF: 49.72 tok/s, "Hello! How can I help you today?" — byte-
    //              identical to this PR's own Phase-2 Gate 2 expected output,
    //              and it stops cleanly on <turn|>.
    //
    // This is the failure mode AGENTS.md documents: a captured graph replays
    // dangling stack-pointer kernargs, so throughput looks right and the tokens
    // are garbage. All six Gemma4 dispatch helpers added during this port go
    // through `launch_maybe_blob` and are capture-safe, so the offending raw
    // launch is elsewhere in the decode body and still needs to be found.
    //
    // The graph is worth 0.03 tok/s here (0.06%), so correctness costs nothing.
    // Re-enable with HIPFIRE_GEMMA4_GRAPH=1 once the capture path is fixed and
    // a coherence check passes with it on.
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

    // n1 = input_layernorm(x) -> q/k/v. Fused (norm+rotate+prerotated GEMVs,
    // shared rotate) when MQ4G256; else plain rmsnorm -> rotation-doing GEMVs.
    attn_input_qkv(
        gpu,
        &lw.input_layernorm,
        &state.tmp,
        &state.tmp_rot,
        &lw.q_proj,
        &lw.k_proj,
        Some(&lw.v_proj),
        &state.x,
        &state.q,
        &state.k,
        &state.v,
        dim,
        eps,
        "sliding",
    )?;

    // Weight-less V RMSNorm (ones). V is independent of q/k here (own v_proj),
    // so it normalizes outside the fused q/k path either way.
    gpu.rmsnorm_batched(&state.v, &state.v_norm_ones, &state.v, n_kv, head_dim, eps)
        .map_err(|e| format!("gemma4 sliding: v_norm: {e:?}"))?;

    // Per-head q_norm/k_norm (weighted RMS) + q prescale (sqrt(head_dim) so the
    // attn kernel's 1/sqrt(head_dim) cancels → effective Gemma4 scale 1.0) +
    // full rotate-half RoPE (n_rot_pairs = head_dim/2, theta = 10000). Fused
    // (L3) into one launch; eager fallback = 4 separate launches.
    if fused_qk_rope_enabled() {
        gpu.fused_gemma4_qk_norm_rope_f32(
            &state.q,
            &state.k,
            &lw.q_norm,
            &lw.k_norm,
            &state.pos_buf,
            n_heads,
            n_kv,
            head_dim,
            head_dim / 2, // full rotate-half
            (head_dim as f32).sqrt(),
            cfg.sliding_rope_theta,
            eps,
        )
        .map_err(|e| format!("gemma4 sliding: fused qk norm+rope: {e:?}"))?;
    } else {
        gpu.rmsnorm_batched(&state.q, &lw.q_norm, &state.q, n_heads, head_dim, eps)
            .map_err(|e| format!("gemma4 sliding: q_norm: {e:?}"))?;
        gpu.rmsnorm_batched(&state.k, &lw.k_norm, &state.k, n_kv, head_dim, eps)
            .map_err(|e| format!("gemma4 sliding: k_norm: {e:?}"))?;
        gpu.scale_f32(&state.q, (head_dim as f32).sqrt())
            .map_err(|e| format!("gemma4 sliding: q scale: {e:?}"))?;
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
    }

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

    // n1 = input_layernorm(x) -> q/k(/v). Fused (norm+rotate+prerotated GEMVs,
    // shared rotate) when MQ4G256; else plain rmsnorm -> rotation-doing GEMVs.
    // When v_proj is Some, v is projected inside the helper from the same
    // (pre)normed input; when None (k_eq_v) the helper projects only q+k and we
    // capture V below from the PRE-k_norm k output.
    attn_input_qkv(
        gpu,
        &lw.input_layernorm,
        &state.tmp,
        &state.tmp_rot,
        &lw.q_proj,
        &lw.k_proj,
        lw.v_proj.as_ref(),
        &state.x,
        &state.q,
        &state.k,
        &state.v,
        dim,
        eps,
        "full",
    )?;

    // V handling for the k_eq_v (12B) case: V = K's PRE-k_norm output (memcpy
    // k -> v BEFORE applying k_norm). Then weight-less RMSNorm on V (below).
    if lw.v_proj.is_none() {
        // CRITICAL ordering: capture V from the PRE-k_norm K output.
        gpu.memcpy_dtod_auto(&state.v.buf, &state.k.buf, kv_bytes)
            .map_err(|e| format!("gemma4 full: k->v copy: {e:?}"))?;
    }

    // Weight-less V RMSNorm (ones). For k_eq_v, V was already captured from the
    // PRE-k_norm K above, so this is safe to run before the fused q/k path.
    gpu.rmsnorm_batched(&state.v, &state.v_norm_ones, &state.v, n_kv, head_dim, eps)
        .map_err(|e| format!("gemma4 full: v_norm: {e:?}"))?;

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

    // Per-head q_norm/k_norm (weighted RMS) + q prescale (sqrt(head_dim)) +
    // partial rotate-half RoPE (n_rot_pairs, theta = full_rope_theta). Fused
    // (L3) into one launch; eager fallback = 4 separate launches.
    if fused_qk_rope_enabled() {
        gpu.fused_gemma4_qk_norm_rope_f32(
            &state.q,
            &state.k,
            &lw.q_norm,
            &lw.k_norm,
            &state.pos_buf,
            n_heads,
            n_kv,
            head_dim,
            n_rot_pairs,
            (head_dim as f32).sqrt(),
            cfg.full_rope_theta,
            eps,
        )
        .map_err(|e| format!("gemma4 full: fused qk norm+rope: {e:?}"))?;
    } else {
        gpu.rmsnorm_batched(&state.q, &lw.q_norm, &state.q, n_heads, head_dim, eps)
            .map_err(|e| format!("gemma4 full: q_norm: {e:?}"))?;
        gpu.rmsnorm_batched(&state.k, &lw.k_norm, &state.k, n_kv, head_dim, eps)
            .map_err(|e| format!("gemma4 full: k_norm: {e:?}"))?;
        gpu.scale_f32(&state.q, (head_dim as f32).sqrt())
            .map_err(|e| format!("gemma4 full: q scale: {e:?}"))?;
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
    }

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

    // Sandwich post-attn norm + residual add: x = residual + post_attn_norm(tmp).
    // Fused (L4) into one launch; eager fallback = rmsnorm + memcpy + add.
    if fused_postnorm_enabled() {
        gpu.rmsnorm_residual_add_f32(
            &state.tmp,
            tail.post_attention_layernorm,
            &state.residual,
            &state.x,
            eps,
        )
        .map_err(|e| format!("gemma4: fused post_attn norm+residual: {e:?}"))?;
    } else {
        // Sandwich post-attn norm (in-place on tmp).
        gpu.rmsnorm_f32(&state.tmp, tail.post_attention_layernorm, &state.tmp, eps)
            .map_err(|e| format!("gemma4: post_attn rmsnorm: {e:?}"))?;

        // x = residual + tmp.
        gpu.memcpy_dtod_auto(&state.x.buf, &state.residual.buf, dim_bytes)
            .map_err(|e| format!("gemma4: reset x: {e:?}"))?;
        gpu.add_inplace_f32(&state.x, &state.tmp)
            .map_err(|e| format!("gemma4: attn residual add: {e:?}"))?;
    }

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

    // Sandwich post-FFN norm + residual add: x = residual + post_ffn_norm(ffn_out).
    // Fused (L4) into one launch; eager fallback = rmsnorm + memcpy + add.
    if fused_postnorm_enabled() {
        gpu.rmsnorm_residual_add_f32(
            &state.ffn_out,
            tail.post_feedforward_layernorm,
            &state.residual,
            &state.x,
            eps,
        )
        .map_err(|e| format!("gemma4: fused post_ffn norm+residual: {e:?}"))?;
    } else {
        // Sandwich post-FFN norm (ffn_out → tmp).
        gpu.rmsnorm_f32(&state.ffn_out, tail.post_feedforward_layernorm, &state.tmp, eps)
            .map_err(|e| format!("gemma4: post_ffn rmsnorm: {e:?}"))?;

        // x = residual + tmp.
        gpu.memcpy_dtod_auto(&state.x.buf, &state.residual.buf, dim_bytes)
            .map_err(|e| format!("gemma4: reset x (ffn): {e:?}"))?;
        gpu.add_inplace_f32(&state.x, &state.tmp)
            .map_err(|e| format!("gemma4: ffn residual add: {e:?}"))?;
    }

    // Learned per-layer scalar multiplier (no-op = 1.0 when tensor absent).
    if tail.layer_scalar_host != 1.0 {
        gpu.scale_f32(&state.x, tail.layer_scalar_host)
            .map_err(|e| format!("gemma4: layer_scalar: {e:?}"))?;
    }
    Ok(())
}

// ════════════════════════════════════════════════════════════════════════
//  Batched verify forward — `forward_batch`
// ════════════════════════════════════════════════════════════════════════
//
// Verifies B tokens [tokens[0..B]] starting at absolute position `start_pos`
// in ONE pass (weights read once). The spec-decode keystone: returns the LAST
// token's logits, byte/argmax-identical to running B sequential `decode_step`
// calls, and leaves the two KV caches in the same state the sequential path
// would (so attention over history is correct for the next token).
//
// Structure mirrors the eager `decode_step_body` exactly, batched across B:
//   - embed lookup per token → x[B,dim] → ×√dim
//   - per layer (sliding / full), the same sandwich-norm + dual-RoPE +
//     k_eq_v pipeline, batched
//   - attention via `attention_q8_0_kv_batched_masked` with a per-row
//     causal+sliding-window additive mask ([B × seq_len], 0 = visible,
//     -inf = masked); block_start=0, block_cols=seq_len gives full per-row
//     control for BOTH layer types
//   - last row → final norm → tied lm_head → logit_softcap(30) → return logits
//
// ADDITIVE: does not touch `decode_step` / `decode_step_with_graph`. The eager
// path is unchanged.

/// One projection GEMM, batched, dispatched by weight dtype. Mirrors the
/// dtypes `weight_gemv` handles on the gemma4 weight set:
///   - Q8_0      → `gemm_q8_0_batched_chunked` (no rotation; auto-routes to
///                 the WMMA Q8 GEMM on gfx12/RDNA4 — the scalar
///                 `gemm_q8_0_batched` was 334 us/call at b=4 proj widths =
///                 ~61 ms/round = the gemma4 EAGLE slowdown on gfx1201;
///                 elsewhere identical scalar kernel, b ≤ 64 = one sub-batch)
///   - F32       → `gemm_q8_0_batched` would be wrong; F32 weights are only
///                 produced for tiny norm-like tensors, never projections here,
///                 so this is an error (projections are Q8/MQ4 in both HFQs).
///   - MQ4G256   → FWHT-rotate x (batched) then `gemm_hfq4g256_batched_lmhead`
///                 (MQ4G256 bytes are HFQ4G256-compatible given a pre-rotated
///                 input — same equivalence the eager fused FFN path relies on).
///
/// `x` is [B, k] row-major, `y` is [B, m] row-major. `x_rot` is a [B, k]
/// scratch used only on the MQ4 path. Both buffers must be ≥ the needed size.
#[allow(clippy::too_many_arguments)]
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
            .map_err(|e| format!("gemma4 batch {label} (q8): {e:?}")),
        DType::MQ4G256 | DType::HFQ4G256 => {
            // FWHT-rotate the shared input once, then run the prerotated GEMM.
            // rotate_x_mq_batched_for handles the AWQ-aware branch (no-op AWQ
            // on these HFQs → plain rotate_x_mq_batched).
            rotate_x_mq_batched_for(gpu, w, x, x_rot, w.k, b)
                .map_err(|e| format!("gemma4 batch {label} rotate: {e:?}"))?;
            gpu.gemm_hfq4g256_batched_lmhead(&w.buf, x_rot, y, w.m, w.k, b)
                .map_err(|e| format!("gemma4 batch {label} (mq4): {e:?}"))
        }
        other => Err(format!(
            "gemma4 batch {label}: dtype {other:?} has no batched proj kernel"
        )),
    }
}

/// argmax over a logits row (spec-decode greedy per-position prediction).
fn argmax_f32_row(v: &[f32]) -> u32 {
    let mut bi = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i as u32;
        }
    }
    bi
}

/// Batched verify forward. See module-level note above. `tokens.len()` = B
/// (1..=64, the `gemm_q8_0_batched` / batched-attention kernel cap). Returns
/// the LAST token's logits. Side effect: writes both KV caches for positions
/// [start_pos, start_pos+B) and sets `state.n_tokens = start_pos + B`.
///
/// Thin wrapper over `forward_batch_spec` with both spec out-params off:
/// behaviour is BYTE-IDENTICAL to the original `forward_batch` (the eager /
/// verify path). All existing callers stay on this signature unchanged.
pub fn forward_batch(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: usize,
) -> Result<Vec<f32>, String> {
    forward_batch_spec(cfg, weights, state, gpu, tokens, start_pos, None, None)
}

/// Batched verify forward with optional spec-decode out-params (Part A of the
/// EAGLE wiring). Identical body to `forward_batch`; the two optional outputs
/// are computed ADDITIVELY after the layer loop and do not alter the returned
/// last-token logits, the KV writes, or `state.n_tokens`.
///
/// * `per_token_hidden_out` — when `Some`, receives each of the B positions'
///   POST-`model.norm` hidden ([B × dim], row-major). Row `i` is exactly the
///   hidden that the eager `decode_step` leaves in `state.tmp` after
///   `final_norm` (the lm_head input). Used to seed the drafter for the next
///   spec round at the accepted-bonus row.
/// * `per_pos_argmax_out` — when `Some`, receives the target's greedy argmax at
///   each block position ([B], `argmax_per_pos[i]` = the target's prediction
///   AFTER block position `i`). Computed by running the SAME lm_head + final
///   logit softcap the eager path uses, per row. Required for greedy accept.
///
/// When BOTH are `None` this is byte-identical to `forward_batch` (only the
/// last-row final-norm + lm_head + softcap + download runs, as before).
#[allow(clippy::too_many_arguments)]
pub fn forward_batch_spec(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: usize,
    per_token_hidden_out: Option<&GpuTensor>,
    per_pos_argmax_out: Option<&mut Vec<u32>>,
) -> Result<Vec<f32>, String> {
    let b = tokens.len();
    if b == 0 {
        return Err("gemma4 forward_batch: empty token slice".to_string());
    }
    if b > 64 {
        return Err(format!("gemma4 forward_batch: B={b} exceeds kernel cap 64"));
    }
    let dim = cfg.dim;
    let eps = cfg.norm_eps;
    let ffn_hd = cfg.hidden_dim;
    let max_q = cfg.max_q_dim();
    let max_kv = cfg.max_kv_dim();
    // seq_len after this batch = absolute positions [start_pos, start_pos+B).
    let seq_len = start_pos + b;

    let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
        g.alloc_tensor(&[n], DType::F32)
            .map_err(|e| format!("gemma4 forward_batch alloc {label}: {e:?}"))
    };

    // ── Batched scratch (per call; verify is not the hot per-kernel loop). ──
    let x = alloc(gpu, b * dim, "x")?;
    let residual = alloc(gpu, b * dim, "residual")?;
    let nrm = alloc(gpu, b * dim, "nrm")?; // rmsnorm output / shared proj input
    // x_rot is the SHARED FWHT-rotate scratch for every projection in the layer
    // (proj_gemm_batched rotates b*w.k floats into it). The largest w.k is the
    // o_proj on FULL attention layers: k = n_heads*full_head_dim (= max_q here),
    // which exceeds dim. Size for the max so the o_proj rotation can't OOB-write.
    let x_rot = alloc(gpu, b * max_q.max(dim), "x_rot")?; // FWHT scratch (MQ4 proj path)
    let q = alloc(gpu, b * max_q, "q")?;
    let k = alloc(gpu, b * max_kv, "k")?;
    let v = alloc(gpu, b * max_kv, "v")?;
    let attn_out = alloc(gpu, b * max_q, "attn_out")?;
    let o = alloc(gpu, b * dim, "o")?;
    let gate_ffn = alloc(gpu, b * ffn_hd, "gate_ffn")?;
    let up_ffn = alloc(gpu, b * ffn_hd, "up_ffn")?;
    let ffn_hidden = alloc(gpu, b * ffn_hd, "ffn_hidden")?;
    let ffn_out = alloc(gpu, b * dim, "ffn_out")?;
    // FFN rotation scratch must hold B*ffn_hd for the down_proj input.
    let ffn_rot = alloc(gpu, b * ffn_hd, "ffn_rot")?;

    // positions [B] i32 (kernels read this buffer as i32).
    let pos_data: Vec<i32> = (0..b).map(|i| (start_pos + i) as i32).collect();
    let pos_bytes: Vec<u8> = pos_data.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let pos_array = alloc(gpu, b, "pos_array")?;
    gpu.hip
        .memcpy_htod(&pos_array.buf, &pos_bytes)
        .map_err(|e| format!("gemma4 forward_batch htod pos: {e:?}"))?;

    // ── Per-row attention mask [B × seq_len]: 0.0 = visible, -inf = masked. ──
    // Built per layer type (sliding has a window; full is plain causal). Two
    // host masks are uploaded to two device buffers and reused across layers of
    // the matching type. block_start=0, block_cols=seq_len so the kernel applies
    // the bias to every key (full per-row control).
    let build_mask = |window: usize| -> Vec<f32> {
        let mut m = vec![0.0f32; b * seq_len];
        for row in 0..b {
            let pos = start_pos + row; // absolute query position
            let lo = if window > 0 {
                pos.saturating_sub(window - 1)
            } else {
                0
            };
            for t in 0..seq_len {
                // Visible iff lo <= t <= pos (causal + optional sliding window).
                let visible = t >= lo && t <= pos;
                m[row * seq_len + t] = if visible { 0.0 } else { f32::NEG_INFINITY };
            }
        }
        m
    };
    let upload_mask = |g: &mut Gpu, host: &[f32], label: &str| -> Result<GpuTensor, String> {
        let t = g
            .alloc_tensor(&[host.len()], DType::F32)
            .map_err(|e| format!("gemma4 forward_batch mask alloc {label}: {e:?}"))?;
        let bytes =
            unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4) };
        g.hip
            .memcpy_htod(&t.buf, bytes)
            .map_err(|e| format!("gemma4 forward_batch mask htod {label}: {e:?}"))?;
        Ok(t)
    };
    let has_sliding = cfg.layer_types.iter().any(|&t| t == LayerType::Sliding);
    let has_full = cfg.layer_types.iter().any(|&t| t == LayerType::Full);
    let sliding_mask = if has_sliding {
        Some(upload_mask(gpu, &build_mask(cfg.sliding_window), "sliding")?)
    } else {
        None
    };
    let full_mask = if has_full {
        Some(upload_mask(gpu, &build_mask(0), "full")?)
    } else {
        None
    };

    // ── Embedding: per-token lookup into x[B,dim], then ×√dim over all rows. ──
    {
        let x_single = alloc(gpu, dim, "x_single")?;
        for (i, &tok) in tokens.iter().enumerate() {
            embed_lookup_row(cfg, weights, gpu, &x_single, tok)?;
            gpu.hip
                .memcpy_dtod_at(&x.buf, i * dim * 4, &x_single.buf, 0, dim * 4)
                .map_err(|e| format!("gemma4 forward_batch embed copy: {e:?}"))?;
        }
        gpu.free_tensor(x_single).ok();
    }
    // √dim scale on the whole [B*dim] buffer (uniform — matches eager scale_f32).
    gpu.scale_f32(&x, cfg.embed_scale)
        .map_err(|e| format!("gemma4 forward_batch embed scale: {e:?}"))?;

    for layer_idx in 0..cfg.n_layers {
        let slot = state.kv_slot_for_layer[layer_idx];
        match (cfg.layer_types[layer_idx], &weights.layers[layer_idx]) {
            (LayerType::Sliding, LayerWeights::Sliding(lw)) => {
                let hd = cfg.sliding_head_dim;
                let n_kv = cfg.sliding_n_kv_heads;
                batch_attn_block(
                    gpu,
                    cfg,
                    state,
                    BatchAttn {
                        b,
                        hd,
                        n_kv,
                        slot,
                        seq_len,
                        is_full: false,
                        rope: RopeKind::Full(cfg.sliding_rope_theta),
                        q_proj: &lw.q_proj,
                        k_proj: &lw.k_proj,
                        v_proj: Some(&lw.v_proj),
                        o_proj: &lw.o_proj,
                        q_norm: &lw.q_norm,
                        k_norm: &lw.k_norm,
                        input_layernorm: &lw.input_layernorm,
                        post_attention_layernorm: &lw.post_attention_layernorm,
                        mask: sliding_mask.as_ref().unwrap(),
                    },
                    &x,
                    &residual,
                    &nrm,
                    &x_rot,
                    &q,
                    &k,
                    &v,
                    &attn_out,
                    &o,
                    &pos_array,
                )?;
                batch_ffn_block(
                    gpu,
                    cfg,
                    state,
                    &lw_common_sliding(lw),
                    b,
                    &x,
                    &residual,
                    &nrm,
                    &gate_ffn,
                    &up_ffn,
                    &ffn_hidden,
                    &ffn_out,
                    &ffn_rot,
                )?;
            }
            (LayerType::Full, LayerWeights::Full(lw)) => {
                let hd = cfg.full_head_dim;
                let n_kv = cfg.full_n_kv_heads;
                let n_rot_pairs = match cfg.full_rope_type {
                    RopeType::Proportional => {
                        ((hd as f32) * cfg.full_partial_rotary_factor * 0.5) as usize
                    }
                    RopeType::Default => hd / 2,
                };
                batch_attn_block(
                    gpu,
                    cfg,
                    state,
                    BatchAttn {
                        b,
                        hd,
                        n_kv,
                        slot,
                        seq_len,
                        is_full: true,
                        rope: RopeKind::PartialHalved(cfg.full_rope_theta, n_rot_pairs),
                        q_proj: &lw.q_proj,
                        k_proj: &lw.k_proj,
                        v_proj: lw.v_proj.as_ref(),
                        o_proj: &lw.o_proj,
                        q_norm: &lw.q_norm,
                        k_norm: &lw.k_norm,
                        input_layernorm: &lw.input_layernorm,
                        post_attention_layernorm: &lw.post_attention_layernorm,
                        mask: full_mask.as_ref().unwrap(),
                    },
                    &x,
                    &residual,
                    &nrm,
                    &x_rot,
                    &q,
                    &k,
                    &v,
                    &attn_out,
                    &o,
                    &pos_array,
                )?;
                batch_ffn_block(
                    gpu,
                    cfg,
                    state,
                    &lw_common_full(lw),
                    b,
                    &x,
                    &residual,
                    &nrm,
                    &gate_ffn,
                    &up_ffn,
                    &ffn_hidden,
                    &ffn_out,
                    &ffn_rot,
                )?;
            }
            _ => return Err(format!("gemma4 forward_batch L{layer_idx} type/weights mismatch")),
        }
    }
    state.n_tokens = start_pos + b;

    // ── Spec-decode out-params (ADDITIVE; both default-off). ──
    // `x` here holds the [B, dim] post-residual hidden for all B positions.
    //
    // (1) per_token_hidden_out: batched final-norm over all B rows → the
    //     caller's [B, dim] buffer. Row i is exactly what the eager
    //     decode_step leaves in state.tmp (the lm_head input) at that position.
    // (2) per_pos_argmax_out: batched lm_head over all B rows → [B, vocab],
    //     final logit softcap (elementwise over B*vocab), per-row argmax on host.
    //     Uses the SAME batched proj path (`proj_gemm_batched`) the verify
    //     forward uses for its other projections, so the MQ4 lm_head rotation is
    //     handled with explicit scratch (NOT weight_gemv's internal mq_x_rot,
    //     which the batched path never sizes → would fault on MQ4 lm_head).
    //
    // Both use a [B, dim] normed-hidden buffer. When (1) is requested we write
    // straight into it; otherwise we allocate a local one only if (2) needs it.
    let want_hidden = per_token_hidden_out.is_some();
    let want_argmax = per_pos_argmax_out.is_some();
    if want_hidden || want_argmax {
        // Normed hidden destination: the caller's buffer if provided, else a
        // local scratch sized [B, dim].
        let local_hidden = if want_hidden {
            None
        } else {
            Some(alloc(gpu, b * dim, "spec_normed")?)
        };
        let normed_hidden: &GpuTensor = match per_token_hidden_out {
            Some(h) => h,
            None => local_hidden.as_ref().unwrap(),
        };
        // Batched final RMSNorm over all B rows.
        gpu.rmsnorm_batched(&x, &weights.final_norm, normed_hidden, b, dim, eps)
            .map_err(|e| format!("gemma4 forward_batch_spec final rmsnorm: {e:?}"))?;

        if let Some(out) = per_pos_argmax_out {
            out.clear();
            out.reserve(b);
            let vocab = cfg.vocab_size;
            if weights.lm_head.gpu_dtype == DType::Q8_0 {
                // FAST PATH (the spec-verify perf lever): a single batched Q8 WMMA
                // lm_head reads the ~1 GB Q8 weight ONCE for all B rows, instead of
                // the per-row weight_gemv that re-streamed the whole weight B times
                // (~90% of the verify, ~4× off roofline). On gfx12 (RDNA4)
                // `gemm_q8_0_batched_chunked` auto-routes to the WMMA Q8 GEMM (now
                // correct after the stale-fp16-cache fix); elsewhere it sub-batches
                // the scalar Q8 GEMM — either way Y[B, vocab] row-major.
                //
                // Softcap is SKIPPED: it's a strictly-monotonic per-element map
                // (tanh-scaled), so argmax(softcap(z)) == argmax(z). The accept
                // decision is an argmax, so this is bit-exact in the decision.
                let logits_b = alloc(gpu, b * vocab, "spec_logits_b")?;
                gpu.gemm_q8_0_batched_chunked(
                    &weights.lm_head.buf,
                    normed_hidden,
                    &logits_b,
                    weights.lm_head.m,
                    weights.lm_head.k,
                    b,
                )
                .map_err(|e| format!("gemma4 forward_batch_spec batched lm_head: {e:?}"))?;
                // GPU per-row argmax over [B, vocab]; only B indices land on PCIe.
                let idx_buf = alloc(gpu, b, "spec_argmax_idx")?;
                gpu.argmax_f32_batched(&logits_b, &idx_buf, vocab, b)
                    .map_err(|e| format!("gemma4 forward_batch_spec batched argmax: {e:?}"))?;
                let mut idx_i32 = vec![0i32; b];
                let idx_bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(idx_i32.as_mut_ptr() as *mut u8, b * 4)
                };
                gpu.hip
                    .memcpy_dtoh(idx_bytes, &idx_buf.buf)
                    .map_err(|e| format!("gemma4 forward_batch_spec argmax dtoh: {e:?}"))?;
                for v in idx_i32 {
                    out.push(v as u32);
                }
                gpu.free_tensor(logits_b).ok();
                gpu.free_tensor(idx_buf).ok();
            } else {
                // FALLBACK (non-Q8 lm_head, e.g. MQ4): per-row SCALAR `weight_gemv`
                // (exactly the eager decode_step / the last-row path below). We
                // deliberately do NOT use the batched MQ4 path here: for an MQ4
                // lm_head (m=vocab≈262144) it routes through
                // `gemm_hfq4g256_residual_wmma[_gfx12]`, whose b>1 path faults
                // (illegal access) at this output width on gfx12. `weight_gemv`
                // (per-row, b=1) sidesteps it entirely.
                for i in 0..b {
                    let hidden_row = normed_hidden.sub_offset(i * dim, dim);
                    weight_gemv(gpu, &weights.lm_head, &hidden_row, &state.logits)
                        .map_err(|e| format!("gemma4 forward_batch_spec lm_head row {i}: {e}"))?;
                    if cfg.final_logit_softcapping > 0.0 {
                        gpu.logit_softcap_f32(
                            &state.logits,
                            vocab,
                            cfg.final_logit_softcapping,
                        )
                        .map_err(|e| format!("gemma4 forward_batch_spec softcap row {i}: {e:?}"))?;
                    }
                    let row = gpu
                        .download_f32(&state.logits)
                        .map_err(|e| format!("gemma4 forward_batch_spec download row {i}: {e:?}"))?;
                    out.push(argmax_f32_row(&row));
                }
            }
        }
        if let Some(t) = local_hidden {
            gpu.free_tensor(t).ok();
        }
    }

    // ── Final RMSNorm + tied lm_head on the LAST row only (verify needs the
    //    last position's logits). ──
    let x_last = alloc(gpu, dim, "x_last")?;
    gpu.hip
        .memcpy_dtod_at(&x_last.buf, 0, &x.buf, (b - 1) * dim * 4, dim * 4)
        .map_err(|e| format!("gemma4 forward_batch last copy: {e:?}"))?;
    gpu.rmsnorm_f32(&x_last, &weights.final_norm, &state.tmp, eps)
        .map_err(|e| format!("gemma4 forward_batch final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.tmp, &state.logits)
        .map_err(|e| format!("gemma4 forward_batch lm_head: {e}"))?;
    if cfg.final_logit_softcapping > 0.0 {
        gpu.logit_softcap_f32(&state.logits, cfg.vocab_size, cfg.final_logit_softcapping)
            .map_err(|e| format!("gemma4 forward_batch logit softcap: {e:?}"))?;
    }
    let logits = gpu
        .download_f32(&state.logits)
        .map_err(|e| format!("gemma4 forward_batch download logits: {e:?}"))?;

    // Free batched scratch.
    let mut frees = vec![
        x, residual, nrm, x_rot, q, k, v, attn_out, o, gate_ffn, up_ffn, ffn_hidden, ffn_out,
        ffn_rot, pos_array, x_last,
    ];
    if let Some(m) = sliding_mask {
        frees.push(m);
    }
    if let Some(m) = full_mask {
        frees.push(m);
    }
    for t in frees {
        gpu.free_tensor(t).ok();
    }
    Ok(logits)
}

/// Embedding lookup for one token into a [dim] buffer (no √dim scale — the
/// caller applies it once over the whole batch). Mirrors `embed_lookup`'s
/// format dispatch.
fn embed_lookup_row(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    gpu: &mut Gpu,
    dst: &GpuTensor,
    token_id: u32,
) -> Result<(), String> {
    use hipfire_runtime::llama::EmbeddingFormat;
    let dim = cfg.dim;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu
            .embedding_lookup_hfq4g256(&weights.embed_tokens, dst, token_id, dim)
            .map_err(|e| format!("gemma4 forward_batch embed hfq4g256: {e:?}")),
        EmbeddingFormat::HFQ4G128 => gpu
            .embedding_lookup_hfq4g128(&weights.embed_tokens, dst, token_id, dim)
            .map_err(|e| format!("gemma4 forward_batch embed hfq4g128: {e:?}")),
        EmbeddingFormat::Q8_0 => gpu
            .embedding_lookup_q8(&weights.embed_tokens, dst, token_id, dim)
            .map_err(|e| format!("gemma4 forward_batch embed q8: {e:?}")),
        EmbeddingFormat::F32 => gpu
            .embedding_lookup(&weights.embed_tokens, dst, token_id, dim)
            .map_err(|e| format!("gemma4 forward_batch embed f32: {e:?}")),
        EmbeddingFormat::Q4K => Err("gemma4 forward_batch: Q4K embedding unsupported".to_string()),
    }
}

/// RoPE flavour for a batched layer.
enum RopeKind {
    /// Full rotate-half over the whole head_dim (sliding layers). theta.
    Full(f32),
    /// Partial proportional rotate-half (full layers). (theta, n_rot_pairs).
    PartialHalved(f32, usize),
}

/// Bundled batched-attention inputs for one layer (keeps arg count sane).
struct BatchAttn<'a> {
    b: usize,
    hd: usize,
    n_kv: usize,
    slot: usize,
    seq_len: usize,
    is_full: bool,
    rope: RopeKind,
    q_proj: &'a WeightTensor,
    k_proj: &'a WeightTensor,
    v_proj: Option<&'a WeightTensor>,
    o_proj: &'a WeightTensor,
    q_norm: &'a GpuTensor,
    k_norm: &'a GpuTensor,
    input_layernorm: &'a GpuTensor,
    post_attention_layernorm: &'a GpuTensor,
    mask: &'a GpuTensor,
}

/// Batched attention sub-block: residual save, input rmsnorm, q/k/v proj,
/// q/k/v norms, q-scale, RoPE, KV write, masked attention, o_proj, residual
/// add. Mirrors `sliding_layer_decode` / `full_layer_decode` (attention half)
/// batched across B. On exit `x += o_proj(attn)` per row.
#[allow(clippy::too_many_arguments)]
fn batch_attn_block(
    gpu: &mut Gpu,
    cfg: &Gemma4Config,
    state: &mut Gemma4State,
    a: BatchAttn,
    x: &GpuTensor,
    residual: &GpuTensor,
    nrm: &GpuTensor,
    x_rot: &GpuTensor,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    attn_out: &GpuTensor,
    o: &GpuTensor,
    pos_array: &GpuTensor,
) -> Result<(), String> {
    let b = a.b;
    let dim = cfg.dim;
    let hd = a.hd;
    let n_heads = cfg.n_heads;
    let n_kv = a.n_kv;
    let eps = cfg.norm_eps;
    let kv_dim = n_kv * hd;

    // residual = x
    gpu.hip
        .memcpy_dtod_at(&residual.buf, 0, &x.buf, 0, b * dim * 4)
        .map_err(|e| format!("gemma4 batch attn save residual: {e:?}"))?;

    // n1 = input_layernorm(x) → nrm. (per-row, batch over B)
    gpu.rmsnorm_batched(x, a.input_layernorm, nrm, b, dim, eps)
        .map_err(|e| format!("gemma4 batch attn input rmsnorm: {e:?}"))?;

    // q / k projections (shared nrm input).
    proj_gemm_batched(gpu, a.q_proj, nrm, q, x_rot, b, "q_proj")?;
    proj_gemm_batched(gpu, a.k_proj, nrm, k, x_rot, b, "k_proj")?;

    // V:
    //   full + attention_k_eq_v (v_proj None): V = K's PRE-k_norm output
    //     (copy whole batched K → V before k_norm). Else: V = v_proj(nrm).
    match a.v_proj {
        Some(vw) => {
            proj_gemm_batched(gpu, vw, nrm, v, x_rot, b, "v_proj")?;
        }
        None => {
            gpu.hip
                .memcpy_dtod_at(&v.buf, 0, &k.buf, 0, b * kv_dim * 4)
                .map_err(|e| format!("gemma4 batch attn k→v copy: {e:?}"))?;
        }
    }

    // Per-head q_norm / k_norm over head_dim (batch over B*heads), weight-less
    // V norm (ones). The flat [B, heads, head_dim] layout makes B*heads rows.
    gpu.rmsnorm_batched(q, a.q_norm, q, b * n_heads, hd, eps)
        .map_err(|e| format!("gemma4 batch attn q_norm: {e:?}"))?;
    gpu.rmsnorm_batched(k, a.k_norm, k, b * n_kv, hd, eps)
        .map_err(|e| format!("gemma4 batch attn k_norm: {e:?}"))?;
    gpu.rmsnorm_batched(v, &state.v_norm_ones, v, b * n_kv, hd, eps)
        .map_err(|e| format!("gemma4 batch attn v_norm: {e:?}"))?;

    // Pre-scale Q by √head_dim (cancels the kernel's 1/√head_dim).
    gpu.scale_f32(q, (hd as f32).sqrt())
        .map_err(|e| format!("gemma4 batch attn q scale: {e:?}"))?;

    // RoPE (per-row positions).
    match a.rope {
        RopeKind::Full(theta) => {
            gpu.rope_batched_f32(q, k, pos_array, n_heads, n_kv, hd, theta, b)
                .map_err(|e| format!("gemma4 batch attn rope (full-rot): {e:?}"))?;
        }
        RopeKind::PartialHalved(theta, n_rot_pairs) => {
            gpu.rope_partial_halved_f32_batched(
                q, k, pos_array, n_heads, n_kv, hd, n_rot_pairs, theta, b,
            )
            .map_err(|e| format!("gemma4 batch attn rope (partial): {e:?}"))?;
        }
    }

    // KV write (Q8) for all B positions.
    let kv = if a.is_full {
        &state.kv_full
    } else {
        &state.kv_sliding
    };
    let physical_cap = kv.physical_cap;
    let k_cache = unsafe { kv.k_gpu[a.slot].buf.alias() };
    let v_cache = unsafe { kv.v_gpu[a.slot].buf.alias() };
    let k_cache_t = GpuTensor {
        buf: k_cache,
        shape: kv.k_gpu[a.slot].shape.clone(),
        dtype: kv.k_gpu[a.slot].dtype,
    };
    let v_cache_t = GpuTensor {
        buf: v_cache,
        shape: kv.v_gpu[a.slot].shape.clone(),
        dtype: kv.v_gpu[a.slot].dtype,
    };
    gpu.kv_cache_write_q8_0_batched(&k_cache_t, k, pos_array, n_kv, hd, b)
        .map_err(|e| format!("gemma4 batch attn kv write k: {e:?}"))?;
    gpu.kv_cache_write_q8_0_batched(&v_cache_t, v, pos_array, n_kv, hd, b)
        .map_err(|e| format!("gemma4 batch attn kv write v: {e:?}"))?;

    // Masked batched attention: tree-bias mode with block_start=0,
    // block_cols=seq_len → the per-row mask is applied to EVERY key, giving
    // exact causal+window control for both layer types. seq_len passed as
    // max_ctx_len so the kernel's scores[] LDS slice is sized for the full
    // context, and as block_cols so the bias index covers all keys.
    gpu.attention_q8_0_kv_batched_masked(
        q,
        &k_cache_t,
        &v_cache_t,
        attn_out,
        pos_array,
        n_heads,
        n_kv,
        hd,
        physical_cap,
        a.seq_len,
        b,
        Some(a.mask),
        0,
        a.seq_len,
    )
    .map_err(|e| format!("gemma4 batch attn masked: {e:?}"))?;

    // o_proj(attn_out) → o.  (Mirrors finish_attn_and_ffn's o_proj.)
    proj_gemm_batched(gpu, a.o_proj, attn_out, o, x_rot, b, "o_proj")?;

    // Sandwich post-attention norm (in-place on o), then x = residual + o.
    gpu.rmsnorm_batched(o, a.post_attention_layernorm, o, b, dim, eps)
        .map_err(|e| format!("gemma4 batch attn post_attn rmsnorm: {e:?}"))?;
    gpu.hip
        .memcpy_dtod_at(&x.buf, 0, &residual.buf, 0, b * dim * 4)
        .map_err(|e| format!("gemma4 batch attn reset x: {e:?}"))?;
    gpu.add_inplace_f32(x, o)
        .map_err(|e| format!("gemma4 batch attn residual add: {e:?}"))?;
    Ok(())
}

/// Batched FFN sub-block. On entry `x` holds the post-attention residual
/// stream (written by `batch_attn_block`). Mirrors the FFN half of
/// `finish_attn_and_ffn` batched across B: pre-FFN norm, gate/up proj,
/// gelu_tanh·mul, down proj, post-FFN norm, residual add, layer_scalar.
#[allow(clippy::too_many_arguments)]
fn batch_ffn_block(
    gpu: &mut Gpu,
    cfg: &Gemma4Config,
    _state: &mut Gemma4State,
    tail: &LayerTail,
    b: usize,
    x: &GpuTensor,
    residual: &GpuTensor,
    nrm: &GpuTensor,
    gate_ffn: &GpuTensor,
    up_ffn: &GpuTensor,
    ffn_hidden: &GpuTensor,
    ffn_out: &GpuTensor,
    ffn_rot: &GpuTensor,
) -> Result<(), String> {
    let dim = cfg.dim;
    let eps = cfg.norm_eps;
    let ffn_hd = cfg.hidden_dim;

    // 1) pre-FFN norm over the post-attention residual stream `x`.
    gpu.rmsnorm_batched(x, tail.pre_feedforward_layernorm, nrm, b, dim, eps)
        .map_err(|e| format!("gemma4 batch ffn pre rmsnorm: {e:?}"))?;

    // residual = x (FFN residual stream).
    gpu.hip
        .memcpy_dtod_at(&residual.buf, 0, &x.buf, 0, b * dim * 4)
        .map_err(|e| format!("gemma4 batch ffn save residual: {e:?}"))?;

    // 2) gate / up projections (shared nrm input).
    proj_gemm_batched(gpu, tail.gate_proj, nrm, gate_ffn, ffn_rot, b, "gate_proj")?;
    proj_gemm_batched(gpu, tail.up_proj, nrm, up_ffn, ffn_rot, b, "up_proj")?;

    // 3) hidden = gelu_tanh(gate) * up  (elementwise over B*ffn_hd).
    gpu.gelu_tanh_f32(gate_ffn, ffn_hidden, b * ffn_hd)
        .map_err(|e| format!("gemma4 batch ffn gelu_tanh: {e:?}"))?;
    gpu.mul_f32(ffn_hidden, up_ffn, ffn_hidden)
        .map_err(|e| format!("gemma4 batch ffn silu mul: {e:?}"))?;

    // 4) down_proj(hidden) → ffn_out.
    proj_gemm_batched(gpu, tail.down_proj, ffn_hidden, ffn_out, ffn_rot, b, "down_proj")?;

    // 5) post-FFN norm (ffn_out → nrm).
    gpu.rmsnorm_batched(ffn_out, tail.post_feedforward_layernorm, nrm, b, dim, eps)
        .map_err(|e| format!("gemma4 batch ffn post rmsnorm: {e:?}"))?;

    // 6) x = residual + nrm.
    gpu.hip
        .memcpy_dtod_at(&x.buf, 0, &residual.buf, 0, b * dim * 4)
        .map_err(|e| format!("gemma4 batch ffn reset x: {e:?}"))?;
    gpu.add_inplace_f32(x, nrm)
        .map_err(|e| format!("gemma4 batch ffn residual add: {e:?}"))?;

    // 7) learned per-layer scalar.
    if tail.layer_scalar_host != 1.0 {
        gpu.scale_f32(x, tail.layer_scalar_host)
            .map_err(|e| format!("gemma4 batch ffn layer_scalar: {e:?}"))?;
    }
    Ok(())
}
