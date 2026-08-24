// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

use crate::config_cache;
use crate::deepseek4::{DeepseekV4Config, DeepseekV4State, DeepseekV4Weights};
use crate::forward::{
    Deepseek4Bindings, OloraSchedule, apply_tail_rope, apply_tail_rope_batched, attn_stub, ds4_superop, ffn_routed, ffn_stub,
    gemv_auto, hc_attn_mix, hc_ffn_mix, kv_joint, mhc_pre, q_lora,
    weight_needs_fwht, precompute_attn_state_batched, precompute_positions_batched,
};
use crate::forward::{
    attention_block_batched_swa_only, ffn_batched, gemv_auto_batched_wmma,
    hc_attn_mix_batched, hc_ffn_mix_batched, kv_joint_batched, mhc_pre_batched,
    q_lora_batched,
};
use hipfire_dispatch::pipeline::superop::SuperOpKind;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::forward::PrefillBatchScratch;
/// DeepSeek V4 Multi-Token Prediction (MTP) forward step — DeepSeek V3 §4.
///
/// Predicts the **next-next** token given:
///   - `h_n`         : hidden state at absolute position N (the output of
///                     the main forward at that position, before the head)
///   - `next_token`  : the token that was emitted at position N+1
///   - `position`    : absolute position N+1 (used by tail-RoPE)
///
/// Output: logits over the vocab for position N+2.
///
/// Architecture (from `mtp.0.*` weights in DeepSeek V4-MTP HFQ files):
/// ```text
/// e_norm     = enorm(embed_lookup(next_token))
/// h_norm     = hnorm(h_n)
/// x_in       = e_proj @ e_norm + h_proj @ h_norm         (Q8F16 GEMVs)
/// x_attn     = attention(attn_norm(x_in))   + x_in        (SWA-only — no compressor)
/// x_ffn      = ffn(ffn_norm(x_attn))        + x_attn      (shared + routed MoE)
/// h_n_plus_1 = mtp_final_norm(x_ffn)
/// logits     = shared_head @ h_n_plus_1                   (reuses main lm_head)
/// ```
///
/// The MTP layer has NO compressor and NO indexer (verified against the
/// safetensors tensor table: only standard attn + FFN weights). Its
/// attention block is the SWA-only path (same as a hash-routed main
/// layer's attention).
///
/// **Status**: M1+M2 (weights ingest) are landed; the standard layer
/// block (attn + FFN with MTP weights) is still pending — the existing
/// per-layer helpers (`q_lora`, `kv_joint`, `attn_stub`, ...) all read
/// `weights.layers[layer_idx]` and need refactoring to accept a
/// `&DeepseekV4LayerWeights` parameter so they can run against
/// `weights.mtp_layer`. Filling in that refactor is M3-complete; it
/// will land alongside validation against the new HFQ that contains
/// the MTP layer.
///
/// Until then this function returns a clear error so callers can stub
/// out the spec-decode path without false-positive bring-up.
pub fn mtp_forward(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    h_n: &GpuTensor,
    next_token: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    let mtp_layer_idx = cfg.num_hidden_layers;
    // Steps 0–6: validate MTP weights + embed/norm/HC plumbing + attention.
    mtp_pre_ffn(cfg, weights, state, gpu, h_n, next_token, position)?;
    // FFN block (== ds4_moe_block_core at the MTP layer: mhc_pre(ffn) + shared
    // ffn_stub + routed ffn_routed + hc_ffn_mix). Single-GPU: routed combines
    // into ffn_out alongside the shared expert; the mix folds it.
    mhc_pre(
        cfg,
        weights,
        state,
        gpu,
        mtp_layer_idx,
        /*is_attn=*/ false,
    )?;
    ffn_stub(cfg, weights, state, gpu, mtp_layer_idx)?;
    ffn_routed(cfg, weights, state, gpu, mtp_layer_idx, None)?;
    hc_ffn_mix(cfg, weights, state, gpu, mtp_layer_idx)?;
    // Step 7: capture full HC residual → mtp_last_hidden (chaining input).
    mtp_capture_hidden(cfg, state, gpu)?;
    // SKIP_HEAD short-circuit (prefill MTP-fill: only the SWA write matters).
    if hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD")
        .ok()
        .as_deref()
        == Some("1")
    {
        return Ok(Vec::new());
    }
    // Steps 8–9: final norm + lm_head + download.
    mtp_head(cfg, weights, state, gpu)
}

/// Steps 0–6 of the MTP forward: validate MTP weights, embed `next_token`,
/// rmsnorm both inputs, populate the `[hc_mult, hidden]` residual streams via
/// the HC plumbing, and run the MTP-layer attention block (up to `hc_attn_mix`).
/// Shared by [`mtp_forward`] (single-GPU) and [`mtp_forward_ep`] (per rank,
/// replicated — only the FFN routed experts are sharded under EP).
fn mtp_pre_ffn(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    h_n: &GpuTensor,
    next_token: u32,
    position: u32,
) -> Result<(), String> {
    // ── 0. Validate MTP weights are present ────────────────────────────
    let mtp = weights.mtp_layer.as_ref().ok_or_else(|| {
        "mtp_forward: weights.mtp_layer is None — \
            re-quantize DeepSeek V4 with --format deepseek4-q8-mtp to include the \
            mtp.0.* tensors, then HIPFIRE_DEEPSEEK4_LOAD_MTP=1 at load time. \
            Files without an MTP layer cannot run spec-decode."
            .to_string()
    })?;
    let mtp_enorm = mtp
        .mtp_enorm
        .as_ref()
        .ok_or("mtp_forward: mtp_enorm missing")?;
    let mtp_hnorm = mtp
        .mtp_hnorm
        .as_ref()
        .ok_or("mtp_forward: mtp_hnorm missing")?;
    let mtp_e_proj = mtp
        .mtp_e_proj
        .as_ref()
        .ok_or("mtp_forward: mtp_e_proj missing")?;
    let mtp_h_proj = mtp
        .mtp_h_proj
        .as_ref()
        .ok_or("mtp_forward: mtp_h_proj missing")?;

    // Defensive: step 4 below passes `dummy_rotated` aliasing the OTHER
    // norm scratch (not a real FWHT rotation). That's safe for Q8_0 /
    // F16 / F32 dtypes since gemv_auto reads only x_plain on those
    // paths. For MQ4 (Raw dtype) gemv_auto reads x_rotated → we'd feed
    // garbage and produce silent NaN cascades. Reject upfront with a
    // clear message; if someone wants MQ4 MTP they need to plumb proper
    // rotated buffers through step 4.
    for (name, t) in [("mtp_e_proj", mtp_e_proj), ("mtp_h_proj", mtp_h_proj)] {
        match t.dtype {
            DType::F32 | DType::F16 | DType::Q8_0 => {}
            other => {
                return Err(format!(
                    "mtp_forward: {name} dtype {other:?} unsupported — step 4 \
                 only plumbs plain input (no FWHT rotation). Add rotated \
                 buffers or re-quant MTP at Q8F16 / F16."
                ));
            }
        }
    }

    // Full-HC plumbing: h_n must be the previous position's complete
    // [hc_mult, hidden] residual stream (per antirez/ds4 reference). The
    // legacy `[hidden]` shape (stream 0 only) is rejected — it produces
    // the broken ~50% acceptance path. final_norm_and_head and the
    // verify-pass capture in spec_decode now populate the full stream.
    let expected_shape = [cfg.hc_mult, cfg.hidden_size];
    if h_n.shape != expected_shape && h_n.numel() != cfg.hc_mult * cfg.hidden_size {
        return Err(format!(
            "mtp_forward: h_n shape {:?} != [hc_mult={}, hidden_size={}]",
            h_n.shape, cfg.hc_mult, cfg.hidden_size,
        ));
    }
    if cfg.num_nextn_predict_layers == 0 {
        return Err("mtp_forward: cfg.num_nextn_predict_layers == 0; MTP not enabled".to_string());
    }

    let hidden = cfg.hidden_size;
    let hc_mult = cfg.hc_mult;
    // MTP layer occupies the slot just past the main layers; resolve_layer
    // routes the per-layer helpers below to `weights.mtp_layer`.
    let mtp_layer_idx = cfg.num_hidden_layers;

    // ── 1. Lazy state allocation ───────────────────────────────────────
    if state.embed_scratch.is_none() {
        state.embed_scratch = Some(
            gpu.alloc_tensor(&[hidden], DType::F32)
                .map_err(|e| format!("alloc embed_scratch: {e:?}"))?,
        );
    }
    if state.residual_streams.is_none() {
        let t = gpu
            .zeros(&[hc_mult, hidden], DType::F32)
            .map_err(|e| format!("alloc residual_streams: {e:?}"))?;
        state.residual_streams = Some(t);
    }
    if state.tmp.is_none() {
        state.tmp = Some(
            gpu.alloc_tensor(&[hidden], DType::F32)
                .map_err(|e| format!("alloc tmp: {e:?}"))?,
        );
    }
    if state.mtp_e_norm_scratch.is_none() {
        state.mtp_e_norm_scratch = Some(
            gpu.alloc_tensor(&[hidden], DType::F32)
                .map_err(|e| format!("alloc mtp_e_norm_scratch: {e:?}"))?,
        );
    }
    // mtp_h_norm_scratch holds the per-HC-row rmsnorm output, sized
    // [hc_mult, hidden] for the full-HC pipeline. Realloc if shape grew
    // from a legacy [hidden] allocation.
    let h_norm_len = hc_mult * hidden;
    let h_norm_needs_realloc = state
        .mtp_h_norm_scratch
        .as_ref()
        .map(|t| t.numel() != h_norm_len)
        .unwrap_or(true);
    if h_norm_needs_realloc {
        state.mtp_h_norm_scratch = Some(
            gpu.alloc_tensor(&[hc_mult, hidden], DType::F32)
                .map_err(|e| format!("alloc mtp_h_norm_scratch: {e:?}"))?,
        );
    }
    if state.logits.is_none() {
        state.logits = Some(
            gpu.alloc_tensor(&[cfg.vocab_size], DType::F32)
                .map_err(|e| format!("alloc logits: {e:?}"))?,
        );
    }

    let token_embd = weights
        .token_embd
        .as_ref()
        .ok_or("mtp_forward: token_embd not uploaded")?;

    // ── 2. Embed next_token → embed_scratch [hidden] ───────────────────
    {
        let embed_scratch = state.embed_scratch.as_ref().unwrap();
        gpu.embedding_lookup_q8(token_embd, embed_scratch, next_token, hidden)
            .map_err(|e| format!("mtp embedding_lookup_q8: {e:?}"))?;
    }

    // ── 3. RMSNorm both inputs ─────────────────────────────────────────
    // e_norm = mtp_enorm(embed)              → mtp_e_norm_scratch [hidden]
    // h_norm = mtp_hnorm(h_n) per HC row    → mtp_h_norm_scratch [hc_mult, hidden]
    {
        let embed_scratch = state.embed_scratch.as_ref().unwrap();
        let e_out = state.mtp_e_norm_scratch.as_ref().unwrap();
        gpu.rmsnorm_f32(embed_scratch, mtp_enorm, e_out, cfg.rms_norm_eps)
            .map_err(|e| format!("mtp rmsnorm_e: {e:?}"))?;
    }
    {
        let h_out = state.mtp_h_norm_scratch.as_ref().unwrap();
        gpu.rmsnorm_batched(h_n, mtp_hnorm, h_out, hc_mult, hidden, cfg.rms_norm_eps)
            .map_err(|e| format!("mtp rmsnorm_h batched: {e:?}"))?;
    }

    // ── 4. Populate residual_streams from full HC plumbing ─────────────
    // Per antirez/ds4 reference:
    //   1. x_e   = mtp_e_proj @ e_norm                  (single [hidden])
    //   2. residual_streams[h] = mtp_h_proj @ h_norm[h] for each HC row h
    //   3. residual_streams[h] += x_e                  (broadcast e_proj)
    //
    // gemv_auto dispatches on the weight's GpuTensor.dtype:
    //   - Q8F16 → gemv_q8_0   (plain input)
    //   - F16   → gemv_f16_xf32 / gemm_f16_x_f16_wmma  (plain input)
    //   - MQ4   → gemv_mq4g256_prerotated              (rotated input)
    //
    // For MQ4 (Raw) MTP weights we'd need FWHT-rotated norm outputs; the
    // upfront dtype check (above) rejects MQ4 e_proj/h_proj for that
    // reason. With Q8/F16/F32 the `x_rotated` argument is unused; we pass
    // mtp_h_norm_scratch (any tensor of correct size) as a dummy.
    {
        let e_norm = state.mtp_e_norm_scratch.as_ref().unwrap();
        let dummy_rotated = state.mtp_h_norm_scratch.as_ref().unwrap();
        let tmp = state.tmp.as_ref().unwrap();
        gemv_auto(
            gpu,
            weights.mq2r_backend,
            mtp_e_proj,
            dummy_rotated,
            e_norm,
            tmp,
            hidden,
            hidden,
        )?;
    }
    {
        let h_norm_full = state.mtp_h_norm_scratch.as_ref().unwrap();
        let streams = state.residual_streams.as_ref().unwrap();
        let dummy_rotated = state.mtp_e_norm_scratch.as_ref().unwrap();
        // Per-HC-row h_proj. mtp_h_proj is the same [hidden, hidden]
        // weight matrix for every row; the inputs differ (per-row h_norm).
        // Tried batched GEMM (B=hc_mult=4) for weight-load amortization —
        // measured 5% SLOWER (17.07 → 16.05 tok/s at K=3) because the
        // batched-chunked Q8 path has setup overhead that beats the
        // amortization at B=4. Keep the per-row loop.
        for h in 0..hc_mult {
            let h_norm_row = h_norm_full.sub_offset(h * hidden, hidden);
            let dst_row = streams.sub_offset(h * hidden, hidden);
            gemv_auto(
                gpu,
                weights.mq2r_backend,
                mtp_h_proj,
                dummy_rotated,
                &h_norm_row,
                &dst_row,
                hidden,
                hidden,
            )?;
        }
    }
    // ── 5. Broadcast-add x_e (in state.tmp) into every HC row ─────────
    {
        let streams = state.residual_streams.as_ref().unwrap();
        let src = state.tmp.as_ref().unwrap();
        for h in 0..hc_mult {
            let row = streams.sub_offset(h * hidden, hidden);
            gpu.add_inplace_f32(&row, src)
                .map_err(|e| format!("mtp x_e broadcast-add stream {h}: {e:?}"))?;
        }
    }

    // ── 6. Standard layer block at layer_idx = num_hidden_layers ───────
    // All per-layer helpers below call `weights.resolve_layer(layer_idx)`
    // internally, which routes to `weights.mtp_layer`. The MTP layer has
    // NO compressor/indexer (compress_ratio = 0 by construction), and is
    // NOT a hash layer (mtp_layer_idx >= num_hash_layers), so we use the
    // standard MoE router (`ffn_routed`) rather than `ffn_hash_routed`.
    mhc_pre(
        cfg,
        weights,
        state,
        gpu,
        mtp_layer_idx,
        /*is_attn=*/ true,
    )?;
    q_lora(cfg, weights, state, gpu, mtp_layer_idx)?;
    kv_joint(cfg, weights, state, gpu, mtp_layer_idx, false)?;
    apply_tail_rope(cfg, weights, state, gpu, position, mtp_layer_idx)?;
    // (No compressor / indexer for MTP — compress_ratio == 0.)
    attn_stub(
        cfg,
        weights,
        state,
        gpu,
        mtp_layer_idx,
        OloraSchedule::Default,
    )?;
    hc_attn_mix(cfg, weights, state, gpu, mtp_layer_idx)?;
    Ok(())
}

/// Step 7 of the MTP forward: capture the full `[hc_mult, hidden]` residual
/// stream into `state.mtp_last_hidden` (the chaining input to the next MTP
/// iteration). Shared by [`mtp_forward`] and [`mtp_forward_ep`].
fn mtp_capture_hidden(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let hc_mult = cfg.hc_mult;
    // ── 7. Capture FULL [hc_mult, hidden] residual stream for chaining ─
    // Subsequent MTP iterations consume this as their h_n input. The
    // full-HC capture matches the antirez/ds4 reference pattern; legacy
    // stream-0-only capture is what pinned K=2 accept at ~50%.
    {
        let stream_len = hc_mult * hidden;
        let need_realloc = state
            .mtp_last_hidden
            .as_ref()
            .map(|t| t.numel() != stream_len)
            .unwrap_or(true);
        if need_realloc {
            state.mtp_last_hidden = Some(
                gpu.alloc_tensor(&[hc_mult, hidden], DType::F32)
                    .map_err(|e| format!("alloc mtp_last_hidden: {e:?}"))?,
            );
        }
        let streams = state.residual_streams.as_ref().unwrap();
        let dst = state.mtp_last_hidden.as_ref().unwrap();
        gpu.memcpy_dtod_auto(&dst.buf, &streams.buf, stream_len * 4)
            .map_err(|e| format!("capture full HC → mtp_last_hidden: {e:?}"))?;
    }
    Ok(())
}

/// Step 8 of the MTP forward: final norm (stream-0 or head-HC mix) + lm_head →
/// `state.logits` (NO download). Mirrors the main-model `final_norm_and_head`.
/// Shared by [`mtp_head`] (single-GPU, adds the download) and [`mtp_forward_ep`]
/// (rank 0 only, downloads after an all-ranks sync). The `MTP_SKIP_HEAD`
/// short-circuit lives in the callers (it must skip this entirely).
fn mtp_head_compute(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let hc_mult = cfg.hc_mult;
    let mtp = weights
        .mtp_layer
        .as_ref()
        .ok_or("mtp_head: weights.mtp_layer is None")?;
    let mtp_final = mtp
        .mtp_final_norm
        .as_ref()
        .ok_or("mtp_head: mtp_final_norm missing")?;
    let head = weights.head.as_ref().ok_or("mtp_head: head not uploaded")?;
    // ── 8. final_norm + lm_head → logits ──────────────────────────────
    // Two paths (mirrors the main-model `final_norm_and_head`):
    //   - default (legacy): stream 0 → mtp_final_norm → lm_head.
    //   - HIPFIRE_DEEPSEEK4_MTP_HEAD_HC=1: head-HC mix(streams, mtp.0.hc_head_*)
    //     → mtp_final_norm → lm_head (reduces [hc_mult, hidden] → [hidden]).
    if state.final_norm.is_none() {
        state.final_norm = Some(
            gpu.alloc_tensor(&[hidden], DType::F32)
                .map_err(|e| format!("alloc final_norm: {e:?}"))?,
        );
    }
    if state.final_norm_rot.is_none() {
        state.final_norm_rot = Some(
            gpu.alloc_tensor(&[hidden], DType::F32)
                .map_err(|e| format!("alloc final_norm_rot: {e:?}"))?,
        );
    }
    // Run head-HC mix or legacy stream-0 path; result lands in
    // `state.final_norm` via rmsnorm.
    let use_head_hc = config_cache::mtp_head_hc_on()
        && mtp.mtp_hc_head_fn.is_some()
        && mtp.mtp_hc_head_base.is_some();
    if use_head_hc {
        if state.head_hc_pre.is_none() {
            state.head_hc_pre = Some(
                gpu.alloc_tensor(&[hc_mult], DType::F32)
                    .map_err(|e| format!("alloc head_hc_pre (mtp): {e:?}"))?,
            );
        }
        if state.head_hc_out.is_none() {
            state.head_hc_out = Some(
                gpu.alloc_tensor(&[hidden], DType::F32)
                    .map_err(|e| format!("alloc head_hc_out (mtp): {e:?}"))?,
            );
        }
        let streams = state.residual_streams.as_ref().unwrap();
        let head_hc_pre = state.head_hc_pre.as_ref().unwrap();
        let head_hc_out = state.head_hc_out.as_ref().unwrap();
        let hc_head_fn = mtp.mtp_hc_head_fn.as_ref().unwrap();
        let hc_head_base = mtp.mtp_hc_head_base.as_ref().unwrap();
        let x_dim = hidden * hc_mult;
        gpu.hc_head_compute_pre(
            streams,
            hc_head_fn,
            hc_head_base,
            head_hc_pre,
            hc_mult as i32,
            x_dim as i32,
            mtp.mtp_hc_head_scale,
            cfg.rms_norm_eps,
            cfg.hc_eps,
        )
        .map_err(|e| format!("mtp hc_head_compute_pre: {e:?}"))?;
        gpu.hc_input_map_4stream(head_hc_pre, streams, head_hc_out, hidden as i32)
            .map_err(|e| format!("mtp hc_input_map: {e:?}"))?;
        let final_norm = state.final_norm.as_ref().unwrap();
        gpu.rmsnorm_f32(head_hc_out, mtp_final, final_norm, cfg.rms_norm_eps)
            .map_err(|e| format!("mtp final rmsnorm (head-HC): {e:?}"))?;
    } else {
        let streams = state.residual_streams.as_ref().unwrap();
        let stream0 = streams.sub_offset(0, hidden);
        let final_norm = state.final_norm.as_ref().unwrap();
        gpu.rmsnorm_f32(&stream0, mtp_final, final_norm, cfg.rms_norm_eps)
            .map_err(|e| format!("mtp final rmsnorm (stream0): {e:?}"))?;
    }
    let final_norm = state.final_norm.as_ref().unwrap();
    let final_norm_rot = state.final_norm_rot.as_ref().unwrap();
    if weight_needs_fwht(head) {
        gpu.rotate_x_mq(final_norm, final_norm_rot, hidden)
            .map_err(|e| format!("mtp rotate final: {e:?}"))?;
    }
    {
        let logits = state.logits.as_ref().unwrap();
        gemv_auto(
            gpu,
            weights.mq2r_backend,
            head,
            final_norm_rot,
            final_norm,
            logits,
            cfg.vocab_size,
            hidden,
        )?;
    }

    Ok(())
}

/// Single-GPU MTP head: [`mtp_head_compute`] + download logits → host `Vec`.
/// EP (`mtp_forward_ep`) uses `mtp_head_compute` directly + an all-ranks sync
/// before the caller downloads, to avoid racing the head GEMV on `active_stream`.
fn mtp_head(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<Vec<f32>, String> {
    mtp_head_compute(cfg, weights, state, gpu)?;
    let logits = state.logits.as_ref().unwrap();
    gpu.download_f32(logits)
        .map_err(|e| format!("mtp download logits: {e:?}"))
}

/// EP (Ship 6 substrate-EP) MTP **draft** forward across N ranks for ONE
/// next-next prediction — the spec-decode drafter under expert parallelism.
///
/// Mirror of [`mtp_forward`], fanned across `gpus.devices`: the MTP-specific
/// pre-FFN (embed / norm / HC plumbing + attention) runs replicated per rank,
/// the MTP-layer FFN runs through the SAME EP executor as the main layers
/// (shared `ffn_stub` replicated in `state.ffn_out`; the 256 routed experts
/// sharded → all-reduced partial; `hc_ffn_mix` deferred to
/// `ep_add_into_residual`), the residual capture runs per rank, and the head
/// runs on rank 0. Returns rank 0's downloaded logits (over the next-next
/// vocab). `mtp_last_hidden` is updated per rank (replicated) for chaining.
///
/// `h_n_per_rank[r]` is rank r's previous-position full `[hc_mult, hidden]`
/// residual stream (replicated; the chaining input) — it MUST be a buffer
/// DISTINCT from `state_per_rank[r].residual_streams` (pre-FFN reads `h_n` then
/// overwrites `residual_streams`). Every device needs an `active_stream`
/// ([`hipfire_runtime::ep::ensure_rank_streams`]) + peer access for the
/// all-reduce.
#[allow(clippy::too_many_arguments)]
pub fn mtp_forward_ep(
    gpus: &mut hipfire_runtime::multi_gpu::Gpus,
    weights_per_rank: &[DeepseekV4Weights],
    cfg: &DeepseekV4Config,
    state_per_rank: &mut [DeepseekV4State],
    partials: &[GpuTensor],
    h_n_per_rank: &[GpuTensor],
    next_token: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    let n = gpus.devices.len();
    assert_eq!(
        weights_per_rank.len(),
        n,
        "mtp_forward_ep: weights_per_rank len"
    );
    assert_eq!(
        state_per_rank.len(),
        n,
        "mtp_forward_ep: state_per_rank len"
    );
    assert_eq!(partials.len(), n, "mtp_forward_ep: partials len");
    assert_eq!(h_n_per_rank.len(), n, "mtp_forward_ep: h_n_per_rank len");
    let hidden = cfg.hidden_size;
    let mtp_layer_idx = cfg.num_hidden_layers;

    // 1. Per-rank pre-FFN (embed/norm/HC + attention), replicated. attn_stub
    //    reads state.n_tokens for the MTP-layer SWA ring slot → set it to
    //    `position` per rank (matches spec_decode's bookkeeping).
    for r in 0..n {
        gpus.devices[r]
            .bind_thread()
            .map_err(|e| format!("mtp_forward_ep bind {r}: {e:?}"))?;
        state_per_rank[r].n_tokens = position as u64;
        mtp_pre_ffn(
            cfg,
            &weights_per_rank[r],
            &mut state_per_rank[r],
            &mut gpus.devices[r],
            &h_n_per_rank[r],
            next_token,
            position,
        )?;
    }

    // 2. MTP-layer FFN via the EP executor: a single [Moe] program at
    //    layer_idx = mtp_layer_idx. run_moe_ep = mhc_pre(ffn) + shared ffn_stub
    //    + routed ffn_routed→partial; the executor all-reduces the partial;
    //    ep_add_into_residual = ffn_out += partial, then hc_ffn_mix.
    {
        let program = vec![ds4_superop(SuperOpKind::Moe)];
        let mut binds: Vec<Deepseek4Bindings> = Vec::with_capacity(n);
        for (r, st) in state_per_rank.iter_mut().enumerate() {
            binds.push(Deepseek4Bindings {
                cfg,
                weights: &weights_per_rank[r],
                state: st,
                layer_idx: mtp_layer_idx,
                position,
                token_id: next_token,
                skip_ffn: false,
            });
        }
        hipfire_runtime::ep::run_layer_program_ep(
            gpus,
            binds.as_mut_slice(),
            partials,
            &program,
            hidden,
        )
        .map_err(|e| format!("mtp_forward_ep run_layer_program_ep: {e}"))?;
    }

    // 3. Per-rank capture (residual_streams → mtp_last_hidden), replicated.
    for r in 0..n {
        gpus.devices[r]
            .bind_thread()
            .map_err(|e| format!("mtp_forward_ep cap bind {r}: {e:?}"))?;
        mtp_capture_hidden(cfg, &mut state_per_rank[r], &mut gpus.devices[r])?;
    }

    // 4. Head COMPUTE on rank 0 (no download — drained by the all-ranks sync).
    gpus.devices[0]
        .bind_thread()
        .map_err(|e| format!("mtp_forward_ep head bind0: {e:?}"))?;
    mtp_head_compute(
        cfg,
        &weights_per_rank[0],
        &mut state_per_rank[0],
        &mut gpus.devices[0],
    )?;

    // 5. Sync every rank, then download rank 0's logits.
    for r in 0..n {
        gpus.devices[r]
            .bind_thread()
            .map_err(|e| format!("mtp_forward_ep sync bind {r}: {e:?}"))?;
        gpus.devices[r]
            .hip
            .device_synchronize()
            .map_err(|e| format!("mtp_forward_ep sync {r}: {e:?}"))?;
    }
    gpus.devices[0]
        .bind_thread()
        .map_err(|e| format!("mtp_forward_ep dl bind0: {e:?}"))?;
    let logits = state_per_rank[0]
        .logits
        .as_ref()
        .ok_or("mtp_forward_ep: rank0 logits unset")?;
    gpus.devices[0]
        .download_f32(logits)
        .map_err(|e| format!("mtp_forward_ep download logits: {e:?}"))
}

/// Batched twin of `mtp_forward` — processes `batch_size` MTP positions
/// in a single pass through the MTP layer block (Phase A4, 2026-05-22).
///
/// Inputs:
/// - `h_n_streams`: `[batch_size, hc_mult, hidden]` — the per-batch full
///   HC residual streams from the main forward's `pbs.streams_batch`.
/// - `next_tokens`: `[batch_size]` — the next-position tokens (T_{i+1}).
/// - `start_pos`: absolute position of the first batch slot.
///
/// Post-state:
/// - `pbs.streams_batch` contains the per-batch MTP-layer output residuals.
/// - `state.mtp_last_hidden` contains the LAST batch position's MTP
///   output stream (the chaining input to subsequent spec-decode windows).
/// - `state._attention[mtp_layer_idx]` SWA cache has slots for the
///   processed positions written.
///
/// Skips lm_head + logits d2h (only the SWA-fill purpose is exercised).
///
/// At batch_size == 1 this is byte-equivalent to `mtp_forward` modulo
/// FP reduction-order noise inherent to the batched kernels.
#[allow(clippy::too_many_arguments)]
pub fn mtp_forward_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    pbs: &PrefillBatchScratch,
    h_n_streams: &GpuTensor,
    next_tokens: &[u32],
    start_pos: u32,
    batch_size: usize,
) -> Result<(), String> {
    if batch_size == 0 {
        return Err("mtp_forward_batched: batch_size == 0".to_string());
    }
    if batch_size > pbs.max_batch {
        return Err(format!(
            "mtp_forward_batched: batch_size {batch_size} > pbs.max_batch {}",
            pbs.max_batch
        ));
    }
    if next_tokens.len() != batch_size {
        return Err(format!(
            "mtp_forward_batched: next_tokens.len {} != batch_size {batch_size}",
            next_tokens.len()
        ));
    }
    let mtp = weights
        .mtp_layer
        .as_ref()
        .ok_or_else(|| "mtp_forward_batched: weights.mtp_layer is None".to_string())?;
    let mtp_enorm = mtp
        .mtp_enorm
        .as_ref()
        .ok_or("mtp_forward_batched: mtp_enorm missing")?;
    let mtp_hnorm = mtp
        .mtp_hnorm
        .as_ref()
        .ok_or("mtp_forward_batched: mtp_hnorm missing")?;
    let mtp_e_proj = mtp
        .mtp_e_proj
        .as_ref()
        .ok_or("mtp_forward_batched: mtp_e_proj missing")?;
    let mtp_h_proj = mtp
        .mtp_h_proj
        .as_ref()
        .ok_or("mtp_forward_batched: mtp_h_proj missing")?;
    for (name, t) in [("mtp_e_proj", mtp_e_proj), ("mtp_h_proj", mtp_h_proj)] {
        match t.dtype {
            DType::F32 | DType::F16 | DType::Q8_0 => {}
            other => {
                return Err(format!(
                    "mtp_forward_batched: {name} dtype {other:?} unsupported"
                ));
            }
        }
    }
    if cfg.num_nextn_predict_layers == 0 {
        return Err("mtp_forward_batched: cfg.num_nextn_predict_layers == 0".to_string());
    }

    let hidden = cfg.hidden_size;
    let hc_mult = cfg.hc_mult;
    let stream_len = hc_mult * hidden;
    let mtp_layer_idx = cfg.num_hidden_layers;

    // Lazy alloc state.mtp_last_hidden.
    if state.mtp_last_hidden.is_none() {
        state.mtp_last_hidden = Some(
            gpu.alloc_tensor(&[hc_mult, hidden], DType::F32)
                .map_err(|e| format!("alloc mtp_last_hidden: {e:?}"))?,
        );
    }

    let token_embd = weights
        .token_embd
        .as_ref()
        .ok_or("mtp_forward_batched: token_embd not uploaded")?;

    // ── 1. Upload next_tokens [batch_size] ─────────────────────────────
    let tokens_host: Vec<i32> = next_tokens.iter().map(|&t| t as i32).collect();
    let token_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, batch_size * 4) };
    gpu.memcpy_htod_auto(&pbs.mtp_tokens_batch.buf, token_bytes)
        .map_err(|e| format!("mtp_forward_batched htod tokens: {e:?}"))?;

    // ── 2. Batched embed → pbs.mtp_embed_batch ─────────────────────────
    gpu.embedding_lookup_q8_batched(
        token_embd,
        &pbs.mtp_embed_batch,
        &pbs.mtp_tokens_batch,
        batch_size,
        hidden,
    )
    .map_err(|e| format!("mtp embedding_lookup_q8_batched: {e:?}"))?;

    // ── 3. Batched RMSNorm both inputs ─────────────────────────────────
    // e_norm = mtp_enorm(embed_batch) → mtp_e_norm_batch [B, hidden]
    gpu.rmsnorm_batched(
        &pbs.mtp_embed_batch,
        mtp_enorm,
        &pbs.mtp_e_norm_batch,
        batch_size,
        hidden,
        cfg.rms_norm_eps,
    )
    .map_err(|e| format!("mtp rmsnorm_e batched: {e:?}"))?;
    // h_norm = mtp_hnorm(h_n_streams) per (batch, HC row) — treat as
    // batch_size * hc_mult rows of length hidden.
    gpu.rmsnorm_batched(
        h_n_streams,
        mtp_hnorm,
        &pbs.mtp_h_norm_batch,
        batch_size * hc_mult,
        hidden,
        cfg.rms_norm_eps,
    )
    .map_err(|e| format!("mtp rmsnorm_h batched: {e:?}"))?;

    // ── 4. Batched e_proj GEMV: mtp_e_norm_batch → mtp_x_e_batch ───────
    // dummy_rotated is unused for F32/F16/Q8 weight dtypes (guarded above).
    gemv_auto_batched_wmma(
        gpu,
        weights.mq2r_backend,
        mtp_e_proj,
        &pbs.mtp_h_norm_batch,
        &pbs.mtp_e_norm_batch,
        &pbs.mtp_x_e_batch,
        hidden,
        hidden,
        batch_size,
        None,
    )?;

    // ── 5. Batched h_proj GEMV — flatten (B, hc_mult) as one batch dim.
    // Input mtp_h_norm_batch [B * hc_mult, hidden] → output streams_batch
    // [B * hc_mult, hidden]. mtp_h_proj is the same weight for every
    // (batch, HC) row.
    gemv_auto_batched_wmma(
        gpu,
        weights.mq2r_backend,
        mtp_h_proj,
        &pbs.mtp_e_norm_batch,
        &pbs.mtp_h_norm_batch,
        &pbs.streams_batch,
        hidden,
        hidden,
        batch_size * hc_mult,
        None,
    )?;

    // ── 6. Broadcast-add x_e_b into every HC row of streams_batch_b ───
    // streams_batch[b][h] += mtp_x_e_batch[b] for h in 0..hc_mult, b in 0..B.
    for b in 0..batch_size {
        let x_e_b = pbs.mtp_x_e_batch.sub_offset(b * hidden, hidden);
        for h in 0..hc_mult {
            let off = b * stream_len + h * hidden;
            let row = pbs.streams_batch.sub_offset(off, hidden);
            gpu.add_inplace_f32(&row, &x_e_b)
                .map_err(|e| format!("mtp x_e add b={b} h={h}: {e:?}"))?;
        }
    }

    // ── 7. Populate per-batch positions + attn_state for the MTP layer.
    //   Positions: start_pos + b.
    //   attn_state: slot = (start_pos + b) % swa_window; n_valid = min(start_pos + b + 1, swa_window).
    precompute_positions_batched(cfg, pbs, gpu, start_pos, batch_size)?;
    precompute_attn_state_batched(cfg, pbs, gpu, start_pos, batch_size)?;

    // ── 8. Standard batched layer block at layer_idx = mtp_layer_idx ──
    // The MTP layer has compress_ratio = 0 so attention_block_batched_swa_only
    // is the right path. Hash routing is N/A (mtp_layer_idx >= num_hash_layers).
    let n = batch_size;
    let mtp_layer = weights.resolve_layer(mtp_layer_idx);
    mhc_pre_batched(
        cfg,
        mtp_layer,
        pbs,
        gpu,
        mtp_layer_idx,
        /*is_attn=*/ true,
        n,
    )?;
    let attention_input_precomputed = q_lora_batched(
        cfg,
        mtp_layer,
        weights.mq2r_backend,
        pbs,
        &pbs.hc_x_in_batch,
        gpu,
        mtp_layer_idx,
        n,
    )?;
    kv_joint_batched(
        cfg,
        mtp_layer,
        weights.mq2r_backend,
        pbs,
        gpu,
        mtp_layer_idx,
        n,
        attention_input_precomputed,
    )?;
    apply_tail_rope_batched(cfg, mtp_layer, pbs, gpu, mtp_layer_idx, n)?;
    attention_block_batched_swa_only(
        cfg,
        weights,
        state,
        pbs,
        gpu,
        mtp_layer_idx,
        start_pos,
        n,
        false,
    )?;
    hc_attn_mix_batched(cfg, pbs, gpu, n)?;
    let mtp_layer = weights.resolve_layer(mtp_layer_idx);
    mhc_pre_batched(
        cfg,
        mtp_layer,
        pbs,
        gpu,
        mtp_layer_idx,
        /*is_attn=*/ false,
        n,
    )?;
    // ffn_batched takes `tokens` for the hash-routed path; MTP layer is
    // not hash-routed (mtp_layer_idx >= num_hash_layers), so the value
    // is ignored. Pass an empty slice.
    let tokens_dummy: &[u32] = &[];
    ffn_batched(
        cfg,
        mtp_layer,
        weights.mq2r_backend,
        pbs,
        gpu,
        mtp_layer_idx,
        /*hash_routing=*/ false,
        n,
        tokens_dummy,
        // No pager on the MTP path: the addon's experts are resident.
        None,
    )?;
    hc_ffn_mix_batched(cfg, pbs, gpu, n)?;

    // ── 9. Capture the LAST batch position's residual stream → mtp_last_hidden.
    //    Subsequent spec-decode windows read from this.
    {
        let last_off = (batch_size - 1) * stream_len;
        let last_slice = pbs.streams_batch.sub_offset(last_off, stream_len);
        let dst = state.mtp_last_hidden.as_ref().unwrap();
        gpu.memcpy_dtod_auto(&dst.buf, &last_slice.buf, stream_len * 4)
            .map_err(|e| format!("mtp d2d streams[last] → mtp_last_hidden: {e:?}"))?;
    }

    Ok(())
}

