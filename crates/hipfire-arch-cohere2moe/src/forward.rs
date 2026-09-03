// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Cohere2-MoE (North-Mini-Code) forward pass (free functions, hot-path static
//! dispatch).
//!
//! The defining structural trait is the **parallel block**: a SINGLE
//! `RMSNorm` (cohere2_moe uses RMSNorm at `rms_norm_eps`, NOT base Cohere2's
//! mean-centered LayerNorm) feeds BOTH the attention and the FFN branch, and
//! both add into the residual —
//!   `h = h + o_proj(attn(RMSNorm(h))) + ffn(RMSNorm(h))`
//! (note: the FFN reads the SAME `RMSNorm(h)` as attention, NOT the
//! post-attention residual). Per layer:
//!   normed = rmsnorm(h, input_layernorm)                       [gamma scale, no mean-center / no β]
//!   q,k,v  = proj(normed); RoPE only if sliding (full=NoPE); attn; h += o_proj
//!   if dense (first_k_dense_replace prefix): h += down(silu(gate(normed))·up(normed))
//!   if moe:  router = sigmoid(gate·normed); top-8 (NO renorm: norm_topk_prob=false);
//!            h += Σ w_e · expert_e(normed)
//! then logits = lm_head(rmsnorm(h, model.norm)) · logit_scale.
//!
//! Routed experts: the MQ4/MQ6 tiers use the FWHT-pre-rotated indexed-MoE GEMV
//! kernels (exactly the qwen35/lfm2/minimax path). The F16 oracle and Q8 expert
//! tiers have no indexed kernel, so they take a per-expert `weight_gemv` loop
//! (correctness over speed — the KLD/PPL harness is offline).
//!
//! Attention: a per-layer NoPE/RoPE split (full_attention layers are NoPE;
//! sliding layers use interleaved RoPE), plus a windowed-mask flash path
//! (`attention_flash_q8_0_windowed`) that applies the sliding-window mask so
//! context beyond the 4096 window attends correctly. At ≤4096, sliding == full
//! causal, so the window is a no-op there.

use crate::cohere2moe::{Cohere2MoeState, Cohere2MoeWeights, Ffn};
use crate::config::{AttnKind, Cohere2MoeConfig};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::moe::{MoeDtypes, MoePrefillParams};
use hipfire_runtime::llama::KvCacheExt;
use hipfire_runtime::llama::{
    fused_silu_mul_rotate_mq_batched_for, moe_family, rotate_x_mq_batched_for, rotate_x_mq_for,
    weight_gemv, weight_gemv_residual,
};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Grouped-MoE prefill tiling constant — must match `run_moe_prefill`'s
/// `MOE_GROUPED_BLOCK_M` (tokens are scattered into per-expert groups padded to
/// a multiple of this).
const MOE_GROUPED_BLOCK_M: usize = 16;
#[inline]
fn align_up_usize(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}
/// Upper bound on the padded total scattered-slot count: every live expert can
/// waste up to `BLOCK_M-1` pad slots. `total_slots = batch * k_top`.
#[inline]
fn moe_grouped_m_total_bound(total_slots: usize, n_exp: usize) -> usize {
    let live = total_slots.min(n_exp);
    align_up_usize(
        total_slots + live * (MOE_GROUPED_BLOCK_M - 1),
        MOE_GROUPED_BLOCK_M,
    )
}

/// Batched Q8_0 projection GEMM for prefill (`Y[b,m] = X[b,k] @ W_q8[m,k]^T`).
/// On WMMA archs (gfx11+/RDNA3.5/RDNA4) with K%32==0 it routes to the
/// matrix-core `gemm_q8_0_wmma` (activation widened to F16 internally) — no
/// MAX_BATCH cap and far faster than the scalar 1-wave-per-row kernel at large
/// batch (mirrors the deepseek4 gfx11 Q8 prefill path). Falls back to the
/// scalar-sub-batched chunked driver on CDNA or when K%32≠0.
/// Opt out of the WMMA Q8 prefill projections (force the scalar sub-batched
/// chunked driver) via `HIPFIRE_COHERE2MOE_Q8_SCALAR=1` — a debugging /
/// precision fallback. WMMA is the default: bit-identical to scalar on the
/// validation prompt and ~9× faster prefill.
fn q8_wmma_enabled() -> bool {
    static EN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EN.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_COHERE2MOE_Q8_SCALAR").as_deref() != Ok("1")
    })
}

#[inline]
fn q8_proj_raw(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    b: usize,
    x_f16: &GpuTensor,
) -> hip_bridge::HipResult<()> {
    if q8_wmma_enabled() && gpu.arch_caps.has_wmma() && k % 32 == 0 {
        // Convert THIS activation to F16 fresh into our own buffer, then feed
        // the F16 tensor to gemm_q8_0_wmma. We must NOT let gemm_q8_0_wmma run
        // its internal `ensure_fp16_x`: that cache is keyed on the source
        // POINTER and only reconverts when the pointer changes — but `normed`
        // and `attn_out` are single buffers reused every layer with NEW
        // contents, so it would return layer-0's STALE F16 for layers 1..N.
        gpu.deepseek4_convert_f32_to_f16(x, x_f16, (b * k) as i64)?;
        gpu.gemm_q8_0_wmma(w, x_f16, y, m, k, b)
    } else {
        gpu.gemm_q8_0_batched_chunked(w, x, y, m, k, b)
    }
}

/// Decode one token; returns the full logits vector.
pub fn decode_step(
    cfg: &Cohere2MoeConfig,
    weights: &Cohere2MoeWeights,
    state: &mut Cohere2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    // Device position scalar (i32) for rope / kv-write / attention.
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("cohere2moe: htod pos: {e:?}"))?;
    embed_lookup(gpu, weights, cfg.hidden_size, token_id, &mut state.h)?;
    decode_step_body(cfg, weights, state, gpu, position)?;
    let mut logits = gpu
        .download_f32(&state.logits)
        .map_err(|e| format!("cohere2moe: download logits: {e:?}"))?;
    // logit_scale (1.0 for North-Mini-Code → no-op; applied host-side so the
    // device logits stay the raw lm_head output for any downstream re-use).
    if (cfg.logit_scale - 1.0).abs() > f32::EPSILON {
        for v in &mut logits {
            *v *= cfg.logit_scale;
        }
    }
    Ok(logits)
}

/// Seed the residual stream `out` with the embedding row for `token_id`.
/// Dispatches on the (tied) embed dtype: Q8 → dequant kernel; F32 → raw row
/// copy. (The F16 path is unused by the current tiers — embed/lm_head stay Q8
/// across the whole sweep, an engine constraint of the tied-embedding lookup.)
fn embed_lookup(
    gpu: &mut Gpu,
    weights: &Cohere2MoeWeights,
    hidden: usize,
    token_id: u32,
    out: &rdna_compute::GpuTensor,
) -> Result<(), String> {
    match weights.embed_dtype {
        DType::Q8_0 => gpu
            .embedding_lookup_q8(&weights.embed, out, token_id, hidden)
            .map_err(|e| format!("cohere2moe: embed lookup q8: {e:?}")),
        DType::F32 => gpu
            .embedding_lookup(&weights.embed, out, token_id, hidden)
            .map_err(|e| format!("cohere2moe: embed lookup f32: {e:?}")),
        other => Err(format!(
            "cohere2moe: embed dtype {other:?} has no lookup path (use Q8 or F32 tied embeddings)"
        )),
    }
}

/// Per-layer parallel-block stack + final norm + lm_head. Reads `state.h`
/// (seeded by the embedding lookup) and `state.pos_buf` (already staged).
fn decode_step_body(
    cfg: &Cohere2MoeConfig,
    weights: &Cohere2MoeWeights,
    state: &mut Cohere2MoeState,
    gpu: &mut Gpu,
    position: u32,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let moe_inter = cfg.moe_intermediate_size;
    let n_exp = cfg.num_experts;
    let k_top = cfg.num_experts_per_tok;
    let eps = cfg.rms_norm_eps;
    let seq_len = position as usize + 1;

    for (l, layer) in weights.layers.iter().enumerate() {
        // ── Parallel block: ONE RMSNorm → `normed`, fed to BOTH the attention
        //    and the FFN branch. cohere2_moe uses plain RMSNorm (LlamaRMSNorm,
        //    rms_norm_eps), NOT base Cohere2's mean-centered LayerNorm. ────────
        gpu.rmsnorm_batched(&state.h, &layer.input_norm, &state.normed, 1, hidden, eps)
            .map_err(|e| format!("cohere2moe L{l}: input rmsnorm: {e:?}"))?;

        // ── Attention branch (reads `normed`) ──────────────────────────────
        weight_gemv(gpu, &layer.wq, &state.normed, &state.fa_q)
            .map_err(|e| format!("cohere2moe L{l}: q_proj: {e}"))?;
        weight_gemv(gpu, &layer.wk, &state.normed, &state.fa_k)
            .map_err(|e| format!("cohere2moe L{l}: k_proj: {e}"))?;
        weight_gemv(gpu, &layer.wv, &state.normed, &state.fa_v)
            .map_err(|e| format!("cohere2moe L{l}: v_proj: {e}"))?;

        // NoPE: only `sliding_attention` layers apply RoPE. `full_attention`
        // (global) layers use NO positional embedding (Cohere2 sets
        // sliding_window=None there, gating off rotary).
        //
        // Cohere2 uses the **interleaved (GPT-J)** rotary convention — pairs
        // adjacent dims (2i, 2i+1) — NOT Llama's half-split. The HF
        // `rotate_half` is explicitly commented "different from e.g. Llama":
        // `x1=x[..., ::2]; x2=x[..., 1::2]; rot=stack([-x2,x1]).flatten`. So we
        // MUST use `rope_partial_interleaved_f32` (pairs 2i/2i+1), NOT
        // `rope_f32` (pairs i / i+head_dim/2). Rotary covers the FULL head_dim
        // (no partial_rotary_factor in the config) → n_rot = head_dim.
        // RoPE on sliding layers AND the dense prefix layers (force_rope:
        // l < first_k_dense_replace && prefix_dense_sliding_window_pattern == 1).
        // North layer 0 is full_attention but STILL rotated; the other global
        // full_attention layers are NoPE. (Matches Cohere2MoeAttention.)
        if layer.attn_kind == AttnKind::Sliding
            || (l < cfg.first_k_dense_replace && cfg.prefix_dense_sliding_window_pattern == 1)
        {
            gpu.rope_interleaved_f32(
                &state.fa_q,
                &state.fa_k,
                &state.pos_buf,
                n_heads,
                n_kv,
                head_dim,
                head_dim, // n_rot = full head_dim (no partial_rotary_factor)
                cfg.rope_theta,
            )
            .map_err(|e| format!("cohere2moe L{l}: rope: {e:?}"))?;
        }

        // KV write (Q8) + sliding-window flash attention via the shared KV-usage
        // abstraction. `sliding_attention` layers clip to the last
        // `sliding_window` keys (window>0); `full_attention` layers are full
        // causal (window=0). cohere routes through AttnFlashQ8_0Windowed
        // UNCONDITIONALLY (q8_windowed) — NOT the q8_attend_key flash/non-flash
        // heuristic, whose non-windowed AttnQ8_0Kv would drop the window and
        // attend the full context at ctx>window. (Tiled, O(1)-LDS, online-softmax
        // — no seq-bound shared-memory ceiling.)
        let window = if layer.attn_kind == AttnKind::Sliding {
            cfg.sliding_window as i32
        } else {
            0
        };
        let ctx = DispatchCtx::new(gpu);
        let plan = hipfire_dispatch::families::kv_tier::KvTierPlan::derive(
            hipfire_dispatch::families::kv_tier::KvTierInputs {
                pos: seq_len - 1,
                q8_windowed: true,
                window,
                ..state.kv.tier_inputs()
            },
        )
        .map_err(|e| format!("cohere2moe L{l}: kv tier: {e}"))?;
        let io = hipfire_dispatch::families::attention::AttnParams {
            q: &state.fa_q,
            k: &state.fa_k,
            v: &state.fa_v,
            k_cache: &state.kv.k_gpu[l],
            v_cache: &state.kv.v_gpu[l],
            k_scales: None,
            v_scales: None,
            pos_buf: &state.pos_buf,
            pos: seq_len - 1,
            positions: None,
            n_heads,
            n_kv_heads: n_kv,
            head_dim,
            physical_cap: state.kv.physical_cap,
            batch_size: 1,
            max_ctx_len: 0,
            flash_partials: Some(&state.flash_partials),
            givens_cos: None,
            givens_sin: None,
            tree_bias: None,
            block_start: 0,
            block_cols: 0,
            output_gate: None,
            output: &state.fa_attn_out,
        };
        hipfire_dispatch::pipeline::execute_steps(
            gpu,
            &ctx,
            &[hipfire_dispatch::pipeline::Step::Attend { plan, io }],
        )
        .map_err(|e| format!("cohere2moe L{l}: attention: {e:?}"))?;

        // h += o_proj · attn_out  (attention into the residual).
        weight_gemv_residual(gpu, &layer.wo, &state.fa_attn_out, &state.h)
            .map_err(|e| format!("cohere2moe L{l}: o_proj: {e}"))?;

        // ── FFN branch (reads the SAME `normed`, NOT post-attention `h`) ─────
        match &layer.ffn {
            Ffn::Dense(d) => {
                weight_gemv(gpu, &d.gate, &state.normed, &state.dense_gate)
                    .map_err(|e| format!("cohere2moe L{l}: dense gate_proj: {e}"))?;
                weight_gemv(gpu, &d.up, &state.normed, &state.dense_up)
                    .map_err(|e| format!("cohere2moe L{l}: dense up_proj: {e}"))?;
                gpu.silu_mul_f32(&state.dense_gate, &state.dense_up, &state.dense_act)
                    .map_err(|e| format!("cohere2moe L{l}: dense silu_mul: {e:?}"))?;
                weight_gemv_residual(gpu, &d.down, &state.dense_act, &state.h)
                    .map_err(|e| format!("cohere2moe L{l}: dense down_proj: {e}"))?;
            }
            Ffn::Moe(m) => {
                // Router: sigmoid(logits) → top-k. `norm_topk_prob=false` for
                // North-Mini-Code, so the top-8 raw sigmoid scores are the
                // combine weights (NO renormalization). Selection by sigmoid is
                // monotonic in the logits, so it matches HF `expert_selection_fn`.
                weight_gemv(gpu, &m.router, &state.normed, &state.router_logits)
                    .map_err(|e| format!("cohere2moe L{l}: router: {e}"))?;
                gpu.sigmoid_f32(&state.router_logits)
                    .map_err(|e| format!("cohere2moe L{l}: sigmoid: {e:?}"))?;
                gpu.moe_topk_renorm_k8(
                    &state.router_logits,
                    &state.topk_indices,
                    &state.topk_weights,
                    n_exp,
                    cfg.norm_topk_prob,
                )
                .map_err(|e| format!("cohere2moe L{l}: topk: {e:?}"))?;

                let edt = m.experts[0].gate_up.gpu_dtype;
                match edt {
                    // FWHT-pre-rotated indexed MoE GEMV (MQ4/MQ6 tiers).
                    DType::MQ4G256 | DType::HFQ4G256 | DType::MQ6G256 | DType::HFQ6G256 => {
                        let mq6 = matches!(edt, DType::MQ6G256 | DType::HFQ6G256);
                        rotate_x_mq_for(
                            gpu,
                            &m.experts[0].gate_up,
                            &state.normed,
                            &state.ffn_x_rot,
                            hidden,
                        )
                        .map_err(|e| format!("cohere2moe L{l}: ffn rotate: {e:?}"))?;
                        if mq6 {
                            gpu.gemv_hfq6g256_moe_gate_up_k8_indexed_batched(
                                &m.expert_gate_up_ptrs,
                                &state.topk_indices,
                                &state.ffn_x_rot,
                                &state.gate_batch,
                                &state.up_batch,
                                2 * moe_inter,
                                hidden,
                                k_top,
                                1,
                            )
                            .map_err(|e| format!("cohere2moe L{l}: gate_up(mq6): {e:?}"))?;
                        } else {
                            gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
                                &m.expert_gate_up_ptrs,
                                &state.topk_indices,
                                &state.ffn_x_rot,
                                &state.gate_batch,
                                &state.up_batch,
                                2 * moe_inter,
                                hidden,
                                k_top,
                                1,
                            )
                            .map_err(|e| format!("cohere2moe L{l}: gate_up(mq4): {e:?}"))?;
                        }
                        fused_silu_mul_rotate_mq_batched_for(
                            gpu,
                            &m.experts[0].down,
                            &state.gate_batch,
                            &state.up_batch,
                            &state.rot_batch,
                            moe_inter,
                            k_top,
                        )
                        .map_err(|e| format!("cohere2moe L{l}: silu_mul_rotate: {e:?}"))?;
                        if mq6 {
                            gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
                                &m.expert_down_ptrs,
                                &state.topk_indices,
                                &state.rot_batch,
                                &state.down_expanded,
                                hidden,
                                moe_inter,
                                k_top,
                                1,
                            )
                            .map_err(|e| format!("cohere2moe L{l}: down(mq6): {e:?}"))?;
                        } else {
                            gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                                &m.expert_down_ptrs,
                                &state.topk_indices,
                                &state.rot_batch,
                                &state.down_expanded,
                                hidden,
                                moe_inter,
                                k_top,
                                1,
                            )
                            .map_err(|e| format!("cohere2moe L{l}: down(mq4): {e:?}"))?;
                        }
                        gpu.moe_down_combine_k8_batched(
                            &state.down_expanded,
                            &state.topk_weights,
                            &state.h,
                            hidden,
                            k_top,
                            1,
                        )
                        .map_err(|e| format!("cohere2moe L{l}: combine: {e:?}"))?;
                    }
                    // Per-expert path for the bf16 oracle + Q8 tier (no indexed
                    // kernel for these dtypes). Reads the 8 selected experts off
                    // the device topk buffers and runs a plain GEMV each
                    // (weight_gemv → run_auto handles BF16/F16/F32/Q8).
                    DType::BF16 | DType::Q8_0 | DType::F16 | DType::F32 => {
                        moe_per_expert(gpu, m, state, moe_inter, k_top, l)?;
                    }
                    other => {
                        return Err(format!(
                            "cohere2moe L{l}: unsupported expert dtype {other:?}"
                        ))
                    }
                }
            }
        }
        if hipfire_config::developer_var_os("HIPFIRE_COHERE_DEBUG").is_some() {
            if let Ok(hv) = gpu.download_f32(&state.h) {
                let l2 = hv.iter().map(|v| v * v).sum::<f32>().sqrt();
                let mx = hv.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let nan = hv.iter().filter(|v| v.is_nan()).count();
                eprintln!(
                    "[dbg] L{l} {:?} h.l2={l2:.2} max={mx:.3} nan={nan}",
                    layer.attn_kind
                );
            }
        }
    }
    state.n_tokens = seq_len;

    // Final RMSNorm + lm_head (tied embed).
    gpu.rmsnorm_batched(
        &state.h,
        &weights.final_norm,
        &state.final_norm_buf,
        1,
        hidden,
        eps,
    )
    .map_err(|e| format!("cohere2moe: final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
        .map_err(|e| format!("cohere2moe: lm_head: {e}"))?;
    Ok(())
}

/// Per-expert SwiGLU for non-indexable expert dtypes (F16 oracle / Q8 tier).
/// Recovers the 8 selected expert ids from the device topk buffers (the
/// i32 indices are bit-preserved through `download_f32`), runs a plain
/// `weight_gemv` per selected expert, and accumulates `w_e · down(silu(gate)·up)`
/// into the residual. `normed` is the parallel-block layernorm output.
fn moe_per_expert(
    gpu: &mut Gpu,
    m: &crate::cohere2moe::MoeFfn,
    state: &Cohere2MoeState,
    moe_inter: usize,
    k_top: usize,
    l: usize,
) -> Result<(), String> {
    // i32 expert ids are stored in an F32-typed tensor; download_f32 is a
    // bit-preserving copy, so `.to_bits()` recovers the original index.
    let idx_bits = gpu
        .download_f32(&state.topk_indices)
        .map_err(|e| format!("cohere2moe L{l}: dl topk idx: {e:?}"))?;
    let weights = gpu
        .download_f32(&state.topk_weights)
        .map_err(|e| format!("cohere2moe L{l}: dl topk w: {e:?}"))?;
    for j in 0..k_top {
        // The router must produce in-range expert ids; a silent `.min()` clamp
        // would mask a routing/topk bug as quietly-wrong output (worst place to
        // be silent — this is the oracle/Q8 correctness path). Fail loudly.
        // (The batched `run_prefill` path instead clamps an OOB id to expert 0
        // in-kernel — bounded-wrong but memory-safe, since the indexed GEMM
        // can't take a runtime loop bound; that asymmetry is intentional.)
        let e = idx_bits[j].to_bits() as usize;
        if e >= m.experts.len() {
            return Err(format!(
                "cohere2moe L{l}: router produced OOB expert id {e} (n_experts={})",
                m.experts.len()
            ));
        }
        let w = weights[j];
        let expert = &m.experts[e];
        // gate_up = [2*moe_inter] (gate ‖ up), then split into halves.
        weight_gemv(gpu, &expert.gate_up, &state.normed, &state.expert_gate_up)
            .map_err(|e2| format!("cohere2moe L{l}E{e}: gate_up gemv: {e2}"))?;
        let gate_view = state.expert_gate_up.sub_offset(0, moe_inter);
        let up_view = state.expert_gate_up.sub_offset(moe_inter, moe_inter);
        gpu.silu_mul_f32(&gate_view, &up_view, &state.expert_act)
            .map_err(|e2| format!("cohere2moe L{l}E{e}: silu_mul: {e2:?}"))?;
        // Fold the router weight into the activation (down is linear:
        // w·down(act) = down(w·act)), then accumulate down(·) into h.
        gpu.scale_f32(&state.expert_act, w)
            .map_err(|e2| format!("cohere2moe L{l}E{e}: scale: {e2:?}"))?;
        weight_gemv_residual(gpu, &expert.down, &state.expert_act, &state.h)
            .map_err(|e2| format!("cohere2moe L{l}E{e}: down gemv: {e2}"))?;
    }
    Ok(())
}

/// True iff `forward_batch` supports this model: Q8 attention/dense/router and
/// indexed (MQ4/MQ6/HFQ4/HFQ6) experts. The bf16 oracle and Q8-expert tiers
/// fall back to per-token `decode_step` (no indexed-MoE / batched-bf16 path).
pub fn forward_batch_supported(weights: &Cohere2MoeWeights) -> bool {
    weights.layers.iter().all(|l| {
        l.wq.gpu_dtype == DType::Q8_0
            && match &l.ffn {
                Ffn::Dense(_) => true,
                Ffn::Moe(m) => matches!(
                    m.experts[0].gate_up.gpu_dtype,
                    DType::MQ4G256 | DType::HFQ4G256 | DType::MQ6G256 | DType::HFQ6G256
                ),
            }
    })
}

/// Batched prefill over `B` (≤64) tokens — the parallel-block forward run ONCE
/// per weight matrix for all B tokens (bandwidth-amortized) instead of B
/// per-token `decode_step`s. Fills the KV cache for positions
/// `[start_pos, start_pos+B)` and returns the LAST token's logits. Supports the
/// MQ4/MQ6 serving tiers (Q8 attention/dense + indexed batched MoE); see
/// `forward_batch_supported` (caller falls back to per-token otherwise).
#[allow(clippy::too_many_arguments)]
/// Env-gated long-context localization probe. `HIPFIRE_C2M_NORMDUMP=<step>`
/// (default 4096 when set non-numeric) logs the LAST-token hidden-state L2 /
/// max-abs / NaN per layer on chunks that cross a `step` boundary, so a
/// long-context collapse can be pinned to the exact layer + op (post-attn vs
/// post-ffn) where the residual blows up or flattens. Off by default.
fn c2m_normdump_step() -> Option<usize> {
    hipfire_config::developer_var("HIPFIRE_C2M_NORMDUMP")
        .ok()
        .map(|s| s.trim().parse().unwrap_or(4096))
}
fn dbg_row_stats(
    gpu: &mut Gpu,
    t: &GpuTensor,
    b: usize,
    hidden: usize,
    l: usize,
    ctx: usize,
    tag: &str,
) {
    if let Ok(v) = gpu.download_f32(t) {
        let off = (b - 1) * hidden;
        if off + hidden > v.len() {
            return;
        }
        let row = &v[off..off + hidden];
        let mut l2 = 0.0f64;
        let mut maxa = 0.0f32;
        let mut nan = false;
        for &x in row {
            if !x.is_finite() {
                nan = true;
            }
            l2 += (x as f64) * (x as f64);
            if x.abs() > maxa {
                maxa = x.abs();
            }
        }
        eprintln!(
            "[normdump ctx={ctx} L{l:02} {tag}] l2={:.4} maxabs={:.5} nan={nan}",
            l2.sqrt(),
            maxa
        );
    }
}

pub fn forward_batch(
    cfg: &Cohere2MoeConfig,
    weights: &Cohere2MoeWeights,
    state: &mut Cohere2MoeState,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: usize,
) -> Result<Vec<f32>, String> {
    let b = tokens.len();
    if b == 0 {
        return Err("cohere2moe forward_batch: empty token slice".to_string());
    }
    // The Q8 projections go through `gemm_q8_0_batched_chunked`, which handles
    // arbitrary batch (sub-batches the MAX_BATCH=64 scalar kernel, and upgrades
    // to WMMA on RDNA4). The remaining batched ops (attention, layernorm, topk,
    // rotate, MoE scatter/grouped) carry no 64-cap. Larger batch is what makes
    // the grouped-WMMA MoE path pay off: tokens-per-expert = b·k_top/n_exp grows
    // (≈4 at b=64 → ≈16 at b=256 with n_exp=128), shrinking the BLOCK_M=16
    // padding waste. 512 is a generous scratch-memory ceiling.
    if b > 512 {
        return Err(format!(
            "cohere2moe forward_batch: B={b} exceeds scratch cap 512"
        ));
    }
    if !forward_batch_supported(weights) {
        return Err(
            "cohere2moe forward_batch: unsupported tier (needs Q8 attn + indexed experts)"
                .to_string(),
        );
    }
    let hidden = cfg.hidden_size;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let moe_inter = cfg.moe_intermediate_size;
    let dense_inter = cfg.dense_intermediate_size;
    let n_exp = cfg.num_experts;
    let k_top = cfg.num_experts_per_tok;
    let eps = cfg.rms_norm_eps;
    let max_ctx = start_pos + b;
    let max_seq = state.kv.physical_cap;

    let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
        g.alloc_tensor(&[n], DType::F32)
            .map_err(|e| format!("forward_batch alloc {label}: {e:?}"))
    };
    let x = alloc(gpu, b * hidden, "x")?;
    let normed = alloc(gpu, b * hidden, "normed")?;
    let fq = alloc(gpu, b * q_dim, "fq")?;
    let fk = alloc(gpu, b * kv_dim, "fk")?;
    let fv = alloc(gpu, b * kv_dim, "fv")?;
    let attn_out = alloc(gpu, b * q_dim, "attn_out")?;
    let o = alloc(gpu, b * hidden, "o")?;
    let ffn_x_rot = alloc(gpu, b * hidden, "ffn_x_rot")?;
    let router_logits = alloc(gpu, b * n_exp, "router_logits")?;
    let topk_idx = alloc(gpu, b * k_top, "topk_idx")?;
    let topk_w = alloc(gpu, b * k_top, "topk_w")?;
    let gate = alloc(gpu, b * k_top * moe_inter, "gate")?;
    let up = alloc(gpu, b * k_top * moe_inter, "up")?;
    let rot = alloc(gpu, b * k_top * moe_inter, "rot")?;
    let down_exp = alloc(gpu, b * k_top * hidden, "down_exp")?;
    let dense_gate = alloc(gpu, b * dense_inter, "dense_gate")?;
    let dense_up = alloc(gpu, b * dense_inter, "dense_up")?;
    let dense_act = alloc(gpu, b * dense_inter, "dense_act")?;
    // Grouped-MoE prefill scratch — scatter-by-expert indices (i32, held in
    // DType::Raw byte tensors at 4 bytes/elem) + grouped-WMMA output buffers.
    // Path 1 (indexed) ignores these; Path 2 (grouped) uses them. Always
    // allocated since MoePrefillParams takes every buffer by reference.
    let m_total_max = moe_grouped_m_total_bound(b * k_top, n_exp);
    let raw = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
        g.alloc_tensor(&[n], DType::Raw)
            .map_err(|e| format!("forward_batch alloc {label}: {e:?}"))
    };
    let expert_token_counts = raw(gpu, n_exp * 4, "etc")?;
    let expert_offsets = raw(gpu, (n_exp + 1) * 4, "eoff")?;
    let sorted_slot_index = raw(gpu, m_total_max * 4, "ssi")?;
    let inverse_perm = raw(gpu, b * k_top * 4, "iperm")?;
    let expert_tile_ids = raw(gpu, (m_total_max / MOE_GROUPED_BLOCK_M) * 4, "etid")?;
    let y_gate_up_grouped = alloc(gpu, m_total_max * 2 * moe_inter, "ygu")?;
    let y_down_grouped = alloc(gpu, m_total_max * hidden, "ydn")?;
    // F16 activation scratch for the WMMA Q8 projections (converted fresh per
    // q8_proj_raw call). Sized for the widest projection input K.
    let kmax = hidden.max(q_dim).max(dense_inter);
    let x_f16 = gpu
        .alloc_tensor(&[b * kmax], DType::F16)
        .map_err(|e| format!("forward_batch alloc x_f16: {e:?}"))?;

    // positions [B] i32 (stored in an f32-sized buffer; kernels read it as i32).
    let pos_bytes: Vec<u8> = (0..b)
        .flat_map(|i| ((start_pos + i) as i32).to_ne_bytes())
        .collect();
    let pos_array = alloc(gpu, b, "pos_array")?;
    gpu.hip
        .memcpy_htod(&pos_array.buf, &pos_bytes)
        .map_err(|e| format!("forward_batch htod pos: {e:?}"))?;

    // Embedding (Q8) per token → x[B, hidden].
    {
        let xs = alloc(gpu, hidden, "xs")?;
        for (i, &tok) in tokens.iter().enumerate() {
            embed_lookup(gpu, weights, hidden, tok, &xs)?;
            gpu.hip
                .memcpy_dtod_at(&x.buf, i * hidden * 4, &xs.buf, 0, hidden * 4)
                .map_err(|e| format!("forward_batch embed copy: {e:?}"))?;
        }
        gpu.free_tensor(xs).ok();
    }

    let normdump = c2m_normdump_step().map_or(false, |step| {
        step > 0 && (start_pos / step != max_ctx / step)
    });
    for (l, layer) in weights.layers.iter().enumerate() {
        // Parallel block: normed = RMSNorm(x), fed to BOTH branches.
        gpu.rmsnorm_batched(&x, &layer.input_norm, &normed, b, hidden, eps)
            .map_err(|e| format!("cohere2moe L{l} batch ln: {e:?}"))?;
        // Attention from `normed` (Q8 projections).
        q8_proj_raw(gpu, &layer.wq.buf, &normed, &fq, q_dim, hidden, b, &x_f16)
            .map_err(|e| format!("cohere2moe L{l} batch q: {e:?}"))?;
        q8_proj_raw(gpu, &layer.wk.buf, &normed, &fk, kv_dim, hidden, b, &x_f16)
            .map_err(|e| format!("cohere2moe L{l} batch k: {e:?}"))?;
        q8_proj_raw(gpu, &layer.wv.buf, &normed, &fv, kv_dim, hidden, b, &x_f16)
            .map_err(|e| format!("cohere2moe L{l} batch v: {e:?}"))?;
        // NoPE on full layers; interleaved RoPE on sliding layers.
        // RoPE on sliding layers AND the dense prefix layers (force_rope) — see
        // the decode path. North layer 0 (dense, full_attention) is rotated.
        if layer.attn_kind == AttnKind::Sliding
            || (l < cfg.first_k_dense_replace && cfg.prefix_dense_sliding_window_pattern == 1)
        {
            gpu.rope_interleaved_f32_batched(
                &fq,
                &fk,
                &pos_array,
                n_heads,
                n_kv,
                head_dim,
                head_dim,
                cfg.rope_theta,
                b,
            )
            .map_err(|e| format!("cohere2moe L{l} batch rope: {e:?}"))?;
        }
        // Batched KV write (Q8) + sliding-window flash attention via the shared
        // KV-usage abstraction (causal: tree_bias=None). q8_windowed selects
        // AttnQ8_0KvBatchedMaskedWindowed unconditionally (window>0 clips
        // `sliding_attention` layers to the last `sliding_window` keys; window=0
        // is full causal for `full_attention`/NoPE layers) — NOT the LDS-bound
        // non-windowed batched key, which would drop the window.
        let window = if layer.attn_kind == AttnKind::Sliding {
            cfg.sliding_window as i32
        } else {
            0
        };
        let ctx = DispatchCtx::new(gpu);
        let plan = hipfire_dispatch::families::kv_tier::KvTierPlan::derive(
            hipfire_dispatch::families::kv_tier::KvTierInputs {
                q8_windowed: true,
                window,
                batch_size: b,
                ..state.kv.tier_inputs()
            },
        )
        .map_err(|e| format!("cohere2moe L{l} batch kv tier: {e}"))?;
        let io = hipfire_dispatch::families::attention::AttnParams {
            q: &fq,
            k: &fk,
            v: &fv,
            k_cache: &state.kv.k_gpu[l],
            v_cache: &state.kv.v_gpu[l],
            k_scales: None,
            v_scales: None,
            pos_buf: &state.pos_buf,
            pos: 0,
            positions: Some(&pos_array),
            n_heads,
            n_kv_heads: n_kv,
            head_dim,
            physical_cap: max_seq,
            batch_size: b,
            max_ctx_len: max_ctx,
            flash_partials: Some(&state.flash_partials),
            givens_cos: None,
            givens_sin: None,
            tree_bias: None,
            block_start: 0,
            block_cols: 0,
            output_gate: None,
            output: &attn_out,
        };
        hipfire_dispatch::pipeline::execute_steps(
            gpu,
            &ctx,
            &[hipfire_dispatch::pipeline::Step::Attend { plan, io }],
        )
        .map_err(|e| format!("cohere2moe L{l} batch attn: {e:?}"))?;
        q8_proj_raw(gpu, &layer.wo.buf, &attn_out, &o, hidden, q_dim, b, &x_f16)
            .map_err(|e| format!("cohere2moe L{l} batch o: {e:?}"))?;
        gpu.add_inplace_f32(&x, &o)
            .map_err(|e| format!("cohere2moe L{l} batch o-resid: {e:?}"))?;
        if normdump {
            dbg_row_stats(gpu, &x, b, hidden, l, max_ctx, "post_attn");
        }

        // FFN from the SAME `normed` (parallel block).
        match &layer.ffn {
            Ffn::Dense(d) => {
                q8_proj_raw(
                    gpu,
                    &d.gate.buf,
                    &normed,
                    &dense_gate,
                    dense_inter,
                    hidden,
                    b,
                    &x_f16,
                )
                .map_err(|e| format!("cohere2moe L{l} batch dgate: {e:?}"))?;
                q8_proj_raw(
                    gpu,
                    &d.up.buf,
                    &normed,
                    &dense_up,
                    dense_inter,
                    hidden,
                    b,
                    &x_f16,
                )
                .map_err(|e| format!("cohere2moe L{l} batch dup: {e:?}"))?;
                gpu.silu_mul_f32(&dense_gate, &dense_up, &dense_act)
                    .map_err(|e| format!("cohere2moe L{l} batch dsilu: {e:?}"))?;
                q8_proj_raw(
                    gpu,
                    &d.down.buf,
                    &dense_act,
                    &o,
                    hidden,
                    dense_inter,
                    b,
                    &x_f16,
                )
                .map_err(|e| format!("cohere2moe L{l} batch ddown: {e:?}"))?;
                gpu.add_inplace_f32(&x, &o)
                    .map_err(|e| format!("cohere2moe L{l} batch ddown-resid: {e:?}"))?;
            }
            Ffn::Moe(m) => {
                q8_proj_raw(
                    gpu,
                    &m.router.buf,
                    &normed,
                    &router_logits,
                    n_exp,
                    hidden,
                    b,
                    &x_f16,
                )
                .map_err(|e| format!("cohere2moe L{l} batch router: {e:?}"))?;
                gpu.sigmoid_f32(&router_logits)
                    .map_err(|e| format!("cohere2moe L{l} batch sigmoid: {e:?}"))?;
                gpu.moe_topk_renorm_k8_batched(
                    &router_logits,
                    &topk_idx,
                    &topk_w,
                    n_exp,
                    cfg.norm_topk_prob,
                    b,
                )
                .map_err(|e| format!("cohere2moe L{l} batch topk: {e:?}"))?;
                // `ffn_x_rot` ← FWHT(normed): run_moe_prefill's MQ4/MQ6 path
                // requires the activations pre-rotated by the model (it rotates
                // only in the paro path). Dropping this was the bug behind the
                // earlier garbage output on both paths.
                rotate_x_mq_batched_for(gpu, &m.experts[0].gate_up, &normed, &ffn_x_rot, hidden, b)
                    .map_err(|e| format!("cohere2moe L{l} batch rot: {e}"))?;
                // Shared dispatch executor: Path 1 (indexed batched GEMV, the
                // default — identical to the hand-rolled loop this replaced) or
                // Path 2 (scatter-by-expert → grouped-WMMA GEMM, under
                // HIPFIRE_MOE_GROUPED_GEMM=1). Routed expert outputs accumulate
                // into `x`, which IS the parallel-block residual add. No shared
                // expert in Cohere2-MoE → shared_* dtypes are inert placeholders.
                let ctx = DispatchCtx::new(gpu);
                let edt = m.experts[0].gate_up.gpu_dtype;
                let params = MoePrefillParams {
                    dtypes: MoeDtypes {
                        router: DType::Q8_0,
                        shared_gate: DType::Q8_0,
                        shared_expert_gate: DType::Q8_0,
                        shared_expert_up: DType::Q8_0,
                        shared_expert_down: DType::Q8_0,
                        experts_all_gate_up_mq4: edt == DType::MQ4G256,
                        routed_gate_up: edt,
                        routed_down: m.experts[0].down.gpu_dtype,
                        routed_has_mixed_experts: false,
                        per_expert_gate_up: None,
                        per_expert_down: None,
                        routed_escha_transforms: false,
                        has_paro_shared: false,
                    },
                    batch_size: b,
                    mi: moe_inter,
                    down_m: hidden,
                    down_k: moe_inter,
                    gate_up_k: hidden,
                    k_top,
                    n_exp,
                    m_total_max,
                    force_mq4_grouped_fp16: false,
                    topk_indices: &topk_idx,
                    topk_weights: &topk_w,
                    x_batch: &x,
                    x_norm_batch: &normed,
                    x_rot_batch: &ffn_x_rot,
                    expert_gate_up_ptrs: &m.expert_gate_up_ptrs,
                    expert_down_ptrs: &m.expert_down_ptrs,
                    expert_down_awq_ptrs: None,
                    expert_dtype_tags: None,
                    gate_batch: &gate,
                    up_batch: &up,
                    rot_batch: &rot,
                    down_expanded: &down_exp,
                    expert_token_counts: &expert_token_counts,
                    expert_offsets: &expert_offsets,
                    sorted_slot_index: &sorted_slot_index,
                    expert_tile_ids: &expert_tile_ids,
                    inverse_perm: &inverse_perm,
                    y_gate_up_grouped: &y_gate_up_grouped,
                    y_down_grouped: &y_down_grouped,
                    paro_gate_up: None,
                    paro_down: None,
                    down_awq_scale: None,
                    routed_out: None,
                    // Not an escha model: the escha branch in
                    // `run_moe_prefill` is skipped and Path 1 / Path 2 run
                    // exactly as before.
                    escha: None,
                    hidden,
                };
                moe_family()
                    .run_prefill(&ctx, gpu, &params)
                    .map_err(|e| format!("cohere2moe L{l} run_prefill: {e:?}"))?;
            }
        }
        if normdump {
            dbg_row_stats(gpu, &x, b, hidden, l, max_ctx, "post_ffn");
        }
    }
    state.n_tokens = start_pos + b;

    // Final RMSNorm + lm_head on the LAST row only (prefill needs the
    // last position's logits to seed decode).
    let x_last = alloc(gpu, hidden, "x_last")?;
    gpu.hip
        .memcpy_dtod_at(&x_last.buf, 0, &x.buf, (b - 1) * hidden * 4, hidden * 4)
        .map_err(|e| format!("forward_batch last copy: {e:?}"))?;
    gpu.rmsnorm_batched(
        &x_last,
        &weights.final_norm,
        &state.final_norm_buf,
        1,
        hidden,
        eps,
    )
    .map_err(|e| format!("forward_batch final ln: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
        .map_err(|e| format!("forward_batch lm_head: {e}"))?;
    let logits = gpu
        .download_f32(&state.logits)
        .map_err(|e| format!("forward_batch download logits: {e:?}"))?;

    for t in [
        x,
        normed,
        fq,
        fk,
        fv,
        attn_out,
        o,
        ffn_x_rot,
        router_logits,
        topk_idx,
        topk_w,
        gate,
        up,
        rot,
        down_exp,
        dense_gate,
        dense_up,
        dense_act,
        pos_array,
        x_last,
        expert_token_counts,
        expert_offsets,
        sorted_slot_index,
        inverse_perm,
        expert_tile_ids,
        y_gate_up_grouped,
        y_down_grouped,
        x_f16,
    ] {
        gpu.free_tensor(t).ok();
    }
    Ok(logits)
}
