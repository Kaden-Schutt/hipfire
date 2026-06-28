// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! ZAYA1 CPU reference forward (f32), a faithful port of the upstream
//! `modeling_zaya.py` (vendored `third_party/transformers` @ `zaya1`). This is the
//! correctness oracle the GPU path is validated against; it captures per-block
//! hidden states so they can be compared to `golden/zaya_golden.npz` (≥0.999
//! cosine). Batch is implicitly 1; activations are `[seq, dim]` row-major.
//!
//! Pipeline per hybrid block (`ZayaDecoderLayer`):
//!   residual = h
//!   h = input_layernorm(h)                          # RMSNorm
//!   h = cca_attention(h)                            # CCA mixer
//!   residual = post_attention_residual_scale(h, residual)
//!   h = post_attention_layernorm(residual)
//!   (h, router_state) = moe(h, router_state)        # EDA/MoD-routed MoE
//!   h = post_mlp_residual_scale(h, residual)

use crate::weights::{ZayaCcaWeights, ZayaLayerWeights, ZayaResidualScale, ZayaWeights};
use crate::ZayaConfig;

/// Captured activations for golden comparison.
pub struct ForwardTrace {
    /// Input to block 0 (post global input residual scale). `[seq, hidden]`.
    pub embed_scaled: Vec<f32>,
    /// Output hidden states of each hybrid block. `num_blocks × [seq, hidden]`.
    pub block: Vec<Vec<f32>>,
    /// Top-1 expert id per position per block (`num_experts` == MoD skip).
    pub router_idx: Vec<Vec<usize>>,
    /// Hidden states after the final RMSNorm. `[seq, hidden]`.
    pub final_norm: Vec<f32>,
    /// lm_head logits. `[seq, vocab]`.
    pub logits: Vec<f32>,
    pub seq: usize,
}

/// Run the full CPU forward for `input_ids`, capturing per-block activations.
pub fn forward_cpu(w: &ZayaWeights, cfg: &ZayaConfig, input_ids: &[u32]) -> ForwardTrace {
    let s = input_ids.len();
    let h = cfg.hidden_size;

    // Embedding lookup → global input residual affine (kept fp32).
    let mut hidden = vec![0f32; s * h];
    for (t, &id) in input_ids.iter().enumerate() {
        let row = &w.embed_tokens[id as usize * h..(id as usize + 1) * h];
        for i in 0..h {
            hidden[t * h + i] =
                (row[i] + w.input_hidden_states_bias[i]) * w.input_hidden_states_scale[i];
        }
    }
    let embed_scaled = hidden.clone();

    // Partial-rotary tables (default rope, first n_rot dims).
    let rope = RopeTables::new(s, cfg.attn.head_dim, cfg.attn.n_rot, cfg.attn.rope_theta);

    let mut block = Vec::with_capacity(cfg.num_blocks);
    let mut router_idx = Vec::with_capacity(cfg.num_blocks);
    // EDA cross-layer router state (router_hidden after down_proj), [seq, router_hidden].
    let mut router_state: Option<Vec<f32>> = None;

    for (l, lw) in w.layers.iter().enumerate() {
        let residual = hidden.clone();
        let normed = rmsnorm(&hidden, &lw.input_layernorm, s, h, cfg.rms_norm_eps);
        let attn_out = cca_attention(&normed, lw, cfg, &rope, s);
        // residual = post_attention_residual_scale(attn_out, residual)
        let residual = residual_scale(
            &attn_out,
            &residual,
            &lw.post_attention_residual_scale,
            s,
            h,
        );
        let normed = rmsnorm(
            &residual,
            &lw.post_attention_layernorm,
            s,
            h,
            cfg.rms_norm_eps,
        );
        let (moe_out, idx, next_state) = moe(&normed, lw, cfg, l, router_state.as_deref(), s);
        router_state = Some(next_state);
        // hidden = post_mlp_residual_scale(moe_out, residual)
        hidden = residual_scale(&moe_out, &residual, &lw.post_mlp_residual_scale, s, h);
        block.push(hidden.clone());
        router_idx.push(idx);
    }

    let final_norm = rmsnorm(&hidden, &w.norm, s, h, cfg.rms_norm_eps);
    // Tied lm_head: logits = final_norm @ embed_tokens^T.
    let logits = linear(&final_norm, s, &w.lm_head, cfg.vocab_size, h, None);

    ForwardTrace {
        embed_scaled,
        block,
        router_idx,
        final_norm,
        logits,
        seq: s,
    }
}

// ── primitives ───────────────────────────────────────────────────────────────

/// `y[t,o] = sum_i x[t,i] * w[o,i] (+ bias[o])`. weight is row-major `[out, in]`.
fn linear(
    x: &[f32],
    s: usize,
    w: &[f32],
    out: usize,
    in_: usize,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let mut y = vec![0f32; s * out];
    for t in 0..s {
        let xt = &x[t * in_..(t + 1) * in_];
        let yt = &mut y[t * out..(t + 1) * out];
        for o in 0..out {
            let wr = &w[o * in_..(o + 1) * in_];
            let mut acc = 0f32;
            for i in 0..in_ {
                acc += xt[i] * wr[i];
            }
            yt[o] = acc + bias.map_or(0.0, |b| b[o]);
        }
    }
    y
}

/// RMSNorm: `y = x / sqrt(mean(x^2)+eps) * weight`.
fn rmsnorm(x: &[f32], weight: &[f32], s: usize, d: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0f32; s * d];
    for t in 0..s {
        let xt = &x[t * d..(t + 1) * d];
        let mut ss = 0f32;
        for &v in xt {
            ss += v * v;
        }
        let inv = 1.0 / (ss / d as f32 + eps).sqrt();
        let yt = &mut y[t * d..(t + 1) * d];
        for i in 0..d {
            yt[i] = xt[i] * inv * weight[i];
        }
    }
    y
}

/// `(h+h_bias)*h_scale + (res+res_bias)*res_scale`, elementwise per feature.
fn residual_scale(
    hidden: &[f32],
    residual: &[f32],
    rs: &ZayaResidualScale,
    s: usize,
    d: usize,
) -> Vec<f32> {
    let mut y = vec![0f32; s * d];
    for t in 0..s {
        for i in 0..d {
            let hi = (hidden[t * d + i] + rs.hidden_states_bias[i]) * rs.hidden_states_scale[i];
            let ri = (residual[t * d + i] + rs.residual_bias[i]) * rs.residual_scale[i];
            y[t * d + i] = hi + ri;
        }
    }
    y
}

/// Exact GELU (erf form), matching `nn.GELU()`.
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// erf via Abramowitz & Stegun 7.1.26 (|err| < 1.5e-7).
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

// ── partial rotary ───────────────────────────────────────────────────────────

struct RopeTables {
    /// `[seq * n_rot/2]` cos and sin of `pos * inv_freq`.
    cos: Vec<f32>,
    sin: Vec<f32>,
    n_rot: usize,
    half: usize,
}

impl RopeTables {
    fn new(seq: usize, _head_dim: usize, n_rot: usize, theta: f32) -> Self {
        let half = n_rot / 2;
        let mut cos = vec![0f32; seq * half];
        let mut sin = vec![0f32; seq * half];
        for p in 0..seq {
            for i in 0..half {
                let inv_freq = (theta as f64).powf(-(2.0 * i as f64) / n_rot as f64);
                let ang = p as f64 * inv_freq;
                cos[p * half + i] = ang.cos() as f32;
                sin[p * half + i] = ang.sin() as f32;
            }
        }
        Self {
            cos,
            sin,
            n_rot,
            half,
        }
    }

    /// Apply half-split partial rotary in-place to one head vector `v[head_dim]`
    /// at position `p` (only the first `n_rot` dims rotate).
    fn apply(&self, v: &mut [f32], p: usize) {
        let half = self.half;
        let c = &self.cos[p * half..(p + 1) * half];
        let sn = &self.sin[p * half..(p + 1) * half];
        for d in 0..half {
            let a = v[d];
            let b = v[d + half];
            v[d] = a * c[d] - b * sn[d];
            v[d + half] = b * c[d] + a * sn[d];
        }
        let _ = self.n_rot;
    }
}

// ── CCA attention ────────────────────────────────────────────────────────────

#[allow(clippy::too_many_lines)]
fn cca_attention(
    x: &[f32],
    lw: &ZayaLayerWeights,
    cfg: &ZayaConfig,
    rope: &RopeTables,
    s: usize,
) -> Vec<f32> {
    let a = &cfg.attn;
    let hd = a.head_dim;
    let nq = a.num_heads;
    let nkv = a.num_kv_heads;
    let groups = nq / nkv;
    let h = cfg.hidden_size;
    let q_dim = nq * hd; // 1024
    let k_dim = nkv * hd; // 256
    let v_half = k_dim / 2; // 128
    let conv_ch = q_dim + k_dim; // 1280
    let cca: &ZayaCcaWeights = &lw.cca;

    // Projections.
    let q = linear(x, s, &cca.q_proj, q_dim, h, None); // [s, 1024]
    let k = linear(x, s, &cca.k_proj, k_dim, h, None); // [s, 256]
    let v_cur = linear(x, s, &cca.v_proj_current, v_half, h, None); // [s, 128]
    let v_del = linear(x, s, &cca.v_proj_delayed, v_half, h, None); // [s, 128]

    // q/k residual paths (pre-conv), in [s, head, hd] layout.
    // query_residual[t,head] = (q[t,head] + key_repeated[t,head]) * 0.5
    // key_repeated: kv head kh broadcast to q heads kh*groups..+groups.
    let mut query_residual = vec![0f32; s * nq * hd];
    for t in 0..s {
        for head in 0..nq {
            let kh = head / groups;
            for d in 0..hd {
                let qv = q[t * q_dim + head * hd + d];
                let kv = k[t * k_dim + kh * hd + d];
                query_residual[(t * nq + head) * hd + d] = (qv + kv) * 0.5;
            }
        }
    }
    // key_residual[t,kh] = mean over the `groups` q-heads of query_residual.
    let mut key_residual = vec![0f32; s * nkv * hd];
    for t in 0..s {
        for kh in 0..nkv {
            for d in 0..hd {
                let mut acc = 0f32;
                for g in 0..groups {
                    acc += query_residual[(t * nq + kh * groups + g) * hd + d];
                }
                key_residual[(t * nkv + kh) * hd + d] = acc / groups as f32;
            }
        }
    }

    // Concatenated q||k stream in channel-major [conv_ch, s], left-padded by the
    // total conv kernel reach, then two causal convs.
    let pad = (a.conv_depthwise_kernel - 1) + (a.conv_grouped_kernel - 1);
    let padded_len = s + pad;
    let mut stream = vec![0f32; conv_ch * padded_len];
    for t in 0..s {
        for c in 0..q_dim {
            stream[c * padded_len + pad + t] = q[t * q_dim + c];
        }
        for c in 0..k_dim {
            stream[(q_dim + c) * padded_len + pad + t] = k[t * k_dim + c];
        }
    }
    // Depthwise conv (groups == channels): out[c,t] = b[c] + sum_k w[c,k]*in[c,t+k].
    let kd = a.conv_depthwise_kernel;
    let dw_len = padded_len - kd + 1;
    let mut dw = vec![0f32; conv_ch * dw_len];
    for c in 0..conv_ch {
        for t in 0..dw_len {
            let mut acc = cca.conv_qk_depthwise_b[c];
            for kk in 0..kd {
                acc += cca.conv_qk_depthwise_w[c * kd + kk] * stream[c * padded_len + t + kk];
            }
            dw[c * dw_len + t] = acc;
        }
    }
    // Grouped conv: groups = nq + nkv (= 10), in/group = conv_ch/groups (= 128).
    let kg = a.conv_grouped_kernel;
    let n_groups = nq + nkv;
    let in_per_group = conv_ch / n_groups;
    let gw_len = dw_len - kg + 1; // == s
    debug_assert_eq!(gw_len, s);
    let mut gw = vec![0f32; conv_ch * s];
    for c in 0..conv_ch {
        let group = c / in_per_group;
        let base = group * in_per_group;
        for t in 0..s {
            let mut acc = cca.conv_qk_grouped_b[c];
            for j in 0..in_per_group {
                for kk in 0..kg {
                    let wv = cca.conv_qk_grouped_w[(c * in_per_group + j) * kg + kk];
                    acc += wv * dw[(base + j) * dw_len + t + kk];
                }
            }
            gw[c * s + t] = acc;
        }
    }

    // Split conv output back to [s, head, hd] and add residuals.
    // query = gw[:q_dim] + query_residual ; key = gw[q_dim:] + key_residual.
    let mut query = vec![0f32; s * nq * hd];
    for t in 0..s {
        for head in 0..nq {
            for d in 0..hd {
                let c = head * hd + d;
                query[(t * nq + head) * hd + d] =
                    gw[c * s + t] + query_residual[(t * nq + head) * hd + d];
            }
        }
    }
    let mut key = vec![0f32; s * nkv * hd];
    for t in 0..s {
        for kh in 0..nkv {
            for d in 0..hd {
                let c = q_dim + kh * hd + d;
                key[(t * nkv + kh) * hd + d] =
                    gw[c * s + t] + key_residual[(t * nkv + kh) * hd + d];
            }
        }
    }

    // Value: head 0 = current-token v, head 1 = previous-token delayed v.
    // value[t] = concat(v_cur[t], v_delayed[t]) viewed as [nkv, hd] where
    // v_delayed[t] = v_del[t-1] (0 at t==0).
    let mut value = vec![0f32; s * nkv * hd];
    for t in 0..s {
        for d in 0..v_half {
            value[(t * nkv + 0) * hd + d] = v_cur[t * v_half + d];
            value[(t * nkv + 1) * hd + d] = if t == 0 {
                0.0
            } else {
                v_del[(t - 1) * v_half + d]
            };
        }
    }

    // QK-norm: L2-normalize each head to sqrt(head_dim); key scaled by per-kv temp.
    let scale = (hd as f32).sqrt();
    let eps = f32::EPSILON;
    qk_l2_norm(&mut query, s, nq, hd, scale, eps);
    qk_l2_norm(&mut key, s, nkv, hd, scale, eps);
    for t in 0..s {
        for kh in 0..nkv {
            let temp = cca.qk_norm_temp[kh];
            for d in 0..hd {
                key[(t * nkv + kh) * hd + d] *= temp;
            }
        }
    }

    // Partial rotary on q and k.
    for t in 0..s {
        for head in 0..nq {
            rope.apply(
                &mut query[(t * nq + head) * hd..(t * nq + head + 1) * hd],
                t,
            );
        }
        for kh in 0..nkv {
            rope.apply(&mut key[(t * nkv + kh) * hd..(t * nkv + kh + 1) * hd], t);
        }
    }

    // GQA causal attention. scaling = head_dim^-0.5.
    let attn_scale = 1.0 / (hd as f32).sqrt();
    let mut ctx = vec![0f32; s * nq * hd]; // [s, nq, hd]
    for head in 0..nq {
        let kh = head / groups;
        for i in 0..s {
            let qv = &query[(i * nq + head) * hd..(i * nq + head + 1) * hd];
            // scores over j <= i
            let mut scores = vec![0f32; i + 1];
            let mut maxv = f32::NEG_INFINITY;
            for j in 0..=i {
                let kv = &key[(j * nkv + kh) * hd..(j * nkv + kh + 1) * hd];
                let mut dot = 0f32;
                for d in 0..hd {
                    dot += qv[d] * kv[d];
                }
                let sc = dot * attn_scale;
                scores[j] = sc;
                if sc > maxv {
                    maxv = sc;
                }
            }
            let mut denom = 0f32;
            for sc in scores.iter_mut() {
                *sc = (*sc - maxv).exp();
                denom += *sc;
            }
            let out = &mut ctx[(i * nq + head) * hd..(i * nq + head + 1) * hd];
            for j in 0..=i {
                let p = scores[j] / denom;
                let vv = &value[(j * nkv + kh) * hd..(j * nkv + kh + 1) * hd];
                for d in 0..hd {
                    out[d] += p * vv[d];
                }
            }
        }
    }

    // o_proj over the flattened [s, nq*hd] context.
    linear(&ctx, s, &cca.o_proj, h, q_dim, None)
}

/// L2-normalize each `[head, hd]` row to `scale` (= sqrt(head_dim)).
fn qk_l2_norm(x: &mut [f32], s: usize, heads: usize, hd: usize, scale: f32, eps: f32) {
    for t in 0..s {
        for head in 0..heads {
            let row = &mut x[(t * heads + head) * hd..(t * heads + head + 1) * hd];
            let mut norm = 0f32;
            for &v in row.iter() {
                norm += v * v;
            }
            let inv = scale / norm.sqrt().max(eps);
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }
}

// ── EDA/MoD-routed MoE ───────────────────────────────────────────────────────

/// Returns (moe_output `[s,hidden]`, top-1 expert id per token, next router state).
fn moe(
    x: &[f32],
    lw: &ZayaLayerWeights,
    cfg: &ZayaConfig,
    layer_idx: usize,
    router_state: Option<&[f32]>,
    s: usize,
) -> (Vec<f32>, Vec<usize>, Vec<f32>) {
    let h = cfg.hidden_size;
    let rh = cfg.moe.router_hidden_size;
    let n_exp = cfg.moe.num_experts;
    let n_route = cfg.moe.num_router_experts(); // n_exp + 1
    let r = &lw.router;

    // down_proj → optional EDA cross-layer state add → next state.
    let mut router_hidden = linear(x, s, &r.down_proj_w, rh, h, Some(&r.down_proj_b));
    if layer_idx != 0 {
        if let (Some(scale), Some(prev)) = (r.router_states_scale.as_ref(), router_state) {
            for t in 0..s {
                for i in 0..rh {
                    router_hidden[t * rh + i] += prev[t * rh + i] * scale[i];
                }
            }
        }
    }
    let next_state = router_hidden.clone();

    // router_mlp: RMSNorm → fc1 → gelu → fc2 → gelu → out_proj (→ n_route logits).
    let normed = rmsnorm(&router_hidden, &r.norm_w, s, rh, cfg.rms_norm_eps);
    let mut a1 = linear(&normed, s, &r.fc1_w, rh, rh, Some(&r.fc1_b));
    for v in a1.iter_mut() {
        *v = gelu(*v);
    }
    let mut a2 = linear(&a1, s, &r.fc2_w, rh, rh, Some(&r.fc2_b));
    for v in a2.iter_mut() {
        *v = gelu(*v);
    }
    let logits = linear(&a2, s, &r.out_proj_w, n_route, rh, None); // [s, n_route]

    let mut out = vec![0f32; s * h];
    let mut idx_out = vec![0usize; s];
    let moe_int = cfg.moe.moe_intermediate_size;
    for t in 0..s {
        // softmax over router logits.
        let row = &logits[t * n_route..(t + 1) * n_route];
        let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs = vec![0f32; n_route];
        let mut denom = 0f32;
        for e in 0..n_route {
            probs[e] = (row[e] - maxv).exp();
            denom += probs[e];
        }
        for p in probs.iter_mut() {
            *p /= denom;
        }
        // top-1 over (probs + balancing_biases); last expert is the MoD skip.
        let mut best = 0usize;
        let mut bestv = f32::NEG_INFINITY;
        for e in 0..n_route {
            let v = probs[e] + r.balancing_biases[e];
            if v > bestv {
                bestv = v;
                best = e;
            }
        }
        idx_out[t] = best;
        if best == n_exp {
            // MoD skip: token bypasses the FFN (zero contribution).
            continue;
        }
        let weight = probs[best];
        // SwiGLU expert: gate_up = x@W_gu^T → [2*moe_int]; silu(gate)*up → down.
        let ew = &lw.experts[best];
        let xt = &x[t * h..(t + 1) * h];
        let mut gu = vec![0f32; 2 * moe_int];
        for o in 0..2 * moe_int {
            let wr = &ew.gate_up_proj[o * h..(o + 1) * h];
            let mut acc = 0f32;
            for i in 0..h {
                acc += xt[i] * wr[i];
            }
            gu[o] = acc;
        }
        let mut act = vec![0f32; moe_int];
        for i in 0..moe_int {
            let g = gu[i];
            let up = gu[moe_int + i];
            act[i] = (g / (1.0 + (-g).exp())) * up; // silu(gate)*up
        }
        let ot = &mut out[t * h..(t + 1) * h];
        for o in 0..h {
            let wr = &ew.down_proj[o * moe_int..(o + 1) * moe_int];
            let mut acc = 0f32;
            for i in 0..moe_int {
                acc += act[i] * wr[i];
            }
            ot[o] = acc * weight;
        }
    }

    (out, idx_out, next_state)
}
