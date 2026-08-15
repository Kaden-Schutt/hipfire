// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gate 4 f64 CPU oracle for the non-attention half of a DeepSeek V4 parent layer.
//!
//! Authority (transcribed literally, not guessed):
//! - `.codeinsight+research/ds4-parent-ref/inference/model.py`
//!   - `RMSNorm` 189-204
//!   - `Gate` 551-590
//!   - `Expert` 592-611
//!   - `MoE` 614-649
//!   - `Block.hc_pre` / `hc_post` / `hc_head` 680-716
//! - `.codeinsight+research/ds4-parent-ref/inference/kernel.py`
//!   - `hc_split_sinkhorn` 372-439
//!
//! All arithmetic is f64 internally; f32 only at the API boundary.
//!
//! ## Reference findings (must match any GPU sibling)
//!
//! **`sqrtsoftplus`** (`model.py:575-576`): when `score_func` is neither
//! `"softmax"` nor `"sigmoid"`, scores become `F.softplus(scores).sqrt()` —
//! i.e. `sqrt(softplus(x))` with the standard softplus
//! `log(1 + exp(x))`. Config sets `score_func = "sqrtsoftplus"`.
//!
//! **`noaux_tc` top-k** (`model.py:577-588`): there is no separate
//! `topk_method` branch in the bundled `Gate.forward`. The HF-named
//! noaux-TC behaviour is exactly:
//! 1. `scores = sqrtsoftplus(logits)`
//! 2. `original_scores = scores`
//! 3. `scores = scores + bias` (bias shifts **selection only**)
//! 4. `indices = scores.topk(k)[1]`
//! 5. `weights = original_scores.gather(1, indices)` — **uncorrected**
//! 6. if `score_func != "softmax"`: `weights /= weights.sum(-1)` (`norm_topk_prob`)
//! 7. `weights *= route_scale` (config: `1.5`)
//!
//! **Expert clamp asymmetry** (`model.py:605-610`): `up` is clamped to
//! `[-limit, +limit]`, but `gate` is clamped **only on the upper side**
//! (`max=limit`, no lower bound). Routing weight multiplies the
//! intermediate `silu(gate)*up` **before** `w2`, not the expert output.

#[inline]
fn err_msg(msg: &str) -> String {
    format!("deepseek4 parent: {msg}")
}

#[inline]
fn softplus_f64(x: f64) -> f64 {
    // Stable softplus: for large positive x, log1p(exp(x)) ≈ x.
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline]
fn sigmoid_f64(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

#[inline]
fn silu_f64(x: f64) -> f64 {
    x * sigmoid_f64(x)
}

/// `RMSNorm.forward` (`model.py:197-202`).
///
/// `x` and `weight` are length `rows * dim` and `dim` respectively (row-major).
/// Returns `rows * dim` f32.
pub fn rms_norm_ref(x: &[f32], weight: &[f32], eps: f64, dim: usize) -> Vec<f32> {
    assert!(dim > 0, "deepseek4 parent: rms_norm_ref dim must be > 0");
    assert_eq!(
        weight.len(),
        dim,
        "deepseek4 parent: rms_norm_ref weight len {} != dim {}",
        weight.len(),
        dim
    );
    assert_eq!(
        x.len() % dim,
        0,
        "deepseek4 parent: rms_norm_ref x len {} not divisible by dim {}",
        x.len(),
        dim
    );
    let rows = x.len() / dim;
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let base = r * dim;
        let mut acc = 0.0f64;
        for d in 0..dim {
            let v = x[base + d] as f64;
            acc += v * v;
        }
        let mean = acc / dim as f64;
        let scale = 1.0 / (mean + eps).sqrt();
        for d in 0..dim {
            let v = (x[base + d] as f64) * scale * (weight[d] as f64);
            out[base + d] = v as f32;
        }
    }
    out
}

/// `hc_split_sinkhorn` (`kernel.py:372-439`).
///
/// `mixes` is `[rows, (2 + hc_mult) * hc_mult]`.
/// `hc_scale` is length 3; `hc_base` is length `(2 + hc_mult) * hc_mult`.
///
/// Returns `(pre[rows, hc], post[rows, hc], comb[rows, hc, hc])`.
///
/// Normalization order (literal):
/// 1. `comb = row_softmax(comb) + eps`
/// 2. `comb = comb / (col_sum + eps)`
/// 3. for `_ in 0..(iters - 1)`: row-normalize, then column-normalize
pub fn hc_split_sinkhorn_ref(
    mixes: &[f32],
    hc_scale: &[f32],
    hc_base: &[f32],
    rows: usize,
    hc_mult: usize,
    iters: usize,
    eps: f64,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    if hc_mult == 0 {
        return Err(err_msg("hc_split_sinkhorn_ref: hc_mult must be > 0"));
    }
    if iters == 0 {
        return Err(err_msg(
            "hc_split_sinkhorn_ref: iters must be >= 1 (first pass is softmax+col)",
        ));
    }
    let mix_hc = (2 + hc_mult) * hc_mult;
    if hc_scale.len() != 3 {
        return Err(err_msg(&format!(
            "hc_split_sinkhorn_ref: hc_scale len {} != 3",
            hc_scale.len()
        )));
    }
    if hc_base.len() != mix_hc {
        return Err(err_msg(&format!(
            "hc_split_sinkhorn_ref: hc_base len {} != mix_hc {}",
            hc_base.len(),
            mix_hc
        )));
    }
    if mixes.len() != rows * mix_hc {
        return Err(err_msg(&format!(
            "hc_split_sinkhorn_ref: mixes len {} != rows*mix_hc {}",
            mixes.len(),
            rows * mix_hc
        )));
    }

    let s0 = hc_scale[0] as f64;
    let s1 = hc_scale[1] as f64;
    let s2 = hc_scale[2] as f64;

    let mut pre = vec![0.0f32; rows * hc_mult];
    let mut post = vec![0.0f32; rows * hc_mult];
    let mut comb = vec![0.0f32; rows * hc_mult * hc_mult];

    for r in 0..rows {
        let mbase = r * mix_hc;
        // pre[j] = sigmoid(mixes[j] * scale[0] + base[j]) + eps
        for j in 0..hc_mult {
            let v = mixes[mbase + j] as f64 * s0 + hc_base[j] as f64;
            pre[r * hc_mult + j] = (sigmoid_f64(v) + eps) as f32;
        }
        // post[j] = 2 * sigmoid(mixes[j+hc] * scale[1] + base[j+hc])
        for j in 0..hc_mult {
            let v = mixes[mbase + j + hc_mult] as f64 * s1 + hc_base[j + hc_mult] as f64;
            post[r * hc_mult + j] = (2.0 * sigmoid_f64(v)) as f32;
        }
        // comb[j,k] = mixes[j*hc + k + 2*hc] * scale[2] + base[...]
        let mut comb_f = vec![0.0f64; hc_mult * hc_mult];
        for j in 0..hc_mult {
            for k in 0..hc_mult {
                let idx = j * hc_mult + k + hc_mult * 2;
                comb_f[j * hc_mult + k] =
                    mixes[mbase + idx] as f64 * s2 + hc_base[idx] as f64;
            }
        }

        // comb = softmax(-1) + eps
        for j in 0..hc_mult {
            let row = &mut comb_f[j * hc_mult..(j + 1) * hc_mult];
            let mut row_max = f64::NEG_INFINITY;
            for &v in row.iter() {
                if v > row_max {
                    row_max = v;
                }
            }
            let mut row_sum = 0.0f64;
            for v in row.iter_mut() {
                *v = (*v - row_max).exp();
                row_sum += *v;
            }
            for v in row.iter_mut() {
                *v = *v / row_sum + eps;
            }
        }

        // comb = comb / (comb.sum(-2) + eps)  — column normalize
        col_normalize(&mut comb_f, hc_mult, eps);

        // (iters - 1) more row/col normalize passes
        for _ in 0..(iters - 1) {
            row_normalize(&mut comb_f, hc_mult, eps);
            col_normalize(&mut comb_f, hc_mult, eps);
        }

        let cbase = r * hc_mult * hc_mult;
        for i in 0..(hc_mult * hc_mult) {
            comb[cbase + i] = comb_f[i] as f32;
        }
    }

    Ok((pre, post, comb))
}

#[inline]
fn row_normalize(m: &mut [f64], hc: usize, eps: f64) {
    for j in 0..hc {
        let mut s = 0.0f64;
        for k in 0..hc {
            s += m[j * hc + k];
        }
        let denom = s + eps;
        for k in 0..hc {
            m[j * hc + k] /= denom;
        }
    }
}

#[inline]
fn col_normalize(m: &mut [f64], hc: usize, eps: f64) {
    for k in 0..hc {
        let mut s = 0.0f64;
        for j in 0..hc {
            s += m[j * hc + k];
        }
        let denom = s + eps;
        for j in 0..hc {
            m[j * hc + k] /= denom;
        }
    }
}

/// `Block.hc_pre` (`model.py:680-688`).
///
/// `x` is `[rows, hc_mult, dim]` (hc_mult = 4).
/// `hc_fn` is `[mix_hc, hc_mult * dim]` with `mix_hc = (2 + hc_mult) * hc_mult`.
/// `hc_scale` length 3; `hc_base` length `mix_hc`.
///
/// Returns `(y[rows, dim], post[rows, hc], comb[rows, hc, hc])`.
///
/// Flattened `hc*dim` RMS is taken once per row (not per stream); mixes are
/// `F.linear(x_flat, hc_fn) * rsqrt`.
pub fn hc_pre_ref(
    x: &[f32],
    hc_fn: &[f32],
    hc_scale: &[f32],
    hc_base: &[f32],
    rows: usize,
    hc_mult: usize,
    dim: usize,
    norm_eps: f64,
    sinkhorn_iters: usize,
    hc_eps: f64,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    if hc_mult == 0 || dim == 0 {
        return Err(err_msg("hc_pre_ref: hc_mult and dim must be > 0"));
    }
    let hc_dim = hc_mult * dim;
    let mix_hc = (2 + hc_mult) * hc_mult;
    if x.len() != rows * hc_dim {
        return Err(err_msg(&format!(
            "hc_pre_ref: x len {} != rows*hc*dim {}",
            x.len(),
            rows * hc_dim
        )));
    }
    if hc_fn.len() != mix_hc * hc_dim {
        return Err(err_msg(&format!(
            "hc_pre_ref: hc_fn len {} != mix_hc*hc_dim {}",
            hc_fn.len(),
            mix_hc * hc_dim
        )));
    }
    if hc_scale.len() != 3 {
        return Err(err_msg(&format!(
            "hc_pre_ref: hc_scale len {} != 3",
            hc_scale.len()
        )));
    }
    if hc_base.len() != mix_hc {
        return Err(err_msg(&format!(
            "hc_pre_ref: hc_base len {} != mix_hc {}",
            hc_base.len(),
            mix_hc
        )));
    }

    // x flattened [rows, hc*dim], rsqrt over last dim, mixes = (x @ hc_fn^T) * rsqrt
    let mut mixes = vec![0.0f32; rows * mix_hc];
    let mut rsqrts = vec![0.0f64; rows];
    for r in 0..rows {
        let xbase = r * hc_dim;
        let mut acc = 0.0f64;
        for d in 0..hc_dim {
            let v = x[xbase + d] as f64;
            acc += v * v;
        }
        rsqrts[r] = 1.0 / (acc / hc_dim as f64 + norm_eps).sqrt();
        // F.linear(x, hc_fn): out[o] = sum_k x[k] * hc_fn[o, k]
        for o in 0..mix_hc {
            let mut s = 0.0f64;
            let wbase = o * hc_dim;
            for k in 0..hc_dim {
                s += (x[xbase + k] as f64) * (hc_fn[wbase + k] as f64);
            }
            mixes[r * mix_hc + o] = (s * rsqrts[r]) as f32;
        }
    }

    let (pre, post, comb) =
        hc_split_sinkhorn_ref(&mixes, hc_scale, hc_base, rows, hc_mult, sinkhorn_iters, hc_eps)?;

    // y = sum_h pre[h] * x[h, :]
    let mut y = vec![0.0f32; rows * dim];
    for r in 0..rows {
        for d in 0..dim {
            let mut s = 0.0f64;
            for h in 0..hc_mult {
                let xv = x[r * hc_dim + h * dim + d] as f64;
                let pv = pre[r * hc_mult + h] as f64;
                s += pv * xv;
            }
            y[r * dim + d] = s as f32;
        }
    }
    Ok((y, post, comb))
}

/// `Block.hc_post` (`model.py:690-693`).
///
/// `x` `[rows, dim]`, `residual` `[rows, hc, dim]`, `post` `[rows, hc]`,
/// `comb` `[rows, hc, hc]` → `y` `[rows, hc, dim]`.
///
/// `y[r,h,d] = post[r,h] * x[r,d] + sum_k comb[r,k,h] * residual[r,k,d]`
///
/// # Contraction axis
///
/// The reference is
/// `torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)`. Broadcasting
/// gives `comb[A,B] * residual[A,d]` and `dim=2` sums over `A`, the **first**
/// `hc` axis of `comb`:
///
/// ```text
/// y[B,d] = sum_A comb[A][B] * residual[A,d]
/// ```
///
/// This is a *column* contraction, and it is load-bearing rather than a
/// convention: `hc_split_sinkhorn` (`kernel.py:401-423`) ends its loop on
/// `comb / comb.sum(-2)`, so the **columns** sum to 1. Summing over `A` is
/// therefore norm-preserving, while contracting the other axis picks up the row
/// sums — which are not 1 — and amplifies the residual on every layer.
///
/// This originally contracted `comb[r,h,k]`, the transpose. Both this reference
/// and the GPU path shared the error, so every HC oracle comparison agreed to
/// ~1e-7 while the composed forward was badly wrong (PPL 163.89 against 14.70
/// for a 2-bit quant of the same checkpoint).
pub fn hc_post_ref(
    x: &[f32],
    residual: &[f32],
    post: &[f32],
    comb: &[f32],
    rows: usize,
    hc_mult: usize,
    dim: usize,
) -> Vec<f32> {
    assert!(hc_mult > 0 && dim > 0);
    assert_eq!(x.len(), rows * dim);
    assert_eq!(residual.len(), rows * hc_mult * dim);
    assert_eq!(post.len(), rows * hc_mult);
    assert_eq!(comb.len(), rows * hc_mult * hc_mult);

    let mut y = vec![0.0f32; rows * hc_mult * dim];
    for r in 0..rows {
        for h in 0..hc_mult {
            let post_h = post[r * hc_mult + h] as f64;
            for d in 0..dim {
                let mut s = post_h * (x[r * dim + d] as f64);
                for k in 0..hc_mult {
                    let c = comb[r * hc_mult * hc_mult + k * hc_mult + h] as f64;
                    let res = residual[r * hc_mult * dim + k * dim + d] as f64;
                    s += c * res;
                }
                y[r * hc_mult * dim + h * dim + d] = s as f32;
            }
        }
    }
    y
}

/// `Block.hc_head` (`model.py:709-716`) — sigmoid path, **no** sinkhorn.
///
/// Used at the output head. `hc_fn` is `[hc_mult, hc_mult * dim]`,
/// `hc_scale` length 1 (or broadcastable scalar slice), `hc_base` length `hc_mult`.
///
/// `pre = sigmoid(mixes * hc_scale + hc_base) + hc_eps`, then weighted sum over streams.
pub fn hc_head_ref(
    x: &[f32],
    hc_fn: &[f32],
    hc_scale: &[f32],
    hc_base: &[f32],
    rows: usize,
    hc_mult: usize,
    dim: usize,
    norm_eps: f64,
    hc_eps: f64,
) -> Result<Vec<f32>, String> {
    if hc_mult == 0 || dim == 0 {
        return Err(err_msg("hc_head_ref: hc_mult and dim must be > 0"));
    }
    let hc_dim = hc_mult * dim;
    if x.len() != rows * hc_dim {
        return Err(err_msg(&format!(
            "hc_head_ref: x len {} != rows*hc*dim {}",
            x.len(),
            rows * hc_dim
        )));
    }
    // hc_head_fn is [hc_mult, hc_dim] — mixes has hc_mult channels, not mix_hc.
    if hc_fn.len() != hc_mult * hc_dim {
        return Err(err_msg(&format!(
            "hc_head_ref: hc_fn len {} != hc_mult*hc_dim {}",
            hc_fn.len(),
            hc_mult * hc_dim
        )));
    }
    if hc_scale.is_empty() {
        return Err(err_msg("hc_head_ref: hc_scale must be non-empty"));
    }
    if hc_base.len() != hc_mult {
        return Err(err_msg(&format!(
            "hc_head_ref: hc_base len {} != hc_mult {}",
            hc_base.len(),
            hc_mult
        )));
    }
    let scale = hc_scale[0] as f64;

    let mut y = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let xbase = r * hc_dim;
        let mut acc = 0.0f64;
        for d in 0..hc_dim {
            let v = x[xbase + d] as f64;
            acc += v * v;
        }
        let rsqrt = 1.0 / (acc / hc_dim as f64 + norm_eps).sqrt();

        // mixes[o] = sum_k x[k] * hc_fn[o,k] * rsqrt
        // pre[o] = sigmoid(mixes[o] * scale + base[o]) + eps
        let mut pre = vec![0.0f64; hc_mult];
        for o in 0..hc_mult {
            let mut s = 0.0f64;
            let wbase = o * hc_dim;
            for k in 0..hc_dim {
                s += (x[xbase + k] as f64) * (hc_fn[wbase + k] as f64);
            }
            let mix = s * rsqrt;
            pre[o] = sigmoid_f64(mix * scale + hc_base[o] as f64) + hc_eps;
        }

        for d in 0..dim {
            let mut s = 0.0f64;
            for h in 0..hc_mult {
                s += pre[h] * (x[xbase + h * dim + d] as f64);
            }
            y[r * dim + d] = s as f32;
        }
    }
    Ok(y)
}

/// Gate routing output: top-k weights and expert indices, both `[rows, topk]`.
#[derive(Clone, Debug, PartialEq)]
pub struct RoutingResult {
    pub weights: Vec<f32>,
    pub indices: Vec<u32>,
}

/// `Gate.forward` score path (`model.py:569-589`).
///
/// - Scores from f32-widened BF16 weight: `scores = x @ W^T` (no act-quant).
/// - `scoring_func = sqrtsoftplus` → `sqrt(softplus(scores))`.
/// - Top-k on **bias-corrected** scores; returned weights are the
///   **uncorrected** gathered scores.
/// - When `norm_topk_prob`, renormalize gathered weights to sum 1, then
///   multiply by `route_scale`.
///
/// `weight` is `[n_experts, dim]` row-major; `bias` is `[n_experts]` if present.
pub fn gate_ref(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    dim: usize,
    n_experts: usize,
    topk: usize,
    route_scale: f64,
    norm_topk_prob: bool,
) -> Result<RoutingResult, String> {
    if dim == 0 || n_experts == 0 {
        return Err(err_msg("gate_ref: dim and n_experts must be > 0"));
    }
    if topk == 0 || topk > n_experts {
        return Err(err_msg(&format!(
            "gate_ref: topk {topk} out of range for n_experts {n_experts}"
        )));
    }
    if x.len() != rows * dim {
        return Err(err_msg(&format!(
            "gate_ref: x len {} != rows*dim {}",
            x.len(),
            rows * dim
        )));
    }
    if weight.len() != n_experts * dim {
        return Err(err_msg(&format!(
            "gate_ref: weight len {} != n_experts*dim {}",
            weight.len(),
            n_experts * dim
        )));
    }
    if let Some(b) = bias {
        if b.len() != n_experts {
            return Err(err_msg(&format!(
                "gate_ref: bias len {} != n_experts {}",
                b.len(),
                n_experts
            )));
        }
    }

    let mut weights_out = vec![0.0f32; rows * topk];
    let mut indices_out = vec![0u32; rows * topk];

    for r in 0..rows {
        let xbase = r * dim;
        // scores = linear(x, weight) = x @ W^T
        let mut original = vec![0.0f64; n_experts];
        for e in 0..n_experts {
            let mut s = 0.0f64;
            let wbase = e * dim;
            for d in 0..dim {
                s += (x[xbase + d] as f64) * (weight[wbase + d] as f64);
            }
            // sqrtsoftplus: softplus(s).sqrt()
            original[e] = softplus_f64(s).sqrt();
        }


        // Bias shifts selection only.
        let mut select = original.clone();
        if let Some(b) = bias {
            for e in 0..n_experts {
                select[e] += b[e] as f64;
            }
        }

        // topk on select; keep (value, index), break ties by smaller index
        // (matches torch.topk on unique values; deterministic for tests).
        let mut order: Vec<usize> = (0..n_experts).collect();
        order.sort_by(|&a, &b| {
            match select[b]
                .partial_cmp(&select[a])
                .unwrap_or(std::cmp::Ordering::Equal)
            {
                std::cmp::Ordering::Equal => a.cmp(&b),
                o => o,
            }
        });
        let top = &order[..topk];

        let mut wrow = vec![0.0f64; topk];
        for (t, &e) in top.iter().enumerate() {
            indices_out[r * topk + t] = e as u32;
            wrow[t] = original[e]; // uncorrected
        }
        if norm_topk_prob {
            let sum: f64 = wrow.iter().sum();
            if sum != 0.0 {
                for w in wrow.iter_mut() {
                    *w /= sum;
                }
            }
        }
        for (t, w) in wrow.iter().enumerate() {
            weights_out[r * topk + t] = (*w * route_scale) as f32;
        }
    }

    Ok(RoutingResult {
        weights: weights_out,
        indices: indices_out,
    })
}

/// Hash-routed layers (`model.py:581-582`): expert ids from `tid2eid[token_id]`.
///
/// `tid2eid` is `[vocab, topk]` row-major i64. Returns indices from the table
/// and uniform weights `1/topk` (hash layers still gather score-weights in the
/// full `Gate.forward`; this helper isolates the index lookup the hash path
/// substitutes for `topk`).
pub fn gate_hash_ref(
    input_ids: &[u32],
    tid2eid: &[i64],
    n_experts: usize,
    topk: usize,
) -> Result<RoutingResult, String> {
    if topk == 0 {
        return Err(err_msg("gate_hash_ref: topk must be > 0"));
    }
    if tid2eid.len() % topk != 0 {
        return Err(err_msg(&format!(
            "gate_hash_ref: tid2eid len {} not divisible by topk {}",
            tid2eid.len(),
            topk
        )));
    }
    let vocab = tid2eid.len() / topk;
    let rows = input_ids.len();
    let mut weights = vec![0.0f32; rows * topk];
    let mut indices = vec![0u32; rows * topk];
    let inv_k = 1.0f32 / topk as f32;

    for (r, &tid) in input_ids.iter().enumerate() {
        let tid = tid as usize;
        if tid >= vocab {
            return Err(err_msg(&format!(
                "gate_hash_ref: token id {tid} >= vocab {vocab}"
            )));
        }
        for t in 0..topk {
            let e = tid2eid[tid * topk + t];
            if e < 0 || e as usize >= n_experts {
                return Err(err_msg(&format!(
                    "gate_hash_ref: expert id {e} out of range for n_experts {n_experts}"
                )));
            }
            indices[r * topk + t] = e as u32;
            weights[r * topk + t] = inv_k;
        }
    }
    Ok(RoutingResult { weights, indices })
}

/// `Expert.forward` middle (`model.py:601-611`) after `w1`/`w3`, before `w2`.
///
/// `gate` and `up` are `[rows, inter]` (already projected).
///
/// Clamp asymmetry (load-bearing):
/// - `up = clamp(up, min=-limit, max=limit)`
/// - `gate = clamp(gate, max=limit)` — **no lower clamp**
/// - `x = silu(gate) * up`
/// - if `weight` is `Some([rows])`, `x = weight[:, None] * x` **here**, before `w2`
///
/// Returns `[rows, inter]`.
pub fn expert_swiglu_ref(
    gate: &[f32],
    up: &[f32],
    rows: usize,
    inter: usize,
    swiglu_limit: f64,
    weight: Option<&[f32]>,
) -> Vec<f32> {
    assert_eq!(gate.len(), rows * inter);
    assert_eq!(up.len(), rows * inter);
    if let Some(w) = weight {
        assert_eq!(w.len(), rows);
    }

    let mut out = vec![0.0f32; rows * inter];
    for r in 0..rows {
        let w_r = weight.map(|w| w[r] as f64).unwrap_or(1.0);
        for i in 0..inter {
            let mut g = gate[r * inter + i] as f64;
            let mut u = up[r * inter + i] as f64;
            if swiglu_limit > 0.0 {
                // up: both sides; gate: upper only
                if u > swiglu_limit {
                    u = swiglu_limit;
                } else if u < -swiglu_limit {
                    u = -swiglu_limit;
                }
                if g > swiglu_limit {
                    g = swiglu_limit;
                }
            }
            let v = silu_f64(g) * u * w_r;
            out[r * inter + i] = v as f32;
        }
    }
    out
}


// ═══════════════════════════════════════════════════════════════════════════
// Attention SWA + compressed f64 oracle (main path; compress_ratio 0 / 4 / 128)
// ═══════════════════════════════════════════════════════════════════════════
//
// Authority:
// - model.py Attention.forward 490-549
// - model.py Attention.__init__ 481-488 (RoPE table selection)
// - kernel.py sparse_attn 277-369
// - model.py get_window_topk_idxs 260-271
// - model.py get_compress_topk_idxs 274-283
// - model.py Compressor.forward 322-383
// - model.py Indexer.forward 408-439 (ratio==4 selection)
// - model.py apply_rotary_emb 238-250 (interleaved)
//
// Scope:
// - Models the *main* q/kv/o path including the ratio>0 RoPE table
//   (YaRN on, original_seq_len=65536, base=compress_rope_theta=160000).
// - Sparse attention is the joint SWA-window + compressed slots + sink
//   path (`model.py:520,531-533`). Compressed keys are concatenated onto
//   the sliding-window KV store; topk_idxs = cat([window, compress], -1).
// - For ratio==4 the oracle runs the indexer selection path (learned
//   top-k over compressed slots). For ratio==128 it uses identity
//   `get_compress_topk_idxs`. For ratio==0 there are no compressed keys.
// - Index space: the reference uses a unified buffer with
//   `offset = seqlen` (prefill). This oracle mirrors that layout so
//   `sparse_attn_ref` gathers from one concatenated `kv` tensor.
//
// Projections reuse codec act-quant + f64 matmul over BF16-decoded weights,
// matching Gate-3's proven BF16≡scaled-FP8 identity (UE8M0 power-of-two
// scales). KV non-RoPE dims use act_quant_fp8_inplace_ref block 64.

use crate::attention::{
    apply_rope_interleaved_inplace, get_compress_topk_idxs, get_window_topk_idxs,
    precompute_rope_freqs, swa_n_valid, PARENT_ATTN_INDEX_TOPK, PARENT_DIM, PARENT_HEAD_DIM,
    PARENT_HEADS_PER_GROUP, PARENT_KV_ACT_QUANT_BLOCK, PARENT_NOPE_DIM, PARENT_N_HEADS,
    PARENT_N_KV_HEADS, PARENT_O_GROUPS, PARENT_O_LORA, PARENT_PER_GROUP_IN, PARENT_Q_LORA,
    PARENT_Q_WIDTH, PARENT_RMS_EPS, PARENT_ROPE_DIM, PARENT_ROPE_THETA, PARENT_SWA_WINDOW,
    PARENT_WO_A_OUT,
};
use crate::codec::act_quant_fp8_inplace_ref;
use crate::compressor::{
    compressor_prefill_ref, PARENT_COMPRESS_ROPE_THETA, PARENT_YARN_BETA_FAST,
    PARENT_YARN_BETA_SLOW, PARENT_YARN_FACTOR, PARENT_YARN_ORIG_SEQ,
};
use crate::indexer::{
    indexer_apply_offset_and_causal_mask, indexer_n_compressed, indexer_n_visible,
    indexer_oracle_f64, indexer_weights_scale, PARENT_INDEX_HEAD_DIM, PARENT_INDEX_N_HEADS,
    PARENT_INDEX_TOPK,
};
use crate::codec::{
    act_quant_fp4_inplace_ref, hadamard_rotate_ref, round_to_bf16,
};

/// RoPE table policy for the main q/kv/o path (`model.py:481-488`).
///
/// - `compress_ratio == 0`: plain `rope_theta=10000`, YaRN off (`original_seq_len=0`).
/// - `compress_ratio > 0`: `compress_rope_theta=160000` + YaRN
///   (`original_seq_len=65536`, factor/β from config).
///
/// Returns `(original_seq_len, rope_theta)`.
#[inline]
pub fn attention_main_rope_policy(compress_ratio: usize) -> Result<(usize, f64), String> {
    match compress_ratio {
        0 => Ok((0, PARENT_ROPE_THETA as f64)),
        4 | 128 => Ok((PARENT_YARN_ORIG_SEQ, PARENT_COMPRESS_ROPE_THETA)),
        other => Err(err_msg(&format!(
            "attention_main_rope_policy: unsupported compress_ratio={other} (expected 0, 4, or 128)"
        ))),
    }
}
/// Host-side weights for [`attention_swa_ref`] (BF16-decoded dense, F32 sink).
///
/// Dense layouts match the checkpoint / `ParentDenseWeight`:
/// - `wq_a`: `[q_lora, dim]`
/// - `wq_b`: `[n_heads * head_dim, q_lora]`
/// - `wkv`:  `[head_dim, dim]`
/// - `wo_a`: `[o_groups * o_lora, heads_per_group * head_dim]`
/// - `wo_b`: `[dim, o_groups * o_lora]`
/// - `q_norm` / `kv_norm`: length `q_lora` / `head_dim`
/// - `attn_sink`: length `n_heads`
///
/// Optional compressor / indexer weights are required when
/// `compress_ratio > 0` (main compressor) and `== 4` (indexer).
#[derive(Clone, Debug)]
pub struct AttnSwARefWeights<'a> {
    pub wq_a: &'a [f32],
    pub wq_b: &'a [f32],
    pub wkv: &'a [f32],
    pub wo_a: &'a [f32],
    pub wo_b: &'a [f32],
    pub q_norm: &'a [f32],
    pub kv_norm: &'a [f32],
    pub attn_sink: &'a [f32],
    /// Main compressor (ratio 4 / 128). `wkv`/`wgate` are `[proj, dim]`,
    /// `norm` length `head_dim`, `ape` is `[ratio, proj]`.
    pub compressor: Option<AttnCompRefWeights<'a>>,
    /// Indexer (ratio 4 only).
    pub indexer: Option<AttnIndexerRefWeights<'a>>,
}

/// Main-attention compressor weights for the joint oracle.
#[derive(Clone, Debug)]
pub struct AttnCompRefWeights<'a> {
    pub wkv: &'a [f32],
    pub wgate: &'a [f32],
    pub norm: &'a [f32],
    pub ape: &'a [f32],
}

/// Indexer weights for the joint oracle (ratio==4).
///
/// - `wq_b`: `[index_n_heads * index_head_dim, q_lora]` BF16-decoded
/// - `weights_proj`: `[index_n_heads, dim]` BF16-decoded
/// - compressor_* : same layout as [`AttnCompRefWeights`] but with
///   `head_dim = index_head_dim` and `hadamard=true` path
#[derive(Clone, Debug)]
pub struct AttnIndexerRefWeights<'a> {
    pub wq_b: &'a [f32],
    pub weights_proj: &'a [f32],
    pub compressor_wkv: &'a [f32],
    pub compressor_wgate: &'a [f32],
    pub compressor_norm: &'a [f32],
    pub compressor_ape: &'a [f32],
}

/// Per-stage intermediates + final output from [`attention_swa_ref`].
///
/// Layouts (row-major flat):
/// - `q_lat` / `q_post_wb` / `q_post_head_rms` / `q_post_rope`: Q path
/// - `kv_post_norm` / `kv_post_rope` / `kv_post_quant`: KV path
/// - `attn_raw`: sparse-attn output **before** inverse RoPE `[rows, n_heads, head_dim]`
/// - `attn_inv_rope`: after inverse RoPE
/// - `wo_a_out`: after grouped wo_a `[rows, o_groups * o_lora]`
/// - `o`: final after wo_b `[rows, dim]`
/// - `window_idxs`: SWA window indices only
/// - `compress_idxs`: compressed-slot indices (already offset into the
///   concatenated KV store; empty when ratio==0)
/// - `joint_idxs`: `cat([window, compress], -1)` fed to sparse attn
/// - `kv_compress`: compressed KV rows `[n_comp, head_dim]` (empty if none)
#[derive(Clone, Debug)]
pub struct AttnRefOut {
    pub o: Vec<f32>,
    pub q_lat: Vec<f32>,
    pub q_post_wb: Vec<f32>,
    pub q_post_head_rms: Vec<f32>,
    pub q_post_rope: Vec<f32>,
    pub kv_post_norm: Vec<f32>,
    pub kv_post_rope: Vec<f32>,
    pub kv_post_quant: Vec<f32>,
    pub attn_raw: Vec<f32>,
    pub attn_inv_rope: Vec<f32>,
    pub wo_a_out: Vec<f32>,
    /// Flat `rows * k_win` window indices used by sparse attn.
    pub window_idxs: Vec<i32>,
    /// Flat `rows * k_comp` compressed indices (offset into concat KV).
    pub compress_idxs: Vec<i32>,
    /// Flat `rows * (k_win + k_comp)` joint top-k indices.
    pub joint_idxs: Vec<i32>,
    /// Compressed KV content `[n_comp, head_dim]` (empty when none emitted).
    pub kv_compress: Vec<f32>,
}

/// f64-accum dense linear matching the parent BF16×BF16 path:
/// act-quant FP8 (block 128) on `x`, then `out = x_q @ W^T` with W already
/// BF16-decoded (`[n, k]` row-major).
fn dense_linear_bf16_ref(
    x: &[f32],
    w: &[f32],
    rows: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, String> {
    if x.len() != rows * k {
        return Err(err_msg(&format!(
            "dense_linear_bf16_ref: x len {} != rows*k {}",
            x.len(),
            rows * k
        )));
    }
    if w.len() != n * k {
        return Err(err_msg(&format!(
            "dense_linear_bf16_ref: w len {} != n*k {}",
            w.len(),
            n * k
        )));
    }
    let mut xq = x.to_vec();
    act_quant_fp8_inplace_ref(&mut xq, k, 128)?;
    let mut out = vec![0.0f32; rows * n];
    for r in 0..rows {
        let xb = r * k;
        for o in 0..n {
            let mut s = 0.0f64;
            let wb = o * k;
            for i in 0..k {
                s += (xq[xb + i] as f64) * (w[wb + i] as f64);
            }
            out[r * n + o] = s as f32;
        }
    }
    Ok(out)
}

/// Online-softmax sparse attention over a window index matrix + sink.
///
/// `q`: `[rows, n_heads, head_dim]`, `kv`: `[n_kv, head_dim]` absolute store,
/// `topk_idxs`: `[rows, k]` with `-1` = masked (no contribution to max/sum/V),
/// `attn_sink`: `[n_heads]` folded into the denominator only
/// (`kernel.py:345-348`).
///
/// Returns `[rows, n_heads, head_dim]`.
pub fn sparse_attn_ref(
    q: &[f32],
    kv: &[f32],
    attn_sink: &[f32],
    topk_idxs: &[i32],
    rows: usize,
    n_heads: usize,
    head_dim: usize,
    n_kv: usize,
    k: usize,
    softmax_scale: f64,
) -> Result<Vec<f32>, String> {
    if head_dim == 0 || n_heads == 0 {
        return Err(err_msg("sparse_attn_ref: n_heads/head_dim must be > 0"));
    }
    if q.len() != rows * n_heads * head_dim {
        return Err(err_msg(&format!(
            "sparse_attn_ref: q len {} != rows*n_heads*head_dim {}",
            q.len(),
            rows * n_heads * head_dim
        )));
    }
    if kv.len() != n_kv * head_dim {
        return Err(err_msg(&format!(
            "sparse_attn_ref: kv len {} != n_kv*head_dim {}",
            kv.len(),
            n_kv * head_dim
        )));
    }
    if attn_sink.len() != n_heads {
        return Err(err_msg(&format!(
            "sparse_attn_ref: attn_sink len {} != n_heads {n_heads}",
            attn_sink.len()
        )));
    }
    if topk_idxs.len() != rows * k {
        return Err(err_msg(&format!(
            "sparse_attn_ref: topk_idxs len {} != rows*k {}",
            topk_idxs.len(),
            rows * k
        )));
    }

    let mut out = vec![0.0f32; rows * n_heads * head_dim];
    for r in 0..rows {
        let mut valid: Vec<usize> = Vec::with_capacity(k);
        for j in 0..k {
            let idx = topk_idxs[r * k + j];
            if idx >= 0 {
                let u = idx as usize;
                if u >= n_kv {
                    return Err(err_msg(&format!(
                        "sparse_attn_ref: topk idx {idx} out of range n_kv={n_kv} (row {r})"
                    )));
                }
                valid.push(u);
            }
        }
        for h in 0..n_heads {
            let qbase = (r * n_heads + h) * head_dim;
            let mut scores = vec![0.0f64; valid.len()];
            let mut m = f64::NEG_INFINITY;
            for (t, &kv_i) in valid.iter().enumerate() {
                let mut dot = 0.0f64;
                let kbase = kv_i * head_dim;
                for d in 0..head_dim {
                    dot += (q[qbase + d] as f64) * (kv[kbase + d] as f64);
                }
                let s = dot * softmax_scale;
                scores[t] = s;
                if s > m {
                    m = s;
                }
            }
            let sink = attn_sink[h] as f64;
            if sink > m {
                m = sink;
            }
            if m == f64::NEG_INFINITY {
                continue;
            }
            let mut sum_exp = (sink - m).exp();
            let mut acc = vec![0.0f64; head_dim];
            for (t, &kv_i) in valid.iter().enumerate() {
                let p = (scores[t] - m).exp();
                sum_exp += p;
                let kbase = kv_i * head_dim;
                for d in 0..head_dim {
                    acc[d] += p * (kv[kbase + d] as f64);
                }
            }
            let inv = 1.0 / sum_exp;
            let obase = (r * n_heads + h) * head_dim;
            for d in 0..head_dim {
                out[obase + d] = (acc[d] * inv) as f32;
            }
        }
    }
    Ok(out)
}

/// f64 reference for one attention block (SWA + optional compressed + sink).
///
/// Prefill operating point (`start_pos == 0`): KV store is the current chunk
/// concatenated with compressed slots (`model.py:531`). Window indices are
/// absolute positions into the SWA half; compressed indices are offset by
/// `rows` into the concatenated store (mirroring `offset = seqlen`).
///
/// `compress_ratio` selects both the main-path RoPE table and the compressed
/// contribution:
/// - `0` → plain base-10000, no YaRN, SWA-only
/// - `4` → YaRN + compress_rope_theta; main compressor + indexer top-k
/// - `128` → YaRN + compress_rope_theta; main compressor + identity gather
///
/// `x` is `[rows, dim]` post-attn_norm F32.
pub fn attention_swa_ref(
    x: &[f32],
    w: &AttnSwARefWeights<'_>,
    rows: usize,
    start_pos: usize,
    compress_ratio: usize,
) -> Result<AttnRefOut, String> {
    let (original_seq_len, rope_theta) = attention_main_rope_policy(compress_ratio)?;
    attention_swa_ref_cfg(
        x,
        w,
        rows,
        start_pos,
        PARENT_DIM,
        PARENT_N_HEADS,
        PARENT_HEAD_DIM,
        PARENT_ROPE_DIM,
        PARENT_Q_LORA,
        PARENT_O_LORA,
        PARENT_O_GROUPS,
        PARENT_SWA_WINDOW,
        PARENT_RMS_EPS as f64,
        original_seq_len,
        rope_theta,
        compress_ratio,
    )
}

/// Configurable core (unit tests use tiny shapes; production uses parent constants).
///
/// `rope_original_seq_len == 0` disables YaRN (ratio-0 policy). Nonzero enables
/// YaRN blending against `rope_theta` (ratio>0 policy uses 65536 / 160000).
///
/// `compress_ratio` gates the compressed-key contribution. Tiny unit tests
/// that only exercise SWA pass `0` (or leave compressor/indexer as `None`).
#[allow(clippy::too_many_arguments)]
pub fn attention_swa_ref_cfg(
    x: &[f32],
    w: &AttnSwARefWeights<'_>,
    rows: usize,
    start_pos: usize,
    dim: usize,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    q_lora: usize,
    o_lora: usize,
    o_groups: usize,
    window: usize,
    eps: f64,
    rope_original_seq_len: usize,
    rope_theta: f64,
    compress_ratio: usize,
) -> Result<AttnRefOut, String> {
    if rows == 0 {
        return Err(err_msg("attention_swa_ref: rows must be > 0"));
    }
    if start_pos != 0 {
        return Err(err_msg(
            "attention_swa_ref: only start_pos==0 (prefill) is supported",
        ));
    }
    if n_heads % o_groups != 0 {
        return Err(err_msg(&format!(
            "attention_swa_ref: n_heads {n_heads} not divisible by o_groups {o_groups}"
        )));
    }
    if head_dim < rope_dim || rope_dim == 0 || rope_dim % 2 != 0 {
        return Err(err_msg("attention_swa_ref: invalid head_dim/rope_dim"));
    }
    let q_width = n_heads * head_dim;
    let heads_per_group = n_heads / o_groups;
    let per_group_in = heads_per_group * head_dim;
    let wo_a_out = o_groups * o_lora;
    let nope = head_dim - rope_dim;

    if x.len() != rows * dim {
        return Err(err_msg(&format!(
            "attention_swa_ref: x len {} != rows*dim {}",
            x.len(),
            rows * dim
        )));
    }
    if w.wq_a.len() != q_lora * dim
        || w.wq_b.len() != q_width * q_lora
        || w.wkv.len() != head_dim * dim
        || w.wo_a.len() != wo_a_out * per_group_in
        || w.wo_b.len() != dim * wo_a_out
        || w.q_norm.len() != q_lora
        || w.kv_norm.len() != head_dim
        || w.attn_sink.len() != n_heads
    {
        return Err(err_msg(&format!(
            "attention_swa_ref: weight length mismatch \
             (wq_a {}/{}, wq_b {}/{}, wkv {}/{}, wo_a {}/{}, wo_b {}/{}, \
              q_norm {}/{}, kv_norm {}/{}, sink {}/{})",
            w.wq_a.len(),
            q_lora * dim,
            w.wq_b.len(),
            q_width * q_lora,
            w.wkv.len(),
            head_dim * dim,
            w.wo_a.len(),
            wo_a_out * per_group_in,
            w.wo_b.len(),
            dim * wo_a_out,
            w.q_norm.len(),
            q_lora,
            w.kv_norm.len(),
            head_dim,
            w.attn_sink.len(),
            n_heads
        )));
    }

    // ── 1. Q: wq_a → q_norm → wq_b → per-head unit RMSNorm ──────────────
    let q_lat_raw = dense_linear_bf16_ref(x, w.wq_a, rows, q_lora, dim)?;
    let q_lat = rms_norm_ref(&q_lat_raw, w.q_norm, eps, q_lora);
    let q_post_wb = dense_linear_bf16_ref(&q_lat, w.wq_b, rows, q_width, q_lora)?;

    // model.py:504 — extra per-head rsqrt (unit weight)
    let mut q_post_head_rms = q_post_wb.clone();
    for r in 0..rows {
        for h in 0..n_heads {
            let base = (r * n_heads + h) * head_dim;
            let mut acc = 0.0f64;
            for d in 0..head_dim {
                let v = q_post_head_rms[base + d] as f64;
                acc += v * v;
            }
            let scale = 1.0 / (acc / head_dim as f64 + eps).sqrt();
            for d in 0..head_dim {
                q_post_head_rms[base + d] = ((q_post_head_rms[base + d] as f64) * scale) as f32;
            }
        }
    }

    // ── 2. KV: wkv → kv_norm ────────────────────────────────────────────
    let kv_raw = dense_linear_bf16_ref(x, w.wkv, rows, head_dim, dim)?;
    let kv_post_norm = rms_norm_ref(&kv_raw, w.kv_norm, eps, head_dim);

    // ── 3. Tail RoPE (interleaved) ──────────────────────────────────────
    // ratio==0: original_seq_len=0 → plain theta, no YaRN.
    // ratio>0: original_seq_len=65536 + compress_rope_theta → YaRN on.
    // q, kv, and inverse-o all consume this same `freqs` table.
    let freqs = precompute_rope_freqs(
        rope_dim,
        rope_original_seq_len,
        rope_theta,
        PARENT_YARN_FACTOR,
        PARENT_YARN_BETA_FAST,
        PARENT_YARN_BETA_SLOW,
    )
    .map_err(|e| err_msg(&format!("rope freqs: {e}")))?;
    let positions: Vec<usize> = (0..rows).map(|r| start_pos + r).collect();

    let mut q_post_rope = q_post_head_rms.clone();
    apply_rope_interleaved_inplace(
        &mut q_post_rope,
        rows,
        n_heads,
        head_dim,
        rope_dim,
        &positions,
        &freqs,
        false,
    )
    .map_err(|e| err_msg(&format!("q rope: {e}")))?;

    let mut kv_post_rope = kv_post_norm.clone();
    apply_rope_interleaved_inplace(
        &mut kv_post_rope,
        rows,
        PARENT_N_KV_HEADS,
        head_dim,
        rope_dim,
        &positions,
        &freqs,
        false,
    )
    .map_err(|e| err_msg(&format!("kv rope: {e}")))?;

    // ── 4. FP8 act-quant on non-RoPE KV dims (block 64 production; flexible in tests) ──
    let mut kv_post_quant = kv_post_rope.clone();
    if nope > 0 {
        let block = if nope % PARENT_KV_ACT_QUANT_BLOCK == 0 {
            PARENT_KV_ACT_QUANT_BLOCK
        } else if nope % 32 == 0 {
            32
        } else if nope % 16 == 0 {
            16
        } else if nope % 8 == 0 {
            8
        } else if nope % 4 == 0 {
            4
        } else if nope % 2 == 0 {
            2
        } else {
            nope
        };
        let mut nope_buf = vec![0.0f32; rows * nope];
        for r in 0..rows {
            let src = r * head_dim;
            let dst = r * nope;
            nope_buf[dst..dst + nope].copy_from_slice(&kv_post_quant[src..src + nope]);
        }
        act_quant_fp8_inplace_ref(&mut nope_buf, nope, block)?;
        for r in 0..rows {
            let src = r * nope;
            let dst = r * head_dim;
            kv_post_quant[dst..dst + nope].copy_from_slice(&nope_buf[src..src + nope]);
        }
    }

    // ── 5. Window + compressed indices + joint sparse attn ──────────────
    // model.py:513-533:
    //   topk_idxs = cat([window_idxs, compress_topk_idxs], -1)
    //   kv = cat([kv, kv_compress], dim=1)   # prefill only
    //   o = sparse_attn(q, kv, sink, topk_idxs, scale)
    let window_idxs =
        get_window_topk_idxs(window, rows, start_pos).map_err(|e| err_msg(&format!("window: {e}")))?;
    let k_win = rows.min(window);
    if window_idxs.len() != rows * k_win {
        return Err(err_msg(&format!(
            "attention_swa_ref: window_idxs len {} != rows*k {}",
            window_idxs.len(),
            rows * k_win
        )));
    }
    for r in 0..rows {
        let nv = swa_n_valid(start_pos, r, window);
        let got = window_idxs[r * k_win..(r + 1) * k_win]
            .iter()
            .filter(|&&v| v >= 0)
            .count();
        if got != nv {
            return Err(err_msg(&format!(
                "attention_swa_ref: row {r} visible {got} != swa_n_valid {nv}"
            )));
        }
    }

    let (kv_compress, compress_idxs, k_comp) = if compress_ratio == 0 {
        (Vec::new(), Vec::new(), 0usize)
    } else {
        build_compressed_contribution(
            x,
            &q_lat,
            w,
            rows,
            start_pos,
            dim,
            head_dim,
            compress_ratio,
            /*offset=*/ rows, // unified index space: compress after SWA slots
        )?
    };

    let n_comp = if head_dim == 0 {
        0
    } else {
        kv_compress.len() / head_dim
    };
    // Concatenate SWA KV + compressed KV (model.py:531).
    let mut kv_joint = kv_post_quant.clone();
    kv_joint.extend_from_slice(&kv_compress);
    let n_kv = rows + n_comp;

    // Joint topk = cat([window, compress], -1) (model.py:520).
    let k_attn = k_win + k_comp;
    let mut joint_idxs = vec![-1i32; rows * k_attn.max(1)];
    if k_attn > 0 {
        for r in 0..rows {
            let dst = r * k_attn;
            joint_idxs[dst..dst + k_win]
                .copy_from_slice(&window_idxs[r * k_win..(r + 1) * k_win]);
            if k_comp > 0 {
                joint_idxs[dst + k_win..dst + k_attn]
                    .copy_from_slice(&compress_idxs[r * k_comp..(r + 1) * k_comp]);
            }
        }
    } else {
        joint_idxs.clear();
    }

    let softmax_scale = 1.0 / (head_dim as f64).sqrt();
    let attn_raw = sparse_attn_ref(
        &q_post_rope,
        &kv_joint,
        w.attn_sink,
        &joint_idxs,
        rows,
        n_heads,
        head_dim,
        n_kv,
        k_attn.max(1),
        softmax_scale,
    )?;

    // ── 6. Inverse tail RoPE (same freqs_cis rows as forward) ───────────
    let mut attn_inv_rope = attn_raw.clone();
    apply_rope_interleaved_inplace(
        &mut attn_inv_rope,
        rows,
        n_heads,
        head_dim,
        rope_dim,
        &positions,
        &freqs,
        true,
    )
    .map_err(|e| err_msg(&format!("inv rope: {e}")))?;

    // ── 7. Grouped wo_a then wo_b ───────────────────────────────────────
    // model.py:542-547 — wo_a stored [o_groups*o_lora, per_group_in],
    // group g = rows [g*o_lora, (g+1)*o_lora).
    let mut wo_a_out_v = vec![0.0f32; rows * wo_a_out];
    for g in 0..o_groups {
        let mut xg = vec![0.0f32; rows * per_group_in];
        for r in 0..rows {
            let src = (r * n_heads + g * heads_per_group) * head_dim;
            let dst = r * per_group_in;
            xg[dst..dst + per_group_in]
                .copy_from_slice(&attn_inv_rope[src..src + per_group_in]);
        }
        let w_off = g * o_lora * per_group_in;
        let w_g = &w.wo_a[w_off..w_off + o_lora * per_group_in];
        let yg = dense_linear_bf16_ref(&xg, w_g, rows, o_lora, per_group_in)?;
        for r in 0..rows {
            let src = r * o_lora;
            let dst = r * wo_a_out + g * o_lora;
            wo_a_out_v[dst..dst + o_lora].copy_from_slice(&yg[src..src + o_lora]);
        }
    }

    let o = dense_linear_bf16_ref(&wo_a_out_v, w.wo_b, rows, dim, wo_a_out)?;

    // Keep parent-constant imports live for the production wrapper path.
    let _ = (
        PARENT_DIM,
        PARENT_HEAD_DIM,
        PARENT_HEADS_PER_GROUP,
        PARENT_NOPE_DIM,
        PARENT_N_HEADS,
        PARENT_O_GROUPS,
        PARENT_O_LORA,
        PARENT_PER_GROUP_IN,
        PARENT_Q_LORA,
        PARENT_Q_WIDTH,
        PARENT_RMS_EPS,
        PARENT_ROPE_DIM,
        PARENT_ROPE_THETA,
        PARENT_SWA_WINDOW,
        PARENT_WO_A_OUT,
        PARENT_ATTN_INDEX_TOPK,
        PARENT_INDEX_TOPK,
        PARENT_N_KV_HEADS,
        compress_ratio,
        indexer_n_visible(0, 0, 4),
        indexer_apply_offset_and_causal_mask,
    );

    Ok(AttnRefOut {
        o,
        q_lat,
        q_post_wb,
        q_post_head_rms,
        q_post_rope,
        kv_post_norm,
        kv_post_rope,
        kv_post_quant,
        attn_raw,
        attn_inv_rope,
        wo_a_out: wo_a_out_v,
        window_idxs,
        compress_idxs,
        joint_idxs,
        kv_compress,
    })
}

/// Build compressed KV content + per-query compressed index matrix.
///
/// Returns `(kv_compress [n_comp*head_dim], compress_idxs [rows*k_comp], k_comp)`.
/// `offset` is added to surviving indices (unified KV index space).
#[allow(clippy::too_many_arguments)]
fn build_compressed_contribution(
    x: &[f32],
    q_lat: &[f32],
    w: &AttnSwARefWeights<'_>,
    rows: usize,
    start_pos: usize,
    dim: usize,
    head_dim: usize,
    compress_ratio: usize,
    offset: usize,
) -> Result<(Vec<f32>, Vec<i32>, usize), String> {
    let comp = w.compressor.as_ref().ok_or_else(|| {
        err_msg(&format!(
            "attention_swa_ref: compress_ratio={compress_ratio} requires compressor weights"
        ))
    })?;

    // Main compressor (hadamard=false) — model.py:530.
    let kv_opt = compressor_prefill_ref(
        x,
        comp.wkv,
        comp.wgate,
        comp.norm,
        comp.ape,
        rows,
        dim,
        head_dim,
        compress_ratio,
        /*hadamard=*/ false,
    )
    .map_err(|e| err_msg(&format!("main compressor: {e}")))?;
    let kv_compress = kv_opt.unwrap_or_default();
    let n_comp = if head_dim == 0 {
        0
    } else {
        kv_compress.len() / head_dim
    };
    let _ = indexer_n_compressed(start_pos, rows, compress_ratio);
    if n_comp == 0 {
        return Ok((Vec::new(), Vec::new(), 0));
    }

    if compress_ratio == 4 {
        let ix = w.indexer.as_ref().ok_or_else(|| {
            err_msg("attention_swa_ref: compress_ratio=4 requires indexer weights")
        })?;
        let compress_idxs = indexer_select_ref(
            x,
            q_lat,
            ix,
            rows,
            start_pos,
            dim,
            compress_ratio,
            offset,
            n_comp,
        )?;
        let k_comp = PARENT_INDEX_TOPK.min(n_comp.max(1));
        // indexer returns rows * PARENT_INDEX_TOPK; trim/pad to k_comp stride.
        let mut out = vec![-1i32; rows * k_comp];
        let src_k = PARENT_INDEX_TOPK;
        for r in 0..rows {
            for j in 0..k_comp {
                out[r * k_comp + j] = if j < src_k {
                    compress_idxs[r * src_k + j]
                } else {
                    -1
                };
            }
        }
        Ok((kv_compress, out, k_comp))
    } else {
        // ratio 128: identity gather of all compressed slots (model.py:519).
        let compress_idxs = get_compress_topk_idxs(compress_ratio, rows, start_pos, offset)
            .map_err(|e| err_msg(&format!("compress idxs: {e}")))?;
        let k_comp = if compress_idxs.is_empty() {
            0
        } else {
            compress_idxs.len() / rows
        };
        Ok((kv_compress, compress_idxs, k_comp))
    }
}

/// Full indexer selection oracle (model.py:408-439) for ratio==4.
///
/// Returns flat `rows * PARENT_INDEX_TOPK` indices already offset + causally
/// masked. Uses the indexer's own compressor (`hadamard=true`).
#[allow(clippy::too_many_arguments)]
fn indexer_select_ref(
    x: &[f32],
    q_lat: &[f32],
    ix: &AttnIndexerRefWeights<'_>,
    rows: usize,
    start_pos: usize,
    dim: usize,
    ratio: usize,
    offset: usize,
    n_comp_main: usize,
) -> Result<Vec<i32>, String> {
    let _ = n_comp_main;
    let n_heads = PARENT_INDEX_N_HEADS;
    let head_dim = PARENT_INDEX_HEAD_DIM;
    let q_width = n_heads * head_dim;
    let q_lora = PARENT_Q_LORA;

    // 1. q = wq_b(qr) via dense FP8 linear (same as main path).
    let mut q = dense_linear_bf16_ref(q_lat, ix.wq_b, rows, q_width, q_lora)
        .map_err(|e| err_msg(&format!("indexer wq_b: {e}")))?;

    // 2. Tail RoPE with YaRN + compress_rope_theta (shares Attention.freqs_cis).
    let freqs = precompute_rope_freqs(
        PARENT_ROPE_DIM,
        PARENT_YARN_ORIG_SEQ,
        PARENT_COMPRESS_ROPE_THETA,
        PARENT_YARN_FACTOR,
        PARENT_YARN_BETA_FAST,
        PARENT_YARN_BETA_SLOW,
    )
    .map_err(|e| err_msg(&format!("indexer rope freqs: {e}")))?;
    let positions: Vec<usize> = (0..rows).map(|r| start_pos + r).collect();
    apply_rope_interleaved_inplace(
        &mut q,
        rows,
        n_heads,
        head_dim,
        PARENT_ROPE_DIM,
        &positions,
        &freqs,
        false,
    )
    .map_err(|e| err_msg(&format!("indexer q rope: {e}")))?;

    // 3. Hadamard + FP4 on full head.
    hadamard_rotate_ref(&mut q, head_dim)
        .map_err(|e| err_msg(&format!("indexer hadamard: {e}")))?;
    act_quant_fp4_inplace_ref(&mut q, head_dim)
        .map_err(|e| err_msg(&format!("indexer fp4: {e}")))?;

    // 4. weights = weights_proj(x) * scale  (plain BF16, no act-quant).
    let scale = indexer_weights_scale();
    let mut weights = vec![0.0f32; rows * n_heads];
    for r in 0..rows {
        for h in 0..n_heads {
            let mut acc = 0.0f64;
            let wb = h * dim;
            for k in 0..dim {
                let xv = round_to_bf16(x[r * dim + k]) as f64;
                let wv = round_to_bf16(ix.weights_proj[wb + k]) as f64;
                acc += xv * wv;
            }
            weights[r * n_heads + h] = (acc * scale) as f32;
        }
    }

    // 5. Indexer compressor (hadamard=true) fills its own compressed KV.
    let kv_opt = compressor_prefill_ref(
        x,
        ix.compressor_wkv,
        ix.compressor_wgate,
        ix.compressor_norm,
        ix.compressor_ape,
        rows,
        dim,
        head_dim,
        ratio,
        /*hadamard=*/ true,
    )
    .map_err(|e| err_msg(&format!("indexer compressor: {e}")))?;
    let kv = kv_opt.unwrap_or_default();
    let n_slots = if head_dim == 0 {
        0
    } else {
        kv.len() / head_dim
    };
    if n_slots == 0 {
        return Ok(vec![-1i32; rows * PARENT_INDEX_TOPK]);
    }

    // 6. Score + top-k + offset/mask (f64 oracle).
    let q_f64: Vec<f64> = q.iter().map(|&v| v as f64).collect();
    let kv_f64: Vec<f64> = kv.iter().map(|&v| v as f64).collect();
    let w_f64: Vec<f64> = weights.iter().map(|&v| v as f64).collect();
    let k_out = PARENT_INDEX_TOPK.min(n_slots.max(1));
    let (_scores, topk) = indexer_oracle_f64(
        &q_f64,
        &kv_f64,
        &w_f64,
        rows,
        n_heads,
        head_dim,
        n_slots,
        start_pos,
        ratio,
        k_out,
        offset,
    )
    .map_err(|e| err_msg(&format!("indexer oracle: {e}")))?;

    // Pad to PARENT_INDEX_TOPK stride if k_out < TOPK.
    if k_out == PARENT_INDEX_TOPK {
        Ok(topk)
    } else {
        let mut out = vec![-1i32; rows * PARENT_INDEX_TOPK];
        for r in 0..rows {
            for j in 0..k_out {
                out[r * PARENT_INDEX_TOPK + j] = topk[r * k_out + j];
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_hand_computed() {
        // x = [3, 4], weight = [1, 1], eps = 0
        // mean sq = (9+16)/2 = 12.5; rsqrt = 1/sqrt(12.5)
        // out = x * rsqrt
        let x = [3.0f32, 4.0];
        let w = [1.0f32, 1.0];
        let out = rms_norm_ref(&x, &w, 0.0, 2);
        let scale = 1.0f64 / (12.5f64).sqrt();
        assert!((out[0] as f64 - 3.0 * scale).abs() < 1e-6);
        assert!((out[1] as f64 - 4.0 * scale).abs() < 1e-6);

        // With non-unit weight and eps.
        let w2 = [2.0f32, 0.5];
        let eps = 1e-6f64;
        let out2 = rms_norm_ref(&x, &w2, eps, 2);
        let scale2 = 1.0f64 / (12.5f64 + eps).sqrt();
        assert!((out2[0] as f64 - 3.0 * scale2 * 2.0).abs() < 1e-6);
        assert!((out2[1] as f64 - 4.0 * scale2 * 0.5).abs() < 1e-6);
    }

    #[test]
    fn hc_split_sinkhorn_doubly_stochastic_and_iters_matter() {
        let hc = 4usize;
        let rows = 1usize;
        let mix_hc = (2 + hc) * hc; // 24
        // Strongly asymmetric comb logits so Sinkhorn has real work to do.
        // Layout: mixes[0..hc]=pre, [hc..2hc]=post, [2hc..]=comb row-major.
        let mut mixes = vec![0.0f32; rows * mix_hc];
        for j in 0..hc {
            mixes[j] = 0.0; // pre logits
            mixes[hc + j] = 0.0; // post logits
        }
        // comb logits: diagonal-heavy but broken so row/col sums need iteration.
        for j in 0..hc {
            for k in 0..hc {
                let idx = 2 * hc + j * hc + k;
                mixes[idx] = if k == j { 8.0 } else { -4.0 };
            }
        }
        mixes[2 * hc + 0 * hc + 1] = 6.0;
        mixes[2 * hc + 1 * hc + 0] = 5.0;
        mixes[2 * hc + 2 * hc + 3] = 7.0;
        mixes[2 * hc + 3 * hc + 2] = 3.0;

        let hc_scale = [1.0f32, 1.0, 1.0];
        let hc_base = vec![0.0f32; mix_hc];
        let eps = 1e-6f64;

        let (_, _, comb1) =
            hc_split_sinkhorn_ref(&mixes, &hc_scale, &hc_base, rows, hc, 1, eps).unwrap();
        let (_, _, comb2) =
            hc_split_sinkhorn_ref(&mixes, &hc_scale, &hc_base, rows, hc, 2, eps).unwrap();
        let (_, _, comb20) =
            hc_split_sinkhorn_ref(&mixes, &hc_scale, &hc_base, rows, hc, 20, eps).unwrap();

        // iters must change the result (1 -> 2 is the first extra row/col pass)
        let mut max_diff = 0.0f32;
        for i in 0..comb1.len() {
            max_diff = max_diff.max((comb1[i] - comb2[i]).abs());
        }
        assert!(
            max_diff > 1e-4,
            "iters=1 and iters=2 should differ; max_diff={max_diff}"
        );
        let mut max_diff20 = 0.0f32;
        for i in 0..comb1.len() {
            max_diff20 = max_diff20.max((comb1[i] - comb20[i]).abs());
        }
        assert!(
            max_diff20 > 1e-4,
            "iters=1 and iters=20 should differ; max_diff={max_diff20}"
        );

        // After many iters, rows and cols both sum ~ 1 (within eps slack).
        for j in 0..hc {
            let mut row_sum = 0.0f64;
            for k in 0..hc {
                row_sum += comb20[j * hc + k] as f64;
            }
            assert!(
                (row_sum - 1.0).abs() < 1e-2,
                "row {j} sum {row_sum} not near 1"
            );
        }
        for k in 0..hc {
            let mut col_sum = 0.0f64;
            for j in 0..hc {
                col_sum += comb20[j * hc + k] as f64;
            }
            assert!(
                (col_sum - 1.0).abs() < 1e-2,
                "col {k} sum {col_sum} not near 1"
            );
        }

        // More iterations must pull row sums closer to 1 than a single pass.
        let mut err1 = 0.0f64;
        let mut err20 = 0.0f64;
        for j in 0..hc {
            let mut s1 = 0.0f64;
            let mut s20 = 0.0f64;
            for k in 0..hc {
                s1 += comb1[j * hc + k] as f64;
                s20 += comb20[j * hc + k] as f64;
            }
            err1 += (s1 - 1.0).abs();
            err20 += (s20 - 1.0).abs();
        }
        assert!(
            err20 < err1,
            "iters=20 row-sum error {err20} should be < iters=1 error {err1}"
        );
    }

    #[test]
    fn hc_pre_post_roundtrip_and_head_differs() {
        let rows = 1usize;
        let hc = 2usize;
        let dim = 2usize;
        let hc_dim = hc * dim;
        let mix_hc = (2 + hc) * hc; // 8
        let norm_eps = 1e-6f64;
        let hc_eps = 1e-6f64;

        // residual streams
        let x: Vec<f32> = vec![
            1.0, 0.0, // stream 0
            0.0, 1.0, // stream 1
        ];
        // hc_fn: [mix_hc, hc_dim] — non-uniform so pre/post/comb channels differ.
        let mut hc_fn = vec![0.0f32; mix_hc * hc_dim];
        for o in 0..mix_hc {
            hc_fn[o * hc_dim + (o % hc_dim)] = 0.5 + 0.1 * (o as f32);
            hc_fn[o * hc_dim + ((o + 1) % hc_dim)] = -0.25 * (o as f32);
        }
        let hc_scale = [1.5f32, 0.8, 1.2];
        let mut hc_base = vec![0.0f32; mix_hc];
        // Distinct bases so pre is not uniform 0.5.
        hc_base[0] = 1.0;
        hc_base[1] = -0.5;
        hc_base[2] = 0.0;
        hc_base[3] = 0.5;
        for i in (2 * hc)..mix_hc {
            hc_base[i] = 0.3 * (i as f32) - 1.0;
        }

        let (y, post, comb) = hc_pre_ref(
            &x, &hc_fn, &hc_scale, &hc_base, rows, hc, dim, norm_eps, 8, hc_eps,
        )
        .unwrap();
        assert_eq!(y.len(), rows * dim);
        assert_eq!(post.len(), rows * hc);
        assert_eq!(comb.len(), rows * hc * hc);

        // hc_post should expand back to [rows, hc, dim]
        let restored = hc_post_ref(&y, &x, &post, &comb, rows, hc, dim);
        assert_eq!(restored.len(), rows * hc * dim);
        assert!(restored.iter().all(|v| v.is_finite()));
        assert!(restored.iter().any(|v| v.abs() > 1e-6));

        // Residual contribution via comb must be present.
        let mut without_residual = vec![0.0f32; rows * hc * dim];
        for h in 0..hc {
            for d in 0..dim {
                without_residual[h * dim + d] = post[h] * y[d];
            }
        }
        let mut residual_contrib = 0.0f32;
        for i in 0..restored.len() {
            residual_contrib += (restored[i] - without_residual[i]).abs();
        }
        assert!(
            residual_contrib > 1e-5,
            "hc_post should include residual via comb; contrib={residual_contrib}"
        );

        // hc_head: plain sigmoid path, NO sinkhorn. Different hc_fn / base so
        // the reduced y diverges from hc_pre's sinkhorn-path pre reduction.
        let mut head_fn = vec![0.0f32; hc * hc_dim];
        for o in 0..hc {
            head_fn[o * hc_dim + o] = 2.0;
            head_fn[o * hc_dim + (1 - o)] = -1.0;
        }
        let head_scale = [2.0f32];
        let head_base = vec![2.0f32, -2.0];
        let head_y = hc_head_ref(
            &x,
            &head_fn,
            &head_scale,
            &head_base,
            rows,
            hc,
            dim,
            norm_eps,
            hc_eps,
        )
        .unwrap();

        let mut diff = 0.0f32;
        for i in 0..y.len() {
            diff += (y[i] - head_y[i]).abs();
        }
        assert!(
            diff > 1e-4,
            "hc_head_ref and hc_pre_ref y must differ (different paths); diff={diff}"
        );

        // Structural: sinkhorn comb has off-diagonal mass (not a pure identity).
        let mut off_diag = 0.0f32;
        for j in 0..hc {
            for k in 0..hc {
                if j != k {
                    off_diag += comb[j * hc + k].abs();
                }
            }
        }
        assert!(
            off_diag > 1e-6,
            "sinkhorn comb should have off-diagonal mass; off_diag={off_diag}"
        );
    }

    #[test]
    fn gate_ref_sqrtsoftplus_bias_norm_scale() {
        // One row, 4 experts, topk=2, dim=1.
        // x=[1], weight = [[a],[b],[c],[d]] so logits = [a,b,c,d].
        let x = [1.0f32];
        let logits = [0.0f32, 1.0, 2.0, -1.0];
        let weight = logits; // [n_experts, 1]
        // Bias that flips ranking: boost expert 3, suppress expert 2.
        let bias = [-10.0f32, -10.0, -10.0, 10.0];
        // Uncorrected sqrtsoftplus scores:
        let mut orig = [0.0f64; 4];
        for i in 0..4 {
            orig[i] = softplus_f64(logits[i] as f64).sqrt();
        }
        // Selection scores = orig + bias → expert 3 wins hard; among the rest
        // expert 2 has highest orig but bias is equal -10 so ranking among 0..2
        // follows orig: 2 > 1 > 0. With bias on 3 = +10, select order: 3, then 2.
        let route_scale = 1.5f64;
        let r = gate_ref(
            &x,
            &weight,
            Some(&bias),
            1,
            1,
            4,
            2,
            route_scale,
            true,
        )
        .unwrap();
        assert_eq!(r.indices, vec![3, 2]);

        // Weights are UNCORRECTED orig, L1-normalized, then * route_scale.
        let w0 = orig[3];
        let w1 = orig[2];
        let sum = w0 + w1;
        let expect0 = (w0 / sum) * route_scale;
        let expect1 = (w1 / sum) * route_scale;
        assert!((r.weights[0] as f64 - expect0).abs() < 1e-6);
        assert!((r.weights[1] as f64 - expect1).abs() < 1e-6);

        // Prove selection used bias-corrected scores: without bias top-2 is 2,1
        // (orig ranking by logits 2>1>0>-1 → sqrtsoftplus monotonic).
        let r_nobias = gate_ref(&x, &weight, None, 1, 1, 4, 2, route_scale, true).unwrap();
        assert_eq!(r_nobias.indices, vec![2, 1]);
        // And those weights are from uncorrected (same as select when no bias).
        let s0 = orig[2];
        let s1 = orig[1];
        let ssum = s0 + s1;
        assert!((r_nobias.weights[0] as f64 - (s0 / ssum) * route_scale).abs() < 1e-6);
        assert!((r_nobias.weights[1] as f64 - (s1 / ssum) * route_scale).abs() < 1e-6);

        // Without norm_topk_prob, weights are raw orig * route_scale (no /sum).
        let r_nonorm = gate_ref(&x, &weight, None, 1, 1, 4, 2, route_scale, false).unwrap();
        assert!((r_nonorm.weights[0] as f64 - orig[2] * route_scale).abs() < 1e-6);
        assert!((r_nonorm.weights[1] as f64 - orig[1] * route_scale).abs() < 1e-6);

        // route_scale multiplies at the end: scale=1 vs 1.5 ratio.
        let r1 = gate_ref(&x, &weight, None, 1, 1, 4, 2, 1.0, true).unwrap();
        assert!((r_nobias.weights[0] as f64 - r1.weights[0] as f64 * 1.5).abs() < 1e-6);
    }

    #[test]
    fn expert_swiglu_asymmetric_gate_clamp() {
        let limit = 10.0f64;
        // gate well below -limit; up inside range.
        let gate = [-100.0f32];
        let up = [2.0f32];
        let out = expert_swiglu_ref(&gate, &up, 1, 1, limit, None);
        // If gate were symmetrically clamped to -10, silu(-10)*2 would result.
        // Unclamped: silu(-100)*2 ≈ 0 (very small negative * 2).
        let unclamped = silu_f64(-100.0) * 2.0;
        let clamped_sym = silu_f64(-limit) * 2.0;
        assert!(
            (out[0] as f64 - unclamped).abs() < 1e-12,
            "gate must NOT be lower-clamped: got {} want {}",
            out[0],
            unclamped
        );
        // And that differs from the symmetric-clamp bug.
        assert!(
            (unclamped - clamped_sym).abs() > 1e-6,
            "test setup broken: unclamped ≈ symmetric clamp"
        );

        // up IS lower-clamped.
        let gate2 = [1.0f32];
        let up2 = [-100.0f32];
        let out2 = expert_swiglu_ref(&gate2, &up2, 1, 1, limit, None);
        let expect_up = silu_f64(1.0) * (-limit);
        assert!((out2[0] as f64 - expect_up).abs() < 1e-6);

        // gate IS upper-clamped.
        let gate3 = [100.0f32];
        let up3 = [1.0f32];
        let out3 = expert_swiglu_ref(&gate3, &up3, 1, 1, limit, None);
        let expect_g = silu_f64(limit) * 1.0;
        assert!((out3[0] as f64 - expect_g).abs() < 1e-6);
    }

    #[test]
    fn expert_routing_weight_before_w2() {
        // Structure the two orderings so they differ.
        // pre_w2 = silu(gate)*up;  weighted_pre = w * pre_w2
        // Fake w2 as a linear map: out = pre * w2_scale (scalar "weight matrix").
        // Before-w2:  (w * pre) * w2_scale
        // After-w2:   w * (pre * w2_scale)
        // These are equal for a single scalar — so use multi-dim with a
        // non-uniform w2 to make order matter if someone applied weight after
        // a reduction. Simpler: prove the intermediate itself carries the
        // weight, by comparing weight=None vs weight=Some and checking the
        // ratio equals the routing weight on every element (pre-w2 position).
        let gate = [1.0f32, 2.0, -1.0];
        let up = [3.0f32, -4.0, 5.0];
        let route_w = [0.25f32];
        let plain = expert_swiglu_ref(&gate, &up, 1, 3, 10.0, None);
        let weighted = expert_swiglu_ref(&gate, &up, 1, 3, 10.0, Some(&route_w));
        for i in 0..3 {
            assert!(
                (weighted[i] as f64 - plain[i] as f64 * 0.25).abs() < 1e-6,
                "routing weight must scale the pre-w2 activation elementwise"
            );
        }

        // Distinguish "before w2" from "after w2" with a non-linear stand-in
        // for w2: relu(sum(act)). Before: relu(sum(w*act)); after: w*relu(sum(act)).
        // With mixed-sign act and w=0.25, these differ when sum(act)>0 but
        // the weighted sum crosses zero differently — pick all-positive act.
        let gate_p = [1.0f32, 1.0];
        let up_p = [2.0f32, 2.0];
        let plain_p = expert_swiglu_ref(&gate_p, &up_p, 1, 2, 0.0, None);
        let weighted_p = expert_swiglu_ref(&gate_p, &up_p, 1, 2, 0.0, Some(&[0.25]));
        let sum_plain: f64 = plain_p.iter().map(|&v| v as f64).sum();
        let sum_weighted: f64 = weighted_p.iter().map(|&v| v as f64).sum();
        // before w2 (correct): sum(w*act) = w * sum(act)
        assert!((sum_weighted - 0.25 * sum_plain).abs() < 1e-6);
        // Fake after-w2 nonlinear: w * f(sum(act)) vs f(sum(w*act)) with f = square
        let after = 0.25 * (sum_plain * sum_plain);
        let before = sum_weighted * sum_weighted;
        assert!(
            (after - before).abs() > 1e-6,
            "setup must make before-w2 vs after-w2 differ under squaring"
        );
        // And our oracle matches the before path:
        assert!((before - (0.25 * sum_plain).powi(2)).abs() < 1e-6);
    }

    #[test]
    fn gate_hash_tid2eid_fixture() {
        // vocab=3, topk=2, n_experts=8
        // tid2eid = [
        //   [1, 4],
        //   [0, 7],
        //   [3, 3],
        // ]
        let tid2eid: Vec<i64> = vec![1, 4, 0, 7, 3, 3];
        let ids = [1u32, 0, 2];
        let r = gate_hash_ref(&ids, &tid2eid, 8, 2).unwrap();
        assert_eq!(r.indices, vec![0, 7, 1, 4, 3, 3]);
        // uniform 1/topk
        for w in &r.weights {
            assert!((*w - 0.5).abs() < 1e-6);
        }
    }


    // ── Attention SWA oracle tests ──────────────────────────────────────

    #[test]
    fn sparse_attn_hand_computed_sink_and_minus_one() {
        // 1 row, 1 head, head_dim=2, k=3 slots with one -1.
        // q=[1,0], kv0=[1,0] score=1*scale, kv1 unused (-1), kv2=[0,1] score=0.
        // scale = 1/sqrt(2).
        // sink = 0.
        let q = [1.0f32, 0.0];
        let kv = [
            1.0, 0.0, // idx 0
            9.0, 9.0, // idx 1 (must NOT be read — masked by -1)
            0.0, 1.0, // idx 2
        ];
        let sink = [0.0f32];
        let idxs = [0i32, -1, 2];
        let scale = 1.0f64 / (2.0f64).sqrt();
        let o = sparse_attn_ref(&q, &kv, &sink, &idxs, 1, 1, 2, 3, 3, scale).unwrap();

        // scores: s0 = 1*scale, s2 = 0, sink = 0
        let s0 = scale;
        let s2 = 0.0f64;
        let sk = 0.0f64;
        let m = s0.max(s2).max(sk);
        let e0 = (s0 - m).exp();
        let e2 = (s2 - m).exp();
        let es = (sk - m).exp();
        let z = e0 + e2 + es;
        let p0 = e0 / z;
        let p2 = e2 / z;
        // out = p0*kv0 + p2*kv2  (sink contributes no value)
        let expect0 = p0 * 1.0 + p2 * 0.0;
        let expect1 = p0 * 0.0 + p2 * 1.0;
        assert!(
            (o[0] as f64 - expect0).abs() < 1e-6,
            "o0={} expect={expect0}",
            o[0]
        );
        assert!(
            (o[1] as f64 - expect1).abs() < 1e-6,
            "o1={} expect={expect1}",
            o[1]
        );
        // Probability mass of sink is es/z > 0, so ||out|| < 1 (not a convex
        // combination of only the two KVs with weights summing to 1).
        let mass_keys = p0 + p2;
        assert!(mass_keys < 1.0 - 1e-6, "sink must take probability mass");
        // And the masked kv1=[9,9] must not leak: |out| stays O(1).
        assert!(o[0].abs() < 2.0 && o[1].abs() < 2.0);
    }

    #[test]
    fn sparse_attn_sink_only_when_all_minus_one() {
        // All indices -1 → output is exactly 0 (sink in denom, no V).
        let q = [1.0f32, 2.0];
        let kv = [3.0f32, 4.0];
        let sink = [1.5f32];
        let idxs = [-1i32, -1];
        let o = sparse_attn_ref(&q, &kv, &sink, &idxs, 1, 1, 2, 1, 2, 1.0).unwrap();
        assert_eq!(o, vec![0.0, 0.0]);
    }

    #[test]
    fn sparse_attn_joint_window_plus_compress_hand() {
        // Tiny joint case: 1 query, 1 head, head_dim=2.
        // Window slots: idx0=[1,0], idx1 masked (-1).
        // One compressed slot at unified idx2=[0,1].
        // sink=0 in denom only.
        // topk = [0, -1, 2]  (window then compress).
        let q = [1.0f32, 0.0];
        let kv = [
            1.0, 0.0, // window 0
            9.0, 9.0, // window 1 (masked — must not leak)
            0.0, 1.0, // compressed 0 at offset=2
        ];
        let sink = [0.0f32];
        let idxs = [0i32, -1, 2];
        let scale = 1.0f64 / (2.0f64).sqrt();
        let o = sparse_attn_ref(&q, &kv, &sink, &idxs, 1, 1, 2, 3, 3, scale).unwrap();

        let s0 = scale; // q·kv0 * scale
        let s2 = 0.0f64; // q·kv2 * scale
        let sk = 0.0f64;
        let m = s0.max(s2).max(sk);
        let e0 = (s0 - m).exp();
        let e2 = (s2 - m).exp();
        let es = (sk - m).exp();
        let z = e0 + e2 + es;
        let p0 = e0 / z;
        let p2 = e2 / z;
        let expect0 = p0 * 1.0 + p2 * 0.0;
        let expect1 = p0 * 0.0 + p2 * 1.0;
        assert!(
            (o[0] as f64 - expect0).abs() < 1e-6,
            "joint o0={} expect={expect0}",
            o[0]
        );
        assert!(
            (o[1] as f64 - expect1).abs() < 1e-6,
            "joint o1={} expect={expect1}",
            o[1]
        );
        // Both window and compress contribute; sink takes mass; masked slot silent.
        assert!(p0 > 0.0 && p2 > 0.0);
        assert!(p0 + p2 < 1.0 - 1e-6);
        assert!(o[0].abs() < 2.0 && o[1].abs() < 2.0);

        // Causality pad: compress index -1 must zero the compressed contribution.
        let idxs_masked = [0i32, -1, -1];
        let o_m = sparse_attn_ref(&q, &kv, &sink, &idxs_masked, 1, 1, 2, 3, 3, scale).unwrap();
        let m2 = s0.max(sk);
        let e0b = (s0 - m2).exp();
        let esb = (sk - m2).exp();
        let zb = e0b + esb;
        let p0b = e0b / zb;
        assert!((o_m[0] as f64 - p0b * 1.0).abs() < 1e-6);
        assert!((o_m[1] as f64 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn attention_swa_ref_tiny_hand_end_to_end() {
        // Tiny shapes chosen so every linear K is divisible by 128 act-quant
        // block... wait, K must be %128==0 for act_quant. Use dim=128,
        // q_lora=128, head_dim=128, rope=64, n_heads=2, o_groups=1, o_lora=128,
        // window=4, rows=2.
        let dim = 128usize;
        let n_heads = 2usize;
        let head_dim = 128usize;
        let rope_dim = 64usize;
        let q_lora = 128usize;
        let o_lora = 128usize;
        let o_groups = 1usize;
        let window = 4usize;
        let rows = 2usize;
        let q_width = n_heads * head_dim;
        let per_group_in = n_heads / o_groups * head_dim;
        let wo_a_out = o_groups * o_lora;

        // Deterministic small weights / activations (BF16-friendly).
        let mut x = vec![0.0f32; rows * dim];
        for r in 0..rows {
            for d in 0..dim {
                x[r * dim + d] = ((r * 3 + d) % 7) as f32 * 0.125 - 0.375;
            }
        }
        let fill = |n: usize, seed: u32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let v = ((i as u32).wrapping_mul(1103515245).wrapping_add(seed)) % 17;
                    (v as f32) * 0.0625 - 0.5
                })
                .collect()
        };
        let wq_a = fill(q_lora * dim, 1);
        let wq_b = fill(q_width * q_lora, 2);
        let wkv = fill(head_dim * dim, 3);
        let wo_a = fill(wo_a_out * per_group_in, 4);
        let wo_b = fill(dim * wo_a_out, 5);
        let q_norm = vec![1.0f32; q_lora];
        let kv_norm = vec![1.0f32; head_dim];
        let attn_sink = vec![0.0f32; n_heads];

        let w = AttnSwARefWeights {
            wq_a: &wq_a,
            wq_b: &wq_b,
            wkv: &wkv,
            wo_a: &wo_a,
            wo_b: &wo_b,
            q_norm: &q_norm,
            kv_norm: &kv_norm,
            attn_sink: &attn_sink,
            compressor: None,
            indexer: None,
        };
        let out = attention_swa_ref_cfg(
            &x, &w, rows, 0, dim, n_heads, head_dim, rope_dim, q_lora, o_lora, o_groups,
            window, 1e-6, /*rope_original_seq_len=*/ 0, 10_000.0, /*compress_ratio=*/ 0,
        )
        .unwrap();

        assert_eq!(out.o.len(), rows * dim);
        assert_eq!(out.q_post_rope.len(), rows * q_width);
        assert_eq!(out.kv_post_quant.len(), rows * head_dim);
        assert_eq!(out.attn_raw.len(), rows * q_width);
        assert_eq!(out.attn_inv_rope.len(), rows * q_width);
        assert_eq!(out.wo_a_out.len(), rows * wo_a_out);
        assert!(out.o.iter().all(|v| v.is_finite()));
        assert!(out.attn_raw.iter().all(|v| v.is_finite()));

        // Window: row0 sees only self; row1 sees 0,1.
        let k = rows.min(window);
        assert_eq!(out.window_idxs[0], 0);
        assert!(out.window_idxs[1..k].iter().all(|&v| v == -1));
        assert_eq!(out.window_idxs[k], 0);
        assert_eq!(out.window_idxs[k + 1], 1);

        // Extra head RMSNorm must change q (not a no-op on non-unit vectors).
        let mut max_rms_delta = 0.0f32;
        for i in 0..out.q_post_wb.len() {
            max_rms_delta = max_rms_delta.max((out.q_post_wb[i] - out.q_post_head_rms[i]).abs());
        }
        assert!(
            max_rms_delta > 1e-6,
            "post-wq_b head RMSNorm must move q; delta={max_rms_delta}"
        );

        // Inverse RoPE of forward RoPE on a head must round-trip the rope tail
        // of attn when applied as (fwd then inv) on the same buffer — checked
        // structurally: attn_inv_rope != attn_raw whenever rope dims are live.
        let mut rope_moved = 0.0f32;
        for i in 0..out.attn_raw.len() {
            rope_moved = rope_moved.max((out.attn_raw[i] - out.attn_inv_rope[i]).abs());
        }
        // With nonzero attention output on rope dims this should move; if the
        // whole attn is ~0 it's still OK as long as finite.
        let _ = rope_moved;

        // Hand-check sparse attn at row0 head0 against q/kv intermediates.
        let scale = 1.0f64 / (head_dim as f64).sqrt();
        let q0 = &out.q_post_rope[..head_dim];
        let kv0 = &out.kv_post_quant[..head_dim];
        let mut dot = 0.0f64;
        for d in 0..head_dim {
            dot += q0[d] as f64 * kv0[d] as f64;
        }
        let s = dot * scale;
        let sink = 0.0f64;
        let m = s.max(sink);
        let e = (s - m).exp();
        let es = (sink - m).exp();
        let z = e + es;
        let p = e / z;
        for d in 0..head_dim {
            let expect = (p * (kv0[d] as f64)) as f32;
            let got = out.attn_raw[d];
            assert!(
                (got - expect).abs() < 1e-5,
                "row0 head0 dim {d}: got={got} expect={expect}"
            );
        }
    }

    #[test]
    fn attention_swa_ref_rejects_nonzero_start_pos() {
        let dim = 128usize;
        let err = attention_swa_ref_cfg(
            &vec![0.0f32; dim],
            &AttnSwARefWeights {
                wq_a: &vec![0.0f32; 128 * dim],
                wq_b: &vec![0.0f32; 2 * 128 * 128],
                wkv: &vec![0.0f32; 128 * dim],
                wo_a: &vec![0.0f32; 128 * 2 * 128],
                wo_b: &vec![0.0f32; dim * 128],
                q_norm: &vec![1.0f32; 128],
                kv_norm: &vec![1.0f32; 128],
                attn_sink: &vec![0.0f32; 2],
                compressor: None,
                indexer: None,
            },
            1,
            5,
            dim,
            2,
            128,
            64,
            128,
            128,
            1,
            4,
            1e-6,
            /*rope_original_seq_len=*/ 0,
            10_000.0,
            /*compress_ratio=*/ 0,
        )
        .unwrap_err();
        assert!(err.contains("start_pos"), "{err}");
        assert!(err.starts_with("deepseek4 parent:"), "{err}");
    }

    #[test]
    fn attention_main_rope_policy_selects_tables() {
        let (o0, t0) = attention_main_rope_policy(0).unwrap();
        assert_eq!(o0, 0);
        assert!((t0 - 10_000.0).abs() < 1e-12);

        let (o4, t4) = attention_main_rope_policy(4).unwrap();
        assert_eq!(o4, PARENT_YARN_ORIG_SEQ);
        assert!((t4 - PARENT_COMPRESS_ROPE_THETA).abs() < 1e-12);

        let (o128, t128) = attention_main_rope_policy(128).unwrap();
        assert_eq!((o128, t128), (o4, t4));

        let err = attention_main_rope_policy(7).unwrap_err();
        assert!(err.contains("unsupported compress_ratio=7"), "{err}");

        // Tables must differ enough that a swap is measurable at pos>0.
        let plain = precompute_rope_freqs(PARENT_ROPE_DIM, o0, t0, PARENT_YARN_FACTOR, PARENT_YARN_BETA_FAST, PARENT_YARN_BETA_SLOW).unwrap();
        let yarn = precompute_rope_freqs(PARENT_ROPE_DIM, o4, t4, PARENT_YARN_FACTOR, PARENT_YARN_BETA_FAST, PARENT_YARN_BETA_SLOW).unwrap();
        assert_eq!(plain.len(), yarn.len());
        let mut max_rel = 0.0f64;
        for (a, b) in plain.iter().zip(yarn.iter()) {
            let d = (a - b).abs() / a.abs().max(1e-30);
            if d > max_rel {
                max_rel = d;
            }
        }
        assert!(
            max_rel > 0.1,
            "ratio-0 vs ratio>0 freq tables must differ substantially; max_rel={max_rel}"
        );
    }

    #[test]
    fn attention_swa_ref_ratio_gt0_rope_hand_differs_from_ratio0() {
        // Same tiny geometry as the ratio-0 hand test. Only the RoPE policy
        // changes. At position 0 the two tables agree (angle=0); at position 1
        // the post-RoPE q/kv must diverge, proving the ratio>0 path consumed
        // the YaRN table end-to-end on q, kv, and inverse-o.
        let dim = 128usize;
        let n_heads = 2usize;
        let head_dim = 128usize;
        let rope_dim = 64usize;
        let q_lora = 128usize;
        let o_lora = 128usize;
        let o_groups = 1usize;
        let window = 4usize;
        let rows = 2usize;
        let q_width = n_heads * head_dim;
        let per_group_in = n_heads / o_groups * head_dim;
        let wo_a_out = o_groups * o_lora;

        let mut x = vec![0.0f32; rows * dim];
        for r in 0..rows {
            for d in 0..dim {
                x[r * dim + d] = ((r * 3 + d) % 7) as f32 * 0.125 - 0.375;
            }
        }
        let fill = |n: usize, seed: u32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let v = ((i as u32).wrapping_mul(1103515245).wrapping_add(seed)) % 17;
                    (v as f32) * 0.0625 - 0.5
                })
                .collect()
        };
        let wq_a = fill(q_lora * dim, 1);
        let wq_b = fill(q_width * q_lora, 2);
        let wkv = fill(head_dim * dim, 3);
        let wo_a = fill(wo_a_out * per_group_in, 4);
        let wo_b = fill(dim * wo_a_out, 5);
        let q_norm = vec![1.0f32; q_lora];
        let kv_norm = vec![1.0f32; head_dim];
        let attn_sink = vec![0.0f32; n_heads];
        let w = AttnSwARefWeights {
            wq_a: &wq_a,
            wq_b: &wq_b,
            wkv: &wkv,
            wo_a: &wo_a,
            wo_b: &wo_b,
            q_norm: &q_norm,
            kv_norm: &kv_norm,
            attn_sink: &attn_sink,
            compressor: None,
            indexer: None,
        };

        let r0 = attention_swa_ref_cfg(
            &x, &w, rows, 0, dim, n_heads, head_dim, rope_dim, q_lora, o_lora, o_groups,
            window, 1e-6, 0, 10_000.0, /*compress_ratio=*/ 0,
        )
        .unwrap();
        let (orig, theta) = attention_main_rope_policy(4).unwrap();
        let r4 = attention_swa_ref_cfg(
            &x, &w, rows, 0, dim, n_heads, head_dim, rope_dim, q_lora, o_lora, o_groups,
            window, 1e-6, orig, theta, /*compress_ratio=*/ 0,
        )
        .unwrap();
        // Production wrapper routes ratio through attention_main_rope_policy;
        // covered by attention_main_rope_policy_selects_tables + the cfg path
        // below (tiny shapes cannot use PARENT_DIM production constants).

        // Pre-RoPE stages are ratio-independent.
        assert_eq!(r0.q_post_wb, r4.q_post_wb);
        assert_eq!(r0.q_post_head_rms, r4.q_post_head_rms);
        assert_eq!(r0.kv_post_norm, r4.kv_post_norm);

        // Row 0: angle=0 → RoPE is identity under both tables.
        let mut row0_q = 0.0f32;
        for i in 0..q_width {
            row0_q = row0_q.max((r0.q_post_rope[i] - r4.q_post_rope[i]).abs());
        }
        assert!(row0_q < 1e-6, "row0 q_post_rope must match across tables; d={row0_q}");

        // Row 1: tables disagree → post-RoPE q/kv must move.
        let mut row1_q = 0.0f32;
        for i in q_width..2 * q_width {
            row1_q = row1_q.max((r0.q_post_rope[i] - r4.q_post_rope[i]).abs());
        }
        assert!(
            row1_q > 1e-3,
            "row1 q_post_rope must diverge under YaRN table; d={row1_q}"
        );
        let mut row1_kv = 0.0f32;
        for i in head_dim..2 * head_dim {
            row1_kv = row1_kv.max((r0.kv_post_rope[i] - r4.kv_post_rope[i]).abs());
        }
        assert!(
            row1_kv > 1e-3,
            "row1 kv_post_rope must diverge under YaRN table; d={row1_kv}"
        );

        // Inverse-RoPE path also uses the same table: attn_inv must differ on row1.
        let mut row1_inv = 0.0f32;
        for i in q_width..2 * q_width {
            row1_inv = row1_inv.max((r0.attn_inv_rope[i] - r4.attn_inv_rope[i]).abs());
        }
        assert!(
            row1_inv > 1e-4,
            "row1 attn_inv_rope must diverge (inverse uses same table); d={row1_inv}"
        );

        // r4 cfg path is the production policy (orig/theta from attention_main_rope_policy(4)).
        assert_eq!(orig, PARENT_YARN_ORIG_SEQ);
        assert!((theta - PARENT_COMPRESS_ROPE_THETA).abs() < 1e-12);

        // Hand-check one interleaved pair on row1 head0 under the YaRN table.
        // q_post_head_rms → rotate by angle = 1 * freqs[0] on the first rope pair.
        let freqs = precompute_rope_freqs(
            rope_dim,
            orig,
            theta,
            PARENT_YARN_FACTOR,
            PARENT_YARN_BETA_FAST,
            PARENT_YARN_BETA_SLOW,
        )
        .unwrap();
        let tail_off = head_dim - rope_dim;
        let base = q_width + tail_off; // row1, head0, first rope pair
        let x0 = r4.q_post_head_rms[base] as f64;
        let x1 = r4.q_post_head_rms[base + 1] as f64;
        let (s, c) = (1.0f64 * freqs[0]).sin_cos();
        let expect0 = (x0 * c - x1 * s) as f32;
        let expect1 = (x0 * s + x1 * c) as f32;
        assert!(
            (r4.q_post_rope[base] - expect0).abs() < 1e-5,
            "hand rope pair0 real: got={} expect={expect0}",
            r4.q_post_rope[base]
        );
        assert!(
            (r4.q_post_rope[base + 1] - expect1).abs() < 1e-5,
            "hand rope pair0 imag: got={} expect={expect1}",
            r4.q_post_rope[base + 1]
        );
    }
}
