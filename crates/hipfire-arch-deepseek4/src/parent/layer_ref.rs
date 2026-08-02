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
/// `y[r,h,d] = post[r,h] * x[r,d] + sum_k comb[r,h,k] * residual[r,k,d]`
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
                    let c = comb[r * hc_mult * hc_mult + h * hc_mult + k] as f64;
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
}
