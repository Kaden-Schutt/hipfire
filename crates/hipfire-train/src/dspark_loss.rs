// SPDX-License-Identifier: Apache-2.0
//! DSpark drafter training loss — forward value + backward gradients.
//!
//! Authoritative reference:
//! `third_party/dspark/deepspec/modeling/dspark/loss.py`
//! (`compute_dspark_loss` / `_collect_local_terms`). This is the single-process
//! (`world_size == 1`) case: no distributed all-reduce of the denominators, so
//! the local and global denominators coincide and the `world_size` scale is 1.
//!
//! Per draft position we combine three terms, each a weighted mean over valid
//! positions with the SAME weight mask `w`:
//!
//! * **CE** — hard cross-entropy of the draft logits against the target next
//!   token. Reuses the fused [`crate::ops::cross_entropy`] op, whose per-row
//!   `d_logits = softmax(draft) − onehot(target)` is exactly the CE gradient
//!   building block.
//! * **L1 / TV** — L1 distance between `softmax(draft)` and `softmax(target)`.
//!   Reuses [`crate::ops::softmax`] for both distributions.
//! * **confidence** — BCE-with-logits of the confidence head against a DETACHED
//!   accept-rate target `ar = clamp(1 − 0.5·l1_per_tok, 0, 1)`.
//!
//! The GPU is used for the two heavy per-row primitives (cross-entropy, softmax)
//! via the existing training ops; the cheap per-row reductions (sign, softmax-of
//! the L1 gradient, BCE, the weighted numerators/denominators) run on the host
//! in f32, then the assembled gradients are uploaded. This keeps the op simple
//! and obviously correct — no new device kernel is required (the row-reduction
//! primitives are composed on the host). Correctness, not throughput, is the
//! goal: it runs once per training micro-step over `[N·block, vocab]`.

use crate::ops::cross_entropy::cross_entropy;
use crate::ops::softmax::softmax_forward;
use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};

/// DSpark loss hyper-parameters. Defaults match the reference config
/// (`ce_alpha=0.1`, `l1_alpha=0.9`, `confidence_alpha=1.0`,
/// `loss_decay_gamma=4.0`). `block_size` is the number of draft positions per
/// block and must be set to the drafter's block size so the positional decay
/// `exp(-pos/gamma)` is computed against the correct within-block position.
#[derive(Clone, Copy, Debug)]
pub struct DsparkLossCfg {
    /// Draft positions per block (row `r` has within-block position
    /// `r % block_size`).
    pub block_size: usize,
    /// CE term weight.
    pub ce_alpha: f32,
    /// L1/TV term weight.
    pub l1_alpha: f32,
    /// Confidence (BCE) term weight.
    pub confidence_alpha: f32,
    /// Positional decay `w *= exp(-pos/gamma)`; `<= 0` disables the decay.
    pub loss_decay_gamma: f32,
}

impl Default for DsparkLossCfg {
    fn default() -> Self {
        Self {
            block_size: 4,
            ce_alpha: 0.1,
            l1_alpha: 0.9,
            confidence_alpha: 1.0,
            loss_decay_gamma: 4.0,
        }
    }
}

impl DsparkLossCfg {
    /// Reference defaults with an explicit `block_size`.
    pub fn with_block_size(block_size: usize) -> Self {
        Self {
            block_size,
            ..Self::default()
        }
    }
}

/// Forward value (+ per-term logging scalars) and the two gradients the drafter
/// backward consumes.
pub struct DsparkLossOut {
    /// `ce_alpha·ce + l1_alpha·l1 + confidence_alpha·conf`.
    pub total: f32,
    /// Normalized CE term `ce_num / ce_den` (before `ce_alpha`), for logging.
    pub ce: f32,
    /// Normalized L1 term `l1_num / l1_den` (before `l1_alpha`), for logging.
    pub l1: f32,
    /// Normalized confidence term `conf_num / conf_den` (before
    /// `confidence_alpha`), for logging.
    pub conf: f32,
    /// `d(total)/d(draft_logits)`, shape `[rows, vocab]` flattened to
    /// `[rows*vocab]`.
    pub d_draft_logits: GpuTensor,
    /// `d(total)/d(confidence_logit)`, shape `[rows]`.
    pub d_confidence_logit: GpuTensor,
}

/// Numerically stable `sigmoid`.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `BCE_with_logits(x, t) = max(x,0) − x·t + ln(1 + e^{-|x|})` (stable form).
#[inline]
fn bce_with_logits(x: f32, t: f32) -> f32 {
    x.max(0.0) - x * t + (1.0 + (-x.abs()).exp()).ln()
}

/// Small denominator guard, matching `loss.py`'s `den + 1e-6`.
const DEN_EPS: f32 = 1e-6;

/// Compute the DSpark training loss and its gradients for a whole batch of
/// blocks flattened to `rows = N·block_size` positions.
///
/// * `draft_logits` `[rows, vocab]` — the drafter's per-position logits.
/// * `confidence_logit` `[rows]` — pre-sigmoid confidence head output.
/// * `target_logits` `[rows, vocab]` — teacher soft logits (for the L1 term).
/// * `target_next_tokens` `[rows]` — hard next-token class ids (f32), `-100`
///   marks ignored/padding positions (see `ignore_index`).
/// * `eval_mask` `[rows]` — `1.0` for valid positions, `0.0` otherwise.
///
/// Grad formulas (den's are constant in the logits, being sums of `w`):
/// * `d_draft += ce_alpha·w/ce_den · (softmax(draft) − onehot(target))`.
/// * `d_draft += l1_alpha·w/l1_den · pd·(sign − Σ_u pd_u·sign_u)` with
///   `sign = sign(pd − pt)` and `pd = softmax(draft)`.
/// * `d_confidence_logit = confidence_alpha·w/conf_den · (sigmoid(logit) − ar)`,
///   `ar` DETACHED (contributes no draft-logit gradient).
pub fn dspark_loss_forward_backward(
    gpu: &mut Gpu,
    draft_logits: &GpuTensor,
    confidence_logit: &GpuTensor,
    target_logits: &GpuTensor,
    target_next_tokens: &GpuTensor,
    eval_mask: &GpuTensor,
    cfg: &DsparkLossCfg,
) -> HipResult<DsparkLossOut> {
    let rows = eval_mask.numel();
    assert!(rows > 0, "dspark_loss: empty batch");
    let total_elems = draft_logits.numel();
    assert_eq!(
        total_elems % rows,
        0,
        "dspark_loss: draft_logits {total_elems} not divisible by rows {rows}"
    );
    let v = total_elems / rows;
    assert_eq!(
        target_logits.numel(),
        total_elems,
        "dspark_loss: target_logits shape mismatch"
    );
    assert!(cfg.block_size > 0, "dspark_loss: block_size must be > 0");

    // ---- weight mask w[r] = eval_mask[r] * exp(-(r % block_size)/gamma) ------
    let eval_host = gpu.download_f32(eval_mask)?;
    let gamma = cfg.loss_decay_gamma;
    let decay: Vec<f32> = (0..cfg.block_size)
        .map(|pos| {
            if gamma > 0.0 {
                (-(pos as f32) / gamma).exp()
            } else {
                1.0
            }
        })
        .collect();
    let mut w = vec![0.0f32; rows];
    for r in 0..rows {
        w[r] = eval_host[r] * decay[r % cfg.block_size];
    }
    let w_sum: f32 = w.iter().sum();
    // All three terms share the same weight mask, so all dens equal w_sum.
    let ce_den = w_sum;
    let l1_den = w_sum;
    let conf_den = w_sum;

    // ---- CE term (reuse fused cross_entropy: loss + softmax−onehot grad) ----
    // `ignore_index = -100` matches F.cross_entropy's default in loss.py.
    let ce_loss = gpu.zeros(&[rows], DType::F32)?;
    let ce_dlog = gpu.zeros(&[total_elems], DType::F32)?;
    cross_entropy(
        gpu,
        draft_logits,
        target_next_tokens,
        &ce_loss,
        &ce_dlog,
        rows,
        v,
        -100,
    )?;
    let ce_per_tok = gpu.download_f32(&ce_loss)?; // [rows]
    let ce_grad_raw = gpu.download_f32(&ce_dlog)?; // [rows*v], = softmax(draft) − onehot

    // ---- pd = softmax(draft), pt = softmax(target) --------------------------
    let pd_t = gpu.zeros(&[total_elems], DType::F32)?;
    let pt_t = gpu.zeros(&[total_elems], DType::F32)?;
    softmax_forward(gpu, draft_logits, &pd_t, rows, v)?;
    softmax_forward(gpu, target_logits, &pt_t, rows, v)?;
    let pd = gpu.download_f32(&pd_t)?;
    let pt = gpu.download_f32(&pt_t)?;

    let conf_logit_host = gpu.download_f32(confidence_logit)?; // [rows]

    // ---- per-row reductions + gradient assembly (host, f32) -----------------
    let mut ce_num = 0.0f32;
    let mut l1_num = 0.0f32;
    let mut conf_num = 0.0f32;

    let mut d_draft = vec![0.0f32; total_elems];
    let mut d_conf = vec![0.0f32; rows];

    let ce_scale_den = ce_den + DEN_EPS;
    let l1_scale_den = l1_den + DEN_EPS;
    let conf_scale_den = conf_den + DEN_EPS;

    for r in 0..rows {
        let wr = w[r];
        let base = r * v;

        // L1 per-token distance and its softmax-gradient dot product.
        // l1_per_tok = Σ_v |pd − pt|;  dot = Σ_v pd_v · sign_v.
        let mut l1_per_tok = 0.0f32;
        let mut dot = 0.0f32;
        for j in 0..v {
            let diff = pd[base + j] - pt[base + j];
            l1_per_tok += diff.abs();
            let sign = if diff > 0.0 {
                1.0
            } else if diff < 0.0 {
                -1.0
            } else {
                0.0
            };
            dot += pd[base + j] * sign;
        }

        // accept-rate target (DETACHED — no draft-logit gradient).
        let ar = (1.0 - 0.5 * l1_per_tok).clamp(0.0, 1.0);

        // numerators
        ce_num += ce_per_tok[r] * wr;
        l1_num += l1_per_tok * wr;
        conf_num += bce_with_logits(conf_logit_host[r], ar) * wr;

        // ---- gradients ----
        let ce_row_scale = cfg.ce_alpha * wr / ce_scale_den;
        let l1_row_scale = cfg.l1_alpha * wr / l1_scale_den;
        for j in 0..v {
            // CE grad: scale (softmax − onehot).
            let mut g = ce_row_scale * ce_grad_raw[base + j];
            // L1/TV grad through softmax: pd_v·(sign_v − Σ_u pd_u·sign_u).
            let diff = pd[base + j] - pt[base + j];
            let sign = if diff > 0.0 {
                1.0
            } else if diff < 0.0 {
                -1.0
            } else {
                0.0
            };
            g += l1_row_scale * pd[base + j] * (sign - dot);
            d_draft[base + j] = g;
        }

        // Confidence grad: standard BCE-with-logits, ar detached.
        d_conf[r] = cfg.confidence_alpha * wr / conf_scale_den * (sigmoid(conf_logit_host[r]) - ar);
    }

    let ce = ce_num / ce_scale_den;
    let l1 = l1_num / l1_scale_den;
    let conf = conf_num / conf_scale_den;
    let total = cfg.ce_alpha * ce + cfg.l1_alpha * l1 + cfg.confidence_alpha * conf;

    let d_draft_logits = gpu.upload_f32(&d_draft, &[rows, v])?;
    let d_confidence_logit = gpu.upload_f32(&d_conf, &[rows])?;

    Ok(DsparkLossOut {
        total,
        ce,
        l1,
        conf,
        d_draft_logits,
        d_confidence_logit,
    })
}
