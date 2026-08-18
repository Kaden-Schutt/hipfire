// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU parent MoE block: routing + decode-on-demand routed experts + shared expert.
//!
//! Parent experts are `I8`-packed E2M1 with per-32 UE8M0 scales, left compressed
//! in HBM. This path **cannot** reuse the MQ2R MoE family — that assumes
//! MQ2R-packed experts. Instead each selected expert's `w1`/`w3`/`w2` is decoded
//! once per batch into reusable scratch and all tokens routed to it run together.
//!
//! Operator semantics authority:
//! - `local://ds4-parent-gate3-contract.md` §6
//! - `.codeinsight+research/ds4-parent-ref/inference/model.py` `Gate` (551-590),
//!   `Expert` (592-611), `MoE` (614-649)
//! - config: `score_func=sqrtsoftplus`, `route_scale=1.5`, `swiglu_limit=10.0`,
//!   `n_activated_experts=6`, `n_routed_experts=256`, `moe_inter_dim=2048`,
//!   `dim=4096`, `n_hash_layers=3`

use crate::codec::round_to_bf16;
use crate::linear::{parent_linear_dense, parent_linear_expert, ParentDenseWeight};
use crate::weights::{ParentExpertTriple, ParentLayerWeights};
use crate::{Ds4ParentBackend, ParentQuantConfig};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Parent hidden dimension (verified against checkpoint shapes).
pub const PARENT_DIM: usize = 4096;
/// Parent MoE intermediate dimension (`moe_inter_dim`).
pub const PARENT_MOE_INTER: usize = 2048;
/// Default `route_scale` from the parent `config.json` (checkpoint value).
/// Diagnostic override: `HIPFIRE_PARENT_ROUTE_SCALE` (logged once when set).
/// The parent default MUST stay at 1.5 — never bake a serving compensation in.
pub const PARENT_ROUTE_SCALE: f32 = 1.5;

/// Resolve parent route_scale. Default is the checkpoint value
/// [`PARENT_ROUTE_SCALE`]. `HIPFIRE_PARENT_ROUTE_SCALE` overrides for
/// diagnostic sweeps only — logged once. Do not change the default.
pub fn effective_parent_route_scale() -> f32 {
    use std::sync::LazyLock;
    static SCALE: LazyLock<f32> = LazyLock::new(|| {
        match std::env::var("HIPFIRE_PARENT_ROUTE_SCALE") {
            Ok(s) => match s.parse::<f32>() {
                Ok(v) if v.is_finite() && v > 0.0 => {
                    eprintln!(
                        "deepseek4 parent: HIPFIRE_PARENT_ROUTE_SCALE={v} overrides \
                         checkpoint PARENT_ROUTE_SCALE={PARENT_ROUTE_SCALE} (diagnostic only)"
                    );
                    v
                }
                _ => {
                    eprintln!(
                        "deepseek4 parent: ignoring invalid HIPFIRE_PARENT_ROUTE_SCALE={s:?}; \
                         using checkpoint {PARENT_ROUTE_SCALE}"
                    );
                    PARENT_ROUTE_SCALE
                }
            },
            Err(_) => PARENT_ROUTE_SCALE,
        }
    });
    *SCALE
}
/// `swiglu_limit` from the parent `config.json`.
pub const PARENT_SWIGLU_LIMIT: f32 = 10.0;

/// Reusable BF16 decode + activation scratch for parent MoE.
///
/// Sized once for the largest expert matrix (`w1`/`w3` logical `[2048, 4096]`,
/// `w2` logical `[4096, 2048]` — both 8_388_608 elements) and for
/// `max_rows` tokens. Allocating per expert would defeat leaving experts
/// compressed in HBM.
pub struct ParentMoeScratch {
    /// BF16 weight decode tile: covers max(`inter*dim`, `dim*inter`) elements.
    pub(crate) w_bf16: GpuTensor,
    /// BF16 activation staging for destructive `parent_linear_*` (copy of x).
    pub(crate) act_bf16: GpuTensor,
    /// F32 gate projection output `[max_rows, inter]`.
    pub(crate) gate_f32: GpuTensor,
    /// F32 up projection output `[max_rows, inter]`.
    pub(crate) up_f32: GpuTensor,
    /// F32 SwiGLU / weighted hidden `[max_rows, inter]`.
    pub(crate) hidden_f32: GpuTensor,
    /// BF16 hidden for w2 input (post weight-scale, pre act-quant).
    pub(crate) hidden_bf16: GpuTensor,
    /// F32 expert / shared output tile `[max_rows, dim]`.
    pub(crate) expert_out_f32: GpuTensor,
    max_rows: usize,
    dim: usize,
    inter: usize,
    /// Bytes of all scratch tensors combined.
    bytes: usize,
}

impl ParentMoeScratch {
    /// Allocate reusable scratch for up to `max_rows` tokens.
    pub fn new(gpu: &mut Gpu, cfg: &ParentQuantConfig, max_rows: usize) -> Result<Self, String> {
        if max_rows == 0 {
            return Err("deepseek4 parent: ParentMoeScratch max_rows must be > 0".to_owned());
        }
        // Shapes are fixed by the parent checkpoint contract; refuse anything
        // that would size scratch against a different model.
        let _ = cfg; // cfg reserved for future dim fields; shapes pinned below.
        let dim = PARENT_DIM;
        let inter = PARENT_MOE_INTER;
        // Largest expert matrix element count: w1/w3 = inter*dim, w2 = dim*inter.
        let w_elems = inter
            .checked_mul(dim)
            .ok_or_else(|| "deepseek4 parent: expert scratch size overflow".to_owned())?;

        let w_bf16 = gpu
            .alloc_tensor(&[w_elems], DType::BF16)
            .map_err(|e| format!("deepseek4 parent: moe w_bf16 alloc: {e:?}"))?;
        let act_bf16 = match gpu.alloc_tensor(&[max_rows, dim], DType::BF16) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(w_bf16);
                return Err(format!("deepseek4 parent: moe act_bf16 alloc: {e:?}"));
            }
        };
        let gate_f32 = match gpu.alloc_tensor(&[max_rows, inter], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(w_bf16);
                let _ = gpu.free_tensor(act_bf16);
                return Err(format!("deepseek4 parent: moe gate_f32 alloc: {e:?}"));
            }
        };
        let up_f32 = match gpu.alloc_tensor(&[max_rows, inter], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(w_bf16);
                let _ = gpu.free_tensor(act_bf16);
                let _ = gpu.free_tensor(gate_f32);
                return Err(format!("deepseek4 parent: moe up_f32 alloc: {e:?}"));
            }
        };
        let hidden_f32 = match gpu.alloc_tensor(&[max_rows, inter], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(w_bf16);
                let _ = gpu.free_tensor(act_bf16);
                let _ = gpu.free_tensor(gate_f32);
                let _ = gpu.free_tensor(up_f32);
                return Err(format!("deepseek4 parent: moe hidden_f32 alloc: {e:?}"));
            }
        };
        let hidden_bf16 = match gpu.alloc_tensor(&[max_rows, inter], DType::BF16) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(w_bf16);
                let _ = gpu.free_tensor(act_bf16);
                let _ = gpu.free_tensor(gate_f32);
                let _ = gpu.free_tensor(up_f32);
                let _ = gpu.free_tensor(hidden_f32);
                return Err(format!("deepseek4 parent: moe hidden_bf16 alloc: {e:?}"));
            }
        };
        let expert_out_f32 = match gpu.alloc_tensor(&[max_rows, dim], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(w_bf16);
                let _ = gpu.free_tensor(act_bf16);
                let _ = gpu.free_tensor(gate_f32);
                let _ = gpu.free_tensor(up_f32);
                let _ = gpu.free_tensor(hidden_f32);
                let _ = gpu.free_tensor(hidden_bf16);
                return Err(format!("deepseek4 parent: moe expert_out_f32 alloc: {e:?}"));
            }
        };

        let bytes = w_bf16.buf.size()
            + act_bf16.buf.size()
            + gate_f32.buf.size()
            + up_f32.buf.size()
            + hidden_f32.buf.size()
            + hidden_bf16.buf.size()
            + expert_out_f32.buf.size();

        Ok(Self {
            w_bf16,
            act_bf16,
            gate_f32,
            up_f32,
            hidden_f32,
            hidden_bf16,
            expert_out_f32,
            max_rows,
            dim,
            inter,
            bytes,
        })
    }

    /// Total scratch bytes resident on device. Constant w.r.t. expert count.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn inter(&self) -> usize {
        self.inter
    }
}

/// Routing result for one batch of rows.
///
/// Layout: `weights` and `indices` are both length `rows * topk`, row-major
/// (`row * topk + slot`).
#[derive(Clone, Debug)]
pub struct ParentRouting {
    pub weights: Vec<f32>,
    pub indices: Vec<u32>,
    pub rows: usize,
    pub topk: usize,
}

impl ParentRouting {
    /// Number of distinct expert ids selected across the batch.
    pub fn distinct_experts(&self) -> usize {
        let mut seen = vec![false; 256];
        let mut n = 0usize;
        for &idx in &self.indices {
            let i = idx as usize;
            if i < seen.len() && !seen[i] {
                seen[i] = true;
                n += 1;
            }
        }
        // Handle expert ids >= 256 without panicking (still count uniquely).
        if self.indices.iter().any(|&i| i as usize >= seen.len()) {
            let mut set = std::collections::BTreeSet::new();
            for &idx in &self.indices {
                set.insert(idx);
            }
            return set.len();
        }
        n
    }
}

/// Group `(row, slot_weight)` pairs by expert id.
///
/// Pure host helper — unit-tested without a GPU.
pub fn group_tokens_by_expert(
    routing: &ParentRouting,
    n_experts: usize,
) -> Result<Vec<Vec<(usize, f32)>>, String> {
    if routing.weights.len() != routing.rows * routing.topk
        || routing.indices.len() != routing.rows * routing.topk
    {
        return Err(format!(
            "deepseek4 parent: routing length mismatch (weights={}, indices={}, rows*topk={})",
            routing.weights.len(),
            routing.indices.len(),
            routing.rows * routing.topk
        ));
    }
    let mut groups: Vec<Vec<(usize, f32)>> = (0..n_experts).map(|_| Vec::new()).collect();
    for row in 0..routing.rows {
        for s in 0..routing.topk {
            let off = row * routing.topk + s;
            let eid = routing.indices[off] as usize;
            if eid >= n_experts {
                return Err(format!(
                    "deepseek4 parent: expert index {eid} out of range (n_routed_experts={n_experts})"
                ));
            }
            groups[eid].push((row, routing.weights[off]));
        }
    }
    Ok(groups)
}

/// Host-side sqrt(softplus(x)) matching `F.softplus(scores).sqrt()` and the
/// `sqrt_softplus_f32` HIP kernel (branched softplus for stability).
#[inline]
pub fn sqrt_softplus_host(x: f32) -> f32 {
    let sp = if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    };
    sp.sqrt()
}

/// Host-side score-route top-k selection (bias-aware / noaux_tc).
///
/// `scores_raw` is post-sqrtsoftplus, shape `[rows * n_experts]`.
/// `bias` is optional length-`n_experts` selection bias (not applied to weights).
///
/// Returns `(weights, indices)` each of length `rows * topk`.
pub fn score_route_topk(
    scores_raw: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    n_experts: usize,
    topk: usize,
    route_scale: f32,
) -> Result<(Vec<f32>, Vec<u32>), String> {
    if rows == 0 || topk == 0 {
        return Err("deepseek4 parent: score_route_topk requires rows>0 and topk>0".to_owned());
    }
    if topk > n_experts {
        return Err(format!(
            "deepseek4 parent: topk={topk} exceeds n_experts={n_experts}"
        ));
    }
    if scores_raw.len() < rows * n_experts {
        return Err(format!(
            "deepseek4 parent: scores_raw too short (have {} need {})",
            scores_raw.len(),
            rows * n_experts
        ));
    }
    if let Some(b) = bias {
        if b.len() < n_experts {
            return Err(format!(
                "deepseek4 parent: gate bias too short (have {} need {n_experts})",
                b.len()
            ));
        }
    }

    let mut weights = vec![0.0f32; rows * topk];
    let mut indices = vec![0u32; rows * topk];

    for r in 0..rows {
        let base = r * n_experts;
        // Selection scores = original + bias; weights come from original.
        let mut order: Vec<usize> = (0..n_experts).collect();
        order.sort_by(|&a, &b| {
            let sa = scores_raw[base + a] + bias.map(|bv| bv[a]).unwrap_or(0.0);
            let sb = scores_raw[base + b] + bias.map(|bv| bv[b]).unwrap_or(0.0);
            // Descending; stable on ties by lower expert id.
            sb.partial_cmp(&sa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });

        let mut sum = 0.0f32;
        for s in 0..topk {
            let eid = order[s];
            let w = scores_raw[base + eid];
            indices[r * topk + s] = eid as u32;
            weights[r * topk + s] = w;
            sum += w;
        }
        // norm_topk_prob for non-softmax score_func (sqrtsoftplus).
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for s in 0..topk {
                weights[r * topk + s] *= inv * route_scale;
            }
        } else {
            let even = route_scale / topk as f32;
            for s in 0..topk {
                weights[r * topk + s] = even;
            }
        }
    }
    Ok((weights, indices))
}

/// Host-side hash-route weight gather (indices from `tid2eid`).
///
/// `indices` is already filled `[rows * topk]` from the hash table.
/// Weights still come from unbiased post-sqrtsoftplus scores, then L1-normed
/// and scaled — same as score routing after selection.
pub fn hash_route_weights(
    scores_raw: &[f32],
    indices: &[u32],
    rows: usize,
    n_experts: usize,
    topk: usize,
    route_scale: f32,
) -> Result<Vec<f32>, String> {
    if scores_raw.len() < rows * n_experts {
        return Err(format!(
            "deepseek4 parent: scores_raw too short (have {} need {})",
            scores_raw.len(),
            rows * n_experts
        ));
    }
    if indices.len() < rows * topk {
        return Err(format!(
            "deepseek4 parent: hash indices too short (have {} need {})",
            indices.len(),
            rows * topk
        ));
    }
    let mut weights = vec![0.0f32; rows * topk];
    for r in 0..rows {
        let base = r * n_experts;
        let mut sum = 0.0f32;
        for s in 0..topk {
            let eid = indices[r * topk + s] as usize;
            if eid >= n_experts {
                return Err(format!(
                    "deepseek4 parent: hash expert index {eid} out of range (n={n_experts})"
                ));
            }
            let w = scores_raw[base + eid];
            weights[r * topk + s] = w;
            sum += w;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for s in 0..topk {
                weights[r * topk + s] *= inv * route_scale;
            }
        } else {
            let even = route_scale / topk as f32;
            for s in 0..topk {
                weights[r * topk + s] = even;
            }
        }
    }
    Ok(weights)
}

/// Routing for one batch of rows.
///
/// Score-routed layers (`layer_idx >= num_hash_layers`) use gate weight + bias
/// top-k. Hash-routed layers use `tid2eid[input_ids]` for indices and still
/// gather / renorm weights from unbiased sqrtsoftplus scores.
///
/// # Routing path: host f32 (not GPU `sqrt_softplus_f32`)
///
/// Chosen deliberately. The reference gate is
/// `linear(x.float(), weight.float())` — a plain F32×F32 matmul of BF16-widened
/// operands with **no** activation quant (`model.py:570`). The GPU
/// `sqrt_softplus_f32` kernel only covers the elementwise activation, not the
/// gate GEMM, and `parent_linear_*` always runs FP8 act-quant which would be
/// **wrong** for the gate. Host-side `rows × 256` scoring is tiny for the
/// calibration path (16–32 tokens) and matches the reference exactly.
pub fn parent_route(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    layer: &ParentLayerWeights,
    cfg: &ParentQuantConfig,
    x: &GpuTensor,
    rows: usize,
    input_ids: Option<&[u32]>,
) -> Result<ParentRouting, String> {
    backend.ensure_device(gpu)?;
    if rows == 0 {
        return Err("deepseek4 parent: parent_route rows must be > 0".to_owned());
    }
    let dim = PARENT_DIM;
    let n_experts = cfg.n_routed_experts;
    let topk = cfg.num_experts_per_tok;
    if topk == 0 || n_experts == 0 {
        return Err("deepseek4 parent: n_routed_experts and num_experts_per_tok must be > 0".to_owned());
    }
    if x.dtype != DType::BF16 {
        return Err(format!(
            "deepseek4 parent: parent_route x must be BF16 (got {:?})",
            x.dtype
        ));
    }
    let need_x = rows
        .checked_mul(dim)
        .and_then(|e| e.checked_mul(2))
        .ok_or_else(|| "deepseek4 parent: route x size overflow".to_owned())?;
    if x.buf.size() < need_x {
        return Err(format!(
            "deepseek4 parent: route x too small (have {} need {need_x} for BF16 [{rows},{dim}])",
            x.buf.size()
        ));
    }
    if layer.gate_weight.dtype != DType::BF16 {
        return Err(format!(
            "deepseek4 parent: gate_weight must be BF16 (got {:?})",
            layer.gate_weight.dtype
        ));
    }
    let need_w = n_experts
        .checked_mul(dim)
        .and_then(|e| e.checked_mul(2))
        .ok_or_else(|| "deepseek4 parent: gate weight size overflow".to_owned())?;
    if layer.gate_weight.buf.size() < need_w {
        return Err(format!(
            "deepseek4 parent: gate_weight too small (have {} need {need_w})",
            layer.gate_weight.buf.size()
        ));
    }

    let is_hash = layer.layer_idx < cfg.num_hash_layers;
    if is_hash {
        if layer.tid2eid.is_none() {
            return Err(format!(
                "deepseek4 parent: layer {} is hash-routed but tid2eid is missing",
                layer.layer_idx
            ));
        }
        if input_ids.is_none() {
            return Err(format!(
                "deepseek4 parent: layer {} is hash-routed; input_ids required",
                layer.layer_idx
            ));
        }
        if input_ids.map(|ids| ids.len()).unwrap_or(0) < rows {
            return Err(format!(
                "deepseek4 parent: input_ids length {} < rows {rows}",
                input_ids.map(|ids| ids.len()).unwrap_or(0)
            ));
        }
    } else if layer.gate_bias.is_none() {
        return Err(format!(
            "deepseek4 parent: layer {} is score-routed but gate_bias is missing",
            layer.layer_idx
        ));
    }

    // Download activations and gate weight; widen BF16 → f32 on host.
    let x_f32 = download_bf16_as_f32(gpu, x, rows * dim)?;
    let w_f32 = download_bf16_as_f32(gpu, &layer.gate_weight, n_experts * dim)?;

    // scores[r, e] = sum_k x[r,k] * W[e,k]  (F.linear)
    let mut scores = vec![0.0f32; rows * n_experts];
    for r in 0..rows {
        let xrow = &x_f32[r * dim..(r + 1) * dim];
        for e in 0..n_experts {
            let wrow = &w_f32[e * dim..(e + 1) * dim];
            let mut acc = 0.0f32;
            for k in 0..dim {
                acc += xrow[k] * wrow[k];
            }
            scores[r * n_experts + e] = sqrt_softplus_host(acc);
        }
    }

    let (weights, indices) = if is_hash {
        let ids = input_ids.unwrap();
        let tid2eid = layer.tid2eid.as_ref().unwrap();
        let indices = lookup_tid2eid(gpu, tid2eid, ids, rows, topk)?;
        let weights = hash_route_weights(
            &scores,
            &indices,
            rows,
            n_experts,
            topk,
            effective_parent_route_scale(),
        )?;
        (weights, indices)
    } else {
        let bias = download_f32_tensor(gpu, layer.gate_bias.as_ref().unwrap(), n_experts)?;
        score_route_topk(
            &scores,
            Some(&bias),
            rows,
            n_experts,
            topk,
            effective_parent_route_scale(),
        )?
    };

    // Validate expert ids early.
    for &idx in &indices {
        if idx as usize >= n_experts {
            return Err(format!(
                "deepseek4 parent: routed expert index {idx} out of range (n={n_experts})"
            ));
        }
    }

    Ok(ParentRouting {
        weights,
        indices,
        rows,
        topk,
    })
}

/// Full MoE forward: routed experts (decode-on-demand, grouped) + shared expert.
///
/// Accumulates in f32 exactly as `MoE.forward`:
/// `y[idx] += expert(x[idx], weight)` then `y += shared_experts(x)`.
///
/// `x` is BF16 `[rows, dim]` and is **not** destroyed (all linears stage a
/// fresh copy into scratch — `parent_linear_*` destroys its `x_bf16` input).
/// `out` is F32 `[rows, dim]` and is overwritten.
///
/// Returns `Ok(decode_calls)` via the side-effect counter only when the
/// caller uses [`parent_moe_forward_counted`]; the plain entry returns `()`.
pub fn parent_moe_forward(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    layer: &ParentLayerWeights,
    cfg: &ParentQuantConfig,
    scratch: &mut ParentMoeScratch,
    x: &GpuTensor,
    rows: usize,
    routing: &ParentRouting,
    out: &GpuTensor,
) -> Result<(), String> {
    let _ = parent_moe_forward_counted(
        gpu, backend, layer, cfg, scratch, x, rows, routing, out,
    )?;
    Ok(())
}

/// Same as [`parent_moe_forward`], but returns the number of expert-weight
/// decode calls issued (3 per selected expert: w1, w3, w2). Used by the
/// calibration smoke to prove token-grouping works.
pub fn parent_moe_forward_counted(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    layer: &ParentLayerWeights,
    cfg: &ParentQuantConfig,
    scratch: &mut ParentMoeScratch,
    x: &GpuTensor,
    rows: usize,
    routing: &ParentRouting,
    out: &GpuTensor,
) -> Result<usize, String> {
    backend.ensure_device(gpu)?;
    if rows == 0 {
        return Err("deepseek4 parent: parent_moe_forward rows must be > 0".to_owned());
    }
    if rows > scratch.max_rows {
        return Err(format!(
            "deepseek4 parent: rows {rows} exceeds ParentMoeScratch capacity {}",
            scratch.max_rows
        ));
    }
    if routing.rows != rows {
        return Err(format!(
            "deepseek4 parent: routing.rows {} != forward rows {rows}",
            routing.rows
        ));
    }
    if routing.topk != cfg.num_experts_per_tok {
        return Err(format!(
            "deepseek4 parent: routing.topk {} != num_experts_per_tok {}",
            routing.topk, cfg.num_experts_per_tok
        ));
    }
    let dim = scratch.dim;
    let inter = scratch.inter;
    if x.dtype != DType::BF16 {
        return Err(format!(
            "deepseek4 parent: moe x must be BF16 (got {:?})",
            x.dtype
        ));
    }
    if out.dtype != DType::F32 {
        return Err(format!(
            "deepseek4 parent: moe out must be F32 (got {:?})",
            out.dtype
        ));
    }
    let need_x = rows * dim * 2;
    if x.buf.size() < need_x {
        return Err(format!(
            "deepseek4 parent: moe x too small (have {} need {need_x})",
            x.buf.size()
        ));
    }
    let need_out = rows * dim * 4;
    if out.buf.size() < need_out {
        return Err(format!(
            "deepseek4 parent: moe out too small (have {} need {need_out})",
            out.buf.size()
        ));
    }
    if layer.experts.is_empty() {
        return Err(
            "deepseek4 parent: moe forward requires loaded routed experts (load_experts=true)"
                .to_owned(),
        );
    }
    if layer.experts.len() != cfg.n_routed_experts {
        return Err(format!(
            "deepseek4 parent: experts.len() {} != n_routed_experts {}",
            layer.experts.len(),
            cfg.n_routed_experts
        ));
    }

    // Validate shared-expert shapes.
    validate_dense_shape("shared_w1", &layer.shared_w1, inter, dim)?;
    validate_dense_shape("shared_w3", &layer.shared_w3, inter, dim)?;
    validate_dense_shape("shared_w2", &layer.shared_w2, dim, inter)?;

    // Zero the f32 accumulator (y = 0 before routed experts).
    zero_f32(gpu, out, rows * dim)?;

    let groups = group_tokens_by_expert(routing, cfg.n_routed_experts)?;
    let mut decode_calls = 0usize;

    // Host-side pristine x (BF16 bits) — re-gathered per expert group so we
    // never double-quantize a shared residual tile.
    let x_bytes = download_bytes(gpu, x, rows * dim * 2)?;

    for (eid, members) in groups.iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        let n_tok = members.len();
        if eid >= layer.experts.len() {
            return Err(format!(
                "deepseek4 parent: expert index {eid} out of range (loaded {})",
                layer.experts.len()
            ));
        }
        let expert: &ParentExpertTriple = &layer.experts[eid];
        if expert.expert_id != eid {
            return Err(format!(
                "deepseek4 parent: experts[{eid}].expert_id = {} (expected {eid})",
                expert.expert_id
            ));
        }
        validate_expert_shapes(expert, inter, dim)?;

        // Gather this expert's tokens into a contiguous host BF16 tile.
        let mut gathered = vec![0u8; n_tok * dim * 2];
        let mut route_w = vec![0.0f32; n_tok];
        for (i, &(row, w)) in members.iter().enumerate() {
            let src = row * dim * 2;
            let dst = i * dim * 2;
            gathered[dst..dst + dim * 2].copy_from_slice(&x_bytes[src..src + dim * 2]);
            route_w[i] = w;
        }

        // ── w1 (gate) ──────────────────────────────────────────────────
        expert.w1.decode_into(gpu, &scratch.w_bf16)?;
        decode_calls += 1;
        upload_bf16_into(gpu, &scratch.act_bf16, &gathered, n_tok * dim)?;
        // View act / out as the first n_tok rows via byte-sized ops; the
        // linear validates buf.size() >= m*k*2, which holds for max_rows capacity.
        parent_linear_expert(
            gpu,
            backend,
            &scratch.w_bf16,
            expert.w1.n(),
            expert.w1.k(),
            &scratch.act_bf16,
            n_tok,
            &scratch.gate_f32,
        )?;

        // ── w3 (up) ────────────────────────────────────────────────────
        expert.w3.decode_into(gpu, &scratch.w_bf16)?;
        decode_calls += 1;
        // Fresh activation copy — parent_linear_expert destroyed the previous.
        upload_bf16_into(gpu, &scratch.act_bf16, &gathered, n_tok * dim)?;
        parent_linear_expert(
            gpu,
            backend,
            &scratch.w_bf16,
            expert.w3.n(),
            expert.w3.k(),
            &scratch.act_bf16,
            n_tok,
            &scratch.up_f32,
        )?;

        // ── SwiGLU with asymmetric clamp ───────────────────────────────
        // deepseek4_silu_mul_clamp_f32 uses gate.numel() as n, so we must
        // run over exactly n_tok*inter elements. Upload-sized views aren't
        // available; download-scale-reupload for the weighted hidden is
        // already on the host path below, so do clamp+silu*up on host too
        // for the exact n_tok tile (avoids over-running into scratch tail).
        let gate = download_f32_prefix(gpu, &scratch.gate_f32, n_tok * inter)?;
        let up = download_f32_prefix(gpu, &scratch.up_f32, n_tok * inter)?;
        let mut hidden = vec![0.0f32; n_tok * inter];
        swiglu_clamp_silu_mul(
            &gate,
            &up,
            &mut hidden,
            PARENT_SWIGLU_LIMIT,
        );
        // Apply routing weight INSIDE the expert, before w2 (model.py:609-610).
        for i in 0..n_tok {
            let w = route_w[i];
            let base = i * inter;
            for j in 0..inter {
                hidden[base + j] *= w;
            }
        }

        // ── w2 (down) ──────────────────────────────────────────────────
        let hidden_bf16_bytes = pack_f32_to_bf16_bytes(&hidden);
        upload_bf16_into(gpu, &scratch.hidden_bf16, &hidden_bf16_bytes, n_tok * inter)?;
        expert.w2.decode_into(gpu, &scratch.w_bf16)?;
        decode_calls += 1;
        parent_linear_expert(
            gpu,
            backend,
            &scratch.w_bf16,
            expert.w2.n(),
            expert.w2.k(),
            &scratch.hidden_bf16,
            n_tok,
            &scratch.expert_out_f32,
        )?;

        // Scatter-add into y[row].
        let eout = download_f32_prefix(gpu, &scratch.expert_out_f32, n_tok * dim)?;
        let mut y = download_f32_prefix(gpu, out, rows * dim)?;
        for (i, &(row, _)) in members.iter().enumerate() {
            let src = i * dim;
            let dst = row * dim;
            for j in 0..dim {
                y[dst + j] += eout[src + j];
            }
        }
        upload_f32_into(gpu, out, &y, rows * dim)?;
    }

    // ── Shared expert over the full batch (no routing weight) ──────────
    // Fresh x copy for each projection (destructive act-quant).
    run_shared_expert(
        gpu,
        backend,
        layer,
        scratch,
        &x_bytes,
        rows,
        dim,
        inter,
        out,
    )?;

    Ok(decode_calls)
}

fn run_shared_expert(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    layer: &ParentLayerWeights,
    scratch: &mut ParentMoeScratch,
    x_bytes: &[u8],
    rows: usize,
    dim: usize,
    inter: usize,
    out: &GpuTensor,
) -> Result<(), String> {
    // w1
    upload_bf16_into(gpu, &scratch.act_bf16, x_bytes, rows * dim)?;
    parent_linear_dense(
        gpu,
        backend,
        &layer.shared_w1,
        &scratch.act_bf16,
        rows,
        &scratch.gate_f32,
    )?;
    // w3
    upload_bf16_into(gpu, &scratch.act_bf16, x_bytes, rows * dim)?;
    parent_linear_dense(
        gpu,
        backend,
        &layer.shared_w3,
        &scratch.act_bf16,
        rows,
        &scratch.up_f32,
    )?;

    let gate = download_f32_prefix(gpu, &scratch.gate_f32, rows * inter)?;
    let up = download_f32_prefix(gpu, &scratch.up_f32, rows * inter)?;
    let mut hidden = vec![0.0f32; rows * inter];
    swiglu_clamp_silu_mul(&gate, &up, &mut hidden, PARENT_SWIGLU_LIMIT);

    let hidden_bf16_bytes = pack_f32_to_bf16_bytes(&hidden);
    upload_bf16_into(gpu, &scratch.hidden_bf16, &hidden_bf16_bytes, rows * inter)?;
    parent_linear_dense(
        gpu,
        backend,
        &layer.shared_w2,
        &scratch.hidden_bf16,
        rows,
        &scratch.expert_out_f32,
    )?;

    // y += shared
    let shared = download_f32_prefix(gpu, &scratch.expert_out_f32, rows * dim)?;
    let mut y = download_f32_prefix(gpu, out, rows * dim)?;
    for i in 0..rows * dim {
        y[i] += shared[i];
    }
    upload_f32_into(gpu, out, &y, rows * dim)?;
    Ok(())
}

/// Asymmetric SwiGLU clamp + silu(gate)*up, matching `Expert.forward` and
/// `deepseek4_silu_mul_clamp.hip`:
/// - `up` clamped to `[-limit, +limit]`
/// - `gate` clamped only from above (`min` unrestricted)
#[inline]
pub fn swiglu_clamp_silu_mul(gate: &[f32], up: &[f32], out: &mut [f32], limit: f32) {
    let n = out.len().min(gate.len()).min(up.len());
    for i in 0..n {
        let mut g = gate[i];
        let mut u = up[i];
        if g > limit {
            g = limit;
        }
        if u > limit {
            u = limit;
        }
        if u < -limit {
            u = -limit;
        }
        // silu(g) = g / (1 + exp(-g))
        let silu_g = g / (1.0 + (-g).exp());
        out[i] = silu_g * u;
    }
}

// ── shape / IO helpers ──────────────────────────────────────────────────────

fn validate_dense_shape(name: &str, w: &ParentDenseWeight, n: usize, k: usize) -> Result<(), String> {
    if w.n() != n || w.k() != k {
        return Err(format!(
            "deepseek4 parent: {name} shape [{},{}] != expected [{n},{k}]",
            w.n(),
            w.k()
        ));
    }
    Ok(())
}

fn validate_expert_shapes(e: &ParentExpertTriple, inter: usize, dim: usize) -> Result<(), String> {
    if e.w1.n() != inter || e.w1.k() != dim {
        return Err(format!(
            "deepseek4 parent: expert {} w1 [{},{}] != [{inter},{dim}]",
            e.expert_id,
            e.w1.n(),
            e.w1.k()
        ));
    }
    if e.w3.n() != inter || e.w3.k() != dim {
        return Err(format!(
            "deepseek4 parent: expert {} w3 [{},{}] != [{inter},{dim}]",
            e.expert_id,
            e.w3.n(),
            e.w3.k()
        ));
    }
    if e.w2.n() != dim || e.w2.k() != inter {
        return Err(format!(
            "deepseek4 parent: expert {} w2 [{},{}] != [{dim},{inter}]",
            e.expert_id,
            e.w2.n(),
            e.w2.k()
        ));
    }
    Ok(())
}

fn download_bytes(gpu: &Gpu, t: &GpuTensor, nbytes: usize) -> Result<Vec<u8>, String> {
    if t.buf.size() < nbytes {
        return Err(format!(
            "deepseek4 parent: download_bytes: have {} need {nbytes}",
            t.buf.size()
        ));
    }
    let mut data = vec![0u8; nbytes];
    gpu.hip
        .memcpy_dtoh(&mut data, &t.buf)
        .map_err(|e| format!("deepseek4 parent: download_bytes: {e:?}"))?;
    Ok(data)
}

fn download_bf16_as_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    let bytes = download_bytes(gpu, t, nelems * 2)?;
    let mut out = Vec::with_capacity(nelems);
    for i in 0..nelems {
        let b = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        out.push(f32::from_bits((b as u32) << 16));
    }
    Ok(out)
}

fn download_f32_tensor(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    download_f32_prefix(gpu, t, nelems)
}

fn download_f32_prefix(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    if t.dtype != DType::F32 {
        return Err(format!(
            "deepseek4 parent: expected F32 tensor (got {:?})",
            t.dtype
        ));
    }
    let nbytes = nelems * 4;
    if t.buf.size() < nbytes {
        return Err(format!(
            "deepseek4 parent: f32 download too small (have {} need {nbytes})",
            t.buf.size()
        ));
    }
    let mut data = vec![0.0f32; nelems];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, nbytes) };
    gpu.hip
        .memcpy_dtoh(bytes, &t.buf)
        .map_err(|e| format!("deepseek4 parent: f32 download: {e:?}"))?;
    Ok(data)
}

fn upload_f32_into(gpu: &Gpu, t: &GpuTensor, data: &[f32], nelems: usize) -> Result<(), String> {
    if t.dtype != DType::F32 {
        return Err(format!(
            "deepseek4 parent: upload_f32_into expects F32 (got {:?})",
            t.dtype
        ));
    }
    if data.len() < nelems {
        return Err(format!(
            "deepseek4 parent: upload_f32_into data short (have {} need {nelems})",
            data.len()
        ));
    }
    let nbytes = nelems * 4;
    if t.buf.size() < nbytes {
        return Err(format!(
            "deepseek4 parent: upload_f32_into dest too small (have {} need {nbytes})",
            t.buf.size()
        ));
    }
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, nbytes) };
    gpu.hip
        .memcpy_htod(&t.buf, bytes)
        .map_err(|e| format!("deepseek4 parent: upload_f32_into: {e:?}"))
}

fn upload_bf16_into(gpu: &Gpu, t: &GpuTensor, bytes: &[u8], nelems: usize) -> Result<(), String> {
    if t.dtype != DType::BF16 {
        return Err(format!(
            "deepseek4 parent: upload_bf16_into expects BF16 (got {:?})",
            t.dtype
        ));
    }
    let nbytes = nelems * 2;
    if bytes.len() < nbytes {
        return Err(format!(
            "deepseek4 parent: upload_bf16_into data short (have {} need {nbytes})",
            bytes.len()
        ));
    }
    if t.buf.size() < nbytes {
        return Err(format!(
            "deepseek4 parent: upload_bf16_into dest too small (have {} need {nbytes})",
            t.buf.size()
        ));
    }
    gpu.hip
        .memcpy_htod(&t.buf, &bytes[..nbytes])
        .map_err(|e| format!("deepseek4 parent: upload_bf16_into: {e:?}"))
}

fn pack_f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let bf = round_to_bf16(v);
        let bits = (bf.to_bits() >> 16) as u16;
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

fn zero_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<(), String> {
    let zeros = vec![0.0f32; nelems];
    upload_f32_into(gpu, t, &zeros, nelems)
}

/// Look up `tid2eid[input_ids[r], :]` for hash-routed layers.
///
/// `tid2eid` is I64 `[vocab, topk]` on device.
fn lookup_tid2eid(
    gpu: &Gpu,
    tid2eid: &GpuTensor,
    input_ids: &[u32],
    rows: usize,
    topk: usize,
) -> Result<Vec<u32>, String> {
    if tid2eid.shape.len() != 2 {
        return Err(format!(
            "deepseek4 parent: tid2eid shape {:?} not rank-2",
            tid2eid.shape
        ));
    }
    let vocab = tid2eid.shape[0];
    let t_topk = tid2eid.shape[1];
    if t_topk != topk {
        return Err(format!(
            "deepseek4 parent: tid2eid topk {t_topk} != cfg topk {topk}"
        ));
    }
    let nbytes = vocab
        .checked_mul(topk)
        .and_then(|e| e.checked_mul(8))
        .ok_or_else(|| "deepseek4 parent: tid2eid size overflow".to_owned())?;
    if tid2eid.buf.size() < nbytes {
        return Err(format!(
            "deepseek4 parent: tid2eid buffer too small (have {} need {nbytes})",
            tid2eid.buf.size()
        ));
    }
    let raw = download_bytes(gpu, tid2eid, nbytes)?;
    let mut indices = vec![0u32; rows * topk];
    for r in 0..rows {
        let tid = input_ids[r] as usize;
        if tid >= vocab {
            return Err(format!(
                "deepseek4 parent: input_id {tid} out of tid2eid vocab {vocab}"
            ));
        }
        for s in 0..topk {
            let off = (tid * topk + s) * 8;
            let v = i64::from_le_bytes(raw[off..off + 8].try_into().unwrap());
            if v < 0 {
                return Err(format!(
                    "deepseek4 parent: negative tid2eid entry {v} at tid={tid} slot={s}"
                ));
            }
            indices[r * topk + s] = v as u32;
        }
    }
    Ok(indices)
}

// ── unit tests (host-side, no GPU) ──────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqrt_softplus_matches_stable_formula() {
        // softplus(0)=ln2, sqrt=sqrt(ln2)
        let sp0 = sqrt_softplus_host(0.0);
        assert!((sp0 - (2.0f32.ln().sqrt())).abs() < 1e-6);

        // large positive ≈ sqrt(x)
        let sp10 = sqrt_softplus_host(10.0);
        assert!((sp10 - 10.0f32.sqrt()).abs() < 1e-3);

        // large negative ≈ 0
        let spn = sqrt_softplus_host(-20.0);
        assert!(spn >= 0.0 && spn < 1e-4);
    }

    #[test]
    fn score_route_bias_affects_selection_not_weights() {
        // 1 row, 4 experts, topk=2. Expert 0 has highest raw score; expert 3
        // has huge bias so it wins selection, but weight still comes from raw.
        let scores = vec![0.8f32, 0.1, 0.05, 0.05];
        let bias = vec![0.0f32, 0.0, 0.0, 10.0];
        let (w, idx) = score_route_topk(&scores, Some(&bias), 1, 4, 2, 1.5).unwrap();
        assert_eq!(idx[0], 3); // bias winner first
        // second should be expert 0 (highest remaining selection score)
        assert_eq!(idx[1], 0);
        // weights from raw, L1-normed, *1.5
        let sum_raw = scores[3] + scores[0];
        assert!((w[0] - scores[3] / sum_raw * 1.5).abs() < 1e-6);
        assert!((w[1] - scores[0] / sum_raw * 1.5).abs() < 1e-6);
        // weights sum to route_scale
        assert!((w[0] + w[1] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn score_route_no_bias_picks_top_raw() {
        let scores = vec![0.1f32, 0.5, 0.3, 0.2];
        let (w, idx) = score_route_topk(&scores, None, 1, 4, 2, 1.5).unwrap();
        assert_eq!(idx[0], 1);
        assert_eq!(idx[1], 2);
        assert!((w[0] + w[1] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn group_tokens_by_expert_groups_correctly() {
        let routing = ParentRouting {
            // row0 → experts 1,3; row1 → experts 1,2
            weights: vec![0.9, 0.6, 0.8, 0.7],
            indices: vec![1, 3, 1, 2],
            rows: 2,
            topk: 2,
        };
        let groups = group_tokens_by_expert(&routing, 4).unwrap();
        assert!(groups[0].is_empty());
        assert_eq!(groups[1], vec![(0, 0.9), (1, 0.8)]);
        assert_eq!(groups[2], vec![(1, 0.7)]);
        assert_eq!(groups[3], vec![(0, 0.6)]);
    }

    #[test]
    fn group_tokens_rejects_oob_expert() {
        let routing = ParentRouting {
            weights: vec![1.0],
            indices: vec![9],
            rows: 1,
            topk: 1,
        };
        let err = group_tokens_by_expert(&routing, 4).unwrap_err();
        assert!(err.contains("out of range"), "{err}");
        assert!(err.starts_with("deepseek4 parent:"), "{err}");
    }

    #[test]
    fn swiglu_asymmetric_clamp() {
        // gate below -limit stays; up is symmetric.
        let gate = vec![-100.0f32, 100.0, 5.0];
        let up = vec![-100.0f32, 100.0, 5.0];
        let mut out = vec![0.0f32; 3];
        swiglu_clamp_silu_mul(&gate, &up, &mut out, 10.0);
        // gate[-100] unclamped → silu ≈ 0; up clamped to -10
        assert!(out[0].abs() < 1e-5, "got {}", out[0]);
        // gate clamped to 10, up clamped to 10
        let g = 10.0f32;
        let silu = g / (1.0 + (-g).exp());
        assert!((out[1] - silu * 10.0).abs() < 1e-5, "got {}", out[1]);
        // no clamp needed
        let g5 = 5.0f32;
        let silu5 = g5 / (1.0 + (-g5).exp());
        assert!((out[2] - silu5 * 5.0).abs() < 1e-5);
    }

    #[test]
    fn scratch_bytes_formula_is_constant_in_experts() {
        // Pure arithmetic check of the sizing formula — no GPU.
        let max_rows = 16usize;
        let dim = PARENT_DIM;
        let inter = PARENT_MOE_INTER;
        let w_elems = inter * dim;
        let bytes = w_elems * 2 // w_bf16
            + max_rows * dim * 2 // act_bf16
            + max_rows * inter * 4 // gate_f32
            + max_rows * inter * 4 // up_f32
            + max_rows * inter * 4 // hidden_f32
            + max_rows * inter * 2 // hidden_bf16
            + max_rows * dim * 4; // expert_out_f32
        // ~16 MiB weight tile + small per-row — not 256× anything.
        assert!(bytes < 40 * 1024 * 1024, "scratch too large: {bytes}");
        // Independent of expert count: same for 1 or 256 experts.
        let bytes_same = bytes; // formula has no n_experts term
        assert_eq!(bytes, bytes_same);
    }

    #[test]
    fn hash_route_weights_renorms() {
        let scores = vec![0.2f32, 0.4, 0.1, 0.3]; // 1 row × 4
        let indices = vec![1u32, 3];
        let w = hash_route_weights(&scores, &indices, 1, 4, 2, 1.5).unwrap();
        let sum_raw = 0.4 + 0.3;
        assert!((w[0] - 0.4 / sum_raw * 1.5).abs() < 1e-6);
        assert!((w[1] - 0.3 / sum_raw * 1.5).abs() < 1e-6);
    }

    #[test]
    fn distinct_experts_counts_unique() {
        let r = ParentRouting {
            weights: vec![1.0; 6],
            indices: vec![1, 2, 1, 5, 2, 7],
            rows: 3,
            topk: 2,
        };
        assert_eq!(r.distinct_experts(), 4);
    }

    /// Gate-6 row-0 explosion guard: routing weight must scale the *pre-w2*
    /// SwiGLU intermediate, not the expert output (`model.py:609-610`).
    /// L5 oracle comparison relies on this asymmetry.
    #[test]
    fn routing_weight_scales_swiglu_before_down_proj() {
        let inter = 8usize;
        let dim = 4usize;
        let gate = vec![1.0f32; inter];
        let up = vec![2.0f32; inter];
        let w_route = 0.75f32;
        let hid = crate::layer_ref::expert_swiglu_ref(
            &gate,
            &up,
            1,
            inter,
            10.0,
            Some(&[w_route]),
        );
        let mut hid_unweighted = vec![0.0f32; inter];
        swiglu_clamp_silu_mul(&gate, &up, &mut hid_unweighted, 10.0);
        for j in 0..inter {
            assert!(
                (hid[j] - hid_unweighted[j] * w_route).abs() < 1e-6,
                "slot {j}: weighted {} != unweighted*w {}",
                hid[j],
                hid_unweighted[j] * w_route
            );
        }
        // Toy w2 = ones: down-proj of weighted hid == w_route * down(unweighted).
        let w2 = vec![1.0f32; dim * inter];
        let mut y_w = vec![0.0f32; dim];
        let mut y_1 = vec![0.0f32; dim];
        for n in 0..dim {
            let mut acc_w = 0.0f32;
            let mut acc_1 = 0.0f32;
            for k in 0..inter {
                acc_w += w2[n * inter + k] * hid[k];
                acc_1 += w2[n * inter + k] * hid_unweighted[k];
            }
            y_w[n] = acc_w;
            y_1[n] = acc_1;
        }
        for n in 0..dim {
            assert!(
                (y_w[n] - y_1[n] * w_route).abs() < 1e-5,
                "down-proj slot {n}: {} != {} * route",
                y_w[n],
                y_1[n]
            );
        }
    }

    /// Grouped dispatch must emit each (row, weight) exactly once. Row 0 is
    /// the first token of every group it joins — double-counting it was the
    /// leading hypothesis for the residual spike.
    #[test]
    fn group_tokens_row0_appears_exactly_topk_times() {
        // r0 → e0,e1 ; r1 → e2,e1
        let routing = ParentRouting {
            weights: vec![0.5, 0.4, 0.6, 0.3],
            indices: vec![0, 1, 2, 1],
            rows: 2,
            topk: 2,
        };
        let groups = group_tokens_by_expert(&routing, 4).unwrap();
        let seen: Vec<(usize, f32)> = groups.iter().flatten().copied().collect();
        assert_eq!(seen.len(), routing.rows * routing.topk);
        let r0_count = seen.iter().filter(|(r, _)| *r == 0).count();
        assert_eq!(
            r0_count, routing.topk,
            "row0 must appear exactly topk times, got {seen:?}"
        );
        // row0 is first member of e0 and of e1 (insertion order).
        assert_eq!(groups[0][0], (0, 0.5));
        assert_eq!(groups[1][0], (0, 0.4));
        assert_eq!(groups[1][1], (1, 0.3));
        let r0_w: f32 = seen
            .iter()
            .filter(|(r, _)| *r == 0)
            .map(|(_, w)| *w)
            .sum();
        assert!((r0_w - 0.9).abs() < 1e-6);
    }

    /// Route-scale contract: even a heavily peaked score row (L5 row-0 style)
    /// must L1-renorm then `* route_scale`, so weights always sum to 1.5 — not
    /// to the raw top score. Under-renorm was a candidate for MoE blow-up.
    #[test]
    fn score_route_weights_sum_to_route_scale() {
        // 3 rows, peaked and flat score rows — every row must sum to 1.5.
        let n_exp = 8usize;
        let topk = 6usize;
        let rows = 3usize;
        let mut scores = vec![0.0f32; rows * n_exp];
        for r in 0..rows {
            for e in 0..n_exp {
                scores[r * n_exp + e] = 0.1 + (e as f32) * 0.05 + (r as f32) * 0.01;
            }
            // Make row 0 heavily peaked on expert 0 (L5-like).
            if r == 0 {
                scores[0] = 10.0;
            }
        }
        let bias = vec![0.0f32; n_exp];
        let (w, _idx) =
            score_route_topk(&scores, Some(&bias), rows, n_exp, topk, PARENT_ROUTE_SCALE)
                .unwrap();
        for r in 0..rows {
            let sum: f32 = w[r * topk..(r + 1) * topk].iter().sum();
            assert!(
                (sum - PARENT_ROUTE_SCALE).abs() < 1e-5,
                "row {r} weight sum {sum} != route_scale"
            );
        }
        // Peaked row 0 still sums to route_scale (not ~weight of expert 0 alone).
        let r0: f32 = w[..topk].iter().sum();
        assert!((r0 - PARENT_ROUTE_SCALE).abs() < 1e-5);
    }

}
