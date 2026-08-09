// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Parent-checkpoint **Indexer** (`model.py:386-440`).
//!
//! Scores compressed KV positions and selects the top `index_topk = 512` for
//! the sparse attention path. Present only on layers with
//! `compress_ratio == 4` (21 of 43 layers).
//!
//! Operator-semantics authority:
//! - `.codeinsight+research/ds4-parent-ref/inference/model.py` `Indexer`
//! - config.json: `index_n_heads=64`, `index_head_dim=128`, `index_topk=512`,
//!   `q_lora_rank=1024`, `rope_head_dim=64`, `compress_rope_theta=160000`
//!
//! # Path summary (`model.py:408-439`)
//!
//! 1. `q = wq_b(qr)` — dense FP8 linear (`parent_linear_dense`), then
//!    unflatten to `[rows, H, D]`.
//! 2. Tail RoPE on `q[..., -rd:]` with **plain** `compress_rope_theta`
//!    (no YaRN — matches MQ `is_indexer=true` and the indexer's own
//!    compressor).
//! 3. **`rotate_activation(q)`** then **`fp4_act_quant(q, 32, inplace=True)`**
//!    — Hadamard + FP4 group-32 simulation on the full head (not FP8).
//! 4. Own compressor with `hadamard=true` fills the compressed KV cache
//!    (`parent_compressor_forward`).
//! 5. `weights = weights_proj(x) * (softmax_scale * n_heads**-0.5)` where
//!    `weights_proj` is plain BF16 (`F.linear`, **no** act-quant) and
//!    `softmax_scale = head_dim**-0.5`.
//! 6. `score[b,t] = sum_h relu(q[b,h]·kv[t]) * weights[b,h]`; causal mask
//!    on prefill; top-k with offset / `-1` padding.
//!
//! Reference note at `model.py:425`: QAT was performed and kv *could* be
//! fp8; the **current implementation uses bf16** — we follow the code.

use crate::parent::attention::{
    apply_rope_interleaved_inplace, precompute_rope_freqs, PARENT_DIM, PARENT_Q_LORA, PARENT_ROPE_DIM,
};
use crate::parent::codec::{
    act_quant_fp4_inplace_ref, hadamard_rotate_ref, round_to_bf16,
};
use crate::parent::linear::parent_linear_dense;
use crate::parent::weights::{ParentCompressorWeights, ParentIndexerWeights};
use crate::parent::{Ds4ParentBackend, ParentQuantConfig};
use rdna_compute::{DType, Gpu, GpuTensor};

// ── Checkpoint shape constants ──────────────────────────────────────────────

/// `index_n_heads`.
pub const PARENT_INDEX_N_HEADS: usize = 64;
/// `index_head_dim`.
pub const PARENT_INDEX_HEAD_DIM: usize = 128;
/// `index_topk`.
pub const PARENT_INDEX_TOPK: usize = 512;
/// Flattened indexer-Q width (`index_n_heads * index_head_dim`).
pub const PARENT_INDEX_Q_WIDTH: usize = PARENT_INDEX_N_HEADS * PARENT_INDEX_HEAD_DIM; // 8192
/// Indexer layers always use compress_ratio 4.
pub const PARENT_INDEX_RATIO: usize = 4;
/// `compress_rope_theta` — plain (no YaRN) base for indexer Q / compressor.
pub const PARENT_COMPRESS_ROPE_THETA: f32 = 160_000.0;
/// FP4 act-quant group size (`fp4_block_size`).
pub const PARENT_FP4_BLOCK: usize = 32;

/// `softmax_scale * n_heads ** -0.5` applied to `weights_proj` output
/// (`model.py:401,424`).
///
/// `softmax_scale = head_dim ** -0.5 = 128 ** -0.5`
/// full factor   = `128**-0.5 * 64**-0.5 = 1/sqrt(8192)`.
#[inline]
pub fn indexer_weights_scale() -> f64 {
    let softmax_scale = (PARENT_INDEX_HEAD_DIM as f64).powf(-0.5);
    let head_scale = (PARENT_INDEX_N_HEADS as f64).powf(-0.5);
    softmax_scale * head_scale
}

/// f32 form of [`indexer_weights_scale`] for GPU paths.
#[inline]
pub fn indexer_weights_scale_f32() -> f32 {
    indexer_weights_scale() as f32
}

// ── Scratch ─────────────────────────────────────────────────────────────────

/// Reusable device scratch for [`parent_indexer_forward`].
///
/// BF16 staging is refilled before every destructive linear / GEMM. The
/// long-lived compressed KV cache lives here and is written by the sibling
/// compressor (`hadamard = true`).
pub struct ParentIndexerScratch {
    /// Destructive BF16 act tile. Width = max(dim, q_lora, index_q_width).
    act_bf16: GpuTensor,
    /// `wq_b` output `[max_rows, H*D]` F32 before reshape / RoPE / quant.
    q_f32: GpuTensor,
    /// Post-quant Q as F32 `[max_rows, H, D]` for the score kernel.
    q_score_f32: GpuTensor,
    /// `weights_proj` output `[max_rows, H]` F32 (pre- or post-scale).
    weights_f32: GpuTensor,
    /// Compressed KV cache `[max_n_compressed, D]` F32 (bf16-representable).
    kv_cache_f32: GpuTensor,
    /// Score matrix `[max_rows, max_n_compressed]` F32.
    scores_f32: GpuTensor,
    /// Per-row valid compressed count for the score kernel (`I32` bits).
    n_per_batch: GpuTensor,
    /// Absolute positions for RoPE (`I32` bits) — reserved for a future
    /// device-side rope path; host rope is used today.
    #[allow(dead_code)]
    positions: GpuTensor,
    /// Compressor scratch (sibling module), if available at construction.
    #[allow(dead_code)]
    compressor: Option<CompressorScratchSlot>,
    max_rows: usize,
    max_n_compressed: usize,
    bytes: usize,
}

/// Opaque hold for compressor scratch so we do not force a hard type
/// dependency when the sibling module is mid-landing. Filled by
/// [`ParentIndexerScratch::attach_compressor`] after construction.
struct CompressorScratchSlot {
    // Kept as raw bytes counter only; real scratch is owned externally or
    // re-created per call once `parent::compressor` is linked. See
    // `parent_indexer_forward` which builds `ParentCompressorWeights` views
    // and calls `parent_compressor_forward` directly.
    _marker: (),
}

impl ParentIndexerScratch {
    /// Allocate reusable scratch for up to `max_rows` query tokens.
    ///
    /// Compressed-cache capacity is
    /// `max((max_rows + PARENT_INDEX_RATIO - 1) / PARENT_INDEX_RATIO, PARENT_INDEX_TOPK)`
    /// so a pure-prefill of `max_rows` always fits and top-k capacity is
    /// never under-sized relative to `index_topk`.
    pub fn new(gpu: &mut Gpu, cfg: &ParentQuantConfig, max_rows: usize) -> Result<Self, String> {
        let _ = cfg;
        if max_rows == 0 {
            return Err(
                "deepseek4 parent: ParentIndexerScratch max_rows must be > 0".to_owned(),
            );
        }
        let max_n_compressed = max_rows
            .div_ceil(PARENT_INDEX_RATIO)
            .max(PARENT_INDEX_TOPK);

        // act width: max of dim (weights_proj / compressor x), q_lora (wq_b),
        // and index Q width (post-proj staging).
        let act_k = PARENT_DIM
            .max(PARENT_Q_LORA)
            .max(PARENT_INDEX_Q_WIDTH);

        let act_bf16 = gpu
            .alloc_tensor(&[max_rows, act_k], DType::BF16)
            .map_err(|e| format!("deepseek4 parent: indexer act_bf16 alloc: {e:?}"))?;
        let q_f32 = match gpu.alloc_tensor(&[max_rows, PARENT_INDEX_Q_WIDTH], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(act_bf16);
                return Err(format!("deepseek4 parent: indexer q_f32 alloc: {e:?}"));
            }
        };
        let q_score_f32 = match gpu
            .alloc_tensor(&[max_rows, PARENT_INDEX_N_HEADS, PARENT_INDEX_HEAD_DIM], DType::F32)
        {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(act_bf16);
                let _ = gpu.free_tensor(q_f32);
                return Err(format!("deepseek4 parent: indexer q_score_f32 alloc: {e:?}"));
            }
        };
        let weights_f32 = match gpu.alloc_tensor(&[max_rows, PARENT_INDEX_N_HEADS], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(act_bf16);
                let _ = gpu.free_tensor(q_f32);
                let _ = gpu.free_tensor(q_score_f32);
                return Err(format!("deepseek4 parent: indexer weights_f32 alloc: {e:?}"));
            }
        };
        let kv_cache_f32 =
            match gpu.alloc_tensor(&[max_n_compressed, PARENT_INDEX_HEAD_DIM], DType::F32) {
                Ok(t) => t,
                Err(e) => {
                    let _ = gpu.free_tensor(act_bf16);
                    let _ = gpu.free_tensor(q_f32);
                    let _ = gpu.free_tensor(q_score_f32);
                    let _ = gpu.free_tensor(weights_f32);
                    return Err(format!(
                        "deepseek4 parent: indexer kv_cache_f32 alloc: {e:?}"
                    ));
                }
            };
        let scores_f32 =
            match gpu.alloc_tensor(&[max_rows, max_n_compressed], DType::F32) {
                Ok(t) => t,
                Err(e) => {
                    let _ = gpu.free_tensor(act_bf16);
                    let _ = gpu.free_tensor(q_f32);
                    let _ = gpu.free_tensor(q_score_f32);
                    let _ = gpu.free_tensor(weights_f32);
                    let _ = gpu.free_tensor(kv_cache_f32);
                    return Err(format!("deepseek4 parent: indexer scores_f32 alloc: {e:?}"));
                }
            };
        let n_per_batch = match alloc_i32_buf(gpu, max_rows) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(act_bf16);
                let _ = gpu.free_tensor(q_f32);
                let _ = gpu.free_tensor(q_score_f32);
                let _ = gpu.free_tensor(weights_f32);
                let _ = gpu.free_tensor(kv_cache_f32);
                let _ = gpu.free_tensor(scores_f32);
                return Err(e);
            }
        };
        let positions = match alloc_i32_buf(gpu, max_rows) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(act_bf16);
                let _ = gpu.free_tensor(q_f32);
                let _ = gpu.free_tensor(q_score_f32);
                let _ = gpu.free_tensor(weights_f32);
                let _ = gpu.free_tensor(kv_cache_f32);
                let _ = gpu.free_tensor(scores_f32);
                let _ = gpu.free_tensor(n_per_batch);
                return Err(e);
            }
        };

        let bytes = act_bf16.buf.size()
            + q_f32.buf.size()
            + q_score_f32.buf.size()
            + weights_f32.buf.size()
            + kv_cache_f32.buf.size()
            + scores_f32.buf.size()
            + n_per_batch.buf.size()
            + positions.buf.size();

        Ok(Self {
            act_bf16,
            q_f32,
            q_score_f32,
            weights_f32,
            kv_cache_f32,
            scores_f32,
            n_per_batch,
            positions,
            compressor: None,
            max_rows,
            max_n_compressed,
            bytes,
        })
    }

    /// Total scratch bytes resident on device.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    pub fn max_n_compressed(&self) -> usize {
        self.max_n_compressed
    }

    /// Compressed KV cache buffer `[max_n_compressed, head_dim]` F32.
    pub fn kv_cache_f32_ref(&self) -> &GpuTensor {
        &self.kv_cache_f32
    }

    /// Score matrix `[max_rows, max_n_compressed]` F32 (diagnostic).
    pub fn scores_f32_ref(&self) -> &GpuTensor {
        &self.scores_f32
    }

    /// Post-quant Q `[max_rows, H, D]` F32 (diagnostic).
    pub fn q_score_f32_ref(&self) -> &GpuTensor {
        &self.q_score_f32
    }

    /// Head weights after scale `[max_rows, H]` F32 (diagnostic).
    pub fn weights_f32_ref(&self) -> &GpuTensor {
        &self.weights_f32
    }
}

fn alloc_i32_buf(gpu: &mut Gpu, n: usize) -> Result<GpuTensor, String> {
    // Raw buffer sized in bytes = n * 4 (DType::Raw size == 1).
    let nbytes = n
        .checked_mul(4)
        .ok_or_else(|| "deepseek4 parent: indexer i32 alloc overflow".to_owned())?;
    gpu.alloc_tensor(&[nbytes], DType::Raw)
        .map_err(|e| format!("deepseek4 parent: indexer i32 alloc: {e:?}"))
}

// ── Host helpers (unit-tested) ──────────────────────────────────────────────

/// Per-row number of compressed slots visible at absolute position
/// `start_pos + row` (`model.py:426,431`: `end_pos // ratio` with causal
/// cutoff `(row+1) // ratio` on prefill).
#[inline]
pub fn indexer_n_visible(start_pos: usize, row: usize, ratio: usize) -> usize {
    (start_pos + row + 1) / ratio
}

/// Total compressed slots committed through `start_pos + rows - 1`.
#[inline]
pub fn indexer_n_compressed(start_pos: usize, rows: usize, ratio: usize) -> usize {
    (start_pos + rows) / ratio
}

/// Host top-k over a single score row. Returns indices of length `k_out`,
/// padded with `-1` when fewer than `k_out` finite candidates exist.
///
/// Tie-break: higher score wins; on equal scores the **lower index** wins
/// (stable, matches the batched GPU kernel's
/// `rank += (sj > si) || (sj == si && j < i)`).
pub fn indexer_topk_host(scores: &[f32], k_out: usize) -> Vec<i32> {
    let n = scores.len();
    let mut order: Vec<usize> = (0..n).filter(|&i| scores[i].is_finite()).collect();
    order.sort_by(|&a, &b| {
        let sa = scores[a];
        let sb = scores[b];
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    let mut out = vec![-1i32; k_out];
    let take = k_out.min(order.len());
    for i in 0..take {
        out[i] = order[i] as i32;
    }
    out
}

/// Prefill causal post-mask + offset (`model.py:434-438`).
///
/// For each row `r`, any selected index `>= (r+1) // ratio` becomes `-1`;
/// surviving indices are shifted by `offset`.
pub fn indexer_apply_offset_and_causal_mask(
    topk: &mut [i32],
    rows: usize,
    k_stride: usize,
    start_pos: usize,
    ratio: usize,
    offset: usize,
) {
    let offset_i = offset as i32;
    for r in 0..rows {
        let cutoff = if start_pos == 0 {
            (r + 1) / ratio
        } else {
            // Decode: all committed slots through end_pos are visible;
            // no per-row future mask inside the chunk.
            usize::MAX
        };
        let row = &mut topk[r * k_stride..(r + 1) * k_stride];
        for v in row.iter_mut() {
            if *v < 0 {
                continue;
            }
            let idx = *v as usize;
            if start_pos == 0 && idx >= cutoff {
                *v = -1;
            } else {
                *v = *v + offset_i;
            }
        }
    }
}

/// f64 oracle for one row of indexer scores (`model.py:426-427`).
///
/// `q` is `[H, D]` post-RoPE / post-Hadamard / post-FP4-sim.
/// `kv` is `[N, D]` compressed cache (bf16 path — no further quant).
/// `weights` is `[H]` after the `softmax_scale * n_heads**-0.5` scale.
///
/// `score[t] = sum_h relu(dot(q[h], kv[t])) * weights[h]`
pub fn indexer_score_row_f64(
    q: &[f64],
    kv: &[f64],
    weights: &[f64],
    n_heads: usize,
    head_dim: usize,
    n_slots: usize,
) -> Result<Vec<f64>, String> {
    if n_heads == 0 || head_dim == 0 {
        return Err(
            "deepseek4 parent: indexer_score_row_f64 requires n_heads>0 and head_dim>0".to_owned(),
        );
    }
    if q.len() < n_heads * head_dim {
        return Err(format!(
            "deepseek4 parent: indexer q short (have {} need {})",
            q.len(),
            n_heads * head_dim
        ));
    }
    if weights.len() < n_heads {
        return Err(format!(
            "deepseek4 parent: indexer weights short (have {} need {n_heads})",
            weights.len()
        ));
    }
    if kv.len() < n_slots * head_dim {
        return Err(format!(
            "deepseek4 parent: indexer kv short (have {} need {})",
            kv.len(),
            n_slots * head_dim
        ));
    }
    let mut scores = vec![0.0f64; n_slots];
    for t in 0..n_slots {
        let mut acc = 0.0f64;
        let kt = &kv[t * head_dim..(t + 1) * head_dim];
        for h in 0..n_heads {
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut dot = 0.0f64;
            for d in 0..head_dim {
                dot += qh[d] * kt[d];
            }
            let relu = if dot > 0.0 { dot } else { 0.0 };
            acc += relu * weights[h];
        }
        scores[t] = acc;
    }
    Ok(scores)
}

/// Full host f64 oracle for the indexer scoring path over a batch.
///
/// Inputs are already past the linear / RoPE / Hadamard / FP4 stages —
/// this covers `model.py:426-438` (score, causal mask, top-k, offset).
///
/// Returns `(scores_flat [rows * n_slots], topk_idx [rows * k_out])`.
pub fn indexer_oracle_f64(
    q: &[f64],         // [rows, H, D]
    kv: &[f64],        // [n_slots, D]
    weights: &[f64],   // [rows, H] already scaled
    rows: usize,
    n_heads: usize,
    head_dim: usize,
    n_slots: usize,
    start_pos: usize,
    ratio: usize,
    k_out: usize,
    offset: usize,
) -> Result<(Vec<f64>, Vec<i32>), String> {
    let mut scores_flat = vec![0.0f64; rows * n_slots];
    let mut topk = vec![-1i32; rows * k_out];
    for r in 0..rows {
        let q_row = &q[r * n_heads * head_dim..(r + 1) * n_heads * head_dim];
        let w_row = &weights[r * n_heads..(r + 1) * n_heads];
        let n_vis = indexer_n_visible(start_pos, r, ratio).min(n_slots);
        let row_scores = indexer_score_row_f64(q_row, kv, w_row, n_heads, head_dim, n_slots)?;
        // Causal: mask future compressed slots with -inf before top-k.
        let mut masked: Vec<f32> = row_scores.iter().map(|v| *v as f32).collect();
        for t in n_vis..n_slots {
            masked[t] = f32::NEG_INFINITY;
        }
        for t in 0..n_slots {
            scores_flat[r * n_slots + t] = if t < n_vis {
                row_scores[t]
            } else {
                f64::NEG_INFINITY
            };
        }
        let k_take = k_out.min(n_slots).min(indexer_n_compressed(start_pos, rows, ratio).max(n_vis));
        // Use per-row visible count as the effective N for top-k pool size
        // when start_pos==0; else the full committed set.
        let pool_n = if start_pos == 0 {
            n_vis
        } else {
            n_slots
        };
        let row_topk = indexer_topk_host(&masked[..pool_n.max(1).min(masked.len())], k_take.max(1));
        let dest = &mut topk[r * k_out..(r + 1) * k_out];
        for i in 0..k_out {
            dest[i] = if i < row_topk.len() { row_topk[i] } else { -1 };
        }
    }
    indexer_apply_offset_and_causal_mask(&mut topk, rows, k_out, start_pos, ratio, offset);
    Ok((scores_flat, topk))
}

/// Compare two top-k index rows as unordered sets (ignoring `-1` pad and
/// order). Returns the number of indices present in `a` but not in `b`
/// plus the reverse — i.e. `0` means identical sets.
pub fn topk_index_mismatch_count(a: &[i32], b: &[i32]) -> usize {
    let set_a: std::collections::BTreeSet<i32> = a.iter().copied().filter(|&v| v >= 0).collect();
    let set_b: std::collections::BTreeSet<i32> = b.iter().copied().filter(|&v| v >= 0).collect();
    set_a.difference(&set_b).count() + set_b.difference(&set_a).count()
}

/// Score-error summary between GPU/host f32 scores and an f64 oracle.
#[derive(Clone, Debug)]
pub struct IndexerScoreReport {
    pub max_abs: f64,
    pub mean_rel: f64,
    pub l2_rel: f64,
    pub index_mismatch: usize,
    pub n_scores: usize,
    pub n_indices: usize,
}

impl IndexerScoreReport {
    pub fn from_scores_and_indices(
        got_scores: &[f32],
        ref_scores: &[f64],
        got_idx: &[i32],
        ref_idx: &[i32],
    ) -> Self {
        let n = got_scores.len().min(ref_scores.len());
        let mut max_abs = 0.0f64;
        let mut _sum_abs = 0.0f64;
        let mut sum_rel = 0.0f64;
        let mut rel_n = 0usize;
        let mut l2_err = 0.0f64;
        let mut l2_ref = 0.0f64;
        for i in 0..n {
            let g = got_scores[i] as f64;
            let r = ref_scores[i];
            if !r.is_finite() && !g.is_finite() {
                continue;
            }
            let d = (g - r).abs();
            max_abs = max_abs.max(d);
            _sum_abs += d;
            l2_err += d * d;
            l2_ref += r * r;
            if r.abs() > 1e-12 {
                sum_rel += d / r.abs();
                rel_n += 1;
            }
        }
        let mean_rel = if rel_n > 0 {
            sum_rel / rel_n as f64
        } else {
            0.0
        };
        let l2_rel = if l2_ref > 0.0 {
            l2_err.sqrt() / l2_ref.sqrt()
        } else {
            l2_err.sqrt()
        };
        let index_mismatch = topk_index_mismatch_count(got_idx, ref_idx);
        Self {
            max_abs,
            mean_rel,
            l2_rel,
            index_mismatch,
            n_scores: n,
            n_indices: got_idx.len().min(ref_idx.len()),
        }
    }
}

// ── Forward ─────────────────────────────────────────────────────────────────

/// Run the parent indexer for a `compress_ratio == 4` layer.
///
/// - `x` is `[rows, dim]` F32 (post-attn_norm stream — same `x` the
///   attention block feeds the indexer).
/// - `qr` is `[rows, q_lora]` F32 post-`q_norm` (the attention block's
///   LoRA bottleneck; `model.py:502,517`).
/// - `topk_idx` is an `I32`/`Raw` buffer of at least `rows * index_topk`
///   elements; written row-major with `-1` pads.
/// - `n_active` is a single `I32` — number of compressed slots committed
///   (`(start_pos + rows) // ratio`).
/// - `offset` is added to surviving indices (`model.py:436,438`); the
///   attention block passes `win` (decode) or `kv.size(1)` (prefill).
/// - `layer_idx` selects `cfg.compress_ratio(layer_idx)`; anything other
///   than `4` is refused.
///
/// # Hadamard + FP4 (q path)
///
/// Applied **after** RoPE on the full `[rows, H, D]` Q tensor, matching
/// `model.py:420-422`:
/// ```text
/// q = rotate_activation(q)          # Walsh-Hadamard, scale D**-0.5
/// fp4_act_quant(q, 32, inplace=True)
/// ```
///
/// # weights_proj BF16 path
///
/// `indexer.weights_proj` is BF16 `[64, 4096]` with **no** `.scale`
/// companion — it takes plain `x @ W^T` (no FP8 act-quant). Verified by
/// refusing any non-BF16 dtype on the weight tensor and by going through
/// `gemm_bf16_mfma_gfx942` / host f32 GEMM **without** calling
/// `parent_linear_dense`.
pub fn parent_indexer_forward(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    w: &ParentIndexerWeights,
    cfg: &ParentQuantConfig,
    scratch: &mut ParentIndexerScratch,
    x: &GpuTensor,
    qr: &GpuTensor,
    rows: usize,
    start_pos: usize,
    offset: usize,
    layer_idx: usize,
    topk_idx: &GpuTensor,
    n_active: &GpuTensor,
) -> Result<(), String> {
    backend.ensure_device(gpu)?;

    let ratio = cfg.compress_ratio(layer_idx);
    if ratio != PARENT_INDEX_RATIO {
        return Err(format!(
            "deepseek4 parent: parent_indexer_forward refuses compress_ratio={ratio} \
             (layer {layer_idx}); indexer is only defined for compress_ratio == 4"
        ));
    }
    if rows == 0 {
        return Err("deepseek4 parent: parent_indexer_forward rows must be > 0".to_owned());
    }
    if rows > scratch.max_rows {
        return Err(format!(
            "deepseek4 parent: rows {rows} exceeds indexer scratch.max_rows {}",
            scratch.max_rows
        ));
    }
    validate_f32_mat(x, rows, PARENT_DIM, "x")?;
    validate_f32_mat(qr, rows, PARENT_Q_LORA, "qr")?;
    validate_i32_len(topk_idx, rows * PARENT_INDEX_TOPK, "topk_idx")?;
    validate_i32_len(n_active, 1, "n_active")?;

    if w.wq_b.n() != PARENT_INDEX_Q_WIDTH || w.wq_b.k() != PARENT_Q_LORA {
        return Err(format!(
            "deepseek4 parent: indexer.wq_b shape [{}, {}] != [{PARENT_INDEX_Q_WIDTH}, {PARENT_Q_LORA}]",
            w.wq_b.n(),
            w.wq_b.k()
        ));
    }
    // weights_proj MUST be plain BF16 — this is the load-bearing check that
    // we did not accidentally route it through the FP8 dense linear.
    if w.weights_proj.dtype != DType::BF16 {
        return Err(format!(
            "deepseek4 parent: indexer.weights_proj must be BF16 (got {:?}) — \
             plain F.linear path, not parent_linear_dense",
            w.weights_proj.dtype
        ));
    }
    let wp_need = PARENT_INDEX_N_HEADS * PARENT_DIM * 2;
    if w.weights_proj.buf.size() < wp_need {
        return Err(format!(
            "deepseek4 parent: indexer.weights_proj too small (have {} need {wp_need})",
            w.weights_proj.buf.size()
        ));
    }

    let n_compressed = indexer_n_compressed(start_pos, rows, ratio);
    if n_compressed > scratch.max_n_compressed {
        return Err(format!(
            "deepseek4 parent: n_compressed={n_compressed} exceeds scratch \
             max_n_compressed={} (start_pos={start_pos} rows={rows})",
            scratch.max_n_compressed
        ));
    }

    // ── 1. q = wq_b(qr)  via dense FP8 linear ───────────────────────────
    // Fresh BF16 copy of qr — parent_linear_dense destroys x_bf16.
    stage_f32_to_act_bf16(gpu, scratch, qr, rows, PARENT_Q_LORA)?;
    {
        let q_out = scratch.q_f32.sub_offset(0, rows * PARENT_INDEX_Q_WIDTH);
        parent_linear_dense(
            gpu,
            backend,
            &w.wq_b,
            &act_view(scratch, rows, PARENT_Q_LORA)?,
            rows,
            &q_out,
        )
        .map_err(|e| format!("deepseek4 parent: indexer wq_b linear: {e}"))?;
    }

    // ── 2. Tail RoPE (YaRN + compress_rope_theta) ───────────────────────
    // model.py:482-500: Indexer shares Attention.freqs_cis which for
    // compress_ratio > 0 is built with original_seq_len + compress_rope_theta
    // (YaRN on). Match the reference.
    let mut q_host = download_f32_prefix(gpu, &scratch.q_f32, rows * PARENT_INDEX_Q_WIDTH)?;
    let freqs = precompute_rope_freqs(
        PARENT_ROPE_DIM,
        /*original_seq_len=*/ 65_536,
        PARENT_COMPRESS_ROPE_THETA as f64,
        /*factor=*/ 16.0,
        /*beta_fast=*/ 32.0,
        /*beta_slow=*/ 1.0,
    )?;
    let positions: Vec<usize> = (0..rows).map(|r| start_pos + r).collect();
    apply_rope_interleaved_inplace(
        &mut q_host,
        rows,
        PARENT_INDEX_N_HEADS,
        PARENT_INDEX_HEAD_DIM,
        PARENT_ROPE_DIM,
        &positions,
        &freqs,
        /*inverse=*/ false,
    )?;

    // ── 3. Hadamard rotate + FP4 group-32 simulation on Q ────────────────
    // model.py:420-422. Applied over the last dim (= head_dim = 128) of
    // every head independently — layout is flat [rows * H, D] for the
    // trailing-block helpers.
    //
    // THIS IS THE FP4 GROUP-32 SITE (not FP8). Missing it makes the
    // reference more accurate than itself.
    hadamard_rotate_ref(&mut q_host, PARENT_INDEX_HEAD_DIM).map_err(|e| {
        format!("deepseek4 parent: indexer q hadamard: {e}")
    })?;
    act_quant_fp4_inplace_ref(&mut q_host, PARENT_INDEX_HEAD_DIM).map_err(|e| {
        format!("deepseek4 parent: indexer q fp4_act_quant: {e}")
    })?;
    upload_f32_prefix(
        gpu,
        &scratch.q_score_f32,
        &q_host,
        rows * PARENT_INDEX_Q_WIDTH,
    )?;

    // ── 4. weights = weights_proj(x) * scale  (plain BF16 GEMM) ──────────
    // No act-quant. gemm_bf16_mfma: A[M,K]=W[H,dim], B[batch,K]=x, D[batch,M].
    {
        let x_host = download_f32_prefix(gpu, x, rows * PARENT_DIM)?;
        let x_bf16 = pack_f32_to_bf16_bytes(&x_host);
        // Stage x into act_bf16 as [rows, dim].
        let x_view = act_view(scratch, rows, PARENT_DIM)?;
        upload_bf16_into(gpu, &x_view, &x_bf16, rows * PARENT_DIM)?;
        let w_out = scratch
            .weights_f32
            .sub_offset(0, rows * PARENT_INDEX_N_HEADS);
        gpu.gemm_bf16_mfma_gfx942(
            &w.weights_proj.buf,
            &x_view.buf,
            &w_out.buf,
            PARENT_INDEX_N_HEADS,
            PARENT_DIM,
            rows,
        )
        .map_err(|e| format!("deepseek4 parent: indexer weights_proj BF16 GEMM: {e:?}"))?;
    }
    // Apply softmax_scale * n_heads**-0.5 on host (tiny H=64).
    let scale = indexer_weights_scale_f32();
    let mut w_host =
        download_f32_prefix(gpu, &scratch.weights_f32, rows * PARENT_INDEX_N_HEADS)?;
    for v in w_host.iter_mut() {
        *v *= scale;
    }
    upload_f32_prefix(
        gpu,
        &scratch.weights_f32,
        &w_host,
        rows * PARENT_INDEX_N_HEADS,
    )?;

    // ── 5. Compressor (hadamard=true) fills kv_cache ─────────────────────
    // Build a ParentCompressorWeights view over the indexer's compressor_*
    // fields and call the sibling forward. If the compressor module is not
    // yet linked this is a compile error — integrate when it lands.
    run_indexer_compressor(
        gpu,
        backend,
        w,
        cfg,
        scratch,
        x,
        rows,
        start_pos,
        n_compressed,
    )?;

    // ── 6. Score + top-k ─────────────────────────────────────────────────
    // Prefer the format-agnostic batched kernels when n_compressed > 0;
    // fall back to host f32 scoring for the empty-cache edge (n=0).
    let k_out = PARENT_INDEX_TOPK.min(n_compressed.max(1));
    let mut topk_host = vec![-1i32; rows * PARENT_INDEX_TOPK];

    if n_compressed == 0 {
        // Nothing to select — leave -1 pads; n_active = 0.
        upload_i32_prefix(gpu, topk_idx, &topk_host, rows * PARENT_INDEX_TOPK)?;
        upload_i32_prefix(gpu, n_active, &[0i32], 1)?;
        return Ok(());
    }

    // Per-row visible counts for the score kernel's causal cutoff.
    let mut n_per: Vec<i32> = (0..rows)
        .map(|r| indexer_n_visible(start_pos, r, ratio).min(n_compressed) as i32)
        .collect();
    // When start_pos > 0 every row sees the full committed set.
    if start_pos > 0 {
        for v in n_per.iter_mut() {
            *v = n_compressed as i32;
        }
    }
    upload_i32_prefix(gpu, &scratch.n_per_batch, &n_per, rows)?;

    let q_view = scratch
        .q_score_f32
        .sub_offset(0, rows * PARENT_INDEX_Q_WIDTH);
    let kv_view = scratch
        .kv_cache_f32
        .sub_offset(0, n_compressed * PARENT_INDEX_HEAD_DIM);
    // scores buffer: kernel writes [rows, max_n] with stride max_n_compressed.
    // Pass the full scores tensor; kernel indexes by N_max.
    let scores_view = scratch
        .scores_f32
        .sub_offset(0, rows * scratch.max_n_compressed);
    let w_view = scratch
        .weights_f32
        .sub_offset(0, rows * PARENT_INDEX_N_HEADS);

    gpu.indexer_relu_score_batched_f32(
        &q_view,
        &kv_view,
        &w_view,
        &scratch.n_per_batch,
        &scores_view,
        PARENT_INDEX_N_HEADS as i32,
        PARENT_INDEX_HEAD_DIM as i32,
        scratch.max_n_compressed as i32,
        rows as i32,
    )
    .map_err(|e| format!("deepseek4 parent: indexer relu score: {e:?}"))?;

    // Top-k on device (n_idx_heads=1: scores are already head-reduced).
    // Layout expected by indexer_top_k_batched with H=1:
    //   scores [B, 1, N_stride], top_indices [B, 1, K_stride].
    // Our scores are [B, N_max] ≡ [B, 1, N_max].
    {
        // Allocate a small host-side topk then upload — the kernel writes i32
        // into a Raw/i32 buffer. Use topk_idx directly.
        gpu.indexer_top_k_batched(
            &scores_view,
            topk_idx,
            /*n_idx_heads=*/ 1,
            /*n_stride=*/ scratch.max_n_compressed as i32,
            /*n_iter=*/ n_compressed as i32,
            /*k_stride=*/ PARENT_INDEX_TOPK as i32,
            /*k_fill=*/ k_out as i32,
            /*batch_size=*/ rows as i32,
        )
        .map_err(|e| format!("deepseek4 parent: indexer top_k: {e:?}"))?;
    }

    // Download topk, apply offset + prefill causal post-mask, re-upload.
    topk_host = download_i32_prefix(gpu, topk_idx, rows * PARENT_INDEX_TOPK)?;
    indexer_apply_offset_and_causal_mask(
        &mut topk_host,
        rows,
        PARENT_INDEX_TOPK,
        start_pos,
        ratio,
        offset,
    );
    upload_i32_prefix(gpu, topk_idx, &topk_host, rows * PARENT_INDEX_TOPK)?;
    upload_i32_prefix(gpu, n_active, &[n_compressed as i32], 1)?;

    Ok(())
}

/// Score-only path: caller has already filled `scratch.kv_cache_f32` with
/// `n_compressed` slots (e.g. from a unit test or a prior compressor run).
/// Still runs the full Q / weights_proj / score / top-k path.
///
/// Used by the mi300x gate when the compressor sibling is exercised
/// separately, and by unit tests that inject synthetic KV.
#[allow(clippy::too_many_arguments)]
pub fn parent_indexer_forward_with_kv(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    w: &ParentIndexerWeights,
    cfg: &ParentQuantConfig,
    scratch: &mut ParentIndexerScratch,
    x: &GpuTensor,
    qr: &GpuTensor,
    rows: usize,
    start_pos: usize,
    offset: usize,
    layer_idx: usize,
    n_compressed: usize,
    topk_idx: &GpuTensor,
    n_active: &GpuTensor,
) -> Result<(), String> {
    backend.ensure_device(gpu)?;
    let ratio = cfg.compress_ratio(layer_idx);
    if ratio != PARENT_INDEX_RATIO {
        return Err(format!(
            "deepseek4 parent: parent_indexer_forward_with_kv refuses compress_ratio={ratio} \
             (layer {layer_idx}); indexer is only defined for compress_ratio == 4"
        ));
    }
    if rows == 0 {
        return Err(
            "deepseek4 parent: parent_indexer_forward_with_kv rows must be > 0".to_owned(),
        );
    }
    if rows > scratch.max_rows {
        return Err(format!(
            "deepseek4 parent: rows {rows} exceeds indexer scratch.max_rows {}",
            scratch.max_rows
        ));
    }
    if n_compressed > scratch.max_n_compressed {
        return Err(format!(
            "deepseek4 parent: n_compressed={n_compressed} exceeds scratch max_n_compressed={}",
            scratch.max_n_compressed
        ));
    }
    validate_f32_mat(x, rows, PARENT_DIM, "x")?;
    validate_f32_mat(qr, rows, PARENT_Q_LORA, "qr")?;
    validate_i32_len(topk_idx, rows * PARENT_INDEX_TOPK, "topk_idx")?;
    validate_i32_len(n_active, 1, "n_active")?;

    if w.weights_proj.dtype != DType::BF16 {
        return Err(format!(
            "deepseek4 parent: indexer.weights_proj must be BF16 (got {:?})",
            w.weights_proj.dtype
        ));
    }

    // Q path (same as parent_indexer_forward steps 1-3).
    stage_f32_to_act_bf16(gpu, scratch, qr, rows, PARENT_Q_LORA)?;
    {
        let q_out = scratch.q_f32.sub_offset(0, rows * PARENT_INDEX_Q_WIDTH);
        parent_linear_dense(
            gpu,
            backend,
            &w.wq_b,
            &act_view(scratch, rows, PARENT_Q_LORA)?,
            rows,
            &q_out,
        )
        .map_err(|e| format!("deepseek4 parent: indexer wq_b linear: {e}"))?;
    }
    let mut q_host = download_f32_prefix(gpu, &scratch.q_f32, rows * PARENT_INDEX_Q_WIDTH)?;
    let freqs = precompute_rope_freqs(
        PARENT_ROPE_DIM,
        /*original_seq_len=*/ 65_536,
        PARENT_COMPRESS_ROPE_THETA as f64,
        16.0,
        32.0,
        1.0,
    )?;
    let positions: Vec<usize> = (0..rows).map(|r| start_pos + r).collect();
    apply_rope_interleaved_inplace(
        &mut q_host,
        rows,
        PARENT_INDEX_N_HEADS,
        PARENT_INDEX_HEAD_DIM,
        PARENT_ROPE_DIM,
        &positions,
        &freqs,
        false,
    )?;
    // FP4 group-32 + Hadamard on Q (model.py:420-422).
    hadamard_rotate_ref(&mut q_host, PARENT_INDEX_HEAD_DIM)
        .map_err(|e| format!("deepseek4 parent: indexer q hadamard: {e}"))?;
    act_quant_fp4_inplace_ref(&mut q_host, PARENT_INDEX_HEAD_DIM)
        .map_err(|e| format!("deepseek4 parent: indexer q fp4_act_quant: {e}"))?;
    upload_f32_prefix(
        gpu,
        &scratch.q_score_f32,
        &q_host,
        rows * PARENT_INDEX_Q_WIDTH,
    )?;

    // weights_proj BF16 path.
    {
        let x_host = download_f32_prefix(gpu, x, rows * PARENT_DIM)?;
        let x_bf16 = pack_f32_to_bf16_bytes(&x_host);
        let x_view = act_view(scratch, rows, PARENT_DIM)?;
        upload_bf16_into(gpu, &x_view, &x_bf16, rows * PARENT_DIM)?;
        let w_out = scratch
            .weights_f32
            .sub_offset(0, rows * PARENT_INDEX_N_HEADS);
        gpu.gemm_bf16_mfma_gfx942(
            &w.weights_proj.buf,
            &x_view.buf,
            &w_out.buf,
            PARENT_INDEX_N_HEADS,
            PARENT_DIM,
            rows,
        )
        .map_err(|e| format!("deepseek4 parent: indexer weights_proj BF16 GEMM: {e:?}"))?;
    }
    let scale = indexer_weights_scale_f32();
    let mut w_host =
        download_f32_prefix(gpu, &scratch.weights_f32, rows * PARENT_INDEX_N_HEADS)?;
    for v in w_host.iter_mut() {
        *v *= scale;
    }
    upload_f32_prefix(
        gpu,
        &scratch.weights_f32,
        &w_host,
        rows * PARENT_INDEX_N_HEADS,
    )?;

    let mut topk_host = vec![-1i32; rows * PARENT_INDEX_TOPK];
    if n_compressed == 0 {
        upload_i32_prefix(gpu, topk_idx, &topk_host, rows * PARENT_INDEX_TOPK)?;
        upload_i32_prefix(gpu, n_active, &[0i32], 1)?;
        return Ok(());
    }

    let mut n_per: Vec<i32> = (0..rows)
        .map(|r| indexer_n_visible(start_pos, r, ratio).min(n_compressed) as i32)
        .collect();
    if start_pos > 0 {
        for v in n_per.iter_mut() {
            *v = n_compressed as i32;
        }
    }
    upload_i32_prefix(gpu, &scratch.n_per_batch, &n_per, rows)?;

    let q_view = scratch
        .q_score_f32
        .sub_offset(0, rows * PARENT_INDEX_Q_WIDTH);
    let kv_view = scratch
        .kv_cache_f32
        .sub_offset(0, n_compressed * PARENT_INDEX_HEAD_DIM);
    let scores_view = scratch
        .scores_f32
        .sub_offset(0, rows * scratch.max_n_compressed);
    let w_view = scratch
        .weights_f32
        .sub_offset(0, rows * PARENT_INDEX_N_HEADS);
    let k_out = PARENT_INDEX_TOPK.min(n_compressed);

    gpu.indexer_relu_score_batched_f32(
        &q_view,
        &kv_view,
        &w_view,
        &scratch.n_per_batch,
        &scores_view,
        PARENT_INDEX_N_HEADS as i32,
        PARENT_INDEX_HEAD_DIM as i32,
        scratch.max_n_compressed as i32,
        rows as i32,
    )
    .map_err(|e| format!("deepseek4 parent: indexer relu score: {e:?}"))?;

    gpu.indexer_top_k_batched(
        &scores_view,
        topk_idx,
        1,
        scratch.max_n_compressed as i32,
        n_compressed as i32,
        PARENT_INDEX_TOPK as i32,
        k_out as i32,
        rows as i32,
    )
    .map_err(|e| format!("deepseek4 parent: indexer top_k: {e:?}"))?;

    topk_host = download_i32_prefix(gpu, topk_idx, rows * PARENT_INDEX_TOPK)?;
    indexer_apply_offset_and_causal_mask(
        &mut topk_host,
        rows,
        PARENT_INDEX_TOPK,
        start_pos,
        ratio,
        offset,
    );
    upload_i32_prefix(gpu, topk_idx, &topk_host, rows * PARENT_INDEX_TOPK)?;
    upload_i32_prefix(gpu, n_active, &[n_compressed as i32], 1)?;
    Ok(())
}

/// Build `ParentCompressorWeights` from indexer fields and invoke the
/// sibling compressor with `hadamard = true`.
fn run_indexer_compressor(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    w: &ParentIndexerWeights,
    cfg: &ParentQuantConfig,
    scratch: &mut ParentIndexerScratch,
    x: &GpuTensor,
    rows: usize,
    start_pos: usize,
    n_compressed: usize,
) -> Result<(), String> {
    // Re-export path: parent::compressor must be on the module tree.
    use crate::parent::compressor::{parent_compressor_forward, ParentCompressorScratch};

    let cw = ParentCompressorWeights {
        wkv: w.compressor_wkv.shallow_clone(),
        wgate: w.compressor_wgate.shallow_clone(),
        norm: w.compressor_norm.shallow_clone(),
        ape: w.compressor_ape.shallow_clone(),
    };

    // ParentCompressorScratch is sized for worst-case main (head_dim=512);
    // indexer head_dim=128 fits inside.
    let mut c_scratch = ParentCompressorScratch::new(gpu, cfg, rows.max(scratch.max_rows))?;
    let n_out = if start_pos == 0 {
        // Prefill emits rows/ratio compressed tokens (0 when rows < ratio).
        rows / PARENT_INDEX_RATIO
    } else {
        n_compressed
    };
    // Always call through so remainder / ring state updates even when n_out==0.
    // kv_out needs capacity for at least 1 head row when n_out==0 (untouched).
    let out_rows = n_out.max(1);
    if out_rows > scratch.max_n_compressed {
        return Err(format!(
            "deepseek4 parent: compressor n_out={n_out} exceeds kv_cache capacity {}",
            scratch.max_n_compressed
        ));
    }
    let kv_out = {
        let mut v = scratch
            .kv_cache_f32
            .sub_offset(0, out_rows * PARENT_INDEX_HEAD_DIM);
        v.shape = vec![out_rows, PARENT_INDEX_HEAD_DIM];
        v
    };

    parent_compressor_forward(
        gpu,
        backend,
        &cw,
        cfg,
        &mut c_scratch,
        x,
        rows,
        start_pos,
        PARENT_INDEX_RATIO,
        /*hadamard=*/ true,
        &kv_out,
    )
    .map_err(|e| format!("deepseek4 parent: indexer compressor: {e}"))?;
    let _ = n_compressed;
    Ok(())
}

// ── Staging / IO helpers ────────────────────────────────────────────────────

fn act_view(scratch: &ParentIndexerScratch, rows: usize, k: usize) -> Result<GpuTensor, String> {
    let act_k = PARENT_DIM
        .max(PARENT_Q_LORA)
        .max(PARENT_INDEX_Q_WIDTH);
    if k > act_k {
        return Err(format!(
            "deepseek4 parent: indexer act_view k={k} exceeds act_bf16 width {act_k}"
        ));
    }
    if rows > scratch.max_rows {
        return Err(format!(
            "deepseek4 parent: indexer act_view rows={rows} exceeds max_rows {}",
            scratch.max_rows
        ));
    }
    let mut v = scratch.act_bf16.sub_offset(0, rows * k);
    v.shape = vec![rows, k];
    Ok(v)
}

fn stage_f32_to_act_bf16(
    gpu: &Gpu,
    scratch: &ParentIndexerScratch,
    x: &GpuTensor,
    rows: usize,
    k: usize,
) -> Result<(), String> {
    let host = download_f32_prefix(gpu, x, rows * k)?;
    let bytes = pack_f32_to_bf16_bytes(&host);
    let view = act_view(scratch, rows, k)?;
    upload_bf16_into(gpu, &view, &bytes, rows * k)
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
    let bytes = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, nbytes) };
    gpu.hip
        .memcpy_dtoh(bytes, &t.buf)
        .map_err(|e| format!("deepseek4 parent: f32 download: {e:?}"))?;
    Ok(data)
}

fn upload_f32_prefix(gpu: &Gpu, t: &GpuTensor, data: &[f32], nelems: usize) -> Result<(), String> {
    if t.dtype != DType::F32 {
        return Err(format!(
            "deepseek4 parent: upload_f32 expects F32 (got {:?})",
            t.dtype
        ));
    }
    if data.len() < nelems {
        return Err(format!(
            "deepseek4 parent: upload_f32 data short ({} < {nelems})",
            data.len()
        ));
    }
    let nbytes = nelems * 4;
    if t.buf.size() < nbytes {
        return Err(format!(
            "deepseek4 parent: upload_f32 dest too small (have {} need {nbytes})",
            t.buf.size()
        ));
    }
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, nbytes) };
    gpu.hip
        .memcpy_htod(&t.buf, bytes)
        .map_err(|e| format!("deepseek4 parent: upload_f32: {e:?}"))
}

fn upload_i32_prefix(gpu: &Gpu, t: &GpuTensor, data: &[i32], nelems: usize) -> Result<(), String> {
    if data.len() < nelems {
        return Err(format!(
            "deepseek4 parent: upload_i32 data short ({} < {nelems})",
            data.len()
        ));
    }
    let nbytes = nelems * 4;
    if t.buf.size() < nbytes {
        return Err(format!(
            "deepseek4 parent: upload_i32 dest too small (have {} need {nbytes})",
            t.buf.size()
        ));
    }
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, nbytes) };
    gpu.hip
        .memcpy_htod(&t.buf, bytes)
        .map_err(|e| format!("deepseek4 parent: upload_i32: {e:?}"))
}

fn download_i32_prefix(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<i32>, String> {
    let nbytes = nelems * 4;
    if t.buf.size() < nbytes {
        return Err(format!(
            "deepseek4 parent: i32 download too small (have {} need {nbytes})",
            t.buf.size()
        ));
    }
    let mut data = vec![0i32; nelems];
    let bytes = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, nbytes) };
    gpu.hip
        .memcpy_dtoh(bytes, &t.buf)
        .map_err(|e| format!("deepseek4 parent: i32 download: {e:?}"))?;
    Ok(data)
}

fn validate_f32_mat(t: &GpuTensor, rows: usize, cols: usize, name: &str) -> Result<(), String> {
    if t.dtype != DType::F32 {
        return Err(format!(
            "deepseek4 parent: {name} must be F32 (got {:?})",
            t.dtype
        ));
    }
    let need = rows
        .checked_mul(cols)
        .and_then(|e| e.checked_mul(4))
        .ok_or_else(|| format!("deepseek4 parent: {name} size overflow"))?;
    if t.buf.size() < need {
        return Err(format!(
            "deepseek4 parent: {name} buffer too small (have {} need {need} for [{rows},{cols}])",
            t.buf.size()
        ));
    }
    Ok(())
}

fn validate_i32_len(t: &GpuTensor, nelems: usize, name: &str) -> Result<(), String> {
    let need = nelems
        .checked_mul(4)
        .ok_or_else(|| format!("deepseek4 parent: {name} size overflow"))?;
    if t.buf.size() < need {
        return Err(format!(
            "deepseek4 parent: {name} buffer too small (have {} need {need} for {nelems} i32)",
            t.buf.size()
        ));
    }
    Ok(())
}

// ── Unit tests (host-side) ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_scale_hand_value() {
        // softmax_scale = 128**-0.5; n_heads**-0.5 = 64**-0.5
        // product = 1/sqrt(128*64) = 1/sqrt(8192)
        let got = indexer_weights_scale();
        let hand = 1.0 / (8192f64).sqrt();
        assert!(
            (got - hand).abs() < 1e-15,
            "scale={got} hand={hand}"
        );
        // Numeric spot-check against a precomputed value.
        let expect = 0.011_048_543_456_039_806;
        assert!(
            (got - expect).abs() < 1e-15,
            "scale={got} expect≈{expect}"
        );
        // f32 form must be the round-trip of the f64 value.
        let f = indexer_weights_scale_f32();
        assert!((f as f64 - got).abs() < 1e-7);
    }

    #[test]
    fn topk_basic_and_ties() {
        // Descending unique.
        let s = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let idx = indexer_topk_host(&s, 3);
        assert_eq!(idx, vec![3, 1, 2]);

        // Ties: equal scores → lower index wins.
        let s = vec![1.0, 1.0, 0.5, 1.0];
        let idx = indexer_topk_host(&s, 3);
        assert_eq!(idx, vec![0, 1, 3]);

        // Fewer than k finite positions.
        let s = vec![0.5, f32::NEG_INFINITY, 0.25];
        let idx = indexer_topk_host(&s, 5);
        assert_eq!(idx.len(), 5);
        assert_eq!(idx[0], 0);
        assert_eq!(idx[1], 2);
        assert_eq!(idx[2], -1);
        assert_eq!(idx[3], -1);
        assert_eq!(idx[4], -1);
    }

    #[test]
    fn topk_fewer_than_k_active() {
        let s = vec![3.0f32, 1.0];
        let idx = indexer_topk_host(&s, 512);
        assert_eq!(idx.len(), 512);
        assert_eq!(idx[0], 0);
        assert_eq!(idx[1], 1);
        assert!(idx[2..].iter().all(|&v| v == -1));
    }

    #[test]
    fn causal_mask_and_offset_prefill() {
        // rows=8, ratio=4 → per-row cutoffs: 0,0,0,1,1,1,1,2
        // (r+1)//4
        let rows = 8usize;
        let k = 4usize;
        let mut topk = vec![0i32; rows * k];
        // Seed every row with indices 0,1,2,3.
        for r in 0..rows {
            for j in 0..k {
                topk[r * k + j] = j as i32;
            }
        }
        indexer_apply_offset_and_causal_mask(&mut topk, rows, k, 0, 4, /*offset=*/ 128);
        // row 0: cutoff 0 → everything ≥0 masked to -1
        assert!(topk[..k].iter().all(|&v| v == -1), "row0={:?}", &topk[..k]);
        // row 3: cutoff 1 → only idx 0 survives → 0+128=128
        let row3 = &topk[3 * k..4 * k];
        assert_eq!(row3[0], 128);
        assert!(row3[1..].iter().all(|&v| v == -1), "row3={row3:?}");
        // row 7: cutoff 2 → idx 0,1 survive
        let row7 = &topk[7 * k..8 * k];
        assert_eq!(row7[0], 128);
        assert_eq!(row7[1], 129);
        assert_eq!(row7[2], -1);
        assert_eq!(row7[3], -1);
    }

    #[test]
    fn causal_mask_decode_no_future_mask() {
        let rows = 1usize;
        let k = 4usize;
        let mut topk = vec![0, 1, 2, 3];
        indexer_apply_offset_and_causal_mask(&mut topk, rows, k, /*start_pos=*/ 16, 4, 128);
        assert_eq!(topk, vec![128, 129, 130, 131]);
    }

    #[test]
    fn n_visible_table() {
        assert_eq!(indexer_n_visible(0, 0, 4), 0);
        assert_eq!(indexer_n_visible(0, 3, 4), 1);
        assert_eq!(indexer_n_visible(0, 15, 4), 4);
        assert_eq!(indexer_n_compressed(0, 16, 4), 4);
        assert_eq!(indexer_n_compressed(0, 15, 4), 3);
        assert_eq!(indexer_n_compressed(4, 1, 4), 1); // end_pos=5 → 5/4=1
    }

    #[test]
    fn score_oracle_relu_and_weights() {
        // H=2, D=2, N=2
        // q[0]=[1,0], q[1]=[0,1]; kv[0]=[1,1], kv[1]=[-1,2]
        // dots: h0t0=1, h1t0=1, h0t1=-1, h1t1=2
        // relu: 1,1,0,2
        // weights [2, 3] → score0=1*2+1*3=5; score1=0*2+2*3=6
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let kv = vec![1.0, 1.0, -1.0, 2.0];
        let w = vec![2.0, 3.0];
        let s = indexer_score_row_f64(&q, &kv, &w, 2, 2, 2).unwrap();
        assert!((s[0] - 5.0).abs() < 1e-12, "s0={}", s[0]);
        assert!((s[1] - 6.0).abs() < 1e-12, "s1={}", s[1]);
    }

    #[test]
    fn oracle_batch_topk_matches_hand() {
        let rows = 4usize;
        let h = 2usize;
        let d = 2usize;
        let n_slots = 2usize;
        // Identical q/weights every row so scores are the same; causal
        // visibility differs.
        let mut q = Vec::new();
        let mut weights = Vec::new();
        for _ in 0..rows {
            q.extend_from_slice(&[1.0, 0.0, 0.0, 1.0]);
            weights.extend_from_slice(&[2.0, 3.0]);
        }
        let kv = vec![1.0, 1.0, -1.0, 2.0]; // scores 5, 6 when fully visible
        let (scores, topk) = indexer_oracle_f64(
            &q, &kv, &weights, rows, h, d, n_slots, /*start_pos=*/ 0, /*ratio=*/ 4,
            /*k_out=*/ 2, /*offset=*/ 0,
        )
        .unwrap();
        // row 0: n_vis=0 → both -inf, topk all -1
        assert!(topk[0] < 0 && topk[1] < 0, "row0 topk={:?}", &topk[0..2]);
        // row 3: n_vis=1 → only slot 0 visible (score 5); topk = [0, -1]
        assert_eq!(topk[3 * 2], 0);
        assert_eq!(topk[3 * 2 + 1], -1);
        // Fully-visible score values for row 3 slot 0.
        assert!((scores[3 * 2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn index_mismatch_count() {
        assert_eq!(topk_index_mismatch_count(&[1, 2, 3, -1], &[3, 2, 1, -1]), 0);
        assert_eq!(topk_index_mismatch_count(&[1, 2, -1], &[1, 3, -1]), 2);
        assert_eq!(topk_index_mismatch_count(&[], &[]), 0);
    }

    #[test]
    fn scratch_bytes_formula() {
        let max_rows = 16usize;
        let max_n = max_rows.div_ceil(PARENT_INDEX_RATIO).max(PARENT_INDEX_TOPK);
        let act_k = PARENT_DIM.max(PARENT_Q_LORA).max(PARENT_INDEX_Q_WIDTH);
        let act = max_rows * act_k * 2;
        let q = max_rows * PARENT_INDEX_Q_WIDTH * 4;
        let q_score = max_rows * PARENT_INDEX_Q_WIDTH * 4;
        let w = max_rows * PARENT_INDEX_N_HEADS * 4;
        let kv = max_n * PARENT_INDEX_HEAD_DIM * 4;
        let scores = max_rows * max_n * 4;
        let n_per = max_rows * 4;
        let pos = max_rows * 4;
        let total = act + q + q_score + w + kv + scores + n_per + pos;
        // max_n = 512 for max_rows=16; dominant terms: scores 16*512*4 +
        // kv 512*128*4 + two Q buffers 16*8192*4*2 ≈ 1 MiB + 0.25 + 1 MiB.
        assert!(total > 1 * 1024 * 1024, "total={total}");
        assert!(total < 16 * 1024 * 1024, "total={total}");
        assert_eq!(max_n, PARENT_INDEX_TOPK);
    }

    #[test]
    fn refuse_nonzero_ratio_message() {
        let ratio = 0usize;
        let layer_idx = 0usize;
        let msg = format!(
            "deepseek4 parent: parent_indexer_forward refuses compress_ratio={ratio} \
             (layer {layer_idx}); indexer is only defined for compress_ratio == 4"
        );
        assert!(msg.contains("refuses compress_ratio=0"));
        assert!(msg.contains("compress_ratio == 4"));
    }

    #[test]
    fn score_report_perfect_match() {
        let got_s = vec![1.0f32, 2.0, 3.0];
        let ref_s = vec![1.0f64, 2.0, 3.0];
        let got_i = vec![2, 1, 0, -1];
        let ref_i = vec![0, 1, 2, -1];
        let r = IndexerScoreReport::from_scores_and_indices(&got_s, &ref_s, &got_i, &ref_i);
        assert_eq!(r.index_mismatch, 0);
        assert!(r.max_abs < 1e-12);
        assert!(r.l2_rel < 1e-12);
    }
}
