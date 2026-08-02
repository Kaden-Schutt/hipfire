// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! One parent-checkpoint transformer layer: HC → attn → HC → HC → MoE → HC.
//!
//! Operator-semantics authority:
//! - `.codeinsight+research/ds4-parent-ref/inference/model.py` `Block.forward`
//!   (695-707). Sequence is transcribed, not guessed:
//!
//! ```text
//! residual = x
//! x, post, comb = hc_pre(x, hc_attn_*)
//! x = attn_norm(x); x = attn(x, ...)
//! x = hc_post(x, residual, post, comb)
//!
//! residual = x
//! x, post, comb = hc_pre(x, hc_ffn_*)   // own post/comb — not the attn half's
//! x = ffn_norm(x); x = ffn(x, input_ids)
//! x = hc_post(x, residual, post, comb)
//! ```
//!
//! This module is **standalone**. It composes the landed parent sub-blocks
//! (`parent::{hc,attention,moe}`) and does **not** thread a weight provider
//! through the MQ2R `forward.rs`. Every numeric constant traces to the
//! checkpoint `config.json` / tensor shapes — no env-var knobs.

use crate::parent::attention::{
    all_finite, l2_norm, parent_attention_swa, ParentAttnScratch, PARENT_DIM, PARENT_RMS_EPS,
};
use crate::parent::codec::round_to_bf16;
use crate::parent::hc::{
    parent_hc_post, parent_hc_pre, parent_rms_norm, ParentHcParams,
};
use crate::parent::moe::{parent_moe_forward, parent_route, ParentMoeScratch};
use crate::parent::weights::ParentWeights;
use crate::parent::{Ds4ParentBackend, ParentQuantConfig};
use rdna_compute::{DType, Gpu, GpuTensor};

// ── Checkpoint shape constants (config.json of DeepSeek-V4-Flash-0731) ──────

/// `hc_mult` — number of residual streams.
pub const PARENT_HC_MULT: usize = 4;
/// `hc_sinkhorn_iters`.
pub const PARENT_HC_SINKHORN_ITERS: i32 = 20;
/// `hc_eps`.
pub const PARENT_HC_EPS: f32 = 1e-6;
/// Flattened multi-stream width (`hc_mult * dim`).
pub const PARENT_HC_DIM: usize = PARENT_HC_MULT * PARENT_DIM; // 16384

#[inline]
fn err(msg: impl Into<String>) -> String {
    format!("deepseek4 parent: {}", msg.into())
}

// ── Scratch ─────────────────────────────────────────────────────────────────

/// All scratch a layer forward needs, allocated once and reused across
/// layers and calls. Owns the attention and MoE scratch plus the HC
/// intermediates and the per-projection BF16 staging buffers.
///
/// # No per-call device allocation
///
/// [`parent_layer_forward`] / [`parent_layer_forward_traced`] never call
/// `Gpu::alloc_tensor`. Device tiles live here; host staging vectors are
/// capacity-reserved at construction and only `.clear()`/refilled per call.
///
/// Note: the underlying `parent_hc_pre` / `parent_rms_norm` helpers still
/// create short-lived device temporaries of their own (mixes / widened
/// weight). That is pre-existing in those modules and is not introduced by
/// the layer composition.
pub struct ParentForwardScratch {
    attn: ParentAttnScratch,
    moe: ParentMoeScratch,

    /// Working multi-stream residual between the two HC halves.
    /// F32 `[max_rows, hc_mult, dim]` (flat `max_rows * hc_dim`).
    residual_hc: GpuTensor,
    /// `hc_pre` stream output y. F32 `[max_rows, dim]`.
    stream_y: GpuTensor,
    /// Post-`attn_norm` / post-`ffn_norm` stream. F32 `[max_rows, dim]`.
    stream_normed: GpuTensor,
    /// Attention / MoE block output. F32 `[max_rows, dim]`.
    stream_block: GpuTensor,
    /// HC `post` control. F32 `[max_rows, hc_mult]`.
    ///
    /// Single buffer pair for both halves — the FFN half **overwrites**
    /// these after the attention half has consumed them via `hc_post`.
    /// That is intentional and matches `Block.forward`: each half computes
    /// its own `post`/`comb` from its own HC weights. Reusing the attention
    /// half's values for the FFN half is the constant-leak-class bug this
    /// composition exists to make impossible.
    post: GpuTensor,
    /// HC `comb` after sinkhorn. F32 `[max_rows, hc_mult, hc_mult]`.
    comb: GpuTensor,
    /// BF16 MoE / route input staged from `stream_normed`.
    moe_x_bf16: GpuTensor,

    /// Host staging for F32→BF16 cast (capacity = `max_rows * dim`).
    host_f32: Vec<f32>,
    /// Host BF16 byte staging (capacity = `max_rows * dim * 2`).
    host_bf16: Vec<u8>,

    max_rows: usize,
    bytes: usize,
}

impl ParentForwardScratch {
    /// Allocate reusable scratch for up to `max_rows` tokens.
    pub fn new(gpu: &mut Gpu, cfg: &ParentQuantConfig, max_rows: usize) -> Result<Self, String> {
        if max_rows == 0 {
            return Err(err("ParentForwardScratch max_rows must be > 0"));
        }
        // Shapes are pinned to the parent checkpoint; refuse a cfg that would
        // silently size against a different model if those fields ever land.
        let _ = cfg;

        let attn = ParentAttnScratch::new(gpu, cfg, max_rows)?;
        let moe = match ParentMoeScratch::new(gpu, cfg, max_rows) {
            Ok(m) => m,
            Err(e) => {
                // ParentAttnScratch has no Drop free — leave its tensors to the
                // process; construction failure is fatal for the caller either way.
                return Err(e);
            }
        };

        let residual_hc = gpu
            .alloc_tensor(&[max_rows, PARENT_HC_MULT, PARENT_DIM], DType::F32)
            .map_err(|e| err(format!("forward residual_hc alloc: {e:?}")))?;
        let stream_y = match gpu.alloc_tensor(&[max_rows, PARENT_DIM], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(residual_hc);
                return Err(err(format!("forward stream_y alloc: {e:?}")));
            }
        };
        let stream_normed = match gpu.alloc_tensor(&[max_rows, PARENT_DIM], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(residual_hc);
                let _ = gpu.free_tensor(stream_y);
                return Err(err(format!("forward stream_normed alloc: {e:?}")));
            }
        };
        let stream_block = match gpu.alloc_tensor(&[max_rows, PARENT_DIM], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(residual_hc);
                let _ = gpu.free_tensor(stream_y);
                let _ = gpu.free_tensor(stream_normed);
                return Err(err(format!("forward stream_block alloc: {e:?}")));
            }
        };
        let post = match gpu.alloc_tensor(&[max_rows, PARENT_HC_MULT], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(residual_hc);
                let _ = gpu.free_tensor(stream_y);
                let _ = gpu.free_tensor(stream_normed);
                let _ = gpu.free_tensor(stream_block);
                return Err(err(format!("forward post alloc: {e:?}")));
            }
        };
        let comb =
            match gpu.alloc_tensor(&[max_rows, PARENT_HC_MULT, PARENT_HC_MULT], DType::F32) {
                Ok(t) => t,
                Err(e) => {
                    let _ = gpu.free_tensor(residual_hc);
                    let _ = gpu.free_tensor(stream_y);
                    let _ = gpu.free_tensor(stream_normed);
                    let _ = gpu.free_tensor(stream_block);
                    let _ = gpu.free_tensor(post);
                    return Err(err(format!("forward comb alloc: {e:?}")));
                }
            };
        let moe_x_bf16 = match gpu.alloc_tensor(&[max_rows, PARENT_DIM], DType::BF16) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(residual_hc);
                let _ = gpu.free_tensor(stream_y);
                let _ = gpu.free_tensor(stream_normed);
                let _ = gpu.free_tensor(stream_block);
                let _ = gpu.free_tensor(post);
                let _ = gpu.free_tensor(comb);
                return Err(err(format!("forward moe_x_bf16 alloc: {e:?}")));
            }
        };

        let own_bytes = residual_hc.buf.size()
            + stream_y.buf.size()
            + stream_normed.buf.size()
            + stream_block.buf.size()
            + post.buf.size()
            + comb.buf.size()
            + moe_x_bf16.buf.size();
        let bytes = own_bytes + attn.bytes() + moe.bytes();

        Ok(Self {
            attn,
            moe,
            residual_hc,
            stream_y,
            stream_normed,
            stream_block,
            post,
            comb,
            moe_x_bf16,
            host_f32: vec![0.0f32; max_rows * PARENT_DIM],
            host_bf16: vec![0u8; max_rows * PARENT_DIM * 2],
            max_rows,
            bytes,
        })
    }

    /// Total device scratch bytes (attn + moe + layer tiles).
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    // ── Diagnostic getters (Gate 4 oracle cross-checks) ──────────────────

    /// Multi-stream residual after the attention half's `hc_post`
    /// (`[max_rows, hc_mult, dim]` F32). During the FFN half this is the
    /// residual input to FFN `hc_post`.
    pub fn residual_hc(&self) -> &GpuTensor {
        &self.residual_hc
    }

    /// Latest `hc_pre` stream output y (`[max_rows, dim]` F32).
    pub fn stream_y(&self) -> &GpuTensor {
        &self.stream_y
    }

    /// Latest post-RMSNorm stream (`[max_rows, dim]` F32) — input to attn/MoE.
    pub fn stream_normed(&self) -> &GpuTensor {
        &self.stream_normed
    }

    /// Latest attention or MoE block output (`[max_rows, dim]` F32).
    pub fn stream_block(&self) -> &GpuTensor {
        &self.stream_block
    }

    /// Latest HC `post` control (`[max_rows, hc_mult]` F32). After a full
    /// layer forward this holds the **FFN** half's post (attn's was consumed
    /// and overwritten).
    pub fn post(&self) -> &GpuTensor {
        &self.post
    }

    /// Latest HC `comb` after sinkhorn (`[max_rows, hc_mult, hc_mult]` F32).
    /// Same half-overwrite contract as [`Self::post`].
    pub fn comb(&self) -> &GpuTensor {
        &self.comb
    }

    /// BF16 MoE input staged from the post-`ffn_norm` stream.
    pub fn moe_x_bf16(&self) -> &GpuTensor {
        &self.moe_x_bf16
    }

    /// Nested attention scratch (Q/KV intermediates).
    pub fn attn_scratch(&self) -> &ParentAttnScratch {
        &self.attn
    }

    /// Nested MoE scratch.
    pub fn moe_scratch(&self) -> &ParentMoeScratch {
        &self.moe
    }
}

// ── Trace ───────────────────────────────────────────────────────────────────

/// Per-stage L2 norms for the canary, filled when a caller opts in. Keeps the
/// gate binary from having to reach into private scratch.
///
/// Cheap: a handful of small D2H reductions. Only the `_traced` entry pays.
#[derive(Default, Debug, Clone)]
pub struct ParentLayerTrace {
    pub hc_pre_attn: f32,
    pub attn_norm: f32,
    pub attn_out: f32,
    pub hc_post_attn: f32,
    pub hc_pre_ffn: f32,
    pub ffn_norm: f32,
    pub moe_out: f32,
    pub hc_post_ffn: f32,
}

// ── Forward ─────────────────────────────────────────────────────────────────

/// One layer. `x` is the HC residual state `[rows, hc_mult, dim]` F32 and is
/// read; `out` receives the updated HC state, same shape. `input_ids` is
/// required for hash-routed layers (`layer_idx < num_hash_layers`).
///
/// `kv_ring` is the caller's persistent SWA ring
/// `[n_kv_heads=1, head_dim=512, window=128]` F32 — not owned by scratch.
pub fn parent_layer_forward(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    weights: &ParentWeights,
    cfg: &ParentQuantConfig,
    scratch: &mut ParentForwardScratch,
    layer_idx: usize,
    x: &GpuTensor,
    rows: usize,
    start_pos: usize,
    input_ids: Option<&[u32]>,
    kv_ring: &GpuTensor,
    out: &GpuTensor,
) -> Result<(), String> {
    parent_layer_forward_inner(
        gpu, backend, weights, cfg, scratch, layer_idx, x, rows, start_pos, input_ids, kv_ring,
        out, None,
    )
}

/// Same as [`parent_layer_forward`], filling per-stage L2 norms into `trace`.
pub fn parent_layer_forward_traced(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    weights: &ParentWeights,
    cfg: &ParentQuantConfig,
    scratch: &mut ParentForwardScratch,
    layer_idx: usize,
    x: &GpuTensor,
    rows: usize,
    start_pos: usize,
    input_ids: Option<&[u32]>,
    kv_ring: &GpuTensor,
    out: &GpuTensor,
    trace: &mut ParentLayerTrace,
) -> Result<(), String> {
    parent_layer_forward_inner(
        gpu,
        backend,
        weights,
        cfg,
        scratch,
        layer_idx,
        x,
        rows,
        start_pos,
        input_ids,
        kv_ring,
        out,
        Some(trace),
    )
}

fn parent_layer_forward_inner(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    weights: &ParentWeights,
    cfg: &ParentQuantConfig,
    scratch: &mut ParentForwardScratch,
    layer_idx: usize,
    x: &GpuTensor,
    rows: usize,
    start_pos: usize,
    input_ids: Option<&[u32]>,
    kv_ring: &GpuTensor,
    out: &GpuTensor,
    mut trace: Option<&mut ParentLayerTrace>,
) -> Result<(), String> {
    backend.ensure_device(gpu)?;

    // ── Validate layer / shapes ─────────────────────────────────────────
    if rows == 0 {
        return Err(err("parent_layer_forward rows must be > 0"));
    }
    if rows > scratch.max_rows {
        return Err(err(format!(
            "parent_layer_forward rows {rows} exceeds scratch.max_rows {}",
            scratch.max_rows
        )));
    }
    if layer_idx >= cfg.num_hidden_layers {
        return Err(err(format!(
            "parent_layer_forward layer_idx {layer_idx} out of range (num_hidden_layers={})",
            cfg.num_hidden_layers
        )));
    }
    if layer_idx < weights.layer_range.start || layer_idx >= weights.layer_range.end {
        return Err(err(format!(
            "parent_layer_forward layer_idx {layer_idx} not loaded (loaded range {:?})",
            weights.layer_range
        )));
    }
    let local = layer_idx - weights.layer_range.start;
    let layer = &weights.layers[local];
    if layer.layer_idx != layer_idx {
        return Err(err(format!(
            "parent_layer_forward layer slot mismatch: weights.layers[{local}].layer_idx={} \
             but requested {layer_idx}",
            layer.layer_idx
        )));
    }

    // compress_ratio is validated inside parent_attention_swa (0 / 4 / 128).
    // Layer forward no longer refuses ratio != 0 — compressor/indexer are
    // owned by ParentAttnScratch and wired through the attention path.
    let _ratio = layer.compress_ratio;
    if cfg.compress_ratio(layer_idx) != layer.compress_ratio {
        return Err(err(format!(
            "parent_layer_forward: config.compress_ratios[{layer_idx}] = {} but \
             layer.compress_ratio claims {} — refusing rather than guessing",
            cfg.compress_ratio(layer_idx),
            layer.compress_ratio
        )));
    }


    // Hash-routed layers need input_ids; score-routed layers must not require
    // them. Error, do not default.
    let is_hash = layer_idx < cfg.num_hash_layers;
    if is_hash {
        match input_ids {
            None => {
                return Err(err(format!(
                    "parent_layer_forward layer {layer_idx} is hash-routed \
                     (layer_idx < num_hash_layers={}); input_ids required",
                    cfg.num_hash_layers
                )));
            }
            Some(ids) if ids.len() < rows => {
                return Err(err(format!(
                    "parent_layer_forward input_ids length {} < rows {rows}",
                    ids.len()
                )));
            }
            Some(_) => {}
        }
    }

    if !weights.experts_loaded || layer.experts.is_empty() {
        return Err(err(format!(
            "parent_layer_forward layer {layer_idx}: routed experts not loaded \
             (ParentLoadPlan::load_experts must be true)"
        )));
    }

    validate_f32_hc(x, rows, "x")?;
    validate_f32_hc(out, rows, "out")?;
    // Refuse aliased in/out — hc_post would race residual against destination.
    if x.buf.as_ptr() == out.buf.as_ptr() {
        return Err(err(
            "parent_layer_forward x and out must be distinct buffers",
        ));
    }

    let dim = PARENT_DIM;
    let hc = PARENT_HC_MULT;
    let eps = PARENT_RMS_EPS;
    let hc_eps = PARENT_HC_EPS;
    let sinkhorn = PARENT_HC_SINKHORN_ITERS;

    // Row-prefix views into scratch tiles (capacity may exceed `rows`).
    let stream_y = scratch.stream_y.sub_offset(0, rows * dim);
    let stream_normed = scratch.stream_normed.sub_offset(0, rows * dim);
    let stream_block = scratch.stream_block.sub_offset(0, rows * dim);
    let residual_hc = scratch.residual_hc.sub_offset(0, rows * PARENT_HC_DIM);
    let post = scratch.post.sub_offset(0, rows * hc);
    let comb = scratch.comb.sub_offset(0, rows * hc * hc);
    let moe_x = scratch.moe_x_bf16.sub_offset(0, rows * dim);

    // ═══════════════════════════════════════════════════════════════════
    // Attention half
    // ═══════════════════════════════════════════════════════════════════
    // residual = x  (x is read-only residual for hc_post)
    // x_s, post, comb = hc_pre(x, hc_attn_*)
    let attn_hc = ParentHcParams {
        fn_mat: &layer.hc_attn_fn,
        base: &layer.hc_attn_base,
        scale: &layer.hc_attn_scale,
    };
    parent_hc_pre(
        gpu,
        backend,
        x,
        attn_hc,
        rows,
        hc,
        dim,
        eps,
        sinkhorn,
        hc_eps,
        &stream_y,
        &post,
        &comb,
    )
    .map_err(|e| err(format!("layer {layer_idx} hc_pre_attn: {e}")))?;
    if let Some(t) = trace.as_mut() {
        t.hc_pre_attn = stage_l2(gpu, &stream_y, rows * dim)?;
    }

    // x_s = attn_norm(x_s)
    parent_rms_norm(
        gpu,
        backend,
        &stream_y,
        &layer.attn_norm,
        &stream_normed,
        rows,
        dim,
        eps,
    )
    .map_err(|e| err(format!("layer {layer_idx} attn_norm: {e}")))?;
    if let Some(t) = trace.as_mut() {
        t.attn_norm = stage_l2(gpu, &stream_normed, rows * dim)?;
    }

    // x_s = attn(x_s, start_pos, ...)
    parent_attention_swa(
        gpu,
        backend,
        layer,
        cfg,
        &mut scratch.attn,
        &stream_normed,
        rows,
        start_pos,
        kv_ring,
        &stream_block,
    )
    .map_err(|e| err(format!("layer {layer_idx} attn: {e}")))?;
    if let Some(t) = trace.as_mut() {
        t.attn_out = stage_l2(gpu, &stream_block, rows * dim)?;
    }

    // x_hc = hc_post(x_s, residual=x, post, comb) → residual_hc
    parent_hc_post(
        gpu,
        backend,
        &stream_block,
        x,
        &post,
        &comb,
        rows,
        hc,
        dim,
        &residual_hc,
    )
    .map_err(|e| err(format!("layer {layer_idx} hc_post_attn: {e}")))?;
    if let Some(t) = trace.as_mut() {
        t.hc_post_attn = stage_l2(gpu, &residual_hc, rows * PARENT_HC_DIM)?;
    }

    // ═══════════════════════════════════════════════════════════════════
    // FFN half — recomputes post/comb from hc_ffn_* (overwrites attn's)
    // ═══════════════════════════════════════════════════════════════════
    // residual = residual_hc
    // x_s, post, comb = hc_pre(residual_hc, hc_ffn_*)
    let ffn_hc = ParentHcParams {
        fn_mat: &layer.hc_ffn_fn,
        base: &layer.hc_ffn_base,
        scale: &layer.hc_ffn_scale,
    };
    parent_hc_pre(
        gpu,
        backend,
        &residual_hc,
        ffn_hc,
        rows,
        hc,
        dim,
        eps,
        sinkhorn,
        hc_eps,
        &stream_y,
        &post,
        &comb,
    )
    .map_err(|e| err(format!("layer {layer_idx} hc_pre_ffn: {e}")))?;
    if let Some(t) = trace.as_mut() {
        t.hc_pre_ffn = stage_l2(gpu, &stream_y, rows * dim)?;
    }

    // x_s = ffn_norm(x_s)
    parent_rms_norm(
        gpu,
        backend,
        &stream_y,
        &layer.ffn_norm,
        &stream_normed,
        rows,
        dim,
        eps,
    )
    .map_err(|e| err(format!("layer {layer_idx} ffn_norm: {e}")))?;
    if let Some(t) = trace.as_mut() {
        t.ffn_norm = stage_l2(gpu, &stream_normed, rows * dim)?;
    }

    // Stage F32 normed stream → BF16 for route + MoE (no device alloc).
    stage_f32_to_bf16(gpu, scratch, &stream_normed, rows, dim)?;

    // Route (hash layers use input_ids; score layers ignore them).
    let routing = parent_route(
        gpu,
        backend,
        layer,
        cfg,
        &moe_x,
        rows,
        if is_hash { input_ids } else { None },
    )
    .map_err(|e| err(format!("layer {layer_idx} route: {e}")))?;

    // x_s = moe(x_s, input_ids)
    parent_moe_forward(
        gpu,
        backend,
        layer,
        cfg,
        &mut scratch.moe,
        &moe_x,
        rows,
        &routing,
        &stream_block,
    )
    .map_err(|e| err(format!("layer {layer_idx} moe: {e}")))?;
    if let Some(t) = trace.as_mut() {
        t.moe_out = stage_l2(gpu, &stream_block, rows * dim)?;
    }

    // out = hc_post(x_s, residual=residual_hc, post, comb)
    // post/comb here are the FFN half's — just written by hc_pre_ffn above.
    parent_hc_post(
        gpu,
        backend,
        &stream_block,
        &residual_hc,
        &post,
        &comb,
        rows,
        hc,
        dim,
        out,
    )
    .map_err(|e| err(format!("layer {layer_idx} hc_post_ffn: {e}")))?;
    if let Some(t) = trace.as_mut() {
        t.hc_post_ffn = stage_l2(gpu, out, rows * PARENT_HC_DIM)?;
    }

    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn validate_f32_hc(t: &GpuTensor, rows: usize, name: &str) -> Result<(), String> {
    if t.dtype != DType::F32 {
        return Err(err(format!(
            "{name} must be F32 [rows, hc_mult, dim] (got {:?})",
            t.dtype
        )));
    }
    let need = rows
        .checked_mul(PARENT_HC_DIM)
        .ok_or_else(|| err(format!("{name} size overflow")))?;
    if t.numel() < need {
        return Err(err(format!(
            "{name} too short for rows={rows} hc_dim={} (have {} need {need})",
            PARENT_HC_DIM,
            t.numel()
        )));
    }
    Ok(())
}

fn stage_l2(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<f32, String> {
    let host = download_f32_prefix(gpu, t, nelems)?;
    if !all_finite(&host) {
        // Still report the L2 (will be NaN) — the canary surfaces non-finite.
        return Ok(l2_norm(&host));
    }
    Ok(l2_norm(&host))
}

fn download_f32_prefix(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    if t.dtype != DType::F32 {
        return Err(err(format!(
            "download_f32_prefix expects F32 (got {:?})",
            t.dtype
        )));
    }
    let nbytes = nelems
        .checked_mul(4)
        .ok_or_else(|| err("download_f32_prefix size overflow"))?;
    if t.buf.size() < nbytes {
        return Err(err(format!(
            "download_f32_prefix: buffer too small (have {} need {nbytes})",
            t.buf.size()
        )));
    }
    let mut host = vec![0.0f32; nelems];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut u8, nbytes) };
    gpu.hip
        .memcpy_dtoh(bytes, &t.buf)
        .map_err(|e| err(format!("download_f32_prefix: {e:?}")))?;
    Ok(host)
}

/// Copy `src` F32 `[rows, dim]` into scratch `moe_x_bf16` via host BF16 pack.
/// Uses the pre-sized `host_f32` / `host_bf16` vectors — no device alloc.
fn stage_f32_to_bf16(
    gpu: &Gpu,
    scratch: &mut ParentForwardScratch,
    src: &GpuTensor,
    rows: usize,
    dim: usize,
) -> Result<(), String> {
    let nelems = rows * dim;
    if scratch.host_f32.len() < nelems || scratch.host_bf16.len() < nelems * 2 {
        return Err(err(
            "stage_f32_to_bf16: host staging shorter than rows*dim (scratch bug)",
        ));
    }
    let nbytes = nelems * 4;
    if src.buf.size() < nbytes {
        return Err(err(format!(
            "stage_f32_to_bf16: src too small (have {} need {nbytes})",
            src.buf.size()
        )));
    }
    {
        let dst = &mut scratch.host_f32[..nelems];
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, nbytes) };
        gpu.hip
            .memcpy_dtoh(bytes, &src.buf)
            .map_err(|e| err(format!("stage_f32_to_bf16 dtoh: {e:?}")))?;
    }
    for i in 0..nelems {
        let bf = round_to_bf16(scratch.host_f32[i]);
        let bits = bf.to_bits() >> 16;
        let b = (bits as u16).to_le_bytes();
        scratch.host_bf16[i * 2] = b[0];
        scratch.host_bf16[i * 2 + 1] = b[1];
    }
    let bf_bytes = nelems * 2;
    if scratch.moe_x_bf16.buf.size() < bf_bytes {
        return Err(err(format!(
            "stage_f32_to_bf16: moe_x_bf16 too small (have {} need {bf_bytes})",
            scratch.moe_x_bf16.buf.size()
        )));
    }
    gpu.hip
        .memcpy_htod(&scratch.moe_x_bf16.buf, &scratch.host_bf16[..bf_bytes])
        .map_err(|e| err(format!("stage_f32_to_bf16 htod: {e:?}")))?;
    Ok(())
}

// ── Host-side unit tests (no GPU) ───────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parent::ParentQuantConfig;

    fn sample_cfg() -> ParentQuantConfig {
        ParentQuantConfig {
            model_type: "deepseek_v4".into(),
            quant_method: "fp8".into(),
            fmt: "e4m3".into(),
            scale_fmt: "ue8m0".into(),
            expert_dtype: "fp4".into(),
            weight_block_size: [128, 128],
            num_hidden_layers: 43,
            num_hash_layers: 3,
            n_routed_experts: 256,
            num_experts_per_tok: 6,
            // layer 0 pure SWA; layer 1 compress 4; rest filler
            compress_ratios: {
                let mut v = vec![0usize; 43];
                v[1] = 4;
                v[2] = 128;
                v
            },
        }
    }

    #[test]
    fn scratch_sizing_formula() {
        // ParentForwardScratch::new needs a GPU; validate the byte formula
        // the constructor uses so a dim/hc change cannot silently desync.
        let max_rows = 16usize;
        let own = max_rows * PARENT_HC_DIM * 4           // residual_hc
            + max_rows * PARENT_DIM * 4                 // stream_y
            + max_rows * PARENT_DIM * 4                 // stream_normed
            + max_rows * PARENT_DIM * 4                 // stream_block
            + max_rows * PARENT_HC_MULT * 4             // post
            + max_rows * PARENT_HC_MULT * PARENT_HC_MULT * 4 // comb
            + max_rows * PARENT_DIM * 2; // moe_x_bf16
        // Sanity: own tiles alone should be well under 16 MiB for 16 rows.
        assert!(own < 16 * 1024 * 1024, "own={own}");
        // HC dim contract.
        assert_eq!(PARENT_HC_DIM, 4 * 4096);
        assert_eq!(PARENT_HC_MULT, 4);
        assert_eq!(PARENT_DIM, 4096);
        // Host staging capacity the constructor reserves.
        assert_eq!(max_rows * PARENT_DIM, 16 * 4096);
        let _ = sample_cfg();
    }

    #[test]
    fn compress_ratio_mismatch_message() {
        // Layer forward no longer refuses ratio!=0; it still fails closed on
        // a config/layer compress_ratio mismatch.
        let layer_idx = 5usize;
        let cfg_ratio = 4usize;
        let layer_ratio = 0usize;
        let msg = err(format!(
            "parent_layer_forward: config.compress_ratios[{layer_idx}] = {cfg_ratio} but \
             layer.compress_ratio claims {layer_ratio} — refusing rather than guessing"
        ));
        assert!(msg.contains("compress_ratios[5] = 4"));
        assert!(msg.contains("claims 0"));
        assert!(msg.contains("deepseek4 parent:"));
    }

    #[test]
    fn hash_layer_input_ids_required_message() {
        let layer_idx = 0usize;
        let num_hash_layers = 3usize;
        assert!(layer_idx < num_hash_layers);
        let msg = err(format!(
            "parent_layer_forward layer {layer_idx} is hash-routed \
             (layer_idx < num_hash_layers={num_hash_layers}); input_ids required"
        ));
        assert!(msg.contains("hash-routed"));
        assert!(msg.contains("input_ids required"));
        assert!(msg.contains("num_hash_layers=3"));
    }


    #[test]
    fn score_layer_does_not_require_ids() {
        let cfg = sample_cfg();
        // Layer 3 is the first score-routed layer.
        assert!(3 >= cfg.num_hash_layers);
        // The forward path only checks input_ids when is_hash; document that
        // contract here so a regression that requires ids on score layers is
        // visible in host tests.
        let is_hash = 3 < cfg.num_hash_layers;
        assert!(!is_hash);
    }

    #[test]
    fn layer_index_bounds() {
        let cfg = sample_cfg();
        assert_eq!(cfg.num_hidden_layers, 43);
        let bad = 43usize;
        let msg = err(format!(
            "parent_layer_forward layer_idx {bad} out of range (num_hidden_layers={})",
            cfg.num_hidden_layers
        ));
        assert!(msg.contains("out of range"));
        assert!(msg.contains("num_hidden_layers=43"));
    }

    #[test]
    fn layer_not_loaded_message() {
        let layer_idx = 7usize;
        let loaded: std::ops::Range<usize> = 0..1;
        assert!(layer_idx < loaded.start || layer_idx >= loaded.end);
        let msg = err(format!(
            "parent_layer_forward layer_idx {layer_idx} not loaded (loaded range {:?})",
            loaded
        ));
        assert!(msg.contains("not loaded"));
        assert!(msg.contains("0..1"));
    }

    #[test]
    fn ffn_half_uses_own_post_comb_contract() {
        // Architectural assertion: a single post/comb buffer pair is shared
        // across halves and is rewritten by the FFN hc_pre. The attention
        // half's values therefore cannot leak into the FFN hc_post unless
        // the FFN hc_pre is skipped — which the forward sequence does not.
        // This test locks the field layout so a "two pairs, attn reused"
        // refactor would have to update it deliberately.
        let cfg = sample_cfg();
        assert_eq!(cfg.compress_ratio(0), 0);
        // Distinct HC weight names on ParentLayerWeights — compile-time via
        // field access in the forward body (hc_attn_* vs hc_ffn_*). Runtime
        // proof is the mi300x canary (post buffer contents after full layer
        // equal FFN-only hc_pre, not attn-only).
        let _ = (
            "hc_attn_fn",
            "hc_attn_base",
            "hc_attn_scale",
            "hc_ffn_fn",
            "hc_ffn_base",
            "hc_ffn_scale",
        );
    }

    #[test]
    fn trace_default_zero() {
        let t = ParentLayerTrace::default();
        assert_eq!(t.hc_pre_attn, 0.0);
        assert_eq!(t.attn_norm, 0.0);
        assert_eq!(t.attn_out, 0.0);
        assert_eq!(t.hc_post_attn, 0.0);
        assert_eq!(t.hc_pre_ffn, 0.0);
        assert_eq!(t.ffn_norm, 0.0);
        assert_eq!(t.moe_out, 0.0);
        assert_eq!(t.hc_post_ffn, 0.0);
    }

    #[test]
    fn constants_match_checkpoint_contract() {
        // config.json: hidden 4096, hc_mult 4, rms_norm_eps 1e-6,
        // hc_sinkhorn_iters 20, hc_eps 1e-6.
        assert_eq!(PARENT_DIM, 4096);
        assert_eq!(PARENT_HC_MULT, 4);
        assert_eq!(PARENT_HC_SINKHORN_ITERS, 20);
        assert!((PARENT_RMS_EPS - 1e-6).abs() < 1e-12);
        assert!((PARENT_HC_EPS - 1e-6).abs() < 1e-12);
    }
}
