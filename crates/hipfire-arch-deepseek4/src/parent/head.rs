// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Parent embedding input and output head.
//!
//! Operator-semantics authority
//! (`.codeinsight+research/ds4-parent-ref/inference/model.py`):
//!
//! - `ParallelEmbedding.forward` (102-111) — vocab row gather. Single-rank
//!   path is plain `F.embedding`; out-of-range ids are a hard error here
//!   (fail closed — the distributed mask/zero path is multi-rank only).
//! - `Transformer.forward` (914-923) — embed → unsqueeze/repeat over
//!   `hc_mult` streams → …layers… → `hc_head` (plain sigmoid, no Sinkhorn)
//!   → `RMSNorm` → `ParallelHead`.
//! - `Block.hc_head` (709-716) — delegated to [`super::hc::parent_hc_head`].
//! - `ParallelHead.forward` (731-740) — `F.linear(x.float(), weight)` where
//!   the checkpoint stores `head.weight` as BF16 and the reference widens it
//!   to F32. **Not** an FP8 act-quant projection: do not route through
//!   [`super::linear::parent_linear_dense`].

use crate::parent::attention::{PARENT_DIM, PARENT_RMS_EPS};
use crate::parent::codec::round_to_bf16;
use crate::parent::hc::{parent_hc_head, parent_rms_norm, ParentHcParams};
use crate::parent::layer_ref::{hc_head_ref, rms_norm_ref};
use crate::parent::plog::PlogWriter;
use crate::parent::weights::ParentWeights;
use crate::parent::{Ds4ParentBackend, ParentQuantConfig};
use rdna_compute::{DType, Gpu, GpuTensor};

// ── Checkpoint shape constants (config.json of DeepSeek-V4-Flash-0731) ──────

/// `vocab_size`.
pub const PARENT_VOCAB: usize = 129_280;
/// `hc_mult` — residual stream count after embedding expand.
pub const PARENT_HC_MULT: usize = 4;
/// `hc_eps` for the head HC mix (no Sinkhorn).
pub const PARENT_HC_EPS: f32 = 1e-6;
/// Flattened multi-stream width (`hc_mult * dim`).
pub const PARENT_HC_DIM: usize = PARENT_HC_MULT * PARENT_DIM; // 16_384

#[inline]
fn err(msg: impl Into<String>) -> String {
    format!("deepseek4 parent: {}", msg.into())
}

// ── Scratch ─────────────────────────────────────────────────────────────────

/// Reusable device scratch for [`parent_embed`] / [`parent_head`].
///
/// Logits are **caller-owned** and intentionally not held here: a 1K-token
/// capture is ~530 MiB and is streamed via [`parent_logits_to_plog`]. Scratch
/// only covers the intermediate HC / norm / BF16 staging tiles.
pub struct ParentHeadScratch {
    /// `hc_head` output. F32 `[max_rows, dim]`.
    stream_y: GpuTensor,
    /// Post-final-RMSNorm activation. F32 `[max_rows, dim]`.
    stream_normed: GpuTensor,
    /// BF16 staging of the normed activation for the head GEMM.
    /// BF16 `[max_rows, dim]`.
    x_bf16: GpuTensor,
    /// Single-row BF16 gather staging for the embedding table (Gate-5 streaming).
    /// BF16 `[dim]`.
    #[allow(dead_code)]
    embed_row_bf16: GpuTensor,
    /// Single-row F32 logits staging for the plog bridge (streaming).
    /// F32 `[vocab]`.
    #[allow(dead_code)]
    logits_row: GpuTensor,

    max_rows: usize,
    bytes: usize,
}

impl ParentHeadScratch {
    /// Allocate reusable scratch for up to `max_rows` tokens.
    pub fn new(gpu: &mut Gpu, cfg: &ParentQuantConfig, max_rows: usize) -> Result<Self, String> {
        let _ = cfg; // shapes pinned to the parent checkpoint contract
        if max_rows == 0 {
            return Err(err("ParentHeadScratch max_rows must be > 0"));
        }

        let stream_y = gpu
            .alloc_tensor(&[max_rows, PARENT_DIM], DType::F32)
            .map_err(|e| err(format!("ParentHeadScratch stream_y: {e:?}")))?;
        let stream_normed = match gpu.alloc_tensor(&[max_rows, PARENT_DIM], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(stream_y);
                return Err(err(format!("ParentHeadScratch stream_normed: {e:?}")));
            }
        };
        let x_bf16 = match gpu.alloc_tensor(&[max_rows, PARENT_DIM], DType::BF16) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(stream_y);
                let _ = gpu.free_tensor(stream_normed);
                return Err(err(format!("ParentHeadScratch x_bf16: {e:?}")));
            }
        };
        let embed_row_bf16 = match gpu.alloc_tensor(&[PARENT_DIM], DType::BF16) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(stream_y);
                let _ = gpu.free_tensor(stream_normed);
                let _ = gpu.free_tensor(x_bf16);
                return Err(err(format!("ParentHeadScratch embed_row_bf16: {e:?}")));
            }
        };
        let logits_row = match gpu.alloc_tensor(&[PARENT_VOCAB], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(stream_y);
                let _ = gpu.free_tensor(stream_normed);
                let _ = gpu.free_tensor(x_bf16);
                let _ = gpu.free_tensor(embed_row_bf16);
                return Err(err(format!("ParentHeadScratch logits_row: {e:?}")));
            }
        };

        let bytes = max_rows * PARENT_DIM * 4 // stream_y
            + max_rows * PARENT_DIM * 4 // stream_normed
            + max_rows * PARENT_DIM * 2 // x_bf16
            + PARENT_DIM * 2 // embed_row_bf16
            + PARENT_VOCAB * 4; // logits_row

        Ok(Self {
            stream_y,
            stream_normed,
            x_bf16,
            embed_row_bf16,
            logits_row,
            max_rows,
            bytes,
        })
    }

    /// Total scratch bytes resident on device.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Peak capacity in rows.
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    /// Peak device bytes a `n_tokens`-long logits capture needs under the
    /// streaming design: one full logits tile of `rows_cap × vocab` plus this
    /// scratch. Callers that stream one row at a time via
    /// [`parent_logits_to_plog`] only need `vocab × 4` of logits on device at
    /// a time (already included in [`Self::bytes`]).
    pub fn peak_logits_capture_bytes(n_tokens: usize, stream_rows: usize) -> usize {
        let rows = stream_rows.max(1);
        // Streaming design: keep at most `stream_rows` logit rows on device.
        let logits = rows
            .saturating_mul(PARENT_VOCAB)
            .saturating_mul(4);
        // Scratch sized for `stream_rows`.
        let scratch = stream_rows * PARENT_DIM * 4
            + stream_rows * PARENT_DIM * 4
            + stream_rows * PARENT_DIM * 2
            + PARENT_DIM * 2
            + PARENT_VOCAB * 4;
        let _ = n_tokens; // file-side only; not device-resident
        logits.saturating_add(scratch)
    }
}

// ── Embed ───────────────────────────────────────────────────────────────────

/// Token ids → initial HC stream state `[rows, hc_mult, dim]` f32.
///
/// `embed.weight` is BF16 `[vocab, dim]`. Each token row is gathered, widened
/// BF16→F32 (lossless high-16-bit shift), then repeated across `hc_mult`
/// streams — matching `Transformer.forward`:
/// `h = embed(ids).unsqueeze(2).repeat(1, 1, hc_mult, 1)`.
///
/// `out` must be F32 with at least `token_ids.len() * hc_mult * dim` elements.
pub fn parent_embed(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    weights: &ParentWeights,
    cfg: &ParentQuantConfig,
    token_ids: &[u32],
    out: &GpuTensor,
) -> Result<(), String> {
    backend.ensure_device(gpu)?;
    let _ = cfg;
    let rows = token_ids.len();
    if rows == 0 {
        return Err(err("parent_embed: token_ids must be non-empty"));
    }
    require_dtype(&weights.embed, DType::BF16, "embed.weight")?;
    require_elems(&weights.embed, PARENT_VOCAB * PARENT_DIM, "embed.weight")?;
    require_dtype(out, DType::F32, "parent_embed out")?;
    require_elems(out, rows * PARENT_HC_MULT * PARENT_DIM, "parent_embed out")?;

    // Host-side gather + expand. A 1K-token tile is 1K × 4096 × 4 B × 4
    // streams ≈ 64 MiB — fine for calibration chunks; the alternative is a
    // dedicated BF16-gather kernel we do not have.
    let mut host = vec![0.0f32; rows * PARENT_HC_MULT * PARENT_DIM];
    let mut row_bytes = vec![0u8; PARENT_DIM * 2];
    for (r, &tid) in token_ids.iter().enumerate() {
        let id = tid as usize;
        if id >= PARENT_VOCAB {
            return Err(err(format!(
                "parent_embed: token id {tid} out of range (vocab={PARENT_VOCAB})"
            )));
        }
        let byte_off = id
            .checked_mul(PARENT_DIM * 2)
            .ok_or_else(|| err("parent_embed: embed byte offset overflow"))?;
        gpu.hip
            .memcpy_dtoh_at(&mut row_bytes, &weights.embed.buf, byte_off)
            .map_err(|e| err(format!("parent_embed download row {tid}: {e:?}")))?;
        // Widen BF16 → F32 and splat across hc_mult streams.
        for d in 0..PARENT_DIM {
            let bits = u16::from_le_bytes([row_bytes[2 * d], row_bytes[2 * d + 1]]);
            let v = f32::from_bits((bits as u32) << 16);
            for h in 0..PARENT_HC_MULT {
                host[(r * PARENT_HC_MULT + h) * PARENT_DIM + d] = v;
            }
        }
    }

    let nbytes = host.len() * 4;
    let bytes = f32_slice_as_le_bytes(&host);
    if out.buf.size() < nbytes {
        return Err(err(format!(
            "parent_embed out too small (have {} need {nbytes})",
            out.buf.size()
        )));
    }
    gpu.hip
        .memcpy_htod(&out.buf, bytes)
        .map_err(|e| err(format!("parent_embed upload: {e:?}")))?;
    Ok(())
}

/// Host-only embedding gather used by unit tests and the f64 oracle path.
///
/// `table_bf16` is the raw little-endian BF16 bytes of `embed.weight`
/// (`vocab * dim` elements). Out-of-range token ids refuse.
pub fn embed_gather_ref(
    table_bf16: &[u8],
    token_ids: &[u32],
    vocab: usize,
    dim: usize,
    hc_mult: usize,
) -> Result<Vec<f32>, String> {
    if dim == 0 || hc_mult == 0 {
        return Err(err("embed_gather_ref: dim and hc_mult must be > 0"));
    }
    if vocab == 0 {
        return Err(err("embed_gather_ref: vocab must be > 0"));
    }
    let need = vocab
        .checked_mul(dim)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| err("embed_gather_ref: table size overflow"))?;
    if table_bf16.len() < need {
        return Err(err(format!(
            "embed_gather_ref: table too short (have {} need {need})",
            table_bf16.len()
        )));
    }
    let rows = token_ids.len();
    let mut out = vec![0.0f32; rows * hc_mult * dim];
    for (r, &tid) in token_ids.iter().enumerate() {
        let id = tid as usize;
        if id >= vocab {
            return Err(err(format!(
                "embed_gather_ref: token id {tid} out of range (vocab={vocab})"
            )));
        }
        let base = id * dim * 2;
        for d in 0..dim {
            let bits = u16::from_le_bytes([table_bf16[base + 2 * d], table_bf16[base + 2 * d + 1]]);
            let v = f32::from_bits((bits as u32) << 16);
            for h in 0..hc_mult {
                out[(r * hc_mult + h) * dim + d] = v;
            }
        }
    }
    Ok(out)
}

// ── Head ────────────────────────────────────────────────────────────────────

/// Final HC state → logits `[rows, vocab]` f32.
///
/// Pipeline (model.py:922-923):
/// 1. `hc_head` — plain sigmoid path, **no** Sinkhorn
///    (`hc_head_fn [4, 16384]`, `hc_head_base [4]`, `hc_head_scale [1]`)
/// 2. final `RMSNorm` over `norm.weight` BF16 `[dim]`, `eps = 1e-6`
/// 3. head projection: BF16 `head.weight [vocab, dim]` × F32 acts → F32 logits
///    via BF16×BF16→F32 MFMA after staging the normed acts as BF16. Matches
///    the reference's BF16-held weight (widened in the ref to F32); activations
///    are rounded to BF16 for the GEMM the same way every other parent dense
///    path stages BF16 tiles. **Not** routed through act-quant/`parent_linear_dense`.
///
/// `x` is F32 `[rows, hc_mult, dim]` (flat `rows * hc_mult * dim`).
/// `logits` is F32 `[rows, vocab]`.
pub fn parent_head(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    weights: &ParentWeights,
    cfg: &ParentQuantConfig,
    x: &GpuTensor,
    rows: usize,
    logits: &GpuTensor,
) -> Result<(), String> {
    backend.ensure_device(gpu)?;
    let _ = cfg;
    if rows == 0 {
        return Err(err("parent_head: rows must be > 0"));
    }
    require_dtype(x, DType::F32, "parent_head x")?;
    require_elems(x, rows * PARENT_HC_DIM, "parent_head x")?;
    require_dtype(logits, DType::F32, "parent_head logits")?;
    require_elems(logits, rows * PARENT_VOCAB, "parent_head logits")?;
    require_dtype(&weights.head, DType::BF16, "head.weight")?;
    require_elems(&weights.head, PARENT_VOCAB * PARENT_DIM, "head.weight")?;
    require_dtype(&weights.norm, DType::BF16, "norm.weight")?;
    require_elems(&weights.norm, PARENT_DIM, "norm.weight")?;
    require_dtype(&weights.hc_head_fn, DType::F32, "hc_head_fn")?;
    require_elems(
        &weights.hc_head_fn,
        PARENT_HC_MULT * PARENT_HC_DIM,
        "hc_head_fn",
    )?;
    require_dtype(&weights.hc_head_base, DType::F32, "hc_head_base")?;
    require_elems(&weights.hc_head_base, PARENT_HC_MULT, "hc_head_base")?;
    require_dtype(&weights.hc_head_scale, DType::F32, "hc_head_scale")?;
    require_elems(&weights.hc_head_scale, 1, "hc_head_scale")?;

    // Transient tiles (hc_head / rms_norm helpers also allocate short-lived
    // scratch of their own). Freed on every exit path below.
    let y = gpu
        .alloc_tensor(&[rows, PARENT_DIM], DType::F32)
        .map_err(|e| err(format!("parent_head y alloc: {e:?}")))?;
    let normed = match gpu.alloc_tensor(&[rows, PARENT_DIM], DType::F32) {
        Ok(t) => t,
        Err(e) => {
            free_scratch(gpu, y);
            return Err(err(format!("parent_head normed alloc: {e:?}")));
        }
    };
    let x_bf16 = match gpu.alloc_tensor(&[rows, PARENT_DIM], DType::BF16) {
        Ok(t) => t,
        Err(e) => {
            free_scratch(gpu, y);
            free_scratch(gpu, normed);
            return Err(err(format!("parent_head x_bf16 alloc: {e:?}")));
        }
    };

    let result = (|| {
        // 1. hc_head
        let p = ParentHcParams {
            fn_mat: &weights.hc_head_fn,
            base: &weights.hc_head_base,
            scale: &weights.hc_head_scale,
        };
        parent_hc_head(
            gpu,
            backend,
            x,
            p,
            rows,
            PARENT_HC_MULT,
            PARENT_DIM,
            PARENT_RMS_EPS,
            PARENT_HC_EPS,
            &y,
        )?;

        // 2. final RMSNorm
        parent_rms_norm(
            gpu,
            backend,
            &y,
            &weights.norm,
            &normed,
            rows,
            PARENT_DIM,
            PARENT_RMS_EPS,
        )?;

        // 3. Stage normed acts as BF16 and project through head.weight.
        stage_f32_as_bf16(gpu, &normed, &x_bf16, rows * PARENT_DIM)?;
        gpu.gemm_bf16_mfma_gfx942(
            &weights.head.buf,
            &x_bf16.buf,
            &logits.buf,
            PARENT_VOCAB,
            PARENT_DIM,
            rows,
        )
        .map_err(|e| err(format!("parent_head BF16 GEMM: {e:?}")))?;
        Ok(())
    })();

    free_scratch(gpu, y);
    free_scratch(gpu, normed);
    free_scratch(gpu, x_bf16);
    result
}

/// Like [`parent_head`] but reuses caller-owned [`ParentHeadScratch`] so the
/// forward driver can avoid per-call device allocation for the intermediate
/// tiles. `logits` remains caller-owned (streamed / sized externally).
pub fn parent_head_with_scratch(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    weights: &ParentWeights,
    cfg: &ParentQuantConfig,
    scratch: &mut ParentHeadScratch,
    x: &GpuTensor,
    rows: usize,
    logits: &GpuTensor,
) -> Result<(), String> {
    backend.ensure_device(gpu)?;
    let _ = cfg;
    if rows == 0 {
        return Err(err("parent_head: rows must be > 0"));
    }
    if rows > scratch.max_rows {
        return Err(err(format!(
            "parent_head: rows {rows} exceeds ParentHeadScratch capacity {}",
            scratch.max_rows
        )));
    }
    require_dtype(x, DType::F32, "parent_head x")?;
    require_elems(x, rows * PARENT_HC_DIM, "parent_head x")?;
    require_dtype(logits, DType::F32, "parent_head logits")?;
    require_elems(logits, rows * PARENT_VOCAB, "parent_head logits")?;
    require_dtype(&weights.head, DType::BF16, "head.weight")?;
    require_elems(&weights.head, PARENT_VOCAB * PARENT_DIM, "head.weight")?;
    require_dtype(&weights.norm, DType::BF16, "norm.weight")?;
    require_elems(&weights.norm, PARENT_DIM, "norm.weight")?;

    let y = scratch.stream_y.sub_offset(0, rows * PARENT_DIM);
    let normed = scratch.stream_normed.sub_offset(0, rows * PARENT_DIM);
    let x_bf16 = scratch.x_bf16.sub_offset(0, rows * PARENT_DIM);

    let p = ParentHcParams {
        fn_mat: &weights.hc_head_fn,
        base: &weights.hc_head_base,
        scale: &weights.hc_head_scale,
    };
    parent_hc_head(
        gpu,
        backend,
        x,
        p,
        rows,
        PARENT_HC_MULT,
        PARENT_DIM,
        PARENT_RMS_EPS,
        PARENT_HC_EPS,
        &y,
    )?;
    parent_rms_norm(
        gpu,
        backend,
        &y,
        &weights.norm,
        &normed,
        rows,
        PARENT_DIM,
        PARENT_RMS_EPS,
    )?;
    stage_f32_as_bf16(gpu, &normed, &x_bf16, rows * PARENT_DIM)?;
    gpu.gemm_bf16_mfma_gfx942(
        &weights.head.buf,
        &x_bf16.buf,
        &logits.buf,
        PARENT_VOCAB,
        PARENT_DIM,
        rows,
    )
    .map_err(|e| err(format!("parent_head BF16 GEMM: {e:?}")))?;
    Ok(())
}

// ── Head projection oracle ──────────────────────────────────────────────────

/// f64 oracle for `F.linear(x, head_weight)` with BF16-held weights.
///
/// `weight_bf16` is raw little-endian BF16 bytes of `head.weight` row-major
/// `[vocab, dim]`. `x` is f32 `[rows, dim]` (post-RMSNorm activations).
/// Each weight element is widened BF16→f64 via the high-16-bit shift (exact).
pub fn head_proj_ref(
    x: &[f32],
    weight_bf16: &[u8],
    rows: usize,
    dim: usize,
    vocab: usize,
) -> Result<Vec<f32>, String> {
    if dim == 0 || vocab == 0 {
        return Err(err("head_proj_ref: dim and vocab must be > 0"));
    }
    if x.len() != rows * dim {
        return Err(err(format!(
            "head_proj_ref: x len {} != rows*dim {}",
            x.len(),
            rows * dim
        )));
    }
    let need = vocab
        .checked_mul(dim)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| err("head_proj_ref: weight size overflow"))?;
    if weight_bf16.len() < need {
        return Err(err(format!(
            "head_proj_ref: weight too short (have {} need {need})",
            weight_bf16.len()
        )));
    }

    let mut out = vec![0.0f32; rows * vocab];
    for r in 0..rows {
        let xbase = r * dim;
        for v in 0..vocab {
            let wbase = v * dim * 2;
            let mut acc = 0.0f64;
            for k in 0..dim {
                let bits = u16::from_le_bytes([
                    weight_bf16[wbase + 2 * k],
                    weight_bf16[wbase + 2 * k + 1],
                ]);
                let w = f32::from_bits((bits as u32) << 16) as f64;
                acc += (x[xbase + k] as f64) * w;
            }
            out[r * vocab + v] = acc as f32;
        }
    }
    Ok(out)
}

/// End-to-end f64 oracle for the head pipeline:
/// `hc_head` → `rms_norm` → `head_proj`.
///
/// `x` is f32 `[rows, hc_mult, dim]`. HC / norm weights are f32 slices
/// (already widened). `head_weight_bf16` is raw BF16 bytes.
pub fn parent_head_ref(
    x: &[f32],
    hc_fn: &[f32],
    hc_scale: &[f32],
    hc_base: &[f32],
    norm_weight: &[f32],
    head_weight_bf16: &[u8],
    rows: usize,
    hc_mult: usize,
    dim: usize,
    vocab: usize,
    norm_eps: f64,
    hc_eps: f64,
) -> Result<Vec<f32>, String> {
    let y = hc_head_ref(
        x, hc_fn, hc_scale, hc_base, rows, hc_mult, dim, norm_eps, hc_eps,
    )?;
    let normed = rms_norm_ref(&y, norm_weight, norm_eps, dim);
    // Match the GPU path: round activations to BF16 before the projection so
    // the oracle agrees with `gemm_bf16_mfma` staging, not just the pure-f32
    // ParallelHead formulation.
    let mut normed_bf16 = normed;
    for v in &mut normed_bf16 {
        *v = round_to_bf16(*v);
    }
    head_proj_ref(&normed_bf16, head_weight_bf16, rows, dim, vocab)
}

// ── Plog bridge ─────────────────────────────────────────────────────────────

/// Download logits row by row and append to an open [`PlogWriter`].
///
/// Streams one vocab-row at a time so a multi-thousand-token capture never
/// materialises the full `n_tokens × vocab` host tensor.
pub fn parent_logits_to_plog(
    gpu: &Gpu,
    logits: &GpuTensor,
    rows: usize,
    vocab: usize,
    w: &mut PlogWriter,
) -> Result<(), String> {
    if rows == 0 {
        return Err(err("parent_logits_to_plog: rows must be > 0"));
    }
    if vocab == 0 {
        return Err(err("parent_logits_to_plog: vocab must be > 0"));
    }
    require_dtype(logits, DType::F32, "parent_logits_to_plog logits")?;
    require_elems(logits, rows * vocab, "parent_logits_to_plog logits")?;

    let mut row_bytes = vec![0u8; vocab * 4];
    for r in 0..rows {
        let byte_off = r
            .checked_mul(vocab * 4)
            .ok_or_else(|| err("parent_logits_to_plog: row offset overflow"))?;
        gpu.hip
            .memcpy_dtoh_at(&mut row_bytes, &logits.buf, byte_off)
            .map_err(|e| err(format!("parent_logits_to_plog download row {r}: {e:?}")))?;
        let row = bytes_as_f32_slice(&row_bytes);
        w.push_row(row)?;
    }
    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn require_dtype(t: &GpuTensor, want: DType, name: &str) -> Result<(), String> {
    if t.dtype != want {
        return Err(err(format!(
            "{name} dtype {:?} != expected {want:?}",
            t.dtype
        )));
    }
    Ok(())
}

fn require_elems(t: &GpuTensor, n: usize, name: &str) -> Result<(), String> {
    // Prefer byte capacity: sub-views report numel via shape, full tensors too.
    let need = n
        .checked_mul(t.dtype.size())
        .ok_or_else(|| err(format!("{name}: elem byte size overflow")))?;
    if t.buf.size() < need {
        return Err(err(format!(
            "{name} too small (have {} bytes need {need} for {n} × {:?} elems)",
            t.buf.size(),
            t.dtype
        )));
    }
    Ok(())
}

fn free_scratch(gpu: &mut Gpu, t: GpuTensor) {
    let _ = gpu.free_tensor(t);
}

fn stage_f32_as_bf16(
    gpu: &Gpu,
    src_f32: &GpuTensor,
    dst_bf16: &GpuTensor,
    nelems: usize,
) -> Result<(), String> {
    require_dtype(src_f32, DType::F32, "stage_f32_as_bf16 src")?;
    require_dtype(dst_bf16, DType::BF16, "stage_f32_as_bf16 dst")?;
    require_elems(src_f32, nelems, "stage_f32_as_bf16 src")?;
    require_elems(dst_bf16, nelems, "stage_f32_as_bf16 dst")?;

    let mut f32_bytes = vec![0u8; nelems * 4];
    gpu.hip
        .memcpy_dtoh(&mut f32_bytes, &src_f32.buf)
        .map_err(|e| err(format!("stage_f32_as_bf16 download: {e:?}")))?;
    let vals = bytes_as_f32_slice(&f32_bytes);
    let mut bf16_bytes = vec![0u8; nelems * 2];
    for (i, &v) in vals.iter().enumerate() {
        let bits = (round_to_bf16(v).to_bits() >> 16) as u16;
        let b = bits.to_le_bytes();
        bf16_bytes[2 * i] = b[0];
        bf16_bytes[2 * i + 1] = b[1];
    }
    gpu.hip
        .memcpy_htod(&dst_bf16.buf, &bf16_bytes)
        .map_err(|e| err(format!("stage_f32_as_bf16 upload: {e:?}")))?;
    Ok(())
}

fn f32_slice_as_le_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn bytes_as_f32_slice(bytes: &[u8]) -> &[f32] {
    debug_assert_eq!(bytes.len() % 4, 0);
    debug_assert_eq!(bytes.as_ptr() as usize % std::mem::align_of::<f32>(), 0);
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }
}

// ── Unit tests (host-testable) ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parent::plog::PlogReader;
    use std::path::PathBuf;

    fn pack_bf16(vals: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 2);
        for &v in vals {
            let bits = (round_to_bf16(v).to_bits() >> 16) as u16;
            out.extend_from_slice(&bits.to_le_bytes());
        }
        out
    }

    #[test]
    fn embed_gather_indexing_and_expand() {
        let vocab = 8usize;
        let dim = 4usize;
        let hc = 4usize;
        // table[id, d] = id * 10 + d
        let mut table_f = vec![0.0f32; vocab * dim];
        for id in 0..vocab {
            for d in 0..dim {
                table_f[id * dim + d] = (id * 10 + d) as f32;
            }
        }
        let table = pack_bf16(&table_f);
        let ids = [0u32, 3, 7];
        let out = embed_gather_ref(&table, &ids, vocab, dim, hc).unwrap();
        assert_eq!(out.len(), ids.len() * hc * dim);
        for (r, &tid) in ids.iter().enumerate() {
            for h in 0..hc {
                for d in 0..dim {
                    let got = out[(r * hc + h) * dim + d];
                    let want = round_to_bf16((tid as usize * 10 + d) as f32);
                    assert_eq!(got, want, "r={r} h={h} d={d}");
                }
            }
        }
    }

    #[test]
    fn embed_gather_rejects_oob_token() {
        let vocab = 4usize;
        let dim = 2usize;
        let table = pack_bf16(&[0.0; 8]);
        let err = embed_gather_ref(&table, &[0, 4], vocab, dim, 4).unwrap_err();
        assert!(
            err.contains("out of range"),
            "expected oob refusal, got: {err}"
        );
    }

    #[test]
    fn scratch_bytes_formula() {
        let max_rows = 16usize;
        let expect = max_rows * PARENT_DIM * 4
            + max_rows * PARENT_DIM * 4
            + max_rows * PARENT_DIM * 2
            + PARENT_DIM * 2
            + PARENT_VOCAB * 4;
        // Constructor needs a GPU; validate the closed-form the constructor uses.
        assert_eq!(expect, 16 * 4096 * 4 * 2 + 16 * 4096 * 2 + 4096 * 2 + 129_280 * 4);
        // 1K-token capture streaming 16 rows at a time.
        let peak = ParentHeadScratch::peak_logits_capture_bytes(1024, 16);
        let logits_tile = 16 * PARENT_VOCAB * 4;
        assert_eq!(peak, logits_tile + expect);
        // Sanity: well under a full 1K×vocab materialisation (~530 MiB).
        let full = 1024 * PARENT_VOCAB * 4;
        assert!(peak < full / 4, "peak {peak} should stream, full={full}");
    }

    #[test]
    fn head_proj_ref_identity_like() {
        // dim=2, vocab=3, one row. weight rows are e0, e1-ish.
        let x = [1.0f32, 2.0];
        let w_f = [
            1.0, 0.0, // → 1
            0.0, 1.0, // → 2
            1.0, 1.0, // → 3
        ];
        let w = pack_bf16(&w_f);
        let y = head_proj_ref(&x, &w, 1, 2, 3).unwrap();
        assert_eq!(y.len(), 3);
        assert!((y[0] - 1.0).abs() < 1e-5);
        assert!((y[1] - 2.0).abs() < 1e-5);
        assert!((y[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn plog_bridge_roundtrip_synthetic() {
        // Host-side stand-in for the GPU bridge: same push_row contract.
        let rows = 3usize;
        let vocab = 5usize;
        let mut logits = vec![0.0f32; rows * vocab];
        for r in 0..rows {
            for v in 0..vocab {
                logits[r * vocab + v] = (r * 10 + v) as f32;
            }
        }

        let dir = std::env::temp_dir().join(format!(
            "ds4_parent_head_plog_{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("synth.plog");

        {
            let mut w = PlogWriter::create(&path, rows, vocab).unwrap();
            for r in 0..rows {
                w.push_row(&logits[r * vocab..(r + 1) * vocab]).unwrap();
            }
            w.finish().unwrap();
        }

        let reader = PlogReader::open(&path).unwrap();
        assert_eq!(reader.n_tokens(), rows);
        assert_eq!(reader.vocab(), vocab);
        for r in 0..rows {
            let row = reader.row(r).unwrap();
            assert_eq!(row, &logits[r * vocab..(r + 1) * vocab]);
        }
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
        let _: PathBuf = dir; // silence
    }

    /// Host-side mirror of [`parent_logits_to_plog`]'s row slicing, proving
    /// the bridge's byte math without a GPU.
    #[test]
    fn plog_bridge_row_slicing_matches_layout() {
        let rows = 4usize;
        let vocab = 7usize;
        let mut logits = vec![0.0f32; rows * vocab];
        for i in 0..logits.len() {
            logits[i] = i as f32 * 0.5;
        }
        // Simulate the per-row byte view the GPU download produces.
        let bytes = f32_slice_as_le_bytes(&logits);
        assert_eq!(bytes.len(), rows * vocab * 4);
        for r in 0..rows {
            let off = r * vocab * 4;
            let row = bytes_as_f32_slice(&bytes[off..off + vocab * 4]);
            assert_eq!(row, &logits[r * vocab..(r + 1) * vocab]);
        }
    }
}
