// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! DFlash draft forward pass — native Rust+HIP.
//!
//! Minimal dependency surface: only reads HFQ draft files (arch_id = 20),
//! writes F32 GpuTensor weights, and runs a bidirectional cross-attention
//! Qwen3-flavored decoder over a block of masked positions.
//!
//! The draft model does not own a vocab head. Its output is the final
//! hidden state per block position; the caller applies the target's
//! `lm_head` to map to logits. This matches the upstream z-lab/dflash
//! reference and lets a single tokenizer / embedding table be shared.
//!
//! Architectural notes:
//! - 5-layer Qwen3 decoder, all full attention, non-causal.
//! - Per-layer cross-attention over `target_hidden` (the projected
//!   concatenation of hidden states from a configured set of target
//!   layers, default `[1, 8, 15, 22, 29]` for a 32-layer target).
//! - Q length = `block_size`, K/V length = `ctx_len + block_size`
//!   (K/V = concat of projected target_hidden and current hidden_states).
//! - MVP simplification: draft has NO persistent KV cache; `k_ctx` /
//!   `v_ctx` are recomputed from the (caller-managed) cumulative
//!   `target_hidden` buffer on every step. This is functionally
//!   equivalent to the reference's cropped draft-KV cache and avoids
//!   one whole layer of persistence bookkeeping.

use crate::hfq::{load_awq_scale, HfqFile};
use crate::llama::WeightTensor;
use hip_bridge::{Graph, GraphExec, HipResult};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::{HashMap, HashSet};

/// Max rows per call into `gemm_dispatch` for the MQ (FWHT-rotated)
/// path. The activation rotation scratch (`DflashScratch.mq_x_rot`) is
/// sized to this many rows × `max(inter, q_dim, num_extract * hidden)`,
/// regardless of context length. Calls with `batch > MQ_X_ROT_CHUNK_ROWS`
/// are chunked transparently inside `gemm_dispatch`.
///
/// Sizing rationale: at chunk=1024 and 27B (ne*h = 25600 floats per row),
/// the scratch is `1024 × 25600 × 4 ≈ 100 MB`. The pre-2026-05-15
/// allocator sized this buffer to `max_seq × ne × h`, which at ctx=17K
/// reached 1.74 GB on 27B — a multi-GB waste that scaled with `max_seq`.
/// Chunking adds `ceil(batch / 1024)` extra kernel launches on the
/// first-call `fc` rotation (one-shot per prompt, negligible vs prefill).
const MQ_X_ROT_CHUNK_ROWS: usize = 1024;

// ─── Config ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DflashConfig {
    pub n_layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub block_size: usize,
    pub mask_token_id: u32,
    pub target_layer_ids: Vec<usize>,
    pub num_target_layers: usize,
}

impl DflashConfig {
    /// Returns the number of target hidden layers concatenated into fc input.
    pub fn num_extract(&self) -> usize {
        self.target_layer_ids.len()
    }

    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// Parse from an HFQ file's metadata JSON. Expects the top-level
    /// `dflash` object written by `dflash_convert`.
    pub fn from_hfq(hfq: &HfqFile) -> Option<Self> {
        let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).ok()?;
        let df = meta.get("dflash")?;

        let n_layers = df.get("num_hidden_layers").and_then(|v| v.as_u64())? as usize;
        let hidden = df.get("hidden_size").and_then(|v| v.as_u64())? as usize;
        let intermediate = df.get("intermediate_size").and_then(|v| v.as_u64())? as usize;
        let n_heads = df.get("num_attention_heads").and_then(|v| v.as_u64())? as usize;
        let n_kv_heads = df.get("num_key_value_heads").and_then(|v| v.as_u64())? as usize;
        let head_dim = df
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or((hidden / n_heads) as u64) as usize;
        let vocab_size = df.get("vocab_size").and_then(|v| v.as_u64())? as usize;
        let norm_eps = df
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6) as f32;
        let rope_theta = df
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(10_000_000.0) as f32;
        let block_size = df.get("block_size").and_then(|v| v.as_u64())? as usize;
        let mask_token_id = df.get("mask_token_id").and_then(|v| v.as_u64())? as u32;
        let target_layer_ids: Vec<usize> = df
            .get("target_layer_ids")?
            .as_array()?
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as usize))
            .collect();
        let num_target_layers = df.get("num_target_layers").and_then(|v| v.as_u64())? as usize;

        Some(DflashConfig {
            n_layers,
            hidden,
            intermediate,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            norm_eps,
            rope_theta,
            block_size,
            mask_token_id,
            target_layer_ids,
            num_target_layers,
        })
    }
}

// ─── Weights ───────────────────────────────────────────────────────────────

pub struct DflashLayerWeights {
    pub attn_norm: GpuTensor, // [hidden] — F32, RMSNorm weight
    pub wq: WeightTensor,     // [q_dim, hidden]
    pub wk: WeightTensor,     // [kv_dim, hidden]
    pub wv: WeightTensor,     // [kv_dim, hidden]
    pub wo: WeightTensor,     // [hidden, q_dim]
    pub q_norm: GpuTensor,    // [head_dim] — F32
    pub k_norm: GpuTensor,    // [head_dim] — F32
    pub ffn_norm: GpuTensor,  // [hidden] — F32
    pub w_gate: WeightTensor, // [intermediate, hidden]
    pub w_up: WeightTensor,   // [intermediate, hidden]
    pub w_down: WeightTensor, // [hidden, intermediate]
}

pub struct DflashWeights {
    /// `fc`: Linear(num_extract × hidden → hidden). Shape: [hidden, num_extract × hidden].
    pub fc: WeightTensor,
    pub hidden_norm: GpuTensor, // [hidden] — F32
    pub norm: GpuTensor,        // [hidden] — F32, final output norm
    pub layers: Vec<DflashLayerWeights>,
    /// True when at least one matrix weight is MQ4G256 — drives whether
    /// the draft_forward path needs to allocate FWHT rotation scratches.
    pub has_mq: bool,
}

/// Load a F32-only tensor (norms, embedding-shaped scalars). Always F32 on GPU.
fn hfq_tensor_f32(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    shape: Vec<usize>,
) -> HipResult<GpuTensor> {
    let (info, data) = hfq
        .tensor_data(name)
        .unwrap_or_else(|| panic!("dflash tensor missing: {name}"));
    let f32_data: Vec<f32> = match info.quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| crate::llama::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        q => panic!("dflash: unsupported quant_type {q} for {name}"),
    };
    let expected: usize = shape.iter().product();
    assert_eq!(
        f32_data.len(),
        expected,
        "dflash: shape mismatch for {name}: have {}, expected {}",
        f32_data.len(),
        expected,
    );
    gpu.upload_f32(&f32_data, &shape)
}

/// Load a matrix tensor as a `WeightTensor` carrying its native dtype.
/// Supported quant_types:
///   1  (F16)      → lifted to F32 on GPU (legacy path).
///   2  (F32)      → uploaded as F32.
///   13 (MQ4-G256) → uploaded raw, kernel dispatch will FWHT-rotate x at use.
///   15 (MQ6-G256) → uploaded raw, kernel dispatch will FWHT-rotate x at use.
///   17 (MQ3-G256) → uploaded raw, kernel dispatch will FWHT-rotate x at use.
///
/// `shape = [m, k]` so m=output_dim and k=input_dim. The HFQ index stores
/// the unaligned byte length; for MQ formats we skip shape verification (the
/// quantized bytes are not a function of m*k alone — group padding can add
/// up to 255 trailing bytes per row group).
fn hfq_weight(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    let (info, data) = hfq
        .tensor_data(name)
        .unwrap_or_else(|| panic!("dflash tensor missing: {name}"));
    let mut wt = match info.quant_type {
        1 => {
            // F16 on disk. Default: upload as F16 (no lift) and dispatch through
            // the mw16 WMMA kernel — 3-5× faster draft at B=16 on gfx1100 than
            // the F32 lift path (which bypassed WMMA entirely via the naive
            // gemm_f32_batched kernel at ~100 GB/s / 10 % peak).
            //
            // HIPFIRE_DRAFT_F16=0 falls back to the legacy F16→F32 lift for
            // A/B comparison.
            let use_f16 = crate::config::get().draft_f16;
            if use_f16 {
                assert_eq!(
                    data.len(),
                    m * k * 2,
                    "dflash {name} F16 byte-size mismatch"
                );
                let buf = gpu.upload_raw(data, &[m * k])?;
                Ok::<WeightTensor, hip_bridge::HipError>(WeightTensor {
                    buf,
                    gpu_dtype: DType::F16,
                    m,
                    k,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                })
            } else {
                let f32_data: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|c| crate::llama::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                assert_eq!(f32_data.len(), m * k, "dflash {name} F16 size mismatch");
                let buf = gpu.upload_f32(&f32_data, &[m * k])?;
                Ok(WeightTensor {
                    buf,
                    gpu_dtype: DType::F32,
                    m,
                    k,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                })
            }
        }
        2 => {
            let f32_data: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(f32_data.len(), m * k, "dflash {name} F32 size mismatch");
            let buf = gpu.upload_f32(&f32_data, &[m * k])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        13 => {
            // MQ4-G256: 136 bytes per 256 weights. The buffer is opaque to
            // the engine; the gemm_hfq4g256 kernel reads it directly.
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ4G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        15 => {
            // MQ6-G256: 200 bytes per 256 weights. Same opaque-buffer pattern
            // as MQ4/MQ3; dispatch rotates activations and calls HFQ6 GEMM.
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ6G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        17 => {
            // MQ3-G256: 104 bytes per 256 weights. Same opaque-buffer pattern
            // as MQ4. Dispatch path (`gemm_dispatch`) routes through
            // `rotate_x_mq_batched` + `gemm_hfq3g256_batched_lmhead`.
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ3G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        q => panic!("dflash: unsupported matrix quant_type {q} for {name}"),
    }?;
    // AWQ sidecar attachment — same pattern as hfq.rs::load_weight_tensor
    // and qwen35.rs::load_weight_tensor. Routed through the centralized
    // `DType::supports_awq_sidecar` allow-list so future widening (MQ6,
    // MQ2, MQ3-Lloyd, MFP4) is a single helper edit. Sidecar absent →
    // `awq_scale` stays None, dispatch path matches the pre-fix behavior.
    if wt.gpu_dtype.supports_awq_sidecar() {
        wt.awq_scale = load_awq_scale(hfq, gpu, name, k);
    }
    Ok(wt)
}

impl DflashWeights {
    pub fn load(gpu: &mut Gpu, hfq: &HfqFile, cfg: &DflashConfig) -> HipResult<Self> {
        let fc = hfq_weight(
            hfq,
            gpu,
            "fc.weight",
            cfg.hidden,
            cfg.num_extract() * cfg.hidden,
        )?;
        let hidden_norm = hfq_tensor_f32(hfq, gpu, "hidden_norm.weight", vec![cfg.hidden])?;
        let norm = hfq_tensor_f32(hfq, gpu, "norm.weight", vec![cfg.hidden])?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("layers.{i}");
            let layer = DflashLayerWeights {
                attn_norm: hfq_tensor_f32(
                    hfq,
                    gpu,
                    &format!("{p}.input_layernorm.weight"),
                    vec![cfg.hidden],
                )?,
                wq: hfq_weight(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.q_proj.weight"),
                    cfg.q_dim(),
                    cfg.hidden,
                )?,
                wk: hfq_weight(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.k_proj.weight"),
                    cfg.kv_dim(),
                    cfg.hidden,
                )?,
                wv: hfq_weight(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.v_proj.weight"),
                    cfg.kv_dim(),
                    cfg.hidden,
                )?,
                wo: hfq_weight(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.o_proj.weight"),
                    cfg.hidden,
                    cfg.q_dim(),
                )?,
                q_norm: hfq_tensor_f32(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.q_norm.weight"),
                    vec![cfg.head_dim],
                )?,
                k_norm: hfq_tensor_f32(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.k_norm.weight"),
                    vec![cfg.head_dim],
                )?,
                ffn_norm: hfq_tensor_f32(
                    hfq,
                    gpu,
                    &format!("{p}.post_attention_layernorm.weight"),
                    vec![cfg.hidden],
                )?,
                w_gate: hfq_weight(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.gate_proj.weight"),
                    cfg.intermediate,
                    cfg.hidden,
                )?,
                w_up: hfq_weight(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.up_proj.weight"),
                    cfg.intermediate,
                    cfg.hidden,
                )?,
                w_down: hfq_weight(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.down_proj.weight"),
                    cfg.hidden,
                    cfg.intermediate,
                )?,
            };
            layers.push(layer);
        }

        let has_mq = std::iter::once(&fc)
            .chain(layers.iter().flat_map(|l| {
                [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down].into_iter()
            }))
            .any(|w| {
                matches!(
                    w.gpu_dtype,
                    DType::MQ4G256 | DType::MQ6G256 | DType::MQ3G256
                )
            });
        if has_mq {
            // MQ dispatch needs the engine's FWHT sign tables uploaded
            // (matches `gemv_mq4g256_with_rotate`'s setup).
            gpu.ensure_mq_signs()?;
        }

        Ok(DflashWeights {
            fc,
            hidden_norm,
            norm,
            layers,
            has_mq,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        // free_all (not .buf) so the awq_scale / paro sidecars are released too —
        // on an AWQ-trunk drafter every weight carries an awq_scale GpuTensor, so
        // .buf-only freeing leaks one tensor per weight per layer on each unload.
        self.fc.free_all(gpu);
        let _ = gpu.free_tensor(self.hidden_norm);
        let _ = gpu.free_tensor(self.norm);
        for l in self.layers {
            let _ = gpu.free_tensor(l.attn_norm);
            l.wq.free_all(gpu);
            l.wk.free_all(gpu);
            l.wv.free_all(gpu);
            l.wo.free_all(gpu);
            let _ = gpu.free_tensor(l.q_norm);
            let _ = gpu.free_tensor(l.k_norm);
            let _ = gpu.free_tensor(l.ffn_norm);
            l.w_gate.free_all(gpu);
            l.w_up.free_all(gpu);
            l.w_down.free_all(gpu);
        }
    }
}

// ─── target_hidden bookkeeping ───────────────────────────────────────────────

/// Encapsulated cursors describing how much of the draft's `target_hidden`
/// context is live on GPU. Three quantities that MUST move together:
///
/// - `uploaded_rows` — rows of `target_hidden` already uploaded H2D (the
///   delta-upload watermark `draft_forward` reads/advances);
/// - `abs_positions` — the absolute (pre-compaction) position of every
///   populated row; its length is always exactly `uploaded_rows`;
/// - `proj_cached_rows` — rows whose per-layer `k_ctx`/`v_ctx` projection is
///   cached (`≤ uploaded_rows`).
///
/// The fields are PRIVATE (this is a submodule), so the only ways to mutate
/// them are the invariant-preserving operations below — `seed_prompt`,
/// `append_committed`, `rebuild_after_eviction`, `reset`, plus the two
/// `draft_forward`-owned watermarks `mark_uploaded` / `mark_proj_cached`.
/// A caller can no longer set `uploaded_rows` without `abs_positions` staying
/// consistent (the desync that bit the generic DFlash path in 89856eab and
/// the #462 class): that error is now defined out of existence — it does not
/// compile.
mod target_hidden_log {
    /// See module-level intent. Construct via [`TargetHiddenLog::new`].
    #[derive(Default)]
    pub struct TargetHiddenLog {
        uploaded_rows: usize,
        abs_positions: Vec<i32>,
        proj_cached_rows: usize,
        /// Separate K/V-fill watermark for the windowed mode's last
        /// (full-attention) layer. Rows older than `l − swa_w` are not
        /// resident in the proj ring, so the last layer's fill for them runs
        /// in the post-seed backfill (host shadow), tracked here. Always
        /// `== proj_cached_rows` in Legacy mode.
        full_cached_rows: usize,
    }

    impl TargetHiddenLog {
        pub fn new() -> Self {
            Self::default()
        }

        // ── reads ────────────────────────────────────────────────────────
        /// Rows of `target_hidden` already uploaded to GPU (delta watermark).
        pub fn uploaded_rows(&self) -> usize {
            self.uploaded_rows
        }
        /// Absolute (pre-compaction) position of each populated row;
        /// `len() == uploaded_rows`.
        pub fn abs_positions(&self) -> &[i32] {
            &self.abs_positions
        }
        /// Rows whose per-layer k_ctx/v_ctx projection is cached.
        pub fn proj_cached_rows(&self) -> usize {
            self.proj_cached_rows
        }
        /// Rows whose LAST-layer (full-attention) K/V projection is cached
        /// (windowed mode; mirrors `proj_cached_rows` in Legacy).
        pub fn full_cached_rows(&self) -> usize {
            self.full_cached_rows
        }

        // ── invariant-preserving mutations ────────────────────────────────
        /// New-prompt / session boundary: forget all GPU-resident rows.
        pub fn reset(&mut self) {
            self.uploaded_rows = 0;
            self.abs_positions.clear();
            self.proj_cached_rows = 0;
            self.full_cached_rows = 0;
        }

        /// Prompt-prefill seed: `rows` contiguous rows `[0..rows)` are live on
        /// GPU at contiguous absolute positions.
        pub fn seed_prompt(&mut self, rows: usize) {
            self.uploaded_rows = rows;
            self.abs_positions = (0..rows as i32).collect();
        }

        /// Divergent-render resume: drop the projection cache back to a
        /// checkpoint position so the next `draft_forward` re-projects from it.
        /// The last-layer (windowed full) watermark clamps down too — rows at
        /// >= ckpt in its ring are stale (pre-divergence); they re-fill from
        /// the live window as new rows stream (out-of-window stale rows are a
        /// τ-only degradation; verify stays exact).
        pub fn set_resume_checkpoint(&mut self, ckpt: usize) {
            self.proj_cached_rows = ckpt;
            self.full_cached_rows = self.full_cached_rows.min(ckpt);
        }

        /// Post-commit append: `n` newly committed rows starting at logical
        /// `base_pos` (with the target KV `compact_offset`) become live. After
        /// this, `uploaded_rows == base_pos + n` and `abs_positions` has one
        /// entry per committed row.
        pub fn append_committed(&mut self, base_pos: usize, n: usize, compact_offset: i32) {
            debug_assert_eq!(
                self.abs_positions.len(),
                base_pos,
                "append_committed: abs_positions out of sync with base_pos"
            );
            // Release-visible companion to the assert above. A `base_pos` ahead
            // of the row count means positions advanced WITHOUT committing their
            // target_hidden rows, so the buffer keeps an unwritten hole —
            // uninitialized VRAM, i.e. NaN — which poisons every later draft
            // forward, collapses acceptance to zero for the rest of the session,
            // and survives a prompt-cache HIT (whose `seed_prompt` rebuilds
            // `abs_positions` contiguously, hiding the skew but not the hole).
            // The `debug_assert` is compiled out in release, which is exactly how
            // the think-budget force-close path went unnoticed; warn loudly once
            // instead of silently poisoning the drafter.
            if self.abs_positions.len() != base_pos {
                static GAP_WARNED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !GAP_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "[dflash] WARNING target-hidden row gap: have {} rows but committing at \
                         position {} ({} row(s) never written) — acceptance will degrade until \
                         the drafter is reseeded. This is a bug in whichever path advanced the \
                         target without committing its hidden rows.",
                        self.abs_positions.len(),
                        base_pos,
                        base_pos.saturating_sub(self.abs_positions.len())
                    );
                }
            }
            self.uploaded_rows = base_pos + n;
            for p in 0..n {
                self.abs_positions
                    .push(base_pos as i32 + p as i32 + compact_offset);
            }
        }

        /// Post-eviction rebuild: `new_abs` is the compacted absolute-position
        /// list (one entry per retained row). Replaces the row layout and
        /// invalidates the projection cache (row indices shifted).
        pub fn rebuild_after_eviction(&mut self, new_abs: Vec<i32>) {
            self.uploaded_rows = new_abs.len();
            self.abs_positions = new_abs;
            self.proj_cached_rows = 0;
            self.full_cached_rows = 0;
        }

        /// Invalidate only the per-layer projection cache (eviction mirror that
        /// keeps the uploaded rows but shifts their indices).
        pub fn invalidate_proj_cache(&mut self) {
            self.proj_cached_rows = 0;
            self.full_cached_rows = 0;
        }

        /// `draft_forward`-owned: record that `l` rows are now uploaded H2D.
        pub fn mark_uploaded(&mut self, l: usize) {
            self.uploaded_rows = l;
        }

        /// `draft_forward`-owned: record that `l` rows are now projection-cached.
        /// Also advances the last-layer watermark: a completed forward filled
        /// every layer's K/V through `l` (windowed mode relies on the
        /// post-seed backfill for the out-of-window prefix).
        pub fn mark_proj_cached(&mut self, l: usize) {
            self.proj_cached_rows = l;
            self.full_cached_rows = self.full_cached_rows.max(l);
        }
        /// Windowed backfill-owned: record that the last (full-attention)
        /// layer's K/V cache is valid through row `l`.
        pub fn mark_full_cached(&mut self, l: usize) {
            self.full_cached_rows = l;
        }
    }
}
pub use target_hidden_log::TargetHiddenLog;

// ─── Scratch ───────────────────────────────────────────────────────────────

/// Draft context mode (HIPFIRE_DFLASH_WINDOW).
///
/// `Legacy` (default): every draft layer attends over the full context; all
/// context-indexed buffers are sized to the (capped) ctx capacity and a
/// request past the cap falls back to AR in the daemon.
///
/// `Windowed { w, w_full }`: layers `0..n-2` attend over the last `w` rows
/// (SWA), layer `n-1` over the last `w_full` rows — the NInfer companion-
/// drafter pattern (W=4096 SWA + one full layer) ported inference-side.
/// Every per-layer K/V cache is a ring (`slot = abs_row % window`), so the
/// layout at `l <= w` is byte-identical to Legacy and all fills stay
/// row-local-exact; past `w` acceptance (τ) degrades gracefully instead of
/// the AR cliff. Draft quality never affects emitted tokens (verify-gated).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftCtxMode {
    Legacy,
    Windowed { w: usize, w_full: usize },
}

/// Iterate the contiguous ring segments covering rows `[start, end)` in a
/// ring of `modulus` slots (slot = row % modulus). Yields
/// `(row_start, slot_start, len)` per segment — at most two for any range
/// smaller than `modulus`. `modulus == usize::MAX` degenerates to a single
/// identity segment (Legacy layout), keeping the windowed code path
/// byte-exact with the legacy one when no wrap is possible.
pub fn ring_segments(start: usize, end: usize, modulus: usize) -> Vec<(usize, usize, usize)> {
    debug_assert!(end >= start);
    if start == end {
        return Vec::new();
    }
    if modulus == usize::MAX {
        return vec![(start, start, end - start)];
    }
    let mut out = Vec::with_capacity(2);
    let mut row = start;
    while row < end {
        let slot = row % modulus;
        let len = (modulus - slot).min(end - row);
        out.push((row, slot, len));
        row += len;
    }
    out
}

/// Activation buffers for one forward pass. Sized for up to
/// `max_block_size` query positions and up to `max_ctx_len` context
/// positions. A single scratch is reused across all speculative steps.
pub struct DflashScratch {
    pub max_block_size: usize,
    pub max_ctx_len: usize,

    // Block-sized activations (B rows).
    pub x: GpuTensor,             // [B, hidden] — hidden state rolled across layers
    pub x_norm: GpuTensor,        // [B, hidden]
    pub q: GpuTensor,             // [B, q_dim]
    pub k_noise: GpuTensor,       // [B, kv_dim]
    pub v_noise: GpuTensor,       // [B, kv_dim]
    pub gate: GpuTensor,          // [B, intermediate]
    pub up: GpuTensor,            // [B, intermediate]
    pub gate_up: GpuTensor,       // [B, intermediate]
    pub attn_out: GpuTensor,      // [B, q_dim]
    pub attn_proj: GpuTensor,     // [B, hidden]
    pub residual_attn: GpuTensor, // [B, hidden]
    pub residual_ffn: GpuTensor,  // [B, hidden]

    // Context activations (L rows), where L ≤ max_ctx_len.
    pub target_hidden: GpuTensor,      // [L, num_extract × hidden]
    pub target_hidden_proj: GpuTensor, // [L, hidden]
    pub k_ctx: GpuTensor,              // [L, kv_dim]
    pub v_ctx: GpuTensor,              // [L, kv_dim]

    // Concatenated K/V (L + B rows).
    pub k_cat: GpuTensor, // [L + B, kv_dim]
    pub v_cat: GpuTensor, // [L + B, kv_dim]

    // Positions (i32).
    pub positions_q: GpuTensor, // [B]       i32
    pub positions_k: GpuTensor, // [L + B]   i32

    // FWHT rotation scratch for MQ4 weight paths. Sized to the largest
    // single-call requirement: max(max_ctx × num_extract*hidden,
    // max_block × max_layer_K). Allocated only when DflashWeights.has_mq.
    pub mq_x_rot: Option<GpuTensor>,

    // Encapsulated `target_hidden` cursors: uploaded-row watermark, per-row
    // absolute positions, and projection-cache extent. The delta-upload
    // tracker drops per-cycle H2D from ~90 MB (full ctx at 1100 tokens × 5
    // layers × 4096 × 4 B) to ~700 KB. See `TargetHiddenLog` for the
    // invariant-preserving API (seed_prompt / append_committed /
    // rebuild_after_eviction / reset); raw poking is no longer possible.
    pub thlog: TargetHiddenLog,

    /// Per-layer cache of `k_ctx` and `v_ctx` (post-GEMM-of-target_hidden_proj,
    /// K additionally post-RMSNorm-via-k_norm, both pre-RoPE). Filled
    /// incrementally as draft_forward sees new target_hidden rows.
    ///
    /// The win: without this cache, each `draft_forward` call re-ran 2
    /// big GEMMs per layer over ALL L context rows, even though only the
    /// tail (accept+1 new rows) had changed since the previous cycle. On
    /// 27B at L=512, that cost ~230 ms/cycle. With the cache, only the
    /// delta rows are recomputed and appended — ~5 ms/cycle for typical
    /// τ ≈ 5.
    ///
    /// Lucebox calls the same structure a "rolling target_feat ring" in
    /// its DFlash-on-ggml writeup; this is our equivalent.
    ///
    /// Shapes: each entry is `[max_ctx, kv_dim]` f32.
    /// The valid extent of these caches (rows `[0..thlog.proj_cached_rows())`
    /// have finished fc + hidden_norm projection, per-layer wk/wv GEMMs, and
    /// k_norm; still pre-RoPE). Tracked by `thlog` so it stays consistent with
    /// the upload watermark.
    pub k_ctx_cached: Vec<GpuTensor>,
    pub v_ctx_cached: Vec<GpuTensor>,

    /// Context mode this scratch was sized for (see [`DraftCtxMode`]).
    pub ctx_mode: DraftCtxMode,
    /// Windowed-mode last-layer (full-attention) K/V rings, `[w_full × kvd]`.
    /// The last layer keeps a 4× longer reach than the SWA layers — the only
    /// unbounded-context structure in windowed mode. `None` in Legacy.
    pub k_full_cached: Option<GpuTensor>,
    pub v_full_cached: Option<GpuTensor>,
    /// Windowed-mode last-layer concat assembly, `[(w_full + B) × kvd]`.
    pub k_cat_full: Option<GpuTensor>,
    pub v_cat_full: Option<GpuTensor>,

    /// Per-layer, per-B graph cache for the fixed-shape FFN tail inside
    /// `draft_forward_opts`. The attention/context part depends on `ctx_len`;
    /// this FFN subgraph does not, so it can replay across DFlash cycles.
    pub draft_ffn_graphs: Vec<HashMap<usize, (Graph, GraphExec, Vec<Vec<u8>>)>>,
    pub draft_ffn_warmed_up: Vec<HashSet<usize>>,
}

impl DflashScratch {
    pub fn new(
        gpu: &mut Gpu,
        cfg: &DflashConfig,
        max_block_size: usize,
        max_ctx_len: usize,
    ) -> HipResult<Self> {
        Self::new_with_mq(gpu, cfg, max_block_size, max_ctx_len, false)
    }

    /// Windowed-mode constructor (HIPFIRE_DFLASH_WINDOW). SWA layers
    /// `0..n-2` get `w`-row K/V rings and `(w + B)` concat buffers; the last
    /// layer gets `w_full`-row rings and its own concat pair. All other
    /// buffers are sized exactly as `new_with_mq(gpu, cfg, b, w, ..)` — so
    /// windowed-mode VRAM is the Legacy-at-ctx=w footprint plus one
    /// `w_full`-sized layer (≈270 MB at w_full=32K, kvd=1024, f32).
    /// `max_ctx` is the target's physical context capacity: the rings wrap
    /// forever, so `l` may grow past `w_full` (spans stay suffixes).
    pub fn new_windowed(
        gpu: &mut Gpu,
        cfg: &DflashConfig,
        max_block_size: usize,
        w: usize,
        w_full: usize,
        max_ctx: usize,
        with_mq: bool,
    ) -> HipResult<Self> {
        let kvd = cfg.kv_dim();
        let b = max_block_size;
        // The long-reach layer never has a SHORTER window than the SWA layers
        // (possible when physical_cap < w): its span must contain theirs.
        let w_full = w_full.max(w);
        // Base at ctx=w: SWA rings, (w+B) concat buffers, w-row target_hidden
        // ring — the entire draft footprint except the one long-reach layer.
        let mut s = Self::new_with_mq(gpu, cfg, b, w, with_mq)?;
        // The last (full-attention) layer gets w_full-row rings + its own
        // concat pair; its w-sized base caches are freed.
        if let Some(k) = s.k_ctx_cached.pop() {
            let _ = gpu.free_tensor(k);
        }
        if let Some(v) = s.v_ctx_cached.pop() {
            let _ = gpu.free_tensor(v);
        }
        s.k_full_cached = Some(gpu.alloc_tensor(&[w_full * kvd], DType::F32)?);
        s.v_full_cached = Some(gpu.alloc_tensor(&[w_full * kvd], DType::F32)?);
        s.k_cat_full = Some(gpu.alloc_tensor(&[(w_full + b) * kvd], DType::F32)?);
        s.v_cat_full = Some(gpu.alloc_tensor(&[(w_full + b) * kvd], DType::F32)?);
        // positions_k holds the last w_full context rows + the B noise rows
        // (the forward uploads only that suffix; every layer's span is one).
        let new_positions_k = gpu.alloc_tensor(&[w_full + b], DType::F32)?;
        let _ = gpu.free_tensor(std::mem::replace(&mut s.positions_k, new_positions_k));
        // The ctx bound is the target's physical capacity, not the window —
        // l may cross w_full (the last layer's span just slides).
        s.max_ctx_len = max_ctx;
        s.ctx_mode = DraftCtxMode::Windowed { w, w_full };
        Ok(s)
    }

    /// `with_mq` allocates the FWHT rotation scratch needed when at least
    /// one matrix weight is MQ4-G256. Sized to handle every per-call
    /// rotation in the draft forward.
    pub fn new_with_mq(
        gpu: &mut Gpu,
        cfg: &DflashConfig,
        max_block_size: usize,
        max_ctx_len: usize,
        with_mq: bool,
    ) -> HipResult<Self> {
        let b = max_block_size;
        let l = max_ctx_len;
        let tot = l + b;
        let ne = cfg.num_extract();
        let h = cfg.hidden;
        let inter = cfg.intermediate;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();

        let mq_x_rot = if with_mq {
            // Sized for a CHUNK of the worst-case MQ rotation, not the whole
            // first-call prefix. The rotations called through `gemm_dispatch`
            // are:
            //   - first-call `fc` (target_hidden):  batch up to `l`, w.k = ne*h
            //   - per-cycle wq/wk/wv/gate/up:       batch = b,         w.k = h
            //   - per-cycle wo:                     batch = b,         w.k = q_dim
            //   - per-cycle w_down:                 batch = b,         w.k = intermediate
            //   - first-call wk/wv on prefix:       batch up to `l`,   w.k = h
            //
            // Steady-state cycles only need `b × max(inter, qd, ne*h)`. The
            // first-call rotations against the full prefix used to pin the
            // buffer to `l × ne × h` (1.7 GB at ctx=17K on 27B). That sizing
            // forced VRAM bloat that scales with max_seq.
            //
            // Fix: cap the scratch at `MQ_X_ROT_CHUNK_ROWS × max(inter, qd, ne*h)`
            // floats and chunk any call where `batch × w.k > scratch.size()`
            // inside `gemm_dispatch`. The first-call rotations are split into
            // `ceil(batch / chunk_rows)` smaller GEMMs — adds ~1-2 launches per
            // 1K prefix tokens (negligible vs seconds-scale prefill).
            let widest = MQ_X_ROT_CHUNK_ROWS * std::cmp::max(inter, std::cmp::max(qd, ne * h));
            Some(gpu.alloc_tensor(&[widest], DType::F32)?)
        } else {
            None
        };

        // Per-layer cache buffers for k_ctx/v_ctx (post-norm-for-K, pre-rope).
        // Size each at [max_ctx × kv_dim] f32 = l × kvd × 4 bytes. Memory
        // cost for 16-layer / 4096-ctx / 256-kv_dim draft ≈ 2 × 16 × 4 MB
        // = 128 MB. Trivial vs 24 GB VRAM.
        let mut k_ctx_cached = Vec::with_capacity(cfg.n_layers);
        let mut v_ctx_cached = Vec::with_capacity(cfg.n_layers);
        let mut draft_ffn_graphs = Vec::with_capacity(cfg.n_layers);
        let mut draft_ffn_warmed_up = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            k_ctx_cached.push(gpu.alloc_tensor(&[l * kvd], DType::F32)?);
            v_ctx_cached.push(gpu.alloc_tensor(&[l * kvd], DType::F32)?);
            draft_ffn_graphs.push(HashMap::new());
            draft_ffn_warmed_up.push(HashSet::new());
        }

        Ok(DflashScratch {
            max_block_size: b,
            max_ctx_len: l,

            x: gpu.alloc_tensor(&[b * h], DType::F32)?,
            x_norm: gpu.alloc_tensor(&[b * h], DType::F32)?,
            q: gpu.alloc_tensor(&[b * qd], DType::F32)?,
            k_noise: gpu.alloc_tensor(&[b * kvd], DType::F32)?,
            v_noise: gpu.alloc_tensor(&[b * kvd], DType::F32)?,
            gate: gpu.alloc_tensor(&[b * inter], DType::F32)?,
            up: gpu.alloc_tensor(&[b * inter], DType::F32)?,
            gate_up: gpu.alloc_tensor(&[b * inter], DType::F32)?,
            attn_out: gpu.alloc_tensor(&[b * qd], DType::F32)?,
            attn_proj: gpu.alloc_tensor(&[b * h], DType::F32)?,
            residual_attn: gpu.alloc_tensor(&[b * h], DType::F32)?,
            residual_ffn: gpu.alloc_tensor(&[b * h], DType::F32)?,

            target_hidden: gpu.alloc_tensor(&[l * ne * h], DType::F32)?,
            target_hidden_proj: gpu.alloc_tensor(&[l * h], DType::F32)?,
            k_ctx: gpu.alloc_tensor(&[l * kvd], DType::F32)?,
            v_ctx: gpu.alloc_tensor(&[l * kvd], DType::F32)?,

            k_cat: gpu.alloc_tensor(&[tot * kvd], DType::F32)?,
            v_cat: gpu.alloc_tensor(&[tot * kvd], DType::F32)?,

            positions_q: gpu.alloc_tensor(&[b], DType::F32)?,
            positions_k: gpu.alloc_tensor(&[tot], DType::F32)?,

            mq_x_rot,
            thlog: TargetHiddenLog::new(),
            k_ctx_cached,
            v_ctx_cached,
            ctx_mode: DraftCtxMode::Legacy,
            k_full_cached: None,
            v_full_cached: None,
            k_cat_full: None,
            v_cat_full: None,
            draft_ffn_graphs,
            draft_ffn_warmed_up,
        })
    }

    /// Reset the incremental-upload tracker for target_hidden. Call this
    /// at the start of a new prompt / session — otherwise stale tracker
    /// state from a prior prompt would cause the next draft_forward to
    /// skip required rows. Also clears the draft-ctx projection cache so
    /// the first draft_forward after reset does a full rebuild.
    pub fn reset_upload_tracking(&mut self) {
        self.thlog.reset();
    }

    /// The dst ring modulus for absolute-row scatters into
    /// `target_hidden` — `usize::MAX` (identity) in Legacy, `w` in
    /// Windowed mode.
    pub fn ctx_modulus(&self) -> usize {
        match self.ctx_mode {
            DraftCtxMode::Legacy => usize::MAX,
            DraftCtxMode::Windowed { w, .. } => w,
        }
    }

    /// Invalidate the per-layer k_ctx/v_ctx projection cache. Called from
    /// `apply_eviction_retain_to_draft` (in speculative.rs) when CASK
    /// evicts positions — the cached rows no longer correspond to the
    /// right absolute positions, so the simplest correct thing is to
    /// rebuild on the next cycle. A finer mirror (applying retain_mask
    /// to the cache) could preserve the cache across eviction but adds
    /// complexity; the rebuild cost is bounded by one slow cycle per
    /// eviction which is rare relative to total cycles.
    pub fn invalidate_draft_ctx_cache(&mut self) {
        self.thlog.invalidate_proj_cache();
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for per_layer in self.draft_ffn_graphs {
            for (_, (graph, exec, _blobs)) in per_layer {
                let _ = gpu.hip.graph_exec_destroy(exec);
                let _ = gpu.hip.graph_destroy(graph);
            }
        }
        let _ = gpu.free_tensor(self.x);
        let _ = gpu.free_tensor(self.x_norm);
        let _ = gpu.free_tensor(self.q);
        let _ = gpu.free_tensor(self.k_noise);
        let _ = gpu.free_tensor(self.v_noise);
        let _ = gpu.free_tensor(self.gate);
        let _ = gpu.free_tensor(self.up);
        let _ = gpu.free_tensor(self.gate_up);
        let _ = gpu.free_tensor(self.attn_out);
        let _ = gpu.free_tensor(self.attn_proj);
        let _ = gpu.free_tensor(self.residual_attn);
        let _ = gpu.free_tensor(self.residual_ffn);
        let _ = gpu.free_tensor(self.target_hidden);
        let _ = gpu.free_tensor(self.target_hidden_proj);
        let _ = gpu.free_tensor(self.k_ctx);
        let _ = gpu.free_tensor(self.v_ctx);
        let _ = gpu.free_tensor(self.k_cat);
        let _ = gpu.free_tensor(self.v_cat);
        let _ = gpu.free_tensor(self.positions_q);
        let _ = gpu.free_tensor(self.positions_k);
        for t in self.k_ctx_cached {
            let _ = gpu.free_tensor(t);
        }
        for t in self.v_ctx_cached {
            let _ = gpu.free_tensor(t);
        }
        for t in [
            self.k_full_cached,
            self.v_full_cached,
            self.k_cat_full,
            self.v_cat_full,
        ]
        .into_iter()
        .flatten()
        {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.mq_x_rot {
            let _ = gpu.free_tensor(t);
        }
    }
}

// ─── Forward ───────────────────────────────────────────────────────────────

/// Dispatch a batched GEMM by weight dtype.
///
/// Layout (row-major):
///   x [batch × k]  F32 input activations
///   w.buf [m × k]  weight, format depends on w.gpu_dtype
///   y [batch × m]  F32 output
///
/// For MQ-G256, the kernel needs the input FWHT-rotated. We do that into
/// `mq_x_rot` (sized to the per-call max in `DflashScratch`), then call the
/// HFQ4-G256 GEMM kernel against the pre-rotated weights.
fn gemm_dispatch(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &WeightTensor,
    y: &GpuTensor,
    batch: usize,
    mq_x_rot: Option<&GpuTensor>,
) -> HipResult<()> {
    // Route HFQ4/MQ4 batched paths through the WMMA lm_head helper — the
    // DFlash draft forward's per-layer projections (wq/wk/wv/wo/gate/up/down)
    // and fc are ALL batched > 1, and share the same "y = A @ x" shape as
    // lm_head. Using the WMMA residual-pre-zeroed path here unlocks ~8-10×
    // on the same matmuls without touching AR-greedy numerics (AR on
    // Qwen3.5 doesn't call `gpu.gemm_hfq4g256` directly — it uses the
    // fused qkvza / gate_up / residual WMMA variants instead).
    // HIPFIRE_DRAFT_GEMM_DUMP=1: per-call (dtype, M, K, B, us, GB/s) dump for
    // draft GEMM triage. Cached via OnceLock so the fast path pays a single
    // atomic load per call rather than an env lookup.
    static DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let dump = *DUMP.get_or_init(|| crate::config::get().draft_gemm_dump);
    if dump {
        gpu.hip.device_synchronize()?;
    }
    let t0 = if dump {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let result = match w.gpu_dtype {
        DType::F32 => gpu.gemm_f32_batched(x, &w.buf, y, batch, w.k, w.m),
        DType::F16 => gpu.gemm_f16_batched_lmhead(&w.buf, x, y, w.m, w.k, batch),
        DType::HFQ4G256 => gpu.gemm_hfq4g256_batched_lmhead(&w.buf, x, y, w.m, w.k, batch),
        DType::MQ4G256 => {
            // Chunk on `batch` when the request exceeds the scratch capacity
            // for this w.k. `mq_x_rot` is sized to MQ_X_ROT_CHUNK_ROWS × max(...)
            // — first-call rotations against the full prefix split into
            // `ceil(batch / max_chunk)` GEMMs.
            let scratch = mq_x_rot.expect("MQ4 dispatch requires mq_x_rot scratch");
            let max_chunk = (scratch.shape[0] / w.k).max(1);
            let mut chunked: HipResult<()> = Ok(());
            let mut row = 0;
            while row < batch {
                let n = std::cmp::min(max_chunk, batch - row);
                let x_chunk = x.sub_offset(row * w.k, n * w.k);
                let y_chunk = y.sub_offset(row * w.m, n * w.m);
                let rot_view = scratch.sub_offset(0, n * w.k);
                // AWQ-aware FWHT rotation. When the drafter weight ships an
                // AWQ sidecar (`w.awq_scale.is_some()`), `_for` dispatches
                // the `x /= awq_scale` + FWHT kernel; otherwise falls
                // through to the plain `rotate_x_mq_batched` and is
                // numerically identical to the prior dispatch.
                if let Err(e) =
                    crate::llama::rotate_x_mq_batched_for(gpu, w, &x_chunk, &rot_view, w.k, n)
                {
                    chunked = Err(e);
                    break;
                }
                if let Err(e) =
                    gpu.gemm_hfq4g256_batched_lmhead(&w.buf, &rot_view, &y_chunk, w.m, w.k, n)
                {
                    chunked = Err(e);
                    break;
                }
                row += n;
            }
            chunked
        }
        DType::MQ3G256 => {
            // Mirrors the MQ4 path: pre-rotate x via FWHT (same shared signs
            // as MQ4 — rotate_x_mq_batched is dtype-agnostic for the activation
            // side), invalidate the FP16 x cache because the rotated bytes
            // share the same source pointer, then dispatch the HFQ3 batched
            // lm_head WMMA kernel. Chunked symmetrically with MQ4.
            //
            // `fp16_x_source_ptr` is invalidated ONCE before the chunk loop —
            // not per-iteration. The MQ3 dispatch always overwrites the
            // shared rotation scratch from scratch each call (no chunk can
            // re-read FP16 from the previous chunk's output), so the
            // invalidation only needs to fire once per gemm_dispatch entry.
            // Previously the assignment was inside the loop, firing
            // `ceil(batch / max_chunk)` times for no extra correctness.
            let scratch = mq_x_rot.expect("MQ3 dispatch requires mq_x_rot scratch");
            let max_chunk = (scratch.shape[0] / w.k).max(1);
            gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
            let mut chunked: HipResult<()> = Ok(());
            let mut row = 0;
            while row < batch {
                let n = std::cmp::min(max_chunk, batch - row);
                let x_chunk = x.sub_offset(row * w.k, n * w.k);
                let y_chunk = y.sub_offset(row * w.m, n * w.m);
                let rot_view = scratch.sub_offset(0, n * w.k);
                // Same AWQ-aware FWHT rotation as the MQ4 arm. Drafters
                // that ship MQ3 AWQ sidecars (via the loader fix in
                // `hfq_weight`) now actually receive the `x /= s` divide
                // before the HFQ3 GEMM; pre-fix this silently produced
                // wrong drafts on AWQ-calibrated drafters.
                if let Err(e) =
                    crate::llama::rotate_x_mq_batched_for(gpu, w, &x_chunk, &rot_view, w.k, n)
                {
                    chunked = Err(e);
                    break;
                }
                if let Err(e) =
                    gpu.gemm_hfq3g256_batched_lmhead(&w.buf, &rot_view, &y_chunk, w.m, w.k, n)
                {
                    chunked = Err(e);
                    break;
                }
                row += n;
            }
            chunked
        }
        DType::MQ6G256 => {
            // Mirrors the MQ4/MQ3 path: pre-rotate x via FWHT, then dispatch
            // the HFQ6 batched lm_head WMMA kernel. Chunked symmetrically with
            // the other MQ formats.
            let scratch = mq_x_rot.expect("MQ6 dispatch requires mq_x_rot scratch");
            let max_chunk = (scratch.shape[0] / w.k).max(1);
            gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
            let mut chunked: HipResult<()> = Ok(());
            let mut row = 0;
            while row < batch {
                let n = std::cmp::min(max_chunk, batch - row);
                let x_chunk = x.sub_offset(row * w.k, n * w.k);
                let y_chunk = y.sub_offset(row * w.m, n * w.m);
                let rot_view = scratch.sub_offset(0, n * w.k);
                if let Err(e) =
                    crate::llama::rotate_x_mq_batched_for(gpu, w, &x_chunk, &rot_view, w.k, n)
                {
                    chunked = Err(e);
                    break;
                }
                if let Err(e) =
                    gpu.gemm_hfq6g256_batched_lmhead(&w.buf, &rot_view, &y_chunk, w.m, w.k, n)
                {
                    chunked = Err(e);
                    break;
                }
                row += n;
            }
            chunked
        }
        other => panic!("dflash gemm_dispatch: unsupported weight dtype {:?}", other),
    };
    if let Some(t) = t0 {
        gpu.hip.device_synchronize()?;
        let us = t.elapsed().as_micros();
        let weight_bytes = match w.gpu_dtype {
            DType::F32 => w.m * w.k * 4,
            DType::F16 => w.m * w.k * 2,
            DType::MQ3G256 => w.m * (w.k / 256).max(1) * 104,
            DType::HFQ4G256 | DType::MQ4G256 => w.m * (w.k / 256).max(1) * 136,
            DType::MQ6G256 => w.m * (w.k / 256).max(1) * 200,
            _ => w.m * w.k,
        };
        let bytes = weight_bytes + batch * w.k * 4 + batch * w.m * 4 * 2;
        let gbs = (bytes as f64) / (us.max(1) as f64) / 1000.0;
        eprintln!(
            "[draft-gemm] dtype={:?} M={} K={} B={} us={} bytes={}KB GB/s={:.1}",
            w.gpu_dtype,
            w.m,
            w.k,
            batch,
            us,
            bytes / 1024,
            gbs
        );
    }
    result
}

fn begin_draft_ffn_graph_capture(gpu: &mut Gpu) -> HipResult<()> {
    gpu.graphs.capture_blobs.clear();
    gpu.graphs.capture_mode = true;
    let stream = gpu
        .active_stream
        .as_ref()
        .expect("draft FFN graph capture requires an explicit stream");
    gpu.hip.stream_begin_capture(stream, 0)
}

fn end_draft_ffn_graph_capture(gpu: &mut Gpu) -> HipResult<(Graph, GraphExec, Vec<Vec<u8>>)> {
    gpu.graphs.capture_mode = false;
    let stream = gpu.active_stream.as_ref().unwrap();
    let graph = gpu.hip.stream_end_capture(stream)?;
    let exec = gpu.hip.graph_instantiate(&graph)?;
    let blobs = std::mem::take(&mut gpu.graphs.capture_blobs);
    Ok((graph, exec, blobs))
}

fn abort_draft_ffn_graph_capture(gpu: &mut Gpu) {
    if gpu.graphs.capture_mode {
        if let Some(stream) = gpu.active_stream.as_ref() {
            let _ = gpu.hip.stream_end_capture(stream);
        }
        gpu.graphs.capture_mode = false;
    }
    gpu.graphs.capture_blobs.clear();
}

fn draft_ffn_layer(
    gpu: &mut Gpu,
    layer: &DflashLayerWeights,
    scratch: &mut DflashScratch,
    b: usize,
    h: usize,
    eps: f32,
    graph_safe: bool,
) -> HipResult<()> {
    if graph_safe {
        gpu.memcpy_dtod_auto(&scratch.residual_ffn.buf, &scratch.x.buf, (b * h) * 4)?;
    } else {
        gpu.hip
            .memcpy_dtod(&scratch.residual_ffn.buf, &scratch.x.buf, (b * h) * 4)?;
    }

    gpu.rmsnorm_batched(&scratch.x, &layer.ffn_norm, &scratch.x_norm, b, h, eps)?;
    gemm_dispatch(
        gpu,
        &scratch.x_norm,
        &layer.w_gate,
        &scratch.gate,
        b,
        scratch.mq_x_rot.as_ref(),
    )?;
    gemm_dispatch(
        gpu,
        &scratch.x_norm,
        &layer.w_up,
        &scratch.up,
        b,
        scratch.mq_x_rot.as_ref(),
    )?;
    gpu.silu_mul_f32(&scratch.gate, &scratch.up, &scratch.gate_up)?;
    gemm_dispatch(
        gpu,
        &scratch.gate_up,
        &layer.w_down,
        &scratch.x,
        b,
        scratch.mq_x_rot.as_ref(),
    )?;
    if graph_safe {
        gpu.add_f32_graph_safe(&scratch.residual_ffn, &scratch.x, &scratch.x)
    } else {
        gpu.add_f32(&scratch.residual_ffn, &scratch.x, &scratch.x)
    }
}

fn draft_ffn_layer_maybe_graph(
    gpu: &mut Gpu,
    layer: &DflashLayerWeights,
    scratch: &mut DflashScratch,
    layer_idx: usize,
    b: usize,
    h: usize,
    eps: f32,
    use_graph: bool,
) -> HipResult<()> {
    if !use_graph {
        return draft_ffn_layer(gpu, layer, scratch, b, h, eps, false);
    }

    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    if scratch.draft_ffn_graphs[layer_idx].contains_key(&b) {
        let stream = gpu.active_stream.as_ref().unwrap();
        let entry = scratch.draft_ffn_graphs[layer_idx]
            .get(&b)
            .unwrap_or_else(|| panic!("missing draft FFN graph for layer={layer_idx} b={b}"));
        return gpu.hip.graph_launch(&entry.1, stream);
    }

    if !scratch.draft_ffn_warmed_up[layer_idx].contains(&b) {
        scratch.draft_ffn_warmed_up[layer_idx].insert(b);
        return draft_ffn_layer(gpu, layer, scratch, b, h, eps, false);
    }

    begin_draft_ffn_graph_capture(gpu)?;
    let r = draft_ffn_layer(gpu, layer, scratch, b, h, eps, true);
    if r.is_ok() {
        let entry = end_draft_ffn_graph_capture(gpu)?;
        scratch.draft_ffn_graphs[layer_idx].insert(b, entry);
        let stream = gpu.active_stream.as_ref().unwrap();
        let entry = scratch.draft_ffn_graphs[layer_idx]
            .get(&b)
            .unwrap_or_else(|| {
                panic!("missing captured draft FFN graph for layer={layer_idx} b={b}")
            });
        gpu.hip.graph_launch(&entry.1, stream)
    } else {
        abort_draft_ffn_graph_capture(gpu);
        r
    }
}

/// Upload f32 slice into a GPU tensor (bytes via memcpy_htod).
fn upload_slice_f32(gpu: &Gpu, dst: &GpuTensor, data: &[f32]) -> HipResult<()> {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    gpu.hip.memcpy_htod(&dst.buf, bytes)
}

/// Upload i32 slice into a GPU tensor (interpreted as i32 by kernels).
fn upload_slice_i32(gpu: &Gpu, dst: &GpuTensor, data: &[i32]) -> HipResult<()> {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    gpu.hip.memcpy_htod(&dst.buf, bytes)
}

/// Rows per chunk for the windowed-mode post-seed backfill.
pub const SEED_BACKFILL_CHUNK: usize = 512;

/// Windowed-mode post-seed backfill: after a cold prefill longer than the
/// SWA window, the last (full-attention) layer still needs K/V for EVERY
/// context row — but `hidden_rb` / the draft ring only retain the last
/// `swa_w`. The host shadow (`target_hidden_host`, cumulative on the cold
/// path) has the full prompt, so stream it through the ring in chunks:
/// upload → fc → hidden_norm → last-layer wk/wv + k_norm into its
/// `full_w`-row ring. SWA layers are deliberately skipped — the first
/// `draft_forward` lazy fill covers their in-window rows bit-identically.
///
/// No-op in Legacy mode or when `prompt_len <= swa_w`. Sets the thlog
/// full-layer watermark to `prompt_len` so the forward's lazy fill resumes
/// from the window edge. All GEMMs/norms are row-local: chunked ==
/// monolithic bit-exactly.
pub fn draft_seed_backfill(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    scratch: &mut DflashScratch,
    host_hidden: &[f32],
    prompt_len: usize,
) -> HipResult<()> {
    let (swa_w, full_w) = match scratch.ctx_mode {
        DraftCtxMode::Legacy => return Ok(()),
        DraftCtxMode::Windowed { w, w_full } => (w, w_full),
    };
    if prompt_len <= swa_w {
        return Ok(());
    }
    let h = cfg.hidden;
    let ne = cfg.num_extract();
    let kvd = cfg.kv_dim();
    let hd = cfg.head_dim;
    let eps = cfg.norm_eps;
    let row_f32 = ne * h;
    assert_eq!(
        host_hidden.len(),
        prompt_len * row_f32,
        "backfill: host shadow must hold the full prompt"
    );
    let last_layer = weights.layers.last().expect("draft has no layers");
    let k_full = scratch.k_full_cached.as_ref().expect("windowed k_full");
    let v_full = scratch.v_full_cached.as_ref().expect("windowed v_full");

    let mut row = 0usize;
    while row < prompt_len {
        let len = SEED_BACKFILL_CHUNK.min(prompt_len - row);
        for (row0, slot0, seg_len) in ring_segments(row, row + len, swa_w) {
            // H2D this host segment into the draft ring.
            let seg = &host_hidden[row0 * row_f32..(row0 + seg_len) * row_f32];
            let src_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(seg.as_ptr() as *const u8, seg.len() * 4)
            };
            gpu.hip
                .memcpy_htod_offset(&scratch.target_hidden.buf, slot0 * row_f32 * 4, src_bytes)?;
            // fc + hidden_norm into the proj ring (same slots).
            let th = scratch
                .target_hidden
                .sub_offset(slot0 * row_f32, seg_len * row_f32);
            let thp = scratch
                .target_hidden_proj
                .sub_offset(slot0 * h, seg_len * h);
            gemm_dispatch(gpu, &th, &weights.fc, &thp, seg_len, scratch.mq_x_rot.as_ref())?;
            gpu.rmsnorm_batched(&thp, &weights.hidden_norm, &thp, seg_len, h, eps)?;
            // Last-layer wk/wv into the full_w ring (its own modulus).
            let mut r2 = row0;
            while r2 < row0 + seg_len {
                let step = (full_w - r2 % full_w).min(row0 + seg_len - r2);
                let thp2 = scratch
                    .target_hidden_proj
                    .sub_offset((slot0 + (r2 - row0)) * h, step * h);
                let c_slot = r2 % full_w;
                let k_slot = k_full.sub_offset(c_slot * kvd, step * kvd);
                let v_slot = v_full.sub_offset(c_slot * kvd, step * kvd);
                gemm_dispatch(gpu, &thp2, &last_layer.wk, &k_slot, step, scratch.mq_x_rot.as_ref())?;
                gemm_dispatch(gpu, &thp2, &last_layer.wv, &v_slot, step, scratch.mq_x_rot.as_ref())?;
                gpu.rmsnorm_batched(
                    &k_slot,
                    &last_layer.k_norm,
                    &k_slot,
                    step * cfg.n_kv_heads,
                    hd,
                    eps,
                )?;
                r2 += step;
            }
        }
        row += len;
    }
    scratch.thlog.mark_full_cached(prompt_len);
    Ok(())
}

/// Run one draft forward. Inputs:
/// - `noise_embedding`: `[block_size × hidden]` f32, row-major. Comes from
///   `target.embed_tokens(block_output_ids)` on the caller side.
/// - `target_hidden`:   `[ctx_len × num_extract × hidden]` f32, row-major
///   (5-way concat of target's chosen-layer hidden states at `ctx_len`
///   accepted positions).
/// - `positions_q`:     `[block_size]` i32 — absolute position index of
///   each block position in the full sequence (used for RoPE on Q).
/// - `positions_k`:     `[ctx_len + block_size]` i32 — absolute position
///   index for every ctx position followed by every block position
///   (used for RoPE on K = concat(ctx, noise)).
///
/// Output: writes final hidden states `[block_size × hidden]` into
/// `scratch.x`. Caller applies target's `lm_head` over the last
/// `block_size - 1` rows to produce logits for the mask slots.
///
/// Precondition: `block_size ≤ scratch.max_block_size`,
/// `ctx_len ≤ scratch.max_ctx_len`.
/// Run one draft forward over `block_size` positions with `ctx_len` cached
/// context rows.
///
/// `noise_embedding`: if `Some`, uploaded into `scratch.x` before the forward.
///     If `None`, the caller must have already filled `scratch.x` with B × hidden
///     F32 embeddings — this avoids the target→host→draft round-trip in the
///     spec-decode hot loop (both target and draft share the same GPU, so
///     D2D copies into `scratch.x` suffice).
/// `target_hidden`: if `Some`, uploaded into `scratch.target_hidden`.
///     If `None`, the caller must have already filled `scratch.target_hidden`
///     with `ctx_len × num_extract × hidden` F32 rows.
#[allow(clippy::too_many_arguments)]
pub fn draft_forward(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    noise_embedding: Option<&[f32]>,
    target_hidden: Option<&[f32]>,
    positions_q: &[i32],
    positions_k: &[i32],
    block_size: usize,
    ctx_len: usize,
    scratch: &mut DflashScratch,
) -> HipResult<()> {
    draft_forward_opts(
        gpu,
        weights,
        cfg,
        noise_embedding,
        target_hidden,
        positions_q,
        positions_k,
        block_size,
        ctx_len,
        scratch,
        false,
    )
}

pub fn draft_forward_opts(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    noise_embedding: Option<&[f32]>,
    target_hidden: Option<&[f32]>,
    positions_q: &[i32],
    positions_k: &[i32],
    block_size: usize,
    ctx_len: usize,
    scratch: &mut DflashScratch,
    graph_ffn: bool,
) -> HipResult<()> {
    let b = block_size;
    let l = ctx_len;
    let tot = l + b;
    let h = cfg.hidden;
    let ne = cfg.num_extract();
    let qd = cfg.q_dim();
    let kvd = cfg.kv_dim();
    let hd = cfg.head_dim;
    let eps = cfg.norm_eps;
    let theta = cfg.rope_theta;

    assert!(b <= scratch.max_block_size, "block_size > scratch max");
    assert!(l <= scratch.max_ctx_len, "ctx_len > scratch max");
    if let Some(ne_slice) = noise_embedding {
        assert_eq!(ne_slice.len(), b * h, "noise_embedding size");
    }
    if let Some(th_slice) = target_hidden {
        assert_eq!(th_slice.len(), l * ne * h, "target_hidden size");
    }
    assert_eq!(positions_q.len(), b, "positions_q size");
    assert_eq!(positions_k.len(), tot, "positions_k size");

    // Draft context windowing (DraftCtxMode::Windowed). `swa_w`/`full_w` are
    // the per-layer ring moduli (usize::MAX in Legacy ⇒ identity ring math,
    // single segment, byte-identical offsets). The last (full-attention)
    // layer uses `full_w`; all others `swa_w`. At l <= swa_w every span is
    // [0..l) and every slot == row: the windowed code path degenerates to
    // Legacy exactly.
    let (swa_w, full_w) = match scratch.ctx_mode {
        DraftCtxMode::Legacy => (usize::MAX, usize::MAX),
        DraftCtxMode::Windowed { w, w_full } => (w, w_full),
    };
    let windowed = !matches!(scratch.ctx_mode, DraftCtxMode::Legacy);
    // Windowed positions base: the device positions_k buffer holds only the
    // suffix [pos_base..tot) (last full_w context rows + B noise) — every
    // layer's attention span lives inside it. Device index of row r is
    // r − pos_base. Legacy: pos_base = 0, the full upload as before.
    let pos_base = if windowed {
        l.saturating_sub(full_w)
    } else {
        0
    };
    // Backfill-invariant probe: past the SWA window the last layer's
    // out-of-window K/V must come from the post-seed backfill. If a driver
    // skipped it (research demos on non-speculator paths), its ring holds
    // stale rows — τ-only damage (verify is exact), but name it once.
    if windowed && l > swa_w && scratch.thlog.full_cached_rows() < l.saturating_sub(swa_w) {
        static BACKFILL_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !BACKFILL_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "[dflash] windowed: last-layer backfill watermark {} < l−W {} — stale \
                 out-of-window K/V (acceptance degraded; output still verify-exact). \
                 Call draft_seed_backfill after cold seeds.",
                scratch.thlog.full_cached_rows(),
                l.saturating_sub(swa_w)
            );
        }
    }

    // ── 0. Uploads ────────────────────────────────────────────────────────
    if let Some(ne_slice) = noise_embedding {
        upload_slice_f32(gpu, &scratch.x, ne_slice)?;
    }
    if let Some(th_slice) = target_hidden {
        // Incremental-upload fast path: the caller passes a rolling prefix
        // (rows 0..l) of target_hidden. In DFlash's common steady-state
        // (ctx_slice == None), the prefix grows by accept+1 rows per cycle
        // and rows [0..prev_l) are unchanged since the previous call.
        // Upload only the new tail when detected. This cuts the H2D from
        // `l × ne × hidden × 4` (e.g. ~90 MB at l=1100 on 9B) to
        // `(l - uploaded) × ne × hidden × 4` (~700 KB at accept=8).
        //
        // The optimization only fires when `th_slice.len() == l × ne × h`
        // and it matches what the caller told us (matches a non-sliced
        // cumulative buffer). ctx_slice callers pass a DIFFERENT slice
        // every cycle (last N rows shift) — for them, force full upload.
        //
        // Windowed mode: the GPU buffer is an `swa_w`-row ring (slot = row %
        // swa_w); only rows [max(prev, l−swa_w)..l) are uploaded, wrap-split
        // at the ring boundary. Out-of-window rows are never read (fc is the
        // only consumer and its fill is span-limited too).
        let row_f32 = ne * h;
        let expected_full_len = l * row_f32;
        let prev = scratch.thlog.uploaded_rows();
        // Full-upload conditions: first call, reset flagged, caller shrank
        // the context, or the slice length suggests ctx_slice (unusual l).
        if prev == 0 || prev > l || th_slice.len() != expected_full_len {
            if windowed {
                let up_start = l.saturating_sub(swa_w);
                for (row0, slot0, len) in ring_segments(up_start, l, swa_w) {
                    let seg = &th_slice[row0 * row_f32..(row0 + len) * row_f32];
                    let dst_byte_off = slot0 * row_f32 * 4;
                    let src_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(seg.as_ptr() as *const u8, seg.len() * 4)
                    };
                    gpu.hip
                        .memcpy_htod_offset(&scratch.target_hidden.buf, dst_byte_off, src_bytes)?;
                }
            } else {
                upload_slice_f32(gpu, &scratch.target_hidden, th_slice)?;
            }
            scratch.thlog.mark_uploaded(l);
        } else if prev < l {
            // Delta-upload: rows [prev..l) need to land at byte offset
            // prev * row_f32 * 4 of scratch.target_hidden.
            if windowed {
                let up_start = prev.max(l.saturating_sub(swa_w));
                for (row0, slot0, len) in ring_segments(up_start, l, swa_w) {
                    let seg = &th_slice[row0 * row_f32..(row0 + len) * row_f32];
                    let dst_byte_off = slot0 * row_f32 * 4;
                    let src_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(seg.as_ptr() as *const u8, seg.len() * 4)
                    };
                    gpu.hip
                        .memcpy_htod_offset(&scratch.target_hidden.buf, dst_byte_off, src_bytes)?;
                }
            } else {
                let tail = &th_slice[prev * row_f32..];
                let dst_byte_off = prev * row_f32 * 4;
                let src_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(tail.as_ptr() as *const u8, tail.len() * 4)
                };
                gpu.hip
                    .memcpy_htod_offset(&scratch.target_hidden.buf, dst_byte_off, src_bytes)?;
            }
            scratch.thlog.mark_uploaded(l);
        }
        // prev == l: nothing new to upload (wouldn't happen in practice
        // since caller always appends, but harmless).
    }
    upload_slice_i32(gpu, &scratch.positions_q, positions_q)?;
    if windowed {
        upload_slice_i32(gpu, &scratch.positions_k, &positions_k[pos_base..])?;
    } else {
        upload_slice_i32(gpu, &scratch.positions_k, positions_k)?;
    }

    // ── 1. target_hidden_proj = hidden_norm(fc @ target_hidden) ──────────
    // Incremental-projection fast path (2026-04-20): only compute the
    // delta rows [cached..L) that haven't been projected yet. Rows
    // [0..cached) were projected and cached by a previous draft_forward
    // call and are still valid. After this block, rows [0..L) of
    // target_hidden_proj are usable by the per-layer K/V step below.
    //
    // `draft_ctx_cached_rows` is the scratch-owned invariant: how many
    // prefix rows have been computed end-to-end (fc + hidden_norm +
    // per-layer wk + k_norm + per-layer wv). It resets to 0 at new
    // prompts (reset_upload_tracking) and on eviction
    // (invalidate_draft_ctx_cache).
    //
    // Full-rebuild cases: delta == L (first cycle after reset), or
    // delta > L somehow (shouldn't happen — this would be a bug). Those
    // go through the same code path with delta == L, same cost as before.
    //
    // Dispatch on fc weight dtype: F32 → gemm_f32_batched (legacy),
    // MQ4 → FWHT-rotate target_hidden then gemm_hfq4g256.
    //
    // Windowed: the proj ring only needs the rows any layer can still read —
    // clamp the fill to [max(cached, l−swa_w)..l) (out-of-window rows were
    // covered by the post-seed backfill for the last layer, and SWA layers
    // never read them). Ring-slot writes are wrap-split; every GEMM/norm is
    // row-local, so segments compose bit-exactly. Legacy: swa_w=MAX ⇒ one
    // identity segment, offsets identical to the pre-windowing code.
    let cached_rows = scratch.thlog.proj_cached_rows();
    let fc_start = cached_rows.max(l.saturating_sub(swa_w));
    let delta = l.saturating_sub(fc_start);
    if delta > 0 {
        for (row0, slot0, len) in ring_segments(fc_start, l, swa_w) {
            let th_slice = scratch
                .target_hidden
                .sub_offset(slot0 * ne * h, len * ne * h);
            let thp_slice = scratch.target_hidden_proj.sub_offset(slot0 * h, len * h);
            gemm_dispatch(
                gpu,
                &th_slice,
                &weights.fc,
                &thp_slice,
                len,
                scratch.mq_x_rot.as_ref(),
            )?;
            gpu.rmsnorm_batched(&thp_slice, &weights.hidden_norm, &thp_slice, len, h, eps)?;
        }
    }

    // HIPFIRE_DRAFT_SUBPHASE=1: per-layer-section timing inside draft_forward.
    // Diagnostic only — device_synchronize at each boundary makes this 2-3×
    // slower than a production run. Printed once per forward.
    //
    // First measurement (2026-04-21, 27B HumanEval, B=16, steady-state):
    //   attn_gemm: 7.4 ms  (wq + wk/v_noise + wk/v_ctx)
    //   concat:    0.4 ms  (K/V cache concat D2Ds)
    //   attn_krn:  0.6 ms  (attention_dflash_f32)
    //   ffn_gemm:  56 ms   (wo + w_gate + w_up + w_down + silu_mul + adds)
    //
    // 87 % of draft_forward lives in the FFN GEMM block. w_gate/w_up/w_down
    // at M=17408/K=5120 should be ~0.5 ms/layer BW-bound but is ~11 ms/layer
    // observed. Next lever: route draft's w_gate/w_up through the fused
    // gemm_gate_up_hfq4g256_wmma kernel (measured 73 µs/call on the same
    // shape in target; vs ksplit's 288 µs/call on a different shape). Or
    // find the real cause of the ksplit slowdown on draft shapes.
    let dbg = crate::config::get().draft_subphase;
    let mut us_attn_gemm: u128 = 0;
    let mut us_attn_kernel: u128 = 0;
    let mut us_ffn_gemm: u128 = 0;
    let mut us_concat: u128 = 0;
    if dbg {
        gpu.hip.device_synchronize()?;
    }

    // ── 2. Per-layer decoder ─────────────────────────────────────────────
    for li in 0..cfg.n_layers {
        let layer = &weights.layers[li];

        // Residual.
        gpu.hip
            .memcpy_dtod(&scratch.residual_attn.buf, &scratch.x.buf, (b * h) * 4)?;

        // attn_norm.
        gpu.rmsnorm_batched(&scratch.x, &layer.attn_norm, &scratch.x_norm, b, h, eps)?;

        let t0 = if dbg {
            gpu.hip.device_synchronize()?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Q/K/V projections — dispatched on each weight's dtype.
        // Q and K/V noise (over the B block positions) must be computed
        // every cycle. K_ctx and V_ctx (over the L context positions)
        // are *incrementally* cached — see the per-layer block below.
        gemm_dispatch(
            gpu,
            &scratch.x_norm,
            &layer.wq,
            &scratch.q,
            b,
            scratch.mq_x_rot.as_ref(),
        )?;
        gemm_dispatch(
            gpu,
            &scratch.x_norm,
            &layer.wk,
            &scratch.k_noise,
            b,
            scratch.mq_x_rot.as_ref(),
        )?;
        gemm_dispatch(
            gpu,
            &scratch.x_norm,
            &layer.wv,
            &scratch.v_noise,
            b,
            scratch.mq_x_rot.as_ref(),
        )?;

        // K_ctx / V_ctx — same wk/wv weights but projected over the L
        // accepted-context rows of target_hidden_proj. INCREMENTAL PATH:
        // only rows past the layer's fill watermark need projection; earlier
        // rows were projected in a prior call and stored in the per-layer
        // k_ctx_cached / v_ctx_cached buffers (post-k_norm for K).
        //
        // Windowed spans: SWA layers read the last `swa_w` context rows, the
        // last (full-attention) layer the last `full_w`. Each cache is a ring
        // (slot = row % layer_w); fills clamp to rows still readable
        // ([max(watermark, span_start)..l)) and source the proj ring, whose
        // modulus is `swa_w` — the last layer's (watermark, l−swa_w) band is
        // the post-seed backfill's responsibility. Segments split at BOTH
        // ring boundaries (proj + cache) so every GEMM sees contiguous src
        // and dst. Legacy: moduli are MAX — one identity segment, offsets
        // identical to the pre-windowing incremental path.
        //
        // For correctness note: the per-head K RMSNorm is row-local
        // (normalizes each kv_head row of size hd independently), so
        // applying it to delta rows of the cache is exactly equivalent
        // to applying it to the full k_cat post-concat. V has no
        // draft-level norm so v_ctx_cached stores raw GEMM output.
        let last_layer = li + 1 == cfg.n_layers;
        let layer_w = if last_layer { full_w } else { swa_w };
        let (k_cache_layer, v_cache_layer) = if last_layer && windowed {
            (
                scratch.k_full_cached.as_ref().expect("windowed k_full"),
                scratch.v_full_cached.as_ref().expect("windowed v_full"),
            )
        } else {
            (&scratch.k_ctx_cached[li], &scratch.v_ctx_cached[li])
        };
        let (k_cat_l, v_cat_l) = if last_layer && windowed {
            (
                scratch.k_cat_full.as_ref().expect("windowed k_cat_full"),
                scratch.v_cat_full.as_ref().expect("windowed v_cat_full"),
            )
        } else {
            (&scratch.k_cat, &scratch.v_cat)
        };
        let span_start = l.saturating_sub(layer_w);
        let span = l - span_start;
        let wm = if last_layer {
            scratch.thlog.full_cached_rows()
        } else {
            cached_rows
        };
        // Rows older than the proj ring's live window cannot be (re)filled
        // here — the post-seed backfill owns that band for the last layer.
        let fill_start = wm.max(span_start).max(l.saturating_sub(swa_w));
        if fill_start < l {
            let mut row = fill_start;
            while row < l {
                // Next row where the proj ring or the cache ring wraps.
                let step = (swa_w.saturating_sub(if swa_w == usize::MAX {
                    0
                } else {
                    row % swa_w
                }))
                .min(layer_w.saturating_sub(if layer_w == usize::MAX {
                    0
                } else {
                    row % layer_w
                }))
                .min(l - row);
                let p_slot = if swa_w == usize::MAX {
                    row
                } else {
                    row % swa_w
                };
                let c_slot = if layer_w == usize::MAX {
                    row
                } else {
                    row % layer_w
                };
                let thp_slice = scratch.target_hidden_proj.sub_offset(p_slot * h, step * h);
                let k_slot = k_cache_layer.sub_offset(c_slot * kvd, step * kvd);
                let v_slot = v_cache_layer.sub_offset(c_slot * kvd, step * kvd);
                gemm_dispatch(
                    gpu,
                    &thp_slice,
                    &layer.wk,
                    &k_slot,
                    step,
                    scratch.mq_x_rot.as_ref(),
                )?;
                gemm_dispatch(
                    gpu,
                    &thp_slice,
                    &layer.wv,
                    &v_slot,
                    step,
                    scratch.mq_x_rot.as_ref(),
                )?;
                // Per-head RMSNorm on K delta rows only. batch = step × n_kv_heads.
                gpu.rmsnorm_batched(&k_slot, &layer.k_norm, &k_slot, step * cfg.n_kv_heads, hd, eps)?;
                row += step;
            }
        }

        if let Some(t) = t0 {
            gpu.hip.device_synchronize()?;
            us_attn_gemm += t.elapsed().as_micros();
        }
        let t1 = if dbg {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Concat K = [K_ctx ring span | K_noise] → k_cat [span + B, kv_dim].
        // The cached K prefix is already post-k_norm (applied incrementally
        // above); the noise tail still needs k_norm applied below. Ring
        // segments assemble in absolute-row order; Legacy = one segment.
        let noise_bytes = (b * kvd) * 4;
        let mut cat_off = 0usize;
        for (_row0, slot0, len) in ring_segments(span_start, l, layer_w) {
            let seg_bytes = len * kvd * 4;
            gpu.hip
                .memcpy_dtod_at(&k_cat_l.buf, cat_off, &k_cache_layer.buf, slot0 * kvd * 4, seg_bytes)?;
            gpu.hip
                .memcpy_dtod_at(&v_cat_l.buf, cat_off, &v_cache_layer.buf, slot0 * kvd * 4, seg_bytes)?;
            cat_off += seg_bytes;
        }
        gpu.hip
            .memcpy_dtod_at(&k_cat_l.buf, cat_off, &scratch.k_noise.buf, 0, noise_bytes)?;
        gpu.hip
            .memcpy_dtod_at(&v_cat_l.buf, cat_off, &scratch.v_noise.buf, 0, noise_bytes)?;

        // Per-head RMSNorm on Q: each of B*n_heads rows, size head_dim,
        // weight [head_dim].
        gpu.rmsnorm_batched(
            &scratch.q,
            &layer.q_norm,
            &scratch.q,
            b * cfg.n_heads,
            hd,
            eps,
        )?;
        // Per-head RMSNorm on the NOISE tail of K_cat only — the cached
        // prefix was already normed when it was inserted into the layer's
        // k_ctx_cached. batch = B × n_kv_heads, applied to the last B rows
        // of k_cat.
        {
            let noise_slot = k_cat_l.sub_offset(span * kvd, b * kvd);
            gpu.rmsnorm_batched(
                &noise_slot,
                &layer.k_norm,
                &noise_slot,
                b * cfg.n_kv_heads,
                hd,
                eps,
            )?;
        }

        // RoPE. rope_batched_f32 expects q and k at the SAME batch size,
        // rotating at per-row positions. We call it twice with a zero
        // "head count" on the inactive tensor so its loop doesn't execute.
        // Call 1: rotate Q with positions_q. Pass k as a valid buffer
        // (scratch.k_noise is shape-compatible; n_heads_k=0 skips its loop).
        gpu.rope_batched_f32(
            &scratch.q,
            &scratch.k_noise,     // ignored because n_heads_k = 0
            &scratch.positions_q, // [B]
            cfg.n_heads,
            0,
            hd,
            theta,
            b,
        )?;
        // Call 2: rotate K_cat with positions_k. n_heads_q = 0 skips Q.
        // Windowed: the layer's context rows are the suffix [span_start..l),
        // so its positions slice is the device buffer's contiguous
        // sub-range [span_start−pos_base..tot−pos_base) — zero extra upload,
        // absolute RoPE phases preserved. Legacy: both bases 0 ⇒ full buffer.
        let positions_k_span = scratch
            .positions_k
            .sub_offset(span_start - pos_base, span + b);
        gpu.rope_batched_f32(
            &scratch.q, // ignored because n_heads_q = 0
            k_cat_l,
            &positions_k_span,
            0,
            cfg.n_kv_heads,
            hd,
            theta,
            span + b,
        )?;

        if let Some(t) = t1 {
            gpu.hip.device_synchronize()?;
            us_concat += t.elapsed().as_micros();
        }
        let t2 = if dbg {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Attention: Q [B, n_heads, hd] × K [span+B, n_kv_heads, hd]^T → scores
        // (with GQA expansion) → softmax → @V.
        // Dispatched via unified full-attention path (AttnFullF32 → DflashScalar).
        // NOTE: adding a WMMA rung here changes spec-decode numerics → draft logits
        // → acceptance patterns. Any future WMMA draft rung requires a dedicated
        // feedback_attention_precision sweep with acceptance-rate tracking.
        {
            use crate::llama::{attention_family, DispatchCtx, FullAttnParams, KernelKey};
            let ctx = DispatchCtx::new(gpu);
            let family = attention_family();
            family
                .run_full_attention(
                    &ctx,
                    gpu,
                    &FullAttnParams {
                        key: KernelKey::AttnFullF32,
                        q: &scratch.q,
                        k: k_cat_l,
                        v: v_cat_l,
                        out: &scratch.attn_out,
                        n: b,
                        seq_len: span + b,
                        n_heads: cfg.n_heads,
                        n_kv_heads: cfg.n_kv_heads,
                        head_dim: hd,
                    },
                )
                .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
        };
        if let Some(t) = t2 {
            gpu.hip.device_synchronize()?;
            us_attn_kernel += t.elapsed().as_micros();
        }
        let t3 = if dbg {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // attn_proj = attn_out @ wo^T → [B, hidden]
        gemm_dispatch(
            gpu,
            &scratch.attn_out,
            &layer.wo,
            &scratch.attn_proj,
            b,
            scratch.mq_x_rot.as_ref(),
        )?;

        // x = residual_attn + attn_proj
        gpu.add_f32(&scratch.residual_attn, &scratch.attn_proj, &scratch.x)?;

        // Fixed-shape FFN tail. MoE DFlash can optionally capture this as a
        // per-layer/per-B hipGraph; the attention/context work above is left
        // direct because `ctx_len` changes every accepted cycle.
        let graph_ffn_active = graph_ffn && !dbg && !crate::config::get().draft_gemm_dump;
        draft_ffn_layer_maybe_graph(gpu, layer, scratch, li, b, h, eps, graph_ffn_active)?;
        // 2026-04-21: tried target's fused gemm_gate_up_hfq4g256 here (shared
        // FP16-X convert + interleaved gate/up GEMMs). Byte-exact A/B neutral
        // on 27B HumanEval (median 76.47 fused vs 76.74 baseline; ±7 % run-to-
        // run variance from ksplit's non-deterministic atomicAdd dominates).
        // Kept per-weight dispatch for clarity. The real draft perf lever is
        // the ~56 ms of ffn_gemm per cycle (see HIPFIRE_DRAFT_SUBPHASE=1);
        // fusion alone doesn't move that number — kernel engineering does.

        if let Some(t) = t3 {
            gpu.hip.device_synchronize()?;
            us_ffn_gemm += t.elapsed().as_micros();
        }
    }

    if dbg {
        gpu.hip.device_synchronize()?;
        eprintln!(
            "[draft-sub] attn_gemm={}µs concat={}µs attn_kernel={}µs ffn_gemm={}µs (B={} L={})",
            us_attn_gemm, us_concat, us_attn_kernel, us_ffn_gemm, b, l,
        );
    }

    // ── 3. Final norm ────────────────────────────────────────────────────
    gpu.rmsnorm_batched(&scratch.x, &weights.norm, &scratch.x, b, h, eps)?;

    // ── 4. Advance the draft-ctx projection cache pointer ────────────────
    // All rows [0..l) of target_hidden_proj and every layer's
    // k_ctx_cached / v_ctx_cached now contain finalized per-layer
    // projections. Next call's delta starts from here.
    scratch.thlog.mark_proj_cached(l);

    Ok(())
}

#[cfg(test)]
mod ring_tests {
    use super::ring_segments;

    #[test]
    fn identity_modulus_is_single_segment() {
        assert_eq!(ring_segments(3, 10, usize::MAX), vec![(3, 3, 7)]);
        assert!(ring_segments(5, 5, usize::MAX).is_empty());
    }

    #[test]
    fn no_wrap_within_ring() {
        // rows < modulus: slot == row, single segment
        assert_eq!(ring_segments(2, 7, 16), vec![(2, 2, 5)]);
    }

    #[test]
    fn wrap_splits_at_boundary() {
        // rows 14..18 in a 16-ring: [14..16) at slots 14.., [16..18) at slots 0..
        assert_eq!(ring_segments(14, 18, 16), vec![(14, 14, 2), (16, 0, 2)]);
    }

    #[test]
    fn span_never_exceeds_ring() {
        // windowed fill spans are <= modulus, so at most two segments
        for start in [0usize, 1, 7, 15, 16, 17, 31, 32] {
            for len in [1usize, 3, 8, 16] {
                let segs = ring_segments(start, start + len, 16);
                assert!(segs.len() <= 2, "{start}+{len}: {segs:?}");
                let covered: usize = segs.iter().map(|s| s.2).sum();
                assert_eq!(covered, len);
                // slots contiguous per segment, rows contiguous across
                let mut row = start;
                for (r0, s0, l) in &segs {
                    assert_eq!(*r0, row);
                    assert_eq!(*s0, row % 16);
                    row += l;
                }
            }
        }
    }
}
