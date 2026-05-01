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

use crate::hfq::HfqFile;
use crate::llama::WeightTensor;
use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

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
        let head_dim = df.get("head_dim").and_then(|v| v.as_u64()).unwrap_or(
            (hidden / n_heads) as u64,
        ) as usize;
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
        let num_target_layers = df
            .get("num_target_layers")
            .and_then(|v| v.as_u64())? as usize;

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
    pub attn_norm: GpuTensor,        // [hidden] — F32, RMSNorm weight
    pub wq: WeightTensor,            // [q_dim, hidden]
    pub wk: WeightTensor,            // [kv_dim, hidden]
    pub wv: WeightTensor,            // [kv_dim, hidden]
    pub wo: WeightTensor,            // [hidden, q_dim]
    pub q_norm: GpuTensor,           // [head_dim] — F32
    pub k_norm: GpuTensor,           // [head_dim] — F32
    pub ffn_norm: GpuTensor,         // [hidden] — F32
    pub w_gate: WeightTensor,        // [intermediate, hidden]
    pub w_up: WeightTensor,          // [intermediate, hidden]
    pub w_down: WeightTensor,        // [hidden, intermediate]
}

pub struct DflashWeights {
    /// `fc`: Linear(num_extract × hidden → hidden). Shape: [hidden, num_extract × hidden].
    pub fc: WeightTensor,
    pub hidden_norm: GpuTensor,    // [hidden] — F32
    pub norm: GpuTensor,           // [hidden] — F32, final output norm
    pub layers: Vec<DflashLayerWeights>,
    /// True when at least one matrix weight is MQ4G256 — drives whether
    /// the draft_forward path needs to allocate FWHT rotation scratches.
    pub has_mq: bool,
}

/// Load a F32-only tensor (norms, embedding-shaped scalars). Always F32 on GPU.
fn hfq_tensor_f32(hfq: &HfqFile, gpu: &mut Gpu, name: &str, shape: Vec<usize>) -> HipResult<GpuTensor> {
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
///
/// `shape = [m, k]` so m=output_dim and k=input_dim. The HFQ index stores
/// the unaligned byte length; for MQ4 we skip shape verification (the
/// quantized bytes are not a function of m*k alone — group padding can add
/// up to 255 trailing bytes per row group).
fn hfq_weight(hfq: &HfqFile, gpu: &mut Gpu, name: &str, m: usize, k: usize) -> HipResult<WeightTensor> {
    let (info, data) = hfq
        .tensor_data(name)
        .unwrap_or_else(|| panic!("dflash tensor missing: {name}"));
    match info.quant_type {
        1 => {
            // F16 on disk. Default: upload as F16 (no lift) and dispatch through
            // the mw16 WMMA kernel — 3-5× faster draft at B=16 on gfx1100 than
            // the F32 lift path (which bypassed WMMA entirely via the naive
            // gemm_f32_batched kernel at ~100 GB/s / 10 % peak).
            //
            // HIPFIRE_DRAFT_F16=0 falls back to the legacy F16→F32 lift for
            // A/B comparison.
            let use_f16 = std::env::var("HIPFIRE_DRAFT_F16").ok().as_deref() != Some("0");
            if use_f16 {
                assert_eq!(data.len(), m * k * 2, "dflash {name} F16 byte-size mismatch");
                let buf = gpu.upload_raw(data, &[m * k])?;
                Ok(WeightTensor { buf, gpu_dtype: DType::F16, m, k, row_stride: 0 })
            } else {
                let f32_data: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|c| crate::llama::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                assert_eq!(f32_data.len(), m * k, "dflash {name} F16 size mismatch");
                let buf = gpu.upload_f32(&f32_data, &[m * k])?;
                Ok(WeightTensor { buf, gpu_dtype: DType::F32, m, k, row_stride: 0 })
            }
        }
        2 => {
            let f32_data: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(f32_data.len(), m * k, "dflash {name} F32 size mismatch");
            let buf = gpu.upload_f32(&f32_data, &[m * k])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::F32, m, k, row_stride: 0 })
        }
        13 => {
            // MQ4-G256: 136 bytes per 256 weights. The buffer is opaque to
            // the engine; the gemm_hfq4g256 kernel reads it directly.
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::MQ4G256, m, k, row_stride: 0 })
        }
        17 => {
            // MQ3-G256: 104 bytes per 256 weights. Same opaque-buffer pattern
            // as MQ4. Dispatch path (`gemm_dispatch`) routes through
            // `rotate_x_mq_batched` + `gemm_hfq3g256_batched_lmhead`.
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::MQ3G256, m, k, row_stride: 0 })
        }
        q => panic!("dflash: unsupported matrix quant_type {q} for {name}"),
    }
}

impl DflashWeights {
    pub fn load(gpu: &mut Gpu, hfq: &HfqFile, cfg: &DflashConfig) -> HipResult<Self> {
        let fc = hfq_weight(hfq, gpu, "fc.weight", cfg.hidden, cfg.num_extract() * cfg.hidden)?;
        let hidden_norm = hfq_tensor_f32(hfq, gpu, "hidden_norm.weight", vec![cfg.hidden])?;
        let norm = hfq_tensor_f32(hfq, gpu, "norm.weight", vec![cfg.hidden])?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("layers.{i}");
            let layer = DflashLayerWeights {
                attn_norm: hfq_tensor_f32(hfq, gpu, &format!("{p}.input_layernorm.weight"), vec![cfg.hidden])?,
                wq: hfq_weight(hfq, gpu, &format!("{p}.self_attn.q_proj.weight"), cfg.q_dim(), cfg.hidden)?,
                wk: hfq_weight(hfq, gpu, &format!("{p}.self_attn.k_proj.weight"), cfg.kv_dim(), cfg.hidden)?,
                wv: hfq_weight(hfq, gpu, &format!("{p}.self_attn.v_proj.weight"), cfg.kv_dim(), cfg.hidden)?,
                wo: hfq_weight(hfq, gpu, &format!("{p}.self_attn.o_proj.weight"), cfg.hidden, cfg.q_dim())?,
                q_norm: hfq_tensor_f32(hfq, gpu, &format!("{p}.self_attn.q_norm.weight"), vec![cfg.head_dim])?,
                k_norm: hfq_tensor_f32(hfq, gpu, &format!("{p}.self_attn.k_norm.weight"), vec![cfg.head_dim])?,
                ffn_norm: hfq_tensor_f32(hfq, gpu, &format!("{p}.post_attention_layernorm.weight"), vec![cfg.hidden])?,
                w_gate: hfq_weight(hfq, gpu, &format!("{p}.mlp.gate_proj.weight"), cfg.intermediate, cfg.hidden)?,
                w_up: hfq_weight(hfq, gpu, &format!("{p}.mlp.up_proj.weight"), cfg.intermediate, cfg.hidden)?,
                w_down: hfq_weight(hfq, gpu, &format!("{p}.mlp.down_proj.weight"), cfg.hidden, cfg.intermediate)?,
            };
            layers.push(layer);
        }

        let has_mq = std::iter::once(&fc)
            .chain(layers.iter().flat_map(|l| {
                [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down].into_iter()
            }))
            .any(|w| matches!(w.gpu_dtype, DType::MQ4G256 | DType::MQ3G256));
        if has_mq {
            // The MQ4 dispatch needs the engine's FWHT sign tables uploaded
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
        let _ = gpu.free_tensor(self.fc.buf);
        let _ = gpu.free_tensor(self.hidden_norm);
        let _ = gpu.free_tensor(self.norm);
        for l in self.layers {
            let _ = gpu.free_tensor(l.attn_norm);
            let _ = gpu.free_tensor(l.wq.buf);
            let _ = gpu.free_tensor(l.wk.buf);
            let _ = gpu.free_tensor(l.wv.buf);
            let _ = gpu.free_tensor(l.wo.buf);
            let _ = gpu.free_tensor(l.q_norm);
            let _ = gpu.free_tensor(l.k_norm);
            let _ = gpu.free_tensor(l.ffn_norm);
            let _ = gpu.free_tensor(l.w_gate.buf);
            let _ = gpu.free_tensor(l.w_up.buf);
            let _ = gpu.free_tensor(l.w_down.buf);
        }
    }
}

// ─── Scratch ───────────────────────────────────────────────────────────────

/// Activation buffers for one forward pass. Sized for up to
/// `max_block_size` query positions and up to `max_ctx_len` context
/// positions. A single scratch is reused across all speculative steps.
pub struct DflashScratch {
    pub max_block_size: usize,
    pub max_ctx_len: usize,

    // Block-sized activations (B rows).
    pub x: GpuTensor,              // [B, hidden] — hidden state rolled across layers
    pub x_norm: GpuTensor,         // [B, hidden]
    pub q: GpuTensor,              // [B, q_dim]
    pub k_noise: GpuTensor,        // [B, kv_dim]
    pub v_noise: GpuTensor,        // [B, kv_dim]
    pub gate: GpuTensor,           // [B, intermediate]
    pub up: GpuTensor,             // [B, intermediate]
    pub gate_up: GpuTensor,        // [B, intermediate]
    pub attn_out: GpuTensor,       // [B, q_dim]
    pub attn_proj: GpuTensor,      // [B, hidden]
    pub residual_attn: GpuTensor,  // [B, hidden]
    pub residual_ffn: GpuTensor,   // [B, hidden]

    // Context activations (L rows), where L ≤ max_ctx_len.
    pub target_hidden: GpuTensor,        // [L, num_extract × hidden]
    pub target_hidden_proj: GpuTensor,   // [L, hidden]
    pub k_ctx: GpuTensor,                // [L, kv_dim]
    pub v_ctx: GpuTensor,                // [L, kv_dim]

    // Concatenated K/V (L + B rows).
    pub k_cat: GpuTensor,                // [L + B, kv_dim]
    pub v_cat: GpuTensor,                // [L + B, kv_dim]

    // Positions (i32).
    pub positions_q: GpuTensor,          // [B]       i32
    pub positions_k: GpuTensor,          // [L + B]   i32

    // FWHT rotation scratch for MQ4 weight paths. Sized to the largest
    // single-call requirement: max(max_ctx × num_extract*hidden,
    // max_block × max_layer_K). Allocated only when DflashWeights.has_mq.
    pub mq_x_rot: Option<GpuTensor>,

    // Incremental-upload tracker for `target_hidden`. On each draft_forward
    // call, the caller passes `target_hidden_host` with `l` total rows. If
    // `uploaded_target_hidden_rows` ≤ l and the caller indicates we can
    // stream-append (ctx_slice == None in spec_step_dflash), we upload only
    // the tail [uploaded..l) rows instead of re-sending the full cumulative
    // context. Drops per-cycle H2D from ~90 MB (full ctx at 1100 tokens ×
    // 5 layers × 4096 × 4 B) to (accept+1) × 5 × 4096 × 4 B ≈ 700 KB —
    // saves ~3–5 ms per cycle on mid-length math prompts.
    //
    // Set to 0 by `reset_upload_tracking` (called at new-prompt boundary).
    // draft_forward updates it after each partial upload.
    pub uploaded_target_hidden_rows: usize,

    /// Absolute (pre-compaction) position of every populated row of
    /// `target_hidden` on GPU. Length always equals the number of valid rows.
    /// Used by `spec_step_dflash` to build non-contiguous `positions_k` when
    /// a TriAttention eviction has compacted `target_hidden` out of order.
    /// Seeded during prompt ingestion and updated on every cycle commit and
    /// every eviction mirror. Empty on the ctx_slice=Some path (caller
    /// manages positions explicitly for that diagnostic mode).
    pub target_hidden_abs_positions: Vec<i32>,

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
    pub k_ctx_cached: Vec<GpuTensor>,
    pub v_ctx_cached: Vec<GpuTensor>,

    /// Number of rows valid in `target_hidden_proj`, `k_ctx_cached[*]`, and
    /// `v_ctx_cached[*]`. Rows `[0..draft_ctx_cached_rows)` have finished
    /// all of (a) fc + hidden_norm projection into target_hidden_proj,
    /// (b) per-layer wk/wv GEMMs, (c) per-layer k_norm. They are still
    /// pre-RoPE — RoPE applies to the full concatenated k_cat each cycle
    /// (cheap; memory-bound on tiny kv_dim tensors).
    ///
    /// Reset to 0 on `reset_upload_tracking` (new prompt) and on
    /// eviction via `invalidate_draft_ctx_cache`. Next cycle after a
    /// reset rebuilds the full prefix in one shot — same cost as a
    /// pre-cache cycle, but amortized thereafter.
    pub draft_ctx_cached_rows: usize,
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
            // The widest single rotation: max(max_ctx × ne*h, max_block × max(intermediate, q_dim)).
            // ne*h on ctx is the `fc` rotation (target_hidden). intermediate is the `w_down`
            // rotation. q_dim is the `wo` rotation. Take the max so a single
            // buffer covers them all.
            let widest = std::cmp::max(l * ne * h, b * std::cmp::max(inter, qd));
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
        for _ in 0..cfg.n_layers {
            k_ctx_cached.push(gpu.alloc_tensor(&[l * kvd], DType::F32)?);
            v_ctx_cached.push(gpu.alloc_tensor(&[l * kvd], DType::F32)?);
        }

        Ok(DflashScratch {
            max_block_size: b,
            max_ctx_len: l,

            x:             gpu.alloc_tensor(&[b * h], DType::F32)?,
            x_norm:        gpu.alloc_tensor(&[b * h], DType::F32)?,
            q:             gpu.alloc_tensor(&[b * qd], DType::F32)?,
            k_noise:       gpu.alloc_tensor(&[b * kvd], DType::F32)?,
            v_noise:       gpu.alloc_tensor(&[b * kvd], DType::F32)?,
            gate:          gpu.alloc_tensor(&[b * inter], DType::F32)?,
            up:            gpu.alloc_tensor(&[b * inter], DType::F32)?,
            gate_up:       gpu.alloc_tensor(&[b * inter], DType::F32)?,
            attn_out:      gpu.alloc_tensor(&[b * qd], DType::F32)?,
            attn_proj:     gpu.alloc_tensor(&[b * h], DType::F32)?,
            residual_attn: gpu.alloc_tensor(&[b * h], DType::F32)?,
            residual_ffn:  gpu.alloc_tensor(&[b * h], DType::F32)?,

            target_hidden:      gpu.alloc_tensor(&[l * ne * h], DType::F32)?,
            target_hidden_proj: gpu.alloc_tensor(&[l * h], DType::F32)?,
            k_ctx:              gpu.alloc_tensor(&[l * kvd], DType::F32)?,
            v_ctx:              gpu.alloc_tensor(&[l * kvd], DType::F32)?,

            k_cat: gpu.alloc_tensor(&[tot * kvd], DType::F32)?,
            v_cat: gpu.alloc_tensor(&[tot * kvd], DType::F32)?,

            positions_q: gpu.alloc_tensor(&[b],   DType::F32)?,
            positions_k: gpu.alloc_tensor(&[tot], DType::F32)?,

            mq_x_rot,
            uploaded_target_hidden_rows: 0,
            target_hidden_abs_positions: Vec::new(),
            k_ctx_cached,
            v_ctx_cached,
            draft_ctx_cached_rows: 0,
        })
    }

    /// Reset the incremental-upload tracker for target_hidden. Call this
    /// at the start of a new prompt / session — otherwise stale tracker
    /// state from a prior prompt would cause the next draft_forward to
    /// skip required rows. Also clears the draft-ctx projection cache so
    /// the first draft_forward after reset does a full rebuild.
    pub fn reset_upload_tracking(&mut self) {
        self.uploaded_target_hidden_rows = 0;
        self.target_hidden_abs_positions.clear();
        self.draft_ctx_cached_rows = 0;
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
        self.draft_ctx_cached_rows = 0;
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
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
/// For MQ4-G256, the kernel needs the input FWHT-rotated. We do that into
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
    let dump = *DUMP.get_or_init(|| {
        std::env::var("HIPFIRE_DRAFT_GEMM_DUMP").ok().as_deref() == Some("1")
    });
    if dump { gpu.hip.device_synchronize()?; }
    let t0 = if dump { Some(std::time::Instant::now()) } else { None };
    let result = match w.gpu_dtype {
        DType::F32 => gpu.gemm_f32_batched(x, &w.buf, y, batch, w.k, w.m),
        DType::F16 => gpu.gemm_f16_batched_lmhead(&w.buf, x, y, w.m, w.k, batch),
        DType::HFQ4G256 => gpu.gemm_hfq4g256_batched_lmhead(&w.buf, x, y, w.m, w.k, batch),
        DType::MQ4G256 => {
            let scratch = mq_x_rot.expect("MQ4 dispatch requires mq_x_rot scratch");
            // Use the prefix [0, batch * k) of the rotation scratch.
            let rot_view = scratch.sub_offset(0, batch * w.k);
            gpu.rotate_x_mq_batched(x, &rot_view, w.k, batch)?;
            gpu.gemm_hfq4g256_batched_lmhead(&w.buf, &rot_view, y, w.m, w.k, batch)
        }
        DType::MQ3G256 => {
            // Mirrors the MQ4 path: pre-rotate x via FWHT (same shared signs
            // as MQ4 — rotate_x_mq_batched is dtype-agnostic for the activation
            // side), invalidate the FP16 x cache because the rotated bytes
            // share the same source pointer, then dispatch the HFQ3 batched
            // lm_head WMMA kernel.
            let scratch = mq_x_rot.expect("MQ3 dispatch requires mq_x_rot scratch");
            let rot_view = scratch.sub_offset(0, batch * w.k);
            gpu.rotate_x_mq_batched(x, &rot_view, w.k, batch)?;
            gpu.fp16_x_source_ptr = std::ptr::null_mut();
            gpu.gemm_hfq3g256_batched_lmhead(&w.buf, &rot_view, y, w.m, w.k, batch)
        }
        other => panic!("dflash gemm_dispatch: unsupported weight dtype {:?}", other),
    };
    if let Some(t) = t0 {
        gpu.hip.device_synchronize()?;
        let us = t.elapsed().as_micros();
        let weight_bytes = match w.gpu_dtype {
            DType::F32 => w.m * w.k * 4,
            DType::F16 => w.m * w.k * 2,
            // HFQ4/MQ4: 136B per group of 256
            _ => w.m * (w.k / 256).max(1) * 136,
        };
        let bytes = weight_bytes + batch * w.k * 4 + batch * w.m * 4 * 2;
        let gbs = (bytes as f64) / (us.max(1) as f64) / 1000.0;
        eprintln!("[draft-gemm] dtype={:?} M={} K={} B={} us={} bytes={}KB GB/s={:.1}",
            w.gpu_dtype, w.m, w.k, batch, us, bytes / 1024, gbs);
    }
    result
}

/// Upload f32 slice into a GPU tensor (bytes via memcpy_htod).
fn upload_slice_f32(gpu: &Gpu, dst: &GpuTensor, data: &[f32]) -> HipResult<()> {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    gpu.hip.memcpy_htod(&dst.buf, bytes)
}

/// Upload i32 slice into a GPU tensor (interpreted as i32 by kernels).
fn upload_slice_i32(gpu: &Gpu, dst: &GpuTensor, data: &[i32]) -> HipResult<()> {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    gpu.hip.memcpy_htod(&dst.buf, bytes)
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
        let row_f32 = ne * h;
        let expected_full_len = l * row_f32;
        let prev = scratch.uploaded_target_hidden_rows;
        // Full-upload conditions: first call, reset flagged, caller shrank
        // the context, or the slice length suggests ctx_slice (unusual l).
        if prev == 0
            || prev > l
            || th_slice.len() != expected_full_len
        {
            upload_slice_f32(gpu, &scratch.target_hidden, th_slice)?;
            scratch.uploaded_target_hidden_rows = l;
        } else if prev < l {
            // Delta-upload: rows [prev..l) need to land at byte offset
            // prev * row_f32 * 4 of scratch.target_hidden.
            let tail = &th_slice[prev * row_f32..];
            let dst_byte_off = prev * row_f32 * 4;
            let src_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    tail.as_ptr() as *const u8,
                    tail.len() * 4,
                )
            };
            gpu.hip.memcpy_htod_offset(&scratch.target_hidden.buf, dst_byte_off, src_bytes)?;
            scratch.uploaded_target_hidden_rows = l;
        }
        // prev == l: nothing new to upload (wouldn't happen in practice
        // since caller always appends, but harmless).
    }
    upload_slice_i32(gpu, &scratch.positions_q, positions_q)?;
    upload_slice_i32(gpu, &scratch.positions_k, positions_k)?;

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
    let cached_rows = scratch.draft_ctx_cached_rows;
    let delta = l.saturating_sub(cached_rows);
    if delta > 0 {
        let src_offset_elems = cached_rows * ne * h;
        let dst_offset_elems = cached_rows * h;
        let th_slice = scratch.target_hidden.sub_offset(src_offset_elems, delta * ne * h);
        let thp_slice = scratch.target_hidden_proj.sub_offset(dst_offset_elems, delta * h);
        gemm_dispatch(
            gpu,
            &th_slice,
            &weights.fc,
            &thp_slice,
            delta,
            scratch.mq_x_rot.as_ref(),
        )?;
        gpu.rmsnorm_batched(
            &thp_slice,
            &weights.hidden_norm,
            &thp_slice,
            delta,
            h,
            eps,
        )?;
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
    let dbg = std::env::var("HIPFIRE_DRAFT_SUBPHASE").ok().as_deref() == Some("1");
    let mut us_attn_gemm: u128 = 0;
    let mut us_attn_kernel: u128 = 0;
    let mut us_ffn_gemm: u128 = 0;
    let mut us_concat: u128 = 0;
    if dbg { gpu.hip.device_synchronize()?; }

    // ── 2. Per-layer decoder ─────────────────────────────────────────────
    for li in 0..cfg.n_layers {
        let layer = &weights.layers[li];

        // Residual.
        gpu.hip.memcpy_dtod(&scratch.residual_attn.buf, &scratch.x.buf, (b * h) * 4)?;

        // attn_norm.
        gpu.rmsnorm_batched(
            &scratch.x,
            &layer.attn_norm,
            &scratch.x_norm,
            b,
            h,
            eps,
        )?;

        let t0 = if dbg {
            gpu.hip.device_synchronize()?;
            Some(std::time::Instant::now())
        } else { None };

        // Q/K/V projections — dispatched on each weight's dtype.
        // Q and K/V noise (over the B block positions) must be computed
        // every cycle. K_ctx and V_ctx (over the L context positions)
        // are *incrementally* cached — see the per-layer block below.
        gemm_dispatch(gpu, &scratch.x_norm, &layer.wq, &scratch.q,       b, scratch.mq_x_rot.as_ref())?;
        gemm_dispatch(gpu, &scratch.x_norm, &layer.wk, &scratch.k_noise, b, scratch.mq_x_rot.as_ref())?;
        gemm_dispatch(gpu, &scratch.x_norm, &layer.wv, &scratch.v_noise, b, scratch.mq_x_rot.as_ref())?;

        // K_ctx / V_ctx — same wk/wv weights but projected over the L
        // accepted-context rows of target_hidden_proj. INCREMENTAL PATH:
        // only rows [cached_rows..L) need projection; rows [0..cached_rows)
        // were projected in a prior call and stored in the per-layer
        // k_ctx_cached / v_ctx_cached buffers (post-k_norm for K).
        //
        // If delta > 0, run wk/wv on the delta slice of target_hidden_proj
        // and write into the tail of the per-layer cache. Then run k_norm
        // on those same delta rows (per-head) in-place on the cache.
        //
        // For correctness note: the per-head K RMSNorm is row-local
        // (normalizes each kv_head row of size hd independently), so
        // applying it to delta rows of the cache is exactly equivalent
        // to applying it to the full k_cat post-concat. V has no
        // draft-level norm so v_ctx_cached stores raw GEMM output.
        let k_cache_layer = &scratch.k_ctx_cached[li];
        let v_cache_layer = &scratch.v_ctx_cached[li];
        if delta > 0 {
            let src_offset_elems = cached_rows * h;
            let dst_offset_elems = cached_rows * kvd;
            let thp_slice = scratch.target_hidden_proj.sub_offset(src_offset_elems, delta * h);
            let k_slot = k_cache_layer.sub_offset(dst_offset_elems, delta * kvd);
            let v_slot = v_cache_layer.sub_offset(dst_offset_elems, delta * kvd);
            gemm_dispatch(gpu, &thp_slice, &layer.wk, &k_slot, delta, scratch.mq_x_rot.as_ref())?;
            gemm_dispatch(gpu, &thp_slice, &layer.wv, &v_slot, delta, scratch.mq_x_rot.as_ref())?;
            // Per-head RMSNorm on K delta rows only. batch = delta × n_kv_heads.
            gpu.rmsnorm_batched(&k_slot, &layer.k_norm, &k_slot, delta * cfg.n_kv_heads, hd, eps)?;
        }

        if let Some(t) = t0 {
            gpu.hip.device_synchronize()?;
            us_attn_gemm += t.elapsed().as_micros();
        }
        let t1 = if dbg {
            Some(std::time::Instant::now())
        } else { None };

        // Concat K = [K_ctx_cached | K_noise] → k_cat [L + B, kv_dim].
        // The cached K prefix is already post-k_norm (applied incrementally
        // above); the noise tail still needs k_norm applied below.
        let ctx_bytes   = (l * kvd) * 4;
        let noise_bytes = (b * kvd) * 4;
        gpu.hip.memcpy_dtod_at(&scratch.k_cat.buf, 0,          &k_cache_layer.buf,   0, ctx_bytes)?;
        gpu.hip.memcpy_dtod_at(&scratch.k_cat.buf, ctx_bytes,  &scratch.k_noise.buf, 0, noise_bytes)?;
        gpu.hip.memcpy_dtod_at(&scratch.v_cat.buf, 0,          &v_cache_layer.buf,   0, ctx_bytes)?;
        gpu.hip.memcpy_dtod_at(&scratch.v_cat.buf, ctx_bytes,  &scratch.v_noise.buf, 0, noise_bytes)?;

        // Per-head RMSNorm on Q: each of B*n_heads rows, size head_dim,
        // weight [head_dim].
        gpu.rmsnorm_batched(&scratch.q, &layer.q_norm, &scratch.q, b * cfg.n_heads, hd, eps)?;
        // Per-head RMSNorm on the NOISE tail of K_cat only — the cached
        // prefix was already normed when it was inserted into the layer's
        // k_ctx_cached. batch = B × n_kv_heads, applied to the last B rows
        // of k_cat.
        {
            let noise_slot = scratch.k_cat.sub_offset(l * kvd, b * kvd);
            gpu.rmsnorm_batched(&noise_slot, &layer.k_norm, &noise_slot, b * cfg.n_kv_heads, hd, eps)?;
        }

        // RoPE. rope_batched_f32 expects q and k at the SAME batch size,
        // rotating at per-row positions. We call it twice with a zero
        // "head count" on the inactive tensor so its loop doesn't execute.
        // Call 1: rotate Q with positions_q. Pass k as a valid buffer
        // (scratch.k_noise is shape-compatible; n_heads_k=0 skips its loop).
        gpu.rope_batched_f32(
            &scratch.q,
            &scratch.k_noise,      // ignored because n_heads_k = 0
            &scratch.positions_q,  // [B]
            cfg.n_heads,
            0,
            hd,
            theta,
            b,
        )?;
        // Call 2: rotate K_cat with positions_k. n_heads_q = 0 skips Q.
        gpu.rope_batched_f32(
            &scratch.q,            // ignored because n_heads_q = 0
            &scratch.k_cat,
            &scratch.positions_k,  // [L + B]
            0,
            cfg.n_kv_heads,
            hd,
            theta,
            tot,
        )?;

        if let Some(t) = t1 {
            gpu.hip.device_synchronize()?;
            us_concat += t.elapsed().as_micros();
        }
        let t2 = if dbg {
            Some(std::time::Instant::now())
        } else { None };

        // Attention: Q [B, n_heads, hd] × K [tot, n_kv_heads, hd]^T → scores
        // (with GQA expansion) → softmax → @V.
        gpu.attention_dflash_f32(
            &scratch.q,
            &scratch.k_cat,
            &scratch.v_cat,
            &scratch.attn_out,
            b,
            tot,
            cfg.n_heads,
            cfg.n_kv_heads,
            hd,
        )?;
        if let Some(t) = t2 {
            gpu.hip.device_synchronize()?;
            us_attn_kernel += t.elapsed().as_micros();
        }
        let t3 = if dbg { Some(std::time::Instant::now()) } else { None };

        // attn_proj = attn_out @ wo^T → [B, hidden]
        gemm_dispatch(gpu, &scratch.attn_out, &layer.wo, &scratch.attn_proj, b, scratch.mq_x_rot.as_ref())?;

        // x = residual_attn + attn_proj
        gpu.add_f32(&scratch.residual_attn, &scratch.attn_proj, &scratch.x)?;

        // Residual for FFN.
        gpu.hip.memcpy_dtod(&scratch.residual_ffn.buf, &scratch.x.buf, (b * h) * 4)?;

        // ffn_norm.
        gpu.rmsnorm_batched(&scratch.x, &layer.ffn_norm, &scratch.x_norm, b, h, eps)?;

        // gate = x_norm @ w_gate^T; up = x_norm @ w_up^T
        gemm_dispatch(gpu, &scratch.x_norm, &layer.w_gate, &scratch.gate, b, scratch.mq_x_rot.as_ref())?;
        gemm_dispatch(gpu, &scratch.x_norm, &layer.w_up,   &scratch.up,   b, scratch.mq_x_rot.as_ref())?;
        // 2026-04-21: tried target's fused gemm_gate_up_hfq4g256 here (shared
        // FP16-X convert + interleaved gate/up GEMMs). Byte-exact A/B neutral
        // on 27B HumanEval (median 76.47 fused vs 76.74 baseline; ±7 % run-to-
        // run variance from ksplit's non-deterministic atomicAdd dominates).
        // Kept per-weight dispatch for clarity. The real draft perf lever is
        // the ~56 ms of ffn_gemm per cycle (see HIPFIRE_DRAFT_SUBPHASE=1);
        // fusion alone doesn't move that number — kernel engineering does.

        // SiLU(gate) * up → gate_up
        gpu.silu_mul_f32(&scratch.gate, &scratch.up, &scratch.gate_up)?;

        // x = w_down @ gate_up^T  (output [B, hidden])
        gemm_dispatch(gpu, &scratch.gate_up, &layer.w_down, &scratch.x, b, scratch.mq_x_rot.as_ref())?;

        // x = residual_ffn + x
        gpu.add_f32(&scratch.residual_ffn, &scratch.x, &scratch.x)?;

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
    scratch.draft_ctx_cached_rows = l;

    Ok(())
}
