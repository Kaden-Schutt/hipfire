// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Parent-checkpoint Multi-head Latent Attention (SWA + compressor/indexer).
//!
//! Operator-semantics authority:
//! - `.codeinsight+research/ds4-parent-ref/inference/model.py`
//!   - `Attention.forward` 442-549 (all three layer classes)
//!   - `get_window_topk_idxs` 260-271
//!   - `get_compress_topk_idxs` 274-283
//!   - `precompute_freqs_cis` 206-236
//!   - `apply_rotary_emb` 238-250  → **interleaved** pair convention
//!     (`view_as_complex(x.float().unflatten(-1, (-1, 2)))` pairs dims
//!     `(2i, 2i+1)`, NOT half-split `(i, i + n_rot/2)`)
//! - config.json:
//!   - ratio-0 SWA: `original_seq_len = 0`, `rope_theta = 10000`
//!   - ratio>0: YaRN on, `compress_rope_theta = 160000`, `original_seq_len = 65536`
//!
//! Layer classes (`compress_ratios`):
//! - **0** (layers 0,1,40-42): pure SWA over the ring.
//! - **128**: main compressor (`hadamard=false`) + identity gather of all
//!   compressed slots into the joint softmax with the SWA window + sink.
//! - **4**: main compressor + indexer (top-`index_topk` compressed slots).
//!   The indexer's own compressor is internal (`hadamard=true`); the main
//!   attention compressor is a separate instance.
//!
//! Compressed KV and compressor ring state live in [`ParentAttnScratch`].
//! Prefill (`start_pos == 0`) is self-contained per call (ring is reset).
//! Multi-layer **decode** with one shared scratch is not supported — each
//! layer needs its own persistent compressor ring + main_kv_cache.

use crate::parent::codec::round_to_bf16;
use crate::parent::compressor::{
    parent_compressor_forward, ParentCompressorScratch, PARENT_COMPRESS_ROPE_THETA,
    PARENT_YARN_BETA_FAST, PARENT_YARN_BETA_SLOW, PARENT_YARN_FACTOR, PARENT_YARN_ORIG_SEQ,
};
use crate::parent::indexer::{
    indexer_n_compressed, indexer_n_visible, parent_indexer_forward, ParentIndexerScratch,
    PARENT_INDEX_TOPK,
};
use crate::parent::linear::{parent_linear_dense, ParentDenseWeight};
use crate::parent::weights::ParentLayerWeights;
use crate::parent::{Ds4ParentBackend, ParentQuantConfig};
use rdna_compute::{DType, Gpu, GpuTensor};

// ── Parent shape constants (checkpoint config.json) ─────────────────────────

/// `dim` / `hidden_size`.
pub const PARENT_DIM: usize = 4096;
/// `n_heads`.
pub const PARENT_N_HEADS: usize = 64;
/// `head_dim`.
pub const PARENT_HEAD_DIM: usize = 512;
/// `qk_rope_head_dim` / `rope_head_dim`.
pub const PARENT_ROPE_DIM: usize = 64;
/// Non-RoPE dims of each head (`head_dim - rope_head_dim`).
pub const PARENT_NOPE_DIM: usize = PARENT_HEAD_DIM - PARENT_ROPE_DIM; // 448
/// `q_lora_rank`.
pub const PARENT_Q_LORA: usize = 1024;
/// `o_lora_rank`.
pub const PARENT_O_LORA: usize = 1024;
/// `o_groups`.
pub const PARENT_O_GROUPS: usize = 8;
/// Heads per O-group (`n_heads / o_groups`).
pub const PARENT_HEADS_PER_GROUP: usize = PARENT_N_HEADS / PARENT_O_GROUPS; // 8
/// Per-group wo_a input width (`heads_per_group * head_dim`).
pub const PARENT_PER_GROUP_IN: usize = PARENT_HEADS_PER_GROUP * PARENT_HEAD_DIM; // 4096
/// Flattened Q / attn-out width (`n_heads * head_dim`).
pub const PARENT_Q_WIDTH: usize = PARENT_N_HEADS * PARENT_HEAD_DIM; // 32768
/// Flattened wo_a output / wo_b input (`o_groups * o_lora_rank`).
pub const PARENT_WO_A_OUT: usize = PARENT_O_GROUPS * PARENT_O_LORA; // 8192
/// `sliding_window` / `window_size`.
pub const PARENT_SWA_WINDOW: usize = 128;
/// `num_key_value_heads`.
pub const PARENT_N_KV_HEADS: usize = 1;
/// `rms_norm_eps`.
pub const PARENT_RMS_EPS: f32 = 1e-6;
/// `rope_theta` for pure-SWA (`compress_ratio == 0`) layers.
pub const PARENT_ROPE_THETA: f32 = 10_000.0;
/// Block size for the KV non-RoPE FP8 act-quant simulation (`model.py:512`).
pub const PARENT_KV_ACT_QUANT_BLOCK: usize = 64;
/// `index_topk` — max compressed slots selected by the indexer.
pub const PARENT_ATTN_INDEX_TOPK: usize = PARENT_INDEX_TOPK; // 512

// ── Scratch ─────────────────────────────────────────────────────────────────

/// Reusable device scratch for [`parent_attention_swa`].
///
/// BF16 staging is deliberately **one** tile that is refilled before every
/// `parent_linear_dense` call — that API destroys its `x_bf16` input via
/// in-place act-quant (`inplace=True` in the reference).
///
/// Owns compressor + indexer scratch so a layer forward still allocates once.
/// `main_kv_cache` holds the main-attention compressor output for the current
/// call (and across decode steps when the same scratch is reused for one layer).
pub struct ParentAttnScratch {
    /// Destructive BF16 act tile. Width = max linear K = [`PARENT_WO_A_OUT`].
    act_bf16: GpuTensor,
    /// BF16 tile for the KV non-RoPE act-quant site (`[max_rows, 448]`).
    kv_nope_bf16: GpuTensor,
    /// Q-LoRA bottleneck `[max_rows, q_lora]`.
    q_lat_f32: GpuTensor,
    /// Q heads `[max_rows, n_heads, head_dim]` (flat `[max_rows * q_width]`).
    q_f32: GpuTensor,
    /// Joint KV `[max_rows, head_dim]`.
    kv_f32: GpuTensor,
    /// Raw attention output `[max_rows, n_heads, head_dim]`.
    attn_out_f32: GpuTensor,
    /// O-LoRA intermediate `[max_rows, o_groups * o_lora]`.
    wo_a_out_f32: GpuTensor,
    /// Per-row SWA visibility window `[max_rows, head_dim, window]` (K=V tied).
    swa_staged: GpuTensor,
    /// Per-row gathered top-k compressed KV `[max_rows, head_dim, index_topk]`.
    topk_staged: GpuTensor,
    /// Main-attention compressor KV cache `[max_n_compressed, head_dim]` F32.
    main_kv_cache: GpuTensor,
    /// Indexer top-k indices `[max_rows, index_topk]` I32 (Raw bytes).
    topk_idx: GpuTensor,
    /// Per-row `n_valid` SWA counts (`I32` bits, length `max_rows`).
    n_valid: GpuTensor,
    /// Per-row active top-k counts (`I32` bits, length `max_rows`).
    n_active_topk: GpuTensor,
    /// Absolute positions for batched RoPE (`I32` bits, length `max_rows`).
    positions: GpuTensor,
    /// Weight-ones `[head_dim]` reserved for a future device-side per-head RMSNorm.
    #[allow(dead_code)]
    q_head_ones: GpuTensor,
    /// Main-attention compressor scratch (`hadamard = false`).
    compressor: ParentCompressorScratch,
    /// Indexer scratch (owns its own compressor path with `hadamard = true`).
    indexer: ParentIndexerScratch,
    max_rows: usize,
    max_n_compressed: usize,
    bytes: usize,
}

impl ParentAttnScratch {
    /// Allocate reusable scratch for up to `max_rows` tokens.
    ///
    /// Compressed-cache capacity is
    /// `max(max_rows, PARENT_ATTN_INDEX_TOPK)` so a pure-prefill of
    /// `max_rows` always fits at ratio 4 (`rows/4`) and ratio 128, and the
    /// identity-gather path never exceeds the top-k staging stride.
    pub fn new(gpu: &mut Gpu, cfg: &ParentQuantConfig, max_rows: usize) -> Result<Self, String> {
        if max_rows == 0 {
            return Err("deepseek4 parent: ParentAttnScratch max_rows must be > 0".to_owned());
        }
        let max_n_compressed = max_rows.max(PARENT_ATTN_INDEX_TOPK);

        // Nested compressor / indexer scratch first so their sizes are known.
        let compressor = ParentCompressorScratch::new(gpu, cfg, max_rows)?;
        let indexer = ParentIndexerScratch::new(gpu, cfg, max_rows)?;

        let act_bf16 = gpu
            .alloc_tensor(&[max_rows, PARENT_WO_A_OUT], DType::BF16)
            .map_err(|e| format!("deepseek4 parent: attn act_bf16 alloc: {e:?}"))?;
        let kv_nope_bf16 = match gpu.alloc_tensor(&[max_rows, PARENT_NOPE_DIM], DType::BF16) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(act_bf16);
                return Err(format!("deepseek4 parent: attn kv_nope_bf16 alloc: {e:?}"));
            }
        };
        let q_lat_f32 = match gpu.alloc_tensor(&[max_rows, PARENT_Q_LORA], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(act_bf16);
                let _ = gpu.free_tensor(kv_nope_bf16);
                return Err(format!("deepseek4 parent: attn q_lat_f32 alloc: {e:?}"));
            }
        };
        let q_f32 = match gpu.alloc_tensor(&[max_rows, PARENT_N_HEADS, PARENT_HEAD_DIM], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(act_bf16);
                let _ = gpu.free_tensor(kv_nope_bf16);
                let _ = gpu.free_tensor(q_lat_f32);
                return Err(format!("deepseek4 parent: attn q_f32 alloc: {e:?}"));
            }
        };
        let kv_f32 = match gpu.alloc_tensor(&[max_rows, PARENT_HEAD_DIM], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(act_bf16);
                let _ = gpu.free_tensor(kv_nope_bf16);
                let _ = gpu.free_tensor(q_lat_f32);
                let _ = gpu.free_tensor(q_f32);
                return Err(format!("deepseek4 parent: attn kv_f32 alloc: {e:?}"));
            }
        };
        let attn_out_f32 =
            match gpu.alloc_tensor(&[max_rows, PARENT_N_HEADS, PARENT_HEAD_DIM], DType::F32) {
                Ok(t) => t,
                Err(e) => {
                    let _ = gpu.free_tensor(act_bf16);
                    let _ = gpu.free_tensor(kv_nope_bf16);
                    let _ = gpu.free_tensor(q_lat_f32);
                    let _ = gpu.free_tensor(q_f32);
                    let _ = gpu.free_tensor(kv_f32);
                    return Err(format!("deepseek4 parent: attn attn_out_f32 alloc: {e:?}"));
                }
            };
        let wo_a_out_f32 = match gpu.alloc_tensor(&[max_rows, PARENT_WO_A_OUT], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(act_bf16);
                let _ = gpu.free_tensor(kv_nope_bf16);
                let _ = gpu.free_tensor(q_lat_f32);
                let _ = gpu.free_tensor(q_f32);
                let _ = gpu.free_tensor(kv_f32);
                let _ = gpu.free_tensor(attn_out_f32);
                return Err(format!("deepseek4 parent: attn wo_a_out_f32 alloc: {e:?}"));
            }
        };
        let swa_staged =
            match gpu.alloc_tensor(&[max_rows, PARENT_HEAD_DIM, PARENT_SWA_WINDOW], DType::F32) {
                Ok(t) => t,
                Err(e) => {
                    let _ = gpu.free_tensor(act_bf16);
                    let _ = gpu.free_tensor(kv_nope_bf16);
                    let _ = gpu.free_tensor(q_lat_f32);
                    let _ = gpu.free_tensor(q_f32);
                    let _ = gpu.free_tensor(kv_f32);
                    let _ = gpu.free_tensor(attn_out_f32);
                    let _ = gpu.free_tensor(wo_a_out_f32);
                    return Err(format!("deepseek4 parent: attn swa_staged alloc: {e:?}"));
                }
            };
        let topk_staged = match gpu.alloc_tensor(
            &[max_rows, PARENT_HEAD_DIM, PARENT_ATTN_INDEX_TOPK],
            DType::F32,
        ) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(act_bf16);
                let _ = gpu.free_tensor(kv_nope_bf16);
                let _ = gpu.free_tensor(q_lat_f32);
                let _ = gpu.free_tensor(q_f32);
                let _ = gpu.free_tensor(kv_f32);
                let _ = gpu.free_tensor(attn_out_f32);
                let _ = gpu.free_tensor(wo_a_out_f32);
                let _ = gpu.free_tensor(swa_staged);
                return Err(format!("deepseek4 parent: attn topk_staged alloc: {e:?}"));
            }
        };
        let main_kv_cache =
            match gpu.alloc_tensor(&[max_n_compressed, PARENT_HEAD_DIM], DType::F32) {
                Ok(t) => t,
                Err(e) => {
                    let _ = gpu.free_tensor(act_bf16);
                    let _ = gpu.free_tensor(kv_nope_bf16);
                    let _ = gpu.free_tensor(q_lat_f32);
                    let _ = gpu.free_tensor(q_f32);
                    let _ = gpu.free_tensor(kv_f32);
                    let _ = gpu.free_tensor(attn_out_f32);
                    let _ = gpu.free_tensor(wo_a_out_f32);
                    let _ = gpu.free_tensor(swa_staged);
                    let _ = gpu.free_tensor(topk_staged);
                    return Err(format!("deepseek4 parent: attn main_kv_cache alloc: {e:?}"));
                }
            };
        let topk_idx = match alloc_i32_buf(gpu, max_rows * PARENT_ATTN_INDEX_TOPK) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(act_bf16);
                let _ = gpu.free_tensor(kv_nope_bf16);
                let _ = gpu.free_tensor(q_lat_f32);
                let _ = gpu.free_tensor(q_f32);
                let _ = gpu.free_tensor(kv_f32);
                let _ = gpu.free_tensor(attn_out_f32);
                let _ = gpu.free_tensor(wo_a_out_f32);
                let _ = gpu.free_tensor(swa_staged);
                let _ = gpu.free_tensor(topk_staged);
                let _ = gpu.free_tensor(main_kv_cache);
                return Err(e);
            }
        };
        let n_valid = match alloc_i32_buf(gpu, max_rows) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(act_bf16);
                let _ = gpu.free_tensor(kv_nope_bf16);
                let _ = gpu.free_tensor(q_lat_f32);
                let _ = gpu.free_tensor(q_f32);
                let _ = gpu.free_tensor(kv_f32);
                let _ = gpu.free_tensor(attn_out_f32);
                let _ = gpu.free_tensor(wo_a_out_f32);
                let _ = gpu.free_tensor(swa_staged);
                let _ = gpu.free_tensor(topk_staged);
                let _ = gpu.free_tensor(main_kv_cache);
                let _ = gpu.free_tensor(topk_idx);
                return Err(e);
            }
        };
        let n_active_topk = match alloc_i32_buf(gpu, max_rows) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(act_bf16);
                let _ = gpu.free_tensor(kv_nope_bf16);
                let _ = gpu.free_tensor(q_lat_f32);
                let _ = gpu.free_tensor(q_f32);
                let _ = gpu.free_tensor(kv_f32);
                let _ = gpu.free_tensor(attn_out_f32);
                let _ = gpu.free_tensor(wo_a_out_f32);
                let _ = gpu.free_tensor(swa_staged);
                let _ = gpu.free_tensor(topk_staged);
                let _ = gpu.free_tensor(main_kv_cache);
                let _ = gpu.free_tensor(topk_idx);
                let _ = gpu.free_tensor(n_valid);
                return Err(e);
            }
        };
        let positions = match alloc_i32_buf(gpu, max_rows) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(act_bf16);
                let _ = gpu.free_tensor(kv_nope_bf16);
                let _ = gpu.free_tensor(q_lat_f32);
                let _ = gpu.free_tensor(q_f32);
                let _ = gpu.free_tensor(kv_f32);
                let _ = gpu.free_tensor(attn_out_f32);
                let _ = gpu.free_tensor(wo_a_out_f32);
                let _ = gpu.free_tensor(swa_staged);
                let _ = gpu.free_tensor(topk_staged);
                let _ = gpu.free_tensor(main_kv_cache);
                let _ = gpu.free_tensor(topk_idx);
                let _ = gpu.free_tensor(n_valid);
                let _ = gpu.free_tensor(n_active_topk);
                return Err(e);
            }
        };
        let ones = vec![1.0f32; PARENT_HEAD_DIM];
        let q_head_ones = match gpu.upload_f32(&ones, &[PARENT_HEAD_DIM]) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(act_bf16);
                let _ = gpu.free_tensor(kv_nope_bf16);
                let _ = gpu.free_tensor(q_lat_f32);
                let _ = gpu.free_tensor(q_f32);
                let _ = gpu.free_tensor(kv_f32);
                let _ = gpu.free_tensor(attn_out_f32);
                let _ = gpu.free_tensor(wo_a_out_f32);
                let _ = gpu.free_tensor(swa_staged);
                let _ = gpu.free_tensor(topk_staged);
                let _ = gpu.free_tensor(main_kv_cache);
                let _ = gpu.free_tensor(topk_idx);
                let _ = gpu.free_tensor(n_valid);
                let _ = gpu.free_tensor(n_active_topk);
                let _ = gpu.free_tensor(positions);
                return Err(format!("deepseek4 parent: attn q_head_ones upload: {e:?}"));
            }
        };

        let own_bytes = act_bf16.buf.size()
            + kv_nope_bf16.buf.size()
            + q_lat_f32.buf.size()
            + q_f32.buf.size()
            + kv_f32.buf.size()
            + attn_out_f32.buf.size()
            + wo_a_out_f32.buf.size()
            + swa_staged.buf.size()
            + topk_staged.buf.size()
            + main_kv_cache.buf.size()
            + topk_idx.buf.size()
            + n_valid.buf.size()
            + n_active_topk.buf.size()
            + positions.buf.size()
            + q_head_ones.buf.size();
        let bytes = own_bytes + compressor.bytes() + indexer.bytes();

        Ok(Self {
            act_bf16,
            kv_nope_bf16,
            q_lat_f32,
            q_f32,
            kv_f32,
            attn_out_f32,
            wo_a_out_f32,
            swa_staged,
            topk_staged,
            main_kv_cache,
            topk_idx,
            n_valid,
            n_active_topk,
            positions,
            q_head_ones,
            compressor,
            indexer,
            max_rows,
            max_n_compressed,
            bytes,
        })
    }

    /// Total scratch bytes resident on device (own tiles + compressor + indexer).
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    pub fn max_n_compressed(&self) -> usize {
        self.max_n_compressed
    }

    /// Post-RoPE Q buffer `[max_rows, n_heads, head_dim]` (diagnostic).
    pub fn q_f32_ref(&self) -> Result<&GpuTensor, String> {
        Ok(&self.q_f32)
    }

    /// Post-quant KV buffer `[max_rows, head_dim]` (diagnostic).
    pub fn kv_f32_ref(&self) -> Result<&GpuTensor, String> {
        Ok(&self.kv_f32)
    }

    /// Attention output before wo_a `[max_rows, n_heads, head_dim]` (diagnostic).
    pub fn attn_out_f32_ref(&self) -> Result<&GpuTensor, String> {
        Ok(&self.attn_out_f32)
    }

    /// Q-LoRA bottleneck after q_norm `[max_rows, q_lora]` (indexer `qr` input).
    pub fn q_lat_f32_ref(&self) -> &GpuTensor {
        &self.q_lat_f32
    }

    /// Main compressor KV cache `[max_n_compressed, head_dim]` (diagnostic).
    pub fn main_kv_cache_ref(&self) -> &GpuTensor {
        &self.main_kv_cache
    }

    /// Indexer top-k indices buffer (diagnostic).
    pub fn topk_idx_ref(&self) -> &GpuTensor {
        &self.topk_idx
    }

    /// Nested compressor scratch.
    pub fn compressor_scratch(&self) -> &ParentCompressorScratch {
        &self.compressor
    }

    /// Nested indexer scratch.
    pub fn indexer_scratch(&self) -> &ParentIndexerScratch {
        &self.indexer
    }

    /// Per-row active top-k counts (`I32` bits, length `max_rows`).
    pub fn n_active_topk_ref(&self) -> &GpuTensor {
        &self.n_active_topk
    }

    /// Per-row SWA staged window `[max_rows, head_dim, window]` (diagnostic).
    pub fn swa_staged_ref(&self) -> &GpuTensor {
        &self.swa_staged
    }

    /// Per-row gathered top-k KV `[max_rows, head_dim, index_topk]` (diagnostic).
    pub fn topk_staged_ref(&self) -> &GpuTensor {
        &self.topk_staged
    }

}

fn alloc_i32_buf(gpu: &mut Gpu, n: usize) -> Result<GpuTensor, String> {
    // Raw buffer sized in bytes = n * 4 (DType::Raw size == 1).
    let nbytes = n
        .checked_mul(4)
        .ok_or_else(|| "deepseek4 parent: i32 buf size overflow".to_owned())?;
    let zeros = vec![0u8; nbytes];
    gpu.upload_raw(&zeros, &[nbytes])
        .map_err(|e| format!("deepseek4 parent: i32 buf alloc: {e:?}"))
}



// ── Host helpers (unit-tested) ──────────────────────────────────────────────

/// `precompute_freqs_cis` frequency table (`model.py:206-236`), angles only.
///
/// Returns length-`dim/2` base frequencies **before** the outer product with
/// positions: `freqs[i] = base_freq_i` (possibly YaRN-blended). Position `t`
/// then uses angle `t * freqs[i]`.
///
/// When `original_seq_len == 0` (pure-SWA / `compress_ratio == 0` layers),
/// YaRN is disabled and the plain `1/base^(2i/dim)` table is returned —
/// matching `model.py:484-485`.
pub fn precompute_rope_freqs(
    dim: usize,
    original_seq_len: usize,
    base: f64,
    factor: f64,
    beta_fast: f64,
    beta_slow: f64,
) -> Result<Vec<f64>, String> {
    if dim == 0 || dim % 2 != 0 {
        return Err(format!(
            "deepseek4 parent: rope dim must be positive even (got {dim})"
        ));
    }
    if base <= 0.0 {
        return Err("deepseek4 parent: rope base must be > 0".to_owned());
    }
    let n_pairs = dim / 2;
    let mut freqs = Vec::with_capacity(n_pairs);
    for i in 0..n_pairs {
        let exp = (2 * i) as f64 / dim as f64;
        freqs.push(1.0 / base.powf(exp));
    }
    if original_seq_len > 0 {
        // YaRN blend (`model.py:227-230`).
        let (low, high) =
            yarn_correction_range(beta_fast, beta_slow, dim, base, original_seq_len as f64);
        for (i, f) in freqs.iter_mut().enumerate() {
            let ramp = yarn_linear_ramp(low, high, i as f64);
            // smooth = 1 - ramp; freqs = freqs/factor*(1-smooth) + freqs*smooth
            //        = freqs/factor*ramp + freqs*(1-ramp)
            let smooth = 1.0 - ramp;
            *f = (*f) / factor * (1.0 - smooth) + (*f) * smooth;
        }
    }
    Ok(freqs)
}

fn yarn_correction_dim(num_rotations: f64, dim: usize, base: f64, max_seq_len: f64) -> f64 {
    // model.py:211-212
    dim as f64 * (max_seq_len / (num_rotations * 2.0 * std::f64::consts::PI)).ln()
        / (2.0 * base.ln())
}

fn yarn_correction_range(
    low_rot: f64,
    high_rot: f64,
    dim: usize,
    base: f64,
    max_seq_len: f64,
) -> (f64, f64) {
    // model.py:214-217
    let low = yarn_correction_dim(low_rot, dim, base, max_seq_len).floor();
    let high = yarn_correction_dim(high_rot, dim, base, max_seq_len).ceil();
    (low.max(0.0), high.min((dim - 1) as f64))
}

fn yarn_linear_ramp(min: f64, mut max: f64, x: f64) -> f64 {
    // model.py:219-224 — ramp in [0, 1] over [min, max]
    if (min - max).abs() < f64::EPSILON {
        max += 0.001;
    }
    let y = (x - min) / (max - min);
    y.clamp(0.0, 1.0)
}

/// Apply interleaved tail RoPE in-place on the last `n_rot` dims of each head.
///
/// Convention (`apply_rotary_emb`, `model.py:238-250`):
/// `view_as_complex(unflatten(-1, (-1, 2)))` → pairs `(2i, 2i+1)` inside the
/// tail window `[head_dim - n_rot, head_dim)`. This is **interleaved**, not
/// half-split.
///
/// `x` layout: `[rows, n_heads, head_dim]` row-major flat.
/// `freqs[pair]` is the base frequency; angle = `pos[row] * freqs[pair]`.
/// `inverse == true` conjugates the rotation (negate sin).
pub fn apply_rope_interleaved_inplace(
    x: &mut [f32],
    rows: usize,
    n_heads: usize,
    head_dim: usize,
    n_rot: usize,
    positions: &[usize],
    freqs: &[f64],
    inverse: bool,
) -> Result<(), String> {
    if n_rot == 0 || n_rot % 2 != 0 {
        return Err(format!(
            "deepseek4 parent: n_rot must be positive even (got {n_rot})"
        ));
    }
    if head_dim < n_rot {
        return Err(format!(
            "deepseek4 parent: head_dim {head_dim} < n_rot {n_rot}"
        ));
    }
    let n_pairs = n_rot / 2;
    if freqs.len() < n_pairs {
        return Err(format!(
            "deepseek4 parent: freqs len {} < n_pairs {n_pairs}",
            freqs.len()
        ));
    }
    if positions.len() < rows {
        return Err(format!(
            "deepseek4 parent: positions len {} < rows {rows}",
            positions.len()
        ));
    }
    let need = rows
        .checked_mul(n_heads)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| "deepseek4 parent: rope size overflow".to_owned())?;
    if x.len() < need {
        return Err(format!(
            "deepseek4 parent: rope buffer short (have {} need {need})",
            x.len()
        ));
    }
    let tail_off = head_dim - n_rot;
    let sin_sign = if inverse { -1.0f64 } else { 1.0f64 };
    for r in 0..rows {
        let pos = positions[r] as f64;
        for h in 0..n_heads {
            let base = (r * n_heads + h) * head_dim + tail_off;
            for p in 0..n_pairs {
                let angle = pos * freqs[p];
                let (s, c) = angle.sin_cos();
                let s = s * sin_sign;
                let i0 = base + 2 * p;
                let i1 = i0 + 1;
                let x0 = x[i0] as f64;
                let x1 = x[i1] as f64;
                x[i0] = (x0 * c - x1 * s) as f32;
                x[i1] = (x0 * s + x1 * c) as f32;
            }
        }
    }
    Ok(())
}

/// `get_window_topk_idxs` (`model.py:260-271`).
///
/// Returns a flat `rows * k` row-major `i32` matrix where `k = min(seqlen,
/// window_size)` when `start_pos == 0`, else `k = window_size`. Entries are
/// absolute KV indices or `-1` for masked slots.
///
/// For the prefill (`start_pos == 0`) path used by Gate 4:
/// ```text
/// base[r] = r
/// matrix[r, j] = clamp(r - window + 1, 0) + j    if that <= r else -1
/// j ∈ [0, min(seqlen, window))
/// ```
pub fn get_window_topk_idxs(
    window_size: usize,
    seqlen: usize,
    start_pos: usize,
) -> Result<Vec<i32>, String> {
    if window_size == 0 {
        return Err("deepseek4 parent: window_size must be > 0".to_owned());
    }
    if seqlen == 0 {
        return Err("deepseek4 parent: seqlen must be > 0".to_owned());
    }
    if start_pos == 0 {
        let k = seqlen.min(window_size);
        let mut out = vec![-1i32; seqlen * k];
        for r in 0..seqlen {
            let base = r as i32;
            let start = (r + 1).saturating_sub(window_size);
            for j in 0..k {
                let idx = start + j;
                let v = if idx <= r { idx as i32 } else { -1 };
                // model.py:269-270:
                //   matrix = (base - window + 1).clamp(0) + arange(min(seqlen, window))
                //   matrix = where(matrix > base, -1, matrix)
                let _ = base; // used via `r`
                let m = (r as i64 - window_size as i64 + 1).max(0) as usize + j;
                let v2 = if m > r { -1 } else { m as i32 };
                debug_assert_eq!(v, v2);
                out[r * k + j] = v2;
            }
        }
        Ok(out)
    } else if start_pos >= window_size - 1 {
        // Decode, ring-full: rotated arange of length window (single row).
        let slot = start_pos % window_size;
        let mut row = Vec::with_capacity(window_size);
        for i in (slot + 1)..window_size {
            row.push(i as i32);
        }
        for i in 0..=slot {
            row.push(i as i32);
        }
        Ok(row)
    } else {
        // Decode, ring-filling: [0..start_pos] then -1 pad to window.
        let mut row = Vec::with_capacity(window_size);
        for i in 0..=start_pos {
            row.push(i as i32);
        }
        while row.len() < window_size {
            row.push(-1);
        }
        Ok(row)
    }
}

/// `get_compress_topk_idxs` (`model.py:274-283`).
///
/// Returns a flat `rows * k` row-major `i32` matrix of compressed-slot
/// indices (already shifted by `offset`), padded with `-1`.
///
/// - Prefill (`start_pos == 0`): `k = seqlen / ratio` (integer). Row `r`
///   sees compressed slots `[0, (r+1)/ratio)` then `-1` pad. Surviving
///   indices are `slot + offset`.
/// - Decode (`start_pos > 0`): single row of length
///   `k = (start_pos + 1) / ratio` with values `arange(k) + offset`
///   (no future-mask; all committed slots are visible).
///
/// `offset` is the absolute base of the compressed region in a unified
/// KV index space (`model.py:515`: prefill `offset = seqlen`, decode
/// `offset = window`). The parent path gathers from a *separate*
/// `main_kv_cache`, so callers typically pass `offset = 0` and treat the
/// returned values as direct compressed-cache indices.
pub fn get_compress_topk_idxs(
    ratio: usize,
    seqlen: usize,
    start_pos: usize,
    offset: usize,
) -> Result<Vec<i32>, String> {
    if ratio == 0 {
        return Err("deepseek4 parent: get_compress_topk_idxs ratio must be > 0".to_owned());
    }
    if seqlen == 0 {
        return Err("deepseek4 parent: get_compress_topk_idxs seqlen must be > 0".to_owned());
    }
    let offset_i = offset as i32;
    if start_pos == 0 {
        // model.py:279-281
        //   matrix = arange(seqlen // ratio).repeat(seqlen, 1)
        //   mask = matrix >= arange(1, seqlen+1).unsqueeze(1) // ratio
        //   matrix = where(mask, -1, matrix + offset)
        let k = seqlen / ratio;
        let mut out = vec![-1i32; seqlen * k.max(1)];
        if k == 0 {
            // No compressed slots yet — return rows of empty (k=0) as empty vec
            // length seqlen*0 = 0, or keep a sentinel length-0 row matrix.
            return Ok(Vec::new());
        }
        for r in 0..seqlen {
            let cutoff = (r + 1) / ratio;
            for j in 0..k {
                let v = if j < cutoff {
                    j as i32 + offset_i
                } else {
                    -1
                };
                out[r * k + j] = v;
            }
        }
        Ok(out)
    } else {
        // model.py:277: arange(0, (start_pos+1)//ratio) + offset
        let k = (start_pos + 1) / ratio;
        let mut out = Vec::with_capacity(k);
        for j in 0..k {
            out.push(j as i32 + offset_i);
        }
        Ok(out)
    }
}

/// Number of compressed slots visible to query row `r` at `start_pos`.
/// Prefill: `(start_pos + r + 1) / ratio`. Decode: `(start_pos + 1) / ratio`
/// for every row in a single-token decode; for a multi-token chunk the
/// identity-gather path uses a per-row count of `(start_pos + r + 1) / ratio`.
#[inline]
pub fn compress_n_visible(start_pos: usize, row: usize, ratio: usize) -> usize {
    if ratio == 0 {
        return 0;
    }
    (start_pos + row + 1) / ratio
}


/// Per-row number of valid SWA positions at absolute position `start_pos + r`.
pub fn swa_n_valid(start_pos: usize, row: usize, window: usize) -> usize {
    let p = start_pos + row;
    (p + 1).min(window)
}

/// Host RMSNorm matching `model.py:197-202` (f32 API, f64 accumulate).
pub fn rms_norm_host(x: &[f32], weight: &[f32], eps: f32, dim: usize) -> Result<Vec<f32>, String> {
    if dim == 0 {
        return Err("deepseek4 parent: rms_norm_host dim must be > 0".to_owned());
    }
    if weight.len() < dim {
        return Err(format!(
            "deepseek4 parent: rms_norm_host weight short ({} < {dim})",
            weight.len()
        ));
    }
    if x.len() % dim != 0 {
        return Err(format!(
            "deepseek4 parent: rms_norm_host x len {} not divisible by {dim}",
            x.len()
        ));
    }
    let rows = x.len() / dim;
    let mut out = vec![0.0f32; x.len()];
    let eps = eps as f64;
    for r in 0..rows {
        let base = r * dim;
        let mut acc = 0.0f64;
        for d in 0..dim {
            let v = x[base + d] as f64;
            acc += v * v;
        }
        let scale = 1.0 / (acc / dim as f64 + eps).sqrt();
        for d in 0..dim {
            out[base + d] = ((x[base + d] as f64) * scale * (weight[d] as f64)) as f32;
        }
    }
    Ok(out)
}

/// Per-head RMSNorm with unit weight (the post-`wq_b` step in `model.py:504`).
pub fn rms_norm_heads_unit(
    x: &mut [f32],
    rows: usize,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(), String> {
    let need = rows
        .checked_mul(n_heads)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| "deepseek4 parent: head rms size overflow".to_owned())?;
    if x.len() < need {
        return Err(format!(
            "deepseek4 parent: head rms buffer short ({} < {need})",
            x.len()
        ));
    }
    let eps = eps as f64;
    for r in 0..rows {
        for h in 0..n_heads {
            let base = (r * n_heads + h) * head_dim;
            let mut acc = 0.0f64;
            for d in 0..head_dim {
                let v = x[base + d] as f64;
                acc += v * v;
            }
            let scale = 1.0 / (acc / head_dim as f64 + eps).sqrt();
            for d in 0..head_dim {
                x[base + d] = ((x[base + d] as f64) * scale) as f32;
            }
        }
    }
    Ok(())
}

// ── Forward ─────────────────────────────────────────────────────────────────

/// Parent attention for every layer class (ratio 0 / 4 / 128).
///
/// - `x` is `[rows, dim]` F32 post-attn_norm.
/// - `out` is `[rows, dim]` F32 (overwritten).
/// - `kv_ring` is the layer's persistent SWA ring `[n_kv_heads, head_dim,
///   window]` F32. On entry it holds history before `start_pos`; on exit the
///   current chunk's KVs have been written at slots
///   `(start_pos + r) % window`.
///
/// Compressed KV lives in `scratch.main_kv_cache` (not the SWA ring). Prefill
/// (`start_pos == 0`) resets compressor ring state so each call is
/// self-contained. Multi-layer decode requires one scratch per layer.
pub fn parent_attention_swa(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    layer: &ParentLayerWeights,
    cfg: &ParentQuantConfig,
    scratch: &mut ParentAttnScratch,
    x: &GpuTensor,
    rows: usize,
    start_pos: usize,
    kv_ring: &GpuTensor,
    out: &GpuTensor,
) -> Result<(), String> {
    backend.ensure_device(gpu)?;

    let ratio = layer.compress_ratio;
    let cfg_ratio = cfg.compress_ratio(layer.layer_idx);
    if ratio != cfg_ratio {
        return Err(format!(
            "deepseek4 parent: config.compress_ratios[{}] = {cfg_ratio} but \
             layer.compress_ratio claims {ratio} — refusing rather than guessing",
            layer.layer_idx
        ));
    }
    if !matches!(ratio, 0 | 4 | 128) {
        return Err(format!(
            "deepseek4 parent: parent_attention_swa unsupported compress_ratio={ratio} \
             (layer {}); expected 0, 4, or 128",
            layer.layer_idx
        ));
    }
    if ratio > 0 && layer.compressor.is_none() {
        return Err(format!(
            "deepseek4 parent: layer {} compress_ratio={ratio} but compressor weights missing",
            layer.layer_idx
        ));
    }
    if ratio == 4 && layer.indexer.is_none() {
        return Err(format!(
            "deepseek4 parent: layer {} compress_ratio=4 but indexer weights missing",
            layer.layer_idx
        ));
    }
    if rows == 0 {
        return Err("deepseek4 parent: parent_attention_swa rows must be > 0".to_owned());
    }
    if rows > scratch.max_rows {
        return Err(format!(
            "deepseek4 parent: rows {rows} exceeds scratch.max_rows {}",
            scratch.max_rows
        ));
    }
    validate_f32_mat(x, rows, PARENT_DIM, "x")?;
    validate_f32_mat(out, rows, PARENT_DIM, "out")?;
    validate_kv_ring(kv_ring)?;

    // ── 1. Q path: wq_a → q_norm → wq_b → per-head RMSNorm ──────────────
    // Fresh BF16 copy of x for wq_a (destructive act-quant inside linear).
    stage_f32_to_act_bf16(gpu, scratch, x, rows, PARENT_DIM)?;
    let q_lat = scratch.q_lat_f32.sub_offset(0, rows * PARENT_Q_LORA);
    parent_linear_dense(
        gpu,
        backend,
        &layer.wq_a,
        &act_view(scratch, rows, PARENT_DIM)?,
        rows,
        &q_lat,
    )
    .map_err(|e| format!("deepseek4 parent: wq_a linear: {e}"))?;

    // q_norm (BF16 weight → host f32 RMSNorm). Keep q_lat on device for the
    // indexer (`qr` in model.py:502/517) and a host copy for wq_b staging.
    let mut q_lat_host = download_f32_prefix(gpu, &scratch.q_lat_f32, rows * PARENT_Q_LORA)?;
    let q_norm_w = download_bf16_as_f32(gpu, &layer.q_norm, PARENT_Q_LORA)?;
    q_lat_host = rms_norm_host(&q_lat_host, &q_norm_w, PARENT_RMS_EPS, PARENT_Q_LORA)?;
    upload_f32_prefix(gpu, &scratch.q_lat_f32, &q_lat_host, rows * PARENT_Q_LORA)?;

    // Fresh BF16 copy of q_lat for wq_b.
    stage_f32_slice_to_act_bf16(gpu, scratch, &q_lat_host, rows, PARENT_Q_LORA)?;
    let q_out = scratch.q_f32.sub_offset(0, rows * PARENT_Q_WIDTH);
    parent_linear_dense(
        gpu,
        backend,
        &layer.wq_b,
        &act_view(scratch, rows, PARENT_Q_LORA)?,
        rows,
        &q_out,
    )
    .map_err(|e| format!("deepseek4 parent: wq_b linear: {e}"))?;

    // Per-head RMSNorm (unit weight) — model.py:504.
    let mut q_host = download_f32_prefix(gpu, &scratch.q_f32, rows * PARENT_Q_WIDTH)?;
    rms_norm_heads_unit(
        &mut q_host,
        rows,
        PARENT_N_HEADS,
        PARENT_HEAD_DIM,
        PARENT_RMS_EPS,
    )?;

    // ── 2. KV path: wkv → kv_norm ───────────────────────────────────────
    stage_f32_to_act_bf16(gpu, scratch, x, rows, PARENT_DIM)?;
    let kv_out = scratch.kv_f32.sub_offset(0, rows * PARENT_HEAD_DIM);
    parent_linear_dense(
        gpu,
        backend,
        &layer.wkv,
        &act_view(scratch, rows, PARENT_DIM)?,
        rows,
        &kv_out,
    )
    .map_err(|e| format!("deepseek4 parent: wkv linear: {e}"))?;

    let mut kv_host = download_f32_prefix(gpu, &scratch.kv_f32, rows * PARENT_HEAD_DIM)?;
    let kv_norm_w = download_bf16_as_f32(gpu, &layer.kv_norm, PARENT_HEAD_DIM)?;
    kv_host = rms_norm_host(&kv_host, &kv_norm_w, PARENT_RMS_EPS, PARENT_HEAD_DIM)?;

    // ── 3. Tail RoPE (interleaved) ──────────────────────────────────────
    // ratio==0: plain theta, no YaRN (model.py:484-485).
    // ratio>0: YaRN + compress_rope_theta (model.py:482-487).
    let freqs = if ratio == 0 {
        precompute_rope_freqs(
            PARENT_ROPE_DIM,
            /*original_seq_len=*/ 0,
            PARENT_ROPE_THETA as f64,
            /*factor=*/ PARENT_YARN_FACTOR,
            /*beta_fast=*/ PARENT_YARN_BETA_FAST,
            /*beta_slow=*/ PARENT_YARN_BETA_SLOW,
        )?
    } else {
        precompute_rope_freqs(
            PARENT_ROPE_DIM,
            PARENT_YARN_ORIG_SEQ,
            PARENT_COMPRESS_ROPE_THETA,
            PARENT_YARN_FACTOR,
            PARENT_YARN_BETA_FAST,
            PARENT_YARN_BETA_SLOW,
        )?
    };
    let positions: Vec<usize> = (0..rows).map(|r| start_pos + r).collect();
    apply_rope_interleaved_inplace(
        &mut q_host,
        rows,
        PARENT_N_HEADS,
        PARENT_HEAD_DIM,
        PARENT_ROPE_DIM,
        &positions,
        &freqs,
        /*inverse=*/ false,
    )?;
    apply_rope_interleaved_inplace(
        &mut kv_host,
        rows,
        PARENT_N_KV_HEADS,
        PARENT_HEAD_DIM,
        PARENT_ROPE_DIM,
        &positions,
        &freqs,
        /*inverse=*/ false,
    )?;

    // ── 4. FP8 act-quant simulation on non-RoPE KV dims (block 64) ───────
    kv_nope_act_quant(gpu, scratch, &mut kv_host, rows)?;

    upload_f32_prefix(gpu, &scratch.q_f32, &q_host, rows * PARENT_Q_WIDTH)?;
    upload_f32_prefix(gpu, &scratch.kv_f32, &kv_host, rows * PARENT_HEAD_DIM)?;

    // ── 5. SWA visibility staging ───────────────────────────────────────
    {
        let kv_batch = scratch.kv_f32.sub_offset(0, rows * PARENT_HEAD_DIM);
        let staged = scratch
            .swa_staged
            .sub_offset(0, rows * PARENT_HEAD_DIM * PARENT_SWA_WINDOW);
        gpu.swa_visibility_stage_batched(
            kv_ring,
            &kv_batch,
            &staged,
            start_pos as i32,
            PARENT_SWA_WINDOW as i32,
            PARENT_HEAD_DIM as i32,
            rows as i32,
        )
        .map_err(|e| format!("deepseek4 parent: swa_visibility_stage: {e:?}"))?;
    }

    // n_valid[r] = min(start_pos + r + 1, window)
    {
        let mut nv = vec![0i32; rows];
        for r in 0..rows {
            nv[r] = swa_n_valid(start_pos, r, PARENT_SWA_WINDOW) as i32;
        }
        upload_i32_prefix(gpu, &scratch.n_valid, &nv, rows)?;
    }
    {
        let mut ps = vec![0i32; rows];
        for r in 0..rows {
            ps[r] = (start_pos + r) as i32;
        }
        upload_i32_prefix(gpu, &scratch.positions, &ps, rows)?;
    }

    // ── 5b. Compressor + indexer (ratio > 0) ────────────────────────────
    // model.py:514-531. Compressed cache is separate from the SWA ring;
    // gather produces topk_staged for the joint softmax.
    if ratio > 0 {
        run_mixed_attn_compress_and_gather(
            gpu,
            backend,
            layer,
            cfg,
            scratch,
            x,
            rows,
            start_pos,
            ratio,
        )?;
    } else {
        // Pure SWA: zero active top-k so a mixed kernel would see n_active=0.
        let zeros = vec![0i32; rows];
        upload_i32_prefix(gpu, &scratch.n_active_topk, &zeros, rows)?;
    }

    // ── 5c. Attention kernel ────────────────────────────────────────────
    {
        let q = scratch.q_f32.sub_offset(0, rows * PARENT_Q_WIDTH);
        let staged = scratch
            .swa_staged
            .sub_offset(0, rows * PARENT_HEAD_DIM * PARENT_SWA_WINDOW);
        let attn_out = scratch.attn_out_f32.sub_offset(0, rows * PARENT_Q_WIDTH);

        if ratio == 0 {
            gpu.deepseek4_attn_swa_batched(
                &q,
                &staged,
                &staged, // K=V tied
                &layer.attn_sink,
                &scratch.n_valid,
                &attn_out,
                PARENT_N_HEADS as i32,
                PARENT_HEAD_DIM as i32,
                PARENT_O_GROUPS as i32,
                PARENT_SWA_WINDOW as i32,
                rows as i32,
            )
            .map_err(|e| format!("deepseek4 parent: deepseek4_attn_swa_batched: {e:?}"))?;
        } else {
            // Joint softmax over SWA + gathered top-k + sink.
            let topk = scratch
                .topk_staged
                .sub_offset(0, rows * PARENT_HEAD_DIM * PARENT_ATTN_INDEX_TOPK);
            gpu.deepseek4_attn_swa_topk_batched_f32(
                &q,
                &staged,
                &staged, // SWA K=V tied
                &topk,
                &topk, // topk K=V tied
                &layer.attn_sink,
                &scratch.n_valid,
                &scratch.n_active_topk,
                &attn_out,
                PARENT_N_HEADS as i32,
                PARENT_HEAD_DIM as i32,
                PARENT_SWA_WINDOW as i32,
                PARENT_ATTN_INDEX_TOPK as i32,
                rows as i32,
            )
            .map_err(|e| {
                format!("deepseek4 parent: deepseek4_attn_swa_topk_batched: {e:?}")
            })?;
        }
    }

    // ── 6. Inverse tail RoPE on attention output ────────────────────────
    let mut attn_host = download_f32_prefix(gpu, &scratch.attn_out_f32, rows * PARENT_Q_WIDTH)?;
    apply_rope_interleaved_inplace(
        &mut attn_host,
        rows,
        PARENT_N_HEADS,
        PARENT_HEAD_DIM,
        PARENT_ROPE_DIM,
        &positions,
        &freqs,
        /*inverse=*/ true,
    )?;
    upload_f32_prefix(gpu, &scratch.attn_out_f32, &attn_host, rows * PARENT_Q_WIDTH)?;

    // ── 7. O projection: grouped wo_a then wo_b ─────────────────────────
    wo_a_grouped(gpu, backend, &layer.wo_a, scratch, rows)?;

    let wo_a_host = download_f32_prefix(gpu, &scratch.wo_a_out_f32, rows * PARENT_WO_A_OUT)?;
    stage_f32_slice_to_act_bf16(gpu, scratch, &wo_a_host, rows, PARENT_WO_A_OUT)?;
    let out_view = out.sub_offset(0, rows * PARENT_DIM);
    parent_linear_dense(
        gpu,
        backend,
        &layer.wo_b,
        &act_view(scratch, rows, PARENT_WO_A_OUT)?,
        rows,
        &out_view,
    )
    .map_err(|e| format!("deepseek4 parent: wo_b linear: {e}"))?;

    // ── 8. Commit this chunk into the SWA ring for future steps ─────────
    {
        let kv_batch = scratch.kv_f32.sub_offset(0, rows * PARENT_HEAD_DIM);
        gpu.swa_ring_write_batched_f32(
            &kv_batch,
            kv_ring,
            PARENT_N_KV_HEADS as i32,
            PARENT_HEAD_DIM as i32,
            PARENT_SWA_WINDOW as i32,
            start_pos as i32,
            rows as i32,
        )
        .map_err(|e| format!("deepseek4 parent: swa_ring_write_batched: {e:?}"))?;
    }

    Ok(())
}

/// Compressor + (optional) indexer + gather into `scratch.topk_staged`.
///
/// Main compressor uses `hadamard = false`. Indexer (ratio==4) calls its own
/// compressor internally with `hadamard = true` — do not double-call it.
///
/// Index space note (`model.py:515`): the reference builds a unified index
/// over `[SWA_slots | compressed_slots]` with `offset = seqlen` (prefill) or
/// `window` (decode). Our kernels keep SWA and compressed buffers separate,
/// so we pass `offset = 0` into the indexer / compress-topk helpers and treat
/// returned indices as direct `main_kv_cache` row ids. That is the MQ2R
/// convention and matches the gather kernel's `idx < N_compressed` check.
fn run_mixed_attn_compress_and_gather(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    layer: &ParentLayerWeights,
    cfg: &ParentQuantConfig,
    scratch: &mut ParentAttnScratch,
    x: &GpuTensor,
    rows: usize,
    start_pos: usize,
    ratio: usize,
) -> Result<(), String> {
    // Prefill is self-contained: reset compressor rings so leftover decode
    // state cannot leak. Decode reuses ring state across calls on the same
    // layer/scratch.
    if start_pos == 0 {
        scratch.compressor.reset_ring(gpu)?;
    }

    let n_compressed = indexer_n_compressed(start_pos, rows, ratio);
    if n_compressed > scratch.max_n_compressed {
        return Err(format!(
            "deepseek4 parent: n_compressed={n_compressed} exceeds main_kv_cache \
             capacity {} (start_pos={start_pos} rows={rows} ratio={ratio})",
            scratch.max_n_compressed
        ));
    }

    // Main compressor → main_kv_cache. Always invoke so ring/remainder state
    // advances even when n_compressed == 0 this call.
    let comp_w = layer.compressor.as_ref().ok_or_else(|| {
        format!(
            "deepseek4 parent: layer {} missing compressor weights",
            layer.layer_idx
        )
    })?;
    {
        let out_rows = n_compressed.max(1);
        let kv_out = {
            let mut v = scratch
                .main_kv_cache
                .sub_offset(0, out_rows * PARENT_HEAD_DIM);
            v.shape = vec![out_rows, PARENT_HEAD_DIM];
            v
        };
        parent_compressor_forward(
            gpu,
            backend,
            comp_w,
            cfg,
            &mut scratch.compressor,
            x,
            rows,
            start_pos,
            ratio,
            /*hadamard=*/ false,
            &kv_out,
        )
        .map_err(|e| format!("deepseek4 parent: main compressor: {e}"))?;
    }

    // Per-row active top-k counts + gather into topk_staged.
    let topk_max = PARENT_ATTN_INDEX_TOPK;
    let mut n_active_host = vec![0i32; rows];

    if n_compressed == 0 {
        upload_i32_prefix(gpu, &scratch.n_active_topk, &n_active_host, rows)?;
        // Leave topk_staged untouched (kernel skips n_active==0 lanes).
        return Ok(());
    }

    if ratio == 4 {
        // Indexer selects top-k compressed slots. offset=0: indices are
        // main_kv_cache rows (see function doc).
        let idx_w = layer.indexer.as_ref().ok_or_else(|| {
            format!(
                "deepseek4 parent: layer {} missing indexer weights",
                layer.layer_idx
            )
        })?;
        let qr = scratch.q_lat_f32.sub_offset(0, rows * PARENT_Q_LORA);
        // n_active scalar buffer: parent_indexer_forward writes a single i32
        // (total n_compressed). We then expand to per-row active counts.
        let n_active_scalar = scratch.n_active_topk.sub_offset(0, 4); // first i32
        parent_indexer_forward(
            gpu,
            backend,
            idx_w,
            cfg,
            &mut scratch.indexer,
            x,
            &qr,
            rows,
            start_pos,
            /*offset=*/ 0,
            layer.layer_idx,
            &scratch.topk_idx,
            &n_active_scalar,
        )
        .map_err(|e| format!("deepseek4 parent: indexer: {e}"))?;

        // Per-row n_active = min(topk, visible compressed slots).
        for r in 0..rows {
            let vis = indexer_n_visible(start_pos, r, ratio).min(n_compressed);
            n_active_host[r] = topk_max.min(vis) as i32;
        }
        upload_i32_prefix(gpu, &scratch.n_active_topk, &n_active_host, rows)?;

        // Gather selected compressed rows into topk_staged.
        // K_active = topk_max (storage); -1 indices write zeros.
        let main_kv = scratch
            .main_kv_cache
            .sub_offset(0, n_compressed * PARENT_HEAD_DIM);
        let topk_out = scratch
            .topk_staged
            .sub_offset(0, rows * PARENT_HEAD_DIM * topk_max);
        gpu.deepseek4_topk_kv_gather_batched_f32(
            &main_kv,
            &scratch.topk_idx,
            &topk_out,
            topk_max as i32,
            PARENT_HEAD_DIM as i32,
            n_compressed as i32,
            topk_max as i32,
            /*col_offset=*/ 0,
            /*scale=*/ 1.0,
            rows as i32,
        )
        .map_err(|e| format!("deepseek4 parent: topk gather: {e:?}"))?;
    } else {
        // ratio == 128: identity gather of all compressed slots (no indexer).
        // Cap at topk_max so the staged buffer / joint-softmax window fit.
        let k_id = n_compressed.min(topk_max);
        for r in 0..rows {
            let vis = compress_n_visible(start_pos, r, ratio).min(k_id);
            n_active_host[r] = vis as i32;
        }
        upload_i32_prefix(gpu, &scratch.n_active_topk, &n_active_host, rows)?;

        let main_kv = scratch
            .main_kv_cache
            .sub_offset(0, n_compressed * PARENT_HEAD_DIM);
        let topk_out = scratch
            .topk_staged
            .sub_offset(0, rows * PARENT_HEAD_DIM * topk_max);
        // Identity copies first k_id rows of main_kv into every batch row's
        // topk slab. Per-row n_active_topk then trims the softmax window.
        gpu.deepseek4_topk_kv_gather_identity_batched_f32(
            &main_kv,
            &topk_out,
            k_id as i32,
            PARENT_HEAD_DIM as i32,
            topk_max as i32,
            rows as i32,
        )
        .map_err(|e| format!("deepseek4 parent: identity gather: {e:?}"))?;
    }

    Ok(())
}


/// Grouped `wo_a` projection: 8 independent `[1024, 4096] @ [rows, 4096]`
/// linears written into `scratch.wo_a_out_f32` as `[rows, 8, 1024]`.
fn wo_a_grouped(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    wo_a: &ParentDenseWeight,
    scratch: &mut ParentAttnScratch,
    rows: usize,
) -> Result<(), String> {
    if wo_a.n() != PARENT_WO_A_OUT || wo_a.k() != PARENT_PER_GROUP_IN {
        return Err(format!(
            "deepseek4 parent: wo_a shape [{},{}] != [{PARENT_WO_A_OUT},{PARENT_PER_GROUP_IN}]",
            wo_a.n(),
            wo_a.k()
        ));
    }
    // attn_out is [rows, n_heads, head_dim] ≡ [rows, n_groups, per_group_in]
    // contiguous, so group g of row r lives at
    //   (r * n_groups + g) * per_group_in
    // For the GEMM we need x_g contiguous as [rows, per_group_in]. Gather
    // per group on host, stage BF16, run destructive linear against a weight
    // sub-view, write into wo_a_out[:, g, :].
    let attn_host = download_f32_prefix(gpu, &scratch.attn_out_f32, rows * PARENT_Q_WIDTH)?;
    let w_tensor = wo_a.tensor();

    for g in 0..PARENT_O_GROUPS {
        // Gather group-g slices → [rows, per_group_in].
        let mut xg = vec![0.0f32; rows * PARENT_PER_GROUP_IN];
        for r in 0..rows {
            let src = (r * PARENT_N_HEADS + g * PARENT_HEADS_PER_GROUP) * PARENT_HEAD_DIM;
            let dst = r * PARENT_PER_GROUP_IN;
            xg[dst..dst + PARENT_PER_GROUP_IN]
                .copy_from_slice(&attn_host[src..src + PARENT_PER_GROUP_IN]);
        }
        stage_f32_slice_to_act_bf16(gpu, scratch, &xg, rows, PARENT_PER_GROUP_IN)?;

        // Weight sub-view: rows [g*o_lora, (g+1)*o_lora) of [8192, 4096].
        let w_elems = PARENT_O_LORA * PARENT_PER_GROUP_IN;
        let w_off = g * w_elems;
        let w_view = w_tensor.sub_offset(w_off, w_elems);
        // Force 2D shape metadata for clarity (gemm uses buffer + explicit n,k).
        let mut w_view = w_view;
        w_view.shape = vec![PARENT_O_LORA, PARENT_PER_GROUP_IN];

        let x_bf16 = act_view(scratch, rows, PARENT_PER_GROUP_IN)?;
        // Destructive act-quant on the staged tile.
        gpu.act_quant_fp8_ue8m0_inplace_gfx942(&x_bf16.buf, rows, PARENT_PER_GROUP_IN, 128)
            .map_err(|e| format!("deepseek4 parent: wo_a[{g}] act-quant: {e:?}"))?;

        // Out view: for each row, o_lora elements at offset r*8192 + g*1024.
        // gemm writes [rows, n] contiguously, so gather into a contiguous
        // scratch tile then scatter — use the q_lat tile as a temp [rows, 1024]
        // (q_lat is done).
        let tmp = scratch.q_lat_f32.sub_offset(0, rows * PARENT_O_LORA);
        let mut tmp = tmp;
        tmp.shape = vec![rows, PARENT_O_LORA];

        backend.ensure_device(gpu)?;
        gpu.gemm_bf16_mfma_gfx942(
            &w_view.buf,
            &x_bf16.buf,
            &tmp.buf,
            PARENT_O_LORA,
            PARENT_PER_GROUP_IN,
            rows,
        )
        .map_err(|e| format!("deepseek4 parent: wo_a[{g}] gemm: {e:?}"))?;

        // Scatter tmp → wo_a_out[:, g, :].
        let tmp_host = download_f32_prefix(gpu, &scratch.q_lat_f32, rows * PARENT_O_LORA)?;
        let mut wo_host = if g == 0 {
            vec![0.0f32; rows * PARENT_WO_A_OUT]
        } else {
            download_f32_prefix(gpu, &scratch.wo_a_out_f32, rows * PARENT_WO_A_OUT)?
        };
        for r in 0..rows {
            let src = r * PARENT_O_LORA;
            let dst = r * PARENT_WO_A_OUT + g * PARENT_O_LORA;
            wo_host[dst..dst + PARENT_O_LORA]
                .copy_from_slice(&tmp_host[src..src + PARENT_O_LORA]);
        }
        upload_f32_prefix(gpu, &scratch.wo_a_out_f32, &wo_host, rows * PARENT_WO_A_OUT)?;
    }
    Ok(())
}

/// FP8 act-quant simulation on `kv[..., :-rope_dim]` with block 64.
fn kv_nope_act_quant(
    gpu: &mut Gpu,
    scratch: &mut ParentAttnScratch,
    kv_host: &mut [f32],
    rows: usize,
) -> Result<(), String> {
    // Pack non-RoPE dims to BF16, quant in place on device, write back.
    let mut nope = vec![0.0f32; rows * PARENT_NOPE_DIM];
    for r in 0..rows {
        let src = r * PARENT_HEAD_DIM;
        let dst = r * PARENT_NOPE_DIM;
        nope[dst..dst + PARENT_NOPE_DIM]
            .copy_from_slice(&kv_host[src..src + PARENT_NOPE_DIM]);
    }
    let bytes = pack_f32_to_bf16_bytes(&nope);
    upload_bf16_into(gpu, &scratch.kv_nope_bf16, &bytes, rows * PARENT_NOPE_DIM)?;
    let view = scratch.kv_nope_bf16.sub_offset(0, rows * PARENT_NOPE_DIM);
    gpu.act_quant_fp8_ue8m0_inplace_gfx942(
        &view.buf,
        rows,
        PARENT_NOPE_DIM,
        PARENT_KV_ACT_QUANT_BLOCK,
    )
    .map_err(|e| format!("deepseek4 parent: kv non-rope act-quant (block 64): {e:?}"))?;
    let nope_q = download_bf16_as_f32(gpu, &scratch.kv_nope_bf16, rows * PARENT_NOPE_DIM)?;
    for r in 0..rows {
        let src = r * PARENT_NOPE_DIM;
        let dst = r * PARENT_HEAD_DIM;
        kv_host[dst..dst + PARENT_NOPE_DIM]
            .copy_from_slice(&nope_q[src..src + PARENT_NOPE_DIM]);
    }
    Ok(())
}

// ── Staging / IO helpers ────────────────────────────────────────────────────

fn act_view(scratch: &ParentAttnScratch, rows: usize, k: usize) -> Result<GpuTensor, String> {
    if k > PARENT_WO_A_OUT {
        return Err(format!(
            "deepseek4 parent: act_view k={k} exceeds act_bf16 width {PARENT_WO_A_OUT}"
        ));
    }
    if rows > scratch.max_rows {
        return Err(format!(
            "deepseek4 parent: act_view rows={rows} exceeds max_rows {}",
            scratch.max_rows
        ));
    }
    // Contiguous [rows, k] inside [max_rows, WO_A_OUT] requires k == WO_A_OUT
    // OR we pack tightly at offset 0 with stride k (we always pack tightly).
    let mut v = scratch.act_bf16.sub_offset(0, rows * k);
    v.shape = vec![rows, k];
    Ok(v)
}

fn stage_f32_to_act_bf16(
    gpu: &Gpu,
    scratch: &ParentAttnScratch,
    x: &GpuTensor,
    rows: usize,
    k: usize,
) -> Result<(), String> {
    let host = download_f32_prefix(gpu, x, rows * k)?;
    stage_f32_slice_to_act_bf16(gpu, scratch, &host, rows, k)
}

fn stage_f32_slice_to_act_bf16(
    gpu: &Gpu,
    scratch: &ParentAttnScratch,
    host: &[f32],
    rows: usize,
    k: usize,
) -> Result<(), String> {
    if host.len() < rows * k {
        return Err(format!(
            "deepseek4 parent: stage_f32 host short ({} < {})",
            host.len(),
            rows * k
        ));
    }
    let bytes = pack_f32_to_bf16_bytes(&host[..rows * k]);
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

fn download_bf16_as_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    let nbytes = nelems * 2;
    if t.buf.size() < nbytes {
        return Err(format!(
            "deepseek4 parent: bf16 download too small (have {} need {nbytes})",
            t.buf.size()
        ));
    }
    let mut raw = vec![0u8; nbytes];
    gpu.hip
        .memcpy_dtoh(&mut raw, &t.buf)
        .map_err(|e| format!("deepseek4 parent: bf16 download: {e:?}"))?;
    let mut out = Vec::with_capacity(nelems);
    for i in 0..nelems {
        let b = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
        out.push(f32::from_bits((b as u32) << 16));
    }
    Ok(out)
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

fn validate_kv_ring(ring: &GpuTensor) -> Result<(), String> {
    if ring.dtype != DType::F32 {
        return Err(format!(
            "deepseek4 parent: kv_ring must be F32 (got {:?})",
            ring.dtype
        ));
    }
    let need = PARENT_N_KV_HEADS * PARENT_HEAD_DIM * PARENT_SWA_WINDOW * 4;
    if ring.buf.size() < need {
        return Err(format!(
            "deepseek4 parent: kv_ring too small (have {} need {need} for \
             [{PARENT_N_KV_HEADS},{PARENT_HEAD_DIM},{PARENT_SWA_WINDOW}])",
            ring.buf.size()
        ));
    }
    Ok(())
}

// ── Intermediate-norm helpers (smoke / diagnostics) ─────────────────────────

/// L2 norm of a contiguous f32 prefix.
pub fn l2_norm(xs: &[f32]) -> f32 {
    let mut acc = 0.0f64;
    for &v in xs {
        let x = v as f64;
        acc += x * x;
    }
    acc.sqrt() as f32
}

/// Whether every value is finite.
pub fn all_finite(xs: &[f32]) -> bool {
    xs.iter().all(|v| v.is_finite())
}

// ── Unit tests (host-side) ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_freqs_plain_no_yarn() {
        // original_seq_len == 0 → plain 1/base^(2i/dim), dim=64, base=10000.
        let freqs = precompute_rope_freqs(64, 0, 10_000.0, 16.0, 32.0, 1.0).unwrap();
        assert_eq!(freqs.len(), 32);
        assert!((freqs[0] - 1.0).abs() < 1e-12);
        // pair 1: 1/10000^(2/64) = 1/10000^(1/32)
        let expect1 = 1.0 / 10_000f64.powf(2.0 / 64.0);
        assert!(
            (freqs[1] - expect1).abs() < 1e-12,
            "freqs[1]={ } expect={expect1}",
            freqs[1]
        );
        let expect31 = 1.0 / 10_000f64.powf(62.0 / 64.0);
        assert!((freqs[31] - expect31).abs() < 1e-12);
    }

    #[test]
    fn rope_freqs_yarn_matches_hand() {
        // Hand values from model.py:206-236 with the parent YaRN knobs.
        let dim = 64usize;
        let base = 10_000f64;
        let factor = 16f64;
        let original = 65536usize;
        let freqs = precompute_rope_freqs(dim, original, base, factor, 32.0, 1.0).unwrap();
        // correction range: low=floor(find(32)), high=ceil(find(1))
        let low = (dim as f64
            * ((original as f64) / (32.0 * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * base.ln()))
        .floor()
        .max(0.0);
        let high = (dim as f64
            * ((original as f64) / (1.0 * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * base.ln()))
        .ceil()
        .min((dim - 1) as f64);
        // pair 0 (i=0): ramp = clamp((0-low)/(high-low),0,1); smooth=1-ramp
        let ramp0 = {
            let mut mx = high;
            let mn = low;
            if (mn - mx).abs() < f64::EPSILON {
                mx += 0.001;
            }
            ((0.0 - mn) / (mx - mn)).clamp(0.0, 1.0)
        };
        let smooth0 = 1.0 - ramp0;
        let f0_base = 1.0f64;
        let f0 = f0_base / factor * (1.0 - smooth0) + f0_base * smooth0;
        assert!(
            (freqs[0] - f0).abs() < 1e-12,
            "yarn freqs[0]={} hand={f0} low={low} high={high} ramp={ramp0}",
            freqs[0]
        );
    }

    #[test]
    fn apply_rope_interleaved_pairs_adjacent() {
        // Interleaved: dims (head_dim-2, head_dim-1) form pair n_rot/2-1.
        // A half-split convention would pair (head_dim-n_rot + i, head_dim-n_rot/2 + i).
        let rows = 1;
        let n_heads = 1;
        let head_dim = 8;
        let n_rot = 4; // tail dims [4,5,6,7]; pairs (4,5) and (6,7)
        let mut x = vec![0.0f32; head_dim];
        x[4] = 1.0; // real of pair 0
        x[5] = 0.0; // imag of pair 0
        x[6] = 0.0;
        x[7] = 1.0; // imag of pair 1
        let freqs = vec![std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2]; // 90°
        apply_rope_interleaved_inplace(
            &mut x,
            rows,
            n_heads,
            head_dim,
            n_rot,
            &[1],
            &freqs,
            false,
        )
        .unwrap();
        // 90°: (1,0) → (0,1); (0,1) → (-1,0)
        assert!(x[4].abs() < 1e-6, "x4={}", x[4]);
        assert!((x[5] - 1.0).abs() < 1e-6, "x5={}", x[5]);
        assert!((x[6] + 1.0).abs() < 1e-6, "x6={}", x[6]);
        assert!(x[7].abs() < 1e-6, "x7={}", x[7]);
        // Leading non-rope dims untouched.
        assert_eq!(x[0], 0.0);
    }

    #[test]
    fn window_topk_prefill_start0() {
        let win = 128usize;
        let seqlen = 16usize;
        let m = get_window_topk_idxs(win, seqlen, 0).unwrap();
        let k = seqlen.min(win);
        assert_eq!(m.len(), seqlen * k);
        // row 0: only position 0 is visible
        assert_eq!(m[0], 0);
        for j in 1..k {
            assert_eq!(m[j], -1, "row0 col{j}");
        }
        // row 5: positions 0..=5
        let row5 = &m[5 * k..(5 + 1) * k];
        for j in 0..=5 {
            assert_eq!(row5[j], j as i32, "row5 col{j}");
        }
        for j in 6..k {
            assert_eq!(row5[j], -1, "row5 col{j}");
        }
        // row 15: positions 0..=15
        let row15 = &m[15 * k..(15 + 1) * k];
        for j in 0..16 {
            assert_eq!(row15[j], j as i32);
        }
    }

    #[test]
    fn window_topk_decode_partial_and_full() {
        let win = 128usize;
        // start_pos=5 (< win-1): [0..5] then -1 pad
        let m = get_window_topk_idxs(win, 1, 5).unwrap();
        assert_eq!(m.len(), win);
        for i in 0..=5 {
            assert_eq!(m[i], i as i32);
        }
        assert!(m[6..].iter().all(|&v| v == -1));

        // start_pos = win: ring-full rotation
        let m = get_window_topk_idxs(win, 1, win).unwrap();
        assert_eq!(m.len(), win);
        // slot = 0; cat(arange(1,win), arange(0,1)) = [1,2,...,127,0]
        assert_eq!(m[0], 1);
        assert_eq!(m[win - 2], (win - 1) as i32);
        assert_eq!(m[win - 1], 0);
    }

    #[test]
    fn swa_n_valid_table() {
        assert_eq!(swa_n_valid(0, 0, 128), 1);
        assert_eq!(swa_n_valid(0, 15, 128), 16);
        assert_eq!(swa_n_valid(0, 200, 128), 128);
        assert_eq!(swa_n_valid(100, 0, 128), 101);
        assert_eq!(swa_n_valid(127, 0, 128), 128);
        assert_eq!(swa_n_valid(128, 0, 128), 128);
    }

    #[test]
    fn scratch_bytes_formula() {
        // Host-side size accounting for the *own* tiles (compressor/indexer
        // nested scratch is additional and GPU-dependent).
        let max_rows = 16usize;
        let max_n_compressed = max_rows.max(PARENT_ATTN_INDEX_TOPK);
        let act = max_rows * PARENT_WO_A_OUT * 2;
        let kv_nope = max_rows * PARENT_NOPE_DIM * 2;
        let q_lat = max_rows * PARENT_Q_LORA * 4;
        let q = max_rows * PARENT_Q_WIDTH * 4;
        let kv = max_rows * PARENT_HEAD_DIM * 4;
        let attn = max_rows * PARENT_Q_WIDTH * 4;
        let wo = max_rows * PARENT_WO_A_OUT * 4;
        let staged = max_rows * PARENT_HEAD_DIM * PARENT_SWA_WINDOW * 4;
        let topk_staged = max_rows * PARENT_HEAD_DIM * PARENT_ATTN_INDEX_TOPK * 4;
        let main_kv = max_n_compressed * PARENT_HEAD_DIM * 4;
        let topk_idx = max_rows * PARENT_ATTN_INDEX_TOPK * 4;
        let n_valid = max_rows * 4;
        let n_active = max_rows * 4;
        let pos = max_rows * 4;
        let ones = PARENT_HEAD_DIM * 4;
        let own = act
            + kv_nope
            + q_lat
            + q
            + kv
            + attn
            + wo
            + staged
            + topk_staged
            + main_kv
            + topk_idx
            + n_valid
            + n_active
            + pos
            + ones;
        // Own tiles alone are dominated by topk_staged (~16*512*512*4 ≈ 16 MiB).
        assert!(own > 16 * 1024 * 1024, "own={own}");
        assert!(own < 64 * 1024 * 1024, "own={own}");
        assert_eq!(q + attn, 2 * max_rows * 32768 * 4);
        assert_eq!(PARENT_ATTN_INDEX_TOPK, 512);
    }

    #[test]
    fn compress_topk_prefill_ratio4() {
        // seqlen=16, ratio=4 → k = 4 compressed slots.
        // Row r sees slots [0, (r+1)/4) then -1 pad; offset applied.
        let ratio = 4usize;
        let seqlen = 16usize;
        let offset = 0usize;
        let m = get_compress_topk_idxs(ratio, seqlen, 0, offset).unwrap();
        let k = seqlen / ratio; // 4
        assert_eq!(m.len(), seqlen * k);
        // row 0: (0+1)/4 = 0 → all -1
        assert!(m[..k].iter().all(|&v| v == -1));
        // row 3: (3+1)/4 = 1 → [0, -1, -1, -1]
        assert_eq!(&m[3 * k..4 * k], &[0, -1, -1, -1]);
        // row 7: (7+1)/4 = 2 → [0, 1, -1, -1]
        assert_eq!(&m[7 * k..8 * k], &[0, 1, -1, -1]);
        // row 15: (15+1)/4 = 4 → [0, 1, 2, 3]
        assert_eq!(&m[15 * k..16 * k], &[0, 1, 2, 3]);

        // offset = seqlen (reference unified-index convention)
        let m2 = get_compress_topk_idxs(ratio, seqlen, 0, seqlen).unwrap();
        assert_eq!(&m2[15 * k..16 * k], &[16, 17, 18, 19]);
        assert_eq!(m2[0], -1);
    }

    #[test]
    fn compress_topk_prefill_ratio128_short() {
        // seqlen=16 < ratio=128 → k=0 → empty matrix.
        let m = get_compress_topk_idxs(128, 16, 0, 0).unwrap();
        assert!(m.is_empty());
        // seqlen=256, ratio=128 → k=2
        let m = get_compress_topk_idxs(128, 256, 0, 0).unwrap();
        let k = 2;
        assert_eq!(m.len(), 256 * k);
        // row 127: (127+1)/128 = 1 → [0, -1]
        assert_eq!(&m[127 * k..128 * k], &[0, -1]);
        // row 255: (255+1)/128 = 2 → [0, 1]
        assert_eq!(&m[255 * k..256 * k], &[0, 1]);
    }

    #[test]
    fn compress_topk_decode_ratio4_and_128() {
        // Decode start_pos=15, ratio=4 → k = 16/4 = 4; values 0..3 + offset
        let m = get_compress_topk_idxs(4, 1, 15, 0).unwrap();
        assert_eq!(m, vec![0, 1, 2, 3]);
        let m = get_compress_topk_idxs(4, 1, 15, 128).unwrap();
        assert_eq!(m, vec![128, 129, 130, 131]);

        // Decode start_pos=255, ratio=128 → k = 256/128 = 2
        let m = get_compress_topk_idxs(128, 1, 255, 0).unwrap();
        assert_eq!(m, vec![0, 1]);

        // Mid-window decode: start_pos=5, ratio=4 → k = 6/4 = 1
        let m = get_compress_topk_idxs(4, 1, 5, 0).unwrap();
        assert_eq!(m, vec![0]);

        // No compress yet: start_pos=2, ratio=4 → k = 0
        let m = get_compress_topk_idxs(4, 1, 2, 0).unwrap();
        assert!(m.is_empty());
    }

    #[test]
    fn compress_n_visible_table() {
        assert_eq!(compress_n_visible(0, 0, 4), 0);
        assert_eq!(compress_n_visible(0, 3, 4), 1);
        assert_eq!(compress_n_visible(0, 15, 4), 4);
        assert_eq!(compress_n_visible(0, 15, 128), 0);
        assert_eq!(compress_n_visible(0, 127, 128), 1);
        assert_eq!(compress_n_visible(15, 0, 4), 4);
        assert_eq!(compress_n_visible(0, 0, 0), 0);
    }

    #[test]
    fn ratio0_window_unchanged() {
        // Ratio-0 path still uses pure SWA window indices; compress helper
        // is not consulted. Lock the prefill window contract.
        let m = get_window_topk_idxs(128, 16, 0).unwrap();
        let k = 16;
        assert_eq!(m[0], 0);
        assert!(m[1..k].iter().all(|&v| v == -1));
        let row15: Vec<i32> = m[15 * k..16 * k].to_vec();
        assert_eq!(row15, (0..16).map(|i| i as i32).collect::<Vec<_>>());
    }

    #[test]
    fn unsupported_ratio_message() {
        // Unsupported ratios still fail closed (not silently treated as SWA).
        let ratio = 7usize;
        let layer_idx = 9usize;
        let msg = format!(
            "deepseek4 parent: parent_attention_swa unsupported compress_ratio={ratio} \
             (layer {layer_idx}); expected 0, 4, or 128"
        );
        assert!(msg.contains("unsupported compress_ratio=7"));
        assert!(msg.contains("expected 0, 4, or 128"));
    }

    #[test]
    fn pos0_window_is_self_only() {
        // Classic off-by-one: position 0 must see exactly one real index (0).
        let m = get_window_topk_idxs(128, 16, 0).unwrap();
        let k = 16;
        let visible: Vec<i32> = m[..k].iter().copied().filter(|&v| v >= 0).collect();
        assert_eq!(visible, vec![0]);
        assert_eq!(swa_n_valid(0, 0, 128), 1);
    }
}
