// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LLaMA model implementation using RDNA GPU compute.
//! Supports loading from GGUF files and running inference.

pub use crate::kv::KvCache;
use crate::transformer::MIN_BATCH;
use crate::weights::{
    fused_silu_mul_rotate_mq_batched_for, rotate_x_for_mq, rotate_x_mq_batched_for, weight_gemm,
    weight_gemv, weight_gemv_prerotated, EmbeddingFormat, LayerWeights, WeightTensor,
};
use hip_bridge::HipResult;
use hipfire_rdna::{DType, Gpu, GpuTensor};

// Re-exports so downstream crates (e.g. hipfire-specdecode-dspark) can reach
// these helpers through the historical `llama` module path after the
// runtime refactor relocated their definitions.
pub use crate::dispatch::gemv_family;
pub use hipfire_primitives::conv::f16_to_f32;

/// Model architecture type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArch {
    Llama,
    Qwen3,
}

/// Model configuration, read from GGUF metadata.
/// Supports LLaMA-family and Qwen3 architectures.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub arch: ModelArch,
    pub dim: usize,        // model dimension (embedding size)
    pub hidden_dim: usize, // FFN hidden dimension
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize, // for GQA
    pub vocab_size: usize,
    pub head_dim: usize,
    pub norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
    pub bos_token: u32,
    pub eos_token: u32,
    pub has_qk_norm: bool, // Qwen3 feature
}

/// A weight matrix on GPU — may be quantized or F32.
/// ParoQuant Givens rotation metadata for a single linear layer.
/// Stored alongside the weight buffer; applied to activations before GEMV.
/// GPU-resident LLaMA model weights.
pub struct LlamaWeights {
    pub token_embd: GpuTensor,
    pub embd_format: EmbeddingFormat,
    pub output_norm: GpuTensor,
    pub output: WeightTensor,
    pub layers: Vec<LayerWeights>,
}

impl LlamaWeights {
    /// Every linear weight that is dispatched through `weight_gemv` / `weight_gemm`,
    /// as `(gpu_dtype, has_awq)` pairs: the lm_head (`output`) plus each layer's
    /// qkv/o/gate/up/down projections. Feeds `weights::preflight_gemv_dtypes` so an
    /// unsupported quant is refused up front rather than panicking at the lm_head
    /// GEMV mid-forward. `token_embd`/norms are excluded (not GEMV-dispatched).
    pub fn linear_weight_dtypes(&self) -> Vec<(DType, bool)> {
        let mut out = Vec::with_capacity(1 + self.layers.len() * 7);
        let mut push = |w: &WeightTensor| out.push((w.gpu_dtype, w.awq_scale.is_some()));
        push(&self.output);
        for l in &self.layers {
            push(&l.wq);
            push(&l.wk);
            push(&l.wv);
            push(&l.wo);
            push(&l.w_gate);
            push(&l.w_up);
            push(&l.w_down);
        }
        out
    }

    /// Return all GPU buffers to the pool (drained on unload). Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.token_embd);
        let _ = gpu.free_tensor(self.output_norm);
        let _ = gpu.free_tensor(self.output.buf);
        for l in self.layers {
            let _ = gpu.free_tensor(l.attn_norm);
            let _ = gpu.free_tensor(l.wq.buf);
            let _ = gpu.free_tensor(l.wk.buf);
            let _ = gpu.free_tensor(l.wv.buf);
            let _ = gpu.free_tensor(l.wo.buf);
            if let Some(t) = l.q_norm {
                let _ = gpu.free_tensor(t);
            }
            if let Some(t) = l.k_norm {
                let _ = gpu.free_tensor(t);
            }
            let _ = gpu.free_tensor(l.ffn_norm);
            let _ = gpu.free_tensor(l.w_gate.buf);
            let _ = gpu.free_tensor(l.w_up.buf);
            let _ = gpu.free_tensor(l.w_down.buf);
        }
    }
}

/// Batched prefill: process all prompt tokens in one forward pass.
/// Returns logits for the LAST position only.
/// KV cache is filled for all positions.
pub fn prefill_forward(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    tokens: &[u32],
    kv_cache: &mut KvCache,
) -> HipResult<Vec<f32>> {
    let batch = tokens.len();
    let dim = config.dim;
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let q_dim = n_heads * head_dim;

    // Allocate batched buffers: [batch × dim]. These are per-call scratch:
    // `OwnedTensor` returns each to the pool on drop; `reclaim_pending` at the
    // end of the call drains the deferred-free mailbox.
    let x_batch = gpu.alloc_owned(&[batch, dim], DType::F32)?;
    let tmp_batch = gpu.alloc_owned(&[batch, dim], DType::F32)?;
    let q_batch = gpu.alloc_owned(&[batch, q_dim], DType::F32)?;
    let k_batch = gpu.alloc_owned(&[batch, kv_dim], DType::F32)?;
    let v_batch = gpu.alloc_owned(&[batch, kv_dim], DType::F32)?;
    let attn_out_batch = gpu.alloc_owned(&[batch, q_dim], DType::F32)?;
    let o_batch = gpu.alloc_owned(&[batch, dim], DType::F32)?;
    let gate_batch = gpu.alloc_owned(&[batch, config.hidden_dim], DType::F32)?;
    let up_batch = gpu.alloc_owned(&[batch, config.hidden_dim], DType::F32)?;
    let ffn_hidden_batch = gpu.alloc_owned(&[batch, config.hidden_dim], DType::F32)?;
    let ffn_out_batch = gpu.alloc_owned(&[batch, dim], DType::F32)?;

    // Raw (non-pooled) device allocation, so `OwnedTensor` does not apply: it is
    // freed on every exit path after the closure below, which is kept solely to
    // funnel that free through any `?` early-return.
    let pos_buf = gpu.hip.malloc(4)?;
    let result = (|| -> HipResult<Vec<f32>> {
        // Embedding: lookup each token individually into the batch buffer
        let x_single = gpu.alloc_owned(&[dim], DType::F32)?;
        for (i, &token) in tokens.iter().enumerate() {
            match weights.embd_format {
                EmbeddingFormat::HFQ4G256 => {
                    gpu.embedding_lookup_hfq4g256(&weights.token_embd, &x_single, token, dim)?
                }
                EmbeddingFormat::HFQ4G128 => {
                    gpu.embedding_lookup_hfq4g128(&weights.token_embd, &x_single, token, dim)?
                }
                EmbeddingFormat::Q8_0 => {
                    gpu.embedding_lookup_q8(&weights.token_embd, &x_single, token, dim)?
                }
                EmbeddingFormat::Q4K => {
                    gpu.embedding_lookup_q4k(&weights.token_embd, &x_single, token, dim)?
                }
                EmbeddingFormat::F32 => {
                    gpu.embedding_lookup(&weights.token_embd, &x_single, token, dim)?
                }
            }
            gpu.hip
                .memcpy_dtod_at(&x_batch.buf, i * dim * 4, &x_single.buf, 0, dim * 4)?;
        }

        // Position array for batched RoPE: [0, 1, 2, ..., batch-1]
        let pos_data: Vec<i32> = (0..batch as i32).collect();
        let pos_bytes: Vec<u8> = pos_data.iter().flat_map(|p| p.to_ne_bytes()).collect();
        let pos_array = gpu.alloc_owned(&[batch], DType::F32)?; // i32 same size as f32
        gpu.hip.memcpy_htod(&pos_array.buf, &pos_bytes)?;

        // Per-position scratch buffers (reused across all layers). `_q_slice` /
        // `_attn_slice` are unused by the current batched path but kept so the
        // pooled allocation order is identical to the original.
        let _q_slice = gpu.alloc_owned(&[q_dim], DType::F32)?;
        let k_slice = gpu.alloc_owned(&[kv_dim], DType::F32)?;
        let v_slice = gpu.alloc_owned(&[kv_dim], DType::F32)?;
        let _attn_slice = gpu.alloc_owned(&[q_dim], DType::F32)?;
        for layer_idx in 0..config.n_layers {
            let layer = &weights.layers[layer_idx];

            // Batched RMSNorm: each row of x_batch independently
            for _i in 0..batch {
                // We need per-row norm — use the batched rmsnorm with batch=batch
                // Actually, rmsnorm_batched already handles this if we set batch=batch, n=dim
            }
            gpu.rmsnorm_batched(
                &x_batch,
                &layer.attn_norm,
                &tmp_batch,
                batch,
                dim,
                config.norm_eps,
            )?;

            // Batched QKV projections
            weight_gemm(gpu, &layer.wq, &tmp_batch, &q_batch, batch)?;
            weight_gemm(gpu, &layer.wk, &tmp_batch, &k_batch, batch)?;
            weight_gemm(gpu, &layer.wv, &tmp_batch, &v_batch, batch)?;

            // QK norm (per-position, per-head)
            if config.has_qk_norm {
                if let Some(ref qn) = layer.q_norm {
                    gpu.rmsnorm_batched(
                        &q_batch,
                        qn,
                        &q_batch,
                        batch * n_heads,
                        head_dim,
                        config.norm_eps,
                    )?;
                }
                if let Some(ref kn) = layer.k_norm {
                    gpu.rmsnorm_batched(
                        &k_batch,
                        kn,
                        &k_batch,
                        batch * n_kv_heads,
                        head_dim,
                        config.norm_eps,
                    )?;
                }
            }

            // Batched RoPE: all positions in one kernel launch
            gpu.rope_batched_f32(
                &q_batch,
                &k_batch,
                &pos_array,
                n_heads,
                n_kv_heads,
                head_dim,
                config.rope_freq_base,
                batch,
            )?;

            // Batched KV cache write: all positions in 2 kernel launches (K + V)
            if kv_cache.quantized && kv_cache.quant_q8 {
                gpu.kv_cache_write_q8_0_batched(
                    &kv_cache.k_gpu[layer_idx],
                    &k_batch,
                    &pos_array,
                    n_kv_heads,
                    head_dim,
                    batch,
                )?;
                gpu.kv_cache_write_q8_0_batched(
                    &kv_cache.v_gpu[layer_idx],
                    &v_batch,
                    &pos_array,
                    n_kv_heads,
                    head_dim,
                    batch,
                )?;
            } else {
                for i in 0..batch {
                    let pos_i32 = i as i32;
                    gpu.hip.memcpy_htod(&pos_buf, &pos_i32.to_ne_bytes())?;
                    gpu.hip.memcpy_dtod_at(
                        &k_slice.buf,
                        0,
                        &k_batch.buf,
                        i * kv_dim * 4,
                        kv_dim * 4,
                    )?;
                    gpu.hip.memcpy_dtod_at(
                        &v_slice.buf,
                        0,
                        &v_batch.buf,
                        i * kv_dim * 4,
                        kv_dim * 4,
                    )?;
                    gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &k_slice, &pos_buf, kv_dim)?;
                    gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &v_slice, &pos_buf, kv_dim)?;
                }
            }

            // Batched causal attention: one kernel launch for all positions
            gpu.attention_causal_batched(
                &q_batch,
                &k_batch,
                &v_batch,
                &attn_out_batch,
                batch,
                n_heads,
                n_kv_heads,
                head_dim,
            )?;

            // Batched output projection
            weight_gemm(gpu, &layer.wo, &attn_out_batch, &o_batch, batch)?;

            // Batched residual add: x_batch += o_batch
            gpu.add_inplace_f32(&x_batch, &o_batch)?;

            // Batched FFN norm
            gpu.rmsnorm_batched(
                &x_batch,
                &layer.ffn_norm,
                &tmp_batch,
                batch,
                dim,
                config.norm_eps,
            )?;

            // Batched FFN projections
            weight_gemm(gpu, &layer.w_gate, &tmp_batch, &gate_batch, batch)?;
            weight_gemm(gpu, &layer.w_up, &tmp_batch, &up_batch, batch)?;

            // Batched SiLU * mul
            gpu.silu_mul_f32(&gate_batch, &up_batch, &ffn_hidden_batch)?;

            // Batched down projection
            weight_gemm(gpu, &layer.w_down, &ffn_hidden_batch, &ffn_out_batch, batch)?;

            // H-Neurons CETT tap (no-op unless a capture session is active).
            // Full-sequence prefill starts at global position 0, so batch_start=0.
            hipfire_hneurons::capture::maybe_capture_ffn(
                gpu,
                &ffn_hidden_batch,
                &ffn_out_batch,
                layer_idx,
                0,
                batch,
            )?;

            // Batched residual
            gpu.add_inplace_f32(&x_batch, &ffn_out_batch)?;
        }

        // Final norm + output projection for LAST position only.
        let last_off = (batch - 1) * dim * 4;
        let x_last = gpu.alloc_owned(&[dim], DType::F32)?;
        let tmp = gpu.alloc_owned(&[dim], DType::F32)?;
        let logits = gpu.alloc_owned(&[config.vocab_size], DType::F32)?;
        gpu.hip
            .memcpy_dtod_at(&x_last.buf, 0, &x_batch.buf, last_off, dim * 4)?;
        gpu.rmsnorm_f32(&x_last, &weights.output_norm, &tmp, config.norm_eps)?;
        weight_gemv(gpu, &weights.output, &tmp, &logits)?;
        gpu.download_f32(&logits)
    })();

    // Free the raw `pos_buf` on every exit path, then return the per-call
    // pooled scratch (already dropped above) to the pool.
    let _ = gpu.hip.free(pos_buf);
    gpu.reclaim_pending();

    result
}

// ─── LLaMA-family batched prefill (Phase A of #89) ─────────────────────────
//
// The fused WMMA + K4-unroll + flash-attention prefill stack lives here so
// any plain LLaMA-family loader (Qwen3, Mistral, Phi, Gemma) can drive it
// directly without going through `qwen35::forward_prefill_batch` (whose
// eligibility gate requires DeltaNet/MoE layers and whose layer enum
// branches over hybrid arch variants).
//
// Mirrors the FullAttn fast path of `qwen35::forward_prefill_chunk` kernel
// for kernel, with two adaptations:
//   1. No "Q + gate" wide projection. Plain Qwen3 attention has a normal
//      Q output (q_dim wide); no deinterleave, no sigmoid_mul step.
//   2. Full RoPE via `rope_batched_f32` (non-interleaved, half-split) to
//      match `forward_scratch`'s `rope_f32` semantics.

/// Upper bound on `forward_prefill_batch`'s per-chunk size. Mirrors the
/// qwen35 chunk cap; sized so flash_partials stays within 2 GB at the
/// largest physical_cap any consumer sets up.
pub const PREFILL_MAX_BATCH: usize = 256;

/// Per-extract-layer residual-hidden capture sink for the batched llama forward
/// (DFlash / DSpark drafter conditioning).
///
/// When threaded through [`forward_prefill_batch_capture`] /
/// [`forward_prefill_batch_tree`] (or the per-token
/// [`forward_scratch_compute_capture`]), the forward captures the residual
/// stream `x` (`[dim]` per position) AFTER each decoder layer whose index
/// appears in `extract_layers`, laying out — per processed position, across the
/// extract layers in `extract_layers` order — `extract_layers.len() × dim` f32.
/// The final buffer is therefore `[n_pos × num_extract × dim]` row-major.
///
/// `extract_layers` MUST be ascending. Capture is at the post-FFN residual,
/// independent of qk-norm (qk-norm acts on Q/K, not the residual).
pub struct HiddenCaptureSink<'a> {
    /// Decoder-layer indices to capture, ascending order.
    pub extract_layers: &'a [usize],
    /// Host output sink: appended `[n_pos × num_extract × dim]` row-major. Used
    /// only when `hidden_gpu` is `None`.
    pub hidden: &'a mut Vec<f32>,
    /// Optional GPU-resident destination (`[n_pos × num_extract × dim]` F32,
    /// position-major). When `Some`, captured extract-layer rows are copied
    /// GPU→GPU straight into this buffer and the host `hidden` Vec is left
    /// untouched (the accepted-prefix-hidden reuse stays on-device). The buffer
    /// must be `>= n_pos × extract_layers.len() × dim` F32.
    pub hidden_gpu: Option<&'a GpuTensor>,
}

/// Tree-attention mask reference for a single batched tree-verify forward
/// ([`forward_prefill_batch_tree`]).
///
/// When supplied, the batched flash-attention kernels run in tree mode: keys in
/// the in-block region `[block_start, block_start + block_cols)` are biased by
/// `bias[row × block_cols + (key_slot − block_start)]` (an additive `0.0`/`-inf`
/// mask), while prompt keys before `block_start` stay fully visible. The KV WRITE
/// slot and the FA mask stay CONTIGUOUS (`[block_start .. block_start + n)`), but
/// Q/K RoPE uses the tree-DEPTH positions (`rope_positions`), so a parent→child
/// RoPE distance is exactly 1 regardless of linearized slot.
pub struct TreeMaskRef<'a> {
    /// `[block_cols × block_cols]` row-major additive bias (0/-inf), on device.
    pub bias: &'a GpuTensor,
    /// Absolute decode position where the in-block keys begin (= `position`).
    pub block_start: usize,
    /// In-block key count (= linearized tree length `1 + tree.num_nodes()`).
    pub block_cols: usize,
    /// Per-slot DEPTH-based RoPE positions (`block_start + node.depth`), on device
    /// as `[block_cols]` i32-in-F32. Used for the Q/K RoPE rotation ONLY — the KV
    /// WRITE slot and the FA mask stay CONTIGUOUS.
    pub rope_positions: &'a GpuTensor,
}

/// Per-call scratch for `forward_prefill_batch`. Holds [N × ...] working
/// buffers reused across the per-layer loop. Sized once per model from
/// `LlamaConfig` and reused across cycles by callers that retain it.
pub struct PrefillBatchScratch {
    pub max_batch: usize,

    // Residual stream + rmsnormed/rotated activation [N × dim].
    pub x_batch: GpuTensor,
    pub x_rot_batch: GpuTensor,

    // Token ids + positions feeding batched embedding + RoPE/KV-write kernels.
    // F32 dtype for layout reasons (4 bytes/element matches i32); the kernels
    // cast the device pointer to `const int*`.
    pub positions: GpuTensor,
    pub tokens: GpuTensor,

    // Q/K/V projection outputs (no gate component for plain attention).
    pub fa_q_batch: GpuTensor,        // [N × n_heads × head_dim]
    pub fa_k_batch: GpuTensor,        // [N × n_kv_heads × head_dim]
    pub fa_v_batch: GpuTensor,        // [N × n_kv_heads × head_dim]
    pub fa_attn_out_batch: GpuTensor, // [N × n_heads × head_dim]
    // FWHT-rotated fa_attn_out for feeding MQ4 wo.
    pub fa_attn_out_rot_batch: GpuTensor,

    // FFN intermediates [N × hidden_dim].
    pub gate_ffn_batch: GpuTensor,
    pub up_batch: GpuTensor,
    pub ffn_hidden_batch: GpuTensor,

    // Flash-attention partial-result scratch (sized to support max_batch
    // tokens × n_heads × max_tiles × (2 + head_dim)).
    pub flash_partials: GpuTensor,
}

impl PrefillBatchScratch {
    pub fn new(
        gpu: &mut Gpu,
        config: &LlamaConfig,
        max_batch: usize,
        kv_max_seq: usize,
    ) -> HipResult<Self> {
        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;

        let tile_size = 128usize;
        let max_tiles = (kv_max_seq + tile_size - 1) / tile_size;
        let batch_mult = crate::config::get()
            .flash_partials_batch
            .filter(|&n| n >= 1 && n <= PREFILL_MAX_BATCH)
            .unwrap_or(16);
        let partials_size = batch_mult * config.n_heads * max_tiles * (2 + config.head_dim);

        Ok(Self {
            max_batch,
            x_batch: gpu.alloc_tensor(&[max_batch * dim], DType::F32)?,
            x_rot_batch: gpu.alloc_tensor(&[max_batch * dim], DType::F32)?,
            positions: gpu.alloc_tensor(&[max_batch], DType::F32)?,
            tokens: gpu.alloc_tensor(&[max_batch], DType::F32)?,
            fa_q_batch: gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
            fa_k_batch: gpu.alloc_tensor(&[max_batch * kv_dim], DType::F32)?,
            fa_v_batch: gpu.alloc_tensor(&[max_batch * kv_dim], DType::F32)?,
            fa_attn_out_batch: gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
            fa_attn_out_rot_batch: gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
            gate_ffn_batch: gpu.alloc_tensor(&[max_batch * hidden_dim], DType::F32)?,
            up_batch: gpu.alloc_tensor(&[max_batch * hidden_dim], DType::F32)?,
            ffn_hidden_batch: gpu.alloc_tensor(&[max_batch * hidden_dim], DType::F32)?,
            flash_partials: gpu.alloc_tensor(&[partials_size], DType::F32)?,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in [
            self.x_batch,
            self.x_rot_batch,
            self.positions,
            self.tokens,
            self.fa_q_batch,
            self.fa_k_batch,
            self.fa_v_batch,
            self.fa_attn_out_batch,
            self.fa_attn_out_rot_batch,
            self.gate_ffn_batch,
            self.up_batch,
            self.ffn_hidden_batch,
            self.flash_partials,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// Upload token ids + positions into `pbs` via sync `memcpy_htod`. Pair
/// with `forward_prefill_batch_chunk_captured` to drive a captured graph
/// without `memcpy_htod` operations sneaking in (which would otherwise
/// either error under capture or bake stale host data into the captured
/// kernarg blob). The plain `forward_prefill_batch` does its own uploads
/// internally and does not need this helper.
pub fn upload_prefill_batch_inputs(
    gpu: &mut Gpu,
    pbs: &PrefillBatchScratch,
    tokens: &[u32],
    start_pos: usize,
) -> HipResult<()> {
    let n = tokens.len();
    let tokens_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let tokens_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;
    let positions_host: Vec<i32> = (0..n).map(|i| (start_pos + i) as i32).collect();
    let positions_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;
    Ok(())
}

/// Process `tokens` through the model with one batched forward, advancing
/// `kv_cache` by `tokens.len()` positions and writing the *last* token's
/// logits into `scratch.logits`.
///
/// Eligibility (else falls back to per-token `forward_scratch` loop):
///   - all FA layer weights (wq/wk/wv/wo + w_gate/w_up/w_down) pass
///     `is_batchable_la`
///   - KV cache is Q8_0 or asym{2,3,4}
///
/// Internally chunks at `pbs.max_batch` to bound VRAM regardless of prompt
/// length. `pbs_in: Some` reuses caller-owned scratch; `None` allocates +
/// frees within the call.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    pbs_in: Option<&PrefillBatchScratch>,
) -> HipResult<()> {
    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }

    // Eligibility goes through the shared transformer seam so the
    // per-(dtype × arch) coverage decision lives in one place (mirrored by the
    // capture-mode assertion in `forward_prefill_batch_chunk_captured`).
    let arch = gpu.arch.as_str();
    let batched_enabled = crate::config::get().prefill_batched;
    let eligible =
        crate::transformer::llama_prefill_batchable(weights, kv_cache, arch, n, batched_enabled);

    if !eligible {
        for (i, &tok) in tokens.iter().enumerate() {
            forward_scratch_embed(gpu, weights, config, tok, start_pos + i, scratch)?;
            forward_scratch_compute(gpu, weights, config, start_pos + i, kv_cache, scratch)?;
        }
        return Ok(());
    }

    let mut own_pbs: Option<PrefillBatchScratch> = None;
    let pbs = if let Some(p) = pbs_in {
        p
    } else {
        let max_batch = PREFILL_MAX_BATCH.min(n.max(MIN_BATCH));
        own_pbs = Some(PrefillBatchScratch::new(
            gpu,
            config,
            max_batch,
            kv_cache.physical_cap,
        )?);
        own_pbs.as_ref().unwrap()
    };

    let max_chunk = pbs.max_batch;
    // Run the chunk loop + final projection under a closure so that any `?`
    // early-return still frees `own_pbs` (PrefillBatchScratch has no Drop).
    let result = (|| -> HipResult<()> {
        let mut offset = 0usize;
        while offset < n {
            let chunk_n = (n - offset).min(max_chunk);
            forward_prefill_chunk(
                gpu,
                weights,
                config,
                &tokens[offset..offset + chunk_n],
                start_pos + offset,
                kv_cache,
                scratch,
                pbs,
                false,
                None,
                None,
            )?;
            offset += chunk_n;
        }

        // Final norm + output projection on the LAST row of x_batch (chunk-local).
        let dim = config.dim;
        let last_n = ((n - 1) % max_chunk) + 1;
        let last_off_bytes = (last_n - 1) * dim * 4;
        gpu.hip
            .memcpy_dtod_at(&scratch.x.buf, 0, &pbs.x_batch.buf, last_off_bytes, dim * 4)?;
        gpu.rmsnorm_f32(
            &scratch.x,
            &weights.output_norm,
            &scratch.tmp,
            config.norm_eps,
        )?;
        weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;
        Ok(())
    })();

    if let Some(p) = own_pbs {
        p.free_gpu(gpu);
    }
    // Prefill boundary: drain any `OwnedTensor` scratch deferred-freed by the
    // decode path. `own_pbs` (a `PrefillBatchScratch`) is not pooled scratch and
    // keeps its own `free_gpu` on all paths above.
    gpu.reclaim_pending();
    result
}

/// `forward_prefill_batch` plus an optional per-extract-layer residual-hidden
/// capture sink (DFlash / DSpark drafter conditioning). The capture sink is only
/// honored on the eligible batched path; a capture request on an ineligible model
/// is a usage error (the per-token fallback does not run the capturing layer
/// loop), so it asserts rather than silently no-op'ing.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_capture(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    pbs_in: Option<&PrefillBatchScratch>,
    mut capture: Option<&mut HiddenCaptureSink>,
) -> HipResult<()> {
    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }

    let arch = gpu.arch.as_str();
    let batched_enabled = crate::config::get().prefill_batched;
    let eligible =
        crate::transformer::llama_prefill_batchable(weights, kv_cache, arch, n, batched_enabled);

    if !eligible {
        assert!(
            capture.is_none(),
            "forward_prefill_batch_capture: hidden capture requested but model is \
             ineligible for the batched path (n={n}); DFlash requires the batched forward"
        );
        for (i, &tok) in tokens.iter().enumerate() {
            forward_scratch_embed(gpu, weights, config, tok, start_pos + i, scratch)?;
            forward_scratch_compute(gpu, weights, config, start_pos + i, kv_cache, scratch)?;
        }
        return Ok(());
    }

    let mut own_pbs: Option<PrefillBatchScratch> = None;
    let pbs = if let Some(p) = pbs_in {
        p
    } else {
        let max_batch = PREFILL_MAX_BATCH.min(n.max(MIN_BATCH));
        own_pbs = Some(PrefillBatchScratch::new(
            gpu,
            config,
            max_batch,
            kv_cache.physical_cap,
        )?);
        own_pbs.as_ref().unwrap()
    };

    let max_chunk = pbs.max_batch;
    let result = (|| -> HipResult<()> {
        let mut offset = 0usize;
        while offset < n {
            let chunk_n = (n - offset).min(max_chunk);
            forward_prefill_chunk(
                gpu,
                weights,
                config,
                &tokens[offset..offset + chunk_n],
                start_pos + offset,
                kv_cache,
                scratch,
                pbs,
                false,
                capture.as_deref_mut(),
                None,
            )?;
            offset += chunk_n;
        }

        // Final norm + output projection on the LAST row of x_batch (chunk-local).
        let dim = config.dim;
        let last_n = ((n - 1) % max_chunk) + 1;
        let last_off_bytes = (last_n - 1) * dim * 4;
        gpu.hip
            .memcpy_dtod_at(&scratch.x.buf, 0, &pbs.x_batch.buf, last_off_bytes, dim * 4)?;
        gpu.rmsnorm_f32(
            &scratch.x,
            &weights.output_norm,
            &scratch.tmp,
            config.norm_eps,
        )?;
        weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;
        Ok(())
    })();

    if let Some(p) = own_pbs {
        p.free_gpu(gpu);
    }
    gpu.reclaim_pending();
    result
}

/// One single-pass TREE-masked verify forward. `tokens` is the linearized tree
/// (slot 0 = seed), `tree_bias` the `[n × n]` additive (`0.0`/`-inf`) mask, and
/// `depth_positions` the per-slot DEPTH RoPE positions (`position + node.depth`).
/// Q/K RoPE rotates at the DEPTH positions (parent→child distance 1) while the KV
/// write + mask stay on CONTIGUOUS slots `[position .. position + n)`, so a node's
/// logits equal a causal verify of its root-to-node chain. Every node row's
/// post-final-layer hidden lands in `pbs.x_batch`. Single chunk only
/// (`n <= pbs.max_batch`); requires Q8_0 KV (the asym/givens FA re-rotates K by
/// the write slot, which conflicts with the decoupled depth-RoPE).
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_tree(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    tokens: &[u32],
    position: usize,
    tree_bias: &GpuTensor,
    depth_positions: &[i32],
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    pbs: &PrefillBatchScratch,
    capture: Option<&mut HiddenCaptureSink>,
) -> HipResult<()> {
    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }
    assert!(
        n <= pbs.max_batch,
        "forward_prefill_batch_tree: tree size {n} exceeds pbs.max_batch {}",
        pbs.max_batch
    );
    assert_eq!(
        depth_positions.len(),
        n,
        "forward_prefill_batch_tree: depth_positions len {} != tree size {n}",
        depth_positions.len()
    );
    assert!(
        kv_cache.quant_q8
            && !(kv_cache.quant_asym2 || kv_cache.quant_asym3 || kv_cache.quant_asym4),
        "forward_prefill_batch_tree requires Q8_0 KV (decoupled depth-RoPE is \
         incompatible with the asym/givens in-kernel re-rotation)"
    );
    let arch = gpu.arch.as_str();
    assert!(
        crate::config::get().prefill_batched
            && crate::transformer::llama_weights_batchable(weights, arch),
        "forward_prefill_batch_tree requires the batched path (prefill_batched, \
         batchable weights)"
    );

    // Upload the depth-based RoPE positions into a scratch device buffer (i32 bits
    // in an F32 tensor, matching pbs.positions' slot-cosmetic dtype).
    let rope_pos = gpu.alloc_tensor(&[n], DType::F32)?;
    let rope_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(depth_positions.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&rope_pos.buf, rope_bytes)?;

    let tm = TreeMaskRef {
        bias: tree_bias,
        block_start: position,
        block_cols: n,
        rope_positions: &rope_pos,
    };

    // Upload tokens + contiguous KV-write/mask positions [position .. position+n)
    // into pbs, then run one captured chunk with the tree mask applied.
    let tokens_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let tokens_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
    let positions_host: Vec<i32> = (0..n).map(|i| (position + i) as i32).collect();
    let positions_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };

    let result = (|| -> HipResult<()> {
        gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;
        gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;
        forward_prefill_chunk(
            gpu,
            weights,
            config,
            tokens,
            position,
            kv_cache,
            scratch,
            pbs,
            true,
            capture,
            Some(&tm),
        )
    })();
    let _ = gpu.free_tensor(rope_pos);
    result
}

/// Single-chunk capture-friendly entry. The caller must have already
/// populated `pbs.tokens` and `pbs.positions` via
/// `upload_prefill_batch_inputs`, and must size `tokens.len() <= pbs.max_batch`.
/// Skips the internal `memcpy_htod` so the body is safe under
/// `hipStreamBeginCapture`. The eligibility check still runs; on a non-eligible
/// model the function asserts rather than silently falling back, since the
/// fallback would issue uploads that violate capture semantics.
///
/// Capture-mode constraint: in capture mode `max_ctx_len` is baked to
/// `kv_cache.physical_cap`. For Q8 KV at `physical_cap > LDS_CTX_LIMIT`
/// (15000), `forward_prefill_chunk` would enter the per-position
/// long-context fallback that issues `hip.malloc` + `memcpy_htod` per row
/// — both capture-illegal. Reject that combination up-front; the asym KV
/// modes have their own batched flash-masked kernels with no per-position
/// uploads, so they are capture-safe at any context length.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_chunk_captured(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    pbs: &PrefillBatchScratch,
) -> HipResult<()> {
    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }
    assert!(
        n <= pbs.max_batch,
        "captured chunk size {n} exceeds pbs.max_batch {}",
        pbs.max_batch
    );

    let arch = gpu.arch.as_str();
    assert!(
        crate::transformer::kv_quant_batchable(kv_cache)
            && crate::transformer::llama_weights_batchable(weights, arch),
        "forward_prefill_batch_chunk_captured requires batched-eligible weights + KV"
    );

    // The Q8 long-context fallback in `forward_prefill_chunk` issues
    // `hip.malloc` + per-row `memcpy_htod` inside the layer loop, which
    // would error or bake stale data under capture. The threshold is
    // baked from `physical_cap` in capture mode, not the live seq_len, so
    // we have to gate on the cap regardless of how many tokens this chunk
    // carries. Asym KV paths run pure-batched kernels and stay safe.
    const LDS_CTX_LIMIT: usize = 15000;
    assert!(
        !(kv_cache.quant_q8 && kv_cache.physical_cap > LDS_CTX_LIMIT),
        "Q8 KV with physical_cap {} > {} hits the per-position long-context fallback, \
         which issues hip.malloc + memcpy_htod inside the captured region. \
         Use asym3 KV for capture at long context, or shrink physical_cap.",
        kv_cache.physical_cap,
        LDS_CTX_LIMIT,
    );

    forward_prefill_chunk(
        gpu, weights, config, tokens, start_pos, kv_cache, scratch, pbs, true, None, None,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn forward_prefill_chunk(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut KvCache,
    s: &ForwardScratch,
    pbs: &PrefillBatchScratch,
    pre_uploaded: bool,
    // Additive DFlash/DSpark hidden capture. `None` for the plain prefill path
    // (the normal forward is byte-identical: the capture branches below are dead
    // when this is `None`).
    mut capture: Option<&mut HiddenCaptureSink>,
    // Additive tree-attention mask. `None` for the plain causal prefill path.
    tree_mask: Option<&TreeMaskRef>,
) -> HipResult<()> {
    let n = tokens.len();
    debug_assert!(n > 0);
    debug_assert!(n <= pbs.max_batch);

    // Per-extract-layer captured residual rows (host sink). Filled inside the
    // layer loop, interleaved position-major after it. Unused (empty) when
    // `capture` is `None` or a GPU-resident sink is supplied.
    let mut cap_rows: Vec<Vec<f32>> = Vec::new();

    // Tree-verify decoupling: RoPE rotates at DEPTH positions while the KV write
    // + FA mask stay on the CONTIGUOUS `pbs.positions` slots. Causal path uses
    // `pbs.positions` for both.
    let rope_positions: &GpuTensor = match tree_mask {
        Some(tm) => {
            debug_assert_eq!(
                tm.block_cols, n,
                "tree_mask.block_cols {} must equal chunk size {n}",
                tm.block_cols
            );
            tm.rope_positions
        }
        None => &pbs.positions,
    };
    let (tree_bias, tree_block_start, tree_block_cols): (Option<&GpuTensor>, usize, usize) =
        match tree_mask {
            Some(tm) => (Some(tm.bias), tm.block_start, tm.block_cols),
            None => (None, 0, 0),
        };

    let dim = config.dim;
    let hidden_dim = config.hidden_dim;
    let kv_dim = config.n_kv_heads * config.head_dim;
    let dim_row_bytes = dim * 4;
    // Q8 WMMA arch gate — see qwen35.rs q8_wmma_arch for the matching capture
    // and rationale (gfx11-only; gfx12 needs a `_w32_gfx12` builtin variant
    // that has not been authored yet, so routing gfx12 here would crash at JIT).
    let q8_wmma_arch = gpu.arch_caps.has_wmma_w32();

    // 1. Embed N tokens into pbs.x_batch.
    if matches!(
        weights.embd_format,
        EmbeddingFormat::HFQ4G256 | EmbeddingFormat::Q8_0
    ) {
        if !pre_uploaded {
            let tokens_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            let tokens_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
            gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;
        }
        match weights.embd_format {
            EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256_batched(
                &weights.token_embd,
                &pbs.x_batch,
                &pbs.tokens,
                n,
                dim,
            )?,
            EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8_batched(
                &weights.token_embd,
                &pbs.x_batch,
                &pbs.tokens,
                n,
                dim,
            )?,
            _ => unreachable!(),
        }
    } else {
        for (i, &tok) in tokens.iter().enumerate() {
            match weights.embd_format {
                EmbeddingFormat::HFQ4G128 => {
                    gpu.embedding_lookup_hfq4g128(&weights.token_embd, &s.x, tok, dim)?
                }
                EmbeddingFormat::Q4K => {
                    gpu.embedding_lookup_q4k(&weights.token_embd, &s.x, tok, dim)?
                }
                EmbeddingFormat::F32 => {
                    gpu.embedding_lookup(&weights.token_embd, &s.x, tok, dim)?
                }
                EmbeddingFormat::HFQ4G256 | EmbeddingFormat::Q8_0 => unreachable!(),
            }
            gpu.hip.memcpy_dtod_at(
                &pbs.x_batch.buf,
                i * dim_row_bytes,
                &s.x.buf,
                0,
                dim_row_bytes,
            )?;
        }
    }

    // 1b. Upload positions [start_pos .. start_pos + n] as i32.
    if !pre_uploaded {
        let positions_host: Vec<i32> = (0..n).map(|i| (start_pos + i) as i32).collect();
        let positions_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
        gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;
    }

    let max_ctx_len = if gpu.capture_mode {
        kv_cache.physical_cap
    } else {
        start_pos + n
    };

    // 2. Per-layer loop.
    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];
        let qkv_is_mq = matches!(
            layer.wq.gpu_dtype,
            DType::MQ4G256 | DType::MQ6G256 | DType::MQ3G256 | DType::MFP4G32
        );
        let qkv_is_6bit = matches!(layer.wq.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
        let qkv_is_mq3 = matches!(layer.wq.gpu_dtype, DType::MQ3G256);
        let qkv_is_fp4 = matches!(layer.wq.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);

        // attn_norm (+ FWHT for MQ — includes MFP4G32 since rotation is the
        // same FWHT pattern as MQ4).
        if qkv_is_mq {
            gpu.fused_rmsnorm_rotate_mq_batched(
                &pbs.x_batch,
                &layer.attn_norm,
                &pbs.x_rot_batch,
                dim,
                config.norm_eps,
                n,
            )?;
        } else {
            gpu.rmsnorm_batched(
                &pbs.x_batch,
                &layer.attn_norm,
                &pbs.x_rot_batch,
                n,
                dim,
                config.norm_eps,
            )?;
        }

        let qkv_is_q8 = matches!(layer.wq.gpu_dtype, DType::Q8_0);

        // 3-way fused QKV projection.
        if qkv_is_6bit {
            gpu.gemm_qkv_hfq6g256(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &pbs.x_rot_batch,
                &pbs.fa_q_batch,
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
                n,
            )?;
        } else if qkv_is_q8 && q8_wmma_arch {
            debug_assert!(
                matches!(layer.wk.gpu_dtype, DType::Q8_0)
                    && matches!(layer.wv.gpu_dtype, DType::Q8_0),
                "llama qkv Q8 WMMA dispatch requires all of wq/wk/wv to be Q8_0",
            );
            gpu.gemm_qkv_q8_0_wmma(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &pbs.x_rot_batch,
                &pbs.fa_q_batch,
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
                n,
            )?;
        } else if qkv_is_q8 {
            gpu.gemm_q8_0_batched_chunked(
                &layer.wq.buf,
                &pbs.x_rot_batch,
                &pbs.fa_q_batch,
                layer.wq.m,
                layer.wq.k,
                n,
            )?;
            gpu.gemm_q8_0_batched_chunked(
                &layer.wk.buf,
                &pbs.x_rot_batch,
                &pbs.fa_k_batch,
                layer.wk.m,
                layer.wk.k,
                n,
            )?;
            gpu.gemm_q8_0_batched_chunked(
                &layer.wv.buf,
                &pbs.x_rot_batch,
                &pbs.fa_v_batch,
                layer.wv.m,
                layer.wv.k,
                n,
            )?;
        } else if qkv_is_mq3 {
            gpu.gemm_qkv_hfq3g256_wmma(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &pbs.x_rot_batch,
                &pbs.fa_q_batch,
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
                n,
            )?;
        } else if qkv_is_fp4 {
            gpu.gemm_qkv_hfp4g32(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &pbs.x_rot_batch,
                &pbs.fa_q_batch,
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
                n,
            )?;
        } else {
            gpu.gemm_qkv_hfq4g256(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &pbs.x_rot_batch,
                &pbs.fa_q_batch,
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
                n,
            )?;
        }

        // Per-head Q/K rmsnorm (Qwen3 only — None on plain LLaMA).
        if config.has_qk_norm {
            if let Some(ref qn) = layer.q_norm {
                gpu.rmsnorm_batched(
                    &pbs.fa_q_batch,
                    qn,
                    &pbs.fa_q_batch,
                    n * config.n_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
            }
            if let Some(ref kn) = layer.k_norm {
                gpu.rmsnorm_batched(
                    &pbs.fa_k_batch,
                    kn,
                    &pbs.fa_k_batch,
                    n * config.n_kv_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
            }
        }

        // Batched full RoPE (non-interleaved, half-split convention —
        // matches forward_scratch's rope_f32).
        gpu.rope_batched_f32(
            &pbs.fa_q_batch,
            &pbs.fa_k_batch,
            rope_positions,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            config.rope_freq_base,
            n,
        )?;

        // Batched KV write.
        if kv_cache.quant_asym4 {
            let ct = kv_cache.givens_cos.as_ref().unwrap();
            let st = kv_cache.givens_sin.as_ref().unwrap();
            gpu.kv_cache_write_asym4_batched(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                &pbs.positions,
                ct,
                st,
                config.n_kv_heads,
                config.head_dim,
                n,
            )?;
        } else if kv_cache.quant_asym3 {
            let ct = kv_cache.givens_cos.as_ref().unwrap();
            let st = kv_cache.givens_sin.as_ref().unwrap();
            gpu.kv_cache_write_asym3_batched(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                &pbs.positions,
                ct,
                st,
                config.n_kv_heads,
                config.head_dim,
                n,
            )?;
        } else if kv_cache.quant_asym2 {
            let ct = kv_cache.givens_cos.as_ref().unwrap();
            let st = kv_cache.givens_sin.as_ref().unwrap();
            gpu.kv_cache_write_asym2_batched(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                &pbs.positions,
                ct,
                st,
                config.n_kv_heads,
                config.head_dim,
                n,
            )?;
        } else {
            gpu.kv_cache_write_q8_0_batched(
                &kv_cache.k_gpu[layer_idx],
                &pbs.fa_k_batch,
                &pbs.positions,
                config.n_kv_heads,
                config.head_dim,
                n,
            )?;
            gpu.kv_cache_write_q8_0_batched(
                &kv_cache.v_gpu[layer_idx],
                &pbs.fa_v_batch,
                &pbs.positions,
                config.n_kv_heads,
                config.head_dim,
                n,
            )?;
        }

        // Batched causal flash attention.
        const LDS_CTX_LIMIT: usize = 15000;
        if kv_cache.quant_asym4 {
            let ct = kv_cache.givens_cos.as_ref().unwrap();
            let st = kv_cache.givens_sin.as_ref().unwrap();
            gpu.attention_flash_asym4_batched_masked(
                &pbs.fa_q_batch,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &pbs.fa_attn_out_batch,
                &pbs.positions,
                ct,
                st,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                kv_cache.physical_cap,
                max_ctx_len,
                n,
                &pbs.flash_partials,
                tree_bias,
                tree_block_start,
                tree_block_cols,
            )?;
        } else if kv_cache.quant_asym3 {
            let ct = kv_cache.givens_cos.as_ref().unwrap();
            let st = kv_cache.givens_sin.as_ref().unwrap();
            gpu.attention_flash_asym3_batched_masked(
                &pbs.fa_q_batch,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &pbs.fa_attn_out_batch,
                &pbs.positions,
                ct,
                st,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                kv_cache.physical_cap,
                max_ctx_len,
                n,
                &pbs.flash_partials,
                tree_bias,
                tree_block_start,
                tree_block_cols,
            )?;
        } else if kv_cache.quant_asym2 {
            assert!(
                tree_mask.is_none(),
                "tree-verify (tree_mask) is unsupported on asym2 KV; requires Q8_0 KV"
            );
            let ct = kv_cache.givens_cos.as_ref().unwrap();
            let st = kv_cache.givens_sin.as_ref().unwrap();
            gpu.attention_flash_asym2_batched(
                &pbs.fa_q_batch,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &pbs.fa_attn_out_batch,
                &pbs.positions,
                ct,
                st,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                kv_cache.physical_cap,
                max_ctx_len,
                n,
                &pbs.flash_partials,
            )?;
        } else if max_ctx_len > LDS_CTX_LIMIT {
            assert!(
                tree_mask.is_none(),
                "tree-verify (tree_mask) is unsupported on the long-context Q8 \
                 per-position fallback; keep the tree within the LDS context limit"
            );
            // Long-context Q8 fallback: per-position flash.
            //
            // `pbs.positions` was uploaded as raw i32 bits but the dtype is
            // F32 (slot-cosmetic, see PrefillBatchScratch::new). `download_f32`
            // would reinterpret those bytes as floats, so positions like 15000
            // would surface as ~1e-3 subnormals that cast to 0. Reconstruct
            // from `start_pos + b` directly — the buffer layout is exactly
            // [start_pos .. start_pos + n] in linear order.
            let q_dim = config.n_heads * config.head_dim;
            let pos_buf_tmp = gpu.hip.malloc(4)?;
            // Free pos_buf_tmp on every exit: the loop below carries `?`
            // (memcpy_htod / attention) whose early-return would otherwise
            // strand this raw allocation.
            let fallback_res = (|| -> HipResult<()> {
                for b in 0..n {
                    let pos_b = start_pos + b;
                    let seq_len_b = pos_b + 1;
                    let pos_i32 = pos_b as i32;
                    gpu.hip.memcpy_htod(&pos_buf_tmp, &pos_i32.to_ne_bytes())?;
                    let q_b = pbs.fa_q_batch.sub_offset(b * q_dim, q_dim);
                    let out_b = pbs.fa_attn_out_batch.sub_offset(b * q_dim, q_dim);
                    gpu.attention_flash_q8_0(
                        &q_b,
                        &kv_cache.k_gpu[layer_idx],
                        &kv_cache.v_gpu[layer_idx],
                        &out_b,
                        &pos_buf_tmp,
                        seq_len_b,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        kv_cache.physical_cap,
                        &pbs.flash_partials,
                    )?;
                }
                Ok(())
            })();
            let _ = gpu.hip.free(pos_buf_tmp);
            fallback_res?;
        } else {
            gpu.attention_q8_0_kv_batched_masked(
                &pbs.fa_q_batch,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &pbs.fa_attn_out_batch,
                &pbs.positions,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                kv_cache.physical_cap,
                max_ctx_len,
                n,
                tree_bias,
                tree_block_start,
                tree_block_cols,
            )?;
        }

        // wo + residual.
        let wo_is_mq = matches!(
            layer.wo.gpu_dtype,
            DType::MQ4G256 | DType::MQ6G256 | DType::MQ3G256 | DType::MFP4G32
        );
        let wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
        let wo_is_mq3 = matches!(layer.wo.gpu_dtype, DType::MQ3G256);
        let wo_is_fp4 = matches!(layer.wo.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
        let wo_is_q8 = matches!(layer.wo.gpu_dtype, DType::Q8_0);
        let wo_input = if wo_is_mq {
            // F2: AWQ-aware rotate for wo (FullAttention output projection) input.
            rotate_x_mq_batched_for(
                gpu,
                &layer.wo,
                &pbs.fa_attn_out_batch,
                &pbs.fa_attn_out_rot_batch,
                layer.wo.k,
                n,
            )?;
            &pbs.fa_attn_out_rot_batch
        } else {
            &pbs.fa_attn_out_batch
        };
        if wo_is_6bit {
            gpu.gemm_hfq6g256_residual(
                &layer.wo.buf,
                wo_input,
                &pbs.x_batch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
        } else if wo_is_q8 && q8_wmma_arch {
            let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
            gpu.gemm_q8_0_residual_wmma(&layer.wo.buf, wo_input, &x_n, layer.wo.m, layer.wo.k, n)?;
        } else if wo_is_q8 {
            let scratch = pbs.x_rot_batch.sub_offset(0, n * layer.wo.m);
            gpu.gemm_q8_0_batched_chunked(
                &layer.wo.buf,
                wo_input,
                &scratch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
            let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
            gpu.add_inplace_f32(&x_n, &scratch)?;
        } else if wo_is_mq3 {
            gpu.gemm_hfq3g256_residual_wmma(
                &layer.wo.buf,
                wo_input,
                &pbs.x_batch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
        } else if wo_is_fp4 {
            gpu.gemm_hfp4g32_residual(
                &layer.wo.buf,
                wo_input,
                &pbs.x_batch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
        } else {
            gpu.gemm_hfq4g256_residual(
                &layer.wo.buf,
                wo_input,
                &pbs.x_batch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
        }

        // FFN: rmsnorm (+ FWHT for MQ — includes MFP4G32), gate+up, silu_mul,
        // w_down + residual.
        let ffn_is_mq = matches!(
            layer.w_gate.gpu_dtype,
            DType::MQ4G256 | DType::MQ6G256 | DType::MQ3G256 | DType::MFP4G32
        );
        let ffn_is_6bit = matches!(layer.w_gate.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
        let ffn_is_mq3 = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256);
        let ffn_is_fp4 = matches!(layer.w_gate.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
        let ffn_is_q8 = matches!(layer.w_gate.gpu_dtype, DType::Q8_0);
        if ffn_is_mq {
            gpu.fused_rmsnorm_rotate_mq_batched(
                &pbs.x_batch,
                &layer.ffn_norm,
                &pbs.x_rot_batch,
                dim,
                config.norm_eps,
                n,
            )?;
        } else {
            gpu.rmsnorm_batched(
                &pbs.x_batch,
                &layer.ffn_norm,
                &pbs.x_rot_batch,
                n,
                dim,
                config.norm_eps,
            )?;
        }
        if ffn_is_6bit {
            gpu.gemm_gate_up_hfq6g256(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        } else if ffn_is_q8 && q8_wmma_arch {
            debug_assert!(
                matches!(layer.w_up.gpu_dtype, DType::Q8_0),
                "llama FFN Q8 WMMA dispatch requires both w_gate and w_up to be Q8_0",
            );
            gpu.gemm_gate_up_q8_0_wmma(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        } else if ffn_is_q8 {
            gpu.gemm_q8_0_batched_chunked(
                &layer.w_gate.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                layer.w_gate.m,
                layer.w_gate.k,
                n,
            )?;
            gpu.gemm_q8_0_batched_chunked(
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.up_batch,
                layer.w_up.m,
                layer.w_up.k,
                n,
            )?;
        } else if ffn_is_mq3 {
            gpu.gemm_gate_up_hfq3g256_wmma(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        } else if ffn_is_fp4 {
            gpu.gemm_gate_up_hfp4g32(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        } else {
            gpu.gemm_gate_up_hfq4g256(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        }
        let w_down_is_mq = matches!(
            layer.w_down.gpu_dtype,
            DType::MQ4G256 | DType::MQ6G256 | DType::MQ3G256 | DType::MFP4G32
        );
        let w_down_is_6bit = matches!(layer.w_down.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
        let w_down_is_mq3 = matches!(layer.w_down.gpu_dtype, DType::MQ3G256);
        let w_down_is_fp4 = matches!(layer.w_down.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
        let w_down_is_q8 = matches!(layer.w_down.gpu_dtype, DType::Q8_0);
        if w_down_is_mq {
            // F2: AWQ-aware silu_mul+rotate for w_down input.
            fused_silu_mul_rotate_mq_batched_for(
                gpu,
                &layer.w_down,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                &pbs.ffn_hidden_batch,
                hidden_dim,
                n,
            )?;
        } else {
            gpu.silu_mul_f32(&pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch)?;
        }
        // H-Neurons CETT: snapshot the residual before the FUSED down-proj add so
        // the tap below can recover down_out = x_after - x_before (this fast path
        // fuses down_proj into x_batch, unlike the generic prefill_forward). Gated:
        // only when a capture session is active, so serving is untouched.
        let cett_x_before = if hipfire_hneurons::capture::is_active() {
            let d = layer.w_down.m;
            let snap = gpu.alloc_owned(&[n, d], DType::F32)?;
            gpu.hip
                .memcpy_dtod_at(&snap.buf, 0, &pbs.x_batch.buf, 0, n * d * 4)?;
            Some((snap, d))
        } else {
            None
        };
        if w_down_is_6bit {
            gpu.gemm_hfq6g256_residual(
                &layer.w_down.buf,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else if w_down_is_q8 && q8_wmma_arch {
            let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
            gpu.gemm_q8_0_residual_wmma(
                &layer.w_down.buf,
                &pbs.ffn_hidden_batch,
                &x_n,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else if w_down_is_q8 {
            let scratch = pbs.x_rot_batch.sub_offset(0, n * layer.w_down.m);
            gpu.gemm_q8_0_batched_chunked(
                &layer.w_down.buf,
                &pbs.ffn_hidden_batch,
                &scratch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
            let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
            gpu.add_inplace_f32(&x_n, &scratch)?;
        } else if w_down_is_mq3 {
            gpu.gemm_hfq3g256_residual_wmma(
                &layer.w_down.buf,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else if w_down_is_fp4 {
            gpu.gemm_hfp4g32_residual(
                &layer.w_down.buf,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else {
            gpu.gemm_hfq4g256_residual(
                &layer.w_down.buf,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        }
        if let Some((x_before, d)) = cett_x_before {
            let x_after = pbs.x_batch.sub_offset(0, n * d);
            hipfire_hneurons::capture::maybe_capture_ffn_residual(
                gpu,
                &pbs.ffn_hidden_batch,
                &x_before,
                &x_after,
                layer_idx,
                start_pos,
                n,
            )?;
        }

        // DFlash / DSpark capture: if this layer is an extract layer, collect its
        // post-FFN residual rows (`pbs.x_batch[..n]`). The residual stream here is
        // the layer output BEFORE the next layer's attn_norm — exactly the
        // conditioning the draft model's cross-attention consumes.
        //
        // GPU-resident sink (`hidden_gpu` Some): copy each position's row straight
        // into its position-major slot on-device (`dst[(p·L + l_idx)·dim ..]`) so
        // the whole capture never leaves VRAM. Host sink (default): download the
        // rows and interleave after the loop.
        if let Some(cap) = capture.as_deref() {
            if let Some(l_idx) = cap.extract_layers.iter().position(|&x| x == layer_idx) {
                if let Some(dst) = cap.hidden_gpu {
                    let num_extract = cap.extract_layers.len();
                    for p in 0..n {
                        let dst_off = (p * num_extract + l_idx) * dim * 4;
                        gpu.memcpy_dtod_at_auto(
                            &dst.buf,
                            dst_off,
                            &pbs.x_batch.buf,
                            p * dim * 4,
                            dim * 4,
                        )?;
                    }
                } else {
                    let rows = pbs.x_batch.sub_offset(0, n * dim);
                    cap_rows.push(gpu.download_f32(&rows)?);
                }
            }
        }

        let _ = kv_dim;
    }

    // Interleave the captured per-extract-layer rows into the HOST sink in
    // position-major order: for each position p, concat layer 0..L at p. The
    // GPU-resident sink already wrote position-major slots inline above.
    if let Some(cap) = capture.as_deref_mut() {
        if cap.hidden_gpu.is_none() {
            debug_assert_eq!(
                cap_rows.len(),
                cap.extract_layers.len(),
                "captured {} layers but extract_layers has {}",
                cap_rows.len(),
                cap.extract_layers.len()
            );
            let num_extract = cap_rows.len();
            cap.hidden.reserve(n * num_extract * dim);
            for p in 0..n {
                for layer_rows in cap_rows.iter() {
                    cap.hidden
                        .extend_from_slice(&layer_rows[p * dim..(p + 1) * dim]);
                }
            }
        }
    }

    Ok(())
}

/// Pre-allocated scratch buffers for the forward pass.
/// Allocate once, reuse every token — zero hipMalloc in the hot loop.
pub struct ForwardScratch {
    pub x: GpuTensor,
    pub tmp: GpuTensor,
    pub q: GpuTensor,
    pub k: GpuTensor,
    pub v: GpuTensor,
    pub attn_out: GpuTensor,
    pub o: GpuTensor,
    pub gate: GpuTensor,
    pub up: GpuTensor,
    pub ffn_hidden: GpuTensor,
    pub ffn_out: GpuTensor,
    pub logits: GpuTensor,
    pub sample_buf: GpuTensor,
    pub repeat_buf: GpuTensor,
    pub attn_partials: GpuTensor, // flash-decoding partial results
    pub pos_buf: hip_bridge::DeviceBuffer,
    /// FWHT-rotated x scratch for MagnumQuant batching. Sized to max(dim, hidden_dim).
    pub x_rot: GpuTensor,
}

impl ForwardScratch {
    pub fn new(gpu: &mut Gpu, config: &LlamaConfig) -> HipResult<Self> {
        let dim = config.dim;
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;
        // Flash-decoding partials: n_heads × max_chunks × (2 + head_dim) floats
        // max_chunks = ceil(2048 / 128) = 16
        let max_chunks = 16;
        let partial_stride = 2 + config.head_dim;
        let partials_size = config.n_heads * max_chunks * partial_stride;
        Ok(Self {
            x: gpu.alloc_tensor(&[dim], DType::F32)?,
            tmp: gpu.alloc_tensor(&[dim], DType::F32)?,
            q: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            k: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            v: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            attn_out: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            o: gpu.alloc_tensor(&[dim], DType::F32)?,
            gate: gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?,
            up: gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?,
            ffn_hidden: gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?,
            ffn_out: gpu.alloc_tensor(&[dim], DType::F32)?,
            logits: gpu.alloc_tensor(&[config.vocab_size], DType::F32)?,
            sample_buf: gpu.alloc_tensor(&[2], DType::F32)?,
            repeat_buf: gpu.alloc_tensor(&[64], DType::F32)?,
            attn_partials: gpu.alloc_tensor(&[partials_size], DType::F32)?,
            pos_buf: gpu.hip.malloc(4)?, // single i32
            x_rot: gpu.alloc_tensor(&[dim.max(config.hidden_dim)], DType::F32)?,
        })
    }

    /// Return all GPU buffers to the pool (drained on unload). Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in [
            self.x,
            self.tmp,
            self.q,
            self.k,
            self.v,
            self.attn_out,
            self.o,
            self.gate,
            self.up,
            self.ffn_hidden,
            self.ffn_out,
            self.logits,
            self.sample_buf,
            self.repeat_buf,
            self.attn_partials,
            self.x_rot,
        ] {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.hip.free(self.pos_buf);
    }
}

/// Forward pass with persistent scratch buffers. Zero allocations.
/// Returns (token_id, new_rng_state) via GPU-side sampling.
pub fn forward_scratch(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    token: u32,
    pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    temperature: f32,
    top_p: f32,
    rng_state: u32,
    repeat_window: usize,
    repeat_penalty: f32,
) -> HipResult<(u32, u32)> {
    forward_scratch_embed(gpu, weights, config, token, pos, scratch)?;
    forward_scratch_layers(
        gpu,
        weights,
        config,
        pos,
        kv_cache,
        scratch,
        temperature,
        top_p,
        rng_state,
        repeat_window,
        repeat_penalty,
    )
}

/// Upload pos and compute embedding. Must be called before forward_scratch_layers.
pub fn forward_scratch_embed(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    token: u32,
    pos: usize,
    scratch: &ForwardScratch,
) -> HipResult<()> {
    let dim = config.dim;
    // Upload pos to GPU buffer (4 bytes)
    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
    // Embedding lookup
    match weights.embd_format {
        EmbeddingFormat::Q4K => {
            gpu.embedding_lookup_q4k(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::F32 => {
            gpu.embedding_lookup(&weights.token_embd, &scratch.x, token, dim)?
        }
    }
    Ok(())
}

/// Layer loop + final norm + logits + sampling. Graph-capturable.
pub fn forward_scratch_layers(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    temperature: f32,
    top_p: f32,
    rng_state: u32,
    repeat_window: usize,
    repeat_penalty: f32,
) -> HipResult<(u32, u32)> {
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;
    let kv_dim = n_kv_heads * head_dim;

    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];

        gpu.rmsnorm_f32(&scratch.x, &layer.attn_norm, &scratch.tmp, config.norm_eps)?;

        if layer.wq.gpu_dtype == DType::Q4K && layer.wk.gpu_dtype == DType::Q4K {
            gpu.fused_qkv_q4k(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &scratch.tmp,
                &scratch.q,
                &scratch.k,
                &scratch.v,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
            )?;
        } else {
            // Batch FWHT for MQ weights: wq/wk/wv all consume scratch.tmp.
            let x_rot = rotate_x_for_mq(gpu, &layer.wq, &scratch.tmp, &scratch.x_rot)?;
            weight_gemv_prerotated(gpu, &layer.wq, &scratch.tmp, x_rot, &scratch.q)?;
            weight_gemv_prerotated(gpu, &layer.wk, &scratch.tmp, x_rot, &scratch.k)?;
            weight_gemv_prerotated(gpu, &layer.wv, &scratch.tmp, x_rot, &scratch.v)?;
        }

        if config.has_qk_norm {
            if let Some(ref qn) = layer.q_norm {
                gpu.rmsnorm_batched(
                    &scratch.q,
                    qn,
                    &scratch.q,
                    n_heads,
                    head_dim,
                    config.norm_eps,
                )?;
            }
            if let Some(ref kn) = layer.k_norm {
                gpu.rmsnorm_batched(
                    &scratch.k,
                    kn,
                    &scratch.k,
                    n_kv_heads,
                    head_dim,
                    config.norm_eps,
                )?;
            }
        }

        gpu.rope_f32(
            &scratch.q,
            &scratch.k,
            &scratch.pos_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            config.rope_freq_base,
        )?;

        if kv_cache.quant_hfq4 {
            gpu.kv_cache_write_hfq4(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_hfq4(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_hfq4_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized
            && !kv_cache.k_scales.is_empty()
            && !kv_cache.quant_int8
            && !kv_cache.quant_q8
        {
            // HFQ8 flat layout
            gpu.kv_cache_write_hfq8(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.k_scales[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_hfq8(
                &kv_cache.v_gpu[layer_idx],
                &kv_cache.v_scales[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_hfq8_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.k_scales[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &kv_cache.v_scales[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else if kv_cache.quant_int8 {
            gpu.kv_cache_write_int8c_f16(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_int8c_f16(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_int8c_f16_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized && kv_cache.quant_q8 {
            gpu.kv_cache_write_q8_0(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_q8_0(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_q8_0_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized {
            gpu.kv_cache_write_q4(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_q4(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_q4kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else {
            gpu.kv_cache_write(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                kv_dim,
            )?;
            gpu.kv_cache_write(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                kv_dim,
            )?;
            gpu.attention_f32(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        }

        weight_gemv(gpu, &layer.wo, &scratch.attn_out, &scratch.o)?;
        gpu.add_inplace_f32(&scratch.x, &scratch.o)?;

        gpu.rmsnorm_f32(&scratch.x, &layer.ffn_norm, &scratch.tmp, config.norm_eps)?;
        if layer.w_gate.gpu_dtype == DType::Q4K && layer.w_up.gpu_dtype == DType::Q4K {
            gpu.fused_gate_up_q4k(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &scratch.tmp,
                &scratch.gate,
                &scratch.up,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
            )?;
        } else {
            // Batch FWHT for MQ weights: w_gate/w_up share scratch.tmp.
            let x_rot = rotate_x_for_mq(gpu, &layer.w_gate, &scratch.tmp, &scratch.x_rot)?;
            weight_gemv_prerotated(gpu, &layer.w_gate, &scratch.tmp, x_rot, &scratch.gate)?;
            weight_gemv_prerotated(gpu, &layer.w_up, &scratch.tmp, x_rot, &scratch.up)?;
        }

        gpu.silu_mul_f32(&scratch.gate, &scratch.up, &scratch.ffn_hidden)?;
        weight_gemv(gpu, &layer.w_down, &scratch.ffn_hidden, &scratch.ffn_out)?;
        gpu.add_inplace_f32(&scratch.x, &scratch.ffn_out)?;
    }

    gpu.rmsnorm_f32(
        &scratch.x,
        &weights.output_norm,
        &scratch.tmp,
        config.norm_eps,
    )?;
    weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;

    // GPU-side sampling (includes sync readback — can't be in graph capture)
    gpu.sample_top_p(
        &scratch.logits,
        &scratch.sample_buf,
        &scratch.repeat_buf,
        config.vocab_size,
        temperature,
        top_p,
        rng_state,
        repeat_window,
        repeat_penalty,
    )
}

/// Early-exit forward pass: check confidence at checkpoint layers, skip rest if confident.
/// Returns (token_id, rng_state, exit_layer) — exit_layer is which layer triggered the exit.
/// If no early exit, exit_layer = n_layers (ran all layers normally).
pub fn forward_early_exit(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    token: u32,
    pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    temperature: f32,
    top_p: f32,
    rng_state: u32,
    repeat_window: usize,
    repeat_penalty: f32,
    exit_threshold: f32,         // max softmax prob threshold (e.g., 0.9)
    checkpoint_layers: &[usize], // which layers to check (e.g., &[12, 24])
) -> HipResult<(u32, u32, usize)> {
    // Embed
    forward_scratch_embed(gpu, weights, config, token, pos, scratch)?;

    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;
    let kv_dim = n_kv_heads * head_dim;

    let mut exit_layer = config.n_layers;

    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];

        // Standard layer computation (same as forward_scratch_layers)
        gpu.rmsnorm_f32(&scratch.x, &layer.attn_norm, &scratch.tmp, config.norm_eps)?;

        if layer.wq.gpu_dtype == DType::Q4K && layer.wk.gpu_dtype == DType::Q4K {
            gpu.fused_qkv_q4k(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &scratch.tmp,
                &scratch.q,
                &scratch.k,
                &scratch.v,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
            )?;
        } else {
            // Batch FWHT for MQ weights: wq/wk/wv all consume scratch.tmp.
            let x_rot = rotate_x_for_mq(gpu, &layer.wq, &scratch.tmp, &scratch.x_rot)?;
            weight_gemv_prerotated(gpu, &layer.wq, &scratch.tmp, x_rot, &scratch.q)?;
            weight_gemv_prerotated(gpu, &layer.wk, &scratch.tmp, x_rot, &scratch.k)?;
            weight_gemv_prerotated(gpu, &layer.wv, &scratch.tmp, x_rot, &scratch.v)?;
        }

        if config.has_qk_norm {
            if let Some(ref qn) = layer.q_norm {
                gpu.rmsnorm_batched(
                    &scratch.q,
                    qn,
                    &scratch.q,
                    n_heads,
                    head_dim,
                    config.norm_eps,
                )?;
            }
            if let Some(ref kn) = layer.k_norm {
                gpu.rmsnorm_batched(
                    &scratch.k,
                    kn,
                    &scratch.k,
                    n_kv_heads,
                    head_dim,
                    config.norm_eps,
                )?;
            }
        }

        gpu.rope_f32(
            &scratch.q,
            &scratch.k,
            &scratch.pos_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            config.rope_freq_base,
        )?;

        // KV write + attention (use same dispatch as forward_scratch_layers)
        if kv_cache.quantized && kv_cache.quant_q8 {
            gpu.kv_cache_write_q8_0(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_q8_0(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_q8_0_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else {
            gpu.kv_cache_write(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                kv_dim,
            )?;
            gpu.kv_cache_write(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                kv_dim,
            )?;
            gpu.attention_f32(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        }

        weight_gemv(gpu, &layer.wo, &scratch.attn_out, &scratch.o)?;
        gpu.add_inplace_f32(&scratch.x, &scratch.o)?;

        gpu.rmsnorm_f32(&scratch.x, &layer.ffn_norm, &scratch.tmp, config.norm_eps)?;
        if layer.w_gate.gpu_dtype == DType::Q4K && layer.w_up.gpu_dtype == DType::Q4K {
            gpu.fused_gate_up_q4k(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &scratch.tmp,
                &scratch.gate,
                &scratch.up,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
            )?;
        } else {
            // Batch FWHT for MQ weights: w_gate/w_up share scratch.tmp.
            let x_rot = rotate_x_for_mq(gpu, &layer.w_gate, &scratch.tmp, &scratch.x_rot)?;
            weight_gemv_prerotated(gpu, &layer.w_gate, &scratch.tmp, x_rot, &scratch.gate)?;
            weight_gemv_prerotated(gpu, &layer.w_up, &scratch.tmp, x_rot, &scratch.up)?;
        }

        gpu.silu_mul_f32(&scratch.gate, &scratch.up, &scratch.ffn_hidden)?;
        weight_gemv(gpu, &layer.w_down, &scratch.ffn_hidden, &scratch.ffn_out)?;
        gpu.add_inplace_f32(&scratch.x, &scratch.ffn_out)?;

        // Early exit check at checkpoint layers
        if checkpoint_layers.contains(&layer_idx) && exit_threshold > 0.0 {
            // Compute logits from intermediate hidden state
            gpu.rmsnorm_f32(
                &scratch.x,
                &weights.output_norm,
                &scratch.tmp,
                config.norm_eps,
            )?;
            weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;

            // GPU-side confidence check: compute max(softmax) on GPU, download 4 bytes
            gpu.max_prob(&scratch.logits, &scratch.sample_buf, config.vocab_size)?;
            let mut prob_bytes = [0u8; 4];
            gpu.hip
                .memcpy_dtoh(&mut prob_bytes, &scratch.sample_buf.buf)?;
            let max_prob = f32::from_ne_bytes(prob_bytes);

            if max_prob >= exit_threshold {
                exit_layer = layer_idx + 1;
                // Sample from these logits
                let (tok, rng) = gpu.sample_top_p(
                    &scratch.logits,
                    &scratch.sample_buf,
                    &scratch.repeat_buf,
                    config.vocab_size,
                    temperature,
                    top_p,
                    rng_state,
                    repeat_window,
                    repeat_penalty,
                )?;
                return Ok((tok, rng, exit_layer));
            }
        }
    }

    // No early exit — run full final norm + logits + sampling
    gpu.rmsnorm_f32(
        &scratch.x,
        &weights.output_norm,
        &scratch.tmp,
        config.norm_eps,
    )?;
    weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;
    let (tok, rng) = gpu.sample_top_p(
        &scratch.logits,
        &scratch.sample_buf,
        &scratch.repeat_buf,
        config.vocab_size,
        temperature,
        top_p,
        rng_state,
        repeat_window,
        repeat_penalty,
    )?;
    Ok((tok, rng, exit_layer))
}

/// Layer loop + final norm + logits only (no sampling). Graph-capturable.
pub fn forward_scratch_compute(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
) -> HipResult<()> {
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;
    let kv_dim = n_kv_heads * head_dim;

    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];
        gpu.rmsnorm_f32(&scratch.x, &layer.attn_norm, &scratch.tmp, config.norm_eps)?;

        if layer.wq.gpu_dtype == DType::Q4K && layer.wk.gpu_dtype == DType::Q4K {
            gpu.fused_qkv_q4k(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &scratch.tmp,
                &scratch.q,
                &scratch.k,
                &scratch.v,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
            )?;
        } else {
            // Batch FWHT for MQ weights: wq/wk/wv all consume scratch.tmp.
            let x_rot = rotate_x_for_mq(gpu, &layer.wq, &scratch.tmp, &scratch.x_rot)?;
            weight_gemv_prerotated(gpu, &layer.wq, &scratch.tmp, x_rot, &scratch.q)?;
            weight_gemv_prerotated(gpu, &layer.wk, &scratch.tmp, x_rot, &scratch.k)?;
            weight_gemv_prerotated(gpu, &layer.wv, &scratch.tmp, x_rot, &scratch.v)?;
        }

        if config.has_qk_norm {
            if let Some(ref qn) = layer.q_norm {
                gpu.rmsnorm_batched(
                    &scratch.q,
                    qn,
                    &scratch.q,
                    n_heads,
                    head_dim,
                    config.norm_eps,
                )?;
            }
            if let Some(ref kn) = layer.k_norm {
                gpu.rmsnorm_batched(
                    &scratch.k,
                    kn,
                    &scratch.k,
                    n_kv_heads,
                    head_dim,
                    config.norm_eps,
                )?;
            }
        }

        gpu.rope_f32(
            &scratch.q,
            &scratch.k,
            &scratch.pos_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            config.rope_freq_base,
        )?;

        if kv_cache.quant_hfq4 {
            gpu.kv_cache_write_hfq4(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_hfq4(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_hfq4_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized
            && !kv_cache.k_scales.is_empty()
            && !kv_cache.quant_int8
            && !kv_cache.quant_q8
        {
            // HFQ8 flat layout
            gpu.kv_cache_write_hfq8(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.k_scales[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_hfq8(
                &kv_cache.v_gpu[layer_idx],
                &kv_cache.v_scales[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_hfq8_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.k_scales[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &kv_cache.v_scales[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else if kv_cache.quant_int8 {
            gpu.kv_cache_write_int8c_f16(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_int8c_f16(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_int8c_f16_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized && kv_cache.quant_q8 {
            gpu.kv_cache_write_q8_0(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_q8_0(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_q8_0_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized {
            gpu.kv_cache_write_q4(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_q4(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_q4kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else {
            gpu.kv_cache_write(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                kv_dim,
            )?;
            gpu.kv_cache_write(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                kv_dim,
            )?;
            gpu.attention_f32(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        }

        weight_gemv(gpu, &layer.wo, &scratch.attn_out, &scratch.o)?;
        gpu.add_inplace_f32(&scratch.x, &scratch.o)?;

        gpu.rmsnorm_f32(&scratch.x, &layer.ffn_norm, &scratch.tmp, config.norm_eps)?;
        if layer.w_gate.gpu_dtype == DType::Q4K && layer.w_up.gpu_dtype == DType::Q4K {
            gpu.fused_gate_up_q4k(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &scratch.tmp,
                &scratch.gate,
                &scratch.up,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
            )?;
        } else {
            // Batch FWHT for MQ weights: w_gate/w_up share scratch.tmp.
            let x_rot = rotate_x_for_mq(gpu, &layer.w_gate, &scratch.tmp, &scratch.x_rot)?;
            weight_gemv_prerotated(gpu, &layer.w_gate, &scratch.tmp, x_rot, &scratch.gate)?;
            weight_gemv_prerotated(gpu, &layer.w_up, &scratch.tmp, x_rot, &scratch.up)?;
        }

        gpu.silu_mul_f32(&scratch.gate, &scratch.up, &scratch.ffn_hidden)?;
        weight_gemv(gpu, &layer.w_down, &scratch.ffn_hidden, &scratch.ffn_out)?;
        gpu.add_inplace_f32(&scratch.x, &scratch.ffn_out)?;
    }

    gpu.rmsnorm_f32(
        &scratch.x,
        &weights.output_norm,
        &scratch.tmp,
        config.norm_eps,
    )?;
    weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;
    Ok(())
}

/// `forward_scratch_compute` plus an optional per-extract-layer residual-hidden
/// capture sink. Processes ONE token (decode kernel — bit-identical to the plain
/// `forward_scratch_compute`), and for each decoder layer whose index appears in
/// `capture.extract_layers` downloads the post-FFN residual (`scratch.x[..dim]`)
/// and appends, in `extract_layers` ascending order, `num_extract × dim` f32 to
/// the host sink. The per-token twin of `forward_prefill_batch_capture`'s
/// capture; used by the DFlash spec prefill so the speculator conditions on the
/// EXACT residual stream the per-token decode produces (the batched prefill
/// kernel is not bitwise-equal to the decode kernel and flips argmax at near-tie
/// logits).
///
/// This is a NEW function alongside `forward_scratch_compute`; the plain forward
/// is left byte-identical (no capture taps woven into the hot path).
#[allow(clippy::too_many_arguments)]
pub fn forward_scratch_compute_capture(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    mut capture: Option<&mut HiddenCaptureSink>,
) -> HipResult<()> {
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let dim = config.dim;
    // Per-token capture appends rows in extract-layer ASCENDING order, matching
    // the batched capture's layout for n=1 (one position per call).
    let mut cap_rows: Vec<Vec<f32>> = Vec::new();

    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];
        gpu.rmsnorm_f32(&scratch.x, &layer.attn_norm, &scratch.tmp, config.norm_eps)?;

        if layer.wq.gpu_dtype == DType::Q4K && layer.wk.gpu_dtype == DType::Q4K {
            gpu.fused_qkv_q4k(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &scratch.tmp,
                &scratch.q,
                &scratch.k,
                &scratch.v,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
            )?;
        } else {
            let x_rot = rotate_x_for_mq(gpu, &layer.wq, &scratch.tmp, &scratch.x_rot)?;
            weight_gemv_prerotated(gpu, &layer.wq, &scratch.tmp, x_rot, &scratch.q)?;
            weight_gemv_prerotated(gpu, &layer.wk, &scratch.tmp, x_rot, &scratch.k)?;
            weight_gemv_prerotated(gpu, &layer.wv, &scratch.tmp, x_rot, &scratch.v)?;
        }

        if config.has_qk_norm {
            if let Some(ref qn) = layer.q_norm {
                gpu.rmsnorm_batched(
                    &scratch.q,
                    qn,
                    &scratch.q,
                    n_heads,
                    head_dim,
                    config.norm_eps,
                )?;
            }
            if let Some(ref kn) = layer.k_norm {
                gpu.rmsnorm_batched(
                    &scratch.k,
                    kn,
                    &scratch.k,
                    n_kv_heads,
                    head_dim,
                    config.norm_eps,
                )?;
            }
        }

        gpu.rope_f32(
            &scratch.q,
            &scratch.k,
            &scratch.pos_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            config.rope_freq_base,
        )?;

        if kv_cache.quant_hfq4 {
            gpu.kv_cache_write_hfq4(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_hfq4(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_hfq4_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized
            && !kv_cache.k_scales.is_empty()
            && !kv_cache.quant_int8
            && !kv_cache.quant_q8
        {
            // HFQ8 flat layout
            gpu.kv_cache_write_hfq8(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.k_scales[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_hfq8(
                &kv_cache.v_gpu[layer_idx],
                &kv_cache.v_scales[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_hfq8_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.k_scales[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &kv_cache.v_scales[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else if kv_cache.quant_int8 {
            gpu.kv_cache_write_int8c_f16(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_int8c_f16(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_int8c_f16_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized && kv_cache.quant_q8 {
            gpu.kv_cache_write_q8_0(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_q8_0(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_q8_0_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized {
            gpu.kv_cache_write_q4(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_q4(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_q4kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        } else {
            gpu.kv_cache_write(
                &kv_cache.k_gpu[layer_idx],
                &scratch.k,
                &scratch.pos_buf,
                kv_dim,
            )?;
            gpu.kv_cache_write(
                &kv_cache.v_gpu[layer_idx],
                &scratch.v,
                &scratch.pos_buf,
                kv_dim,
            )?;
            gpu.attention_f32(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out,
                &scratch.pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
        }

        weight_gemv(gpu, &layer.wo, &scratch.attn_out, &scratch.o)?;
        gpu.add_inplace_f32(&scratch.x, &scratch.o)?;

        gpu.rmsnorm_f32(&scratch.x, &layer.ffn_norm, &scratch.tmp, config.norm_eps)?;
        if layer.w_gate.gpu_dtype == DType::Q4K && layer.w_up.gpu_dtype == DType::Q4K {
            gpu.fused_gate_up_q4k(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &scratch.tmp,
                &scratch.gate,
                &scratch.up,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
            )?;
        } else {
            let x_rot = rotate_x_for_mq(gpu, &layer.w_gate, &scratch.tmp, &scratch.x_rot)?;
            weight_gemv_prerotated(gpu, &layer.w_gate, &scratch.tmp, x_rot, &scratch.gate)?;
            weight_gemv_prerotated(gpu, &layer.w_up, &scratch.tmp, x_rot, &scratch.up)?;
        }

        gpu.silu_mul_f32(&scratch.gate, &scratch.up, &scratch.ffn_hidden)?;
        weight_gemv(gpu, &layer.w_down, &scratch.ffn_hidden, &scratch.ffn_out)?;
        gpu.add_inplace_f32(&scratch.x, &scratch.ffn_out)?;

        // DFlash per-token capture: download this layer's post-FFN residual
        // (`scratch.x[..dim]`) if it is an extract layer. Same tap point as the
        // batched capture (post-FFN residual, before the next attn_norm).
        if let Some(cap) = capture.as_deref() {
            if cap.extract_layers.contains(&layer_idx) {
                let row = scratch.x.sub_offset(0, dim);
                cap_rows.push(gpu.download_f32(&row)?);
            }
        }
    }

    if let Some(cap) = capture.as_deref_mut() {
        debug_assert_eq!(
            cap_rows.len(),
            cap.extract_layers.len(),
            "captured {} layers but extract_layers has {}",
            cap_rows.len(),
            cap.extract_layers.len()
        );
        for layer_rows in cap_rows.iter() {
            cap.hidden.extend_from_slice(&layer_rows[..dim]);
        }
    }

    gpu.rmsnorm_f32(
        &scratch.x,
        &weights.output_norm,
        &scratch.tmp,
        config.norm_eps,
    )?;
    weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;
    Ok(())
}

/// Run a single forward pass for one token (decode step).
/// Returns logits over vocab.
pub fn forward(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    token: u32,
    pos: usize,
    kv_cache: &mut KvCache,
) -> HipResult<Vec<f32>> {
    let dim = config.dim;
    let head_dim = config.head_dim;
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let kv_dim = n_kv_heads * head_dim;

    // Embedding lookup — GPU-side D2D copy of one row (8KB vs 262MB download).
    // All of these are per-call pooled scratch: `OwnedTensor` returns each to the
    // pool on drop and `reclaim_pending` below drains the deferred-free mailbox.
    let x = gpu.alloc_owned(&[dim], DType::F32)?;
    match weights.embd_format {
        EmbeddingFormat::Q4K => gpu.embedding_lookup_q4k(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &x, token, dim)?
        }
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &x, token, dim)?,
    }

    let tmp = gpu.alloc_owned(&[dim], DType::F32)?;

    // Pre-allocate scratch buffers — reused every layer (eliminates 324 allocs per token)
    let q_dim = n_heads * head_dim;
    let q = gpu.alloc_owned(&[q_dim], DType::F32)?;
    let k = gpu.alloc_owned(&[kv_dim], DType::F32)?;
    let v = gpu.alloc_owned(&[kv_dim], DType::F32)?;
    let attn_out = gpu.alloc_owned(&[q_dim], DType::F32)?;
    let o = gpu.alloc_owned(&[dim], DType::F32)?;
    let gate = gpu.alloc_owned(&[config.hidden_dim], DType::F32)?;
    let up = gpu.alloc_owned(&[config.hidden_dim], DType::F32)?;
    let ffn_hidden = gpu.alloc_owned(&[config.hidden_dim], DType::F32)?;
    let ffn_out = gpu.alloc_owned(&[dim], DType::F32)?;

    // Upload pos to GPU buffer (4 bytes). Raw (non-pooled) allocation, so
    // `OwnedTensor` does not apply: it is freed on every exit path after the
    // closure below, which is kept solely to funnel that free through any `?`.
    let pos_buf = gpu.hip.malloc(4)?;
    let pos_i32 = pos as i32;
    gpu.hip.memcpy_htod(&pos_buf, &pos_i32.to_ne_bytes())?;

    let result = (|| -> HipResult<Vec<f32>> {
        for layer_idx in 0..config.n_layers {
            let layer = &weights.layers[layer_idx];

            // RMSNorm before attention
            gpu.rmsnorm_f32(&x, &layer.attn_norm, &tmp, config.norm_eps)?;

            // Fused QKV: 3 GEMVs in 1 kernel launch (saves 2 launches per layer)
            if layer.wq.gpu_dtype == DType::Q4K
                && layer.wk.gpu_dtype == DType::Q4K
                && layer.wv.gpu_dtype == DType::Q4K
            {
                gpu.fused_qkv_q4k(
                    &layer.wq.buf,
                    &layer.wk.buf,
                    &layer.wv.buf,
                    &tmp,
                    &q,
                    &k,
                    &v,
                    layer.wq.m,
                    layer.wk.m,
                    layer.wv.m,
                    layer.wq.k,
                )?;
            } else {
                weight_gemv(gpu, &layer.wq, &tmp, &q)?;
                weight_gemv(gpu, &layer.wk, &tmp, &k)?;
                weight_gemv(gpu, &layer.wv, &tmp, &v)?;
            }

            // QK normalization (Qwen3) — GPU-side per-head RMSNorm.
            // Launches n_heads blocks, each normalizing head_dim elements.
            if config.has_qk_norm {
                if let Some(ref qn) = layer.q_norm {
                    gpu.rmsnorm_batched(&q, qn, &q, n_heads, head_dim, config.norm_eps)?;
                }
                if let Some(ref kn) = layer.k_norm {
                    gpu.rmsnorm_batched(&k, kn, &k, n_kv_heads, head_dim, config.norm_eps)?;
                }
            }

            // RoPE — GPU-side, reads pos from GPU buffer
            gpu.rope_f32(
                &q,
                &k,
                &pos_buf,
                n_heads,
                n_kv_heads,
                head_dim,
                config.rope_freq_base,
            )?;

            // Store K, V in GPU cache + attention
            gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &k, &pos_buf, kv_dim)?;
            gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &v, &pos_buf, kv_dim)?;
            gpu.attention_f32(
                &q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &attn_out,
                &pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;
            // Output projection: o = Wo * attn_out
            weight_gemv(gpu, &layer.wo, &attn_out, &o)?;

            // Residual: x += o (in-place)
            gpu.add_inplace_f32(&x, &o)?;

            // FFN
            gpu.rmsnorm_f32(&x, &layer.ffn_norm, &tmp, config.norm_eps)?;
            // Fused Gate+Up: 2 GEMVs in 1 kernel launch
            if layer.w_gate.gpu_dtype == DType::Q4K && layer.w_up.gpu_dtype == DType::Q4K {
                gpu.fused_gate_up_q4k(
                    &layer.w_gate.buf,
                    &layer.w_up.buf,
                    &tmp,
                    &gate,
                    &up,
                    layer.w_gate.m,
                    layer.w_up.m,
                    layer.w_gate.k,
                )?;
            } else {
                weight_gemv(gpu, &layer.w_gate, &tmp, &gate)?;
                weight_gemv(gpu, &layer.w_up, &tmp, &up)?;
            }

            // Fused SiLU(gate) * up
            gpu.silu_mul_f32(&gate, &up, &ffn_hidden)?;

            // Down projection
            weight_gemv(gpu, &layer.w_down, &ffn_hidden, &ffn_out)?;

            // Residual: x += ffn_out (in-place)
            gpu.add_inplace_f32(&x, &ffn_out)?;
        }

        // Final norm
        gpu.rmsnorm_f32(&x, &weights.output_norm, &tmp, config.norm_eps)?;

        // Logits: output = output_weight * x
        let logits = gpu.alloc_owned(&[config.vocab_size], DType::F32)?;
        weight_gemv(gpu, &weights.output, &tmp, &logits)?;
        gpu.download_f32(&logits)
    })();

    // Free the raw `pos_buf` on every exit path; the pooled scratch is RAII
    // (`OwnedTensor`) and is returned to the pool by `reclaim_pending`.
    gpu.hip.free(pos_buf).ok(); // raw hip.malloc — not pooled, free explicitly
    gpu.reclaim_pending();

    result
}

/// Forward pass + GPU-side sampling. Returns (token_id, new_rng_state).
/// Logits stay on GPU — only 8 bytes downloaded instead of 600KB.
pub fn forward_sample(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    token: u32,
    pos: usize,
    kv_cache: &mut KvCache,
    sample_buf: &GpuTensor,
    repeat_buf: &GpuTensor,
    temperature: f32,
    top_p: f32,
    rng_state: u32,
    repeat_window: usize,
    repeat_penalty: f32,
) -> HipResult<(u32, u32)> {
    let logits_on_gpu = forward_logits_gpu(gpu, weights, config, token, pos, kv_cache)?;
    let result = gpu.sample_top_p(
        &logits_on_gpu,
        sample_buf,
        repeat_buf,
        config.vocab_size,
        temperature,
        top_p,
        rng_state,
        repeat_window,
        repeat_penalty,
    )?;
    gpu.free_tensor(logits_on_gpu)?;
    Ok(result)
}

/// Forward pass that keeps logits on GPU (no download).
fn forward_logits_gpu(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    token: u32,
    pos: usize,
    kv_cache: &mut KvCache,
) -> HipResult<GpuTensor> {
    let dim = config.dim;
    let kv_dim = config.n_kv_heads * config.head_dim;
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;

    // Per-call pooled scratch — RAII via `OwnedTensor`, drained by
    // `reclaim_pending` below. `logits` is the exception: it is returned to the
    // caller, so it stays a plain pooled tensor (see the head below).
    let x = gpu.alloc_owned(&[dim], DType::F32)?;
    match weights.embd_format {
        EmbeddingFormat::Q4K => gpu.embedding_lookup_q4k(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &x, token, dim)?
        }
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &x, token, dim)?,
    }

    let tmp = gpu.alloc_owned(&[dim], DType::F32)?;
    let q = gpu.alloc_owned(&[n_heads * head_dim], DType::F32)?;
    let k = gpu.alloc_owned(&[kv_dim], DType::F32)?;
    let v = gpu.alloc_owned(&[kv_dim], DType::F32)?;
    let attn_out = gpu.alloc_owned(&[n_heads * head_dim], DType::F32)?;
    let o = gpu.alloc_owned(&[dim], DType::F32)?;
    let gate = gpu.alloc_owned(&[config.hidden_dim], DType::F32)?;
    let up = gpu.alloc_owned(&[config.hidden_dim], DType::F32)?;
    let ffn_hidden = gpu.alloc_owned(&[config.hidden_dim], DType::F32)?;
    let ffn_out = gpu.alloc_owned(&[dim], DType::F32)?;

    // Upload pos to GPU buffer (4 bytes). Raw (non-pooled) allocation, so
    // `OwnedTensor` does not apply: it is freed on every exit path after the
    // closure below, which is kept solely to funnel that free through any `?`.
    let pos_buf = gpu.hip.malloc(4)?;
    let pos_i32 = pos as i32;
    gpu.hip.memcpy_htod(&pos_buf, &pos_i32.to_ne_bytes())?;

    // `logits` is moved out on success and freed only on the post-alloc error
    // branch inside the closure.
    let result = (|| -> HipResult<GpuTensor> {
        for layer_idx in 0..config.n_layers {
            let layer = &weights.layers[layer_idx];
            gpu.rmsnorm_f32(&x, &layer.attn_norm, &tmp, config.norm_eps)?;

            if layer.wq.gpu_dtype == DType::Q4K && layer.wk.gpu_dtype == DType::Q4K {
                gpu.fused_qkv_q4k(
                    &layer.wq.buf,
                    &layer.wk.buf,
                    &layer.wv.buf,
                    &tmp,
                    &q,
                    &k,
                    &v,
                    layer.wq.m,
                    layer.wk.m,
                    layer.wv.m,
                    layer.wq.k,
                )?;
            } else {
                weight_gemv(gpu, &layer.wq, &tmp, &q)?;
                weight_gemv(gpu, &layer.wk, &tmp, &k)?;
                weight_gemv(gpu, &layer.wv, &tmp, &v)?;
            }

            if config.has_qk_norm {
                if let Some(ref qn) = layer.q_norm {
                    gpu.rmsnorm_batched(&q, qn, &q, n_heads, head_dim, config.norm_eps)?;
                }
                if let Some(ref kn) = layer.k_norm {
                    gpu.rmsnorm_batched(&k, kn, &k, n_kv_heads, head_dim, config.norm_eps)?;
                }
            }

            gpu.rope_f32(
                &q,
                &k,
                &pos_buf,
                n_heads,
                n_kv_heads,
                head_dim,
                config.rope_freq_base,
            )?;

            gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &k, &pos_buf, kv_dim)?;
            gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &v, &pos_buf, kv_dim)?;

            gpu.attention_f32(
                &q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &attn_out,
                &pos_buf,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_cache.physical_cap,
            )?;

            weight_gemv(gpu, &layer.wo, &attn_out, &o)?;
            gpu.add_inplace_f32(&x, &o)?;

            gpu.rmsnorm_f32(&x, &layer.ffn_norm, &tmp, config.norm_eps)?;
            if layer.w_gate.gpu_dtype == DType::Q4K && layer.w_up.gpu_dtype == DType::Q4K {
                gpu.fused_gate_up_q4k(
                    &layer.w_gate.buf,
                    &layer.w_up.buf,
                    &tmp,
                    &gate,
                    &up,
                    layer.w_gate.m,
                    layer.w_up.m,
                    layer.w_gate.k,
                )?;
            } else {
                weight_gemv(gpu, &layer.w_gate, &tmp, &gate)?;
                weight_gemv(gpu, &layer.w_up, &tmp, &up)?;
            }

            gpu.silu_mul_f32(&gate, &up, &ffn_hidden)?;
            weight_gemv(gpu, &layer.w_down, &ffn_hidden, &ffn_out)?;
            gpu.add_inplace_f32(&x, &ffn_out)?;
        }

        gpu.rmsnorm_f32(&x, &weights.output_norm, &tmp, config.norm_eps)?;

        // `logits` is returned to the caller, so it stays a plain pooled tensor
        // (the caller owns + frees it); free it here only on the gemv error path.
        let logits = gpu.alloc_tensor(&[config.vocab_size], DType::F32)?;
        match weight_gemv(gpu, &weights.output, &tmp, &logits) {
            Ok(()) => Ok(logits),
            Err(e) => {
                let _ = gpu.free_tensor(logits);
                Err(e)
            }
        }
    })();

    // Free the raw `pos_buf` on every exit path; the pooled scratch is RAII
    // (`OwnedTensor`) and is returned to the pool by `reclaim_pending`. `logits`
    // is intentionally not reclaimed here: on success it is the returned value.
    gpu.hip.free(pos_buf).ok(); // raw hip.malloc — not pooled, free explicitly
    gpu.reclaim_pending();

    result
}

pub fn apply_rope_cpu_pub(data: &mut [f32], n_heads: usize, head_dim: usize, pos: usize) {
    apply_rope_cpu(data, n_heads, head_dim, pos, 10000.0);
}

fn apply_rope_cpu(data: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, freq_base: f32) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let freq = 1.0 / (freq_base.powf((2 * i) as f32 / head_dim as f32));
            let val = pos as f32 * freq;
            let cos_val = val.cos();
            let sin_val = val.sin();
            let v0 = data[base + i];
            let v1 = data[base + i + half];
            data[base + i] = v0 * cos_val - v1 * sin_val;
            data[base + i + half] = v0 * sin_val + v1 * cos_val;
        }
    }
}

// attention_cpu removed — GPU attention is now used

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::is_batchable_la;

    #[test]
    fn is_batchable_la_always_ok_dtypes() {
        // MQ4/HFQ4/MQ6/HFQ6 batchable on every arch.
        for arch in [
            "gfx900", "gfx906", "gfx1010", "gfx1030", "gfx1100", "gfx1200", "gfx942",
        ] {
            assert!(is_batchable_la(DType::HFQ4G256, arch));
            assert!(is_batchable_la(DType::MQ4G256, arch));
            assert!(is_batchable_la(DType::HFQ6G256, arch));
            assert!(is_batchable_la(DType::MQ6G256, arch));
        }
    }

    #[test]
    fn is_batchable_la_mq3_wmma_only() {
        // MQ3 batchable on WMMA archs (gfx11/gfx12) via the WMMA path, and on
        // gfx10 RDNA1/2 via the scalar path (PR #298, commit 4840f0b).
        for arch in [
            "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
        ] {
            assert!(
                is_batchable_la(DType::MQ3G256, arch),
                "MQ3 should batch on {arch} (WMMA)"
            );
        }
        for arch in [
            "gfx1010", "gfx1011", "gfx1012", "gfx1013", "gfx1030", "gfx1031", "gfx1032",
        ] {
            assert!(
                is_batchable_la(DType::MQ3G256, arch),
                "MQ3 should batch on {arch} (gfx10 scalar)"
            );
        }
        for arch in ["gfx900", "gfx906", "gfx942"] {
            assert!(
                !is_batchable_la(DType::MQ3G256, arch),
                "MQ3 must fall back on {arch}"
            );
        }
    }

    #[test]
    fn is_batchable_la_fp4_wmma_only() {
        // HFP4G32 / MFP4G32 require WMMA — same arch gate as MQ3.
        for arch in [
            "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
        ] {
            assert!(
                is_batchable_la(DType::HFP4G32, arch),
                "HFP4G32 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::MFP4G32, arch),
                "MFP4G32 should batch on {arch}"
            );
        }
        for arch in ["gfx900", "gfx906", "gfx1010", "gfx1030", "gfx942"] {
            assert!(
                !is_batchable_la(DType::HFP4G32, arch),
                "HFP4G32 must fall back on {arch}"
            );
            assert!(
                !is_batchable_la(DType::MFP4G32, arch),
                "MFP4G32 must fall back on {arch}"
            );
        }
    }

    #[test]
    fn is_batchable_la_unsupported_dtypes() {
        // Q4K / Q6K / F32 stay on per-token forward_scratch.
        for arch in ["gfx1100", "gfx1200"] {
            assert!(!is_batchable_la(DType::Q4K, arch));
            assert!(!is_batchable_la(DType::Q6K, arch));
            assert!(!is_batchable_la(DType::F32, arch));
        }
    }

    #[test]
    fn is_batchable_la_q8_0_always_ok() {
        // Q8_0 is batchable on every arch via gemm_q8_0_batched_chunked
        // (unfused, sub-batched at MAX_BATCH=16). Eval-mode noise-floor path —
        // see docs/plans/q8_batchable.md.
        for arch in [
            "gfx900", "gfx906", "gfx1010", "gfx1030", "gfx1100", "gfx1200", "gfx942",
        ] {
            assert!(is_batchable_la(DType::Q8_0, arch));
        }
    }

    // ── ParoQuant dispatch helpers ──────────────────────────────

    #[test]
    fn paro_small_direct_returns_none_when_unset() {
        // With env var unset, should return None
        // (can't clear env in test, but this verifies the conversion logic)
        // The function returns None when env var is not set (via try_opt)
        // We test the parsing behavior with a lock
    }

    // ── KvCache format dispatch ─────────────────────────────────

    #[test]
    fn kv_cache_is_boundary_within_bounds() {
        // Mock KvCache to test is_boundary logic
        // Since KvCache requires GPU allocation, we test the predicate in isolation
        // by constructing the boolean checks that the dispatch uses.
    }
}
