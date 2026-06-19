// SPDX-License-Identifier: Apache-2.0
// hipfire — Gemma3 single-token decode forward. See LICENSE / NOTICE.

//! `Gemma3State` (per-decode GPU scratch + F32 KV cache) and the per-token
//! `forward_step`. Correctness-first bring-up: direct `gpu.*` kernel calls,
//! greedy/caller-sampled, one token at a time (prefill = N sequential calls),
//! modelled on `hipfire-arch-qwen2::forward_step`.
//!
//! Gemma3 layer body (4 norms; post-norms sit *inside* the residual, so the
//! fused gemv+residual can't be used — gemv → post-norm → explicit add):
//! ```text
//!   resid = x
//!   h = input_layernorm(x);  q,k,v = proj(h)
//!   q = q_norm(q); k = k_norm(k)              # per-head, q_norm carries the Q pre-scale
//!   rope(q,k, θ = global or local per layer)
//!   attn_out = GQA_attention(q,k,v, kv$);  o = o_proj(attn_out)
//!   x = resid + post_attention_layernorm(o)
//!   resid = x
//!   h = pre_feedforward_layernorm(x);  g,u = gate/up(h);  ffn = gelu_mul(g,u)
//!   o = down(ffn)
//!   x = resid + post_feedforward_layernorm(o)
//! ```
//! Embedding is scaled by √hidden_size before layer 0; norms are `(1+w)`-baked
//! at ingest so the plain rmsnorm kernel is correct. The attention kernel's
//! built-in `1/√head_dim` is corrected to `1/√query_pre_attn_scalar` by the Q
//! pre-scale baked into `q_norm` (see `load_weights`).

use hip_bridge::{DeviceBuffer, HipResult};
use hipfire_runtime::llama::{weight_gemv, EmbeddingFormat};
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::config::Gemma3Config;
use crate::weights::Gemma3Weights;

/// Default KV budget for bring-up validation (slots).
pub const DEFAULT_MAX_SEQ: usize = 4096;

/// Per-decode GPU scratch + F32 KV cache. `tmp` (size `hidden_size`) is reused
/// for every norm output; `o` (size `hidden_size`) holds both the attn and FFN
/// projection outputs before their post-norm + residual add.
pub struct Gemma3State {
    pub x: GpuTensor,          // residual stream [hidden]
    pub tmp: GpuTensor,        // norm-output scratch [hidden]
    pub q: GpuTensor,          // [n_heads*head_dim]
    pub k: GpuTensor,          // [n_kv*head_dim]
    pub v: GpuTensor,          // [n_kv*head_dim]
    pub attn_out: GpuTensor,   // [n_heads*head_dim]
    pub o: GpuTensor,          // projection output [hidden]
    pub gate: GpuTensor,       // [intermediate]
    pub up: GpuTensor,         // [intermediate]
    pub ffn_hidden: GpuTensor, // [intermediate]
    pub logits: GpuTensor,     // [vocab]
    pub pos_buf: DeviceBuffer,
    pub k_cache: Vec<GpuTensor>,
    pub v_cache: Vec<GpuTensor>,
    pub max_seq: usize,
    /// Next absolute KV write slot; bumped by `forward_step`.
    pub next_pos: usize,
}

impl Gemma3State {
    pub fn new(gpu: &mut Gpu, cfg: &Gemma3Config) -> Result<Self, String> {
        Self::new_with_max_seq(gpu, cfg, DEFAULT_MAX_SEQ)
            .map_err(|e| format!("gemma3: Gemma3State::new failed: {e:?}"))
    }

    pub fn new_with_max_seq(gpu: &mut Gpu, cfg: &Gemma3Config, max_seq: usize) -> HipResult<Self> {
        let dim = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let hidden_dim = cfg.intermediate_size;

        let mut k_cache = Vec::with_capacity(cfg.num_hidden_layers);
        let mut v_cache = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            k_cache.push(gpu.zeros(&[max_seq * kv_dim], DType::F32)?);
            v_cache.push(gpu.zeros(&[max_seq * kv_dim], DType::F32)?);
        }

        Ok(Self {
            x: gpu.alloc_tensor(&[dim], DType::F32)?,
            tmp: gpu.alloc_tensor(&[dim], DType::F32)?,
            q: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            k: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            v: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            attn_out: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            o: gpu.alloc_tensor(&[dim], DType::F32)?,
            gate: gpu.alloc_tensor(&[hidden_dim], DType::F32)?,
            up: gpu.alloc_tensor(&[hidden_dim], DType::F32)?,
            ffn_hidden: gpu.alloc_tensor(&[hidden_dim], DType::F32)?,
            logits: gpu.alloc_tensor(&[cfg.vocab_size], DType::F32)?,
            pos_buf: gpu.hip.malloc(4)?,
            k_cache,
            v_cache,
            max_seq,
            next_pos: 0,
        })
    }

    /// Rewind to position 0 (fresh conversation). KV slots are overwritten in
    /// place, so this is O(1).
    pub fn reset(&mut self) {
        self.next_pos = 0;
    }

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
            self.logits,
        ] {
            let _ = gpu.free_tensor(t);
        }
        for t in self.k_cache {
            let _ = gpu.free_tensor(t);
        }
        for t in self.v_cache {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.hip.free(self.pos_buf);
    }
}

fn prelude(gpu: &mut Gpu, state: &Gemma3State) -> HipResult<usize> {
    let pos = state.next_pos;
    if pos >= state.max_seq {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "gemma3: forward_step pos={pos} >= max_seq={}; rebuild Gemma3State \
                 with a larger budget",
                state.max_seq
            ),
        ));
    }
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(pos as i32).to_ne_bytes())?;
    Ok(pos)
}

/// Single-token decode: read `token` at `state.next_pos`, run the full stack,
/// write K/V at that position, leave logits in `state.logits`, bump `next_pos`.
pub fn forward_step(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    cfg: &Gemma3Config,
    state: &mut Gemma3State,
    token: u32,
) -> HipResult<()> {
    let pos = prelude(gpu, state)?;
    let dim = cfg.hidden_size;

    // Embedding lookup → x.
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &state.x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &state.x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8(&weights.token_embd, &state.x, token, dim)?
        }
        EmbeddingFormat::Q4K => {
            gpu.embedding_lookup_q4k(&weights.token_embd, &state.x, token, dim)?
        }
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &state.x, token, dim)?,
    }
    // Gemma scales the embedding by √hidden_size before the first layer; the
    // scale rides the residual stream (it is NOT normalized away — rmsnorm
    // cancels it locally but each residual add re-injects it).
    gpu.scale_f32(&state.x, cfg.embed_scale())?;

    forward_after_x(gpu, weights, cfg, state, pos)?;
    state.next_pos += 1;
    Ok(())
}

/// Decode one position from a **prebuilt embedding** instead of an embedded
/// token — the image-token splice primitive for gemma3-vl. The multimodal
/// projector output already lives in the text embedding space and is inserted
/// into the (already-scaled) text stream **unscaled**, so this path does NOT
/// apply the `√hidden` embed scale. `embedding` is one row of `hidden_size`
/// F32s. Mirrors `hipfire-arch-qwen2::forward_step_with_embed`.
pub fn forward_step_with_embed(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    cfg: &Gemma3Config,
    state: &mut Gemma3State,
    embedding: &[f32],
) -> HipResult<()> {
    let dim = cfg.hidden_size;
    if embedding.len() != dim {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "gemma3: forward_step_with_embed expects {dim} F32s, got {}",
                embedding.len()
            ),
        ));
    }
    let pos = prelude(gpu, state)?;
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(embedding.as_ptr() as *const u8, embedding.len() * 4) };
    gpu.hip.memcpy_htod(&state.x.buf, bytes)?;
    // NB: no embed_scale here — the image embedding is inserted at its own
    // magnitude (only token embeddings get the √hidden normalizer).
    forward_after_x(gpu, weights, cfg, state, pos)?;
    state.next_pos += 1;
    Ok(())
}

fn forward_after_x(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    cfg: &Gemma3Config,
    state: &mut Gemma3State,
    pos: usize,
) -> HipResult<()> {
    let n_heads = cfg.num_attention_heads;
    let n_kv_heads = cfg.num_key_value_heads;
    let head_dim = cfg.head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let eps = cfg.rms_norm_eps;

    for layer_idx in 0..cfg.num_hidden_layers {
        let layer = &weights.layers[layer_idx];

        // ── Attention block ──────────────────────────────────────────
        gpu.rmsnorm_f32(&state.x, &layer.input_norm, &state.tmp, eps)?;
        weight_gemv(gpu, &layer.wq, &state.tmp, &state.q)?;
        weight_gemv(gpu, &layer.wk, &state.tmp, &state.k)?;
        weight_gemv(gpu, &layer.wv, &state.tmp, &state.v)?;

        // Per-head QK-norm (q_norm carries the baked Q pre-scale).
        Gpu::rmsnorm_batched(
            gpu,
            &state.q,
            &layer.q_norm,
            &state.q,
            n_heads,
            head_dim,
            eps,
        )?;
        Gpu::rmsnorm_batched(
            gpu,
            &state.k,
            &layer.k_norm,
            &state.k,
            n_kv_heads,
            head_dim,
            eps,
        )?;

        // Dual-θ RoPE: global layers use rope_theta, local use rope_local_base_freq.
        gpu.rope_f32(
            &state.q,
            &state.k,
            &state.pos_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            cfg.rope_base_for_layer(layer_idx),
        )?;

        gpu.kv_cache_write(&state.k_cache[layer_idx], &state.k, &state.pos_buf, kv_dim)?;
        gpu.kv_cache_write(&state.v_cache[layer_idx], &state.v, &state.pos_buf, kv_dim)?;

        // GQA attention (full causal; sliding-window mask deferred — only
        // affects ctx > sliding_window, see the bring-up plan).
        Gpu::attention_f32(
            gpu,
            &state.q,
            &state.k_cache[layer_idx],
            &state.v_cache[layer_idx],
            &state.attn_out,
            &state.pos_buf,
            pos + 1,
            n_heads,
            n_kv_heads,
            head_dim,
            state.max_seq,
        )?;

        weight_gemv(gpu, &layer.wo, &state.attn_out, &state.o)?;
        // post_attention_layernorm sits INSIDE the residual: norm(o) then add.
        gpu.rmsnorm_f32(&state.o, &layer.post_attn_norm, &state.tmp, eps)?;
        gpu.add_f32(&state.x, &state.tmp, &state.x)?;

        // ── FFN block (GeGLU) ────────────────────────────────────────
        gpu.rmsnorm_f32(&state.x, &layer.pre_ffn_norm, &state.tmp, eps)?;
        weight_gemv(gpu, &layer.w_gate, &state.tmp, &state.gate)?;
        weight_gemv(gpu, &layer.w_up, &state.tmp, &state.up)?;
        gpu.gelu_mul_f32(&state.gate, &state.up, &state.ffn_hidden)?;
        weight_gemv(gpu, &layer.w_down, &state.ffn_hidden, &state.o)?;
        // post_feedforward_layernorm, also inside the residual.
        gpu.rmsnorm_f32(&state.o, &layer.post_ffn_norm, &state.tmp, eps)?;
        gpu.add_f32(&state.x, &state.tmp, &state.x)?;
    }

    // Final norm + lm_head.
    gpu.rmsnorm_f32(&state.x, &weights.output_norm, &state.tmp, eps)?;
    weight_gemv(gpu, &weights.output, &state.tmp, &state.logits)?;
    Ok(())
}

/// Greedy variant: run a step, then return argmax of the resulting logits.
pub fn forward_step_greedy(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    cfg: &Gemma3Config,
    state: &mut Gemma3State,
    token: u32,
) -> HipResult<u32> {
    forward_step(gpu, weights, cfg, state, token)?;
    gpu.argmax_f32(&state.logits, cfg.vocab_size)
}
