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
use hipfire_runtime::weights::{weight_gemm, weight_gemv, EmbeddingFormat, WeightTensor};
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
    /// When true, `k_cache`/`v_cache` hold q8_0 (int8 + per-32-block fp16 scale)
    /// instead of F32 — ~4× smaller KV, letting larger contexts fit. Requires
    /// `head_dim % 32 == 0` (true for gemma3: 128 @27b, 256 @4b).
    pub kv_quant_q8: bool,
    /// Next absolute KV write slot; bumped by `forward_step`.
    pub next_pos: usize,
}

impl Gemma3State {
    pub fn new(gpu: &mut Gpu, cfg: &Gemma3Config) -> Result<Self, String> {
        Self::new_with_max_seq(gpu, cfg, DEFAULT_MAX_SEQ, false)
            .map_err(|e| format!("gemma3: Gemma3State::new failed: {e:?}"))
    }

    pub fn new_with_max_seq(
        gpu: &mut Gpu,
        cfg: &Gemma3Config,
        max_seq: usize,
        quant_q8: bool,
    ) -> HipResult<Self> {
        let dim = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let hidden_dim = cfg.intermediate_size;

        // q8_0 KV requires head_dim divisible by the 32-element block; fall back
        // to F32 otherwise. K/V cache slots are q8_0 blocks (32 int8 + fp16
        // scale = 34 bytes each), allocated as an F32 tensor sized by bytes.
        let kv_quant_q8 = quant_q8 && cfg.head_dim % 32 == 0;
        let mut k_cache = Vec::with_capacity(cfg.num_hidden_layers);
        let mut v_cache = Vec::with_capacity(cfg.num_hidden_layers);
        let cache_elems = if kv_quant_q8 {
            let total_blocks = cfg.num_key_value_heads * (cfg.head_dim / 32);
            (max_seq * total_blocks * 34).div_ceil(4)
        } else {
            max_seq * kv_dim
        };
        for _ in 0..cfg.num_hidden_layers {
            k_cache.push(gpu.zeros(&[cache_elems], DType::F32)?);
            v_cache.push(gpu.zeros(&[cache_elems], DType::F32)?);
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
            kv_quant_q8,
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

    // Embedding lookup + Gemma √hidden scale → x.
    embed_token(gpu, weights, cfg, &state.x, token)?;

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

/// Debug: when `HIPFIRE_LM_DUMP=<dir>` is set, write a decoder-stage tensor to
/// `<dir>/<name>.bin` (f32 LE) + `<dir>/<name>.json` for validation against an
/// HF reference (benchmarks/vision/diff_dumps.py). Called per forward, so for a
/// per-token prefill the LAST call's tensors (= last prompt position) survive.
/// No-op when unset.
fn maybe_dump_lm(gpu: &mut Gpu, t: &GpuTensor, name: &str, shape: &[usize]) {
    let dir = match std::env::var("HIPFIRE_LM_DUMP") {
        Ok(d) if !d.is_empty() => d,
        _ => return,
    };
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    if let Ok(data) = gpu.download_f32(t) {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let _ = std::fs::write(format!("{dir}/{name}.bin"), bytes);
        let _ = std::fs::write(
            format!("{dir}/{name}.json"),
            format!("{{\"shape\":{shape:?}}}"),
        );
    }
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

        // GQA attention (full causal; sliding-window mask deferred — only
        // affects ctx > sliding_window, see the bring-up plan). KV is stored
        // F32 or q8_0 depending on the state's quant mode.
        if state.kv_quant_q8 {
            gpu.kv_cache_write_q8_0(
                &state.k_cache[layer_idx],
                &state.k,
                &state.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.kv_cache_write_q8_0(
                &state.v_cache[layer_idx],
                &state.v,
                &state.pos_buf,
                n_kv_heads,
                head_dim,
            )?;
            gpu.attention_q8_0_kv(
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
        } else {
            gpu.kv_cache_write(&state.k_cache[layer_idx], &state.k, &state.pos_buf, kv_dim)?;
            gpu.kv_cache_write(&state.v_cache[layer_idx], &state.v, &state.pos_buf, kv_dim)?;
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
        }

        weight_gemv(gpu, &layer.wo, &state.attn_out, &state.o)?;
        // post_attention_layernorm sits INSIDE the residual: norm(o) then add.
        gpu.rmsnorm_f32(&state.o, &layer.post_attn_norm, &state.tmp, eps)?;
        gpu.add_f32(&state.x, &state.tmp, &state.x)?;

        // ── FFN block (GeGLU) ────────────────────────────────────────
        gpu.rmsnorm_f32(&state.x, &layer.pre_ffn_norm, &state.tmp, eps)?;
        weight_gemv(gpu, &layer.w_gate, &state.tmp, &state.gate)?;
        weight_gemv(gpu, &layer.w_up, &state.tmp, &state.up)?;
        gpu.gelu_mul_f32(&state.gate, &state.up, &state.ffn_hidden)?;
        // H-Neuron intervention gain (no-op unless a session is active): scale the
        // down_proj input in place before down. Decode is a single position.
        hipfire_hneurons::intervene::maybe_intervene_ffn(gpu, &state.ffn_hidden, layer_idx, 1)?;
        weight_gemv(gpu, &layer.w_down, &state.ffn_hidden, &state.o)?;
        // post_feedforward_layernorm, also inside the residual.
        gpu.rmsnorm_f32(&state.o, &layer.post_ffn_norm, &state.tmp, eps)?;
        gpu.add_f32(&state.x, &state.tmp, &state.x)?;
        // Block-boundary steering/abliteration hook (no-op unless a session is
        // active). `state.x` is the settled post-residual stream — fusion-proof.
        hipfire_steer::maybe_steer_block(gpu, &state.x, layer_idx)?;
        maybe_dump_lm(
            gpu,
            &state.x,
            &format!("lm_block_{layer_idx:02}"),
            &[cfg.hidden_size],
        );
    }

    // Final norm + lm_head.
    gpu.rmsnorm_f32(&state.x, &weights.output_norm, &state.tmp, eps)?;
    weight_gemv(gpu, &weights.output, &state.tmp, &state.logits)?;
    maybe_dump_lm(gpu, &state.logits, "lm_logits", &[cfg.vocab_size]);
    Ok(())
}

/// Embed one text token (format-dispatched lookup + Gemma √hidden scale) into
/// `dest` (`[dim]`). The embedding step shared by `forward_step` and the
/// batched-prefill input builder. Image rows bypass this (spliced unscaled).
pub fn embed_token(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    cfg: &Gemma3Config,
    dest: &GpuTensor,
    token: u32,
) -> HipResult<()> {
    let dim = cfg.hidden_size;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, dest, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, dest, token, dim)?
        }
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, dest, token, dim)?,
        EmbeddingFormat::Q4K => gpu.embedding_lookup_q4k(&weights.token_embd, dest, token, dim)?,
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, dest, token, dim)?,
    }
    // Gemma scales the embedding by √hidden_size before the first layer; the
    // scale rides the residual stream (rmsnorm cancels it locally but each
    // residual add re-injects it).
    gpu.scale_f32(dest, cfg.embed_scale())?;
    Ok(())
}

/// Prefill linear for the batched gemma3 path. Routes Q8_0 weights through the
/// no-LDS WMMA kernel (`gemm_q8_0_wmma`, ~11–30× over the scalar
/// `gemm_q8_0_batched` that `weight_gemm`'s chunked driver otherwise selects on
/// non-RDNA4 arches) whenever the GPU has w32 WMMA (RDNA3/3.5/4) and `K % 32 ==
/// 0`; anything else falls back to the generic `weight_gemm` dispatch. Preserves
/// the imatrix/Hessian activation tap `weight_gemm` performs so batched
/// calibration still works on gemma3. Portability: RDNA2 (no WMMA) stays on the
/// scalar path via the fallback, and the kernel is register-tiled (no LDS), so
/// it is safe on the gfx1103 LDS-fault class.
fn prefill_linear(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
) -> HipResult<()> {
    if w.gpu_dtype == DType::Q8_0 && gpu.arch_caps.has_wmma_w32() && w.k % 32 == 0 {
        // Mirror `weight_gemm`'s batched calibration tap before dispatch.
        gpu.maybe_capture_activation(&w.buf, x, m, w.k);
        gpu.gemm_q8_0_wmma(&w.buf, x, y, w.m, w.k, m)
    } else {
        weight_gemm(gpu, w, x, y, m)
    }
}

/// Batched prefill of `m` already-embedded tokens. `x_batch` is `[m, dim]`
/// row-major — the caller embeds text tokens (×`embed_scale`) and splices image
/// rows (unscaled) just like the per-token path. Writes KV for absolute
/// positions `start_pos..start_pos+m`, advances `state.next_pos`, and leaves the
/// LAST position's logits in `state.logits`. Numerically equivalent to running
/// `m` sequential `forward_step`s (both full-causal) but reads each weight once
/// instead of `m` times — the multi-image / long-prompt prefill speedup.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    cfg: &Gemma3Config,
    state: &mut Gemma3State,
    x_batch: &GpuTensor,
    m: usize,
    start_pos: usize,
) -> HipResult<()> {
    let dim = cfg.hidden_size;
    let n_heads = cfg.num_attention_heads;
    let n_kv_heads = cfg.num_key_value_heads;
    let head_dim = cfg.head_dim;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let inter = cfg.intermediate_size;
    let eps = cfg.rms_norm_eps;
    let max_ctx = start_pos + m;

    // Absolute positions start_pos..start_pos+m as an i32 device table (dtype is
    // cosmetic — the rope/attention/kv kernels read it as `const int*`).
    let pos_vals: Vec<i32> = (start_pos as i32..(start_pos + m) as i32).collect();
    let positions = gpu.alloc_owned(&[m], DType::F32)?;
    {
        let bytes: Vec<u8> = pos_vals.iter().flat_map(|p| p.to_ne_bytes()).collect();
        gpu.hip.memcpy_htod(&positions.buf, &bytes)?;
    }

    // Batched scratch. `OwnedTensor` frees itself back to the pool on drop, on
    // EVERY exit path (`?` errors and panics included).
    let tmp = gpu.alloc_owned(&[m * dim], DType::F32)?;
    let q = gpu.alloc_owned(&[m * q_dim], DType::F32)?;
    let k = gpu.alloc_owned(&[m * kv_dim], DType::F32)?;
    let v = gpu.alloc_owned(&[m * kv_dim], DType::F32)?;
    let attn_out = gpu.alloc_owned(&[m * q_dim], DType::F32)?;
    let o = gpu.alloc_owned(&[m * dim], DType::F32)?;
    let gate = gpu.alloc_owned(&[m * inter], DType::F32)?;
    let up = gpu.alloc_owned(&[m * inter], DType::F32)?;
    let ffn = gpu.alloc_owned(&[m * inter], DType::F32)?;

    for layer_idx in 0..cfg.num_hidden_layers {
        let layer = &weights.layers[layer_idx];

        // ── Attention block ──
        gpu.rmsnorm_batched(x_batch, &layer.input_norm, &tmp, m, dim, eps)?;
        prefill_linear(gpu, &layer.wq, &tmp, &q, m)?;
        prefill_linear(gpu, &layer.wk, &tmp, &k, m)?;
        prefill_linear(gpu, &layer.wv, &tmp, &v, m)?;

        // Per-head QK-norm (q_norm carries the baked Q pre-scale): m*heads groups.
        gpu.rmsnorm_batched(&q, &layer.q_norm, &q, m * n_heads, head_dim, eps)?;
        gpu.rmsnorm_batched(&k, &layer.k_norm, &k, m * n_kv_heads, head_dim, eps)?;

        gpu.rope_batched_f32(
            &q,
            &k,
            &positions,
            n_heads,
            n_kv_heads,
            head_dim,
            cfg.rope_base_for_layer(layer_idx),
            m,
        )?;

        if state.kv_quant_q8 {
            gpu.kv_cache_write_q8_0_batched(
                &state.k_cache[layer_idx],
                &k,
                &positions,
                n_kv_heads,
                head_dim,
                m,
            )?;
            gpu.kv_cache_write_q8_0_batched(
                &state.v_cache[layer_idx],
                &v,
                &positions,
                n_kv_heads,
                head_dim,
                m,
            )?;
            gpu.attention_q8_0_kv_batched(
                &q,
                &state.k_cache[layer_idx],
                &state.v_cache[layer_idx],
                &attn_out,
                &positions,
                n_heads,
                n_kv_heads,
                head_dim,
                state.max_seq,
                max_ctx,
                m,
            )?;
        } else {
            gpu.kv_cache_write_f32_batched(&state.k_cache[layer_idx], &k, &positions, kv_dim, m)?;
            gpu.kv_cache_write_f32_batched(&state.v_cache[layer_idx], &v, &positions, kv_dim, m)?;
            gpu.attention_f32_batched(
                &q,
                &state.k_cache[layer_idx],
                &state.v_cache[layer_idx],
                &attn_out,
                &positions,
                n_heads,
                n_kv_heads,
                head_dim,
                state.max_seq,
                max_ctx,
                m,
            )?;
        }

        prefill_linear(gpu, &layer.wo, &attn_out, &o, m)?;
        gpu.rmsnorm_batched(&o, &layer.post_attn_norm, &tmp, m, dim, eps)?;
        gpu.add_f32(x_batch, &tmp, x_batch)?;

        // ── FFN block (GeGLU) ──
        gpu.rmsnorm_batched(x_batch, &layer.pre_ffn_norm, &tmp, m, dim, eps)?;
        prefill_linear(gpu, &layer.w_gate, &tmp, &gate, m)?;
        prefill_linear(gpu, &layer.w_up, &tmp, &up, m)?;
        gpu.gelu_mul_f32(&gate, &up, &ffn)?;
        // H-Neuron intervention gain (no-op unless active): scale the down_proj
        // input in place before down, across all m positions.
        hipfire_hneurons::intervene::maybe_intervene_ffn(gpu, &ffn, layer_idx, m)?;
        prefill_linear(gpu, &layer.w_down, &ffn, &o, m)?;
        // H-Neurons CETT tap (no-op unless a capture session is active). `o` holds
        // the raw down_proj output here — the residual add below folds
        // post_ffn_norm(o), so both down_proj input (`ffn`) and output (`o`) stay
        // materialized. `start_pos` is the global position of this chunk's row 0.
        hipfire_hneurons::capture::maybe_capture_ffn(gpu, &ffn, &o, layer_idx, start_pos, m)?;
        gpu.rmsnorm_batched(&o, &layer.post_ffn_norm, &tmp, m, dim, eps)?;
        gpu.add_f32(x_batch, &tmp, x_batch)?;
        // Block-boundary steering/abliteration hook (no-op unless active).
        // Prefill convention: capture folds the last position, apply hits all.
        hipfire_steer::maybe_steer_block_batched(gpu, x_batch, layer_idx, m, dim)?;
    }

    // Final norm + lm_head on the LAST position only (the next-token logits).
    gpu.memcpy_dtod_at_auto(&state.x.buf, 0, &x_batch.buf, (m - 1) * dim * 4, dim * 4)?;
    gpu.rmsnorm_f32(&state.x, &weights.output_norm, &state.tmp, eps)?;
    weight_gemv(gpu, &weights.output, &state.tmp, &state.logits)?;

    // The per-call pooled scratch above (`OwnedTensor`) returned itself to the
    // deferred-free mailbox on drop; drain the mailbox at this forward boundary.
    gpu.reclaim_pending();
    state.next_pos = start_pos + m;
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
