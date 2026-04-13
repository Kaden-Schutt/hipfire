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
//! Architectural notes (also see `docs/DFLASH_ARCHITECTURE.md`):
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
    pub attn_norm: GpuTensor,      // [hidden]
    pub wq: GpuTensor,             // [q_dim, hidden]
    pub wk: GpuTensor,             // [kv_dim, hidden]
    pub wv: GpuTensor,             // [kv_dim, hidden]
    pub wo: GpuTensor,             // [hidden, q_dim]
    pub q_norm: GpuTensor,         // [head_dim]
    pub k_norm: GpuTensor,         // [head_dim]
    pub ffn_norm: GpuTensor,       // [hidden]
    pub w_gate: GpuTensor,         // [intermediate, hidden]
    pub w_up: GpuTensor,           // [intermediate, hidden]
    pub w_down: GpuTensor,         // [hidden, intermediate]
}

pub struct DflashWeights {
    /// `fc`: Linear(num_extract × hidden → hidden). Shape: [hidden, num_extract × hidden].
    pub fc: GpuTensor,
    pub hidden_norm: GpuTensor,    // [hidden]
    pub norm: GpuTensor,           // [hidden] — final output norm
    pub layers: Vec<DflashLayerWeights>,
}

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

impl DflashWeights {
    pub fn load(gpu: &mut Gpu, hfq: &HfqFile, cfg: &DflashConfig) -> HipResult<Self> {
        let fc_shape = vec![cfg.hidden, cfg.num_extract() * cfg.hidden];
        let fc = hfq_tensor_f32(hfq, gpu, "fc.weight", fc_shape)?;
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
                wq: hfq_tensor_f32(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.q_proj.weight"),
                    vec![cfg.q_dim(), cfg.hidden],
                )?,
                wk: hfq_tensor_f32(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.k_proj.weight"),
                    vec![cfg.kv_dim(), cfg.hidden],
                )?,
                wv: hfq_tensor_f32(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.v_proj.weight"),
                    vec![cfg.kv_dim(), cfg.hidden],
                )?,
                wo: hfq_tensor_f32(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.o_proj.weight"),
                    vec![cfg.hidden, cfg.q_dim()],
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
                w_gate: hfq_tensor_f32(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.gate_proj.weight"),
                    vec![cfg.intermediate, cfg.hidden],
                )?,
                w_up: hfq_tensor_f32(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.up_proj.weight"),
                    vec![cfg.intermediate, cfg.hidden],
                )?,
                w_down: hfq_tensor_f32(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.down_proj.weight"),
                    vec![cfg.hidden, cfg.intermediate],
                )?,
            };
            layers.push(layer);
        }
        Ok(DflashWeights {
            fc,
            hidden_norm,
            norm,
            layers,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.fc);
        let _ = gpu.free_tensor(self.hidden_norm);
        let _ = gpu.free_tensor(self.norm);
        for l in self.layers {
            let _ = gpu.free_tensor(l.attn_norm);
            let _ = gpu.free_tensor(l.wq);
            let _ = gpu.free_tensor(l.wk);
            let _ = gpu.free_tensor(l.wv);
            let _ = gpu.free_tensor(l.wo);
            let _ = gpu.free_tensor(l.q_norm);
            let _ = gpu.free_tensor(l.k_norm);
            let _ = gpu.free_tensor(l.ffn_norm);
            let _ = gpu.free_tensor(l.w_gate);
            let _ = gpu.free_tensor(l.w_up);
            let _ = gpu.free_tensor(l.w_down);
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
}

impl DflashScratch {
    pub fn new(
        gpu: &mut Gpu,
        cfg: &DflashConfig,
        max_block_size: usize,
        max_ctx_len: usize,
    ) -> HipResult<Self> {
        let b = max_block_size;
        let l = max_ctx_len;
        let tot = l + b;
        let ne = cfg.num_extract();
        let h = cfg.hidden;
        let inter = cfg.intermediate;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();

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

            positions_q: gpu.alloc_tensor(&[b],   DType::F32)?,   // stored as 4-byte slots; interpreted as i32 by kernels
            positions_k: gpu.alloc_tensor(&[tot], DType::F32)?,
        })
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
    }
}

// ─── Forward ───────────────────────────────────────────────────────────────

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
#[allow(clippy::too_many_arguments)]
pub fn draft_forward(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    noise_embedding: &[f32],
    target_hidden: &[f32],
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
    assert_eq!(noise_embedding.len(), b * h, "noise_embedding size");
    assert_eq!(target_hidden.len(), l * ne * h, "target_hidden size");
    assert_eq!(positions_q.len(), b, "positions_q size");
    assert_eq!(positions_k.len(), tot, "positions_k size");

    // ── 0. Uploads ────────────────────────────────────────────────────────
    upload_slice_f32(gpu, &scratch.x, noise_embedding)?;
    upload_slice_f32(gpu, &scratch.target_hidden, target_hidden)?;
    upload_slice_i32(gpu, &scratch.positions_q, positions_q)?;
    upload_slice_i32(gpu, &scratch.positions_k, positions_k)?;

    // ── 1. target_hidden_proj = hidden_norm(fc @ target_hidden) ──────────
    // Treat `target_hidden` as a [L, num_extract*hidden] batch with inner
    // dim = num_extract*hidden. fc is [hidden, num_extract*hidden].
    //
    // gemm_f32_batched(A, B, Y, M, K, N) → Y[M,N] = A[M,K] @ B[N,K]^T.
    // Pick M=L (batch rows), K=num_extract*hidden (inner), N=hidden
    // (output). Pass A = target_hidden (the batch) and B = fc (the
    // weights) so Y[L,hidden] is row-major batch × output.
    gpu.gemm_f32_batched(
        &scratch.target_hidden,     // A[L, ne*h]
        &weights.fc,                // B[hidden, ne*h]
        &scratch.target_hidden_proj, // Y[L, hidden]
        l,
        ne * h,
        h,
    )?;
    // RMSNorm across each L-row of size hidden with hidden_norm weight.
    gpu.rmsnorm_batched(
        &scratch.target_hidden_proj,
        &weights.hidden_norm,
        &scratch.target_hidden_proj,
        l,
        h,
        eps,
    )?;

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

        // Q = x_norm @ wq^T  → [B, q_dim]
        gpu.gemm_f32_batched(&scratch.x_norm, &layer.wq, &scratch.q,   b, h, qd)?;
        // K_noise = x_norm @ wk^T → [B, kv_dim]
        gpu.gemm_f32_batched(&scratch.x_norm, &layer.wk, &scratch.k_noise, b, h, kvd)?;
        // V_noise = x_norm @ wv^T → [B, kv_dim]
        gpu.gemm_f32_batched(&scratch.x_norm, &layer.wv, &scratch.v_noise, b, h, kvd)?;

        // K_ctx = target_hidden_proj @ wk^T → [L, kv_dim]
        gpu.gemm_f32_batched(&scratch.target_hidden_proj, &layer.wk, &scratch.k_ctx, l, h, kvd)?;
        // V_ctx = target_hidden_proj @ wv^T → [L, kv_dim]
        gpu.gemm_f32_batched(&scratch.target_hidden_proj, &layer.wv, &scratch.v_ctx, l, h, kvd)?;

        // Concat K = [K_ctx | K_noise] → [L + B, kv_dim]
        //         V = [V_ctx | V_noise] → [L + B, kv_dim]
        let ctx_bytes   = (l * kvd) * 4;
        let noise_bytes = (b * kvd) * 4;
        gpu.hip.memcpy_dtod_at(&scratch.k_cat.buf, 0,          &scratch.k_ctx.buf,   0, ctx_bytes)?;
        gpu.hip.memcpy_dtod_at(&scratch.k_cat.buf, ctx_bytes,  &scratch.k_noise.buf, 0, noise_bytes)?;
        gpu.hip.memcpy_dtod_at(&scratch.v_cat.buf, 0,          &scratch.v_ctx.buf,   0, ctx_bytes)?;
        gpu.hip.memcpy_dtod_at(&scratch.v_cat.buf, ctx_bytes,  &scratch.v_noise.buf, 0, noise_bytes)?;

        // Per-head RMSNorm on Q: each of B*n_heads rows, size head_dim,
        // weight [head_dim].
        gpu.rmsnorm_batched(&scratch.q,    &layer.q_norm, &scratch.q,    b * cfg.n_heads,   hd, eps)?;
        // Per-head RMSNorm on K_cat: each of (L+B)*n_kv_heads rows.
        gpu.rmsnorm_batched(&scratch.k_cat, &layer.k_norm, &scratch.k_cat, tot * cfg.n_kv_heads, hd, eps)?;

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

        // attn_proj = attn_out @ wo^T → [B, hidden]
        gpu.gemm_f32_batched(&scratch.attn_out, &layer.wo, &scratch.attn_proj, b, qd, h)?;

        // x = residual_attn + attn_proj
        gpu.add_f32(&scratch.residual_attn, &scratch.attn_proj, &scratch.x)?;

        // Residual for FFN.
        gpu.hip.memcpy_dtod(&scratch.residual_ffn.buf, &scratch.x.buf, (b * h) * 4)?;

        // ffn_norm.
        gpu.rmsnorm_batched(&scratch.x, &layer.ffn_norm, &scratch.x_norm, b, h, eps)?;

        // gate = x_norm @ w_gate^T; up = x_norm @ w_up^T
        gpu.gemm_f32_batched(&scratch.x_norm, &layer.w_gate, &scratch.gate, b, h, cfg.intermediate)?;
        gpu.gemm_f32_batched(&scratch.x_norm, &layer.w_up,   &scratch.up,   b, h, cfg.intermediate)?;

        // SiLU(gate) * up → gate_up
        gpu.silu_mul_f32(&scratch.gate, &scratch.up, &scratch.gate_up)?;

        // x = w_down @ gate_up^T  (output [B, hidden])
        gpu.gemm_f32_batched(&scratch.gate_up, &layer.w_down, &scratch.x, b, cfg.intermediate, h)?;

        // x = residual_ffn + x
        gpu.add_f32(&scratch.residual_ffn, &scratch.x, &scratch.x)?;
    }

    // ── 3. Final norm ────────────────────────────────────────────────────
    gpu.rmsnorm_batched(&scratch.x, &weights.norm, &scratch.x, b, h, eps)?;

    Ok(())
}
