//! Qwen3.5 native MTP (Multi-Token Prediction / NextN) head.
//!
//! This module loads the single transformer-decoder block + NextN overlay
//! that ships in Qwen3.5/3.6 dense checkpoints (`mtp.*` tensor namespace),
//! quantized into a `.mtp` file by `crates/hipfire-quantize/src/bin/mtp_extract.rs`
//! (arch_id = 21, `QWEN35_MTP_HEAD`). Loading + forward path are independent
//! of the trunk's full forward — the head consumes a single `prev_hidden`
//! activation produced by the trunk + the next committed token, and emits
//! logits over the full vocab via the trunk's shared `lm_head`.
//!
//! ## Forward (per llama.cpp `qwen35_mtp.cpp` and HF reference):
//!
//! ```text
//! tok_embd = embed[next_token]                 # via trunk's tok_embd
//! e_norm   = RMSNorm(tok_embd, enorm, eps)     # F32 norm weight
//! h_norm   = RMSNorm(prev_hidden, hnorm, eps)
//! cur      = eh_proj @ concat(e_norm, h_norm)  # 2d → d
//! inpSA    = cur                               # save for residual
//! cur      = RMSNorm(cur, attn_norm)
//! Q_full   = wq @ cur                          # 2 * head_dim * n_head
//! Q, gate  = deinterleave(Q_full)              # split per-head
//! Q        = RMSNorm(Q, attn_q_norm)           # per-head
//! K, V     = wk @ cur, wv @ cur
//! K        = RMSNorm(K, attn_k_norm)
//! Q, K     = rope_partial_interleaved(Q, K, pos)  # default RoPE for Qwen3.5
//! kv[pos]  = K, V                              # MTP-private KV cache
//! attn     = attention(Q, kv[..=pos], V_cache, scale = 1/sqrt(head_dim))
//! attn     = sigmoid_mul(attn, gate)           # gated-Q output
//! cur      = wo @ attn + inpSA                 # residual
//! ffn_in   = cur
//! cur      = RMSNorm(cur, attn_post_norm)      # POST-attn norm (NOT pre-FFN)
//! ffn      = ffn_down(silu(ffn_gate @ cur) * (ffn_up @ cur))
//! cur      = ffn + ffn_in
//! cur      = RMSNorm(cur, shared_head_norm)    # pre-LM-head norm
//! logits   = lm_head_weights @ cur             # caller supplies trunk's lm_head
//! ```
//!
//! ## Per-call alloc: zero
//!
//! All intermediates live in [`Qwen35MtpHeadScratch`], which is allocated
//! once per slot. The forward writes logits into `scratch.logits` (caller
//! reads via `gpu.download_f32`). KV cache is in [`Qwen35MtpHeadKvCache`]
//! (single-layer F32, separate from the trunk).
//!
//! ## RoPE choice
//!
//! Qwen3.5 spec says M-RoPE multi-section, but the trunk's full-attention
//! layer also uses `rope_partial_interleaved_f32` (qwen35.rs:2431,2611) —
//! the M-RoPE sections + partial-rotary-factor=0.25 reduce to the same
//! single-section partial RoPE for text-only tokens. We mirror trunk
//! behavior so the MTP-head numerics stay in sync with trunk-trained
//! distillation targets.
//!
//! ## What this module does NOT do
//!
//! - tree-decode multi-step recursion: caller composes step N+1 by feeding
//!   `scratch.t_mtp_out` back as `prev_hidden`.
//! - sampling: caller takes argmax / does temperature / top-k.
//! - KV rollback / batched verify: this is a single-token forward, the
//!   verify-loop equivalent for MTP would be Task 11.

use crate::qwen35::{Qwen35Config, Qwen35Weights};
use hip_bridge::{DeviceBuffer, HipResult};
use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};
use hipfire_runtime::llama::{f16_to_f32, weight_gemv, EmbeddingFormat, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;

// ─── Config ──────────────────────────────────────────────────────────────

/// All dimensions and hyperparams the MTP head needs. Loaded from the
/// `.mtp` file's metadata JSON; nothing is hardcoded per model size.
#[derive(Debug, Clone)]
pub struct Qwen35MtpHeadConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    /// Partial-rotary factor; defaults to 0.25 to mirror Qwen3.5 trunk.
    /// Stored as the absolute n_rot (head_dim * factor).
    pub n_rot: usize,
    pub rms_norm_eps: f32,
    /// Maximum positions the head's KV cache can store. Caller picks at
    /// allocation time.
    pub max_seq: usize,
    /// True iff the source model's `tie_word_embeddings` is true. Lets the
    /// caller know whether trunk's embed_tokens + lm_head are aliases.
    pub tie_word_embeddings: bool,
}

impl Qwen35MtpHeadConfig {
    /// Parse from a `.mtp` file's metadata JSON. Defaults match the Task 8
    /// extractor's canonical layout.
    fn from_metadata(meta: &serde_json::Value, max_seq: usize) -> Self {
        let g = |k: &str, default: f64| -> f64 {
            meta.get(k).and_then(|v| v.as_f64()).unwrap_or(default)
        };
        let gu = |k: &str| -> usize {
            meta.get(k)
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| panic!(".mtp metadata missing required key '{k}'"))
                as usize
        };
        let n_embd = gu("n_embd");
        let n_head = gu("n_head");
        let n_head_kv = gu("n_head_kv");
        let head_dim = gu("n_embd_head");
        let n_ff = gu("n_ff");
        let vocab_size = gu("vocab_size");
        // partial_rotary_factor lives nested under config_text_config; fall
        // back to 0.25 (Qwen3.5 default) when absent.
        let prf = meta
            .get("config_text_config")
            .and_then(|c| c.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.25);
        let n_rot = (head_dim as f64 * prf) as usize;
        let rope_theta = g("rope_theta", 10_000_000.0) as f32;
        let rms_norm_eps = g("rms_norm_eps", 1e-6) as f32;
        let tie_word_embeddings = meta
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        Self {
            n_embd, n_head, n_head_kv, head_dim, n_ff,
            vocab_size, rope_theta, n_rot, rms_norm_eps, max_seq,
            tie_word_embeddings,
        }
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────

/// All 15 GPU-resident MTP head tensors. Ownership is tied to the head
/// instance; [`Qwen35MtpHeadWeights::free_gpu`] releases them at unload.
pub struct Qwen35MtpHeadWeights {
    // Norms (F32, 1D)
    pub shared_head_norm: GpuTensor,
    pub enorm: GpuTensor,
    pub hnorm: GpuTensor,
    pub attn_norm: GpuTensor,
    pub attn_post_norm: GpuTensor,
    pub attn_q_norm: GpuTensor,
    pub attn_k_norm: GpuTensor,
    // 2D weights (MQ4 / Q8)
    pub eh_proj: WeightTensor,   // [n_embd, 2 * n_embd]
    pub wq: WeightTensor,        // [2 * head_dim * n_head, n_embd]
    pub wk: WeightTensor,        // [head_dim * n_head_kv, n_embd]
    pub wv: WeightTensor,        // [head_dim * n_head_kv, n_embd]
    pub wo: WeightTensor,        // [n_embd, head_dim * n_head]
    pub ffn_gate: WeightTensor,  // [n_ff, n_embd]
    pub ffn_up: WeightTensor,    // [n_ff, n_embd]
    pub ffn_down: WeightTensor,  // [n_embd, n_ff]
}

impl Qwen35MtpHeadWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.shared_head_norm);
        let _ = gpu.free_tensor(self.enorm);
        let _ = gpu.free_tensor(self.hnorm);
        let _ = gpu.free_tensor(self.attn_norm);
        let _ = gpu.free_tensor(self.attn_post_norm);
        let _ = gpu.free_tensor(self.attn_q_norm);
        let _ = gpu.free_tensor(self.attn_k_norm);
        let _ = gpu.free_tensor(self.eh_proj.buf);
        let _ = gpu.free_tensor(self.wq.buf);
        let _ = gpu.free_tensor(self.wk.buf);
        let _ = gpu.free_tensor(self.wv.buf);
        let _ = gpu.free_tensor(self.wo.buf);
        let _ = gpu.free_tensor(self.ffn_gate.buf);
        let _ = gpu.free_tensor(self.ffn_up.buf);
        let _ = gpu.free_tensor(self.ffn_down.buf);
    }
}

// ─── Scratch ─────────────────────────────────────────────────────────────

/// Per-call GPU scratch for the MTP head forward — allocated once via
/// [`Qwen35MtpHeadScratch::new`], reused on every call. Mirrors
/// `Qwen35Scratch` but sized for the MTP head's single block + LM head.
pub struct Qwen35MtpHeadScratch {
    // Activation stages
    pub tok_embd: GpuTensor,    // [n_embd]
    pub e_norm: GpuTensor,      // [n_embd]
    pub h_norm: GpuTensor,      // [n_embd]
    pub concat: GpuTensor,      // [2 * n_embd]
    pub cur: GpuTensor,         // [n_embd] — primary residual stream
    pub residual: GpuTensor,    // [n_embd] — saved for inpSA
    pub tmp: GpuTensor,         // [n_embd] — RMSNorm output scratch

    // Attention sub-block
    pub q_full: GpuTensor,      // [2 * head_dim * n_head]
    pub q: GpuTensor,           // [head_dim * n_head]
    pub gate: GpuTensor,        // [head_dim * n_head]
    pub k: GpuTensor,           // [head_dim * n_head_kv]
    pub v: GpuTensor,           // [head_dim * n_head_kv]
    pub attn_out: GpuTensor,    // [head_dim * n_head]
    pub o: GpuTensor,           // [n_embd]

    // FFN sub-block
    pub gate_ffn: GpuTensor,    // [n_ff]
    pub up: GpuTensor,          // [n_ff]
    pub ffn_hidden: GpuTensor,  // [n_ff]
    pub ffn_out: GpuTensor,     // [n_embd]

    // Snapshot of the post-FFN, pre-LM-head-norm hidden — caller can
    // capture this and feed back as `prev_hidden` for an n+2 prediction.
    pub t_mtp_out: GpuTensor,   // [n_embd]

    // LM head output
    pub logits: GpuTensor,      // [vocab_size]

    // Position scalar — uploaded each forward into a 4-byte device buffer.
    pub pos_buf: DeviceBuffer,
}

impl Qwen35MtpHeadScratch {
    pub fn new(gpu: &mut Gpu, config: &Qwen35MtpHeadConfig) -> HipResult<Self> {
        let dim = config.n_embd;
        let q_dim = config.head_dim * config.n_head;
        let kv_dim = config.head_dim * config.n_head_kv;
        Ok(Self {
            tok_embd: gpu.alloc_tensor(&[dim], DType::F32)?,
            e_norm: gpu.alloc_tensor(&[dim], DType::F32)?,
            h_norm: gpu.alloc_tensor(&[dim], DType::F32)?,
            concat: gpu.alloc_tensor(&[2 * dim], DType::F32)?,
            cur: gpu.alloc_tensor(&[dim], DType::F32)?,
            residual: gpu.alloc_tensor(&[dim], DType::F32)?,
            tmp: gpu.alloc_tensor(&[dim], DType::F32)?,
            q_full: gpu.alloc_tensor(&[2 * q_dim], DType::F32)?,
            q: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            gate: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            k: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            v: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            attn_out: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            o: gpu.alloc_tensor(&[dim], DType::F32)?,
            gate_ffn: gpu.alloc_tensor(&[config.n_ff], DType::F32)?,
            up: gpu.alloc_tensor(&[config.n_ff], DType::F32)?,
            ffn_hidden: gpu.alloc_tensor(&[config.n_ff], DType::F32)?,
            ffn_out: gpu.alloc_tensor(&[dim], DType::F32)?,
            t_mtp_out: gpu.alloc_tensor(&[dim], DType::F32)?,
            logits: gpu.alloc_tensor(&[config.vocab_size], DType::F32)?,
            pos_buf: gpu.hip.malloc(4)?,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.tok_embd);
        let _ = gpu.free_tensor(self.e_norm);
        let _ = gpu.free_tensor(self.h_norm);
        let _ = gpu.free_tensor(self.concat);
        let _ = gpu.free_tensor(self.cur);
        let _ = gpu.free_tensor(self.residual);
        let _ = gpu.free_tensor(self.tmp);
        let _ = gpu.free_tensor(self.q_full);
        let _ = gpu.free_tensor(self.q);
        let _ = gpu.free_tensor(self.gate);
        let _ = gpu.free_tensor(self.k);
        let _ = gpu.free_tensor(self.v);
        let _ = gpu.free_tensor(self.attn_out);
        let _ = gpu.free_tensor(self.o);
        let _ = gpu.free_tensor(self.gate_ffn);
        let _ = gpu.free_tensor(self.up);
        let _ = gpu.free_tensor(self.ffn_hidden);
        let _ = gpu.free_tensor(self.ffn_out);
        let _ = gpu.free_tensor(self.t_mtp_out);
        let _ = gpu.free_tensor(self.logits);
        let _ = gpu.hip.free(self.pos_buf);
    }
}

// ─── KV cache (single-layer, MTP-private) ────────────────────────────────

/// The MTP head has a single attention block, so its KV cache is one
/// per-layer F32 K + V buffer. Separate from the trunk's `KvCache` since
/// the MTP head writes the SAME absolute position the trunk just emitted —
/// reusing the trunk's cache would mean either double-write or
/// snapshot/restore on every cycle.
pub struct Qwen35MtpHeadKvCache {
    pub k_gpu: GpuTensor,    // [max_seq * n_head_kv * head_dim] F32
    pub v_gpu: GpuTensor,    // [max_seq * n_head_kv * head_dim] F32
    pub max_seq: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
}

impl Qwen35MtpHeadKvCache {
    pub fn new(gpu: &mut Gpu, config: &Qwen35MtpHeadConfig) -> HipResult<Self> {
        let kv_dim = config.n_head_kv * config.head_dim;
        let total = config.max_seq * kv_dim;
        Ok(Self {
            k_gpu: gpu.zeros(&[total], DType::F32)?,
            v_gpu: gpu.zeros(&[total], DType::F32)?,
            max_seq: config.max_seq,
            n_head_kv: config.n_head_kv,
            head_dim: config.head_dim,
        })
    }

    /// Reset positions 0..=highest_written. Cheap since we re-zero the
    /// whole buffer; MTP cache is tiny (single layer × max_seq positions).
    pub fn reset(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        let kv_dim = self.n_head_kv * self.head_dim;
        let zeros = vec![0.0f32; self.max_seq * kv_dim];
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(zeros.as_ptr() as *const u8, zeros.len() * 4)
        };
        gpu.hip.memcpy_htod(&self.k_gpu.buf, bytes)?;
        gpu.hip.memcpy_htod(&self.v_gpu.buf, bytes)?;
        Ok(())
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.k_gpu);
        let _ = gpu.free_tensor(self.v_gpu);
    }
}

// ─── Top-level handle ────────────────────────────────────────────────────

/// Loaded MTP head: config + weights, ready for `mtp_head_forward`.
/// Caller separately allocates [`Qwen35MtpHeadScratch`] (per inference slot)
/// and [`Qwen35MtpHeadKvCache`] (per generation).
pub struct Qwen35MtpHead {
    pub config: Qwen35MtpHeadConfig,
    pub weights: Qwen35MtpHeadWeights,
}

impl Qwen35MtpHead {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.weights.free_gpu(gpu);
    }
}

// ─── Loader ──────────────────────────────────────────────────────────────

/// Load a `.mtp` file (arch_id = 21) created by `mtp_extract` (Task 8).
/// Returns the head ready for `mtp_head_forward`.
///
/// `max_seq` bounds the per-position KV cache later allocated by
/// [`Qwen35MtpHeadKvCache::new`]; pick to match your decode budget.
pub fn load_mtp_head(
    path: &Path,
    gpu: &mut Gpu,
    max_seq: usize,
) -> HipResult<Qwen35MtpHead> {
    let hfq = HfqFile::open(path)
        .unwrap_or_else(|e| panic!("open .mtp file {}: {e}", path.display()));
    assert_eq!(
        hfq.arch_id, 21,
        ".mtp file at {} has arch_id={} (expected 21 = QWEN35_MTP_HEAD); \
         is this actually an MTP head extracted by mtp_extract?",
        path.display(), hfq.arch_id
    );
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
        .expect(".mtp metadata JSON parse failed");
    let config = Qwen35MtpHeadConfig::from_metadata(&meta, max_seq);

    // ── Norms (F32, 1D) ─────────────────────────────────────────────────
    //
    // The .mtp file uses bare tensor names ("enorm", "wq", ...) — no
    // "model.language_model." prefix. We read raw bytes and upload
    // directly; `load_weight_tensor` from qwen35.rs unconditionally
    // prepends that prefix and so cannot be reused here.
    let n_embd = config.n_embd;
    let head_dim = config.head_dim;

    let shared_head_norm = load_norm_raw(&hfq, gpu, "shared_head_norm", n_embd)?;
    let enorm           = load_norm_raw(&hfq, gpu, "enorm",            n_embd)?;
    let hnorm           = load_norm_raw(&hfq, gpu, "hnorm",            n_embd)?;
    let attn_norm       = load_norm_raw(&hfq, gpu, "attn_norm",        n_embd)?;
    let attn_post_norm  = load_norm_raw(&hfq, gpu, "attn_post_norm",   n_embd)?;
    let attn_q_norm     = load_norm_raw(&hfq, gpu, "attn_q_norm",      head_dim)?;
    let attn_k_norm     = load_norm_raw(&hfq, gpu, "attn_k_norm",      head_dim)?;

    // ── 2D weights ──────────────────────────────────────────────────────
    let q_full_dim = 2 * head_dim * config.n_head;
    let kv_dim     = head_dim * config.n_head_kv;
    let q_dim      = head_dim * config.n_head;

    let eh_proj  = load_weight_raw(&hfq, gpu, "eh_proj",  n_embd,    2 * n_embd)?;
    let wq       = load_weight_raw(&hfq, gpu, "wq",       q_full_dim, n_embd)?;
    let wk       = load_weight_raw(&hfq, gpu, "wk",       kv_dim,    n_embd)?;
    let wv       = load_weight_raw(&hfq, gpu, "wv",       kv_dim,    n_embd)?;
    let wo       = load_weight_raw(&hfq, gpu, "wo",       n_embd,    q_dim)?;
    let ffn_gate = load_weight_raw(&hfq, gpu, "ffn_gate", config.n_ff, n_embd)?;
    let ffn_up   = load_weight_raw(&hfq, gpu, "ffn_up",   config.n_ff, n_embd)?;
    let ffn_down = load_weight_raw(&hfq, gpu, "ffn_down", n_embd,      config.n_ff)?;

    let weights = Qwen35MtpHeadWeights {
        shared_head_norm, enorm, hnorm, attn_norm, attn_post_norm,
        attn_q_norm, attn_k_norm,
        eh_proj, wq, wk, wv, wo, ffn_gate, ffn_up, ffn_down,
    };

    Ok(Qwen35MtpHead { config, weights })
}

/// Load a 1D F32 norm tensor from a .mtp file. Mirrors the +1.0 offset
/// convention used by the trunk (Qwen3.5 RMSNorm: `out = x · rsqrt(var+eps)
/// · (1 + weight)`). All `.mtp` norms ship as quant_type=2 (F32) per the
/// `mtp_extract` packing rule.
fn load_norm_raw(
    hfq: &HfqFile, gpu: &mut Gpu, name: &str, expected_n: usize,
) -> HipResult<GpuTensor> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .unwrap_or_else(|| panic!(".mtp tensor '{name}' missing"));
    let mut f32_data: Vec<f32> = match info.quant_type {
        1 => data.chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => panic!("norm '{name}' has unexpected qt={other} (expected F16 or F32)"),
    };
    assert_eq!(
        f32_data.len(), expected_n,
        "norm '{name}': loaded {} elems but expected {expected_n}",
        f32_data.len(),
    );
    // Qwen3.5 / 3.6 RMSNorm convention: weight is stored as the offset from
    // 1.0. Trunk's `load_norm_weight` does the same `+= 1.0` pre-upload
    // step for every per-layer norm. The MTP head's `mtp.norm.weight`
    // (mapped to `shared_head_norm`) is the equivalent of the trunk's
    // `model.norm.weight` — but the trunk treats the FINAL norm as raw
    // (no +1 offset, see `load_norm_weight_raw`). We follow the trunk's
    // per-layer convention here for ALL norms because the safetensors
    // shipped values for `mtp.norm.weight` ≈ 0 (consistent with offset
    // representation). Off-by-one risk is small: the trunk does the same
    // +1 for `shared_expert_intermediate.norm` etc.
    for v in &mut f32_data { *v += 1.0; }
    gpu.upload_f32(&f32_data, &[expected_n])
}

/// Load a 2D quantized weight tensor from a .mtp file. Resolves any of
/// the supported quant types into a [`WeightTensor`]; m and k are passed
/// in (the mtp container stores shape but we trust caller-supplied dims).
fn load_weight_raw(
    hfq: &HfqFile, gpu: &mut Gpu, name: &str, m: usize, k: usize,
) -> HipResult<WeightTensor> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .unwrap_or_else(|| panic!(".mtp tensor '{name}' missing"));
    sanity_check_2d_shape(name, info, m, k);
    weight_tensor_from_raw(gpu, info.quant_type, &data, m, k, name)
}

/// Cross-check the on-disk shape against the caller's expected (m, k).
/// Catches silent dim mismatches (e.g. tied vocab, head split done wrong).
fn sanity_check_2d_shape(name: &str, info: &HfqTensorInfo, m: usize, k: usize) {
    if info.shape.len() != 2 {
        panic!(
            ".mtp tensor '{name}': expected 2D shape, got {}D = {:?}",
            info.shape.len(), info.shape,
        );
    }
    let on_disk_m = info.shape[0] as usize;
    let on_disk_k = info.shape[1] as usize;
    assert_eq!(
        on_disk_m, m,
        ".mtp tensor '{name}': shape[0]={on_disk_m} but expected m={m}"
    );
    assert_eq!(
        on_disk_k, k,
        ".mtp tensor '{name}': shape[1]={on_disk_k} but expected k={k}"
    );
}

/// Wrap raw quantized bytes into a [`WeightTensor`]. Local copy of the
/// dispatch table from `qwen35::load_weight_tensor_raw`, restricted to the
/// quant types `mtp_extract` actually emits (MQ4, Q8_F16=Q8_0, F16, F32).
fn weight_tensor_from_raw(
    gpu: &Gpu, quant_type: u8, data: &[u8], m: usize, k: usize, name: &str,
) -> HipResult<WeightTensor> {
    match quant_type {
        13 => {
            // MQ4G256 — must be K%256-aligned (kernel requirement).
            assert!(
                k % 256 == 0,
                ".mtp tensor '{name}' is MQ4G256 with K={k} not divisible by 256"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::MQ4G256, m, k, row_stride: 0 })
        }
        3 => {
            // Q8_F16 (group_size=32, 34 bytes/group) — same byte layout as
            // GGML Q8_0; existing gemv_q8_0 dispatch works directly.
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::Q8_0, m, k, row_stride: 0 })
        }
        1 => {
            // F16 → dequantize on host, upload as F32 (no GPU-native F16
            // GEMV; the trunk does the same conversion in load_weight_tensor_raw).
            let f32_data: Vec<f32> = data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    f32_data.as_ptr() as *const u8,
                    f32_data.len() * 4,
                )
            };
            let buf = gpu.upload_raw(bytes, &[m, k])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::F32, m, k, row_stride: 0 })
        }
        2 => {
            // F32 raw.
            let buf = gpu.upload_raw(data, &[m, k])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::F32, m, k, row_stride: 0 })
        }
        other => panic!(
            ".mtp tensor '{name}': unsupported quant_type={other} \
             (mtp_extract emits MQ4G256=13, Q8_F16=3, F16=1, F32=2)"
        ),
    }
}

// ─── Forward pass ────────────────────────────────────────────────────────

/// Single MTP-head forward. Produces logits over the full vocab in
/// `scratch.logits` for position `pos + 1`. Caller does sampling.
///
/// - `next_token`: the most-recently committed token id at position `pos`.
///   Embedded via the trunk's `weights.token_embd` and used as the
///   prediction-target signal.
/// - `prev_hidden`: the trunk's post-final-norm hidden state at position
///   `pos` (or any other contextually-equivalent activation that's
///   distillation-aligned with the MTP block's training input).
/// - `pos`: current position (the slot the MTP block writes its K/V into,
///   then attends to all positions 0..=pos).
/// - `lm_head_weights`: the trunk's `weights.output`. The MTP file
///   intentionally does NOT pack a separate LM head — Qwen3.5/3.6 share
///   the trunk's lm_head with the MTP head ("shared_lm_head_with_trunk":
///   true in the mtp metadata).
///
/// Side effects:
/// - Writes K/V at slot `pos` into `kv` (overwriting any prior data there).
/// - Writes logits into `scratch.logits` (caller `download_f32` to read).
/// - Writes the post-FFN, pre-LM-head-norm hidden into `scratch.t_mtp_out`
///   so callers wanting an n+2 prediction can feed it back as `prev_hidden`.
pub fn mtp_head_forward(
    gpu: &mut Gpu,
    head: &Qwen35MtpHead,
    scratch: &Qwen35MtpHeadScratch,
    kv: &mut Qwen35MtpHeadKvCache,
    next_token: u32,
    prev_hidden: &GpuTensor,
    pos: usize,
    trunk_weights: &Qwen35Weights,
    lm_head_weights: &WeightTensor,
) -> HipResult<()> {
    let cfg = &head.config;
    let n_embd = cfg.n_embd;

    assert_eq!(
        lm_head_weights.k, n_embd,
        "mtp_head_forward: lm_head_weights.k={} but n_embd={n_embd}; \
         lm_head must accept the MTP head's hidden",
        lm_head_weights.k,
    );

    // Run the block (NextN concat + eh_proj + attn + FFN). Writes
    // `scratch.t_mtp_out` and leaves `scratch.ffn_out` holding the same
    // hidden (alias for the LM-head path).
    mtp_head_forward_block_only(
        gpu, head, scratch, kv, next_token, prev_hidden, None, pos,
        trunk_weights,
    )?;

    // Standard single-step lm_head: shared_head_norm + GEMV over t_mtp_out.
    let w = &head.weights;
    gpu.rmsnorm_f32(&scratch.t_mtp_out, &w.shared_head_norm, &scratch.tmp, cfg.rms_norm_eps)?;
    weight_gemv(gpu, lm_head_weights, &scratch.tmp, &scratch.logits)?;

    Ok(())
}

/// Block-only variant of [`mtp_head_forward`]: runs the NextN concat +
/// eh_proj + attention + FFN, but **stops before** `shared_head_norm` and
/// `lm_head`. Writes the post-FFN, pre-shared-head-norm hidden into
/// `scratch.t_mtp_out`. Caller is responsible for running
/// [`mtp_head_apply_lm_head_batched`] on a stack of N `t_mtp_out`s to
/// recover the predicted-token logits in one batched GEMM.
///
/// ## Optional embedding override (lossy K-step chaining)
///
/// In the standard path (`next_token_embed = None`), `next_token` is
/// embedded via the trunk's `token_embd` table — same as
/// `mtp_head_forward`. The K-step batched-lm_head optimization in
/// `spec_step_mtp` chains forwards WITHOUT yet knowing each step's
/// predicted token (we postpone all K argmaxes to a single end-of-chain
/// batched lm_head). To allow step k+1 to proceed before step k's
/// `lm_head` runs, the caller passes `next_token_embed = Some(prev_step_t_mtp_out)`,
/// which BYPASSES the embedding lookup and feeds the previous step's
/// `t_mtp_out` directly as the "embedding of the predicted token."
///
/// This is **architecturally lossy** — the MTP head was trained with
/// discrete-token round-trips through `token_embd`, so feeding the
/// continuous post-FFN hidden as a substitute for `embed[token]` is OOD
/// for the head. Acceptance rate (τ) may degrade. Lossless guarantee is
/// preserved at the trunk-verify level: any incorrect MTP candidate is
/// rejected by the trunk's argmax check and the cycle just re-AR-decodes
/// from the bonus token.
///
/// `next_token` is IGNORED when `next_token_embed` is `Some(_)`. The
/// caller may pass any sentinel (e.g. 0).
#[allow(clippy::too_many_arguments)]
pub fn mtp_head_forward_block_only(
    gpu: &mut Gpu,
    head: &Qwen35MtpHead,
    scratch: &Qwen35MtpHeadScratch,
    kv: &mut Qwen35MtpHeadKvCache,
    next_token: u32,
    prev_hidden: &GpuTensor,
    next_token_embed: Option<&GpuTensor>,
    pos: usize,
    trunk_weights: &Qwen35Weights,
) -> HipResult<()> {
    let cfg = &head.config;
    let w = &head.weights;
    let n_embd = cfg.n_embd;
    let kv_dim = cfg.head_dim * cfg.n_head_kv;

    assert_eq!(
        prev_hidden.numel(), n_embd,
        "mtp_head_forward_block_only: prev_hidden has {} elems but expected n_embd={n_embd}",
        prev_hidden.numel(),
    );
    assert!(
        pos < kv.max_seq,
        "mtp_head_forward_block_only: pos={pos} >= kv.max_seq={}",
        kv.max_seq,
    );

    // Upload position scalar for the RoPE / attention kernels.
    let pos_i32 = pos as i32;
    gpu.hip.memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;

    // ── 1. Token embedding (table lookup OR caller-supplied override) ────
    let dim_bytes = n_embd * 4;
    if let Some(embed) = next_token_embed {
        assert_eq!(
            embed.numel(), n_embd,
            "mtp_head_forward_block_only: next_token_embed has {} elems but expected n_embd={n_embd}",
            embed.numel(),
        );
        // Lossy-recursion path: feed the caller's pre-computed activation
        // directly into the e_norm input slot. Aliasing-safe (separate
        // backing buffers — caller passes the previous step's t_mtp_out).
        gpu.hip.memcpy_dtod_at(&scratch.tok_embd.buf, 0, &embed.buf, 0, dim_bytes)?;
    } else {
        embed_lookup_into(gpu, trunk_weights, &scratch.tok_embd, next_token, n_embd)?;
    }

    // ── 2. RMSNorm both inputs to the NextN projection ───────────────────
    gpu.rmsnorm_f32(&scratch.tok_embd, &w.enorm, &scratch.e_norm, cfg.rms_norm_eps)?;
    gpu.rmsnorm_f32(prev_hidden, &w.hnorm, &scratch.h_norm, cfg.rms_norm_eps)?;

    // ── 3. concat = [e_norm | h_norm], then cur = eh_proj @ concat ───────
    gpu.hip.memcpy_dtod_at(&scratch.concat.buf, 0, &scratch.e_norm.buf, 0, dim_bytes)?;
    gpu.hip.memcpy_dtod_at(&scratch.concat.buf, dim_bytes, &scratch.h_norm.buf, 0, dim_bytes)?;
    weight_gemv(gpu, &w.eh_proj, &scratch.concat, &scratch.cur)?;

    // Save inpSA for the attention residual (cur is about to be norm'd
    // out-of-place into scratch.tmp).
    gpu.hip.memcpy_dtod_at(&scratch.residual.buf, 0, &scratch.cur.buf, 0, dim_bytes)?;

    // ── 4. Pre-attn norm + Q/K/V projections ─────────────────────────────
    gpu.rmsnorm_f32(&scratch.cur, &w.attn_norm, &scratch.tmp, cfg.rms_norm_eps)?;

    // Qwen3.5 gated-Q: wq emits 2 * head_dim * n_head, deinterleaved into
    // Q (head-major first half) and gate (second half) per-head. Mirror
    // qwen35.rs:2402-2414.
    weight_gemv(gpu, &w.wq, &scratch.tmp, &scratch.q_full)?;
    gpu.deinterleave_f32(&scratch.q_full, &scratch.q, &scratch.gate, cfg.n_head, cfg.head_dim)?;
    gpu.rmsnorm_batched(
        &scratch.q, &w.attn_q_norm, &scratch.q,
        cfg.n_head, cfg.head_dim, cfg.rms_norm_eps,
    )?;

    weight_gemv(gpu, &w.wk, &scratch.tmp, &scratch.k)?;
    weight_gemv(gpu, &w.wv, &scratch.tmp, &scratch.v)?;
    gpu.rmsnorm_batched(
        &scratch.k, &w.attn_k_norm, &scratch.k,
        cfg.n_head_kv, cfg.head_dim, cfg.rms_norm_eps,
    )?;

    // ── 5. RoPE (partial-interleaved, mirrors trunk's full-attn layer) ───
    gpu.rope_partial_interleaved_f32(
        &scratch.q, &scratch.k, &scratch.pos_buf,
        cfg.n_head, cfg.n_head_kv, cfg.head_dim, cfg.n_rot, cfg.rope_theta,
    )?;

    // ── 6. KV cache write at slot `pos` ──────────────────────────────────
    //
    // F32 cache: kv_cache_write copies kv_dim floats into slot pos.
    gpu.kv_cache_write(&kv.k_gpu, &scratch.k, &scratch.pos_buf, kv_dim)?;
    gpu.kv_cache_write(&kv.v_gpu, &scratch.v, &scratch.pos_buf, kv_dim)?;

    // ── 7. Attention ────────────────────────────────────────────────────
    //
    // attention_f32 computes attn_out = softmax(Q · Kᵀ / sqrt(hd)) · V over
    // positions [0, seq_len_hint). seq_len_hint = pos + 1 — every position
    // up to and including the slot we just wrote.
    gpu.attention_f32(
        &scratch.q, &kv.k_gpu, &kv.v_gpu, &scratch.attn_out, &scratch.pos_buf,
        pos + 1, cfg.n_head, cfg.n_head_kv, cfg.head_dim, kv.max_seq,
    )?;

    // ── 8. Apply gate (sigmoid(gate) * attn_out, in-place on attn_out) ───
    gpu.sigmoid_mul_f32(&scratch.attn_out, &scratch.gate)?;

    // ── 9. Output projection + residual ──────────────────────────────────
    weight_gemv(gpu, &w.wo, &scratch.attn_out, &scratch.o)?;
    gpu.add_inplace_f32(&scratch.o, &scratch.residual)?;
    // scratch.o now holds (attn_out @ wo + inpSA); this is the FFN residual base.

    // ── 10. POST-attn norm + SwiGLU FFN + residual ───────────────────────
    //
    // Note attn_post_norm runs BEFORE the FFN and the residual is taken
    // from the pre-norm activation, mirroring the standard Qwen3.5 layer
    // (post-attention-layernorm in HF lingo = pre-FFN norm here, with the
    // "attn_post_norm" name reflecting its source position in the .mtp
    // metadata file).
    gpu.rmsnorm_f32(&scratch.o, &w.attn_post_norm, &scratch.tmp, cfg.rms_norm_eps)?;
    weight_gemv(gpu, &w.ffn_gate, &scratch.tmp, &scratch.gate_ffn)?;
    weight_gemv(gpu, &w.ffn_up,   &scratch.tmp, &scratch.up)?;
    gpu.silu_mul_f32(&scratch.gate_ffn, &scratch.up, &scratch.ffn_hidden)?;
    weight_gemv(gpu, &w.ffn_down, &scratch.ffn_hidden, &scratch.ffn_out)?;
    gpu.add_inplace_f32(&scratch.ffn_out, &scratch.o)?;
    // scratch.ffn_out now holds the post-FFN, pre-LM-head-norm hidden.

    // Snapshot for callers that want to chain into n+2 prediction OR feed
    // into the batched `mtp_head_apply_lm_head_batched` end-of-chain reduce.
    gpu.hip.memcpy_dtod_at(&scratch.t_mtp_out.buf, 0, &scratch.ffn_out.buf, 0, dim_bytes)?;

    Ok(())
}

/// Batched end-of-chain LM head: applies `shared_head_norm` to `n` stacked
/// `t_mtp_out` rows and runs the trunk's lm_head as a single batched GEMM.
///
/// `t_mtp_outs_stacked` has shape `[n, n_embd]` (row-major, contiguous).
/// `logits_batched` is the caller-allocated output of shape `[n, vocab]`.
/// `tmp_batched` is `[n, n_embd]` scratch for the rmsnorm output (caller
/// owns; reused across cycles).
/// `rot_batched` is `[n, n_embd]` scratch used for FWHT-rotated x for
/// MagnumQuant lm_heads (MQ4/MQ3/MQ6); ignored for non-MQ dtypes.
///
/// Mirrors the per-dtype dispatch in `mtp_probe::probe_one_step` and
/// `speculative::verify_dflash_block_inner`.
#[allow(clippy::too_many_arguments)]
pub fn mtp_head_apply_lm_head_batched(
    gpu: &mut Gpu,
    head: &Qwen35MtpHead,
    lm_head_weights: &WeightTensor,
    t_mtp_outs_stacked: &GpuTensor,
    tmp_batched: &GpuTensor,
    rot_batched: &GpuTensor,
    logits_batched: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    let cfg = &head.config;
    let w = &head.weights;
    let n_embd = cfg.n_embd;
    let vocab = cfg.vocab_size;
    assert_eq!(
        lm_head_weights.k, n_embd,
        "mtp_head_apply_lm_head_batched: lm_head_weights.k={} but n_embd={n_embd}",
        lm_head_weights.k,
    );
    assert!(
        t_mtp_outs_stacked.numel() >= n * n_embd,
        "t_mtp_outs_stacked too small: {} < n*n_embd ({})",
        t_mtp_outs_stacked.numel(), n * n_embd,
    );
    assert!(
        tmp_batched.numel() >= n * n_embd,
        "tmp_batched too small: {} < n*n_embd ({})",
        tmp_batched.numel(), n * n_embd,
    );
    assert!(
        logits_batched.numel() >= n * vocab,
        "logits_batched too small: {} < n*vocab ({})",
        logits_batched.numel(), n * vocab,
    );

    // Per-row shared_head_norm.
    gpu.rmsnorm_batched(t_mtp_outs_stacked, &w.shared_head_norm, tmp_batched,
                        n, n_embd, cfg.rms_norm_eps)?;

    // Per-dtype batched LM head dispatch (mirrors mtp_probe.rs:278+).
    let logits_view = logits_batched.sub_offset(0, n * vocab);
    match lm_head_weights.gpu_dtype {
        DType::Q8_0 => {
            gpu.gemm_q8_0_batched(
                &lm_head_weights.buf, tmp_batched, &logits_view,
                lm_head_weights.m, lm_head_weights.k, n,
            )?;
        }
        DType::HFQ4G256 => {
            gpu.gemm_hfq4g256_batched_lmhead(
                &lm_head_weights.buf, tmp_batched, &logits_view,
                lm_head_weights.m, lm_head_weights.k, n,
            )?;
        }
        DType::MQ4G256 => {
            let rot_view = rot_batched.sub_offset(0, n * lm_head_weights.k);
            gpu.rotate_x_mq_batched(tmp_batched, &rot_view, lm_head_weights.k, n)?;
            gpu.gemm_hfq4g256_batched_lmhead(
                &lm_head_weights.buf, &rot_view, &logits_view,
                lm_head_weights.m, lm_head_weights.k, n,
            )?;
        }
        DType::MQ3G256 => {
            let rot_view = rot_batched.sub_offset(0, n * lm_head_weights.k);
            gpu.rotate_x_mq_batched(tmp_batched, &rot_view, lm_head_weights.k, n)?;
            gpu.gemm_hfq3g256_batched_lmhead(
                &lm_head_weights.buf, &rot_view, &logits_view,
                lm_head_weights.m, lm_head_weights.k, n,
            )?;
        }
        DType::HFQ6G256 => {
            gpu.gemm_hfq6g256_batched_lmhead(
                &lm_head_weights.buf, tmp_batched, &logits_view,
                lm_head_weights.m, lm_head_weights.k, n,
            )?;
        }
        DType::MQ6G256 => {
            let rot_view = rot_batched.sub_offset(0, n * lm_head_weights.k);
            gpu.rotate_x_mq_batched(tmp_batched, &rot_view, lm_head_weights.k, n)?;
            gpu.gemm_hfq6g256_batched_lmhead(
                &lm_head_weights.buf, &rot_view, &logits_view,
                lm_head_weights.m, lm_head_weights.k, n,
            )?;
        }
        _ => {
            // Fallback: per-row weight_gemv. Same path mtp_probe uses for
            // unrecognized dtypes. Defeats the K-amortization but keeps
            // correctness for less-common lm_head formats.
            for i in 0..n {
                let row = tmp_batched.sub_offset(i * n_embd, n_embd);
                let logits_row = logits_view.sub_offset(i * vocab, vocab);
                weight_gemv(gpu, lm_head_weights, &row, &logits_row)?;
            }
        }
    }
    Ok(())
}

/// Per-format embedding-lookup dispatch. Mirrors `mtp_probe::embed_lookup_to_scratch`.
fn embed_lookup_into(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    out: &GpuTensor,
    token: u32,
    dim: usize,
) -> HipResult<()> {
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.token_embd, out, token, dim),
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.token_embd, out, token, dim),
        EmbeddingFormat::Q8_0     => gpu.embedding_lookup_q8(&weights.token_embd, out, token, dim),
        EmbeddingFormat::Q4K      => gpu.embedding_lookup_q4k(&weights.token_embd, out, token, dim),
        EmbeddingFormat::F32      => gpu.embedding_lookup(&weights.token_embd, out, token, dim),
    }
}
