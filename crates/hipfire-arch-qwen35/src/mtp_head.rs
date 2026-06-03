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

use crate::qwen35::Qwen35Weights;
use hip_bridge::{DeviceBuffer, HipResult};
use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};
use hipfire_runtime::llama::{
    self, f16_to_f32, fused_silu_mul_rotate_mq_batched_for, fused_silu_mul_rotate_mq_for,
    rotate_x_mq_for, weight_gemv, EmbeddingFormat, WeightTensor,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;

// ─── Config ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35MtpFfnKind {
    Dense,
    Moe,
}

/// All dimensions and hyperparams the MTP head needs. Loaded from the
/// `.mtp` file's metadata JSON; nothing is hardcoded per model size.
#[derive(Debug, Clone)]
pub struct Qwen35MtpHeadConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub ffn_kind: Qwen35MtpFfnKind,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,
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
        let ffn_kind = match meta
            .get("ffn_kind")
            .and_then(|v| v.as_str())
            .unwrap_or("dense")
        {
            "dense" => Qwen35MtpFfnKind::Dense,
            "moe" => Qwen35MtpFfnKind::Moe,
            other => panic!(".mtp metadata has unsupported ffn_kind='{other}'"),
        };
        let num_experts = meta
            .get("num_experts")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let num_experts_per_tok = meta
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let moe_intermediate_size = meta
            .get("moe_intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let shared_expert_intermediate_size = meta
            .get("shared_expert_intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let norm_topk_prob = meta
            .get("norm_topk_prob")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
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
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            n_ff,
            ffn_kind,
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            norm_topk_prob,
            vocab_size,
            rope_theta,
            n_rot,
            rms_norm_eps,
            max_seq,
            tie_word_embeddings,
        }
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────

pub struct Qwen35MtpDenseFfnWeights {
    pub gate: WeightTensor, // [n_ff, n_embd]
    pub up: WeightTensor,   // [n_ff, n_embd]
    pub down: WeightTensor, // [n_embd, n_ff]
}

pub struct Qwen35MtpMoeExpertWeights {
    pub gate_up: WeightTensor, // [2 * moe_intermediate, n_embd]
    pub down: WeightTensor,    // [n_embd, moe_intermediate]
}

pub struct Qwen35MtpMoeSharedExpertWeights {
    pub gate: WeightTensor, // [shared_expert_intermediate, n_embd]
    pub up: WeightTensor,   // [shared_expert_intermediate, n_embd]
    pub down: WeightTensor, // [n_embd, shared_expert_intermediate]
}

pub struct Qwen35MtpMoeFfnWeights {
    pub router: WeightTensor, // [num_experts, n_embd]
    pub shared_expert: Qwen35MtpMoeSharedExpertWeights,
    pub shared_expert_gate: WeightTensor, // [1, n_embd]
    pub experts: Vec<Qwen35MtpMoeExpertWeights>,
    pub expert_gate_up_ptrs: GpuTensor,
    pub expert_down_ptrs: GpuTensor,
}

pub enum Qwen35MtpFfnWeights {
    Dense(Qwen35MtpDenseFfnWeights),
    Moe(Qwen35MtpMoeFfnWeights),
}

/// All 15 GPU-resident MTP head tensors (+2 optional FastMTP-style
/// vocab-compression sidecar tensors). Ownership is tied to the head
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
    pub eh_proj: WeightTensor, // [n_embd, 2 * n_embd]
    pub wq: WeightTensor,      // [2 * head_dim * n_head, n_embd]
    pub wk: WeightTensor,      // [head_dim * n_head_kv, n_embd]
    pub wv: WeightTensor,      // [head_dim * n_head_kv, n_embd]
    pub wo: WeightTensor,      // [n_embd, head_dim * n_head]
    pub ffn: Qwen35MtpFfnWeights,

    // Optional FastMTP-style vocab-compression sidecar (present iff the
    // .mtp metadata says `has_compressed_lm_head_draft: true`).
    //
    // - `lm_head_draft`: top-K rows of trunk lm_head (MQ4 / Q8), used by
    //   `mtp_head_forward_compressed` in place of the trunk's full lm_head.
    //   Shape [compressed_vocab_size, n_embd]. ~7.7x BW reduction at K=32K
    //   on a 248K-vocab Qwen3.5/3.6.
    // - `lm_head_draft_vocab_map`: host-side u32 array mapping draft idx
    //   -> full vocab idx. Kept for host fallback paths and diagnostics.
    // - `lm_head_draft_vocab_map_gpu`: same map on-device for the greedy
    //   MTP token-chain path, where draft idx -> full token must happen
    //   without a per-step D2H sync.
    // - `compressed_vocab_size`: K. Cached for fast access in forward.
    pub lm_head_draft: Option<WeightTensor>,
    pub lm_head_draft_vocab_map: Option<Vec<u32>>,
    pub lm_head_draft_vocab_map_gpu: Option<GpuTensor>,
    pub compressed_vocab_size: Option<usize>,
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
        match self.ffn {
            Qwen35MtpFfnWeights::Dense(ffn) => {
                let _ = gpu.free_tensor(ffn.gate.buf);
                let _ = gpu.free_tensor(ffn.up.buf);
                let _ = gpu.free_tensor(ffn.down.buf);
            }
            Qwen35MtpFfnWeights::Moe(ffn) => {
                let _ = gpu.free_tensor(ffn.router.buf);
                let _ = gpu.free_tensor(ffn.shared_expert_gate.buf);
                let _ = gpu.free_tensor(ffn.shared_expert.gate.buf);
                let _ = gpu.free_tensor(ffn.shared_expert.up.buf);
                let _ = gpu.free_tensor(ffn.shared_expert.down.buf);
                let _ = gpu.free_tensor(ffn.expert_gate_up_ptrs);
                let _ = gpu.free_tensor(ffn.expert_down_ptrs);
                for expert in ffn.experts {
                    let _ = gpu.free_tensor(expert.gate_up.buf);
                    let _ = gpu.free_tensor(expert.down.buf);
                }
            }
        }
        if let Some(lm_d) = self.lm_head_draft {
            let _ = gpu.free_tensor(lm_d.buf);
        }
        if let Some(vmap) = self.lm_head_draft_vocab_map_gpu {
            let _ = gpu.free_tensor(vmap);
        }
    }
}

// ─── Scratch ─────────────────────────────────────────────────────────────

/// Per-call GPU scratch for the MTP head forward — allocated once via
/// [`Qwen35MtpHeadScratch::new`], reused on every call. Mirrors
/// `Qwen35Scratch` but sized for the MTP head's single block + LM head.
pub struct Qwen35MtpHeadScratch {
    // Activation stages
    pub tok_embd: GpuTensor, // [n_embd]
    pub e_norm: GpuTensor,   // [n_embd]
    pub h_norm: GpuTensor,   // [n_embd]
    pub concat: GpuTensor,   // [2 * n_embd]
    pub cur: GpuTensor,      // [n_embd] — primary residual stream
    pub residual: GpuTensor, // [n_embd] — saved for inpSA
    pub tmp: GpuTensor,      // [n_embd] — RMSNorm output scratch

    // Attention sub-block
    pub q_full: GpuTensor,   // [2 * head_dim * n_head]
    pub q: GpuTensor,        // [head_dim * n_head]
    pub gate: GpuTensor,     // [head_dim * n_head]
    pub k: GpuTensor,        // [head_dim * n_head_kv]
    pub v: GpuTensor,        // [head_dim * n_head_kv]
    pub attn_out: GpuTensor, // [head_dim * n_head]
    pub o: GpuTensor,        // [n_embd]

    // FFN sub-block
    pub gate_ffn: GpuTensor,   // [n_ff]
    pub up: GpuTensor,         // [n_ff]
    pub ffn_hidden: GpuTensor, // [n_ff]
    pub ffn_out: GpuTensor,    // [n_embd]

    // MoE FFN scratch, allocated only for ffn_kind=moe.
    pub moe_router_logits: Option<GpuTensor>, // [num_experts]
    pub moe_scalar_buf: Option<GpuTensor>,    // [1]
    pub moe_x_rot: Option<GpuTensor>,         // [n_embd]
    pub moe_gate_up_buf: Option<GpuTensor>,   // [2 * max(moe_intermediate, shared_intermediate)]
    pub moe_gate_buf: Option<GpuTensor>,      // [max(moe_intermediate, shared_intermediate)]
    pub moe_up_buf: Option<GpuTensor>,        // [max(moe_intermediate, shared_intermediate)]
    pub moe_ffn_hidden: Option<GpuTensor>,    // [max(moe_intermediate, shared_intermediate)]
    pub moe_ffn_out: Option<GpuTensor>,       // [n_embd]
    pub moe_gate_batch: Option<GpuTensor>,    // [top_k * moe_intermediate]
    pub moe_up_batch: Option<GpuTensor>,      // [top_k * moe_intermediate]
    pub moe_rot_batch: Option<GpuTensor>,     // [top_k * moe_intermediate]
    pub moe_topk_indices: Option<GpuTensor>,  // [top_k]
    pub moe_topk_weights: Option<GpuTensor>,  // [top_k]
    pub moe_down_expanded: Option<GpuTensor>, // [top_k * n_embd]

    // Snapshot of the post-FFN, pre-LM-head-norm hidden — caller can
    // capture this and feed back as `prev_hidden` for an n+2 prediction.
    pub t_mtp_out: GpuTensor, // [n_embd]

    // LM head output
    pub logits: GpuTensor, // [vocab_size]

    // Optional FastMTP-style compressed-logits scratch — sized
    // [compressed_vocab_size] when the head was loaded with a sidecar.
    // Populated by `mtp_head_forward_compressed` instead of `logits`.
    pub logits_compressed: Option<GpuTensor>,

    // FlashAttention partials buffer for the asym3 attention kernel.
    // Sized [n_heads * max_tiles * (2 + head_dim)] F32 where max_tiles =
    // ceil(max_seq / 128). Required by `attention_flash_asym3`. Trunk
    // sizes this with batch_mult=16 for batched prefill; MTP single-step
    // forward uses batch_mult=1.
    pub flash_partials: GpuTensor,

    // Position scalar — uploaded each forward into a 4-byte device buffer.
    pub pos_buf: DeviceBuffer,
}

impl Qwen35MtpHeadScratch {
    pub fn new(gpu: &mut Gpu, config: &Qwen35MtpHeadConfig) -> HipResult<Self> {
        let dim = config.n_embd;
        let q_dim = config.head_dim * config.n_head;
        let kv_dim = config.head_dim * config.n_head_kv;
        let moe_max_inter = config
            .moe_intermediate_size
            .max(config.shared_expert_intermediate_size);
        let alloc_moe = config.ffn_kind == Qwen35MtpFfnKind::Moe;
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
            moe_router_logits: if alloc_moe {
                Some(gpu.alloc_tensor(&[config.num_experts], DType::F32)?)
            } else {
                None
            },
            moe_scalar_buf: if alloc_moe {
                Some(gpu.alloc_tensor(&[1], DType::F32)?)
            } else {
                None
            },
            moe_x_rot: if alloc_moe {
                Some(gpu.alloc_tensor(&[dim], DType::F32)?)
            } else {
                None
            },
            moe_gate_up_buf: if alloc_moe {
                Some(gpu.alloc_tensor(&[2 * moe_max_inter], DType::F32)?)
            } else {
                None
            },
            moe_gate_buf: if alloc_moe {
                Some(gpu.alloc_tensor(&[moe_max_inter], DType::F32)?)
            } else {
                None
            },
            moe_up_buf: if alloc_moe {
                Some(gpu.alloc_tensor(&[moe_max_inter], DType::F32)?)
            } else {
                None
            },
            moe_ffn_hidden: if alloc_moe {
                Some(gpu.alloc_tensor(&[moe_max_inter], DType::F32)?)
            } else {
                None
            },
            moe_ffn_out: if alloc_moe {
                Some(gpu.alloc_tensor(&[dim], DType::F32)?)
            } else {
                None
            },
            moe_gate_batch: if alloc_moe {
                Some(gpu.alloc_tensor(
                    &[config.num_experts_per_tok * config.moe_intermediate_size],
                    DType::F32,
                )?)
            } else {
                None
            },
            moe_up_batch: if alloc_moe {
                Some(gpu.alloc_tensor(
                    &[config.num_experts_per_tok * config.moe_intermediate_size],
                    DType::F32,
                )?)
            } else {
                None
            },
            moe_rot_batch: if alloc_moe {
                Some(gpu.alloc_tensor(
                    &[config.num_experts_per_tok * config.moe_intermediate_size],
                    DType::F32,
                )?)
            } else {
                None
            },
            moe_topk_indices: if alloc_moe {
                Some(gpu.alloc_tensor(&[config.num_experts_per_tok], DType::F32)?)
            } else {
                None
            },
            moe_topk_weights: if alloc_moe {
                Some(gpu.alloc_tensor(&[config.num_experts_per_tok], DType::F32)?)
            } else {
                None
            },
            moe_down_expanded: if alloc_moe {
                Some(gpu.alloc_tensor(&[config.num_experts_per_tok * dim], DType::F32)?)
            } else {
                None
            },
            t_mtp_out: gpu.alloc_tensor(&[dim], DType::F32)?,
            logits: gpu.alloc_tensor(&[config.vocab_size], DType::F32)?,
            logits_compressed: None,
            flash_partials: {
                // Same sizing as trunk's prefill_partials at qwen35.rs:2822
                // (TILE_SIZE=128) but with batch_mult=1 since MTP forward
                // is single-token. Allocated per scratch instance, lives
                // for the lifetime of the slot.
                let tile_size = 128usize;
                let max_tiles = (config.max_seq + tile_size - 1) / tile_size;
                gpu.alloc_tensor(
                    &[config.n_head * max_tiles * (2 + config.head_dim)],
                    DType::F32,
                )?
            },
            pos_buf: gpu.hip.malloc(4)?,
        })
    }

    /// Allocate the compressed-logits scratch for FastMTP-style forwards.
    /// Idempotent — a no-op if already allocated to the right size. Call
    /// once after head load when the head reports a compressed_vocab_size.
    pub fn ensure_compressed_logits(
        &mut self,
        gpu: &mut Gpu,
        compressed_vocab_size: usize,
    ) -> HipResult<()> {
        if let Some(existing) = self.logits_compressed.as_ref() {
            if existing.numel() == compressed_vocab_size {
                return Ok(());
            }
            // Size changed — drop and reallocate. This shouldn't happen in
            // normal use (sidecar K is fixed at extract time) but we guard
            // anyway.
            if let Some(old) = self.logits_compressed.take() {
                let _ = gpu.free_tensor(old);
            }
        }
        self.logits_compressed = Some(gpu.alloc_tensor(&[compressed_vocab_size], DType::F32)?);
        Ok(())
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
        if let Some(t) = self.moe_router_logits {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_scalar_buf {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_x_rot {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_gate_up_buf {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_gate_buf {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_up_buf {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_ffn_hidden {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_ffn_out {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_gate_batch {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_up_batch {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_rot_batch {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_topk_indices {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_topk_weights {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_down_expanded {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.free_tensor(self.t_mtp_out);
        let _ = gpu.free_tensor(self.logits);
        if let Some(lc) = self.logits_compressed {
            let _ = gpu.free_tensor(lc);
        }
        let _ = gpu.free_tensor(self.flash_partials);
        let _ = gpu.hip.free(self.pos_buf);
    }
}

// ─── KV cache (single-layer, MTP-private) ────────────────────────────────

/// The MTP head has a single attention block, so its KV cache is one
/// per-layer K + V buffer. Separate from the trunk's `KvCache` since the
/// MTP head writes the SAME absolute position the trunk just emitted —
/// reusing the trunk's cache would mean either double-write or
/// snapshot/restore on every cycle.
///
/// **Format:** Q8_0 (8-bit per-block quantized K and V, 34 B/block of 32
/// elements). Picked over F32 (4× larger, no flash-attn kernel) and
/// asym3 (3-bit, ~9% perf regression on canonical due to lower attention
/// fidelity → lower τ at K=5; benchmarked 2026-05-15). Q8 runs the
/// `attention_flash_q8_0` flash-attn tile kernel, preserves attention
/// quality close to F32 (much higher precision than 3-bit), and is 4×
/// smaller than F32 (2 KB / token vs 8 KB at hd=256, n_kv_heads=4).
///
/// Internally wraps a single-layer [`llama::KvCache`] built via
/// `new_gpu_q8` so we share buffer-sizing semantics with the trunk's
/// `--kv-mode q8` path. Single-layer-ness means `inner.k_gpu[0]` /
/// `inner.v_gpu[0]` are the only slots used.
///
/// History (this branch):
/// - v1: hardcoded F32 + naive `attention_f32` (canonical 49 tok/s K=5)
/// - v2: tried asym3 — coherent but -9% perf at K=5 (44 tok/s) due to
///       3-bit K quant lowering attention quality → lower τ ceiling
/// - v3 (current): Q8 — flash-attn enabled, attention quality preserved
/// KV format for the MTP head's per-token decode path. Default is Q8.
///
/// - **Q8**: 8-bit per-block K and V, `attention_q8_0_kv` (non-flash).
///   Reference: -3% perf vs F32 at K=5 per [[mtp-session-state-2026-05-15-compaction]],
///   externally validated by Unsloth's 76.45% Q8 hit-rate claim.
/// - **Asym3**: 3-bit Givens-rotated K + Q8 V, `attention_flash_asym3`.
///   Prior session v2 attempt: -9% perf vs F32 (lower attention fidelity → lower τ).
///   Reverted in favor of Q8. Re-running with the new ±1-3% methodology
///   to confirm/refute that delta.
/// - **Fwht4**: 4-bit signed-FWHT-rotated K + Q8 V, `attention_flash_fwht4`.
///   Master's feat/fwht Phase 2 bench (3.5-27B prose, dflash): Fwht3 fixes
///   asym3 collapse (τ 1.27→3.60) and Fwht2 EXCEEDS Q8 (τ 4.08 vs 3.91).
///   Wired here as the MTP-head's matching test.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MtpKvMode {
    Q8,
    Asym3,
    Fwht4,
}

impl MtpKvMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "q8" => Ok(Self::Q8),
            "asym3" | "turbo3" => Ok(Self::Asym3),
            "fwht4" => Ok(Self::Fwht4),
            other => Err(format!(
                "unknown MTP kv-mode '{other}' (expected: q8, asym3, fwht4)"
            )),
        }
    }
}

pub struct Qwen35MtpHeadKvCache {
    pub inner: hipfire_runtime::llama::KvCache,
    pub max_seq: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub kv_mode: MtpKvMode,
}

impl Qwen35MtpHeadKvCache {
    /// Default constructor: Q8 KV (current ship).
    pub fn new(gpu: &mut Gpu, config: &Qwen35MtpHeadConfig) -> HipResult<Self> {
        Self::new_with_kv_mode(gpu, config, MtpKvMode::Q8)
    }

    /// Allocate the MTP head's single-layer KV cache in the requested format.
    pub fn new_with_kv_mode(
        gpu: &mut Gpu,
        config: &Qwen35MtpHeadConfig,
        kv_mode: MtpKvMode,
    ) -> HipResult<Self> {
        let inner = match kv_mode {
            MtpKvMode::Q8 => hipfire_runtime::llama::KvCache::new_gpu_q8(
                gpu,
                /* n_layers */ 1,
                config.n_head_kv,
                config.head_dim,
                config.max_seq,
            )?,
            MtpKvMode::Asym3 => hipfire_runtime::llama::KvCache::new_gpu_asym3(
                gpu,
                /* n_layers */ 1,
                config.n_head_kv,
                config.head_dim,
                config.max_seq,
            )?,
            MtpKvMode::Fwht4 => hipfire_runtime::llama::KvCache::new_gpu_fwht4(
                gpu,
                /* n_layers */ 1,
                config.n_head_kv,
                config.head_dim,
                config.max_seq,
            )?,
        };
        Ok(Self {
            inner,
            max_seq: config.max_seq,
            n_head_kv: config.n_head_kv,
            head_dim: config.head_dim,
            kv_mode,
        })
    }

    /// Reset positions to all-zeros. Q8 K/V slots are 34 B per 32-element
    /// block stored inside an F32-typed buffer; zeroing the underlying
    /// buffer is equivalent to "no positions written" at the kernel level.
    pub fn reset(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        for layer in 0..self.inner.k_gpu.len() {
            let k_n = self.inner.k_gpu[layer].numel();
            let v_n = self.inner.v_gpu[layer].numel();
            let zeros_k = vec![0.0f32; k_n];
            let zeros_v = vec![0.0f32; v_n];
            let kb: &[u8] =
                unsafe { std::slice::from_raw_parts(zeros_k.as_ptr() as *const u8, k_n * 4) };
            let vb: &[u8] =
                unsafe { std::slice::from_raw_parts(zeros_v.as_ptr() as *const u8, v_n * 4) };
            gpu.hip.memcpy_htod(&self.inner.k_gpu[layer].buf, kb)?;
            gpu.hip.memcpy_htod(&self.inner.v_gpu[layer].buf, vb)?;
        }
        Ok(())
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        // llama::KvCache holds GpuTensors with NO Drop impl — dropping it
        // leaks the k_gpu/v_gpu buffers. Free them explicitly. (The old
        // "they free on Drop" comment was false; see mtp_spec/mtp_compose
        // which already bypass this wrapper for the same reason.)
        self.inner.free_gpu(gpu);
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

// ─── Bundled .mq4-mtp loader (trunk + MTP head in one file) ───────────────

/// Trailer magic written by `mq4_merge_mtp`. Indicates the file is a bundle
/// of a trunk `.mq4` followed by an MTP `.mtp` section.
pub const BUNDLE_TRAILER_MAGIC: &[u8; 8] = b"HFBNDMTP";
/// Trailer is 8 bytes magic + 8 bytes u64 mtp-section offset = 16 bytes.
pub const BUNDLE_TRAILER_LEN: u64 = 16;

/// Inspect a file's trailing 16 bytes for the `mq4_merge_mtp` trailer. If
/// present, returns the byte offset where the embedded MTP `.mtp` section
/// starts. Returns `None` for plain `.mq4` trunk files (no MTP bundled).
///
/// Cheap operation — single 16-byte read from end of file. Safe to call
/// on any path: returns `Ok(None)` rather than erroring when the file is
/// too small or the magic doesn't match.
pub fn detect_bundled_mtp_offset(path: &Path) -> std::io::Result<Option<u64>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path)?;
    let file_size = f.metadata()?.len();
    if file_size < BUNDLE_TRAILER_LEN {
        return Ok(None);
    }
    f.seek(SeekFrom::End(-(BUNDLE_TRAILER_LEN as i64)))?;
    let mut trailer = [0u8; BUNDLE_TRAILER_LEN as usize];
    f.read_exact(&mut trailer)?;
    if &trailer[..8] != BUNDLE_TRAILER_MAGIC {
        return Ok(None);
    }
    let mtp_offset = u64::from_le_bytes(trailer[8..16].try_into().unwrap());
    if mtp_offset >= file_size - BUNDLE_TRAILER_LEN {
        // Corrupt trailer (offset past EOF). Treat as not-bundled rather
        // than crashing — the trunk loader's HFQM parse will succeed
        // independent of trailer state.
        return Ok(None);
    }
    Ok(Some(mtp_offset))
}

/// Load the MTP head section embedded inside a bundled `.mq4-mtp` file.
/// `path` is the bundle file path; the trailer is read to find the embedded
/// MTP section offset, then [`load_mtp_head_at_offset`] parses it.
///
/// Returns `Ok(None)` if `path` is a plain trunk `.mq4` (no bundle trailer).
pub fn load_mtp_head_bundled(
    path: &Path,
    gpu: &mut Gpu,
    max_seq: usize,
) -> HipResult<Option<Qwen35MtpHead>> {
    let mtp_offset = match detect_bundled_mtp_offset(path) {
        Ok(Some(off)) => off,
        Ok(None) => return Ok(None),
        Err(e) => panic!("read bundle trailer from {}: {e}", path.display()),
    };
    let head = load_mtp_head_at_offset(path, gpu, max_seq, mtp_offset)?;
    Ok(Some(head))
}

// ─── Loader ──────────────────────────────────────────────────────────────

/// Load a `.mtp` file (arch_id = 21) created by `mtp_extract` (Task 8).
/// Returns the head ready for `mtp_head_forward`.
///
/// `max_seq` bounds the per-position KV cache later allocated by
/// [`Qwen35MtpHeadKvCache::new`]; pick to match your decode budget.
pub fn load_mtp_head(path: &Path, gpu: &mut Gpu, max_seq: usize) -> HipResult<Qwen35MtpHead> {
    load_mtp_head_at_offset(path, gpu, max_seq, 0)
}

/// Like [`load_mtp_head`] but opens the HFQM container at `base_offset`
/// inside `path`. Pass `0` for a standalone `.mtp` file; for a bundled
/// `.mq4-mtp` file pass the offset returned by [`detect_bundled_mtp_offset`].
pub fn load_mtp_head_at_offset(
    path: &Path,
    gpu: &mut Gpu,
    max_seq: usize,
    base_offset: u64,
) -> HipResult<Qwen35MtpHead> {
    let hfq = HfqFile::open_at_offset(path, base_offset).unwrap_or_else(|e| {
        panic!(
            "open .mtp file {} @ offset {base_offset}: {e}",
            path.display()
        )
    });
    assert_eq!(
        hfq.arch_id,
        21,
        ".mtp file at {} has arch_id={} (expected 21 = QWEN35_MTP_HEAD); \
         is this actually an MTP head extracted by mtp_extract?",
        path.display(),
        hfq.arch_id
    );
    let meta: serde_json::Value =
        serde_json::from_str(&hfq.metadata_json).expect(".mtp metadata JSON parse failed");
    let config = Qwen35MtpHeadConfig::from_metadata(&meta, max_seq);

    // ── Norms (F32, 1D) ─────────────────────────────────────────────────
    //
    // The .mtp file uses bare tensor names ("enorm", "wq", ...) — no
    // "model.language_model." prefix. We read raw bytes and upload
    // directly; `load_weight_tensor` from qwen35.rs unconditionally
    // prepends that prefix and so cannot be reused here.
    let n_embd = config.n_embd;
    let head_dim = config.head_dim;

    // shared_head_norm gets +1.0 like the per-layer norms — empirically
    // verified 2026-05-15 A/B: removing +1.0 regressed K=3 from τ=3.08 to
    // τ=2.00 on 27B-3.5 LRU bench. The MTP head trains its `mtp.norm` with
    // the trunk per-layer convention, NOT the trunk final-norm convention.
    let shared_head_norm = load_norm_raw(&hfq, gpu, "shared_head_norm", n_embd)?;
    let enorm = load_norm_raw(&hfq, gpu, "enorm", n_embd)?;
    let hnorm = load_norm_raw(&hfq, gpu, "hnorm", n_embd)?;
    let attn_norm = load_norm_raw(&hfq, gpu, "attn_norm", n_embd)?;
    let attn_post_norm = load_norm_raw(&hfq, gpu, "attn_post_norm", n_embd)?;
    let attn_q_norm = load_norm_raw(&hfq, gpu, "attn_q_norm", head_dim)?;
    let attn_k_norm = load_norm_raw(&hfq, gpu, "attn_k_norm", head_dim)?;

    // ── 2D weights ──────────────────────────────────────────────────────
    let q_full_dim = 2 * head_dim * config.n_head;
    let kv_dim = head_dim * config.n_head_kv;
    let q_dim = head_dim * config.n_head;

    let eh_proj = load_weight_raw(&hfq, gpu, "eh_proj", n_embd, 2 * n_embd)?;
    let wq = load_weight_raw(&hfq, gpu, "wq", q_full_dim, n_embd)?;
    let wk = load_weight_raw(&hfq, gpu, "wk", kv_dim, n_embd)?;
    let wv = load_weight_raw(&hfq, gpu, "wv", kv_dim, n_embd)?;
    let wo = load_weight_raw(&hfq, gpu, "wo", n_embd, q_dim)?;
    let ffn = match config.ffn_kind {
        Qwen35MtpFfnKind::Dense => Qwen35MtpFfnWeights::Dense(Qwen35MtpDenseFfnWeights {
            gate: load_weight_raw(&hfq, gpu, "ffn_gate", config.n_ff, n_embd)?,
            up: load_weight_raw(&hfq, gpu, "ffn_up", config.n_ff, n_embd)?,
            down: load_weight_raw(&hfq, gpu, "ffn_down", n_embd, config.n_ff)?,
        }),
        Qwen35MtpFfnKind::Moe => {
            assert_eq!(
                config.num_experts_per_tok, 8,
                "MoE MTP runtime currently supports top_k=8, got {}",
                config.num_experts_per_tok
            );
            Qwen35MtpFfnWeights::Moe(load_mtp_moe_ffn(&hfq, gpu, &config)?)
        }
    };

    // ── Optional FastMTP-style compressed lm_head_draft + vocab map ────
    let has_compressed = meta
        .get("has_compressed_lm_head_draft")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let (
        lm_head_draft,
        lm_head_draft_vocab_map,
        lm_head_draft_vocab_map_gpu,
        compressed_vocab_size,
    ) = if has_compressed {
        let cvs = meta
            .get("compressed_vocab_size")
            .and_then(|v| v.as_u64())
            .expect("metadata claims has_compressed_lm_head_draft but lacks compressed_vocab_size")
            as usize;
        assert!(cvs > 0, "compressed_vocab_size must be positive");
        let lm_d = load_weight_raw(&hfq, gpu, "lm_head_draft.weight", cvs, n_embd)?;
        let (vmap_info, vmap_bytes) = hfq
            .tensor_data_vec("lm_head_draft.vocab_map")
            .expect("compressed sidecar missing vocab_map tensor");
        assert_eq!(
            vmap_info.shape,
            vec![cvs as u32],
            "lm_head_draft.vocab_map shape != [compressed_vocab_size]"
        );
        assert_eq!(
            vmap_bytes.len(),
            cvs * 4,
            "lm_head_draft.vocab_map byte count != {} (got {})",
            cvs * 4,
            vmap_bytes.len()
        );
        let vmap: Vec<u32> = vmap_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let vmap_gpu = gpu.upload_raw(&vmap_bytes, &[vmap_bytes.len()])?;
        (Some(lm_d), Some(vmap), Some(vmap_gpu), Some(cvs))
    } else {
        (None, None, None, None)
    };

    let weights = Qwen35MtpHeadWeights {
        shared_head_norm,
        enorm,
        hnorm,
        attn_norm,
        attn_post_norm,
        attn_q_norm,
        attn_k_norm,
        eh_proj,
        wq,
        wk,
        wv,
        wo,
        ffn,
        lm_head_draft,
        lm_head_draft_vocab_map,
        lm_head_draft_vocab_map_gpu,
        compressed_vocab_size,
    };

    Ok(Qwen35MtpHead { config, weights })
}

/// Load a 1D F32 norm tensor from a .mtp file. Mirrors the +1.0 offset
/// convention used by the trunk (Qwen3.5 RMSNorm: `out = x · rsqrt(var+eps)
/// · (1 + weight)`). All `.mtp` norms ship as quant_type=2 (F32) per the
/// `mtp_extract` packing rule.
fn load_norm_raw(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    expected_n: usize,
) -> HipResult<GpuTensor> {
    load_norm_raw_with_offset(hfq, gpu, name, expected_n, true)
}

/// Like load_norm_raw but allows callers to control whether the +1.0 trunk
/// offset is applied. Trunk's per-layer norms get +1; trunk's FINAL norm
/// (`model.norm.weight` → `output_norm`) is RAW. The MTP head's
/// `mtp.norm.weight` (shared_head_norm) is the equivalent of the trunk's
/// final norm — should also be RAW. Pass apply_plus_one=false for that one.
fn load_norm_raw_with_offset(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    expected_n: usize,
    apply_plus_one: bool,
) -> HipResult<GpuTensor> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .unwrap_or_else(|| panic!(".mtp tensor '{name}' missing"));
    let mut f32_data: Vec<f32> = match info.quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => panic!("norm '{name}' has unexpected qt={other} (expected F16 or F32)"),
    };
    assert_eq!(
        f32_data.len(),
        expected_n,
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
    if apply_plus_one {
        for v in &mut f32_data {
            *v += 1.0;
        }
    }
    gpu.upload_f32(&f32_data, &[expected_n])
}

/// Load a 2D quantized weight tensor from a .mtp file. Resolves any of
/// the supported quant types into a [`WeightTensor`]; m and k are passed
/// in (the mtp container stores shape but we trust caller-supplied dims).
fn load_weight_raw(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .unwrap_or_else(|| panic!(".mtp tensor '{name}' missing"));
    sanity_check_2d_shape(name, info, m, k);
    weight_tensor_from_raw(gpu, info.quant_type, &data, m, k, name)
}

fn load_mtp_moe_ffn(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    config: &Qwen35MtpHeadConfig,
) -> HipResult<Qwen35MtpMoeFfnWeights> {
    let n_exp = config.num_experts;
    let dim = config.n_embd;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    assert!(n_exp > 0, "MoE MTP config has num_experts=0");
    assert!(mi > 0, "MoE MTP config has moe_intermediate_size=0");
    assert!(
        smi > 0,
        "MoE MTP config has shared_expert_intermediate_size=0"
    );

    let router = load_weight_raw(hfq, gpu, "moe_router", n_exp, dim)?;
    let shared_expert_gate = load_weight_raw(hfq, gpu, "moe_shared_expert_gate", 1, dim)?;
    let shared_expert = Qwen35MtpMoeSharedExpertWeights {
        gate: load_weight_raw(hfq, gpu, "moe_shared_gate", smi, dim)?,
        up: load_weight_raw(hfq, gpu, "moe_shared_up", smi, dim)?,
        down: load_weight_raw(hfq, gpu, "moe_shared_down", dim, smi)?,
    };

    let mut experts = Vec::with_capacity(n_exp);
    for x in 0..n_exp {
        let gate_up = load_weight_raw(hfq, gpu, &format!("moe_experts.{x}.gate_up"), 2 * mi, dim)?;
        let down = load_weight_raw(hfq, gpu, &format!("moe_experts.{x}.down"), dim, mi)?;
        experts.push(Qwen35MtpMoeExpertWeights { gate_up, down });
    }

    let mut gu_ptrs: Vec<u64> = Vec::with_capacity(n_exp);
    let mut dn_ptrs: Vec<u64> = Vec::with_capacity(n_exp);
    for e in &experts {
        gu_ptrs.push(e.gate_up.buf.buf.as_ptr() as u64);
        dn_ptrs.push(e.down.buf.buf.as_ptr() as u64);
    }
    let gu_bytes: Vec<u8> = gu_ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let dn_bytes: Vec<u8> = dn_ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let expert_gate_up_ptrs = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    let expert_down_ptrs = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    gpu.hip.memcpy_htod(&expert_gate_up_ptrs.buf, &gu_bytes)?;
    gpu.hip.memcpy_htod(&expert_down_ptrs.buf, &dn_bytes)?;

    Ok(Qwen35MtpMoeFfnWeights {
        router,
        shared_expert,
        shared_expert_gate,
        experts,
        expert_gate_up_ptrs,
        expert_down_ptrs,
    })
}

/// Cross-check the on-disk shape against the caller's expected (m, k).
/// Catches silent dim mismatches (e.g. tied vocab, head split done wrong).
fn sanity_check_2d_shape(name: &str, info: &HfqTensorInfo, m: usize, k: usize) {
    if info.shape.len() != 2 {
        panic!(
            ".mtp tensor '{name}': expected 2D shape, got {}D = {:?}",
            info.shape.len(),
            info.shape,
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
    gpu: &Gpu,
    quant_type: u8,
    data: &[u8],
    m: usize,
    k: usize,
    name: &str,
) -> HipResult<WeightTensor> {
    match quant_type {
        13 => {
            // MQ4G256 — must be K%256-aligned (kernel requirement).
            assert!(
                k % 256 == 0,
                ".mtp tensor '{name}' is MQ4G256 with K={k} not divisible by 256"
            );
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
        3 => {
            // Q8_F16 (group_size=32, 34 bytes/group) — same byte layout as
            // GGML Q8_0; existing gemv_q8_0 dispatch works directly.
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::Q8_0,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        1 => {
            // F16 → dequantize on host, upload as F32 (no GPU-native F16
            // GEMV; the trunk does the same conversion in load_weight_tensor_raw).
            let f32_data: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
            let buf = gpu.upload_raw(bytes, &[m, k])?;
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
        2 => {
            // F32 raw.
            let buf = gpu.upload_raw(data, &[m, k])?;
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
        gpu,
        head,
        scratch,
        kv,
        next_token,
        prev_hidden,
        None,
        pos,
        trunk_weights,
    )?;

    // Standard single-step lm_head: shared_head_norm + GEMV over t_mtp_out.
    let w = &head.weights;
    gpu.rmsnorm_f32(
        &scratch.t_mtp_out,
        &w.shared_head_norm,
        &scratch.tmp,
        cfg.rms_norm_eps,
    )?;
    weight_gemv(gpu, lm_head_weights, &scratch.tmp, &scratch.logits)?;

    Ok(())
}

/// FastMTP-style compressed forward. Identical to [`mtp_head_forward`]
/// but dispatches the LM head against the head's bundled compressed
/// `lm_head_draft` (shape `[K, n_embd]`) instead of the trunk's full
/// `[V, n_embd]` head. Writes logits into `scratch.logits_compressed`
/// (must be pre-allocated via [`Qwen35MtpHeadScratch::ensure_compressed_logits`]).
///
/// Caller does argmax over the K-element compressed logit vector and
/// remaps the index to a full-vocab token id via `weights.lm_head_draft_vocab_map`
/// (a tiny CPU-side `Vec<u32>`).
///
/// Lossless greedy guarantee: trunk verification is unchanged (still
/// uses the full vocab head), so any out-of-K-vocab token the trunk
/// would have emitted is automatically rejected by the argmax mismatch.
/// Only effect of an unrepresentative sidecar is degraded τ — the engine
/// emits the same tokens it would have emitted in plain AR.
///
/// Panics if the head was loaded without a compressed sidecar (check
/// `head.weights.lm_head_draft.is_some()` before calling).
pub fn mtp_head_forward_compressed(
    gpu: &mut Gpu,
    head: &Qwen35MtpHead,
    scratch: &Qwen35MtpHeadScratch,
    kv: &mut Qwen35MtpHeadKvCache,
    next_token: u32,
    prev_hidden: &GpuTensor,
    pos: usize,
    trunk_weights: &Qwen35Weights,
) -> HipResult<()> {
    // Block forward — identical to mtp_head_forward; produces t_mtp_out.
    mtp_head_forward_block_only(
        gpu,
        head,
        scratch,
        kv,
        next_token,
        prev_hidden,
        None,
        pos,
        trunk_weights,
    )?;

    mtp_head_apply_lm_head_draft(gpu, head, scratch)?;

    Ok(())
}

/// Applies the compressed draft LM head to `scratch.t_mtp_out`.
///
/// This is the shared tail of the host-mediated compressed path and the
/// device-token-chain path. Keeping it in one helper prevents those paths from
/// drifting if the compressed head gains another post-block transform.
pub fn mtp_head_apply_lm_head_draft(
    gpu: &mut Gpu,
    head: &Qwen35MtpHead,
    scratch: &Qwen35MtpHeadScratch,
) -> HipResult<()> {
    let cfg = &head.config;
    let w = &head.weights;
    let lm_head_draft = w
        .lm_head_draft
        .as_ref()
        .expect("mtp_head_apply_lm_head_draft called but head has no lm_head_draft sidecar");
    let logits_c = scratch.logits_compressed.as_ref().expect(
        "mtp_head_apply_lm_head_draft: scratch.logits_compressed not allocated; \
                 call Qwen35MtpHeadScratch::ensure_compressed_logits first",
    );

    assert_eq!(
        lm_head_draft.k, cfg.n_embd,
        "mtp_head_apply_lm_head_draft: lm_head_draft.k={} but n_embd={}",
        lm_head_draft.k, cfg.n_embd,
    );

    gpu.rmsnorm_f32(
        &scratch.t_mtp_out,
        &w.shared_head_norm,
        &scratch.tmp,
        cfg.rms_norm_eps,
    )?;
    weight_gemv(gpu, lm_head_draft, &scratch.tmp, logits_c)
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
    let n_embd = cfg.n_embd;

    assert_eq!(
        prev_hidden.numel(),
        n_embd,
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
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;

    mtp_head_forward_block_only_with_pos_buf(
        gpu,
        head,
        scratch,
        kv,
        next_token,
        prev_hidden,
        next_token_embed,
        &scratch.pos_buf,
        pos,
        pos + 1,
        trunk_weights,
    )
}

/// Graph-capturable variant of [`mtp_head_forward_block_only`].
///
/// `pos_buf` must point at an i32 device slot containing `pos`. Captured
/// proposal graphs pass a stable per-step slot here and update the slot bytes
/// before each graph launch, avoiding scalar position arguments baked into
/// captured nodes. `seq_len_hint` controls the attention launch shape and
/// shared-memory reservation; graph callers pass a tier cap that is >= every
/// replayed `pos + 1`.
#[allow(clippy::too_many_arguments)]
pub fn mtp_head_forward_block_only_with_pos_buf(
    gpu: &mut Gpu,
    head: &Qwen35MtpHead,
    scratch: &Qwen35MtpHeadScratch,
    kv: &mut Qwen35MtpHeadKvCache,
    next_token: u32,
    prev_hidden: &GpuTensor,
    next_token_embed: Option<&GpuTensor>,
    pos_buf: &DeviceBuffer,
    pos: usize,
    seq_len_hint: usize,
    trunk_weights: &Qwen35Weights,
) -> HipResult<()> {
    let cfg = &head.config;
    let w = &head.weights;
    let n_embd = cfg.n_embd;

    assert_eq!(
        prev_hidden.numel(),
        n_embd,
        "mtp_head_forward_block_only_with_pos_buf: prev_hidden has {} elems but expected n_embd={n_embd}",
        prev_hidden.numel(),
    );
    assert!(
        pos < kv.max_seq,
        "mtp_head_forward_block_only_with_pos_buf: pos={pos} >= kv.max_seq={}",
        kv.max_seq,
    );

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
        gpu.memcpy_dtod_at_auto(&scratch.tok_embd.buf, 0, &embed.buf, 0, dim_bytes)?;
    } else {
        embed_lookup_into(gpu, trunk_weights, &scratch.tok_embd, next_token, n_embd)?;
    }

    // ── 2. RMSNorm both inputs to the NextN projection ───────────────────
    gpu.rmsnorm_f32(
        &scratch.tok_embd,
        &w.enorm,
        &scratch.e_norm,
        cfg.rms_norm_eps,
    )?;
    gpu.rmsnorm_f32(prev_hidden, &w.hnorm, &scratch.h_norm, cfg.rms_norm_eps)?;

    // ── 3. concat = [e_norm | h_norm], then cur = eh_proj @ concat ───────
    gpu.memcpy_dtod_at_auto(&scratch.concat.buf, 0, &scratch.e_norm.buf, 0, dim_bytes)?;
    gpu.memcpy_dtod_at_auto(
        &scratch.concat.buf,
        dim_bytes,
        &scratch.h_norm.buf,
        0,
        dim_bytes,
    )?;
    weight_gemv(gpu, &w.eh_proj, &scratch.concat, &scratch.cur)?;

    // Save inpSA for the attention residual (cur is about to be norm'd
    // out-of-place into scratch.tmp).
    gpu.memcpy_dtod_at_auto(&scratch.residual.buf, 0, &scratch.cur.buf, 0, dim_bytes)?;

    // ── 4. Pre-attn norm + Q/K/V projections ─────────────────────────────
    gpu.rmsnorm_f32(&scratch.cur, &w.attn_norm, &scratch.tmp, cfg.rms_norm_eps)?;

    // Qwen3.5 gated-Q: wq emits 2 * head_dim * n_head, deinterleaved into
    // Q (head-major first half) and gate (second half) per-head. Mirror
    // qwen35.rs:2402-2414.
    weight_gemv(gpu, &w.wq, &scratch.tmp, &scratch.q_full)?;
    gpu.deinterleave_f32(
        &scratch.q_full,
        &scratch.q,
        &scratch.gate,
        cfg.n_head,
        cfg.head_dim,
    )?;
    gpu.rmsnorm_batched(
        &scratch.q,
        &w.attn_q_norm,
        &scratch.q,
        cfg.n_head,
        cfg.head_dim,
        cfg.rms_norm_eps,
    )?;

    weight_gemv(gpu, &w.wk, &scratch.tmp, &scratch.k)?;
    weight_gemv(gpu, &w.wv, &scratch.tmp, &scratch.v)?;
    gpu.rmsnorm_batched(
        &scratch.k,
        &w.attn_k_norm,
        &scratch.k,
        cfg.n_head_kv,
        cfg.head_dim,
        cfg.rms_norm_eps,
    )?;

    // ── 5. RoPE (partial-interleaved, mirrors trunk's full-attn layer) ───
    gpu.rope_partial_interleaved_f32(
        &scratch.q,
        &scratch.k,
        pos_buf,
        cfg.n_head,
        cfg.n_head_kv,
        cfg.head_dim,
        cfg.n_rot,
        cfg.rope_theta,
    )?;

    // ── 6+7. KV cache write + attention (dispatch on kv.kv_mode) ─────────
    //
    // Mirrors trunk's per-token decode dispatch at qwen35.rs:6062-6138.
    // - Q8: 2-call kv_cache_write_q8_0 + attention_q8_0_kv (no flash partials)
    // - Asym3: kv_cache_write_asym3_fused + attention_flash_asym3 (Givens cos/sin)
    // - Fwht4: kv_cache_write_fwht4_fused + attention_flash_fwht4 (FWHT signs
    //   stored in kv_cache.givens_cos/givens_sin slots — field-name reuse
    //   per Phase 1 fwht4 commit `c64c0e3f`).
    match kv.kv_mode {
        MtpKvMode::Q8 => {
            gpu.kv_cache_write_q8_0(
                &kv.inner.k_gpu[0],
                &scratch.k,
                pos_buf,
                cfg.n_head_kv,
                cfg.head_dim,
            )?;
            gpu.kv_cache_write_q8_0(
                &kv.inner.v_gpu[0],
                &scratch.v,
                pos_buf,
                cfg.n_head_kv,
                cfg.head_dim,
            )?;
            gpu.attention_q8_0_kv(
                &scratch.q,
                &kv.inner.k_gpu[0],
                &kv.inner.v_gpu[0],
                &scratch.attn_out,
                pos_buf,
                seq_len_hint,
                cfg.n_head,
                cfg.n_head_kv,
                cfg.head_dim,
                kv.inner.physical_cap,
            )?;
        }
        MtpKvMode::Asym3 => {
            let ct = kv
                .inner
                .givens_cos
                .as_ref()
                .expect("MtpKvMode::Asym3 requires kv.inner.givens_cos to be Some");
            let st = kv
                .inner
                .givens_sin
                .as_ref()
                .expect("MtpKvMode::Asym3 requires kv.inner.givens_sin to be Some");
            gpu.kv_cache_write_asym3_fused(
                &kv.inner.k_gpu[0],
                &kv.inner.v_gpu[0],
                &scratch.k,
                &scratch.v,
                pos_buf,
                ct,
                st,
                cfg.n_head_kv,
                cfg.head_dim,
            )?;
            gpu.attention_flash_asym3(
                &scratch.q,
                &kv.inner.k_gpu[0],
                &kv.inner.v_gpu[0],
                &scratch.attn_out,
                pos_buf,
                ct,
                st,
                seq_len_hint,
                cfg.n_head,
                cfg.n_head_kv,
                cfg.head_dim,
                kv.inner.physical_cap,
                &scratch.flash_partials,
            )?;
        }
        MtpKvMode::Fwht4 => {
            // Fwht uses the asym4-shaped k/v buffers + the cos/sin slots hold
            // signs1/signs2 (128 elements each). FWHT operates on 128-element
            // halves; head_dim=256 processes 2 halves reusing the same signs.
            let ct = kv
                .inner
                .givens_cos
                .as_ref()
                .expect("MtpKvMode::Fwht4 requires kv.inner.givens_cos (signs1) to be Some");
            let st = kv
                .inner
                .givens_sin
                .as_ref()
                .expect("MtpKvMode::Fwht4 requires kv.inner.givens_sin (signs2) to be Some");
            gpu.kv_cache_write_fwht4_fused(
                &kv.inner.k_gpu[0],
                &kv.inner.v_gpu[0],
                &scratch.k,
                &scratch.v,
                pos_buf,
                ct,
                st,
                cfg.n_head_kv,
                cfg.head_dim,
                kv.inner.v_mode_bits(),
            )?;
            gpu.attention_flash_fwht4(
                &scratch.q,
                &kv.inner.k_gpu[0],
                &kv.inner.v_gpu[0],
                &scratch.attn_out,
                pos_buf,
                ct,
                st,
                seq_len_hint,
                cfg.n_head,
                cfg.n_head_kv,
                cfg.head_dim,
                kv.inner.physical_cap,
                &scratch.flash_partials,
                kv.inner.v_mode_bits(),
            )?;
        }
    }

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
    gpu.rmsnorm_f32(
        &scratch.o,
        &w.attn_post_norm,
        &scratch.tmp,
        cfg.rms_norm_eps,
    )?;
    match &w.ffn {
        Qwen35MtpFfnWeights::Dense(ffn) => {
            weight_gemv(gpu, &ffn.gate, &scratch.tmp, &scratch.gate_ffn)?;
            weight_gemv(gpu, &ffn.up, &scratch.tmp, &scratch.up)?;
            gpu.silu_mul_f32(&scratch.gate_ffn, &scratch.up, &scratch.ffn_hidden)?;
            weight_gemv(gpu, &ffn.down, &scratch.ffn_hidden, &scratch.ffn_out)?;
            gpu.add_inplace_f32(&scratch.ffn_out, &scratch.o)?;
        }
        Qwen35MtpFfnWeights::Moe(ffn) => {
            gpu.memcpy_dtod_at_auto(&scratch.ffn_out.buf, 0, &scratch.o.buf, 0, dim_bytes)?;
            mtp_moe_ffn_decode(gpu, ffn, &scratch.tmp, &scratch.ffn_out, cfg, scratch)?;
        }
    }
    // scratch.ffn_out now holds the post-FFN, pre-LM-head-norm hidden.

    // Snapshot for callers that want to chain into n+2 prediction OR feed
    // into the batched `mtp_head_apply_lm_head_batched` end-of-chain reduce.
    gpu.memcpy_dtod_at_auto(
        &scratch.t_mtp_out.buf,
        0,
        &scratch.ffn_out.buf,
        0,
        dim_bytes,
    )?;

    Ok(())
}

fn mtp_moe_ffn_decode(
    gpu: &mut Gpu,
    ffn: &Qwen35MtpMoeFfnWeights,
    x_norm: &GpuTensor,
    x_residual: &GpuTensor,
    cfg: &Qwen35MtpHeadConfig,
    scratch: &Qwen35MtpHeadScratch,
) -> HipResult<()> {
    let dim = cfg.n_embd;
    let mi = cfg.moe_intermediate_size;
    let smi = cfg.shared_expert_intermediate_size;
    let k_top = cfg.num_experts_per_tok;
    assert_eq!(k_top, 8, "MoE MTP decode currently expects top_k=8");
    assert_eq!(ffn.experts.len(), cfg.num_experts);

    let router_logits = scratch
        .moe_router_logits
        .as_ref()
        .expect("MoE MTP scratch not allocated");
    let scalar_buf = scratch.moe_scalar_buf.as_ref().expect("MoE MTP scratch");
    let x_rot = scratch.moe_x_rot.as_ref().expect("MoE MTP scratch");
    let gate_buf = scratch.moe_gate_buf.as_ref().expect("MoE MTP scratch");
    let up_buf = scratch.moe_up_buf.as_ref().expect("MoE MTP scratch");
    let ffn_hidden = scratch.moe_ffn_hidden.as_ref().expect("MoE MTP scratch");
    let ffn_out = scratch.moe_ffn_out.as_ref().expect("MoE MTP scratch");
    let gate_batch = scratch.moe_gate_batch.as_ref().expect("MoE MTP scratch");
    let up_batch = scratch.moe_up_batch.as_ref().expect("MoE MTP scratch");
    let rot_batch = scratch.moe_rot_batch.as_ref().expect("MoE MTP scratch");
    let topk_indices = scratch.moe_topk_indices.as_ref().expect("MoE MTP scratch");
    let topk_weights = scratch.moe_topk_weights.as_ref().expect("MoE MTP scratch");
    let down_expanded = scratch.moe_down_expanded.as_ref().expect("MoE MTP scratch");

    weight_gemv(gpu, &ffn.router, x_norm, router_logits)?;
    gpu.softmax_f32(router_logits)?;
    gpu.moe_topk_renorm_k8(
        router_logits,
        topk_indices,
        topk_weights,
        cfg.num_experts,
        cfg.norm_topk_prob,
    )?;

    weight_gemv(gpu, &ffn.shared_expert_gate, x_norm, scalar_buf)?;
    let shared_gate = gate_buf.sub_offset(0, smi);
    let shared_up = up_buf.sub_offset(0, smi);
    weight_gemv(gpu, &ffn.shared_expert.gate, x_norm, &shared_gate)?;
    weight_gemv(gpu, &ffn.shared_expert.up, x_norm, &shared_up)?;
    if ffn.shared_expert.down.gpu_dtype == DType::MQ4G256 {
        gpu.ensure_mq_signs()?;
        let x_rot_alias = GpuTensor {
            buf: unsafe { gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias() },
            shape: vec![gpu.scratch.mq_x_rot.as_ref().unwrap().buf.size() / 4],
            dtype: DType::F32,
        };
        fused_silu_mul_rotate_mq_for(
            gpu,
            &ffn.shared_expert.down,
            &shared_gate,
            &shared_up,
            &x_rot_alias,
            smi,
        )?;
        gpu.gemv_hfq4g256_residual_sigmoid_scaled_gpu(
            &ffn.shared_expert.down.buf,
            &x_rot_alias,
            x_residual,
            scalar_buf,
            ffn.shared_expert.down.m,
            ffn.shared_expert.down.k,
        )?;
    } else {
        gpu.sigmoid_f32(scalar_buf)?;
        let shared_hid = ffn_hidden.sub_offset(0, smi);
        gpu.silu_mul_f32(&shared_gate, &shared_up, &shared_hid)?;
        weight_gemv(gpu, &ffn.shared_expert.down, &shared_hid, ffn_out)?;
        gpu.scaled_add_inplace_gpu_scalar_f32(x_residual, ffn_out, scalar_buf)?;
    }

    let e0 = ffn.experts.first().expect("MoE MTP has no routed experts");
    assert_eq!(
        e0.gate_up.gpu_dtype,
        DType::MQ4G256,
        "MoE MTP routed gate_up currently requires MQ4G256"
    );
    assert_eq!(
        e0.down.gpu_dtype,
        DType::MQ4G256,
        "MoE MTP routed down currently requires MQ4G256"
    );
    rotate_x_mq_for(gpu, &e0.gate_up, x_norm, x_rot, dim)?;
    gpu.gemv_hfq4g256_moe_gate_up_k8_indexed(
        &ffn.expert_gate_up_ptrs,
        topk_indices,
        x_rot,
        gate_batch,
        up_batch,
        2 * mi,
        e0.gate_up.k,
    )?;
    fused_silu_mul_rotate_mq_batched_for(
        gpu, &e0.down, gate_batch, up_batch, rot_batch, mi, k_top,
    )?;
    gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
        &ffn.expert_down_ptrs,
        topk_indices,
        rot_batch,
        down_expanded,
        e0.down.m,
        e0.down.k,
        k_top,
        1,
    )?;
    gpu.moe_down_combine_k8_batched(down_expanded, topk_weights, x_residual, e0.down.m, k_top, 1)?;

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
    // Output dim = the lm_head weight's row count. For the FULL trunk lm_head
    // this matches cfg.vocab_size; for FastMTP-style compressed lm_head_draft
    // this is the (smaller) compressed_vocab_size — the function is parametric
    // on whichever WeightTensor the caller passes in.
    let vocab = lm_head_weights.m;
    assert_eq!(
        lm_head_weights.k, n_embd,
        "mtp_head_apply_lm_head_batched: lm_head_weights.k={} but n_embd={n_embd}",
        lm_head_weights.k,
    );
    assert!(
        t_mtp_outs_stacked.numel() >= n * n_embd,
        "t_mtp_outs_stacked too small: {} < n*n_embd ({})",
        t_mtp_outs_stacked.numel(),
        n * n_embd,
    );
    assert!(
        tmp_batched.numel() >= n * n_embd,
        "tmp_batched too small: {} < n*n_embd ({})",
        tmp_batched.numel(),
        n * n_embd,
    );
    assert!(
        logits_batched.numel() >= n * vocab,
        "logits_batched too small: {} < n*vocab ({})",
        logits_batched.numel(),
        n * vocab,
    );

    // Per-row shared_head_norm.
    gpu.rmsnorm_batched(
        t_mtp_outs_stacked,
        &w.shared_head_norm,
        tmp_batched,
        n,
        n_embd,
        cfg.rms_norm_eps,
    )?;

    // Per-dtype batched LM head dispatch (mirrors mtp_probe.rs:278+).
    let logits_view = logits_batched.sub_offset(0, n * vocab);
    match lm_head_weights.gpu_dtype {
        DType::Q8_0 => {
            gpu.gemm_q8_0_batched(
                &lm_head_weights.buf,
                tmp_batched,
                &logits_view,
                lm_head_weights.m,
                lm_head_weights.k,
                n,
            )?;
        }
        DType::HFQ4G256 => {
            gpu.gemm_hfq4g256_batched_lmhead(
                &lm_head_weights.buf,
                tmp_batched,
                &logits_view,
                lm_head_weights.m,
                lm_head_weights.k,
                n,
            )?;
        }
        DType::MQ4G256 => {
            let rot_view = rot_batched.sub_offset(0, n * lm_head_weights.k);
            llama::rotate_x_mq_batched_for(
                gpu,
                lm_head_weights,
                tmp_batched,
                &rot_view,
                lm_head_weights.k,
                n,
            )?;
            gpu.gemm_hfq4g256_batched_lmhead(
                &lm_head_weights.buf,
                &rot_view,
                &logits_view,
                lm_head_weights.m,
                lm_head_weights.k,
                n,
            )?;
        }
        DType::MQ3G256 => {
            let rot_view = rot_batched.sub_offset(0, n * lm_head_weights.k);
            llama::rotate_x_mq_batched_for(
                gpu,
                lm_head_weights,
                tmp_batched,
                &rot_view,
                lm_head_weights.k,
                n,
            )?;
            gpu.gemm_hfq3g256_batched_lmhead(
                &lm_head_weights.buf,
                &rot_view,
                &logits_view,
                lm_head_weights.m,
                lm_head_weights.k,
                n,
            )?;
        }
        DType::HFQ6G256 => {
            gpu.gemm_hfq6g256_batched_lmhead(
                &lm_head_weights.buf,
                tmp_batched,
                &logits_view,
                lm_head_weights.m,
                lm_head_weights.k,
                n,
            )?;
        }
        DType::MQ6G256 => {
            let rot_view = rot_batched.sub_offset(0, n * lm_head_weights.k);
            llama::rotate_x_mq_batched_for(
                gpu,
                lm_head_weights,
                tmp_batched,
                &rot_view,
                lm_head_weights.k,
                n,
            )?;
            gpu.gemm_hfq6g256_batched_lmhead(
                &lm_head_weights.buf,
                &rot_view,
                &logits_view,
                lm_head_weights.m,
                lm_head_weights.k,
                n,
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
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, out, token, dim)
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, out, token, dim)
        }
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, out, token, dim),
        EmbeddingFormat::Q4K => gpu.embedding_lookup_q4k(&weights.token_embd, out, token, dim),
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, out, token, dim),
    }
}

// ─── Batched MTP head forward (Task 11b) ─────────────────────────────────

/// Per-call GPU scratch for the batched MTP head forward — sized for up to
/// `max_n` MTP forwards processed in a single per-layer GEMM batch. Mirrors
/// [`Qwen35MtpHeadScratch`] but with each tensor sized `max_n × <dim>`.
///
/// All buffers are row-major contiguous: row `i` for slot i lives at byte
/// offset `i * <dim> * 4`, exactly as the trunk's batched-prefill scratch
/// expects (and matches the layout the existing batched GEMM /
/// rmsnorm_batched / rope_batched / deinterleave_batched kernels read).
pub struct Qwen35MtpHeadBatchedScratch {
    pub max_n: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    // Activations
    pub tok_embd: GpuTensor, // [max_n × n_embd]
    pub e_norm: GpuTensor,   // [max_n × n_embd]
    pub h_norm: GpuTensor,   // [max_n × n_embd]
    pub concat: GpuTensor,   // [max_n × 2 × n_embd]
    pub cur: GpuTensor,      // [max_n × n_embd]
    pub residual: GpuTensor, // [max_n × n_embd]
    pub tmp: GpuTensor,      // [max_n × n_embd]
    // Attention sub-block
    pub q_full: GpuTensor,   // [max_n × 2 × q_dim]
    pub q: GpuTensor,        // [max_n × q_dim]
    pub gate: GpuTensor,     // [max_n × q_dim]
    pub k: GpuTensor,        // [max_n × kv_dim]
    pub v: GpuTensor,        // [max_n × kv_dim]
    pub attn_out: GpuTensor, // [max_n × q_dim]
    pub o: GpuTensor,        // [max_n × n_embd]
    // FFN sub-block
    pub gate_ffn: GpuTensor,   // [max_n × n_ff]
    pub up: GpuTensor,         // [max_n × n_ff]
    pub ffn_hidden: GpuTensor, // [max_n × n_ff]
    pub ffn_out: GpuTensor,    // [max_n × n_embd]
    /// Stacked post-FFN, pre-LM-head-norm hidden — feeds
    /// `mtp_head_apply_lm_head_batched` directly. [max_n × n_embd]
    pub t_mtp_outs: GpuTensor,
    /// Per-slot positions device buffer. [max_n] i32, uploaded fresh each call.
    pub positions: GpuTensor,
}

impl Qwen35MtpHeadBatchedScratch {
    pub fn new(gpu: &mut Gpu, config: &Qwen35MtpHeadConfig, max_n: usize) -> HipResult<Self> {
        assert!(
            max_n >= 1,
            "Qwen35MtpHeadBatchedScratch: max_n must be >= 1"
        );
        let dim = config.n_embd;
        let q_dim = config.head_dim * config.n_head;
        let kv_dim = config.head_dim * config.n_head_kv;
        Ok(Self {
            max_n,
            n_embd: dim,
            n_ff: config.n_ff,
            q_dim,
            kv_dim,
            tok_embd: gpu.alloc_tensor(&[max_n * dim], DType::F32)?,
            e_norm: gpu.alloc_tensor(&[max_n * dim], DType::F32)?,
            h_norm: gpu.alloc_tensor(&[max_n * dim], DType::F32)?,
            concat: gpu.alloc_tensor(&[max_n * 2 * dim], DType::F32)?,
            cur: gpu.alloc_tensor(&[max_n * dim], DType::F32)?,
            residual: gpu.alloc_tensor(&[max_n * dim], DType::F32)?,
            tmp: gpu.alloc_tensor(&[max_n * dim], DType::F32)?,
            q_full: gpu.alloc_tensor(&[max_n * 2 * q_dim], DType::F32)?,
            q: gpu.alloc_tensor(&[max_n * q_dim], DType::F32)?,
            gate: gpu.alloc_tensor(&[max_n * q_dim], DType::F32)?,
            k: gpu.alloc_tensor(&[max_n * kv_dim], DType::F32)?,
            v: gpu.alloc_tensor(&[max_n * kv_dim], DType::F32)?,
            attn_out: gpu.alloc_tensor(&[max_n * q_dim], DType::F32)?,
            o: gpu.alloc_tensor(&[max_n * dim], DType::F32)?,
            gate_ffn: gpu.alloc_tensor(&[max_n * config.n_ff], DType::F32)?,
            up: gpu.alloc_tensor(&[max_n * config.n_ff], DType::F32)?,
            ffn_hidden: gpu.alloc_tensor(&[max_n * config.n_ff], DType::F32)?,
            ffn_out: gpu.alloc_tensor(&[max_n * dim], DType::F32)?,
            t_mtp_outs: gpu.alloc_tensor(&[max_n * dim], DType::F32)?,
            positions: gpu.alloc_tensor(&[max_n], DType::F32)?, // F32 alias for i32 storage
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
        let _ = gpu.free_tensor(self.t_mtp_outs);
        let _ = gpu.free_tensor(self.positions);
    }
}

/// Per-format BATCHED weight GEMM dispatch for the MTP head's inner-layer
/// projections. M is the OUTPUT row count, K is the input dim. Each batch
/// row `i` reads `x[i * k..(i+1) * k]` and writes `y[i * m..(i+1) * m]`.
///
/// Reuses the same kernels the trunk's prefill batch + lm_head batched paths
/// use (no new kernels).
fn weight_gemm_batched(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x_batched: &GpuTensor,
    y_batched: &GpuTensor,
    n: usize,
    rotated_x_scratch: Option<&GpuTensor>,
) -> HipResult<()> {
    match w.gpu_dtype {
        DType::Q8_0 => gpu.gemm_q8_0_batched(&w.buf, x_batched, y_batched, w.m, w.k, n),
        DType::HFQ4G256 => gpu.gemm_hfq4g256(&w.buf, x_batched, y_batched, w.m, w.k, n),
        DType::MQ4G256 => {
            // MQ4 needs an FWHT-rotated x first (matches trunk lm_head + dflash patterns).
            let rot = rotated_x_scratch.expect("MQ4 batched gemm requires rotated_x_scratch");
            llama::rotate_x_mq_batched_for(gpu, w, x_batched, rot, w.k, n)?;
            gpu.gemm_hfq4g256(&w.buf, rot, y_batched, w.m, w.k, n)
        }
        DType::F32 => {
            // Fallback: per-row gemv (slow but correct). MTP head loaded via
            // load_weight_raw can ship F32 for rare formats — keep functional.
            for i in 0..n {
                let x_row = x_batched.sub_offset(i * w.k, w.k);
                let y_row = y_batched.sub_offset(i * w.m, w.m);
                weight_gemv(gpu, w, &x_row, &y_row)?;
            }
            Ok(())
        }
        other => panic!("weight_gemm_batched: unsupported dtype {:?}", other),
    }
}

/// Batched MTP head block forward (Task 11b).
///
/// Runs the NextN concat + eh_proj + gated-Q attention + SwiGLU FFN over
/// `n` parallel slots in a single per-layer GEMM batch. Each slot has its
/// OWN `prev_hidden` and `next_token`; the K/V for slot `i` are written to
/// the MTP head's KV cache at `positions[i]`.
///
/// ## v1 simplification: per-slot independent attention
///
/// Each slot's attention reads ONLY its own freshly-written K/V at the
/// caller-supplied position. There's no cross-slot attention, no historical
/// reads from previous cycles' MTP slots. This is the v1 simplification —
/// the MTP head was trained with full historical attention, so τ_mtp may
/// drop, but the trunk verify is the correctness gate: bad MTP candidates
/// just get rejected.
///
/// Concretely, attention reduces to: `out[h, d] = V[h, d]` (single-key
/// attention with self-only key collapses softmax to 1). The gate
/// (sigmoid_mul) then modulates V before `wo`. We still WRITE K/V to the
/// per-slot KV slot so future cycles' historical reads (when wired) would
/// see the correct cache state — slot writes are cheap and keep the cache
/// snapshot invariant simple.
///
/// ## Inputs
///
/// - `prev_hiddens_stacked`: `[n × n_embd]` row-major. Slot i's drafter /
///   trunk hidden that gets fed into `hnorm`.
/// - `next_tokens`: length-`n` slice of u32 token ids — embedded via the
///   trunk's `token_embd` table, then RMSNorm'd via `enorm`.
/// - `positions`: length-`n` slice of i32 absolute MTP-cache slots. Each
///   `positions[i]` MUST be < `kv.max_seq` and unique (or the K/V writes
///   collide). Caller is responsible for slot allocation.
///
/// ## Output
///
/// - `scratch.t_mtp_outs[i]` holds the post-FFN hidden for slot i. Pass the
///   full `t_mtp_outs` view to [`mtp_head_apply_lm_head_batched`] for the
///   batched lm_head step.
///
/// ## Side effects
///
/// - K/V for slot i written into `kv` at slot `positions[i]`. Caller may
///   call `kv.reset(gpu)` after the cycle if the writes shouldn't persist.
#[allow(clippy::too_many_arguments)]
pub fn mtp_head_forward_block_batched(
    gpu: &mut Gpu,
    head: &Qwen35MtpHead,
    scratch: &mut Qwen35MtpHeadBatchedScratch,
    kv: &mut Qwen35MtpHeadKvCache,
    next_tokens: &[u32],
    prev_hiddens_stacked: &GpuTensor,
    positions: &[i32],
    n: usize,
    trunk_weights: &Qwen35Weights,
    rotated_x_scratch: Option<&GpuTensor>,
) -> HipResult<()> {
    let cfg = &head.config;
    let w = &head.weights;
    let dim = cfg.n_embd;
    let q_dim = cfg.head_dim * cfg.n_head;
    let kv_dim = cfg.head_dim * cfg.n_head_kv;
    let dim_bytes = dim * 4;

    assert_eq!(next_tokens.len(), n, "next_tokens.len() != n");
    assert_eq!(positions.len(), n, "positions.len() != n");
    assert!(
        n <= scratch.max_n,
        "n={n} > scratch.max_n={}",
        scratch.max_n
    );
    assert!(
        prev_hiddens_stacked.numel() >= n * dim,
        "prev_hiddens_stacked too small: {} < n*dim ({})",
        prev_hiddens_stacked.numel(),
        n * dim
    );
    for &p in positions {
        assert!(
            (p as usize) < kv.max_seq,
            "position {p} >= kv.max_seq {}",
            kv.max_seq,
        );
    }

    // Upload positions (i32 stored in F32-typed buffer; aliasing is fine since
    // the kernels read raw bytes via memcpy_htod into the F32 buffer).
    {
        let bytes = unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, n * 4) };
        gpu.hip.memcpy_htod(&scratch.positions.buf, bytes)?;
    }

    // ── 1. Per-slot token embeddings into stacked tok_embd ───────────────
    // No batched embedding-lookup kernel; per-slot dispatch is cheap (n×fast lookups).
    for (i, &tok) in next_tokens.iter().enumerate() {
        let dst = scratch.tok_embd.sub_offset(i * dim, dim);
        embed_lookup_into(gpu, trunk_weights, &dst, tok, dim)?;
    }

    // ── 2. Batched RMSNorm both inputs to NextN projection ───────────────
    let tok_embd_view = scratch.tok_embd.sub_offset(0, n * dim);
    let e_norm_view = scratch.e_norm.sub_offset(0, n * dim);
    let prev_view = prev_hiddens_stacked.sub_offset(0, n * dim);
    let h_norm_view = scratch.h_norm.sub_offset(0, n * dim);
    gpu.rmsnorm_batched(
        &tok_embd_view,
        &w.enorm,
        &e_norm_view,
        n,
        dim,
        cfg.rms_norm_eps,
    )?;
    gpu.rmsnorm_batched(&prev_view, &w.hnorm, &h_norm_view, n, dim, cfg.rms_norm_eps)?;

    // ── 3. Build concat = [e_norm | h_norm] per slot ─────────────────────
    // Layout: concat[i] = [e_norm[i, 0..dim], h_norm[i, 0..dim]] (length 2*dim).
    for i in 0..n {
        gpu.hip.memcpy_dtod_at(
            &scratch.concat.buf,
            i * 2 * dim_bytes,
            &scratch.e_norm.buf,
            i * dim_bytes,
            dim_bytes,
        )?;
        gpu.hip.memcpy_dtod_at(
            &scratch.concat.buf,
            i * 2 * dim_bytes + dim_bytes,
            &scratch.h_norm.buf,
            i * dim_bytes,
            dim_bytes,
        )?;
    }

    // cur = eh_proj @ concat (per-slot). w.eh_proj has m=n_embd, k=2*n_embd.
    let concat_view = scratch.concat.sub_offset(0, n * 2 * dim);
    let cur_view = scratch.cur.sub_offset(0, n * dim);
    weight_gemm_batched(
        gpu,
        &w.eh_proj,
        &concat_view,
        &cur_view,
        n,
        rotated_x_scratch,
    )?;

    // Save residual (inpSA) for the post-attn add.
    gpu.hip
        .memcpy_dtod_at(&scratch.residual.buf, 0, &scratch.cur.buf, 0, n * dim_bytes)?;

    // ── 4. Pre-attn norm + Q/K/V projections ─────────────────────────────
    let tmp_view = scratch.tmp.sub_offset(0, n * dim);
    gpu.rmsnorm_batched(&cur_view, &w.attn_norm, &tmp_view, n, dim, cfg.rms_norm_eps)?;

    // wq emits 2 * q_dim per slot.
    let q_full_view = scratch.q_full.sub_offset(0, n * 2 * q_dim);
    weight_gemm_batched(gpu, &w.wq, &tmp_view, &q_full_view, n, rotated_x_scratch)?;

    // Deinterleave per-head into Q + Gate. Output: [n × n_head × head_dim] each.
    let q_view = scratch.q.sub_offset(0, n * q_dim);
    let gate_view = scratch.gate.sub_offset(0, n * q_dim);
    gpu.deinterleave_f32_batched(
        &q_full_view,
        &q_view,
        &gate_view,
        cfg.n_head,
        cfg.head_dim,
        n,
    )?;

    // attn_q_norm: per-head normalization of Q.
    // rmsnorm_batched treats `batch` as "rows of length n", and gates the
    // single norm-weight broadcast across all rows. For per-head norm here,
    // each Q row is `n_head` heads of length `head_dim`, so we want
    // `batch = n * n_head`, `n = head_dim`. Same convention as the trunk's
    // per-head q_norm at qwen35.rs.
    gpu.rmsnorm_batched(
        &q_view,
        &w.attn_q_norm,
        &q_view,
        n * cfg.n_head,
        cfg.head_dim,
        cfg.rms_norm_eps,
    )?;

    // K, V projections.
    let k_view = scratch.k.sub_offset(0, n * kv_dim);
    let v_view = scratch.v.sub_offset(0, n * kv_dim);
    weight_gemm_batched(gpu, &w.wk, &tmp_view, &k_view, n, rotated_x_scratch)?;
    weight_gemm_batched(gpu, &w.wv, &tmp_view, &v_view, n, rotated_x_scratch)?;
    gpu.rmsnorm_batched(
        &k_view,
        &w.attn_k_norm,
        &k_view,
        n * cfg.n_head_kv,
        cfg.head_dim,
        cfg.rms_norm_eps,
    )?;

    // ── 5. Batched RoPE on Q + K ─────────────────────────────────────────
    gpu.rope_partial_interleaved_f32_batched(
        &q_view,
        &k_view,
        &scratch.positions,
        cfg.n_head,
        cfg.n_head_kv,
        cfg.head_dim,
        cfg.n_rot,
        cfg.rope_theta,
        n,
        // pos_offset=0: MTP head has its own non-compacted KV. Unchanged behavior.
        0,
    )?;

    // ── 6. v1 simplification: per-slot K/V writes + attention = V
    // ─────────────────────────────────────────────────────────────────────
    //
    // Each slot's attention reads its own KV slot only, so attn_out = V
    // (single-token softmax = 1). We still write K/V to make the KV cache
    // available for future cycles that wire historical reads.
    //
    // Per-slot K/V writes use the F32 KV cache write helper. Each write is
    // `kv_dim` floats; positions[i] is the slot. We construct a per-slot
    // device pos_buf by sub-offsetting `scratch.positions`, but
    // kv_cache_write expects a separate i32 device buffer per call. To avoid
    // n separate allocs, we pass sub-offset views of `scratch.positions`
    // (each one i32-sized), since the kernel reads `*pos` as the slot index.
    for i in 0..n {
        let k_row = scratch.k.sub_offset(i * kv_dim, kv_dim);
        let v_row = scratch.v.sub_offset(i * kv_dim, kv_dim);
        let pos_slot = scratch.positions.sub_offset(i, 1);
        gpu.kv_cache_write_q8_0(
            &kv.inner.k_gpu[0],
            &k_row,
            &pos_slot.buf,
            cfg.n_head_kv,
            cfg.head_dim,
        )?;
        gpu.kv_cache_write_q8_0(
            &kv.inner.v_gpu[0],
            &v_row,
            &pos_slot.buf,
            cfg.n_head_kv,
            cfg.head_dim,
        )?;
    }

    // attn_out = V (self-only attention with 1 key collapses softmax to 1).
    // V layout = [n × n_head_kv × head_dim]; attn_out layout = [n × n_head × head_dim].
    // For GQA (n_head > n_head_kv), repeat each KV head `n_head/n_head_kv` times.
    let attn_out_view = scratch.attn_out.sub_offset(0, n * q_dim);
    if cfg.n_head == cfg.n_head_kv {
        // No GQA expansion: attn_out := V row-by-row.
        gpu.hip
            .memcpy_dtod_at(&scratch.attn_out.buf, 0, &scratch.v.buf, 0, n * kv_dim * 4)?;
    } else {
        let ratio = cfg.n_head / cfg.n_head_kv;
        assert!(
            cfg.n_head % cfg.n_head_kv == 0,
            "n_head ({}) must be divisible by n_head_kv ({})",
            cfg.n_head,
            cfg.n_head_kv,
        );
        // Per-slot per-head expand: attn_out[i, h, d] = V[i, h/ratio, d].
        for i in 0..n {
            for h in 0..cfg.n_head {
                let kv_h = h / ratio;
                let src_off = (i * kv_dim + kv_h * cfg.head_dim) * 4;
                let dst_off = (i * q_dim + h * cfg.head_dim) * 4;
                gpu.hip.memcpy_dtod_at(
                    &scratch.attn_out.buf,
                    dst_off,
                    &scratch.v.buf,
                    src_off,
                    cfg.head_dim * 4,
                )?;
            }
        }
    }

    // ── 7. Apply gate (sigmoid * attn_out, in-place) ─────────────────────
    // sigmoid_mul_f32 is shape-agnostic (operates on flat buffer); pass the
    // full `n * q_dim` views.
    gpu.sigmoid_mul_f32(&attn_out_view, &gate_view)?;

    // ── 8. Output projection + residual ──────────────────────────────────
    let o_view = scratch.o.sub_offset(0, n * dim);
    weight_gemm_batched(gpu, &w.wo, &attn_out_view, &o_view, n, rotated_x_scratch)?;
    let res_view = scratch.residual.sub_offset(0, n * dim);
    gpu.add_inplace_f32(&o_view, &res_view)?;

    // ── 9. POST-attn norm + SwiGLU FFN + residual ────────────────────────
    gpu.rmsnorm_batched(
        &o_view,
        &w.attn_post_norm,
        &tmp_view,
        n,
        dim,
        cfg.rms_norm_eps,
    )?;

    let gate_ffn_view = scratch.gate_ffn.sub_offset(0, n * cfg.n_ff);
    let up_view = scratch.up.sub_offset(0, n * cfg.n_ff);
    let ffn_hidden_view = scratch.ffn_hidden.sub_offset(0, n * cfg.n_ff);
    let ffn_out_view = scratch.ffn_out.sub_offset(0, n * dim);
    match &w.ffn {
        Qwen35MtpFfnWeights::Dense(ffn) => {
            weight_gemm_batched(
                gpu,
                &ffn.gate,
                &tmp_view,
                &gate_ffn_view,
                n,
                rotated_x_scratch,
            )?;
            weight_gemm_batched(gpu, &ffn.up, &tmp_view, &up_view, n, rotated_x_scratch)?;
            gpu.silu_mul_f32(&gate_ffn_view, &up_view, &ffn_hidden_view)?;
            weight_gemm_batched(
                gpu,
                &ffn.down,
                &ffn_hidden_view,
                &ffn_out_view,
                n,
                rotated_x_scratch,
            )?;
            gpu.add_inplace_f32(&ffn_out_view, &o_view)?;
        }
        Qwen35MtpFfnWeights::Moe(_) => {
            panic!("batched MTP head forward does not support MoE FFN yet; use serial MTP");
        }
    }

    // ── 10. Snapshot post-FFN hidden into t_mtp_outs ─────────────────────
    gpu.hip.memcpy_dtod_at(
        &scratch.t_mtp_outs.buf,
        0,
        &scratch.ffn_out.buf,
        0,
        n * dim_bytes,
    )?;

    Ok(())
}
