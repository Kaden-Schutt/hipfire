//! hipfire-arch-qwen2: plain Qwen2 dense text decoder.
//!
//! Implements [`hipfire_runtime::arch::Architecture`] for the Qwen2 family
//! (GQA, RMSNorm, SwiGLU, 1-D RoPE, `attention_bias=true` on Q/K/V
//! projections). Validated against Qwen2-1.5B-Instruct.
//!
//! arch_id = 7. See `docs/architecture-ids.md`.
//!
//! # Bring-up status (rev 0)
//!
//! Skeleton phase. `config_from_hfq` and `load_weights` are stubs from
//! the toy template — they compile but do not load real weights. The
//! forward path will be ported from `hipfire-arch-qwen35::qwen35` with
//! DeltaNet/MoE/q-k-norm-pre-RoPE removed and QKV bias added.
//!
//! See `docs/plans/qwen_2.5_vlm.md` phase 1 for the full plan.
//!
//! # Relation to plain Qwen3
//!
//! Qwen2 and Qwen3 share most of the dense forward shape. The deltas
//! that justify a separate crate (vs. flag-toggles in qwen35):
//!
//! - Qwen3 applies q/k RMSNorm *before* RoPE; Qwen2 does not.
//! - Qwen2 has `attention_bias=true` on Q/K/V projections; qwen35's
//!   `fused_qkv_hfq4g256` does not currently accept a bias buffer.
//! - Qwen2-1.5B-Instruct uses `tie_word_embeddings=true` (no separate
//!   lm_head tensor); dots.ocr uses `tie_word_embeddings=false`. The
//!   loader handles both.

pub mod arch;
pub mod qwen2;

pub use arch::Qwen2;
