//! hipfire-arch-qwen2: plain Qwen2 dense text decoder.
//!
//! Implements [`hipfire_runtime::arch::Architecture`] for the Qwen2 family
//! (GQA, RMSNorm, SwiGLU, 1-D RoPE, `attention_bias=true` on Q/K/V
//! projections). Validated against Qwen2-1.5B-Instruct.
//!
//! arch_id = 7. See `docs/architecture-ids.md`.
//!
//! # Bring-up status (rev 2)
//!
//! Real config parser, weight loader, and HFQ4 quantisation are all
//! landed and load-verified on gfx1151 via `inspect_hfq --load`:
//!
//! - [`qwen2::config_from_metadata_json`] parses 13 Qwen2Config fields
//!   with sensible defaults; covered by 4 unit tests.
//! - [`qwen2::load_weights`] reads embed_tokens + final norm + tied
//!   lm_head + 28 layers (input_layernorm + qkv with bias + o_proj +
//!   post_attention_layernorm + gate/up/down). Supports
//!   HFQ4G256 / HFQ4G128 / F16 weight quant types.
//! - Tied-embedding detection + F16→F32 host expansion for the tied
//!   lm_head (the latter is load-bearing — see the doc on
//!   `load_lm_head` for the corruption mode it avoids).
//!
//! Still pending: forward pass + real `Qwen2State` (KV cache + scratch
//! graph) + HF reference capture + token-id validation.
//!
//! See `docs/plans/qwen_2.0_vlm_plus_dots_ocr.md` phase 1 for the bring-up plan
//! (the new R2–R5 risk entries in §6 capture the rev-2 review findings
//! that drove this revision; the standalone review files were folded
//! into the plan and then dropped).
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
//! - Qwen2 uses standard RMSNorm (`weight * x * rsqrt(...)`); Qwen3.5
//!   uses `(1 + weight) * ...`. The loader does **not** apply the
//!   `+= 1.0` offset (see `load_norm_weight_raw`).

pub mod arch;
pub mod qwen2;

pub use arch::Qwen2;
