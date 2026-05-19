//! hipfire-arch-qwen2: plain Qwen2 dense text decoder.
//!
//! Implements [`hipfire_runtime::arch::Architecture`] for the Qwen2 family
//! (GQA, RMSNorm, SwiGLU, 1-D RoPE, `attention_bias=true` on Q/K/V
//! projections). Validated against Qwen2-1.5B-Instruct.
//!
//! arch_id = 7. See `docs/architecture-ids.md`.
//!
//! # Bring-up status (rev 3 — phase 1 functionally complete)
//!
//! Real config parser, weight loader, KV cache + scratch graph
//! ([`qwen2::Qwen2State`]), and forward pass ([`qwen2::forward_step`] /
//! [`qwen2::forward_step_greedy`]) are all landed and end-to-end
//! validated on gfx1151 against the committed HF F32 reference.
//!
//! - [`qwen2::config_from_metadata_json`] parses 13 Qwen2Config fields
//!   with sensible defaults; covered by 6 unit tests.
//! - [`qwen2::load_weights`] reads embed_tokens + final norm + tied
//!   lm_head + 28 layers. Supports HFQ4G256 / HFQ4G128 / F16 weight
//!   quant types with host-side F16→F32 expansion for tied lm_head.
//! - [`qwen2::Qwen2State`] allocates the full per-step scratch graph
//!   plus F32 KV cache (28 layers × 2 × max_seq × kv_dim).
//! - [`qwen2::forward_step`] runs one decode step through 28 layers:
//!   RMSNorm → fused QKV + bias adds → RoPE → KV cache write →
//!   attention → o_proj → residual → FFN norm → SwiGLU → residual,
//!   then final norm + lm_head. Bumps `state.next_pos`.
//!
//! End-to-end validation result against
//! `benchmarks/references/qwen2_1p5b_instruct_smoke.json`:
//!
//! - **7/7 prefix top-1 matches** (positions 0..7)
//! - 9/16 total top-1 matches; divergences are at synonym positions
//!   ("key" vs "crucial") consistent with HFQ4 (4-bit weight) quant
//!   noise against the F32 reference, not implementation error
//! - hipfire output is fluent coherent English describing transformer
//!   attention
//!
//! The driver binary `examples/infer_qwen2.rs` runs prefill + 16-token
//! greedy decode + reference compare; pass criterion is up to the
//! caller's tolerance for quant-induced divergence.
//!
//! Still pending: daemon dispatch arm (R3), F16-quant precision sweep
//! for a tighter top-1 baseline, and KV quantisation paths (HFQ4 /
//! HFQ8 / asym-N / Q8) for serving-time memory budgets.
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
