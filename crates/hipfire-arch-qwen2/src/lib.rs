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
//! End-to-end validation results against
//! `benchmarks/references/qwen2_1p5b_instruct_smoke.json`:
//!
//! - **Q8F16 (qt=3) weights: 16/16 top-1 matches** — definitive
//!   correctness lock-in. Forward + greedy decode of 16 tokens
//!   completes in ~300 ms on gfx1151 (140 ms prefill).
//! - HFQ4G256 (4-bit) weights: 9/16 top-1 matches with a perfect
//!   7/7 prefix; divergences at synonym positions ("key" vs "crucial")
//!   are the expected signature of 4-bit weight quant against the
//!   F32 reference, NOT implementation error (confirmed by the Q8
//!   sweep above).
//!
//! The driver binary `examples/infer_qwen2.rs` runs prefill + N-token
//! greedy decode + reference compare; the Q8 path is the recommended
//! correctness baseline.
//!
//! R3 resolved: daemon arm wired (arch_id=7 → `generate_qwen2` in
//! daemon.rs). `hipfire run /path/to/qwen2.hfq "prompt"` works
//! end-to-end at ~96 tok/s on the Q8 model. Bring-up scope —
//! DFlash / CASK / PFlash / VL / ChatML / repeat penalty / top-p /
//! `<think>` / multi-GPU are explicitly refused or skipped on this
//! path.
//!
//! Still pending: KV quantisation paths (HFQ4 / HFQ8 / asym-N / Q8)
//! for serving-time memory budgets, and prefill batching for
//! serving-time latency.
//!
//! See `docs/plans/dots-ocr-prd.md` phase 1 for the bring-up plan
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
