// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt, Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.
//! hipfire-arch-gemma4: Gemma 4 architecture (text-only + vision tower).
//!
//! Implements the [`hipfire_runtime::arch::Architecture`] trait for the
//! Gemma 4 family (`gemma-4-12B`, `gemma-4-31B`, `gemma-4-26B-A4B`,
//! `gemma-4-E4B`, `gemma-4-E2B`).
//!
//! Architectural distinctives vs. Qwen3.5:
//!   - Hybrid attention: 5 sliding-window layers (head_dim=256)
//!     per 1 full-attention layer (head_dim=512, K=V shared via
//!     `attention_k_eq_v`).
//!   - Proportional partial RoPE on full layers (rotates 64 of 256
//!     pairs, theta=1e6); standard full-rotation RoPE on sliding layers
//!     (theta=1e4).
//!   - Sandwich RMSNorm (input + post-attn + pre-FFN + post-FFN per
//!     layer) plus a learned per-layer scalar `layer_scalar [1]`.
//!   - Final logit softcap `tanh(x / 30) * 30` before sampling.
//!   - Tied LM head (`lm_head` aliases `embed_tokens`).
//!   - `embed_scale = sqrt(hidden_size)` multiplied at every embed lookup.
//!   - SPM-BPE tokenizer (vocab=262144, BOS-prepend).
//!
//! Status (2026-06-07): ported from `feat/gemma4-128k-ring-buffer` onto
//! `integration/dispatch-unification`. Uses old-style dispatch initially
//! (backward-compatible wrappers on the dispatch branch); incremental
//! migration to `execute_steps` / `AttentionFamily` planned in Phase 2+.

pub mod arch;
pub mod gemma4;
pub mod gemma4_vision;

pub use arch::Gemma4;
