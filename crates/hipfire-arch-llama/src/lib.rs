// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-arch-llama: LLaMA / Mistral / plain-Qwen3 architecture.
//!
//! This crate implements the [`hipfire_runtime::arch::Architecture`] trait
//! for the dense LLaMA-family (`arch_id = 0` for LLaMA / Mistral, `arch_id = 1`
//! for plain Qwen3 / Qwen2). It provides the bring-up triple
//! (`config_from_hfq` / `load_weights` / `new_state`) and re-exports the
//! LLaMA-arch forward path under [`hipfire_arch_llama::llama`] so
//! daemon and example call sites can dispatch via either the trait (for
//! bring-up) or direct static calls (for the hot loop).
//!
//! # Why the model module is a re-export
//!
//! PR 8 moved `qwen35.rs` physically into `hipfire-arch-qwen35` because
//! the qwen35 hybrid path was self-contained. The LLaMA-family forward
//! pass in `crates/hipfire-runtime/src/llama.rs` is a different shape:
//! it hosts the **shared transformer infrastructure** that the qwen35
//! arch (and its `pflash.rs` "Plain LLaMA-family drafter" branch) reach
//! into directly — `KvCache`, `WeightTensor`, `EmbeddingFormat`, the
//! GEMV dispatch helpers (`weight_gemv`, `weight_gemv_prerotated`,
//! `weight_gemv_residual`, `weight_gemv_swiglu_residual`,
//! `fused_rmsnorm_rotate_for_mq`), the dequantisers (`dequantize_q4_*`,
//! `dequantize_q8_0`, `dequantize_q6_k`, `convert_q4k_to_q4f16_g{32,64}`),
//! the f16/f32 conversions, the sampler primitives (re-exported from
//! `hipfire_runtime::sampler`), and the Llama-shaped types
//! (`LlamaConfig`, `LlamaWeights`, `LayerWeights`, `ForwardScratch`,
//! `PrefillBatchScratch`, `KvCache`, `SamplingConfig`).
//!
//! Physically moving the file would force `hipfire-arch-qwen35` to take
//! a build-time dependency on this crate. PR 11's task spec forbids
//! touching the qwen35 crate, so the LLaMA module body stays in the
//! runtime crate as the canonical home for those shared primitives.
//! `hipfire_arch_llama::llama` is a re-export of
//! `hipfire_runtime::llama` so future call sites — and any new arch
//! crates that want to use the LLaMA-family forward — can import via
//! the arch-named path consistently with PR 8's pattern.
//!
//! Once cross-arch shared utilities are extracted into a dedicated
//! `hipfire_runtime::transformer` (or similar) sub-module in a
//! follow-up PR, the truly LLaMA-arch-only forward functions
//! (`forward`, `forward_scratch*`, `forward_prefill_batch*`,
//! `prefill_forward`, `forward_early_exit`, `forward_sample`,
//! `forward_logits_gpu`, `is_batchable_la`, `upload_prefill_batch_inputs`,
//! `LlamaConfig::from_gguf`, the GGUF-only `load_weights`) can be
//! physically moved here without breaking arch-qwen35's pflash branch.

pub mod arch;

/// Re-export the LLaMA-family model module so callers can write
/// `hipfire_arch_llama::llama::forward_scratch(...)` etc., matching the
/// PR 8 idiom of `hipfire_arch_qwen35::qwen35::forward_prefill_batch(...)`.
///
/// All symbols here are hosted in `crates/hipfire-runtime/src/llama.rs`.
/// See the module-level doc above for why the body lives in runtime.
pub use hipfire_runtime::llama;

pub use arch::Llama;
