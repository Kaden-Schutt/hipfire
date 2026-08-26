// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-arch-minimax: MiniMax-M2 (Mixtral-style MoE) for hipfire.
//!
//! Architecture (verified against `modeling_minimax_m2.py`): a textbook
//! pre-norm transformer, every one of `num_hidden_layers` blocks being
//!   `h += attn(input_layernorm(h)); h += moe(post_attention_layernorm(h))`
//! where:
//!   - attention = GQA (no bias) + **per-layer QK-norm** (RMSNorm on the
//!     flat q[`n_heads*head_dim`] / k[`n_kv*head_dim`] BEFORE head reshape)
//!     + **partial rotate_half RoPE** (first `rotary_dim` of `head_dim`).
//!   - moe = `sigmoid(router_logits) + e_score_correction_bias` top-k
//!     selection, gather-sigmoid-weights + normalize (DeepSeek-V3 style),
//!     SwiGLU experts, NO shared expert.
//!
//! arch_id = 10 (see docs/architecture-ids.md). Maps onto hipfire's
//! existing kernels with NO new kernels: qwen35 GQA + `rope_partial_*`,
//! deepseek4 `moe_topk_bias_aware` routing, the grouped-MQ4 MoE GEMM +
//! `moe_scatter/unscatter/down_combine_k8` family.

pub mod arch;
pub mod arch_model;
pub mod carrier;
pub mod forward;
pub mod minimax;
/// MiniMax-M2 impl of the arch-generic `hipfire_runtime::spec` seam
/// (`impl SpecTarget for MiniMaxBundle`) — lets the model-free n-gram
/// speculator drive MiniMax-M2 (arch_id=10).
pub mod spec_impl;

pub use arch::MiniMaxM2;
pub use carrier::load_minimax_bundle;
pub use minimax::{
    config_from_safetensors, load_weights_from_safetensors, MiniMaxConfig, MiniMaxExpertWeights,
    MiniMaxLayerWeights, MiniMaxState, MiniMaxWeights,
};
pub use spec_impl::{MiniMaxBundle, MiniMaxSpecScratch};
