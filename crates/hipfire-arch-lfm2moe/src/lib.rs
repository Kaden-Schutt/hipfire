// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! LFM2.5-8B-A1B architecture (arch_id 11): a HYBRID model interleaving
//! double-gated LIV short-convolution mixers (18 layers) with GQA+QK-norm
//! attention mixers (6 layers), feeding a DeepSeek-style sigmoid-bias top-4
//! MoE FFN (layers >= num_dense_layers) or a dense SwiGLU MLP (the first
//! `num_dense_layers` layers). Per-layer mixer choice comes from the
//! checkpoint's `layer_types`. Forward pass is free functions in `forward.rs`,
//! mirroring the MiniMax-M2 (arch_id 10) port.
//!
//! Kernel coverage (all pre-existing except one small conv variant):
//!   * attention   -> per-head QK-norm (rmsnorm_batched) + full-dim rotate_half
//!                    RoPE (rope_f32) + Q8 GQA flash attention
//!   * MoE routing -> deepseek4_moe_topk_bias_aware (sigmoid + expert_bias,
//!                    runtime k_top = num_experts_per_tok = 4)
//!   * MoE experts -> gemv_hfq4g256_moe_{gate_up,down} indexed (FWHT MQ4, k=4)
//!   * dense MLP   -> Q8 SwiGLU (w1 gate, w3 up, silu_mul, w2 down)
//!   * LIV conv    -> conv1d_gated_decode_f32 (NEW: K=3, fused B*x / C*conv_out
//!                    gates + rolling conv-state cache)
pub mod config;
pub mod forward;
pub mod lfm2moe;

pub use config::{Lfm2MoeConfig, MixerKind};
pub use forward::{decode_step, decode_step_capture};
pub use lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};

/// Architecture id for LFM2.5-MoE.
pub const ARCH_ID: u32 = 11;
