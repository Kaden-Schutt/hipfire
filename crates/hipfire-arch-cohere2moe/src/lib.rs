// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Cohere2Moe architecture (arch_id 12) — CohereLabs BLS-Mini-Code-1.0 family.
//!
//! A Command-R-style decoder with MoE: GQA attention (32 q / 4 kv heads,
//! head_dim 128) + a **parallel decoder block** (one `input_layernorm` shared
//! by attention and the MLP, both summed into the residual) feeding either a
//! dense SwiGLU MLP (the first `first_k_dense_replace` layers) or a 128-expert
//! top-8 sigmoid-routed MoE. Forward pass = free functions in `forward.rs`,
//! mirroring the MiniMax-M2 (arch_id 10) / LFM2.5-MoE (arch_id 11) ports.
//!
//! Kernel coverage — **zero new kernels** (the precondition that makes this
//! port tractable per docs/methodology/arch-port-validation.md):
//!   * attention   -> interleaved full-dim RoPE (`rope_partial_interleaved_f32`
//!                    with rotary_dim = head_dim) + Q8 GQA flash attention
//!   * dense MLP   -> Q8 SwiGLU (gate_proj, up_proj, silu_mul, down_proj)
//!   * MoE routing -> sigmoid + (zero-bias) top-8 `deepseek4_moe_topk_bias_aware`
//!   * MoE experts -> gemv_hfq{4,6}g256_moe_{gate_up,down} indexed (FWHT MQ4/6)
//!   * norm        -> standard RMSNorm (`weight * x̂`)
//!
//! What is NOT yet wired (see NEXT-STEPS.md): the hipfire-quantize converter arm
//! (safetensors → HFQ), the daemon `arch_id == 12` dispatch, sliding-window
//! attention (deferred — full attention is correct for prompts < 4096), and the
//! `norm_topk_prob=false` no-renorm routing variant (oracle-loop target T1).

pub mod arch;
pub mod cohere2moe;
pub mod config;
pub mod forward;

pub use arch::Cohere2Moe;
pub use cohere2moe::{CohereState, CohereWeights};
pub use config::{AttnKind, Cohere2MoeConfig};
pub use forward::{decode_step, decode_step_capture};

/// Architecture id for Cohere2Moe.
pub const ARCH_ID: u32 = 12;
