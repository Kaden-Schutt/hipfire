// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt, Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 dense text plus strictly admitted E2B/E4B text variants for hipfire.
//!
//! A DENSE 12B port of the Gemma 4 architecture. Distinguishing features vs a
//! plain pre-norm transformer:
//!   - **Hybrid attention**: 5 sliding-window layers (head_dim 256, window
//!     1024, RoPE θ=10000 full rotate-half) per 1 full layer (head_dim 512,
//!     partial proportional RoPE θ=1e6 over the first 25% of head_dim). The
//!     period is READ from the `layer_types` array, not assumed.
//!   - **K=V sharing** on full layers (`attention_k_eq_v`): no v_proj — V is
//!     the pre-k_norm output of k_proj, renormed by a weight-less RMSNorm.
//!   - **Sandwich RMSNorm**: input + post-attention + pre-FFN + post-FFN per
//!     layer (plain `x * w`, no +1), plus a learned per-layer `layer_scalar`.
//!   - **Attention scale 1.0** (not 1/√d): Q is pre-scaled by √head_dim so the
//!     kernel's internal 1/√head_dim cancels.
//!   - **gelu_pytorch_tanh** SwiGLU MLP; **embed scaled by √hidden_size**;
//!     **tied LM head**; **final logit softcap** `tanh(x/30)*30`.
//!
//! DROPPED from the broader Gemma 4 family (the dense 12B uses none of these):
//! MoE blocks, the E-series per-layer-embedding / KV-sharing-layer /
//! double-wide-MLP machinery, and vision/audio.
//!
//! arch_id = 13 (see docs/architecture-ids.md). Reuses existing kernels:
//! `rope_f32` / `rope_partial_halved_f32`, `attention_q8_0_kv_swa`,
//! `gelu_tanh_f32`, `logit_softcap_f32`, plus the shared GEMV path.

pub mod arch;
pub mod config;
pub mod drafter;
pub mod forward;
pub mod gemma4;
pub mod lowered;
pub mod speculative;
pub mod carrier;
pub use carrier::{
    load_gemma4_bundle, load_gemma4_bundle_with_route, lowered_kv_policy, preflight_gemma4,
    preflight_lowered_kv_budget, required_kv_bytes, Gemma4Bundle, Gemma4EagerBundle,
    Gemma4LoweredBundle, Gemma4LoweredKvPolicy, Gemma4Route,
};

pub use arch::{Gemma4, ARCH_ID};
pub use config::{Gemma4Config, LayerType, RopeType};
pub use drafter::{
    drafter_step, drafter_step_from_concat, DrafterLayerWeights, DrafterStepOut,
    Gemma4DrafterConfig, Gemma4DrafterScratch, Gemma4DrafterWeights, DRAFTER_ARCH_ID,
};
pub use forward::{forward_batch, forward_batch_spec};
pub use gemma4::{FullLayerWeights, Gemma4State, Gemma4Weights, LayerWeights, SlidingLayerWeights};
pub use speculative::{spec_step_gemma4_eagle, Gemma4SpecScratch, SpecStepOut};
