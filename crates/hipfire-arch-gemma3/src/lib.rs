// SPDX-License-Identifier: Apache-2.0
// hipfire — Gemma3 text decoder arch crate. See LICENSE / NOTICE.

//! Gemma3 (text) architecture — `arch_id = 12`.
//!
//! A dense, full-attention-only decoder, but with several deltas from the
//! llama/qwen path that the forward and ingest must honor (see
//! `docs/plans/2026-06-19-gemma3-bringup.md`):
//!
//! 1. **(1+w) zero-centered RMSNorm** — the quantizer bakes `+1` into every
//!    norm weight at ingest (`gemma_norm_offset` metadata marker), so the
//!    standard rmsnorm kernel is correct here with no runtime offset.
//! 2. **Embedding scaled by √hidden_size** (see [`Gemma3Config::embed_scale`]).
//! 3. **4 norms/layer** — input + post-attn (pre-residual) and pre-FFN +
//!    post-FFN (pre-residual); the post-norms sit between the projection and
//!    the residual add, so the fused gemv+residual step can't be reused.
//! 4. **Per-head QK-norm** (RMSNorm over `head_dim` on q and k).
//! 5. **head_dim independent of dim/n_heads** (128 @27b, 256 @4b).
//! 6. **Custom attention scale** `query_pre_attn_scalar^-0.5`
//!    (see [`Gemma3Config::attn_scale`]) — NOT `1/√head_dim` (168≠128 @27b).
//! 7. **Dual-theta sliding-window interleave** — 5 local (θ=`rope_local_base_freq`,
//!    SWA `sliding_window`) : 1 global (θ=`rope_theta`, full causal); see
//!    [`Gemma3Config::is_global_layer`].
//! 8. **GeGLU `gelu_pytorch_tanh`** (not SwiGLU/silu).
//! 9. **No logit/attn soft-capping** in Gemma3.
#![allow(clippy::too_many_arguments)]

pub mod arch;
pub mod calibration;
pub mod config;
pub mod forward;
pub mod weights;

pub use arch::{Gemma3, Gemma3Backend};
pub use config::{config_from_hfq, config_from_metadata_json, Gemma3Config};
pub use forward::{
    embed_token, forward_prefill_batch, forward_step, forward_step_greedy, forward_step_with_embed,
    Gemma3State,
};
pub use weights::{load_weights, load_weights_prefixed, Gemma3LayerWeights, Gemma3Weights};
