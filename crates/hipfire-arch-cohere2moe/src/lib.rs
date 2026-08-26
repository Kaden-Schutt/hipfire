// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-arch-cohere2moe: Cohere2-MoE (CohereLabs/North-Mini-Code-1.0) for
//! hipfire.
//!
//! Architecture (`architectures: ["Cohere2MoeForCausalLM"]`, `model_type:
//! cohere2_moe`): a **parallel-block** transformer — every layer is
//!   `h += attn(input_layernorm(h)); h += ffn(input_layernorm(h))`
//! using a SINGLE mean-centered `Cohere2LayerNorm` for both branches, where:
//!   - attention = GQA (no bias, no QK-norm), interleaved per `layer_types`:
//!     `sliding_attention` (window 4096, **RoPE**) vs `full_attention`
//!     (global, **NoPE** — no positional embedding). 32 q / 4 kv heads,
//!     head_dim 128.
//!   - ffn = dense SwiGLU for the first `first_k_dense_replace` (=1) layers,
//!     else 128-expert MoE: `sigmoid(router)` top-8, `norm_topk_prob=false`
//!     (raw sigmoid combine weights), no routing bias, no shared expert.
//!   - tied embeddings; `logit_scale=1.0`.
//!
//! arch_id = 12 (see docs/architecture-ids.md). Reuses hipfire's existing
//! kernels with NO new GPU kernels — the mean-centered `layernorm_batched`
//! already shipped, and `moe_topk_renorm_k8(norm_topk=false)` expresses the
//! sigmoid/no-renorm router.

pub mod arch;
pub mod arch_model;
pub mod carrier;
pub mod cohere2moe;
pub mod config;
pub mod forward;
pub mod paro_dir;
pub mod spec_emit;
pub mod spec_impl;
pub use arch::Cohere2Moe;
pub use carrier::load_cohere2moe_bundle;
pub use cohere2moe::{
    Cohere2MoeLayerWeights, Cohere2MoeState, Cohere2MoeWeights, ExpertWeights, MoeFfn,
};
pub use config::{AttnKind, Cohere2MoeConfig};
pub use forward::decode_step;
pub use spec_impl::Cohere2MoeBundle;
