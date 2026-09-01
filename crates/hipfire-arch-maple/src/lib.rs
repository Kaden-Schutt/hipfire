// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-arch-maple: Maple-Preview 20B-A1B (`model_type: maple`, arch_id 15).
//!
//! A **natively ternary** MoE. Every linear weight in the published checkpoint
//! is already `{-s_r, 0, +s_r}` with one bf16 scale per output row, so the
//! weights are carried losslessly by `MQ2G256LloydU` (qt=51) — the unrotated
//! sibling of MQ2-Lloyd. Nothing here quantizes; the format is a packing.
//!
//! Shape, from the released `config.json` (NOT inferred):
//!   - 24 layers, hidden 2048, vocab 151936, untied lm_head
//!   - GQA 16 query / 4 KV heads, head_dim 128, QK-norm on both
//!   - `layer_types`: 18 `sliding_attention` (window 512, RoPE) + 6
//!     `full_attention` (global, **NoPE**), in a strict [L,L,L,G] 3:1 pattern
//!   - MoE on EVERY layer: 256 experts, top-8, `moe_intermediate_size` 512,
//!     no shared experts, no expert bias, softmax → top-k → renormalise
//!   - `partial_rotary_factor` 0.5 ⇒ only the first 64 of 128 head dims rotate
//!
//! Three things that are easy to get silently wrong:
//!   1. **QK-norm runs BEFORE RoPE** (`modeling_maple.py:300-303`). Swapping
//!      the order still generates text; it just generates worse text.
//!   2. **Clamped SwiGLU, asymmetrically:**
//!      `silu(clamp(gate, max=7)) * clamp(up, -7, 7)` — the gate clamps only
//!      from above, `up` clamps both ways.
//!   3. **`intermediate_size` (4096) is a dead key.** Experts are 512 wide.
//!
//! No new GPU kernels: the whole MoE decode pipeline already exists in
//! `quant/quality` (`moe_router_softmax_topk_k8_wave64_exact`,
//! `moe_topk_renorm_k8`, `moe_unscatter_silu_clamp_k8` — the clamped SwiGLU —
//! plus the scatter/permute/combine suite), and MQ2G256LloydU shares
//! MQ2-Lloyd's 72 B/group layout byte-for-byte so every existing kernel binds.

pub mod batch;
pub mod bundle;
pub mod carrier;
pub mod config;
pub mod forward;
pub mod maple;

pub use bundle::{load_maple_from_hfq, load_maple_from_hfq_with_head, MapleBundle};
pub use carrier::load_maple_bundle;
pub use forward::decode_step;

pub use config::{MapleConfig, MapleLayerType, MAPLE_SWIGLU_CLAMP};
pub use maple::{
    expert_tensor_name, router_tensor_name, ExpertProj, MapleExpert, MapleLayerWeights,
    MapleMoeFfn, MapleState, MapleWeights, EMBED_TENSOR_NAME, FINAL_NORM_TENSOR_NAME,
    LM_HEAD_TENSOR_NAME,
};
