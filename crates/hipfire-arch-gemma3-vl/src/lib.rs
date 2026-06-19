// SPDX-License-Identifier: Apache-2.0
// hipfire — Gemma3 multimodal (SigLIP vision + projector). See LICENSE / NOTICE.

//! Gemma3 multimodal (`Gemma3ForConditionalGeneration`) — `arch_id = 13`.
//!
//! Pipeline (see `docs/plans/2026-06-19-medgemma-vision-bringup.md`):
//! image → **SigLIP** vision encoder (`vision_tower.vision_model.*`) → **multimodal
//! projector** (avg-pool 4096→256 → `mm_soft_emb_norm` → `mm_input_projection`) →
//! splice the 256 image embeddings at the `image_token_index` placeholders →
//! the **gemma3 text decoder** (`hipfire-arch-gemma3`, the E1 forward, reused).
//!
//! V1 (this): config only. SigLIP is a standard ViT — LayerNorm (+bias),
//! bidirectional attention, gelu-tanh MLP, **learned** position embeddings (no
//! RoPE) — so its forward reuses the qwen35-vl vision kernels.

pub mod config;
pub mod forward;
pub mod vision;

pub use config::{vl_config_from_hfq, Gemma3VlConfig, SigLipConfig};
pub use forward::vision_forward;
pub use vision::{SigLipLayerWeights, SigLipWeights};
