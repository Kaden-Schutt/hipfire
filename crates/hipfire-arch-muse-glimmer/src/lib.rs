// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Muse Glimmer (`model_type: muse_glimmer`, arch_id 14) — dense text tower.
//!
//! Shape, from the released `config.json` (NOT inferred):
//!   - 52 layers, hidden 6656, intermediate 19968, vocab 202048
//!   - GQA 32 query / 2 KV heads, head_dim 128 (uniform — unlike Gemma4's 256/512)
//!   - `layer_types`: 39 `sliding_attention` + 13 `full_attention` (3:1, [L,L,L,G])
//!   - `layer_rope_theta`: 500000.0 on the 39 sliding layers, **0 on the 13 full
//!     layers** — theta 0 means NoPE. Key off this array, not the type string.
//!   - `sliding_window` 2048, `max_position_embeddings` 131072
//!   - `hidden_activation` silu (Gemma4 is gelu_pytorch_tanh)
//!   - `attention_bias` false, `tie_word_embeddings` **false** (separate lm_head)
//!
//! Three things that are easy to get silently wrong:
//!   1. **Split norm eps.** `rms_norm_eps` 1e-5 for the pre-norms, `post_norm_eps`
//!      1e-8 for the post-norms. Using one value everywhere is wrong but coherent.
//!   2. **Scale-less QK-norm.** There are no `q_norm`/`k_norm` weight tensors, but
//!      the RMSNorm still runs on Q and K per head, followed by `Q *= qk_scale_factor`
//!      (3.87). Skipping it wrong-scales attention. Do NOT copy Gemma4's
//!      pre-scale-by-sqrt(head_dim) trick, which cancels the kernel's 1/sqrt.
//!   3. **Gated attention.** `self_attn.gate_proj` is NOT the MLP gate. The gate is
//!      applied to the attention output BEFORE `o_proj`:
//!      `attn_out *= sigmoid(gate_proj(post_input_layernorm_hidden))`.
//!
//! Logits: `logits *= output_multiplier` (0.196116135 == 1/sqrt(6656/256)) and then
//! `tanh(x/cap)*cap` with cap 20.0. Gemma4 softcaps at 30 with no multiplier.

pub mod config;
pub mod drafter;
pub mod forward;
pub mod glimmer;
