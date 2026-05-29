// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MiniMax-M2 forward pass (free functions — hot-path static dispatch,
//! NOT trait methods; called by the daemon's `arch_id == 10` branch).
//!
//! SCAFFOLD: signatures + the planned per-layer pipeline are sketched
//! here as the bring-up contract; bodies are filled component-by-
//! component and validated against the PyTorch oracle (cosine≈1 per
//! layer) before the real-weight path is enabled.
//!
//! Planned decode/prefill pipeline per layer (verified vs modeling_minimax_m2.py):
//!   1. h_norm = rmsnorm(h, input_layernorm)                  [rmsnorm_f32 / _batched]
//!   2. q = q_proj·h_norm ; k = k_proj·h_norm ; v = v_proj·h_norm   [gemv_auto*]
//!   3. if use_qk_norm: q = rmsnorm(q_flat, q_norm); k = rmsnorm(k_flat, k_norm)
//!        (per-LAYER, over the flat [n_heads*head_dim]/[n_kv*head_dim] vector)
//!   4. reshape to heads; partial RoPE first `rotary_dim` dims     [rope_partial_interleaved_f32]
//!   5. write KV cache; GQA attention                            [attention_flash_gqa / attention_q8_0_kv_batched]
//!   6. attn_out = o_proj·attn ; h += attn_out
//!   7. h_norm2 = rmsnorm(h, post_attention_layernorm)
//!   8. router_logits = gate·h_norm2 ; scores = sigmoid(router_logits)   [sigmoid_f32]
//!   9. top-8 via scores + e_score_correction_bias ; gather sigmoid ; normalize
//!        [deepseek4 moe_topk_bias_aware_f32 / _batched]
//!  10. scatter → grouped SwiGLU experts (gate=w1, up=w3, down=w2) → combine
//!        [moe_scatter_fused_k8, gemm_hfq4g256_moe_grouped_wmma_k2, silu·mul,
//!         moe_down_combine_grouped_k8]   (NO shared expert)
//!  11. h += moe_out
//! Final: rmsnorm(h, norm) → lm_head [gemv_auto] → logits.

// Bring-up bodies land next; the crate compiles as a registered arch
// (arch_id=10) with config parsing + stub weights/state in the meantime.
