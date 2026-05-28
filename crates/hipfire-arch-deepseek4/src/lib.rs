// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// Suppress stylistic-only clippy lints that aren't structural — DeepSeek V4 kernel
// wrappers routinely exceed the default 7-arg ceiling (kernels.launch_kernel
// signatures), and doc-indent rules cover bullet lists that read fine in
// IDEs but trip Rust 1.80+'s heuristic.
#![allow(
    clippy::too_many_arguments,
    clippy::doc_overindented_list_items,
    clippy::doc_lazy_continuation,
    clippy::needless_range_loop
)]

//! hipfire-arch-deepseek4: DeepSeek V4 Flash architecture.
//!
//! DeepSeek V4 Flash architecture (`arch_id = 9`). DeepSeek V4 is a hybrid
//! attention + MoE arch with several pieces not present in the
//! Qwen3.5 / LLaMA paths:
//!
//! - **Hyper-Connections** (`hc_mult = 4`, `hc_sinkhorn_iters = 20`):
//!   four residual streams mixed via a Sinkhorn-normalised gating
//!   matrix every layer, replacing the single pre-norm residual.
//! - **Compressed-KV indexer** (`index_n_heads = 64`,
//!   `index_head_dim = 128`, `index_topk = 512`): a separate small
//!   attention surface that scores tokens to gate which positions
//!   the main attention attends to. `compress_ratios` per layer
//!   (mostly `4, 128` alternating) controls compression strength.
//! - **Tail-only RoPE** (`qk_rope_head_dim = 64` of `head_dim = 512`):
//!   only the last 64 dims of each head carry rotary positional
//!   encoding; the rest is straight Q · K matmul.
//! - **Q-LoRA + O-LoRA** (`q_lora_rank = 1024`, `o_lora_rank = 1024`):
//!   the query and output projections factor through a rank-1024
//!   bottleneck rather than going full hidden × hidden.
//! - **FP4 expert weights** (`expert_dtype = "fp4"`, block scales
//!   `128 × 128`, scale fmt `ue8m0`, weight fmt `e4m3`): the routed
//!   experts ship as FP4 with per-block UE8M0 scales. Conversion to
//!   our MQ-family quants happens at `hipfire-quantize` time.
//! - **Raw SWA cache** (`sliding_window = 128`): a bounded ring of
//!   the last 128 tokens for attention; longer-range context comes
//!   through the compressed-KV indexer path.
//!
//! See `crates/hipfire-arch-deepseek4/src/arch.rs` for the trait
//! impl shape, `src/deepseek4.rs` for the Config / Weights / State
//! definitions, `src/forward.rs` for the inference path, and
//! `src/spec_decode.rs` for the MTP speculative-decode head.
//!
//! Upstream references:
//! - Reference C99 implementation: <https://github.com/antirez/ds4>
//!   (source of truth for MTP wiring, Hyper-Connections head reduction,
//!   and the raw SWA + compressed-KV KV-cache layout).

pub mod arch;
pub mod deepseek4;
pub mod dsml;
pub mod forward;
pub mod grammar;
pub mod sampling;
pub mod spec_decode;

pub use arch::DeepseekV4;
pub use deepseek4::{
    DeepseekV4Config, DeepseekV4State, DeepseekV4Weights, IndexerLayerState,
    MainAttentionLayerState,
};
