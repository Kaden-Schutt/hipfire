// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-runtime: GGUF model loading and LLaMA inference on RDNA GPUs.
//!
//! This crate is arch-agnostic. Architecture implementations live in
//! sibling crates (`hipfire-arch-qwen35`, `hipfire-arch-qwen35-vl`,
//! future `hipfire-arch-llama`, etc.) and depend on this crate for
//! shared infrastructure: HFQ/GGUF file readers, the LLaMA-style
//! scratch / KV / sampler primitives, tokenizer, prompt framing, eos
//! filter, loop guard, eviction (TriAttn, CASK), spec-decode primitives
//! (DFlash, DDTree), demand paging (cpu_router, weight_pager), and the
//! [`arch::Architecture`] trait.

pub mod arch;
pub mod bf16_loader;
pub mod calibration;
#[cfg(feature = "deltanet")]
pub mod cask;
pub mod config;
#[cfg(feature = "deltanet")]
pub mod cpu_router;
#[cfg(feature = "deltanet")]
pub mod ddtree;
#[cfg(feature = "deltanet")]
pub mod dflash;
pub mod env_docs;
pub mod eos_filter;
pub mod ep;
pub mod gguf;
pub mod hfq;
pub mod hfq_modules;
pub mod host_profile;
pub mod llama;
// Neutral homes for the GENERIC primitives historically defined under `llama`
// (they are not llama-specific — every arch uses them). Callers should import
// from these modules, not `llama`. The definitions still physically live in
// `llama` for now; relocating them is a no-API-change follow-up. See the
// de-llama-ify cleanup.
pub mod kv {
    //! Generic KV cache (not llama-specific).
    pub use crate::llama::KvCache;
}
pub mod weights {
    //! Generic weight/embedding types + the GEMV/GEMM/rotate operations on them
    //! (not llama-specific).
    pub use crate::llama::{
        fused_rmsnorm_rotate_for_mq, fused_rmsnorm_rotate_for_paro,
        fused_rmsnorm_rotate_mq_batched_for, fused_silu_mul_rotate_mq_batched_for,
        fused_silu_mul_rotate_mq_for, rotate_x_mq_128_for, rotate_x_mq_batched_for,
        rotate_x_mq_for, rotate_x_paro_for, weight_gemm, weight_gemv, weight_gemv_prerotated,
        weight_gemv_residual, weight_gemv_swiglu_residual, EmbeddingFormat, LayerWeights,
        ParoRotation, WeightTensor,
    };
}
pub mod quant {
    //! Generic dequant codecs + half/bf16 conversions (not llama-specific).
    pub use crate::llama::{
        convert_q4k_to_q4f16_g32, convert_q4k_to_q4f16_g64, dequantize_q4_0, dequantize_q4_k,
        dequantize_q6_k, dequantize_q8_0, f16_to_f32, f32_to_f16,
    };
}
pub mod dispatch {
    //! Generic kernel-dispatch family accessors + dispatch types (not
    //! llama-specific; the family accessors and `hipfire_dispatch` re-exports
    //! historically lived under `llama`).
    pub use crate::llama::{
        attention_family, fused_qkv_family, gemm_family, gemv_family, is_batchable_la, moe_family,
        AttnParams, DispatchCtx, FullAttnParams, FusedQkvParams, GemvVariant, KernelKey,
        KvTierInputs, KvTierPlan, RotInput, RotateInputs, RotatedActivation, ShapeInfo,
    };
}
pub mod logging;
pub mod loop_guard;
pub mod model_source;
pub mod mtp_mirror;
pub mod multi_gpu;
pub mod safetensors_source;
pub mod sampler;
pub mod speed_bench;
pub mod tokenizer;
pub mod tool_call;
pub mod tp_shard;
#[cfg(feature = "deltanet")]
pub mod triattn;
#[cfg(feature = "deltanet")]
pub mod weight_pager;
