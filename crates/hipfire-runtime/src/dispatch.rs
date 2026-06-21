// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Generic kernel-dispatch family accessors and dispatch-type re-exports.
//!
//! Process-global accessors for the centralized `hipfire_dispatch` kernel
//! families (gemv/gemm/fused_qkv/moe/attention) plus the dispatch parameter
//! types. These are arch-agnostic — every dense/MoE arch routes its launches
//! through them. They historically lived in `llama.rs`; relocated here as part
//! of the de-llama-ify cleanup. The parameter/plan types are thin re-exports of
//! the `hipfire_dispatch` crate's own definitions.

use rdna_compute::DType;

pub use hipfire_dispatch::context::DispatchCtx;
pub use hipfire_dispatch::families::attention::{AttnParams, FullAttnParams};
pub use hipfire_dispatch::families::fused_qkv::FusedQkvParams;
pub use hipfire_dispatch::families::gemv::{RotInput, RotateInputs, RotatedActivation};
pub use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
pub use hipfire_dispatch::types::{GemvVariant, KernelKey, ShapeInfo};

pub fn gemv_family() -> &'static hipfire_dispatch::families::gemv::GemvFamily {
    use std::sync::OnceLock;
    static GEMV: OnceLock<hipfire_dispatch::families::gemv::GemvFamily> = OnceLock::new();
    GEMV.get_or_init(hipfire_dispatch::families::gemv::GemvFamily::new)
}

/// Process-global [`GemmFamily`], mirroring [`gemv_family`]. #397 Ship 5.2:
/// arches route their batched-prefill plain-GEMM launches through
/// `gemm_family().run_key(..)` so the dispatcher-entry kernel selection lives in
/// the dispatch crate. `run_key` (explicit KernelKey) preserves the direct
/// `gpu.gemm_*` call's own internal arch dispatch byte-for-byte.
pub fn gemm_family() -> &'static hipfire_dispatch::families::gemm::GemmFamily {
    use std::sync::OnceLock;
    static GEMM: OnceLock<hipfire_dispatch::families::gemm::GemmFamily> = OnceLock::new();
    GEMM.get_or_init(hipfire_dispatch::families::gemm::GemmFamily::new)
}

/// Process-global [`FusedQkvFamily`], mirroring [`gemv_family`]. Used by the
/// dense-arch forward paths to route fused QKV / gate-up launches through the
/// centralized dispatch tables (arch gating + 1:1 KernelKey→kernel launch).
pub fn fused_qkv_family() -> &'static hipfire_dispatch::families::fused_qkv::FusedQkvFamily {
    use std::sync::OnceLock;
    static FUSED_QKV: OnceLock<hipfire_dispatch::families::fused_qkv::FusedQkvFamily> =
        OnceLock::new();
    FUSED_QKV.get_or_init(hipfire_dispatch::families::fused_qkv::FusedQkvFamily::new)
}

/// Process-global [`MoeFamily`], mirroring [`gemv_family`]. The centralized MoE
/// decode entry (Ship 4): arches route their per-layer MoE decode through
/// `moe_family().run(..)` so expert dispatch lives in the dispatch crate rather
/// than per-model kernel calls.
pub fn moe_family() -> &'static hipfire_dispatch::families::moe::MoeFamily {
    use std::sync::OnceLock;
    static MOE: OnceLock<hipfire_dispatch::families::moe::MoeFamily> = OnceLock::new();
    MOE.get_or_init(hipfire_dispatch::families::moe::MoeFamily::new)
}

/// Process-global [`AttentionFamily`], mirroring [`gemv_family`]. Ship 3:
/// arches route their per-layer attention decode through
/// `attention_family().run_attention(..)` so KV-write + flash-attention dispatch
/// lives in the dispatch crate rather than per-model inline match trees.
pub fn attention_family() -> &'static hipfire_dispatch::families::attention::AttentionFamily {
    use std::sync::OnceLock;
    static ATTENTION: OnceLock<hipfire_dispatch::families::attention::AttentionFamily> =
        OnceLock::new();
    ATTENTION.get_or_init(hipfire_dispatch::families::attention::AttentionFamily::new)
}

pub fn is_batchable_la(dt: DType, arch: &str) -> bool {
    let always_ok = matches!(
        dt,
        DType::MQ4G256 | DType::HFQ4G256 | DType::MQ6G256 | DType::HFQ6G256 | DType::Q8_0
    );
    if always_ok {
        return true;
    }
    // HFP4G32 / MFP4G32 + MQ3G256 require WMMA. Same arch gate as MQ3.
    let wmma_only = matches!(dt, DType::MQ3G256 | DType::HFP4G32 | DType::MFP4G32)
        && matches!(
            arch,
            "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" | "gfx1200" | "gfx1201"
        );
    // gfx10 RDNA1/2 scalar HFQ3 batched-prefill (Phase 1 of
    // docs/plans/gfx10_mq3_prefill.md). Mirrors the
    // `mq3_uniform_with_gfx10_scalar` arm in qwen35.rs::is_batchable_la —
    // both must stay in sync per the matching-pair comment there.
    let mq3_gfx10_scalar = matches!(dt, DType::MQ3G256)
        && matches!(
            arch,
            "gfx1010" | "gfx1011" | "gfx1012" | "gfx1013" | "gfx1030" | "gfx1031" | "gfx1032"
        );
    wmma_only || mq3_gfx10_scalar
}
