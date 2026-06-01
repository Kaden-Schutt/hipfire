// SPDX-License-Identifier: MIT OR Apache-2.0
//! MoE kernel family: dispatching expert GEMM operations.
//!
//! Supports 3 variants:
//! - **IndexedGateUp**: gate+up projection for a single expert (indexed by token)
//! - **IndexedDown**: down projection for a single expert (indexed by token)
//! - **GroupedGemm**: batched grouped-expert GEMM (all experts in one launch)
//!
//! # Current status
//!
//! MoE dispatch is **model-specific** — expert compute is orchestrated through
//! per-model `moe_ffn_decode_with_scratch` paths (e.g. `qwen35`) that manage
//! their own per-expert GEMV loops, scatter/gather, and softmax top-k routing.
//! This family exists as a future entry point for fused grouped-expert kernels.
//! Today, `run()` returns `UnsupportedVariant` with a clear explanation.

use rdna_compute::GpuTensor;

use crate::context::DispatchCtx;
use crate::families::gemv::WeightRef;
use crate::tables::moe_table;
use crate::tables::KernelRegistry;
use crate::traits::KernelFamily;
use crate::types::*;

// ── Dispatch parameters ────────────────────────────────

pub struct MoeParams<'a> {
    pub variant: MoeVariant,
    pub weights: &'a [&'a WeightRef<'a>],
    pub x: &'a GpuTensor,
    pub y: &'a [&'a GpuTensor],
}

// ── Family ─────────────────────────────────────────────

pub struct MoeFamily {
    registry: KernelRegistry,
}

impl MoeFamily {
    pub fn new() -> Self {
        let registry = KernelRegistry::new();
        moe_table::populate(&registry);
        Self { registry }
    }

    pub fn registry(&self) -> &KernelRegistry {
        &self.registry
    }

    /// Resolve the best kernel key for the given MoE variant.
    ///
    /// Applies arch gating through `KernelRegistry::resolve`.
    pub fn resolve(
        &self,
        variant: MoeVariant,
        ctx: &DispatchCtx,
    ) -> Result<KernelKey, DispatchError> {
        let key = match variant {
            MoeVariant::IndexedGateUp => KernelKey::MoeIndexedGateUpLloyd,
            MoeVariant::IndexedDown => KernelKey::MoeIndexedDownLloyd,
            MoeVariant::GroupedGemm => KernelKey::MoeGroupedGemm,
        };
        self.registry.resolve(key, ctx, None)
    }

    /// Run a MoE expert operation.
    ///
    /// Currently returns `UnsupportedVariant` because expert dispatch is
    /// model-specific — it lives in per-model `moe_ffn_decode_with_scratch`
    /// paths that handle per-expert GEMV loops, routing, and scratch layout.
    /// This is a placeholder for when fused grouped-expert kernels land.
    pub fn run(
        &self,
        ctx: &DispatchCtx,
        _gpu: &mut rdna_compute::Gpu,
        params: &MoeParams,
    ) -> Result<(), DispatchError> {
        let _resolved = self.resolve(params.variant, ctx)?;

        match params.variant {
            MoeVariant::GroupedGemm => Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "grouped_gemm",
                arch: "",
                quant: "",
            }),
            MoeVariant::IndexedGateUp | MoeVariant::IndexedDown => Err(
                DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "indexed",
                    arch: "",
                    quant: "",
                },
            ),
        }
    }
}

impl KernelFamily for MoeFamily {
    fn name(&self) -> &'static str {
        "moe"
    }
}
