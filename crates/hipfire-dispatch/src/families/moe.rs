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

use rdna_compute::DType;
use rdna_compute::GpuTensor;

use crate::context::DispatchCtx;
use crate::families::gemv::WeightRef;
use crate::tables::moe_table;
use crate::tables::KernelRegistry;
use crate::traits::KernelFamily;
use crate::types::*;

// ── MoE eligibility lattice ────────────────────────────

/// Per-layer dtype snapshot the MoE eligibility lattice reads. Built by the
/// model from its weight structs; kept dtype-only so this stays GPU-free and
/// the dispatch crate needs no dependency on any arch crate.
///
/// `experts_all_gate_up_mq4` mirrors the `ffn.experts.iter().all(..)` clause
/// the original `gate_side_mq4` check used (qwen35.rs:4598-4605); the routed
/// fields use experts[0] as representative (the loader builds all experts in a
/// layer with matching dtype, so [0] == all — same invariant the original
/// routed_* checks relied on).
pub struct MoeDtypes {
    pub router: DType,
    pub shared_gate: DType,          // ffn.shared_expert_gate
    pub shared_expert_gate: DType,   // ffn.shared_expert.gate
    pub shared_expert_up: DType,     // ffn.shared_expert.up
    pub experts_all_gate_up_mq4: bool,
    pub routed_gate_up: DType,       // ffn.experts[0].gate_up
    pub routed_down: DType,          // ffn.experts[0].down
    pub has_paro_shared: bool,       // ffn.paro_shared.is_some()
}

/// Resolved fused-vs-fallback eligibility for one MoE decode layer. This IS the
/// routing-config logic, relocated from `moe_ffn_decode_impl` into one typed,
/// testable place (review finding #1). Pure function of `MoeDtypes` + k.
#[derive(Clone, Copy, Debug)]
pub struct MoeResolution {
    pub gate_side_mq4: bool,
    pub routed_indexable_mq4: bool,
    pub routed_indexable_mq6: bool,
    pub routed_indexable_paro: bool,
    pub use_gpu_topk: bool,
    pub needs_x_rot_local: bool,
}

impl MoeResolution {
    pub fn resolve(d: &MoeDtypes, k: usize) -> Self {
        use DType::*;
        let gate_side_mq4 = d.router == MQ4G256
            && d.shared_gate == MQ4G256
            && d.shared_expert_gate == MQ4G256
            && d.shared_expert_up == MQ4G256
            && d.experts_all_gate_up_mq4;

        let routed_gate_up_mq4 = d.routed_gate_up == MQ4G256;
        let routed_gate_up_mq6 = d.routed_gate_up == MQ6G256;
        let routed_gate_up_paro = d.routed_gate_up == ParoQ4G128 && d.has_paro_shared;

        let routed_indexable_mq4 = (d.routed_down == MQ4G256) && routed_gate_up_mq4;
        let routed_indexable_mq6 = (d.routed_down == MQ6G256) && routed_gate_up_mq6;
        let routed_indexable_paro =
            (d.routed_down == ParoQ4G128 && d.has_paro_shared) && routed_gate_up_paro;

        let routed_dtype_indexable =
            routed_indexable_mq4 || routed_indexable_mq6 || routed_indexable_paro;

        let use_gpu_topk = k == 8 && routed_dtype_indexable;
        let needs_x_rot_local = gate_side_mq4
            || routed_gate_up_mq4
            || routed_gate_up_mq6
            || routed_gate_up_paro;

        Self {
            gate_side_mq4,
            routed_indexable_mq4,
            routed_indexable_mq6,
            routed_indexable_paro,
            use_gpu_topk,
            needs_x_rot_local,
        }
    }

    pub fn routed_indexable(&self) -> bool {
        self.routed_indexable_mq4 || self.routed_indexable_mq6 || self.routed_indexable_paro
    }
}

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
        let mut registry = KernelRegistry::new();
        moe_table::populate(&mut registry);
        registry.validate().expect("moe kernel table has empty entries");
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
        shape: Option<&ShapeInfo>,
    ) -> Result<&KernelVariant, DispatchError> {
        let key = match variant {
            MoeVariant::IndexedGateUp => KernelKey::MoeIndexedGateUpLloyd,
            MoeVariant::IndexedDown => KernelKey::MoeIndexedDownLloyd,
            MoeVariant::GroupedGemm => KernelKey::MoeGroupedGemm,
        };
        self.registry.resolve(key, ctx, shape)
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
        let _key = self.resolve(params.variant, ctx, None)?.key;

        Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "all",
            arch: "",
            quant: "",
        })
    }
}

impl KernelFamily for MoeFamily {
    fn name(&self) -> &'static str {
        "moe"
    }
}
