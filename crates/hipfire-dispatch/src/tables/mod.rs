// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
pub mod gemm_table;
pub mod gemv_table;
pub mod moe_table;
pub mod rotation_table;
pub mod attention_table;
pub mod fused_qkv_table;

use std::collections::HashMap;
use crate::context::DispatchCtx;
use crate::types::{ArchPredicate, DispatchError, KernelKey, KernelVariant, ShapeInfo, ShapePredicate};

/// Kernel registry. Built once via `register`, frozen, read-only thereafter.
pub struct KernelRegistry {
    table: HashMap<KernelKey, Vec<KernelVariant>>,
}

impl KernelRegistry {
    pub fn new() -> Self {
        Self { table: HashMap::new() }
    }

    pub fn register(&mut self, entry: KernelVariant) {
        self.table.entry(entry.key).or_default().push(entry);
    }

    /// Resolve `key` to the first registered variant that passes both the
    /// arch predicate and (when provided) the shape predicate.
    ///
    /// Pass `shape: None` to bypass shape gating entirely — useful for arch
    /// probing and validation where tensor dimensions are not yet known.
    pub fn resolve(
        &self,
        key: KernelKey,
        ctx: &DispatchCtx,
        shape: Option<&ShapeInfo>,
    ) -> Result<&KernelVariant, DispatchError> {
        let variants = self.table.get(&key)
            .ok_or(DispatchError::NotFound { key })?;

        for variant in variants {
            if !variant.arch_required.eval_arch(ctx) {
                continue;
            }
            if let Some(ref gate) = variant.shape_gate {
                if let Some(s) = shape {
                    if !gate.eval(s) {
                        continue;
                    }
                }
                // shape is None → bypass shape gating for this call
            }
            return Ok(variant);
        }

        Err(DispatchError::MissingImpl { key })
    }

    pub fn validate(&self) -> Result<(), DispatchError> {
        for (key, variants) in self.table.iter() {
            if variants.is_empty() {
                return Err(DispatchError::EmptyEntry { key: *key });
            }
        }
        Ok(())
    }

    pub fn all_keys(&self) -> Vec<KernelKey> {
        self.table.keys().copied().collect()
    }

    /// All registered variants for a given key (arch/shape-ungated).
    /// For completeness tests that need to verify tile coverage.
    pub fn variants_for(&self, key: KernelKey) -> &[KernelVariant] {
        self.table.get(&key).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

impl ArchPredicate {
    pub fn eval_arch(&self, ctx: &DispatchCtx) -> bool {
        match self {
            Self::Always => true,
            // RDNA3 (gfx11) wave32 WMMA or RDNA4 (gfx12) WMMA.
            // Collapsed from the old HasWmmaW32 / HasWmmaW32Gfx12 pair — the gfx12-only
            // predicate was dead (zero kernel registrations) and the gfx12-admit fix for
            // every WMMA-family quant (MQ3, Lloyd, fused QKV/gate-up, MoE grouped,
            // GQA-fused attn) was the || gfx12 OR on HasWmmaW32. A single HasWmma
            // predicate backed by ArchCaps::has_wmma() is equivalent and enforces that
            // a new ArchPredicate variant only lands with the kernel it gates.
            Self::HasWmma => ctx.arch.has_wmma(),
            // gfx11-family wave32 WMMA only (RDNA3 + RDNA3.5), EXCLUDES gfx12/RDNA4 —
            // for WMMA kernels with no gfx12 source sibling (mb4, q8_0_wmma_x64).
            Self::HasWmmaW32 => ctx.arch.has_wmma_w32(),
            Self::HasWmmaGfx12 => ctx.arch.has_wmma_w32_gfx12(),
            Self::HasDot2F32F16 => ctx.arch.has_dot2_f32_f16(),
            Self::HasSdot4 => ctx.arch.has_hfq3_sdot4(),
            // MQ6/HFQ6 GEMV ships on RDNA4 (gemv_mq6g256_prerotated has a gfx12 build);
            // has_mmq is gfx906||rdna3 only, so admit RDNA4 explicitly.
            Self::HasMmq => ctx.arch.has_mmq() || ctx.arch.is_rdna4(),
            Self::HasCdna3LdsGemv => ctx.arch.has_cdna3_lds_gemv(),
            Self::HasDp4a => ctx.arch.gemv_dp4a_enabled(),
        }
    }
}

impl ShapePredicate {
    pub fn eval(&self, shape: &ShapeInfo) -> bool {
        match self {
            Self::BatchGt(n) => shape.batch_size > *n,
            Self::BatchGe(n) => shape.batch_size >= *n,
            Self::BatchEq(n) => shape.batch_size == *n,
            Self::HeadDimEq(n) => shape.head_dim == *n,
            Self::HeadDimLe(n) => shape.head_dim <= *n,
            Self::HeadDimMultipleOf(n) => shape.head_dim % *n == 0,
            Self::HeadDimIn(set) => set.contains(&shape.head_dim),
            Self::MLt(n) => shape.m < *n,
            Self::IsTree(b) => shape.is_tree == *b,
            Self::And(preds) => preds.iter().all(|p| p.eval(shape)),
        }
    }
}
