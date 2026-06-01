// SPDX-License-Identifier: MIT OR Apache-2.0
pub mod gemm_table;
pub mod gemv_table;
pub mod moe_table;
pub mod rotation_table;
pub mod attention_table;
pub mod fused_qkv_table;

use std::collections::HashMap;
use std::sync::Mutex;
use crate::context::DispatchCtx;
use crate::types::{ArchPredicate, DispatchError, KernelKey, KernelVariant, ShapeInfo, ShapePredicate};

/// Thread-safe kernel registry. Populated once at init, read-only thereafter.
pub struct KernelRegistry {
    table: Mutex<HashMap<KernelKey, Vec<KernelVariant>>>,
}

impl KernelRegistry {
    pub fn new() -> Self {
        Self { table: Mutex::new(HashMap::new()) }
    }

    pub fn register(&self, entry: KernelVariant) {
        let mut table = self.table.lock().unwrap();
        table.entry(entry.key).or_default().push(entry);
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
    ) -> Result<KernelKey, DispatchError> {
        let table = self.table.lock().unwrap();
        let variants = table.get(&key)
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
            return Ok(variant.key);
        }

        Err(DispatchError::MissingImpl { key })
    }

    pub fn validate(&self) -> Result<(), DispatchError> {
        let table = self.table.lock().unwrap();
        for (key, variants) in table.iter() {
            if variants.is_empty() {
                return Err(DispatchError::EmptyEntry { key: *key });
            }
        }
        Ok(())
    }

    pub fn all_keys(&self) -> Vec<KernelKey> {
        let table = self.table.lock().unwrap();
        table.keys().copied().collect()
    }
}

impl ArchPredicate {
    pub fn eval_arch(&self, ctx: &DispatchCtx) -> bool {
        match self {
            Self::Always => true,
            Self::HasWmmaW32 => ctx.arch.has_wmma_w32(),
            Self::HasWmmaW32Gfx12 => ctx.arch.has_wmma_w32_gfx12(),
            Self::HasDp4a => ctx.arch.has_dot2_f32_f16(),
            Self::HasSdot4 => ctx.arch.has_hfq3_sdot4(),
            Self::HasMmq => ctx.arch.has_mmq(),
            Self::HasCdna3LdsGemv => ctx.arch.has_cdna3_lds_gemv(),
        }
    }
}

impl ShapePredicate {
    pub fn eval(&self, shape: &ShapeInfo) -> bool {
        match self {
            Self::BatchGt(n) => shape.batch_size > *n,
            Self::HeadDimEq(n) => shape.head_dim == *n,
            Self::MLt(n) => shape.m < *n,
        }
    }
}
