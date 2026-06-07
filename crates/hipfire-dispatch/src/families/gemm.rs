// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
//! GEMM kernel family: dispatching batched matrix-matrix multiply across quant formats.
//!
//! GEMM is a single-variant family (no Prerotated / WithResidual distinction —
//! those are layer-level concerns handled by the caller). Dispatch is by dtype
//! only, with WMMA-preferred routing where available.

use rdna_compute::{DType, Gpu, GpuTensor};

use crate::context::DispatchCtx;
use crate::families::gemv::WeightRef;
use crate::tables::gemm_table;
use crate::tables::KernelRegistry;
use crate::traits::KernelFamily;
use crate::types::*;

// ── Dispatch parameters ────────────────────────────────

pub struct GemmParams<'a> {
    pub w: &'a WeightRef<'a>,
    pub x: &'a GpuTensor,
    pub y: &'a GpuTensor,
    pub batch_size: usize,
}

// ── Family ─────────────────────────────────────────────

pub struct GemmFamily {
    registry: KernelRegistry,
}

impl GemmFamily {
    pub fn new() -> Self {
        let mut registry = KernelRegistry::new();
        gemm_table::populate(&mut registry);
        registry.validate().expect("gemm kernel table has empty entries");
        Self { registry }
    }

    pub fn registry(&self) -> &KernelRegistry {
        &self.registry
    }

    /// Resolve the best kernel key for the given dtype.
    ///
    /// Applies arch gating through `KernelRegistry::resolve`. For dtypes that
    /// have both a WMMA and a non-WMMA path (Q8_0, HFQ4G256), the WMMA variant
    /// is preferred when the arch supports it.
    pub fn resolve(
        &self,
        dtype: DType,
        ctx: &DispatchCtx,
        shape: Option<&ShapeInfo>,
    ) -> Result<&KernelVariant, DispatchError> {
        let key = match dtype {
            DType::F32 => KernelKey::GemmF32RegisterTiled,
            DType::F16 => KernelKey::GemmF16XF16Wmma,
            DType::Q8_0 => {
                let preferred = KernelKey::GemmQ8_0Wmma;
                if self.registry.resolve(preferred, ctx, shape).is_ok() {
                    preferred
                } else {
                    KernelKey::GemmQ8_0BatchedChunked
                }
            }
            DType::HFQ4G256 => {
                let preferred = KernelKey::GemmHfq4G256Wmma;
                if self.registry.resolve(preferred, ctx, shape).is_ok() {
                    preferred
                } else {
                    KernelKey::GemmHfq4G256
                }
            }
            DType::HFQ4G128 => KernelKey::GemmHfq4G128,
            _ => {
                return Err(DispatchError::UnsupportedVariant {
                    family: "gemm", variant: "plain",
                    arch: "", quant: "",
                })
            }
        };
        self.registry.resolve(key, ctx, shape)
    }

    /// Run a GEMM operation.
    ///
    /// Validates arch compatibility via `resolve()`, then dispatches to the
    /// correct `Gpu` method.
    pub fn run(
        &self,
        ctx: &DispatchCtx,
        gpu: &mut Gpu,
        params: &GemmParams,
    ) -> Result<(), DispatchError> {
        let key = self.resolve(params.w.dtype, ctx, None)?.key;

        let w = params.w;
        let x = params.x;
        let y = params.y;
        let batch_size = params.batch_size;
        let m = w.m;
        let k = w.k;

        macro_rules! hip {
            ($e:expr) => {
                $e.map_err(|e| DispatchError::Hip(e.to_string()))
            };
        }

        use KernelKey as K;
        match key {
            K::GemmF32RegisterTiled => hip!(gpu.gemm_f32_register_tiled(w.buf, x, y, m, k, batch_size)),
            K::GemmF16XF16Wmma => hip!(gpu.gemm_f16_x_f16_wmma(w.buf, x, y, m, k, batch_size)),
            K::GemmQ8_0Wmma => hip!(gpu.gemm_q8_0_wmma(w.buf, x, y, m, k, batch_size)),
            K::GemmQ8_0BatchedChunked => hip!(gpu.gemm_q8_0_batched_chunked(w.buf, x, y, m, k, batch_size)),
            K::GemmHfq4G256Wmma => hip!(gpu.gemm_hfq4g256_wmma(w.buf, x, y, m, k, batch_size)),
            K::GemmHfq4G256 => hip!(gpu.gemm_hfq4g256(w.buf, x, y, m, k, batch_size)),
            K::GemmHfq4G128 => hip!(gpu.gemm_hfq4g128(w.buf, x, y, m, k, batch_size)),
            // #397 Ship 5.1: plain-GEMM catalog. Each arm maps the registered
            // KernelKey to the exact rdna-compute method with the canonical
            // `(a, x, y, m, k, batch_size)` signature.
            K::GemmF16 => hip!(gpu.gemm_f16(w.buf, x, y, m, k, batch_size)),
            K::GemmF16Tiled => hip!(gpu.gemm_f16_tiled(w.buf, x, y, m, k, batch_size)),
            K::GemmF16WmmaMb4 => hip!(gpu.gemm_f16_wmma_mb4(w.buf, x, y, m, k, batch_size)),
            K::GemmF16WmmaMb8 => hip!(gpu.gemm_f16_wmma_mb8(w.buf, x, y, m, k, batch_size)),
            K::GemmF32Batched => hip!(gpu.gemm_f32_batched(w.buf, x, y, m, k, batch_size)),
            K::GemmQ8_0WmmaX64 => hip!(gpu.gemm_q8_0_wmma_x64(w.buf, x, y, m, k, batch_size)),
            K::GemmQ8_0ResidualWmma => hip!(gpu.gemm_q8_0_residual_wmma(w.buf, x, y, m, k, batch_size)),
            K::GemmQ8_0ResidualWmmaGfx12 => hip!(gpu.gemm_q8_0_residual_wmma_gfx12(w.buf, x, y, m, k, batch_size)),
            K::GemmHfq4G256Dp4a => hip!(gpu.gemm_hfq4g256_dp4a(w.buf, x, y, m, k, batch_size)),
            K::GemmHfq4G256MmqSet => hip!(gpu.gemm_hfq4g256_mmq_set(w.buf, x, y, m, k, batch_size)),
            other => Err(DispatchError::MissingImpl { key: other }),
        }
    }
}

impl KernelFamily for GemmFamily {
    fn name(&self) -> &'static str {
        "gemm"
    }
}
