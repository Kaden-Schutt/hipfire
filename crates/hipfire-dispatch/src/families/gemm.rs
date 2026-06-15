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
        self.run_key(key, ctx, gpu, params)
    }

    /// Run a GEMM operation against an *explicit* [`KernelKey`], bypassing the
    /// dtype-keyed WMMA-preference heuristic in [`resolve`].
    ///
    /// This is the behavior-preserving migration primitive for prefill call
    /// sites that historically invoked a *specific* `gpu.gemm_*` method whose
    /// own internal arch dispatch (e.g. `gemm_hfq4g256` routing to dp4a /
    /// rocBLAS / WMMA, or `gemm_q8_0_batched_chunked` routing to RDNA4 WMMA)
    /// must be preserved exactly. Passing the dispatcher-entry key
    /// (`GemmHfq4G256`, `GemmQ8_0BatchedChunked`, `GemmF32Batched`, …) routes to
    /// the identical `gpu.gemm_*` method the direct call used, so output is
    /// byte-identical on every (dtype × arch × shape).
    ///
    /// `resolve` (dtype-keyed) would instead *front-run* the kernel's internal
    /// dispatch by preferring a single WMMA variant, which can diverge from the
    /// direct call on some arches — so it is NOT appropriate for migrating a
    /// site that called the dispatcher entry point directly. Use this method for
    /// those; use [`run`] only where the dtype-keyed heuristic matches the
    /// site's prior behavior.
    pub fn run_key(
        &self,
        key: KernelKey,
        ctx: &DispatchCtx,
        gpu: &mut Gpu,
        params: &GemmParams,
    ) -> Result<(), DispatchError> {
        // Validate the explicit key is registered and arch-admissible. The
        // dispatcher-entry keys used at migrated prefill sites are registered
        // `ArchPredicate::Always`, so this never rejects on a supported build.
        let key = self.registry.resolve(key, ctx, None)?.key;

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
            // #397 Ship 5.2 FINAL: residual-fused GEMM catalog. Each arm computes
            // `y += a·x` IN-PLACE (the add is internal to the kernel; `y` carries
            // the residual stream and is never reused as GEMV scratch). The
            // operand order `(w.buf, x, y, m, k, batch_size)` is byte-identical to
            // the prior direct `gpu.gemm_*_residual(&w.buf, x, y, m, k, n)` call,
            // so each kernel's internal arch routing is preserved exactly.
            K::GemmHfq6G256Residual => hip!(gpu.gemm_hfq6g256_residual(w.buf, x, y, m, k, batch_size)),
            K::GemmHfq4G256Residual => hip!(gpu.gemm_hfq4g256_residual(w.buf, x, y, m, k, batch_size)),
            // HFQ3 residual mirrors the qwen35 call site's WMMA-vs-base arch split
            // (`if arch_has_wmma { _wmma } else { base }`); has_wmma() includes
            // gfx12, and the _wmma method routes the gfx12 sibling internally.
            K::GemmHfq3G256Residual => {
                if gpu.arch_caps.has_wmma() {
                    hip!(gpu.gemm_hfq3g256_residual_wmma(w.buf, x, y, m, k, batch_size))
                } else {
                    hip!(gpu.gemm_hfq3g256_residual(w.buf, x, y, m, k, batch_size))
                }
            }
            // HFP4 / MQ3-Lloyd residual are WMMA-only dispatcher entries; each
            // routes its own gfx12-vs-gfx11 WMMA sibling internally.
            K::GemmHfp4G32Residual => hip!(gpu.gemm_hfp4g32_residual(w.buf, x, y, m, k, batch_size)),
            K::GemmMq3G256LloydResidual => hip!(gpu.gemm_mq3g256_lloyd_residual_wmma(w.buf, x, y, m, k, batch_size)),
            // #397 Ship 5.3: spec-decode (DFlash) batched lm_head catalog. Each
            // arm maps the explicit key to the exact rdna-compute method the prior
            // direct spec-decode call used. The operand order
            // `(w.buf, x, y, m, k, batch_size)` is byte-identical, and each method
            // keeps its own internal arch routing (WMMA for batch>1 on gfx11/12,
            // dp4a on gfx906, fp16/scalar fallback) so output is preserved exactly.
            K::GemmQ8_0Batched => hip!(gpu.gemm_q8_0_batched(w.buf, x, y, m, k, batch_size)),
            K::GemmHfq4G256BatchedLmhead => hip!(gpu.gemm_hfq4g256_batched_lmhead(w.buf, x, y, m, k, batch_size)),
            K::GemmHfq3G256BatchedLmhead => hip!(gpu.gemm_hfq3g256_batched_lmhead(w.buf, x, y, m, k, batch_size)),
            K::GemmHfq6G256BatchedLmhead => hip!(gpu.gemm_hfq6g256_batched_lmhead(w.buf, x, y, m, k, batch_size)),
            other => Err(DispatchError::MissingImpl { key: other }),
        }
    }
}

impl KernelFamily for GemmFamily {
    fn name(&self) -> &'static str {
        "gemm"
    }
}
