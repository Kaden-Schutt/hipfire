// SPDX-License-Identifier: MIT OR Apache-2.0
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
    ) -> Result<KernelKey, DispatchError> {
        let key = match dtype {
            DType::F32 => KernelKey::GemmF32RegisterTiled,
            DType::F16 => KernelKey::GemmF16XF16Wmma,
            DType::Q8_0 => {
                let preferred = KernelKey::GemmQ8_0Wmma;
                if self.registry.resolve(preferred, ctx, None).is_ok() {
                    preferred
                } else {
                    KernelKey::GemmQ8_0BatchedChunked
                }
            }
            DType::HFQ4G256 => {
                let preferred = KernelKey::GemmHfq4G256Wmma;
                if self.registry.resolve(preferred, ctx, None).is_ok() {
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
        self.registry.resolve(key, ctx, None)
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
        let _resolved = self.resolve(params.w.dtype, ctx)?;

        let w = params.w;
        let x = params.x;
        let y = params.y;
        let batch_size = params.batch_size;
        let m = w.m;
        let k = w.k;
        use DType::*;

        macro_rules! hip {
            ($e:expr) => {
                $e.map_err(|e| DispatchError::Hip(e.to_string()))
            };
        }

        match w.dtype {
            F32 => hip!(gpu.gemm_f32_register_tiled(w.buf, x, y, m, k, batch_size)),
            F16 => hip!(gpu.gemm_f16_x_f16_wmma(w.buf, x, y, m, k, batch_size)),
            Q8_0 => {
                let preferred = KernelKey::GemmQ8_0Wmma;
                if self.registry.resolve(preferred, ctx, None).is_ok() {
                    hip!(gpu.gemm_q8_0_wmma(w.buf, x, y, m, k, batch_size))
                } else {
                    hip!(gpu.gemm_q8_0_batched_chunked(w.buf, x, y, m, k, batch_size))
                }
            }
            HFQ4G256 => {
                let preferred = KernelKey::GemmHfq4G256Wmma;
                if self.registry.resolve(preferred, ctx, None).is_ok() {
                    hip!(gpu.gemm_hfq4g256_wmma(w.buf, x, y, m, k, batch_size))
                } else {
                    hip!(gpu.gemm_hfq4g256(w.buf, x, y, m, k, batch_size))
                }
            }
            HFQ4G128 => hip!(gpu.gemm_hfq4g128(w.buf, x, y, m, k, batch_size)),
            _ => Err(DispatchError::UnsupportedVariant {
                family: "gemm", variant: "plain",
                arch: "", quant: "",
            }),
        }
    }
}

impl KernelFamily for GemmFamily {
    fn name(&self) -> &'static str {
        "gemm"
    }
}
