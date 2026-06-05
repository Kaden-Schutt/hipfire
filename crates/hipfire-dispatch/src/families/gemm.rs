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

/// Which batched lm_head GEMM kernel family to dispatch.
///
/// The lm_head kernel is selected by the *kernel* family (HFQ3 / HFQ4 / HFQ6),
/// which is NOT the same as the weight's source dtype: MQ4 weights route to the
/// HFQ4 kernel (after the caller FWHT-rotates x), MQ3 → HFQ3, MQ6 → HFQ6. So the
/// caller names the kernel explicitly rather than letting `run` infer it from
/// `WeightRef::dtype`. Each maps to a `gemm_hfqXg256_batched_lmhead` wrapper in
/// rdna-compute, which zero-inits Y and self-selects WMMA-residual / dp4a /
/// per-row GEMV per arch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GemmLmHeadKernel {
    Hfq4G256,
    Hfq3G256,
    Hfq6G256,
}

impl GemmLmHeadKernel {
    fn key(self) -> KernelKey {
        match self {
            GemmLmHeadKernel::Hfq4G256 => KernelKey::GemmHfq4G256BatchedLmhead,
            GemmLmHeadKernel::Hfq3G256 => KernelKey::GemmHfq3G256BatchedLmhead,
            GemmLmHeadKernel::Hfq6G256 => KernelKey::GemmHfq6G256BatchedLmhead,
        }
    }
}

/// Parameters for a batched lm_head GEMM dispatch.
///
/// `x` is the (already-rotated, when the source weights are MQ) hidden-state
/// batch; `y` is the `batch * m` logits output. Mirrors the direct
/// `gpu.gemm_hfqXg256_batched_lmhead(buf, x, y, m, k, batch)` call shape.
pub struct GemmLmHeadParams<'a> {
    pub kernel: GemmLmHeadKernel,
    pub w_buf: &'a GpuTensor,
    pub x: &'a GpuTensor,
    pub y: &'a GpuTensor,
    pub m: usize,
    pub k: usize,
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
            other => Err(DispatchError::MissingImpl { key: other }),
        }
    }

    /// Run a batched lm_head GEMM (`y[b][row] = A[row] · x[b]`, zero-init Y).
    ///
    /// Dispatches the `gemm_hfqXg256_batched_lmhead` wrapper for the named
    /// kernel family. The wrapper self-selects WMMA-residual / dp4a / per-row
    /// GEMV per arch, so `resolve` only confirms the key is registered (the
    /// arch predicate is `Always`). Byte-identical to the prior direct
    /// `gpu.gemm_hfqXg256_batched_lmhead(...)` calls in the DFlash path.
    pub fn run_batched_lmhead(
        &self,
        ctx: &DispatchCtx,
        gpu: &mut Gpu,
        params: &GemmLmHeadParams,
    ) -> Result<(), DispatchError> {
        let key = self.registry.resolve(params.kernel.key(), ctx, None)?.key;

        let w_buf = params.w_buf;
        let x = params.x;
        let y = params.y;
        let m = params.m;
        let k = params.k;
        let batch_size = params.batch_size;

        macro_rules! hip {
            ($e:expr) => {
                $e.map_err(|e| DispatchError::Hip(e.to_string()))
            };
        }

        use KernelKey as K;
        match key {
            K::GemmHfq4G256BatchedLmhead => {
                hip!(gpu.gemm_hfq4g256_batched_lmhead(w_buf, x, y, m, k, batch_size))
            }
            K::GemmHfq3G256BatchedLmhead => {
                hip!(gpu.gemm_hfq3g256_batched_lmhead(w_buf, x, y, m, k, batch_size))
            }
            K::GemmHfq6G256BatchedLmhead => {
                hip!(gpu.gemm_hfq6g256_batched_lmhead(w_buf, x, y, m, k, batch_size))
            }
            other => Err(DispatchError::MissingImpl { key: other }),
        }
    }
}

impl KernelFamily for GemmFamily {
    fn name(&self) -> &'static str {
        "gemm"
    }
}
