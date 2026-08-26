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

fn is_gemm_hfq4_key(key: KernelKey) -> bool {
    matches!(
        key,
        KernelKey::GemmHfq4G256
            | KernelKey::GemmHfq4G256Residual
            | KernelKey::GemmHfq4G256BatchedLmhead
    )
}

fn is_gemm_mq4v2_key(key: KernelKey) -> bool {
    matches!(
        key,
        KernelKey::GemmMq4G256V2
            | KernelKey::GemmMq4G256V2Residual
            | KernelKey::GemmMq4G256V2BatchedLmhead
    )
}

fn is_gemm_mq3v2_key(key: KernelKey) -> bool {
    matches!(
        key,
        KernelKey::GemmMq3G256V2
            | KernelKey::GemmMq3G256V2Residual
            | KernelKey::GemmMq3G256V2BatchedLmhead
    )
}

fn is_gemm_mq2v2_key(key: KernelKey) -> bool {
    matches!(
        key,
        KernelKey::GemmMq2G256V2
            | KernelKey::GemmMq2G256V2Residual
            | KernelKey::GemmMq2G256V2BatchedLmhead
    )
}

fn is_gemm_mq5v2_key(key: KernelKey) -> bool {
    matches!(
        key,
        KernelKey::GemmMq5G256V2
            | KernelKey::GemmMq5G256V2Residual
            | KernelKey::GemmMq5G256V2BatchedLmhead
    )
}

fn is_gemm_mq6v2_key(key: KernelKey) -> bool {
    matches!(
        key,
        KernelKey::GemmMq6G256V2
            | KernelKey::GemmMq6G256V2Residual
            | KernelKey::GemmMq6G256V2BatchedLmhead
    )
}

fn is_gemm_mq4c_key(key: KernelKey) -> bool {
    matches!(
        key,
        KernelKey::GemmMq4CG256
            | KernelKey::GemmMq4CG256Residual
            | KernelKey::GemmMq4CG256BatchedLmhead
    )
}

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
        registry
            .validate()
            .expect("gemm kernel table has empty entries");
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
            // Native BF16 stays BF16: on gfx942 this is the MFMA GEMM. On any
            // other arch the registry rejects IsGfx942 and resolve() returns
            // UnsupportedVariant, so a caller must fall back explicitly rather
            // than silently landing on a scalar kernel.
            DType::BF16 => KernelKey::GemmBf16Mfma,
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
            DType::TQ2G128 => KernelKey::GemmTQ2G128Prefill,
            DType::BQ1G128 => KernelKey::GemmBQ1G128Prefill,
            DType::MQ4G256V2 => KernelKey::GemmMq4G256V2,
            DType::MQ6G256V2 => KernelKey::GemmMq6G256V2,
            DType::MQ5G256V2 => KernelKey::GemmMq5G256V2,
            DType::MQ3G256V2 => KernelKey::GemmMq3G256V2,
            DType::MQ2G256V2 => KernelKey::GemmMq2G256V2,
            DType::MQ4CG256 => KernelKey::GemmMq4CG256,
            _ => {
                return Err(DispatchError::UnsupportedVariant {
                    family: "gemm",
                    variant: "plain",
                    arch: "",
                    quant: "",
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

        // Calibration tap. This is the batched chokepoint for every arch that
        // migrated off `llama::weight_gemm` onto dispatcher-entry keys —
        // gemma4 (`lowered.rs:173`), muse-glimmer (`forward.rs:80`) and
        // qwen35's migrated prefill sites. Without it those arches capture
        // NOTHING on the batched path, because `weight_gemm`'s tap never runs
        // for them. `batch_size` is the real row count, so one prefill call
        // contributes that many rows to H and Σx².
        // Zero cost when unarmed (`active_capture.is_none()` early-returns).
        //
        // A BF16 `x` is a STAGED copy of an F32 activation, produced by an arch
        // for GemmBf16Mfma. Do not capture it: the collector copies `n*k` F32
        // elements, so a 2-byte-per-element buffer overruns (assert in
        // `memcpy_dtod_at`), and even sized correctly it would record the
        // bf16-rounded activation rather than the true one. Arches that stage
        // MUST capture the original F32 source before converting.
        if x.dtype != DType::BF16 {
            gpu.maybe_capture_activation(w.buf, x, batch_size, k);
        }

        // Guard: V2/v1.5 bytes through wrong-header kernels is silent noise.
        // qt=44 = 136 B/group (two fp16 half-grids); qt=45 = 136 B/group (one
        // packed fp16 scale/zero dword at [0..4), 4 B pad at [4..8), 128 B
        // nibbles at [8..136)); v1 = 136 B/group (f32 scale/zero). Equal stride
        // does not make the layouts interchangeable — a mis-route decodes every
        // group with the wrong header and returns noise at full speed with no
        // HIP error.
        if w.dtype == DType::MQ4G256V2 && is_gemm_hfq4_key(key) {
            return Err(DispatchError::Hip(format!(
                "qt=44 (MQ4G256V2) weight (dtype {:?}) routed to v1 kernel key {:?}: \
                 v2 stores fp16 scale/zero per 128 weights (s0/z0 for 0..127, s1/z1 for 128..255) \
                 where v1 stores f32 scale/zero per 256, so the v1 kernel decodes every weight \
                 to ~1e-14. This is a missing v2 routing arm at the callsite, not a valid configuration.",
                w.dtype, key
            )));
        }
        if w.dtype == DType::MQ6G256V2 && is_gemm_hfq4_key(key) {
            return Err(DispatchError::Hip(format!(
                "qt=47 (MQ6G256V2) weight (dtype {:?}) routed to v1 kernel key {:?}: v2 stores fp16 scale/zero per 128 where v1 stores f32; mis-route is silent.",
                w.dtype, key
            )));
        }
        if w.dtype == DType::MQ5G256V2 && is_gemm_hfq4_key(key) {
            return Err(DispatchError::Hip(format!(
                "qt=48 (MQ5G256V2) weight (dtype {:?}) routed to v1 kernel key {:?}: v2 stores fp16 scale/zero per 128 where v1 stores f32; mis-route is silent.",
                w.dtype, key
            )));
        }
        if w.dtype == DType::MQ3G256V2 && is_gemm_hfq4_key(key) {
            return Err(DispatchError::Hip(format!(
                "qt=49 (MQ3G256V2) weight (dtype {:?}) routed to v1 kernel key {:?}: \
                 v2 stores fp16 scale/zero per 128 weights (s0/z0 for 0..127, s1/z1 for 128..255) \
                 where v1 stores f32 scale/zero per 256, so the v1 kernel decodes every weight \
                 to ~1e-14. This is a missing v2 routing arm at the callsite, not a valid configuration.",
                w.dtype, key
            )));
        }
        if w.dtype == DType::MQ6G256V2
            && (is_gemm_mq4v2_key(key)
                || is_gemm_mq3v2_key(key)
                || is_gemm_mq2v2_key(key)
                || is_gemm_mq5v2_key(key)
                || is_gemm_mq4c_key(key))
        {
            return Err(DispatchError::Hip(format!(
                "qt=47 (MQ6G256V2) weight (dtype {:?}) routed to incompatible kernel key {:?}: mq6v2 is 200 B/group (dual fp16 per 128, 192 B 6-bit) while target differs; mis-route decodes every group wrong.",
                w.dtype, key
            )));
        }
        if w.dtype == DType::MQ5G256V2
            && (is_gemm_mq4v2_key(key)
                || is_gemm_mq3v2_key(key)
                || is_gemm_mq2v2_key(key)
                || is_gemm_mq6v2_key(key)
                || is_gemm_mq4c_key(key))
        {
            return Err(DispatchError::Hip(format!(
                "qt=48 (MQ5G256V2) weight (dtype {:?}) routed to incompatible kernel key {:?}: mq5v2 is 168 B/group; mis-route decodes every group wrong.",
                w.dtype, key
            )));
        }
        if w.dtype == DType::MQ3G256V2
            && (is_gemm_mq4v2_key(key) || is_gemm_mq4c_key(key) || is_gemm_mq2v2_key(key))
        {
            return Err(DispatchError::Hip(format!(
                "qt=49 (MQ3G256V2) weight (dtype {:?}) routed to incompatible kernel key {:?}: \
                 mq3v2 is 104 B/group (dual fp16 per 128, 96 B 3-bit) while the target is 136/72 B; \
                 equal stride is not interchangeable — a mis-route decodes every group wrong.",
                w.dtype, key
            )));
        }
        if w.dtype == DType::MQ2G256V2 && is_gemm_hfq4_key(key) {
            return Err(DispatchError::Hip(format!(
                "qt=50 (MQ2G256V2) weight (dtype {:?}) routed to v1 kernel key {:?}: \
                 v2 stores fp16 scale/zero per 128 weights (s0/z0 for 0..127, s1/z1 for 128..255) \
                 where v1 stores f32 scale/zero per 256, so the v1 kernel decodes every weight \
                 to ~1e-14. This is a missing v2 routing arm at the callsite, not a valid configuration.",
                w.dtype, key
            )));
        }
        if (w.dtype == DType::HFQ2G256 || w.dtype == DType::MQ2G256) && is_gemm_mq2v2_key(key) {
            return Err(DispatchError::Hip(format!(
                "v1 weight (dtype {:?}) routed to v2 kernel key {:?}: \
                 v2 expects fp16 s0/z0/s1/z1 per 128 weights while v1 stores f32 scale/zero per 256. \
                 Routing a v1 payload through a v2 kernel is equally wrong and silent.",
                w.dtype, key
            )));
        }
        if w.dtype == DType::MQ4CG256
            && (is_gemm_hfq4_key(key) || is_gemm_mq4v2_key(key) || is_gemm_mq3v2_key(key))
        {
            return Err(DispatchError::Hip(format!(
                "qt=45 (MQ4CG256) weight (dtype {:?}) routed to incompatible kernel key {:?}: \
                 mq4c stores one packed fp16 scale/zero dword at [0..4), 4 B pad at [4..8), and \
                 128 B nibbles at [8..136) (136 B/group). v1/v2 also use 136 B groups but with \
                 different headers (f32 scale/zero or two fp16 half-grids), so equal stride is \
                 not interchangeable — a mis-route decodes every group wrong and returns noise \
                 at full speed with no error.",
                w.dtype, key
            )));
        }
        if (w.dtype == DType::HFQ4G256
            || w.dtype == DType::MQ4G256
            || w.dtype == DType::MQ3G256
            || w.dtype == DType::MQ2G256
            || w.dtype == DType::MQ5G256
            || w.dtype == DType::MQ6G256)
            && is_gemm_mq6v2_key(key)
        {
            return Err(DispatchError::Hip(format!(
                "v1 weight (dtype {:?}) routed to mq6v2 kernel key {:?}: \
                 mq6v2 expects 200 B dual-half groups while v1 uses different stride; mis-route is silent.",
                w.dtype, key
            )));
        }
        if (w.dtype == DType::HFQ4G256
            || w.dtype == DType::MQ4G256
            || w.dtype == DType::MQ3G256
            || w.dtype == DType::MQ2G256
            || w.dtype == DType::MQ5G256
            || w.dtype == DType::MQ6G256)
            && is_gemm_mq3v2_key(key)
        {
            return Err(DispatchError::Hip(format!(
                "v1 weight (dtype {:?}) routed to mq3v2 kernel key {:?}: \
                 mq3v2 expects 104 B dual-half groups while v1 uses 136/104 B single-scale; mis-route is silent.",
                w.dtype, key
            )));
        }
        if (w.dtype == DType::HFQ4G256 || w.dtype == DType::MQ4G256) && is_gemm_mq4v2_key(key) {
            return Err(DispatchError::Hip(format!(
                "v1 weight (dtype {:?}) routed to v2 kernel key {:?}: \
                 v2 expects fp16 s0/z0/s1/z1 per 128 weights while v1 stores f32 scale/zero per 256. \
                 Routing a v1 payload through a v2 kernel is equally wrong and silent.",
                w.dtype, key
            )));
        }
        if (w.dtype == DType::HFQ4G256
            || w.dtype == DType::MQ4G256
            || w.dtype == DType::MQ4G256V2
            || w.dtype == DType::MQ2G256V2)
            && is_gemm_mq4c_key(key)
        {
            return Err(DispatchError::Hip(format!(
                "non-mq4c weight (dtype {:?}) routed to mq4c (qt=45) kernel key {:?}: \
                 mq4c expects a 136 B group (packed fp16 scale/zero dword + 4 B pad + 128 B \
                 nibbles) while v1/v2 use different 136 B headers. Equal stride does not make \
                 the layouts interchangeable — reverse mis-route is equally silent and wrong.",
                w.dtype, key
            )));
        }
        macro_rules! hip {
            ($e:expr) => {
                $e.map_err(|e| DispatchError::Hip(e.to_string()))
            };
        }

        use KernelKey as K;
        match key {
            K::GemmF32RegisterTiled => {
                hip!(gpu.gemm_f32_register_tiled(w.buf, x, y, m, k, batch_size))
            }
            K::GemmF16XF16Wmma => hip!(gpu.gemm_f16_x_f16_wmma(w.buf, x, y, m, k, batch_size)),
            K::GemmQ8_0Wmma => hip!(gpu.gemm_q8_0_wmma(w.buf, x, y, m, k, batch_size)),
            K::GemmQ8_0BatchedChunked => {
                hip!(gpu.gemm_q8_0_batched_chunked(w.buf, x, y, m, k, batch_size))
            }
            K::GemmHfq4G256Wmma => hip!(gpu.gemm_hfq4g256_wmma(w.buf, x, y, m, k, batch_size)),
            K::GemmTQ2G128Prefill => {
                hip!(gpu.gemm_tq2g128_prefill(w.buf, x, y, m, k, batch_size))
            }
            K::GemmBQ1G128Prefill => {
                hip!(gpu.gemm_bq1g128_prefill(w.buf, x, y, m, k, batch_size))
            }
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
            // Pure BF16 x BF16 -> F32. Takes `&DeviceBuffer` rather than the
            // family's `&GpuTensor` convention, so reach through to the buffers.
            //
            // The activation MUST already be BF16. The wrapper only checks that
            // `B` is large enough (batch*k*2), and an F32 activation buffer is
            // batch*k*4 bytes — it passes that check and the kernel then reads
            // F32 bytes as BF16, producing garbage with no diagnostic. Refuse
            // instead: callers stage to BF16 before dispatching here.
            K::GemmBf16Mfma => {
                if x.dtype != DType::BF16 {
                    return Err(DispatchError::UnsupportedVariant {
                        family: "gemm",
                        variant: "bf16_mfma_needs_bf16_activation",
                        arch: "gfx942",
                        quant: "",
                    });
                }
                hip!(gpu.gemm_bf16_mfma_gfx942(&w.buf.buf, &x.buf, &y.buf, m, k, batch_size))
            }
            K::GemmQ8_0WmmaX64 => hip!(gpu.gemm_q8_0_wmma_x64(w.buf, x, y, m, k, batch_size)),
            K::GemmQ8_0ResidualWmma => {
                hip!(gpu.gemm_q8_0_residual_wmma(w.buf, x, y, m, k, batch_size))
            }
            K::GemmQ8_0ResidualWmmaGfx12 => {
                hip!(gpu.gemm_q8_0_residual_wmma_gfx12(w.buf, x, y, m, k, batch_size))
            }
            K::GemmHfq4G256Dp4a => hip!(gpu.gemm_hfq4g256_dp4a(w.buf, x, y, m, k, batch_size)),
            K::GemmHfq4G256MmqSet => hip!(gpu.gemm_hfq4g256_mmq_set(w.buf, x, y, m, k, batch_size)),
            // #397 Ship 5.2 FINAL: residual-fused GEMM catalog. Each arm computes
            // `y += a·x` IN-PLACE (the add is internal to the kernel; `y` carries
            // the residual stream and is never reused as GEMV scratch). The
            // operand order `(w.buf, x, y, m, k, batch_size)` is byte-identical to
            // the prior direct `gpu.gemm_*_residual(&w.buf, x, y, m, k, n)` call,
            // so each kernel's internal arch routing is preserved exactly.
            K::GemmHfq6G256Residual => {
                hip!(gpu.gemm_hfq6g256_residual(w.buf, x, y, m, k, batch_size))
            }
            K::GemmHfq4G256Residual => {
                hip!(gpu.gemm_hfq4g256_residual(w.buf, x, y, m, k, batch_size))
            }
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
            K::GemmHfp4G32Residual => {
                hip!(gpu.gemm_hfp4g32_residual(w.buf, x, y, m, k, batch_size))
            }
            K::GemmMq3G256LloydResidual => {
                hip!(gpu.gemm_mq3g256_lloyd_residual_wmma(w.buf, x, y, m, k, batch_size))
            }
            // #397 Ship 5.3: spec-decode (DFlash) batched lm_head catalog. Each
            // arm maps the explicit key to the exact rdna-compute method the prior
            // direct spec-decode call used. The operand order
            // `(w.buf, x, y, m, k, batch_size)` is byte-identical, and each method
            // keeps its own internal arch routing (WMMA for batch>1 on gfx11/12,
            // dp4a on gfx906, fp16/scalar fallback) so output is preserved exactly.
            K::GemmQ8_0Batched => hip!(gpu.gemm_q8_0_batched(w.buf, x, y, m, k, batch_size)),
            K::GemmHfq4G256BatchedLmhead => {
                hip!(gpu.gemm_hfq4g256_batched_lmhead(w.buf, x, y, m, k, batch_size))
            }
            K::GemmHfq3G256BatchedLmhead => {
                hip!(gpu.gemm_hfq3g256_batched_lmhead(w.buf, x, y, m, k, batch_size))
            }
            K::GemmHfq6G256BatchedLmhead => {
                hip!(gpu.gemm_hfq6g256_batched_lmhead(w.buf, x, y, m, k, batch_size))
            }
            K::GemmQ8_0BatchedWideExact => {
                hip!(gpu.gemm_q8_0_batched_wide_exact(w.buf, x, y, m, k, batch_size))
            }
            K::GemmMq4G256V2 => hip!(gpu.gemm_mq4g256v2(w.buf, x, y, m, k, batch_size)),
            K::GemmMq5G256V2 => hip!(gpu.gemm_mq5g256v2(w.buf, x, y, m, k, batch_size)),
            K::GemmMq6G256V2 => hip!(gpu.gemm_mq6g256v2(w.buf, x, y, m, k, batch_size)),
            K::GemmMq3G256V2 => hip!(gpu.gemm_mq3g256v2(w.buf, x, y, m, k, batch_size)),
            K::GemmMq2G256V2 => hip!(gpu.gemm_mq2g256v2(w.buf, x, y, m, k, batch_size)),
            K::GemmMq4G256V2Residual => {
                hip!(gpu.gemm_hfq4g256_residual_mq4v2(w.buf, x, y, m, k, batch_size))
            }
            K::GemmMq5G256V2Residual => {
                hip!(gpu.gemm_mq5g256v2_residual_wmma(w.buf, x, y, m, k, batch_size))
            }
            K::GemmMq6G256V2Residual => {
                hip!(gpu.gemm_mq6g256v2_residual_wmma(w.buf, x, y, m, k, batch_size))
            }
            K::GemmMq3G256V2Residual => {
                hip!(gpu.gemm_mq3g256v2_residual_wmma(w.buf, x, y, m, k, batch_size))
            }
            K::GemmMq2G256V2Residual => {
                hip!(gpu.gemm_mq2g256v2_residual_wmma(w.buf, x, y, m, k, batch_size))
            }
            K::GemmMq4G256V2BatchedLmhead => {
                hip!(gpu.gemm_mq4g256v2_batched_lmhead(w.buf, x, y, m, k, batch_size))
            }
            K::GemmMq5G256V2BatchedLmhead => {
                hip!(gpu.gemm_mq5g256v2_batched_lmhead(w.buf, x, y, m, k, batch_size))
            }
            K::GemmMq6G256V2BatchedLmhead => {
                hip!(gpu.gemm_mq6g256v2_batched_lmhead(w.buf, x, y, m, k, batch_size))
            }
            K::GemmMq3G256V2BatchedLmhead => {
                hip!(gpu.gemm_mq3g256v2_batched_lmhead(w.buf, x, y, m, k, batch_size))
            }
            K::GemmMq2G256V2BatchedLmhead => {
                hip!(gpu.gemm_mq2g256v2_batched_lmhead(w.buf, x, y, m, k, batch_size))
            }
            K::GemmMq4CG256 => hip!(gpu.gemm_mq4cg256(w.buf, x, y, m, k, batch_size)),
            K::GemmMq4CG256Residual => {
                hip!(gpu.gemm_mq4cg256_residual(w.buf, x, y, m, k, batch_size))
            }
            K::GemmMq4CG256BatchedLmhead => {
                hip!(gpu.gemm_mq4cg256_batched_lmhead(w.buf, x, y, m, k, batch_size))
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

#[cfg(test)]
mod tests {
    use super::GemmFamily;
    use crate::context::DispatchCtx;
    use crate::types::{DispatchError, KernelKey};
    use rdna_compute::DType;

    #[test]
    fn v2_plain_resolves_exact_not_hfq4() {
        // All five V2 widths (MQ4V2 qt44 + MQ6/5/3/2V2 qt47-50) admit on both gfx11 (HasWmma) and gfx12 (HasWmma) via the shared WMMA dispatch;
        // MQ4C (qt45) remains gfx12-only (HasWmmaGfx12).
        let v2_cases: &[(DType, KernelKey)] = &[
            (DType::MQ6G256V2, KernelKey::GemmMq6G256V2),
            (DType::MQ5G256V2, KernelKey::GemmMq5G256V2),
            (DType::MQ3G256V2, KernelKey::GemmMq3G256V2),
            (DType::MQ2G256V2, KernelKey::GemmMq2G256V2),
            (DType::MQ4G256V2, KernelKey::GemmMq4G256V2),
        ];
        let gfx12_ctx = DispatchCtx::for_test("gfx1200");
        let gfx11_ctx = DispatchCtx::for_test("gfx1100");
        let rdna1_ctx = DispatchCtx::for_test("gfx1010");
        for (dt, exp_key) in v2_cases {
            let fam = GemmFamily::new();
            for (label, ctx) in [("gfx1200", &gfx12_ctx), ("gfx1100", &gfx11_ctx)] {
                let variant = fam
                    .resolve(*dt, ctx, None)
                    .unwrap_or_else(|e| panic!("{label} must admit V2 {dt:?}, got {e:?}"));
                assert_eq!(
                    variant.key, *exp_key,
                    "plain resolve mismatch for {:?} on {label}",
                    dt
                );
                assert_ne!(
                    variant.key,
                    KernelKey::GemmHfq4G256,
                    "V2 must not resolve to HFQ4 for {:?}",
                    dt
                );
            }
            let err = fam.resolve(*dt, &rdna1_ctx, None).unwrap_err();
            assert!(
                matches!(err, DispatchError::MissingImpl { .. }),
                "gfx1010 should reject {:?}, got {:?}",
                dt,
                err
            );
        }
        // MQ4C stays gfx12-only
        let fam = GemmFamily::new();
        let dt = DType::MQ4CG256;
        let exp = KernelKey::GemmMq4CG256;
        let v = fam
            .resolve(dt, &gfx12_ctx, None)
            .expect("gfx1200 must admit MQ4C");
        assert_eq!(v.key, exp);
        assert_ne!(v.key, KernelKey::GemmHfq4G256);
        let err = fam.resolve(dt, &gfx11_ctx, None).unwrap_err();
        assert!(
            matches!(err, DispatchError::MissingImpl { .. }),
            "gfx1100 should reject MQ4C, got {:?}",
            err
        );
    }

    #[test]
    fn v2_residual_run_key_not_hfq4() {
        let gfx12_ctx = DispatchCtx::for_test("gfx1200");
        let gfx11_ctx = DispatchCtx::for_test("gfx1100");
        let fam = GemmFamily::new();
        for key in [
            KernelKey::GemmMq6G256V2Residual,
            KernelKey::GemmMq5G256V2Residual,
            KernelKey::GemmMq3G256V2Residual,
            KernelKey::GemmMq2G256V2Residual,
            KernelKey::GemmMq4G256V2Residual,
        ] {
            assert!(
                fam.registry().resolve(key, &gfx12_ctx, None).is_ok(),
                "residual {:?} should be admitted on gfx1200",
                key
            );
            assert!(
                fam.registry().resolve(key, &gfx11_ctx, None).is_ok(),
                "residual {:?} should be admitted on gfx1100 (HasWmma)",
                key
            );
            assert_ne!(key, KernelKey::GemmHfq4G256Residual);
        }
        // MQ4C remains gfx12-only
        assert!(fam
            .registry()
            .resolve(KernelKey::GemmMq4CG256Residual, &gfx12_ctx, None)
            .is_ok());
        assert!(
            fam.registry()
                .resolve(KernelKey::GemmMq4CG256Residual, &gfx11_ctx, None)
                .is_err(),
            "MQ4C residual should be gfx12-only"
        );
    }
}
