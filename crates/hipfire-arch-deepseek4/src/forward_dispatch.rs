// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! New-dispatch GEMV wrappers for the DeepSeek V4 forward pass.
//!
//! Replaces the inline `gemv_auto` / `gemv_auto_batched_wmma` dispatch
//! with `GemvFamily::run()` from hipfire-dispatch.

use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::{GemvFamily, WeightRef};
use rdna_compute::{Gpu, GpuTensor};

/// Forward dispatch context: owns the GemvFamily registry and DispatchCtx.
pub struct ForwardDispatch {
    pub gemv: GemvFamily,
    pub ctx: DispatchCtx,
}

impl ForwardDispatch {
    pub fn new(gpu: &Gpu) -> Self {
        Self {
            gemv: GemvFamily::new(),
            ctx: DispatchCtx::new(gpu),
        }
    }

    /// Single-row GEMV dispatch via `GemvFamily::run_auto`.
    ///
    /// Automatically selects `Prerotated` or `Plain` variant based on
    /// the weight's dtype. Caller must pass the correctly rotated (or
    /// plain) `x` — `run_auto` chooses the variant, not the input.
    pub fn dispatch_gemv(
        &self,
        gpu: &mut Gpu,
        weight: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> Result<(), String> {
        let w = WeightRef {
            buf: weight,
            dtype: weight.dtype,
            m,
            k,
        };
        self.gemv
            .run_auto(&self.ctx, gpu, &w, x, y)
            .map_err(|e| format!("gemv dispatch: {e}"))
    }

    /// Batched GEMV dispatch via `GemvFamily::run_auto`.
    ///
    /// Same as `dispatch_gemv` but for batched `x` input. Variant
    /// selection is automatic based on the weight dtype.
    pub fn dispatch_gemv_batched(
        &self,
        gpu: &mut Gpu,
        weight: &GpuTensor,
        x_batch: &GpuTensor,
        y: &GpuTensor,
        _batch_size: usize,
        m: usize,
        k: usize,
    ) -> Result<(), String> {
        let w = WeightRef {
            buf: weight,
            dtype: weight.dtype,
            m,
            k,
        };
        self.gemv
            .run_auto(&self.ctx, gpu, &w, x_batch, y)
            .map_err(|e| format!("gemv batched dispatch: {e}"))
    }
}
