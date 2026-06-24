// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! A linear weight that is either plain F32 (safetensors path) or a quantized
//! `WeightTensor` (HFQ path), with one `gemv` entry that dispatches correctly
//! for both — so the nemotron block structs share one code path across f32 and
//! mq4/hfq4/q8 weights (FU4). The F32 arm is byte-identical to the prior
//! `gemv_f32` path (keeping the validated forward unchanged); the Quant arm
//! routes through `hipfire_dispatch` `Step::Gemv` with `GemvInput::Raw`, which
//! auto-applies the FWHT rotation for MQ-family dtypes and skips it for
//! HFQ/Q8/F32.

use hip_bridge::{HipError, HipResult};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::{execute_steps, GemvInput, Step};
use hipfire_runtime::weights::WeightTensor;
use rdna_compute::{Gpu, GpuTensor};

/// A `[out, in]` linear weight, plain-f32 or quantized.
pub enum LinearWeight {
    /// Row-major `[out, in]` f32 weight (safetensors path).
    F32(GpuTensor),
    /// Quantized weight (mq4 / hfq4 / q8 …) loaded from an HFQ.
    Quant(Box<WeightTensor>),
}

impl LinearWeight {
    /// `out = W · x`. F32 uses `gemv_f32`; Quant routes through the dispatched
    /// gemv (auto-rotates for MQ-family, plain for HFQ/Q8).
    pub fn gemv(&self, gpu: &mut Gpu, x: &GpuTensor, out: &GpuTensor) -> HipResult<()> {
        match self {
            LinearWeight::F32(w) => gpu.gemv_f32(w, x, out),
            LinearWeight::Quant(wt) => {
                let ctx = DispatchCtx::new(gpu);
                execute_steps(
                    gpu,
                    &ctx,
                    &[Step::Gemv {
                        w: &wt.dispatch_ref(),
                        input: GemvInput::Raw(x),
                        out,
                    }],
                )
                .map_err(|e| HipError::new(0, &format!("nemotron quant gemv: {e}")))
            }
        }
    }

    /// Free the GPU storage (consumes the weight).
    pub fn free(self, gpu: &mut Gpu) {
        match self {
            LinearWeight::F32(w) => {
                let _ = gpu.free_tensor(w);
            }
            LinearWeight::Quant(wt) => wt.free_all(gpu),
        }
    }
}

/// The token embedding table — plain f32 (safetensors) or Q8 (HFQ). Looked up
/// by row, not gemv'd, so it needs its own dispatch.
pub enum EmbeddingTable {
    F32(GpuTensor),
    /// Q8_0 storage (`embedding_lookup_q8` dequantizes the looked-up row).
    Q8(GpuTensor),
}

impl EmbeddingTable {
    /// Copy/dequantize the embedding row for `token` into `out` `[dim]`.
    pub fn lookup(&self, gpu: &mut Gpu, out: &GpuTensor, token: u32, dim: usize) -> HipResult<()> {
        match self {
            EmbeddingTable::F32(t) => gpu.embedding_lookup(t, out, token, dim),
            EmbeddingTable::Q8(t) => gpu.embedding_lookup_q8(t, out, token, dim),
        }
    }

    pub fn free(self, gpu: &mut Gpu) {
        match self {
            EmbeddingTable::F32(t) | EmbeddingTable::Q8(t) => {
                let _ = gpu.free_tensor(t);
            }
        }
    }
}
