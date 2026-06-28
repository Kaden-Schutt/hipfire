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
use rdna_compute::{DType, Gpu, GpuTensor};

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
        // Calibration tap: when a Hessian/imatrix collector is armed
        // (`gpu.active_capture`) and this weight is a capture target
        // (`gpu.capture_names`, keyed by weight-buffer ptr — see [`Self::buf_ptr`]
        // and `calibration::build_capture_names`), accumulate the input
        // activation. Zero-cost when no collector is armed. nemotron linears
        // bypass the runtime `weight_gemv` chokepoint, so the tap lives here.
        match self {
            LinearWeight::F32(w) => {
                let k = x.shape.iter().product::<usize>().max(1);
                gpu.maybe_capture_activation(w, x, 1, k);
                gpu.gemv_f32(w, x, out)
            }
            LinearWeight::Quant(wt) => {
                // OQ4/OQ8 (Opus Quant) decode: FWHT rotation + AWQ activation
                // scaling live in the shared weight_gemv (which also self-taps for
                // calibration). The MQ4/HFQ/Q8 execute_steps dispatch below does
                // not apply the OQ rotation/AWQ, so OQ takes the shared path
                // (mirrors the OQ4 prefill arm delegating to weight_gemm).
                if matches!(wt.gpu_dtype, DType::Oq4G256 | DType::Oq8G256) {
                    return hipfire_runtime::weights::weight_gemv(gpu, wt, x, out);
                }
                gpu.maybe_capture_activation(&wt.buf, x, 1, wt.k);
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

    /// Weight-buffer pointer — the key the calibration capture map uses; must
    /// match the pointer `maybe_capture_activation` derives in [`Self::gemv`].
    pub(crate) fn buf_ptr(&self) -> usize {
        match self {
            LinearWeight::F32(w) => w.buf.as_ptr() as usize,
            LinearWeight::Quant(wt) => wt.buf.buf.as_ptr() as usize,
        }
    }

    /// Batched prefill matmul: `out[seq, m] = x[seq, k] · Wᵀ` (W stored `[m, k]`,
    /// like the gemv path). F32 uses `gemm_f32_train` with `trans_b`; quantized
    /// weights route through the existing batched GEMM kernels. MQ4G256
    /// pre-rotates the whole `[seq, k]` activation once, then reuses the HFQ4G256
    /// GEMM because MQ4 weights are stored in the same rotated HFQ4 layout.
    pub fn gemm_seq(
        &self,
        gpu: &mut Gpu,
        x: &GpuTensor,
        out: &GpuTensor,
        seq: usize,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        match self {
            LinearWeight::F32(w) => {
                // C[seq,m] = x[seq,k] · Wᵀ ; op(A)=[seq,k] (lda=k), op(B)=Wᵀ=[k,m]
                // from stored W[m,k] (trans_b, ldb=k).
                gpu.gemm_f32_train(x, w, out, seq, m, k, k, k, false, true)
            }
            LinearWeight::Quant(wt) => {
                debug_assert_eq!(wt.m, m);
                debug_assert_eq!(wt.k, k);
                match wt.gpu_dtype {
                    DType::Q8_0 => gpu.gemm_q8_0_batched_chunked(&wt.buf, x, out, m, k, seq),
                    DType::HFQ4G256 => gpu.gemm_hfq4g256(&wt.buf, x, out, m, k, seq),
                    DType::HFQ4G128 => gpu.gemm_hfq4g128(&wt.buf, x, out, m, k, seq),
                    DType::MQ4G256 => {
                        let x_rot = gpu.zeros(&[seq * k], DType::F32)?;
                        let res = (|| {
                            if let Some(awq) = wt.awq_scale.as_ref() {
                                gpu.rotate_x_mq_awq_batched(x, awq, &x_rot, k, seq)?;
                            } else {
                                gpu.rotate_x_mq_batched(x, &x_rot, k, seq)?;
                            }
                            gpu.gemm_hfq4g256(&wt.buf, &x_rot, out, m, k, seq)
                        })();
                        let _ = gpu.free_tensor(x_rot);
                        res
                    }
                    // OQ4 batched prefill (rotation + act-bit/batch heuristics) is
                    // already implemented canonically in the shared weight_gemm.
                    DType::Oq4G256 => hipfire_runtime::weights::weight_gemm(gpu, wt, x, out, seq),
                    other => Err(HipError::unsupported(&format!(
                        "nemotron prefill: no quantized batched gemm for {other:?}"
                    ))),
                }
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
