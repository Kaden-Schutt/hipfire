// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! rocBLAS GEMM fallback wrappers + arch-eligibility helpers. Pure move (Phase 1 M1).

use super::Gpu;
use hip_bridge::{DeviceBuffer, HipResult};
use std::ffi::c_void;
use std::sync::OnceLock;

impl Gpu {
    /// CDNA3-only: prefill GEMM used by `gemm_hfq4g256` rocBLAS path.
    ///
    /// Computes Y_rowmajor[N × M] = X_rowmajor[N × K] · W_transposed, where
    /// the weight is stored row-major [M × K] but the operation needs W^T.
    /// This matches the engine's convention (weight dotted with each row of X
    /// produces one output column per batch row).
    ///
    /// rocBLAS is column-major. A row-major [M × K] matrix is byte-identical
    /// to a column-major [K × M] matrix. So the call is:
    ///   col-major C[M × N] = op_A(W) · X_col[K × N]
    /// with op_A = T (transpose the col-major [K × M] view of W to get [M × K]).
    /// X_row[N × K] viewed col-major is [K × N] with ld=K. Y_row[N × M] viewed
    /// col-major is [M × N] with ld=M — so pointer+ld match C directly.
    pub fn rocblas_gemm_hfq4_prefill(
        &self,
        w_fp16: &DeviceBuffer, // row-major [M × K]
        x_fp16: &DeviceBuffer, // row-major [N × K]
        y_fp32: &DeviceBuffer, // row-major [N × M]
        m: usize,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rocblas_gemm_hfq4_generic(w_fp16, x_fp16, y_fp32, m, n, k, 1.0, 0.0)
    }
    /// Same op as `rocblas_gemm_hfq4_prefill` but with Y += alpha·(X·W^T) +
    /// beta·Y. Covers the residual-GEMM pattern (w_down on LA path, wo on
    /// attention path) where the existing hand-rolled kernels fuse the add.
    pub fn rocblas_gemm_hfq4_prefill_residual(
        &self,
        w_fp16: &DeviceBuffer,
        x_fp16: &DeviceBuffer,
        y_fp32: &DeviceBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rocblas_gemm_hfq4_generic(w_fp16, x_fp16, y_fp32, m, n, k, 1.0, 1.0)
    }
    fn rocblas_gemm_hfq4_generic(
        &self,
        w_fp16: &DeviceBuffer,
        x_fp16: &DeviceBuffer,
        y_fp32: &DeviceBuffer,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> HipResult<()> {
        use hip_bridge::{RocblasDatatype, RocblasOperation};
        let rb = self
            .rocblas
            .as_ref()
            .expect("rocblas_gemm_hfq4: rocBLAS not initialized");
        unsafe {
            rb.gemm_ex(
                RocblasOperation::Transpose,
                RocblasOperation::None,
                m as i32,
                n as i32,
                k as i32,
                &alpha as *const f32 as *const c_void,
                w_fp16.as_ptr(),
                RocblasDatatype::F16,
                k as i32,
                x_fp16.as_ptr(),
                RocblasDatatype::F16,
                k as i32,
                &beta as *const f32 as *const c_void,
                y_fp32.as_ptr(),
                RocblasDatatype::F32,
                m as i32,
                y_fp32.as_ptr(),
                RocblasDatatype::F32,
                m as i32,
                RocblasDatatype::F32,
            )
            .map_err(|e| {
                hip_bridge::HipError::new(e.status, &format!("rocblas_gemm: {}", e.context))
            })
        }
    }
    /// Whether the arch is eligible for the rocBLAS/MFMA batched-prefill
    /// path. Default: CDNA3 only (MI300-series, gfx94x). Override with
    /// `HIPFIRE_ROCBLAS_ALL_ARCHS=1` for local testing on RDNA3+ — rocBLAS
    /// runs fine there (uses WMMA backends on RDNA3, not MFMA) so this is
    /// a useful smoke-path in the absence of an MI300.
    pub(crate) fn rocblas_arch_eligible(&self) -> bool {
        static CACHE: OnceLock<bool> = OnceLock::new();
        let all_archs = *CACHE.get_or_init(|| self.flags.rocblas_all_archs);
        if all_archs {
            return self.rocblas.is_some();
        }
        self.arch_caps.is_cdna3()
    }
    /// Configurable batch threshold for MFMA dispatch. Below this we stay on
    /// the hand-rolled GEMV — rocBLAS launch overhead eats the compute win
    /// at tiny batches. Overridable via `HIPFIRE_ROCBLAS_MIN_BATCH` env var.
    ///
    /// Kill-switch: `HIPFIRE_ROCBLAS_OFF=1` forces the threshold to usize::MAX,
    /// which disables the rocBLAS path entirely for A/B benchmarking against
    /// the hand-rolled GEMV baseline.
    pub(crate) fn rocblas_min_batch(&self) -> usize {
        static CACHE: OnceLock<usize> = OnceLock::new();
        *CACHE.get_or_init(|| {
            if self.flags.rocblas_off {
                return usize::MAX;
            }
            self.flags.rocblas_min_batch.unwrap_or(4)
        })
    }
}
