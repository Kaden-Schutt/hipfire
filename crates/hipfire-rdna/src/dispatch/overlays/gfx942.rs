// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx942 (CDNA3 / MI300) kernel-dispatch overlays. Phase 2.

use super::super::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    /// gfx942 two-kernel split: rmsnorm_reduce + rotate_with_rms.
    ///
    /// Replaces the single-WG-per-batch fused kernel with two kernels that
    /// each scale better on MI300X's 304 CUs. Kernel A computes rms per
    /// batch (1 WG/batch × 16 wave64s). Kernel B applies rmsnorm + FWHT
    /// per (group, batch) cell (K/256 × batch WGs × 1 wave64 each).
    ///
    /// For batch=256 K=5120: 20×256 = 5120 wave64s on Kernel B vs 1024 on
    /// the fused path → 5× more in-flight waves on prefill.
    ///
    /// Math byte-identical to fused_rmsnorm_mq_rotate.
    pub(crate) fn fused_rmsnorm_rotate_mq_split_gfx942(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "rmsnorm_reduce_gfx942",
            kernels::RMSNORM_REDUCE_GFX942_SRC,
            "rmsnorm_reduce_gfx942",
        )?;
        self.ensure_kernel(
            "rotate_with_rms_gfx942",
            kernels::ROTATE_WITH_RMS_GFX942_SRC,
            "rotate_with_rms_gfx942",
        )?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();

        // Allocate scratch tensor for rms_out (batch_size f32s).
        let rms_tensor = self.alloc_tensor(&[batch_size], DType::F32)?;
        let rms_ptr = rms_tensor.buf.as_ptr();

        // ─── Kernel A: rmsnorm_reduce ────────────────────────────────────
        let xp_a = x.buf.as_ptr();
        let kv_a = k as i32;
        let eps_a = eps;
        let bytes_a = batch_size * k * 4;
        let timer_a =
            crate::profile::begin_timer(&self.hip, "fused", "rmsnorm_reduce_gfx942", bytes_a);
        self.launch_kernargs(
            "rmsnorm_reduce_gfx942",
            [batch_size as u32, 1, 1],
            [1024, 1, 1],
            0,
            &kernargs![ptr xp_a, ptr rms_ptr, i32 kv_a, f32 eps_a],
        )?;
        if let Some(t) = timer_a {
            t.finish(&self.hip);
        }

        // ─── Kernel B: rotate_with_rms ───────────────────────────────────
        let xp_b = x.buf.as_ptr();
        let wp_b = weight.buf.as_ptr();
        let xrp_b = x_rot.buf.as_ptr();
        let s1_b = s1_ptr;
        let s2_b = s2_ptr;
        let kv_b = k as i32;
        let groups = (k / 256) as u32;
        let bytes_b = batch_size * (k * 4 * 3 + 2 * 256 * 4);
        let timer_b =
            crate::profile::begin_timer(&self.hip, "fused", "rotate_with_rms_gfx942", bytes_b);
        let result = self.launch_kernargs(
            "rotate_with_rms_gfx942",
            [groups, batch_size as u32, 1],
            [64, 1, 1],
            0,
            &kernargs![ptr xp_b, ptr wp_b, ptr s1_b, ptr s2_b, ptr rms_ptr, ptr xrp_b, i32 kv_b],
        );
        if let Some(t) = timer_b {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp_b);
        result
    }
    /// Wave64 FP16 hybrid batched HFQ4-G256 GEMM with fused residual add.
    /// Combines wave64 block structure (2 rows/block, full lane utilization) with
    /// FP16 packed arithmetic (__hfma2). Target: gfx906 (MI50) prefill optimization.
    #[allow(clippy::too_many_arguments)]
    /// MFMA-direct HFQ4G256 GEMM with residual add for gfx942 (MI300X CDNA3).
    /// Channel-test verified at max_rel_err = 2e-5 vs FP16 scalar reference.
    pub fn gemm_hfq4g256_residual_mfma_gfx942(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq4g256_residual_mfma_gfx942",
            kernels::GEMM_HFQ4G256_RESIDUAL_MFMA_GFX942_SRC,
            "gemm_hfq4g256_residual_mfma_gfx942",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];
        let grid_x = ((m as u32) + 15) / 16;
        let grid_y = ((batch_size as u32) + 15) / 16;
        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq4g256_residual_mfma_gfx942",
            bytes,
        );
        let result = unsafe {
            self.hip.launch_kernel(
                &self.functions["gemm_hfq4g256_residual_mfma_gfx942"],
                [grid_x, grid_y, 1],
                [64, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn gemm_hfq4g256_residual_mfma_v2_gfx942(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq4g256_residual_mfma_v2_gfx942",
            kernels::GEMM_HFQ4G256_RESIDUAL_MFMA_V2_GFX942_SRC,
            "gemm_hfq4g256_residual_mfma_v2_gfx942",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];
        let grid_x = ((m as u32) + 31) / 32;
        let grid_y = ((batch_size as u32) + 31) / 32;
        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq4g256_residual_mfma_v2_gfx942",
            bytes,
        );
        let result = unsafe {
            self.hip.launch_kernel(
                &self.functions["gemm_hfq4g256_residual_mfma_v2_gfx942"],
                [grid_x, grid_y, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn gemm_hfq4g256_residual_mfma_v3_gfx942(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq4g256_residual_mfma_v3_gfx942",
            kernels::GEMM_HFQ4G256_RESIDUAL_MFMA_V3_GFX942_SRC,
            "gemm_hfq4g256_residual_mfma_v3_gfx942",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];
        let grid_x = ((m as u32) + 31) / 32;
        let grid_y = ((batch_size as u32) + 31) / 32;
        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq4g256_residual_mfma_v3_gfx942",
            bytes,
        );
        let result = unsafe {
            self.hip.launch_kernel(
                &self.functions["gemm_hfq4g256_residual_mfma_v3_gfx942"],
                [grid_x, grid_y, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn gemm_hfq4g256_residual_mfma_v4_gfx942(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq4g256_residual_mfma_v4_gfx942",
            kernels::GEMM_HFQ4G256_RESIDUAL_MFMA_V4_GFX942_SRC,
            "gemm_hfq4g256_residual_mfma_v4_gfx942",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];
        let grid_x = ((m as u32) + 15) / 16;
        let grid_y = ((batch_size as u32) + 63) / 64;
        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq4g256_residual_mfma_v4_gfx942",
            bytes,
        );
        let result = unsafe {
            self.hip.launch_kernel(
                &self.functions["gemm_hfq4g256_residual_mfma_v4_gfx942"],
                [grid_x, grid_y, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
