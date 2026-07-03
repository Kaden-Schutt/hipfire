// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Fused QKV and QKV+Z+A projection GEMMs (all dtypes). Pure move (Phase 1 M6).

use super::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{DeviceBuffer, HipResult};
use std::ffi::c_void;
use std::sync::OnceLock;

impl Gpu {
    /// MQ4-Lloyd WMMA fused QKVZA GEMM (LA preamble: qkv + z + beta + alpha).
    /// 4-way fused — one launch covers all four projections of the LA layer.
    /// Phase B1 sibling of `gemm_mq4g256_lloyd_residual_wmma` (kernels-only,
    /// dead-code-safe — wired via the consolidated parity test only; matcher
    /// updates land together with corruption-prevention in Phase B2).
    pub fn gemm_qkvza_mq4g256_lloyd_wmma(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — path selector; concrete launch path binds before HIP use
        // Phase D-B path selector — same gate as residual_mb4.
        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let arch_supports_mb4 = self.arch_caps.is_rdna3()
            && matches!(
                self.arch.as_str(),
                "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151"
            );
        let use_mb4 = match self.flags.lloyd_mb4 {
            None => arch_supports_mb4 && n >= 128 && total_m >= 4096,
            Some(_) => arch_supports_mb4,
        };
        if use_mb4 {
            return self.gemm_qkvza_mq4g256_lloyd_wmma_mb4(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, n,
            );
        }
        self.bind_thread()?;
        let (src, module) = kernels::gemm_qkvza_mq4g256_lloyd_wmma_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_qkvza_mq4g256_lloyd_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_qkv_p = a_qkv.buf.as_ptr();
        let mut a_z_p = a_z.buf.as_ptr();
        let mut a_beta_p = a_beta.buf.as_ptr();
        let mut a_alpha_p = a_alpha.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_qkv_p = y_qkv.buf.as_ptr();
        let mut y_z_p = y_z.buf.as_ptr();
        let mut y_beta_p = y_beta.buf.as_ptr();
        let mut y_alpha_p = y_alpha.buf.as_ptr();
        let mut qkv_m_v = qkv_m as i32;
        let mut z_m_v = z_m as i32;
        let mut beta_m_v = beta_m as i32;
        let mut alpha_m_v = alpha_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_qkv_p as *mut _ as *mut c_void,
            &mut a_z_p as *mut _ as *mut c_void,
            &mut a_beta_p as *mut _ as *mut c_void,
            &mut a_alpha_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_qkv_p as *mut _ as *mut c_void,
            &mut y_z_p as *mut _ as *mut c_void,
            &mut y_beta_p as *mut _ as *mut c_void,
            &mut y_alpha_p as *mut _ as *mut c_void,
            &mut qkv_m_v as *mut _ as *mut c_void,
            &mut z_m_v as *mut _ as *mut c_void,
            &mut beta_m_v as *mut _ as *mut c_void,
            &mut alpha_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 15) / 16;
        let weight_bytes = total_m * (k / 256) * super::LLOYD_MQ4_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_mq4g256_lloyd_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_mq4g256_lloyd_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_qkv_p);
                b.push_ptr(a_z_p);
                b.push_ptr(a_beta_p);
                b.push_ptr(a_alpha_p);
                b.push_ptr(x_p);
                b.push_ptr(y_qkv_p);
                b.push_ptr(y_z_p);
                b.push_ptr(y_beta_p);
                b.push_ptr(y_alpha_p);
                b.push_i32(qkv_m_v);
                b.push_i32(z_m_v);
                b.push_i32(beta_m_v);
                b.push_i32(alpha_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// MQ4-Lloyd WMMA fused QKV GEMM (FullAttention preamble: q + k + v).
    /// 3-way fused. Phase B1 sibling.
    pub fn gemm_qkv_mq4g256_lloyd_wmma(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — path selector; concrete launch path binds before HIP use
        let total_m = q_m + k_m + v_m;
        let arch_supports_mb4 = self.arch_caps.is_rdna3()
            && matches!(
                self.arch.as_str(),
                "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151"
            );
        let use_mb4 = match self.flags.lloyd_mb4 {
            None => arch_supports_mb4 && n >= 128 && total_m >= 4096,
            Some(_) => arch_supports_mb4,
        };
        if use_mb4 {
            return self.gemm_qkv_mq4g256_lloyd_wmma_mb4(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, n,
            );
        }
        self.bind_thread()?;
        let (src, module) = kernels::gemm_qkv_mq4g256_lloyd_wmma_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_qkv_mq4g256_lloyd_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_q_p = a_q.buf.as_ptr();
        let mut a_k_p = a_k.buf.as_ptr();
        let mut a_v_p = a_v.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_q_p = y_q.buf.as_ptr();
        let mut y_k_p = y_k.buf.as_ptr();
        let mut y_v_p = y_v.buf.as_ptr();
        let mut q_m_v = q_m as i32;
        let mut k_m_v = k_m as i32;
        let mut v_m_v = v_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_q_p as *mut _ as *mut c_void,
            &mut a_k_p as *mut _ as *mut c_void,
            &mut a_v_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_q_p as *mut _ as *mut c_void,
            &mut y_k_p as *mut _ as *mut c_void,
            &mut y_v_p as *mut _ as *mut c_void,
            &mut q_m_v as *mut _ as *mut c_void,
            &mut k_m_v as *mut _ as *mut c_void,
            &mut v_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 15) / 16;
        let weight_bytes = total_m * (k / 256) * super::LLOYD_MQ4_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_mq4g256_lloyd_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_mq4g256_lloyd_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_q_p);
                b.push_ptr(a_k_p);
                b.push_ptr(a_v_p);
                b.push_ptr(x_p);
                b.push_ptr(y_q_p);
                b.push_ptr(y_k_p);
                b.push_ptr(y_v_p);
                b.push_i32(q_m_v);
                b.push_i32(k_m_v);
                b.push_i32(v_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Phase D-B: 16×64 fanout sibling of `gemm_qkvza_mq4g256_lloyd_wmma`.
    pub fn gemm_qkvza_mq4g256_lloyd_wmma_mb4(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemm_qkvza_mq4g256_lloyd_wmma_mb4_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_qkvza_mq4g256_lloyd_wmma_mb4")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_qkv_p = a_qkv.buf.as_ptr();
        let mut a_z_p = a_z.buf.as_ptr();
        let mut a_beta_p = a_beta.buf.as_ptr();
        let mut a_alpha_p = a_alpha.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_qkv_p = y_qkv.buf.as_ptr();
        let mut y_z_p = y_z.buf.as_ptr();
        let mut y_beta_p = y_beta.buf.as_ptr();
        let mut y_alpha_p = y_alpha.buf.as_ptr();
        let mut qkv_m_v = qkv_m as i32;
        let mut z_m_v = z_m as i32;
        let mut beta_m_v = beta_m as i32;
        let mut alpha_m_v = alpha_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_qkv_p as *mut _ as *mut c_void,
            &mut a_z_p as *mut _ as *mut c_void,
            &mut a_beta_p as *mut _ as *mut c_void,
            &mut a_alpha_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_qkv_p as *mut _ as *mut c_void,
            &mut y_z_p as *mut _ as *mut c_void,
            &mut y_beta_p as *mut _ as *mut c_void,
            &mut y_alpha_p as *mut _ as *mut c_void,
            &mut qkv_m_v as *mut _ as *mut c_void,
            &mut z_m_v as *mut _ as *mut c_void,
            &mut beta_m_v as *mut _ as *mut c_void,
            &mut alpha_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 63) / 64;
        let weight_bytes = total_m * (k / 256) * super::LLOYD_MQ4_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_qkvza_mq4g256_lloyd_wmma_mb4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_qkvza_mq4g256_lloyd_wmma_mb4",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_qkv_p);
                b.push_ptr(a_z_p);
                b.push_ptr(a_beta_p);
                b.push_ptr(a_alpha_p);
                b.push_ptr(x_p);
                b.push_ptr(y_qkv_p);
                b.push_ptr(y_z_p);
                b.push_ptr(y_beta_p);
                b.push_ptr(y_alpha_p);
                b.push_i32(qkv_m_v);
                b.push_i32(z_m_v);
                b.push_i32(beta_m_v);
                b.push_i32(alpha_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Phase D-B: 16×64 fanout sibling of `gemm_qkv_mq4g256_lloyd_wmma`.
    pub fn gemm_qkv_mq4g256_lloyd_wmma_mb4(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemm_qkv_mq4g256_lloyd_wmma_mb4_for_arch(
            &self.arch_caps,
            self.flags.lloyd_force_baseline,
        );
        self.ensure_kernel(module, src, "gemm_qkv_mq4g256_lloyd_wmma_mb4")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_q_p = a_q.buf.as_ptr();
        let mut a_k_p = a_k.buf.as_ptr();
        let mut a_v_p = a_v.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_q_p = y_q.buf.as_ptr();
        let mut y_k_p = y_k.buf.as_ptr();
        let mut y_v_p = y_v.buf.as_ptr();
        let mut q_m_v = q_m as i32;
        let mut k_m_v = k_m as i32;
        let mut v_m_v = v_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_q_p as *mut _ as *mut c_void,
            &mut a_k_p as *mut _ as *mut c_void,
            &mut a_v_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_q_p as *mut _ as *mut c_void,
            &mut y_k_p as *mut _ as *mut c_void,
            &mut y_v_p as *mut _ as *mut c_void,
            &mut q_m_v as *mut _ as *mut c_void,
            &mut k_m_v as *mut _ as *mut c_void,
            &mut v_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 63) / 64;
        let weight_bytes = total_m * (k / 256) * super::LLOYD_MQ4_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_qkv_mq4g256_lloyd_wmma_mb4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_qkv_mq4g256_lloyd_wmma_mb4",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_q_p);
                b.push_ptr(a_k_p);
                b.push_ptr(a_v_p);
                b.push_ptr(x_p);
                b.push_ptr(y_q_p);
                b.push_ptr(y_k_p);
                b.push_ptr(y_v_p);
                b.push_i32(q_m_v);
                b.push_i32(k_m_v);
                b.push_i32(v_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// MQ3-Lloyd WMMA fused QKVZA GEMM (LA preamble: qkv + z + beta + alpha).
    /// 4-way fused — one launch covers all four projections of the LA layer.
    /// Caller pre-rotates X (FWHT) for MQ3-Lloyd dtype.
    pub fn gemm_qkvza_mq3g256_lloyd_wmma(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let arch_supports_mb4 = self.arch_caps.is_rdna3()
            && !self.arch_caps.is_gfx1152()
            && !self.arch_caps.is_gfx1103();
        let use_mb4 = match self.flags.mq3_mb4 {
            None => arch_supports_mb4 && n >= 128 && total_m >= 4096,
            Some(_) => arch_supports_mb4,
        };
        if use_mb4 {
            return self.gemm_qkvza_mq3g256_lloyd_wmma_mb4(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, n,
            );
        }
        let (src, module) = kernels::gemm_qkvza_mq3g256_lloyd_wmma_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_qkvza_mq3g256_lloyd_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_qkv_p = a_qkv.buf.as_ptr();
        let mut a_z_p = a_z.buf.as_ptr();
        let mut a_beta_p = a_beta.buf.as_ptr();
        let mut a_alpha_p = a_alpha.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_qkv_p = y_qkv.buf.as_ptr();
        let mut y_z_p = y_z.buf.as_ptr();
        let mut y_beta_p = y_beta.buf.as_ptr();
        let mut y_alpha_p = y_alpha.buf.as_ptr();
        let mut qkv_m_v = qkv_m as i32;
        let mut z_m_v = z_m as i32;
        let mut beta_m_v = beta_m as i32;
        let mut alpha_m_v = alpha_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_qkv_p as *mut _ as *mut c_void,
            &mut a_z_p as *mut _ as *mut c_void,
            &mut a_beta_p as *mut _ as *mut c_void,
            &mut a_alpha_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_qkv_p as *mut _ as *mut c_void,
            &mut y_z_p as *mut _ as *mut c_void,
            &mut y_beta_p as *mut _ as *mut c_void,
            &mut y_alpha_p as *mut _ as *mut c_void,
            &mut qkv_m_v as *mut _ as *mut c_void,
            &mut z_m_v as *mut _ as *mut c_void,
            &mut beta_m_v as *mut _ as *mut c_void,
            &mut alpha_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 15) / 16;
        let weight_bytes = total_m * (k / 256) * super::LLOYD_MQ3_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_mq3g256_lloyd_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_mq3g256_lloyd_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_qkv_p);
                b.push_ptr(a_z_p);
                b.push_ptr(a_beta_p);
                b.push_ptr(a_alpha_p);
                b.push_ptr(x_p);
                b.push_ptr(y_qkv_p);
                b.push_ptr(y_z_p);
                b.push_ptr(y_beta_p);
                b.push_ptr(y_alpha_p);
                b.push_i32(qkv_m_v);
                b.push_i32(z_m_v);
                b.push_i32(beta_m_v);
                b.push_i32(alpha_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// MQ3-Lloyd WMMA fused QKV GEMM (FA preamble: q + k + v).
    /// MQ3-Lloyd qkvza mb4 dispatch.
    pub fn gemm_qkvza_mq3g256_lloyd_wmma_mb4(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemm_qkvza_mq3g256_lloyd_wmma_mb4_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_qkvza_mq3g256_lloyd_wmma_mb4")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_qkv_p = a_qkv.buf.as_ptr();
        let mut a_z_p = a_z.buf.as_ptr();
        let mut a_beta_p = a_beta.buf.as_ptr();
        let mut a_alpha_p = a_alpha.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_qkv_p = y_qkv.buf.as_ptr();
        let mut y_z_p = y_z.buf.as_ptr();
        let mut y_beta_p = y_beta.buf.as_ptr();
        let mut y_alpha_p = y_alpha.buf.as_ptr();
        let mut qkv_m_v = qkv_m as i32;
        let mut z_m_v = z_m as i32;
        let mut beta_m_v = beta_m as i32;
        let mut alpha_m_v = alpha_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_qkv_p as *mut _ as *mut c_void,
            &mut a_z_p as *mut _ as *mut c_void,
            &mut a_beta_p as *mut _ as *mut c_void,
            &mut a_alpha_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_qkv_p as *mut _ as *mut c_void,
            &mut y_z_p as *mut _ as *mut c_void,
            &mut y_beta_p as *mut _ as *mut c_void,
            &mut y_alpha_p as *mut _ as *mut c_void,
            &mut qkv_m_v as *mut _ as *mut c_void,
            &mut z_m_v as *mut _ as *mut c_void,
            &mut beta_m_v as *mut _ as *mut c_void,
            &mut alpha_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 63) / 64;
        let weight_bytes = total_m * (k / 256) * super::LLOYD_MQ3_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_qkvza_mq3g256_lloyd_wmma_mb4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_qkvza_mq3g256_lloyd_wmma_mb4",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_qkv_p);
                b.push_ptr(a_z_p);
                b.push_ptr(a_beta_p);
                b.push_ptr(a_alpha_p);
                b.push_ptr(x_p);
                b.push_ptr(y_qkv_p);
                b.push_ptr(y_z_p);
                b.push_ptr(y_beta_p);
                b.push_ptr(y_alpha_p);
                b.push_i32(qkv_m_v);
                b.push_i32(z_m_v);
                b.push_i32(beta_m_v);
                b.push_i32(alpha_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn gemm_qkv_mq3g256_lloyd_wmma(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let total_m = q_m + k_m + v_m;
        let arch_supports_mb4 = self.arch_caps.is_rdna3()
            && !self.arch_caps.is_gfx1152()
            && !self.arch_caps.is_gfx1103();
        let use_mb4 = match self.flags.mq3_mb4 {
            None => arch_supports_mb4 && n >= 128 && total_m >= 4096,
            Some(_) => arch_supports_mb4,
        };
        if use_mb4 {
            return self.gemm_qkv_mq3g256_lloyd_wmma_mb4(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, n,
            );
        }
        let (src, module) = kernels::gemm_qkv_mq3g256_lloyd_wmma_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_qkv_mq3g256_lloyd_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_q_p = a_q.buf.as_ptr();
        let mut a_k_p = a_k.buf.as_ptr();
        let mut a_v_p = a_v.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_q_p = y_q.buf.as_ptr();
        let mut y_k_p = y_k.buf.as_ptr();
        let mut y_v_p = y_v.buf.as_ptr();
        let mut q_m_v = q_m as i32;
        let mut k_m_v = k_m as i32;
        let mut v_m_v = v_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_q_p as *mut _ as *mut c_void,
            &mut a_k_p as *mut _ as *mut c_void,
            &mut a_v_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_q_p as *mut _ as *mut c_void,
            &mut y_k_p as *mut _ as *mut c_void,
            &mut y_v_p as *mut _ as *mut c_void,
            &mut q_m_v as *mut _ as *mut c_void,
            &mut k_m_v as *mut _ as *mut c_void,
            &mut v_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 15) / 16;
        let weight_bytes = total_m * (k / 256) * super::LLOYD_MQ3_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_mq3g256_lloyd_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_mq3g256_lloyd_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_q_p);
                b.push_ptr(a_k_p);
                b.push_ptr(a_v_p);
                b.push_ptr(x_p);
                b.push_ptr(y_q_p);
                b.push_ptr(y_k_p);
                b.push_ptr(y_v_p);
                b.push_i32(q_m_v);
                b.push_i32(k_m_v);
                b.push_i32(v_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// MQ3-Lloyd WMMA fused gate+up GEMM (FFN preamble).
    /// MQ3-Lloyd qkv mb4 dispatch.
    pub fn gemm_qkv_mq3g256_lloyd_wmma_mb4(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemm_qkv_mq3g256_lloyd_wmma_mb4_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_qkv_mq3g256_lloyd_wmma_mb4")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_q_p = a_q.buf.as_ptr();
        let mut a_k_p = a_k.buf.as_ptr();
        let mut a_v_p = a_v.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_q_p = y_q.buf.as_ptr();
        let mut y_k_p = y_k.buf.as_ptr();
        let mut y_v_p = y_v.buf.as_ptr();
        let mut q_m_v = q_m as i32;
        let mut k_m_v = k_m as i32;
        let mut v_m_v = v_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_q_p as *mut _ as *mut c_void,
            &mut a_k_p as *mut _ as *mut c_void,
            &mut a_v_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_q_p as *mut _ as *mut c_void,
            &mut y_k_p as *mut _ as *mut c_void,
            &mut y_v_p as *mut _ as *mut c_void,
            &mut q_m_v as *mut _ as *mut c_void,
            &mut k_m_v as *mut _ as *mut c_void,
            &mut v_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 63) / 64;
        let weight_bytes = total_m * (k / 256) * super::LLOYD_MQ3_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_qkv_mq3g256_lloyd_wmma_mb4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_qkv_mq3g256_lloyd_wmma_mb4",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_q_p);
                b.push_ptr(a_k_p);
                b.push_ptr(a_v_p);
                b.push_ptr(x_p);
                b.push_ptr(y_q_p);
                b.push_ptr(y_k_p);
                b.push_ptr(y_v_p);
                b.push_i32(q_m_v);
                b.push_i32(k_m_v);
                b.push_i32(v_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn gemm_qkvza_hfq4g256(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // CDNA3 MFMA path — 4 back-to-back rocBLAS calls. The last two
        // matrices (beta, alpha) are tiny (n_v_heads = 128 on A3B) so we
        // could skip them and stay on the GEMV path, but dispatching all
        // four via rocBLAS keeps the codepath uniform. Amortizes well.
        if self.rocblas_arch_eligible()
            && batch_size >= self.rocblas_min_batch()
            && self.rocblas.is_some()
            && !self.capture_mode
        {
            let shadow_qkv = self.ensure_fp16_shadow(a_qkv, qkv_m, k)?;
            let shadow_z = self.ensure_fp16_shadow(a_z, z_m, k)?;
            let shadow_beta = self.ensure_fp16_shadow(a_beta, beta_m, k)?;
            let shadow_alpha = self.ensure_fp16_shadow(a_alpha, alpha_m, k)?;
            if let (Some(pq), Some(pz), Some(pb), Some(pa)) =
                (shadow_qkv, shadow_z, shadow_beta, shadow_alpha)
            {
                let x_fp16 = self.ensure_fp16_x(x, batch_size * k)?;
                let xb = unsafe { DeviceBuffer::from_raw(x_fp16, (batch_size * k) * 2) };
                let wq = unsafe { DeviceBuffer::from_raw(pq, (qkv_m * k) * 2) };
                let wz_b = unsafe { DeviceBuffer::from_raw(pz, (z_m * k) * 2) };
                let wb = unsafe { DeviceBuffer::from_raw(pb, (beta_m * k) * 2) };
                let wa = unsafe { DeviceBuffer::from_raw(pa, (alpha_m * k) * 2) };
                let timer = crate::profile::begin_timer(
                    &self.hip,
                    "gemm",
                    "gemm_qkvza_hfq4g256_rocblas",
                    crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                        + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                        + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                        + crate::profile::gemv_hfq4g256_bytes(alpha_m, k),
                );
                let r1 = self.rocblas_gemm_hfq4_prefill(&wq, &xb, &y_qkv.buf, qkv_m, batch_size, k);
                let r2 = if r1.is_ok() {
                    self.rocblas_gemm_hfq4_prefill(&wz_b, &xb, &y_z.buf, z_m, batch_size, k)
                } else {
                    Ok(())
                };
                let r3 = if r2.is_ok() {
                    self.rocblas_gemm_hfq4_prefill(&wb, &xb, &y_beta.buf, beta_m, batch_size, k)
                } else {
                    Ok(())
                };
                let r4 = if r3.is_ok() {
                    self.rocblas_gemm_hfq4_prefill(&wa, &xb, &y_alpha.buf, alpha_m, batch_size, k)
                } else {
                    Ok(())
                };
                std::mem::forget(xb);
                std::mem::forget(wq);
                std::mem::forget(wz_b);
                std::mem::forget(wb);
                std::mem::forget(wa);
                if let Some(t) = timer {
                    t.finish(&self.hip);
                }
                return r1.and(r2).and(r3).and(r4);
            }
        }
        // Fast paths for prefill (batch_size > 1). Disable globally with
        // HIPFIRE_FP16=0 or only for this LA qkvza projection with
        // HIPFIRE_HFQ4_QKVZA_FAST=0.
        if batch_size > 1 && !self.flags.fp16_disabled && self.flags.hfq4_qkvza_fast {
            // Wave64 FP16 hybrid — best of both worlds for gfx906 (MI50).
            if self.arch_caps.is_gcn5_wave64() {
                // gfx906 dp4a MMQ split: qkv + z route through the new MMQ
                // kernel (large-M outputs); beta + alpha keep the fused
                // wave64 kernel because their M (=linear_num_value_heads,
                // typically 32) is far below MMQ_Y=128 — bounds-checked
                // MMQ would waste ~75% of each row-tile.
                //
                // The fused wave64 kernel accepts qkv_m=0, z_m=0 to handle
                // the beta+alpha tail alone (its row-routing logic skips
                // the qkv/z branches when those Ms are zero). See
                // kernels/src/gemm_qkvza_hfq4g256_fp16_wave64.hip:54-61.
                //
                // Routes through MMQ at batch_size ≥ 16 (per
                // should_use_mmq's gfx906 default). Falls through to the
                // fused wave64 if any of qkv/z screening rejects (matches
                // gate_up's behavior in gemm_gate_up_hfq4g256).
                // gfx906 MMQ split — qkv + z through MMQ (large-M), beta + alpha
                // through a fused-projection kernel (tail M typically 32, below
                // MMQ_Y=128). Distinguishes two reasons MMQ might not fire:
                //   (a) batch_size below cutover → fall to dp4a 4-way fused
                //   (b) qkv or z screening rejected → fall to fp16 4-way fused
                //       (screen-reject path preserves higher-precision intent;
                //       dp4a shares Q8_1 quant step that MMQ failed on).
                let mut mmq_screen_rejected = false;
                if self.arch_caps.is_gfx906() && self.arch_caps.should_use_mmq(batch_size) {
                    let qz_safe = if self.mmq_screen {
                        self.mmq_screen_weight(a_qkv, qkv_m, k)
                            && self.mmq_screen_weight(a_z, z_m, k)
                    } else {
                        true
                    };
                    if qz_safe {
                        let xq = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
                        let (r1, r2) = if qkv_m % 128 == 0 && z_m % 128 == 0 {
                            (
                                self.gemm_gate_up_hfq4g256_mmq_gfx906_prequant(
                                    a_qkv, a_z, xq, y_qkv, y_z, qkv_m, z_m, k, batch_size,
                                ),
                                Ok(()),
                            )
                        } else {
                            let r_qkv = self.gemm_hfq4g256_mmq_set_gfx906(
                                a_qkv, xq, y_qkv, qkv_m, k, batch_size,
                            );
                            let r_z = if r_qkv.is_ok() {
                                self.gemm_hfq4g256_mmq_set_gfx906(a_z, xq, y_z, z_m, k, batch_size)
                            } else {
                                Ok(())
                            };
                            (r_qkv, r_z)
                        };
                        // Tail: beta+alpha. Use dp4a-prequant when available
                        // (reuses the Q8_1 scratch we just produced above, no
                        // re-quantize). Falls back to fp16_wave64 in capture
                        // mode (ensure_kernel first-use JIT is unsafe inside
                        // capture; the dp4a kernel may not be compiled yet on
                        // a fresh process).
                        let r3 = if r2.is_ok() {
                            if self.arch_caps.gemv_dp4a_enabled() && !self.capture_mode {
                                self.gemm_qkvza_hfq4g256_wave64_dp4a_prequant(
                                    a_qkv, a_z, a_beta, a_alpha, xq, y_qkv, y_z, y_beta, y_alpha,
                                    0, 0, beta_m, alpha_m, k, batch_size,
                                )
                            } else {
                                self.gemm_qkvza_hfq4g256_fp16_wave64(
                                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, 0,
                                    0, beta_m, alpha_m, k, batch_size,
                                )
                            }
                        } else {
                            Ok(())
                        };
                        return r1.and(r2).and(r3);
                    }
                    mmq_screen_rejected = self.mmq_screen;
                    // qkv or z screening rejected — fall through; screen-reject
                    // path goes to fp16, NOT dp4a.
                }
                // gfx906 dp4a 4-way fused (issue #276 Gap 2). Fires when
                // batch_size > 1 below the MMQ cutover or when capture mode
                // prevents MMQ. Skipped on screen-reject to preserve the
                // higher-precision fallback intent.
                if !mmq_screen_rejected && self.arch_caps.gemv_dp4a_enabled() && !self.capture_mode
                {
                    return self.gemm_qkvza_hfq4g256_wave64_dp4a(
                        a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                        beta_m, alpha_m, k, batch_size,
                    );
                }
                return self.gemm_qkvza_hfq4g256_fp16_wave64(
                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                    beta_m, alpha_m, k, batch_size,
                );
            }
            if self.arch_caps.should_use_mmq(batch_size) {
                let use_mmq = if self.mmq_screen {
                    self.mmq_screen_weight(a_qkv, qkv_m, k)
                        && self.mmq_screen_weight(a_z, z_m, k)
                        && self.mmq_screen_weight(a_beta, beta_m, k)
                        && self.mmq_screen_weight(a_alpha, alpha_m, k)
                } else {
                    true
                };
                if use_mmq {
                    let xq = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
                    let r1 =
                        self.gemm_hfq4g256_mmq_set_prequant(a_qkv, xq, y_qkv, qkv_m, k, batch_size);
                    let r2 = if r1.is_ok() {
                        self.gemm_hfq4g256_mmq_set_prequant(a_z, xq, y_z, z_m, k, batch_size)
                    } else {
                        Ok(())
                    };
                    let r3 = if r2.is_ok() {
                        self.gemm_hfq4g256_mmq_set_prequant(
                            a_beta, xq, y_beta, beta_m, k, batch_size,
                        )
                    } else {
                        Ok(())
                    };
                    let r4 = if r3.is_ok() {
                        self.gemm_hfq4g256_mmq_set_prequant(
                            a_alpha, xq, y_alpha, alpha_m, k, batch_size,
                        )
                    } else {
                        Ok(())
                    };
                    return r1.and(r2).and(r3).and(r4);
                }
            }
            // HFQ4 wave32 MMQ RDNA2 path (issue #299 Phase 4). Three modes:
            //   (a) all 4 Ms aligned to MMQ_Y=128 → single 4-way fused MMQ kernel
            //   (b) qkv_m and z_m aligned but beta_m/alpha_m not (LinearAttention
            //       β+α are typically tiny, well below 128) → split routing:
            //       2-way gate_up MMQ on (wqkv, wz) + 2-way gate_up dot2 on
            //       (w_beta, w_alpha). Mirrors MQ3 phase-2 finding that
            //       gave +22% prefill on Qwen3.5 LA layers.
            //   (c) something else not aligned → fall through to dot2/wmma.
            if self.arch_caps.has_hfq4_mmq() {
                let all_aligned =
                    qkv_m % 128 == 0 && z_m % 128 == 0 && beta_m % 128 == 0 && alpha_m % 128 == 0;
                if all_aligned {
                    return self.gemm_qkvza_hfq4g256_mmq(
                        a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                        beta_m, alpha_m, k, batch_size,
                    );
                }
                if qkv_m % 128 == 0 && z_m % 128 == 0 {
                    let r1 = self.gemm_gate_up_hfq4g256_mmq(
                        a_qkv, a_z, x, y_qkv, y_z, qkv_m, z_m, k, batch_size,
                    );
                    let r2 = if r1.is_ok() {
                        self.gemm_gate_up_hfq4g256_dot2(
                            a_beta, a_alpha, x, y_beta, y_alpha, beta_m, alpha_m, k, batch_size,
                        )
                    } else {
                        Ok(())
                    };
                    return r1.and(r2);
                }
            }
            if self.arch_caps.has_wmma_w32_gfx12() {
                return self.gemm_qkvza_hfq4g256_wmma_gfx12(
                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                    beta_m, alpha_m, k, batch_size,
                );
            }
            if self.arch_caps.has_wmma_w32() {
                return self.gemm_qkvza_hfq4g256_wmma(
                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                    beta_m, alpha_m, k, batch_size,
                );
            }
            // v_dot2_f32_f16 on archs that have it (gfx1011/1012/1030-1032).
            // Excludes gfx1010 (Navi 10, 5700 XT) and gfx1013 (Van Gogh/BC-250 APU).
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_qkvza_hfq4g256_dot2(
                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                    beta_m, alpha_m, k, batch_size,
                );
            }
            // FP16 packed (v_pk_fma_f16) for gfx1010/1013 — 2× scalar FP32.
            return self.gemm_qkvza_hfq4g256_fp16(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            );
        }
        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let (func_name, block, grid_div): (&str, [u32; 3], u32) = if cdna_wave64 {
            self.ensure_kernel(
                "gemm_qkvza_hfq4g256_wave64",
                kernels::GEMM_QKVZA_HFQ4G256_WAVE64_SRC,
                "gemm_qkvza_hfq4g256_wave64",
            )?;
            ("gemm_qkvza_hfq4g256_wave64", [64, 1, 1], 2)
        } else {
            self.ensure_kernel(
                "gemm_qkvza_hfq4g256",
                kernels::GEMM_QKVZA_HFQ4G256_SRC,
                "gemm_qkvza_hfq4g256",
            )?;
            ("gemm_qkvza_hfq4g256", [32, 1, 1], 1)
        };
        let func = &self.functions[func_name];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let grid_x = (total_m + grid_div - 1) / grid_div;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq4g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_tiles as u32, 1],
                block,
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
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq4g256_exact(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let (func_name, block, grid_div): (&str, [u32; 3], u32) = if cdna_wave64 {
            self.ensure_kernel(
                "gemm_qkvza_hfq4g256_wave64",
                kernels::GEMM_QKVZA_HFQ4G256_WAVE64_SRC,
                "gemm_qkvza_hfq4g256_wave64",
            )?;
            ("gemm_qkvza_hfq4g256_wave64", [64, 1, 1], 2)
        } else {
            self.ensure_kernel(
                "gemm_qkvza_hfq4g256",
                kernels::GEMM_QKVZA_HFQ4G256_SRC,
                "gemm_qkvza_hfq4g256",
            )?;
            ("gemm_qkvza_hfq4g256", [32, 1, 1], 1)
        };
        let func = &self.functions[func_name];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let grid_x = (total_m + grid_div - 1) / grid_div;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k);
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq4g256_exact", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_tiles as u32, 1],
                block,
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
    /// Batched 4-way fused HFQ3-G256 GEMM for the LA preamble (MQ3 path).
    ///
    /// HFQ3 sibling of `gemm_qkvza_hfq4g256` — single scalar variant only.
    /// Phase 1 of the gfx10 MQ3 prefill plan. Wires the dense Qwen3.5
    /// LA layer's 4-way fused projection (wqkv + wz + w_beta + w_alpha)
    /// onto the batched path; previously gfx10 MQ3 LA fell back to
    /// per-token forward_scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq3g256(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Phase 3 MMQ (auto-tile-selecting). Default-on for gfx10 sdot4 archs
        // (issue #300 MQ3 gate removal; escape hatch HIPFIRE_HFQ3_MMQ=0).
        // Auto-selector falls back to dot2 at small batch. Layer-gate
        // (HIPFIRE_HFQ3_MMQ_LAYER_{MIN,MAX}) is a no-op when unset; supports
        // per-layer KLD attribution sweeps (#302).
        if batch_size > 1 && self.arch_caps.has_hfq3_mmq() && self.flags.hfq3_mmq_layer_gate_pass()
        {
            return self.gemm_qkvza_hfq3g256_mmq(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            );
        }
        // FP16 fast paths — Phase 2b (dot2) + Phase 2c (fp16 fallback).
        // Layer-aware FP16 gate (#302): falls through to scalar when the
        // current layer falls in HIPFIRE_FP16_LAYER_MIN..=MAX. No-op when
        // those env vars are unset.
        if batch_size > 1 && !self.flags.fp16_disabled_for_current_layer() {
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_qkvza_hfq3g256_dot2(
                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                    beta_m, alpha_m, k, batch_size,
                );
            }
            return self.gemm_qkvza_hfq3g256_fp16(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_qkvza_hfq3g256",
            kernels::GEMM_QKVZA_HFQ3G256_SRC,
            "gemm_qkvza_hfq3g256",
        )?;
        let func = &self.functions["gemm_qkvza_hfq3g256"];

        let mut aqkv = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yqkv = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aqkv as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yqkv as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let bytes = crate::profile::gemm_hfq3g256_bytes(qkv_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(z_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(beta_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(alpha_m, k, batch_size);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq3g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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
    /// v_dot2_f32_f16-accelerated batched 4-way fused HFQ3-G256 GEMM (qkv + z + beta + alpha).
    /// RDNA2 (gfx1011/1012/1030-1032) + RDNA3/4 fast path; HFQ3 sibling of
    /// `gemm_qkvza_hfq4g256_dot2`. Phase 2b of the gfx10 MQ3 prefill plan.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq3g256_dot2(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq3g256_dot2",
            kernels::GEMM_QKVZA_HFQ3G256_DOT2_SRC,
            "gemm_qkvza_hfq3g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkvza_hfq3g256_dot2"];

        let mut aqkv = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yqkv = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aqkv as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yqkv as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let bytes = crate::profile::gemm_hfq3g256_bytes(qkv_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(z_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(beta_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(alpha_m, k, batch_size)
            + batch_size * k * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq3g256_dot2", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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
    /// v_pk_fma_f16-accelerated batched 4-way fused HFQ3-G256 GEMM.
    /// Fallback for archs without the dot extension (gfx1010, gfx1013).
    /// Phase 2c of the gfx10 MQ3 prefill plan.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq3g256_fp16(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq3g256_fp16",
            kernels::GEMM_QKVZA_HFQ3G256_FP16_SRC,
            "gemm_qkvza_hfq3g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkvza_hfq3g256_fp16"];

        let mut aqkv = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yqkv = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aqkv as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yqkv as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let bytes = crate::profile::gemm_hfq3g256_bytes(qkv_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(z_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(beta_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(alpha_m, k, batch_size)
            + batch_size * k * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq3g256_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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
    /// FP16-packed batched 4-way fused HFQ4-G256 GEMM (qkv + z + beta + alpha).
    /// RDNA1/2 fast path — v_pk_fma_f16 inner loop, 2× scalar FP32 throughput.
    /// Requires FP16-converted X (provided via ensure_fp16_x).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq4g256_fp16(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq4g256_fp16",
            kernels::GEMM_QKVZA_HFQ4G256_FP16_SRC,
            "gemm_qkvza_hfq4g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkvza_hfq4g256_fp16"];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (qkv_m + z_m + beta_m + alpha_m) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq4g256_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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
    /// Wave64 FP16 hybrid batched 4-way fused HFQ4-G256 GEMM (qkv + z + beta + alpha).
    /// Combines wave64 block structure (2 rows/block, full lane utilization) with
    /// FP16 packed arithmetic (__hfma2). Target: gfx906 (MI50) prefill optimization.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq4g256_fp16_wave64(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq4g256_fp16_wave64",
            kernels::GEMM_QKVZA_HFQ4G256_FP16_WAVE64_SRC,
            "gemm_qkvza_hfq4g256_fp16_wave64",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkvza_hfq4g256_fp16_wave64"];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let grid_x = (total_m + 1) / 2;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (qkv_m + z_m + beta_m + alpha_m) * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_qkvza_hfq4g256_fp16_wave64",
            bytes,
        );
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_tiles as u32, 1],
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
    /// v_dot2_f32_f16-accelerated batched 4-way fused HFQ4-G256 GEMM (qkv + z + beta + alpha).
    /// RDNA2 (gfx1011/1012/1030-1032) fast path using `amd_mixed_dot`.
    /// One instruction per half2 dot with FP32 accumulation — 1.2-1.5× over FP16 packed.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq4g256_dot2(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq4g256_dot2",
            kernels::GEMM_QKVZA_HFQ4G256_DOT2_SRC,
            "gemm_qkvza_hfq4g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkvza_hfq4g256_dot2"];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
            + batch_size * k * 2
            + batch_size * (qkv_m + z_m + beta_m + alpha_m) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq4g256_dot2", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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
    /// Batched 3-way fused HFQ4-G256 GEMM for the FA preamble.
    ///
    /// Processes N tokens × three projections (wq + wk + wv) in one launch.
    /// Bitwise-identical to calling `fused_qkv_hfq4g256` N times on the same
    /// x[b] — 4-accumulator interleave + pairwise combine preserved per
    /// batch element.
    pub fn gemm_qkv_hfq4g256(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // CDNA3 MFMA path — 3 back-to-back rocBLAS calls for Q, K, V.
        if self.rocblas_arch_eligible()
            && batch_size >= self.rocblas_min_batch()
            && self.rocblas.is_some()
            && !self.capture_mode
        {
            let sq = self.ensure_fp16_shadow(a_q, q_m, k)?;
            let sk = self.ensure_fp16_shadow(a_k, k_m, k)?;
            let sv = self.ensure_fp16_shadow(a_v, v_m, k)?;
            if let (Some(pq), Some(pk), Some(pv)) = (sq, sk, sv) {
                let x_fp16 = self.ensure_fp16_x(x, batch_size * k)?;
                let xb = unsafe { DeviceBuffer::from_raw(x_fp16, (batch_size * k) * 2) };
                let wq = unsafe { DeviceBuffer::from_raw(pq, (q_m * k) * 2) };
                let wk = unsafe { DeviceBuffer::from_raw(pk, (k_m * k) * 2) };
                let wv = unsafe { DeviceBuffer::from_raw(pv, (v_m * k) * 2) };
                let timer = crate::profile::begin_timer(
                    &self.hip,
                    "gemm",
                    "gemm_qkv_hfq4g256_rocblas",
                    crate::profile::gemv_hfq4g256_bytes(q_m, k)
                        + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                        + crate::profile::gemv_hfq4g256_bytes(v_m, k),
                );
                let r1 = self.rocblas_gemm_hfq4_prefill(&wq, &xb, &y_q.buf, q_m, batch_size, k);
                let r2 = if r1.is_ok() {
                    self.rocblas_gemm_hfq4_prefill(&wk, &xb, &y_k.buf, k_m, batch_size, k)
                } else {
                    Ok(())
                };
                let r3 = if r2.is_ok() {
                    self.rocblas_gemm_hfq4_prefill(&wv, &xb, &y_v.buf, v_m, batch_size, k)
                } else {
                    Ok(())
                };
                std::mem::forget(xb);
                std::mem::forget(wq);
                std::mem::forget(wk);
                std::mem::forget(wv);
                if let Some(t) = timer {
                    t.finish(&self.hip);
                }
                return r1.and(r2).and(r3);
            }
        }
        // Fast paths for prefill (batch_size > 1). Disable globally with
        // HIPFIRE_FP16=0 or only for this FA q/k/v projection with
        // HIPFIRE_HFQ4_QKV_FAST=0.
        if batch_size > 1 && !self.flags.fp16_disabled && self.flags.hfq4_qkv_fast {
            // Wave64 FP16 hybrid — best of both worlds for gfx906 (MI50).
            if self.arch_caps.is_gcn5_wave64() {
                // gfx906 dp4a MMQ: route q+k+v through the new MMQ kernel.
                // Unlike qkvza, all three qkv outputs have M well above
                // MMQ_Y=128 (Qwen 9B full-attn: q_m=4096, k_m=v_m=1024),
                // so no tail kernel is needed — straight 3× MMQ-set.
                //
                // Routes through MMQ at batch_size ≥ 16 (per
                // should_use_mmq's gfx906 default). Falls through to the
                // fused wave64 if any of q/k/v screening rejects.
                let mut mmq_screen_rejected = false;
                if self.arch_caps.is_gfx906() && self.arch_caps.should_use_mmq(batch_size) {
                    let qkv_safe = if self.mmq_screen {
                        self.mmq_screen_weight(a_q, q_m, k)
                            && self.mmq_screen_weight(a_k, k_m, k)
                            && self.mmq_screen_weight(a_v, v_m, k)
                    } else {
                        true
                    };
                    if qkv_safe {
                        if q_m % 128 == 0 && k_m % 128 == 0 && v_m % 128 == 0 {
                            return self.gemm_qkv_hfq4g256_mmq_gfx906(
                                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                            );
                        }
                        let xq = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
                        let r1 =
                            self.gemm_hfq4g256_mmq_set_gfx906(a_q, xq, y_q, q_m, k, batch_size);
                        let r2 = if r1.is_ok() {
                            self.gemm_hfq4g256_mmq_set_gfx906(a_k, xq, y_k, k_m, k, batch_size)
                        } else {
                            Ok(())
                        };
                        let r3 = if r2.is_ok() {
                            self.gemm_hfq4g256_mmq_set_gfx906(a_v, xq, y_v, v_m, k, batch_size)
                        } else {
                            Ok(())
                        };
                        return r1.and(r2).and(r3);
                    }
                    mmq_screen_rejected = self.mmq_screen;
                    // q/k/v screening rejected — fall through; screen-reject
                    // path goes to fp16, NOT dp4a (preserves the screen's
                    // higher-precision fallback intent).
                }
                // gfx906 dp4a 3-way fused (issue #276 Gap 2). Fires when
                // batch_size > 1 below the MMQ cutover or in capture mode.
                // Skipped on screen-reject (dp4a shares Q8_1 quant step with
                // MMQ; routing rejected weights here would defeat the screen).
                if !mmq_screen_rejected && self.arch_caps.gemv_dp4a_enabled() && !self.capture_mode
                {
                    return self.gemm_qkv_hfq4g256_wave64_dp4a(
                        a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                    );
                }
                return self.gemm_qkv_hfq4g256_fp16_wave64(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            if self.arch_caps.should_use_mmq(batch_size) {
                let use_mmq = if self.mmq_screen {
                    self.mmq_screen_weight(a_q, q_m, k)
                        && self.mmq_screen_weight(a_k, k_m, k)
                        && self.mmq_screen_weight(a_v, v_m, k)
                } else {
                    true
                };
                if use_mmq {
                    let xq = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
                    let r1 = self.gemm_hfq4g256_mmq_set_prequant(a_q, xq, y_q, q_m, k, batch_size);
                    let r2 = if r1.is_ok() {
                        self.gemm_hfq4g256_mmq_set_prequant(a_k, xq, y_k, k_m, k, batch_size)
                    } else {
                        Ok(())
                    };
                    let r3 = if r2.is_ok() {
                        self.gemm_hfq4g256_mmq_set_prequant(a_v, xq, y_v, v_m, k, batch_size)
                    } else {
                        Ok(())
                    };
                    return r1.and(r2).and(r3);
                }
            }
            // HFQ4 wave32 MMQ RDNA2 path (issue #299 Phase 2). Routes
            // ahead of dot2/wmma fallbacks; default-on for the allowlist
            // arch set (issue #300 gate removal, escape hatch
            // HIPFIRE_HFQ4_MMQ_RDNA2=0). All q_m/k_m/v_m for the Qwen3.5
            // family are MMQ_Y(128)-aligned.
            if self.arch_caps.has_hfq4_mmq() && q_m % 128 == 0 && k_m % 128 == 0 && v_m % 128 == 0 {
                return self.gemm_qkv_hfq4g256_mmq(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            if self.arch_caps.has_wmma_w32_gfx12() {
                return self.gemm_qkv_hfq4g256_wmma_gfx12(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            if self.arch_caps.has_wmma_w32() {
                return self.gemm_qkv_hfq4g256_wmma(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            // v_dot2_f32_f16 on archs that have it (gfx1011/1012/1030-1032).
            // Excludes gfx1010 (Navi 10, 5700 XT) and gfx1013 (Van Gogh/BC-250 APU).
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_qkv_hfq4g256_dot2(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            // FP16 packed (v_pk_fma_f16) for gfx1010/1013 — 2× scalar FP32.
            return self.gemm_qkv_hfq4g256_fp16(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let (func_name, block, grid_div): (&str, [u32; 3], u32) = if cdna_wave64 {
            self.ensure_kernel(
                "gemm_qkv_hfq4g256_wave64",
                kernels::GEMM_QKV_HFQ4G256_WAVE64_SRC,
                "gemm_qkv_hfq4g256_wave64",
            )?;
            ("gemm_qkv_hfq4g256_wave64", [64, 1, 1], 2)
        } else {
            self.ensure_kernel(
                "gemm_qkv_hfq4g256",
                kernels::GEMM_QKV_HFQ4G256_SRC,
                "gemm_qkv_hfq4g256",
            )?;
            ("gemm_qkv_hfq4g256", [32, 1, 1], 1)
        };
        let func = &self.functions[func_name];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (q_m + k_m + v_m) as u32;
        let grid_x = (total_m + grid_div - 1) / grid_div;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_tiles as u32, 1],
                block,
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
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq4g256_exact(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let (func_name, block, grid_div): (&str, [u32; 3], u32) = if cdna_wave64 {
            self.ensure_kernel(
                "gemm_qkv_hfq4g256_wave64",
                kernels::GEMM_QKV_HFQ4G256_WAVE64_SRC,
                "gemm_qkv_hfq4g256_wave64",
            )?;
            ("gemm_qkv_hfq4g256_wave64", [64, 1, 1], 2)
        } else {
            self.ensure_kernel(
                "gemm_qkv_hfq4g256",
                kernels::GEMM_QKV_HFQ4G256_SRC,
                "gemm_qkv_hfq4g256",
            )?;
            ("gemm_qkv_hfq4g256", [32, 1, 1], 1)
        };
        let func = &self.functions[func_name];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (q_m + k_m + v_m) as u32;
        let grid_x = (total_m + grid_div - 1) / grid_div;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k);
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256_exact", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_tiles as u32, 1],
                block,
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
    /// Batched 3-way fused HFQ3-G256 GEMM for the FA preamble (MQ3 path).
    ///
    /// HFQ3 sibling of `gemm_qkv_hfq4g256` — same dispatch shape, 104 B
    /// group stride and 3-bit unpack. Single scalar variant only (no
    /// rocBLAS / wave64 / fp16 / dp4a fast paths yet) — Phase 1 of the
    /// gfx10 MQ3 prefill plan. Bitwise-identical to running the
    /// single-row HFQ3 GEMV N times for N=1.
    pub fn gemm_qkv_hfq3g256(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Phase 3 MMQ family (auto-tile-selecting). Default-on for gfx10
        // sdot4 archs (issue #300, escape hatch HIPFIRE_HFQ3_MMQ=0) when
        // q_m/k_m/v_m are MMQ_Y-aligned. The
        // auto-selector itself falls back to dot2 at batch ≤ 12, so it's
        // safe at any batch_size. Layer-gate is a no-op when unset (#302).
        if batch_size > 1
            && self.arch_caps.has_hfq3_mmq()
            && self.flags.hfq3_mmq_layer_gate_pass()
            && q_m % 128 == 0
            && k_m % 128 == 0
            && v_m % 128 == 0
        {
            return self.gemm_qkv_hfq3g256_mmq(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        // Phase 2 experimental: wave32 dp4a if HIPFIRE_HFQ3_DP4A=1.
        if batch_size > 1 && self.arch_caps.has_hfq3_dp4a() {
            return self.gemm_qkv_hfq3g256_dp4a(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        // FP16 fast paths — gfx10xx admits MQ3 via is_batchable_la, all of
        // these archs support FP16 ISA. Phase 2b (dot2) + Phase 2c (fp16).
        // Layer-aware FP16 gate (#302) falls through to scalar when layer
        // in HIPFIRE_FP16_LAYER_MIN..=MAX. No-op when those vars are unset.
        if batch_size > 1 && !self.flags.fp16_disabled_for_current_layer() {
            // v_dot2_f32_f16 on archs with the dot extension
            // (gfx1011/1012/1030-1032, gfx11/12).
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_qkv_hfq3g256_dot2(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            // v_pk_fma_f16 fallback for gfx1010 (Navi 10 / 5700 XT) and
            // gfx1013 (BC-250 APU), which lack the dot extension but have FP16.
            return self.gemm_qkv_hfq3g256_fp16(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_qkv_hfq3g256",
            kernels::GEMM_QKV_HFQ3G256_SRC,
            "gemm_qkv_hfq3g256",
        )?;
        let func = &self.functions["gemm_qkv_hfq3g256"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (q_m + k_m + v_m) as u32;

        let bytes = crate::profile::gemm_hfq3g256_bytes(q_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(k_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(v_m, k, batch_size);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq3g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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
    /// v_dot2_f32_f16-accelerated batched 3-way fused HFQ3-G256 GEMM (Q + K + V).
    /// RDNA2 (gfx1011/1012/1030-1032) + RDNA3/4 fast path. HFQ3 sibling of
    /// `gemm_qkv_hfq4g256_dot2` — same dispatch shape, FP16 X via
    /// `ensure_fp16_x`, only the weight unpack differs (104 B/group, uint24
    /// byte-combine, 8 3-bit trits per group per thread). Phase 2b of the
    /// gfx10 MQ3 prefill plan.
    pub fn gemm_qkv_hfq3g256_dot2(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq3g256_dot2",
            kernels::GEMM_QKV_HFQ3G256_DOT2_SRC,
            "gemm_qkv_hfq3g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkv_hfq3g256_dot2"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (q_m + k_m + v_m) as u32;

        let bytes = crate::profile::gemm_hfq3g256_bytes(q_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(k_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(v_m, k, batch_size)
            + batch_size * k * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq3g256_dot2", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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
    /// v_pk_fma_f16-accelerated batched 3-way fused HFQ3-G256 GEMM (Q + K + V).
    /// Fallback for archs without the dot extension (gfx1010, gfx1013).
    /// Phase 2c of the gfx10 MQ3 prefill plan.
    pub fn gemm_qkv_hfq3g256_fp16(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq3g256_fp16",
            kernels::GEMM_QKV_HFQ3G256_FP16_SRC,
            "gemm_qkv_hfq3g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkv_hfq3g256_fp16"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (q_m + k_m + v_m) as u32;

        let bytes = crate::profile::gemm_hfq3g256_bytes(q_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(k_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(v_m, k, batch_size)
            + batch_size * k * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq3g256_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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
    /// Wave32+dp4a batched 3-way fused HFQ3-G256 GEMM (Q + K + V).
    /// Phase 2 experimental — port of `gemm_qkv_hfq4g256_wave64_dp4a` from
    /// gfx906 wave64 to wave32 + HFQ3 unpack. Available on the gfx10 sdot4
    /// subset. Gated by `HIPFIRE_HFQ3_DP4A=1`.
    pub fn gemm_qkv_hfq3g256_dp4a(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.has_hfq3_sdot4() {
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_qkv_hfq3g256_dot2(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            return self.gemm_qkv_hfq3g256_fp16(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_qkv_hfq3g256_dp4a",
            kernels::GEMM_QKV_HFQ3G256_DP4A_SRC,
            "gemm_qkv_hfq3g256_dp4a",
        )?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions["gemm_qkv_hfq3g256_dp4a"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 16;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (q_m + k_m + v_m) as u32;

        let bytes = crate::profile::gemm_hfq3g256_bytes(q_m, k, batch_size)
                  + crate::profile::gemm_hfq3g256_bytes(k_m, k, batch_size)
                  + crate::profile::gemm_hfq3g256_bytes(v_m, k, batch_size)
                  + batch_size * k  // Q8_1 mmq X is ~1 byte per element
                  + batch_size * (q_m + k_m + v_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq3g256_dp4a", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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
    /// FP16-packed batched 3-way fused HFQ4-G256 GEMM (Q + K + V).
    /// RDNA1/2 fast path — v_pk_fma_f16 inner loop, 2× scalar FP32 throughput.
    /// Requires FP16-converted X (provided via ensure_fp16_x).
    pub fn gemm_qkv_hfq4g256_fp16(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq4g256_fp16",
            kernels::GEMM_QKV_HFQ4G256_FP16_SRC,
            "gemm_qkv_hfq4g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkv_hfq4g256_fp16"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (q_m + k_m + v_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(v_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (q_m + k_m + v_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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
    /// Wave64 FP16 hybrid batched 3-way fused HFQ4-G256 GEMM (Q + K + V).
    /// Combines wave64 block structure (2 rows/block, full lane utilization) with
    /// FP16 packed arithmetic (__hfma2). Target: gfx906 (MI50) prefill optimization.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq4g256_fp16_wave64(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq4g256_fp16_wave64",
            kernels::GEMM_QKV_HFQ4G256_FP16_WAVE64_SRC,
            "gemm_qkv_hfq4g256_fp16_wave64",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkv_hfq4g256_fp16_wave64"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (q_m + k_m + v_m) as u32;
        let grid_x = (total_m + 1) / 2;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(v_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (q_m + k_m + v_m) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256_fp16_wave64", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_tiles as u32, 1],
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
    /// v_dot2_f32_f16-accelerated batched 3-way fused HFQ4-G256 GEMM (Q + K + V).
    /// RDNA2 (gfx1011/1012/1030-1032) fast path using `amd_mixed_dot`.
    /// One instruction per half2 dot with FP32 accumulation — 1.2-1.5× over FP16 packed.
    pub fn gemm_qkv_hfq4g256_dot2(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq4g256_dot2",
            kernels::GEMM_QKV_HFQ4G256_DOT2_SRC,
            "gemm_qkv_hfq4g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkv_hfq4g256_dot2"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (q_m + k_m + v_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k)
            + batch_size * k * 2
            + batch_size * (q_m + k_m + v_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256_dot2", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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
    /// HFQ3 qkv MMQ auto-selector. Default-on unless `HIPFIRE_HFQ3_MMQ=0`.
    /// CALLER INVARIANT: q_m, k_m, v_m must each be multiples of 128.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq3g256_mmq(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to gemm_qkv_hfq3g256_{dot2,mmq_xN} which bind.
        if !self.arch_caps.has_hfq3_sdot4() {
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_qkv_hfq3g256_dot2(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            return self.gemm_qkv_hfq3g256_fp16(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        if batch_size <= 12 {
            self.gemm_qkv_hfq3g256_dot2(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            )
        } else if batch_size <= 127 {
            self.gemm_qkv_hfq3g256_mmq_x16(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            )
        } else {
            self.gemm_qkv_hfq3g256_mmq_x32(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            )
        }
    }
    /// HFQ3 qkv MMQ at mmq_x=8 (short-prefill tile).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq3g256_mmq_x8(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_qkv_hfq3_mmq_tile(
            a_q,
            a_k,
            a_v,
            x,
            y_q,
            y_k,
            y_v,
            q_m,
            k_m,
            v_m,
            k,
            batch_size,
            8,
            "gemm_qkv_hfq3g256_mmq_x8",
            kernels::GEMM_QKV_HFQ3G256_MMQ_X8_SRC,
        )
    }
    /// HFQ3 qkv MMQ at mmq_x=16 (mid-prefill tile, auto-selector default for 13-127).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq3g256_mmq_x16(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_qkv_hfq3_mmq_tile(
            a_q,
            a_k,
            a_v,
            x,
            y_q,
            y_k,
            y_v,
            q_m,
            k_m,
            v_m,
            k,
            batch_size,
            16,
            "gemm_qkv_hfq3g256_mmq_x16",
            kernels::GEMM_QKV_HFQ3G256_MMQ_X16_SRC,
        )
    }
    /// HFQ3 qkv MMQ at mmq_x=32 (long-prefill tile, b128 LDS).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq3g256_mmq_x32(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_qkv_hfq3_mmq_tile(
            a_q,
            a_k,
            a_v,
            x,
            y_q,
            y_k,
            y_v,
            q_m,
            k_m,
            v_m,
            k,
            batch_size,
            32,
            "gemm_qkv_hfq3g256_mmq_x32",
            kernels::GEMM_QKV_HFQ3G256_MMQ_X32_SRC,
        )
    }
    /// HFQ3 qkvza MMQ auto-selector (wqkv + wz + w_beta + w_alpha). Default-on
    /// unless `HIPFIRE_HFQ3_MMQ=0`. CALLER INVARIANT: qkv_m, z_m, beta_m,
    /// alpha_m must each be multiples of 128.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq3g256_mmq(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to gemm_qkvza_hfq3g256_{dot2,mmq_xN} which bind.
        if !self.arch_caps.has_hfq3_sdot4() {
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_qkvza_hfq3g256_dot2(
                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                    beta_m, alpha_m, k, batch_size,
                );
            }
            return self.gemm_qkvza_hfq3g256_fp16(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            );
        }
        if batch_size <= 12 {
            self.gemm_qkvza_hfq3g256_dot2(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            )
        } else if batch_size <= 127 {
            self.gemm_qkvza_hfq3g256_mmq_x16(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            )
        } else {
            self.gemm_qkvza_hfq3g256_mmq_x32(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            )
        }
    }
    /// HFQ3 qkvza MMQ at mmq_x=8.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq3g256_mmq_x8(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_qkvza_hfq3_mmq_tile(
            a_qkv,
            a_z,
            a_beta,
            a_alpha,
            x,
            y_qkv,
            y_z,
            y_beta,
            y_alpha,
            qkv_m,
            z_m,
            beta_m,
            alpha_m,
            k,
            batch_size,
            8,
            "gemm_qkvza_hfq3g256_mmq_x8",
            kernels::GEMM_QKVZA_HFQ3G256_MMQ_X8_SRC,
        )
    }
    /// HFQ3 qkvza MMQ at mmq_x=16 (auto-selector default for batch 13-127).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq3g256_mmq_x16(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_qkvza_hfq3_mmq_tile(
            a_qkv,
            a_z,
            a_beta,
            a_alpha,
            x,
            y_qkv,
            y_z,
            y_beta,
            y_alpha,
            qkv_m,
            z_m,
            beta_m,
            alpha_m,
            k,
            batch_size,
            16,
            "gemm_qkvza_hfq3g256_mmq_x16",
            kernels::GEMM_QKVZA_HFQ3G256_MMQ_X16_SRC,
        )
    }
    /// HFQ3 qkvza MMQ at mmq_x=32.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq3g256_mmq_x32(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_qkvza_hfq3_mmq_tile(
            a_qkv,
            a_z,
            a_beta,
            a_alpha,
            x,
            y_qkv,
            y_z,
            y_beta,
            y_alpha,
            qkv_m,
            z_m,
            beta_m,
            alpha_m,
            k,
            batch_size,
            32,
            "gemm_qkvza_hfq3g256_mmq_x32",
            kernels::GEMM_QKVZA_HFQ3G256_MMQ_X32_SRC,
        )
    }
    pub fn gemm_qkv_hfq4g256_mmq_x16(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_qkv_hfq4_mmq_tile(
            a_q,
            a_k,
            a_v,
            x,
            y_q,
            y_k,
            y_v,
            q_m,
            k_m,
            v_m,
            k,
            batch_size,
            16,
            "gemm_qkv_hfq4g256_mmq_x16",
            kernels::GEMM_QKV_HFQ4G256_MMQ_X16_SRC,
        )
    }
    pub fn gemm_qkv_hfq4g256_mmq_x32(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_qkv_hfq4_mmq_tile(
            a_q,
            a_k,
            a_v,
            x,
            y_q,
            y_k,
            y_v,
            q_m,
            k_m,
            v_m,
            k,
            batch_size,
            32,
            "gemm_qkv_hfq4g256_mmq_x32",
            kernels::GEMM_QKV_HFQ4G256_MMQ_X32_SRC,
        )
    }
    pub fn gemm_qkv_hfq4g256_mmq(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if batch_size <= 63 {
            self.gemm_qkv_hfq4g256_mmq_x16(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            )
        } else {
            self.gemm_qkv_hfq4g256_mmq_x32(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq4g256_mmq_x16(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_qkvza_hfq4_mmq_tile(
            a_qkv,
            a_z,
            a_beta,
            a_alpha,
            x,
            y_qkv,
            y_z,
            y_beta,
            y_alpha,
            qkv_m,
            z_m,
            beta_m,
            alpha_m,
            k,
            batch_size,
            16,
            "gemm_qkvza_hfq4g256_mmq_x16",
            kernels::GEMM_QKVZA_HFQ4G256_MMQ_X16_SRC,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq4g256_mmq_x32(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_qkvza_hfq4_mmq_tile(
            a_qkv,
            a_z,
            a_beta,
            a_alpha,
            x,
            y_qkv,
            y_z,
            y_beta,
            y_alpha,
            qkv_m,
            z_m,
            beta_m,
            alpha_m,
            k,
            batch_size,
            32,
            "gemm_qkvza_hfq4g256_mmq_x32",
            kernels::GEMM_QKVZA_HFQ4G256_MMQ_X32_SRC,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq4g256_mmq(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if batch_size <= 63 {
            self.gemm_qkvza_hfq4g256_mmq_x16(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            )
        } else {
            self.gemm_qkvza_hfq4g256_mmq_x32(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            )
        }
    }
    /// WMMA-accelerated batched 5-way fused HFQ4-G256 GEMM (qkv + z + beta + alpha).
    /// gfx1100+ only. 16x16 output tiles via wave32 WMMA.
    pub fn gemm_qkvza_hfq4g256_wmma(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq4g256_wmma",
            kernels::GEMM_QKVZA_HFQ4G256_WMMA_SRC,
            "gemm_qkvza_hfq4g256_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq4g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfq4g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(q_m);
                b.push_i32(z_m_val);
                b.push_i32(b_m);
                b.push_i32(a_m);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// HFP4-G32 batched 4-way fused GEMM (qkv + z + beta + alpha) for
    /// the Qwen3.5 DeltaNet LA preamble. Routes gfx11 / gfx12. Used for
    /// HFP4G32 (raw X) and MFP4G32 (FWHT-rotated X handled upstream).
    pub fn gemm_qkvza_hfp4g32(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch_caps.has_wmma_w32_gfx12() {
            return self.gemm_qkvza_hfp4g32_wmma_gfx12(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            );
        }
        self.gemm_qkvza_hfp4g32_wmma(
            a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
            alpha_m, k, batch_size,
        )
    }
    pub fn gemm_qkvza_hfp4g32_wmma(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfp4g32_wmma",
            kernels::GEMM_QKVZA_HFP4G32_WMMA_SRC,
            "gemm_qkvza_hfp4g32_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfp4g32_bytes(qkv_m, k)
            + crate::profile::gemv_hfp4g32_bytes(z_m, k)
            + crate::profile::gemv_hfp4g32_bytes(beta_m, k)
            + crate::profile::gemv_hfp4g32_bytes(alpha_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfp4g32_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfp4g32_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(q_m);
                b.push_i32(z_m_val);
                b.push_i32(b_m);
                b.push_i32(a_m);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// HFQ3-G256 sister of `gemm_qkvza_hfq4g256_wmma`. Same WMMA shape +
    /// lane decomposition; only the inner K-tile unpack differs (3-bit
    /// cross-byte vs 4-bit nibble) and the per-group byte stride is 104
    /// instead of 136. Used for MQ3 prefill via dispatch wrappers that
    /// pre-rotate `x` (see `gemm_qkvza_mq3g256_wmma` below). gfx11 K2
    /// unroll variant — gfx12 K4 to follow once K2 is validated.
    pub fn gemm_qkvza_hfq3g256_wmma(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // HFQ3 mb4 path selector. Only triggers on gfx11; gfx12 keeps its
        // existing fast path (line below) since mb4 sibling not ported.
        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let arch_supports_mb4 = self.arch_caps.is_rdna3()
            && !self.arch_caps.is_gfx1152()
            && !self.arch_caps.is_gfx1103();
        let use_mb4 = match self.flags.mq3_mb4 {
            None => arch_supports_mb4 && batch_size >= 128 && total_m >= 4096,
            Some(_) => arch_supports_mb4,
        };
        if use_mb4 {
            return self.gemm_qkvza_hfq3g256_wmma_mb4(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            );
        }
        if self.arch_caps.has_wmma_w32_gfx12() {
            return self.gemm_qkvza_hfq3g256_wmma_gfx12(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_qkvza_hfq3g256_wmma",
            kernels::GEMM_QKVZA_HFQ3G256_WMMA_SRC,
            "gemm_qkvza_hfq3g256_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        // HFQ3 storage = 104 B/group → ~3.06 bits/weight (vs HFQ4's 4.25).
        let weight_bytes = (qkv_m + z_m + beta_m + alpha_m) * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq3g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfq3g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(q_m);
                b.push_i32(z_m_val);
                b.push_i32(b_m);
                b.push_i32(a_m);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// HFQ3 qkvza mb4 dispatch: 16×64 output tile per WG.
    pub fn gemm_qkvza_hfq3g256_wmma_mb4(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq3g256_wmma_mb4",
            kernels::GEMM_QKVZA_HFQ3G256_WMMA_MB4_SRC,
            "gemm_qkvza_hfq3g256_wmma_mb4",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m_v = qkv_m as i32;
        let mut z_m_v = z_m as i32;
        let mut b_m_v = beta_m as i32;
        let mut a_m_v = alpha_m as i32;
        let mut k_v = k as i32;
        let mut n_v = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m_v as *mut _ as *mut c_void,
            &mut z_m_v as *mut _ as *mut c_void,
            &mut b_m_v as *mut _ as *mut c_void,
            &mut a_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 63) / 64;
        let bytes = total_m * (k / 256) * 104 + batch_size * k * 2 + batch_size * total_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq3g256_wmma_mb4", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfq3g256_wmma_mb4",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(q_m_v);
                b.push_i32(z_m_v);
                b.push_i32(b_m_v);
                b.push_i32(a_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// MQ3 wrapper: rotates `x` via `mq_rotate_x` (FWHT with shared sign
    /// vectors) into the caller-provided `x_rot` scratch, then invokes
    /// `gemm_qkvza_hfq3g256_wmma`. Mirror of `gemm_qkvza_mq4g256_wmma`.
    /// Caller is responsible for `x_rot` being [batch × K] f32 scratch.
    pub fn gemm_qkvza_mq3g256_wmma(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Rotate batched x. mq_rotate_x_batched applies FWHT per-row.
        for b in 0..batch_size {
            let x_row = x.sub_offset(b * k, k);
            let x_rot_row = x_rot.sub_offset(b * k, k);
            self.rotate_x_mq(&x_row, &x_rot_row, k)?;
        }
        // Invalidate the fp16-conversion cache: `x_rot`'s pointer is stable
        // across consecutive MQ3 wrapper calls (same scratch buffer reused
        // per layer), but the underlying data was just rewritten by the
        // rotate loop above. Without this, `ensure_fp16_x` would see the
        // matching `fp16_x_source_ptr` and skip the f32→fp16 conversion,
        // and the kernel would read stale fp16 values from the previous
        // layer's rotation.
        self.fp16_x_source_ptr = std::ptr::null_mut();
        self.gemm_qkvza_hfq3g256_wmma(
            a_qkv, a_z, a_beta, a_alpha, x_rot, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
            alpha_m, k, batch_size,
        )
    }
    /// WMMA-accelerated batched 3-way fused HFQ4-G256 GEMM (Q + K + V).
    /// gfx1100+ only. 16x16 output tiles via wave32 WMMA.
    pub fn gemm_qkv_hfq4g256_wmma(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq4g256_wmma",
            kernels::GEMM_QKV_HFQ4G256_WMMA_SRC,
            "gemm_qkv_hfq4g256_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq4g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// HFQ3-G256 sister of `gemm_qkv_hfq4g256_wmma`. Same WMMA shape +
    /// lane decomposition; only the inner K-tile unpack differs (3-bit
    /// cross-byte vs 4-bit nibble) and the per-group byte stride is 104
    /// instead of 136. Used for MQ3 prefill via dispatch sites in
    /// qwen35.rs FullAttention branch (X is pre-rotated upstream).
    pub fn gemm_qkv_hfq3g256_wmma(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let total_m = q_m + k_m + v_m;
        let arch_supports_mb4 = self.arch_caps.is_rdna3()
            && !self.arch_caps.is_gfx1152()
            && !self.arch_caps.is_gfx1103();
        let use_mb4 = match self.flags.mq3_mb4 {
            None => arch_supports_mb4 && batch_size >= 128 && total_m >= 4096,
            Some(_) => arch_supports_mb4,
        };
        if use_mb4 {
            return self.gemm_qkv_hfq3g256_wmma_mb4(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        if self.arch_caps.has_wmma_w32_gfx12() {
            return self.gemm_qkv_hfq3g256_wmma_gfx12(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_qkv_hfq3g256_wmma",
            kernels::GEMM_QKV_HFQ3G256_WMMA_SRC,
            "gemm_qkv_hfq3g256_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let weight_bytes = (q_m + k_m + v_m) * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq3g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq3g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// HFQ3 qkv mb4 dispatch: 16×64 output tile per WG.
    pub fn gemm_qkv_hfq3g256_wmma_mb4(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq3g256_wmma_mb4",
            kernels::GEMM_QKV_HFQ3G256_WMMA_MB4_SRC,
            "gemm_qkv_hfq3g256_wmma_mb4",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_v = q_m as i32;
        let mut k_m_v = k_m as i32;
        let mut v_m_v = v_m as i32;
        let mut k_v = k as i32;
        let mut n_v = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_v as *mut _ as *mut c_void,
            &mut k_m_v as *mut _ as *mut c_void,
            &mut v_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 63) / 64;
        let bytes = total_m * (k / 256) * 104 + batch_size * k * 2 + batch_size * total_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq3g256_wmma_mb4", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq3g256_wmma_mb4",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_v);
                b.push_i32(k_m_v);
                b.push_i32(v_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// WMMA-accelerated batched 3-way fused HFP4-G32 GEMM (Q + K + V).
    /// Sister of `gemm_qkv_hfq4g256_wmma` for the FP4 (E2M1 + UE8M0 g32 +
    /// FP16 row scale) family. Routes to the gfx11 or gfx12 variant by
    /// arch. Asserts a WMMA-capable arch — callers must gate via
    /// `is_batchable_la` (which restricts HFP4G32 to gfx11+/gfx12 archs).
    ///
    /// Used for both HFP4G32 (raw, X is the rmsnormed activation) and
    /// MFP4G32 (X is the FWHT-rotated activation; rotation happens
    /// upstream via `mq_rotate_x` so the kernel is identical).
    pub fn gemm_qkv_hfp4g32(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // FP8 WMMA gate: only at batch sizes where the prefill bench
        // measured ≥1× vs FP16 WMMA. At small batches (decode FA QKV
        // calls this with batch_size=1) the FP8 path measures
        // 0.71-0.84×, so we keep the FP16 path there. Threshold is
        // conservative — see project_fp8_wmma_hfp4g32_2026_05_10.md
        // for the full N sweep. The decode-path FP8 win is on the
        // GEMV side (gemv_hfp4g32_fp8_gfx12), not WMMA.
        if self.arch_caps.has_wmma_w32_gfx12()
            && self.flags.fp8_wmma
            && batch_size >= super::FP8_WMMA_MIN_BATCH
        {
            return self.gemm_qkv_hfp4g32_wmma_fp8_gfx12(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        if self.arch_caps.has_wmma_w32_gfx12() {
            return self.gemm_qkv_hfp4g32_wmma_gfx12(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        self.gemm_qkv_hfp4g32_wmma(
            a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
        )
    }
    /// gfx11 (RDNA3) variant of `gemm_qkv_hfp4g32`. Direct entry point
    /// for tests; production callers should use `gemm_qkv_hfp4g32` to
    /// pick up the gfx12 sister automatically.
    pub fn gemm_qkv_hfp4g32_wmma(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfp4g32_wmma",
            kernels::GEMM_QKV_HFP4G32_WMMA_SRC,
            "gemm_qkv_hfp4g32_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfp4g32_bytes(q_m, k)
            + crate::profile::gemv_hfp4g32_bytes(k_m, k)
            + crate::profile::gemv_hfp4g32_bytes(v_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfp4g32_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfp4g32_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Batched HFQ4-G256 fused 4-way QKVZA (qkv + z + beta + alpha) GEMM
    /// with dp4a inner loop on gfx906. HFQ4 sibling of
    /// `gemm_qkvza_hfq6g256_wave64_dp4a` (HFQ6 Phase A.3, merged via #187).
    /// Closes the dispatch fallthrough where MQ4 at gfx906 batched DeltaNet
    /// preamble (B>1) drops to `gemm_qkvza_hfq4g256_fp16_wave64`. Issue #276
    /// Gap 2 part 2. Uses `BATCH_TILE=16` matching the kernel's `#define`.
    pub fn gemm_qkvza_hfq4g256_wave64_dp4a(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        self.gemm_qkvza_hfq4g256_wave64_dp4a_prequant(
            a_qkv, a_z, a_beta, a_alpha, xq_ptr, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
            alpha_m, k, batch_size,
        )
    }
    /// Prequant entry point: caller has already populated the Q8_1 scratch.
    /// Skips the Q8_1 conversion. Use when X has just been quantized for a
    /// sibling kernel (e.g. the MMQ-split qkv+z path's beta+alpha tail) to
    /// avoid a redundant FP32→Q8_1 conversion of the entire X tensor.
    pub fn gemm_qkvza_hfq4g256_wave64_dp4a_prequant(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        xq_ptr: *mut c_void,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq4g256_wave64_dp4a",
            kernels::GEMM_QKVZA_HFQ4G256_WAVE64_DP4A_SRC,
            "gemm_qkvza_hfq4g256_wave64_dp4a",
        )?;

        let aq = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let yq = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let qkv_m_val = qkv_m as i32;
        let z_m_val = z_m as i32;
        let beta_m_val = beta_m as i32;
        let alpha_m_val = alpha_m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &az as *const _ as *mut c_void,
            &ab as *const _ as *mut c_void,
            &aa as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yz as *const _ as *mut c_void,
            &yb as *const _ as *mut c_void,
            &ya as *const _ as *mut c_void,
            &qkv_m_val as *const _ as *mut c_void,
            &z_m_val as *const _ as *mut c_void,
            &beta_m_val as *const _ as *mut c_void,
            &alpha_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        // BATCH_TILE MUST match the kernel's `#define BATCH_TILE 16`.
        const BATCH_TILE: usize = 16;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let grid_x = (total_m + 1) / 2;

        // bytes = weight (4 matrices, 136 B/group each) + Q8_1 X read +
        // 4× Y writes (overwrite semantic, no read).
        let bytes = crate::profile::hfq4g256_weight_bytes(qkv_m, k)
            + crate::profile::hfq4g256_weight_bytes(z_m, k)
            + crate::profile::hfq4g256_weight_bytes(beta_m, k)
            + crate::profile::hfq4g256_weight_bytes(alpha_m, k)
            + batch_size * k
            + batch_size * (qkv_m + z_m + beta_m + alpha_m) * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_qkvza_hfq4g256_wave64_dp4a",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfq4g256_wave64_dp4a",
            [grid_x, batch_tiles as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xq);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(qkv_m_val);
                b.push_i32(z_m_val);
                b.push_i32(beta_m_val);
                b.push_i32(alpha_m_val);
                b.push_i32(k_val);
                b.push_i32(bs_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Batched HFQ4-G256 fused 3-way QKV GEMM with dp4a inner loop on
    /// gfx906. HFQ4 sibling of `gemm_qkv_hfq6g256_wave64_dp4a`. Closes the
    /// dispatch fallthrough where MQ4 at gfx906 batched FullAttention
    /// preamble drops to `gemm_qkv_hfq4g256_fp16_wave64`. Issue #276 Gap 2.
    pub fn gemm_qkv_hfq4g256_wave64_dp4a(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        self.gemm_qkv_hfq4g256_wave64_dp4a_prequant(
            a_q, a_k, a_v, xq_ptr, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
        )
    }
    /// Prequant entry point — see `gemm_qkvza_hfq4g256_wave64_dp4a_prequant`
    /// for rationale. Skips the FP32→Q8_1 conversion of X; caller must have
    /// populated the Q8_1 scratch beforehand.
    pub fn gemm_qkv_hfq4g256_wave64_dp4a_prequant(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        xq_ptr: *mut c_void,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq4g256_wave64_dp4a",
            kernels::GEMM_QKV_HFQ4G256_WAVE64_DP4A_SRC,
            "gemm_qkv_hfq4g256_wave64_dp4a",
        )?;

        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_val = q_m as i32;
        let k_m_val = k_m as i32;
        let v_m_val = v_m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &ak as *const _ as *mut c_void,
            &av as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yk as *const _ as *mut c_void,
            &yv as *const _ as *mut c_void,
            &q_m_val as *const _ as *mut c_void,
            &k_m_val as *const _ as *mut c_void,
            &v_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 16;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (q_m + k_m + v_m) as u32;
        let grid_x = (total_m + 1) / 2;

        let bytes = crate::profile::hfq4g256_weight_bytes(q_m, k)
            + crate::profile::hfq4g256_weight_bytes(k_m, k)
            + crate::profile::hfq4g256_weight_bytes(v_m, k)
            + batch_size * k
            + batch_size * (q_m + k_m + v_m) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256_wave64_dp4a", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq4g256_wave64_dp4a",
            [grid_x, batch_tiles as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xq);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b.push_i32(bs_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Batched 4-way fused HFQ6-G256 GEMM (qkv + z + beta + alpha).
    /// Auto-selects: gfx11 -> WMMA, else -> scalar.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq6g256(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if batch_size > 1 && !self.flags.fp16_disabled {
            if self.arch_caps.has_wmma_w32_gfx12() {
                return self.gemm_qkvza_hfq6g256_wmma_gfx12(
                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                    beta_m, alpha_m, k, batch_size,
                );
            }
            static HFQ6_QKVZA_4W: OnceLock<bool> = OnceLock::new();
            let hfq6_qkvza_4w = *HFQ6_QKVZA_4W.get_or_init(|| {
                !matches!(
                    std::env::var("HIPFIRE_HFQ6_QKVZA_4W").ok().as_deref(),
                    Some("0" | "off" | "false" | "no")
                )
            });
            if hfq6_qkvza_4w && self.arch == "gfx1151" && batch_size % 64 == 0 && batch_size >= 128
            {
                return self.gemm_qkvza_hfq6g256_wmma_4w_gfx1151(
                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                    beta_m, alpha_m, k, batch_size,
                );
            }
            if self.arch_caps.has_wmma_w32() {
                return self.gemm_qkvza_hfq6g256_wmma(
                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                    beta_m, alpha_m, k, batch_size,
                );
            }
            // gfx906: wave64+dp4a batched fused (Phase A.3, plan v3.2.3 §5.1
            // item 3). Pre-quantize x to Q8_1 and dispatch the dp4a kernel.
            // Skip in capture mode (Q8_1 quantize launch must be reachable
            // from captured graph or pre-baked) — matches HFQ4 sibling pattern.
            if self.arch_caps.gemv_dp4a_enabled() && !self.capture_mode {
                return self.gemm_qkvza_hfq6g256_wave64_dp4a(
                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                    beta_m, alpha_m, k, batch_size,
                );
            }
            // v_dot2_f32_f16 on archs that have it (gfx1011/1012/1030-1032).
            // Excludes gfx1010 (Navi 10, 5700 XT) and gfx1013 (Van Gogh/BC-250 APU).
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_qkvza_hfq6g256_dot2(
                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                    beta_m, alpha_m, k, batch_size,
                );
            }
            // FP16 packed (v_pk_fma_f16) for gfx1010/1013 — 2× scalar FP32.
            return self.gemm_qkvza_hfq6g256_fp16(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_qkvza_hfq6g256",
            kernels::GEMM_QKVZA_HFQ6G256_SRC,
            "gemm_qkvza_hfq6g256",
        )?;
        let func = &self.functions["gemm_qkvza_hfq6g256"];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq6g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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
    /// FP16-packed batched 4-way fused HFQ6-G256 GEMM (qkv + z + beta + alpha).
    /// RDNA1/2 fast path — v_pk_fma_f16 inner loop, 2× scalar FP32 throughput.
    /// Requires FP16-converted X (provided via ensure_fp16_x).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq6g256_fp16(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq6g256_fp16",
            kernels::GEMM_QKVZA_HFQ6G256_FP16_SRC,
            "gemm_qkvza_hfq6g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkvza_hfq6g256_fp16"];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (qkv_m + z_m + beta_m + alpha_m) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq6g256_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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
    /// v_dot2_f32_f16-accelerated batched 4-way fused HFQ6-G256 GEMM (qkv + z + beta + alpha).
    /// RDNA2 (gfx1011/1012/1030-1032) fast path using `amd_mixed_dot`.
    /// One instruction per half2 dot with FP32 accumulation — 1.2-1.5× over FP16 packed.
    #[allow(clippy::too_many_arguments)]
    /// gfx906 wave64+dp4a batched 4-way fused QKVZA GEMM. Phase A.3
    /// (plan v3.2.3 §5.1 item 3). Uses Q8_1 activation pre-quantize
    /// (shared with A.1c GEMV-shape dp4a kernels) and HFQ6 6-bit
    /// unsigned weight unpack.
    pub fn gemm_qkvza_hfq6g256_wave64_dp4a(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;

        self.ensure_kernel(
            "gemm_qkvza_hfq6g256_wave64_dp4a",
            kernels::GEMM_QKVZA_HFQ6G256_WAVE64_DP4A_SRC,
            "gemm_qkvza_hfq6g256_wave64_dp4a",
        )?;

        let aq = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let yq = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let qkv_m_val = qkv_m as i32;
        let z_m_val = z_m as i32;
        let beta_m_val = beta_m as i32;
        let alpha_m_val = alpha_m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &az as *const _ as *mut c_void,
            &ab as *const _ as *mut c_void,
            &aa as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yz as *const _ as *mut c_void,
            &yb as *const _ as *mut c_void,
            &ya as *const _ as *mut c_void,
            &qkv_m_val as *const _ as *mut c_void,
            &z_m_val as *const _ as *mut c_void,
            &beta_m_val as *const _ as *mut c_void,
            &alpha_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 8;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let grid_x = (total_m + 1) / 2;

        self.launch_maybe_blob(
            "gemm_qkvza_hfq6g256_wave64_dp4a",
            [grid_x, batch_tiles as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xq);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(qkv_m_val);
                b.push_i32(z_m_val);
                b.push_i32(beta_m_val);
                b.push_i32(alpha_m_val);
                b.push_i32(k_val);
                b.push_i32(bs_val);
                b
            },
        )
    }
    pub fn gemm_qkvza_hfq6g256_dot2(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq6g256_dot2",
            kernels::GEMM_QKVZA_HFQ6G256_DOT2_SRC,
            "gemm_qkvza_hfq6g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkvza_hfq6g256_dot2"];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;

        unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// WMMA-accelerated batched 4-way fused HFQ6-G256 GEMM (qkv + z + beta + alpha).
    /// gfx1100+ only. 16x16 output tiles via wave32 WMMA.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq6g256_wmma(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq6g256_wmma",
            kernels::GEMM_QKVZA_HFQ6G256_WMMA_SRC,
            "gemm_qkvza_hfq6g256_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq6g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfq6g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(q_m);
                b.push_i32(z_m_val);
                b.push_i32(b_m);
                b.push_i32(a_m);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Batched 3-way fused HFQ6-G256 GEMM for the FA preamble (Q + K + V).
    /// Auto-selects: gfx11 -> WMMA, else -> scalar.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq6g256(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if batch_size > 1 && !self.flags.fp16_disabled {
            if self.arch_caps.has_wmma_w32_gfx12() {
                return self.gemm_qkv_hfq6g256_wmma_gfx12(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            static HFQ6_QKV_4W: OnceLock<bool> = OnceLock::new();
            let hfq6_qkv_4w = *HFQ6_QKV_4W.get_or_init(|| {
                !matches!(
                    std::env::var("HIPFIRE_HFQ6_QKV_4W").ok().as_deref(),
                    Some("0" | "off" | "false" | "no")
                )
            });
            if hfq6_qkv_4w && self.arch == "gfx1151" && batch_size % 64 == 0 && batch_size >= 128 {
                return self.gemm_qkv_hfq6g256_wmma_4w_gfx1151(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            if self.arch_caps.has_wmma_w32() {
                return self.gemm_qkv_hfq6g256_wmma(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            // gfx906: wave64+dp4a batched fused (Phase A.3).
            // Skip in capture mode (Q8_1 quantize) — matches HFQ4 sibling.
            if self.arch_caps.gemv_dp4a_enabled() && !self.capture_mode {
                return self.gemm_qkv_hfq6g256_wave64_dp4a(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            // v_dot2_f32_f16 on archs that have it (gfx1011/1012/1030-1032).
            // Excludes gfx1010 (Navi 10, 5700 XT) and gfx1013 (Van Gogh/BC-250 APU).
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_qkv_hfq6g256_dot2(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            // FP16 packed (v_pk_fma_f16) for gfx1010/1013 — 2× scalar FP32.
            return self.gemm_qkv_hfq6g256_fp16(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_qkv_hfq6g256",
            kernels::GEMM_QKV_HFQ6G256_SRC,
            "gemm_qkv_hfq6g256",
        )?;
        let func = &self.functions["gemm_qkv_hfq6g256"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (q_m + k_m + v_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq6g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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
    /// FP16-packed batched 3-way fused HFQ6-G256 GEMM (Q + K + V).
    /// RDNA1/2 fast path — v_pk_fma_f16 inner loop, 2× scalar FP32 throughput.
    /// Requires FP16-converted X (provided via ensure_fp16_x).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq6g256_fp16(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq6g256_fp16",
            kernels::GEMM_QKV_HFQ6G256_FP16_SRC,
            "gemm_qkv_hfq6g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkv_hfq6g256_fp16"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (q_m + k_m + v_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(v_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (q_m + k_m + v_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq6g256_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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
    /// v_dot2_f32_f16-accelerated batched 3-way fused HFQ6-G256 GEMM (Q + K + V).
    /// RDNA2 (gfx1011/1012/1030-1032) fast path using `amd_mixed_dot`.
    /// One instruction per half2 dot with FP32 accumulation — 1.2-1.5× over FP16 packed.
    #[allow(clippy::too_many_arguments)]
    /// gfx906 wave64+dp4a batched 3-way fused QKV GEMM. Phase A.3
    /// (plan v3.2.3 §5.1 item 3). Sibling of qkvza_wave64_dp4a.
    pub fn gemm_qkv_hfq6g256_wave64_dp4a(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;

        self.ensure_kernel(
            "gemm_qkv_hfq6g256_wave64_dp4a",
            kernels::GEMM_QKV_HFQ6G256_WAVE64_DP4A_SRC,
            "gemm_qkv_hfq6g256_wave64_dp4a",
        )?;

        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_val = q_m as i32;
        let k_m_val = k_m as i32;
        let v_m_val = v_m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &ak as *const _ as *mut c_void,
            &av as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yk as *const _ as *mut c_void,
            &yv as *const _ as *mut c_void,
            &q_m_val as *const _ as *mut c_void,
            &k_m_val as *const _ as *mut c_void,
            &v_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 8;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (q_m + k_m + v_m) as u32;
        let grid_x = (total_m + 1) / 2;

        self.launch_maybe_blob(
            "gemm_qkv_hfq6g256_wave64_dp4a",
            [grid_x, batch_tiles as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xq);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b.push_i32(bs_val);
                b
            },
        )
    }
    pub fn gemm_qkv_hfq6g256_dot2(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq6g256_dot2",
            kernels::GEMM_QKV_HFQ6G256_DOT2_SRC,
            "gemm_qkv_hfq6g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkv_hfq6g256_dot2"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (q_m + k_m + v_m) as u32;

        unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// WMMA-accelerated batched 3-way fused HFQ6-G256 GEMM (Q + K + V).
    /// gfx1100+ only. 16x16 output tiles via wave32 WMMA.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq6g256_wmma(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq6g256_wmma",
            kernels::GEMM_QKV_HFQ6G256_WMMA_SRC,
            "gemm_qkv_hfq6g256_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq6g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq6g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// WMMA 4-way fused Q8_0 GEMM (wqkv + wz + w_beta + w_alpha).
    /// DeltaNet LA preamble. Auto-routes to gfx12 sibling on RDNA4.
    pub fn gemm_qkvza_q8_0_wmma(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch_caps.is_rdna4() {
            return self.gemm_qkvza_q8_0_wmma_gfx12(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            );
        }
        // Q8_0 packs 32 elements per block (34 bytes); the kernel iterates
        // `K/32` blocks per row and silently drops any tail if K is not a
        // multiple of 32. All current production shapes satisfy this; guard
        // here to catch future shape regressions before they corrupt output.
        debug_assert_eq!(
            k % 32,
            0,
            "gemm_qkvza_q8_0_wmma: K must be a multiple of 32 (got K={k})"
        );
        static Q8_QKVZA_4W: OnceLock<Option<bool>> = OnceLock::new();
        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let auto_q8_qkvza_4w =
            batch_size >= 128 || (batch_size >= 64 && k == 3072 && total_m >= 8192);
        let q8_qkvza_4w = Self::gfx1151_q8_4w_enabled(
            *Q8_QKVZA_4W.get_or_init(|| Self::q8_4w_mode("HIPFIRE_Q8_QKVZA_4W")),
            auto_q8_qkvza_4w,
        );
        if q8_qkvza_4w && self.arch == "gfx1151" && batch_size % 64 == 0 {
            return self.gemm_qkvza_q8_0_wmma_4w_gfx1151(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_qkvza_q8_0_wmma",
            kernels::GEMM_QKVZA_Q8_0_WMMA_SRC,
            "gemm_qkvza_q8_0_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_qkv_p = a_qkv.buf.as_ptr();
        let mut a_z_p = a_z.buf.as_ptr();
        let mut a_beta_p = a_beta.buf.as_ptr();
        let mut a_alpha_p = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut y_qkv_p = y_qkv.buf.as_ptr();
        let mut y_z_p = y_z.buf.as_ptr();
        let mut y_beta_p = y_beta.buf.as_ptr();
        let mut y_alpha_p = y_alpha.buf.as_ptr();
        let mut qkv_m_val = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut beta_m_val = beta_m as i32;
        let mut alpha_m_val = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_qkv_p as *mut _ as *mut c_void,
            &mut a_z_p as *mut _ as *mut c_void,
            &mut a_beta_p as *mut _ as *mut c_void,
            &mut a_alpha_p as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut y_qkv_p as *mut _ as *mut c_void,
            &mut y_z_p as *mut _ as *mut c_void,
            &mut y_beta_p as *mut _ as *mut c_void,
            &mut y_alpha_p as *mut _ as *mut c_void,
            &mut qkv_m_val as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut beta_m_val as *mut _ as *mut c_void,
            &mut alpha_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;
        let q8_bytes = |m: usize| m * (k / 32) * 34;
        let bytes = q8_bytes(qkv_m)
            + q8_bytes(z_m)
            + q8_bytes(beta_m)
            + q8_bytes(alpha_m)
            + batch_size * k * 2
            + batch_size * total_m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_q8_0_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_q8_0_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_qkv_p);
                b.push_ptr(a_z_p);
                b.push_ptr(a_beta_p);
                b.push_ptr(a_alpha_p);
                b.push_ptr(xp);
                b.push_ptr(y_qkv_p);
                b.push_ptr(y_z_p);
                b.push_ptr(y_beta_p);
                b.push_ptr(y_alpha_p);
                b.push_i32(qkv_m_val);
                b.push_i32(z_m_val);
                b.push_i32(beta_m_val);
                b.push_i32(alpha_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// WMMA-accelerated batched 3-way fused Q8_0 GEMM (Q + K + V projections).
    /// Auto-routes to the gfx12 sibling on RDNA4 archs; gfx11 path is the
    /// canonical implementation (X is converted from F32 to FP16 via
    /// `ensure_fp16_x`). Mirrors `gemm_qkv_hfq4g256_wmma`.
    pub fn gemm_qkv_q8_0_wmma(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch_caps.is_rdna4() {
            return self.gemm_qkv_q8_0_wmma_gfx12(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        debug_assert_eq!(
            k % 32,
            0,
            "gemm_qkv_q8_0_wmma: K must be a multiple of 32 (got K={k})"
        );
        static Q8_QKV_4W: OnceLock<Option<bool>> = OnceLock::new();
        let total_m = q_m + k_m + v_m;
        let auto_q8_qkv_4w =
            batch_size >= 128 || (batch_size >= 64 && k == 3072 && total_m >= 8192);
        let q8_qkv_4w = Self::gfx1151_q8_4w_enabled(
            *Q8_QKV_4W.get_or_init(|| Self::q8_4w_mode("HIPFIRE_Q8_QKV_4W")),
            auto_q8_qkv_4w,
        );
        if q8_qkv_4w && self.arch == "gfx1151" && batch_size % 64 == 0 {
            return self.gemm_qkv_q8_0_wmma_4w_gfx1151(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_qkv_q8_0_wmma",
            kernels::GEMM_QKV_Q8_0_WMMA_SRC,
            "gemm_qkv_q8_0_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        // Byte accounting: per-weight Q8_0 = (k/32)*34 bytes/row × m rows,
        // plus X (fp16) and 3× Y (f32).
        let q8_bytes = |m: usize| m * (k / 32) * 34;
        let bytes = q8_bytes(q_m)
            + q8_bytes(k_m)
            + q8_bytes(v_m)
            + batch_size * k * 2
            + batch_size * total_m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_q8_0_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_q8_0_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
