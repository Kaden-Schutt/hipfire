// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Decode-shape GEMV dispatch (all dtypes: f16/bf16/f32, hfq/mq/lloyd/oq/q8, Paro, MoE-scalar-indexed). Pure move (Phase 1 M5).

use super::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    /// Q4_LUT GEMV: 4-bit with LDS codebook lookup. 48 bytes per 32 elements.
    pub fn gemv_q4lut(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_q4lut", kernels::GEMV_Q4LUT_SRC, "gemv_q4lut")?;
        let func = &self.functions["gemv_q4lut"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        // LDS: 8 codebooks × 16 entries × 2 bytes = 256 bytes
        let shared_mem = 256u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                shared_mem,
                None,
                &mut params,
            )
        }
    }
    /// Wave-cooperative Q4 GEMV (Q4_F16_G32 format, 0.625 B/w). Shuffle-based nibble distribution.
    pub fn gemv_q4wave(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_q4wave", kernels::GEMV_Q4WAVE_SRC, "gemv_q4wave")?;
        let func = &self.functions["gemv_q4wave"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip
                .launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, None, &mut params)
        }
    }
    /// Q4-as-Q8 GEMV: 4-bit precision stored in Q8_0 format (1.0625 B/w). Gets Q8 occupancy.
    pub fn gemv_q4as8(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_q4as8", kernels::GEMV_Q4AS8_SRC, "gemv_q4as8")?;
        let func = &self.functions["gemv_q4as8"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip
                .launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, None, &mut params)
        }
    }
    /// y = A * x (matrix-vector multiply, A is [M, K], x is [K], y is [M])
    pub fn gemv_f32(&mut self, a: &GpuTensor, x: &GpuTensor, y: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv", kernels::GEMV_SRC, "gemv_f32")?;
        let func = &self.functions["gemv_f32"];

        let m = a.shape[0] as i32;
        let k = a.shape[1] as i32;
        let alpha = 1.0f32;
        let beta = 0.0f32;

        let mut a_ptr = a.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m;
        let mut k_val = k;
        let mut alpha_val = alpha;
        let mut beta_val = beta;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut alpha_val as *mut _ as *mut c_void,
            &mut beta_val as *mut _ as *mut c_void,
        ];

        // One block per row, 256 threads per block with shared memory reduction
        let block_size = 256u32.min(k as u32);
        let shared_mem = block_size * 4; // one float per thread
        let bytes = (m as usize) * (k as usize) * 4 + (k as usize) * 4 + (m as usize) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_f32", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// y = A_q4k * x (quantized matrix-vector multiply, A stored as Q4_K on GPU)
    /// a_raw: raw Q4_K bytes on GPU, x: F32 input, y: F32 output
    /// m: number of output rows, k: number of input columns (must be multiple of 256)
    pub fn gemv_q4k(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_q4k", kernels::GEMV_Q4K_SRC, "gemv_q4k")?;
        let func = &self.functions["gemv_q4k"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 32u32; // single warp — no shared memory needed
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// HFQ4-G128 GEMV: flat 4-bit with 128-weight groups.
    /// K must be multiple of 128.
    pub fn gemv_hfq4g128(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq4g128", kernels::GEMV_HFQ4G128_SRC, "gemv_hfq4g128")?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let bytes = crate::profile::gemv_hfq4g128_bytes(m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_hfq4g128", bytes);
        let result = self.launch_maybe_blob(
            "gemv_hfq4g128",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// PARO4-G128 GEMV: ParoQuant pair-rotated activation + W4 weights.
    /// K must be multiple of 128 and M must be a multiple of the AWQ pack size
    /// (8). Each block computes one packed output column (8 output rows).
    pub fn gemv_paro4g128(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(m % 8, 0, "PARO4G128 GEMV requires M multiple of 8, got {m}");
        assert_eq!(
            k % 128,
            0,
            "PARO4G128 GEMV requires K multiple of 128, got {k}"
        );
        self.ensure_kernel(
            "gemv_paro4g128",
            kernels::GEMV_PARO4G128_SRC,
            "gemv_paro4g128",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let grid_x = (m / 8) as u32;
        let bytes = crate::profile::gemv_paro4g128_prerotated_bytes(m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_paro4g128", bytes);
        let result = self.launch_maybe_blob(
            "gemv_paro4g128",
            [grid_x, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Residual PARO4-G128 GEMV: y += A(x) where x is pair-rotated per
    /// ParoQuant metadata. One block computes one AWQ packed output column.
    pub fn gemv_paro4g128_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            m % 8,
            0,
            "PARO4G128 residual GEMV requires M multiple of 8, got {m}"
        );
        assert_eq!(
            k % 128,
            0,
            "PARO4G128 residual GEMV requires K multiple of 128, got {k}"
        );
        self.ensure_kernel(
            "gemv_paro4g128",
            kernels::GEMV_PARO4G128_SRC,
            "gemv_paro4g128_residual",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let grid_x = (m / 8) as u32;
        let bytes = crate::profile::gemv_paro4g128_prerotated_bytes(m, k) + m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemv", "gemv_paro4g128_residual", bytes);
        let result = self.launch_maybe_blob(
            "gemv_paro4g128_residual",
            [grid_x, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// PARO4-G128 fused SwiGLU down projection: y += W * (silu(gate) * up).
    /// Saves the standalone `silu_mul_f32` launch and ffn_hidden global write/read.
    pub fn gemv_paro4g128_swiglu_residual(
        &mut self,
        a_raw: &GpuTensor,
        gate: &GpuTensor,
        up: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            m % 8,
            0,
            "PARO4G128 SwiGLU residual GEMV requires M multiple of 8, got {m}"
        );
        assert_eq!(
            k % 128,
            0,
            "PARO4G128 SwiGLU residual GEMV requires K multiple of 128, got {k}"
        );
        self.ensure_kernel(
            "gemv_paro4g128",
            kernels::GEMV_PARO4G128_SRC,
            "gemv_paro4g128_swiglu_residual",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let gate_ptr = gate.buf.as_ptr();
        let up_ptr = up.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &gate_ptr as *const _ as *mut c_void,
            &up_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let grid_x = (m / 8) as u32;
        let bytes = crate::profile::gemv_paro4g128_prerotated_bytes(m, k) + k * 4 + m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemv", "gemv_paro4g128_swiglu_residual", bytes);
        let result = self.launch_maybe_blob(
            "gemv_paro4g128_swiglu_residual",
            [grid_x, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(gate_ptr);
                b.push_ptr(up_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// PARO4-G128T direct GEMV for tiny-M projections. This keeps the Paro
    /// rotation inside the GEMV block instead of materializing x_rot globally.
    pub fn gemv_paro4g128t_direct(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            m % 8,
            0,
            "PARO4G128T direct GEMV requires M multiple of 8, got {m}"
        );
        assert_eq!(
            k % 128,
            0,
            "PARO4G128T direct GEMV requires K multiple of 128, got {k}"
        );
        self.ensure_kernel(
            "gemv_paro4g128",
            kernels::GEMV_PARO4G128_SRC,
            "gemv_paro4g128t_direct",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let grid_x = (m / 8) as u32;
        let bytes = crate::profile::gemv_paro4g128_prerotated_bytes(m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_paro4g128t_direct", bytes);
        let result = self.launch_maybe_blob(
            "gemv_paro4g128t_direct",
            [grid_x, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Residual PARO4-G128T direct GEMV for tiny-M projections.
    pub fn gemv_paro4g128t_direct_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            m % 8,
            0,
            "PARO4G128T direct residual GEMV requires M multiple of 8, got {m}"
        );
        assert_eq!(
            k % 128,
            0,
            "PARO4G128T direct residual GEMV requires K multiple of 128, got {k}"
        );
        self.ensure_kernel(
            "gemv_paro4g128",
            kernels::GEMV_PARO4G128_SRC,
            "gemv_paro4g128t_direct_residual",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let grid_x = (m / 8) as u32;
        let bytes = crate::profile::gemv_paro4g128_prerotated_bytes(m, k) + m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_paro4g128t_direct_residual",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_paro4g128t_direct_residual",
            [grid_x, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// PARO4-G128 GEMV over an already materialized Paro-rotated activation.
    pub fn gemv_paro4g128_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            m % 8,
            0,
            "PARO4G128 prerotated GEMV requires M multiple of 8, got {m}"
        );
        assert_eq!(
            k % 128,
            0,
            "PARO4G128 prerotated GEMV requires K multiple of 128, got {k}"
        );
        self.ensure_kernel(
            "gemv_paro4g128",
            kernels::GEMV_PARO4G128_SRC,
            "gemv_paro4g128_prerotated",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x_rot.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let grid_x = (m / 8) as u32;
        let bytes = crate::profile::gemv_paro4g128_prerotated_bytes(m, k);
        let timer =
            crate::profile::begin_timer(&self.hip, "gemv", "gemv_paro4g128_prerotated", bytes);
        let result = self.launch_maybe_blob(
            "gemv_paro4g128_prerotated",
            [grid_x, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Residual PARO4-G128 GEMV over an already materialized Paro-rotated
    /// activation.
    pub fn gemv_paro4g128_prerotated_residual(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            m % 8,
            0,
            "PARO4G128 prerotated residual GEMV requires M multiple of 8, got {m}"
        );
        assert_eq!(
            k % 128,
            0,
            "PARO4G128 prerotated residual GEMV requires K multiple of 128, got {k}"
        );
        self.ensure_kernel(
            "gemv_paro4g128",
            kernels::GEMV_PARO4G128_SRC,
            "gemv_paro4g128_prerotated_residual",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x_rot.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let grid_x = (m / 8) as u32;
        let bytes = crate::profile::gemv_paro4g128_prerotated_bytes(m, k) + m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_paro4g128_prerotated_residual",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_paro4g128_prerotated_residual",
            [grid_x, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// PARO4-G128T GEMV over an already materialized Paro-rotated activation.
    /// The payload stores qweight as [M/8, K], making the inner-loop reads
    /// contiguous for the GEMV access pattern.
    pub fn gemv_paro4g128t_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            m % 8,
            0,
            "PARO4G128T prerotated GEMV requires M multiple of 8, got {m}"
        );
        assert_eq!(
            k % 128,
            0,
            "PARO4G128T prerotated GEMV requires K multiple of 128, got {k}"
        );
        self.ensure_kernel(
            "gemv_paro4g128",
            kernels::GEMV_PARO4G128_SRC,
            "gemv_paro4g128t_prerotated",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x_rot.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let grid_x = (m / 8) as u32;
        let bytes = crate::profile::gemv_paro4g128_prerotated_bytes(m, k);
        let timer =
            crate::profile::begin_timer(&self.hip, "gemv", "gemv_paro4g128t_prerotated", bytes);
        let result = self.launch_maybe_blob(
            "gemv_paro4g128t_prerotated",
            [grid_x, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Residual PARO4-G128T GEMV over an already materialized Paro-rotated
    /// activation.
    pub fn gemv_paro4g128t_prerotated_residual(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            m % 8,
            0,
            "PARO4G128T prerotated residual GEMV requires M multiple of 8, got {m}"
        );
        assert_eq!(
            k % 128,
            0,
            "PARO4G128T prerotated residual GEMV requires K multiple of 128, got {k}"
        );
        self.ensure_kernel(
            "gemv_paro4g128",
            kernels::GEMV_PARO4G128_SRC,
            "gemv_paro4g128t_prerotated_residual",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x_rot.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let grid_x = (m / 8) as u32;
        let bytes = crate::profile::gemv_paro4g128_prerotated_bytes(m, k) + m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_paro4g128t_prerotated_residual",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_paro4g128t_prerotated_residual",
            [grid_x, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// PARO4-G128T prerotated GEMV with four output lanes per block. This
    /// duplicates qweight reads relative to the 8-lane pack but lowers
    /// accumulator/register pressure for empirical Atlas testing.
    pub fn gemv_paro4g128t_prerotated_pack4(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            m % 8,
            0,
            "PARO4G128T pack4 GEMV requires M multiple of 8, got {m}"
        );
        assert_eq!(
            k % 128,
            0,
            "PARO4G128T pack4 GEMV requires K multiple of 128, got {k}"
        );
        self.ensure_kernel(
            "gemv_paro4g128",
            kernels::GEMV_PARO4G128_SRC,
            "gemv_paro4g128t_prerotated_pack4",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x_rot.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let grid_x = (m / 4) as u32;
        let bytes = crate::profile::gemv_paro4g128_prerotated_bytes(m, k) * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_paro4g128t_prerotated_pack4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_paro4g128t_prerotated_pack4",
            [grid_x, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Residual PARO4-G128T pack4 prerotated GEMV.
    pub fn gemv_paro4g128t_prerotated_residual_pack4(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            m % 8,
            0,
            "PARO4G128T pack4 residual GEMV requires M multiple of 8, got {m}"
        );
        assert_eq!(
            k % 128,
            0,
            "PARO4G128T pack4 residual GEMV requires K multiple of 128, got {k}"
        );
        self.ensure_kernel(
            "gemv_paro4g128",
            kernels::GEMV_PARO4G128_SRC,
            "gemv_paro4g128t_prerotated_residual_pack4",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x_rot.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let grid_x = (m / 4) as u32;
        let bytes = crate::profile::gemv_paro4g128_prerotated_bytes(m, k) * 2 + m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_paro4g128t_prerotated_residual_pack4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_paro4g128t_prerotated_residual_pack4",
            [grid_x, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// PARO4-G128T prerotated GEMV with two output lanes per block. This is
    /// an Atlas probe for whether lower accumulator pressure beats duplicate
    /// qweight traffic on the residual/down hot path.
    pub fn gemv_paro4g128t_prerotated_pack2(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            m % 8,
            0,
            "PARO4G128T pack2 GEMV requires M multiple of 8, got {m}"
        );
        assert_eq!(
            k % 128,
            0,
            "PARO4G128T pack2 GEMV requires K multiple of 128, got {k}"
        );
        self.ensure_kernel(
            "gemv_paro4g128",
            kernels::GEMV_PARO4G128_SRC,
            "gemv_paro4g128t_prerotated_pack2",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x_rot.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let grid_x = (m / 2) as u32;
        let bytes = crate::profile::gemv_paro4g128_prerotated_bytes(m, k) * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_paro4g128t_prerotated_pack2",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_paro4g128t_prerotated_pack2",
            [grid_x, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Residual PARO4-G128T pack2 prerotated GEMV.
    pub fn gemv_paro4g128t_prerotated_residual_pack2(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            m % 8,
            0,
            "PARO4G128T pack2 residual GEMV requires M multiple of 8, got {m}"
        );
        assert_eq!(
            k % 128,
            0,
            "PARO4G128T pack2 residual GEMV requires K multiple of 128, got {k}"
        );
        self.ensure_kernel(
            "gemv_paro4g128",
            kernels::GEMV_PARO4G128_SRC,
            "gemv_paro4g128t_prerotated_residual_pack2",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x_rot.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let grid_x = (m / 2) as u32;
        let bytes = crate::profile::gemv_paro4g128_prerotated_bytes(m, k) * 4 + m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_paro4g128t_prerotated_residual_pack2",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_paro4g128t_prerotated_residual_pack2",
            [grid_x, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// PARO4-G128T prerotated GEMV with one output lane per block.
    pub fn gemv_paro4g128t_prerotated_pack1(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            m % 8,
            0,
            "PARO4G128T pack1 GEMV requires M multiple of 8, got {m}"
        );
        assert_eq!(
            k % 128,
            0,
            "PARO4G128T pack1 GEMV requires K multiple of 128, got {k}"
        );
        self.ensure_kernel(
            "gemv_paro4g128",
            kernels::GEMV_PARO4G128_SRC,
            "gemv_paro4g128t_prerotated_pack1",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x_rot.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let bytes = crate::profile::gemv_paro4g128_prerotated_bytes(m, k) * 8;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_paro4g128t_prerotated_pack1",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_paro4g128t_prerotated_pack1",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Residual PARO4-G128T pack1 prerotated GEMV.
    pub fn gemv_paro4g128t_prerotated_residual_pack1(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            m % 8,
            0,
            "PARO4G128T pack1 residual GEMV requires M multiple of 8, got {m}"
        );
        assert_eq!(
            k % 128,
            0,
            "PARO4G128T pack1 residual GEMV requires K multiple of 128, got {k}"
        );
        self.ensure_kernel(
            "gemv_paro4g128",
            kernels::GEMV_PARO4G128_SRC,
            "gemv_paro4g128t_prerotated_residual_pack1",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x_rot.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let bytes = crate::profile::gemv_paro4g128_prerotated_bytes(m, k) * 8 + m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_paro4g128t_prerotated_residual_pack1",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_paro4g128t_prerotated_residual_pack1",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// PARO4-G128 rotate-once wrapper used for env-gated runtime probes.
    pub fn gemv_paro4g128_with_prerotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to `paro4g128_rotate` which binds
        self.paro4g128_rotate(a_raw, x, x_rot, m, k)?;
        self.gemv_paro4g128_prerotated(a_raw, x_rot, y, m, k)
    }
    /// PARO4-G128 rotate-once residual wrapper used for env-gated runtime probes.
    pub fn gemv_paro4g128_residual_with_prerotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to `paro4g128_rotate` which binds
        self.paro4g128_rotate(a_raw, x, x_rot, m, k)?;
        self.gemv_paro4g128_prerotated_residual(a_raw, x_rot, y, m, k)
    }
    /// PARO4-G128 fused SwiGLU rotate-once down projection.
    pub fn gemv_paro4g128_swiglu_residual_with_prerotate(
        &mut self,
        a_raw: &GpuTensor,
        gate: &GpuTensor,
        up: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to `paro4g128_swiglu_rotate` which binds
        self.paro4g128_swiglu_rotate(a_raw, gate, up, x_rot, m, k)?;
        self.gemv_paro4g128_prerotated_residual(a_raw, x_rot, y, m, k)
    }
    /// PARO4-G128T rotate-once wrapper for engine-tiled qweight payloads.
    pub fn gemv_paro4g128t_with_prerotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to `paro4g128t_rotate` which binds
        self.paro4g128t_rotate(a_raw, x, x_rot, m, k)?;
        if std::env::var_os("HIPFIRE_PARO_PACK1").is_some() {
            return self.gemv_paro4g128t_prerotated_pack1(a_raw, x_rot, y, m, k);
        }
        if std::env::var_os("HIPFIRE_PARO_PACK2").is_some() {
            return self.gemv_paro4g128t_prerotated_pack2(a_raw, x_rot, y, m, k);
        }
        if std::env::var_os("HIPFIRE_PARO_PACK4").is_some() {
            return self.gemv_paro4g128t_prerotated_pack4(a_raw, x_rot, y, m, k);
        }
        self.gemv_paro4g128t_prerotated(a_raw, x_rot, y, m, k)
    }
    /// PARO4-G128T rotate-once residual wrapper for engine-tiled qweight payloads.
    pub fn gemv_paro4g128t_residual_with_prerotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to `paro4g128t_rotate` which binds
        self.paro4g128t_rotate(a_raw, x, x_rot, m, k)?;
        if std::env::var_os("HIPFIRE_PARO_PACK1").is_some() {
            return self.gemv_paro4g128t_prerotated_residual_pack1(a_raw, x_rot, y, m, k);
        }
        if std::env::var_os("HIPFIRE_PARO_PACK2").is_some() {
            return self.gemv_paro4g128t_prerotated_residual_pack2(a_raw, x_rot, y, m, k);
        }
        if std::env::var_os("HIPFIRE_PARO_PACK4").is_some() {
            return self.gemv_paro4g128t_prerotated_residual_pack4(a_raw, x_rot, y, m, k);
        }
        self.gemv_paro4g128t_prerotated_residual(a_raw, x_rot, y, m, k)
    }
    /// PARO4-G128T fused SwiGLU rotate-once down projection.
    pub fn gemv_paro4g128t_swiglu_residual_with_prerotate(
        &mut self,
        a_raw: &GpuTensor,
        gate: &GpuTensor,
        up: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to `paro4g128t_swiglu_rotate` which binds
        self.paro4g128t_swiglu_rotate(a_raw, gate, up, x_rot, m, k)?;
        if std::env::var_os("HIPFIRE_PARO_PACK1").is_some() {
            return self.gemv_paro4g128t_prerotated_residual_pack1(a_raw, x_rot, y, m, k);
        }
        if std::env::var_os("HIPFIRE_PARO_PACK2").is_some() {
            return self.gemv_paro4g128t_prerotated_residual_pack2(a_raw, x_rot, y, m, k);
        }
        if std::env::var_os("HIPFIRE_PARO_PACK4").is_some() {
            return self.gemv_paro4g128t_prerotated_residual_pack4(a_raw, x_rot, y, m, k);
        }
        self.gemv_paro4g128t_prerotated_residual(a_raw, x_rot, y, m, k)
    }
    /// HFQ2-G256 GEMV. K must be multiple of 256.
    pub fn gemv_hfq2g256(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq2g256", kernels::GEMV_HFQ2G256_SRC, "gemv_hfq2g256")?;
        let func = &self.functions["gemv_hfq2g256"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// MQ2-Lloyd GEMV (2-bit + per-block 4-entry fp16 codebook). K must be a
    /// multiple of 256. Same launch shape as gemv_hfq2g256 — header is the
    /// only layout difference.
    pub fn gemv_mq2g256_lloyd(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_mq2g256_lloyd",
            kernels::GEMV_MQ2G256_LLOYD_SRC,
            "gemv_mq2g256_lloyd",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "gemv_mq2g256_lloyd",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        )
    }
    /// MQ2-Lloyd GEMV with engine-side x rotation (matches `gemv_mq2g256_with_rotate`).
    pub fn gemv_mq2g256_lloyd_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to rotate_x_mq + gemv_mq2g256_lloyd, both of which bind.
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_mq2g256_lloyd(a_raw, x_rot, y, m, k)
    }
    /// MQ3-Lloyd GEMV (3-bit + per-block 8-entry fp16 codebook). K must be a
    /// multiple of 256. gfx1100/1101/1102 use the K4-unrolled + LDS-codebook
    /// variant; other archs fall back to the baseline switch-dispatch path.
    pub fn gemv_mq3g256_lloyd(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) =
            kernels::gemv_mq3g256_lloyd_for_arch(&self.arch_caps, self.flags.lloyd_force_baseline);
        self.ensure_kernel(module, src, "gemv_mq3g256_lloyd")?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gemv_mq3g256_lloyd_bytes(m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_mq3g256_lloyd", bytes);
        let result = self.launch_maybe_blob(
            "gemv_mq3g256_lloyd",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// MQ3-Lloyd GEMV with engine-side x rotation.
    pub fn gemv_mq3g256_lloyd_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to rotate_x_mq + gemv_mq3g256_lloyd, both of which bind.
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_mq3g256_lloyd(a_raw, x_rot, y, m, k)
    }
    /// MQ4-Lloyd GEMV (4-bit + per-block 16-entry fp16 codebook). K must be a
    /// multiple of 256. gfx1100/1101/1102/1151 use the K4-unrolled + LDS-codebook
    /// variant (cooperative double-load for the 64-entry table). Other archs
    /// fall back to the chip-agnostic baseline switch-dispatch path.
    pub fn gemv_mq4g256_lloyd(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) =
            kernels::gemv_mq4g256_lloyd_for_arch(&self.arch_caps, self.flags.lloyd_force_baseline);
        self.ensure_kernel(module, src, "gemv_mq4g256_lloyd")?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gemv_mq4g256_lloyd_bytes(m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_mq4g256_lloyd", bytes);
        let result = self.launch_maybe_blob(
            "gemv_mq4g256_lloyd",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// MQ4-Lloyd GEMV with engine-side x rotation.
    pub fn gemv_mq4g256_lloyd_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to rotate_x_mq + gemv_mq4g256_lloyd.
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_mq4g256_lloyd(a_raw, x_rot, y, m, k)
    }
    /// DIAGNOSTIC ONLY: K4 multi-accumulator MQ4-Lloyd GEMV. NOT for production.
    /// Used by examples/diag_mq4_lloyd_multiacc.rs to compare against the slow
    /// generic kernel on real model rows. See the kernel header for the
    /// open question this exists to investigate.
    pub fn gemv_mq4g256_lloyd_multiacc_diag(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_mq4g256_lloyd_multiacc_diag",
            kernels::GEMV_MQ4G256_LLOYD_MULTIACC_DIAG_GFX1100_SRC,
            "gemv_mq4g256_lloyd_multiacc_diag",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "gemv_mq4g256_lloyd_multiacc_diag",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        )
    }
    /// MQ4-Lloyd GEMV with fused residual add: y[row] += A[row] · x. Mirrors
    /// gemv_mq3g256_lloyd_residual; same single-acc bug fix applies.
    pub fn gemv_mq4g256_lloyd_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemv_mq4g256_lloyd_residual_for_arch(
            &self.arch_caps,
            self.flags.lloyd_force_baseline,
        );
        self.ensure_kernel(module, src, "gemv_mq4g256_lloyd_residual")?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gemv_mq4g256_lloyd_bytes(m, k);
        let timer =
            crate::profile::begin_timer(&self.hip, "gemv", "gemv_mq4g256_lloyd_residual", bytes);
        let result = self.launch_maybe_blob(
            "gemv_mq4g256_lloyd_residual",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// MQ4-Lloyd residual GEMV with engine-side x rotation.
    pub fn gemv_mq4g256_lloyd_residual_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_mq4g256_lloyd_residual(a_raw, x_rot, y, m, k)
    }
    /// MQ3-Lloyd GEMV with fused residual add: y[row] += A[row] · x. Used by
    /// `weight_gemv_residual` MQ3-Lloyd arm to eliminate the alloc + gemv +
    /// add_inplace_f32 + free fallback chain (saves ~4.4% of decode time on
    /// 9B Lloyd-MQ3, gfx1100, per the 2026-05-06 decode profile).
    pub fn gemv_mq3g256_lloyd_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemv_mq3g256_lloyd_residual_for_arch(
            &self.arch_caps,
            self.flags.lloyd_force_baseline,
        );
        self.ensure_kernel(module, src, "gemv_mq3g256_lloyd_residual")?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gemv_mq3g256_lloyd_bytes(m, k);
        let timer =
            crate::profile::begin_timer(&self.hip, "gemv", "gemv_mq3g256_lloyd_residual", bytes);
        let result = self.launch_maybe_blob(
            "gemv_mq3g256_lloyd_residual",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// MQ3-Lloyd residual GEMV with engine-side x rotation.
    pub fn gemv_mq3g256_lloyd_residual_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to rotate_x_mq + gemv_mq3g256_lloyd_residual.
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_mq3g256_lloyd_residual(a_raw, x_rot, y, m, k)
    }
    /// MagnumQuant GEMV: FWHT-rotated HFQ4-G256. Rotates x per group via ds_swizzle,
    /// then standard 4-bit dot product. signs1/signs2 are the FWHT sign tables (256 floats each).
    pub fn gemv_mq4g256(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_mq4g256", kernels::GEMV_MQ4G256_SRC, "gemv_mq4g256")?;
        let func = &self.functions["gemv_mq4g256"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut s1_ptr = signs1.buf.as_ptr();
        let mut s2_ptr = signs2.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut s1_ptr as *mut _ as *mut c_void,
            &mut s2_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        // LDS for rotated x: 256 floats = 1024 bytes
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                1024,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// HFP4-G32 GEMV — RDNA-optimal FP4 (E2M1 + UE8M0 g32 + FP16 row scale).
    ///
    /// v1 correctness anchor: no WMMA, no FP8, no rotation. K must be a multiple of 256
    /// (the kernel's 4-accumulator + tail-by-g%4 outer loop assumes the 256-element
    /// "iter window" stride; v2 will lift this to k%32==0). See `kernels/src/gemv_hfp4g32.hip`
    /// and `docs/quant-formats/hfp4.md`.
    pub fn gemv_hfp4g32(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            k % 256 == 0,
            "gemv_hfp4g32 requires K%256==0 in v1, got K={}",
            k
        );
        // Shape-gated: FP8 dot4 only when M is large enough that it
        // actually wins (FFN shapes). At M < 4096 the fallback wins or
        // ties; uniform-FP8 was net-negative in 9B Qwen 3.5 decode.
        if self.arch_caps.has_wmma_w32_gfx12() && self.flags.fp8_wmma && m >= super::FP8_GEMV_MIN_M
        {
            return self.gemv_hfp4g32_fp8_gfx12(a_raw, x, y, m, k);
        }
        // gfx11 (RDNA3) v_dot2_f32_f16 trickle-down: replaces the
        // fallback's F32 mul+fma chain with one fdot2 per 2 elements.
        // No new scratch (reuses ensure_fp16_x), no cross-kernel
        // context cost like the FP8 path had. Default-on for gfx11.
        // Kill switch HIPFIRE_DOT2_GEMV=0 for A/B benching.
        if self.arch_caps.has_wmma_w32() && self.flags.dot2_gemv {
            return self.gemv_hfp4g32_dot2_gfx11(a_raw, x, y, m, k);
        }
        self.gemv_hfp4g32_fallback(a_raw, x, y, m, k)
    }
    /// Direct fallback entry point (F32 mul+fma chain). Useful for
    /// A/B benchmarking against the dot2/fp8 variants.
    pub fn gemv_hfp4g32_fallback(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemv_hfp4g32_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemv_hfp4g32")?;
        let func = &self.functions["gemv_hfp4g32"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        // LDS: 16-entry FP16 LUT = 32 bytes.
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                32,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// MagnumQuant MQ4: rotate x once, then GEMV against rotated x.
    /// MQ4 weights are stored in HFQ4-G256 format with FWHT pre-applied, so the GEMV
    /// inner loop is identical to standard HFQ4 — we reuse the arch-tuned HFQ4 kernel.
    pub fn gemv_mq4g256_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rotate_x_mq(x, x_rot, k)?;
        // MQ4 = FWHT-rotated HFQ4-G256. dot(rot(W), rot(x)) = dot(W, x).
        // Route through the arch-specific HFQ4 kernel (4x unroll on gfx1100, etc).
        self.gemv_hfq4g256(a_raw, x_rot, y, m, k)
    }
    /// MagnumQuant MQ4 with pre-rotated x. Skips the rotation step entirely —
    /// caller must have called `rotate_x_mq` into `x_rot` first.
    pub fn gemv_mq4g256_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_hfq4g256(a_raw, x_rot, y, m, k)
    }
    /// MagnumQuant MQ4-G128 with pre-rotated x. Skips the rotation step entirely —
    /// caller must have called `rotate_x_mq_128` into `x_rot` first.
    pub fn gemv_mq4g128_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_hfq4g128(a_raw, x_rot, y, m, k)
    }
    /// MFP4G32: rotate x once via FWHT, then HFP4G32 GEMV against rotated x.
    /// MFP4 weights are stored in HFP4G32 format (E2M1 + UE8M0 g32 + FP16 row scale)
    /// with the same 256-element FWHT pre-applied, so the GEMV inner loop is
    /// identical to standard HFP4 — we reuse `gemv_hfp4g32`.
    pub fn gemv_mfp4g32_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Shape-gated FP8 routing (Option α empirical embodiment): only
        // when M ≥ super::FP8_GEMV_MIN_M does FP8 dot4 win measurably on this
        // path. Below threshold (e.g. wo M=2048), the FP8 fused-rotation
        // costs more than the dot4 ALU savings — keep the F32 fallback.
        if self.arch_caps.has_wmma_w32_gfx12() && self.flags.fp8_wmma && m >= super::FP8_GEMV_MIN_M
        {
            let x_fp8_ptr = self.rotate_x_mq_dual_fp8(x, x_rot, k)?;
            return self.gemv_hfp4g32_fp8_gfx12_with_fp8_ptr(a_raw, x_fp8_ptr, y, m, k);
        }
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_hfp4g32(a_raw, x_rot, y, m, k)
    }
    /// MFP4G32 with pre-rotated x. Skips the rotation step entirely — caller must
    /// have called `rotate_x_mq` into `x_rot` first.
    pub fn gemv_mfp4g32_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_hfp4g32(a_raw, x_rot, y, m, k)
    }
    /// MagnumQuant MQ3: rotate x once, then HFQ3-G256 GEMV against rotated x.
    /// MQ3 weights are stored in HFQ3-G256 format (104 B/group) with FWHT pre-applied,
    /// so the GEMV inner loop is identical to standard HFQ3.
    pub fn gemv_mq3g256_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_hfq3g256(a_raw, x_rot, y, m, k)
    }
    /// MagnumQuant MQ3 with pre-rotated x.
    pub fn gemv_mq3g256_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_hfq3g256(a_raw, x_rot, y, m, k)
    }
    /// MagnumQuant MQ2: rotate x once, then HFQ2-G256 GEMV against rotated x.
    /// MQ2 weights are stored in HFQ2-G256 format (72 B/group) with FWHT pre-applied.
    pub fn gemv_mq2g256_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_hfq2g256(a_raw, x_rot, y, m, k)
    }
    /// MagnumQuant MQ2 with pre-rotated x.
    pub fn gemv_mq2g256_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_hfq2g256(a_raw, x_rot, y, m, k)
    }
    /// MagnumQuant MQ6: rotate x via FWHT, then HFQ6 GEMV.
    pub fn gemv_mq6g256_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_hfq6g256(a_raw, x_rot, y, m, k)
    }
    /// MagnumQuant MQ6 with pre-rotated x.
    pub fn gemv_mq6g256_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_hfq6g256(a_raw, x_rot, y, m, k)
    }
    /// MQ8 dp4a GEMV using pre-rotated+quantized x. Caller must have called
    /// `rotate_quantize_x_mq8(x, k)` first — results use the internal `mq_x_q8`/`mq_x_scales`.
    pub fn gemv_mq8g256_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_mq8g256", kernels::GEMV_MQ8G256_SRC, "gemv_mq8g256")?;

        let xq_ptr = self.mq_x_q8.as_ref().unwrap().as_ptr();
        let xs_ptr = self.mq_x_scales.as_ref().unwrap().as_ptr();

        let func = &self.functions["gemv_mq8g256"];
        let mut ap = a_raw.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut xs = xs_ptr;
        let mut yp = y.buf.as_ptr();
        let mut mv = m as i32;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut xs as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mv as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// MagnumQuant MQ8: FWHT rotate + INT8 quantize x, then dp4a GEMV.
    pub fn gemv_mq8g256_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rotate_quantize_x_mq8(x, k)?;
        self.gemv_mq8g256_prerotated(a_raw, y, m, k)
    }
    /// HFQ3-G256 GEMV. K must be multiple of 256.
    /// Per-arch dispatch: gfx1100/1101/1102 uses the K4-unrolled
    /// 4-accumulator variant. The default kernel was re-ported to match
    /// the same ordering so non-RDNA3 archs (gfx1010, gfx1030, gfx12,
    /// gfx9xx) produce byte-exact results against the RDNA3 baseline.
    /// Uses `launch_maybe_blob` for HIPFIRE_GRAPH=1 capture safety.
    pub fn gemv_hfq3g256(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemv_hfq3g256_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemv_hfq3g256")?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "gemv_hfq3g256",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        )
    }
    /// HFQ3-G256 GEMV with fused residual add: y[row] += A[row] dot x.
    /// Used by `weight_gemv_residual` MQ3 arm to eliminate the
    /// alloc+gemv+add+free fallback chain (saves ~3 launches per residual).
    /// gfx1100 selects the K4-unrolled chip-specific variant (commit 0003103,
    /// 9B MQ3 decode 114 to 141 tok/s); other archs use the K4-ported default
    /// (re-port in 9fdba4d keeps non-RDNA3 archs byte-exact with the prior
    /// gemv + add_inplace path). Uses launch_maybe_blob for HIPFIRE_GRAPH=1
    /// capture safety.
    pub fn gemv_hfq3g256_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemv_hfq3g256_residual_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemv_hfq3g256_residual")?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "gemv_hfq3g256_residual",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        )
    }
    /// QTIP-3 GEMV (plain, y = W·x). Fused on-the-fly trellis decode via
    /// gemv_qtip3g256 (computed 1MAD codebook, zero LDS). x must be pre-FWHT-
    /// rotated by the caller (rotate_x_mq_for), same contract as MQ3/MQ4.
    pub fn gemv_qtip3g256(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_qtip3g256",
            kernels::GEMV_QTIP3G256_SRC,
            "gemv_qtip3g256",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "gemv_qtip3g256",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        )
    }
    /// QTIP-3 GEMV with fused residual add (y += W·x). Same decode as
    /// `gemv_qtip3g256`; the kernel's final store accumulates.
    pub fn gemv_qtip3g256_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_qtip3g256",
            kernels::GEMV_QTIP3G256_SRC,
            "gemv_qtip3g256_residual",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "gemv_qtip3g256_residual",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        )
    }
    /// MagnumQuant MQ3-G256 GEMV with fused residual add. The pre-rotation
    /// happens in a separate kernel via fused_silu_mul_mq_rotate or
    /// rotate_x_for_mq; this function just dispatches the underlying
    /// hfq3g256_residual against the already-rotated x.
    pub fn gemv_mq3g256_residual_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_hfq3g256_residual(a_raw, x_rot, y, m, k)
    }
    /// HFQ3-G128 GEMV. K must be multiple of 128. Finer granularity than G256.
    pub fn gemv_hfq3g128(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq3g128", kernels::GEMV_HFQ3G128_SRC, "gemv_hfq3g128")?;
        let func = &self.functions["gemv_hfq3g128"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// HFQ2-G128 GEMV. K must be multiple of 128. Finer granularity than G256.
    pub fn gemv_hfq2g128(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq2g128", kernels::GEMV_HFQ2G128_SRC, "gemv_hfq2g128")?;
        let func = &self.functions["gemv_hfq2g128"];
        let mut ap = a_raw.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mv = m as i32;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mv as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// HFQ6-G256 GEMV with fused residual add: y[row] += A[row] . x.
    /// Same shape as gemv_hfq6g256; only the final write differs (+= vs =).
    /// Used for wo and w_down in HFQ6 / MQ6 forward paths so the
    /// add_inplace_f32 follow-up launch can be elided.
    pub fn gemv_hfq6g256_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        // Wave64-native fast path (gfx906/908/94x): 2 rows per block, halves
        // grid.x. Mirrors the HFQ4 sibling at line ~5378. Plan §3.1.1 item 2
        // (gfx906-mq6-mq8-port.md v3.2.1 + v3.2.2). Byte-exact with the
        // wave32 base since each warp's 32-lane reduction stays in-warp.
        // ILP-prefetch variant gates on gemv_prefetch_enabled(arch) — default
        // on for gfx906 (Phase A.1b, mirror of HFQ4 +4.8% lever from `3ef127d`).
        if self.arch_caps.is_wave64_native() {
            let (kname, ksrc): (&str, &str) = if self.arch_caps.gemv_prefetch_enabled() {
                (
                    "gemv_hfq6g256_residual_wave64_prefetch",
                    kernels::GEMV_HFQ6G256_RESIDUAL_WAVE64_PREFETCH_SRC,
                )
            } else {
                (
                    "gemv_hfq6g256_residual_wave64",
                    kernels::GEMV_HFQ6G256_RESIDUAL_WAVE64_SRC,
                )
            };
            self.ensure_kernel(kname, ksrc, kname)?;
            let func = &self.functions[kname];
            let grid = ((m as u32) + 1) / 2;
            return unsafe {
                self.hip.launch_kernel(
                    func,
                    [grid, 1, 1],
                    [64, 1, 1],
                    0,
                    self.stream_ref(),
                    &mut params,
                )
            };
        }

        self.ensure_kernel(
            "gemv_hfq6g256_residual",
            kernels::GEMV_HFQ6G256_RESIDUAL_SRC,
            "gemv_hfq6g256_residual",
        )?;
        let func = &self.functions["gemv_hfq6g256_residual"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// HFQ6-G256 GEMV. K must be multiple of 256.
    pub fn gemv_hfq6g256(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq6g256", kernels::GEMV_HFQ6G256_SRC, "gemv_hfq6g256")?;
        let func = &self.functions["gemv_hfq6g256"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// HFQ8-G256 GEMV. K must be multiple of 256.
    pub fn gemv_hfq8g256(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq8g256", kernels::GEMV_HFQ8G256_SRC, "gemv_hfq8g256")?;
        let func = &self.functions["gemv_hfq8g256"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// HFQ4-G512 GEMV. K must be multiple of 512.
    pub fn gemv_hfq4g512(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq4g512", kernels::GEMV_HFQ4G512_SRC, "gemv_hfq4g512")?;
        let func = &self.functions["gemv_hfq4g512"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// HFQ4-G1024 GEMV. K must be multiple of 1024.
    pub fn gemv_hfq4g1024(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g1024",
            kernels::GEMV_HFQ4G1024_SRC,
            "gemv_hfq4g1024",
        )?;
        let func = &self.functions["gemv_hfq4g1024"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// HFQ4-G256 GEMV: flat 4-bit with 256-weight groups. K must be multiple of 256.
    pub fn gemv_hfq4g256(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (hfq4g256_src, hfq4g256_module) =
            kernels::gemv_hfq4g256_for_arch(&self.arch_caps, self.flags.rdna2_variant);
        self.ensure_kernel(hfq4g256_module, hfq4g256_src, "gemv_hfq4g256")?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(a_ptr);
            b.push_ptr(x_ptr);
            b.push_ptr(y_ptr);
            b.push_i32(m_val);
            b.push_i32(k_val);
            b
        };

        // Multi-row GEMV: one warp computes R output rows, sharing x register
        // state across rows. Measured RDNA3/RDNA3.5 defaults stay at R=1;
        // override any arch with HIPFIRE_GEMV_ROWS in {1, 2, 4, 8}.
        //
        // See gemv_rows_default() for the measurement data that motivates
        // the per-arch defaults.
        let rdna3 = self.arch_caps.is_rdna3();
        let rows = self.arch_caps.gemv_rows_default();
        let use_multirow = rows > 1;

        // RDNA2 (gfx1030/1031): always use the arch-optimized narrow kernel.
        // Other non-RDNA3 archs: use wide kernel (2 rows/block) for large M.
        let use_wide =
            !use_multirow && m >= 64 && !(self.arch_caps.is_rdna2() || self.arch_caps.is_rdna3());

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_hfq4g256", bytes);
        let result = if use_multirow {
            let (func_name, grid_div) = match rows {
                2 => ("gemv_hfq4g256_multirow_r2", 2u32),
                4 => ("gemv_hfq4g256_multirow_r4", 4u32),
                8 => ("gemv_hfq4g256_multirow_r8", 8u32),
                _ => unreachable!(),
            };
            let (mr_name, mr_src) = if rdna3 {
                (
                    "gemv_hfq4g256_multirow_rdna3",
                    kernels::GEMV_HFQ4G256_MULTIROW_GFX1100_SRC,
                )
            } else {
                (
                    "gemv_hfq4g256_multirow_default",
                    kernels::GEMV_HFQ4G256_MULTIROW_SRC,
                )
            };
            self.ensure_kernel(mr_name, mr_src, func_name)?;
            let grid = ((m as u32) + grid_div - 1) / grid_div;
            self.launch_maybe_blob(
                func_name,
                [grid, 1, 1],
                [32, 1, 1],
                0,
                &mut params,
                blob_builder,
            )
        } else if use_wide {
            self.ensure_kernel(
                "gemv_hfq4g256_wide",
                kernels::GEMV_HFQ4G256_WIDE_SRC,
                "gemv_hfq4g256_wide",
            )?;
            let grid = ((m + 1) / 2) as u32;
            self.launch_maybe_blob(
                "gemv_hfq4g256_wide",
                [grid, 1, 1],
                [64, 1, 1],
                0,
                &mut params,
                blob_builder,
            )
        } else {
            self.launch_maybe_blob(
                "gemv_hfq4g256",
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                &mut params,
                blob_builder,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// HFQ4-G256 GEMV with fused residual add: y[row] += A[row] · x.
    /// Same math as `gemv_hfq4g256` but the final write accumulates into `y`
    /// instead of overwriting. Used for wo / w_down projections where the
    /// following step would have been `x += gemv_out` via add_inplace_f32.
    pub fn gemv_hfq4g256_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemv_hfq4g256_residual_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemv_hfq4g256_residual")?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        // CDNA3 wave64 fast path: 2 rows per block, halves grid.x. The base
        // kernel runs at half throughput on a wave64-native arch because
        // half the wave masks out per `__shfl_down`. Byte-exact with base.
        let cdna3 = self.arch_caps.is_wave64_native();

        // RDNA3 multi-row path. RDNA3/RDNA3.5 defaults to R=1 from measured
        // regressions; HIPFIRE_GEMV_ROWS can still opt into R=2/4/8.
        // Non-RDNA3 archs take the single-row residual path because only the
        // RDNA3 residual multi-row source exists.
        let rdna3 = self.arch_caps.is_rdna3();
        let rows = if rdna3 {
            self.arch_caps.gemv_rows_default()
        } else {
            1
        };
        let use_multirow = rdna3 && rows > 1;

        // Bandwidth: weight + x + y_read (for residual) + y_write.
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_hfq4g256_residual", bytes);
        let result = if cdna3 {
            // gfx94x (CDNA3 / MI300X) takes the LDS-cached 8-rows-per-WG path
            // when enabled; gfx906/908 (or env override) keep wave64 base.
            if self.flags.gfx942_gemv_v3 {
                let kname = "gemv_hfq4g256_residual_v3_gfx942";
                self.ensure_kernel(kname, kernels::GEMV_HFQ4G256_RESIDUAL_V3_GFX942_SRC, kname)?;
                let grid = ((m as u32) + 7) / 8;
                self.launch_maybe_blob(kname, [grid, 1, 1], [256, 1, 1], 0, &mut params, || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(a_ptr);
                    b.push_ptr(x_ptr);
                    b.push_ptr(y_ptr);
                    b.push_i32(m_val);
                    b.push_i32(k_val);
                    b
                })
            } else if self.arch_caps.is_cdna3() && self.flags.gfx942_gemv_v2.unwrap_or(true) {
                let kname = "gemv_hfq4g256_residual_v2_gfx942";
                self.ensure_kernel(kname, kernels::GEMV_HFQ4G256_RESIDUAL_V2_GFX942_SRC, kname)?;
                let grid = ((m as u32) + 3) / 4;
                self.launch_maybe_blob(kname, [grid, 1, 1], [128, 1, 1], 0, &mut params, || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(a_ptr);
                    b.push_ptr(x_ptr);
                    b.push_ptr(y_ptr);
                    b.push_i32(m_val);
                    b.push_i32(k_val);
                    b
                })
            } else if self.arch_caps.has_cdna3_lds_gemv()
                && !self.arch_caps.gemv_prefetch_enabled()
                && (k as u32) * 4 <= 32768
            {
                let kname = "gemv_hfq4g256_residual_gfx942";
                self.ensure_kernel(kname, kernels::GEMV_HFQ4G256_RESIDUAL_GFX942_SRC, kname)?;
                let grid = ((m as u32) + 7) / 8;
                let lds_bytes = (k as u32) * 4;
                self.launch_maybe_blob(
                    kname,
                    [grid, 1, 1],
                    [256, 1, 1],
                    lds_bytes,
                    &mut params,
                    || {
                        let mut b = hip_bridge::KernargBlob::new();
                        b.push_ptr(a_ptr);
                        b.push_ptr(x_ptr);
                        b.push_ptr(y_ptr);
                        b.push_i32(m_val);
                        b.push_i32(k_val);
                        b
                    },
                )
            } else {
                let (kname, ksrc): (&str, &str) = if self.arch_caps.gemv_prefetch_enabled() {
                    (
                        "gemv_hfq4g256_residual_wave64_prefetch",
                        kernels::GEMV_HFQ4G256_RESIDUAL_WAVE64_PREFETCH_SRC,
                    )
                } else {
                    (
                        "gemv_hfq4g256_residual_wave64",
                        kernels::GEMV_HFQ4G256_RESIDUAL_WAVE64_SRC,
                    )
                };
                self.ensure_kernel(kname, ksrc, kname)?;
                let grid = ((m as u32) + 1) / 2;
                self.launch_maybe_blob(kname, [grid, 1, 1], [64, 1, 1], 0, &mut params, || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(a_ptr);
                    b.push_ptr(x_ptr);
                    b.push_ptr(y_ptr);
                    b.push_i32(m_val);
                    b.push_i32(k_val);
                    b
                })
            }
        } else if use_multirow {
            let (func_name, grid_div) = match rows {
                2 => ("gemv_hfq4g256_residual_multirow_r2", 2u32),
                4 => ("gemv_hfq4g256_residual_multirow_r4", 4u32),
                8 => ("gemv_hfq4g256_residual_multirow_r8", 8u32),
                _ => unreachable!(),
            };
            self.ensure_kernel(
                "gemv_hfq4g256_residual_multirow_rdna3",
                kernels::GEMV_HFQ4G256_RESIDUAL_MULTIROW_GFX1100_SRC,
                func_name,
            )?;
            let grid = ((m as u32) + grid_div - 1) / grid_div;
            self.launch_maybe_blob(func_name, [grid, 1, 1], [32, 1, 1], 0, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            })
        } else {
            self.launch_maybe_blob(
                "gemv_hfq4g256_residual",
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(a_ptr);
                    b.push_ptr(x_ptr);
                    b.push_ptr(y_ptr);
                    b.push_i32(m_val);
                    b.push_i32(k_val);
                    b
                },
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// HFQ4-G256 GEMV with fused SCALED residual add, CPU-scalar variant:
    ///   y[row] += scale * (A[row] · x)
    /// where `scale` is host-supplied by kernarg. Replaces the three-kernel
    /// tail of the MoE routed-expert epilogue (gemv → scale → add_inplace)
    /// with a single launch. Bit-exact with gemv_hfq4g256_residual followed
    /// by scaled_add_inplace_cpu_scalar when the inputs are identical —
    /// same accumulator layout, same pairwise combine.
    pub fn gemv_hfq4g256_residual_scaled_cpu(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        scale: f32,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_residual_scaled",
            kernels::GEMV_HFQ4G256_RESIDUAL_SCALED_SRC,
            "gemv_hfq4g256_residual_scaled_cpu",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let s_val = scale;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &s_val as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_residual_scaled_cpu",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_residual_scaled_cpu",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_f32(s_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// HFQ4-G256 GEMV with fused SCALED residual add, GPU-scalar variant:
    ///   y[row] += c_buf[0] * (A[row] · x)
    /// Reads the scale from a 1-element device buffer. Used by the MoE
    /// shared-expert epilogue where `c_buf` holds sigmoid(gate · x) computed
    /// entirely on-device, avoiding a D2H sync.
    pub fn gemv_hfq4g256_residual_scaled_gpu(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        c_buf: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_residual_scaled",
            kernels::GEMV_HFQ4G256_RESIDUAL_SCALED_SRC,
            "gemv_hfq4g256_residual_scaled_gpu",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let c_ptr = c_buf.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &c_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_residual_scaled_gpu",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_residual_scaled_gpu",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_ptr(c_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Same as `gemv_hfq4g256_residual_scaled_gpu` but applies sigmoid to
    /// `c_buf[0]` before scaling — lets the caller skip a separate
    /// `sigmoid_f32` launch on the 1-elem shared-expert gate scalar.
    /// Used by the A3B MoE FFN shared-expert down path.
    pub fn gemv_hfq4g256_residual_sigmoid_scaled_gpu(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        c_buf: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_residual_scaled",
            kernels::GEMV_HFQ4G256_RESIDUAL_SCALED_SRC,
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let c_ptr = c_buf.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &c_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_ptr(c_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// N-batched variant of `gemv_hfq4g256_residual_sigmoid_scaled_gpu`.
    /// `x_batch` is [N × K], `y_batch` is [N × M], `c_batch` is [N]. Each
    /// (row, token) block runs the HFQ4G256 GEMV body on its token's x
    /// row and atomicAdd's `sigmoid(c_batch[token]) * acc` into
    /// `y_batch[token × M + row]`. Used by the batched MoE FFN shared-
    /// expert down projection to eliminate N per-token launches.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched(
        &mut self,
        a_raw: &GpuTensor,
        x_batch: &GpuTensor,
        y_batch: &GpuTensor,
        c_batch: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_residual_scaled",
            kernels::GEMV_HFQ4G256_RESIDUAL_SCALED_SRC,
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x_batch.buf.as_ptr();
        let y_ptr = y_batch.buf.as_ptr();
        let c_ptr = c_batch.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &c_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched",
            [m as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_ptr(c_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// HFQ3/MQ3 analogue of `gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched`.
    /// Same grid shape, but reads HFQ3's 104 B / group layout. MQ3G256 shares
    /// storage with HFQ3G256; callers apply the FWHT rotation upstream.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq3g256_residual_sigmoid_scaled_gpu_batched(
        &mut self,
        a_raw: &GpuTensor,
        x_batch: &GpuTensor,
        y_batch: &GpuTensor,
        c_batch: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq3g256_residual_sigmoid_scaled",
            kernels::GEMV_HFQ3G256_RESIDUAL_SIGMOID_SCALED_SRC,
            "gemv_hfq3g256_residual_sigmoid_scaled_gpu_batched",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x_batch.buf.as_ptr();
        let y_ptr = y_batch.buf.as_ptr();
        let c_ptr = c_batch.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &c_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let groups = k / 256;
        let weight_bytes = m * groups * 104;
        let bytes = batch_size * (weight_bytes + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq3g256_residual_sigmoid_scaled_gpu_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq3g256_residual_sigmoid_scaled_gpu_batched",
            [m as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_ptr(c_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// HFQ4-G128 batched GEMV with fused per-token sigmoid-scaled residual.
    ///
    /// y_batch[token, row] += sigmoid(c_batch[token]) * (A[row] · x_batch[token])
    ///
    /// HFQ4-G128 layout: 72 bytes per 128-element group (vs HFQ4-G256's
    /// 136 B/256-element group). Used by the PARO shared-expert down
    /// dispatch in `prefill_moe_ffn_body_batched` (Phase 2 — admit gated
    /// behind HIPFIRE_PARO_BATCHED=1). Same grid/block contract as the
    /// HFQ4-G256 sister: grid=[M × batch_size × 1], block=[32 × 1 × 1].
    pub fn gemv_hfq4g128_residual_sigmoid_scaled_gpu_batched(
        &mut self,
        a_raw: &GpuTensor,
        x_batch: &GpuTensor,
        y_batch: &GpuTensor,
        c_batch: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g128_residual_sigmoid_scaled",
            kernels::GEMV_HFQ4G128_RESIDUAL_SIGMOID_SCALED_SRC,
            "gemv_hfq4g128_residual_sigmoid_scaled_gpu_batched",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x_batch.buf.as_ptr();
        let y_ptr = y_batch.buf.as_ptr();
        let c_ptr = c_batch.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &c_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * (crate::profile::gemv_hfq4g128_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g128_residual_sigmoid_scaled_gpu_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq4g128_residual_sigmoid_scaled_gpu_batched",
            [m as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_ptr(c_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// HFQ6/MQ6 analogue of `gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched`.
    /// Same kernel shape (grid = `M × batch`, block = 32, one warp per
    /// `(row, token)`), but reads HFQ6's 200 B / group layout (4 B scale +
    /// 4 B zero + 192 B packed 6-bit nibbles). MQ6G256 shares storage with
    /// HFQ6G256 — caller applies the FWHT rotation upstream, same convention
    /// as MQ4 / HFQ4. Used by the batched MoE FFN shared-expert `down`
    /// projection in the AWQ-style mixed-precision path where shared.down
    /// is MQ6 (12 of 40 layers in AWQ A3B fall into this case).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched(
        &mut self,
        a_raw: &GpuTensor,
        x_batch: &GpuTensor,
        y_batch: &GpuTensor,
        c_batch: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq6g256_residual_sigmoid_scaled",
            kernels::GEMV_HFQ6G256_RESIDUAL_SIGMOID_SCALED_SRC,
            "gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x_batch.buf.as_ptr();
        let y_ptr = y_batch.buf.as_ptr();
        let c_ptr = c_batch.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &c_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        // HFQ6 weight footprint: m * (k / 256) * 200 bytes per row + 4 B per
        // input/output cell. No dedicated profile helper yet (HFQ6 GEMV
        // currently doesn't appear in profile.rs); inlined here.
        let groups = k / 256;
        let weight_bytes = m * groups * 200;
        let bytes = batch_size * (weight_bytes + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched",
            [m as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_ptr(c_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn gemv_moe_scalar_residual_sigmoid_scaled_batched(
        &mut self,
        kernel_name: &'static str,
        weight_stride: usize,
        a_raw: &GpuTensor,
        x_batch: &GpuTensor,
        y_batch: &GpuTensor,
        c_batch: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            kernel_name,
            kernels::MOE_MQ_GFX1151_SCALAR_BATCHED_SRC,
            kernel_name,
        )?;
        let ap = a_raw.buf.as_ptr();
        let xp = x_batch.buf.as_ptr();
        let yp = y_batch.buf.as_ptr();
        let cp = c_batch.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * (m * (k / 256) * weight_stride + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [m as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_ptr(cp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemv_moe_scalar_gate_up_indexed_batched(
        &mut self,
        kernel_name: &'static str,
        weight_stride: usize,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            kernel_name,
            kernels::MOE_MQ_GFX1151_SCALAR_BATCHED_SRC,
            kernel_name,
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * k_top * (m * (k / 256) * weight_stride + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(ygp);
                b.push_ptr(yup);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemv_moe_scalar_down_indexed_batched_expanded(
        &mut self,
        kernel_name: &'static str,
        weight_stride: usize,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        rot_batch: &GpuTensor,
        expert_outputs: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            kernel_name,
            kernels::MOE_MQ_GFX1151_SCALAR_BATCHED_SRC,
            kernel_name,
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = rot_batch.buf.as_ptr();
        let yp = expert_outputs.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * k_top * (m * (k / 256) * weight_stride + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq2g256_residual_sigmoid_scaled_gpu_batched(
        &mut self,
        a_raw: &GpuTensor,
        x_batch: &GpuTensor,
        y_batch: &GpuTensor,
        c_batch: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_moe_scalar_residual_sigmoid_scaled_batched(
            "gemv_hfq2g256_residual_sigmoid_scaled_gpu_batched",
            72,
            a_raw,
            x_batch,
            y_batch,
            c_batch,
            m,
            k,
            batch_size,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq8g256_residual_sigmoid_scaled_gpu_batched(
        &mut self,
        a_raw: &GpuTensor,
        x_batch: &GpuTensor,
        y_batch: &GpuTensor,
        c_batch: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_moe_scalar_residual_sigmoid_scaled_batched(
            "gemv_hfq8g256_residual_sigmoid_scaled_gpu_batched",
            258,
            a_raw,
            x_batch,
            y_batch,
            c_batch,
            m,
            k,
            batch_size,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_mq2g256_lloyd_residual_sigmoid_scaled_gpu_batched(
        &mut self,
        a_raw: &GpuTensor,
        x_batch: &GpuTensor,
        y_batch: &GpuTensor,
        c_batch: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_moe_scalar_residual_sigmoid_scaled_batched(
            "gemv_mq2g256_lloyd_residual_sigmoid_scaled_gpu_batched",
            72,
            a_raw,
            x_batch,
            y_batch,
            c_batch,
            m,
            k,
            batch_size,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_mq3g256_lloyd_residual_sigmoid_scaled_gpu_batched(
        &mut self,
        a_raw: &GpuTensor,
        x_batch: &GpuTensor,
        y_batch: &GpuTensor,
        c_batch: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_moe_scalar_residual_sigmoid_scaled_batched(
            "gemv_mq3g256_lloyd_residual_sigmoid_scaled_gpu_batched",
            112,
            a_raw,
            x_batch,
            y_batch,
            c_batch,
            m,
            k,
            batch_size,
        )
    }
    /// MoE fused gate_up GEMV: runs 8 top-K experts' HFQ4-G256 GEMV in a
    /// single launch. Caller passes the 8 selected experts' weight
    /// tensors (in top-K order); the kernel's grid.y picks which expert
    /// each block uses. Outputs are SPLIT into `y_gate` (first mi rows of
    /// each expert) and `y_up` (second mi rows), both `[k_top × mi]`
    /// row-major, so the next-stage batched silu_mul_rotate can consume
    /// them as plain [batch × K] buffers without extra strided reads.
    ///
    /// Bit-exact with running `gemv_hfq4g256` 8 times (same accumulator
    /// layout and pairwise final combine). `k_top` is currently hardcoded
    /// to 8 to match A3B; a generic path can follow alongside Phase 2b.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_gate_up_k8(
        &mut self,
        w0: &GpuTensor,
        w1: &GpuTensor,
        w2: &GpuTensor,
        w3: &GpuTensor,
        w4: &GpuTensor,
        w5: &GpuTensor,
        w6: &GpuTensor,
        w7: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor, // [k_top × mi] — first half
        y_up: &GpuTensor,   // [k_top × mi] — second half
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_moe_gate_up",
            kernels::GEMV_HFQ4G256_MOE_GATE_UP_SRC,
            "gemv_hfq4g256_moe_gate_up_k8",
        )?;
        let w0p = w0.buf.as_ptr();
        let w1p = w1.buf.as_ptr();
        let w2p = w2.buf.as_ptr();
        let w3p = w3.buf.as_ptr();
        let w4p = w4.buf.as_ptr();
        let w5p = w5.buf.as_ptr();
        let w6p = w6.buf.as_ptr();
        let w7p = w7.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &w0p as *const _ as *mut c_void,
            &w1p as *const _ as *mut c_void,
            &w2p as *const _ as *mut c_void,
            &w3p as *const _ as *mut c_void,
            &w4p as *const _ as *mut c_void,
            &w5p as *const _ as *mut c_void,
            &w6p as *const _ as *mut c_void,
            &w7p as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        // Bandwidth: 8× weight, x read 8× (cached in practice), 8×m writes.
        let bytes = 8 * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer =
            crate::profile::begin_timer(&self.hip, "gemv", "gemv_hfq4g256_moe_gate_up_k8", bytes);
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_moe_gate_up_k8",
            [m as u32, 8, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(w0p);
                b.push_ptr(w1p);
                b.push_ptr(w2p);
                b.push_ptr(w3p);
                b.push_ptr(w4p);
                b.push_ptr(w5p);
                b.push_ptr(w6p);
                b.push_ptr(w7p);
                b.push_ptr(xp);
                b.push_ptr(ygp);
                b.push_ptr(yup);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// MoE fused down GEMV with scaled residual: accumulates 8 top-K
    /// experts' weighted contributions into `x_residual` in a single
    /// kernel launch. Grid.y selects the expert; each block atomicAdds
    /// `s_rank * (W_rank[row] · rot_batch[rank, :])` into `x_residual[row]`.
    /// Replaces 8 separate `gemv_hfq4g256_residual_scaled_cpu` calls.
    ///
    /// Atomic-add summation order is non-deterministic, so bit-exactness
    /// across runs isn't guaranteed (vs the sequential per-expert path).
    /// For A3B the MoE contribution is added on top of a non-trivial base,
    /// so the ordering-dependent FP noise is tiny in practice and the
    /// smoke-test decode still matches the Phase 2c step 2 output.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_down_residual_scaled_k8(
        &mut self,
        w0: &GpuTensor,
        w1: &GpuTensor,
        w2: &GpuTensor,
        w3: &GpuTensor,
        w4: &GpuTensor,
        w5: &GpuTensor,
        w6: &GpuTensor,
        w7: &GpuTensor,
        rot_batch: &GpuTensor,
        x_residual: &GpuTensor,
        scales: [f32; 8],
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_moe_down",
            kernels::GEMV_HFQ4G256_MOE_DOWN_SRC,
            "gemv_hfq4g256_moe_down_residual_scaled_k8",
        )?;
        let w0p = w0.buf.as_ptr();
        let w1p = w1.buf.as_ptr();
        let w2p = w2.buf.as_ptr();
        let w3p = w3.buf.as_ptr();
        let w4p = w4.buf.as_ptr();
        let w5p = w5.buf.as_ptr();
        let w6p = w6.buf.as_ptr();
        let w7p = w7.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let xrp = x_residual.buf.as_ptr();
        let [s0, s1, s2, s3, s4, s5, s6, s7] = scales;
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &w0p as *const _ as *mut c_void,
            &w1p as *const _ as *mut c_void,
            &w2p as *const _ as *mut c_void,
            &w3p as *const _ as *mut c_void,
            &w4p as *const _ as *mut c_void,
            &w5p as *const _ as *mut c_void,
            &w6p as *const _ as *mut c_void,
            &w7p as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &s0 as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &s3 as *const _ as *mut c_void,
            &s4 as *const _ as *mut c_void,
            &s5 as *const _ as *mut c_void,
            &s6 as *const _ as *mut c_void,
            &s7 as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = 8 * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_moe_down_residual_scaled_k8",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_moe_down_residual_scaled_k8",
            [m as u32, 8, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(w0p);
                b.push_ptr(w1p);
                b.push_ptr(w2p);
                b.push_ptr(w3p);
                b.push_ptr(w4p);
                b.push_ptr(w5p);
                b.push_ptr(w6p);
                b.push_ptr(w7p);
                b.push_ptr(rbp);
                b.push_ptr(xrp);
                b.push_f32(s0);
                b.push_f32(s1);
                b.push_f32(s2);
                b.push_f32(s3);
                b.push_f32(s4);
                b.push_f32(s5);
                b.push_f32(s6);
                b.push_f32(s7);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Index-aware MoE gate_up GEMV. Reads expert_ids from a device-side
    /// topk_indices buffer and weight bases from expert_ptrs[expert_id].
    /// hipGraph-capture-safe replacement for the kernarg-pointer variant.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_gate_up_k8_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,  // [n_exp] of u64 device pointers
        topk_indices: &GpuTensor, // [k_top] i32
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let two_row = self.arch_caps.is_wave64_native() || self.gfx1151_moe_indexed_2row_enabled();
        let (func_name, block, grid_x) = if two_row {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_gate_up_indexed_wave64",
                kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_WAVE64_SRC,
                "gemv_hfq4g256_moe_gate_up_k8_indexed_wave64",
            )?;
            (
                "gemv_hfq4g256_moe_gate_up_k8_indexed_wave64",
                [64u32, 1, 1],
                ((m as u32) + 1) / 2,
            )
        } else {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_gate_up_indexed",
                kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_SRC,
                "gemv_hfq4g256_moe_gate_up_k8_indexed",
            )?;
            (
                "gemv_hfq4g256_moe_gate_up_k8_indexed",
                [32u32, 1, 1],
                m as u32,
            )
        };
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = 8 * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_moe_gate_up_k8_indexed",
            bytes,
        );
        let result =
            self.launch_maybe_blob(func_name, [grid_x, 8, 1], block, 0, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(ygp);
                b.push_ptr(yup);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// HFQ4G128 (ParoQuant) variant of the indexed MoE gate_up GEMV.
    /// wave32-only (gfx10/11/12) — no wave64 path yet because ParoQuant
    /// A3B is not currently validated on gfx94x.
    pub fn gemv_paro_q4g128_moe_gate_up_k8_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,  // [n_exp] of u64 device pointers
        topk_indices: &GpuTensor, // [k_top] i32
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_paro_q4g128_moe_gate_up_indexed",
            kernels::GEMV_PARO_Q4G128_MOE_GATE_UP_INDEXED_SRC,
            "gemv_paro_q4g128_moe_gate_up_k8_indexed",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = 8 * (crate::profile::gemv_hfq4g128_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_paro_q4g128_moe_gate_up_k8_indexed",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_paro_q4g128_moe_gate_up_k8_indexed",
            [m as u32, 8, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(ygp);
                b.push_ptr(yup);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Index-aware MoE down GEMV with scaled residual. Same pattern as
    /// the indexed gate_up; also reads scales from a device topk_weights
    /// buffer and atomicAdds the contribution into x_residual.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_down_residual_scaled_k8_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        rot_batch: &GpuTensor,
        x_residual: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let (func_name, block, grid_x) = if cdna_wave64 {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_down_indexed_wave64",
                kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_WAVE64_SRC,
                "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_wave64",
            )?;
            (
                "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_wave64",
                [64u32, 1, 1],
                ((m as u32) + 1) / 2,
            )
        } else {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_down_indexed",
                kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_SRC,
                "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed",
            )?;
            (
                "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed",
                [32u32, 1, 1],
                m as u32,
            )
        };
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let wp = topk_weights.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let xrp = x_residual.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = 8 * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed",
            bytes,
        );
        let result =
            self.launch_maybe_blob(func_name, [grid_x, 8, 1], block, 0, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_ptr(rbp);
                b.push_ptr(xrp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// N-batched indexed MoE gate_up. Grid = (M, K_TOP, N). `x` is
    /// [N × K], `topk_indices` is [N × K_TOP] i32, `y_gate` and `y_up`
    /// are [N × K_TOP × MI] where MI = M / 2.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let two_row = self.arch_caps.is_wave64_native() || self.gfx1151_moe_indexed_2row_enabled();
        let (func_name, block, grid_div): (&str, [u32; 3], u32) = if two_row {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_gate_up_indexed_batched_wave64",
                kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_BATCHED_WAVE64_SRC,
                "gemv_hfq4g256_moe_gate_up_k8_indexed_batched_wave64",
            )?;
            (
                "gemv_hfq4g256_moe_gate_up_k8_indexed_batched_wave64",
                [64, 1, 1],
                2,
            )
        } else {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_gate_up_indexed_batched",
                kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_BATCHED_SRC,
                "gemv_hfq4g256_moe_gate_up_k8_indexed_batched",
            )?;
            (
                "gemv_hfq4g256_moe_gate_up_k8_indexed_batched",
                [32, 1, 1],
                1,
            )
        };
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * k_top * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_batched",
            bytes,
        );
        let grid_x = (m as u32 + grid_div - 1) / grid_div;
        let result = self.launch_maybe_blob(
            func_name,
            [grid_x, k_top as u32, batch_size as u32],
            block,
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(ygp);
                b.push_ptr(yup);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// N-batched indexed MoE down + scaled residual. Grid = (M, K_TOP, N).
    /// `rot_batch` is [N × K_TOP × K], `x_residual` is [N × M]; the kernel
    /// atomicAdd's per-token slices. `topk_indices` / `topk_weights` are
    /// [N × K_TOP].
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        rot_batch: &GpuTensor,
        x_residual: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let (func_name, block, grid_div): (&str, [u32; 3], u32) = if cdna_wave64 {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_down_indexed_batched_wave64",
                kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_BATCHED_WAVE64_SRC,
                "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched_wave64",
            )?;
            (
                "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched_wave64",
                [64, 1, 1],
                2,
            )
        } else {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_down_indexed_batched",
                kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_BATCHED_SRC,
                "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched",
            )?;
            (
                "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched",
                [32, 1, 1],
                1,
            )
        };
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let wp = topk_weights.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let xrp = x_residual.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * k_top * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched",
            bytes,
        );
        let grid_x = (m as u32 + grid_div - 1) / grid_div;
        let result = self.launch_maybe_blob(
            func_name,
            [grid_x, k_top as u32, batch_size as u32],
            block,
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_ptr(rbp);
                b.push_ptr(xrp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Atomic-free counterpart to
    /// `gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched`. Writes
    /// each (token, krank) result to its own row of `expert_outputs`
    /// ([N × K_TOP × M], f32) instead of atomicAdd'ing the scaled sum into
    /// `x_residual`. Pair with `moe_down_combine_k8_batched` to fold the
    /// K_TOP slots back into the residual with topk_weights applied.
    ///
    /// Observed lift on R9700/gfx1201: 387 → ~900 GiB/s for the down GEMV
    /// (no K_TOP-way atomic contention per output cell). Wave32-only
    /// (RDNA) for now — the CDNA wave64 path stays on the residual_scaled
    /// kernel; atomicAdd on HBM is faster there and the contention pattern
    /// is different.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        rot_batch: &GpuTensor,
        expert_outputs: &GpuTensor, // [batch_size × k_top × m] f32
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let two_row = self.gfx1151_moe_indexed_2row_enabled();
        let (kernel_key, kernel_src, func_name, block, grid_div): (
            &str,
            &str,
            &str,
            [u32; 3],
            u32,
        ) = if two_row {
            (
                "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_2row_gfx1151",
                kernels::GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_2ROW_GFX1151_SRC,
                "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_2row_gfx1151",
                [64, 1, 1],
                2,
            )
        } else {
            (
                "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded",
                kernels::GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_SRC,
                "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded",
                [32, 1, 1],
                1,
            )
        };
        self.ensure_kernel(kernel_key, kernel_src, func_name)?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let eop = expert_outputs.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &eop as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * k_top * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded",
            bytes,
        );
        let grid_x = (m as u32 + grid_div - 1) / grid_div;
        let result = self.launch_maybe_blob(
            func_name,
            [grid_x, k_top as u32, batch_size as u32],
            block,
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(rbp);
                b.push_ptr(eop);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// HFQ4G128 (ParoQuant) variant of the atomic-free batched indexed
    /// MoE down. Same expanded-output contract as the HFQ4G256 sibling;
    /// caller must follow with `moe_down_combine_k8_batched` to fold the
    /// K_TOP slots into x_residual with topk_weights applied. wave32-only.
    #[allow(clippy::too_many_arguments)]
    /// N-batched indexed MoE gate_up GEMV for HFQ4G128 (ParoQuant routed
    /// experts). Sister of `gemv_hfq4g256_moe_gate_up_k8_indexed_batched`
    /// with 72 B/group stride. The caller MUST pre-rotate x using the
    /// layer's shared `gate_up` Givens sidecar (givens_rotate_to into
    /// x_rot_batch) before calling — this kernel is rotation-agnostic and
    /// just reads HFQ4G128 nibbles. Grid: (M, K_TOP, N) wave32.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_paro_q4g128_moe_gate_up_k8_indexed_batched(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_paro_q4g128_moe_gate_up_k8_indexed_batched",
            kernels::GEMV_PARO_Q4G128_MOE_GATE_UP_K8_INDEXED_BATCHED_SRC,
            "gemv_paro_q4g128_moe_gate_up_k8_indexed_batched",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * k_top * (crate::profile::gemv_hfq4g128_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_paro_q4g128_moe_gate_up_k8_indexed_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_paro_q4g128_moe_gate_up_k8_indexed_batched",
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(ygp);
                b.push_ptr(yup);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn gemv_paro_q4g128_moe_down_k8_indexed_batched(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        rot_batch: &GpuTensor,
        expert_outputs: &GpuTensor, // [batch_size × k_top × m] f32
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_paro_q4g128_moe_down_k8_indexed_batched",
            kernels::GEMV_PARO_Q4G128_MOE_DOWN_K8_INDEXED_BATCHED_SRC,
            "gemv_paro_q4g128_moe_down_k8_indexed_batched",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let eop = expert_outputs.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &eop as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * k_top * (crate::profile::gemv_hfq4g128_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_paro_q4g128_moe_down_k8_indexed_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_paro_q4g128_moe_down_k8_indexed_batched",
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(rbp);
                b.push_ptr(eop);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Index-aware MoE gate_up GEMV for HFQ6G256-layout routed experts.
    /// Wave32 (RDNA) only — CDNA wave64 path stays on the residual_scaled
    /// kernel family. Used to keep mixed-kmap A3B (post-PR-199 alternating
    /// MQ4→MQ6 promotion) on the device-side top-K path under hipGraph
    /// capture.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq6g256_moe_gate_up_k8_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq6g256_moe_gate_up_indexed",
            kernels::GEMV_HFQ6G256_MOE_GATE_UP_INDEXED_SRC,
            "gemv_hfq6g256_moe_gate_up_k8_indexed",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        // HFQ6 uses 200 bytes/group vs HFQ4's 136. Bytes estimate scales
        // accordingly. Reuse the existing profile helper with a 200/136
        // ratio so timer estimates are roughly correct.
        let hfq4_bytes = crate::profile::gemv_hfq4g256_bytes(m, k);
        let bytes = 8 * (hfq4_bytes * 200 / 136 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq6g256_moe_gate_up_k8_indexed",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq6g256_moe_gate_up_k8_indexed",
            [m as u32, 8, 1],
            [32u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(ygp);
                b.push_ptr(yup);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// HFQ6G256 counterpart to `gemv_hfq4g256_moe_gate_up_k8_indexed_batched`.
    /// N-batched indexed MoE gate_up GEMV for 6-bit (200 B/group) routed
    /// experts. Same kernarg + grid (M, K_TOP, N) + gate/up output split as
    /// the HFQ4 sibling; only the per-group dequant differs (handled inside
    /// the kernel). Wave32 (RDNA) only — no wave64 variant.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq6g256_moe_gate_up_k8_indexed_batched(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq6g256_moe_gate_up_indexed_batched",
            kernels::GEMV_HFQ6G256_MOE_GATE_UP_INDEXED_BATCHED_SRC,
            "gemv_hfq6g256_moe_gate_up_k8_indexed_batched",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        // 200 vs 136 B/group: scale the HFQ4 byte estimate by 200/136.
        let hfq4_bytes = crate::profile::gemv_hfq4g256_bytes(m, k);
        let bytes = batch_size * k_top * (hfq4_bytes * 200 / 136 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq6g256_moe_gate_up_k8_indexed_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq6g256_moe_gate_up_k8_indexed_batched",
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(ygp);
                b.push_ptr(yup);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// HFQ6G256 counterpart to `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded`.
    /// Atomic-free expand-then-combine for the MoE down step. Pairs with
    /// `moe_down_combine_k8_batched` (dtype-independent — operates on the
    /// f32 expanded buffer). Wave32 (RDNA) only.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        rot_batch: &GpuTensor,
        expert_outputs: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq6g256_moe_down_k8_indexed_batched_expanded",
            kernels::GEMV_HFQ6G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_SRC,
            "gemv_hfq6g256_moe_down_k8_indexed_batched_expanded",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let eop = expert_outputs.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &eop as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let hfq4_bytes = crate::profile::gemv_hfq4g256_bytes(m, k);
        let bytes = batch_size * k_top * (hfq4_bytes * 200 / 136 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq6g256_moe_down_k8_indexed_batched_expanded",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq6g256_moe_down_k8_indexed_batched_expanded",
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(rbp);
                b.push_ptr(eop);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// y = A_q8_0 * x (quantized GEMV for Q8_0)
    pub fn gemv_q8_0(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(a_ptr);
            b.push_ptr(x_ptr);
            b.push_ptr(y_ptr);
            b.push_i32(m_val);
            b.push_i32(k_val);
            b
        };

        // Adaptive dispatch: wide kernel for small K (more threads per row),
        // narrow kernel for large K (more blocks, better occupancy).
        if k <= 1536 {
            self.ensure_kernel(
                "gemv_q8_0_wide",
                kernels::GEMV_Q8_0_WIDE_SRC,
                "gemv_q8_0_wide",
            )?;
            let block_size = 64u32; // 2 warps, each processes one row
            let grid = ((m + 1) / 2) as u32; // ceil(M/2)
            return self.launch_maybe_blob(
                "gemv_q8_0_wide",
                [grid, 1, 1],
                [block_size, 1, 1],
                0,
                &mut params,
                blob_builder,
            );
        }

        self.ensure_kernel("gemv_q8_0", kernels::GEMV_Q8_0_SRC, "gemv_q8_0")?;
        let block_size = 32u32;
        self.launch_maybe_blob(
            "gemv_q8_0",
            [m as u32, 1, 1],
            [block_size, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }
    /// y = A_q8hfq * x (split-metadata Q8 GEMV, row_stride = padded row bytes)
    pub fn gemv_q8hfq(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        row_stride: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut rs_val = row_stride as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut rs_val as *mut _ as *mut c_void,
        ];

        if k <= 1536 {
            self.ensure_kernel(
                "gemv_q8hfq_wide",
                kernels::GEMV_Q8HFQ_WIDE_SRC,
                "gemv_q8hfq_wide",
            )?;
            let func = &self.functions["gemv_q8hfq_wide"];
            let block_size = 64u32;
            let grid = ((m + 1) / 2) as u32;
            return unsafe {
                self.hip
                    .launch_kernel(func, [grid, 1, 1], [block_size, 1, 1], 0, None, &mut params)
            };
        }

        self.ensure_kernel("gemv_q8hfq", kernels::GEMV_Q8HFQ_SRC, "gemv_q8hfq")?;
        let func = &self.functions["gemv_q8hfq"];
        unsafe {
            self.hip
                .launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, None, &mut params)
        }
    }
    /// y = A_q6k * x (quantized GEMV for Q6_K)
    pub fn gemv_q6k(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_q6k", kernels::GEMV_Q6K_SRC, "gemv_q6k")?;
        let func = &self.functions["gemv_q6k"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 256u32;
        let shared_mem = block_size * 4;
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// y = A_q4f16 * x (RDNA-native Q4_F16 GEMV, group size 64)
    /// a_raw: raw Q4_F16_G64 bytes on GPU, x: F32 input, y: F32 output
    /// Block: 36 bytes per 64 elements. K must be multiple of 64.
    /// Uses 128 threads (4 warps) with shared memory reduction for increased MLP.
    pub fn gemv_q4f16_g64(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_q4f16_g64",
            kernels::GEMV_Q4F16_G64_SRC,
            "gemv_q4f16_g64",
        )?;
        let func = &self.functions["gemv_q4f16_g64"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 32u32; // single warp — no shared memory
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// y = A_q4f16 * x (256-thread wide variant for occupancy testing)
    /// Element-strided access pattern matching F32 GEMV. Shared memory reduction.
    pub fn gemv_q4f16_g64_wide(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_q4f16_g64_wide",
            kernels::GEMV_Q4F16_G64_WIDE_SRC,
            "gemv_q4f16_g64_wide",
        )?;
        let func = &self.functions["gemv_q4f16_g64_wide"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 256u32;
        let shared_mem = block_size * 4; // one float per thread
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// y = A_q4f16 * x (RDNA-native Q4_F16 GEMV, group size 32)
    /// Block: 20 bytes per 32 elements. K must be multiple of 32.
    pub fn gemv_q4f16_g32(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_q4f16_g32",
            kernels::GEMV_Q4F16_G32_SRC,
            "gemv_q4f16_g32",
        )?;
        let func = &self.functions["gemv_q4f16_g32"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 32u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Generic GEMV F16×F16 → F32: `w` [M,K], `x` [K], `y` [M]. gfx1103 wave32.
    pub fn gemv_f16_f32(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.launch_gemv_generic("gemv_f16_f32", kernels::GEMV_F16_F32_SRC, w, x, y, m, k)
    }
    /// Generic GEMV F16×F16 → F16: `w` [M,K], `x` [K], `y` [M]. gfx1103 wave32.
    pub fn gemv_f16_f16(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.launch_gemv_generic("gemv_f16_f16", kernels::GEMV_F16_F16_SRC, w, x, y, m, k)
    }
    /// Generic GEMV BF16×BF16 → F32: `w` [M,K], `x` [K], `y` [M]. gfx1103 wave32.
    pub fn gemv_bf16_f32(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.launch_gemv_generic("gemv_bf16_f32", kernels::GEMV_BF16_F32_SRC, w, x, y, m, k)
    }
    /// Generic GEMV BF16×BF16 → BF16: `w` [M,K], `x` [K], `y` [M]. gfx1103 wave32.
    pub fn gemv_bf16_bf16(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.launch_gemv_generic("gemv_bf16_bf16", kernels::GEMV_BF16_BF16_SRC, w, x, y, m, k)
    }
    /// Generic GEMV signed-INT8×INT8 → INT32: `w` [M,K], `x` [K], `y` [M].
    pub fn gemv_iu8_i32(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.launch_gemv_generic("gemv_iu8_i32", kernels::GEMV_IU8_I32_SRC, w, x, y, m, k)
    }
    /// Generic GEMV signed-INT4×INT4 → INT32: `w` [M,K/2], `x` [K/2] (packed
    /// nibbles), `y` [M]. `k` is the logical K. gfx1103 wave32.
    pub fn gemv_iu4_i32(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.launch_gemv_generic("gemv_iu4_i32", kernels::GEMV_IU4_I32_SRC, w, x, y, m, k)
    }
    /// Opus Quant W4A4 DECODE GEMV (batch=1): one wave32 per output row, no WMMA
    /// N-tile waste. `w_i4`/`w_scales` are the combined-buffer base + sub_offset
    /// scale view (same as gemm_oq4_grouped_wmma); `x_i4`/`x_scales` are the
    /// int4-quantized activation. `group` must be 256.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_oq4_grouped(
        &mut self,
        w_i4: &GpuTensor,
        w_scales: &GpuTensor,
        x_i4: &GpuTensor,
        x_scales: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(group, 256, "gemv_oq4_grouped: group must be 256");
        assert_eq!(
            k % group,
            0,
            "gemv_oq4_grouped: K must be a multiple of group"
        );
        self.ensure_kernel(
            "gemv_oq4_grouped",
            kernels::GEMV_OQ4_GROUPED_SRC,
            "gemv_oq4_grouped",
        )?;
        let wp = w_i4.buf.as_ptr();
        let wsp = w_scales.buf.as_ptr();
        let xp = x_i4.buf.as_ptr();
        let xsp = x_scales.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut gi = group as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &wsp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
        ];
        let func = &self.functions["gemv_oq4_grouped"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// y = A_q8_0 * x (quantized GEMV for Q8_0)
    /// F16-weight × F32-input GEMV. y[m] = W_f16[m, k] @ x_f32[k].
    /// Keeps full F32 input precision — use this for full-precision F16
    /// weights instead of the WMMA F16×F16 path (which converts the F32
    /// input to F16 first, losing ~13 bits of mantissa).
    pub fn gemv_f16_xf32(
        &mut self,
        weight: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_f16_xf32", kernels::GEMV_F16_XF32_SRC, "gemv_f16_xf32")?;

        let w_ptr = weight.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &w_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(w_ptr);
            b.push_ptr(x_ptr);
            b.push_ptr(y_ptr);
            b.push_i32(m_val);
            b.push_i32(k_val);
            b
        };

        let r = self.launch_maybe_blob(
            "gemv_f16_xf32",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            blob_builder,
        );
        self.maybe_capture_activation(weight, x, 1, k);
        r
    }
    /// F16-weight × F32-input GEMV with fused residual add.
    pub fn gemv_f16_xf32_residual(
        &mut self,
        weight: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_f16_xf32_residual",
            kernels::GEMV_F16_XF32_RESIDUAL_SRC,
            "gemv_f16_xf32_residual",
        )?;

        let w_ptr = weight.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &w_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "gemv_f16_xf32_residual",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(w_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        )
    }
    /// Batched F16-weight × F32-input GEMV with fused residual add.
    pub fn gemv_f16_xf32_residual_batched(
        &mut self,
        weight: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_f16_xf32_residual_batched",
            kernels::GEMV_F16_XF32_RESIDUAL_BATCHED_SRC,
            "gemv_f16_xf32_residual_batched",
        )?;

        let w_ptr = weight.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let b_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &w_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &b_val as *const _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "gemv_f16_xf32_residual_batched",
            [m as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(w_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(b_val);
                b
            },
        )
    }
}
