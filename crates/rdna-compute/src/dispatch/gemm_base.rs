// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Base-dtype GEMM dispatch (f16/bf16/f32, incl. WMMA + train). Pure move (Phase 1 M6).

use super::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    /// gfx1151 raw FP16 routed-MoE grouped WMMA. Same scatter contract as
    /// `gemm_hfq4g256_moe_grouped_wmma_k2`; X is staged through FP16 scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16_moe_grouped_wmma_gfx1151(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch != "gfx1151" {
            panic!(
                "gemm_f16_moe_grouped_wmma_gfx1151: only gfx1151 is supported; arch={}",
                self.arch
            );
        }
        assert_eq!(
            k % 16,
            0,
            "gemm_f16_moe_grouped_wmma_gfx1151: K must be a multiple of 16"
        );
        self.ensure_kernel(
            "gemm_f16_moe_grouped_wmma_gfx1151",
            kernels::GEMM_F16_MOE_GROUPED_WMMA_GFX1151_SRC,
            "gemm_f16_moe_grouped_wmma_gfx1151",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x_src, x_src_rows * k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_f16_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 127) / 128) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let bytes = (m * k * 2) + (x_src_rows * k * 2) + (m_total * m * 4);
        let profile_name = if x_row_div > 1 {
            "gemm_f16_moe_grouped_wmma_gfx1151_gate_up"
        } else {
            "gemm_f16_moe_grouped_wmma_gfx1151_down"
        };
        let timer = crate::profile::begin_timer(&self.hip, "gemm", profile_name, bytes);
        let result = self.launch_maybe_blob(
            "gemm_f16_moe_grouped_wmma_gfx1151",
            [row_tiles, slot_tiles, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// gfx1151 raw BF16 routed-MoE grouped WMMA. Same scatter contract as
    /// `gemm_f16_moe_grouped_wmma_gfx1151`; X is staged through BF16 scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_bf16_moe_grouped_wmma_gfx1151(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch != "gfx1151" {
            panic!(
                "gemm_bf16_moe_grouped_wmma_gfx1151: only gfx1151 is supported; arch={}",
                self.arch
            );
        }
        assert_eq!(
            k % 16,
            0,
            "gemm_bf16_moe_grouped_wmma_gfx1151: K must be a multiple of 16"
        );
        let use_m256 = std::env::var("HIPFIRE_BF16_MOE_M256").ok().as_deref() == Some("1");
        let kernel_name = if use_m256 {
            "gemm_bf16_moe_grouped_wmma_gfx1151_m256"
        } else {
            "gemm_bf16_moe_grouped_wmma_gfx1151"
        };
        self.ensure_kernel(
            kernel_name,
            kernels::GEMM_BF16_MOE_GROUPED_WMMA_GFX1151_SRC,
            kernel_name,
        )?;
        let x_bf16_ptr = self.ensure_bf16_x(x_src, x_src_rows * k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_bf16_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
        ];

        let row_tile = if use_m256 { 256usize } else { 128usize };
        let block_threads = if use_m256 { 512u32 } else { 256u32 };
        let row_tiles = ((m + row_tile - 1) / row_tile) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let bytes = (m * k * 2) + (x_src_rows * k * 2) + (m_total * m * 4);
        let profile_name = if x_row_div > 1 {
            if use_m256 {
                "gemm_bf16_moe_grouped_wmma_gfx1151_gate_up_m256"
            } else {
                "gemm_bf16_moe_grouped_wmma_gfx1151_gate_up"
            }
        } else {
            if use_m256 {
                "gemm_bf16_moe_grouped_wmma_gfx1151_down_m256"
            } else {
                "gemm_bf16_moe_grouped_wmma_gfx1151_down"
            }
        };
        let timer = crate::profile::begin_timer(&self.hip, "gemm", profile_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [block_threads, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// FP16-weight lm_head fast path for DFlash drafts that ship F16 (not
    /// quantized) weights. Routes through `gemm_mw16_residual_wmma` with the
    /// usual memset-then-atomicAdd residual pattern.
    ///
    /// Shape requirements: K must be a multiple of 32 (mw16 processes 32 K
    /// elements per WMMA iteration). All 27B/9B draft shapes satisfy this
    /// (hidden=5120, intermediate=17408, q_dim=4096, kv_dim=1024, fc-K=25600).
    ///
    /// Non-gfx11 falls through to row-by-row F16 GEMM so lm_head output keeps
    /// the `[batch, vocab]` layout expected by callers. Set
    /// `HIPFIRE_LM_HEAD_F16=f32` at load time to use the legacy F32-expanded
    /// storage and bypass this path entirely.
    pub fn gemm_f16_batched_lmhead(
        &mut self,
        w_f16: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Calibration capture: this is the F16/BF16 linear chokepoint the lowered
        // super-op path routes through (GemvFamily K::GemvF16), so capturing here
        // fires for the resident bf16 calibration forward. No-op when unarmed.
        self.maybe_capture_activation(w_f16, x, batch_size, k);
        if !self.arch_caps.has_wmma_w32() {
            // No mw16 WMMA on non-RDNA3. The generic F16 GEMM writes [M,N],
            // while lm_head consumers expect [N,M], so preserve layout by
            // launching one row at a time.
            for b in 0..batch_size {
                let x_row = x.sub_offset(b * k, k);
                let y_row = y.sub_offset(b * m, m);
                self.gemm_f16_tiled(w_f16, &x_row, &y_row, m, k, 1)?;
            }
            return Ok(());
        }
        self.ensure_kernel(
            "gemm_mw16_residual_wmma",
            kernels::GEMM_MW16_RESIDUAL_WMMA_SRC,
            "gemm_mw16_residual_wmma",
        )?;
        // Pre-zero Y (residual WMMA does y += acc) and force FP16-X reconversion
        // (the draft reuses the same scratch pointer every cycle with new data).
        self.fp16_x_source_ptr = std::ptr::null_mut();
        match self.active_stream.as_ref() {
            Some(stream) => self
                .hip
                .memset_async(&y.buf, 0, batch_size * m * 4, stream)?,
            None => self.hip.memset(&y.buf, 0, batch_size * m * 4)?,
        }
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let wp = w_f16.buf.as_ptr();
        let xp = x_f16_ptr;
        let yp = y.buf.as_ptr();
        let mi = m as i32;
        let ki = k as i32;
        let ni = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mi as *const _ as *mut c_void,
            &ki as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        let rows = ((m + 15) / 16) as u32;
        let batches = ((batch_size + 15) / 16) as u32;
        // Bytes: FP16 weight + FP16 x + FP32 y (read+write).
        let bytes = m * k * 2 + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_mw16_residual_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_mw16_residual_wmma",
            [rows, batches, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(wp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(mi);
                b.push_i32(ki);
                b.push_i32(ni);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Batched GEMV (GEMM) for F16 weights: Y[M,N] = W_f16[M,K] @ X_f32[N,K]^T
    pub fn gemm_f16(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_f16", kernels::GEMM_F16_SRC, "gemm_f16")?;
        let func = &self.functions["gemm_f16"];
        let mut wp = w.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, n as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// WMMA-accelerated batched GEMM for F16 weights × F32 activations (gfx1100+).
    /// Y[M,N] = W_f16[M,K] @ X_f32[N,K]^T.  Tiled 16×16 WMMA matrix multiply.
    /// Grid=[ceil(M/16), ceil(N/16)], Block=[32].  Replaces naive gemm_f16 for vision encoder.
    pub fn gemm_f16_wmma(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_f16_wmma", kernels::GEMM_F16_WMMA_SRC, "gemm_f16_wmma")?;
        let func = &self.functions["gemm_f16_wmma"];
        let mut wp = w.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        let grid_m = ((m + 15) / 16) as u32;
        let grid_n = ((n + 15) / 16) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_n, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Tiled F16 GEMM — 4-way ILP unrolled, no shared memory (high occupancy).
    /// Grid=[M, N], Block=[32], LDS=0.
    pub fn gemm_f16_tiled(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_f16_tiled",
            kernels::GEMM_F16_TILED_SRC,
            "gemm_f16_tiled",
        )?;
        let func = &self.functions["gemm_f16_tiled"];
        let mut wp = w.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        // Same grid as naive: [M, N], block [32], no LDS
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, n as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Fused GEMM + bias: Y[N,M] = X[N,K] @ W_f16[M,K]^T + bias[M].
    /// Replaces gemm_f16 + transpose_f32 + bias_add_f32 (3 ops → 1).
    /// Grid=[N, 1], Block=[256].
    pub fn gemm_f16_bias(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        bias: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_f16_bias", kernels::GEMM_F16_BIAS_SRC, "gemm_f16_bias")?;
        let func = &self.functions["gemm_f16_bias"];
        let mut wp = w.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut bp = bias.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        // One block per row of X, 256 threads, no LDS
        unsafe {
            self.hip.launch_kernel(
                func,
                [n as u32, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Batched GEMM for F32: Y[M,N] = A[M,K] @ B[N,K]^T
    pub fn gemm_f32_batched(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_f32_batched",
            kernels::GEMM_F32_SRC,
            "gemm_f32_batched",
        )?;
        let func = &self.functions["gemm_f32_batched"];
        let mut ap = a.buf.as_ptr();
        let mut bp = b.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, n as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// General fp32 GEMM with per-operand transpose flags (training path).
    ///
    /// Computes `C[M,N] = op(A)·op(B)`, C row-major. `op(A)` is `[M,K]`, `op(B)`
    /// is `[K,N]`. `lda`/`ldb` are the leading dims of the *stored* operands;
    /// `trans_a`/`trans_b` select whether the stored matrix is read transposed.
    /// One general kernel covers the forward and both backward matmuls of a
    /// linear — see `kernels/src/gemm_f32_train.hip`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f32_train(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
        m: usize,
        n: usize,
        k: usize,
        lda: usize,
        ldb: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_f32_train",
            kernels::GEMM_F32_TRAIN_SRC,
            "gemm_f32_train",
        )?;
        let func = &self.functions["gemm_f32_train"];
        let mut ap = a.buf.as_ptr();
        let mut bp = b.buf.as_ptr();
        let mut cp = c.buf.as_ptr();
        let mut mi = m as i32;
        let mut ni = n as i32;
        let mut ki = k as i32;
        let mut ldai = lda as i32;
        let mut ldbi = ldb as i32;
        let mut tai = trans_a as i32;
        let mut tbi = trans_b as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut cp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ldai as *mut _ as *mut c_void,
            &mut ldbi as *mut _ as *mut c_void,
            &mut tai as *mut _ as *mut c_void,
            &mut tbi as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [((n + 63) / 64) as u32, ((m + 63) / 64) as u32, 1],
                [16, 16, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// `gemm_f32_train` variant that accumulates: `C = beta*C + op(A)·op(B)`.
    /// Used where a gradient lands on a buffer that already holds a partial.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f32_train_accum(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
        m: usize,
        n: usize,
        k: usize,
        lda: usize,
        ldb: usize,
        trans_a: bool,
        trans_b: bool,
        beta: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_f32_train_accum",
            kernels::GEMM_F32_TRAIN_SRC,
            "gemm_f32_train_accum",
        )?;
        let func = &self.functions["gemm_f32_train_accum"];
        let mut ap = a.buf.as_ptr();
        let mut bp = b.buf.as_ptr();
        let mut cp = c.buf.as_ptr();
        let mut mi = m as i32;
        let mut ni = n as i32;
        let mut ki = k as i32;
        let mut ldai = lda as i32;
        let mut ldbi = ldb as i32;
        let mut tai = trans_a as i32;
        let mut tbi = trans_b as i32;
        let mut betaf = beta;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut cp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ldai as *mut _ as *mut c_void,
            &mut ldbi as *mut _ as *mut c_void,
            &mut tai as *mut _ as *mut c_void,
            &mut tbi as *mut _ as *mut c_void,
            &mut betaf as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [((n + 63) / 64) as u32, ((m + 63) / 64) as u32, 1],
                [16, 16, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn gemm_f16_wmma_mb4(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to gemm_f16 (which binds)
        self.gemm_f16(w, x, y, m, k, n)
    }
    pub fn gemm_f16_wmma_mb8(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to gemm_f16 (which binds)
        self.gemm_f16(w, x, y, m, k, n)
    }
    /// WMMA F16 weight × F16 input → F32 output GEMM with (B, M)
    /// output layout. Drop-in for `gemm_f32_register_tiled` once the
    /// weight has been kept on device as F16 and the input has been
    /// staged through `convert_f32_to_f16`.
    pub fn gemm_f16_x_f16_wmma(
        &mut self,
        a_f16: &GpuTensor,
        x_f16: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_f16_x_f16_wmma",
            kernels::GEMM_F16_X_F16_WMMA_SRC,
            "gemm_f16_x_f16_wmma",
        )?;
        let ap = a_f16.buf.as_ptr();
        let xp = x_f16.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
        ];
        let grid_m = ((m + 15) / 16) as u32;
        let grid_b = ((batch_size + 15) / 16) as u32;
        self.launch_maybe_blob(
            "gemm_f16_x_f16_wmma",
            [grid_m, grid_b, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(mi);
                b.push_i32(ki);
                b.push_i32(bi);
                b
            },
        )
    }
    /// WMMA F16 weight × FP32 input → F32 output GEMM with (B, M)
    /// output layout. This stages `x_f32` through the cached FP16 scratch
    /// before launching `gemm_f16_x_f16_wmma`.
    pub fn gemm_f16_x_f32_wmma(
        &mut self,
        a_f16: &GpuTensor,
        x_f32: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            matches!(a_f16.dtype, DType::F16 | DType::Raw),
            "gemm_f16_x_f32_wmma: weights must be F16 or raw F16 payload"
        );
        assert_eq!(
            x_f32.dtype,
            DType::F32,
            "gemm_f16_x_f32_wmma: input must be F32 before FP16 staging"
        );
        assert_eq!(
            y_f32.dtype,
            DType::F32,
            "gemm_f16_x_f32_wmma: output must be F32"
        );
        self.ensure_kernel(
            "gemm_f16_x_f16_wmma",
            kernels::GEMM_F16_X_F16_WMMA_SRC,
            "gemm_f16_x_f16_wmma",
        )?;
        let ap = a_f16.buf.as_ptr();
        let xp = self.ensure_fp16_x(x_f32, batch_size * k)?;
        let yp = y_f32.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
        ];
        let grid_m = ((m + 15) / 16) as u32;
        let grid_b = ((batch_size + 15) / 16) as u32;
        self.launch_maybe_blob(
            "gemm_f16_x_f16_wmma",
            [grid_m, grid_b, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(mi);
                b.push_i32(ki);
                b.push_i32(bi);
                b
            },
        )
    }
    /// WMMA BF16 weight × BF16-staged input → F32 output GEMM with
    /// (B, M) output layout. `a_bf16` is raw BF16 row-major [M, K].
    /// `x_f32` is staged through the cached BF16 scratch once per source
    /// pointer, then consumed by the RDNA wave32 BF16 WMMA kernel.
    /// Generic kernel library: WMMA GEMM, BF16 inputs → BF16 output.
    /// `a_bf16` [M,K], `x_bf16` [B,K], `y_bf16` [B,M], all raw BF16 (u16)
    /// payloads. gfx1103/RDNA3 wave32, zero LDS. Accumulation is in bf16
    /// precision (16-bit-output WMMA), so this trades accuracy for a bf16
    /// output; use the `→F32` sibling when long-K accuracy matters.
    /// Requires `k % 16 == 0` and wave32 WMMA.
    pub fn gemm_bf16_bf16_wmma(
        &mut self,
        a_bf16: &GpuTensor,
        x_bf16: &GpuTensor,
        y_bf16: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(k % 16, 0, "gemm_bf16_bf16_wmma: K must be a multiple of 16");
        self.ensure_kernel(
            "gemm_bf16_bf16_wmma",
            kernels::GEMM_BF16_BF16_WMMA_SRC,
            "gemm_bf16_bf16_wmma",
        )?;
        let ap = a_bf16.buf.as_ptr();
        let xp = x_bf16.buf.as_ptr();
        let yp = y_bf16.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
        ];
        let grid_m = ((m + 15) / 16) as u32;
        let grid_b = ((batch_size + 15) / 16) as u32;
        let func = &self.functions["gemm_bf16_bf16_wmma"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn gemm_f16_f16_wmma(
        &mut self,
        a_f16: &GpuTensor,
        x_f16: &GpuTensor,
        y_f16: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(k % 16, 0, "gemm_f16_f16_wmma: K must be a multiple of 16");
        self.ensure_kernel(
            "gemm_f16_f16_wmma",
            kernels::GEMM_F16_F16_WMMA_SRC,
            "gemm_f16_f16_wmma",
        )?;
        let ap = a_f16.buf.as_ptr();
        let xp = x_f16.buf.as_ptr();
        let yp = y_f16.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
        ];
        let grid_m = ((m + 15) / 16) as u32;
        let grid_b = ((batch_size + 15) / 16) as u32;
        let func = &self.functions["gemm_f16_f16_wmma"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn gemm_bf16_x_bf16_wmma(
        &mut self,
        a_bf16: &GpuTensor,
        x_f32: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.gemm_bf16_x_bf16_wmma_labeled(
            a_bf16,
            x_f32,
            y_f32,
            m,
            k,
            batch_size,
            "gemm_bf16_x_bf16_wmma",
        )
    }
    pub fn gemm_bf16_x_bf16_wmma_labeled(
        &mut self,
        a_bf16: &GpuTensor,
        x_f32: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        profile_label: &'static str,
    ) -> HipResult<()> {
        // Calibration capture: the BF16 linear chokepoint (the lowered super-op
        // path's bf16 gemv funnels here via gemm_bf16_x_bf16_wmma). Captures the
        // input activation before the compute. No-op when unarmed.
        self.maybe_capture_activation(a_bf16, x_f32, batch_size, k);
        if self.arch == "gfx1151"
            && m >= 128
            && batch_size >= 16
            && k % 16 == 0
            && std::env::var("HIPFIRE_BF16_DENSE_M128").ok().as_deref() != Some("0")
        {
            return self.gemm_bf16_x_bf16_wmma_gfx1151_m128_labeled(
                a_bf16,
                x_f32,
                y_f32,
                m,
                k,
                batch_size,
                profile_label,
            );
        }
        self.bind_thread()?;
        assert_eq!(
            a_bf16.dtype,
            DType::BF16,
            "gemm_bf16_x_bf16_wmma: weights must be BF16"
        );
        assert_eq!(
            x_f32.dtype,
            DType::F32,
            "gemm_bf16_x_bf16_wmma: input must be F32 before BF16 staging"
        );
        assert_eq!(
            y_f32.dtype,
            DType::F32,
            "gemm_bf16_x_bf16_wmma: output must be F32"
        );
        self.ensure_kernel(
            "gemm_bf16_x_bf16_wmma",
            kernels::GEMM_BF16_X_BF16_WMMA_SRC,
            "gemm_bf16_x_bf16_wmma",
        )?;
        let ap = a_bf16.buf.as_ptr();
        let xp = self.ensure_bf16_x(x_f32, batch_size * k)?;
        let yp = y_f32.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
        ];
        let grid_m = ((m + 15) / 16) as u32;
        let grid_b = ((batch_size + 15) / 16) as u32;
        let bytes = m * k * 2 + batch_size * k * 2 + batch_size * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", profile_label, bytes);
        let result = self.launch_maybe_blob(
            "gemm_bf16_x_bf16_wmma",
            [grid_m, grid_b, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(mi);
                b.push_i32(ki);
                b.push_i32(bi);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn gemm_bf16_x_bf16_wmma_gfx1151_m128_labeled(
        &mut self,
        a_bf16: &GpuTensor,
        x_f32: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        profile_label: &'static str,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            self.arch, "gfx1151",
            "gemm_bf16_x_bf16_wmma_gfx1151_m128 is gfx1151-only"
        );
        assert_eq!(
            a_bf16.dtype,
            DType::BF16,
            "gemm_bf16_x_bf16_wmma_gfx1151_m128: weights must be BF16"
        );
        assert_eq!(
            x_f32.dtype,
            DType::F32,
            "gemm_bf16_x_bf16_wmma_gfx1151_m128: input must be F32 before BF16 staging"
        );
        assert_eq!(
            y_f32.dtype,
            DType::F32,
            "gemm_bf16_x_bf16_wmma_gfx1151_m128: output must be F32"
        );
        assert!(
            k % 16 == 0,
            "gemm_bf16_x_bf16_wmma_gfx1151_m128: K={k} must be divisible by 16",
        );
        self.ensure_kernel(
            "gemm_bf16_x_bf16_wmma_gfx1151_m128",
            kernels::GEMM_BF16_X_BF16_WMMA_SRC,
            "gemm_bf16_x_bf16_wmma_gfx1151_m128",
        )?;
        let ap = a_bf16.buf.as_ptr();
        let xp = self.ensure_bf16_x(x_f32, batch_size * k)?;
        let yp = y_f32.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
        ];
        let grid_m = ((m + 127) / 128) as u32;
        let grid_b = ((batch_size + 15) / 16) as u32;
        let bytes = m * k * 2 + batch_size * k * 2 + batch_size * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", profile_label, bytes);
        let result = self.launch_maybe_blob(
            "gemm_bf16_x_bf16_wmma_gfx1151_m128",
            [grid_m, grid_b, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(mi);
                b.push_i32(ki);
                b.push_i32(bi);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Register-tiled F32 batched GEMM. Y[batch, M] = A[M,K] @ x[batch,K]^T.
    /// Each block holds BATCH_TILE=8 accumulators in registers and
    /// reuses each loaded weight element across them — amortizing
    /// weight bandwidth, which is the prefill bottleneck on unified-
    /// memory Strix Halo. Replaces the fake-batched `gemm_f32_batched`
    /// for prefill (where weight reads dominate).
    pub fn gemm_f32_register_tiled(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (kname, src, batch_tile, block_x) = (
            "gemm_f32_register_tiled",
            kernels::GEMM_F32_REGISTER_TILED_SRC,
            8u32,
            32u32,
        );
        self.ensure_kernel(kname, src, kname)?;
        let ap = a.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yp = y.buf.as_ptr();
        let mi = m as i32;
        let ki = k as i32;
        let bs = batch_size as i32;
        let mut ap_m = ap;
        let mut xp_m = xp;
        let mut yp_m = yp;
        let mut mi_m = mi;
        let mut ki_m = ki;
        let mut bs_m = bs;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap_m as *mut _ as *mut c_void,
            &mut xp_m as *mut _ as *mut c_void,
            &mut yp_m as *mut _ as *mut c_void,
            &mut mi_m as *mut _ as *mut c_void,
            &mut ki_m as *mut _ as *mut c_void,
            &mut bs_m as *mut _ as *mut c_void,
        ];
        let grid_y = (batch_size as u32 + batch_tile - 1) / batch_tile;
        let bytes = m * k * 4 + batch_size * k * 4 + batch_size * m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_f32_register_tiled", bytes);
        let result = self.launch_maybe_blob(
            kname,
            [m as u32, grid_y, 1],
            [block_x, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(mi);
                b.push_i32(ki);
                b.push_i32(bs);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// BATCH_TILE=16 sibling of `gemm_f32_register_tiled`. This is useful for
    /// large-vocab lm_head/reference paths where the M dimension is enormous
    /// and reducing CTA count matters more than preserving the BATCH_TILE=8
    /// register budget used by production prefill paths.
    pub fn gemm_f32_register_tiled_b16(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (kname, src, batch_tile, block_x) = (
            "gemm_f32_register_tiled_b16",
            kernels::GEMM_F32_REGISTER_TILED_B16_SRC,
            16u32,
            32u32,
        );
        self.ensure_kernel(kname, src, kname)?;
        let ap = a.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yp = y.buf.as_ptr();
        let mi = m as i32;
        let ki = k as i32;
        let bs = batch_size as i32;
        let mut ap_m = ap;
        let mut xp_m = xp;
        let mut yp_m = yp;
        let mut mi_m = mi;
        let mut ki_m = ki;
        let mut bs_m = bs;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap_m as *mut _ as *mut c_void,
            &mut xp_m as *mut _ as *mut c_void,
            &mut yp_m as *mut _ as *mut c_void,
            &mut mi_m as *mut _ as *mut c_void,
            &mut ki_m as *mut _ as *mut c_void,
            &mut bs_m as *mut _ as *mut c_void,
        ];
        let grid_y = (batch_size as u32 + batch_tile - 1) / batch_tile;
        let bytes = m * k * 4 + batch_size * k * 4 + batch_size * m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_f32_register_tiled_b16", bytes);
        let result = self.launch_maybe_blob(
            kname,
            [m as u32, grid_y, 1],
            [block_x, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(mi);
                b.push_i32(ki);
                b.push_i32(bs);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// BATCH_TILE=32 sibling for large-vocab lm_head/reference paths.
    pub fn gemm_f32_register_tiled_b32(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (kname, src, batch_tile, block_x) = (
            "gemm_f32_register_tiled_b32",
            kernels::GEMM_F32_REGISTER_TILED_B32_SRC,
            32u32,
            32u32,
        );
        self.ensure_kernel(kname, src, kname)?;
        let ap = a.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yp = y.buf.as_ptr();
        let mi = m as i32;
        let ki = k as i32;
        let bs = batch_size as i32;
        let mut ap_m = ap;
        let mut xp_m = xp;
        let mut yp_m = yp;
        let mut mi_m = mi;
        let mut ki_m = ki;
        let mut bs_m = bs;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap_m as *mut _ as *mut c_void,
            &mut xp_m as *mut _ as *mut c_void,
            &mut yp_m as *mut _ as *mut c_void,
            &mut mi_m as *mut _ as *mut c_void,
            &mut ki_m as *mut _ as *mut c_void,
            &mut bs_m as *mut _ as *mut c_void,
        ];
        let grid_y = (batch_size as u32 + batch_tile - 1) / batch_tile;
        let bytes = m * k * 4 + batch_size * k * 4 + batch_size * m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_f32_register_tiled_b32", bytes);
        let result = self.launch_maybe_blob(
            kname,
            [m as u32, grid_y, 1],
            [block_x, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(mi);
                b.push_i32(ki);
                b.push_i32(bs);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// BATCH_TILE=64 sibling for large-vocab lm_head/reference paths.
    pub fn gemm_f32_register_tiled_b64(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (kname, src, batch_tile, block_x) = (
            "gemm_f32_register_tiled_b64",
            kernels::GEMM_F32_REGISTER_TILED_B64_SRC,
            64u32,
            32u32,
        );
        self.ensure_kernel(kname, src, kname)?;
        let ap = a.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yp = y.buf.as_ptr();
        let mi = m as i32;
        let ki = k as i32;
        let bs = batch_size as i32;
        let mut ap_m = ap;
        let mut xp_m = xp;
        let mut yp_m = yp;
        let mut mi_m = mi;
        let mut ki_m = ki;
        let mut bs_m = bs;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap_m as *mut _ as *mut c_void,
            &mut xp_m as *mut _ as *mut c_void,
            &mut yp_m as *mut _ as *mut c_void,
            &mut mi_m as *mut _ as *mut c_void,
            &mut ki_m as *mut _ as *mut c_void,
            &mut bs_m as *mut _ as *mut c_void,
        ];
        let grid_y = (batch_size as u32 + batch_tile - 1) / batch_tile;
        let bytes = m * k * 4 + batch_size * k * 4 + batch_size * m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_f32_register_tiled_b64", bytes);
        let result = self.launch_maybe_blob(
            kname,
            [m as u32, grid_y, 1],
            [block_x, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(mi);
                b.push_i32(ki);
                b.push_i32(bs);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
