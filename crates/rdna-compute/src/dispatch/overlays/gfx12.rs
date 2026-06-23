// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx12xx (RDNA4) kernel-dispatch overlays. Phase 2.

use super::super::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym4_wmma_tile_batched_gfx12(
        &mut self,
        _q: &GpuTensor,
        _k_cache: &GpuTensor,
        _v_cache: &GpuTensor,
        _out: &GpuTensor,
        _positions: &GpuTensor,
        _ct: &GpuTensor,
        _st: &GpuTensor,
        _n_heads: usize,
        _n_kv_heads: usize,
        _head_dim: usize,
        _physical_cap: usize,
        _max_ctx_len: usize,
        _batch_size: usize,
        _partials: &GpuTensor,
        _tree_bias: Option<&GpuTensor>,
        _block_start: usize,
        _block_cols: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — unimplemented stub (no GPU work; returns Err)
        Err(hip_bridge::HipError::new(801, "not yet implemented"))
    }
    pub fn gemm_gate_up_hfp4g32_wmma_gfx12(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfp4g32_wmma_gfx12",
            kernels::GEMM_GATE_UP_HFP4G32_WMMA_GFX12_SRC,
            "gemm_gate_up_hfp4g32_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut gm_val = gate_m as i32;
        let mut um_val = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut gm_val as *mut _ as *mut c_void,
            &mut um_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfp4g32_bytes(gate_m, k)
            + crate::profile::gemv_hfp4g32_bytes(up_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_gate_up_hfp4g32_wmma_gfx12",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfp4g32_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg);
                b.push_ptr(yu);
                b.push_i32(gm_val);
                b.push_i32(um_val);
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
    /// gfx12 (RDNA4) sister of `gemm_gate_up_hfq3g256_wmma`.
    pub fn gemm_gate_up_hfq3g256_wmma_gfx12(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq3g256_wmma_gfx12",
            kernels::GEMM_GATE_UP_HFQ3G256_WMMA_GFX12_SRC,
            "gemm_gate_up_hfq3g256_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let weight_bytes = total_m * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_gate_up_hfq3g256_wmma_gfx12",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfq3g256_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg);
                b.push_ptr(yu);
                b.push_i32(g_m);
                b.push_i32(u_m);
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
    /// gfx12 (RDNA4) sister of `gemm_gate_up_hfq4g256_wmma`. Same recipe
    /// as the QKV gfx12 scaffold (validated on R9700). Not yet wired into
    /// the public dispatch tree — exposed only for the channel-test
    /// harness. See issue #54.
    pub fn gemm_gate_up_hfq4g256_wmma_gfx12(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq4g256_wmma_gfx12",
            kernels::GEMM_GATE_UP_HFQ4G256_WMMA_GFX12_SRC,
            "gemm_gate_up_hfq4g256_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
            + crate::profile::gemv_hfq4g256_bytes(up_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_gate_up_hfq4g256_wmma_gfx12",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfq4g256_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg);
                b.push_ptr(yu);
                b.push_i32(g_m);
                b.push_i32(u_m);
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
    /// gfx12 (RDNA4) sister of `gemm_gate_up_hfq6g256_wmma`. Same gfx12
    /// recipe as the other scaffolds (validated on R9700). Not yet wired
    /// into the public dispatch tree — exposed only for the channel-test
    /// harness. See issue #54.
    pub fn gemm_gate_up_hfq6g256_wmma_gfx12(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq6g256_wmma_gfx12",
            kernels::GEMM_GATE_UP_HFQ6G256_WMMA_GFX12_SRC,
            "gemm_gate_up_hfq6g256_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
            + crate::profile::gemv_hfq4g256_bytes(up_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_gate_up_hfq6g256_wmma_gfx12",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfq6g256_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg);
                b.push_ptr(yu);
                b.push_i32(g_m);
                b.push_i32(u_m);
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
    /// gfx12 sister of `gemm_gate_up_q8_0_wmma` (FFN preamble).
    pub fn gemm_gate_up_q8_0_wmma_gfx12(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(
            k % 32,
            0,
            "gemm_gate_up_q8_0_wmma_gfx12: K must be a multiple of 32 (got K={k})"
        );
        self.ensure_kernel(
            "gemm_gate_up_q8_0_wmma_gfx12",
            kernels::GEMM_GATE_UP_Q8_0_WMMA_GFX12_SRC,
            "gemm_gate_up_q8_0_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_g = a_gate.buf.as_ptr();
        let mut a_u = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut y_g = y_gate.buf.as_ptr();
        let mut y_u = y_up.buf.as_ptr();
        let mut gate_m_val = gate_m as i32;
        let mut up_m_val = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_g as *mut _ as *mut c_void,
            &mut a_u as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut y_g as *mut _ as *mut c_void,
            &mut y_u as *mut _ as *mut c_void,
            &mut gate_m_val as *mut _ as *mut c_void,
            &mut up_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;
        let q8_bytes = |m: usize| m * (k / 32) * 34;
        let bytes =
            q8_bytes(gate_m) + q8_bytes(up_m) + batch_size * k * 2 + batch_size * total_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_q8_0_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_gate_up_q8_0_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_g);
                b.push_ptr(a_u);
                b.push_ptr(xp);
                b.push_ptr(y_g);
                b.push_ptr(y_u);
                b.push_i32(gate_m_val);
                b.push_i32(up_m_val);
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
    /// gfx12 (RDNA4) sister of `gemm_hfq4g256_residual_wmma` (specifically
    /// the `_k2` variant — the gfx11 dispatch default for M >= 8192, with
    /// the validated C-output mapping).
    ///
    /// Closes the residual-GEMM gap on 9B prefill: before this kernel,
    /// gfx12 fell through to the dot2 fp16 fallback for the residual call
    /// site (attn-out + ffn-down), which accounted for ~42% of 9B prefill
    /// time on R9700. The other six gfx12 WMMA kernels shipped in PR #62.
    ///
    /// Same recipe as the qkv / qkvza / gate_up gfx12 ports: `_w32_gfx12`
    /// builtin, half8_t operands, K-split via `tid >> 4`, contiguous
    /// C-row mapping (`acc[j] = C[8*(tid>>4) + j][tid & 15]`). Validated
    /// on R9700 by the `test_wmma_residual_gfx12` channel-test against
    /// the dot2 reference path.
    pub fn gemm_hfq4g256_residual_wmma_gfx12(
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
            "gemm_hfq4g256_residual_wmma_gfx12",
            kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_GFX12_SRC,
            "gemm_hfq4g256_residual_wmma_gfx12",
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

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq4g256_residual_wmma_gfx12",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_hfq4g256_residual_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
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
                b.push_i32(bs_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// gfx12 lm_head sibling of `gemm_hfq4g256_residual_wmma_gfx12`.
    /// Uses the same WMMA tile mapping but overwrites Y instead of reading
    /// and adding to it, so lm_head callers can skip a full-vocab zero-fill.
    pub fn gemm_hfq4g256_lmhead_wmma_gfx12(
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
            "gemm_hfq4g256_lmhead_wmma_gfx12",
            kernels::GEMM_HFQ4G256_LMHEAD_WMMA_GFX12_SRC,
            "gemm_hfq4g256_lmhead_wmma_gfx12",
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

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq4g256_lmhead_wmma_gfx12",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_hfq4g256_lmhead_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
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
                b.push_i32(bs_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// gfx12 (RDNA4 — R9700/gfx1201, gfx1200) i8 MMQ MoE grouped GEMM. Ports
    /// the i8 WMMA MMQ pattern to the SGLang grouped scatter dispatch using
    /// the gfx12-specific WMMA intrinsic (`wmma_i32_16x16x16_iu8_w32_gfx12`)
    /// at 2× the FLOP rate of FP16 WMMA on gfx12. X is pre-quantized to Q8_1
    /// via `ensure_q8_1_mmq_x` (same scratch buffer as the residual MMQ path).
    ///
    /// Kernarg layout matches the gfx1151 sister: FP16 args + `x_src_rows`
    /// extra arg for the Q8_1 layout stride (`[K/128 × x_src_rows]`).
    ///
    /// Used as a drop-in replacement for `gemm_hfq4g256_moe_grouped_wmma_k2`
    /// on gfx12 when `HIPFIRE_MOE_GROUPED_I8 != "0"` (default ON for gfx12).
    /// The FP16 sister still owns the env=0 opt-out path.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_moe_grouped_mmq_gfx12(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to Q8_1)
        y_grouped: &GpuTensor,          // [m_total × M] f32, written direct
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let kernel_name = "gemm_hfq4g256_moe_grouped_mmq_gfx12";
        let kernel_src = kernels::GEMM_HFQ4G256_MOE_GROUPED_MMQ_GFX12_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        // Q8_1 pre-pass (reuses the shared MMQ X scratch).
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x_src, x_src_rows, k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_q8_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;
        let xsr_val = x_src_rows as i32;

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
            &xsr_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW estimate: Q8_1 X reads (~1B/elem + ds4 metadata) + HFQ4 weights + Y writes.
        let bytes = (m_total * k) + (m_total * m) * 4 + (crate::profile::gemv_hfq4g256_bytes(m, k));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
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
                b.push_i32(xsr_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// gfx12 (RDNA4 — R9700/gfx1201, gfx1200) i8 MMQ MoE grouped GEMM —
    /// k4 (deeper K-tile pipeline) variant. Drop-in for
    /// `gemm_hfq4g256_moe_grouped_mmq_gfx12` — same kernarg layout, same
    /// grid/block geometry, same scatter contract. Processes all 4
    /// sub-blocks of one Q8_1 block per inner iteration (8 WMMAs into 4
    /// independent int32 accumulators) before the per-sub-block scale FMA
    /// chain resolves. Numerically equivalent to k2 modulo floating-point
    /// summation order on the scale FMA chain.
    ///
    /// Opt-IN via `HIPFIRE_MOE_GROUPED_I8=1 HIPFIRE_MOE_GROUPED_I8_K4_GFX12=1`
    /// (both default OFF). Routes through the same wrapper as k2
    /// (`gemm_hfq4g256_moe_grouped_wmma_k2`), which gates on
    /// `HIPFIRE_MOE_GROUPED_I8 == "1"` for gfx12 first.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_moe_grouped_mmq_k4_gfx12(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to Q8_1)
        y_grouped: &GpuTensor,          // [m_total × M] f32, written direct
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let kernel_name = "gemm_hfq4g256_moe_grouped_mmq_k4_gfx12";
        let kernel_src = kernels::GEMM_HFQ4G256_MOE_GROUPED_MMQ_K4_GFX12_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        // Q8_1 pre-pass (reuses the shared MMQ X scratch).
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x_src, x_src_rows, k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_q8_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;
        let xsr_val = x_src_rows as i32;

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
            &xsr_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW estimate: same as the k2 sibling. k4 is a pure unroll-depth
        // change, no extra memory traffic.
        let bytes = (m_total * k) + (m_total * m) * 4 + (crate::profile::gemv_hfq4g256_bytes(m, k));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
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
                b.push_i32(xsr_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// MQ3 wrapper for `gemm_hfq3g256_residual_wmma`: pre-rotates X then
    /// dispatches the HFQ3 kernel. See `gemm_qkvza_mq3g256_wmma` for
    /// the cache-invalidation rationale.
    /// gfx12 (RDNA4) sister of `gemm_hfq3g256_residual_wmma`.
    pub fn gemm_hfq3g256_residual_wmma_gfx12(
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
            "gemm_hfq3g256_residual_wmma_gfx12",
            kernels::GEMM_HFQ3G256_RESIDUAL_WMMA_GFX12_SRC,
            "gemm_hfq3g256_residual_wmma_gfx12",
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

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let weight_bytes = m * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq3g256_residual_wmma_gfx12",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_hfq3g256_residual_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
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
                b.push_i32(bs_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// gfx12 sister of `gemm_hfq6g256_residual_wmma` (wo / w_down post-projection
    /// for MQ6 LA/FA attention). Caller seeds Y with the residual; kernel does
    /// `Y += X @ A^T`. Mirrors `gemm_q8_0_residual_wmma_gfx12` kernarg layout
    /// and grid.
    pub fn gemm_hfq6g256_residual_wmma_gfx12(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(
            k % 256,
            0,
            "gemm_hfq6g256_residual_wmma_gfx12: K must be a multiple of 256 (got K={k})"
        );
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq6g256_residual_wmma_gfx12",
            kernels::GEMM_HFQ6G256_RESIDUAL_WMMA_GFX12_SRC,
            "gemm_hfq6g256_residual_wmma_gfx12",
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

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * k * 2  // FP16 X
            + batch_size * m * 4 * 2; // residual read + write
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq6g256_residual_wmma_gfx12",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_hfq6g256_residual_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
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
                b.push_i32(bs_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn gemm_hfp4g32_residual_wmma_gfx12(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfp4g32_residual_wmma_gfx12",
            kernels::GEMM_HFP4G32_RESIDUAL_WMMA_GFX12_SRC,
            "gemm_hfp4g32_residual_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ap = a.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yp = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes =
            crate::profile::gemv_hfp4g32_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfp4g32_residual_wmma_gfx12",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_hfp4g32_residual_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
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
    /// gfx12 sister of `gemm_q8_0_residual_wmma` (wo / w_down post-projection).
    /// Caller seeds Y with the residual; kernel does `Y += X @ A^T`.
    pub fn gemm_q8_0_residual_wmma_gfx12(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(
            k % 32,
            0,
            "gemm_q8_0_residual_wmma_gfx12: K must be a multiple of 32 (got K={k})"
        );
        self.ensure_kernel(
            "gemm_q8_0_residual_wmma_gfx12",
            kernels::GEMM_Q8_0_RESIDUAL_WMMA_GFX12_SRC,
            "gemm_q8_0_residual_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_p = a.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut y_p = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_p as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut y_p as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;
        let bytes = m * (k / 32) * 34 + batch_size * k * 2 + batch_size * m * 4 * 2; // residual read + write
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_q8_0_residual_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_q8_0_residual_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_p);
                b.push_ptr(xp);
                b.push_ptr(y_p);
                b.push_i32(m_val);
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
    pub fn gemm_qkvza_hfp4g32_wmma_gfx12(
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
            "gemm_qkvza_hfp4g32_wmma_gfx12",
            kernels::GEMM_QKVZA_HFP4G32_WMMA_GFX12_SRC,
            "gemm_qkvza_hfp4g32_wmma_gfx12",
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
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfp4g32_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfp4g32_wmma_gfx12",
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
    /// gfx12 (RDNA4) sister of `gemm_qkvza_hfq3g256_wmma`. K4-unrolled
    /// half8_t lane-split per `gemm_qkvza_hfq4g256_wmma_gfx12`. Wired via
    /// the `gemm_qkvza_hfq3g256_wmma` arch dispatch — direct callers can
    /// also use this if they know they're on gfx12.
    pub fn gemm_qkvza_hfq3g256_wmma_gfx12(
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
            "gemm_qkvza_hfq3g256_wmma_gfx12",
            kernels::GEMM_QKVZA_HFQ3G256_WMMA_GFX12_SRC,
            "gemm_qkvza_hfq3g256_wmma_gfx12",
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

        let weight_bytes = total_m * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq3g256_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfq3g256_wmma_gfx12",
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
    /// gfx12 (RDNA4) sister of `gemm_qkvza_hfq4g256_wmma`. Same gfx12
    /// recipe as the other scaffolds (validated on R9700) extended to
    /// 4-output qkv/z/beta/alpha routing. Not yet wired into the public
    /// dispatch tree — exposed only for the channel-test harness.
    pub fn gemm_qkvza_hfq4g256_wmma_gfx12(
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
            "gemm_qkvza_hfq4g256_wmma_gfx12",
            kernels::GEMM_QKVZA_HFQ4G256_WMMA_GFX12_SRC,
            "gemm_qkvza_hfq4g256_wmma_gfx12",
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
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq4g256_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfq4g256_wmma_gfx12",
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
    /// gfx12 (RDNA4) sister of `gemm_qkv_hfq4g256_wmma`. Identical signature
    /// and grid/block; only the kernel-side intrinsic + operand vector size
    /// differs. NOT yet wired into the public dispatch tree — exposed only
    /// for the channel-test (`test_wmma_qkv_gfx12`) that validates the
    /// gfx12 C-output mapping hypothesis on real RDNA4 silicon. See issue
    /// #54 and `.skills/hipfire-arch-port/wmma-matrix.md`.
    pub fn gemm_qkv_hfq4g256_wmma_gfx12(
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
            "gemm_qkv_hfq4g256_wmma_gfx12",
            kernels::GEMM_QKV_HFQ4G256_WMMA_GFX12_SRC,
            "gemm_qkv_hfq4g256_wmma_gfx12",
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
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq4g256_wmma_gfx12",
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
    /// gfx12 (RDNA4) variant of `gemm_qkv_hfp4g32`. half8_t lane-split +
    /// K4 unroll. Same C-output mapping as `gemm_qkv_hfq4g256_wmma_gfx12`.
    pub fn gemm_qkv_hfp4g32_wmma_gfx12(
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
            "gemm_qkv_hfp4g32_wmma_gfx12",
            kernels::GEMM_QKV_HFP4G32_WMMA_GFX12_SRC,
            "gemm_qkv_hfp4g32_wmma_gfx12",
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
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfp4g32_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfp4g32_wmma_gfx12",
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
    /// gfx12 FP8-WMMA variant of `gemm_qkv_hfp4g32_wmma_gfx12`. Same
    /// 16x16x16 tile shape, same C-mapping; weight LUT pre-converts
    /// E2M1->E4M3 bytes (no scale) and per-output-row row_scale * UE8M0
    /// is applied to the F32 accumulator after each WMMA pair via
    /// lane-shuffle. Activation is converted FP16->FP8 inline by
    /// cvt_pk_fp8_f32 (unscaled — post-RMSNorm magnitudes are bounded
    /// well below E4M3 saturation). Opt-in via HIPFIRE_FP8_WMMA=1.
    pub fn gemm_qkv_hfp4g32_wmma_fp8_gfx12(
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
            "gemm_qkv_hfp4g32_wmma_fp8_gfx12",
            kernels::GEMM_QKV_HFP4G32_WMMA_FP8_GFX12_SRC,
            "gemm_qkv_hfp4g32_wmma_fp8_gfx12",
        )?;
        let x_fp8_ptr = self.ensure_fp8_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_fp8_ptr;
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
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_qkv_hfp4g32_wmma_fp8_gfx12",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfp4g32_wmma_fp8_gfx12",
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
    /// HFQ3-G256 sister of `gemm_gate_up_hfq4g256_wmma`. Same WMMA shape
    /// + lane decomposition; only the inner K-tile unpack differs (3-bit
    /// cross-byte vs 4-bit nibble) and the per-group byte stride is 104
    /// instead of 136. Used for MQ3 prefill via `gemm_gate_up_mq3g256_wmma`.
    /// gfx12 (RDNA4) sister of `gemm_qkv_hfq3g256_wmma`.
    pub fn gemm_qkv_hfq3g256_wmma_gfx12(
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
            "gemm_qkv_hfq3g256_wmma_gfx12",
            kernels::GEMM_QKV_HFQ3G256_WMMA_GFX12_SRC,
            "gemm_qkv_hfq3g256_wmma_gfx12",
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

        let weight_bytes = total_m * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq3g256_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq3g256_wmma_gfx12",
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
    /// gfx12 (RDNA4) sister of `gemm_qkvza_hfq6g256_wmma`. Pure scaffold
    /// composition (hfq6 dequant + 4-output qkvza routing, both validated
    /// on R9700). Not yet wired into the public dispatch tree — exposed
    /// only for the channel-test harness. See issue #54.
    pub fn gemm_qkvza_hfq6g256_wmma_gfx12(
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
            "gemm_qkvza_hfq6g256_wmma_gfx12",
            kernels::GEMM_QKVZA_HFQ6G256_WMMA_GFX12_SRC,
            "gemm_qkvza_hfq6g256_wmma_gfx12",
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
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq6g256_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfq6g256_wmma_gfx12",
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
    /// gfx12 (RDNA4) sister of `gemm_qkv_hfq6g256_wmma`. Same gfx12 recipe
    /// as the hfq4 QKV scaffold (validated on R9700) with the hfq6 dequant
    /// inner loop carried over. Not yet wired into the public dispatch
    /// tree — exposed only for the channel-test harness. See issue #54.
    pub fn gemm_qkv_hfq6g256_wmma_gfx12(
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
            "gemm_qkv_hfq6g256_wmma_gfx12",
            kernels::GEMM_QKV_HFQ6G256_WMMA_GFX12_SRC,
            "gemm_qkv_hfq6g256_wmma_gfx12",
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
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq6g256_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq6g256_wmma_gfx12",
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
    /// gfx12 (RDNA4) sister of `gemm_qkv_q8_0_wmma`. Uses
    /// `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` (vs the gfx11 `_w32`)
    /// and half8_t operands with K split across 2 lane-groups. Mirrors the
    /// `gemm_qkv_hfq4g256_wmma_gfx12` pattern.
    pub fn gemm_qkv_q8_0_wmma_gfx12(
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
        debug_assert_eq!(
            k % 32,
            0,
            "gemm_qkv_q8_0_wmma_gfx12: K must be a multiple of 32 (got K={k})"
        );
        self.ensure_kernel(
            "gemm_qkv_q8_0_wmma_gfx12",
            kernels::GEMM_QKV_Q8_0_WMMA_GFX12_SRC,
            "gemm_qkv_q8_0_wmma_gfx12",
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

        let q8_bytes = |m: usize| m * (k / 32) * 34;
        let bytes = q8_bytes(q_m)
            + q8_bytes(k_m)
            + q8_bytes(v_m)
            + batch_size * k * 2
            + batch_size * total_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_q8_0_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_q8_0_wmma_gfx12",
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
    /// gfx12 sister of `gemm_qkvza_q8_0_wmma` (DeltaNet LA preamble).
    pub fn gemm_qkvza_q8_0_wmma_gfx12(
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
        debug_assert_eq!(
            k % 32,
            0,
            "gemm_qkvza_q8_0_wmma_gfx12: K must be a multiple of 32 (got K={k})"
        );
        self.ensure_kernel(
            "gemm_qkvza_q8_0_wmma_gfx12",
            kernels::GEMM_QKVZA_Q8_0_WMMA_GFX12_SRC,
            "gemm_qkvza_q8_0_wmma_gfx12",
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

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;
        let q8_bytes = |m: usize| m * (k / 32) * 34;
        let bytes = q8_bytes(qkv_m)
            + q8_bytes(z_m)
            + q8_bytes(beta_m)
            + q8_bytes(alpha_m)
            + batch_size * k * 2
            + batch_size * total_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_q8_0_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_q8_0_wmma_gfx12",
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
    /// gfx12 FP8-dot4 decode-path GEMV for HFP4G32. Uses
    /// `dot4_f32_fp8_fp8` to cut inner-loop ALU vs the dequant/FMA
    /// fallback. Activation X is consumed as FP8 (E4M3); when called
    /// via `gemv_hfp4g32` (env-gated routing for HFP4G32 weights, no
    /// rotation), this function calls `ensure_fp8_x` to pack F32 → FP8
    /// scratch. The MFP4G32 rotation path uses
    /// `rotate_x_mq_dual_fp8` + `gemv_hfp4g32_fp8_gfx12_with_fp8_ptr`
    /// instead so the FP8 pack is fused into the rotation kernel.
    pub fn gemv_hfp4g32_fp8_gfx12(
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
            "gemv_hfp4g32_fp8 requires K%256==0, got K={}",
            k
        );
        self.ensure_kernel(
            "gemv_hfp4g32_fp8_gfx12",
            kernels::GEMV_HFP4G32_FP8_GFX12_SRC,
            "gemv_hfp4g32_fp8_gfx12",
        )?;
        let x_fp8_ptr = self.ensure_fp8_x(x, k)?;
        self.gemv_hfp4g32_fp8_gfx12_with_fp8_ptr(a_raw, x_fp8_ptr, y, m, k)
    }
    /// FP8-dot4 GEMV variant that takes an FP8 device pointer directly
    /// (bypassing `ensure_fp8_x`). Used by `gemv_mfp4g32_with_rotate`
    /// after the fused rotation+pack kernel produces the FP8 buffer
    /// in-place.
    pub(crate) fn gemv_hfp4g32_fp8_gfx12_with_fp8_ptr(
        &mut self,
        a_raw: &GpuTensor,
        x_fp8_ptr: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        assert!(
            k % 256 == 0,
            "gemv_hfp4g32_fp8 requires K%256==0, got K={}",
            k
        );
        self.ensure_kernel(
            "gemv_hfp4g32_fp8_gfx12",
            kernels::GEMV_HFP4G32_FP8_GFX12_SRC,
            "gemv_hfp4g32_fp8_gfx12",
        )?;
        let func = &self.functions["gemv_hfp4g32_fp8_gfx12"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_fp8_ptr;
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
                32,
                self.stream_ref(),
                &mut params,
            )
        }
    }
}
