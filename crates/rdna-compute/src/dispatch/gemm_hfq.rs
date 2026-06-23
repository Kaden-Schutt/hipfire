// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! HFQ-quantized GEMM dispatch (hfq2/3/4/6/8 g128/g256). Pure move (Phase 1 M6).

use super::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{DeviceBuffer, HipResult};
use std::ffi::c_void;
use std::sync::OnceLock;

impl Gpu {
    /// Synthetic gfx1151 HFQ4-G256 x signed-Q4 activation IU4-WMMA probe.
    /// Caller supplies a prequantized S4 activation scratch plus per-32-K
    /// scale/sum metadata. This validates the affine correction path but is
    /// deliberately not routed into Qwen model code.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_s4_mmq_gfx1151(
        &mut self,
        a: &GpuTensor,
        x_qs: &GpuTensor,
        x_d: &GpuTensor,
        x_sum: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch != "gfx1151" {
            panic!(
                "gemm_hfq4g256_s4_mmq_gfx1151: only gfx1151 is supported; arch={}",
                self.arch
            );
        }
        assert_eq!(
            k % 256,
            0,
            "gemm_hfq4g256_s4_mmq_gfx1151 requires K multiple of 256"
        );
        self.ensure_kernel(
            "gemm_hfq4g256_s4_mmq_gfx1151",
            kernels::GEMM_HFQ4G256_S4_MMQ_GFX1151_SRC,
            "gemm_hfq4g256_s4_mmq_gfx1151",
        )?;

        let a_ptr = a.buf.as_ptr();
        let x_qs_ptr = x_qs.buf.as_ptr();
        let x_d_ptr = x_d.buf.as_ptr();
        let x_sum_ptr = x_sum.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let n_val = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_qs_ptr as *const _ as *mut c_void,
            &x_d_ptr as *const _ as *mut c_void,
            &x_sum_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "gemm_hfq4g256_s4_mmq_gfx1151",
            [m.div_ceil(16) as u32, n.div_ceil(16) as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_qs_ptr);
                b.push_ptr(x_d_ptr);
                b.push_ptr(x_sum_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        )
    }
    /// Batched HFQ4-G128 GEMM. Same tiled approach as G256.
    ///
    /// gfx1151 i8 MMQ fast-path (default ON; opt out via HIPFIRE_HFQ4G128_MMQ=0):
    /// when batch_size and M are 16-tile aligned, pre-quantize X to Q8_1 and
    /// route to `gemm_hfq4g128_mmq_gfx1151`. Closes the rocprof finding that
    /// this kernel was 66% of pp256 prefill on A3B-PARO; A/B median +129.5%
    /// (427 → 980 tok/s pp256) on shisa-Qwen3.6-35B-A3B-PARO. Mirror of the
    /// routed-MoE MMQ k8 default-on flip at 949f51db.
    pub fn gemm_hfq4g128(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let use_mmq = self.arch.starts_with("gfx1151")
            && std::env::var("HIPFIRE_HFQ4G128_MMQ").as_deref() != Ok("0")
            && batch_size >= 16
            && batch_size % 16 == 0
            && m % 16 == 0
            && k % 128 == 0;
        if use_mmq {
            return self.gemm_hfq4g128_mmq_gfx1151(a_raw, x, y, m, k, batch_size);
        }
        self.ensure_kernel("gemm_hfq4g128", kernels::GEMM_HFQ4G128_SRC, "gemm_hfq4g128")?;
        let func = &self.functions["gemm_hfq4g128"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
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
        let batch_tiles = ((batch_size + 7) / 8) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_tiles, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// gfx1151 i8 MMQ dispatch helper for HFQ4-G128. Pre-quantizes X to
    /// Q8_1 mmq DS4 then launches `gemm_hfq4g128_mmq_gfx1151`. Caller must
    /// have already verified the alignment constraints.
    fn gemm_hfq4g128_mmq_gfx1151(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        self.ensure_kernel(
            "gemm_hfq4g128_mmq_gfx1151",
            kernels::GEMM_HFQ4G128_MMQ_GFX1151_SRC,
            "gemm_hfq4g128_mmq_gfx1151",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let n_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_q8_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
        ];
        let bytes = (m * k / 2)               // HFQ4 weight (4 bits / elem)
                  + (batch_size * k * 4 / 3)  // Q8_1 activation (approx, includes ds4 headers)
                  + (batch_size * m * 4); // F32 output
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g128_mmq_gfx1151", bytes);
        let result = self.launch_maybe_blob(
            "gemm_hfq4g128_mmq_gfx1151",
            [(m / 16) as u32, (batch_size / 16) as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_q8_ptr);
                b.push_ptr(y_ptr);
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
    /// Wave32 MMQ residual kernel for HFQ3 on RDNA2+ — Phase 3 tile-size
    /// family auto-selector. Picks the best path per batch_size, falling
    /// back to `gemm_hfq3g256_residual_dot2` when MMQ would lose at small
    /// N. Gate boundaries from the microbench at
    /// `examples/bench_hfq3_mmq_sweep.rs` (m=4096, k=2048 on gfx1031):
    ///   batch ≤ 12       → dot2 (MMQ tile granularity wastes compute)
    ///   13 ≤ batch ≤ 127 → mmq_x=16 (best across this whole range,
    ///                       within ~5% of mmq_x=32 even at N=96)
    ///   batch ≥ 128      → mmq_x=32 (b128 LDS path pulls ahead +4-10%)
    /// Default-on on the supported allowlist unless `HIPFIRE_HFQ3_MMQ=0`.
    /// mmq_x=8 is never best in the
    /// sweep (lost to scalar/dot2 at small N, lost to mmq_x=16 at large
    /// N) so it's not in the auto-selector — kept available as
    /// `gemm_hfq3g256_residual_mmq_x8` for further experimentation.
    pub fn gemm_hfq3g256_residual_mmq(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to gemm_hfq3g256_residual_{dot2,mmq_xN} which bind.
        if !self.arch_caps.has_hfq3_sdot4() {
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_hfq3g256_residual_dot2(a_raw, x, y, m, k, batch_size);
            }
            return self.gemm_hfq3g256_residual_fp16(a_raw, x, y, m, k, batch_size);
        }
        if batch_size <= 12 {
            self.gemm_hfq3g256_residual_dot2(a_raw, x, y, m, k, batch_size)
        } else if batch_size <= 63 {
            self.gemm_hfq3g256_residual_mmq_x16(a_raw, x, y, m, k, batch_size)
        } else {
            self.gemm_hfq3g256_residual_mmq_x32_y64(a_raw, x, y, m, k, batch_size)
        }
    }
    /// HFQ3 MMQ residual at mmq_x=8 (short-prefill tile).
    pub fn gemm_hfq3g256_residual_mmq_x8(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_hfq3_mmq_tile(
            a_raw,
            x,
            y,
            m,
            k,
            batch_size,
            8,
            "gemm_hfq3g256_residual_mmq_x8",
            kernels::GEMM_HFQ3G256_RESIDUAL_MMQ_X8_SRC,
        )
    }
    /// HFQ3 MMQ residual at mmq_x=16 (mid-prefill tile).
    pub fn gemm_hfq3g256_residual_mmq_x16(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_hfq3_mmq_tile(
            a_raw,
            x,
            y,
            m,
            k,
            batch_size,
            16,
            "gemm_hfq3g256_residual_mmq_x16",
            kernels::GEMM_HFQ3G256_RESIDUAL_MMQ_X16_SRC,
        )
    }
    /// HFQ3 MMQ residual at mmq_x=32 (long-prefill tile, b128 LDS path).
    pub fn gemm_hfq3g256_residual_mmq_x32(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_hfq3_mmq_tile(
            a_raw,
            x,
            y,
            m,
            k,
            batch_size,
            32,
            "gemm_hfq3g256_residual_mmq_x32",
            kernels::GEMM_HFQ3G256_RESIDUAL_MMQ_X32_SRC,
        )
    }
    /// HFQ3 MMQ residual experimental MMQ_Y=64 variant (mmq_x=32).
    pub fn gemm_hfq3g256_residual_mmq_x32_y64(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_hfq3_mmq_tile_with_y(
            a_raw,
            x,
            y,
            m,
            k,
            batch_size,
            32,
            64,
            "gemm_hfq3g256_residual_mmq_x32_y64",
            kernels::GEMM_HFQ3G256_RESIDUAL_MMQ_X32_Y64_SRC,
        )
    }
    /// HFQ3 MMQ residual experimental MMQ_Y=32 variant (mmq_x=32).
    pub fn gemm_hfq3g256_residual_mmq_x32_y32(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_hfq3_mmq_tile_with_y(
            a_raw,
            x,
            y,
            m,
            k,
            batch_size,
            32,
            32,
            "gemm_hfq3g256_residual_mmq_x32_y32",
            kernels::GEMM_HFQ3G256_RESIDUAL_MMQ_X32_Y32_SRC,
        )
    }
    pub fn gemm_hfq4g256_residual_mmq_x16(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_hfq4_mmq_tile_with_y(
            a_raw,
            x,
            y,
            m,
            k,
            batch_size,
            16,
            128,
            "gemm_hfq4g256_residual_mmq_x16",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_X16_SRC,
        )
    }
    pub fn gemm_hfq4g256_residual_mmq_x32(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_hfq4_mmq_tile_with_y(
            a_raw,
            x,
            y,
            m,
            k,
            batch_size,
            32,
            128,
            "gemm_hfq4g256_residual_mmq_x32",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_X32_SRC,
        )
    }
    pub fn gemm_hfq4g256_residual_mmq_x32_y64(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_hfq4_mmq_tile_with_y(
            a_raw,
            x,
            y,
            m,
            k,
            batch_size,
            32,
            64,
            "gemm_hfq4g256_residual_mmq_x32_y64",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_X32_Y64_SRC,
        )
    }
    pub fn gemm_hfq4g256_residual_mmq_rdna2_auto(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if batch_size <= 63 {
            self.gemm_hfq4g256_residual_mmq_x16(a_raw, x, y, m, k, batch_size)
        } else {
            self.gemm_hfq4g256_residual_mmq_x32_y64(a_raw, x, y, m, k, batch_size)
        }
    }
    /// Wave32 MMQ residual kernel for HFQ4 on RDNA2+ — Phase 3 side-win probe.
    /// Same topology as the HFQ3 sibling; differs only in 4-bit nibble unpack
    /// (vs 3-bit trit). Default-on unless `HIPFIRE_HFQ4_MMQ_RDNA2=0`.
    pub fn gemm_hfq4g256_residual_mmq_rdna2(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Distinct module + function name from the pre-existing
        // `gemm_hfq4g256_residual_mmq` (llama.cpp-style, RDNA3+ via
        // HIPFIRE_WO_MMQ=1) to avoid kernel-cache collision.
        self.ensure_kernel(
            "gemm_hfq4g256_residual_mmq_rdna2",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_RDNA2_SRC,
            "gemm_hfq4g256_residual_mmq_rdna2",
        )?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions["gemm_hfq4g256_residual_mmq_rdna2"];

        let mut ap = a_raw.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut yp = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        const MMQ_Y: usize = 128;
        const MMQ_X: usize = 32;
        const X_STRIDE: usize = 40;
        const Y_STRIDE: usize = 36;
        let shared_mem = (MMQ_Y * X_STRIDE * 4 + MMQ_Y * 8 + MMQ_X * Y_STRIDE * 4) as u32;

        let row_tiles = (m + MMQ_Y - 1) / MMQ_Y;
        let col_tiles = (batch_size + MMQ_X - 1) / MMQ_X;

        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq4g256_residual_mmq_rdna2",
            bytes,
        );
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, col_tiles as u32, 1],
                [32, 4, 1],
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
    /// Path 2 grouped-WMMA-GEMM for MoE prefill (gate_up or down).
    /// Each WMMA tile picks its expert via `expert_tile_ids[tile_y]` and
    /// gathers its B-operand rows via `sorted_slot_index`; -1 padding
    /// lanes contribute zeros. Writes `Y_grouped[m_total × M]` direct.
    ///
    /// The companion combine kernel (Stage 3) fans Y_grouped back to the
    /// per-token gate_batch/up_batch streams (or applies topk_weights for
    /// the down combine).
    /// `x_row_div` selects the X gather layout:
    ///   gate_up: x_src = x_rot_batch [N × K], x_row_div = K_TOP
    ///   down:    x_src = rot_batch [N*K_TOP × K], x_row_div = 1
    /// `x_src_rows` is the number of rows in x_src (N or N*K_TOP).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_moe_grouped_wmma_k2(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to FP16)
        y_grouped: &GpuTensor,          // [m_total × M] f32, written direct
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // gfx1151 (Strix Halo iGPU) i8 MMQ port: lift the compute ceiling
        // from ~71 (FP16 WMMA) to ~140 TFLOPS (i8 WMMA). Opt-out via
        // HIPFIRE_MOE_GROUPED_I8=0; default ON for gfx1151 only.
        let use_i8_gfx1151 =
            self.arch_caps.is_gfx1151() && self.flags.moe_grouped_i8.unwrap_or(true);
        if use_i8_gfx1151 {
            // Optional deeper-pipeline variants. On gfx1151, k8 is default ON
            // after the 122B A10B pp128 profile showed ~2.2x over k2 for the
            // HFQ4 grouped-MoE hotspot; opt out with HIPFIRE_MOE_GROUPED_I8_K8=0.
            // k4 remains available via HIPFIRE_MOE_GROUPED_I8_K4=1.
            // Same kernarg layout + scatter contract as the k2 default.
            // - k8: processes all 4 sub-blocks of one Q8_1 block per inner
            //   iteration (8 WMMAs into 4 independent int32 accumulators).
            // - k4: pairs adjacent Q8_1 sub-blocks (4 WMMAs into 2 accumulators).
            // - k2: one sub-block per inner iteration.
            let use_4w = self.flags.moe_grouped_i8_4w;
            let use_k8 = self.flags.moe_grouped_i8_k8;
            let use_k4 = self.flags.moe_grouped_i8_k4;
            if use_4w {
                return self.gemm_hfq4g256_moe_grouped_mmq_k8_4w_gfx1151(
                    expert_weight_ptrs,
                    expert_tile_ids,
                    sorted_slot_index,
                    x_src,
                    y_grouped,
                    m,
                    k,
                    x_row_div,
                    m_total,
                    x_src_rows,
                );
            }
            if use_k8 {
                return self.gemm_hfq4g256_moe_grouped_mmq_k8_gfx1151(
                    expert_weight_ptrs,
                    expert_tile_ids,
                    sorted_slot_index,
                    x_src,
                    y_grouped,
                    m,
                    k,
                    x_row_div,
                    m_total,
                    x_src_rows,
                );
            }
            if use_k4 {
                return self.gemm_hfq4g256_moe_grouped_mmq_k4_gfx1151(
                    expert_weight_ptrs,
                    expert_tile_ids,
                    sorted_slot_index,
                    x_src,
                    y_grouped,
                    m,
                    k,
                    x_row_div,
                    m_total,
                    x_src_rows,
                );
            }
            return self.gemm_hfq4g256_moe_grouped_mmq_gfx1151(
                expert_weight_ptrs,
                expert_tile_ids,
                sorted_slot_index,
                x_src,
                y_grouped,
                m,
                k,
                x_row_div,
                m_total,
                x_src_rows,
            );
        }
        // gfx11 dGPU i8 MMQ port (gfx1100/1101/1102/1103 — 7900 XTX, 7800/
        // 7700, 7600, Phoenix mobile). Same lift as gfx1151: doubles the
        // compute ceiling on this compute-bound grouped GEMM path.
        // Opt-out via HIPFIRE_MOE_GROUPED_I8=0; default ON for these archs.
        let use_i8_gfx11_dgpu = (self.arch.starts_with("gfx1100")
            || self.arch.starts_with("gfx1101")
            || self.arch.starts_with("gfx1102")
            || self.arch.starts_with("gfx1103"))
            && self.flags.moe_grouped_i8.unwrap_or(true);
        if use_i8_gfx11_dgpu {
            // k4 default ON: deeper K-tile pipeline gives +2.8% over k2 on
            // gfx1100 (A/B confirmed 2026-05-19 k9lin 7900 XTX); same
            // structural pattern as gfx1151's +4.6%. k2 alone was a wash vs
            // FP16, so k4 is what makes the dGPU i8 path actually worth
            // shipping. Opt out with HIPFIRE_MOE_GROUPED_I8_K4=0.
            let use_k4 = self.flags.moe_grouped_i8_k4;
            if use_k4 {
                return self.gemm_hfq4g256_moe_grouped_mmq_k4_gfx11_dgpu(
                    expert_weight_ptrs,
                    expert_tile_ids,
                    sorted_slot_index,
                    x_src,
                    y_grouped,
                    m,
                    k,
                    x_row_div,
                    m_total,
                    x_src_rows,
                );
            }
            return self.gemm_hfq4g256_moe_grouped_mmq_gfx11_dgpu(
                expert_weight_ptrs,
                expert_tile_ids,
                sorted_slot_index,
                x_src,
                y_grouped,
                m,
                k,
                x_row_div,
                m_total,
                x_src_rows,
            );
        }
        // gfx12 (RDNA4 — R9700/gfx1201, gfx1200) i8 MMQ port. Correctness PASS
        // (NRMSE ~0.4% on A3B shapes vs FP16 reference) but empirical perf on
        // 2026-05-19 R9700 A3B prefill (256-token, 5-run median): 2960 → 2607
        // tok/s = **-11.6% regression**. Per-call kernel time 279µs (FP16) →
        // 408µs (i8) = +46% kernel slowdown. Theoretical 2× i8 WMMA FLOP rate
        // is offset by per-sub-block scale FMA dependency chain (8 INT→FLOAT
        // conversions + 16 FMAs per sub-block, fully serial after each WMMA
        // pair). Same pattern as documented synth-win → prod-falsify cases
        // (FP8 WMMA HFQ4G32 2026-05-10, gfx11 dot2 trickle-down 2026-05-11).
        // Shipped as opt-in research artifact; default OFF for gfx12.
        // Opt-in via HIPFIRE_MOE_GROUPED_I8=1 to evaluate on other shapes.
        let use_i8_gfx12 = self.arch_caps.is_rdna4() && self.flags.moe_grouped_i8.unwrap_or(false);
        if use_i8_gfx12 {
            // k4 variant: 4 sub-blocks paired per inner iteration, 8 WMMAs
            // into 4 independent int32 accumulators before scale-FMA chain
            // resolves. Experimental — separate gate from the gfx11_dgpu k4
            // (which is default-on) because the gfx12 i8 path itself is
            // default-off pending recovery from the -11.6% regression vs FP16.
            let use_k4 = self.flags.moe_grouped_i8_k4_gfx12;
            if use_k4 {
                return self.gemm_hfq4g256_moe_grouped_mmq_k4_gfx12(
                    expert_weight_ptrs,
                    expert_tile_ids,
                    sorted_slot_index,
                    x_src,
                    y_grouped,
                    m,
                    k,
                    x_row_div,
                    m_total,
                    x_src_rows,
                );
            }
            return self.gemm_hfq4g256_moe_grouped_mmq_gfx12(
                expert_weight_ptrs,
                expert_tile_ids,
                sorted_slot_index,
                x_src,
                y_grouped,
                m,
                k,
                x_row_div,
                m_total,
                x_src_rows,
            );
        }
        // gfx12 (RDNA4) needs the _gfx12 WMMA intrinsic; gfx11 (RDNA3) and
        // older RDNA archs use the base _w32 intrinsic from the k2 sibling.
        let is_gfx12 = self.arch_caps.is_rdna4();
        // 2×1 M-direction reg-blocked variant (gfx12 only for now). Env-gated.
        let use_m2 = is_gfx12 && self.flags.moe_grouped_m2;
        let (kernel_name, kernel_src) = if use_m2 {
            (
                "gemm_hfq4g256_moe_grouped_wmma_m2_gfx12",
                kernels::GEMM_HFQ4G256_MOE_GROUPED_WMMA_M2_GFX12_SRC,
            )
        } else if is_gfx12 {
            (
                "gemm_hfq4g256_moe_grouped_wmma_gfx12",
                kernels::GEMM_HFQ4G256_MOE_GROUPED_WMMA_GFX12_SRC,
            )
        } else {
            (
                "gemm_hfq4g256_moe_grouped_wmma_k2",
                kernels::GEMM_HFQ4G256_MOE_GROUPED_WMMA_K2_SRC,
            )
        };
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
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

        let row_tile_stride = if use_m2 { 32 } else { 16 };
        let row_tiles = ((m + row_tile_stride - 1) / row_tile_stride) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW estimate: each tile loads one expert weight row band (m_total/16 tiles
        // share the same expert avg ~ m_total/E times) + gathers X + writes Y.
        let bytes =
            m_total * k * 2 + (m_total * m) * 4 + (crate::profile::gemv_hfq4g256_bytes(m, k));
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
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// gfx1151 (Strix Halo iGPU) i8 MMQ MoE grouped GEMM. Ports the i8
    /// WMMA MMQ pattern from `gemm_hfq4g256_residual_mmq` to the SGLang
    /// grouped scatter dispatch. X is pre-quantized to Q8_1 via
    /// `ensure_q8_1_mmq_x` (same buffer/scratch as the residual MMQ path).
    ///
    /// Kernarg layout matches the FP16 sister except the X pointer is the
    /// Q8_1 packed scratch (not the FP16 conversion buffer) and there is
    /// one extra `x_src_rows` arg (Q8_1 layout is `[K/128 × x_src_rows]`,
    /// so the kernel needs `x_src_rows` to compute the row stride).
    ///
    /// Used as a drop-in replacement for `gemm_hfq4g256_moe_grouped_wmma_k2`
    /// on gfx1151 when `HIPFIRE_MOE_GROUPED_I8 != "0"` (default ON for
    /// gfx1151). The FP16 sister still owns gfx12/gfx11-non-1151 paths.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_moe_grouped_mmq_gfx1151(
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
        let kernel_name = "gemm_hfq4g256_moe_grouped_mmq_gfx1151";
        let kernel_src = kernels::GEMM_HFQ4G256_MOE_GROUPED_MMQ_GFX1151_SRC;
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
        // BW estimate: Q8_1 X reads + HFQ4 weights + Y writes. Q8_1 = ~1B/elem
        // (slightly more for the per-sub-block (d,sum) metadata) vs FP16 = 2B/elem.
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
    /// gfx1151 (Strix Halo iGPU) i8 MMQ MoE grouped GEMM — k4 (deeper
    /// K-tile pipeline) variant. Drop-in for `gemm_hfq4g256_moe_grouped_mmq_gfx1151`
    /// — same kernarg layout, same grid/block geometry, same scatter
    /// contract. The kernel pairs adjacent Q8_1 sub-blocks so each inner
    /// iteration issues 4 WMMAs into 2 independent int32 accumulators
    /// before the per-sub-block scale FMA resolves. Output is
    /// numerically equivalent to k2 modulo int32 summation-order
    /// (commutative; integer-addition reductions are exact).
    ///
    /// Opt-IN via `HIPFIRE_MOE_GROUPED_I8_K4=1` (default OFF). Routes
    /// through the same wrapper as k2 (`gemm_hfq4g256_moe_grouped_wmma_k2`),
    /// which gates on `HIPFIRE_MOE_GROUPED_I8 != "0"` first.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_moe_grouped_mmq_k4_gfx1151(
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
        let kernel_name = "gemm_hfq4g256_moe_grouped_mmq_k4_gfx1151";
        let kernel_src = kernels::GEMM_HFQ4G256_MOE_GROUPED_MMQ_K4_GFX1151_SRC;
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
        // BW estimate: same as the k2 sibling — Q8_1 X reads + HFQ4 weights
        // + Y writes. k4 is a pure unroll-depth change, no extra memory traffic.
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
    /// gfx1151 (Strix Halo iGPU) i8 MMQ MoE grouped GEMM — k8 (deepest
    /// K-tile pipeline) variant. Drop-in for `gemm_hfq4g256_moe_grouped_mmq_gfx1151`
    /// — same kernarg layout, same grid/block geometry, same scatter
    /// contract. The kernel processes all 4 sub-blocks of one Q8_1 block
    /// per inner iteration — 8 WMMAs into 4 independent int32 accumulators
    /// before the per-sub-block scale FMA resolves. Output is numerically
    /// equivalent to k2/k4 modulo int32 summation-order (commutative;
    /// integer-addition reductions are exact).
    ///
    /// Opt-IN via `HIPFIRE_MOE_GROUPED_I8_K8=1` (default OFF). Routes
    /// through the same wrapper as k2/k4 (`gemm_hfq4g256_moe_grouped_wmma_k2`),
    /// which gates on `HIPFIRE_MOE_GROUPED_I8 != "0"` first; k8 takes
    /// priority over k4 if both env vars are set.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_moe_grouped_mmq_k8_gfx1151(
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
        let kernel_name = "gemm_hfq4g256_moe_grouped_mmq_k8_gfx1151";
        let kernel_src = kernels::GEMM_HFQ4G256_MOE_GROUPED_MMQ_K8_GFX1151_SRC;
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
        // BW estimate: same as the k2/k4 siblings — Q8_1 X reads + HFQ4 weights
        // + Y writes. k8 is a pure unroll-depth change, no extra memory traffic.
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
    /// gfx1151 i8 MMQ MoE grouped GEMM — 4-warp k8 variant. Shares the
    /// routed Q8_1 activation blocks across four adjacent 16-row tiles via
    /// LDS while preserving the default k8 per-warp math.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_moe_grouped_mmq_k8_4w_gfx1151(
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
        let kernel_name = "gemm_hfq4g256_moe_grouped_mmq_k8_4w_gfx1151";
        let kernel_src = kernels::GEMM_HFQ4G256_MOE_GROUPED_MMQ_K8_4W_GFX1151_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
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

        let row_tiles = ((m + 63) / 64) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let bytes = (m_total * k) + (m_total * m) * 4 + (crate::profile::gemv_hfq4g256_bytes(m, k));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [128, 1, 1],
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
    /// k4 (deeper K-tile pipeline) variant for gfx11 dGPUs. Mirrors the
    /// gfx1151 k4 design (validated +4.6% over k2 there with zero spills).
    /// Pairs adjacent Q8_1 sub-blocks for 4 WMMAs into 2 independent int32
    /// accumulators per inner iteration; numerically equivalent to k2
    /// modulo int32 summation order. Opt-IN via
    /// `HIPFIRE_MOE_GROUPED_I8_K4=1` (default OFF on dGPU — k2 was no-op
    /// vs FP16, so k4 is the real test of the pipeline-depth hypothesis).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_moe_grouped_mmq_k4_gfx11_dgpu(
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
        let kernel_name = "gemm_hfq4g256_moe_grouped_mmq_k4_gfx11_dgpu";
        let kernel_src = kernels::GEMM_HFQ4G256_MOE_GROUPED_MMQ_K4_GFX11_DGPU_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
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
    /// gfx11 dGPU i8 MMQ MoE grouped GEMM (gfx1100/1101/1102/1103 — 7900 XTX,
    /// 7800/7700, 7600, Phoenix mobile). Same kernarg layout as the gfx1151
    /// i8 sister (10-arg variant with explicit `x_src_rows` for the Q8_1
    /// K-block stride). X pre-quantized to Q8_1 via `ensure_q8_1_mmq_x`.
    ///
    /// Used as a drop-in replacement for `gemm_hfq4g256_moe_grouped_wmma_k2`
    /// on gfx11 dGPUs when `HIPFIRE_MOE_GROUPED_I8 != "0"` (default ON for
    /// gfx1100/1101/1102/1103). Roughly 2× the FLOP rate of the FP16 sister
    /// on this compute-bound grouped MoE GEMM path.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_moe_grouped_mmq_gfx11_dgpu(
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
        let kernel_name = "gemm_hfq4g256_moe_grouped_mmq_gfx11_dgpu";
        let kernel_src = kernels::GEMM_HFQ4G256_MOE_GROUPED_MMQ_GFX11_DGPU_SRC;
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
        // BW estimate: Q8_1 X reads (~1 B/elem incl. (d,sum) metadata) +
        // HFQ4 weights + Y writes. Distinct from the FP16 sister (which
        // uses 2 B/elem for X).
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
    /// HFQ6/MQ6 sister of `gemm_hfq4g256_moe_grouped_wmma_k2`. Same kernarg
    /// layout + grouped dispatch contract; differs only in the 200 B/group
    /// HFQ6 dequant inner loop. Unblocks AWQ A3B prefill (where ~50% of
    /// experts are MQ6 not MQ4 in the production AWQ A3B build at
    /// /mnt/nas/kaden/hipfire/mi300x-v3/qwen3-35b-a3b-awq-mq4.hfq).
    ///
    /// `x_row_div` selects the X gather layout (identical to the HFQ4 sister):
    ///   gate_up: x_src = x_rot_batch [N × K], x_row_div = K_TOP
    ///   down:    x_src = rot_batch [N*K_TOP × K], x_row_div = 1
    /// `x_src_rows` is the number of rows in x_src (N or N*K_TOP).
    ///
    /// **gfx12 + gfx1151 only.** Panics with a clear message on other archs;
    /// broader gfx11 can be added later by mirroring the HFQ4 `_k2` sibling.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq6g256_moe_grouped_wmma(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to FP16)
        y_grouped: &GpuTensor,          // [m_total × M] f32, written direct
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !(self.arch_caps.is_rdna4() || self.arch == "gfx1151") {
            panic!(
                "gemm_hfq6g256_moe_grouped_wmma: only gfx12/gfx1151 kernels are wired \
                 (current arch = {}). Add a sibling before admitting this arch.",
                self.arch
            );
        }
        // gfx1151 default: 4-warp 64-row tile sharing one staged 16-slot
        // X slice across four row-warps. Opt out with HIPFIRE_MOE_HFQ6_4W=0.
        //
        // v2 lever (M-direction 2×1 reg-block, env-gated) remains available
        // behind HIPFIRE_MOE_HFQ6_V2=1 for gfx12 or gfx1151 when the 4w path
        // is explicitly disabled. Each warp covers 32 rows × 16 slots (vs
        // 16×16); B-load halved per output. Compatible with existing
        // BLOCK_M=16 scatter — only the M (row) dimension is restrided.
        // The slot tile stride stays at 16 so expert-boundary safety is
        // unchanged from v1.
        let use_4w = self.arch == "gfx1151" && self.flags.moe_hfq6_4w;
        let use_v2 = self.flags.moe_hfq6_v2;
        let (kernel_name, kernel_src, row_tile_stride, block_dim) = if use_4w {
            (
                "gemm_hfq6g256_moe_grouped_wmma_4w_gfx1151",
                kernels::GEMM_HFQ6G256_MOE_GROUPED_WMMA_4W_GFX1151_SRC,
                64usize,
                128u32,
            )
        } else if self.arch == "gfx1151" && use_v2 {
            (
                "gemm_hfq6g256_moe_grouped_wmma_v2_gfx1151",
                kernels::GEMM_HFQ6G256_MOE_GROUPED_WMMA_V2_GFX1151_SRC,
                32usize,
                32u32,
            )
        } else if self.arch == "gfx1151" {
            (
                "gemm_hfq6g256_moe_grouped_wmma_gfx1151",
                kernels::GEMM_HFQ6G256_MOE_GROUPED_WMMA_GFX1151_SRC,
                16usize,
                32u32,
            )
        } else if use_v2 {
            (
                "gemm_hfq6g256_moe_grouped_wmma_v2_gfx12",
                kernels::GEMM_HFQ6G256_MOE_GROUPED_WMMA_V2_GFX12_SRC,
                32usize,
                32u32,
            )
        } else {
            (
                "gemm_hfq6g256_moe_grouped_wmma_gfx12",
                kernels::GEMM_HFQ6G256_MOE_GROUPED_WMMA_GFX12_SRC,
                16usize,
                32u32,
            )
        };
        let slot_tile_stride = 16usize;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
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

        let row_tiles = ((m + row_tile_stride - 1) / row_tile_stride) as u32;
        let slot_tiles = ((m_total + slot_tile_stride - 1) / slot_tile_stride) as u32;
        // BW estimate uses the HFQ6 weight footprint (200 B/group vs HFQ4's 136 B).
        let bytes =
            m_total * k * 2 + (m_total * m) * 4 + (crate::profile::gemv_hfq6g256_bytes(m, k));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [block_dim, 1, 1],
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
    /// HFQ3/MQ3 sister of `gemm_hfq4g256_moe_grouped_wmma_k2` for the
    /// MoE Path-2 grouped-WMMA-GEMM. Same contract: each WMMA tile picks
    /// its expert via `expert_tile_ids[tile_y]` (-1 sentinel = early
    /// return) and gathers its B-operand rows via `sorted_slot_index`
    /// (-1 padding lanes contribute zeros). Writes `Y_grouped[m_total ×
    /// M]` direct.
    ///
    /// `x_row_div` selects the X gather layout:
    ///   gate_up: x_src = x_rot_batch [N × K], x_row_div = K_TOP
    ///   down:    x_src = rot_batch [N*K_TOP × K], x_row_div = 1
    /// `x_src_rows` is the number of rows in x_src (N or N*K_TOP).
    ///
    /// **gfx12 + gfx1151 only** for now. Other archs panic; integration
    /// with qwen35 MoE admission is gated on this same arch set.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq3g256_moe_grouped_wmma(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to FP16)
        y_grouped: &GpuTensor,          // [m_total × M] f32, written direct
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !(self.arch_caps.is_rdna4() || self.arch == "gfx1151") {
            panic!(
                "gemm_hfq3g256_moe_grouped_wmma: only gfx12/gfx1151 is supported; \
                 caller must gate before dispatch. Arch: {}",
                self.arch
            );
        }
        let (kernel_name, kernel_src) = if self.arch == "gfx1151" {
            (
                "gemm_hfq3g256_moe_grouped_wmma_gfx1151",
                kernels::GEMM_HFQ3G256_MOE_GROUPED_WMMA_GFX1151_SRC,
            )
        } else {
            (
                "gemm_hfq3g256_moe_grouped_wmma_gfx12",
                kernels::GEMM_HFQ3G256_MOE_GROUPED_WMMA_GFX12_SRC,
            )
        };
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
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

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW estimate: HFQ3 row footprint is groups_per_row × 104 B (vs
        // HFQ4's 136 B); reuse the gemv_hfq3g256_bytes profile helper.
        let bytes =
            m_total * k * 2 + (m_total * m) * 4 + (crate::profile::gemv_hfq3g256_bytes(m, k));
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
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Batched HFQ4-G256 GEMM with fused residual add:
    ///   for b in 0..batch_size: y[b][row] += A[row] · x[b]
    ///
    /// Bitwise-identical output to calling `gemv_hfq4g256_residual` N times
    /// (preserves the 4-accumulator interleave and pairwise final combine),
    /// so safe to use in the quality-gated forward path. Each block handles
    /// one row × up to BATCH_TILE batch elements, amortizing the weight
    /// fetch across the batch loop.
    ///
    /// `x`: [batch_size × K] row-major, `y`: [batch_size × M] row-major.
    /// `y` must already hold the residual summand to accumulate into.
    pub fn gemm_hfq4g256_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // gfx94x MFMA-direct opt-in: skips FP16 shadow + rocBLAS launch.
        // Opt-in via HIPFIRE_GFX942_MFMA_PREFILL=1 while validating; this
        // fires BEFORE the rocBLAS branch on purpose (rocBLAS goes through
        // FP16 dequant shadow, which is the cost we want to avoid).
        {
            let mfma_v = self.flags.gfx942_mfma_prefill.clone();
            let want = mfma_v.as_deref();
            if (want == Some("1") || want == Some("2") || want == Some("3") || want == Some("4"))
                && self.arch_caps.is_cdna3()
                && batch_size >= 16
                && m % 16 == 0
                && k % 256 == 0
                && !self.capture_mode
            {
                if want == Some("4") && batch_size % 64 == 0 && m % 16 == 0 {
                    return self
                        .gemm_hfq4g256_residual_mfma_v4_gfx942(a_raw, x, y, m, k, batch_size);
                }
                if want == Some("3") && batch_size % 32 == 0 && m % 32 == 0 {
                    return self
                        .gemm_hfq4g256_residual_mfma_v3_gfx942(a_raw, x, y, m, k, batch_size);
                }
                if want == Some("2") && batch_size % 32 == 0 && m % 32 == 0 {
                    return self
                        .gemm_hfq4g256_residual_mfma_v2_gfx942(a_raw, x, y, m, k, batch_size);
                }
                return self.gemm_hfq4g256_residual_mfma_gfx942(a_raw, x, y, m, k, batch_size);
            }
        }
        // CDNA3 MFMA path — Y += X·W^T via rocBLAS with beta=1.
        if self.rocblas_arch_eligible()
            && batch_size >= self.rocblas_min_batch()
            && self.rocblas.is_some()
            && !self.capture_mode
        {
            if let Ok(Some(shadow_ptr)) = self.ensure_fp16_shadow(a_raw, m, k) {
                let x_fp16 = self.ensure_fp16_x(x, batch_size * k)?;
                let w_buf = unsafe { DeviceBuffer::from_raw(shadow_ptr, (m * k) * 2) };
                let x_buf = unsafe { DeviceBuffer::from_raw(x_fp16, (batch_size * k) * 2) };
                let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
                    + batch_size * k * 4
                    + batch_size * m * 4 * 2;
                let timer = crate::profile::begin_timer(
                    &self.hip,
                    "gemm",
                    "gemm_hfq4g256_residual_rocblas",
                    bytes,
                );
                let result = self
                    .rocblas_gemm_hfq4_prefill_residual(&w_buf, &x_buf, &y.buf, m, batch_size, k);
                std::mem::forget(w_buf);
                std::mem::forget(x_buf);
                if let Some(t) = timer {
                    t.finish(&self.hip);
                }
                return result;
            }
        }

        // HFQ4 wave32 MMQ residual on RDNA2+. Default-on for the allowlist
        // arch set (issue #300 gate removal — +210% prefill on gfx1031 4B
        // MQ4 pp128, KLD-neutral; escape hatch HIPFIRE_HFQ4_MMQ_RDNA2=0).
        // HFQ4's cheaper 4-bit nibble unpack lets MMQ beat the fp16
        // fallback. Env gate is OnceLock-cached.
        //
        // Issue #299 follow-up: route through the tile-size auto-selector
        // so narrow-batch calls pick mmq_x=16 and long-prefill picks
        // mmq_x=32_y64 (MQ3 phase-2 finding). All variants clamp M-tail
        // internally, so no alignment check needed.
        if self.flags.hfq4_residual_fast && self.hfq4g256_mmq_gfx1151_enabled(m, k, batch_size) {
            return self.gemm_hfq4g256_mmq_gfx1151(a_raw, x, y, m, k, batch_size, true);
        }
        if self.flags.hfq4_residual_fast && batch_size > 1 && self.arch_caps.has_hfq4_mmq() {
            return self.gemm_hfq4g256_residual_mmq_rdna2_auto(a_raw, x, y, m, k, batch_size);
        }

        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if self.flags.hfq4_residual_fast && batch_size > 1 && !self.flags.fp16_disabled {
            // gfx906 dp4a MMQ residual path — default-on at batch ≥ 8 per
            // should_use_mmq's gfx906 default. Distinguishes two reasons
            // MMQ might NOT fire:
            //   (a) batch_size below cutover → fall to dp4a batched residual
            //   (b) mmq_screen_weight rejected the weight → fall to fp16
            //       (preserves screen's design intent: rejected weights go
            //       to a higher-precision fallback, NOT to dp4a which has
            //       the same Q8_1 quantization step that MMQ failed on).
            let mut mmq_screen_rejected = false;
            if self.arch_caps.is_gfx906() && self.arch_caps.should_use_mmq(batch_size) {
                let use_mmq = if self.mmq_screen {
                    self.mmq_screen_weight(a_raw, m, k)
                } else {
                    true
                };
                if use_mmq {
                    return self.gemm_hfq4g256_residual_mmq_gfx906(a_raw, x, y, m, k, batch_size);
                }
                mmq_screen_rejected = self.mmq_screen;
            }

            // gfx906 dp4a batched residual (issue #276 Gap 2, HFQ4 sibling of
            // HFQ6 Phase A.2). Fires for B>1 below the MMQ cutover (B ∈
            // [2, 7] on gfx906 by should_use_mmq's default). Wins on
            // per-call ALU (dp4a issues 4 int8 multiplies + 4 accumulates
            // per cycle, vs FP wave64 hybrid's hfma2 at 2 mul + 2 add per
            // cycle → 2× FLOPs/cycle) and reuses the existing Q8_1 scratch.
            //
            // Skipped when MMQ screening rejected (preserves screen's
            // higher-precision fallback intent — dp4a has the same Q8_1
            // quantization step that MMQ already failed on for this
            // weight).
            //
            // The `!self.capture_mode` guard: `ensure_q8_1_mmq_x` (and the
            // downstream `ensure_kernel` for this kernel) can fire `hipMalloc`
            // / JIT-compile on first use, both unsafe inside an active capture.
            // The internal Q8_1 quantize launch itself goes through
            // `launch_maybe_blob` and IS recorded into the captured graph;
            // the guard protects only first-use-only side effects.
            if !mmq_screen_rejected && self.arch_caps.gemv_dp4a_enabled() && !self.capture_mode {
                return self.gemm_hfq4g256_residual_wave64_dp4a(a_raw, x, y, m, k, batch_size);
            }

            // Wave64 FP16 hybrid — best of both worlds for gfx906 (MI50).
            // Also the safe fallback when MMQ screen rejected the weight.
            if self.arch_caps.is_gcn5_wave64() {
                return self.gemm_hfq4g256_residual_fp16_wave64(a_raw, x, y, m, k, batch_size);
            }

            // Opt-in MMQ path (RDNA3/3.5, HIPFIRE_MMQ=1 or HIPFIRE_WO_MMQ=1).
            if self.flags.wo_mmq || self.arch_caps.should_use_mmq(batch_size) {
                let use_mmq = if self.mmq_screen {
                    self.mmq_screen_weight(a_raw, m, k)
                } else {
                    true
                };
                if use_mmq {
                    return self.gemm_hfq4g256_residual_mmq(a_raw, x, y, m, k, batch_size);
                }
            }

            // WMMA on gfx12 (RDNA4): K2-unroll port
            if self.arch_caps.has_wmma_w32_gfx12() {
                return self.gemm_hfq4g256_residual_wmma_gfx12(a_raw, x, y, m, k, batch_size);
            }

            // WMMA on gfx11+ (RDNA3): 16×16 tiled, ~8-10× over scalar
            if self.arch_caps.has_wmma_w32() {
                return self.gemm_hfq4g256_residual_wmma(a_raw, x, y, m, k, batch_size);
            }

            // FP16 packed on all other RDNA: ~15% prefill improvement
            return self.gemm_hfq4g256_residual_fp16(a_raw, x, y, m, k, batch_size);
        }

        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let (func_name, block, grid_div): (&str, [u32; 3], u32) = if cdna_wave64 {
            self.ensure_kernel(
                "gemm_hfq4g256_residual_wave64",
                kernels::GEMM_HFQ4G256_RESIDUAL_WAVE64_SRC,
                "gemm_hfq4g256_residual_wave64",
            )?;
            ("gemm_hfq4g256_residual_wave64", [64, 1, 1], 2)
        } else {
            self.ensure_kernel(
                "gemm_hfq4g256_residual",
                kernels::GEMM_HFQ4G256_RESIDUAL_SRC,
                "gemm_hfq4g256_residual",
            )?;
            ("gemm_hfq4g256_residual", [32, 1, 1], 1)
        };
        let func = &self.functions[func_name];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
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

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let grid_x = (m as u32 + grid_div - 1) / grid_div;

        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 4 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_residual", bytes);
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
    pub fn gemm_hfq4g256_residual_exact(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let (func_name, block, grid_div): (&str, [u32; 3], u32) = if cdna_wave64 {
            self.ensure_kernel(
                "gemm_hfq4g256_residual_wave64",
                kernels::GEMM_HFQ4G256_RESIDUAL_WAVE64_SRC,
                "gemm_hfq4g256_residual_wave64",
            )?;
            ("gemm_hfq4g256_residual_wave64", [64, 1, 1], 2)
        } else {
            self.ensure_kernel(
                "gemm_hfq4g256_residual",
                kernels::GEMM_HFQ4G256_RESIDUAL_SRC,
                "gemm_hfq4g256_residual",
            )?;
            ("gemm_hfq4g256_residual", [32, 1, 1], 1)
        };
        let func = &self.functions[func_name];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
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

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let grid_x = (m as u32 + grid_div - 1) / grid_div;

        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 4 + batch_size * m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_residual_exact", bytes);
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
    /// Batched HFQ3-G256 GEMM with fused residual add (MQ3 path).
    ///
    /// HFQ3 sibling of `gemm_hfq4g256_residual` — single scalar variant,
    /// 104 B group stride and 3-bit unpack. Phase 1 of the gfx10 MQ3
    /// prefill plan. Used for batched prefill of the post-attention
    /// (wo) and post-FFN (w_down) projections.
    ///
    /// `x`: [batch_size × K] row-major, `y`: [batch_size × M] row-major.
    /// `y` must already hold the residual summand to accumulate into.
    pub fn gemm_hfq3g256_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Phase 3 experimental: wave32 MMQ is default-on for the supported
        // allowlist unless HIPFIRE_HFQ3_MMQ=0. Layer-gate is a no-op when
        // unset (#302).
        if batch_size > 1 && self.arch_caps.has_hfq3_mmq() && self.flags.hfq3_mmq_layer_gate_pass()
        {
            return self.gemm_hfq3g256_residual_mmq(a_raw, x, y, m, k, batch_size);
        }
        // FP16 fast paths — Phase 2b (dot2) + Phase 2c (fp16 fallback).
        // Layer-aware FP16 gate (#302).
        if batch_size > 1 && !self.flags.fp16_disabled_for_current_layer() {
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_hfq3g256_residual_dot2(a_raw, x, y, m, k, batch_size);
            }
            return self.gemm_hfq3g256_residual_fp16(a_raw, x, y, m, k, batch_size);
        }
        self.ensure_kernel(
            "gemm_hfq3g256_residual",
            kernels::GEMM_HFQ3G256_RESIDUAL_SRC,
            "gemm_hfq3g256_residual",
        )?;
        let func = &self.functions["gemm_hfq3g256_residual"];

        let mut ap = a_raw.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let bytes = crate::profile::gemm_hfq3g256_bytes(m, k, batch_size);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq3g256_residual", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_tiles as u32, 1],
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
    /// v_dot2_f32_f16-accelerated batched HFQ3-G256 residual GEMM (Y += A·X).
    /// HFQ3 sibling of `gemm_hfq4g256_residual_fp16`, upgraded from
    /// v_pk_fma_f16 to v_dot2_f32_f16. Phase 2b.
    pub fn gemm_hfq3g256_residual_dot2(
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
            "gemm_hfq3g256_residual_dot2",
            kernels::GEMM_HFQ3G256_RESIDUAL_DOT2_SRC,
            "gemm_hfq3g256_residual_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_hfq3g256_residual_dot2"];

        let mut ap = a_raw.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yp = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let bytes = crate::profile::gemm_hfq3g256_bytes(m, k, batch_size) + batch_size * k * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq3g256_residual_dot2", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_tiles as u32, 1],
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
    /// v_pk_fma_f16-accelerated batched HFQ3-G256 residual GEMM (Y += A·X).
    /// Fallback for archs without the dot extension (gfx1010, gfx1013).
    /// Phase 2c of the gfx10 MQ3 prefill plan.
    pub fn gemm_hfq3g256_residual_fp16(
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
            "gemm_hfq3g256_residual_fp16",
            kernels::GEMM_HFQ3G256_RESIDUAL_FP16_SRC,
            "gemm_hfq3g256_residual_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_hfq3g256_residual_fp16"];

        let mut ap = a_raw.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yp = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let bytes = crate::profile::gemm_hfq3g256_bytes(m, k, batch_size) + batch_size * k * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq3g256_residual_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_tiles as u32, 1],
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
    /// FP16-input batched HFQ4-G256 GEMM with residual add.
    /// Converts X from FP32 to FP16 (halving X bandwidth), then runs the
    /// FP16-packed GEMM kernel. The conversion is a one-shot pass amortized
    /// across M rows.
    pub fn gemm_hfq4g256_residual_fp16(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor, // FP32 [batch_size × K]
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq4g256_residual_fp16",
            kernels::GEMM_HFQ4G256_RESIDUAL_FP16_SRC,
            "gemm_hfq4g256_residual_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        // FP16 GEMM
        let func = &self.functions["gemm_hfq4g256_residual_fp16"];
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

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * k * 2  // FP16 X (half bandwidth!)
            + batch_size * m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_residual_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_tiles as u32, 1],
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
    pub fn gemm_hfq4g256_residual_fp16_wave64(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor, // FP32 [batch_size × K]
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq4g256_residual_fp16_wave64",
            kernels::GEMM_HFQ4G256_RESIDUAL_FP16_WAVE64_SRC,
            "gemm_hfq4g256_residual_fp16_wave64",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let func = &self.functions["gemm_hfq4g256_residual_fp16_wave64"];
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

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let grid_x = (m as u32 + 1) / 2;

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * k * 2  // FP16 X (half bandwidth!)
            + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq4g256_residual_fp16_wave64",
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
    /// Experimental llama.cpp-style MMQ residual GEMM for HFQ4-G256.
    /// Opt-in only via `HIPFIRE_WO_MMQ=1` while the tiled path is validated.
    pub fn gemm_hfq4g256_residual_mmq(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.hfq4g256_mmq_gfx1151_enabled(m, k, batch_size) {
            return self.gemm_hfq4g256_mmq_gfx1151(a_raw, x, y, m, k, batch_size, true);
        }
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let kernel_name = if m % 128 == 0 && batch_size % 128 == 0 {
            "gemm_hfq4g256_residual_mmq_full_add"
        } else {
            "gemm_hfq4g256_residual_mmq"
        };
        self.ensure_kernel(
            "gemm_hfq4g256_residual_mmq",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_SRC,
            kernel_name,
        )?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut xq_ptr = x_q8_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut add_val = 1i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut add_val as *mut _ as *mut c_void,
        ];

        const MMQ_X: usize = 128;
        const MMQ_Y: usize = 128;
        const MMQ_TILE_Y_K: usize = 36;
        const MMQ_TILE_X_K: usize = 76;
        let row_tiles = (m + MMQ_Y - 1) / MMQ_Y;
        let batch_tiles = (batch_size + MMQ_X - 1) / MMQ_X;
        let shared_mem =
            ((MMQ_X * MMQ_TILE_Y_K + MMQ_Y * MMQ_TILE_X_K) * std::mem::size_of::<i32>()) as u32;

        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k + batch_size * m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_residual_mmq", bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 8, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b.push_i32(add_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// gfx906 dp4a MMQ residual GEMM. Wave-native topology (block 64×2,
    /// tile 128×64) per llama.cpp-gfx906 reference. Distinct from the
    /// RDNA3 i8-WMMA variant above — different block dim, different
    /// LDS layout, different kernel symbols.
    ///
    /// Phase 1 implementation; opt-in via `HIPFIRE_MMQ=1` while correctness
    /// is being validated. See plans/gfx906_mmq_plan.md and
    /// plans/p1.2_dp4a_mmq_design.md.
    pub fn gemm_hfq4g256_residual_mmq_gfx906(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Quantize activations to Q8_1.
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;

        // Diagnostic: HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY=1 isolates the cost of
        // the Q8_1 activation pre-quantize by running the FP16 wave64 path
        // *after* paying the quantize cost. The flag is read once at init
        // (see `Gpu::new`) so this check is a single bool load, not a
        // per-call env::var lookup.
        if self.flags.mmq_diag_quantize_only {
            let _ = x_q8_ptr;
            return self.gemm_hfq4g256_residual_fp16_wave64(a_raw, x, y, m, k, batch_size);
        }

        // Greedy mmq_x selection matching stock.
        let mmq_x = if batch_size <= 8 {
            8
        } else if batch_size <= 16 {
            16
        } else if batch_size <= 24 {
            24
        } else if batch_size <= 32 {
            32
        } else if batch_size <= 40 {
            40
        } else if batch_size <= 48 {
            48
        } else if batch_size <= 56 {
            56
        } else {
            64
        };

        // Pick variant name and source.
        let is_full = m % 128 == 0 && batch_size % mmq_x == 0;
        let base_name = "gemm_hfq4g256_residual_mmq_gfx906";
        let kernel_name = if is_full {
            format!("{}_full_add_x{}", base_name, mmq_x)
        } else {
            format!("{}_x{}", base_name, mmq_x)
        };

        let wrapper_src = match mmq_x {
            8 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X8_SRC,
            16 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X16_SRC,
            24 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X24_SRC,
            32 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X32_SRC,
            40 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X40_SRC,
            48 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X48_SRC,
            56 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X56_SRC,
            64 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X64_SRC,
            _ => unreachable!(),
        };
        // Inline the body .cuh: the runtime hipcc compiles from cache_dir,
        // which doesn't have kernels/src on its -I path. Strip the
        // `#include "..._body.cuh"` line and prepend the body content.
        let inlined = wrapper_src.replace(
            "#include \"gemm_hfq4g256_residual_mmq_gfx906_body.cuh\"",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_BODY_CUH,
        );

        self.ensure_kernel(&format!("{}_x{}", base_name, mmq_x), &inlined, &kernel_name)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut xq_ptr = x_q8_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut add_val = 1i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut add_val as *mut _ as *mut c_void,
        ];

        // Option C streaming topology — KEEP IN SYNC WITH body.cuh:
        //   x_qs   : MMQ_Y * x_stride ints  (per-mmq_x: 40 if mmq_x≥32 else 33)
        //   x_dm   : MMQ_Y float2
        //   tile_y : mmq_x * Y_STRIDE ints
        const MMQ_Y: usize = 128;
        let x_stride: usize = if mmq_x >= 32 { 40 } else { 33 };
        const Y_STRIDE: usize = 36;
        const X_DM_HALF2: usize = 128;
        let row_tiles = (m + MMQ_Y - 1) / MMQ_Y;
        let batch_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let shared_mem =
            ((MMQ_Y * x_stride * 4) + (X_DM_HALF2 * 8) + (mmq_x * Y_STRIDE * 4)) as u32;
        // 2 WGs/CU on gfx906 needs ≤32 KiB/WG (64 KiB cap).
        debug_assert!(
            shared_mem as usize <= 32 * 1024,
            "gfx906 MMQ LDS budget exceeded: {} B > 32 KiB",
            shared_mem
        );

        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", base_name, bytes);
        let result = self.launch_maybe_blob(
            &kernel_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [64, 4, 1], // nwarps=4
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b.push_i32(add_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Set-mode (add=0) variant of the gfx906 MMQ kernel.
    pub fn gemm_hfq4g256_mmq_set_gfx906(
        &mut self,
        a_raw: &GpuTensor,
        x_q8_ptr: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Caller (fused dispatcher) is expected to gate via `should_use_mmq`;
        // the assert below enforces that contract so a future caller can't
        // silently route a non-winning batch through MMQ. Mirrors the MQ6
        // sibling's `hfq6_mmq_route` assert added in 5ea9050.
        debug_assert!(
            self.arch_caps.should_use_mmq(batch_size) || self.capture_mode,
            "_mmq_set_gfx906 called at non-winning B={} (capture={}) — \
             caller must gate via should_use_mmq first",
            batch_size,
            self.capture_mode,
        );
        let mmq_x = if batch_size <= 8 {
            8
        } else if batch_size <= 16 {
            16
        } else if batch_size <= 24 {
            24
        } else if batch_size <= 32 {
            32
        } else if batch_size <= 40 {
            40
        } else if batch_size <= 48 {
            48
        } else if batch_size <= 56 {
            56
        } else {
            64
        };

        let is_full = m % 128 == 0 && batch_size % mmq_x == 0;
        let base_name = "gemm_hfq4g256_residual_mmq_gfx906";
        let kernel_name = if is_full {
            format!("{}_full_set_x{}", base_name, mmq_x)
        } else {
            format!("{}_x{}", base_name, mmq_x)
        };

        let wrapper_src = match mmq_x {
            8 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X8_SRC,
            16 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X16_SRC,
            24 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X24_SRC,
            32 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X32_SRC,
            40 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X40_SRC,
            48 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X48_SRC,
            56 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X56_SRC,
            64 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X64_SRC,
            _ => unreachable!(),
        };
        let inlined = wrapper_src.replace(
            "#include \"gemm_hfq4g256_residual_mmq_gfx906_body.cuh\"",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_BODY_CUH,
        );

        self.ensure_kernel(&format!("{}_x{}", base_name, mmq_x), &inlined, &kernel_name)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut xq_ptr = x_q8_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut add_val = 0i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut add_val as *mut _ as *mut c_void,
        ];

        // Option C streaming topology — KEEP IN SYNC WITH body.cuh
        // (same layout invariant as residual variant above).
        const MMQ_Y: usize = 128;
        let x_stride: usize = if mmq_x >= 32 { 40 } else { 33 };
        const Y_STRIDE: usize = 36;
        const X_DM_HALF2: usize = 128;
        let row_tiles = (m + MMQ_Y - 1) / MMQ_Y;
        let batch_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let shared_mem =
            ((MMQ_Y * x_stride * 4) + (X_DM_HALF2 * 8) + (mmq_x * Y_STRIDE * 4)) as u32;
        debug_assert!(
            shared_mem as usize <= 32 * 1024,
            "gfx906 MMQ LDS budget exceeded: {} B > 32 KiB",
            shared_mem
        );

        // bytes = weight read + X read (Q8_1, ~1 byte/element + scale) + Y write (set, no read).
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k + batch_size * m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_mmq_set_gfx906", bytes);
        let result = self.launch_maybe_blob(
            &kernel_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [64, 4, 1], // nwarps=4
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b.push_i32(add_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn gemm_hfq4g256_mmq_set(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.hfq4g256_mmq_gfx1151_enabled(m, k, batch_size) {
            return self.gemm_hfq4g256_mmq_gfx1151(a_raw, x, y, m, k, batch_size, false);
        }
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let kernel_name = if m % 128 == 0 && batch_size % 128 == 0 {
            "gemm_hfq4g256_residual_mmq_full_set"
        } else {
            "gemm_hfq4g256_residual_mmq"
        };
        self.ensure_kernel(
            "gemm_hfq4g256_residual_mmq",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_SRC,
            kernel_name,
        )?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut xq_ptr = x_q8_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut add_val = 0i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut add_val as *mut _ as *mut c_void,
        ];

        const MMQ_X: usize = 128;
        const MMQ_Y: usize = 128;
        const MMQ_TILE_Y_K: usize = 36;
        const MMQ_TILE_X_K: usize = 76;
        let row_tiles = (m + MMQ_Y - 1) / MMQ_Y;
        let batch_tiles = (batch_size + MMQ_X - 1) / MMQ_X;
        let shared_mem =
            ((MMQ_X * MMQ_TILE_Y_K + MMQ_Y * MMQ_TILE_X_K) * std::mem::size_of::<i32>()) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k + batch_size * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_mmq_set", bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 8, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b.push_i32(add_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn gemm_hfq4g256_mmq_set_prequant(
        &mut self,
        a_raw: &GpuTensor,
        x_q8_ptr: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch_caps.is_gfx906() {
            // gfx906 has its own dispatcher (`gemm_hfq4g256_residual_mmq_gfx906`)
            // that handles its own quantize internally, called directly from
            // mmq_screen_weight on gfx906. _set_prequant is RDNA3-only.
            return Err(hip_bridge::HipError::new(
                0,
                "gemm_hfq4g256_mmq_set_prequant is not supported on gfx906; \
                 callers should route to gemm_hfq4g256_residual_mmq_gfx906 directly",
            ));
        }
        if self.hfq4g256_mmq_gfx1151_enabled(m, k, batch_size) {
            return self
                .gemm_hfq4g256_mmq_gfx1151_prequant(a_raw, x_q8_ptr, y, m, k, batch_size, false);
        }
        let kernel_name = if m % 128 == 0 && batch_size % 128 == 0 {
            "gemm_hfq4g256_residual_mmq_full_set"
        } else {
            "gemm_hfq4g256_residual_mmq"
        };
        self.ensure_kernel(
            "gemm_hfq4g256_residual_mmq",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_SRC,
            kernel_name,
        )?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut xq_ptr = x_q8_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut add_val = 0i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut add_val as *mut _ as *mut c_void,
        ];

        const MMQ_X: usize = 128;
        const MMQ_Y: usize = 128;
        const MMQ_TILE_Y_K: usize = 36;
        const MMQ_TILE_X_K: usize = 76;
        let row_tiles = (m + MMQ_Y - 1) / MMQ_Y;
        let batch_tiles = (batch_size + MMQ_X - 1) / MMQ_X;
        let shared_mem =
            ((MMQ_X * MMQ_TILE_Y_K + MMQ_Y * MMQ_TILE_X_K) * std::mem::size_of::<i32>()) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_mmq_set", bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 8, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b.push_i32(add_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    fn gemm_hfq4g256_mmq_gfx1151(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        add: bool,
    ) -> HipResult<()> {
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        self.gemm_hfq4g256_mmq_gfx1151_prequant(a_raw, x_q8_ptr, y, m, k, batch_size, add)
    }
    fn gemm_hfq4g256_mmq_gfx1151_prequant(
        &mut self,
        a_raw: &GpuTensor,
        x_q8_ptr: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        add: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq4g256_mmq_gfx1151",
            kernels::GEMM_HFQ4G256_MMQ_GFX1151_SRC,
            "gemm_hfq4g256_mmq_gfx1151",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let n_val = batch_size as i32;
        let add_val = if add { 1i32 } else { 0i32 };
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_q8_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
            &add_val as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * k
            + batch_size * m * 4 * if add { 2 } else { 1 };
        let label = if add {
            "gemm_hfq4g256_mmq_gfx1151_add"
        } else {
            "gemm_hfq4g256_mmq_gfx1151_set"
        };
        let timer = crate::profile::begin_timer(&self.hip, "gemm", label, bytes);
        let result = self.launch_maybe_blob(
            "gemm_hfq4g256_mmq_gfx1151",
            [(m / 16) as u32, (batch_size / 16) as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_q8_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b.push_i32(add_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// WMMA-accelerated batched HFQ4-G256 GEMM with residual add.
    /// gfx1100+ only. 16×16 output tiles via wave32 WMMA.
    /// Converts X to FP16, then uses __builtin_amdgcn_wmma_f32_16x16x16_f16_w32.
    pub fn gemm_hfq4g256_residual_wmma(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Compile both kernels (convert + WMMA GEMM share the FP16 convert)
        // Kernel variant selection
        // MW16 path: dequant weights to FP16 per-call, then run no-dequant WMMA
        if self.flags.mw16 {
            return self.gemm_mw16_residual_wmma_via_dequant(a_raw, x, y, m, k, batch_size);
        }
        // Shape-aware default: ksplit only pays for itself when the un-split
        // grid is CU-starved (target wo_residual at M=5120 → 320 blocks,
        // ~3.3/CU on gfx1100 — ksplit 4×'s it to 13/CU). For draft-FFN shapes
        // (M=17408, K=5120, B=16) the un-split grid is already 1088 blocks
        // (~11/CU) and the atomicAdd reduce is pure overhead. k2 removes the
        // split + atomics and runs deterministically.
        //
        // Threshold picked at M=8192: covers M∈{5120,6144} (target wo) on the
        // ksplit side and M∈{17408} (draft gate/up/down) on the k2 side. lm_head
        // (M=vocab) is always way above threshold → k2.
        //
        // HIPFIRE_WO_WMMA_VARIANT=ksplit|k2|k2x32|k4|wmma|wmma2 overrides the
        // auto selection (applies to every call, both target and draft).
        //   ksplit — K-split + atomicAdd (non-deterministic accum order)
        //   k2     — 2× K-tile pipeline (byte-exact accum order)
        //   k2x32  — 32-row block with shared X fragment per K-tile. Slower
        //            than k2 on gfx1100, but faster on gfx1151 Strix Halo for
        //            small-M residual projections at prefill-sized batches.
        //            DFlash verify/lm_head runs at B<=16 and large-M draft
        //            FFN/lm_head also prefer k2.
        //   k4     — 4× K-tile pipeline. Fixed 2026-05-01 (commit pending):
        //            output mapping was swapped relative to K2's canonical
        //            wave32 WMMA C-mapping. Channel-test passes at K∈{256,512,4096}
        //            × batch∈{1,2,4,16}. At m<8192 (9B residual at m=4096) K4
        //            ties K2 within FP drift but loses to ksplit by ~33%
        //            per-call at small batch (CU-starved grid: 3.3 vs 13
        //            blocks/CU); auto-dispatch correctly stays on ksplit. K4
        //            vs K2 at m≥8192 not yet benched on available models. See
        //            plans/k4_plan.md.
        //   wmma   — base WMMA         (output-mapping bug — debug only)
        //   wmma2  — 2-wave block, 32 rows × 16 batch (output-mapping bug — debug only)
        let is_gfx115x = self.arch_caps.is_rdna3p5();
        // ksplit's atomicAdd reduction across K_SPLITS partials is fp-non-
        // associative — order varies with warp scheduling, so output bytes
        // drift between processes and between cold/hot runs. The drift is
        // sub-argmax-margin per call but cascades on long greedy decode
        // (>50 tokens). HIPFIRE_DETERMINISTIC=1 forces k2 (single-block
        // K reduction) at the cost of ~33% perf on small-batch / small-M.
        // Required when chasing multi-GPU parity: pp=1 vs pp=2 outputs
        // can't be compared byte-for-byte when the underlying single-GPU
        // path itself is non-deterministic.
        // Cached — getenv on every decode token would re-parse 6× per layer
        // × N layers per step. Read once at first dispatch.
        static FORCE_DET: OnceLock<bool> = OnceLock::new();
        let force_det = *FORCE_DET.get_or_init(|| self.flags.deterministic);
        let auto_variant = if force_det {
            "k2"
        } else if is_gfx115x && batch_size <= 16 {
            "k2"
        } else if is_gfx115x && m < 8192 {
            "k2x32"
        } else if m >= 8192 {
            "k2"
        } else {
            "ksplit"
        };
        let variant_override = self.flags.wo_wmma_variant.clone();
        let variant = variant_override.as_deref().unwrap_or(auto_variant);
        let (kernel_name, kernel_src, block_size, row_step, k_splits) = match variant {
            "k2" => (
                "gemm_hfq4g256_residual_wmma_k2",
                kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_K2_SRC,
                32u32,
                16usize,
                1u32,
            ),
            "k2x32" => (
                "gemm_hfq4g256_residual_wmma_k2x32",
                kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_K2X32_SRC,
                32u32,
                32usize,
                1u32,
            ),
            "k4" => (
                "gemm_hfq4g256_residual_wmma_k4",
                kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_K4_SRC,
                32u32,
                16usize,
                1u32,
            ),
            "wmma" => (
                "gemm_hfq4g256_residual_wmma",
                kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_SRC,
                32u32,
                16usize,
                1u32,
            ),
            "wmma2" => (
                "gemm_hfq4g256_residual_wmma2",
                kernels::GEMM_HFQ4G256_RESIDUAL_WMMA2_SRC,
                64u32,
                32usize,
                1u32,
            ),
            _ => (
                "gemm_hfq4g256_residual_wmma_ksplit",
                kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_KSPLIT_SRC,
                32u32,
                16usize,
                4u32,
            ),
        };
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
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

        let row_tiles = (m + row_step - 1) / row_step;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        // HIPFIRE_GEMM_DUMP=1: per-call shape+wall-clock dump of this kernel.
        // Synchronously times only the ksplit kernel launch (not memset / convert).
        // Measures actual GPU execution time via device_synchronize pre+post —
        // costs latency vs async pipelining but gives shape-accurate µs.
        static DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let dump = *DUMP.get_or_init(|| self.flags.gemm_dump);
        if dump {
            self.hip.device_synchronize()?;
        }
        let dump_start = if dump {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles as u32, batch_tiles as u32, k_splits],
            [block_size, 1, 1],
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
        if let Some(t) = dump_start {
            self.hip.device_synchronize()?;
            let us = t.elapsed().as_micros();
            let gbs = (bytes as f64) / (us.max(1) as f64) / 1000.0; // MB/ms == GB/s
            eprintln!(
                "[gemm-dump] {} M={} K={} B={} bytes={}KB us={} GB/s={:.1}",
                kernel_name,
                m,
                k,
                batch_size,
                bytes / 1024,
                us,
                gbs
            );
        }
        result
    }
    /// HFQ3-G256 sister of `gemm_hfq4g256_residual_wmma` (basic WMMA
    /// variant). Same WMMA shape + lane decomposition; only the inner
    /// K-tile unpack differs (3-bit cross-byte vs 4-bit nibble) and the
    /// per-group byte stride is 104 instead of 136. Y += acc[j] (fused
    /// residual add — caller must initialize Y with the residual stream
    /// before launching).
    pub fn gemm_hfq3g256_residual_wmma(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let arch_supports_mb4 = self.arch_caps.is_rdna3()
            && !self.arch_caps.is_gfx1152()
            && !self.arch_caps.is_gfx1103();
        let use_mb4 = match self.flags.mq3_mb4 {
            Some(_) => arch_supports_mb4,
            None => arch_supports_mb4 && batch_size >= 128 && m >= 4096,
        };
        if use_mb4 {
            return self.gemm_hfq3g256_residual_wmma_mb4(a_raw, x, y, m, k, batch_size);
        }
        if self.arch_caps.has_wmma_w32_gfx12() {
            return self.gemm_hfq3g256_residual_wmma_gfx12(a_raw, x, y, m, k, batch_size);
        }
        self.ensure_kernel(
            "gemm_hfq3g256_residual_wmma",
            kernels::GEMM_HFQ3G256_RESIDUAL_WMMA_SRC,
            "gemm_hfq3g256_residual_wmma",
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
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq3g256_residual_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_hfq3g256_residual_wmma",
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
    /// HFQ3 residual mb4 dispatch: 16×64 output tile per WG.
    pub fn gemm_hfq3g256_residual_wmma_mb4(
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
            "gemm_hfq3g256_residual_wmma_mb4",
            kernels::GEMM_HFQ3G256_RESIDUAL_WMMA_MB4_SRC,
            "gemm_hfq3g256_residual_wmma_mb4",
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
        let batch_tiles = (batch_size + 63) / 64;
        let bytes = m * (k / 256) * 104 + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq3g256_residual_wmma_mb4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_hfq3g256_residual_wmma_mb4",
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
    /// Batched HFQ4-G256 GEMM: y[b][row] = A[row] · x[b] for all batch elements.
    /// x: [batch_size × K], y: [batch_size × M], both row-major.
    ///
    /// This is the portable scalar kernel — stays byte-exact with the AR
    /// greedy prefill's numerical baseline. For the DFlash lm_head fast
    /// path (batched, tolerates small FP16 drift for 8-10× speedup), use
    /// `gemm_hfq4g256_batched_lmhead` instead.
    pub fn gemm_hfq4g256(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // gfx906 dp4a opt-in for the LM-head batched GEMM. PMC at 2026-05-06
        // showed gemm_hfq4g256_wave64 was 17 % of DFlash 27B steady-state
        // decode time on the FP wave64 path. The dp4a port pre-quantizes x
        // to Q8_1 (shared scratch with the prefill MMQ + the gate_up/qkv/qkvza
        // GEMV ports) and runs v_dot4_i32_i8.
        //
        // Only fires on gfx906 (other wave64-native archs have rocBLAS or
        // larger MFMA paths that beat dp4a at large batches). Skip in
        // capture mode (matches the rocBLAS branch's caveat — Q8_1
        // quantize launch must be reachable from the captured graph or
        // pre-baked).
        if self.arch_caps.gemv_dp4a_enabled() && !self.capture_mode {
            return self.gemm_hfq4g256_dp4a(a_raw, x, y, m, k, batch_size);
        }

        // CDNA3 MFMA path (task #130): when rocBLAS is loaded and batch is
        // big enough for the launch overhead to amortize, route through the
        // dequantize-once FP16 shadow + rocBLAS GEMM. Expected 20-100× over
        // the wave64 GEMV on prefill-heavy workloads (sidecar cal, DFlash
        // target verify). Falls back to wave64 GEMV on: single-token decode
        // (batch<4), capture mode (rocBLAS launches don't graph-capture
        // cleanly; revisit if hipGraph becomes critical for CDNA3 prefill),
        // or if the fp16 shadow alloc fails under VRAM pressure.
        if self.rocblas_arch_eligible()
            && batch_size >= self.rocblas_min_batch()
            && self.rocblas.is_some()
            && !self.capture_mode
        {
            if let Ok(Some(shadow_ptr)) = self.ensure_fp16_shadow(a_raw, m, k) {
                // Convert X to FP16 via the existing ensure_fp16_x helper.
                let x_fp16 = self.ensure_fp16_x(x, batch_size * k)?;
                // Wrap the raw device pointers as non-owning DeviceBuffers so
                // the rocBLAS helper's signature works. The underlying memory
                // is owned by the fp16 shadow cache / fp16_x_scratch / caller's
                // y GpuTensor — all live beyond this call.
                let w_buf = unsafe { DeviceBuffer::from_raw(shadow_ptr, (m * k) * 2) };
                let x_buf = unsafe { DeviceBuffer::from_raw(x_fp16, (batch_size * k) * 2) };
                let bytes = crate::profile::gemm_hfq4g256_bytes(m, k, batch_size);
                let timer =
                    crate::profile::begin_timer(&self.hip, "gemv", "gemm_hfq4g256_rocblas", bytes);
                let result =
                    self.rocblas_gemm_hfq4_prefill(&w_buf, &x_buf, &y.buf, m, batch_size, k);
                // Suppress the non-owning DeviceBuffer drop; HipError::Drop on
                // hip_free would clobber memory we don't own.
                std::mem::forget(w_buf);
                std::mem::forget(x_buf);
                if let Some(t) = timer {
                    t.finish(&self.hip);
                }
                return result;
            }
            // Shadow allocation failed — fall through to the GEMV path.
        }

        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let (func_name, block, grid_div): (&str, [u32; 3], u32) = if cdna_wave64 {
            self.ensure_kernel(
                "gemm_hfq4g256_wave64",
                kernels::GEMM_HFQ4G256_WAVE64_SRC,
                "gemm_hfq4g256_wave64",
            )?;
            ("gemm_hfq4g256_wave64", [64, 1, 1], 2)
        } else {
            self.ensure_kernel("gemm_hfq4g256", kernels::GEMM_HFQ4G256_SRC, "gemm_hfq4g256")?;
            ("gemm_hfq4g256", [32, 1, 1], 1)
        };

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
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

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            ((batch_size + BATCH_TILE - 1) / BATCH_TILE) as u32
        };
        let grid_x = (m as u32 + grid_div - 1) / grid_div;
        let bytes = crate::profile::gemm_hfq4g256_bytes(m, k, batch_size);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemm_hfq4g256", bytes);
        let result = self.launch_maybe_blob(
            func_name,
            [grid_x, batch_tiles, 1],
            block,
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
    /// dp4a-port of gemm_hfq4g256 for gfx906. Pre-quantizes x to Q8_1 via
    /// the shared MMQ x-scratch (kblock-major: `[K/128, batch_size]`),
    /// then dispatches the wave64 dp4a GEMM. Math is identical modulo
    /// Q8_1 quant noise.
    ///
    /// Targets the LM-head batched GEMM hot path on DFlash 27B (PMC at
    /// 2026-05-06 showed 17 % of decode time was here on the FP path).
    /// Same Q8_1 layout as the prefill MMQ kernel + the four PR-158
    /// fused GEMVs, so `ensure_q8_1_mmq_x` reuses the existing scratch.
    pub fn gemm_hfq4g256_dp4a(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Quantize x → Xq[K/128 * batch_size] block_q8_1_mmq via the
        // shared scratch. Stride layout: kblock-major (matches
        // quantize_q8_1_mmq_ds4 at gemm_hfq4g256_residual_mmq.hip:80).
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;

        self.ensure_kernel(
            "gemm_hfq4g256_wave64_dp4a",
            kernels::GEMM_HFQ4G256_WAVE64_DP4A_SRC,
            "gemm_hfq4g256_wave64_dp4a",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;
        let grid_x = (m as u32 + 1) / 2;
        const BATCH_TILE: usize = 8;
        let grid_y = ((batch_size + BATCH_TILE - 1) / BATCH_TILE) as u32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        let bytes = crate::profile::gemm_hfq4g256_bytes(m, k, batch_size);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemm_hfq4g256_dp4a", bytes);
        let result = self.launch_maybe_blob(
            "gemm_hfq4g256_wave64_dp4a",
            [grid_x, grid_y, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq);
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
    /// Batched HFQ4-G256 residual GEMM with fused dp4a inner loop on gfx906.
    /// HFQ4 sibling of `gemm_hfq6g256_residual_wave64_dp4a` (HFQ6 Phase A.2,
    /// commit 1b9f3747 → merged via #187). Closes the dispatch gap where MQ4
    /// at gfx906 B>1 below the MMQ cutover (B ∈ [2, 7] per `should_use_mmq`'s
    /// gfx906 default) falls to `gemm_hfq4g256_residual_fp16_wave64`; the
    /// dp4a path wins on per-call ALU (sdot4 issues 4 int8 mul + 4 acc-add
    /// per cycle, vs FP wave64 hybrid's hfma2 at 2 mul + 2 add per cycle →
    /// ~2× FLOPs/cycle) and reuses the existing Q8_1 activation scratch
    /// (shared with HFQ4 MMQ + the GEMV-shape fused dp4a kernels).
    ///
    /// Issue #276 Gap 2. Ships with `BATCH_TILE = 16` from the start per the
    /// HFQ6 Phase B.1.1 measurement (commit ff9e2105: BT=8→16 halves A-reload
    /// trips per row, +7-17% per-call on the structurally identical HFQ6
    /// sibling). MUST stay in sync with the kernel's `#define BATCH_TILE 16`
    /// at `kernels/src/gemm_hfq4g256_residual_wave64_dp4a.hip:38`.
    pub fn gemm_hfq4g256_residual_wave64_dp4a(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        self.gemm_hfq4g256_residual_wave64_dp4a_prequant(a_raw, xq_ptr, y, m, k, batch_size)
    }
    /// Prequant entry point: caller has already populated the Q8_1 scratch
    /// (see `ensure_q8_1_mmq_x`). Skips the Q8_1 conversion. Use when X has
    /// just been quantized for a sibling kernel (e.g. MMQ split or fused
    /// QKVZA tail) to avoid a redundant ~k·batch_size byte memset+convert.
    pub fn gemm_hfq4g256_residual_wave64_dp4a_prequant(
        &mut self,
        a_raw: &GpuTensor,
        xq_ptr: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq4g256_residual_wave64_dp4a",
            kernels::GEMM_HFQ4G256_RESIDUAL_WAVE64_DP4A_SRC,
            "gemm_hfq4g256_residual_wave64_dp4a",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        // BATCH_TILE MUST match the kernel's `#define BATCH_TILE 16`.
        const BATCH_TILE: usize = 16;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let grid_x = ((m as u32) + 1) / 2;

        // bytes = weight (HFQ4: 136 B/group, 0.53 B/weight) + Q8_1 X scratch
        // (~33 B per Q8_1 block of 32 K-elems = ~1.03 B/element, but for
        // bandwidth accounting use the dominant int8 qs term: batch*k bytes)
        // + Y read+write (residual: batch*m*4 each way).
        let bytes =
            crate::profile::hfq4g256_weight_bytes(m, k) + batch_size * k + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq4g256_residual_wave64_dp4a",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_hfq4g256_residual_wave64_dp4a",
            [grid_x, batch_tiles as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq);
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
    /// WMMA lm_head fast path for DFlash. Computes y = A @ x at batch>1 via
    /// the residual-WMMA kernel on pre-zeroed y — 8-10× faster than the
    /// scalar `gemm_hfq4g256` on 9B lm_head (batch=16, vocab=248K, k=2560).
    /// HIPFIRE_LM_HEAD_OVERWRITE=1 opts into a gfx12 overwrite sibling that
    /// skips the zero-fill; measured as a small win but not enough to default.
    ///
    /// NOT numerically identical to `gemm_hfq4g256`. Uses FP16 tensor cores
    /// with the accumulators in FP32 the residual kernel ships. On the
    /// DFlash target-verify + draft-lm_head hot path this is a win (~13 ms
    /// saved per cycle), and the small FP16 drift doesn't meaningfully
    /// affect greedy acceptance. Do NOT use for AR greedy prefill — it will
    /// break byte-exact quality-gate reproducibility.
    ///
    /// Fallbacks: non-gfx11 or HIPFIRE_FP16=0 or HIPFIRE_LM_HEAD_WMMA=0 →
    /// routes to plain `gemm_hfq4g256`.
    ///
    /// Subtle: the residual-WMMA kernel goes through `ensure_fp16_x`, which
    /// caches the FP32→FP16 conversion keyed on source pointer. DFlash
    /// callers reuse the SAME hidden buffer pointer every cycle (draft
    /// scratch sub-offset, verify's persistent final_hidden) but with NEW
    /// data — so the cache entry is silently stale. Stomp the cache pointer
    /// before the dispatch to force reconversion.
    pub fn gemm_hfq4g256_batched_lmhead(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Gate covers gfx11 (RDNA3) AND gfx12 (RDNA4). The gfx12 family
        // ships its own residual_wmma kernel sibling
        // (gemm_hfq4g256_residual_wmma_gfx12); without this dispatch,
        // gfx12 falls through to the scalar `gemm_hfq4g256` and pays the
        // 8-10× per-call penalty (rocprof on R9700 / gfx1201 measured
        // ~26.68% of composition cycle wall in this scalar path).
        let arch = self.arch.as_str();
        let wmma_eligible = batch_size > 1
            && (self.arch_caps.has_wmma_w32() || self.arch_caps.has_wmma_w32_gfx12())
            && !self.flags.fp16_disabled
            && !self.flags.lm_head_wmma_disabled;
        if wmma_eligible {
            self.fp16_x_source_ptr = std::ptr::null_mut();
            if arch.starts_with("gfx12") && self.flags.lm_head_overwrite {
                self.gemm_hfq4g256_lmhead_wmma_gfx12(a_raw, x, y, m, k, batch_size)
            } else {
                match self.active_stream.as_ref() {
                    Some(stream) => self
                        .hip
                        .memset_async(&y.buf, 0, batch_size * m * 4, stream)?,
                    None => self.hip.memset(&y.buf, 0, batch_size * m * 4)?,
                }
                if arch.starts_with("gfx12") {
                    self.gemm_hfq4g256_residual_wmma_gfx12(a_raw, x, y, m, k, batch_size)
                } else {
                    self.gemm_hfq4g256_residual_wmma(a_raw, x, y, m, k, batch_size)
                }
            }
        } else {
            self.gemm_hfq4g256(a_raw, x, y, m, k, batch_size)
        }
    }
    /// HFQ6-G256 sister of `gemm_hfq4g256_batched_lmhead`. Phase A.4
    /// (plan v3.2.3 §5.1 item 4). On gfx906 uses the dp4a residual GEMM
    /// (Phase A.2) with a zero-init of Y, mirroring the HFQ4 WMMA pattern
    /// at line 8019-8022. Lets the residual `+=` collapse to `=` semantics
    /// without needing a separate non-residual kernel.
    ///
    /// Caller is responsible for FWHT-rotating x first when the weights
    /// are MQ6 (FWHT-rotated at quant time).
    pub fn gemm_hfq6g256_batched_lmhead(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // gfx906: dp4a residual + zero-init Y for `=` semantics.
        // Skip in capture mode (the residual kernel calls ensure_q8_1_mmq_x
        // which launches an internal quantize kernel — matches HFQ4 sibling).
        if batch_size > 1 && self.arch_caps.gemv_dp4a_enabled() && !self.capture_mode {
            match self.active_stream.as_ref() {
                Some(stream) => self
                    .hip
                    .memset_async(&y.buf, 0, batch_size * m * 4, stream)?,
                None => self.hip.memset(&y.buf, 0, batch_size * m * 4)?,
            }
            return self.gemm_hfq6g256_residual_wave64_dp4a(a_raw, x, y, m, k, batch_size);
        }
        // gfx11+ AND gfx12: WMMA residual + zero-init. Symmetric to the
        // HFQ4 fix (commit 48dd8ba4) — gfx12 sibling kernel already ships
        // (gemm_hfq6g256_residual_wmma_gfx12, see line ~15431 dispatch);
        // this wrapper just needed the gate widened.
        let arch_str = self.arch.as_str();
        let wmma_eligible = batch_size > 1
            && self.arch_caps.has_wmma_w32()
            && !self.flags.fp16_disabled
            && !self.flags.lm_head_wmma_disabled;
        if wmma_eligible {
            self.fp16_x_source_ptr = std::ptr::null_mut();
            match self.active_stream.as_ref() {
                Some(stream) => self
                    .hip
                    .memset_async(&y.buf, 0, batch_size * m * 4, stream)?,
                None => self.hip.memset(&y.buf, 0, batch_size * m * 4)?,
            }
            return if arch_str.starts_with("gfx12") {
                self.gemm_hfq6g256_residual_wmma_gfx12(a_raw, x, y, m, k, batch_size)
            } else {
                self.gemm_hfq6g256_residual_wmma(a_raw, x, y, m, k, batch_size)
            };
        }
        // Fallback: use the residual dispatcher with zero-init Y. This
        // routes to fp16-packed or scalar depending on arch.
        match self.active_stream.as_ref() {
            Some(stream) => self
                .hip
                .memset_async(&y.buf, 0, batch_size * m * 4, stream)?,
            None => self.hip.memset(&y.buf, 0, batch_size * m * 4)?,
        }
        self.gemm_hfq6g256_residual(a_raw, x, y, m, k, batch_size)
    }
    /// HFQ3-G256 sister of `gemm_hfq4g256_batched_lmhead`. Same FP16-X cache
    /// stomp + zero-init of Y, then `gemm_hfq3g256_residual_wmma` to compute
    /// y[b][row] = A[row] · x[b]. Used by `dflash::gemm_dispatch` for MQ3
    /// drafts so DFlash works with MQ3-quantized draft weights.
    ///
    /// Caller is responsible for FWHT-rotating x first when the weights are
    /// MQ3 (FWHT-rotated at quant time) — `dflash::gemm_dispatch` handles
    /// that via `rotate_x_mq_batched`. This wrapper is dtype-agnostic in
    /// the same sense as `gemm_hfq4g256_batched_lmhead`.
    pub fn gemm_hfq3g256_batched_lmhead(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // WMMA eligibility: any arch with an MQ3 WMMA family ported. Today
        // that's gfx11 (RDNA3, _w32 builtin) and gfx12 (RDNA4, _w32_gfx12
        // builtin) — `gemm_hfq3g256_residual_wmma` dispatches internally to
        // the correct variant per arch. Other archs (gfx10/906/94x) fall
        // through to the per-row GEMV path.
        let wmma_eligible = batch_size > 1
            && (self.arch_caps.has_wmma_w32() || self.arch_caps.has_wmma_w32_gfx12())
            && !self.flags.fp16_disabled
            && !self.flags.lm_head_wmma_disabled;
        if wmma_eligible {
            self.fp16_x_source_ptr = std::ptr::null_mut();
            match self.active_stream.as_ref() {
                Some(stream) => self
                    .hip
                    .memset_async(&y.buf, 0, batch_size * m * 4, stream)?,
                None => self.hip.memset(&y.buf, 0, batch_size * m * 4)?,
            }
            return self.gemm_hfq3g256_residual_wmma(a_raw, x, y, m, k, batch_size);
        }
        // Non-WMMA fallback: per-batch GEMV. Slow but functional. DFlash on
        // non-gfx11/gfx12 archs is already gated upstream by the daemon's
        // DFlash refusal guard (lm_head whitelist requires gfx11 or gfx12
        // for MQ3) — this fallback is reachable only via direct callers
        // that bypass the daemon (e.g., bench harnesses, channel tests).
        for b in 0..batch_size {
            let x_row = x.sub_offset(b * k, k);
            let y_row = y.sub_offset(b * m, m);
            self.gemv_hfq3g256(a_raw, &x_row, &y_row, m, k)?;
        }
        Ok(())
    }
    /// gfx906 wave64+dp4a batched residual GEMM for HFQ6/MQ6.
    /// Phase A.2 (plan v3.2.3 §5.1 item 2). Pre-quantizes x to Q8_1 and
    /// dispatches the dp4a kernel; output is residual `+=` semantics.
    ///
    /// Math identity: same as the fused-GEMV dp4a kernels (plan §2.2
    /// Option A — HFQ6 unsigned weights, no zp shift correction).
    pub fn gemm_hfq6g256_residual_wave64_dp4a(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;

        self.ensure_kernel(
            "gemm_hfq6g256_residual_wave64_dp4a",
            kernels::GEMM_HFQ6G256_RESIDUAL_WAVE64_DP4A_SRC,
            "gemm_hfq6g256_residual_wave64_dp4a",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 8;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let grid_x = ((m as u32) + 1) / 2;

        self.launch_maybe_blob(
            "gemm_hfq6g256_residual_wave64_dp4a",
            [grid_x, batch_tiles as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(bs_val);
                b
            },
        )
    }
    /// Batched HFQ6-G256 GEMM with fused residual add:
    ///   for b in 0..batch_size: y[b][row] += A[row] · x[b]
    ///
    /// Auto-selects: gfx11 -> WMMA, gfx906 -> dp4a (Phase A.2),
    /// else -> FP16 packed, fallback -> FP32 scalar.
    pub fn gemm_hfq6g256_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if batch_size > 1 && !self.flags.fp16_disabled {
            // WMMA on gfx12 (RDNA4): _w32_gfx12 builtin (gfx11 builtin
            // does NOT pattern-match on gfx12 — see has_wmma_f16 comment).
            if self.arch_caps.has_wmma_w32_gfx12() {
                return self.gemm_hfq6g256_residual_wmma_gfx12(a_raw, x, y, m, k, batch_size);
            }
            static HFQ6_RESIDUAL_4W: OnceLock<bool> = OnceLock::new();
            let hfq6_residual_4w = *HFQ6_RESIDUAL_4W.get_or_init(|| {
                matches!(
                    std::env::var("HIPFIRE_HFQ6_RESIDUAL_4W").ok().as_deref(),
                    Some("1" | "on" | "true" | "yes")
                )
            });
            if hfq6_residual_4w
                && self.arch == "gfx1151"
                && batch_size % 64 == 0
                && batch_size >= 128
            {
                return self.gemm_hfq6g256_residual_wmma_4w_gfx1151(a_raw, x, y, m, k, batch_size);
            }
            // WMMA on gfx11+ (RDNA3): 16x16 tiled
            if self.arch_caps.has_wmma_w32() {
                return self.gemm_hfq6g256_residual_wmma(a_raw, x, y, m, k, batch_size);
            }
            // gfx906: dp4a + wave64 batched residual (Phase A.2, plan v3.2.3
            // §5.1 item 2). Pre-quantize x to Q8_1 and dispatch the dp4a
            // kernel. Mirror of the HFQ4 sibling pattern at gemm_hfq4g256_wave64_dp4a.
            // Skip in capture mode: ensure_q8_1_mmq_x launches an internal
            // quantize kernel that the captured graph may not record (matches
            // gemm_hfq4g256_dp4a's `&& !self.capture_mode` guard at line ~7889).
            if self.arch_caps.gemv_dp4a_enabled() && !self.capture_mode {
                return self.gemm_hfq6g256_residual_wave64_dp4a(a_raw, x, y, m, k, batch_size);
            }
            // FP16 packed on all other RDNA
            return self.gemm_hfq6g256_residual_fp16(a_raw, x, y, m, k, batch_size);
        }
        self.ensure_kernel(
            "gemm_hfq6g256_residual",
            kernels::GEMM_HFQ6G256_RESIDUAL_SRC,
            "gemm_hfq6g256_residual",
        )?;
        let func = &self.functions["gemm_hfq6g256_residual"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
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

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };

        // Bandwidth: weight (HFQ6: 200 bytes/group vs HFQ4: 136), per-batch x read, per-batch y RMW.
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)  // placeholder until hfq6 profiling added
            + batch_size * k * 4
            + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq6g256_residual", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_tiles as u32, 1],
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
    /// FP16-input batched HFQ6-G256 GEMM with residual add.
    /// Converts X from FP32 to FP16 (halving X bandwidth), then runs the
    /// FP16-packed GEMM kernel.
    pub fn gemm_hfq6g256_residual_fp16(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor, // FP32 [batch_size x K]
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq6g256_residual_fp16",
            kernels::GEMM_HFQ6G256_RESIDUAL_FP16_SRC,
            "gemm_hfq6g256_residual_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        // FP16 GEMM
        let func = &self.functions["gemm_hfq6g256_residual_fp16"];
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

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * k * 2  // FP16 X (half bandwidth)
            + batch_size * m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq6g256_residual_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_tiles as u32, 1],
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
    /// WMMA-accelerated batched HFQ6-G256 GEMM with residual add.
    /// gfx1100+ only. 16x16 output tiles via wave32 WMMA.
    pub fn gemm_hfq6g256_residual_wmma(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (kernel_name, kernel_src, block_size, row_step) = (
            "gemm_hfq6g256_residual_wmma_k2",
            kernels::GEMM_HFQ6G256_RESIDUAL_WMMA_K2_SRC,
            32u32,
            16usize,
        );
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        // WMMA GEMM
        let func = &self.functions[kernel_name];
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

        let row_tiles = (m + row_step - 1) / row_step;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, batch_tiles as u32, 1],
                [block_size, 1, 1],
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
    /// gfx1151 4-warp 64x64 WMMA HFQ6 residual GEMM.
    pub fn gemm_hfq6g256_residual_wmma_4w_gfx1151(
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
            self.arch, "gfx1151",
            "gemm_hfq6g256_residual_wmma_4w_gfx1151 is gfx1151-only"
        );
        debug_assert_eq!(
            k % 256,
            0,
            "gemm_hfq6g256_residual_wmma_4w_gfx1151: K must be a multiple of 256"
        );
        debug_assert_eq!(
            batch_size % 64,
            0,
            "gemm_hfq6g256_residual_wmma_4w_gfx1151: N must be a multiple of 64"
        );
        self.ensure_kernel(
            "gemm_hfq6g256_residual_wmma_4w_gfx1151",
            kernels::GEMM_HFQ6G256_RESIDUAL_WMMA_4W_GFX1151_SRC,
            "gemm_hfq6g256_residual_wmma_4w_gfx1151",
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

        let row_tiles = (m + 63) / 64;
        let batch_tiles = (batch_size + 63) / 64;

        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq6g256_residual_wmma_4w_gfx1151",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_hfq6g256_residual_wmma_4w_gfx1151",
            [row_tiles as u32, batch_tiles as u32, 1],
            [128, 1, 1],
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
    /// WMMA HFQ4G256 weight × F16 input → F32 output GEMM with
    /// (B, M) output layout. Drop-in for `gemm_hfq4g256` once
    /// activations have been staged through `convert_f32_to_f16`.
    pub fn gemm_hfq4g256_wmma(
        &mut self,
        a_raw: &GpuTensor,
        x_f16: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq4g256_wmma",
            kernels::GEMM_HFQ4G256_WMMA_SRC,
            "gemm_hfq4g256_wmma",
        )?;
        let func = &self.functions["gemm_hfq4g256_wmma"];
        let ap = a_raw.buf.as_ptr();
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
}
