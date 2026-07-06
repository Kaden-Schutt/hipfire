// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx11xx (generic RDNA3 / gfx1100) kernel-dispatch overlays. Phase 2.

use super::super::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
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

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let bytes = (m_total * k) + (m_total * m) * 4 + (crate::profile::gemv_hfq4g256_bytes(m, k));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_kernargs(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr ep, ptr tp, ptr sp, ptr xp, ptr yp, i32 m_val, i32 k_val, i32 xrd_val, i32 mt_val, i32 xsr_val],
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

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW estimate: Q8_1 X reads (~1 B/elem incl. (d,sum) metadata) +
        // HFQ4 weights + Y writes. Distinct from the FP16 sister (which
        // uses 2 B/elem for X).
        let bytes = (m_total * k) + (m_total * m) * 4 + (crate::profile::gemv_hfq4g256_bytes(m, k));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_kernargs(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr ep, ptr tp, ptr sp, ptr xp, ptr yp, i32 m_val, i32 k_val, i32 xrd_val, i32 mt_val, i32 xsr_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// gfx11 (RDNA3) v_dot2_f32_f16 decode-path GEMV for HFP4G32.
    /// Takes F32 x and converts to FP16 INLINE in the inner loop;
    /// `__builtin_amdgcn_fdot2` (v_dot2_f32_f16) does 2 FP16 muls +
    /// 1 FP32 add per VALU. Reduces inner-loop multiply count ~4×
    /// vs the fallback F32 mul+fma chain on ALU-bound shapes.
    /// Routed automatically from `gemv_hfp4g32` when on gfx11+ archs
    /// (gfx1100/1101/1102/1150/1151). NO ensure_fp16_x pre-pass —
    /// that's the v1 trap (eats the dot2 savings in production).
    pub fn gemv_hfp4g32_dot2_gfx11(
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
            "gemv_hfp4g32_dot2 requires K%256==0, got K={}",
            k
        );
        self.ensure_kernel(
            "gemv_hfp4g32_dot2_gfx11",
            kernels::GEMV_HFP4G32_DOT2_GFX11_SRC,
            "gemv_hfp4g32_dot2_gfx11",
        )?;
        let func = &self.functions["gemv_hfp4g32_dot2_gfx11"];
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
                32,
                self.stream_ref(),
                &mut params,
            )
        }
    }
}
