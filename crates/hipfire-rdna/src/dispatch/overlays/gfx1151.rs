// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1151 (RDNA3.5 / Strix Halo APU) kernel-dispatch overlays. Phase 2.

use super::super::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    #[cfg(feature = "deltanet")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gated_delta_net_q8_reg_gfx1151(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        gate: &GpuTensor,
        beta: &GpuTensor,
        s_q8: &GpuTensor,
        s_scales: &GpuTensor,
        output: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        Self::ensure_gdn_hd128(head_dim)?;
        self.ensure_kernel(
            "gated_delta_net_q8_reg_gfx1151",
            kernels::GATED_DELTA_NET_Q8_REG_GFX1151_SRC,
            "gated_delta_net_q8_reg_gfx1151",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let vp = v.buf.as_ptr();
        let gp = gate.buf.as_ptr();
        let bp = beta.buf.as_ptr();
        let sp = s_q8.buf.as_ptr();
        let scp = s_scales.buf.as_ptr();
        let op = output.buf.as_ptr();
        let nt = n_tokens as i32;
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let fr = super::super::GDN_REQUANT_FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            as i32;
        let bytes = crate::profile::gated_delta_net_q8_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "gated_delta_net_q8_reg_gfx1151",
            bytes,
        );
        let result = self.launch_kernargs(
            "gated_delta_net_q8_reg_gfx1151",
            [n_heads as u32, 1, 1],
            [128, 1, 1],
            0,
            &kernargs![ptr qp, ptr kp, ptr vp, ptr gp, ptr bp, ptr sp, ptr scp, ptr op, i32 nt, i32 nh, i32 hd, i32 fr],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
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
        let row_tiles = ((m + 127) / 128) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let bytes = (m * k * 2) + (x_src_rows * k * 2) + (m_total * m * 4);
        let profile_name = if x_row_div > 1 {
            "gemm_f16_moe_grouped_wmma_gfx1151_gate_up"
        } else {
            "gemm_f16_moe_grouped_wmma_gfx1151_down"
        };
        let timer = crate::profile::begin_timer(&self.hip, "gemm", profile_name, bytes);
        let result = self.launch_kernargs(
            "gemm_f16_moe_grouped_wmma_gfx1151",
            [row_tiles, slot_tiles, 1],
            [256, 1, 1],
            0,
            &kernargs![ptr ep, ptr tp, ptr sp, ptr xp, ptr yp, i32 m_val, i32 k_val, i32 xrd_val, i32 mt_val],
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
        let result = self.launch_kernargs(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [block_threads, 1, 1],
            0,
            &kernargs![ptr ep, ptr tp, ptr sp, ptr xp, ptr yp, i32 m_val, i32 k_val, i32 xrd_val, i32 mt_val],
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
        let mi = m as i32;
        let ki = k as i32;
        let bi = batch_size as i32;
        let grid_m = ((m + 127) / 128) as u32;
        let grid_b = ((batch_size + 15) / 16) as u32;
        let bytes = m * k * 2 + batch_size * k * 2 + batch_size * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", profile_label, bytes);
        let result = self.launch_kernargs(
            "gemm_bf16_x_bf16_wmma_gfx1151_m128",
            [grid_m, grid_b, 1],
            [256, 1, 1],
            0,
            &kernargs![ptr ap, ptr xp, ptr yp, i32 mi, i32 ki, i32 bi],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// gfx1151 4-warp 64x64 Q8_0 gate+up GEMM for large prefill shapes.
    pub fn gemm_gate_up_q8_0_wmma_4w_gfx1151(
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
            self.arch, "gfx1151",
            "gemm_gate_up_q8_0_wmma_4w_gfx1151 is gfx1151-only"
        );
        debug_assert_eq!(
            k % 32,
            0,
            "gemm_gate_up_q8_0_wmma_4w_gfx1151: K must be a multiple of 32"
        );
        debug_assert_eq!(
            batch_size % 64,
            0,
            "gemm_gate_up_q8_0_wmma_4w_gfx1151: N must be a multiple of 64"
        );
        self.ensure_kernel(
            "gemm_gate_up_q8_0_wmma_4w_gfx1151",
            kernels::GEMM_GATE_UP_Q8_0_WMMA_4W_GFX1151_SRC,
            "gemm_gate_up_q8_0_wmma_4w_gfx1151",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let a_g = a_gate.buf.as_ptr();
        let a_u = a_up.buf.as_ptr();
        let xp = x_f16_ptr;
        let y_g = y_gate.buf.as_ptr();
        let y_u = y_up.buf.as_ptr();
        let gate_m_val = gate_m as i32;
        let up_m_val = up_m as i32;
        let k_val = k as i32;
        let n_val = batch_size as i32;

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 63) / 64;
        let batch_tiles = (batch_size + 63) / 64;
        let q8_bytes = |m: usize| m * (k / 32) * 34;
        let bytes =
            q8_bytes(gate_m) + q8_bytes(up_m) + batch_size * k * 2 + batch_size * total_m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_gate_up_q8_0_wmma_4w_gfx1151",
            bytes,
        );
        let result = self.launch_kernargs(
            "gemm_gate_up_q8_0_wmma_4w_gfx1151",
            [row_tiles as u32, batch_tiles as u32, 1],
            [128, 1, 1],
            0,
            &kernargs![ptr a_g, ptr a_u, ptr xp, ptr y_g, ptr y_u, i32 gate_m_val, i32 up_m_val, i32 k_val, i32 n_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
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
        self.launch_kernargs(
            "gemm_hfq4g256_s4_mmq_gfx1151",
            [m.div_ceil(16) as u32, n.div_ceil(16) as u32, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr a_ptr, ptr x_qs_ptr, ptr x_d_ptr, ptr x_sum_ptr, ptr y_ptr, i32 m_val, i32 k_val, i32 n_val],
        )
    }
    /// gfx1151 i8 MMQ dispatch helper for HFQ4-G128. Pre-quantizes X to
    /// Q8_1 mmq DS4 then launches `gemm_hfq4g128_mmq_gfx1151`. Caller must
    /// have already verified the alignment constraints.
    pub(crate) fn gemm_hfq4g128_mmq_gfx1151(
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
        let bytes = (m * k / 2)               // HFQ4 weight (4 bits / elem)
                  + (batch_size * k * 4 / 3)  // Q8_1 activation (approx, includes ds4 headers)
                  + (batch_size * m * 4); // F32 output
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g128_mmq_gfx1151", bytes);
        let result = self.launch_kernargs(
            "gemm_hfq4g128_mmq_gfx1151",
            [(m / 16) as u32, (batch_size / 16) as u32, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr a_ptr, ptr x_q8_ptr, ptr y_ptr, i32 m_val, i32 k_val, i32 n_val],
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

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW estimate: Q8_1 X reads + HFQ4 weights + Y writes. Q8_1 = ~1B/elem
        // (slightly more for the per-sub-block (d,sum) metadata) vs FP16 = 2B/elem.
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

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW estimate: same as the k2 sibling — Q8_1 X reads + HFQ4 weights
        // + Y writes. k4 is a pure unroll-depth change, no extra memory traffic.
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

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW estimate: same as the k2/k4 siblings — Q8_1 X reads + HFQ4 weights
        // + Y writes. k8 is a pure unroll-depth change, no extra memory traffic.
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

        let row_tiles = ((m + 63) / 64) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let bytes = (m_total * k) + (m_total * m) * 4 + (crate::profile::gemv_hfq4g256_bytes(m, k));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_kernargs(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [128, 1, 1],
            0,
            &kernargs![ptr ep, ptr tp, ptr sp, ptr xp, ptr yp, i32 m_val, i32 k_val, i32 xrd_val, i32 mt_val, i32 xsr_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub(crate) fn gemm_hfq4g256_mmq_gfx1151(
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
    pub(crate) fn gemm_hfq4g256_mmq_gfx1151_prequant(
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
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * k
            + batch_size * m * 4 * if add { 2 } else { 1 };
        let label = if add {
            "gemm_hfq4g256_mmq_gfx1151_add"
        } else {
            "gemm_hfq4g256_mmq_gfx1151_set"
        };
        let timer = crate::profile::begin_timer(&self.hip, "gemm", label, bytes);
        let result = self.launch_kernargs(
            "gemm_hfq4g256_mmq_gfx1151",
            [(m / 16) as u32, (batch_size / 16) as u32, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr a_ptr, ptr x_q8_ptr, ptr y_ptr, i32 m_val, i32 k_val, i32 n_val, i32 add_val],
        );
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

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x_f16_ptr;
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;

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
        let result = self.launch_kernargs(
            "gemm_hfq6g256_residual_wmma_4w_gfx1151",
            [row_tiles as u32, batch_tiles as u32, 1],
            [128, 1, 1],
            0,
            &kernargs![ptr a_ptr, ptr x_ptr, ptr y_ptr, i32 m_val, i32 k_val, i32 bs_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Synthetic gfx1151 S4 x S4 -> I32 WMMA tile probe. This validates the
    /// signed packed-Q4 operand layout needed before any HFQ4/MQ4 IU4
    /// approximation is evaluated. It is deliberately not routed into model
    /// code.
    pub fn gemm_s4s4_wmma_tile_gfx1151(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch != "gfx1151" {
            panic!(
                "gemm_s4s4_wmma_tile_gfx1151: only gfx1151 is supported; arch={}",
                self.arch
            );
        }
        self.ensure_kernel(
            "gemm_s4s4_wmma_tile_gfx1151",
            kernels::GEMM_S4S4_WMMA_TILE_GFX1151_SRC,
            "gemm_s4s4_wmma_tile_gfx1151",
        )?;

        let a_ptr = a.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let n_val = n as i32;
        self.launch_kernargs(
            "gemm_s4s4_wmma_tile_gfx1151",
            [m.div_ceil(16) as u32, n.div_ceil(16) as u32, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr a_ptr, ptr x_ptr, ptr y_ptr, i32 m_val, i32 k_val, i32 n_val],
        )
    }
    /// k8 deepest-pipeline sibling of `gemm_paro_q4g128_moe_grouped_mmq_gfx1151`.
    /// 8 WMMAs into 4 independent int32 accumulators per HFQ4G128 group.
    /// Same kernarg layout + grid as k2. Used via HIPFIRE_MOE_PARO_I8_K8=1.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_paro_q4g128_moe_grouped_mmq_k8_gfx1151(
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
        let kernel_name = "gemm_paro_q4g128_moe_grouped_mmq_k8_gfx1151";
        let kernel_src = kernels::GEMM_PARO_Q4G128_MOE_GROUPED_MMQ_K8_GFX1151_SRC;
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
        let bytes = (m_total * k) + (m_total * m) * 4 + (crate::profile::gemv_hfq4g128_bytes(m, k));
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
    /// gfx1151 i8 WMMA MMQ MoE grouped GEMM for HFQ4G128 (ParoQuant). Same
    /// scatter contract + per-sub-block scale-FMA convention as the
    /// HFQ4G256 sister `gemm_hfq4g256_moe_grouped_mmq_gfx1151`. Auto-
    /// quantizes F32 x_src to Q8_1 via `ensure_q8_1_mmq_x` (shared
    /// scratch). Compute-bound regime: ~140 TFLOPS i8 WMMA vs ~71 TFLOPS
    /// FP16 WMMA. gfx1151-only — the kernel guards on `__gfx1151__` and
    /// is a no-op stub on other archs.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_paro_q4g128_moe_grouped_mmq_gfx1151(
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
        let kernel_name = "gemm_paro_q4g128_moe_grouped_mmq_gfx1151";
        let kernel_src = kernels::GEMM_PARO_Q4G128_MOE_GROUPED_MMQ_GFX1151_SRC;
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
        // BW estimate: Q8_1 X reads + HFQ4G128 weights + Y writes.
        let bytes = (m_total * k) + (m_total * m) * 4 + (crate::profile::gemv_hfq4g128_bytes(m, k));
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
    /// gfx1151 4-warp 64x64 Q8_0 GEMM with fused residual add.
    pub fn gemm_q8_0_residual_wmma_4w_gfx1151(
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
            self.arch, "gfx1151",
            "gemm_q8_0_residual_wmma_4w_gfx1151 is gfx1151-only"
        );
        debug_assert_eq!(
            k % 32,
            0,
            "gemm_q8_0_residual_wmma_4w_gfx1151: K must be a multiple of 32"
        );
        debug_assert_eq!(
            batch_size % 64,
            0,
            "gemm_q8_0_residual_wmma_4w_gfx1151: N must be a multiple of 64"
        );
        self.ensure_kernel(
            "gemm_q8_0_residual_wmma_4w_gfx1151",
            kernels::GEMM_Q8_0_RESIDUAL_WMMA_4W_GFX1151_SRC,
            "gemm_q8_0_residual_wmma_4w_gfx1151",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let a_p = a.buf.as_ptr();
        let xp = x_f16_ptr;
        let y_p = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let n_val = batch_size as i32;

        let row_tiles = (m + 63) / 64;
        let batch_tiles = (batch_size + 63) / 64;
        let bytes = m * (k / 32) * 34 + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_q8_0_residual_wmma_4w_gfx1151",
            bytes,
        );
        let result = self.launch_kernargs(
            "gemm_q8_0_residual_wmma_4w_gfx1151",
            [row_tiles as u32, batch_tiles as u32, 1],
            [128, 1, 1],
            0,
            &kernargs![ptr a_p, ptr xp, ptr y_p, i32 m_val, i32 k_val, i32 n_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// gfx1151 4-warp 64x64 WMMA QKVZA GEMM for HFQ6/MQ6 large prefill.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq6g256_wmma_4w_gfx1151(
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
            self.arch, "gfx1151",
            "gemm_qkvza_hfq6g256_wmma_4w_gfx1151 is gfx1151-only"
        );
        debug_assert_eq!(
            k % 256,
            0,
            "gemm_qkvza_hfq6g256_wmma_4w_gfx1151: K must be a multiple of 256"
        );
        debug_assert_eq!(
            batch_size % 64,
            0,
            "gemm_qkvza_hfq6g256_wmma_4w_gfx1151: N must be a multiple of 64"
        );
        self.ensure_kernel(
            "gemm_qkvza_hfq6g256_wmma_4w_gfx1151",
            kernels::GEMM_QKVZA_HFQ6G256_WMMA_4W_GFX1151_SRC,
            "gemm_qkvza_hfq6g256_wmma_4w_gfx1151",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let aq = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let xp = x_f16_ptr;
        let yq = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let q_m = qkv_m as i32;
        let z_m_val = z_m as i32;
        let b_m = beta_m as i32;
        let a_m = alpha_m as i32;
        let k_val = k as i32;
        let n_val = batch_size as i32;

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 63) / 64;
        let batch_tiles = (batch_size + 63) / 64;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_qkvza_hfq6g256_wmma_4w_gfx1151",
            bytes,
        );
        let result = self.launch_kernargs(
            "gemm_qkvza_hfq6g256_wmma_4w_gfx1151",
            [row_tiles as u32, batch_tiles as u32, 1],
            [128, 1, 1],
            0,
            &kernargs![ptr aq, ptr az, ptr ab, ptr aa, ptr xp, ptr yq, ptr yz, ptr yb, ptr ya, i32 q_m, i32 z_m_val, i32 b_m, i32 a_m, i32 k_val, i32 n_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// gfx1151 4-warp 64x64 WMMA QKV GEMM for HFQ6/MQ6 large prefill.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq6g256_wmma_4w_gfx1151(
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
            self.arch, "gfx1151",
            "gemm_qkv_hfq6g256_wmma_4w_gfx1151 is gfx1151-only"
        );
        debug_assert_eq!(
            k % 256,
            0,
            "gemm_qkv_hfq6g256_wmma_4w_gfx1151: K must be a multiple of 256"
        );
        debug_assert_eq!(
            batch_size % 64,
            0,
            "gemm_qkv_hfq6g256_wmma_4w_gfx1151: N must be a multiple of 64"
        );
        self.ensure_kernel(
            "gemm_qkv_hfq6g256_wmma_4w_gfx1151",
            kernels::GEMM_QKV_HFQ6G256_WMMA_4W_GFX1151_SRC,
            "gemm_qkv_hfq6g256_wmma_4w_gfx1151",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let xp = x_f16_ptr;
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_val = q_m as i32;
        let k_m_val = k_m as i32;
        let v_m_val = v_m as i32;
        let k_val = k as i32;
        let n_val = batch_size as i32;

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 63) / 64;
        let batch_tiles = (batch_size + 63) / 64;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_qkv_hfq6g256_wmma_4w_gfx1151",
            bytes,
        );
        let result = self.launch_kernargs(
            "gemm_qkv_hfq6g256_wmma_4w_gfx1151",
            [row_tiles as u32, batch_tiles as u32, 1],
            [128, 1, 1],
            0,
            &kernargs![ptr aq, ptr ak, ptr av, ptr xp, ptr yq, ptr yk, ptr yv, i32 q_m_val, i32 k_m_val, i32 v_m_val, i32 k_val, i32 n_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// gfx1151 4-warp 64x64 Q8_0 QKVZA GEMM. Explicit opt-in until the
    /// end-to-end Qwen35 profile proves the shape gate is consistently a win.
    pub fn gemm_qkvza_q8_0_wmma_4w_gfx1151(
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
            self.arch, "gfx1151",
            "gemm_qkvza_q8_0_wmma_4w_gfx1151 is gfx1151-only"
        );
        debug_assert_eq!(
            k % 32,
            0,
            "gemm_qkvza_q8_0_wmma_4w_gfx1151: K must be a multiple of 32"
        );
        debug_assert_eq!(
            batch_size % 64,
            0,
            "gemm_qkvza_q8_0_wmma_4w_gfx1151: N must be a multiple of 64"
        );
        self.ensure_kernel(
            "gemm_qkvza_q8_0_wmma_4w_gfx1151",
            kernels::GEMM_QKVZA_Q8_0_WMMA_4W_GFX1151_SRC,
            "gemm_qkvza_q8_0_wmma_4w_gfx1151",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let a_qkv_p = a_qkv.buf.as_ptr();
        let a_z_p = a_z.buf.as_ptr();
        let a_beta_p = a_beta.buf.as_ptr();
        let a_alpha_p = a_alpha.buf.as_ptr();
        let xp = x_f16_ptr;
        let y_qkv_p = y_qkv.buf.as_ptr();
        let y_z_p = y_z.buf.as_ptr();
        let y_beta_p = y_beta.buf.as_ptr();
        let y_alpha_p = y_alpha.buf.as_ptr();
        let qkv_m_val = qkv_m as i32;
        let z_m_val = z_m as i32;
        let beta_m_val = beta_m as i32;
        let alpha_m_val = alpha_m as i32;
        let k_val = k as i32;
        let n_val = batch_size as i32;

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 63) / 64;
        let batch_tiles = (batch_size + 63) / 64;
        let q8_bytes = |m: usize| m * (k / 32) * 34;
        let bytes = q8_bytes(qkv_m)
            + q8_bytes(z_m)
            + q8_bytes(beta_m)
            + q8_bytes(alpha_m)
            + batch_size * k * 2
            + batch_size * total_m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_qkvza_q8_0_wmma_4w_gfx1151",
            bytes,
        );
        let result = self.launch_kernargs(
            "gemm_qkvza_q8_0_wmma_4w_gfx1151",
            [row_tiles as u32, batch_tiles as u32, 1],
            [128, 1, 1],
            0,
            &kernargs![ptr a_qkv_p, ptr a_z_p, ptr a_beta_p, ptr a_alpha_p, ptr xp, ptr y_qkv_p, ptr y_z_p, ptr y_beta_p, ptr y_alpha_p, i32 qkv_m_val, i32 z_m_val, i32 beta_m_val, i32 alpha_m_val, i32 k_val, i32 n_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// gfx1151 4-warp 64x64 Q8_0 QKV GEMM for large prefill shapes.
    pub fn gemm_qkv_q8_0_wmma_4w_gfx1151(
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
            self.arch, "gfx1151",
            "gemm_qkv_q8_0_wmma_4w_gfx1151 is gfx1151-only"
        );
        debug_assert_eq!(
            k % 32,
            0,
            "gemm_qkv_q8_0_wmma_4w_gfx1151: K must be a multiple of 32"
        );
        debug_assert_eq!(
            batch_size % 64,
            0,
            "gemm_qkv_q8_0_wmma_4w_gfx1151: N must be a multiple of 64"
        );
        self.ensure_kernel(
            "gemm_qkv_q8_0_wmma_4w_gfx1151",
            kernels::GEMM_QKV_Q8_0_WMMA_4W_GFX1151_SRC,
            "gemm_qkv_q8_0_wmma_4w_gfx1151",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let xp = x_f16_ptr;
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_val = q_m as i32;
        let k_m_val = k_m as i32;
        let v_m_val = v_m as i32;
        let k_val = k as i32;
        let n_val = batch_size as i32;

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 63) / 64;
        let batch_tiles = (batch_size + 63) / 64;

        let q8_bytes = |m: usize| m * (k / 32) * 34;
        let bytes = q8_bytes(q_m)
            + q8_bytes(k_m)
            + q8_bytes(v_m)
            + batch_size * k * 2
            + batch_size * total_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_q8_0_wmma_4w_gfx1151", bytes);
        let result = self.launch_kernargs(
            "gemm_qkv_q8_0_wmma_4w_gfx1151",
            [row_tiles as u32, batch_tiles as u32, 1],
            [128, 1, 1],
            0,
            &kernargs![ptr aq, ptr ak, ptr av, ptr xp, ptr yq, ptr yk, ptr yv, i32 q_m_val, i32 k_m_val, i32 v_m_val, i32 k_val, i32 n_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// gfx1151 F16 routed-MoE indexed gate/up GEMV. This is the compact
    /// small-prefill path for full-precision experts; it computes N*K_TOP
    /// real routed slots and avoids grouped-GEMM padding.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_f16_moe_gate_up_k8_indexed_batched_gfx1151(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x_src: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch != "gfx1151" {
            panic!(
                "gemv_f16_moe_gate_up_k8_indexed_batched_gfx1151: only gfx1151 is supported; arch={}",
                self.arch
            );
        }
        self.ensure_kernel(
            "gemv_fp16_moe_gate_up_indexed_batched_gfx1151",
            kernels::GEMV_FP16_MOE_GATE_UP_INDEXED_BATCHED_GFX1151_SRC,
            "gemv_f16_moe_gate_up_k8_indexed_batched_gfx1151",
        )?;
        let xp = self.ensure_fp16_x(x_src, batch_size * k)?;
        let ep = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let gp = y_gate.buf.as_ptr();
        let up = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let bytes = m * k * 2 + batch_size * k * 2 + batch_size * k_top * (m / 2) * 2 * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_f16_moe_gate_up_k8_indexed_batched_gfx1151",
            bytes,
        );
        let result = self.launch_kernargs(
            "gemv_f16_moe_gate_up_k8_indexed_batched_gfx1151",
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &kernargs![ptr ep, ptr ip, ptr xp, ptr gp, ptr up, i32 m_val, i32 k_val, i32 kt_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// gfx1151 BF16 routed-MoE indexed gate/up GEMV. Input activations are
    /// staged to BF16 first, matching the grouped BF16 WMMA path's precision.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_bf16_moe_gate_up_k8_indexed_batched_gfx1151(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x_src: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch != "gfx1151" {
            panic!(
                "gemv_bf16_moe_gate_up_k8_indexed_batched_gfx1151: only gfx1151 is supported; arch={}",
                self.arch
            );
        }
        self.ensure_kernel(
            "gemv_fp16_moe_gate_up_indexed_batched_gfx1151",
            kernels::GEMV_FP16_MOE_GATE_UP_INDEXED_BATCHED_GFX1151_SRC,
            "gemv_bf16_moe_gate_up_k8_indexed_batched_gfx1151",
        )?;
        let xp = self.ensure_bf16_x(x_src, batch_size * k)?;
        let ep = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let gp = y_gate.buf.as_ptr();
        let up = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let bytes = m * k * 2 + batch_size * k * 2 + batch_size * k_top * (m / 2) * 2 * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_bf16_moe_gate_up_k8_indexed_batched_gfx1151",
            bytes,
        );
        let result = self.launch_kernargs(
            "gemv_bf16_moe_gate_up_k8_indexed_batched_gfx1151",
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &kernargs![ptr ep, ptr ip, ptr xp, ptr gp, ptr up, i32 m_val, i32 k_val, i32 kt_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
