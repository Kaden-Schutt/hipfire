// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Remaining GEMM dtypes (q8_0, mq/lloyd, fp4/oq/iu, paro, s4s4). Pure move (Phase 1 M6).

use super::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;
use std::sync::OnceLock;

impl Gpu {
    /// MQ4-Lloyd WMMA-accelerated batched residual GEMM (Phase 5b / issue #182,
    /// Phase B1). Mirrors gemm_mq3g256_lloyd_residual_wmma's wiring, with 160 B/
    /// group + 16-entry codebook + nibble-pair decode. fp16-LDS staging — fp16
    /// won the MQ3 Phase A bench by 7.15% (decision inherited).
    /// gfx11/gfx12 wave32 WMMA; other archs fall through to baseline (which
    /// itself currently requires WMMA — caller must check arch first).
    pub fn gemm_mq4g256_lloyd_residual_wmma(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — path selector; concrete launch path binds before HIP use
        // Phase D-A path selector: route to `_mb4` (16×64 output tile, 4× weight
        // reuse per WG) when shape clears the size gate. Bench (gfx1151,
        // benchmarks/results/devlog_20260509_mq4_lloyd_gfx1151_bench.md):
        // 1.40-2.24× speedup at production shapes (M ≥ 4096, batch ≥ 128);
        // small shapes regress (4× WG reduction + 106 VGPR leaves CUs idle).
        // Threshold tuning open — see Phase D plan §"Open questions" #3.
        // Env override: HIPFIRE_LLOYD_MB4=1 force-on, =0 force-off.
        let arch_supports_mb4 = self.arch_caps.is_rdna3()
            && matches!(
                self.arch.as_str(),
                "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151"
            );
        let use_mb4 = match self.flags.lloyd_mb4 {
            Some(_) => arch_supports_mb4,
            None => arch_supports_mb4 && batch_size >= 128 && m >= 4096,
        };
        if use_mb4 {
            return self.gemm_mq4g256_lloyd_residual_wmma_mb4(a_raw, x, y, m, k, batch_size);
        }
        self.bind_thread()?;
        let (src, module) = kernels::gemm_mq4g256_lloyd_residual_wmma_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_mq4g256_lloyd_residual_wmma")?;
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

        let weight_bytes = m * (k / 256) * super::LLOYD_MQ4_GROUP_BYTES;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_mq4g256_lloyd_residual_wmma",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_mq4g256_lloyd_residual_wmma",
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
    /// MQ4-Lloyd WMMA residual GEMM, 4× batch-tile fanout per WG (Phase D-A).
    /// Same args as `gemm_mq4g256_lloyd_residual_wmma`; only the grid shape and
    /// per-WG output tile differ (16×64 vs 16×16).
    ///
    /// Caller is responsible for the path-selection gate. This kernel is shipped
    /// dead-code-safe: parity test wires it directly; production matcher routing
    /// lands in Phase D-C.
    pub fn gemm_mq4g256_lloyd_residual_wmma_mb4(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemm_mq4g256_lloyd_residual_wmma_mb4_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_mq4g256_lloyd_residual_wmma_mb4")?;
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

        let weight_bytes = m * (k / 256) * super::LLOYD_MQ4_GROUP_BYTES;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_mq4g256_lloyd_residual_wmma_mb4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_mq4g256_lloyd_residual_wmma_mb4",
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
    /// Phase D experiment: 16×32 fanout sibling of `gemm_mq4g256_lloyd_residual_wmma`.
    /// Half the per-WG weight reuse of mb4 but 2× the WG count and ~21 fewer
    /// VGPRs — targets the small-M residual case where mb4 is occupancy-bound.
    pub fn gemm_mq4g256_lloyd_residual_wmma_mb2(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemm_mq4g256_lloyd_residual_wmma_mb2_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_mq4g256_lloyd_residual_wmma_mb2")?;
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
        let batch_tiles = (batch_size + 31) / 32;

        let weight_bytes = m * (k / 256) * super::LLOYD_MQ4_GROUP_BYTES;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_mq4g256_lloyd_residual_wmma_mb2",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_mq4g256_lloyd_residual_wmma_mb2",
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
    /// MQ3-Lloyd WMMA residual GEMM (Phase 5 / issue #116, Phase B1).
    /// Mirrors `gemm_hfq3g256_residual_wmma` shape + grid; group stride is 112 B
    /// (16 B fp16 codebook + 96 B 3-bit indices) instead of HFQ3's 104. K must
    /// be a multiple of 256. gfx11/gfx12 wave32 WMMA; other archs fall through
    /// to the baseline kernel (which itself currently requires WMMA — caller
    /// must check arch before dispatching).
    /// Caller is responsible for pre-rotating X (FWHT) for the MQ3-Lloyd dtype;
    /// this dispatch mirrors `gemm_hfq3g256_residual_wmma` and does not rotate.
    /// fp16-LDS staging — fp16 won the Phase A bench by 7.15% (devlog
    /// 2026-05-07).
    pub fn gemm_mq3g256_lloyd_residual_wmma(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // mb4 path selector — same gate as MQ4-Lloyd's mb4 family.
        let arch_supports_mb4 = self.arch_caps.is_rdna3()
            && !self.arch_caps.is_gfx1152()
            && !self.arch_caps.is_gfx1103();
        let use_mb4 = match self.flags.mq3_mb4 {
            Some(_) => arch_supports_mb4,
            None => arch_supports_mb4 && batch_size >= 128 && m >= 4096,
        };
        if use_mb4 {
            return self.gemm_mq3g256_lloyd_residual_wmma_mb4(a_raw, x, y, m, k, batch_size);
        }
        let (src, module) = kernels::gemm_mq3g256_lloyd_residual_wmma_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_mq3g256_lloyd_residual_wmma")?;
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

        let weight_bytes = m * (k / 256) * super::LLOYD_MQ3_GROUP_BYTES;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_mq3g256_lloyd_residual_wmma",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_mq3g256_lloyd_residual_wmma",
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
    /// MQ3-Lloyd WMMA residual mb4: 16×64 output tile per WG. Sibling of
    /// `gemm_mq4g256_lloyd_residual_wmma_mb4` ported to the MQ3 codebook
    /// (8 entries) + 3-bit cross-byte K-tile decode.
    pub fn gemm_mq3g256_lloyd_residual_wmma_mb4(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemm_mq3g256_lloyd_residual_wmma_mb4_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_mq3g256_lloyd_residual_wmma_mb4")?;
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

        let weight_bytes = m * (k / 256) * super::LLOYD_MQ3_GROUP_BYTES;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_mq3g256_lloyd_residual_wmma_mb4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_mq3g256_lloyd_residual_wmma_mb4",
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
    /// HFP4-G32 batched residual GEMM with fused += semantics.
    /// Sister of `gemm_hfq4g256_residual_wmma_k2`. Used for wo + w_down
    /// projections in the batched prefill path. Routes to gfx11/gfx12.
    /// Caller must initialize Y to the residual stream before this call.
    pub fn gemm_hfp4g32_residual(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch_caps.has_wmma_w32_gfx12() {
            return self.gemm_hfp4g32_residual_wmma_gfx12(a, x, y, m, k, batch_size);
        }
        self.gemm_hfp4g32_residual_wmma(a, x, y, m, k, batch_size)
    }
    pub fn gemm_hfp4g32_residual_wmma(
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
            "gemm_hfp4g32_residual_wmma",
            kernels::GEMM_HFP4G32_RESIDUAL_WMMA_SRC,
            "gemm_hfp4g32_residual_wmma",
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
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfp4g32_residual_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_hfp4g32_residual_wmma",
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
    /// SGLang-style grouped-WMMA-GEMM for HFQ4G128 (ParoQuant) routed
    /// experts. Sister of `gemm_hfq4g256_moe_grouped_wmma_k2` with the
    /// 72 B/group HFQ4G128 stride. F32 x_src is auto-converted to F16
    /// via `ensure_fp16_x` (same convention as the G256 sister). Used
    /// by the Path 2 routed-expert dispatch in
    /// `prefill_moe_ffn_body_batched` on gfx11/gfx12 when ParoQ4G128
    /// experts are admitted (HIPFIRE_PARO_BATCHED=1). No i8 MMQ variant
    /// today — HFQ4G128 doesn't have a Q8_1 prequant pipeline; if needed
    /// later this would parallel `gemm_hfq4g256_moe_grouped_mmq_gfx1151`.
    ///
    /// `x_src_rows` is the number of rows in x_src (N for gate_up,
    /// N*K_TOP for down).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_paro_q4g128_moe_grouped_wmma_k2(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to FP16)
        y_grouped: &GpuTensor,          // [m_total × M] f32
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_paro_q4g128_moe_grouped_wmma_k2",
            kernels::GEMM_PARO_Q4G128_MOE_GROUPED_WMMA_K2_SRC,
            "gemm_paro_q4g128_moe_grouped_wmma_k2",
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
        let bytes =
            m_total * k * 2 + (m_total * m) * 4 + (crate::profile::gemv_hfq4g128_bytes(m, k));
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_paro_q4g128_moe_grouped_wmma_k2",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_paro_q4g128_moe_grouped_wmma_k2",
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
    pub fn gemm_mq3g256_residual_wmma(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        for b in 0..batch_size {
            let x_row = x.sub_offset(b * k, k);
            let x_rot_row = x_rot.sub_offset(b * k, k);
            self.rotate_x_mq(&x_row, &x_rot_row, k)?;
        }
        self.fp16_x_source_ptr = std::ptr::null_mut();
        self.gemm_hfq3g256_residual_wmma(a_raw, x_rot, y, m, k, batch_size)
    }
    /// MW16: dequant 4-bit weights to FP16, then run the no-dequant WMMA kernel.
    /// Per-call dequant (wasteful) — for benchmarking only. Production would
    /// dequant at model load time.
    pub(crate) fn gemm_mw16_residual_wmma_via_dequant(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_kernel(
            "dequant_hfq4g256_to_f16",
            kernels::DEQUANT_HFQ4G256_TO_F16_SRC,
            "dequant_hfq4g256_to_f16",
        )?;
        self.ensure_kernel(
            "gemm_mw16_residual_wmma",
            kernels::GEMM_MW16_RESIDUAL_WMMA_SRC,
            "gemm_mw16_residual_wmma",
        )?;
        let x_f16 = self.ensure_fp16_x(x, batch_size * k)?;

        // Dequant weights to FP16 scratch
        let w_elems = m * k;
        let w_f16 = self.hip.malloc(w_elems * 2)?;
        {
            let f = &self.functions["dequant_hfq4g256_to_f16"];
            let groups = k / 256;
            let mut ap = a_raw.buf.as_ptr();
            let mut wp = w_f16.as_ptr();
            let mut mv = m as i32;
            let mut kv = k as i32;
            let mut p: Vec<*mut c_void> = vec![
                &mut ap as *mut _ as *mut c_void,
                &mut wp as *mut _ as *mut c_void,
                &mut mv as *mut _ as *mut c_void,
                &mut kv as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    f,
                    [m as u32, groups as u32, 1],
                    [32, 1, 1],
                    0,
                    self.stream_ref(),
                    &mut p,
                )?;
            }
        }

        // MW16 WMMA GEMM
        let f = &self.functions["gemm_mw16_residual_wmma"];
        let mut wp = w_f16.as_ptr();
        let mut xp = x_f16;
        let mut yp = y.buf.as_ptr();
        let mut mv = m as i32;
        let mut kv = k as i32;
        let mut nv = batch_size as i32;
        let mut p: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mv as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
        ];
        let rows = (m + 15) / 16;
        let batches = (batch_size + 15) / 16;
        let bytes = m * k * 2 + batch_size * k * 2 + batch_size * m * 8;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_mw16_residual_wmma", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                f,
                [rows as u32, batches as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut p,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        drop(w_f16);
        result
    }
    /// Y[batch, M] = X[batch, K] @ A_q8[M, K]^T — batched Q8_0 GEMM.
    /// One block per output row (32 threads, one wave). Each thread holds
    /// MAX_BATCH=16 per-batch accumulators and broadcasts each weight load.
    /// Drops the (batch_size − 1)× weight re-reads of the GEMV-loop path
    /// without splitting launches.
    pub fn gemm_q8_0_batched(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            batch_size <= 64,
            "gemm_q8_0_batched: batch_size {batch_size} exceeds kernel MAX_BATCH=64"
        );
        self.ensure_kernel(
            "gemm_q8_0_batched",
            kernels::GEMM_Q8_0_BATCHED_SRC,
            "gemm_q8_0_batched",
        )?;

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

        self.launch_maybe_blob(
            "gemm_q8_0_batched",
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
                b.push_i32(bs_val);
                b
            },
        )
    }
    /// Q8_0 batched GEMM driver that handles `n` rows by sub-batching at the
    /// kernel's MAX_BATCH=64. Y[n, m] = X[n, k] @ A_q8[m, k]^T.
    ///
    /// On gfx12 (RDNA4) with K % 32 == 0, routes the entire call through
    /// the WMMA Q8 GEMM (`gemm_q8_0_wmma_gfx12`) which is ~3-4× faster
    /// than the scalar `gemm_q8_0_batched` per output. Opt out via
    /// HIPFIRE_Q8_BATCHED_LEGACY=1.
    pub fn gemm_q8_0_batched_chunked(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        static USE_LEGACY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_legacy = *USE_LEGACY.get_or_init(|| self.flags.q8_batched_legacy);
        // 2026-05-26: 4-warp 64×64 WMMA path (gemm_q8_0_wmma_4w). Matches
        // llama.cpp's gfx1151 MMQ pattern from issue #21284 (mmq_x=48,
        // mmq_y=64, nwarps=4). LDS-staged X, 4× weight reuse per block
        // vs the single-warp 16×16 baseline.
        //
        // Microbench (bench_q8_wmma_4w, 30 trials, gfx1151):
        //   M=32768 K=1536 B=1024 (wq_b shape): 30.8ms → 9.3ms = 3.30× faster
        //   M=4096  K=4096 B=1024:               7.5ms → 3.6ms = 2.09×
        //   M=4096  K=4096 B=256:                2.1ms → 1.1ms = 1.95×
        //   M=4096  K=4096 B=64:                 374µs → 450µs = 0.83× (slower)
        //   M≤1024  B=64:                                       0.27-0.48× (slower)
        //
        // Gate: route through 4w when batch_size is large enough to amortize
        // the 64×64 block tile. Threshold B≥128 keeps the small-batch path
        // on the single-warp kernel. M%64==0 strictly required by 4w
        // (kernel ASSERT). When M doesn't divide 64, fall through to the
        // single-warp variant.
        //
        // Opt-in via HIPFIRE_Q8_WMMA_4W=1. Default OFF: commit be57d8d
        // originally shipped default-ON on gfx11/gfx12, but the auto-enable
        // measured ~15% prefill TPS regression (49.3 → 41.7) per the
        // commit message itself. Restore documented behavior; the kernel
        // needs a tighter shape gate before the default can flip on.
        let use_4w = std::env::var("HIPFIRE_Q8_WMMA_4W").as_deref() == Ok("1");
        if !use_legacy && use_4w && k % 32 == 0 && m % 64 == 0 && n % 64 == 0 && n >= 128 {
            return self.gemm_q8_0_wmma_4w(a_raw, x, y, m, k, n);
        }
        if !use_legacy
            && k % 32 == 0
            && n >= 64
            && (n % 64 == 0)
            && std::env::var("HIPFIRE_Q8_WMMA_X64").as_deref() == Ok("1")
        {
            return self.gemm_q8_0_wmma_x64(a_raw, x, y, m, k, n);
        }
        if !use_legacy && self.arch_caps.is_rdna4() && k % 32 == 0 && n > 0 {
            return self.gemm_q8_0_wmma(a_raw, x, y, m, k, n);
        }

        const MAX_BATCH: usize = 64;
        let mut off = 0;
        while off < n {
            let take = (n - off).min(MAX_BATCH);
            let x_sub = x.sub_offset(off * k, take * k);
            let y_sub = y.sub_offset(off * m, take * m);
            self.gemm_q8_0_batched(a_raw, &x_sub, &y_sub, m, k, take)?;
            off += take;
        }
        Ok(())
    }
    /// Wider-N Q8_0 WMMA: 16×64 output tile instead of 16×16. Same
    /// single-warp wave32 structure as `gemm_q8_0_wmma` but each K-step
    /// issues 4 back-to-back WMMAs sharing one A fragment against 4
    /// different B fragments — 4× weight reuse per block.
    ///
    /// Caller-checked: K % 32 == 0 AND N % 64 == 0.
    pub fn gemm_q8_0_wmma_x64(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(k % 32, 0, "gemm_q8_0_wmma_x64: K must be %32");
        debug_assert_eq!(batch_size % 64, 0, "gemm_q8_0_wmma_x64: N must be %64");
        self.ensure_kernel(
            "gemm_q8_0_wmma_x64",
            kernels::GEMM_Q8_0_WMMA_X64_SRC,
            "gemm_q8_0_wmma_x64",
        )?;
        let xp_owned = x.buf.as_ptr();
        let mut xp = if matches!(x.dtype, DType::F16) {
            xp_owned
        } else {
            self.ensure_fp16_x(x, batch_size * k)?
        };
        let mut a_p = a.buf.as_ptr();
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
        let batch_tiles = (batch_size + 63) / 64;
        let bytes = m * (k / 32) * 34 + batch_size * k * 2 + batch_size * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_q8_0_wmma_x64", bytes);
        let result = self.launch_maybe_blob(
            "gemm_q8_0_wmma_x64",
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
    /// 4-warp 64×64 Q8_0 GEMM for gfx1151. Drop-in alternate for
    /// `gemm_q8_0_wmma`. Requires M % 64 == 0 and N % 64 == 0 (caller
    /// pads if needed). LDS-staged X cooperative load amortizes weight
    /// reads across 64 batch positions (vs 16 in the single-warp variant).
    pub fn gemm_q8_0_wmma_4w(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(k % 32, 0, "gemm_q8_0_wmma_4w: K must be a multiple of 32");
        debug_assert_eq!(
            m % 64,
            0,
            "gemm_q8_0_wmma_4w: M must be a multiple of 64 (got {m})"
        );
        debug_assert_eq!(
            batch_size % 64,
            0,
            "gemm_q8_0_wmma_4w: N must be a multiple of 64 (got {batch_size})"
        );
        self.ensure_kernel(
            "gemm_q8_0_wmma_4w",
            kernels::GEMM_Q8_0_WMMA_4W_SRC,
            "gemm_q8_0_wmma_4w",
        )?;
        // Stage F32 → F16 input if needed.
        let xp_owned = x.buf.as_ptr();
        let mut xp = if matches!(x.dtype, DType::F16) {
            xp_owned
        } else {
            self.ensure_fp16_x(x, batch_size * k)?
        };

        let mut a_p = a.buf.as_ptr();
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
        let row_tiles = (m + 63) / 64;
        let batch_tiles = (batch_size + 63) / 64;
        let func = &self.functions["gemm_q8_0_wmma_4w"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, batch_tiles as u32, 1],
                [128, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// WMMA Q8_0 GEMM (no residual). Y[N, M] = X[N, K] @ A_q8[M, K]^T.
    /// Arch-aware: dispatches the gfx12-specific intrinsic on RDNA4 and
    /// the cross-RDNA (RDNA3+, gfx1100/gfx115x/gfx1200) intrinsic on
    /// older archs. X may be F32 (auto-converted internally) or F16
    /// (passed through). Drop-in replacement for `gemm_q8_0_batched`;
    /// the scalar 1-wave-per-row kernel was 65% of A3B prefill GPU time
    /// per rocprofv3 2026-05-19.
    pub fn gemm_q8_0_wmma(
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
            "gemm_q8_0_wmma: K must be a multiple of 32 (got K={k})"
        );

        // 4-warp 64×64 fast path for gfx1151 (RDNA3.5). Microbench shows
        // 2.0-3.3× speedup at the actual deepseek4 prefill shapes (M ∈
        // {1536, 4096, 32768}, B ∈ {256, 1024}) — bench_q8_wmma_4w.
        // OPT-IN via HIPFIRE_Q8_WMMA_4W=1 — end-to-end auto-enable on
        // gfx1151 measured a ~15% prefill TPS regression in one bench
        // run; suspected thermal/cache-pollution interaction with the
        // surrounding kernels, but unconfirmed.  Default off until the
        // end-to-end gap is understood (see project_llamacpp_21284
        // execution memory for the open thread).
        if m % 64 == 0
            && batch_size % 64 == 0
            && std::env::var("HIPFIRE_Q8_WMMA_4W").as_deref() == Ok("1")
        {
            return self.gemm_q8_0_wmma_4w(a, x, y, m, k, batch_size);
        }

        // Arch-aware kernel selection. The gfx12-specific intrinsic
        // `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` is RDNA4-only;
        // gfx115x (Strix Halo / RDNA 3.5) uses the cross-RDNA intrinsic
        // `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`. Both kernels share
        // the same signature (A_q8, X_f16, Y_f32, M, K, N) and launch shape.
        // NOTE: origin/master refactored this to gfx12-only — keeping the
        // cross-RDNA branch here because deepseek4 V4F forward calls this
        // directly on gfx1151 (RDNA3.5).
        let (kname, src) = if self.arch_caps.is_rdna4() {
            ("gemm_q8_0_wmma_gfx12", kernels::GEMM_Q8_0_WMMA_GFX12_SRC)
        } else {
            ("gemm_q8_0_wmma", kernels::GEMM_Q8_0_WMMA_SRC)
        };
        self.ensure_kernel(kname, src, kname)?;

        // If caller pre-converted X to F16, use it directly. Otherwise
        // convert F32 → F16 UNCONDITIONALLY (no pointer-keyed cache).
        //
        // STALE-CACHE BUG (origin/master 49881383): the pointer-keyed
        // `ensure_fp16_x` skips reconversion when the source ptr matches the
        // last call. In the MTP decode path the `tmp_batched` lm_head scratch
        // is a FIXED allocation (same device address) refilled with NEW content
        // by `rmsnorm_batched` each step — so after step 0 warms the cache,
        // every later step read stale step-0 FP16, producing wrong draft logits
        // and collapsing τ (~1.85 → ~1.01 on gfx11). gfx12 / batched-verify
        // dodged it by churning the shared scratch between proposals.
        // `convert_fp16_x_uncached` always re-converts (one extra cheap F32→F16
        // kernel per call, negligible vs the GEMM).
        let xp_owned = x.buf.as_ptr();
        let mut xp = if matches!(x.dtype, DType::F16) {
            xp_owned
        } else {
            self.convert_fp16_x_uncached(x, batch_size * k)?
        };

        let mut a_p = a.buf.as_ptr();
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
        let bytes = m * (k / 32) * 34 + batch_size * k * 2 + batch_size * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kname, bytes);
        let result = self.launch_maybe_blob(
            kname,
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
    /// WMMA Q8_0 GEMM with fused residual add (Y += X @ A^T).
    /// Caller seeds Y with the residual. Auto-routes to gfx12 sibling
    /// on RDNA4.
    pub fn gemm_q8_0_residual_wmma(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch_caps.is_rdna4() {
            return self.gemm_q8_0_residual_wmma_gfx12(a, x, y, m, k, batch_size);
        }
        debug_assert_eq!(
            k % 32,
            0,
            "gemm_q8_0_residual_wmma: K must be a multiple of 32 (got K={k})"
        );
        static Q8_RESIDUAL_4W: OnceLock<Option<bool>> = OnceLock::new();
        let auto_q8_residual_4w = batch_size >= 128 || (batch_size >= 64 && m == 3072 && k >= 8192);
        let q8_residual_4w = Self::gfx1151_q8_4w_enabled(
            *Q8_RESIDUAL_4W.get_or_init(|| Self::q8_4w_mode("HIPFIRE_Q8_RESIDUAL_4W")),
            auto_q8_residual_4w,
        );
        if q8_residual_4w && self.arch == "gfx1151" && batch_size % 64 == 0 {
            return self.gemm_q8_0_residual_wmma_4w_gfx1151(a, x, y, m, k, batch_size);
        }
        self.ensure_kernel(
            "gemm_q8_0_residual_wmma",
            kernels::GEMM_Q8_0_RESIDUAL_WMMA_SRC,
            "gemm_q8_0_residual_wmma",
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
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_q8_0_residual_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_q8_0_residual_wmma",
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
    /// HFP4G32 / MFP4G32 grouped-WMMA-GEMM for MoE prefill — sister of
    /// `gemm_hfq4g256_moe_grouped_wmma_k2` but on the FP4G32 dequant. Same
    /// kernarg layout, same expert_tile_ids / sorted_slot_index contract.
    /// Tile (blockIdx.x, blockIdx.y) gathers row `sorted_slot_index[slot_start
    /// + m_lane]` (with -1 meaning "padding lane — zero B") and applies the
    /// weights of expert `expert_tile_ids[blockIdx.y]`. Sentinel `< 0` early-
    /// returns the tile so the dispatcher can launch up to m_total_max/16
    /// tiles without an m_total dtoh sync.
    ///
    /// `x_row_div` selects the X gather layout (same as HFQ4 sister):
    ///   gate_up: x_src = x_rot_batch [N × K], x_row_div = K_TOP
    ///   down:    x_src = rot_batch [N*K_TOP × K], x_row_div = 1
    /// `x_src_rows` is the number of rows in x_src (N or N*K_TOP).
    ///
    /// **gfx12 only.** gfx11 and older archs are not implemented and will
    /// panic — call sites should arch-gate before dispatching here.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfp4g32_moe_grouped_wmma(
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
        if !self.arch_caps.is_rdna4() {
            panic!(
                "gemm_hfp4g32_moe_grouped_wmma: only gfx12 (RDNA4) is implemented; \
                 got arch={}. Add a wave32 sister kernel for non-gfx12 archs.",
                self.arch
            );
        }
        let kernel_name = "gemm_hfp4g32_moe_grouped_wmma_gfx12";
        let kernel_src = kernels::GEMM_HFP4G32_MOE_GROUPED_WMMA_GFX12_SRC;
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
        // BW estimate: gather X (fp16) + weight rows (HFP4G32: 18 B/group, K/32 groups
        // per row, ~m_total/E shared per tile) + write Y. Use the existing helper.
        let bytes = m_total * k * 2 + (m_total * m) * 4 + crate::profile::gemv_hfp4g32_bytes(m, k);
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
    /// Generic kernel library: WMMA GEMM, signed INT4 inputs → INT32 output.
    /// `a_i4` [M,K/2], `x_i4` [B,K/2] (packed nibbles, `k_even | k_odd<<4`),
    /// `y_i32` [B,M] (int32). gfx1103/RDNA3 wave32, zero LDS. Requires
    /// `k % 16 == 0` and wave32 WMMA.
    pub fn gemm_iu4_i32_wmma(
        &mut self,
        a_i4: &GpuTensor,
        x_i4: &GpuTensor,
        y_i32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(k % 16, 0, "gemm_iu4_i32_wmma: K must be a multiple of 16");
        self.ensure_kernel(
            "gemm_iu4_i32_wmma",
            kernels::GEMM_IU4_I32_WMMA_SRC,
            "gemm_iu4_i32_wmma",
        )?;
        let ap = a_i4.buf.as_ptr();
        let xp = x_i4.buf.as_ptr();
        let yp = y_i32.buf.as_ptr();
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
        let func = &self.functions["gemm_iu4_i32_wmma"];
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
    /// SpinQuant R1 working copy of [`Self::gemm_iu4_i32_wmma`] (Phase 1b
    /// sandbox, symbol `gemm_iu4_i32_wmma_r1`). Identical contract; kept separate
    /// so the learned-rotation W4A4 path can evolve the kernel without touching
    /// production. Same args: `a_i4` [M,K/2], `x_i4` [B,K/2], `y_i32` [B,M].
    pub fn gemm_iu4_i32_wmma_r1(
        &mut self,
        a_i4: &GpuTensor,
        x_i4: &GpuTensor,
        y_i32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            k % 16,
            0,
            "gemm_iu4_i32_wmma_r1: K must be a multiple of 16"
        );
        self.ensure_kernel(
            "gemm_iu4_i32_wmma_r1",
            kernels::GEMM_IU4_I32_WMMA_R1_SRC,
            "gemm_iu4_i32_wmma_r1",
        )?;
        let ap = a_i4.buf.as_ptr();
        let xp = x_i4.buf.as_ptr();
        let yp = y_i32.buf.as_ptr();
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
        let func = &self.functions["gemm_iu4_i32_wmma_r1"];
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

    /// Tuned wave64 LDS-staged W4A4 GEMM — identical contract to
    /// [`Self::gemm_iu4_i32_wmma`] (`a_i4` [M,K/2], `x_i4` [B,K/2], `y_i32` [B,M]),
    /// ~14× faster on large prefill GEMMs on gfx1151. `K % 64 == 0` (Oq4G256
    /// guarantees %256). wave64 kernel (compiled `-mwavefrontsize64` via the source
    /// magic comment); block = 256 threads = 4 wave64 waves, block tile 64×256.
    /// The caller gates this to RDNA3.5+ prefill; decode/gfx1103 stay on the
    /// single-chain kernel. Parity: `parity_gemm_iu4_i32_wmma_lds`.
    pub fn gemm_iu4_i32_wmma_lds(
        &mut self,
        a_i4: &GpuTensor,
        x_i4: &GpuTensor,
        y_i32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            k % 64,
            0,
            "gemm_iu4_i32_wmma_lds: K must be a multiple of 64"
        );
        self.ensure_kernel(
            "gemm_iu4_i32_wmma_lds",
            kernels::GEMM_IU4_I32_WMMA_LDS_SRC,
            "gemm_iu4_i32_wmma_lds",
        )?;
        let ap = a_i4.buf.as_ptr();
        let xp = x_i4.buf.as_ptr();
        let yp = y_i32.buf.as_ptr();
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
        let grid_m = ((m + 63) / 64) as u32; // BM = 64
        let grid_b = ((batch_size + 255) / 256) as u32; // BN = 256
        let func = &self.functions["gemm_iu4_i32_wmma_lds"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_b, 1],
                [256, 1, 1], // 4 wave64 waves
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Opus Quant W4A4 core: grouped signed-INT4 × INT4 GEMM with per-group scale
    /// rescale in the f32 epilogue. `w_i4` [M,K/2] + `w_scales` [M,K/group] (f32),
    /// `x_i4` [B,K/2] + `x_scales` [B,K/group] (f32), `y_f32` [B,M]. Requires
    /// `k % group == 0` and `group % 16 == 0`. gfx1103 wave32, zero LDS.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_oq4_grouped_wmma(
        &mut self,
        w_i4: &GpuTensor,
        w_scales: &GpuTensor,
        x_i4: &GpuTensor,
        x_scales: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            k % group,
            0,
            "gemm_oq4_grouped_wmma: K must be a multiple of group"
        );
        assert_eq!(
            group % 16,
            0,
            "gemm_oq4_grouped_wmma: group must be a multiple of 16"
        );
        self.ensure_kernel(
            "gemm_oq4_grouped_wmma",
            kernels::GEMM_OQ4_GROUPED_WMMA_SRC,
            "gemm_oq4_grouped_wmma",
        )?;
        let wp = w_i4.buf.as_ptr();
        let wsp = w_scales.buf.as_ptr();
        let xp = x_i4.buf.as_ptr();
        let xsp = x_scales.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut gi = group as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &wsp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
        ];
        let grid_m = ((m + 15) / 16) as u32;
        let grid_b = ((batch_size + 15) / 16) as u32;
        let func = &self.functions["gemm_oq4_grouped_wmma"];
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

    /// OQ4+ batched prefill: W4A16 grouped GEMM. `x_f32` is the f32
    /// FWHT(+AWQ)-rotated activation batch [B,K]; it is converted to f16 once
    /// and the 4-bit-resident weight is dequantized to f16 inline.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_oq4_grouped_f16_wmma(
        &mut self,
        w_i4: &GpuTensor,
        x_f32: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(group, 256, "gemm_oq4_grouped_f16_wmma: group must be 256");
        assert_eq!(
            k % group,
            0,
            "gemm_oq4_grouped_f16_wmma: K must be a multiple of group"
        );
        self.ensure_kernel(
            "gemm_oq4_grouped_f16_wmma",
            kernels::GEMM_OQ4_GROUPED_F16_WMMA_SRC,
            "gemm_oq4_grouped_f16_wmma",
        )?;
        let x_fp16 = self.ensure_fp16_x(x_f32, batch_size * k)?;
        let wp = w_i4.buf.as_ptr();
        let mut xp = x_fp16;
        let yp = y_f32.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut gi = group as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
        ];
        let grid_x = m.div_ceil(16) as u32;
        let grid_y = batch_size.div_ceil(16) as u32;
        let func = &self.functions["gemm_oq4_grouped_f16_wmma"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, grid_y, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// OQ4+ MMQ batched GEMM. Quantizes the f32 activation to block_q8_1 once,
    /// then uses int8 WMMA over the 4-bit-resident weight. `add=false` writes
    /// `Y = W*x`; `add=true` accumulates `Y += W*x`.
    pub fn gemm_oq4_residual_mmq(
        &mut self,
        w_combined: &GpuTensor,
        x_f32: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
        add: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x_f32, n, k)?;
        self.gemm_oq4_mmq_launch(w_combined, x_q8_ptr, y, m, k, n, add)
    }

    /// OQ4+ MMQ GEMM over a pre-quantized q8_1 activation pointer.
    pub fn gemm_oq4_mmq_launch(
        &mut self,
        w_combined: &GpuTensor,
        x_q8_ptr: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
        add: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let kernel_name = if m % 128 == 0 && n % 128 == 0 {
            if add {
                "gemm_oq4_residual_mmq_full_add"
            } else {
                "gemm_oq4_residual_mmq_full_set"
            }
        } else {
            "gemm_oq4_residual_mmq"
        };
        self.ensure_kernel(
            "gemm_oq4_residual_mmq",
            kernels::GEMM_OQ4_RESIDUAL_MMQ_SRC,
            kernel_name,
        )?;
        let a_ptr = w_combined.buf.as_ptr();
        let mut xq_ptr = x_q8_ptr;
        let y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = n as i32;
        let mut add_val = if add { 1i32 } else { 0i32 };
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut add_val as *mut _ as *mut c_void,
        ];
        const MMQ_X: usize = 128;
        const MMQ_Y: usize = 128;
        const MMQ_TILE_Y_K: usize = 36;
        const MMQ_TILE_X_K: usize = 76;
        let row_tiles = m.div_ceil(MMQ_Y);
        let batch_tiles = n.div_ceil(MMQ_X);
        let shared_mem =
            ((MMQ_X * MMQ_TILE_Y_K + MMQ_Y * MMQ_TILE_X_K) * std::mem::size_of::<i32>()) as u32;
        self.launch_maybe_blob(
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
        )
    }

    /// OQ4+ MMQ 4-way QKVZA prefill: quantize once, then one MMQ GEMM per
    /// projection.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_oq4_qkvza_mmq(
        &mut self,
        w_qkv: &GpuTensor,
        w_z: &GpuTensor,
        w_beta: &GpuTensor,
        w_alpha: &GpuTensor,
        x_f32: &GpuTensor,
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
        let xq = self.ensure_q8_1_mmq_x(x_f32, n, k)?;
        self.gemm_oq4_mmq_launch(w_qkv, xq, y_qkv, qkv_m, k, n, false)?;
        self.gemm_oq4_mmq_launch(w_z, xq, y_z, z_m, k, n, false)?;
        self.gemm_oq4_mmq_launch(w_beta, xq, y_beta, beta_m, k, n, false)?;
        self.gemm_oq4_mmq_launch(w_alpha, xq, y_alpha, alpha_m, k, n, false)
    }

    /// OQ4+ MMQ 2-way gate+up prefill: quantize once, then one MMQ GEMM per
    /// projection.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_oq4_gate_up_mmq(
        &mut self,
        w_gate: &GpuTensor,
        w_up: &GpuTensor,
        x_f32: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        let xq = self.ensure_q8_1_mmq_x(x_f32, n, k)?;
        self.gemm_oq4_mmq_launch(w_gate, xq, y_gate, gate_m, k, n, false)?;
        self.gemm_oq4_mmq_launch(w_up, xq, y_up, up_m, k, n, false)
    }

    /// OQ4+ MMQ 3-way QKV prefill: quantize once, then one MMQ GEMM per
    /// projection.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_oq4_qkv_mmq(
        &mut self,
        w_q: &GpuTensor,
        w_k: &GpuTensor,
        w_v: &GpuTensor,
        x_f32: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        let xq = self.ensure_q8_1_mmq_x(x_f32, n, k)?;
        self.gemm_oq4_mmq_launch(w_q, xq, y_q, q_m, k, n, false)?;
        self.gemm_oq4_mmq_launch(w_k, xq, y_k, k_m, k, n, false)?;
        self.gemm_oq4_mmq_launch(w_v, xq, y_v, v_m, k, n, false)
    }

    /// MQ2-Lloyd grouped GEMM (F16 WMMA k2). DeepSeek V4 port of the HFQ4 grouped
    /// pattern. Same scatter pipeline + kernarg layout as
    /// `gemm_hfq4g256_moe_grouped_wmma_k2`; the kernel decodes MQ2-Lloyd's
    /// 72 B/group codebook layout instead of HFQ4's 136 B/group affine.
    /// Used by DeepSeek V4 prefill `ffn_batched` when `chunk_size ≥ 256` (the tile-
    /// fill threshold from Gate 1); falls back to scalar K4 below that.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq2g256_lloyd_moe_grouped_wmma_k2(
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
        let kernel_name = "gemm_mq2g256_lloyd_moe_grouped_wmma_k2";
        let kernel_src = kernels::GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_K2_SRC;
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
        // BW: MQ2-Lloyd weight is 72 B/group, half of HFQ4's 136 B/group.
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = m_total * k * 2 + (m_total * m) * 4 + mq2_weight_bytes;
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
    /// 4-warp MoE-grouped MQ2-Lloyd WMMA GEMM (gfx1151 RDNA3.5). Drop-in
    /// replacement for `gemm_mq2g256_lloyd_moe_grouped_wmma_k2` with a
    /// 64-row × 16-slot output tile (4× more output rows per block via
    /// 4 cooperating warps). LDS-staged X is shared across warps → 4×
    /// less B-fragment memory traffic per FLOP. Caller-checked:
    /// `m % 64 == 0` (kernel ASSERTs); slot dim stays at 16 due to
    /// the expert-spanning constraint (one block = one expert).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2(
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
        debug_assert_eq!(
            m % 64,
            0,
            "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2: M must be a multiple of 64 (got {m})"
        );
        debug_assert_eq!(
            k % 256,
            0,
            "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2: K must be a multiple of 256 (got {k})"
        );
        let kernel_name = "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2";
        let kernel_src = kernels::GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_4W_K2_SRC;
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

        // Row tiles widen 16 → 64; slot tiles unchanged at 16.
        let row_tiles = ((m + 63) / 64) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW unchanged from baseline: same total data movement, the
        // win is in B-fragment cache reuse across warps.
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = m_total * k * 2 + (m_total * m) * 4 + mq2_weight_bytes;
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
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// MMQ-style preload variant of the 4w MQ2-Lloyd MoE grouped GEMM.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload(
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
        debug_assert_eq!(m % 64, 0, "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload: M must be a multiple of 64 (got {m})");
        debug_assert_eq!(k % 256, 0, "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload: K must be a multiple of 256 (got {k})");
        let kernel_name = "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload";
        let kernel_src = kernels::GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_4W_K2_MMQLOAD_SRC;
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

        let row_tiles = ((m + 63) / 64) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = m_total * k * 2 + (m_total * m) * 4 + mq2_weight_bytes;
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
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Barrier-free variant of the mmqload kernel.
    /// Eliminates __syncthreads() and LDS X staging.
    pub fn gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload_nosync(
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
        debug_assert_eq!(m % 64, 0, "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload_nosync: M must be a multiple of 64 (got {m})");
        debug_assert_eq!(k % 256, 0, "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload_nosync: K must be a multiple of 256 (got {k})");
        let kernel_name = "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload_nosync";
        let kernel_src = kernels::GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_4W_K2_MMQLOAD_NOSYNC_SRC;
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

        let row_tiles = ((m + 63) / 64) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = m_total * k * 2 + (m_total * m) * 4 + mq2_weight_bytes;
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
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// `gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2` with N_TILE=32 tile-pairing:
    /// each block processes two consecutive 16-slot tiles and, when both map to
    /// the same expert (the common case after the expert-sorted scatter),
    /// decodes the per-warp MQ2-Lloyd A-fragment ONCE for two WMMAs — halving
    /// the dominant dequant ALU per token. `grid.y` halves to `(m_total/16+1)/2`;
    /// scatter (BLOCK_M=16) and the slot-indexed `Y_grouped` layout are unchanged.
    pub fn gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32(
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
        debug_assert_eq!(
            m % 64,
            0,
            "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32: M must be a multiple of 64 (got {m})"
        );
        debug_assert_eq!(
            k % 256,
            0,
            "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32: K must be a multiple of 256 (got {k})"
        );
        let kernel_name = "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32";
        let kernel_src = kernels::GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_4W_K2_N32_SRC;
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

        let row_tiles = ((m + 63) / 64) as u32;
        // Each block handles TWO 16-slot tiles → halve the slot-tile grid dim.
        let slot_tiles = (((m_total + 15) / 16 + 1) / 2) as u32;
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = m_total * k * 2 + (m_total * m) * 4 + mq2_weight_bytes;
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
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// `gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2` with the per-weight MQ2
    /// dequant done via f16 cndmask selects instead of int→f32→f16 — bit-exact,
    /// shorter dependency chain, identical geometry/LDS/occupancy. Same grid as
    /// the 4w baseline.
    pub fn gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_cnd(
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
        debug_assert_eq!(
            m % 64,
            0,
            "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_cnd: M must be a multiple of 64 (got {m})"
        );
        debug_assert_eq!(
            k % 256,
            0,
            "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_cnd: K must be a multiple of 256 (got {k})"
        );
        let kernel_name = "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_cnd";
        let kernel_src = kernels::GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_4W_K2_CND_SRC;
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

        let row_tiles = ((m + 63) / 64) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = m_total * k * 2 + (m_total * m) * 4 + mq2_weight_bytes;
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
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// 8-warp (M_TILE=128) variant of the grouped MQ2-Lloyd GEMM. Shares the
    /// staged X across 8 warps (half the X-load traffic per M-row vs the 4w).
    /// 256-thread block, grid.x = (m+127)/128; slot-tile grid unchanged.
    pub fn gemm_mq2g256_lloyd_moe_grouped_wmma_8w_k2(
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
        debug_assert_eq!(
            m % 64,
            0,
            "gemm_mq2g256_lloyd_moe_grouped_wmma_8w_k2: M must be a multiple of 64 (got {m})"
        );
        debug_assert_eq!(
            k % 256,
            0,
            "gemm_mq2g256_lloyd_moe_grouped_wmma_8w_k2: K must be a multiple of 256 (got {k})"
        );
        let kernel_name = "gemm_mq2g256_lloyd_moe_grouped_wmma_8w_k2";
        let kernel_src = kernels::GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_8W_K2_SRC;
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

        let row_tiles = ((m + 127) / 128) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = m_total * k * 2 + (m_total * m) * 4 + mq2_weight_bytes;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
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
}
